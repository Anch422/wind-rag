import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit, QFileDialog, QMessageBox,
    QFormLayout, QGroupBox, QProgressBar, QSplitter, QHeaderView, QAbstractItemView, QCheckBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from .data import load_corpus, starter_questions, load_questions, read_csv
from .engine import Engine, Settings, METHODS, Ollama
from .evaluation import benchmark, save_run, METRICS
from .branding import APP_NAME, APP_DESCRIPTION, APP_FULL_NAME

ROOT = Path(__file__).resolve().parent.parent


class Worker(QThread):
    progress = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        try:
            result = self.task(self.progress.emit, self.isInterruptionRequested)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def fill_table(table, rows, columns):
    sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    table.blockSignals(True)
    table.clear()
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels([title for key, title in columns])
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, (key, _) in enumerate(columns):
            value = row.get(key, "")
            if value is None:
                text = "N/A"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            elif isinstance(value, (list, dict)):
                text = json.dumps(value, ensure_ascii=False)
            else:
                text = str(value)
            item = QTableWidgetItem()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                item.setData(Qt.ItemDataRole.DisplayRole, value)
            else:
                item.setText(text)
            item.setData(Qt.ItemDataRole.UserRole, r)
            item.setToolTip(text)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    table.setSortingEnabled(sorting)
    table.blockSignals(False)


def table_widget():
    table = QTableWidget()
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().hide()
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    return table


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Qt's offscreen Windows backend may not enumerate system fonts.
        if "Segoe UI" not in QFontDatabase.families():
            for filename in ("segoeui.ttf", "segoeuib.ttf"):
                font_path = Path("C:/Windows/Fonts") / filename
                if font_path.exists():
                    QFontDatabase.addApplicationFont(str(font_path))
        self.setWindowTitle(APP_FULL_NAME)
        self.resize(1400, 940)
        self.docs, self.events, self.features, self.questions = [], [], [], []
        self.engine = None
        self.corpus_hash = ""
        self.run_data = None
        self.worker = None
        self.query_results = []
        self._busy = False
        self.mutating_controls = []
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 18, 24, 18)
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel(APP_DESCRIPTION)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.setup_tab()
        self.data_tab()
        self.ask_tab()
        self.benchmark_tab()
        self.results_tab()
        self.help_tab()
        bottom = QHBoxLayout()
        self.status = QLabel("Choose a mode, then load the dataset and build the index.")
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(160)
        self.bar.setRange(0, 1)
        self.cancel = QPushButton("Cancel task")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_task)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.bar)
        bottom.addWidget(self.cancel)
        layout.addLayout(bottom)
        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #1e293b; }
            QMainWindow, QTabWidget::pane { background: #f6f8fc; }
            QLabel#title { font-size: 29px; font-weight: 700; color: #0f766e; }
            QLabel#note { color: #526174; }
            QGroupBox { font-weight: 600; border: 1px solid #d6dee8; border-radius: 7px; margin-top: 15px; padding: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; }
            QPushButton { background: #0f766e; color: white; padding: 8px 14px; border-radius: 5px; }
            QPushButton:disabled { background: #cbd5e1; color: #64748b; }
            QPushButton:hover { background: #115e59; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTextBrowser { background: white; border: 1px solid #ccd5e1; padding: 6px; border-radius: 4px; }
            QTabBar::tab { padding: 11px 18px; }
            QTabBar::tab:selected { background: white; color: #0f766e; }
            QTableWidget { background: white; alternate-background-color: #eef3f8; gridline-color: #e2e8f0; }
            QHeaderView::section { background: #e2e8f0; padding: 7px; font-weight: 600; border: 0; }
        """)
        QTimer.singleShot(0, lambda: self.check_ollama(automatic=True))
        self.host.editingFinished.connect(lambda: self.check_ollama(automatic=True))

    def button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        self.mutating_controls.append(button)
        return button

    def setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Dataset and models")
        form = QFormLayout(group)
        path_row = QHBoxLayout()
        self.dataset_path = QLineEdit(str(ROOT / "datasets" / "CARE_To_Compare"))
        path_row.addWidget(self.dataset_path, 1)
        self.button("Browse", self.browse_dataset, path_row)
        form.addRow("Extracted dataset", path_row)
        self.include_reports = QCheckBox("Include technician PDF reports")
        self.include_reports.setChecked(True)
        form.addRow("Additional evidence", self.include_reports)
        report_row = QHBoxLayout()
        self.reports_path = QLineEdit(str(ROOT / 'datasets' / 'technician_reports'))
        report_row.addWidget(self.reports_path, 1)
        self.button('Browse reports', self.browse_reports, report_row)
        form.addRow('Technician reports folder', report_row)
        self.mode = QComboBox()
        self.mode.addItems(["Ollama · local RAG", "Offline demo · LSA + extractive answers"])
        form.addRow("Run mode", self.mode)
        self.host = QLineEdit("http://localhost:11434")
        self.embedding_model = QComboBox()
        self.embedding_model.setEditable(True)
        self.embedding_model.addItem("nomic-embed-text")
        self.answer_model = QComboBox()
        self.answer_model.setEditable(True)
        self.answer_model.addItem("qwen2.5:3b")
        form.addRow("Ollama server", self.host)
        form.addRow("Embedding model", self.embedding_model)
        form.addRow("Answer / reranker model", self.answer_model)
        self.k = QSpinBox()
        self.k.setRange(1, 20)
        self.k.setValue(5)
        self.candidates = QSpinBox()
        self.candidates.setRange(5, 80)
        self.candidates.setValue(20)
        self.seed = QSpinBox()
        self.seed.setRange(0, 1000000)
        self.seed.setValue(42)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0, 2)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0)
        form.addRow("Retrieved sources (top-k)", self.k)
        form.addRow("Reranking candidates", self.candidates)
        form.addRow("Random seed", self.seed)
        form.addRow("Generation temperature", self.temperature)
        layout.addWidget(group)
        actions = QHBoxLayout()
        self.button("Refresh models", lambda: self.check_ollama(), actions)
        self.button("Load / reuse index", self.build_index, actions)
        actions.addStretch()
        layout.addLayout(actions)
        self.index_info = QTextBrowser()
        self.index_info.setPlainText("Ollama mode uses local neural embeddings, BM25 fusion, and LLM relevance reranking.\n\nOffline demo uses LSA vectors, BM25 fusion, character TF-IDF reranking, and an extractive answer. Demo scores are not neural RAG results.\n\nOnly event metadata and feature definitions are indexed. Raw SCADA remains on disk and can be plotted in Dataset.")
        layout.addWidget(self.index_info, 1)
        self.tabs.addTab(tab, "1 · Setup")
        self.mutating_controls.extend([self.dataset_path, self.mode, self.host, self.embedding_model,
                                       self.answer_model, self.k, self.candidates, self.seed, self.temperature,
                                       self.include_reports, self.reports_path])

    def data_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.data_summary = QLabel("Load the dataset to explore event records and source documents.")
        layout.addWidget(self.data_summary)
        self.event_table = table_widget()
        self.event_table.itemSelectionChanged.connect(self.event_selected)
        self.source_table = table_widget()
        subtabs = QTabWidget()
        subtabs.addTab(self.event_table, "Events")
        subtabs.addTab(self.source_table, "Source IDs / corpus")
        layout.addWidget(subtabs, 1)
        row = QHBoxLayout()
        self.sensor = QComboBox()
        self.sensor.setMinimumWidth(380)
        self.sensor.addItem("Select an event first")
        row.addWidget(self.sensor, 1)
        self.button("Plot selected event sensor", self.plot_sensor, row)
        self.button("Export source catalog", self.export_corpus, row)
        layout.addLayout(row)
        self.scada_figure = Figure(figsize=(8, 2.5), tight_layout=True)
        self.scada_canvas = FigureCanvasQTAgg(self.scada_figure)
        layout.addWidget(self.scada_canvas, 1)
        self.tabs.addTab(tab, "2 · Dataset")

    def ask_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.question = QTextEdit()
        self.question.setMaximumHeight(95)
        self.question.setPlaceholderText("Ask about recorded faults, event history, or sensor definitions…")
        self.question.setPlainText("What does sensor_0 measure at Wind Farm A, and what is its unit?")
        layout.addWidget(self.question)
        row = QHBoxLayout()
        self.button("Compare all three", self.ask, row)
        row.addStretch()
        layout.addLayout(row)
        self.answer_tabs = QTabWidget()
        self.answer_views = {}
        for method in METHODS:
            view = QTextBrowser()
            self.answer_views[method] = view
            self.answer_tabs.addTab(view, method)
        layout.addWidget(self.answer_tabs, 1)
        self.tabs.addTab(tab, "3 · Ask & compare")

    def benchmark_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        note = QLabel("Starter questions come from indexed metadata. Review or replace them before reporting thesis results.\nImported benchmarks need: id, question, reference_answer, relevant_ids (source IDs).")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.button("Generate starter questions", self.generate_questions, row)
        self.button("Import JSON / CSV", self.import_questions, row)
        self.button("Export questions", self.export_questions, row)
        row.addStretch()
        layout.addLayout(row)
        self.question_table = table_widget()
        self.question_table.itemSelectionChanged.connect(self.question_selected)
        layout.addWidget(self.question_table, 1)
        editor = QGroupBox("Review selected question")
        form = QFormLayout(editor)
        self.edit_question = QLineEdit()
        self.edit_reference = QLineEdit()
        self.edit_ids = QLineEdit()
        form.addRow("Question", self.edit_question)
        form.addRow("Reference answer", self.edit_reference)
        form.addRow("Relevant IDs (separate with |)", self.edit_ids)
        edit_actions = QHBoxLayout()
        self.button("Save edits", self.save_question_edits, edit_actions)
        self.button("Delete selected", self.delete_question, edit_actions)
        self.button("Add question", self.add_question, edit_actions)
        form.addRow(edit_actions)
        layout.addWidget(editor)
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Questions to run (first N)"))
        self.run_count = QSpinBox()
        self.run_count.setRange(1, 10000)
        self.run_count.setValue(12)
        run_row.addWidget(self.run_count)
        self.button("Run benchmark + save results", self.run_benchmark, run_row)
        run_row.addStretch()
        layout.addLayout(run_row)
        self.tabs.addTab(tab, "4 · Benchmark")
        self.mutating_controls.extend([self.edit_question, self.edit_reference, self.edit_ids, self.run_count])

    def results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.button("Open saved run", self.open_run, row)
        self.button("Save chart as PNG", self.export_chart, row)
        self.metric_choice = QComboBox()
        self.metric_choice.addItems(METRICS)
        self.metric_choice.currentTextChanged.connect(self.draw_comparison)
        row.addWidget(QLabel("Chart metric"))
        row.addWidget(self.metric_choice)
        row.addStretch()
        layout.addLayout(row)
        self.run_info = QLabel("No benchmark run yet.")
        self.run_info.setWordWrap(True)
        layout.addWidget(self.run_info)
        self.summary_table = table_widget()
        layout.addWidget(self.summary_table)
        self.figure = Figure(figsize=(9, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.result_table = table_widget()
        self.result_table.itemSelectionChanged.connect(self.result_selected)
        self.result_detail = QTextBrowser()
        self.result_detail.setMinimumHeight(140)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.result_table)
        split.addWidget(self.result_detail)
        layout.addWidget(split, 1)
        self.tabs.addTab(tab, "5 · Results")

    def help_tab(self):
        tab = QTextBrowser()
        tab.setOpenExternalLinks(True)
        tab.setPlainText("QUICK START\n\n1. Setup: select Ollama or Offline demo, then Load / reuse index.\n2. Dataset: inspect events, source IDs, and one sensor at a time.\n3. Ask & compare: run one question and inspect answers with retrieved evidence.\n4. Benchmark: generate starter questions or import a reviewed JSON/CSV benchmark. Edit questions and their evidence IDs.\n5. Run benchmark: the app compares identical questions across all three methods and saves CSV/JSON files.\n6. Results: compare charts, confidence intervals, per-question answers, and sources.\n\nMETHODS\nDense: cosine similarity over vectors.\nHybrid: reciprocal rank fusion of dense retrieval and BM25.\nHybrid + reranking: reorder the hybrid candidate pool using local LLM relevance scores (or character TF-IDF in demo mode).\n\nMETRICS\nRecall@k: fraction of annotated relevant sources retrieved.\nPrecision@k: fraction of retrieved sources annotated relevant.\nMRR: reciprocal rank of the first relevant source.\nnDCG@k: rewards relevant sources near the top.\nAnswer F1: normalized token overlap against a reference answer.\nExact match: normalized answer equality.\nCitation validity: fraction of cited IDs present in retrieved context; this does not check claim support.\nCitation recall: fraction of annotated relevant sources cited.\nLatency: milliseconds, including query embedding and reranking for retrieval; index building is separate.\n\nREAD README.md FOR THE COMPLETE BEGINNER COURSE.\n\nThe app answers historical dataset questions. It does not diagnose turbines, predict faults, or provide validated maintenance instructions. No API keys are required. Ollama must be installed separately with an embedding model and an answer model.")
        self.tabs.addTab(tab, "Help")

    def settings(self):
        return Settings(mode="ollama" if self.mode.currentIndex() == 0 else "offline",
                        host=self.host.text().strip(), embedding_model=self.embedding_model.currentText().strip(),
                        answer_model=self.answer_model.currentText().strip(), top_k=self.k.value(),
                        candidates=max(self.k.value(), self.candidates.value()), seed=self.seed.value(),
                        temperature=self.temperature.value())

    def current_engine(self):
        if self.engine is None:
            raise ValueError("Build an index in Setup first.")
        if str(Path(self.dataset_path.text()).resolve()) != self.engine.dataset_root:
            raise ValueError("Dataset folder changed. Load its index in Setup.")
        reports_root = str(Path(self.reports_path.text()).resolve()) if self.include_reports.isChecked() else None
        if getattr(self.engine, 'reports_root', None) != reports_root:
            raise ValueError('Technician reports settings changed. Waiting for automatic index initialization.')
        _, _, _, current_hash = load_corpus(self.engine.dataset_root, reports_root)
        if current_hash != self.corpus_hash:
            raise ValueError("Dataset metadata changed. Load its updated index in Setup.")
        self.engine.apply_settings(self.settings())
        return self.engine

    def perform(self, task, callback, quiet=False, worker=None):
        if self._busy:
            return
        self._busy = True
        for control in self.mutating_controls:
            control.setEnabled(False)
        self.cancel.setEnabled(True)
        self.bar.setRange(0, 0)
        self.worker = worker if worker is not None else Worker(task)
        self.worker.progress.connect(self.status.setText)
        self.worker.succeeded.connect(callback)
        self.worker.failed.connect(self.status.setText if quiet else self.task_error)
        self.worker.finished.connect(self.task_finished)
        self.worker.start()

    def task_finished(self):
        self._busy = False
        for control in self.mutating_controls:
            control.setEnabled(True)
        self.cancel.setEnabled(False)
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        if self.worker:
            self.worker.deleteLater()
        self.worker = None

    def task_error(self, message):
        self.status.setText(message)
        QMessageBox.warning(self, "Task did not complete", message)

    def cancel_task(self):
        if self.worker:
            self.worker.requestInterruption()
            self.status.setText("Cancellation requested; waiting for the current model request to finish.")

    def guard(self, action):
        try:
            action()
        except Exception as exc:
            QMessageBox.warning(self, "Action unavailable", str(exc))

    def browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Choose extracted CARE_To_Compare folder", self.dataset_path.text())
        if path:
            self.dataset_path.setText(path)

    def browse_reports(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose technician PDF reports folder', self.reports_path.text())
        if path:
            self.reports_path.setText(path)

    def build_index(self):
        path, settings = self.dataset_path.text(), self.settings()
        reports_root = str(Path(self.reports_path.text()).resolve()) if self.include_reports.isChecked() else None
        def task(progress, cancelled):
            if settings.mode == "ollama":
                progress("Checking selected Ollama models…")
                Ollama(settings).validate_models(include_answer=False)
            progress("Reading event and feature metadata…")
            docs, events, features, digest = load_corpus(path, reports_root)
            engine = Engine(docs, settings, progress, cancelled, cache_dir=ROOT / ".cache" / "indexes")
            engine.dataset_root = str(Path(path).resolve())
            engine.reports_root = reports_root
            return docs, events, features, digest, engine
        self.perform(task, self.index_ready)

    def index_ready(self, result):
        previous_hash = self.corpus_hash
        self.docs, self.events, self.features, self.corpus_hash, self.engine = result
        if previous_hash != self.corpus_hash:
            self.questions = []
        self.refresh_questions()
        normal = sum(e["label"] == "normal" for e in self.events)
        reports = sum(d.kind == 'report' for d in self.docs)
        self.data_summary.setText(f"{len(self.events)} events · {len(self.events) - normal} anomalies · {normal} normal events · {len(self.features)} sensor definitions · {reports} PDF report chunks · {len(self.docs)} indexed documents")
        fill_table(self.event_table, self.events, [("farm", "Farm"), ("event_id", "Event"), ("asset", "Asset"),
                  ("label", "Label"), ("start", "Start"), ("description", "Recorded description")])
        fill_table(self.source_table, [d.__dict__ for d in self.docs], [("id", "Source ID"), ("kind", "Type"), ("title", "Title"), ("text", "Text")])
        self.index_info.setPlainText(self.data_summary.text() + "\n\n" + json.dumps(self.engine.config(), indent=2) + "\n\nCorpus SHA256: " + self.corpus_hash)
        self.status.setText("Saved index loaded." if self.engine.cache_hit else "Index built and saved for reuse.")

    def check_ollama(self, automatic=False):
        host = self.host.text().strip()
        def task(progress, cancelled):
            progress("Connecting to Ollama…")
            response = requests.get(host.rstrip("/") + "/api/tags", timeout=(5, 10))
            response.raise_for_status()
            return response.json().get("models", [])
        def done(models):
            for combo, capability in ((self.embedding_model, "embedding"), (self.answer_model, "completion")):
                previous = combo.currentText()
                names = [m["name"] for m in models if capability in m.get("capabilities", [])]
                if not names:
                    names = [m["name"] for m in models]
                combo.clear()
                combo.addItems(names)
                canonical = previous if ":" in previous else previous + ":latest"
                if canonical in names:
                    combo.setCurrentText(canonical)
                elif combo is self.answer_model and "qwen2.5:7b" in names:
                    combo.setCurrentText("qwen2.5:7b")
                elif names:
                    combo.setCurrentIndex(0)
                else:
                    combo.setCurrentText(previous)
            self.status.setText("Ollama connected. Answer models can change without rebuilding. Installed: " + (", ".join(m["name"] for m in models) or "none"))
        self.perform(task, done, quiet=automatic)

    def event_selected(self):
        row = self.event_table.currentRow()
        if row >= 0:
            row = self.event_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if 0 <= row < len(self.events):
            farm = self.events[row]["farm"]
            self.sensor.clear()
            for feature in self.features:
                if feature["farm"] == farm and "average" in feature["statistics"]:
                    self.sensor.addItem(f"{feature['sensor']} · {feature['description']} ({feature['unit']})", feature)

    def plot_sensor(self):
        def action():
            row = self.event_table.currentRow()
            if row >= 0:
                row = self.event_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            feature = self.sensor.currentData()
            if row < 0 or not feature:
                raise ValueError("Select an event and an average sensor first.")
            event = self.events[row]
            def task(progress, cancelled):
                path = Path(event["scada_path"])
                progress(f"Reading {path.name}: one sensor column, all rows…")
                columns = read_csv(path, nrows=0).columns
                sensor = feature["sensor"]
                column = sensor + "_avg" if sensor + "_avg" in columns else sensor
                if column not in columns:
                    raise ValueError(f"Sensor column {column} is absent in this event file.")
                pieces = []
                for chunk in read_csv(path, usecols=["time_stamp", column], chunksize=10000):
                    if cancelled():
                        raise InterruptedError("Plot cancelled.")
                    pieces.append(chunk)
                values = pd.concat(pieces, ignore_index=True)
                values["time_stamp"] = pd.to_datetime(values["time_stamp"], errors="coerce")
                values[column] = pd.to_numeric(values[column], errors="coerce")
                return event, feature, values, column
            self.perform(task, self.sensor_ready)
        self.guard(action)

    def sensor_ready(self, result):
        event, feature, values, column = result
        self.scada_figure.clear()
        axis = self.scada_figure.add_subplot(111)
        # Display daily averages to keep dense 10-minute histories readable; compute from every row.
        plot = values.set_index("time_stamp")[column].resample("1D").mean()
        axis.plot(plot.index, plot.values, color="#0f766e", linewidth=1.2)
        axis.axvspan(pd.to_datetime(event["start"]), pd.to_datetime(event["end"]), color="#f59e0b", alpha=0.18, label="Recorded event interval")
        axis.set_title(f"{event['farm']} · Event {event['event_id']} · {feature['description']} · daily average")
        axis.set_ylabel(feature["unit"])
        axis.legend(fontsize=8)
        self.scada_figure.autofmt_xdate()
        self.scada_canvas.draw()
        self.status.setText(f"Plotted daily averages from all {len(values):,} rows. Timestamps are anonymized.")

    def ask(self):
        def action():
            engine = self.current_engine()
            query = self.question.toPlainText().strip()
            if not query:
                raise ValueError("Enter a question first.")
            self.perform(lambda p, c: engine.compare(query, p, c, reuse_answers=True), self.answers_ready)
        self.guard(action)

    def answers_ready(self, rows):
        self.query_results = rows
        for row in rows:
            sources = "\n\n".join(f"[{s['id']}] score {s['score']:.4f}\n{s['text']}\nSource: {s['source']}" for s in row["sources"])
            reused = "\nAnswer reused: identical ordered evidence in this interactive comparison." if row.get("answer_reused") else ""
            self.answer_views[row["method"]].setPlainText(f"{row['answer']}\n\nRetrieval: {row['retrieval_ms']:.1f} ms · Generation: {row['generation_ms']:.1f} ms{reused}\n\nRETRIEVED EVIDENCE\n\n{sources}")
        self.status.setText("Comparison complete. Inspect each answer and its retrieved evidence.")

    def refresh_questions(self):
        fill_table(self.question_table, self.questions, [("id", "Question ID"), ("question", "Question"),
                   ("reference_answer", "Reference answer"), ("relevant_ids", "Relevant source IDs"), ("origin", "Origin")])

    def generate_questions(self):
        def action():
            self.current_engine()
            if self.questions and QMessageBox.question(self, "Replace questions?", "Replace the current benchmark with generated starter questions?") != QMessageBox.StandardButton.Yes:
                return
            self.questions = starter_questions(self.events, self.features)
            self.refresh_questions()
            self.status.setText(f"Generated {len(self.questions)} starter questions. Review before research use.")
        self.guard(action)

    def question_selected(self):
        row = self.question_table.currentRow()
        if 0 <= row < len(self.questions):
            question = self.questions[row]
            self.edit_question.setText(question["question"])
            self.edit_reference.setText(question["reference_answer"])
            self.edit_ids.setText("|".join(question["relevant_ids"]))

    def save_question_edits(self):
        def action():
            row = self.question_table.currentRow()
            if row < 0:
                raise ValueError("Select a question to edit.")
            ids = [x.strip() for x in self.edit_ids.text().split("|") if x.strip()]
            unknown = set(ids) - {d.id for d in self.docs}
            if unknown:
                raise ValueError(f"Unknown source IDs: {sorted(unknown)}")
            if not self.edit_question.text().strip():
                raise ValueError("Question cannot be empty.")
            old_origin = self.questions[row].get("origin", "unspecified")
            self.questions[row].update(question=self.edit_question.text().strip(), reference_answer=self.edit_reference.text().strip(),
                                       relevant_ids=ids, origin=old_origin + "; edited by user")
            self.refresh_questions()
            self.question_table.selectRow(row)
        self.guard(action)

    def delete_question(self):
        row = self.question_table.currentRow()
        if 0 <= row < len(self.questions):
            del self.questions[row]
            self.refresh_questions()

    def add_question(self):
        identifier = 1
        existing = {q["id"] for q in self.questions}
        while f"custom-{identifier}" in existing:
            identifier += 1
        self.questions.append({"id": f"custom-{identifier}", "question": "Edit this question",
                               "reference_answer": "", "relevant_ids": [], "origin": "user authored"})
        self.refresh_questions()
        self.question_table.selectRow(len(self.questions) - 1)

    def import_questions(self):
        def action():
            self.current_engine()
            path, _ = QFileDialog.getOpenFileName(self, "Import benchmark", str(ROOT / "benchmarks"), "Benchmarks (*.json *.csv)")
            if path:
                self.questions = load_questions(path, [d.id for d in self.docs])
                self.refresh_questions()
                self.status.setText(f"Imported {len(self.questions)} benchmark questions.")
        self.guard(action)

    def export_questions(self):
        def action():
            if not self.questions:
                raise ValueError("No questions to export.")
            path, _ = QFileDialog.getSaveFileName(self, "Export questions", str(ROOT / "benchmarks" / "questions.json"), "JSON (*.json);;CSV (*.csv)")
            if path:
                if Path(path).suffix.lower() == ".csv":
                    rows = [q | {"relevant_ids": "|".join(q["relevant_ids"])} for q in self.questions]
                    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
                else:
                    Path(path).write_text(json.dumps(self.questions, indent=2, ensure_ascii=False), encoding="utf-8")
                self.status.setText(f"Questions exported to {path}")
        self.guard(action)

    def run_benchmark(self):
        def action():
            engine = self.current_engine()
            questions = json.loads(json.dumps(self.questions[:self.run_count.value()]))
            if not questions:
                raise ValueError("Generate or import benchmark questions first.")
            if any(not q["reference_answer"].strip() or q["question"] == "Edit this question" for q in questions):
                raise ValueError("Complete the questions and reference answers before running.")
            digest = self.corpus_hash
            def task(progress, cancelled):
                rows, summary = benchmark(engine, questions, progress, cancelled)
                if cancelled():
                    raise InterruptedError("Benchmark cancelled; incomplete run not saved.")
                progress("Saving complete run…")
                return save_run(ROOT / "results", engine, digest, questions, rows, summary)
            self.perform(task, self.benchmark_ready)
        self.guard(action)

    def benchmark_ready(self, result):
        folder, self.run_data = result
        self.populate_results()
        self.tabs.setCurrentIndex(4)
        self.figure.savefig(Path(folder) / "comparison.png", dpi=160)
        self.status.setText(f"Complete run saved: {folder}")

    def populate_results(self):
        run = self.run_data
        mode = run["config"]["mode"]
        self.run_info.setText(f"Run {run['run_id']} · {mode.upper()} · {len(run['questions'])} questions · {run['config']['dense_backend']}\nStarter benchmarks are not independent; citation validity is not factual faithfulness. Scores use exact annotated evidence IDs.")
        fill_table(self.summary_table, run["summary"], [("method", "Method"), ("recall_at_k", "Recall@k"),
            ("precision_at_k", "Precision@k"), ("mrr", "MRR"), ("ndcg_at_k", "nDCG@k"),
            ("answer_f1", "Answer F1"), ("exact_match", "Exact match"), ("citation_validity", "Citation validity"),
            ("citation_recall", "Citation recall"), ("total_ms", "Total ms")])
        fill_table(self.result_table, run["results"], [("question_id", "Question ID"), ("method", "Method"),
             ("recall_at_k", "Recall@k"), ("answer_f1", "Answer F1"), ("total_ms", "Total ms"), ("answer", "Answer")])
        self.draw_comparison()

    def draw_comparison(self):
        if not self.run_data:
            return
        metric = self.metric_choice.currentText()
        summaries = self.run_data["summary"]
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        values = [s.get(metric) for s in summaries]
        heights = [0 if v is None else v for v in values]
        intervals = [s.get(metric + "_ci95") for s in summaries]
        error = [[max(0, h - ci[0]) if ci else 0 for h, ci in zip(heights, intervals)],
                 [max(0, ci[1] - h) if ci else 0 for h, ci in zip(heights, intervals)]]
        axis.bar(range(len(heights)), heights, color=["#0f766e", "#2563eb", "#d97706"], yerr=error, capsize=5)
        axis.set_xticks(range(len(heights)), [s["method"] for s in summaries])
        axis.set_ylabel("Milliseconds" if metric.endswith("_ms") else "Score")
        axis.set_title(f"{metric} · mean with 95% question-bootstrap intervals · {self.run_data['config']['mode']}")
        if not metric.endswith("_ms"):
            axis.set_ylim(0, 1.13)
        for i, v in enumerate(values):
            axis.text(i, heights[i], "N/A" if v is None else f"{v:.3f}", ha="center", va="bottom")
        axis.grid(axis="y", alpha=0.15)
        self.canvas.draw()

    def result_selected(self):
        row = self.result_table.currentRow()
        if self.run_data and 0 <= row < len(self.run_data["results"]):
            self.result_detail.setPlainText(json.dumps(self.run_data["results"][row], indent=2, ensure_ascii=False))

    def open_run(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(self, "Open saved run", str(ROOT / "results"), "Run JSON (run.json)")
            if path:
                run = json.loads(Path(path).read_text(encoding="utf-8"))
                for key in ("config", "summary", "questions", "results", "run_id"):
                    if key not in run:
                        raise ValueError("This file is not a saved Wind RAG run.")
                self.run_data = run
                self.populate_results()
        self.guard(action)

    def export_chart(self):
        def action():
            if not self.run_data:
                raise ValueError("Run or open a benchmark first.")
            path, _ = QFileDialog.getSaveFileName(self, "Save chart", str(ROOT / "results" / "comparison.png"), "PNG (*.png)")
            if path:
                self.figure.savefig(path, dpi=200)
        self.guard(action)

    def export_corpus(self):
        def action():
            if not self.docs:
                raise ValueError("Load the dataset first.")
            path, _ = QFileDialog.getSaveFileName(self, "Export source catalog", str(ROOT / "benchmarks" / "corpus.csv"), "CSV (*.csv)")
            if path:
                pd.DataFrame([d.__dict__ for d in self.docs]).to_csv(path, index=False, encoding="utf-8-sig")
        self.guard(action)

    def closeEvent(self, event):
        if self._busy:
            QMessageBox.information(self, "Task running", "Cancel the task and wait for it to finish before closing.")
            event.ignore()
        else:
            event.accept()


def launch():
    (ROOT / "benchmarks").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

