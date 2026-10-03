"""Task-focused interface: chat, sequential benchmarking, saved-run comparisons."""
import html
import json
import sys
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QTextEdit, QTextBrowser, QScrollArea, QFrame,
    QListWidget, QListWidgetItem, QSpinBox, QSplitter, QFileDialog, QGroupBox, QTabWidget, QProgressBar)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import requests
from .ui import MainWindow, ROOT, table_widget, fill_table
from .engine import METHODS, METHOD_DETAILS, Engine
from .data import load_questions
from .experiments import (validation_set, sequential_runs, scan_runs, aggregate,
                          export_comparison, plot_metric, metric_label)
from .evaluation import METRICS
from .benchmark_worker import BenchmarkWorker
from .branding import APP_NAME, APP_FULL_NAME
from .chat_store import ChatStore


class MessageEditor(QTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier):
            event.accept()
            self.send_requested.emit()
        else:
            super().keyPressEvent(event)


class Workbench(MainWindow):
    def __init__(self, chat_database=None):
        self.chat_messages = []
        self.chat_store = ChatStore(chat_database or ROOT / 'sessions' / 'chat_history.sqlite3')
        self.active_chat_id = None
        self._restoring_chat = False
        self.run_entries = []
        self.chat_pending = False
        self._initializing = True
        self._init_stage = "models"
        self._init_failed = False
        self._closed = False
        self._auto_index_pending = False
        super().__init__()
        self.tabs.setTabText(0, "Setup")
        self.tabs.setTabText(1, "Dataset")
        self.tabs.setTabText(2, "Chat")
        self.tabs.setTabText(3, "Benchmark")
        self.tabs.setTabText(4, "Results")
        self.event_table.setSortingEnabled(True)
        self.source_table.setSortingEnabled(True)
        self.tabs.removeTab(5)
        self.k.setValue(3)
        self.candidates.setValue(10)
        # Chat and benchmarking have their own explicit answer-model selectors.
        self.answer_model.hide()
        form = self.answer_model.parentWidget().layout()
        if hasattr(form, "setRowVisible"):
            form.setRowVisible(self.answer_model, False)
        self.setWindowTitle(APP_FULL_NAME)
        self.setStyleSheet(self.styleSheet() + """
            QPushButton[color='blue'] { background: #2563eb; }
            QPushButton[color='green'] { background: #15803d; }
            QPushButton[color='orange'] { background: #c2410c; }
            QPushButton[color='gray'] { background: #64748b; }
            QPushButton:disabled { background: #cbd5e1; color: #64748b; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 22px; }
            QPushButton { min-height: 20px; }
        """)
        QTimer.singleShot(0, self.refresh_runs)
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.refresh_runs)
        self.scan_timer.start(15000)
        for control, signal in ((self.mode, 'currentIndexChanged'), (self.embedding_model, 'currentTextChanged'),
                                (self.host, 'textChanged'), (self.dataset_path, 'textChanged'),
                                (self.include_reports, 'toggled'), (self.reports_path, 'textChanged')):
            getattr(control, signal).connect(self.refresh_rag_readiness)
        self.refresh_rag_readiness()
        self.initialization_panel = QGroupBox(f'Initializing {APP_NAME}')
        loading = QVBoxLayout(self.initialization_panel)
        self.initialization_message = QLabel("Starting: discovering models, loading dataset and indexes, scanning results…")
        self.initialization_message.setWordWrap(True)
        loading.addWidget(self.initialization_message)
        self.initialization_bar = QProgressBar()
        self.initialization_bar.setRange(0, 100)
        self.initialization_bar.setValue(5)
        loading.addWidget(self.initialization_bar)
        self.retry_initialization = QPushButton("Retry initialization")
        self.retry_initialization.clicked.connect(self.initialize)
        self.retry_initialization.hide()
        loading.addWidget(self.retry_initialization)
        self.centralWidget().layout().insertWidget(2, self.initialization_panel)
        self.tabs.setEnabled(False)
        self.auto_index_timer = QTimer(self)
        self.auto_index_timer.setSingleShot(True)
        self.auto_index_timer.setInterval(700)
        self.auto_index_timer.timeout.connect(self.auto_prepare_index)
        for control, signal in ((self.mode, 'currentIndexChanged'), (self.embedding_model, 'currentTextChanged'),
                                (self.host, 'editingFinished'), (self.dataset_path, 'editingFinished'),
                                (self.include_reports, 'toggled'), (self.reports_path, 'editingFinished')):
            getattr(control, signal).connect(self.schedule_auto_index)

    def initialize(self):
        if self._busy:
            return
        self._initializing = True
        self._init_failed = False
        self._init_stage = "models"
        self.initialization_panel.show()
        self.retry_initialization.hide()
        self.initialization_bar.setValue(5)
        self.initialization_message.setText("Discovering installed Ollama models…")
        self.tabs.setEnabled(False)
        if self.settings().mode == "offline":
            self._init_stage = "index"
            self.build_index()
        else:
            self.check_ollama()

    def schedule_auto_index(self, *_):
        if not self._initializing:
            self._auto_index_pending = True
            if not self._busy:
                self.auto_index_timer.start()

    def browse_dataset(self):
        super().browse_dataset()
        self.schedule_auto_index()

    def browse_reports(self):
        super().browse_reports()
        self.schedule_auto_index()

    def auto_prepare_index(self):
        if self._busy or self._initializing or self._closed:
            return
        self._auto_index_pending = False
        settings = self.settings()
        reports_root = str(Path(self.reports_path.text()).resolve()) if self.include_reports.isChecked() else None
        if self.engine is None or self.engine.index_spec != Engine.index_settings(settings) or getattr(self.engine, 'dataset_root', '') != str(Path(self.dataset_path.text()).resolve()) or getattr(self.engine, 'reports_root', None) != reports_root:
            self.build_index()

    def build_index(self):
        if self._busy or self._closed:
            return
        self._initializing = True
        self._init_failed = False
        self._init_stage = "index"
        self.initialization_panel.show()
        self.retry_initialization.hide()
        self.initialization_bar.setValue(30)
        self.initialization_message.setText("Loading dataset and preparing or reusing saved RAG indexes…")
        self.tabs.setEnabled(False)
        super().build_index()

    def perform(self, task, callback, quiet=False, worker=None):
        super().perform(task, callback, quiet=quiet and not self._initializing, worker=worker)
        if isinstance(self.worker, BenchmarkWorker):
            self.cancel.setText('Force cancel benchmark')
        if self._initializing and self.worker:
            self.worker.progress.connect(self.initialization_progress)

    def initialization_progress(self, message):
        self.initialization_message.setText(message)
        if "Embedding documents" in message:
            import re
            match = re.search(r"–(\d+) / (\d+)", message)
            if match:
                self.initialization_bar.setValue(35 + int(50 * int(match[1]) / int(match[2])))
        elif "Loaded saved index" in message or "Index ready" in message:
            self.initialization_bar.setValue(90)

    def setup_tab(self):
        super().setup_tab()
        layout = self.tabs.widget(0).layout()
        group = QGroupBox("RAG algorithms and indexes")
        body = QVBoxLayout(group)
        note = QLabel("Indexes initialize automatically when the app opens or embedding settings change. All six algorithms share the saved index; no duplicate document embedding is needed.")
        note.setWordWrap(True)
        body.addWidget(note)
        self.rag_index_table = table_widget()
        self.rag_index_table.setMinimumHeight(210)
        body.addWidget(self.rag_index_table)
        actions = QHBoxLayout()
        # Move the existing preparation action into the RAG section.
        old_actions = layout.itemAt(1).layout()
        for i in range(old_actions.count()):
            widget = old_actions.itemAt(i).widget()
            if isinstance(widget, QPushButton) and widget.text() == "Load / reuse index":
                old_actions.removeWidget(widget)
                widget.setText("Refresh / retry indexes")
                widget.setProperty("color", "blue")
                actions.addWidget(widget)
                break
        self.colored("Reset to defaults", self.reset_defaults, old_actions, "gray")
        body.addLayout(actions)
        general = QWidget()
        general_layout = QVBoxLayout(general)
        dataset_group = layout.takeAt(0).widget()
        general_layout.addWidget(dataset_group)
        general_layout.addLayout(layout.takeAt(0).layout())
        general_layout.addStretch()
        info = layout.takeAt(0).widget()
        indexes = QWidget()
        indexes_layout = QVBoxLayout(indexes)
        indexes_layout.addWidget(group, 1)
        indexes_layout.addWidget(info)
        self.index_info.setMaximumHeight(130)
        self.setup_pages = QTabWidget()
        for page, title in ((general, "General"), (indexes, "RAG indexes")):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            self.setup_pages.addTab(scroll, title)
        layout.addWidget(self.setup_pages)

    def refresh_rag_readiness(self, *_):
        reports_root = str(Path(self.reports_path.text()).resolve()) if self.include_reports.isChecked() else None
        ready = self.engine is not None and self.engine.index_spec == Engine.index_settings(self.settings()) and getattr(self.engine, 'dataset_root', '') == str(Path(self.dataset_path.text()).resolve()) and getattr(self.engine, 'reports_root', None) == reports_root
        rows = [{"method": method, "description": METHOD_DETAILS[method][0], "index": METHOD_DETAILS[method][1],
                 "status": "Ready · saved index reused" if ready and self.engine.cache_hit else "Ready · index prepared" if ready else "Waiting for initialization"} for method in METHODS]
        fill_table(self.rag_index_table, rows, [("method", "RAG algorithm"), ("description", "How it retrieves"), ("index", "Index required"), ("status", "Status")])
        self.chat_rag.setEnabled(ready and not self._busy)
        for i in range(self.chat_rag.count()):
            self.chat_rag.model().item(i).setEnabled(ready)
        for i in range(self.test_rags.count()):
            item = self.test_rags.item(i)
            flags = item.flags()
            item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled if ready else flags & ~Qt.ItemFlag.ItemIsEnabled)

    def reset_defaults(self):
        if self._busy:
            return
        self.dataset_path.setText(str(ROOT / "datasets" / "CARE_To_Compare"))
        self.include_reports.setChecked(True)
        self.reports_path.setText(str(ROOT / 'datasets' / 'technician_reports'))
        self.mode.setCurrentIndex(0)
        self.host.setText("http://localhost:11434")
        default_embed = "nomic-embed-text:latest" if self.embedding_model.findText("nomic-embed-text:latest") >= 0 else "nomic-embed-text"
        self.embedding_model.setCurrentText(default_embed)
        self.k.setValue(3)
        self.candidates.setValue(10)
        self.seed.setValue(42)
        self.temperature.setValue(0)
        self.run_count.setValue(24)
        if self.chat_model.findText("qwen2.5:7b") >= 0:
            self.chat_model.setCurrentText("qwen2.5:7b")
        elif self.chat_model.count():
            self.chat_model.setCurrentIndex(0)
        self.answer_model.setCurrentText(self.chat_model.currentText())
        self.chat_rag.setCurrentText("Dense")
        for i in range(self.test_rags.count()):
            self.test_rags.item(i).setCheckState(Qt.CheckState.Checked if i < 3 else Qt.CheckState.Unchecked)
        for i in range(self.test_models.count()):
            item = self.test_models.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.text() == self.chat_model.currentText() else Qt.CheckState.Unchecked)
        self.refresh_rag_readiness()
        self.schedule_auto_index()
        self.status.setText("Defaults restored. Saved indexes and results are kept; matching indexes load automatically.")

    def colored(self, text, callback, layout, color="blue"):
        button = self.button(text, callback, layout)
        button.setProperty("color", color)
        return button

    def ask_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split)
        sidebar = QWidget()
        sidebar.setMinimumWidth(190)
        sidebar.setMaximumWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.addWidget(QLabel('Your chats'))
        self.colored('New chat', self.new_chat, sidebar_layout, 'blue')
        self.chat_list = QListWidget()
        self.chat_list.setWordWrap(True)
        self.chat_list.currentItemChanged.connect(self.select_chat)
        sidebar_layout.addWidget(self.chat_list, 1)
        self.colored('Delete chat', self.delete_chat, sidebar_layout, 'orange')
        self.chat_save_status = QLabel('Saved automatically on this computer')
        self.chat_save_status.setWordWrap(True)
        sidebar_layout.addWidget(self.chat_save_status)
        split.addWidget(sidebar)
        conversation = QWidget()
        layout = QVBoxLayout(conversation)
        split.addWidget(conversation)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 1000])
        row = QHBoxLayout()
        row.addWidget(QLabel("Model"))
        self.chat_model = QComboBox()
        self.chat_model.setEditable(True)
        self.chat_model.addItem("qwen2.5:7b")
        row.addWidget(self.chat_model, 1)
        row.addWidget(QLabel("RAG"))
        self.chat_rag = QComboBox()
        self.chat_rag.addItems(METHODS)
        row.addWidget(self.chat_rag)
        layout.addLayout(row)
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll, 1)
        note = QLabel("Ask about turbine records or repairs. Repair and cause questions automatically search technician PDF reports. Show evidence to inspect the report and page.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.question = MessageEditor()
        self.question.send_requested.connect(self.ask)
        self.question.setMaximumHeight(85)
        self.question.setPlaceholderText("Type your question… Enter to send · Shift+Enter for a new line")
        layout.addWidget(self.question)
        actions = QHBoxLayout()
        self.colored("Send message", self.ask, actions, "blue")
        actions.addStretch()
        layout.addLayout(actions)
        self.tabs.addTab(tab, "Chat")
        self.mutating_controls.extend([self.chat_model, self.chat_rag, self.chat_list])
        self.typing_label = QLabel()
        self.typing_label.setMinimumWidth(430)
        self.typing_label.setMaximumWidth(980)
        self.typing_label.setStyleSheet("background: #ecfdf5; border-radius: 12px; padding: 14px; color: #15803d;")
        self.typing_label.hide()
        self.typing_timer = QTimer(self)
        self.typing_timer.setInterval(350)
        self.typing_timer.timeout.connect(self.animate_typing)
        self.typing_step = 0
        self.chat_save_timer = QTimer(self)
        self.chat_save_timer.setSingleShot(True)
        self.chat_save_timer.setInterval(300)
        self.chat_save_timer.timeout.connect(self.save_chat_settings)
        self.question.textChanged.connect(self.schedule_chat_save)
        self.chat_model.currentTextChanged.connect(self.schedule_chat_save)
        self.chat_rag.currentTextChanged.connect(self.schedule_chat_save)
        chat_id = self.chat_store.active()
        if chat_id is None:
            existing = self.chat_store.list_chats()
            chat_id = existing[0]['id'] if existing else self.chat_store.create(self.chat_model.currentText(), self.chat_rag.currentText())
        self.load_chat(chat_id)

    def schedule_chat_save(self, *_):
        if not self._restoring_chat and not self._initializing:
            self.chat_save_timer.start()

    def save_chat_settings(self):
        if self.active_chat_id and not self._restoring_chat:
            try:
                self.chat_store.save_settings(self.active_chat_id, self.chat_model.currentText(), self.chat_rag.currentText(), self.question.toPlainText())
                self.chat_save_status.setText('Saved automatically on this computer')
            except Exception as exc:
                self.chat_save_status.setText(f'Could not save chat: {exc}')

    def refresh_chat_list(self):
        self.chat_list.blockSignals(True)
        self.chat_list.clear()
        for chat in self.chat_store.list_chats():
            item = QListWidgetItem(chat['title'])
            item.setData(Qt.ItemDataRole.UserRole, chat['id'])
            item.setToolTip(f"{chat['title']}\n{chat['model']} · {chat['rag']}")
            self.chat_list.addItem(item)
            if chat['id'] == self.active_chat_id:
                self.chat_list.setCurrentItem(item)
        self.chat_list.blockSignals(False)

    def load_chat(self, chat_id):
        self.chat_save_timer.stop()
        self._restoring_chat = True
        try:
            chat = self.chat_store.load(chat_id)
            self.clear_chat()
            self.active_chat_id = chat_id
            self.chat_store.set_active(chat_id)
            self.chat_model.setCurrentText(chat['model'])
            self.chat_rag.setCurrentText(chat['rag'])
            self.question.setPlainText(chat['draft'])
            for message in chat['messages']:
                role = 'You' if message['role'] == 'You' else APP_NAME
                self.bubble(role, message['text'], message['metadata'], message['sources'], persist=False)
            self.refresh_chat_list()
        finally:
            self._restoring_chat = False
        if self.engine is not None and not self.chat_messages:
            self.chat_greeting()

    def select_chat(self, item, previous=None):
        if self._busy or self._restoring_chat or item is None:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        if chat_id != self.active_chat_id:
            self.save_chat_settings()
            self.load_chat(chat_id)

    def new_chat(self):
        if self._busy:
            return
        self.save_chat_settings()
        chat_id = self.chat_store.create(self.chat_model.currentText(), self.chat_rag.currentText())
        self.load_chat(chat_id)
        self.question.setFocus()

    def delete_chat(self):
        if self._busy or not self.active_chat_id:
            return
        self.chat_save_timer.stop()
        self.chat_store.delete(self.active_chat_id)
        existing = self.chat_store.list_chats()
        chat_id = existing[0]['id'] if existing else self.chat_store.create(self.chat_model.currentText(), self.chat_rag.currentText())
        self.load_chat(chat_id)

    def chat_greeting(self):
        self.bubble(APP_NAME, 'Ready. Choose a model and a RAG algorithm, then send a question.',
                    'Offline demo · extractive replies' if self.engine.settings.mode == 'offline' else 'Local Ollama')

    def animate_typing(self):
        self.typing_step = self.typing_step % 3 + 1
        self.typing_label.setText(f'{APP_NAME} is typing ' + "● " * self.typing_step)

    def stop_typing(self):
        self.typing_timer.stop()
        self.chat_layout.removeWidget(self.typing_label)
        self.typing_label.hide()

    def bubble(self, role, text, metadata="", sources=None, persist=True):
        row = QHBoxLayout()
        frame = QFrame()
        frame.setMinimumWidth(430)
        frame.setMaximumWidth(980)
        frame.setStyleSheet("QFrame { background: " + ("#dbeafe" if role == "You" else "#ecfdf5") + "; border-radius: 12px; }")
        body = QVBoxLayout(frame)
        title = QLabel(role + (" · " + metadata if metadata else ""))
        title.setWordWrap(True)
        body.addWidget(title)
        message = QLabel(text)
        message.setTextFormat(Qt.TextFormat.PlainText)
        message.setWordWrap(True)
        message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(message)
        if sources:
            evidence = QTextBrowser()
            evidence.setMaximumHeight(220)
            evidence.setOpenLinks(False)
            evidence.setOpenExternalLinks(False)
            evidence.anchorClicked.connect(self.open_evidence_pdf)
            blocks = []
            for source in sources:
                path = Path(source.get('source', ''))
                link = ''
                if path.suffix.lower() == '.pdf':
                    url = QUrl.fromLocalFile(str(path.resolve())).toString()
                    link = f'<br><a href="{html.escape(url, quote=True)}">Open PDF</a>'
                blocks.append(f"<p><b>[{html.escape(source['id'])}] {html.escape(source.get('title', ''))}</b>{link}</p>"
                              f"<p>{html.escape(source['text']).replace(chr(10), '<br>')}<br>"
                              f"<b>Source:</b> {html.escape(source.get('source', ''))}</p>")
            evidence.setHtml(''.join(blocks))
            evidence.hide()
            toggle = QPushButton("Show evidence")
            toggle.clicked.connect(lambda: (evidence.setVisible(not evidence.isVisible()), toggle.setText("Hide evidence" if evidence.isVisible() else "Show evidence")))
            body.addWidget(toggle)
            body.addWidget(evidence)
        if role == "You":
            row.addStretch()
            row.addWidget(frame)
        else:
            row.addWidget(frame)
            row.addStretch()
        self.chat_layout.addLayout(row)
        self.chat_messages.append({"role": role, "text": text, "metadata": metadata, 'sources': sources or []})
        if persist and not self._restoring_chat and self.active_chat_id:
            try:
                self.chat_store.append(self.active_chat_id, role, text, metadata, sources)
                self.refresh_chat_list()
            except Exception as exc:
                self.chat_save_status.setText(f'Could not save message: {exc}')
        QTimer.singleShot(0, lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))

    def open_evidence_pdf(self, url):
        if not url.isLocalFile():
            self.status.setText('This evidence link is not a local PDF file.')
            return
        path = Path(url.toLocalFile())
        if path.suffix.lower() != '.pdf' or not path.is_file():
            self.status.setText(f'PDF report not found: {path}')
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            self.status.setText('Could not open the report. Set a default PDF viewer in Windows.')
        else:
            self.status.setText(f'Opened PDF report: {path.name}')

    def clear_chat(self):
        self.stop_typing()
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.layout():
                child = item.layout()
                while child.count():
                    widget = child.takeAt(0).widget()
                    if widget:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
                child.deleteLater()
        self.chat_messages = []

    def ask(self):
        if self._busy or not self.question.toPlainText().strip():
            return
        def action():
            model, rag = self.chat_model.currentText().strip(), self.chat_rag.currentText()
            self.answer_model.setCurrentText(model)
            engine = self.current_engine()
            query = self.question.toPlainText().strip()
            if not query:
                raise ValueError("Type a question first.")
            self.bubble("You", query)
            self.question.clear()
            self.save_chat_settings()
            self.chat_pending = True
            self.chat_layout.addWidget(self.typing_label, 0, Qt.AlignmentFlag.AlignLeft)
            self.animate_typing()
            self.typing_label.show()
            self.typing_timer.start()
            QTimer.singleShot(0, lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))
            self.perform(lambda p, c: engine.compare(query, p, c, order=[rag]),
                         lambda rows: self.chat_reply(rows, model if engine.settings.mode == "ollama" else "offline-demo", rag))
        self.guard(action)

    def chat_reply(self, rows, model, rag):
        self.stop_typing()
        self.chat_pending = False
        self.query_results = rows
        row = rows[0]
        scope = ' · Technician PDF reports' if row.get('evidence_scope') == 'technician_reports' else ''
        self.bubble(APP_NAME, row["answer"], f"{model} · {rag} · {row['total_ms']/1000:.1f} s{scope}", row["sources"])
        self.status.setText("Reply ready. Show evidence to inspect the retrieved sources.")

    def task_error(self, message):
        self.stop_typing()
        if self._initializing:
            self._init_failed = True
            self.initialization_message.setText("Initialization could not finish: " + message + "\nCheck Setup, start Ollama if needed, then retry. Offline demo can also initialize without Ollama.")
        if self.chat_pending:
            self.chat_pending = False
            self.bubble(APP_NAME, message, "Request failed")
        self.status.setText(message)

    def benchmark_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("1. Choose models and algorithms  →  2. Prepare validation questions  →  3. Run tests sequentially"))
        selections = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Models to test (check one or more)"))
        self.test_models = QListWidget()
        self.test_models.setMaximumHeight(125)
        left.addWidget(self.test_models)
        selections.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("RAG algorithms to test"))
        self.test_rags = QListWidget()
        for method in METHODS:
            item = QListWidgetItem(method)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if method in METHODS[:3] else Qt.CheckState.Unchecked)
            self.test_rags.addItem(item)
        self.test_rags.setMaximumHeight(170)
        right.addWidget(self.test_rags)
        selections.addLayout(right, 1)
        layout.addLayout(selections)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("Validation size"))
        self.run_count = QSpinBox()
        self.run_count.setRange(1, 1000)
        self.run_count.setValue(24)
        actions.addWidget(self.run_count)
        self.colored("Generate validation", self.generate_questions, actions, "blue")
        self.colored("Import validation", self.import_questions, actions, "gray")
        self.colored("Export validation", self.export_questions, actions, "green")
        layout.addLayout(actions)
        note = QLabel("Generated questions are balanced by farm and source type, but need human review before thesis reporting. Edit the question, reference answer, and relevant source IDs directly in the table.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.question_table = table_widget()
        self.question_table.setEditTriggers(self.question_table.EditTrigger.DoubleClicked | self.question_table.EditTrigger.EditKeyPressed)
        layout.addWidget(self.question_table, 1)
        self.validation_status = QLabel("No validation dataset yet.")
        layout.addWidget(self.validation_status)
        run_row = QHBoxLayout()
        self.colored("Start sequential benchmark", self.run_benchmark, run_row, "orange")
        run_row.addWidget(QLabel("One complete results folder is saved for each model × RAG pair."))
        layout.addLayout(run_row)
        self.tabs.addTab(tab, "Benchmark")
        self.mutating_controls.extend([self.test_models, self.test_rags, self.question_table, self.run_count])

    def refresh_questions(self):
        fill_table(self.question_table, self.questions, [("id", "ID"), ("question", "Question"),
            ("reference_answer", "Reference answer"), ("relevant_ids", "Relevant source IDs (JSON list)"), ("origin", "Origin")])
        if hasattr(self, "validation_status"):
            self.validation_status.setText(f"{len(self.questions)} validation questions. The same set will be used for every selected model and RAG.")

    def collect_questions(self):
        if not self.questions:
            raise ValueError("Generate or import validation questions first.")
        rows = []
        for r in range(self.question_table.rowCount()):
            cells = [self.question_table.item(r, c).text().strip() for c in range(5)]
            rows.append(dict(zip(("id", "question", "reference_answer", "relevant_ids", "origin"), cells)))
        # Reuse the import validator without creating a permanent intermediate file.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            validated = load_questions(path, [d.id for d in self.docs])
        if any(not q["reference_answer"] for q in validated):
            raise ValueError("Every question needs a reference answer.")
        self.questions = validated
        return validated

    def generate_questions(self):
        def action():
            self.current_engine()
            self.questions = validation_set(self.events, self.features, self.run_count.value(), self.seed.value(), self.docs)
            self.refresh_questions()
        self.guard(action)

    def export_questions(self):
        def action():
            self.collect_questions()
            MainWindow.export_questions(self)
        self.guard(action)

    def task_finished(self):
        self.stop_typing()
        super().task_finished()
        self.cancel.setText('Cancel task')
        self.refresh_rag_readiness()
        self.refresh_runs()
        if self._initializing:
            if self._init_failed:
                self._initializing = False
                self.tabs.setEnabled(True)
                self.retry_initialization.show()
            elif self._init_stage == "models":
                self._init_stage = "index"
                QTimer.singleShot(0, self.build_index)
            else:
                self.initialization_bar.setValue(100)
                self.initialization_message.setText("Ready: models, dataset, RAG indexes and saved results loaded.")
                self._initializing = False
                self.tabs.setEnabled(True)
                self.initialization_panel.hide()
                self.status.setText("Ready. Dataset, RAG indexes and saved results loaded automatically.")
        if not self._initializing and self._auto_index_pending:
            self.auto_index_timer.start()

    def closeEvent(self, event):
        super().closeEvent(event)
        if event.isAccepted():
            self.save_chat_settings()
            self.chat_save_timer.stop()
            self._closed = True
            self.auto_index_timer.stop()
            self.scan_timer.stop()

    def cancel_task(self):
        if isinstance(self.worker, BenchmarkWorker):
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            self.status.setText('Stopping benchmark immediately. Completed runs will remain saved.')
        else:
            super().cancel_task()

    def index_ready(self, result):
        super().index_ready(result)
        self.refresh_rag_readiness()
        if not self.chat_messages:
            self.chat_greeting()

    @staticmethod
    def checked(widget):
        return [widget.item(i).text() for i in range(widget.count()) if widget.item(i).checkState() == Qt.CheckState.Checked]

    def run_benchmark(self):
        def action():
            engine = self.current_engine()
            questions = self.collect_questions()
            models = self.checked(self.test_models) if engine.settings.mode == "ollama" else ["offline-demo"]
            methods = self.checked(self.test_rags)
            if not models or not methods:
                raise ValueError("Check at least one model and one RAG algorithm.")
            digest = self.corpus_hash
            worker = BenchmarkWorker(engine, digest, questions, models, methods, ROOT / 'results', ROOT / '.cache' / 'benchmark_jobs')
            self.perform(None, self.batch_ready, worker=worker)
        self.guard(action)

    def batch_ready(self, folders):
        self.refresh_runs()
        self.tabs.setCurrentIndex(4)
        failed = sum(Path(folder).name.endswith("_failed") for folder in folders)
        self.status.setText(f"Finished {len(folders)} runs ({failed} failed). Select successful runs in Results to compare or export.")

    def check_ollama(self, automatic=False):
        if self._closed:
            return
        if automatic and self._initializing and self.settings().mode == "offline":
            self.build_index()
            return
        host = self.host.text().strip()
        def task(progress, cancelled):
            response = requests.get(host.rstrip("/") + "/api/tags", timeout=(5, 10))
            response.raise_for_status()
            return response.json().get("models", [])
        def done(models):
            chat = [m["name"] for m in models if "completion" in m.get("capabilities", [])]
            embeds = [m["name"] for m in models if "embedding" in m.get("capabilities", [])]
            if not chat:
                chat = [m["name"] for m in models if "embed" not in m["name"]]
            if not embeds:
                embeds = [m["name"] for m in models if "embed" in m["name"]]
            for combo, names in ((self.embedding_model, embeds), (self.chat_model, chat), (self.answer_model, chat)):
                previous = combo.currentText()
                combo.clear()
                combo.addItems(names)
                canonical = previous if ":" in previous else previous + ":latest"
                if canonical in names:
                    combo.setCurrentText(canonical)
                elif "qwen2.5:7b" in names:
                    combo.setCurrentText("qwen2.5:7b")
            selected = set(self.checked(self.test_models))
            self.test_models.clear()
            for name in chat:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if name in selected or not selected and name == self.chat_model.currentText() else Qt.CheckState.Unchecked)
                self.test_models.addItem(item)
            self.status.setText(f"Found {len(chat)} chat models and {len(embeds)} embedding models.")
        self.perform(task, done, quiet=automatic)

    def results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        actions = QHBoxLayout()
        self.colored("Refresh saved runs", self.refresh_runs, actions, "gray")
        self.colored("Compare selected runs", self.compare_runs, actions, "blue")
        self.comparison_name = QLineEdit("comparison")
        self.comparison_name.setPlaceholderText("Comparison export name")
        actions.addWidget(self.comparison_name)
        self.colored("Export comparison + graphs", self.export_selected, actions, "green")
        layout.addLayout(actions)
        self.saved_runs = QListWidget()
        self.saved_runs.setMaximumHeight(120)
        layout.addWidget(self.saved_runs)
        self.run_info = QLabel("Results are scanned automatically. Check the runs you want to compare.")
        self.run_info.setWordWrap(True)
        layout.addWidget(self.run_info)
        self.summary_table = table_widget()
        self.summary_table.setMaximumHeight(140)
        layout.addWidget(self.summary_table)
        row = QHBoxLayout()
        row.addWidget(QLabel("Graph"))
        self.metric_choice = QComboBox()
        for metric in METRICS:
            self.metric_choice.addItem(metric_label(metric), metric)
        self.metric_choice.currentIndexChanged.connect(self.draw_comparison)
        row.addWidget(self.metric_choice, 1)
        layout.addLayout(row)
        self.figure = Figure(figsize=(10, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(250)
        layout.addWidget(self.canvas, 1)
        self.result_table = table_widget()
        self.result_table.setMaximumHeight(110)
        layout.addWidget(self.result_table)
        self.aggregate_summary = []
        self.aggregate_rows = []
        self.tabs.addTab(tab, "Results")
        self.mutating_controls.extend([self.saved_runs, self.comparison_name])

    def refresh_runs(self):
        if not hasattr(self, "saved_runs") or self._busy:
            return
        selected = {self.saved_runs.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.saved_runs.count()) if self.saved_runs.item(i).checkState() == Qt.CheckState.Checked}
        self.run_entries, errors = scan_runs(ROOT / "results")
        self.saved_runs.clear()
        for index, entry in enumerate(self.run_entries):
            key = entry["path"] + "|" + entry["rag"]
            item = QListWidgetItem(f"{entry['run_id']} · {entry['label']} · {entry['summary']['questions']} questions")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setCheckState(Qt.CheckState.Checked if key in selected else Qt.CheckState.Unchecked)
            self.saved_runs.addItem(item)
        self.run_info.setText(f"{len(self.run_entries)} saved run/algorithm entries. Automatically refreshed every 15 seconds." + (f" {len(errors)} failed or unreadable runs excluded." if errors else ""))

    def selected_entries(self):
        return [entry for i, entry in enumerate(self.run_entries) if self.saved_runs.item(i).checkState() == Qt.CheckState.Checked]

    def compare_runs(self):
        def action():
            entries = self.selected_entries()
            self.aggregate_summary, self.aggregate_rows = aggregate(entries)
            fill_table(self.summary_table, self.aggregate_summary, [("label", "Model / RAG"), ("runs", "Runs"),
                ("recall_at_k", "Recall ↑"), ("answer_f1", "Answer F1 ↑"), ("citation_validity", "Citation ID validity ↑"), ("total_ms", "Latency ms ↓")])
            fill_table(self.result_table, self.aggregate_rows, [("model", "Model"), ("method", "RAG"), ("question_id", "Question"), ("answer_f1", "F1 ↑"), ("answer", "Reply")])
            self.draw_comparison()
            self.run_info.setText(f"Comparing {len(entries)} selected runs. Repeated runs averaged per question. ↑ higher is better; ↓ lower is better.")
        self.guard(action)

    def draw_comparison(self):
        if getattr(self, "aggregate_summary", None):
            plot_metric(self.aggregate_summary, self.metric_choice.currentData(), self.figure)
            self.canvas.draw()

    def export_selected(self):
        def action():
            folder = export_comparison(ROOT / "results", self.selected_entries(), self.comparison_name.text())
            self.status.setText(f"Comparison exported: {folder}")
        self.guard(action)


def launch():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "benchmarks").mkdir(exist_ok=True)
    app = QApplication(sys.argv)
    window = Workbench()
    window.show()
    sys.exit(app.exec())
