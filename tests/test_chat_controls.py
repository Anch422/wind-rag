import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import tempfile
from unittest.mock import patch
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton, QTextBrowser
from windrag.workbench import MessageEditor, Workbench
from windrag.ui import fill_table, table_widget
from windrag.engine import Engine, Settings, METHODS
from test_more_rags import DOCS
from pathlib import Path

app = QApplication.instance() or QApplication([])


class ChatControlsTests(unittest.TestCase):
    def setUp(self):
        self.chat_folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.chat_folder.cleanup)

    def make_window(self):
        return Workbench(chat_database=Path(self.chat_folder.name) / 'chats.sqlite3')

    def test_pdf_evidence_link_opens_correct_local_file(self):
        import tempfile
        with patch.object(Workbench, 'check_ollama'), patch.object(Workbench, 'refresh_runs'), tempfile.TemporaryDirectory() as directory:
            window = self.make_window()
            report = Path(directory) / 'Report A & B.pdf'
            report.write_bytes(b'%PDF-1.4\n')
            window.bubble('Assistant', 'Answer', sources=[{'id': 'REPORT:a:p2:c1', 'title': 'Report <A>', 'text': '<script>text</script>', 'source': str(report)},
                                                        {'id': 'A:event:1', 'title': 'Event', 'text': 'Metadata', 'source': 'events.csv'}])
            frame = window.chat_layout.itemAt(0).layout().itemAt(0).widget()
            evidence = frame.findChild(QTextBrowser)
            self.assertEqual(evidence.toPlainText().count('Open PDF'), 1)
            self.assertIn('<script>text</script>', evidence.toPlainText())
            with patch('windrag.workbench.QDesktopServices.openUrl', return_value=True) as opener:
                evidence.anchorClicked.emit(QUrl.fromLocalFile(str(report)))
                self.assertEqual(Path(opener.call_args.args[0].toLocalFile()), report.resolve())
                window.open_evidence_pdf(QUrl.fromLocalFile(str(report.parent / 'missing.pdf')))
                window.open_evidence_pdf(QUrl('https://example.com/report.pdf'))
                self.assertEqual(opener.call_count, 1)
            window.close()

    def test_setup_layout_is_readable_at_multiple_sizes(self):
        with patch.object(Workbench, 'check_ollama'), patch.object(Workbench, 'refresh_runs'):
            window = self.make_window()
            window._initializing = False
            window.initialization_panel.hide()
            window.tabs.setEnabled(True)
            window.show()
            for width, height in ((1000, 700), (1400, 940), (1920, 1080)):
                window.resize(width, height)
                window.setup_pages.setCurrentIndex(0)
                app.processEvents()
                for control in (window.dataset_path, window.mode, window.host, window.embedding_model, window.k, window.candidates, window.seed, window.temperature):
                    self.assertGreaterEqual(control.height(), control.fontMetrics().height() + 8)
                window.setup_pages.setCurrentIndex(1)
                app.processEvents()
                button = next(button for button in window.findChildren(QPushButton) if button.text() == 'Refresh / retry indexes')
                self.assertFalse(window.rag_index_table.geometry().intersects(button.geometry()))
                self.assertGreaterEqual(window.rag_index_table.height(), 210)
            window.close()

    def test_rag_readiness_and_reset(self):
        with patch.object(Workbench, 'check_ollama'), patch.object(Workbench, 'refresh_runs'):
            window = self.make_window()
            window._initializing = False
            window.tabs.setEnabled(True)
            self.assertFalse(window.chat_rag.isEnabled())
            self.assertTrue(all(not window.test_rags.item(i).flags() & Qt.ItemFlag.ItemIsEnabled for i in range(len(METHODS))))
            window.mode.setCurrentIndex(1)
            window.engine = Engine(DOCS, Settings(mode='offline'))
            window.engine.dataset_root = str(Path(window.dataset_path.text()).resolve())
            window.engine.reports_root = str(Path(window.reports_path.text()).resolve())
            window.refresh_rag_readiness()
            self.assertTrue(window.chat_rag.isEnabled())
            window.k.setValue(9)
            window.refresh_rag_readiness()
            self.assertTrue(window.chat_rag.isEnabled())
            window.dataset_path.setText('different-dataset')
            self.assertFalse(window.chat_rag.isEnabled())
            window.reset_defaults()
            self.assertEqual(window.k.value(), 3)
            self.assertEqual(window.candidates.value(), 10)
            self.assertEqual(window.seed.value(), 42)
            self.assertEqual(window.mode.currentIndex(), 0)
            self.assertEqual(window.checked(window.test_rags), list(METHODS[:3]))
            self.assertFalse(window.chat_rag.isEnabled())
            self.assertIsNotNone(window.engine)
            window.close()

    def test_enter_sends_and_shift_enter_inserts_newline(self):
        editor = MessageEditor()
        sent = []
        editor.send_requested.connect(lambda: sent.append(editor.toPlainText()))
        editor.setPlainText('Question')
        QTest.keyClick(editor, Qt.Key.Key_Return)
        self.assertEqual(sent, ['Question'])
        editor.moveCursor(editor.textCursor().MoveOperation.End)
        QTest.keyClick(editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(editor.toPlainText(), 'Question\n')
        self.assertEqual(len(sent), 1)
        editor.close()

    def test_typing_lifecycle_and_busy_draft(self):
        with patch.object(Workbench, 'check_ollama'), patch.object(Workbench, 'refresh_runs'):
            window = self.make_window()
            window.typing_label.show()
            window.typing_timer.start()
            window.animate_typing()
            first = window.typing_label.text()
            QTest.qWait(400)
            self.assertNotEqual(first, window.typing_label.text())
            window._busy = True
            window.question.setPlainText('Next question')
            window.ask()
            self.assertEqual(window.question.toPlainText(), 'Next question')
            self.assertEqual(window.chat_messages, [])
            window.chat_pending = True
            window.task_error('Failed')
            self.assertFalse(window.typing_timer.isActive())
            self.assertTrue(window.typing_label.isHidden())
            self.assertFalse(window.chat_pending)
            window._busy = False
            window.close()

    def test_sort_keeps_original_event_identity(self):
        table = table_widget()
        table.setSortingEnabled(True)
        fill_table(table, [{'farm': 'B', 'event': 10}, {'farm': 'A', 'event': 2}], [('farm', 'Farm'), ('event', 'Event')])
        table.sortItems(1, Qt.SortOrder.AscendingOrder)
        self.assertEqual(table.item(0, 1).text(), '2')
        self.assertEqual(table.item(0, 0).text(), 'A')
        self.assertEqual(table.item(0, 0).data(Qt.ItemDataRole.UserRole), 1)
        table.close()
