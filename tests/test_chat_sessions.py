import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from windrag.chat_store import ChatStore
from windrag.workbench import Workbench
from windrag.branding import APP_NAME

app = QApplication.instance() or QApplication([])


class ChatSessionsTests(unittest.TestCase):
    def test_store_reopens_and_deletion_cascades(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.sqlite3'
            store = ChatStore(path)
            first = store.create('model-a', 'Dense')
            source = {'id': 'report:1', 'source': 'repair.pdf', 'text': 'Replace bearing °C'}
            store.append(first, 'You', 'How do I repair the bearing?')
            store.append(first, APP_NAME, 'Replace it.', 'Timing', [source])
            store.save_settings(first, 'model-b', 'BM25', 'Unsent draft')
            reopened = ChatStore(path)
            self.assertEqual(reopened.active(), first)
            saved = reopened.load(first)
            self.assertEqual(saved['title'], 'How do I repair the bearing?')
            self.assertEqual(saved['messages'][1]['sources'], [source])
            self.assertEqual((saved['model'], saved['rag'], saved['draft']), ('model-b', 'BM25', 'Unsent draft'))
            other = store.create('model-c', 'Dense')
            reopened.delete(first)
            self.assertEqual(reopened.active(), other)
            with reopened.connection() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)
            self.assertEqual(len(reopened.list_chats()), 1)

    def test_sidebar_switch_delete_and_restart_restore(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(Workbench, 'check_ollama'), patch.object(Workbench, 'refresh_runs'):
            path = Path(folder) / 'history.sqlite3'
            window = Workbench(chat_database=path)
            window._initializing = False
            first = window.active_chat_id
            window.bubble('You', 'What failed?')
            sources = [{'id': 'REPORT:a:p2:c1', 'title': 'Repair report', 'text': 'Bearing repair', 'source': str(Path(folder) / 'repair.pdf')}]
            window.bubble(APP_NAME, 'The bearing.', 'Local model', sources)
            window.chat_model.setCurrentText('custom:7b')
            window.chat_rag.setCurrentText('BM25')
            window.question.setPlainText('Draft follow-up')
            window.new_chat()
            second = window.active_chat_id
            self.assertNotEqual(first, second)
            self.assertEqual(window.chat_messages, [])
            self.assertEqual(window.question.toPlainText(), '')
            for i in range(window.chat_list.count()):
                item = window.chat_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == first:
                    window.chat_list.setCurrentItem(item)
                    break
            self.assertEqual(window.active_chat_id, first)
            self.assertEqual(window.question.toPlainText(), 'Draft follow-up')
            self.assertEqual(window.chat_messages[1]['sources'], sources)
            window.close()
            reopened = Workbench(chat_database=path)
            self.assertEqual(reopened.active_chat_id, first)
            self.assertEqual(len(reopened.chat_messages), 2)
            self.assertEqual(reopened.chat_model.currentText(), 'custom:7b')
            self.assertEqual(reopened.chat_rag.currentText(), 'BM25')
            self.assertEqual(reopened.question.toPlainText(), 'Draft follow-up')
            reopened.delete_chat()
            self.assertEqual(reopened.active_chat_id, second)
            self.assertEqual(reopened.chat_messages, [])
            reopened.delete_chat()
            self.assertEqual(reopened.chat_list.count(), 1)
            self.assertNotEqual(reopened.active_chat_id, second)
            reopened.close()
