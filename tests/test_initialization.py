import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import requests
from PyQt6.QtWidgets import QApplication
from windrag.workbench import Workbench
from windrag.engine import Ollama
from test_more_rags import DOCS

app = QApplication.instance() or QApplication([])


def wait_ready(window):
    deadline = time.monotonic() + 15
    while (window._initializing or window._busy) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert not window._busy and not window._initializing


class InitializationTests(unittest.TestCase):
    def test_startup_discovers_models_and_reuses_index_on_reopen(self):
        response = Mock()
        response.json.return_value = {'models': [
            {'name': 'nomic-embed-text:latest', 'capabilities': ['embedding'], 'digest': 'test'},
            {'name': 'qwen2.5:7b', 'capabilities': ['completion']}]}
        with tempfile.TemporaryDirectory() as directory, patch('windrag.ui.ROOT', Path(directory)), patch('windrag.ui.load_corpus', return_value=(DOCS, [], [], 'test-corpus')), patch('windrag.workbench.requests.get', return_value=response), patch.object(Ollama, 'embed', return_value=np.eye(3)) as embed:
            for cache_hit in (False, True):
                window = Workbench(chat_database=Path(directory) / 'chats.sqlite3')
                window.show()
                self.assertFalse(window.tabs.isEnabled())
                wait_ready(window)
                self.assertTrue(window.tabs.isEnabled())
                self.assertTrue(window.initialization_panel.isHidden())
                self.assertEqual(window.initialization_bar.value(), 100)
                self.assertEqual(window.engine.cache_hit, cache_hit)
                self.assertEqual(window.test_models.count(), 1)
                self.assertTrue(window.chat_rag.isEnabled())
                window.close()
            self.assertEqual(embed.call_count, 1)

    def test_failed_startup_can_recover_automatically_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as directory, patch('windrag.ui.ROOT', Path(directory)), patch('windrag.ui.load_corpus', return_value=(DOCS, [], [], 'test-corpus')), patch('windrag.workbench.requests.get', side_effect=requests.ConnectionError('Ollama unavailable')):
            window = Workbench(chat_database=Path(directory) / 'chats.sqlite3')
            window.show()
            wait_ready(window)
            self.assertTrue(window.tabs.isEnabled())
            self.assertFalse(window.retry_initialization.isHidden())
            self.assertIsNone(window.engine)
            window.mode.setCurrentIndex(1)
            deadline = time.monotonic() + 15
            while (window.auto_index_timer.isActive() or window._initializing or window._busy) and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            self.assertIsNotNone(window.engine)
            self.assertEqual(window.engine.settings.mode, 'offline')
            self.assertTrue(window.initialization_panel.isHidden())
            window.close()
