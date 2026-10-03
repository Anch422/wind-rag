"""Save offscreen screenshots of both Setup pages for visual review."""
import os
import tempfile
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from windrag.workbench import Workbench

app = QApplication([])
with patch.object(Workbench, 'check_ollama'), patch.object(Workbench, 'refresh_runs'), tempfile.TemporaryDirectory() as directory:
    window = Workbench(chat_database=Path(directory) / 'chats.sqlite3')
    window.resize(1400, 940)
    window.show()
    app.processEvents()
    window.grab().save(str(Path(__file__).resolve().parents[1] / 'docs' / 'app-initializing.png'))
    window._initializing = False
    window.initialization_panel.hide()
    window.tabs.setEnabled(True)
    for index, name in enumerate(('general', 'indexes')):
        window.setup_pages.setCurrentIndex(index)
        app.processEvents()
        path = Path(__file__).resolve().parents[1] / 'docs' / f'app-setup-{name}.png'
        window.grab().save(str(path))
        print(path)
    window.close()
