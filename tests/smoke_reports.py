"""Exercise automatic PDF loading, report chat and report validation in Qt."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
import time
import tempfile
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from windrag.workbench import Workbench

app = QApplication([])
chat_folder = tempfile.TemporaryDirectory()
window = Workbench(chat_database=Path(chat_folder.name) / 'chats.sqlite3')
window.mode.setCurrentIndex(1)
window.show()


def wait():
    deadline = time.monotonic() + 90
    while (window._initializing or window._busy) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not window._initializing and not window._busy
    assert not window._init_failed, window.status.text()
    app.processEvents()


wait()
assert len([doc for doc in window.docs if doc.kind == 'report']) == 90
window.chat_rag.setCurrentText('BM25')
window.question.setPlainText('How should I repair a damaged generator bearing at Wind Farm A, according to the technician reports?')
window.ask()
wait()
row = window.query_results[0]
assert row['evidence_scope'] == 'technician_reports'
assert all(source['kind'] == 'report' for source in row['sources'])
assert not any(word in row['answer'].lower() for word in ('mock', 'fictional', 'synthetic'))
assert 'bearing' in row['answer'].lower()
window.tabs.setCurrentIndex(2)
app.processEvents()
window.grab().save(str(Path(__file__).resolve().parents[1] / 'docs' / 'app-report-chat.png'))
window.run_count.setValue(24)
window.generate_questions()
assert any(q['relevant_ids'][0].startswith('REPORT:') for q in window.questions)
print(json.dumps({'automatic_pdf_load': 90, 'repair_chat': 'passed', 'report_validation': 'passed', 'index_reused': window.engine.cache_hit}, indent=2))
window.close()
chat_folder.cleanup()
