import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from windrag.workbench import Workbench
from windrag.engine import METHODS

ROOT = Path(__file__).resolve().parents[1]
app = QApplication([])
chat_folder = tempfile.TemporaryDirectory()
window = Workbench(chat_database=Path(chat_folder.name) / 'chats.sqlite3')
window.show()


def wait_task():
    deadline = time.monotonic() + 120
    while (window._busy or window._initializing) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not window._busy and not window._initializing, "Worker timeout"
    app.processEvents()


response = Mock()
response.json.return_value = {"models": [{"name": "nomic-embed-text:latest", "capabilities": ["embedding"]},
                                         {"name": "qwen2.5:7b", "capabilities": ["completion"]}]}
with patch("windrag.workbench.requests.get", return_value=response):
    window.mode.setCurrentIndex(1)
    app.processEvents()
    wait_task()
    window.check_ollama()
    wait_task()
assert window.chat_model.currentText() == "qwen2.5:7b"
assert window.test_models.count() == 1
assert window.engine
window.chat_rag.setCurrentText("Hybrid")
window.question.setPlainText('What does sensor_0 measure at Wind Farm A, and what is its unit?')
window.ask()
wait_task()
assert len(window.chat_messages) == 3
assert len(window.query_results) == 1
assert "Ambient temperature" in window.chat_messages[-1]["text"]
window.run_count.setValue(24)
window.generate_questions()
old_run_ids = {entry['run']['run_id'] for entry in window.run_entries}
for i in range(window.test_rags.count()):
    window.test_rags.item(i).setCheckState(Qt.CheckState.Checked)
window.run_benchmark()
wait_task()
assert f"Finished {len(METHODS)} runs" in window.status.text()
# Choose only this new validation set; all older results remain on disk.
chosen = 0
for i, entry in enumerate(window.run_entries):
    if entry['run']['run_id'] not in old_run_ids and len(entry["run"]["questions"]) == 24 and entry["run"]["config"]["top_k"] == 3:
        window.saved_runs.item(i).setCheckState(Qt.CheckState.Checked)
        chosen += 1
        if chosen == len(METHODS):
            break
assert chosen == len(METHODS)
window.compare_runs()
assert len(window.aggregate_summary) == len(METHODS)
window.comparison_name.setText("validated-offline-demo")
window.export_selected()
assert "Comparison exported" in window.status.text()
folder = ROOT / "docs"
comparison = max((ROOT / 'results' / 'comparisons').glob('*_validated-offline-demo'), key=lambda path: path.stat().st_mtime_ns)
for metric in ('recall_at_k', 'total_ms'):
    shutil.copy2(comparison / 'graphs' / f'{metric}.png', folder / f'benchmark-{metric}.png')
window.new_chat()
window.question.setPlainText("What's the problem with Wind Farm C?")
window.ask()
wait_task()
assert all(source['id'].startswith('C:event:') for source in window.query_results[0]['sources'])
window.tabs.setCurrentIndex(2)
app.processEvents()
window.grab().save(str(folder / "app-chat.png"))
window.tabs.setCurrentIndex(1)
app.processEvents()
window.grab().save(str(folder / 'app-dataset.png'))
window.tabs.setCurrentIndex(3)
app.processEvents()
window.grab().save(str(folder / "app-benchmark.png"))
window.tabs.setCurrentIndex(4)
app.processEvents()
window.grab().save(str(folder / "app-comparisons.png"))
print(json.dumps({"chat": "passed", "sequential_runs": len(METHODS), "questions_per_run": 24, "comparison_export": "passed", "screenshots": 4, "graph_examples": 2}, indent=2))
window.close()
chat_folder.cleanup()
