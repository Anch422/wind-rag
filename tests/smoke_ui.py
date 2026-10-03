"""Exercise actual threaded desktop workflows using Qt's offscreen renderer."""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import json
import time
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from windrag.ui import MainWindow

ROOT = Path(__file__).resolve().parents[1]
app = QApplication([])
window = MainWindow()
window.show()
window.mode.setCurrentIndex(1)


def wait_task():
    deadline = time.monotonic() + 90
    while window._busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not window._busy, "UI worker timed out"
    app.processEvents()


window.build_index()
wait_task()
assert window.engine is not None
assert window.event_table.rowCount() == 95
window.ask()
wait_task()
assert len(window.query_results) == 3
window.generate_questions()
assert len(window.questions) > 95
original_vectors = window.engine.vectors
window.k.setValue(3)
window.seed.setValue(99)
window.answer_model.setCurrentText("another-answer-model")
assert window.current_engine().vectors is original_vectors
question_count = len(window.questions)
window.build_index()
wait_task()
assert window.engine.cache_hit
assert len(window.questions) == question_count
window.run_count.setValue(12)
window.run_benchmark()
wait_task()
assert window.run_data and len(window.run_data["results"]) == 36
assert window.summary_table.rowCount() == 3
window.event_table.selectRow(0)
window.plot_sensor()
wait_task()
assert len(window.scada_figure.axes) == 1
folder = ROOT / "docs"
folder.mkdir(exist_ok=True)
window.tabs.setCurrentIndex(4)
app.processEvents()
window.grab().save(str(folder / "app-results.png"))
window.tabs.setCurrentIndex(0)
app.processEvents()
window.grab().save(str(folder / "app-setup.png"))
print(json.dumps({"ui": "passed", "indexed_documents": len(window.docs), "questions_run": 12,
                  "run_id": window.run_data["run_id"], "scada_plot": "passed"}, indent=2))
window.close()
