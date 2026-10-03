import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from windrag.benchmark_worker import BenchmarkWorker
from windrag.data import Document
from windrag.engine import Engine, Settings
from windrag.experiments import scan_runs

app = QApplication.instance() or QApplication([])
DOCS = [Document('a', 'Gearbox', 'gearbox failure', 'events.csv', 'event', 'A', 'failure'),
        Document('b', 'Blade', 'blade inspection', 'events.csv', 'event', 'A', 'inspection')]
QUESTIONS = [{'id': 'q1', 'question': 'gearbox failure', 'reference_answer': 'failure', 'relevant_ids': ['a']}]


class ForceCancelTests(unittest.TestCase):
    def test_force_cancel_interrupts_hanging_http_and_preserves_completed_run(self):
        blocked = threading.Event()
        release = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/api/embed':
                    data = {'embeddings': [[1., 0.] if 'gearbox' in text else [0., 1.] for text in payload['input']]}
                else:
                    if payload['model'] == 'blocked-model':
                        blocked.set()
                        release.wait(30)
                    data = {'message': {'content': 'failure [a]'}}
                content = json.dumps(data).encode()
                try:
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                except OSError:
                    pass  # Expected disconnect when the benchmark process is stopped.
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        worker = None
        try:
            engine = Engine(DOCS, Settings(mode='ollama', host=f'http://127.0.0.1:{server.server_port}', top_k=1, candidates=2))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                worker = BenchmarkWorker(engine, 'test-corpus', QUESTIONS, ['fast-model', 'blocked-model'], ['BM25'], root / 'results', root / 'jobs')
                failures = []
                successes = []
                worker.failed.connect(failures.append)
                worker.succeeded.connect(successes.append)
                worker.start()
                deadline = time.monotonic() + 30
                while not blocked.is_set() and worker.isRunning() and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                self.assertTrue(blocked.is_set(), failures)
                self.assertEqual(len(scan_runs(root / 'results')[0]), 1)
                start = time.monotonic()
                worker.requestInterruption()
                self.assertTrue(worker.wait(3000), 'Cancellation waited for the model')
                app.processEvents()
                self.assertLess(time.monotonic() - start, 3)
                self.assertFalse(release.is_set(), 'The model finished before cancellation')
                self.assertIsNotNone(worker.process.poll())
                self.assertEqual(successes, [])
                self.assertTrue(any('cancelled immediately' in text for text in failures))
                entries, errors = scan_runs(root / 'results')
                self.assertEqual(len(entries), 1)
                self.assertFalse(errors)
                self.assertEqual(entries[0]['model'], 'fast-model')
                self.assertEqual(len(list((Path(entries[0]['path']).parent / 'graphs').glob('*.png'))), 13)
                self.assertFalse(worker.staging.exists())
        finally:
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(5000)
            release.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_scanner_ignores_in_progress_run(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / '.in_progress' / 'job' / 'unfinished'
            folder.mkdir(parents=True)
            (folder / 'run.json').write_text('{}')
            self.assertEqual(scan_runs(directory), ([], []))
