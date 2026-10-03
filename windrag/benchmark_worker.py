"""Qt supervisor for a force-cancellable isolated benchmark process."""
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import uuid
import joblib
from .ui import Worker


class BenchmarkWorker(Worker):
    def __init__(self, engine, corpus_hash, questions, models, methods, directory, jobs_dir):
        super().__init__(None)
        self.directory = Path(directory).resolve()
        self.jobs_dir = Path(jobs_dir).resolve()
        self.job_id = uuid.uuid4().hex
        self.staging = self.directory / '.in_progress' / self.job_id
        self.payload = dict(engine=engine, corpus_hash=corpus_hash, questions=questions,
                            models=models, methods=methods, directory=str(self.directory), staging=str(self.staging))
        self.process = None

    def run(self):
        payload_path = self.jobs_dir / (self.job_id + '.joblib')
        cancelled = False
        try:
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            self.progress.emit('Starting isolated benchmark. Force cancel is available throughout the run.')
            if self.isInterruptionRequested():
                raise InterruptedError('Benchmark cancelled immediately. Completed runs remain saved.')
            joblib.dump(self.payload, payload_path)
            if self.isInterruptionRequested():
                raise InterruptedError('Benchmark cancelled immediately. Completed runs remain saved.')
            self.process = subprocess.Popen([sys.executable, '-u', '-m', 'windrag.benchmark_process', str(payload_path)],
                                            cwd=str(Path(__file__).resolve().parents[1]), stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            messages = queue.Queue()
            def read_output():
                for line in self.process.stdout:
                    messages.put(line)
            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            folders = None
            error = None
            while self.process.poll() is None or reader.is_alive() or not messages.empty():
                if self.isInterruptionRequested():
                    cancelled = True
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=1)
                    break
                try:
                    line = messages.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self.progress.emit(line.strip())
                    continue
                if message['event'] == 'progress':
                    self.progress.emit(message['data'])
                elif message['event'] == 'done':
                    folders = message['data']
                elif message['event'] == 'error':
                    error = message['data']
            reader.join(timeout=1)
            if cancelled or self.isInterruptionRequested():
                raise InterruptedError('Benchmark cancelled immediately. Completed runs remain saved.')
            if error or self.process.returncode != 0 or folders is None:
                raise RuntimeError(error or f'Benchmark process exited with code {self.process.returncode}.')
            self.succeeded.emit(folders)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
        finally:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=2)
                self.process.stdout.close()
            payload_path.unlink(missing_ok=True)
            # Only the unique staging folder owned by this job may be removed.
            if self.staging.is_dir() and self.staging.resolve() == self.directory / '.in_progress' / self.job_id:
                shutil.rmtree(self.staging)
