"""Real HTTP round trips to a deterministic test server, not a real model."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import re
import threading
import unittest
from windrag.data import Document
from windrag.engine import Engine, Settings, METHODS
from windrag.evaluation import benchmark


class Handler(BaseHTTPRequestHandler):
    requests_seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests_seen.append((self.path, payload))
        if self.path == "/api/embed":
            embeddings = []
            for text in payload["input"]:
                vector = [0.0] * 64
                for token in re.findall(r"\w+", text.lower()):
                    index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 64
                    vector[index] += 1
                embeddings.append(vector)
            response = {"embeddings": embeddings}
        elif self.path == "/api/chat" and "format" in payload:
            sources = json.loads(payload["messages"][-1]["content"])["sources"]
            content = json.dumps({"scores": [10 if "gearbox" in s["text"] else 1 for s in sources]})
            response = {"message": {"content": content}, "prompt_eval_count": 80, "eval_count": 15}
        else:
            context = payload["messages"][-1]["content"].split("Sources:\n", 1)[1]
            identifier = re.search(r"\[([^\]]+)\]", context).group(1)
            response = {"message": {"content": f"Gearbox failure [{identifier}]"},
                        "prompt_eval_count": 80, "eval_count": 8}
        content = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class HttpIntegrationTests(unittest.TestCase):
    def test_all_methods_embedding_reranking_generation_and_metrics(self):
        Handler.requests_seen = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = Settings(mode="ollama", host=f"http://127.0.0.1:{server.server_port}", top_k=1, candidates=2)
            docs = [Document("A:event:1", "Gearbox", "gearbox failure turbine", "events.csv", "event", "A", "Gearbox failure"),
                    Document("A:event:2", "Blade", "blade inspection turbine", "events.csv", "event", "A", "Blade inspection")]
            engine = Engine(docs, settings)
            questions = [{"id": "q1", "question": "gearbox failure", "reference_answer": "Gearbox failure", "relevant_ids": ["A:event:1"]}]
            rows, summaries = benchmark(engine, questions)
            self.assertEqual({r["method"] for r in rows}, set(METHODS))
            self.assertTrue(all(r["recall_at_k"] == 1 and r["exact_match"] == 1 for r in rows))
            self.assertEqual(len(summaries), len(METHODS))
            calls = Handler.requests_seen
            self.assertEqual(sum(path == "/api/embed" for path, _ in calls), 2)
            self.assertEqual(sum(path == "/api/chat" for path, _ in calls), len(METHODS) + 1)
            generation = [payload for path, payload in calls if path == "/api/chat" and "format" not in payload]
            self.assertEqual(len(generation), len(METHODS))
            self.assertTrue(all(p["options"]["temperature"] == 0 and p["options"]["seed"] == 42 for p in generation))
            Handler.requests_seen = []
            interactive = engine.compare("gearbox failure", reuse_answers=True)
            self.assertEqual(sum(path == "/api/embed" for path, _ in Handler.requests_seen), 1)
            self.assertEqual(sum(path == "/api/chat" for path, _ in Handler.requests_seen), 2)
            self.assertEqual(sum(r["answer_reused"] for r in interactive), len(METHODS) - 1)
            Handler.requests_seen = []
            lexical = engine.compare("gearbox failure", order=["BM25", "TF-IDF"])
            self.assertFalse(any(path == "/api/embed" for path, _ in Handler.requests_seen))
            self.assertTrue(all(row["query_embedding_ms"] == 0 for row in lexical))
            import tempfile
            from unittest.mock import patch
            from windrag.experiments import sequential_runs, scan_runs, aggregate
            with tempfile.TemporaryDirectory() as directory, patch("windrag.experiments.export_graphs"):
                folders = sequential_runs(engine, "test-corpus", questions,
                                          ["test-model-one", "test-model-two"], list(METHODS), directory)
                self.assertEqual(len(folders), 2 * len(METHODS))
                entries, errors = scan_runs(directory)
                self.assertFalse(errors)
                self.assertEqual({e["model"] for e in entries}, {"test-model-one", "test-model-two"})
                self.assertEqual(len(aggregate(entries)[0]), 2 * len(METHODS))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

