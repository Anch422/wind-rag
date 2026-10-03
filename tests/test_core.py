import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from windrag.data import Document, load_corpus, load_questions, starter_questions
from windrag.engine import Engine, Settings, Ollama, METHODS
from windrag.evaluation import score_result, benchmark, save_run

ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_known_retrieval_ranks_and_token_f1(self):
        result = {"answer": "gearbox failure [b]", "sources": [{"id": "x"}, {"id": "b"}, {"id": "a"}]}
        scores = score_result(result, {"relevant_ids": ["a", "b"], "reference_answer": "gearbox failure"})
        self.assertEqual(scores["recall_at_k"], 1)
        self.assertAlmostEqual(scores["precision_at_k"], 2 / 3)
        self.assertEqual(scores["mrr"], 0.5)
        self.assertAlmostEqual(scores["ndcg_at_k"], (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
        self.assertEqual(scores["answer_f1"], 1)
        self.assertEqual(scores["citation_recall"], 0.5)

    def test_empty_ground_truth_is_not_recall_success(self):
        scores = score_result({"answer": "Insufficient evidence.", "sources": []},
                              {"relevant_ids": [], "reference_answer": "Insufficient evidence."})
        self.assertIsNone(scores["recall_at_k"])
        self.assertIsNone(scores["mrr"])
        self.assertEqual(scores["exact_match"], 1)

    def test_invalid_citation(self):
        scores = score_result({"answer": "failure [invented]", "sources": [{"id": "a"}]},
                              {"relevant_ids": ["a"], "reference_answer": "failure"})
        self.assertEqual(scores["citation_validity"], 0)


class CorpusTests(unittest.TestCase):
    def test_actual_dataset_and_all_methods(self):
        docs, events, features, digest = load_corpus(ROOT / "datasets" / "CARE_To_Compare")
        self.assertEqual(len(events), 95)
        self.assertEqual(sum(e["label"] == "anomaly" for e in events), 45)
        self.assertEqual(len(docs), 450)
        engine = Engine(docs, Settings(mode="offline"))
        questions = starter_questions(events, features)[:4]
        rows, summary = benchmark(engine, questions)
        self.assertEqual(len(rows), len(questions) * len(METHODS))
        self.assertEqual({r["method"] for r in rows}, set(METHODS))
        for row in rows:
            self.assertGreater(len(row["sources"]), 0)
            self.assertLessEqual(len(row["sources"]), engine.settings.top_k)
            self.assertTrue(0 <= row["answer_f1"] <= 1)
        with tempfile.TemporaryDirectory() as directory:
            folder, run = save_run(directory, engine, digest, questions, rows, summary)
            loaded = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["corpus_sha256"], digest)
            self.assertTrue((folder / "results.csv").exists())
            self.assertTrue((folder / "corpus.json").exists())

    def test_bad_benchmark_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([{"id": "q1", "question": "test", "reference_answer": "test", "relevant_ids": ["missing"]}]))
            with self.assertRaises(ValueError):
                load_questions(path, ["a"])


class OllamaContractTests(unittest.TestCase):
    def test_missing_model_error_preserves_server_message(self):
        from unittest.mock import Mock
        response = Mock(ok=False, status_code=404, text="model not found")
        response.json.return_value = {"error": "model 'qwen2.5:3b' not found"}
        with patch("windrag.engine.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "qwen2.5:3b.*not found"):
                Ollama(Settings()).chat("system", "question")

    def test_model_validation_accepts_latest_alias_and_rejects_missing_chat_model(self):
        from unittest.mock import Mock
        response = Mock()
        response.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:7b"}]}
        with patch("windrag.engine.requests.get", return_value=response):
            Ollama(Settings(answer_model="qwen2.5:7b")).validate_models()
            with self.assertRaisesRegex(ValueError, "answer / reranker model.*not installed"):
                Ollama(Settings()).validate_models()

    def test_embedding_normalization_and_shape(self):
        client = Ollama(Settings(mode="ollama"))
        with patch.object(client, "post", return_value={"embeddings": [[3, 4]]}):
            self.assertAlmostEqual(float(np.linalg.norm(client.embed(["test"])[0])), 1)
        with patch.object(client, "post", return_value={"embeddings": [[0, 0]]}):
            with self.assertRaises(ValueError):
                client.embed(["test"])

    def test_reranker_does_not_silently_accept_missing_scores(self):
        docs = [Document("a", "A", "gearbox failure", "test.csv", "event", "A", "failure"),
                Document("b", "B", "blade inspection", "test.csv", "event", "A", "inspection")]
        engine = Engine(docs, Settings(mode="offline", top_k=1, candidates=2))
        engine.settings.mode = "ollama"
        with patch.object(engine.client, "embed", return_value=engine.vectors[:1]), patch.object(engine.client, "chat", return_value=(json.dumps({"scores": [7]}), {})):
            with self.assertRaises(ValueError):
                engine.retrieve("gearbox", "Hybrid + reranking")


if __name__ == "__main__":
    unittest.main()

