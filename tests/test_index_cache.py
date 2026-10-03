import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch
import numpy as np
from windrag.data import Document
from windrag.engine import Engine, Settings, Ollama

DOCS = [Document("a", "A", "gearbox failure", "events.csv", "event", "A", "failure"),
        Document("b", "B", "blade inspection", "events.csv", "event", "A", "inspection")]


class IndexCacheTests(unittest.TestCase):
    def test_restart_reuses_index_and_new_runtime_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Engine(DOCS, Settings(mode="offline"), cache_dir=folder)
            changed = Settings(mode="offline", answer_model="another-model", top_k=1, candidates=6, seed=7, temperature=0.5)
            with patch("windrag.engine.TfidfVectorizer.fit_transform", side_effect=AssertionError("Index was rebuilt")):
                second = Engine(DOCS, changed, cache_dir=folder)
            self.assertTrue(second.cache_hit)
            np.testing.assert_array_equal(first.vectors, second.vectors)
            self.assertEqual(len(second.retrieve("gearbox", "Dense")), 1)
            before = second.vectors
            second.apply_settings(replace(changed, top_k=2, answer_model="third-model"))
            self.assertIs(second.vectors, before)
            self.assertEqual(second.client.settings.answer_model, "third-model")
            self.assertEqual(len(second.retrieve("gearbox", "Dense")), 2)

    def test_corpus_change_creates_new_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Engine(DOCS, Settings(mode="offline"), cache_dir=folder)
            changed_docs = [replace(DOCS[0], text="transformer overheating"), DOCS[1]]
            second = Engine(changed_docs, Settings(mode="offline"), cache_dir=folder)
            self.assertFalse(second.cache_hit)
            self.assertNotEqual(first.cache_path, second.cache_path)

    def test_neural_cache_uses_embedding_digest_not_answer_model(self):
        response = Mock()
        response.json.return_value = {"models": [{"name": "embed:latest", "digest": "version-one"}]}
        settings = Settings(mode="ollama", embedding_model="embed", answer_model="chat-one")
        with tempfile.TemporaryDirectory() as folder, patch("windrag.engine.requests.get", return_value=response), patch.object(Ollama, "embed", return_value=np.eye(2)) as embed:
            first = Engine(DOCS, settings, cache_dir=folder)
            second = Engine(DOCS, replace(settings, embedding_model="embed:latest", answer_model="chat-two"), cache_dir=folder)
            self.assertTrue(second.cache_hit)
            self.assertEqual(embed.call_count, 1)
            second.apply_settings(replace(settings, answer_model="chat-three", seed=2))
            with self.assertRaisesRegex(ValueError, "Embedding model"):
                second.apply_settings(replace(settings, embedding_model="different"))
            response.json.return_value["models"][0]["digest"] = "version-two"
            third = Engine(DOCS, settings, cache_dir=folder)
            self.assertFalse(third.cache_hit)
            self.assertNotEqual(first.cache_path, third.cache_path)
            self.assertEqual(embed.call_count, 2)

    def test_corrupt_cache_is_rebuilt(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            first = Engine(DOCS, Settings(mode="offline"), cache_dir=folder)
            Path(first.cache_path).write_bytes(b"broken cache")
            second = Engine(DOCS, Settings(mode="offline"), cache_dir=folder)
            self.assertFalse(second.cache_hit)
            self.assertEqual(len(second.retrieve("gearbox", "Dense")), 2)
