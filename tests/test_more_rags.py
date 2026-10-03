import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from windrag.data import Document
from windrag.engine import Engine, Settings, METHODS

DOCS = [Document('a', 'A', 'gearbox failure', 'events.csv', 'event', 'A', 'failure'),
        Document('b', 'B', 'gearbox inspection', 'events.csv', 'event', 'A', 'inspection'),
        Document('c', 'C', 'blade temperature', 'events.csv', 'event', 'A', 'temperature')]


class AddedRagTests(unittest.TestCase):
    def test_lexical_methods_do_not_embed_question(self):
        engine = Engine(DOCS, Settings(top_k=1))
        with patch.object(engine, 'query_vector', side_effect=AssertionError('Unexpected embedding')):
            rows = engine.compare('gearbox failure', order=['BM25', 'TF-IDF'])
        for row in rows:
            self.assertEqual(row['sources'][0]['id'], 'a')
            self.assertEqual(row['query_embedding_ms'], 0)

    def test_mmr_diversifies_dense_candidates(self):
        engine = Engine(DOCS, Settings(top_k=2, candidates=3))
        engine.vectors = np.array([[1., 0.], [0.99995, 0.01], [0., 1.]])
        vector = np.array([[2**-0.5, 2**-0.5]])
        dense = engine.retrieve('test', 'Dense', vector)
        diverse = engine.retrieve('test', 'Dense + MMR', vector)
        self.assertEqual([doc.id for doc, _ in dense], ['b', 'a'])
        self.assertEqual([doc.id for doc, _ in diverse], ['b', 'c'])

    def test_all_algorithms_reuse_saved_index(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Engine(DOCS, Settings(top_k=2), cache_dir=directory)
            with patch('windrag.engine.TfidfVectorizer.fit_transform', side_effect=AssertionError('Rebuilt index')):
                loaded = Engine(DOCS, Settings(top_k=2, answer_model='another-llm'), cache_dir=directory)
                rows = loaded.compare('gearbox failure')
            self.assertTrue(loaded.cache_hit)
            self.assertEqual(first.cache_path, loaded.cache_path)
            self.assertEqual({row['method'] for row in rows}, set(METHODS))
            self.assertTrue(all(len(row['sources']) == 2 for row in rows))
