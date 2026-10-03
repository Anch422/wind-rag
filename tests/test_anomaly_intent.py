import unittest
from pathlib import Path
from windrag.data import load_corpus
from windrag.engine import Engine, Settings, METHODS
from windrag.query_intent import analyze_query, matches_document
from windrag.data import Document


class AnomalyIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs, _, _, _ = load_corpus(Path(__file__).resolve().parents[1] / 'datasets' / 'CARE_To_Compare')
        cls.engine = Engine(docs, Settings(mode='offline', top_k=3))

    def test_every_algorithm_returns_farm_c_anomalies(self):
        query = "What's the problem with wind farm C?"
        for result in self.engine.compare(query):
            self.assertEqual(result['evidence_scope'], 'anomaly_events')
            self.assertEqual(len(result['sources']), 3)
            for source in result['sources']:
                self.assertTrue(source['id'].startswith('C:event:'))
                self.assertIn('Recorded label: anomaly.', source['text'])
            self.assertIn('historical anomaly records', result['answer'])

    def test_intent_and_farm_boundaries(self):
        for query in ('What is wrong with Wind Farm C?', 'What issues are recorded at Farm C?', 'List anomalies at Wind Farm C'):
            self.assertEqual(self.engine.evidence_scope(query), 'anomaly_events')
        self.assertEqual(self.engine.evidence_scope('How is Wind Farm C?'), 'events')
        self.assertEqual(self.engine.evidence_scope('How to repair Wind Farm C?'), 'technician_reports')
        query = 'What does sensor_0 measure at Wind Farm C?'
        self.assertEqual(self.engine.evidence_scope(query), 'all')
        self.assertTrue(all(self.engine.docs[i].farm == 'Wind Farm C' for i in self.engine.eligible_documents(query)))
        answer, _ = self.engine.answer('What problems are at Wind Farm Z?', [])
        self.assertIn('No matching recorded anomaly events', answer)

    def test_paraphrases_and_explicit_filters(self):
        cases = [
            ('What’s wrong with WIND FARM C?', 'anomaly_events', {'Wind Farm C'}),
            ('Show breakdowns in farm-c', 'anomaly_events', {'Wind Farm C'}),
            ('Compare problems at wind farms A and C', 'anomaly_events', {'Wind Farm A', 'Wind Farm C'}),
            ('How are wind farms B/C operating?', 'events', {'Wind Farm B', 'Wind Farm C'}),
            ('Are there normal events in farm C?', 'normal_events', {'Wind Farm C'}),
            ('Compare normal and anomalous events at farm C', 'events', {'Wind Farm C'}),
            ('What happened in event 55 at farm C?', 'events', {'Wind Farm C'}),
            ('What problems occurred for turbine 50 at farm C?', 'events', {'Wind Farm C'}),
            ('What is the unit of wind speed at farm C?', 'all', {'Wind Farm C'}),
            ('What causes the converter failure at farm C?', 'technician_reports', {'Wind Farm C'}),
            ('Why does this sensor measure temperature?', 'all', set()),
        ]
        for query, scope, farms in cases:
            with self.subTest(query=query):
                self.assertEqual(self.engine.evidence_scope(query), scope)
                selected = [self.engine.docs[i] for i in self.engine.eligible_documents(query)]
                if scope != 'technician_reports':
                    self.assertTrue(selected)
                if farms:
                    self.assertTrue(all(doc.farm in farms for doc in selected))
        self.assertEqual([self.engine.docs[i].id for i in self.engine.eligible_documents('event 55 at farm C')], ['C:event:55'])
        selected = [self.engine.docs[i] for i in self.engine.eligible_documents('turbine 50 at farm C')]
        self.assertTrue(all('Turbine asset 50.' in doc.text for doc in selected))

    def test_normal_negation_and_missing_metadata(self):
        for query in ('List normal events at farm C', 'Events without anomalies at farm C', 'Records with no problems at farm C'):
            selected = [self.engine.docs[i] for i in self.engine.eligible_documents(query)]
            self.assertTrue(selected)
            self.assertTrue(all('Recorded label: normal.' in doc.text for doc in selected))
        unknown = Document('unknown', 'Failure', 'Converter failure documented', 'events.csv', 'event', 'Wind Farm C')
        self.assertTrue(matches_document(unknown, analyze_query('What failed at farm C?')))
        self.assertFalse(matches_document(unknown, analyze_query('What failed at farm A?')))

    def test_all_farms_overview_covers_each_farm_in_every_algorithm(self):
        for query in ('Check the condition of all my windfarms', 'How are all wind farms?', 'List problems across farms A and B and C'):
            for method in METHODS:
                with self.subTest(query=query, method=method):
                    sources = self.engine.retrieve(query, method)
                    self.assertEqual({doc.farm for doc, _ in sources}, {'Wind Farm A', 'Wind Farm B', 'Wind Farm C'})
                    self.assertEqual(len(sources), 3)

    def test_overview_insufficient_top_k_is_explicit(self):
        previous = self.engine.settings.top_k
        try:
            self.engine.settings.top_k = 2
            with self.assertRaisesRegex(ValueError, 'at least 3'):
                self.engine.retrieve('Check the condition of all my windfarms', 'BM25')
            sources = self.engine.retrieve('Compare problems at farms A and C', 'BM25')
            self.assertEqual({doc.farm for doc, _ in sources}, {'Wind Farm A', 'Wind Farm C'})
        finally:
            self.engine.settings.top_k = previous
