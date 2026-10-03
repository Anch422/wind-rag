import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pypdf import PdfReader, PdfWriter
from windrag.data import load_corpus, load_report_documents
from windrag.engine import Engine, Settings, METHODS
from windrag.experiments import validation_set

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'datasets' / 'technician_reports'


class TechnicianReportTests(unittest.TestCase):
    def test_complete_pdf_dataset_covers_every_anomaly(self):
        _, events, _, _ = load_corpus(ROOT / 'datasets' / 'CARE_To_Compare')
        manifest = json.loads((REPORTS / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['report_count'], 45)
        self.assertEqual({row['care_event_document_id'] for row in manifest['reports']}, {e['document_id'] for e in events if e['label'] == 'anomaly'})
        readers = {path.name: PdfReader(path) for path in REPORTS.glob('*.pdf')}
        self.assertEqual(len(readers), 45)
        self.assertTrue(all(len(reader.pages) == 2 for reader in readers.values()))
        self.assertEqual(sum(len(reader.pages) for reader in readers.values()), 90)
        for row in manifest['reports']:
            for key, heading in (('inspection_page', 'Possible cause'), ('repair_page', 'Repair record')):
                text = readers[row['pdf']].pages[row[key] - 1].extract_text()
                for value in (row['report_id'], row['farm'], row['technician'], row['company'], heading):
                    self.assertIn(value, text)
                self.assertNotRegex(text.lower(), r'\b(mock|fictional|synthetic|fabricated)\b')
        self.assertEqual(manifest['data_origin'], 'generated_research_fixture')
        self.assertFalse(manifest['externally_verified'])
        self.assertGreater(len({r['service_date'] for r in manifest['reports']}), 20)
        self.assertEqual(len({r['company'] for r in manifest['reports']}), 6)
        self.assertEqual(len({r['technician'] for r in manifest['reports']}), 12)

    def test_combined_corpus_and_validation_include_page_citations(self):
        care = ROOT / 'datasets' / 'CARE_To_Compare'
        base, _, _, original_hash = load_corpus(care)
        docs, events, features, combined_hash = load_corpus(care, REPORTS)
        report_docs = [doc for doc in docs if doc.kind == 'report']
        self.assertEqual(len(base), 450)
        self.assertEqual(len(report_docs), 90)
        self.assertNotEqual(original_hash, combined_hash)
        self.assertEqual(len({doc.id for doc in docs}), len(docs))
        self.assertTrue(all('PDF page' in doc.text and Path(doc.source).is_file() for doc in report_docs))
        questions = validation_set(events, features, size=24, report_docs=docs)
        self.assertTrue(any(q['relevant_ids'][0].startswith('REPORT:') for q in questions))

    def test_all_algorithms_route_repair_questions_to_reports(self):
        docs, _, _, _ = load_corpus(ROOT / 'datasets' / 'CARE_To_Compare', REPORTS)
        engine = Engine(docs, Settings(mode='offline', top_k=3, candidates=10))
        rows = engine.compare('How should I repair a damaged generator bearing at Wind Farm A?')
        self.assertEqual(len(rows), len(METHODS))
        for row in rows:
            self.assertEqual(row['evidence_scope'], 'technician_reports')
            self.assertTrue(row['sources'])
            self.assertTrue(all(source['kind'] == 'report' and source['farm'] == 'Wind Farm A' for source in row['sources']))
            self.assertNotRegex(row['answer'].lower(), r'\b(mock|fictional|synthetic)\b')
        self.assertEqual(engine.evidence_scope('What was reported for event 40?'), 'events')
        with patch.object(engine.client, 'chat', return_value=('Report recommendation', {})) as chat:
            engine.settings.mode = 'ollama'
            engine.answer('How to repair it?', engine.retrieve('How to repair it?', 'BM25'))
            self.assertNotRegex(chat.call_args.args[0].lower(), r'\b(mock|fictional|synthetic)\b')
            self.assertIn('cite their source IDs', chat.call_args.args[0])

    def test_missing_report_evidence_does_not_invent_repair(self):
        docs, _, _, _ = load_corpus(ROOT / 'datasets' / 'CARE_To_Compare')
        engine = Engine(docs, Settings(mode='offline'))
        row = engine.compare('How to repair a generator bearing?', order=['BM25'])[0]
        self.assertEqual(row['sources'], [])
        self.assertIn('Insufficient evidence', row['answer'])

    def test_image_only_pdf_is_rejected_with_ocr_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            writer.write(str(Path(directory) / 'scan.pdf'))
            with self.assertRaisesRegex(ValueError, 'OCR'):
                load_report_documents(directory)
