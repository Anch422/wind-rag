import json
import tempfile
import unittest
from pathlib import Path
from windrag.data import load_corpus
from windrag.engine import Engine, Settings, METHODS
from windrag.experiments import validation_set, sequential_runs, scan_runs, aggregate, export_comparison

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def test_sequential_export_scan_aggregate_and_comparison(self):
        docs, events, features, digest = load_corpus(ROOT / "datasets/CARE_To_Compare")
        sensor = next(d for d in docs if d.id == "A:sensor:sensor_0")
        self.assertIn("°C", sensor.text)
        self.assertNotIn("�", sensor.text)
        questions = validation_set(events, features, size=6)
        self.assertEqual(len(questions), 6)
        self.assertEqual(len({q["relevant_ids"][0][0] for q in questions}), 3)
        self.assertEqual(questions, validation_set(events, features, size=6))
        engine = Engine(docs, Settings(mode="offline", top_k=3, candidates=10))
        with tempfile.TemporaryDirectory() as directory:
            folders = sequential_runs(engine, digest, questions, ["offline-demo"], list(METHODS), directory)
            self.assertEqual(len(folders), len(METHODS))
            for folder in map(Path, folders):
                self.assertIn("_offline-demo_", folder.name)
                for name in ("run.json", "results.csv", "summary.csv", "validation.json", "validation.csv", "corpus.json", "report.md"):
                    self.assertTrue((folder / name).exists(), name)
                self.assertEqual(len(list((folder / "graphs").glob("*.png"))), 13)
                run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(len(run["summary"]), 1)
            entries, errors = scan_runs(directory)
            self.assertFalse(errors)
            self.assertEqual(len(entries), len(METHODS))
            summary, rows = aggregate(entries)
            self.assertEqual(len(summary), len(METHODS))
            self.assertEqual(len(rows), 6 * len(METHODS))
            comparison = export_comparison(directory, entries, "my comparison")
            self.assertEqual(comparison.parent.name, "comparisons")
            self.assertTrue((comparison / "comparison.json").exists())
            self.assertTrue((comparison / "validation.json").exists())
            self.assertEqual(len(list((comparison / "source_runs").glob("*.json"))), len(METHODS))
            self.assertEqual(len(scan_runs(directory)[0]), len(METHODS))
            entries[0]["run"]["benchmark_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "different validation"):
                aggregate(entries)

    def test_cancel_preserves_completed_runs(self):
        docs, events, features, digest = load_corpus(ROOT / "datasets/CARE_To_Compare")
        engine = Engine(docs, Settings(mode="offline"))
        completed = []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                sequential_runs(engine, digest, validation_set(events, features, 1), ["offline-demo"], list(METHODS), directory,
                                cancelled=lambda: bool(completed), completed=completed.append)
            self.assertEqual(len(completed), 1)
            self.assertEqual(len(scan_runs(directory)[0]), 1)

    def test_failed_run_is_saved_and_next_algorithm_continues(self):
        from unittest.mock import patch
        docs, events, features, digest = load_corpus(ROOT / "datasets/CARE_To_Compare")
        engine = Engine(docs, Settings(mode="offline"))
        from windrag.experiments import benchmark as real_benchmark
        def flaky(*args, **kwargs):
            if kwargs["methods"] == ["Dense"]:
                raise RuntimeError("simulated model failure")
            return real_benchmark(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch("windrag.experiments.benchmark", side_effect=flaky), patch("windrag.experiments.export_graphs"):
            folders = sequential_runs(engine, digest, validation_set(events, features, 1), ["offline-demo"], ["Dense", "Hybrid"], directory)
            self.assertEqual(len(folders), 2)
            failure = json.loads((Path(folders[0]) / "run.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(len(scan_runs(directory)[0]), 1)
