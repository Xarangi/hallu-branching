from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.archive import archive_run
from branching_hallucinations.compare import compare_runs, compare_and_write
from branching_hallucinations.storage import write_json


def _minimal_summary() -> dict:
    return {
        "n_verified_seeds": 2,
        "domain_composition": {"research": 2},
        "n_nodes": 4,
        "n_judgments_ok": 4,
        "n_judgments_failed": 0,
        "t1": {
            "by_action": {
                "D": {
                    "DROP": {"count": 0},
                    "RETRACT": {"count": 0},
                    "REPEAT": {"count": 0},
                    "DEPEND": {"count": 2},
                    "n": 2,
                    "active": [1.0, 0.5, 1.0],
                },
                "N": {
                    "DROP": {"count": 1},
                    "RETRACT": {"count": 0},
                    "REPEAT": {"count": 0},
                    "DEPEND": {"count": 1},
                    "n": 2,
                    "active": [0.5, 0.0, 1.0],
                },
                "V": {
                    "DROP": {"count": 0},
                    "RETRACT": {"count": 0},
                    "REPEAT": {"count": 2},
                    "DEPEND": {"count": 0},
                    "n": 2,
                    "active": [1.0, 0.5, 1.0],
                },
            },
            "paired_mcnemar_active": {},
            "n_seeds": 2,
        },
        "t2": {"seed_cluster_bootstrap_active": {}},
        "action_compliance": {"N": {"compliance": 0.5}},
        "verification_attrition": {"n_candidates": 10, "by_status": {"VERIFIED_FALSE": 2}},
    }


class ArchiveCompareTests(unittest.TestCase):
    def test_archive_run_copies_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "demo"
            dest = Path(tmp) / "archive"
            (run / "reports").mkdir(parents=True)
            (run / "analysis").mkdir(parents=True)
            (run / "seeds").mkdir(parents=True)
            write_json(run / "manifest.json", {"experiment_id": "demo"})
            write_json(run / "reports" / "summary.json", _minimal_summary())
            write_json(run / "analysis" / "summary.json", _minimal_summary())
            (run / "seeds" / "verified.jsonl").write_text("{}\n", encoding="utf-8")
            meta = archive_run(run, dest, label="demo")
            self.assertIn("reports/summary.json", meta["copied"])
            self.assertTrue((dest / "archive_meta.json").exists())
            self.assertTrue((dest / "seeds" / "verified.jsonl").exists())

    def test_compare_runs_from_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            for path in (left, right):
                (path / "reports").mkdir(parents=True)
                summary = _minimal_summary()
                write_json(path / "reports" / "summary.json", summary)
            comparison = compare_runs({"a": left, "b": right})
            self.assertEqual(comparison["labels"], ["a", "b"])
            self.assertEqual(comparison["t1_active"]["a"]["D"], 1.0)
            self.assertEqual(comparison["t1_active"]["b"]["N"], 0.5)

    def test_compare_and_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            out = Path(tmp) / "out"
            (run / "reports").mkdir(parents=True)
            write_json(run / "reports" / "summary.json", _minimal_summary())
            compare_and_write({"solo": run}, out)
            self.assertTrue((out / "comparison.json").exists())
            self.assertTrue((out / "comparison.md").exists())
            payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
            self.assertIn("solo", payload["overview"])


if __name__ == "__main__":
    unittest.main()
