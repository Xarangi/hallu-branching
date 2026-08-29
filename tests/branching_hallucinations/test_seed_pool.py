from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.config import load_config
from branching_hallucinations.schemas import (
    CandidateClaim,
    GeneratedSeed,
    ParseStatus,
    VerificationResult,
    VerificationStatus,
    VerifiedSeed,
)
from branching_hallucinations.seed_pool import (
    SeedPool,
    claim_cache_key,
    verification_stack_fingerprint,
)
from branching_hallucinations.storage import RunStore, append_jsonl, write_json


class SeedPoolTests(unittest.TestCase):
    def test_fingerprint_stable_for_same_config(self) -> None:
        config = load_config("configs/branching_pilot.toml")
        a = verification_stack_fingerprint(config)
        b = verification_stack_fingerprint(config)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_import_reuses_verification_without_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool_root = Path(tmp) / "pools"
            run_a = Path(tmp) / "run-a"
            run_b = Path(tmp) / "run-b"
            config = load_config("configs/branching_pilot.toml")
            pool = SeedPool.for_config(config, pool_root=pool_root)

            store_a = RunStore(run_a)
            store_a.ensure()
            store_a.write_manifest(config)
            seed = GeneratedSeed(
                seed_id="seed1",
                question_id=1,
                domain="research",
                question="Q?",
                seed_answer="A.",
                answer_model="test",
            )
            store_a.append_generated(seed)
            claim = CandidateClaim(claim_id="seed1/c0", seed_id="seed1", text="False claim.")
            store_a.append_claim(claim)
            supported = VerificationResult(
                claim_id="seed1/c0",
                status=VerificationStatus.SUPPORTED,
                reason="ok",
                parse_status=ParseStatus.OK,
            )
            store_a.append_verification(supported)
            append_jsonl(store_a.exhausted_seeds_path, {"seed_id": "seed1"})
            pool.publish_run(store_a)

            store_b = RunStore(run_b)
            store_b.ensure()
            store_b.write_manifest(config)
            store_b.append_generated(seed)
            store_b.append_claim(claim)
            stats = pool.import_into_run(store_b)
            self.assertEqual(stats.verifications, 1)
            self.assertEqual(stats.exhausted, 1)
            self.assertEqual(len(store_b.verifications()), 1)
            self.assertEqual(store_b.verifications()[0].status, VerificationStatus.SUPPORTED)
            self.assertIn("seed1", store_b.completed_ids(store_b.exhausted_seeds_path, "seed_id"))

    def test_import_verified_false_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool_root = Path(tmp) / "pools"
            config = load_config("configs/branching_pilot.toml")
            pool = SeedPool.for_config(config, pool_root=pool_root)
            run_a = Path(tmp) / "run-a"
            store_a = RunStore(run_a)
            store_a.ensure()
            verified = VerifiedSeed(
                seed_id="seed2",
                question_id=2,
                domain="research",
                question="Q2?",
                seed_answer="A2.",
                tracked_claim="Bad.",
                tracked_claim_id="seed2/c0",
            )
            store_a.append_verified(verified)
            pool.publish_run(store_a)

            run_b = Path(tmp) / "run-b"
            store_b = RunStore(run_b)
            store_b.ensure()
            stats = pool.import_into_run(store_b)
            self.assertEqual(stats.verified, 1)
            self.assertEqual(len(store_b.verified_seeds()), 1)


if __name__ == "__main__":
    unittest.main()
