from __future__ import annotations

import argparse
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.cli import cmd_verify_seeds
from branching_hallucinations.schemas import (
    CandidateClaim,
    GeneratedSeed,
    ParseStatus,
    VerificationResult,
    VerificationStatus,
)
from branching_hallucinations.storage import RunStore


class _DummySamplers:
    search_planner = object()
    grounded_verifier = object()


def _seed(index: int) -> GeneratedSeed:
    return GeneratedSeed(
        seed_id=f"seed{index}",
        question_id=index,
        domain="research",
        question=f"question {index}",
        seed_answer=f"answer {index}",
        answer_model="mock",
    )


def _claim(index: int, claim_index: int = 0) -> CandidateClaim:
    seed_id = f"seed{index}"
    return CandidateClaim(
        claim_id=f"{seed_id}/c{claim_index}",
        seed_id=seed_id,
        text=f"claim {index}.{claim_index}",
    )


def _args(run: str, *, concurrency: int, max_verified: int) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(ROOT / "configs" / "branching_pilot.toml"),
        run=run,
        concurrency=concurrency,
        max_verified=max_verified,
    )


class VerifyConcurrencyTests(unittest.TestCase):
    def test_freeze_order_follows_seed_file_not_finish_order(self):
        async def fake_verify_claim(**kwargs):
            claim_id = kwargs["claim_id"]
            seed_id = claim_id.rsplit("/c", 1)[0]
            if seed_id == "seed0":
                await asyncio.sleep(0.04)
            return VerificationResult(
                claim_id=claim_id,
                status=VerificationStatus.VERIFIED_FALSE,
                reason="mock contradiction",
                parse_status=ParseStatus.OK,
            )

        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                store = RunStore(tmp)
                store.ensure()
                for index in range(3):
                    store.append_generated(_seed(index))
                    store.append_claim(_claim(index))
                with patch(
                    "branching_hallucinations.cli.ExperimentSamplers.from_config",
                    return_value=_DummySamplers(),
                ), patch(
                    "branching_hallucinations.grounding.verify_claim",
                    side_effect=fake_verify_claim,
                ):
                    await cmd_verify_seeds(_args(tmp, concurrency=3, max_verified=2))
                return [seed.seed_id for seed in store.verified_seeds()]

        self.assertEqual(asyncio.run(_run()), ["seed0", "seed1"])

    def test_does_not_start_extra_seeds_beyond_remaining_need(self):
        calls: list[str] = []

        async def fake_verify_claim(**kwargs):
            claim_id = kwargs["claim_id"]
            calls.append(claim_id)
            return VerificationResult(
                claim_id=claim_id,
                status=VerificationStatus.VERIFIED_FALSE,
                reason="mock contradiction",
                parse_status=ParseStatus.OK,
            )

        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                store = RunStore(tmp)
                store.ensure()
                for index in range(4):
                    store.append_generated(_seed(index))
                    store.append_claim(_claim(index))
                with patch(
                    "branching_hallucinations.cli.ExperimentSamplers.from_config",
                    return_value=_DummySamplers(),
                ), patch(
                    "branching_hallucinations.grounding.verify_claim",
                    side_effect=fake_verify_claim,
                ):
                    await cmd_verify_seeds(_args(tmp, concurrency=8, max_verified=1))
                return [seed.seed_id for seed in store.verified_seeds()]

        frozen = asyncio.run(_run())
        self.assertEqual(frozen, ["seed0"])
        self.assertEqual(calls, ["seed0/c0"])

    def test_claims_inside_a_seed_stay_serial_and_stop_at_first_false(self):
        calls: list[str] = []

        async def fake_verify_claim(**kwargs):
            claim_id = kwargs["claim_id"]
            calls.append(claim_id)
            status = (
                VerificationStatus.SUPPORTED
                if claim_id.endswith("/c0")
                else VerificationStatus.VERIFIED_FALSE
            )
            return VerificationResult(
                claim_id=claim_id,
                status=status,
                reason="mock",
                parse_status=ParseStatus.OK,
            )

        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                store = RunStore(tmp)
                store.ensure()
                store.append_generated(_seed(0))
                store.append_claim(_claim(0, 0))
                store.append_claim(_claim(0, 1))
                store.append_claim(_claim(0, 2))
                with patch(
                    "branching_hallucinations.cli.ExperimentSamplers.from_config",
                    return_value=_DummySamplers(),
                ), patch(
                    "branching_hallucinations.grounding.verify_claim",
                    side_effect=fake_verify_claim,
                ):
                    await cmd_verify_seeds(_args(tmp, concurrency=4, max_verified=1))
                frozen = store.verified_seeds()
                return frozen[0].tracked_claim_id if frozen else None

        self.assertEqual(asyncio.run(_run()), "seed0/c1")
        self.assertEqual(calls, ["seed0/c0", "seed0/c1"])


if __name__ == "__main__":
    unittest.main()
