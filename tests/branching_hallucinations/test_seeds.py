from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.config import DatasetConfig
from branching_hallucinations.seeds import generate_seeds
from branching_hallucinations.storage import RunStore
from libs.types import SamplerResponse


class DelayedAnswer:
    """Later questions finish first so commit order is not finish order."""

    async def __call__(self, messages):
        question = messages[0]["content"]
        index = int(question.split()[-1])
        await asyncio.sleep(0.03 * (5 - index))
        return SamplerResponse(
            response_text=f"answer-{index}",
            actual_queried_message_list=messages,
            response_metadata={"backend": "mock"},
            token_usage={},
        )


class EmptyThenAnswer:
    async def __call__(self, messages):
        question = messages[0]["content"]
        index = int(question.split()[-1])
        text = "" if index == 0 else f"answer-{index}"
        return SamplerResponse(
            response_text=text,
            actual_queried_message_list=messages,
            response_metadata={"backend": "mock"},
            token_usage={},
        )


def _questions(path: Path, n: int) -> DatasetConfig:
    path.write_text(
        "".join(
            json.dumps({"id": i, "domain": "research", "question": f"question {i}"}) + "\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )
    return DatasetConfig(name="jsonl", path=path, domain_field="domain", id_field="id")


class GenerateSeedsConcurrencyTests(unittest.TestCase):
    def test_writes_n_seeds_in_question_order_not_finish_order(self):
        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = RunStore(root / "run")
                store.ensure()
                dataset = _questions(root / "q.jsonl", 5)
                created = await generate_seeds(
                    store,
                    answer_model=DelayedAnswer(),
                    domains=("research",),
                    n_seeds=3,
                    dataset=dataset,
                    concurrency=4,
                )
                return [seed.seed_id for seed in created], [
                    seed.seed_id for seed in store.generated_seeds()
                ]

        created_ids, file_ids = asyncio.run(_run())
        self.assertEqual(created_ids, ["seed0", "seed1", "seed2"])
        self.assertEqual(file_ids, ["seed0", "seed1", "seed2"])

    def test_skips_empty_answers_and_still_fills_n_seeds(self):
        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = RunStore(root / "run")
                store.ensure()
                dataset = _questions(root / "q.jsonl", 4)
                created = await generate_seeds(
                    store,
                    answer_model=EmptyThenAnswer(),
                    domains=("research",),
                    n_seeds=2,
                    dataset=dataset,
                    concurrency=3,
                )
                return [seed.seed_id for seed in created]

        self.assertEqual(asyncio.run(_run()), ["seed1", "seed2"])


if __name__ == "__main__":
    unittest.main()
