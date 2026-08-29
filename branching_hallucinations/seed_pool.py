"""Cross-run cache for verified-false seeds and completed verifications.

Pools are keyed by the verification stack (answer model, dataset, extractor,
verifier, search planner, grounding, and verifier/extractor prompt hashes).
Trajectory judge is intentionally excluded — it only affects tree labels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig, REPO_ROOT, prompt_hash
from .schemas import (
    CandidateClaim,
    GeneratedSeed,
    ParseStatus,
    VerificationResult,
    VerificationStatus,
    VerifiedSeed,
)
from .storage import RunStore, append_jsonl, read_jsonl, utc_now, write_json, write_jsonl

POOL_SCHEMA = "branching_hallucinations.seed_pool.v1"
DEFAULT_POOL_ROOT = REPO_ROOT / "data" / "seed_pools"


def normalize_claim_text(text: str) -> str:
    return " ".join((text or "").split())


def claim_cache_key(seed_id: str, claim_text: str) -> str:
    payload = f"{seed_id}\n{normalize_claim_text(claim_text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verification_stack_spec(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema": POOL_SCHEMA,
        "answer_model": config.answer.to_dict(),
        "claim_extractor_model": config.claim_extractor.to_dict(),
        "grounded_verifier_model": config.grounded_verifier.to_dict(),
        "search_planner_model": config.search_planner.to_dict(),
        "dataset": config.dataset.to_dict(),
        "grounding_method": config.grounding_method,
        "max_searches": config.max_searches,
        "max_claims_per_seed": config.max_claims_per_seed,
        "prompt_hashes": {
            "claim_extraction": prompt_hash("claim_extraction"),
            "grounded_verifier": prompt_hash("grounded_verifier"),
        },
    }


def verification_stack_fingerprint(config: ExperimentConfig) -> str:
    canonical = json.dumps(verification_stack_spec(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class ImportStats:
    generated: int = 0
    candidates: int = 0
    verifications: int = 0
    verified: int = 0
    exhausted: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "generated": self.generated,
            "candidates": self.candidates,
            "verifications": self.verifications,
            "verified": self.verified,
            "exhausted": self.exhausted,
        }


@dataclass
class SeedPool:
    root: Path
    spec: dict[str, Any]
    fingerprint: str = ""
    generated: dict[str, GeneratedSeed] = field(default_factory=dict)
    candidates: dict[str, CandidateClaim] = field(default_factory=dict)
    candidates_by_cache: dict[str, CandidateClaim] = field(default_factory=dict)
    verifications: dict[str, VerificationResult] = field(default_factory=dict)
    verifications_by_cache: dict[str, VerificationResult] = field(default_factory=dict)
    verified: dict[str, VerifiedSeed] = field(default_factory=dict)
    exhausted: set[str] = field(default_factory=set)
    claim_text_by_id: dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_config(cls, config: ExperimentConfig, pool_root: Path | None = None) -> SeedPool:
        fingerprint = verification_stack_fingerprint(config)
        root = (pool_root or DEFAULT_POOL_ROOT) / fingerprint
        spec = verification_stack_spec(config)
        pool = cls(root=root, spec=spec, fingerprint=fingerprint)
        if root.exists():
            pool._load()
        return pool

    @classmethod
    def from_run(cls, run_dir: str | Path, config: ExperimentConfig | None = None) -> tuple[SeedPool, dict[str, int]]:
        store = RunStore(run_dir)
        manifest = store.load_manifest()
        if config is None:
            from .config import load_config

            config_path = manifest.get("config_file")
            config = load_config(config_path) if config_path else None
        if config is None:
            raise ValueError(f"Cannot infer config for run {run_dir}")
        pool = cls.for_config(config)
        counts = pool.merge_from_run(store, source=str(store.root.resolve()))
        return pool, counts

    def _paths(self) -> dict[str, Path]:
        return {
            "manifest": self.root / "pool_manifest.json",
            "generated": self.root / "generated.jsonl",
            "candidates": self.root / "candidates.jsonl",
            "verifications": self.root / "verifications.jsonl",
            "verified": self.root / "verified.jsonl",
            "exhausted": self.root / "exhausted_seeds.jsonl",
        }

    def _load(self) -> None:
        paths = self._paths()
        if paths["manifest"].exists():
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.spec = manifest.get("stack", self.spec)
            self.fingerprint = manifest.get("fingerprint", self.fingerprint)
        for row in read_jsonl(paths["generated"]):
            seed = GeneratedSeed.from_dict(row)
            self.generated[seed.seed_id] = seed
        for row in read_jsonl(paths["candidates"]):
            claim = CandidateClaim.from_dict(row)
            self._index_candidate(claim)
        for row in read_jsonl(paths["verifications"]):
            payload = dict(row)
            payload.pop("cache_key", None)
            payload.pop("claim_text", None)
            payload.pop("seed_id", None)
            result = VerificationResult.from_dict(payload)
            text = str(row.get("claim_text") or self.claim_text_by_id.get(result.claim_id, ""))
            seed_id = str(row.get("seed_id") or "")
            self._index_verification(result, seed_id=seed_id, claim_text=text)
        for row in read_jsonl(paths["verified"]):
            seed = VerifiedSeed.from_dict(row)
            self.verified[seed.seed_id] = seed
        for row in read_jsonl(paths["exhausted"]):
            seed_id = str(row.get("seed_id") or "")
            if seed_id:
                self.exhausted.add(seed_id)

    def _index_candidate(self, claim: CandidateClaim) -> None:
        self.candidates[claim.claim_id] = claim
        self.claim_text_by_id[claim.claim_id] = claim.text
        self.candidates_by_cache[claim_cache_key(claim.seed_id, claim.text)] = claim

    def _index_verification(
        self,
        result: VerificationResult,
        *,
        seed_id: str,
        claim_text: str,
    ) -> None:
        self.verifications[result.claim_id] = result
        if seed_id and claim_text:
            self.verifications_by_cache[claim_cache_key(seed_id, claim_text)] = result

    def _extractor_hash(self) -> str:
        return str(self.spec.get("prompt_hashes", {}).get("claim_extraction", ""))

    def _candidate_extractor_hash(self, claim: CandidateClaim) -> str:
        meta = claim.extractor_metadata or {}
        return str(meta.get("prompt_hash") or "")

    def merge_from_run(self, store: RunStore, *, source: str = "") -> dict[str, int]:
        counts = {"generated": 0, "candidates": 0, "verifications": 0, "verified": 0, "exhausted": 0}
        claim_text: dict[str, str] = {c.claim_id: c.text for c in store.candidate_claims()}
        for seed in store.generated_seeds():
            if seed.seed_id not in self.generated:
                self.generated[seed.seed_id] = seed
                counts["generated"] += 1
        extractor_hash = self._extractor_hash()
        for claim in store.candidate_claims():
            if extractor_hash and self._candidate_extractor_hash(claim) not in ("", extractor_hash):
                continue
            if claim.claim_id not in self.candidates:
                self._index_candidate(claim)
                counts["candidates"] += 1
        for result in store.verifications():
            if result.parse_status is ParseStatus.FAILED:
                continue
            text = claim_text.get(result.claim_id, "")
            seed_id = ""
            claim = self.candidates.get(result.claim_id)
            if claim is not None:
                seed_id = claim.seed_id
                text = claim.text
            cache = claim_cache_key(seed_id, text) if seed_id and text else ""
            if cache and cache in self.verifications_by_cache:
                continue
            self._index_verification(result, seed_id=seed_id, claim_text=text)
            counts["verifications"] += 1
        for seed in store.verified_seeds():
            if seed.seed_id not in self.verified:
                self.verified[seed.seed_id] = seed
                counts["verified"] += 1
            if seed.seed_id not in self.generated:
                self.generated[seed.seed_id] = GeneratedSeed(
                    seed_id=seed.seed_id,
                    question_id=seed.question_id,
                    domain=seed.domain,
                    question=seed.question,
                    seed_answer=seed.seed_answer,
                    answer_model=str((seed.answer_model_metadata or {}).get("model") or ""),
                    generation_metadata=dict(seed.answer_model_metadata or {}),
                )
                counts["generated"] += 1
        for row in read_jsonl(store.exhausted_seeds_path):
            seed_id = str(row.get("seed_id") or "")
            if seed_id and seed_id not in self.exhausted:
                self.exhausted.add(seed_id)
                counts["exhausted"] += 1
        self._recompute_exhausted_from_verifications()
        self._write(source_runs=[source] if source else [])
        return counts

    def _recompute_exhausted_from_verifications(self) -> None:
        by_seed: dict[str, list[CandidateClaim]] = {}
        for claim in self.candidates.values():
            by_seed.setdefault(claim.seed_id, []).append(claim)
        for seed_id, claims in by_seed.items():
            if seed_id in self.verified:
                continue
            if not claims:
                continue
            statuses = []
            for claim in claims:
                result = self.verifications.get(claim.claim_id) or self.verifications_by_cache.get(
                    claim_cache_key(claim.seed_id, claim.text)
                )
                if result is None or result.parse_status is ParseStatus.FAILED:
                    statuses = []
                    break
                statuses.append(result.status)
            if statuses and all(status is not VerificationStatus.VERIFIED_FALSE for status in statuses):
                self.exhausted.add(seed_id)

    def _write(self, *, source_runs: Iterable[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        paths = self._paths()
        existing_sources = []
        if paths["manifest"].exists():
            existing_sources = json.loads(paths["manifest"].read_text(encoding="utf-8")).get("source_runs", [])
        merged_sources = list(dict.fromkeys([*existing_sources, *source_runs]))
        write_json(
            paths["manifest"],
            {
                "schema": POOL_SCHEMA,
                "fingerprint": self.fingerprint,
                "stack": self.spec,
                "updated_at": utc_now(),
                "source_runs": merged_sources,
                "counts": {
                    "generated": len(self.generated),
                    "candidates": len(self.candidates),
                    "verifications": len(self.verifications),
                    "verified": len(self.verified),
                    "exhausted": len(self.exhausted),
                },
            },
        )
        write_jsonl(paths["generated"], (seed.to_dict() for seed in self.generated.values()))
        write_jsonl(paths["candidates"], (claim.to_dict() for claim in self.candidates.values()))
        verification_rows = []
        for claim_id, result in self.verifications.items():
            claim = self.candidates.get(claim_id)
            row = result.to_dict()
            if claim is not None:
                row["seed_id"] = claim.seed_id
                row["claim_text"] = claim.text
                row["cache_key"] = claim_cache_key(claim.seed_id, claim.text)
            verification_rows.append(row)
        write_jsonl(paths["verifications"], verification_rows)
        write_jsonl(paths["verified"], (seed.to_dict() for seed in self.verified.values()))
        write_jsonl(paths["exhausted"], ({"seed_id": seed_id} for seed_id in sorted(self.exhausted)))

    def import_into_run(self, store: RunStore) -> ImportStats:
        stats = ImportStats()
        store.ensure()
        done_generated = store.completed_ids(store.generated_seeds_path, "seed_id")
        done_claims = store.completed_ids(store.candidate_claims_path, "claim_id")
        done_verifications = {
            item.claim_id
            for item in store.verifications()
            if item.parse_status is not ParseStatus.FAILED
        }
        done_verified = store.completed_ids(store.verified_seeds_path, "seed_id")
        done_exhausted = store.completed_ids(store.exhausted_seeds_path, "seed_id")
        extractor_hash = self._extractor_hash()

        for seed_id in sorted(self.verified):
            seed = self.verified[seed_id]
            if seed_id not in done_generated and seed_id in self.generated:
                store.append_generated(self.generated[seed_id])
                done_generated.add(seed_id)
                stats.generated += 1
            elif seed_id not in done_generated:
                store.append_generated(
                    GeneratedSeed(
                        seed_id=seed.seed_id,
                        question_id=seed.question_id,
                        domain=seed.domain,
                        question=seed.question,
                        seed_answer=seed.seed_answer,
                        answer_model=str((seed.answer_model_metadata or {}).get("model") or ""),
                        generation_metadata=dict(seed.answer_model_metadata or {}),
                    )
                )
                done_generated.add(seed_id)
                stats.generated += 1

        for seed_id in sorted(self.exhausted):
            if seed_id in self.verified:
                continue
            if seed_id not in done_generated and seed_id in self.generated:
                store.append_generated(self.generated[seed_id])
                done_generated.add(seed_id)
                stats.generated += 1

        for claim in self.candidates.values():
            if extractor_hash and self._candidate_extractor_hash(claim) not in ("", extractor_hash):
                continue
            if claim.claim_id in done_claims:
                continue
            store.append_claim(claim)
            done_claims.add(claim.claim_id)
            stats.candidates += 1

        for run_claim in store.candidate_claims():
            if run_claim.claim_id in done_verifications:
                continue
            cache = claim_cache_key(run_claim.seed_id, run_claim.text)
            result = self.verifications_by_cache.get(cache)
            if result is None:
                result = self.verifications.get(run_claim.claim_id)
            if result is None or result.parse_status is ParseStatus.FAILED:
                continue
            payload = result.to_dict()
            payload["claim_id"] = run_claim.claim_id
            store.append_verification(VerificationResult.from_dict(payload))
            done_verifications.add(run_claim.claim_id)
            stats.verifications += 1

        for seed_id, seed in self.verified.items():
            if seed_id in done_verified:
                continue
            store.append_verified(seed)
            done_verified.add(seed_id)
            stats.verified += 1

        for seed_id in sorted(self.exhausted):
            if seed_id in done_exhausted:
                continue
            append_jsonl(store.exhausted_seeds_path, {"seed_id": seed_id})
            done_exhausted.add(seed_id)
            stats.exhausted += 1

        return stats

    def publish_run(self, store: RunStore) -> dict[str, int]:
        counts = self.merge_from_run(store, source=str(store.root.resolve()))
        return counts

    def skip_seed_ids(self) -> set[str]:
        return set(self.verified) | set(self.exhausted)
