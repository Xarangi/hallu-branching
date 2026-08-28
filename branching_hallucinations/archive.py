"""Archive pilot artifacts from a live run directory for version control."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .storage import RunStore, git_sha, utc_now, write_json

# Aggregated analysis and frozen seeds (always attempted).
ARCHIVE_SUMMARY_PATHS = (
    "manifest.json",
    "reports/summary.json",
    "reports/seed_audit.csv",
    "analysis/summary.json",
    "analysis/t1_states.csv",
    "analysis/transitions.csv",
    "analysis/bootstrap.csv",
    "seeds/verified.jsonl",
)

# Per-turn audit trail: follow-up questions, answers, labels, action audits.
ARCHIVE_AUDIT_PATHS = (
    "tree/nodes.jsonl",
    "reports/intervention_audit.csv",
    "reports/trajectory_audit.csv",
    "seeds/generated.jsonl",
    "seeds/candidates.jsonl",
    "seeds/verifications.jsonl",
)


def _copy_rel(store: RunStore, dest: Path, rel: str) -> bool:
    src = store.root / rel
    if not src.exists():
        return False
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return True


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def archive_run(
    run_dir: str | Path,
    dest_dir: str | Path,
    *,
    label: str | None = None,
    trajectory_version: str = "v1",
    include_audit_trail: bool = True,
) -> dict[str, Any]:
    """Copy summaries and optional per-turn logs into a tracked archive directory."""
    store = RunStore(run_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []

    for rel in ARCHIVE_SUMMARY_PATHS:
        if _copy_rel(store, dest, rel):
            copied.append(rel)
        else:
            missing.append(rel)

    if include_audit_trail:
        for rel in ARCHIVE_AUDIT_PATHS:
            if _copy_rel(store, dest, rel):
                copied.append(rel)
            else:
                missing.append(rel)
        judgments_dir = store.judgments_dir
        if judgments_dir.exists():
            for src in sorted(judgments_dir.glob("*.jsonl")):
                rel = _rel_posix(src, store.root)
                if _copy_rel(store, dest, rel):
                    copied.append(rel)

    if "reports/summary.json" in missing:
        raise FileNotFoundError(
            f"{store.root}: missing reports/summary.json — run analyze and export-audit first"
        )

    meta = {
        "label": label or store.root.name,
        "source_run": str(store.root.resolve()),
        "archived_at": utc_now(),
        "git_sha": git_sha(),
        "trajectory_version": trajectory_version,
        "include_audit_trail": include_audit_trail,
        "copied": copied,
        "missing": missing,
    }
    write_json(dest / "archive_meta.json", meta)
    return meta


def default_archive_dest(run_dir: str | Path, archive_root: str | Path | None = None) -> Path:
    root = Path(archive_root or "results/pilots")
    return root / Path(run_dir).name
