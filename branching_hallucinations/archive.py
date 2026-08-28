"""Archive compact, committable pilot artifacts from a live run directory."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .storage import RunStore, git_sha, utc_now, write_json

# Full conversation audit CSVs are omitted (multi-MB). Re-export from the live run if needed.
ARCHIVE_REL_PATHS = (
    "manifest.json",
    "reports/summary.json",
    "reports/seed_audit.csv",
    "analysis/summary.json",
    "analysis/t1_states.csv",
    "analysis/transitions.csv",
    "analysis/bootstrap.csv",
    "seeds/verified.jsonl",
)


def archive_run(
    run_dir: str | Path,
    dest_dir: str | Path,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Copy scientific summaries from a gitignored run into a tracked archive directory."""
    store = RunStore(run_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for rel in ARCHIVE_REL_PATHS:
        src = store.root / rel
        if not src.exists():
            missing.append(rel)
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
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
        "copied": copied,
        "missing": missing,
    }
    write_json(dest / "archive_meta.json", meta)
    return meta


def default_archive_dest(run_dir: str | Path, archive_root: str | Path | None = None) -> Path:
    root = Path(archive_root or "results/pilots")
    return root / Path(run_dir).name
