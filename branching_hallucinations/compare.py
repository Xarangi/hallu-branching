"""Compare analysis summaries across multiple pilot runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .storage import write_json


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_path(run_or_summary: str | Path) -> Path:
    path = Path(run_or_summary)
    if path.is_dir():
        for candidate in (path / "reports" / "summary.json", path / "summary.json"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No summary.json under {path}")
    return path


def _t1_active(summary: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for action, payload in summary.get("t1", {}).get("by_action", {}).items():
        active = payload.get("active")
        if isinstance(active, list) and active:
            out[action] = float(active[0])
    return out


def _t1_counts(summary: dict[str, Any], action: str) -> dict[str, int]:
    payload = summary.get("t1", {}).get("by_action", {}).get(action, {})
    return {
        state: int(payload.get(state, {}).get("count", 0))
        for state in ("DROP", "RETRACT", "REPEAT", "DEPEND")
    }


def compare_runs(labeled_paths: dict[str, str | Path]) -> dict[str, Any]:
    """Build a side-by-side comparison from run dirs or summary.json paths."""
    summaries: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for label, run_or_summary in labeled_paths.items():
        summary_path = _summary_path(run_or_summary)
        summaries[label] = _load_summary(summary_path)
        paths[label] = str(summary_path.resolve())

    labels = list(labeled_paths.keys())
    comparison: dict[str, Any] = {
        "labels": labels,
        "paths": paths,
        "overview": {},
        "t1_active": {},
        "t1_by_action": {},
        "t2_bootstrap_active": {},
        "action_compliance": {},
        "verification_attrition": {},
    }

    for label in labels:
        summary = summaries[label]
        comparison["overview"][label] = {
            "n_verified_seeds": summary.get("n_verified_seeds"),
            "domain_composition": summary.get("domain_composition", {}),
            "n_nodes": summary.get("n_nodes"),
            "n_judgments_ok": summary.get("n_judgments_ok"),
            "n_judgments_failed": summary.get("n_judgments_failed"),
        }
        comparison["t1_active"][label] = _t1_active(summary)
        comparison["t1_by_action"][label] = {
            action: _t1_counts(summary, action) for action in ("D", "N", "V")
        }
        comparison["t2_bootstrap_active"][label] = summary.get("t2", {}).get(
            "seed_cluster_bootstrap_active", {}
        )
        comparison["action_compliance"][label] = summary.get("action_compliance", {})
        comparison["verification_attrition"][label] = summary.get("verification_attrition", {})

    if len(labels) == 2:
        left, right = labels
        comparison["paired_mcnemar_active"] = {
            left: summaries[left].get("t1", {}).get("paired_mcnemar_active", {}),
            right: summaries[right].get("t1", {}).get("paired_mcnemar_active", {}),
        }

    return comparison


def write_comparison_markdown(comparison: dict[str, Any], path: Path) -> None:
    labels = comparison["labels"]
    lines = ["# Pilot comparison", ""]
    lines.append("## Overview")
    lines.append("")
    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "|".join(["---"] * len(labels)) + "|")
    for key in ("n_verified_seeds", "n_nodes", "n_judgments_ok", "n_judgments_failed"):
        row = [key] + [str(comparison["overview"][label].get(key, "")) for label in labels]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## T1 P(active)")
    lines.append("")
    lines.append("| Action | " + " | ".join(labels) + " |")
    lines.append("|---|" + "|".join(["---"] * len(labels)) + "|")
    for action in ("D", "N", "V"):
        row = [action] + [
            f"{comparison['t1_active'][label].get(action, 0):.0%}"
            if action in comparison["t1_active"][label]
            else "—"
            for label in labels
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## N intervention compliance")
    lines.append("")
    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "|".join(["---"] * len(labels)) + "|")
    compliance_row = [
        f"{comparison['action_compliance'][label].get('N', {}).get('compliance', 0):.0%}"
        for label in labels
    ]
    lines.append("| N compliance | " + " | ".join(compliance_row) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_and_write(
    labeled_paths: dict[str, str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    out = Path(out_dir)
    comparison = compare_runs(labeled_paths)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "comparison.json", comparison)
    write_comparison_markdown(comparison, out / "comparison.md")
    return comparison
