"""Generate until n verified-false seeds exist, then run the remaining stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

from branching_hallucinations.env import load_branching_env

load_branching_env()

from branching_hallucinations.cli import main as cli_main
from branching_hallucinations.storage import RunStore


def _count_verified(run: str) -> int:
    store = RunStore(run)
    if not store.verified_seeds_path.exists():
        return 0
    return len(store.verified_seeds())


def _count_generated(run: str) -> int:
    store = RunStore(run)
    if not store.generated_seeds_path.exists():
        return 0
    return len(store.generated_seeds())


def _python() -> Path:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        return venv
    return Path(sys.executable)


def _run(command: list[str]) -> None:
    py = str(_python())
    print("+", py, "-m", "branching_hallucinations", *command, flush=True)
    code = cli_main(command)
    if code:
        raise SystemExit(code)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/branching_pilot.toml")
    parser.add_argument("--run", default="runs/pilot-oss120b")
    parser.add_argument("--target-verified", type=int, default=10)
    parser.add_argument("--batch", type=int, default=24)
    args = parser.parse_args()

    common = ["--config", args.config, "--run", args.run]
    _run(["init-run", *common])

    target = args.target_verified
    batch = args.batch
    pool = max(batch, target)
    while _count_verified(args.run) < target:
        before = _count_generated(args.run)
        pool = max(pool, before + batch)
        _run(["generate-seeds", *common, "--n", str(pool)])
        after = _count_generated(args.run)
        if after == before:
            print(
                f"No new generated seeds ({after}). "
                f"Verified-false={_count_verified(args.run)}/{target}. Stopping fill loop.",
                flush=True,
            )
            break
        _run(["extract-claims", *common])
        _run(["verify-seeds", *common, "--max-verified", str(target)])
        print(
            f"Progress: generated={after} verified-false={_count_verified(args.run)}/{target}",
            flush=True,
        )
        pool += batch

    n_verified = _count_verified(args.run)
    if n_verified < target:
        raise SystemExit(
            f"Only {n_verified} verified-false seeds; wanted {target}."
        )

    _run(["generate-tree", *common])
    _run(["audit-actions", *common, "--version", "v1"])
    _run(["judge-trajectories", *common, "--version", "v1"])
    _run(["analyze", *common, "--trajectory-version", "v1"])
    _run(["export-audit", *common, "--trajectory-version", "v1"])
    print(f"Done. verified-false={n_verified} run={args.run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
