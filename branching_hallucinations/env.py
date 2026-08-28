"""Load repo .env for the branching experiment and ignore unrelated machine keys."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .config import REPO_ROOT

_ENV_PATH = REPO_ROOT / ".env"


def load_branching_env() -> None:
    """Load project secrets and prefer Azure over a stale system OPENAI_API_KEY."""
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=True)
    if os.getenv("AZURE_OPENAI_API_KEY", "").strip():
        # HalluHard content filtering uses embeddings. A leftover machine-wide
        # OPENAI_API_KEY must not override the Azure credentials you configured.
        os.environ.pop("OPENAI_API_KEY", None)
