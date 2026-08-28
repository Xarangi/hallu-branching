"""Shared Azure OpenAI endpoint/credential helpers."""

from __future__ import annotations

import os

DEFAULT_API_VERSION = "2024-12-01-preview"


def normalize_azure_endpoint(raw: str) -> str:
    """Strip /openai/responses query URLs down to the Azure resource root."""
    value = (raw or "").strip()
    if "?" in value:
        value = value.split("?", 1)[0]
    value = value.rstrip("/")
    for suffix in ("/openai/responses", "/openai/v1", "/openai"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.rstrip("/")


def azure_credentials() -> tuple[str, str]:
    endpoint = normalize_azure_endpoint(
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_ENDPOINT")
        or ""
    )
    key = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_API_KEY")
        or ""
    ).strip()
    return endpoint, key


def azure_api_version() -> str:
    return (os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION).strip()


def azure_endpoint_for_api(api: str) -> str:
    default, _key = azure_credentials()
    if api == "responses":
        alt = os.environ.get("AZURE_GPT5_ENDPOINT") or os.environ.get("AZURE_RESPONSES_ENDPOINT") or ""
        if alt.strip():
            return normalize_azure_endpoint(alt)
    return default
