"""Shared helpers for the Azure Container Apps sandbox example scenarios.

All scenarios authenticate through :class:`azure.identity.aio.DefaultAzureCredential`
and build a :class:`SandboxGroupClient` from a small set of environment
variables. Keeping the boilerplate here lets the scenario files focus on the
agent logic.
"""

from __future__ import annotations

import os
from typing import Any


def require_env(name: str) -> str:
    """Return ``os.environ[name]`` or exit with a helpful error message."""

    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} must be set. The Azure Container Apps sandbox examples "
            "require ACA_SUBSCRIPTION_ID, ACA_RESOURCE_GROUP, ACA_SANDBOX_GROUP, "
            "and ACA_REGION."
        )
    return value


def build_group_client() -> Any:
    """Build an Azure Container Apps SandboxGroupClient from ACA_* env vars."""

    try:
        from azure.containerapps.sandbox.aio import SandboxGroupClient
        from azure.identity.aio import DefaultAzureCredential
    except Exception as exc:  # pragma: no cover - import guarded by extras
        raise SystemExit(
            "Azure Container Apps sandbox examples require the optional repo extra.\n"
            "Install it with: uv sync --extra aca-sandbox"
        ) from exc

    subscription_id = require_env("ACA_SUBSCRIPTION_ID")
    resource_group = require_env("ACA_RESOURCE_GROUP")
    sandbox_group = require_env("ACA_SANDBOX_GROUP")
    region = require_env("ACA_REGION")

    credential = DefaultAzureCredential()
    return SandboxGroupClient(
        endpoint=f"https://management.{region}.azuredevcompute.io",
        subscription_id=subscription_id,
        resource_group=resource_group,
        sandbox_group=sandbox_group,
        credential=credential,
    )


async def close_group_client(group_client: Any) -> None:
    """Best-effort teardown of a SandboxGroupClient and its credential."""

    close = getattr(group_client, "close", None)
    if callable(close):
        try:
            await close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
