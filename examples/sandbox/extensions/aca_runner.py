"""
Minimal Azure Container Apps Sandboxes example for manual validation.

This example provisions a sandbox via :class:`ACASandboxClient`, asks the agent a
single question against a small in-memory manifest, then deletes the sandbox.

Prerequisites:

- ``OPENAI_API_KEY``
- An Azure subscription with a sandbox group ready to use, identified by:
    - ``ACA_SUBSCRIPTION_ID``
    - ``ACA_RESOURCE_GROUP``
    - ``ACA_SANDBOX_GROUP``
    - ``ACA_REGION``
- Azure credentials available to :class:`azure.identity.aio.DefaultAzureCredential`
  (e.g. ``az login`` for local development or a managed identity in CI).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from openai.types.responses import ResponseTextDeltaEvent

from agents import ModelSettings, Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.sandbox.misc.example_support import text_manifest
from examples.sandbox.misc.workspace_shell import WorkspaceShellCapability

try:
    from agents.extensions.sandbox import (
        ACASandboxClient,
        ACASandboxClientOptions,
    )
except Exception as exc:  # pragma: no cover - import path depends on optional extras
    raise SystemExit(
        "Azure Container Apps sandbox examples require the optional repo extra.\n"
        "Install it with: uv sync --extra aca-sandbox"
    ) from exc


DEFAULT_QUESTION = "Summarize this workspace in 2 sentences."


def _build_manifest() -> Manifest:
    return text_manifest(
        {
            "README.md": (
                "# Azure Container Apps Sandbox Demo Workspace\n\n"
                "This workspace exists to validate the Azure Container Apps sandbox "
                "backend manually.\n"
            ),
            "notes.md": (
                "# Notes\n\n"
                "- Sandbox group resources are provisioned via the Azure SDK.\n"
                "- The workspace root is `/workspace` inside the sandbox.\n"
            ),
        }
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set before running this example.")
    return value


def _build_group_client() -> Any:
    """Build an Azure Container Apps SandboxGroupClient from environment + credentials."""

    try:
        from azure.containerapps.sandbox.aio import SandboxGroupClient
        from azure.identity.aio import DefaultAzureCredential
    except Exception as exc:  # pragma: no cover - guarded by the extras import above
        raise SystemExit(
            "Azure Container Apps sandbox examples require the optional repo extra.\n"
            "Install it with: uv sync --extra aca-sandbox"
        ) from exc

    subscription_id = _require_env("ACA_SUBSCRIPTION_ID")
    resource_group = _require_env("ACA_RESOURCE_GROUP")
    sandbox_group = _require_env("ACA_SANDBOX_GROUP")
    region = _require_env("ACA_REGION")

    credential = DefaultAzureCredential()
    return SandboxGroupClient(
        endpoint=f"https://management.{region}.azuredevcompute.io",
        subscription_id=subscription_id,
        resource_group=resource_group,
        sandbox_group=sandbox_group,
        credential=credential,
    )


async def main(*, model: str, question: str, disk: str, stream: bool) -> None:
    _require_env("OPENAI_API_KEY")
    manifest = _build_manifest()
    group_client = _build_group_client()

    agent = SandboxAgent(
        name="ACA Sandbox Assistant",
        model=model,
        instructions=(
            "Answer questions about the sandbox workspace. Inspect the files before "
            "answering and keep the response concise. Do not invent files that are not "
            "present in the workspace. Cite the file names you inspected."
        ),
        default_manifest=manifest,
        capabilities=[WorkspaceShellCapability()],
        model_settings=ModelSettings(tool_choice="required"),
    )

    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=ACASandboxClient(group_client),
            options=ACASandboxClientOptions(disk=disk),
        ),
        workflow_name="Azure Container Apps sandbox example",
    )

    try:
        if not stream:
            result = await Runner.run(agent, question, run_config=run_config)
            print(result.final_output)
            return

        stream_result = Runner.run_streamed(agent, question, run_config=run_config)
        saw_text_delta = False
        async for event in stream_result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                if not saw_text_delta:
                    print("assistant> ", end="", flush=True)
                    saw_text_delta = True
                print(event.data.delta, end="", flush=True)

        if saw_text_delta:
            print()
    finally:
        close = getattr(group_client, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.5", help="Model name to use.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Prompt to send to the agent.")
    parser.add_argument(
        "--disk",
        default="ubuntu",
        help="Public disk image name (see your sandbox group's disk catalog).",
    )
    parser.add_argument("--stream", action="store_true", default=False, help="Stream the response.")
    args = parser.parse_args()

    asyncio.run(
        main(
            model=args.model,
            question=args.question,
            disk=args.disk,
            stream=args.stream,
        )
    )
