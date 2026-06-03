"""Scenario 3 — nested sandboxes.

The orchestrator agent itself runs inside an Azure Container Apps sandbox and
spawns additional sandboxes for its workers. This file is both the launcher
(run on your machine) and the supervisor (re-executed inside the orchestrator
sandbox) — selected by ``--mode``.

Flow::

    [your machine] launcher mode
         │  provisions orchestrator sandbox
         │  uploads this script + a JSON config
         │  pip-installs the configured SDK package spec
         ▼
    [orchestrator sandbox] supervisor mode
         │  builds an ACASandboxClient against the worker group
         │  planner Agent decomposes the task into N subtasks
         │  spawns N worker SandboxAgents in parallel (each in its own sandbox)
         │  synthesizer Agent merges their findings
         ▼
    final answer (printed back to launcher stdout)

Prerequisites:

- ``OPENAI_API_KEY`` and ``ACA_*`` variables for the orchestrator group.
- ``ACA_WORKER_*`` variables for the worker group (subscription, resource
  group, sandbox group, region).
- Azure service-principal env vars (``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``,
  ``AZURE_CLIENT_SECRET``) so that the supervisor — running inside the
  orchestrator — can authenticate as itself against the worker group. The
  principal must have data-plane permissions on the worker group.
- See ``examples/sandbox/extensions/aca/README.md`` for the full setup notes,
  including the secrets-in-orchestrator-workspace tradeoff.

Run::

    uv run python examples/sandbox/extensions/aca/nested_sandboxes.py \\
        --task "Compare Django and FastAPI for high-throughput APIs." \\
        --workers 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ---- mode selection -----------------------------------------------------------------

LAUNCHER_REQUIRED_ENV = (
    "OPENAI_API_KEY",
    "ACA_SUBSCRIPTION_ID",
    "ACA_RESOURCE_GROUP",
    "ACA_SANDBOX_GROUP",
    "ACA_REGION",
    "ACA_WORKER_SUBSCRIPTION_ID",
    "ACA_WORKER_RESOURCE_GROUP",
    "ACA_WORKER_SANDBOX_GROUP",
    "ACA_WORKER_REGION",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
)

SUPERVISOR_REQUIRED_ENV = (
    "OPENAI_API_KEY",
    "ACA_SUBSCRIPTION_ID",
    "ACA_RESOURCE_GROUP",
    "ACA_SANDBOX_GROUP",
    "ACA_REGION",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
)


# =====================================================================================
# Launcher (runs on your machine)
# =====================================================================================


BOOTSTRAP_SCRIPT = """\
set -euo pipefail
cd /workspace/orchestrator
python3 -m pip install --quiet --break-system-packages \\
    -r requirements.txt 2>&1 | tail -n 5
exec python3 nested.py --mode supervisor config.json
"""


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _normalize_git_url(url: str) -> str:
    if url.startswith("git@github.com:"):
        return f"https://github.com/{url.removeprefix('git@github.com:')}"
    return url


def _default_package_spec() -> str:
    override = os.environ.get("OPENAI_AGENTS_PACKAGE_SPEC")
    if override:
        return override

    remote = _run_git("remote", "get-url", "origin")
    commit = _run_git("rev-parse", "HEAD")
    if remote and commit:
        return f"openai-agents[aca-sandbox] @ git+{_normalize_git_url(remote)}@{commit}"

    return "openai-agents[aca-sandbox]"


async def _run_launcher(
    task: str,
    workers: int,
    model: str,
    disk: str,
    package_spec: str,
) -> int:
    if __package__ is None or __package__ == "":
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from examples.sandbox.extensions.aca._support import (
        build_group_client,
        close_group_client,
    )

    try:
        from agents.extensions.sandbox import (
            ACASandboxClient,
            ACASandboxClientOptions,
        )
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Install the optional extra: uv sync --extra aca-sandbox") from exc

    missing = [v for v in LAUNCHER_REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        raise SystemExit("Missing required env vars: " + ", ".join(missing))

    run_id = uuid.uuid4().hex[:8]
    print(f"==> Run id  : {run_id}")
    print(f"==> Task    : {task}")
    print(f"==> Workers : {workers}")
    print(f"==> Package : {package_spec}")
    print()
    print("[1/4] Provisioning orchestrator sandbox...")

    group_client = build_group_client()
    sandbox_client = ACASandboxClient(group_client)
    options = ACASandboxClientOptions(
        disk=disk,
        labels={"demo": "nested-sandboxes", "role": "orchestrator", "run-id": run_id},
    )
    session = await sandbox_client.create(options=options)

    try:
        print("[2/4] Uploading supervisor script and configuration...")
        script_bytes = Path(__file__).read_bytes()
        import io as _io  # local alias to avoid leaking import name into module

        await session.write(Path("orchestrator/nested.py"), _io.BytesIO(script_bytes))

        # Worker-group config + secrets handed to the supervisor. These end up
        # in the orchestrator sandbox workspace; see README for tradeoffs.
        config = {
            "task": task,
            "workers": workers,
            "model": model,
            "disk": disk,
            "run_id": run_id,
            "package_spec": package_spec,
            "worker_group": {
                "subscription_id": os.environ["ACA_WORKER_SUBSCRIPTION_ID"],
                "resource_group": os.environ["ACA_WORKER_RESOURCE_GROUP"],
                "sandbox_group": os.environ["ACA_WORKER_SANDBOX_GROUP"],
                "region": os.environ["ACA_WORKER_REGION"],
            },
            "env": {
                k: os.environ[k]
                for k in (
                    "OPENAI_API_KEY",
                    "AZURE_TENANT_ID",
                    "AZURE_CLIENT_ID",
                    "AZURE_CLIENT_SECRET",
                )
            },
        }
        await session.write(
            Path("orchestrator/config.json"),
            _io.BytesIO(json.dumps(config).encode("utf-8")),
        )
        await session.write(
            Path("orchestrator/bootstrap.sh"),
            _io.BytesIO(BOOTSTRAP_SCRIPT.encode("utf-8")),
        )
        await session.write(
            Path("orchestrator/requirements.txt"),
            _io.BytesIO(f"{package_spec}\n".encode()),
        )

        print("[3/4] Installing SDK and starting supervisor inside the orchestrator...")
        started = time.perf_counter()
        result = await session.exec("bash /workspace/orchestrator/bootstrap.sh", timeout=900.0)
        elapsed = time.perf_counter() - started

        print()
        print(f"[4/4] Orchestrator finished in {elapsed:.1f}s (exit={result.exit_code})")
        print("-" * 60)
        sys.stdout.write(result.stdout.decode("utf-8", errors="replace"))
        if result.exit_code != 0:
            sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
            return 1
        return 0
    finally:
        print()
        print("Cleaning up orchestrator sandbox...")
        try:
            await sandbox_client.delete(session)
        finally:
            await close_group_client(group_client)


# =====================================================================================
# Supervisor (runs inside the orchestrator sandbox)
# =====================================================================================


DECOMPOSE_PROMPT = """\
You are a research planner. Decompose the user's task into between {min_n} and \
{max_n} focused, independent subtasks that can be researched in parallel.

Respond with ONLY a JSON array (no markdown fences, no preamble). Each item:
  {{"key": "<short-id>", "title": "<concise title>", "question": "<focused research question>"}}

The "key" must be a short, lowercase, alphanumeric/hyphen identifier unique
across items. The "question" must be answerable by a single web/source agent
without further decomposition.
"""

SYNTHESIZE_PROMPT = """\
You are a synthesizer. Combine the worker findings below into a single, \
coherent answer to the original task. Preserve citations from the workers. \
Prefer concrete facts over speculation; if the workers disagree, surface that.

Original task: {task}

Worker findings (Markdown):
{findings}

Write the final answer in Markdown.
"""

WORKER_INSTRUCTIONS = """\
You are a research worker. Investigate exactly the question below and return \
a concise Markdown answer with citations to any sources you consulted. Use the \
shell and filesystem tools to explore.

Question: {question}
"""


def _parse_subtasks(raw: str, min_n: int, max_n: int) -> list[dict[str, str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("planner output is not a JSON array")
    if not (min_n <= len(data) <= max_n):
        raise ValueError(f"planner returned {len(data)} subtasks, want {min_n}-{max_n}")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, item in enumerate(data):
        key = str(item["key"]).strip().lower()
        if not key or key in seen:
            raise ValueError(f"subtask {i}: invalid or duplicate key {key!r}")
        seen.add(key)
        out.append(
            {
                "key": key,
                "title": str(item["title"]).strip(),
                "question": str(item["question"]).strip(),
            }
        )
    return out


async def _run_supervisor(config_path: str) -> int:
    cfg: dict[str, Any] = json.loads(Path(config_path).read_text(encoding="utf-8"))

    # Propagate secrets handed in by the launcher.
    for k, v in cfg.get("env", {}).items():
        os.environ.setdefault(k, v)
    wg = cfg["worker_group"]
    os.environ["ACA_SUBSCRIPTION_ID"] = wg["subscription_id"]
    os.environ["ACA_RESOURCE_GROUP"] = wg["resource_group"]
    os.environ["ACA_SANDBOX_GROUP"] = wg["sandbox_group"]
    os.environ["ACA_REGION"] = wg["region"]

    missing = [v for v in SUPERVISOR_REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        raise SystemExit("Supervisor missing env vars: " + ", ".join(missing))

    # Imports happen here so the launcher does not require the SDK to be
    # locally importable from inside the orchestrator's interpreter.
    from azure.containerapps.sandbox.aio import SandboxGroupClient
    from azure.identity.aio import DefaultAzureCredential

    from agents import Agent, Runner
    from agents.extensions.sandbox import ACASandboxClient, ACASandboxClientOptions
    from agents.run import RunConfig
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Filesystem, Shell

    task: str = cfg["task"]
    workers: int = int(cfg["workers"])
    model: str = cfg["model"]
    disk: str = cfg["disk"]
    run_id: str = cfg["run_id"]

    print(f"[supervisor] run_id={run_id} task={task!r} workers={workers}")

    planner = Agent(
        name="planner",
        model=model,
        instructions=DECOMPOSE_PROMPT.format(min_n=workers, max_n=workers),
    )
    plan_result = await Runner.run(planner, task, max_turns=3)
    subtasks = _parse_subtasks(plan_result.final_output or "", workers, workers)
    print(
        f"[supervisor] decomposed into {len(subtasks)} subtasks: "
        f"{', '.join(s['key'] for s in subtasks)}"
    )

    credential = DefaultAzureCredential()
    endpoint = f"https://management.{wg['region']}.azuredevcompute.io"
    group_client = SandboxGroupClient(
        endpoint=endpoint,
        subscription_id=wg["subscription_id"],
        resource_group=wg["resource_group"],
        sandbox_group=wg["sandbox_group"],
        credential=credential,
    )
    worker_client = ACASandboxClient(group_client)
    worker_options = ACASandboxClientOptions(
        disk=disk,
        labels={"demo": "nested-sandboxes", "role": "worker", "run-id": run_id},
    )

    async def _run_worker(sub: dict[str, str]) -> tuple[str, str]:
        agent = SandboxAgent(
            name=f"worker-{sub['key']}",
            model=model,
            instructions=WORKER_INSTRUCTIONS.format(question=sub["question"]),
            capabilities=[Shell(), Filesystem()],
        )
        run_config = RunConfig(
            sandbox=SandboxRunConfig(client=worker_client, options=worker_options),
            workflow_name=f"nested worker {sub['key']}",
        )
        print(f"[supervisor] launching worker sandbox for {sub['key']}")
        result = await Runner.run(agent, sub["question"], run_config=run_config, max_turns=12)
        return sub["title"], (result.final_output or "").strip()

    try:
        findings = await asyncio.gather(*(_run_worker(s) for s in subtasks))
    finally:
        await group_client.close()
        await credential.close()

    findings_md = "\n\n".join(f"### {title}\n\n{answer}" for title, answer in findings)
    synth = Agent(
        name="synthesizer",
        model=model,
        instructions="You produce concise, well-cited Markdown.",
    )
    synth_result = await Runner.run(
        synth,
        SYNTHESIZE_PROMPT.format(task=task, findings=findings_md),
        max_turns=3,
    )

    print()
    print("===== FINAL ANSWER =====")
    print(synth_result.final_output or "")
    return 0


# =====================================================================================
# Entry point
# =====================================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ACA sandbox example: orchestrator sandbox that spawns worker sandboxes.",
    )
    parser.add_argument(
        "--mode",
        choices=("launcher", "supervisor"),
        default="launcher",
        help="Run on your machine ('launcher', default) or inside the orchestrator sandbox.",
    )
    parser.add_argument("--task", help="Research task (launcher mode).")
    parser.add_argument("--workers", type=int, default=3, help="Number of worker sandboxes.")
    parser.add_argument("--model", default="gpt-5.5", help="Model name to use.")
    parser.add_argument("--disk", default="ubuntu", help="Public disk image name.")
    parser.add_argument(
        "--package-spec",
        default=_default_package_spec(),
        help=(
            "Package spec installed inside the orchestrator sandbox. Defaults to "
            "OPENAI_AGENTS_PACKAGE_SPEC, then the current git origin+HEAD, then "
            "'openai-agents[aca-sandbox]'."
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to config.json (supervisor mode only).",
    )
    args = parser.parse_args(argv)

    if args.mode == "launcher":
        if not args.task:
            parser.error("--task is required in launcher mode.")
        if args.workers < 1 or args.workers > 8:
            parser.error("--workers must be between 1 and 8.")
        return asyncio.run(
            _run_launcher(args.task, args.workers, args.model, args.disk, args.package_spec)
        )

    if not args.config:
        parser.error("supervisor mode requires a config path argument.")
    return asyncio.run(_run_supervisor(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
