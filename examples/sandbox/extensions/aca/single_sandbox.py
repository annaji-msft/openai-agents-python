"""Scenario 1 — single sandbox.

One Azure Container Apps sandbox, one :class:`SandboxAgent` with shell and
filesystem capabilities. The agent clones a GitHub repository and answers a
natural-language question about it, citing the files it inspected.

Run::

    uv run python examples/sandbox/extensions/aca/single_sandbox.py \\
        --repo https://github.com/openai/openai-agents-python \\
        "What are the main agent capabilities and how do they fit together?"

Prerequisites:

- ``OPENAI_API_KEY``
- The standard ``ACA_*`` environment variables documented in
  ``examples/sandbox/extensions/aca/README.md``.
- Azure credentials usable by :class:`azure.identity.aio.DefaultAzureCredential`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from agents import Runner  # noqa: E402
from agents.run import RunConfig  # noqa: E402
from agents.sandbox import SandboxAgent, SandboxRunConfig  # noqa: E402
from agents.sandbox.capabilities import Filesystem, Shell  # noqa: E402
from examples.sandbox.extensions.aca._support import (  # noqa: E402
    build_group_client,
    close_group_client,
)

try:
    from agents.extensions.sandbox import (
        ACASandboxClient,
        ACASandboxClientOptions,
    )
except Exception as exc:  # pragma: no cover - import depends on optional extras
    raise SystemExit(
        "Azure Container Apps sandbox examples require the optional repo extra.\n"
        "Install it with: uv sync --extra aca-sandbox"
    ) from exc

DEFAULT_REPO = "https://github.com/openai/openai-agents-python"
DEFAULT_QUESTION = "Summarize the main purpose and high-level structure of this repository."

INSTRUCTIONS_TEMPLATE = """\
You are a research agent running inside an isolated Azure Container Apps sandbox.
Your task is to analyze a GitHub repository and answer questions about it.

Repository location: {repo_path}

Operating procedure:
1. Clone the repository (shallow):
       cd /workspace && git clone --depth 1 {repo_url} repo
2. Explore the repository structure:
       ls -la /workspace/repo/
       read any README.* at the root
       look for key manifests (pyproject.toml, package.json, go.mod, etc.)
3. To answer the question:
       grep -RIn "keyword" /workspace/repo/
       read relevant files with cat or the filesystem tool
4. Cite the files you inspected; include line numbers for code references.
5. If the answer cannot be found in the repository, say so explicitly.

Output format:
- Direct answer to the question.
- Key findings with file citations.
- Optional: short, relevant code snippets.
"""


async def run_demo(repo_url: str, question: str, *, model: str, disk: str) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set.")

    run_id = uuid.uuid4().hex[:8]
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")

    print(f"==> Run id     : {run_id}")
    print(f"==> Repository : {repo_url}")
    print(f"==> Question   : {question}")
    print()

    group_client = build_group_client()
    try:
        sandbox_client = ACASandboxClient(group_client)
        options = ACASandboxClientOptions(
            disk=disk,
            labels={"demo": "single-sandbox", "run-id": run_id, "repo": repo_name},
        )

        agent = SandboxAgent(
            name="repo-researcher",
            model=model,
            instructions=INSTRUCTIONS_TEMPLATE.format(
                repo_url=repo_url,
                repo_path="/workspace/repo",
            ),
            capabilities=[Shell(), Filesystem()],
        )

        run_config = RunConfig(
            sandbox=SandboxRunConfig(client=sandbox_client, options=options),
            workflow_name="ACA single sandbox",
        )

        print("Provisioning sandbox and starting research...")
        started = time.perf_counter()
        result = await Runner.run(agent, question, run_config=run_config, max_turns=15)
        elapsed = time.perf_counter() - started

        print()
        print(f"Research complete in {elapsed:.1f}s")
        print("-" * 60)
        print(result.final_output)
        print()
        return 0
    finally:
        await close_group_client(group_client)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ACA sandbox example: one agent inside one sandbox.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository URL to analyze.")
    parser.add_argument("--model", default="gpt-5.5", help="Model name to use.")
    parser.add_argument("--disk", default="ubuntu", help="Public disk image name.")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question to answer about the repository.",
    )
    args = parser.parse_args(argv)
    return asyncio.run(run_demo(args.repo, args.question, model=args.model, disk=args.disk))


if __name__ == "__main__":
    raise SystemExit(main())
