"""Scenario 2 — parallel sandboxes.

Many worker agents, each in its own Azure Container Apps sandbox, run in
parallel against different research topics. Concurrency is bounded by an
``asyncio.Semaphore`` and the per-topic findings are aggregated into Markdown
and a ``summary.json`` under ``./.run-output/parallel-<run-id>/``.

Run::

    uv run python examples/sandbox/extensions/aca/parallel_sandboxes.py \\
        --topic "fastapi|FastAPI|https://github.com/fastapi/fastapi|What makes FastAPI performant?" \\
        --topic "django|Django|https://github.com/django/django|What are Django's key features?" \\
        --concurrency 2

Topic format::

    key|name|source_url|question

Prerequisites:

- ``OPENAI_API_KEY``
- The standard ``ACA_*`` environment variables documented in
  ``examples/sandbox/extensions/aca/README.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


WORKER_INSTRUCTIONS = """\
You are a research worker analyzing a specific topic inside an isolated sandbox.

Task: {question}

Instructions:
1. Clone the source (shallow):
       cd /workspace && git clone --depth 1 {source} repo
2. Explore: ls /workspace/repo/, then read README files and the most relevant
   directories.
3. Research thoroughly to answer the question. Focus on: what it is, how it
   works, key features, and notable trade-offs.

Respond with a short summary (2-3 sentences) followed by key findings as
bullet points with file citations.
"""


@dataclass(frozen=True)
class ResearchTopic:
    key: str
    name: str
    source: str
    question: str


@dataclass
class WorkerResult:
    topic: ResearchTopic
    success: bool
    answer: str
    elapsed: float
    error: str = ""


def parse_topic_arg(spec: str) -> ResearchTopic:
    parts = spec.split("|", 3)
    if len(parts) != 4:
        raise ValueError(f"Topic must be 'key|name|source|question' (got {len(parts)} parts).")
    key, name, source, question = (p.strip() for p in parts)
    key = key.lower()
    if not key or not all(ch.isalnum() or ch == "-" for ch in key):
        raise ValueError(f"Topic key must be lowercase alphanumeric+dashes; got {key!r}.")
    return ResearchTopic(key=key, name=name, source=source, question=question)


async def research_topic(
    topic: ResearchTopic,
    sandbox_client: Any,
    options: ACASandboxClientOptions,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> WorkerResult:
    async with semaphore:
        started = time.perf_counter()
        print(f"    [{topic.key}] provisioning sandbox...")
        try:
            async with asyncio.timeout(timeout):
                agent = SandboxAgent(
                    name=f"research-{topic.key}",
                    model=model,
                    instructions=WORKER_INSTRUCTIONS.format(
                        question=topic.question,
                        source=topic.source,
                    ),
                    capabilities=[Shell(), Filesystem()],
                )
                run_config = RunConfig(
                    sandbox=SandboxRunConfig(client=sandbox_client, options=options),
                    workflow_name=f"ACA parallel worker: {topic.key}",
                )
                result = await Runner.run(
                    agent, topic.question, run_config=run_config, max_turns=15
                )
                elapsed = time.perf_counter() - started
                print(f"    [{topic.key}] done in {elapsed:.1f}s")
                return WorkerResult(
                    topic=topic,
                    success=True,
                    answer=(result.final_output or "").strip(),
                    elapsed=elapsed,
                )
        except TimeoutError:
            elapsed = time.perf_counter() - started
            print(f"    [{topic.key}] timeout after {elapsed:.1f}s")
            return WorkerResult(
                topic=topic,
                success=False,
                answer="",
                elapsed=elapsed,
                error=f"Timeout after {timeout:.0f}s",
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            elapsed = time.perf_counter() - started
            print(f"    [{topic.key}] error: {exc}")
            return WorkerResult(
                topic=topic,
                success=False,
                answer="",
                elapsed=elapsed,
                error=str(exc),
            )


def _write_outputs(
    output_dir: Path,
    results: list[WorkerResult],
    *,
    run_id: str,
    elapsed_total: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir = output_dir / "findings"
    findings_dir.mkdir(exist_ok=True)

    for r in results:
        finding_file = findings_dir / f"{r.topic.key}.md"
        lines = [f"# {r.topic.name}", "", f"**Source:** {r.topic.source}", ""]
        if r.success:
            lines.extend([r.answer, ""])
        else:
            lines.extend([f"**ERROR:** {r.error}", ""])
        lines.append(f"*Elapsed: {r.elapsed:.1f}s*")
        finding_file.write_text("\n".join(lines), encoding="utf-8")

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    final_md = output_dir / "final-answer.md"
    sections = [
        "# Research Findings",
        "",
        f"**Analyzed:** {len(successful)}/{len(results)} topics",
        "",
    ]
    for r in successful:
        sections.extend([f"## {r.topic.name}", "", r.answer, "", "---", ""])
    if failed:
        sections.extend(["## Failed Topics", ""])
        for r in failed:
            sections.append(f"- **{r.topic.name}**: {r.error}")
    final_md.write_text("\n".join(sections), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "total_topics": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "elapsed_total": elapsed_total,
        "topics": [
            {
                "key": r.topic.key,
                "name": r.topic.name,
                "success": r.success,
                "elapsed": r.elapsed,
                "error": r.error if not r.success else None,
            }
            for r in results
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return final_md


async def run_parallel(
    topics: list[ResearchTopic],
    *,
    concurrency: int,
    timeout_per_worker: float,
    run_id: str,
    output_dir: Path,
    model: str,
    disk: str,
) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set.")

    keys = [t.key for t in topics]
    if len(keys) != len(set(keys)):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"Duplicate topic keys: {', '.join(duplicates)}")

    print(f"==> Run id           : {run_id}")
    print(f"==> Topics           : {', '.join(t.key for t in topics)}")
    print(f"==> Concurrency      : {concurrency}")
    print(f"==> Timeout / worker : {timeout_per_worker:.0f}s")
    print(f"==> Output directory : {output_dir}")
    print()

    group_client = build_group_client()
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    try:
        sandbox_client = ACASandboxClient(group_client)
        options = ACASandboxClientOptions(
            disk=disk,
            labels={"demo": "parallel-sandboxes", "run-id": run_id},
        )
        results = await asyncio.gather(
            *(
                research_topic(t, sandbox_client, options, model, semaphore, timeout_per_worker)
                for t in topics
            )
        )
    finally:
        await close_group_client(group_client)

    elapsed_total = time.perf_counter() - started
    final_md = _write_outputs(output_dir, results, run_id=run_id, elapsed_total=elapsed_total)

    successful = sum(1 for r in results if r.success)
    print()
    print(f"Run complete: {successful}/{len(results)} ok in {elapsed_total:.1f}s")
    print(f"Final answer: {final_md}")
    return 0 if successful == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ACA sandbox example: many agents, each in its own sandbox.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="Add a topic 'key|name|source|question'. Repeat for multiple topics.",
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel workers.")
    parser.add_argument(
        "--timeout-per-worker",
        type=float,
        default=180.0,
        help="Per-worker timeout in seconds.",
    )
    parser.add_argument("--model", default="gpt-5.5", help="Model name to use.")
    parser.add_argument("--disk", default="ubuntu", help="Public disk image name.")
    args = parser.parse_args(argv)

    if not args.topics:
        parser.error("at least one --topic is required.")

    try:
        topics = [parse_topic_arg(spec) for spec in args.topics]
    except ValueError as exc:
        parser.error(str(exc))

    run_id = uuid.uuid4().hex[:8]
    output_dir = Path.cwd() / ".run-output" / f"parallel-{run_id}"
    return asyncio.run(
        run_parallel(
            topics,
            concurrency=args.concurrency,
            timeout_per_worker=args.timeout_per_worker,
            run_id=run_id,
            output_dir=output_dir,
            model=args.model,
            disk=args.disk,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
