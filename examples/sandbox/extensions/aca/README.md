# Azure Container Apps sandbox scenarios

Three progressive examples that build on the `ACASandboxClient` provider:

| File | Pattern | What it shows |
| --- | --- | --- |
| [`single_sandbox.py`](./single_sandbox.py) | One agent, one sandbox | A `SandboxAgent` with shell + filesystem capabilities runs inside a single Azure Container Apps sandbox to clone and analyze a GitHub repository. |
| [`parallel_sandboxes.py`](./parallel_sandboxes.py) | Many agents, many sandboxes | Multiple worker agents run in parallel, each in its own sandbox, against different topics. Concurrency is bounded by an `asyncio.Semaphore` and results are aggregated into Markdown on the host. |
| [`nested_sandboxes.py`](./nested_sandboxes.py) | Orchestrator-in-sandbox that spawns more sandboxes | The orchestrator agent itself runs inside an Azure Container Apps sandbox. It plans the task with a local planner agent, spawns N worker sandboxes in parallel, and synthesizes their findings — all from inside its own sandbox. |

## Install

```bash
uv sync --extra aca-sandbox
```

This installs `azure-containerapps-sandbox` and `azure-identity` alongside the SDK.

## Common prerequisites

All three scenarios need:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Your OpenAI API key. |
| `ACA_SUBSCRIPTION_ID` | Azure subscription that owns the sandbox group. |
| `ACA_RESOURCE_GROUP` | Resource group containing the sandbox group. |
| `ACA_SANDBOX_GROUP` | Name of the Azure Container Apps sandbox group. |
| `ACA_REGION` | Region of the sandbox group (e.g. `eastus2`). |

Azure credentials are picked up via `azure.identity.aio.DefaultAzureCredential`, so any of the usual sources work — `az login`, managed identity, or service-principal env vars (`AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`). The identity must have data-plane permissions on the sandbox group.

## Scenario 1 — single sandbox

```bash
uv run python examples/sandbox/extensions/aca/single_sandbox.py \
    --repo https://github.com/openai/openai-agents-python \
    "What are the main agent capabilities and how do they fit together?"
```

Provisions a single sandbox, runs one agent inside it with `Shell` and `Filesystem` capabilities, and prints the answer.

## Scenario 2 — parallel sandboxes from your machine

```bash
uv run python examples/sandbox/extensions/aca/parallel_sandboxes.py \
    --topic "fastapi|FastAPI|https://github.com/fastapi/fastapi|What makes FastAPI fast?" \
    --topic "django|Django|https://github.com/django/django|What are Django's key features?" \
    --concurrency 2
```

Each `--topic` (format: `key|name|source_url|question`) becomes one worker agent in its own sandbox. The host runs the orchestration and aggregates results into `./.run-output/parallel-<run-id>/`.

## Scenario 3 — orchestrator sandbox that spawns worker sandboxes

In addition to the common prerequisites, this scenario needs a second sandbox group for the workers, plus service-principal credentials so the orchestrator can authenticate to that group from inside its own sandbox:

| Variable | Description |
| --- | --- |
| `ACA_WORKER_SUBSCRIPTION_ID` | Subscription that owns the **worker** sandbox group. |
| `ACA_WORKER_RESOURCE_GROUP` | Resource group of the worker sandbox group. |
| `ACA_WORKER_SANDBOX_GROUP` | Name of the worker sandbox group. |
| `ACA_WORKER_REGION` | Region of the worker sandbox group. |
| `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` | Service principal with data-plane permissions on the worker group. |

Run:

```bash
uv run python examples/sandbox/extensions/aca/nested_sandboxes.py \
    --task "Compare Django and FastAPI for high-throughput APIs." \
    --workers 3
```

What happens:

1. The launcher (your machine) provisions the orchestrator sandbox using the `ACA_*` group.
2. It uploads `nested_sandboxes.py` and a JSON config to `/tmp/` in the orchestrator and `pip install`s `openai-agents[aca-sandbox]` there.
3. The same file is re-executed inside the orchestrator with `--mode supervisor`. The supervisor builds a fresh `ACASandboxClient` against the worker group, plans the task with a local planner `Agent`, spawns N worker `SandboxAgent`s in parallel (each in its own worker-group sandbox), and synthesizes their findings.
4. The final answer is streamed back to the launcher's stdout. The orchestrator sandbox is deleted on exit.

### Security tradeoff

The launcher writes the service-principal secrets into `/tmp/config.json` on the orchestrator sandbox so the supervisor can authenticate. Treat the orchestrator sandbox accordingly: scope the principal narrowly (data-plane on the worker group only), use a short-lived secret if you can, and rely on `delete()` to tear the sandbox down at the end of the run. If you have an alternative such as managed identity available on the orchestrator group, prefer that instead and drop the `env` block from the launcher's config.

## See also

- [Provider reference](../../../../docs/sandbox/clients.md)
- [`ACASandboxClient` API](../../../../docs/ref/extensions/sandbox/aca_sandbox/sandbox.md)
- [`aca_runner.py`](../aca_runner.py) — the minimal one-shot runner.
