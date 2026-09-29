# Codoctopus 🐙

A provider-neutral agent orchestration framework: give it a goal, and it decomposes the
goal into a verified task DAG, then runs each step with agents that can actually act —
read and write files, run shell commands and tests, and make HTTP requests.

- **Provider-neutral** — Anthropic, OpenAI, Gemini, Ollama, or any OpenAI-compatible
  gateway (LM Studio, vLLM, ...). Switching models is a config change, not a code change.
- **Structured protocol** — steps pass typed objects (Pydantic), not prose to be parsed.
- **Agents with tools** — verification means the code was really run, not re-read by an LLM.
- **Pluggable domains** — `coding` and `research` ship built in.
- **Zero infrastructure by default** — plans run in-process on asyncio; optionally hand
  them to [Coworkify](https://github.com/ccoliu/coworkify) for retries and persistence.

See [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) for the full design.

## Install

Requires Python 3.11+.

```bash
pip install -e ".[all]"          # core + every provider SDK
pip install -e ".[anthropic]"    # or just the providers you need: anthropic / openai / gemini
pip install -e ".[server]"       # adds FastAPI + uvicorn for the GUI backend
pip install -e ".[dev,all]"      # for development (pytest, ruff)
```

## CLI

```bash
# Plan and execute a goal
codoctopus run "Write a CLI that converts CSV to JSON, with tests" --domain coding

# Only show the plan
codoctopus run "Survey Python async HTTP clients" --domain research --dry-run

# Pick models per role (provider:model)
codoctopus run "..." --model anthropic:claude-opus-5 --worker-model ollama:llama3.1
```

| Option | Description |
|---|---|
| `--domain` | Domain pack to plan within (`coding`, `research`) |
| `--model` | `provider:model` used for planning |
| `--worker-model` | `provider:model` used to run each step |
| `--workspace` | Directory the agent tools may read and write in |
| `--executor` | `local` (default) or `coworkify` |
| `--dry-run` | Plan only, don't execute |
| `--json` | Machine-readable output |

## GUI

The GUI is two processes: a FastAPI backend (`codoctopus serve`) and a Vite dev server
(`webapp/`) that proxies `/api/*` to it. Both need to be running.

```bash
# Terminal 1 — backend (needs the `server` extra)
codoctopus serve --port 8420

# Terminal 2 — frontend
cd webapp
npm install   # first time only
npm run dev   # http://localhost:5174
```

If the frontend logs `ECONNREFUSED 127.0.0.1:8420` for `/api/*` requests, the backend
isn't running (or died) — start it in Terminal 1 and refresh. Provider API keys are
entered in the GUI's own Settings page (saved to the browser's `localStorage`, sent
only with the runs you start) — no `.env` needed on the backend for that.

## Python SDK

```python
import asyncio
from pathlib import Path

from codoctopus.domains import get_domain
from codoctopus.llm import get_provider
from codoctopus.planning import make_plan
from codoctopus.runtime import LocalExecutor
from codoctopus.tools.builtin import BUILTIN_TOOLS, build_tool_registry


async def main() -> None:
    plan = await make_plan(
        get_provider("anthropic:claude-opus-5"),
        "Write a function that parses ISO dates, with tests",
        domain=get_domain("coding"),
        available_tools=list(BUILTIN_TOOLS),
    )
    workspace = Path(".codoctopus/workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    executor = LocalExecutor(
        get_provider("anthropic:claude-sonnet-5"),
        workspace=workspace,
        tools=build_tool_registry(workspace),
    )
    result = await executor.run(plan)
    print(result.status, result.step_results)


asyncio.run(main())
```

## Providers

Models are referenced as `provider:model`; a bare provider name uses its default model.

| Provider | Example | Credentials |
|---|---|---|
| `anthropic` | `anthropic:claude-opus-5` | `ANTHROPIC_API_KEY` |
| `openai` | `openai:gpt-4.1` | `OPENAI_API_KEY` |
| `gemini` | `gemini:gemini-2.0-flash` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `ollama` | `ollama:llama3.1` | none (local) |
| `gateway` | `gateway:<model>` | OpenAI-compatible server; requires `base_url` |

Custom providers can be added with `codoctopus.llm.registry.register_provider`.

## Configuration

All settings come from environment variables:

| Variable | Default | Description |
|---|---|---|
| `CODOCTOPUS_PLANNER_MODEL` | `anthropic:claude-opus-5` | Model used for planning |
| `CODOCTOPUS_WORKER_MODEL` | `anthropic:claude-sonnet-5` | Model used for each step |
| `CODOCTOPUS_EFFORT` | `high` | Reasoning effort |
| `CODOCTOPUS_MAX_TOKENS` | `16000` | Max output tokens per call |
| `CODOCTOPUS_WORKSPACE` | `.codoctopus/workspace` | Agents may only read/write below this directory |
| `CODOCTOPUS_MAX_TOOL_TURNS` | `25` | Tool-use iteration cap per agent step |
| `CODOCTOPUS_COWORKIFY_URL` | *(unset)* | Coworkify base URL; unset means run locally |
| `CODOCTOPUS_COWORKIFY_TOKEN` | *(unset)* | Coworkify auth token |

## Project Layout

```
codoctopus/
├── cli.py          # `codoctopus run` / `codoctopus serve`
├── config.py       # Settings from environment variables
├── planning/       # goal → Plan (DAG) + validation
├── agents/         # Agent tool-use loop
├── domains/        # Domain packs: coding, research
├── tools/          # filesystem, shell, testing, http
├── llm/            # Provider interface + anthropic / openai / gemini / ollama
├── runtime/        # LocalExecutor (asyncio), CoworkifyExecutor
└── server/         # FastAPI backend for the GUI (HTTP + WebSocket)
webapp/             # React + Vite frontend
tests/              # pytest suite (providers use scripted fakes; no API keys needed)
docs/               # Architecture docs
```

## Development

```bash
pip install -e ".[dev,all]"
ruff check codoctopus tests
pytest -q
```

## Legacy v1

`backend/`, `frontend/`, `run.py`, `gunicorn.conf.py`, `render.yaml`, `requirements.txt`
and `.env.example` belong to the old v1 Flask app (Gemini-based code analysis, generation
and plagiarism checking, backed by MongoDB). They are superseded by the `codoctopus`
package above and kept only for reference.

## License

MIT
