# ---------------------------------------------------
# Codoctopus — Settings
#
# Everything tunable lives here and comes from the environment, so switching
# models or providers never means editing code (v1's failure: three API keys
# read from Keys/openai_key.txt and a model name pinned in Config.yaml).
# ---------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    """
    Runtime configuration.

    `planner_model` is separated from `worker_model` on purpose: planning is
    low-volume and benefits from the strongest model, while the many worker
    steps are where cost accumulates.
    """

    planner_model: str = field(default_factory=lambda: _env("CODOCTOPUS_PLANNER_MODEL", "anthropic:claude-opus-5"))
    worker_model: str = field(default_factory=lambda: _env("CODOCTOPUS_WORKER_MODEL", "anthropic:claude-sonnet-5"))
    effort: str = field(default_factory=lambda: _env("CODOCTOPUS_EFFORT", "high"))
    max_tokens: int = field(default_factory=lambda: _env_int("CODOCTOPUS_MAX_TOKENS", 16000))

    #: Agents may only read and write below this directory.
    workspace: Path = field(default_factory=lambda: Path(_env("CODOCTOPUS_WORKSPACE", ".codoctopus/workspace")))

    #: SQLite file the GUI server keeps its run history in, so it survives a restart.
    runs_db: Path = field(default_factory=lambda: Path(_env("CODOCTOPUS_RUNS_DB", ".codoctopus/runs.db")))

    #: Ceiling on tool-use iterations in one agent step, so a loop cannot run away.
    max_tool_turns: int = field(default_factory=lambda: _env_int("CODOCTOPUS_MAX_TOOL_TURNS", 25))

    #: Base URL of a Coworkify instance; unset means run locally.
    coworkify_url: str = field(default_factory=lambda: _env("CODOCTOPUS_COWORKIFY_URL", ""))
    coworkify_token: str = field(default_factory=lambda: _env("CODOCTOPUS_COWORKIFY_TOKEN", ""))

    @property
    def uses_coworkify(self) -> bool:
        return bool(self.coworkify_url)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings, read from the environment on first use."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings; used by tests that patch the environment."""
    global _settings
    _settings = None
