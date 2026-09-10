# ---------------------------------------------------
# Codoctopus — GUI backend request/response models
# ---------------------------------------------------

from __future__ import annotations

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    goal: str = Field(min_length=1)
    domain: str | None = None
    model: str | None = Field(default=None, description="provider:model for planning")
    worker_model: str | None = Field(default=None, description="provider:model for each step")
    executor: str = Field(default="local", pattern="^(local|coworkify)$")
    workspace: str | None = None

    # Credentials the browser holds (its Settings page, backed by localStorage
    # — never persisted server-side) and attaches per run instead of the
    # server process needing them in its own environment. Separate for
    # planner vs worker since they may be different providers.
    planner_api_key: str | None = None
    planner_base_url: str | None = None
    worker_api_key: str | None = None
    worker_base_url: str | None = None


class ModelsRequest(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
