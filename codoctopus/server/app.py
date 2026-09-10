# ---------------------------------------------------
# Codoctopus — GUI backend
#
# Thin HTTP/WebSocket wrapper around the same public API the CLI uses
# (codoctopus.planning.make_plan + codoctopus.runtime executors) so a browser
# can submit a goal and watch it plan and run instead of reading stdout.
# No auth, no persistence beyond the process's lifetime — a local dev tool
# for one user watching their own runs, same trust boundary as the CLI.
# ---------------------------------------------------

from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from codoctopus.config import get_settings
from codoctopus.domains import available_domains
from codoctopus.llm import ProviderError, available_providers
from codoctopus.llm import list_models as llm_list_models
from codoctopus.server.runs import DONE, RunManager, ScheduleError
from codoctopus.server.schemas import ModelsRequest, RunCreate, ScheduleCreate


def create_app() -> FastAPI:
    app = FastAPI(title="Codoctopus")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = RunManager(get_settings())

    @app.get("/api/domains")
    async def list_domains() -> dict[str, list[str]]:
        return {"domains": available_domains()}

    @app.get("/api/providers")
    async def list_providers() -> dict[str, list[str]]:
        return {"providers": available_providers()}

    @app.post("/api/providers/{name}/models")
    async def list_provider_models(name: str, body: ModelsRequest) -> dict[str, list[str]]:
        kwargs = {}
        if body.api_key:
            kwargs["api_key"] = body.api_key
        if body.base_url:
            kwargs["base_url"] = body.base_url
        try:
            models = await llm_list_models(name, **kwargs)
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"models": models}

    @app.post("/api/runs", status_code=201)
    async def create_run(body: RunCreate) -> dict:
        # Must run on the event loop thread, not FastAPI's sync threadpool:
        # RunManager.create() calls asyncio.create_task() to start the run in
        # the background, which needs a running loop in the calling thread.
        run = manager.create(
            body.goal,
            domain=body.domain,
            model=body.model,
            worker_model=body.worker_model,
            executor=body.executor,
            workspace=body.workspace,
            planner_api_key=body.planner_api_key,
            planner_base_url=body.planner_base_url,
            worker_api_key=body.worker_api_key,
            worker_base_url=body.worker_base_url,
        )
        return run.to_detail()

    @app.get("/api/runs")
    async def list_runs() -> dict[str, list[dict]]:
        return {"runs": [r.to_summary() for r in manager.list()]}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.to_detail()

    @app.post("/api/runs/{run_id}/schedule", status_code=201)
    async def create_run_schedule(run_id: str, body: ScheduleCreate) -> dict:
        if manager.get(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            return await manager.create_schedule(run_id, cron_expression=body.cron_expression, name=body.name)
        except ScheduleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/api/runs/{run_id}/stream")
    async def stream_run(websocket: WebSocket, run_id: str) -> None:
        run = manager.get(run_id)
        if run is None:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        await websocket.send_json({"event": "snapshot", "data": run.to_detail()})
        for entry in manager.history(run_id):
            await websocket.send_json(entry)

        if run.status in ("success", "failed"):
            await websocket.close()
            return

        queue = await manager.subscribe(run_id)
        try:
            while True:
                entry = await queue.get()
                if entry is DONE:
                    break
                await websocket.send_json(entry)
        except WebSocketDisconnect:
            pass
        finally:
            manager.unsubscribe(run_id, queue)
            try:
                await websocket.close()
            except RuntimeError:
                pass  # already closed (client disconnected)

    return app
