# ---------------------------------------------------
# Codoctopus — Run history persistence for the GUI backend
#
# RunManager keeps everything in memory for speed and writes through to this
# SQLite file on every change, so a restarted server can load the full
# history back. Each run is stored as its to_detail() JSON plus its ordered
# event log (what a reconnecting WebSocket replays).
# ---------------------------------------------------

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class RunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # RunManager is built on one thread and used from the event loop's
        # (e.g. TestClient's portal thread); all access is still serialized.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)

    def save_run(self, detail: dict[str, Any]) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO runs (id, created_at, data) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (detail["id"], detail["created_at"], json.dumps(detail)),
            )

    def append_event(self, run_id: str, seq: int, entry: dict[str, Any]) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO run_events (run_id, seq, data) VALUES (?, ?, ?)",
                (run_id, seq, json.dumps(entry)),
            )

    def load(self) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Every stored run's detail dict with its events, oldest run first."""
        events: dict[str, list[dict[str, Any]]] = {}
        for run_id, data in self._db.execute("SELECT run_id, data FROM run_events ORDER BY run_id, seq"):
            events.setdefault(run_id, []).append(json.loads(data))
        return [
            (json.loads(data), events.get(run_id, []))
            for run_id, data in self._db.execute("SELECT id, data FROM runs ORDER BY created_at")
        ]
