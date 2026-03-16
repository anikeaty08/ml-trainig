from __future__ import annotations

from typing import Any


class JobStateManager:
    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}

    def bootstrap_from_database(self, jobs: list[dict[str, Any]]) -> None:
        for job in jobs:
            self._state[job["id"]] = {
                "job_id": job["id"],
                "stage": job["stage"],
                "progress": job["progress"],
                "message": job["message"],
                "status": job["status"],
            }

    def set_state(self, job_id: str, payload: dict[str, Any]) -> None:
        self._state[job_id] = payload

    def get_state(self, job_id: str) -> dict[str, Any] | None:
        return self._state.get(job_id)
