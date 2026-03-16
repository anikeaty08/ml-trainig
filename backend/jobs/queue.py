from __future__ import annotations

from queue import Queue
from typing import Callable

from .worker import Worker


class JobQueue:
    def __init__(self, processor: Callable[[str, str], None]) -> None:
        self.queue: Queue[tuple[str, str]] = Queue()
        self.worker = Worker(self.queue, processor)
        self.worker.start()

    def enqueue(self, job_id: str, filepath: str) -> None:
        self.queue.put((job_id, filepath))
