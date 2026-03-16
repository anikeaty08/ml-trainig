from __future__ import annotations

import threading
from queue import Queue
from typing import Callable


class Worker:
    def __init__(self, queue: Queue[tuple[str, str]], processor: Callable[[str, str], None]) -> None:
        self.queue = queue
        self.processor = processor
        self.thread = threading.Thread(target=self._run, daemon=True, name="ml-pipeline-worker")

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def _run(self) -> None:
        while True:
            job_id, filepath = self.queue.get()
            try:
                self.processor(job_id, filepath)
            finally:
                self.queue.task_done()
