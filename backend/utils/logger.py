from __future__ import annotations

import logging

from ..config import LOGS_DIR, ensure_directories


def get_job_logger(job_id: str) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(f"job-{job_id}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    handler = logging.FileHandler(LOGS_DIR / f"{job_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
