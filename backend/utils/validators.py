from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def validate_upload(filename: str, allowed_extensions: set[str]) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Allowed: {', '.join(sorted(allowed_extensions))}",
        )
