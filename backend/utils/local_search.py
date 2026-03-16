from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import BACKEND_DIR, ROOT_DIR


SEARCH_DIRS = [
    ROOT_DIR / "docs",
    ROOT_DIR / "README.md",
    BACKEND_DIR / "data" / "reports",
]


def search_local_knowledge(query: str, limit: int = 10) -> list[dict[str, Any]]:
    terms = [term.lower() for term in query.split() if term.strip()]
    if not terms:
        return []

    hits: list[dict[str, Any]] = []
    for path in SEARCH_DIRS:
        if not path.exists():
            continue
        files = [path] if path.is_file() else list(path.rglob("*"))
        for file_path in files:
            if not file_path.is_file():
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            content_lower = content.lower()
            if not all(term in content_lower for term in terms):
                continue
            first_term = terms[0]
            index = content_lower.find(first_term)
            start = max(index - 120, 0)
            end = min(index + 200, len(content))
            snippet = content[start:end].replace("\n", " ").strip()
            hits.append({"path": str(file_path), "snippet": snippet})
            if len(hits) >= limit:
                return hits
    return hits
