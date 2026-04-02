from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _query_terms(query: str) -> List[str]:
    parts = [part.lower() for part in re.split(r"[^A-Za-z0-9_\-]+", query) if part.strip()]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(parts))


def _iter_chunks(path: Path) -> Iterable[Dict[str, Any]]:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        yield {"source": path.name, "line": line_no, "chunk": line.strip()}


def _score_chunk(chunk: Dict[str, Any], query: str, terms: List[str]) -> Dict[str, Any] | None:
    text = str(chunk["chunk"])
    lowered = text.lower()
    matched_terms = [term for term in terms if term in lowered]

    regex_matched = False
    if query:
        try:
            regex_matched = bool(re.search(query, text, re.IGNORECASE))
        except re.error:
            regex_matched = False

    phrase_match = query.lower() in lowered if query else False
    if not matched_terms and not regex_matched and not phrase_match:
        return None

    score = 0.0
    score += float(len(matched_terms)) * 1.0
    if regex_matched:
        score += 1.25
    if phrase_match:
        score += 0.75
    if any(term in chunk["source"].lower() for term in matched_terms):
        score += 0.2

    return {
        **chunk,
        "text": text,
        "score": round(score, 4),
        "matched_terms": matched_terms,
    }


def search_local_references(
    query: str,
    candidate_paths: Iterable[Path],
    *,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    terms = _query_terms(query)
    scored: List[Dict[str, Any]] = []
    for path in candidate_paths:
        if not path.is_file():
            continue
        for chunk in _iter_chunks(path):
            matched = _score_chunk(chunk, query, terms)
            if matched is not None:
                scored.append(matched)

    scored.sort(key=lambda item: (-float(item["score"]), item["source"], int(item["line"])))
    results = []
    for rank, item in enumerate(scored[: max(1, int(top_k))], start=1):
        results.append(
            {
                "source": item["source"],
                "path": item["source"],
                "line": item["line"],
                "chunk": item["chunk"],
                "text": item["text"],
                "score": item["score"],
                "rank": rank,
                "matched_terms": item["matched_terms"],
            }
        )
    return results
