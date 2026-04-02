from __future__ import annotations

import re
from typing import Any, Dict, List

from ..agent.schema import ToolArgumentConstraint
from ..retrieval import search_local_references
from ..utils.paths import data_dir
from .base import Tool, ToolResult
from .registry import register


class DocSearchTool(Tool):
    name: str = "doc_search"
    description: str = "Retrieve ranked evidence chunks from local reference documents under the data directory"
    risk_level: str = "low"
    output_type: str = "retrieval_results"
    read_only: bool = True
    mutating: bool = False
    input_schema: dict = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "query": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "top_k": {"type": "integer"},
        },
        "required": [],
    }
    argument_constraints: Dict[str, ToolArgumentConstraint] = {
        "pattern": ToolArgumentConstraint(
            type="string",
            required=False,
            non_empty=True,
            description="Regex or substring pattern to search in local documents.",
        ),
        "query": ToolArgumentConstraint(
            type="string",
            required=False,
            non_empty=True,
            description="Preferred retrieval query text.",
        ),
        "files": ToolArgumentConstraint(
            type="array",
            required=False,
            min_items=1,
            description="Optional list of file names under the data directory.",
        ),
        "top_k": ToolArgumentConstraint(
            type="integer",
            required=False,
            description="Maximum number of ranked chunks to return.",
        ),
    }
    allowed_memory_types: Dict[str, List[str]] = {"pattern": ["text", "file_text"], "query": ["text", "file_text"], "files": ["json"]}
    examples: List[Dict[str, Any]] = [{"query": "INV-9 ownership evidence", "files": ["invoice_casebook.txt", "incident_brief.txt"], "top_k": 3}]

    def run(self, inp: Dict[str, Any]) -> ToolResult:
        pattern = inp.get("query") or inp.get("pattern", "")
        if not isinstance(pattern, str) or not pattern.strip():
            return ToolResult(ok=False, error="missing retrieval query")
        files = inp.get("files") or []
        top_k = int(inp.get("top_k", 5) or 5)
        candidate_paths = []
        if files:
            candidate_paths = [data_dir() / name for name in files if isinstance(name, str)]
        else:
            candidate_paths = sorted(path for path in data_dir().iterdir() if path.is_file() and path.suffix in {".txt", ".md"})

        try:
            results = search_local_references(pattern, candidate_paths, top_k=top_k)
            matched_terms = sorted(
                {
                    term
                    for item in results
                    for term in item.get("matched_terms", [])
                }
            )
            return ToolResult(
                ok=True,
                output=results,
                metadata={
                    "query": pattern,
                    "top_k": top_k,
                    "candidate_source_count": len(candidate_paths),
                    "returned_count": len(results),
                    "matched_terms": matched_terms,
                },
            )
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))


register(DocSearchTool())
