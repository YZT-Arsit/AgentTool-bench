from __future__ import annotations

import sqlite3
from typing import Any, Dict, List

from ..agent.schema import ToolArgumentConstraint
from ..utils.paths import db_path
from .base import Tool, ToolResult
from .registry import register


class SQLiteTool(Tool):
    name: str = "sql_query"
    description: str = "Execute read-only SQL on local sample.db"
    risk_level: str = "low"
    output_type: str = "sql_result"
    read_only: bool = True
    mutating: bool = False
    input_schema: dict = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    argument_constraints: Dict[str, ToolArgumentConstraint] = {
        "query": ToolArgumentConstraint(
            type="string",
            required=True,
            non_empty=True,
            description="Read-only SQL query.",
        )
    }
    allowed_memory_types: Dict[str, List[str]] = {"query": ["text", "file_text"]}
    examples: List[Dict[str, Any]] = [{"query": "SELECT name FROM users ORDER BY id"}]

    def run(self, inp: Dict[str, Any]) -> ToolResult:
        q = inp.get("query", "")
        try:
            conn = sqlite3.connect(db_path())
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(q)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return ToolResult(ok=True, output=rows)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

register(SQLiteTool())
