from __future__ import annotations

import re
from typing import Any, Dict, List

from ..agent.schema import ToolArgumentConstraint
from ..utils.paths import app_log_path
from .base import Tool, ToolResult
from .registry import register


class LogSearchTool(Tool):
    name: str = "log_search"
    description: str = "Search app.log for keyword or regex"
    risk_level: str = "low"
    output_type: str = "log_lines"
    read_only: bool = True
    mutating: bool = False
    input_schema: dict = {"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}
    argument_constraints: Dict[str, ToolArgumentConstraint] = {
        "pattern": ToolArgumentConstraint(
            type="string",
            required=True,
            non_empty=True,
            description="Regex or substring pattern to search in logs.",
        )
    }
    allowed_memory_types: Dict[str, List[str]] = {"pattern": ["text", "file_text"]}
    examples: List[Dict[str, Any]] = [{"pattern": "request_id=REQ-404"}]

    def run(self, inp: Dict[str, Any]) -> ToolResult:
        pat = inp.get("pattern","")
        try:
            lines = app_log_path().read_text(encoding="utf-8").splitlines()
            res: List[Dict[str, Any]] = []
            try:
                regex = re.compile(pat)
                for idx, line in enumerate(lines, start=1):
                    if regex.search(line):
                        res.append({"line": idx, "text": line.strip()})
            except re.error:
                # treat as substring
                for idx, line in enumerate(lines, start=1):
                    if pat in line:
                        res.append({"line": idx, "text": line.strip()})
            return ToolResult(ok=True, output=res)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

register(LogSearchTool())
