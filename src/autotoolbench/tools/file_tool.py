from __future__ import annotations

from typing import Any, Dict, List

from ..agent.schema import ToolArgumentConstraint
from ..utils.paths import data_dir, resolve_data_path
from .base import Tool, ToolResult
from .registry import register


class FileTool(Tool):
    name: str = "file_read"
    description: str = "Read file under data directory"
    risk_level: str = "low"
    output_type: str = "file_text"
    read_only: bool = True
    mutating: bool = False
    input_schema: dict = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    argument_constraints: Dict[str, ToolArgumentConstraint] = {
        "path": ToolArgumentConstraint(
            type="string",
            required=True,
            non_empty=True,
            description="Relative path under the data directory.",
        )
    }
    examples: List[Dict[str, Any]] = [{"path": "incident_brief.txt"}]

    def run(self, inp: Dict[str, Any]) -> ToolResult:
        p = inp.get("path")
        if not isinstance(p, str):
            return ToolResult(ok=False, error="invalid path")
        try:
            target = resolve_data_path(p)
        except ValueError:
            return ToolResult(ok=False, error="Path escape")
        if not target.is_file():
            return ToolResult(ok=False, error="Not found")
        try:
            data = target.read_text(encoding="utf-8")
            return ToolResult(ok=True, output=data)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

class FileWriteTool(FileTool):
    name: str = "file_write"
    description: str = "Write file under data directory"
    risk_level: str = "medium"
    output_type: str = "text"
    read_only: bool = False
    mutating: bool = True
    input_schema: dict = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path","content"]}
    argument_constraints: Dict[str, ToolArgumentConstraint] = {
        "path": ToolArgumentConstraint(
            type="string",
            required=True,
            non_empty=True,
            description="Relative output path under the data directory.",
        ),
        "content": ToolArgumentConstraint(
            type="string",
            required=True,
            description="String content to write. Structured values will be serialized before writing.",
        ),
    }
    allowed_memory_types: Dict[str, List[str]] = {
        "content": ["text", "file_text", "json", "rows", "log_lines", "sql_result", "retrieval_results", "unknown"]
    }
    examples: List[Dict[str, Any]] = [{"path": "users_report.json", "content": "[{\"name\": \"Alice\"}]"}]

    def run(self, inp: Dict[str, Any]) -> ToolResult:
        p = inp.get("path")
        if not isinstance(p, str):
            return ToolResult(ok=False, error="invalid path")
        try:
            target = resolve_data_path(p)
        except ValueError:
            return ToolResult(ok=False, error="Path escape")
        try:
            data_dir().mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(inp.get("content",""), encoding="utf-8")
            return ToolResult(ok=True, output="written")
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

register(FileTool())
register(FileWriteTool())
