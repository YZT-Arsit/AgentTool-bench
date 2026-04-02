from __future__ import annotations

from typing import Any, Dict, List

from ..agent.schema import ToolArgumentConstraint
from .base import Tool, ToolResult
from .registry import register


class NoopTool(Tool):
    name: str = "noop"
    description: str = "No operation tool used as default"
    risk_level: str = "low"
    output_type: str = "text"
    read_only: bool = True
    mutating: bool = False
    input_schema: dict = {"type":"object"}
    argument_constraints: Dict[str, ToolArgumentConstraint] = {}
    examples: List[Dict[str, Any]] = [{}]

    def run(self, inp: Dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, output="noop")

register(NoopTool())
