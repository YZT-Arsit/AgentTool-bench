from __future__ import annotations

from typing import Any, Dict, List

from ..agent.schema import ToolArgumentConstraint
from .base import Tool, ToolResult
from .registry import register

SAFE_BUILTINS = {
    'abs': abs,
    'min': min,
    'max': max,
    'sum': sum,
    'len': len,
    'range': range,
    'print': print,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'dict': dict,
    'list': list,
    'set': set,
}

class PythonExecTool(Tool):
    name: str = "python_exec"
    description: str = "Execute simple Python code in sandbox"
    risk_level: str = "high"
    output_type: str = "json"
    read_only: bool = False
    mutating: bool = True
    input_schema: dict = {"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}
    argument_constraints: Dict[str, ToolArgumentConstraint] = {
        "code": ToolArgumentConstraint(
            type="string",
            required=True,
            non_empty=True,
            description="Python source code to execute in the sandbox.",
        )
    }
    allowed_memory_types: Dict[str, List[str]] = {"code": ["text", "file_text"]}
    examples: List[Dict[str, Any]] = [{"code": "a = 1 + 2"}]

    def run(self, inp: Dict[str, Any]) -> ToolResult:
        code = inp.get("code","")
        # restricted exec environment
        loc = {}
        glb = {"__builtins__": SAFE_BUILTINS}
        try:
            exec(code, glb, loc)
            return ToolResult(ok=True, output=loc)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

register(PythonExecTool())
