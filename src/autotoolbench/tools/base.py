from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..agent.budget import estimate_token_count
from ..agent.schema import ToolArgumentConstraint


class ToolResult(BaseModel):
    ok: bool
    output: Any = None
    error: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    risk_level: str = "low"
    output_type: str = "unknown"
    argument_constraints: Dict[str, ToolArgumentConstraint] = Field(default_factory=dict)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    read_only: bool = True
    mutating: bool = False
    allowed_memory_types: Dict[str, List[str]] = Field(default_factory=dict)

    def estimate_cost(self, inp: Dict[str, Any]) -> Dict[str, float]:
        return {
            "time": 0.1,
            "tokens": float(estimate_token_count(inp)),
            "calls": 1,
            "steps": 1,
        }

    def run(self, inp: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "output_type": self.output_type,
            "read_only": self.read_only,
            "mutating": self.mutating,
            "input_schema": self.input_schema,
            "argument_constraints": {key: constraint.model_dump() for key, constraint in self.argument_constraints.items()},
            "allowed_memory_types": self.allowed_memory_types,
            "examples": self.examples,
        }
