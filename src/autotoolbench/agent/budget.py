from __future__ import annotations

import json
import time
from typing import Any, Dict


def estimate_token_count(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except TypeError:
            text = str(payload)
    return max(1, (len(text) + 3) // 4)


class BudgetController:
    def __init__(self, max_calls: int = 10, max_steps: int = 10, max_time: float = 60, max_tokens: int = 1000):
        self.max_calls = max_calls
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens

    def initial(self) -> Dict[str, Any]:
        return {
            "calls": 0,
            "steps": 0,
            "time": 0.0,
            "tokens": 0,
            "llm_tokens": 0,
            "tool_tokens": 0,
            "estimated_tool_time": 0.0,
            "started_at": time.perf_counter(),
        }

    def limits(self) -> Dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "max_steps": self.max_steps,
            "max_time": self.max_time,
            "max_tokens": self.max_tokens,
        }

    def check(self, budget: Dict[str, Any]) -> bool:
        return (
            budget.get("calls", 0) < self.max_calls
            and budget.get("steps", 0) < self.max_steps
            and budget.get("time", 0.0) <= self.max_time
            and budget.get("tokens", 0) <= self.max_tokens
        )

    def can_afford_tool(self, budget: Dict[str, Any], estimate: Dict[str, Any]) -> bool:
        projected_calls = budget.get("calls", 0) + int(estimate.get("calls", 1))
        projected_steps = budget.get("steps", 0) + int(estimate.get("steps", 1))
        projected_time = budget.get("time", 0.0) + float(estimate.get("time", 0.0))
        projected_tokens = budget.get("tokens", 0) + int(estimate.get("tokens", 0))
        return (
            projected_calls <= self.max_calls
            and projected_steps <= self.max_steps
            and projected_time <= self.max_time
            and projected_tokens <= self.max_tokens
        )

    def consume_llm_tokens(self, budget: Dict[str, Any], estimated_tokens: int) -> None:
        amount = max(0, int(estimated_tokens))
        budget["tokens"] = budget.get("tokens", 0) + amount
        budget["llm_tokens"] = budget.get("llm_tokens", 0) + amount

    def reserve_tool_estimate(self, budget: Dict[str, Any], estimate: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.normalize_cost(estimate)
        budget["tool_tokens"] = budget.get("tool_tokens", 0) + int(normalized["tokens"])
        budget["tokens"] = budget.get("tokens", 0) + int(normalized["tokens"])
        budget["estimated_tool_time"] = budget.get("estimated_tool_time", 0.0) + float(normalized["time"])
        return normalized

    def record_tool_call(self, budget: Dict[str, Any], runtime_seconds: float) -> None:
        budget["time"] = budget.get("time", 0.0) + float(runtime_seconds)
        budget["calls"] = budget.get("calls", 0) + 1
        budget["steps"] = budget.get("steps", 0) + 1

    def snapshot(self, budget: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "calls": int(budget.get("calls", 0)),
            "steps": int(budget.get("steps", 0)),
            "time": float(budget.get("time", 0.0)),
            "tokens": int(budget.get("tokens", 0)),
            "llm_tokens": int(budget.get("llm_tokens", 0)),
            "tool_tokens": int(budget.get("tool_tokens", 0)),
            "estimated_tool_time": float(budget.get("estimated_tool_time", 0.0)),
        }

    def normalize_cost(self, estimate: Dict[str, Any] | None) -> Dict[str, Any]:
        estimate = estimate or {}
        return {
            "calls": int(estimate.get("calls", 1)),
            "steps": int(estimate.get("steps", 1)),
            "time": float(estimate.get("time", 0.0)),
            "tokens": int(estimate.get("tokens", 0)),
        }

    @classmethod
    def from_preset(cls, preset: str) -> "BudgetController":
        if preset == "tight":
            return cls(max_calls=6, max_steps=8, max_time=10, max_tokens=4000)
        if preset == "loose":
            return cls(max_calls=16, max_steps=20, max_time=30, max_tokens=12000)
        return cls()
