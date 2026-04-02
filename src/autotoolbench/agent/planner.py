from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..env.tasks import Task
from ..tools.registry import all_tools
from .budget import estimate_token_count
from .json_utils import bounded_retry, validate_plan
from .schema import PlanPayload


class Planner:
    def __init__(self, llm):
        self.llm = llm
        self.last_trace: Dict[str, Any] = {}

    def plan(
        self,
        task_or_instruction: Task | str,
        budget_mode: str = "default",
        scenario: str = "default",
        replan_count: int = 0,
    ) -> List[Dict[str, Any]]:
        if isinstance(task_or_instruction, Task) or (
            hasattr(task_or_instruction, "instruction")
            and hasattr(task_or_instruction, "task_id")
            and hasattr(task_or_instruction, "task_type")
            and hasattr(task_or_instruction, "plan_hints")
        ):
            instruction = task_or_instruction.instruction
            task_id = task_or_instruction.task_id
            task_type = task_or_instruction.task_type
            hints = task_or_instruction.plan_hints
            hint_steps = hints.get(budget_mode) or hints.get("default") or self._heuristic_plan(instruction)
        else:
            instruction = task_or_instruction
            task_id = "ad-hoc"
            task_type = "default"
            hint_steps = self._heuristic_plan(instruction)

        context = {
            "task_id": task_id,
            "task_instruction": instruction,
            "task_type": task_type,
            "budget_mode": budget_mode,
            "scenario": scenario,
            "replan_count": replan_count,
            "memory_reference_syntax": "$memory:key",
            "memory_slot_schema": {
                "key": "slot name",
                "value": "stored value",
                "value_type": "text|rows|json|log_lines|file_text|sql_result|retrieval_results|unknown",
                "source_step_id": "producing step",
                "source_tool": "producing tool",
            },
            "tool_schema": self._tool_schema(),
            "plan_hints": self._normalize_hints(hint_steps),
        }
        if any(step.get("branch_group") or step.get("merge_into") for step in hint_steps):
            context["branch_schema"] = {
                "branch_group": "shared group id for independent sibling steps",
                "branch_id": "branch-local id inside the group",
                "independent": "true when the step can run without waiting on sibling branches",
                "merge_into": "set on the merge step that consumes a branch group",
                "merge_requirements": ["optional list of branch ids expected before merge"],
            }
        messages = self._messages(context)
        result = bounded_retry(lambda: self.llm.generate(messages, schema_name="plan"), validate_plan, max_retries=2)
        llm_meta = getattr(self.llm, "last_generation_metadata", {})
        if result["ok"]:
            payload = PlanPayload.model_validate(result["payload"])
            steps = [self._step_to_legacy(step.model_dump()) for step in payload.steps]
            self.last_trace = {
                "llm_raw_text": result["raw"],
                "parsed_json": result["payload"],
                "validation_errors": [],
                "fallback_used": False,
                "fallback_reason": "",
                "parse_failures": result["parse_failures"],
                "attempts": result["attempts"],
                "estimated_tokens": self._estimate_trace_tokens(messages, result["attempts"]),
                "injection_metadata": llm_meta.get("injection"),
            }
            return steps

        self.last_trace = {
            "llm_raw_text": result["raw"],
            "parsed_json": result["payload"],
            "validation_errors": result["validation_errors"],
            "fallback_used": True,
            "fallback_reason": "plan_json_invalid",
            "parse_failures": result["parse_failures"],
            "attempts": result["attempts"],
            "estimated_tokens": self._estimate_trace_tokens(messages, result["attempts"]),
            "injection_metadata": llm_meta.get("injection"),
        }
        return [dict(step) for step in hint_steps]

    def _messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        supports_branches = "branch_schema" in context
        schema_suffix = (
            ',"branch_group":"optional_group","branch_id":"optional_branch","independent":false,"merge_into":"optional_group","merge_requirements":["A"]'
            if supports_branches
            else ""
        )
        branch_guidance = (
            " Only use branch_group/branch_id for truly independent sibling steps, and use merge_into on the merge step."
            if supports_branches
            else ""
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are the planner for AutoToolBench. Return JSON only. "
                    'Schema: {"steps":[{"step_id":"S1","subgoal":"...","tool":"sql_query","args_hint":{},'
                    f'"success_criteria":["..."],"optional":false,"save_as":"slot"{schema_suffix}]}}. '
                    f"Use save_as plus $memory:key when later steps need earlier outputs.{branch_guidance}"
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

    def _normalize_hints(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for idx, step in enumerate(steps, start=1):
            normalized.append(
                {
                    "step_id": step.get("step_id", f"S{idx}"),
                    "subgoal": step.get("subgoal", f"step {idx}"),
                    "tool": step.get("tool", "noop"),
                    "args": step.get("args", step.get("args_hint", {})) or {},
                    "success_criteria": step.get("success_criteria", [f"complete step {idx}"]),
                    "optional": bool(step.get("optional", False)),
                    "save_as": step.get("save_as"),
                    "branch_group": step.get("branch_group"),
                    "branch_id": step.get("branch_id"),
                    "independent": bool(step.get("independent", False)),
                    "merge_into": step.get("merge_into"),
                    "merge_requirements": step.get("merge_requirements", []),
                }
            )
        return normalized

    def _tool_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {}
        for name, tool in all_tools().items():
            if name not in {"file_read", "file_write", "log_search", "sql_query", "doc_search"}:
                continue
            schema[name] = {
                "risk_level": tool.risk_level,
                "output_type": tool.output_type,
                "read_only": tool.read_only,
                "mutating": tool.mutating,
                "required_args": list(tool.input_schema.get("required", [])),
                "argument_constraints": {
                    key: {
                        "type": constraint.type,
                        "required": constraint.required,
                        "non_empty": constraint.non_empty,
                        "min_items": constraint.min_items,
                        "pattern": constraint.pattern,
                        "enum": list(constraint.enum),
                    }
                    for key, constraint in tool.argument_constraints.items()
                },
                "allowed_memory_types": tool.allowed_memory_types,
            }
        return schema

    def _step_to_legacy(self, step: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step_id": step["step_id"],
            "subgoal": step["subgoal"],
            "tool": step["tool"],
            "args": step.get("args_hint", {}),
            "success_criteria": step.get("success_criteria", []),
            "optional": step.get("optional", False),
            "save_as": step.get("save_as"),
            "branch_group": step.get("branch_group"),
            "branch_id": step.get("branch_id"),
            "independent": step.get("independent", False),
            "merge_into": step.get("merge_into"),
            "merge_requirements": step.get("merge_requirements", []),
        }

    def _heuristic_plan(self, instruction: str) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        parts = [p.strip() for p in instruction.split(" and ")]
        for i, part in enumerate(parts):
            if not part:
                continue
            tool = "noop"
            args: Dict[str, Any] = {}
            lower = part.lower()
            if "select" in lower:
                tool = "sql_query"
                args["query"] = part
            elif "write" in lower or "save" in lower:
                tool = "file_write"
                match = re.search(r"(?:to|save) (?:file )?([\w\.]+)", lower)
                if match:
                    args["path"] = match.group(1)
                args.setdefault("content", "")
            elif "read" in lower:
                tool = "file_read"
                match = re.search(r"read (?:file )?([\w\.]+)", lower)
                if match:
                    args["path"] = match.group(1)
            elif "log" in lower:
                tool = "log_search"
                match = re.search(r"search logs for (.+)", part, re.IGNORECASE)
                args["pattern"] = match.group(1).strip() if match else ("ERROR" if "error" in lower else "")
            steps.append(
                {
                    "step_id": f"S{i + 1}",
                    "subgoal": part,
                    "tool": tool,
                    "args": args,
                    "success_criteria": [f"complete step {i + 1}"],
                    "optional": False,
                    "save_as": f"s{i + 1}",
                }
            )
        return steps

    def _estimate_trace_tokens(self, messages: List[Dict[str, str]], attempts: List[Dict[str, Any]]) -> int:
        prompt_tokens = sum(estimate_token_count(message.get("content")) for message in messages)
        completion_tokens = sum(estimate_token_count(attempt.get("raw")) for attempt in attempts)
        return prompt_tokens + completion_tokens
