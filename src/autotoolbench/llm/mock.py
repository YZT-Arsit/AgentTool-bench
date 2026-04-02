from __future__ import annotations

import json
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .base import LLMBase


class MockLLM(LLMBase):
    model_name = "mock-json"

    def __init__(self, seed: Optional[int] = None, noise: float = 0.0, json_error_rate: float = 0.0):
        self.noise = noise
        self.seed = seed
        self.json_error_rate = json_error_rate
        self.last_generation_metadata: Dict[str, Any] = {}

    def generate(self, messages: List[Dict], schema_name: str = "text") -> str:
        context = self._parse_context(messages[-1]["content"])
        raw_payload = self._build_payload(schema_name, context)
        original_payload = deepcopy(raw_payload)
        corrupted_payload, injection = self._inject_noise(schema_name, context, raw_payload)
        raw_text = json.dumps(corrupted_payload, ensure_ascii=False)
        malformed = self._should_malformed(schema_name, context)
        if malformed:
            raw_text = self._malform_json(raw_text)
        self.last_generation_metadata = {
            "schema_name": schema_name,
            "context": context,
            "original_json": json.dumps(original_payload, ensure_ascii=False),
            "corrupted_json": raw_text,
            "injection": injection,
            "json_error_injected": malformed,
        }
        return raw_text

    def maybe_corrupt(self, data: Dict) -> Dict:
        return data

    def corrupt_action(
        self,
        action: Dict[str, Any],
        *,
        task_id: str,
        step_index: int,
        available_tools: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        context = {
            "task_id": task_id,
            "step_index": step_index,
            "available_tools": available_tools,
            "plan_step": {"tool": action.get("tool"), "args_hint": action.get("args", {})},
            "task_instruction": "",
            "budget_mode": "default",
        }
        corrupted, injection = self._inject_noise("action", context, {"tool": action.get("tool"), "args": action.get("args", {})})
        return corrupted, injection or {}

    def _parse_context(self, content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"text": content}
        return {"text": str(content)}

    def _build_payload(self, schema_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if schema_name == "plan":
            steps = []
            for idx, hint in enumerate(context.get("plan_hints") or [], start=1):
                step = {
                    "step_id": hint.get("step_id", f"S{idx}"),
                    "subgoal": hint.get("subgoal", f"step {idx}"),
                    "tool": hint.get("tool", "noop"),
                    "args_hint": hint.get("args", hint.get("args_hint", {})) or {},
                    "success_criteria": hint.get("success_criteria", [f"step {idx} completed"]),
                    "optional": bool(hint.get("optional", False)),
                    "save_as": hint.get("save_as"),
                }
                steps.append(step)
            return {"steps": steps}
        if schema_name == "action":
            plan_step = context.get("plan_step", {})
            tool = plan_step.get("tool", "noop")
            args = deepcopy(plan_step.get("args_hint", {}))
            if tool == "file_write" and not args.get("content"):
                last_output = context.get("last_output")
                if isinstance(last_output, str):
                    args["content"] = last_output
                elif last_output is not None:
                    args["content"] = json.dumps(last_output, ensure_ascii=False, indent=2)
            return {
                "tool": tool,
                "args": args,
                "rationale": f"Execute {plan_step.get('subgoal', 'current step')} with {tool}",
            }
        if schema_name == "reflection":
            label = self._infer_reflection_label(context)
            patch = self._build_patch(context, label)
            recommended_strategy = self._recommended_strategy(label)
            return {
                "label": label,
                "explanation": f"Recover from {label.lower()} based on the latest observation.",
                "recommended_strategy": recommended_strategy,
                "fix_action": "patch" if recommended_strategy in {"patch_args", "patch_tool", "retry_safe"} else recommended_strategy,
                "replan_needed": recommended_strategy == "replan",
                "recovery_reason": f"Prefer {recommended_strategy} for {label.lower()}",
                "patch": patch,
            }
        return {"text": "mock"}

    def _inject_noise(
        self,
        schema_name: str,
        context: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
        signature = self._signature(schema_name, context)
        local_random = random.Random(signature)
        if self.noise <= 0 or schema_name not in {"plan", "action", "reflection"}:
            return payload, None
        effective_noise = self.noise if schema_name != "reflection" else min(0.5, self.noise)
        if local_random.random() >= effective_noise:
            return payload, None
        original = deepcopy(payload)
        corrupted = deepcopy(payload)
        injection_type = local_random.choice(["TOOL_CHOICE_ERROR", "TOOL_ARGS_ERROR", "MISSING_STEP"])
        if schema_name == "plan":
            corrupted = self._corrupt_plan(corrupted, injection_type)
        elif schema_name == "action":
            corrupted = self._corrupt_action_payload(corrupted, injection_type, context, local_random)
        else:
            corrupted = self._corrupt_reflection(corrupted, injection_type)
        return corrupted, {
            "injection_type": injection_type,
            "affected_step": context.get("step_index", 0),
            "original_json": original,
            "corrupted_json": corrupted,
        }

    def _signature(self, schema_name: str, context: Dict[str, Any]) -> str:
        return ":".join(
            [
                str(self.seed),
                schema_name,
                str(context.get("task_id", "task")),
                str(context.get("step_index", 0)),
                str(context.get("replan_count", 0)),
            ]
        )

    def _should_malformed(self, schema_name: str, context: Dict[str, Any]) -> bool:
        if self.json_error_rate <= 0 or schema_name not in {"plan", "action", "reflection"}:
            return False
        local_random = random.Random(self._signature(f"json:{schema_name}", context))
        return local_random.random() < self.json_error_rate

    def _malform_json(self, text: str) -> str:
        if text.startswith("{"):
            return text[1:]
        if text.startswith("["):
            return text[1:]
        return text + " trailing"

    def _corrupt_plan(self, payload: Dict[str, Any], injection_type: str) -> Dict[str, Any]:
        steps = payload.get("steps", [])
        if not steps:
            return payload
        if injection_type == "MISSING_STEP":
            payload["steps"] = steps[:-1] or steps
            return payload
        if injection_type == "TOOL_CHOICE_ERROR":
            steps[0]["tool"] = "noop"
            steps[0]["args_hint"] = {}
            return payload
        steps[0]["args_hint"] = self._corrupt_args(steps[0].get("args_hint", {}))
        return payload

    def _corrupt_action_payload(
        self,
        payload: Dict[str, Any],
        injection_type: str,
        context: Dict[str, Any],
        local_random: random.Random,
    ) -> Dict[str, Any]:
        if injection_type == "TOOL_CHOICE_ERROR":
            safe_tools = {"noop", "file_read", "file_write", "log_search", "sql_query"}
            alternatives = [tool for tool in context.get("available_tools", []) if tool in safe_tools and tool != payload.get("tool")]
            if alternatives:
                payload["tool"] = local_random.choice(sorted(alternatives))
                payload["args"] = {}
        elif injection_type == "TOOL_ARGS_ERROR":
            payload["args"] = self._corrupt_args(payload.get("args", {}))
        else:
            payload["tool"] = "noop"
            payload["args"] = {}
        return payload

    def _corrupt_reflection(self, payload: Dict[str, Any], injection_type: str) -> Dict[str, Any]:
        if injection_type == "MISSING_STEP":
            payload["label"] = "PLAN_MISMATCH"
            payload["recommended_strategy"] = "replan"
            payload["replan_needed"] = True
            payload["patch"] = None
        elif injection_type == "TOOL_ARGS_ERROR":
            payload["recommended_strategy"] = "patch_args"
            payload["patch"] = None
            payload["replan_needed"] = False
        return payload

    def _infer_reflection_label(self, context: Dict[str, Any]) -> str:
        injection_type = (context.get("injection_metadata") or {}).get("injection_type")
        error = (context.get("error") or "").lower()
        tool = context.get("tool")
        if injection_type == "MISSING_STEP":
            return "MISSING_PREREQUISITE"
        if error == "tool_not_found":
            return "TOOL_NOT_FOUND"
        if error == "budget_exhausted":
            return "BUDGET_EXHAUSTED"
        if injection_type == "TOOL_ARGS_ERROR":
            return "BAD_TOOL_ARGS"
        if error:
            return "TOOL_EXECUTION_FAILED"
        if injection_type == "TOOL_CHOICE_ERROR" or tool == "noop":
            return "PLAN_MISMATCH"
        return "PLAN_MISMATCH"

    def _build_patch(self, context: Dict[str, Any], label: str) -> Dict[str, Any] | None:
        step = context.get("plan_step", {})
        if label == "BAD_TOOL_ARGS":
            return {"tool": step.get("tool"), "args": deepcopy(step.get("args_hint", {}))}
        if label == "PLAN_MISMATCH" and step.get("tool"):
            return {"tool": step.get("tool"), "args": deepcopy(step.get("args_hint", {}))}
        return None

    def _recommended_strategy(self, label: str) -> str:
        if label in {"BAD_TOOL_ARGS", "EMPTY_RESULT", "VALIDATION_FAILED"}:
            return "patch_args"
        if label in {"MISSING_PREREQUISITE", "PLAN_MISMATCH"}:
            return "replan"
        if label == "JSON_MALFORMED":
            return "retry_safe"
        if label == "BUDGET_EXHAUSTED":
            return "terminate"
        if label == "TOOL_NOT_FOUND":
            return "fail_fast"
        return "fail_fast"

    def _corrupt_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not args:
            return {}
        corrupted = deepcopy(args)
        for key, value in list(corrupted.items()):
            if isinstance(value, str):
                upper_value = value.upper()
                if "REQ-404" in value:
                    corrupted[key] = value.replace("REQ-404", "REQ-405")
                elif "cache_node_timeout" in value:
                    corrupted[key] = value.replace("cache_node_timeout", "cache_timeout")
                elif "Alice" in value:
                    corrupted[key] = value.replace("Alice", "Alicia")
                elif "LIMIT 2" in upper_value:
                    corrupted[key] = value.replace("LIMIT 2", "LIMIT 3")
                elif "AGE < 30" in upper_value:
                    corrupted[key] = value.replace("< 30", "<= 30")
                elif "paid" in value:
                    corrupted[key] = value.replace("paid", "pending", 1)
                else:
                    corrupted[key] = value + "_noise"
                break
            if isinstance(value, int):
                corrupted[key] = value + 1
                break
        return corrupted
