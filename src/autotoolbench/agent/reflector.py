from __future__ import annotations

import json
from typing import Any, Dict, List

from .budget import estimate_token_count
from .json_utils import bounded_retry, validate_reflection
from .schema import ReflectionPayload

FAILURE_LABELS = {
    "TOOL_NOT_FOUND",
    "BAD_TOOL_ARGS",
    "EMPTY_RESULT",
    "MISSING_PREREQUISITE",
    "PLAN_MISMATCH",
    "VALIDATION_FAILED",
    "BUDGET_EXHAUSTED",
    "JSON_MALFORMED",
    "TOOL_EXECUTION_FAILED",
}

RECOVERY_STRATEGIES = {
    "patch_args",
    "patch_tool",
    "replan",
    "fail_fast",
    "retry_safe",
    "terminate",
}


class Reflector:
    def __init__(self, llm=None):
        self.llm = llm
        self.last_trace: Dict[str, Any] = {}

    def classify(self, traj_step: Dict[str, Any]) -> str:
        metadata = traj_step.get("metadata", {})
        injection_type = metadata.get("injection_type")
        error = (traj_step.get("error") or "").lower()
        output = traj_step.get("output")
        if metadata.get("fallback_reason") == "action_json_invalid" or metadata.get("parse_failures", 0):
            return "JSON_MALFORMED"
        if injection_type == "MISSING_STEP":
            return "MISSING_PREREQUISITE"
        if injection_type == "TOOL_CHOICE_ERROR":
            return "PLAN_MISMATCH"
        if injection_type == "TOOL_ARGS_ERROR":
            return "BAD_TOOL_ARGS"
        if error == "tool_not_found":
            return "TOOL_NOT_FOUND"
        if error == "budget_exhausted":
            return "BUDGET_EXHAUSTED"
        if error in {"invalid path", "path escape", "not found"}:
            return "BAD_TOOL_ARGS"
        if error:
            return "TOOL_EXECUTION_FAILED"
        if output in (None, "", [], {}):
            return "EMPTY_RESULT"
        if traj_step.get("tool") == "noop":
            return "PLAN_MISMATCH"
        return "VALIDATION_FAILED"

    def suggest_fix(self, issue: str) -> str:
        suggestions = {
            "TOOL_NOT_FOUND": "The requested tool is unavailable, so stop instead of fabricating a replacement.",
            "BAD_TOOL_ARGS": "Patch the arguments using the plan hints or working memory and retry safely.",
            "EMPTY_RESULT": "Tighten or broaden the query/condition before retrying the same step.",
            "MISSING_PREREQUISITE": "Replan to insert the missing prerequisite step before continuing.",
            "PLAN_MISMATCH": "Replan because the current plan/tool choice is not aligned with the task.",
            "VALIDATION_FAILED": "If the trajectory looks close to success, patch the final action; otherwise replan.",
            "BUDGET_EXHAUSTED": "Terminate immediately and report the exhausted budget.",
            "JSON_MALFORMED": "Use a conservative fallback action and avoid aggressive recovery.",
            "TOOL_EXECUTION_FAILED": "Patch arguments for recoverable errors and fail fast for structural tool failures.",
        }
        return suggestions.get(issue, f"Please fix {issue}")

    def recommend_strategy(
        self,
        label: str,
        *,
        plan_step: Dict[str, Any],
        recent_steps: List[Dict[str, Any]],
        error: str | None,
    ) -> Dict[str, Any]:
        normalized = self._normalize_label(label, fallback="VALIDATION_FAILED")
        latest_step = recent_steps[-1] if recent_steps else {}
        latest_output = latest_step.get("output")
        lowered_error = (error or latest_step.get("error") or "").lower()
        close_to_success = bool(latest_output not in (None, "", [], {}) and not lowered_error)

        strategy = "replan"
        patch: Dict[str, Any] | None = None
        reason = self.suggest_fix(normalized)

        if normalized == "BAD_TOOL_ARGS":
            strategy = "patch_args"
            patch = {"tool": plan_step.get("tool"), "args": dict(plan_step.get("args", {}))}
        elif normalized == "EMPTY_RESULT":
            strategy = "patch_args"
            patch = {"tool": plan_step.get("tool"), "args": dict(plan_step.get("args", {}))}
        elif normalized == "MISSING_PREREQUISITE":
            strategy = "replan"
            reason = "Replan to add the missing prerequisite before retrying downstream steps."
        elif normalized == "PLAN_MISMATCH":
            strategy = "replan"
            reason = "Replan because the current tool choice diverged from the intended plan."
        elif normalized == "TOOL_NOT_FOUND":
            strategy = "fail_fast"
            reason = "Fail fast because the tool is not registered and a safe patch is not available."
        elif normalized == "JSON_MALFORMED":
            strategy = "retry_safe"
            patch = {"tool": plan_step.get("tool"), "args": dict(plan_step.get("args", {}))} if plan_step.get("tool") else None
            reason = "Use a conservative fallback action based on the plan hints after malformed JSON."
        elif normalized == "TOOL_EXECUTION_FAILED":
            recoverable = any(token in lowered_error for token in ["syntax", "invalid", "missing", "path", "query"])
            if recoverable and plan_step.get("tool"):
                strategy = "patch_args"
                patch = {"tool": plan_step.get("tool"), "args": dict(plan_step.get("args", {}))}
                reason = "Patch the arguments because the tool error looks recoverable."
            else:
                strategy = "fail_fast"
                reason = "Fail fast because the tool execution error looks structural rather than recoverable."
        elif normalized == "VALIDATION_FAILED":
            if close_to_success and plan_step.get("tool"):
                strategy = "patch_args"
                patch = {"tool": plan_step.get("tool"), "args": dict(plan_step.get("args", {}))}
                reason = "Patch the final action because the trajectory appears close to satisfying the validator."
            else:
                strategy = "replan"
                reason = "Replan because the validator failed and the trajectory is not close enough to success."
        elif normalized == "BUDGET_EXHAUSTED":
            strategy = "terminate"
            reason = "Terminate immediately because the budget is exhausted."

        return {
            "label": normalized,
            "recommended_strategy": strategy,
            "patch": patch,
            "recovery_reason": reason,
            "fix_action": "patch" if strategy in {"patch_args", "patch_tool", "retry_safe"} else strategy,
            "replan_needed": strategy == "replan",
        }

    def reflect(
        self,
        *,
        task_id: str,
        plan_step: Dict[str, Any],
        recent_steps: List[Dict[str, Any]],
        error: str | None,
        injection_metadata: Dict[str, Any] | None,
        step_index: int,
        budget_mode: str,
        scenario: str,
    ) -> Dict[str, Any]:
        fallback_label = self.classify(recent_steps[-1] if recent_steps else {"metadata": injection_metadata or {}, "error": error})
        context = {
            "task_id": task_id,
            "plan_step": plan_step,
            "recent_steps": recent_steps,
            "error": error,
            "tool": recent_steps[-1].get("tool") if recent_steps else None,
            "step_index": step_index,
            "budget_mode": budget_mode,
            "scenario": scenario,
            "injection_metadata": injection_metadata or {},
            "failure_taxonomy": sorted(FAILURE_LABELS),
            "recovery_strategies": sorted(RECOVERY_STRATEGIES),
        }
        messages = self._messages(context)
        result = bounded_retry(lambda: self.llm.generate(messages, schema_name="reflection"), validate_reflection, max_retries=2)
        llm_meta = getattr(self.llm, "last_generation_metadata", {})
        estimated_tokens = self._estimate_trace_tokens(messages, result["attempts"])
        if result["ok"]:
            payload = ReflectionPayload.model_validate(result["payload"]).model_dump()
            payload["label"] = self._normalize_label(payload["label"], fallback=fallback_label)
            strategy_info = self.recommend_strategy(
                payload["label"],
                plan_step=plan_step,
                recent_steps=recent_steps,
                error=error,
            )
            payload["recommended_strategy"] = self._normalize_strategy(payload.get("recommended_strategy"), strategy_info["recommended_strategy"])
            payload["fix_action"] = strategy_info["fix_action"]
            payload["replan_needed"] = strategy_info["replan_needed"]
            payload["recovery_reason"] = payload.get("recovery_reason") or strategy_info["recovery_reason"]
            if payload["patch"] is None:
                payload["patch"] = strategy_info["patch"]
            self.last_trace = {
                "llm_raw_text": result["raw"],
                "parsed_json": payload,
                "validation_errors": [],
                "fallback_used": False,
                "fallback_reason": "",
                "parse_failures": result["parse_failures"],
                "attempts": result["attempts"],
                "estimated_tokens": estimated_tokens,
                "injection_metadata": llm_meta.get("injection"),
            }
            return payload

        strategy_info = self.recommend_strategy(
            fallback_label,
            plan_step=plan_step,
            recent_steps=recent_steps,
            error=error,
        )
        fallback = {
            "label": strategy_info["label"],
            "explanation": self.suggest_fix(strategy_info["label"]),
            "recommended_strategy": strategy_info["recommended_strategy"],
            "fix_action": strategy_info["fix_action"],
            "replan_needed": strategy_info["replan_needed"],
            "recovery_reason": strategy_info["recovery_reason"],
            "patch": strategy_info["patch"],
        }
        self.last_trace = {
            "llm_raw_text": result["raw"],
            "parsed_json": result["payload"],
            "validation_errors": result["validation_errors"],
            "fallback_used": True,
            "fallback_reason": "reflection_json_invalid",
            "parse_failures": result["parse_failures"],
            "attempts": result["attempts"],
            "estimated_tokens": estimated_tokens,
            "injection_metadata": llm_meta.get("injection"),
        }
        return fallback

    def _normalize_label(self, label: str, fallback: str) -> str:
        normalized = (label or "").strip().upper()
        if normalized in FAILURE_LABELS:
            return normalized
        alias_map = {
            "MISSING_PREREQ": "MISSING_PREREQUISITE",
            "PLAN_ERROR": "PLAN_MISMATCH",
        }
        return alias_map.get(normalized, fallback)

    def _normalize_strategy(self, strategy: str | None, fallback: str) -> str:
        normalized = (strategy or "").strip().lower()
        if normalized in RECOVERY_STRATEGIES:
            return normalized
        return fallback

    def _messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are the reflector for AutoToolBench. Return JSON only. "
                    'Schema: {"label":"BAD_TOOL_ARGS","explanation":"...","recommended_strategy":"patch_args",'
                    '"fix_action":"patch","replan_needed":false,"recovery_reason":"...",'
                    '"patch":{"tool":"sql_query","args":{}}}'
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

    def _estimate_trace_tokens(self, messages: List[Dict[str, str]], attempts: List[Dict[str, Any]]) -> int:
        prompt_tokens = sum(estimate_token_count(message.get("content")) for message in messages)
        completion_tokens = sum(estimate_token_count(attempt.get("raw")) for attempt in attempts)
        return prompt_tokens + completion_tokens
