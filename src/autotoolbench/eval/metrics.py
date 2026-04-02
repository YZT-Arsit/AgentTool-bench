from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from ..agent.schema import StepRecord, Trajectory

RECOVERY_CONTINUE_ACTIONS = {"patch_args", "patch_tool", "replan", "safe_fallback"}
FAILURE_STAGES = ("planner", "action_generation", "tool_execution", "validator", "reflector_recovery")


def _budget_usage(traj: Trajectory) -> Dict[str, Any]:
    if isinstance(traj.metadata.get("budget_usage"), dict):
        return traj.metadata["budget_usage"]
    if traj.steps:
        return traj.steps[-1].budget
    return {}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _recovery_steps(traj: Trajectory) -> List[StepRecord]:
    return [step for step in traj.steps if (step.actual_recovery_action or "") in RECOVERY_CONTINUE_ACTIONS]


def _recovery_events(traj: Trajectory) -> List[Dict[str, Any]]:
    return [
        event
        for event in traj.metadata.get("recovery_events", [])
        if str(event.get("actual_recovery_action", "")) in RECOVERY_CONTINUE_ACTIONS
    ]


def _recovery_cost_from_first_attempt(traj: Trajectory) -> Dict[str, float]:
    budget = _budget_usage(traj)
    recovery_steps = _recovery_steps(traj)
    if not recovery_steps:
        return {"calls": 0.0, "tokens": 0.0, "runtime": 0.0}
    first = recovery_steps[0]
    return {
        "calls": max(0.0, float(budget.get("calls", 0)) - float(first.budget.get("calls", 0))),
        "tokens": max(0.0, float(budget.get("tokens", 0)) - float(first.budget.get("tokens", 0))),
        "runtime": max(0.0, float(budget.get("time", 0.0)) - float(first.budget.get("time", 0.0))),
    }


def _planner_trace_failed(traj: Trajectory) -> bool:
    planner_stats = traj.metadata.get("component_stats", {}).get("planner", [])
    return any(
        int(item.get("parse_failures", 0)) > 0
        or bool(item.get("fallback_used"))
        or bool(item.get("validation_errors"))
        for item in planner_stats
    )


def _step_failure_label(step: StepRecord) -> str | None:
    label = step.failure_label or step.metadata.get("failure_label") or step.reflection
    return str(label) if label else None


def _stage_from_step(step: StepRecord) -> str | None:
    label = _step_failure_label(step)
    if step.actual_recovery_action in {"fail_fast", "terminate", "replan_unavailable"}:
        return "reflector_recovery"
    if label in {"PLAN_MISMATCH", "MISSING_PREREQUISITE"}:
        return "planner"
    if label in {"JSON_MALFORMED", "BAD_TOOL_ARGS", "TOOL_NOT_FOUND"}:
        return "action_generation"
    if label in {"EMPTY_RESULT", "TOOL_EXECUTION_FAILED", "BUDGET_EXHAUSTED"}:
        return "tool_execution"
    if label == "VALIDATION_FAILED":
        return "validator"
    if step.metadata.get("fallback_reason") == "action_json_invalid" or int(step.metadata.get("parse_failures", 0)) > 0:
        return "action_generation"
    if step.error in {"tool_not_found", "missing_memory_key", "budget_exhausted"} or str(step.error).startswith("missing_memory_key:"):
        return "action_generation" if "memory" in str(step.error) else "tool_execution"
    return None


def failure_profile(traj: Trajectory) -> Dict[str, Any]:
    recovery_events = _recovery_events(traj)
    first_failure_stage: str | None = None
    first_failure_step_index: int | None = None

    if _planner_trace_failed(traj):
        first_failure_stage = "planner"
        first_failure_step_index = -1

    for idx, step in enumerate(traj.steps):
        stage = _stage_from_step(step)
        if stage is not None:
            first_failure_stage = stage
            first_failure_step_index = idx
            break

    if first_failure_stage is None and not traj.success and str(traj.metadata.get("failure_label") or "") == "VALIDATION_FAILED":
        first_failure_stage = "validator"
        first_failure_step_index = len(traj.steps)

    if traj.success:
        final_failure_stage = "recovered" if first_failure_stage is not None else ""
    elif recovery_events and str(traj.metadata.get("actual_recovery_action") or "") in {"fail_fast", "terminate", "replan_unavailable", "no_recovery"}:
        final_failure_stage = "reflector_recovery"
    elif str(traj.metadata.get("failure_label") or "") == "VALIDATION_FAILED":
        final_failure_stage = "validator"
    else:
        final_failure_stage = ""
        for step in reversed(traj.steps):
            stage = _stage_from_step(step)
            if stage is not None:
                final_failure_stage = stage
                break
        if not final_failure_stage and not traj.success:
            final_failure_stage = "validator"

    failure_recovered = bool(first_failure_stage) and bool(traj.success)
    propagation_candidates = sum(1 for step in traj.steps if _stage_from_step(step) is not None)
    later_steps_exist = first_failure_step_index is not None and first_failure_step_index >= 0 and first_failure_step_index < len(traj.steps) - 1
    failure_propagated = bool(
        first_failure_stage
        and (
            later_steps_exist
            or len(recovery_events) > 0
            or propagation_candidates > 1
            or (not traj.success and final_failure_stage and final_failure_stage != first_failure_stage)
        )
    )

    return {
        "first_failure_stage": first_failure_stage or "",
        "final_failure_stage": final_failure_stage or "",
        "final_failure_label": str(traj.metadata.get("failure_label") or ""),
        "failure_recovered": failure_recovered,
        "recovery_attempt_count": len(recovery_events),
        "failure_propagated": failure_propagated,
    }


def summarize(trajs: List[Trajectory], runtimes: List[float] | None = None) -> Dict[str, Any]:
    total = len(trajs)
    success = sum(1 for traj in trajs if traj.success)
    budgets = [_budget_usage(traj) for traj in trajs]
    failure_breakdown: Counter[str] = Counter()
    recommended_strategy_breakdown: Counter[str] = Counter()
    actual_recovery_action_breakdown: Counter[str] = Counter()
    first_failure_stage_breakdown: Counter[str] = Counter()
    recovered_by_stage: Counter[str] = Counter()
    unrecovered_by_stage: Counter[str] = Counter()
    stage_to_stage_summary: Counter[str] = Counter()
    propagated_failure_count = 0
    trajectories_with_failure_origin = 0
    retrieval_task_count = 0
    retrieval_hit_count = 0
    retrieval_source_coverage_total = 0.0
    retrieval_term_coverage_total = 0.0
    retrieval_noise_ratio_total = 0.0
    retrieval_evidence_usage_count = 0

    recovery_attempt_count = 0
    recovery_task_count = 0
    recovery_success_count = 0
    patched_task_count = 0
    patched_success_count = 0
    replanned_task_count = 0
    replanned_success_count = 0
    recovery_cost_calls_total = 0.0
    recovery_cost_tokens_total = 0.0
    recovery_cost_runtime_total = 0.0

    for traj in trajs:
        profile = failure_profile(traj)
        if not traj.success:
            failure_breakdown[str(traj.metadata.get("failure_label") or "UNKNOWN")] += 1
        if profile["first_failure_stage"]:
            trajectories_with_failure_origin += 1
            first_failure_stage_breakdown[profile["first_failure_stage"]] += 1
            stage_to_stage_summary[f"{profile['first_failure_stage']}->{profile['final_failure_stage'] or 'unknown'}"] += 1
            if profile["failure_recovered"]:
                recovered_by_stage[profile["first_failure_stage"]] += 1
            else:
                unrecovered_by_stage[profile["first_failure_stage"]] += 1
            if profile["failure_propagated"]:
                propagated_failure_count += 1

        for event in traj.metadata.get("recovery_events", []):
            if event.get("recommended_strategy"):
                recommended_strategy_breakdown[str(event["recommended_strategy"])] += 1
            if event.get("actual_recovery_action"):
                actual_recovery_action_breakdown[str(event["actual_recovery_action"])] += 1

        recovery_events = _recovery_events(traj)
        recovery_attempt_count += len(recovery_events)
        if recovery_events:
            recovery_task_count += 1
            if traj.success:
                recovery_success_count += 1
            recovery_cost = _recovery_cost_from_first_attempt(traj)
            recovery_cost_calls_total += recovery_cost["calls"]
            recovery_cost_tokens_total += recovery_cost["tokens"]
            recovery_cost_runtime_total += recovery_cost["runtime"]

        if int(traj.metadata.get("patch_count", 0)) > 0:
            patched_task_count += 1
            if traj.success:
                patched_success_count += 1

        if int(traj.metadata.get("replan_count", 0)) > 0:
            replanned_task_count += 1
            if traj.success:
                replanned_success_count += 1

        retrieval_analysis = traj.metadata.get("retrieval_analysis", {})
        if retrieval_analysis:
            retrieval_task_count += 1
            if retrieval_analysis.get("hit"):
                retrieval_hit_count += 1
            retrieval_source_coverage_total += float(retrieval_analysis.get("source_coverage", 0.0))
            retrieval_term_coverage_total += float(retrieval_analysis.get("term_coverage", 0.0))
            retrieval_noise_ratio_total += float(retrieval_analysis.get("noise_ratio", 0.0))
            if retrieval_analysis.get("used_memory"):
                retrieval_evidence_usage_count += 1

    avg_steps = sum(int(budget.get("steps", len(traj.steps))) for traj, budget in zip(trajs, budgets)) / total if total else 0.0
    avg_calls = sum(int(budget.get("calls", 0)) for budget in budgets) / total if total else 0.0
    avg_runtime = sum(float(budget.get("time", 0.0)) for budget in budgets) / total if total else 0.0
    avg_tokens = sum(int(budget.get("tokens", 0)) for budget in budgets) / total if total else 0.0
    avg_parse_failures = sum(int(traj.metadata.get("parse_failures", 0)) for traj in trajs) / total if total else 0.0
    avg_fallbacks = sum(int(traj.metadata.get("fallback_count", 0)) for traj in trajs) / total if total else 0.0
    avg_replans = sum(int(traj.metadata.get("replan_count", 0)) for traj in trajs) / total if total else 0.0
    avg_patches = sum(int(traj.metadata.get("patch_count", 0)) for traj in trajs) / total if total else 0.0
    budget_exhaustion_count = sum(1 for traj in trajs if traj.metadata.get("failure_label") == "BUDGET_EXHAUSTED")
    wall_clock_runtime = sum(runtimes or []) / total if total else 0.0

    total_calls = sum(float(budget.get("calls", 0)) for budget in budgets)
    total_tokens = sum(float(budget.get("tokens", 0)) for budget in budgets)
    total_runtime = sum(float(budget.get("time", 0.0)) for budget in budgets)

    return {
        "total": total,
        "success_rate": _safe_div(success, total),
        "avg_steps": avg_steps,
        "avg_calls": avg_calls,
        "avg_runtime": avg_runtime,
        "avg_time": avg_runtime,
        "avg_wall_clock_runtime": wall_clock_runtime,
        "avg_tokens": avg_tokens,
        "avg_parse_failures": avg_parse_failures,
        "avg_fallbacks": avg_fallbacks,
        "avg_replans": avg_replans,
        "avg_patches": avg_patches,
        "budget_exhaustion_count": budget_exhaustion_count,
        "failure_breakdown": dict(sorted(failure_breakdown.items())),
        "failure_types": dict(sorted(failure_breakdown.items())),
        "first_failure_stage_breakdown": dict(sorted(first_failure_stage_breakdown.items())),
        "recovered_by_stage": dict(sorted(recovered_by_stage.items())),
        "unrecovered_by_stage": dict(sorted(unrecovered_by_stage.items())),
        "failure_propagation_rate": _safe_div(propagated_failure_count, trajectories_with_failure_origin),
        "stage_to_stage_propagation_summary": dict(sorted(stage_to_stage_summary.items())),
        "recommended_strategy_breakdown": dict(sorted(recommended_strategy_breakdown.items())),
        "actual_recovery_action_breakdown": dict(sorted(actual_recovery_action_breakdown.items())),
        "parse_failure_rate": avg_parse_failures,
        "fallback_count": sum(int(traj.metadata.get("fallback_count", 0)) for traj in trajs),
        "replan_count": sum(int(traj.metadata.get("replan_count", 0)) for traj in trajs),
        "patch_count": sum(int(traj.metadata.get("patch_count", 0)) for traj in trajs),
        "success_per_call": _safe_div(success, total_calls),
        "success_per_estimated_token": _safe_div(success, total_tokens),
        "success_per_runtime": _safe_div(success, total_runtime),
        "recovery_attempt_count": recovery_attempt_count,
        "recovery_task_count": recovery_task_count,
        "recovery_success_count": recovery_success_count,
        "recovery_success_rate": _safe_div(recovery_success_count, recovery_task_count),
        "avg_recovery_cost_calls": _safe_div(recovery_cost_calls_total, recovery_task_count),
        "avg_recovery_cost_tokens": _safe_div(recovery_cost_tokens_total, recovery_task_count),
        "avg_recovery_cost_runtime": _safe_div(recovery_cost_runtime_total, recovery_task_count),
        "patched_success_rate": _safe_div(patched_success_count, patched_task_count),
        "replanned_success_rate": _safe_div(replanned_success_count, replanned_task_count),
        "retrieval_task_count": retrieval_task_count,
        "retrieval_hit_rate": _safe_div(retrieval_hit_count, retrieval_task_count),
        "avg_retrieval_source_coverage": _safe_div(retrieval_source_coverage_total, retrieval_task_count),
        "avg_retrieval_term_coverage": _safe_div(retrieval_term_coverage_total, retrieval_task_count),
        "avg_retrieval_noise_ratio": _safe_div(retrieval_noise_ratio_total, retrieval_task_count),
        "retrieval_evidence_usage_rate": _safe_div(retrieval_evidence_usage_count, retrieval_task_count),
    }
