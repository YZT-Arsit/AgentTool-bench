from __future__ import annotations

from typing import Any, Dict, List

from ..env.tasks import Task
from .budget import BudgetController
from .executor import Executor
from .planner import Planner
from .reflector import Reflector
from .schema import Trajectory


class AdaptiveAgent:
    def __init__(
        self,
        llm,
        budget: BudgetController | None = None,
        disable_reflector: bool = False,
        disable_budget: bool = False,
        disable_replan: bool = False,
        disable_memory: bool = False,
    ):
        self.llm = llm
        self.planner = Planner(llm)
        self.budget_ctrl = budget or (
            BudgetController() if not disable_budget else BudgetController(max_calls=9999, max_steps=9999, max_time=9999, max_tokens=10**9)
        )
        self.executor = Executor(llm, budget_ctrl=self.budget_ctrl, disable_memory=disable_memory)
        self.reflector = Reflector(llm) if not disable_reflector else None
        self.max_replans = 0 if disable_replan else 2
        self.disable_reflector = disable_reflector
        self.disable_budget = disable_budget
        self.disable_replan = disable_replan
        self.disable_memory = disable_memory

    def run(self, task: Task, seed: int = 0, noise: float = 0.0, budget_mode: str = "default") -> Trajectory:
        budget = self.budget_ctrl.initial()
        self.executor.reset()
        traj = Trajectory(
            task_id=task.task_id,
            metadata={
                "task_type": task.task_type,
                "budget_mode": budget_mode,
                "agent": "adaptive",
                "seed": seed,
                "noise": noise,
                "parse_failures": 0,
                "fallback_count": 0,
                "replan_count": 0,
                "patch_count": 0,
                "failure_label": None,
                "budget_limits": self.budget_ctrl.limits(),
                "recovery_events": [],
            },
        )
        plan = self.planner.plan(task, budget_mode=budget_mode, scenario=f"budget={budget_mode}", replan_count=0)
        self._merge_component_trace(traj, budget, self.planner.last_trace, "planner")
        replans = 0
        next_index = 0
        last_obs: Any = None

        while True:
            while next_index < len(plan):
                if not self.budget_ctrl.check(budget):
                    self._append_recovery_event(
                        traj,
                        failure_label="BUDGET_EXHAUSTED",
                        recommended_strategy="terminate",
                        actual_recovery_action="terminate",
                        recovery_reason="Budget exhausted before the next step could run.",
                    )
                    return self._finalize_trajectory(
                        traj,
                        task,
                        budget,
                        budget_mode,
                        success=False,
                        failure_label="BUDGET_EXHAUSTED",
                    )

                step = plan[next_index]
                if self._is_branch_step(step):
                    branch_steps, merge_step, merge_index = self._collect_branch_group(plan, next_index)
                    resolution = self._execute_branch_group(
                        traj=traj,
                        task=task,
                        budget=budget,
                        branch_steps=branch_steps,
                        merge_step=merge_step,
                        start_index=next_index,
                        budget_mode=budget_mode,
                        last_obs=last_obs,
                        replans=replans,
                    )
                    replans = resolution["replans"]
                    if resolution["terminate"]:
                        return self._finalize_trajectory(
                            traj,
                            task,
                            budget,
                            budget_mode,
                            success=False,
                            failure_label=resolution["failure_label"],
                        )
                    if resolution["replan"]:
                        plan = self.planner.plan(task, budget_mode=budget_mode, scenario=f"budget={budget_mode}", replan_count=replans)
                        self._merge_component_trace(traj, budget, self.planner.last_trace, "planner")
                        next_index = 0
                        self.executor.reset()
                        traj.memory = {}
                        traj.metadata["branch_groups"] = {}
                        last_obs = None
                        break
                    last_obs = resolution["last_obs"]
                    next_index = merge_index if merge_step is not None else merge_index
                    continue

                resolution = self._execute_plan_step(
                    traj=traj,
                    task=task,
                    budget=budget,
                    step=step,
                    step_index=next_index,
                    budget_mode=budget_mode,
                    last_obs=last_obs,
                    replans=replans,
                    has_remaining_steps=next_index < len(plan) - 1,
                )
                replans = resolution["replans"]
                if resolution["done"]:
                    return self._finalize_trajectory(
                        traj,
                        task,
                        budget,
                        budget_mode,
                        success=resolution["success"],
                        failure_label=resolution["failure_label"],
                        validation_payload=resolution.get("validation_payload"),
                    )
                if resolution["terminate"]:
                    return self._finalize_trajectory(
                        traj,
                        task,
                        budget,
                        budget_mode,
                        success=False,
                        failure_label=resolution["failure_label"],
                    )
                if resolution["replan"]:
                    plan = self.planner.plan(task, budget_mode=budget_mode, scenario=f"budget={budget_mode}", replan_count=replans)
                    self._merge_component_trace(traj, budget, self.planner.last_trace, "planner")
                    next_index = 0
                    self.executor.reset()
                    traj.memory = {}
                    traj.metadata["branch_groups"] = {}
                    last_obs = None
                    break
                last_obs = resolution["last_obs"]
                next_index += 1
            else:
                validation = task.validate_result(budget_mode=budget_mode)
                if not validation.ok and not traj.metadata["recovery_events"]:
                    self._append_recovery_event(
                        traj,
                        failure_label="VALIDATION_FAILED",
                        recommended_strategy="replan",
                        actual_recovery_action="no_recovery",
                        recovery_reason="The plan ended without satisfying the validator and no explicit recovery was attempted.",
                    )
                return self._finalize_trajectory(
                    traj,
                    task,
                    budget,
                    budget_mode,
                    success=validation.ok,
                    failure_label=None if validation.ok else "VALIDATION_FAILED",
                    validation_payload=validation.to_dict(),
                )

    def _execute_plan_step(
        self,
        *,
        traj: Trajectory,
        task: Task,
        budget: Dict[str, Any],
        step: Dict[str, Any],
        step_index: int,
        budget_mode: str,
        last_obs: Any,
        replans: int,
        has_remaining_steps: bool,
    ) -> Dict[str, Any]:
        pending_patch: Dict[str, Any] | None = None
        while True:
            step_traj = self.executor.execute_step(
                task,
                step,
                budget,
                step_index,
                budget_mode,
                f"budget={budget_mode}",
                last_obs=last_obs,
                action_override=pending_patch,
            )
            pending_patch = None
            self._merge_component_trace(traj, budget, self.executor.last_trace, "executor")
            traj.steps.extend(step_traj.steps)
            traj.memory = dict(self.executor.memory)
            current = traj.steps[-1]
            current.metadata["plan_step_id"] = step.get("step_id", f"S{step_index + 1}")
            current.metadata["subgoal"] = step.get("subgoal")
            current.metadata["candidate_action_count"] = len(current.candidate_actions)
            if current.chosen_action is not None:
                current.metadata["chosen_action_source"] = current.chosen_action.source
            current.failure_label = self._classify_step_failure(current)
            current.metadata["failure_label"] = current.failure_label
            if last_obs is not None:
                current.metadata["last_obs_available"] = True
            if step.get("branch_group"):
                current.metadata["branch_group"] = step.get("branch_group")
                current.metadata["branch_id"] = step.get("branch_id")
            if step.get("merge_into"):
                current.metadata["merge_into"] = step.get("merge_into")

            validation = task.validate_result(budget_mode=budget_mode)
            traj.metadata["validation"] = validation.to_dict()
            if validation.ok:
                return {
                    "done": True,
                    "success": True,
                    "failure_label": None,
                    "validation_payload": validation.to_dict(),
                    "replans": replans,
                    "last_obs": current.output,
                    "terminate": False,
                    "replan": False,
                }

            issue_payload: Dict[str, Any] | None = None
            injection_metadata = current.metadata.get("injection_metadata") or {}
            if self._should_reflect(current, injection_metadata, validation.ok, has_remaining_steps=has_remaining_steps):
                recent_steps = [record.model_dump() for record in traj.steps[-2:]]
                issue_payload = self.reflector.reflect(
                    task_id=task.task_id,
                    plan_step=step,
                    recent_steps=recent_steps,
                    error=current.error,
                    injection_metadata=injection_metadata,
                    step_index=step_index,
                    budget_mode=budget_mode,
                    scenario=f"budget={budget_mode}",
                )
                self._merge_component_trace(traj, budget, self.reflector.last_trace, "reflector")
                current.reflection = issue_payload["label"]
                current.failure_label = issue_payload["label"]
                current.recommended_strategy = issue_payload["recommended_strategy"]
                current.recovery_reason = issue_payload.get("recovery_reason") or issue_payload.get("explanation")
                current.metadata["reflection_json"] = issue_payload
                current.metadata["failure_label"] = current.failure_label
                current.metadata["recommended_strategy"] = current.recommended_strategy
                current.metadata["recovery_reason"] = current.recovery_reason

            if current.error == "budget_exhausted":
                self._record_step_recovery(
                    traj,
                    current,
                    recommended_strategy="terminate",
                    actual_recovery_action="terminate",
                    recovery_reason="Terminate because the tool call cannot fit within the remaining budget.",
                )
                return {
                    "done": False,
                    "success": False,
                    "failure_label": "BUDGET_EXHAUSTED",
                    "replans": replans,
                    "last_obs": last_obs,
                    "terminate": True,
                    "replan": False,
                }

            if issue_payload:
                resolution = self._apply_recovery_strategy(
                    traj=traj,
                    task=task,
                    budget=budget,
                    step=step,
                    current_step=current,
                    issue_payload=issue_payload,
                    replans=replans,
                )
                replans = resolution["replans"]
                if resolution["terminate"]:
                    return {
                        "done": False,
                        "success": False,
                        "failure_label": current.failure_label,
                        "replans": replans,
                        "last_obs": last_obs,
                        "terminate": True,
                        "replan": False,
                    }
                if resolution["replan"]:
                    return {
                        "done": False,
                        "success": False,
                        "failure_label": current.failure_label,
                        "replans": replans,
                        "last_obs": last_obs,
                        "terminate": False,
                        "replan": True,
                    }
                if resolution["retry_current"]:
                    pending_patch = resolution["pending_patch"]
                    continue

            return {
                "done": False,
                "success": False,
                "failure_label": current.failure_label,
                "replans": replans,
                "last_obs": current.output,
                "terminate": False,
                "replan": False,
            }

    def _is_branch_step(self, step: Dict[str, Any]) -> bool:
        return bool(step.get("branch_group") and step.get("branch_id"))

    def _collect_branch_group(self, plan: List[Dict[str, Any]], start_index: int) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, int]:
        group_id = plan[start_index].get("branch_group")
        branch_steps: List[Dict[str, Any]] = []
        idx = start_index
        while idx < len(plan):
            step = plan[idx]
            if step.get("branch_group") == group_id and step.get("branch_id"):
                branch_steps.append(step)
                idx += 1
                continue
            break
        merge_step = None
        if idx < len(plan) and plan[idx].get("merge_into") == group_id:
            merge_step = dict(plan[idx])
            idx += 1
        return branch_steps, merge_step, idx

    def _execute_branch_group(
        self,
        *,
        traj: Trajectory,
        task: Task,
        budget: Dict[str, Any],
        branch_steps: List[Dict[str, Any]],
        merge_step: Dict[str, Any] | None,
        start_index: int,
        budget_mode: str,
        last_obs: Any,
        replans: int,
    ) -> Dict[str, Any]:
        group_id = str(branch_steps[0].get("branch_group") or f"BG{start_index + 1}")
        branch_summary = {
            "branch_group": group_id,
            "execution_mode": "sequential_parallel_simulation",
            "branches": [],
            "failed_branches": [],
            "succeeded_branches": [],
            "merge_target": merge_step.get("step_id") if merge_step else "",
        }
        traj.metadata.setdefault("branch_groups", {})[group_id] = branch_summary
        base_last_output = self.executor.last_output

        for offset, branch_step in enumerate(branch_steps):
            branch_id = str(branch_step.get("branch_id") or f"B{offset + 1}")
            branch_step = dict(branch_step)
            branch_step["branch_group"] = group_id
            branch_step["branch_id"] = branch_id
            branch_memory_before = self.executor.memory_snapshot()
            self.executor.last_output = base_last_output
            result = self._execute_plan_step(
                traj=traj,
                task=task,
                budget=budget,
                step=branch_step,
                step_index=start_index + offset,
                budget_mode=budget_mode,
                last_obs=last_obs,
                replans=replans,
                has_remaining_steps=True,
            )
            replans = result["replans"]
            current = traj.steps[-1]
            branch_record = {
                "branch_id": branch_id,
                "step_id": branch_step.get("step_id"),
                "success": current.error is None and current.failure_label not in {"TOOL_NOT_FOUND", "BUDGET_EXHAUSTED", "TOOL_EXECUTION_FAILED", "BAD_TOOL_ARGS"},
                "failure_label": current.failure_label or "",
                "memory_before_keys": sorted(branch_memory_before.keys()),
                "memory_delta_keys": sorted((current.memory_delta or {}).keys()),
            }
            branch_summary["branches"].append(branch_record)
            if branch_record["success"]:
                branch_summary["succeeded_branches"].append(branch_id)
            else:
                branch_summary["failed_branches"].append(branch_id)
                current.metadata["branch_failure"] = True
            if result["done"] or result["terminate"] or result["replan"]:
                if merge_step is not None:
                    traj.metadata["branch_groups"][group_id] = branch_summary
                return {
                    "terminate": result["terminate"],
                    "replan": result["replan"],
                    "failure_label": result["failure_label"],
                    "replans": replans,
                    "last_obs": result["last_obs"],
                }

        branch_summary["merged_memory_keys"] = sorted(
            {
                key
                for step in traj.steps[-len(branch_steps) :]
                for key in (step.memory_delta or {}).keys()
            }
        )
        traj.metadata["branch_groups"][group_id] = branch_summary
        if merge_step is not None:
            merge_step["merge_summary"] = branch_summary
            merge_result = self._execute_plan_step(
                traj=traj,
                task=task,
                budget=budget,
                step=merge_step,
                step_index=start_index + len(branch_steps),
                budget_mode=budget_mode,
                last_obs={"branch_group": group_id, "branches": branch_summary["branches"]},
                replans=replans,
                has_remaining_steps=True,
            )
            replans = merge_result["replans"]
            return {
                "terminate": merge_result["terminate"],
                "replan": merge_result["replan"],
                "failure_label": merge_result["failure_label"],
                "replans": replans,
                "last_obs": merge_result["last_obs"],
            }
        return {
            "terminate": False,
            "replan": False,
            "failure_label": "",
            "replans": replans,
            "last_obs": {"branch_group": group_id, "branches": branch_summary["branches"]},
        }

    def _should_reflect(self, current_step, injection_metadata: Dict[str, Any], validation_ok: bool, *, has_remaining_steps: bool) -> bool:
        return bool(
            self.reflector
            and (
                current_step.error
                or injection_metadata.get("injection_type")
                or current_step.output in (None, "", [], {})
                or (not validation_ok and not has_remaining_steps)
                or current_step.metadata.get("fallback_reason") == "action_json_invalid"
            )
        )

    def _classify_step_failure(self, step) -> str | None:
        if step.reflection:
            return step.reflection
        if step.metadata.get("fallback_reason") == "action_json_invalid" or step.metadata.get("parse_failures", 0):
            return "JSON_MALFORMED"
        if step.error == "tool_not_found":
            return "TOOL_NOT_FOUND"
        if step.error == "budget_exhausted":
            return "BUDGET_EXHAUSTED"
        if step.error:
            return "TOOL_EXECUTION_FAILED"
        if step.output in (None, "", [], {}):
            return "EMPTY_RESULT"
        if step.tool == "noop":
            return "PLAN_MISMATCH"
        return "VALIDATION_FAILED"

    def _apply_recovery_strategy(
        self,
        *,
        traj: Trajectory,
        task: Task,
        budget: Dict[str, Any],
        step: Dict[str, Any],
        current_step,
        issue_payload: Dict[str, Any],
        replans: int,
    ) -> Dict[str, Any]:
        strategy = issue_payload["recommended_strategy"]
        current_step.recommended_strategy = strategy
        current_step.recovery_reason = issue_payload.get("recovery_reason") or issue_payload.get("explanation")

        if strategy == "patch_args":
            patch = issue_payload.get("patch") or {"tool": step.get("tool"), "args": step.get("args", {})}
            self._record_step_recovery(
                traj,
                current_step,
                recommended_strategy=strategy,
                actual_recovery_action="patch_args",
                recovery_reason=current_step.recovery_reason,
            )
            traj.metadata["patch_count"] += 1
            return {
                "terminate": False,
                "replan": False,
                "retry_current": True,
                "pending_patch": {
                    "tool": patch.get("tool") or step.get("tool"),
                    "args": patch.get("args") or step.get("args", {}),
                    "save_as": step.get("save_as"),
                },
                "replans": replans,
            }

        if strategy == "patch_tool":
            patch = issue_payload.get("patch") or {"tool": step.get("tool"), "args": step.get("args", {})}
            self._record_step_recovery(
                traj,
                current_step,
                recommended_strategy=strategy,
                actual_recovery_action="patch_tool",
                recovery_reason=current_step.recovery_reason,
            )
            traj.metadata["patch_count"] += 1
            return {
                "terminate": False,
                "replan": False,
                "retry_current": True,
                "pending_patch": {
                    "tool": patch.get("tool") or step.get("tool"),
                    "args": patch.get("args") or step.get("args", {}),
                    "save_as": step.get("save_as"),
                },
                "replans": replans,
            }

        if strategy == "retry_safe":
            self._record_step_recovery(
                traj,
                current_step,
                recommended_strategy=strategy,
                actual_recovery_action="safe_fallback",
                recovery_reason=current_step.recovery_reason,
            )
            return {"terminate": False, "replan": False, "retry_current": False, "pending_patch": None, "replans": replans}

        if strategy == "replan":
            if replans < self.max_replans:
                current_step.replanned = True
                self._record_step_recovery(
                    traj,
                    current_step,
                    recommended_strategy=strategy,
                    actual_recovery_action="replan",
                    recovery_reason=current_step.recovery_reason,
                )
                traj.metadata["replan_count"] = replans + 1
                return {"terminate": False, "replan": True, "retry_current": False, "pending_patch": None, "replans": replans + 1}
            self._record_step_recovery(
                traj,
                current_step,
                recommended_strategy=strategy,
                actual_recovery_action="replan_unavailable",
                recovery_reason="Replan was recommended but the agent already exhausted its replan budget.",
            )
            return {"terminate": False, "replan": False, "retry_current": False, "pending_patch": None, "replans": replans}

        if strategy == "fail_fast":
            self._record_step_recovery(
                traj,
                current_step,
                recommended_strategy=strategy,
                actual_recovery_action="fail_fast",
                recovery_reason=current_step.recovery_reason,
            )
            return {"terminate": True, "replan": False, "retry_current": False, "pending_patch": None, "replans": replans}

        if strategy == "terminate":
            self._record_step_recovery(
                traj,
                current_step,
                recommended_strategy=strategy,
                actual_recovery_action="terminate",
                recovery_reason=current_step.recovery_reason,
            )
            return {"terminate": True, "replan": False, "retry_current": False, "pending_patch": None, "replans": replans}

        self._record_step_recovery(
            traj,
            current_step,
            recommended_strategy=strategy,
            actual_recovery_action="no_recovery",
            recovery_reason="No matching recovery handler was found, so execution will continue without intervention.",
        )
        return {"terminate": False, "replan": False, "retry_current": False, "pending_patch": None, "replans": replans}

    def _record_step_recovery(
        self,
        traj: Trajectory,
        current_step,
        *,
        recommended_strategy: str,
        actual_recovery_action: str,
        recovery_reason: str,
    ) -> None:
        current_step.recommended_strategy = recommended_strategy
        current_step.actual_recovery_action = actual_recovery_action
        current_step.recovery_reason = recovery_reason
        current_step.metadata["recommended_strategy"] = recommended_strategy
        current_step.metadata["actual_recovery_action"] = actual_recovery_action
        current_step.metadata["recovery_reason"] = recovery_reason
        self._append_recovery_event(
            traj,
            failure_label=current_step.failure_label or "UNKNOWN",
            recommended_strategy=recommended_strategy,
            actual_recovery_action=actual_recovery_action,
            recovery_reason=recovery_reason,
        )

    def _append_recovery_event(
        self,
        traj: Trajectory,
        *,
        failure_label: str,
        recommended_strategy: str,
        actual_recovery_action: str,
        recovery_reason: str,
    ) -> None:
        traj.metadata.setdefault("recovery_events", []).append(
            {
                "failure_label": failure_label,
                "recommended_strategy": recommended_strategy,
                "actual_recovery_action": actual_recovery_action,
                "recovery_reason": recovery_reason,
            }
        )

    def _merge_component_trace(self, traj: Trajectory, budget: Dict[str, Any], trace: Dict[str, Any], component: str) -> None:
        if not trace:
            return
        traj.metadata["parse_failures"] = traj.metadata.get("parse_failures", 0) + int(trace.get("parse_failures", 0))
        if trace.get("fallback_used"):
            traj.metadata["fallback_count"] = traj.metadata.get("fallback_count", 0) + 1
        if trace.get("estimated_tokens"):
            self.budget_ctrl.consume_llm_tokens(budget, int(trace["estimated_tokens"]))
        component_stats = traj.metadata.setdefault("component_stats", {})
        component_stats.setdefault(component, []).append(
            {
                "llm_raw_text": trace.get("llm_raw_text"),
                "parsed_json": trace.get("parsed_json"),
                "validation_errors": trace.get("validation_errors", []),
                "fallback_used": trace.get("fallback_used", False),
                "fallback_reason": trace.get("fallback_reason", ""),
                "parse_failures": trace.get("parse_failures", 0),
                "estimated_tokens": trace.get("estimated_tokens", 0),
                "injection_metadata": trace.get("injection_metadata"),
            }
        )

    def _finalize_trajectory(
        self,
        traj: Trajectory,
        task: Task,
        budget: Dict[str, Any],
        budget_mode: str,
        *,
        success: bool,
        failure_label: str | None,
        validation_payload: Dict[str, Any] | None = None,
    ) -> Trajectory:
        validation = validation_payload or task.validate_result(budget_mode=budget_mode).to_dict()
        traj.success = success
        traj.memory = dict(self.executor.memory)
        traj.metadata["validation"] = validation
        traj.metadata["failure_label"] = failure_label or self._derive_failure_label(traj, success)
        if traj.metadata["recovery_events"]:
            latest = traj.metadata["recovery_events"][-1]
            traj.metadata["recommended_strategy"] = latest["recommended_strategy"]
            traj.metadata["actual_recovery_action"] = latest["actual_recovery_action"]
            traj.metadata["recovery_reason"] = latest["recovery_reason"]
        return self._attach_runtime_metadata(traj, budget)

    def _attach_runtime_metadata(self, traj: Trajectory, budget: Dict[str, Any]) -> Trajectory:
        traj.metadata["budget_usage"] = self.budget_ctrl.snapshot(budget)
        traj.metadata["memory_keys"] = sorted(traj.memory.keys())
        return traj

    def _derive_failure_label(self, traj: Trajectory, success: bool) -> str | None:
        if success:
            return None
        for step in reversed(traj.steps):
            label = step.failure_label or step.metadata.get("failure_label") or step.reflection
            if label:
                return str(label)
        return "VALIDATION_FAILED"
