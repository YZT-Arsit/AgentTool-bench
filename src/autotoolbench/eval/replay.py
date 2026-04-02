from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..agent.schema import StepRecord, Trajectory
from ..env.tasks import Task
from .metrics import failure_profile


def _budget_text(budget: Dict[str, Any]) -> str:
    return (
        f"calls={budget.get('calls', 0)}, "
        f"steps={budget.get('steps', 0)}, "
        f"tokens={budget.get('tokens', 0)}, "
        f"time={float(budget.get('time', 0.0)):.4f}"
    )


def _recovery_actions(traj: Trajectory) -> list[str]:
    actions: list[str] = []
    for event in traj.metadata.get("recovery_events", []):
        action = str(event.get("actual_recovery_action") or "").strip()
        if action and action not in actions:
            actions.append(action)
    return actions


def _memory_refs(traj: Trajectory) -> list[str]:
    refs: list[str] = []
    for step in traj.steps:
        for key in step.metadata.get("referenced_memory_keys", []):
            key = str(key)
            if key and key not in refs:
                refs.append(key)
    return refs[:6]


def _first_failure_step(traj: Trajectory) -> tuple[int | None, StepRecord | None]:
    for idx, step in enumerate(traj.steps, start=1):
        if step.failure_label or step.error or step.reflection or step.safety_decision == "blocked":
            return idx, step
    return None, None


def _key_success_path(traj: Trajectory) -> str:
    tools = [step.tool for step in traj.steps if step.tool]
    if not tools:
        return "No tool steps were recorded."
    unique_tools: list[str] = []
    for tool in tools:
        if tool not in unique_tools:
            unique_tools.append(tool)
    return " -> ".join(unique_tools[:4])


def summarize_trajectory_markdown(traj: Trajectory, task: Task | None = None) -> str:
    profile = failure_profile(traj)
    validation = traj.metadata.get("validation", {})
    budget_usage = traj.metadata.get("budget_usage", {})
    task_id = task.task_id if task is not None else traj.task_id
    task_type = task.task_type if task is not None else str(traj.metadata.get("task_type") or "unknown")
    recovery_actions = _recovery_actions(traj)
    memory_refs = _memory_refs(traj)
    first_failure_idx, first_failure_step = _first_failure_step(traj)

    lines = [
        f"### Replay Summary: {task_id}",
        "",
        f"- Task Type: `{task_type}`",
        f"- Outcome: `{'success' if traj.success else 'failure'}`",
        f"- First Failure Stage: `{profile.get('first_failure_stage') or 'none'}`",
        f"- Final Failure Label: `{traj.metadata.get('failure_label') or 'none'}`",
        f"- Recovery Actions: `{', '.join(recovery_actions) if recovery_actions else 'none'}`",
        f"- Budget Usage: `{_budget_text(budget_usage)}`",
        f"- Key Memory References: `{', '.join(memory_refs) if memory_refs else 'none'}`",
        f"- Final Validation: `{validation.get('validator', 'unknown')}` -> `{validation.get('message', 'no validation summary')}`",
        "",
    ]

    if traj.success:
        lines.extend(
            [
                "**Success Path**",
                f"- Key tool path: `{_key_success_path(traj)}`",
                f"- Recovery encountered: `{'yes' if recovery_actions else 'no'}`",
                f"- Validator passed because: `{validation.get('message', 'validator reported success')}`",
            ]
        )
    else:
        failure_reason = ""
        if first_failure_step is not None:
            failure_reason = (
                first_failure_step.safety_reason
                or first_failure_step.recovery_reason
                or first_failure_step.error
                or first_failure_step.failure_label
                or "unknown failure"
            )
        final_reason = validation.get("message") or traj.metadata.get("recovery_reason") or "task never reached a passing validator state"
        lines.extend(
            [
                "**Failure Diagnosis**",
                f"- Failure first appeared at step: `{first_failure_idx or 'n/a'}`",
                f"- Why it failed: `{failure_reason or 'unknown failure'}`",
                f"- Recovery attempts made: `{', '.join(recovery_actions) if recovery_actions else 'none'}`",
                f"- Why it still failed: `{final_reason}`",
            ]
        )

    return "\n".join(lines) + "\n"


def write_trajectory_summary(traj: Trajectory, path: Path, task: Task | None = None) -> str:
    summary_text = summarize_trajectory_markdown(traj, task=task)
    path.write_text(summary_text, encoding="utf-8")
    return summary_text
