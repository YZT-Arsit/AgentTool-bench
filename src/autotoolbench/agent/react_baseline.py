from __future__ import annotations

from typing import Any

from ..env.tasks import Task
from .budget import BudgetController
from .executor import Executor
from .planner import Planner
from .schema import Trajectory


class ReactAgent:
    def __init__(self, llm, max_steps: int = 10):
        self.llm = llm
        self.max_steps = max_steps
        self.planner = Planner(llm)
        self.budget_ctrl = BudgetController(max_calls=max_steps, max_steps=max_steps, max_time=9999, max_tokens=10**9)
        self.executor = Executor(llm, budget_ctrl=self.budget_ctrl)

    def run(self, task: Task, seed: int = 0, noise: float = 0.0, budget_mode: str = "default") -> Trajectory:
        traj = Trajectory(
            task_id=task.task_id,
            metadata={
                "task_type": task.task_type,
                "budget_mode": budget_mode,
                "agent": "react",
                "parse_failures": 0,
                "fallback_count": 0,
                "replan_count": 0,
                "patch_count": 0,
                "budget_limits": self.budget_ctrl.limits(),
            },
        )
        self.executor.reset()
        plan = self.planner.plan(task, budget_mode=budget_mode, scenario="react", replan_count=0)
        traj.metadata["parse_failures"] += int(self.planner.last_trace.get("parse_failures", 0))
        if self.planner.last_trace.get("fallback_used"):
            traj.metadata["fallback_count"] += 1
        last_obs: Any = None
        budget = self.budget_ctrl.initial()
        if self.planner.last_trace.get("estimated_tokens"):
            self.budget_ctrl.consume_llm_tokens(budget, int(self.planner.last_trace["estimated_tokens"]))
        for index, step in enumerate(plan[: self.max_steps]):
            step_traj = self.executor.execute_step(task, step, budget, index, budget_mode, "react", last_obs=last_obs)
            traj.metadata["parse_failures"] += int(self.executor.last_trace.get("parse_failures", 0))
            if self.executor.last_trace.get("fallback_used"):
                traj.metadata["fallback_count"] += 1
            traj.steps.extend(step_traj.steps)
            traj.memory = dict(self.executor.memory)
            if self.executor.last_trace.get("estimated_tokens"):
                self.budget_ctrl.consume_llm_tokens(budget, int(self.executor.last_trace["estimated_tokens"]))
            current = traj.steps[-1]
            current.metadata["plan_step_id"] = step.get("step_id", f"S{index + 1}")
            current.metadata["subgoal"] = step.get("subgoal")
            last_obs = current.output
            if task.validate(budget_mode=budget_mode):
                traj.success = True
                traj.metadata["validation"] = task.validate_result(budget_mode=budget_mode).to_dict()
                traj.metadata["failure_label"] = None
                traj.metadata["budget_usage"] = self.budget_ctrl.snapshot(budget)
                traj.metadata["memory_keys"] = sorted(traj.memory.keys())
                return traj
        traj.success = False
        traj.metadata["validation"] = task.validate_result(budget_mode=budget_mode).to_dict()
        traj.metadata["failure_label"] = next(
            (step.metadata.get("failure_label") for step in reversed(traj.steps) if step.metadata.get("failure_label")),
            "VALIDATION_FAILED",
        )
        traj.metadata["budget_usage"] = self.budget_ctrl.snapshot(budget)
        traj.metadata["memory_keys"] = sorted(traj.memory.keys())
        return traj
