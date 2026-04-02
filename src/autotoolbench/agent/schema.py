from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: str
    subgoal: str
    tool: str
    args_hint: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: List[str] = Field(default_factory=list)
    optional: bool = False
    save_as: Optional[str] = None
    branch_group: Optional[str] = None
    branch_id: Optional[str] = None
    independent: bool = False
    merge_into: Optional[str] = None
    merge_requirements: List[str] = Field(default_factory=list)


class PlanPayload(BaseModel):
    steps: List[PlanStep] = Field(default_factory=list)


class ToolArgumentConstraint(BaseModel):
    type: str = "any"
    required: bool = False
    non_empty: bool = False
    min_items: Optional[int] = None
    pattern: Optional[str] = None
    description: Optional[str] = None
    enum: List[Any] = Field(default_factory=list)


class ActionPayload(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    save_as: Optional[str] = None


class ActionCandidate(BaseModel):
    candidate_id: str
    source: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    save_as: Optional[str] = None


class ActionScore(BaseModel):
    candidate_id: str
    estimated_success_likelihood: float
    estimated_budget_cost: float
    estimated_cost_breakdown: Dict[str, float] = Field(default_factory=dict)
    risk_level: float
    tool_compatibility: float
    total_score: float
    reasons: List[str] = Field(default_factory=list)


class ReflectionPatch(BaseModel):
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None


class ReflectionPayload(BaseModel):
    label: str
    explanation: str
    recommended_strategy: str
    fix_action: str
    replan_needed: bool
    recovery_reason: Optional[str] = None
    patch: Optional[ReflectionPatch] = None


class MemoryEntry(BaseModel):
    key: str
    value: Any = None
    value_type: str = "unknown"
    source_step_id: Optional[str] = None
    source_tool: Optional[str] = None
    updated_at: float = Field(default_factory=time.time)


class StepRecord(BaseModel):
    timestamp: float
    subgoal: Optional[str]
    tool: Optional[str]
    input: Dict[str, Any]
    output: Any = None
    error: Optional[str] = None
    reflection: Optional[str] = None
    failure_label: Optional[str] = None
    recommended_strategy: Optional[str] = None
    actual_recovery_action: Optional[str] = None
    recovery_reason: Optional[str] = None
    candidate_actions: List[ActionCandidate] = Field(default_factory=list)
    chosen_action: Optional[ActionCandidate] = None
    action_scores: List[ActionScore] = Field(default_factory=list)
    selection_reason: Optional[str] = None
    tool_risk_level: Optional[str] = None
    action_allowed: bool = True
    safety_decision: str = "allowed"
    safety_reason: Optional[str] = None
    safety_level: Optional[str] = None
    branch_group: Optional[str] = None
    branch_id: Optional[str] = None
    merge_point: bool = False
    replanned: bool = False
    budget: Dict[str, Any] = Field(default_factory=dict)
    cost: Dict[str, Any] = Field(default_factory=dict)
    memory_delta: Dict[str, Any] = Field(default_factory=dict)
    merge_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    task_id: str
    steps: List[StepRecord] = Field(default_factory=list)
    success: Optional[bool] = None
    memory: Dict[str, MemoryEntry] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_step(self, **kwargs: Any) -> None:
        kwargs.setdefault("timestamp", time.time())
        self.steps.append(StepRecord(**kwargs))
