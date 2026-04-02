from __future__ import annotations

import json
import subprocess
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..agent.adaptive_agent import AdaptiveAgent
from ..agent.budget import BudgetController
from ..agent.react_baseline import ReactAgent
from ..agent.schema import Trajectory
from ..constants import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
    LATEST_TASKS_SNAPSHOT_NAME,
    LATEST_TOOL_SCHEMA_NAME,
)
from ..data_gen import main as generate_data
from ..data_gen import reset_runtime_artifacts
from ..env.tasks import Task, load_tasks
from ..llm.mock import MockLLM
from ..llm.openai_client import OpenAIClient
from ..tools.registry import all_tools
from ..utils.paths import create_run_dir, reports_dir, trajectories_dir
from .metrics import failure_profile, summarize
from .replay import summarize_trajectory_markdown, write_trajectory_summary

DEFAULT_SCENARIOS = [
    {"label": "noise=0.0", "noise": 0.0, "budget_preset": "loose"},
    {"label": "noise=0.2 tight", "noise": 0.2, "budget_preset": "tight"},
    {"label": "noise=0.2 loose", "noise": 0.2, "budget_preset": "loose"},
]

ABLATION_AGENTS = ["no_reflector", "no_replan", "no_budget", "no_memory", "weak_validation"]


class TaskValidationWrapper:
    def __init__(self, task: Task, mode: str):
        self._task = task
        self._mode = mode
        self.task_id = task.task_id
        self.instruction = task.instruction
        self.expected_artifacts = task.expected_artifacts
        self.validator = task.validator
        self.validator_params = task.validator_params
        self.category = task.category
        self.task_type = task.task_type
        self.difficulty = task.difficulty
        self.budget_mode = task.budget_mode
        self.plan_hints = task.plan_hints
        self.retrieval_expectations = getattr(task, "retrieval_expectations", {})
        self.raw = dict(task.raw)

    def resolve_validator_params(self, budget_mode: str = "default") -> Dict[str, Any]:
        params = deepcopy(self._task.resolve_validator_params(budget_mode))
        if self._mode != "weak_validation":
            return params
        if self.validator == "multi_artifact" and isinstance(params, dict) and params.get("validators"):
            return {"validators": [params["validators"][0]]}
        return params

    def validate_result(self, budget_mode: str = "default"):
        from ..env.validators import ValidationResult, run_validator

        result = run_validator(self.validator, self.resolve_validator_params(budget_mode))
        if self._mode == "weak_validation":
            return ValidationResult(
                ok=result.ok,
                validator=f"weak::{result.validator}",
                message="Weak validation mode uses a reduced validator set for ablation comparability.",
                details={"base_validator": result.validator, "mode": self._mode},
                children=[],
            )
        return result

    def validate(self, budget_mode: str = "default") -> bool:
        return self.validate_result(budget_mode).ok


def _build_llm(
    seed: int,
    noise: float,
    llm_type: str,
    json_error_rate: float,
    openai_model: str,
    temperature: float,
):
    if llm_type == "openai":
        return OpenAIClient(model=openai_model, temperature=temperature)
    return MockLLM(seed=seed, noise=noise, json_error_rate=json_error_rate)


def _build_agent(
    name: str,
    seed: int,
    noise: float,
    budget_preset: str,
    llm_type: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
):
    llm = _build_llm(seed, noise, llm_type, json_error_rate, openai_model, temperature)
    budget = None if name == "react" else BudgetController.from_preset(budget_preset)
    if name == "react":
        return ReactAgent(llm)
    if name == "plan":
        return AdaptiveAgent(llm, budget=budget, disable_reflector=True, disable_replan=True)
    if name == "adaptive":
        return AdaptiveAgent(llm, budget=budget)
    if name == "no_reflector":
        return AdaptiveAgent(llm, budget=budget, disable_reflector=True)
    if name == "no_replan":
        return AdaptiveAgent(llm, budget=budget, disable_replan=True)
    if name == "no_budget":
        return AdaptiveAgent(llm, budget=BudgetController(max_calls=9999, max_steps=9999, max_time=9999, max_tokens=10**9), disable_budget=True)
    if name == "no_memory":
        return AdaptiveAgent(llm, budget=budget, disable_memory=True)
    if name == "weak_validation":
        return AdaptiveAgent(llm, budget=budget)
    raise KeyError(name)


def _prepare_task_for_agent(task: Task, agent_name: str) -> Task:
    if agent_name == "weak_validation":
        return TaskValidationWrapper(task, mode="weak_validation")
    return task


def _failure_label(traj: Trajectory) -> str:
    if traj.metadata.get("failure_label"):
        return str(traj.metadata["failure_label"])
    labels: list[str] = []
    for step in traj.steps:
        if getattr(step, "failure_label", None):
            labels.append(str(step.failure_label))
        elif step.metadata.get("failure_label"):
            labels.append(str(step.metadata["failure_label"]))
        elif step.reflection:
            labels.append(step.reflection)
        elif step.error:
            labels.append(step.error)
    if not labels:
        return "UNKNOWN"
    return Counter(labels).most_common(1)[0][0]


def _save_trajectory(traj: Trajectory, task: Task, run_dir: Path, agent_name: str) -> tuple[str, str]:
    agent_dir = trajectories_dir(run_dir) / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / f"{traj.task_id}.json"
    path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    summary_path = agent_dir / f"{traj.task_id}.summary.md"
    write_trajectory_summary(traj, summary_path, task=task)
    return str(path), str(summary_path)


def _task_type_rates(task_outcomes: Dict[str, list[bool]]) -> Dict[str, float]:
    return {task_type: (sum(values) / len(values) if values else 0.0) for task_type, values in sorted(task_outcomes.items())}


def _task_type_stage_analysis(trajectories: List[dict[str, Any]], agent_name: str, scenario_label: str) -> Dict[str, Dict[str, Any]]:
    filtered = [item for item in trajectories if item.get("agent") == agent_name and item.get("scenario") == scenario_label]
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in filtered:
        task_type = str(item.get("task_type") or "unknown")
        stage = str(item.get("first_failure_stage") or "none")
        stats = grouped.setdefault(
            task_type,
            {
                "total": 0,
                "with_failure_origin": 0,
                "propagated_count": 0,
                "first_failure_stage_breakdown": Counter(),
                "recovered_by_stage": Counter(),
                "unrecovered_by_stage": Counter(),
            },
        )
        stats["total"] += 1
        if stage != "none":
            stats["with_failure_origin"] += 1
            stats["first_failure_stage_breakdown"][stage] += 1
            if item.get("failure_propagated"):
                stats["propagated_count"] += 1
            if item.get("failure_recovered"):
                stats["recovered_by_stage"][stage] += 1
            else:
                stats["unrecovered_by_stage"][stage] += 1
    return {
        task_type: {
            "total": data["total"],
            "with_failure_origin": data["with_failure_origin"],
            "failure_propagation_rate": (
                data["propagated_count"] / data["with_failure_origin"] if data["with_failure_origin"] else 0.0
            ),
            "first_failure_stage_breakdown": dict(sorted(data["first_failure_stage_breakdown"].items())),
            "recovered_by_stage": dict(sorted(data["recovered_by_stage"].items())),
            "unrecovered_by_stage": dict(sorted(data["unrecovered_by_stage"].items())),
        }
        for task_type, data in sorted(grouped.items())
    }


def _task_type_stage_strategy_analysis(trajs: List[Trajectory], agent_name: str, scenario_label: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for traj in trajs:
        if traj.metadata.get("agent") != agent_name or traj.metadata.get("scenario") != scenario_label:
            continue
        task_type = str(traj.metadata.get("task_type") or "unknown")
        stage = str(traj.metadata.get("first_failure_stage") or "none")
        stats = grouped.setdefault(
            task_type,
            {
                "recommended_strategy_by_stage": defaultdict(Counter),
                "actual_recovery_action_by_stage": defaultdict(Counter),
                "stage_event_count": Counter(),
            },
        )
        if stage == "none":
            continue
        recovery_events = traj.metadata.get("recovery_events", [])
        if not recovery_events:
            stats["stage_event_count"][stage] += 1
            stats["actual_recovery_action_by_stage"][stage]["no_recovery"] += 1
            continue
        for event in recovery_events:
            stats["stage_event_count"][stage] += 1
            recommended = str(event.get("recommended_strategy") or "none")
            actual = str(event.get("actual_recovery_action") or "none")
            stats["recommended_strategy_by_stage"][stage][recommended] += 1
            stats["actual_recovery_action_by_stage"][stage][actual] += 1
    return {
        task_type: {
            "stage_event_count": dict(sorted(data["stage_event_count"].items())),
            "recommended_strategy_by_stage": {
                stage: dict(sorted(counter.items())) for stage, counter in sorted(data["recommended_strategy_by_stage"].items())
            },
            "actual_recovery_action_by_stage": {
                stage: dict(sorted(counter.items())) for stage, counter in sorted(data["actual_recovery_action_by_stage"].items())
            },
        }
        for task_type, data in sorted(grouped.items())
    }


def _trajectory_index_entry(traj: Trajectory, task: Task, agent_name: str, scenario_label: str) -> dict[str, Any]:
    budget_usage = traj.metadata.get("budget_usage", {})
    profile = failure_profile(traj)
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "agent": agent_name,
        "scenario": scenario_label,
        "success": bool(traj.success),
        "failure_label": traj.metadata.get("failure_label", ""),
        "first_failure_stage": profile["first_failure_stage"],
        "final_failure_stage": profile["final_failure_stage"],
        "failure_recovered": profile["failure_recovered"],
        "recovery_attempt_count": profile["recovery_attempt_count"],
        "failure_propagated": profile["failure_propagated"],
        "recommended_strategy": traj.metadata.get("recommended_strategy", ""),
        "actual_recovery_action": traj.metadata.get("actual_recovery_action", ""),
        "recovery_reason": traj.metadata.get("recovery_reason", ""),
        "budget_usage": budget_usage,
        "parse_failures": int(traj.metadata.get("parse_failures", 0)),
        "fallback_count": int(traj.metadata.get("fallback_count", 0)),
        "replan_count": int(traj.metadata.get("replan_count", 0)),
        "patch_count": int(traj.metadata.get("patch_count", 0)),
        "retrieval_analysis": traj.metadata.get("retrieval_analysis", {}),
        "trajectory_path": traj.metadata.get("trajectory_path", ""),
        "replay_summary_path": traj.metadata.get("replay_summary_path", ""),
        "replay_summary": traj.metadata.get("replay_summary", ""),
    }


def _analyze_retrieval(traj: Trajectory, task: Task) -> dict[str, Any]:
    expectations = getattr(task, "retrieval_expectations", {}) or {}
    if not expectations:
        return {}
    retrieval_steps = [step for step in traj.steps if step.tool == "doc_search" and isinstance(step.output, list)]
    slot = str(expectations.get("slot") or "")
    if not retrieval_steps:
        return {
            "task_id": task.task_id,
            "retrieved": False,
            "hit": False,
            "slot": slot,
            "required_sources": list(expectations.get("required_sources", [])),
            "required_terms": list(expectations.get("required_terms", [])),
            "source_hits": [],
            "term_hits": [],
            "source_coverage": 0.0,
            "term_coverage": 0.0,
            "noise_count": 0,
            "noise_ratio": 1.0,
            "used_memory": False,
            "result_count": 0,
        }

    target_step = retrieval_steps[0]
    if slot:
        for step in retrieval_steps:
            action_json = step.metadata.get("action_json", {})
            if action_json.get("save_as") == slot:
                target_step = step
                break

    results = target_step.output if isinstance(target_step.output, list) else []
    required_sources = list(expectations.get("required_sources", []))
    required_terms = list(expectations.get("required_terms", []))
    source_hits = set()
    term_hits = set()
    noise_count = 0
    for item in results:
        if not isinstance(item, dict):
            noise_count += 1
            continue
        source = str(item.get("source") or item.get("path") or "")
        chunk = str(item.get("chunk") or item.get("text") or "")
        if source in required_sources:
            source_hits.add(source)
        matched_terms = [term for term in required_terms if term in chunk]
        term_hits.update(matched_terms)
        if source not in required_sources and not matched_terms:
            noise_count += 1

    source_coverage = len(source_hits) / len(required_sources) if required_sources else 1.0
    term_coverage = len(term_hits) / len(required_terms) if required_terms else 1.0
    used_memory = False
    if slot:
        for step in traj.steps:
            if step.tool == "doc_search":
                continue
            if slot in step.metadata.get("referenced_memory_keys", []):
                used_memory = True
                break

    return {
        "task_id": task.task_id,
        "retrieved": True,
        "hit": source_coverage >= 1.0 and term_coverage >= 1.0,
        "slot": slot,
        "required_sources": required_sources,
        "required_terms": required_terms,
        "source_hits": sorted(source_hits),
        "term_hits": sorted(term_hits),
        "source_coverage": source_coverage,
        "term_coverage": term_coverage,
        "noise_count": noise_count,
        "noise_ratio": (noise_count / len(results)) if results else 0.0,
        "used_memory": used_memory,
        "result_count": len(results),
    }


def run_agents(
    agent_names: List[str],
    seed: int = 0,
    noise: float = 0.0,
    budget_preset: str = "loose",
    scenario_label: str = "single",
    save_traces: bool = False,
    llm_type: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    run_dir: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    details = run_agents_detailed(
        agent_names,
        seed=seed,
        noise=noise,
        budget_preset=budget_preset,
        scenario_label=scenario_label,
        save_traces=save_traces,
        llm_type=llm_type,
        json_error_rate=json_error_rate,
        openai_model=openai_model,
        temperature=temperature,
        run_dir=run_dir,
    )
    return details["summary"]


def run_agents_detailed(
    agent_names: List[str],
    seed: int = 0,
    noise: float = 0.0,
    budget_preset: str = "loose",
    scenario_label: str = "single",
    save_traces: bool = False,
    llm_type: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    run_dir: Path | None = None,
) -> Dict[str, Any]:
    tasks = load_tasks()
    summary: Dict[str, Dict[str, Any]] = {}
    by_task_type: Dict[str, Dict[str, float]] = {}
    agent_trajs: Dict[str, List[Trajectory]] = {}
    failures: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    effective_run_dir = run_dir or create_run_dir("eval")

    for name in agent_names:
        reset_runtime_artifacts()
        agent = _build_agent(
            name,
            seed=seed,
            noise=noise,
            budget_preset=budget_preset,
            llm_type=llm_type,
            json_error_rate=json_error_rate,
            openai_model=openai_model,
            temperature=temperature,
        )
        trajs: list[Trajectory] = []
        runtimes: list[float] = []
        task_outcomes: dict[str, list[bool]] = defaultdict(list)
        for task in tasks:
            reset_runtime_artifacts()
            effective_task = _prepare_task_for_agent(task, name)
            started = time.perf_counter()
            traj = agent.run(effective_task, seed=seed, noise=noise, budget_mode=budget_preset)
            runtimes.append(time.perf_counter() - started)
            traj.metadata.update(
                {
                    "task_type": task.task_type,
                    "scenario": scenario_label,
                    "agent": name,
                    "budget_preset": budget_preset,
                    "noise": noise,
                    "llm_type": llm_type,
                    "llm_model": getattr(getattr(agent, "llm", None), "model_name", llm_type),
                    "ablation_mode": name if name in ABLATION_AGENTS else "",
                }
            )
            traj.metadata["retrieval_analysis"] = _analyze_retrieval(traj, task)
            traj.metadata.setdefault("failure_label", _failure_label(traj) if not traj.success else None)
            traj.metadata.setdefault("recommended_strategy", "")
            traj.metadata.setdefault("actual_recovery_action", "")
            traj.metadata.setdefault("recovery_reason", "")
            traj.metadata.setdefault("budget_usage", traj.steps[-1].budget if traj.steps else {"calls": 0, "steps": 0, "time": 0.0, "tokens": 0})
            traj.metadata.update(failure_profile(traj))
            traj.metadata["replay_summary"] = summarize_trajectory_markdown(traj, task=task)
            if save_traces:
                path, summary_path = _save_trajectory(traj, task, effective_run_dir / scenario_label.replace(" ", "_").replace("=", "_"), name)
                traj.metadata["trajectory_path"] = path
                traj.metadata["replay_summary_path"] = summary_path
            trajs.append(traj)
            task_outcomes[task.task_type].append(bool(traj.success))
            trajectories.append(_trajectory_index_entry(traj, task, name, scenario_label))
            if not traj.success:
                failures.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "agent": name,
                        "scenario": scenario_label,
                        "label": _failure_label(traj),
                        "first_failure_stage": traj.metadata.get("first_failure_stage", ""),
                        "final_failure_stage": traj.metadata.get("final_failure_stage", ""),
                        "failure_propagated": bool(traj.metadata.get("failure_propagated", False)),
                        "validation": traj.metadata.get("validation", {}),
                        "trajectory_path": traj.metadata.get("trajectory_path", ""),
                    }
                )
        summary[name] = summarize(trajs, runtimes)
        by_task_type[name] = _task_type_rates(task_outcomes)
        agent_trajs[name] = list(trajs)

    task_type_stage_analysis = {name: _task_type_stage_analysis(trajectories, name, scenario_label) for name in agent_names}
    task_type_stage_strategy_analysis = {
        name: _task_type_stage_strategy_analysis(agent_trajs.get(name, []), name, scenario_label) for name in agent_names
    }

    return {
        "summary": summary,
        "task_type_rates": by_task_type,
        "task_type_stage_analysis": task_type_stage_analysis,
        "task_type_stage_strategy_analysis": task_type_stage_strategy_analysis,
        "failures": failures,
        "trajectories": trajectories,
        "run_dir": str(effective_run_dir),
    }


def run_matrix(
    agent_names: List[str],
    seed: int = 0,
    include_ablation: bool = True,
    llm_type: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    generate_data(seed=seed)
    effective_agents = list(agent_names)
    if include_ablation:
        effective_agents.extend(ABLATION_AGENTS)

    run_dir = create_run_dir("eval")
    scenario_results: dict[str, dict[str, Any]] = {}
    all_failures: list[dict[str, Any]] = []
    all_trajectories: list[dict[str, Any]] = []
    for scenario in DEFAULT_SCENARIOS:
        details = run_agents_detailed(
            effective_agents,
            seed=seed,
            noise=scenario["noise"],
            budget_preset=scenario["budget_preset"],
            scenario_label=scenario["label"],
            save_traces=True,
            llm_type=llm_type,
            json_error_rate=json_error_rate,
            openai_model=openai_model,
            temperature=temperature,
            run_dir=run_dir,
        )
        scenario_results[scenario["label"]] = details
        all_failures.extend(details["failures"])
        all_trajectories.extend(details["trajectories"])

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({label: details["summary"] for label, details in scenario_results.items()}, indent=2), encoding="utf-8")

    return {
        "scenarios": DEFAULT_SCENARIOS,
        "results": scenario_results,
        "failures": all_failures,
        "trajectories": all_trajectories,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
    }


def evaluation_snapshot(
    agent_names: Iterable[str],
    seed: int,
    matrix: Dict[str, Any],
    llm_type: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    commit = "unknown"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    run_dir = Path(matrix["run_dir"])
    tasks = load_tasks()
    tasks_snapshot_path = run_dir / "tasks_snapshot.jsonl"
    with tasks_snapshot_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.raw) + "\n")
    tool_schema_path = run_dir / "tool_schema.json"
    tool_schema = {name: tool.input_schema for name, tool in all_tools().items()}
    tool_schema_path.write_text(json.dumps(tool_schema, indent=2), encoding="utf-8")
    snapshot = {
        "agent_names": list(agent_names),
        "seed": seed,
        "noise": [scenario["noise"] for scenario in DEFAULT_SCENARIOS],
        "budget_presets": [scenario["budget_preset"] for scenario in DEFAULT_SCENARIOS],
        "task_count": len(tasks),
        "git_commit": commit,
        "tasks_snapshot": str(tasks_snapshot_path),
        "tool_schema_snapshot": str(tool_schema_path),
        "tool_exception_enabled": False,
        "scenarios": matrix["scenarios"],
        "llm_type": llm_type,
        "json_error_rate": json_error_rate,
        "openai_model": openai_model if llm_type == "openai" else "",
        "temperature": temperature,
        "agent_version": "p2-json-chain",
        "run_dir": str(run_dir),
        "summary_path": matrix.get("summary_path", ""),
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    latest_dir = reports_dir()
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / LATEST_TASKS_SNAPSHOT_NAME).write_text(tasks_snapshot_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / LATEST_TOOL_SCHEMA_NAME).write_text(tool_schema_path.read_text(encoding="utf-8"), encoding="utf-8")
    return snapshot
