from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from ..constants import LATEST_CONFIG_NAME, LATEST_REPORT_NAME, LATEST_SUMMARY_NAME
from ..utils.paths import reports_dir


def _top_failures(matrix: Dict[str, Any], scenario_label: str, limit: int = 8) -> list[dict[str, Any]]:
    filtered = [item for item in matrix["failures"] if item["scenario"] == scenario_label]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in filtered:
        grouped[(item["task_id"], item["task_type"])].append(item)
    ranking = []
    for (task_id, task_type), failures in grouped.items():
        common_label = Counter(item["label"] for item in failures).most_common(1)[0][0]
        ranking.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "count": len(failures),
                "label": common_label,
                "trajectory_path": failures[0]["trajectory_path"],
            }
        )
    ranking.sort(key=lambda item: (-item["count"], item["task_id"]))
    return ranking[:limit]


def _scenario_trajectories(matrix: Dict[str, Any], scenario_label: str) -> list[dict[str, Any]]:
    return [item for item in matrix.get("trajectories", []) if item.get("scenario") == scenario_label]


def _adaptive_taxonomy(matrix: Dict[str, Any], scenario_label: str) -> Dict[str, int]:
    scenario = matrix["results"][scenario_label]
    return scenario["summary"]["adaptive"].get("failure_breakdown", {})


def _render_overall_matrix(handle, matrix: Dict[str, Any]) -> None:
    handle.write("## Overall Matrix\n\n")
    handle.write("| Scenario | Agent | Success Rate | Avg Calls | Avg Steps | Avg Runtime | Avg Parse Failures | Avg Fallbacks | Avg Replans | Avg Patches |\n")
    handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for scenario in matrix["scenarios"]:
        label = scenario["label"]
        for agent, stats in matrix["results"][label]["summary"].items():
            handle.write(
                f"| {label} | {agent} | {stats['success_rate']:.2f} | {stats['avg_calls']:.2f} | "
                f"{stats['avg_steps']:.2f} | {stats.get('avg_runtime', 0):.4f} | {stats.get('avg_parse_failures', 0):.2f} | "
                f"{stats.get('avg_fallbacks', 0):.2f} | {stats.get('avg_replans', 0):.2f} | {stats.get('avg_patches', 0):.2f} |\n"
            )


def _render_scenario_summaries(handle, matrix: Dict[str, Any]) -> None:
    handle.write("\n## Scenario Summaries\n\n")
    handle.write(
        "| Scenario | Agent | Success Rate | Success/Call | Recovery Success Rate | Avg Tokens | Budget Exhaustions | Failure Labels |\n"
    )
    handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for scenario in matrix["scenarios"]:
        label = scenario["label"]
        for agent, stats in matrix["results"][label]["summary"].items():
            failure_labels = ", ".join(sorted(stats.get("failure_breakdown", {}).keys())[:4]) or "none"
            handle.write(
                f"| {label} | {agent} | {stats.get('success_rate', 0):.2f} | {stats.get('success_per_call', 0):.4f} | "
                f"{stats.get('recovery_success_rate', 0):.2f} | {stats.get('avg_tokens', 0):.2f} | "
                f"{stats.get('budget_exhaustion_count', 0)} | {failure_labels} |\n"
            )


def _render_recovery_metrics(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Recovery Efficiency (`noise=0.2 tight`)\n\n")
    handle.write(
        "| Agent | Success/Call | Success/Token | Success/Runtime | Recovery Attempts | Recovery Success Rate | "
        "Avg Recovery Calls | Avg Recovery Tokens | Avg Recovery Runtime | Patched Success Rate | Replanned Success Rate |\n"
    )
    handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for agent, stats in matrix["results"][scenario_label]["summary"].items():
        handle.write(
            f"| {agent} | {stats.get('success_per_call', 0):.4f} | {stats.get('success_per_estimated_token', 0):.6f} | "
            f"{stats.get('success_per_runtime', 0):.4f} | {stats.get('recovery_attempt_count', 0)} | "
            f"{stats.get('recovery_success_rate', 0):.2f} | {stats.get('avg_recovery_cost_calls', 0):.2f} | "
            f"{stats.get('avg_recovery_cost_tokens', 0):.2f} | {stats.get('avg_recovery_cost_runtime', 0):.4f} | "
            f"{stats.get('patched_success_rate', 0):.2f} | {stats.get('replanned_success_rate', 0):.2f} |\n"
        )


def _render_task_type_matrix(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Task Type Success Rates (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Task Type | Success Rate |\n")
    handle.write("| --- | --- | --- |\n")
    for agent, rates in matrix["results"][scenario_label]["task_type_rates"].items():
        for task_type, success_rate in rates.items():
            handle.write(f"| {agent} | {task_type} | {success_rate:.2f} |\n")


def _render_failure_breakdown(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Failure Breakdown (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Failure Label | Count |\n")
    handle.write("| --- | --- | --- |\n")
    for agent, stats in matrix["results"][scenario_label]["summary"].items():
        breakdown = stats.get("failure_breakdown", {})
        if not breakdown:
            handle.write(f"| {agent} | none | 0 |\n")
            continue
        for label, count in sorted(breakdown.items()):
            handle.write(f"| {agent} | {label} | {count} |\n")


def _render_recovery_breakdown(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Recovery Action Breakdown (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Recommended Strategy | Actual Recovery Action | Count |\n")
    handle.write("| --- | --- | --- | --- |\n")
    for agent, stats in matrix["results"][scenario_label]["summary"].items():
        recommended = stats.get("recommended_strategy_breakdown", {})
        actual = stats.get("actual_recovery_action_breakdown", {})
        keys = sorted(set(recommended) | set(actual))
        if not keys:
            handle.write(f"| {agent} | none | none | 0 |\n")
            continue
        for key in keys:
            handle.write(f"| {agent} | {key if key in recommended else '-'} | {key if key in actual else '-'} | {max(recommended.get(key, 0), actual.get(key, 0))} |\n")


def _render_failure_propagation(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Failure Propagation Summary (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Propagation Rate | First Failure Stages | Recovered By Stage | Unrecovered By Stage |\n")
    handle.write("| --- | --- | --- | --- | --- |\n")
    for agent, stats in matrix["results"][scenario_label]["summary"].items():
        first = ", ".join(f"{key}:{value}" for key, value in sorted(stats.get("first_failure_stage_breakdown", {}).items())) or "none"
        recovered = ", ".join(f"{key}:{value}" for key, value in sorted(stats.get("recovered_by_stage", {}).items())) or "none"
        unrecovered = ", ".join(f"{key}:{value}" for key, value in sorted(stats.get("unrecovered_by_stage", {}).items())) or "none"
        handle.write(
            f"| {agent} | {stats.get('failure_propagation_rate', 0):.2f} | {first} | {recovered} | {unrecovered} |\n"
        )


def _render_stage_to_stage(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Stage To Stage Propagation (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Transition | Count |\n")
    handle.write("| --- | --- | --- |\n")
    for agent, stats in matrix["results"][scenario_label]["summary"].items():
        transitions = stats.get("stage_to_stage_propagation_summary", {})
        if not transitions:
            handle.write(f"| {agent} | none | 0 |\n")
            continue
        for transition, count in sorted(transitions.items()):
            handle.write(f"| {agent} | {transition} | {count} |\n")


def _render_task_type_stage_cross_table(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Task Type X Failure Stage (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Task Type | Failure Origin Count | Propagation Rate | First Failure Stages | Recovered By Stage | Unrecovered By Stage |\n")
    handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    scenario = matrix["results"][scenario_label]
    for agent, task_type_stats in scenario.get("task_type_stage_analysis", {}).items():
        if not task_type_stats:
            handle.write(f"| {agent} | none | 0 | 0.00 | none | none | none |\n")
            continue
        for task_type, stats in task_type_stats.items():
            first = ", ".join(f"{key}:{value}" for key, value in sorted(stats.get("first_failure_stage_breakdown", {}).items())) or "none"
            recovered = ", ".join(f"{key}:{value}" for key, value in sorted(stats.get("recovered_by_stage", {}).items())) or "none"
            unrecovered = ", ".join(f"{key}:{value}" for key, value in sorted(stats.get("unrecovered_by_stage", {}).items())) or "none"
            handle.write(
                f"| {agent} | {task_type} | {stats.get('with_failure_origin', 0)} | {stats.get('failure_propagation_rate', 0):.2f} | "
                f"{first} | {recovered} | {unrecovered} |\n"
            )


def _render_task_type_stage_strategy_cross_table(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Task Type X Stage X Strategy (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Task Type | Stage | Recovery Events | Recommended Strategies | Actual Recovery Actions |\n")
    handle.write("| --- | --- | --- | --- | --- | --- |\n")
    scenario = matrix["results"][scenario_label]
    for agent, task_type_stats in scenario.get("task_type_stage_strategy_analysis", {}).items():
        if not task_type_stats:
            handle.write(f"| {agent} | none | none | 0 | none | none |\n")
            continue
        for task_type, stats in task_type_stats.items():
            stages = sorted(
                set(stats.get("stage_event_count", {}))
                | set(stats.get("recommended_strategy_by_stage", {}))
                | set(stats.get("actual_recovery_action_by_stage", {}))
            )
            if not stages:
                handle.write(f"| {agent} | {task_type} | none | 0 | none | none |\n")
                continue
            for stage in stages:
                recommended = ", ".join(
                    f"{key}:{value}" for key, value in sorted(stats.get("recommended_strategy_by_stage", {}).get(stage, {}).items())
                ) or "none"
                actual = ", ".join(
                    f"{key}:{value}" for key, value in sorted(stats.get("actual_recovery_action_by_stage", {}).get(stage, {}).items())
                ) or "none"
                handle.write(
                    f"| {agent} | {task_type} | {stage} | {stats.get('stage_event_count', {}).get(stage, 0)} | {recommended} | {actual} |\n"
                )


def _render_budget_summary(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Budget Usage Summary (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Avg Calls | Avg Steps | Avg Tokens | Avg Runtime | Budget Exhaustions |\n")
    handle.write("| --- | --- | --- | --- | --- | --- |\n")
    for agent, stats in matrix["results"][scenario_label]["summary"].items():
        handle.write(
            f"| {agent} | {stats.get('avg_calls', 0):.2f} | {stats.get('avg_steps', 0):.2f} | "
            f"{stats.get('avg_tokens', 0):.2f} | {stats.get('avg_runtime', 0):.4f} | {stats.get('budget_exhaustion_count', 0)} |\n"
        )


def _render_parse_summary(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Parse And Recovery Summary (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Avg Parse Failures | Total Fallbacks | Total Replans | Total Patches |\n")
    handle.write("| --- | --- | --- | --- | --- |\n")
    for agent, stats in matrix["results"][scenario_label]["summary"].items():
        handle.write(
            f"| {agent} | {stats.get('avg_parse_failures', 0):.2f} | {stats.get('fallback_count', 0)} | "
            f"{stats.get('replan_count', 0)} | {stats.get('patch_count', 0)} |\n"
        )


def _render_retrieval_quality(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Retrieval Quality (`noise=0.2 tight`)\n\n")
    handle.write("| Agent | Retrieval Tasks | Hit Rate | Avg Source Coverage | Avg Term Coverage | Avg Noise Ratio | Evidence Usage Rate |\n")
    handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for agent, stats in matrix["results"][scenario_label]["summary"].items():
        handle.write(
            f"| {agent} | {stats.get('retrieval_task_count', 0)} | {stats.get('retrieval_hit_rate', 0):.2f} | "
            f"{stats.get('avg_retrieval_source_coverage', 0):.2f} | {stats.get('avg_retrieval_term_coverage', 0):.2f} | "
            f"{stats.get('avg_retrieval_noise_ratio', 0):.2f} | {stats.get('retrieval_evidence_usage_rate', 0):.2f} |\n"
        )


def _render_top_failures(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Top Failed Tasks (`noise=0.2 tight`)\n\n")
    handle.write("| Task ID | Task Type | Failures | Common Label | Trajectory |\n")
    handle.write("| --- | --- | --- | --- | --- |\n")
    for item in _top_failures(matrix, scenario_label):
        handle.write(
            f"| {item['task_id']} | {item['task_type']} | {item['count']} | {item['label']} | {item['trajectory_path']} |\n"
        )


def _render_typical_failures(handle, matrix: Dict[str, Any], scenario_label: str, limit: int = 5) -> None:
    handle.write("\n## Typical Failure Cases (`noise=0.2 tight`)\n\n")
    handle.write("| Task ID | Agent | First Failure Stage | Failure Label | Recommended Strategy | Actual Recovery Action | Budget Usage | Trajectory |\n")
    handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    rows = [item for item in _scenario_trajectories(matrix, scenario_label) if not item.get("success")]
    rows.sort(key=lambda item: (item.get("failure_label", ""), item.get("task_id", "")))
    if not rows:
        handle.write("| none | none | none | none | none | none | none |\n")
        return
    for item in rows[:limit]:
        budget_usage = item.get("budget_usage", {})
        budget_text = f"calls={budget_usage.get('calls', 0)}, tokens={budget_usage.get('tokens', 0)}, time={budget_usage.get('time', 0):.4f}"
        handle.write(
            f"| {item.get('task_id', '')} | {item.get('agent', '')} | {item.get('first_failure_stage', '') or '-'} | {item.get('failure_label', '')} | "
            f"{item.get('recommended_strategy', '') or '-'} | {item.get('actual_recovery_action', '') or '-'} | "
            f"{budget_text} | {item.get('trajectory_path', '')} |\n"
        )


def _render_replay_samples(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Single-Task Replay Samples (`noise=0.2 tight`)\n\n")
    rows = _scenario_trajectories(matrix, scenario_label)
    success_item = next((item for item in rows if item.get("success") and item.get("replay_summary")), None)
    failure_item = next((item for item in rows if not item.get("success") and item.get("replay_summary")), None)
    if success_item is None and failure_item is None:
        handle.write("No replay summaries were available.\n")
        return
    if success_item is not None:
        handle.write("### Success Sample\n\n")
        handle.write(success_item["replay_summary"].strip() + "\n\n")
        if success_item.get("replay_summary_path"):
            handle.write(f"- Summary Path: {success_item['replay_summary_path']}\n\n")
    if failure_item is not None:
        handle.write("### Failure Sample\n\n")
        handle.write(failure_item["replay_summary"].strip() + "\n\n")
        if failure_item.get("replay_summary_path"):
            handle.write(f"- Summary Path: {failure_item['replay_summary_path']}\n\n")


def _render_adaptive_taxonomy(handle, matrix: Dict[str, Any], scenario_label: str) -> None:
    handle.write("\n## Adaptive Failure Taxonomy (`noise=0.2 tight`)\n\n")
    taxonomy = _adaptive_taxonomy(matrix, scenario_label)
    if taxonomy:
        handle.write("| Label | Count |\n")
        handle.write("| --- | --- |\n")
        for label, count in sorted(taxonomy.items()):
            handle.write(f"| {label} | {count} |\n")
    else:
        handle.write("No adaptive failures recorded.\n")


def _write_latest_pointers(report_path: Path, snapshot: Dict[str, Any] | None, matrix: Dict[str, Any]) -> None:
    report_dir = reports_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / LATEST_REPORT_NAME).write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    if snapshot is not None:
        (report_dir / LATEST_CONFIG_NAME).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    summary_payload = {label: details["summary"] for label, details in matrix["results"].items()}
    (report_dir / LATEST_SUMMARY_NAME).write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def generate(matrix: Dict[str, Any], snapshot: Dict[str, Any] | None = None) -> str:
    run_dir = Path(matrix.get("run_dir", reports_dir()))
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.md"

    noisy_tight_label = "noise=0.2 tight"
    if noisy_tight_label not in matrix["results"]:
        noisy_tight_label = matrix["scenarios"][0]["label"]
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# AutoToolBench Report\n\n")
        if snapshot:
            handle.write(
                f"- llm={snapshot.get('llm_type', 'mock')}"
                f", model={snapshot.get('openai_model') or 'mock-json'}"
                f", temperature={snapshot.get('temperature', 0)}\n"
            )
            handle.write(f"- agent_version={snapshot.get('agent_version', 'unknown')}\n")
            handle.write(f"- run_dir={snapshot.get('run_dir', str(run_dir))}\n\n")
        handle.write("- Metrics note: parse failure rate is reported as average parse failures per task.\n")
        handle.write("- Replay note: failure, recovery, and budget fields below are sourced from persisted trajectory metadata.\n\n")
        _render_overall_matrix(handle, matrix)
        _render_scenario_summaries(handle, matrix)
        _render_recovery_metrics(handle, matrix, noisy_tight_label)
        _render_failure_breakdown(handle, matrix, noisy_tight_label)
        _render_recovery_breakdown(handle, matrix, noisy_tight_label)
        _render_failure_propagation(handle, matrix, noisy_tight_label)
        _render_stage_to_stage(handle, matrix, noisy_tight_label)
        _render_task_type_stage_cross_table(handle, matrix, noisy_tight_label)
        _render_task_type_stage_strategy_cross_table(handle, matrix, noisy_tight_label)
        _render_budget_summary(handle, matrix, noisy_tight_label)
        _render_parse_summary(handle, matrix, noisy_tight_label)
        _render_retrieval_quality(handle, matrix, noisy_tight_label)
        _render_task_type_matrix(handle, matrix, noisy_tight_label)
        _render_top_failures(handle, matrix, noisy_tight_label)
        _render_typical_failures(handle, matrix, noisy_tight_label)
        _render_replay_samples(handle, matrix, noisy_tight_label)
        _render_adaptive_taxonomy(handle, matrix, noisy_tight_label)

    _write_latest_pointers(report_path, snapshot, matrix)
    return str(report_path)
