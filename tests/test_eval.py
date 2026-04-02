import json
from collections import Counter

from autotoolbench.agent.adaptive_agent import AdaptiveAgent
from autotoolbench.agent.budget import BudgetController
from autotoolbench.agent.schema import StepRecord, Trajectory
from autotoolbench.data_gen import main as generate_data
from autotoolbench.env import validators
from autotoolbench.env.tasks import load_tasks
from autotoolbench.eval.ablation import ablate
from autotoolbench.eval.metrics import failure_profile, summarize
from autotoolbench.eval.replay import summarize_trajectory_markdown
from autotoolbench.eval.report import generate
from autotoolbench.eval.runner import evaluation_snapshot, run_agents_detailed, run_matrix
from autotoolbench.llm.mock import MockLLM
from autotoolbench.utils.paths import data_dir, reports_dir


def test_task_inventory_and_coverage():
    generate_data(seed=0)
    tasks = load_tasks()
    counts = Counter(task.task_type for task in tasks)
    assert len(tasks) >= 28
    assert set(counts) == {
        "single_tool_easy",
        "multi_tool_chain",
        "prerequisite_dependency",
        "tool_confusion",
        "args_brittle",
        "budget_tradeoff",
        "retrieval_heavy",
        "ambiguity_heavy",
        "long_horizon",
        "partial_success",
    }
    assert all(count >= 3 for count in counts.values())


def test_retrieval_and_ambiguity_tasks_are_present_and_counted():
    generate_data(seed=0)
    tasks = load_tasks()
    counts = Counter(task.task_type for task in tasks)
    assert counts["retrieval_heavy"] >= 3
    assert counts["ambiguity_heavy"] >= 3
    assert counts["long_horizon"] >= 3
    assert counts["partial_success"] >= 3


def test_matrix_report_contains_required_sections():
    generate_data(seed=0)
    matrix = run_matrix(["react", "plan", "adaptive"], seed=0)
    snapshot = evaluation_snapshot(["react", "plan", "adaptive"], seed=0, matrix=matrix)
    report_path = generate(matrix, snapshot=snapshot)
    report_text = open(report_path, encoding="utf-8").read()
    assert "noise=0.0" in report_text
    assert "noise=0.2 tight" in report_text
    assert "Top Failed Tasks" in report_text
    assert "Adaptive Failure Taxonomy" in report_text
    assert "Recovery Efficiency" in report_text
    assert "Failure Breakdown" in report_text
    assert "Recovery Action Breakdown" in report_text
    assert "Failure Propagation Summary" in report_text
    assert "Stage To Stage Propagation" in report_text
    assert "Task Type X Failure Stage" in report_text
    assert "Task Type X Stage X Strategy" in report_text
    assert "Budget Usage Summary" in report_text
    assert "Parse And Recovery Summary" in report_text
    assert "Retrieval Quality" in report_text
    assert "Typical Failure Cases" in report_text
    assert "Single-Task Replay Samples" in report_text
    assert "Scenario Summaries" in report_text
    assert "parse failure rate" in report_text.lower()
    config = json.loads((reports_dir() / "latest_config.json").read_text(encoding="utf-8"))
    assert config["task_count"] >= 22
    assert config["tool_exception_enabled"] is False
    assert "tool_schema_snapshot" in config
    assert config["llm_type"] == "mock"


def test_tight_noise_matrix_reports_consistent_metrics():
    generate_data(seed=0)
    matrix = run_matrix(["react", "plan", "adaptive"], seed=0)
    tight_summary = matrix["results"]["noise=0.2 tight"]["summary"]
    assert tight_summary["adaptive"]["avg_calls"] <= tight_summary["adaptive"]["avg_steps"]
    assert "failure_breakdown" in tight_summary["adaptive"]
    assert "budget_exhaustion_count" in tight_summary["adaptive"]
    assert tight_summary["adaptive"]["avg_parse_failures"] >= 0


def test_trajectory_index_contains_replay_friendly_fields():
    generate_data(seed=0)
    matrix = run_matrix(["adaptive"], seed=0)
    entry = next(item for item in matrix["trajectories"] if item["scenario"] == "noise=0.2 tight")
    assert "failure_label" in entry
    assert "recommended_strategy" in entry
    assert "actual_recovery_action" in entry
    assert "budget_usage" in entry
    assert "parse_failures" in entry
    assert "first_failure_stage" in entry
    assert "final_failure_stage" in entry
    assert "failure_propagated" in entry
    assert "replay_summary" in entry


def test_report_budget_and_failure_sections_match_summary():
    generate_data(seed=0)
    matrix = run_matrix(["adaptive"], seed=0)
    tight_summary = matrix["results"]["noise=0.2 tight"]["summary"]["adaptive"]
    failures = [item for item in matrix["failures"] if item["scenario"] == "noise=0.2 tight" and item["agent"] == "adaptive"]
    assert sum(tight_summary["failure_breakdown"].values()) == len(failures)
    snapshot = evaluation_snapshot(["adaptive"], seed=0, matrix=matrix)
    report_path = generate(matrix, snapshot=snapshot)
    report_text = open(report_path, encoding="utf-8").read()
    assert f"| adaptive | {tight_summary['avg_calls']:.2f} | {tight_summary['avg_steps']:.2f}" in report_text
    for label, count in sorted(tight_summary["failure_breakdown"].items()):
        assert f"| adaptive | {label} | {count} |" in report_text


def test_ablation_command_payload():
    generate_data(seed=0)
    res = ablate(seed=0, noise=0.2, budget_preset="tight")
    assert "adaptive" in res
    assert "no_replan" in res
    assert "no_memory" in res
    assert "weak_validation" in res


def test_new_ablations_enter_summary_and_matrix():
    generate_data(seed=0)
    details = run_agents_detailed(["adaptive", "no_memory", "weak_validation"], seed=0, noise=0.0, budget_preset="loose", scenario_label="ablation-check")
    assert "no_memory" in details["summary"]
    assert "weak_validation" in details["summary"]
    assert all(item["agent"] in {"adaptive", "no_memory", "weak_validation"} for item in details["trajectories"])
    assert "task_type_stage_analysis" in details
    assert "task_type_stage_strategy_analysis" in details
    assert "adaptive" in details["task_type_stage_analysis"]


def test_new_validator_is_explainable():
    generate_data(seed=0)
    result = summarize([])
    assert result["total"] == 0
    detailed = validators.run_validator(
        "file_json_array_contains",
        {"path": "missing.json", "records": [{"path": "invoice_casebook.txt", "line": 1}]},
    )
    assert detailed.ok is False
    assert detailed.validator == "file_json_array_contains"


def test_partial_success_validator_status_is_explainable():
    generate_data(seed=0)
    partial_path = data_dir() / "partial_validator_probe.json"
    partial_path.write_text('[{"name": "Bob", "team": "platform"}]\n', encoding="utf-8")
    result = validators.run_validator(
        "file_json_quality",
        {"path": "partial_validator_probe.json", "required_keys": ["name", "age", "team"], "min_items": 1},
    )
    assert result.ok is False
    assert result.status == "partial_success"
    assert "partial" in result.message.lower()


def test_retrieval_heavy_task_can_run_through_agent():
    generate_data(seed=0)
    task = next(task for task in load_tasks() if task.task_type == "retrieval_heavy")
    traj = AdaptiveAgent(MockLLM(seed=0, noise=0.0), budget=BudgetController.from_preset("loose")).run(task, budget_mode="loose")
    assert traj.success is True
    assert traj.metadata["validation"]["status"] == "full_success"


def test_retrieval_metrics_are_summarized_from_real_trajectories():
    generate_data(seed=0)
    details = run_agents_detailed(["adaptive"], seed=0, noise=0.0, budget_preset="loose", scenario_label="noise=0.0")
    summary = details["summary"]["adaptive"]
    assert summary["retrieval_task_count"] >= 1
    assert 0.0 <= summary["retrieval_hit_rate"] <= 1.0
    assert 0.0 <= summary["avg_retrieval_source_coverage"] <= 1.0
    assert 0.0 <= summary["avg_retrieval_term_coverage"] <= 1.0
    assert 0.0 <= summary["retrieval_evidence_usage_rate"] <= 1.0


def test_ambiguity_heavy_task_can_run_through_agent():
    generate_data(seed=0)
    task = next(task for task in load_tasks() if task.task_type == "ambiguity_heavy")
    traj = AdaptiveAgent(MockLLM(seed=0, noise=0.0), budget=BudgetController.from_preset("loose")).run(task, budget_mode="loose")
    assert traj.success is True


def test_long_horizon_task_can_run_through_agent():
    generate_data(seed=0)
    task = next(task for task in load_tasks() if task.task_type == "long_horizon")
    traj = AdaptiveAgent(MockLLM(seed=0, noise=0.0), budget=BudgetController.from_preset("loose")).run(task, budget_mode="loose")
    assert traj.success is True
    assert traj.metadata["validation"]["status"] == "full_success"


def test_advanced_recovery_metrics_are_computed_with_clear_denominators():
    recovered_success = Trajectory(
        task_id="R1",
        success=True,
        steps=[
            StepRecord(
                timestamp=1.0,
                subgoal="bad step",
                tool="sql_query",
                input={},
                output=[],
                failure_label="BAD_TOOL_ARGS",
                actual_recovery_action="patch_args",
                budget={"calls": 1, "steps": 1, "time": 1.0, "tokens": 100},
                metadata={},
            ),
            StepRecord(
                timestamp=2.0,
                subgoal="patched step",
                tool="sql_query",
                input={},
                output=[{"id": 1}],
                budget={"calls": 3, "steps": 3, "time": 5.0, "tokens": 300},
                metadata={},
            ),
        ],
        metadata={
            "budget_usage": {"calls": 3, "steps": 3, "time": 5.0, "tokens": 300},
            "patch_count": 1,
            "replan_count": 0,
            "fallback_count": 0,
            "parse_failures": 0,
            "recovery_events": [
                {
                    "failure_label": "BAD_TOOL_ARGS",
                    "recommended_strategy": "patch_args",
                    "actual_recovery_action": "patch_args",
                    "recovery_reason": "patch and continue",
                }
            ],
        },
    )
    recovered_fail = Trajectory(
        task_id="R2",
        success=False,
        steps=[
            StepRecord(
                timestamp=1.0,
                subgoal="wrong plan",
                tool="noop",
                input={},
                output="noop",
                failure_label="PLAN_MISMATCH",
                actual_recovery_action="replan",
                budget={"calls": 1, "steps": 1, "time": 2.0, "tokens": 200},
                metadata={},
            ),
            StepRecord(
                timestamp=2.0,
                subgoal="replanned but still failed",
                tool="file_write",
                input={},
                output="written",
                budget={"calls": 4, "steps": 4, "time": 8.0, "tokens": 500},
                metadata={},
            ),
        ],
        metadata={
            "failure_label": "VALIDATION_FAILED",
            "budget_usage": {"calls": 4, "steps": 4, "time": 8.0, "tokens": 500},
            "patch_count": 0,
            "replan_count": 1,
            "fallback_count": 0,
            "parse_failures": 0,
            "recovery_events": [
                {
                    "failure_label": "PLAN_MISMATCH",
                    "recommended_strategy": "replan",
                    "actual_recovery_action": "replan",
                    "recovery_reason": "replan and continue",
                }
            ],
        },
    )
    no_recovery_success = Trajectory(
        task_id="R3",
        success=True,
        steps=[],
        metadata={
            "budget_usage": {"calls": 2, "steps": 2, "time": 4.0, "tokens": 200},
            "patch_count": 0,
            "replan_count": 0,
            "fallback_count": 0,
            "parse_failures": 0,
            "recovery_events": [],
        },
    )

    summary = summarize([recovered_success, recovered_fail, no_recovery_success], runtimes=[5.5, 8.5, 4.5])

    assert summary["recovery_attempt_count"] == 2
    assert summary["recovery_task_count"] == 2
    assert summary["recovery_success_count"] == 1
    assert summary["recovery_success_rate"] == 0.5
    assert summary["patched_success_rate"] == 1.0
    assert summary["replanned_success_rate"] == 0.0
    assert summary["success_per_call"] == 2 / 9
    assert summary["success_per_estimated_token"] == 2 / 1000
    assert summary["success_per_runtime"] == 2 / 17.0
    assert summary["avg_recovery_cost_calls"] == 2.5
    assert summary["avg_recovery_cost_tokens"] == 250.0
    assert summary["avg_recovery_cost_runtime"] == 5.0
    assert summary["first_failure_stage_breakdown"]["action_generation"] == 1
    assert summary["first_failure_stage_breakdown"]["planner"] == 1
    assert summary["recovered_by_stage"]["action_generation"] == 1
    assert summary["unrecovered_by_stage"]["planner"] == 1
    assert summary["stage_to_stage_propagation_summary"]["action_generation->recovered"] == 1
    assert summary["stage_to_stage_propagation_summary"]["planner->validator"] == 1


def test_failure_profile_maps_stages_and_recovery_consistently():
    traj = Trajectory(
        task_id="FP1",
        success=False,
        steps=[
            StepRecord(
                timestamp=1.0,
                subgoal="bad action json",
                tool="sql_query",
                input={},
                output=None,
                failure_label="JSON_MALFORMED",
                budget={"calls": 1, "steps": 1, "time": 1.0, "tokens": 80},
                metadata={"fallback_reason": "action_json_invalid", "parse_failures": 1},
            ),
            StepRecord(
                timestamp=2.0,
                subgoal="late validation miss",
                tool="file_write",
                input={},
                output="written",
                budget={"calls": 2, "steps": 2, "time": 2.0, "tokens": 120},
                metadata={},
            ),
        ],
        metadata={
            "failure_label": "VALIDATION_FAILED",
            "recovery_events": [
                {
                    "failure_label": "JSON_MALFORMED",
                    "recommended_strategy": "retry_safe",
                    "actual_recovery_action": "safe_fallback",
                    "recovery_reason": "fallback",
                }
            ],
        },
    )
    profile = failure_profile(traj)
    assert profile["first_failure_stage"] == "action_generation"
    assert profile["final_failure_stage"] == "validator"
    assert profile["failure_recovered"] is False
    assert profile["recovery_attempt_count"] == 1
    assert profile["failure_propagated"] is True


def test_replay_summary_formats_success_and_failure_cases():
    success_traj = Trajectory(
        task_id="RS1",
        success=True,
        steps=[
            StepRecord(
                timestamp=1.0,
                subgoal="query",
                tool="sql_query",
                input={"query": "SELECT 1"},
                output=[{"value": 1}],
                budget={"calls": 1, "steps": 1, "time": 0.2, "tokens": 50},
                metadata={"referenced_memory_keys": ["brief"]},
            )
        ],
        metadata={
            "task_type": "single_tool_easy",
            "budget_usage": {"calls": 1, "steps": 1, "time": 0.2, "tokens": 50},
            "validation": {"validator": "sql_result_equals", "message": "matched expected rows"},
            "recovery_events": [],
        },
    )
    failure_traj = Trajectory(
        task_id="RS2",
        success=False,
        steps=[
            StepRecord(
                timestamp=1.0,
                subgoal="wrong plan",
                tool="noop",
                input={},
                output="noop",
                error="noop",
                failure_label="PLAN_MISMATCH",
                actual_recovery_action="replan",
                recovery_reason="The noop step did not advance the task.",
                budget={"calls": 1, "steps": 1, "time": 0.3, "tokens": 60},
                metadata={},
            )
        ],
        metadata={
            "task_type": "ambiguity_heavy",
            "failure_label": "VALIDATION_FAILED",
            "first_failure_stage": "planner",
            "budget_usage": {"calls": 1, "steps": 1, "time": 0.3, "tokens": 60},
            "validation": {"validator": "file_contains_regex", "message": "pattern missing"},
            "recovery_events": [
                {
                    "failure_label": "PLAN_MISMATCH",
                    "recommended_strategy": "replan",
                    "actual_recovery_action": "replan",
                    "recovery_reason": "Need a different tool path.",
                }
            ],
        },
    )
    success_md = summarize_trajectory_markdown(success_traj)
    failure_md = summarize_trajectory_markdown(failure_traj)
    assert "Replay Summary: RS1" in success_md
    assert "Outcome: `success`" in success_md
    assert "Key Memory References: `brief`" in success_md
    assert "matched expected rows" in success_md
    assert "Replay Summary: RS2" in failure_md
    assert "Outcome: `failure`" in failure_md
    assert "Failure first appeared at step: `1`" in failure_md
    assert "Recovery attempts made: `replan`" in failure_md
    assert "pattern missing" in failure_md


def test_planner_stage_is_detected_from_component_stats():
    traj = Trajectory(
        task_id="FP2",
        success=True,
        steps=[],
        metadata={
            "component_stats": {"planner": [{"parse_failures": 1, "fallback_used": True, "validation_errors": ["bad"]}]},
            "recovery_events": [],
        },
    )
    profile = failure_profile(traj)
    assert profile["first_failure_stage"] == "planner"
    assert profile["final_failure_stage"] == "recovered"
    assert profile["failure_recovered"] is True


def test_no_recovery_tasks_do_not_pollute_recovery_metrics():
    summary = summarize(
        [
            Trajectory(
                task_id="N1",
                success=True,
                steps=[],
                metadata={
                    "budget_usage": {"calls": 0, "steps": 0, "time": 0.0, "tokens": 0},
                    "patch_count": 0,
                    "replan_count": 0,
                    "fallback_count": 0,
                    "parse_failures": 0,
                    "recovery_events": [],
                },
            )
        ],
        runtimes=[0.0],
    )
    assert summary["recovery_attempt_count"] == 0
    assert summary["recovery_success_count"] == 0
    assert summary["recovery_success_rate"] == 0.0
    assert summary["avg_recovery_cost_calls"] == 0.0
    assert summary["patched_success_rate"] == 0.0
    assert summary["replanned_success_rate"] == 0.0


def test_regression_summary_fields_remain_stable_for_seed_zero():
    generate_data(seed=0)
    details = run_agents_detailed(["adaptive"], seed=0, noise=0.0, budget_preset="loose", scenario_label="regression")
    summary = details["summary"]["adaptive"]
    required = {
        "success_rate",
        "avg_calls",
        "avg_steps",
        "avg_runtime",
        "avg_parse_failures",
        "failure_breakdown",
        "first_failure_stage_breakdown",
        "recovered_by_stage",
        "unrecovered_by_stage",
        "failure_propagation_rate",
        "stage_to_stage_propagation_summary",
        "recommended_strategy_breakdown",
        "actual_recovery_action_breakdown",
        "budget_exhaustion_count",
    }
    assert required.issubset(summary.keys())
    trajectory = details["trajectories"][0]
    assert {
        "failure_label",
        "first_failure_stage",
        "final_failure_stage",
        "failure_recovered",
        "recovery_attempt_count",
        "failure_propagated",
        "replay_summary",
        "recommended_strategy",
        "actual_recovery_action",
        "budget_usage",
    }.issubset(trajectory.keys())


def test_task_type_stage_cross_table_is_aggregated_consistently():
    generate_data(seed=0)
    details = run_agents_detailed(["adaptive"], seed=0, noise=0.0, budget_preset="loose", scenario_label="cross-check")
    cross = details["task_type_stage_analysis"]["adaptive"]
    assert "retrieval_heavy" in cross
    assert "ambiguity_heavy" in cross
    retrieval = cross["retrieval_heavy"]
    assert {"with_failure_origin", "failure_propagation_rate", "first_failure_stage_breakdown", "recovered_by_stage", "unrecovered_by_stage"}.issubset(
        retrieval.keys()
    )
    total_failure_origin = sum(
        stats["with_failure_origin"] for stats in details["task_type_stage_analysis"]["adaptive"].values()
    )
    summary_total = sum(details["summary"]["adaptive"]["first_failure_stage_breakdown"].values())
    assert total_failure_origin == summary_total


def test_task_type_stage_strategy_cross_table_is_present_and_explainable():
    generate_data(seed=0)
    details = run_agents_detailed(["adaptive"], seed=0, noise=0.0, budget_preset="loose", scenario_label="strategy-cross-check")
    cross = details["task_type_stage_strategy_analysis"]["adaptive"]
    assert "retrieval_heavy" in cross
    retrieval = cross["retrieval_heavy"]
    assert {"stage_event_count", "recommended_strategy_by_stage", "actual_recovery_action_by_stage"}.issubset(retrieval.keys())
    stage_count = sum(retrieval["stage_event_count"].values())
    recommended_total = sum(
        count for counter in retrieval["recommended_strategy_by_stage"].values() for count in counter.values()
    )
    actual_total = sum(
        count for counter in retrieval["actual_recovery_action_by_stage"].values() for count in counter.values()
    )
    assert stage_count >= recommended_total
    assert stage_count >= actual_total


def test_report_includes_task_type_stage_rows():
    generate_data(seed=0)
    matrix = run_matrix(["adaptive"], seed=0)
    snapshot = evaluation_snapshot(["adaptive"], seed=0, matrix=matrix)
    report_path = generate(matrix, snapshot=snapshot)
    report_text = open(report_path, encoding="utf-8").read()
    assert "| adaptive | retrieval_heavy |" in report_text
    assert "| adaptive | long_horizon |" in report_text
    assert "| adaptive | partial_success |" in report_text
    assert "Task Type X Stage X Strategy" in report_text
    assert "### Success Sample" in report_text
