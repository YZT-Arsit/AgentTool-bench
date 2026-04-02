import pytest
from typer.testing import CliRunner

from autotoolbench.agent.adaptive_agent import AdaptiveAgent
from autotoolbench.agent.budget import BudgetController
from autotoolbench.agent.executor import Executor
from autotoolbench.agent.json_utils import (
    extract_json,
    validate_action,
    validate_plan,
    validate_reflection,
)
from autotoolbench.agent.planner import Planner
from autotoolbench.agent.reflector import Reflector
from autotoolbench.cli import app
from autotoolbench.data_gen import main as generate_data
from autotoolbench.env.tasks import Task, get_task
from autotoolbench.llm.mock import MockLLM
from autotoolbench.tools.registry import get
from autotoolbench.utils.paths import reports_dir


def test_budget_controller_presets():
    tight = BudgetController.from_preset("tight")
    loose = BudgetController.from_preset("loose")
    assert tight.max_calls < loose.max_calls
    assert tight.max_steps < loose.max_steps


def test_reflector_uses_injection_metadata():
    reflector = Reflector()
    assert reflector.classify({"metadata": {"injection_type": "MISSING_STEP"}}) == "MISSING_PREREQUISITE"
    assert reflector.classify({"metadata": {"injection_type": "TOOL_ARGS_ERROR"}}) == "BAD_TOOL_ARGS"


def test_planner_uses_budget_specific_hints():
    generate_data(seed=0)
    task = get_task("T021")
    planner = Planner(MockLLM(seed=0, noise=0))
    tight_plan = planner.plan(task, budget_mode="tight")
    loose_plan = planner.plan(task, budget_mode="loose")
    assert tight_plan != loose_plan
    assert "REQ-404" in tight_plan[0]["args"]["pattern"]
    assert "invoice fetched" in loose_plan[0]["args"]["pattern"]


def test_noise_injection_is_reproducible():
    llm_a = MockLLM(seed=0, noise=0.2)
    llm_b = MockLLM(seed=0, noise=0.2)
    action = {"tool": "sql_query", "args": {"query": "SELECT * FROM users"}}
    corrupted_a, meta_a = llm_a.corrupt_action(action, task_id="T999", step_index=1, available_tools=["sql_query", "file_write", "noop"])
    corrupted_b, meta_b = llm_b.corrupt_action(action, task_id="T999", step_index=1, available_tools=["sql_query", "file_write", "noop"])
    assert corrupted_a == corrupted_b
    assert meta_a == meta_b


def test_json_utils_extract_and_validate_structured_payloads():
    wrapped = 'preface ```json {"tool":"sql_query","args":{"query":"SELECT 1"},"rationale":"test"} ``` tail'
    action = extract_json(wrapped)
    assert validate_action(action)[0] is True

    plan = {
        "steps": [
            {
                "step_id": "S1",
                "subgoal": "query",
                "tool": "sql_query",
                "args_hint": {"query": "SELECT 1"},
                "success_criteria": ["rows"],
                "optional": False,
                "branch_group": "BG1",
                "branch_id": "A",
                "independent": True,
            }
        ]
    }
    assert validate_plan(plan)[0] is True

    reflection = {
        "label": "BAD_TOOL_ARGS",
        "explanation": "bad args",
        "recommended_strategy": "patch_args",
        "fix_action": "patch",
        "replan_needed": False,
        "recovery_reason": "Patch the query",
        "patch": {"tool": "sql_query", "args": {"query": "SELECT 1"}},
    }
    assert validate_reflection(reflection)[0] is True


def test_mock_json_error_triggers_fallback_without_crashing():
    generate_data(seed=0)
    task = get_task("T001")
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0, json_error_rate=1.0))
    traj = agent.run(task, seed=0, noise=0.0, budget_mode="tight")
    assert traj.metadata["fallback_count"] >= 1
    assert traj.steps
    assert traj.steps[0].metadata["fallback_used"] is True


def test_executor_working_memory_tracks_named_outputs():
    generate_data(seed=0)
    task = get_task("T001")
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0))
    traj = agent.run(task, seed=0, noise=0.0, budget_mode="tight")
    assert traj.success is True
    assert "S1" in traj.memory
    assert "last_output" in traj.memory
    assert traj.memory["S1"].value_type == "sql_result"
    assert traj.steps[0].memory_delta["S1"]["value"] == traj.steps[0].output
    assert traj.steps[0].metadata["memory_before"] == {}
    assert "S1" in traj.steps[0].metadata["memory_after"]


def test_memory_reference_and_provenance_are_recorded():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T903",
            "instruction": "read then write via memory",
            "validator": "file_contains_regex",
            "validator_params": {"path": "memory_copy.txt", "patterns": ["REQ-404", "payments-api"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    plan = [
        {"step_id": "S1", "subgoal": "read source", "tool": "file_read", "args": {"path": "incident_brief.txt"}, "save_as": "incident_text"},
        {"step_id": "S2", "subgoal": "write copy", "tool": "file_write", "args": {"path": "memory_copy.txt", "content": "$memory:incident_text"}},
    ]

    first = executor.execute_step(task, plan[0], budget, 0, "tight", "test")
    second = executor.execute_step(task, plan[1], budget, 1, "tight", "test", last_obs=first.steps[-1].output)

    assert first.steps[0].memory_delta["incident_text"]["value_type"] == "file_text"
    assert second.steps[0].metadata["referenced_memory_keys"] == ["incident_text"]
    assert second.steps[0].metadata["memory_before"]["incident_text"]["source_tool"] == "file_read"
    assert second.steps[0].input["content"].startswith("Incident: REQ-404")


def test_no_memory_ablation_disables_memory_slots():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T906",
            "instruction": "read then write via memory",
            "validator": "file_contains_regex",
            "validator_params": {"path": "memory_disabled.txt", "patterns": ["REQ-404"]},
            "task_type": "multi_tool_chain",
        }
    )
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0), disable_memory=True)
    monkey_plan = [
        {"step_id": "S1", "subgoal": "read", "tool": "file_read", "args": {"path": "incident_brief.txt"}, "save_as": "brief"},
        {"step_id": "S2", "subgoal": "write", "tool": "file_write", "args": {"path": "memory_disabled.txt", "content": "$memory:brief"}},
    ]
    agent.planner.plan = lambda *args, **kwargs: monkey_plan
    traj = agent.run(task, budget_mode="tight")
    assert traj.success is False
    assert traj.metadata["failure_label"] in {"BAD_TOOL_ARGS", "VALIDATION_FAILED", "TOOL_EXECUTION_FAILED"}
    assert traj.steps[0].memory_delta == {}
    assert traj.memory == {}


def test_missing_memory_key_returns_clear_error():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T904",
            "instruction": "write missing memory",
            "validator": "file_contains_regex",
            "validator_params": {"path": "missing_memory.txt", "patterns": ["unused"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {"step_id": "S1", "subgoal": "write missing memory", "tool": "file_write", "args": {"path": "missing_memory.txt", "content": "$memory:not_here"}}
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    assert traj.steps[0].error == "missing_memory_key:not_here"
    assert traj.steps[0].metadata["memory_resolution_error"] == "Missing memory key: not_here"
    assert traj.steps[0].metadata["referenced_memory_keys"] == ["not_here"]


def test_argument_constraint_validation_blocks_missing_required_arg():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T913",
            "instruction": "invalid file write",
            "validator": "file_contains_regex",
            "validator_params": {"path": "unused.txt", "patterns": ["x"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {"step_id": "S1", "subgoal": "write without path", "tool": "file_write", "args": {"content": "hello"}}
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    record = traj.steps[0]
    assert record.error == "missing_required_argument"
    assert record.metadata["failure_label"] == "ARGUMENT_CONSTRAINT_VIOLATION"
    assert "argument_validation" in record.metadata
    assert "path" in record.metadata["argument_validation"]["message"]


def test_memory_type_mismatch_is_detected_before_tool_run():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T914",
            "instruction": "memory type mismatch",
            "validator": "file_contains_regex",
            "validator_params": {"path": "unused.txt", "patterns": ["x"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    first = executor.execute_step(
        task,
        {"step_id": "S1", "subgoal": "collect lines", "tool": "log_search", "args": {"pattern": "REQ-404"}, "save_as": "lines"},
        budget,
        0,
        "tight",
        "test",
    )
    second = executor.execute_step(
        task,
        {"step_id": "S2", "subgoal": "use log lines as sql", "tool": "sql_query", "args": {"query": "$memory:lines"}},
        budget,
        1,
        "tight",
        "test",
        last_obs=first.steps[-1].output,
    )
    record = second.steps[0]
    assert record.error == "memory_type_mismatch"
    assert record.metadata["failure_label"] == "MEMORY_TYPE_MISMATCH"
    assert record.metadata["argument_validation"]["details"]["memory_type"] == "log_lines"


def test_downstream_tool_io_mismatch_gets_clear_label():
    executor = Executor(MockLLM(seed=0, noise=0.0))
    validation = executor._validate_tool_args(
        get("file_write"),
        "file_write",
        {"path": "bad.txt", "content": object()},
        {"path": "bad.txt", "content": object()},
    )
    assert validation["ok"] is False
    assert validation["error_code"] in {"argument_type_mismatch", "downstream_input_incompatible"}
    assert validation["label"] in {"ARGUMENT_CONSTRAINT_VIOLATION", "TOOL_IO_MISMATCH"}


def test_planner_exposes_hardened_tool_schema():
    planner = Planner(MockLLM(seed=0, noise=0.0))
    schema = planner._tool_schema()
    assert schema["sql_query"]["read_only"] is True
    assert schema["file_write"]["mutating"] is True
    assert schema["file_write"]["allowed_memory_types"]["content"]
    assert schema["log_search"]["output_type"] == "log_lines"
    assert schema["doc_search"]["output_type"] == "retrieval_results"


def test_retrieval_results_enter_typed_memory_slots():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T916",
            "instruction": "retrieve invoice evidence",
            "validator": "file_contains_regex",
            "validator_params": {"path": "unused.txt", "patterns": ["x"]},
            "task_type": "retrieval_heavy",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    traj = executor.execute_step(
        task,
        {
            "step_id": "S1",
            "subgoal": "retrieve invoice evidence",
            "tool": "doc_search",
            "args": {"query": "INV-9 ownership evidence", "files": ["invoice_casebook.txt", "incident_brief.txt"], "top_k": 2},
            "save_as": "invoice_evidence",
        },
        budget,
        0,
        "tight",
        "test",
    )
    assert traj.steps[0].error is None
    assert "invoice_evidence" in traj.memory
    assert traj.memory["invoice_evidence"].value_type == "retrieval_results"


def test_branch_aware_execution_records_branch_metadata_and_merge_summary():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T917",
            "instruction": "collect independent evidence branches and merge them",
            "validator": "file_contains_regex",
            "validator_params": {"path": "branch_lines.json", "patterns": ["REQ-404", "payments-api"]},
            "task_type": "long_horizon",
        }
    )
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0), budget=BudgetController.from_preset("loose"))
    branch_plan = [
        {
            "step_id": "S1",
            "subgoal": "fetch live incident lines",
            "tool": "log_search",
            "args": {"pattern": "REQ-404|payments-api"},
            "save_as": "live_lines",
            "branch_group": "BG1",
            "branch_id": "A",
            "independent": True,
        },
        {
            "step_id": "S2",
            "subgoal": "read incident brief",
            "tool": "file_read",
            "args": {"path": "incident_brief.txt"},
            "save_as": "brief_text",
            "branch_group": "BG1",
            "branch_id": "B",
            "independent": True,
        },
        {
            "step_id": "S3",
            "subgoal": "merge branch evidence into output",
            "tool": "file_write",
            "args": {"path": "branch_lines.json", "content": "$memory:live_lines"},
            "merge_into": "BG1",
            "merge_requirements": ["A", "B"],
        },
    ]
    agent.planner.plan = lambda *args, **kwargs: branch_plan
    traj = agent.run(task, budget_mode="loose")
    assert traj.success is True
    assert "BG1" in traj.metadata["branch_groups"]
    branch_meta = traj.metadata["branch_groups"]["BG1"]
    assert sorted(branch_meta["succeeded_branches"]) == ["A", "B"]
    assert "brief_text" in branch_meta["merged_memory_keys"]
    assert "live_lines" in branch_meta["merged_memory_keys"]
    merge_step = traj.steps[-1]
    assert merge_step.merge_point is True
    assert merge_step.metadata["merge_into"] == "BG1"
    assert merge_step.merge_summary["branch_group"] == "BG1"
    assert traj.steps[0].branch_id == "A"
    assert traj.steps[1].branch_id == "B"


def test_branch_failure_is_clearly_marked():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T918",
            "instruction": "branch failure task",
            "validator": "file_contains_regex",
            "validator_params": {"path": "branch_failure.json", "patterns": ["REQ-404"]},
            "task_type": "long_horizon",
        }
    )
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0), budget=BudgetController.from_preset("loose"))
    branch_plan = [
        {
            "step_id": "S1",
            "subgoal": "bad branch",
            "tool": "missing_tool",
            "args": {},
            "branch_group": "BG2",
            "branch_id": "A",
            "independent": True,
        },
        {
            "step_id": "S2",
            "subgoal": "good branch",
            "tool": "log_search",
            "args": {"pattern": "REQ-404"},
            "save_as": "good_lines",
            "branch_group": "BG2",
            "branch_id": "B",
            "independent": True,
        },
        {
            "step_id": "S3",
            "subgoal": "merge branch output",
            "tool": "file_write",
            "args": {"path": "branch_failure.json", "content": "$memory:good_lines"},
            "merge_into": "BG2",
            "merge_requirements": ["A", "B"],
        },
    ]
    agent.planner.plan = lambda *args, **kwargs: branch_plan
    traj = agent.run(task, budget_mode="loose")
    assert traj.success is False
    assert traj.metadata["failure_label"] == "TOOL_NOT_FOUND"
    failed_step = traj.steps[0]
    assert failed_step.branch_group == "BG2"
    assert failed_step.branch_id == "A"
    assert failed_step.failure_label == "TOOL_NOT_FOUND"
    assert failed_step.metadata["branch_failure"] is True


def test_dangerous_write_path_is_blocked_by_safety_guardrail():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T911",
            "instruction": "attempt protected write",
            "validator": "file_contains_regex",
            "validator_params": {"path": "unused.txt", "patterns": ["x"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {"step_id": "S1", "subgoal": "write protected file", "tool": "file_write", "args": {"path": ".env", "content": "secret=1"}}
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    record = traj.steps[0]
    assert record.error == "safety_blocked"
    assert record.action_allowed is False
    assert record.safety_decision == "blocked"
    assert "protected" in (record.safety_reason or "").lower() or "hidden" in (record.safety_reason or "").lower()
    assert record.tool_risk_level == "medium"


def test_non_read_only_sql_is_blocked_by_safety_guardrail():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T912",
            "instruction": "attempt dangerous sql",
            "validator": "file_contains_regex",
            "validator_params": {"path": "unused.txt", "patterns": ["x"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {"step_id": "S1", "subgoal": "dangerous sql", "tool": "sql_query", "args": {"query": "DROP TABLE users"}}
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    record = traj.steps[0]
    assert record.error == "safety_blocked"
    assert record.action_allowed is False
    assert record.safety_decision == "blocked"
    assert "read-only" in (record.safety_reason or "").lower() or "non-read-only" in (record.safety_reason or "").lower()
    assert record.tool_risk_level == "low"


def test_normal_task_records_allowed_safety_decision():
    generate_data(seed=0)
    task = get_task("T001")
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {"step_id": "S1", "subgoal": "query users", "tool": "sql_query", "args": {"query": "SELECT name FROM users ORDER BY id"}}
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    record = traj.steps[0]
    assert record.error is None
    assert record.action_allowed is True
    assert record.safety_decision == "allowed"
    assert record.metadata["safety_decision"] == "allowed"
    assert record.metadata["action_allowed"] is True


def test_executor_ranks_candidates_and_prefers_compatible_action(monkeypatch):
    generate_data(seed=0)
    task = get_task("T001")
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {
        "step_id": "S1",
        "subgoal": "query users",
        "tool": "sql_query",
        "args": {"query": "SELECT name FROM users ORDER BY id"},
        "save_as": "rows",
    }

    def fake_decide_action(*args, **kwargs):
        executor.last_trace = {
            "llm_raw_text": '{"tool":"noop","args":{}}',
            "parsed_json": {"tool": "noop", "args": {}},
            "validation_errors": [],
            "fallback_used": False,
            "fallback_reason": "",
            "parse_failures": 0,
            "attempts": [],
            "estimated_tokens": 32,
            "referenced_memory_keys": [],
            "injection_metadata": None,
        }
        return {"tool": "noop", "args": {}, "rationale": "bad suggestion"}

    monkeypatch.setattr(executor, "decide_action", fake_decide_action)
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    record = traj.steps[0]
    assert len(record.candidate_actions) >= 2
    assert record.chosen_action.tool == "sql_query"
    assert record.selection_reason
    assert record.metadata["chosen_action"]["tool"] == "sql_query"


def test_high_cost_candidate_is_not_selected_when_score_is_worse(monkeypatch):
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T909",
            "instruction": "write a small marker",
            "validator": "file_contains_regex",
            "validator_params": {"path": "cost_sensitive.txt", "patterns": ["done"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {
        "step_id": "S1",
        "subgoal": "write small file",
        "tool": "file_write",
        "args": {"path": "cost_sensitive.txt", "content": "done"},
    }

    def fake_decide_action(*args, **kwargs):
        executor.last_trace = {
            "llm_raw_text": "",
            "parsed_json": {},
            "validation_errors": [],
            "fallback_used": False,
            "fallback_reason": "",
            "parse_failures": 0,
            "attempts": [],
            "estimated_tokens": 20,
            "referenced_memory_keys": [],
            "injection_metadata": None,
        }
        return {"tool": "file_write", "args": {"path": "cost_sensitive.txt", "content": "X" * 20000}, "rationale": "expensive"}

    monkeypatch.setattr(executor, "decide_action", fake_decide_action)
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    record = traj.steps[0]
    expensive_score = next(score for score in record.action_scores if score.candidate_id == "C1")
    chosen_score = next(score for score in record.action_scores if score.candidate_id == record.chosen_action.candidate_id)
    assert record.chosen_action.args["content"] == "done"
    assert expensive_score.estimated_budget_cost > chosen_score.estimated_budget_cost
    assert expensive_score.total_score < chosen_score.total_score


def test_patch_candidate_participates_in_ranking(monkeypatch):
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T910",
            "instruction": "write the repaired artifact",
            "validator": "file_contains_regex",
            "validator_params": {"path": "patched_ranked.txt", "patterns": ["done"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {
        "step_id": "S1",
        "subgoal": "write file",
        "tool": "file_write",
        "args": {"path": "wrong_ranked.txt", "content": "miss"},
    }

    def fake_decide_action(*args, **kwargs):
        executor.last_trace = {
            "llm_raw_text": "",
            "parsed_json": {},
            "validation_errors": [],
            "fallback_used": False,
            "fallback_reason": "",
            "parse_failures": 0,
            "attempts": [],
            "estimated_tokens": 18,
            "referenced_memory_keys": [],
            "injection_metadata": None,
        }
        return {"tool": "noop", "args": {}, "rationale": "wrong"}

    monkeypatch.setattr(executor, "decide_action", fake_decide_action)
    patch_candidate = {"tool": "file_write", "args": {"path": "patched_ranked.txt", "content": "done"}, "rationale": "patch"}
    traj = executor.execute_step(task, step, budget, 0, "tight", "test", action_override=patch_candidate)
    record = traj.steps[0]
    assert any(candidate.source == "patch_candidate" for candidate in record.candidate_actions)
    assert record.chosen_action.source == "patch_candidate"
    assert record.chosen_action.args["path"] == "patched_ranked.txt"


def test_ranking_metadata_is_recorded_for_replay(monkeypatch):
    generate_data(seed=0)
    task = get_task("T001")
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {"step_id": "S1", "subgoal": "query", "tool": "sql_query", "args": {"query": "SELECT 1"}}

    def fake_decide_action(*args, **kwargs):
        executor.last_trace = {
            "llm_raw_text": "",
            "parsed_json": {},
            "validation_errors": [],
            "fallback_used": False,
            "fallback_reason": "",
            "parse_failures": 0,
            "attempts": [],
            "estimated_tokens": 10,
            "referenced_memory_keys": [],
            "injection_metadata": None,
        }
        return {"tool": "sql_query", "args": {"query": "SELECT 1"}, "rationale": "direct"}

    monkeypatch.setattr(executor, "decide_action", fake_decide_action)
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    record = traj.steps[0]
    assert record.candidate_actions
    assert record.action_scores
    assert record.chosen_action is not None
    assert record.selection_reason
    assert record.metadata["candidate_actions"]
    assert record.metadata["action_scores"]
    assert record.metadata["selection_reason"] == record.selection_reason


@pytest.mark.parametrize(
    ("step", "expected_type"),
    [
        ({"step_id": "S1", "subgoal": "read", "tool": "file_read", "args": {"path": "incident_brief.txt"}}, "file_text"),
        ({"step_id": "S1", "subgoal": "logs", "tool": "log_search", "args": {"pattern": "REQ-404"}}, "log_lines"),
        ({"step_id": "S1", "subgoal": "sql", "tool": "sql_query", "args": {"query": "SELECT name FROM users ORDER BY id"}}, "sql_result"),
        ({"step_id": "S1", "subgoal": "write", "tool": "file_write", "args": {"path": "typed_write.txt", "content": "done"}}, "text"),
    ],
)
def test_tool_outputs_are_classified_into_memory_types(step, expected_type):
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T905",
            "instruction": "classify memory output",
            "validator": "file_contains_regex",
            "validator_params": {"path": "typed_write.txt", "patterns": ["done"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    traj = executor.execute_step(task, {**step, "save_as": "slot1"}, budget, 0, "tight", "test")
    assert traj.steps[0].memory_delta["slot1"]["value_type"] == expected_type


def test_reflector_maps_labels_to_strategies():
    reflector = Reflector()
    assert reflector.recommend_strategy("BAD_TOOL_ARGS", plan_step={"tool": "sql_query", "args": {"query": "SELECT 1"}}, recent_steps=[], error=None)["recommended_strategy"] == "patch_args"
    assert reflector.recommend_strategy("PLAN_MISMATCH", plan_step={"tool": "sql_query", "args": {"query": "SELECT 1"}}, recent_steps=[], error=None)["recommended_strategy"] == "replan"
    assert reflector.recommend_strategy("TOOL_NOT_FOUND", plan_step={"tool": "missing_tool", "args": {}}, recent_steps=[], error="tool_not_found")["recommended_strategy"] == "fail_fast"
    assert reflector.recommend_strategy("BUDGET_EXHAUSTED", plan_step={"tool": "sql_query", "args": {}}, recent_steps=[], error="budget_exhausted")["recommended_strategy"] == "terminate"


def test_empty_result_is_classified_consistently():
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T907",
            "instruction": "empty result",
            "validator": "file_contains_regex",
            "validator_params": {"path": "empty.txt", "patterns": ["x"]},
            "task_type": "multi_tool_chain",
        }
    )
    executor = Executor(MockLLM(seed=0, noise=0.0))
    budget = BudgetController.from_preset("tight").initial()
    step = {"step_id": "S1", "subgoal": "search unmatched logs", "tool": "log_search", "args": {"pattern": "DOES_NOT_EXIST"}}
    traj = executor.execute_step(task, step, budget, 0, "tight", "test")
    assert traj.steps[0].metadata["failure_label"] == "EMPTY_RESULT"


def test_validation_failure_is_reported_without_fake_success(monkeypatch):
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T908",
            "instruction": "write exact artifact",
            "validator": "file_contains_regex",
            "validator_params": {"path": "validation_fail.txt", "patterns": ["^done$"]},
            "task_type": "multi_tool_chain",
        }
    )
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0), disable_reflector=True, disable_replan=True)
    monkeypatch.setattr(
        agent.planner,
        "plan",
        lambda *args, **kwargs: [{"step_id": "S1", "subgoal": "write wrong artifact", "tool": "file_write", "args": {"path": "validation_fail.txt", "content": "not-done"}, "success_criteria": []}],
    )
    traj = agent.run(task, budget_mode="tight")
    assert traj.success is False
    assert traj.metadata["failure_label"] == "VALIDATION_FAILED"


def test_task_validation_result_is_explainable():
    generate_data(seed=0)
    task = get_task("T001")
    result = task.validate_result(budget_mode="tight")
    assert result.ok is True
    assert result.validator == "sql_result_equals"
    assert "matched" in result.message.lower()


def test_fail_fast_does_not_continue_execution(monkeypatch):
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T900",
            "instruction": "write expected artifact",
            "validator": "file_contains_regex",
            "validator_params": {"path": "fail_fast_target.txt", "patterns": ["done"]},
            "task_type": "multi_tool_chain",
        }
    )
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0))

    monkeypatch.setattr(
        agent.planner,
        "plan",
        lambda *args, **kwargs: [
            {"step_id": "S1", "subgoal": "bad tool", "tool": "missing_tool", "args": {}, "success_criteria": []},
            {"step_id": "S2", "subgoal": "should not run", "tool": "sql_query", "args": {"query": "SELECT 1"}, "success_criteria": []},
        ],
    )
    monkeypatch.setattr(
        agent.reflector,
        "reflect",
        lambda **kwargs: {
            "label": "TOOL_NOT_FOUND",
            "explanation": "fail fast",
            "recommended_strategy": "fail_fast",
            "fix_action": "fail_fast",
            "replan_needed": False,
            "recovery_reason": "Missing tool cannot be recovered safely.",
            "patch": None,
        },
    )

    traj = agent.run(task, budget_mode="tight")
    assert traj.success is False
    assert len(traj.steps) == 1
    assert traj.steps[0].actual_recovery_action == "fail_fast"
    assert traj.metadata["failure_label"] == "TOOL_NOT_FOUND"


def test_weak_validation_wrapper_changes_outcome_without_changing_execution():
    generate_data(seed=0)
    from autotoolbench.eval.runner import TaskValidationWrapper

    base_task = get_task("T005")
    assert base_task is not None
    weak_task = TaskValidationWrapper(base_task, mode="weak_validation")
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0), budget=BudgetController.from_preset("loose"))
    base_traj = agent.run(base_task, budget_mode="loose")
    weak_traj = AdaptiveAgent(MockLLM(seed=0, noise=0.0), budget=BudgetController.from_preset("loose")).run(weak_task, budget_mode="loose")
    assert weak_task.validate_result("loose").validator.startswith("weak::")
    assert weak_traj.metadata["validation"]["validator"].startswith("weak::")
    assert weak_traj.metadata["budget_usage"]["calls"] <= base_traj.metadata["budget_usage"]["calls"]
    assert int(weak_traj.success) >= int(base_traj.success)


def test_patch_strategy_retries_current_step(monkeypatch):
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T901",
            "instruction": "write expected artifact",
            "validator": "file_contains_regex",
            "validator_params": {"path": "patched_target.txt", "patterns": ["done"]},
            "task_type": "multi_tool_chain",
        }
    )
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0))

    monkeypatch.setattr(
        agent.planner,
        "plan",
        lambda *args, **kwargs: [
            {
                "step_id": "S1",
                "subgoal": "write wrong file first",
                "tool": "file_write",
                "args": {"path": "wrong_target.txt", "content": "miss"},
                "success_criteria": [],
            }
        ],
    )
    monkeypatch.setattr(
        agent.reflector,
        "reflect",
        lambda **kwargs: {
            "label": "BAD_TOOL_ARGS",
            "explanation": "patch args",
            "recommended_strategy": "patch_args",
            "fix_action": "patch",
            "replan_needed": False,
            "recovery_reason": "Retry with the known-good query.",
            "patch": {"tool": "file_write", "args": {"path": "patched_target.txt", "content": "done"}},
        },
    )

    traj = agent.run(task, budget_mode="tight")
    assert traj.success is True
    assert traj.metadata["patch_count"] == 1
    assert traj.steps[0].actual_recovery_action == "patch_args"
    assert len(traj.steps) >= 2


def test_replan_strategy_rebuilds_plan(monkeypatch):
    generate_data(seed=0)
    task = Task(
        {
            "task_id": "T902",
            "instruction": "write expected artifact",
            "validator": "file_contains_regex",
            "validator_params": {"path": "replanned_target.txt", "patterns": ["done"]},
            "task_type": "multi_tool_chain",
        }
    )
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0))
    plans = [
        [{"step_id": "S1", "subgoal": "wrong step", "tool": "noop", "args": {}, "success_criteria": []}],
        [{"step_id": "S1", "subgoal": "correct step", "tool": "file_write", "args": {"path": "replanned_target.txt", "content": "done"}, "success_criteria": []}],
    ]

    def fake_plan(*args, **kwargs):
        return plans[min(kwargs.get("replan_count", 0), 1)]

    monkeypatch.setattr(agent.planner, "plan", fake_plan)
    monkeypatch.setattr(
        agent.reflector,
        "reflect",
        lambda **kwargs: {
            "label": "PLAN_MISMATCH",
            "explanation": "replan",
            "recommended_strategy": "replan",
            "fix_action": "replan",
            "replan_needed": True,
            "recovery_reason": "The noop step does not advance the task.",
            "patch": None,
        },
    )

    traj = agent.run(task, budget_mode="tight")
    assert traj.success is True
    assert traj.metadata["replan_count"] == 1
    assert any(step.actual_recovery_action == "replan" for step in traj.steps)


def test_budget_exhausted_does_not_attempt_recovery():
    generate_data(seed=0)
    task = get_task("T001")
    agent = AdaptiveAgent(MockLLM(seed=0, noise=0.0), budget=BudgetController(max_calls=0, max_steps=0, max_time=0, max_tokens=0))
    traj = agent.run(task, budget_mode="tight")
    assert traj.success is False
    assert traj.metadata["failure_label"] == "BUDGET_EXHAUSTED"
    assert traj.metadata["patch_count"] == 0
    assert traj.metadata["replan_count"] == 0
    assert traj.metadata["actual_recovery_action"] == "terminate"


def test_cli_eval_outputs_report_and_config():
    runner = CliRunner()
    assert runner.invoke(app, ["make-data", "--seed", "0"]).exit_code == 0
    result = runner.invoke(app, ["eval", "--agent", "all", "--seed", "0"])
    assert result.exit_code == 0
    report_text = (reports_dir() / "latest_report.md").read_text(encoding="utf-8")
    assert "Overall Matrix" in report_text
    assert "Task Type Success Rates" in report_text
    assert "parse failure rate" in report_text.lower()
