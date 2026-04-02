from __future__ import annotations

from enum import StrEnum

PACKAGE_NAME = "autotoolbench"
PACKAGE_VERSION = "0.1.0"

DATA_DIRNAME = "data"
REPORTS_DIRNAME = "reports"
RUNS_DIRNAME = "runs"
TRAJECTORIES_DIRNAME = "trajectories"

LATEST_REPORT_NAME = "latest_report.md"
LATEST_CONFIG_NAME = "latest_config.json"
LATEST_SUMMARY_NAME = "latest_summary.json"
LATEST_TASKS_SNAPSHOT_NAME = "latest_tasks.jsonl"
LATEST_TOOL_SCHEMA_NAME = "latest_tool_schema.json"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 0


class BudgetPreset(StrEnum):
    TIGHT = "tight"
    LOOSE = "loose"


class InjectionType(StrEnum):
    TOOL_CHOICE_ERROR = "TOOL_CHOICE_ERROR"
    TOOL_ARGS_ERROR = "TOOL_ARGS_ERROR"
    MISSING_STEP = "MISSING_STEP"


class FailureLabel(StrEnum):
    PLAN_ERROR = "PLAN_ERROR"
    BAD_TOOL_ARGS = "BAD_TOOL_ARGS"
    MISSING_PREREQ = "MISSING_PREREQ"
    TASK_VALIDATION_FAILED = "TASK_VALIDATION_FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
