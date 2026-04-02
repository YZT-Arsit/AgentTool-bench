from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Dict

from ..utils.paths import data_dir, resolve_data_path

WRITE_PROTECTED_NAMES = {
    ".env",
    "pyproject.toml",
    "README.md",
    "tasks.jsonl",
    "sample.db",
    "app.log",
    "latest_report.md",
    "latest_summary.json",
    "latest_config.json",
}
WRITE_PROTECTED_SUFFIXES = {".py", ".sh", ".db", ".sqlite", ".sqlite3", ".jsonl", ".toml"}
READ_ONLY_SQL_PREFIXES = ("select", "with", "explain select", "explain query plan select")
DANGEROUS_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|pragma|attach|detach|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


def _decision(action_allowed: bool, decision: str, reason: str, safety_level: str, tool_risk_level: str) -> Dict[str, Any]:
    return {
        "action_allowed": action_allowed,
        "safety_decision": decision,
        "safety_reason": reason,
        "safety_level": safety_level,
        "tool_risk_level": tool_risk_level,
    }


def _inspect_file_write(args: Dict[str, Any], tool_risk_level: str) -> Dict[str, Any]:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return _decision(False, "blocked", "file_write requires a non-empty relative path", "high", tool_risk_level)
    if raw_path.startswith(("/", "\\")) or "\x00" in raw_path or "~" in raw_path:
        return _decision(False, "blocked", "file_write path looks abnormal or absolute", "high", tool_risk_level)

    path_obj = PurePosixPath(raw_path)
    if any(part in {"..", ""} for part in path_obj.parts):
        return _decision(False, "blocked", "file_write path attempts directory traversal", "high", tool_risk_level)
    if any(part.startswith(".") for part in path_obj.parts):
        return _decision(False, "blocked", "file_write path targets a hidden or protected location", "high", tool_risk_level)
    if path_obj.name in WRITE_PROTECTED_NAMES or path_obj.suffix.lower() in WRITE_PROTECTED_SUFFIXES:
        return _decision(False, "blocked", "file_write path could overwrite a protected file type", "high", tool_risk_level)

    try:
        target = resolve_data_path(raw_path)
    except ValueError:
        return _decision(False, "blocked", "file_write path escapes the data directory", "high", tool_risk_level)

    if target.exists() and target.is_file() and target.suffix.lower() not in {".txt", ".json", ".md", ".log"}:
        return _decision(True, "warned", "file_write will overwrite an existing non-artifact file", "medium", tool_risk_level)
    if target.exists() and target.is_file() and target.resolve().parent != data_dir().resolve():
        return _decision(True, "warned", "file_write will overwrite a nested file under the data directory", "medium", tool_risk_level)
    return _decision(True, "allowed", "write path passed lightweight safety checks", tool_risk_level, tool_risk_level)


def _inspect_sql_query(args: Dict[str, Any], tool_risk_level: str) -> Dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        return _decision(False, "blocked", "sql_query requires a non-empty query", "high", tool_risk_level)
    lowered = " ".join(query.lower().split())
    statements = [part.strip() for part in query.split(";") if part.strip()]
    if len(statements) > 1:
        return _decision(False, "blocked", "multiple SQL statements are not allowed", "high", tool_risk_level)
    if DANGEROUS_SQL_PATTERN.search(lowered):
        return _decision(False, "blocked", "non-read-only SQL keyword detected", "high", tool_risk_level)
    if not any(lowered.startswith(prefix) for prefix in READ_ONLY_SQL_PREFIXES):
        return _decision(False, "blocked", "only read-only SQL queries are allowed", "high", tool_risk_level)
    return _decision(True, "allowed", "query passed read-only SQL safety checks", tool_risk_level, tool_risk_level)


def inspect_action(tool_name: str, args: Dict[str, Any], tool_risk_level: str = "low") -> Dict[str, Any]:
    if tool_name == "file_write":
        return _inspect_file_write(args, tool_risk_level)
    if tool_name == "sql_query":
        return _inspect_sql_query(args, tool_risk_level)
    if tool_risk_level == "high":
        return _decision(True, "warned", f"{tool_name} is a high-risk tool and was allowed with a warning", "medium", tool_risk_level)
    return _decision(True, "allowed", "tool passed lightweight safety checks", tool_risk_level, tool_risk_level)
