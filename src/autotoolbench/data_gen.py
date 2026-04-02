from __future__ import annotations

import json
import random
import shutil
import sqlite3
import time
from typing import Any

from .utils.paths import app_log_path, data_dir, db_path, logs_dir, tasks_file


def _write_seed_file(seed: int) -> None:
    (data_dir() / "seed.json").write_text(json.dumps({"seed": seed}, indent=2), encoding="utf-8")


def _init_database(seed: int) -> None:
    random.seed(seed)
    for _ in range(10):
        conn = None
        try:
            db_path().unlink(missing_ok=True)
            conn = sqlite3.connect(db_path(), timeout=1)
            cur = conn.cursor()
            cur.executescript(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    team TEXT NOT NULL
                );
                CREATE TABLE orders (
                    order_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.executemany(
                "INSERT INTO users (id, name, age, team) VALUES (?, ?, ?, ?)",
                [
                    (1, "Alice", 30, "analytics"),
                    (2, "Bob", 25, "platform"),
                    (3, "Carla", 29, "infra"),
                ],
            )
            cur.executemany(
                "INSERT INTO orders (order_id, user_id, amount, status, created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (1001, 1, 120.50, "paid", "2025-02-01"),
                    (1002, 1, 80.00, "pending", "2025-02-03"),
                    (1003, 2, 200.00, "paid", "2025-02-04"),
                    (1004, 3, 50.25, "failed", "2025-02-05"),
                    (1005, 3, 75.75, "paid", "2025-02-06"),
                ],
            )
            conn.commit()
            conn.close()
            return
        except sqlite3.OperationalError:
            if conn is not None:
                conn.close()
            time.sleep(0.1)
    raise RuntimeError("failed to initialize sample.db after retries")


def _write_logs() -> None:
    log_lines = [
        "INFO request_id=REQ-100 service=payments msg=boot complete",
        "INFO request_id=REQ-101 service=payments msg=invoice fetched invoice_id=INV-9",
        (
            "ERROR request_id=REQ-404 service=payments "
            "root_cause=database_pool_exhausted "
            'detail="connection pool exhausted while loading invoice INV-9"'
        ),
        "WARN request_id=REQ-404 service=payments retry=false owner=payments-api",
        (
            "ERROR request_id=REQ-777 service=search "
            'root_cause=cache_node_timeout detail="redis shard 2 timed out"'
        ),
        "INFO request_id=REQ-777 service=search msg=recovered",
    ]
    tmp_log = app_log_path().with_name("app.log.tmp")
    tmp_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    tmp_log.replace(app_log_path())


def _write_reference_files() -> None:
    references = {
        "incident_brief.txt": "Incident: REQ-404\nOwner: payments-api\nAction: escalate to database team\n",
        "active_user.txt": "Alice\n",
        "lookup_request.txt": "REQ-404\n",
        "target_team.txt": "platform\n",
        "invoice_lookup.txt": "INV-9\n",
        "team_lookup_note.txt": "Use the platform team roster from target_team.txt\n",
        "invoice_casebook.txt": "Customer invoice INV-9 is tied to request REQ-404.\nUse incident logs for live evidence.\n",
        "owner_directory.txt": "Service owner mapping:\npayments-api -> database team\nsearch -> cache team\n",
        "audit_playbook.txt": "Audit target: platform team.\nUse the live user roster from the database, not stale notes.\n",
        "cache_lookup_note.txt": "Cache timeout reference points to REQ-777.\nConfirm with the live logs.\n",
        "customer_audit_note.txt": "Current customer audit target: Alice.\nPaid order review should use live order rows.\n",
        "stale_platform_roster.txt": "Old roster snapshot:\nplatform team = Brenda\n",
        "ops_triage_hint.txt": "If the brief mentions a request, collect live incident lines rather than copying the brief text.\n",
    }
    for name, content in references.items():
        (data_dir() / name).write_text(content, encoding="utf-8")


def _cleanup_generated_artifacts(clear_logs: bool = True) -> None:
    keep = {
        "seed.json",
        "incident_brief.txt",
        "active_user.txt",
        "lookup_request.txt",
        "target_team.txt",
        "invoice_lookup.txt",
        "team_lookup_note.txt",
        "invoice_casebook.txt",
        "owner_directory.txt",
        "audit_playbook.txt",
        "cache_lookup_note.txt",
        "customer_audit_note.txt",
        "stale_platform_roster.txt",
        "ops_triage_hint.txt",
        "tasks.jsonl",
        "sample.db",
    }
    for path in data_dir().iterdir():
        if path.name in keep or path.name == "logs":
            continue
        if path.is_file():
            path.unlink(missing_ok=True)
    if clear_logs and logs_dir().exists():
        for subdir in logs_dir().iterdir():
            if subdir.is_dir():
                shutil.rmtree(subdir, ignore_errors=True)
            else:
                subdir.unlink(missing_ok=True)


def reset_runtime_artifacts() -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    logs_dir().mkdir(parents=True, exist_ok=True)
    _cleanup_generated_artifacts(clear_logs=False)


def _task(
    task_id: str,
    instruction: str,
    validator: str,
    validator_params: dict[str, Any],
    *,
    expected_artifacts: list[str] | None = None,
    task_type: str,
    difficulty: str,
    budget_mode: str = "both",
    plan_hints: dict[str, list[dict[str, Any]]] | None = None,
    retrieval_expectations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "instruction": instruction,
        "expected_artifacts": expected_artifacts or [],
        "validator": validator,
        "validator_params": validator_params,
        "category": task_type,
        "task_type": task_type,
        "difficulty": difficulty,
        "budget_mode": budget_mode,
        "plan_hints": plan_hints or {},
        "retrieval_expectations": retrieval_expectations or {},
    }


def _multi(*validators: dict[str, Any]) -> dict[str, Any]:
    return {"validators": list(validators)}


def _validator(name: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "params": params}


def _write_tasks() -> None:
    tasks: list[dict[str, Any]] = []
    task_id = 1

    def next_id() -> str:
        nonlocal task_id
        current = f"T{task_id:03d}"
        task_id += 1
        return current

    single_queries = [
        (
            "select id, name, team from users order by id",
            "SELECT id, name, team FROM users ORDER BY id",
            [{"id": 1, "name": "Alice", "team": "analytics"}, {"id": 2, "name": "Bob", "team": "platform"}, {"id": 3, "name": "Carla", "team": "infra"}],
        ),
        (
            "select status, count(*) as total from orders group by status order by status",
            "SELECT status, count(*) AS total FROM orders GROUP BY status ORDER BY status",
            [{"status": "failed", "total": 1}, {"status": "paid", "total": 3}, {"status": "pending", "total": 1}],
        ),
        (
            "select count(*) as paid_orders from orders where status = 'paid'",
            "SELECT count(*) AS paid_orders FROM orders WHERE status = 'paid'",
            [{"paid_orders": 3}],
        ),
        (
            "select order_id, amount from orders where order_id = 1003",
            "SELECT order_id, amount FROM orders WHERE order_id = 1003",
            [{"order_id": 1003, "amount": 200.0}],
        ),
    ]
    for instruction, query, expected_rows in single_queries:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "sql_result_equals",
                {"query": query, "expected_rows": expected_rows},
                task_type="single_tool_easy",
                difficulty="easy",
                plan_hints={"default": [{"tool": "sql_query", "args": {"query": query}}]},
            )
        )

    chain_tasks = [
        (
            "select name, age from users order by age desc and write to file users_report.json",
            "users_report.json",
            "SELECT name, age FROM users ORDER BY age DESC",
            _multi(
                _validator(
                    "file_json_schema",
                    {"path": "users_report.json", "schema": {"type": "array", "min_items": 3, "items": {"type": "object", "required": ["name", "age"]}}},
                ),
                _validator("file_contains_regex", {"path": "users_report.json", "patterns": ["Alice", "Carla", "Bob"]}),
            ),
        ),
        (
            "search logs for request_id=REQ-404 and write to file req404_lines.json",
            "req404_lines.json",
            None,
            _multi(
                _validator("file_json_schema", {"path": "req404_lines.json", "schema": {"type": "array", "min_items": 2, "items": {"type": "object", "required": ["line", "text"]}}}),
                _validator("file_contains_regex", {"path": "req404_lines.json", "patterns": ["REQ-404", "database_pool_exhausted", "payments-api"]}),
            ),
        ),
        (
            "select user_id, sum(amount) as total_amount from orders where status = 'paid' group by user_id order by user_id and write to file paid_totals.json",
            "paid_totals.json",
            "SELECT user_id, sum(amount) AS total_amount FROM orders WHERE status = 'paid' GROUP BY user_id ORDER BY user_id",
            _multi(
                _validator("file_json_schema", {"path": "paid_totals.json", "schema": {"type": "array", "min_items": 3, "items": {"type": "object", "required": ["user_id", "total_amount"]}}}),
                _validator("file_contains_regex", {"path": "paid_totals.json", "patterns": ["120.5", "200.0", "75.75"]}),
            ),
        ),
        (
            "read file incident_brief.txt and write to file incident_brief_copy.txt",
            "incident_brief_copy.txt",
            None,
            _multi(_validator("file_contains_regex", {"path": "incident_brief_copy.txt", "patterns": ["REQ-404", "payments-api", "database team"]})),
        ),
    ]
    for instruction, output_path, query, validator_params in chain_tasks:
        default_plan = []
        if instruction.startswith("select"):
            default_plan = [
                {"tool": "sql_query", "args": {"query": query}},
                {"tool": "file_write", "args": {"path": output_path, "content": ""}},
            ]
        elif instruction.startswith("search logs"):
            default_plan = [
                {"tool": "log_search", "args": {"pattern": "request_id=REQ-404"}},
                {"tool": "file_write", "args": {"path": output_path, "content": ""}},
            ]
        else:
            default_plan = [
                {"tool": "file_read", "args": {"path": "incident_brief.txt"}},
                {"tool": "file_write", "args": {"path": output_path, "content": ""}},
            ]
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="multi_tool_chain",
                difficulty="medium",
                plan_hints={"default": default_plan},
            )
        )

    prereq_tasks = [
        (
            "read file active_user.txt and write to file active_user_orders.json",
            "active_user_orders.json",
            [
                {"tool": "file_read", "args": {"path": "active_user.txt"}},
                {"tool": "sql_query", "args": {"query": "SELECT order_id, status, amount FROM orders WHERE user_id = 1 ORDER BY order_id"}},
                {"tool": "file_write", "args": {"path": "active_user_orders.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "active_user_orders.json", "schema": {"type": "array", "min_items": 2, "items": {"type": "object", "required": ["order_id", "status", "amount"]}}}),
                _validator("file_contains_regex", {"path": "active_user_orders.json", "patterns": ["1001", "1002", "pending"]}),
            ),
        ),
        (
            "read file lookup_request.txt and write to file lookup_request_incident.json",
            "lookup_request_incident.json",
            [
                {"tool": "file_read", "args": {"path": "lookup_request.txt"}},
                {"tool": "log_search", "args": {"pattern": "request_id=REQ-404"}},
                {"tool": "file_write", "args": {"path": "lookup_request_incident.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "lookup_request_incident.json", "schema": {"type": "array", "min_items": 2, "items": {"type": "object", "required": ["line", "text"]}}}),
                _validator("file_contains_regex", {"path": "lookup_request_incident.json", "patterns": ["REQ-404", "payments-api"]}),
            ),
        ),
        (
            "read file target_team.txt and write to file target_team_members.json",
            "target_team_members.json",
            [
                {"tool": "file_read", "args": {"path": "target_team.txt"}},
                {"tool": "sql_query", "args": {"query": "SELECT name, age FROM users WHERE team = 'platform' ORDER BY id"}},
                {"tool": "file_write", "args": {"path": "target_team_members.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "target_team_members.json", "schema": {"type": "array", "min_items": 1, "items": {"type": "object", "required": ["name", "age"]}}}),
                _validator("file_contains_regex", {"path": "target_team_members.json", "patterns": ["Bob", "25"]}),
            ),
        ),
        (
            "read file invoice_lookup.txt and write to file invoice_lookup_incident.json",
            "invoice_lookup_incident.json",
            [
                {"tool": "file_read", "args": {"path": "invoice_lookup.txt"}},
                {"tool": "log_search", "args": {"pattern": "INV-9|REQ-404"}},
                {"tool": "file_write", "args": {"path": "invoice_lookup_incident.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "invoice_lookup_incident.json", "schema": {"type": "array", "min_items": 3, "items": {"type": "object", "required": ["line", "text"]}}}),
                _validator("file_contains_regex", {"path": "invoice_lookup_incident.json", "patterns": ["INV-9", "REQ-404", "database_pool_exhausted"]}),
            ),
        ),
    ]
    for instruction, output_path, plan, validator_params in prereq_tasks:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="prerequisite_dependency",
                difficulty="hard",
                plan_hints={"default": plan},
            )
        )

    confusion_tasks = [
        (
            "from the incident brief text, write the owner line to incident_owner.txt",
            "incident_owner.txt",
            [{"tool": "file_read", "args": {"path": "incident_brief.txt"}}, {"tool": "file_write", "args": {"path": "incident_owner.txt", "content": ""}}],
            _multi(_validator("file_contains_regex", {"path": "incident_owner.txt", "patterns": ["Owner", "payments-api"]})),
        ),
        (
            "from the payments logs, write the REQ-404 evidence lines to payments_incident.json",
            "payments_incident.json",
            [{"tool": "log_search", "args": {"pattern": "request_id=REQ-404"}}, {"tool": "file_write", "args": {"path": "payments_incident.json", "content": ""}}],
            _multi(_validator("file_contains_regex", {"path": "payments_incident.json", "patterns": ["REQ-404", "database_pool_exhausted", "payments-api"]})),
        ),
        (
            "from the orders table, write the paid totals grouped by user to paid_totals_copy.json",
            "paid_totals_copy.json",
            [
                {"tool": "sql_query", "args": {"query": "SELECT user_id, sum(amount) AS total_amount FROM orders WHERE status = 'paid' GROUP BY user_id ORDER BY user_id"}},
                {"tool": "file_write", "args": {"path": "paid_totals_copy.json", "content": ""}},
            ],
            _multi(_validator("file_contains_regex", {"path": "paid_totals_copy.json", "patterns": ["120.5", "200.0", "75.75"]})),
        ),
        (
            "from the team lookup note, produce a platform roster in platform_roster.json",
            "platform_roster.json",
            [
                {"tool": "file_read", "args": {"path": "team_lookup_note.txt"}},
                {"tool": "sql_query", "args": {"query": "SELECT name, team FROM users WHERE team = 'platform' ORDER BY id"}},
                {"tool": "file_write", "args": {"path": "platform_roster.json", "content": ""}},
            ],
            _multi(_validator("file_contains_regex", {"path": "platform_roster.json", "patterns": ["Bob", "platform"]})),
        ),
    ]
    for instruction, output_path, plan, validator_params in confusion_tasks:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="tool_confusion",
                difficulty="medium",
                plan_hints={"default": plan},
            )
        )

    brittle_tasks = [
        (
            "write the top two paid orders by amount to top_paid_orders.json",
            "top_paid_orders.json",
            [{"tool": "sql_query", "args": {"query": "SELECT order_id, amount FROM orders WHERE status = 'paid' ORDER BY amount DESC LIMIT 2"}}, {"tool": "file_write", "args": {"path": "top_paid_orders.json", "content": ""}}],
            _multi(
                _validator("file_contains_regex", {"path": "top_paid_orders.json", "patterns": ["1003", "1001"]}),
                _validator("file_not_contains_regex", {"path": "top_paid_orders.json", "patterns": ["1005"]}),
            ),
        ),
        (
            "write only the REQ-404 log lines to req404_only.json",
            "req404_only.json",
            [{"tool": "log_search", "args": {"pattern": "request_id=REQ-404"}}, {"tool": "file_write", "args": {"path": "req404_only.json", "content": ""}}],
            _multi(
                _validator("file_contains_regex", {"path": "req404_only.json", "patterns": ["REQ-404"]}),
                _validator("file_not_contains_regex", {"path": "req404_only.json", "patterns": ["REQ-777"]}),
            ),
        ),
        (
            "write users younger than 30 sorted ascending to under_30.json",
            "under_30.json",
            [{"tool": "sql_query", "args": {"query": "SELECT name, age FROM users WHERE age < 30 ORDER BY age ASC"}}, {"tool": "file_write", "args": {"path": "under_30.json", "content": ""}}],
            _multi(
                _validator("file_contains_regex", {"path": "under_30.json", "patterns": ["Bob", "Carla"]}),
                _validator("file_not_contains_regex", {"path": "under_30.json", "patterns": ["Alice"]}),
            ),
        ),
        (
            "write only cache timeout log lines to cache_timeout_only.json",
            "cache_timeout_only.json",
            [{"tool": "log_search", "args": {"pattern": "root_cause=cache_node_timeout"}}, {"tool": "file_write", "args": {"path": "cache_timeout_only.json", "content": ""}}],
            _multi(
                _validator("file_contains_regex", {"path": "cache_timeout_only.json", "patterns": ["REQ-777", "cache_node_timeout"]}),
                _validator("file_not_contains_regex", {"path": "cache_timeout_only.json", "patterns": ["REQ-404"]}),
            ),
        ),
    ]
    for instruction, output_path, plan, validator_params in brittle_tasks:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="args_brittle",
                difficulty="hard",
                plan_hints={"default": plan},
            )
        )

    budget_tasks = [
        (
            "analyze REQ-404 incident and write to file req404_budget_report.json",
            "req404_budget_report.json",
            {
                "tight": [
                    {"tool": "log_search", "args": {"pattern": "request_id=REQ-404"}},
                    {"tool": "file_write", "args": {"path": "req404_budget_report.json", "content": ""}},
                ],
                "loose": [
                    {"tool": "log_search", "args": {"pattern": "invoice fetched|REQ-404|payments-api"}},
                    {"tool": "file_write", "args": {"path": "req404_budget_report.json", "content": ""}},
                ],
            },
            {
                "tight": _multi(_validator("file_contains_regex", {"path": "req404_budget_report.json", "patterns": ["REQ-404", "database_pool_exhausted"]})),
                "loose": _multi(_validator("file_contains_regex", {"path": "req404_budget_report.json", "patterns": ['"line": 2', '"line": 3', '"line": 4', "invoice fetched", "payments-api"]})),
            },
        ),
        (
            "analyze Alice revenue and write to file alice_revenue_audit.json",
            "alice_revenue_audit.json",
            {
                "tight": [
                    {"tool": "sql_query", "args": {"query": "SELECT user_id, sum(amount) AS total_amount FROM orders WHERE user_id = 1 AND status = 'paid' GROUP BY user_id"}},
                    {"tool": "file_write", "args": {"path": "alice_revenue_audit.json", "content": ""}},
                ],
                "loose": [
                    {"tool": "sql_query", "args": {"query": "SELECT order_id, status, amount FROM orders WHERE user_id = 1 ORDER BY order_id"}},
                    {"tool": "file_write", "args": {"path": "alice_revenue_audit.json", "content": ""}},
                ],
            },
            {
                "tight": _multi(_validator("file_contains_regex", {"path": "alice_revenue_audit.json", "patterns": ["120.5"]})),
                "loose": _multi(_validator("file_contains_regex", {"path": "alice_revenue_audit.json", "patterns": ["1001", "1002", "pending", "paid"]})),
            },
        ),
        (
            "analyze platform team roster and write to file platform_roster_audit.json",
            "platform_roster_audit.json",
            {
                "tight": [
                    {"tool": "sql_query", "args": {"query": "SELECT name FROM users WHERE team = 'platform' ORDER BY id"}},
                    {"tool": "file_write", "args": {"path": "platform_roster_audit.json", "content": ""}},
                ],
                "loose": [
                    {"tool": "sql_query", "args": {"query": "SELECT name, age, team FROM users WHERE team = 'platform' ORDER BY id"}},
                    {"tool": "file_write", "args": {"path": "platform_roster_audit.json", "content": ""}},
                ],
            },
            {
                "tight": _multi(_validator("file_contains_regex", {"path": "platform_roster_audit.json", "patterns": ["Bob"]})),
                "loose": _multi(_validator("file_contains_regex", {"path": "platform_roster_audit.json", "patterns": ["Bob", "25", "platform"]})),
            },
        ),
        (
            "analyze cache timeout incident and write to file cache_budget_report.json",
            "cache_budget_report.json",
            {
                "tight": [
                    {"tool": "log_search", "args": {"pattern": "root_cause=cache_node_timeout"}},
                    {"tool": "file_write", "args": {"path": "cache_budget_report.json", "content": ""}},
                ],
                "loose": [
                    {"tool": "log_search", "args": {"pattern": "cache_node_timeout|recovered"}},
                    {"tool": "file_write", "args": {"path": "cache_budget_report.json", "content": ""}},
                ],
            },
            {
                "tight": _multi(_validator("file_contains_regex", {"path": "cache_budget_report.json", "patterns": ["REQ-777", "cache_node_timeout"]})),
                "loose": _multi(_validator("file_contains_regex", {"path": "cache_budget_report.json", "patterns": ['"line": 5', '"line": 6', "cache_node_timeout", "recovered"]})),
            },
        ),
    ]
    for instruction, output_path, plan_hints, validator_params in budget_tasks:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="budget_tradeoff",
                difficulty="hard",
                budget_mode="both",
                plan_hints=plan_hints,
            )
        )

    retrieval_tasks = [
        (
            "search the local reference notes for invoice INV-9 ownership evidence and write to file inv9_retrieval.json",
            "inv9_retrieval.json",
            [
                {
                    "tool": "doc_search",
                    "args": {"query": "INV-9 ownership evidence REQ-404 payments-api database team", "files": ["invoice_casebook.txt", "incident_brief.txt", "owner_directory.txt"], "top_k": 4},
                    "save_as": "inv9_evidence",
                },
                {"tool": "file_write", "args": {"path": "inv9_retrieval.json", "content": "$memory:inv9_evidence"}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "inv9_retrieval.json", "schema": {"type": "array", "min_items": 3, "items": {"type": "object", "required": ["source", "chunk", "score", "rank"]}}}),
                _validator(
                    "retrieval_results_quality",
                    {
                        "path": "inv9_retrieval.json",
                        "required_sources": ["invoice_casebook.txt", "incident_brief.txt", "owner_directory.txt"],
                        "required_terms": ["INV-9", "REQ-404", "payments-api", "database team"],
                        "max_noise": 1,
                        "max_rank": 4,
                    },
                ),
            ),
            {"slot": "inv9_evidence", "required_sources": ["invoice_casebook.txt", "incident_brief.txt", "owner_directory.txt"], "required_terms": ["INV-9", "REQ-404", "payments-api", "database team"], "max_noise": 1},
        ),
        (
            "use the local audit notes to find which team should be audited, then write the live roster to retrieved_team_roster.json",
            "retrieved_team_roster.json",
            [
                {
                    "tool": "doc_search",
                    "args": {"query": "audit target platform live user roster", "files": ["audit_playbook.txt", "stale_platform_roster.txt"], "top_k": 3},
                    "save_as": "audit_evidence",
                },
                {"tool": "sql_query", "args": {"query": "SELECT name, age, team FROM users WHERE team = 'platform' ORDER BY id"}},
                {"tool": "file_write", "args": {"path": "retrieved_team_roster.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "retrieved_team_roster.json", "schema": {"type": "array", "min_items": 1, "items": {"type": "object", "required": ["name", "age", "team"]}}}),
                _validator("file_contains_regex", {"path": "retrieved_team_roster.json", "patterns": ["Bob", "25", "platform"]}),
                _validator("file_not_contains_regex", {"path": "retrieved_team_roster.json", "patterns": ["Brenda"]}),
            ),
            {"slot": "audit_evidence", "required_sources": ["audit_playbook.txt"], "required_terms": ["Audit target", "platform", "live user roster"], "max_noise": 1},
        ),
        (
            "use the local cache lookup notes to find which request is linked to the cache timeout and write matching incident lines to retrieved_cache_incident.json",
            "retrieved_cache_incident.json",
            [
                {
                    "tool": "doc_search",
                    "args": {"query": "cache timeout REQ-777", "files": ["cache_lookup_note.txt", "owner_directory.txt"], "top_k": 3},
                    "save_as": "cache_lookup",
                },
                {"tool": "log_search", "args": {"pattern": "REQ-777|cache_node_timeout|recovered"}},
                {"tool": "file_write", "args": {"path": "retrieved_cache_incident.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "retrieved_cache_incident.json", "schema": {"type": "array", "min_items": 2, "items": {"type": "object", "required": ["line", "text"]}}}),
                _validator("file_contains_regex", {"path": "retrieved_cache_incident.json", "patterns": ["REQ-777", "cache_node_timeout", "recovered"]}),
            ),
            {"slot": "cache_lookup", "required_sources": ["cache_lookup_note.txt"], "required_terms": ["Cache timeout", "REQ-777"], "max_noise": 1},
        ),
        (
            "use the local customer audit notes to find the current customer to review, then write their paid order totals to retrieved_customer_paid.json",
            "retrieved_customer_paid.json",
            [
                {
                    "tool": "doc_search",
                    "args": {"query": "Alice customer audit target paid order review", "files": ["customer_audit_note.txt", "active_user.txt"], "top_k": 3},
                    "save_as": "customer_evidence",
                },
                {"tool": "sql_query", "args": {"query": "SELECT order_id, amount, status FROM orders WHERE user_id = 1 AND status = 'paid' ORDER BY order_id"}},
                {"tool": "file_write", "args": {"path": "retrieved_customer_paid.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "retrieved_customer_paid.json", "schema": {"type": "array", "min_items": 1, "items": {"type": "object", "required": ["order_id", "amount", "status"]}}}),
                _validator("file_contains_regex", {"path": "retrieved_customer_paid.json", "patterns": ["1001", "120.5", "paid"]}),
                _validator("file_not_contains_regex", {"path": "retrieved_customer_paid.json", "patterns": ["1002", "pending"]}),
            ),
            {"slot": "customer_evidence", "required_sources": ["customer_audit_note.txt"], "required_terms": ["Alice", "customer audit target", "Paid order review"], "max_noise": 1},
        ),
    ]
    for instruction, output_path, plan, validator_params, retrieval_expectations in retrieval_tasks:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="retrieval_heavy",
                difficulty="hard",
                plan_hints={"default": plan},
                retrieval_expectations=retrieval_expectations,
            )
        )

    ambiguity_tasks = [
        (
            "using the platform hint, produce the current platform roster in ambiguous_platform_roster.json",
            "ambiguous_platform_roster.json",
            [
                {"tool": "file_read", "args": {"path": "team_lookup_note.txt"}},
                {"tool": "sql_query", "args": {"query": "SELECT name, age, team FROM users WHERE team = 'platform' ORDER BY id"}},
                {"tool": "file_write", "args": {"path": "ambiguous_platform_roster.json", "content": ""}},
            ],
            _multi(
                _validator("file_contains_regex", {"path": "ambiguous_platform_roster.json", "patterns": ["Bob", "25", "platform"]}),
                _validator("file_not_contains_regex", {"path": "ambiguous_platform_roster.json", "patterns": ["Brenda"]}),
            ),
        ),
        (
            "investigate the INV-9 issue and write live incident evidence to ambiguous_inv9_incident.json",
            "ambiguous_inv9_incident.json",
            [
                {"tool": "file_read", "args": {"path": "invoice_lookup.txt"}},
                {"tool": "log_search", "args": {"pattern": "INV-9|REQ-404|payments-api"}},
                {"tool": "file_write", "args": {"path": "ambiguous_inv9_incident.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "ambiguous_inv9_incident.json", "schema": {"type": "array", "min_items": 3, "items": {"type": "object", "required": ["line", "text"]}}}),
                _validator("file_contains_regex", {"path": "ambiguous_inv9_incident.json", "patterns": ['"line": 2', '"line": 3', '"line": 4', "INV-9", "REQ-404", "payments-api"]}),
            ),
        ),
        (
            "find the active customer's order history and write it to ambiguous_active_orders.json",
            "ambiguous_active_orders.json",
            [
                {"tool": "file_read", "args": {"path": "active_user.txt"}},
                {"tool": "sql_query", "args": {"query": "SELECT order_id, status, amount FROM orders WHERE user_id = 1 ORDER BY order_id"}},
                {"tool": "file_write", "args": {"path": "ambiguous_active_orders.json", "content": ""}},
            ],
            _multi(
                _validator("file_contains_regex", {"path": "ambiguous_active_orders.json", "patterns": ["1001", "1002", "pending", "paid"]}),
                _validator("file_not_contains_regex", {"path": "ambiguous_active_orders.json", "patterns": ["REQ-404"]}),
            ),
        ),
        (
            "for the request mentioned in the brief, produce live owner evidence lines in ambiguous_owner_lines.json",
            "ambiguous_owner_lines.json",
            [
                {"tool": "file_read", "args": {"path": "incident_brief.txt"}},
                {"tool": "log_search", "args": {"pattern": "REQ-404|payments-api"}},
                {"tool": "file_write", "args": {"path": "ambiguous_owner_lines.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "ambiguous_owner_lines.json", "schema": {"type": "array", "min_items": 2, "items": {"type": "object", "required": ["line", "text"]}}}),
                _validator("file_contains_regex", {"path": "ambiguous_owner_lines.json", "patterns": ['"line": 3', '"line": 4', "payments-api", "REQ-404"]}),
            ),
        ),
    ]
    for instruction, output_path, plan, validator_params in ambiguity_tasks:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="ambiguity_heavy",
                difficulty="hard",
                plan_hints={"default": plan},
            )
        )

    long_horizon_tasks = [
        (
            "trace the INV-9 issue through notes and live logs, then write the live incident evidence to long_horizon_inv9_incident.json",
            "long_horizon_inv9_incident.json",
            [
                {"tool": "file_read", "args": {"path": "invoice_lookup.txt"}, "save_as": "invoice_lookup"},
                {"tool": "doc_search", "args": {"pattern": "INV-9|REQ-404", "files": ["invoice_casebook.txt", "incident_brief.txt"]}, "save_as": "invoice_refs"},
                {"tool": "file_read", "args": {"path": "incident_brief.txt"}, "save_as": "incident_brief"},
                {"tool": "log_search", "args": {"pattern": "INV-9|REQ-404|payments-api"}, "save_as": "incident_lines"},
                {"tool": "file_write", "args": {"path": "long_horizon_inv9_incident.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "long_horizon_inv9_incident.json", "schema": {"type": "array", "min_items": 3, "items": {"type": "object", "required": ["line", "text"]}}}),
                _validator("file_contains_regex", {"path": "long_horizon_inv9_incident.json", "patterns": ['"line": 2', '"line": 3', '"line": 4', "INV-9", "REQ-404", "payments-api"]}),
            ),
        ),
        (
            "follow the platform audit trail and write the live platform roster to long_horizon_platform_roster.json",
            "long_horizon_platform_roster.json",
            [
                {"tool": "file_read", "args": {"path": "team_lookup_note.txt"}, "save_as": "team_hint"},
                {"tool": "doc_search", "args": {"pattern": "platform|live user roster", "files": ["audit_playbook.txt", "stale_platform_roster.txt"]}, "save_as": "audit_evidence"},
                {"tool": "file_read", "args": {"path": "target_team.txt"}, "save_as": "target_team"},
                {"tool": "sql_query", "args": {"query": "SELECT name, age, team FROM users WHERE team = 'platform' ORDER BY id"}, "save_as": "platform_roster"},
                {"tool": "file_write", "args": {"path": "long_horizon_platform_roster.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "long_horizon_platform_roster.json", "schema": {"type": "array", "min_items": 1, "items": {"type": "object", "required": ["name", "age", "team"]}}}),
                _validator("file_contains_regex", {"path": "long_horizon_platform_roster.json", "patterns": ["Bob", "25", "platform"]}),
                _validator("file_not_contains_regex", {"path": "long_horizon_platform_roster.json", "patterns": ["Brenda"]}),
            ),
        ),
        (
            "use the customer audit trail to write the paid order review for the active customer to long_horizon_paid_review.json",
            "long_horizon_paid_review.json",
            [
                {"tool": "file_read", "args": {"path": "active_user.txt"}, "save_as": "active_customer"},
                {"tool": "doc_search", "args": {"pattern": "Alice|customer audit target|Paid order review", "files": ["customer_audit_note.txt", "active_user.txt"]}, "save_as": "customer_evidence"},
                {"tool": "sql_query", "args": {"query": "SELECT order_id, amount, status FROM orders WHERE user_id = 1 ORDER BY order_id"}, "save_as": "all_customer_orders"},
                {"tool": "sql_query", "args": {"query": "SELECT order_id, amount, status FROM orders WHERE user_id = 1 AND status = 'paid' ORDER BY order_id"}, "save_as": "paid_customer_orders"},
                {"tool": "file_write", "args": {"path": "long_horizon_paid_review.json", "content": ""}},
            ],
            _multi(
                _validator("file_json_schema", {"path": "long_horizon_paid_review.json", "schema": {"type": "array", "min_items": 1, "items": {"type": "object", "required": ["order_id", "amount", "status"]}}}),
                _validator("file_contains_regex", {"path": "long_horizon_paid_review.json", "patterns": ["1001", "120.5", "paid"]}),
                _validator("file_not_contains_regex", {"path": "long_horizon_paid_review.json", "patterns": ["1002", "pending"]}),
            ),
        ),
    ]
    for instruction, output_path, plan, validator_params in long_horizon_tasks:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="long_horizon",
                difficulty="hard",
                plan_hints={"default": plan},
            )
        )

    partial_success_tasks = [
        (
            "write the current platform roster to partial_platform_roster.json with all required fields and no stale names",
            "partial_platform_roster.json",
            [
                {"tool": "sql_query", "args": {"query": "SELECT name, age, team FROM users WHERE team = 'platform' ORDER BY id"}},
                {"tool": "file_write", "args": {"path": "partial_platform_roster.json", "content": ""}},
            ],
            _multi(
                _validator(
                    "file_json_quality",
                    {"path": "partial_platform_roster.json", "required_keys": ["name", "age", "team"], "min_items": 1, "forbidden_patterns": ["Brenda"]},
                ),
            ),
        ),
        (
            "write only the clean REQ-404 incident evidence to partial_req404_clean.json",
            "partial_req404_clean.json",
            [
                {"tool": "log_search", "args": {"pattern": "request_id=REQ-404"}},
                {"tool": "file_write", "args": {"path": "partial_req404_clean.json", "content": ""}},
            ],
            _multi(
                _validator(
                    "file_contains_quality",
                    {
                        "path": "partial_req404_clean.json",
                        "required_patterns": ["REQ-404", "database_pool_exhausted", "payments-api"],
                        "forbidden_patterns": ["REQ-777"],
                    },
                ),
            ),
        ),
        (
            "write only paid orders with complete fields to partial_paid_orders.json",
            "partial_paid_orders.json",
            [
                {"tool": "sql_query", "args": {"query": "SELECT order_id, amount, status FROM orders WHERE status = 'paid' ORDER BY order_id"}},
                {"tool": "file_write", "args": {"path": "partial_paid_orders.json", "content": ""}},
            ],
            _multi(
                _validator(
                    "file_json_quality",
                    {"path": "partial_paid_orders.json", "required_keys": ["order_id", "amount", "status"], "min_items": 3, "forbidden_patterns": ["pending", "failed"]},
                ),
            ),
        ),
    ]
    for instruction, output_path, plan, validator_params in partial_success_tasks:
        tasks.append(
            _task(
                next_id(),
                instruction,
                "multi_artifact",
                validator_params,
                expected_artifacts=[output_path],
                task_type="partial_success",
                difficulty="hard",
                plan_hints={"default": plan},
            )
        )

    with tasks_file().open("w", encoding="utf-8") as handle:
        for item in tasks:
            handle.write(json.dumps(item) + "\n")


def main(seed: int = 0) -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    logs_dir().mkdir(parents=True, exist_ok=True)
    _cleanup_generated_artifacts(clear_logs=True)
    _write_seed_file(seed)
    _init_database(seed)
    _write_logs()
    _write_reference_files()
    _write_tasks()
