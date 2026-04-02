import json

from autotoolbench.data_gen import main as generate_data
from autotoolbench.env import validators
from autotoolbench.tools.doc_search_tool import DocSearchTool
from autotoolbench.tools.file_tool import FileTool, FileWriteTool
from autotoolbench.tools.log_tool import LogSearchTool
from autotoolbench.tools.python_tool import PythonExecTool
from autotoolbench.tools.sqlite_tool import SQLiteTool
from autotoolbench.utils.paths import data_dir


def test_file_read_write():
    generate_data(seed=0)
    writer = FileWriteTool()
    res = writer.run({"path": "test.txt", "content": "hello"})
    assert res.ok
    assert writer.output_type == "text"
    assert writer.mutating is True
    assert "content" in writer.describe()["argument_constraints"]

    reader = FileTool()
    res2 = reader.run({"path": "incident_brief.txt"})
    assert res2.ok
    assert "REQ-404" in res2.output
    assert not reader.run({"path": "../etc/passwd"}).ok
    assert reader.risk_level == "low"
    assert writer.risk_level == "medium"


def test_sqlite_read():
    generate_data(seed=0)
    tool = SQLiteTool()
    assert tool.risk_level == "low"
    assert tool.read_only is True
    assert tool.output_type == "sql_result"
    res = tool.run({"query": "SELECT name FROM users ORDER BY id"})
    assert res.ok
    assert res.output == [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carla"}]


def test_log_tool_returns_line_metadata():
    generate_data(seed=0)
    tool = LogSearchTool()
    res = tool.run({"pattern": "request_id=REQ-404"})
    assert res.ok
    assert any(item["line"] == 3 for item in res.output)


def test_python_exec():
    tool = PythonExecTool()
    assert tool.risk_level == "high"
    res = tool.run({"code": "a=1+2"})
    assert res.ok
    assert res.output.get("a") == 3
    assert not tool.run({"code": "import os"}).ok


def test_doc_search_tool_and_json_array_validator():
    generate_data(seed=0)
    tool = DocSearchTool()
    res = tool.run({"query": "INV-9 ownership evidence", "files": ["invoice_casebook.txt", "incident_brief.txt"], "top_k": 3})
    assert res.ok
    assert res.output[0]["rank"] == 1
    assert "score" in res.output[0]
    assert "source" in res.output[0]
    assert any(item["source"] == "invoice_casebook.txt" for item in res.output)
    (data_dir() / "inv9_probe.json").write_text(json.dumps(res.output, ensure_ascii=False), encoding="utf-8")
    quality = validators.run_validator(
        "retrieval_results_quality",
        {
            "path": "inv9_probe.json",
            "required_sources": ["invoice_casebook.txt"],
            "required_terms": ["INV-9"],
            "max_noise": 1,
            "max_rank": 3,
        },
    )
    assert quality.ok is True
    assert quality.status == "full_success"


def test_validators():
    generate_data(seed=0)
    assert validators.file_contains_regex({"path": "incident_brief.txt", "patterns": ["REQ-404", "payments-api"]})
    assert validators.file_not_contains_regex({"path": "incident_brief.txt", "patterns": ["REQ-777"]})
    detailed = validators.run_validator("file_contains_regex", {"path": "incident_brief.txt", "patterns": ["REQ-404"]})
    assert detailed.ok is True
    assert detailed.validator == "file_contains_regex"
