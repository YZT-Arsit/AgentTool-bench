from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    _run([sys.executable, "-m", "autotoolbench", "make-data", "--seed", "0"])
    _run(
        [
            sys.executable,
            "-m",
            "autotoolbench",
            "run",
            "--task-id",
            "T001",
            "--agent",
            "adaptive",
            "--seed",
            "0",
            "--noise",
            "0.0",
            "--budget-preset",
            "loose",
            "--llm",
            "mock",
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "autotoolbench",
            "eval",
            "--agent",
            "adaptive",
            "--seed",
            "0",
            "--noise",
            "0.0",
        ]
    )

    latest_report = REPO_ROOT / "reports" / "latest_report.md"
    latest_summary = REPO_ROOT / "reports" / "latest_summary.json"
    print("\nDemo artifacts")
    print(f"- report: {latest_report}")
    print(f"- summary: {latest_summary}")
    if latest_summary.exists():
        data = json.loads(latest_summary.read_text(encoding="utf-8"))
        first_scenario = next(iter(data.values()), {})
        adaptive = first_scenario.get("adaptive", {})
        print(
            "- adaptive summary: "
            f"success_rate={adaptive.get('success_rate', 0):.2f}, "
            f"avg_calls={adaptive.get('avg_calls', 0):.2f}, "
            f"recovery_success_rate={adaptive.get('recovery_success_rate', 0):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
