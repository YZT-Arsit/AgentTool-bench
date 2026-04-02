#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$ROOT_DIR/docs"
mkdir -p "$DOCS_DIR"

MAKE_DATA_LOG="$DOCS_DIR/demo_make_data.txt"
RUN_LOG="$DOCS_DIR/demo_run.txt"
EVAL_LOG="$DOCS_DIR/demo_eval.txt"
OUTPUT_LOG="$DOCS_DIR/demo_output.txt"
REPORT_EXCERPT="$DOCS_DIR/demo_report_excerpt.md"
TRAJECTORY_EXCERPT="$DOCS_DIR/demo_trajectory_excerpt.json"

python -m autotoolbench make-data --seed 0 > "$MAKE_DATA_LOG"
python -m autotoolbench run --task-id T001 --agent adaptive --seed 0 --noise 0.2 --llm mock > "$RUN_LOG"
LATEST_RUN_DIR="$(find "$ROOT_DIR/runs" -maxdepth 1 -type d -name '*_run_t001' | sort | tail -n 1)"
python -m autotoolbench eval --agent all --seed 0 --noise 0.0 > "$EVAL_LOG"

{
  echo "## make-data"
  cat "$MAKE_DATA_LOG"
  echo
  echo "## run"
  cat "$RUN_LOG"
  echo
  echo "## eval"
  cat "$EVAL_LOG"
} > "$OUTPUT_LOG"

sed -n '1,24p' "$ROOT_DIR/reports/latest_report.md" > "$REPORT_EXCERPT"
python - <<PY
from pathlib import Path
root = Path("$ROOT_DIR")
source = Path("$LATEST_RUN_DIR") / "trajectory.json"
target = root / "docs" / "demo_trajectory_excerpt.json"
lines = source.read_text(encoding="utf-8").splitlines()
target.write_text("\n".join(lines[:80]) + "\n", encoding="utf-8")
PY

echo "demo artifacts written to $DOCS_DIR"
