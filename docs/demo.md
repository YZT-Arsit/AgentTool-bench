# Minimal Demo: Single Task To Report

This demo is the recommended path for quickly verifying the project and for showing it in an interview.

It does three things:
- generates local benchmark assets
- runs one adaptive-agent task
- generates one markdown report

## Fastest path

```bash
python scripts/run_minimal_demo.py
```

## Equivalent manual commands

### 1. Generate data

```bash
python -m autotoolbench make-data --seed 0
```

### 2. Run a single task

```bash
python -m autotoolbench run \
  --task-id T001 \
  --agent adaptive \
  --seed 0 \
  --noise 0.0 \
  --budget-preset loose \
  --llm mock
```

### 3. Generate a small report

```bash
python -m autotoolbench eval \
  --agent adaptive \
  --seed 0 \
  --noise 0.0
```

## What to show afterwards

### Single task trajectory

- `runs/..._run_t001/trajectory.json`

Look for:
- `steps`
- `memory_delta`
- `failure_label`
- `validation`

### Replay summary

- `runs/..._eval/.../trajectories/adaptive/<task>.summary.md`

Look for:
- first failure stage
- recovery actions
- budget usage
- validator summary

### Report

- [reports/latest_report.md](/Users/Hoshino/Downloads/AutoTool-Agent/reports/latest_report.md)

Good sections to show live:
- `Overall Matrix`
- `Failure Breakdown`
- `Recovery Action Breakdown`
- `Budget Usage Summary`
- `Retrieval Quality`
- `Single-Task Replay Samples`

## Why this demo works well

It is small enough to reproduce locally, but rich enough to demonstrate:
- tool execution
- validation
- trajectory persistence
- replay
- reporting

That makes it a much better interview artifact than a screenshot-only demo.
