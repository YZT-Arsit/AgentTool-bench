# AutoToolBench One-Pager

<p align="center">
  <strong>A local benchmark and runtime for enterprise-style tool-use agents, focused on robustness under structured failure, budget limits, retrieval noise, and explicit validation.</strong>
</p>

<p align="center">
  <a href="../README.md">README</a> ·
  <a href="demo.md">Demo</a> ·
  <a href="benchmark_methodology.md">Methodology</a> ·
  <a href="failure_taxonomy.md">Failure Taxonomy</a>
</p>

---

## One-line Pitch

AutoToolBench is a benchmark-first agent runtime for local enterprise toolchains. It evaluates whether an agent can plan, execute, recover, validate, and explain its behavior under realistic structured failure.

## Problem

Many agent demos answer:
- can the model call a tool once?

This project instead asks:
- can the system finish a workflow when planning, arguments, retrieval, or execution go wrong?

That is much closer to what makes or breaks real internal agents.

## Core Loop

```text
Task -> Planner -> Executor -> Tools -> Validator -> Reflector -> Patch / Replan / Terminate
```

## What Makes It Interesting

| Capability | Why it matters |
| --- | --- |
| typed working memory | intermediate state is reusable and debuggable |
| failure-aware recovery | errors are classified and mapped to explicit strategies |
| budget-aware execution | robustness is measured together with cost |
| candidate action ranking | executor does more than blindly follow one action |
| lightweight retrieval subsystem | evidence retrieval is explicit and measurable |
| safety guardrails | risky writes and unsafe queries are visible and gated |
| replay summaries | single runs are easy to present and diagnose |
| ablation-ready design | you can explain which component creates the gain |

## Why The Benchmark Is Credible

- success comes from real execution + validators
- failure and recovery are trajectory-backed
- budgets are explicit
- retrieval quality is measured
- report fields are consistent with saved trajectories

## Good Talking Points In An Interview

### Systems thinking

The codebase separates:
- planning
- execution
- recovery
- validation
- evaluation

### Runtime design

It shows how to build a robust loop instead of a single-shot prompt chain.

### Evaluation rigor

The benchmark avoids hidden success rewriting and keeps explanation close to the data.

### Observability

The project can answer:
- what failed first
- what recovery was attempted
- what memory was used
- what budget was consumed
- why the validator passed or failed

## Recommended Live Demo

1. Run one task
2. Open the trajectory JSON
3. Open the replay summary
4. Open the markdown report

That sequence usually communicates both implementation depth and evaluation maturity very quickly.

## Limitations To State Honestly

- local benchmark, not internet-scale agent evaluation
- lightweight retrieval, not semantic vector search
- estimated token cost, not production billing
- simple branch-aware execution, not a full workflow engine

## Supporting Docs

- [README.md](../README.md)
- [benchmark_methodology.md](benchmark_methodology.md)
- [failure_taxonomy.md](failure_taxonomy.md)
- [ablation_guide.md](ablation_guide.md)
- [trajectory_reading_guide.md](trajectory_reading_guide.md)
- [demo.md](demo.md)
