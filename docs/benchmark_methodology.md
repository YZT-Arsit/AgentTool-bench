# Benchmark Methodology

## Goal

AutoToolBench evaluates whether an agent can complete tool-use tasks under realistic structured failure, not just whether it can emit plausible text.

The benchmark is meant to answer:
- Can the agent pick the right tool?
- Can it generate valid arguments?
- Can it preserve and reuse intermediate state?
- Can it recover when execution goes wrong?
- Can it do so under explicit budget constraints?

## Core Principles

### 1. Success comes from execution plus validation

Primary task outcome is determined by:
- real tool execution
- persisted artifacts
- validator results

It is not determined by:
- prompt heuristics
- runner-side patching of success labels
- “close enough” post hoc result rewriting

### 2. Failures should be explainable

Each trajectory should make it possible to answer:
- where the first failure started
- what failure label was assigned
- what recovery strategy was recommended
- what actually happened next
- why the task still failed or eventually succeeded

### 3. Cost matters

The runtime tracks:
- calls
- steps
- runtime
- estimated tokens
- estimated tool cost

This allows the benchmark to compare not only success, but success-per-cost and recovery cost.

### 4. The benchmark should remain local and lightweight

The project intentionally uses:
- local files
- local SQLite
- local logs
- heuristic retrieval
- mock LLM corruption for stress tests

This keeps the system reproducible and understandable.

## Task Families

Current task families are designed to expose different failure modes:

- `single_tool_easy`
  Simple correctness sanity checks.
- `multi_tool_chain`
  Basic chaining and artifact production.
- `prerequisite_dependency`
  Requires ordering and dependency awareness.
- `tool_confusion`
  Exposes tool selection mistakes.
- `args_brittle`
  Exposes argument generation fragility.
- `budget_tradeoff`
  Rewards efficient planning under constraints.
- `retrieval_heavy`
  Requires explicit evidence retrieval before downstream actions.
- `ambiguity_heavy`
  Requires better tool-path selection under vague instructions.
- `long_horizon`
  Tests longer dependency chains and state carrying.
- `partial_success`
  Distinguishes incomplete-but-promising outputs from hard failure.

## Validation Philosophy

Validators are intentionally artifact-centric.

Examples include:
- exact SQL row checks
- file regex checks
- JSON schema checks
- retrieval evidence quality checks
- multi-artifact aggregation

Structured validator result supports:
- `full_success`
- `partial_success`
- `failure`

Compatibility note:
- the old boolean `validate()` path is still supported
- benchmark result semantics still treat only `full_success` as success

## Recovery Methodology

Recovery is explicitly labeled and tracked.

The system distinguishes at least:
- `BAD_TOOL_ARGS`
- `EMPTY_RESULT`
- `MISSING_PREREQUISITE`
- `PLAN_MISMATCH`
- `TOOL_NOT_FOUND`
- `JSON_MALFORMED`
- `TOOL_EXECUTION_FAILED`
- `VALIDATION_FAILED`
- `BUDGET_EXHAUSTED`

Recovery strategies include:
- `patch_args`
- `patch_tool`
- `replan`
- `retry_safe`
- `fail_fast`
- `terminate`

## Why This Is Useful In Interviews

This methodology helps explain that the project is not “just an agent demo”.

It demonstrates:
- runtime design
- evaluation rigor
- systems thinking around observability and failure analysis
- realistic tradeoffs between robustness and cost
