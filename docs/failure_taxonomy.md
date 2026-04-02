# Failure Taxonomy

## Why a taxonomy exists

AutoToolBench does not treat all failures as one bucket.

The taxonomy exists so we can distinguish:
- where the problem started
- whether recovery was appropriate
- whether the system became more or less stable over time

## Labels

### `BAD_TOOL_ARGS`

Meaning:
- the tool call exists, but the generated arguments are wrong, incomplete, or malformed for that tool

Typical recovery:
- `patch_args`

### `EMPTY_RESULT`

Meaning:
- the tool call executed, but returned no useful output

Typical recovery:
- adjust query/condition, then retry

### `MISSING_PREREQUISITE`

Meaning:
- the current step depends on evidence or state that was never prepared

Typical recovery:
- `replan`

### `PLAN_MISMATCH`

Meaning:
- the executed action diverged from the plan in a way that is unlikely to satisfy the task

Typical recovery:
- `replan`

### `TOOL_NOT_FOUND`

Meaning:
- the selected tool is not available in the registry

Typical recovery:
- `fail_fast`

### `JSON_MALFORMED`

Meaning:
- planner/action/reflection JSON could not be parsed or validated cleanly

Typical recovery:
- `retry_safe`

### `TOOL_EXECUTION_FAILED`

Meaning:
- the tool was invoked, but execution failed at runtime

Typical recovery:
- depends on error shape
- may patch args
- may fail fast

### `VALIDATION_FAILED`

Meaning:
- execution completed, but produced the wrong artifact or insufficient evidence

Typical recovery:
- patch if close
- otherwise replan

### `BUDGET_EXHAUSTED`

Meaning:
- the remaining budget cannot support the next action or tool call

Typical recovery:
- `terminate`

## Failure origin vs final failure

A trajectory may have:
- a first failure stage
- a final failure label
- recovery attempts in between

This matters because a stable agent is not only one that fails less, but one that:
- contains failures earlier
- avoids propagating them
- recovers them at lower cost

## How to use the taxonomy in discussion

Good interview framing:

- `BAD_TOOL_ARGS` means the planner/executor interface is brittle.
- `MISSING_PREREQUISITE` means the plan structure is weak.
- `VALIDATION_FAILED` means the runtime got close but did not finish correctly.
- `BUDGET_EXHAUSTED` means the agent design may be robust but inefficient.

That distinction makes the benchmark much more informative than a single success rate.
