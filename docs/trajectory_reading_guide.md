# How To Read A Trajectory And Replay Summary

## What to open first

For a single task run, the most useful files are:
- trajectory JSON under `runs/.../trajectories/...`
- replay summary markdown under the same directory
- latest markdown report if you want scenario context

## Recommended reading order

### 1. Task header

Look at:
- `task_id`
- `task_type`
- `budget_mode`
- `agent`

This tells you what kind of failure or behavior to expect.

### 2. Step sequence

For each step, inspect:
- `tool`
- `input`
- `output`
- `error`
- `failure_label`

This is the fastest way to reconstruct the runtime path.

### 3. Recovery fields

If a step failed or looked suspicious, check:
- `recommended_strategy`
- `actual_recovery_action`
- `recovery_reason`

These fields explain whether the runtime:
- patched
- replanned
- retried conservatively
- failed fast
- terminated

### 4. Working memory

Check:
- `memory_delta`
- `memory_before`
- `memory_after`
- `referenced_memory_keys`

This reveals whether the agent:
- stored useful intermediate state
- reused it later
- referenced the wrong slot

### 5. Budget usage

Check:
- per-step `budget`
- final `budget_usage`

This helps answer whether failure was:
- a correctness issue
- a recovery issue
- or simply a cost issue

### 6. Validation

Finally inspect final `validation`:
- `validator`
- `status`
- `message`
- `details`

That tells you why the task counted as success, partial success, or failure.

## How to read the replay summary

Replay summary is the human-friendly version of the trajectory.

For a successful case, look for:
- key tool path
- whether recovery happened
- why validator passed

For a failed case, look for:
- first failure stage
- final failure label
- which recovery attempts were tried
- why validation still did not pass

## Best interview use

A replay summary is often the best live demo artifact because it is:
- short
- readable
- grounded in real run data

It lets you explain both system design and debugging ability without scrolling through large JSON blobs.
