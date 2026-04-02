# Ablation Reading Guide

## Why ablations matter here

AutoToolBench is a systems project, so “overall success rate” is not enough.

Ablations answer:
- which component actually contributes to robustness
- whether an observed gain is methodological or accidental
- whether stronger results come from better runtime behavior or weaker evaluation

## Current ablation variants

### `no_reflector`

Removes explicit reflection and structured recovery recommendation.

Use it to ask:
- how much of the gain comes from failure classification and recovery policy?

### `no_replan`

Keeps the adaptive runtime but removes replanning.

Use it to ask:
- is patching enough, or do some failures require rebuilding the plan?

### `no_budget`

Removes realistic budget pressure.

Use it to ask:
- does the agent stay efficient, or does it only succeed when cost limits are relaxed?

### `no_memory`

Disables working memory.

Use it to ask:
- how much robustness comes from named intermediate state rather than only recent observations?

### `weak_validation`

Reduces validator strictness while keeping basic validation flow.

Use it to ask:
- does the success gain survive under stronger validation, or is it partly an evaluation looseness artifact?

## How to read ablation results

### If `adaptive` beats `no_reflector`

Interpretation:
- failure-aware recovery matters

### If `adaptive` beats `no_memory`

Interpretation:
- named intermediate state is important, especially for retrieval-heavy and long-horizon tasks

### If `adaptive` and `weak_validation` are similar

Interpretation:
- success gains may be methodological, not runtime-driven

### If `no_budget` is much better than `adaptive`

Interpretation:
- the agent may be correct in principle but inefficient in practice

## Best metrics to pair with ablations

Do not only look at:
- `success_rate`

Also look at:
- `success_per_call`
- `success_per_estimated_token`
- `recovery_success_rate`
- `avg_recovery_cost_calls`
- `failure_breakdown`
- `failure_propagation_rate`
- retrieval metrics for `retrieval_heavy`

## Best interview takeaway

The ablation suite shows that the project was built with causal reasoning in mind:
- not just “did it work”
- but “which runtime capability made it work”
