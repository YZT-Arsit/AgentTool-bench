# Project Roadmap

This roadmap is intentionally lightweight. The project aims to stay readable and benchmark-first rather than grow into a large orchestration framework.

## Near-term improvements

- stronger `partial_success` reporting in summary and report outputs
- more explicit branch-aware benchmark tasks
- richer retrieval quality breakdown by task type
- improved single-task replay CLI ergonomics

## Medium-term improvements

- more task families around ambiguity, recovery, and long-horizon coordination
- deeper branch-aware recovery analysis
- more explicit validator composability and validation debugging UX
- clearer comparison baselines and fixed regression snapshots

## Non-goals

- building a hosted platform
- introducing a large workflow engine
- replacing local reproducibility with external infrastructure dependencies
- turning the benchmark into a generic agent framework with many plugins

## Guiding principle

When there is a tradeoff, prefer:
- rigor over flashy demos
- explainability over cleverness
- lightweight structure over heavyweight abstractions
