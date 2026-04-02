# Contributing

Thanks for improving Adaptive-Agent. This project is optimized for reproducible agent evaluation, so changes should preserve determinism, validator quality, and artifact traceability.

## Development setup

```bash
git clone https://github.com/YZT-Arsit/AutoTool-Agent.git
cd AutoTool-Agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[dev]
```

## Local validation

Before opening a pull request, run:

```bash
ruff check .
pytest -q
python -m autotoolbench make-data --seed 0
python -m autotoolbench eval --agent all --seed 0 --noise 0.0
```

You can also use the repo shortcuts:

```bash
make lint
make test
make smoke
```

## Style

- Follow the existing module boundaries: `agent`, `env`, `eval`, `llm`, `tools`, `utils`.
- Prefer explicit types on public functions and data models.
- Keep comments short and engineering-oriented.
- Reuse shared constants and path helpers instead of introducing ad hoc strings.

## Adding a task or validator

1. Update `src/autotoolbench/data_gen.py`.
2. If needed, add the validator to `src/autotoolbench/env/validators.py`.
3. Add or update tests under `tests/`.
4. Regenerate the dataset and verify the benchmark still runs end to end.

## Pull requests

- Keep PRs focused.
- Include a short rationale and validation notes.
- If behavior changes, update the relevant documentation and changelog entry.
- Prefer branch names like `docs/...`, `feat/...`, or `fix/...`.
- Include trajectory or report paths when a change affects benchmark behavior.
