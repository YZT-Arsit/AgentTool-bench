.PHONY: lint test smoke

lint:
	ruff check .

test:
	pytest -q

smoke:
	python -m autotoolbench make-data --seed 0
	python -m autotoolbench eval --agent all --seed 0 --noise 0.0
