.PHONY: install dev test lint run benchmark

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

coverage:
	pytest tests/ --cov=autolabel --cov-report=html --cov-report=term

lint:
	ruff check autolabel/ tests/
	ruff format --check autolabel/ tests/

format:
	ruff format autolabel/ tests/

run:
	autolabel run -d airline_tweets -t "Extract airline names" -p anthropic -n 50

benchmark:
	autolabel benchmark -d airline_tweets -p anthropic
