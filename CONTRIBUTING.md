# Contributing to AutoLabel

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/yashpatil/autolabel.git
cd autolabel
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

All tests use `MockLLMProvider` — no API keys needed.

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check autolabel/ tests/
ruff format autolabel/ tests/
```

## Adding a New Dataset

1. Add a loader function in `autolabel/data/loaders.py`
2. Register it in the `DATASET_LOADERS` dict
3. For HuggingFace datasets, use the `_hf_load_and_build()` helper

## Adding a New LF Strategy

1. Add the strategy instructions in `autolabel/lf/templates.py`
2. Register it in `STRATEGY_TEMPLATES`
3. Add it to `autolabel/core/strategy.py` `STRATEGIES` list
4. Add tests

## Adding a New LLM Provider

1. Create a new file in `autolabel/llm/`
2. Implement the `BaseLLMProvider` interface
3. Register in `autolabel/llm/__init__.py`

## Adding a New Language

1. Add language supplement in `autolabel/lf/templates.py` `LANGUAGE_SUPPLEMENTS`
2. Add language note in `autolabel/core/strategy.py` `_LANGUAGE_NOTES`
3. Add dataset loader if available
4. Add tests in `tests/test_multilingual.py`

## Pull Requests

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Run the full test suite before submitting
- Update CHANGELOG.md for user-facing changes
