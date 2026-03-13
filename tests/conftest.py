"""Shared fixtures for AutoLabel tests."""

from __future__ import annotations

import pytest
import numpy as np

from autolabel.data.dataset import AutoLabelDataset
from autolabel.lf.base import LabelingFunction
from autolabel.llm.base import BaseLLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """Deterministic LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="mock-model")
        self._responses = responses or ["Mock response"]
        self._call_idx = 0

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        text = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1
        return LLMResponse(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            model="mock-model",
            provider="mock",
        )


# ---------------------------------------------------------------------------
# Sample dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dataset() -> AutoLabelDataset:
    """Small 3-class dataset for testing."""
    texts = [
        "I love flying with Delta, great service!",
        "Delta has the best in-flight entertainment",
        "My Delta flight was on time",
        "United Airlines lost my baggage again",
        "United is always delayed",
        "Flew United from Chicago, not impressed",
        "Air Canada has friendly staff",
        "Air Canada serves great food",
        "I enjoyed my Air Canada experience",
        "Delta rerouted me without notice",
        "United finally improved their meals",
        "Air Canada cancelled my connection",
    ]
    labels = [
        "Delta Air Lines", "Delta Air Lines", "Delta Air Lines",
        "United Airlines", "United Airlines", "United Airlines",
        "Air Canada", "Air Canada", "Air Canada",
        "Delta Air Lines", "United Airlines", "Air Canada",
    ]
    return AutoLabelDataset(
        name="test_airlines",
        task_description="Extract the airline mentioned in this tweet",
        label_space=["Air Canada", "Delta Air Lines", "United Airlines"],
        texts=texts,
        labels=labels,
        train_indices=[0, 1, 3, 4, 6, 7],
        dev_indices=[2, 5, 8],
        test_indices=[9, 10, 11],
    )


@pytest.fixture
def sample_label_matrix() -> np.ndarray:
    """5 samples, 3 LFs, 3 classes. -1 = abstain."""
    return np.array([
        [0,  0, -1],  # two votes for class 0
        [1, -1,  1],  # two votes for class 1
        [2,  2,  2],  # unanimous class 2
        [-1, 0,  1],  # conflict
        [-1, -1, -1], # no coverage
    ])


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def sample_lf() -> LabelingFunction:
    """A simple keyword LF for testing."""
    return LabelingFunction(
        name="lf_keyword_delta_01",
        source='def lf_keyword_delta_01(text: str):\n    """Detects Delta by keyword."""\n    if "delta" in text.lower():\n        return "Delta Air Lines"\n    return None\n',
        strategy="keyword",
        description="Detects Delta by keyword match",
        target_label="Delta Air Lines",
        iteration=1,
    )
