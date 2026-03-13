"""Tests for LLM provider infrastructure."""

from __future__ import annotations

import pytest

from autolabel.llm import get_provider
from autolabel.llm.base import LLMResponse
from autolabel.llm.cost_tracker import CostTracker
from tests.conftest import MockLLMProvider


class TimeoutCaptureProvider(MockLLMProvider):
    def __init__(self) -> None:
        super().__init__(responses=["structured"])
        self.last_timeout: float | None = None

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_timeout_seconds: float | None = None,
    ) -> LLMResponse:
        self.last_timeout = request_timeout_seconds
        return super().generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )


class TestLLMResponse:
    def test_fields(self):
        r = LLMResponse(
            text="hello", input_tokens=10, output_tokens=5, model="test", provider="test"
        )
        assert r.text == "hello"
        assert r.input_tokens == 10


class TestMockProvider:
    def test_generate(self):
        p = MockLLMProvider(responses=["response 1", "response 2"])
        r1 = p.generate("prompt 1")
        assert r1.text == "response 1"
        r2 = p.generate("prompt 2")
        assert r2.text == "response 2"

    def test_generate_structured(self):
        p = MockLLMProvider(responses=["structured"])
        r = p.generate_structured("prompt")
        assert r.text == "structured"

    def test_generate_structured_forwards_timeout(self):
        p = TimeoutCaptureProvider()
        r = p.generate_structured("prompt", request_timeout_seconds=7.5)
        assert r.text == "structured"
        assert p.last_timeout == 7.5


class TestCostTracker:
    def test_record_and_total(self):
        tracker = CostTracker()
        r = LLMResponse(
            text="x",
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
        )
        tracker.record(r)
        assert tracker.total_cost() > 0
        tokens = tracker.total_tokens()
        assert tokens["input_tokens"] == 1000
        assert tokens["output_tokens"] == 500

    def test_summary(self):
        tracker = CostTracker()
        r = LLMResponse(
            text="x", input_tokens=100, output_tokens=50, model="mock-model", provider="mock"
        )
        tracker.record(r)
        summary = tracker.summary()
        assert "Total" in summary


class TestGetProvider:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_provider("nonexistent")
