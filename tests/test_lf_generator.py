"""Tests for the LF generator."""

from autolabel.lf.generator import LFGenerator
from tests.conftest import MockLLMProvider


MOCK_LLM_RESPONSE = '''Here are 2 labeling functions:

```python
def lf_keyword_delta_01(text: str):
    """Checks for 'delta' keyword."""
    if "delta" in text.lower():
        return "Delta Air Lines"
    return None
```

```python
def lf_keyword_delta_02(text: str):
    """Checks for 'DL' abbreviation."""
    if " DL " in text or text.startswith("DL "):
        return "Delta Air Lines"
    return None
```
'''


def test_generator_parses_lfs():
    provider = MockLLMProvider(responses=[MOCK_LLM_RESPONSE])
    gen = LFGenerator(
        provider=provider,
        label_space=["Air Canada", "Delta Air Lines", "United Airlines"],
        task_description="Extract airline",
    )
    lfs = gen.generate(
        strategy="keyword",
        target_label="Delta Air Lines",
        examples=["I flew Delta today"],
        existing_lf_descriptions=[],
        num_lfs=2,
        iteration=1,
    )
    assert len(lfs) == 2
    assert lfs[0].name == "lf_keyword_delta_01"
    assert lfs[1].name == "lf_keyword_delta_02"
    assert all(lf.strategy == "keyword" for lf in lfs)


def test_generator_filters_invalid():
    bad_response = '''```python
def lf_keyword_bad_01(text: str):
    import os
    return os.environ.get("SECRET")
```

```python
def lf_keyword_good_01(text: str):
    """Good function."""
    if "test" in text.lower():
        return "Test"
    return None
```
'''
    provider = MockLLMProvider(responses=[bad_response])
    gen = LFGenerator(
        provider=provider,
        label_space=["Test"],
        task_description="test",
    )
    lfs = gen.generate(
        strategy="keyword",
        target_label="Test",
        examples=["test text"],
        existing_lf_descriptions=[],
        num_lfs=2,
    )
    assert len(lfs) == 1
    assert lfs[0].name == "lf_keyword_good_01"
