"""Tests for the LF applicator."""

from autolabel.lf.applicator import LFApplicator
from autolabel.lf.base import LabelingFunction


def _make_lf(name: str, keyword: str, label: str) -> LabelingFunction:
    source = f'''def {name}(text: str):
    if "{keyword}" in text.lower():
        return "{label}"
    return None
'''
    return LabelingFunction(
        name=name,
        source=source,
        strategy="keyword",
        description=f"Detects {keyword}",
        target_label=label,
        iteration=0,
    )


def test_apply_lfs_basic():
    lfs = [
        _make_lf("lf_delta", "delta", "Delta Air Lines"),
        _make_lf("lf_united", "united", "United Airlines"),
    ]
    texts = [
        "I love Delta flights",
        "United is okay",
        "No airline mentioned",
    ]
    label_space = ["Delta Air Lines", "United Airlines"]
    matrix = LFApplicator.apply_lfs(lfs, texts, label_space)

    assert matrix.shape == (3, 2)
    assert matrix[0, 0] == 0  # Delta LF votes Delta
    assert matrix[0, 1] == -1  # United LF abstains
    assert matrix[1, 0] == -1
    assert matrix[1, 1] == 1  # United LF votes United
    assert matrix[2, 0] == -1
    assert matrix[2, 1] == -1  # both abstain


def test_apply_lfs_empty():
    matrix = LFApplicator.apply_lfs([], ["text"], ["A"])
    assert matrix.shape == (1, 0)
