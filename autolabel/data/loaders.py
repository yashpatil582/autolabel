"""Dataset loader functions for AutoLabel.

Each loader returns an :class:`AutoLabelDataset` with pre-computed
train / dev / test splits.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split

from autolabel.data.dataset import AutoLabelDataset

logger = logging.getLogger(__name__)

_SEED = 42


# ---------------------------------------------------------------------------
# Helper: stratified three-way split
# ---------------------------------------------------------------------------


def _stratified_split(
    n: int,
    labels: list[str],
    train_frac: float = 0.60,
    dev_frac: float = 0.15,
    seed: int = _SEED,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train_indices, dev_indices, test_indices) via stratified split."""
    indices = np.arange(n)
    test_frac = 1.0 - train_frac - dev_frac

    # First split: train vs (dev + test)
    train_idx, rest_idx, train_labels, rest_labels = train_test_split(
        indices,
        labels,
        test_size=dev_frac + test_frac,
        stratify=labels,
        random_state=seed,
    )

    # Second split: dev vs test within the remainder
    relative_dev = dev_frac / (dev_frac + test_frac)
    dev_idx, test_idx = train_test_split(
        rest_idx,
        test_size=1.0 - relative_dev,
        stratify=rest_labels,
        random_state=seed,
    )

    return sorted(train_idx.tolist()), sorted(dev_idx.tolist()), sorted(test_idx.tolist())


# ---------------------------------------------------------------------------
# Local dataset loaders
# ---------------------------------------------------------------------------


def load_airline_tweets(data_dir: Path) -> AutoLabelDataset:
    """Load the airline-tweets dataset from a local JSONL file.

    Expected path: ``<data_dir>/airline_tweets/dataset_airlines.jsonl``

    Each JSON row has keys *uid*, *tweet*, and *labels* (a stringified Python
    dict with an ``'airlines'`` key mapping to a single-element list).
    """
    filepath = data_dir / "airline_tweets" / "dataset_airlines.jsonl"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Airline tweets dataset not found at {filepath}. "
            "Make sure the datasets/ directory is in the project root."
        )

    texts: list[str] = []
    labels: list[str] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            texts.append(row["tweet"])
            # labels field is a stringified Python dict, e.g.
            # "{'airlines': ['Air Canada'], 'topic': 'Clean aircraft'}"
            label_dict = ast.literal_eval(row["labels"])
            airline = label_dict["airlines"][0]
            labels.append(airline)

    label_space = sorted(set(labels))
    train_idx, dev_idx, test_idx = _stratified_split(len(texts), labels)

    return AutoLabelDataset(
        name="airline_tweets",
        task_description="Extract the airline mentioned in this tweet",
        label_space=label_space,
        texts=texts,
        labels=labels,
        train_indices=train_idx,
        dev_indices=dev_idx,
        test_indices=test_idx,
    )


# ---------------------------------------------------------------------------
# HuggingFace dataset loaders
# ---------------------------------------------------------------------------


def _try_import_datasets():
    """Import the ``datasets`` library or raise a helpful error."""
    try:
        import datasets  # noqa: F811

        return datasets
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to load HuggingFace datasets. "
            "Install it with:  pip install datasets"
        )


def _hf_load_and_build(
    hf_path: str,
    hf_name: str | None,
    text_column: str,
    label_column: str,
    label_names: list[str] | None,
    dataset_name: str,
    task_description: str,
    max_examples: int = 2000,
    split: str = "train",
) -> AutoLabelDataset:
    """Generic helper to load a HuggingFace dataset and wrap it."""
    hf_datasets = _try_import_datasets()

    logger.info("Loading HuggingFace dataset %s (config=%s) ...", hf_path, hf_name)
    ds = hf_datasets.load_dataset(hf_path, hf_name, split=split, trust_remote_code=True)

    # Subsample if needed
    if len(ds) > max_examples:
        ds = ds.shuffle(seed=_SEED).select(range(max_examples))

    texts: list[str] = ds[text_column]
    raw_labels = ds[label_column]

    # Resolve label names
    if label_names is None:
        # Try to get from dataset features
        features = ds.features[label_column]
        if hasattr(features, "names"):
            label_names = features.names
        else:
            label_names = sorted(set(str(lbl) for lbl in raw_labels))

    # Convert integer labels to string names
    if isinstance(raw_labels[0], int):
        labels = [label_names[i] for i in raw_labels]
    else:
        labels = [str(lbl) for lbl in raw_labels]

    label_space = sorted(set(labels))
    train_idx, dev_idx, test_idx = _stratified_split(len(texts), labels)

    return AutoLabelDataset(
        name=dataset_name,
        task_description=task_description,
        label_space=label_space,
        texts=texts,
        labels=labels,
        train_indices=train_idx,
        dev_indices=dev_idx,
        test_indices=test_idx,
    )


def load_imdb(data_dir: Path) -> AutoLabelDataset:
    """Load the IMDB sentiment dataset from HuggingFace (first 2000 examples)."""
    return _hf_load_and_build(
        hf_path="imdb",
        hf_name=None,
        text_column="text",
        label_column="label",
        label_names=["neg", "pos"],
        dataset_name="imdb",
        task_description="Classify the sentiment of this movie review as positive or negative",
        max_examples=2000,
        split="train",
    )


def load_ag_news(data_dir: Path) -> AutoLabelDataset:
    """Load the AG News topic dataset from HuggingFace (first 2000 examples)."""
    return _hf_load_and_build(
        hf_path="ag_news",
        hf_name=None,
        text_column="text",
        label_column="label",
        label_names=["World", "Sports", "Business", "Sci/Tech"],
        dataset_name="ag_news",
        task_description="Classify this news article into one of: World, Sports, Business, Sci/Tech",
        max_examples=2000,
        split="train",
    )


def load_yelp(data_dir: Path) -> AutoLabelDataset:
    """Load the Yelp review sentiment dataset from HuggingFace (first 2000 examples)."""
    return _hf_load_and_build(
        hf_path="yelp_review_full",
        hf_name=None,
        text_column="text",
        label_column="label",
        label_names=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        dataset_name="yelp",
        task_description="Classify this Yelp review by its star rating (1-5)",
        max_examples=2000,
        split="train",
    )


def load_sms_spam(data_dir: Path) -> AutoLabelDataset:
    """Load the SMS Spam dataset from HuggingFace (first 2000 examples)."""
    return _hf_load_and_build(
        hf_path="sms_spam",
        hf_name=None,
        text_column="sms",
        label_column="label",
        label_names=["ham", "spam"],
        dataset_name="sms_spam",
        task_description="Classify this SMS message as ham (legitimate) or spam",
        max_examples=2000,
        split="train",
    )


def load_trec(data_dir: Path) -> AutoLabelDataset:
    """Load the TREC question classification dataset from HuggingFace (first 2000 examples)."""
    return _hf_load_and_build(
        hf_path="trec",
        hf_name=None,
        text_column="text",
        label_column="coarse_label",
        label_names=["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"],
        dataset_name="trec",
        task_description="Classify this question into one of: ABBR, ENTY, DESC, HUM, LOC, NUM",
        max_examples=2000,
        split="train",
    )


# ---------------------------------------------------------------------------
# Multilingual dataset loaders
# ---------------------------------------------------------------------------


def load_hindi_headlines(data_dir: Path) -> AutoLabelDataset:
    """Load a Hindi news headline classification dataset from HuggingFace.

    Uses the IndicNLP News Article Dataset (Hindi subset).
    """
    return _hf_load_and_build(
        hf_path="ai4bharat/IndicNLP-News-Articles",
        hf_name="hi",
        text_column="text",
        label_column="label",
        label_names=None,  # auto-detect from features
        dataset_name="hindi_headlines",
        task_description="इस समाचार शीर्षक को उचित श्रेणी में वर्गीकृत करें (Classify this Hindi news headline)",
        max_examples=2000,
        split="train",
    )


def load_marathi_headlines(data_dir: Path) -> AutoLabelDataset:
    """Load a Marathi news headline classification dataset from HuggingFace.

    Uses the IndicNLP News Article Dataset (Marathi subset).
    """
    return _hf_load_and_build(
        hf_path="ai4bharat/IndicNLP-News-Articles",
        hf_name="mr",
        text_column="text",
        label_column="label",
        label_names=None,  # auto-detect from features
        dataset_name="marathi_headlines",
        task_description="या मराठी बातमीच्या शीर्षकाचे वर्गीकरण करा (Classify this Marathi news headline)",
        max_examples=2000,
        split="train",
    )


# ---------------------------------------------------------------------------
# Generic unlabeled loader (Feature 3)
# ---------------------------------------------------------------------------


def load_unlabeled(
    filepath: Path,
    label_space: list[str],
    task_description: str,
    name: str = "unlabeled",
) -> AutoLabelDataset:
    """Load unlabeled texts from a plain text file (one text per line).

    Returns a dataset with empty labels, suitable for zero-label bootstrap.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Unlabeled text file not found: {filepath}")

    texts: list[str] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(line)

    return AutoLabelDataset.from_unlabeled(
        texts=texts,
        label_space=label_space,
        task_description=task_description,
        name=name,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_LOADERS: dict[str, Callable[[Path], AutoLabelDataset]] = {
    "airline_tweets": load_airline_tweets,
    "imdb": load_imdb,
    "ag_news": load_ag_news,
    "yelp": load_yelp,
    "sms_spam": load_sms_spam,
    "trec": load_trec,
    "hindi_headlines": load_hindi_headlines,
    "marathi_headlines": load_marathi_headlines,
}
