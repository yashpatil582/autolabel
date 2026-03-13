"""Data loading and dataset abstractions for AutoLabel."""

from __future__ import annotations

from autolabel.data.dataset import AutoLabelDataset
from autolabel.data.loaders import DATASET_LOADERS, load_airline_tweets

__all__ = [
    "AutoLabelDataset",
    "DATASET_LOADERS",
    "load_airline_tweets",
]
