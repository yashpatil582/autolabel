"""Classical ML and LLM baselines for benchmarking."""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from autolabel.data.dataset import AutoLabelDataset
from autolabel.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class BaselineRunner:
    """Runs baseline methods on a dataset and returns results."""

    def __init__(self, dataset: AutoLabelDataset, provider: Any = None) -> None:
        self.dataset = dataset
        self.provider = provider
        self.evaluator = Evaluator(dataset)

    def run_all(self) -> list[dict[str, Any]]:
        """Run all baselines and return results."""
        results = []
        results.append(self.run_random())
        results.append(self.run_majority_class())
        results.append(self.run_tfidf_logreg())

        if self.provider is not None:
            results.append(self.run_zero_shot_llm())
            results.append(self.run_few_shot_llm())

        return results

    def run_random(self) -> dict[str, Any]:
        """Random label assignment."""
        random.seed(42)
        predictions = [random.choice(self.dataset.label_space) for _ in self.dataset.test_texts]
        result = self.evaluator.evaluate(predictions, split="test")
        return {"method": "Random", "f1": result.f1, "accuracy": result.accuracy, "coverage": 1.0}

    def run_majority_class(self) -> dict[str, Any]:
        """Always predict the most common class."""
        counter = Counter(self.dataset.train_labels)
        majority = counter.most_common(1)[0][0]
        predictions = [majority] * len(self.dataset.test_texts)
        result = self.evaluator.evaluate(predictions, split="test")
        return {"method": "Majority Class", "f1": result.f1, "accuracy": result.accuracy, "coverage": 1.0}

    def run_tfidf_logreg(self) -> dict[str, Any]:
        """TF-IDF + Logistic Regression."""
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 3), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipeline.fit(self.dataset.train_texts, self.dataset.train_labels)
        predictions = pipeline.predict(self.dataset.test_texts).tolist()
        result = self.evaluator.evaluate(predictions, split="test")
        return {
            "method": "TF-IDF + LogReg",
            "f1": result.f1,
            "accuracy": result.accuracy,
            "coverage": 1.0,
        }

    def run_zero_shot_llm(self) -> dict[str, Any]:
        """Zero-shot LLM classification."""
        if self.provider is None:
            return {"method": "Zero-shot LLM", "f1": 0.0, "accuracy": 0.0, "coverage": 0.0}

        labels_str = ", ".join(self.dataset.label_space)
        predictions = []

        for text in self.dataset.test_texts:
            prompt = (
                f"Task: {self.dataset.task_description}\n"
                f"Labels: {labels_str}\n\n"
                f"Text: {text}\n\n"
                f"Reply with ONLY the label name, nothing else."
            )
            try:
                response = self.provider.generate_structured(prompt=prompt)
                pred = response.text.strip()
                # Try to match to valid label
                matched = self._match_label(pred)
                predictions.append(matched)
            except Exception as e:
                logger.warning("Zero-shot failed for text: %s", e)
                predictions.append(None)

        result = self.evaluator.evaluate(predictions, split="test")
        from autolabel.evaluation.metrics import compute_coverage
        return {
            "method": "Zero-shot LLM",
            "f1": result.f1,
            "accuracy": result.accuracy,
            "coverage": compute_coverage(predictions),
        }

    def run_few_shot_llm(self, n_examples: int = 3) -> dict[str, Any]:
        """Few-shot LLM classification."""
        if self.provider is None:
            return {"method": "Few-shot LLM", "f1": 0.0, "accuracy": 0.0, "coverage": 0.0}

        labels_str = ", ".join(self.dataset.label_space)

        # Build few-shot examples from training data
        examples_by_label: dict[str, list[str]] = {}
        for text, label in zip(self.dataset.train_texts, self.dataset.train_labels):
            examples_by_label.setdefault(label, []).append(text)

        few_shot_str = ""
        for label in self.dataset.label_space:
            exs = examples_by_label.get(label, [])[:n_examples]
            for ex in exs:
                few_shot_str += f"Text: {ex}\nLabel: {label}\n\n"

        predictions = []
        for text in self.dataset.test_texts:
            prompt = (
                f"Task: {self.dataset.task_description}\n"
                f"Labels: {labels_str}\n\n"
                f"Examples:\n{few_shot_str}"
                f"Text: {text}\n\n"
                f"Reply with ONLY the label name, nothing else."
            )
            try:
                response = self.provider.generate_structured(prompt=prompt)
                pred = response.text.strip()
                matched = self._match_label(pred)
                predictions.append(matched)
            except Exception as e:
                logger.warning("Few-shot failed for text: %s", e)
                predictions.append(None)

        result = self.evaluator.evaluate(predictions, split="test")
        from autolabel.evaluation.metrics import compute_coverage
        return {
            "method": "Few-shot LLM",
            "f1": result.f1,
            "accuracy": result.accuracy,
            "coverage": compute_coverage(predictions),
        }

    def _match_label(self, pred: str) -> str | None:
        """Match an LLM prediction to the closest valid label."""
        pred_lower = pred.lower().strip()
        # Exact match
        for label in self.dataset.label_space:
            if label.lower() == pred_lower:
                return label
        # Substring match
        for label in self.dataset.label_space:
            if label.lower() in pred_lower or pred_lower in label.lower():
                return label
        return None
