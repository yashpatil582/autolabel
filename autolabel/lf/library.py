"""Persistent LF library for cross-dataset transfer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from autolabel.lf.base import LabelingFunction

logger = logging.getLogger(__name__)


class LFLibrary:
    """Stores and retrieves high-scoring LFs for transfer across datasets.

    LFs are persisted as JSON with their source code, strategy, scores,
    and domain metadata. On a new run, transferable LFs can be loaded,
    adapted to the new domain, and seeded into the registry.
    """

    def __init__(self, library_path: str | Path) -> None:
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.library_path / "lf_index.json"
        self._index: list[dict[str, Any]] = self._load_index()

    def save(
        self,
        lfs: list[LabelingFunction],
        domain: str,
        scores: dict[str, float] | None = None,
    ) -> int:
        """Persist high-scoring LFs to the library.

        Args:
            lfs: Labeling functions to save.
            domain: Domain identifier (e.g., dataset name).
            scores: Optional dict mapping LF name to composite score.

        Returns:
            Number of LFs saved.
        """
        scores = scores or {}
        saved = 0

        for lf in lfs:
            entry = {
                "name": lf.name,
                "source": lf.source,
                "strategy": lf.strategy,
                "description": lf.description,
                "target_label": lf.target_label,
                "domain": domain,
                "score": scores.get(lf.name, 0.0),
                "abstract_pattern": getattr(lf, "abstract_pattern", ""),
                "transferability_score": getattr(lf, "transferability_score", 0.0),
            }

            # Check for duplicate
            if not any(e["name"] == lf.name and e["domain"] == domain for e in self._index):
                self._index.append(entry)
                saved += 1

        self._save_index()
        logger.info("Saved %d LFs to library for domain '%s'", saved, domain)
        return saved

    def find_transferable(
        self,
        target_domain: str,
        label_space: list[str],
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Find LFs that may transfer to a new domain.

        Filters by:
        - Strategy type (keyword/regex/semantic are most transferable)
        - Score threshold
        - Label overlap with target domain

        Args:
            target_domain: The domain to transfer to.
            label_space: Label space of the target task.
            min_score: Minimum composite score to consider.

        Returns:
            List of LF entry dicts.
        """
        transferable_strategies = {"keyword", "regex", "semantic", "context", "compositional"}

        candidates = []
        for entry in self._index:
            # Skip same domain
            if entry["domain"] == target_domain:
                continue

            # Strategy filter
            if entry["strategy"] not in transferable_strategies:
                continue

            # Score filter
            if entry.get("score", 0) < min_score:
                continue

            candidates.append(entry)

        # Sort by score descending
        candidates.sort(key=lambda e: e.get("score", 0), reverse=True)
        return candidates[:20]  # Cap at 20

    def adapt_lf(
        self,
        entry: dict[str, Any],
        provider: Any,
        new_domain: str,
        new_label_space: list[str],
        task_description: str,
    ) -> LabelingFunction | None:
        """Ask the LLM to adapt an LF from the library to a new domain.

        Args:
            entry: Library entry dict with source code and metadata.
            provider: LLM provider for adaptation.
            new_domain: Target domain name.
            new_label_space: Labels in the target task.
            task_description: Description of the target task.

        Returns:
            Adapted LabelingFunction or None if adaptation fails.
        """
        from autolabel.lf.sandbox import SandboxedExecutor

        prompt = (
            f"Adapt this labeling function for a new task.\n\n"
            f"Original function (domain: {entry['domain']}, "
            f"label: {entry['target_label']}):\n"
            f"```python\n{entry['source']}\n```\n\n"
            f"New task: {task_description}\n"
            f"New labels: {', '.join(new_label_space)}\n\n"
            f"Rewrite the function for the new task. Keep the same strategy "
            f"({entry['strategy']}) but adapt keywords, patterns, and the "
            f"return label to match the new domain.\n\n"
            f"Return the adapted function in a ```python``` code fence."
        )

        try:
            response = provider.generate(prompt=prompt, temperature=0.3, max_tokens=1024)

            import re

            match = re.search(r"```(?:python)?\s*\n(.*?)```", response.text, re.DOTALL)
            if not match:
                return None

            source = match.group(1).strip()

            # Validate
            ok, reason = SandboxedExecutor.validate_source(source, max_lines=50)
            if not ok:
                logger.warning("Adapted LF failed validation: %s", reason)
                return None

            # Extract function name
            fn_match = re.search(r"def\s+(lf_\w+)\s*\(", source)
            if not fn_match:
                return None

            fn_name = fn_match.group(1)

            # Determine target label from the source
            target_label = None
            for label in new_label_space:
                if f'"{label}"' in source or f"'{label}'" in source:
                    target_label = label
                    break
            if target_label is None:
                target_label = new_label_space[0]

            lf = LabelingFunction(
                name=fn_name,
                source=source,
                strategy=entry["strategy"],
                description=f"Adapted from {entry['domain']}: {entry['description']}",
                target_label=target_label,
                iteration=0,
            )
            lf.compile()
            return lf

        except Exception as e:
            logger.warning("LF adaptation failed: %s", e)
            return None

    def _load_index(self) -> list[dict[str, Any]]:
        """Load the library index from disk."""
        if self._index_file.exists():
            try:
                return json.loads(self._index_file.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_index(self) -> None:
        """Persist the library index to disk."""
        self._index_file.write_text(json.dumps(self._index, indent=2))

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        domains = set(e.get("domain", "?") for e in self._index)
        return f"LFLibrary(entries={len(self._index)}, domains={domains})"
