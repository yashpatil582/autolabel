"""Base labeling function data structure and ABSTAIN sentinel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

ABSTAIN = "__ABSTAIN__"


@dataclass
class LabelingFunction:
    """A single labeling function that votes on text classification.

    Each LF encapsulates a small Python function that examines a text input
    and either returns a target label string or ``ABSTAIN`` to indicate that
    it has no opinion.

    Attributes:
        name: Unique identifier, e.g. ``"lf_keyword_delta_01"``.
        source: Python source code of the function body.
        strategy: Generation strategy used (e.g. keyword, regex, semantic).
        description: Human-readable explanation of what the LF detects.
        target_label: The label this LF votes for (e.g. ``"Delta Air Lines"``).
        iteration: Which optimisation loop iteration created this LF.
    """

    name: str
    source: str
    strategy: str
    description: str
    target_label: str
    iteration: int

    _compiled_fn: Callable[[str], str | None] | None = field(default=None, repr=False)

    def compile(self) -> None:
        """Compile *source* into a callable function.

        The compiled function is stored in ``_compiled_fn``.  Only the
        ``re`` standard-library module and the ``ABSTAIN`` sentinel are
        injected into the execution namespace.

        Raises:
            ValueError: If no function whose name starts with ``lf_`` is
                found in the compiled source.
        """
        namespace: dict = {"re": __import__("re"), "ABSTAIN": ABSTAIN}
        exec(self.source, namespace)  # noqa: S102 – validated before reaching here
        for v in namespace.values():
            if callable(v) and not isinstance(v, type) and getattr(v, "__name__", "") != "":
                if v.__name__.startswith("lf_"):
                    self._compiled_fn = v
                    break
        if self._compiled_fn is None:
            raise ValueError(f"No lf_* function found in source for {self.name}")

    def apply(self, text: str) -> str:
        """Apply this labeling function to *text*.

        Returns the target label string when the LF matches, or
        :data:`ABSTAIN` otherwise.
        """
        if self._compiled_fn is None:
            self.compile()
        assert self._compiled_fn is not None
        result = self._compiled_fn(text)
        return result if result is not None else ABSTAIN
