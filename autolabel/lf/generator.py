"""LLM-powered labeling function generator."""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Any

from autolabel.lf.base import LabelingFunction
from autolabel.lf.sandbox import SandboxedExecutor
from autolabel.lf.templates import (
    FEW_SHOT_EXAMPLES,
    LANGUAGE_SUPPLEMENTS,
    STRATEGY_TEMPLATES,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Regex to extract fenced Python code blocks from LLM output
_CODE_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```", re.DOTALL
)


class LFGenerator:
    """Generates labeling functions by prompting an LLM provider.

    The generator formats a strategy-specific prompt, calls the LLM,
    parses the response for Python function definitions, validates each
    one through :class:`SandboxedExecutor`, and returns only the functions
    that pass validation.
    """

    def __init__(
        self,
        provider: Any,  # BaseLLMProvider at runtime
        label_space: list[str],
        task_description: str,
        max_lf_lines: int = 100,
        language: str = "en",
        small_model_mode: bool = False,
    ) -> None:
        self.provider = provider
        self.label_space = label_space
        self.task_description = task_description
        self.max_lf_lines = max_lf_lines
        self.language = language
        self.small_model_mode = small_model_mode
        self.max_retries = 2

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        strategy: str,
        target_label: str,
        examples: list[str],
        existing_lf_descriptions: list[str],
        failure_examples: list[str] | None = None,
        num_lfs: int = 5,
        iteration: int = 0,
    ) -> list[LabelingFunction]:
        """Generate labeling functions for a given strategy and label.

        Args:
            strategy: One of the keys in
                :data:`~autolabel.lf.templates.STRATEGY_TEMPLATES`.
            target_label: The label these LFs should vote for.
            examples: Positive example texts for the target label.
            existing_lf_descriptions: Descriptions of LFs already in the
                registry (to avoid duplicates).
            failure_examples: Texts that existing LFs misclassify, used to
                guide the LLM toward better coverage.
            num_lfs: How many LFs to request from the LLM.
            iteration: Current optimisation loop iteration number.

        Returns:
            A list of validated :class:`LabelingFunction` objects.  The
            list may be shorter than *num_lfs* if some generated
            functions fail validation.
        """
        if strategy not in STRATEGY_TEMPLATES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {sorted(STRATEGY_TEMPLATES)}"
            )

        # Small model mode overrides
        effective_num_lfs = min(num_lfs, 3) if self.small_model_mode else num_lfs
        effective_max_tokens = 2048 if self.small_model_mode else 4096

        prompt = self._build_prompt(
            strategy=strategy,
            target_label=target_label,
            examples=examples,
            existing_lf_descriptions=existing_lf_descriptions,
            failure_examples=failure_examples,
            num_lfs=effective_num_lfs,
        )

        logger.info(
            "Generating %d '%s' LFs for label '%s'",
            effective_num_lfs,
            strategy,
            target_label,
        )

        # Retry loop: retry up to max_retries times if 0 valid LFs parsed
        temperature = 0.7
        valid_lfs: list[LabelingFunction] = []

        for attempt in range(1 + self.max_retries):
            if attempt > 0:
                logger.info("Retry %d/%d (lowering temperature)", attempt, self.max_retries)
                temperature = 0.4
                # Add explicit fence reminder on retry
                retry_prompt = (
                    prompt + "\n\nIMPORTANT: Wrap each function in its own "
                    "```python\n...\n``` code fence. Start each function with `def lf_`."
                )
            else:
                retry_prompt = prompt

            response = self.provider.generate(
                prompt=retry_prompt,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )

            raw_functions = self._parse_response(response.text)
            logger.info(
                "Parsed %d code blocks from LLM response (attempt %d)",
                len(raw_functions),
                attempt + 1,
            )

            for idx, source in enumerate(raw_functions):
                # Auto-fix: prepend `import re` if LF uses re. but forgot import
                source = self._auto_fix_imports(source)

                fn_name = self._extract_function_name(source)
                if fn_name is None:
                    logger.warning(
                        "Skipping code block %d – no lf_* function name found",
                        idx,
                    )
                    continue

                effective_max_lines = (
                    min(self.max_lf_lines, 30) if self.small_model_mode else self.max_lf_lines
                )
                ok, reason = SandboxedExecutor.validate_source(
                    source, max_lines=effective_max_lines
                )
                if not ok:
                    logger.warning(
                        "Skipping %s – validation failed: %s", fn_name, reason
                    )
                    continue

                lf = LabelingFunction(
                    name=fn_name,
                    source=source,
                    strategy=strategy,
                    description=self._extract_docstring(source),
                    target_label=target_label,
                    iteration=iteration,
                )

                # Smoke-test compilation
                try:
                    lf.compile()
                except Exception as exc:
                    logger.warning(
                        "Skipping %s – compilation failed: %s", fn_name, exc
                    )
                    continue

                valid_lfs.append(lf)

            if valid_lfs:
                break  # Got at least one valid LF, no need to retry

        logger.info(
            "Returning %d valid LFs",
            len(valid_lfs),
        )
        return valid_lfs

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        strategy: str,
        target_label: str,
        examples: list[str],
        existing_lf_descriptions: list[str],
        failure_examples: list[str] | None,
        num_lfs: int,
    ) -> str:
        template = STRATEGY_TEMPLATES[strategy]

        # Format examples as a numbered list
        examples_str = "\n".join(
            f"  {i + 1}. {ex}" for i, ex in enumerate(examples)
        )

        # Failure examples section
        if failure_examples:
            failure_section = (
                "The following texts were MISCLASSIFIED by existing LFs. "
                "Write new LFs that handle these correctly:\n"
                + "\n".join(
                    f"  - {ex}" for ex in failure_examples
                )
            )
        else:
            failure_section = ""

        # Existing LF descriptions
        if existing_lf_descriptions:
            existing_lfs_str = "\n".join(
                f"  - {desc}" for desc in existing_lf_descriptions
            )
        else:
            existing_lfs_str = "  (none yet)"

        # Language supplement
        language_supplement = LANGUAGE_SUPPLEMENTS.get(self.language, "")
        if language_supplement:
            language_supplement = "\n" + language_supplement + "\n"

        # Few-shot examples (always for small model mode, available in template)
        few_shot_section = FEW_SHOT_EXAMPLES if self.small_model_mode else ""

        return template.format(
            task_description=self.task_description,
            target_label=target_label,
            label_space=", ".join(self.label_space),
            examples=examples_str,
            failure_section=failure_section,
            existing_lfs=existing_lfs_str,
            num_lfs=num_lfs,
            strategy=strategy,
            language_supplement=language_supplement,
            few_shot_section=few_shot_section,
        )

    @staticmethod
    def _auto_fix_imports(source: str) -> str:
        """Prepend ``import re`` if the source uses ``re.`` but lacks the import."""
        if "re." in source and "import re" not in source:
            source = "import re\n" + source
        return source

    @staticmethod
    def _parse_response(text: str) -> list[str]:
        """Extract Python code blocks from the LLM's markdown response."""
        blocks: list[str] = []
        for match in _CODE_FENCE_RE.finditer(text):
            code = textwrap.dedent(match.group(1)).strip()
            if code and "def lf_" in code:
                blocks.append(code)
        return blocks

    @staticmethod
    def _extract_function_name(source: str) -> str | None:
        """Return the first ``lf_*`` function name found in *source*."""
        match = re.search(r"def\s+(lf_\w+)\s*\(", source)
        return match.group(1) if match else None

    @staticmethod
    def _extract_docstring(source: str) -> str:
        """Return the docstring of the first function, or a fallback."""
        try:
            import ast

            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        return docstring
        except Exception:
            pass
        return "(no description)"
