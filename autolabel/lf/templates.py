"""Prompt templates used by LFGenerator to instruct the LLM."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are an expert Python programmer specializing in weak supervision and \
programmatic labeling. Your task is to write small, focused Python labeling \
functions (LFs) that classify text into a set of known labels.

Rules for every function you write:
1. The function signature MUST be `def lf_<strategy>_<label_slug>_<number>(text: str):`.
2. The function takes a single argument `text` (a plain string) and returns \
either the target label as a string literal, or `None` to abstain.
3. You may ONLY use: built-in string methods, the `re` module, basic Python \
control flow (if/elif/else, for, while), and built-in data structures \
(list, dict, set, tuple).
4. Do NOT import anything other than `re`.
5. Do NOT use `open`, `exec`, `eval`, `os`, `sys`, `subprocess`, or any I/O.
6. Each function must be self-contained and stateless.
7. Wrap every function in a ```python ... ``` code fence.
8. Provide a one-line docstring describing what the function detects.
9. KEEP FUNCTIONS SHORT — under 30 lines. Use simple, direct logic. \
Prefer 5-15 line functions. Avoid verbose patterns or unnecessary variables.
10. Be PRECISE — only return the target label when there is strong evidence. \
It is better to abstain (return None) than to over-fire. A function that \
fires on too many texts is useless.
"""

# ---------------------------------------------------------------------------
# Few-shot examples (critical for small models)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: str = """\

Here are example labeling functions showing the expected format:

```python
def lf_keyword_sports_01(text: str):
    \"\"\"Detects sports content via ball-game keywords.\"\"\"
    keywords = ["touchdown", "goalkeeper", "home run", "slam dunk"]
    lower = text.lower()
    if any(kw in lower for kw in keywords):
        return "Sports"
    return None
```

```python
import re
def lf_regex_email_01(text: str):
    \"\"\"Detects email addresses using a regex pattern.\"\"\"
    if re.search(r'[\\w.+-]+@[\\w-]+\\.[a-zA-Z]{2,}', text):
        return "Contact Info"
    return None
```

```python
def lf_compositional_urgent_01(text: str):
    \"\"\"Detects urgent messages by combining multiple signals.\"\"\"
    lower = text.lower()
    has_urgency = any(w in lower for w in ["asap", "urgent", "immediately"])
    has_action = any(w in lower for w in ["fix", "resolve", "respond"])
    if has_urgency and has_action:
        return "Urgent"
    return None
```

Now write NEW functions for the task below. Follow the exact same format.
"""

# ---------------------------------------------------------------------------
# Language-specific supplements
# ---------------------------------------------------------------------------

LANGUAGE_SUPPLEMENTS: dict[str, str] = {
    "en": "",
    "hi": """\
LANGUAGE NOTE — Hindi (Devanagari script):
- Use Unicode Devanagari range [\\u0900-\\u097F] in regex patterns.
- Common Hindi postpositions: का, की, के, में, से, को, पर, ने.
- For keyword matching use the actual Devanagari strings, e.g. \
`if "क्रिकेट" in text`.
- Hindi text may contain mixed Hindi-English (Hinglish); check both scripts.
- Use `text.strip()` and normalize whitespace before matching.
""",
    "mr": """\
LANGUAGE NOTE — Marathi (Devanagari script):
- Use Unicode Devanagari range [\\u0900-\\u097F] in regex patterns.
- Common Marathi postpositions: चा, ची, चे, ला, ने, त, मध्ये.
- For keyword matching use Devanagari strings, e.g. `if "क्रीडा" in text`.
- Marathi shares the Devanagari script with Hindi but has distinct vocabulary.
- Use `text.strip()` and normalize whitespace before matching.
""",
    "ta": """\
LANGUAGE NOTE — Tamil (Tamil script):
- Use Unicode Tamil range [\\u0B80-\\u0BFF] in regex patterns.
- Tamil is agglutinative; consider suffix patterns.
""",
    "bn": """\
LANGUAGE NOTE — Bengali (Bengali script):
- Use Unicode Bengali range [\\u0980-\\u09FF] in regex patterns.
- Consider conjuncts and vowel signs in pattern matching.
""",
}

# ---------------------------------------------------------------------------
# Strategy-specific templates
# ---------------------------------------------------------------------------

_BASE_TEMPLATE: str = """\
Task description: {task_description}
Target label: {target_label}
Full label space: {label_space}
{language_supplement}
Here are example texts that belong to the target label:
{examples}

{failure_section}

Existing labeling function descriptions (avoid duplicating these):
{existing_lfs}
{few_shot_section}
Write exactly {num_lfs} NEW Python labeling functions using the \
**{strategy}** strategy described below.

{strategy_instructions}

Return each function inside its own ```python``` code fence.
"""

# Strategy-specific instruction blocks
_KEYWORD_INSTRUCTIONS: str = """\
KEYWORD strategy: Write functions that check for the presence of specific \
keywords or phrases (case-insensitive) that are strong indicators of the \
target label. Use `str.lower()`, `in`, and similar string operations. \
Choose keywords that are distinctive and unlikely to appear in other classes."""

_REGEX_INSTRUCTIONS: str = """\
REGEX strategy: Write functions that use `re.search()` or `re.findall()` \
with regular expressions to match patterns indicative of the target label. \
Use word boundaries (`\\b`), alternation (`|`), and character classes where \
appropriate. Compile patterns inline."""

_FUZZY_INSTRUCTIONS: str = """\
FUZZY strategy: Write functions that perform approximate / fuzzy string \
matching. Check for common misspellings, typos, partial matches, or \
character-level variations of the target label or its key identifiers. \
Use techniques like substring checks with lowered text, character set \
overlap, or Levenshtein-style heuristics using only built-in operations."""

_SEMANTIC_INSTRUCTIONS: str = """\
SEMANTIC strategy: Write functions that detect semantic cues, contextual \
phrases, or domain-specific terminology strongly associated with the target \
label. Look for co-occurring terms, descriptive phrases, or topical \
language rather than exact keyword matches."""

_ABBREVIATION_INSTRUCTIONS: str = """\
ABBREVIATION strategy: Write functions that detect abbreviations, acronyms, \
ticker symbols, short forms, or code names commonly used to refer to the \
target label. Consider uppercase patterns, dotted abbreviations, and \
informal short-hands."""

_NEGATION_INSTRUCTIONS: str = """\
NEGATION strategy: Write functions that use negation or exclusion logic to \
identify the target label by ruling out other labels. Check for the ABSENCE \
of features that would indicate a different label, or detect explicit \
negation patterns (e.g. "not X", "unlike Y")."""

_CONTEXT_INSTRUCTIONS: str = """\
CONTEXT strategy: Write functions that examine the surrounding context or \
sentence structure to identify the target label. Look for typical sentence \
patterns, prepositional phrases, or syntactic structures that are \
characteristic of how the target label is discussed."""

_COMPOSITIONAL_INSTRUCTIONS: str = """\
COMPOSITIONAL strategy: Write functions that combine multiple weak signals \
to make a labeling decision. Each function should check for at least two \
independent indicators and return the target label only when enough \
evidence is present. Use boolean combinations (and/or) of simpler checks."""


STRATEGY_TEMPLATES: dict[str, str] = {
    "keyword": _BASE_TEMPLATE.replace("{strategy_instructions}", _KEYWORD_INSTRUCTIONS),
    "regex": _BASE_TEMPLATE.replace("{strategy_instructions}", _REGEX_INSTRUCTIONS),
    "fuzzy": _BASE_TEMPLATE.replace("{strategy_instructions}", _FUZZY_INSTRUCTIONS),
    "semantic": _BASE_TEMPLATE.replace("{strategy_instructions}", _SEMANTIC_INSTRUCTIONS),
    "abbreviation": _BASE_TEMPLATE.replace("{strategy_instructions}", _ABBREVIATION_INSTRUCTIONS),
    "negation": _BASE_TEMPLATE.replace("{strategy_instructions}", _NEGATION_INSTRUCTIONS),
    "context": _BASE_TEMPLATE.replace("{strategy_instructions}", _CONTEXT_INSTRUCTIONS),
    "compositional": _BASE_TEMPLATE.replace("{strategy_instructions}", _COMPOSITIONAL_INSTRUCTIONS),
}

# ---------------------------------------------------------------------------
# Refinement template for agentic self-debugging (Feature 2)
# ---------------------------------------------------------------------------

REFINEMENT_TEMPLATE: str = """\
You previously wrote this labeling function:

```python
{prior_source}
```

It was tested and had these problems:
{failure_report}

Task description: {task_description}
Target label: {target_label}
Full label space: {label_space}

Here are example texts that belong to the target label:
{examples}

Please rewrite the function to fix these issues. Be MORE PRECISE:
- If the function is overly broad, add more specific conditions
- If it's firing on the wrong texts, adjust the pattern matching
- If it's too narrow, widen the criteria slightly while keeping precision

Return the improved function in a ```python``` code fence.
Keep the same function name and strategy, but fix the logic.
"""
