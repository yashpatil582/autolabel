"""Tests for multilingual support."""

from autolabel.lf.generator import LFGenerator
from autolabel.lf.sandbox import SandboxedExecutor
from autolabel.lf.templates import LANGUAGE_SUPPLEMENTS
from autolabel.text.normalize import contains_devanagari, normalize_text
from tests.conftest import MockLLMProvider


class TestUnicodeNormalization:
    def test_normalize_nfkc(self):
        # Devanagari text with potential combining characters
        text = "नमस्ते दुनिया"
        result = normalize_text(text, form="NFKC")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_preserves_ascii(self):
        text = "hello world"
        assert normalize_text(text) == text

    def test_contains_devanagari_true(self):
        assert contains_devanagari("यह हिंदी है")

    def test_contains_devanagari_false(self):
        assert not contains_devanagari("this is english")

    def test_contains_devanagari_mixed(self):
        assert contains_devanagari("Hello नमस्ते World")


class TestLanguageSupplements:
    def test_hindi_supplement_exists(self):
        assert "hi" in LANGUAGE_SUPPLEMENTS
        supplement = LANGUAGE_SUPPLEMENTS["hi"]
        assert "Devanagari" in supplement
        assert "\\u0900" in supplement

    def test_marathi_supplement_exists(self):
        assert "mr" in LANGUAGE_SUPPLEMENTS
        supplement = LANGUAGE_SUPPLEMENTS["mr"]
        assert "Devanagari" in supplement
        assert "Marathi" in supplement

    def test_english_supplement_empty(self):
        assert LANGUAGE_SUPPLEMENTS["en"] == ""


class TestLanguageInGenerator:
    def test_hindi_prompt_includes_supplement(self):
        provider = MockLLMProvider(responses=["no code"])
        gen = LFGenerator(
            provider=provider,
            label_space=["खेल", "राजनीति"],
            task_description="Classify Hindi news",
            language="hi",
        )
        prompt = gen._build_prompt(
            strategy="keyword",
            target_label="खेल",
            examples=["क्रिकेट मैच में भारत जीता"],
            existing_lf_descriptions=[],
            failure_examples=None,
            num_lfs=2,
        )
        assert "Devanagari" in prompt
        assert "\\u0900" in prompt

    def test_english_prompt_no_supplement(self):
        provider = MockLLMProvider(responses=["no code"])
        gen = LFGenerator(
            provider=provider,
            label_space=["pos", "neg"],
            task_description="Classify sentiment",
            language="en",
        )
        prompt = gen._build_prompt(
            strategy="keyword",
            target_label="pos",
            examples=["great movie"],
            existing_lf_descriptions=[],
            failure_examples=None,
            num_lfs=2,
        )
        # Should not contain Devanagari note
        assert "Devanagari" not in prompt


class TestSandboxDevanagari:
    def test_validates_lf_with_devanagari_strings(self):
        src = '''def lf_keyword_sports_01(text: str):
    """Detects sports via Hindi keyword."""
    if "क्रिकेट" in text:
        return "खेल"
    return None
'''
        ok, reason = SandboxedExecutor.validate_source(src)
        assert ok, reason

    def test_validates_lf_with_devanagari_regex(self):
        src = r'''import re
def lf_regex_devanagari_01(text: str):
    """Detects Devanagari text."""
    if re.search(r'[\u0900-\u097F]+', text):
        return "Hindi"
    return None
'''
        ok, reason = SandboxedExecutor.validate_source(src)
        assert ok, reason


class TestAutoFixImports:
    def test_auto_fix_adds_import_re(self):
        source = '''def lf_regex_test_01(text: str):
    if re.search(r"hello", text):
        return "Test"
    return None
'''
        fixed = LFGenerator._auto_fix_imports(source)
        assert fixed.startswith("import re\n")

    def test_auto_fix_no_change_when_present(self):
        source = '''import re
def lf_regex_test_01(text: str):
    if re.search(r"hello", text):
        return "Test"
    return None
'''
        fixed = LFGenerator._auto_fix_imports(source)
        assert fixed == source
