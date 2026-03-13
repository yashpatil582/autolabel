"""Tests for the sandboxed executor."""

from autolabel.lf.sandbox import SandboxedExecutor
from autolabel.lf.base import LabelingFunction, ABSTAIN


class TestValidateSource:
    def test_valid_keyword_function(self):
        src = """def lf_keyword_test(text: str):
    if "delta" in text.lower():
        return "Delta"
    return None
"""
        ok, reason = SandboxedExecutor.validate_source(src)
        assert ok, reason

    def test_valid_regex_function(self):
        src = """import re
def lf_regex_test(text: str):
    if re.search(r"\\bdelta\\b", text, re.IGNORECASE):
        return "Delta"
    return None
"""
        ok, reason = SandboxedExecutor.validate_source(src)
        assert ok, reason

    def test_blocked_import_os(self):
        src = """import os
def lf_bad(text: str):
    return os.environ.get("SECRET")
"""
        ok, reason = SandboxedExecutor.validate_source(src)
        assert not ok
        assert "import" in reason.lower() or "blocked" in reason.lower()

    def test_blocked_import_subprocess(self):
        src = """import subprocess
def lf_bad(text: str):
    return subprocess.check_output(["cat", "/etc/passwd"])
"""
        ok, reason = SandboxedExecutor.validate_source(src)
        assert not ok

    def test_blocked_eval(self):
        src = """def lf_bad(text: str):
    return eval("1+1")
"""
        ok, reason = SandboxedExecutor.validate_source(src)
        assert not ok
        assert "eval" in reason.lower()

    def test_blocked_exec(self):
        src = """def lf_bad(text: str):
    exec("import os")
    return None
"""
        ok, reason = SandboxedExecutor.validate_source(src)
        assert not ok

    def test_blocked_open(self):
        src = """def lf_bad(text: str):
    f = open("/etc/passwd")
    return f.read()
"""
        ok, reason = SandboxedExecutor.validate_source(src)
        assert not ok

    def test_max_lines_exceeded(self):
        lines = ["def lf_long(text: str):"] + ["    x = 1"] * 60
        src = "\n".join(lines)
        ok, reason = SandboxedExecutor.validate_source(src, max_lines=50)
        assert not ok
        assert "lines" in reason.lower()

    def test_syntax_error(self):
        src = "def lf_bad(text str):\n    return None"
        ok, reason = SandboxedExecutor.validate_source(src)
        assert not ok
        assert "syntax" in reason.lower()


class TestExecuteSafe:
    def test_normal_execution(self, sample_lf):
        result = SandboxedExecutor.execute_safe(sample_lf, "I flew Delta today")
        assert result == "Delta Air Lines"

    def test_abstain(self, sample_lf):
        result = SandboxedExecutor.execute_safe(sample_lf, "I flew United today")
        assert result == ABSTAIN

    def test_error_returns_abstain(self):
        lf = LabelingFunction(
            name="lf_error",
            source='def lf_error(text: str):\n    raise ValueError("boom")\n',
            strategy="keyword",
            description="Always errors",
            target_label="Test",
            iteration=0,
        )
        result = SandboxedExecutor.execute_safe(lf, "test")
        assert result == ABSTAIN
