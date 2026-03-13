"""Sandboxed execution environment for labeling functions.

Provides AST-level validation and signal-based timeout execution so
that LLM-generated code cannot perform dangerous operations.
"""

from __future__ import annotations

import ast
import signal
from typing import Any

from autolabel.lf.base import ABSTAIN, LabelingFunction

# ---------------------------------------------------------------------------
# AST node whitelist
# ---------------------------------------------------------------------------

_ALLOWED_NODES: set[type] = {
    # Structural
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Expr,
    ast.Pass,
    ast.Break,
    ast.Continue,
    # Control flow
    ast.If,
    ast.For,
    ast.While,
    # Expressions
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Name,
    ast.Constant,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.IfExp,
    ast.Starred,
    ast.keyword,
    # Assignments
    ast.Assign,
    ast.AugAssign,
    # Collections / comprehensions
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.comprehension,
    # Slice helpers
    ast.Slice,
    # Context nodes
    ast.Store,
    ast.Load,
    ast.Del,
    ast.alias,
    # Operators
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.FloorDiv,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    # Unary operators
    ast.USub,
    ast.UAdd,
    ast.Invert,
    # Bitwise operators (used in set operations)
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    # Other common nodes
    ast.Pow,
    ast.Delete,
    ast.Assert,
    ast.NamedExpr,
    # Try/except (LLMs frequently generate these)
    ast.Try,
    ast.ExceptHandler,
    # With statement
    ast.With,
    ast.withitem,
    # Annotated assignment (e.g. x: int = 5)
    ast.AnnAssign,
    # MatMult operator
    ast.MatMult,
}

# Deprecated nodes still emitted by older Python versions
for _name in ("Index", "Str", "Num", "TryStar"):
    _node = getattr(ast, _name, None)
    if _node is not None:
        _ALLOWED_NODES.add(_node)

# ---------------------------------------------------------------------------
# Blocked callable names
# ---------------------------------------------------------------------------

_BLOCKED_CALLS: set[str] = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "input",
    "print",  # not dangerous but unnecessary
}

# Blocked module-level attribute access
_BLOCKED_MODULES: set[str] = {"os", "sys", "subprocess", "shutil", "pathlib"}


class SandboxedExecutor:
    """Validates and safely executes labeling function source code."""

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_source(
        source: str, max_lines: int = 50
    ) -> tuple[bool, str]:
        """Validate *source* against the AST whitelist.

        Returns:
            A ``(valid, reason)`` tuple.  ``valid`` is ``True`` when the
            source passes all checks; otherwise ``reason`` describes the
            first violation encountered.
        """
        # Line count check
        lines = source.strip().splitlines()
        if len(lines) > max_lines:
            return False, f"Source exceeds {max_lines} lines ({len(lines)})"

        # Parse
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        # Walk AST
        for node in ast.walk(tree):
            node_type = type(node)

            # --- Import handling ---
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name != "re":
                        return False, f"Blocked import: {alias.name}"
                continue

            if isinstance(node, ast.ImportFrom):
                if node.module is None or not node.module.startswith("re"):
                    return (
                        False,
                        f"Blocked from-import: {node.module}",
                    )
                continue

            # --- General whitelist ---
            if node_type not in _ALLOWED_NODES:
                return (
                    False,
                    f"Disallowed AST node: {node_type.__name__}",
                )

            # --- Dangerous calls ---
            if isinstance(node, ast.Call):
                func = node.func
                # Direct call: eval(...), exec(...)
                if isinstance(func, ast.Name):
                    if func.id in _BLOCKED_CALLS:
                        return False, f"Blocked call: {func.id}"
                # Attribute call: os.system(...)
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        if func.value.id in _BLOCKED_MODULES:
                            return (
                                False,
                                f"Blocked module access: {func.value.id}",
                            )
                # type() with 3 args (dynamic class creation)
                if isinstance(func, ast.Name) and func.id == "type":
                    if len(node.args) == 3:
                        return False, "Blocked: type() with 3 arguments"

            # --- Bare name references to blocked modules ---
            if isinstance(node, ast.Name) and node.id in _BLOCKED_MODULES:
                # Allow only load-context references that are part of an
                # attribute chain (caught above); standalone is suspicious.
                pass  # attribute-access blocked above is sufficient

        return True, "OK"

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    @staticmethod
    def execute_safe(
        lf: LabelingFunction, text: str, timeout: int = 10
    ) -> str:
        """Execute *lf* on *text* with a signal-based timeout.

        Returns:
            The label string produced by the LF, or :data:`ABSTAIN` on
            any error, timeout, or exception.
        """

        def _timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError("LF execution timed out")

        # Compile if needed
        try:
            if lf._compiled_fn is None:
                lf.compile()
        except Exception:
            return ABSTAIN

        # Set alarm (Unix only)
        use_alarm = hasattr(signal, "SIGALRM")
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        try:
            result = lf.apply(text)
            return result
        except Exception:
            return ABSTAIN
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
