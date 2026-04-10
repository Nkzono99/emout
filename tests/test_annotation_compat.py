"""Static checks that keep modern type annotations import-safe on Python 3.8."""

from __future__ import annotations

import ast
from pathlib import Path


def _uses_modern_annotation_syntax(tree: ast.AST) -> bool:
    """Return True when annotations use ``|`` or builtin generics like ``list[str]``."""
    for node in ast.walk(tree):
        annotation = None
        if isinstance(node, (ast.AnnAssign, ast.arg)):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        if annotation is None:
            continue

        text = ast.unparse(annotation)
        if "|" in text or any(token in text for token in ("list[", "dict[", "set[", "tuple[")):
            return True
    return False


def test_files_with_modern_annotations_enable_future_annotations():
    """Files using modern annotation syntax must defer evaluation for Python 3.8 support."""
    root = Path(__file__).resolve().parents[1] / "emout"
    offending = []

    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        if _uses_modern_annotation_syntax(tree) and "from __future__ import annotations" not in source:
            offending.append(path.relative_to(root.parent).as_posix())

    assert offending == []
