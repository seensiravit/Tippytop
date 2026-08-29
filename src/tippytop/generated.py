from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any


MAX_SOURCE_LENGTH = 100_000

_DATA_SCIENCE_IMPORTS = {
    "joblib",
    "lightgbm",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "threadpoolctl",
    "typing_extensions",
}
_BLOCKED_STANDARD_IMPORTS = {
    "asyncio",
    "builtins",
    "code",
    "codeop",
    "concurrent",
    "ctypes",
    "importlib",
    "inspect",
    "multiprocessing",
    "os",
    "pathlib",
    "pkgutil",
    "pty",
    "resource",
    "subprocess",
    "sys",
}
_ALLOWED_IMPORTS = (set(sys.stdlib_module_names) - _BLOCKED_STANDARD_IMPORTS) | _DATA_SCIENCE_IMPORTS
_ALLOWED_TIPPYTOP_IMPORTS = {"tippytop.models", "tippytop.research"}
_DANGEROUS_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


@dataclass(frozen=True)
class GeneratedExperiment:
    hypothesis: str
    expected_effect: str
    source: str

    def __post_init__(self) -> None:
        for field_name in ("hypothesis", "expected_effect", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        object.__setattr__(self, "hypothesis", self.hypothesis.strip())
        object.__setattr__(self, "expected_effect", self.expected_effect.strip())
        _validate_source(self.source)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GeneratedExperiment:
        if not isinstance(value, dict):
            raise ValueError("generated experiment must be an object")

        expected = {"hypothesis", "expected_effect", "source"}
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing:
            raise ValueError(f"generated experiment is missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"generated experiment has unknown fields: {sorted(unknown)}")
        for field_name in expected:
            if not isinstance(value[field_name], str) or not value[field_name].strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        return cls(
            hypothesis=value["hypothesis"],
            expected_effect=value["expected_effect"],
            source=value["source"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "hypothesis": self.hypothesis,
            "expected_effect": self.expected_effect,
            "source": self.source,
        }

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract exactly one JSON object, tolerating a Markdown code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            raise ValueError("LLM response must contain a JSON object")
        if stripped[index + end :].strip():
            raise ValueError("LLM response contains text after the JSON object")
        return value
    raise ValueError("LLM response does not contain valid JSON")


def executable_fingerprint(source: str) -> str:
    """Hash executable syntax while ignoring formatting and comments."""

    tree = ast.parse(source)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_source(source: str) -> None:
    if len(source) > MAX_SOURCE_LENGTH:
        raise ValueError(f"source exceeds the {MAX_SOURCE_LENGTH}-character limit")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"source is not valid Python: {exc}") from exc

    _SourceValidator().visit(tree)
    _validate_entry_point(tree, "fit", ("train_rows", "seed"))
    _validate_entry_point(tree, "predict", ("model", "rows"))


def _validate_entry_point(
    tree: ast.Module, name: str, expected_parameters: tuple[str, str]
) -> None:
    functions = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == name
    ]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise ValueError(f"source must define exactly one top-level {name}{expected_parameters}")

    arguments = functions[0].args
    positional = (*arguments.posonlyargs, *arguments.args)
    if (
        len(positional) != len(expected_parameters)
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kwarg is not None
    ):
        rendered = ", ".join(expected_parameters)
        raise ValueError(f"{name} must have exactly the positional parameters ({rendered})")


def _is_allowed_import(module: str) -> bool:
    return any(
        module == allowed or module.startswith(f"{allowed}.")
        for allowed in _ALLOWED_IMPORTS | _ALLOWED_TIPPYTOP_IMPORTS
    )


def _has_dunder_identifier(name: str) -> bool:
    return any(part.startswith("__") for part in name.split("."))


class _SourceValidator(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            self._validate_import(imported.name)
            self._validate_identifier(imported.asname, "import alias")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module is None:
            raise ValueError("relative imports are not allowed")
        self._validate_import(node.module)
        for imported in node.names:
            if imported.name == "*":
                raise ValueError("wildcard imports are not allowed")
            self._validate_identifier(imported.name, "imported name")
            self._validate_identifier(imported.asname, "import alias")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        called_name: str | None = None
        if isinstance(node.func, ast.Name):
            called_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called_name = node.func.attr
        if called_name in _DANGEROUS_CALLS:
            raise ValueError(f"call to {called_name} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _DANGEROUS_CALLS:
            raise ValueError(f"reference to {node.id} is not allowed")
        self._validate_identifier(node.id, "identifier")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._validate_identifier(node.attr, "attribute")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._validate_identifier(node.arg, "parameter")
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        self._validate_identifier(node.arg, "keyword")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._validate_identifier(name, "global")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._validate_identifier(name, "nonlocal")

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._validate_identifier(node.name, "exception target")
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self._validate_identifier(node.name, "pattern target")
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._validate_identifier(node.name, "pattern target")

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self._validate_identifier(node.rest, "pattern target")
        self.generic_visit(node)

    @staticmethod
    def _validate_import(module: str) -> None:
        if _has_dunder_identifier(module) or not _is_allowed_import(module):
            raise ValueError(f"import from {module!r} is not allowed")

    @staticmethod
    def _validate_identifier(name: str | None, kind: str) -> None:
        if name is not None and _has_dunder_identifier(name):
            raise ValueError(f"{kind} {name!r} may not begin with '__'")
