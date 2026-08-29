"""Static checks against installed estimator APIs for generated repair patches."""

from __future__ import annotations

import ast
import importlib
import inspect
from typing import TypeVar


_VALIDATED_LIBRARY_ROOTS = {"lightgbm", "sklearn"}
T = TypeVar("T")


def validate_installed_api_calls(source: str) -> None:
    """Reject statically resolvable method keywords unsupported by this environment."""

    tree = ast.parse(source)
    imports = _imports(tree)
    index = _ScopedIndex(imports)
    index.visit(tree)

    failures: list[str] = []
    for lexical_scope, receiver_scope, node in index.calls:
        receiver = ast.unparse(node.func.value)
        key = (receiver_scope, receiver)
        qualified_class = _latest(index.constructors.get(key, []), node.lineno)
        if qualified_class is None:
            continue
        method_name = node.func.attr
        method = _resolve_attribute(qualified_class, method_name)
        if method is None:
            continue
        signature = inspect.signature(method)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            continue
        supported = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind is not inspect.Parameter.POSITIONAL_ONLY and name != "self"
        }
        supplied = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
        for keyword in node.keywords:
            if keyword.arg is None and isinstance(keyword.value, ast.Name):
                dictionary_key = (lexical_scope, keyword.value.id)
                supplied.update(_latest(index.dictionaries.get(dictionary_key, []), node.lineno) or set())
        unsupported = sorted(keyword for keyword in supplied if keyword not in supported)
        if unsupported:
            failures.append(
                f'generated experiment.py", line {node.lineno}: '
                f"{qualified_class}.{method_name} has unsupported keywords {unsupported}; "
                f"installed signature is {method_name}{signature}"
            )
        if (
            qualified_class.endswith(".LGBMRanker")
            and method_name == "fit"
            and len(node.args) >= 2
            and _latest(
                index.constant_one_arrays.get(
                    (
                        receiver_scope
                        if ast.unparse(node.args[1]).startswith("self.")
                        else lexical_scope,
                        ast.unparse(node.args[1]),
                    ),
                    [],
                ),
                node.lineno,
            )
            is True
        ):
            failures.append(
                f'generated experiment.py", line {node.lineno}: {qualified_class}.fit receives '
                "labels created only with numpy.ones; a ranker needs "
                "non-constant relevance labels within each query group and feature rows for every "
                "label represented by group"
            )

    if failures:
        raise ValueError("installed API validation failed: " + "; ".join(failures))


Scope = tuple[str, int]
BindingKey = tuple[Scope, str]


class _ScopedIndex(ast.NodeVisitor):
    """Index bindings by lexical scope and source order so stale names cannot leak."""

    def __init__(self, imports: dict[str, str]) -> None:
        self.imports = imports
        self.class_stack: list[int] = []
        self.function_stack: list[int] = []
        self.constructors: dict[BindingKey, list[tuple[int, str | None]]] = {}
        self.dictionaries: dict[BindingKey, list[tuple[int, set[str] | None]]] = {}
        self.constant_one_arrays: dict[BindingKey, list[tuple[int, bool]]] = {}
        self.calls: list[tuple[Scope, Scope, ast.Call]] = []

    def current_scope(self) -> Scope:
        if self.function_stack:
            return ("function", self.function_stack[-1])
        if self.class_stack:
            return ("class", self.class_stack[-1])
        return ("module", 0)

    def binding_scope(self, target: str, scope: Scope | None = None) -> Scope:
        if target.startswith("self.") and self.class_stack:
            return ("class", self.class_stack[-1])
        return scope or self.current_scope()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.lineno)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.lineno)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1:
            self._record_assignment(node.targets[0], node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_assignment(node.target, node.value, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            receiver = ast.unparse(node.func.value)
            self.calls.append((self.current_scope(), self.binding_scope(receiver), node))
        self.generic_visit(node)

    def _record_assignment(self, target_node: ast.expr, value: ast.expr, lineno: int) -> None:
        target = ast.unparse(target_node)
        key = (self.binding_scope(target), target)
        qualified = _qualified_name(value.func, self.imports) if isinstance(value, ast.Call) else None
        constructor = (
            qualified
            if qualified and qualified.split(".", 1)[0] in _VALIDATED_LIBRARY_ROOTS
            else None
        )
        self.constructors.setdefault(key, []).append((lineno, constructor))

        dictionary: set[str] | None = None
        if isinstance(value, ast.Dict):
            keys = {
                item.value
                for item in value.keys
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            if len(keys) == len(value.keys):
                dictionary = keys
        self.dictionaries.setdefault(key, []).append((lineno, dictionary))
        self.constant_one_arrays.setdefault(key, []).append((lineno, qualified == "numpy.ones"))


def _latest(bindings: list[tuple[int, T]], lineno: int) -> T | None:
    eligible = [binding for binding in bindings if binding[0] <= lineno]
    return max(eligible, key=lambda binding: binding[0])[1] if eligible else None


def _imports(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                aliases[imported.asname or imported.name.split(".", 1)[0]] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"
    return aliases


def _qualified_name(node: ast.expr, imports: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return imports.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value, imports)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _resolve_attribute(qualified_class: str, method_name: str) -> object | None:
    parts = qualified_class.split(".")
    try:
        value: object = importlib.import_module(parts[0])
        for part in parts[1:]:
            value = getattr(value, part)
        return getattr(value, method_name)
    except (AttributeError, ImportError):
        return None
