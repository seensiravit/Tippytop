"""Static contract checks for generated train/prediction feature paths."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..research.data import AUXILIARY_COLUMNS


_TRAINING_ONLY_COLUMNS = {"long_view", *AUXILIARY_COLUMNS}


@dataclass(frozen=True)
class _Function:
    owner: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def qualified_name(self) -> str:
        return f"{self.owner}.{self.name}" if self.owner else self.name


def validate_prediction_paths(source: str) -> None:
    """Reject prediction paths that attempt to read training-only outcomes."""

    tree = ast.parse(source)
    functions = _functions(tree)
    reachable = _prediction_reachable(functions)
    failures: list[str] = []
    for function in sorted(reachable, key=lambda item: item.node.lineno):
        referenced = sorted(
            {
                node.value
                for node in ast.walk(function.node)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in _TRAINING_ONLY_COLUMNS
            }
        )
        if referenced:
            failures.append(
                f'generated experiment.py", line {function.node.lineno}: '
                f"prediction-reachable function {function.qualified_name} references "
                f"training-only columns {referenced}"
            )
    if failures:
        raise ValueError(
            "prediction-path validation failed: "
            + "; ".join(failures)
            + ". Fit outcome-derived aggregates on training rows, persist maps/statistics in the "
            "returned model state, and use a separate prediction transform that reads only "
            "prediction-time columns and stored state."
        )


def _functions(tree: ast.Module) -> list[_Function]:
    functions: list[_Function] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_Function(None, node))
        elif isinstance(node, ast.ClassDef):
            functions.extend(
                _Function(node.name, child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return functions


def _prediction_reachable(functions: list[_Function]) -> set[_Function]:
    by_key = {(function.owner, function.name): function for function in functions}
    reachable = {function for function in functions if function.name == "predict"}
    pending = list(reachable)
    while pending:
        function = pending.pop()
        instance_types = _instance_types(function.node)
        for call in (node for node in ast.walk(function.node) if isinstance(node, ast.Call)):
            target: _Function | None = None
            if isinstance(call.func, ast.Name):
                target = by_key.get((None, call.func.id))
            elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                receiver = call.func.value.id
                if receiver == "self" and function.owner:
                    target = by_key.get((function.owner, call.func.attr))
                elif receiver in instance_types:
                    target = by_key.get((instance_types[receiver], call.func.attr))
            if target is not None and target not in reachable:
                reachable.add(target)
                pending.append(target)
    return reachable


def _instance_types(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    instances: dict[str, str] = {}
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Assign)
            and len(child.targets) == 1
            and isinstance(child.targets[0], ast.Name)
            and isinstance(child.value, ast.Call)
            and isinstance(child.value.func, ast.Name)
        ):
            instances[child.targets[0].id] = child.value.func.id
    return instances
