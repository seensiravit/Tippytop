"""Best-effort static guard on generated code.

NOT a real sandbox — it blocks the *obvious* ways a script could read the test
split or reach the network. The real hidden-test wall is that the harness never
scores test until finalize (see orchestrator). Returns a list of violation
strings; empty means clean.
"""
from __future__ import annotations
import ast
import re

_BANNED_IMPORTS = {
    "socket", "urllib", "requests", "http", "httplib", "ftplib", "smtplib",
    "subprocess", "ctypes", "telnetlib", "asyncio", "aiohttp",
}
_BANNED_CALLS = {"eval", "exec", "__import__", "compile"}
# hardcoded test-split access, e.g. splits['test'], .splits["test"], split='test'
_TEST_RE = re.compile(r"""(?ix)
    (splits|enc)\s*\[\s*['"]test['"]\s*\]      # splits['test']
  | \.\s*(X|y|users|rows)\s*\(\s*['"]test['"]  # .X('test')
  | \bsplit\s*=\s*['"]test['"]                 # split='test'
""")


def scan(code: str) -> list[str]:
    violations: list[str] = []

    for m in _TEST_RE.finditer(code or ""):
        violations.append(f"hardcoded test-split access: {m.group(0)!r}")

    try:
        tree = ast.parse(code or "")
    except SyntaxError as e:
        return violations + [f"syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BANNED_IMPORTS:
                    violations.append(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _BANNED_IMPORTS:
                violations.append(f"banned import from: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                violations.append(f"banned call: {node.func.id}()")
        elif isinstance(node, ast.Attribute):
            # os.system / os.popen
            if node.attr in ("system", "popen") and isinstance(node.value, ast.Name) \
                    and node.value.id == "os":
                violations.append(f"banned call: os.{node.attr}()")

    # de-dup, keep order
    seen, out = set(), []
    for v in violations:
        if v not in seen:
            seen.add(v); out.append(v)
    return out
