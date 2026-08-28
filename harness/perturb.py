"""Perturbation generators.

Every variant comes from one normalized baseline, so a judge comparing baseline
against any condition sees exactly one intended difference and no incidental
formatting noise:

    baseline = ast.unparse(ast.parse(original))

Most transforms work on the AST. reformat cannot, since ast.unparse normalizes
whitespace away, so it edits source text and uses tokenize to skip lines spanned
by multi-line strings; docstring contents are never re-indented.

Docstrings survive the AST round-trip because they are ordinary string
expressions. `#` comments do not, which is why every documentation condition
here is carried by the docstring.

Nothing in this file is trusted: oracle.py re-checks every variant and discards
preserving transforms that change behaviour and mutations that do not.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class Variant:
    condition: str
    code: str
    meta: dict = field(default_factory=dict)


def normalize(src: str) -> str:
    """The common ancestor of every variant."""
    return ast.unparse(ast.parse(src))


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise KeyError(f"entry point {name!r} not found")


def get_docstring(src: str, entry_point: str) -> str | None:
    return ast.get_docstring(find_function(ast.parse(src), entry_point), clean=False)


def set_docstring(src: str, entry_point: str, new_doc: str | None) -> str:
    """Replace (or with None, remove) the entry point's docstring."""
    tree = ast.parse(src)
    fn = find_function(tree, entry_point)
    has_doc = (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    )
    if has_doc:
        fn.body = fn.body[1:]
    if new_doc is not None:
        fn.body.insert(0, ast.Expr(value=ast.Constant(value=new_doc)))
    if not fn.body:
        fn.body = [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


class _BoundNames(ast.NodeVisitor):
    """Names bound inside the module: parameters, assignment targets, loop and
    comprehension variables, nested function names.

    Names appearing only in Load context (builtins, imported modules, the entry
    point) are never collected, so renaming cannot break a call to len."""

    def __init__(self, protect: set[str]):
        self.bound: set[str] = set()
        self.protect = protect

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id not in self.protect:
            self.bound.add(node.id)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg):
        if node.arg not in self.protect:
            self.bound.add(node.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name not in self.protect:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name and node.name not in self.protect:
            self.bound.add(node.name)
        self.generic_visit(node)


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.m = mapping

    def visit_Name(self, node: ast.Name):
        node.id = self.m.get(node.id, node.id)
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = self.m.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self.m.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            node.name = self.m.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        node.names = [self.m.get(n, n) for n in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal):
        node.names = [self.m.get(n, n) for n in node.names]
        return node


def rename(src: str, entry_point: str) -> str:
    """Replace every meaningful identifier with an opaque one. Semantics are
    untouched; every naming cue a judge might read is destroyed."""
    tree = ast.parse(src)
    # Imported names bind through alias rather than a Name node, so they are
    # already safe; protect them explicitly anyway.
    imported = {
        (a.asname or a.name.split(".")[0])
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    }
    # self and cls are kept. Renaming them is legal but introduces a PEP 8
    # violation, and a judge penalising that would be reacting to a real style
    # defect rather than to the loss of naming information.
    collector = _BoundNames(protect={entry_point, "self", "cls"} | imported)
    collector.visit(tree)
    mapping = {name: f"v{i}" for i, name in enumerate(sorted(collector.bound))}
    if not mapping:
        raise ValueError("nothing to rename")
    return ast.unparse(_Renamer(mapping).visit(tree))


def strip_docstring(src: str, entry_point: str) -> str:
    if get_docstring(src, entry_point) is None:
        raise ValueError("no docstring to strip")
    return set_docstring(src, entry_point, None)


class _ConstantHoister(ast.NodeTransformer):
    """Collects int/float/str literals and replaces them with references.

    Skips f-strings (a Constant inside a JoinedStr is not an expression you can
    substitute a Name for), annotations, and default arguments (evaluated at
    def time, before the hoisted assignments would run)."""

    def __init__(self):
        self.consts: list[ast.Constant] = []

    def visit_JoinedStr(self, node):
        return node                      # do not descend

    def visit_arguments(self, node):
        return node                      # defaults + annotations: do not descend

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool) or node.value is None:
            return node
        if not isinstance(node.value, (int, float, str)):
            return node
        self.consts.append(node)
        return ast.Name(id=f"_k{len(self.consts) - 1}", ctx=ast.Load())


def extract_constants(src: str, entry_point: str) -> str:
    """Hoist literals into named locals at the top of the function. The
    computation is identical; the code looks more 'engineered'."""
    tree = ast.parse(src)
    fn = find_function(tree, entry_point)

    start = 0
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        start = 1                        # keep the docstring where it is

    hoister = _ConstantHoister()
    body = [hoister.visit(stmt) for stmt in fn.body[start:]]
    if not hoister.consts:
        raise ValueError("no hoistable literals")

    assigns = [
        ast.Assign(targets=[ast.Name(id=f"_k{i}", ctx=ast.Store())], value=c)
        for i, c in enumerate(hoister.consts)
    ]
    fn.body = fn.body[:start] + assigns + body
    return ast.unparse(ast.fix_missing_locations(tree))


def _string_spanned_lines(src: str) -> set[int]:
    """1-indexed lines covered by a multi-line string token. Their contents are
    data, not code, so they must not be re-indented."""
    protected: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                protected.update(range(tok.start[0] + 1, tok.end[0] + 1))
    except tokenize.TokenError:
        pass
    return protected


def reformat(src: str, entry_point: str) -> str:
    """Change indentation width and add vertical whitespace. Purely visual."""
    protected = _string_spanned_lines(src)
    out: list[str] = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        if lineno in protected or not line.strip():
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        reindented = " " * ((indent // 4) * 2) + line.lstrip(" ")
        # A blank line inside a block is legal anywhere except directly after a
        # `:` header line, which we detect by looking at what we just emitted.
        prev = out[-1].rstrip() if out else ""
        if out and prev and not prev.endswith(":") and reindented.strip().startswith(("return", "for ", "if ", "while ")):
            out.append("")
        out.append(reindented)
    result = "\n".join(out)
    ast.parse(result)                    # fail loudly rather than emit garbage
    return result


PRESERVING_FNS: dict[str, Callable[[str, str], str]] = {
    "rename": rename,
    "strip_docstring": strip_docstring,
    "reformat": reformat,
    "extract_constants": extract_constants,
}


def _sites(tree: ast.Module, pred) -> list[ast.AST]:
    return [n for n in ast.walk(tree) if pred(n)]


def _mutate_family(src: str, pred, apply_fn, limit: int) -> list[tuple[str, str]]:
    """Yield one mutant per eligible site, up to `limit`.

    Each mutant is built from a freshly parsed tree so mutations never compound,
    and ast.walk order is deterministic so site k is stable across runs. The
    caller oracle-checks these and keeps the first that actually breaks, which
    is how equivalent mutants are filtered out."""
    out: list[tuple[str, str]] = []
    n_sites = len(_sites(ast.parse(src), pred))
    for k in range(min(n_sites, limit)):
        tree = ast.parse(src)
        node = _sites(tree, pred)[k]
        desc = apply_fn(node)
        if desc is None:
            continue
        out.append((desc, ast.unparse(ast.fix_missing_locations(tree))))
    return out


_CMP_FLIP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}


def mut_compare(src: str, limit: int) -> list[tuple[str, str]]:
    """Boundary bugs: `<` becomes `<=`, `==` becomes `!=`."""
    def pred(n):
        return isinstance(n, ast.Compare) and any(type(o) in _CMP_FLIP for o in n.ops)

    def apply(node: ast.Compare):
        for i, op in enumerate(node.ops):
            if type(op) in _CMP_FLIP:
                new = _CMP_FLIP[type(op)]
                node.ops[i] = new()
                return f"{type(op).__name__}->{new.__name__}"
        return None

    return _mutate_family(src, pred, apply, limit)


def mut_offbyone(src: str, limit: int) -> list[tuple[str, str]]:
    """Off-by-one: shift an integer literal."""
    def pred(n):
        return isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool)

    results: list[tuple[str, str]] = []
    for delta in (1, -1):
        def apply(node: ast.Constant, d=delta):
            node.value = node.value + d
            return f"int{d:+d}"
        results.extend(_mutate_family(src, pred, apply, limit))
    return results


def mut_swap(src: str, limit: int) -> list[tuple[str, str]]:
    """Argument/operand order: only non-commutative operators and calls."""
    non_commutative = (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift)

    def pred(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, non_commutative):
            return True
        return isinstance(n, ast.Call) and len(n.args) >= 2

    def apply(node):
        if isinstance(node, ast.BinOp):
            node.left, node.right = node.right, node.left
            return f"binop-{type(node.op).__name__}"
        node.args[0], node.args[1] = node.args[1], node.args[0]
        return "call-args"

    return _mutate_family(src, pred, apply, limit)


BREAKING_FNS: dict[str, Callable[[str, int], list[tuple[str, str]]]] = {
    "mut_compare": mut_compare,
    "mut_offbyone": mut_offbyone,
    "mut_swap": mut_swap,
}
