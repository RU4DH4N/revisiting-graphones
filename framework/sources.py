import ast
from functools import cache
from pathlib import Path

from framework.paths import ROOT


def module_file(module: str) -> Path | None:
    base = ROOT.joinpath(*module.split("."))
    candidates = (base.with_suffix(".py"), base / "__init__.py")
    return next((path for path in candidates if path.is_file()), None)


def absolute(node: ast.ImportFrom, package: str) -> str:
    base = node.module or ""
    if not node.level:
        return base
    parent = package.split(".")[: len(package.split(".")) - node.level + 1]
    return ".".join([*parent, *([base] if base else [])])


@cache
def imports_of(path: Path, module: str) -> frozenset[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = absolute(node, package)
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return frozenset(names)


@cache
def closure(module: str) -> tuple[Path, ...]:
    seen: dict[str, Path] = {}
    pending = [module]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = module_file(name)
        if path is None:
            continue
        seen[name] = path
        parts = name.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        pending.extend(imports_of(path, name))
    if module not in seen:
        raise ValueError(f"{module} is not a module in this repository")
    return tuple(sorted(set(seen.values())))
