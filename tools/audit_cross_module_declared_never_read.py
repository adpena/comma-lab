#!/usr/bin/env python3
"""Audit cross-module declarations whose live value is never read.

This is a narrow #904 detector.  It does not try to prove whole-repo liveness.
It proves the class that mattered for the positive control: a constructor
parameter is declared, assigned to ``self.<field>``, externally constructed from
another module, and then no production path reads the stored field.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "cross_module_declared_never_read.v1"
POSITIVE_CONTROL = (
    "tac.optimization.direct_description_joint_descent."
    "DirectDescriptionJointDescentMLXModule.margin_targets"
)
NEGATIVE_CONTROLS = (
    "tac.optimization.direct_description_joint_descent."
    "DirectDescriptionJointDescentMLXModule.seg_targets",
    "tac.optimization.direct_description_joint_descent."
    "DirectDescriptionJointDescentMLXModule.pose_targets",
    "tac.optimization.direct_description_joint_descent."
    "DirectDescriptionJointDescentMLXModule.margin_hinge_weight",
)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel: str
    module: str
    root: str
    is_test: bool
    source: str | None
    tree: ast.AST | None
    parse_error: str | None = None


@dataclass(frozen=True)
class ImportRef:
    local: str
    source_module: str
    source_name: str
    line: int


@dataclass(frozen=True)
class ModuleInfo:
    source: SourceFile
    class_defs: dict[str, str]
    imports: tuple[ImportRef, ...]
    module_aliases: dict[str, str]


@dataclass(frozen=True)
class FieldDecl:
    key: str
    class_fqn: str
    class_name: str
    module: str
    attr: str
    param: str
    rel: str
    declaration_line: int
    assignment_line: int
    init_param_order: tuple[str, ...]


@dataclass(frozen=True)
class ArgparseDecl:
    key: str
    module: str
    dest: str
    flag: str
    rel: str
    line: int


@dataclass(frozen=True)
class ReadSite:
    rel: str
    line: int
    module: str
    scope: str
    via: str


@dataclass(frozen=True)
class CallSite:
    rel: str
    line: int
    module: str
    import_hops: tuple[str, ...]
    bindings: dict[str, str]


def module_name_for_rel(rel: str) -> str:
    if rel.startswith("src/") and rel.endswith(".py"):
        return rel[len("src/") : -3].replace("/", ".")
    if rel.endswith(".py"):
        return rel[:-3].replace("/", ".")
    return rel.replace("/", ".")


def is_test_rel(rel: str) -> bool:
    name = Path(rel).name
    return "/tests/" in rel or rel.startswith("tests/") or name.startswith("test_")


DN1_SRC_FILES = (
    "src/tac/optimization/direct_description_joint_descent.py",
    "src/tac/optimization/lane_guard.py",
)
DN1_TOOL_FILES = (
    "tools/launch_ddm_joint_descent.py",
    "tools/run_ddm_j12_receiver_coordinate_custody.py",
    "tools/measure_ddm_fd2_posenull_gn_disambiguation.py",
    "tools/smoke_ddm_fd1_gn_engine.py",
)
DN1_EXPERIMENT_FILES = (
    "experiments/train_levelset_witness_realized_through_R_mlx.py",
)


def _dn1_scope(repo_root: Path, *, include_tests: bool) -> dict[str, list[Path]]:
    roots: dict[str, list[Path]] = {
        "src/tac/witness_dsl": [],
        "src/tac/optimization_controls": [],
        "tools/direct_description_callers": [],
        "experiments/trainer_argparse": [],
    }
    witness = repo_root / "src" / "tac" / "witness_dsl"
    if witness.is_dir():
        roots["src/tac/witness_dsl"] = sorted(
            path for path in witness.glob("*.py")
            if include_tests or not is_test_rel(path.relative_to(repo_root).as_posix())
        )
    for rel in DN1_SRC_FILES:
        path = repo_root / rel
        if path.is_file():
            roots["src/tac/optimization_controls"].append(path)
    for rel in DN1_TOOL_FILES:
        path = repo_root / rel
        if path.is_file():
            roots["tools/direct_description_callers"].append(path)
    for rel in DN1_EXPERIMENT_FILES:
        path = repo_root / rel
        if path.is_file():
            roots["experiments/trainer_argparse"].append(path)
    return roots


def _broad_scope(repo_root: Path, *, include_tests: bool) -> tuple[dict[str, list[Path]], int]:
    roots: dict[str, list[Path]] = {"src/tac": [], "tools": [], "experiments_top": []}
    excluded_results = 0
    src_root = repo_root / "src" / "tac"
    if src_root.is_dir():
        roots["src/tac"] = sorted(
            path for path in src_root.rglob("*.py")
            if include_tests or not is_test_rel(path.relative_to(repo_root).as_posix())
        )
    tools_root = repo_root / "tools"
    if tools_root.is_dir():
        roots["tools"] = sorted(
            path for path in tools_root.rglob("*.py")
            if include_tests or not is_test_rel(path.relative_to(repo_root).as_posix())
        )
    experiments_root = repo_root / "experiments"
    if experiments_root.is_dir():
        excluded_results = sum(1 for _ in (experiments_root / "results").rglob("*.py")) if (experiments_root / "results").is_dir() else 0
        roots["experiments_top"] = sorted(
            path for path in experiments_root.glob("*.py")
            if include_tests or not is_test_rel(path.relative_to(repo_root).as_posix())
        )
    return roots, excluded_results


def iter_python_scope(
    repo_root: Path,
    *,
    include_tests: bool = False,
    scope: str = "dn1",
) -> tuple[list[Path], dict[str, Any]]:
    if scope not in {"dn1", "broad"}:
        raise ValueError(f"unknown audit scope {scope!r}")
    if scope == "dn1":
        roots = _dn1_scope(repo_root, include_tests=include_tests)
        excluded_results = 0
        if not any(roots.values()):
            roots, excluded_results = _broad_scope(repo_root, include_tests=include_tests)
            scope = "broad"
    else:
        roots, excluded_results = _broad_scope(repo_root, include_tests=include_tests)
    paths: list[Path] = []
    for group in roots.values():
        paths.extend(group)
    denominator = {
        "roots": {name: len(group) for name, group in roots.items()},
        "scope": scope,
        "files_total": len(paths),
        "excluded_results_files": excluded_results,
    }
    return paths, denominator


def load_sources(
    repo_root: Path,
    *,
    include_tests: bool = False,
    scope: str = "dn1",
) -> tuple[list[SourceFile], dict[str, Any]]:
    paths, denominator = iter_python_scope(
        repo_root, include_tests=include_tests, scope=scope
    )
    sources: list[SourceFile] = []
    parse_errors: list[dict[str, Any]] = []
    production_files = 0
    test_files = 0
    for path in paths:
        rel = path.relative_to(repo_root).as_posix()
        test = is_test_rel(rel)
        production_files += 0 if test else 1
        test_files += 1 if test else 0
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
            sources.append(
                SourceFile(
                    path=path,
                    rel=rel,
                    module=module_name_for_rel(rel),
                    root=rel.split("/", 1)[0],
                    is_test=test,
                    source=source,
                    tree=tree,
                )
            )
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            parse_errors.append({"path": rel, "error": str(exc)})
            sources.append(
                SourceFile(
                    path=path,
                    rel=rel,
                    module=module_name_for_rel(rel),
                    root=rel.split("/", 1)[0],
                    is_test=test,
                    source=None,
                    tree=None,
                    parse_error=str(exc),
                )
            )
    denominator.update(
        {
            "production_files": production_files,
            "test_files": test_files,
            "parsed_files": sum(1 for item in sources if item.tree is not None),
            "parse_error_count": len(parse_errors),
            "parse_errors": parse_errors,
            "test_read_classification_enabled": include_tests,
        }
    )
    return sources, denominator


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur: ast.AST | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def value_mentions_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def assignment_targets_self_attr(node: ast.AST) -> tuple[ast.Attribute, ...]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    out: list[ast.Attribute] = []
    for target in targets:
        for child in ast.walk(target):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
                and isinstance(child.ctx, ast.Store)
            ):
                out.append(child)
    return tuple(out)


def function_param_order(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    args = node.args
    ordered = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    return tuple(name for name in ordered if name != "self")


def arg_line(init_node: ast.FunctionDef | ast.AsyncFunctionDef, param: str) -> int:
    for arg in (
        init_node.args.posonlyargs
        + init_node.args.args
        + init_node.args.kwonlyargs
    ):
        if arg.arg == param:
            return getattr(arg, "lineno", init_node.lineno)
    return init_node.lineno


def import_level_to_module(current: str, module: str | None, level: int) -> str | None:
    if level == 0:
        return module
    parts = current.split(".")[:-1]
    if level > len(parts) + 1:
        return None
    base = parts[: len(parts) - level + 1]
    if module:
        base.extend(module.split("."))
    return ".".join(part for part in base if part)


def collect_module_info(sources: list[SourceFile]) -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for source in sources:
        class_defs: dict[str, str] = {}
        imports: list[ImportRef] = []
        module_aliases: dict[str, str] = {}
        if source.tree is not None:
            for node in source.tree.body:
                if isinstance(node, ast.ClassDef):
                    class_defs[node.name] = f"{source.module}.{node.name}"
            for node in ast.walk(source.tree):
                if isinstance(node, ast.ImportFrom):
                    mod = import_level_to_module(source.module, node.module, node.level)
                    if mod is None:
                        continue
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        imports.append(
                            ImportRef(
                                local=alias.asname or alias.name,
                                source_module=mod,
                                source_name=alias.name,
                                line=node.lineno,
                            )
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        local = alias.asname or alias.name.split(".", 1)[0]
                        module_aliases[local] = alias.name
        modules[source.module] = ModuleInfo(
            source=source,
            class_defs=class_defs,
            imports=tuple(imports),
            module_aliases=module_aliases,
        )
    return modules


def resolve_imported_name(
    modules: dict[str, ModuleInfo], exported: dict[str, dict[str, tuple[str, tuple[str, ...]]]],
    source_module: str, source_name: str
) -> tuple[str, tuple[str, ...]] | None:
    info = modules.get(source_module)
    if info is None:
        return None
    if source_name in info.class_defs:
        return info.class_defs[source_name], (source_module,)
    target = exported.get(source_module, {}).get(source_name)
    if target is None:
        return None
    fqn, hops = target
    return fqn, (source_module, *hops)


def build_exported_aliases(
    modules: dict[str, ModuleInfo],
) -> dict[str, dict[str, tuple[str, tuple[str, ...]]]]:
    exported: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
        module: {name: (fqn, (module,)) for name, fqn in info.class_defs.items()}
        for module, info in modules.items()
    }
    changed = True
    while changed:
        changed = False
        for module, info in modules.items():
            bucket = exported.setdefault(module, {})
            for ref in info.imports:
                resolved = resolve_imported_name(
                    modules, exported, ref.source_module, ref.source_name
                )
                if resolved is None:
                    continue
                if bucket.get(ref.local) != resolved:
                    bucket[ref.local] = resolved
                    changed = True
    return exported


def collect_field_declarations(
    sources: list[SourceFile],
) -> tuple[dict[str, FieldDecl], dict[str, tuple[str, ...]]]:
    declarations: dict[str, FieldDecl] = {}
    class_param_orders: dict[str, tuple[str, ...]] = {}
    for source in sources:
        if source.tree is None or source.is_test:
            continue
        for class_node in [n for n in source.tree.body if isinstance(n, ast.ClassDef)]:
            class_fqn = f"{source.module}.{class_node.name}"
            for item in class_node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                    params = function_param_order(item)
                    class_param_orders[class_fqn] = params
                    for node in ast.walk(item):
                        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                            continue
                        for target in assignment_targets_self_attr(node):
                            for param in params:
                                if value_mentions_name(node.value, param):
                                    key = f"{class_fqn}.{target.attr}"
                                    declarations[key] = FieldDecl(
                                        key=key,
                                        class_fqn=class_fqn,
                                        class_name=class_node.name,
                                        module=source.module,
                                        attr=target.attr,
                                        param=param,
                                        rel=source.rel,
                                        declaration_line=arg_line(item, param),
                                        assignment_line=target.lineno,
                                        init_param_order=params,
                                    )
                                    break
    return declarations, class_param_orders


def collect_internal_self_reads(
    sources: list[SourceFile],
    declarations: dict[str, FieldDecl],
) -> dict[str, list[ReadSite]]:
    by_class_attr = {(d.class_fqn, d.attr): key for key, d in declarations.items()}
    reads: dict[str, list[ReadSite]] = {key: [] for key in declarations}
    for source in sources:
        if source.tree is None:
            continue
        for class_node in [n for n in source.tree.body if isinstance(n, ast.ClassDef)]:
            class_fqn = f"{source.module}.{class_node.name}"
            for node in ast.walk(class_node):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and isinstance(node.ctx, ast.Load)
                ):
                    key = by_class_attr.get((class_fqn, node.attr))
                    if key is not None:
                        reads[key].append(
                            ReadSite(
                                rel=source.rel,
                                line=node.lineno,
                                module=source.module,
                                scope="test" if source.is_test else "production",
                                via="self_attribute_load",
                            )
                        )
    return reads


def aliases_for_module(
    info: ModuleInfo,
    modules: dict[str, ModuleInfo],
    exported: dict[str, dict[str, tuple[str, tuple[str, ...]]]],
) -> dict[str, tuple[str, tuple[str, ...]]]:
    aliases: dict[str, tuple[str, tuple[str, ...]]] = {}
    for ref in info.imports:
        resolved = resolve_imported_name(modules, exported, ref.source_module, ref.source_name)
        if resolved is not None:
            aliases[ref.local] = resolved
    return aliases


def resolve_class_expr(
    expr: ast.AST,
    info: ModuleInfo,
    name_aliases: dict[str, tuple[str, tuple[str, ...]]],
    modules: dict[str, ModuleInfo],
    exported: dict[str, dict[str, tuple[str, tuple[str, ...]]]],
) -> tuple[str, tuple[str, ...]] | None:
    if isinstance(expr, ast.Name):
        return name_aliases.get(expr.id)
    if isinstance(expr, ast.Attribute):
        dotted = dotted_name(expr)
        if dotted is None:
            return None
        parts = dotted.split(".")
        if len(parts) == 2 and parts[0] in info.module_aliases:
            mod = info.module_aliases[parts[0]]
            return resolve_imported_name(modules, exported, mod, parts[1])
        for idx in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:idx])
            suffix = parts[idx:]
            if len(suffix) != 1:
                continue
            mod = info.module_aliases.get(parts[0])
            if mod and prefix.startswith(parts[0]):
                candidate_module = ".".join([mod, *parts[1:idx]])
            else:
                candidate_module = prefix
            resolved = resolve_imported_name(modules, exported, candidate_module, suffix[0])
            if resolved is not None:
                return resolved
    return None


def call_bindings(node: ast.Call, params: tuple[str, ...]) -> dict[str, str]:
    bindings = {param: "defaulted" for param in params}
    for idx, _arg in enumerate(node.args):
        if idx < len(params):
            bindings[params[idx]] = "explicit-positional"
    for keyword in node.keywords:
        if keyword.arg in bindings:
            bindings[keyword.arg] = "explicit-keyword"
    return bindings


def collect_calls_and_external_reads(
    sources: list[SourceFile],
    modules: dict[str, ModuleInfo],
    exported: dict[str, dict[str, tuple[str, tuple[str, ...]]]],
    declarations: dict[str, FieldDecl],
    class_param_orders: dict[str, tuple[str, ...]],
    reads: dict[str, list[ReadSite]],
) -> dict[str, list[CallSite]]:
    calls: dict[str, list[CallSite]] = {key: [] for key in declarations}
    keys_by_class_attr = {(d.class_fqn, d.attr): key for key, d in declarations.items()}
    for source in sources:
        info = modules[source.module]
        name_aliases = aliases_for_module(info, modules, exported)
        var_types: dict[str, str] = {}
        var_type_lines: dict[str, int] = {}
        if source.tree is None:
            continue
        for node in ast.walk(source.tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                resolved = resolve_class_expr(
                    node.value.func, info, name_aliases, modules, exported
                )
                if resolved is None:
                    continue
                class_fqn, _hops = resolved
                if class_fqn not in class_param_orders:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_types[target.id] = class_fqn
                        var_type_lines[target.id] = node.lineno
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
                resolved = resolve_class_expr(
                    node.value.func, info, name_aliases, modules, exported
                )
                if resolved is not None and isinstance(node.target, ast.Name):
                    class_fqn, _hops = resolved
                    if class_fqn in class_param_orders:
                        var_types[node.target.id] = class_fqn
                        var_type_lines[node.target.id] = node.lineno

        for node in ast.walk(source.tree):
            if isinstance(node, ast.Call):
                resolved = resolve_class_expr(node.func, info, name_aliases, modules, exported)
                if resolved is None:
                    continue
                class_fqn, hops = resolved
                params = class_param_orders.get(class_fqn)
                if params is None:
                    continue
                if source.module == class_fqn.rsplit(".", 1)[0]:
                    continue
                bindings = call_bindings(node, params)
                for key, decl in declarations.items():
                    if decl.class_fqn == class_fqn:
                        calls[key].append(
                            CallSite(
                                rel=source.rel,
                                line=node.lineno,
                                module=source.module,
                                import_hops=hops,
                                bindings=bindings,
                            )
                        )
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.value.id in var_types
            ):
                key = keys_by_class_attr.get((var_types[node.value.id], node.attr))
                if key is not None:
                    reads[key].append(
                        ReadSite(
                            rel=source.rel,
                            line=node.lineno,
                            module=source.module,
                            scope="test" if source.is_test else "production",
                            via=f"instance_attribute_load:{node.value.id}@{var_type_lines[node.value.id]}",
                        )
                    )
    return calls


def argparse_dest(flag: str, call: ast.Call) -> str:
    for keyword in call.keywords:
        if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return flag.lstrip("-").replace("-", "_")


def collect_argparse_declarations(sources: list[SourceFile]) -> dict[str, ArgparseDecl]:
    out: dict[str, ArgparseDecl] = {}
    for source in sources:
        if source.tree is None or source.is_test:
            continue
        for node in ast.walk(source.tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith("--")
            ):
                continue
            flag = node.args[0].value
            dest = argparse_dest(flag, node)
            key = f"{source.module}:argparse:{dest}"
            out[key] = ArgparseDecl(
                key=key,
                module=source.module,
                dest=dest,
                flag=flag,
                rel=source.rel,
                line=node.lineno,
            )
    return out


def collect_argparse_reads(
    sources: list[SourceFile], declarations: dict[str, ArgparseDecl]
) -> dict[str, list[ReadSite]]:
    by_module_dest = {(decl.module, decl.dest): key for key, decl in declarations.items()}
    reads: dict[str, list[ReadSite]] = {key: [] for key in declarations}
    for source in sources:
        if source.tree is None:
            continue
        for node in ast.walk(source.tree):
            dest: str | None = None
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "args"
                and isinstance(node.ctx, ast.Load)
            ):
                dest = node.attr
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "args"
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                dest = node.args[1].value
            if dest is None:
                continue
            key = by_module_dest.get((source.module, dest))
            if key is not None:
                reads[key].append(
                    ReadSite(
                        rel=source.rel,
                        line=node.lineno,
                        module=source.module,
                        scope="test" if source.is_test else "production",
                        via="argparse_namespace_load",
                    )
                )
    return reads


def site_dict(site: ReadSite | CallSite) -> dict[str, Any]:
    if isinstance(site, ReadSite):
        return {
            "path": site.rel,
            "line": site.line,
            "module": site.module,
            "scope": site.scope,
            "via": site.via,
        }
    return {
        "path": site.rel,
        "line": site.line,
        "module": site.module,
        "import_hops": list(site.import_hops),
        "bindings": dict(site.bindings),
    }


def blast_radius_rank(external_calls: list[CallSite], test_reads: list[ReadSite]) -> int:
    defaulted = sum(
        1
        for call in external_calls
        if any(mode == "defaulted" for mode in call.bindings.values())
    )
    max_hops = max((len(call.import_hops) for call in external_calls), default=0)
    return 10 * len(external_calls) + 3 * max_hops + 2 * defaulted + (0 if test_reads else 1)


def build_field_hits(
    declarations: dict[str, FieldDecl],
    reads: dict[str, list[ReadSite]],
    calls: dict[str, list[CallSite]],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for key, decl in declarations.items():
        production_reads = [r for r in reads.get(key, []) if r.scope == "production"]
        test_reads = [r for r in reads.get(key, []) if r.scope == "test"]
        external_calls = calls.get(key, [])
        if production_reads or not external_calls:
            continue
        rank = blast_radius_rank(external_calls, test_reads)
        hits.append(
            {
                "kind": "constructor_field",
                "key": key,
                "class": decl.class_fqn,
                "field": decl.attr,
                "constructor_param": decl.param,
                "declaration": {
                    "path": decl.rel,
                    "line": decl.declaration_line,
                    "assignment_line": decl.assignment_line,
                },
                "production_reads": [],
                "test_only_reads": [site_dict(site) for site in test_reads],
                "external_constructor_calls": [site_dict(site) for site in external_calls],
                "shortest_import_hops": min(len(call.import_hops) for call in external_calls),
                "blast_radius_rank": rank,
                "rank_formula": "10*external_calls + 3*max_import_hops + 2*defaulted_calls + 1_if_no_test_read",
                "routing": {
                    "owner_surface": decl.class_fqn,
                    "severed_at": f"{decl.rel}:{decl.assignment_line}",
                    "cheapest_rewire_or_retire": (
                        f"read self.{decl.attr} on the live consumer path, or remove the "
                        f"{decl.param} parameter from external callers and docs"
                    ),
                },
            }
        )
    return sorted(hits, key=lambda h: (-h["blast_radius_rank"], h["key"]))


def build_argparse_hits(
    declarations: dict[str, ArgparseDecl],
    reads: dict[str, list[ReadSite]],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for key, decl in declarations.items():
        production_reads = [r for r in reads.get(key, []) if r.scope == "production"]
        test_reads = [r for r in reads.get(key, []) if r.scope == "test"]
        if production_reads:
            continue
        hits.append(
            {
                "kind": "argparse_flag",
                "key": key,
                "flag": decl.flag,
                "dest": decl.dest,
                "declaration": {"path": decl.rel, "line": decl.line},
                "production_reads": [],
                "test_only_reads": [site_dict(site) for site in test_reads],
                "external_constructor_calls": [],
                "shortest_import_hops": 0,
                "blast_radius_rank": 1 if not test_reads else 0,
                "rank_formula": "argparse flags rank 1 when no production read, 0 when only test-read",
                "routing": {
                    "owner_surface": decl.module,
                    "severed_at": f"{decl.rel}:{decl.line}",
                    "cheapest_rewire_or_retire": (
                        f"read args.{decl.dest} in the executable path, or retire {decl.flag}"
                    ),
                },
            }
        )
    return sorted(hits, key=lambda h: (-h["blast_radius_rank"], h["key"]))


def audit_repository(
    repo_root: Path | str,
    *,
    include_tests: bool = False,
    scope: str = "dn1",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    sources, denominator = load_sources(root, include_tests=include_tests, scope=scope)
    modules = collect_module_info(sources)
    exported = build_exported_aliases(modules)
    field_decls, class_param_orders = collect_field_declarations(sources)
    field_reads = collect_internal_self_reads(sources, field_decls)
    field_calls = collect_calls_and_external_reads(
        sources, modules, exported, field_decls, class_param_orders, field_reads
    )
    argparse_decls = collect_argparse_declarations(sources)
    argparse_reads = collect_argparse_reads(sources, argparse_decls)
    field_hits = build_field_hits(field_decls, field_reads, field_calls)
    argparse_hits = build_argparse_hits(argparse_decls, argparse_reads)
    hits = sorted(
        [*field_hits, *argparse_hits],
        key=lambda h: (-int(h["blast_radius_rank"]), h["kind"], h["key"]),
    )
    denominator.update(
        {
            "constructor_field_declarations": len(field_decls),
            "argparse_declarations": len(argparse_decls),
            "chains_traced": sum(len(v) for v in field_calls.values()),
            "none_path_rows": len(hits),
            "constructor_field_none_path_rows": len(field_hits),
            "argparse_none_path_rows": len(argparse_hits),
        }
    )
    report = {
        "schema": SCHEMA,
        "repo_root": str(root),
        "denominator": denominator,
        "controls": {},
        "hits": hits,
        "limitations": [
            "Constructor-field detection is intentionally scoped to __init__ params assigned to self fields.",
            "A production self/instance attribute load is treated as a conservative read, not proof of score effect.",
            "Dynamic imports, reflection, setattr/getattr on arbitrary instances, and whole-repo dataflow are out of scope.",
            "Argparse support is a namespace-read screen; it does not prove behavioral influence after the read.",
        ],
    }
    report["controls"] = run_canonical_controls(report)
    return report


def run_canonical_controls(report: dict[str, Any]) -> dict[str, Any]:
    hit_keys = {hit["key"] for hit in report.get("hits", [])}
    positive = POSITIVE_CONTROL in hit_keys
    negatives = {key: key not in hit_keys for key in NEGATIVE_CONTROLS}
    return {
        "positive": {
            "key": POSITIVE_CONTROL,
            "expected": "hit",
            "passed": positive,
        },
        "negatives": [
            {"key": key, "expected": "not_hit", "passed": passed}
            for key, passed in negatives.items()
        ],
        "passed": positive and all(negatives.values()),
    }


def controls_failure_text(controls: dict[str, Any]) -> str:
    failures: list[str] = []
    if not controls["positive"]["passed"]:
        failures.append(f"positive control did not flag: {controls['positive']['key']}")
    for item in controls["negatives"]:
        if not item["passed"]:
            failures.append(f"negative control incorrectly flagged: {item['key']}")
    return "; ".join(failures)


def summarize(report: dict[str, Any]) -> str:
    denom = report["denominator"]
    controls = report["controls"]
    return (
        f"{SCHEMA}: controls={'PASS' if controls['passed'] else 'FAIL'}; "
        f"files={denom['files_total']} parsed={denom['parsed_files']} "
        f"parse_errors={denom['parse_error_count']} declarations="
        f"{denom['constructor_field_declarations']} fields + "
        f"{denom['argparse_declarations']} argparse; none_path_rows={denom['none_path_rows']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scope", choices=("dn1", "broad"), default="dn1")
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="include test files so test-only reads are classified in the sweep",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_repository(
        args.repo_root, include_tests=args.include_tests, scope=args.scope
    )
    controls = report["controls"]
    if not controls["passed"]:
        print(f"CONTROL FAILURE: {controls_failure_text(controls)}", file=sys.stderr)
        return 2
    if args.controls_only:
        print("controls PASS")
        return 0
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(summarize(report))
    for hit in report["hits"][:20]:
        decl = hit["declaration"]
        print(
            f"{hit['kind']} {hit['key']} @ {decl['path']}:{decl['line']} "
            f"rank={hit['blast_radius_rank']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
