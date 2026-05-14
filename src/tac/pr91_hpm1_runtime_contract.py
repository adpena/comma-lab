# SPDX-License-Identifier: MIT
"""Static runtime-contract audit for public PR91/HPM1 HPAC replay."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from tac.pr91_hpm1_codec import DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR, REPO_ROOT
from tac.repo_io import repo_relative, sha256_file

SCHEMA_VERSION = 1
KIND = "pr91_hpm1_runtime_contract"

DECOMPRESS_FUNCTION = "decompress_tokens_hpac"
DEVICE_ARG_INDEX = 8


def _source_file_record(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": repo_relative(path, REPO_ROOT),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else "",
    }


def _parse(path: Path) -> ast.Module | None:
    if not path.is_file():
        return None
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())


def _import_aliases(tree: ast.Module) -> set[str]:
    aliases = {DECOMPRESS_FUNCTION}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "pr86_hpac":
            continue
        for alias in node.names:
            if alias.name == DECOMPRESS_FUNCTION:
                aliases.add(alias.asname or alias.name)
    return aliases


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _device_arg(call: ast.Call, source: str) -> tuple[str, str]:
    for keyword in call.keywords:
        if keyword.arg == "device":
            return ast.get_source_segment(source, keyword.value) or "", _classify_expr(keyword.value)
    if len(call.args) > DEVICE_ARG_INDEX:
        arg = call.args[DEVICE_ARG_INDEX]
        return ast.get_source_segment(source, arg) or "", _classify_expr(arg)
    return "", "missing"


def _classify_expr(expr: ast.AST) -> str:
    if isinstance(expr, ast.Constant) and expr.value == "cpu":
        return "literal_cpu"
    if isinstance(expr, ast.Constant) and expr.value == "cuda":
        return "literal_cuda"
    if isinstance(expr, ast.Name) and expr.id == "device":
        return "ambient_device"
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "str"
        and len(expr.args) == 1
        and isinstance(expr.args[0], ast.Name)
        and expr.args[0].id == "device"
    ):
        return "ambient_device_stringified"
    return type(expr).__name__


def _cpu_intent(lines: list[str], lineno: int) -> tuple[bool, list[str]]:
    start = max(0, lineno - 7)
    window = lines[start:lineno]
    hits = [
        line.strip()
        for line in window
        if "Force HPAC decode onto CPU" in line
        or "CPU encoder bit-exactly" in line
        or "CPU FP32" in line
    ]
    return bool(hits), hits


def _function_signature(tree: ast.Module) -> dict[str, Any]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != DECOMPRESS_FUNCTION:
            continue
        args = [arg.arg for arg in node.args.args]
        return {
            "found": True,
            "line": node.lineno,
            "args": args,
            "device_arg_index": args.index("device") if "device" in args else None,
        }
    return {"found": False, "line": None, "args": [], "device_arg_index": None}


def _call_sites(path: Path, tree: ast.Module | None) -> list[dict[str, Any]]:
    if tree is None:
        return []
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    aliases = _import_aliases(tree)
    rows: list[dict[str, Any]] = []
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _call_name(node.func)
        if callee not in aliases:
            continue
        device_expression, device_class = _device_arg(node, source)
        cpu_intent, comments = _cpu_intent(lines, node.lineno)
        parent = parents.get(node)
        while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = parents.get(parent)
        rows.append(
            {
                "path": repo_relative(path, REPO_ROOT),
                "line": node.lineno,
                "function": parent.name if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) else "",
                "callee": callee,
                "device_expression": device_expression,
                "device_class": device_class,
                "preceding_cpu_intent_comment": cpu_intent,
                "cpu_intent_comments": comments,
                "contradiction": cpu_intent and device_class not in {"literal_cpu"},
            }
        )
    return sorted(rows, key=lambda row: (row["path"], row["line"], row["callee"]))


def _device_contract(call_sites: list[dict[str, Any]], *, missing_files: list[str]) -> dict[str, Any]:
    if missing_files:
        return {
            "status": "blocked_missing_runtime_sources",
            "resolved_device": None,
            "passed": False,
            "reason": "runtime sources are missing",
        }
    if not call_sites:
        return {
            "status": "blocked_no_hpac_call_sites",
            "resolved_device": None,
            "passed": False,
            "reason": "no HPAC decompress call sites were statically visible",
        }
    classes = {str(row["device_class"]) for row in call_sites}
    contradictions = [row for row in call_sites if row["contradiction"]]
    if not contradictions and classes == {"literal_cpu"}:
        return {
            "status": "resolved_cpu_only",
            "resolved_device": "cpu",
            "passed": True,
            "reason": "all visible HPAC decompress call sites pass literal CPU",
        }
    return {
        "status": "blocked_ambient_or_contradictory",
        "resolved_device": None,
        "passed": False,
        "reason": "HPAC decode device contract is ambient or contradictory; byte parity must resolve CPU vs CUDA",
        "device_classes": sorted(classes),
        "contradiction_count": len(contradictions),
    }


def audit_pr91_hpm1_runtime_contract(
    *,
    source_dir: str | Path = DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR,
) -> dict[str, Any]:
    """Audit HPAC device-contract evidence in the public PR91 runtime sources."""

    root = Path(source_dir)
    inflate_py = root / "inflate.py"
    pr86_py = root / "pr86_hpac.py"
    inflate_tree = _parse(inflate_py)
    pr86_tree = _parse(pr86_py)
    call_sites = _call_sites(inflate_py, inflate_tree) + _call_sites(pr86_py, pr86_tree)
    ambient_calls = [
        row
        for row in call_sites
        if row["device_class"] in {"ambient_device", "ambient_device_stringified", "literal_cuda"}
    ]
    contradictions = [row for row in call_sites if row["contradiction"]]
    missing_files = [
        record["path"]
        for record in (_source_file_record(inflate_py), _source_file_record(pr86_py))
        if not record["exists"]
    ]
    device_contract = _device_contract(call_sites, missing_files=missing_files)
    gates = {
        "runtime_sources_present": {
            "passed": not missing_files,
            "status": "passed" if not missing_files else "missing",
            "required_for_dispatch": True,
            "reason": "public PR91 inflate.py and pr86_hpac.py are present"
            if not missing_files
            else "public PR91 runtime sources are missing",
        },
        "hpac_decoder_signature_found": {
            "passed": bool(pr86_tree is not None and _function_signature(pr86_tree)["found"]),
            "status": "passed"
            if pr86_tree is not None and _function_signature(pr86_tree)["found"]
            else "failed_closed",
            "required_for_dispatch": True,
            "reason": "decompress_tokens_hpac signature is statically visible",
        },
        "hpac_device_contract_resolved": {
            "passed": bool(device_contract["passed"]),
            "status": "passed" if device_contract["passed"] else "blocked",
            "required_for_dispatch": True,
            "reason": device_contract["reason"],
        },
        "runtime_consumer_sidecar_free_hpm1": {
            "passed": False,
            "status": "blocked",
            "required_for_dispatch": True,
            "reason": "static source audit does not prove sidecar-free contest HPM1 loading",
        },
    }
    blockers = [
        name
        for name, gate in sorted(gates.items())
        if gate["required_for_dispatch"] and not gate["passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "static_runtime_contract_audit",
        "source_dir": repo_relative(root, REPO_ROOT),
        "files": {
            "inflate_py": _source_file_record(inflate_py),
            "pr86_hpac_py": _source_file_record(pr86_py),
        },
        "hpac_decoder_signature": _function_signature(pr86_tree) if pr86_tree is not None else {"found": False},
        "call_sites": call_sites,
        "ambient_device_call_count": len(ambient_calls),
        "contradiction_count": len(contradictions),
        "contradictions": contradictions,
        "hpac_device_contract": device_contract,
        "gates": gates,
        "dispatch_blockers": blockers,
        "next_safe_actions": [
            "Resolve whether public PR91 HPAC arithmetic decode must be CPU-only or CUDA-compatible.",
            "Pin a byte-exact device contract in local replay before full 600-frame decode/reencode work.",
            "Require runtime-consumption proof before any PR91/HPM1 exact CUDA dispatch.",
        ],
    }


__all__ = [
    "DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR",
    "KIND",
    "SCHEMA_VERSION",
    "audit_pr91_hpm1_runtime_contract",
]
