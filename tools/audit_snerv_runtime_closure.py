#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit SNeRV generated runtime byte closure without pruning it."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import sha256_bytes, sha256_file, tree_sha256, write_json  # noqa: E402
from tac.submission_archive import safe_extract_zip  # noqa: E402

SCHEMA = "snerv_runtime_closure_audit.v1"
AXIS_TAG = "[receiver-safe:false-authority]"

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}

DEFAULT_IMPORT_TARGETS = ("inflate",)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_snerv_runtime_closure(
        archive_zip_path=args.archive_zip,
        runtime_package_dir=args.runtime_package_dir,
        receiver_proof_json=args.receiver_proof_json,
        run_import_smoke=not args.no_import_smoke,
        scratch_dir=args.scratch_dir,
        generated_utc=datetime.now(UTC).isoformat(),
    )
    write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_json": str(args.output_json),
                "archive_zip_bytes": report["archive_zip"]["bytes"],
                "payload_member_compressed_bytes": report["byte_accounting"]["payload_member_compressed_bytes"],
                "runtime_member_compressed_bytes": report["byte_accounting"]["runtime_member_compressed_bytes"],
                "unused_runtime_member_compressed_bytes": report["byte_accounting"][
                    "unused_runtime_member_compressed_bytes"
                ],
                "import_smoke_passed": report["import_smoke"]["passed"],
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0


def audit_snerv_runtime_closure(
    *,
    archive_zip_path: str | Path,
    runtime_package_dir: str | Path | None = None,
    receiver_proof_json: str | Path | None = None,
    run_import_smoke: bool = True,
    scratch_dir: str | Path | None = None,
    generated_utc: str,
) -> dict[str, Any]:
    archive_zip = Path(archive_zip_path).expanduser().resolve(strict=False)
    if not archive_zip.is_file():
        raise FileNotFoundError(f"archive zip not found: {archive_zip}")

    runtime_package = (
        None if runtime_package_dir is None else Path(runtime_package_dir).expanduser().resolve(strict=False)
    )
    proof_path = _resolve_receiver_proof_path(
        runtime_package_dir=runtime_package,
        explicit=receiver_proof_json,
    )
    zip_members, member_payloads = _zip_member_rows(archive_zip)
    source_minification = _source_minification_estimates(zip_members, member_payloads)
    upstream_contract = _upstream_contest_bundle_contract(zip_members, member_payloads)
    submission_root_context = _submission_root_context(
        archive_zip=archive_zip,
        runtime_package_dir=runtime_package,
        scratch_dir=scratch_dir,
    )
    with submission_root_context as submission_root:
        python_files = _python_module_index(submission_root)
        import_graph = _build_static_import_graph(python_files)
        import_smoke = (
            _run_import_smoke(submission_root, DEFAULT_IMPORT_TARGETS)
            if run_import_smoke
            else {"passed": None, "skipped": True, "blockers": []}
        )
        runtime_tree_sha = tree_sha256(submission_root) if submission_root.exists() else None

    reachability = _runtime_reachability(zip_members, import_graph)
    receiver_proof = _read_receiver_proof(proof_path)
    byte_accounting = _byte_accounting(zip_members, reachability)
    blockers = _blockers(
        import_graph=import_graph,
        import_smoke=import_smoke,
        receiver_proof=receiver_proof,
    )
    report = {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "generated_utc": generated_utc,
        "operation": "snerv_runtime_static_closure_and_byte_profile",
        "inputs": {
            "archive_zip_path": archive_zip.as_posix(),
            "runtime_package_dir": (None if runtime_package is None else runtime_package.as_posix()),
            "receiver_proof_json": None if proof_path is None else proof_path.as_posix(),
            "scratch_dir": None if scratch_dir is None else str(scratch_dir),
        },
        "archive_zip": {
            "path": archive_zip.as_posix(),
            "bytes": archive_zip.stat().st_size,
            "sha256": sha256_file(archive_zip),
            "member_count": len(zip_members),
        },
        "runtime_package": {
            "path": None if runtime_package is None else runtime_package.as_posix(),
            "submission_tree_sha256": runtime_tree_sha,
            "source_kind": (
                "runtime_package_submission_dir"
                if runtime_package is not None and (runtime_package / "submission").is_dir()
                else "archive_zip_extracted_view"
            ),
        },
        "byte_accounting": byte_accounting,
        "upstream_contest_bundle_contract": upstream_contract,
        "zip_members": zip_members,
        "source_minification_estimates": source_minification,
        "static_import_graph": import_graph,
        "runtime_reachability": reachability,
        "import_smoke": import_smoke,
        "receiver_proof": receiver_proof,
        "materialization_candidates": _materialization_candidates(
            byte_accounting,
            reachability,
        ),
        "launchability": {
            "audit_only": True,
            "candidate_package_launchable": False,
            "blocked_long_training_rows_must_not_launch": True,
            "reason": (
                "runtime pruning/specialization has not been materialized and "
                "receiver-proven as its own byte-closed archive"
            ),
        },
        "blockers": blockers,
        "next_actions": [
            "materialize_minimal_runtime_candidate_only_from_static_reachable_set",
            "run_full_video_receiver_replay_on_pruned_runtime_candidate",
            "compare_pruned_runtime_archive_bytes_against_source_archive_bytes",
            "keep_exact_score_and_launch_authority_false_until_paired_contest_eval",
        ],
        **FALSE_AUTHORITY,
    }
    # Keep payload bytes out of the report while ensuring member data was read,
    # hashed, and profiled during zip inspection.
    del member_payloads
    return report


def _zip_member_rows(archive_zip: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(archive_zip, "r") as zf:
        for index, info in enumerate(zf.infolist()):
            payload = zf.read(info.filename)
            payloads[info.filename] = payload
            rows.append(
                {
                    "index": index,
                    "filename": info.filename,
                    "kind": _member_kind(info.filename),
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "crc": int(info.CRC),
                    "sha256": sha256_bytes(payload),
                }
            )
    return rows, payloads


class _SubmissionRootContext:
    def __init__(
        self,
        *,
        archive_zip: Path,
        runtime_package_dir: Path | None,
        scratch_dir: str | Path | None,
    ) -> None:
        self.archive_zip = archive_zip
        self.runtime_package_dir = runtime_package_dir
        self.scratch_dir = scratch_dir
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        if self.runtime_package_dir is not None:
            submission = self.runtime_package_dir / "submission"
            if submission.is_dir():
                return submission
        temp_parent = _scratch_parent(self.scratch_dir, fallback=self.archive_zip.parent)
        self._tmp = tempfile.TemporaryDirectory(
            prefix="snerv_runtime_closure_import_",
            dir=str(temp_parent),
        )
        root = Path(self._tmp.name)
        safe_extract_zip(self.archive_zip, root)
        return root

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


def _submission_root_context(
    *,
    archive_zip: Path,
    runtime_package_dir: Path | None,
    scratch_dir: str | Path | None,
) -> _SubmissionRootContext:
    return _SubmissionRootContext(
        archive_zip=archive_zip,
        runtime_package_dir=runtime_package_dir,
        scratch_dir=scratch_dir,
    )


def _scratch_parent(scratch_dir: str | Path | None, *, fallback: Path) -> Path:
    parent = fallback if scratch_dir is None else Path(scratch_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _python_module_index(root: Path) -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        module = _module_name_for_relpath(rel)
        if module is None:
            continue
        modules[module] = {
            "module": module,
            "path": path,
            "relpath": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return modules


def _module_name_for_relpath(rel: str) -> str | None:
    if rel == "inflate.py":
        return "inflate"
    if not rel.startswith("src/") or not rel.endswith(".py"):
        return None
    stem = rel[:-3]
    parts = stem.split("/")[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def _build_static_import_graph(python_files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    edges: dict[str, list[str]] = {}
    external_import_roots: set[str] = set()
    parse_errors: list[dict[str, str]] = []
    for module, row in python_files.items():
        path = Path(row["path"])
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            parse_errors.append({"module": module, "path": row["relpath"], "error": str(exc)})
            continue
        imports = _imports_from_ast(tree, module=module)
        edges[module] = sorted(imports)
        for imported in imports:
            if not imported.startswith(("tac.", "inflate")):
                external_import_roots.add(imported.split(".", 1)[0])

    reachable = _reachable_modules(edges, python_files)
    missing_tac_imports = sorted(
        imported for imported in _all_imports(edges) if imported.startswith("tac.") and imported not in python_files
    )
    return {
        "entry_modules": ["inflate"],
        "module_count": len(python_files),
        "reachable_modules": sorted(reachable),
        "reachable_module_count": len(reachable),
        "missing_tac_imports": missing_tac_imports,
        "external_import_roots": sorted(external_import_roots),
        "parse_errors": parse_errors,
        "edges": {key: edges[key] for key in sorted(edges)},
    }


def _imports_from_ast(tree: ast.AST, *, module: str) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from(node, module=module)
            if resolved:
                imports.add(resolved)
    return imports


def _resolve_import_from(node: ast.ImportFrom, *, module: str) -> str | None:
    if node.level <= 0:
        return node.module
    package_parts = module.split(".")[:-1]
    if module.endswith("__init__"):
        package_parts = module.split(".")[:-1]
    drop = max(0, int(node.level) - 1)
    if drop:
        package_parts = package_parts[:-drop]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(part for part in package_parts if part)


def _reachable_modules(
    edges: Mapping[str, Sequence[str]],
    python_files: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    reachable: set[str] = set()
    queue: deque[str] = deque(["inflate"])
    while queue:
        module = queue.popleft()
        if module in reachable:
            continue
        if module not in python_files:
            continue
        reachable.add(module)
        for imported in edges.get(module, ()):
            if imported in python_files and imported not in reachable:
                queue.append(imported)
    for module in list(reachable):
        parts = module.split(".")
        for end in range(1, len(parts)):
            package = ".".join(parts[:end])
            if package in python_files:
                reachable.add(package)
    return reachable


def _all_imports(edges: Mapping[str, Sequence[str]]) -> set[str]:
    imports: set[str] = set()
    for values in edges.values():
        imports.update(values)
    return imports


def _runtime_reachability(
    zip_members: Sequence[Mapping[str, Any]],
    import_graph: Mapping[str, Any],
) -> dict[str, Any]:
    reachable_modules = set(import_graph.get("reachable_modules") or ())
    rows: list[dict[str, Any]] = []
    for member in zip_members:
        filename = str(member["filename"])
        module = _module_name_for_relpath(filename)
        reachable = filename in {"inflate.sh", "0.bin"} or (module is not None and module in reachable_modules)
        rows.append(
            {
                "filename": filename,
                "module": module,
                "kind": member["kind"],
                "reachable_from_inflate_entrypoint": bool(reachable),
                "file_size": int(member["file_size"]),
                "compress_size": int(member["compress_size"]),
            }
        )
    unused = [
        row
        for row in rows
        if row["kind"] in {"runtime_python", "runtime_entrypoint"} and not row["reachable_from_inflate_entrypoint"]
    ]
    return {
        "rows": rows,
        "unused_runtime_members": unused,
        "unused_runtime_member_count": len(unused),
    }


def _byte_accounting(
    zip_members: Sequence[Mapping[str, Any]],
    reachability: Mapping[str, Any],
) -> dict[str, Any]:
    archive_member_compressed = sum(int(row["compress_size"]) for row in zip_members)
    payload_member_compressed = sum(int(row["compress_size"]) for row in zip_members if row["filename"] == "0.bin")
    runtime_member_compressed = sum(int(row["compress_size"]) for row in zip_members if row["filename"] != "0.bin")
    runtime_python_compressed = sum(
        int(row["compress_size"]) for row in zip_members if str(row["filename"]).endswith(".py")
    )
    unused_runtime_member_compressed = sum(
        int(row["compress_size"]) for row in reachability.get("unused_runtime_members", ())
    )
    return {
        "archive_member_compressed_bytes": archive_member_compressed,
        "payload_member_compressed_bytes": payload_member_compressed,
        "runtime_member_compressed_bytes": runtime_member_compressed,
        "runtime_python_compressed_bytes": runtime_python_compressed,
        "unused_runtime_member_compressed_bytes": unused_runtime_member_compressed,
        "runtime_over_payload_compressed_ratio": (
            None if payload_member_compressed == 0 else runtime_member_compressed / payload_member_compressed
        ),
    }


def _upstream_contest_bundle_contract(
    zip_members: Sequence[Mapping[str, Any]],
    member_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    payload_members = {
        str(row["filename"]): member_payloads[str(row["filename"])]
        for row in zip_members
        if row["kind"] == "payload_packet"
    }
    data_only_zip = _deterministic_zip_bytes(payload_members)
    current_archive_runtime_members = [
        str(row["filename"]) for row in zip_members if row["kind"] != "payload_packet"
    ]
    return {
        "schema": "snerv_upstream_contest_bundle_contract.v1",
        "sources_checked": [
            "upstream/README.md",
            "upstream/evaluate.py",
            "upstream/evaluate.sh",
            "upstream/.github/workflows/eval.yml",
            "src/comma_lab/evaluate.py",
            "src/comma_lab/install.py",
        ],
        "upstream_rate_uses_archive_zip_stat_only": True,
        "upstream_evaluate_py_rate_expression": (
            "compressed_size = (args.submission_dir / 'archive.zip').stat().st_size"
        ),
        "upstream_evaluate_sh_unzips_archive_before_inflate": True,
        "upstream_inflate_sh_runs_from_submission_dir_not_archive_member": True,
        "upstream_workflow_downloads_archive_zip_into_pr_submission_dir": True,
        "runtime_source_in_pr_checkout_not_counted_by_upstream_rate_formula": True,
        "internal_rule_faithful_payload_note": (
            "src/comma_lab/install.py may sum archive.zip plus runtime files for "
            "local rule-faithful accounting; keep this separate from upstream "
            "evaluate.py score authority"
        ),
        "current_archive_contains_runtime_members": bool(current_archive_runtime_members),
        "current_archive_runtime_member_count": len(current_archive_runtime_members),
        "current_archive_runtime_members": current_archive_runtime_members,
        "data_only_archive_zip_estimate": {
            "member_names": sorted(payload_members),
            "zip_bytes": len(data_only_zip),
            "zip_sha256": sha256_bytes(data_only_zip),
            "payload_member_count": len(payload_members),
        },
        "score_authority": "false_until_exact_upstream_evaluate_replay",
        **FALSE_AUTHORITY,
    }


def _source_minification_estimates(
    zip_members: Sequence[Mapping[str, Any]],
    member_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for member in zip_members:
        filename = str(member["filename"])
        if not filename.endswith(".py"):
            continue
        payload = member_payloads[filename]
        estimated = _ast_minify_python_source(payload)
        row = {
            "filename": filename,
            "source_bytes": len(payload),
            "source_zip_compress_size": int(member["compress_size"]),
            "ast_minified_source_bytes": estimated.get("bytes"),
            "ast_minified_zip_compress_size": estimated.get("zip_compress_size"),
            "ast_minified_source_delta_bytes": estimated.get("source_delta_bytes"),
            "ast_minified_zip_delta_bytes": estimated.get("zip_delta_bytes"),
            "identifier_renaming_attempted": False,
            "semantics_proven": False,
            "blockers": list(estimated.get("blockers") or ()),
        }
        rows.append(row)
    total_source = sum(int(row["source_zip_compress_size"]) for row in rows)
    total_minified = sum(
        int(row["ast_minified_zip_compress_size"] or row["source_zip_compress_size"])
        for row in rows
    )
    rows.sort(key=lambda row: int(row["source_zip_compress_size"]), reverse=True)
    return {
        "schema": "snerv_runtime_source_minification_estimate.v1",
        "method": "strip_docstrings_and_comments_via_ast_unparse_estimate_only",
        "runtime_python_member_count": len(rows),
        "source_zip_compressed_bytes": total_source,
        "ast_minified_zip_compressed_bytes": total_minified,
        "ast_minified_estimated_saved_zip_bytes": total_source - total_minified,
        "rows": rows,
        "materialized": False,
        "receiver_replay_required": True,
        "identifier_renaming_required_for_no_human_symbols": True,
        "blockers": [
            "runtime_source_minification_not_materialized",
            "identifier_renaming_not_implemented",
            "minified_runtime_receiver_replay_missing",
        ],
        **FALSE_AUTHORITY,
    }


def _ast_minify_python_source(payload: bytes) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        text = payload.decode("utf-8")
        tree = ast.parse(text)
        tree = _DocstringStripper().visit(tree)
        ast.fix_missing_locations(tree)
        minified = ast.unparse(tree).encode("utf-8")
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        return {
            "bytes": None,
            "zip_compress_size": None,
            "source_delta_bytes": None,
            "zip_delta_bytes": None,
            "blockers": [f"ast_minification_failed:{type(exc).__name__}"],
        }
    source_zip = _zip_member_compress_size(payload)
    minified_zip = _zip_member_compress_size(minified)
    if minified_zip > source_zip:
        blockers.append("ast_minified_source_compresses_larger_than_original")
    return {
        "bytes": len(minified),
        "zip_compress_size": minified_zip,
        "source_delta_bytes": len(minified) - len(payload),
        "zip_delta_bytes": minified_zip - source_zip,
        "blockers": blockers,
    }


class _DocstringStripper(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = _strip_docstring_expr(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = _strip_docstring_expr(node.body)
        return node

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = _strip_docstring_expr(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = _strip_docstring_expr(node.body)
        return node


def _strip_docstring_expr(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:] or [ast.Pass()]
    return body


def _zip_member_compress_size(payload: bytes) -> int:
    import io

    buf = io.BytesIO()
    info = zipfile.ZipInfo("x.py", date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(info, payload)
        return int(zf.infolist()[0].compress_size)


def _deterministic_zip_bytes(members: Mapping[str, bytes]) -> bytes:
    import io

    buf = io.BytesIO()
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in sorted(members):
            info = zipfile.ZipInfo(filename, date_time=fixed_ts)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, members[filename])
    return buf.getvalue()


def _run_import_smoke(root: Path, targets: Sequence[str]) -> dict[str, Any]:
    code = (
        "import importlib, pathlib, sys\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "sys.path.insert(0, str(root))\n"
        "sys.path.insert(0, str(root / 'src'))\n"
        "targets = sys.argv[2:]\n"
        "for target in targets:\n"
        "    importlib.import_module(target)\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, root.as_posix(), *targets],
        check=False,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        timeout=30,
    )
    passed = result.returncode == 0
    return {
        "passed": passed,
        "skipped": False,
        "targets": list(targets),
        "returncode": int(result.returncode),
        "stdout": _safe_text(result.stdout),
        "stderr": _safe_text(result.stderr),
        "blockers": [] if passed else ["snerv_runtime_import_smoke_failed"],
    }


def _read_receiver_proof(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "path": None if path is None else path.as_posix(),
            "present": False,
            "runtime_consumption_proof_passed": False,
            "receiver_contract_satisfied": False,
            "blockers": ["snerv_runtime_receiver_proof_missing"],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "present": True,
        "schema": payload.get("schema"),
        "archive_sha256": payload.get("archive_sha256"),
        "archive_bytes": payload.get("archive_bytes"),
        "runtime_tree_sha256": payload.get("runtime_tree_sha256"),
        "runtime_consumption_proof_passed": (payload.get("runtime_consumption_proof_passed") is True),
        "receiver_contract_satisfied": (payload.get("receiver_contract_satisfied") is True),
        "receiver_output_bytes": payload.get("receiver_output_bytes"),
        "receiver_output_sha256": payload.get("receiver_output_sha256"),
        "receiver_output_retained": payload.get("receiver_output_retained"),
        "blockers": list(payload.get("blockers") or ()),
    }


def _resolve_receiver_proof_path(
    *,
    runtime_package_dir: Path | None,
    explicit: str | Path | None,
) -> Path | None:
    if explicit is not None:
        return Path(explicit).expanduser().resolve(strict=False)
    if runtime_package_dir is None:
        return None
    candidates = sorted((runtime_package_dir / "receiver_proof").glob("*.json"))
    return candidates[0] if candidates else None


def _materialization_candidates(
    byte_accounting: Mapping[str, Any],
    reachability: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    unused_bytes = int(byte_accounting["unused_runtime_member_compressed_bytes"])
    if unused_bytes > 0:
        candidates.append(
            {
                "id": "snerv_runtime_static_unreachable_member_prune",
                "operation": "drop_static_unreachable_python_members",
                "candidate_saved_bytes_upper_bound": unused_bytes,
                "source_rows": list(reachability.get("unused_runtime_members") or ()),
                "materialized": False,
                "receiver_replay_required": True,
                "blockers": [
                    "snerv_runtime_pruned_archive_not_materialized",
                    "snerv_runtime_pruned_archive_receiver_replay_missing",
                ],
                **FALSE_AUTHORITY,
            }
        )
    candidates.append(
        {
            "id": "snerv_runtime_candidate_specific_inline_inflate",
            "operation": "candidate_specific_static_inflate_runtime_specialization",
            "candidate_saved_bytes_upper_bound": int(byte_accounting["runtime_member_compressed_bytes"]),
            "materialized": False,
            "receiver_replay_required": True,
            "contest_compliance_notes": [
                "no scorer imports",
                "no external sidecars",
                "same inflate.sh signature",
                "full raw output parity must be reproven before use",
            ],
            "blockers": [
                "minimal_snerv_runtime_closure_not_materialized",
                "contest_inflate_dependency_closure_not_proven_for_pruned_runtime",
                "full_raw_output_parity_missing_for_pruned_runtime",
            ],
            **FALSE_AUTHORITY,
        }
    )
    return candidates


def _blockers(
    *,
    import_graph: Mapping[str, Any],
    import_smoke: Mapping[str, Any],
    receiver_proof: Mapping[str, Any],
) -> list[str]:
    blockers = [
        "minimal_snerv_runtime_closure_not_materialized",
        "contest_inflate_dependency_closure_not_proven_for_pruned_runtime",
        "runtime_source_minification_not_materialized",
        "full_video_scorer_replay_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
    ]
    if import_graph.get("missing_tac_imports"):
        blockers.append("snerv_runtime_static_import_closure_missing_members")
    if import_graph.get("parse_errors"):
        blockers.append("snerv_runtime_static_import_parse_errors")
    if import_smoke.get("passed") is False:
        blockers.extend(import_smoke.get("blockers") or ())
    if receiver_proof.get("runtime_consumption_proof_passed") is not True:
        blockers.append("snerv_runtime_source_receiver_proof_missing_or_failed")
    if receiver_proof.get("receiver_contract_satisfied") is not True:
        blockers.append("snerv_runtime_source_receiver_contract_missing_or_failed")
    return _dedupe(blockers)


def _member_kind(filename: str) -> str:
    if filename == "0.bin":
        return "payload_packet"
    if filename == "inflate.sh":
        return "runtime_shell_entrypoint"
    if filename == "inflate.py":
        return "runtime_entrypoint"
    if filename.endswith("__init__.py"):
        return "runtime_namespace_stub"
    if filename.endswith(".py"):
        return "runtime_python"
    return "runtime_other"


def _safe_text(value: object, *, max_chars: int = 4096) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _dedupe(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", required=True, type=Path)
    parser.add_argument("--runtime-package-dir", type=Path)
    parser.add_argument("--receiver-proof-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument(
        "--no-import-smoke",
        action="store_true",
        help="Skip import-only dependency smoke.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
