#!/usr/bin/env python3
"""Audit Modal image method ordering and repo import path contracts.

Modal local mounts (``Image.add_local_file`` / ``Image.add_local_dir``) must be
the final phase of an image chain. Build/environment steps after a local mount
fail at app startup with ``InvalidError`` before any remote work begins.

Images that mount the repo ``src`` tree must also set ``PYTHONPATH`` before the
mount phase. Modal imports the function module on the worker before user code
runs; without the image-level import path, top-level ``tac`` imports fail before
the eval or training entrypoint can repair ``sys.path``.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.repo_io import json_text  # noqa: E402

LOCAL_MOUNT_METHODS = {"add_local_dir", "add_local_file"}


@dataclass(frozen=True)
class ModalImageOrderViolation:
    path: str
    line: int
    method: str
    message: str


def tracked_python_paths(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {proc.stderr.strip()}")
    paths: list[Path] = []
    for line in proc.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        if rel.startswith(("experiments/results/", "reports/raw/")):
            continue
        paths.append(root / rel)
    return paths


def _call_chain_nodes(node: ast.AST) -> list[tuple[str, int, ast.Call]]:
    """Return fluent method call nodes in execution order."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return []
    return [*_call_chain_nodes(node.func.value), (node.func.attr, node.lineno, node)]


def _contains_string_literal(node: ast.AST, expected: str) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value == expected or node.value.endswith(f"/{expected}")
    return any(_contains_string_literal(child, expected) for child in ast.iter_child_nodes(node))


def _mounts_repo_src(method: str, call: ast.Call) -> bool:
    if method != "add_local_dir":
        return False
    for arg in call.args:
        if _contains_string_literal(arg, "src"):
            return True
    for keyword in call.keywords:
        if keyword.arg in {None, "local_path", "remote_path"} and _contains_string_literal(
            keyword.value, "src"
        ):
            return True
    return False


def _env_sets_pythonpath(method: str, call: ast.Call) -> bool:
    if method != "env":
        return False
    for arg in call.args:
        if isinstance(arg, ast.Dict):
            for key in arg.keys:
                if isinstance(key, ast.Constant) and key.value == "PYTHONPATH":
                    return True
    return any(keyword.arg == "PYTHONPATH" for keyword in call.keywords)


def _image_chain_violations(path: Path, root: Path) -> list[ModalImageOrderViolation]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        rel = path.relative_to(root).as_posix()
        return [
            ModalImageOrderViolation(
                path=rel,
                line=getattr(exc, "lineno", 1) or 1,
                method="parse",
                message=f"could not parse Python file while auditing Modal image order: {exc}",
            )
        ]

    rel = path.relative_to(root).as_posix()
    violations_by_key: dict[tuple[str, int, str], ModalImageOrderViolation] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _call_chain_nodes(node)
        if not chain or not any(method in LOCAL_MOUNT_METHODS for method, _line, _call in chain):
            continue
        first_local = next(
            index
            for index, (method, _line, _call) in enumerate(chain)
            if method in LOCAL_MOUNT_METHODS
        )
        for method, line, _call in chain[first_local + 1:]:
            if method in LOCAL_MOUNT_METHODS:
                continue
            violation = ModalImageOrderViolation(
                path=rel,
                line=line,
                method=method,
                message=(
                    f"{method}() appears after add_local_*() in a Modal image chain; "
                    "move build/env steps before local mounts or use copy=True for "
                    "the mounted file if a later build step is intentionally required"
                )
            )
            violations_by_key[(rel, line, method)] = violation

        src_mount_indexes = [
            index for index, (method, _line, call) in enumerate(chain) if _mounts_repo_src(method, call)
        ]
        if src_mount_indexes:
            first_src_mount = src_mount_indexes[0]
            has_pythonpath = any(
                _env_sets_pythonpath(method, call)
                for method, _line, call in chain[:first_src_mount]
            )
            if not has_pythonpath:
                _method, line, _call = chain[first_src_mount]
                violation = ModalImageOrderViolation(
                    path=rel,
                    line=line,
                    method="PYTHONPATH",
                    message=(
                        "Modal image mounts repo src without setting PYTHONPATH before "
                        "the local-mount phase; worker import of top-level tac modules "
                        "can fail before entrypoint code runs"
                    ),
                )
                violations_by_key[(rel, line, "PYTHONPATH")] = violation
    return list(violations_by_key.values())


def audit_modal_image_build_order(root: Path) -> dict[str, object]:
    violations: list[ModalImageOrderViolation] = []
    scanned = 0
    for path in tracked_python_paths(root):
        scanned += 1
        violations.extend(_image_chain_violations(path, root))
    return {
        "schema": "pact.modal_image_build_order_audit.v1",
        "scanned_python_files": scanned,
        "violation_count": len(violations),
        "violations": [asdict(v) for v in violations],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    payload = audit_modal_image_build_order(root)
    if args.format == "json":
        print(json_text(payload), end="")
    elif payload["violation_count"]:
        print(
            "modal image build order: FAIL "
            f"({payload['violation_count']} violation(s) across "
            f"{payload['scanned_python_files']} tracked Python files)"
        )
        for row in payload["violations"][:30]:
            print(f"  - {row['path']}:{row['line']} {row['message']}")
        remaining = int(payload["violation_count"]) - 30
        if remaining > 0:
            print(f"  - ... {remaining} more")
    else:
        print(
            "modal image build order: PASS "
            f"({payload['scanned_python_files']} tracked Python files scanned)"
        )

    if args.strict and payload["violation_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
