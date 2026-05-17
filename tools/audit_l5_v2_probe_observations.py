#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit existing artifacts for the L5 v2 C1/Z5/TT5L probe gate.

The command emits a fail-closed intake report and, optionally, the gate artifact
consumed by `l5_v2_dispatch_readiness`. It does not run eval, does not claim a
score, and exits nonzero until the probe verdict allows architecture lock-in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)


def _reexec_repo_venv_if_available(repo_root: Path) -> None:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if os.environ.get("PACT_ALLOW_SYSTEM_PYTHON") == "1" or not venv_python.is_file():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_repo_venv_if_available(REPO_ROOT)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.l5_v2_probe_intake import (  # noqa: E402
    build_l5_v2_probe_observation_intake,
    default_l5_v2_probe_source_paths,
    render_l5_v2_probe_observation_intake_markdown,
)


def _safe_output_path(path: Path) -> Path:
    output = path if path.is_absolute() else REPO_ROOT / path
    resolved = output.resolve(strict=False)
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"refusing output outside repo root: {path}") from exc
    text = str(resolved)
    if text.startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise ValueError(f"refusing output in tmp: {path}")
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-json",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional or replacement source artifact. If omitted, scans the "
            "canonical L5 v2 source list."
        ),
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument(
        "--probe-gate-out",
        default=None,
        type=Path,
        help="Optional path for the wrapped L5 v2 probe gate artifact.",
    )
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    output = _safe_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    output = _safe_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_paths = args.source_json or [
        Path(item) for item in default_l5_v2_probe_source_paths(repo_root=REPO_ROOT)
    ]
    intake = build_l5_v2_probe_observation_intake(source_paths, repo_root=REPO_ROOT)
    _write_json(args.output_json, intake)
    if args.output_md is not None:
        _write_text(args.output_md, render_l5_v2_probe_observation_intake_markdown(intake))
    if args.probe_gate_out is not None:
        _write_json(
            args.probe_gate_out,
            intake["probe_gate_artifact"],
        )
    print(f"wrote {_safe_output_path(args.output_json).relative_to(REPO_ROOT)}")
    if args.output_md is not None:
        print(f"wrote {_safe_output_path(args.output_md).relative_to(REPO_ROOT)}")
    if args.probe_gate_out is not None:
        print(f"wrote {_safe_output_path(args.probe_gate_out).relative_to(REPO_ROOT)}")
    allowed = intake.get("architecture_lock_allowed") is True
    print(f"architecture_lock_allowed={str(allowed).lower()}")
    return 0 if allowed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
