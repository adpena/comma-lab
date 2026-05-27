#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Safely rerun and diff an MLX master-gradient replay bundle.

The rerun path is deliberately side-effect contained: it rewrites the original
``--out`` and replay-bundle paths under an operator-selected output directory,
strips anchor writes, and disables canonical manifest appends by default. The
result is still a macOS-MLX research-signal verifier, never score authority.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.diff_mlx_master_gradient_replay import diff_replay_payloads
except ModuleNotFoundError:  # pragma: no cover
    from diff_mlx_master_gradient_replay import diff_replay_payloads

REPO_ROOT = Path(__file__).resolve().parents[1]
RERUN_SCHEMA = "mlx_master_gradient_replay_rerun.v1"
REPLAY_BUNDLE_SCHEMA = "mlx_master_gradient_replay_bundle.v1"
FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
VALUE_OPTIONS_TO_REWRITE = {
    "--out",
    "--replay-bundle-path",
    "--manifest-jsonl",
}
SIDE_EFFECT_FLAGS_TO_STRIP = {
    "--write-anchor",
    "--no-replay-bundle",
}


class MLXReplayRerunError(ValueError):
    """Raised when a replay bundle cannot be safely rerun."""


def _utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)
    return out.strip("._-").lower() or "mlx_master_gradient_replay"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MLXReplayRerunError(f"{path}: expected JSON object")
    return payload


def _expect_bundle(payload: Mapping[str, Any], *, path: Path) -> None:
    if payload.get("schema") != REPLAY_BUNDLE_SCHEMA:
        raise MLXReplayRerunError(
            f"{path}: schema must be {REPLAY_BUNDLE_SCHEMA}, got {payload.get('schema')!r}"
        )
    if payload.get("tool") != "tools/extract_master_gradient_mlx.py":
        raise MLXReplayRerunError(
            f"{path}: replay tool must be tools/extract_master_gradient_mlx.py"
        )
    gate = payload.get("calibration_gate")
    if not isinstance(gate, Mapping):
        raise MLXReplayRerunError(f"{path}: calibration_gate missing")
    for key, expected in FALSE_AUTHORITY.items():
        if gate.get(key) is not expected:
            raise MLXReplayRerunError(f"{path}: calibration_gate.{key} must be false")


def _strip_side_effect_options(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in VALUE_OPTIONS_TO_REWRITE:
            skip_next = True
            continue
        if any(token.startswith(f"{option}=") for option in VALUE_OPTIONS_TO_REWRITE):
            continue
        if token in SIDE_EFFECT_FLAGS_TO_STRIP:
            continue
        out.append(token)
    return out


def build_rerun_command(
    bundle: Mapping[str, Any],
    *,
    bundle_path: Path,
    output_dir: Path,
    python_executable: str,
    append_manifest: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a side-effect-contained rerun command from a replay bundle."""

    _expect_bundle(bundle, path=bundle_path)
    argv = bundle.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise MLXReplayRerunError(f"{bundle_path}: argv must be a string list")

    archive = bundle.get("archive")
    archive_sha = ""
    if isinstance(archive, Mapping):
        archive_sha = str(archive.get("sha256") or "")[:12]
    resolved_run_id = run_id or f"{bundle_path.stem}_{archive_sha}_{_utc_compact()}"
    run_dir = output_dir / _safe_id(resolved_run_id)
    out_path = run_dir / "gradient.npy"
    rerun_bundle_path = out_path.with_suffix(out_path.suffix + ".replay_bundle.json")
    diff_path = run_dir / "replay_diff.json"
    summary_path = run_dir / "replay_rerun_summary.json"

    rerun_argv = _strip_side_effect_options(argv)
    rerun_argv.extend(
        [
            "--out",
            str(out_path),
            "--replay-bundle-path",
            str(rerun_bundle_path),
        ]
    )
    if not append_manifest:
        rerun_argv.append("--no-manifest")
    command = [
        python_executable,
        str(REPO_ROOT / "tools" / "extract_master_gradient_mlx.py"),
        *rerun_argv,
    ]
    return {
        "schema": "mlx_master_gradient_replay_rerun_command.v1",
        "command": command,
        "run_dir": str(run_dir),
        "out_path": str(out_path),
        "rerun_bundle_path": str(rerun_bundle_path),
        "diff_path": str(diff_path),
        "summary_path": str(summary_path),
        "append_manifest": bool(append_manifest),
        "stripped_side_effects": sorted(SIDE_EFFECT_FLAGS_TO_STRIP),
        "rewritten_options": sorted(VALUE_OPTIONS_TO_REWRITE),
        **FALSE_AUTHORITY,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rerun_and_diff(
    *,
    bundle_path: Path,
    output_dir: Path,
    python_executable: str,
    append_manifest: bool = False,
    run_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    bundle = _load_json(bundle_path)
    command_record = build_rerun_command(
        bundle,
        bundle_path=bundle_path,
        output_dir=output_dir,
        python_executable=python_executable,
        append_manifest=append_manifest,
        run_id=run_id,
    )
    run_dir = Path(command_record["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        command_record["command"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    rerun_bundle_path = Path(command_record["rerun_bundle_path"])
    diff: dict[str, Any] | None = None
    if proc.returncode == 0 and rerun_bundle_path.is_file():
        diff = diff_replay_payloads(bundle, _load_json(rerun_bundle_path))
        _write_json(Path(command_record["diff_path"]), diff)

    summary = {
        "schema": RERUN_SCHEMA,
        "bundle_path": str(bundle_path),
        "command_record": command_record,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "diff": diff,
        "matched": bool(diff and diff.get("matched") is True),
        "side_effect_policy": {
            "canonical_manifest_append_default": bool(append_manifest),
            "anchor_write_stripped": True,
            "canonical_anchor_append_disabled": True,
        },
        **FALSE_AUTHORITY,
    }
    _write_json(Path(command_record["summary_path"]), summary)
    return summary, 0 if proc.returncode == 0 and summary["matched"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable for the rerun command (default: current interpreter).",
    )
    parser.add_argument(
        "--append-manifest",
        action="store_true",
        help=(
            "Allow the rerun extractor to append its MLX research-signal manifest row. "
            "Anchor writes are still stripped."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Stable rerun directory name under --output-dir. Defaults to a "
            "timestamped id when omitted."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the rerun bundle differs from the source bundle.",
    )
    args = parser.parse_args(argv)

    try:
        summary, mismatch_code = rerun_and_diff(
            bundle_path=args.bundle,
            output_dir=args.output_dir,
            python_executable=args.python_executable,
            append_manifest=args.append_manifest,
            run_id=args.run_id,
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        MLXReplayRerunError,
    ) as exc:
        print(f"[mlx-replay-rerun] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return mismatch_code if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
