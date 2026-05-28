#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a deterministic replay bundle for MLX-local advisory work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_replay_bundle import (  # noqa: E402
    MlxReplayBundleError,
    build_mlx_local_replay_bundle,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-out", required=True, type=Path)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--axis", default="[macOS-MLX research-signal]")
    parser.add_argument(
        "--command-json",
        required=True,
        help="JSON list of replay argv lists.",
    )
    parser.add_argument("--artifact", action="append", default=[], type=Path)
    parser.add_argument("--input-artifact", action="append", default=[], type=Path)
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _command_lists(raw: str) -> list[list[str]]:
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise MlxReplayBundleError("--command-json must be a non-empty list")
    commands: list[list[str]] = []
    for index, command in enumerate(payload):
        if not isinstance(command, list) or not command:
            raise MlxReplayBundleError(f"--command-json[{index}] must be a non-empty list")
        commands.append([str(item) for item in command])
    return commands


def _metadata(raw: str) -> dict[str, object]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise MlxReplayBundleError("--metadata-json must be a JSON object")
    return dict(payload)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    try:
        bundle = build_mlx_local_replay_bundle(
            repo_root=REPO_ROOT,
            bundle_id=args.bundle_id,
            axis=args.axis,
            commands=_command_lists(args.command_json),
            artifact_paths=args.artifact,
            input_artifact_paths=args.input_artifact,
            metadata=_metadata(args.metadata_json),
            argv=[Path(sys.argv[0]).name, *raw_argv],
        )
        bundle_out = _resolve(args.bundle_out)
        expected_existing_sha256 = (
            sha256_file(bundle_out) if bundle_out.is_file() and args.overwrite else None
        )
        write = write_json_artifact(
            bundle_out,
            bundle,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except (
        ArtifactWriteError,
        OSError,
        MlxReplayBundleError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: MLX replay bundle failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "mlx_local_replay_bundle_cli_result.v1",
                "bundle_out": str(args.bundle_out),
                "bundle_id": args.bundle_id,
                "local_replay_ready": bundle["replay_readiness"]["local_replay_ready"],
                "missing_artifact_count": len(bundle["missing_artifacts"]),
                "bytes_written": write.bytes_written,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
