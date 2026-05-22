#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize an MLX scorer-input cache from a recovered auth-eval artifact.

The tool re-runs only the submission inflate step locally, verifies the
inflated raw surface against the recovered auth-eval manifest, and then writes a
full NumPy scorer-input tensor cache. It does not score anything and never
creates score authority.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.contest_auth_eval import (  # noqa: E402  # pyright: ignore[reportPrivateUsage]
    _ensure_uv_available,
    _extract_archive,
    _record_inflated_output_artifacts,
    _run_inflate,
    _runtime_dependency_manifest,
    _sha256,
    _validate_archive_members,
)
from tac.local_acceleration.mlx_cache_audit import (  # noqa: E402
    audit_mlx_scorer_input_cache_against_auth_eval,
    write_cache_audit,
)
from tac.local_acceleration.mlx_preprocess import write_scorer_input_cache_from_raw_file  # noqa: E402
from tac.repo_io import read_json, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-eval-dir", required=True, type=Path)
    parser.add_argument("--output-cache-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--upstream-dir", default=Path("upstream"), type=Path)
    parser.add_argument(
        "--video-names-file",
        default=None,
        type=Path,
        help="Defaults to <upstream-dir>/public_test_video_names.txt.",
    )
    parser.add_argument("--archive", default=None, type=Path)
    parser.add_argument("--submission-dir", default=None, type=Path)
    parser.add_argument("--inflate-sh", default=None)
    parser.add_argument("--inflate-timeout", default=1800, type=int)
    parser.add_argument("--batch-pairs", default=8, type=int)
    parser.add_argument(
        "--large-cache-pair-threshold",
        default=64,
        type=int,
        help="Refuse full tensor caches above this pair count unless acknowledged.",
    )
    parser.add_argument("--allow-large-tensor-cache", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.batch_pairs < 1:
        raise SystemExit("--batch-pairs must be >= 1")
    if args.inflate_timeout < 1:
        raise SystemExit("--inflate-timeout must be >= 1")
    if args.large_cache_pair_threshold < 1:
        raise SystemExit("--large-cache-pair-threshold must be >= 1")

    auth_dir = args.auth_eval_dir.resolve()
    auth_eval_path = auth_dir / "contest_auth_eval.json"
    auth_eval = _read_object(auth_eval_path)
    recovered_manifest = _read_object(auth_dir / "inflated_outputs_manifest.json")
    local_request = _load_local_request(auth_dir)

    archive = (args.archive or Path(str(local_request.get("archive_path") or ""))).expanduser()
    submission_dir = (
        args.submission_dir
        or Path(str(local_request.get("submission_dir") or "")).expanduser()
    )
    inflate_rel = args.inflate_sh or str(local_request.get("inflate_sh") or "inflate.sh")
    if not archive.is_absolute():
        archive = archive.resolve()
    if not submission_dir.is_absolute():
        submission_dir = submission_dir.resolve()
    inflate_sh = (submission_dir / inflate_rel).resolve()
    upstream_dir = args.upstream_dir.resolve()
    video_names_file = (
        args.video_names_file.resolve()
        if args.video_names_file is not None
        else (upstream_dir / "public_test_video_names.txt").resolve()
    )

    _require_file(archive, "archive")
    _require_file(inflate_sh, "inflate_sh")
    _require_file(video_names_file, "video_names_file")
    _require_file(auth_eval_path, "auth_eval")

    output_cache = args.output_cache_dir.resolve()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        if not args.force:
            raise SystemExit(f"--work-dir already exists; pass --force to replace: {work_dir}")
        shutil.rmtree(work_dir)
    if output_cache.exists():
        if not args.force:
            raise SystemExit(
                f"--output-cache-dir already exists; pass --force to replace: {output_cache}"
            )
        shutil.rmtree(output_cache)

    archive_sha = _sha256(archive, prefix=0)
    expected_archive = _auth_archive_sha256(auth_eval)
    if archive_sha != expected_archive:
        raise SystemExit(
            "archive SHA mismatch against auth eval: "
            f"archive={archive_sha} auth={expected_archive}"
        )
    expected_archive_size = _auth_archive_size_bytes(auth_eval)
    if expected_archive_size is not None and archive.stat().st_size != expected_archive_size:
        raise SystemExit(
            "archive byte count mismatch against auth eval: "
            f"archive={archive.stat().st_size} auth={expected_archive_size}"
        )
    inflate_policy = _validate_default_inflate_policy(auth_eval)
    runtime_custody = _validate_runtime_custody(
        auth_eval,
        local_request,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
    )
    _ensure_uv_available()

    extracted_dir = work_dir / "archive"
    inflated_dir = work_dir / "inflated"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_inflate_env = _local_inflate_env_bridge(work_dir)
    members = _extract_archive(archive, extracted_dir)
    _validate_archive_members(members)
    inflate_elapsed = _run_inflate(
        inflate_sh,
        extracted_dir,
        inflated_dir,
        video_names_file,
        timeout=int(args.inflate_timeout),
        extra_env=local_inflate_env,
    )

    provenance: dict[str, Any] = {
        "schema_version": "mlx_scorer_cache_materialize_from_auth_eval.v1",
        "tool": "tools/materialize_mlx_scorer_cache_from_auth_eval.py",
        "auth_eval_dir": str(auth_dir),
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive.stat().st_size,
        "inflate_script": str(inflate_sh),
        "inflate_script_sha256": _sha256(inflate_sh, prefix=0),
        "inflate_elapsed_seconds": inflate_elapsed,
        "inflate_policy_replay": inflate_policy,
        "local_inflate_env_bridge": {
            "python": local_inflate_env.get("PYTHON"),
            "python_bin": local_inflate_env.get("PYTHON_BIN"),
            "path_prefix": str(work_dir / "bin"),
            "score_claim": False,
            "promotion_eligible": False,
        },
        "runtime_custody": runtime_custody,
    }
    write_json(work_dir / "provenance.json", provenance)
    inflated_manifest = _record_inflated_output_artifacts(
        provenance,
        work_dir,
        inflated_dir,
        video_names_file,
    )
    expected_inflated = _manifest_aggregate_sha256(recovered_manifest)
    actual_inflated = _manifest_aggregate_sha256(inflated_manifest)
    if actual_inflated != expected_inflated:
        raise SystemExit(
            "inflated output aggregate SHA mismatch against recovered auth eval: "
            f"local={actual_inflated} auth={expected_inflated}"
        )

    raw_path = _single_raw_path(inflated_manifest, inflated_dir)
    pair_count = _pair_count_from_video_names(video_names_file)
    if pair_count > args.large_cache_pair_threshold and not args.allow_large_tensor_cache:
        raise SystemExit(
            "refusing full tensor cache for "
            f"{pair_count} pairs (> threshold {args.large_cache_pair_threshold}); "
            "pass --allow-large-tensor-cache explicitly"
        )
    cache_manifest = write_scorer_input_cache_from_raw_file(
        raw_path,
        output_cache,
        archive_sha256=archive_sha,
        inflated_outputs_aggregate_sha256=actual_inflated,
        batch_pairs=int(args.batch_pairs),
    )
    audit = audit_mlx_scorer_input_cache_against_auth_eval(cache_manifest, auth_eval)
    write_cache_audit(audit, args.audit_output)
    cache_manifest = _finalize_cache_after_audit(
        output_cache=output_cache,
        audit_output=args.audit_output.resolve(),
        audit=audit,
        auth_dir=auth_dir,
        auth_eval_path=auth_eval_path,
    )

    if not args.keep_raw:
        shutil.rmtree(extracted_dir, ignore_errors=True)
        shutil.rmtree(inflated_dir, ignore_errors=True)

    summary = {
        "cache_manifest": str(output_cache / "manifest.json") if audit["passed"] else None,
        "audit_output": str(args.audit_output),
        "audit_passed": audit["passed"],
        "audit_verdict": audit["verdict"],
        "archive_sha256": archive_sha,
        "inflated_outputs_aggregate_sha256": actual_inflated,
        "pair_count": cache_manifest.get("pair_count", pair_count),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if audit["passed"] else 2


def _load_local_request(auth_dir: Path) -> dict[str, Any]:
    for path in sorted(auth_dir.glob("modal_*_auth_eval_local_request.json")):
        payload = read_json(path)
        if isinstance(payload, dict):
            return payload
    raise SystemExit(f"no modal_*_auth_eval_local_request.json found in {auth_dir}")


def _read_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def _auth_archive_sha256(payload: dict[str, Any]) -> str | None:
    value = payload.get("archive_sha256")
    if isinstance(value, str) and value:
        return value
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("archive_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _auth_archive_size_bytes(payload: dict[str, Any]) -> int | None:
    value = payload.get("archive_size_bytes")
    if isinstance(value, int):
        return value
    provenance = payload.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("archive_size_bytes"), int):
        return provenance.get("archive_size_bytes")
    return None


def _auth_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    return provenance if isinstance(provenance, dict) else {}


def _validate_default_inflate_policy(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = _auth_provenance(payload)
    policy = str(provenance.get("inflate_device_policy") or "auto").strip().lower()
    overrides = provenance.get("inflate_env_overrides")
    if policy != "auto":
        raise SystemExit(
            "auth eval used non-default inflate_device_policy; local cache materialization "
            f"currently refuses policy replay: {policy}"
        )
    if isinstance(overrides, dict) and overrides:
        raise SystemExit(
            "auth eval used inflate_env_overrides; local cache materialization refuses "
            "non-default inflate env replay"
        )
    return {
        "inflate_device_policy": policy,
        "inflate_env_overrides_present": False,
        "policy_replay_mode": "default_auto_only",
    }


def _validate_runtime_custody(
    auth_eval: dict[str, Any],
    local_request: dict[str, Any],
    *,
    inflate_sh: Path,
    upstream_dir: Path,
) -> dict[str, Any]:
    provenance = _auth_provenance(auth_eval)
    local_inflate_sha = _sha256(inflate_sh, prefix=0)
    expected_inflate_sha = _string(provenance.get("inflate_script_sha256"))
    if expected_inflate_sha and local_inflate_sha != expected_inflate_sha:
        raise SystemExit(
            "inflate script SHA mismatch against recovered auth eval: "
            f"local={local_inflate_sha} auth={expected_inflate_sha}"
        )

    local_manifest = _runtime_dependency_manifest(inflate_sh, upstream_dir)
    auth_manifest = provenance.get("inflate_runtime_manifest")
    if not isinstance(auth_manifest, dict):
        raise SystemExit("auth eval missing provenance.inflate_runtime_manifest")

    local_content = _string(local_manifest.get("runtime_content_tree_sha256"))
    auth_content = _string(auth_manifest.get("runtime_content_tree_sha256"))
    local_tree = _string(local_manifest.get("runtime_tree_sha256"))
    auth_tree = _string(auth_manifest.get("runtime_tree_sha256"))
    expected_tree = _string(local_request.get("expected_runtime_tree_sha256"))
    if auth_content:
        if local_content != auth_content:
            raise SystemExit(
                "runtime content-tree SHA mismatch against recovered auth eval: "
                f"local={local_content} auth={auth_content}"
            )
        comparison = "runtime_content_tree_sha256"
    elif expected_tree:
        if local_tree != expected_tree and local_content != expected_tree:
            raise SystemExit(
                "runtime SHA mismatch against local request expected runtime tree: "
                f"local_tree={local_tree} local_content={local_content} expected={expected_tree}"
            )
        comparison = "local_request_expected_runtime_tree_sha256"
    else:
        raise SystemExit(
            "auth eval does not provide runtime_content_tree_sha256 and local request "
            "does not provide expected_runtime_tree_sha256"
        )

    return {
        "comparison": comparison,
        "local_runtime_tree_sha256": local_tree,
        "auth_runtime_tree_sha256": auth_tree,
        "local_runtime_content_tree_sha256": local_content,
        "auth_runtime_content_tree_sha256": auth_content,
        "local_request_expected_runtime_tree_sha256": expected_tree,
        "local_inflate_script_sha256": local_inflate_sha,
        "auth_inflate_script_sha256": expected_inflate_sha,
        "runtime_file_count": local_manifest.get("runtime_file_count"),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _local_inflate_env_bridge(work_dir: Path) -> dict[str, str]:
    """Expose this interpreter as ``python`` for public runtimes that hardcode it."""

    bin_dir = work_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    python_link = bin_dir / "python"
    if python_link.exists() or python_link.is_symlink():
        python_link.unlink()
    python_link.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"exec {json.dumps(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    python_link.chmod(0o755)
    path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return {
        "PATH": path,
        "PYTHON": sys.executable,
        "PYTHON_BIN": sys.executable,
    }


def _finalize_cache_after_audit(
    *,
    output_cache: Path,
    audit_output: Path,
    audit: dict[str, Any],
    auth_dir: Path,
    auth_eval_path: Path,
) -> dict[str, Any]:
    manifest_path = output_cache / "manifest.json"
    if audit.get("passed") is not True:
        shutil.rmtree(output_cache, ignore_errors=True)
        return {}

    manifest = _read_object(manifest_path)
    manifest["eligible_for_local_mlx_transfer_calibration"] = True
    manifest["auth_eval_identity_audit"] = {
        "schema_version": audit.get("schema_version"),
        "path": str(audit_output),
        "sha256": _sha256(audit_output, prefix=0),
        "verdict": audit.get("verdict"),
        "passed": True,
        "identity_residual": audit.get("identity_residual"),
        "auth_eval_dir": str(auth_dir),
        "auth_eval_path": str(auth_eval_path),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def _manifest_aggregate_sha256(payload: dict[str, Any]) -> str | None:
    value = payload.get("aggregate_sha256")
    if isinstance(value, str) and value:
        return value
    nested = payload.get("payload")
    if isinstance(nested, dict):
        value = nested.get("aggregate_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _single_raw_path(manifest: dict[str, Any], inflated_dir: Path) -> Path:
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise SystemExit("expected exactly one inflated raw file in manifest")
    row = files[0]
    if not isinstance(row, dict):
        raise SystemExit("inflated raw manifest row is not an object")
    rel = row.get("relative_path")
    if not isinstance(rel, str) or not rel:
        raise SystemExit("inflated raw manifest missing relative_path")
    path = inflated_dir / rel
    if not path.is_file():
        raise SystemExit(f"inflated raw file not found: {path}")
    return path


def _pair_count_from_video_names(video_names_file: Path) -> int:
    # The current public challenge file is one 1200-frame video. Keep this
    # explicit so accidental multi-video calibration fails before tensor writes.
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    if names != ["0.mkv"]:
        raise SystemExit(f"unsupported video_names_file for MLX cache materialization: {names}")
    return 600


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
