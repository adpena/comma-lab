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
    AUTHORITY_FALSE_FIELDS,
    audit_mlx_scorer_input_cache_against_auth_eval,
    write_cache_audit,
)
from tac.local_acceleration.mlx_preprocess import write_scorer_input_cache_from_raw_file  # noqa: E402
from tac.repo_io import read_json, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-eval-dir", required=True, type=Path)
    parser.add_argument("--output-cache-dir", required=True, type=Path)
    parser.add_argument("--work-dir", default=None, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument(
        "--downloaded-tensor-cache-dir",
        default=None,
        type=Path,
        help=(
            "No-reinflate mode: directory downloaded from the Modal auth-cache "
            "volume containing manifest.json and scorer input .npy tensors."
        ),
    )
    parser.add_argument(
        "--tensor-volume-manifest",
        default=None,
        type=Path,
        help=(
            "No-reinflate mode: scorer_input_cache_tensor_volume_manifest.json "
            "from the recovered auth-eval artifact."
        ),
    )
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
    if bool(args.downloaded_tensor_cache_dir) != bool(args.tensor_volume_manifest):
        raise SystemExit(
            "--downloaded-tensor-cache-dir and --tensor-volume-manifest must be passed together"
        )
    if args.downloaded_tensor_cache_dir is not None:
        return _main_downloaded_tensor_cache(
            args=args,
            auth_dir=auth_dir,
            auth_eval_path=auth_eval_path,
            auth_eval=auth_eval,
        )
    if args.work_dir is None:
        raise SystemExit("--work-dir is required unless --downloaded-tensor-cache-dir is used")
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


def _main_downloaded_tensor_cache(
    *,
    args: argparse.Namespace,
    auth_dir: Path,
    auth_eval_path: Path,
    auth_eval: dict[str, Any],
) -> int:
    source_cache = args.downloaded_tensor_cache_dir.resolve()
    volume_manifest_path = args.tensor_volume_manifest.resolve()
    output_cache = args.output_cache_dir.resolve()
    audit_output = args.audit_output.resolve()
    if not source_cache.is_dir():
        raise SystemExit(f"downloaded tensor cache dir not found: {source_cache}")
    _require_file(volume_manifest_path, "tensor_volume_manifest")

    validation = _validate_downloaded_tensor_cache(
        downloaded_cache_dir=source_cache,
        tensor_volume_manifest=_read_object(volume_manifest_path),
        auth_eval=auth_eval,
    )
    normalized_manifest = validation["normalized_manifest"]
    audit = audit_mlx_scorer_input_cache_against_auth_eval(normalized_manifest, auth_eval)
    write_cache_audit(audit, audit_output)

    if not audit["passed"]:
        summary = {
            "cache_manifest": None,
            "audit_output": str(audit_output),
            "audit_passed": False,
            "audit_verdict": audit["verdict"],
            "downloaded_tensor_cache_dir": str(source_cache),
            "tensor_volume_manifest": str(volume_manifest_path),
            "validation": _public_validation_summary(validation),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        print(json.dumps(summary, sort_keys=True))
        return 2

    same_output = source_cache == output_cache
    if output_cache.exists() and not same_output:
        if not args.force:
            raise SystemExit(
                f"--output-cache-dir already exists; pass --force to replace: {output_cache}"
            )
        shutil.rmtree(output_cache)
    if not same_output:
        shutil.copytree(source_cache, output_cache)
    write_json(output_cache / "manifest.json", normalized_manifest)
    cache_manifest = _finalize_cache_after_audit(
        output_cache=output_cache,
        audit_output=audit_output,
        audit=audit,
        auth_dir=auth_dir,
        auth_eval_path=auth_eval_path,
        delete_on_fail=False,
    )
    cache_manifest["downloaded_tensor_cache_identity"] = _public_validation_summary(validation)
    write_json(output_cache / "manifest.json", cache_manifest)

    summary = {
        "cache_manifest": str(output_cache / "manifest.json"),
        "audit_output": str(audit_output),
        "audit_passed": True,
        "audit_verdict": audit["verdict"],
        "downloaded_tensor_cache_dir": str(source_cache),
        "tensor_volume_manifest": str(volume_manifest_path),
        "validation": _public_validation_summary(validation),
        "archive_sha256": cache_manifest.get("archive_sha256"),
        "inflated_outputs_aggregate_sha256": cache_manifest.get(
            "inflated_outputs_aggregate_sha256"
        ),
        "pair_count": cache_manifest.get("pair_count"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _load_local_request(auth_dir: Path) -> dict[str, Any]:
    for path in sorted(auth_dir.glob("modal_*_auth_eval_local_request.json")):
        payload = read_json(path)
        if isinstance(payload, dict):
            return payload
    raise SystemExit(f"no modal_*_auth_eval_local_request.json found in {auth_dir}")


def _validate_downloaded_tensor_cache(
    *,
    downloaded_cache_dir: Path,
    tensor_volume_manifest: dict[str, Any],
    auth_eval: dict[str, Any],
) -> dict[str, Any]:
    if tensor_volume_manifest.get("schema_version") != "modal_auth_eval_tensor_volume_manifest.v1":
        raise SystemExit("tensor volume manifest has unexpected schema_version")
    volume_payload = _dict(tensor_volume_manifest.get("payload"), "tensor volume payload")
    manifest_path = downloaded_cache_dir / "manifest.json"
    _require_file(manifest_path, "downloaded tensor cache manifest")
    downloaded_manifest = _read_object(manifest_path)

    manifest_sha_expected = _string(tensor_volume_manifest.get("manifest_sha256"))
    manifest_sha_actual = _sha256(manifest_path, prefix=0)
    pre_stamp_manifest = _manifest_without_local_audit_stamp(downloaded_manifest)
    manifest_sha_exact = bool(manifest_sha_expected and manifest_sha_actual == manifest_sha_expected)
    if not manifest_sha_exact and pre_stamp_manifest != volume_payload:
        raise SystemExit(
            "downloaded tensor cache manifest mismatch against tensor volume manifest "
            f"(local_sha={manifest_sha_actual} expected_sha={manifest_sha_expected})"
        )
    if manifest_sha_exact and downloaded_manifest != volume_payload:
        raise SystemExit(
            "downloaded tensor cache manifest SHA matches but JSON payload differs "
            "from tensor volume manifest payload"
        )

    auth_tensor_record = _auth_tensor_manifest_record(auth_eval)
    if auth_tensor_record:
        auth_payload = _dict(auth_tensor_record.get("payload"), "auth eval tensor payload")
        if auth_payload != volume_payload:
            raise SystemExit("tensor volume manifest payload differs from auth eval provenance")
        auth_manifest_sha = _string(auth_tensor_record.get("sha256"))
        if auth_manifest_sha and manifest_sha_expected and auth_manifest_sha != manifest_sha_expected:
            raise SystemExit(
                "tensor volume manifest SHA differs from auth eval provenance: "
                f"volume={manifest_sha_expected} auth={auth_manifest_sha}"
            )

    artifact_checks = _validate_downloaded_tensor_artifacts(
        downloaded_cache_dir=downloaded_cache_dir,
        manifest_payload=volume_payload,
    )
    normalized_manifest = _complete_false_authority_contract(dict(pre_stamp_manifest))
    return {
        "schema_version": "downloaded_mlx_tensor_cache_identity.v1",
        "downloaded_cache_dir": str(downloaded_cache_dir),
        "manifest_sha256_expected": manifest_sha_expected,
        "manifest_sha256_actual": manifest_sha_actual,
        "manifest_sha256_exact_match": manifest_sha_exact,
        "preexisting_local_audit_stamp": pre_stamp_manifest != downloaded_manifest,
        "authority_contract_completed_fields": [
            field
            for field in AUTHORITY_FALSE_FIELDS
            if pre_stamp_manifest.get(field) is not False
        ],
        "artifact_checks": artifact_checks,
        "normalized_manifest": normalized_manifest,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _validate_downloaded_tensor_artifacts(
    *,
    downloaded_cache_dir: Path,
    manifest_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = _dict(manifest_payload.get("artifacts"), "tensor cache artifacts")
    checks: list[dict[str, Any]] = []
    for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        record = _dict(artifacts.get(key), f"tensor artifact {key}")
        remote_path = _string(record.get("path"))
        if remote_path is None:
            raise SystemExit(f"tensor artifact {key} missing path")
        local_path = downloaded_cache_dir / Path(remote_path).name
        _require_file(local_path, f"downloaded tensor artifact {key}")
        expected_bytes = _int(record.get("bytes"))
        if expected_bytes is None:
            raise SystemExit(f"tensor artifact {key} missing byte count")
        actual_bytes = local_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"tensor artifact byte count mismatch for {key}: "
                f"local={actual_bytes} expected={expected_bytes}"
            )
        expected_sha = _string(record.get("sha256"))
        if expected_sha is None:
            raise SystemExit(f"tensor artifact {key} missing sha256")
        actual_sha = _sha256(local_path, prefix=0)
        if actual_sha != expected_sha:
            raise SystemExit(
                f"tensor artifact SHA mismatch for {key}: "
                f"local={actual_sha} expected={expected_sha}"
            )
        checks.append(
            {
                "name": key,
                "path": str(local_path),
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        )
    return checks


def _manifest_without_local_audit_stamp(manifest: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(manifest)
    stripped.pop("auth_eval_identity_audit", None)
    stripped.pop("eligible_for_local_mlx_transfer_calibration", None)
    stripped.pop("downloaded_tensor_cache_identity", None)
    return stripped


def _complete_false_authority_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    for field in AUTHORITY_FALSE_FIELDS:
        if (
            field in manifest
            and manifest.get(field) is not None
            and manifest.get(field) is not False
        ):
            raise SystemExit(
                "downloaded tensor cache manifest authority field "
                f"{field} must be false"
            )
        manifest[field] = False
    return manifest


def _auth_tensor_manifest_record(auth_eval: dict[str, Any]) -> dict[str, Any]:
    provenance = auth_eval.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    record = provenance.get("scorer_input_cache_tensor_manifest")
    return record if isinstance(record, dict) else {}


def _public_validation_summary(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in validation.items()
        if key not in {"normalized_manifest"}
    }


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
    delete_on_fail: bool = True,
) -> dict[str, Any]:
    manifest_path = output_cache / "manifest.json"
    if audit.get("passed") is not True:
        if delete_on_fail:
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


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _dict(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise SystemExit(f"expected JSON object for {label}")


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
