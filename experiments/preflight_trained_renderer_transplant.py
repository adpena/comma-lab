#!/usr/bin/env python3
"""Fail-closed preflight for trained renderer Block-FP transplants.

This tool is local infrastructure for the C067 trained/self-compressed
JointFrameGenerator lane.  It replaces the logical ``renderer.bin`` from a
C067-compatible source archive with a trained renderer export, repacks that
state into deterministic QBF1 Block-FP candidates, verifies the contest runtime
unpack/load path, and emits JSON with the exact Lightning command shapes needed
after a dispatch claim.

It never dispatches remote work and never makes a score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

BUILDER_PATH = REPO_ROOT / "experiments" / "build_blockfp_c067_archive.py"
SCHEMA = "trained_renderer_blockfp_transplant_preflight_v1"
POSE_SAFETY_SCHEMA = "renderer_transplant_pose_safety_preflight_v1"
LANE_ID = "c067_trained_renderer_self_compression_blockfp"
DEFAULT_DIAGNOSTIC_MACHINE = "g7e.4xlarge"
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip"
)
DEFAULT_BLOCK_SIZES = (64, 128, 256, 512, 1024)
RENDERER_MEMBER = "renderer.bin"
MASK_MEMBER = "masks.mkv"
POSE_MEMBER = "optimized_poses.bin"
ORIGINAL_VIDEO_BYTES = 37_545_489


def _load_builder_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_trained_renderer_transplant_blockfp_builder",
        BUILDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load builder module: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BLOCKFP_BUILDER = _load_builder_module()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_pose_safety_reports(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            payload = json.loads(resolved.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid pose-safety JSON: {resolved}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"pose-safety JSON must contain an object: {resolved}")
        payload = dict(payload)
        payload["_pose_safety_json_path"] = str(resolved)
        reports.append(payload)
    return reports


def _pose_safety_gate(
    *,
    reports: list[dict[str, Any]],
    source_archive_sha256: str,
    candidate_archive_sha256: str,
    non_surrogate_export: bool,
) -> dict[str, Any]:
    """Return the fail-closed renderer transplant dispatch gate for one archive."""

    if not non_surrogate_export:
        return {
            "required": False,
            "status": "not_applicable_source_renderer_surrogate",
            "safe_for_exact_eval_dispatch": False,
            "matching_report_path": None,
            "blockers": ["source renderer surrogate is never dispatchable"],
        }
    matching: list[dict[str, Any]] = []
    for report in reports:
        source = report.get("source_archive") or {}
        candidate = report.get("candidate_archive") or {}
        if (
            source.get("sha256") == source_archive_sha256
            and candidate.get("sha256") == candidate_archive_sha256
        ):
            matching.append(report)
    if not matching:
        return {
            "required": True,
            "status": "missing_pose_safety_report",
            "safe_for_exact_eval_dispatch": False,
            "matching_report_path": None,
            "blockers": [
                "missing renderer transplant pose-safety preflight for exact source/candidate archive SHA pair"
            ],
        }
    report = sorted(
        matching,
        key=lambda item: str(item.get("_pose_safety_json_path") or ""),
    )[-1]
    blockers: list[str] = []
    if report.get("schema") != POSE_SAFETY_SCHEMA:
        blockers.append("pose-safety report has unexpected schema")
    if report.get("score_claim") is not False:
        blockers.append("pose-safety report must be no-score evidence")
    if report.get("promotion_eligible") is not False:
        blockers.append("pose-safety report must not claim promotion eligibility")
    if report.get("remote_gpu_dispatch_performed") is not False:
        blockers.append("pose-safety report must be local-only")
    if report.get("safe_for_exact_eval_dispatch") is not True:
        blockers.extend(report.get("fail_closed_reasons") or ["pose-safety report failed closed"])
    safe = not blockers
    return {
        "required": True,
        "status": "pass" if safe else "failed",
        "safe_for_exact_eval_dispatch": safe,
        "matching_report_path": report.get("_pose_safety_json_path"),
        "failure_class": report.get("failure_class"),
        "fail_closed_reasons": report.get("fail_closed_reasons") or [],
        "blockers": sorted(set(str(item) for item in blockers if item)),
    }


def parse_block_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated QBF1 block-size list."""

    return BLOCKFP_BUILDER.parse_block_sizes(value)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _decode_renderer_export(renderer_bytes: bytes) -> tuple[str, dict[str, Any]]:
    magic = renderer_bytes[:4]
    if magic == b"QZS3":
        from tac.quantizr_qzs3_codec import decode_qzs3_state_dict

        return "QZS3", decode_qzs3_state_dict(renderer_bytes, device="cpu")
    if magic == b"MQZ1":
        from tac.quantizr_qzs3_codec import decode_mixed_qzs_block_state_dict

        return "MQZ1", decode_mixed_qzs_block_state_dict(renderer_bytes, device="cpu")
    if magic == b"QBF1":
        from tac.qbf1_renderer_codec import decode_qbf1_state_dict

        return "QBF1", decode_qbf1_state_dict(renderer_bytes, device="cpu")
    raise ValueError(
        "trained renderer transplant preflight supports only pickle-free "
        f"JointFrameGenerator exports with QZS3, MQZ1, or QBF1 magic; got {magic!r}"
    )


def _validate_joint_frame_generator_state(state: dict[str, Any]) -> dict[str, Any]:
    import torch
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

    template = build_quantizr_faithful_renderer().state_dict()
    missing = [name for name in template if name not in state]
    extra = [name for name in state if name not in template]
    shape_mismatches: list[dict[str, Any]] = []
    nonfinite: list[str] = []
    for name, expected in template.items():
        actual = state.get(name)
        if actual is None:
            continue
        if tuple(actual.shape) != tuple(expected.shape):
            shape_mismatches.append(
                {
                    "name": name,
                    "expected": list(expected.shape),
                    "actual": list(actual.shape),
                }
            )
            continue
        if torch.is_floating_point(actual) and not bool(torch.isfinite(actual).all().item()):
            nonfinite.append(name)

    ok = not missing and not extra and not shape_mismatches and not nonfinite
    if not ok:
        raise ValueError(
            "trained renderer export is not JointFrameGenerator-compatible: "
            f"missing={missing[:5]} extra={extra[:5]} "
            f"shape_mismatches={shape_mismatches[:3]} nonfinite={nonfinite[:5]}"
        )
    return {
        "state_tensor_count": len(state),
        "template_tensor_count": len(template),
        "missing": missing,
        "extra": extra,
        "shape_mismatches": shape_mismatches,
        "nonfinite_tensors": nonfinite,
        "compatible": True,
    }


def _read_renderer_for_preflight(
    *,
    source_runtime_members: dict[str, bytes],
    renderer_export: Path | None,
    allow_source_renderer_surrogate: bool,
) -> tuple[bytes, dict[str, Any]]:
    source_renderer = source_runtime_members[RENDERER_MEMBER]
    source_sha = _sha256_bytes(source_renderer)
    if renderer_export is None:
        if not allow_source_renderer_surrogate:
            raise ValueError(
                "--renderer-export is required unless "
                "--allow-source-renderer-surrogate is set for local plumbing tests"
            )
        return source_renderer, {
            "mode": "source_renderer_surrogate",
            "path": None,
            "bytes": len(source_renderer),
            "sha256": source_sha,
            "same_as_source_renderer": True,
            "dispatchable_trained_export": False,
            "reason": "surrogate proves archive/runtime plumbing only; it is not a trained transplant",
        }

    raw = renderer_export.resolve().read_bytes()
    if not raw:
        raise ValueError(f"trained renderer export is empty: {renderer_export}")
    return raw, {
        "mode": "trained_renderer_export",
        "path": str(renderer_export.resolve()),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "same_as_source_renderer": _sha256_bytes(raw) == source_sha,
        "dispatchable_trained_export": _sha256_bytes(raw) != source_sha,
        "reason": None,
    }


def _lightning_commands(
    *,
    candidate_id: str,
    archive_path: Path,
    archive_bytes: int,
    archive_sha256: str,
    renderer_sha256: str,
    source_archive_sha256: str,
) -> dict[str, Any]:
    job_name = f"exact_eval_{candidate_id}_fastdiag_20260502T_READY"
    archive_rel = _repo_rel(archive_path)
    claim = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        LANE_ID,
        "--platform",
        "lightning",
        "--instance-job-id",
        job_name,
        "--agent",
        "codex:gpt-5",
        "--predicted-eta-utc",
        "2026-05-02T23:59Z",
        "--status",
        "eval",
        "--notes",
        f"fast_gpu_diagnostic candidate={candidate_id} archive_sha256={archive_sha256}",
    ]
    exact_eval_base = [
        ".venv/bin/python",
        "scripts/launch_lightning_batch_job.py",
        "exact-eval",
        "--job-name",
        job_name,
        "--archive",
        archive_rel,
        "--repo-dir",
        "/teamspace/studios/this_studio/pact",
        "--upstream-dir",
        "/teamspace/studios/this_studio/pact/upstream",
        "--machine",
        DEFAULT_DIAGNOSTIC_MACHINE,
        "--adjudicate",
        "--baseline-score",
        "0.31561703078448233",
        "--baseline-archive-bytes",
        "276214",
        "--predicted-band",
        "0.0",
        "10.0",
        "--regression-threshold",
        "10.0",
        "--infer-expected-archive",
        "--dispatch-lane-id",
        LANE_ID,
        "--queue-metadata",
        f"lane_id={LANE_ID}",
        "--queue-metadata",
        f"candidate_id={candidate_id}",
        "--queue-metadata",
        f"source_archive_sha256={source_archive_sha256}",
        "--queue-metadata",
        f"trained_renderer_sha256={renderer_sha256}",
        "--queue-metadata",
        "purpose=trained_renderer_blockfp_fast_gpu_diagnostic",
        "--component-trace",
        "--component-trace-top-k",
        "80",
        "--max-sane-score",
        "10.0",
    ]
    dry_run = [*exact_eval_base, "--dry-run"]
    submit_shape = [
        *exact_eval_base,
        "--studio",
        "${LIGHTNING_STUDIO}",
        "--source-manifest",
        "${LIGHTNING_SOURCE_MANIFEST_JSON}",
        "--remote-preflight-ssh-target",
        "${LIGHTNING_PREFLIGHT_SSH_TARGET}",
    ]
    return {
        "job_name": job_name,
        "lane_id": LANE_ID,
        "archive_path": archive_rel,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "claim_command": claim,
        "lightning_exact_eval_dry_run_command": dry_run,
        "lightning_exact_eval_submit_command_shape": submit_shape,
        "remote_gpu_dispatch_performed": False,
    }


def build_preflight(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    output_dir: Path,
    renderer_export: Path | None = None,
    block_sizes: tuple[int, ...] = DEFAULT_BLOCK_SIZES,
    brotli_quality: int = 11,
    payload_member_name: str = BLOCKFP_BUILDER.PACKER.SHORT_PAYLOAD_MEMBER_NAME,
    payload_format: str = BLOCKFP_BUILDER.PACKER.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    allow_source_renderer_surrogate: bool = False,
    pose_safety_json: tuple[Path, ...] = (),
    force: bool = False,
) -> dict[str, Any]:
    """Build deterministic local transplant candidates and readiness JSON."""

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is non-empty; pass --force to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    block_sizes = parse_block_sizes(",".join(str(item) for item in block_sizes))

    source_archive = source_archive.resolve()
    source_bytes = source_archive.read_bytes()
    source_sha = _sha256_bytes(source_bytes)
    runtime_members, source_packaging = BLOCKFP_BUILDER.extract_runtime_members(source_archive)
    required_missing = [
        name for name in (RENDERER_MEMBER, MASK_MEMBER, POSE_MEMBER) if name not in runtime_members
    ]
    if required_missing:
        raise ValueError(f"source archive missing required logical members: {required_missing}")

    renderer_bytes, renderer_export_meta = _read_renderer_for_preflight(
        source_runtime_members=runtime_members,
        renderer_export=renderer_export,
        allow_source_renderer_surrogate=allow_source_renderer_surrogate,
    )
    pose_safety_reports = _read_pose_safety_reports(pose_safety_json)
    non_surrogate_export = bool(
        renderer_export_meta["dispatchable_trained_export"]
        and not renderer_export_meta["same_as_source_renderer"]
    )
    export_format, state = _decode_renderer_export(renderer_bytes)
    state_contract = _validate_joint_frame_generator_state(state)

    import brotli
    from tac.qbf1_renderer_codec import load_qbf1, pack_qbf1_state_dict, qbf1_byte_accounting

    candidates: list[dict[str, Any]] = []
    for block_size in block_sizes:
        transformed_renderer = pack_qbf1_state_dict(state, block_size=block_size)
        accounting = qbf1_byte_accounting(transformed_renderer)
        load_qbf1(transformed_renderer, device="cpu")
        candidate_members = dict(runtime_members)
        candidate_members[RENDERER_MEMBER] = transformed_renderer
        ordered = BLOCKFP_BUILDER._ordered_three_member_payload(candidate_members)
        payload, payload_header = BLOCKFP_BUILDER._build_payload(
            ordered,
            source_archive_sha256=source_sha,
            payload_format=payload_format,
        )
        compressed = brotli.compress(payload, quality=brotli_quality, lgwin=24)
        if brotli.decompress(compressed) != payload:
            raise RuntimeError("Brotli round-trip mismatch for transplant payload")

        candidate_id = f"trained_qbf1_b{block_size:04d}"
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        archive_path = candidate_dir / "archive.zip"
        BLOCKFP_BUILDER.PACKER.write_deterministic_payload_archive(
            archive_path,
            compressed,
            payload_member_name=payload_member_name,
        )
        runtime_unpack_check = BLOCKFP_BUILDER._verify_output_archive(
            archive_path,
            payload_member_name=payload_member_name,
            expected_renderer=transformed_renderer,
        )
        archive_sha = _sha256_file(archive_path)
        archive_bytes = archive_path.stat().st_size
        pose_safety_gate = _pose_safety_gate(
            reports=pose_safety_reports,
            source_archive_sha256=source_sha,
            candidate_archive_sha256=archive_sha,
            non_surrogate_export=non_surrogate_export,
        )
        manifest = {
            "schema": "trained_renderer_blockfp_candidate_manifest_v1",
            "candidate_id": candidate_id,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
            "source_archive": {
                "path": str(source_archive),
                "bytes": len(source_bytes),
                "sha256": source_sha,
                **source_packaging,
            },
            "renderer_export": renderer_export_meta,
            "renderer_export_format": export_format,
            "state_contract": state_contract,
            "transformed_renderer_payload": {
                "member_name": RENDERER_MEMBER,
                "wire_format": "QBF1",
                "block_size": block_size,
                "bytes": len(transformed_renderer),
                "sha256": _sha256_bytes(transformed_renderer),
                "byte_accounting": {
                    "header_nbytes": accounting.header_nbytes,
                    "metadata_nbytes": accounting.metadata_nbytes,
                    "payload_nbytes": accounting.payload_nbytes,
                    "tensor_payload_nbytes": accounting.tensor_payload_nbytes,
                    "total_nbytes": accounting.total_nbytes,
                },
            },
            "output_archive": {
                "path": str(archive_path),
                "bytes": archive_bytes,
                "sha256": archive_sha,
                "delta_bytes_vs_source_archive": archive_bytes - len(source_bytes),
                "formula_only_rate_delta_vs_source_archive": (
                    25.0 * (archive_bytes - len(source_bytes)) / ORIGINAL_VIDEO_BYTES
                ),
            },
            "packed_payload": {
                "payload_format": payload_format,
                "payload_member": payload_member_name,
                "payload_raw_bytes": len(payload),
                "payload_compressed_bytes": len(compressed),
                "brotli_quality": brotli_quality,
                "header": payload_header,
                **runtime_unpack_check,
            },
            "runtime_contract": {
                "byte_closed": True,
                "runtime_unpack_verified": True,
                "qbf1_runtime_load_verified_cpu": True,
                "inflate_loader": (
                    "submissions/robust_current/inflate_renderer.py::_load_renderer "
                    "QBF1 branch"
                ),
                "scorer_imports_at_inflate_time": False,
                "sidecars_required": False,
                "score_affecting_payload_charged_in_archive": True,
                "diagnostic_exact_eval_machine": DEFAULT_DIAGNOSTIC_MACHINE,
                "canonical_score_source_required": (
                    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                    "experiments/contest_auth_eval.py --device cuda"
                ),
            },
            "pose_safety_gate": pose_safety_gate,
        }
        (candidate_dir / "preflight_manifest.json").write_bytes(_json_bytes(manifest))
        commands = _lightning_commands(
            candidate_id=candidate_id,
            archive_path=archive_path,
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha,
            renderer_sha256=str(renderer_export_meta["sha256"]),
            source_archive_sha256=source_sha,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "archive_path": str(archive_path),
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha,
                "manifest_path": str(candidate_dir / "preflight_manifest.json"),
                "block_size": block_size,
                "renderer_bytes": len(transformed_renderer),
                "renderer_sha256": _sha256_bytes(transformed_renderer),
                "delta_bytes_vs_source_archive": archive_bytes - len(source_bytes),
                "runtime_compatible": True,
                "byte_closed": True,
                "pose_safety_gate": pose_safety_gate,
                "score_claim": False,
                "promotion_eligible": False,
                "commands": commands,
            }
        )

    best = min(candidates, key=lambda item: (item["archive_bytes"], item["block_size"]))
    dispatchable_candidates = [
        item
        for item in candidates
        if item["pose_safety_gate"].get("safe_for_exact_eval_dispatch") is True
    ]
    best_dispatchable = (
        min(dispatchable_candidates, key=lambda item: (item["archive_bytes"], item["block_size"]))
        if dispatchable_candidates
        else None
    )
    h100_ready = bool(best_dispatchable)
    if h100_ready:
        readiness_reason = (
            "trained renderer export is byte-closed, QBF1-loadable, materially "
            "differs from the source renderer, and passed renderer pose-safety preflight"
        )
    elif not non_surrogate_export:
        readiness_reason = (
            "source-renderer surrogate or no-op export only proves local "
            "plumbing; wait for a trained renderer export before exact-eval spend"
        )
    elif not pose_safety_reports:
        readiness_reason = (
            "trained renderer export is byte-closed, but exact-eval dispatch is "
            "blocked until renderer pose-safety preflight is supplied for the "
            "exact source/candidate archive SHA pair"
        )
    else:
        readiness_reason = (
            "trained renderer export is byte-closed, but renderer pose-safety "
            "preflight failed for every candidate archive"
    )
    summary = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": {
            "path": str(source_archive),
            "bytes": len(source_bytes),
            "sha256": source_sha,
        },
        "renderer_export": renderer_export_meta,
        "renderer_export_format": export_format,
        "state_contract": state_contract,
        "payload_format": payload_format,
        "payload_member_name": payload_member_name,
        "pose_safety_reports": [str(path.resolve()) for path in pose_safety_json],
        "pose_safety_preflight_required_for_exact_eval": True,
        "block_sizes": list(block_sizes),
        "candidate_count": len(candidates),
        "best_by_archive_bytes": {
            "candidate_id": best["candidate_id"],
            "archive_path": best["archive_path"],
            "archive_bytes": best["archive_bytes"],
            "archive_sha256": best["archive_sha256"],
            "manifest_path": best["manifest_path"],
            "delta_bytes_vs_source_archive": best["delta_bytes_vs_source_archive"],
            "pose_safety_gate": best["pose_safety_gate"],
            "commands": best["commands"],
        },
        "best_dispatchable_after_pose_safety": (
            {
                "candidate_id": best_dispatchable["candidate_id"],
                "archive_path": best_dispatchable["archive_path"],
                "archive_bytes": best_dispatchable["archive_bytes"],
                "archive_sha256": best_dispatchable["archive_sha256"],
                "manifest_path": best_dispatchable["manifest_path"],
                "delta_bytes_vs_source_archive": best_dispatchable["delta_bytes_vs_source_archive"],
                "pose_safety_gate": best_dispatchable["pose_safety_gate"],
                "commands": best_dispatchable["commands"],
            }
            if best_dispatchable is not None
            else None
        ),
        "candidates": candidates,
        "h100_lightning_readiness": {
            "ready": h100_ready,
            "reason": readiness_reason,
            "pose_safety_required": True,
            "selected_pose_safety_gate": (
                best_dispatchable["pose_safety_gate"]
                if best_dispatchable is not None
                else None
            ),
            "dispatch_lane_id": LANE_ID,
            "next_commands_if_ready": best_dispatchable["commands"] if best_dispatchable is not None else None,
        },
    }
    (output_dir / "trained_renderer_blockfp_preflight.json").write_bytes(
        _json_bytes(summary)
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--renderer-export", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-sizes", type=parse_block_sizes, default=DEFAULT_BLOCK_SIZES)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument(
        "--payload-member-name",
        choices=BLOCKFP_BUILDER.PACKER.ALLOWED_PAYLOAD_MEMBER_NAMES,
        default=BLOCKFP_BUILDER.PACKER.SHORT_PAYLOAD_MEMBER_NAME,
    )
    parser.add_argument(
        "--payload-format",
        choices=BLOCKFP_BUILDER.PACKER.ALLOWED_PAYLOAD_FORMATS,
        default=BLOCKFP_BUILDER.PACKER.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    )
    parser.add_argument(
        "--allow-source-renderer-surrogate",
        action="store_true",
        help="Use the source renderer as a no-op local plumbing surrogate.",
    )
    parser.add_argument(
        "--pose-safety-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Renderer transplant pose-safety preflight JSON for an exact "
            "source/candidate archive SHA pair. Required before this tool emits "
            "dispatchable Lightning exact-eval commands for non-surrogate exports."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_preflight(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        renderer_export=args.renderer_export,
        block_sizes=args.block_sizes,
        brotli_quality=args.brotli_quality,
        payload_member_name=args.payload_member_name,
        payload_format=args.payload_format,
        allow_source_renderer_surrogate=args.allow_source_renderer_surrogate,
        pose_safety_json=tuple(args.pose_safety_json),
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
