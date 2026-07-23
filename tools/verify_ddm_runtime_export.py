#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify Build #636 raw identity, decode timing, and advisory scorer row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402, I001
from tac.canonical_equations.ddm_runtime_export_identity_20260723 import (  # noqa: E402
    export_identity,
    score_row,
)
from tac.optimization.ddm_runtime_exporter import (  # noqa: E402
    EVIDENCE_AXIS,
    REPO_ROOT as EXPORT_REPO_ROOT,
    _compile_seed_state,
    _publish_or_verify,
    _sha256,
    _sha256_file,
    load_config,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    _load_models,
    _measure_candidate,
)
from tools.measure_ddm_v15_scorer_solved_templates import (  # noqa: E402
    DDMV15ScorerSolvedTemplateConfigV1,
)


class VerificationError(ValueError):
    """The runtime proof or scorer custody failed closed."""


def _load_canonical_json(path: Path) -> dict:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"malformed JSON: {path}") from exc
    canonical = rfc8785_canonicalize(value)
    if payload not in (canonical, canonical + b"\n"):
        raise VerificationError(f"JSON is not canonical: {path}")
    return value


def _find_fresh_runtime_receipt(proof_root: Path) -> tuple[Path, dict]:
    matches = sorted(
        (proof_root / "fresh" / ".ddm_runtime_checkpoints").glob(
            "*/*/inflate_receipt.json"
        )
    )
    if len(matches) != 1:
        raise VerificationError(
            f"expected one fresh runtime receipt, observed {len(matches)}"
        )
    path = matches[0]
    return path, _load_canonical_json(path)


def verify(
    *,
    export_config_path: Path,
    scorer_config_path: Path,
) -> tuple[dict, Path]:
    export_config = load_config(export_config_path)
    scorer_value = json.loads(scorer_config_path.read_bytes())
    scorer_config = DDMV15ScorerSolvedTemplateConfigV1.model_validate(scorer_value)
    if scorer_config.pair_start != 0 or scorer_config.pair_count != 600:
        raise VerificationError("scorer config must bind the full n600 window")
    if EXPORT_REPO_ROOT != REPO_ROOT:
        raise VerificationError("exporter repository root drifted")

    output_root = (
        REPO_ROOT / export_config.output_directory
    ).resolve()
    export_receipt_path = (
        output_root.parent / "ddm_e1_runtime_export_receipt.json"
    )
    export_receipt = _load_canonical_json(export_receipt_path)
    upstream_receipt_path = (
        output_root.parent / "ddm_e1_upstream_harness_receipt.json"
    )
    upstream_receipt = _load_canonical_json(upstream_receipt_path)
    archive_path = output_root / "archive.zip"
    archive_bytes, archive_sha256 = _sha256_file(archive_path)
    if (archive_bytes, archive_sha256) != (
        export_receipt["archive"]["bytes"],
        export_receipt["archive"]["sha256"],
    ):
        raise VerificationError("exported archive changed after compilation")
    if (
        upstream_receipt["status"] != "PASS"
        or upstream_receipt["score_claim"] is not False
        or upstream_receipt["failure_reasons"] != []
        or upstream_receipt["under_1800_seconds"] is not True
        or upstream_receipt["packet"]["archive.zip"]
        != {"bytes": archive_bytes, "sha256": archive_sha256}
    ):
        raise VerificationError("upstream harness receipt does not bind this archive")
    runtime_path = output_root / "inflate.py"
    runtime_bytes, runtime_sha256 = _sha256_file(runtime_path)
    if (runtime_bytes, runtime_sha256) != (
        export_receipt["runtime"]["inflate_py"]["bytes"],
        export_receipt["runtime"]["inflate_py"]["sha256"],
    ):
        raise VerificationError("runtime source changed after compilation")

    proof_root = Path(export_config.proof_root)
    runtime_receipt_path, runtime_receipt = _find_fresh_runtime_receipt(proof_root)
    final_path = Path(runtime_receipt["final"]["path"])
    final_identity = _sha256_file(final_path)
    if final_identity != (
        runtime_receipt["final"]["bytes"],
        runtime_receipt["final"]["sha256"],
    ):
        raise VerificationError("fresh final raw changed after inflate")
    identity = export_identity(
        pair_count=600,
        source_bytes=export_receipt["output_identity"]["bytes"],
        source_sha256=export_receipt["output_identity"]["sha256"],
        packaged_bytes=final_identity[0],
        packaged_sha256=final_identity[1],
    )
    if not identity.byte_identical:
        raise VerificationError("packaged raw is not byte-identical to the source receiver")
    if upstream_receipt["raw"] != {
        "bytes": final_identity[0],
        "sha256": final_identity[1],
    }:
        raise VerificationError("upstream harness raw differs from clean receiver raw")
    if float(runtime_receipt["total_seconds"]) >= 1800.0:
        raise VerificationError("fresh single-thread inflate exceeded 30 minutes")

    source_path = (REPO_ROOT / export_config.source_archive_path).resolve()
    source = source_path.read_bytes()
    if (len(source), _sha256(source)) != (
        export_config.source_archive_bytes,
        export_config.source_archive_sha256,
    ):
        raise VerificationError("sealed source archive changed")
    state_archive, state_receiver, dofs = _compile_seed_state(source)

    cache_path = Path(scorer_config.target_cache_path)
    cache_identity = _sha256_file(cache_path)
    if cache_identity != (
        scorer_config.target_cache_bytes,
        scorer_config.target_cache_sha256,
    ):
        raise VerificationError("target scorer cache custody mismatch")
    labels = open_stored_npy_memmap(cache_path, "lstars")
    poses = open_stored_npy_memmap(cache_path, "gt_poses")
    segnet, posenet, model_custody = _load_models(scorer_config)
    measurement_root = proof_root / "scorer_measurement"
    measurement_root.mkdir(parents=True, exist_ok=True)
    measured = _measure_candidate(
        name="seeded_source_state",
        archive=state_archive,
        receiver=state_receiver,
        config=scorer_config,
        root=measurement_root,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
    )
    terms = score_row(
        archive_bytes=archive_bytes,
        d_seg=float(measured["d_seg"]),
        d_pose=float(measured["d_pose"]),
    )
    payload_bytes = {
        row["member"]: int(row["member_payload_range"]["bytes"])
        for row in export_receipt["archive"]["member_homes"]
        if row["member"] is not None
    }
    block_bytes = [
        {
            "block": "L",
            "bytes": payload_bytes["base/chart.ddb"]
            + payload_bytes["semantic/composed.dds"],
            "contents": "composed semantic field plus base chart",
        },
        {
            "block": "D2",
            "bytes": 0,
            "contents": "per-element tolerance dual section inactive",
        },
        {
            "block": "D1",
            "bytes": 0,
            "contents": "amplitude field section inactive",
        },
        {
            "block": "D4",
            "bytes": archive_bytes - sum(payload_bytes.values()),
            "contents": "ZIP framing and container custody",
        },
        {
            "block": "D6",
            "bytes": payload_bytes["manifest.json"],
            "contents": "realization and archive metadata",
        },
        {
            "block": "D5",
            "bytes": 0,
            "contents": "texture-quotient residual-stat section inactive",
        },
    ]
    if sum(int(row["bytes"]) for row in block_bytes) != archive_bytes:
        raise VerificationError("joint-cycle block bytes do not close to archive bytes")
    total_camera_pixels = 600 * 2 * 874 * 1164
    described_camera_paint_fraction = (
        int(export_receipt["paint_jacobian"]["painted_camera_pixels_all_pairs_all_frames"])
        / total_camera_pixels
    )
    result = {
        "archive": {
            "bytes": archive_bytes,
            "member_homes": export_receipt["archive"]["member_homes"],
            "receiver_byte_home_bijection": export_receipt["archive"][
                "receiver_byte_home_bijection"
            ],
            "sha256": archive_sha256,
        },
        "cleanup": {
            "bulk_artifacts_preserved_on_ssd": True,
            "certify_or_block": "no proof, source, cache, or checkpoint bytes deleted",
            "proof_root": str(proof_root),
        },
        "dofs": dofs,
        "evidence_axis": EVIDENCE_AXIS,
        "identity": {
            "byte_identical": identity.byte_identical,
            "bytes": final_identity[0],
            "packaged_sha256": identity.packaged_sha256,
            "source_sha256": identity.source_sha256,
        },
        "joint_iteration_curve": {
            "block_order": ["L", "D2", "D1", "D4", "D6", "D5"],
            "cycles": [
                {
                    "archive_bytes": archive_bytes,
                    "apparatus_validity": {
                        "coder_refit_applied": True,
                        "input_hashes_verified_at_consumption": True,
                        "realized_residual_fresh": True,
                    },
                    "best_so_far": True,
                    "bytes_by_block": block_bytes,
                    "cycle_index": 0,
                    "described_camera_paint_fraction": format(
                        described_camera_paint_fraction, ".12f"
                    ),
                    "d_pose": measured["d_pose"],
                    "d_seg": measured["d_seg"],
                    "realized_argmax_agreement_proxy": format(
                        1.0 - float(measured["d_seg"]), ".12f"
                    ),
                    "score_total": terms["total"],
                    "staleness_input_hashes": {
                        "archive_sha256": archive_sha256,
                        "raw_sha256": final_identity[1],
                        "scorer_cache_sha256": cache_identity[1],
                        "state_archive_sha256": _sha256(state_archive),
                    },
                }
            ],
            "fixed_budget_bytes": archive_bytes,
            "fixed_point_status": "OPEN_ONE_MEASURED_EXPORT_CYCLE_ONLY",
            "global_reinvestment": True,
            "successor_cycle_measurement_policy": {
                "pose": "skip_only_while_pose_state_hash_is_unchanged",
                "seg": "consume_argmax_bit_identical_exact_forward_checkpoint_when_fresh",
                "stale_cache": "rederive_or_mark_stale_advisory",
            },
            "schema": "ddm_joint_fixed_budget_iteration_curve.v1",
            "stop_law": "full_joint_cycle_no_net_gain_at_constant_bytes",
            "successor_cycles_required": True,
        },
        "main_landing_review_required": True,
        "paint_jacobian": export_receipt["paint_jacobian"],
        "pointer_moved": False,
        "rederive_argv": [
            "/Users/adpena/Projects/pact/.venv/bin/python",
            "tools/verify_ddm_runtime_export.py",
            "--export-config",
            str(export_config_path.relative_to(REPO_ROOT)),
            "--scorer-config",
            str(scorer_config_path.relative_to(REPO_ROOT)),
        ],
        "research_only": True,
        "runtime": {
            "dependencies": ["torch", "brotli"],
            "fresh_receipt_path": str(runtime_receipt_path),
            "member_consumption": runtime_receipt["member_consumption"],
            "render_seconds": runtime_receipt["render_seconds"],
            "runtime_bytes": runtime_bytes,
            "runtime_sha256": runtime_sha256,
            "single_thread_cpu": runtime_receipt["single_thread_cpu"],
            "stage_count": runtime_receipt["resume"]["stage_count"],
            "total_seconds": runtime_receipt["total_seconds"],
            "under_1800_seconds": True,
        },
        "schema": "ddm_e1_runtime_verification_receipt.v1",
        "score": {
            "archive_bytes": archive_bytes,
            "d_pose": measured["d_pose"],
            "d_seg": measured["d_seg"],
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            "terms": terms,
        },
        "score_claim": False,
        "scorer": {
            "batch_count": measured["batch_count"],
            "batch_size": measured["batch_size"],
            "batch_digest_chain_sha256": measured[
                "batch_digest_chain_sha256"
            ],
            "custody": model_custody,
            "measurement_root": str(measurement_root),
            "target_cache": {
                "bytes": cache_identity[0],
                "path": str(cache_path),
                "sha256": cache_identity[1],
            },
        },
        "state": {
            "archive_bytes": len(state_archive),
            "archive_sha256": _sha256(state_archive),
            "name": export_config.state_name,
        },
        "upstream_harness": upstream_receipt,
        "verdict": "PASS_EXACT_N600_RUNTIME_EXPORT_ADVISORY_ONLY",
        "verdict_scope": (
            "Exact n600 source-versus-packaged camera raw identity and local "
            "single-thread decode timing; frozen-scorer row is macOS-CPU advisory, "
            "not contest-CPU/CUDA authority or promotion evidence."
        ),
    }
    receipt_path = _publish_or_verify(
        output_root.parent / "ddm_e1_runtime_verification_receipt.json",
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-config", required=True)
    parser.add_argument("--scorer-config", required=True)
    args = parser.parse_args(argv)
    export_config_path = Path(args.export_config).resolve()
    scorer_config_path = Path(args.scorer_config).resolve()
    for path in (export_config_path, scorer_config_path):
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise VerificationError("config paths must be inside the repository") from exc
    result, receipt_path = verify(
        export_config_path=export_config_path,
        scorer_config_path=scorer_config_path,
    )
    print(
        json.dumps(
            {
                "receipt_path": str(receipt_path),
                "score": result["score"],
                "verdict": result["verdict"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
