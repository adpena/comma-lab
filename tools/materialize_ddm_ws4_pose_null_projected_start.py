#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the receiver-closed WS4 warm starts and custody receipts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.ddm_ws4_pose_null_projected_start import (  # noqa: E402
    EVIDENCE_AXIS,
    POINTER,
    SCHEMA,
    WSEG_N600,
    BoundArtifact,
    WS4PoseNullError,
    build_arbitration_receipt,
    canonical_json_bytes,
    classify_projection_components,
    sha256_bytes,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    DirectDescriptionJointDescentTypedConfigV1,
    compile_parameterized_archive,
    lift_v15_archive,
    load_stage_checkpoint,
)

DEFAULT_CONFIG = REPO_ROOT / ".omx/research/configs/ddm_ws4_pose_null_projected_seg_start_20260725.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _storage_preflight(root: Path, required_free_bytes: int = 1 << 30) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < required_free_bytes:
        raise WS4PoseNullError(f"SSD preflight refused: {free} free bytes < {required_free_bytes}")
    return {
        "tier": str(root),
        "free_bytes": free,
        "required_free_bytes": required_free_bytes,
        "status": "PASS",
        "cleanup": "immutable small archives and receipts retained; no scratch survives success",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema") != "ddm_ws4_pose_null_projected_start_config.v1":
        raise WS4PoseNullError("WS4 typed config schema differs")
    bindings = {
        name: BoundArtifact.from_mapping(value, repo_root=REPO_ROOT) for name, value in config["bindings"].items()
    }
    payloads = {name: artifact.read() for name, artifact in bindings.items()}
    json_inputs = {
        name: json.loads(payload)
        for name, payload in payloads.items()
        if name
        not in {
            "authority",
            "wseg_archive",
            "wjoint_step50_checkpoint",
        }
    }

    classification = classify_projection_components(
        ws1=json_inputs["ws1_receipt"],
        ws2=json_inputs["ws2_receipt"],
        dm2=json_inputs["dm2_receipt"],
        dm4=json_inputs["dm4_receipt"],
        ws3_arbitration=json_inputs["ws3_arbitration"],
        cc3=json_inputs["cc3_receipt"],
        j9_ticket=json_inputs["j9_ticket"],
    )

    ssd_root = Path(config["outputs"]["ssd_root"])
    storage = _storage_preflight(ssd_root)
    archive_dir = ssd_root / "01_archives"

    # No joinable pose-punished component exists, so W_seg_perp is the exact
    # receiver-closed identity materialization.  Re-lift validates the J5
    # consumer boundary without changing bytes.
    wseg = payloads["wseg_archive"]
    wseg_lift = lift_v15_archive(wseg)
    wseg_reemit = wseg_lift.exact_reemit()
    if wseg_reemit != wseg:
        raise WS4PoseNullError("W_seg J5 lift/reemit is not byte-identical")
    wseg_perp_path = archive_dir / "W_seg_perp.zip.receipt-bytes"
    _atomic_write(wseg_perp_path, wseg_reemit)
    wseg_perp = {
        "path": str(wseg_perp_path),
        "sha256": sha256_bytes(wseg_reemit),
        "bytes": len(wseg_reemit),
        "receiver_parse_reemit_byte_identical": True,
        "j5_consumer_lift_reemit_byte_identical": True,
        "j5_low_dim_parameter_count": len(wseg_lift.parameter_names),
        "projection_applied": False,
        "projection_component_count": 0,
    }

    # Preserve the exact live resume state.  The adjacent EMA verdict is not
    # relabeled as a live-state score; J10 owns the shadow-consistent reseal.
    ticket_path = bindings["j9_ticket"].path
    typed = DirectDescriptionJointDescentTypedConfigV1.from_ticket(ticket_path)
    state, checkpoint_metadata = load_stage_checkpoint(
        bindings["wjoint_step50_checkpoint"].path,
        config=typed,
    )
    source_archive = Path(typed.source_archive_path).read_bytes()
    source_lift = lift_v15_archive(source_archive)
    wjoint_live, realized = compile_parameterized_archive(
        source_lift,
        state.theta,
        include_lane_programs=False,
    )
    realized_custody = checkpoint_metadata.get("realized_archive")
    if not isinstance(realized_custody, dict):
        raise WS4PoseNullError("W_joint checkpoint lacks realized-archive custody")
    expected_live_sha = realized_custody.get("sha256")
    expected_live_bytes = realized_custody.get("bytes")
    if sha256_bytes(wjoint_live) != expected_live_sha or len(wjoint_live) != expected_live_bytes:
        raise WS4PoseNullError("W_joint step50 live materialization differs from checkpoint metadata")
    wjoint_path = archive_dir / "W_joint_step50_live.zip.receipt-bytes"
    _atomic_write(wjoint_path, wjoint_live)
    wjoint = {
        "path": str(wjoint_path),
        "sha256": sha256_bytes(wjoint_live),
        "bytes": len(wjoint_live),
        "checkpoint_global_step": int(state.step),
        "parameter_shadow": "live_resume_state",
        "realized_parameter_count": int((realized != 0).sum()),
        "receiver_closed": True,
        "n600_status": "PENDING_J10_SHADOW_CONSISTENT_RESEAL",
        "ema_step50_verdict_not_relabelled": bindings["wjoint_ema_step50_verdict"].receipt(),
    }

    arbitration_inputs = {
        "W_seg_perp_start": wseg_perp,
        "W_seg_full_run_receipt": bindings["ws3_wseg_full_run"].receipt(),
        "W_seg_terminal_proposal": bindings["ws3_wseg_terminal_proposal"].receipt(),
        "W_joint_step50_checkpoint": bindings["wjoint_step50_checkpoint"].receipt(),
        "W_joint_full_run_receipt": bindings["wjoint_full_run"].receipt(),
        "source_ws3_arbitration": bindings["ws3_arbitration"].receipt(),
    }
    arbitration = build_arbitration_receipt(
        ws3_arbitration=json_inputs["ws3_arbitration"],
        terminal_proposal=json_inputs["ws3_wseg_terminal_proposal"],
        wseg_perp_custody=wseg_perp,
        wjoint_step50_custody=wjoint,
        inputs=arbitration_inputs,
    )
    arbitration_path = _resolve(config["outputs"]["arbitration_receipt"])
    _atomic_write(arbitration_path, canonical_json_bytes(arbitration))

    receipt = {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "delegation_checkpoint_key": config["delegation_checkpoint_key"],
        "authority": bindings["authority"].receipt(),
        "typed_config": {
            "path": str(args.config),
            "sha256": sha256_bytes(config_bytes),
            "bytes": len(config_bytes),
        },
        "inputs": {name: artifact.receipt() for name, artifact in bindings.items()},
        "storage_preflight": storage,
        "pose_coupling_classification": classification,
        "materialized_archives": {
            "W_seg_perp": wseg_perp,
            "W_joint_step50_live": wjoint,
        },
        "n600_endpoint": {
            "W_seg_perp": {
                **WSEG_N600,
                "custody": "REUSED_SETTLED_EXACT_N600_BY_BYTE_IDENTICAL_ARCHIVE_SHA",
                "fresh_scorer_pass_invoked": False,
                "reuse_reason": "ALREADY_SETTLED_TABLE_AND_EXACT_ARCHIVE_BYTE_IDENTITY",
                "evidence_axis": EVIDENCE_AXIS,
                "score_claim": False,
            },
            "W_joint_step50_live": {
                "status": "PENDING_J10_SHADOW_CONSISTENT_N600_RESEAL",
                "ema_result_reuse_forbidden": True,
                "score_claim": False,
            },
        },
        "arbitration_receipt": {
            "path": str(arbitration_path),
            "sha256": sha256_bytes(arbitration_path.read_bytes()),
            "bytes": arbitration_path.stat().st_size,
            "schema": arbitration["schema"],
        },
        "verdict": "POST_HOC_PROJECTION_FORMULATION_LOSS_KEEP_WJOINT_STEP50_LIVE",
        "verdict_scope": (
            "FORMULATION: sealed WS2 W_seg has no lawfully joinable pose-punished component; "
            "identity W_seg_perp inherits WS3 exact first-proposal SEG_REGRESSION. "
            "No negative on future jointly measured per-correction projection."
        ),
        "score_claim": False,
        "research_only": True,
        "execution_allowed": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "main_review_required": True,
    }
    receipt_path = _resolve(config["outputs"]["receipt"])
    _atomic_write(receipt_path, canonical_json_bytes(receipt))

    ssd_receipt = ssd_root / "ddm_ws4_pose_null_projected_seg_start_receipt.json"
    _atomic_write(ssd_receipt, receipt_path.read_bytes())
    print(
        json.dumps(
            {
                "status": "PASS_FORMULATION_STOP",
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_bytes(receipt_path.read_bytes()),
                "W_seg_perp": wseg_perp,
                "W_joint_step50_live": wjoint,
                "arbitration": arbitration["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
