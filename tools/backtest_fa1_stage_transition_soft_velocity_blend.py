#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scorer-free FB1 audit/backtest receipt for FA1 soft velocity blend.

The tool inspects banked J3/J4/JD-line artifacts and writes a machine-readable
receipt.  It does not run training, invoke a scorer, or synthesize missing
gradients.  A corpus that lacks the previous optimizer state or the recorded
post-boundary gradient sequence is marked REFUSED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.reset_operator import effective_lr_multiplier  # noqa: E402
from tac.optimization.stage_transition_soft_velocity_blend import (  # noqa: E402
    StageTransitionSoftVelocityBlendConfig,
    replay_custody_issues,
    rms,
)

J3_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_j3_366_fullrun_finalsmoke_64c421698c_20260723T030700Z"
)
J4_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_j4_366_warmstart_smoke_9c3575aa_20260723T042700Z"
)
JD4_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_jd4_20260805")
J4_DIAGNOSIS = REPO_ROOT / ".omx/research/ddm_j4_366_warm_start_diagnosis_receipt_20260723.json"


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _read_first_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                data = json.loads(line)
                return data if isinstance(data, dict) else {}
    return {}


def _npz_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=True) as z:
        return sorted(str(k) for k in z.files)


def _j3_partial_row(required_vectors: int) -> dict[str, Any]:
    step1_ckpt = J3_ROOT / "checkpoints/01_island_worldsheet_joint_descent_intra_global000001.npz"
    step4_ckpt = J3_ROOT / "checkpoints/01_island_worldsheet_joint_descent_intra_global000004.npz"
    tel1 = _read_json(J3_ROOT / "telemetry/step000001.json")
    verdict4 = _read_json(J3_ROOT / "verdicts/01_island_worldsheet_joint_descent_step000004_n600.json")
    gradient_rms = None
    first_moment_rms = None
    if step1_ckpt.exists():
        with np.load(step1_ckpt, allow_pickle=True) as z:
            if "first_moment" in z.files:
                first_moment = np.asarray(z["first_moment"], dtype=np.float64)
                first_moment_rms = rms(first_moment)
                gradient_rms = rms(first_moment / 0.1)
    issues = replay_custody_issues(
        previous_mapped_m_available=False,
        gradient_vectors_available=1 if gradient_rms is not None else 0,
        required_gradient_vectors=required_vectors,
        baseline_controls_available=True,
    )
    return {
        "corpus": "J3 finalsmoke 366 opening window",
        "axis": "[optimizer-state arithmetic / scorer-free audit]",
        "status": "PARTIAL_STEP1_ONLY_REFUSED",
        "required_gradient_vectors": required_vectors,
        "reconstructable_gradient_vectors": 1 if gradient_rms is not None else 0,
        "previous_mapped_m_available": False,
        "soft_blend_backtest": "REFUSED",
        "refusal_reasons": list(issues),
        "observed_opening_signal": {
            "step1_gradient_norm_telemetry": tel1.get("gradient_norm"),
            "step1_local_initial_d_seg": (tel1.get("initial") or {}).get("d_seg"),
            "step1_local_final_d_seg": (tel1.get("final") or {}).get("d_seg"),
            "step1_reconstructed_gradient_rms_from_first_moment": gradient_rms,
            "step1_first_moment_rms": first_moment_rms,
            "uncorrected_adam_eta_step1": effective_lr_multiplier(1),
            "step4_n600_d_seg": verdict4.get("d_seg"),
            "step4_n600_d_pose": verdict4.get("d_pose"),
        },
        "sources": [
            _source(step1_ckpt),
            _source(step4_ckpt),
            _source(J3_ROOT / "telemetry/step000001.json"),
            _source(J3_ROOT / "full_run_receipt.json"),
            _source(J3_ROOT / "verdicts/01_island_worldsheet_joint_descent_step000004_n600.json"),
        ],
    }


def _j4_refused_row(required_vectors: int) -> dict[str, Any]:
    ckpt = J4_ROOT / "checkpoints/01_island_worldsheet_joint_descent_blocked_global000004.npz"
    tel1 = _read_json(J4_ROOT / "telemetry/step000001.json")
    diagnosis = _read_json(J4_DIAGNOSIS)
    diagnosis_opt = (diagnosis.get("exact_failure_localization") or {}).get("optimizer_state") or {}
    issues = replay_custody_issues(
        previous_mapped_m_available=False,
        gradient_vectors_available=0,
        required_gradient_vectors=required_vectors,
        baseline_controls_available=True,
    )
    return {
        "corpus": "J4 warm-start reform 366 opening window",
        "axis": "[optimizer-state arithmetic / scorer-free audit]",
        "status": "REFUSED_MISSING_REPLAY_CUSTODY",
        "required_gradient_vectors": required_vectors,
        "reconstructable_gradient_vectors": 0,
        "previous_mapped_m_available": False,
        "soft_blend_backtest": "REFUSED",
        "refusal_reasons": list(issues),
        "observed_opening_signal": {
            "step1_gradient_norm_telemetry": tel1.get("gradient_norm"),
            "step1_lr_rewarmup_factor": tel1.get("lr_rewarmup_factor"),
            "step1_realized_boundary_crossed": tel1.get("realized_boundary_crossed"),
            "diagnosis_warm_start_optimizer_state_loadable": diagnosis_opt.get(
                "warm_start_optimizer_state_loadable"
            ),
        },
        "checkpoint_keys": _npz_keys(ckpt),
        "sources": [
            _source(ckpt),
            _source(J4_ROOT / "telemetry/step000001.json"),
            _source(J4_ROOT / "full_run_receipt.json"),
            _source(J4_DIAGNOSIS),
        ],
    }


def _jd_line_refused_row(required_vectors: int) -> dict[str, Any]:
    receipts = sorted(JD4_ROOT.glob("tr1_jd4_cont_ep*/tr1_window_receipt.json"))
    telemetry = sorted(JD4_ROOT.glob("tr1_jd4_cont_ep*/telemetry.jsonl"))
    sample_receipt = receipts[-1] if receipts else JD4_ROOT / "MISSING"
    sample_telemetry = sample_receipt.with_name("telemetry.jsonl") if receipts else JD4_ROOT / "MISSING"
    sample_ckpt = (
        sorted((sample_receipt.parent / "checkpoints").glob("*.npz"))[0]
        if receipts and sorted((sample_receipt.parent / "checkpoints").glob("*.npz"))
        else JD4_ROOT / "MISSING"
    )
    first_row = _read_first_jsonl(sample_telemetry)
    receipt = _read_json(sample_receipt)
    ckpt_keys = _npz_keys(sample_ckpt)
    opt_m_keys = [key for key in ckpt_keys if key.startswith("opt::") and key.endswith(".m")]
    issues = replay_custody_issues(
        previous_mapped_m_available=bool(opt_m_keys),
        gradient_vectors_available=0,
        required_gradient_vectors=required_vectors,
        baseline_controls_available=bool(receipt.get("cfg")),
    )
    return {
        "corpus": "JD-line TR1 continuation windows",
        "axis": "[optimizer-state arithmetic / scorer-free audit]",
        "status": "REFUSED_SCALAR_TELEMETRY_ONLY",
        "windows_found": len(receipts),
        "sample_window": str(sample_receipt.parent) if receipts else None,
        "required_gradient_vectors": required_vectors,
        "reconstructable_gradient_vectors": 0,
        "previous_mapped_m_available": bool(opt_m_keys),
        "soft_blend_backtest": "REFUSED",
        "refusal_reasons": list(issues),
        "observed_opening_signal": {
            "sample_event": first_row.get("event"),
            "sample_has_gradient_vector": any(
                key in first_row for key in ("gradient", "grad_vector", "gradients", "first_moment")
            ),
            "sample_reset_arm": receipt.get("reset_arm"),
            "sample_adam_bias_correction": (receipt.get("cfg") or {}).get("adam_bias_correction"),
            "sample_optimizer_moment_keys_found": len(opt_m_keys),
        },
        "sources": [
            _source(sample_receipt),
            _source(sample_telemetry),
            _source(sample_ckpt),
        ],
    }


def build_receipt(required_vectors: int) -> dict[str, Any]:
    cfg = StageTransitionSoftVelocityBlendConfig.from_beta2(0.999, c=2.0)
    rows = [
        _j3_partial_row(required_vectors),
        _j4_refused_row(required_vectors),
        _jd_line_refused_row(required_vectors),
    ]
    return {
        "schema": "ddm_fb1_stage_transition_soft_velocity_blend_backtest.v1",
        "axis": "[optimizer-state arithmetic / scorer-free audit]",
        "score_claim": False,
        "training_launched": False,
        "scorer_launched": False,
        "soft_blend_treatment": {
            "formula": "m_new=(1-alpha(t))*m_mapped+alpha(t)*m_fresh",
            "window_steps": cfg.window_steps,
            "window_derivation": "ceil(2.0/(1-0.999)) optimizer steps",
            "alpha_start": cfg.alpha_start,
            "alpha_end": cfg.alpha_end,
            "shape": cfg.shape,
            "clip_rms": cfg.clip_rms,
        },
        "backtest_rows": rows,
        "verdict": "LESSON-ONLY-confirmed",
        "consumer_trainer": "experiments/train_levelset_witness_realized_through_R_mlx.py",
        "fire_condition": (
            "Capture a real stage-boundary checkpoint with previous mapped optimizer state, "
            "the deterministic post-boundary gradient vectors for the beta2 window or a "
            "pre-registered shorter opening slice, and matched cold/reset controls; then rerun "
            "this scorer-free arithmetic backtest and only enable the trainer consumer if the "
            "soft blend wins both controls at matched update RMS."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=".omx/research/ddm_fb1_20260807/backtest_receipt.json",
        help="Receipt JSON path under repo custody.",
    )
    ap.add_argument("--required-vectors", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    if "/tmp/" in str(out):
        raise SystemExit("refusing to persist FB1 evidence under /tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = build_receipt(args.required_vectors)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "verdict": receipt["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
