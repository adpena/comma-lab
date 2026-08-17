"""ddm_rc4 - re-score the RETAINED pose vectors against the AUTHORITY-lineage GT.

ddm_pi2 (MAIN relay 2026-08-16, supersedes the additive-floor fallback): the
advisory pose gap was OUR OWN TOOLING reading two ground truths - the seg half
off a DALI-lineage cache, the pose half decoding GT fresh with PyAV every run.
The FIX is to score pose against
/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt["pose"],
measured to track contest authority at 1.00081x.

Because ddm_rc4 RETAINED every per-pair pose vector (base and dropped), the fix
costs a re-score, not a re-render.  This is what ALWAYS-KEEP-THE-PAYLOAD buys.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np
import torch

STORE = pathlib.Path("/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816")
DALI = pathlib.Path("/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")
DALI_SHA16 = "a91d98252fe377c5"
PYAV_GT = pathlib.Path(
    "/Volumes/APDataStore/pact/ddm_mt1_t4_sign_gate_20260814_custody/"
    "ddm_mt1_t4_sign_gate_20260814/inputs/gt_pose.npy"
)
D_POSE_AUTH = 0.0082945765**2 / 10.0


def ds_pose(delta_abs: float) -> float:
    return math.sqrt(10.0 * (D_POSE_AUTH + delta_abs)) - math.sqrt(10.0 * D_POSE_AUTH)


def main() -> int:
    if hashlib.sha256(DALI.read_bytes()).hexdigest()[:16] != DALI_SHA16:
        raise SystemExit("DALI GT cache sha mismatch - refusing")
    gt_dali = torch.load(DALI, map_location="cpu", weights_only=False)["pose"].numpy().astype(np.float64)
    gt_pyav = np.load(PYAV_GT).astype(np.float64)

    rows = json.loads((STORE / "retained" / "pose_leg" / "pose_u7.0.json").read_text())
    out = {}
    for label, gt in (("dali_authority_lineage", gt_dali), ("pyav_prior_lineage", gt_pyav)):
        se_b, se_d = [], []
        for r in rows:
            g = gt[r["pair"]]
            se_b.append(float(((np.asarray(r["base"], dtype=np.float64) - g) ** 2).mean()))
            se_d.append(float(((np.asarray(r["drop"], dtype=np.float64) - g) ** 2).mean()))
        db, dd = float(np.mean(se_b)), float(np.mean(se_d))
        out[label] = {
            "d_pose_base": db,
            "d_pose_drop": dd,
            "delta_d_pose_absolute": dd - db,
            "delta_S_pose_at_authority_baseline": ds_pose(dd - db),
            "base_over_authority": db / D_POSE_AUTH,
        }

    fix, fallback = out["dali_authority_lineage"], out["pyav_prior_lineage"]
    result = {
        "arm": "ddm_rc4",
        "stage": "1c_pose_rescored_against_authority_lineage_gt",
        "axis": "[macOS-CPU advisory, stratified-random n=48] COMPONENT-ONLY NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "gt_dali_sha256_prefix": DALI_SHA16,
        "u": 7.0,
        "p_max_threshold": 1.0 - 2.0**-7.0,
        "sample_pairs": len(rows),
        "results": out,
        "fix_vs_fallback": {
            "delta_S_pose_fix": fix["delta_S_pose_at_authority_baseline"],
            "delta_S_pose_fallback": fallback["delta_S_pose_at_authority_baseline"],
            "ratio_fallback_over_fix": (
                fallback["delta_S_pose_at_authority_baseline"]
                / fix["delta_S_pose_at_authority_baseline"]
                if fix["delta_S_pose_at_authority_baseline"]
                else float("inf")
            ),
        },
    }
    (STORE / "POSE_RESCORED_DALI.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for k, v in out.items():
        print(f"{k:>28}: base {v['d_pose_base']:.6e} ({v['base_over_authority']:.2f}x authority) "
              f"delta {v['delta_d_pose_absolute']:.6e}  dS_pose {v['delta_S_pose_at_authority_baseline']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
