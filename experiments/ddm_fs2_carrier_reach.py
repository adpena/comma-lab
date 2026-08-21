"""ddm_fs2 - is the frame-0 carrier's pose reach a function of the frame-1 damage?

`ddm_rc4` refused rung 4 on pose by 517x. That refusal was measured with the
carrier held BYTE-IDENTICAL between arms (a stale carrier), so what it priced was
the UNCOMPENSATED damage. `ddm_jg5` then proved on the same vehicle that
re-solving the 12 frame-0 carrier coefficients against the 6 PoseNet equations
turns most of that damage into nothing - but jg5's own compensation factor is a
cross-regime bracket and must not be carried onto a different perturbation
family ([[cross-regime-constant-transfer-genus-finishing-stage]]).

What CAN transfer is a structural property, if it is measured as one: **does the
carrier's reach degrade as the frame-1 perturbation grows?** If it does, rc4's
amplitude (mean uncompensated `delta_d_pose` 3.33e-3) is out of reach and the
refusal stands under compensation too. If it does not, the reach is a property of
the pair, the recovery is bimodal at every amplitude, and a per-pair waterfill can
select the reachable pairs at any damage level.

This is a $0 re-read of four retained ddm_jg5 n600 arrays, content-hashed at read
time. It measures the amplitude dependence directly instead of transferring a
constant. It is NOT a verdict on rung 4: it sizes the job and sets the prior that
the direct measurement then confirms or refutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

JG5 = Path("/Volumes/APDataStore/pact/ddm_jg5/retained/final")
STORE = Path("/Volumes/APDataStore/pact/ddm_fs2")

#: ddm_rc4 POSE_RESCORED_DALI.json, authority-lineage GT: the mean per-pair
#: uncompensated `delta_d_pose` the rung-4 drop produced at p_max >= 0.9921875.
RC4_UNCOMPENSATED_DELTA_D_POSE = 3.327899e-3


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> tuple[np.ndarray, str]:
    path = JG5 / name
    return np.load(path), sha256_file(path)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse() -> dict:
    base, s_base = load("d_pose_per_pair_base_odd_frames.npy")
    cand, s_cand = load("d_pose_per_pair_candidate.npy")
    ref, s_ref = load("d_pose_per_pair_refined_matched.npy")

    uncompensated = cand - base
    residual = ref - base
    edited = np.abs(uncompensated) > 1e-12
    u = uncompensated[edited]
    r = residual[edited]

    quantiles = [0, 20, 40, 60, 80, 90, 100]
    edges = np.percentile(u, quantiles)
    bins = []
    for i in range(len(edges) - 1):
        low, high = float(edges[i]), float(edges[i + 1])
        last = i == len(edges) - 2
        mask = (u >= low) & (u <= high) if last else (u >= low) & (u < high)
        if not mask.any():
            continue
        bins.append(
            {
                "u_low": low,
                "u_high": high,
                "n": int(mask.sum()),
                "median_uncompensated": float(np.median(u[mask])),
                "median_residual": float(np.median(r[mask])),
                "mean_residual": float(r[mask].mean()),
                "fraction_at_or_below_base": float((r[mask] <= 0.0).mean()),
                "sum_ratio": float(u[mask].sum() / r[mask].sum()) if r[mask].sum() else None,
            }
        )

    # The band ddm_rc4's own measured amplitude sits in.
    band_low, band_high = 2.0e-3, 6.0e-3
    band = (u >= band_low) & (u < band_high)
    rc4_band = {
        "band": [band_low, band_high],
        "rc4_measured_uncompensated_delta_d_pose": RC4_UNCOMPENSATED_DELTA_D_POSE,
        "n": int(band.sum()),
        "median_uncompensated": float(np.median(u[band])),
        "median_residual": float(np.median(r[band])),
        "mean_residual": float(r[band].mean()),
        "fraction_at_or_below_base": float((r[band] <= 0.0).mean()),
        "sum_ratio": float(u[band].sum() / r[band].sum()),
    }

    fractions = np.array([b["fraction_at_or_below_base"] for b in bins])
    return {
        "arm": "ddm_fs2",
        "stage": "carrier_reach_amplitude_dependence",
        "axis": "[macOS-CPU advisory] re-read of retained ddm_jg5 n600 arrays; no new forward",
        "score_claim": False,
        "promotable": False,
        "verdict_scope": "FORMULATION: the jg3 seg-edit family on the br1/rc2 body",
        "inputs": {
            "d_pose_per_pair_base_odd_frames.npy": s_base,
            "d_pose_per_pair_candidate.npy": s_cand,
            "d_pose_per_pair_refined_matched.npy": s_ref,
        },
        "totals": {
            "edited_pairs": int(edited.sum()),
            "base_mean": float(base.mean()),
            "candidate_mean": float(cand.mean()),
            "refined_mean": float(ref.mean()),
            "full_set_sum_ratio": float(u.sum() / r.sum()),
        },
        "amplitude_bins": bins,
        "rc4_amplitude_band": rc4_band,
        "spearman_uncompensated_vs_residual": spearman(u, r),
        "amplitude_span_of_bins": float(bins[-1]["median_uncompensated"] / bins[0]["median_uncompensated"]),
        "fraction_at_or_below_base_spread": [float(fractions.min()), float(fractions.max())],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STORE / "FS2_CARRIER_REACH.json")
    args = parser.parse_args(argv)
    out = analyse()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
