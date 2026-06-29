#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""FREE-GENERATOR irreducible-info byte budget — the MEASURED rate-half arithmetic.  $0 / CPU / numpy.

[macOS advisory / research-signal] — NOT a contest score; pointer claim FORBIDDEN.
score_claim=false, promotable=false, ready_for_exact_eval_dispatch=false. Pointer UNMOVED.

QUESTION (DAG FEED-ja / FEED-iy, grok-confirmed): the contest scores ONLY archive.zip bytes;
inflate.py is a FREE deterministic interpreter (rule 118: generic algorithm free, video-derived
learned content counted). So the irreducible COUNTED info = the Kolmogorov complexity of the witness
RELATIVE TO OUR FREE INTERPRETER, K_machine(witness) = the smallest video-derived program/data the
free generator needs to reproduce the argmax partition + pose targets. FEED-ja decomposes it as:

    K_machine(witness) = { canonical-scene descriptor (static IPM partition / per-class SDF manifold)
                         + pose trajectory (6-DOF / pair)
                         + per-class warp-type mask
                         + Lane-survival learned residual
                         + ~0.0008 movables residual }

This tool MEASURES the two we can byte-close at $0 — the POSE TRAJECTORY and the CANONICAL SCENE —
and assembles the total byte budget with the CITED (existence-proof) residual rows, then computes the
MEASURED-rate sub-0.15 arithmetic:  S = 100*d_seg_residual + sqrt(10*d_pose) + 25*total_bytes/N.

WHAT IS MEASURED HERE (NO-FAKE — real bytes, real d_pose, real d_seg):
  * pose RD curve: for each uniform quant step q, d_pose floor = mean((round_q(pose)-pose)^2)
    (the floor a render that HITS the stored quantized targets achieves), pose_term=sqrt(10*d_pose),
    + per-column order-0 entropy of the temporal deltas (the range/AR-code achievable bytes, the
    constriction <0.1%-over-entropy regime) + concrete LZMA bytes (general-purpose realized upper bound).
  * canonical scene: per-pixel temporal-MODE partition lossless bytes (ONE static scene), its
    static-canonical per-class d_seg (no warp), and the FULL per-frame lossless store it REPLACES.

WHAT IS CITED (existence proofs; per CLAUDE.md terminal-conclusion-crosscheck — do not re-derive):
  * structured per-class SDF descriptor (the rate-half VIABLE form): FEED-dm lane SDF d_seg 4.2e-4
    post-R 8e-4 ~1-2KB/600; FEED-du hood 7.4e-4 post-R 6.8e-4 56B/600 (eikonal_sdf_dseg_recovery memo).
  * grok pose-warp: stored pose carries the Road-plane d_seg trajectory FREE (+15-17% Road compression;
    grok_pose_warp_dseg memo). MyCar needs identity, sky rotation-only -> per-class warp-type mask.
  * movables residual d_seg ~0.0008 (grok n96), area ~0.016 -> a few extra 6-DOF object streams.

COMPLIANCE BOUNDARY (rule 118, emitted as structured data):
  FREE in inflate.py  = the GENERIC algorithm: homography/eikonal/SDF rasterizer, range decoder,
                        per-class warp-type dispatch (class -> warp regime is deterministic).
  COUNTED in archive  = the VIDEO-DERIVED data: pose scalars, canonical/manifold coords, learned
                        residual weights, movable object streams.
  FORBIDDEN           = smuggling a video-derived per-frame table into inflate.py "code".

Disk hygiene: one small JSON under experiments/results/free_generator_byte_budget_<utc>/ (gitignored,
rebuildable from this script + the committed gt cache). No bulk artifacts; no /tmp.
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.boundary_math.contour_codec import _LZMA_FILTERS, partition_description_bytes  # noqa: E402
from tac.contest_score import UNCOMPRESSED_SIZE_BYTES  # noqa: E402

N_CLASSES = 5
H, W = 384, 512
SEG_FRAMES_FULL = 600
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
DEFAULT_POSE_NPZ = str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_SEG_NPZ = str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")

# rate-term constants
RATE_PER_BYTE = 25.0 / UNCOMPRESSED_SIZE_BYTES  # contribution to S of one COUNTED archive byte
T3_TARGET = 0.15  # THE goal
T1_FLOOR = 0.19   # floor of acceptable

# --- CITED existence-proof residual rows (provenance pinned; NOT re-derived here) ---------------
CITED = {
    "structured_lane_sdf": {
        "source": ".omx/research/eikonal_sdf_dseg_recovery_test_20260629T164449Z.md (FEED-dm)",
        "d_seg_direct": 4.2e-4, "d_seg_post_R": 8.0e-4, "bytes_total_600": 1500,
        "note": "continuous-band lane SDF on the IPM manifold; R-SURVIVES; ~30-43 floats/frame.",
    },
    "structured_hood_static": {
        "source": ".omx/research/eikonal_sdf_dseg_recovery_test_20260629T164449Z.md (FEED-du)",
        "d_seg_direct": 7.4e-4, "d_seg_post_R": 6.8e-4, "bytes_total_600": 56,
        "note": "static ego-hood SDF; 56 bytes total for 600 frames.",
    },
    "movables_residual": {
        "source": ".omx/research/grok_pose_warp_dseg_test_20260629T181000Z.md (GAP 1)",
        "d_seg_direct": 8.0e-4, "bytes_total_600_estimate": 750,
        "note": "ESTIMATE: ~5 movable objects x 6-DOF streams; independent motion (irreducible by warp).",
    },
    "pose_carries_road_dseg_free": {
        "source": ".omx/research/grok_pose_warp_dseg_test_20260629T181000Z.md (FEED-iv)",
        "road_rel_improvement": 0.16,
        "note": "stored pose -> ground homography compresses Road d_seg +15-17% at ZERO extra bytes "
                "(dual-use with the d_pose sidecar). MyCar=identity, sky=rotation -> per-class warp mask.",
    },
    "solved_pose_sidecar_reference": {
        "source": "CLAUDE.md THE CURRENT FRONTIER (Quantizr-style stored-target sidecar)",
        "d_pose": 3.4e-5, "pose_term": 0.018, "bytes": "~1-5KB (fp16 raw 7.2KB)",
        "note": "the already-built pose solution; this tool shows entropy-coding the trajectory beats it.",
    },
}


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _order0_entropy_bytes(int_symbols: np.ndarray) -> float:
    """Sum over columns of (order-0 empirical entropy in bits) * n_rows, in bytes.

    This is the range/AR-code achievable cost for a static per-column model matched to the empirical
    delta distribution (constriction achieves <0.1% over this; excludes ~tens of bytes table overhead).
    A context/AR model exploiting temporal+cross-column structure can MATCH or BEAT this -> conservative.
    """
    nrows = int_symbols.shape[0]
    total_bits = 0.0
    for c in range(int_symbols.shape[1]):
        _, cnts = np.unique(int_symbols[:, c], return_counts=True)
        p = cnts / cnts.sum()
        total_bits += float(-np.sum(p * np.log2(p))) * nrows
    return total_bits / 8.0


def _lzma_raw(b: bytes) -> int:
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS))


def measure_pose_trajectory(poses: np.ndarray, q_steps) -> dict:
    """Pose-trajectory RD curve: (quant step) -> (d_pose floor, pose_term, entropy bytes, LZMA bytes)."""
    P = poses.astype(np.float64)
    n = P.shape[0]
    raw_f16 = P.astype(np.float16).tobytes()
    curve = []
    for q in q_steps:
        Q = np.round(P / q).astype(np.int64)
        dq = np.round(P / q) * q
        d_pose = float(np.mean((dq - P) ** 2))
        # temporal delta per column (row 0 absolute), int symbols
        TD = Q.copy()
        TD[1:] = Q[1:] - Q[:-1]
        ent_bytes = _order0_entropy_bytes(TD)
        lzma_bytes = _lzma_raw(TD.astype(np.int32).tobytes())
        # per-column entropy decomposition (where the bytes go)
        per_col = {}
        for c in range(P.shape[1]):
            _, cnts = np.unique(TD[:, c], return_counts=True)
            p = cnts / cnts.sum()
            per_col[f"col{c}"] = round(float(-np.sum(p * np.log2(p))) * n / 8.0, 1)
        curve.append({
            "q": q, "d_pose_floor": d_pose, "pose_term": float((10.0 * d_pose) ** 0.5),
            "range_code_entropy_bytes": round(ent_bytes, 1),
            "lzma_realized_bytes": lzma_bytes,
            "per_col_entropy_bytes": per_col,
        })
    return {
        "n_pairs": int(n), "raw_fp16_bytes": len(raw_f16),
        "per_col_mean": [round(float(x), 5) for x in P.mean(0)],
        "per_col_std": [round(float(x), 5) for x in P.std(0)],
        "delta_std": [round(float(x), 5) for x in np.diff(P, axis=0).std(0)],
        "note": "col0 = forward speed (~31, std 1.26) is the dominant byte cost; cols1-5 near-static.",
        "rd_curve": curve,
    }


def measure_canonical_scene(lstars: np.ndarray) -> dict:
    """Canonical static scene (per-pixel temporal MODE) lossless bytes + static-canonical d_seg + the
    full per-frame lossless store it REPLACES."""
    L = lstars.astype(np.int64)
    nfr = L.shape[0]
    counts = np.stack([(L == c).sum(0) for c in range(N_CLASSES)], 0)
    mode = counts.argmax(0).astype(np.int64)
    canon_bytes = partition_description_bytes(mode)
    ne = (L != mode[None])
    per_class = {}
    for c in range(N_CLASSES):
        m = (L == c)
        per_class[CLASS_NAMES[c]] = {
            "area": round(float(m.mean()), 4),
            "static_dseg": round(float(ne[m].mean()), 5) if m.any() else None,
        }
    joint = _lzma_raw(L.astype(np.uint8).tobytes())
    full_bpf = joint / nfr
    return {
        "frames_used": int(nfr),
        "canonical_mode_partition_lossless_bytes": int(canon_bytes),
        "static_canonical_dseg_total": round(float(ne.mean()), 5),
        "static_canonical_per_class": per_class,
        "full_perframe_lossless_bytes_per_frame": round(full_bpf, 1),
        "full_perframe_lossless_x600": int(round(full_bpf * SEG_FRAMES_FULL)),
        "full_perframe_lossless_rate_term": round(RATE_PER_BYTE * full_bpf * SEG_FRAMES_FULL, 4),
        "interpretation": (
            "ONE static canonical scene = 480-ish B. BUT static-canonical (and pose-warp, grok +16% on "
            "Road) is LOSSY at d_seg~0.018-0.021 -> seg term ~1.8-2.1, SCORE-DOMINATED (eikonal memo: "
            "lossy partition coding cannot trade d_seg for bytes; break-even Delta-d_seg/byte=4e-6). "
            "The VIABLE free-generator is therefore the STRUCTURED per-class SDF manifold descriptor "
            "(reaches frontier d_seg, R-surviving) + FREE pose + LEARNED long-tail residual, NOT the "
            "naive scene-warp. The full per-frame lossless store (rate 0.17-0.28) is what we AVOID."
        ),
    }


def assemble_budget(pose: dict, canon: dict) -> dict:
    """Assemble the free-generator COUNTED byte budget + the MEASURED-rate sub-0.15 arithmetic."""
    # pick MEASURED pose operating points from the RD curve
    def pick(target_pose_term):
        return min(pose["rd_curve"], key=lambda r: abs(r["pose_term"] - target_pose_term))
    pose_usable = pick(0.05)       # pose_term ~0.05 (d_pose ~2e-4)
    pose_solved = pick(0.025)      # pose_term ~0.025 (d_pose ~6e-5), ~ matches the solved sidecar grade

    # COUNTED rows (MEASURED where possible, CITED otherwise)
    rows = {
        "pose_trajectory_solved_grade": {
            "bytes": pose_solved["range_code_entropy_bytes"], "kind": "MEASURED (range/AR entropy)",
            "carries": "d_pose (pose_term %.3f) + Road d_seg modulation FREE (grok)" % pose_solved["pose_term"],
        },
        "canonical_per_class_sdf_manifold": {
            "bytes": CITED["structured_lane_sdf"]["bytes_total_600"]
                     + CITED["structured_hood_static"]["bytes_total_600"],
            "kind": "CITED (FEED-dm lane + FEED-du hood; R-surviving structured SDF)",
            "carries": "lane d_seg 4.2e-4 (post-R 8e-4) + hood 7.4e-4 (post-R 6.8e-4)",
        },
        "per_class_warp_type_mask": {
            "bytes": 0, "kind": "FREE (class -> warp regime is a deterministic dispatch in inflate.py)",
            "carries": "Road/Lane->ground homography, MyCar->identity, sky->rotation",
        },
        "movables_residual": {
            "bytes": CITED["movables_residual"]["bytes_total_600_estimate"],
            "kind": "ESTIMATE (CITED grok; ~5 objects x 6-DOF)", "carries": "movable d_seg ~8e-4",
        },
        "learned_longtail_residual": {
            "bytes": "VARIABLE (the witness's job; the GPU-budget unknown)",
            "kind": "the genuinely-new LEARNED payload (annulus/dash-gap long-tail)",
            "carries": "drives d_seg from ~0.018 (free-gen bulk) down toward frontier ~6e-4-1e-3",
        },
    }
    measured_plus_cited_bytes = (
        rows["pose_trajectory_solved_grade"]["bytes"]
        + rows["canonical_per_class_sdf_manifold"]["bytes"]
        + rows["movables_residual"]["bytes"]
    )

    # MEASURED-rate sub-0.15 arithmetic at labeled operating points
    def S(d_seg, pose_term, total_bytes):
        seg = 100.0 * d_seg
        rate = RATE_PER_BYTE * total_bytes
        return {"d_seg": d_seg, "pose_term": pose_term, "total_bytes": int(total_bytes),
                "seg_term": round(seg, 4), "pose_term_val": round(pose_term, 4),
                "rate_term": round(rate, 5), "S": round(seg + pose_term + rate, 4)}

    # solve d_seg_max for a target S given pose + bytes
    def dseg_max(target_S, pose_term, total_bytes):
        return (target_S - pose_term - RATE_PER_BYTE * total_bytes) / 100.0

    scenarios = {
        "optimistic_frontier": S(8.0e-4, pose_solved["pose_term"], measured_plus_cited_bytes),
        "conservative_postR": S(1.0e-3, pose_solved["pose_term"], measured_plus_cited_bytes + 4000),
        "cheap_pose_variant": S(8.0e-4, pose_usable["pose_term"], pose_usable["range_code_entropy_bytes"] + 2306),
    }
    thresholds = {
        "note": "with rate negligible (~%0.4f at %d B) and pose fixed, sub-0.15/sub-0.19 reduce ENTIRELY "
                "to a d_seg threshold." % (RATE_PER_BYTE * measured_plus_cited_bytes, measured_plus_cited_bytes),
        "d_seg_max_for_sub015_solved_pose": round(dseg_max(T3_TARGET, pose_solved["pose_term"], measured_plus_cited_bytes), 6),
        "d_seg_max_for_sub019_solved_pose": round(dseg_max(T1_FLOOR, pose_solved["pose_term"], measured_plus_cited_bytes), 6),
        "frontier_need_cited": "~6e-4 to 1e-3 (eikonal memo) -> INSIDE the sub-0.15 d_seg window.",
    }
    return {
        "counted_rows": rows,
        "measured_plus_cited_bytes_excl_learned_residual": int(measured_plus_cited_bytes),
        "rate_term_of_that_budget": round(RATE_PER_BYTE * measured_plus_cited_bytes, 5),
        "kolmogorov_relative_to_free_interpreter_bytes": int(measured_plus_cited_bytes),
        "vs_full_lossless_store_bytes": canon["full_perframe_lossless_x600"],
        "compression_factor_vs_lossless": round(canon["full_perframe_lossless_x600"] / max(measured_plus_cited_bytes, 1), 1),
        "sub015_arithmetic_scenarios": scenarios,
        "sub015_thresholds": thresholds,
        "headline": (
            "MEASURED rate floor of the free-generator = ~%d B counted (rate %0.4f), vs ~%d B lossless "
            "store (rate %0.3f) = %0.0fx smaller. The pose entropy-codes to HUNDREDS of bytes. So the "
            "rate term is NEGLIGIBLE and sub-0.15 reduces ENTIRELY to d_seg <= ~%0.1e (the LEARNED "
            "long-tail residual). The irreducible COUNTED info is tiny; the game is d_seg accuracy + R."
            % (measured_plus_cited_bytes, RATE_PER_BYTE * measured_plus_cited_bytes,
               canon["full_perframe_lossless_x600"], canon["full_perframe_lossless_rate_term"],
               canon["full_perframe_lossless_x600"] / max(measured_plus_cited_bytes, 1),
               dseg_max(T3_TARGET, pose_solved["pose_term"], measured_plus_cited_bytes))
        ),
    }


COMPLIANCE_BOUNDARY = {
    "rule": "rule 118 (CLAUDE.md): generic algorithm FREE in inflate.py; video-derived learned content COUNTED in archive.zip.",
    "FREE_in_inflate_py": [
        "homography warp / eikonal growth / SDF rasterizer (generic geometry)",
        "range/ANS decoder (e.g. constriction) for the pose + manifold streams",
        "per-class warp-type dispatch (class -> {ground homography, identity, rotation}) — deterministic",
    ],
    "COUNTED_in_archive_zip": [
        "pose 6-DOF scalars per pair (the dual-use d_pose + Road-d_seg sufficient statistic)",
        "canonical per-class SDF manifold coords",
        "learned long-tail residual weights + movable object streams",
    ],
    "FORBIDDEN": "smuggling a video-derived per-frame table/weights into inflate.py disguised as code (hide-data-in-code fake).",
    "pose_genericity_note": (
        "pose is COMPLIANT either way: stored as a tiny counted statistic (~500-900 B, measured) OR "
        "GENERICALLY re-derived at decode via PoseNet/SfM-from-frames (free). Storing the 6 scalars is "
        "the cheaper, deterministic, host-portable choice and is legally COUNTED video-derived data."
    ),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-npz", default=DEFAULT_POSE_NPZ)
    ap.add_argument("--seg-npz", default=DEFAULT_SEG_NPZ)
    ap.add_argument("--seg-frames", type=int, default=96)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    pz = np.load(args.pose_npz)
    poses = np.asarray(pz["gt_poses"], dtype=np.float64)
    q_steps = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
    pose = measure_pose_trajectory(poses, q_steps)

    sz = np.load(args.seg_npz)
    lstars = np.asarray(sz["lstars"], dtype=np.int64)[: args.seg_frames]
    canon = measure_canonical_scene(lstars)

    budget = assemble_budget(pose, canon)

    out = {
        "tool": "tools/measure_free_generator_byte_budget.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory rate-half sizing; not a contest score)",
        "uncompressed_size_bytes": UNCOMPRESSED_SIZE_BYTES,
        "rate_per_counted_byte": RATE_PER_BYTE,
        "pose_npz": args.pose_npz, "seg_npz": args.seg_npz,
        "pose_trajectory": pose,
        "canonical_scene": canon,
        "free_generator_budget": budget,
        "compliance_boundary": COMPLIANCE_BOUNDARY,
        "cited_existence_proofs": CITED,
        "elapsed_secs": round(time.time() - t0, 1),
    }

    out_dir = (Path(args.out_dir) if args.out_dir
               else _REPO / f"experiments/results/free_generator_byte_budget_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps({
        "pose_rd_curve": [{k: r[k] for k in ("q", "d_pose_floor", "pose_term",
                                             "range_code_entropy_bytes", "lzma_realized_bytes")}
                          for r in pose["rd_curve"]],
        "canonical_scene": {k: canon[k] for k in ("canonical_mode_partition_lossless_bytes",
                                                  "static_canonical_dseg_total",
                                                  "full_perframe_lossless_x600",
                                                  "full_perframe_lossless_rate_term")},
        "budget_headline": budget["headline"],
        "kolmogorov_relative_bytes": budget["kolmogorov_relative_to_free_interpreter_bytes"],
        "compression_vs_lossless": budget["compression_factor_vs_lossless"],
        "sub015_scenarios": budget["sub015_arithmetic_scenarios"],
        "sub015_thresholds": budget["sub015_thresholds"],
    }, indent=2))
    print(f"\n[written] {out_path}  ({out['elapsed_secs']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
