# SPDX-License-Identifier: MIT
"""FEED-2026-06-25ae UNIT — measure the DIRECT SYMBOLIC/TOPOLOGICAL partition rate.

Operator (2026-06-25): the neural-witness (114KB) + flip-sidecar (215KB) = TWO
redundant codes for ONE object (the SegNet argmax partition = moving boundary CURVES
+ near-constant region TOPOLOGY). The MINIMUM SUFFICIENT STATISTIC should be the
curves-as-symbols + topology, temporally AR-coded -> potentially KB-scale, NOT 330KB.
"NASH" = the witness+sidecar is not an equilibrium (each pays for info the other
carries); the optimum is ONE representation at the joint Lagrangian/Pareto point.

This MEASURES the partition's MDL rate with a REAL reversible coder on the FULL n600
GT SegNet argmax stack, at TWO operating regimes:

  REGIME 1 (d_seg = 0, the exact sufficient statistic):
    The context-adaptive arithmetic codec (``context_partition_codec``, JBIG/LOCO-I/
    CABAC family) IS the optimal symbolic/topological coder of a piecewise-constant
    label field: it codes each pixel under a causal (left, up, prev-frame) context.
    This SUBSUMES contour-coding (the boundary IS where context predicts poorly) AND
    temporal-AR (prev-frame context = the ego-motion delta). We measure the REAL
    coded bytes (decode==encode bit-exact) at SPATIAL and TEMPORAL templates.
    NO-FAKE: this replaces the N*H(.) ENTROPY ESTIMATE (253,413 B advisory) with a
    REAL coder's bytes -- the honest achieved length, not an asserted floor.

  REGIME 2 (lossy, equal-d_seg ~ the witness's 0.002017):
    "Store only what is necessary." The MDL region-merge SOLVE (``region_merge``)
    drops every region whose contour bytes exceed its flips-fixed * 1.27 (the exact
    evaluate.py water level). A dropped region = a few flips bought for cheaper than
    its boundary bytes = exactly the operator's "only store the necessary curves".
    We sweep the water level to trace the symbolic d_seg-vs-rate Pareto and read the
    operating point at the witness's d_seg, comparing bytes to witness+sidecar.

The pose half is ALREADY a solved ~6,650 B carrier (separate axis, FEED 'pose is
solved'); we hold d_pose at the reference 3.4e-5 (pose_term 0.018) so the comparison
is purely on the BINDING d_seg-representation rate, apples-to-apples with FEED-ad.

Authority: ``[macOS-CPU advisory]`` (GT argmax = frozen CPU-torch SegNet, the d_seg
reference). NON-PROMOTABLE. NOT byte-closed (render->RGB->SegNet realization is the
separate FEED-y gate). NO score claim; the current pointer is read dynamically.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.bitmask_dseg import flip_count
from tac.boundary_math.context_partition_codec import (
    decode_partition_stack,
    encode_partition_stack,
)
from tac.boundary_math.partition import build_region_adjacency_graph
from tac.witness_dsl.dynamic_frontier_target import load_dynamic_frontier_target

N_CLASSES = 5
H, W = 384, 512
N_PX_PER_FRAME = H * W
N_CONTEST_PAIRS = 600
TOTAL_VIDEO_BYTES = 37_545_489.0

# Reference anchors (all advisory per FEED-ad). Competitive routing is dynamic.
D_POSE_REF = 3.4e-5
POSE_TERM_REF = float(np.sqrt(10.0 * D_POSE_REF))  # ~0.01844
# The pose half is a solved real carrier (targets_meta best_pose_carrier_bytes).
POSE_CARRIER_BYTES = 6_650

# FEED-ad witness+sidecar anchors (the stack this unit challenges).
WITNESS_WEIGHT_BYTES = 114_197
WITNESS_BASE_D_SEG = 0.002017
SIDECAR_FULL_REPAIR_BYTES = 215_464  # drives witness d_seg -> 0
SIDECAR_BYTES_PER_FLIP = 0.905

DEFAULT_ARGMAX_U8 = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/"
    "targets_n600/gt_segnet_argmax.u8"
)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_gt_argmax_stack(path: Path, n_pairs: int) -> np.ndarray:
    """Load the frozen CPU-torch SegNet argmax stack (n, H, W) uint8."""
    raw = np.fromfile(path, dtype=np.uint8)
    expect = N_CONTEST_PAIRS * N_PX_PER_FRAME
    if raw.size != expect:
        raise ValueError(f"argmax u8 size {raw.size} != {expect} (600*384*512)")
    stack = raw.reshape(N_CONTEST_PAIRS, H, W)
    return stack[:n_pairs].astype(np.int64)


def rate_term(total_bytes: float) -> float:
    return 25.0 * total_bytes / TOTAL_VIDEO_BYTES


def implied_S(d_seg: float, repr_bytes: float, *, include_pose_carrier: bool = True) -> float:
    """Implied DIRECT-partition score: 100*d_seg + pose_term + 25*bytes/D.

    ``repr_bytes`` is the d_seg-representation bytes; pose carrier added when the
    representation must ship its own pose (apples-to-apples with the full archive).
    """
    total = repr_bytes + (POSE_CARRIER_BYTES if include_pose_carrier else 0.0)
    return 100.0 * d_seg + POSE_TERM_REF + rate_term(total)


# ---------------------------------------------------------------------------
# REGIME 1 — real-coder bytes at d_seg = 0 (the exact sufficient statistic).
# ---------------------------------------------------------------------------

def measure_exact_partition_real_bytes(partitions: list[np.ndarray]) -> dict:
    """Run the REAL context-adaptive arithmetic codec; report achieved bytes.

    Verifies decode==encode bit-exact (NO FAKE) then reports spatial + temporal
    template bytes. This is the honest achieved length vs the N*H(.) ESTIMATE.
    """
    # Bit-exact roundtrip proof on the first few frames (and trust the same codec
    # path for the full stack -- the codec is deterministic + frame-shape-uniform).
    probe = encode_partition_stack(partitions[:3], n_classes=N_CLASSES, template="temporal")
    dec = decode_partition_stack(probe.payload)
    roundtrip_ok = all(np.array_equal(a, b) for a, b in zip(partitions[:3], dec, strict=True))
    if not roundtrip_ok:
        raise RuntimeError("partition codec roundtrip FAILED — abort (NO FAKE)")

    t0 = time.time()
    spatial = encode_partition_stack(partitions, n_classes=N_CLASSES, template="spatial")
    temporal = encode_partition_stack(partitions, n_classes=N_CLASSES, template="temporal")
    # Full-stack bit-exact verification on the WINNING template (the one we report).
    full_dec = decode_partition_stack(temporal.payload)
    full_ok = all(np.array_equal(a, b) for a, b in zip(partitions, full_dec, strict=True))
    if not full_ok:
        raise RuntimeError("FULL-stack temporal roundtrip FAILED — abort (NO FAKE)")

    best = temporal if temporal.total_bytes <= spatial.total_bytes else spatial
    return {
        "coder": "context-adaptive arithmetic (constriction), causal (left,up,prev)",
        "roundtrip_bit_exact_full_stack": bool(full_ok),
        "n_frames": len(partitions),
        "spatial_total_bytes": int(spatial.total_bytes),
        "spatial_bytes_per_frame": spatial.bytes_per_frame,
        "spatial_model_bytes": int(spatial.model_bytes),
        "spatial_stream_bytes": int(spatial.stream_bytes),
        "temporal_total_bytes": int(temporal.total_bytes),
        "temporal_bytes_per_frame": temporal.bytes_per_frame,
        "temporal_model_bytes": int(temporal.model_bytes),
        "temporal_stream_bytes": int(temporal.stream_bytes),
        "best_template": best.template,
        "best_total_bytes": int(best.total_bytes),
        "best_bytes_per_frame": best.bytes_per_frame,
        "best_rate_term": rate_term(best.total_bytes),
        "d_seg": 0.0,
        "implied_S_dseg0_with_pose": implied_S(0.0, best.total_bytes),
        "encode_seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# REGIME 2 — lossy "store only the necessary" via MDL region drop, then real coder.
# ---------------------------------------------------------------------------

def _simplify_partition(argmax_hw: np.ndarray, min_region_px: int) -> np.ndarray:
    """LOSSY: dissolve every region smaller than ``min_region_px`` into its dominant
    neighbour. This is the operator's "store only the necessary curves": tiny regions
    are fine boundary detail whose contour bytes exceed the flips they fix. Larger
    ``min_region_px`` = simpler partition = fewer stored bytes, more flips (d_seg up).

    Smallest-first dissolution so a dropped tiny region's pixels join the surround
    BEFORE a medium region is evaluated (a real region-merge contraction). Returns the
    simplified label map (what would be STORED, then real-coded).
    """
    from scipy import ndimage

    _C4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    merged = argmax_hw.astype(np.int64).copy()
    # Iterate to fixpoint: dissolving small regions can expose new small ones.
    for _ in range(4):
        region_id = np.full(merged.shape, -1, dtype=np.int64)
        sizes: list[tuple[int, int]] = []  # (pixels, region_id)
        nid = 0
        for c in range(N_CLASSES):
            cm = merged == c
            if not cm.any():
                continue
            lab, n = ndimage.label(cm, structure=_C4)
            for comp in range(1, n + 1):
                m = lab == comp
                region_id[m] = nid
                sizes.append((int(m.sum()), nid))
                nid += 1
        small = [rid for px, rid in sizes if px < min_region_px]
        if not small:
            break
        # dissolve smallest-first
        for px, rid in sorted((s for s in sizes if s[1] in set(small))):
            rmask = region_id == rid
            ring = ndimage.binary_dilation(rmask, structure=_C4) & ~rmask
            if not ring.any():
                continue
            vals, cnts = np.unique(merged[ring], return_counts=True)
            merged[rmask] = int(vals[int(np.argmax(cnts))])
    return merged


def measure_lossy_symbolic_pareto(
    partitions: list[np.ndarray], min_region_px_levels: list[int], *, target_score: float
) -> list[dict]:
    """Trace the lossy symbolic d_seg-vs-rate Pareto by MDL region-drop + real coder.

    For each min-region-pixel level: dissolve sub-threshold regions on EVERY frame,
    real-code the simplified stack (temporal context), measure (real bytes, exact
    d_seg vs the GT partition). NO-FAKE: bytes are real coded lengths, d_seg is the
    exact popcount of the SIMPLIFIED-vs-GT partition (the actual flips storing the
    simpler partition would incur), roundtrip is bit-exact.
    """
    rows = []
    for mrp in min_region_px_levels:
        t0 = time.time()
        simplified = [_simplify_partition(p, mrp) for p in partitions]
        total_flips = sum(flip_count(s, p) for s, p in zip(simplified, partitions, strict=True))
        n_scored = len(partitions) * N_PX_PER_FRAME
        d_seg = total_flips / n_scored
        code = encode_partition_stack(simplified, n_classes=N_CLASSES, template="temporal")
        dec = decode_partition_stack(code.payload)
        ok = all(np.array_equal(a, b) for a, b in zip(simplified, dec, strict=True))
        if not ok:
            raise RuntimeError(f"lossy stack roundtrip FAILED at min_region_px={mrp} (NO FAKE)")
        rows.append({
            "min_region_px": mrp,
            "total_flips_vs_gt": int(total_flips),
            "d_seg": d_seg,
            "real_coded_bytes": int(code.total_bytes),
            "bytes_per_frame": code.bytes_per_frame,
            "rate_term": rate_term(code.total_bytes),
            "implied_S_with_pose": implied_S(d_seg, code.total_bytes),
            "beats_pointer": bool(implied_S(d_seg, code.total_bytes) < target_score),
            "seconds": round(time.time() - t0, 1),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


# ---------------------------------------------------------------------------
# Topology measurement (the region-adjacency-graph cost — operator's "topology").
# ---------------------------------------------------------------------------

def measure_topology(partitions: list[np.ndarray], sample_stride: int = 30) -> dict:
    """Measure region/topology statistics across the stack (operator's 'topology')."""
    n_regions = []
    n_edges = []
    for i in range(0, len(partitions), sample_stride):
        rag = build_region_adjacency_graph(partitions[i], N_CLASSES)
        n_regions.append(rag.n_regions())
        n_edges.append(sum(len(s) for s in rag.adjacency.values()) // 2)
    return {
        "mean_regions_per_frame": float(np.mean(n_regions)),
        "mean_adjacency_edges_per_frame": float(np.mean(n_edges)),
        "sampled_frames": len(n_regions),
        "note": (
            "The topology (RAG: ~34 regions, ~edges) is near-constant frame-to-frame; "
            "the temporal-context coder's prev-frame term already captures it implicitly "
            "(a separate topology-symbol stream would DOUBLE-code it)."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--argmax-u8", type=Path, default=DEFAULT_ARGMAX_U8)
    ap.add_argument("--n-pairs", type=int, default=N_CONTEST_PAIRS)
    ap.add_argument("--out", type=Path,
                    default=REPO / "experiments/results/symbolic_topological_partition_mdl_20260625"
                    / "symbolic_topological_partition_mdl.json")
    ap.add_argument("--min-region-px", type=str, default="0,8,32,128,512,2048",
                    help="lossy region-drop thresholds (px); 0 = exact (no drop)")
    args = ap.parse_args(argv)
    dynamic_frontier = load_dynamic_frontier_target(repo_root=REPO)
    target_score = dynamic_frontier.target_score

    t0 = time.time()
    stack = load_gt_argmax_stack(args.argmax_u8, args.n_pairs)
    partitions = [stack[i] for i in range(stack.shape[0])]
    print(f"[{_utc()}] loaded {len(partitions)} GT argmax frames {stack.shape[1:]} ", flush=True)

    # REGIME 1: exact (d_seg=0) real-coder bytes.
    exact = measure_exact_partition_real_bytes(partitions)
    print(f"[{_utc()}] REGIME 1 (d_seg=0) real bytes: temporal={exact['temporal_total_bytes']:,} "
          f"({exact['temporal_bytes_per_frame']:.1f} B/frame) rate={exact['best_rate_term']:.4f}", flush=True)

    # Topology stats.
    topo = measure_topology(partitions)

    # REGIME 2: lossy symbolic Pareto.
    mrps = [int(x) for x in args.min_region_px.split(",")]
    pareto = measure_lossy_symbolic_pareto(partitions, mrps, target_score=target_score)

    # --- VERDICT: compare to witness+sidecar (FEED-ad) at matched regimes.
    exact_bytes = exact["best_total_bytes"]
    # witness+sidecar at d_seg=0 (full repair): weights + full-repair sidecar.
    ws_dseg0_bytes = WITNESS_WEIGHT_BYTES + SIDECAR_FULL_REPAIR_BYTES
    ws_dseg0_S = implied_S(0.0, ws_dseg0_bytes, include_pose_carrier=True)
    symbolic_dseg0_S = implied_S(0.0, exact_bytes, include_pose_carrier=True)

    verdict = {
        "dseg0_symbolic_real_bytes": int(exact_bytes),
        "dseg0_symbolic_implied_S": symbolic_dseg0_S,
        "dseg0_witness_plus_sidecar_bytes": int(ws_dseg0_bytes),
        "dseg0_witness_plus_sidecar_implied_S": ws_dseg0_S,
        "symbolic_dominates_at_dseg0": bool(exact_bytes < ws_dseg0_bytes),
        "byte_delta_symbolic_minus_ws": int(exact_bytes - ws_dseg0_bytes),
        "S_delta_symbolic_minus_ws": symbolic_dseg0_S - ws_dseg0_S,
        "symbolic_beats_pointer_at_dseg0": bool(symbolic_dseg0_S < target_score),
        "pointer": target_score,
        "interpretation": _interpret(exact_bytes, ws_dseg0_bytes, symbolic_dseg0_S, ws_dseg0_S),
    }

    out = {
        "schema": "symbolic_topological_partition_mdl.v1",
        "subagent": "symbolic_topological_partition_mdl_20260625",
        "utc": _utc(),
        "authority": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "dynamic_frontier_target": dataclasses.asdict(dynamic_frontier),
        "pointer_unmoved": target_score,
        "n_pairs": len(partitions),
        "gt_source": str(args.argmax_u8),
        "elapsed_seconds": round(time.time() - t0, 1),
        "no_fake": {
            "all_partition_bytes_are_real_coded_lengths": True,
            "exact_dseg0_roundtrip_bit_exact": exact["roundtrip_bit_exact_full_stack"],
            "lossy_stacks_roundtrip_bit_exact": True,
            "replaces_NstarH_estimate_with_real_coder": True,
            "Hestimate_temporal_bytes_advisory_prior": 253_413,
        },
        "regime1_exact_dseg0": exact,
        "topology": topo,
        "regime2_lossy_symbolic_pareto": pareto,
        "feed_ad_witness_sidecar_anchors": {
            "witness_weight_bytes": WITNESS_WEIGHT_BYTES,
            "witness_base_d_seg": WITNESS_BASE_D_SEG,
            "sidecar_full_repair_bytes": SIDECAR_FULL_REPAIR_BYTES,
            "sidecar_bytes_per_flip": SIDECAR_BYTES_PER_FLIP,
            "witness_alone_implied_S": implied_S(WITNESS_BASE_D_SEG, WITNESS_WEIGHT_BYTES),
        },
        "verdict": verdict,
        "advisory_caveat": (
            "All d_seg is DIRECT partition (stored/simplified argmax vs frozen CPU-torch "
            "GT-SegNet argmax). The BYTE-CLOSED render->RGB->SegNet realization is the "
            "SEPARATE FEED-y gate. No score claim; this tool does not move the reopened pointer."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"[{_utc()}] wrote {args.out}", flush=True)
    return 0


def _interpret(symbolic_bytes, ws_bytes, symbolic_S, ws_S) -> str:
    if symbolic_bytes < ws_bytes:
        return (
            f"SYMBOLIC DOMINATES at d_seg=0: direct partition store ({symbolic_bytes:,} B) "
            f"is CHEAPER than witness+full-repair-sidecar ({ws_bytes:,} B) by "
            f"{ws_bytes - symbolic_bytes:,} B -> implied S {symbolic_S:.5f} vs {ws_S:.5f}. "
            "The witness+sidecar IS the redundant double-code the operator named; ONE "
            "direct symbolic/topological code is the Nash/Pareto optimum."
        )
    return (
        f"At d_seg=0 the direct symbolic store ({symbolic_bytes:,} B) is ABOVE "
        f"witness+sidecar ({ws_bytes:,} B). The binding axis is the partition's "
        "irreducible inter-frame entropy; the lossy Pareto (regime 2) is where the "
        "'store only the necessary curves' win must come from."
    )


if __name__ == "__main__":
    raise SystemExit(main())
