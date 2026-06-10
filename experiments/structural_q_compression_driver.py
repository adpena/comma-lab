#!/usr/bin/env python
"""Thin CLI driver for #71 structural Q* compression.

Delegates to ``tac.optimization.structural_q_compression`` (the byte-cost
re-encoder + transforms) and the non-MPS exact-scorer harness
(``render_and_score_lib``) to: apply a structural transform to the frontier
HNeRV decoder, measure the EXACT decoder-blob brotli bytes AND the EXACT
d_seg/d_pose on N pairs, and emit a ``scorer_quotient_candidate_row.v1``
(candidate_kind=structural_compression). The firewall promotes ONLY a
contest-tier exact-evaluate row with recomputed ΔS<0.

Usage (from repo root, with the pr110pp analysis dir on sys.path):
    .venv/bin/python experiments/structural_q_compression_driver.py \
        --transform prune --keep 0.5 --n-eval 16 --json out.json

Tag: [macOS-CPU advisory] — candidate-generator. A pointer move requires the
confirming paired contest CPU+CUDA exact eval (this driver's authority_tier is
``exact_cpu_advisory`` / metric_family ``exact_pair_scorer``; it does NOT promote).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_ANALYSIS = REPO / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
for _p in (str(REPO / "src"), str(_ANALYSIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.optimization import structural_q_compression as SQ  # noqa: E402
from tac.optimization.scorer_quotient_candidate_row import (  # noqa: E402
    ScorerQuotientCandidateRow,
)

ARCHIVE_BYTES = 177_169  # frontier archive (177,169 B); decoder blob is 162,127 of it
FRONTIER_SHA = "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e"


def _build_codes(parse_frontier, HNeRVDecoder):
    p = parse_frontier()
    sd = p["decoder_sd"]
    probe = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(384, 512))
    order = [name for name, _ in probe.state_dict().items()]
    items = [(name, sd[name].numpy().astype(np.float64)) for name in order]
    codes = SQ.recover_all_codes(items)
    return p, sd, codes


def _transform_codes(codes, sd, transform, keep, rank_frac):
    """Apply the structural transform; return (modified_codes, modified_sd)."""
    import torch  # noqa: PLC0415

    big = {idx: tc for idx, tc in codes.items()
           if tc.codes_i8.size > 1000 and tc.codes_i8.ndim >= 2}
    mod_codes = dict(codes)
    mod_sd = {k: v.clone() for k, v in sd.items()}
    for idx, tc in big.items():
        if transform == "prune":
            sens = SQ.magnitude_sensitivity(tc.codes_i8)
            new_codes = SQ.score_aware_prune_codes(tc.codes_i8, sens, keep)
        elif transform == "lowrank":
            full = min(tc.shape[0], int(np.prod(tc.shape[1:])))
            r = max(1, int(round(full * rank_frac)))
            lr = SQ.low_rank_truncate_weight(
                (tc.codes_i8.astype(np.float64) * tc.scale).reshape(tc.shape), r
            )
            new_codes = np.clip(np.round(lr / tc.scale), -127, 127).astype(np.int8)
        else:
            raise ValueError(f"unknown transform {transform}")
        mod_codes[idx] = SQ.TensorCode(tc.name, idx, tc.shape, new_codes, tc.scale)
        mod_sd[tc.name] = torch.from_numpy(new_codes.astype(np.float32) * tc.scale)
    return mod_codes, mod_sd


def _score(parsed_template, decoder_sd, pairs, L, scorer, gt_bthwc):
    parsed = copy.copy(parsed_template)
    parsed["decoder_sd"] = decoder_sd
    r = L.FrontierRenderer(parsed=parsed)
    comp = r.render_baseline_pairs(pairs)
    segs, poses = [], []
    for pi in pairs:
        pose, seg = scorer.score_batch(gt_bthwc[pi], L.comp_pair_to_bthwc(comp[pi]))
        segs.append(float(seg[0]))
        poses.append(float(pose[0]))
    return float(np.mean(segs)), float(np.mean(poses))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transform", choices=["prune", "lowrank"], default="prune")
    ap.add_argument("--keep", type=float, default=0.5, help="prune keep fraction")
    ap.add_argument("--rank-frac", type=float, default=0.5, help="lowrank fraction of full rank")
    ap.add_argument("--n-eval", type=int, default=16, help="pairs for exact distortion")
    ap.add_argument("--json", type=str, default=None, help="write the candidate row JSON here")
    args = ap.parse_args(argv)

    import render_and_score_lib as L  # noqa: PLC0415
    from model import HNeRVDecoder  # noqa: PLC0415

    p, sd, codes = _build_codes(L.parse_frontier, HNeRVDecoder)
    base_blob = SQ.encode_decoder_blob_bytes(codes)
    pairs = list(range(args.n_eval))
    scorer = L.ExactScorer()
    gt = L.decode_gt_pairs(pairs)
    gt_bthwc = {pi: L.comp_pair_to_bthwc(gt[pi].float().permute(0, 3, 1, 2)) for pi in pairs}

    t0 = time.time()
    base_seg, base_pose = _score(p, sd, pairs, L, scorer, gt_bthwc)
    mod_codes, mod_sd = _transform_codes(codes, sd, args.transform, args.keep, args.rank_frac)
    mod_blob = SQ.encode_decoder_blob_bytes(mod_codes)
    mod_seg, mod_pose = _score(p, mod_sd, pairs, L, scorer, gt_bthwc)
    elapsed = time.time() - t0

    bytes_after = ARCHIVE_BYTES + (mod_blob - base_blob)
    row = ScorerQuotientCandidateRow(
        lever_id=f"71_structural_{args.transform}_keep{args.keep}_rf{args.rank_frac}",
        candidate_kind="structural_compression",
        base_archive_sha256=FRONTIER_SHA,
        bytes_before=ARCHIVE_BYTES,
        bytes_after=bytes_after,
        d_seg_before=base_seg,
        d_seg_after=mod_seg,
        d_pose_before=base_pose,
        d_pose_after=mod_pose,
        # advisory: macOS-CPU exact pair-scorer, NOT contest-tier exact_evaluate.
        authority_tier="exact_cpu_advisory",
        metric_family="exact_pair_scorer",
        decision="reject" if (mod_seg + mod_pose) > (base_seg + base_pose) else "continue",
        runtime_seconds=elapsed,
    )
    out = {
        "lever_id": row.lever_id,
        "candidate_kind": row.candidate_kind,
        "transform": args.transform,
        "keep": args.keep,
        "rank_frac": args.rank_frac,
        "n_eval_pairs": args.n_eval,
        "base_decoder_blob": base_blob,
        "mod_decoder_blob": mod_blob,
        "decoder_blob_delta": mod_blob - base_blob,
        "bytes_before": row.bytes_before,
        "bytes_after": row.bytes_after,
        "d_seg_before": base_seg,
        "d_seg_after": mod_seg,
        "d_pose_before": base_pose,
        "d_pose_after": mod_pose,
        "score_before_subset": row.score_before,
        "score_after_subset": row.score_after,
        "delta_score_total_subset": row.delta_score_total,
        "rate_only_delta_score": SQ.rate_only_delta_score(mod_blob, base_blob),
        "pointer_update_eligible": row.pointer_update_eligible,
        "authority_tier": row.authority_tier,
        "metric_family": row.metric_family,
        "decision": row.decision,
        "runtime_seconds": elapsed,
        "note": "advisory candidate-generator; pointer move requires contest CPU+CUDA exact eval",
    }
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
