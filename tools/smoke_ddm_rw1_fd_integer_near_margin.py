#!/usr/bin/env python
"""Bounded rw1 FD smoke for integer near-margin proposals.

This does not relaunch the FD campaign.  It runs at most n<=8 pairs and at
most the requested proposal count, then writes a durable accept-rate receipt.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "experiments"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    Scorer,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_stage_decomposition_and_solved_paint import resize_to_scorer  # noqa: E402

from tac.optimization.fd_integer_near_margin_proposals import (  # noqa: E402
    IntegerNearMarginProposalGenerator,
)
from tac.optimization.rw1_true_domain_instruments import (  # noqa: E402
    element_grade_vector,
    write_json_atomic,
)

DEFAULT_RAW = Path("/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/boxsolve_inflate/0.raw")
DEFAULT_GT_MKV = REPO / "upstream/videos/0.mkv"
DEFAULT_OUT = REPO / ".omx/research/ddm_rw1_20260806/fd_integer_near_margin_smoke.json"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as exc:  # pragma: no cover
        return f"UNKNOWN:{type(exc).__name__}:{exc}"


def _logits_chw(sc: Scorer, pair_u8: np.ndarray) -> np.ndarray:
    x = sc._to_bthwc(pair_u8)
    with torch.inference_mode():
        logits = sc.net.segnet(sc.net.segnet.preprocess_input(x))[0]
    return logits.detach().cpu().numpy().astype(np.float64)


def _scorer_rgb_u8(frame: np.ndarray) -> np.ndarray:
    value = torch.round(resize_to_scorer(frame))[0].permute(1, 2, 0).numpy()
    return np.clip(value, 0, 255).astype(np.uint8)


def _selection(limit: int, stride: int, offset: int) -> list[int]:
    if limit < 1 or limit > 8:
        raise RuntimeError(f"rw1 FD smoke is bounded to 1<=limit<=8, got {limit}")
    pairs = [int(offset + stride * i) for i in range(limit)]
    bad = [p for p in pairs if p < 0 or p >= N_PAIRS_TOTAL]
    if bad:
        raise RuntimeError(f"selected pairs outside population: {bad}")
    return pairs


def _payload(args: argparse.Namespace, rows: list[dict[str, Any]], started: float) -> dict[str, Any]:
    n_prop = int(sum(row["result"]["n_proposals"] for row in rows))
    n_acc = int(sum(row["result"]["n_accepted"] for row in rows))
    grade = element_grade_vector(
        chain_name="fd1_fd2_integer_near_margin_reopen",
        overrides={
            "init": ("OPTIMAL-RECEIPT", "FD1 inflated substrate plus decoded GT, bounded pair selection"),
            "step_rule": ("OPTIMAL-RECEIPT", "ascending realized current-minus-target logit margin"),
            "stopping_rule": ("OPTIMAL-RECEIPT", "bounded n<=8 smoke and max proposal count, no campaign relaunch"),
            "metric": ("OPTIMAL-RECEIPT", "proposal acceptance requires realized SegNet argmax improvement"),
            "subset": ("OPTIMAL-RECEIPT", f"strided limit={args.limit} stride={args.stride} offset={args.offset}"),
            "realization": ("OPTIMAL-RECEIPT", f"dk1 {args.method} private-block uint8 camera lattice"),
            "projection": ("OPTIMAL-RECEIPT", "no post-hoc pose-null-only projection; proposals are realized before validation"),
            "tie_breaks": ("NAIVE-NAMED", "stable lexical tie-break after margin sort"),
            "seed": ("UNKNOWN", "no stochastic seed consumed"),
            "caches": ("NAIVE-NAMED", "FD1 raw substrate read directly; no new cache authority"),
        },
    )
    return {
        "schema": "ddm_rw1_fd_integer_near_margin_smoke.v1",
        "utc": _utc(),
        "git": _git_hash(),
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "inputs": {
            "fd1_raw": str(args.raw),
            "gt_mkv": str(args.gt_mkv),
        },
        "bounds": {
            "limit_pairs": int(args.limit),
            "max_proposals_total": int(args.max_proposals),
            "scorer_slot": "small-n<=8 smoke only; no n600 scorer job",
        },
        "baseline": {
            "fd1_accept_rate": "0/6",
            "fd1_source": ".omx/research/ddm_fd1_family_d_gn_DAG_FEED_20260728.md",
            "fd2_source": ".omx/research/ddm_fd2_posenull_gn_disambiguation_20260728.md",
        },
        "instrument": {
            "method": args.method,
            "dk1_cvp_tap_radius": int(args.cvp_tap_radius),
            "dk1_dykstra_iterations": int(args.dykstra_iterations),
        },
        "aggregate": {
            "n_pairs": len(rows),
            "n_proposals": n_prop,
            "n_accepted": n_acc,
            "accept_rate": n_acc / n_prop if n_prop else None,
            "delta_vs_fd1_baseline_accepts": n_acc,
            "elapsed_sec": time.time() - started,
        },
        "element_grade_vector": grade,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--gt-mkv", type=Path, default=DEFAULT_GT_MKV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-proposals", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--method", choices=["cvp", "dykstra", "naive"], default="cvp")
    ap.add_argument("--cvp-tap-radius", type=int, default=1)
    ap.add_argument("--dykstra-iterations", type=int, default=8)
    args = ap.parse_args()

    if args.max_proposals < 1:
        raise RuntimeError("--max-proposals must be positive")
    started = time.time()
    pairs = _selection(args.limit, args.stride, args.offset)
    raw = np.memmap(
        args.raw,
        dtype=np.uint8,
        mode="r",
        shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3),
    )
    wanted = {seq_len * p + k for p in pairs for k in (0, 1)}
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    generator = IntegerNearMarginProposalGenerator(
        method=args.method,
        dykstra_iterations=args.dykstra_iterations,
        cvp_tap_radius=args.cvp_tap_radius,
    )
    remaining = int(args.max_proposals)
    rows: list[dict[str, Any]] = []
    for p in pairs:
        if remaining <= 0:
            break
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        flips0 = lstar != lgt
        before = int(flips0.sum())
        logits = _logits_chw(sc, dec)
        base_sc = _scorer_rgb_u8(dec[1])
        target_sc = _scorer_rgb_u8(gt[1])

        def validator(
            candidate: np.ndarray,
            context: Mapping[str, Any],
            *,
            dec_f0: np.ndarray = dec[0],
            lgt_pair: np.ndarray = lgt,
            flips0_pair: np.ndarray = flips0,
            before_pair: int = before,
        ) -> Mapping[str, Any]:
            lam = sc.seg_argmax(np.stack([dec_f0, candidate]))
            after = lam != lgt_pair
            after_count = int(after.sum())
            return {
                "accepted": bool(after_count < before_pair),
                "flips_before": before_pair,
                "flips_after": after_count,
                "fixed": int((flips0_pair & ~after).sum()),
                "introduced": int((~flips0_pair & after).sum()),
                "accept_rule": "flips_after < baseline_flips_before on realized SegNet argmax",
                "site": context["site"],
            }

        result = generator.generate(
            camera_frame=dec[1],
            base_scorer_hwc=base_sc,
            target_scorer_hwc=target_sc,
            logits_chw=logits,
            realized_argmax=lstar,
            target_argmax=lgt,
            max_proposals=remaining,
            validator=validator,
        )
        remaining -= int(result["n_proposals"])
        rows.append(
            {
                "pair": int(p),
                "flips_before": before,
                "result": result,
            }
        )
        payload = _payload(args, rows, started)
        write_json_atomic(args.out, payload)
        print(
            f"[rw1-fd] pair={p} proposals={result['n_proposals']} "
            f"accepted={result['n_accepted']} remaining={remaining}",
            flush=True,
        )

    payload = _payload(args, rows, started)
    write_json_atomic(args.out, payload)
    print(
        f"[rw1-fd] DONE accepted={payload['aggregate']['n_accepted']}/"
        f"{payload['aggregate']['n_proposals']} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
