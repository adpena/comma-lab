#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""PTQ-on-frontier FLOOR — the decisive $0 gate (Path-B QAT floor measurement).

THE QUESTION (CLAUDE.md "THE GOAL — SUB-0.15"; desk-calc memo
``.omx/research/capacity_rd_score_aware_qat_pivot_20260618T175040Z.md``): the 0.19110
contest-CPU frontier (lane pr110, archive 177,169 B) already bought near-frontier d_seg
(~0.00056) by spending decoder capacity; its problem is PURELY RATE (62% of S is bytes,
the 161,104 B decoder section = 90.9% of the archive). The desk model says QAT-shrinking
the decoder to an int4-perfect-hold → S=0.1399 (sub-0.15). The OPEN question this probe
answers EMPIRICALLY: how far does NAIVE PTQ (no finetune) — and SCORE-AWARE per-stage
allocated PTQ — land from those perfect-hold models? That distance is the floor score-aware
QAT must recover from (or, if PTQ already beats 0.191, a possible $0 pointer-mover).

WHAT IT MEASURES (all from EXISTING reusable pieces — tac.frontier_decoder_ptq):
  * decode the frontier decoder state_dict (proven byte-identical identity round-trip).
  * for each grid (int8 baseline, int7/6/5/4 uniform, score-aware int4-low/int8-high,
    score-aware int4-low/int6-high): requantize → re-encode → MEASURED archive bytes.
  * full 600-pair exact d_seg/d_pose/rate/score via RealScorerContext.exact_eval (CPU
    AUTHORITY — the G3-proven path; NEVER MPS).

VERDICT: (1) PTQ floor vs 0.19110; (2) gap to the desk int4-perfect-hold 0.1399 = the
distortion QAT must recover; (3) score-aware vs uniform at matched bytes.

AUTHORITY: ``[contest-CPU advisory]`` NON-PROMOTABLE. Frontier pointer UNMOVED at 0.19110.
CPU only. $0 — no GPU/paid dispatch.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

_RATE_DENOM = 37_545_489
# Desk-calc reference points (from the pivot memo; all [advisory]).
_FRONTIER_S = 0.19109982419209975
_DESK_INT4_PERFECT_HOLD_S = 0.1399
_SUB015 = 0.15


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--targets-cache",
        default="experiments/results/capstone_gt_targets_cache",
        help="capped-GT targets cache dir (reused across probes)",
    )
    ap.add_argument(
        "--scratch-dir",
        default=".omx/tmp/ptq_frontier_floor",
        help="where re-encoded candidate archives are written (auto-cleaned)",
    )
    ap.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help="override the mode list (default: full uniform + score-aware sweep)",
    )
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))

    from tac.frontier_decoder_ptq import (
        FRONTIER_ARCHIVE,
        build_frontier_decoder,
        decode_frontier_member,
        reencode_frontier_archive,
        requantize_decoder_state_dict,
        score_aware_requant_state_dict,
    )
    from tac.frontier_decoder_ptq import (
        score as score_fn,
    )
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    import_vendored_bundle()
    video_path = import_vendored("data").get_default_video_path()

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    t_decode = time.time()
    member = decode_frontier_member(FRONTIER_ARCHIVE)
    n_pairs = int(member.latents.shape[0])
    n_params = sum(int(v.numel()) for v in member.state_dict.values())
    print(
        f"[frontier] archive={FRONTIER_ARCHIVE} member_bytes={member.member_bytes} "
        f"decoder_params={n_params} n_pairs={n_pairs} ({time.time() - t_decode:.1f}s)",
        flush=True,
    )

    # Mode dispatch: each entry is (label, requant_callable). int8 first (baseline).
    def _uniform(mode: str):
        return lambda sd: requantize_decoder_state_dict(sd, mode)

    all_modes: dict[str, object] = {
        "int8": _uniform("int8"),  # codec baseline (identity grid)
        "int7": _uniform("int7"),
        "int6": _uniform("int6"),
        "int5": _uniform("int5"),
        "int4": _uniform("int4"),
        # score-aware per-stage: coarsen d_seg-blind early/low-res stages to int4,
        # protect the tiny d_seg-critical output-proximal heads at higher precision.
        "sa_int4lo_int8hi": lambda sd: score_aware_requant_state_dict(
            sd, low_nbits=4, high_nbits=8
        ),
        "sa_int4lo_int6hi": lambda sd: score_aware_requant_state_dict(
            sd, low_nbits=4, high_nbits=6
        ),
        "sa_int5lo_int8hi": lambda sd: score_aware_requant_state_dict(
            sd, low_nbits=5, high_nbits=8
        ),
    }
    mode_list = args.modes if args.modes else list(all_modes)

    ctx = RealScorerContext(
        video_path,
        device="cpu",
        train_device="cpu",
        split_by_head=False,
        max_pairs=n_pairs,
        targets_cache=args.targets_cache,
    )

    rows: dict[str, dict] = {}
    for mode in mode_list:
        if mode not in all_modes:
            raise SystemExit(f"unknown mode {mode!r}; have {sorted(all_modes)}")
        t0 = time.time()
        shrunk = all_modes[mode](member.state_dict)  # type: ignore[operator]
        cand = scratch / f"frontier_ptq_{mode}.zip"
        bytes_meas = reencode_frontier_archive(member, shrunk, cand)
        decoder = build_frontier_decoder(shrunk)
        res = ctx.exact_eval(decoder, member.latents, bytes_meas)
        d_seg = float(res["seg_distortion"])
        d_pose = float(res["pose_distortion"])
        s = score_fn(d_seg, d_pose, bytes_meas)
        rows[mode] = {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": bytes_meas,
            "rate_term": 25.0 * bytes_meas / _RATE_DENOM,
            "score": s,
            "wall_s": round(time.time() - t0, 1),
        }
        # auto-clean the candidate archive (certified rebuildable: deterministic from
        # the frontier + the mode; the bytes are recorded in this report).
        try:
            cand.unlink()
        except OSError:
            pass
        print(
            f"[{mode:18}] d_seg={d_seg:.6f} d_pose={d_pose:.6f} bytes={bytes_meas:>6} "
            f"rate={rows[mode]['rate_term']:.4f} S={s:.4f}  ΔS_vs_frontier={s - _FRONTIER_S:+.4f}  "
            f"({rows[mode]['wall_s']}s) [contest-CPU advisory]",
            flush=True,
        )

    # ── verdict ──────────────────────────────────────────────────────────────
    i8 = rows.get("int8")
    ranked = sorted(rows.items(), key=lambda kv: kv[1]["score"])
    best_mode, best = ranked[0]
    uniform_modes = [m for m in ("int7", "int6", "int5", "int4") if m in rows]
    sa_modes = [m for m in rows if m.startswith("sa_")]

    # (1) PTQ floor vs frontier 0.19110.
    beats_frontier = {m: r for m, r in rows.items() if r["score"] < _FRONTIER_S - 1e-9}
    # (2) gap to desk int4-perfect-hold 0.1399.
    gap_to_perfect_hold = best["score"] - _DESK_INT4_PERFECT_HOLD_S
    # (3) score-aware vs uniform at MATCHED bytes (compare each SA row to the uniform row
    # with the closest bytes).
    sa_vs_uniform = {}
    for sm in sa_modes:
        sb = rows[sm]["archive_bytes"]
        # nearest uniform by bytes
        nearest = min(uniform_modes, key=lambda um: abs(rows[um]["archive_bytes"] - sb)) if uniform_modes else None
        if nearest is not None:
            sa_vs_uniform[sm] = {
                "nearest_uniform": nearest,
                "sa_bytes": sb,
                "uniform_bytes": rows[nearest]["archive_bytes"],
                "sa_S": rows[sm]["score"],
                "uniform_S": rows[nearest]["score"],
                "sa_beats_uniform_dS": round(rows[sm]["score"] - rows[nearest]["score"], 5),
                "sa_d_seg": rows[sm]["d_seg"],
                "uniform_d_seg": rows[nearest]["d_seg"],
            }

    verdict = {
        "frontier_baseline_S": _FRONTIER_S,
        "desk_int4_perfect_hold_S": _DESK_INT4_PERFECT_HOLD_S,
        "sub_015_target": _SUB015,
        "int8_identity": {
            "d_seg": i8["d_seg"] if i8 else None,
            "d_pose": i8["d_pose"] if i8 else None,
            "archive_bytes": i8["archive_bytes"] if i8 else None,
            "score": i8["score"] if i8 else None,
        },
        "best_ptq_mode": best_mode,
        "best_ptq_S": best["score"],
        "best_ptq_dS_vs_frontier": round(best["score"] - _FRONTIER_S, 5),
        "ptq_beats_frontier": sorted(beats_frontier),  # [] => PTQ collapses (QAT required)
        "ptq_floor_gap_to_perfect_hold": round(gap_to_perfect_hold, 5),
        "qat_distortion_recovery_needed_S": round(gap_to_perfect_hold, 5),
        "score_aware_vs_uniform_matched_bytes": sa_vs_uniform,
    }
    if beats_frontier:
        verdict["headline_part1"] = (
            f"PTQ-ALREADY-BEATS-FRONTIER: modes {sorted(beats_frontier)} score below 0.19110 "
            f"byte-closed (best {best_mode} S={best['score']:.4f}, ΔS {best['score'] - _FRONTIER_S:+.4f}) "
            f"→ POSSIBLE $0 POINTER-MOVER. FLAG for paired dual CPU/CUDA exact eval (do NOT self-promote)."
        )
    else:
        verdict["headline_part1"] = (
            f"PTQ-COLLAPSES: NO grid beats 0.19110 byte-closed (best {best_mode} S={best['score']:.4f}, "
            f"ΔS {best['score'] - _FRONTIER_S:+.4f}) — naive PTQ spills distortion that buries the rate "
            f"win. CONFIRMS score-aware QAT is required (the operator's pivot)."
        )
    verdict["headline_part2"] = (
        f"QAT-GAP: best PTQ S={best['score']:.4f} is {gap_to_perfect_hold:+.4f} from the desk "
        f"int4-perfect-hold 0.1399 — that is the distortion QAT-finetune must recover."
    )
    if sa_vs_uniform:
        wins = [m for m, d in sa_vs_uniform.items() if d["sa_beats_uniform_dS"] < -1e-9]
        detail = "; ".join(
            "{m} dS {ds:+.4f} vs {u}".format(
                m=m,
                ds=sa_vs_uniform[m]["sa_beats_uniform_dS"],
                u=sa_vs_uniform[m]["nearest_uniform"],
            )
            for m in sa_vs_uniform
        )
        beats_word = "beats" if wins else "does NOT beat"
        validates_word = "VALIDATES" if wins else "does not yet validate"
        verdict["headline_part3"] = (
            f"SCORE-AWARE: {beats_word} uniform at matched bytes ({detail})"
            f" → per-stage bit-allocation {validates_word} the taper-on-the-bit-axis insight."
        )

    report = {
        "probe": "ptq_on_frontier_floor",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer_moved": False,
        "frontier_archive": str(FRONTIER_ARCHIVE),
        "frontier_archive_sha256": "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e",
        "decoder_params": n_params,
        "n_pairs": n_pairs,
        "rate_denom": _RATE_DENOM,
        "rows": rows,
        "verdict": verdict,
        "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Bytes MEASURED via the real frontier codec (PR101 split-brotli + CTXR recode, "
            "STORED zip); identity (int8) round-trip is byte-identical to the on-disk frontier "
            "(177169 B). d_seg/d_pose are full-600-pair exact via RealScorerContext.exact_eval "
            "(CPU AUTHORITY). Local CPU ≈ contest-CPU per G3. NON-PROMOTABLE feasibility floor."
        ),
    }
    print("\n=== PTQ-ON-FRONTIER FLOOR VERDICT ===")
    print(f"  {verdict['headline_part1']}")
    print(f"  {verdict['headline_part2']}")
    if "headline_part3" in verdict:
        print(f"  {verdict['headline_part3']}")

    text = json.dumps(report, indent=2)
    out = Path(args.out_json) if args.out_json else Path("reports/ptq_on_frontier_floor.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[wrote] {out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
