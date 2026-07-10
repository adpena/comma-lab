#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""CLI for the exact-score-gated latent CLICK-POLISH SEARCH (task #399).

HONEST NAMING (NO-FAKE #6): this is a search/polish, not a solver. See
``tac.click_polish`` for the full mechanism + borrowed-substrate accounting.

Subcommands:
  verify         codec round-trip + fold-sidecar byte custody (fast, no GPU/GT).
  verify-render  pair-locality + diagonal-batch==sequential equivalence (needs
                 scorers + a small GT cache; the render/scorer greens).
  estimate       timing smoke: measure sec/render at a small n, extrapolate n600.
  smoke          tiny end-to-end search on n4 pairs [macOS-CPU advisory].
  search         full n600 exact-gated search (phase-2 Modal CPU / Linux x86_64).

Axis discipline: selection runs scorers on CPU only. macOS rows are
[macOS-CPU advisory] NON-PROMOTABLE; only upstream/evaluate.py on Linux x86_64 over
the exact candidate bytes is [contest-CPU] authority (run separately / in-dispatch).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# repo root on path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tac import click_polish as cp  # noqa: E402


def _packet(args):
    return cp.FrozenPacket.parse(args.archive, args.submission_dir)


def cmd_verify(args):
    pkt = _packet(args)
    rt = pkt.verify_roundtrip()
    fold = pkt.fold_sidecar_custody()
    out = {"roundtrip": rt, "fold_sidecar_custody": fold}
    print(json.dumps(out, indent=2))
    ok = (
        rt["latent_raw_roundtrip_byte_exact"]
        and rt["member_byte_exact"]
        and rt["archive_byte_exact"]
        and fold["delta_matches_sidecar"]
    )
    print(f"\nVERIFY {'GREEN' if ok else 'RED'}: "
          f"roundtrip byte-exact={rt['archive_byte_exact']} "
          f"fold delta={fold['archive_byte_delta']} (expect {fold['expected_delta']})")
    return 0 if ok else 1


def cmd_verify_render(args):
    pkt = _packet(args)
    rnd = cp.Renderer(pkt, device=args.device)
    scorer = cp.Scorer(upstream_dir=args.upstream, device=args.device)
    n = max(4, args.n)
    lstars, poses, gtsrc = cp.load_gt_targets(args.gt_cache, n)
    loc = cp.verify_pair_locality(pkt, rnd)
    beq = cp.verify_batch_equivalence(pkt, rnd, scorer, lstars, poses, pairs=tuple(range(4)))
    out = {"gt_source": gtsrc, "locality": loc, "batch_equivalence": beq}
    print(json.dumps(out, indent=2, default=float))
    ok = loc["locality_holds"] and beq["equivalence_holds"]
    print(f"\nVERIFY-RENDER {'GREEN' if ok else 'RED'}: "
          f"locality={loc['locality_holds']} batch_equiv={beq['equivalence_holds']}")
    return 0 if ok else 1


def cmd_estimate(args):
    pkt = _packet(args)
    rnd = cp.Renderer(pkt, device=args.device)
    scorer = cp.Scorer(upstream_dir=args.upstream, device=args.device)
    n = args.n
    lstars, poses, _ = cp.load_gt_targets(args.gt_cache, n)
    pairs = list(range(n))
    t0 = time.time()
    frames = rnd.render(pkt.Q0, pairs)
    scorer.per_pair(frames, lstars, poses)
    dt = time.time() - t0
    per_pair = dt / n
    # a full sweep = len(SWEEP_DELTAS)*LATENT_DIM renders of n600 + 1 baseline
    renders_per_round = len(cp.SWEEP_DELTAS) * cp.LATENT_DIM + 1
    sec_per_render_600 = per_pair * 600
    sec_per_round = renders_per_round * sec_per_render_600
    out = {
        "n_timed": n, "render+score_sec": dt, "sec_per_pair": per_pair,
        "renders_per_round": renders_per_round,
        "est_sec_per_render_n600": sec_per_render_600,
        "est_sec_per_round_n600": sec_per_round,
        "est_hours_per_round_n600": sec_per_round / 3600.0,
        "note": "wall-clock at n600 scales ~linearly in pairs; Modal CPU has more cores "
                "than this macOS timing, so treat as an UPPER bound.",
    }
    print(json.dumps(out, indent=2))
    return 0


def _run_search(args, n):
    pkt = _packet(args)
    rnd = cp.Renderer(pkt, device=args.device, drop_sidecar=args.drop_sidecar)
    scorer = cp.Scorer(upstream_dir=args.upstream, device=args.device)
    lstars, poses, gtsrc = cp.load_gt_targets(args.gt_cache, n)
    search = cp.ClickPolishSearch(
        packet=pkt, renderer=rnd, scorer=scorer, gt_lstars=lstars, gt_poses=poses,
        out_dir=args.out_dir, axis_tag=args.axis_tag, max_rounds=args.max_rounds,
    )
    result = search.run()
    result["gt_source"] = gtsrc
    result["n_pairs"] = n
    with open(os.path.join(args.out_dir, "search_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))
    return 0


def cmd_smoke(args):
    args.axis_tag = "[macOS-CPU advisory]"
    return _run_search(args, n=4)


def cmd_search(args):
    return _run_search(args, n=args.n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default=os.path.join(ROOT, cp.DEFAULT_ARCHIVE))
    ap.add_argument("--submission-dir", default=os.path.join(ROOT, cp.DEFAULT_SUBMISSION_DIR))
    ap.add_argument("--upstream", default=os.path.join(ROOT, cp.DEFAULT_UPSTREAM))
    ap.add_argument("--gt-cache", default=os.path.join(ROOT, cp.DEFAULT_GT_CACHE))
    ap.add_argument("--device", default="cpu", choices=["cpu"],
                    help="CPU only — selection axis (MPS/CUDA never an authority here)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("verify"); sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("verify-render")
    sp.add_argument("--n", type=int, default=6)
    sp.set_defaults(func=cmd_verify_render)

    sp = sub.add_parser("estimate")
    sp.add_argument("--n", type=int, default=6)
    sp.set_defaults(func=cmd_estimate)

    sp = sub.add_parser("smoke")
    sp.add_argument("--out-dir", default=os.path.join(ROOT, ".omx/tmp/clickpolish_smoke"))
    sp.add_argument("--max-rounds", type=int, default=3)
    sp.add_argument("--drop-sidecar", action="store_true")
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("search")
    sp.add_argument("--n", type=int, default=600)
    sp.add_argument("--out-dir", default=os.path.join(ROOT, "experiments/results/clickpolish_pr110_20260710"))
    sp.add_argument("--max-rounds", type=int, default=40)
    sp.add_argument("--axis-tag", default="[macOS-CPU advisory]")
    sp.add_argument("--drop-sidecar", action="store_true")
    sp.set_defaults(func=cmd_search)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
