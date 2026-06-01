#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 macOS-CPU advisory CLI for the SNeRV inverse-steganalysis carrier.

Runs the complete byte-closed SNeRV stack on REAL ``upstream/videos/0.mkv`` frames
and emits a NON-PROMOTABLE ``[macOS-CPU advisory]`` JSON (Catalog #341/#192/#127/
#323). Reports the achieved rate term (bytes), advisory d_seg/d_pose, the
Z8-falsification verdict, the G3 DWT-adjoint exactness number, and the L-inf-vs-L2
score delta.

NO paid dispatch, NO cloud GPU, NO PR, NO MPS-as-authority. The scorer is the
bit-exact CPU mirror (offline oracle + advisory re-measure only); it never crosses
the receiver boundary.

Usage:
    .venv/bin/python tools/run_snerv_inverse_steg_advisory.py \
        --n-pairs 4 --levels 3 --bits-per-coeff 2.5 \
        --out .omx/research/snerv_inverse_steg_advisory_<utc>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.snerv_inverse_steg_carrier.advisory import (  # noqa: E402  (sys.path bootstrap above)
    run_snerv_advisory,
)


def _default_out() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f".omx/research/snerv_inverse_steg_advisory_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--bits-per-coeff", type=float, default=2.5)
    ap.add_argument("--wavelet", type=str, default="db2")
    ap.add_argument("--pair-stride", type=int, default=1)
    ap.add_argument("--start-pair", type=int, default=0)
    ap.add_argument("--pr101-frontier-bytes", type=int, default=178_493)
    ap.add_argument("--video-path", type=str, default="upstream/videos/0.mkv")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    res = run_snerv_advisory(
        n_pairs=args.n_pairs,
        levels=args.levels,
        wavelet=args.wavelet,
        target_bits_per_coeff=args.bits_per_coeff,
        pair_stride=args.pair_stride,
        start_pair=args.start_pair,
        pr101_frontier_bytes=args.pr101_frontier_bytes,
        video_path=args.video_path,
    )
    payload = res.as_jsonable()
    out_path = Path(args.out or _default_out())
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"[SNeRV advisory] {res.axis_tag} NON-PROMOTABLE (Catalog #341/#192/#127/#323)")
    print(f"  carrier {res.carrier_hw}  n_pairs={res.n_pairs} levels={res.levels} {res.wavelet}")
    print(f"  G3 adjoint rel-residual = {res.adjoint_rel_residual:.3e} (exact iff < 1e-12)")
    print(f"  LF stored coeffs = {res.lf_coeff_count_total}  LF payload = {res.lf_payload_bytes} B")
    print(f"  decoder = {res.decoder_bytes} B  archive_total = {res.archive_bytes_total} B")
    print(f"  rate_term = {res.rate_term:.5f}  (frontier {res.pr101_frontier_bytes} B = {res.pr101_frontier_rate:.5f})")
    print(f"  beats_frontier_rate = {res.beats_frontier_rate}")
    print(f"  d_seg(linf) = {res.d_seg_mean_linf:.5f}  d_pose(linf) = {res.d_pose_mean_linf:.5f}  score_linf = {res.score_linf:.5f}")
    print(f"  d_seg(l2)   = {res.d_seg_mean_l2:.5f}  d_pose(l2)   = {res.d_pose_mean_l2:.5f}  score_l2   = {res.score_l2:.5f}")
    print(f"  Z8 detail-store-frac = {res.z8_disease_detail_store_frac:.3f}")
    print(f"  Z8 falsification: {res.z8_falsification_verdict}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
