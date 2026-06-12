# SPDX-License-Identifier: MIT
"""Measure the Cool-Chic AR-Gaussian entropy coder on the real byte-closed smoke.

Re-encodes the trained latents from a CCV1 smoke ``0.bin`` with the REAL AR-Gaussian
entropy coder (CCV2) and reports the apples-to-apples byte reduction + lossless
round-trip confirmation + the new total archive size + the advisory rate term.

This is the deliverable headline. Evidence grade: ``[contest-CPU advisory]`` —
NON-PROMOTABLE (no scorer eval here; the byte count + losslessness are exact, but
no contest score is claimed; the rate term is advisory).

Usage:
    .venv/bin/python experiments/measure_cool_chic_ar_entropy_coder.py \
        --smoke-bin experiments/results/cool_chic_mps_smoke_20260612T121501Z/0.bin \
        --bits-per-std 5.0 --out-json experiments/results/<...>/ar_entropy_measure.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import torch

from tac.substrates.cool_chic.architecture import CoolChicConfig, CoolChicSubstrate
from tac.substrates.cool_chic.archive import (
    CCV1_HEADER_FMT,
    CCV1_HEADER_SIZE,
    pack_archive_ccv2,
    parse_archive,
)
from tac.substrates.cool_chic.entropy_coder import (
    LatentGrid,
    choose_grid_step,
    encode_latent_chain,
)

CONTEST_N = 37_545_489.0


def _rebuild_priors(arc, ar_hidden: int):
    cfg = CoolChicConfig(num_pairs=int(arc.latents_coarse.shape[0]), ar_prior_hidden=ar_hidden)
    m = CoolChicSubstrate(cfg).eval()
    m.ar_prior_coarse.load_state_dict(
        {k.replace("coarse.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("coarse.")},
        strict=False,
    )
    m.ar_prior_fine.load_state_dict(
        {k.replace("fine.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("fine.")},
        strict=False,
    )
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-bin", required=True)
    p.add_argument("--bits-per-std", type=float, default=5.0)
    p.add_argument("--out-json", default=None)
    p.add_argument("--verify-decode", action="store_true", help="full CCV2 parse round-trip (slower)")
    args = p.parse_args()

    raw = Path(args.smoke_bin).read_bytes()
    header = struct.unpack(CCV1_HEADER_FMT, raw[:CCV1_HEADER_SIZE])
    meta_len = header[-1]
    raw_meta = json.loads(raw[-meta_len:].decode("utf-8"))
    ar_hidden = int(raw_meta["ar_prior_hidden"])

    arc = parse_archive(raw)  # CCV1 dequantized latents (the trained latents)
    m = _rebuild_priors(arc, ar_hidden)
    clean_meta = {k: v for k, v in raw_meta.items() if not k.startswith("_quant")}

    lc = arc.latents_coarse.to(torch.float32)
    lf = arc.latents_fine.to(torch.float32)

    # --- apples-to-apples: raw-int16-at-the-SAME-grid vs AR-coded-at-the-same-grid
    step_c = choose_grid_step(lc, bits_per_std=args.bits_per_std)
    step_f = choose_grid_step(lf, bits_per_std=args.bits_per_std)
    grid_c, grid_f = LatentGrid(step_c), LatentGrid(step_f)
    k_c = grid_c.quantize(lc)
    k_f = grid_f.quantize(lf)

    def _fixed_width_bytes(k: torch.Tensor) -> int:
        krange = int(k.max() - k.min()) + 1
        bits = max(1, math.ceil(math.log2(max(2, krange))))
        return math.ceil(k.numel() * bits / 8)

    fixed_c = _fixed_width_bytes(k_c)
    fixed_f = _fixed_width_bytes(k_f)
    int16_c = lc.numel() * 2
    int16_f = lf.numel() * 2

    t0 = time.time()
    coded_c = encode_latent_chain(lc, m.ar_prior_coarse, grid_c)
    t_c = time.time() - t0
    t0 = time.time()
    coded_f = encode_latent_chain(lf, m.ar_prior_fine, grid_f)
    t_f = time.time() - t0
    ar_c, ar_f = len(coded_c), len(coded_f)

    # --- full CCV2 archive (the real new 0.bin) + lossless verify
    ccv2 = pack_archive_ccv2(
        arc.synthesis_state_dict, arc.ar_prior_state_dict, lc, lf, clean_meta,
        grid_step_coarse=step_c, grid_step_fine=step_f,
    )
    lossless = None
    if args.verify_decode:
        arc2 = parse_archive(ccv2)
        lossless = (
            bool(torch.equal(grid_c.quantize(lc), grid_c.quantize(arc2.latents_coarse)))
            and bool(torch.equal(grid_f.quantize(lf), grid_f.quantize(arc2.latents_fine)))
        )

    result = {
        "schema": "cool_chic_ar_entropy_measure_v1",
        "evidence_grade": "contest-CPU advisory",
        "promotable": False,
        "score_claim": False,
        "smoke_bin": str(args.smoke_bin),
        "bits_per_std": args.bits_per_std,
        "grid_step_coarse": step_c,
        "grid_step_fine": step_f,
        # raw int16 (the L0 grammar baseline) per section
        "raw_int16_coarse_bytes": int16_c,
        "raw_int16_fine_bytes": int16_f,
        "raw_int16_latent_bytes": int16_c + int16_f,
        # fixed-width-at-this-grid (isolates entropy gain from requant gain)
        "fixed_width_coarse_bytes": fixed_c,
        "fixed_width_fine_bytes": fixed_f,
        "fixed_width_latent_bytes": fixed_c + fixed_f,
        # AR-entropy-coded
        "ar_coded_coarse_bytes": ar_c,
        "ar_coded_fine_bytes": ar_f,
        "ar_coded_latent_bytes": ar_c + ar_f,
        # reductions
        "reduction_vs_int16_pct": (1 - (ar_c + ar_f) / (int16_c + int16_f)) * 100,
        "reduction_vs_int16_combined_note": "requant(16->grid) + AR entropy, both lossless-to-grid",
        "pure_entropy_gain_vs_fixed_width_pct": (1 - (ar_c + ar_f) / (fixed_c + fixed_f)) * 100,
        "pure_entropy_gain_note": "AR entropy only, same grid as fixed-width baseline",
        # totals
        "ccv1_total_bytes": len(raw),
        "ccv2_total_bytes": len(ccv2),
        "total_reduction_pct": (1 - len(ccv2) / len(raw)) * 100,
        # advisory rate terms (25*B/N)
        "ccv1_advisory_rate_term": 25 * len(raw) / CONTEST_N,
        "ccv2_advisory_rate_term": 25 * len(ccv2) / CONTEST_N,
        # losslessness
        "lossless_roundtrip_verified": lossless,
        "encode_seconds_coarse": t_c,
        "encode_seconds_fine": t_f,
    }
    print(json.dumps(result, indent=2))
    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[measure] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
