# SPDX-License-Identifier: MIT
"""Measure the REAL byte-closed capstone archive across decoder sizes + dtypes.

Replaces the prior memo's §4 PROJECTION with measured len(brotli(...)) bytes
from the actual build_capstone_archive_bytes, at base_channels in {36,24,20,16}
and decoder_dtype in {fp16,int8}. The decoder weights are the actual MLX bundle
params (untrained values — the rate is structure/dtype-bound, the magnitude is
what the codec measures). Emits a JSON table.
"""

from __future__ import annotations

import json

import numpy as np

from tac.capstone_vq_nerv.export import build_capstone_archive_bytes
from tac.capstone_vq_nerv.vq_nerv_bundle import (
    CapstoneVqNervBundle,
    CapstoneVqNervConfig,
)

CONTEST_SOURCE_BYTES = 37_545_489
NUM_PAIRS = 600


def _bundle_arrays(base_channels: int):
    cfg = CapstoneVqNervConfig(num_pairs=NUM_PAIRS, base_channels=base_channels)
    bundle = CapstoneVqNervBundle(cfg)
    from mlx.utils import tree_flatten

    flat = tree_flatten(bundle.trainable_parameters())
    decoder_weights = {}
    for k, v in flat:
        # decoder backbone + FiLM are the "free basis"; latents are NOT stored
        # (the VQ index is the carrier). Codebook stored separately.
        if k.startswith("latents"):
            continue
        decoder_weights[k] = np.asarray(v, dtype=np.float32)
    codebook = np.asarray(bundle.quantizer._codebook, dtype=np.float32)
    # deterministic synthetic indices + pose (rate is index-width/pose-count bound).
    rng = np.random.default_rng(0)
    vq_indices = rng.integers(0, cfg.codebook_size, size=NUM_PAIRS).astype(np.int32)
    pose = rng.standard_normal((NUM_PAIRS, 6)).astype(np.float32)
    return decoder_weights, codebook, vq_indices, pose, cfg.codebook_size


def main() -> int:
    rows = []
    for base_channels in (36, 24, 20, 16):
        dec_w, cb, idx, pose, K = _bundle_arrays(base_channels)
        dec_param_count = sum(int(a.size) for a in dec_w.values())
        for dtype in ("fp16", "int8"):
            archive, account = build_capstone_archive_bytes(
                decoder_weights=dec_w,
                codebook=cb,
                vq_indices=idx,
                pose_scalars=pose,
                codebook_size=K,
                decoder_dtype=dtype,
            )
            rate = 25.0 * account.total_bytes / CONTEST_SOURCE_BYTES
            rows.append(
                {
                    "base_channels": base_channels,
                    "decoder_params": dec_param_count,
                    "decoder_dtype": dtype,
                    "decoder_bytes": account.decoder_bytes,
                    "codebook_bytes": account.codebook_bytes,
                    "index_bytes": account.index_bytes,
                    "pose_bytes": account.pose_bytes,
                    "total_bytes": account.total_bytes,
                    "rate_term": round(rate, 5),
                    "bytes_per_decoder_param": round(
                        account.decoder_bytes / max(dec_param_count, 1), 3
                    ),
                }
            )
    out = {
        "schema": "capstone_byte_budget_measured.v1",
        "axis_tag": "[exact byte measurement]",
        "contest_source_bytes": CONTEST_SOURCE_BYTES,
        "num_pairs": NUM_PAIRS,
        "rows": rows,
    }
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="", help="optional JSON output path (durable, NOT /tmp)")
    args = ap.parse_args()
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
