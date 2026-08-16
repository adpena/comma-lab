#!/usr/bin/env python3
"""Scorer-free exact pre-proof for the rfo2 rung-2 carrier rank/refit question.

Answers, at $0 and with no scorer, the question #1058 gated: can a rank/precision
cut of the shipped 22,161 B pose carrier supply the rate rung that the sub-0.15
crossing needs, and at what exact carrier-product reconstruction error?

Every candidate is encoded through the EXACT shipped CPR1 codec and the EXACT
shipped Brotli cell, so every byte number is a real coded length, never a model.
The reconstruction error is exact: the receiver's carrier field is bilinear in
the coefficients, so the error of any rank-r approximation is a closed form in
the 12x12 Gram matrix of the RMS-normalised basis at evaluation resolution.

The rank-r REFIT is the least-squares optimum
    c_r = (Br^T Br)^-1 Br^T B c
which is a LOWER BOUND on the error of every rank-r carrier that keeps the
shipped receiver's linear synthesis. A cut that misses the rung under this
bound misses it under every refit heuristic.

Axis: exact coded bytes [MEASURED]; carrier-product MSE [MEASURED, exact
closed form]; d_pose [NOT measured here -- this is the scorer-free pre-gate].
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
AP_ROOT = Path("/Volumes/APDataStore/pact")
DEFAULT_OUTPUT = AP_ROOT / "ddm_ra1_carrier_rank_refit_20260816/retained"

GEN = (
    AP_ROOT
    / "ddm_mp2_mixed_precision_receiver_close_20260815"
    / "generations/hv1_base_control"
)
BASE = GEN / "retained"
OUTER_CARRIER = BASE / "receiver_decode/outer_carrier.bin"
CARRIER_RAW = BASE / "carrier.raw.bin"
CARRIER_BR = BASE / "carrier.br"
ARCHIVE = BASE / "archive.repeat.zip"
CODEC_SOURCE = GEN / "cpr1/carrier_codec.py"           # shipped receiver: decode
ENCODER_SOURCE = REPO / "src/tac/pr130_runtime/fx1_runtime_tree/carrier_codec.py"
# VERIFIED this unit: ENCODER_SOURCE re-encodes the shipped carrier BYTE-IDENTICALLY
# (22,307 B, sha 709ea928c2d73c59...), so candidate byte numbers are real coded
# lengths from the same encoder that produced the frontier bytes.

# Custody: these are the frontier bytes (hv1 ep0634, contest-CUDA T4 n600).
EXPECTED = {
    OUTER_CARRIER: (22_242, "196f0e5136f4d6bfd22c4cf24ad779eee55f6e95a4f5f5994ae09a4fc268b6ef"),
    CARRIER_RAW: (22_219, "065fce08fc3d44e49d29ad624561cbef86d01282cc73dcd32533b5d63115bd9f"),
    CARRIER_BR: (22_161, "fd14aabcb9daa5f1dd1c9c6e63e745a88f2978766e3129b184dd3a9ac7334de0"),
    ARCHIVE: (182_759, "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"),
}

# Receiver constants, read from src/tac/pr130_runtime/fx1_runtime_tree/inflate.py.
N = 600
EVAL_H, EVAL_W = 384, 512
CARRIER_DIM = 12
CARRIER_H, CARRIER_W = 24, 32
CARRIER_AMPLITUDE = 64.0
BASIS_COUNT = CARRIER_DIM * 3 * CARRIER_H * CARRIER_W  # 27,648
BROTLI_QUALITY = 11  # the shipped cell (mp2 exact race: q11 ties the incumbent)

# Score arithmetic, verified against upstream/evaluate.py:63-64 this unit.
UNCOMPRESSED_BYTES = 37_545_489  # == size of upstream/videos/0.mkv
S_PER_BYTE = 25.0 / UNCOMPRESSED_BYTES
FRONTIER_S = 0.15959729295498598
FRONTIER_BYTES = 182_759
# Strict crossing on the hv1 base (integral bytes, target strictly < 0.15).
RUNG_BYTES = 14_414

# PK2's pre-registered scorer-free gate (ddm_pk2 NEXT_IF_RESUMED fire_trigger).
PK2_MSE_GATE = 2.5e-6
PK2_MIN_BYTES_GATE = 2_000

# Populated by decode_reference() with the exact canonical CPR1 carrier bytes.
CANONICAL_CPR1 = b""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_custody() -> dict:
    receipts = {}
    for path, (size, digest) in EXPECTED.items():
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != size or actual_sha != digest:
            raise SystemExit(
                f"CUSTODY FAIL {path}: {actual_size}/{actual_sha} != {size}/{digest}"
            )
        receipts[path.name] = {"bytes": actual_size, "sha256": actual_sha}
    return receipts


def load_codec():
    """Load the SHIPPED F26 receiver codec + CAP1->CPR1 materialiser, read-only."""
    for extra in (str(GEN / "cpr1"), str(GEN)):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    spec = importlib.util.spec_from_file_location("ra1_carrier_codec", CODEC_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    import runtime.carrier_repack as repack

    espec = importlib.util.spec_from_file_location("ra1_encoder", ENCODER_SOURCE)
    encoder = importlib.util.module_from_spec(espec)
    assert espec.loader is not None
    espec.loader.exec_module(encoder)
    return module, repack, encoder


def normalized_basis(raw_basis: torch.Tensor) -> torch.Tensor:
    """Byte-for-byte the receiver's normalized_basis (inflate.py:621-627)."""
    basis = F.interpolate(
        raw_basis, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
    )
    basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
    rms = basis.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
    return basis / rms


def decode_reference(codec, repack):
    """Exact shipped chain: F0C1 split -> CAP1 -> canonical CPR1 -> decode."""
    import types

    shim = types.SimpleNamespace(N=N, CARRIER_DIM=CARRIER_DIM)
    canonical = repack.materialize_cpr1(OUTER_CARRIER.read_bytes(), shim)
    basis_scales, basis_codes, coeff_scales, encoded = codec.decode_compact_carrier(
        canonical, basis_count=BASIS_COUNT, frames=N, dimensions=CARRIER_DIM
    )
    globals()["CANONICAL_CPR1"] = canonical
    # Receiver-exact reconstruction of the signed int12 coefficients
    # (inflate.py:275-281): delta + zigzag + cumulative sum in 12-bit wrap.
    codes = torch.from_numpy(encoded.astype(np.int32, copy=False)).reshape(
        N, CARRIER_DIM
    ) & 0xFFF
    delta = (codes >> 1) ^ -(codes & 1)
    codes = torch.cumsum(delta, dim=0) & 0xFFF
    codes = torch.where(codes >= 0x800, codes - 0x1000, codes)
    return basis_scales, basis_codes, coeff_scales, encoded, codes.numpy()


def encoded_bytes(codec, basis_scales, basis_codes, coeff_scales, encoded) -> dict:
    """Exact coded length through the shipped CPR1 codec + shipped Brotli cell."""
    raw = codec.encode_compact_carrier(
        np.asarray(basis_scales, dtype="<f4"),
        np.asarray(basis_codes, dtype=np.int64),
        np.asarray(coeff_scales, dtype="<f4"),
        np.asarray(encoded, dtype=np.int64),
    )
    comp = brotli.compress(bytes(raw), quality=BROTLI_QUALITY)
    return {"raw_bytes": len(raw), "coded_bytes": len(comp), "raw": bytes(raw), "br": comp}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.time()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    payload_dir = out / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    custody = verify_custody()
    codec, repack, encoder = load_codec()
    basis_scales, basis_codes, coeff_scales, encoded, coeff_codes = decode_reference(
        codec, repack
    )
    # Price every candidate in ONE consistent, real, decodable domain:
    # canonical CPR1 + the shipped Brotli cell. The shipped container (F0C1/CAP1)
    # is 86-117 B tighter on the baseline; that offset is recorded, and it cannot
    # move a verdict whose scale is ~14,000 B.
    roundtrip = bytes(
        encoder.encode_compact_carrier(
            np.asarray(basis_scales, dtype="<f4"),
            np.asarray(basis_codes, dtype=np.int64),
            np.asarray(coeff_scales, dtype="<f4"),
            np.asarray(encoded, dtype=np.int64),
        )
    )
    if roundtrip != CANONICAL_CPR1:
        raise SystemExit("ENCODER FAIL: does not reproduce the shipped carrier bytes")
    baseline_coded = len(brotli.compress(CANONICAL_CPR1, quality=BROTLI_QUALITY))

    # --- exact receiver synthesis geometry -------------------------------
    raw_basis = (
        torch.from_numpy(basis_codes.astype(np.float32))
        .reshape(CARRIER_DIM, 3, CARRIER_H, CARRIER_W)
        * torch.from_numpy(np.asarray(basis_scales, dtype=np.float32))[:, None, None, None]
    )
    Bn = normalized_basis(raw_basis.to(torch.float64))  # (12,3,H,W)
    flat = Bn.reshape(CARRIER_DIM, -1)
    npix = flat.shape[1]
    gram = (flat @ flat.T).numpy() / npix  # mean-per-pixel Gram

    coeff = coeff_codes.astype(np.float64) * np.asarray(coeff_scales, dtype=np.float64)

    # Pixel-space error scale: slave_eval = 127.5 + A * (c . Bn)/sqrt(12).
    pix_scale = (CARRIER_AMPLITUDE ** 2) / CARRIER_DIM

    def field_mse(delta_c: np.ndarray) -> float:
        """Exact mean-squared carrier-field error, in 0-255 pixel units."""
        quad = np.einsum("bi,ij,bj->b", delta_c, gram, delta_c)
        return float(pix_scale * quad.mean())

    ref_energy = field_mse(coeff)  # signal energy in the same units

    # --- the exact least-squares rank-r refit ----------------------------
    rows = []
    order_energy = np.argsort(-(coeff**2).mean(axis=0) * np.diag(gram))

    for r in range(CARRIER_DIM, 0, -1):
        keep = np.sort(order_energy[:r])
        Grr = gram[np.ix_(keep, keep)]
        Grc = gram[np.ix_(keep, np.arange(CARRIER_DIM))]
        # least-squares optimal refit of the kept coefficients
        c_refit = np.linalg.lstsq(Grr, Grc @ coeff.T, rcond=None)[0].T  # (N, r)

        # residual field error of the OPTIMAL rank-r refit (lower bound)
        full = coeff.copy()
        approx = np.zeros_like(full)
        approx[:, keep] = c_refit
        mse_refit = field_mse(full - approx)

        # naive drop (no refit) for contrast
        drop = coeff.copy()
        drop[:, [i for i in range(CARRIER_DIM) if i not in set(keep.tolist())]] = 0.0
        mse_drop = field_mse(full - drop)

        # ---- exact coded bytes of the rank-r carrier -------------------
        # requantise the refit coefficients onto the shipped int12 lattice
        # with a per-dimension scale, then re-encode delta+zigzag exactly.
        sub_scale = np.maximum(np.abs(c_refit).max(axis=0) / 2047.0, 1e-12)
        q = np.clip(np.rint(c_refit / sub_scale), -2048, 2047).astype(np.int64)
        delta = np.diff(np.concatenate([np.zeros((1, r), dtype=np.int64), q]), axis=0)
        zz = ((delta << 1) ^ (delta >> 63)) & 0xFFF

        sub_basis = basis_codes.reshape(CARRIER_DIM, -1)[keep].reshape(-1)
        enc = encoded_bytes(
            encoder,
            np.asarray(basis_scales)[keep],
            sub_basis,
            sub_scale.astype("<f4"),
            zz,
        )
        # quantisation adds error on top of the refit lower bound
        approx_q = np.zeros_like(full)
        approx_q[:, keep] = q.astype(np.float64) * sub_scale
        mse_realised = field_mse(full - approx_q)

        saved = baseline_coded - enc["coded_bytes"]
        name = f"rank{r:02d}_refit"
        (payload_dir / f"{name}.raw.bin").write_bytes(enc["raw"])
        (payload_dir / f"{name}.br").write_bytes(enc["br"])

        rows.append(
            {
                "candidate": name,
                "rank": r,
                "kept_atoms": keep.tolist(),
                "coded_bytes": enc["coded_bytes"],
                "raw_bytes": enc["raw_bytes"],
                "bytes_saved_vs_shipped_carrier": saved,
                "archive_bytes_if_adopted": FRONTIER_BYTES - saved,
                "rate_credit_S": saved * S_PER_BYTE,
                "mse_refit_lower_bound_pixel_units": mse_refit,
                "mse_naive_drop_pixel_units": mse_drop,
                "mse_realised_int12_pixel_units": mse_realised,
                "mse_relative_to_signal_energy": mse_refit / ref_energy,
                "supplies_rung": bool(saved >= RUNG_BYTES),
                "payload_sha256": hashlib.sha256(enc["br"]).hexdigest(),
                "payload_path": str(payload_dir / f"{name}.br"),
            }
        )
        print(
            f"  rank {r:2d}  coded {enc['coded_bytes']:6d} B  saved {saved:+7d} B"
            f"  mse_refit {mse_refit:.6e}  rung={'YES' if saved >= RUNG_BYTES else 'no'}",
            flush=True,
        )

    result = {
        "schema": "ddm_ra1_carrier_rank_refit_preproof.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "axis": "scorer-free exact coded bytes + exact carrier-field MSE; NOT a score",
        "verdict_scope": "the shipped hv1 CPR1 carrier under the receiver's linear synthesis",
        "custody": custody,
        "frontier": {
            "S": FRONTIER_S,
            "archive_bytes": FRONTIER_BYTES,
            "archive_sha256": EXPECTED[ARCHIVE][1],
        },
        "arithmetic": {
            "uncompressed_bytes": UNCOMPRESSED_BYTES,
            "S_per_byte": S_PER_BYTE,
            "rung_bytes_on_hv1_base": RUNG_BYTES,
            "max_archive_for_sub_0_15": FRONTIER_BYTES - RUNG_BYTES,
        },
        "carrier": {
            "shipped_coded_bytes": EXPECTED[CARRIER_BR][0],
            "pricing_domain": "canonical CPR1 + shipped Brotli q11",
            "baseline_coded_bytes_in_pricing_domain": baseline_coded,
            "container_offset_vs_shipped": baseline_coded - EXPECTED[CARRIER_BR][0],
            "canonical_cpr1_bytes": len(CANONICAL_CPR1),
            "raw_bytes": EXPECTED[CARRIER_RAW][0],
            "dim": CARRIER_DIM,
            "basis_count": BASIS_COUNT,
            "signal_energy_pixel_units": ref_energy,
            "gram_diagonal": np.diag(gram).tolist(),
        },
        "pk2_pregate": {"mse": PK2_MSE_GATE, "min_bytes": PK2_MIN_BYTES_GATE},
        "rows": rows,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "brotli": brotli.__version__ if hasattr(brotli, "__version__") else "unknown",
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "elapsed_s": time.time() - started,
    }
    (out / "CARRIER_RANK_REFIT_PREPROOF.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out / 'CARRIER_RANK_REFIT_PREPROOF.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
