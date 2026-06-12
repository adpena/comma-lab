# SPDX-License-Identifier: MIT
"""MED-1 probe — validate Lever-1's entropy surrogate against REAL brotli bytes.

The R1 audit (`layer2_levers_independent_audit_20260612T151829Z.md` MEDIUM-1) flagged
a train/deploy scan-order gap: Lever-1's ``conditional_weight_entropy``
(``src/tac/losses/rate_surrogate.py``) computes an order-1 conditional entropy
``H(W_i | W_{i-1})`` over the FIRST ``pair_sample_size`` contiguous weights of each
Conv2d/Linear tensor, iterated in ``named_modules()`` order. The REAL archive bytes
are produced by the vendored codec ``encode_decoder(quantize_state_dict(sd))`` —
brotli q=11 over the WHOLE ``state_dict`` byte stream (weights AND biases, in
state-dict iteration order, with per-tensor name/shape/scale metadata interleaved).

This probe measures whether the abstract entropy is a PREDICTIVE proxy for real brotli
decoder-blob bytes across several real weight configurations. It is $0 / local /
torch-CPU (TRUSTED). Every number it prints is ``[macOS-CPU advisory]`` NON-PROMOTABLE
— it is a proxy-validation probe, NOT a score claim.

It computes, for each config:
  * ``h_cond`` (legacy)  — the LEGACY per-tensor mode (codec_scan_order=False:
                           named_modules WEIGHTS only, first 2000/tensor). Diagnostic
                           baseline that motivated the MED-1 fix.
  * ``h_cond_full``      — the legacy mode over the FULL tensor (no 2000 truncation).
  * ``h_cond_scan``      — the CODEC-SCAN-ORDER conditional entropy of the actual
                           zigzag-INT8 decoder byte stream the codec brotli-compresses
                           (full state_dict, weights+biases, state-dict order). **This is
                           what the FIX (driver ``codec_scan_order=True``) computes.**
  * ``brotli_bytes``     — the REAL ``len(encode_decoder(quantize_state_dict(sd)))``.

Then it Spearman/Pearson-correlates each proxy against ``brotli_bytes``. A proxy is
PREDICTIVE if its correlation with real brotli bytes is strong + monotone across the
configs (higher entropy → more bytes). The verdict distinguishes:
  * FIX-VALIDATED — the scan-order proxy (what the driver now uses) is a strong rank
    proxy for real brotli bytes AND beats the legacy per-tensor mode → train-time rate
    tracks deploy-time bytes (gap CLOSED via the driver + rate_surrogate fix).
  * INDETERMINATE — even the scan-order proxy is weak → Lever-1 must NOT be claimed as
    a rate lever.

Usage::

    .venv/bin/python experiments/probe_lever1_entropy_vs_real_brotli.py
    .venv/bin/python experiments/probe_lever1_entropy_vs_real_brotli.py --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.losses.rate_surrogate import (
    RateSurrogateConfig,
    conditional_weight_entropy,
)
from tac.torch_vehicle.vendored_imports import import_vendored

_BASIN_CKPT = (
    _REPO_ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
    / "torch_vehicle_checkpoint_state.pt"
)


# --------------------------------------------------------------------------- #
# The codec-scan-order conditional entropy (the exact byte stream brotli sees).
# --------------------------------------------------------------------------- #
def codec_scan_order_conditional_entropy(
    state_dict: dict[str, torch.Tensor],
    codec,
    cfg: RateSurrogateConfig | None = None,
) -> float:
    """Order-1 conditional entropy of the ACTUAL zigzag-INT8 decoder byte stream.

    This reconstructs the exact symbol sequence the codec concatenates (per-tensor
    zigzag-INT8 weights+biases, in ``state_dict`` order — NOT named_modules-weights
    only), then computes the order-1 conditional entropy of the *zigzag uint8* symbol
    stream (the density brotli's LZ/order-N model actually consumes). It is computed
    with NUMPY hard histograms (a measurement, not a training surrogate), so it is the
    fairest non-differentiable estimate of the deployed stream's order-1 redundancy.
    """
    cfg = cfg or RateSurrogateConfig()
    q_sd = codec.quantize_state_dict(state_dict)
    # The exact symbol stream encode_decoder writes (zigzag uint8 of every tensor, in
    # the SAME order encode_decoder iterates q_sd.items()).
    chunks: list[np.ndarray] = []
    for _name, (q, _scale, _shape) in q_sd.items():
        chunks.append(codec.zigzag_encode_i8(q))
    if not chunks:
        return 0.0
    sym = np.concatenate(chunks).astype(np.int64)  # uint8 zigzag symbols 0..254
    if sym.size < 2:
        return 0.0
    prev = sym[:-1]
    cur = sym[1:]
    # Hard 2-D joint histogram over the observed symbol alphabet.
    n_sym = int(sym.max()) + 1
    joint = np.zeros((n_sym, n_sym), dtype=np.float64)
    np.add.at(joint, (prev, cur), 1.0)
    joint /= joint.sum()
    marg_prev = joint.sum(axis=1)  # P(prev=a)
    eps = cfg.eps
    h = 0.0
    for a in range(n_sym):
        pa = marg_prev[a]
        if pa <= 0:
            continue
        cond = joint[a] / pa
        nz = cond > 0
        h_a = -(cond[nz] * np.log2(cond[nz] + eps)).sum()
        h += pa * h_a
    return float(h)


def real_brotli_decoder_bytes(state_dict: dict[str, torch.Tensor], codec) -> int:
    """The REAL brotli decoder-blob byte count: ``len(encode_decoder(quantize(sd)))``."""
    q_sd = codec.quantize_state_dict(state_dict)
    return len(codec.encode_decoder(q_sd))


def real_brotli_full_archive_bytes(
    state_dict: dict[str, torch.Tensor], latents: torch.Tensor, codec
) -> int:
    """The REAL full archive byte count (meta + decoder + latents)."""
    meta = {"base_channels": 20, "latent_dim": int(latents.shape[1])}
    return len(codec.build_archive(state_dict, latents, meta))


# --------------------------------------------------------------------------- #
# Weight-config constructors (real configs that span a range of redundancy).
# --------------------------------------------------------------------------- #
def _build_decoder(codec_model, state_dict: dict[str, torch.Tensor] | None) -> nn.Module:
    dec = codec_model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    if state_dict is not None:
        dec.load_state_dict(state_dict)
    return dec


def _clone_sd(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in sd.items()}


def _sorted_smoothed(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Sort each tensor's flattened weights ascending then reshape — maximal scan-order
    smoothness (long monotone runs → minimal order-1 entropy → minimal brotli bytes)."""
    out = {}
    for k, v in sd.items():
        flat = v.detach().flatten()
        srt, _ = torch.sort(flat)
        out[k] = srt.reshape(v.shape).clone()
    return out


def _shuffled(sd: dict[str, torch.Tensor], seed: int = 0) -> dict[str, torch.Tensor]:
    """Randomly permute each tensor's elements (destroys scan-order runs → higher
    order-1 entropy → more brotli bytes), keeping the marginal histogram identical."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        flat = v.detach().flatten()
        perm = torch.randperm(flat.numel(), generator=g)
        out[k] = flat[perm].reshape(v.shape).clone()
    return out


def _scaled_noise(sd: dict[str, torch.Tensor], scale: float, seed: int) -> dict[str, torch.Tensor]:
    """Add iid Gaussian noise at a fraction of each tensor's std (raises entropy +
    bytes monotonically with ``scale``)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        std = v.detach().float().std().clamp_min(1e-9)
        noise = torch.randn(v.shape, generator=g) * (scale * std)
        out[k] = (v.detach() + noise).clone()
    return out


# --------------------------------------------------------------------------- #
# Correlation helpers (no scipy dependency).
# --------------------------------------------------------------------------- #
def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def _rankdata(a: list[float]) -> list[float]:
    order = sorted(range(len(a)), key=lambda i: a[i])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    if not _BASIN_CKPT.exists():
        raise FileNotFoundError(f"basin checkpoint not found: {_BASIN_CKPT}")

    codec = import_vendored("codec")
    model = import_vendored("model")

    ck = torch.load(_BASIN_CKPT, map_location="cpu", weights_only=False)
    basin_sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}
    basin_latents = ck["ema_latents"].detach().float()

    # Build the config matrix: real trained + degraded/smoothed variants spanning a
    # wide redundancy range so the correlation is meaningful.
    configs: dict[str, dict[str, torch.Tensor]] = {
        "basin_ema": _clone_sd(basin_sd),
        "sorted_smoothed": _sorted_smoothed(basin_sd),
        "shuffled": _shuffled(basin_sd, seed=11),
        "noise_0p10": _scaled_noise(basin_sd, 0.10, seed=1),
        "noise_0p25": _scaled_noise(basin_sd, 0.25, seed=2),
        "noise_0p50": _scaled_noise(basin_sd, 0.50, seed=3),
        "noise_1p00": _scaled_noise(basin_sd, 1.00, seed=4),
        "random_init": _build_decoder(model, None).state_dict(),
    }

    cfg = RateSurrogateConfig()
    rows = []
    for name, sd in configs.items():
        dec = _build_decoder(model, sd)
        with torch.no_grad():
            h_cond = float(conditional_weight_entropy(dec, cfg).item())
            # Full-tensor (no 2000 truncation) variant.
            cfg_full = RateSurrogateConfig(pair_sample_size=10**9)
            h_cond_full = float(conditional_weight_entropy(dec, cfg_full).item())
        h_cond_scan = codec_scan_order_conditional_entropy(sd, codec, cfg)
        brotli_bytes = real_brotli_decoder_bytes(sd, codec)
        full_bytes = real_brotli_full_archive_bytes(sd, basin_latents, codec)
        rows.append(
            {
                "config": name,
                "h_cond_as_wired": round(h_cond, 5),
                "h_cond_full": round(h_cond_full, 5),
                "h_cond_scan_order": round(h_cond_scan, 5),
                "brotli_decoder_bytes": brotli_bytes,
                "full_archive_bytes": full_bytes,
            }
        )

    bytes_vec = [r["brotli_decoder_bytes"] for r in rows]
    corr = {
        "as_wired": {
            "pearson": round(_pearson([r["h_cond_as_wired"] for r in rows], bytes_vec), 4),
            "spearman": round(_spearman([r["h_cond_as_wired"] for r in rows], bytes_vec), 4),
        },
        "full_tensor": {
            "pearson": round(_pearson([r["h_cond_full"] for r in rows], bytes_vec), 4),
            "spearman": round(_spearman([r["h_cond_full"] for r in rows], bytes_vec), 4),
        },
        "scan_order": {
            "pearson": round(_pearson([r["h_cond_scan_order"] for r in rows], bytes_vec), 4),
            "spearman": round(_spearman([r["h_cond_scan_order"] for r in rows], bytes_vec), 4),
        },
    }

    # Verdict logic: a proxy is a VALID training signal if its Spearman (monotone) and
    # Pearson (linear) correlation with real brotli bytes are both strongly positive
    # (>= 0.7). The columns:
    #   * ``as_wired``  = the LEGACY per-tensor mode (codec_scan_order=False): the
    #     diagnostic baseline that motivated the fix. EXPECTED to be a WEAK rank proxy.
    #   * ``scan_order`` = the deploy-faithful conditional entropy of the ACTUAL codec
    #     byte stream — this is what the FIX (driver uses codec_scan_order=True) computes.
    # POST-FIX the driver runs ``scan_order``, so the gate is: does the scan-order proxy
    # clear the bar AND beat the legacy as-wired mode? If yes → FIX-VALIDATED.
    THRESH = 0.70
    aw_s = corr["as_wired"]["spearman"]
    scan_s = corr["scan_order"]["spearman"]
    scan_p = corr["scan_order"]["pearson"]
    scan_ok = (scan_s >= THRESH) and (scan_p >= THRESH)
    scan_beats_legacy = scan_s - aw_s > 0.1
    if scan_ok and scan_beats_legacy:
        verdict = "FIX-VALIDATED"
        verdict_note = (
            f"the deploy-faithful scan-order proxy (codec_scan_order=True, what the "
            f"driver now uses) is a STRONG rank proxy for real brotli decoder bytes "
            f"(Spearman={scan_s} Pearson={scan_p} >= {THRESH}); the LEGACY per-tensor "
            f"mode was WEAK (Spearman={aw_s}). The MED-1 fix (driver uses "
            f"codec_scan_order=True) makes train-time rate track deploy-time bytes."
        )
    elif scan_ok:
        verdict = "FIX-VALIDATED-MARGINAL"
        verdict_note = (
            f"scan-order proxy clears the bar (Spearman={scan_s} Pearson={scan_p}) but "
            f"the legacy mode was not much worse (Spearman={aw_s}); the fix is still "
            "correct (scan-order is the deploy-faithful density) but the gap is small "
            "on these configs."
        )
    else:
        verdict = "INDETERMINATE"
        verdict_note = (
            f"the scan-order proxy (Spearman={scan_s} Pearson={scan_p}) does not clear "
            f"{THRESH}; the entropy↔bytes link is weak across these configs — Lever-1 "
            "should NOT be claimed as a rate lever on this basis."
        )

    result = {
        "probe": "lever1_entropy_vs_real_brotli",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE (proxy-validation probe)",
        "basin_checkpoint": str(_BASIN_CKPT.relative_to(_REPO_ROOT)),
        "rows": rows,
        "correlations_vs_brotli_decoder_bytes": corr,
        "threshold": THRESH,
        "verdict": verdict,
        "verdict_note": verdict_note,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 78)
        print("MED-1 PROBE — Lever-1 entropy surrogate vs REAL brotli decoder bytes")
        print("=" * 78)
        print(f"basin: {result['basin_checkpoint']}")
        print(f"authority: {result['authority']}")
        print()
        hdr = f"{'config':<16}{'h_cond(legacy)':>16}{'h_cond(full)':>14}{'h_scan(FIX)':>13}{'dec_bytes':>11}{'arc_bytes':>11}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(
                f"{r['config']:<16}{r['h_cond_as_wired']:>16}{r['h_cond_full']:>14}"
                f"{r['h_cond_scan_order']:>13}{r['brotli_decoder_bytes']:>11}{r['full_archive_bytes']:>11}"
            )
        print()
        print("Correlation vs real brotli decoder bytes:")
        for k, v in corr.items():
            print(f"  {k:<12} Pearson={v['pearson']:>7}  Spearman={v['spearman']:>7}")
        print()
        print(f"VERDICT: {verdict}")
        print(f"  {verdict_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
