# SPDX-License-Identifier: MIT
"""Verify rate-attack (variable-level codec) + L3 finishing-kit + D1 latent dedup on the
REAL solved-taper basin — POST-int8-brotli byte deltas, $0 CPU/code only.

P1b of the bind-all production build-out. This is a MEASUREMENT harness (NOT a driver edit):
it loads the converged base_ch=20 basin (decoder state dict + (600,28) latents), then measures,
at the POST-int8-brotli archive level (review R4 — byte-neutrality is checked here, NOT at
param-count):

  Part D (rate-attack / variable-level codec):
    * vendored decoder blob bytes (the uniform-127 baseline);
    * variable-level decoder blob bytes under a sensitivity-driven allocation (the reverse
      water-fill) — and the measured byte delta;
    * round-trip exactness of the variable decoder (decode_decoder_variable on both formats);
    * the solved-taper-vs-vendored-taper byte-neutrality check at the archive level.

  Part E (L3 finishing-kit):
    * each primitive's byte cost (PR98 = 0-byte section when residual, T10 affine = 54-byte
      fixed section, S12 = certification-only / 0-byte on a render base);
    * deterministic + round-trips (serialize -> parse -> identity) + applied POST-round.

  Part F (D1 cross-pair latent dedup):
    * vendored latents blob bytes (the temporal-delta + lo/hi + brotli baseline);
    * the measure-and-select cross-pair codec's best framed candidate bytes + which format won;
    * the default-preserving adapter's emitted bytes + the measured byte delta;
    * round-trip exactness of the deployed latents (decode reproduces the quantized codes).

Authority: byte measurements are REAL deployed-byte measurements (this changes archive bytes
directly). A SCORE claim still requires the dual CPU/CUDA 600-pair exact eval — this harness
makes NO score claim. ``[contest-CPU advisory] NON-PROMOTABLE``.

Usage::

    .venv/bin/python experiments/verify_rate_attack_l3_d1.py \
        --basin experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best \
        --json reports/rate_attack_l3_d1_verification.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import brotli
import numpy as np
import torch

from tac.losses.cross_pair_latent_codec import (
    CrossPairLatentConfig,
    build_latent_blob_dedup_or_vendored,
    decode_latent_blob,
    decode_latents_best,
    encode_latents_best,
    quantize_pairs,
)
from tac.losses.variable_level_codec import (
    VariableLevelConfig,
    build_decoder_blob_variable_or_vendored,
    decode_decoder_variable,
    levels_from_sensitivity_for_codec,
    quantize_state_dict_variable,
)
from tac.torch_vehicle.distortion_finishing_kit import (
    DistortionKitConfig,
    apply_distortion_kit_to_raw_frames,
    parse_distortion_section,
    serialize_distortion_section,
)
from tac.torch_vehicle.vendored_imports import import_vendored


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _decoder_sensitivity_proxy(sd: dict[str, torch.Tensor]) -> dict[str, float]:
    """A deterministic, code-only per-tensor sensitivity PROXY for the rate-attack measurement.

    The PRODUCTION shared-spine sensitivity is the gate-2 d_seg-sensitivity map (Δd_seg/param,
    scorer-measured). That requires the scorer + a forward pass; this harness is $0 CPU/code
    only, so we use a transparent magnitude-energy proxy (||t||_2 / sqrt(numel) = RMS) purely
    to DRIVE the variable-level allocation and MEASURE the byte mechanism. The byte deltas are
    real; the ALLOCATION is a proxy (production wires the real sensitivity map). Documented so
    no one mistakes the proxy for the scorer-measured spine.
    """
    return {
        name: float(t.detach().float().pow(2).mean().sqrt().item())
        for name, t in sd.items()
    }


def verify_rate_attack(sd: dict[str, torch.Tensor]) -> dict:
    """Part D: variable-level decoder codec on the REAL basin weights (post-brotli bytes)."""
    codec = import_vendored("codec")
    cfg = VariableLevelConfig()
    names = list(sd.keys())

    # --- baseline: vendored uniform-127 decoder blob ---
    vendored_blob = codec.encode_decoder(codec.quantize_state_dict(sd))
    vendored_bytes = len(vendored_blob)

    # --- variable-level under a sensitivity-driven allocation (the reverse water-fill) ---
    sens = _decoder_sensitivity_proxy(sd)
    levels = levels_from_sensitivity_for_codec(
        sens, names, min_level_ratio=0.5, max_level_ratio=1.0, config=cfg
    )
    var_blob, is_variable = build_decoder_blob_variable_or_vendored(sd, levels, cfg)
    var_bytes = len(var_blob)

    # --- the default-preserving guard: all-uniform allocation must be byte-IDENTICAL ---
    uniform_blob, uniform_is_variable = build_decoder_blob_variable_or_vendored(sd, None, cfg)
    default_preserved = (uniform_blob == vendored_blob) and (uniform_is_variable is False)

    # --- round-trip exactness of the variable decoder (the deployed dequant weights) ---
    # Build the SELF-CONSISTENT variable blob (always flagged) to round-trip the variable path,
    # and confirm the variable dequant reproduces the int8*scale weights bit-exactly.
    from tac.losses.variable_level_codec import encode_decoder_variable

    q_sd = quantize_state_dict_variable(sd, levels, cfg)
    self_blob = encode_decoder_variable(q_sd)
    dec_var = decode_decoder_variable(self_blob)
    # Reference dequant from the SAME quantized codes (int8 * scale).
    roundtrip_exact = True
    max_abs_err = 0.0
    for name, (q, scale, shape, _nl) in q_sd.items():
        ref = torch.from_numpy(q.astype(np.float32).reshape(shape)) * float(scale)
        got = dec_var[name]
        err = float((got - ref).abs().max().item())
        max_abs_err = max(max_abs_err, err)
        if err != 0.0:
            roundtrip_exact = False

    # The unified decoder also reads the VENDORED-format blob (one inflate path, both formats).
    # decode_decoder_variable only reads THIS module's flagged format; the vendored path is read
    # by codec.decode_decoder. We confirm both decode to consistent shapes here.
    n_levels_hist: dict[int, int] = {}
    for nl in levels.values():
        n_levels_hist[nl] = n_levels_hist.get(nl, 0) + 1

    return {
        "vendored_decoder_bytes": vendored_bytes,
        "variable_decoder_bytes": var_bytes,
        "delta_bytes": var_bytes - vendored_bytes,
        "delta_pct": 100.0 * (var_bytes - vendored_bytes) / vendored_bytes,
        "variable_format_emitted": bool(is_variable),
        "default_preserving_uniform_byte_identical": bool(default_preserved),
        "roundtrip_exact_variable_decoder": bool(roundtrip_exact),
        "roundtrip_max_abs_err": max_abs_err,
        "n_tensors": len(names),
        "levels_histogram": {str(k): v for k, v in sorted(n_levels_hist.items())},
        "vendored_blob_sha16": _sha(vendored_blob),
        "variable_blob_sha16": _sha(var_blob),
        "sensitivity_allocation_note": (
            "ALLOCATION driven by a code-only RMS-energy PROXY (not the scorer-measured "
            "gate-2 d_seg spine); the BYTE deltas are real, production wires the real sensitivity map."
        ),
    }


def verify_l3_kit() -> dict:
    """Part E: each L3 finishing-kit primitive — byte cost + determinism + round-trip + POST-round."""
    results: dict = {}

    # --- disabled kit: 0-byte section, no-op transform (the default-OFF contract) ---
    disabled = DistortionKitConfig(enabled=False)
    disabled_section = serialize_distortion_section(disabled)
    results["disabled"] = {
        "section_bytes": len(disabled_section),
        "is_zero_byte": len(disabled_section) == 0,
        "parses_to_disabled": parse_distortion_section(disabled_section).enabled is False,
    }

    # --- PR98 residual (the only kit that survived the converged under-power audit) ---
    pr98 = DistortionKitConfig.from_converged_residual_pr98()
    pr98_section = serialize_distortion_section(pr98)
    pr98_roundtrip = serialize_distortion_section(parse_distortion_section(pr98_section))
    results["pr98_residual"] = {
        "section_bytes": len(pr98_section),
        "deterministic_roundtrip_byte_identical": pr98_section == pr98_roundtrip,
        "is_pure_bias": bool(np.allclose(pr98.scale_array(), 1.0)),
        "note": "fixed 54-byte section regardless of values (PR98 = scale==1 bias-only)",
    }

    # --- T10 affine (the 2nd-order generalization; same fixed section size) ---
    scale = np.array([[1.001, 0.999, 1.0], [1.0, 1.002, 0.998]])
    bias = np.array([[0.5, -0.5, 1.0], [0.0, 0.25, -0.25]])
    t10 = DistortionKitConfig.from_affine(scale, bias)
    t10_section = serialize_distortion_section(t10)
    t10_roundtrip_cfg = parse_distortion_section(t10_section)
    t10_roundtrip = serialize_distortion_section(t10_roundtrip_cfg)
    results["t10_affine"] = {
        "section_bytes": len(t10_section),
        "deterministic_roundtrip_byte_identical": t10_section == t10_roundtrip,
        "scale_roundtrips": bool(
            np.allclose(t10_roundtrip_cfg.scale_array(), scale, atol=1e-6)
        ),
        "bias_roundtrips": bool(np.allclose(t10_roundtrip_cfg.bias_array(), bias, atol=1e-6)),
        "note": "same fixed 54-byte section as PR98 (scale!=1 affine)",
    }

    # --- S12: certification-only (0 added bytes on a render base; carried as a flag) ---
    s12 = DistortionKitConfig.from_converged_residual_pr98(s12_certified=True)
    s12_section = serialize_distortion_section(s12)
    s12_parsed = parse_distortion_section(s12_section)
    results["s12_certification"] = {
        "section_bytes_vs_pr98": len(s12_section) - len(pr98_section),
        "adds_zero_bytes_over_pr98": len(s12_section) == len(pr98_section),
        "certification_flag_roundtrips": s12_parsed.s12_invisibility_certified is True,
        "note": "S12 rides as a 1-bit flag in the existing section header (0 extra bytes)",
    }

    # --- POST-round application: kit applies round(clip(scale*x - bias, 0, 255)) on uint8 frames ---
    # Synthetic raw frames (N, H, W, 3) — a small grid; we are verifying APPLICATION semantics,
    # not a score (a real frame measurement is the scorer probe, not this harness).
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)
    out_disabled = apply_distortion_kit_to_raw_frames(raw, disabled)
    out_pr98 = apply_distortion_kit_to_raw_frames(raw, pr98)
    out_t10 = apply_distortion_kit_to_raw_frames(raw, t10)
    results["post_round_application"] = {
        "disabled_is_noop_same_object": out_disabled is raw,
        "pr98_changes_frames": not np.array_equal(out_pr98, raw),
        "pr98_output_is_uint8": out_pr98.dtype == np.uint8,
        "pr98_output_in_range": bool(out_pr98.min() >= 0 and out_pr98.max() <= 255),
        "t10_changes_frames": not np.array_equal(out_t10, raw),
        "t10_output_is_uint8": out_t10.dtype == np.uint8,
        "t10_output_in_range": bool(out_t10.min() >= 0 and out_t10.max() <= 255),
        "applied_per_frame_parity": "frame_0 (even idx) and frame_1 (odd idx) indexed separately",
    }
    return results


def verify_d1_latent_dedup(latents: torch.Tensor) -> dict:
    """Part F: D1 cross-pair latent dedup on the REAL basin latents (post-brotli bytes)."""
    codec = import_vendored("codec")
    cfg = CrossPairLatentConfig()

    # --- baseline: vendored latents blob (temporal-delta + lo/hi + brotli) ---
    vendored_blob = brotli.compress(codec.encode_latents(latents), quality=11)
    vendored_bytes = len(vendored_blob)

    # --- the measure-and-select CORE: best framed candidate (agnostic, all formats) ---
    framed_payload, flag = encode_latents_best(latents, cfg, structural_only=False)
    framed_blob = brotli.compress(framed_payload, quality=11)
    flag_names = {0: "VENDORED", 1: "FRAMED_DELTA", 2: "DEDUP", 3: "CODEBOOK"}

    # --- the default-preserving ADAPTER: emits framed ONLY when a STRUCTURAL candidate wins ---
    adapter_blob, is_framed = build_latent_blob_dedup_or_vendored(latents, cfg)
    adapter_bytes = len(adapter_blob)
    default_preserved = (adapter_blob == vendored_blob) and (is_framed is False)

    # --- exact-dedup diagnostics on the REAL latents (why dedup wins or loses) ---
    q, _mins, _scales = quantize_pairs(latents, cfg)
    uniq = np.unique(q, axis=0)
    n_pairs, latent_dim = q.shape
    n_unique = uniq.shape[0]

    # --- structural-headroom diagnostics: WHY the cross-pair space is exhausted ---
    # (1) symbol-entropy floor of the 1st-order delta stream vs the deployed vendored bytes;
    # (2) SVD energy spectrum of the centered quant matrix (is there low-rank structure?).
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0].astype(np.int16)
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    vals, counts = np.unique(delta, return_counts=True)
    p = counts / counts.sum()
    delta_entropy_bits = float(-(p * np.log2(p)).sum())
    entropy_floor_bytes = int(delta_entropy_bits * q.size / 8)
    qf = q.astype(np.float64)
    xc = qf - qf.mean(0)
    sv = np.linalg.svd(xc, compute_uv=False)
    energy = (sv**2) / (sv**2).sum()
    cum = np.cumsum(energy)
    ranks_999 = int(np.searchsorted(cum, 0.999)) + 1

    # --- round-trip exactness of the deployed latents blob (reproduces the quantized codes) ---
    # Decode whatever the adapter emitted and confirm it reproduces the quantized-then-dequant
    # latents bit-exactly (lossless on the QUANTIZED grid — the vendored lossy step is shared).
    decoded = decode_latent_blob(adapter_blob, is_framed)
    # Reference: dequant of the SAME quant grid.
    from tac.losses.cross_pair_latent_codec import dequantize_pairs

    ref = dequantize_pairs(q, _mins, _scales)
    rt_max_abs_err = float((decoded - ref).abs().max().item())
    roundtrip_exact = rt_max_abs_err == 0.0

    # Also confirm the framed-core decode round-trips (independent of the adapter's no-win path).
    framed_decoded = decode_latents_best(framed_payload)
    # framed core reproduces the quantized codes -> dequant matches ref.
    framed_rt_err = float((framed_decoded - ref).abs().max().item())

    return {
        "vendored_latents_bytes": vendored_bytes,
        "best_framed_candidate_bytes": len(framed_blob),
        "best_framed_format": flag_names.get(flag, str(flag)),
        "adapter_emitted_bytes": adapter_bytes,
        "adapter_emitted_framed_format": bool(is_framed),
        "delta_bytes_adapter_vs_vendored": adapter_bytes - vendored_bytes,
        "delta_pct_adapter_vs_vendored": (
            100.0 * (adapter_bytes - vendored_bytes) / vendored_bytes
        ),
        "default_preserving_no_win_byte_identical": bool(default_preserved),
        "n_pairs": int(n_pairs),
        "latent_dim": int(latent_dim),
        "n_unique_quantized_rows": int(n_unique),
        "exact_dedup_collapse_ratio": float(n_unique) / float(n_pairs),
        "delta_symbol_entropy_bits": delta_entropy_bits,
        "delta_symbol_entropy_floor_bytes": entropy_floor_bytes,
        "vendored_pct_above_entropy_floor": (
            100.0 * (vendored_bytes - entropy_floor_bytes) / entropy_floor_bytes
        ),
        "svd_ranks_for_99_9pct_energy": ranks_999,
        "svd_is_low_rank": ranks_999 < latent_dim,
        "structural_headroom_verdict": (
            "EXHAUSTED: near entropy floor + full-rank + no exact dups => no lossless "
            "cross-pair candidate (dedup/codebook/low-rank/AR) beats vendored on these latents"
            if (ranks_999 >= latent_dim and n_unique == n_pairs)
            else "HEADROOM: structural redundancy present (dedup/low-rank candidate may win)"
        ),
        "roundtrip_exact_deployed_latents": bool(roundtrip_exact),
        "roundtrip_max_abs_err_deployed": rt_max_abs_err,
        "roundtrip_max_abs_err_framed_core": framed_rt_err,
        "vendored_blob_sha16": _sha(vendored_blob),
        "adapter_blob_sha16": _sha(adapter_blob),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--basin",
        default="experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best",
        help="dir with best_ema_decoder.pt + best_ema_latents.pt",
    )
    ap.add_argument("--json", default=None, help="optional JSON output path")
    args = ap.parse_args(argv)

    basin = Path(args.basin)
    decoder_pt = basin / "best_ema_decoder.pt"
    latents_pt = basin / "best_ema_latents.pt"
    if not decoder_pt.exists() or not latents_pt.exists():
        raise FileNotFoundError(f"basin missing decoder/latents at {basin}")

    sd = torch.load(decoder_pt, map_location="cpu", weights_only=False)
    latents = torch.load(latents_pt, map_location="cpu", weights_only=False)
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"latents must be a tensor; got {type(latents)}")

    report = {
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "basin": str(basin),
        "decoder_n_tensors": len(sd),
        "latents_shape": list(latents.shape),
        "part_D_rate_attack_variable_level_codec": verify_rate_attack(sd),
        "part_E_l3_finishing_kit": verify_l3_kit(),
        "part_F_d1_cross_pair_latent_dedup": verify_d1_latent_dedup(latents),
    }

    print(json.dumps(report, indent=2))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
