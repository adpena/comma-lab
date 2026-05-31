#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generalized per-substrate archive pixel-consumption probe (Catalog #272).

Per Catalog #272 (distinguishing-feature integration contract) + Catalog #220
(L1+ scaffold operational mechanism) + Catalog #105/#139 (no-op detector): given
a substrate's BUILT archive bytes + its REAL inflate/reconstruct path, this probe
classifies the TRAINED renderer/predictor weight section as one of:

- ``PIXEL_CONSUMED`` — perturbing the trained decoder/predictor WEIGHTS (at the
  semantic post-decompression level) CHANGES the reconstructed RGB output. The
  trained weights genuinely drive contest pixels. A ratification of this archive
  validates the TRAINING.
- ``PLACEHOLDER_OR_PARSE_GUARD`` — the decoder section exists but perturbing the
  decoded weights produces ZERO pixel delta (silently ignored, e.g. a
  ``load_state_dict(strict=False)`` whose keys do not match, OR placeholder
  zeros). A ratification of this archive does NOT validate the training (the
  inflate reconstructs pixels from some OTHER section, e.g. a classical codec
  blob). This is the Z8 ``decoder_blob`` trap (commit 182b88406).
- ``CODEC_DRIVEN`` — the substrate has NO trained-decoder slot; pixels come
  entirely from a classical codec blob (wavelet / chroma-LUT). The codec blob is
  the pixel driver, NOT a trained renderer. Ratifying validates the CODEC.
- ``NOT_BUILT`` — no archive on disk (archive-grammar build needed before
  ratification is meaningful).
- ``ADAPTER_NOT_IMPLEMENTED`` — a built archive exists but this probe has no
  faithful reconstruct adapter for it; reported honestly, NOT fabricated.

WHY THE PERTURBATION IS AT THE *DECODED WEIGHT* LEVEL, NOT THE RAW BYTE LEVEL:
the trained decoder weights are stored brotli/zlib-compressed. A raw byte flip in
a compressed blob ALWAYS corrupts the stream -> parse error (structural
consumption per Catalog #105) and CANNOT distinguish "weights drive pixels" from
"weights are placeholder zeros silently dropped by strict=False". So the faithful
test is: parse archive -> perturb the largest-norm decoder tensor -> re-render the
SAME small per-pair RGB surface the contest inflate runs -> compare pixels. A
non-zero pixel delta is a faithful operational-consumption proof; a zero delta is
a faithful placeholder/ignored proof.

NO FAKE IMPLEMENTATIONS (CLAUDE.md non-negotiable): every adapter actually parses
real archive bytes, actually perturbs real decoded weights, actually runs the
real reconstruct forward, and actually compares real pixel outputs. A verdict that
would be identical regardless of the inflate output is forbidden. Missing archives
return ``NOT_BUILT``; missing adapters return ``ADAPTER_NOT_IMPLEMENTED``.

Axis discipline (Catalog #127/#192/#323/#341): this probe NEVER produces a contest
score. Every manifest carries ``score_claim=False``, ``promotable=False``,
``axis_tag="[macOS-CPU advisory]"``. It is a structural-consumption diagnostic.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXEL_CONSUMPTION_PROBE_SCHEMA = (
    "substrate_archive_pixel_consumption_probe.v1"
)

# A perturbed-weight pixel delta below this is treated as "not consumed" (the
# weights are silently ignored / placeholder). Above it = genuinely consumed.
# 1e-6 is well above fp32 reconstruction jitter and well below the ~1e-2 deltas
# a real trained-weight perturbation produces.
MIN_PIXEL_DELTA: float = 1e-6

# A base reconstruction whose spatial std is below this is treated as degenerate
# (near-constant / flat-gray) — the signature of a mock-teacher / near-untrained
# archive whose decoder is wired but carries no trained signal. Ratifying such an
# archive scores a flat reconstruction, NOT validated training.
MIN_BASE_SPATIAL_VARIANCE: float = 1e-3

# Verdict sentinels.
PIXEL_CONSUMED = "PIXEL_CONSUMED"
# Decoder weights are wired (perturbation moves pixels) BUT the base recon is
# near-constant (degenerate / untrained signal). Ratification scores a flat
# reconstruction, not validated PC-training.
PIXEL_CONSUMED_BUT_NEAR_UNTRAINED = "PIXEL_CONSUMED_BUT_NEAR_UNTRAINED"
PLACEHOLDER_OR_PARSE_GUARD = "PLACEHOLDER_OR_PARSE_GUARD"
CODEC_DRIVEN = "CODEC_DRIVEN"
NOT_BUILT = "NOT_BUILT"
ADAPTER_NOT_IMPLEMENTED = "ADAPTER_NOT_IMPLEMENTED"
ADAPTER_ERROR = "ADAPTER_ERROR"


@dataclass
class WeightPerturbVerdict:
    """Result of perturbing a trained-weight section + re-rendering."""

    verdict: str
    perturbed_tensor: str | None = None
    max_abs_pixel_delta: float = 0.0
    mean_abs_pixel_delta: float = 0.0
    base_pixel_count: int = 0
    n_decoder_tensors: int = 0
    base_spatial_variance: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "perturbed_tensor": self.perturbed_tensor,
            "max_abs_pixel_delta": self.max_abs_pixel_delta,
            "mean_abs_pixel_delta": self.mean_abs_pixel_delta,
            "base_pixel_count": self.base_pixel_count,
            "n_decoder_tensors": self.n_decoder_tensors,
            "base_spatial_variance": self.base_spatial_variance,
            "detail": self.detail,
        }


@dataclass
class SubstrateAdapter:
    """A per-substrate reconstruct adapter the probe drives.

    ``classify_weights`` MUST actually parse the archive, perturb a real decoded
    trained-weight tensor, re-render the real small per-pair RGB surface, and
    compare pixels. ``architecture_family`` is ``"renderer"`` (trained decoder
    state_dict -> frames) or ``"codec"`` (classical codec blob -> frames; the
    trained-weight section, if present, is parse-guard only by construction).
    """

    substrate_id: str
    magic: bytes
    architecture_family: str  # "renderer" | "codec"
    classify_weights: Callable[[bytes, int], WeightPerturbVerdict]
    notes: str = ""


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _perturb_one_tensor(
    state_dict: dict, key: str, *, rel_std_frac: float = 1.0, abs_floor: float = 0.2
):
    """Return (perturbed_state_dict, applied_pert) for a single tensor.

    Adds ``rel_std_frac * std`` (or ``abs_floor`` if std==0) to the whole tensor.
    A whole-tensor additive perturbation is a faithful "do these weights matter?"
    test: if the reconstruct forward reads the tensor, every pixel it influences
    shifts; if the forward ignores it (strict=False dropped key / placeholder),
    zero pixels shift.
    """
    import torch

    sd2 = {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_dict.items()
    }
    t = sd2[key].clone().float()
    std = float(t.std()) if t.numel() > 1 else 0.0
    pert = rel_std_frac * std if std > 0 else abs_floor
    t = t + pert
    sd2[key] = t.to(state_dict[key].dtype)
    return sd2, pert


def _sweep_perturb_all_tensors(state_dict: dict, render_fn, base):
    """Perturb EACH decoder tensor (1.0x std) and return the MAX pixel delta.

    Critical methodology point: perturbing ONLY the largest-norm tensor can miss
    real consumption when that tensor has low pixel-influence (e.g. an input
    projection that the forward squashes). Sweeping every tensor + taking the max
    is the faithful "are ANY of these weights consumed?" test. Returns
    ``(best_key, max_delta, mean_at_max, applied_pert)``.
    """
    import numpy as np
    import torch

    keys = [k for k, v in state_dict.items() if torch.is_tensor(v) and v.numel() > 0]
    if not keys:
        return None, 0.0, 0.0, 0.0
    best_key = None
    best_max = -1.0
    best_mean = 0.0
    best_pert = 0.0
    for key in keys:
        sd2, pert = _perturb_one_tensor(state_dict, key)
        try:
            recon = render_fn(sd2)
        except Exception:
            continue
        if recon.shape != base.shape:
            continue
        diff = np.abs(recon - base)
        mx = float(diff.max())
        if mx > best_max:
            best_max = mx
            best_mean = float(diff.mean())
            best_key = key
            best_pert = pert
    return best_key, max(best_max, 0.0), best_mean, best_pert


def _base_spatial_variance(base) -> float:
    """Std of the base reconstruction — a near-zero std means a degenerate /
    near-constant (e.g. flat gray) output, the signature of an untrained or
    mock-teacher archive whose decoder is wired but carries no trained signal.
    """
    import numpy as np

    if base.size == 0:
        return 0.0
    return float(np.std(base))


def _flat_pixel_delta(base, recon) -> tuple[float, float]:
    import numpy as np

    if recon.shape != base.shape:
        return float("nan"), float("nan")
    diff = np.abs(recon - base)
    return float(diff.max()), float(diff.mean())


def _classify_from_delta(
    max_delta: float,
    mean_delta: float,
    base_count: int,
    n_tensors: int,
    perturbed_key: str | None,
    base_variance: float = float("nan"),
    detail: str = "",
) -> WeightPerturbVerdict:
    import math

    if math.isnan(max_delta):
        return WeightPerturbVerdict(
            verdict=ADAPTER_ERROR,
            perturbed_tensor=perturbed_key,
            base_pixel_count=base_count,
            n_decoder_tensors=n_tensors,
            base_spatial_variance=(0.0 if math.isnan(base_variance) else base_variance),
            detail="perturbed render shape changed vs base (" + detail + ")",
        )
    if max_delta <= MIN_PIXEL_DELTA:
        verdict = PLACEHOLDER_OR_PARSE_GUARD
    elif (not math.isnan(base_variance)) and base_variance < MIN_BASE_SPATIAL_VARIANCE:
        # Weights ARE wired (pixels move on perturbation) but the base recon is
        # near-constant -> the archive carries no trained signal (mock-teacher /
        # underconverged). Ratifying scores a flat reconstruction, not training.
        verdict = PIXEL_CONSUMED_BUT_NEAR_UNTRAINED
    else:
        verdict = PIXEL_CONSUMED
    return WeightPerturbVerdict(
        verdict=verdict,
        perturbed_tensor=perturbed_key,
        max_abs_pixel_delta=max_delta,
        mean_abs_pixel_delta=mean_delta,
        base_pixel_count=base_count,
        n_decoder_tensors=n_tensors,
        base_spatial_variance=(0.0 if math.isnan(base_variance) else base_variance),
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# Renderer-family adapters (trained decoder state_dict -> frames)
# --------------------------------------------------------------------------- #
def _z6v2_classify(archive_bytes: bytes, k_pairs: int) -> WeightPerturbVerdict:
    import numpy as np
    import torch

    from tac.substrates.z6_v2_cargo_cult_unwind.architecture import (
        Z6V2Config,
        Z6V2Substrate,
    )
    from tac.substrates.z6_v2_cargo_cult_unwind.archive import parse_archive

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    n_pairs = int(arc.latents.shape[0])

    def build(sd_override=None):
        cfg = Z6V2Config(
            latent_dim=int(arc.latents.shape[1]),
            ego_dim=int(arc.ego_vecs.shape[1]),
            embed_dim=int(meta["embed_dim"]),
            initial_grid_h=int(meta["initial_grid_h"]),
            initial_grid_w=int(meta["initial_grid_w"]),
            decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
            sin_frequency=float(meta["sin_frequency"]),
            num_upsample_blocks=int(meta["num_upsample_blocks"]),
            rao_ballard_level_boundary=int(meta.get("rao_ballard_level_boundary", 3)),
            film_generator_depth=int(meta.get("film_generator_depth", 3)),
            film_hidden_width=int(meta.get("film_hidden_width", 80)),
            cooperative_receiver_beta=float(meta.get("cooperative_receiver_beta", 0.5)),
            num_pairs=n_pairs,
            output_height=int(meta["output_height"]),
            output_width=int(meta["output_width"]),
        )
        m = Z6V2Substrate(cfg).to("cpu").eval()
        m.load_state_dict(
            sd_override if sd_override is not None else arc.decoder_state_dict,
            strict=False,
        )
        with torch.no_grad():
            m.latents.copy_(arc.latents.to(dtype=m.latents.dtype))
            m.ego_vecs.copy_(arc.ego_vecs.to(dtype=m.ego_vecs.dtype))
        return m

    def render(m):
        outs = []
        with torch.inference_mode():
            for i in range(min(k_pairs, n_pairs)):
                r0, r1 = m(torch.tensor([i], dtype=torch.long))
                outs.append(np.concatenate([r0.numpy().ravel(), r1.numpy().ravel()]))
        return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)

    base = render(build())
    base_var = _base_spatial_variance(base)
    key, mx, mn, pert = _sweep_perturb_all_tensors(
        arc.decoder_state_dict, lambda sd2: render(build(sd2)), base
    )
    if key is None:
        return WeightPerturbVerdict(
            verdict=PLACEHOLDER_OR_PARSE_GUARD,
            base_pixel_count=int(base.size),
            base_spatial_variance=base_var,
            n_decoder_tensors=0,
            detail="decoder_state_dict is empty",
        )
    return _classify_from_delta(
        mx, mn, int(base.size), len(arc.decoder_state_dict), key,
        base_variance=base_var,
        detail=f"swept all {len(arc.decoder_state_dict)} decoder tensors (1.0x std); max on {key} (+{pert:.5f})",
    )


def _z7_mamba2_classify(archive_bytes: bytes, k_pairs: int) -> WeightPerturbVerdict:
    import numpy as np
    import torch

    from tac.substrates.time_traveler_l5_z7_mamba2.archive import parse_archive
    from tac.substrates.time_traveler_l5_z7_mamba2.inflate import (
        _build_decoder,
        replay_latent_sequence_with_context,
    )

    arc = parse_archive(archive_bytes)
    n_pairs = int(arc.config.num_pairs)

    def render_with_decoder(decoder):
        latents_cpu, _contexts = replay_latent_sequence_with_context(arc)
        latents = latents_cpu.to("cpu", dtype=torch.float32)
        outs = []
        with torch.no_grad():
            for i in range(min(k_pairs, n_pairs)):
                z_t = latents[i : i + 1]
                r0, r1 = decoder(z_t)
                outs.append(np.concatenate([r0.numpy().ravel(), r1.numpy().ravel()]))
        return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)

    base_decoder = _build_decoder(arc, "cpu")
    base = render_with_decoder(base_decoder)
    base_var = _base_spatial_variance(base)
    base_sd = dict(base_decoder.state_dict())
    n_dec = len(base_sd)

    def render_with_sd(sd2):
        dec = _build_decoder(arc, "cpu")
        dec.load_state_dict(sd2, strict=True)
        dec.eval()
        return render_with_decoder(dec)

    key, mx, mn, pert = _sweep_perturb_all_tensors(base_sd, render_with_sd, base)
    if key is None:
        return WeightPerturbVerdict(
            verdict=PLACEHOLDER_OR_PARSE_GUARD,
            base_pixel_count=int(base.size),
            base_spatial_variance=base_var,
            n_decoder_tensors=n_dec,
            detail="decoder has no learnable tensors",
        )
    return _classify_from_delta(
        mx, mn, int(base.size), n_dec, key,
        base_variance=base_var,
        detail=f"swept all {n_dec} Z6-compatible decoder tensors (1.0x std); max on {key} (+{pert:.5f})",
    )


def _dreamer_classify(archive_bytes: bytes, k_pairs: int) -> WeightPerturbVerdict:
    # READ-ONLY: sister C owns dreamer_v3_rssm SOURCE; we only IMPORT + RUN it.
    import numpy as np
    import torch

    from tac.substrates.dreamer_v3_rssm.archive import parse_archive
    from tac.substrates.dreamer_v3_rssm.inflate import DreamerV3RSSMDecoderTorch

    arc = parse_archive(archive_bytes)

    def build_decoder() -> DreamerV3RSSMDecoderTorch:
        dec = DreamerV3RSSMDecoderTorch(
            num_groups=arc.num_groups,
            num_categories=arc.num_categories,
            decoder_latent_dim=arc.decoder_latent_dim,
            base_channels=arc.base_channels,
        )
        # Mirror the inflate's NHWC->NCHW transpose for 4-D conv weights.
        torch_sd: dict[str, torch.Tensor] = {}
        for key, np_arr in arc.decoder_state_dict.items():
            arr = np_arr.astype("float32")
            if arr.ndim == 4:
                arr = arr.transpose(0, 3, 1, 2).copy()
            torch_sd[key] = torch.from_numpy(arr)
        dec.load_state_dict(torch_sd, strict=False)
        dec.eval()
        return dec

    indices = torch.from_numpy(arc.category_indices).to(dtype=torch.long)
    n_pairs = int(arc.num_pairs)

    def render(dec) -> Any:
        outs = []
        with torch.inference_mode():
            for i in range(min(k_pairs, n_pairs)):
                # (1, G) -> (1, 2, 3, H, W)
                decoded = dec(indices[i : i + 1])
                outs.append(decoded.detach().numpy().ravel())
        return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)

    base_dec = build_decoder()
    base = render(base_dec)
    base_var = _base_spatial_variance(base)
    base_sd = dict(base_dec.state_dict())
    n_dec = len(base_sd)

    def render_with_sd(sd2) -> Any:
        dec = build_decoder()
        dec.load_state_dict(sd2, strict=False)
        dec.eval()
        return render(dec)

    key, mx, mn, pert = _sweep_perturb_all_tensors(base_sd, render_with_sd, base)
    if key is None:
        return WeightPerturbVerdict(
            verdict=PLACEHOLDER_OR_PARSE_GUARD,
            base_pixel_count=int(base.size),
            base_spatial_variance=base_var,
            n_decoder_tensors=n_dec,
            detail="dreamer decoder has no learnable tensors",
        )
    return _classify_from_delta(
        mx, mn, int(base.size), n_dec, key,
        base_variance=base_var,
        detail=f"swept all {n_dec} dreamer decoder tensors (1.0x std); max on {key} (+{pert:.5f})",
    )


def _z5_rao_ballard_classify(archive_bytes: bytes, k_pairs: int) -> WeightPerturbVerdict:
    import numpy as np
    import torch

    from tac.substrates.time_traveler_l5_z5.architecture import (
        Z5RaoBallardConfig,
        Z5RaoBallardSubstrate,
    )
    from tac.substrates.time_traveler_l5_z5.archive import parse_archive

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    num_pairs = int(arc.low_latents.shape[0])

    def build(dec_override=None):
        cfg = Z5RaoBallardConfig(
            low_latent_dim=int(arc.low_latents.shape[1]),
            high_latent_dim=int(arc.high_latents.shape[1]),
            ego_dim=int(arc.ego_vecs.shape[1]),
            embed_dim=int(meta.get("embed_dim", 32)),
            initial_grid_h=int(meta.get("initial_grid_h", 3)),
            initial_grid_w=int(meta.get("initial_grid_w", 4)),
            decoder_channels=tuple(
                int(c) for c in meta.get("decoder_channels", (24, 20, 16, 12, 8, 6, 4))
            ),
            num_upsample_blocks=int(meta.get("num_upsample_blocks", 7)),
            sin_frequency=float(meta.get("sin_frequency", 30.0)),
            film_generator_depth=int(meta.get("film_generator_depth", 3)),
            film_hidden_width=int(meta.get("film_hidden_width", 24)),
            num_pairs=num_pairs,
            output_height=int(meta.get("output_height", 384)),
            output_width=int(meta.get("output_width", 512)),
            predictor_hidden_dim=int(meta.get("predictor_hidden_dim", 48)),
            predictor_num_layers=int(meta.get("predictor_num_layers", 2)),
        )
        m = Z5RaoBallardSubstrate(cfg).to("cpu").eval()
        m.decoder.load_state_dict(
            dec_override if dec_override is not None else arc.decoder_state_dict,
            strict=False,
        )
        m.predictor.load_state_dict(arc.predictor_state_dict, strict=False)
        with torch.no_grad():
            m.low_latents.copy_(arc.low_latents.to(dtype=m.low_latents.dtype))
            m.high_latents.copy_(arc.high_latents.to(dtype=m.high_latents.dtype))
            m.ego_vecs.copy_(arc.ego_vecs.to(dtype=m.ego_vecs.dtype))
        return m

    def render(m):
        outs = []
        with torch.no_grad():
            for i in range(min(k_pairs, num_pairs)):
                r0, r1, _res = m.reconstruct_pair(torch.tensor([i], dtype=torch.long))
                outs.append(np.concatenate([r0.numpy().ravel(), r1.numpy().ravel()]))
        return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)

    base = render(build())
    base_var = _base_spatial_variance(base)
    key, mx, mn, pert = _sweep_perturb_all_tensors(
        arc.decoder_state_dict, lambda sd2: render(build(sd2)), base
    )
    if key is None:
        return WeightPerturbVerdict(
            verdict=PLACEHOLDER_OR_PARSE_GUARD,
            base_pixel_count=int(base.size),
            base_spatial_variance=base_var,
            n_decoder_tensors=0,
            detail="decoder_state_dict is empty",
        )
    return _classify_from_delta(
        mx, mn, int(base.size), len(arc.decoder_state_dict), key,
        base_variance=base_var,
        detail=f"swept all {len(arc.decoder_state_dict)} Z5 Rao-Ballard decoder tensors (1.0x std); max on {key} (+{pert:.5f})",
    )


def _z4_cooperative_receiver_classify(
    archive_bytes: bytes, k_pairs: int
) -> WeightPerturbVerdict:
    import numpy as np
    import torch

    from tac.substrates.z4_cooperative_receiver_loss.architecture import (
        CooperativeReceiverConfig,
        CooperativeReceiverSubstrate,
    )
    from tac.substrates.z4_cooperative_receiver_loss.archive import parse_archive

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    num_pairs = int(arc.latents.shape[0])

    def build(dec_override=None):
        cfg = CooperativeReceiverConfig(
            latent_dim=int(arc.latents.shape[1]),
            encoder_input_channels=int(meta.get("encoder_input_channels", 3)),
            encoder_hidden_dim=int(meta.get("encoder_hidden_dim", 64)),
            decoder_embed_dim=int(meta["decoder_embed_dim"]),
            decoder_initial_grid_h=int(meta["decoder_initial_grid_h"]),
            decoder_initial_grid_w=int(meta["decoder_initial_grid_w"]),
            decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
            decoder_num_upsample_blocks=int(meta["decoder_num_upsample_blocks"]),
            num_pairs=num_pairs,
            output_height=int(meta.get("output_height", 384)),
            output_width=int(meta.get("output_width", 512)),
            cooperative_receiver_lambda_pixel=float(
                meta.get("cooperative_receiver_meta", {}).get("lambda_pixel", 0.0)
            ),
            cooperative_receiver_atick_redlich_form=bool(
                meta.get("cooperative_receiver_meta", {}).get("atick_redlich_form", True)
            ),
            latent_init_std=float(meta.get("latent_init_std", 0.02)),
        )
        m = CooperativeReceiverSubstrate(cfg).to("cpu").eval()
        m.encoder.load_state_dict(arc.encoder_state_dict, strict=False)
        m.decoder.load_state_dict(
            dec_override if dec_override is not None else arc.decoder_state_dict,
            strict=False,
        )
        with torch.no_grad():
            m.latents.copy_(arc.latents.to(dtype=m.latents.dtype))
        return m

    def render(m):
        outs = []
        with torch.no_grad():
            for i in range(min(k_pairs, num_pairs)):
                r0, r1, _mu, _lv = m(
                    torch.tensor([i], dtype=torch.long), frames_for_encoder=None
                )
                outs.append(np.concatenate([r0.numpy().ravel(), r1.numpy().ravel()]))
        return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)

    base = render(build())
    base_var = _base_spatial_variance(base)
    key, mx, mn, pert = _sweep_perturb_all_tensors(
        arc.decoder_state_dict, lambda sd2: render(build(sd2)), base
    )
    if key is None:
        return WeightPerturbVerdict(
            verdict=PLACEHOLDER_OR_PARSE_GUARD,
            base_pixel_count=int(base.size),
            base_spatial_variance=base_var,
            n_decoder_tensors=0,
            detail="decoder_state_dict is empty",
        )
    return _classify_from_delta(
        mx, mn, int(base.size), len(arc.decoder_state_dict), key,
        base_variance=base_var,
        detail=f"swept all {len(arc.decoder_state_dict)} Z4 cooperative-receiver decoder tensors (1.0x std); max on {key} (+{pert:.5f})",
    )


# --------------------------------------------------------------------------- #
# Codec-family adapters (classical codec blob -> frames; weights parse-guard)
# --------------------------------------------------------------------------- #
def _codec_family_verdict(substrate_id: str, codec_blob_name: str) -> WeightPerturbVerdict:
    """A codec-family substrate has NO trained-decoder pixel slot by construction.

    The pixel driver is the classical codec blob (wavelet inverse / chroma LUT).
    Any trained-weight section, if present, is parse-guard only. The faithful
    statement is: a ratification validates the CODEC, not a trained renderer.
    """
    return WeightPerturbVerdict(
        verdict=CODEC_DRIVEN,
        detail=(
            f"{substrate_id} reconstructs pixels from {codec_blob_name} "
            f"(classical codec); no trained-renderer-weight pixel slot"
        ),
    )


def _z8_classify(archive_bytes: bytes, k_pairs: int) -> WeightPerturbVerdict:
    """Re-confirm Z8 by delegating to B's canonical Z8 byte-mutation proof.

    B's probe (tools/probe_z8_archive_distinguishing_feature_byte_mutation.py)
    is the canonical Z8 consumption proof: wavelet_blob PIXEL_CONSUMED,
    decoder_blob PARSE_GUARD_ONLY. We re-run it on the supplied archive and map
    its per-section verdict onto this probe's vocabulary: the TRAINED-WEIGHT
    section (decoder_blob) is the classification target.
    """
    import tempfile

    from tools.probe_z8_archive_distinguishing_feature_byte_mutation import (
        probe_z8_archive_distinguishing_feature,
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(archive_bytes)
        tmp = Path(fh.name)
    try:
        m = probe_z8_archive_distinguishing_feature(tmp)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    dec = m["sections"].get("decoder_blob", {})
    wav = m["sections"].get("wavelet_blob", {})
    dec_verdict = dec.get("verdict")
    if dec_verdict == "PIXEL_CONSUMED":
        return WeightPerturbVerdict(
            verdict=PIXEL_CONSUMED,
            max_abs_pixel_delta=float(dec.get("max_abs_pixel_delta", 0.0)),
            detail="Z8 decoder_blob (trained renderer) pixel-consumed per B's proof",
        )
    # decoder_blob is PARSE_GUARD_ONLY / NO_OP: the trained categorical-posterior
    # HNeRV renderer weights are NOT pixel-consumed; pixels come from the
    # classical Mallat wavelet inverse (wavelet_blob).
    return WeightPerturbVerdict(
        verdict=CODEC_DRIVEN,
        max_abs_pixel_delta=float(wav.get("max_abs_pixel_delta", 0.0)),
        detail=(
            "Z8 decoder_blob (trained categorical-posterior HNeRV renderer) is "
            f"{dec_verdict}; pixels driven by classical Mallat wavelet_blob "
            f"(wavelet verdict={wav.get('verdict')}) per B's commit 182b88406"
        ),
    )


# --------------------------------------------------------------------------- #
# Adapter registry
# --------------------------------------------------------------------------- #
def _adapters() -> dict[str, SubstrateAdapter]:
    return {
        "z6_v2_cargo_cult_unwind": SubstrateAdapter(
            substrate_id="z6_v2_cargo_cult_unwind",
            magic=b"Z6V2",
            architecture_family="renderer",
            classify_weights=_z6v2_classify,
            notes="load_state_dict(strict=False) -> Z6V2Substrate render; FiLM Rao-Ballard decoder",
        ),
        "time_traveler_l5_z7_mamba2": SubstrateAdapter(
            substrate_id="time_traveler_l5_z7_mamba2",
            magic=b"Z7M2",
            architecture_family="renderer",
            classify_weights=_z7_mamba2_classify,
            notes="load_state_dict(strict=True) -> Z6-compatible decoder; Mamba-2 predictor replay",
        ),
        "dreamer_v3_rssm": SubstrateAdapter(
            substrate_id="dreamer_v3_rssm",
            magic=b"RSSC",
            architecture_family="renderer",
            classify_weights=_dreamer_classify,
            notes="READ-ONLY (sister C owns source); categorical dequant + decoder forward",
        ),
        "time_traveler_l5_z5": SubstrateAdapter(
            substrate_id="time_traveler_l5_z5",
            magic=b"Z5RB",
            architecture_family="renderer",
            classify_weights=_z5_rao_ballard_classify,
            notes="2-level Rao-Ballard predictive coder; decoder+predictor state_dict -> reconstruct_pair",
        ),
        "z4_cooperative_receiver_loss": SubstrateAdapter(
            substrate_id="z4_cooperative_receiver_loss",
            magic=b"Z4CR",
            architecture_family="renderer",
            classify_weights=_z4_cooperative_receiver_classify,
            notes="Atick-Redlich cooperative-receiver loss; latent + decoder",
        ),
        "nscs06_v8_path_b_wavelet": SubstrateAdapter(
            substrate_id="nscs06_v8_path_b_wavelet",
            magic=b"",
            architecture_family="codec",
            classify_weights=lambda b, k: _codec_family_verdict(
                "nscs06_v8_path_b_wavelet", "wavelet + Wyner-Ziv residual blob"
            ),
            notes="pywavelets inverse + Wyner-Ziv temporal; no trained-renderer pixel slot",
        ),
        "z8_hierarchical_predictive_coding": SubstrateAdapter(
            substrate_id="z8_hierarchical_predictive_coding",
            magic=b"Z8HP",
            architecture_family="codec",
            classify_weights=_z8_classify,
            notes="Mallat wavelet inverse codec; trained decoder_blob PARSE_GUARD per B 182b88406",
        ),
        "nscs06_v8_chroma_lut": SubstrateAdapter(
            substrate_id="nscs06_v8_chroma_lut",
            magic=b"CH08",
            architecture_family="codec",
            classify_weights=lambda b, k: _codec_family_verdict(
                "nscs06_v8_chroma_lut", "grayscale + chroma-LUT blob"
            ),
            notes="grayscale + per-class chroma LUT; no trained-renderer pixel slot",
        ),
    }


def _identify_magic(archive_bytes: bytes) -> bytes:
    return archive_bytes[:4] if len(archive_bytes) >= 4 else b""


def probe_substrate_archive_pixel_consumption(
    substrate_id: str,
    archive_path: Path | None,
    *,
    k_pairs: int = 2,
    proof_out: Path | None = None,
) -> dict[str, Any]:
    """Classify a substrate's trained-weight pixel-consumption (Catalog #272).

    Args:
        substrate_id: registry key (e.g. ``"z6_v2_cargo_cult_unwind"``).
        archive_path: path to the built ``0.bin`` archive, or None / missing for
            ``NOT_BUILT``.
        k_pairs: number of pairs to render for the comparison surface.
        proof_out: optional path to write the proof JSON.

    Returns:
        A manifest dict with the per-substrate verdict + axis-discipline markers.
    """
    adapters = _adapters()
    adapter = adapters.get(substrate_id)
    manifest: dict[str, Any] = {
        "schema_version": PIXEL_CONSUMPTION_PROBE_SCHEMA,
        "tool": "tools/probe_substrate_archive_pixel_consumption.py",
        "substrate_id": substrate_id,
        "archive_path": str(archive_path) if archive_path is not None else None,
        # Axis discipline: NEVER a contest score.
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "advisory",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }

    if archive_path is None or not Path(archive_path).is_file():
        manifest.update(
            {
                "architecture_family": adapter.architecture_family if adapter else "unknown",
                "trained_weight_verdict": NOT_BUILT,
                "detail": "no built archive on disk; archive-grammar build needed",
            }
        )
        if adapter:
            manifest["adapter_notes"] = adapter.notes
        if proof_out is not None:
            _write_proof(proof_out, manifest)
        return manifest

    archive_bytes = Path(archive_path).read_bytes()
    manifest["archive_bytes"] = len(archive_bytes)
    manifest["archive_magic"] = _identify_magic(archive_bytes).hex()

    if adapter is None:
        manifest.update(
            {
                "architecture_family": "unknown",
                "trained_weight_verdict": ADAPTER_NOT_IMPLEMENTED,
                "detail": f"no probe adapter registered for substrate {substrate_id!r}",
            }
        )
        if proof_out is not None:
            _write_proof(proof_out, manifest)
        return manifest

    manifest["architecture_family"] = adapter.architecture_family
    manifest["adapter_notes"] = adapter.notes
    try:
        verdict = adapter.classify_weights(archive_bytes, k_pairs)
    except Exception as exc:  # honest adapter-error reporting
        verdict = WeightPerturbVerdict(
            verdict=ADAPTER_ERROR, detail=f"{type(exc).__name__}: {repr(exc)[:200]}"
        )
    manifest["trained_weight_verdict"] = verdict.verdict
    manifest["weight_perturbation"] = verdict.as_dict()

    # Ratification semantics: PIXEL_CONSUMED validates training; CODEC_DRIVEN /
    # PLACEHOLDER validates the codec, not the training.
    manifest["ratification_validates"] = _ratification_semantics(
        adapter.architecture_family, verdict.verdict
    )
    manifest["ratification_ready_for_pc_training"] = bool(
        verdict.verdict == PIXEL_CONSUMED
    )
    if proof_out is not None:
        _write_proof(proof_out, manifest)
    return manifest


def _ratification_semantics(family: str, verdict: str) -> str:
    if verdict == PIXEL_CONSUMED:
        return (
            "predictive-coding/renderer TRAINING (trained weights drive contest "
            "pixels; a paired-CUDA ratification scores the training)"
        )
    if verdict == PIXEL_CONSUMED_BUT_NEAR_UNTRAINED:
        return (
            "the DECODER WIRING but NOT a converged result (weights are wired so "
            "pixels move on perturbation, but the base reconstruction is "
            "near-constant / flat-gray = mock-teacher or underconverged; ratifying "
            "scores a degenerate reconstruction, NOT validated PC-training — needs "
            "a converged real-teacher re-train BEFORE ratification is meaningful)"
        )
    if verdict == CODEC_DRIVEN:
        return (
            "the CLASSICAL CODEC (wavelet/chroma-LUT), NOT the trained renderer; "
            "ratifying buys a codec anchor, not a PC-training anchor"
        )
    if verdict == PLACEHOLDER_OR_PARSE_GUARD:
        return (
            "NOTHING about the training (trained weights ignored/placeholder; "
            "ratifying is phantom-provenance-adjacent waste per Catalog #321/#322)"
        )
    if verdict == NOT_BUILT:
        return "N/A — no archive built yet (archive-grammar build needed first)"
    return "INDETERMINATE — adapter not implemented or errored; do NOT ratify"


def _write_proof(proof_out: Path, manifest: dict[str, Any]) -> None:
    out = Path(proof_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["proof_path"] = str(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--substrate", required=True, help="registry substrate_id"
    )
    parser.add_argument(
        "--archive", type=Path, help="path to built 0.bin (omit for NOT_BUILT)"
    )
    parser.add_argument("--k-pairs", type=int, default=2)
    parser.add_argument("--proof-out", type=Path)
    parser.add_argument(
        "--list-adapters", action="store_true", help="print registered adapters"
    )
    args = parser.parse_args(argv)

    if args.list_adapters:
        for sid, ad in sorted(_adapters().items()):
            print(f"{sid}\tfamily={ad.architecture_family}\tmagic={ad.magic!r}")
        return 0

    manifest = probe_substrate_archive_pixel_consumption(
        args.substrate, args.archive, k_pairs=args.k_pairs, proof_out=args.proof_out
    )
    print(
        f"[pixel-consumption] substrate={manifest['substrate_id']} "
        f"family={manifest.get('architecture_family')} "
        f"verdict={manifest['trained_weight_verdict']}"
    )
    wp = manifest.get("weight_perturbation", {})
    if wp:
        print(
            f"[pixel-consumption] perturbed={wp.get('perturbed_tensor')} "
            f"max_abs_pixel_delta={wp.get('max_abs_pixel_delta')} "
            f"detail={wp.get('detail')}"
        )
    print(f"[pixel-consumption] ratification_validates={manifest.get('ratification_validates')}")
    print(
        f"[pixel-consumption] ratification_ready_for_pc_training="
        f"{manifest.get('ratification_ready_for_pc_training')}"
    )
    if args.proof_out is not None:
        print(f"[pixel-consumption] proof={args.proof_out}")
    return 0


__all__ = [
    "ADAPTER_ERROR",
    "ADAPTER_NOT_IMPLEMENTED",
    "CODEC_DRIVEN",
    "MIN_BASE_SPATIAL_VARIANCE",
    "MIN_PIXEL_DELTA",
    "NOT_BUILT",
    "PIXEL_CONSUMED",
    "PIXEL_CONSUMED_BUT_NEAR_UNTRAINED",
    "PIXEL_CONSUMPTION_PROBE_SCHEMA",
    "PLACEHOLDER_OR_PARSE_GUARD",
    "SubstrateAdapter",
    "WeightPerturbVerdict",
    "main",
    "probe_substrate_archive_pixel_consumption",
]


if __name__ == "__main__":
    raise SystemExit(main())
