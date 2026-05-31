# SPDX-License-Identifier: MIT
"""Z4 Atick-Redlich archive candidate builder — canonical bridge to Z4ATR bytes.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
+ Catalog #146 contest-compliant runtime contract + Catalog #220
operational mechanism: this helper bridges a trained
``Z4AtickRedlichSubstrate`` into Z4ATR archive bytes that the inflate
runtime consumes faithfully.

Canonical pattern (mirrors sister
``z6_v2_cargo_cult_unwind/archive_candidate.py``):

1. Extract decoder state_dict (everything EXCEPT ``latents``,
   ``decorrelator.proj.weight``, ``decorrelator.proj.bias``).
2. Extract latents tensor + decorrelator weight + decorrelator bias as
   separate canonical blob sections.
3. Pack meta JSON with architectural hyperparameters required by
   ``inflate.py::inflate_one_video`` to re-instantiate the substrate
   without ambient state.
4. Delegate to ``archive.pack_archive`` for byte-deterministic encoding.

The canonical extraction discipline keeps the decorrelator blob as a
SEPARATE archive section (per Catalog #272 distinguishing-feature
integration contract) — it does NOT get folded into the decoder
state_dict. This is the operational mechanism that Catalog #220 +
Catalog #272 + Catalog #105/#139 no-op detector contracts verify.

[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/archive_candidate.py
 sister bridge pattern (extracts latents + ego_vecs as separate sections)]
[verified-against: Catalog #146 + Catalog #220 + Catalog #272 contracts]
"""

from __future__ import annotations

import torch

from .archive import Z4ATR_SCHEMA_VERSION, pack_archive
from .architecture import Z4AtickRedlichConfig, Z4AtickRedlichSubstrate


# Canonical state_dict keys that are NOT part of the decoder blob (they
# get their own dedicated archive sections per Catalog #272 distinguishing-
# feature integration contract).
DECODER_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "latents",
        "decorrelator.proj.weight",
        "decorrelator.proj.bias",
    }
)


def extract_decoder_state_dict(
    model: Z4AtickRedlichSubstrate,
) -> dict[str, torch.Tensor]:
    """Extract the decoder state_dict (everything EXCEPT latents + decorrelator).

    Per Catalog #272 distinguishing-feature contract: the decorrelator
    weight + bias are stored in their OWN archive section (the
    distinguishing-feature payload). The latents are stored in their OWN
    int16-quantized section. Only the renderer weights remain in the
    "decoder" state_dict.
    """
    full = model.state_dict()
    decoder = {
        k: v for k, v in full.items() if k not in DECODER_EXCLUDED_KEYS
    }
    return decoder


def build_meta(cfg: Z4AtickRedlichConfig) -> dict[str, object]:
    """Build the canonical meta dict for Z4ATR archive packing.

    Per Catalog #146 inflate runtime contract: the meta dict MUST contain
    every architectural hyperparameter needed to re-instantiate
    ``Z4AtickRedlichConfig`` at inflate time.
    """
    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": list(cfg.decoder_channels),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "sin_frequency": float(cfg.sin_frequency),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "apply_decorrelator": bool(cfg.apply_decorrelator),
        "cooperative_receiver_beta": float(cfg.cooperative_receiver_beta),
        "_substrate_id": "time_traveler_l5_z4",
        "_substrate_variant": "atick_redlich_cooperative_receiver",
        "_archive_grammar_version": Z4ATR_SCHEMA_VERSION,
    }


def build_archive_bytes(
    model: Z4AtickRedlichSubstrate,
    *,
    extra_meta: dict[str, object] | None = None,
) -> bytes:
    """Build canonical Z4ATR archive bytes from a trained substrate.

    Per CLAUDE.md "Bit-level deconstruction and entropy discipline" +
    "Canonical leaderboard binding-depth discipline" L20 + L21 + L29 +
    L32: byte-deterministic under deterministic input.
    """
    decoder = extract_decoder_state_dict(model)
    latents = model.latents.detach().cpu()
    decorrelator_w = model.decorrelator.proj.weight.detach().cpu()
    decorrelator_b = model.decorrelator.proj.bias.detach().cpu()
    meta = build_meta(model.cfg)
    if extra_meta:
        # extra_meta keys MUST NOT collide with canonical meta keys
        # (defensive guard so callers cannot silently overwrite the
        # canonical architectural hyperparameters).
        for k in extra_meta:
            if k in meta:
                raise ValueError(
                    f"extra_meta key {k!r} collides with canonical meta key"
                )
        meta.update(extra_meta)
    return pack_archive(
        decoder_state_dict=decoder,
        latents=latents,
        decorrelator_weight=decorrelator_w,
        decorrelator_bias=decorrelator_b,
        meta=meta,
    )


__all__ = [
    "DECODER_EXCLUDED_KEYS",
    "build_archive_bytes",
    "build_meta",
    "extract_decoder_state_dict",
]
