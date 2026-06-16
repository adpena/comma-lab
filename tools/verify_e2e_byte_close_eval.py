#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""End-to-end BYTE-CLOSE -> EVAL verification harness (the #107 readiness actuator).

ONE command turns a trained small-basis checkpoint dir into an advisory score S: it
REALLY byte-closes the checkpoint into the contest ``archive.zip`` (containing the
vendored per-video ``0.bin``) + assembles the full runtime tree, REALLY runs the
authority-faithful eval path, and reports the score COMPONENTS (recomputed from
d_seg / d_pose / the REAL ``archive.zip`` ``st_size`` — never a rounded field).

This is the pipeline the production small-basis run consumes to convert a converged
checkpoint into an advisory ``S``; on contest hardware (Linux x86_64 CPU / CUDA) the
identical packet is the input to the G3 dual exact row. NO new codec is reimplemented:
the byte-close is the vendored ``codec.build_archive`` / ``parse_archive`` already wired
in ``driver._build_archive_and_eval_decoder``; the eval is the same
``RealScorerContext.exact_eval`` the driver's BEST-tracker uses (the in-Python authority
path that does the SAME decoder forward -> bicubic 874x1164 -> uint8 -> frozen
SegNet/PoseNet that ``inflate.py`` + ``evaluate.py`` do; GT decodes ONLY via
``frame_utils.yuv420_to_rgb`` per CLAUDE.md).

WHAT IT REUSES (search-first, named):
  * ``tac.torch_vehicle.driver.import_vendored_bundle`` -> vendored ``build_archive`` /
    ``parse_archive`` (the schedule-agnostic codec; G2 proved bilinear-skip round-trips).
  * ``tac.torch_vehicle.vendored_imports.import_vendored`` -> ``model.HNeRVDecoder`` +
    the vendored ``inflate.py`` / ``inflate.sh`` / ``src/`` runtime tree (assembled into
    the contest packet exactly as ``compress.sh`` does: ``archive.zip`` holds ONLY 0.bin).
  * ``tac.torch_vehicle.configurable_taper_decoder.ConfigurableTaperHNeRVDecoder`` for the
    ``--taper-channels`` small-basis vehicle (the faithful taper generalization).
  * ``tac.torch_vehicle.scorer_context.RealScorerContext.exact_eval`` -> the authority
    advisory d_seg / d_pose / rate / score (vendored ``evaluate_decoder`` + ``compute_score``).
  * The G2 byte-close parity contract (``test_bilinear_skip_byte_close_g2.py``):
    parse-back is a fixed point (weights + latents bit-exact on re-parse).
  * FiLM-v2 (the arm_b production run): when the checkpoint is a pose-FiLM WRAPPER
    state dict (``decoder.*`` inner decoder + ``pose_mlp`` + ``film_resid`` + ``stored_pose``),
    the harness AUTO-DETECTS the version and routes the byte-close + eval through the
    DRIVER's PROVEN FiLM path (``pose_film_v2.build_archive_with_pose`` /
    ``parse_pose_section`` / ``wrapper_sd_to_archive_decoder_sd`` / ``_FiLMEvalDecoder``) —
    the SAME helpers ``driver._build_archive_and_eval_decoder`` uses for the in-loop
    byte-close. NO codec is reimplemented: the additive ~1 KB pose section is appended to
    the pristine vendored 3-section archive, and the parse-back parity additionally proves
    the ``stored_pose`` section round-trips bit-exact. A PLAIN decoder keeps the legacy
    (byte-identical) route — backward-compatible.
  * The byte-close path is the PRISTINE vendored 3-section archive (meta + decoder + latents),
    byte-identical to the basin's ``best_archive.bin``. The P1b export hooks (commit 460c59efc:
    ``losses.variable_level_codec.build_decoder_blob_variable_or_vendored`` /
    ``torch_vehicle.distortion_finishing_kit.serialize_distortion_section`` /
    ``losses.cross_pair_latent_codec.build_latent_blob_dedup_or_vendored``) are the
    documented extension points a future ``--variable-level`` / ``--distortion-kit`` /
    ``--latent-dedup`` flag would call in step 2 — NOT yet wired (the driver already wires
    them via ``cfg.lever4_variable_level_export_enabled``; this harness verifies the default
    vendored packet a production run emits today).

BYTE ACCOUNTING (the rate term the contest actually scores):
  ``upstream/evaluate.py:63`` measures ``(submission_dir / 'archive.zip').stat().st_size``
  and ``rate = compressed_size / uncompressed_size`` where ``uncompressed_size`` =
  ``upstream/videos/`` total (= ``0.mkv`` = 37,545,489 B). The harness reports BOTH the raw
  ``0.bin`` size (what the in-loop advisory uses as a rate proxy) AND the REAL ``archive.zip``
  ``st_size`` (the contest rate term, with the small zip-container overhead), and computes the
  HEADLINE S from the ``archive.zip`` size so the advisory S matches the contest's accounting.

AUTHORITY: ``[contest-CPU advisory] NON-PROMOTABLE``. CPU only (no MPS; no CUDA; no remote;
no paid eval). The reported S is advisory until the SAME packet is run through
``upstream/evaluate.py`` on contest-compliant Linux x86_64 CPU / CUDA (the G3 exact row).
NO score / frontier / promotion claim is made.

Usage:
    .venv/bin/python tools/verify_e2e_byte_close_eval.py \\
        --ckpt-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best \\
        [--max-pairs 600] [--taper-channels 22,16,15,14,15,14,10] \\
        [--out reports/e2e_byte_close_eval_<ts>.json] [--keep-packet]

Default ckpt-dir = the converged n600 base_ch=20 basin. ``--max-pairs`` caps the eval
subset for SPEED (the full 600-pair CPU eval streams the whole video and is slow); the
byte-close ALWAYS uses ALL latents (the archive bytes are the full vehicle). Writes a JSON
report to ``reports/`` (tracked -> durable signal).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RATE_DENOM_DEFAULT = 37_545_489  # upstream/videos/0.mkv st_size (the contest rate denom)
_AUTHORITY = "[contest-CPU advisory] NON-PROMOTABLE"
_EVAL_H, _EVAL_W = 384, 512


# ---------------------------------------------------------------------------
# checkpoint loading (mirrors probe_dseg_sensitivity_map_basin_n600.py)
# ---------------------------------------------------------------------------
def _load_checkpoint(ckpt_dir: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Load (decoder_state_dict, latents, meta) from a driver ``best/`` dir.

    Refuses to fabricate anything (NO-FAKE): missing files raise.
    """
    dec_path = ckpt_dir / "best_ema_decoder.pt"
    lat_path = ckpt_dir / "best_ema_latents.pt"
    if not dec_path.exists() or not lat_path.exists():
        raise FileNotFoundError(
            f"checkpoint missing: {dec_path} / {lat_path}. "
            "Refusing to fabricate an archive/score (NO-FAKE)."
        )
    dec_sd = torch.load(dec_path, map_location="cpu", weights_only=False)
    if isinstance(dec_sd, dict) and "state_dict" in dec_sd:
        dec_sd = dec_sd["state_dict"]
    latents = torch.load(lat_path, map_location="cpu", weights_only=False)
    if isinstance(latents, dict):
        latents = latents.get("latents", next(iter(latents.values())))
    if not torch.is_tensor(latents) or latents.dim() != 2:
        raise ValueError(f"latents must be a 2-D (n_pairs, latent_dim) tensor; got {type(latents)}")

    meta_path = ckpt_dir / "best_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return dec_sd, latents, meta


# ---------------------------------------------------------------------------
# FiLM-v2 wrapper detection (the arm_b production checkpoint is FiLM-v2)
# ---------------------------------------------------------------------------
# A driver ``best/`` dir written by a pose-FiLM run saves the WRAPPER state dict
# (``best_ema_decoder.pt`` = the EMA shadow of the wrapper), NOT a bare vendored
# decoder. The wrapper holds the vendored decoder under the ``decoder.`` submodule
# prefix + the FiLM submodule weights + the non-trainable ``stored_pose`` buffer.
# Detection mirrors ``driver._load_warm_start_into`` / ``_FILM_PARAM_PREFIXES``
# EXACTLY so the harness routes identically to the production byte-close:
#   * v1 (``pose_film.PoseFiLMHNeRVWrapper``): stem-injection FiLM -> ``pose_film.*``
#   * v2 (``pose_film_v2.PoseFiLMHNeRVWrapperV2``): residual-rgb0 FiLM -> ``pose_mlp.*`` +
#     ``film_resid.*`` (the arm_b production wrapper).
# A PLAIN vendored / ConfigurableTaper decoder has NO ``decoder.`` prefix and no
# ``stored_pose`` buffer, so ``stored_pose`` presence uniquely flags a FiLM wrapper.
_FILM_PARAM_PREFIXES: dict[int, tuple[str, ...]] = {
    1: ("pose_film.",),
    2: ("pose_mlp.", "film_resid."),
}


def _detect_film_version(dec_sd: dict[str, torch.Tensor]) -> int | None:
    """Return the pose-FiLM wrapper version (1 or 2) for a wrapper state dict, or
    ``None`` for a plain vendored / ConfigurableTaper decoder (backward-compatible).

    The ``stored_pose`` buffer is the unambiguous FiLM marker (a plain decoder never
    has it); the FiLM-key prefixes then disambiguate v1 (``pose_film.``) from v2
    (``pose_mlp.`` + ``film_resid.``). Keying on ``stored_pose`` + the prefix set is
    the SAME signature ``driver._FILM_PARAM_PREFIXES`` / ``_load_warm_start_into`` use.
    """
    keys = set(dec_sd.keys())
    if "stored_pose" not in keys:
        return None  # plain decoder — the legacy path is unchanged
    has_v2 = any(k.startswith("pose_mlp.") or k.startswith("film_resid.") for k in keys)
    has_v1 = any(k.startswith("pose_film.") for k in keys)
    if has_v2:
        return 2
    if has_v1:
        return 1
    raise ValueError(
        "checkpoint has a stored_pose buffer but no recognized FiLM keys "
        "(pose_mlp.* / film_resid.* for v2, pose_film.* for v1); refusing to "
        "guess the wrapper version (NO-FAKE)."
    )


def _infer_dims(
    dec_sd: dict[str, torch.Tensor], latents: torch.Tensor, meta: dict[str, Any]
) -> tuple[int, int, int]:
    """Return (latent_dim, base_channels, n_pairs), inferring from tensors when meta is silent.

    ``base_channels`` is inferred from ``stem.weight`` = (C0*48, latent_dim) -> C0 = rows/48
    (the vendored decoder's channels[0] == base_channels), with meta override honored.
    Handles BOTH a plain decoder (bare ``stem.weight``) and a FiLM wrapper (the vendored
    stem lives under the ``decoder.stem.weight`` prefix).
    """
    latent_dim = int(meta.get("latent_dim") or latents.shape[1])
    n_pairs = int(latents.shape[0])
    base_channels = meta.get("base_channels")
    if base_channels is None:
        stem_w = dec_sd.get("stem.weight")
        if stem_w is None:
            stem_w = dec_sd.get("decoder.stem.weight")  # FiLM-wrapper inner decoder
        if stem_w is None:
            raise ValueError("cannot infer base_channels: meta lacks it and stem.weight absent")
        base_channels = int(stem_w.shape[0] // (6 * 8))  # C0 * base_h(6) * base_w(8)
    return latent_dim, int(base_channels), n_pairs


def _infer_film_hidden(dec_sd: dict[str, torch.Tensor], film_version: int) -> int:
    """Infer the FiLM ``film_hidden`` (cond_dim) from the wrapper state dict so the
    rebuilt eval wrapper is bit-identical to the production one.

    v2: ``pose_mlp.fc1.weight`` is ``(cond_dim, pose_dim=6)`` -> rows = cond_dim.
    v1: ``pose_film`` MLP first layer ``(hidden, pose_dim)`` -> rows = hidden.
    Defaults to the driver default (8) only when the tensor is somehow absent.
    """
    if film_version == 2:
        w = dec_sd.get("pose_mlp.fc1.weight")
        if w is not None and w.dim() == 2:
            return int(w.shape[0])
    else:
        for k, v in dec_sd.items():
            if k.startswith("pose_film.") and torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == 6:
                return int(v.shape[0])
    return 8  # driver default (pose_film_hidden: int = 8)


# ---------------------------------------------------------------------------
# decoder factory (reuses vendored HNeRVDecoder OR ConfigurableTaperHNeRVDecoder)
# ---------------------------------------------------------------------------
def _build_decoder(
    latent_dim: int,
    base_channels: int,
    taper_channels: list[int] | None,
    device: torch.device,
) -> nn.Module:
    """Build a fresh eval decoder on ``device`` — the vendored one (DEFAULT) or the
    faithful configurable-taper generalization when ``--taper-channels`` is given.

    Mirrors ``driver._new_vendored_decoder`` exactly so the harness decoder is bit-identical
    to what the production run / inflate.py reconstructs.
    """
    if taper_channels is not None:
        from tac.torch_vehicle.configurable_taper_decoder import (
            ConfigurableTaperHNeRVDecoder,
        )

        return ConfigurableTaperHNeRVDecoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            eval_size=(_EVAL_H, _EVAL_W),
            channels=list(taper_channels),
        ).to(device).eval()
    from tac.torch_vehicle.vendored_imports import import_vendored

    model_mod = import_vendored("model")
    return model_mod.HNeRVDecoder(
        latent_dim=latent_dim, base_channels=base_channels, eval_size=(_EVAL_H, _EVAL_W)
    ).to(device).eval()


# ---------------------------------------------------------------------------
# archive section byte breakdown (the vendored 3-section grammar; codec.py:145-167)
# ---------------------------------------------------------------------------
def _section_byte_breakdown(archive: bytes) -> dict[str, int]:
    """Decompose the vendored archive into its 3 length-prefixed sections.

    Grammar (codec.py:148-152): [meta_len:u32][meta][dec_len:u32][dec][lat_len:u32][lat].
    Returns the per-section payload byte counts + the 12 B of u32 length prefixes, so the
    operator sees exactly where the 0.bin bytes live. Tolerant: returns ``{}`` if the bytes
    are not the vendored 3-section grammar (e.g. an additive FiLM-pose section is appended).
    """
    try:
        buf = io.BytesIO(archive)
        out: dict[str, int] = {}
        for name in ("meta_brotli", "decoder_blob", "latents_brotli"):
            (n,) = struct.unpack("<I", buf.read(4))
            buf.read(n)
            out[name] = int(n)
        out["length_prefixes"] = 12
        out["consumed"] = buf.tell()
        out["trailing"] = len(archive) - buf.tell()  # >0 if an extra (e.g. pose) section follows
        return out
    except (struct.error, ValueError):
        return {}


# ---------------------------------------------------------------------------
# byte-close: vendored codec build + parse-back parity (reuses the G2 contract)
# ---------------------------------------------------------------------------
def _byte_close_and_verify_parity(
    vendored_bundle: Any,
    dec_sd: dict[str, torch.Tensor],
    latents: torch.Tensor,
    meta_dict: dict[str, Any],
) -> tuple[bytes, dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Build the REAL 0.bin via the vendored codec, then VERIFY parse-back is a fixed point.

    The G2 parity contract (test_bilinear_skip_byte_close_g2.py): re-building the archive
    from the parse-back state and re-parsing is BIT-EXACT on weights + latents. We assert it
    here so a silent codec drift fails the harness (NO-FAKE).

    Returns (archive_bytes, parseback_decoder_sd, parseback_latents, parity_status).
    """
    archive = vendored_bundle.build_archive(dec_sd, latents, meta_dict=meta_dict)
    dec_sd_back, lat_back, meta_back = vendored_bundle.parse_archive(archive)

    # Determinism: identical inputs -> byte-identical archive (G2 Proof C).
    archive2 = vendored_bundle.build_archive(dec_sd, latents, meta_dict=meta_dict)
    deterministic = archive == archive2

    # Fixed-point: re-encode the parse-back state, re-parse -> bit-exact (G2 Proof B).
    arc_fp = vendored_bundle.build_archive(dec_sd_back, lat_back, meta_dict=meta_dict)
    sd_fp, lat_fp, _ = vendored_bundle.parse_archive(arc_fp)
    weights_fixed_point = all(
        torch.equal(dec_sd_back[k], sd_fp[k]) for k in dec_sd_back if torch.is_tensor(dec_sd_back[k])
    )
    latents_fixed_point = bool(torch.equal(lat_back, lat_fp))

    # Key parity: the parse-back state dict must carry EXACTLY the decoder's keys.
    keys_match = set(dec_sd_back.keys()) == {k for k in dec_sd if torch.is_tensor(dec_sd[k])}

    status = {
        "build_deterministic": bool(deterministic),
        "weights_fixed_point": bool(weights_fixed_point),
        "latents_fixed_point": bool(latents_fixed_point),
        "keys_match": bool(keys_match),
        "meta_roundtrip_base_channels": int(meta_back.get("base_channels", -1)),
        "parity_ok": bool(deterministic and weights_fixed_point and latents_fixed_point and keys_match),
    }
    if not status["parity_ok"]:
        raise RuntimeError(
            f"byte-close parse-back parity FAILED (NO-FAKE): {status}. "
            "The codec round-trip is not a fixed point; refusing to report a score."
        )
    return archive, dec_sd_back, lat_back, status


# ---------------------------------------------------------------------------
# FiLM-v2 byte-close: REUSE the driver's PROVEN additive-pose-section path
# ---------------------------------------------------------------------------
def _film_helpers(film_version: int):
    """Import the canonical version-aware FiLM helpers the DRIVER reuses (no codec
    is reimplemented here — these are the exact modules ``driver._build_archive_and_
    eval_decoder`` imports for the in-loop byte-close).

    Returns (PoseFiLMWrapper, _FiLMEvalDecoder, build_archive_with_pose,
    parse_pose_section, wrapper_sd_to_archive_decoder_sd).
    """
    if film_version == 2:
        from tac.torch_vehicle.pose_film_v2 import (
            PoseFiLMHNeRVWrapperV2 as _PoseFiLMWrapper,
        )
        from tac.torch_vehicle.pose_film_v2 import (
            _FiLMEvalDecoder,
            build_archive_with_pose,
            parse_pose_section,
            wrapper_sd_to_archive_decoder_sd,
        )
    else:
        from tac.torch_vehicle.pose_film import (
            PoseFiLMHNeRVWrapper as _PoseFiLMWrapper,
        )
        from tac.torch_vehicle.pose_film import (
            _FiLMEvalDecoder,
            build_archive_with_pose,
            parse_pose_section,
            wrapper_sd_to_archive_decoder_sd,
        )
    return (
        _PoseFiLMWrapper,
        _FiLMEvalDecoder,
        build_archive_with_pose,
        parse_pose_section,
        wrapper_sd_to_archive_decoder_sd,
    )


def _byte_close_and_verify_parity_film(
    vendored_bundle: Any,
    wrapper_sd: dict[str, torch.Tensor],
    latents: torch.Tensor,
    meta_dict: dict[str, Any],
    film_version: int,
) -> tuple[bytes, dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Byte-close a FiLM-wrapped checkpoint via the DRIVER's additive-pose-section
    grammar (``build_archive_with_pose``), then VERIFY parse-back is a fixed point —
    INCLUDING the additive ``stored_pose`` section round-trip.

    Mirrors ``driver._build_archive_and_eval_decoder`` step (1)-(2): split the wrapper
    state dict into the codec decoder blob (bare vendored keys + FiLM weights, via
    ``wrapper_sd_to_archive_decoder_sd``) + the ``stored_pose`` buffer (additive pose
    section), build ``vendored_archive + encode_pose_section``, and re-parse all 4
    sections. The vendored codec stays PRISTINE (the pose section is strictly appended).

    Returns (archive_bytes, archive_decoder_sd_back, parseback_latents, parity_status).
    ``archive_decoder_sd_back`` carries the bare vendored keys + the FiLM keys (NOT the
    ``decoder.`` prefix), exactly what the FiLM wrapper's eval rebuild consumes.
    """
    (_, _, build_archive_with_pose, parse_pose_section, wrapper_sd_to_archive_decoder_sd) = (
        _film_helpers(film_version)
    )

    archive_decoder_sd = wrapper_sd_to_archive_decoder_sd(wrapper_sd)
    if "stored_pose" not in wrapper_sd:
        raise ValueError("FiLM byte-close requires a stored_pose buffer (NO-FAKE)")
    stored_pose = wrapper_sd["stored_pose"]

    def _build(md: dict[str, Any]) -> bytes:
        return build_archive_with_pose(
            vendored_bundle.build_archive, archive_decoder_sd, latents, md, stored_pose
        )

    archive = _build(meta_dict)
    # The vendored parse reads only the 3 base sections (it ignores the trailing pose
    # bytes); the additive pose section is parsed separately (the additive grammar).
    dec_sd_back, lat_back, meta_back = vendored_bundle.parse_archive(archive)
    pose_back = parse_pose_section(archive, vendored_bundle.parse_archive)

    # Determinism (G2 Proof C): identical inputs -> byte-identical archive.
    deterministic = archive == _build(meta_dict)

    # Fixed-point (G2 Proof B): re-encode the parse-back state -> bit-exact re-parse,
    # AND the additive pose section round-trips bit-exact.
    arc_fp = build_archive_with_pose(
        vendored_bundle.build_archive, dec_sd_back, lat_back, meta_dict, pose_back
    )
    sd_fp, lat_fp, _ = vendored_bundle.parse_archive(arc_fp)
    pose_fp = parse_pose_section(arc_fp, vendored_bundle.parse_archive)
    weights_fixed_point = all(
        torch.equal(dec_sd_back[k], sd_fp[k]) for k in dec_sd_back if torch.is_tensor(dec_sd_back[k])
    )
    latents_fixed_point = bool(torch.equal(lat_back, lat_fp))
    pose_present = pose_back is not None and pose_fp is not None
    pose_fixed_point = bool(pose_present and torch.equal(pose_back, pose_fp))

    # Key parity: the parse-back decoder blob carries EXACTLY the archive decoder keys
    # (the bare vendored keys + the FiLM keys; the stored_pose buffer is in the pose
    # section, NOT the blob, so it is excluded here by construction).
    keys_match = set(dec_sd_back.keys()) == {
        k for k in archive_decoder_sd if torch.is_tensor(archive_decoder_sd[k])
    }

    status = {
        "film_version": int(film_version),
        "build_deterministic": bool(deterministic),
        "weights_fixed_point": bool(weights_fixed_point),
        "latents_fixed_point": bool(latents_fixed_point),
        "pose_section_present": bool(pose_present),
        "pose_section_fixed_point": bool(pose_fixed_point),
        "keys_match": bool(keys_match),
        "meta_roundtrip_base_channels": int(meta_back.get("base_channels", -1)),
        "parity_ok": bool(
            deterministic
            and weights_fixed_point
            and latents_fixed_point
            and pose_fixed_point
            and keys_match
        ),
    }
    if not status["parity_ok"]:
        raise RuntimeError(
            f"FiLM byte-close parse-back parity FAILED (NO-FAKE): {status}. "
            "The additive-pose codec round-trip is not a fixed point; refusing to report a score."
        )
    return archive, dec_sd_back, lat_back, status


def _build_film_eval_decoder(
    archive_decoder_sd: dict[str, torch.Tensor],
    parsed_pose: torch.Tensor,
    latent_dim: int,
    base_channels: int,
    n_pairs: int,
    film_version: int,
    film_hidden: int,
    taper_channels: list[int] | None,
    device: torch.device,
) -> nn.Module:
    """Rebuild the FiLM wrapper + cursor eval adapter from the PARSE-BACK blob, EXACTLY
    as ``driver._build_archive_and_eval_decoder`` step (3) does, so the harness eval
    renders the SAME FiLM-conditioned frames the inflate path produces (byte-closed
    faithful). The FiLM keys keep their wrapper prefixes; the rest are the bare vendored
    decoder keys.
    """
    (PoseFiLMWrapper, _FiLMEvalDecoder, _, _, _) = _film_helpers(film_version)
    prefixes = _FILM_PARAM_PREFIXES[film_version]
    film_sd = {
        k: v for k, v in archive_decoder_sd.items() if any(k.startswith(p) for p in prefixes)
    }
    dec_sd = {
        k: v for k, v in archive_decoder_sd.items() if not any(k.startswith(p) for p in prefixes)
    }
    # The inner vendored decoder — vendored (DEFAULT) or the configurable taper (the
    # v2 wrapper wraps a ConfigurableTaperHNeRVDecoder when --taper-channels is given).
    vendored = _build_decoder(latent_dim, base_channels, taper_channels, device)
    vendored.load_state_dict({k: v.to(device) for k, v in dec_sd.items()})
    # v1 + v2 share the (decoder, *, n_pairs, film_hidden) constructor signature.
    wrapper = PoseFiLMWrapper(vendored, n_pairs=n_pairs, film_hidden=film_hidden).to(device)
    # Load the FiLM submodule weights over the freshly-built (identity-init) FiLM,
    # keeping the vendored decoder keys already loaded (strict=False: the merged dict
    # is the FULL wrapper state, no missing/unexpected in practice).
    wrapper.load_state_dict(
        {**wrapper.state_dict(), **{k: v.to(device) for k, v in film_sd.items()}},
        strict=False,
    )
    if parsed_pose is not None:
        wrapper.set_stored_pose(parsed_pose.to(device))
    wrapper.eval()
    eval_dec = _FiLMEvalDecoder(wrapper)
    eval_dec.eval()  # resets the cursor to pair 0
    return eval_dec


# ---------------------------------------------------------------------------
# authority-advisory eval (the SAME vendored primitives RealScorerContext.exact_eval uses)
# ---------------------------------------------------------------------------
def _authority_exact_eval(
    eval_decoder: nn.Module,
    eval_latents: torch.Tensor,
    archive_bytes: int,
    video_path: str | Path,
    rate_denom: int,
) -> dict[str, float]:
    """Run the authority-faithful advisory eval on the parse-back decoder.

    This is byte-for-byte the same eval ``RealScorerContext.exact_eval`` performs (vendored
    ``score.evaluate_decoder`` streams GT via ``frame_utils.yuv420_to_rgb`` + does the SAME
    decoder forward -> bicubic 874x1164 -> uint8 -> frozen SegNet/PoseNet that ``inflate.py`` +
    ``evaluate.py`` do; then ``score.compute_score``). We call those vendored primitives
    DIRECTLY — rather than via ``RealScorerContext`` — because the eval path does NOT use the
    per-step ``seg_targets_hard`` / ``pose_targets`` precompute (it streams GT itself), so the
    context's target precompute (which for n=600 loads a ~900 MB cache into RAM) is pure
    overhead the eval-only harness must not pay. Authority device is CPU (NEVER MPS).

    The rate term uses ``archive_bytes`` (we pass the REAL ``archive.zip`` st_size) over the
    contest denominator ``rate_denom`` (default = the source video st_size, exactly
    ``evaluate.py:64``'s ``uncompressed_size``).
    """
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.vendored_imports import import_vendored

    score_mod = import_vendored("score")
    distortion_net = load_frozen_distortion_net(device="cpu")  # CPU AUTHORITY (no MPS)
    dist = score_mod.evaluate_decoder(
        eval_decoder, eval_latents.to("cpu"), distortion_net, str(video_path),
        batch_pairs=8, device="cpu",
    )
    # compute_score uses total_video_bytes(video) as the denom; we pass our explicit denom so
    # the harness honors --rate-denom while defaulting to the contest's uncompressed_size.
    result = score_mod.compute_score(
        dist["seg_distortion"], dist["pose_distortion"], archive_bytes, rate_denom
    )
    return {
        "seg_distortion": float(result["seg_distortion"]),
        "pose_distortion": float(result["pose_distortion"]),
        "rate": float(result["rate"]),
        "score": float(result["score"]),
    }


# ---------------------------------------------------------------------------
# contest packet assembly (mirrors compress.sh + evaluate.sh: archive.zip holds ONLY 0.bin)
# ---------------------------------------------------------------------------
def _assemble_contest_packet(
    archive_bin: bytes, packet_dir: Path
) -> tuple[Path, int, list[str]]:
    """Assemble the contest submission packet exactly as the vendored ``compress.sh`` /
    ``evaluate.sh`` contract requires, and return (archive_zip_path, zip_st_size, runtime_files).

    The packet layout (per evaluate.sh): the rate term measures
    ``(submission_dir / 'archive.zip').stat().st_size`` and ``archive.zip`` contains ONLY the
    per-video ``0.bin`` (``compress.sh``: ``zip -j archive.zip 0.bin``). The runtime tree
    (``inflate.sh`` / ``inflate.py`` / ``src/``) lives BESIDE the zip in ``submission_dir/`` and
    is NOT counted in the rate. We write the real ``0.bin`` + zip it (deterministic: fixed
    timestamp, no extra members) and copy the pristine vendored runtime so the packet is a
    runnable contest submission_dir (the G3 input). Returns the zip st_size = the contest rate
    numerator.
    """
    packet_dir.mkdir(parents=True, exist_ok=True)
    bin_path = packet_dir / "0.bin"
    bin_path.write_bytes(archive_bin)

    # Deterministic zip: a single ZipInfo with a fixed date so the st_size is reproducible
    # (matching `zip -j archive.zip 0.bin` semantics: the basename only, no dir component).
    zip_path = packet_dir / "archive.zip"
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, archive_bin)

    # Copy the pristine vendored runtime tree (inflate.sh / inflate.py / src/) BESIDE the
    # zip so the packet_dir is a runnable submission_dir for the contest evaluate.sh / G3.
    from tac.torch_vehicle.vendored_imports import VENDORED_SRC

    vendored_root = VENDORED_SRC.parent  # .../submissions/hnerv_muon (holds inflate.{py,sh} + src/)
    runtime_files: list[str] = []
    for name in ("inflate.py", "inflate.sh", "README.md"):
        src = vendored_root / name
        if src.exists():
            shutil.copy2(src, packet_dir / name)
            runtime_files.append(name)
    src_dir = vendored_root / "src"
    if src_dir.exists():
        dst_src = packet_dir / "src"
        if dst_src.exists():
            shutil.rmtree(dst_src)
        shutil.copytree(
            src_dir, dst_src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        runtime_files.append("src/")

    return zip_path, int(zip_path.stat().st_size), runtime_files


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(
    ckpt_dir: Path,
    *,
    max_pairs: int | None,
    taper_channels: list[int] | None,
    rate_denom: int | None,
    keep_packet: bool,
    packet_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the full byte-close -> eval pipeline and return the advisory report dict.

    The headline S uses the REAL ``archive.zip`` st_size for the rate term (the contest's
    accounting). Always ``[contest-CPU advisory] NON-PROMOTABLE``.
    """
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.vendored_imports import import_vendored

    vb = import_vendored_bundle()
    video_path = import_vendored("data").get_default_video_path()

    # Rate denominator: default to the SOURCE video st_size (= evaluate.py:64 uncompressed_size).
    if rate_denom:
        denom = int(rate_denom)
    else:
        vp = Path(video_path)
        denom = vp.stat().st_size if vp.exists() else _RATE_DENOM_DEFAULT

    dec_sd, latents, meta = _load_checkpoint(ckpt_dir)
    latent_dim, base_channels, n_pairs = _infer_dims(dec_sd, latents, meta)
    decoder_params = int(sum(v.numel() for v in dec_sd.values() if torch.is_tensor(v)))

    # FiLM-wrapper detection: a pose-FiLM run (the arm_b production run is FiLM-v2)
    # saves the WRAPPER state dict (decoder.* + FiLM weights + stored_pose). When
    # detected, route the byte-close + eval through the DRIVER's proven additive-pose
    # path; a plain decoder keeps the legacy (byte-identical) route.
    film_version = _detect_film_version(dec_sd)
    film_hidden = _infer_film_hidden(dec_sd, film_version) if film_version else None

    meta_dict = {
        "n_pairs": n_pairs,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "eval_size": [_EVAL_H, _EVAL_W],
    }

    film_tag = f"v{film_version}(hidden={film_hidden})" if film_version else "off"
    print(f"[ckpt] {ckpt_dir}  latents={tuple(latents.shape)}  base_ch={base_channels}  "
          f"params={decoder_params}  film={film_tag}  {_AUTHORITY}", flush=True)

    # 1+2. Byte-close (ALL latents) + parse-back parity (G2 contract). The FiLM route
    # uses the additive-pose-section grammar (build_archive_with_pose) and additionally
    # verifies the stored_pose section round-trips bit-exact.
    if film_version:
        archive, dec_sd_back, lat_back, parity = _byte_close_and_verify_parity_film(
            vb, dec_sd, latents, meta_dict, film_version
        )
    else:
        archive, dec_sd_back, lat_back, parity = _byte_close_and_verify_parity(
            vb, dec_sd, latents, meta_dict
        )
    bin_bytes = len(archive)
    section_bytes = _section_byte_breakdown(archive)
    print(f"[byte-close] 0.bin={bin_bytes} B  parity_ok={parity['parity_ok']}  "
          f"sections={section_bytes}", flush=True)

    # 3. Assemble the REAL contest archive.zip (containing 0.bin) + runtime tree.
    packet_dir = packet_dir or (
        _REPO_ROOT / "experiments" / "results"
        / f"e2e_byte_close_packet_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    zip_path, zip_bytes, runtime_files = _assemble_contest_packet(archive, packet_dir)
    zip_overhead = zip_bytes - bin_bytes
    print(f"[packet] archive.zip={zip_bytes} B (0.bin {bin_bytes} + zip overhead {zip_overhead})  "
          f"runtime={runtime_files}  dir={packet_dir}", flush=True)

    # 4. Authority-advisory eval on the PARSE-BACK decoder (the contest-visible artifact).
    if film_version:
        # FiLM route: rebuild the wrapper + cursor eval adapter from the parse-back blob
        # + the parsed stored-pose section (the SAME FiLM-conditioned render inflate.py
        # produces). REUSES the driver's step-(3) logic via _build_film_eval_decoder.
        (_, _, _, parse_pose_section, _) = _film_helpers(film_version)
        parsed_pose = parse_pose_section(archive, vb.parse_archive)
        eval_dec = _build_film_eval_decoder(
            dec_sd_back, parsed_pose, latent_dim, base_channels, n_pairs,
            film_version, int(film_hidden), taper_channels, torch.device("cpu"),
        )
    else:
        eval_dec = _build_decoder(latent_dim, base_channels, taper_channels, torch.device("cpu"))
        missing, unexpected = eval_dec.load_state_dict(dict(dec_sd_back), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"parse-back decoder load mismatch (NO-FAKE): missing={missing} "
                f"unexpected={unexpected}"
            )

    eval_max = n_pairs if max_pairs is None else min(int(max_pairs), n_pairs)
    eval_latents = lat_back[:eval_max]
    # Authority-advisory eval: the REAL archive.zip st_size feeds the rate term (the contest's
    # accounting), the source-video st_size is the denominator (evaluate.py:64). Direct vendored
    # primitives (no target precompute / no 900 MB cache load) — see _authority_exact_eval.
    ev = _authority_exact_eval(eval_dec, eval_latents, zip_bytes, video_path, denom)
    d_seg = float(ev["seg_distortion"])
    d_pose = float(ev["pose_distortion"])

    # 5. Recompute S FROM components (do NOT trust a rounded field).
    rate = zip_bytes / denom
    seg_term = 100.0 * d_seg
    pose_term = (10.0 * d_pose + 1e-12) ** 0.5
    rate_term = 25.0 * rate
    score = seg_term + pose_term + rate_term

    # Also report the .bin-based S (what the in-loop advisory used) for comparison.
    rate_bin = bin_bytes / denom
    score_bin = seg_term + pose_term + 25.0 * rate_bin

    report: dict[str, Any] = {
        "authority": _AUTHORITY,
        "promotion_claim": False,
        "score_axis": "contest-CPU-advisory",
        "ckpt_dir": str(ckpt_dir),
        "video_path": str(video_path),
        "rate_denominator_bytes": denom,
        "n_pairs_total": n_pairs,
        "eval_pairs": int(eval_latents.shape[0]),
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "taper_channels": list(taper_channels) if taper_channels else None,
        "pose_film_version": film_version,  # None (plain) | 1 (stem) | 2 (residual-rgb0)
        "pose_film_hidden": film_hidden,
        "decoder_params": decoder_params,
        # byte accounting (the rate term)
        "archive_bin_bytes": bin_bytes,
        "archive_zip_bytes": zip_bytes,
        "zip_container_overhead_bytes": zip_overhead,
        "section_byte_breakdown": section_bytes,
        "runtime_files": runtime_files,
        # parse-back parity
        "parseback_parity": parity,
        # score components (HEADLINE uses archive.zip bytes)
        "d_seg": d_seg,
        "d_pose": d_pose,
        "rate": rate,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "score_S": score,
        # comparison: S computed from the raw .bin size (the in-loop advisory proxy)
        "rate_bin": rate_bin,
        "score_S_from_bin_bytes": score_bin,
        "archive_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "packet_dir": str(packet_dir),
    }
    # The basin's recorded advisory (if present) for an apples-to-apples sanity line.
    if meta:
        report["checkpoint_recorded_advisory"] = {
            "d_seg": meta.get("d_seg"),
            "d_pose": meta.get("d_pose"),
            "score": meta.get("score"),
            "archive_bytes": meta.get("archive_bytes"),
        }

    print(
        f"[eval] d_seg={d_seg:.6f} d_pose={d_pose:.6f} | "
        f"seg={seg_term:.4f} pose={pose_term:.4f} rate={rate_term:.4f} | "
        f"S={score:.4f} (zip) / {score_bin:.4f} (bin)  over {eval_latents.shape[0]} pairs  "
        f"{_AUTHORITY}",
        flush=True,
    )

    if not keep_packet:
        shutil.rmtree(packet_dir, ignore_errors=True)
        report["packet_dir"] = "(deleted; pass --keep-packet to retain)"

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ckpt-dir", type=Path,
        default=_REPO_ROOT / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best",
        help="driver best/ dir with best_ema_decoder.pt + best_ema_latents.pt + best_meta.json",
    )
    ap.add_argument(
        "--max-pairs", type=int, default=None,
        help="cap the eval subset for SPEED (default: all latents). Byte-close ALWAYS uses all.",
    )
    ap.add_argument(
        "--taper-channels", type=str, default=None,
        help="7-stage configurable taper, e.g. 22,16,15,14,15,14,10 (default: vendored decoder)",
    )
    ap.add_argument(
        "--rate-denom", type=int, default=None,
        help=f"rate denominator (default {_RATE_DENOM_DEFAULT} = upstream/videos/0.mkv st_size)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="JSON report path (default reports/e2e_byte_close_eval_<ts>.json)",
    )
    ap.add_argument(
        "--keep-packet", action="store_true",
        help="retain the assembled submission_dir (archive.zip + runtime) for the G3 exact row",
    )
    args = ap.parse_args(argv)

    taper = None
    if args.taper_channels:
        taper = [int(x) for x in args.taper_channels.replace(" ", "").split(",")]
        if len(taper) != 7:
            ap.error(f"--taper-channels must have 7 comma-separated stages; got {taper}")

    report = run(
        args.ckpt_dir,
        max_pairs=args.max_pairs,
        taper_channels=taper,
        rate_denom=args.rate_denom,
        keep_packet=args.keep_packet,
    )

    out = args.out or (
        _REPO_ROOT / "reports"
        / f"e2e_byte_close_eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {out}  {_AUTHORITY}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
