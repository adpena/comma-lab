# SPDX-License-Identifier: MIT
"""Z7MCM2 byte grammar for the Z7-Mamba-2 recurrent predictive-coding substrate.

This is the byte-closed archive grammar for ``time_traveler_l5_z7_mamba2``.
It is the canonical sister of Z7PCWM1 (Z7-GRU/LSTM) with the predictor
stream replaced by Mamba-2 selective state-space weights. The grammar
preserves:

- ``encoder_blob``: optional context conditioner state_dict (zlib int8/fp16).
- ``decoder_blob``: Z6-compatible PixelShuffle decoder state_dict.
- ``predictor_blob``: Mamba2Predictor state_dict (selective projection
  matrices A_log, B_proj, C_proj, in_proj, out_proj, dt_proj, conv1d).
- ``latent_init_blob``: int8-quantized 1-D latent_init.
- ``residuals_blob``: int8-quantized 2-D residuals (num_pairs, latent_dim).
- ``ego_motion_blob``: int8-quantized 2-D ego_motion buffer.
- ``meta_blob``: sorted JSON metadata with non-promotable authority tags.

Per CLAUDE.md HNeRV parity discipline L3 (Archive grammar = monolithic
single-file ``0.bin``) + L4 (Inflate ≤200 LOC with substrate-engineering
waiver) + L8 (Eval-roundtrip-aware) + L11 (No-op detector via byte-
mutation smoke per Catalog #139 + #272).

The ``replay_latent_sequence_with_context`` helper proves the predictor
section is parse-consumed by reconstructing the latent autoregression:

    z_t = Mamba2(z_{t-1}, ego_motion[t]) + residual[t]

[verified-against: tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.archive Z7PCWM1 pattern]
[verified-against: tac.optimization.mamba2_predictor.Mamba2Predictor.state_dict]
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass

import torch

from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture import (
    normalize_context_conditioning_mode,
)
from tac.substrates.time_traveler_l5_z7_mamba2.architecture import (
    EVAL_HW,
    Z7Mamba2PredictiveCodingConfig,
)

__all__ = [
    "Z7MCM2_HEADER_FMT",
    "Z7MCM2_HEADER_SIZE",
    "Z7MCM2_MAGIC",
    "Z7MCM2_SCHEMA_VERSION",
    "Z7MCM2_SECTION_ROLES",
    "Z7Mamba2PredictiveCodingArchive",
    "pack_archive",
    "parse_archive",
    "parse_z7mcm2_archive_bytes",
    "replay_latent_sequence",
    "replay_latent_sequence_with_context",
]

Z7MCM2_MAGIC: bytes = b"Z7M2"
Z7MCM2_SCHEMA_VERSION: int = 1

# MAGIC(4), VERSION(1), LATENT_DIM(2), EGO_DIM(2), NUM_PAIRS(2),
# D_MODEL(2), D_STATE(1), EXPAND(1), D_CONV(1), FLAGS(1), 7 x u32 section lengths.
Z7MCM2_HEADER_FMT: str = "<4sBHHHHBBBBIIIIIII"
Z7MCM2_HEADER_SIZE: int = struct.calcsize(Z7MCM2_HEADER_FMT)
assert Z7MCM2_HEADER_SIZE == 45, f"Z7MCM2_HEADER_SIZE expected 45, got {Z7MCM2_HEADER_SIZE}"

_ZLIB_LEVEL: int = 9
_FLAG_STATEFUL: int = 1 << 0
_FLAG_IDENTITY: int = 1 << 1


@dataclass(frozen=True)
class Z7Mamba2PredictiveCodingArchive:
    """Parsed Z7MCM2 archive."""

    encoder_state_dict: dict[str, torch.Tensor]
    decoder_state_dict: dict[str, torch.Tensor]
    predictor_state_dict: dict[str, torch.Tensor]
    latent_init: torch.Tensor
    residuals: torch.Tensor
    ego_motion: torch.Tensor
    meta: dict[str, object]
    schema_version: int
    config: Z7Mamba2PredictiveCodingConfig


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize state_dict to zlib-compressed fp16 byte stream."""
    parts: list[bytes] = []
    for key in sorted(sd.keys()):
        tensor = sd[key].detach().to("cpu", dtype=torch.float16).contiguous()
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"state_dict key too long: {key!r}")
        shape = tuple(int(s) for s in tensor.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"tensor {key!r} has too many dims")
        header = struct.pack(
            f"<H{len(key_bytes)}sB" + "I" * len(shape),
            len(key_bytes),
            key_bytes,
            len(shape),
            *shape,
        )
        parts.append(header)
        parts.append(tensor.numpy().tobytes(order="C"))
    return zlib.compress(b"".join(parts), level=_ZLIB_LEVEL)


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    """Deserialize state_dict from zlib-compressed fp16 byte stream."""
    raw = zlib.decompress(blob)
    sd: dict[str, torch.Tensor] = {}
    pos = 0
    while pos < len(raw):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated reading key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        if pos + key_len + 1 > len(raw):
            raise ValueError("state_dict blob truncated reading key")
        key = raw[pos : pos + key_len].decode("utf-8")
        pos += key_len
        (ndim,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        if pos + 4 * ndim > len(raw):
            raise ValueError("state_dict blob truncated reading shape")
        shape = struct.unpack("<" + "I" * ndim, raw[pos : pos + 4 * ndim])
        pos += 4 * ndim
        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * 2
        if pos + nbytes > len(raw):
            raise ValueError(f"state_dict blob truncated reading tensor {key!r}")
        sd[key] = torch.frombuffer(
            bytearray(raw[pos : pos + nbytes]),
            dtype=torch.float16,
        ).clone().reshape(shape)
        pos += nbytes
    return sd


def _quantize_to_int8(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize float tensor to int8 with returned (quantized, scale, zp).

    Per Catalog #161 degenerate-range clamp: the all-equal branch fills
    quantized with `-(MAX_LEVELS // 2)` so exact dequantization returns
    to ``lo`` (the constant value), not `0` mapped through zero_point.
    """
    if t.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"tensor must be float16/float32; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        # Catalog #161 degenerate-range: fill with sentinel that
        # dequantizes back to lo via (q + 127) * 1.0 + lo = lo for q=-127.
        return torch.full_like(f, -127, dtype=torch.int8), 1.0, lo
    scale = (hi - lo) / 254.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 254.0)
    return (q_unsigned - 127.0).to(torch.int8), scale, lo


def _dequantize_from_int8(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    """Dequantize int8 back to float32."""
    return (q.to(torch.float32) + 127.0) * float(scale) + float(zero_point)


def _false_authority_meta(
    base_meta: dict[str, object],
    *,
    config: Z7Mamba2PredictiveCodingConfig,
) -> dict[str, object]:
    """Stamp non-promotable evidence-grade tags into archive metadata.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" +
    "Forbidden misleading-directory-name (phantom-score directory trap)":
    every archive carries explicit ``score_claim=false`` +
    ``promotion_eligible=false`` + ``ready_for_exact_eval_dispatch=false``
    blockers until a paired Tier-C post-training validation + per-substrate
    symposium PROCEED-unconditional lands.
    """
    meta = dict(base_meta)
    score_aware_used = bool(meta.get("score_aware_scorer_loss_used")) or (
        str(meta.get("loss_mode", "")).lower() == "score_aware"
    )
    blockers = [
        (
            "score_aware_trained_packet_not_auth_eval_validated"
            if score_aware_used
            else "proxy_trained_packet_not_score_aware_or_auth_eval_validated"
        ),
        "paired_exact_eval_json_required_for_z7_mamba2_disambiguator",
        "wave_n_plus_1_council_required_for_z7_mamba2",
        "z7_mamba2_post_training_tier_c_validation_required_per_catalog_324",
    ]
    if not score_aware_used:
        blockers.insert(1, "score_aware_training_absent_prebuild")
    context_mode = normalize_context_conditioning_mode(
        str(meta.get("context_conditioning_mode", config.context_conditioning_mode))
    )
    if context_mode != "none":
        blockers.append("context_conditioned_decoder_requires_paired_exact_eval")
    meta.setdefault("context_conditioning_mode", context_mode)
    meta.setdefault("context_affine_strength", float(config.context_affine_strength))
    meta.setdefault(
        "context_conditioner_state_dict_in_encoder_blob",
        context_mode != "none",
    )
    meta["z7_mamba2_recurrent_predictive_coding_meta"] = {
        "schema": "z7_mamba2_recurrent_predictive_coding_meta_v1",
        "substrate_id": "time_traveler_l5_z7_mamba2",
        "archive_grammar": "Z7MCM2",
        "predictor": "mamba2_selective_state_space_predictor",
        "decoder_context_conditioning": context_mode,
        "context_affine_strength": float(config.context_affine_strength),
        "context_conditioner_state_dict_in_encoder_blob": bool(
            meta.get("context_conditioner_state_dict_in_encoder_blob")
        ),
        "loss_mode": "score_aware" if score_aware_used else "proxy",
        "score_aware_scorer_loss_used": score_aware_used,
        "latent_dim": config.latent_dim,
        "ego_motion_dim": config.ego_motion_dim,
        "mamba2_d_model": config.d_model,
        "mamba2_d_state": config.d_state,
        "mamba2_expand": config.expand,
        "mamba2_d_conv": config.d_conv,
        "stateful": config.stateful,
        "identity_predictor": config.identity_predictor,
        "beta_ib": float(config.beta_ib),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "evidence_grade": "archive_grammar_scaffold_only_NOT_promotable",
        "blockers": blockers,
    }
    return meta


def _validate_tensor_shapes(
    latent_init: torch.Tensor,
    residuals: torch.Tensor,
    ego_motion: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate latent/residual/ego_motion shape consistency."""
    if latent_init.dim() != 1:
        raise ValueError(f"latent_init must be 1-D; got {tuple(latent_init.shape)}")
    if residuals.dim() != 2:
        raise ValueError(f"residuals must be 2-D; got {tuple(residuals.shape)}")
    if ego_motion.dim() != 2:
        raise ValueError(f"ego_motion must be 2-D; got {tuple(ego_motion.shape)}")
    latent_dim = int(latent_init.shape[0])
    num_pairs = int(residuals.shape[0])
    ego_dim = int(ego_motion.shape[1])
    if residuals.shape[1] != latent_dim:
        raise ValueError("residuals second dim must match latent_dim")
    if ego_motion.shape[0] != num_pairs:
        raise ValueError("ego_motion first dim must match residuals num_pairs")
    for name, value in (
        ("latent_dim", latent_dim),
        ("num_pairs", num_pairs),
        ("ego_motion_dim", ego_dim),
    ):
        if value <= 0 or value > 0xFFFF:
            raise ValueError(f"{name} {value} out of u16 range")
    return latent_dim, num_pairs, ego_dim


def pack_archive(
    encoder_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    predictor_state_dict: dict[str, torch.Tensor],
    latent_init: torch.Tensor,
    residuals: torch.Tensor,
    ego_motion: torch.Tensor,
    meta: dict[str, object],
    *,
    config: Z7Mamba2PredictiveCodingConfig,
    schema_version: int = Z7MCM2_SCHEMA_VERSION,
) -> bytes:
    """Serialize Z7-Mamba-2 sections into deterministic Z7MCM2 bytes."""
    if schema_version != Z7MCM2_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    latent_dim, num_pairs, ego_dim = _validate_tensor_shapes(
        latent_init,
        residuals,
        ego_motion,
    )
    if config.latent_dim != latent_dim:
        raise ValueError("config.latent_dim does not match latent_init")
    if config.ego_motion_dim != ego_dim:
        raise ValueError("config.ego_motion_dim does not match ego_motion")
    if config.num_pairs != num_pairs:
        raise ValueError("config.num_pairs does not match residuals")
    if config.d_model <= 0 or config.d_model > 0xFFFF:
        raise ValueError("config.d_model out of u16 range")
    if config.d_state <= 0 or config.d_state > 0xFF:
        raise ValueError("config.d_state out of u8 range")
    if config.expand <= 0 or config.expand > 0xFF:
        raise ValueError("config.expand out of u8 range")
    if config.d_conv <= 0 or config.d_conv > 0xFF:
        raise ValueError("config.d_conv out of u8 range")
    context_mode = normalize_context_conditioning_mode(
        config.context_conditioning_mode
    )
    if context_mode != "none" and not encoder_state_dict:
        raise ValueError(
            "context-conditioned Z7MCM2 archives require a context conditioner "
            "state_dict in encoder_state_dict"
        )

    q_latent_init, scale_li, zp_li = _quantize_to_int8(latent_init)
    q_residuals, scale_r, zp_r = _quantize_to_int8(residuals)
    q_ego, scale_e, zp_e = _quantize_to_int8(ego_motion)
    latent_init_blob = q_latent_init.contiguous().numpy().tobytes()
    residuals_blob = q_residuals.contiguous().numpy().tobytes()
    ego_blob = q_ego.contiguous().numpy().tobytes()

    encoder_blob = _serialize_state_dict(encoder_state_dict)
    decoder_blob = _serialize_state_dict(decoder_state_dict)
    predictor_blob = _serialize_state_dict(predictor_state_dict)

    meta_with_tags = _false_authority_meta(meta, config=config)
    meta_with_tags["_latent_init_scale"] = float(scale_li)
    meta_with_tags["_latent_init_zp"] = float(zp_li)
    meta_with_tags["_residuals_scale"] = float(scale_r)
    meta_with_tags["_residuals_zp"] = float(zp_r)
    meta_with_tags["_ego_motion_scale"] = float(scale_e)
    meta_with_tags["_ego_motion_zp"] = float(zp_e)
    meta_blob = json.dumps(
        meta_with_tags,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    flags = 0
    if config.stateful:
        flags |= _FLAG_STATEFUL
    if config.identity_predictor:
        flags |= _FLAG_IDENTITY

    header = struct.pack(
        Z7MCM2_HEADER_FMT,
        Z7MCM2_MAGIC,
        schema_version,
        latent_dim,
        ego_dim,
        num_pairs,
        config.d_model,
        config.d_state,
        config.expand,
        config.d_conv,
        flags,
        len(encoder_blob),
        len(decoder_blob),
        len(predictor_blob),
        len(latent_init_blob),
        len(residuals_blob),
        len(ego_blob),
        len(meta_blob),
    )
    return (
        header
        + encoder_blob
        + decoder_blob
        + predictor_blob
        + latent_init_blob
        + residuals_blob
        + ego_blob
        + meta_blob
    )


def _unpack_header(blob: bytes) -> tuple[object, ...]:
    """Unpack and validate Z7MCM2 header."""
    if len(blob) < Z7MCM2_HEADER_SIZE:
        raise ValueError(
            f"z7mcm2 archive too short: got {len(blob)} bytes, "
            f"need >= {Z7MCM2_HEADER_SIZE}"
        )
    header = struct.unpack(Z7MCM2_HEADER_FMT, blob[:Z7MCM2_HEADER_SIZE])
    magic = header[0]
    version = header[1]
    if magic != Z7MCM2_MAGIC:
        raise ValueError(f"z7mcm2 bad magic: {magic!r}")
    if version != Z7MCM2_SCHEMA_VERSION:
        raise ValueError(f"z7mcm2 unsupported schema version: {version}")
    return header


def parse_z7mcm2_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for Z7MCM2 bytes."""
    (
        _magic,
        _version,
        latent_dim,
        ego_dim,
        num_pairs,
        _d_model,
        _d_state,
        _expand,
        _d_conv,
        _flags,
        encoder_len,
        decoder_len,
        predictor_len,
        latent_init_len,
        residuals_len,
        ego_len,
        meta_len,
    ) = _unpack_header(archive_bytes)
    if latent_init_len != latent_dim:
        raise ValueError("z7mcm2 latent_init_len != latent_dim")
    if residuals_len != int(num_pairs) * int(latent_dim):
        raise ValueError("z7mcm2 residuals_len != num_pairs*latent_dim")
    if ego_len != int(num_pairs) * int(ego_dim):
        raise ValueError("z7mcm2 ego_motion_len != num_pairs*ego_motion_dim")

    end_header = Z7MCM2_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_predictor = end_decoder + int(predictor_len)
    end_latent_init = end_predictor + int(latent_init_len)
    end_residuals = end_latent_init + int(residuals_len)
    end_ego = end_residuals + int(ego_len)
    end_meta = end_ego + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"z7mcm2 archive size {len(archive_bytes)} != expected {end_meta}"
        )
    return {
        "z7mcm2_header": (0, Z7MCM2_HEADER_SIZE),
        "encoder_blob": (end_header, int(encoder_len)),
        "decoder_blob": (end_encoder, int(decoder_len)),
        "predictor_blob": (end_decoder, int(predictor_len)),
        "latent_init_blob": (end_predictor, int(latent_init_len)),
        "residuals_blob": (end_latent_init, int(residuals_len)),
        "ego_motion_blob": (end_residuals, int(ego_len)),
        "meta_blob": (end_ego, int(meta_len)),
    }


def _slice_section(
    blob: bytes,
    sections: dict[str, tuple[int, int]],
    name: str,
) -> bytes:
    start, length = sections[name]
    return blob[start : start + length]


def parse_archive(blob: bytes) -> Z7Mamba2PredictiveCodingArchive:
    """Parse Z7MCM2 bytes back into tensors, state dicts, and metadata."""
    (
        _magic,
        version,
        latent_dim,
        ego_dim,
        num_pairs,
        d_model,
        d_state,
        expand,
        d_conv,
        flags,
        _encoder_len,
        _decoder_len,
        _predictor_len,
        _latent_init_len,
        _residuals_len,
        _ego_len,
        _meta_len,
    ) = _unpack_header(blob)
    sections = parse_z7mcm2_archive_bytes(blob)
    encoder_blob = _slice_section(blob, sections, "encoder_blob")
    decoder_blob = _slice_section(blob, sections, "decoder_blob")
    predictor_blob = _slice_section(blob, sections, "predictor_blob")
    latent_init_blob = _slice_section(blob, sections, "latent_init_blob")
    residuals_blob = _slice_section(blob, sections, "residuals_blob")
    ego_blob = _slice_section(blob, sections, "ego_motion_blob")
    meta_blob = _slice_section(blob, sections, "meta_blob")

    meta = json.loads(meta_blob.decode("utf-8"))
    scale_li = float(meta.pop("_latent_init_scale"))
    zp_li = float(meta.pop("_latent_init_zp"))
    scale_r = float(meta.pop("_residuals_scale"))
    zp_r = float(meta.pop("_residuals_zp"))
    scale_e = float(meta.pop("_ego_motion_scale"))
    zp_e = float(meta.pop("_ego_motion_zp"))

    latent_q = torch.frombuffer(
        bytearray(latent_init_blob),
        dtype=torch.int8,
    ).clone().view(int(latent_dim))
    residuals_q = torch.frombuffer(
        bytearray(residuals_blob),
        dtype=torch.int8,
    ).clone().view(int(num_pairs), int(latent_dim))
    ego_q = torch.frombuffer(
        bytearray(ego_blob),
        dtype=torch.int8,
    ).clone().view(int(num_pairs), int(ego_dim))

    raw_decoder_channels = meta.get("decoder_channels", (32, 24, 16, 12))
    if isinstance(raw_decoder_channels, str):
        decoder_channels = tuple(
            int(part.strip()) for part in raw_decoder_channels.split(",") if part.strip()
        )
    elif isinstance(raw_decoder_channels, (list, tuple)):
        decoder_channels = tuple(int(part) for part in raw_decoder_channels)
    else:
        raise TypeError("z7mcm2 metadata decoder_channels must be list/tuple/string")

    config = Z7Mamba2PredictiveCodingConfig(
        latent_dim=int(latent_dim),
        ego_motion_dim=int(ego_dim),
        d_model=int(d_model),
        d_state=int(d_state),
        expand=int(expand),
        d_conv=int(d_conv),
        # Backend reference_torch at inflate-time to avoid CUDA-kernel
        # dependency on inflate device (per HNeRV parity L4 + L9 runtime
        # closure: inflate runtime must NOT depend on mamba_ssm CUDA
        # kernels for byte-faithful replay).
        backend="reference_torch",
        stateful=bool(int(flags) & _FLAG_STATEFUL),
        identity_predictor=bool(int(flags) & _FLAG_IDENTITY),
        num_pairs=int(num_pairs),
        decoder_embed_dim=int(meta.get("decoder_embed_dim", 32)),
        decoder_initial_grid_h=int(meta.get("decoder_initial_grid_h", 24)),
        decoder_initial_grid_w=int(meta.get("decoder_initial_grid_w", 32)),
        decoder_channels=decoder_channels,
        decoder_num_upsample_blocks=int(
            meta.get("decoder_num_upsample_blocks", 4)
        ),
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
        latent_init_std=float(meta.get("latent_init_std", 0.02)),
        context_conditioning_mode=normalize_context_conditioning_mode(
            str(meta.get("context_conditioning_mode", "none"))
        ),
        context_affine_strength=float(meta.get("context_affine_strength", 0.125)),
    )
    return Z7Mamba2PredictiveCodingArchive(
        encoder_state_dict=_deserialize_state_dict(encoder_blob),
        decoder_state_dict=_deserialize_state_dict(decoder_blob),
        predictor_state_dict=_deserialize_state_dict(predictor_blob),
        latent_init=_dequantize_from_int8(latent_q, scale_li, zp_li),
        residuals=_dequantize_from_int8(residuals_q, scale_r, zp_r),
        ego_motion=_dequantize_from_int8(ego_q, scale_e, zp_e),
        meta=meta,
        schema_version=int(version),
        config=config,
    )


def replay_latent_sequence_with_context(
    archive: Z7Mamba2PredictiveCodingArchive,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay parsed Mamba-2 stream into latents plus pre-residual contexts.

    Loads the Mamba2Predictor weights from the parsed archive and
    autoregresses through the selective state-space across the 600-pair
    sequence. Returns ``(latents, contexts)`` where ``contexts`` are
    pre-residual predictor outputs (used by optional latent_affine
    context conditioner at inflate time).
    """
    from tac.optimization.mamba2_predictor import Mamba2Predictor

    predictor = Mamba2Predictor(archive.config.to_mamba2_predictor_config())
    if not archive.config.identity_predictor:
        # Cast predictor weights to fp32 for stable replay
        # Wave N+9 Slot 1 self-containment fix 2026-05-28: strip the canonical
        # "predictor." prefix that `export_state_dict` adds before loading into
        # the bare `Mamba2Predictor` module which expects unprefixed keys
        # (sister of the decoder.* prefix-strip in inflate._build_decoder).
        state_dict_fp32 = {
            (k[len("predictor."):] if k.startswith("predictor.") else k): v.to(torch.float32)
            for k, v in archive.predictor_state_dict.items()
        }
        predictor.load_state_dict(state_dict_fp32, strict=True)
    predictor.eval()
    z = archive.latent_init.to(torch.float32).view(1, archive.config.latent_dim)
    ego = archive.ego_motion.to(torch.float32)
    residuals = archive.residuals.to(torch.float32)
    outs: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    with torch.no_grad():
        predictor.reset_state(1, device=z.device)
        for t in range(archive.config.num_pairs):
            pred = predictor(z, ego[t : t + 1])
            z = pred + residuals[t : t + 1]
            contexts.append(pred.squeeze(0).clone())
            outs.append(z.squeeze(0).clone())
    return torch.stack(outs, dim=0), torch.stack(contexts, dim=0)


def replay_latent_sequence(archive: Z7Mamba2PredictiveCodingArchive) -> torch.Tensor:
    """Replay parsed Mamba-2 stream into the latent sequence (no contexts)."""
    latents, _contexts = replay_latent_sequence_with_context(archive)
    return latents


Z7MCM2_SECTION_ROLES: dict[str, str] = {
    "z7mcm2_header": "control_or_metadata",
    "encoder_blob": "context_conditioner_weight_stream_or_training_provenance",
    "decoder_blob": "decoder_weight_stream",
    "predictor_blob": "mamba2_selective_state_space_predictor_weight_stream",
    "latent_init_blob": "latent_stream",
    "residuals_blob": "latent_stream",
    "ego_motion_blob": "sidecar_or_correction_stream",
    "meta_blob": "control_or_metadata",
}
