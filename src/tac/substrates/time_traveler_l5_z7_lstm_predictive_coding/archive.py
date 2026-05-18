# SPDX-License-Identifier: MIT
"""Z7PCWM1 byte grammar scaffold for the GRU recurrent predictor path.

This is not score authority. It is the byte-closed archive grammar needed before
Z7-GRU can graduate from scaffolded runtime closure to the next real build gate.
The grammar stores encoder/decoder/provenance blobs for the
future Z6-compatible full renderer plus the Z7 distinguishing GRU predictor
bytes, latent init, predictive residuals, ego-motion side information, and
sorted JSON metadata.

The helper ``replay_latent_sequence`` proves the GRU predictor section is
parse-consumed by reconstructing the latent autoregression:

    z_t = GRU(z_{t-1}, ego_motion[t]) + residual[t]

Scorer authority remains blocked until the Wave N+1 build, trained packet, and
paired exact eval land.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass

import torch

from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture import (
    GruRecurrentPredictor,
    Z7GruPredictiveCodingConfig,
)

Z7PCWM1_MAGIC: bytes = b"Z7GR"
Z7PCWM1_SCHEMA_VERSION: int = 1

# MAGIC(4), VERSION(1), LATENT_DIM(2), EGO_DIM(2), NUM_PAIRS(2),
# GRU_HIDDEN_DIM(2), GRU_LAYERS(1), FLAGS(1), 7 x u32 section lengths.
Z7PCWM1_HEADER_FMT: str = "<4sBHHHHBBIIIIIII"
Z7PCWM1_HEADER_SIZE: int = struct.calcsize(Z7PCWM1_HEADER_FMT)
assert Z7PCWM1_HEADER_SIZE == 43

_ZLIB_LEVEL: int = 9
_FLAG_STATEFUL: int = 1 << 0
_FLAG_IDENTITY: int = 1 << 1


@dataclass(frozen=True)
class Z7PredictiveCodingArchive:
    """Parsed Z7PCWM1 archive scaffold."""

    encoder_state_dict: dict[str, torch.Tensor]
    decoder_state_dict: dict[str, torch.Tensor]
    predictor_state_dict: dict[str, torch.Tensor]
    latent_init: torch.Tensor
    residuals: torch.Tensor
    ego_motion: torch.Tensor
    meta: dict[str, object]
    schema_version: int
    config: Z7GruPredictiveCodingConfig


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
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
    import numpy as np

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
        arr = np.frombuffer(raw[pos : pos + nbytes], dtype=np.float16)
        sd[key] = torch.from_numpy(arr.reshape(shape).astype(np.float16, copy=True))
        pos += nbytes
    return sd


def _quantize_to_int8(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    if t.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"tensor must be float16/float32; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return torch.full_like(f, -127, dtype=torch.int8), 1.0, lo
    scale = (hi - lo) / 254.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 254.0)
    return (q_unsigned - 127.0).to(torch.int8), scale, lo


def _dequantize_from_int8(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    return (q.to(torch.float32) + 127.0) * float(scale) + float(zero_point)


def _false_authority_meta(
    base_meta: dict[str, object],
    *,
    config: Z7GruPredictiveCodingConfig,
) -> dict[str, object]:
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
        "paired_exact_eval_json_required_for_z7_disambiguator",
        "wave_n_plus_1_council_required",
    ]
    if not score_aware_used:
        blockers.insert(1, "score_aware_training_absent_prebuild")
    meta["z7_recurrent_predictive_coding_meta"] = {
        "schema": "z7_recurrent_predictive_coding_meta_v1",
        "substrate_id": "time_traveler_l5_z7_lstm_predictive_coding",
        "archive_grammar": "Z7PCWM1",
        "predictor": "gru_recurrent_predictor",
        "loss_mode": "score_aware" if score_aware_used else "proxy",
        "score_aware_scorer_loss_used": score_aware_used,
        "latent_dim": config.latent_dim,
        "ego_motion_dim": config.ego_motion_dim,
        "gru_hidden_dim": config.gru_hidden_dim,
        "gru_num_layers": config.gru_num_layers,
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
    config: Z7GruPredictiveCodingConfig,
    schema_version: int = Z7PCWM1_SCHEMA_VERSION,
) -> bytes:
    """Serialize Z7-GRU scaffold sections into deterministic Z7PCWM1 bytes."""

    if schema_version != Z7PCWM1_SCHEMA_VERSION:
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
    if config.gru_hidden_dim <= 0 or config.gru_hidden_dim > 0xFFFF:
        raise ValueError("config.gru_hidden_dim out of u16 range")
    if config.gru_num_layers <= 0 or config.gru_num_layers > 0xFF:
        raise ValueError("config.gru_num_layers out of u8 range")

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
        Z7PCWM1_HEADER_FMT,
        Z7PCWM1_MAGIC,
        schema_version,
        latent_dim,
        ego_dim,
        num_pairs,
        config.gru_hidden_dim,
        config.gru_num_layers,
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
    if len(blob) < Z7PCWM1_HEADER_SIZE:
        raise ValueError(
            f"z7pcwm1 archive too short: got {len(blob)} bytes, "
            f"need >= {Z7PCWM1_HEADER_SIZE}"
        )
    header = struct.unpack(Z7PCWM1_HEADER_FMT, blob[:Z7PCWM1_HEADER_SIZE])
    magic = header[0]
    version = header[1]
    if magic != Z7PCWM1_MAGIC:
        raise ValueError(f"z7pcwm1 bad magic: {magic!r}")
    if version != Z7PCWM1_SCHEMA_VERSION:
        raise ValueError(f"z7pcwm1 unsupported schema version: {version}")
    return header


def parse_z7pcwm1_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for Z7PCWM1 bytes."""

    (
        _magic,
        _version,
        latent_dim,
        ego_dim,
        num_pairs,
        _gru_hidden_dim,
        _gru_layers,
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
        raise ValueError("z7pcwm1 latent_init_len != latent_dim")
    if residuals_len != int(num_pairs) * int(latent_dim):
        raise ValueError("z7pcwm1 residuals_len != num_pairs*latent_dim")
    if ego_len != int(num_pairs) * int(ego_dim):
        raise ValueError("z7pcwm1 ego_motion_len != num_pairs*ego_motion_dim")

    end_header = Z7PCWM1_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_predictor = end_decoder + int(predictor_len)
    end_latent_init = end_predictor + int(latent_init_len)
    end_residuals = end_latent_init + int(residuals_len)
    end_ego = end_residuals + int(ego_len)
    end_meta = end_ego + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"z7pcwm1 archive size {len(archive_bytes)} != expected {end_meta}"
        )
    return {
        "z7pcwm1_header": (0, Z7PCWM1_HEADER_SIZE),
        "encoder_blob": (end_header, int(encoder_len)),
        "decoder_blob": (end_encoder, int(decoder_len)),
        "predictor_blob": (end_decoder, int(predictor_len)),
        "latent_init_blob": (end_predictor, int(latent_init_len)),
        "residuals_blob": (end_latent_init, int(residuals_len)),
        "ego_motion_blob": (end_residuals, int(ego_len)),
        "meta_blob": (end_ego, int(meta_len)),
    }


def parse_archive(blob: bytes) -> Z7PredictiveCodingArchive:
    """Parse Z7PCWM1 bytes back into tensors, state dicts, and metadata."""

    (
        _magic,
        version,
        latent_dim,
        ego_dim,
        num_pairs,
        gru_hidden_dim,
        gru_layers,
        flags,
        _encoder_len,
        _decoder_len,
        _predictor_len,
        _latent_init_len,
        _residuals_len,
        _ego_len,
        _meta_len,
    ) = _unpack_header(blob)
    sections = parse_z7pcwm1_archive_bytes(blob)
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

    import numpy as np

    latent_q = torch.from_numpy(
        np.frombuffer(latent_init_blob, dtype=np.int8).copy()
    ).view(int(latent_dim))
    residuals_q = torch.from_numpy(
        np.frombuffer(residuals_blob, dtype=np.int8).copy()
    ).view(int(num_pairs), int(latent_dim))
    ego_q = torch.from_numpy(
        np.frombuffer(ego_blob, dtype=np.int8).copy()
    ).view(int(num_pairs), int(ego_dim))

    config = Z7GruPredictiveCodingConfig(
        latent_dim=int(latent_dim),
        ego_motion_dim=int(ego_dim),
        gru_hidden_dim=int(gru_hidden_dim),
        gru_num_layers=int(gru_layers),
        stateful=bool(int(flags) & _FLAG_STATEFUL),
        identity_predictor=bool(int(flags) & _FLAG_IDENTITY),
        num_pairs=int(num_pairs),
    )
    return Z7PredictiveCodingArchive(
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


def _slice_section(
    blob: bytes,
    sections: dict[str, tuple[int, int]],
    name: str,
) -> bytes:
    start, length = sections[name]
    return blob[start : start + length]


def replay_latent_sequence(archive: Z7PredictiveCodingArchive) -> torch.Tensor:
    """Replay the parsed GRU predictor + residual stream into latent sequence."""

    predictor = GruRecurrentPredictor(archive.config)
    if not archive.config.identity_predictor:
        predictor.load_state_dict(archive.predictor_state_dict, strict=True)
    predictor.eval()
    z = archive.latent_init.to(torch.float32).view(1, archive.config.latent_dim)
    ego = archive.ego_motion.to(torch.float32)
    residuals = archive.residuals.to(torch.float32)
    outs: list[torch.Tensor] = []
    with torch.no_grad():
        predictor.reset_state(1, device=z.device, dtype=z.dtype)
        for t in range(archive.config.num_pairs):
            pred = predictor(z, ego[t : t + 1])
            z = pred + residuals[t : t + 1]
            outs.append(z.squeeze(0).clone())
    return torch.stack(outs, dim=0)


Z7PCWM1_SECTION_ROLES: dict[str, str] = {
    "z7pcwm1_header": "control_or_metadata",
    "encoder_blob": "training_provenance_only",
    "decoder_blob": "planned_decoder_weight_stream",
    "predictor_blob": "decoder_weight_stream",
    "latent_init_blob": "latent_stream",
    "residuals_blob": "latent_stream",
    "ego_motion_blob": "sidecar_or_correction_stream",
    "meta_blob": "control_or_metadata",
}


__all__ = [
    "Z7PCWM1_HEADER_FMT",
    "Z7PCWM1_HEADER_SIZE",
    "Z7PCWM1_MAGIC",
    "Z7PCWM1_SCHEMA_VERSION",
    "Z7PCWM1_SECTION_ROLES",
    "Z7PredictiveCodingArchive",
    "pack_archive",
    "parse_archive",
    "parse_z7pcwm1_archive_bytes",
    "replay_latent_sequence",
]
