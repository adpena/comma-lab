# SPDX-License-Identifier: MIT
"""BRV2 consumed-sideinfo contract for ``balle_renderer``.

BRV1 stores main latents directly and only closure-checks hyper-latents.
BRV2 changes the runtime authority: inflate reconstructs main latents from
hyper-latents plus a residual stream before rendering. The residual is not a
valid direct-render latent stream without the sideinfo-conditioned predictor.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from tac.substrates._shared.inflate_runtime import (
    select_inflate_device,
    write_rgb_pair_to_raw,
)

from .architecture import BalleRendererConfig, BalleRendererSubstrate
from .archive import (
    _dequantize_from_int16,
    _deserialize_state_dict,
    _quantize_to_int16,
    _serialize_state_dict,
)

BRV2_MAGIC: bytes = b"BRV2"
BRV2_SCHEMA_VERSION: int = 2
BRV2_CONSUMED_SIDEINFO_CONTRACT: str = (
    "brv2_hyper_latents_predict_main_latent_residual"
)
BRV2_SIDEINFO_GAIN_META_KEY: str = "brv2_sideinfo_prediction_gain"

# MAGIC + VERSION + LATENT_DIM + HYPER_DIM + NUM_PAIRS
# + ENC_LEN + DEC_LEN + HP_LEN + HYPER_LEN + RESIDUAL_LEN + META_LEN
BRV2_HEADER_FMT: str = "<4sBHHHIIIIII"
BRV2_HEADER_SIZE: int = struct.calcsize(BRV2_HEADER_FMT)
assert BRV2_HEADER_SIZE == 35, (
    f"BRV2 header size invariant; got {BRV2_HEADER_SIZE}"
)


@dataclass(frozen=True)
class BalleRendererBRV2Archive:
    """Parsed BRV2 archive without direct main-latent authority."""

    encoder_state_dict: dict[str, torch.Tensor]
    decoder_state_dict: dict[str, torch.Tensor]
    hyperprior_state_dict: dict[str, torch.Tensor]
    hyper_latents: torch.Tensor
    latent_residuals: torch.Tensor
    meta: dict[str, object]
    schema_version: int


def is_brv2_archive(blob: bytes) -> bool:
    """Return True when ``blob`` starts with the BRV2 magic."""
    return len(blob) >= 4 and blob[:4] == BRV2_MAGIC


def _require_brv2_contract(meta: dict[str, object]) -> float:
    contract = meta.get("sideinfo_consumption_contract")
    if contract != BRV2_CONSUMED_SIDEINFO_CONTRACT:
        raise RuntimeError(
            "BRV2 archive missing consumed-sideinfo contract: "
            f"expected {BRV2_CONSUMED_SIDEINFO_CONTRACT!r}, got {contract!r}"
        )
    if BRV2_SIDEINFO_GAIN_META_KEY not in meta:
        raise RuntimeError(
            f"BRV2 archive missing {BRV2_SIDEINFO_GAIN_META_KEY!r} meta field"
        )
    gain = float(meta[BRV2_SIDEINFO_GAIN_META_KEY])
    if not math.isfinite(gain) or gain == 0.0:
        raise RuntimeError(
            f"BRV2 sideinfo prediction gain must be finite and non-zero; got {gain!r}"
        )
    return gain


def _sideinfo_prediction(
    model: BalleRendererSubstrate,
    hyper_latents: torch.Tensor,
    meta: dict[str, object],
) -> torch.Tensor:
    gain = _require_brv2_contract(meta)
    prediction = model.hyper_synthesis(hyper_latents)
    if prediction.shape[0] != hyper_latents.shape[0]:
        raise RuntimeError(
            "BRV2 sideinfo predictor changed batch count: "
            f"{tuple(hyper_latents.shape)} -> {tuple(prediction.shape)}"
        )
    return prediction * gain


def encode_brv2_latent_residuals(
    model: BalleRendererSubstrate,
    latents: torch.Tensor,
    hyper_latents: torch.Tensor,
    meta: dict[str, object],
) -> torch.Tensor:
    """Return residuals whose decode requires BRV2 hyper sideinfo."""
    if latents.dim() != 2 or hyper_latents.dim() != 2:
        raise ValueError("BRV2 latents and hyper_latents must both be 2-D")
    with torch.no_grad():
        prediction = _sideinfo_prediction(
            model,
            hyper_latents.to(device=latents.device, dtype=latents.dtype),
            meta,
        )
    if prediction.shape != latents.shape:
        raise ValueError(
            f"BRV2 prediction shape {tuple(prediction.shape)} does not match "
            f"latents {tuple(latents.shape)}"
        )
    return latents.detach() - prediction.detach()


def decode_brv2_consumed_latents(
    model: BalleRendererSubstrate,
    archive: BalleRendererBRV2Archive,
) -> torch.Tensor:
    """Reconstruct main latents through the consumed sideinfo path."""
    if archive.schema_version != BRV2_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported BRV2 schema version {archive.schema_version}")
    param = next(model.parameters())
    dtype = model.latents.dtype
    hyper_latents = archive.hyper_latents.to(device=param.device, dtype=dtype)
    residuals = archive.latent_residuals.to(device=param.device, dtype=dtype)
    with torch.no_grad():
        prediction = _sideinfo_prediction(model, hyper_latents, archive.meta)
        if prediction.shape != residuals.shape:
            raise RuntimeError(
                f"BRV2 prediction shape {tuple(prediction.shape)} does not match "
                f"residuals {tuple(residuals.shape)}"
            )
        return prediction + residuals


def _cfg_from_brv2_meta(archive: BalleRendererBRV2Archive) -> BalleRendererConfig:
    meta = archive.meta
    return BalleRendererConfig(
        latent_dim=int(archive.latent_residuals.shape[1]),
        hyper_latent_dim=int(archive.hyper_latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        hyper_mlp_channels=tuple(int(c) for c in meta["hyper_mlp_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        gdn_eps=float(meta.get("gdn_eps", 1e-12)),
        quantize_noise_std=float(meta.get("quantize_noise_std", 0.0)),
        num_pairs=int(archive.latent_residuals.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
    )


def _load_brv2_archive_state(
    model: BalleRendererSubstrate,
    archive: BalleRendererBRV2Archive,
) -> None:
    merged: dict[str, torch.Tensor] = {}
    merged.update(
        {"hyper_analysis." + k: v for k, v in archive.encoder_state_dict.items()}
    )
    merged.update(archive.decoder_state_dict)
    merged.update(archive.hyperprior_state_dict)
    incompat = model.load_state_dict(merged, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing - {"latents"} or unexpected:
        raise RuntimeError(
            "balle_renderer BRV2 archive state_dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )


def inflate_brv2_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate BRV2 by decoding main latents through hyper sideinfo."""
    archive = parse_brv2_archive(archive_bytes)
    render_device = select_inflate_device(device)
    model = (
        BalleRendererSubstrate(_cfg_from_brv2_meta(archive))
        .to(render_device)
        .eval()
    )
    _load_brv2_archive_state(model, archive)
    with torch.no_grad():
        latents = decode_brv2_consumed_latents(model, archive)
        if latents.shape != model.latents.shape:
            raise RuntimeError(
                f"BRV2 decoded latents shape {tuple(latents.shape)} does not match "
                f"model latents {tuple(model.latents.shape)}"
            )
        model.latents.copy_(latents.to(dtype=model.latents.dtype))

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(model.cfg.num_pairs):
            idx_tensor = torch.tensor(
                [pair_idx],
                device=render_device,
                dtype=torch.long,
            )
            rgb_0, rgb_1, _rate = model(idx_tensor)
            frames_written += write_rgb_pair_to_raw(
                fh,
                rgb_0,
                rgb_1,
                input_range="unit",
            )
    return frames_written


def pack_brv2_archive(
    encoder_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    hyperprior_state_dict: dict[str, torch.Tensor],
    hyper_latents: torch.Tensor,
    latent_residuals: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = BRV2_SCHEMA_VERSION,
) -> bytes:
    """Serialize a BRV2 packet with consumed hyper-latent sideinfo."""
    if schema_version != BRV2_SCHEMA_VERSION:
        raise ValueError(f"unsupported BRV2 schema version: {schema_version}")
    if meta.get("sideinfo_consumption_contract") != BRV2_CONSUMED_SIDEINFO_CONTRACT:
        raise ValueError(
            "BRV2 meta must declare sideinfo_consumption_contract="
            f"{BRV2_CONSUMED_SIDEINFO_CONTRACT!r}"
        )
    _require_brv2_contract(dict(meta))
    if hyper_latents.dim() != 2 or latent_residuals.dim() != 2:
        raise ValueError("BRV2 hyper_latents and latent_residuals must both be 2-D")
    if hyper_latents.shape[0] != latent_residuals.shape[0]:
        raise ValueError(
            "BRV2 num_pairs mismatch: "
            f"hyper_latents {hyper_latents.shape[0]} vs residuals {latent_residuals.shape[0]}"
        )

    num_pairs = int(hyper_latents.shape[0])
    hyper_dim = int(hyper_latents.shape[1])
    latent_dim = int(latent_residuals.shape[1])
    for name, val in (
        ("num_pairs", num_pairs),
        ("latent_dim", latent_dim),
        ("hyper_dim", hyper_dim),
    ):
        if val <= 0 or val > 0xFFFF:
            raise ValueError(f"{name}={val} out of u16 range")

    q_hyp, hyp_scale, hyp_zp = _quantize_to_int16(hyper_latents)
    q_res, res_scale, res_zp = _quantize_to_int16(latent_residuals)
    hyper_bytes = q_hyp.contiguous().numpy().tobytes()
    residual_bytes = q_res.contiguous().numpy().tobytes()

    enc_blob = _serialize_state_dict(encoder_state_dict)
    dec_blob = _serialize_state_dict(decoder_state_dict)
    hp_blob = _serialize_state_dict(hyperprior_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_brv2_hyper_quant_scale"] = float(hyp_scale)
    meta_with_quant["_brv2_hyper_quant_zero_point"] = float(hyp_zp)
    meta_with_quant["_brv2_residual_quant_scale"] = float(res_scale)
    meta_with_quant["_brv2_residual_quant_zero_point"] = float(res_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        BRV2_HEADER_FMT,
        BRV2_MAGIC,
        schema_version,
        latent_dim,
        hyper_dim,
        num_pairs,
        len(enc_blob),
        len(dec_blob),
        len(hp_blob),
        len(hyper_bytes),
        len(residual_bytes),
        len(meta_bytes),
    )
    return header + enc_blob + dec_blob + hp_blob + hyper_bytes + residual_bytes + meta_bytes


def _unpack_brv2_header(
    blob: bytes,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    if len(blob) < BRV2_HEADER_SIZE:
        raise ValueError(
            f"BRV2 archive too short ({len(blob)} bytes; need >= {BRV2_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        hyper_dim,
        num_pairs,
        enc_len,
        dec_len,
        hp_len,
        hyper_len,
        residual_len,
        meta_len,
    ) = struct.unpack(BRV2_HEADER_FMT, blob[:BRV2_HEADER_SIZE])
    if magic != BRV2_MAGIC:
        raise ValueError(f"bad BRV2 magic: {magic!r} (expected {BRV2_MAGIC!r})")
    if version != BRV2_SCHEMA_VERSION:
        raise ValueError(f"unsupported BRV2 schema version: {version}")
    expected_hyper_bytes = num_pairs * hyper_dim * 2
    expected_residual_bytes = num_pairs * latent_dim * 2
    if hyper_len != expected_hyper_bytes:
        raise ValueError(
            f"BRV2 hyper_len {hyper_len} != num_pairs*hyper_dim*2 = {expected_hyper_bytes}"
        )
    if residual_len != expected_residual_bytes:
        raise ValueError(
            "BRV2 residual_len "
            f"{residual_len} != num_pairs*latent_dim*2 = {expected_residual_bytes}"
        )
    return (
        int(latent_dim),
        int(hyper_dim),
        int(num_pairs),
        int(enc_len),
        int(dec_len),
        int(hp_len),
        int(hyper_len),
        int(residual_len),
        int(meta_len),
    )


def brv2_section_offsets(blob: bytes) -> dict[str, tuple[int, int]]:
    """Return byte offsets for BRV2 sections after validating the header."""
    (
        _latent_dim,
        _hyper_dim,
        _num_pairs,
        enc_len,
        dec_len,
        hp_len,
        hyper_len,
        residual_len,
        meta_len,
    ) = _unpack_brv2_header(blob)
    end_header = BRV2_HEADER_SIZE
    end_enc = end_header + enc_len
    end_dec = end_enc + dec_len
    end_hp = end_dec + hp_len
    end_hyper = end_hp + hyper_len
    end_residual = end_hyper + residual_len
    end_meta = end_residual + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"BRV2 archive size {len(blob)} != expected {end_meta} from header"
        )
    return {
        "encoder_state_dict": (end_header, end_enc),
        "decoder_state_dict": (end_enc, end_dec),
        "hyperprior_state_dict": (end_dec, end_hp),
        "hyper_latents": (end_hp, end_hyper),
        "latent_residuals": (end_hyper, end_residual),
        "meta": (end_residual, end_meta),
    }


def parse_brv2_archive(blob: bytes) -> BalleRendererBRV2Archive:
    """Parse BRV2 bytes into sideinfo + residual components."""
    (
        latent_dim,
        hyper_dim,
        num_pairs,
        _enc_len,
        _dec_len,
        _hp_len,
        _hyper_len,
        _residual_len,
        _meta_len,
    ) = _unpack_brv2_header(blob)
    offsets = brv2_section_offsets(blob)

    enc_blob = blob[slice(*offsets["encoder_state_dict"])]
    dec_blob = blob[slice(*offsets["decoder_state_dict"])]
    hp_blob = blob[slice(*offsets["hyperprior_state_dict"])]
    hyper_blob = blob[slice(*offsets["hyper_latents"])]
    residual_blob = blob[slice(*offsets["latent_residuals"])]
    meta_blob = blob[slice(*offsets["meta"])]

    enc_sd = _deserialize_state_dict(enc_blob)
    dec_sd = _deserialize_state_dict(dec_blob)
    hp_sd = _deserialize_state_dict(hp_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local import; keep module import-time light

    q_hyp = torch.from_numpy(
        np.frombuffer(hyper_blob, dtype=np.int16).copy()
    ).view(num_pairs, hyper_dim)
    q_res = torch.from_numpy(
        np.frombuffer(residual_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    hyper_latents = _dequantize_from_int16(
        q_hyp,
        float(meta["_brv2_hyper_quant_scale"]),
        float(meta["_brv2_hyper_quant_zero_point"]),
    )
    latent_residuals = _dequantize_from_int16(
        q_res,
        float(meta["_brv2_residual_quant_scale"]),
        float(meta["_brv2_residual_quant_zero_point"]),
    )

    return BalleRendererBRV2Archive(
        encoder_state_dict=enc_sd,
        decoder_state_dict=dec_sd,
        hyperprior_state_dict=hp_sd,
        hyper_latents=hyper_latents,
        latent_residuals=latent_residuals,
        meta=meta,
        schema_version=BRV2_SCHEMA_VERSION,
    )


__all__ = [
    "BRV2_CONSUMED_SIDEINFO_CONTRACT",
    "BRV2_HEADER_SIZE",
    "BRV2_MAGIC",
    "BRV2_SCHEMA_VERSION",
    "BRV2_SIDEINFO_GAIN_META_KEY",
    "BalleRendererBRV2Archive",
    "brv2_section_offsets",
    "decode_brv2_consumed_latents",
    "encode_brv2_latent_residuals",
    "inflate_brv2_one_video",
    "is_brv2_archive",
    "pack_brv2_archive",
    "parse_brv2_archive",
]
