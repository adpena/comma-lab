# SPDX-License-Identifier: MIT
"""Lane Ω-W-V2 — multi-tensor renderer archive (OWV2 magic).

This module wraps :func:`tac.water_filling_codec_v2.encode_omega_w_v2` /
:func:`tac.water_filling_codec_v2.decode_omega_w_v2` (which encode a SINGLE
4-D conv weight tensor) into a full-renderer .bin archive that the
contest inflate path can load via the ``OWV2`` magic-byte dispatch.

Wire format (little-endian throughout)
--------------------------------------
::

    magic              : 4 bytes  = b"OWV2"
    header_len         : 4 bytes  uint32
    header_json        : header_len bytes UTF-8 JSON
    body_len           : 4 bytes  uint32
    body               : body_len bytes (concatenation of per-layer chunks)

The header JSON records ``arch`` (output of
:func:`tac.renderer_export._infer_asymmetric_config`) plus a ``layers`` list.
Each layer entry has::

    {
        "name": "renderer.stem_conv",
        "kind": "owv2_conv" | "fp16_emb" | "fp16_conv" | "fp16_convt" | "fp16_linear",
        "shape": [O, I, kH, kW] (4-tuple) or [O, I] (2-tuple) or [N, D] (embedding),
        "blob_len": int,
        ...kind-specific fields,
    }

Body layout per kind
~~~~~~~~~~~~~~~~~~~~

* ``owv2_conv`` — single OWV2 payload (per-tensor) for the weight; if the
  conv has a bias, the FP16 bias bytes follow concatenated. The header entry
  records ``weight_blob_len`` + ``has_bias`` + ``bias_blob_len``.
* ``fp16_emb`` — raw FP16 bytes of the embedding table.
* ``fp16_conv`` / ``fp16_convt`` / ``fp16_linear`` — raw FP16 weight bytes
  followed by raw FP16 bias bytes (if ``has_bias``). Header records
  ``weight_blob_len`` + ``bias_blob_len``.

Why not "OWV2 every conv"?
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`encode_omega_w_v2` overhead-gate (``GateRegression``) refuses to
ship a regression on tensors where the OWV2 header dwarfs the payload (e.g.
``renderer.head.weight`` shape ``(3, 36, 1, 1)``). For those tensors we
keep the FP16 raw representation — same policy that OMG1 uses for protected
layers.

CLAUDE.md compliance
--------------------
* No scorer load at decode time (Check H STRICT). The decode path is pure
  ``decode_omega_w_v2`` + numpy ``frombuffer`` + ``copy_`` — no SegNet/PoseNet
  forward pass.
* No silent defaults (Check 81 STRICT). Every public function arg is None
  or explicit.
* Strict-scorer-rule: this archive variant adds ZERO bytes' worth of scorer
  weights to the archive; the per-tensor OWV2 payloads + FP16 fallbacks are
  bit-deterministic CPU functions of the renderer's float weights.

Provenance
----------
Council F SAFE-LOCAL Lane Ω-W-V2 validation (40.98% byte savings empirical
on Lane G v3 renderer.bin) — see ``test_omega_w_v2_real_archive.py``. This
module promotes that validation from "single tensor savings" to "full
archive savings + dispatch enabling for contest-CUDA auth eval".

Cross-references
----------------
* Sister codec module: ``tac.water_filling_codec_v2``
* Magic registry: ``tac.codec_magic_registry`` (entry already present)
* Inflate-side dispatch: ``submissions/robust_current/inflate_renderer.py``
  (``magic == b"OWV2"`` branch added by this lane)
* Companion tests:
  - ``tac.tests.test_owv2_renderer_archive_inflate``
  - ``tac.tests.test_omega_w_v2_real_archive`` (single-tensor empirical anchor)
* Build script: ``experiments/build_lane_g_v3_omega_w_v2_stack.py``
* Remote dispatch: ``scripts/remote_lane_omega_w_v2_stack.sh``
"""
from __future__ import annotations

import json
import struct

import numpy as np
import torch
import torch.nn as nn

from tac.water_filling_codec_v2 import (
    BlockFPIneligible,
    GateRegression,
    decode_omega_w_v2,
    encode_omega_w_v2,
)

OWV2_ARCHIVE_MAGIC: bytes = b"OWV2"
OWV2_ARCHIVE_VERSION: int = 1


# ── exceptions ────────────────────────────────────────────────────────────


class OWV2ArchiveError(ValueError):
    """Raised on malformed OWV2 archive payloads (missing magic, truncated
    header/body, unsupported version, layer name not found in the
    reconstructed model, etc.).
    """


# ── encode ────────────────────────────────────────────────────────────────


def _flat_hessian_for(weight: torch.Tensor) -> torch.Tensor:
    """Synthetic uniform per-output-channel Hessian.

    Conservative — gives Ω-W-V2 the WORST-CASE bit allocation (uniform
    spread). A future calibration loop can replace this with a real
    per-channel Hessian computed via 1-step gradients with eval_roundtrip;
    the on-disk format is unchanged.
    """
    o = int(weight.shape[0])
    return torch.ones(o, dtype=torch.float32)


def _v1_raw_byte_estimate(weight: torch.Tensor) -> int:
    """Mirror :func:`tac.water_filling_codec_v2._v1_raw_qint_byte_estimate`."""
    o, i, kh, kw = weight.shape
    return int(o * i * kh * kw) + int(o * 4) + 32


def _eligible_for_owv2(module: nn.Module) -> bool:
    """A Conv2d weight is OWV2-eligible iff:

    * It is a plain :class:`nn.Conv2d` (not ``nn.Sequential`` wrapper, not
      ``ConvTranspose2d``, not ``Linear``).
    * Its weight tensor has rank 4.
    * Its output-channel count is >= 2 (single-output convs have no
      per-channel spread for the arithmetic coder to exploit).
    """
    if not isinstance(module, nn.Conv2d):
        return False
    w = module.weight
    if w.dim() != 4:
        return False
    if int(w.shape[0]) < 2:
        return False
    return True


def _derive_total_bits(weight: torch.Tensor, ratio: float = 0.7) -> int:
    """Default bit budget derivation: 70% of V1 raw byte estimate × 8 bits.

    This mirrors the budget used in
    ``test_omega_w_v2_real_archive.py`` which empirically lands inside
    Council F's [20%, 60%] band on the Lane G v3 renderer. The factor 0.7
    is a single-knob lever — lowering it (e.g. 0.6) tightens the budget +
    saves more bytes at the cost of distortion; raising it (e.g. 0.8)
    relaxes the budget. Documented here so the build script can override.
    """
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(
            f"_derive_total_bits: ratio={ratio} must be in (0, 1) — "
            f"reasonable values are 0.5..0.85."
        )
    bytes_v1 = _v1_raw_byte_estimate(weight)
    return int(bytes_v1 * ratio * 8)


def encode_owv2_archive(
    model: nn.Module | None = None,
    *,
    bit_budget_ratio: float | None = None,
    arch_extra: dict | None = None,
) -> bytes:
    """Encode an AsymmetricPairGenerator (or compatible) renderer to OWV2.

    Walks every parameterised module:

    * Plain ``nn.Embedding`` → FP16 raw bytes.
    * Plain ``nn.Conv2d`` with eligible weight (rank-4, O>=2) → tries to
      OWV2-encode the weight, falls back to FP16 if the OWV2 overhead-gate
      fires (``GateRegression``). Bias is always FP16.
    * ``nn.ConvTranspose2d`` → FP16 weight + bias (no OWV2 path; the
      water-fill codec's per-channel exponent assumes the conv layout
      ``[O, I, kH, kW]``, not ``[I, O, kH, kW]``).
    * ``nn.Linear`` → FP16 weight + bias.

    Args:
        model: a built renderer (e.g. AsymmetricPairGenerator). REQUIRED;
            no silent default.
        bit_budget_ratio: per-tensor OWV2 bit budget as a fraction of the
            V1 raw byte estimate. Defaults to 0.7 (matches Council F
            empirical anchor). Pass an explicit value if you want tighter
            (lower) or more relaxed (higher) compression.
        arch_extra: optional extra keys to merge into the inferred arch
            config (rare; pass ``None`` for the standard path).

    Returns:
        bytes — the full OWV2 archive payload (ready for ``Path.write_bytes``).

    Raises:
        OWV2ArchiveError on shape/finiteness/encoding errors.
    """
    if model is None:
        raise OWV2ArchiveError(
            "encode_owv2_archive: model is None — pass an AsymmetricPairGenerator "
            "instance explicitly. (CLAUDE.md silent-default audit class.)"
        )
    ratio = 0.7 if bit_budget_ratio is None else float(bit_budget_ratio)

    # Lazy import to avoid pulling renderer_export at module-load time
    # (we want this module importable on machines that don't have the full
    # tac.renderer_export dependency tree resolved).
    from tac.renderer_export import _infer_asymmetric_config

    model.eval()
    arch = _infer_asymmetric_config(model)
    if arch_extra:
        arch.update(arch_extra)

    layers_meta: list[dict] = []
    body_chunks: list[bytes] = []
    seen_emb_ids: set[int] = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if id(module) in seen_emb_ids:
                continue
            seen_emb_ids.add(id(module))
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            body_chunks.append(blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_emb",
                "shape": list(module.weight.shape),
                "blob_len": len(blob),
            })
            continue

        if isinstance(module, nn.ConvTranspose2d):
            arr = module.weight.detach().cpu().float().numpy()
            w_blob = arr.astype("float16").tobytes()
            b_blob = b""
            if module.bias is not None:
                b_blob = (
                    module.bias.detach().cpu().float().numpy()
                    .astype("float16").tobytes()
                )
            body_chunks.append(w_blob + b_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_convt",
                "shape": list(module.weight.shape),
                "stride": (
                    module.stride[0]
                    if isinstance(module.stride, tuple) else module.stride
                ),
                "padding": (
                    module.padding[0]
                    if isinstance(module.padding, tuple) else module.padding
                ),
                "has_bias": module.bias is not None,
                "weight_blob_len": len(w_blob),
                "bias_blob_len": len(b_blob),
            })
            continue

        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().cpu().float()
            b_blob = b""
            if module.bias is not None:
                b_blob = (
                    module.bias.detach().cpu().float().numpy()
                    .astype("float16").tobytes()
                )

            if _eligible_for_owv2(module):
                hess = _flat_hessian_for(w)
                total_bits = _derive_total_bits(w, ratio=ratio)
                try:
                    payload = encode_omega_w_v2(
                        weights_block_fp=w,
                        hessian=hess,
                        total_bits=total_bits,
                    )
                    body_chunks.append(payload + b_blob)
                    layers_meta.append({
                        "name": name,
                        "kind": "owv2_conv",
                        "shape": list(w.shape),
                        "stride": (
                            module.stride[0]
                            if isinstance(module.stride, tuple)
                            else module.stride
                        ),
                        "padding": (
                            module.padding[0]
                            if isinstance(module.padding, tuple)
                            else module.padding
                        ),
                        "dilation": (
                            module.dilation[0]
                            if isinstance(module.dilation, tuple)
                            else module.dilation
                        ),
                        "groups": module.groups,
                        "padding_mode": module.padding_mode,
                        "has_bias": module.bias is not None,
                        "weight_blob_len": len(payload),
                        "bias_blob_len": len(b_blob),
                        "v1_raw_estimate": _v1_raw_byte_estimate(w),
                        "bit_budget_ratio": ratio,
                    })
                    continue
                except (BlockFPIneligible, GateRegression):
                    # Honest fall-through to FP16 — Carmack overhead-gate
                    # rule prevents shipping an OWV2-encoded regression.
                    pass

            # Fallback: FP16 raw bytes for ineligible / overhead-gated layers.
            arr = w.numpy()
            w_blob = arr.astype("float16").tobytes()
            body_chunks.append(w_blob + b_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_conv",
                "shape": list(w.shape),
                "stride": (
                    module.stride[0]
                    if isinstance(module.stride, tuple) else module.stride
                ),
                "padding": (
                    module.padding[0]
                    if isinstance(module.padding, tuple) else module.padding
                ),
                "dilation": (
                    module.dilation[0]
                    if isinstance(module.dilation, tuple) else module.dilation
                ),
                "groups": module.groups,
                "padding_mode": module.padding_mode,
                "has_bias": module.bias is not None,
                "weight_blob_len": len(w_blob),
                "bias_blob_len": len(b_blob),
            })
            continue

        if isinstance(module, nn.Linear):
            arr = module.weight.detach().cpu().float().numpy()
            w_blob = arr.astype("float16").tobytes()
            b_blob = b""
            if module.bias is not None:
                b_blob = (
                    module.bias.detach().cpu().float().numpy()
                    .astype("float16").tobytes()
                )
            body_chunks.append(w_blob + b_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_linear",
                "shape": list(module.weight.shape),
                "has_bias": module.bias is not None,
                "weight_blob_len": len(w_blob),
                "bias_blob_len": len(b_blob),
            })
            continue

    # Capture scalar params not covered by the Conv/Linear/Embedding walk
    # (mirror OMG1 export behaviour).
    captured: set[int] = set()
    for _n, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
            for p in mod.parameters(recurse=False):
                captured.add(id(p))
    scalar_params: dict[str, float] = {}
    for pname, param in model.named_parameters():
        if id(param) in captured:
            continue
        if param.numel() == 1:
            scalar_params[pname] = float(param.item())

    body = b"".join(body_chunks)
    header = {
        "version": OWV2_ARCHIVE_VERSION,
        "format": "owv2_renderer_archive_v1",
        "arch": arch,
        "layers": layers_meta,
        "scalar_params": scalar_params,
        "body_len": len(body),
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    buf = bytearray()
    buf.extend(OWV2_ARCHIVE_MAGIC)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    buf.extend(struct.pack("<I", len(body)))
    buf.extend(body)
    return bytes(buf)


# ── decode ────────────────────────────────────────────────────────────────


def _read_fp16_to_tensor(blob: bytes, shape: list[int]) -> torch.Tensor:
    """Restore a tensor from raw FP16 bytes."""
    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32).copy()
    return torch.from_numpy(arr).reshape(shape)


def decode_owv2_archive(
    data: bytes | None = None,
    device: str | None = None,
) -> nn.Module:
    """Decode an OWV2 archive blob and return the rebuilt renderer.

    Pure-math byte → renderer. NO scorer load (Check H STRICT). The only
    forward pass that occurs is the renderer's own ``__init__`` (which
    constructs sub-modules with random weights that we then overwrite).

    Args:
        data: bytes produced by :func:`encode_owv2_archive`. REQUIRED.
        device: target device string (e.g. ``"cuda"``, ``"cpu"``).
            REQUIRED — no silent fallback. The renderer is moved to this
            device + put in eval mode before return.

    Returns:
        torch.nn.Module — the rebuilt renderer in eval mode on ``device``.
    """
    if data is None:
        raise OWV2ArchiveError(
            "decode_owv2_archive: data is required (no silent default — "
            "Check 81 STRICT)."
        )
    if device is None:
        raise OWV2ArchiveError(
            "decode_owv2_archive: device is required (no silent default — "
            "the renderer must be placed deterministically per CLAUDE.md "
            "no-MPS-fallback non-negotiable)."
        )
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise OWV2ArchiveError(
            f"decode_owv2_archive: data must be bytes-like, got "
            f"{type(data).__name__}"
        )
    blob = bytes(data)
    if len(blob) < 12 or blob[:4] != OWV2_ARCHIVE_MAGIC:
        raise OWV2ArchiveError(
            f"decode_owv2_archive: bad/missing magic (got {blob[:4]!r}, "
            f"expected {OWV2_ARCHIVE_MAGIC!r})"
        )
    offset = 4
    (header_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    header = json.loads(blob[offset:offset + header_len].decode("utf-8"))
    offset += header_len
    if header.get("version") != OWV2_ARCHIVE_VERSION:
        raise OWV2ArchiveError(
            f"decode_owv2_archive: unsupported version "
            f"{header.get('version')!r} (expected {OWV2_ARCHIVE_VERSION})"
        )
    (body_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    body = blob[offset:offset + body_len]
    if len(body) != body_len:
        raise OWV2ArchiveError(
            f"decode_owv2_archive: declared body_len={body_len} but only "
            f"{len(body)} bytes remain in the blob."
        )

    # Build the empty renderer from the arch header. Lazy import keeps this
    # module importable in environments without the full tac dependency tree.
    from tac.renderer import AsymmetricPairGenerator, build_renderer

    arch = header["arch"]
    pair_mode = arch.get("pair_mode", "asymmetric")
    if pair_mode == "asymmetric":
        model = AsymmetricPairGenerator(
            num_classes=arch.get("num_classes", 5),
            embed_dim=arch.get("embed_dim", 6),
            base_ch=arch.get("base_ch", 36),
            mid_ch=arch.get("mid_ch", 60),
            motion_hidden=arch.get("motion_hidden", 32),
            depth=arch.get("depth", 1),
            max_flow_px=arch.get("max_flow_px", 20.0),
            max_residual=arch.get("max_residual", 20.0),
            flow_only=arch.get("flow_only", False),
            pose_dim=arch.get("pose_dim", 0),
            use_dsconv=arch.get("use_dsconv", False),
            use_ghost=arch.get("use_ghost", False),
            use_zoom_flow=bool(arch.get("use_zoom_flow") or False),
            padding_mode=arch.get("padding_mode", "zeros"),
            use_dilation=arch.get("use_dilation", False),
        )
    else:
        model = build_renderer(
            num_classes=arch.get("num_classes", 5),
            embed_dim=arch.get("embed_dim", 6),
            base_ch=arch.get("base_ch", 36),
            mid_ch=arch.get("mid_ch", 60),
            motion_hidden=arch.get("motion_hidden", 32),
            depth=arch.get("depth", 1),
            pose_dim=arch.get("pose_dim", 0),
            use_dsconv=arch.get("use_dsconv", False),
            use_ghost=arch.get("use_ghost", False),
            padding_mode=arch.get("padding_mode", "zeros"),
            use_dilation=arch.get("use_dilation", False),
            use_zoom_flow=bool(arch.get("use_zoom_flow") or False),
            blend_mode=arch.get("blend_mode", "scalar"),
            noise_mode=arch.get("noise_mode", "deterministic"),
            motion_type=arch.get("motion_type", "learned_cnn"),
        )

    name_to_module: dict[str, nn.Module] = dict(model.named_modules())
    body_offset = 0
    seen_emb_ids: set[int] = set()

    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        kind = layer_meta["kind"]
        module = name_to_module.get(name)
        if module is None:
            raise OWV2ArchiveError(
                f"decode_owv2_archive: layer {name!r} not found in "
                f"reconstructed model — arch drift between encode and decode."
            )

        if kind == "fp16_emb":
            blob_len = int(layer_meta["blob_len"])
            chunk = body[body_offset:body_offset + blob_len]
            body_offset += blob_len
            if id(module) in seen_emb_ids:
                continue
            seen_emb_ids.add(id(module))
            t = _read_fp16_to_tensor(chunk, layer_meta["shape"])
            with torch.no_grad():
                module.weight.copy_(t)
            continue

        if kind in ("fp16_conv", "fp16_convt", "fp16_linear"):
            w_len = int(layer_meta["weight_blob_len"])
            b_len = int(layer_meta["bias_blob_len"])
            w_chunk = body[body_offset:body_offset + w_len]
            body_offset += w_len
            b_chunk = body[body_offset:body_offset + b_len]
            body_offset += b_len
            w_t = _read_fp16_to_tensor(w_chunk, layer_meta["shape"])
            with torch.no_grad():
                module.weight.copy_(w_t)
                if layer_meta.get("has_bias") and module.bias is not None:
                    b_t = _read_fp16_to_tensor(b_chunk, [w_t.shape[0] if kind != "fp16_convt" else w_t.shape[1]])
                    module.bias.copy_(b_t)
            continue

        if kind == "owv2_conv":
            w_len = int(layer_meta["weight_blob_len"])
            b_len = int(layer_meta["bias_blob_len"])
            w_chunk = body[body_offset:body_offset + w_len]
            body_offset += w_len
            b_chunk = body[body_offset:body_offset + b_len]
            body_offset += b_len
            w_t = decode_omega_w_v2(blob=w_chunk).reshape(layer_meta["shape"])
            with torch.no_grad():
                module.weight.copy_(w_t)
                if layer_meta.get("has_bias") and module.bias is not None:
                    b_t = _read_fp16_to_tensor(b_chunk, [w_t.shape[0]])
                    module.bias.copy_(b_t)
            continue

        raise OWV2ArchiveError(
            f"decode_owv2_archive: unknown layer kind {kind!r} for {name!r}"
        )

    # Restore scalar params (mirror OMG1 behaviour).
    scalar_params = header.get("scalar_params", {}) or {}
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(float(pval))

    # Sanity guard for shared embedding (renderer + motion typically share one).
    if hasattr(model, "renderer") and hasattr(model, "motion"):
        r_emb = getattr(model.renderer, "embedding", None)
        m_emb = getattr(model.motion, "embedding", None)
        if r_emb is not None and m_emb is not None:
            if r_emb is not m_emb:
                raise OWV2ArchiveError(
                    "decode_owv2_archive: shared-embedding invariant violated "
                    "(renderer.embedding is not motion.embedding) — arch "
                    "drift between encode and decode."
                )

    model = model.to(device)
    model.eval()
    return model


def is_owv2_archive(blob: bytes) -> bool:
    """Magic-byte sniff: True iff ``blob`` is an OWV2 archive."""
    return (
        isinstance(blob, (bytes, bytearray, memoryview))
        and len(blob) >= 4
        and bytes(blob[:4]) == OWV2_ARCHIVE_MAGIC
    )


__all__ = [
    "OWV2_ARCHIVE_MAGIC",
    "OWV2_ARCHIVE_VERSION",
    "OWV2ArchiveError",
    "encode_owv2_archive",
    "decode_owv2_archive",
    "is_owv2_archive",
]
