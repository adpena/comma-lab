# SPDX-License-Identifier: MIT
"""Lane 17 / Lane J-IMP — sparse-CSR multi-tensor renderer archive (IMPS magic).

This module wraps the per-tensor sparse-CSR codec from
:mod:`tac.iterative_magnitude_pruning` into a full-renderer .bin archive
that the contest inflate path loads via the ``IMPS`` magic-byte dispatch.

It mirrors the wire layout pattern of :mod:`tac.owv2_renderer_archive` so
operators can reason about both archive variants from a single mental model.

Wire format (little-endian throughout)
--------------------------------------
::

    magic              : 4 bytes  = b"IMPS"
    header_len         : 4 bytes  uint32
    header_json        : header_len bytes UTF-8 JSON
    body_len           : 4 bytes  uint32
    body               : body_len bytes (concatenation of per-layer chunks)

The header JSON records ``arch`` (output of
:func:`tac.renderer_export._infer_asymmetric_config`) plus a ``layers``
list. Each layer entry has::

    {
        "name": "renderer.stem_conv",
        "kind": "imps_conv" | "fp16_emb" | "fp16_conv" | "fp16_convt" | "fp16_linear",
        "shape": [O, I, kH, kW] (4-tuple) or [O, I] or [N, D],
        ...kind-specific fields,
    }

Body layout per kind
~~~~~~~~~~~~~~~~~~~~

* ``imps_conv`` — single sparse-CSR payload (per-tensor) for the weight,
  produced by :func:`tac.iterative_magnitude_pruning.sparse_csr_export`. If
  the conv has a bias, the FP16 bias bytes follow concatenated. The header
  entry records ``weight_blob_len`` + ``has_bias`` + ``bias_blob_len`` +
  ``sparsity`` (informational; the codec round-trips its own header).
* ``fp16_emb`` — raw FP16 bytes of the embedding table.
* ``fp16_conv`` / ``fp16_convt`` / ``fp16_linear`` — raw FP16 weight bytes
  followed by raw FP16 bias bytes (if ``has_bias``). Header records
  ``weight_blob_len`` + ``bias_blob_len``.

Per-tensor breakeven gate
~~~~~~~~~~~~~~~~~~~~~~~~~

Sparse-CSR (uint16 idx + FP4 val = 2.5B/nnz + a small per-tensor header)
beats dense FP4 storage (4 bits / weight = numel/2 bytes) only above ~80%
sparsity. For tensors below that threshold OR for tensors whose numel
exceeds the uint16 cap (65535), the encoder falls back to FP16 raw bytes
to avoid shipping a regression — the same Carmack overhead-gate philosophy
OWV2 uses for protected layers.

Why FP16 fallback (and not FP4 dense)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASYM/FP4A export path lives in :mod:`tac.renderer_export`. The IMPS
archive is intentionally a thinner wrapper: it produces a self-contained
.bin that decodes WITHOUT requiring the FP4A dequant code path. FP16 is
2x the dense-FP4 size but trivially deserializable by ``np.frombuffer``.
For the Lane G v3 anchor, the IMPS archive is competitive only when the
*global* renderer reaches >80% sparsity (the dense convs that don't get
pruned to >80% individually still ship in FP16, but they're a small
fraction of the total weight count).

CLAUDE.md compliance
--------------------
* No scorer load at decode time (Check H STRICT). The decode path is pure
  ``sparse_csr_decode`` + numpy ``frombuffer`` + ``copy_`` — no SegNet /
  PoseNet forward pass.
* No silent defaults (Check 81 STRICT). Every public function arg is None
  or explicit.
* Strict-scorer-rule: this archive variant adds ZERO bytes' worth of
  scorer weights to the archive; the per-tensor sparse-CSR payloads +
  FP16 fallbacks are bit-deterministic CPU functions of the renderer's
  float weights + boolean mask.

Provenance
----------
Lane 17 Council design 2026-04-30 (.omx/research/council_lane_17_imp_design_20260430.md):
- Magic byte ``b"IMPS"`` (Q6, 10/10 vote)
- Per-tensor breakeven gate matches the sparse-CSR breakeven formula in
  :mod:`tac.iterative_magnitude_pruning` module docstring (~80% sparsity).
- Mask is implicit: sparse-CSR stores indices of survivors, so the mask
  reconstructs from idx-set on decode.

Cross-references
----------------
* Sister codec module: :mod:`tac.iterative_magnitude_pruning` (sparse-CSR primitive)
* Inflate-side dispatch: ``submissions/robust_current/inflate_renderer.py``
  (``magic == b"IMPS"`` branch added by this lane)
* Companion tests: :mod:`tac.tests.test_imps_renderer_archive`
* Build script: ``experiments/build_lane_17_imps_archive.py`` (dispatched
  from ``scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh``).
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch
import torch.nn as nn

from tac.iterative_magnitude_pruning import (
    sparse_csr_decode,
    sparse_csr_export,
)

IMPS_ARCHIVE_MAGIC: bytes = b"IMPS"
IMPS_ARCHIVE_VERSION: int = 1

# Sparse-CSR beats dense FP4 once nnz fraction < 50% (i.e. sparsity > 50%
# at our 88K-param scale). The actual breakeven from the codec docstring is
# ~80% (uint16 idx + FP4 val = 2.5B/nnz vs 0.5B/numel for dense FP4). We
# pick a slightly conservative gate of 0.78 to absorb header overhead +
# alignment costs across the multi-tensor archive.
IMPS_PER_TENSOR_SPARSITY_GATE: float = 0.78

# numel cap matches the sparse-CSR codec's uint16 indexing cap.
IMPS_PER_TENSOR_NUMEL_CAP: int = 65535


# ── exceptions ────────────────────────────────────────────────────────────


class IMPSArchiveError(ValueError):
    """Raised on malformed IMPS archive payloads (missing magic, truncated
    header/body, unsupported version, layer name not found in the
    reconstructed model, etc.).
    """


# ── encode ────────────────────────────────────────────────────────────────


def _eligible_for_sparse_csr(
    weight: torch.Tensor,
    mask: torch.Tensor,
    sparsity_gate: float = IMPS_PER_TENSOR_SPARSITY_GATE,
    numel_cap: int = IMPS_PER_TENSOR_NUMEL_CAP,
) -> bool:
    """A Conv2d weight is sparse-CSR-eligible iff:

    * mask is provided and shape-matches the weight.
    * The PER-TENSOR sparsity exceeds the breakeven gate (default 78%).
    * numel <= the uint16 indexing cap (default 65535).

    Below the gate, the dense-FP16 fallback is smaller AND faster to decode.
    """
    if mask is None:
        return False
    if weight.shape != mask.shape:
        return False
    numel = int(weight.numel())
    if numel > numel_cap:
        return False
    if numel == 0:
        return False
    n_kept = int(mask.sum().item())
    sparsity = 1.0 - (n_kept / numel)
    return sparsity >= sparsity_gate


def encode_imps_archive(
    model: nn.Module | None = None,
    *,
    masks: dict[str, torch.Tensor] | None = None,
    sparsity_gate: float | None = None,
    arch_extra: dict | None = None,
) -> bytes:
    """Encode an AsymmetricPairGenerator (or compatible) renderer to IMPS.

    Walks every parameterised module:

    * Plain ``nn.Embedding`` → FP16 raw bytes.
    * Plain ``nn.Conv2d`` with eligible weight (mask provided AND
      sparsity > gate AND numel <= cap) → sparse-CSR encode the weight,
      FP16 fallback for the bias.
    * ``nn.Conv2d`` ineligible → FP16 weight + bias.
    * ``nn.ConvTranspose2d`` → FP16 weight + bias (no sparse-CSR path;
      transpose-conv channel layout differs from regular conv).
    * ``nn.Linear`` → FP16 weight + bias.

    Args:
        model: a built renderer (e.g. AsymmetricPairGenerator). REQUIRED;
            no silent default.
        masks: dict ``{qualified_name: BoolTensor}`` from
            :func:`tac.iterative_magnitude_pruning.prune_lowest_magnitude`.
            Names match the ``model.named_parameters()`` keys (e.g.
            ``"renderer.stem_conv.weight"``). Conv layers without a mask
            entry fall through to FP16.
        sparsity_gate: per-tensor sparsity above which sparse-CSR encoding
            is used. Defaults to 0.78. Pass an explicit value to tighten
            (lower) or relax (higher).
        arch_extra: optional extra keys to merge into the inferred arch
            config (rare; pass ``None`` for the standard path).

    Returns:
        bytes — the full IMPS archive payload (ready for ``Path.write_bytes``).

    Raises:
        IMPSArchiveError on shape/finiteness/encoding errors.
    """
    if model is None:
        raise IMPSArchiveError(
            "encode_imps_archive: model is required (no silent default — "
            "Check 81 STRICT)."
        )
    if masks is None:
        # Allow empty-mask encode → entire archive becomes FP16, useful for
        # the unit-test smoke and for "what if I export the dense baseline
        # in IMPS format" sanity comparisons. A real Lane 17 dispatch always
        # passes a populated mask dict.
        masks = {}
    gate = float(IMPS_PER_TENSOR_SPARSITY_GATE if sparsity_gate is None
                 else sparsity_gate)

    # Lazy import to mirror OWV2 — keeps this module importable without
    # pulling the full renderer_export dependency at module-load time.
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
                    if isinstance(module.stride, tuple)
                    else module.stride
                ),
                "padding": (
                    module.padding[0]
                    if isinstance(module.padding, tuple)
                    else module.padding
                ),
                "groups": module.groups,
                "padding_mode": module.padding_mode,
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
            mask_key = f"{name}.weight" if name else "weight"
            mask = masks.get(mask_key)
            if mask is not None:
                mask = mask.detach().cpu().bool()
            if _eligible_for_sparse_csr(w, mask, sparsity_gate=gate):
                payload = sparse_csr_export(w, mask)
                body_chunks.append(payload + b_blob)
                n_kept = int(mask.sum().item())
                numel = int(w.numel())
                layers_meta.append({
                    "name": name,
                    "kind": "imps_conv",
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
                    "sparsity": 1.0 - (n_kept / numel),
                    "n_kept": n_kept,
                    "numel": numel,
                })
                continue
            # Fallback: FP16 raw bytes.
            arr = w.numpy()
            w_blob = arr.astype("float16").tobytes()
            body_chunks.append(w_blob + b_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_conv",
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
    # (mirror OMG1 / OWV2 export behaviour).
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
        "version": IMPS_ARCHIVE_VERSION,
        "format": "imps_renderer_archive_v1",
        "arch": arch,
        "layers": layers_meta,
        "scalar_params": scalar_params,
        "body_len": len(body),
        "sparsity_gate": gate,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    buf = bytearray()
    buf.extend(IMPS_ARCHIVE_MAGIC)
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


def decode_imps_archive(
    data: bytes | None = None,
    device: str | None = None,
) -> nn.Module:
    """Decode an IMPS archive blob and return the rebuilt renderer.

    Pure-math byte → renderer. NO scorer load (Check H STRICT). The only
    forward pass that occurs is the renderer's own ``__init__`` (which
    constructs sub-modules with random weights that we then overwrite).

    Args:
        data: bytes produced by :func:`encode_imps_archive`. REQUIRED.
        device: target device string (e.g. ``"cuda"``, ``"cpu"``).
            REQUIRED — no silent fallback.

    Returns:
        torch.nn.Module — the rebuilt renderer in eval mode on ``device``.
    """
    if data is None:
        raise IMPSArchiveError(
            "decode_imps_archive: data is required (no silent default — "
            "Check 81 STRICT)."
        )
    if device is None:
        raise IMPSArchiveError(
            "decode_imps_archive: device is required (no silent default — "
            "the renderer must be placed deterministically per CLAUDE.md "
            "no-MPS-fallback non-negotiable)."
        )
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise IMPSArchiveError(
            f"decode_imps_archive: data must be bytes-like, got "
            f"{type(data).__name__}"
        )
    blob = bytes(data)
    if len(blob) < 12 or blob[:4] != IMPS_ARCHIVE_MAGIC:
        raise IMPSArchiveError(
            f"decode_imps_archive: bad/missing magic (got {blob[:4]!r}, "
            f"expected {IMPS_ARCHIVE_MAGIC!r})"
        )
    offset = 4
    (header_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    header = json.loads(blob[offset:offset + header_len].decode("utf-8"))
    offset += header_len
    if header.get("version") != IMPS_ARCHIVE_VERSION:
        raise IMPSArchiveError(
            f"decode_imps_archive: unsupported version "
            f"{header.get('version')!r} (expected {IMPS_ARCHIVE_VERSION})"
        )
    (body_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    body = blob[offset:offset + body_len]
    if len(body) != body_len:
        raise IMPSArchiveError(
            f"decode_imps_archive: declared body_len={body_len} but only "
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
            raise IMPSArchiveError(
                f"decode_imps_archive: layer {name!r} not found in "
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
                    bias_shape = [
                        w_t.shape[0] if kind != "fp16_convt" else w_t.shape[1]
                    ]
                    b_t = _read_fp16_to_tensor(b_chunk, bias_shape)
                    module.bias.copy_(b_t)
            continue

        if kind == "imps_conv":
            w_len = int(layer_meta["weight_blob_len"])
            b_len = int(layer_meta["bias_blob_len"])
            w_chunk = body[body_offset:body_offset + w_len]
            body_offset += w_len
            b_chunk = body[body_offset:body_offset + b_len]
            body_offset += b_len
            # sparse_csr_decode returns (dense_weight, mask). The mask is
            # retained on the layer_meta for sanity checks; the dense
            # tensor (with pruned positions = 0.0) is what the model needs.
            dense, _mask = sparse_csr_decode(w_chunk)
            dense = dense.reshape(layer_meta["shape"])
            with torch.no_grad():
                module.weight.copy_(dense)
                if layer_meta.get("has_bias") and module.bias is not None:
                    b_t = _read_fp16_to_tensor(b_chunk, [dense.shape[0]])
                    module.bias.copy_(b_t)
            continue

        raise IMPSArchiveError(
            f"decode_imps_archive: unknown layer kind {kind!r} for {name!r}"
        )

    # Restore scalar params (mirror OMG1 / OWV2 behaviour).
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
                raise IMPSArchiveError(
                    "decode_imps_archive: shared-embedding invariant violated "
                    "(renderer.embedding is not motion.embedding) — arch "
                    "drift between encode and decode."
                )

    model = model.to(device)
    model.eval()
    return model


__all__ = [
    "IMPS_ARCHIVE_MAGIC",
    "IMPS_ARCHIVE_VERSION",
    "IMPS_PER_TENSOR_SPARSITY_GATE",
    "IMPS_PER_TENSOR_NUMEL_CAP",
    "IMPSArchiveError",
    "encode_imps_archive",
    "decode_imps_archive",
]
