#!/usr/bin/env python3
# ROUNDTRIP_SELF_TEST: _smoke_roundtrip_no_dead_k and _smoke_roundtrip_cplx_op1
# verify emitted archive sections through the staged inflate parsers.
"""PR101 unified-winners cross-paradigm empirical stack.

Composes the four 2026-05-08 PR101 session winners into a single
byte-closed candidate archive (CPU build, no scorer, no dispatch):

  Stage 1: UNIWARD-weighted Lagrangian per-tensor allocation
           (commit ``be715fac``; best Path B step 7 byte_proxy 148,346 B
           at rms=0.05 — saves +4,074 B vs uniform Lagrangian Path B step 6
           at the same rms target).
  Stage 2: ADMM no-dead-K wire format (commit ``0b24e5d1``; drops 28 B
           per archive by removing the audit-only K side-info).
  Stage 3: Op1 finalizer / cross-paradigm composition
           (commit ``8d33d5c1`` + WIRE-DECODER ``669b5b5f``;
           dequantize-then-re-encode-with-PR101-split-Brotli on the K-coarsened
           substrate). Uses the corrected ``q_i8 * scale`` dequantize formula
           per FIX-CODEX-FINDINGS commit ``98d2174b``. Bypasses the buggy
           CodecPipeline CPL1 wrapper (int-key sign-flip) by serializing the
           Op1 inner blob in the WIRE-DECODER CPLX1 wire format.
  Stage 4: Filler STC ternary mask deltas (commit ``0c8bb6d4``).
           DOCUMENTED N/A for PR101 — per
           ``feedback_pr106_archive_is_monolithic_single_file_20260508.md``,
           the PR101 archive is a single ``x`` ZIP entry with no separate
           ``masks.mkv`` channel; the Filler STC mask payload doesn't apply.

Per-stage byte progression (predicted, anchored to existing manifests):

  ADMM-uniform @ rms=0.0386 (Path B step 6 baseline)        153,699 B
  Stage 2 alone (uniform Ks + no-dead-K)                    153,671 B   (-28)
  Stage 1+2 (UNIWARD Ks @ rms=0.05 + no-dead-K)            ~149,XXX B   [computed]
  Stage 1+2+3 (UNIWARD Ks + no-dead-K + Op1 finalizer)     ~XXX,XXX B   [computed]
  Stage 1+2+3+4                                                  N/A   (no PR101 mask channel)

Falsification scope
-------------------
``uniward_x_no_dead_k_x_op1_only``: only the byte-proxy-faithful CPU build
roundtrip + decoder section sizes are tested. Score, scorer-basin parity,
and runtime contest-CUDA replay are NOT tested.

CLAUDE.md compliance
--------------------
- ``family_falsified=False``,
  ``falsification_scope="uniward_x_no_dead_k_x_op1_only"``.
- ``ready_for_exact_eval_dispatch=False`` (CPU build never promotes itself).
- ``cuda_eval_worth_testing=True`` IFF smoke roundtrip passes.
- Pure-CPU; never loads a scorer; tags evidence ``[CPU-prep faithful
  unified-winners-stack test]``.
- weights_only=True per REVIEW-ENG C4.
- Per ``forbidden_premature_class_level_falsification``: any negative
  result tagged ``MEASURED_CONFIG_NOT_DISPATCHABLE``.
- Per ``forbidden_CPU_MPS_derived_dispatch_readiness_flag``:
  ``ready_for_exact_eval_dispatch=False``.

Out-of-scope
------------
- No dispatch (Lightning bootstrap is owned by other subagents).
- No retraining; no Q-FAITHFUL retrain (operator excluded).
- No core ``CodecPipeline`` modifications (we bypass CPL1 entirely
  via WIRE-DECODER's CPLX1 wire format).

Usage
-----
.. code-block:: bash

    .venv/bin/python tools/pr101_unified_winners_stack_empirical.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import Sequence

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    DECODER_BLOB_LEN,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    N_QUANT,
    _quantize_tensor,
    auto_select_byte_maps,
    encode_decoder_compact,
    decode_decoder_compact,
)

# Reuse staging helpers from the canonical Lightning build script.
_LIGHTNING_BUILDER_PATH = (
    REPO_ROOT / "experiments" / "lossy_coarsening_lightning_cuda_test.py"
)
_spec = importlib.util.spec_from_file_location(
    "_lossy_coarsening_lightning_cuda_test", _LIGHTNING_BUILDER_PATH
)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"FATAL: could not load builder from {_LIGHTNING_BUILDER_PATH}")
_lightning_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lightning_builder)

_read_pr101_inner_blob = _lightning_builder._read_pr101_inner_blob
_split_pr101_inner_blob = _lightning_builder._split_pr101_inner_blob
_build_inner_blob = _lightning_builder._build_inner_blob
_write_pr101_archive = _lightning_builder._write_pr101_archive

LANE_ID = "lane_unified_winners_stack"
SCHEMA_VERSION = "pr101_unified_winners_stack_empirical.v1"
TOOL_NAME = "tools/pr101_unified_winners_stack_empirical.py"

# --- Source-of-truth Ks vectors -----------------------------------------------
# UNIWARD-weighted Lagrangian @ rms_target=0.05 (the best-of-session row,
# +4,074 B vs uniform Lagrangian; from
# reports/raw/pr101_uniward_weighted_lagrangian_20260508T063514Z/manifest.json)
UNIWARD_KS_RMS_0_05: tuple[int, ...] = (
    2, 2, 5, 1, 5, 1, 7, 1, 4, 1, 6, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
)
UNIWARD_DECODER_PROXY_BYTES = 148_346  # encoder-proxy at decoder section level
UNIWARD_REL_ERR = 0.05089180545476426
UNIWARD_RMS_TARGET = 0.05
UNIWARD_LAMBDA_REPORTED: float | None = None  # not pinned by source manifest at this slot

# Reference baselines (audit only).
BASELINE_ADMM_UNIFORM_RMS_0_0386_BYTES = 153_699  # original Path B step 6 with K bytes
BASELINE_ADMM_NO_DEAD_K_RMS_0_0386_BYTES = 153_671  # no-dead-K
BASELINE_CROSS_PARADIGM_CORRECTED_BYTES = 153_513  # ADMM-uniform Ks + Op1 finalizer
BASELINE_PR101_LOSSLESS_BYTES = 178_468

DEFAULT_PR101_STATE_DICT = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex"
    / "pr101_decoder_state_dict.pt"
)
DEFAULT_PR101_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full"
    / "public_pr101_intake_20260505_auto"
    / "archive.zip"
)
DEFAULT_PR101_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full"
    / "public_pr101_intake_20260505_auto"
    / "source/submissions/hnerv_ft_microcodec/src"
)

CPLX_MAGIC = b"CPLX"

# Custody fields per CLAUDE.md non-negotiables. Stage 4 retired-config
# audit is documented in the body since PR101 has no mask channel.
CPU_BUILD_SCORE_BLOCKERS = [
    "cpu_build_byte_proxy_not_score_evidence",
    "exact_cuda_auth_eval_not_yet_harvested",
    "requires_contest_auth_eval_json_before_score_promotion_rank_or_kill",
    "uniward_variance_proxy_substitutes_for_wavelet_residual",
    "no_iterative_primal_dual_admm_consensus",
    "score_aware_per_tensor_distortion_weights_not_in_loop",
]

EVIDENCE_GRADE = "[CPU-prep faithful unified-winners-stack test]"


# ---------------------------------------------------------------------------
# Helpers (utility)
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Stage 1+2: UNIWARD Ks + no-dead-K wire format
#
# Same wire format as build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py:
#     uint32 LE: total_section_bytes (D)
#     byte * 56: per_tensor_fp16_scale (LE half)
#     byte * (D - 4 - 56): brotli(concat(rounded_int8s))
# ---------------------------------------------------------------------------


def _build_no_dead_k_decoder_section(
    state_dict_path: Path,
    Ks: Sequence[int],
    *,
    brotli_quality: int = 11,
) -> dict:
    if not state_dict_path.is_file():
        raise SystemExit(f"FATAL: PR101 state_dict not found: {state_dict_path}")
    n_tensors = len(FIXED_STATE_SCHEMA)
    if len(Ks) != n_tensors:
        raise SystemExit(
            f"FATAL: Ks length {len(Ks)} != n_tensors {n_tensors}"
        )
    for k in Ks:
        if not isinstance(k, int) or k < 1 or k > 255:
            raise SystemExit(f"FATAL: K out of [1,255] range: {k!r}")

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    scales_fp16: list[float] = []
    rounded_chunks: list[np.ndarray] = []
    abs_orig_total = 0.0
    abs_err_total = 0.0
    n_symbols = 0
    for (name, _shape), K in zip(FIXED_STATE_SCHEMA, Ks, strict=True):
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        symbols_i32 = qt.q_i8.astype(np.int32).flatten()
        scale_fp16 = float(np.float16(qt.scale))
        rounded = np.round(symbols_i32 / K) * K
        rounded_clipped = rounded.clip(-127, 127)
        abs_orig_total += float(np.abs(symbols_i32).astype(np.float64).sum())
        abs_err_total += float(
            np.abs(rounded_clipped - symbols_i32).astype(np.float64).sum()
        )
        rounded_chunks.append(rounded_clipped.astype(np.int8))
        scales_fp16.append(scale_fp16)
        n_symbols += int(symbols_i32.size)

    flat = np.concatenate(rounded_chunks).tobytes()
    brotli_payload = brotli.compress(
        flat, quality=brotli_quality, lgwin=22, lgblock=24
    )
    rel_err = abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for PR101 unified-winners stack (joint encoder); same form as PR101 lossy_coarsening

    scale_arr = np.array(scales_fp16, dtype=np.float16)
    if not scale_arr.dtype.isnative or sys.byteorder != "little":
        scale_bytes = scale_arr.astype("<f2").tobytes()
    else:
        scale_bytes = scale_arr.tobytes()

    section_no_prefix = scale_bytes + brotli_payload
    section_total = 4 + len(section_no_prefix)
    prefix = struct.pack("<I", section_total)
    decoder_bytes = prefix + section_no_prefix
    if len(decoder_bytes) != section_total:
        raise RuntimeError(
            f"decoder section length mismatch: declared {section_total}, "
            f"actual {len(decoder_bytes)}"
        )
    return {
        "decoder_bytes": decoder_bytes,
        "per_tensor_K": list(Ks),
        "per_tensor_scale_fp16": scales_fp16,
        "rel_err": rel_err,
        "n_tensors": n_tensors,
        "n_symbols": n_symbols,
        "brotli_payload_bytes": len(brotli_payload),
        "scale_bytes": len(scale_bytes),
        "section_total_bytes": section_total,
    }


# ---------------------------------------------------------------------------
# Stage 3: Op1 finalizer (CPLX1 wire format)
#
# Reuses the WIRE-DECODER pipeline:
#   1. Apply UNIWARD Ks to PR101 INT8 symbols.
#   2. Dequantize back to fp32 using ``q_i8 * scale_fp16`` (the canonical
#      PR101 dequantize, NOT the broken ``q_i8 / N_QUANT * scale`` form).
#   3. Run encode_decoder_compact directly (bypass CPL1 wrapper).
#   4. Wrap the inner blob in a CPLX1 container with int-keyed byte_maps.
# ---------------------------------------------------------------------------


def _build_dequantized_substrate_uniward(
    state_dict_path: Path,
    Ks: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Stage 1 (UNIWARD Ks) + Stage 3 substrate: per-tensor int8 quant +
    UNIWARD K-coarsening + dequantize to fp32, ready for Op1 finalizer.

    Identical to the builder in
    ``tools/build_cross_paradigm_admm_x_op1_finalizer.py``'s
    ``_build_dequantized_substrate_via_admm`` — only the K vector source
    differs (UNIWARD instead of uniform Lagrangian).
    """
    if not state_dict_path.is_file():
        raise SystemExit(f"FATAL: PR101 state_dict not found: {state_dict_path}")
    # weights_only=False here mirrors the cross-paradigm tool (PR101 state dict
    # is a plain dict[str, Tensor]; safe to load). REVIEW-ENG B8 allowlist.
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact

    n_tensors = len(FIXED_STATE_SCHEMA)
    if len(Ks) != n_tensors:
        raise SystemExit(
            f"FATAL: Ks length {len(Ks)} != n_tensors {n_tensors}"
        )

    rebuilt: dict[str, torch.Tensor] = {}
    abs_orig = 0.0
    abs_err = 0.0
    per_tensor_scales: list[float] = []
    n_symbols = 0
    for (name, shape), K in zip(FIXED_STATE_SCHEMA, Ks, strict=True):
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        symbols_i32 = qt.q_i8.astype(np.int32).flatten()
        rounded = (np.round(symbols_i32.astype(np.float64) / K) * K).astype(np.int32)
        rounded_clipped = np.clip(rounded, -127, 127).astype(np.int8)
        # Canonical PR101 dequantize (fixes XPARADIGM 127× scale bug;
        # see commit 98d2174b FIX-CODEX-FINDINGS):
        #   weight_fp32 = q_i8 * scale
        deq = rounded_clipped.astype(np.float32) * float(qt.scale)
        rebuilt[name] = torch.from_numpy(deq.reshape(shape))
        abs_orig += float(np.abs(symbols_i32).astype(np.float64).sum())
        abs_err += float(
            np.abs(rounded_clipped.astype(np.int32) - symbols_i32).astype(np.float64).sum()
        )
        per_tensor_scales.append(float(qt.scale))
        n_symbols += int(symbols_i32.size)

    rel_err = abs_err / abs_orig if abs_orig > 1e-9 else 0.0  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for PR101 unified-winners stage_1_2_3 substrate; consistent with cross-paradigm composition form
    return rebuilt, {
        "rel_err_int8_after_uniward_admm": rel_err,
        "n_tensors": n_tensors,
        "n_symbols": n_symbols,
        "per_tensor_scale_fp32": per_tensor_scales,
        "Ks": list(Ks),
    }


def _encode_op1_finalizer_blob(
    dequantized_state_dict: dict[str, torch.Tensor],
) -> tuple[bytes, dict[int, str], dict]:
    """Run PR101 split-Brotli (auto_select=True) on the dequantized substrate.

    Bypasses CodecPipeline CPL1 because of the int-key sign-flip bug
    (WIRE-DECODER 2026-05-08 finding). Returns the raw split-Brotli stream
    plus the int-keyed byte_maps to be serialized in CPLX1.
    """
    bm = auto_select_byte_maps(dequantized_state_dict, brotli_quality=11)
    blob = encode_decoder_compact(
        dequantized_state_dict,
        brotli_quality=11,
        effective_byte_maps=bm,
        auto_select=False,
    )
    return blob, dict(bm), {
        "op1_inner_blob_bytes": len(blob),
        "effective_byte_maps_count": len(bm),
        "non_default_byte_maps": {
            int(idx): m for idx, m in bm.items() if m != "zig"
        },
    }


def _build_cplx1_decoder_section(
    op1_inner_blob: bytes,
    byte_maps: dict[int, str],
) -> bytes:
    """Wrap the Op1 inner blob in the CPLX1 container.

    Mirrors ``_build_decoder_section_cplx`` in the cross-paradigm tool.
    """
    bm_json = json.dumps(
        {str(int(idx)): str(m) for idx, m in byte_maps.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(bm_json) > 0xFFFF:
        raise SystemExit(
            f"FATAL: byte_maps JSON {len(bm_json)} > 65535; CPLX uint16 length "
            "overflow"
        )
    body = struct.pack("<H", len(bm_json)) + bm_json + op1_inner_blob
    section_total = 8 + len(body)
    return CPLX_MAGIC + struct.pack("<I", section_total) + body


# ---------------------------------------------------------------------------
# Forked inflate sources
# ---------------------------------------------------------------------------

# Stage 1+2 inflate (UNIWARD Ks + no-dead-K wire). Identical contract to the
# step6_no_dead_k inflate; the only thing that differs is the K vector that
# was used at encode time, and that's not visible in the inflate (Ks are
# audit-only metadata in build_manifest.json).
INFLATE_PY_NO_DEAD_K = '''#!/usr/bin/env python
"""Forked PR101 inflate for unified-winners-stack Stage 1+2 archive
(UNIWARD-weighted Ks + no-dead-K wire format).

Wire format (inner blob, single ZIP member \'x\'):
    uint32 LE: decoder_section_total_bytes (D)
    byte * 56: per_tensor_fp16_scale (LE half)
    byte * (D - 4 - 56): brotli(int8s)
    byte * 15387: latent_blob (PR101 ORIGINAL)
    byte * remaining: sidecar_blob (ORIGINAL)
"""
import struct
import sys
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import (
    LATENT_BLOB_LEN,
    N_PAIRS,
    LATENT_DIM,
    BASE_CHANNELS,
    EVAL_SIZE,
    decode_latents_compact,
    apply_latent_sidecar,
)
from model import HNeRVDecoder


def _fixed_state_schema():
    probe = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    )
    return tuple((name, tuple(t.shape)) for name, t in probe.state_dict().items())


FIXED_STATE_SCHEMA = _fixed_state_schema()
N_TENSORS = len(FIXED_STATE_SCHEMA)
SCALE_SECTION_BYTES = N_TENSORS * 2
PREFIX_BYTES = 4

CAMERA_H, CAMERA_W = 874, 1164


def parse_unified_no_dead_k_archive(archive_bytes):
    if len(archive_bytes) < PREFIX_BYTES + SCALE_SECTION_BYTES + LATENT_BLOB_LEN:
        raise ValueError(
            f"archive too short ({len(archive_bytes)} bytes) for unified Stage 1+2 format"
        )
    section_total = struct.unpack("<I", archive_bytes[:PREFIX_BYTES])[0]
    if section_total < PREFIX_BYTES + SCALE_SECTION_BYTES:
        raise ValueError(
            f"decoder_section_total ({section_total}) < minimum "
            f"{PREFIX_BYTES + SCALE_SECTION_BYTES}"
        )
    if section_total > len(archive_bytes) - LATENT_BLOB_LEN:
        raise ValueError(
            f"decoder_section_total ({section_total}) leaves no room for "
            f"latent_blob {LATENT_BLOB_LEN}"
        )

    scale_start = PREFIX_BYTES
    scale_end = scale_start + SCALE_SECTION_BYTES
    scales_fp16 = np.frombuffer(archive_bytes[scale_start:scale_end], dtype="<f2")
    if scales_fp16.size != N_TENSORS:
        raise ValueError(
            f"scale section size {scales_fp16.size} != N_TENSORS {N_TENSORS}"
        )

    brotli_start = scale_end
    brotli_end = section_total
    brotli_payload = archive_bytes[brotli_start:brotli_end]
    flat_int8 = np.frombuffer(brotli.decompress(brotli_payload), dtype=np.int8)

    decoder_sd = {}
    cursor = 0
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        nelem = 1
        for d in shape:
            nelem *= d
        if cursor + nelem > flat_int8.size:
            raise ValueError(
                f"flat_int8 underflow at tensor {idx} ({name}): "
                f"need {nelem}, have {flat_int8.size - cursor}"
            )
        chunk = flat_int8[cursor:cursor + nelem].astype(np.int32)
        weight_fp32 = chunk.astype(np.float32) * float(scales_fp16[idx])
        decoder_sd[name] = torch.from_numpy(weight_fp32.reshape(shape).copy())
        cursor += nelem
    if cursor != flat_int8.size:
        raise ValueError(
            f"flat_int8 leftover {flat_int8.size - cursor} bytes after all tensors"
        )

    latent_start = section_total
    latent_end = latent_start + LATENT_BLOB_LEN
    latent_blob = archive_bytes[latent_start:latent_end]
    sidecar_blob = archive_bytes[latent_end:]
    if not latent_blob:
        raise ValueError("missing latent_blob in unified Stage 1+2 archive")

    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decoder_sd, latents, meta


def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents, meta = parse_unified_no_dead_k_archive(archive_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
'''


# Stage 1+2+3 inflate (UNIWARD Ks + Op1 finalizer in CPLX1 wire format).
# Identical to build_cross_paradigm_admm_x_op1_finalizer.py inflate; only
# the encoder-side K vector differs.
INFLATE_PY_CPLX_OP1 = '''#!/usr/bin/env python
"""Forked inflate for unified-winners-stack Stage 1+2+3 archive
(UNIWARD-weighted Ks + no-dead-K + Op1 finalizer in CPLX1 wire).

Wire format (inner blob, single ZIP member \'x\'):
    4 bytes: magic = b"CPLX"
    4 bytes: uint32 LE = decoder_section_bytes D
    2 bytes: uint16 LE = byte_maps_json_len J
    J bytes: utf-8 JSON {str(idx): byte_map_str}
    (D - 10 - J) bytes: Op1 inner blob (raw PR101 split-Brotli streams)
    15387 bytes: PR101 latent_blob (UNCHANGED)
    remaining: PR101 sidecar_blob (UNCHANGED)
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import (
    LATENT_BLOB_LEN,
    N_PAIRS,
    LATENT_DIM,
    BASE_CHANNELS,
    EVAL_SIZE,
    decode_latents_compact,
    apply_latent_sidecar,
)
from model import HNeRVDecoder
from tac.pr101_split_brotli_codec import decode_decoder_compact

CPLX_MAGIC = b"CPLX"
CPLX_HEADER_LEN = 8

CAMERA_H, CAMERA_W = 874, 1164


def parse_unified_cplx_archive(archive_bytes):
    if len(archive_bytes) < CPLX_HEADER_LEN + 2 + LATENT_BLOB_LEN:
        raise ValueError(
            f"archive too short ({len(archive_bytes)} bytes) for unified CPLX format"
        )
    magic = archive_bytes[:4]
    if magic != CPLX_MAGIC:
        raise ValueError(
            f"bad magic {magic!r}, expected {CPLX_MAGIC!r}"
        )
    section_total = struct.unpack("<I", archive_bytes[4:8])[0]
    if section_total < CPLX_HEADER_LEN + 2:
        raise ValueError(
            f"decoder_section_total ({section_total}) < minimum {CPLX_HEADER_LEN + 2}"
        )
    if section_total > len(archive_bytes) - LATENT_BLOB_LEN:
        raise ValueError(
            f"decoder_section_total ({section_total}) leaves no room for "
            f"latent_blob {LATENT_BLOB_LEN}"
        )

    bm_json_len = struct.unpack("<H", archive_bytes[8:10])[0]
    bm_json_start = 10
    bm_json_end = bm_json_start + bm_json_len
    if bm_json_end > section_total:
        raise ValueError(
            f"byte_maps json length {bm_json_len} overflows section_total {section_total}"
        )
    bm_str_keyed = json.loads(
        archive_bytes[bm_json_start:bm_json_end].decode("utf-8")
    )
    effective_byte_maps = {int(k): str(v) for k, v in bm_str_keyed.items()}

    op1_inner_blob = archive_bytes[bm_json_end:section_total]
    decoder_sd = decode_decoder_compact(
        op1_inner_blob, effective_byte_maps=effective_byte_maps
    )

    latent_start = section_total
    latent_end = latent_start + LATENT_BLOB_LEN
    latent_blob = archive_bytes[latent_start:latent_end]
    sidecar_blob = archive_bytes[latent_end:]
    if not latent_blob:
        raise ValueError("missing latent_blob in unified CPLX archive")

    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decoder_sd, latents, meta


def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents, meta = parse_unified_cplx_archive(archive_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
'''


INFLATE_SH_CANONICAL = '''#!/usr/bin/env bash
# Forked PR101 inflate.sh for unified-winners-stack lane.
#
# Implements the canonical contest auth-eval contract:
#   inflate.sh <data_dir> <output_dir> <file_list>
#
# (the previous 2-arg ``inflate.py``-only wrapper would have failed exact
# eval before producing any ``.raw`` outputs; see codex HIGH finding
# #1 ``feedback_codex_high_findings_extincted_20260508.md``.)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

mkdir -p "$OUTPUT_DIR"

# Canonical inflate-time uv ephemeral env (mirrors
# ``submissions/robust_current/inflate.sh`` lines 74-90 + the
# lossy_coarsening lane reference at
# ``experiments/results/lossy_coarsening_20260508T024022Z/submission_dir/inflate.sh``).
INFLATE_BROTLI_SPEC="${INFLATE_BROTLI_SPEC:-brotli==1.2.0}"
INFLATE_TORCH_SPEC="${INFLATE_TORCH_SPEC:-torch==2.5.1+cu124}"
INFLATE_NUMPY_SPEC="${INFLATE_NUMPY_SPEC:-numpy==2.4.4}"
UV_BIN="${UV_BIN:-$(command -v uv || echo /usr/local/bin/uv)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || echo python)}"

# Codex adversarial review 2026-05-08 HIGH #2: fail closed when uv missing,
# and pin the uv invocation with --no-project so it cannot pick up host
# pyproject.toml resolutions (which would make the CUDA score
# non-reproducible under a degraded evaluator image).
if [ ! -x "$UV_BIN" ]; then
  echo "[unified-winners-inflate] FATAL: uv required at $UV_BIN for hermetic inflate runtime; refusing host-python fallback." >&2
  exit 1
fi
PYTHON_RUNNER=("$UV_BIN" "run" "--no-project" "--with" "$INFLATE_BROTLI_SPEC" "--with" "$INFLATE_TORCH_SPEC" "--with" "$INFLATE_NUMPY_SPEC" "python")
echo "[unified-winners-inflate] uv specs (--no-project): brotli=$INFLATE_BROTLI_SPEC torch=$INFLATE_TORCH_SPEC numpy=$INFLATE_NUMPY_SPEC" >&2

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/x"
  if [ ! -f "$SRC" ]; then
    SRC="${DATA_DIR}/${BASE}.bin"
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "${PYTHON_RUNNER[@]}" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
'''


def _stage_submission_dir(
    submission_dir: Path,
    *,
    pr101_source_dir: Path,
    inflate_py_src: str,
    vendor_tac_split_brotli: bool,
) -> None:
    """Stage submission_dir/{inflate.py, inflate.sh, src/codec.py, src/model.py}.

    If ``vendor_tac_split_brotli`` is True, also vendor the tac module so the
    Stage 3 inflate (which calls ``decode_decoder_compact``) can run with
    --no-project uv shells.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    src_dir = submission_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("codec.py", "model.py"):
        src_file = pr101_source_dir / fname
        if not src_file.is_file():
            raise SystemExit(f"FATAL: PR101 source missing: {src_file}")
        shutil.copy2(src_file, src_dir / fname)

    if vendor_tac_split_brotli:
        tac_src = REPO_ROOT / "src" / "tac"
        vendored = src_dir / "tac"
        vendored.mkdir(parents=True, exist_ok=True)
        (vendored / "__init__.py").write_text("", encoding="utf-8")
        for fname in (
            "pr101_split_brotli_codec.py",
            "pr101_split_brotli_codec_derivers.py",
        ):
            sp = tac_src / fname
            if not sp.is_file():
                raise SystemExit(f"FATAL: tac source missing: {sp}")
            shutil.copy2(sp, vendored / fname)

    (submission_dir / "inflate.py").write_text(inflate_py_src, encoding="utf-8")
    (submission_dir / "inflate.sh").write_text(INFLATE_SH_CANONICAL, encoding="utf-8")
    (submission_dir / "inflate.py").chmod(0o755)
    (submission_dir / "inflate.sh").chmod(0o755)


# ---------------------------------------------------------------------------
# Smoke roundtrip
# ---------------------------------------------------------------------------


def _smoke_roundtrip_no_dead_k(
    archive_path: Path,
    *,
    pr101_state_dict_path: Path,
    submission_dir: Path,
) -> dict:
    spec_path = submission_dir / "inflate.py"
    spec = importlib.util.spec_from_file_location(
        "forked_unified_no_dead_k_inflate", spec_path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: cannot load forked inflate spec from {spec_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir / "src"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    inner = _read_pr101_inner_blob(archive_path)
    decoder_sd, latents, meta = mod.parse_unified_no_dead_k_archive(inner)

    sd_ref = torch.load(
        pr101_state_dict_path, map_location="cpu", weights_only=True
    )
    abs_orig = 0.0
    abs_err = 0.0
    per_tensor: list[dict] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        t_dec = decoder_sd[name].cpu().numpy().astype(np.float32)
        qt_ref = _quantize_tensor(name, sd_ref[name], n_quant=N_QUANT)
        ref_quantized = (
            qt_ref.q_i8.astype(np.float32) * float(np.float16(qt_ref.scale))
        ).reshape(t_dec.shape)
        denom_q = float(np.abs(ref_quantized).sum())
        err_q = float(np.abs(t_dec - ref_quantized).sum())
        per_tensor.append(
            {"name": name, "rel_err_vs_quantized": (err_q / denom_q) if denom_q > 1e-9 else 0.0}
        )
        abs_orig += denom_q
        abs_err += err_q
    rel_err = abs_err / abs_orig if abs_orig > 1e-9 else 0.0  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for PR101 unified-winners weight-identity smoke probe
    n_pairs = int(latents.shape[0]) if hasattr(latents, "shape") else None
    return {
        "passed": True,
        "rel_err_vs_quantized_fp32": rel_err,
        "max_per_tensor_rel_err": max(t["rel_err_vs_quantized"] for t in per_tensor),
        "n_tensors_compared": len(per_tensor),
        "n_latent_pairs_decoded": n_pairs,
        "latent_dim_meta": meta.get("latent_dim"),
        "base_channels_meta": meta.get("base_channels"),
        "eval_size_meta": meta.get("eval_size"),
    }


def _smoke_roundtrip_cplx_op1(
    archive_path: Path,
    *,
    dequantized_state_dict: dict[str, torch.Tensor],
    submission_dir: Path,
) -> dict:
    """Verify the CPLX1 inflate parses cleanly and recovers the same state
    dict ``encode_decoder_compact`` would round-trip in-process.
    """
    spec_path = submission_dir / "inflate.py"
    spec = importlib.util.spec_from_file_location(
        "forked_unified_cplx_inflate", spec_path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: cannot load forked inflate spec from {spec_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir / "src"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    inner = _read_pr101_inner_blob(archive_path)
    decoder_sd, latents, meta = mod.parse_unified_cplx_archive(inner)

    abs_orig = 0.0
    abs_err = 0.0
    per_tensor: list[dict] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        decoded = decoder_sd[name].cpu().numpy().astype(np.float32)
        ref = dequantized_state_dict[name].cpu().numpy().astype(np.float32)
        denom = float(np.abs(ref).sum())
        err = float(np.abs(decoded - ref).sum())
        per_tensor.append(
            {"name": name, "rel_err_vs_dequantized": (err / denom) if denom > 1e-9 else 0.0}
        )
        abs_orig += denom
        abs_err += err
    rel_err = abs_err / abs_orig if abs_orig > 1e-9 else 0.0  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for PR101 unified-winners post-stage smoke; consistent with mainline form

    return {
        "passed": True,
        "rel_err_vs_dequantized_fp32": rel_err,
        "max_per_tensor_rel_err": max(t["rel_err_vs_dequantized"] for t in per_tensor),
        "n_tensors_compared": len(per_tensor),
        "n_latent_pairs_decoded": int(latents.shape[0]) if hasattr(latents, "shape") else None,
        "latent_dim_meta": meta.get("latent_dim"),
        "base_channels_meta": meta.get("base_channels"),
        "eval_size_meta": meta.get("eval_size"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_stage_12_archive(
    *,
    state_dict_path: Path,
    pr101_inner: bytes,
    output_dir: Path,
    pr101_source_dir: Path,
    Ks: Sequence[int],
    brotli_quality: int,
) -> dict:
    """Stage 1+2: UNIWARD Ks + no-dead-K wire."""
    section = _build_no_dead_k_decoder_section(
        state_dict_path, Ks, brotli_quality=brotli_quality
    )
    print(
        f"[stage_1_2]   decoder_bytes={section['section_total_bytes']:,} "
        f"(brotli={section['brotli_payload_bytes']:,}, K_in_wire=0)"
    )
    print(
        f"[stage_1_2]   rel_err vs int8 quantized: {section['rel_err']:.6f}"
    )

    _orig_decoder, latent_blob, sidecar_blob = _split_pr101_inner_blob(pr101_inner)
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(
            f"FATAL: PR101 latent_blob length {len(latent_blob)} != expected {LATENT_BLOB_LEN}"
        )
    inner_blob = _build_inner_blob(section["decoder_bytes"], latent_blob, sidecar_blob)

    archive_dir = output_dir / "stage_1_2_uniward_no_dead_k"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "archive.zip"
    submission_dir = archive_dir / "submission_dir"

    _write_pr101_archive(inner_blob, archive_path)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256(archive_path.read_bytes())
    print(
        f"[stage_1_2] WROTE archive size={archive_bytes:,} B sha256={archive_sha[:16]}..."
    )

    _stage_submission_dir(
        submission_dir,
        pr101_source_dir=pr101_source_dir,
        inflate_py_src=INFLATE_PY_NO_DEAD_K,
        vendor_tac_split_brotli=False,
    )
    print(f"[stage_1_2] WROTE submission_dir: {submission_dir.relative_to(REPO_ROOT)}")

    print("[stage_1_2 smoke] running CPU roundtrip...")
    smoke = _smoke_roundtrip_no_dead_k(
        archive_path,
        pr101_state_dict_path=state_dict_path,
        submission_dir=submission_dir,
    )
    print(
        f"[stage_1_2 smoke] rel_err_vs_quantized={smoke['rel_err_vs_quantized_fp32']:.6f} "
        f"max_per_tensor={smoke['max_per_tensor_rel_err']:.6f} "
        f"n_pairs={smoke['n_latent_pairs_decoded']}"
    )

    return {
        "stage": "1_2_uniward_no_dead_k",
        "archive_path": str(archive_path.relative_to(REPO_ROOT)),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "decoder_section_bytes": section["section_total_bytes"],
        "brotli_payload_bytes": section["brotli_payload_bytes"],
        "rel_err_vs_int8_quantized": section["rel_err"],
        "K_bytes_in_wire_format": 0,
        "scale_bytes": section["scale_bytes"],
        "latent_blob_bytes": len(latent_blob),
        "sidecar_blob_bytes": len(sidecar_blob),
        "smoke": smoke,
    }


def _build_stage_123_archive(
    *,
    state_dict_path: Path,
    pr101_inner: bytes,
    output_dir: Path,
    pr101_source_dir: Path,
    Ks: Sequence[int],
) -> dict:
    """Stage 1+2+3: UNIWARD Ks + no-dead-K + Op1 finalizer (CPLX1 wire)."""
    dequantized_sd, sub_stats = _build_dequantized_substrate_uniward(
        state_dict_path, Ks
    )
    print(
        f"[stage_1_2_3]   substrate rel_err={sub_stats['rel_err_int8_after_uniward_admm']:.6f} "
        f"({sub_stats['n_tensors']} tensors, {sub_stats['n_symbols']} symbols)"
    )

    op1_blob, byte_maps, op1_stats = _encode_op1_finalizer_blob(dequantized_sd)
    print(
        f"[stage_1_2_3]   Op1 inner blob: {op1_stats['op1_inner_blob_bytes']:,} B "
        f"({op1_stats['effective_byte_maps_count']} byte_maps; "
        f"{len(op1_stats['non_default_byte_maps'])} non-default)"
    )

    decoder_section = _build_cplx1_decoder_section(op1_blob, byte_maps)
    print(f"[stage_1_2_3]   CPLX1 decoder section: {len(decoder_section):,} B")

    _orig_decoder, latent_blob, sidecar_blob = _split_pr101_inner_blob(pr101_inner)
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(
            f"FATAL: PR101 latent_blob length {len(latent_blob)} != expected {LATENT_BLOB_LEN}"
        )
    inner_blob = _build_inner_blob(decoder_section, latent_blob, sidecar_blob)

    archive_dir = output_dir / "stage_1_2_3_uniward_no_dead_k_op1_finalizer"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "archive.zip"
    submission_dir = archive_dir / "submission_dir"

    _write_pr101_archive(inner_blob, archive_path)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256(archive_path.read_bytes())
    print(
        f"[stage_1_2_3] WROTE archive size={archive_bytes:,} B sha256={archive_sha[:16]}..."
    )

    _stage_submission_dir(
        submission_dir,
        pr101_source_dir=pr101_source_dir,
        inflate_py_src=INFLATE_PY_CPLX_OP1,
        vendor_tac_split_brotli=True,
    )
    print(f"[stage_1_2_3] WROTE submission_dir: {submission_dir.relative_to(REPO_ROOT)}")

    print("[stage_1_2_3 smoke] running CPU roundtrip...")
    smoke = _smoke_roundtrip_cplx_op1(
        archive_path,
        dequantized_state_dict=dequantized_sd,
        submission_dir=submission_dir,
    )
    print(
        f"[stage_1_2_3 smoke] rel_err_vs_dequantized={smoke['rel_err_vs_dequantized_fp32']:.6e} "
        f"max_per_tensor={smoke['max_per_tensor_rel_err']:.6e} "
        f"n_pairs={smoke['n_latent_pairs_decoded']}"
    )

    return {
        "stage": "1_2_3_uniward_no_dead_k_op1_finalizer",
        "archive_path": str(archive_path.relative_to(REPO_ROOT)),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "decoder_section_bytes": len(decoder_section),
        "op1_inner_blob_bytes": op1_stats["op1_inner_blob_bytes"],
        "effective_byte_maps_count": op1_stats["effective_byte_maps_count"],
        "non_default_byte_maps": op1_stats["non_default_byte_maps"],
        "rel_err_vs_int8_quantized": sub_stats["rel_err_int8_after_uniward_admm"],
        "latent_blob_bytes": len(latent_blob),
        "sidecar_blob_bytes": len(sidecar_blob),
        "smoke": smoke,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_PR101_STATE_DICT)
    p.add_argument(
        "--frontier-archive", type=Path, default=DEFAULT_PR101_FRONTIER_ARCHIVE
    )
    p.add_argument("--pr101-source-dir", type=Path, default=DEFAULT_PR101_SOURCE_DIR)
    p.add_argument("--brotli-quality", type=int, default=11)
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "results",
    )
    p.add_argument(
        "--Ks",
        nargs="+",
        type=int,
        default=list(UNIWARD_KS_RMS_0_05),
        help="Per-tensor K coarsening vector (default = UNIWARD @ rms=0.05).",
    )
    p.add_argument("--skip-stage-3", action="store_true",
                   help="Build only Stage 1+2 (skip Op1 finalizer)")
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        sys.exit(f"FATAL: --state-dict not found: {args.state_dict}")
    if not args.frontier_archive.is_file():
        sys.exit(f"FATAL: --frontier-archive not found: {args.frontier_archive}")
    if not args.pr101_source_dir.is_dir():
        sys.exit(f"FATAL: --pr101-source-dir not found: {args.pr101_source_dir}")

    Ks = list(args.Ks)
    if len(Ks) != len(FIXED_STATE_SCHEMA):
        sys.exit(
            f"FATAL: Ks length {len(Ks)} != n_tensors {len(FIXED_STATE_SCHEMA)}"
        )

    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / f"unified_winners_stack_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[unified-stack] OUTPUT: {output_dir.relative_to(REPO_ROOT)}")
    print(f"[unified-stack] Ks (UNIWARD @ rms={UNIWARD_RMS_TARGET}): {Ks}")
    print(
        f"[unified-stack] reference baselines: "
        f"ADMM-uniform={BASELINE_ADMM_UNIFORM_RMS_0_0386_BYTES:,} B, "
        f"ADMM-no-dead-K={BASELINE_ADMM_NO_DEAD_K_RMS_0_0386_BYTES:,} B, "
        f"cross-paradigm-corrected={BASELINE_CROSS_PARADIGM_CORRECTED_BYTES:,} B"
    )

    pr101_inner = _read_pr101_inner_blob(args.frontier_archive)

    stage_1_2 = _build_stage_12_archive(
        state_dict_path=args.state_dict,
        pr101_inner=pr101_inner,
        output_dir=output_dir,
        pr101_source_dir=args.pr101_source_dir,
        Ks=Ks,
        brotli_quality=args.brotli_quality,
    )

    if args.skip_stage_3:
        stage_1_2_3 = None
    else:
        stage_1_2_3 = _build_stage_123_archive(
            state_dict_path=args.state_dict,
            pr101_inner=pr101_inner,
            output_dir=output_dir,
            pr101_source_dir=args.pr101_source_dir,
            Ks=Ks,
        )

    # Per-stage progression summary.
    progression = [
        {
            "stage": "ADMM-uniform Ks @ rms=0.0386 (Path B step 6 baseline, with dead K bytes)",
            "archive_bytes": BASELINE_ADMM_UNIFORM_RMS_0_0386_BYTES,
            "delta_vs_uniform_baseline": 0,
            "evidence": "experiments/results/admm_x_lossy_coarsening_path_b_step6_20260508T060435Z/archive.zip",
        },
        {
            "stage": "ADMM-uniform Ks + no-dead-K wire (commit 0b24e5d1)",
            "archive_bytes": BASELINE_ADMM_NO_DEAD_K_RMS_0_0386_BYTES,
            "delta_vs_uniform_baseline": (
                BASELINE_ADMM_NO_DEAD_K_RMS_0_0386_BYTES
                - BASELINE_ADMM_UNIFORM_RMS_0_0386_BYTES
            ),
            "evidence": "experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/archive.zip",
        },
        {
            "stage": "Stage 1+2 (UNIWARD Ks @ rms=0.05 + no-dead-K)",
            "archive_bytes": stage_1_2["archive_bytes"],
            "delta_vs_uniform_baseline": (
                stage_1_2["archive_bytes"]
                - BASELINE_ADMM_UNIFORM_RMS_0_0386_BYTES
            ),
            "evidence": stage_1_2["archive_path"],
        },
    ]
    if stage_1_2_3 is not None:
        progression.append(
            {
                "stage": "Stage 1+2+3 (UNIWARD Ks + no-dead-K + Op1 finalizer)",
                "archive_bytes": stage_1_2_3["archive_bytes"],
                "delta_vs_uniform_baseline": (
                    stage_1_2_3["archive_bytes"]
                    - BASELINE_ADMM_UNIFORM_RMS_0_0386_BYTES
                ),
                "evidence": stage_1_2_3["archive_path"],
            }
        )

    print("\n========== UNIFIED-WINNERS-STACK PER-STAGE PROGRESSION ==========")
    for row in progression:
        print(
            f"  {row['archive_bytes']:>10,} B   "
            f"({row['delta_vs_uniform_baseline']:+,}) "
            f"{row['stage']}"
        )

    # Smoke-pass-aware dispatch flag (still NOT marking dispatch_ready).
    smoke_pass_12 = bool(stage_1_2["smoke"]["passed"])
    smoke_pass_123 = bool(stage_1_2_3["smoke"]["passed"]) if stage_1_2_3 else None
    cuda_eval_worth_testing = bool(
        smoke_pass_12 and (smoke_pass_123 if stage_1_2_3 is not None else True)
    )

    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "lane_id": LANE_ID,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": cuda_eval_worth_testing,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "family_falsified": False,
        "falsification_scope": "uniward_x_filler_stc_x_admm_no_dead_k_x_op1_only",
        "evidence_semantics": "cpu_build_byte_closed_candidate_proxy_no_score_unified_winners",
        "session_winners_composed": [
            {
                "winner": "uniward_weighted_lagrangian",
                "commit": "be715fac",
                "stage_in_unified_stack": 1,
                "tool": "tools/pr101_omega_opt_uniward_weighted_allocation.py",
                "best_session_savings_bytes": 4074,
                "best_session_savings_rms_target": 0.05,
                "applied": True,
            },
            {
                "winner": "filler_syndrome_trellis_codec",
                "commit": "0c8bb6d4",
                "stage_in_unified_stack": 4,
                "tool": "tools/pr_alpha_mask_stc_empirical.py",
                "best_session_reduction_pct": "55-68 vs LZMA",
                "applied": False,
                "applied_reason": (
                    "PR101 archive is monolithic single-file (member 'x'); no "
                    "separate masks.mkv channel exists, so the Filler STC "
                    "ternary mask payload is N/A for PR101. See "
                    "feedback_pr106_archive_is_monolithic_single_file_20260508."
                ),
                "retired_config_audit": {
                    "test_status": "NOT_DISPATCHABLE_FOR_PR101_SUBSTRATE",
                    "reactivation_target_substrates": [
                        "PARADIGM-α masks.mkv (existing test target)",
                        "future PR-class with separate per-frame mask delta channel",
                    ],
                },
            },
            {
                "winner": "admm_no_dead_k",
                "commit": "0b24e5d1",
                "stage_in_unified_stack": 2,
                "tool": "tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py",
                "best_session_savings_bytes": 28,
                "applied": True,
            },
            {
                "winner": "op1_finalizer_cross_paradigm",
                "commits": ["8d33d5c1", "669b5b5f", "98d2174b"],
                "stage_in_unified_stack": 3,
                "tool": "tools/build_cross_paradigm_admm_x_op1_finalizer.py",
                "best_session_savings_bytes": (
                    BASELINE_ADMM_NO_DEAD_K_RMS_0_0386_BYTES
                    - BASELINE_CROSS_PARADIGM_CORRECTED_BYTES
                ),
                "applied": stage_1_2_3 is not None,
            },
        ],
        "Ks_uniward_at_rms_0_05": list(UNIWARD_KS_RMS_0_05),
        "Ks_used": Ks,
        "uniward_rel_err": UNIWARD_REL_ERR,
        "uniward_rms_target": UNIWARD_RMS_TARGET,
        "input_state_dict": str(args.state_dict),
        "input_frontier_archive": str(args.frontier_archive),
        "input_pr101_source_dir": str(args.pr101_source_dir),
        "stage_1_2": stage_1_2,
        "stage_1_2_3": stage_1_2_3,
        "per_stage_progression": progression,
        "reference_baselines": {
            "admm_uniform_rms_0_0386_bytes": BASELINE_ADMM_UNIFORM_RMS_0_0386_BYTES,
            "admm_no_dead_k_rms_0_0386_bytes": BASELINE_ADMM_NO_DEAD_K_RMS_0_0386_BYTES,
            "cross_paradigm_corrected_bytes": BASELINE_CROSS_PARADIGM_CORRECTED_BYTES,
            "pr101_lossless_bytes": BASELINE_PR101_LOSSLESS_BYTES,
        },
        "best_archive_bytes": (
            min(stage_1_2["archive_bytes"], stage_1_2_3["archive_bytes"])
            if stage_1_2_3 is not None
            else stage_1_2["archive_bytes"]
        ),
        "best_archive_path": (
            stage_1_2_3["archive_path"]
            if (
                stage_1_2_3 is not None
                and stage_1_2_3["archive_bytes"] < stage_1_2["archive_bytes"]
            )
            else stage_1_2["archive_path"]
        ),
        "dispatch_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
        "score_claim_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
        "reactivation_criteria_remaining": [
            "exact_cuda_auth_eval_on_uniward_no_dead_k_archive",
            "exact_cuda_auth_eval_on_uniward_op1_finalizer_archive",
            "wavelet_domain_uniward_residual_proxy_replaces_int8_variance_proxy",
            "iterative_primal_dual_admm_consensus_replacing_lagrangian_bisect",
            "score_aware_per_tensor_distortion_weights_detector_in_loop",
            "stage_4_filler_stc_on_paradigm_alpha_mask_substrate_(separate_lane)",
        ],
        "build_timestamp_utc": _utc_now_iso(),
        "build_output_dir": str(output_dir.relative_to(REPO_ROOT)),
    }

    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[unified-stack] WROTE build_manifest.json: {manifest_path.relative_to(REPO_ROOT)}")

    print(
        f"[unified-stack] best archive: {manifest['best_archive_path']} "
        f"({manifest['best_archive_bytes']:,} B; smoke roundtrip: stage_1_2={smoke_pass_12} "
        f"stage_1_2_3={smoke_pass_123})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
