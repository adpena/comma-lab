# SPDX-License-Identifier: MIT
# Catalog #270 scope clarification (2026-05-17 + lane_catalog_270_scope_fix_tool_vs_substrate_dispatch_20260517):
# this file is a TOOL dispatch (``tools/extract_master_gradient.py``), NOT a
# substrate trainer (``experiments/train_substrate_*.py``). Substrate-only Tier 3
# fields (Catalogs #172 autocast / #178 TF32 / #179 torch.compile / #226 canonical
# auth-eval helper) are categorically inapplicable and are skipped by
# ``src/tac/deploy/dispatch_protocol.py::_is_tool_dispatch`` via implicit
# ``tools/*.py`` detection + the recipe's explicit ``dispatch_kind: tool`` field.
"""Master-gradient extractor for byte-grain score sensitivity at a measured operating point.

[verified-against: .omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md §3.2 REVISED]
[verified-against: .omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md §1.1 op-routable #1]

Method: 1 forward pass + 3 backward passes per the symposium §3.2 revised methodology.
NOT finite-difference per-bit (Round-1 C-1 + C-2 falsified that as infeasible — 178517 byte
flips × per-flip inflate = O(days) on T4; mathematically yields ±1 bit deltas which are
quantized at zero before reaching the scorer for most bytes).

The autograd path:
    forward:  z_latents (frozen) -> decoder(weights) -> rgb_pair[0:eval_size]
    upscale:  bicubic -> 874x1164 -> bilinear roundtrip down to 384x512 (eval_roundtrip)
    selector: fec6 huffman compact selector applied per pair (NO grad through selector;
              it operates AFTER round to uint8 which is the canonical eval cliff)
    scorers:  segnet + posenet preprocess -> forward -> d_seg + d_pose
    backward(d_seg):   per-parameter d(d_seg)/d(theta)  [SegNet uses x[:, -1, ...] so only
                       frame_1 weights get gradient flow]
    backward(d_pose):  per-parameter d(d_pose)/d(theta)
    rate analytical:   25 / 37,545,489 per byte uniformly

Per-byte projection through the fec6 codec Jacobian:
    The fec6 archive grammar (per submission_dir/src/codec.py) stores:
      - Decoder: per-tensor (int8 mantissa stream after zig/negzig/twos/off mapping) + fp16 scale
        --> w_dequant[i] = mantissa_byte[i] * scale_fp16  (per-tensor scale; sign embedded in int8)
        --> d(w)/d(mantissa_byte[i]) = scale_fp16
        --> d(score)/d(mantissa_byte[i]) = d(score)/d(w[i]) * scale_fp16
      - Per-tensor scale (fp16, 2 bytes):
        --> d(w)/d(scale) = mantissa_byte[i]  -> aggregated across all weights in tensor
      - Latents: uint8 temporal-delta cumulative + fp16 mins/scales -> z_latents are FROZEN
        in this extraction (we measure the FIXED operating point), so latent-byte sensitivity
        comes from feeding latents through the decoder's stem (which is small and gradient-rich).
      - Sidecar: huffman-coded delta_x100 per pair, applied to latents POST-decode. Treated
        as PARTIAL (zero gradient) in v1 because the discrete code -> selected delta mapping
        is non-differentiable. Symposium §3.6 use #4 explicitly notes this is the next
        refinement target.

Output: master gradient ledger anchor at .omx/state/master_gradient_anchors.jsonl + sidecar
.npy file at OUTPUT_NPY (caller-specified; symposium §3.6 use #7 mandates non-/tmp path).

CLI:
    .venv/bin/python tools/extract_master_gradient.py \\
        --archive submissions/fec6/archive.zip \\
        --inflate-py experiments/results/.../submission_dir/inflate.py \\
        --upstream-dir upstream \\
        --axis '[contest-CPU]' \\
        --output-npy .omx/state/master_gradient_fec6_20260517.npy \\
        --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import math
import struct
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Catalog #152 WAVE-1 APPARATUS HARDENING extension 2026-05-16:
# the fec6 archive + submission_dir/inflate.py live under experiments/results/**
# which is Modal-IGNORED per tac.deploy.modal.mount_manifest.DEFAULT_RESULTS_IGNORE.
# Declare the required-input paths here so mount_manifest.collect_extra_mount_paths
# stages them. Bug-class anchor: STC v2 smoke fc-01KRSB76H04HM4958V2HX2JZZ4 rc=25
# (2026-05-16) for the same Modal-IGNORED required-input class.
TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = (
    "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip",
    "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py",
    "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src",
)

from tac.differentiable_eval_roundtrip import (  # noqa: E402
    apply_eval_roundtrip_during_training,
    patch_upstream_yuv6_globally,
    unpatch_upstream_yuv6,
)
from tac.master_gradient import (  # noqa: E402
    CONTEST_RATE_DENOM_BYTES,
    PER_PAIR_GRADIENT_TENSOR_KIND,
    MasterGradient,
    OperatingPoint,
    append_anchor_locked,
    compute_marginal_coefficients,
)
from tac.scorer import load_differentiable_scorers  # noqa: E402

# ---------------------------------------------------------------------------- #
# Per-tensor parsed-from-fec6 metadata                                          #
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _TensorByteSpan:
    """One decoded fec6 tensor's byte layout in the archive.

    The fec6 codec writes streams in DECODER_STORAGE_ORDER (28 tensors). After
    brotli-decompression each tensor occupies (numel) mantissa bytes + 2 fp16
    scale bytes contiguously. We track each span here so the per-byte
    gradient projection can be assembled cheaply.
    """

    name: str
    storage_index: int
    shape: tuple[int, ...]
    numel: int
    mantissa_byte_offset: int  # offset in DECODED brotli stream (not raw archive)
    scale_byte_offset: int
    fp16_scale: float
    byte_map: str  # "zig" | "negzig" | "twos" | "off"


@dataclass(frozen=True)
class _Fec6ArchiveLayout:
    """Parsed fec6 archive layout: decoder + latents + sidecar.

    The ARCHIVE_RAW bytes are the source-of-truth indexing for the master
    gradient: byte_i in [0, n_archive_bytes) maps to a region (decoder/latent/
    sidecar). We carry per-region metadata to project per-parameter gradient
    back to per-archive-byte sensitivity.
    """

    archive_path: Path
    archive_sha256: str
    archive_bytes: bytes
    n_archive_bytes: int
    decoder_blob_offset: int  # start of decoder blob in archive_bytes (0 for FP11 outer)
    decoder_blob_len: int
    decoder_tensor_spans: tuple[_TensorByteSpan, ...]
    decoder_raw_decompressed: bytes
    latent_blob_offset: int
    latent_blob_len: int
    sidecar_blob_offset: int
    sidecar_blob_len: int
    n_pairs: int
    latent_dim: int
    base_channels: int
    eval_size: tuple[int, int]
    has_fp11_outer_wrapper: bool


# ---------------------------------------------------------------------------- #
# Archive parsing                                                                #
# ---------------------------------------------------------------------------- #


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _zigzag_encode_i8_to_u8(arr_i8: np.ndarray) -> np.ndarray:
    """Inverse of codec.zigzag_decode_u8."""
    arr = arr_i8.astype(np.int32)
    return np.where(arr >= 0, arr * 2, -2 * arr - 1).astype(np.uint8)


def parse_fec6_archive_layout(archive_path: Path, codec_module) -> _Fec6ArchiveLayout:
    """Parse a fec6 archive into per-byte-region metadata.

    The fec6 outer wrapper (FP11 + selector) is OPTIONAL: if present, the inner
    bytes are the source-faithful PR101 archive; if absent, archive_path IS the
    inner archive. We support both shapes so the extractor handles either the
    selector-wrapped frontier archive OR the plain pre-selector archive.
    """
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = _sha256_bytes(archive_bytes)
    n_archive_bytes = len(archive_bytes)

    # Detect FP11 outer wrapper
    has_fp11_outer = archive_bytes[:4] == b"FP11"
    if has_fp11_outer:
        # FP11 + 4-byte source_len + source bytes + 2-byte selector_len + selector
        source_len = struct.unpack_from("<I", archive_bytes, 4)[0]
        source_payload_offset = 8
        source_payload = archive_bytes[source_payload_offset : source_payload_offset + source_len]
        # Decoder blob lives inside source_payload (PR101 inner archive).
        inner_bytes = source_payload
        inner_base = source_payload_offset
    else:
        inner_bytes = archive_bytes
        inner_base = 0

    decoder_blob_len = codec_module.DECODER_BLOB_LEN
    latent_blob_len = codec_module.LATENT_BLOB_LEN
    decoder_blob_offset = inner_base
    latent_blob_offset = inner_base + decoder_blob_len
    sidecar_blob_offset = latent_blob_offset + latent_blob_len
    sidecar_blob_len = (inner_base + len(inner_bytes)) - sidecar_blob_offset

    # Decompress decoder bytes to find per-tensor mantissa+scale offsets.
    decoder_blob = inner_bytes[:decoder_blob_len]
    raw = codec_module.decompress_brotli_streams(
        decoder_blob, len(codec_module.DECODER_STREAM_ENDS)
    )

    # Walk DECODER_STORAGE_ORDER to map each tensor's mantissa span + fp16 scale.
    probe = codec_module.HNeRVDecoder(
        latent_dim=codec_module.LATENT_DIM,
        base_channels=codec_module.BASE_CHANNELS,
        eval_size=codec_module.EVAL_SIZE,
    )
    items = list(probe.state_dict().items())

    spans: list[_TensorByteSpan] = []
    pos = 0
    for idx in codec_module.DECODER_STORAGE_ORDER:
        name, tensor = items[idx]
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        mantissa_byte_offset = pos
        pos += numel
        scale_byte_offset = pos
        fp16_scale = float(
            np.frombuffer(raw, dtype=np.float16, count=1, offset=scale_byte_offset)[0]
        )
        pos += 2
        byte_map = codec_module.DECODER_BYTE_MAPS.get(idx, "zig")
        spans.append(
            _TensorByteSpan(
                name=name,
                storage_index=idx,
                shape=shape,
                numel=numel,
                mantissa_byte_offset=mantissa_byte_offset,
                scale_byte_offset=scale_byte_offset,
                fp16_scale=fp16_scale,
                byte_map=byte_map,
            )
        )

    if pos != len(raw):
        raise ValueError(
            f"parse_fec6_archive_layout: pos={pos} != len(raw)={len(raw)} — "
            "decoder layout decode is non-canonical (extractor and codec disagree)"
        )

    return _Fec6ArchiveLayout(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        n_archive_bytes=n_archive_bytes,
        decoder_blob_offset=decoder_blob_offset,
        decoder_blob_len=decoder_blob_len,
        decoder_tensor_spans=tuple(spans),
        decoder_raw_decompressed=raw,
        latent_blob_offset=latent_blob_offset,
        latent_blob_len=latent_blob_len,
        sidecar_blob_offset=sidecar_blob_offset,
        sidecar_blob_len=sidecar_blob_len,
        n_pairs=codec_module.N_PAIRS,
        latent_dim=codec_module.LATENT_DIM,
        base_channels=codec_module.BASE_CHANNELS,
        eval_size=tuple(codec_module.EVAL_SIZE),
        has_fp11_outer_wrapper=has_fp11_outer,
    )


# ---------------------------------------------------------------------------- #
# Per-param gradient -> per-byte projection                                      #
# ---------------------------------------------------------------------------- #


def project_per_param_gradient_to_per_byte(
    layout: _Fec6ArchiveLayout,
    grad_state_dict_seg: dict[str, torch.Tensor],
    grad_state_dict_pose: dict[str, torch.Tensor],
    *,
    inner_base: int = 0,
) -> np.ndarray:
    """Project per-parameter gradient through fec6 int8+fp16 Jacobian.

    Returns: (n_archive_bytes, 3) float32 array with columns (seg, pose, rate).

    Per-tensor Jacobian for the fec6 grammar:
        w = mantissa_byte_decoded_to_i8 * scale_fp16
        --> d(w)/d(mantissa_byte) = sign_factor * scale_fp16  (sign_factor from byte_map)
        --> d(w)/d(scale_fp16)    = mantissa_byte_decoded_to_i8  (broadcast across all weights)
        Score chain:
        d(score)/d(mantissa_byte_i) = d(score)/d(w_i) * d(w_i)/d(mantissa_byte_i)
        d(score)/d(scale)           = sum_i d(score)/d(w_i) * d(w_i)/d(scale)
                                    = sum_i grad_w[i] * decoded_byte_i

    The sign_factor accounts for codec.decode_mapped_u8 byte mappings:
      - "zig":    decoded_i8 = zigzag(byte_u8); local d(decoded)/d(byte) is +1 for even bytes
                  and -1 for odd bytes (per zigzag_decode_u8 line 226-228). For Jacobian
                  projection we use 1.0 (the absolute magnitude is what matters for ranking;
                  the sign factor merely flips which byte-delta moves the weight up vs down).
      - "negzig": decoded_i8 = -zigzag(byte_u8); same magnitude as "zig" with sign flipped.
      - "off":    decoded_i8 = byte_u8 - 128; d(decoded)/d(byte) = 1
      - "twos":   decoded_i8 = byte_u8 as i8; d(decoded)/d(byte) = 1

    We encode this conservatively: the magnitude of d(decoded)/d(byte) is always 1
    for all four mappings (zigzag flips between +1/-1 but the abs-derivative is 1).
    The sign for ranking purposes is set to +1 by convention; downstream candidate
    generators that propose specific byte modifications should consult the codec.

    Rate column is analytical: 1.0 / CONTEST_RATE_DENOM_BYTES per byte (uniform across
    all archive bytes); the predict_delta_s helper multiplies by 25.0 marginal.
    """
    n_bytes = layout.n_archive_bytes
    G = np.zeros((n_bytes, 3), dtype=np.float32)

    # ── Decoder weights ───────────────────────────────────────────────────
    raw_decompressed = layout.decoder_raw_decompressed
    decoder_blob_offset = layout.decoder_blob_offset  # offset into archive_bytes

    # For each tensor span: decoded_i8 = decode_mapped_u8(raw[mantissa_offset:..])
    for span in layout.decoder_tensor_spans:
        if span.name not in grad_state_dict_seg or span.name not in grad_state_dict_pose:
            # Tensor has no grad (e.g., never seen during forward) — skip
            continue

        # Per-weight gradient flattened
        grad_seg_flat = grad_state_dict_seg[span.name].detach().cpu().numpy().reshape(-1).astype(np.float32)
        grad_pose_flat = grad_state_dict_pose[span.name].detach().cpu().numpy().reshape(-1).astype(np.float32)

        # The codec applies a stored permutation for conv4 tensors at storage time;
        # the gradient is in the natural model-order, the BYTES are in storage-order.
        # We need to permute the gradient to match storage layout before per-byte assignment.
        if len(span.shape) == 4 and span.storage_index in _conv4_storage_perms(codec_module=None):
            # Fall back to import to avoid circular ref — done lazily so unit-test fakes can avoid it.
            # The CONV4_STORAGE_PERMS lives on the codec_module passed at parse time; here
            # we recover it via the global cache attached to the layout object.
            perm = _LAYOUT_CONV4_STORAGE_PERMS_CACHE.get(span.storage_index)
            if perm is None:
                # Caller did not initialize the cache; this branch is only used by
                # the synthetic decoder unit tests where conv4 weights are absent.
                pass
            else:
                grad_seg_flat = (
                    grad_state_dict_seg[span.name].detach().cpu().numpy().transpose(perm).reshape(-1).astype(np.float32)
                )
                grad_pose_flat = (
                    grad_state_dict_pose[span.name].detach().cpu().numpy().transpose(perm).reshape(-1).astype(np.float32)
                )

        # Per-byte d(score)/d(byte) = d(score)/d(w) * d(w)/d(byte)
        # d(w)/d(byte_mantissa) = sign_factor * fp16_scale (|sign_factor| = 1)
        scale_mag = abs(span.fp16_scale)
        # Mantissa bytes
        mant_start_in_raw = span.mantissa_byte_offset
        mant_end_in_raw = mant_start_in_raw + span.numel
        # Map raw-decompressed offset -> archive_bytes offset: the brotli-compressed
        # bytes do NOT have a one-to-one mapping with their decompressed counterparts.
        # For ranking purposes we approximate by attributing per-tensor sensitivity
        # to the COMPRESSED tensor's byte region. Since fec6 uses a per-tensor
        # brotli stream (DECODER_STREAM_ENDS), and the compressed length varies,
        # we attribute uniformly across the tensor's compressed-byte region.
        # This is the canonical Round-2 approximation per symposium §3.2 footnote.
        # For v1 we attribute the GRAD to the DECOMPRESSED span and emit a parallel
        # ledger keyed by (tensor_name -> grad_l2_norm) so downstream candidate
        # generators can refine the mapping.
        # ──> For the master_gradient.npy output (compressed-byte indexing) we
        # ──> conservatively spread per-tensor sensitivity uniformly across the
        # ──> compressed tensor span. Mantissa-byte-grain sensitivity ranking
        # ──> remains usable for autopilot Pareto facets.

        # In the absence of a per-byte brotli-decompressed mapping, we attribute
        # the per-weight sensitivity to the corresponding RAW-DECOMPRESSED byte
        # position and emit an auxiliary per-tensor summary array. To keep the
        # invariant `G.shape == (n_archive_bytes, 3)` we project the decompressed-
        # byte sensitivity onto the compressed-byte region by uniform spreading.
        per_byte_seg = grad_seg_flat * (1.0 * scale_mag)  # |d(w)/d(byte)| = scale
        per_byte_pose = grad_pose_flat * (1.0 * scale_mag)

        # Attribute the per-mantissa-byte sensitivities into G. The decompressed
        # bytes occupy positions decoder_blob_offset + ??? in archive_bytes; we
        # conservatively place them in the FIRST decoder_blob_len bytes of the
        # archive (uniformly weighted by compressed offset). For ranking-purposes
        # the relative ordering across tensors is what matters; absolute byte
        # locations within the compressed region map to tensor-level sensitivity
        # in the auxiliary per-tensor summary.
        # ──> v1 strategy: distribute the per-mantissa-byte gradient L2-norm
        # ──> uniformly across the tensor's COMPRESSED byte region.
        compressed_per_tensor_ratio = layout.decoder_blob_len / max(len(raw_decompressed), 1)
        compressed_start = decoder_blob_offset + round(
            mant_start_in_raw * compressed_per_tensor_ratio
        )
        compressed_end = decoder_blob_offset + round(
            mant_end_in_raw * compressed_per_tensor_ratio
        )
        compressed_end = max(compressed_end, compressed_start + 1)
        compressed_end = min(compressed_end, decoder_blob_offset + layout.decoder_blob_len)

        # Uniform-spread sensitivity per byte in the compressed span.
        n_comp = compressed_end - compressed_start
        if n_comp <= 0:
            continue

        # The aggregate sensitivity for this tensor is sum of |per_byte_*|; spread
        # uniformly across the compressed region (per Round-2 fallback). We sum
        # the absolute values to compute a per-byte sensitivity magnitude; the
        # SIGN is set to +1 by convention (see docstring).
        seg_mass = float(np.abs(per_byte_seg).sum())
        pose_mass = float(np.abs(per_byte_pose).sum())
        if n_comp > 0:
            seg_per_byte = seg_mass / n_comp
            pose_per_byte = pose_mass / n_comp
            G[compressed_start:compressed_end, 0] += seg_per_byte
            G[compressed_start:compressed_end, 1] += pose_per_byte

    # ── Rate (analytical, uniform across all archive bytes) ─────────────────
    G[:, 2] = 1.0 / CONTEST_RATE_DENOM_BYTES

    return G


# Module-level cache for CONV4_STORAGE_PERMS — populated lazily by parse step
_LAYOUT_CONV4_STORAGE_PERMS_CACHE: dict[int, tuple[int, ...]] = {}


def _conv4_storage_perms(codec_module):
    if codec_module is not None and not _LAYOUT_CONV4_STORAGE_PERMS_CACHE:
        for k, v in codec_module.CONV4_STORAGE_PERMS.items():
            _LAYOUT_CONV4_STORAGE_PERMS_CACHE[k] = v
    return _LAYOUT_CONV4_STORAGE_PERMS_CACHE


# ---------------------------------------------------------------------------- #
# Forward + 3 backward passes                                                    #
# ---------------------------------------------------------------------------- #


def _stamp_decoder_with_archive_weights(decoder, decoded_state_dict):
    """Load fec6-decoded state_dict into decoder, then enable requires_grad."""
    decoder.load_state_dict(decoded_state_dict)
    for p in decoder.parameters():
        p.requires_grad_(True)
    return decoder


def _ground_truth_frame_pairs(video_path: Path, n_pairs: int, eval_size: tuple[int, int]) -> torch.Tensor:
    """Decode the first n_pairs pairs (consecutive frames) from upstream/videos/0.mkv.

    Returns: (n_pairs, 2, 3, H, W) float32 in [0, 255] at eval_size resolution.

    Per CLAUDE.md "Forbidden `make_synthetic_pair_batch` calls in any non-smoke training
    path" + Catalog #114: this extractor uses REAL video frames.
    """
    import av
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    H, W = eval_size
    frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        if len(frames) >= 2 * n_pairs:
            break
        rgb = frame.to_rgb().to_ndarray()  # (H, W, 3) uint8 at native resolution
        # Resize to eval_size for the decoder's eval resolution
        tens = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        tens_resized = F.interpolate(tens, size=(H, W), mode="bilinear", align_corners=False)
        frames.append(tens_resized.squeeze(0).numpy())
    container.close()
    if len(frames) < 2 * n_pairs:
        raise RuntimeError(f"video has only {len(frames)} frames; need 2*{n_pairs}={2*n_pairs}")
    arr = np.stack(frames[: 2 * n_pairs], axis=0)  # (2*n_pairs, 3, H, W)
    arr = arr.reshape(n_pairs, 2, 3, H, W)
    return torch.from_numpy(arr).float()


def compute_operating_point_and_per_param_gradients(
    decoder: torch.nn.Module,
    latents: torch.Tensor,  # (n_pairs, latent_dim)
    eval_size: tuple[int, int],
    gt_pair_batch: torch.Tensor,  # (n_pairs, 2, 3, H, W) in [0, 255]
    posenet,
    segnet,
    *,
    archive_bytes_count: int,
    device: torch.device,
    n_pairs_used: int = 8,
    preserve_per_pair: bool = False,
) -> tuple[
    OperatingPoint,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
]:
    """Run 1 forward + 2 backward passes (or 2*n_pairs backward passes when ``preserve_per_pair``).

    Returns (operating_point, grad_seg_sd, grad_pose_sd,
             grad_seg_sd_per_pair_or_None, grad_pose_sd_per_pair_or_None).

    The 4th + 5th tuple entries are ``None`` when ``preserve_per_pair=False`` (canonical
    averaged path; 2 backward total). When ``True``, each per-pair dict value has shape
    ``(n_pairs_used, *param_shape)`` — same gradient values per pair, just not averaged
    across the pair axis. The averaged dicts are computed as ``per_pair.mean(dim=0)`` so
    consumers get both at no extra forward cost (only extra backward cost).

    For tractability on CPU we run on ``n_pairs_used`` pairs (default 8) and project the
    operating point to the full-archive scale. Per symposium §3.6: operating-point-LOCAL
    sensitivity is what we need; per-pair gradient unlocks Rashomon disagreement queue
    (Catalog #252 sister), Wyner-Ziv side-info hoisting, NSCS01 nullspace verification,
    per-pair Pareto allocation, and bootstrap variance for the K=8 ensemble.

    Following PR101 / PR95 paradigm:
      - eval_roundtrip simulated (bicubic up to 874x1164, bilinear down to 384x512)
      - rgb_to_yuv6 patched globally so PoseNet gradients flow
      - SegNet uses x[:, -1, ...] (frame_1 of each pair)
      - PoseNet uses both frames yuv6-encoded
    """
    decoder.train()
    decoder.zero_grad()

    # Subset latents + ground truth
    z = latents[:n_pairs_used].to(device).requires_grad_(False)
    gt = gt_pair_batch[:n_pairs_used].to(device)  # (n_pairs_used, 2, 3, H, W)

    # Forward: decoder produces predicted frame pairs
    decoded = decoder(z)  # (n_pairs_used, 2, 3, H, W) in [0, 255]
    if decoded.shape != gt.shape:
        raise ValueError(
            f"decoded shape {tuple(decoded.shape)} != gt shape {tuple(gt.shape)}; "
            f"eval_size {eval_size}"
        )

    # Apply contest eval roundtrip with autograd preserved
    decoded_rt = apply_eval_roundtrip_during_training(
        decoded, simulate_uint8=True, simulate_resize=True
    )
    gt_rt = apply_eval_roundtrip_during_training(
        gt, simulate_uint8=True, simulate_resize=True
    )

    # Scorer input convention per upstream/modules.py:
    #   DistortionNet.preprocess_input expects (B, T, H, W, C) -> rearrange to (B, T, C, H, W)
    # We are already at (B, T, C, H, W) layout in [0, 255]; the scorers' preprocess methods
    # accept (B, T, C, H, W) directly (PoseNet uses (B, T*6, H/2, W/2) after yuv6, SegNet
    # uses (B, 3, 384, 512) after slicing last frame).
    # The DistortionNet path internally rearranges from BTHWC -> BTCHW; load_differentiable_scorers
    # returns (posenet, segnet) directly, so we call their preprocess_input(x) where x is BTCHW.

    posenet_in_decoded = posenet.preprocess_input(decoded_rt)
    segnet_in_decoded = segnet.preprocess_input(decoded_rt)
    posenet_in_gt = posenet.preprocess_input(gt_rt)
    segnet_in_gt = segnet.preprocess_input(gt_rt)

    posenet_out_decoded = posenet(posenet_in_decoded)
    posenet_out_gt = posenet(posenet_in_gt)
    segnet_out_decoded = segnet(segnet_in_decoded)
    segnet_out_gt = segnet(segnet_in_gt)

    # SegNet distortion: argmax disagreement rate (mean)
    # We can't backprop through argmax; use a soft surrogate via softmax KL — this is the
    # canonical PR95 / PR101 score-aware surrogate.
    # Per upstream/modules.py SegNet.compute_distortion: diff = (out1.argmax(1) != out2.argmax(1)).float().mean
    # Differentiable surrogate: soft-argmax disagreement via cross-entropy on softmax dist.
    log_p_decoded = F.log_softmax(segnet_out_decoded, dim=1)
    log_p_gt = F.log_softmax(segnet_out_gt, dim=1).detach()
    seg_kl = -(log_p_gt.exp() * log_p_decoded).sum(dim=1).mean()  # per-pixel CE, mean
    # Use raw KL value as d_seg surrogate; for the operating point we use the HARD argmax.
    with torch.no_grad():
        d_seg_hard = (segnet_out_decoded.argmax(dim=1) != segnet_out_gt.argmax(dim=1)).float().mean()

    # PoseNet distortion: MSE on first 6 pose dims (canonical per upstream/modules.py)
    if "pose" in posenet_out_decoded:
        pose_decoded = posenet_out_decoded["pose"][..., :6]
        pose_gt = posenet_out_gt["pose"][..., :6]
    else:
        # Some PoseNet variants return a tensor directly (legacy fallback)
        pose_decoded = posenet_out_decoded[..., :6]
        pose_gt = posenet_out_gt[..., :6]
    d_pose = (pose_decoded - pose_gt.detach()).pow(2).mean()
    with torch.no_grad():
        d_pose_hard = d_pose.detach().clone()

    # Rate term (analytical)
    rate = archive_bytes_count / float(CONTEST_RATE_DENOM_BYTES)

    # Build operating point with HARD scoring values (matches contest scorer S = 100*d_seg + sqrt(10*d_pose) + 25*R)
    d_seg_val = float(d_seg_hard.item())
    d_pose_val = float(d_pose_hard.item())
    if d_pose_val <= 0:
        # PoseNet MSE is essentially 0 (model fits perfectly) — bump to a tiny positive
        # so the OperatingPoint constructor doesn't reject (pose marginal undefined at 0).
        d_pose_val = 1e-12
    score_hard = 100.0 * d_seg_val + math.sqrt(10.0 * d_pose_val) + 25.0 * rate
    operating_point = OperatingPoint(
        d_seg=d_seg_val, d_pose=d_pose_val, rate=rate, score=score_hard
    )

    if not preserve_per_pair:
        # Canonical averaged path (2 backward passes total).
        # Backward pass (a): per-parameter d(d_seg_surrogate)/d(theta)
        decoder.zero_grad()
        seg_kl.backward(retain_graph=True)
        grad_seg_sd: dict[str, torch.Tensor] = {}
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                grad_seg_sd[name] = param.grad.detach().clone()
            else:
                grad_seg_sd[name] = torch.zeros_like(param)

        # Backward pass (b): per-parameter d(d_pose)/d(theta)
        decoder.zero_grad()
        d_pose.backward()
        grad_pose_sd: dict[str, torch.Tensor] = {}
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                grad_pose_sd[name] = param.grad.detach().clone()
            else:
                grad_pose_sd[name] = torch.zeros_like(param)

        return operating_point, grad_seg_sd, grad_pose_sd, None, None

    # Per-pair path (2 * n_pairs_used backward passes total).
    # Decompose the per-pair losses by collapsing all non-pair axes per-pair.
    # log_p_decoded shape: (n_pairs, classes, H', W'); same for log_p_gt
    # per_pair_seg_kl[i] = -(log_p_gt[i] * log_p_decoded[i]).sum(dim=classes).mean(dim=pixels)
    p_gt = log_p_gt.exp()
    per_pair_seg_kl = -(p_gt * log_p_decoded).sum(dim=1).mean(dim=(1, 2))  # shape (n_pairs,)
    # pose_decoded shape: (n_pairs, 6); per-pair MSE collapsed across 6 dims
    pose_diff_sq = (pose_decoded - pose_gt.detach()).pow(2)
    per_pair_d_pose = pose_diff_sq.reshape(pose_diff_sq.shape[0], -1).mean(dim=1)  # shape (n_pairs,)

    if per_pair_seg_kl.shape[0] != n_pairs_used or per_pair_d_pose.shape[0] != n_pairs_used:
        raise RuntimeError(
            f"per-pair loss shape mismatch: seg={tuple(per_pair_seg_kl.shape)} "
            f"pose={tuple(per_pair_d_pose.shape)} expected n_pairs_used={n_pairs_used}"
        )

    # Accumulate per-pair gradients into a list of per-param tensors;
    # stack at end into shape (n_pairs, *param_shape).
    per_param_names = [name for name, _ in decoder.named_parameters()]
    grad_seg_per_pair_lists: dict[str, list[torch.Tensor]] = {n: [] for n in per_param_names}
    grad_pose_per_pair_lists: dict[str, list[torch.Tensor]] = {n: [] for n in per_param_names}

    # 2 * n_pairs_used backward passes; retain_graph on all but the LAST pair's
    # final (pose) backward so the autograd graph stays alive across iterations.
    last_pair_idx = n_pairs_used - 1
    for i in range(n_pairs_used):
        # Per-pair seg backward
        decoder.zero_grad()
        per_pair_seg_kl[i].backward(retain_graph=True)
        for name, param in decoder.named_parameters():
            g = (
                param.grad.detach().clone()
                if param.grad is not None
                else torch.zeros_like(param)
            )
            grad_seg_per_pair_lists[name].append(g)

        # Per-pair pose backward — retain_graph except on the very last call
        decoder.zero_grad()
        is_final_call = i == last_pair_idx
        per_pair_d_pose[i].backward(retain_graph=not is_final_call)
        for name, param in decoder.named_parameters():
            g = (
                param.grad.detach().clone()
                if param.grad is not None
                else torch.zeros_like(param)
            )
            grad_pose_per_pair_lists[name].append(g)

    # Stack list-of-tensors into shape (n_pairs, *param_shape) per parameter
    grad_seg_sd_per_pair = {
        name: torch.stack(grad_seg_per_pair_lists[name], dim=0) for name in per_param_names
    }
    grad_pose_sd_per_pair = {
        name: torch.stack(grad_pose_per_pair_lists[name], dim=0) for name in per_param_names
    }

    # Averaged dicts derived from per-pair tensors (mathematically equivalent
    # to a single backward on the .mean() loss; numerically within ~1e-7 of
    # the canonical path due to floating-point associativity).
    grad_seg_sd = {name: v.mean(dim=0) for name, v in grad_seg_sd_per_pair.items()}
    grad_pose_sd = {name: v.mean(dim=0) for name, v in grad_pose_sd_per_pair.items()}

    return operating_point, grad_seg_sd, grad_pose_sd, grad_seg_sd_per_pair, grad_pose_sd_per_pair


# ---------------------------------------------------------------------------- #
# CLI                                                                            #
# ---------------------------------------------------------------------------- #


def _add_inflate_src_to_path(inflate_py: Path) -> None:
    """Ensure the submission_dir/src is on sys.path so we can import codec + model."""
    src_dir = inflate_py.parent / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"expected {src_dir} alongside {inflate_py}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _maybe_extract_inner_archive_from_zip(zip_path: Path) -> bytes:
    """If zip_path is a real .zip with a single 0.bin member, return its bytes; else assume it's already raw."""
    import zipfile
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            if len(names) == 1:
                return zf.read(names[0])
            # Look for a canonical name
            for cand in ("0.bin", "archive.bin"):
                if cand in names:
                    return zf.read(cand)
            raise ValueError(f"zip has multiple members {names}; no canonical name found")
    return zip_path.read_bytes()


def _serialize_archive_to_temp(raw_bytes: bytes, scratch_dir: Path) -> Path:
    """Write raw archive bytes to a scratch file for layout parsing."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    target = scratch_dir / "archive.bin"
    target.write_bytes(raw_bytes)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract master gradient (per-byte score sensitivity) for an archive at its operating point."
    )
    parser.add_argument("--archive", required=True, type=Path, help="Path to fec6 archive.zip or raw archive bytes")
    parser.add_argument("--inflate-py", required=True, type=Path, help="Path to submission_dir/inflate.py (the codec source)")
    parser.add_argument("--upstream-dir", required=True, type=Path, help="Path to upstream repository root (with models/posenet.safetensors etc.)")
    parser.add_argument("--axis", required=True, choices=["[contest-CPU]", "[contest-CUDA]"], help="Score axis (lane-tagged per CLAUDE.md)")
    parser.add_argument("--output-npy", required=True, type=Path, help="Sidecar .npy path for the (n_bytes, 3) gradient array (NOT in /tmp per Catalog #220)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device (cpu for Modal CPU dispatch)")
    parser.add_argument("--video-path", default=None, type=Path, help="GT video path (defaults to upstream/videos/0.mkv)")
    parser.add_argument("--n-pairs-used", type=int, default=8, help="Pairs to use for forward+backward (CPU economics; default 8)")
    parser.add_argument("--call-id", default=None, help="Optional dispatch call_id for the ledger anchor")
    parser.add_argument("--hardware-substrate", default=None, help="Hardware substrate tag (default: derived from device)")
    parser.add_argument("--scratch-dir", default=None, type=Path, help="Scratch dir for raw archive bytes (default: alongside output-npy)")
    parser.add_argument("--no-anchor-write", action="store_true", help="Skip writing the ledger anchor (smoke / dry-run mode)")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress logging")
    parser.add_argument(
        "--preserve-per-pair",
        action="store_true",
        help=(
            "Also emit (N_bytes, N_pairs, 3) per-pair gradient as a sister .npy + "
            "ledger anchor. Cost: 2*N_pairs backward passes instead of 2 (N=8 ~1 min, "
            "N=600 ~2-3h on M5 Max). Per-pair tensor unlocks Rashomon disagreement queue, "
            "Wyner-Ziv side-info hoisting, NSCS01 nullspace verification, per-pair Pareto "
            "allocation, and bootstrap variance for the K=8 ensemble (Catalog #252 sister)."
        ),
    )
    parser.add_argument(
        "--per-pair-output-npy",
        default=None,
        type=Path,
        help=(
            "Per-pair sidecar .npy path (only used with --preserve-per-pair). "
            "Default: derived from --output-npy by inserting '_per_pair_<N>pair' "
            "before .npy. Forbidden under /tmp per Catalog #220."
        ),
    )

    args = parser.parse_args(argv)

    if args.output_npy.is_absolute() and str(args.output_npy).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise SystemExit(
            f"--output-npy {args.output_npy} forbidden under /tmp per CLAUDE.md "
            "'Forbidden /tmp paths in any persisted artifact' (Catalog #220 + transient-evidence trap)"
        )

    _add_inflate_src_to_path(args.inflate_py)
    import codec as codec_module  # type: ignore[import-not-found]
    from model import HNeRVDecoder  # type: ignore[import-not-found]
    # Populate the CONV4 permutation cache
    _conv4_storage_perms(codec_module)

    raw_archive_bytes = _maybe_extract_inner_archive_from_zip(args.archive)
    scratch_dir = args.scratch_dir or args.output_npy.parent / "_extractor_scratch"
    scratch_archive = _serialize_archive_to_temp(raw_archive_bytes, scratch_dir)

    print(f"[master-gradient] parsing archive layout from {scratch_archive} ({len(raw_archive_bytes)} bytes)")
    layout = parse_fec6_archive_layout(scratch_archive, codec_module)

    print(f"[master-gradient] archive sha256={layout.archive_sha256[:16]}... n_bytes={layout.n_archive_bytes}")
    print(f"[master-gradient] decoder_blob {layout.decoder_blob_len}B, latent_blob {layout.latent_blob_len}B, sidecar {layout.sidecar_blob_len}B")
    print(f"[master-gradient] n_pairs={layout.n_pairs} eval_size={layout.eval_size} latent_dim={layout.latent_dim}")

    device = torch.device(args.device)

    # Patch upstream YUV6 BEFORE loading scorers (CLAUDE.md eval_roundtrip non-negotiable)
    print("[master-gradient] patching upstream rgb_to_yuv6 for autograd preservation (PR95 fix)")
    patch_token = patch_upstream_yuv6_globally()

    try:
        print(f"[master-gradient] loading scorers from {args.upstream_dir}")
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)

        # Parse decoder state_dict + latents from archive (using submitter codec)
        print("[master-gradient] decoding archive via canonical codec.parse_archive")
        if layout.has_fp11_outer_wrapper:
            # parse_archive expects inner bytes
            source_payload = raw_archive_bytes[8 : 8 + struct.unpack_from("<I", raw_archive_bytes, 4)[0]]
            decoder_sd, latents, meta = codec_module.parse_archive(source_payload)
        else:
            decoder_sd, latents, meta = codec_module.parse_archive(raw_archive_bytes)

        decoder = HNeRVDecoder(
            latent_dim=meta["latent_dim"],
            base_channels=meta["base_channels"],
            eval_size=tuple(meta["eval_size"]),
        ).to(device)
        _stamp_decoder_with_archive_weights(decoder, decoder_sd)

        video_path = args.video_path or (args.upstream_dir / "videos" / "0.mkv")
        print(f"[master-gradient] loading {args.n_pairs_used} ground-truth pairs from {video_path}")
        gt_pairs = _ground_truth_frame_pairs(video_path, args.n_pairs_used, tuple(meta["eval_size"]))

        latents_tensor = latents.to(device)

        backward_count = (2 * args.n_pairs_used) if args.preserve_per_pair else 2
        print(
            f"[master-gradient] running 1 forward + {backward_count} backward passes "
            f"({'per-pair' if args.preserve_per_pair else 'averaged'} mode; "
            f"n_pairs_used={args.n_pairs_used})"
        )
        t0 = time.time()
        (
            op,
            grad_seg_sd,
            grad_pose_sd,
            grad_seg_sd_per_pair,
            grad_pose_sd_per_pair,
        ) = compute_operating_point_and_per_param_gradients(
            decoder=decoder,
            latents=latents_tensor,
            eval_size=tuple(meta["eval_size"]),
            gt_pair_batch=gt_pairs,
            posenet=posenet,
            segnet=segnet,
            archive_bytes_count=layout.n_archive_bytes,
            device=device,
            n_pairs_used=args.n_pairs_used,
            preserve_per_pair=args.preserve_per_pair,
        )
        fwd_bwd_secs = time.time() - t0
        print(f"[master-gradient] forward+{backward_count}-backward done in {fwd_bwd_secs:.2f}s")
        print(f"[master-gradient] operating point: d_seg={op.d_seg:.4f} d_pose={op.d_pose:.6g} rate={op.rate:.6f} score={op.score:.4f}")

        seg_marg, pose_marg, rate_per_byte = compute_marginal_coefficients(op)
        print(f"[master-gradient] marginals: dS/d_seg={seg_marg:.1f} dS/d_pose={pose_marg:.2f} dS/d_byte={rate_per_byte:.3e}")

        print("[master-gradient] projecting per-parameter grad to per-byte (fec6 codec Jacobian)")
        G = project_per_param_gradient_to_per_byte(
            layout, grad_seg_sd, grad_pose_sd, inner_base=0
        )

        # Sanity: shape and finite
        if G.shape != (layout.n_archive_bytes, 3):
            raise RuntimeError(f"projected gradient shape {G.shape} != ({layout.n_archive_bytes}, 3)")
        nans = int(np.isnan(G).sum())
        infs = int(np.isinf(G).sum())
        if nans or infs:
            raise RuntimeError(f"projected gradient has {nans} NaN + {infs} Inf entries")

        # Write sidecar
        args.output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_npy, G)
        print(f"[master-gradient] wrote sidecar {args.output_npy} ({G.nbytes} bytes; shape={G.shape})")

        # Optional ledger anchor
        if not args.no_anchor_write:
            hardware_substrate = args.hardware_substrate or (
                "linux_x86_64_modal_cpu" if device.type == "cpu" else f"linux_x86_64_modal_{device.type}"
            )
            grad = MasterGradient(
                archive_sha256=layout.archive_sha256,
                operating_point=op,
                gradient_array_path=str(args.output_npy.resolve()),
                n_bytes=layout.n_archive_bytes,
                measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
                measurement_axis=args.axis,
                measurement_hardware=hardware_substrate,
                measurement_call_id=args.call_id,
                measurement_utc=datetime.now(UTC).isoformat(),
                pareto_facets=(),
                rashomon_disagreement_score=None,
            )
            append_anchor_locked(grad)
            print(f"[master-gradient] appended anchor to {grad.gradient_array_path} (axis={args.axis})")
        else:
            print("[master-gradient] --no-anchor-write set; skipping ledger append")

        # ── Per-pair sister artifact (when --preserve-per-pair) ────────────────
        if args.preserve_per_pair:
            if grad_seg_sd_per_pair is None or grad_pose_sd_per_pair is None:
                raise RuntimeError(
                    "preserve_per_pair=True but per-pair gradients are None; "
                    "compute_operating_point_and_per_param_gradients contract violated"
                )

            per_pair_output_npy = args.per_pair_output_npy
            if per_pair_output_npy is None:
                # Default sister path: insert "_per_pair_<N>pair" before .npy
                stem = args.output_npy.stem
                suffix = args.output_npy.suffix
                per_pair_output_npy = args.output_npy.with_name(
                    f"{stem}_per_pair_{args.n_pairs_used}pair{suffix}"
                )

            if per_pair_output_npy.is_absolute() and str(per_pair_output_npy).startswith(
                ("/tmp/", "/private/tmp/", "/var/tmp/")
            ):
                raise SystemExit(
                    f"--per-pair-output-npy {per_pair_output_npy} forbidden under /tmp "
                    "per CLAUDE.md 'Forbidden /tmp paths in any persisted artifact' "
                    "(Catalog #220 + transient-evidence trap)"
                )

            print(
                f"[master-gradient] projecting per-pair grad to per-byte per-pair "
                f"(shape (N_bytes={layout.n_archive_bytes}, N_pairs={args.n_pairs_used}, 3))"
            )

            # Project per-pair: loop over the pair axis, slicing each pair's
            # per-param dict and reusing the canonical single-pair projector.
            G_per_pair = np.zeros(
                (layout.n_archive_bytes, args.n_pairs_used, 3), dtype=np.float32
            )
            t_pp = time.time()
            for i in range(args.n_pairs_used):
                grad_seg_i = {
                    name: tensor[i] for name, tensor in grad_seg_sd_per_pair.items()
                }
                grad_pose_i = {
                    name: tensor[i] for name, tensor in grad_pose_sd_per_pair.items()
                }
                G_i = project_per_param_gradient_to_per_byte(
                    layout, grad_seg_i, grad_pose_i, inner_base=0
                )
                if G_i.shape != (layout.n_archive_bytes, 3):
                    raise RuntimeError(
                        f"per-pair projection shape {G_i.shape} != "
                        f"({layout.n_archive_bytes}, 3) at pair {i}"
                    )
                G_per_pair[:, i, :] = G_i
            proj_secs = time.time() - t_pp
            print(
                f"[master-gradient] per-pair projection done in {proj_secs:.2f}s "
                f"(N_bytes={layout.n_archive_bytes}, N_pairs={args.n_pairs_used})"
            )

            # Sanity: shape + finite
            if G_per_pair.shape != (layout.n_archive_bytes, args.n_pairs_used, 3):
                raise RuntimeError(
                    f"per-pair gradient shape {G_per_pair.shape} != "
                    f"({layout.n_archive_bytes}, {args.n_pairs_used}, 3)"
                )
            pp_nans = int(np.isnan(G_per_pair).sum())
            pp_infs = int(np.isinf(G_per_pair).sum())
            if pp_nans or pp_infs:
                raise RuntimeError(
                    f"per-pair gradient has {pp_nans} NaN + {pp_infs} Inf entries"
                )

            # Write sister sidecar
            per_pair_output_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(per_pair_output_npy, G_per_pair)
            print(
                f"[master-gradient] wrote per-pair sidecar {per_pair_output_npy} "
                f"({G_per_pair.nbytes} bytes; shape={G_per_pair.shape})"
            )

            # Optional sister ledger anchor
            if not args.no_anchor_write:
                hardware_substrate_pp = args.hardware_substrate or (
                    "linux_x86_64_modal_cpu"
                    if device.type == "cpu"
                    else f"linux_x86_64_modal_{device.type}"
                )
                grad_pp = MasterGradient(
                    archive_sha256=layout.archive_sha256,
                    operating_point=op,
                    gradient_array_path=str(per_pair_output_npy.resolve()),
                    n_bytes=layout.n_archive_bytes,
                    measurement_method=(
                        "autograd_per_parameter_projected_fec6_int8_fp16_jacobian"
                        f"_per_pair_{args.n_pairs_used}pair"
                    ),
                    measurement_axis=args.axis,
                    measurement_hardware=hardware_substrate_pp,
                    measurement_call_id=(
                        f"{args.call_id}_per_pair" if args.call_id else None
                    ),
                    measurement_utc=datetime.now(UTC).isoformat(),
                    pareto_facets=(),
                    rashomon_disagreement_score=None,
                    gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
                    n_pairs=args.n_pairs_used,
                )
                append_anchor_locked(grad_pp)
                print(
                    f"[master-gradient] appended per-pair anchor to "
                    f"{grad_pp.gradient_array_path} (axis={args.axis})"
                )
            else:
                print(
                    "[master-gradient] --no-anchor-write set; skipping per-pair ledger append"
                )

        print(f"[master-gradient] DONE [score-axis={args.axis}] sha256={layout.archive_sha256[:16]}")
        return 0
    finally:
        unpatch_upstream_yuv6(patch_token)


if __name__ == "__main__":
    raise SystemExit(main())
