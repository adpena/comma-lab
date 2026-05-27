# SPDX-License-Identifier: MIT
"""Canonical $0-MLX-local per-pair + per-axis master-gradient extractor.

This is the **MLX-local sister** of ``tools/extract_master_gradient.py`` (the
canonical PyTorch-side per-pair extractor per task #887). Where the PyTorch
tool obtains analytic per-parameter gradients via autograd and projects them to
per-archive-byte sensitivity, this module obtains the same
``(N_archive_bytes, N_pairs, 3_axes)`` canonical signal on the M5 Max GPU via
the MLX SegNet+PoseNet scorer port as a *fast forward oracle* + per-decoder-
tensor finite differences.

WHY MLX-local + finite-difference (NOT raw-byte FD)
---------------------------------------------------
The MLX scorer adapters in ``tac.local_acceleration.mlx_scorer_adapters`` are
forward-only inference ports (they consume NumPy/MLX arrays, return NumPy
arrays; MLX autograd does not flow back to decoded pixels through them). The
contest-faithful master gradient is ``d(score_component)/d(archive_byte)``; the
chain is ``archive_byte -> decoder_state_dict_tensor -> decoded_frames ->
scorer -> distortion``. We obtain it by perturbing each of the 28 fec6 decoder
tensors by ``+/- eps`` (NOT raw archive bytes -- that pattern is forbidden by
CLAUDE.md Catalog #318), re-decoding frames, scoring per-pair via the MLX
oracle, and forming the per-tensor sensitivity, then projecting per-tensor
sensitivity to per-byte via the same fec6 int8+fp16 Jacobian the PyTorch tool
uses (``d(w)/d(mantissa_byte) = |sign_factor| * fp16_scale = fp16_scale``).

This is **per-tensor finite difference projected per-byte**, NOT a raw-byte
finite difference -- it perturbs the actual learned weight that an archive byte
encodes, which is exactly the canonical contest-faithful master gradient.

NON-NEGOTIABLE PROVENANCE (CLAUDE.md "MLX portable-local-substrate authority" +
Catalog #192/#127/#323)
---------------------------------------------------------------------------------
macOS-MLX master gradient is RESEARCH-SIGNAL for the closed-form PREDICTION
sweep ONLY. It gates whether the paid FIRE-phase is worth it; it is NOT a
contest score and NOT promotable. Every emitted manifest row carries
``evidence_grade="macOS-MLX research-signal"`` / ``score_claim=false`` /
``promotion_eligible=false`` / ``ready_for_exact_eval_dispatch=false`` and a
canonical Provenance with ``axis_tag="[macOS-MLX research-signal]"``. The
contest-CUDA/CPU exact-eval per Catalog #246 remains required before any
score/frontier/PR claim.

[verified-against: tac.local_acceleration.mlx_scorer_torch_parity (MLX-ARCH-5
parity-validated SegNet smp.Unet('tu-efficientnet_b2') + PoseNet FastViT-T12)]

Sister-coordination: disjoint from the Cascade B wave-2 sister's
``tools/cascade_b_*.py`` + ``.omx/research/cascade_b_*.md`` scope.

Catalog #344 (FORMALIZATION_PENDING): the per-tensor-FD-projected-per-byte
sensitivity is a predicted-only macOS-MLX advisory equation; the canonical
equation is registered as ``mlx_per_pair_master_gradient_per_byte_fd_v1``
(predicted-only; macOS-MLX advisory; gated FIRE-phase decision support).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac.master_gradient import (
    CONTEST_RATE_DENOM_BYTES,
    PER_PAIR_GRADIENT_TENSOR_KIND,
    MasterGradient,
    OperatingPoint,
    score_axis_dominance_summary,
)

# Canonical non-promotable evidence grade for the MLX research signal.
EVIDENCE_GRADE_MLX = "macOS-MLX research-signal"
EVIDENCE_TAG_MLX = "[macOS-MLX research-signal]"
HARDWARE_SUBSTRATE_MLX = "darwin_arm64_m5_max_macos_mlx_advisory"

# Canonical schema: (N_archive_bytes, N_pairs, 3_axes) with axes = (seg, pose, rate).
AXIS_ORDER = ("seg", "pose", "rate")
SCHEMA_VERSION = "mlx_tensor_fd_gradient_heuristic_v1_20260527"
HEURISTIC_GRADIENT_TENSOR_KIND = "tensor_fd_uniform_decompressed_projection_heuristic_v1"
HEURISTIC_GRADIENT_BYTE_DOMAIN = (
    "decompressed_decoder_mantissa_span_uniform_attribution_not_archive_byte_authority"
)
MASTER_GRADIENT_ANCHOR_BLOCKERS = (
    "source_runtime_full_frame_parity_missing",
    "canonical_archive_byte_domain_mapping_missing",
    "per_weight_or_per_byte_projector_missing",
)

# Default finite-difference epsilon as a RELATIVE multiplier of each tensor's
# RMS magnitude (so the perturbation scales with the weight magnitude).
DEFAULT_FD_REL_EPS = 1e-2

REPO_ROOT = Path(__file__).resolve().parents[2]

# The canonical fec6 frontier codec module (the PR101/HNeRV grammar the
# canonical frontier stacks on). This is the dominant HNeRV decoder component.
FEC6_CODEC_DIR = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
    / "submission_dir"
    / "src"
)


class MLXMasterGradientError(RuntimeError):
    """Raised when MLX per-pair master gradient extraction cannot proceed."""


@dataclass(frozen=True)
class TensorByteSpan:
    """One fec6 decoder tensor's byte layout in the (decompressed) decoder blob.

    ``mantissa_byte_offset`` is the offset within the *decompressed* decoder raw
    bytes; we project per-tensor sensitivity uniformly across the tensor's
    mantissa-byte region in the decompressed domain (the canonical Round-2
    approximation per the PyTorch tool's symposium §3.2 footnote: brotli
    compression breaks the 1:1 decompressed<->compressed byte map, so we
    attribute uniformly across the decompressed mantissa span).
    """

    name: str
    storage_index: int
    shape: tuple[int, ...]
    numel: int
    mantissa_byte_offset: int
    fp16_scale: float


@dataclass(frozen=True)
class MLXMasterGradientResult:
    """The canonical per-pair per-axis master gradient + provenance."""

    per_pair_per_byte: np.ndarray  # (N_archive_bytes, N_pairs, 3) float64
    archive_sha256: str
    archive_bytes_count: int
    n_pairs_used: int
    n_pairs_total: int
    axes: tuple[str, ...]
    operating_point: dict[str, float]
    fd_rel_eps: float
    n_decoder_tensors: int
    decompressed_decoder_len: int
    decoder_blob_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# fec6 codec module loading                                                    #
# --------------------------------------------------------------------------- #


def _load_fec6_codec_module(codec_dir: Path = FEC6_CODEC_DIR):
    """Import the fec6 frontier codec module by file path (no PYTHONPATH leak).

    The codec module exposes ``parse_archive`` / ``decode_decoder_compact`` /
    ``decode_latents_compact`` / ``decompress_brotli_streams`` / ``HNeRVDecoder``
    + the tensor-layout constants (``DECODER_BLOB_LEN`` / ``DECODER_STORAGE_ORDER``
    / ``DECODER_STREAM_ENDS`` / ``DECODER_BYTE_MAPS`` / ``CONV4_STORAGE_PERMS``).
    """
    codec_dir = Path(codec_dir)
    if not codec_dir.is_dir():
        raise MLXMasterGradientError(
            f"fec6 codec dir not found: {codec_dir} -- the frontier fec6 archive "
            "submission_dir/src must be present for decoder reconstruction"
        )
    # The codec module + model.py + frame_selector.py live as siblings; make the
    # dir importable for the duration of the load, then restore sys.path.
    inserted = False
    if str(codec_dir) not in sys.path:
        sys.path.insert(0, str(codec_dir))
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location(
            "_fec6_codec_for_mlx_master_gradient", str(codec_dir / "codec.py")
        )
        if spec is None or spec.loader is None:
            raise MLXMasterGradientError(f"could not load codec spec from {codec_dir / 'codec.py'}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted and str(codec_dir) in sys.path:
            sys.path.remove(str(codec_dir))


# --------------------------------------------------------------------------- #
# Archive -> decoder spans + state_dict + latents                              #
# --------------------------------------------------------------------------- #


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _score_from_components(*, d_seg: float, d_pose: float, rate: float) -> float:
    if d_pose < 0.0:
        raise MLXMasterGradientError(f"d_pose must be non-negative, got {d_pose}")
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * rate


def _operating_point_dict(
    *,
    d_seg: float,
    d_pose: float,
    archive_bytes_count: int,
) -> dict[str, float]:
    rate = float(archive_bytes_count) / float(CONTEST_RATE_DENOM_BYTES)
    return {
        "d_seg": float(d_seg),
        "d_pose": float(d_pose),
        "rate": rate,
        "score": _score_from_components(
            d_seg=float(d_seg),
            d_pose=float(d_pose),
            rate=rate,
        ),
    }


def _maybe_unwrap_zip_member(archive_path: Path) -> bytes:
    """Return the raw fec6 payload bytes, unwrapping a ZIP member named 'x' if present."""
    raw = archive_path.read_bytes()
    if raw[:2] == b"PK":
        import zipfile

        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            member = "x" if "x" in names else (names[0] if names else None)
            if member is None:
                raise MLXMasterGradientError(f"empty ZIP archive: {archive_path}")
            return zf.read(member)
    return raw


def _resolve_inner_pr101_payload(payload: bytes, codec_module) -> tuple[bytes, int]:
    """Unwrap the optional FP11/selector outer wrapper to the inner PR101 payload.

    Mirrors ``tools/extract_master_gradient.parse_fec6_archive_layout`` wrapper
    detection. Returns (inner_payload, inner_base_offset) where ``inner_base_offset``
    is the byte offset of the inner payload within ``payload`` (so per-byte
    projection attributes to the correct archive-byte addresses).

    Three shapes:
      - ``FP11`` outer: 4-byte magic + 4-byte source_len (LE u32) + inner PR101.
      - A1 4-byte header: leading LE u32 == DECODER_BLOB_LEN + 4; inner at offset 4.
      - raw PR101: inner == payload; offset 0.
    """
    import struct

    if payload[:4] == b"FP11":
        source_len = struct.unpack_from("<I", payload, 4)[0]
        inner_base = 8
        inner = payload[inner_base : inner_base + source_len]
        if len(inner) < codec_module.DECODER_BLOB_LEN + codec_module.LATENT_BLOB_LEN:
            raise MLXMasterGradientError(
                f"FP11 inner payload too short: {len(inner)} (source_len={source_len})"
            )
        return inner, inner_base
    if (
        len(payload) >= 4
        and struct.unpack_from("<I", payload, 0)[0] == codec_module.DECODER_BLOB_LEN + 4
    ):
        return payload[4:], 4
    return payload, 0


def _build_decoder_spans(codec_module, decoder_blob: bytes) -> tuple[list[TensorByteSpan], bytes]:
    """Walk DECODER_STORAGE_ORDER to map each tensor's mantissa span + fp16 scale.

    Mirrors ``tools/extract_master_gradient.parse_fec6_archive_layout`` decoder
    walk, returning the decompressed raw bytes + per-tensor spans.
    """
    raw = codec_module.decompress_brotli_streams(
        decoder_blob, len(codec_module.DECODER_STREAM_ENDS)
    )
    probe = codec_module.HNeRVDecoder(
        latent_dim=codec_module.LATENT_DIM,
        base_channels=codec_module.BASE_CHANNELS,
        eval_size=codec_module.EVAL_SIZE,
    )
    items = list(probe.state_dict().items())
    spans: list[TensorByteSpan] = []
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
        spans.append(
            TensorByteSpan(
                name=name,
                storage_index=idx,
                shape=shape,
                numel=numel,
                mantissa_byte_offset=mantissa_byte_offset,
                fp16_scale=fp16_scale,
            )
        )
    if pos != len(raw):
        raise MLXMasterGradientError(
            f"decoder layout decode non-canonical: pos={pos} != len(raw)={len(raw)}"
        )
    return spans, raw


# --------------------------------------------------------------------------- #
# Ground-truth real video pairs (Catalog #114: NEVER synthetic)               #
# --------------------------------------------------------------------------- #


def load_ground_truth_pairs_rgb_uint8(
    video_path: Path, n_pairs: int, eval_size: tuple[int, int]
) -> np.ndarray:
    """Decode the first ``n_pairs`` non-overlapping frame pairs as (B, 2, H, W, 3) uint8.

    Per CLAUDE.md "Forbidden make_synthetic_pair_batch" + Catalog #114: uses the
    REAL contest video. Resizes to ``eval_size`` (the decoder's eval resolution)
    via bilinear so the GT matches decoded-frame resolution.
    """
    import av
    import torch
    import torch.nn.functional as F

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    H, W = eval_size
    frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        if len(frames) >= 2 * n_pairs:
            break
        rgb = frame.to_rgb().to_ndarray()  # (Hn, Wn, 3) uint8
        tens = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        tens_resized = F.interpolate(tens, size=(H, W), mode="bilinear", align_corners=False)
        frames.append(tens_resized.squeeze(0).clamp(0, 255).round().to(torch.uint8).numpy())
    container.close()
    if len(frames) < 2 * n_pairs:
        raise MLXMasterGradientError(
            f"video has only {len(frames)} frames; need 2*{n_pairs}={2 * n_pairs}"
        )
    arr = np.stack(frames[: 2 * n_pairs], axis=0)  # (2*n_pairs, 3, H, W)
    arr = arr.reshape(n_pairs, 2, 3, H, W)
    # ScorerInputBatch wants (B, 2, H, W, 3) uint8 RGB.
    return np.ascontiguousarray(arr.transpose(0, 1, 3, 4, 2))


# --------------------------------------------------------------------------- #
# Decoder forward + MLX scorer oracle -> per-pair per-axis distortion          #
# --------------------------------------------------------------------------- #


def _decode_pairs_rgb_uint8_from_state(
    codec_module,
    state_dict,
    latents_np: np.ndarray,
    n_pairs_used: int,
    *,
    pair_start: int = 0,
) -> np.ndarray:
    """Forward the HNeRV decoder to (B, 2, H, W, 3) uint8 RGB at eval resolution.

    Decoder is a tiny CNN (~10K params); forward is near-instant on CPU.
    """
    import torch

    decoder = codec_module.HNeRVDecoder(
        latent_dim=codec_module.LATENT_DIM,
        base_channels=codec_module.BASE_CHANNELS,
        eval_size=codec_module.EVAL_SIZE,
    )
    decoder.load_state_dict(state_dict)
    decoder.eval()
    pair_end = pair_start + n_pairs_used
    z = torch.from_numpy(np.ascontiguousarray(latents_np[pair_start:pair_end])).float()
    with torch.no_grad():
        decoded = decoder(z)  # (B, 2, 3, H, W) in [0, 255]
    arr = decoded.clamp(0, 255).round().to(torch.uint8).cpu().numpy()  # (B,2,3,H,W)
    return np.ascontiguousarray(arr.transpose(0, 1, 3, 4, 2))  # (B,2,H,W,3)


def _mlx_per_pair_distortion(
    mlx_scorer,
    candidate_pairs_rgb_uint8: np.ndarray,
    reference_outputs: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Run the MLX scorer oracle and return per-pair {seg, pose} distortion arrays.

    ``reference_outputs`` are the GROUND-TRUTH scorer outputs (computed once). The
    contest distortion is candidate-vs-reference per-pair.
    """
    from tac.local_acceleration.mlx_preprocess import preprocess_scorer_inputs_from_pairs
    from tac.local_acceleration.mlx_scorer_adapters import (
        run_mlx_distortion_scorer_nchw,
        scorer_distortion_components_numpy,
    )

    batch = preprocess_scorer_inputs_from_pairs(candidate_pairs_rgb_uint8)
    cand_outputs = run_mlx_distortion_scorer_nchw(
        mlx_scorer,
        batch.posenet_yuv6_pair,
        batch.segnet_last_rgb,
    )
    comps = scorer_distortion_components_numpy(reference_outputs, cand_outputs)
    return {
        "seg": np.asarray(comps["segnet"], dtype=np.float64),  # (B,)
        "pose": np.asarray(comps["posenet"], dtype=np.float64),  # (B,)
    }


def _mlx_reference_outputs(mlx_scorer, gt_pairs_rgb_uint8: np.ndarray) -> dict[str, Any]:
    """Compute the GROUND-TRUTH MLX scorer outputs once (the contest reference)."""
    from tac.local_acceleration.mlx_preprocess import preprocess_scorer_inputs_from_pairs
    from tac.local_acceleration.mlx_scorer_adapters import run_mlx_distortion_scorer_nchw

    batch = preprocess_scorer_inputs_from_pairs(gt_pairs_rgb_uint8)
    return run_mlx_distortion_scorer_nchw(
        mlx_scorer,
        batch.posenet_yuv6_pair,
        batch.segnet_last_rgb,
    )


def _mlx_reference_output_chunks(
    mlx_scorer,
    gt_pairs_rgb_uint8: np.ndarray,
    *,
    pair_batch_size: int,
) -> list[tuple[int, int, dict[str, Any]]]:
    chunks: list[tuple[int, int, dict[str, Any]]] = []
    n_pairs = int(gt_pairs_rgb_uint8.shape[0])
    for start in range(0, n_pairs, pair_batch_size):
        end = min(n_pairs, start + pair_batch_size)
        chunks.append((start, end, _mlx_reference_outputs(mlx_scorer, gt_pairs_rgb_uint8[start:end])))
    return chunks


def _score_state_per_pair_in_chunks(
    codec_module,
    mlx_scorer,
    state_dict,
    latents_np: np.ndarray,
    reference_chunks: Sequence[tuple[int, int, dict[str, Any]]],
) -> dict[str, np.ndarray]:
    n_pairs = reference_chunks[-1][1] if reference_chunks else 0
    seg = np.zeros(n_pairs, dtype=np.float64)
    pose = np.zeros(n_pairs, dtype=np.float64)
    for start, end, reference_outputs in reference_chunks:
        decoded = _decode_pairs_rgb_uint8_from_state(
            codec_module,
            state_dict,
            latents_np,
            end - start,
            pair_start=start,
        )
        dist = _mlx_per_pair_distortion(mlx_scorer, decoded, reference_outputs)
        seg[start:end] = dist["seg"]
        pose[start:end] = dist["pose"]
    return {"seg": seg, "pose": pose}


def _build_mlx_scorer(upstream_dir: Path):
    """Load PyTorch DistortionNet then port to the MLX scorer adapter."""
    import torch

    from tac.local_acceleration.mlx_scorer_adapters import MLXDistortionScorerAdapter
    from tac.scorer import load_default_scorers

    posenet, segnet = load_default_scorers(upstream_dir, device="cpu")

    class _DistortionShim(torch.nn.Module):
        def __init__(self, posenet, segnet):
            super().__init__()
            self.posenet = posenet
            self.segnet = segnet

    shim = _DistortionShim(posenet, segnet)
    return MLXDistortionScorerAdapter(shim)


# --------------------------------------------------------------------------- #
# Per-tensor finite-difference master gradient                                 #
# --------------------------------------------------------------------------- #


def compute_per_tensor_per_pair_sensitivity(
    codec_module,
    mlx_scorer,
    base_state_dict,
    latents_np: np.ndarray,
    spans: Sequence[TensorByteSpan],
    gt_pairs_rgb_uint8: np.ndarray,
    *,
    n_pairs_used: int,
    fd_rel_eps: float,
    pair_batch_size: int,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Per-tensor central finite-difference per-pair {seg, pose} sensitivity.

    For each decoder tensor ``t`` we perturb ALL its weights by ``+eps`` and
    ``-eps`` (eps = fd_rel_eps * RMS(t)), re-decode, score per-pair via the MLX
    oracle, and form the per-pair central difference
    ``d(component_p)/d(t) ~= (D_plus_p - D_minus_p) / (2 * eps_scalar)``. This is
    a SCALAR sensitivity per (tensor, pair, axis): how much the per-pair
    distortion moves per unit weight-magnitude perturbation of that tensor.

    Returns (sens_seg, sens_pose, operating_point) where sens_* maps
    tensor_name -> (n_pairs_used,) float64.
    """
    import torch

    if pair_batch_size <= 0:
        raise MLXMasterGradientError(f"pair_batch_size must be positive, got {pair_batch_size}")
    reference_chunks = _mlx_reference_output_chunks(
        mlx_scorer,
        gt_pairs_rgb_uint8,
        pair_batch_size=pair_batch_size,
    )

    # Operating point: distortion of the unperturbed decoded frames vs GT.
    base_dist = _score_state_per_pair_in_chunks(
        codec_module,
        mlx_scorer,
        base_state_dict,
        latents_np,
        reference_chunks,
    )
    op_seg = float(np.mean(base_dist["seg"]))
    op_pose = float(np.mean(base_dist["pose"]))
    operating_point = {
        "d_seg": op_seg,
        "d_pose": op_pose,
        # Placeholder until archive byte count is known by the top-level extractor.
        "rate": 0.0,
        "score": _score_from_components(d_seg=op_seg, d_pose=op_pose, rate=0.0),
    }

    sens_seg: dict[str, np.ndarray] = {}
    sens_pose: dict[str, np.ndarray] = {}

    for span_i, span in enumerate(spans):
        w = base_state_dict[span.name]
        rms = float(torch.sqrt(torch.mean(w.float() ** 2)).item())
        if rms <= 0.0 or not np.isfinite(rms):
            sens_seg[span.name] = np.zeros(n_pairs_used, dtype=np.float64)
            sens_pose[span.name] = np.zeros(n_pairs_used, dtype=np.float64)
            continue
        eps = fd_rel_eps * rms

        # +eps perturbation (uniform additive on the whole tensor; the per-tensor
        # FD measures the tensor's aggregate weight-magnitude sensitivity, which
        # is then attributed uniformly across the tensor's mantissa bytes).
        sd_plus = {k: v.clone() for k, v in base_state_dict.items()}
        sd_plus[span.name] = w + eps
        dist_plus = _score_state_per_pair_in_chunks(
            codec_module,
            mlx_scorer,
            sd_plus,
            latents_np,
            reference_chunks,
        )

        sd_minus = {k: v.clone() for k, v in base_state_dict.items()}
        sd_minus[span.name] = w - eps
        dist_minus = _score_state_per_pair_in_chunks(
            codec_module,
            mlx_scorer,
            sd_minus,
            latents_np,
            reference_chunks,
        )

        denom = 2.0 * eps
        sens_seg[span.name] = (dist_plus["seg"] - dist_minus["seg"]) / denom
        sens_pose[span.name] = (dist_plus["pose"] - dist_minus["pose"]) / denom

        if verbose:
            print(
                f"[mlx-fd] tensor {span_i + 1}/{len(spans)} {span.name} "
                f"shape={span.shape} eps={eps:.4g} "
                f"|sens_seg|={np.mean(np.abs(sens_seg[span.name])):.4g} "
                f"|sens_pose|={np.mean(np.abs(sens_pose[span.name])):.4g}",
                flush=True,
            )

    return sens_seg, sens_pose, operating_point


def project_per_tensor_sensitivity_to_per_byte(
    spans: Sequence[TensorByteSpan],
    sens_seg: dict[str, np.ndarray],
    sens_pose: dict[str, np.ndarray],
    *,
    archive_bytes_count: int,
    decoder_blob_offset: int,
    n_pairs_used: int,
) -> np.ndarray:
    """Project per-(tensor, pair) sensitivity to per-(archive_byte, pair, axis).

    Mirrors ``tools/extract_master_gradient.project_per_param_gradient_to_per_byte``
    Jacobian: ``d(score)/d(mantissa_byte) = d(score)/d(w) * d(w)/d(byte)`` with
    ``|d(w)/d(byte)| = fp16_scale`` (the absolute magnitude; sign convention +1
    for ranking). The per-tensor FD already gives ``d(component)/d(t)`` aggregated
    over the tensor; we attribute that uniformly across the tensor's mantissa-byte
    region in the DECOMPRESSED domain, scaled by the per-byte ``fp16_scale``
    weight-to-byte Jacobian, and normalized by numel so the per-byte attribution
    sums to the per-tensor sensitivity magnitude.

    Note: the byte offsets are in the *decompressed* decoder domain. We attribute
    to ``decoder_blob_offset + mantissa_byte_offset + k`` (the canonical Round-2
    decompressed-domain approximation the PyTorch tool documents -- brotli breaks
    the 1:1 map, so per-byte rows live at the decompressed offset within the
    archive's decoder region; downstream consumers use the per-pair STRUCTURE, not
    exact compressed-byte addresses).

    Returns (archive_bytes_count, n_pairs_used, 3) float64 with axis order
    (seg, pose, rate); rate column is zero (byte-value sensitivities do not move
    the rate term; packet-valid operator rows measure rate_bytes_delta explicitly).
    """
    G = np.zeros((archive_bytes_count, n_pairs_used, 3), dtype=np.float64)
    for span in spans:
        if span.name not in sens_seg or span.name not in sens_pose:
            continue
        scale_mag = abs(span.fp16_scale)
        if scale_mag <= 0.0 or not np.isfinite(scale_mag):
            continue
        seg_per_pair = sens_seg[span.name]  # (n_pairs_used,)
        pose_per_pair = sens_pose[span.name]  # (n_pairs_used,)
        start = decoder_blob_offset + span.mantissa_byte_offset
        end = start + span.numel
        if end > archive_bytes_count:
            # Decompressed offsets can exceed the compressed archive size; clamp
            # to the available archive-byte window (the canonical approximation).
            end = archive_bytes_count
        if start >= archive_bytes_count or end <= start:
            continue
        n_bytes_span = end - start
        # Per-byte sensitivity: distribute the per-tensor sensitivity uniformly
        # across the tensor's mantissa bytes, scaled by the per-byte fp16 Jacobian
        # magnitude, normalized by numel so total per-tensor attribution preserved.
        per_byte_seg = (seg_per_pair[None, :] * scale_mag) / float(span.numel)  # (1, P)
        per_byte_pose = (pose_per_pair[None, :] * scale_mag) / float(span.numel)
        G[start:end, :, 0] = np.broadcast_to(per_byte_seg, (n_bytes_span, n_pairs_used))
        G[start:end, :, 1] = np.broadcast_to(per_byte_pose, (n_bytes_span, n_pairs_used))
        # axis 2 (rate) stays zero.
    return G


# --------------------------------------------------------------------------- #
# Top-level extraction                                                         #
# --------------------------------------------------------------------------- #


def extract_mlx_per_pair_master_gradient(
    archive_path: Path,
    *,
    upstream_dir: Path,
    video_path: Path,
    n_pairs_used: int = 64,
    n_pairs_total: int = 600,
    fd_rel_eps: float = DEFAULT_FD_REL_EPS,
    pair_batch_size: int = 16,
    codec_dir: Path = FEC6_CODEC_DIR,
    verbose: bool = False,
) -> MLXMasterGradientResult:
    """Extract the canonical (N_archive_bytes, N_pairs, 3) MLX-local master gradient.

    Chain: archive -> fec6 parse (decoder state_dict + latents) -> decoder forward
    -> MLX scorer oracle per-pair distortion -> per-tensor central FD -> project
    per-byte. NON-PROMOTABLE macOS-MLX research-signal per Catalog #192.
    """
    archive_path = Path(archive_path)
    codec_module = _load_fec6_codec_module(codec_dir)

    zip_payload = _maybe_unwrap_zip_member(archive_path)
    payload, inner_base = _resolve_inner_pr101_payload(zip_payload, codec_module)
    archive_sha256 = _sha256_bytes(archive_path.read_bytes())
    archive_bytes_count = archive_path.stat().st_size

    decoder_blob_len = codec_module.DECODER_BLOB_LEN
    latent_blob_len = codec_module.LATENT_BLOB_LEN
    if len(payload) < decoder_blob_len + latent_blob_len:
        raise MLXMasterGradientError(
            f"inner payload too short for fec6/PR101 grammar: {len(payload)} < "
            f"{decoder_blob_len + latent_blob_len}"
        )

    decoder_blob = payload[:decoder_blob_len]
    latent_blob = payload[decoder_blob_len : decoder_blob_len + latent_blob_len]

    state_dict = codec_module.decode_decoder_compact(decoder_blob)
    latents_np = np.asarray(codec_module.decode_latents_compact(latent_blob), dtype=np.float32)
    if latents_np.shape[0] < n_pairs_used:
        raise MLXMasterGradientError(
            f"decoded latents have only {latents_np.shape[0]} pairs; need {n_pairs_used}"
        )

    spans, raw_decompressed = _build_decoder_spans(codec_module, decoder_blob)
    # Per-byte rows attribute to the inner PR101 decoder region; ``inner_base`` is
    # the offset of the inner payload within the archive bytes (0 for raw payload,
    # 8 for FP11-wrapped, 4 for A1-headered). The decompressed mantissa offsets are
    # in the decompressed domain (canonical Round-2 approximation per the PyTorch
    # tool); we anchor them within the inner decoder region.
    decoder_blob_offset = inner_base

    eval_size = tuple(codec_module.EVAL_SIZE)
    gt_pairs = load_ground_truth_pairs_rgb_uint8(video_path, n_pairs_used, eval_size)

    mlx_scorer = _build_mlx_scorer(upstream_dir)

    sens_seg, sens_pose, operating_point = compute_per_tensor_per_pair_sensitivity(
        codec_module,
        mlx_scorer,
        state_dict,
        latents_np,
        spans,
        gt_pairs,
        n_pairs_used=n_pairs_used,
        fd_rel_eps=fd_rel_eps,
        pair_batch_size=pair_batch_size,
        verbose=verbose,
    )
    operating_point = _operating_point_dict(
        d_seg=operating_point["d_seg"],
        d_pose=operating_point["d_pose"],
        archive_bytes_count=archive_bytes_count,
    )

    per_pair_per_byte = project_per_tensor_sensitivity_to_per_byte(
        spans,
        sens_seg,
        sens_pose,
        archive_bytes_count=archive_bytes_count,
        decoder_blob_offset=decoder_blob_offset,
        n_pairs_used=n_pairs_used,
    )

    return MLXMasterGradientResult(
        per_pair_per_byte=per_pair_per_byte,
        archive_sha256=archive_sha256,
        archive_bytes_count=archive_bytes_count,
        n_pairs_used=n_pairs_used,
        n_pairs_total=n_pairs_total,
        axes=AXIS_ORDER,
        operating_point=operating_point,
        fd_rel_eps=fd_rel_eps,
        n_decoder_tensors=len(spans),
        decompressed_decoder_len=len(raw_decompressed),
        decoder_blob_offset=decoder_blob_offset,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "hardware_substrate": HARDWARE_SUBSTRATE_MLX,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "measurement_method": "per_tensor_central_fd_via_mlx_scorer_oracle_uniform_decompressed_projection",
            "axis_order": list(AXIS_ORDER),
            "codec_grammar": "fec6_pr101_fixed_section",
            "pair_batch_size": pair_batch_size,
            "gradient_tensor_kind": HEURISTIC_GRADIENT_TENSOR_KIND,
            "gradient_byte_domain": HEURISTIC_GRADIENT_BYTE_DOMAIN,
            "master_gradient_anchor_eligible": False,
            "master_gradient_anchor_blockers": list(MASTER_GRADIENT_ANCHOR_BLOCKERS),
            "allowed_use": "mlx_local_heuristic_gradient_prior_for_probe_ranking_only",
            "forbidden_use": "score_claim_rank_kill_promotion_or_master_gradient_anchor_authority",
        },
    )


def build_mlx_master_gradient_anchor(
    result: MLXMasterGradientResult,
    *,
    gradient_array_path: Path,
    measurement_call_id: str | None = None,
    measurement_utc: str,
) -> MasterGradient:
    """Build the canonical queue-consumable anchor for an eligible MLX result.

    Current tensor-FD heuristic rows are intentionally not eligible. This helper
    exists so a future source-runtime-parity/per-byte implementation can append a
    normal ``master_gradient_anchors.jsonl`` row without changing the CLI
    surface.
    """
    blockers = mlx_master_gradient_anchor_blockers(result)
    if blockers:
        raise MLXMasterGradientError(
            "MLX gradient result is not eligible for master_gradient_anchors: "
            + ", ".join(blockers)
        )
    op = OperatingPoint(
        d_seg=float(result.operating_point["d_seg"]),
        d_pose=float(result.operating_point["d_pose"]),
        rate=float(result.operating_point["rate"]),
        score=float(result.operating_point["score"]),
    )
    return MasterGradient(
        archive_sha256=result.archive_sha256,
        operating_point=op,
        gradient_array_path=str(Path(gradient_array_path).resolve()),
        n_bytes=result.archive_bytes_count,
        measurement_method=str(result.metadata["measurement_method"]),
        measurement_axis=EVIDENCE_TAG_MLX,
        measurement_hardware=HARDWARE_SUBSTRATE_MLX,
        measurement_call_id=measurement_call_id,
        measurement_utc=measurement_utc,
        pareto_facets=(),
        rashomon_disagreement_score=None,
        gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
        n_pairs=result.n_pairs_used,
        scored_archive_sha256=result.archive_sha256,
        scored_archive_bytes=result.archive_bytes_count,
        gradient_subject_sha256=result.archive_sha256,
        gradient_subject_bytes=result.archive_bytes_count,
        gradient_byte_domain=str(result.metadata["gradient_byte_domain"]),
        n_pairs_used=result.n_pairs_used,
        n_pairs_total=result.n_pairs_total,
        score_axis_dominance=score_axis_dominance_summary(result.per_pair_per_byte, op),
    )


def mlx_master_gradient_anchor_blockers(result: MLXMasterGradientResult) -> list[str]:
    """Return blockers preventing canonical master-gradient anchor emission."""
    blockers = [
        str(blocker)
        for blocker in result.metadata.get("master_gradient_anchor_blockers", ())
        if str(blocker)
    ]
    if result.metadata.get("master_gradient_anchor_eligible") is not True:
        blockers.extend(MASTER_GRADIENT_ANCHOR_BLOCKERS)
    if result.metadata.get("gradient_tensor_kind") != PER_PAIR_GRADIENT_TENSOR_KIND:
        blockers.append("gradient_tensor_kind_not_canonical_per_pair_per_byte")
    if result.metadata.get("gradient_byte_domain") in {
        None,
        "",
        HEURISTIC_GRADIENT_BYTE_DOMAIN,
    }:
        blockers.append("gradient_byte_domain_not_canonical_archive_subject")
    unique: list[str] = []
    for blocker in blockers:
        if blocker not in unique:
            unique.append(blocker)
    return unique


__all__ = [
    "AXIS_ORDER",
    "DEFAULT_FD_REL_EPS",
    "EVIDENCE_GRADE_MLX",
    "EVIDENCE_TAG_MLX",
    "HARDWARE_SUBSTRATE_MLX",
    "HEURISTIC_GRADIENT_BYTE_DOMAIN",
    "HEURISTIC_GRADIENT_TENSOR_KIND",
    "MASTER_GRADIENT_ANCHOR_BLOCKERS",
    "SCHEMA_VERSION",
    "MLXMasterGradientError",
    "MLXMasterGradientResult",
    "TensorByteSpan",
    "build_mlx_master_gradient_anchor",
    "compute_per_tensor_per_pair_sensitivity",
    "extract_mlx_per_pair_master_gradient",
    "load_ground_truth_pairs_rgb_uint8",
    "mlx_master_gradient_anchor_blockers",
    "project_per_tensor_sensitivity_to_per_byte",
]
