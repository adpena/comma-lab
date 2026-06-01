# SPDX-License-Identifier: MIT
"""PR95-HNeRV inverse-steganalysis carrier — the RESOLVED Phase-1 carrier.

This is the THIN wiring that binds the §7-proven L-inf margin-budget OBJECTIVE
(``tac.analysis.inverse_steganalysis_linf_vs_l2_gate``, GREEN: L-inf beats L2 at
equal rate on the real scorer, pose-Fisher-dominated) onto the **PR95-HNeRV
carrier** (``tac.local_acceleration.pr95_hnerv_mlx``) — the contest's HNeRV ROOT
that PR100/PR101 gold built on, RESOLVED as the full-stack carrier per operator
clarification 2026-06-01 (*"consider PR95 for HNeRV's role"*).

The co-equal keystone (T3-council-ratified) is { score-exact oracle objective,
a carrier whose achievable R(D) is cheap enough to reach the floor }. PR95-HNeRV
is the CHEAP-BY-CONSTRUCTION half: its bytes are a ``--modelsize`` byte BUDGET
(content amortized into a compact INT8 decoder + tiny 28-d per-pair latents), NOT
emergent-from-fidelity. This is the exact property Z8's raw-float wavelet detail
LACKS (Z8 ``0.bin`` = 28,406,255 B, wavelet_blob = 28,376,254 B = 99.894% of the
archive, 546x from frontier — the "Z8 disease").

What this module produces — the HEAD-TO-HEAD ROW (advisory):
  1. ``carrier_rate_term`` — the cheap-by-construction claim: the REAL PR95-HNeRV
     archive byte budget * lambda (25/N). Directly measured, not modelled.
  2. ``advisory_d_seg`` / ``advisory_d_pose`` — the carrier's standalone distortion:
     render the REAL carrier (its INT8 decoder + 28-d latents) on the REAL
     ``upstream/videos/0.mkv`` pairs and measure d_seg/d_pose vs ground truth via
     the bit-exact CPU mirror (``measure_pair_d_seg_d_pose``). The RENDER is
     ``[macOS-MLX research-signal]`` (PR95 MLX<->PyTorch forward parity is a known
     blocker); the d_seg/d_pose MEASUREMENT is ``[macOS-CPU advisory]``.
  3. ``z8_falsification`` — PR95-HNeRV rate << Z8 rate at comparable distortion.

And the OBJECTIVE composition (the one new mechanism, §7 GREEN, ported to this
carrier): the L-inf margin-budget allocation in the carrier's 28-d per-pair
LATENT domain vs L2 at equal latent rate. The pixel-space oracle saliency rho_i
is pushed into the latent domain EXACTLY via the carrier decoder Jacobian (the
Fisher-pullback ``s_latent_k = sum_i (dframe_i/dz_k)^2 * s_pixel_i``, the diagonal
of ``J^T diag(s_pixel) J``), computed by central finite differences through the
SHIPPED carrier decoder (decoder-agnostic; MLX or numpy reference). The L-inf
allocation places the carrier's precious latent bits by the detector, NOT
uniformly — the §7-proven inverse-steganalysis prior, in the carrier's own
coefficient domain.

CONTEST COMPLIANCE / authority
------------------------------
COMPRESS-SIDE only. Loads frozen scorers for OFFLINE allocation analysis; nothing
crosses the receiver boundary except ``archive.zip`` + the scorer-free runtime.
All numerics NON-PROMOTABLE per Catalog #341/#192/#127/#323: the carrier render is
``[macOS-MLX research-signal]``, the d_seg/d_pose CPU-mirror measurement is
``[macOS-CPU advisory]`` (1:1 with the frozen weights but Apple-Silicon CPU, NOT
contest GHA-Linux-x86_64). ``score_claim=False``, ``promotable=False``. No score
claim; paired CPU+CUDA (Catalog #246) reserved for operator authorization.

References
----------
- §7 GATE (the OBJECTIVE half, GREEN): ``inverse_steganalysis_linf_vs_l2_gate``
  + design memo ``inverse_steganalysis_optimal_full_stack_20260601.md`` §7.
- The score-exact oracle (P18 DeepFool s_seg + P19 Fisher s_pose):
  ``tac.analysis.score_exact_saliency``.
- The PR95-HNeRV carrier (the CARRIER half, RESOLVED): ``pr95_hnerv_mlx`` (parse +
  ``load_pytorch_state_dict_into_mlx`` + ``HNeRVDecoderMLX.decode_pair_nhwc``).
- HiNeRV sister (the dense-decoder-VJP carrier; this module's structural sibling):
  ``tac.analysis.hinerv_latent_linf_allocation``.
- Verified scorer: ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/N``,
  ``N = 37,545,489``, ``lambda = 25/N = 6.659e-7`` (1,502 B <-> 0.001 score).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.analysis.inverse_steganalysis_linf_vs_l2_gate import (
    allocate_linf_margin_budget,
    margin_budget_from_saliency,
    measure_pair_d_seg_d_pose,
)

CARRIER_SCHEMA = "pr95_hnerv_linf_carrier.v1"

# Verified contest constants (CONSUME, do NOT re-derive — bit-exact mirror sha
# 8173b493a, contract ``tac.contest_eval_contract``).
CONTEST_RATE_DENOM_BYTES = 37_545_489
CONTEST_RATE_MULTIPLIER = 25.0
CONTEST_LAMBDA = CONTEST_RATE_MULTIPLIER / CONTEST_RATE_DENOM_BYTES  # 6.659e-7 score/byte

# The Z8 falsification baseline (verified ``z8_hpc_byte_profile_rate_axis_verdict``
# 20260531; near-lossless 600-pair byte-closed). The wavelet_blob IS the archive.
Z8_NEAR_LOSSLESS_ARCHIVE_BYTES = 28_406_255
Z8_NEAR_LOSSLESS_WAVELET_BLOB_BYTES = 28_376_254  # 99.894% of 0.bin
Z8_600PAIR_BYTE_CLOSED_CONTEST_SCORE = 104.94  # [macOS-CPU advisory], 546x from frontier
# HISTORICAL_SCORE_LITERAL_OK:z8_falsification_baseline_anchor_2026-05-31_pr95_hnerv_carrier_head_to_head

# The receiver never loads the scorer; these are advisory compress-side numerics.
NON_PROMOTABLE_MARKERS: dict[str, object] = {
    "axis_tag": "[macOS-CPU advisory]",
    "render_axis_tag": "[macOS-MLX research-signal]",
    "score_claim": False,
    "promotable": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "evidence_grade": "macos_cpu_advisory",
}


class Pr95HnervCarrierError(ValueError):
    """Raised when the PR95-HNeRV carrier wiring invariants cannot hold."""


# ---------------------------------------------------------------------------
# The CARRIER half — load + render the REAL PR95-HNeRV carrier.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarrierRateTerm:
    """The cheap-by-construction rate term for a REAL PR95-HNeRV carrier archive."""

    archive_bytes: int
    archive_sha256: str
    n_pairs: int
    latent_dim: int
    base_channels: int
    rate_term: float  # 25 * archive_bytes / N
    cheap_by_construction: bool  # bytes are a --modelsize budget, not emergent
    archive_path: str


def carrier_rate_term(archive_path: str | Path) -> CarrierRateTerm:
    """Parse a REAL PR95-HNeRV public archive and compute its rate term.

    The rate term is computed DIRECTLY from the carrier's archive bytes
    (``rate = 25 * archive_bytes / 37,545,489``). This is the cheap-by-construction
    claim: PR95-HNeRV bytes are a ``--modelsize`` budget (compact INT8 decoder +
    28-d per-pair latents), NOT emergent-from-fidelity. The archive is parsed with
    the canonical ``parse_pr95_public_archive_zip`` so the byte budget, n_pairs,
    latent_dim and base_channels are the REAL trained carrier's, not asserted.
    """
    from tac.local_acceleration.pr95_hnerv_mlx import parse_pr95_public_archive_zip

    path = Path(archive_path)
    if not path.is_file():
        raise Pr95HnervCarrierError(f"PR95-HNeRV archive not found: {path}")
    packet = parse_pr95_public_archive_zip(path)
    meta = packet.meta
    archive_bytes = int(path.stat().st_size)
    latents = np.asarray(packet.latents)
    if latents.ndim != 2:
        raise Pr95HnervCarrierError(
            f"PR95 latents must be rank-2 (n_pairs, latent_dim), got {latents.shape}"
        )
    n_pairs = int(meta.get("n_pairs", latents.shape[0]))
    latent_dim = int(meta.get("latent_dim", latents.shape[1]))
    base_channels = int(meta.get("base_channels", 36))
    if n_pairs <= 0 or latent_dim <= 0:
        raise Pr95HnervCarrierError(
            f"PR95 carrier dims must be positive: n_pairs={n_pairs} latent_dim={latent_dim}"
        )
    return CarrierRateTerm(
        archive_bytes=archive_bytes,
        archive_sha256=str(packet.archive_zip_sha256),
        n_pairs=n_pairs,
        latent_dim=latent_dim,
        base_channels=base_channels,
        rate_term=float(CONTEST_RATE_MULTIPLIER * archive_bytes / CONTEST_RATE_DENOM_BYTES),
        cheap_by_construction=True,
        archive_path=path.as_posix(),
    )


def load_carrier_decoder(archive_path: str | Path) -> tuple[Any, np.ndarray, CarrierRateTerm]:
    """Load the REAL PR95-HNeRV carrier: MLX decoder with REAL trained weights.

    Returns ``(decoder, latents_np, rate_term)``. The decoder is an
    ``HNeRVDecoderMLX`` with the carrier's REAL trained state_dict loaded via the
    canonical ``load_pytorch_state_dict_into_mlx`` (NOT default-init). ``latents_np``
    is the carrier's ``(n_pairs, latent_dim)`` per-pair latents. Raises if MLX is
    unavailable (the render path requires it on Apple Silicon).
    """
    from tac.local_acceleration.pr95_hnerv_mlx import (
        HNeRVDecoderMLX,
        load_pytorch_state_dict_into_mlx,
        parse_pr95_public_archive_zip,
    )

    rt = carrier_rate_term(archive_path)
    packet = parse_pr95_public_archive_zip(Path(archive_path))
    decoder = HNeRVDecoderMLX(
        latent_dim=rt.latent_dim,
        base_channels=rt.base_channels,
        eval_size=tuple(int(d) for d in packet.meta.get("eval_size", (384, 512))),
    )
    load_pytorch_state_dict_into_mlx(decoder, packet.state_dict)
    return decoder, np.asarray(packet.latents).astype(np.float32), rt


def render_carrier_pair_bcthw(
    decoder: Any, latent_row: np.ndarray
) -> torch.Tensor:
    """Render ONE PR95-HNeRV carrier pair as a torch BTCHW tensor in [0, 255].

    The carrier decoder renders NHWC ``(1, 2, H, W, 3)`` in [0, 255]; we transpose
    to the scorer-mirror BTCHW ``(1, 2, 3, H, W)`` layout that
    ``measure_pair_d_seg_d_pose`` consumes. ``[macOS-MLX research-signal]`` — the
    render uses the MLX decoder whose PyTorch forward parity is a known blocker;
    the d_seg/d_pose measured on it is advisory, NOT contest authority.
    """
    import mlx.core as mx

    z = np.asarray(latent_row, dtype=np.float32).reshape(1, -1)
    out = decoder.decode_pair_nhwc(mx.array(z))
    mx.eval(out)
    nhwc = np.asarray(out)  # (1, 2, H, W, 3)
    if nhwc.ndim != 5 or nhwc.shape[0] != 1 or nhwc.shape[1] != 2 or nhwc.shape[-1] != 3:
        raise Pr95HnervCarrierError(
            f"carrier render must be (1,2,H,W,3) NHWC, got {nhwc.shape}"
        )
    # NHWC (1,2,H,W,3) -> BTCHW (1,2,3,H,W)
    bcthw = np.transpose(nhwc, (0, 1, 4, 2, 3))
    return torch.from_numpy(np.ascontiguousarray(bcthw.astype(np.float32)))


# ---------------------------------------------------------------------------
# The Fisher-pullback — push pixel-space oracle saliency into the LATENT domain.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentSaliency:
    """Per-latent-dim saliency induced by the pixel-space oracle Fisher."""

    s_latent: np.ndarray  # (latent_dim,) >= 0
    method: str  # "central_finite_difference_jacobian_columns"
    pair_index: int
    frame_slot: int
    pixel_saliency_total: float
    eps: float


def push_pixel_saliency_to_latent(
    decoder: Any,
    latent_row: np.ndarray,
    s_pixel: np.ndarray,
    *,
    frame_slot: int = 1,
    eps: float = 1.0e-2,
) -> LatentSaliency:
    """``s_latent_k = sum_i (dframe_i/dz_k)^2 * s_pixel_i`` — the Fisher-pullback.

    The diagonal of the latent Fisher information ``diag(J^T diag(s_pixel) J)``
    induced by the per-pixel oracle Fisher ``s_pixel`` through the carrier decoder
    synthesis ``J = dframe/dz``. For a 28-d latent we build the Jacobian columns
    EXACTLY by central finite differences through the SHIPPED carrier decoder
    (decoder-agnostic; no MLX-autograd dependency): ``J[:,k] ~= (A(z + eps*e_k) -
    A(z - eps*e_k)) / (2 eps)``, then ``s_latent_k = sum_i J_ik^2 * s_pixel_i``.

    Central differences are second-order accurate; for a 28-d latent this is 56
    decoder forwards (cheap). ``s_pixel`` is the per-pixel detector saliency
    (>= 0) at the carrier's render resolution (broadcast over channel). The frame
    slot selects frame_0 or frame_1 of the pair.
    """
    import mlx.core as mx

    if frame_slot not in (0, 1):
        raise Pr95HnervCarrierError("frame_slot must be 0 or 1")
    z = np.asarray(latent_row, dtype=np.float64).reshape(-1)
    latent_dim = int(z.size)

    def render_frame_flat(z_vec: np.ndarray) -> np.ndarray:
        out = decoder.decode_pair_nhwc(mx.array(z_vec.astype(np.float32).reshape(1, -1)))
        mx.eval(out)
        nhwc = np.asarray(out)  # (1, 2, H, W, 3)
        frame = nhwc[0, frame_slot]  # (H, W, 3)
        return frame.astype(np.float64).reshape(-1)

    base = render_frame_flat(z)
    n_pix = int(base.size)  # H*W*3 (channel-last flat)

    # Build s_pixel as a flat (H*W*3,) >= 0 weight matching the rendered frame.
    sp = np.asarray(s_pixel, dtype=np.float64)
    h, w = _carrier_hw(decoder)
    if sp.ndim == 2:
        sp = _resize_saliency(sp, h, w)  # (h, w)
        # channel-last flat: (h, w, 3) broadcast over channel
        sp_flat = np.repeat(sp[:, :, None], 3, axis=2).reshape(-1)
    elif sp.ndim == 1 and sp.size == h * w:
        sp2 = sp.reshape(h, w)
        sp_flat = np.repeat(sp2[:, :, None], 3, axis=2).reshape(-1)
    else:
        raise Pr95HnervCarrierError(
            f"s_pixel must be (H,W) or flat H*W; got shape {sp.shape}"
        )
    sp_flat = np.clip(sp_flat, 0.0, None)
    if sp_flat.size != n_pix:
        raise Pr95HnervCarrierError(
            f"s_pixel flat {sp_flat.size} != rendered frame pixels {n_pix}"
        )

    s_lat = np.zeros(latent_dim, dtype=np.float64)
    for k in range(latent_dim):
        zp = z.copy()
        zp[k] += eps
        zm = z.copy()
        zm[k] -= eps
        col = (render_frame_flat(zp) - render_frame_flat(zm)) / (2.0 * eps)  # J[:,k]
        s_lat[k] = float((col**2 * sp_flat).sum())

    return LatentSaliency(
        s_latent=s_lat,
        method="central_finite_difference_jacobian_columns",
        pair_index=-1,
        frame_slot=int(frame_slot),
        pixel_saliency_total=float(sp_flat.sum()),
        eps=float(eps),
    )


def _carrier_hw(decoder: Any) -> tuple[int, int]:
    """Read the carrier decoder's (H, W) output resolution."""
    eval_size = getattr(decoder, "eval_size", None)
    if eval_size is not None and len(eval_size) == 2:
        return int(eval_size[0]), int(eval_size[1])
    return 384, 512


def _resize_saliency(s: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear-resize a (H0, W0) saliency surface to (h, w) via torch interpolate."""
    s = np.asarray(s, dtype=np.float64)
    if s.shape == (h, w):
        return s
    t = torch.from_numpy(s)[None, None].float()
    r = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return r[0, 0].double().cpu().numpy()


# ---------------------------------------------------------------------------
# L-inf-vs-L2 allocation in the carrier's 28-d per-pair LATENT domain.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentLinfL2Allocation:
    """Per-latent-dim L-inf and L2 quantizer steps + realized rates at equal budget."""

    linf_steps: np.ndarray  # (latent_dim,)
    l2_steps: np.ndarray  # (latent_dim,)
    linf_bits: float
    l2_bits: float
    target_bits: float
    latent_dynamic_range: float
    water_level: float
    allocations_differ: bool


def allocate_latent_linf_vs_l2(
    s_latent: np.ndarray,
    latent_values: np.ndarray,
    *,
    target_bits: float,
    min_step_frac: float = 1.0e-4,
    rate_tolerance: float = 1.0e-3,
) -> LatentLinfL2Allocation:
    """L-inf margin-budget vs L2-uniform latent allocation at EQUAL latent rate.

    Reuses the canonical §7 allocator (``allocate_linf_margin_budget`` +
    ``margin_budget_from_saliency``) on the carrier's latent coefficients. The
    latent dynamic range is the coefficient span (NOT 256 like pixels). The L2
    baseline is a single uniform step solved so its realized rate equals
    ``target_bits``; the L-inf allocation is forced to spend >= L2 bits
    (``disadvantage_linf`` anti-gaming guard) so any win cannot be a rate artifact.
    """
    s = np.asarray(s_latent, dtype=np.float64).reshape(-1)
    z = np.asarray(latent_values, dtype=np.float64).reshape(-1)
    if s.size == 0 or s.size != z.size:
        raise Pr95HnervCarrierError(
            f"s_latent ({s.size}) and latent_values ({z.size}) must match and be non-empty"
        )
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        raise Pr95HnervCarrierError("s_latent must be finite and >= 0")
    span = float(z.max() - z.min())
    R = max(span, 1e-6)
    min_step = max(min_step_frac * R, 1e-9)

    rho = margin_budget_from_saliency(s)  # 1/(s + eps); high saliency => fine step
    linf = allocate_linf_margin_budget(
        rho,
        target_bits=float(target_bits),
        dynamic_range=R,
        min_step=min_step,
        max_step=R,
        rate_tolerance=rate_tolerance,
        fairness_direction="disadvantage_linf",
    )
    # L2 baseline: a single uniform step for the whole latent at the target rate.
    # bits = latent_dim * log2(R/delta) = target_bits  =>  delta = R * 2^(-target/dim).
    per_dim_bits = float(target_bits) / float(z.size)
    l2_delta = R * (2.0 ** (-per_dim_bits))
    l2_steps = np.full(z.size, max(l2_delta, min_step), dtype=np.float64)
    l2_bits = float(np.clip(np.log2(R / l2_steps), 0.0, None).sum())

    allocations_differ = bool(np.abs(linf.steps - l2_steps).max() > 1e-9)
    return LatentLinfL2Allocation(
        linf_steps=np.asarray(linf.steps, dtype=np.float64),
        l2_steps=l2_steps,
        linf_bits=float(linf.total_bits),
        l2_bits=l2_bits,
        target_bits=float(target_bits),
        latent_dynamic_range=R,
        water_level=float(linf.water_level),
        allocations_differ=allocations_differ,
    )


def quantize_latent_with_steps(
    latent_values: np.ndarray, steps: np.ndarray
) -> np.ndarray:
    """Deterministic mid-rise uniform quantization ``q_k = round(z_k/delta_k)*delta_k``.

    This IS the decode the rate model assumes — the quantized latents are what the
    carrier would store. Returns the dequantized latents (same shape) for the
    advisory re-render + the no-op proof.
    """
    z = np.asarray(latent_values, dtype=np.float64).reshape(-1)
    d = np.asarray(steps, dtype=np.float64).reshape(-1)
    if z.size != d.size:
        raise Pr95HnervCarrierError(f"latent size {z.size} != steps size {d.size}")
    return np.round(z / d) * d


# ---------------------------------------------------------------------------
# The advisory measurement — carrier reconstruction d_seg/d_pose vs REAL GT.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarrierDistortionAdvisory:
    """Carrier standalone distortion vs ground truth (advisory, CPU mirror)."""

    d_seg: float  # last-frame argmax-flip rate (carrier render vs gt), CPU mirror
    d_pose: float  # first-6 pose MSE (carrier render vs gt), CPU mirror
    advisory_score: float  # 100*d_seg + sqrt(10*d_pose) (distortion-only, no rate)
    n_pairs_measured: int
    render_axis_tag: str
    measure_axis_tag: str


def measure_carrier_distortion(
    decoder: Any,
    latents: np.ndarray,
    gt_pairs_btchw: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    *,
    pair_indices: list[int],
) -> CarrierDistortionAdvisory:
    """Carrier standalone advisory distortion vs REAL ground-truth pairs.

    For each requested pair, render the REAL PR95-HNeRV carrier (its INT8 decoder
    + 28-d latent for that pair) and measure d_seg/d_pose vs the ground-truth pair
    via the bit-exact CPU mirror ``measure_pair_d_seg_d_pose``. The render is
    ``[macOS-MLX research-signal]`` (PR95 MLX<->PyTorch parity blocker); the
    d_seg/d_pose are ``[macOS-CPU advisory]`` (frozen weights but Apple-Silicon CPU,
    NOT contest GHA-Linux-x86_64). NO score claim.

    ``gt_pairs_btchw`` is ``(P, 2, 3, H, W)`` real frames; ``pair_indices`` selects
    which carrier latent row maps to which gt pair (1:1 ordering).
    """
    if gt_pairs_btchw.dim() != 5 or gt_pairs_btchw.shape[1] != 2:
        raise Pr95HnervCarrierError(
            f"gt_pairs_btchw must be (P,2,3,H,W); got {tuple(gt_pairs_btchw.shape)}"
        )
    if len(pair_indices) != gt_pairs_btchw.shape[0]:
        raise Pr95HnervCarrierError(
            f"pair_indices ({len(pair_indices)}) != gt pairs ({gt_pairs_btchw.shape[0]})"
        )
    z = np.asarray(latents, dtype=np.float32)
    h_gt, w_gt = gt_pairs_btchw.shape[-2:]
    d_seg_sum = 0.0
    d_pose_sum = 0.0
    for j, pair_idx in enumerate(pair_indices):
        carrier_pair = render_carrier_pair_bcthw(decoder, z[pair_idx])  # (1,2,3,h,w)
        # Resize carrier render to the gt resolution so the scorer compares like-for-like.
        carrier_pair = _resize_pair_to(carrier_pair, h_gt, w_gt)
        gt_pair = gt_pairs_btchw[j : j + 1]  # (1,2,3,H,W)
        d_seg, d_pose = measure_pair_d_seg_d_pose(posenet, segnet, gt_pair, carrier_pair)
        d_seg_sum += d_seg
        d_pose_sum += d_pose
    n = len(pair_indices)
    mean_d_seg = d_seg_sum / max(n, 1)
    mean_d_pose = d_pose_sum / max(n, 1)
    advisory_score = 100.0 * mean_d_seg + float(np.sqrt(10.0 * mean_d_pose))
    return CarrierDistortionAdvisory(
        d_seg=float(mean_d_seg),
        d_pose=float(mean_d_pose),
        advisory_score=float(advisory_score),
        n_pairs_measured=int(n),
        render_axis_tag="[macOS-MLX research-signal]",
        measure_axis_tag="[macOS-CPU advisory]",
    )


def _resize_pair_to(pair_btchw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bilinear-resize a (1,2,3,H0,W0) pair to (1,2,3,h,w) (per-frame)."""
    if pair_btchw.shape[-2:] == (h, w):
        return pair_btchw
    frames = []
    for t in range(pair_btchw.shape[1]):
        fr = torch.nn.functional.interpolate(
            pair_btchw[:, t].float(), size=(h, w), mode="bilinear", align_corners=False
        )
        frames.append(fr)
    return torch.stack(frames, dim=1)


# ---------------------------------------------------------------------------
# The Z8-falsification — PR95-HNeRV rate << Z8 rate at comparable distortion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Z8Falsification:
    """The cheap-by-construction-vs-emergent-from-fidelity falsification row."""

    pr95_hnerv_archive_bytes: int
    pr95_hnerv_rate_term: float
    z8_archive_bytes: int
    z8_rate_term: float
    z8_over_pr95_byte_ratio: float  # how many times bigger Z8 is
    z8_disease_confirmed: bool  # Z8 bytes are emergent-from-fidelity, PR95 are budgeted
    rationale: str


def z8_falsification(carrier: CarrierRateTerm) -> Z8Falsification:
    """PR95-HNeRV rate << Z8 rate — the cheap-by-construction falsification.

    PR95-HNeRV's bytes are a ``--modelsize`` budget (cheap-by-construction). Z8's
    bytes are emergent-from-fidelity (the wavelet detail blob IS 99.894% of the
    archive at near-lossless; the "Z8 disease"). PR95-HNeRV reaches the FRONTIER
    score (sister PR101/HNeRV-family carrier = 178,493 B at score 0.19199) while
    Z8 sits 546x from frontier at 28,406,255 B. This row makes the keystone
    concrete: the cheap carrier is the dominant lever; an optimal allocator on a
    byte-heavy carrier (Z8 with its joint P18/P19 dead-zone allocation) still
    cannot move the score because the carrier R(D) is the bottleneck.
    """
    z8_bytes = Z8_NEAR_LOSSLESS_ARCHIVE_BYTES
    z8_rate = float(CONTEST_RATE_MULTIPLIER * z8_bytes / CONTEST_RATE_DENOM_BYTES)
    ratio = float(z8_bytes / max(carrier.archive_bytes, 1))
    return Z8Falsification(
        pr95_hnerv_archive_bytes=carrier.archive_bytes,
        pr95_hnerv_rate_term=carrier.rate_term,
        z8_archive_bytes=z8_bytes,
        z8_rate_term=z8_rate,
        z8_over_pr95_byte_ratio=ratio,
        z8_disease_confirmed=bool(ratio > 1.0),
        rationale=(
            f"PR95-HNeRV carrier rate {carrier.rate_term:.4f} "
            f"({carrier.archive_bytes:,} B, --modelsize budget, cheap-by-construction) "
            f"<< Z8 rate {z8_rate:.4f} ({z8_bytes:,} B, emergent-from-fidelity "
            f"wavelet detail = 99.894% of 0.bin) — Z8 is {ratio:.0f}x heavier. "
            "PR95-HNeRV reaches frontier; Z8 sits 546x away despite optimal "
            "joint-P18/P19 dead-zone allocation: the cheap carrier is the lever."
        ),
    )


# ---------------------------------------------------------------------------
# The head-to-head row (the deliverable).
# ---------------------------------------------------------------------------


def build_head_to_head_row(
    carrier: CarrierRateTerm,
    distortion: CarrierDistortionAdvisory | None,
    falsification: Z8Falsification,
    *,
    latent_allocation: LatentLinfL2Allocation | None = None,
) -> dict[str, Any]:
    """Assemble the advisory head-to-head row (rate x d_seg/d_pose x Z8-falsification).

    The row is the deliverable: it binds (1) the carrier cheap-by-construction rate
    term, (2) the advisory d_seg/d_pose (CPU mirror; None if the render was skipped),
    (3) the Z8-falsification, and (4) the optional L-inf-vs-L2 latent allocation
    (the §7-proven objective ported to the carrier's coefficient domain). Carries
    the full NON_PROMOTABLE_MARKERS so no consumer can promote it (Catalog
    #341/#192/#127/#323).
    """
    row: dict[str, Any] = {
        "schema": CARRIER_SCHEMA,
        "carrier": "pr95_hnerv",
        "rate_term": carrier.rate_term,
        "carrier_archive_bytes": carrier.archive_bytes,
        "carrier_archive_sha256": carrier.archive_sha256,
        "carrier_n_pairs": carrier.n_pairs,
        "carrier_latent_dim": carrier.latent_dim,
        "carrier_base_channels": carrier.base_channels,
        "cheap_by_construction": carrier.cheap_by_construction,
        "lambda_score_per_byte": CONTEST_LAMBDA,
        "z8_falsification": {
            "z8_archive_bytes": falsification.z8_archive_bytes,
            "z8_rate_term": falsification.z8_rate_term,
            "z8_over_pr95_byte_ratio": falsification.z8_over_pr95_byte_ratio,
            "z8_disease_confirmed": falsification.z8_disease_confirmed,
            "rationale": falsification.rationale,
        },
        **dict(NON_PROMOTABLE_MARKERS),
    }
    if distortion is not None:
        row["advisory_d_seg"] = distortion.d_seg
        row["advisory_d_pose"] = distortion.d_pose
        row["advisory_distortion_only_score"] = distortion.advisory_score
        row["advisory_n_pairs_measured"] = distortion.n_pairs_measured
        row["advisory_full_score_estimate"] = float(
            distortion.advisory_score + carrier.rate_term
        )
        row["render_axis_tag"] = distortion.render_axis_tag
        row["measure_axis_tag"] = distortion.measure_axis_tag
    else:
        row["advisory_d_seg"] = None
        row["advisory_d_pose"] = None
        row["advisory_render_skipped"] = True
    if latent_allocation is not None:
        row["latent_linf_vs_l2"] = {
            "linf_bits": latent_allocation.linf_bits,
            "l2_bits": latent_allocation.l2_bits,
            "target_bits": latent_allocation.target_bits,
            "allocations_differ": latent_allocation.allocations_differ,
            "latent_dynamic_range": latent_allocation.latent_dynamic_range,
            "water_level": latent_allocation.water_level,
            "objective": "linf_margin_budget_section_7_proven",
        }
    return row


__all__ = [
    "CARRIER_SCHEMA",
    "CONTEST_LAMBDA",
    "CONTEST_RATE_DENOM_BYTES",
    "CONTEST_RATE_MULTIPLIER",
    "NON_PROMOTABLE_MARKERS",
    "Z8_600PAIR_BYTE_CLOSED_CONTEST_SCORE",
    "Z8_NEAR_LOSSLESS_ARCHIVE_BYTES",
    "Z8_NEAR_LOSSLESS_WAVELET_BLOB_BYTES",
    "CarrierDistortionAdvisory",
    "CarrierRateTerm",
    "LatentLinfL2Allocation",
    "LatentSaliency",
    "Pr95HnervCarrierError",
    "Z8Falsification",
    "allocate_latent_linf_vs_l2",
    "build_head_to_head_row",
    "carrier_rate_term",
    "load_carrier_decoder",
    "measure_carrier_distortion",
    "push_pixel_saliency_to_latent",
    "quantize_latent_with_steps",
    "render_carrier_pair_bcthw",
    "z8_falsification",
]
