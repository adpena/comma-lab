# SPDX-License-Identifier: MIT
"""Canonical score-exact per-pixel saliency producer (P18 + P19).

FOUNDATION reverse-engineering producer for the comma video compression
challenge. This module is the canonical UPSTREAM producer of the two
score-exact per-pixel saliency surfaces that the rate-axis allocators
(``tac.optimization.joint_p18_p19_waterfill``) consume:

  * **P18** — ``compute_s_seg_flip_risk``: DeepFool top-2-margin flip-risk on
    the LAST frame of the pair (SegNet scores only ``x[:, -1, ...]``). This is
    the score-exact surrogate for the contest's hard ``100 * d_seg`` term where
    ``d_seg = mean(argmax(out1) != argmax(out2))`` (per
    ``upstream/modules.py::SegNet.compute_distortion``).
  * **P19** — ``compute_s_pose_fisher``: squared input-Jacobian of the first-6
    pose dims over BOTH frames of the pair (PoseNet scores the pair via 12-ch
    YUV6; ``d_pose = MSE`` over the first 6 of 12 pose-head dims, per
    ``upstream/modules.py::PoseNet.compute_distortion``). Pose gradients require
    the differentiable ``rgb_to_yuv6`` mirror because upstream's is
    ``@torch.no_grad`` / in-place-clamp.

Why this module exists (CLAUDE.md "Results must become system intelligence"):
the verified mirror logic in ``tools/verify_upstream_scorer_mirror_fidelity.py``
(commit 8173b493a; bit-exact vs real frozen weights) proved the s_seg/s_pose
surfaces are gradient-reachable. That harness stays as the FIDELITY VERIFIER;
this module CONSOLIDATES its ``_section_s_seg`` / ``_section_s_pose`` logic into
a reusable, profiled, optimized producer with a ``saliency_concentration``
analyzer and provenance per
``tac.contest_eval_contract.build_saliency_verification_contract``.

All results are ``[macOS-CPU advisory]`` — NON-PROMOTABLE (no score claims, no
MPS authority per CLAUDE.md "MPS auth eval is NOISE"). The producer forces CPU
and tags every output ``score_claim=false`` / ``promotable=false`` per
Catalog #341 / #192 / #323.

CONTEST COMPLIANCE: this is a COMPRESS-SIDE analysis producer. It loads frozen
upstream scorers for OFFLINE saliency derivation only; the emitted artifacts
NEVER cross the receiver boundary (only ``archive.zip`` + the scorer-free
deterministic ``inflate`` runtime do), per
``build_saliency_verification_contract`` ``contest_compliance``.

Example::

    from tac.analysis.score_exact_saliency import (
        load_score_exact_scorers,
        compute_s_seg_flip_risk,
        compute_s_pose_fisher,
        saliency_concentration,
    )
    posenet, segnet = load_score_exact_scorers(upstream_dir="upstream")
    s_seg = compute_s_seg_flip_risk(segnet, pair_btchw)      # (H, W) flip-risk
    s_pose = compute_s_pose_fisher(posenet, pair_btchw)      # (H, W) Fisher info
    conc_seg = saliency_concentration(s_seg.flip_risk)       # Gini / top-k% / ratio

References:
    - Moosavi-Dezfooli, Fawzi, Frossard 2016 "DeepFool" (multiclass top-2 margin).
    - Fisher information of a regression output under iid Gaussian pixel noise:
      diag(J^T J) where J = d(pose_1..6)/d(x_i).
    - tac.contest_eval_contract.build_saliency_verification_contract (the 6
      required_numerical_proofs this producer must satisfy).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Canonical scorer loading + real-frame decode (consolidated from the verified
# mirror harness tools/verify_upstream_scorer_mirror_fidelity.py).
# ---------------------------------------------------------------------------


def _ensure_src_on_path() -> None:
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_score_exact_scorers(
    upstream_dir: str | Path = "upstream",
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load frozen upstream PoseNet+SegNet and make them differentiable.

    Consolidates ``_build_upstream_distortion_net`` from the verified mirror
    harness + ``tac.scorer.make_scorers_differentiable`` (the canonical
    differentiable ``rgb_to_yuv6`` + AllNorm reshape patch). The PoseNet preprocess
    becomes gradient-reachable so the P19 Fisher surface is real (not severed).

    Returns ``(posenet, segnet)`` with all params frozen (``requires_grad=False``).
    The fidelity of this mirror vs the real frozen weights is verified by
    ``tools/verify_upstream_scorer_mirror_fidelity.py`` (the SHA-256 + max-abs
    proofs); this loader does NOT re-verify — it consumes the verified contract.
    """
    device = torch.device(device)
    upstream = Path(upstream_dir)
    if not upstream.is_absolute():
        upstream = REPO_ROOT / upstream
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    _ensure_src_on_path()
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore

    dn = DistortionNet().eval().to(device)
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in dn.parameters():
        p.requires_grad = False

    from tac.scorer import make_scorers_differentiable

    make_scorers_differentiable(dn.posenet, dn.segnet)
    return dn.posenet, dn.segnet


def decode_real_pairs(
    video_path: str | Path,
    num_pairs: int,
    *,
    pair_stride: int = 1,
    start_pair: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Decode ``num_pairs`` real frame-pairs from ``upstream/videos/0.mkv``.

    Uses the SAME ``frame_utils.yuv420_to_rgb`` decode path as upstream so the
    pixel values are byte-identical to what the contest scorer sees on CPU
    (per Catalog #213 — NEVER synthetic frames in a non-smoke path).

    ``pair_stride`` lets a diagnostic stratify the 600-pair video (e.g. every
    20th pair) for a representative subset without decoding all 1200 frames.

    Returns a ``(num_pairs, 2, 3, H, W)`` float CHW tensor — exactly the layout
    after the upstream ``DistortionNet.preprocess_input`` rearrange, which is
    what both ``PoseNet.preprocess_input`` and ``SegNet.preprocess_input``
    receive.
    """
    device = torch.device(device)
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    upstream = REPO_ROOT / "upstream"
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import av  # type: ignore
    from frame_utils import yuv420_to_rgb  # type: ignore

    # Frames we must touch: cover start..(start + (num_pairs-1)*stride) pairs.
    last_pair_index = start_pair + (num_pairs - 1) * pair_stride
    frames_needed = (last_pair_index + 1) * 2

    # SINGLE sequential decode pass. We keep ONLY the frames the requested pairs
    # need (the 2 frames of each pair); everything between strided pairs is
    # decoded-and-dropped to advance the stream, so RAM holds <= num_pairs*2
    # native frames at once (RAM hygiene: native 874x1164x3 is ~3 MB/frame).
    wanted_frame_idx: set[int] = set()
    for i in range(num_pairs):
        p = start_pair + i * pair_stride
        wanted_frame_idx.add(2 * p)
        wanted_frame_idx.add(2 * p + 1)

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    kept: dict[int, torch.Tensor] = {}
    idx = 0
    for frame in container.decode(stream):
        if idx in wanted_frame_idx:
            kept[idx] = yuv420_to_rgb(frame)  # (H, W, 3) uint8
        idx += 1
        if idx >= frames_needed:
            break
    container.close()
    if len(kept) < len(wanted_frame_idx):
        raise RuntimeError(
            f"decoded only {idx} frames from {video_path}, needed {frames_needed} "
            f"for {num_pairs} pairs at stride {pair_stride} from start_pair {start_pair}"
        )

    pair_tensors: list[torch.Tensor] = []
    for i in range(num_pairs):
        p = start_pair + i * pair_stride
        pair = torch.stack([kept[2 * p], kept[2 * p + 1]])  # (2, H, W, 3)
        pair_tensors.append(pair)
    pairs = torch.stack(pair_tensors)  # (num_pairs, 2, H, W, 3) uint8
    pairs = pairs.permute(0, 1, 4, 2, 3).float().contiguous().to(device)
    return pairs  # (num_pairs, 2, 3, H, W)


def stream_real_pairs(
    video_path: str | Path,
    *,
    num_pairs: int,
    pair_stride: int = 1,
    start_pair: int = 0,
    device: torch.device | str = "cpu",
):
    """Yield ``(1, 2, 3, H, W)`` pairs in a SINGLE sequential decode pass.

    Unlike repeated ``decode_real_pairs`` calls (which each re-decode from frame
    0 — O(N^2) for strided sampling), this generator opens the container ONCE,
    advances the decoder linearly, and yields each requested pair as it is
    reached. RAM holds at most one pair's two native frames at a time. This is
    the canonical full-video iteration path for the dead-zone diagnostic.
    """
    device = torch.device(device)
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    upstream = REPO_ROOT / "upstream"
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import av  # type: ignore
    from frame_utils import yuv420_to_rgb  # type: ignore

    # Map: frame index -> (pair_position, which_frame_in_pair).
    frame_to_pair: dict[int, tuple[int, int]] = {}
    for i in range(num_pairs):
        p = start_pair + i * pair_stride
        frame_to_pair[2 * p] = (i, 0)
        frame_to_pair[2 * p + 1] = (i, 1)
    last_frame = max(frame_to_pair) if frame_to_pair else -1

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    buffers: dict[int, list] = {}  # pair_position -> [f0, f1]
    idx = 0
    try:
        for frame in container.decode(stream):
            if idx in frame_to_pair:
                pos, which = frame_to_pair[idx]
                slot = buffers.setdefault(pos, [None, None])
                slot[which] = yuv420_to_rgb(frame)  # (H, W, 3) uint8
                if slot[0] is not None and slot[1] is not None:
                    pair = torch.stack([slot[0], slot[1]])  # (2, H, W, 3)
                    pair = (
                        pair.permute(0, 3, 1, 2).float().unsqueeze(0).contiguous().to(device)
                    )  # (1, 2, 3, H, W)
                    del buffers[pos]
                    yield pair
            idx += 1
            if idx > last_frame:
                break
    finally:
        container.close()


# ---------------------------------------------------------------------------
# P18: SegNet flip-risk saliency (DeepFool top-2 margin, LAST frame only).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegFlipRisk:
    """P18 score-exact flip-risk surface for ONE pair (last frame).

    ``flip_risk`` ~ DeepFool flip-risk = ``||grad margin||^2 / margin^2`` (high
    where the top-2 logit margin is small AND the input gradient is large — the
    pixels where a small perturbation flips the argmax and moves ``d_seg``).
    """

    flip_risk: torch.Tensor  # (H, W) at scorer resolution 384x512
    grad_energy: torch.Tensor  # (H, W) ||grad margin||^2 (numerator)
    margin: torch.Tensor  # (H, W) top-2 logit margin (>= 0)
    grad_finite: bool
    grad_nonzero_frac: float
    scorer_input_hw: tuple[int, int]


def compute_s_seg_flip_risk(
    segnet: torch.nn.Module, pair_btchw: torch.Tensor
) -> SegFlipRisk:
    """P18: per-pixel SegNet flip-risk on the LAST frame of one pair.

    SegNet scores only ``x[:, -1, ...]`` (the frame_1 of the pair). We backprop
    the SUMMED top-2 margin once (a single backward gives the per-input-pixel
    gradient field), then form the DeepFool flip-risk ``grad_energy / margin^2``.

    ``pair_btchw`` must be ``(1, 2, 3, H, W)`` or ``(2, 3, H, W)`` (one pair).
    The gradient is taken wrt the SegNet preprocess output (the resized last
    frame at 384x512), so the surface is at scorer resolution.
    """
    pair = _as_single_pair(pair_btchw)  # (1, 2, 3, H, W)
    seg_in = segnet.preprocess_input(pair)  # (1, 3, 384, 512) — last frame resized
    seg_in = seg_in.detach().clone().requires_grad_(True)
    logits = segnet(seg_in)  # (1, 5, 384, 512)
    top2 = logits.topk(2, dim=1).values  # (1, 2, H, W)
    margin = top2[:, 0] - top2[:, 1]  # (1, H, W) winning-minus-runnerup, >= 0
    summed = margin.sum()
    grad = torch.autograd.grad(summed, seg_in, retain_graph=False)[0]  # (1,3,H,W)
    grad_energy = grad.pow(2).sum(dim=1)  # (1, H, W) sum over RGB channels
    flip_risk = grad_energy / (margin.detach().pow(2) + 1e-6)  # (1, H, W)

    finite = bool(torch.isfinite(grad).all().item()) and bool(
        torch.isfinite(flip_risk).all().item()
    )
    return SegFlipRisk(
        flip_risk=flip_risk.detach().squeeze(0),  # (H, W)
        grad_energy=grad_energy.detach().squeeze(0),
        margin=margin.detach().squeeze(0),
        grad_finite=finite,
        grad_nonzero_frac=float((grad.abs() > 0).float().mean().item()),
        scorer_input_hw=tuple(seg_in.shape[-2:]),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# P19: PoseNet Fisher saliency (squared input-Jacobian of first-6 pose dims).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseFisher:
    """P19 score-exact Fisher-information surface for ONE pair (both frames)."""

    s_pose: torch.Tensor  # (H, W) per-pixel Fisher info (summed over 2 frames + RGB)
    s_pose_per_frame: torch.Tensor  # (2, H, W) per-frame Fisher info
    grad_finite: bool
    s_pose_nonzero_frac: float
    method: str  # "loop" | "batched_vjp"
    scorer_input_hw: tuple[int, int]


def compute_s_pose_fisher(
    posenet: torch.nn.Module,
    pair_btchw: torch.Tensor,
    *,
    method: str = "batched_vjp",
) -> PoseFisher:
    """P19: per-pixel PoseNet Fisher info over BOTH frames of one pair.

    Under iid Gaussian pixel noise the per-pixel Fisher information of the first
    6 pose dims is ``sum_{k<6} (d pose_k / d x_i)^2`` = diag(J^T J). We compute
    it wrt the camera-native input (both frames), so the surface lives at native
    resolution after the upstream preprocess chain (resize + YUV6) is
    backpropagated through the differentiable mirror.

    ``method``:
      * ``"loop"`` — baseline: 6 separate backward passes (one per pose dim).
        This is the original harness implementation. Kept for equivalence proof.
      * ``"batched_vjp"`` — OPTIMIZED: a SINGLE ``is_grads_batched=True`` backward
        with a 6-row identity ``grad_outputs``. PyTorch vectorizes the 6 VJPs
        over the batch dimension (vmap under the hood), reusing the forward graph
        once instead of re-traversing it 6x. Numerically identical to ``loop``.
    """
    pair = _as_single_pair(pair_btchw)  # (1, 2, 3, H, W)
    if method == "loop":
        return _pose_fisher_loop(posenet, pair)
    if method == "batched_vjp":
        return _pose_fisher_batched_vjp(posenet, pair)
    raise ValueError(f"unknown method {method!r}; use 'loop' or 'batched_vjp'")


def _pose_fisher_loop(posenet: torch.nn.Module, pair: torch.Tensor) -> PoseFisher:
    """Baseline: 6 separate backward passes (one per pose scalar)."""
    pair = pair.detach().clone().requires_grad_(True)  # (1, 2, 3, H, W)
    pose_in = posenet.preprocess_input(pair)  # (1, 12, 192, 256)
    pose_out = posenet(pose_in)["pose"]  # (1, 12)
    s_pose = torch.zeros_like(pair)
    grads_finite = True
    for k in range(6):
        g = torch.autograd.grad(
            pose_out[0, k], pair, retain_graph=(k < 5), create_graph=False
        )[0]
        if not torch.isfinite(g).all():
            grads_finite = False
        s_pose = s_pose + g.pow(2)
    return _finalize_pose_fisher(s_pose, pair, grads_finite, "loop")


def _pose_fisher_batched_vjp(posenet: torch.nn.Module, pair: torch.Tensor) -> PoseFisher:
    """OPTIMIZED: single is_grads_batched=True backward over a 6-row identity.

    For the 6-output, single-input case this computes the full input-Jacobian of
    the first 6 pose dims in ONE backward. ``grad_outputs`` is a
    ``(6, 12)`` matrix whose row k is e_k (the k-th standard basis vector); with
    ``is_grads_batched=True`` PyTorch returns the stacked input gradients
    ``(6, 1, 2, 3, H, W)`` = J^T applied to each e_k = the rows of the input
    Jacobian. Summing the squares over the 6 rows gives diag(J^T J) — exactly the
    Fisher surface, numerically identical to the loop but ~3-6x faster because
    the forward graph is traversed once (vmap) instead of 6x.
    """
    pair = pair.detach().clone().requires_grad_(True)  # (1, 2, 3, H, W)
    pose_in = posenet.preprocess_input(pair)  # (1, 12, 192, 256)
    pose_out = posenet(pose_in)["pose"]  # (1, 12)
    pose6 = pose_out[0, :6]  # (6,)
    # 6-row identity: row k selects pose dim k.
    eye6 = torch.eye(6, dtype=pose6.dtype, device=pose6.device)  # (6, 6)
    # is_grads_batched vectorizes the backward over the leading dim of grad_outputs.
    jac_rows = torch.autograd.grad(
        outputs=pose6,
        inputs=pair,
        grad_outputs=eye6,  # (6, 6) — batched: row k = e_k
        retain_graph=False,
        create_graph=False,
        is_grads_batched=True,
    )[0]  # (6, 1, 2, 3, H, W)
    grads_finite = bool(torch.isfinite(jac_rows).all().item())
    # diag(J^T J)_i = sum_k (J_ki)^2 = sum over the 6 batched rows of the square.
    s_pose = jac_rows.pow(2).sum(dim=0)  # (1, 2, 3, H, W)
    return _finalize_pose_fisher(s_pose, pair, grads_finite, "batched_vjp")


def _finalize_pose_fisher(
    s_pose: torch.Tensor, pair: torch.Tensor, grads_finite: bool, method: str
) -> PoseFisher:
    # s_pose: (1, 2, 3, H, W) — collapse RGB -> per-frame, and frames -> per-pixel.
    s_pose_per_frame = s_pose.detach().squeeze(0).sum(dim=1)  # (2, H, W)
    s_pose_per_pixel = s_pose_per_frame.sum(dim=0)  # (H, W)
    finite = grads_finite and bool(torch.isfinite(s_pose_per_pixel).all().item())
    return PoseFisher(
        s_pose=s_pose_per_pixel,
        s_pose_per_frame=s_pose_per_frame,
        grad_finite=finite,
        s_pose_nonzero_frac=float((s_pose_per_pixel > 0).float().mean().item()),
        method=method,
        scorer_input_hw=tuple(pair.shape[-2:]),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Concentration analyzer (Lorenz/Gini + top-k% mass + boundary/interior ratio).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaliencyConcentration:
    """Concentration metrics for one per-pixel saliency surface.

    ``gini`` -> 1 means a single pixel holds all mass (maximally concentrated);
    ``gini`` -> 0 means uniform. ``top_k_pct_mass`` maps each k% to the fraction
    of total saliency mass held by the top-k% of pixels.
    """

    gini: float
    top_k_pct_mass: dict[float, float]  # {1.0: frac, 5.0: frac, 10.0: frac}
    boundary_over_interior_ratio: float  # decile-based, optional
    num_pixels: int
    total_mass: float
    nonzero_frac: float


def saliency_concentration(
    saliency_map: torch.Tensor,
    *,
    margin: torch.Tensor | None = None,
    top_k_pcts: tuple[float, ...] = (1.0, 5.0, 10.0),
) -> SaliencyConcentration:
    """Lorenz/Gini + top-k% mass fraction + boundary/interior ratio.

    ``saliency_map`` is any nonnegative per-pixel surface (s_seg flip-risk or
    s_pose Fisher). ``margin`` (optional, only meaningful for s_seg) defines the
    boundary set = lowest-decile margin pixels; if omitted, the boundary ratio
    is computed against the top-decile-vs-bottom-decile of the saliency itself.

    The Gini coefficient is computed from the sorted Lorenz curve:
        G = (2 * sum_i(i * x_i) / (n * sum_i x_i)) - (n + 1) / n
    where x_i is the ascending-sorted saliency. This is the standard
    population Gini, bounded [0, 1] for nonnegative inputs.
    """
    flat = saliency_map.detach().reshape(-1).double()
    flat = flat.clamp_min(0.0)  # saliency surfaces are nonnegative by construction
    n = flat.numel()
    total = float(flat.sum().item())
    nonzero_frac = float((flat > 0).double().mean().item())

    # Gini from the ascending Lorenz curve.
    if total <= 0.0 or n == 0:
        gini = 0.0
    else:
        sorted_vals, _ = torch.sort(flat)
        idx = torch.arange(1, n + 1, dtype=torch.float64, device=flat.device)
        gini = float(
            (2.0 * (idx * sorted_vals).sum() / (n * sorted_vals.sum())).item()
            - (n + 1.0) / n
        )
        gini = max(0.0, min(1.0, gini))

    # Top-k% mass fraction.
    top_k_pct_mass: dict[float, float] = {}
    if total > 0.0 and n > 0:
        sorted_desc, _ = torch.sort(flat, descending=True)
        csum = torch.cumsum(sorted_desc, dim=0)
        for pct in top_k_pcts:
            k = max(1, round(pct / 100.0 * n))
            k = min(k, n)
            top_k_pct_mass[float(pct)] = float((csum[k - 1] / total).item())
    else:
        for pct in top_k_pcts:
            top_k_pct_mass[float(pct)] = 0.0

    # Boundary/interior ratio.
    if margin is not None:
        m = margin.detach().reshape(-1)
        q10 = torch.quantile(m.float(), 0.10)
        q90 = torch.quantile(m.float(), 0.90)
        bmask = (margin.detach().reshape(-1) <= q10)
        imask = (margin.detach().reshape(-1) > q90)
        fr = saliency_map.detach().reshape(-1)
        b_energy = float(fr[bmask].mean().item()) if bool(bmask.any()) else 0.0
        i_energy = float(fr[imask].mean().item()) if bool(imask.any()) else 0.0
    else:
        fr = saliency_map.detach().reshape(-1).float()
        q10 = torch.quantile(fr, 0.10)
        q90 = torch.quantile(fr, 0.90)
        b_energy = float(fr[fr >= q90].mean().item()) if bool((fr >= q90).any()) else 0.0
        i_energy = float(fr[fr <= q10].mean().item()) if bool((fr <= q10).any()) else 0.0
    ratio = float(b_energy / (i_energy + 1e-12))

    return SaliencyConcentration(
        gini=gini,
        top_k_pct_mass=top_k_pct_mass,
        boundary_over_interior_ratio=ratio,
        num_pixels=int(n),
        total_mass=total,
        nonzero_frac=nonzero_frac,
    )


# ---------------------------------------------------------------------------
# Provenance (per build_saliency_verification_contract).
# ---------------------------------------------------------------------------


def build_producer_provenance(
    *,
    upstream_dir: str | Path = "upstream",
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Provenance dict referencing upstream source + mirror SHA-256s + 6 proofs.

    Threads the canonical ``build_saliency_verification_contract`` so every
    emitted artifact names the upstream source SHA-256s, the mirror
    implementation paths, and the 6 ``required_numerical_proofs`` that
    ``tools/verify_upstream_scorer_mirror_fidelity.py`` establishes. The score
    custody is fail-closed: ``score_claim=false``, ``promotable=false``,
    ``axis_tag="[macOS-CPU advisory]"`` per Catalog #341 / #323.
    """
    _ensure_src_on_path()
    from tac.contest_eval_contract import (
        build_saliency_verification_contract,
        build_upstream_eval_contract,
    )

    eval_contract = build_upstream_eval_contract(
        repo_root=repo_root, upstream_dir=str(Path(upstream_dir).name)
        if not Path(upstream_dir).is_absolute()
        else upstream_dir,
    )
    saliency_contract = build_saliency_verification_contract()
    return {
        "schema": "score_exact_saliency_producer_provenance.v1",
        "producer_module": "tac.analysis.score_exact_saliency",
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "upstream_source_custody": eval_contract["source_custody"],
        "upstream_model_custody": eval_contract["model_custody"],
        "upstream_contract_valid": eval_contract["contract_valid"],
        "mirror_paths": {
            "differentiable_yuv6": "tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6",
            "scorer_patch": "tac.scorer.make_scorers_differentiable",
            "fidelity_verifier": "tools/verify_upstream_scorer_mirror_fidelity.py",
        },
        "required_numerical_proofs": saliency_contract["required_numerical_proofs"],
        "contest_compliance": saliency_contract["contest_compliance"],
        "budget_spend_requires": saliency_contract["budget_spend_requires"],
        "p18_method": "DeepFool top-2 margin flip-risk on LAST frame (||grad m||^2 / m^2)",
        "p19_method": "squared input-Jacobian of first-6 pose dims over BOTH frames (diag J^T J)",
    }


# ---------------------------------------------------------------------------
# Profiling helper.
# ---------------------------------------------------------------------------


@dataclass
class ProducerProfile:
    """Per-pair wall-clock for s_seg and s_pose, with projection to full video."""

    s_seg_seconds_per_pair: float
    s_pose_seconds_per_pair: float
    s_pose_method: str
    num_pairs_measured: int
    total_pairs_in_video: int = 600
    per_pair_breakdown: list[dict[str, float]] = field(default_factory=list)

    @property
    def total_seconds_per_pair(self) -> float:
        return self.s_seg_seconds_per_pair + self.s_pose_seconds_per_pair

    @property
    def projected_full_video_seconds(self) -> float:
        return self.total_seconds_per_pair * self.total_pairs_in_video

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "s_seg_seconds_per_pair": self.s_seg_seconds_per_pair,
            "s_pose_seconds_per_pair": self.s_pose_seconds_per_pair,
            "s_pose_method": self.s_pose_method,
            "total_seconds_per_pair": self.total_seconds_per_pair,
            "num_pairs_measured": self.num_pairs_measured,
            "total_pairs_in_video": self.total_pairs_in_video,
            "projected_full_video_seconds": self.projected_full_video_seconds,
            "projected_full_video_minutes": self.projected_full_video_seconds / 60.0,
            "per_pair_breakdown": self.per_pair_breakdown,
        }


def profile_producer(
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    pairs_btchw: torch.Tensor,
    *,
    s_pose_method: str = "batched_vjp",
    total_pairs_in_video: int = 600,
) -> ProducerProfile:
    """Measure per-pair wall-clock for s_seg and s_pose over the given pairs."""
    num_pairs = pairs_btchw.shape[0]
    seg_times: list[float] = []
    pose_times: list[float] = []
    breakdown: list[dict[str, float]] = []
    for i in range(num_pairs):
        pair = pairs_btchw[i : i + 1]
        t0 = time.perf_counter()
        compute_s_seg_flip_risk(segnet, pair)
        t1 = time.perf_counter()
        compute_s_pose_fisher(posenet, pair, method=s_pose_method)
        t2 = time.perf_counter()
        seg_times.append(t1 - t0)
        pose_times.append(t2 - t1)
        breakdown.append({"pair_index": float(i), "s_seg_s": t1 - t0, "s_pose_s": t2 - t1})
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return ProducerProfile(
        s_seg_seconds_per_pair=mean(seg_times),
        s_pose_seconds_per_pair=mean(pose_times),
        s_pose_method=s_pose_method,
        num_pairs_measured=num_pairs,
        total_pairs_in_video=total_pairs_in_video,
        per_pair_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _as_single_pair(pair_btchw: torch.Tensor) -> torch.Tensor:
    """Coerce input to a single-pair (1, 2, 3, H, W) tensor."""
    if pair_btchw.dim() == 4:  # (2, 3, H, W)
        return pair_btchw.unsqueeze(0)
    if pair_btchw.dim() == 5:  # (P, 2, 3, H, W)
        if pair_btchw.shape[0] != 1:
            return pair_btchw[:1]
        return pair_btchw
    raise ValueError(
        f"expected (2,3,H,W) or (P,2,3,H,W) pair tensor, got shape {tuple(pair_btchw.shape)}"
    )


__all__ = [
    "PoseFisher",
    "ProducerProfile",
    "SaliencyConcentration",
    "SegFlipRisk",
    "build_producer_provenance",
    "compute_s_pose_fisher",
    "compute_s_seg_flip_risk",
    "decode_real_pairs",
    "load_score_exact_scorers",
    "profile_producer",
    "saliency_concentration",
    "stream_real_pairs",
]
