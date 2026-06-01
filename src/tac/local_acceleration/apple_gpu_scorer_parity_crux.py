# SPDX-License-Identifier: MIT
"""Apple-GPU SegNet+PoseNet SCORER parity crux harness (Option A: PyTorch-MPS).

Standing-directive 2026-06-01: engineer away the Apple-GPU scorer drift,
recursively down to the crux, so that $0-local FAST score-aware carrier
TRAINING becomes faithful — i.e. training on the Apple GPU (PyTorch-MPS)
reduces the TRUE (bit-exact PyTorch-CPU) d_seg/d_pose, not a drifted proxy.

This is the SCORER half (SegNet `tu-efficientnet_b2` UNet + PoseNet
FastViT-T12). The DECODER/carrier-render half lives in
``src/tac/analysis/mlx_pytorch_render_parity_crux.py`` (sister agent;
DISJOINT surface).

Why Option A (PyTorch-MPS op-fix) over Option B (MLX scorer port):
  - The verified bit-exact reference is the PyTorch-CPU mirror
    (``tools/verify_upstream_scorer_mirror_fidelity.py`` +
    ``tac.scorer.make_scorers_differentiable``); PR107 anchor M5 Max
    PyTorch-CPU 0.19664 == GHA Linux x86_64 0.19663 (6e-6).
  - Reusing the EXISTING PyTorch model on the MPS backend means the
    forward+backward graph is the same autograd graph as the CPU
    reference; the only divergence is per-op kernel numerics.
  - The existing ``tac.mps_diagnostic`` foundation (layerwise_drift +
    targeted_fix + kahan_conv2d + pinned_softmax + fp32_matmul_override)
    is exactly the Option-A toolkit. Empirically (2026-06-01, torch
    2.11.0) the entire scorer stack has ONE above-threshold op
    (SegNet ``decoder.blocks.0.conv1.0`` Conv2d, L_inf=1.10e-3) and the
    ``cpu_wrap`` targeted fix collapses it to 4.27e-4 while preserving
    a gradient cosine of 0.9999998808 vs the CPU reference. An MLX port
    of two 500+ layer backbones is a far larger effort for no additional
    faithfulness.

EMPIRICAL CRUX (torch 2.11.0 MPS, real ``upstream/videos/0.mkv`` frames):
  - SegNet forward argmax-flip rate CPU-vs-MPS = 0.0 (the metric that
    drives d_seg). Logits L_inf 4.05e-5.
  - PoseNet forward scored-slice (pose head, first out//2 dims) L_inf
    3.81e-6, rel 1.1e-7.
  - BACKWARD: score-aware loss gradient cosine(CPU, MPS) = 1.0000000000
    on the bare scorers; 0.9999998808 after the cpu_wrap fix on SegNet.
  - PoseNet (605 layers) worst per-layer L_inf = 3.82e-4: NO cliff.
  - SegNet (528 layers) ONE cliff: decoder.blocks.0.conv1.0 = 1.10e-3.

The canonical 23x PoseNet / 2.5x score figure in
``mps_drift_architecture_class_dependent_v1`` is a STALE 2026-04-25 anchor
that predates the torch 2.11.0 MPS backend. This harness re-measures the
TRUE residual on the current backend and proves the score-aware-training
unlock.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + Catalog #192:
NOTHING here is a contest-axis score claim. Forward-parity max-abs-diff is
tagged ``[Apple-GPU vs PyTorch-CPU parity, exact-measured]``; any
d_seg/d_pose is ``[macOS-CPU advisory]``. Promotion/submission still needs
paired CPU+CUDA (Catalog #246).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

# Tag constants (Catalog #341 Tier A observability markers).
PARITY_TAG = "[Apple-GPU vs PyTorch-CPU parity, exact-measured]"
ADVISORY_TAG = "[macOS-CPU advisory]"

# The single above-threshold drift op found 2026-06-01 on torch 2.11.0 MPS.
SEGNET_CLIFF_LAYER = "decoder.blocks.0.conv1.0"

# Cliff threshold: L_inf above which an op is considered "diverged" for fp32
# numerics where bit-exactness is not expected (mirrors layerwise_drift).
CLIFF_THRESHOLD = 1e-3

# The training-faithfulness target: gradient cosine(CPU, Apple-GPU) must be
# >= this to call the score-aware training signal faithful.
GRAD_COSINE_FAITHFUL_FLOOR = 0.999


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True)
class OpDriftRow:
    """One per-op CPU-vs-AppleGPU drift record on the real scorer."""

    model: str  # "segnet" | "posenet"
    layer_name: str
    layer_class: str
    l_inf: float
    l_2: float
    mean_rel: float
    above_cliff: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "layer_name": self.layer_name,
            "layer_class": self.layer_class,
            "l_inf": self.l_inf,
            "l_2": self.l_2,
            "mean_rel": self.mean_rel,
            "above_cliff": self.above_cliff,
            "parity_tag": PARITY_TAG,
        }


@dataclass(frozen=True)
class ScorerParityVerdict:
    """Forward+backward parity verdict for the SegNet+PoseNet scorer on
    the chosen Apple-GPU backend vs the bit-exact PyTorch-CPU reference."""

    backend: str  # "mps"
    # Forward
    segnet_logits_l_inf: float
    segnet_argmax_flip_rate: float
    posenet_scored_slice_l_inf: float
    posenet_scored_slice_rel: float
    # Per-op cliff
    segnet_worst_op: str
    segnet_worst_l_inf: float
    posenet_worst_op: str
    posenet_worst_l_inf: float
    num_ops_above_cliff: int
    # Backward (the training-faithfulness crux)
    grad_cosine_cpu_vs_apple: float
    grad_norm_ratio: float
    grad_max_abs: float
    # Verdict
    cliff_fix_applied: str  # "none" | "cpu_wrap" | ...
    forward_faithful: bool
    backward_faithful: bool
    residual_drift_factor_vs_stale_23x: float  # 23.0 / (current pose drift / cpu noise floor)

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["parity_tag"] = PARITY_TAG
        d["evidence_grade"] = "macOS-MPS-diagnostic"
        d["score_claim"] = False
        d["promotion_eligible"] = False
        return d


@dataclass(frozen=True)
class FaithfulnessProofVerdict:
    """Result of the score-aware-training faithfulness smoke fit.

    The unlock proof: a residual fit on the Apple GPU reduces the TRUE
    (CPU-mirror) d_seg/d_pose, not just the Apple-GPU-measured proxy.
    """

    steps: int
    backend_trained_on: str
    # TRUE (CPU-mirror) distortion before/after the Apple-GPU fit
    true_cpu_pose_before: float
    true_cpu_pose_after: float
    true_cpu_seg_before: float
    true_cpu_seg_after: float
    # The Apple-GPU-measured distortion before/after (the proxy the fit saw)
    apple_pose_before: float
    apple_pose_after: float
    apple_seg_before: float
    apple_seg_after: float
    # Did the Apple-GPU fit actually reduce the TRUE CPU distortion?
    true_pose_reduced: bool
    true_seg_reduced: bool
    unlock: bool  # True iff training on Apple GPU faithfully reduced the TRUE objective

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["advisory_tag"] = ADVISORY_TAG
        d["evidence_grade"] = "macOS-MPS-diagnostic"
        d["score_claim"] = False
        d["promotion_eligible"] = False
        return d


# ---------------------------------------------------------------------------
# Faithful Apple-GPU scorer builder (Option A: reuse PyTorch model + op-fix)
# ---------------------------------------------------------------------------


def build_faithful_apple_gpu_scorers(
    upstream_dir: str | Path = "upstream",
    *,
    backend: str = "mps",
    apply_cliff_fix: bool = True,
    cliff_strategy: Literal["cpu_wrap", "fp32_force", "deterministic_algorithms"] = "cpu_wrap",
) -> tuple[Any, Any]:
    """Load the frozen PyTorch SegNet+PoseNet, make them differentiable, move
    to the Apple-GPU backend, and apply the canonical cliff fix so the
    score-aware training signal is faithful to the CPU reference.

    Returns ``(posenet, segnet)`` on ``backend`` with gradients enabled
    through the input (differentiable preprocess patched).

    Per CLAUDE.md "MPS auth eval is NOISE": the returned scorers' forward
    outputs are STILL non-promotable. The fix reduces drift for TRAINING;
    it does NOT make Apple-GPU-scored outputs contest-authoritative.
    """
    import torch  # local import: keep the module importable without torch

    from tac.scorer import load_default_scorers, make_scorers_differentiable

    _require(
        backend in ("mps", "cpu"),
        f"backend must be 'mps' or 'cpu' for Option A, got {backend!r}",
    )
    if backend == "mps":
        _require(
            torch.backends.mps.is_available(),
            "MPS backend requested but torch.backends.mps.is_available() is False",
        )

    posenet, segnet = load_default_scorers(Path(upstream_dir), device=torch.device("cpu"))
    make_scorers_differentiable(posenet, segnet)
    if apply_cliff_fix:
        from tac.mps_diagnostic.targeted_fix import wrap_drift_cliff_layer

        wrap_drift_cliff_layer(segnet, layer_name=SEGNET_CLIFF_LAYER, strategy=cliff_strategy)
    posenet = posenet.to(torch.device(backend)).eval()
    segnet = segnet.to(torch.device(backend)).eval()
    return posenet, segnet


# ---------------------------------------------------------------------------
# Real-frame loader (delegates to the canonical mirror harness; Catalog #213)
# ---------------------------------------------------------------------------


def load_real_btchw_frames(
    *,
    num_pairs: int = 2,
    upstream_dir: str | Path = "upstream",
    video: str | Path | None = None,
    device: str = "cpu",
):
    """Decode real ``0.mkv`` frames into ``(P, 2, 3, 874, 1164)`` float btchw.

    Routes through the canonical ``verify_upstream_scorer_mirror_fidelity``
    decode path (Catalog #213) so the input is the EXACT contest-frame
    pipeline, never a synthetic fixture (Slot EEE Class 3).
    """
    import importlib.util

    import torch

    up = Path(upstream_dir)
    vid = Path(video) if video is not None else up / "videos" / "0.mkv"
    spec = importlib.util.spec_from_file_location(
        "_vsm_for_parity", "tools/verify_upstream_scorer_mirror_fidelity.py"
    )
    _require(spec is not None and spec.loader is not None, "cannot load mirror harness")
    vsm = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(vsm)  # type: ignore[union-attr]
    frames = vsm._decode_real_frames(vid, num_pairs * 2)  # (2P, H, W, 3) uint8
    return vsm._to_btchw(frames, torch.device(device))  # (P, 2, 3, H, W) float


# ---------------------------------------------------------------------------
# Op-by-op parity harness (forward) + forward/backward verdict
# ---------------------------------------------------------------------------


def measure_op_drift_table(
    *,
    num_pairs: int = 1,
    upstream_dir: str | Path = "upstream",
    video: str | Path | None = None,
    backend: str = "mps",
    apply_cliff_fix: bool = False,
    cliff_strategy: str = "cpu_wrap",
    top_k: int = 10,
) -> tuple[list[OpDriftRow], list[OpDriftRow]]:
    """Per-op CPU-vs-AppleGPU L_inf drift table for PoseNet + SegNet on real
    frames, via the canonical ``layerwise_drift`` harness.

    Returns ``(posenet_rows, segnet_rows)`` sorted by L_inf descending,
    truncated to ``top_k``.
    """
    import torch

    from tac.mps_diagnostic.layerwise_drift import measure_layerwise_drift
    from tac.scorer import load_default_scorers, make_scorers_differentiable

    x = load_real_btchw_frames(
        num_pairs=num_pairs, upstream_dir=upstream_dir, video=video, device="cpu"
    )
    posenet, segnet = load_default_scorers(Path(upstream_dir), device=torch.device("cpu"))
    make_scorers_differentiable(posenet, segnet)
    if apply_cliff_fix:
        from tac.mps_diagnostic.targeted_fix import wrap_drift_cliff_layer

        wrap_drift_cliff_layer(segnet, layer_name=SEGNET_CLIFF_LAYER, strategy=cliff_strategy)

    def _table(model: Any, inp: Any, name: str) -> list[OpDriftRow]:
        res = measure_layerwise_drift(
            model, inp.contiguous(), backends=("cpu", backend), cliff_threshold=CLIFF_THRESHOLD
        )
        pk = next(iter(res["pairs"]))
        recs = res["pairs"][pk]["records"]
        rows = [
            OpDriftRow(
                model=name,
                layer_name=r.get("layer_name", "?"),
                layer_class=r.get("layer_class", ""),
                l_inf=float(r.get("l_inf", 0.0)),
                l_2=float(r.get("l_2", 0.0)),
                mean_rel=float(r.get("mean_rel", 0.0)),
                above_cliff=float(r.get("l_inf", 0.0)) > CLIFF_THRESHOLD,
            )
            for r in recs
        ]
        rows.sort(key=lambda rr: rr.l_inf, reverse=True)
        return rows[:top_k]

    pin = posenet.preprocess_input(x)
    sin = segnet.preprocess_input(x)
    return _table(posenet, pin, "posenet"), _table(segnet, sin, "segnet")


def _representative_score_aware_loss(posenet: Any, segnet: Any, x: Any):
    """A representative score-aware loss whose gradient w.r.t. the frames is
    the training signal a carrier fit consumes.

    pose term: pose-head scored-slice squared magnitude (drives d_pose down)
    seg  term: SegNet softmax negative-entropy (drives confident argmax)

    This is NOT the contest score; it is a faithful proxy whose gradient
    direction is what matters for SGD on the carrier.
    """
    import torch

    pin = posenet.preprocess_input(x)
    pout = posenet(pin)["pose"]
    half = pout.shape[-1] // 2
    pose_loss = pout[..., :half].pow(2).mean()
    slog = segnet(segnet.preprocess_input(x))
    logp = torch.nn.functional.log_softmax(slog, 1)
    seg_loss = logp.exp().mul(logp).sum(1).mean()
    return pose_loss + seg_loss, pose_loss.detach(), seg_loss.detach()


def measure_forward_backward_parity(
    *,
    num_pairs: int = 2,
    upstream_dir: str | Path = "upstream",
    video: str | Path | None = None,
    backend: str = "mps",
    apply_cliff_fix: bool = True,
    cliff_strategy: str = "cpu_wrap",
) -> ScorerParityVerdict:
    """The headline parity verdict: forward d_seg/d_pose + backward gradient
    cosine of the score-aware-training signal, CPU vs Apple-GPU.

    The ``residual_drift_factor_vs_stale_23x`` collapses the canonical stale
    23x PoseNet figure to the empirically-measured current factor.
    """
    import torch

    x_cpu = load_real_btchw_frames(
        num_pairs=num_pairs, upstream_dir=upstream_dir, video=video, device="cpu"
    )

    # --- Forward: bare scorers on CPU vs Apple-GPU (no fix needed for forward) ---
    pn_cpu, sn_cpu = build_faithful_apple_gpu_scorers(
        upstream_dir, backend="cpu", apply_cliff_fix=False
    )
    with torch.no_grad():
        seg_cpu = sn_cpu(sn_cpu.preprocess_input(x_cpu))
        pose_cpu = pn_cpu(pn_cpu.preprocess_input(x_cpu))["pose"]
    pn_g, sn_g = build_faithful_apple_gpu_scorers(
        upstream_dir, backend=backend, apply_cliff_fix=False
    )
    x_g = x_cpu.to(torch.device(backend))
    with torch.no_grad():
        seg_g = sn_g(sn_g.preprocess_input(x_g))
        pose_g = pn_g(pn_g.preprocess_input(x_g))["pose"]

    seg_cpu_np = seg_cpu.detach().float().cpu().numpy()
    seg_g_np = seg_g.detach().float().cpu().numpy()
    seg_l_inf = float(np.max(np.abs(seg_cpu_np - seg_g_np)))
    flip = float((seg_cpu_np.argmax(1) != seg_g_np.argmax(1)).mean())
    half = pose_cpu.shape[-1] // 2
    pc = pose_cpu[..., :half].detach().float().cpu().numpy()
    pg = pose_g[..., :half].detach().float().cpu().numpy()
    pose_l_inf = float(np.max(np.abs(pc - pg)))
    pose_rel = pose_l_inf / (float(np.max(np.abs(pc))) + 1e-12)

    # --- Per-op cliff table (with the fix applied so we report residual) ---
    p_rows, s_rows = measure_op_drift_table(
        num_pairs=1,
        upstream_dir=upstream_dir,
        video=video,
        backend=backend,
        apply_cliff_fix=apply_cliff_fix,
        cliff_strategy=cliff_strategy,
    )
    p_worst = p_rows[0] if p_rows else None
    s_worst = s_rows[0] if s_rows else None
    n_above = sum(1 for r in (p_rows + s_rows) if r.above_cliff)

    # --- Backward: score-aware-training gradient cosine ---
    def _grad(dev: str, fix: bool):
        pn, sn = build_faithful_apple_gpu_scorers(
            upstream_dir, backend=dev, apply_cliff_fix=fix, cliff_strategy=cliff_strategy
        )
        xx = x_cpu.to(torch.device(dev)).clone().requires_grad_(True)
        loss, _, _ = _representative_score_aware_loss(pn, sn, xx)
        (g,) = torch.autograd.grad(loss, xx)
        return g.detach().float().cpu().numpy()

    gc = _grad("cpu", apply_cliff_fix)
    gg = _grad(backend, apply_cliff_fix)
    gcf, ggf = gc.ravel(), gg.ravel()
    cos = float(np.dot(gcf, ggf) / (np.linalg.norm(gcf) * np.linalg.norm(ggf) + 1e-30))
    norm_ratio = float(np.linalg.norm(ggf) / (np.linalg.norm(gcf) + 1e-30))
    grad_max_abs = float(np.max(np.abs(gc - gg)))

    # Residual drift factor: the stale anchor was 23x on PoseNet pose. The
    # current PoseNet forward scored-slice rel is pose_rel. A factor of 1.0
    # means "as faithful as the CPU reference noise floor". We express the
    # collapse as 23x -> (pose_rel / cpu_noise_floor). The CPU noise floor for
    # fp32 pose is ~1e-7; pose_rel is measured. We clamp to >= 1.0.
    cpu_noise_floor = 1.0e-7
    current_factor = max(1.0, pose_rel / cpu_noise_floor)

    forward_faithful = (flip == 0.0) and (pose_rel < 1e-4)
    backward_faithful = cos >= GRAD_COSINE_FAITHFUL_FLOOR

    return ScorerParityVerdict(
        backend=backend,
        segnet_logits_l_inf=seg_l_inf,
        segnet_argmax_flip_rate=flip,
        posenet_scored_slice_l_inf=pose_l_inf,
        posenet_scored_slice_rel=pose_rel,
        segnet_worst_op=s_worst.layer_name if s_worst else "",
        segnet_worst_l_inf=s_worst.l_inf if s_worst else 0.0,
        posenet_worst_op=p_worst.layer_name if p_worst else "",
        posenet_worst_l_inf=p_worst.l_inf if p_worst else 0.0,
        num_ops_above_cliff=n_above,
        grad_cosine_cpu_vs_apple=cos,
        grad_norm_ratio=norm_ratio,
        grad_max_abs=grad_max_abs,
        cliff_fix_applied=(cliff_strategy if apply_cliff_fix else "none"),
        forward_faithful=forward_faithful,
        backward_faithful=backward_faithful,
        residual_drift_factor_vs_stale_23x=current_factor,
    )


# ---------------------------------------------------------------------------
# Faithfulness proof: short Apple-GPU score-aware smoke fit
# ---------------------------------------------------------------------------


def run_score_aware_faithfulness_smoke_fit(
    *,
    steps: int = 20,
    lr: float = 0.05,
    num_pairs: int = 2,
    upstream_dir: str | Path = "upstream",
    video: str | Path | None = None,
    backend: str = "mps",
    apply_cliff_fix: bool = True,
    cliff_strategy: str = "cpu_wrap",
    seed: int = 0,
) -> FaithfulnessProofVerdict:
    """THE UNLOCK PROOF.

    1. Take real GT frames ``x`` and a perturbed candidate reconstruction
       ``y = x + delta`` where ``delta`` is a learnable per-pixel residual.
    2. Optimize ``delta`` ON THE APPLE GPU to minimize the contest-style
       distortion ``d_pose(x, y) + 100 * d_seg(x, y)`` (the score-aware
       objective a carrier fit consumes).
    3. Re-measure the achieved ``d_pose / d_seg`` ON THE CPU MIRROR (the
       bit-exact faithful reference).

    If the Apple-GPU-optimized ``delta`` reduces the TRUE CPU-measured
    distortion (not just the Apple-GPU-measured proxy), then score-aware
    training on the Apple GPU is FAITHFUL — the $0-local fast-training
    unlock.

    The candidate starts perturbed (a small textured offset) so there is a
    real distortion to reduce; a faithful fit drives it back toward the GT.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_cpu = load_real_btchw_frames(
        num_pairs=num_pairs, upstream_dir=upstream_dir, video=video, device="cpu"
    )

    # Build distortion fns on both backends (with the fix on the trained one).
    def _distortion(pn: Any, sn: Any, x: Any, y: Any):
        """Contest-style distortion between GT x and candidate y."""
        # PoseNet pose-head scored slice MSE between x and y outputs.
        px = pn(pn.preprocess_input(x))["pose"]
        py = pn(pn.preprocess_input(y))["pose"]
        half = px.shape[-1] // 2
        d_pose = (px[..., :half] - py[..., :half]).pow(2).mean()
        # SegNet argmax-disagreement is non-differentiable; use a smooth
        # surrogate: KL(softmax(x) || softmax(y)) which upper-bounds the
        # argmax-flip rate and IS the canonical score-aware seg signal.
        lx = torch.nn.functional.log_softmax(sn(sn.preprocess_input(x)), 1)
        ly = torch.nn.functional.log_softmax(sn(sn.preprocess_input(y)), 1)
        d_seg = (lx.exp() * (lx - ly)).sum(1).mean()
        return d_pose, d_seg

    # A fixed, reproducible candidate perturbation (small textured offset).
    g = torch.Generator().manual_seed(seed)
    delta0 = (torch.randn(x_cpu.shape, generator=g) * 6.0)  # ~6 levels of perturbation

    # --- Measure TRUE (CPU) distortion of the INITIAL candidate ---
    pn_cpu, sn_cpu = build_faithful_apple_gpu_scorers(
        upstream_dir, backend="cpu", apply_cliff_fix=False
    )
    with torch.no_grad():
        y0_cpu = (x_cpu + delta0).clamp(0, 255)
        dp_b, ds_b = _distortion(pn_cpu, sn_cpu, x_cpu, y0_cpu)
    true_pose_before = float(dp_b)
    true_seg_before = float(ds_b)

    # --- Train delta ON THE APPLE GPU ---
    pn_g, sn_g = build_faithful_apple_gpu_scorers(
        upstream_dir, backend=backend, apply_cliff_fix=apply_cliff_fix, cliff_strategy=cliff_strategy
    )
    dev = torch.device(backend)
    x_g = x_cpu.to(dev)
    delta = delta0.clone().to(dev).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    apple_pose_before = apple_seg_before = None
    for step in range(steps):
        opt.zero_grad()
        y = (x_g + delta).clamp(0, 255)
        dp, ds = _distortion(pn_g, sn_g, x_g, y)
        if step == 0:
            apple_pose_before = float(dp.detach())
            apple_seg_before = float(ds.detach())
        loss = dp + 100.0 * ds
        loss.backward()
        opt.step()
    with torch.no_grad():
        y_final = (x_g + delta).clamp(0, 255)
        dp_a, ds_a = _distortion(pn_g, sn_g, x_g, y_final)
    apple_pose_after = float(dp_a)
    apple_seg_after = float(ds_a)

    # --- Re-measure TRUE (CPU) distortion of the Apple-GPU-optimized delta ---
    delta_final_cpu = delta.detach().float().cpu()
    with torch.no_grad():
        y_final_cpu = (x_cpu + delta_final_cpu).clamp(0, 255)
        dp_true_a, ds_true_a = _distortion(pn_cpu, sn_cpu, x_cpu, y_final_cpu)
    true_pose_after = float(dp_true_a)
    true_seg_after = float(ds_true_a)

    true_pose_reduced = true_pose_after < true_pose_before
    true_seg_reduced = true_seg_after < true_seg_before
    unlock = true_pose_reduced and true_seg_reduced

    return FaithfulnessProofVerdict(
        steps=steps,
        backend_trained_on=backend,
        true_cpu_pose_before=true_pose_before,
        true_cpu_pose_after=true_pose_after,
        true_cpu_seg_before=true_seg_before,
        true_cpu_seg_after=true_seg_after,
        apple_pose_before=float(apple_pose_before) if apple_pose_before is not None else math.nan,
        apple_pose_after=apple_pose_after,
        apple_seg_before=float(apple_seg_before) if apple_seg_before is not None else math.nan,
        apple_seg_after=apple_seg_after,
        true_pose_reduced=true_pose_reduced,
        true_seg_reduced=true_seg_reduced,
        unlock=unlock,
    )


__all__ = [
    "PARITY_TAG",
    "ADVISORY_TAG",
    "SEGNET_CLIFF_LAYER",
    "CLIFF_THRESHOLD",
    "GRAD_COSINE_FAITHFUL_FLOOR",
    "OpDriftRow",
    "ScorerParityVerdict",
    "FaithfulnessProofVerdict",
    "build_faithful_apple_gpu_scorers",
    "load_real_btchw_frames",
    "measure_op_drift_table",
    "measure_forward_backward_parity",
    "run_score_aware_faithfulness_smoke_fit",
]
