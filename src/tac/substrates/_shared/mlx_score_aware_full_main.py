# SPDX-License-Identifier: MIT
"""Canonical MLX-first score-aware training harness — sister of ``pact_nerv_full_main.py``.

# NO_GRAD_WAIVED:MLX_substrate_harness_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_harness_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive

MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27. Six MLX-first class-shift substrate
trainers (``experiments/train_substrate_{dreamer_v3_rssm, z8_*, mdl_ibps_j_*,
atw_v2_*, coin_pp_*, faiss_ivf_pq_residual}*.py``) each shipped an L0 SCAFFOLD
``_full_main`` raising ``NotImplementedError`` because the shared trainer
skeleton (``trainer_skeleton.py`` / ``pact_nerv_full_main.py``) is PyTorch-only
(``device_or_die`` rejects MPS per Catalog #1; ``decode_real_pairs`` returns a
``torch.Tensor``; the scorer routes through ``load_differentiable_scorers`` +
``score_pair_components_dispatch`` in PyTorch). This helper is the MLX-FIRST
sister: the substrate-AGNOSTIC MLX score-aware training loop that extinguishes
the ``NotImplementedError`` for substrates whose distinguishing primitive is an
MLX-native trainable renderer, while every variant keeps its UNIQUE primitive
(DreamerV3 RSSM / Z8 hierarchical predictive coding / categorical posterior /
ego-motion cooperative-receiver / etc.).

## The 8th standing directive (MLX-first numpy-portable individually-fractal)

Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL STANDING
DIRECTIVE": NEW substrates train MLX-first on the M5 Max while the INFLATE
path stays numpy/PIL-portable (no MLX dep; ≤200 LOC + ≤2 ext deps per HNeRV
parity L4). This harness binds the TRAINING half of that contract:

- **TRAINING (this harness)**: MLX-native ``value_and_grad`` + AdamW +
  Polyak/Kahan EMA + real-video GT + gradient-reachable score-aware loss.
- **INFLATE (the substrate's own ``inflate.py``)**: numpy/PIL-only decode of
  the archive bytes. :func:`assert_numpy_portable_inflate` statically verifies
  the substrate's ``inflate.py`` imports neither ``mlx`` nor ``torch`` so the
  MLX-trained weights decode via the substrate's numpy/PIL primitives.

The bridge contract (MLX state_dict -> archive bytes -> numpy inflate) is the
substrate's own ``export_archive`` (Protocol method) + ``inflate.py`` pair; this
harness owns only the substrate-AGNOSTIC training loop.

## Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" the
canonical-vs-unique split is:

- ADOPT_CANONICAL_BECAUSE_SERVES (this helper + the canonical L2 harness it
  wraps):
  * train/val pair split, AdamW, Polyak/Kahan EMA shadow, OOM-safe step,
    early-stopping, per-epoch telemetry, canonical Provenance + posterior
    anchor, archive export wiring. These are substrate-AGNOSTIC and route
    through ``tac.training.long_training_canonical.run_long_training`` (the
    canonical L2 harness; the Z6 ``Z6LongTrainingAdapter`` is the proven
    reference adapter this generic adapter generalizes).
  * real-video decode via ``tac.data.decode_video`` (the MLX-side sister of
    ``decode_real_pairs``; returns numpy/torch frames we normalize to MLX
    float32 NHWC in ``[0, 1]``).
  * gradient-reachable score-aware loss via the canonical Hinton-distilled
    KL T=2.0 surrogate math (``tac.substrates.hinton_distilled_scorer_surrogate.
    mlx_loss``) which is the MLX-native gradient-reachable scorer-surrogate
    path per CLAUDE.md "eval_roundtrip" + Catalog #164 sister discipline.
- FORK_BECAUSE_PRINCIPLED_MISMATCH (stays in each substrate package):
  * the MLX renderer (``mlx_renderer.py`` / ``module.py``) — the distinguishing
    primitive, an ``mlx.nn.Module`` (or array-bearing object) with a
    ``reconstruct_pair(idx)`` or ``__call__(idx)`` forward.
  * archive grammar + numpy/PIL-portable inflate (``archive.py`` / ``inflate.py``)
    — each variant's byte layout + decode path differs.
  * any variant-specific extra loss terms (residual L2 / commitment / MINE)
    are wired via the ``extra_loss_terms`` callback so the canonical loop never
    assumes a fixed loss signature.

The variant passes a ``RendererBundle`` describing its renderer + targets +
optional extra-loss callback; only the AGNOSTIC scaffold is shared.

## Non-promotable by construction

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/#192/
#317/#341: this harness does NOT promote, rank, or claim any score. Every
artifact is tagged ``[macOS-MLX research-signal]`` with ``score_claim=False``,
``promotion_eligible=False``, ``ready_for_exact_eval_dispatch=False``. The
canonical L2 harness auto-stamps these markers on the ``TrainingArtifact``.
Promotion requires MLX state_dict -> PyTorch bridge -> archive -> paired
``[contest-CUDA]`` + ``[contest-CPU]`` auth-eval on 1:1 contest-compliant
hardware (operator-routed paid dispatch).

## Dispatch gating (Catalog #325)

This harness runs on the M5 Max via MLX at $0; it NEVER triggers a paid GPU
dispatch. The substrate recipes stay ``dispatch_enabled: false`` +
``research_only: true`` until each substrate clears its per-substrate
symposium. The harness fails closed on a non-MLX host (no silent CPU/CUDA
fallback per Catalog #1 + #317): :func:`require_mlx_for_harness` raises with a
clear diagnostic rather than dispatching to a paid path.

[verified-against: tac.training.long_training_canonical.run_long_training canonical L2 harness]
[verified-against: tac.substrates.time_traveler_l5_z6.long_training_adapter.Z6LongTrainingAdapter proven Style-B reference]
[verified-against: tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss gradient-reachable scorer surrogate]
[verified-against: tac.local_acceleration.pr95_hnerv_mlx canonical tinygrad-like MLX primitives]
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical contest constants (sister of pact_nerv_full_main).
CONTEST_NORMALIZER: float = 37_545_489.0
N_PAIRS_FULL: int = 600
# MLX false-authority canonical markers (sister of pr95_hnerv_mlx FALSE_AUTHORITY).
MLX_EVIDENCE_GRADE: str = "[macOS-MLX research-signal]"
# inflate.py portability contract: NO mlx / torch import at decode time.
FORBIDDEN_INFLATE_IMPORT_ROOTS: tuple[str, ...] = ("mlx", "torch")


class MlxScoreAwareHarnessError(RuntimeError):
    """Raised when the MLX score-aware harness cannot run faithfully."""


def require_mlx_for_harness() -> Any:
    """Return the ``mlx.core`` module or fail closed on a non-MLX host.

    Per Catalog #1 + #317: NO silent CPU/CUDA fallback. The harness is
    MLX-local ($0 M5 Max) by construction; on a non-MLX host the correct
    behaviour is a clear diagnostic, NOT a paid-dispatch leak.

    Returns:
        the imported ``mlx.core`` module.

    Raises:
        MlxScoreAwareHarnessError: MLX is not importable.
    """
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - non-Apple CI.
        raise MlxScoreAwareHarnessError(
            "MLX score-aware harness requires MLX (Apple Silicon only). "
            "Install via `uv pip install mlx` or invoke from a macOS-ARM64 "
            "host. There is NO CPU/CUDA fallback (Catalog #1 + #317): the "
            "PyTorch sister path is `pact_nerv_full_main.py` / "
            "`trainer_skeleton.device_or_die` for CUDA dispatch."
        ) from exc
    return mx


# ---------------------------------------------------------------------------
# Real-video target decode (Catalog #114: real contest video, never synthetic
# outside smoke)
# ---------------------------------------------------------------------------


def decode_mlx_targets(
    video_path: Any,
    *,
    num_pairs: int,
    output_height: int,
    output_width: int,
) -> tuple[Any, Any]:
    """Decode real contest video into per-pair MLX target frame buffers.

    Per Catalog #114 the targets come from the actual contest video, NEVER
    ``make_synthetic_*`` outside ``--smoke``. Decodes ``2 * num_pairs`` frames,
    reshapes to ``(num_pairs, 2, H, W, 3)`` and splits into two NHWC MLX
    float32 buffers normalized to ``[0, 1]`` (the canonical MLX target layout
    consumed by the Z6 reference adapter).

    Args:
        video_path: path to the contest video (e.g. ``upstream/videos/0.mkv``).
        num_pairs: number of adjacent-frame pairs to decode (full = 600).
        output_height / output_width: target spatial size (the substrate
            renderer's output resolution).

    Returns:
        ``(target_rgb_0, target_rgb_1)`` each MLX float32 ``(num_pairs, H, W,
        3)`` in ``[0, 1]``.

    Raises:
        MlxScoreAwareHarnessError: fewer than ``2 * num_pairs`` frames decoded.
    """
    import numpy as np

    mx = require_mlx_for_harness()

    from tac.data import decode_video

    frames = decode_video(
        video_path,
        target_h=output_height,
        target_w=output_width,
        max_frames=2 * num_pairs,
    )
    if len(frames) < 2 * num_pairs:
        raise MlxScoreAwareHarnessError(
            f"decoded {len(frames)} frames from {video_path!s}; need "
            f"{2 * num_pairs} for {num_pairs} pairs at "
            f"({output_height}, {output_width})"
        )
    gt_arr = np.stack([f.numpy() for f in frames[: 2 * num_pairs]], axis=0)
    gt_pairs = gt_arr.reshape(num_pairs, 2, output_height, output_width, 3)
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))
    return target_rgb_0, target_rgb_1


# ---------------------------------------------------------------------------
# RendererBundle: the substrate-specific axis passed into the harness
# ---------------------------------------------------------------------------


@dataclass
class RendererBundle:
    """Substrate-specific renderer + targets + optional extra-loss callback.

    The canonical-vs-unique boundary: everything in this bundle is the
    substrate's UNIQUE axis; the harness owns everything else (AGNOSTIC).

    Attributes:
        model: the MLX renderer. MUST be an ``mlx.nn.Module`` (or expose
            ``.parameters()`` + ``.update()`` so MLX ``value_and_grad`` can
            differentiate it) AND expose ONE of:
              * ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` with each
                ``(B, 3, H, W)`` in ``[0, 1]`` (the Z6 / atw_v2 convention), OR
              * ``__call__(idx) -> (B, 2, 3, H, W)`` in ``[0, 255]`` (the
                dreamer / z8 HNeRV convention).
            The harness auto-detects the convention via ``forward_convention``.
        target_rgb_0: MLX float32 ``(num_pairs, H, W, 3)`` in ``[0, 1]``.
        target_rgb_1: MLX float32 ``(num_pairs, H, W, 3)`` in ``[0, 1]``.
        num_pairs: total trainable pair count.
        forward_convention: ``"reconstruct_pair_nchw01"`` (model returns
            ``(rgb_0, rgb_1)`` NCHW in ``[0, 1]``) or ``"call_b2chw_255"``
            (model returns ``(B, 2, 3, H, W)`` in ``[0, 255]``).
        extra_loss_terms: optional callback ``(model, idx) -> {name: scalar}``
            for the variant's UNIQUE extra terms (residual L2 / commitment /
            MINE). Each scalar is weighted by ``loss_weights[name]`` (default
            weight from ``extra_loss_weights``). The harness adds them to the
            reconstruction + score-aware terms.
        extra_loss_weights: default Lagrangian weights for ``extra_loss_terms``
            keys.
        distillation_weight: weight ``λ`` on the gradient-reachable Hinton-KL
            T=2.0 score-aware surrogate term. ``0.0`` disables it (pure
            reconstruction). Default ``0.0`` so a substrate opts INTO the
            scorer surrogate explicitly (the canonical reconstruction-proxy
            posture mirrors the Z6 reference which DEFERS per-axis to the
            PyTorch sister).
        export_state_dict_fn: optional ``(model, path) -> None`` PyTorch-export
            bridge; threaded into the adapter's ``export_state_dict``.
        export_archive_fn: optional ``(model, output_dir) -> (path, sha, bytes)``
            numpy-portable archive builder; threaded into the adapter's
            ``export_archive``.
    """

    model: Any
    target_rgb_0: Any
    target_rgb_1: Any
    num_pairs: int
    forward_convention: str = "call_b2chw_255"
    extra_loss_terms: Callable[[Any, Any], Mapping[str, Any]] | None = None
    extra_loss_weights: Mapping[str, float] = field(default_factory=dict)
    distillation_weight: float = 0.0
    distillation_temperature: float = 2.0
    distillation_num_classes: int = 5
    export_state_dict_fn: Callable[[Any, Path], None] | None = None
    export_archive_fn: (
        Callable[[Any, Path], tuple[Path, str, int] | None] | None
    ) = None

    def __post_init__(self) -> None:
        valid = {"reconstruct_pair_nchw01", "call_b2chw_255"}
        if self.forward_convention not in valid:
            raise MlxScoreAwareHarnessError(
                f"forward_convention must be one of {sorted(valid)}; got "
                f"{self.forward_convention!r}"
            )
        if self.num_pairs < 1:
            raise MlxScoreAwareHarnessError(
                f"num_pairs must be >= 1; got {self.num_pairs}"
            )
        if self.distillation_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                f"distillation_weight must be >= 0 (0.0 disables); got "
                f"{self.distillation_weight}"
            )


# ---------------------------------------------------------------------------
# Score-aware loss (gradient-reachable, MLX-native)
# ---------------------------------------------------------------------------


def _decode_frames_nhwc01(
    bundle: RendererBundle,
    idx: Any,
) -> tuple[Any, Any]:
    """Decode (rgb_0, rgb_1) as NHWC ``[0, 1]`` regardless of model convention.

    Returns two MLX float32 arrays each ``(B, H, W, 3)`` in ``[0, 1]``, ready
    for MSE against the canonical NHWC ``[0, 1]`` targets.
    """
    mx = require_mlx_for_harness()
    model = bundle.model
    if bundle.forward_convention == "reconstruct_pair_nchw01":
        result = model.reconstruct_pair(idx)
        # The renderer may return (rgb_0, rgb_1) or (rgb_0, rgb_1, z); take the
        # first two. Each is (B, 3, H, W) in [0, 1].
        rgb_0 = result[0]
        rgb_1 = result[1]
        rgb_0 = mx.transpose(rgb_0, (0, 2, 3, 1))
        rgb_1 = mx.transpose(rgb_1, (0, 2, 3, 1))
        return rgb_0, rgb_1
    # call_b2chw_255: model(idx) -> (B, 2, 3, H, W) in [0, 255].
    pair = model(idx)
    pair01 = pair / 255.0
    rgb_0 = mx.transpose(pair01[:, 0], (0, 2, 3, 1))
    rgb_1 = mx.transpose(pair01[:, 1], (0, 2, 3, 1))
    return rgb_0, rgb_1


def score_aware_loss(
    bundle: RendererBundle,
    idx: Any,
    *,
    recon_weight: float = 1.0,
    loss_weights: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute the gradient-reachable MLX score-aware Lagrangian.

    The combined loss is::

        L = recon_weight * (mse(rgb_0, gt_0) + mse(rgb_1, gt_1))
            + distillation_weight * T**2 * KL(student || teacher)
            + sum_k extra_weight[k] * extra_term_k

    The reconstruction MSE is over the canonical NHWC ``[0, 1]`` frames. The
    optional score-aware term is the canonical Hinton-distilled KL T=2.0
    surrogate (gradient-reachable from KL -> decoded frame -> renderer params)
    per CLAUDE.md "eval_roundtrip" + Catalog #164 sister discipline; the
    teacher is the deterministic ``MockTeacherLogitsProvider`` projection on
    the TARGET frame (stop-gradient), the student is the same projection on the
    DECODED frame (gradient-bearing). This avoids the self-KL false positive.

    Args:
        bundle: the substrate RendererBundle.
        idx: MLX int32 ``(B,)`` pair-index batch.
        recon_weight: Lagrangian weight on the reconstruction MSE term.
        loss_weights: optional per-name overrides for the extra-loss terms.

    Returns:
        ``(total_loss_scalar, parts_dict)`` where ``parts_dict`` has float
        component values for telemetry (``total`` / ``recon`` / ``distill`` /
        per-extra).
    """
    mx = require_mlx_for_harness()
    weights = dict(bundle.extra_loss_weights)
    if loss_weights:
        weights.update({k: float(v) for k, v in loss_weights.items()})

    rgb_0, rgb_1 = _decode_frames_nhwc01(bundle, idx)
    gt_0 = bundle.target_rgb_0[idx]
    gt_1 = bundle.target_rgb_1[idx]
    mse_0 = mx.mean((rgb_0 - gt_0) ** 2)
    mse_1 = mx.mean((rgb_1 - gt_1) ** 2)
    recon = mse_0 + mse_1
    total = recon_weight * recon
    parts: dict[str, Any] = {"recon": recon}

    if bundle.distillation_weight > 0.0:
        from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
            MockTeacherLogitsProvider,
            hinton_distilled_kl_t2_loss,
        )

        provider = MockTeacherLogitsProvider(
            num_classes=bundle.distillation_num_classes,
        )
        # Student consumes the DECODED frame_0; teacher consumes the TARGET
        # frame_0 (stop-gradient). Gradient flows KL -> student_logits ->
        # decoded -> renderer params.
        student_logits = provider.teacher_logits(rgb_0)
        teacher_logits = mx.stop_gradient(provider.teacher_logits(gt_0))
        distill = hinton_distilled_kl_t2_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=bundle.distillation_temperature,
        )
        total = total + bundle.distillation_weight * distill
        parts["distill"] = distill

    if bundle.extra_loss_terms is not None:
        extra = bundle.extra_loss_terms(bundle.model, idx)
        for name, term in extra.items():
            w = float(weights.get(name, 1.0))
            total = total + w * term
            parts[name] = term

    parts["total"] = total
    return total, parts


# ---------------------------------------------------------------------------
# MlxScoreAwareAdapter: the canonical Style-B adapter (SubstrateLongTrainingAdapter)
# ---------------------------------------------------------------------------


class MlxScoreAwareAdapter:
    """Generic Style-B MLX adapter satisfying ``SubstrateLongTrainingAdapter``.

    This is the substrate-AGNOSTIC bridge between any substrate ``RendererBundle``
    and the canonical L2 harness ``tac.training.long_training_canonical.
    run_long_training``. It generalizes the proven Z6 ``Z6LongTrainingAdapter``
    so each substrate's ``_full_main`` is ~30 LOC of config + one harness call.

    Style B (combined ``train_step``) is used because MLX's ``value_and_grad``
    requires a combined value+grad+update step (the canonical helper prefers
    ``train_step`` when present per the Protocol contract).
    """

    def __init__(
        self,
        bundle: RendererBundle,
        *,
        substrate_id: str,
    ) -> None:
        mx = require_mlx_for_harness()
        import mlx.nn as mlx_nn
        import mlx.optimizers as mlx_optim

        self._mx = mx
        self._mlx_nn = mlx_nn
        self._mlx_optim = mlx_optim
        self.bundle = bundle
        self.model = bundle.model
        self.substrate_id = substrate_id
        self._optimizer: Any = None
        self._optimizer_lr: float | None = None

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a deterministic batch of pair indices (Catalog #229 PV)."""
        import numpy as np

        mx = self._mx
        num_pairs = self.bundle.num_pairs
        size = min(max(1, batch_size), num_pairs)
        rng = np.random.RandomState(seed)
        sampled = rng.choice(num_pairs, size=size, replace=False)
        return mx.array(sampled.astype("int32"))

    def loss_fn(
        self,
        model: Any,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style A diagnostic loss (no grad/update); Style B train_step is used.

        Provided for Protocol conformance + sister tooling that wants a pure
        loss read. The canonical helper detects ``train_step`` and bypasses
        this.
        """
        mx = self._mx
        _total, parts = score_aware_loss(
            self.bundle, batch, loss_weights=loss_weights
        )
        out: dict[str, float] = {}
        for name, value in parts.items():
            mx.eval(value)
            out[name] = float(value.item())
        return out

    def optimizer_step(
        self, model: Any, loss: Any, learning_rate: float
    ) -> None:
        """Style A stub; this adapter uses Style B ``train_step``.

        Per CLAUDE.md "Comment-only contracts are FORBIDDEN": this raises so a
        caller cannot silently no-op. The canonical helper detects
        ``train_step`` and never calls this.
        """
        raise NotImplementedError(
            "MlxScoreAwareAdapter uses Style B train_step "
            "(combined value+grad+update for MLX value_and_grad). The "
            "canonical helper prefers train_step when present; this "
            "optimizer_step is a Protocol-conformance stub only."
        )

    def train_step(
        self,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style B combined value+grad+update (canonical MLX training step)."""
        mx = self._mx
        mlx_nn = self._mlx_nn
        mlx_optim = self._mlx_optim
        if self._optimizer is None or self._optimizer_lr != learning_rate:
            self._optimizer = mlx_optim.AdamW(learning_rate=learning_rate)
            self._optimizer_lr = learning_rate

        def _loss_fn_inner(model: Any) -> Any:
            # NOTE: score_aware_loss reads bundle.model; the value_and_grad
            # closure differentiates ``self.model`` which IS bundle.model.
            total, _parts = score_aware_loss(
                self.bundle, batch, loss_weights=loss_weights
            )
            return total

        loss_and_grad_fn = mlx_nn.value_and_grad(self.model, _loss_fn_inner)
        loss_value, grads = loss_and_grad_fn(self.model)
        self._optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self._optimizer.state)
        return {"total": float(loss_value.item())}

    def export_state_dict(self, model: Any, path: Path) -> None:
        """Export the model state for checkpointing.

        Two paths:

        1. If the substrate wired ``export_state_dict_fn`` (its MLX->PyTorch
           bridge per Catalog #1251), delegate to it (the promotion path).
        2. Otherwise write a numpy-portable MLX-native checkpoint: flatten the
           model parameters to a ``.npz`` of numpy arrays (NO PyTorch dep).
           This keeps checkpointing functional for any MLX substrate while the
           PyTorch promotion bridge is a later deliverable; the npz is the
           canonical MLX-native portable state (sister of the pr95 long-training
           ``np.savez`` fallback). The checkpoint is non-promotable research
           signal per Catalog #192.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.bundle.export_state_dict_fn is not None:
            self.bundle.export_state_dict_fn(model, path)
            return
        import numpy as np

        flat: dict[str, Any] = {}

        def _flatten(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _flatten(f"{prefix}.{i}" if prefix else str(i), v)
            elif hasattr(obj, "shape"):
                flat[prefix] = np.asarray(obj)

        _flatten("", model.parameters())
        npz_path = path.with_suffix(path.suffix + ".mlx.npz")
        np.savez(str(npz_path), **flat)

    def export_archive(
        self, model: Any, output_dir: Path
    ) -> tuple[Path, str, int] | None:
        """Export the substrate's numpy-portable archive (0.bin)."""
        if self.bundle.export_archive_fn is None:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.bundle.export_archive_fn(model, output_dir)

    def score_aware_components(
        self, model: Any, batch: Any
    ) -> Mapping[str, float] | None:
        """Per-axis decomposition is DEFERRED to the PyTorch sister L2 path.

        Per the Z6 reference adapter + per-substrate symposium discipline:
        true contest-grade per-axis SegNet/PoseNet decomposition routes
        through the PyTorch sister (Catalog #164 + #226). The MLX L2 trainer
        is reconstruction-proxy + Hinton-KL-surrogate only; per-axis is the
        L3+ sister cascade's responsibility. Returns ``None`` (observability-
        only; never fails the run).
        """
        return None


# ---------------------------------------------------------------------------
# Numpy-portable inflate contract (the 8th directive's portability half)
# ---------------------------------------------------------------------------


def assert_numpy_portable_inflate(
    inflate_py_path: Any,
    *,
    forbidden_roots: Sequence[str] = FORBIDDEN_INFLATE_IMPORT_ROOTS,
) -> dict[str, Any]:
    """Statically verify a substrate ``inflate.py`` is numpy/PIL-portable.

    Per the 8th standing directive: the INFLATE path must decode MLX-trained
    weights via numpy/PIL primitives with NO ``mlx`` / ``torch`` import. This
    parses ``inflate.py`` via ``ast`` and refuses any ``import mlx`` /
    ``import torch`` / ``from mlx... import`` / ``from torch... import``
    statement (including dotted submodules ``mlx.core`` / ``torch.nn``).

    String mentions of ``mlx`` / ``torch`` in comments / docstrings are NOT
    flagged (ast import-node scan only).

    Args:
        inflate_py_path: path to the substrate's ``inflate.py``.
        forbidden_roots: import roots that break numpy-portability.

    Returns:
        ``{"numpy_portable": True, "checked_path": ..., "import_roots": [...]}``
        on success.

    Raises:
        MlxScoreAwareHarnessError: ``inflate.py`` missing OR imports a
            forbidden root.
    """
    path = Path(inflate_py_path)
    if not path.is_file():
        raise MlxScoreAwareHarnessError(
            f"inflate.py not found at {path!s}; numpy-portable inflate "
            "contract cannot be verified (HNeRV parity L4)."
        )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    seen_roots: set[str] = set()
    forbidden_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                seen_roots.add(root)
                if root in forbidden_roots:
                    forbidden_hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # node.module is None for `from . import x` (relative); skip those.
            root = node.module.split(".", 1)[0]
            seen_roots.add(root)
            if root in forbidden_roots:
                forbidden_hits.append(node.module)
    if forbidden_hits:
        raise MlxScoreAwareHarnessError(
            f"inflate.py at {path!s} imports forbidden non-portable root(s) "
            f"{sorted(set(forbidden_hits))}; the numpy-portable inflate "
            "contract (8th standing directive + HNeRV parity L4) requires the "
            "decode path to be numpy/PIL-only (no mlx/torch). MLX is a "
            "training-time-only dependency."
        )
    return {
        "numpy_portable": True,
        "checked_path": str(path),
        "import_roots": sorted(seen_roots),
    }


# ---------------------------------------------------------------------------
# The substrate-AGNOSTIC _full_main body
# ---------------------------------------------------------------------------


def run_mlx_score_aware_full_main(
    *,
    bundle: RendererBundle,
    substrate_id: str,
    lane_id: str,
    output_dir: Any,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float = 1e-3,
    seed: int = 0,
    ema_decay: float | None = None,
    checkpoint_interval_epochs: int = 10,
    early_stopping_patience: int | None = None,
    curriculum_stages: Any | None = None,
    inflate_py_path: Any | None = None,
    notes: str = "",
    on_epoch_end: Callable[[Any], None] | None = None,
) -> Any:
    """Run the canonical MLX-first score-aware ``_full_main`` body.

    This is the substrate-AGNOSTIC ``_full_main`` the 6 MLX-first substrate
    trainers route through. It:

    1. Verifies MLX availability (fail-closed; no CPU/CUDA leak per Catalog
       #1 + #317 + #325).
    2. (Optional) Verifies the substrate's ``inflate.py`` is numpy-portable
       (8th directive; HNeRV parity L4) when ``inflate_py_path`` is supplied.
    3. Wraps the substrate ``RendererBundle`` in :class:`MlxScoreAwareAdapter`.
    4. Builds a canonical ``LongTrainingConfig`` (single full-stage curriculum
       by default; the substrate may pass a multi-stage curriculum).
    5. Routes through canonical ``run_long_training`` (EMA / OOM-safe /
       telemetry / checkpoint / Provenance / posterior anchor / archive
       export).

    Args:
        bundle: the substrate RendererBundle (UNIQUE axis).
        substrate_id: canonical substrate id.
        lane_id: canonical lane id per CLAUDE.md "Lane maturity registry".
        output_dir: canonical output dir (MUST NOT be ``/tmp`` per the
            transient-evidence trap; ``run_long_training`` validates this).
        epochs: total epoch budget.
        batch_pair_indices_per_step: training batch size.
        learning_rate / seed / checkpoint_interval_epochs: training hparams.
        ema_decay: optional EMA decay override (default = canonical 0.997).
        early_stopping_patience: optional override (default = epochs + 1, i.e.
            disabled; MLX-local runs are cheap so we run the full budget).
        curriculum_stages: optional ``tuple[CurriculumStage, ...]``; default is
            a single full-budget stage.
        inflate_py_path: optional path to the substrate ``inflate.py`` to
            verify numpy-portability before training (8th directive).
        notes: substantive rationale (Catalog #287 placeholder rejected by the
            config).
        on_epoch_end: optional per-epoch callback.

    Returns:
        the canonical ``TrainingArtifact`` from ``run_long_training``.

    Raises:
        MlxScoreAwareHarnessError: MLX unavailable OR inflate not portable.
    """
    require_mlx_for_harness()
    output_dir = Path(output_dir)

    from tac.training.long_training_canonical import (
        CANONICAL_EMA_DECAY,
        CurriculumStage,
        LongTrainingConfig,
        run_long_training,
    )

    if inflate_py_path is not None:
        assert_numpy_portable_inflate(inflate_py_path)

    if curriculum_stages is None:
        curriculum_stages = (
            CurriculumStage(
                name=f"{substrate_id}_mlx_score_aware_full",
                start_epoch=0,
                end_epoch=epochs,
                notes=(
                    "MLX-first score-aware full-budget stage via canonical "
                    "mlx_score_aware_full_main harness; reconstruction + "
                    "optional Hinton-KL T=2.0 scorer surrogate."
                ),
            ),
        )

    adapter = MlxScoreAwareAdapter(bundle, substrate_id=substrate_id)

    config = LongTrainingConfig(
        substrate_id=substrate_id,
        lane_id=lane_id,
        epochs=epochs,
        batch_pair_indices_per_step=batch_pair_indices_per_step,
        curriculum_stages=curriculum_stages,
        ema_decay=CANONICAL_EMA_DECAY if ema_decay is None else float(ema_decay),
        checkpoint_interval_epochs=checkpoint_interval_epochs,
        early_stopping_patience=(
            epochs + 1 if early_stopping_patience is None else early_stopping_patience
        ),
        learning_rate=learning_rate,
        seed=seed,
        output_dir=output_dir,
        device="mlx",
        evidence_grade=MLX_EVIDENCE_GRADE,
        notes=(
            notes
            or (
                f"{substrate_id} MLX-first score-aware L2 via canonical "
                "mlx_score_aware_full_main harness; non-promotable "
                "[macOS-MLX research-signal] per Catalog #192/#317/#341."
            )
        ),
    )

    return run_long_training(adapter, config, on_epoch_end=on_epoch_end)


__all__ = [
    "CONTEST_NORMALIZER",
    "FORBIDDEN_INFLATE_IMPORT_ROOTS",
    "MLX_EVIDENCE_GRADE",
    "N_PAIRS_FULL",
    "MlxScoreAwareAdapter",
    "MlxScoreAwareHarnessError",
    "RendererBundle",
    "assert_numpy_portable_inflate",
    "decode_mlx_targets",
    "require_mlx_for_harness",
    "run_mlx_score_aware_full_main",
    "score_aware_loss",
]
