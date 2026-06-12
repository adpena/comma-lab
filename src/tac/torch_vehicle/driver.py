# SPDX-License-Identifier: MIT
"""Resumable curriculum driver for the P2 torch-vehicle (vendored PR95 HNeRV-Muon).

This is the ADAPTER's heart. The vendored ``stages/common.py:train_stage`` is a
monolithic loop with NO resume, NO telemetry, and a HARDCODED ``base_channels=36``
(in 3 call sites) — it cannot run the base_ch=20 rate-win config and a death
loses the whole in-flight stage. We do NOT edit the pristine vendored source
(CLAUDE.md "Forbidden in-place edits to public PR intake clones"); instead we
RE-DRIVE the vendored PRIMITIVES (``HNeRVDecoder`` / ``Muon`` /
``partition_params_for_muon`` / the seg-loss fns / ``ema_update`` / ``apply_qat``
/ ``cat_entropy_v2`` / ``build_archive`` / ``parse_archive``) unchanged, with:

* ``base_channels`` threaded into ``HNeRVDecoder(...)`` (the FINDING-1 fix —
  parametrization lives in the driver, NOT in the pristine source);
* COMPLETE checkpoint/resume (decoder + latents + EMA shadow + AdamW state +
  Muon momentum + LR-scheduler state + RNG + curriculum position), so a death
  costs ≤1 checkpoint interval — verified bit-identical by the kill+restart
  test (``tests/test_driver_resume.py``);
* per-epoch telemetry to durable JSONL (the "Max observability" non-negotiable);
* BEST-checkpoint-by-canonical-score tracking from the EMA shadow (the EMA
  non-negotiable — inference/export bytes are the shadow).

The per-step math is a 1:1 port of ``common.py`` (eval_roundtrip bicubic↑→
bilinear↓→uint8-STE, joint clip, ``ema_update`` after each step, sigma weight
noise via QAT, C1a entropy) — verified faithful in ``tests/test_driver_faithful.py``.

The driver is SCORER-INJECTABLE: production wires the real frozen SegNet/PoseNet
via :class:`RealScorerContext` (``precompute_targets`` + ``evaluate_decoder``);
tests inject a tiny synthetic frozen scorer so the resume round-trip (which is
architecture-AGNOSTIC) is fast + deterministic, exactly as the MLX
``test_checkpoint`` does.

Authority: torch-CPU TRUSTED (CLAUDE.md "local CPU + MLX GPU good"); NO MPS. The
in-loop d_seg/d_pose are ``[contest-CPU advisory]`` NON-PROMOTABLE until the
byte-closed archive is run through ``upstream/evaluate.py``.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.torch_vehicle.checkpoint import (
    TorchCheckpointPosition,
    checkpoint_exists,
    load_checkpoint,
    read_manifest,
    save_checkpoint,
    write_done_marker,
)
from tac.torch_vehicle.telemetry import EpochRecord, TelemetryWriter

if TYPE_CHECKING:  # StageSpec used only in annotations at module level
    from tac.torch_vehicle.curriculum import StageSpec

_EVAL_H, _EVAL_W = 384, 512
_SEG_NUM_CLASSES = 5  # contest SegNet head (upstream/modules.py: smp.Unet classes=5)


def _seg_loss_for_spec(
    spec: StageSpec,
    seg_out: torch.Tensor,
    seg_targets_hard: torch.Tensor,
) -> torch.Tensor:
    """Score-domain seg loss router (Lever 2) — DEFAULT-PRESERVING.

    ``spec.seg_surrogate is None`` (the default for every vendored-projected
    ``StageSpec``) returns ``spec.seg_loss_fn(seg_out, seg_targets_hard)`` EXACTLY
    as the legacy non-routed call did — byte-for-byte identical, so a basin that
    resumes onto this code is unchanged (verified by
    ``test_default_seg_surrogate_is_byte_identical_to_vendored_call`` +
    ``test_default_seg_surrogate_gradient_is_byte_identical`` in
    ``tests/test_seg_surrogate_lever.py``).

    When ``spec.seg_surrogate`` is a surrogate name (``"soft_cosine"`` /
    ``"fisher_rao"`` / ``"sinkhorn"``) the seg term is routed through the
    differentiable score-domain d_seg surrogate
    :func:`tac.losses.core.segnet_surrogate_per_pixel`. The contest d_seg is the
    per-pixel SegNet ARGMAX-DISAGREEMENT rate; the surrogate concentrates the
    gradient where the argmax actually flips (HNeRV parity L6), unlike CE which
    spends capacity on confident-interior pixels the argmax already gets right.

    GT side-information: the curriculum caches the HARD per-pixel target argmax
    (``seg_targets_hard``, ``(B, 384, 512)`` int64), not GT logits. The score-domain
    surrogate needs a GT class distribution; we build SHARP one-hot GT LOGITS from
    the hard target (argmax(GT logits) == the cached hard class), so the GT stays
    hard while only the PRED is temperature-softened (see the inline comment below).
    With one-hot GT, soft-cosine ``1 - Σ_c softmax(pred/T)_c · onehot_c =
    1 - softmax(pred/T)[gt]`` is EXACTLY the per-pixel "pred's probability mass NOT
    on the GT-argmax class" — the differentiable argmax-flip surrogate the memo
    specifies, cache-free (argmax is all the surrogate's per-pixel target needs).
    """
    if spec.seg_surrogate is None:
        # Legacy vendored path — UNCHANGED. Do not touch (basin-resume safety).
        return spec.seg_loss_fn(seg_out, seg_targets_hard)

    from tac.losses.core import segnet_surrogate_per_pixel

    # seg_out: (B, 5, H, W) logits. seg_targets_hard: (B, H, W) int64 class indices.
    # Build SHARP one-hot GT LOGITS (B, 5, H, W): the GT-argmax class gets a large
    # positive logit, all others ~0. ``softmax(gt_logits / T)`` is then ~one-hot for
    # ANY reasonable T (at the memo's coldest T=0.05, 30/0.05=600 → exactly one-hot),
    # so the GT is NOT temperature-softened (it is hard, as the contest d_seg is a
    # hard argmax) while ``spec.seg_temperature`` softens ONLY the PREDICTION. With
    # one-hot GT, soft_cosine reduces to ``1 - softmax(pred/T)[gt]`` EXACTLY — the
    # differentiable argmax-flip surrogate the memo specifies (verified
    # bit-equal to a hand-computed reference at T∈{1.0, 0.5, 0.1} in the lever test).
    # Passing GT as LOGITS (``gt_already_probs=False``) — NOT cached probs — is what
    # lets the surrogate honor non-unit ``temperature`` (the helper refuses cached
    # probs at T≠1 because cached probs are softmax@T=1; a one-hot logit has no such
    # constraint). This generalizes to fisher_rao / sinkhorn (same one-hot GT logits).
    _ONEHOT_GT_LOGIT = 30.0  # softmax(30·onehot/T) ≈ one-hot for all T ≥ ~0.02
    gt_onehot = F.one_hot(
        seg_targets_hard.long(), num_classes=_SEG_NUM_CLASSES
    )  # (B, H, W, 5)
    gt_logits = (
        gt_onehot.permute(0, 3, 1, 2).to(seg_out.dtype) * _ONEHOT_GT_LOGIT
    )  # (B, 5, H, W) sharp one-hot logits
    per_pixel = segnet_surrogate_per_pixel(
        seg_out,
        gt_logits,
        surrogate=spec.seg_surrogate,
        temperature=spec.seg_temperature,
        gt_already_probs=False,
        num_classes=_SEG_NUM_CLASSES,
    )  # (B, H, W)
    return per_pixel.mean()


class _SimulatedDeath(Exception):
    """Raised by the test-only ``_stop_after_global_epoch`` hook AFTER a checkpoint
    lands, to simulate a SIGKILL/OOM mid-run for the in-process resume test."""


# ---------------------------------------------------------------------------
# Scorer context protocol: production = real frozen SegNet/PoseNet; test = synthetic
# ---------------------------------------------------------------------------
class ScorerContext(Protocol):
    """The per-run frozen-scorer + GT-targets surface the driver trains against.

    Production binds the REAL contest scorer (``precompute_targets`` →
    ``distortion_net``, ``seg_targets_hard``, ``pose_targets``; GT via
    ``frame_utils.yuv420_to_rgb``). Tests bind a synthetic frozen scorer with the
    same interface so the resume round-trip is fast.
    """

    n_pairs: int
    seg_targets_hard: torch.Tensor  # (n_pairs, 384, 512) int64
    pose_targets: torch.Tensor  # (n_pairs, 6) float32

    def seg_pose_forward(
        self, decoded_bhwc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen scorer on roundtripped decoder output.

        ``decoded_bhwc`` is (B, 2, 384, 512, 3) float in [0,255] (post eval-
        roundtrip). Returns ``(seg_logits, pose_pred6)`` where ``seg_logits`` is
        (B*?, 5, 384, 512) matching ``seg_targets_hard`` indexing and
        ``pose_pred6`` is (B, 6) — so the driver's loss matches ``common.py``.
        """
        ...

    def exact_eval(
        self, ema_decoder: nn.Module, ema_latents: torch.Tensor, archive_bytes: int
    ) -> dict[str, float]:
        """Canonical d_seg / d_pose / rate / score on the EMA shadow (the score
        the BEST checkpoint tracks). Production routes through the real
        ``evaluate_decoder`` + ``compute_score`` on the parse-back archive."""
        ...


@dataclass
class TorchVehicleConfig:
    """Run-level config (architecture + budget + IO); the per-stage schedule
    comes from :func:`tac.torch_vehicle.curriculum.build_curriculum`."""

    base_channels: int = 20
    latent_dim: int = 28
    out_dir: Path = Path("experiments/results/torch_vehicle_run")
    checkpoint_every_epochs: int = 1  # a death costs <= this many epochs
    total_epoch_budget: int | None = None
    ema_decay: float = 0.999
    eval_every: int | None = None
    # ``device`` is the AUTHORITY/eval device — the exact d_seg/d_pose that pick
    # the BEST checkpoint and seed the telemetry MUST be CPU-TRUSTED (or CUDA);
    # MPS is FORBIDDEN here (CLAUDE.md "MPS auth eval is NOISE": 23x pose drift).
    device: str = "cpu"
    # ``train_device`` is the GRADIENT backend ONLY — the per-step forward+backward
    # of the decoder→roundtrip→frozen-scorer graph. MPS is ALLOWED here (and is the
    # whole point: ~104x faster fwd+bwd than torch-CPU per the bench). The MPS
    # gradient is research-signal until it passes the descent-equivalence acceptance
    # gate (BOTH d_seg AND d_pose at the real n); the EXACT metric is always re-run
    # on ``device`` (CPU authority). Defaults to ``device`` (single-device legacy).
    train_device: str | None = None
    # ``split_by_head`` (the pose-axis SALVAGE): run the SegNet-path forward+backward
    # on ``train_device`` (MPS — the 90x lever, validated bit-identical on d_seg) and
    # the PoseNet-path forward+backward on the CPU AUTHORITY ``device`` (zero pose
    # drift). The two per-head frame cotangents are summed at the frame tensor, so the
    # combined decoder gradient is descent-equivalent on BOTH terms BY CONSTRUCTION
    # (the pose part IS the authority gradient). The prior full-MPS gradient REJECTED
    # the descent-equivalence gate on the POSE axis (d_seg PASS / d_pose REJECT, gap
    # 0.06->7.02). Requires a split device. Defaults False (full train_device forward).
    split_by_head: bool = False
    # ``async_eval`` (the throughput salvage): run the CPU AUTHORITY exact eval
    # (byte-close + 600-pair d_seg/d_pose/rate/score) in a BACKGROUND THREAD off a
    # POINT-IN-TIME CPU snapshot of the EMA shadow, so the GPU/MPS training loop is
    # NOT blocked by the ~13-min CPU eval. The eval is the IDENTICAL eval on the
    # IDENTICAL snapshot — only non-blocking — so the authority numbers are
    # bit-for-bit the same as the sync path (proved by the no-regression test).
    # At most ONE eval is in-flight: a new eval epoch arriving while the prior eval
    # is still running is SKIPPED (logged) — the cadence self-throttles. On run
    # completion the in-flight thread is JOINED so the final BEST + last eval row
    # land before exit. Defaults False (the sync path is byte-identical unchanged).
    async_eval: bool = False
    # ``pose_film_enabled`` (Lever 3 — the Quantizr STORE-pose lesson): wrap the
    # vendored ``HNeRVDecoder`` with a stored-pose FiLM at the stem (channels[0])
    # and store the 6 GT pose scalars per pair as side information (Wyner-Ziv), so
    # d_pose collapses toward the stored-pose quant floor and the decoder capacity
    # is freed for d_seg + rate. The stored pose is range-coded into an ADDITIVE
    # archive section (~1 KB charged; the vendored codec stays pristine). DEFAULT
    # FALSE → the driver builds the vendored decoder unchanged and adds NO pose
    # section, so the archive is BYTE-IDENTICAL to today (the live basin is
    # unaffected if it resumes onto this code — proved by the byte-identity test).
    pose_film_enabled: bool = False
    # FiLM MLP bottleneck width (only consulted when ``pose_film_enabled``).
    pose_film_hidden: int = 8
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.device).lower().startswith("mps"):
            raise ValueError(
                "MPS is NEVER trusted as the AUTHORITY device (CLAUDE.md): corrupts "
                "SegNet/PoseNet argmax (23x pose drift). The exact d_seg/d_pose that "
                "pick BEST must be CPU/CUDA. Set device='cpu' and train_device='mps' "
                "to use the Apple GPU as the GRADIENT backend only (split-device)."
            )
        if self.train_device is None:
            self.train_device = self.device
        if self.split_by_head and self.train_device == self.device:
            raise ValueError(
                "split_by_head requires train_device != device (the SegNet path runs "
                "on train_device, the PoseNet path on the CPU authority device). Set "
                "device='cpu' and train_device='mps'."
            )
        self.out_dir = Path(self.out_dir)


@dataclass
class _EvalSnapshot:
    """An immutable POINT-IN-TIME CPU copy of the EMA shadow the eval scores.

    Captured in the MAIN thread (cheap), then either evaluated inline (sync) or
    handed to a background worker (async). Because it is a deep CPU copy, the
    training loop may keep mutating the live (MPS) EMA shadow without racing the
    eval — guaranteeing the async numbers equal the sync numbers on the SAME
    snapshot (the no-regression contract)."""

    ema_sd: dict[str, torch.Tensor]
    ema_latents: torch.Tensor


@dataclass
class _StageRuntime:
    """Mutable per-stage training objects (decoder/optimizers/sched/EMA)."""

    decoder: nn.Module
    latents: nn.Parameter
    ema_decoder: nn.Module
    ema_latents: torch.Tensor
    adamw_opt: torch.optim.Optimizer
    muon_opt: Any  # vendored Muon or None
    adamw_sched: Any
    muon_sched: Any
    muon_params: list
    adamw_params: list


class TorchVehicleDriver:
    """Drives the faithful PR95 8-stage curriculum with resume + telemetry.

    Usage (production)::

        ctx = RealScorerContext(video_path, device)
        driver = TorchVehicleDriver(cfg, scorer=ctx, vendored=import_vendored_bundle())
        driver.run()  # resumes from cfg.out_dir if a checkpoint is present
    """

    def __init__(
        self,
        cfg: TorchVehicleConfig,
        *,
        scorer: ScorerContext,
        vendored: VendoredBundle,
        curriculum: list[StageSpec] | None = None,
    ):
        self.cfg = cfg
        self.scorer = scorer
        self.v = vendored
        # AUTHORITY/eval device (CPU-TRUSTED or CUDA) — the exact d_seg/d_pose live
        # here. ``train_device`` is the GRADIENT backend (may be MPS for the 104x).
        self.device = torch.device(cfg.device)
        self.train_device = torch.device(cfg.train_device or cfg.device)
        self.split_device = self.train_device != self.device
        # Split-by-HEAD: SegNet grad on train_device (MPS), PoseNet grad on the CPU
        # authority, summed at the frame tensor. Honored only when the scorer context
        # exposes the per-head forwards (RealScorerContext / SyntheticScorerContext).
        self.split_by_head = bool(cfg.split_by_head) and getattr(
            scorer, "split_by_head", False
        )
        self.n_pairs = int(scorer.n_pairs)
        if curriculum is None:
            from tac.torch_vehicle.curriculum import build_curriculum

            curriculum = build_curriculum(
                total_epoch_budget=cfg.total_epoch_budget,
                ema_decay=cfg.ema_decay,
                eval_every=cfg.eval_every,
            )
        self.curriculum = curriculum
        self.telemetry = TelemetryWriter(
            cfg.out_dir,
            run_meta={
                "base_channels": cfg.base_channels,
                "latent_dim": cfg.latent_dim,
                "n_pairs": self.n_pairs,
                "total_epoch_budget": cfg.total_epoch_budget,
                "ema_decay": cfg.ema_decay,
                "device": cfg.device,
            },
        )
        self.best_score = self.telemetry.best_score
        self.best_ep = self.telemetry.best_ep
        self.best_stage = self.telemetry.best_stage
        self._global_epoch = 0
        # Test-only hook: simulate a death after this many global epochs (the
        # checkpoint has already landed when the death fires). None in production.
        self._stop_after_global_epoch: int | None = None
        # -- async-eval state (CLAUDE.md "async authority eval" throughput salvage) --
        # The async eval runs in a BACKGROUND THREAD; ``_eval_lock`` serializes the
        # two shared surfaces it touches with the main loop: the BEST-tracker fields
        # (best_score/best_ep/best_stage + the best/ dir write) AND the single
        # TelemetryWriter (which is NOT thread-safe — the main loop also records
        # non-eval rows). ``_eval_thread`` holds the at-most-one in-flight worker so
        # the cadence self-throttles (a new eval epoch arriving while the prior eval
        # is alive is SKIPPED). ``_skipped_evals`` counts the skips for the report.
        self._async_eval = bool(cfg.async_eval)
        self._eval_lock = threading.Lock()
        self._eval_thread: threading.Thread | None = None
        self._skipped_evals = 0
        self._inflight_snapshot_epoch: int | None = None

    # -- architecture construction (base_ch threaded — the FINDING-1 fix) -----
    def _new_vendored_decoder(self, device: torch.device | None = None) -> nn.Module:
        """Build a fresh base_ch-threaded VENDORED decoder (no FiLM)."""
        return self.v.HNeRVDecoder(
            latent_dim=self.cfg.latent_dim,
            base_channels=self.cfg.base_channels,
            eval_size=(_EVAL_H, _EVAL_W),
        ).to(device if device is not None else self.train_device)

    def _new_decoder(self, device: torch.device | None = None) -> nn.Module:
        """Build a fresh base_ch-threaded decoder on ``device`` (default: the
        TRAIN device — the gradient backend). The parse-back EVAL decoder is built
        on the AUTHORITY device explicitly (``device=self.device``).

        Lever 3 (``cfg.pose_film_enabled``): wrap the vendored decoder in a
        :class:`~tac.torch_vehicle.pose_film.PoseFiLMHNeRVWrapper` (stem-injection
        FiLM, identity at init) and SEED its ``stored_pose`` buffer from the GT
        PoseNet pose (``scorer.pose_targets``). DEFAULT-OFF returns the vendored
        decoder unchanged (byte-identical)."""
        dev = device if device is not None else self.train_device
        if not self.cfg.pose_film_enabled:
            return self._new_vendored_decoder(dev)
        from tac.torch_vehicle.pose_film import PoseFiLMHNeRVWrapper

        vendored = self._new_vendored_decoder(dev)
        wrapper = PoseFiLMHNeRVWrapper(
            vendored,
            n_pairs=self.n_pairs,
            film_hidden=self.cfg.pose_film_hidden,
        ).to(dev)
        # Seed the STORED pose from the GT PoseNet pose (side-info, not learned).
        wrapper.set_stored_pose(self.scorer.pose_targets[: self.n_pairs].to(dev))
        return wrapper

    def _build_stage_runtime(
        self,
        spec: StageSpec,
        *,
        decoder: nn.Module,
        latents: nn.Parameter,
        ema_decoder: nn.Module | None,
        ema_latents: torch.Tensor | None,
    ) -> _StageRuntime:
        """Build fresh optimizers + cosine schedulers for a stage (faithful: PR95
        resets the optimizer per stage; weights/EMA carry)."""
        if ema_decoder is None:
            ema_decoder = deepcopy(decoder)
        if ema_latents is None:
            ema_latents = latents.data.clone()

        if spec.use_muon:
            muon_params, adamw_params = self.v.partition_params_for_muon(decoder)
            muon_opt = self.v.Muon(
                muon_params,
                lr=spec.muon_lr,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                weight_decay=spec.muon_weight_decay,
            )
            adamw_opt = torch.optim.AdamW(
                [
                    {"params": adamw_params, "lr": spec.adamw_lr},
                    {"params": [latents], "lr": spec.adamw_lr * spec.latent_lr_mult},
                ],
                weight_decay=0.0,
            )
        else:
            muon_opt = None
            muon_params = []
            adamw_params = list(decoder.parameters())
            adamw_opt = torch.optim.AdamW(
                [
                    {"params": decoder.parameters(), "lr": spec.adamw_lr},
                    {"params": [latents], "lr": spec.adamw_lr * spec.latent_lr_mult},
                ],
                weight_decay=0.0,
            )

        eta_min_ratio = max(spec.lr_floor_ratio / spec.adamw_lr, 1e-3)

        def lr_lambda(epoch: int) -> float:
            return max(0.5 * (1 + math.cos(math.pi * epoch / spec.epochs)), eta_min_ratio)

        adamw_sched = torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda)
        muon_sched = (
            torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda)
            if muon_opt is not None
            else None
        )
        return _StageRuntime(
            decoder=decoder,
            latents=latents,
            ema_decoder=ema_decoder,
            ema_latents=ema_latents,
            adamw_opt=adamw_opt,
            muon_opt=muon_opt,
            adamw_sched=adamw_sched,
            muon_sched=muon_sched,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

    # -- one faithful training epoch (1:1 with common.py) --------------------
    def _train_one_epoch(self, rt: _StageRuntime, spec: StageSpec) -> tuple[float, float, float, float]:
        """Run one epoch; returns (mean_loss, mean_pose_mse, last_grad_adamw, last_grad_muon)."""
        decoder, latents = rt.decoder, rt.latents
        n_pairs = self.n_pairs
        bs = spec.batch_size
        # torch.randperm honors the global torch RNG (captured/restored on resume).
        # Build on CPU then move so the RNG draw is device-independent (MPS has its
        # own RNG stream) — this keeps the resume bit-identical regardless of the
        # train device, and keeps the permutation reproducible vs a CPU-arm A/B.
        pair_indices = torch.randperm(n_pairs).to(self.train_device)
        epoch_loss = 0.0
        epoch_pose = 0.0
        nb = 0
        last_gn_adamw = None
        last_gn_muon = None

        for batch_start in range(0, n_pairs, bs):
            idx = pair_indices[batch_start : batch_start + bs]
            B = len(idx)

            if spec.use_qat:
                originals = self.v.apply_qat(decoder)
            # Lever 3: when pose-FiLM is on the decoder is a PoseFiLMHNeRVWrapper
            # whose forward needs the per-pair index to look up the stored pose.
            # When off, the vendored decoder forward takes only the latents.
            decoded_pair = (
                decoder(latents[idx], idx)
                if self.cfg.pose_film_enabled
                else decoder(latents[idx])
            )
            if spec.use_qat:
                self.v.restore_qat(decoder, originals)

            flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
            up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
            down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
            decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)

            decoded_clamped = decoded_bhwc.clamp(0, 255)
            decoded_rounded = decoded_clamped.round()
            decoded_bhwc = decoded_clamped + (decoded_rounded - decoded_clamped).detach()

            rt.adamw_opt.zero_grad()
            if rt.muon_opt is not None:
                rt.muon_opt.zero_grad()

            if self.split_by_head:
                loss_val, pose_mse_val = self._split_by_head_backward(
                    decoded_bhwc, idx, spec
                )
                # The categorical-entropy regularizer does NOT depend on the frames
                # (it reads the decoder weights), so it backprops straight to the
                # decoder — add it as a separate scalar backward (accumulates into
                # the same .grad buffers the frame-cotangent backward populated).
                if spec.cat_lambda > 0:
                    ent = self.v.cat_entropy_v2(
                        decoder, sigma=spec.cat_sigma, sample_size=2000,
                        device=self.train_device,
                    )
                    (spec.cat_lambda * ent).backward()
                    loss_val += float(spec.cat_lambda * ent.item())
            else:
                seg_out, pose_pred6 = self.scorer.seg_pose_forward(decoded_bhwc)

                seg_l = _seg_loss_for_spec(
                    spec, seg_out, self.scorer.seg_targets_hard[idx]
                )
                pose_mse = F.mse_loss(pose_pred6, self.scorer.pose_targets[idx])
                pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)

                loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
                if spec.cat_lambda > 0:
                    ent = self.v.cat_entropy_v2(
                        decoder, sigma=spec.cat_sigma, sample_size=2000,
                        device=self.train_device,
                    )
                    loss = loss + spec.cat_lambda * ent
                loss.backward()
                loss_val = float(loss.item())
                pose_mse_val = float(pose_mse.item())
            gn_adamw = torch.nn.utils.clip_grad_norm_(
                [*rt.adamw_params, latents], spec.grad_clip
            )
            last_gn_adamw = float(gn_adamw)
            if rt.muon_opt is not None and spec.grad_clip_muon is not None:
                gn_muon = torch.nn.utils.clip_grad_norm_(rt.muon_params, spec.grad_clip_muon)
                last_gn_muon = float(gn_muon)
            rt.adamw_opt.step()
            if rt.muon_opt is not None:
                rt.muon_opt.step()

            # EMA update after each step (the EMA non-negotiable; faithful position).
            self.v.ema_update(
                rt.ema_decoder, decoder, rt.ema_latents, latents, decay=spec.ema_decay
            )

            epoch_loss += loss_val
            epoch_pose += pose_mse_val
            nb += 1

        rt.adamw_sched.step()
        if rt.muon_sched is not None:
            rt.muon_sched.step()
        return epoch_loss / max(nb, 1), epoch_pose / max(nb, 1), last_gn_adamw, last_gn_muon

    # -- split-by-head combined-gradient backward (the pose-axis salvage) -----
    def _split_by_head_backward(
        self, decoded_bhwc: torch.Tensor, idx: torch.Tensor, spec: StageSpec
    ) -> tuple[float, float]:
        """Compute the COMBINED frame-gradient (SegNet path on the fast train device,
        PoseNet path on the CPU authority) and inject it into the decoder graph via a
        single ``decoded_bhwc.backward(gradient=combined_cotangent)``.

        The math (why this is descent-equivalent on BOTH terms BY CONSTRUCTION):
        the full loss ``L = w_seg * seg_l(F) + w_pose * pose_l(F)`` is a sum of two
        terms that each depend on the SAME frame tensor ``F = decoded_bhwc``. By the
        chain rule, ``dL/dF = w_seg * d(seg_l)/dF + w_pose * d(pose_l)/dF`` — the two
        per-head frame cotangents SUM at the frame tensor, and ``dL/dtheta`` (decoder)
        follows from ``dL/dF`` by the SAME vjp regardless of how dL/dF was assembled.
        So we:
          1. detach two frame leaves that share F's values (one on train_device for
             SegNet, one on the CPU authority for PoseNet),
          2. backprop each head's weighted loss to its own leaf (``leaf.grad`` = that
             head's frame cotangent),
          3. sum the two cotangents on the train device (move the CPU pose cotangent
             over — a value transfer, the gradient is already computed on the CPU
             authority PoseNet so it carries ZERO MPS pose drift),
          4. ``decoded_bhwc.backward(gradient=combined)`` — flows the exact combined
             gradient into the decoder.

        The SegNet cotangent is the (validated bit-identical on d_seg) MPS gradient;
        the PoseNet cotangent is the CPU AUTHORITY gradient (zero drift). Returns
        ``(loss_value, pose_mse_value)`` for telemetry (recomputed from the detached
        head losses; not authority — telemetry only)."""
        # --- SegNet path on the TRAIN device (MPS) ---
        frames_seg = decoded_bhwc.detach().requires_grad_(True)  # train_device leaf
        seg_out = self.scorer.seg_forward_train(frames_seg)
        seg_targets = self.scorer.seg_targets_hard[idx]
        seg_l = _seg_loss_for_spec(spec, seg_out, seg_targets)
        (spec.seg_weight * seg_l).backward()
        cot_seg = frames_seg.grad  # d(w_seg*seg_l)/dF on train_device

        # --- PoseNet path on the CPU AUTHORITY (zero pose drift) ---
        frames_pose = (
            decoded_bhwc.detach().to(self.device).requires_grad_(True)
        )  # authority-device leaf, same values
        pose_pred6 = self.scorer.pose_forward_authority(frames_pose)
        pose_targets = self._pose_targets_authority()[idx.to(self.device)]
        pose_mse = F.mse_loss(pose_pred6, pose_targets)
        pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
        (spec.pose_weight * pose_l).backward()
        cot_pose = frames_pose.grad  # d(w_pose*pose_l)/dF on the CPU authority

        # --- sum the two cotangents at the frame tensor + inject into the decoder ---
        combined = cot_seg + cot_pose.to(cot_seg.device)
        decoded_bhwc.backward(gradient=combined)

        loss_val = float((spec.seg_weight * seg_l).item()) + float(
            (spec.pose_weight * pose_l).item()
        )
        return loss_val, float(pose_mse.item())

    def _pose_targets_authority(self) -> torch.Tensor:
        """The pose targets on the AUTHORITY device (cached). For split-by-head the
        PoseNet path runs on the CPU authority, so its targets must live there too
        (the scorer holds them on train_device for the non-split path)."""
        cached = getattr(self, "_pose_targets_authority_cache", None)
        if cached is None or cached.device != self.device:
            cached = self.scorer.pose_targets.to(self.device)
            self._pose_targets_authority_cache = cached
        return cached

    # -- checkpoint state capture / restore ----------------------------------
    def _capture_state(self, rt: _StageRuntime, spec: StageSpec) -> dict[str, Any]:
        return {
            "decoder": {k: v.detach().cpu().clone() for k, v in rt.decoder.state_dict().items()},
            "latents": rt.latents.detach().cpu().clone(),
            "ema_decoder": {k: v.detach().cpu().clone() for k, v in rt.ema_decoder.state_dict().items()},
            "ema_latents": rt.ema_latents.detach().cpu().clone(),
            "adamw": rt.adamw_opt.state_dict(),
            "muon": rt.muon_opt.state_dict() if rt.muon_opt is not None else None,
            "adamw_sched": rt.adamw_sched.state_dict(),
            "muon_sched": rt.muon_sched.state_dict() if rt.muon_sched is not None else None,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "base_channels": self.cfg.base_channels,
            "latent_dim": self.cfg.latent_dim,
            "n_pairs": self.n_pairs,
            "stage_name": spec.name,
            "ema_decay": spec.ema_decay,
            "best_score": self.best_score,
            "best_ep": self.best_ep,
            "best_stage": self.best_stage,
        }

    def _restore_into(self, rt: _StageRuntime, merged: dict[str, Any]) -> None:
        # The TRAIN runtime (decoder/latents/EMA shadow) lives on the train device
        # (the gradient backend, possibly MPS); the checkpoint is CPU-resident.
        td = self.train_device
        rt.decoder.load_state_dict({k: v.to(td) for k, v in merged["decoder"].items()})
        rt.latents.data = merged["latents"].to(td)
        rt.ema_decoder.load_state_dict(
            {k: v.to(td) for k, v in merged["ema_decoder"].items()}
        )
        rt.ema_latents = merged["ema_latents"].to(td)
        # Optimizer/scheduler restore is SKIPPED for a FORK-SEED checkpoint (the
        # Lever-3 A/B seeds the WEIGHTS + EMA from the immutable basin forkpoint but
        # cannot transfer the AdamW/Muon momentum across an architecture change —
        # FiLM adds new params, so the param groups differ). A fork-seed checkpoint
        # sets ``adamw=None`` to signal "fresh optimizers from these weights at this
        # curriculum position". Production checkpoints ALWAYS carry optimizer state,
        # so the default (full) resume is unchanged.
        if merged.get("adamw") is not None:
            rt.adamw_opt.load_state_dict(merged["adamw"])
            if rt.muon_opt is not None and merged.get("muon") is not None:
                rt.muon_opt.load_state_dict(merged["muon"])
            rt.adamw_sched.load_state_dict(merged["adamw_sched"])
            if rt.muon_sched is not None and merged.get("muon_sched") is not None:
                rt.muon_sched.load_state_dict(merged["muon_sched"])
        # RNG restore (so the next epoch's randperm matches the uninterrupted run).
        from tac.torch_vehicle.checkpoint import restore_rng

        restore_rng(merged)

    # -- eval + BEST tracking (EMA shadow is the export bytes) ---------------
    def _snapshot_ema(self, rt: _StageRuntime) -> _EvalSnapshot:
        """Capture a POINT-IN-TIME CPU copy of the EMA shadow the eval will score.

        This is the ONLY part of the eval that touches the live training runtime,
        and it runs in the MAIN thread (cheap — a state_dict + latents deep-copy to
        CPU). After it returns, training may keep mutating the live (MPS) EMA shadow
        without racing the eval: the eval operates exclusively on this immutable
        snapshot. The async path snapshots HERE (main thread), then hands the
        snapshot to a background worker; the sync path snapshots + evals inline. The
        eval math (:meth:`_eval_snapshot`) is IDENTICAL for both — so the authority
        numbers are bit-for-bit the same (the no-regression guarantee)."""
        ema_sd = {
            k: v.detach().cpu().clone() for k, v in rt.ema_decoder.state_dict().items()
        }
        ema_latents = rt.ema_latents.detach().cpu().clone()
        return _EvalSnapshot(ema_sd=ema_sd, ema_latents=ema_latents)

    def _build_archive_and_eval_decoder(
        self,
        ema_sd: dict[str, torch.Tensor],
        ema_latents: torch.Tensor,
        meta_dict: dict[str, Any],
    ) -> tuple[bytes, nn.Module, torch.Tensor]:
        """Build the byte-closed archive from the EMA shadow + return the PARSE-BACK
        eval decoder + parsed latents (on the AUTHORITY device).

        DEFAULT (no FiLM): the vendored 3-section archive + a vendored decoder
        rebuilt from the int8-dequantized parse-back — UNCHANGED from the legacy
        path (byte-identical).

        Lever 3 (``pose_film_enabled``): the EMA shadow is the WRAPPER state dict
        (``decoder.*`` + ``pose_film.*`` + ``stored_pose``). We (1) split the FiLM
        weights into the codec-compatible decoder blob (bare vendored keys + the
        ``pose_film.*`` keys) and the ``stored_pose`` buffer into the ADDITIVE pose
        section, (2) build ``vendored_archive + encode_pose_section`` (pristine
        codec untouched), (3) parse BOTH back and rebuild the FiLM wrapper + a
        cursor-based eval adapter so the exact eval renders the SAME FiLM-conditioned
        frames the inflate path produces (byte-closed faithful)."""
        if not self.cfg.pose_film_enabled:
            archive = self.v.build_archive(ema_sd, ema_latents, meta_dict=meta_dict)
            eval_decoder_sd, eval_latents, _meta = self.v.parse_archive(archive)
            eval_dec = self._new_vendored_decoder(device=self.device)
            eval_dec.load_state_dict(
                {k: v.to(self.device) for k, v in eval_decoder_sd.items()}
            )
            eval_dec.eval()
            return archive, eval_dec, eval_latents

        from tac.torch_vehicle.pose_film import (
            PoseFiLMHNeRVWrapper,
            _FiLMEvalDecoder,
            build_archive_with_pose,
            parse_pose_section,
            wrapper_sd_to_archive_decoder_sd,
        )

        # (1) Split the wrapper state dict: codec decoder blob (vendored + FiLM
        # weights) and the stored-pose buffer (additive pose section).
        archive_decoder_sd = wrapper_sd_to_archive_decoder_sd(ema_sd)
        stored_pose = ema_sd["stored_pose"]
        # (2) Build the additive archive (pristine vendored build_archive + pose).
        archive = build_archive_with_pose(
            self.v.build_archive, archive_decoder_sd, ema_latents, meta_dict, stored_pose
        )
        # (3) Parse BOTH sections back and rebuild the FiLM wrapper for eval.
        eval_decoder_sd, eval_latents, _meta = self.v.parse_archive(archive)
        parsed_pose = parse_pose_section(archive, self.v.parse_archive)
        film_sd = {
            k[len("pose_film.") :]: v
            for k, v in eval_decoder_sd.items()
            if k.startswith("pose_film.")
        }
        dec_sd = {
            k: v for k, v in eval_decoder_sd.items() if not k.startswith("pose_film.")
        }
        vendored = self._new_vendored_decoder(device=self.device)
        vendored.load_state_dict({k: v.to(self.device) for k, v in dec_sd.items()})
        wrapper = PoseFiLMHNeRVWrapper(
            vendored,
            n_pairs=self.n_pairs,
            film_hidden=self.cfg.pose_film_hidden,
        ).to(self.device)
        wrapper.pose_film.load_state_dict(
            {k: v.to(self.device) for k, v in film_sd.items()}
        )
        if parsed_pose is not None:
            wrapper.set_stored_pose(parsed_pose.to(self.device))
        wrapper.eval()
        eval_dec = _FiLMEvalDecoder(wrapper)
        eval_dec.eval()  # resets the cursor to pair 0
        return archive, eval_dec, eval_latents

    def _eval_snapshot(
        self, snap: _EvalSnapshot, spec: StageSpec, stage_index: int, snapshot_epoch: int
    ) -> dict[str, Any]:
        """Build the archive from the SNAPSHOT EMA shadow, exact-eval the PARSE-BACK
        (int8-dequantized) decoder + latents, track BEST, and RECORD the telemetry
        row — all keyed to ``snapshot_epoch`` (the epoch the snapshot was taken at).

        FIDELITY (1:1 with vendored ``common.py:228-238``): the score that picks
        the BEST checkpoint is the score of the **archive the contest sees**, i.e.
        the int8-quantized decoder + delta-coded latents AFTER ``parse_archive``,
        NOT the raw float EMA shadow (which over-estimates quality and would pick
        the wrong checkpoint). We reconstruct the eval decoder from the parse-back
        ``state_dict`` and eval THAT.

        THREAD-SAFETY: this method may run in a BACKGROUND thread (async path) or
        inline (sync path). The pure-eval part (build_archive / parse_archive /
        exact_eval) touches NO shared state — it reads only the immutable snapshot.
        The shared surfaces (best-tracker fields, the ``best/`` dir write, and the
        single non-thread-safe :class:`TelemetryWriter`) are mutated ONLY under
        ``self._eval_lock`` so they never race the main loop's non-eval records.

        Returns the eval dict (d_seg / d_pose / rate / score / archive_bytes /
        is_best). The BEST EMA shadow + archive are written to ``out_dir/best/``.
        """
        ema_sd = snap.ema_sd
        meta_dict = {
            "n_pairs": self.n_pairs,
            "latent_dim": self.cfg.latent_dim,
            "base_channels": self.cfg.base_channels,  # FINDING: base_ch in meta (was 36)
            "eval_size": [_EVAL_H, _EVAL_W],
        }
        archive, eval_dec, eval_latents = self._build_archive_and_eval_decoder(
            ema_sd, snap.ema_latents, meta_dict
        )
        archive_bytes = len(archive)
        # Parse-back: the int8-dequantized decoder + delta-decoded latents — the
        # contest-visible artifact (faithful to common.py). The exact eval runs on
        # the AUTHORITY device (CPU-TRUSTED / CUDA), NEVER the train device — even
        # when training on MPS the score that picks BEST is the CPU authority
        # (CLAUDE.md "MPS auth eval is NOISE"). So the eval decoder is built on
        # self.device and the scorer's exact_eval routes through its authority net.
        ev = self.scorer.exact_eval(eval_dec, eval_latents.to(self.device), archive_bytes)
        score = float(ev["score"])
        d_seg = ev.get("seg_distortion", ev.get("d_seg"))
        d_pose = ev.get("pose_distortion", ev.get("d_pose"))
        rate = ev.get("rate")
        # --- shared-state mutation under the lock (BEST tracker + telemetry) ------
        with self._eval_lock:
            is_best = score < self.best_score
            if is_best:
                self.best_score = score
                self.best_ep = snapshot_epoch
                self.best_stage = stage_index
                best_dir = self.cfg.out_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                (best_dir / "best_archive.bin").write_bytes(archive)
                torch.save(ema_sd, best_dir / "best_ema_decoder.pt")
                torch.save(snap.ema_latents, best_dir / "best_ema_latents.pt")
                (best_dir / "best_meta.json").write_text(
                    _json_dumps(
                        {
                            "stage_index": stage_index,
                            "stage_name": spec.name,
                            "global_epoch": snapshot_epoch,
                            "score": score,
                            "d_seg": d_seg,
                            "d_pose": d_pose,
                            "rate": rate,
                            "archive_bytes": archive_bytes,
                            "base_channels": self.cfg.base_channels,
                            "authority": "[contest-CPU advisory] NON-PROMOTABLE",
                        }
                    )
                )
        return {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "rate": rate,
            "score": score,
            "archive_bytes": archive_bytes,
            "is_best": is_best,
        }

    def _record_eval_row(
        self,
        ev: dict[str, Any],
        spec: StageSpec,
        stage_index: int,
        snapshot_epoch: int,
    ) -> None:
        """Append the EVAL telemetry row, tagged with ``snapshot_epoch`` (the epoch
        the eval's snapshot came from — which may be a few epochs behind the current
        training epoch on the async path). Written under ``_eval_lock`` because the
        single :class:`TelemetryWriter` is not thread-safe and the main loop also
        records non-eval rows."""
        with self._eval_lock:
            self.telemetry.record(
                EpochRecord(
                    stage_index=stage_index,
                    stage_name=spec.name,
                    epoch_in_stage=snapshot_epoch
                    - sum(self.curriculum[i].epochs for i in range(stage_index)),
                    global_epoch=snapshot_epoch,
                    loss=float("nan"),
                    pose_mse=float("nan"),
                    adamw_lr=float("nan"),
                    muon_lr=None,
                    evaluated=True,
                    d_seg=ev.get("d_seg"),
                    d_pose=ev.get("d_pose"),
                    rate=ev.get("rate"),
                    score=ev.get("score"),
                    archive_bytes=ev.get("archive_bytes"),
                    is_best=ev.get("is_best", False),
                    extra={"async_eval_row": True, "snapshot_epoch": snapshot_epoch},
                )
            )

    # -- async eval scheduling (the throughput salvage) ----------------------
    def _async_eval_in_flight(self) -> bool:
        """True iff a background eval worker is currently running."""
        return self._eval_thread is not None and self._eval_thread.is_alive()

    def _schedule_async_eval(
        self, rt: _StageRuntime, spec: StageSpec, stage_index: int, snapshot_epoch: int
    ) -> bool:
        """Snapshot the EMA shadow NOW (main thread) and spawn ONE background worker
        to run the IDENTICAL exact eval off that snapshot, then return immediately so
        training continues. At most ONE eval is in-flight: if a prior worker is still
        running, SKIP (log + count) — the cadence self-throttles (a 13-min eval over-
        runs the 150s eval interval early; once training-only epochs are cheap the
        eval naturally lands within an interval). Returns True if scheduled, False if
        skipped."""
        if self._async_eval_in_flight():
            self._skipped_evals += 1
            print(
                f"[async-eval] SKIP @ global_ep={snapshot_epoch}: prior eval "
                f"(snapshot_ep={self._inflight_snapshot_epoch}) still running "
                f"(total skipped={self._skipped_evals})",
                flush=True,
            )
            return False
        snap = self._snapshot_ema(rt)  # cheap, main-thread, point-in-time
        self._inflight_snapshot_epoch = snapshot_epoch

        def _worker() -> None:
            t0 = time.time()
            try:
                ev = self._eval_snapshot(snap, spec, stage_index, snapshot_epoch)
                self._record_eval_row(ev, spec, stage_index, snapshot_epoch)
                print(
                    f"[async-eval] DONE snapshot_ep={snapshot_epoch} "
                    f"score={ev['score']:.5f} d_seg={ev['d_seg']} d_pose={ev['d_pose']} "
                    f"({time.time() - t0:.0f}s){' *BEST*' if ev['is_best'] else ''}",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                # An eval failure must NOT take down the training loop (it runs in a
                # daemon thread off the main loop). Log loudly; do NOT write a
                # telemetry row (a failed eval has no authority numbers — a row would
                # be misleading). Training continues; the next eval epoch re-schedules
                # off a fresh snapshot.
                print(
                    f"[async-eval] FAILED snapshot_ep={snapshot_epoch}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        self._eval_thread = threading.Thread(
            target=_worker, name=f"async-eval-ep{snapshot_epoch}", daemon=True
        )
        self._eval_thread.start()
        return True

    def _join_async_eval(self, timeout: float | None = None) -> None:
        """JOIN any in-flight eval worker so the final BEST + last eval row land
        before the run exits (the DONE-marker contract). Called at run completion."""
        if self._eval_thread is not None and self._eval_thread.is_alive():
            print("[async-eval] JOIN: waiting for in-flight eval to finish...", flush=True)
            self._eval_thread.join(timeout=timeout)
        self._eval_thread = None

    # -- the resumable run ----------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Run (or resume) the full curriculum. Idempotent on a DONE marker.

        Resume contract: if ``out_dir`` holds a checkpoint, the run restores the
        complete state and continues from ``(stage_index, epoch_in_stage)``; a
        death costs at most ``checkpoint_every_epochs`` epochs. On completion a
        DONE marker is written (marker-on-exit).
        """
        from tac.torch_vehicle.checkpoint import is_done

        if is_done(self.cfg.out_dir):
            return {"status": "already_done", "best_score": self.best_score}

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Resume position (default: start of stage 0).
        resume_pos = TorchCheckpointPosition(0, 0)
        merged: dict[str, Any] | None = None
        if checkpoint_exists(self.cfg.out_dir):
            man = read_manifest(self.cfg.out_dir)
            if int(man["n_pairs"]) != self.n_pairs:
                raise ValueError(
                    f"checkpoint n_pairs={man['n_pairs']} != trainer n_pairs={self.n_pairs}; "
                    "cannot resume a different basis"
                )
            if int(man["base_channels"]) != self.cfg.base_channels:
                raise ValueError(
                    f"checkpoint base_channels={man['base_channels']} != "
                    f"cfg.base_channels={self.cfg.base_channels}; cannot resume a different basis"
                )
            merged = load_checkpoint(self.cfg.out_dir, map_location=self.cfg.device)
            resume_pos = merged["position"]
            self.best_score = float(man["best_score"])
            self.best_ep = int(man["best_ep"])
            self.best_stage = int(man["best_stage"])

        # Carry decoder/latents/EMA across stages.
        carry_decoder: nn.Module | None = None
        carry_latents: nn.Parameter | None = None
        carry_ema_decoder: nn.Module | None = None
        carry_ema_latents: torch.Tensor | None = None

        # Recompute the global-epoch base for stages already completed.
        self._global_epoch = sum(
            self.curriculum[i].epochs for i in range(resume_pos.stage_index)
        ) + resume_pos.epoch_in_stage

        for stage_index in range(resume_pos.stage_index, len(self.curriculum)):
            spec = self.curriculum[stage_index]
            start_epoch = resume_pos.epoch_in_stage if stage_index == resume_pos.stage_index else 0

            # Build decoder/latents for this stage (carry from prior stage, or init).
            resuming_into_this_stage = (
                merged is not None and stage_index == resume_pos.stage_index
            )
            if carry_decoder is None:
                decoder = self._new_decoder()  # train-device (gradient backend)
                if spec.init_latents_random:
                    # Draw on CPU (deterministic via the global seed) then move to the
                    # train device — MPS has its own RNG stream, so a CPU draw keeps
                    # the init reproducible vs a CPU-arm descent-equivalence A/B.
                    latents = nn.Parameter(
                        (torch.randn(self.n_pairs, self.cfg.latent_dim) * 0.1).to(
                            self.train_device
                        )
                    )
                elif resuming_into_this_stage:
                    # The latents will be overwritten by ``_restore_into`` below; a
                    # placeholder of the right shape is sufficient (faithful: the
                    # vendored loop loads ``final_latents.pt`` here — we load from
                    # the checkpoint).
                    latents = nn.Parameter(
                        torch.zeros(self.n_pairs, self.cfg.latent_dim, device=self.train_device)
                    )
                else:
                    # Fail closed (faithful to vendored common.py:122): a non-stage-1
                    # stage with neither an in-memory carry NOR a resume checkpoint
                    # has no latents to start from — this is a misconfigured run, not
                    # a silent zeros-init.
                    raise ValueError(
                        f"stage {spec.name} (index {stage_index}) has init_latents_random="
                        "False but no prior-stage carry and no resume checkpoint to load "
                        "latents from — cannot start a non-stage-1 stage from scratch."
                    )
            else:
                decoder = carry_decoder
                latents = carry_latents  # type: ignore[assignment]

            rt = self._build_stage_runtime(
                spec,
                decoder=decoder,
                latents=latents,
                ema_decoder=carry_ema_decoder,
                ema_latents=carry_ema_latents,
            )

            # If resuming INTO this stage, restore the mid-stage state now.
            if merged is not None and stage_index == resume_pos.stage_index:
                self._restore_into(rt, merged)
                merged = None  # consumed

            for epoch in range(start_epoch, spec.epochs):
                mean_loss, mean_pose_mse, gn_adamw, gn_muon = self._train_one_epoch(rt, spec)
                self._global_epoch += 1
                epoch_in_stage = epoch + 1  # 1-based completed

                # Eval cadence (faithful: eval_every).
                evaluated = (epoch_in_stage % spec.eval_every) == 0
                ev: dict[str, Any] = {}
                # SYNC path (default; byte-identical to the legacy run): eval inline
                # off a snapshot and emit ONE combined train+eval row. The snapshot
                # round-trip is a no-op vs reading rt directly — same archive, same
                # exact numbers — so the sync output is unchanged.
                # ASYNC path (--async-eval): record the TRAIN-ONLY row now and SPAWN
                # the (identical) eval off a point-in-time snapshot in a background
                # thread; the eval row lands LATER, tagged with this snapshot epoch.
                # Training continues immediately — the ~13-min CPU eval no longer
                # blocks the MPS loop.
                if evaluated and not self._async_eval:
                    snap = self._snapshot_ema(rt)
                    ev = self._eval_snapshot(snap, spec, stage_index, self._global_epoch)

                self.telemetry.record(
                    EpochRecord(
                        stage_index=stage_index,
                        stage_name=spec.name,
                        epoch_in_stage=epoch_in_stage,
                        global_epoch=self._global_epoch,
                        loss=mean_loss,
                        pose_mse=mean_pose_mse,
                        adamw_lr=float(rt.adamw_opt.param_groups[0]["lr"]),
                        muon_lr=(
                            float(rt.muon_opt.param_groups[0]["lr"])
                            if rt.muon_opt is not None
                            else None
                        ),
                        grad_norm_adamw=gn_adamw,
                        grad_norm_muon=gn_muon,
                        # In ASYNC mode this row is train-only (the eval row is
                        # emitted separately by the worker, tagged with the snapshot
                        # epoch); ``evaluated`` here reflects the SYNC inline eval.
                        evaluated=evaluated and not self._async_eval,
                        d_seg=ev.get("d_seg"),
                        d_pose=ev.get("d_pose"),
                        rate=ev.get("rate"),
                        score=ev.get("score"),
                        archive_bytes=ev.get("archive_bytes"),
                        is_best=ev.get("is_best", False),
                    )
                )

                # ASYNC eval: schedule AFTER the train row is recorded so the
                # snapshot is taken at this epoch's completed EMA shadow. At most one
                # in-flight; over-cadence evals self-throttle (skip + log).
                if evaluated and self._async_eval:
                    self._schedule_async_eval(rt, spec, stage_index, self._global_epoch)

                # Checkpoint cadence (a death costs <= this many epochs).
                if (epoch_in_stage % self.cfg.checkpoint_every_epochs) == 0:
                    self._checkpoint(rt, spec, stage_index, epoch_in_stage)

                # Test-only simulated death: raise AFTER the checkpoint has landed
                # so a resumed run continues from this exact point. Production
                # never sets this; a real death (SIGKILL/OOM/preempt) is the
                # equivalent and is covered by the subprocess kill+restart test.
                if (
                    self._stop_after_global_epoch is not None
                    and self._global_epoch >= self._stop_after_global_epoch
                ):
                    raise _SimulatedDeath(self._global_epoch)

            # End of stage: ensure a checkpoint at the boundary + carry forward.
            self._checkpoint(rt, spec, stage_index, spec.epochs)
            carry_decoder = rt.decoder
            carry_latents = rt.latents
            carry_ema_decoder = rt.ema_decoder
            carry_ema_latents = rt.ema_latents
            resume_pos = TorchCheckpointPosition(stage_index + 1, 0)

        # JOIN any in-flight async eval so the final BEST + last eval row land
        # before the DONE marker (the marker-on-exit contract). No-op in sync mode.
        self._join_async_eval()

        summary = {
            "status": "complete",
            "best_score": self.best_score,
            "best_ep": self.best_ep,
            "best_stage": self.best_stage,
            "n_stages": len(self.curriculum),
            "base_channels": self.cfg.base_channels,
            "async_eval": self._async_eval,
            "skipped_async_evals": self._skipped_evals,
            "authority": "[contest-CPU advisory] NON-PROMOTABLE — exact via upstream/evaluate.py",
        }
        write_done_marker(self.cfg.out_dir, summary)
        return summary

    def _checkpoint(self, rt: _StageRuntime, spec: StageSpec, stage_index: int, epoch_in_stage: int) -> None:
        state = self._capture_state(rt, spec)
        save_checkpoint(
            state,
            self.cfg.out_dir,
            TorchCheckpointPosition(stage_index, epoch_in_stage),
        )


@dataclass
class VendoredBundle:
    """The vendored PR95 primitives the driver re-drives (NO source edits).

    Built via :func:`import_vendored_bundle`; kept as an explicit handle so the
    driver's vendored-dependency surface is auditable in one place.
    """

    HNeRVDecoder: type
    Muon: type
    partition_params_for_muon: Callable
    ema_update: Callable
    apply_qat: Callable
    restore_qat: Callable
    cat_entropy_v2: Callable
    build_archive: Callable
    parse_archive: Callable


def import_vendored_bundle() -> VendoredBundle:
    """Import the vendored PR95 primitives WITHOUT the challenge-scorer dependency.

    ``model`` / ``optim`` / ``losses`` / ``codec`` import cleanly (no challenge
    repo needed); only ``data`` / ``score`` pull in the frozen scorer (those are
    used by :class:`RealScorerContext`, not here). So this bundle is importable on
    a host without the challenge weights — the resume/checkpoint machinery is
    architecture-agnostic.
    """
    from tac.torch_vehicle.vendored_imports import import_vendored

    model = import_vendored("model")
    optim = import_vendored("optim")
    losses = import_vendored("losses")
    codec = import_vendored("codec")
    return VendoredBundle(
        HNeRVDecoder=model.HNeRVDecoder,
        Muon=optim.Muon,
        partition_params_for_muon=optim.partition_params_for_muon,
        ema_update=losses.ema_update,
        apply_qat=losses.apply_qat,
        restore_qat=losses.restore_qat,
        cat_entropy_v2=losses.cat_entropy_v2,
        build_archive=codec.build_archive,
        parse_archive=codec.parse_archive,
    )


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


__all__ = [
    "ScorerContext",
    "TorchVehicleConfig",
    "TorchVehicleDriver",
    "VendoredBundle",
    "import_vendored_bundle",
]
