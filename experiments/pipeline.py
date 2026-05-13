#!/usr/bin/env python3
"""Canonical compression pipeline: video -> trained renderer -> optimized archive.

Production-grade pipeline for the comma.ai video compression challenge.
Takes a video and a trained renderer checkpoint as input, produces an
optimized submission archive through a multi-step coordinate descent on
the rate-distortion surface.

Pipeline steps per iteration:
    1. Export: float checkpoint -> quantized renderer.bin
    2. Pose TTO: adaptive gradient-based pose optimization (converges to tolerance)
    3. QAT: quantization-aware fine-tuning (monitors quality degradation)
    4. Fridrich refinement: steganalytic polish with texture/L-inf/Markov losses
    5. Weight compression: int4+LZMA2 or FP4 (configurable, auto mode available)
    6. Archive: build submission zip with masks + renderer + poses
    7. Eval: full end-to-end auth evaluation through upstream scorer

The pipeline is idempotent: each step writes a .done marker, and re-running
resumes from the last completed step. Iterative refinement continues until
score improvement falls below a convergence tolerance.

Usage:
    # Compress a single video:
    PYTHONPATH=src:upstream python experiments/pipeline.py compress \\
        --video upstream/videos/0.mkv \\
        --checkpoint path/to/distill_phase2_best.pt \\
        --device cuda --output-dir results/submission_v1

    # Compress with all optimizations:
    PYTHONPATH=src:upstream python experiments/pipeline.py compress \\
        --video upstream/videos/0.mkv \\
        --checkpoint path/to/distill_phase2_best.pt \\
        --device cuda --output-dir results/submission_v1 \\
        --half-frame --binary-poses --brotli --weight-compression auto

    # Evaluate an existing archive:
    PYTHONPATH=src:upstream python experiments/pipeline.py eval \\
        --archive results/submission_v1/archive.zip \\
        --video upstream/videos/0.mkv --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pydantic import BaseModel
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn


# ── Constants ────────────────────────────────────────────────────────────

ORIGINAL_VIDEO_BYTES = 37_545_489
NUM_FRAMES = 1200
SEGNET_H, SEGNET_W = 384, 512


# ── Logging ──────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "pipeline") -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _banner(msg: str) -> None:
    w = max(len(msg) + 4, 60)
    _log("=" * w)
    _log(f"  {msg}")
    _log("=" * w)


# ── Subprocess helper (DX top-1, council 5/0 approved 2026-04-26) ─────────
#
# Bug class this prevents: 5 of 6 subprocess.run() call sites previously
# inherited the parent's stdout/stderr (no capture). When a child crashed
# with a 200-line traceback, the pipeline only saw `exit N` — no log file,
# no replay command. This helper captures stderr + stdout to disk, surfaces
# the last 20 lines on failure, and prints the full shell-quoted command
# for replay. Apply to every new subprocess.run call.

def _run_step(
    cmd: list[str],
    step_name: str,
    iter_dir: Path,
    timeout: int | None = None,
    env: dict | None = None,
) -> int:
    """Subprocess wrapper that captures stderr to <iter_dir>/<step_name>.stderr.log
    and surfaces last 20 lines on failure with the cmd string for replay.

    Returns the subprocess returncode (0 on success). Raises
    subprocess.TimeoutExpired on timeout (after logging it). Caller is
    responsible for converting non-zero returns into RuntimeError.
    """
    iter_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = iter_dir / f"{step_name}.stderr.log"
    stdout_log = iter_dir / f"{step_name}.stdout.log"
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    _log(f"[{step_name}] $ {cmd_str}")
    try:
        with stderr_log.open("w") as err_f, stdout_log.open("w") as out_f:
            result = subprocess.run(  # subprocess-no-check-OK: returncode validated below at line 118; caller converts non-zero to RuntimeError per docstring
                cmd, stdout=out_f, stderr=err_f,
                timeout=timeout, env=env,
            )
    except subprocess.TimeoutExpired:
        _log(f"[{step_name}] TIMEOUT after {timeout}s. Cmd: {cmd_str}", "ERROR")
        raise
    if result.returncode != 0:
        try:
            tail = stderr_log.read_text().splitlines()[-20:]
            for line in tail:
                _log(f"[{step_name}] stderr: {line}", "ERROR")
        except OSError:
            pass
        _log(
            f"[{step_name}] FAILED (exit {result.returncode}). "
            f"Full stderr: {stderr_log}. Replay: {cmd_str}",
            "ERROR",
        )
    return result.returncode


# ── Resume support ───────────────────────────────────────────────────────

def _step_done(output_dir: Path, step: str) -> bool:
    return (output_dir / f".done_{step}").exists()


def _mark_done(output_dir: Path, step: str, meta: dict | None = None) -> None:
    """Atomically write a .done_<step> marker.

    R29 fix: write to a tempfile then os.replace — guarantees a concurrent
    reader sees either the OLD marker or the COMPLETE NEW marker, never a
    partial JSON. Path.write_text is non-atomic on POSIX (open+write+close);
    a concurrent process could read mid-write and treat partial JSON as a
    complete .done marker.
    """
    info = {"step": step, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if meta:
        info.update(meta)
    target = output_dir / f".done_{step}"
    tmp = output_dir / f".done_{step}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, target)


# ── Pipeline configuration ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Production pipeline configuration.

    Architecture defaults match ``tac.renderer.ArchConfig`` (the single source
    of truth).  ``padding_mode`` is overridden to "replicate" because Yousfi's
    analysis showed that zeros creates boundary artifacts.
    """

    # Input
    video: str = ""
    checkpoint: str = ""
    masks: str = ""  # FULL masks (1200 frames) for training/pose TTO. Extracted from video if empty.
    masks_archive: str = ""  # HALF-FRAME masks (600 frames) for archive. Built from full masks if empty.
    output_dir: str = "experiments/results/pipeline"
    device: str = "cuda"
    upstream: str = "upstream"

    # Architecture — defaults from ArchConfig (see tac.renderer.ArchConfig)
    base_ch: int = 36       # ArchConfig default: 36  (was 20 — BUG: diverged from DistillConfig)
    mid_ch: int = 60        # ArchConfig default: 60  (was 28 — BUG: diverged from DistillConfig)
    motion_hidden: int = 32  # ArchConfig default: 32
    depth: int = 1          # ArchConfig default: 1
    pose_dim: int = 6       # ArchConfig default: 6
    embed_dim: int = 6      # ArchConfig default: 6
    use_dsconv: bool = False  # ArchConfig default: False (was True — BUG: diverged from DistillConfig)
    padding_mode: str = "replicate"  # ArchConfig default: "zeros" — override: boundary artifact avoidance
    use_dilation: bool = False  # ArchConfig default: False
    use_zoom_flow: bool = False  # GREEN profile: MotionPredictor 4ch (gate+residual), flow from RadialZoomWarp

    # Training discipline / loss modulators (passed through to train_distill.py)
    use_swa: bool = False
    use_per_class_weights: bool = False
    freeze_motion_phase2: bool = False
    freeze_renderer_phase3: bool = False
    beneficial_quant_noise: bool = False
    # Yousfi #3: spatially-adaptive quantization noise (UNIWARD-aligned)
    use_variance_noise: bool = False
    variance_noise_weight: float = 0.1
    variance_noise_base_std: float = 2.0
    variance_noise_kernel: int = 8
    variance_noise_mode: str = "variance"
    # Yousfi #5: ScanNet-style spatial uncertainty maps
    use_uncertainty_loss: bool = False
    uncertainty_loss_weight: float = 0.1
    uncertainty_loss_floor: float = 0.1
    # Yousfi council R2 CRITICAL (2026-04-26): step_engineered_corrections
    # ran UNCONDITIONALLY on every pipeline run, costing 5-30 min compute
    # AND silently bundling Fridrich-VETOED gradient_corrections.bin into
    # archives. Now opt-in via cfg flag; default off until empirically proven
    # to improve a CUDA contest score (council Fridrich currently VETOes).
    use_engineered_corrections: bool = False
    engineered_max_bytes: int = 51_200  # rate-budget guardrail (50 KB)

    # Pose TTO — adaptive convergence
    pose_max_steps: int = 2000        # upper bound
    pose_min_steps: int = 200         # minimum before checking convergence
    pose_lr: float = 0.01
    pose_convergence_tol: float = 1e-4  # stop when improvement < tol for patience steps
    pose_patience: int = 50
    pose_batch_pairs: int = 16  # 50 OOMs on 4090 with both scorers loaded

    # QAT — adaptive quality monitoring
    qat_max_epochs: int = 500         # upper bound
    qat_min_epochs: int = 100         # minimum
    qat_lr: float = 5e-5
    qat_quality_threshold: float = 1.5  # stop if FP4/float quality ratio exceeds this

    # Archive
    half_frame: bool = False
    binary_poses: bool = False
    brotli: bool = False
    mask_crf: int = 50

    # Mask codec for the ARCHIVE artifact. "av1_monochrome" is the
    # legacy default (lossy, ffmpeg dependency); "argmax_rle" is the
    # Yousfi council #8 lossless codec (pure Python, single-file
    # `tac.lossless.argmax_codec`). The TRAINING masks always use the
    # lossy AV1 path because we already store full-resolution intermediates
    # on disk; only the in-archive copy is affected by this flag.
    mask_codec: str = "av1_monochrome"

    # Weight compression: "fp4" (current FP4+codebook), "int4_lzma2" (int4+LZMA2),
    # "nwc" (Lane J-NWC neural weight compression — requires a trained codec
    # checkpoint at `weight_codec_path`), or "auto" (try int4_lzma2 first, fall
    # back to fp4 if quality degrades too much). Default stays "fp4" so the
    # canonical pipeline is unchanged unless an operator opts in.
    weight_compression: str = "fp4"
    # Lane J-NWC: path to a trained WeightCodec checkpoint produced by
    # experiments/train_neural_weight_codec.py. REQUIRED when
    # weight_compression == "nwc". Empty string disables the NWC branch and
    # the pipeline falls back to FP4 with a warning if NWC was requested.
    weight_codec_path: str = ""

    # Lane I (Cool-Chic / C3 residual neural-mask renderers, 2026-04-27).
    # When `variant` is set, step_export dispatches to the matching builder
    # + serializer (CCh1 / C3R1) instead of the canonical AsymmetricPair-
    # Generator + FP4A path. Empty string / None → legacy ASYM/FP4A path.
    # The trainer's checkpoint __meta__ also carries `variant`; that wins
    # over the cfg value (mirrors the fp4_codebook precedence).
    variant: str = ""
    # Cool-Chic / C3 architecture knobs — used by the rebuild side when the
    # __meta__ doesn't carry them (legacy training checkpoints). New
    # checkpoints embed all of these in __meta__.
    latent_ch: int = 8
    latent_shapes: tuple = ((6, 8), (12, 16), (24, 32))
    residual_hidden: int = 32
    residual_layers: int = 2
    residual_scale: float = 16.0
    # Phase 3 of Lane I: when set, the C3 residual head ships at int{N}
    # per-channel uniform quantization instead of FP4. Hypothesis: FP4's
    # per-block max-scaling collapses the small-magnitude residual signal
    # to 0; int8 preserves it at a tiny rate cost (~3.6% on the smoke
    # config). None = legacy pure-FP4 path. Set to 8 to enable.
    residual_quant_bits: int | None = None

    # Iterative refinement — convergence-driven, not arbitrary count
    max_iterations: int = 10          # safety bound (convergence stops earlier)
    convergence_tol: float = 0.01     # stop when score improvement < tol between cycles
    # The pipeline is a coordinate descent on the rate-distortion surface:
    # each cycle optimizes poses (distortion) then quantizes (rate).
    # Convergence is reached when marginal improvement falls below the
    # marginal rate cost of running another cycle.

    # Lane 8: multi-pass compress with score-feedback loop.
    # When ``multipass=True`` the compress flow wraps step_archive +
    # step_eval inside a ``MultiPassCompressor`` outer loop that iterates
    # encoder parameters (mask CRF) based on per-pass score deltas.
    # Compress-time only — strict-scorer-rule per CLAUDE.md.
    # See `.omx/research/council_lane_8_multipass_design_20260430.md`.
    multipass: bool = False
    multipass_max_passes: int = 3    # council verdict default (Carmack 80/30)
    multipass_target_score: float = 0.0  # 0 == "use baseline - 0.005"
    multipass_eps: float = 1e-3      # below scorer noise floor

    # Cross-paradigm wiring flags (PARADIGM-α/β/γ/la-pose) — registered here
    # so callers can pass them through; dispatch to the per-paradigm modules
    # is staged in follow-up commits (each gate has its own integration test
    # before the dispatch path goes live). Adversarial review 2026-05-06:
    # registering fields without dispatching is safe — every flag has
    # default=False / "" so existing pipeline runs are untouched. The
    # corresponding lane-registry entries are tracked separately.

    # PARADIGM-β (sensitivity-weighted weight compression). Routes
    # ``step_compress_weights`` to ``tac.owv3_sensitivity_weighted`` /
    # ``tac.neural_weight_codec_sensitivity``. Requires
    # ``sensitivity_map_path`` (a serialized SensitivityMap artifact).
    use_sensitivity_weighted: bool = False
    sensitivity_map_path: str = ""
    owv3_bit_budget_ratio: float = 0.7   # [calibration: owv3_sensitivity_weighted]
    owv3_protect_threshold: float = 1e-3 # [calibration: owv3_sensitivity_weighted]

    # PARADIGM-γ (joint score-aware codec stack). Routes the weight-compression
    # step through ``tac.joint_codec_stack_orchestrator`` (ADMM + Ballé +
    # arithmetic terminal codecs jointly). The ``JCSP`` magic byte identifies
    # the resulting container at inflate time.
    use_joint_codec_stack: bool = False
    # Path to a serialized per-tensor score-marginals artifact (mapping
    # tensor name → cached dScore/dByte). Required for JCSP dispatch — the
    # ADMM coordinator uses these marginals to project byte allocations.
    # Without it, all marginals are 0 and the coordinator has no signal.
    jcsp_score_marginals_path: str = ""

    # PARADIGM-α (mask-encoder portfolio). Default ``av1_monochrome`` preserves
    # current behaviour; the four research alternatives are registered here as
    # stubs and raise NotImplementedError at dispatch time until each codec's
    # compress-time training harness lands. mask_codec must be one of:
    #   av1_monochrome (default, wired) | argmax_rle (wired) |
    #   nerv | wavelet | vqvae | grayscale_lut (PARADIGM-α stubs)
    mask_codec: str = "av1_monochrome"

    # PARADIGM-la-pose. ``use_raft_init`` initializes pose TTO from RAFT-derived
    # poses (``experiments/derive_poses_from_raft.py``); ``use_riemannian_tto``
    # routes pose optimization through the SE(3) exp/log maps in ``tac.se3``.
    use_raft_init: bool = False
    use_riemannian_tto: bool = False

    # PARADIGM-δεζ (Phase 1 scaffolding — multi-day post-deadline mandate).
    # Each flag has default=False so existing pipelines run unchanged.
    # Guards mirror the β/γ/α REGISTERED-BUT-NOT-WIRED pattern from commits
    # 999211e5 / 9bdd3d56 / 77dc808a / 80455cf8 — each branch raises a
    # WARN if the flag is set but Phase 2/3 dispatch has not yet landed,
    # preventing the silent-no-op trap. See blueprint at
    # ``.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md``
    # and lane registry entries lane_delta_joint_training,
    # lane_epsilon_learnable_entropy, lane_zeta_self_compress_renderer.

    # PARADIGM-δ (joint scorer-aware training). Routes the renderer training
    # step through ``tac.joint_scorer_aware_training.JointScorerAwareLoss``
    # (compress-time only; strict-scorer-rule). Requires
    # ``joint_training_config_path`` to point at a serialized
    # JointTrainingConfig artifact. Phase 2 implementation pending Gate 2
    # (apogee_int6 [contest-CUDA] eval landing).
    use_joint_scorer_aware: bool = False
    joint_training_config_path: str = ""

    # PARADIGM-ε (learned entropy prior codec). When True, the
    # weight-compression step composes the Ballé hyperprior + arithmetic
    # coder via ``tac.learnable_entropy_model``. Adds an OPTIONAL
    # ``LEPR``-magic archive section (≤5KB). Phase 2 implementation
    # pending Gate 3 (δ Phase-2 [contest-CUDA] empirical improvement).
    use_learnable_entropy: bool = False

    # PARADIGM-ζ (full-renderer self-compression NN). Extends
    # tac.self_compress's postfilter pattern to the 88K-param renderer.
    # Default protect_patterns include FiLM canonical substrings; the
    # archive section uses magic ``ZETA``. Phase 2 implementation pending
    # Gate 3.
    use_full_renderer_self_compress: bool = False


# ── Step implementations ─────────────────────────────────────────────────

def step_extract_masks(cfg: PipelineConfig) -> tuple[Path, Path]:
    """Extract SegNet masks — FULL (1200) for training/TTO, HALF (600) for archive.

    Training and pose TTO need full-frame masks (both frame_t and frame_t1).
    The archive only needs half-frame masks (frame_t1 only, frame_t derived via warp).
    These are DIFFERENT requirements — conflating them caused a batch 38/75 crash
    when pose TTO tried to make 600 pairs from 600 half-frame masks.
    """
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── PARADIGM-α mask_codec stub guard ───────────────────────────────
    # Default ``av1_monochrome`` is the wired path. ``argmax_rle`` is also
    # wired. The four research stubs (nerv / wavelet / vqvae / grayscale_lut)
    # raise NotImplementedError below so that an operator who flips the flag
    # without first landing the codec's compress-time training harness gets a
    # loud failure rather than a silent fallback to AV1.
    _wired_mask_codecs = {"av1_monochrome", "argmax_rle"}
    _alpha_stub_codecs = {"nerv", "wavelet", "vqvae", "grayscale_lut"}
    if cfg.mask_codec not in _wired_mask_codecs:
        if cfg.mask_codec in _alpha_stub_codecs:
            raise NotImplementedError(
                f"PARADIGM-α: cfg.mask_codec={cfg.mask_codec!r} is "
                f"REGISTERED-BUT-NOT-WIRED. The compress-time training harness "
                f"for this codec has not yet landed; running it would either "
                f"silently fall back to AV1 or produce a non-decodable archive. "
                f"To enable, land the dispatch branch + an integration test "
                f"that verifies bit-identical decode roundtrip against the "
                f"contest scorer. See lane_alpha_{cfg.mask_codec}_mask in the "
                f"lane registry."
            )
        raise ValueError(
            f"step_extract_masks: unknown cfg.mask_codec={cfg.mask_codec!r}. "
            f"Wired: {sorted(_wired_mask_codecs)}; "
            f"PARADIGM-α stubs: {sorted(_alpha_stub_codecs)}."
        )

    # Full masks (1200 frames) — for training and pose TTO
    if cfg.masks and Path(cfg.masks).exists():
        full_masks_path = Path(cfg.masks)
        _log(f"Using pre-extracted full masks: {full_masks_path}")
    elif _step_done(out, "masks_full"):
        full_masks_path = out / "masks_full.mkv"
        _log("Full mask extraction already done, skipping")
    else:
        _log("Extracting FULL SegNet masks (1200 frames) from video...")
        from tac.data import decode_video
        from tac.scorer import load_scorers
        from tac.mask_codec import extract_masks, encode_masks

        gt_frames = decode_video(cfg.video, target_h=SEGNET_H, target_w=SEGNET_W)[:NUM_FRAMES]
        _log(f"  Decoded {len(gt_frames)} frames")

        models_dir = Path(cfg.upstream) / "models"
        _, segnet = load_scorers(
            str(models_dir / "posenet.safetensors"),
            str(models_dir / "segnet.safetensors"),
            device=cfg.device,
        )
        masks = extract_masks(gt_frames, segnet, device=cfg.device)
        full_masks_path = out / "masks_full.mkv"
        nbytes = encode_masks(masks, full_masks_path, crf=cfg.mask_crf)
        _log(f"  Full masks: {masks.shape[0]} frames, {nbytes:,} bytes")
        _mark_done(out, "masks_full", {"n_masks": masks.shape[0], "bytes": nbytes})

    # Half-frame masks (600 frames) — for archive only
    archive_suffix = ".amrc" if cfg.mask_codec == "argmax_rle" else ".mkv"
    archive_basename = f"masks_half{archive_suffix}" if cfg.half_frame else None

    if cfg.masks_archive and Path(cfg.masks_archive).exists():
        half_masks_path = Path(cfg.masks_archive)
        _log(f"Using pre-extracted half-frame masks: {half_masks_path}")
    elif cfg.half_frame:
        # Cache key is keyed off codec name so a codec switch invalidates.
        cache_step = f"masks_half_{cfg.mask_codec}"
        cached_path = out / archive_basename
        if _step_done(out, cache_step) and cached_path.exists():
            half_masks_path = cached_path
            _log(f"Reusing cached half-frame archive: {half_masks_path}")
        else:
            _log(
                f"Building half-frame archive masks via codec={cfg.mask_codec}..."
            )
            from tac.mask_codec import decode_masks_auto, encode_masks_auto
            # full_masks_path is always the AV1 intermediate (lossy training cache).
            full = decode_masks_auto(str(full_masks_path), codec="av1_monochrome")
            half = full[1::2]
            half_masks_path = cached_path
            if cfg.mask_codec == "argmax_rle":
                nbytes = encode_masks_auto(half, half_masks_path, codec="argmax_rle")
            else:
                nbytes = encode_masks_auto(
                    half, half_masks_path, codec="av1_monochrome",
                    crf=cfg.mask_crf,
                )
            _log(f"  Half masks: {half.shape[0]} frames, {nbytes:,} bytes "
                 f"({cfg.mask_codec})")
            _mark_done(out, cache_step, {
                "n_masks": int(half.shape[0]),
                "bytes": nbytes,
                "codec": cfg.mask_codec,
            })
    else:
        # No half-frame — archive uses the full mask source.
        if cfg.mask_codec == "argmax_rle":
            # Re-encode the full sequence under AMRC for the archive copy.
            cache_step = f"masks_full_{cfg.mask_codec}"
            cached_path = out / f"masks_full{archive_suffix}"
            if _step_done(out, cache_step) and cached_path.exists():
                half_masks_path = cached_path
                _log(f"Reusing cached AMRC archive masks: {half_masks_path}")
            else:
                from tac.mask_codec import decode_masks_auto, encode_masks_auto
                full = decode_masks_auto(str(full_masks_path), codec="av1_monochrome")
                half_masks_path = cached_path
                nbytes = encode_masks_auto(full, half_masks_path, codec="argmax_rle")
                _log(f"  AMRC archive masks: {full.shape[0]} frames, {nbytes:,} bytes")
                _mark_done(out, cache_step, {
                    "n_masks": int(full.shape[0]),
                    "bytes": nbytes,
                    "codec": cfg.mask_codec,
                })
        else:
            half_masks_path = full_masks_path  # legacy: archive uses AV1 intermediate

    return full_masks_path, half_masks_path


def step_export(cfg: PipelineConfig, iteration: int = 0) -> Path:
    """Export float checkpoint to FP4 renderer.bin."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # R38 fix: invalidate cache when cfg.checkpoint was modified after the
    # .done_export marker (mirrors R36 step_eval mtime invalidation).
    # R39 fix: also verify renderer.bin actually EXISTS on disk before
    # returning; the phantom-path bug class (R23/R34) was reintroduced if
    # someone deleted renderer.bin while .done_export survived.
    done_path = iter_dir / ".done_export"
    bin_path = iter_dir / "renderer.bin"
    if _step_done(iter_dir, "export"):
        try:
            ckpt_mtime = Path(cfg.checkpoint).stat().st_mtime
            done_mtime = done_path.stat().st_mtime
            if ckpt_mtime > done_mtime:
                _log(f"Export cache stale: {cfg.checkpoint} modified after "
                     f".done_export ({ckpt_mtime} > {done_mtime}), rerunning")
            elif not bin_path.exists():
                _log("Export marker present but renderer.bin missing — "
                     "forcing re-export to avoid phantom path", "WARN")
            else:
                _log(f"Export already done (iter {iteration}), skipping")
                return bin_path
        except (OSError, FileNotFoundError):
            if bin_path.exists():
                _log(f"Export already done (iter {iteration}), skipping")
                return bin_path
            _log("Export marker unreadable + bin missing, forcing re-export", "WARN")

    _log("Exporting checkpoint to FP4")

    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # 2026-04-26 fourth-layer arch-drift fix: train_renderer (DEN profile)
    # saves the model with torch.nn.utils.parametrize hooks attached for
    # self-compression / fake-quantization. State keys look like
    # `<layer>.parametrizations.weight.original` and the constructed
    # model (no hooks) expects plain `<layer>.weight`. Normalize here so
    # the same .pt loads cleanly into a fresh build_renderer() output.
    # Drop codebook tensors (`.parametrizations.<n>.0.codebook`) — those
    # are QAT internals not part of the plain weight tensor.
    if any(".parametrizations." in k for k in state.keys()):
        normalized = {}
        for k, v in state.items():
            if ".parametrizations." not in k:
                normalized[k] = v
                continue
            # Pattern: <prefix>.parametrizations.<name>.original  →  <prefix>.<name>
            head, _, tail = k.partition(".parametrizations.")
            name, _, suffix = tail.partition(".")
            if suffix == "original":
                normalized[f"{head}.{name}"] = v
            # else: drop codebook + other parametrize internals
        state = normalized

    # 2026-04-26 council fix (arch drift): read FULL arch from __meta__
    # (saved by train_renderer.py with use_zoom_flow / use_dsconv / etc).
    # Without this we waste GPU runs on arch mismatches between trainer and
    # consumer (DEN burned 1.2h on this exact bug). __meta__ wins over cfg
    # because it's the trainer's exact spec for THIS checkpoint.
    fp4_codebook = "default"
    fp4_robust_scale = False
    meta = ckpt.get("__meta__", {}) if isinstance(ckpt, dict) else {}
    if isinstance(meta, dict):
        fp4_codebook = meta.get("fp4_codebook", fp4_codebook)
        fp4_robust_scale = meta.get("fp4_robust_scale", fp4_robust_scale)

    # Take arch from __meta__ when present, fall back to cfg defaults.
    def _arch(key, default):
        return meta.get(key, getattr(cfg, key, default)) if meta else getattr(cfg, key, default)

    # ── Lane I dispatch (2026-04-27) ────────────────────────────────────
    # When the trainer's __meta__ records variant ∈ {coolchic_renderer,
    # c3_residual_renderer}, the canonical build_renderer + FP4A path does
    # NOT apply: those variants build PairGenerators wrapping a
    # CoolChicLatentRenderer / C3ResidualRenderer with latent
    # ParameterLists that the FP4A walk does not pick up. Dispatch to the
    # CCh1 / C3R1 export instead. cfg.variant wins when __meta__ is absent
    # (e.g. legacy checkpoints saved before the variant key was added).
    variant = _arch("variant", getattr(cfg, "variant", "") or "")
    if variant in ("coolchic_renderer", "c3_residual_renderer"):
        from tac.renderer_export import (
            export_coolchic_renderer,
            export_c3_residual_renderer,
        )
        # Cool-Chic / C3 latent_shapes is stored as a list-of-lists in JSON
        # (the trainer saves a tuple-of-tuples; torch.save round-trips it).
        latent_shapes_raw = _arch("latent_shapes", getattr(cfg, "latent_shapes", ((6, 8), (12, 16), (24, 32))))
        latent_shapes = tuple(tuple(s) for s in latent_shapes_raw)

        if variant == "coolchic_renderer":
            from tac.contrib.coolchic_renderer import build_coolchic_renderer
            model = build_coolchic_renderer(
                num_classes=int(_arch("num_classes", 5)),
                embed_dim=int(_arch("embed_dim", 6)),
                latent_ch=int(_arch("latent_ch", getattr(cfg, "latent_ch", 8))),
                hidden=int(_arch("base_ch", 32)),
                motion_hidden=int(_arch("motion_hidden", 32)),
                latent_shapes=latent_shapes,
                blend_mode=str(_arch("blend_mode", "scalar")),
                noise_mode=str(_arch("noise_mode", "deterministic")),
            )
        else:  # c3_residual_renderer
            from tac.contrib.coolchic_renderer import build_c3_residual_renderer
            model = build_c3_residual_renderer(
                num_classes=int(_arch("num_classes", 5)),
                embed_dim=int(_arch("embed_dim", 6)),
                latent_ch=int(_arch("latent_ch", getattr(cfg, "latent_ch", 6))),
                hidden=int(_arch("base_ch", 24)),
                motion_hidden=int(_arch("motion_hidden", 32)),
                residual_hidden=int(_arch("residual_hidden", getattr(cfg, "residual_hidden", 32))),
                residual_layers=int(_arch("residual_layers", getattr(cfg, "residual_layers", 2))),
                residual_scale=float(_arch("residual_scale", getattr(cfg, "residual_scale", 16.0))),
                latent_shapes=latent_shapes,
                blend_mode=str(_arch("blend_mode", "scalar")),
                noise_mode=str(_arch("noise_mode", "deterministic")),
            )

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint shape mismatch on Lane I variant={variant!r} — "
                f"refuse to export wrong arch. "
                f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}. "
                f"Verify latent_shapes={latent_shapes}, latent_ch=..., "
                f"residual_hidden=... match training."
            )
        n_params = sum(p.numel() for p in model.parameters())

        renderer_bin = iter_dir / "renderer.bin"
        residual_quant_bits = _arch("residual_quant_bits",
                                    getattr(cfg, "residual_quant_bits", None))
        _log(
            f"  Lane I export: variant={variant} codebook={fp4_codebook} "
            f"robust_scale={fp4_robust_scale} "
            f"residual_quant_bits={residual_quant_bits}"
        )
        if variant == "coolchic_renderer":
            nbytes = export_coolchic_renderer(
                model, str(renderer_bin),
                codebook_name=fp4_codebook,
                robust_scale=fp4_robust_scale,
            )
        else:
            nbytes = export_c3_residual_renderer(
                model, str(renderer_bin),
                codebook_name=fp4_codebook,
                robust_scale=fp4_robust_scale,
                residual_quant_bits=residual_quant_bits,
            )
        _log(f"  {n_params:,} params → {nbytes:,} bytes ({nbytes/1024:.1f} KB)")
        _mark_done(iter_dir, "export", {
            "params": n_params, "bytes": nbytes,
            "fp4_codebook": fp4_codebook, "fp4_robust_scale": fp4_robust_scale,
            "variant": variant,
            "residual_quant_bits": residual_quant_bits,
        })
        return renderer_bin

    # ── Legacy ASYM / FP4A path (default) ───────────────────────────────
    from tac.renderer import build_renderer
    from tac.renderer_export import export_asymmetric_checkpoint_fp4

    # 2026-04-26 third-layer arch-drift fix: route through build_renderer
    # — the SAME function train_renderer uses — so PairGenerator vs
    # AsymmetricPairGenerator dispatch is identical on both sides.
    # PairGenerator wraps a MaskRenderer + MotionPredictor (bias [2] for
    # 2-channel flow), AsymmetricPairGenerator inlines its own constructor
    # (bias [6] for the zoom-flow path). Hardcoding either constructor
    # here was the root of the DEN-V2 step_export crash.
    model = build_renderer(
        embed_dim=_arch("embed_dim", 6),
        base_ch=_arch("base_ch", 36),
        mid_ch=_arch("mid_ch", 60),
        motion_hidden=_arch("motion_hidden", 32),
        depth=_arch("depth", 1),
        pose_dim=_arch("pose_dim", 0),
        use_dsconv=_arch("use_dsconv", False),
        use_ghost=_arch("use_ghost", False),
        padding_mode=_arch("padding_mode", "zeros"),
        use_dilation=_arch("use_dilation", False),
        use_zoom_flow=_arch("use_zoom_flow", False),
    )
    # strict=False so we get the missing/unexpected lists; we then raise
    # if either is non-empty — equivalent to strict=True but with a much
    # better error message. (R24 finding: silent strict=False permitted
    # use_zoom_flow channel-count mismatch.)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint shape mismatch — refuse to export wrong arch. "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}. "
            f"Verify use_zoom_flow={cfg.use_zoom_flow}, base_ch={cfg.base_ch}, "
            f"mid_ch={cfg.mid_ch}, motion_hidden={cfg.motion_hidden} match training."
        )
    n_params = sum(p.numel() for p in model.parameters())

    renderer_bin = iter_dir / "renderer.bin"
    _log(f"  FP4 codebook={fp4_codebook}, robust_scale={fp4_robust_scale}")
    nbytes = export_asymmetric_checkpoint_fp4(
        model, str(renderer_bin),
        codebook_name=fp4_codebook,
        robust_scale=fp4_robust_scale,
    )
    _log(f"  {n_params:,} params → {nbytes:,} bytes ({nbytes/1024:.1f} KB)")

    _mark_done(iter_dir, "export", {
        "params": n_params, "bytes": nbytes,
        "fp4_codebook": fp4_codebook, "fp4_robust_scale": fp4_robust_scale,
    })
    return renderer_bin


def _export_refined_checkpoint(cfg: PipelineConfig, ckpt_path: Path,
                                iteration: int) -> Path:
    """Export a refined (Fridrich/QAT) checkpoint to FP4 .bin.

    R32 fix: split out from step_export because step_export is gated by
    .done_export which is set with the ORIGINAL cfg.checkpoint at iteration
    start. Re-calling step_export would silently return stale weights and
    discard whatever refinement just happened. This helper writes to a
    distinct filename so the .done_export gate cannot collide.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    out_path = iter_dir / "renderer_refined.bin"

    # R33 fix: idempotence guard. Match every other step's resume contract
    # so a restart doesn't re-do the load+export. The output is deterministic
    # given (cfg arch, ckpt_path) so reusing the existing file is safe.
    if out_path.exists() and out_path.stat().st_size > 0:
        _log(f"  Refined export already exists, skipping ({out_path.name})")
        return out_path

    from tac.renderer import AsymmetricPairGenerator
    from tac.renderer_export import export_asymmetric_checkpoint_fp4

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch, mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden, depth=cfg.depth,
        pose_dim=cfg.pose_dim, use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode, use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"_export_refined_checkpoint: shape mismatch on {ckpt_path}. "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}. "
            f"Verify arch (use_zoom_flow={cfg.use_zoom_flow}, base_ch={cfg.base_ch}) "
            f"matches the refined checkpoint."
        )
    # R34 fix: atomic write via tmp + os.replace. Without this, a concurrent
    # invocation (or interrupted run + immediate resume) could observe a
    # partial file via the idempotence guard's `out_path.stat().st_size > 0`
    # check and ship truncated weights downstream.
    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.bin")
    try:
        nbytes = export_asymmetric_checkpoint_fp4(model, str(tmp_path))
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    _log(f"  Refined export → {out_path.name} ({nbytes:,}B = {nbytes/1024:.1f}KB)")
    return out_path


def step_pose_tto(cfg: PipelineConfig, iteration: int = 0) -> Path:
    """Adaptive pose TTO with convergence detection."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # ── Cross-paradigm flag guards (PARADIGM-la-pose) ──────────────────
    # Mirror of the WARN-on-unwired pattern from step_compress_weights.
    # ``use_raft_init`` would route warm-start through
    # ``experiments/derive_poses_from_raft.py``;
    # ``use_riemannian_tto`` would dispatch optimization through
    # ``tac.se3`` exp/log maps. Both REGISTERED-BUT-NOT-WIRED until the
    # operator lands the dispatch branch + integration test (see
    # lane_raft_pose_init / lane_riemannian_pose_tto in the registry).
    if cfg.use_raft_init:
        _log(
            "PARADIGM-la-pose: cfg.use_raft_init=True but the RAFT warm-start "
            "dispatch path is REGISTERED-BUT-NOT-WIRED. Continuing with "
            "default pose initialization. To enable, the operator must "
            "land the dispatch branch in step_pose_tto.",
            "WARN",
        )
    # PARADIGM-la-pose use_riemannian_tto: WIRED (commit pending) — passes
    # through ``--optimizer=riemannian-sgd`` to the optimize_poses.py
    # subprocess later in this function. WARN guard removed; INFO log moved
    # to the subprocess-cmd construction site.
    # R29 fix: validate masks BEFORE constructing the subprocess cmd. Prior
    # version passed cfg.masks blindly, so an empty string flowed through
    # to optimize_poses.py and produced a cryptic decode failure deep in the
    # subprocess instead of a clear "masks not set" error here.
    if not cfg.masks:
        raise RuntimeError(
            f"step_pose_tto: cfg.masks is empty. step_extract_masks must run "
            f"first AND populate cfg.masks. iter={iteration} "
            f"output_dir={cfg.output_dir}. "
            f"Fix: verify run_compress() called step_extract_masks before "
            f"step_pose_tto, OR run extraction directly: "
            f"`python -m experiments.pipeline extract-masks --video {cfg.video} "
            f"--output-dir {cfg.output_dir}`."
        )
    if not Path(cfg.masks).exists():
        raise RuntimeError(
            f"step_pose_tto: masks file does not exist: {cfg.masks!r}. "
            f"iter={iteration} output_dir={cfg.output_dir}. "
            f"Fix: re-run mask extraction — "
            f"`python experiments/pipeline.py extract-masks --video {cfg.video} "
            f"--output-dir {cfg.output_dir}` — and confirm the output is "
            f"384x512 via `ffprobe {cfg.masks}` (CLAUDE.md catastrophic mask "
            f"res failure pattern)."
        )

    if _step_done(iter_dir, "pose_tto"):
        _log(f"Pose TTO already done (iter {iteration}), skipping")
        for suffix in [".bin", ".pt"]:
            p = iter_dir / f"optimized_poses{suffix}"
            if p.exists():
                return p
        return iter_dir / "optimized_poses.pt"

    _log(f"Pose TTO (up to {cfg.pose_max_steps} steps, converge at tol={cfg.pose_convergence_tol})")
    # R38 fix: use the FP4-exported renderer.bin from THIS iteration's
    # step_export, NOT the original cfg.checkpoint. Prior version optimized
    # poses against the float training checkpoint while the archive shipped
    # the FP4-quantized version → quantization-induced proxy-auth drift.
    # R39 fix: WARN loudly when falling back so operators see the regression
    # (silent fallback would re-introduce the original bug invisibly).
    pose_tto_checkpoint = iter_dir / "renderer.bin"
    if not pose_tto_checkpoint.exists():
        _log(
            f"renderer.bin missing for iter {iteration} — pose TTO will use "
            f"the float cfg.checkpoint ({cfg.checkpoint}). This silently reverts "
            f"the FP4-checkpoint TTO fix; verify step_export ran.",
            "WARN",
        )
        pose_tto_checkpoint = Path(cfg.checkpoint)

    # 2026-04-26: pose-space TTO requires FiLM conditioning (pose_dim>0).
    # If the renderer has no FiLM (e.g., DEN profile with pose_dim=0), the
    # optimized poses have nothing to influence and optimize_poses.py
    # correctly aborts. Detect that here from the FP4 .bin header and
    # soft-skip — return a sentinel "no poses" path so step_archive +
    # step_eval downstream know not to bundle pose data. Without this
    # skip, every non-FiLM profile fails the whole pipeline at this step.
    if pose_tto_checkpoint.suffix == ".bin":
        try:
            import struct, json
            with pose_tto_checkpoint.open("rb") as f:
                head = f.read(8 + 65536)  # magic + len + (likely-enough) header
            if head[:4] == b"FP4A":
                hlen = struct.unpack("<I", head[4:8])[0]
                hdr = json.loads(head[8:8 + hlen].decode())
                bin_pose_dim = hdr.get("pose_dim", 0)
                if bin_pose_dim == 0:
                    _log(
                        "renderer has pose_dim=0 (no FiLM) — pose TTO is "
                        "architecturally meaningless; skipping. Archive + "
                        "eval will run without optimized poses.", "WARN",
                    )
                    _mark_done(iter_dir, "pose_tto",
                               {"skipped": True, "reason": "no_film"})
                    # Sentinel: an absent path. step_archive + step_eval already
                    # handle missing optimized_poses gracefully.
                    return iter_dir / "optimized_poses.bin"
        except (struct.error, json.JSONDecodeError, OSError):
            # Fall through to the normal path; if the .bin is malformed
            # optimize_poses.py will error with a more specific message.
            pass
    cmd = [
        sys.executable, "-u", "experiments/optimize_poses.py",
        "--checkpoint", str(pose_tto_checkpoint),
        "--masks", cfg.masks,
        "--device", cfg.device,
        "--steps", str(cfg.pose_max_steps),
        "--lr", str(cfg.pose_lr),
        "--batch-pairs", str(cfg.pose_batch_pairs),
        "--eval-roundtrip",
        "--output-dir", str(iter_dir),
    ]

    # PARADIGM-la-pose Riemannian SE(3) dispatch — wired 2026-05-06.
    # The optimize_poses.py CLI already supports ``--optimizer=riemannian-sgd``
    # (Lane RM); cfg.use_riemannian_tto routes to it. The mode requires
    # ``--pose-mode=full-6dof`` and disabled LoRA paths per the CLI's own
    # validation (optimize_poses.py:530-546). We set the optimizer flag here
    # and trust optimize_poses.py to validate the mode-compatibility.
    if cfg.use_riemannian_tto:
        cmd.extend(["--optimizer", "riemannian-sgd"])
        _log(
            "PARADIGM-la-pose: cfg.use_riemannian_tto=True → routing pose TTO "
            "through SE(3) Riemannian SGD (tac.riemannian_pose_optimizer). "
            "Predicted band [1.05, 1.15] vs Lane A's 1.15 [contest-CUDA] "
            "(per optimize_poses.py docstring).",
            "INFO",
        )

    # Reuse cached pose targets from previous runs (saves ~15 min of PoseNet inference).
    # DX #5 (2026-04-26): mtime-gate the cache. iter_dir was created (or
    # touched) when this iteration started; a gt_pose_targets.pt OLDER than
    # iter_dir is left over from a prior session under a different profile,
    # different scorers, or different upstream snapshot. Silent reuse there
    # is exactly the catastrophe pattern (wrong cache → wrong PoseNet
    # targets → silently optimised the wrong objective).
    iter_dir_mtime = iter_dir.stat().st_mtime
    for candidate in [iter_dir / "gt_pose_targets.pt", Path(cfg.output_dir) / "gt_pose_targets.pt"]:
        if not candidate.exists():
            continue
        cand_mtime = candidate.stat().st_mtime
        if cand_mtime < iter_dir_mtime:
            age_h = (iter_dir_mtime - cand_mtime) / 3600.0
            _log(
                f"  Found stale {candidate.name} from {age_h:.1f}h ago "
                f"(older than iter_dir creation) — re-deriving instead of "
                f"silently using.", "WARN",
            )
            continue
        cmd.extend(["--gt-pose-targets", str(candidate)])
        _log(f"  Using cached pose targets: {candidate}")
        break

    t0 = time.monotonic()
    # R30: 4-hour timeout (pose TTO is the longest-running subprocess; A100
    # 4090 typical wall is 30-90 min). A hung scorer or DataLoader would
    # otherwise block the pipeline forever.
    # 2026-04-26: routed through _run_step so a crash surfaces the last
    # 20 stderr lines + replay command instead of silent `exit N`.
    try:
        rc = _run_step(cmd, "pose_tto", iter_dir, timeout=14400)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Pose TTO timed out after 4h — likely hung GPU/DataLoader. "
            f"Fix: SSH in and run `nvidia-smi`; if util=0%% kill the python "
            f"process (`pkill -f optimize_poses`), then "
            f"`rm {iter_dir}/.done_pose_tto` and re-launch. Inspect "
            f"{iter_dir}/pose_tto.stderr.log for stalls."
        )
    elapsed = time.monotonic() - t0

    if rc != 0:
        raise RuntimeError(
            f"Pose TTO failed (exit {rc}); see {iter_dir}/pose_tto.stderr.log. "
            f"Fix: tail -100 the stderr log for the actual exception. Common "
            f"causes: (1) FP4 .bin shape mismatch — verify --base-ch / --mid-ch "
            f"match the checkpoint's profile; (2) CUDA OOM — drop "
            f"--pose-batch-pairs to 4; (3) masks resolution wrong — verify "
            f"{cfg.masks} is 384x512 not 48x64 (cf. CLAUDE.md catastrophic mask "
            f"res failure)."
        )

    _log(f"  Complete in {elapsed/60:.1f} min")

    poses_path = iter_dir / "optimized_poses.bin"
    if not poses_path.exists():
        poses_path = iter_dir / "optimized_poses.pt"

    _mark_done(iter_dir, "pose_tto", {"elapsed_s": round(elapsed)})
    return poses_path


def step_sensitivity_sweep(cfg: PipelineConfig, iteration: int = 0) -> Path | None:
    """Run scorer-Jacobian sensitivity sweep to determine per-layer bit allocation.

    Measures how quantization noise in each renderer layer propagates through
    the actual SegNet + PoseNet scorers. Layers that are scorer-insensitive
    get fewer bits (2), scorer-sensitive layers get more bits (8), and the
    rest stay at 4. The output JSON is consumed by the QAT step and the
    weight compression step for mixed-precision export.

    Returns path to the sensitivity sweep JSON, or None if skipped.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    step_name = "sensitivity_sweep"

    sweep_json = iter_dir / "scorer_sensitivity_sweep.json"

    if _step_done(iter_dir, step_name):
        _log(f"Sensitivity sweep already done (iter {iteration}), skipping")
        return sweep_json if sweep_json.exists() else None

    _log("Scorer-Jacobian sensitivity sweep (Yousfi approach)")
    cmd = [
        sys.executable, "-u", "experiments/scorer_sensitivity_sweep.py",
        "--checkpoint", cfg.checkpoint,
        "--device", cfg.device,
        "--n-pairs", "5",
        "--output", str(sweep_json),
    ]

    t0 = time.monotonic()
    # R30: 1-hour timeout for sensitivity sweep (Yousfi Jacobian; 5 pairs
    # should be 5-15 min on A100). Hung Jacobian must not block the pipeline.
    try:
        result = subprocess.run(cmd, timeout=3600)
    except subprocess.TimeoutExpired:
        _log("Sensitivity sweep timed out after 1h, continuing without", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True, "timeout": True})
        return None
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        _log(f"Sensitivity sweep failed (exit {result.returncode}), continuing without", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True, "elapsed_s": round(elapsed)})
        return None

    _log(f"  Complete in {elapsed/60:.1f} min")

    # Log the allocation summary
    if sweep_json.exists():
        try:
            data = json.loads(sweep_json.read_text())
            alloc = data.get("allocation", {})
            uniform_kb = data.get("uniform_kb", 0)
            mixed_kb = data.get("mixed_kb", 0)
            _log(f"  Allocation: {len(alloc)} layers, "
                 f"uniform={uniform_kb:.1f}KB -> mixed={mixed_kb:.1f}KB "
                 f"({(1 - mixed_kb / max(uniform_kb, 0.1)) * 100:.1f}% savings)")
        except (json.JSONDecodeError, KeyError):
            pass

    _mark_done(iter_dir, step_name, {"elapsed_s": round(elapsed)})
    return sweep_json


def step_qat(cfg: PipelineConfig, iteration: int = 0,
             mixed_precision_json: Path | None = None) -> Path:
    """QAT fine-tune with quality monitoring."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    if _step_done(iter_dir, "qat"):
        _log(f"QAT already done (iter {iteration}), skipping")
        # qat_finetune.py may save under either name depending on path; resolve
        # against what actually exists on disk so the caller never gets a
        # phantom path. (Round 23 finding.)
        for candidate in ("renderer_fp4.bin", "renderer_qat.bin"):
            p = iter_dir / candidate
            if p.exists():
                return p
        existing_bins = sorted(iter_dir.glob("*.bin"))
        if existing_bins:
            return existing_bins[-1]
        raise RuntimeError(
            f"QAT marked done in {iter_dir} but no .bin output found. "
            f"Delete .done_qat marker to force rerun."
        )

    _log(f"QAT fine-tune (up to {cfg.qat_max_epochs} epochs)")
    cmd = [
        sys.executable, "-u", "experiments/qat_finetune.py",
        "--checkpoint", cfg.checkpoint,
        "--upstream", cfg.upstream,
        "--device", cfg.device,
        "--fp4-epochs", str(cfg.qat_max_epochs),
        "--lr", str(cfg.qat_lr),
        "--output-dir", str(iter_dir),
        "--base-ch", str(cfg.base_ch),
        "--mid-ch", str(cfg.mid_ch),
        "--motion-hidden", str(cfg.motion_hidden),
        "--depth", str(cfg.depth),
        "--embed-dim", str(cfg.embed_dim),
        "--pose-dim", str(cfg.pose_dim),
        "--padding-mode", cfg.padding_mode,
    ]
    if cfg.use_dsconv:
        cmd.append("--use-dsconv")
    if cfg.use_dilation:
        cmd.append("--use-dilation")
    if cfg.use_zoom_flow:
        cmd.append("--use-zoom-flow")
    if mixed_precision_json is not None and mixed_precision_json.exists():
        cmd.extend(["--mixed-precision-json", str(mixed_precision_json)])
        _log(f"  Using mixed-precision allocation from: {mixed_precision_json}")

    t0 = time.monotonic()
    # R30: 6-hour timeout for QAT (250 FP4 epochs typical 1-3h on A100).
    # 2026-04-26: routed through _run_step so a crash surfaces the last
    # 20 stderr lines + replay command instead of silent `exit N`.
    try:
        rc = _run_step(cmd, "qat", iter_dir, timeout=21600)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "QAT timed out after 6h — likely hung. "
            f"Fix: SSH in, `nvidia-smi` to confirm util=0%%, then "
            f"`pkill -f qat_finetune` and `rm {iter_dir}/.done_qat` to retry. "
            f"If the renderer is large (>200K params) consider --qat-max-epochs "
            f"100 instead of the profile default."
        )
    elapsed = time.monotonic() - t0

    if rc != 0:
        raise RuntimeError(
            f"QAT failed (exit {rc}); see {iter_dir}/qat.stderr.log. "
            f"Fix: tail -100 the stderr log. Common causes: (1) checkpoint "
            f"shape mismatch — verify the float .pt was trained with the same "
            f"--base-ch / --mid-ch / --depth as the profile in cfg; (2) FP4 "
            f"codebook missing — confirm renderer.py exports FakeQuantFP4 "
            f"correctly; (3) NaN loss — drop --qat-lr by 10x and retry."
        )

    _log(f"  Complete in {elapsed/60:.1f} min")

    # R34 fix: qat_finetune.py saves 'renderer_fp4.bin', NOT 'renderer_qat.bin'.
    # Prior code returned the wrong-name path and only the glob fallback
    # rescued at runtime. Worse, the alphabetical glob could return the
    # PRE-QAT 'renderer.bin' if 'renderer_fp4.bin' wasn't present, silently
    # using the un-quantized model downstream.
    qat_bin = iter_dir / "renderer_fp4.bin"
    if not qat_bin.exists():
        # Try the legacy/alternative names before falling back.
        for alt in ("renderer_qat.bin", "renderer_mixed.bin"):
            cand = iter_dir / alt
            if cand.exists():
                qat_bin = cand
                break
        else:
            raise RuntimeError(
                f"QAT step succeeded but no QAT artifact found in {iter_dir}. "
                f"Expected renderer_fp4.bin (or renderer_qat.bin / "
                f"renderer_mixed.bin). Refusing to fall back to pre-QAT "
                f"renderer.bin (would silently ship un-quantized weights). "
                f"Fix: inspect {iter_dir}/qat.stdout.log for the export step; "
                f"verify qat_finetune.py's `--output-dir` arg matches the "
                f"iter_dir, and that export_asymmetric_checkpoint_fp4() ran "
                f"to completion. Re-run with `rm {iter_dir}/.done_qat` to retry."
            )

    _mark_done(iter_dir, "qat", {"elapsed_s": round(elapsed)})
    return qat_bin


def step_fridrich_refine(cfg: PipelineConfig, iteration: int = 0) -> Path:
    """Step 3.5: Fridrich steganalytic refinement (post-QAT polish).

    100 epochs at very low LR with ONLY Fridrich losses (texture + L-inf + Markov).
    Pushes pixel-level errors further into the scorer's null space without
    disturbing the QAT-optimized scorer distortion. Our competitive advantage.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    step_name = "fridrich_refine"

    def _resolve_fridrich_output() -> Path:
        """Find the best Fridrich refinement output, with explicit fallbacks.

        R31 fix: prior version returned a hardcoded `fridrich/distill_phase2_best.pt`
        path that may not exist if train_distill.py used a different name.
        Glob the fridrich subdir; on resume from failure, fall back to QAT.

        R32 fix: glob sort by mtime (newest first) — alphabetical sort could
        return a partial `*_tmp.pt` save instead of the latest complete one.
        Last-resort fallback raises RuntimeError instead of returning the
        original input — the pipeline made no progress, fail loud.
        """
        fridrich_dir = iter_dir / "fridrich"
        if fridrich_dir.exists():
            # R34: distill_best.pt removed — no producer ever writes it.
            # train_distill.py saves distill_phase2_best.pt and distill_latest.pt.
            for candidate in (fridrich_dir / "distill_phase2_best.pt",
                              fridrich_dir / "distill_latest.pt"):
                if candidate.exists():
                    return candidate
            # Sort by mtime (newest last); last item is most recent successful save.
            pts = sorted(fridrich_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
            # Skip obvious tempfiles in the name.
            pts = [p for p in pts if "tmp" not in p.name.lower()]
            if pts:
                return pts[-1]
        # No Fridrich output exists — fall back to the QAT pre-Fridrich checkpoint.
        for fb in (iter_dir / "qat_best_float.pt", iter_dir / "renderer_qat.bin",
                   iter_dir / "renderer_fp4.bin"):
            if fb.exists():
                return fb
        # Last resort: pipeline made NO progress. Fail loud per R32 finding.
        raise RuntimeError(
            f"_resolve_fridrich_output: no Fridrich, QAT-best, or FP4 output "
            f"found in {iter_dir}. Pipeline made no progress beyond input. "
            f"Delete .done_fridrich_refine and .done_qat to force rerun."
        )

    if _step_done(iter_dir, step_name):
        # R38 fix: if the prior run was a failure/timeout, REMOVE the .done
        # marker and re-run rather than blocking forever. Prior version
        # logged a warning then short-circuited via fallback — the operator
        # had to manually rm the marker to get retry.
        # R39 fix: also re-check skipped precondition (qat_best_float.pt
        # may have appeared since the skip). And on retry, remove any
        # partial fridrich/ output to prevent stale .pt globs.
        def _clear_marker_and_partial() -> None:
            try:
                (iter_dir / f".done_{step_name}").unlink()
            except FileNotFoundError:
                pass
            # Remove only .pt files in fridrich/ — keep config.json etc.
            fridrich_dir = iter_dir / "fridrich"
            if fridrich_dir.exists():
                for stale in fridrich_dir.glob("*.pt"):
                    try:
                        stale.unlink()
                    except OSError:
                        pass
            # R40 fix: ALSO remove iter_dir-level distill_latest.pt if present
            # — it could be from an ad-hoc resumed run (different arch /
            # different training session) and silently become the Fridrich
            # input on retry. The canonical path is qat_best_float.pt;
            # distill_latest.pt at the iter_dir level is suspect.
            stale_top = iter_dir / "distill_latest.pt"
            if stale_top.exists():
                try:
                    stale_top.unlink()
                    _log(f"  Removed suspect {stale_top.name} (ad-hoc artifact)", "WARN")
                except OSError:
                    pass

        try:
            done_meta = json.loads((iter_dir / f".done_{step_name}").read_text())
            if done_meta.get("failed") or done_meta.get("timeout"):
                _log(f"  Prior Fridrich was {done_meta} — clearing marker + partial output, retrying", "WARN")
                _clear_marker_and_partial()
            elif done_meta.get("skipped"):
                # R39: re-check whether the skip precondition still holds.
                if (iter_dir / "qat_best_float.pt").exists():
                    _log("  Fridrich was skipped but qat_best_float.pt now exists — retrying", "WARN")
                    _clear_marker_and_partial()
                else:
                    _log("  Fridrich previously skipped (no QAT input); using fallback")
                    return _resolve_fridrich_output()
            else:
                _log(f"Fridrich refinement already done (iter {iteration}), skipping")
                return _resolve_fridrich_output()
        except (json.JSONDecodeError, FileNotFoundError):
            _log("Fridrich marker unreadable; running fresh", "WARN")
            _clear_marker_and_partial()

    # Find best QAT checkpoint to start from
    # qat_finetune.py saves: qat_best_float.pt, renderer_fp4.bin
    qat_ckpt = iter_dir / "qat_best_float.pt"
    if not qat_ckpt.exists():
        qat_ckpt = iter_dir / "distill_latest.pt"
    if not qat_ckpt.exists():
        _log("No QAT checkpoint found, skipping Fridrich refinement", "WARN")
        _mark_done(iter_dir, step_name, {"skipped": True})
        return _resolve_fridrich_output()

    _log("Fridrich steganalytic refinement (100 epochs, Fridrich-only)")
    cmd = [
        sys.executable, "-u", "experiments/train_distill.py",
        "--checkpoint", str(qat_ckpt),
        "--masks", cfg.masks,
        "--device", cfg.device,
        "--output-dir", str(iter_dir / "fridrich"),
        "--base-ch", str(cfg.base_ch), "--mid-ch", str(cfg.mid_ch),
        "--motion-hidden", str(cfg.motion_hidden),
        "--depth", str(cfg.depth), "--pose-dim", str(cfg.pose_dim),
        "--embed-dim", str(cfg.embed_dim),
        "--padding-mode", cfg.padding_mode,
        "--eval-roundtrip",
        # Zero scorer loss — ONLY Fridrich pixel-space losses
        "--seg-weight", "0.0", "--pose-weight", "0.0", "--pixel-weight", "0.0",
        "--use-texture-loss", "--texture-loss-weight", "1.0",
        "--use-linf-penalty", "--linf-weight", "0.1",
        "--use-markov-loss", "--markov-weight", "0.5",
        "--phase1-epochs", "0", "--phase2-epochs", "100", "--phase3-epochs", "0",
        "--phase2-lr", "1e-5",
        "--ema-decay", "0.997",
        "--checkpoint-every", "50", "--eval-every", "25", "--log-every", "10",
        "--upstream", cfg.upstream,
    ]
    if cfg.use_dsconv:
        cmd.append("--use-dsconv")
    if cfg.use_dilation:
        cmd.append("--use-dilation")
    if cfg.use_zoom_flow:
        cmd.append("--use-zoom-flow")
    # Pass through training-discipline flags so Fridrich refinement runs in
    # the same regime as the original training. Otherwise the refinement
    # diverges from the training distribution it's trying to polish.
    if cfg.use_swa:
        cmd.append("--use-swa")
    if cfg.use_per_class_weights:
        cmd.append("--use-per-class-weights")
    if cfg.freeze_motion_phase2:
        cmd.append("--freeze-motion-phase2")
    if cfg.freeze_renderer_phase3:
        cmd.append("--freeze-renderer-phase3")
    if cfg.beneficial_quant_noise:
        cmd.append("--beneficial-quant-noise")
    if cfg.use_variance_noise:
        cmd.extend([
            "--use-variance-noise",
            "--variance-noise-weight", str(cfg.variance_noise_weight),
            "--variance-noise-base-std", str(cfg.variance_noise_base_std),
            "--variance-noise-kernel", str(cfg.variance_noise_kernel),
            "--variance-noise-mode", cfg.variance_noise_mode,
        ])
    if cfg.use_uncertainty_loss:
        cmd.extend([
            "--use-uncertainty-loss",
            "--uncertainty-loss-weight", str(cfg.uncertainty_loss_weight),
            "--uncertainty-loss-floor", str(cfg.uncertainty_loss_floor),
        ])

    # R30: 2-hour timeout. 2026-04-26 Hotz R3: route through _run_step so
    # stderr is captured to <iter_dir>/fridrich_refine.stderr.log — without
    # this, a 2h training subprocess would crash and leave only "exit N" as
    # diagnostic. Same waste class as the SHIRAZ 16h burn (memory:
    # feedback_no_wasted_resources).
    t0 = time.monotonic()
    try:
        rc = _run_step(cmd, "fridrich_refine", iter_dir, timeout=7200)
    except subprocess.TimeoutExpired:
        _log("Fridrich refinement timed out after 2h, falling back to QAT output", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True, "timeout": True})
        return _resolve_fridrich_output()
    elapsed = time.monotonic() - t0

    if rc != 0:
        _log(f"Fridrich refinement failed (exit {rc}). See iter_{iteration}/fridrich_refine.stderr.log", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True})
        return _resolve_fridrich_output()

    _log(f"  Fridrich refinement complete in {elapsed/60:.1f} min")
    _mark_done(iter_dir, step_name, {"elapsed_s": round(elapsed)})
    # R31: glob for the actual best checkpoint instead of returning a hardcoded
    # path that may not exist if train_distill.py used a different filename.
    return _resolve_fridrich_output()


def _load_renderer_for_jcsp_dry_run(
    cfg: PipelineConfig,
    checkpoint_path: Path,
) -> nn.Module:
    """Load the renderer checkpoint for a JCSP metadata-only dry run.

    This loader expects a PyTorch state_dict pickle (the training-checkpoint
    format produced by experiments/train_*.py). It is NOT a content-detecting
    loader for binary exports — if the operator passes an FP4A/ASYM/DPSM/I4LZ
    archive .bin file, raise a clear error directing them to use the canonical
    ``experiments.precompute_gradient_corrections.load_renderer`` instead. This
    closes the DEN-V2 bug class (2026-04-26) where torch.load on a binary export
    would crash with "could not convert string to float: 'P4AV'".
    """
    from tac.renderer import build_renderer

    # Magic-byte gate: refuse binary export formats up front so
    # ``torch.load`` only ever sees a pickle / zip header.
    with checkpoint_path.open("rb") as fh:
        magic = fh.read(4)
    _BINARY_EXPORT_MAGICS = (b"FP4A", b"ASYM", b"DPSM", b"I4LZ", b"CCh1", b"C3R1")
    if magic in _BINARY_EXPORT_MAGICS:
        raise RuntimeError(
            f"_load_renderer_for_jcsp_dry_run expects a PyTorch state_dict "
            f"pickle but {checkpoint_path} starts with magic {magic!r}, which "
            f"identifies a binary export format ({magic.decode('ascii', errors='replace')}). "
            f"Use experiments.precompute_gradient_corrections.load_renderer "
            f"(the canonical content-detecting loader) instead."
        )
    try:
        ckpt = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        if os.environ.get("PIPELINE_ALLOW_UNSAFE_CKPT", "0") != "1":
            raise
        ckpt = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
    model = build_renderer(
        base_ch=cfg.base_ch,
        mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden,
        depth=cfg.depth,
        pose_dim=cfg.pose_dim,
        embed_dim=cfg.embed_dim,
        use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode,
        use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
    )
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def step_compress_weights(
    cfg: PipelineConfig,
    checkpoint_path: Path,
    iteration: int = 0,
    sensitivity_json: Path | None = None,
) -> Path:
    """Compress model weights using mixed-precision quantization + LZMA2.

    Three compression strategies (choose best for this checkpoint):
    1. Int4 per-tensor + LZMA2 (simplest, ~2.2 bits/weight)
    2. LearnableBitDepth fine-tune + LZMA2 (~1.5-2.0 bits/weight, future)
    3. FP4 + Brotli (current, ~4.4 bits/weight, fallback)

    When cfg.weight_compression == "auto", tries int4_lzma2 first.
    Measures quality degradation on a small forward pass. Falls back
    to fp4 if output divergence exceeds an acceptable threshold.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    step_name = "compress_weights"

    # R40 fix: invalidate cache when cfg.weight_compression mode differs
    # from the mode recorded in the .done marker. Prior version cached
    # `skipped=True` from a fp4 run and silently blocked the next run from
    # actually running int4_lzma2 even after the operator switched modes.
    #
    # Adversarial review 2026-05-06 (Contrarian): the prior cache check only
    # compared against ``cfg.weight_compression``. With β dispatch wired (mode
    # = "owv3_sensitivity_weighted" when use_sensitivity_weighted=True), a
    # cached fp4 .done marker would silent-skip the β branch. The effective
    # mode is now derived from the cross-paradigm flags BEFORE the cache
    # comparison.
    if cfg.use_joint_codec_stack:
        effective_mode = "joint_codec_stack_archive_member"
    elif cfg.use_sensitivity_weighted and cfg.sensitivity_map_path and Path(cfg.sensitivity_map_path).exists():
        effective_mode = "owv3_sensitivity_weighted"
    else:
        effective_mode = cfg.weight_compression
    done_path = iter_dir / f".done_{step_name}"
    if _step_done(iter_dir, step_name):
        prior_mode = None
        try:
            prior_mode = json.loads(done_path.read_text()).get("mode")
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        if prior_mode is not None and prior_mode != effective_mode:
            _log(f"Weight compression mode changed: {prior_mode!r} → "
                 f"{effective_mode!r}; invalidating cache", "WARN")
            try:
                done_path.unlink()
            except FileNotFoundError:
                pass
        else:
            _log(f"Weight compression already done (iter {iteration}), skipping")
            # Return whichever format was produced
            for suffix in ["_int4lzma2.bin", "_owv3_sensitivity.bin", ".bin"]:
                p = iter_dir / f"renderer{suffix}"
                if p.exists():
                    return p
            return checkpoint_path

    mode = cfg.weight_compression

    # ── Cross-paradigm flag guards (PARADIGM-β, PARADIGM-gamma) ─────────
    # Registered fields in PipelineConfig but full dispatch wiring is staged
    # behind operator approval per the cross-paradigm blueprint. Surfacing a
    # WARN here prevents the silent-no-op trap if a caller sets the flag
    # without realizing the dispatch path is not yet active. A future commit
    # will replace each block with the actual module dispatch.
    if cfg.use_sensitivity_weighted:
        # PARADIGM-β dispatch: route through tac.owv3_sensitivity_weighted.
        # Wired 2026-05-06 (cross-paradigm wiring 6/N).
        # Requires:
        # - cfg.sensitivity_map_path points at a serialized SensitivityMap
        #   artifact (.pt produced by the GPU sensitivity sweep, validated
        #   under load_sensitivity_map's weights_only=True allowlist).
        # - the checkpoint loadable via the standard ArchConfig hydrator.
        # If the path is missing or empty, falls back to cfg.weight_compression
        # with a loud WARN so an operator who forgot to provide the artifact
        # gets a clear error, not a silent "did nothing".
        sens_path = cfg.sensitivity_map_path
        if not sens_path or not Path(sens_path).exists():
            _log(
                f"PARADIGM-β: cfg.use_sensitivity_weighted=True but "
                f"sensitivity_map_path={sens_path!r} does not exist. "
                f"Falling through to weight_compression={mode!r}. "
                f"Produce the artifact via the GPU sensitivity sweep first "
                f"(see tac.sensitivity_map / lane_sensitivity_map in the "
                f"registry).",
                "WARN",
            )
        else:
            from tac.owv3_sensitivity_weighted import encode_owv3_archive
            from tac.sensitivity_map import (
                SensitivityMapError,
                load_sensitivity_map,
                validate_real_sensitivity_artifact,
            )
            from tac.renderer import build_renderer

            sensitivities, sens_metadata = load_sensitivity_map(sens_path)
            try:
                validate_real_sensitivity_artifact(sensitivities, sens_metadata)
            except SensitivityMapError as exc:
                raise RuntimeError(
                    "PARADIGM-β: refusing to build OWV3 archive with "
                    f"non-real sensitivity artifact {sens_path!r}: {exc}"
                ) from exc
            # Adversarial review 2026-05-06 fix: weights_only=True is the
            # safe path (CLAUDE.md FORBIDDEN PATTERN: arbitrary torch.load on
            # checkpoints from external sources — Vast.ai/Modal/Lightning
            # artifacts). For full-checkpoint dicts that wrap a state_dict,
            # weights_only=True works since state_dict tensors are allowlisted.
            try:
                ckpt = torch.load(
                    str(checkpoint_path), map_location="cpu", weights_only=True
                )
            except Exception:
                # Legacy checkpoint format may need weights_only=False; only
                # fall back if the operator explicitly opts in via env-var.
                import os as _os
                if _os.environ.get("PIPELINE_ALLOW_UNSAFE_CKPT", "0") != "1":
                    raise
                ckpt = torch.load(
                    str(checkpoint_path), map_location="cpu", weights_only=False
                )
            model = build_renderer(
                base_ch=cfg.base_ch,
                mid_ch=cfg.mid_ch,
                motion_hidden=cfg.motion_hidden,
                depth=cfg.depth,
                pose_dim=cfg.pose_dim,
                embed_dim=cfg.embed_dim,
                use_dsconv=cfg.use_dsconv,
                padding_mode=cfg.padding_mode,
                use_dilation=cfg.use_dilation,
                use_zoom_flow=cfg.use_zoom_flow,
            )
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            # Adversarial review 2026-05-06 fix: explicit eval mode (encode_owv3_archive
            # also calls .eval() but defensive double-call is cheap and
            # protects callers that compose this branch into other harnesses).
            model.eval()
            archive_bytes = encode_owv3_archive(
                model,
                sensitivities=sensitivities,
                bit_budget_ratio=cfg.owv3_bit_budget_ratio,
                protect_threshold=cfg.owv3_protect_threshold,
            )
            out_path = iter_dir / "renderer_owv3_sensitivity.bin"
            out_path.write_bytes(archive_bytes)
            _log(
                f"PARADIGM-β: wrote {len(archive_bytes):,}-byte sensitivity-"
                f"weighted OWV3 archive at {out_path} "
                f"(bit_budget_ratio={cfg.owv3_bit_budget_ratio}, "
                f"protect_threshold={cfg.owv3_protect_threshold}, "
                f"sens_metadata_keys={sorted(sens_metadata.keys()) if sens_metadata else []}).",
                "INFO",
            )
            done_path.write_text(json.dumps({
                "mode": "owv3_sensitivity_weighted",
                "iteration": iteration,
                "archive_bytes": len(archive_bytes),
                "bit_budget_ratio": cfg.owv3_bit_budget_ratio,
                "protect_threshold": cfg.owv3_protect_threshold,
            }))
            (iter_dir / f".done_{step_name}").touch()
            return out_path
    # PARADIGM-gamma JCSP dispatch (joint score-aware codec stack) — when
    # ``cfg.use_joint_codec_stack=True``, decompose the model into JCSP
    # streams via ``tac.jcsp_stream_builder.model_to_stream_sources``,
    # run the ADMM coordinator across {repr, predict, quant, entropy}, and
    # pack the resulting JCSP container as the renderer payload. Both gates
    # must be present (use_joint_codec_stack flag + score_marginals path) or
    # the branch raises NotImplementedError so the operator cannot
    # accidentally ship a stub archive (silent-no-op trap class).
    if cfg.use_joint_codec_stack:
        marginals_path = cfg.jcsp_score_marginals_path or ""
        missing = []
        if not marginals_path or not Path(marginals_path).exists():
            missing.append(f"cfg.jcsp_score_marginals_path={marginals_path!r}")
        if missing:
            raise NotImplementedError(
                f"PARADIGM-gamma JCSP dispatch (use_joint_codec_stack=True): "
                f"missing prerequisites — {', '.join(missing)}. The score "
                f"marginals artifact is produced by the frontier sampler "
                f"(per-tensor cached dScore/dByte from a calibration sweep). "
                f"Without it, the ADMM coordinator has no gradient signal "
                f"and JCSP cannot allocate bytes intelligently. See "
                f"lane_joint_codec_stack in the lane registry. The "
                f"decomposition primitives "
                f"(tac.jcsp_stream_builder.model_to_stream_sources) are "
                f"now landed (commit b2d7928a, 16 tests); the remaining "
                f"work is the per-tensor marginals harness + the "
                f"run_admm/run_sequential_codec_stack invocation."
            )
        from tac.jcsp_stream_builder import jcsp_stream_source_archive_member

        model = _load_renderer_for_jcsp_dry_run(cfg, checkpoint_path)
        archive_path = iter_dir / "jcsp_archive_member.zip"
        manifest_path = iter_dir / "jcsp_archive_member_manifest.json"
        archive_bytes, archive_contract = jcsp_stream_source_archive_member(
            model,
            score_marginals_path=marginals_path,
            archive_path_for_manifest=archive_path,
        )
        archive_path.write_bytes(archive_bytes)
        manifest_path.write_text(
            json.dumps(
                archive_contract,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        _log(
            "PARADIGM-gamma JCSP archive member wrote "
            f"{archive_contract['stream_count']} encoded stream payloads to "
            f"{archive_path} "
            f"(archive_sha256={archive_contract['archive_sha256']}, "
            "archive_bytes_written=True, "
            "ready_for_runtime_loader=True, "
            "ready_for_exact_eval_dispatch=False).",
            "WARN",
        )
        raise NotImplementedError(
            "PARADIGM-gamma JCSP dispatch: byte-closed JCSP archive member "
            f"was written at {archive_path} with manifest {manifest_path}, "
            "but it is not dispatch-ready. The submission runtime detects but "
            "does not emit contest raw outputs from jcsp.bin or prove output "
            "parity, strict preflight proof is missing, no lane was claimed, "
            "and no GPU/remote/eval dispatch was attempted."
        )

    # -- Cross-paradigm flag guards (PARADIGM-delta/epsilon/zeta) --------
    # Phase 1 scaffolding (blueprint at
    # ``.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md``).
    # Mirror of the fail-closed unwired-stub pattern from PARADIGM-alpha.
    # All three branches are REGISTERED-BUT-NOT-WIRED in Phase 1; full
    # dispatch wiring is staged behind:
    #   - delta: Gate 2 (apogee_int6 [contest-CUDA] eval landing)
    #   - epsilon / zeta: Gate 3 (delta Phase-2 [contest-CUDA] empirical improvement)
    # See lane registry entries lane_delta_joint_training,
    # lane_epsilon_learnable_entropy, lane_zeta_self_compress_renderer.
    if cfg.use_joint_scorer_aware:
        raise NotImplementedError(
            "PARADIGM-delta: cfg.use_joint_scorer_aware=True but the joint "
            "scorer-aware training dispatch path "
            "(tac.joint_scorer_aware_training.JointScorerAwareLoss) is "
            "REGISTERED-BUT-NOT-WIRED (Phase 1 scaffold). Refusing to fall "
            f"through to weight_compression={mode!r}. To enable, land "
            "the Phase 2 dispatch branch + integration test (Gate 2 = "
            "apogee_int6 [contest-CUDA] eval landing). See lane "
            "lane_delta_joint_training in the registry. joint_training_config_path="
            f"{cfg.joint_training_config_path!r}."
        )
    if cfg.use_learnable_entropy:
        raise NotImplementedError(
            "PARADIGM-epsilon: cfg.use_learnable_entropy=True but the learned "
            "entropy prior codec dispatch path "
            "(tac.learnable_entropy_model.LearnableEntropyModelCodec) is "
            "REGISTERED-BUT-NOT-WIRED (Phase 1 scaffold). Refusing to fall "
            f"through to weight_compression={mode!r} without a LEPR archive "
            "section. To enable, land the Phase 2 dispatch branch + "
            "integration test (Gate 3 = delta Phase-2 empirical improvement). "
            "See lane lane_epsilon_learnable_entropy in the registry."
        )
    if cfg.use_full_renderer_self_compress:
        raise NotImplementedError(
            "PARADIGM-zeta: cfg.use_full_renderer_self_compress=True but the "
            "full-renderer self-compression dispatch path "
            "(tac.self_compress_full_renderer.FullRendererSelfCompress) is "
            "REGISTERED-BUT-NOT-WIRED (Phase 1 scaffold). Refusing to fall "
            f"through to weight_compression={mode!r} without a ZETA archive "
            "section. To enable, land the Phase 2 dispatch branch + "
            "integration test (Gate 3 = delta Phase-2 empirical improvement) "
            "and the QAT loop MUST honor the >=2000 step minimum + FiLM "
            "protection per Selfcomp/Hotz revisions. See lane "
            "lane_zeta_self_compress_renderer in the registry."
        )

    # ── Lane J-NWC neural-weight-compression branch ─────────────────────
    # Gate analogous to I4LZ: NWC1 *can* faithfully roundtrip arch flags
    # because the header carries `_infer_asymmetric_config(model)` (which
    # records padding_mode / use_dilation / use_zoom_flow / depth / channels)
    # — UNLIKE I4LZ which has no header. The gate that matters for NWC is
    # the codec-checkpoint requirement: without a trained codec we fall back
    # to FP4 with a warning rather than silently shipping a stub.
    # PARADIGM-β NWCS dispatch (sensitivity-weighted neural weight codec) —
    # mirror of the NWC branch with sensitivity-bucketed codebooks.
    # cfg.weight_compression="nwcs_sensitivity" routes through
    # tac.neural_weight_codec_sensitivity.SensitivityAwareWeightCodec which
    # uses per-block sensitivity to choose codebook size (K=256 high-sens →
    # K=4 low-sens). Requires (1) trained codec checkpoint at
    # cfg.weight_codec_path AND (2) sensitivity_map_path. Both gates must be
    # present or the branch raises NotImplementedError so the operator
    # cannot accidentally ship a stub archive (silent-no-op trap class).
    if mode == "nwcs_sensitivity":
        codec_path = cfg.weight_codec_path or ""
        sens_path = cfg.sensitivity_map_path or ""
        missing = []
        if not codec_path or not Path(codec_path).exists():
            missing.append(f"cfg.weight_codec_path={codec_path!r}")
        if not sens_path or not Path(sens_path).exists():
            missing.append(f"cfg.sensitivity_map_path={sens_path!r}")
        if missing:
            raise NotImplementedError(
                f"PARADIGM-β NWCS dispatch ('nwcs_sensitivity'): missing "
                f"prerequisites — {', '.join(missing)}. The codec checkpoint "
                f"is produced by experiments/train_neural_weight_codec_sensitivity.py "
                f"(not yet built); the sensitivity map is produced by the GPU "
                f"sensitivity sweep. Without both, dispatch cannot proceed. "
                f"See lane_nwcs_sensitivity_weighted in the lane registry."
            )
        # Future operator landing: import and call
        # tac.neural_weight_codec_sensitivity.export_nwcs_renderer_container
        # with per-tensor encoded blobs from SensitivityAwareWeightCodec.
        # For now, treat as registered-but-not-fully-implemented:
        raise NotImplementedError(
            "PARADIGM-β NWCS dispatch: codec checkpoint + sensitivity map "
            "are present, but the per-tensor encoding loop "
            "(SensitivityAwareWeightCodec.encode → "
            "export_nwcs_renderer_container) is not yet wired. Operator "
            "must land that integration; reference β branch (commit 107f6fea) "
            "for the dispatch pattern."
        )

    if mode == "nwc":
        codec_path = cfg.weight_codec_path or ""
        if not codec_path or not Path(codec_path).exists():
            _log(
                f"Weight compression: NWC requested but weight_codec_path "
                f"({codec_path!r}) does not exist. Falling back to FP4 (the "
                f"NWC codec MUST be trained ahead of time via "
                f"experiments/train_neural_weight_codec.py). [Memory: "
                f"feedback_canonical_remote_bootstraps]",
                "WARN",
            )
            _mark_done(iter_dir, step_name, {
                "mode": "fp4_fallback",
                "reason": "nwc_codec_missing",
                "weight_codec_path": codec_path,
            })
            return checkpoint_path

        _log(f"Weight compression: NWC (codec={codec_path})")
        from tac.renderer import AsymmetricPairGenerator
        from tac.renderer_export import export_neural_compressed_checkpoint

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model = AsymmetricPairGenerator(
            num_classes=5, embed_dim=cfg.embed_dim,
            base_ch=cfg.base_ch, mid_ch=cfg.mid_ch,
            motion_hidden=cfg.motion_hidden, depth=cfg.depth,
            pose_dim=cfg.pose_dim, use_dsconv=cfg.use_dsconv,
            padding_mode=cfg.padding_mode, use_dilation=cfg.use_dilation,
            use_zoom_flow=cfg.use_zoom_flow,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"step_compress_weights (NWC): shape mismatch on {checkpoint_path}. "
                f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}."
            )
        model.eval()

        nwc_path = iter_dir / "renderer_nwc.bin"
        nwc_bytes = export_neural_compressed_checkpoint(
            model, codec_path=codec_path, output_path=nwc_path,
        )
        _log(f"  NWC: {nwc_bytes:,} bytes ({nwc_bytes/1024:.1f} KB)")
        _mark_done(iter_dir, step_name, {
            "mode": "nwc",
            "bytes": nwc_bytes,
            "weight_codec_path": codec_path,
        })
        return nwc_path

    if mode == "fp4":
        _log("Weight compression: FP4 (default, no additional compression)")
        _mark_done(iter_dir, step_name, {"mode": "fp4", "skipped": True})
        return checkpoint_path

    # I4LZ format has no arch header — inflate cannot recover padding_mode,
    # use_dilation, or use_zoom_flow from the compressed state dict. Forcing
    # I4LZ on a model trained with non-default arch flags would silently use
    # zeros/False at inflate → wrong reconstruction. (R24 finding: SHIRAZ-class
    # bug, but at inflate time instead of QAT time.)
    needs_arch_header = (
        cfg.padding_mode != "zeros"
        or cfg.use_dilation
        or cfg.use_zoom_flow
    )
    if mode in ("int4_lzma2", "auto") and needs_arch_header:
        _log(
            f"Weight compression: I4LZ format unsafe for padding_mode={cfg.padding_mode!r}, "
            f"use_dilation={cfg.use_dilation}, use_zoom_flow={cfg.use_zoom_flow}. "
            f"I4LZ has no arch header → inflate would silently use defaults. "
            f"Falling back to FP4 (which embeds the full arch header).",
            "WARN",
        )
        _mark_done(iter_dir, step_name, {"mode": "fp4_fallback",
                                          "reason": "i4lz_no_arch_header"})
        return checkpoint_path

    _log(f"Weight compression: {mode}")

    from tac.renderer import AsymmetricPairGenerator
    from tac.mixed_precision_export import export_int4_lzma2, load_int4_lzma2

    # Load the checkpoint into a model for re-export
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch, mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden, depth=cfg.depth,
        pose_dim=cfg.pose_dim, use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode, use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
    )
    # R32 fix: same hard-fail-on-shape-mismatch pattern as step_export.
    # Silent strict=False permits use_zoom_flow channel-count mismatches.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"step_compress_weights: shape mismatch on {checkpoint_path}. "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}. "
            f"Verify arch (use_zoom_flow={cfg.use_zoom_flow}, base_ch={cfg.base_ch}) "
            f"matches the checkpoint."
        )
    model.eval()

    int4_path = iter_dir / "renderer_int4lzma2.bin"
    fp4_path = iter_dir / "renderer.bin"

    # Load per-layer bit allocation from sensitivity sweep (if available)
    bit_allocation: dict[str, int] | None = None
    if sensitivity_json is not None and sensitivity_json.exists():
        try:
            sweep_data = json.loads(sensitivity_json.read_text())
            bit_allocation = sweep_data.get("allocation")
            if bit_allocation:
                _log(f"  Using mixed-precision allocation from: {sensitivity_json}")
                bit_counts = {}
                for b in bit_allocation.values():
                    bit_counts[b] = bit_counts.get(b, 0) + 1
                _log(f"  Allocation: {', '.join(f'{b}b:{n}' for b, n in sorted(bit_counts.items()))}")
        except (json.JSONDecodeError, KeyError) as e:
            _log(f"  Failed to load sensitivity sweep: {e}", "WARN")

    # ── Strategy 1: Int4/Mixed + LZMA2 ──
    int4_bytes = export_int4_lzma2(model, int4_path, bit_allocation=bit_allocation)
    label = "Mixed" if bit_allocation else "Int4"
    _log(f"  {label}+LZMA2: {int4_bytes:,} bytes ({int4_bytes/1024:.1f} KB)")

    if mode == "int4_lzma2":
        # Unconditionally use int4_lzma2
        _mark_done(iter_dir, step_name, {"mode": "int4_lzma2", "bytes": int4_bytes})
        return int4_path

    # ── mode == "auto": compare quality ──
    # Quick quality check: load int4 weights, compare forward pass
    restored_sd = load_int4_lzma2(int4_path)
    model_restored = AsymmetricPairGenerator(
        num_classes=5, embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch, mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden, depth=cfg.depth,
        pose_dim=cfg.pose_dim, use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode, use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
    )
    # R33 fix: same hard-fail-on-shape-mismatch pattern as step_export.
    # Without this, a corrupt/truncated I4LZ file silently restores partial
    # weights and the RMSE quality measurement compares against zeros for
    # missing layers — producing artificially low RMSE that would make the
    # 'auto' mode incorrectly choose I4LZ over FP4.
    missing_r, unexpected_r = model_restored.load_state_dict(restored_sd, strict=False)
    if missing_r or unexpected_r:
        raise RuntimeError(
            f"step_compress_weights: I4LZ restore shape mismatch. "
            f"missing={list(missing_r)[:5]} unexpected={list(unexpected_r)[:5]}. "
            f"I4LZ archive may be truncated/corrupt."
        )
    model_restored.eval()

    # Measure parameter-space RMSE
    total_se = 0.0
    total_n = 0
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model_restored.named_parameters()
    ):
        se = ((p1.detach() - p2.detach()) ** 2).sum().item()
        total_se += se
        total_n += p1.numel()
    rmse = (total_se / max(total_n, 1)) ** 0.5

    _log(f"  Int4+LZMA2 weight RMSE: {rmse:.6f}")

    # Also compare FP4 size for the auto decision
    from tac.renderer_export import export_asymmetric_checkpoint_fp4
    fp4_bytes = export_asymmetric_checkpoint_fp4(model, str(fp4_path))
    _log(f"  FP4: {fp4_bytes:,} bytes ({fp4_bytes/1024:.1f} KB)")

    # Auto decision: use int4_lzma2 if it's smaller (it almost always will be)
    # AND weight RMSE is acceptable (< 0.05, which is well within FP4 noise)
    if int4_bytes < fp4_bytes and rmse < 0.05:
        _log(f"  Auto: chose int4_lzma2 (smaller by {fp4_bytes - int4_bytes:,} bytes)")
        chosen = "int4_lzma2"
        result_path = int4_path
    else:
        _log(f"  Auto: chose fp4 (int4 RMSE={rmse:.4f} or size not better)")
        chosen = "fp4"
        result_path = fp4_path

    _mark_done(iter_dir, step_name, {
        "mode": chosen,
        "int4_bytes": int4_bytes,
        "fp4_bytes": fp4_bytes,
        "int4_rmse": round(rmse, 6),
    })
    return result_path


def step_archive(cfg: PipelineConfig, renderer_bin: Path, poses_path: Path | None,
                 iteration: int = 0, corrections_bin: Path | None = None) -> Path:
    """Build optimized submission archive.

    poses_path may be None for non-FiLM renderers (pose_dim=0). In that case
    step_pose_tto is correctly soft-skipped and the archive is built without
    poses. Contrarian council CRITICAL 2026-04-26: prior soft-skip returned
    a phantom path which crashed build_submission_archive with
    FileNotFoundError. Now: detect None or non-existent path, switch to a
    no-poses manifest variant.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    archive_path = iter_dir / "archive.zip"

    if _step_done(iter_dir, "archive"):
        _log(f"Archive already built (iter {iteration}), skipping")
        return archive_path

    _log("Building submission archive")
    from tac.submission_archive import (
        build_submission_archive, RENDERER_SUBMISSION_MANIFEST,
        RENDERER_COMPACT_MANIFEST, RENDERER_AMRC_MANIFEST, ArchiveManifest,
    )

    # 2026-04-26 Contrarian fix: handle no-poses (FiLM-less renderer) case.
    has_poses = poses_path is not None and Path(poses_path).exists()
    if not has_poses:
        _log("  Building archive WITHOUT optimized poses (no-FiLM renderer or "
             "pose_tto soft-skipped). Eval will use zero poses (safe only for "
             "non-FiLM renderers).", "WARN")
        poses_path = None

    is_bin = (poses_path.suffix == ".bin") if has_poses else False
    archive_mask_path = Path(cfg.masks_archive or cfg.masks)
    is_amrc = archive_mask_path.suffix.lower() == ".amrc"

    if is_amrc:
        # AMRC manifest: lossless argmax-RLE codec instead of AV1.
        # The pose field is .pt by default; override to .bin if requested.
        if not has_poses:
            manifest = ArchiveManifest(
                renderer_bin=True, masks_amrc=True,
            )
        elif cfg.binary_poses or is_bin:
            manifest = ArchiveManifest(
                renderer_bin=True, masks_amrc=True, optimized_poses_bin=True,
            )
        else:
            manifest = RENDERER_AMRC_MANIFEST
    else:
        if not has_poses:
            manifest = ArchiveManifest(renderer_bin=True, masks_mkv=True)
        else:
            manifest = (
                RENDERER_COMPACT_MANIFEST if (cfg.binary_poses or is_bin)
                else RENDERER_SUBMISSION_MANIFEST
            )

    # Include gradient corrections if available (Eureka 6 — Contrarian)
    corr_path = corrections_bin if (corrections_bin and corrections_bin.exists()) else None
    if corr_path:
        from tac.submission_archive import ArchiveManifest
        manifest = ArchiveManifest(
            renderer_bin=manifest.renderer_bin,
            masks_mkv=manifest.masks_mkv,
            masks_amrc=manifest.masks_amrc,
            optimized_poses_pt=manifest.optimized_poses_pt,
            optimized_poses_bin=manifest.optimized_poses_bin,
            gradient_corrections_bin=True,
        )
        _log(f"  Including gradient corrections: {corr_path} ({corr_path.stat().st_size:,} bytes)")

    result = build_submission_archive(
        output_path=archive_path,
        renderer_bin=renderer_bin,
        masks_mkv=str(archive_mask_path) if not is_amrc else None,
        masks_amrc=str(archive_mask_path) if is_amrc else None,
        optimized_poses_pt=poses_path if (has_poses and not is_bin) else None,
        optimized_poses_bin=poses_path if (has_poses and is_bin) else None,
        gradient_corrections_bin=corr_path,
        manifest=manifest,
        validate=True,
        use_brotli=cfg.brotli,
    )

    rate = 25 * result.archive_bytes / ORIGINAL_VIDEO_BYTES
    _log(f"  {result.archive_bytes:,} bytes ({result.archive_bytes/1024:.1f} KB)")
    _log(f"  Rate: {rate:.4f}")
    for name, size in sorted(result.files_found.items()):
        _log(f"    {name}: {size:,} bytes")

    _mark_done(iter_dir, "archive", {
        "bytes": result.archive_bytes, "rate": round(rate, 6),
    })
    return archive_path


def step_engineered_corrections(cfg: PipelineConfig, renderer_bin: Path,
                                 iteration: int = 0) -> Path | None:
    """Eureka 6: Compute gradient-directed SegNet corrections.

    Pre-computes sparse pixel perturbations that flip wrong SegNet predictions
    to match GT. Stored as gradient_corrections.bin for inflate-time application.
    Contest-compliant: no scorers at inflate time.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    step_name = "engineered_corrections"
    corrections_bin = iter_dir / "gradient_corrections.bin"

    # R30: use canonical _step_done/_mark_done instead of bare touch() —
    # consistent with every other step + atomic .done write.
    if _step_done(iter_dir, step_name) and corrections_bin.exists():
        _log(f"[{step_name}] Skipping — already done ({corrections_bin})")
        return corrections_bin

    _log(f"[{step_name}] Computing SegNet-targeting corrections...")
    video_path = Path(cfg.upstream) / "videos" / "0.mkv"
    # Yousfi R1 #2 fix (2026-04-26 council 5/0): wire the byte cap from cfg
    # so over-budget runs trigger the explicit top-K-drop / non-zero exit
    # path inside engineered_quant_noise instead of silently bundling a
    # rate-busting artifact (or, worse, conflating exit 2 with an unrelated
    # subprocess crash).
    cmd = [
        sys.executable, "-u", "experiments/engineered_quant_noise.py",
        "--checkpoint", str(renderer_bin),
        "--video", str(video_path),
        "--device", cfg.device,
        "--output-dir", str(iter_dir),
        "--max-artifact-bytes", str(int(cfg.engineered_max_bytes)),
    ]
    # 1-hour timeout — corrections is a single-pass gradient computation that
    # should complete in 5-30 minutes on a 4090. Anything longer is hung.
    # (R28 finding: prior call had no timeout — could block the pipeline forever.)
    # 2026-04-26: routed through _run_step so a crash surfaces the last
    # 20 stderr lines + replay command instead of silent `exit N`.
    try:
        rc = _run_step(cmd, step_name, iter_dir, timeout=3600)
    except subprocess.TimeoutExpired:
        _log(f"[{step_name}] WARNING: Timed out after 1h, skipping corrections")
        return None
    if rc == 2:
        # Exit 2 is the documented "magnitude budget exceeded OR cannot fit
        # under --max-artifact-bytes even after top-K drop" signal from
        # engineered_quant_noise. Surface it loudly so the operator knows to
        # bump the cap or relax the magnitude budget rather than confuse it
        # with a generic crash.
        _log(
            f"[{step_name}] OVER BUDGET (exit 2): magnitude or byte cap "
            f"exceeded — see {iter_dir}/{step_name}.stderr.log. "
            f"Bump cfg.engineered_max_bytes (current "
            f"{int(cfg.engineered_max_bytes):,}) or relax the magnitude "
            f"budget. Skipping corrections."
        )
        return None
    if rc != 0:
        _log(
            f"[{step_name}] WARNING: Failed (exit {rc}), skipping corrections; "
            f"see {iter_dir}/{step_name}.stderr.log"
        )
        return None

    if corrections_bin.exists():
        size = corrections_bin.stat().st_size
        _mark_done(iter_dir, step_name, {"corrections_bytes": size})
        _log(f"[{step_name}] Done: {corrections_bin} ({size:,} bytes)")
        return corrections_bin
    # Subprocess succeeded but didn't produce the expected file.
    _log(f"[{step_name}] WARNING: subprocess returned 0 but {corrections_bin} not produced")
    return None


def step_eval(cfg: PipelineConfig, renderer_bin: Path, archive_path: Path,
              iteration: int = 0, poses_path: Path | None = None) -> dict:
    """Full e2e auth evaluation through upstream scorer.

    Args:
        renderer_bin: path to renderer.bin (checkpoint for auth_eval_renderer.py)
        archive_path: path to archive.zip (for rate calculation via --archive-size-bytes)
        poses_path: optional path to optimized_poses.{bin,pt}. If None, auto-discovers
            in the archive's parent dir. Required for FiLM models (pose_dim>0); without
            it, auth_eval_renderer.py renders with zero poses and PoseNet collapses
            (verified 2026-04-26 with SHIRAZ v4: pose_d 0.342 vs ~0.011 with poses).
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # R36 fix: invalidate the .done_eval cache when the renderer or archive
    # has been modified after the cache was written. Without this, the eval
    # subcommand silently returns a stale score on a re-run with a different
    # archive — a footgun that would invalidate iterative measurement.
    done_path = iter_dir / ".done_eval"
    if _step_done(iter_dir, "eval"):
        try:
            done_mtime = done_path.stat().st_mtime
            stale = False
            for inp_path in (renderer_bin, archive_path):
                if Path(inp_path).exists() and Path(inp_path).stat().st_mtime > done_mtime:
                    stale = True
                    _log(f"Eval cache stale: {inp_path} modified after .done_eval; rerunning")
                    break
            if not stale:
                _log(f"Eval already done (iter {iteration}), skipping")
                return json.loads(done_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass  # fall through to re-run on any cache-read error

    _log("Running full e2e auth evaluation")
    archive_bytes = archive_path.stat().st_size if archive_path.exists() else 0

    # Auto-discover optimized poses if caller didn't supply one. Search both
    # the archive's iter_dir (compress flow) and the archive's parent
    # (standalone `pipeline.py eval`) for the canonical filenames produced by
    # step_pose_tto. Without this, FiLM-conditioned renderers (pose_dim>0)
    # silently render with zero poses → catastrophic PoseNet collapse
    # (verified 2026-04-26 with SHIRAZ v4: pose_d 0.342 vs ~0.011).
    #
    # DX #5 (2026-04-26): mtime-gate. A poses file OLDER than iter_dir is
    # almost certainly from a prior session against a different renderer
    # checkpoint. Silent use of a stale .bin produced the SHIRAZ 4.83
    # pseudo-baseline; warn loudly here so the operator chooses to re-run
    # pose TTO instead of getting a misleading score.
    if poses_path is None:
        try:
            iter_dir_mtime = iter_dir.stat().st_mtime
        except OSError:
            iter_dir_mtime = 0.0
        search_dirs = [iter_dir, archive_path.parent]
        for d in search_dirs:
            for cand_name in ("optimized_poses.bin", "optimized_poses.pt"):
                cand = d / cand_name
                if not cand.exists():
                    continue
                cand_mtime = cand.stat().st_mtime
                if iter_dir_mtime and cand_mtime < iter_dir_mtime:
                    age_h = (iter_dir_mtime - cand_mtime) / 3600.0
                    _log(
                        f"  Found stale {cand.name} from {age_h:.1f}h ago "
                        f"(older than iter_dir) — eval will use ZERO poses "
                        f"instead. Re-run step_pose_tto to refresh.",
                        "WARN",
                    )
                    continue
                poses_path = cand
                break
            if poses_path is not None:
                break

    cmd = [
        sys.executable, "-u", "experiments/auth_eval_renderer.py",
        "--checkpoint", str(renderer_bin),
        "--upstream-dir", cfg.upstream,
        "--device", cfg.device,
        "--archive-size-bytes", str(archive_bytes),
    ]
    if poses_path is not None and poses_path.exists():
        cmd.extend(["--poses", str(poses_path)])
        _log(f"  Eval will use poses: {poses_path.name}")
    else:
        _log("  No optimized_poses.{bin,pt} in iter_dir — eval will use zero poses "
             "(safe only for non-FiLM renderers).", "WARN")

    # R30: stream stdout to a tempfile rather than capture_output=True. A
    # 30-min auth eval emits MB of pair-by-pair logs; buffering all of it in
    # memory could OOM on small instances. The full log lives at
    # iter_dir/auth_eval.stdout.log for postmortem; we read it back to
    # extract the RESULT_JSON sentinel.
    # R34 fix: pid-suffix the log files so concurrent invocations don't
    # truncate each other's output via open("w"). After success, we rename
    # to canonical names with os.replace for postmortem-friendly paths.
    stdout_log_pid = iter_dir / f"auth_eval.stdout.{os.getpid()}.log"
    stderr_log_pid = iter_dir / f"auth_eval.stderr.{os.getpid()}.log"
    stdout_log = stdout_log_pid
    stderr_log = stderr_log_pid
    t0 = time.monotonic()
    timed_out = False
    with stdout_log.open("w") as out_f, stderr_log.open("w") as err_f:
        try:
            result = subprocess.run(cmd, stdout=out_f, stderr=err_f, timeout=3600)
        except subprocess.TimeoutExpired:
            # Mark in-stream so postmortem readers know the log was truncated.
            timed_out = True
    if timed_out:
        # Append the sentinel AFTER the with block closes the file so the
        # marker is the last line, not interleaved with kernel-buffered writes.
        with stdout_log.open("a") as out_f:
            out_f.write("\n[PIPELINE TIMEOUT — eval killed after 1h]\n")
        raise RuntimeError(
            "Auth eval timed out after 1h — should typically take 20-30 min "
            f"on T4. See {stdout_log} (last line marked TIMEOUT). "
            f"Fix: SSH in and `nvidia-smi` to confirm GPU not stuck. Then "
            f"`pkill -f auth_eval_renderer` and inspect the stdout log for "
            f"the last successful pair index — if it stalled mid-frame the "
            f"renderer is producing NaN. Re-run after `rm "
            f"{iter_dir}/.done_eval` (deleting cache so a clean retry runs)."
        )
    elapsed = time.monotonic() - t0

    _log(f"  Complete in {elapsed/60:.1f} min")

    if result.returncode != 0:
        _log(f"  Auth eval FAILED (exit {result.returncode})", "ERROR")
        # Read the last 10 lines of stderr without buffering the whole file
        try:
            stderr_tail = stderr_log.read_text().strip().split("\n")[-10:]
            for line in stderr_tail:
                _log(f"  stderr: {line}")
        except Exception:
            pass
        # NEVER mark a failed eval as done — that would cache the failure and
        # the next pipeline run would skip eval and return a phantom score.
        # (R24 finding.)
        raise RuntimeError(
            f"Auth eval failed (exit {result.returncode}). "
            f"Investigate {stderr_log}; do NOT cache this failure. "
            f"Fix: tail -100 {stderr_log} for the actual exception. Common "
            f"causes: (1) renderer.bin shape/profile mismatch — confirm "
            f"profile metadata in {renderer_bin} matches what auth_eval "
            f"expects; (2) MPS device leak — re-run with --device cuda only "
            f"(MPS auth is NOISE per CLAUDE.md, 23x PoseNet drift); (3) "
            f"stale optimized_poses — delete {iter_dir}/optimized_poses.* "
            f"and re-run step_pose_tto."
        )

    # Parse the RESULT_JSON sentinel from the stdout log (R29 schema-first
    # contract; replaces R27/R28 fragile regex).
    stdout_text = stdout_log.read_text()
    score = _extract_eval_score(stdout_text, logger=_log)

    meta = {"elapsed_s": round(elapsed), "score": score}
    _mark_done(iter_dir, "eval", meta)
    return meta


# R29: replaced fragile regex (caught bugs across rounds 28 + 29) with a
# schema-first contract. auth_eval_renderer.py now emits a single
# `RESULT_JSON: {...}` line; we json.loads + Pydantic-validate. No more
# token-by-token guessing whether a number is a score, a timing measurement,
# or a component breakdown. Schema-first per CLAUDE.md.
# R38 fix: DOTALL so `.` matches newlines. The current emitter uses
# json.dumps(separators=(',',':')) (compact, single-line) — but a future
# pretty-print would silently produce score=None without DOTALL. Defense
# in depth.
_RESULT_JSON_RE = re.compile(r"^RESULT_JSON:\s*(\{.*?\})\s*$",
                              re.MULTILINE | re.DOTALL)


# NON-NEGOTIABLE CONTRACT: any breaking change to the RESULT_JSON payload
# emitted by `experiments/auth_eval_renderer.py` MUST bump BOTH:
#   1. EXPECTED_AUTH_EVAL_SCHEMA below
#   2. The `schema_version` int in `auth_eval_renderer.py`'s _result_payload
# Adding a new optional field is NOT a breaking change (extra='ignore' tolerates
# it). Removing a field, renaming, or changing types IS a breaking change.
# Mismatched versions raise RuntimeError at parse time, surfacing the drift
# loudly per the schema-first standard.
EXPECTED_AUTH_EVAL_SCHEMA = 1


class AuthEvalResult(BaseModel):
    """Validated payload from auth_eval_renderer.py's RESULT_JSON sentinel line.

    Required fields are strictly typed. extra='ignore' so the emitter can add
    new optional fields without breaking older parsers (forward-compat).
    schema_version is checked separately against EXPECTED_AUTH_EVAL_SCHEMA;
    a bump there means coordinated parser change.
    """
    model_config = {"extra": "ignore"}
    schema_version: int
    final_score: float
    score_seg: float
    score_pose: float
    score_rate: float
    avg_segnet_dist: float
    avg_posenet_dist: float
    rate: float
    archive_size_bytes: int
    gt_size_bytes: int


def _extract_eval_score(stdout: str, logger=None) -> float | None:
    """Parse the RESULT_JSON sentinel line from auth_eval_renderer.py stdout.

    Returns the validated final_score, or None if no RESULT_JSON line was
    found. Raises ValidationError if the payload is malformed, or
    RuntimeError if the schema version doesn't match expectations.
    """
    matches = list(_RESULT_JSON_RE.finditer(stdout))
    if not matches:
        if logger:
            logger("  No RESULT_JSON line in auth_eval stdout — eval may be "
                   "running an old version without the sentinel emit", "WARN")
        return None
    # Use the LAST match — defensive against an eval that emits multiple
    # (e.g., per-iteration). The last is the final result.
    payload = matches[-1].group(1)
    parsed = AuthEvalResult.model_validate_json(payload)
    if parsed.schema_version != EXPECTED_AUTH_EVAL_SCHEMA:
        raise RuntimeError(
            f"RESULT_JSON schema_version={parsed.schema_version} does not match "
            f"expected={EXPECTED_AUTH_EVAL_SCHEMA}. Update AuthEvalResult and "
            f"EXPECTED_AUTH_EVAL_SCHEMA in lockstep with auth_eval_renderer.py."
        )
    if logger:
        logger(f"  Parsed RESULT_JSON: final={parsed.final_score:.4f} "
               f"seg={parsed.score_seg:.4f} pose={parsed.score_pose:.4f} "
               f"rate={parsed.score_rate:.4f}")
    return parsed.final_score


# ── Main pipeline ────────────────────────────────────────────────────────

def _capture_provenance(cfg: PipelineConfig) -> dict:
    """Build a full-provenance dict for `pipeline_config.json`. Captures the
    config + git hash + GPU info + library versions per CLAUDE.md non-negotiable
    'Full provenance' rule. (R25 finding: prior config dump had none of this.)

    DX #6 (2026-04-26): added ffmpeg_version, libsvtav1_version, brotli_version,
    env_vars (LD_LIBRARY_PATH / PYTHONPATH / CUBLAS_WORKSPACE_CONFIG /
    PYTORCH_CUDA_ALLOC_CONF / TAC_UPSTREAM_DIR), sys_argv, profile_dict (the
    resolved profile, not just its name), and mask_resolution. These were the
    fields Fridrich + Quantizr asked for so a CUDA score is reproducible from
    the JSON alone with no human archeology.
    """
    prov: dict = {"config": asdict(cfg)}
    from tac.cross_paradigm_wiring import build_cross_paradigm_runtime_contract

    cross_paradigm_contract = build_cross_paradigm_runtime_contract(cfg)
    if cross_paradigm_contract.any_cross_paradigm_opt_in:
        prov["cross_paradigm_runtime_contract"] = cross_paradigm_contract.to_dict()
    # Always capture argv first so even an early exception in another probe
    # leaves a record of "what command was actually run".
    prov["sys_argv"] = list(sys.argv)
    try:
        import torch
        prov["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            prov["gpu_name"] = torch.cuda.get_device_name(0)
            prov["cuda_version"] = torch.version.cuda
    except Exception as e:
        prov["torch_version_error"] = str(e)
    try:
        import platform
        prov["python_version"] = platform.python_version()
        prov["platform"] = platform.platform()
    except Exception:
        pass
    try:
        # R34 fix: timeouts on git subprocesses. NFS-backed home dirs and
        # broken post-checkout hooks can hang `git rev-parse` indefinitely;
        # provenance capture should never block the pipeline. The except
        # clause already records TimeoutExpired (subclass of Exception).
        prov["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).parent.parent, timeout=10,
        ).strip()
        prov["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            cwd=Path(__file__).parent.parent, timeout=10,
        ).strip())
    except Exception as e:
        prov["git_hash_error"] = str(e)

    # DX #6 — toolchain versions. These break the pipeline differently
    # depending on minor version (e.g. ffmpeg 4.4 lacks svtav1-params),
    # so a CUDA score reproduction needs the exact versions to know what
    # to install.
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-version"], text=True,
            stderr=subprocess.STDOUT, timeout=10,
        )
        prov["ffmpeg_version"] = out.splitlines()[0].strip() if out else None
    except Exception as e:  # missing ffmpeg, timeout, etc.
        prov["ffmpeg_version_error"] = str(e)
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-encoders"], text=True,
            stderr=subprocess.STDOUT, timeout=10,
        )
        svt = [ln.strip() for ln in out.splitlines() if "svt" in ln.lower() or "svtav1" in ln.lower()]
        prov["libsvtav1_version"] = svt[0] if svt else None
    except Exception as e:
        prov["libsvtav1_version_error"] = str(e)
    try:
        import brotli  # type: ignore
        prov["brotli_version"] = getattr(brotli, "__version__", "unknown")
    except Exception as e:
        prov["brotli_version_error"] = str(e)

    # DX #6 — env vars that materially change reproducibility. We only
    # capture the ones we know matter; capturing all of os.environ would
    # leak secrets (HF_TOKEN, AWS keys, ...).
    env_keys = (
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TAC_UPSTREAM_DIR",
        "PYTHONHASHSEED",
        "TAC_MODELS_DIR",
        "INFLATE_TTO",  # safety: must be 0 for contest-compliant runs
    )
    prov["env_vars"] = {k: os.environ.get(k) for k in env_keys}

    # DX #6 — full resolved profile. cfg.profile is just the name; the
    # caller may have CLI-overridden specific fields. Capture the dict so a
    # reproduction does not have to chase every CLI flag in argv.
    try:
        from tac.profiles import PROFILES  # type: ignore
        profile_name = getattr(cfg, "profile", None)
        if profile_name and profile_name in PROFILES:
            prof = PROFILES[profile_name]
            # Only record the dict body — strip values that aren't
            # JSON-serialisable (lambdas, classes) so json.dumps doesn't
            # crash. defaultdict / OrderedDict get coerced to plain dict.
            def _safe(v):
                try:
                    json.dumps(v)
                    return v
                except (TypeError, ValueError):
                    return repr(v)
                except Exception:
                    return "<unserialisable>"
            prov["profile_dict"] = {k: _safe(v) for k, v in dict(prof).items()}
    except Exception as e:
        prov["profile_dict_error"] = str(e)

    # DX #6 — mask resolution. Catastrophic failure 2026-04-21 was caused
    # by 48x64 masks; recording H x W in provenance makes it auditable
    # post-hoc rather than buried in the masks.mkv container metadata.
    try:
        masks_path = getattr(cfg, "masks", None) or getattr(cfg, "masks_archive", None)
        if masks_path and Path(masks_path).exists():
            try:
                ffp = subprocess.check_output(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height", "-of", "csv=p=0",
                     str(masks_path)],
                    text=True, stderr=subprocess.STDOUT, timeout=10,
                ).strip()
                if "," in ffp:
                    w, h = ffp.split(",", 1)
                    prov["mask_resolution"] = f"{h.strip()}x{w.strip()}"
            except Exception:
                # Fallback: try torch loading for .pt mask tensors
                pass
    except Exception as e:
        prov["mask_resolution_error"] = str(e)

    prov["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return prov


def _validate_cross_paradigm_runtime_contract(
    cfg: PipelineConfig,
):
    """Refuse cross-paradigm opt-ins that are only registered, not dispatchable."""
    from tac.cross_paradigm_wiring import (
        build_cross_paradigm_runtime_contract,
        raise_for_cross_paradigm_blockers,
    )

    contract = build_cross_paradigm_runtime_contract(cfg)
    if not contract.any_cross_paradigm_opt_in:
        return contract

    for warning in contract.warnings:
        _log(f"Cross-paradigm runtime contract warning: {warning}", "WARN")
    for blocker in contract.blockers:
        _log(f"Cross-paradigm runtime contract blocker: {blocker}", "ERROR")
    raise_for_cross_paradigm_blockers(contract)
    return contract


def step_multipass(
    cfg: PipelineConfig,
    final_renderer: Path,
    poses_path: Path | None,
    iteration: int,
    corrections_bin: Path | None = None,
) -> tuple[Path, dict]:
    """Lane 8 — multi-pass compress with score-feedback.

    Wraps ``step_archive`` + ``step_eval`` inside a ``MultiPassCompressor``
    that iterates encoder parameters (currently ``mask_crf``) based on
    per-pass score deltas. The output is the BEST archive across passes
    (regression-guarded). Inflate side is UNCHANGED.

    Strict-scorer-rule per CLAUDE.md: this runs at COMPRESS time only and
    invokes the contest scorer through ``step_eval`` (which uses
    ``experiments/auth_eval_renderer.py``). Inflate-time scorer loads are
    forbidden — preflight Check 91 enforces this statically.

    Returns ``(archive_path, eval_meta)`` where ``eval_meta`` is the same
    schema as ``step_eval``. The pass history JSONL is written to
    ``iter_dir/multipass_history.jsonl`` for forensics.
    """
    from tac.multipass_compressor import (
        ABSOLUTE_MAX_PASSES, MultiPassCompressor,
    )

    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    history_log = iter_dir / "multipass_history.jsonl"

    if cfg.multipass_max_passes > ABSOLUTE_MAX_PASSES:
        raise ValueError(
            f"--multipass-max-passes={cfg.multipass_max_passes} exceeds "
            f"ABSOLUTE_MAX_PASSES={ABSOLUTE_MAX_PASSES}. Council verdict "
            f"(Shannon log saturation) — see "
            f".omx/research/council_lane_8_multipass_design_20260430.md"
        )

    # Encoder closure: re-runs step_archive with the proposed mask_crf.
    # The encoder must return BYTES (the compressor's contract). We read
    # archive.zip back into memory at the end of each pass.
    def encoder(_state: object, params: dict) -> bytes:
        # Apply parameters: today only mask_crf is wired through to the
        # encoder. Other axes (pose_q_bits, block_fp_block_size,
        # residual_gain) are reserved for future sub-encoders. Coordinate-
        # descent will plateau on those axes immediately and roll forward.
        cfg.mask_crf = int(round(params.get("mask_crf", cfg.mask_crf)))
        # Bust the archive cache so step_archive re-runs.
        done_marker = iter_dir / ".done_archive"
        if done_marker.exists():
            done_marker.unlink()
        archive_path = step_archive(
            cfg, final_renderer, poses_path, iteration,
            corrections_bin=corrections_bin,
        )
        return archive_path.read_bytes()

    # Scorer closure: writes the compressor's archive bytes to a unique
    # path per pass and runs step_eval. step_eval is cached on
    # iter_dir/.done_eval; we bust the cache before each pass.
    def scorer(archive_bytes: bytes) -> float:
        eval_archive = iter_dir / "multipass_archive.zip"
        eval_archive.write_bytes(archive_bytes)
        done_marker = iter_dir / ".done_eval"
        if done_marker.exists():
            done_marker.unlink()
        meta = step_eval(
            cfg, final_renderer, eval_archive, iteration,
            poses_path=poses_path,
        )
        s = meta.get("score")
        if s is None:
            raise RuntimeError(
                f"step_eval returned None score for {eval_archive}; "
                f"see {iter_dir}/auth_eval.stdout.log for the full eval log."
            )
        scorer.last_meta = meta  # keep last meta for the return value
        return float(s)

    scorer.last_meta = {}  # type: ignore[attr-defined]

    target = cfg.multipass_target_score or 0.0
    if target == 0.0:
        # Convention: 0 == "use baseline - 0.005" — but we don't know the
        # baseline until pass 1 runs. We special-case by setting an
        # impossibly-low target so target_hit never short-circuits and the
        # eps/regression guards do the work.
        target = -1.0
    _log(
        f"Lane 8 multipass: max_passes={cfg.multipass_max_passes} "
        f"target={target:.4f} eps={cfg.multipass_eps}"
    )

    result = MultiPassCompressor(
        target_score=target,
        max_passes=cfg.multipass_max_passes,
        eps=cfg.multipass_eps,
        regression_guard=True,
        initial_params={
            "mask_crf": float(cfg.mask_crf),
        },
        log_path=history_log,
    ).compress(None, encoder, scorer)

    _log(
        f"Lane 8 multipass: best_pass_idx={result.best_pass_idx} "
        f"score={result.final_score:.4f} reverted={result.reverted} "
        f"converged={result.converged} target_hit={result.target_hit}"
    )

    # Persist the BEST archive at iter_dir/archive.zip (the canonical name).
    final_archive = iter_dir / "archive.zip"
    final_archive.write_bytes(result.final_archive_bytes)

    # Write a final summary alongside the history log.
    summary_path = iter_dir / "multipass_summary.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2))
    return final_archive, scorer.last_meta  # type: ignore[attr-defined]


def run_compress(cfg: PipelineConfig) -> None:
    """Full compress pipeline: video → archive with iterative optimization."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_cross_paradigm_runtime_contract(cfg)

    # Save full provenance for reproducibility (R25 fix: was just asdict(cfg)).
    # R30 fix: on resume, append a `resumes` list rather than overwrite the
    # original timestamp/git_hash — otherwise the .done markers reference the
    # original run but the JSON timestamp claims today.
    prov_path = output_dir / "pipeline_config.json"
    new_prov = _capture_provenance(cfg)
    if prov_path.exists():
        try:
            existing = json.loads(prov_path.read_text())
            resumes = existing.setdefault("resumes", [])
            resumes.append({
                "timestamp_utc": new_prov.get("timestamp_utc"),
                "git_hash": new_prov.get("git_hash"),
                "git_dirty": new_prov.get("git_dirty"),
                "torch_version": new_prov.get("torch_version"),
            })
            prov_path.write_text(json.dumps(existing, indent=2))
        except (json.JSONDecodeError, KeyError):
            # Existing JSON is corrupted; overwrite.
            prov_path.write_text(json.dumps(new_prov, indent=2))
    else:
        prov_path.write_text(json.dumps(new_prov, indent=2))

    t0 = time.monotonic()
    _banner(f"COMPRESSION PIPELINE — {cfg.max_iterations} iteration(s)")
    _log(f"Video: {cfg.video}")
    _log(f"Checkpoint: {cfg.checkpoint}")
    _log(f"Device: {cfg.device}")

    # 2026-04-26 Quantizr R2 + Contrarian R2 CRITICAL: --half-frame requires
    # use_zoom_flow=True so the inflate side can warp frame_t from frame_t1
    # via RadialZoomWarp. With use_zoom_flow=False the inflate falls back to
    # repeat_interleave (catastrophic distortion spike — Quantizr R1 #6).
    # Auto-disable instead of crashing the lane after 2h of training.
    if cfg.half_frame and not cfg.use_zoom_flow:
        _log("--half-frame requires use_zoom_flow=True (renderer must have "
             "RadialZoomWarp to reconstruct even frames). Profile has "
             "use_zoom_flow=False — disabling --half-frame to prevent "
             "catastrophic frame-duplication at inflate.", "WARN")
        cfg.half_frame = False

    _log(f"Optimizations: half_frame={cfg.half_frame}, binary_poses={cfg.binary_poses}, brotli={cfg.brotli}")

    # Step 0: Extract masks (once, shared across iterations)
    # FULL masks (1200 frames) for training/pose TTO
    # HALF masks (600 frames) for archive only
    full_masks_path, half_masks_path = step_extract_masks(cfg)
    cfg.masks = str(full_masks_path)  # training/TTO always uses full
    cfg.masks_archive = str(half_masks_path)  # archive uses half (if --half-frame)

    prev_score = float("inf")
    for iteration in range(cfg.max_iterations):
        _banner(f"ITERATION {iteration + 1} / {cfg.max_iterations}")

        # Step 1: Export
        renderer_bin = step_export(cfg, iteration)

        # Step 2: Pose TTO
        poses_path = step_pose_tto(cfg, iteration)

        # Step 2.5: Scorer-Jacobian sensitivity sweep (Eureka 3 — Hotz)
        # Runs BEFORE QAT so the bit allocation can inform quantization-aware
        # training. Only runs when weight_compression is not plain fp4.
        sensitivity_json: Path | None = None
        if cfg.weight_compression != "fp4":
            sensitivity_json = step_sensitivity_sweep(cfg, iteration)

        # Step 3: QAT (pass mixed-precision allocation if available)
        qat_bin = step_qat(cfg, iteration, mixed_precision_json=sensitivity_json)

        # Step 3.5: Fridrich steganalytic refinement (post-QAT polish)
        fridrich_ckpt = step_fridrich_refine(cfg, iteration)

        # Step 3.7: Weight compression (int4+LZMA2 or FP4, per cfg.weight_compression)
        if fridrich_ckpt.exists() and fridrich_ckpt.suffix == ".pt":
            best_ckpt = fridrich_ckpt
        else:
            # Fall back to QAT best, then raw export. R33 fix: filename is
            # qat_best_float.pt (per qat_finetune.py), not renderer_qat_best.pt.
            # The wrong name made this fallback ALWAYS fall through to
            # cfg.checkpoint, silently using pre-QAT weights.
            qat_best = Path(cfg.output_dir) / f"iter_{iteration}" / "qat_best_float.pt"
            best_ckpt = qat_best if qat_best.exists() else Path(cfg.checkpoint)

        cross_paradigm_weight_path = (
            cfg.use_sensitivity_weighted
            or cfg.use_joint_codec_stack
            or cfg.use_joint_scorer_aware
            or cfg.use_learnable_entropy
            or cfg.use_full_renderer_self_compress
        )
        if (cfg.weight_compression != "fp4" or cross_paradigm_weight_path) and best_ckpt.exists():
            final_renderer = step_compress_weights(
                cfg, best_ckpt, iteration, sensitivity_json=sensitivity_json,
            )
        elif best_ckpt.exists() and best_ckpt.suffix == ".pt":
            # R32 fix: do NOT re-call step_export — its .done_export marker
            # was set at iteration start with the ORIGINAL cfg.checkpoint
            # path, so it would silently return the stale renderer.bin and
            # discard the Fridrich-refined weights. Inline the export here
            # to a distinct filename (renderer_refined.bin) so the
            # downstream consumer sees the right artifact.
            final_renderer = _export_refined_checkpoint(cfg, best_ckpt, iteration)
        else:
            final_renderer = qat_bin if qat_bin.exists() else renderer_bin

        # Step 3.8: Engineered SegNet corrections (Eureka 6 — Contrarian).
        # 2026-04-26 Yousfi R2 CRITICAL: gated behind cfg.use_engineered_corrections
        # because Fridrich VETOed shipping (square-root-law violation; concentrates
        # max-amplitude perturbations on most-detected pixels) AND every
        # unconditional invocation cost 5-30min of compute.
        corrections_bin = (
            step_engineered_corrections(cfg, final_renderer, iteration)
            if cfg.use_engineered_corrections else None
        )

        # Step 4: Archive (include corrections if available)
        # Lane 8: when --multipass is set, step_archive + step_eval are
        # invoked together inside MultiPassCompressor's outer loop, which
        # iterates encoder parameters based on per-pass score feedback.
        # Compress-time only — strict-scorer-rule per CLAUDE.md.
        if cfg.multipass:
            archive_path, eval_result = step_multipass(
                cfg, final_renderer, poses_path, iteration,
                corrections_bin=corrections_bin,
            )
        else:
            archive_path = step_archive(cfg, final_renderer, poses_path, iteration,
                                         corrections_bin=corrections_bin)

            # Step 5: Eval (pass renderer.bin for scoring, archive.zip for rate,
            # and poses for FiLM conditioning — see step_eval docstring).
            eval_result = step_eval(cfg, final_renderer, archive_path, iteration,
                                     poses_path=poses_path)
        score = eval_result.get("score")

        if score is not None:
            improvement = prev_score - score
            _log(f"  Score: {score:.4f} (improvement: {improvement:+.4f})")

            if iteration > 0 and improvement < cfg.convergence_tol:
                _log(f"  Converged (improvement {improvement:.4f} < tol {cfg.convergence_tol})")
                break

            prev_score = score

        # For next iteration: use QAT output as starting point.
        # R33 fix: filename is qat_best_float.pt (per qat_finetune.py), not
        # renderer_qat_best.pt. The wrong name made this guard ALWAYS False,
        # so cfg.checkpoint never advanced — every multi-iteration run was
        # silently equivalent to --max-iterations 1.
        if iteration < cfg.max_iterations - 1:
            qat_ckpt = Path(cfg.output_dir) / f"iter_{iteration}" / "qat_best_float.pt"
            if qat_ckpt.exists():
                cfg.checkpoint = str(qat_ckpt)
                _log(f"  Next iteration starts from: {qat_ckpt}")
            else:
                _log(f"  WARN: qat_best_float.pt not found in iter_{iteration} — "
                     f"next iteration will re-train from original cfg.checkpoint")

    total = time.monotonic() - t0
    _banner(f"PIPELINE COMPLETE — {total/60:.1f} min total")


def run_eval(args: argparse.Namespace) -> None:
    """Evaluate an existing archive against an existing renderer checkpoint.

    R29 fix: prior version passed `archive` as both args to step_eval, which
    sent the archive .zip as --checkpoint to auth_eval_renderer.py. The eval
    script expected a renderer .pt/.bin → cryptic load error. Every
    `pipeline.py eval` invocation was broken.
    R30 fix: support --profile so arch fields can match training.
    """
    # If --profile given, load matching arch fields onto args.
    if getattr(args, "profile", None):
        _apply_profile(args, args.profile, user_provided_flags=set())
    pcfg_field_names = {f.name for f in fields(PipelineConfig)}
    base_kwargs = {
        "video": args.video, "device": args.device,
        "upstream": getattr(args, "upstream", "upstream"),
        "output_dir": str(Path(args.archive).parent),
    }
    # Pull any arch fields the profile injected; they're already on args.
    for field_name in pcfg_field_names:
        if field_name in base_kwargs:
            continue
        if hasattr(args, field_name):
            base_kwargs[field_name] = getattr(args, field_name)
    cfg = PipelineConfig(**base_kwargs)
    archive = Path(args.archive)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"--checkpoint {checkpoint} does not exist. eval needs both the "
            f"archive (for rate calculation) and the renderer (for forward pass)."
        )
    result = step_eval(cfg, checkpoint, archive)
    score = result.get("score")
    # R30 fix: was `if score:` which is falsy for score=0.0 (a perfect score).
    # Use explicit None check, mirroring the pattern in run_compress.
    if score is not None:
        _log(f"Final score: {score:.4f}")


# ── CLI ──────────────────────────────────────────────────────────────────

def _user_provided_flags(
    parser: argparse.ArgumentParser, argv: list[str],
    *, active_subcommand: str,
) -> set[str]:
    """Return the set of arg.dest values that the user explicitly passed in argv.

    Walks the top-level parser AND the active subparser only — NOT every
    subparser, which could mark the wrong dest if two subparsers share an
    option string with different dest names. (R27 finding.)

    `active_subcommand` is REQUIRED (kw-only, no default). Earlier versions
    accepted None and silently returned an empty set for subparser flags —
    a footgun that would clobber every CLI override with profile values.
    (R28 finding.)
    """
    if not active_subcommand or not isinstance(active_subcommand, str):
        raise ValueError(
            f"active_subcommand must be the name of the active subparser "
            f"(e.g., 'compress'). Got {active_subcommand!r}."
        )
    user_set: set[str] = set()
    actions: list[argparse.Action] = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub = action.choices.get(active_subcommand)
            if sub is not None:
                actions.extend(sub._actions)

    for action in actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            matched = False
            for token in argv:
                if token == opt or token.startswith(opt + "="):
                    user_set.add(action.dest)
                    matched = True
                    break
            if matched:
                break
    return user_set


def _apply_profile(args: argparse.Namespace, profile_name: str,
                   user_provided_flags: set[str] | None = None) -> None:
    """Load PROFILES[profile_name]. Profile fills in defaults; explicit CLI
    flags from `user_provided_flags` win. Profile keys that don't match a
    PipelineConfig field are SILENTLY SKIPPED (they're for training scripts
    like train_distill.py that consume the profile separately).

    However, a profile key that LOOKS like a PipelineConfig field but is
    misspelled (Levenshtein close-match) FAILS LOUDLY — that's the SHIRAZ
    class. E.g. `motino_hidden` is close to `motion_hidden` → fail.
    `use_tto` is not close to anything → silent skip (training-only).

    `user_provided_flags` is a set of arg names that the operator explicitly
    passed on the command line. These are NOT overwritten by the profile.
    """
    import difflib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tac.profiles import PROFILES
    if profile_name not in PROFILES:
        raise SystemExit(
            f"unknown --profile {profile_name!r}. Available: {sorted(PROFILES.keys())}"
        )
    profile = PROFILES[profile_name]
    user_provided_flags = user_provided_flags or set()

    pcfg_fields = {f.name for f in fields(PipelineConfig)}
    likely_typos: list[tuple[str, str]] = []
    for key in profile.keys():
        if key in pcfg_fields:
            continue
        # cutoff=0.85: tight enough to flag motino_hidden→motion_hidden but
        # not loose enough to flag every training-only key as a typo of
        # something. Empirical: 0.85 catches 1-2 char edits on names ≥6 chars.
        close = difflib.get_close_matches(key, pcfg_fields, n=1, cutoff=0.85)
        if close:
            likely_typos.append((key, close[0]))
    if likely_typos:
        msg_parts = [f"  {k!r} → did you mean {v!r}?" for k, v in likely_typos]
        raise SystemExit(
            f"profile {profile_name!r} has likely-typo keys (close-match to "
            f"a real PipelineConfig field):\n" + "\n".join(msg_parts) +
            "\n\nFix the typos or, if intentional, rename the key to something "
            "clearly distinct from existing PipelineConfig field names."
        )

    for key, value in profile.items():
        if key in pcfg_fields and key not in user_provided_flags:
            setattr(args, key, value)
    args._resolved_profile = profile_name


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Canonical compression pipeline for comma.ai video compression challenge",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compress subcommand
    comp = sub.add_parser("compress", help="Compress video to submission archive")
    comp.add_argument("--profile", default=None,
                      help="Named profile from src/tac/profiles.py — overrides every "
                           "matching arch/discipline arg with the profile value. The "
                           "canonical way to launch (no SHIRAZ-class arg drift).")
    comp.add_argument("--video", required=True, help="Input video path")
    comp.add_argument("--checkpoint", required=True, help="Trained renderer checkpoint")
    comp.add_argument("--masks", default="", help="FULL masks (1200 frames) for training/TTO")
    comp.add_argument("--masks-archive", default="", help="HALF-FRAME masks (600 frames) for archive only")
    comp.add_argument("--output-dir", default="experiments/results/pipeline")
    comp.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    comp.add_argument("--upstream", default="upstream")
    # Architecture
    comp.add_argument("--base-ch", type=int, default=36)
    comp.add_argument("--mid-ch", type=int, default=60)
    comp.add_argument("--motion-hidden", type=int, default=32)
    comp.add_argument("--depth", type=int, default=1)
    comp.add_argument("--pose-dim", type=int, default=6)
    comp.add_argument("--embed-dim", type=int, default=6)
    comp.add_argument("--use-dsconv", action="store_true", default=False)
    comp.add_argument("--padding-mode", type=str, default="replicate",
                      choices=["zeros", "reflect", "replicate", "circular"])
    comp.add_argument("--use-dilation", action="store_true")
    comp.add_argument("--use-zoom-flow", action="store_true",
                      help="GREEN profile: 4ch MotionPredictor + RadialZoomWarp")
    # Training-discipline boolean flags (passed through to step_fridrich_refine,
    # which re-invokes train_distill.py). Profile-driven; usually set via --profile.
    comp.add_argument("--use-swa", action="store_true")
    comp.add_argument("--use-per-class-weights", action="store_true")
    comp.add_argument("--freeze-motion-phase2", action="store_true")
    comp.add_argument("--freeze-renderer-phase3", action="store_true")
    comp.add_argument("--beneficial-quant-noise", action="store_true")
    # Yousfi #3: spatially-adaptive quantization noise (UNIWARD)
    comp.add_argument("--use-variance-noise", action="store_true",
                      help="UNIWARD-aligned: noise concentrated in textured regions "
                           "where the scorer is steganographically blind (Yousfi #3)")
    comp.add_argument("--variance-noise-weight", type=float, default=0.1)
    comp.add_argument("--variance-noise-base-std", type=float, default=2.0)
    comp.add_argument("--variance-noise-kernel", type=int, default=8)
    comp.add_argument("--variance-noise-mode", type=str, default="variance",
                      choices=["variance", "inverse_variance", "wavelet_db4"])
    # Yousfi #5: ScanNet-style spatial uncertainty maps
    comp.add_argument("--use-uncertainty-loss", action="store_true",
                      help="Yousfi #5: weight reconstruction loss by inverse SegNet "
                           "softmax entropy on the GT (focus on confident pixels)")
    comp.add_argument("--uncertainty-loss-weight", type=float, default=0.1,
                      help="Multiplier on uncertainty-weighted L1 (default 0.1)")
    comp.add_argument("--uncertainty-loss-floor", type=float, default=0.1,
                      help="Per-pixel weight floor (default 0.1)")
    # Optimization
    comp.add_argument("--pose-max-steps", type=int, default=2000)
    comp.add_argument("--pose-lr", type=float, default=0.01)
    comp.add_argument("--pose-batch-pairs", type=int, default=16,
                      help="OOM mitigation knob — reduce on smaller GPUs (R34)")
    comp.add_argument("--qat-max-epochs", type=int, default=500)
    comp.add_argument("--qat-lr", type=float, default=5e-5)
    # Archive
    comp.add_argument("--half-frame", action="store_true")
    comp.add_argument("--binary-poses", action="store_true")
    comp.add_argument("--brotli", action="store_true")
    comp.add_argument("--mask-crf", type=int, default=50)
    # Weight compression
    comp.add_argument("--weight-compression", type=str, default="fp4",
                      choices=["fp4", "int4_lzma2", "auto", "nwc", "nwcs_sensitivity"],
                      help="Weight compression: fp4 (default), int4_lzma2, auto, "
                           "nwc, or nwcs_sensitivity")
    comp.add_argument("--weight-codec-path", default="",
                      help="Pretrained neural weight codec checkpoint for nwc/nwcs_sensitivity")
    # Cross-paradigm operator-visible controls. Defaults preserve legacy
    # behavior; registered-but-unwired lanes still fail closed in their step
    # guards instead of silently falling through.
    comp.add_argument("--use-sensitivity-weighted", action="store_true",
                      help="PARADIGM-beta: route weight compression through sensitivity-weighted OWV3")
    comp.add_argument("--sensitivity-map-path", default="",
                      help="Serialized real SensitivityMap artifact required by beta lanes")
    comp.add_argument("--owv3-bit-budget-ratio", type=float, default=0.7)
    comp.add_argument("--owv3-protect-threshold", type=float, default=1e-3)
    comp.add_argument("--use-joint-codec-stack", action="store_true",
                      help="PARADIGM-gamma: build JCSP member; fails closed until runtime parity")
    comp.add_argument("--jcsp-score-marginals-path", default="",
                      help="Per-tensor dScore/dByte marginals artifact required by JCSP")
    comp.add_argument("--mask-codec", default="av1_monochrome",
                      choices=["av1_monochrome", "argmax_rle", "nerv", "wavelet", "vqvae", "grayscale_lut"],
                      help="PARADIGM-alpha mask codec; research codecs fail closed until wired")
    comp.add_argument("--use-raft-init", action="store_true",
                      help="PARADIGM-la-pose: request RAFT pose initialization; fails closed until wired")
    comp.add_argument("--use-riemannian-tto", action="store_true",
                      help="PARADIGM-la-pose: route pose TTO through SE(3) Riemannian optimizer")
    comp.add_argument("--use-joint-scorer-aware", action="store_true",
                      help="PARADIGM-delta: registered scorer-aware training flag; fails closed")
    comp.add_argument("--joint-training-config-path", default="",
                      help="JointTrainingConfig artifact for scorer-aware training")
    comp.add_argument("--use-learnable-entropy", action="store_true",
                      help="PARADIGM-epsilon: registered learned entropy prior flag; fails closed")
    comp.add_argument("--use-full-renderer-self-compress", action="store_true",
                      help="PARADIGM-zeta: registered full-renderer self-compression flag; fails closed")
    # Iteration
    comp.add_argument("--max-iterations", type=int, default=10,
                      help="Safety bound on convergence cycles (stops earlier when converged)")
    # Lane 8: multi-pass compress
    comp.add_argument("--multipass", action="store_true",
                      help="Lane 8: wrap step_archive + step_eval in a MultiPassCompressor "
                           "outer loop with score-feedback parameter adjustment "
                           "(compress-time only, strict-scorer-rule per CLAUDE.md). "
                           "See .omx/research/council_lane_8_multipass_design_20260430.md")
    comp.add_argument("--multipass-max-passes", type=int, default=3,
                      help="Multi-pass safety bound (council verdict: 3 default, 5 absolute cap)")
    comp.add_argument("--multipass-target-score", type=float, default=0.0,
                      help="Multi-pass target score (lower=better). "
                           "0 = sentinel — never short-circuit on target_hit; "
                           "rely on eps + regression guard + max-passes for stop. "
                           "Set explicitly to e.g. 1.045 to trip target_hit and "
                           "early-stop the loop.")
    comp.add_argument("--multipass-eps", type=float, default=1e-3,
                      help="Multi-pass eps stop threshold (default 1e-3 = below scorer noise floor)")

    # eval subcommand
    ev = sub.add_parser("eval", help="Evaluate an existing archive")
    ev.add_argument("--archive", required=True, help="Path to archive.zip (for rate calculation)")
    ev.add_argument("--checkpoint", required=True,
                    help="Renderer checkpoint (.pt or .bin) for forward pass. R29 fix.")
    ev.add_argument("--video", required=True, help="GT video for scoring")
    ev.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    # R30: auth_eval_renderer.py reads arch from checkpoint header for .bin
    # formats but inherits PipelineConfig defaults for .pt loading. Make
    # arch overridable so an eval-only invocation can match the training arch.
    ev.add_argument("--profile", default=None,
                    help="Optional profile name to populate arch fields (matches compress)")
    ev.add_argument("--upstream", default="upstream")

    args = parser.parse_args()

    if args.command == "compress":
        # Discover which flags the user explicitly typed (vs argparse defaults)
        # by walking sys.argv. This preserves "explicit CLI overrides profile"
        # semantics so `--profile X --base-ch 999` actually uses base_ch=999.
        # Anything else is filled in from the profile.
        user_provided = _user_provided_flags(parser, sys.argv[1:], active_subcommand="compress")
        if getattr(args, "profile", None):
            _apply_profile(args, args.profile, user_provided_flags=user_provided)
        # Build PipelineConfig from every args field that overlaps a
        # dataclass field. This automatically picks up any new field added
        # to PipelineConfig OR set by _apply_profile, with no risk of
        # forgetting to thread one through (R26 finding: 7 fields were
        # silently stuck at dataclass defaults under the manual approach).
        pcfg_field_names = {f.name for f in fields(PipelineConfig)}
        kwargs = {k: v for k, v in vars(args).items() if k in pcfg_field_names}
        cfg = PipelineConfig(**kwargs)
        run_compress(cfg)
    elif args.command == "eval":
        run_eval(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
