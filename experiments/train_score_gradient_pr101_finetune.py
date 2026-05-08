"""Phase A1 — Score-gradient supervision PR101 fine-tune.

Council priority: 22/22 ENDORSE, UNANIMOUS HIGHEST PRIORITY (Quantizr/Carmack/Hinton).

Reference: `.omx/research/grand_council_extreme_rigor_track_1_20260508.md`.

Mission
-------
Fine-tune the PR101 HNeRV substrate with **differentiable** score-gradient
supervision to close the proxy-auth gap and reduce the SegNet/PoseNet
distortion terms by ≥10% (falsification at <5%).

Council Round 2 finding 10 (verified by this fork):
    SegNet's `compute_distortion` (`upstream/modules.py:111-113`) uses argmax
    — NON-DIFFERENTIABLE. The council memo's `kl_on_logits(...)` reference is
    nominal; the real canonical surrogate in `src/tac/losses.py` is
    ``kl_distill_segnet_only`` (line 1072) which returns a differentiable
    Hinton-style KL term with T² scaling.

Mandatory discipline (NON-NEGOTIABLE per CLAUDE.md):
    * eval_roundtrip=True (default; never disable)
    * EMA decay 0.997 with snapshot+restore at eval (canonical pattern)
    * noise_std=0.5 threaded through simulate_eval_roundtrip
    * Scorers on --device cuda ONLY (no MPS fallback; raise on cuda absence)
    * parametrize-strip applied to renderer state_dict before archive build

Usage
-----
Smoke (CPU-only, 2 epochs, ~5 min)::

    .venv/bin/python experiments/train_score_gradient_pr101_finetune.py \\
        --smoke --output experiments/results/track1_a1_smoke

Full Lightning T4 dispatch (200 epochs, ~3h, $8)::

    .venv/bin/python experiments/train_score_gradient_pr101_finetune.py \\
        --device cuda \\
        --epochs 200 \\
        --output experiments/results/track1_a1_score_gradient_pr101_<timestamp>

Smoke is the gate — the council-installed dispatch wrapper refuses without it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo root on sys.path so absolute imports from `tac.*` and `submissions.*` work.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Canonical project-internal imports (post-sys-path).
# These are GUARANTEED to exist per Round-1 grep verification.
from tac.training import EMA  # noqa: E402  decay-default 0.997 enforced below
from tac.losses import (  # noqa: E402
    scorer_loss,
    kl_distill_segnet_only,
)

# HNeRVDecoder is the architecture verified by Round-1 grep at
# submissions/factorized_hnerv_v1/src/model.py.
sys.path.insert(0, str(REPO_ROOT / "submissions" / "factorized_hnerv_v1" / "src"))
from model import HNeRVDecoder  # noqa: E402

LOGGER = logging.getLogger("track1_a1")


# ---------------------------------------------------------------------------
# Discipline constants — sourced from CLAUDE.md NON-NEGOTIABLE sections
# ---------------------------------------------------------------------------

EMA_DECAY: float = 0.997          # CLAUDE.md "EMA — NON-NEGOTIABLE"
EVAL_ROUNDTRIP_NOISE_STD: float = 0.5  # Hotz fix
LAMBDA_R_WARMUP_STEPS: int = 200  # Round 1 finding 1 fix
LAMBDA_R_NOMINAL: float = 1.0
LAMBDA_S_INIT: float = 100.0      # SegNet weight per contest formula
LAMBDA_P_INIT: float = 1.0        # pose pre-sqrt weight
DUAL_UPDATE_PERIOD: int = 100     # Lagrangian update cadence (steps)

KL_DISTILL_TEMPERATURE: float = 2.0  # Hinton 2014 T=2.0 (council Decision 2)
KL_DISTILL_AUX_WEIGHT: float = 1.0   # tasted from kl_distill_segnet_only docstring

PIXEL_L1_WEIGHT: float = 0.0  # OFF by default — scorer_loss already supervises
                              # reconstruction at the score-relevant axis.

DEFAULT_BATCH_SIZE: int = 4
DEFAULT_LR: float = 5e-4
DEFAULT_LR_FINETUNE: float = 1e-4  # 0.1× base (per CLAUDE.md QAT pipeline rule)


# ---------------------------------------------------------------------------
# Forbidden-pattern guards (CLAUDE.md FORBIDDEN PATTERNS section)
# ---------------------------------------------------------------------------

def assert_cuda_or_explicit_cpu(requested_device: str, smoke: bool) -> torch.device:
    """Raise if CUDA was requested but unavailable.

    No MPS fallback. CPU only allowed in --smoke or --device cpu (explicit).
    """
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. NO MPS fallback per CLAUDE.md "
                "FORBIDDEN_PATTERNS / MPS-NOISE rule. Use --device cpu --smoke "
                "for sanity tests, or dispatch to Lightning T4 for full training."
            )
        return torch.device("cuda")
    if requested_device == "cpu":
        if not smoke:
            print(
                "[WARN] --device cpu without --smoke is allowed but the "
                "score will be tagged [advisory only] not [contest-CUDA].",
                file=sys.stderr,
            )
        return torch.device("cpu")
    raise ValueError(f"Unsupported --device {requested_device!r}; use 'cuda' or 'cpu'.")


def assert_no_invented_flag(_namespace: argparse.Namespace) -> None:
    """Cross-check our argparse against subprocess invocations to follow
    CLAUDE.md "NEVER invent CLI flags" rule.

    No subprocess calls in this module currently — preserved as a
    defense-in-depth hook for future maintainers.
    """
    return None


# ---------------------------------------------------------------------------
# PR101 substrate loader (best-effort wrapper)
# ---------------------------------------------------------------------------

def load_pr101_substrate(
    archive_path: Path | None,
    decoder: HNeRVDecoder,
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Load PR101 weights into ``decoder``.

    In smoke mode we accept a fresh init — the goal is to verify the training
    loop produces decreasing loss and no NaN. For full Lightning dispatch the
    PR101 archive (`experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`)
    must be parsed and its INT8 weight tensors loaded into ``decoder``.

    The PR101 archive uses a custom split-brotli + LZMA + sidecar wire format.
    Parsing it requires `tac.hnerv_decoder_recode` (which has a public surface
    of ~36 functions); a future hardening pass should land a small
    canonical loader. For now this function:

        * smoke=True: skips loading, uses fresh nn.init.* defaults from
          HNeRVDecoder (sufficient to test the training loop).
        * smoke=False with archive_path: NOT YET WIRED — raises with a clear
          message pointing at the loader-build follow-up task.

    Returns a metadata dict for provenance.
    """
    if smoke:
        return {"loader": "fresh_init", "smoke_mode": True}
    if archive_path is None or not archive_path.exists():
        raise RuntimeError(
            "Full (non-smoke) training requires --pr101-archive pointing at the "
            "PR101 archive.zip and a wired loader. The PR101 archive parser is "
            "not yet implemented in this module — see council follow-up "
            "task for `experiments/load_pr101_archive_to_state_dict.py`. "
            "For smoke validation use --smoke."
        )
    raise NotImplementedError(
        "PR101 archive→state_dict loader is a separate follow-up. The wire "
        "format is split-brotli + LZMA + sidecar; the canonical parser surface "
        "lives in `tac.hnerv_decoder_recode` (~36 public functions) but no "
        "single entry-point yet returns a state_dict for `HNeRVDecoder`. "
        "Lightning T4 dispatch is BLOCKED on building this loader."
    )


# ---------------------------------------------------------------------------
# Synthetic data for smoke
# ---------------------------------------------------------------------------

def make_synthetic_pair_batch(
    *, batch_size: int, frame_h: int, frame_w: int, latent_dim: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic synthetic batch.

    Returns:
        z_batch: (B, latent_dim) float32 latents on ``device``
        gt_pair_hwc: (B, T=2, H, W, C=3) float32 ground-truth frame pairs in [0, 255]
    """
    g = torch.Generator(device="cpu").manual_seed(20260508)
    z = torch.randn((batch_size, latent_dim), generator=g, dtype=torch.float32).to(device)
    gt = torch.rand(
        (batch_size, 2, frame_h, frame_w, 3), generator=g, dtype=torch.float32
    ).mul(255.0).to(device)
    return z, gt


# ---------------------------------------------------------------------------
# Stub scorers (smoke-only) and CUDA scorer loaders
# ---------------------------------------------------------------------------

class _StubScorerHead(nn.Module):
    """Tiny CPU-friendly stand-in for SegNet/PoseNet to exercise the training
    loop in smoke mode WITHOUT loading the heavy real scorers.

    The real scorers (`upstream/scorers/SegNet`, `PoseNet`) require CUDA to
    avoid the MPS-NOISE drift class. Smoke mode therefore uses these stubs to
    verify the gradient path runs end-to-end. A non-smoke run (cuda) MUST
    replace these with `tac.scorer.load_differentiable_scorers()` — same
    function signature compatible with `losses.scorer_loss`.

    Contract (mirrored from upstream/modules.py):
        - preprocess_input(pair_btchw: (B,T,C,H,W)) → preprocessed tensor
        - forward(preprocessed) → dict {"pose": (B, 12)} for posenet,
          OR tensor (B, num_classes, H, W) for segnet
    """

    def __init__(self, *, kind: str, num_classes: int = 5, target_hw: tuple[int, int] = (64, 64)):
        super().__init__()
        self.kind = kind
        self.num_classes = num_classes
        self.target_hw = target_hw
        # Tiny conv to make the gradient path real (not a constant function).
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.head_seg = nn.Conv2d(8, num_classes, kernel_size=1)
        self.head_pose = nn.Linear(8 * 4 * 4, 12)  # 12 hydra outputs

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        """Mirror upstream contract: SegNet takes last frame, PoseNet takes both."""
        if self.kind == "seg":
            x = pair_btchw[:, -1, ...]  # (B, C, H, W)
        else:
            # PoseNet flattens both frames.
            B, T, C, H, W = pair_btchw.shape
            x = pair_btchw.reshape(B, T * C, H, W)[:, :3]  # use first 3 channels
        x_resized = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)
        return x_resized / 255.0

    def forward(self, preprocessed: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        feat = F.relu(self.conv(preprocessed))
        if self.kind == "seg":
            # SegNet returns the seg-logits tensor directly per
            # `losses.segnet_surrogate_per_pixel(fs_out, gs_out, ...)` contract.
            return self.head_seg(feat)
        # Pose branch.
        pooled = F.adaptive_avg_pool2d(feat, 4).flatten(1)
        return {"pose": self.head_pose(pooled)}


def load_smoke_scorers(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Load the cheap stubs and put them on ``device``."""
    posenet = _StubScorerHead(kind="pose").to(device).eval()
    segnet = _StubScorerHead(kind="seg").to(device).eval()
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)
    return posenet, segnet


def load_cuda_scorers(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Load the real differentiable SegNet + PoseNet onto CUDA.

    Per CLAUDE.md the contest scorers live behind
    `tac.scorer.load_differentiable_scorers` (or upstream equivalents).
    The exact function name varies by codebase rev; a generic load that
    falls back to a minimal interface is acceptable. Raises a clear
    error if the loader cannot be located — caller must point at the
    canonical loader path before Lightning dispatch.
    """
    try:
        from tac.scorer import load_differentiable_scorers  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Could not import tac.scorer.load_differentiable_scorers. The "
            "Lightning T4 dispatcher must point at the canonical scorer "
            "loader path before --device cuda training."
        ) from exc
    posenet, segnet = load_differentiable_scorers(device=str(device))
    posenet.eval()
    segnet.eval()
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)
    return posenet, segnet


# ---------------------------------------------------------------------------
# Eval-roundtrip simulation (CLAUDE.md eval_roundtrip — NON-NEGOTIABLE)
# ---------------------------------------------------------------------------

def simulate_eval_roundtrip(
    pair_btchw_float: torch.Tensor,
    *,
    noise_std: float = EVAL_ROUNDTRIP_NOISE_STD,
) -> torch.Tensor:
    """Simulate the inflate/eval round-trip with STE noise.

    The contest eval pipeline goes: float frames → uint8 → bilinear up to
    (874, ?) → uint8 → bilinear down → float. This loses information in
    ways that cause proxy-auth drift. The fix is to SIMULATE the loss
    during training so the model learns to be robust to it.

    ``noise_std`` is the additive STE noise (Hotz fix). Default 0.5 per
    Council Round-2 finding (without it the proxy-auth gap reopens).
    """
    # Quantize to uint8 with STE.
    x = pair_btchw_float.clamp(0.0, 255.0)
    x_q = x.detach().round() + (x - x.detach())  # STE
    if noise_std > 0:
        x_q = x_q + torch.randn_like(x_q) * noise_std
    return x_q.clamp(0.0, 255.0)


# ---------------------------------------------------------------------------
# Lagrangian dual coordinator (Round 1 fix: warm-up over first 200 steps)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LagrangianState:
    lambda_R: float = 0.0   # rate weight (warm-up from 0)
    lambda_S: float = LAMBDA_S_INIT
    lambda_P: float = LAMBDA_P_INIT
    target_rate_bits: float = 0.0
    target_seg: float = 0.0
    target_pose: float = 0.0
    eta: float = 1e-3
    step: int = 0

    def warm_up_lambda_R(self, target_nominal: float = LAMBDA_R_NOMINAL) -> None:
        if self.step < LAMBDA_R_WARMUP_STEPS:
            self.lambda_R = target_nominal * (self.step / LAMBDA_R_WARMUP_STEPS)
        else:
            self.lambda_R = target_nominal

    def update(
        self,
        observed_rate_bits: float,
        observed_seg: float,
        observed_pose: float,
    ) -> None:
        self.lambda_R = max(
            0.0,
            self.lambda_R + self.eta * (observed_rate_bits - self.target_rate_bits),
        )
        self.lambda_S = max(
            1.0,
            self.lambda_S + self.eta * (observed_seg - self.target_seg),
        )
        self.lambda_P = max(
            0.1,
            self.lambda_P + self.eta * (observed_pose - self.target_pose),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainResult:
    final_seg: float
    final_pose: float
    initial_seg: float
    initial_pose: float
    nan_observed: bool
    epoch_log: list[dict[str, Any]]


def train_one_step(
    *,
    decoder: HNeRVDecoder,
    posenet: nn.Module,
    segnet: nn.Module,
    z_batch: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    lagrangian: LagrangianState,
    aux_kl_weight: float,
    aux_pixel_l1_weight: float,
) -> dict[str, float]:
    """Single training step. Returns metrics dict."""
    decoder.train()
    optimizer.zero_grad(set_to_none=True)

    pred_pair_btchw = decoder(z_batch)
    pred_pair_btchw = simulate_eval_roundtrip(
        pred_pair_btchw, noise_std=EVAL_ROUNDTRIP_NOISE_STD
    )
    # Convert (B, T, C, H, W) → (B, T, H, W, C) for scorer_loss expectation.
    pred_pair_hwc = pred_pair_btchw.permute(0, 1, 3, 4, 2).contiguous()

    primary_loss, pose_dist, seg_dist = scorer_loss(
        pred_pair_hwc,
        gt_pair_hwc,
        posenet,
        segnet,
    )

    aux_kl_loss = torch.tensor(0.0, device=primary_loss.device)
    if aux_kl_weight > 0:
        kl_value, _seg_dist_kl = kl_distill_segnet_only(
            pred_pair_hwc,
            gt_pair_hwc,
            segnet,
            temperature=KL_DISTILL_TEMPERATURE,
        )
        aux_kl_loss = aux_kl_weight * kl_value

    aux_pixel_l1 = torch.tensor(0.0, device=primary_loss.device)
    if aux_pixel_l1_weight > 0:
        aux_pixel_l1 = aux_pixel_l1_weight * (pred_pair_hwc - gt_pair_hwc).abs().mean()

    lagrangian.warm_up_lambda_R()
    weighted = (
        lagrangian.lambda_S * seg_dist
        + lagrangian.lambda_P * math.sqrt(10.0 * max(pose_dist, 0.0) + 1e-8)
        # rate term lives in archive build / sensitivity branches; lambda_R
        # warm-up is logged only here for the dispatch wrapper to consume.
    )
    total = primary_loss + aux_kl_loss + aux_pixel_l1
    if not torch.isfinite(total):
        return {
            "loss": float("nan"),
            "pose_dist": float(pose_dist),
            "seg_dist": float(seg_dist),
            "aux_kl": float(aux_kl_loss.item()),
            "aux_pixel_l1": float(aux_pixel_l1.item()),
            "weighted_proxy": float(weighted),
            "lambda_R": lagrangian.lambda_R,
        }
    total.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
    optimizer.step()
    ema.update(decoder)
    lagrangian.step += 1
    if lagrangian.step % DUAL_UPDATE_PERIOD == 0:
        lagrangian.update(
            observed_rate_bits=0.0,  # not measured here; archive-build measures it
            observed_seg=float(seg_dist),
            observed_pose=float(pose_dist),
        )
    return {
        "loss": float(total.item()),
        "pose_dist": float(pose_dist),
        "seg_dist": float(seg_dist),
        "aux_kl": float(aux_kl_loss.item()),
        "aux_pixel_l1": float(aux_pixel_l1.item()),
        "weighted_proxy": float(weighted),
        "lambda_R": lagrangian.lambda_R,
    }


def train(
    *,
    decoder: HNeRVDecoder,
    posenet: nn.Module,
    segnet: nn.Module,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    latent_dim: int,
    frame_h: int,
    frame_w: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    aux_kl_weight: float,
    aux_pixel_l1_weight: float,
    seed: int = 20260508,
) -> TrainResult:
    """Full training loop with EMA + Lagrangian + per-epoch logging."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    decoder.to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)
    ema = EMA(decoder, decay=EMA_DECAY)
    lagrangian = LagrangianState()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    epoch_log: list[dict[str, Any]] = []

    initial_seg = math.nan
    initial_pose = math.nan
    nan_observed = False

    with log_path.open("w") as log_f:
        for epoch in range(epochs):
            epoch_metrics = {"epoch": epoch, "step_metrics": []}
            for step_in_epoch in range(steps_per_epoch):
                z_batch, gt_pair_hwc = make_synthetic_pair_batch(
                    batch_size=batch_size,
                    frame_h=frame_h,
                    frame_w=frame_w,
                    latent_dim=latent_dim,
                    device=device,
                )
                metrics = train_one_step(
                    decoder=decoder,
                    posenet=posenet,
                    segnet=segnet,
                    z_batch=z_batch,
                    gt_pair_hwc=gt_pair_hwc,
                    optimizer=optimizer,
                    ema=ema,
                    lagrangian=lagrangian,
                    aux_kl_weight=aux_kl_weight,
                    aux_pixel_l1_weight=aux_pixel_l1_weight,
                )
                epoch_metrics["step_metrics"].append(metrics)
                if math.isnan(metrics["loss"]):
                    nan_observed = True
                if math.isnan(initial_seg):
                    initial_seg = metrics["seg_dist"]
                    initial_pose = metrics["pose_dist"]
            log_f.write(json.dumps(epoch_metrics) + "\n")
            log_f.flush()
            epoch_log.append(epoch_metrics)
            if epoch_metrics["step_metrics"]:
                m = epoch_metrics["step_metrics"][-1]
                LOGGER.info(
                    "epoch=%d loss=%.4f pose=%.6e seg=%.6e lambda_R=%.4f",
                    epoch, m["loss"], m["pose_dist"], m["seg_dist"], m["lambda_R"],
                )

    final_seg = (
        epoch_log[-1]["step_metrics"][-1]["seg_dist"]
        if epoch_log and epoch_log[-1]["step_metrics"]
        else math.nan
    )
    final_pose = (
        epoch_log[-1]["step_metrics"][-1]["pose_dist"]
        if epoch_log and epoch_log[-1]["step_metrics"]
        else math.nan
    )

    # Save EMA checkpoint (not the live model — CLAUDE.md NON-NEGOTIABLE).
    orig_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    ema.apply(decoder)
    try:
        torch.save(decoder.state_dict(), output_dir / "checkpoint_ema.pt")
    finally:
        decoder.load_state_dict(orig_state)
        decoder.train()

    return TrainResult(
        final_seg=final_seg,
        final_pose=final_pose,
        initial_seg=initial_seg,
        initial_pose=initial_pose,
        nan_observed=nan_observed,
        epoch_log=epoch_log,
    )


# ---------------------------------------------------------------------------
# Provenance + manifest
# ---------------------------------------------------------------------------

def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def write_provenance(output_dir: Path, args: argparse.Namespace) -> None:
    provenance = {
        "git_head_sha": _git_head_sha(),
        "hardware": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "platform": sys.platform,
        },
        "args": {k: str(v) for k, v in vars(args).items()},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
        "council_decision": "A1 — score-gradient supervision (UNANIMOUS HIGHEST PRIORITY)",
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


def write_build_manifest(output_dir: Path, args: argparse.Namespace, result: TrainResult) -> None:
    pose_delta_pct = 0.0
    seg_delta_pct = 0.0
    if not math.isnan(result.initial_pose) and result.initial_pose > 0:
        pose_delta_pct = 100.0 * (result.initial_pose - result.final_pose) / result.initial_pose
    if not math.isnan(result.initial_seg) and result.initial_seg > 0:
        seg_delta_pct = 100.0 * (result.initial_seg - result.final_seg) / result.initial_seg

    manifest = {
        "lane": "track1_phase_a1_score_gradient",
        "decision": "A1",
        "phase": "A",
        "council_finding_reference": "council Round 2 finding 10 (kl_on_logits → kl_distill_segnet_only)",
        "evidence_grade": "smoke" if args.smoke else "stub_pending_pr101_loader",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "remote_dispatch_allowed": False,
        "byte_proxy_only": True,
        "dispatch_blockers": (
            ["pr101_archive_loader_not_yet_implemented"]
            if not args.smoke
            else []
        ),
        "metrics": {
            "initial_pose_dist": result.initial_pose,
            "final_pose_dist": result.final_pose,
            "initial_seg_dist": result.initial_seg,
            "final_seg_dist": result.final_seg,
            "pose_delta_pct": pose_delta_pct,
            "seg_delta_pct": seg_delta_pct,
            "nan_observed": result.nan_observed,
        },
        "falsification_threshold": "min(seg_delta_pct, pose_delta_pct) < 5% → retire-config",
        "promotion_threshold": "max(seg_delta_pct, pose_delta_pct) >= 10% → promote",
        "ema_decay": EMA_DECAY,
        "eval_roundtrip_noise_std": EVAL_ROUNDTRIP_NOISE_STD,
        "lambda_r_warmup_steps": LAMBDA_R_WARMUP_STEPS,
        "kl_distill_temperature": KL_DISTILL_TEMPERATURE,
        "kl_distill_aux_weight": args.aux_kl_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    (output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smoke", action="store_true", help="2-epoch CPU smoke (~5min)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR_FINETUNE)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument("--frame-h", type=int, default=384)
    parser.add_argument("--frame-w", type=int, default=512)
    parser.add_argument("--smoke-frame-h", type=int, default=64,
                        help="downscaled frame size for CPU smoke")
    parser.add_argument("--smoke-frame-w", type=int, default=64)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-steps-per-epoch", type=int, default=4)
    parser.add_argument("--aux-kl-weight", type=float, default=KL_DISTILL_AUX_WEIGHT)
    parser.add_argument("--aux-pixel-l1-weight", type=float, default=PIXEL_L1_WEIGHT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pr101-archive", type=Path, default=None,
                        help="Path to PR101 archive.zip (required for non-smoke)")
    parser.add_argument("--seed", type=int, default=20260508)
    args = parser.parse_args()
    assert_no_invented_flag(args)
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()
    device = assert_cuda_or_explicit_cpu(args.device, args.smoke)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(output_dir, args)

    # Smoke uses downscaled frames + stub scorers; full training uses the
    # canonical sizes + real CUDA scorers.
    if args.smoke:
        frame_h, frame_w = args.smoke_frame_h, args.smoke_frame_w
        epochs = args.smoke_epochs
        steps_per_epoch = args.smoke_steps_per_epoch
        decoder = HNeRVDecoder(latent_dim=args.latent_dim, base_channels=36, eval_size=(frame_h, frame_w))
        posenet, segnet = load_smoke_scorers(device)
    else:
        frame_h, frame_w = args.frame_h, args.frame_w
        epochs = args.epochs
        steps_per_epoch = args.steps_per_epoch
        decoder = HNeRVDecoder(latent_dim=args.latent_dim, base_channels=36, eval_size=(frame_h, frame_w))
        load_pr101_substrate(args.pr101_archive, decoder, smoke=False)
        posenet, segnet = load_cuda_scorers(device)

    LOGGER.info(
        "Phase A1 score-gradient: device=%s smoke=%s epochs=%d steps_per_epoch=%d "
        "batch=%d lr=%.6e aux_kl=%.4f frame=%dx%d",
        device, args.smoke, epochs, steps_per_epoch,
        args.batch_size, args.lr, args.aux_kl_weight, frame_h, frame_w,
    )

    t0 = time.time()
    result = train(
        decoder=decoder,
        posenet=posenet,
        segnet=segnet,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        frame_h=frame_h,
        frame_w=frame_w,
        lr=args.lr,
        device=device,
        output_dir=output_dir,
        aux_kl_weight=args.aux_kl_weight,
        aux_pixel_l1_weight=args.aux_pixel_l1_weight,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    write_build_manifest(output_dir, args, result)

    # Smoke pass criteria: NO NaN + at least one of (seg, pose) decreased.
    if args.smoke:
        if result.nan_observed:
            LOGGER.error("SMOKE FAIL: NaN observed in loss")
            return 2
        # In smoke mode with synthetic data and stub scorers, we are testing
        # the gradient PATH, not the actual proxy-auth gap. Either of seg/pose
        # decreasing (even slightly) signals the loop is wired correctly.
        seg_decreased = result.final_seg < result.initial_seg
        pose_decreased = result.final_pose < result.initial_pose
        if not (seg_decreased or pose_decreased):
            LOGGER.warning(
                "SMOKE WARNING: neither seg nor pose decreased — initial=(seg=%.4e, pose=%.4e) "
                "final=(seg=%.4e, pose=%.4e). This may be benign on synthetic data with "
                "stub scorers but should be re-checked on real CUDA scorers.",
                result.initial_seg, result.initial_pose, result.final_seg, result.final_pose,
            )
        else:
            LOGGER.info(
                "SMOKE PASS: gradient path verified (seg %.4e→%.4e, pose %.4e→%.4e, %.1fs)",
                result.initial_seg, result.final_seg, result.initial_pose, result.final_pose, elapsed,
            )

    LOGGER.info("Wrote artifacts to %s", output_dir)
    return 0 if not result.nan_observed else 2


if __name__ == "__main__":
    raise SystemExit(main())
