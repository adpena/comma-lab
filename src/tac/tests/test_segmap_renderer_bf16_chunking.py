"""Tests for Council C OOM-class deep fixes (DF2 bf16 + DF3 scorer chunking).

Memory: .omx/research/council_oom_class_deep_fix_20260429.md.

The 5 tests mandated by the work-scope blueprint:

  (i)   bf16 path runs forward+backward without dtype error on CUDA
        (SKIPPED with marker when CUDA is not available locally — bf16
        is structurally a CUDA-only feature in our trainer per CLAUDE.md
        FORBIDDEN PATTERNS).
  (ii)  scorer_chunk=2 with mb=4 produces output identical (within fp
        tolerance) to scorer_chunk=0 — proves chunking preserves
        autograd correctness.
  (iii) scorer_chunk=1 with mb=8 produces output identical to
        scorer_chunk=0 — same equivalence at the per-pair extreme.
  (iv)  Check 87 fires when a SegMap-class lane script lacks the
        OOM-guard flags.
  (v)   Check 87 passes when a SegMap-class lane script has all flags
        set with B*N<=8.

Structural correctness only — score validation requires Vast.ai 4090
dispatch (parent agent will handle).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.preflight import (
    PreflightError,
    check_segmap_class_lanes_have_oom_guards,
)
from tac.segmap_renderer import (
    SEGMAP_INPUT_SIZE,
    SegMap,
    SegMapTrainer,
)
from tac.training import EMA, TrainConfig


# ─── Mock scorers (mirror tests/test_segmap_renderer.py fixtures) ──────


class _MockPoseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(12, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8, 12)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = F.interpolate(x, size=(96, 128), mode="bilinear", align_corners=False)
        x = x.reshape(b, t * c, 96, 128)
        if x.shape[1] < 12:
            pad = torch.zeros(
                b, 12 - x.shape[1], 96, 128, device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=1)
        return x[:, :12]

    def forward(self, x: torch.Tensor) -> dict:
        h = self.conv(x).mean(dim=(2, 3))
        return {"pose": self.fc(h)}


class _MockSegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, ...]
        return F.interpolate(last, size=(48, 64), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _make_tiny_segmap(num_blocks: int = 2, max_frame_index: int = 32) -> SegMap:
    """Small but architecturally faithful SegMap for unit tests."""
    return SegMap(
        hidden=8,
        block_hidden=8,
        num_blocks=num_blocks,
        max_frame_index=max_frame_index,
    )


def _make_cfg(*, bf16: bool = False, scorer_chunk: int = 0) -> TrainConfig:
    return TrainConfig(
        hidden=8,
        epochs=100,
        warmup_epochs=10,
        tag="test-segmap-bf16",
        lr=1e-3,
        eval_roundtrip=True,
        loss_mode="standard",
        bf16=bf16,
        scorer_chunk=scorer_chunk,
    )


def _make_pairs(b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mask_pairs, gt_pairs) at 1/16 SEGMAP_INPUT_SIZE."""
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16
    t = 2
    masks = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt = torch.rand(b, t, h, w, 3) * 255.0
    return masks, gt


# ─── Test (i): bf16 path forward+backward runs cleanly on CUDA ────────


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bf16 autocast is CUDA-only by design (CLAUDE.md FORBIDDEN "
    "PATTERN: never silently fall back to MPS/CPU). Test requires CUDA.",
)
def test_bf16_forward_backward_no_dtype_error_on_cuda() -> None:
    """DF2: bf16 autocast wraps the forward + scorer call without dtype error.

    PoseNet+SegNet are FROZEN (requires_grad_(False) on every param), so the
    backward pass only flows gradients through the rendered branch — bf16
    casts cleanly without GradScaler. The test asserts:
      1) train_epoch returns finite loss with bf16=True;
      2) at least one renderer parameter actually moves (grad pressure).
    """
    torch.manual_seed(0)
    cfg = _make_cfg(bf16=True, scorer_chunk=0)
    model = _make_tiny_segmap().to("cuda")
    posenet = _MockPoseNet().to("cuda")
    segnet = _MockSegNet().to("cuda")
    for net in (posenet, segnet):
        for p in net.parameters():
            p.requires_grad_(False)
    trainer = SegMapTrainer(model, cfg, posenet, segnet, device="cuda")
    assert trainer.use_bf16 is True

    masks, gt = _make_pairs(b=2)
    masks = masks.to("cuda")
    gt = gt.to("cuda")
    pre = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    stats = trainer.train_epoch(masks, gt, ema=None, batch_size=2)
    post = next(p for p in model.parameters() if p.requires_grad).detach()

    assert math.isfinite(stats["loss"]), f"loss not finite under bf16: {stats}"
    assert stats["num_steps"] == 1
    assert not torch.equal(pre, post), "bf16 path produced no parameter update"


def test_bf16_requires_cuda_raises_on_cpu() -> None:
    """CLAUDE.md FORBIDDEN PATTERN: bf16=True with non-CUDA device is fatal,
    never a silent fallback to MPS/CPU.

    This test runs on CPU (always available) and confirms the trainer
    raises ``PreflightError`` rather than silently downgrading to fp32 on
    a non-CUDA device when ``bf16=True`` is requested.
    """
    cfg = _make_cfg(bf16=True, scorer_chunk=0)
    model = _make_tiny_segmap()
    with pytest.raises(PreflightError, match=r"bf16=True requires device='cuda'"):
        SegMapTrainer(model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu")


# ─── Test (ii): scorer_chunk=2 with mb=4 == scorer_chunk=0 ────────────


def test_scorer_chunk_2_mb_4_matches_unchunked() -> None:
    """DF3 correctness: chunking the scorer call along batch dim must produce
    the same loss / gradients as a single un-chunked call (within fp tol).

    Run two trainers from the same seed, same model state — one with
    scorer_chunk=0 (legacy unchunked) and one with scorer_chunk=2 (chunk
    of 2 pairs at a time). Both should produce identical loss within the
    floating-point tolerance from non-associative reduction order.
    """
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16
    b, t = 4, 2
    torch.manual_seed(123)
    masks_seed = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt_seed = torch.rand(b, t, h, w, 3) * 255.0

    # Path A: legacy scorer_chunk=0 (unchunked).
    torch.manual_seed(7)
    model_a = _make_tiny_segmap()
    cfg_a = _make_cfg(bf16=False, scorer_chunk=0)
    posenet_a = _MockPoseNet()
    segnet_a = _MockSegNet()
    for net in (posenet_a, segnet_a):
        for p in net.parameters():
            p.requires_grad_(False)
    trainer_a = SegMapTrainer(model_a, cfg_a, posenet_a, segnet_a, device="cpu")

    # Path B: chunked scorer_chunk=2 (matched seeds → identical init).
    torch.manual_seed(7)
    model_b = _make_tiny_segmap()
    cfg_b = _make_cfg(bf16=False, scorer_chunk=2)
    posenet_b = _MockPoseNet()
    segnet_b = _MockSegNet()
    for net in (posenet_b, segnet_b):
        for p in net.parameters():
            p.requires_grad_(False)
    # Mirror the scorer weights exactly so the two paths see identical
    # frozen scorer outputs (otherwise the random scorer init diverges).
    posenet_b.load_state_dict(posenet_a.state_dict())
    segnet_b.load_state_dict(segnet_a.state_dict())
    trainer_b = SegMapTrainer(model_b, cfg_b, posenet_b, segnet_b, device="cpu")

    # roundtrip_noise_std=0.0 makes the test deterministic (no Gaussian
    # noise dependent on torch.randn-call-order between the two paths).
    stats_a = trainer_a.train_epoch(
        masks_seed.clone(), gt_seed.clone(), ema=None,
        roundtrip_noise_std=0.0, batch_size=4,
    )
    stats_b = trainer_b.train_epoch(
        masks_seed.clone(), gt_seed.clone(), ema=None,
        roundtrip_noise_std=0.0, batch_size=4,
    )

    # Loss / pose / seg must match within fp tolerance (cat over chunked
    # outputs is mathematically identical; tiny drift can still arise from
    # cross-entropy reduction order differences. Use 1e-5 tolerance).
    assert math.isclose(stats_a["loss"], stats_b["loss"], rel_tol=1e-5, abs_tol=1e-5), (
        f"loss differs: chunked={stats_b['loss']} vs unchunked={stats_a['loss']}"
    )
    assert math.isclose(stats_a["pose_dist"], stats_b["pose_dist"], rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(stats_a["seg_dist"], stats_b["seg_dist"], rel_tol=1e-5, abs_tol=1e-5)


# ─── Test (iii): scorer_chunk=1 with mb=8 == scorer_chunk=0 ───────────


def test_scorer_chunk_1_mb_8_matches_unchunked() -> None:
    """Per-pair extreme chunking (chunk=1) must also match the unchunked path."""
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16
    b, t = 8, 2
    torch.manual_seed(456)
    masks_seed = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt_seed = torch.rand(b, t, h, w, 3) * 255.0

    torch.manual_seed(11)
    model_a = _make_tiny_segmap()
    cfg_a = _make_cfg(bf16=False, scorer_chunk=0)
    posenet_a = _MockPoseNet()
    segnet_a = _MockSegNet()
    for net in (posenet_a, segnet_a):
        for p in net.parameters():
            p.requires_grad_(False)
    trainer_a = SegMapTrainer(model_a, cfg_a, posenet_a, segnet_a, device="cpu")

    torch.manual_seed(11)
    model_b = _make_tiny_segmap()
    cfg_b = _make_cfg(bf16=False, scorer_chunk=1)
    posenet_b = _MockPoseNet()
    segnet_b = _MockSegNet()
    for net in (posenet_b, segnet_b):
        for p in net.parameters():
            p.requires_grad_(False)
    posenet_b.load_state_dict(posenet_a.state_dict())
    segnet_b.load_state_dict(segnet_a.state_dict())
    trainer_b = SegMapTrainer(model_b, cfg_b, posenet_b, segnet_b, device="cpu")

    stats_a = trainer_a.train_epoch(
        masks_seed.clone(), gt_seed.clone(), ema=None,
        roundtrip_noise_std=0.0, batch_size=8,
    )
    stats_b = trainer_b.train_epoch(
        masks_seed.clone(), gt_seed.clone(), ema=None,
        roundtrip_noise_std=0.0, batch_size=8,
    )

    assert math.isclose(stats_a["loss"], stats_b["loss"], rel_tol=1e-5, abs_tol=1e-5), (
        f"loss differs: chunk=1 {stats_b['loss']} vs unchunked {stats_a['loss']}"
    )
    assert math.isclose(stats_a["pose_dist"], stats_b["pose_dist"], rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(stats_a["seg_dist"], stats_b["seg_dist"], rel_tol=1e-5, abs_tol=1e-5)


# ─── Test (iv): Check 87 fires on a missing-flags lane script ─────────


def test_check_87_fires_when_flags_missing(tmp_path: Path) -> None:
    """A synthetic lane script that invokes train_segmap.py without the
    OOM-guard flags must trigger Check 87 with strict=True."""
    repo_root = tmp_path
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)

    bad_script = scripts_dir / "remote_lane_bad_synthetic.sh"
    bad_script.write_text(
        '#!/bin/bash\n'
        'set -euo pipefail\n'
        '"$PYBIN" -u experiments/train_segmap.py \\\n'
        '    --variant kl_distill \\\n'
        '    --hidden 24 --block-hidden 24 --num-blocks 8 \\\n'
        '    --epochs 600 --batch-size 8 --lr 1e-3 \\\n'
        '    --device cuda \\\n'
        '    --tag bad --output-dir /tmp/bad\n'
    )

    violations = check_segmap_class_lanes_have_oom_guards(
        repo_root=repo_root,
        shell_files=["scripts/remote_lane_bad_synthetic.sh"],
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "OOM-class deep fixes" in violations[0]
    assert "bf16=False" in violations[0]
    assert "scorer-chunk=None" in violations[0]

    # strict=True must raise.
    with pytest.raises(PreflightError, match=r"SEGMAP OOM GUARD"):
        check_segmap_class_lanes_have_oom_guards(
            repo_root=repo_root,
            shell_files=["scripts/remote_lane_bad_synthetic.sh"],
            strict=True,
            verbose=False,
        )


def test_check_87_fires_when_bn_product_exceeds_cap(tmp_path: Path) -> None:
    """A script with all 3 flags set but B*N > 8 must still violate."""
    repo_root = tmp_path
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)

    over_cap_script = scripts_dir / "remote_lane_over_cap.sh"
    # B=8, N=2 → B*N=16 (over the cap of 8).
    over_cap_script.write_text(
        '#!/bin/bash\n'
        'set -euo pipefail\n'
        '"$PYBIN" -u experiments/train_segmap.py \\\n'
        '    --variant kl_distill \\\n'
        '    --hidden 24 --block-hidden 24 --num-blocks 8 \\\n'
        '    --epochs 600 --batch-size 8 --lr 1e-3 \\\n'
        '    --bf16 --scorer-chunk 2 \\\n'
        '    --device cuda \\\n'
        '    --tag over --output-dir /tmp/over\n'
    )

    violations = check_segmap_class_lanes_have_oom_guards(
        repo_root=repo_root,
        shell_files=["scripts/remote_lane_over_cap.sh"],
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1, f"expected 1 violation, got {violations}"
    assert "B*N<=8" in violations[0]
    assert "batch-size=8" in violations[0]


# ─── Test (v): Check 87 passes when B*N <= 8 ──────────────────────────


def test_check_87_passes_when_flags_set_with_bn_product_le_8(tmp_path: Path) -> None:
    """A correctly-armed lane script (--bf16 + --scorer-chunk N + --batch-size B
    with B*N<=8) must yield zero Check 87 violations."""
    repo_root = tmp_path
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)

    # B=4, N=2 → B*N=8 (exactly at cap).
    good_script = scripts_dir / "remote_lane_good.sh"
    good_script.write_text(
        '#!/bin/bash\n'
        'set -euo pipefail\n'
        '"$PYBIN" -u experiments/train_segmap.py \\\n'
        '    --variant kl_distill \\\n'
        '    --kl-distill-weight 0.002 \\\n'
        '    --kl-distill-temperature 2.0 \\\n'
        '    --hidden 24 --block-hidden 24 --num-blocks 8 \\\n'
        '    --epochs 600 --batch-size 4 --lr 1e-3 \\\n'
        '    --bf16 --scorer-chunk 2 \\\n'
        '    --device cuda \\\n'
        '    --tag good --output-dir /tmp/good\n'
    )

    violations = check_segmap_class_lanes_have_oom_guards(
        repo_root=repo_root,
        shell_files=["scripts/remote_lane_good.sh"],
        strict=True,
        verbose=False,
    )
    assert violations == [], f"expected 0 violations, got {violations}"

    # Path B opt-out: --gradient-checkpointing + GPU_TIER_HINT=A100.
    a100_script = scripts_dir / "remote_lane_a100_path_b.sh"
    a100_script.write_text(
        '#!/bin/bash\n'
        'set -euo pipefail\n'
        'export GPU_TIER_HINT=A100\n'
        '"$PYBIN" -u experiments/train_segmap.py \\\n'
        '    --variant kl_distill \\\n'
        '    --hidden 24 --block-hidden 24 --num-blocks 8 \\\n'
        '    --epochs 600 --batch-size 8 --lr 1e-3 \\\n'
        '    --gradient-checkpointing \\\n'
        '    --device cuda \\\n'
        '    --tag a100 --output-dir /tmp/a100\n'
    )
    violations = check_segmap_class_lanes_have_oom_guards(
        repo_root=repo_root,
        shell_files=["scripts/remote_lane_a100_path_b.sh"],
        strict=True,
        verbose=False,
    )
    assert violations == [], f"expected 0 Path-B violations, got {violations}"
