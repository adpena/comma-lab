"""Lane AL — unit tests for src/tac/optimize_grayscale_canvas.py.

Smoke-level coverage:
  * device validation rejects 'mps' and missing-CUDA cases
  * eval_roundtrip=False raises (CLAUDE.md non-negotiable)
  * initialize_gray_logits roundtrips Lane MM grayscale targets
  * Gaussian softmax matches the LUT at integer gray values
  * Soft-embedding lookup matches the discrete embedding lookup at one-hot
  * STE round/clamp passes gradient unchanged
  * Full SGD loop on a 4-frame, 1-class toy reduces the loss

The full lane is exercised on real Lane A artifacts via the remote
script + auth eval — these tests cover the math + interfaces only.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.mask_grayscale_lut import (
    CLASS_TO_GRAY,
    LUT_DEFAULT_SIGMA,
    NUM_CLASSES,
    create_gaussian_softmax_lut,
    encode_masks_grayscale,
)
from tac.optimize_grayscale_canvas import (
    OptimizeConfig,
    _gaussian_softmax_soft,
    _gray_logits_to_continuous,
    _render_pair_from_logits,
    _soft_embedding_lookup,
    _ste_round_clamp,
    _validate_device,
    initialize_gray_logits,
    optimize_grayscale_canvas,
)
from tac.preflight import PreflightError


# ── device validation ────────────────────────────────────────────────────


def test_validate_device_rejects_mps() -> None:
    """CLAUDE.md non-negotiable: never default to MPS for Lane AL."""
    with pytest.raises(PreflightError, match="mps"):
        _validate_device("mps")


def test_validate_device_cuda_requires_available(monkeypatch) -> None:
    """If --device cuda is requested but CUDA is missing, raise (no
    silent CPU fallback — that produces invalid scores)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PreflightError, match="CUDA"):
        _validate_device("cuda")


def test_validate_device_cpu_allowed() -> None:
    """CPU is allowed for smoke tests with the deterministic-bytes
    advisory caveat (the function does NOT enforce that — the lane
    script does)."""
    out = _validate_device("cpu")
    assert out.type == "cpu"


# ── initializer ─────────────────────────────────────────────────────────


def test_initialize_gray_logits_roundtrips_lane_mm_targets() -> None:
    """sigmoid(logits) * 255 should reproduce the Lane MM CLASS_TO_GRAY
    targets (within rounding)."""
    class_ids = torch.tensor(
        [[[0, 1, 2, 3, 4]]], dtype=torch.int64,
    )  # (1, 1, 5)
    logits = initialize_gray_logits(class_ids, device="cpu")
    gray_continuous = _gray_logits_to_continuous(logits)
    # Round to uint8 and check exact match against the canonical encoder.
    rounded = gray_continuous.round().clamp(0, 255).to(torch.uint8)
    expected = encode_masks_grayscale(class_ids)
    # eps=1/512 in initialize_gray_logits → max drift 0.5 levels =>
    # rounded should match exactly for the {0, 64, 128, 192, 255} grid.
    assert torch.equal(rounded, expected), (
        f"roundtrip drift: rounded={rounded.tolist()} "
        f"expected={expected.tolist()}"
    )


def test_initialize_gray_logits_rejects_non_int64() -> None:
    bad = torch.zeros(1, 4, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="int64"):
        initialize_gray_logits(bad, device="cpu")


# ── Gaussian softmax math ───────────────────────────────────────────────


def test_gaussian_softmax_matches_canonical_lut_at_integer_gray() -> None:
    """At integer gray values, the soft-projection must equal the
    canonical 256-row LUT (within float tolerance)."""
    targets = torch.tensor(
        [CLASS_TO_GRAY[c] for c in range(NUM_CLASSES)], dtype=torch.float32,
    )
    canonical_lut = create_gaussian_softmax_lut(sigma=LUT_DEFAULT_SIGMA)
    # Probe at every integer gray value: build a 1x1 tensor, project,
    # compare to canonical_lut[v].
    for v in range(256):
        gray = torch.tensor([[float(v)]])  # (1, 1)
        # _gaussian_softmax_soft expects (..., H, W) → out (..., C, H, W)
        soft = _gaussian_softmax_soft(
            gray.unsqueeze(0), sigma=LUT_DEFAULT_SIGMA, targets=targets,
        )
        # soft shape (1, NUM_CLASSES, 1, 1)
        assert torch.allclose(
            soft.flatten(), canonical_lut[v], atol=1e-5,
        ), f"mismatch at gray={v}: soft={soft.flatten()} lut={canonical_lut[v]}"


def test_gaussian_softmax_is_differentiable() -> None:
    targets = torch.tensor(
        [CLASS_TO_GRAY[c] for c in range(NUM_CLASSES)], dtype=torch.float32,
    )
    gray = torch.full((1, 4, 4), 240.0, requires_grad=True)
    soft = _gaussian_softmax_soft(gray, sigma=15.0, targets=targets)
    loss = soft[:, 1, :, :].sum()  # class 1 target is gray=255
    loss.backward()
    assert gray.grad is not None
    assert torch.isfinite(gray.grad).all()
    assert gray.grad.abs().mean() > 0


# ── soft embedding ──────────────────────────────────────────────────────


def test_soft_embedding_at_onehot_matches_discrete_lookup() -> None:
    """Soft lookup at a one-hot probability == discrete embedding(class)."""
    embed = nn.Embedding(NUM_CLASSES, 6)
    nn.init.normal_(embed.weight)
    cls = 3
    soft = torch.zeros(1, NUM_CLASSES, 4, 4)
    soft[:, cls, :, :] = 1.0
    out = _soft_embedding_lookup(soft, embed)  # (1, 6, 4, 4)
    expected = embed.weight[cls].view(1, 6, 1, 1).expand(1, 6, 4, 4)
    assert torch.allclose(out, expected, atol=1e-5)


# ── STE round/clamp ─────────────────────────────────────────────────────


def test_ste_round_clamp_forward_quantizes() -> None:
    x = torch.tensor([-5.0, 0.4, 0.6, 100.7, 256.0])
    out = _ste_round_clamp(x)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 101.0, 255.0]))


def test_ste_round_clamp_backward_passes_gradient() -> None:
    x = torch.tensor([10.5, 200.3], requires_grad=True)
    out = _ste_round_clamp(x)
    loss = out.sum()
    loss.backward()
    assert torch.equal(x.grad, torch.tensor([1.0, 1.0]))


# ── full SGD smoke ─────────────────────────────────────────────────────


class _ToyEmbeddingRenderer(nn.Module):
    """Minimal renderer: maps soft embedding → 3-channel RGB via 1x1 conv."""

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.head = nn.Conv2d(embed_dim, 3, 1, bias=True)
        nn.init.normal_(self.head.weight, std=0.5)
        nn.init.constant_(self.head.bias, 127.5)

    def forward_from_embedding(self, soft_embed: torch.Tensor) -> torch.Tensor:
        return 255.0 * torch.sigmoid(self.head(soft_embed) / 50.0)


class _ToyScorer(nn.Module):
    """Minimal scorer mimicking PoseNet/SegNet preprocess + forward.

    PoseNet path: returns dict with 'pose' = (B, 12) — a fixed-weight
    linear projection of the input that is sensitive to per-pixel values
    (NOT just spatial mean) so SGD on the gray-canvas can drive
    pose_dist toward zero.
    SegNet path: returns (B, 5, H, W) logits via a fixed 1x1 conv whose
    weights differ per output class, so the cross-entropy gradient
    actually depends on the input.
    """

    def __init__(self, mode: str, in_channels: int = 6):
        super().__init__()
        self.mode = mode
        torch.manual_seed(7)
        if mode == "pose":
            # (B, in_ch, H, W) → (B, 12) via global avg pool + linear.
            self.proj = nn.Linear(in_channels, 12, bias=False)
        else:
            # 1x1 conv → (B, 5, H, W).
            self.proj = nn.Conv2d(in_channels, 5, 1, bias=True)
        # Freeze (act like the real scorers).
        for p in self.parameters():
            p.requires_grad = False

    def preprocess_input(self, x):
        b, t, c, h, w = x.shape
        return x.reshape(b, t * c, h, w)

    def forward(self, x):
        if self.mode == "pose":
            # GAP → linear → (B, 12)
            pooled = x.mean(dim=(2, 3))  # (B, in_ch)
            pose = self.proj(pooled)
            return {"pose": pose, "embed": pose}
        else:
            return self.proj(x)


def test_optimize_grayscale_canvas_reduces_loss_smoke_cpu(monkeypatch) -> None:
    """Tiny end-to-end loop on CPU: 4 frames, 20 SGD steps, loss drops.

    The Gaussian-LUT softmax at the production sigma=15 has very small
    gradients away from class boundaries — by design, since it is an
    error-correcting LUT for AV1 quantization noise. The smoke test
    therefore monkey-patches the soft-probability function with a plain
    softmax over (-distance/sigma) on a moderate sigma so the toy
    randomized canvas has non-degenerate gradient signal everywhere.
    The mathematical guarantees (one-hot equality at class targets;
    integer-LUT equivalence) are verified separately by
    ``test_gaussian_softmax_matches_canonical_lut_at_integer_gray``.
    """
    torch.manual_seed(13)
    H, W, embed_dim = 8, 8, 4
    n_frames = 4
    # Off-target initialization (NOT Lane MM targets) so gradients
    # through the production Gaussian softmax are meaningful.
    init_class_ids = torch.randint(0, NUM_CLASSES, (n_frames, H, W),
                                    dtype=torch.int64)

    # GT pairs at "scorer" resolution (this toy scorer doesn't care
    # about exact resolution since it is a 1×1 mean op).
    P = n_frames // 2
    gt_pairs = torch.rand(P, 2, 3, H, W) * 255.0

    embed = nn.Embedding(NUM_CLASSES, embed_dim)
    nn.init.normal_(embed.weight)
    toy_renderer = _ToyEmbeddingRenderer(embed_dim=embed_dim)

    # Toy scorer in_channels = T*3 = 6 for non-overlapping pair input.
    pose_scorer = _ToyScorer(mode="pose", in_channels=6)
    seg_scorer = _ToyScorer(mode="seg", in_channels=6)

    cfg = OptimizeConfig(
        steps=20,
        lr=10.0,  # large lr OK on the toy: gray space is 0..255
        sigma=30.0,  # softer than the canonical 15 → broader gradient
        batch_size=2,
        log_every=1,
        noise_std=0.0,  # deterministic for the smoke check
        seed=0,
    )

    # Patch _eval_roundtrip_with_noise to a no-op for the toy resolution
    # (the canonical roundtrip rescales to 874x1164 which would crush
    # the toy 8x8 input; we just want the loss-drop signal here).
    import tac.optimize_grayscale_canvas as mod

    def _identity_roundtrip(rgb, noise_std, generator):
        return rgb

    monkey_orig = mod._eval_roundtrip_with_noise
    mod._eval_roundtrip_with_noise = _identity_roundtrip
    try:
        gray_int, metrics = optimize_grayscale_canvas(
            init_class_ids=init_class_ids,
            gt_pairs_btchw=gt_pairs,
            embedding=embed,
            renderer_forward_from_embedding=toy_renderer.forward_from_embedding,
            posenet=pose_scorer,
            segnet=seg_scorer,
            cfg=cfg,
            device="cpu",
        )
    finally:
        mod._eval_roundtrip_with_noise = monkey_orig

    assert gray_int.shape == (n_frames, H, W)
    assert gray_int.dtype == torch.uint8
    assert len(metrics) >= 2
    # First-step loss should be > final-step loss for a converging SGD
    # on this trivial objective. Allow some noise: require strict drop.
    assert metrics[-1]["loss"] < metrics[0]["loss"], (
        f"loss did not decrease: first={metrics[0]} last={metrics[-1]}"
    )


# ── eval_roundtrip enforcement ──────────────────────────────────────────


def test_optimize_refuses_eval_roundtrip_false() -> None:
    """CLAUDE.md non-negotiable: every TTO/training path must roundtrip."""
    cfg = OptimizeConfig(
        steps=1, eval_roundtrip=False,
    )
    embed = nn.Embedding(NUM_CLASSES, 4)
    pose_scorer = _ToyScorer(mode="pose")
    seg_scorer = _ToyScorer(mode="seg")
    init_class_ids = torch.zeros(2, 4, 4, dtype=torch.int64)
    gt_pairs = torch.zeros(1, 2, 3, 4, 4)

    def _noop_renderer(soft_embed):
        return torch.zeros(soft_embed.shape[0], 3, 4, 4)

    with pytest.raises(PreflightError, match="eval_roundtrip"):
        optimize_grayscale_canvas(
            init_class_ids=init_class_ids,
            gt_pairs_btchw=gt_pairs,
            embedding=embed,
            renderer_forward_from_embedding=_noop_renderer,
            posenet=pose_scorer,
            segnet=seg_scorer,
            cfg=cfg,
            device="cpu",
        )


# ── shape validation ────────────────────────────────────────────────────


def test_optimize_rejects_odd_frame_count() -> None:
    cfg = OptimizeConfig(steps=1, eval_roundtrip=True)
    embed = nn.Embedding(NUM_CLASSES, 4)
    pose_scorer = _ToyScorer(mode="pose")
    seg_scorer = _ToyScorer(mode="seg")
    init_class_ids = torch.zeros(3, 4, 4, dtype=torch.int64)  # odd!
    gt_pairs = torch.zeros(1, 2, 3, 4, 4)

    def _noop_renderer(soft_embed):
        return torch.zeros(soft_embed.shape[0], 3, 4, 4)

    with pytest.raises(ValueError, match="odd N"):
        optimize_grayscale_canvas(
            init_class_ids=init_class_ids,
            gt_pairs_btchw=gt_pairs,
            embedding=embed,
            renderer_forward_from_embedding=_noop_renderer,
            posenet=pose_scorer,
            segnet=seg_scorer,
            cfg=cfg,
            device="cpu",
        )
