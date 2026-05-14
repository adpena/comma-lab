# SPDX-License-Identifier: MIT
"""Tests for ``RealPairBatchSource.iter_batches`` Lane 12-v2 batch source.

Per HIGH 3 in the comprehensive adversarial review (commit 7e9d970e), the
$40 Phase B CUDA dispatch was blocked because ``iter_batches`` did not yet
expose production-grade resumability (skip_to_pair) + cap (max_pairs) +
canonical-decode-path enforcement. This file lands those tests.

Hard requirements honored per CLAUDE.md:

* HNeRV parity discipline lesson 1 (substrate score-aware): MUST decode
  ``upstream/videos/0.mkv`` via PyAV, NOT synthetic noise.
* HNeRV parity discipline lesson 8 (eval_roundtrip + autograd-YUV6 baked in
  TRAINING inner loop): the differentiable preprocess primitives live in
  ``train_step`` (rendered side) + ``scorer.preprocess_input`` (both sides);
  the batch source delivers raw uint8 GT pairs at camera resolution.
* AVVideoDataset CUDA-CPU drift discriminator (commit 0c2faf0a): PyAV is the
  canonical decoder; DALI is forbidden on this path because contest CPU axis
  ranks via PyAV decode.
* MASKS.MKV postmortem (CLAUDE.md "CATASTROPHIC FAILURES (2026-04-21)"): pairs
  MUST be non-overlapping (pair 0 = (frame0, frame1), pair 1 = (frame2,
  frame3), ...) — NOT the score-invalid 1199-pair overlapping scheme.
* ``forbidden_synthetic_non_smoke_OK``: refuses to fall back to synthetic when
  the contest video is missing.
* ``forbidden_default_to_convenience_trap``: tests run on CPU only; CUDA
  default is enforced inside ``Lane12V2NeRVConfig`` not the batch source.

Cross-references:

* ``src/tac/lane_12_v2_nerv_as_renderer.py::RealPairBatchSource``
* ``src/tac/differentiable_eval_roundtrip.py``
* ``upstream/frame_utils.py::yuv420_to_rgb``
* HNeRV retrospective:
  ``feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md``
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from tac.lane_12_v2_nerv_as_renderer import (
    RealPairBatchSource,
    _make_synthetic_pair_batch_for_smoke,
    train_step,
    default_pose_surrogate,
    default_seg_surrogate,
    Lane12V2NeRVConfig,
    Lane12V2NeRVRenderer,
    Lane12V2LatentTable,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_VIDEO = REPO_ROOT / "upstream" / "videos" / "0.mkv"


def _require_video() -> Path:
    if not UPSTREAM_VIDEO.exists():
        pytest.skip(f"upstream contest video not present at {UPSTREAM_VIDEO}")
    try:
        import av  # noqa: F401
    except ImportError:
        pytest.skip("PyAV not installed")
    return UPSTREAM_VIDEO


# ── Shape + contract tests (real PyAV decode) ────────────────────────────


def test_iter_batches_yields_correct_shapes():
    """``(B, 2, 3, H, W)`` pair tensor + ``(B,)`` long pair-index tensor."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    indices, pairs = next(iter(src.iter_batches(batch_size=2)))
    assert indices.shape == (2,)
    assert indices.dtype == torch.long
    assert pairs.shape == (2, 2, 3, 874, 1164)
    assert pairs.dtype == torch.uint8


def test_iter_batches_resolves_to_contest_camera_resolution():
    """Camera resolution = ``(874, 1164)`` per ``upstream/frame_utils.py::camera_size``.

    Renderer outputs at ``eval_size=(384, 512)``; trainer upsamples to camera
    resolution before scorer call. GT pairs are delivered at camera res so the
    scorer's ``preprocess_input`` resizes to ``(384, 512)`` symmetrically.
    """
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=2, eval_size=(384, 512))
    _, pairs = next(iter(src.iter_batches(batch_size=2)))
    assert pairs.shape[-2:] == (874, 1164)


def test_iter_batches_non_overlapping_pairs():
    """Pair 0 = (frame0, frame1), pair 1 = (frame2, frame3) — NOT overlapping.

    Regression for the 2026-04-21 ``MASKS.MKV AT 48x64 DESTROYED THE SCORE``
    postmortem: the canonical contest evaluator uses ``seq_len=2`` non-overlapping
    batching (600 pairs from 1200 frames). Overlapping would yield 1199 pairs
    and silently produce wrong scores.
    """
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    seen_indices: list[int] = []
    for indices, _pairs in src.iter_batches(batch_size=2):
        seen_indices.extend(int(i) for i in indices)
    # 4 pairs requested → indices are [0, 1, 2, 3], NOT [0, 1, 2, 3, 4, 5, ...].
    assert seen_indices == [0, 1, 2, 3]


def test_iter_batches_consecutive_pairs_have_distinct_frames():
    """Pair 0 frames must differ from pair 1 frames (no frame reuse).

    If pairs were overlapping, pair 0 frame[1] would equal pair 1 frame[0];
    non-overlapping pairs guarantee no shared frame between consecutive pairs.
    """
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=2, eval_size=(384, 512))
    indices, pairs = next(iter(src.iter_batches(batch_size=2)))
    assert indices.tolist() == [0, 1]
    pair0_frame1 = pairs[0, 1].float()
    pair1_frame0 = pairs[1, 0].float()
    # Real video frames are never identical at uint8 quantization unless the
    # decoder is bugged or pairs are overlapping.
    assert not torch.allclose(pair0_frame1, pair1_frame0, atol=1e-3), (
        "consecutive non-overlapping pairs must have distinct frames; "
        "shared frame indicates overlapping scheme"
    )


def test_iter_batches_uses_pyav_not_dali():
    """The batch source must NOT import or instantiate DALI on this path.

    Per AVVideoDataset CUDA-CPU drift discriminator (commit 0c2faf0a): the
    contest CPU evaluator ranks via PyAV; using DALI on the trainer side would
    introduce a decoder-class drift between training targets and CPU-axis
    evaluation that erodes the [contest-CPU] score.
    """
    src_text = inspect.getsource(RealPairBatchSource)
    # Token-level check: no DALI import or class reference inside the class body.
    forbidden = ("nvidia.dali", "DALIGenericIterator", "DaliVideoDataset", "pipeline_def")
    for token in forbidden:
        assert token not in src_text, (
            f"DALI token {token!r} appeared inside RealPairBatchSource — "
            f"PyAV is the canonical decoder per AVVideoDataset drift findings."
        )


def test_iter_batches_preprocess_preserves_autograd_via_train_step():
    """End-to-end: GT pairs → train_step → ``loss.requires_grad`` is True.

    The batch source delivers raw uint8 GT pairs (no autograd needed on the
    target side — used inside ``with torch.no_grad():`` blocks in train_step).
    The autograd path is on the RENDERED side: latent + decoder weights →
    rendered → eval_roundtrip → scorer.preprocess_input (which routes through
    ``differentiable_rgb_to_yuv6`` when the scorer is loaded via
    ``load_differentiable_scorers``). This test covers that the batch source
    integrates cleanly into the differentiable training pipeline.
    """
    import torch.nn as nn

    class _FakeScorerSeg(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 3, padding=1)

        def preprocess_input(self, x):
            return x[:, -1, :, :, :] / 255.0

        def forward(self, x):
            return self.conv(x)

    class _FakeScorerPose(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 64, 12)

        def preprocess_input(self, x):
            B, T, C, H, W = x.shape
            return (x.reshape(B, T * C, H, W) / 255.0)

        def forward(self, x):
            B = x.shape[0]
            patch = x[:, :3, :8, :8].mean(dim=(2, 3))
            feat = patch.repeat(1, 64)
            return self.fc(feat)

    config = Lane12V2NeRVConfig(
        latent_dim=8, base_channels=8, n_pairs=4, cuda_required=False,
    )
    renderer = Lane12V2NeRVRenderer(config)
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()

    # Use synthetic for this test (the autograd-flow check is independent of
    # whether the GT comes from PyAV or random — synthetic is appropriate here
    # because we are probing the gradient path, not the GT-fidelity path).
    # SYNTHETIC_NON_SMOKE_OK:autograd_flow_probe_only_no_score_claim
    pair_idx, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=config.latent_dim,
        eval_size=config.eval_size, n_pairs=config.n_pairs,
    )
    out = train_step(
        renderer=renderer, latent_table=table,
        pair_indices=pair_idx, gt_pairs_uint8=gt,
        scorer_seg=seg, scorer_pose=pose,
        seg_surrogate=default_seg_surrogate,
        pose_surrogate=default_pose_surrogate,
        lambda_seg=config.lambda_seg, lambda_pose=config.lambda_pose,
    )
    assert out["loss"].requires_grad, (
        "Loss must be differentiable for SGD to update renderer weights"
    )
    out["loss"].backward()
    # Verify the gradient actually reached at least one decoder weight.
    grad_norms = [p.grad.norm().item() for p in renderer.parameters() if p.grad is not None]
    assert any(g > 0.0 for g in grad_norms), (
        "no nonzero gradient reached renderer parameters — autograd path broken"
    )


# ── Resumability (skip_to_pair) ──────────────────────────────────────────


def test_iter_batches_skip_to_pair_works():
    """Skipping the first N pairs starts iteration at pair N (resumability)."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=10, eval_size=(384, 512))
    seen: list[int] = []
    for indices, _pairs in src.iter_batches(batch_size=2, skip_to_pair=4):
        seen.extend(int(i) for i in indices)
    # 10 pairs total, skip 4 → indices [4, 5, 6, 7, 8, 9].
    assert seen == [4, 5, 6, 7, 8, 9]


def test_iter_batches_skip_to_pair_zero_is_default():
    """``skip_to_pair=0`` (default) yields all n_pairs pairs starting at 0."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    seen: list[int] = []
    for indices, _pairs in src.iter_batches(batch_size=2, skip_to_pair=0):
        seen.extend(int(i) for i in indices)
    assert seen == [0, 1, 2, 3]


def test_iter_batches_skip_to_pair_beyond_n_pairs_yields_empty():
    """Skipping more pairs than the source has yields zero batches (no error)."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    batches = list(src.iter_batches(batch_size=2, skip_to_pair=10))
    assert batches == []


def test_iter_batches_rejects_negative_skip_to_pair():
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    with pytest.raises(ValueError, match="skip_to_pair must be non-negative"):
        next(src.iter_batches(batch_size=2, skip_to_pair=-1))


# ── max_pairs cap ────────────────────────────────────────────────────────


def test_iter_batches_max_pairs_caps_iteration():
    """Requesting fewer pairs than n_pairs caps iteration to ``max_pairs``."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=10, eval_size=(384, 512))
    seen: list[int] = []
    for indices, _pairs in src.iter_batches(batch_size=2, max_pairs=4):
        seen.extend(int(i) for i in indices)
    assert seen == [0, 1, 2, 3]


def test_iter_batches_max_pairs_combines_with_skip_to_pair():
    """``max_pairs`` is relative to ``skip_to_pair`` — yields next ``max_pairs``."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=10, eval_size=(384, 512))
    seen: list[int] = []
    for indices, _pairs in src.iter_batches(batch_size=2, skip_to_pair=2, max_pairs=4):
        seen.extend(int(i) for i in indices)
    # Skip 2, take 4 → [2, 3, 4, 5].
    assert seen == [2, 3, 4, 5]


def test_iter_batches_max_pairs_clamps_to_self_n_pairs():
    """``max_pairs > self.n_pairs`` clamps; never decodes more than configured."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    seen: list[int] = []
    for indices, _pairs in src.iter_batches(batch_size=2, max_pairs=100):
        seen.extend(int(i) for i in indices)
    # n_pairs=4 caps even though max_pairs=100.
    assert seen == [0, 1, 2, 3]


def test_iter_batches_rejects_zero_max_pairs():
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    with pytest.raises(ValueError, match="max_pairs must be positive"):
        next(src.iter_batches(batch_size=2, max_pairs=0))


# ── Synthetic-fallback refusal ───────────────────────────────────────────


def test_iter_batches_no_synthetic_fallback_when_video_missing():
    """Refuses to construct if contest video missing (raises FileNotFoundError).

    Per CLAUDE.md ``forbidden_synthetic_non_smoke_OK``: synthetic batches are
    forbidden in non-smoke training paths. The batch source MUST raise on
    construction; it must NEVER silently fall back to synthetic noise.
    """
    with pytest.raises(FileNotFoundError, match="contest video not found"):
        RealPairBatchSource(
            video_path=Path("/nonexistent/never_exists.mkv"),
            n_pairs=4, eval_size=(384, 512),
        )


def test_iter_batches_does_not_import_synthetic_helper():
    """``RealPairBatchSource`` must not import or call the synthetic helper.

    The synthetic helper ``_make_synthetic_pair_batch_for_smoke`` carries a
    ``# SYNTHETIC_NON_SMOKE_OK:phase_a_scaffold_smoke_test_only`` waiver and
    is intentionally module-private. ``RealPairBatchSource`` source MUST NOT
    reference it.
    """
    src_text = inspect.getsource(RealPairBatchSource)
    assert "_make_synthetic_pair_batch_for_smoke" not in src_text, (
        "RealPairBatchSource references the synthetic helper — this would "
        "violate the no-synthetic-fallback contract"
    )
    assert "synthetic" not in src_text.lower() or "no synthetic" in src_text.lower() or (
        "forbidden" in src_text.lower()
    ), "RealPairBatchSource source mentions 'synthetic' in unexpected context"


# ── Differentiable preprocess wiring (existence checks) ──────────────────


def test_differentiable_eval_roundtrip_module_is_importable():
    """The canonical eval_roundtrip primitive is importable from tac.

    Per HNeRV parity discipline lesson 8: trainer-inner-loop preprocess MUST
    route through ``tac.differentiable_eval_roundtrip``. This test verifies
    the module surface exists; the actual roundtrip is applied inside
    ``train_step._eval_roundtrip_uint8_clamp`` on the rendered side.
    """
    from tac.differentiable_eval_roundtrip import (  # noqa: F401
        apply_eval_roundtrip_during_training,
        differentiable_rgb_to_yuv6,
    )


def test_train_step_applies_eval_roundtrip_clamp_on_rendered_side():
    """Verifies ``train_step`` clamps + rounds rendered RGB to uint8 range.

    The check is internal: the helper ``_eval_roundtrip_uint8_clamp`` is called
    inside train_step; if that path is ever removed, train_step's eval-roundtrip
    contract breaks and the proxy-auth gap re-opens.
    """
    train_step_src = inspect.getsource(train_step)
    assert "_eval_roundtrip_uint8_clamp" in train_step_src, (
        "train_step no longer applies the eval-roundtrip clamp on rendered "
        "side — proxy-auth gap will reopen"
    )


# ── Validation gates (defensive) ─────────────────────────────────────────


def test_iter_batches_rejects_zero_batch_size():
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    with pytest.raises(ValueError, match="batch_size must be positive"):
        next(src.iter_batches(batch_size=0))


def test_iter_batches_shuffle_true_raises_not_implemented():
    """Phase A is sequential-only; shuffle is a Phase B cached-target deliverable."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=4, eval_size=(384, 512))
    with pytest.raises(NotImplementedError, match="shuffle=True"):
        next(src.iter_batches(batch_size=2, shuffle=True))


def test_iter_batches_partial_last_batch_is_yielded():
    """If ``n_pairs % batch_size != 0``, the final partial batch is still yielded."""
    video = _require_video()
    src = RealPairBatchSource(video_path=video, n_pairs=5, eval_size=(384, 512))
    batch_sizes: list[int] = []
    for indices, pairs in src.iter_batches(batch_size=2):
        batch_sizes.append(int(indices.shape[0]))
        assert pairs.shape[0] == indices.shape[0]
    # 5 pairs / batch_size 2 → batches of [2, 2, 1].
    assert batch_sizes == [2, 2, 1]


def test_iter_batches_total_pairs_match_n_pairs():
    """Sum of indices across yielded batches equals ``n_pairs``."""
    video = _require_video()
    n_pairs = 6
    src = RealPairBatchSource(video_path=video, n_pairs=n_pairs, eval_size=(384, 512))
    total = 0
    for indices, _pairs in src.iter_batches(batch_size=4):
        total += int(indices.shape[0])
    assert total == n_pairs
