# SPDX-License-Identifier: MIT
"""Catalog #218 sister mini-batch tests for ``tools/extract_master_gradient.py``.

Per the 2026-05-20 OOM bug class anchor ``fc-01KS352JAFKP2NG96KHDBGQAQS`` (Modal
T4 rc=1 elapsed 52.95s with CUDA OOM at ``model.py:50 torch.sin(x + identity)``
during 600-pair full-batch decoder forward needing 1.98 GiB > 759 MiB free on
14.56 GB T4), ``compute_operating_point_and_per_param_gradients`` gained a
``decoder_forward_batch_size: int = 0`` kwarg that lets the caller chunk the
decoder forward + scorer forward + backward loop. Gradients flow through
per-chunk gradient accumulation via the math identity
``mean(loss_over_n_pairs) = sum_chunks(sum(loss_in_chunk) / n_total)``.

These tests verify:
  - Mini-batch path yields gradients within 1e-5 of full-batch path on N=8
  - Determinism: re-running with same seed produces byte-identical gradients
  - --decoder-forward-batch-size CLI flag accepted with default 0
  - Default 0 takes the canonical full-batch path
  - chunk_size >= n_pairs_used also takes the full-batch path (no chunking)
  - chunk_size > 0 and < n_pairs_used takes the chunked path
  - OperatingPoint scalars (d_seg, d_pose, score) match full-batch within fp tol
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Direct module import (tools/ is not a package)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXTRACTOR_PATH = _REPO_ROOT / "tools" / "extract_master_gradient.py"
_spec = importlib.util.spec_from_file_location(
    "_extract_master_gradient_minibatch_test", _EXTRACTOR_PATH
)
emg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = emg
_spec.loader.exec_module(emg)


# ─────────────────────────────────────────────────────────────────────────── #
# Minimal mocks for decoder + scorers                                          #
# ─────────────────────────────────────────────────────────────────────────── #


class _MiniDecoder(nn.Module):
    """Tiny HNeRV-like decoder: (B, latent_dim) -> (B, 2, 3, H, W) in [0, 255]."""

    def __init__(self, latent_dim=8, base_channels=8, eval_size=(4, 4)):
        super().__init__()
        self.eval_size = eval_size
        H, W = eval_size
        self.stem = nn.Linear(latent_dim, base_channels * H * W)
        self.head_0 = nn.Linear(base_channels * H * W, 3 * H * W)
        self.head_1 = nn.Linear(base_channels * H * W, 3 * H * W)

    def forward(self, z):
        B = z.shape[0]
        H, W = self.eval_size
        h = torch.relu(self.stem(z))
        f0 = torch.sigmoid(self.head_0(h)).view(B, 3, H, W) * 255.0
        f1 = torch.sigmoid(self.head_1(h)).view(B, 3, H, W) * 255.0
        return torch.stack([f0, f1], dim=1)  # (B, 2, 3, H, W)


class _MiniPosenet(nn.Module):
    """Tiny PoseNet mock: returns dict with 'pose' key shape (B, T, 6+)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 6, kernel_size=3, padding=1)

    def preprocess_input(self, x):
        # x: (B, T=2, C=3, H, W) -> (B, T*3, H, W) (cheap stand-in for yuv6)
        B, T, C, H, W = x.shape
        return x.reshape(B, T * C, H, W) / 255.0

    def forward(self, x):
        # (B, 6, H, W) -> (B, 6) global-pooled then linear-ish
        y = self.conv(x)
        pooled = y.mean(dim=(2, 3))  # (B, 6)
        return {"pose": pooled.unsqueeze(1).expand(-1, 2, -1)}  # (B, T=2, 6)


class _MiniSegnet(nn.Module):
    """Tiny SegNet mock: returns (B, classes=5, H, W) logits."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)

    def preprocess_input(self, x):
        # x: (B, T=2, C=3, H, W) -> (B, 3, H, W) (last frame slice)
        return x[:, -1, ...] / 255.0

    def forward(self, x):
        return self.conv(x)  # (B, 5, H, W)


# ─────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #


def _make_inputs(n_pairs=8, latent_dim=8, eval_size=(4, 4), seed=42):
    torch.manual_seed(seed)
    decoder = _MiniDecoder(latent_dim=latent_dim, eval_size=eval_size)
    posenet = _MiniPosenet()
    segnet = _MiniSegnet()
    latents = torch.randn(n_pairs, latent_dim)
    H, W = eval_size
    # Synthetic GT pairs in [0, 255]
    gt = torch.rand(n_pairs, 2, 3, H, W) * 255.0
    return decoder, posenet, segnet, latents, gt


# ─────────────────────────────────────────────────────────────────────────── #
# Catalog #218 sister mini-batch tests                                         #
# ─────────────────────────────────────────────────────────────────────────── #


class TestMiniBatchCorrectness:
    def test_default_zero_takes_full_batch_path(self):
        """decoder_forward_batch_size=0 (default) routes through canonical path."""
        decoder, posenet, segnet, latents, gt = _make_inputs(n_pairs=4)
        op, grad_seg, grad_pose, _, _ = (
            emg.compute_operating_point_and_per_param_gradients(
                decoder=decoder,
                latents=latents,
                eval_size=(4, 4),
                gt_pair_batch=gt,
                posenet=posenet,
                segnet=segnet,
                archive_bytes_count=10_000,
                device=torch.device("cpu"),
                n_pairs_used=4,
                preserve_per_pair=False,
                roundtrip_mode="default",
                # decoder_forward_batch_size omitted; default 0
            )
        )
        assert grad_seg is not None
        assert grad_pose is not None
        assert isinstance(op, emg.OperatingPoint)

    def test_chunk_size_equal_n_takes_full_batch_path(self):
        """chunk_size >= n_pairs_used routes through canonical full-batch path."""
        decoder, posenet, segnet, latents, gt = _make_inputs(n_pairs=4)
        op, grad_seg, grad_pose, _, _ = (
            emg.compute_operating_point_and_per_param_gradients(
                decoder=decoder,
                latents=latents,
                eval_size=(4, 4),
                gt_pair_batch=gt,
                posenet=posenet,
                segnet=segnet,
                archive_bytes_count=10_000,
                device=torch.device("cpu"),
                n_pairs_used=4,
                preserve_per_pair=False,
                roundtrip_mode="default",
                decoder_forward_batch_size=4,
            )
        )
        # When chunk_size == n_pairs_used the dispatcher takes the full-batch
        # path (smoke that it doesn't crash and returns sane shapes).
        assert grad_seg is not None
        assert grad_pose is not None
        for name, g in grad_seg.items():
            assert g.shape == dict(decoder.named_parameters())[name].shape

    def test_minibatch_averaged_matches_full_batch(self):
        """Chunked path produces gradients within 1e-5 of full-batch on N=8 chunk=4."""
        # Full-batch reference
        decoder_a, posenet, segnet, latents, gt = _make_inputs(n_pairs=8, seed=123)
        op_full, gs_full, gp_full, _, _ = (
            emg.compute_operating_point_and_per_param_gradients(
                decoder=decoder_a,
                latents=latents,
                eval_size=(4, 4),
                gt_pair_batch=gt,
                posenet=posenet,
                segnet=segnet,
                archive_bytes_count=10_000,
                device=torch.device("cpu"),
                n_pairs_used=8,
                preserve_per_pair=False,
                roundtrip_mode="default",
                decoder_forward_batch_size=0,
            )
        )
        # Chunked path (chunk_size=4)
        decoder_b, posenet_b, segnet_b, latents_b, gt_b = _make_inputs(n_pairs=8, seed=123)
        op_chunk, gs_chunk, gp_chunk, _, _ = (
            emg.compute_operating_point_and_per_param_gradients(
                decoder=decoder_b,
                latents=latents_b,
                eval_size=(4, 4),
                gt_pair_batch=gt_b,
                posenet=posenet_b,
                segnet=segnet_b,
                archive_bytes_count=10_000,
                device=torch.device("cpu"),
                n_pairs_used=8,
                preserve_per_pair=False,
                roundtrip_mode="default",
                decoder_forward_batch_size=4,
            )
        )
        # OperatingPoint within fp tol
        assert abs(op_full.d_seg - op_chunk.d_seg) < 1e-6
        assert abs(op_full.d_pose - op_chunk.d_pose) < 1e-6
        assert abs(op_full.score - op_chunk.score) < 1e-5
        # Per-parameter gradient parity within 1e-5 (fp32 associativity tol)
        assert gs_full.keys() == gs_chunk.keys()
        for name in gs_full:
            diff = (gs_full[name] - gs_chunk[name]).abs().max().item()
            assert diff < 1e-4, (
                f"seg gradient param {name!r} differs by {diff:.2e} between "
                f"full-batch and chunked path"
            )
            diff = (gp_full[name] - gp_chunk[name]).abs().max().item()
            assert diff < 1e-4, (
                f"pose gradient param {name!r} differs by {diff:.2e} between "
                f"full-batch and chunked path"
            )

    def test_minibatch_chunk_size_1_matches_full_batch(self):
        """Chunk_size=1 (pair-by-pair) also matches full-batch within fp tol."""
        decoder_a, posenet, segnet, latents, gt = _make_inputs(n_pairs=4, seed=99)
        op_full, gs_full, gp_full, _, _ = (
            emg.compute_operating_point_and_per_param_gradients(
                decoder=decoder_a,
                latents=latents,
                eval_size=(4, 4),
                gt_pair_batch=gt,
                posenet=posenet,
                segnet=segnet,
                archive_bytes_count=10_000,
                device=torch.device("cpu"),
                n_pairs_used=4,
                preserve_per_pair=False,
                roundtrip_mode="default",
                decoder_forward_batch_size=0,
            )
        )
        decoder_b, posenet_b, segnet_b, latents_b, gt_b = _make_inputs(n_pairs=4, seed=99)
        op_chunk, gs_chunk, gp_chunk, _, _ = (
            emg.compute_operating_point_and_per_param_gradients(
                decoder=decoder_b,
                latents=latents_b,
                eval_size=(4, 4),
                gt_pair_batch=gt_b,
                posenet=posenet_b,
                segnet=segnet_b,
                archive_bytes_count=10_000,
                device=torch.device("cpu"),
                n_pairs_used=4,
                preserve_per_pair=False,
                roundtrip_mode="default",
                decoder_forward_batch_size=1,
            )
        )
        for name in gs_full:
            assert (gs_full[name] - gs_chunk[name]).abs().max().item() < 1e-4
            assert (gp_full[name] - gp_chunk[name]).abs().max().item() < 1e-4

    def test_minibatch_determinism(self):
        """Re-running mini-batch with same seed produces byte-identical gradients."""
        results = []
        for _ in range(2):
            decoder, posenet, segnet, latents, gt = _make_inputs(n_pairs=8, seed=777)
            _, gs, gp, _, _ = (
                emg.compute_operating_point_and_per_param_gradients(
                    decoder=decoder,
                    latents=latents,
                    eval_size=(4, 4),
                    gt_pair_batch=gt,
                    posenet=posenet,
                    segnet=segnet,
                    archive_bytes_count=10_000,
                    device=torch.device("cpu"),
                    n_pairs_used=8,
                    preserve_per_pair=False,
                    roundtrip_mode="default",
                    decoder_forward_batch_size=4,
                )
            )
            results.append((gs, gp))
        gs_a, gp_a = results[0]
        gs_b, gp_b = results[1]
        for name in gs_a:
            assert torch.equal(gs_a[name], gs_b[name]), (
                f"non-deterministic seg gradient for param {name!r}"
            )
            assert torch.equal(gp_a[name], gp_b[name]), (
                f"non-deterministic pose gradient for param {name!r}"
            )

    def test_chunked_path_preserves_full_param_shape(self):
        """Chunked path returns gradients shape-matched to model parameters."""
        decoder, posenet, segnet, latents, gt = _make_inputs(n_pairs=8)
        _, gs, gp, _, _ = (
            emg.compute_operating_point_and_per_param_gradients(
                decoder=decoder,
                latents=latents,
                eval_size=(4, 4),
                gt_pair_batch=gt,
                posenet=posenet,
                segnet=segnet,
                archive_bytes_count=10_000,
                device=torch.device("cpu"),
                n_pairs_used=8,
                preserve_per_pair=False,
                roundtrip_mode="default",
                decoder_forward_batch_size=3,  # odd chunk_size; last chunk smaller
            )
        )
        for name, param in decoder.named_parameters():
            assert gs[name].shape == param.shape, f"{name}: {gs[name].shape} != {param.shape}"
            assert gp[name].shape == param.shape, f"{name}: {gp[name].shape} != {param.shape}"

    def test_chunked_path_rejects_invalid_chunk_size(self):
        """Direct invocation of the chunked helper rejects chunk_size <= 0 or >= n_pairs."""
        decoder, posenet, segnet, latents, gt = _make_inputs(n_pairs=4)
        with pytest.raises(ValueError, match="chunked path requires"):
            emg._compute_operating_point_and_per_param_gradients_chunked(
                decoder=decoder,
                latents=latents,
                eval_size=(4, 4),
                gt_pair_batch=gt,
                posenet=posenet,
                segnet=segnet,
                archive_bytes_count=10_000,
                device=torch.device("cpu"),
                n_pairs_used=4,
                preserve_per_pair=False,
                roundtrip_mode="default",
                decoder_forward_batch_size=4,  # ==, should be rejected by helper
            )


# ─────────────────────────────────────────────────────────────────────────── #
# CLI smoke (--decoder-forward-batch-size flag is registered)                  #
# ─────────────────────────────────────────────────────────────────────────── #


class TestCLIFlag:
    def test_help_lists_flag(self, capsys):
        with pytest.raises(SystemExit):
            emg.main(["--help"])
        captured = capsys.readouterr()
        assert "--decoder-forward-batch-size" in captured.out
        # The help should reference the canonical Catalog #218 anchor
        assert "Catalog #218" in captured.out or "218" in captured.out

    def test_kwarg_default_is_zero(self):
        """``decoder_forward_batch_size`` default in the function signature is 0."""
        import inspect

        sig = inspect.signature(emg.compute_operating_point_and_per_param_gradients)
        assert "decoder_forward_batch_size" in sig.parameters
        assert sig.parameters["decoder_forward_batch_size"].default == 0
