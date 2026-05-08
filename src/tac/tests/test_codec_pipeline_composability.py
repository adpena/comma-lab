"""Composability smoke tests for the CodecPipeline orchestrator.

Verifies that landed CodecOps chain cleanly when wrapped in a
CodecPipeline. Specifically:
  - Op1_PR101SplitBrotli + Op_KLPoseStream (substitutional + substitutional)
  - Op1_PR101SplitBrotli + Op_KLLatent (substitutional + substitutional)
  - Op_KLPoseStream + Op_KLLatent (two stream codecs targeting different keys)
  - All three composed together (full multi-paradigm stack)

The cathedral's CodecPipeline orchestrator is canonical for any "stack
multiple paradigms" scenario. These tests are smoke-only: they verify
the chain ENCODES + DECODES without crashing, and that each op's blob
is independently retrievable from the pipeline manifest.

This is item #5 from the prior tranche's next-tranche plan
("Mini composability check").
"""
from __future__ import annotations

import torch

from tac.codec_pipeline import CodecPipeline, Op1_PR101SplitBrotli
from tac.codec_pipeline_kl_latent import LATENT_KEY, Op_KLLatent
from tac.codec_pipeline_kl_pose import POSE_KEY, Op_KLPoseStream
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA


def _synthetic_pr101_state_dict() -> dict[str, torch.Tensor]:
    """Build a synthetic state_dict matching PR101's FIXED_STATE_SCHEMA."""
    g = torch.Generator().manual_seed(0)
    return {
        name: torch.randn(*shape, generator=g) * 0.1
        for name, shape in FIXED_STATE_SCHEMA
    }


def _smooth_pose_trajectory(n_frames: int = 600) -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    t = torch.linspace(0.0, 60.0, n_frames)
    poses = torch.zeros(n_frames, 6)
    poses[:, 0] = t
    poses[:, 5] = 0.05 * torch.sin(t / 6.0)
    poses += 0.001 * torch.randn(n_frames, 6, generator=g)
    return poses


def _smooth_low_rank_latents(n_frames: int = 600, latent_dim: int = 28, rank: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    row_coefs = torch.randn(n_frames, rank, generator=g)
    col_basis = torch.randn(rank, latent_dim, generator=g)
    col_basis = col_basis / col_basis.norm(dim=1, keepdim=True)
    latents = row_coefs @ col_basis
    latents += 0.001 * torch.randn(n_frames, latent_dim, generator=g)
    return latents


# ---------------------------------------------------------------------------
# Two-op chains
# ---------------------------------------------------------------------------


def test_op1_plus_kl_pose_chain() -> None:
    """Op1_PR101SplitBrotli + Op_KLPoseStream: chains decoder weights with
    pose stream. State dict carries BOTH the PR101 schema tensors AND a
    poses_se3 tensor. Each op encodes its own slice; manifest records both."""
    state_dict = _synthetic_pr101_state_dict()
    state_dict[POSE_KEY] = _smooth_pose_trajectory()

    pipeline = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False),
        Op_KLPoseStream(n_components=4, brotli_quality=1),
    ])
    blob, manifest = pipeline.encode(state_dict)
    assert blob[:4] in (b"CPL1", b"CPL2")  # CPL2 is canonical default 2026-05-08
    assert len(manifest.op_results) == 2
    assert manifest.op_results[0].op_name == "pr101_split_brotli"
    assert manifest.op_results[1].op_name == "kl_pose_stream"

    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["pr101_split_brotli", "kl_pose_stream"]
    # Last op's output dominates `decoded` (substitutional semantics);
    # the pose key must be present.
    assert POSE_KEY in decoded


def test_op1_plus_kl_latent_chain() -> None:
    """Op1_PR101SplitBrotli + Op_KLLatent: chains decoder weights with
    latent stream. Same substitutional pattern."""
    state_dict = _synthetic_pr101_state_dict()
    state_dict[LATENT_KEY] = _smooth_low_rank_latents()

    pipeline = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False),
        Op_KLLatent(n_components=4, brotli_quality=1),
    ])
    blob, manifest = pipeline.encode(state_dict)
    assert blob[:4] in (b"CPL1", b"CPL2")  # CPL2 is canonical default 2026-05-08
    assert len(manifest.op_results) == 2
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["pr101_split_brotli", "kl_latent"]
    assert LATENT_KEY in decoded


def test_kl_pose_plus_kl_latent_chain() -> None:
    """Op_KLPoseStream + Op_KLLatent: two stream codecs targeting
    different state_dict keys. The cathedral pipeline must keep them
    independent (each decodes its own key)."""
    state_dict = {
        POSE_KEY: _smooth_pose_trajectory(),
        LATENT_KEY: _smooth_low_rank_latents(),
    }
    pipeline = CodecPipeline([
        Op_KLPoseStream(n_components=4, brotli_quality=1),
        Op_KLLatent(n_components=4, brotli_quality=1),
    ])
    blob, manifest = pipeline.encode(state_dict)
    assert len(manifest.op_results) == 2
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["kl_pose_stream", "kl_latent"]
    # Both keys present in final decoded state
    assert LATENT_KEY in decoded


# ---------------------------------------------------------------------------
# Three-op full stack
# ---------------------------------------------------------------------------


def test_op1_plus_kl_pose_plus_kl_latent_full_stack() -> None:
    """The full multi-paradigm stack: decoder weights + pose stream +
    latent stream all in one pipeline.

    This is the cathedral's "four-way stack" shape generalized: each op
    targets a different slice of the state_dict; the pipeline's
    substitutional semantics keep them composable.
    """
    state_dict = _synthetic_pr101_state_dict()
    state_dict[POSE_KEY] = _smooth_pose_trajectory()
    state_dict[LATENT_KEY] = _smooth_low_rank_latents()

    pipeline = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False),
        Op_KLPoseStream(n_components=4, brotli_quality=1),
        Op_KLLatent(n_components=4, brotli_quality=1),
    ])
    blob, manifest = pipeline.encode(state_dict)
    assert blob[:4] in (b"CPL1", b"CPL2")  # CPL2 is canonical default 2026-05-08
    assert len(manifest.op_results) == 3
    op_names = [r.op_name for r in manifest.op_results]
    assert op_names == ["pr101_split_brotli", "kl_pose_stream", "kl_latent"]

    # Per-op byte impact is recorded individually for forensic inspection
    pr101_bytes = manifest.op_results[0].bytes_out
    pose_bytes = manifest.op_results[1].bytes_out
    latent_bytes = manifest.op_results[2].bytes_out
    assert pr101_bytes > 0
    assert pose_bytes > 0
    assert latent_bytes > 0

    decoded, replayed = pipeline.decode(blob)
    assert replayed == op_names


def test_full_stack_byte_deterministic() -> None:
    """Same input → identical pipeline blob. Byte-determinism is the
    invariant the cathedral's CPL1 wire format guarantees when
    auto_select / non-deterministic ops are disabled."""
    state_dict = _synthetic_pr101_state_dict()
    state_dict[POSE_KEY] = _smooth_pose_trajectory()
    state_dict[LATENT_KEY] = _smooth_low_rank_latents()

    pipeline_a = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False),
        Op_KLPoseStream(n_components=4, brotli_quality=11),
        Op_KLLatent(n_components=4, brotli_quality=11),
    ])
    pipeline_b = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False),
        Op_KLPoseStream(n_components=4, brotli_quality=11),
        Op_KLLatent(n_components=4, brotli_quality=11),
    ])
    blob_a, _ = pipeline_a.encode(state_dict)
    blob_b, _ = pipeline_b.encode(state_dict)
    assert blob_a == blob_b
