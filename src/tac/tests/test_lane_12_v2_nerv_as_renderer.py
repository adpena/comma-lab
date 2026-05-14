# SPDX-License-Identifier: MIT
"""Tests for Lane 12-v2 NeRV-as-renderer Phase A scaffold.

Coverage:
- Forward shape correctness (B, latent_dim) → (B, 2, 3, H, W).
- Score-aware loss returns finite + non-zero on a smoke batch.
- Archive roundtrip determinism (encode → parse → forward → match within
  quantization tolerance).
- Per-tensor quantization bit budget honors INT8 + fp16 scale.
- Per-frame conditioning correctness (different latents → different outputs).
- Validation gates (config, schema, shapes).
- Smoke training step doesn't NaN.
- Inflate LOC budget enforced.
- Inflate runtime dep closure (only torch + brotli + tac.lane_12_v2_*).
- Real-pair batch source raises if contest video missing (fail-fast).
- ARCHIVE_GRAMMAR machine-readable manifest is internally consistent.
- export_to_archive returns sha256 + writes deterministic bytes.
- Phase B preconditions return PENDING in Phase A.
- FiLM raises NotImplementedError per design memo.
"""
from __future__ import annotations

import ast
import hashlib
import struct
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# R5-1 finding (2026-05-13): repo-root-relative test fixtures replace prior
# operator-absolute paths per CLAUDE.md "Public Disclosure Hygiene" + R5-1 fix
# in feedback_recursive_review_r5_LANDED_20260513.md. This file lives at
# src/tac/tests/test_lane_12_v2_nerv_as_renderer.py so parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

from tac.lane_12_v2_nerv_as_renderer import (
    ARCHIVE_GRAMMAR,
    LANE_12_V2_FORMAT_VERSION,
    LANE_12_V2_MAGIC,
    Lane12V2LatentTable,
    Lane12V2NeRVConfig,
    Lane12V2NeRVRenderer,
    RealPairBatchSource,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_to_archive,
    phase_b_preconditions_status,
    train_step,
)


# ── Smoke config (fast tests on CPU) ─────────────────────────────────────


def _smoke_config(latent_dim: int = 8, n_pairs: int = 4) -> Lane12V2NeRVConfig:
    """Tiny config for fast CPU tests. Production config uses latent_dim=16, n_pairs=600."""
    return Lane12V2NeRVConfig(
        latent_dim=latent_dim,
        base_channels=8,  # tiny for speed
        n_pairs=n_pairs,
        cuda_required=False,
    )


# ── Forward shape ────────────────────────────────────────────────────────


def test_renderer_forward_shape_matches_design():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    z = torch.randn(2, config.latent_dim)
    out = renderer(z)
    assert out.shape == (2, 2, 3, *config.eval_size), (
        f"expected (2, 2, 3, 384, 512), got {tuple(out.shape)}"
    )


def test_renderer_forward_outputs_in_0_255_range():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    z = torch.randn(2, config.latent_dim)
    out = renderer(z)
    assert (out >= 0).all() and (out <= 255).all(), "RGB out of [0, 255] range"


def test_renderer_forward_rejects_wrong_latent_dim():
    config = _smoke_config(latent_dim=8)
    renderer = Lane12V2NeRVRenderer(config)
    bad_z = torch.randn(2, 7)  # wrong latent_dim
    with pytest.raises(ValueError, match="forward expected"):
        renderer(bad_z)


def test_renderer_forward_rejects_wrong_dim_count():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    bad_z = torch.randn(2, 3, config.latent_dim)  # 3D not 2D
    with pytest.raises(ValueError, match="forward expected"):
        renderer(bad_z)


# ── Per-frame conditioning correctness ───────────────────────────────────


def test_different_latents_produce_different_outputs():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    z1 = torch.randn(1, config.latent_dim)
    z2 = z1 + 1.0  # noticeably different
    out1 = renderer(z1)
    out2 = renderer(z2)
    diff = (out1 - out2).abs().mean().item()
    assert diff > 0.5, f"different latents should produce different outputs, got mean abs diff {diff}"


def test_same_latent_produces_same_output():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    renderer.eval()
    z = torch.randn(1, config.latent_dim)
    out1 = renderer(z)
    out2 = renderer(z)
    assert torch.allclose(out1, out2), "deterministic forward expected"


# ── Latent table ─────────────────────────────────────────────────────────


def test_latent_table_returns_correct_shape():
    table = Lane12V2LatentTable(n_pairs=10, latent_dim=8)
    indices = torch.tensor([0, 3, 7], dtype=torch.long)
    z = table(indices)
    assert z.shape == (3, 8)


def test_latent_table_init_is_small_random():
    table = Lane12V2LatentTable(n_pairs=10, latent_dim=8)
    # Init std is 0.01; check magnitudes are not gigantic.
    assert table.embedding.weight.abs().max().item() < 0.1


# ── Score-aware loss + train_step ────────────────────────────────────────


class _FakeScorerSeg(nn.Module):
    """Fake SegNet for unit tests — small CNN that returns 5-class logits.

    Mirrors the contest scorer contract: ``preprocess_input(x)`` takes a
    5-D ``(B, T, C, H, W)`` tensor and returns a 4-D ``(B, C, H, W)`` tensor
    (last frame only — matches modules.py SegNet semantics).
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 3, padding=1)
        self.preprocess_calls = 0

    def preprocess_input(self, x):
        self.preprocess_calls += 1
        assert x.ndim == 5
        # x: (B, T, C, H, W); SegNet uses LAST frame only per modules.py.
        return x[:, -1, :, :, :] / 255.0

    def forward(self, x):
        assert x.ndim == 4
        # x: 4-D (B, C, H, W) post-preprocess.
        return self.conv(x)


class _FakeScorerPose(nn.Module):
    """Fake PoseNet for unit tests — small MLP that returns (B, 12) pose.

    Mirrors the contest scorer contract: ``preprocess_input(x)`` takes a
    5-D ``(B, T, C, H, W)`` tensor and returns a 4-D tensor with both frames
    concatenated along the channel dim (matches PoseNet's 12-channel YUV6
    pair input semantics).
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 64, 12)
        self.preprocess_calls = 0

    def preprocess_input(self, x):
        self.preprocess_calls += 1
        assert x.ndim == 5
        # x: (B, T, C, H, W) → (B, T*C, H, W) for the test fake. Real PoseNet
        # does YUV6 conversion + resize; we just collapse T into C for shape.
        B, T, C, H, W = x.shape
        return (x.reshape(B, T * C, H, W) / 255.0)

    def forward(self, x):
        assert x.ndim == 4
        # x: 4-D (B, C, H, W). Reduce to 12 dims via tiny patch + linear.
        B = x.shape[0]
        patch = x[:, :3, :8, :8].mean(dim=(2, 3))  # (B, 3)
        feat = patch.repeat(1, 64)  # (B, 3*64)
        return self.fc(feat)


def test_train_step_returns_finite_nonzero_loss():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()

    # SYNTHETIC_NON_SMOKE_OK:phase_a_scaffold_smoke_test_only
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
    loss = out["loss"]
    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.item() != 0.0, "loss should not be exactly zero"
    assert loss.requires_grad, "loss must be differentiable"
    assert seg.preprocess_calls == 2
    assert pose.preprocess_calls == 2


def test_train_step_eval_roundtrip_false_is_forbidden():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()

    # SYNTHETIC_NON_SMOKE_OK:phase_a_scaffold_smoke_test_only
    pair_idx, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=config.latent_dim,
        eval_size=config.eval_size, n_pairs=config.n_pairs,
    )
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step(
            renderer=renderer, latent_table=table,
            pair_indices=pair_idx, gt_pairs_uint8=gt,
            scorer_seg=seg, scorer_pose=pose,
            seg_surrogate=default_seg_surrogate,
            pose_surrogate=default_pose_surrogate,
            lambda_seg=config.lambda_seg, lambda_pose=config.lambda_pose,
            eval_roundtrip=False,
        )


def test_train_step_loss_does_not_nan_on_smoke_batch():
    """Smoke training step doesn't NaN on synthetic input (regression test)."""
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()

    # SYNTHETIC_NON_SMOKE_OK:phase_a_scaffold_smoke_test_only
    pair_idx, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=config.latent_dim,
        eval_size=config.eval_size, n_pairs=config.n_pairs,
    )
    optim = torch.optim.SGD(
        list(renderer.parameters()) + list(table.parameters()), lr=1e-4
    )
    for _ in range(3):
        out = train_step(
            renderer=renderer, latent_table=table,
            pair_indices=pair_idx, gt_pairs_uint8=gt,
            scorer_seg=seg, scorer_pose=pose,
            seg_surrogate=default_seg_surrogate,
            pose_surrogate=default_pose_surrogate,
            lambda_seg=config.lambda_seg, lambda_pose=config.lambda_pose,
        )
        optim.zero_grad()
        out["loss"].backward()
        optim.step()
        assert torch.isfinite(out["loss"]).all(), "NaN/Inf in loss"


# ── Quantization ─────────────────────────────────────────────────────────


def test_quantize_per_tensor_int8_returns_int8_and_fp16_scale():
    t = torch.randn(10, 10) * 5.0
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert scale.numel() == 1


def test_quantize_per_tensor_recovers_within_tolerance():
    t = torch.randn(100) * 2.0
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    recovered = q.float() * scale.float()
    err = (t - recovered).abs().max().item()
    # int8 + fp16 scale: max quant error ≈ scale; for std=2 input, ~2/127 ≈ 0.016.
    assert err < 0.1, f"quantization error too large: {err}"


def test_quantize_per_tensor_handles_zero_tensor():
    t = torch.zeros(5)
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    # Should not divide by zero; q should be all zeros.
    assert (q == 0).all()


# ── Archive roundtrip determinism ────────────────────────────────────────


def test_export_to_archive_returns_sha256():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)
    out_path = Path("/tmp/lane_12_v2_test_archive.bin")
    sha = export_to_archive(renderer=renderer, latent_table=table, output_path=out_path)
    assert len(sha) == 64
    assert out_path.exists()
    actual = hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert sha == actual


def test_export_to_archive_starts_with_magic():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)
    out_path = Path("/tmp/lane_12_v2_magic_test.bin")
    export_to_archive(renderer=renderer, latent_table=table, output_path=out_path)
    archive_bytes = out_path.read_bytes()
    assert archive_bytes[:4] == LANE_12_V2_MAGIC


def test_export_to_archive_header_fields_match_config():
    config = _smoke_config(latent_dim=12, n_pairs=7)
    renderer = Lane12V2NeRVRenderer(config)
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)
    out_path = Path("/tmp/lane_12_v2_header_test.bin")
    export_to_archive(renderer=renderer, latent_table=table, output_path=out_path)
    archive_bytes = out_path.read_bytes()
    version, latent_dim, n_pairs, base_channels = struct.unpack(
        "<HHHH", archive_bytes[4:12]
    )
    assert version == LANE_12_V2_FORMAT_VERSION
    assert latent_dim == 12
    assert n_pairs == 7
    assert base_channels == config.base_channels


def test_archive_roundtrip_via_inflate_module():
    """Encode → parse → load_state_dict → forward gives same shape + similar values."""
    from tac.inflate.lane_12_v2_inflate import _parse_archive

    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    renderer.eval()
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)

    out_path = Path("/tmp/lane_12_v2_roundtrip_test.bin")
    export_to_archive(renderer=renderer, latent_table=table, output_path=out_path)

    parsed_config, parsed_sd, parsed_latents = _parse_archive(out_path.read_bytes())
    assert parsed_config.latent_dim == config.latent_dim
    assert parsed_config.n_pairs == config.n_pairs
    assert parsed_config.base_channels == config.base_channels

    # Load parsed weights into a fresh renderer; check forward matches up to
    # quantization tolerance.
    fresh = Lane12V2NeRVRenderer(parsed_config)
    fresh.load_state_dict(parsed_sd)
    fresh.eval()

    z = torch.randn(1, config.latent_dim)
    with torch.no_grad():
        out_orig = renderer(z)
        out_fresh = fresh(z)
    # int8 + fp16 + sin-deep-net amplifies error; use loose tolerance.
    diff = (out_orig - out_fresh).abs().mean().item()
    assert diff < 50.0, f"roundtrip diff {diff} exceeds tolerance"


def test_archive_export_is_deterministic():
    """Same config + same weights → same archive bytes (Lesson 11 no-op detector)."""
    config = _smoke_config()
    torch.manual_seed(42)
    renderer = Lane12V2NeRVRenderer(config)
    table = Lane12V2LatentTable(config.n_pairs, config.latent_dim)
    out1 = Path("/tmp/lane_12_v2_det_1.bin")
    out2 = Path("/tmp/lane_12_v2_det_2.bin")
    sha1 = export_to_archive(renderer=renderer, latent_table=table, output_path=out1)
    sha2 = export_to_archive(renderer=renderer, latent_table=table, output_path=out2)
    assert sha1 == sha2, "export should be deterministic for same weights"


# ── Validation gates ─────────────────────────────────────────────────────


def test_config_rejects_zero_latent_dim():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        Lane12V2NeRVConfig(latent_dim=0, cuda_required=False)


def test_config_rejects_film():
    with pytest.raises(NotImplementedError, match="FiLM"):
        Lane12V2NeRVConfig(use_film=True, cuda_required=False)


def test_config_rejects_non_8_bit_quant():
    with pytest.raises(ValueError, match="8-bit per-tensor"):
        Lane12V2NeRVConfig(quantization_bits=4, cuda_required=False)


def test_config_rejects_non_2_frames_per_pair():
    with pytest.raises(ValueError, match="frames_per_pair=2"):
        Lane12V2NeRVConfig(frames_per_pair=1, cuda_required=False)


def test_config_rejects_non_6_stages():
    with pytest.raises(ValueError, match="n_stages=6"):
        Lane12V2NeRVConfig(n_stages=4, cuda_required=False)


# ── Schema correctness ─────────────────────────────────────────────────────


def test_renderer_schema_covers_every_state_dict_key():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    schema_keys = {k for k, _ in renderer.schema}
    sd_keys = set(renderer.state_dict().keys())
    # Schema may exclude Identity-skip layers (no parameters); those are fine.
    missing = sd_keys - schema_keys
    assert not missing, f"schema missing keys: {missing}"


def test_renderer_schema_shapes_match_state_dict():
    config = _smoke_config()
    renderer = Lane12V2NeRVRenderer(config)
    sd = renderer.state_dict()
    for key, shape in renderer.schema:
        assert tuple(sd[key].shape) == shape, (
            f"schema shape mismatch for {key}: schema={shape}, sd={tuple(sd[key].shape)}"
        )


# ── ARCHIVE_GRAMMAR machine-readable manifest ────────────────────────────


def test_archive_grammar_matches_module_constants():
    assert ARCHIVE_GRAMMAR["format_version"] == LANE_12_V2_FORMAT_VERSION
    assert ARCHIVE_GRAMMAR["magic"] == LANE_12_V2_MAGIC.decode("ascii")


def test_archive_grammar_declares_all_sections():
    section_names = {s["name"] for s in ARCHIVE_GRAMMAR["sections"]}
    assert section_names == {
        "header", "decoder_blob", "scale_table", "latent_blob", "sidecar_blob"
    }


def test_archive_grammar_sidecar_is_empty_in_phase_a():
    sidecar = next(
        s for s in ARCHIVE_GRAMMAR["sections"] if s["name"] == "sidecar_blob"
    )
    assert sidecar.get("phase_a_empty") is True


# ── RealPairBatchSource fail-fast ────────────────────────────────────────


def test_real_pair_batch_source_raises_if_video_missing():
    with pytest.raises(FileNotFoundError, match="contest video not found"):
        RealPairBatchSource(
            video_path=Path("/nonexistent/video.mkv"),
            n_pairs=600, eval_size=(384, 512),
        )


def test_real_pair_batch_source_shuffle_requires_cached_phase_b_source():
    upstream_video = _REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not upstream_video.exists():
        pytest.skip("upstream/videos/0.mkv not present in this checkout")
    src = RealPairBatchSource(video_path=upstream_video, n_pairs=600, eval_size=(384, 512))
    with pytest.raises(NotImplementedError, match="shuffle=True"):
        next(src.iter_batches(batch_size=2, shuffle=True))


def test_real_pair_batch_source_yields_real_decoded_pairs():
    """Verify the linter-implemented iter_batches yields real (B, 2, 3, H, W) tensors.

    Per CLAUDE.md HNeRV parity discipline: synthetic pair batches are forbidden
    in non-smoke training paths. This test verifies the real-decode path
    produces real PyAV-decoded contest video frames.
    """
    upstream_video = _REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not upstream_video.exists():
        pytest.skip("upstream/videos/0.mkv not present in this checkout")
    try:
        import av  # noqa: F401
    except ImportError:
        pytest.skip("PyAV not installed in this environment")

    src = RealPairBatchSource(video_path=upstream_video, n_pairs=4, eval_size=(384, 512))
    indices, pairs = next(iter(src.iter_batches(batch_size=2)))
    assert indices.dtype == torch.long
    assert indices.shape == (2,)
    assert pairs.dim() == 5
    assert pairs.shape[0] == 2  # batch
    assert pairs.shape[1] == 2  # frames per pair
    assert pairs.shape[2] == 3  # RGB channels
    # Camera resolution per upstream/frame_utils.py.
    assert pairs.shape[-2:] == (874, 1164)


def test_real_pair_batch_source_rejects_zero_batch_size():
    upstream_video = _REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not upstream_video.exists():
        pytest.skip("upstream/videos/0.mkv not present in this checkout")
    src = RealPairBatchSource(video_path=upstream_video, n_pairs=4, eval_size=(384, 512))
    with pytest.raises(ValueError, match="batch_size must be positive"):
        next(src.iter_batches(batch_size=0))


# ── Phase B preconditions ───────────────────────────────────────────────


def test_phase_b_preconditions_status_is_pending():
    """Legacy snapshot: Phase B remains gated with all 4 PENDING when consulting
    the static 2026-05-09 snapshot (consult_session_state=False).

    The default behaviour now consults actual session state; tests that need
    the static snapshot pass ``consult_session_state=False`` explicitly.
    """
    status = phase_b_preconditions_status(consult_session_state=False)
    # Two MET (scaffold tests + batch source), four PENDING, one bool gate.
    assert status["phase_a_scaffold_tests_pass"] == "MET"
    assert status["real_pair_batch_source_implemented"] == "MET"
    pending = [
        "t7_t8_t11_subadditivity_disambiguator_returned",
        "t13_t19_wired_into_trainer",
        "strict_preflight_124_warn_only_landed",
        "operator_phase_b_authorization",
    ]
    for key in pending:
        assert status[key] == "PENDING", f"{key} should be PENDING, got {status[key]}"
    assert status["any_pending_blocks_phase_b_dispatch"] is True


def test_phase_b_preconditions_status_consults_session_state_by_default(tmp_path, monkeypatch):
    """Default: consults memory dir (PACT_MEMORY_DIR override) for landed evidence.

    Use a clean tmp_path memory dir to make the test hermetic — actual
    session state varies across machines and over time.
    """
    monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path))
    # Empty dir: all 4 dynamic flags are PENDING.
    status = phase_b_preconditions_status()  # default: consult_session_state=True
    assert status["session_state_consulted"] is True
    assert status["phase_a_scaffold_tests_pass"] == "MET"
    assert status["real_pair_batch_source_implemented"] == "MET"
    assert status["t7_t8_t11_subadditivity_disambiguator_returned"] == "PENDING"
    assert status["t13_t19_wired_into_trainer"] == "PENDING"
    assert status["operator_phase_b_authorization"] == "PENDING"
    # strict_preflight_124_warn_only_landed is import-based; depends on
    # tac.preflight, which is always present in this test runner.
    assert status["strict_preflight_124_warn_only_landed"] == "MET"
    assert status["any_pending_blocks_phase_b_dispatch"] is True


def test_phase_b_preconditions_status_detects_memo_evidence(tmp_path, monkeypatch):
    """Plant 2 evidence memos; verify they flip MET."""
    monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path))
    (tmp_path / "feedback_t7_t8_t11_sub_additivity_disambiguator_landed_20260509.md").write_text(
        "scaffold memo body\n"
    )
    (tmp_path / "feedback_t13_t19_phase1_trainer_integration_landed_20260509.md").write_text(
        "scaffold memo body\n"
    )
    status = phase_b_preconditions_status()
    assert status["t7_t8_t11_subadditivity_disambiguator_returned"] == "MET"
    assert status["t13_t19_wired_into_trainer"] == "MET"
    # operator authorization still PENDING (no matching name pattern + no body token)
    assert status["operator_phase_b_authorization"] == "PENDING"


def test_phase_b_preconditions_status_detects_operator_authorization_token(
    tmp_path, monkeypatch
):
    """Plant a properly-named auth memo containing the explicit body token; verify MET.

    Per the hardened ``_check_operator_phase_b_authorization`` (anti-spoofing
    fix from codex round 8 HIGH 2 hardening), the memo MUST match the name
    pattern ``feedback_lane_12_v2*phase_b*authoriz*.md`` AND carry the
    explicit unquoted line-level token. Arbitrary ``feedback_*.md`` files
    are no longer scanned for the token (descriptive mentions in landing
    memos are not authorization).

    Note: this test exercises the legacy `~/.claude` scan path (no
    ``auth_memo_path`` passed). For the Option C path (committed
    repo-relative auth memo), see test_check_150_phase_b_auth_memo_in_repo.py.
    """
    monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path))
    (tmp_path / "feedback_lane_12_v2_phase_b_authorization_20260509.md").write_text(
        "operator_phase_b_authorization=true\n"
    )
    status = phase_b_preconditions_status()
    assert status["operator_phase_b_authorization"] == "MET"


# ── Inflate LOC budget ───────────────────────────────────────────────────


def _effective_loc(path: Path) -> int:
    """Count effective LOC: non-blank, non-pure-comment, non-docstring."""
    src_text = path.read_text()
    src_lines = src_text.splitlines()
    tree = ast.parse(src_text)
    docstring_lines: set[int] = set()

    def collect(node):
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            ds = body[0]
            for ln in range(ds.lineno, ds.end_lineno + 1):
                docstring_lines.add(ln)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                collect(child)

    collect(tree)
    loc = 0
    for i, line in enumerate(src_lines, start=1):
        s = line.strip()
        if not s or s.startswith("#") or i in docstring_lines:
            continue
        loc += 1
    return loc


def test_lane_12_v2_inflate_loc_budget():
    """Inflate.py MUST be ≤ 100 effective LOC per Lesson 4."""
    inflate_path = _REPO_ROOT / "src" / "tac" / "inflate" / "lane_12_v2_inflate.py"
    loc = _effective_loc(inflate_path)
    assert loc <= 100, (
        f"inflate LOC budget exceeded: {loc} > 100. "
        f"Per HNeRV retrospective Lesson 4, inflate.py must be ≤ 100 LOC."
    )


def test_lane_12_v2_inflate_runtime_dep_closure():
    """Reference inflate may depend only on torch + brotli + tac oracle code.

    AST scan of import statements; allowed deps:
    - stdlib (io, struct, sys, pathlib)
    - torch, torch.nn.functional
    - brotli
    - tac.lane_12_v2_nerv_as_renderer
    """
    inflate_path = _REPO_ROOT / "src" / "tac" / "inflate" / "lane_12_v2_inflate.py"
    src = inflate_path.read_text()
    tree = ast.parse(src)
    allowed_top_level = {
        "io", "struct", "sys", "pathlib", "torch", "brotli", "tac",
        # transitive: __future__ for annotations
        "__future__",
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                found.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                found.add(top)
    extras = found - allowed_top_level
    assert not extras, (
        f"inflate has disallowed deps: {extras}. "
        f"Allowed: {allowed_top_level}. This Phase A reference inflate is "
        f"research_only; Phase B contest inflate must remove the tac import."
    )


# ── Default surrogates produce sane outputs ──────────────────────────────


def test_default_seg_surrogate_returns_finite():
    pred = torch.randn(2, 5, 8, 8)
    target = torch.randn(2, 5, 8, 8)
    loss = default_seg_surrogate(pred, target)
    assert torch.isfinite(loss).all()
    assert loss.item() >= 0.0


def test_default_pose_surrogate_returns_finite():
    pred = torch.randn(2, 12)
    target = torch.randn(2, 12)
    loss = default_pose_surrogate(pred, target)
    assert torch.isfinite(loss).all()
    assert loss.item() >= 0.0


def test_default_pose_surrogate_uses_first_6_dims():
    pred = torch.zeros(2, 12)
    target = torch.zeros(2, 12)
    target[:, 6:] = 100.0  # last 6 dims should be ignored
    loss = default_pose_surrogate(pred, target)
    assert loss.item() == 0.0
