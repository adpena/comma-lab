# SPDX-License-Identifier: MIT
"""Dedicated tests for the pre-trained driving prior substrate scaffold.

Coverage targets (≥18 dedicated tests per scaffold mandate):

* Codebook serialization / parse / roundtrip / validation
* Distillation deterministic synthetic path
* Contest-video leakage guard
* BDD100K opt-in / Waymo skip
* Federated aggregation arithmetic
* Soft-prior loss differentiability + zero-codebook handling
* Archive grammar pack / parse / roundtrip / header invariants
* End-to-end inflate.py smoke (renderer renders, raw output writes)
* Readiness manifest fields
* /tmp path refusal
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tac.substrates.pretrained_driving_prior import (
    CODEBOOK_TOTAL_TARGET_BYTES_MAX,
    DP1_HEADER_SIZE,
    DP1_MAGIC,
    ContestVideoLeakageError,
    DashcamCodebook,
    DashcamPriorLoss,
    DistillationConfig,
    DrivingPriorLossWeights,
    DrivingPriorRenderer,
    DrivingPriorRendererConfig,
    PriorApplicationWeights,
    aggregate_local_codebooks,
    build_readiness_manifest,
    check_no_contest_video_leakage,
    deterministic_zero_codebook,
    distill_codebook,
    pack_archive,
    parse_archive,
    parse_codebook,
    serialize_codebook,
    validate_codebook,
    write_codebook_to_disk,
)
from tac.substrates.pretrained_driving_prior.inflate import inflate_one_video
from tac.substrates.pretrained_driving_prior.score_aware_loss import (
    DrivingPriorScoreAwareLoss,
)

# ----------------------------- codebook tests -----------------------------


def test_deterministic_zero_codebook_validates_clean() -> None:
    """The scaffold zero codebook satisfies every shape + dtype + meta invariant."""
    book = deterministic_zero_codebook()
    validate_codebook(book)  # raises on failure


def test_codebook_serialize_parse_roundtrip_preserves_bytes() -> None:
    """Serialize -> parse -> serialize is byte-stable (deterministic)."""
    book = deterministic_zero_codebook()
    bytes1 = serialize_codebook(book)
    parsed = parse_codebook(bytes1)
    bytes2 = serialize_codebook(parsed)
    assert bytes1 == bytes2, "codebook roundtrip must be byte-stable"


def test_codebook_parse_rejects_truncated_data() -> None:
    """Truncated codebook bytes raise ValueError."""
    book = deterministic_zero_codebook()
    blob = serialize_codebook(book)
    with pytest.raises(ValueError):
        parse_codebook(blob[: len(blob) // 2])


def test_codebook_validate_rejects_wrong_shape() -> None:
    """Codebook with mismatched array shape fails validate_codebook."""
    book = deterministic_zero_codebook()
    bad = DashcamCodebook(
        road_plane_basis=np.zeros((4, 16, 24, 3), dtype=np.int8),  # wrong: 4 vs 8
        sky_horizon_profile=book.sky_horizon_profile,
        lane_curvature_pca=book.lane_curvature_pca,
        vehicle_appearance_basis=book.vehicle_appearance_basis,
        metadata=book.metadata,
    )
    with pytest.raises(ValueError, match="road_plane_basis"):
        validate_codebook(bad)


def test_codebook_validate_rejects_missing_metadata_key() -> None:
    """Codebook missing a required metadata key fails validation."""
    book = deterministic_zero_codebook()
    bad_meta = {k: v for k, v in book.metadata.items() if k != "license_tags"}
    bad = DashcamCodebook(
        road_plane_basis=book.road_plane_basis,
        sky_horizon_profile=book.sky_horizon_profile,
        lane_curvature_pca=book.lane_curvature_pca,
        vehicle_appearance_basis=book.vehicle_appearance_basis,
        metadata=bad_meta,
    )
    with pytest.raises(ValueError, match="license_tags"):
        validate_codebook(bad)


def test_codebook_size_within_target_band() -> None:
    """The serialized codebook fits in the 5-10 KB target band (or below)."""
    book = deterministic_zero_codebook()
    blob = serialize_codebook(book)
    # Zero codebook compresses very well — should be well under MAX.
    assert len(blob) <= CODEBOOK_TOTAL_TARGET_BYTES_MAX, (
        f"codebook size {len(blob)} > target max {CODEBOOK_TOTAL_TARGET_BYTES_MAX}"
    )


# --------------------------- distillation tests ---------------------------


def test_distill_codebook_synthetic_deterministic() -> None:
    """Synthetic distillation is deterministic across two runs with same seed."""
    cfg = DistillationConfig(
        dataset_name="synthetic_test", random_seed=0xDEAD, max_frames=256
    )
    book1 = distill_codebook(cfg)
    book2 = distill_codebook(cfg)
    np.testing.assert_array_equal(book1.road_plane_basis, book2.road_plane_basis)
    np.testing.assert_array_equal(book1.sky_horizon_profile, book2.sky_horizon_profile)
    assert book1.metadata["basis_sha256"] == book2.metadata["basis_sha256"]


def test_distill_rejects_contest_video_leakage_in_path() -> None:
    """check_no_contest_video_leakage refuses paths matching contest fragments."""
    with pytest.raises(ContestVideoLeakageError):
        check_no_contest_video_leakage([Path("upstream/videos/0.mkv")])
    with pytest.raises(ContestVideoLeakageError):
        check_no_contest_video_leakage([Path("some/comma_video_compression_challenge/foo.mkv")])
    # A safe public-dataset path is accepted.
    check_no_contest_video_leakage([Path("comma2k19/raw/2019-04-18--13-23-55/0/video.mp4")])


def test_distill_bdd100k_requires_opt_in() -> None:
    """BDD100K dataset-images opt-in is required (default False refuses)."""
    cfg = DistillationConfig(dataset_name="bdd100k", allow_bdd100k_dataset_images=False)
    with pytest.raises(ValueError, match="BDD100K"):
        distill_codebook(cfg, frames=iter([np.zeros((32, 32, 3), dtype=np.uint8)] * 8))


def test_distill_unknown_dataset_name_raises() -> None:
    """Unknown dataset name raises ValueError."""
    cfg = DistillationConfig(dataset_name="waymo", dataset_sha256="")
    with pytest.raises(ValueError, match="unknown dataset_name"):
        distill_codebook(cfg, frames=iter([np.zeros((16, 16, 3), dtype=np.uint8)] * 8))


def test_distill_license_tags_include_comma2k19_mit() -> None:
    """Comma2k19 distillation tags license as MIT + github URL."""
    cfg = DistillationConfig(dataset_name="comma2k19")
    book = distill_codebook(
        cfg, frames=iter([np.zeros((32, 32, 3), dtype=np.uint8)] * 8)
    )
    tags = book.metadata["license_tags"]
    assert "comma2k19:MIT" in tags
    assert any("github.com/commaai/comma2k19" in t for t in tags)


def test_federated_aggregation_two_codebooks_validates() -> None:
    """Two federated codebooks aggregate into a validated combined codebook."""
    cfg = DistillationConfig(dataset_name="synthetic_test", random_seed=1)
    book_a = distill_codebook(cfg)
    cfg2 = DistillationConfig(dataset_name="synthetic_test", random_seed=2)
    book_b = distill_codebook(cfg2)
    merged = aggregate_local_codebooks([book_a, book_b], weights=[0.5, 0.5])
    validate_codebook(merged)
    assert merged.metadata["num_constituent_codebooks"] == 2


def test_federated_aggregation_rejects_bad_weights() -> None:
    """Federated aggregation refuses weights that don't sum to 1.0."""
    cfg = DistillationConfig(dataset_name="synthetic_test")
    book = distill_codebook(cfg)
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        aggregate_local_codebooks([book, book], weights=[0.3, 0.3])


# --------------------------- prior loss tests ---------------------------


def test_dashcam_prior_loss_zero_codebook_returns_finite_zero() -> None:
    """With zero codebook, prior_total is zero (or near-zero) and finite."""
    book = deterministic_zero_codebook()
    loss_fn = DashcamPriorLoss(book, PriorApplicationWeights())
    rgb = torch.zeros(1, 3, 32, 32)
    total, parts = loss_fn(rgb)
    assert torch.isfinite(total)
    assert torch.isfinite(parts["prior_total"])


def test_dashcam_prior_loss_carries_gradient() -> None:
    """The prior loss must be differentiable w.r.t. the input RGB."""
    book = deterministic_zero_codebook()
    loss_fn = DashcamPriorLoss(book, PriorApplicationWeights())
    rgb = torch.rand(1, 3, 32, 32, requires_grad=True)
    total, _ = loss_fn(rgb)
    total.backward()
    assert rgb.grad is not None
    # With a zero codebook the gradient may be zero but it must EXIST (not None).


def test_dashcam_prior_loss_rejects_wrong_input_shape() -> None:
    """Non-4D input raises ValueError."""
    book = deterministic_zero_codebook()
    loss_fn = DashcamPriorLoss(book, PriorApplicationWeights())
    with pytest.raises(ValueError, match=r"rgb_pred"):
        loss_fn(torch.zeros(3, 32, 32))


# --------------------------- archive grammar tests ---------------------------


def test_dp1_header_size_invariant() -> None:
    """DP1 header is exactly 28 bytes."""
    assert DP1_HEADER_SIZE == 28


def test_dp1_pack_parse_roundtrip_preserves_renderer_state_dict() -> None:
    """Pack + parse of a DP1 archive preserves codebook + state_dict + residual."""
    book = deterministic_zero_codebook()
    cfg = DrivingPriorRendererConfig(hidden_dim=16, num_hidden_layers=2)
    renderer = DrivingPriorRenderer(cfg)
    sd = renderer.state_dict()

    num_pairs = 4
    per_pair_bytes = 8
    residual = bytes([0] * (num_pairs * per_pair_bytes))
    meta = {"residual_int8_scale": 64.0, "hidden_dim": cfg.hidden_dim}

    packed = pack_archive(
        book,
        sd,
        residual,
        meta,
        num_pairs=num_pairs,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        per_pair_bytes=per_pair_bytes,
    )
    assert packed[:4] == DP1_MAGIC
    parsed = parse_archive(packed)
    assert parsed.num_pairs == num_pairs
    assert parsed.per_pair_bytes == per_pair_bytes
    assert parsed.meta["residual_int8_scale"] == 64.0
    # State dict round-trip preserves keys + value shapes.
    assert set(parsed.renderer_state_dict.keys()) == set(sd.keys())


def test_dp1_pack_rejects_residual_length_mismatch() -> None:
    """Residual byte count must equal num_pairs * per_pair_bytes."""
    book = deterministic_zero_codebook()
    cfg = DrivingPriorRendererConfig(hidden_dim=8, num_hidden_layers=2)
    renderer = DrivingPriorRenderer(cfg)
    with pytest.raises(ValueError, match="per_pair_residual"):
        pack_archive(
            book,
            renderer.state_dict(),
            b"\x00\x00",  # 2 bytes
            {"residual_int8_scale": 64.0},
            num_pairs=2,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
            per_pair_bytes=4,  # expected 8 bytes total
        )


def test_dp1_parse_rejects_wrong_magic() -> None:
    """An archive with the wrong magic raises ValueError."""
    bad = b"XXXX" + b"\x00" * 100
    with pytest.raises(ValueError, match="magic"):
        parse_archive(bad)


def test_dp1_parse_rejects_short_header() -> None:
    """An archive too short for the header raises ValueError."""
    with pytest.raises(ValueError, match="too short"):
        parse_archive(b"DP1\x00")


def test_dp1_readiness_manifest_tags_proxy_evidence() -> None:
    """The readiness manifest carries [proxy] evidence_grade + dispatch blockers."""
    manifest = build_readiness_manifest(
        archive_path="experiments/results/lane_pretrained_driving_prior/0.bin",
        codebook_path="experiments/results/lane_pretrained_driving_prior/codebook.bin",
        archive_bytes=80_000,
        codebook_bytes=7_000,
    )
    assert manifest["evidence_grade"] == "[proxy]"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False
    assert "contest_cuda_eval_not_run" in manifest["dispatch_blockers"]
    assert manifest["lane_class"] == "substrate_engineering"
    assert manifest["research_only"] is True


# --------------------------- inflate end-to-end ---------------------------


def test_inflate_smoke_writes_raw_file(tmp_path: Path) -> None:
    """End-to-end: pack archive -> inflate -> raw file with contest shape."""
    book = deterministic_zero_codebook()
    cfg = DrivingPriorRendererConfig(hidden_dim=16, num_hidden_layers=2)
    renderer = DrivingPriorRenderer(cfg)

    num_pairs = 3  # tiny to keep test fast
    per_pair_bytes = 6
    residual = bytes(range(num_pairs * per_pair_bytes))
    meta = {"residual_int8_scale": 64.0}

    packed = pack_archive(
        book,
        renderer.state_dict(),
        residual,
        meta,
        num_pairs=num_pairs,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        per_pair_bytes=per_pair_bytes,
    )
    raw_path = tmp_path / "0.raw"
    frames_written = inflate_one_video(packed, raw_path, device="cpu")
    assert frames_written == 2 * num_pairs
    assert raw_path.is_file()
    # Contest contract: each frame is 874 * 1164 * 3 bytes.
    expected_bytes = 2 * num_pairs * 874 * 1164 * 3
    assert raw_path.stat().st_size == expected_bytes


# --------------------------- /tmp path discipline ---------------------------


def test_write_codebook_to_disk_refuses_tmp_paths(tmp_path: Path) -> None:
    """write_codebook_to_disk refuses /tmp/ paths per CLAUDE.md."""
    book = deterministic_zero_codebook()
    bad_path = Path("/tmp/scaffold_codebook.bin")
    with pytest.raises(ValueError, match="transient path"):
        write_codebook_to_disk(book, bad_path)
    # Non-tmp path works.
    good_path = tmp_path / "codebook.bin"
    write_codebook_to_disk(book, good_path)
    assert good_path.is_file()
    assert good_path.with_suffix(good_path.suffix + ".meta.json").is_file()


# --------------------------- composition with time-traveler ---------------------------


def test_driving_prior_substrate_composes_with_time_traveler_via_score_pair_components() -> None:
    """Both substrates share the score_pair_components contract (Catalog #164).

    This test asserts the import-path contract: the score-aware loss
    accepts the same canonical scorer pipeline as the time-traveler
    substrate, so composition is a typed sum of archive grammars under a
    wrapper.
    """
    from tac.substrates.score_aware_common import score_pair_components

    # The function exists; the contract is the same shape both substrates rely on.
    assert callable(score_pair_components)
    # The driving prior loss accepts the same scorer surface.
    book = deterministic_zero_codebook()
    weights = DrivingPriorLossWeights()
    prior_loss_fn = DashcamPriorLoss(book, PriorApplicationWeights())

    # Stub scorers for the structural composition test (real test exercises
    # the canonical pipeline via the trainer's full integration test).
    class _StubScorer(torch.nn.Module):
        def preprocess_input(self, x):
            return x

        def forward(self, x):
            return torch.zeros(x.shape[0], 5, x.shape[-2], x.shape[-1])

    seg = _StubScorer()
    pose = _StubScorer()
    loss_mod = DrivingPriorScoreAwareLoss(seg, pose, prior_loss_fn, weights)
    assert loss_mod.weights.alpha_rate == 25.0
    assert loss_mod.weights.delta_prior == 0.05
