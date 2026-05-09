"""Unit tests for tac.paradigm_delta_epsilon_zeta.frozen_a1_encoder."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch

from tac.paradigm_delta_epsilon_zeta.frozen_a1_encoder import (
    A1_CANONICAL_DIR_NAME,
    CANONICAL_DESIGNATION_PATH,
    FrozenA1Encoder,
    FrozenA1EncoderError,
    _parse_designation_memo,
    load_frozen_a1_encoder,
)


def _make_fake_a1_canonical(
    tmp_path: Path,
    *,
    archive_bytes: bytes = b"FAKE_ARCHIVE_FOR_TEST",
    latents: torch.Tensor | None = None,
    decoder_sd: dict[str, torch.Tensor] | None = None,
) -> tuple[Path, str, int]:
    """Create a fake A1 canonical layout under tmp_path. Returns (repo_root, sha, size)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    canonical = repo / "experiments" / "results" / A1_CANONICAL_DIR_NAME
    (canonical / "finetuned_archive").mkdir(parents=True)
    (canonical / "train").mkdir(parents=True)
    archive_path = canonical / "finetuned_archive" / "archive.zip"
    archive_path.write_bytes(archive_bytes)
    sha = hashlib.sha256(archive_bytes).hexdigest()
    size = len(archive_bytes)

    latents = latents if latents is not None else torch.randn(600, 28)
    decoder_sd = decoder_sd or {"linear.weight": torch.randn(8, 8)}
    state = {"latents": latents.detach().clone(), "decoder": decoder_sd}
    torch.save(state, canonical / "train" / "checkpoint_best_proxy.pt")

    # Designation memo with json fenced block.
    omx_state = repo / ".omx" / "state"
    omx_state.mkdir(parents=True)
    contract = {
        "canonical_lane_id": "track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex",
        "archive_sha256": sha,
        "archive_size_bytes": size,
        "contest_cuda_score": 0.226352,
    }
    memo = (
        "# Canonical A1 designation\n\nDesignation contract:\n\n"
        "```json\n" + json.dumps(contract, indent=2) + "\n```\n"
    )
    (repo / CANONICAL_DESIGNATION_PATH).write_text(memo)
    return repo, sha, size


def test_load_canonical_a1_round_trip(tmp_path):
    repo, sha, size = _make_fake_a1_canonical(tmp_path)
    encoder = load_frozen_a1_encoder(repo_root=repo)
    assert encoder.archive_sha256 == sha
    assert encoder.archive_size_bytes == size
    assert encoder.n_pairs == 600
    assert encoder.latent_dim == 28
    assert encoder.contest_cuda_score == 0.226352


def test_load_refuses_missing_canonical_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(FrozenA1EncoderError, match="canonical A1 directory"):
        load_frozen_a1_encoder(repo_root=repo)


def test_load_refuses_missing_designation_memo(tmp_path):
    repo, _, _ = _make_fake_a1_canonical(tmp_path)
    (repo / CANONICAL_DESIGNATION_PATH).unlink()
    with pytest.raises(FrozenA1EncoderError, match="designation memo not found"):
        load_frozen_a1_encoder(repo_root=repo)


def test_load_refuses_sha_mismatch(tmp_path):
    repo, sha, size = _make_fake_a1_canonical(tmp_path)
    # Mutate the archive after the memo has been written.
    archive_path = repo / "experiments" / "results" / A1_CANONICAL_DIR_NAME / "finetuned_archive" / "archive.zip"
    archive_path.write_bytes(b"DIFFERENT BYTES")
    with pytest.raises(FrozenA1EncoderError, match="archive sha mismatch"):
        load_frozen_a1_encoder(repo_root=repo, strict_sha_check=True)


def test_load_allows_sha_mismatch_when_strict_check_disabled(tmp_path):
    repo, _, _ = _make_fake_a1_canonical(tmp_path)
    archive_path = repo / "experiments" / "results" / A1_CANONICAL_DIR_NAME / "finetuned_archive" / "archive.zip"
    archive_path.write_bytes(b"DIFFERENT BYTES")
    encoder = load_frozen_a1_encoder(repo_root=repo, strict_sha_check=False)
    assert encoder.archive_size_bytes == len(b"DIFFERENT BYTES")


def test_load_refuses_missing_archive(tmp_path):
    repo, _, _ = _make_fake_a1_canonical(tmp_path)
    archive = repo / "experiments" / "results" / A1_CANONICAL_DIR_NAME / "finetuned_archive" / "archive.zip"
    archive.unlink()
    with pytest.raises(FrozenA1EncoderError, match="no archive.zip"):
        load_frozen_a1_encoder(repo_root=repo)


def test_load_refuses_missing_checkpoint(tmp_path):
    repo, _, _ = _make_fake_a1_canonical(tmp_path)
    ckpt = repo / "experiments" / "results" / A1_CANONICAL_DIR_NAME / "train" / "checkpoint_best_proxy.pt"
    ckpt.unlink()
    with pytest.raises(FrozenA1EncoderError, match="no checkpoint"):
        load_frozen_a1_encoder(repo_root=repo)


def test_frozen_encoder_refuses_grad_on_construction():
    latents = torch.randn(4, 8, requires_grad=True)
    with pytest.raises(FrozenA1EncoderError, match="must have requires_grad=False"):
        FrozenA1Encoder(
            latents=latents,
            decoder_state_dict={},
            archive_sha256="x",
            archive_size_bytes=0,
            contest_cuda_score=None,
        )


def test_frozen_encoder_to_returns_new_view(tmp_path):
    repo, _, _ = _make_fake_a1_canonical(tmp_path)
    encoder = load_frozen_a1_encoder(repo_root=repo)
    moved = encoder.to("cpu")
    assert moved is not encoder
    assert moved.latents.requires_grad is False
    assert moved.archive_sha256 == encoder.archive_sha256


def test_designation_memo_parser_rejects_no_fenced_block(tmp_path):
    memo = tmp_path / "memo.md"
    memo.write_text("# No JSON block here\n")
    with pytest.raises(FrozenA1EncoderError, match="no.*fenced block"):
        _parse_designation_memo(memo)


def test_designation_memo_parser_rejects_invalid_json(tmp_path):
    memo = tmp_path / "memo.md"
    memo.write_text("# bad\n```json\n{not valid\n```\n")
    with pytest.raises(FrozenA1EncoderError, match="invalid JSON"):
        _parse_designation_memo(memo)


def test_designation_memo_parser_rejects_missing_keys(tmp_path):
    memo = tmp_path / "memo.md"
    memo.write_text("# missing\n```json\n{\"canonical_lane_id\": \"x\"}\n```\n")
    with pytest.raises(FrozenA1EncoderError, match="missing required keys"):
        _parse_designation_memo(memo)


def test_provenance_records_paths_and_hashes(tmp_path):
    repo, sha, _ = _make_fake_a1_canonical(tmp_path)
    encoder = load_frozen_a1_encoder(repo_root=repo)
    prov = encoder.provenance
    assert prov["loaded_archive_sha256"] == sha
    assert "canonical_dir" in prov["loaded_from"]
    assert "designation_memo" in prov["loaded_from"]
    assert prov["latents_shape"] == [600, 28]
    assert prov["decoder_param_count"] == 64  # 8x8 weight tensor


def test_load_handles_plain_state_dict_layout(tmp_path):
    """A1 checkpoints may be flat dicts without 'latents'/'decoder' keys."""
    repo = tmp_path / "repo"
    repo.mkdir()
    canonical = repo / "experiments" / "results" / A1_CANONICAL_DIR_NAME
    (canonical / "finetuned_archive").mkdir(parents=True)
    (canonical / "train").mkdir(parents=True)
    archive_bytes = b"PLAIN_LAYOUT_TEST"
    (canonical / "finetuned_archive" / "archive.zip").write_bytes(archive_bytes)

    sha = hashlib.sha256(archive_bytes).hexdigest()
    flat_state = {
        "latents": torch.randn(10, 12),
        "stem.weight": torch.randn(8, 12),
        "stem.bias": torch.randn(8),
    }
    torch.save(flat_state, canonical / "train" / "checkpoint_best_proxy.pt")

    omx_state = repo / ".omx" / "state"
    omx_state.mkdir(parents=True)
    contract = {
        "canonical_lane_id": "fake_lane",
        "archive_sha256": sha,
        "archive_size_bytes": len(archive_bytes),
    }
    (repo / CANONICAL_DESIGNATION_PATH).write_text(
        f"# fake\n```json\n{json.dumps(contract)}\n```\n"
    )
    encoder = load_frozen_a1_encoder(repo_root=repo)
    assert encoder.n_pairs == 10
    assert encoder.latent_dim == 12
    assert "stem.weight" in encoder.decoder_state_dict
