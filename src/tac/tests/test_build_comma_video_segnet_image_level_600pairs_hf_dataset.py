# SPDX-License-Identifier: MIT
"""Tests for tools/build_comma_video_segnet_image_level_600pairs_hf_dataset.py (Catalog #342)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# Module load helper — tool lives under tools/ (not on PYTHONPATH); import
# via importlib.util pattern per sister test conventions in the repo.
# --------------------------------------------------------------------------


def _load_tool_module():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "build_comma_video_segnet_image_level_600pairs_hf_dataset.py"
    mod_name = "build_comma_video_segnet_dataset"
    spec = importlib.util.spec_from_file_location(mod_name, tool_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so dataclasses can resolve
    # cls.__module__ during typing-annotation parsing (Python 3.12 behavior).
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder_module():
    return _load_tool_module()


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# Constants + schema invariants
# --------------------------------------------------------------------------


def test_constants_canonical(builder_module):
    """Canonical constants match contest reality."""
    assert builder_module.CONTEST_VIDEO_NPAIRS == 600
    assert builder_module.CONTEST_VIDEO_NFRAMES == 1200
    assert builder_module.CANONICAL_FRAME_HEIGHT == 384
    assert builder_module.CANONICAL_FRAME_WIDTH == 512
    assert builder_module.SEGNET_NUM_CLASSES == 5
    assert builder_module.DEFAULT_LICENSE_TAG == "MIT"
    assert builder_module.DEFAULT_HUB_REPO_ID == "adpena/comma-video-segnet-image-level-600pairs"
    assert builder_module.SCHEMA_VERSION.startswith("comma_video_segnet")


def test_schema_version_pinned(builder_module):
    """Schema version contains catalog # + date for forensic provenance."""
    assert "catalog342" in builder_module.SCHEMA_VERSION
    assert "20260519" in builder_module.SCHEMA_VERSION


# --------------------------------------------------------------------------
# Evidence grade mapping (Catalog #1 + #127 + #192 sister discipline)
# --------------------------------------------------------------------------


def test_device_to_evidence_grade_cpu(builder_module):
    """CPU = contest_cpu_authoritative per CLAUDE.md MPS-noise non-negotiable."""
    assert (
        builder_module._device_to_evidence_grade("cpu")
        == "contest_cpu_authoritative"
    )


def test_device_to_evidence_grade_mps(builder_module):
    """MPS = advisory/proxy per Catalog #1 + #192."""
    grade = builder_module._device_to_evidence_grade("mps")
    assert "advisory" in grade or "proxy" in grade


def test_device_to_evidence_grade_cuda(builder_module):
    assert (
        builder_module._device_to_evidence_grade("cuda")
        == "contest_cuda_authoritative"
    )


def test_device_to_evidence_grade_unknown(builder_module):
    """Unknown device returns explicit unknown tag (not silent default)."""
    grade = builder_module._device_to_evidence_grade("xpu_experimental")
    assert "unknown_device:xpu_experimental" == grade


# --------------------------------------------------------------------------
# sha256 helper
# --------------------------------------------------------------------------


def test_sha256_file_deterministic(builder_module, tmp_path):
    p = tmp_path / "test.txt"
    p.write_bytes(b"hello world")
    sha = builder_module._sha256_file(p)
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha == expected


# --------------------------------------------------------------------------
# Dry-run build (no decode / scorer / upload — sha-chain only)
# --------------------------------------------------------------------------


def test_dry_run_build_emits_canonical_summary(builder_module, repo_root):
    """Dry-run produces sha-chain summary without heavy compute."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")

    # Find any segnet weights variant — canonical under upstream/models/
    weights_path = None
    for cand in (
        "models/segnet.safetensors",
        "models/segnet.pt",
        "segnet.safetensors",
        "segnet.pt",
        "segnet_weights.pt",
    ):
        candidate = repo_root / "upstream" / cand
        if candidate.exists():
            weights_path = candidate
            break
    if weights_path is None:
        pytest.skip("No SegNet weights found under upstream/")

    summary = builder_module.build_dataset(
        video_path=video_path,
        segnet_weights_path=weights_path,
        dry_run=True,
        upload=False,
    )
    assert summary.dry_run is True
    assert summary.uploaded is False
    assert summary.hub_commit_sha is None
    assert summary.n_pairs == 600
    assert summary.n_frames == 1200
    assert len(summary.upstream_video_sha256) == 64  # sha256 hex
    assert len(summary.segnet_weights_sha256) == 64
    assert summary.evidence_grade == "contest_cpu_authoritative"


def test_dry_run_summary_serializable(builder_module, repo_root):
    """Summary serializes to canonical JSON (no non-JSON-safe fields)."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")

    summary = builder_module.build_dataset(
        video_path=video_path,
        dry_run=True,
        upload=False,
    )
    d = summary.to_dict()
    serialized = json.dumps(d, sort_keys=True)
    reparsed = json.loads(serialized)
    assert reparsed["n_pairs"] == 600
    assert reparsed["schema_version"] == builder_module.SCHEMA_VERSION


def test_dry_run_summary_includes_evidence_grade(builder_module, repo_root):
    """Canonical Provenance per Catalog #287/#323 — evidence grade always present."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    summary = builder_module.build_dataset(
        video_path=video_path,
        device="cpu",
        dry_run=True,
        upload=False,
    )
    assert summary.evidence_grade == "contest_cpu_authoritative"


# --------------------------------------------------------------------------
# Idempotency — same inputs produce same manifest_sha
# --------------------------------------------------------------------------


def test_manifest_sha_deterministic(builder_module, repo_root):
    """Re-running dry-run on same inputs produces same manifest_sha."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    s1 = builder_module.build_dataset(video_path=video_path, dry_run=True)
    s2 = builder_module.build_dataset(video_path=video_path, dry_run=True)
    assert s1.manifest_sha256 == s2.manifest_sha256
    assert s1.upstream_video_sha256 == s2.upstream_video_sha256


def test_manifest_sha_differs_by_device(builder_module, repo_root):
    """Different device → different manifest sha (different evidence chain)."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    s_cpu = builder_module.build_dataset(video_path=video_path, dry_run=True, device="cpu")
    s_mps = builder_module.build_dataset(video_path=video_path, dry_run=True, device="mps")
    assert s_cpu.manifest_sha256 != s_mps.manifest_sha256


# --------------------------------------------------------------------------
# Dataset card README
# --------------------------------------------------------------------------


def test_dataset_card_has_license_section(builder_module, repo_root):
    """Card includes MIT license + comma.ai attribution per CLAUDE.md Public Disclosure Hygiene."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    summary = builder_module.build_dataset(video_path=video_path, dry_run=True)
    card = builder_module.build_dataset_card(summary)
    assert "license: mit" in card.lower()
    assert "comma.ai" in card.lower()
    assert "## License" in card


def test_dataset_card_has_provenance_section(builder_module, repo_root):
    """Card includes sha256 chain per Catalog #287 / #323."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    summary = builder_module.build_dataset(video_path=video_path, dry_run=True)
    card = builder_module.build_dataset_card(summary)
    assert summary.upstream_video_sha256 in card
    assert summary.segnet_weights_sha256 in card
    assert summary.manifest_sha256 in card


def test_dataset_card_no_local_absolute_paths(builder_module, repo_root):
    """Card sanitizes local paths per Catalog #208."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    summary = builder_module.build_dataset(video_path=video_path, dry_run=True)
    card = builder_module.build_dataset_card(summary)
    # No machine-local home dir leakage
    assert "/Users/" not in card
    assert "/home/" not in card or "openpilot" in card.lower()  # github.com/commaai/openpilot is allowed


# --------------------------------------------------------------------------
# CLI smoke
# --------------------------------------------------------------------------


def test_cli_dry_run_smoke(builder_module, repo_root, tmp_path):
    """CLI --dry-run invocation produces canonical summary JSON."""
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    summary_out = tmp_path / "summary.json"
    rc = builder_module.main([
        "--dry-run",
        "--output-summary", str(summary_out),
        "--video-path", str(video_path),
    ])
    assert rc == 0
    assert summary_out.exists()
    payload = json.loads(summary_out.read_text())
    assert payload["n_pairs"] == 600
    assert payload["dry_run"] is True


def test_cli_rejects_missing_video(builder_module, tmp_path):
    """Missing video path raises FileNotFoundError (fail-loud)."""
    with pytest.raises(FileNotFoundError):
        builder_module.build_dataset(
            video_path=tmp_path / "nonexistent.mkv",
            dry_run=True,
        )
