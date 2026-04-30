"""Tests for Check 96 — Lane PFP16 pose-stream fp16-or-smaller discipline.

Verifies the regression behaviour of `check_pose_stream_uses_fp16_or_smaller`:

* The live codebase passes with 0 violations (smoke test).
* A synthetic violation (torch.save on poses without canonical encoder) is
  detected.
* The `# POSE_FP32_REQUIRED:<reason>` waiver suppresses the violation.
* Files using `encode_pfp16` / `save_poses_binary` / `encode_pose_deltas` /
  `encode_pose_delta_v2` / `encode_lora_*` are not flagged.
* Build scripts that don't touch poses are not flagged.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_pose_stream_uses_fp16_or_smaller,
)


# ───────────────────────────────────────────────────────────────────────────
# Live-codebase smoke (must always pass)
# ───────────────────────────────────────────────────────────────────────────


def test_live_codebase_passes_strict():
    """Strict-mode scan against the real repo must report 0 violations.

    If THIS test fails, a new archive build script slipped in shipping
    fp32 pose tensors. Add `encode_pfp16` / `save_poses_binary` /
    `encode_pose_deltas` to the encode path, OR add a
    `# POSE_FP32_REQUIRED:<reason>` waiver if fp32 is genuinely required.
    """
    violations = check_pose_stream_uses_fp16_or_smaller(strict=False, verbose=False)
    assert violations == [], (
        "Lane PFP16 fp16-or-smaller discipline regressed:\n  "
        + "\n  ".join(violations)
    )
    # Re-confirm strict raises nothing (defense-in-depth: a future check
    # change might re-classify violations).
    check_pose_stream_uses_fp16_or_smaller(strict=True, verbose=False)


# ───────────────────────────────────────────────────────────────────────────
# Synthetic violation detection
# ───────────────────────────────────────────────────────────────────────────


def _make_repo(tmp_path: Path, build_files: dict[str, str]) -> Path:
    """Create a minimal repo skeleton for testing Check 96.

    Returns the repo root path.
    """
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    for name, content in build_files.items():
        (repo / "experiments" / name).write_text(content)
    return repo


def test_detects_torch_save_on_poses_without_encoder(tmp_path):
    """A build script calling `torch.save(poses, ...)` without a canonical
    encoder MUST be flagged.
    """
    repo = _make_repo(tmp_path, {
        "build_lane_synthetic_stack.py": """
import torch
poses = torch.randn(600, 6)
# Ship raw fp32 pickle - WRONG.
torch.save(poses, "optimized_poses.pt")
""",
    })
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "build_lane_synthetic_stack.py" in violations[0]
    assert "encode_pfp16" in violations[0] or "canonical pose-encode" in violations[0]


def test_strict_raises_on_violation(tmp_path):
    """Strict mode raises PreflightError on any violation."""
    repo = _make_repo(tmp_path, {
        "build_lane_bad_stack.py": """
import torch
poses = torch.randn(10, 6)
torch.save(poses, "/tmp/poses.pt")
""",
    })
    with pytest.raises(PreflightError, match="POSE-STREAM FP16 VIOLATIONS"):
        check_pose_stream_uses_fp16_or_smaller(
            repo_root=repo, strict=True, verbose=False,
        )


# ───────────────────────────────────────────────────────────────────────────
# Waiver / canonical-encoder suppression
# ───────────────────────────────────────────────────────────────────────────


def test_waiver_marker_suppresses_violation(tmp_path):
    """The `# POSE_FP32_REQUIRED:<reason>` marker silences the check."""
    repo = _make_repo(tmp_path, {
        "build_lane_waived_stack.py": """
# POSE_FP32_REQUIRED: this build is for a debug variant where fp32
# precision is required to debug a numerical issue.
import torch
poses = torch.randn(10, 6)
torch.save(poses, "/tmp/poses.pt")
""",
    })
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_encode_pfp16_call_suppresses_violation(tmp_path):
    """A build script that uses `encode_pfp16` is compliant."""
    repo = _make_repo(tmp_path, {
        "build_lane_pfp16_stack.py": """
import torch
from tac.pfp16_codec import encode_pfp16

poses = torch.randn(600, 6)
raw = encode_pfp16(poses)
# Even if the file ALSO calls torch.save on something pose-named, the
# encode function presence exempts the file (it's the canonical path).
torch.save(poses, "/tmp/poses.pt")  # would normally trigger
""",
    })
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_save_poses_binary_call_suppresses_violation(tmp_path):
    """`save_poses_binary` is canonical too."""
    repo = _make_repo(tmp_path, {
        "build_lane_savebin_stack.py": """
import torch
from tac.submission_archive import save_poses_binary
poses = torch.randn(600, 6)
save_poses_binary(poses, "/tmp/poses.bin")
torch.save(poses, "/tmp/poses_dbg.pt")  # would normally trigger
""",
    })
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_encode_pose_deltas_suppresses_violation(tmp_path):
    """Lane PD's `encode_pose_deltas` is also canonical."""
    repo = _make_repo(tmp_path, {
        "build_lane_pd_stack.py": """
import torch
from tac.pose_delta_codec import encode_pose_deltas
poses = torch.randn(600, 6)
encoded = encode_pose_deltas(poses)
torch.save(encoded, "/tmp/poses_pd.pt")  # this is the encoded dict, not raw fp32
""",
    })
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_encode_pose_delta_v2_suppresses_violation(tmp_path):
    """Lane PD-V2's `encode_pose_delta_v2` is also canonical."""
    repo = _make_repo(tmp_path, {
        "build_lane_pdv2_stack.py": """
import torch
from tac.pose_delta_codec_v2 import encode_pose_delta_v2
poses = torch.randn(600, 6)
blob = encode_pose_delta_v2(poses)
torch.save({"format": "pose_delta_v2", "blob": blob}, "/tmp/poses_pdv2.pt")
""",
    })
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


# ───────────────────────────────────────────────────────────────────────────
# Negative cases — files that MUST NOT be flagged
# ───────────────────────────────────────────────────────────────────────────


def test_non_pose_build_script_not_flagged(tmp_path):
    """A build script that doesn't mention poses at all is exempt."""
    repo = _make_repo(tmp_path, {
        "build_lane_renderer_only_stack.py": """
import torch
weights = torch.randn(100, 100)
torch.save(weights, "/tmp/renderer.bin")
""",
    })
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_byte_copy_pose_artifact_not_flagged(tmp_path):
    """A build script that copies bytes from a pre-built pose artifact
    (no torch.save on a tensor) is not flagged. This is the pattern
    `experiments/build_lane_g_v3_omega_w_v2_stack.py` uses — it reads
    Lane G v3 poses_bytes via Path.read_bytes() and writes them
    bit-identically into a new ZipFile entry.
    """
    repo = _make_repo(tmp_path, {
        "build_lane_byte_copy_stack.py": """
from pathlib import Path
import zipfile
poses_bytes = Path("anchor/optimized_poses.pt").read_bytes()
with zipfile.ZipFile("out.zip", "w") as z:
    z.writestr("optimized_poses.pt", poses_bytes)
""",
    })
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_no_candidate_files_returns_empty(tmp_path):
    """If experiments/ has no build_*_stack.py / build_*_archive.py /
    build_lane_*.py files, the check is a no-op.
    """
    repo = tmp_path / "empty_repo"
    (repo / "experiments").mkdir(parents=True)
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=True, verbose=False,
    )
    assert violations == []


def test_no_experiments_dir_returns_empty(tmp_path):
    """If experiments/ doesn't exist, the check is a no-op."""
    repo = tmp_path / "no_experiments"
    repo.mkdir()
    violations = check_pose_stream_uses_fp16_or_smaller(
        repo_root=repo, strict=True, verbose=False,
    )
    assert violations == []
