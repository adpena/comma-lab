"""Tests for experiments/contest_auth_eval.py — the canonical generic
contest-compliant auth evaluator.

Covers parsing + the structural promises the tool makes:
  - Parses upstream/evaluate.py's report.txt format correctly
  - Score components match the reported final score
  - Refuses missing inflate.sh / archive / upstream
  - Records full provenance
"""
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def cae() -> ModuleType:
    """Load contest_auth_eval as a module without depending on it being
    installed (it's a top-level experiments/ script, not a package member)."""
    spec = importlib.util.spec_from_file_location(
        "contest_auth_eval", REPO / "experiments" / "contest_auth_eval.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads(cae):
    """Sanity: the module imports and exposes the expected entry points."""
    assert hasattr(cae, "main")
    assert hasattr(cae, "_parse_report")
    assert hasattr(cae, "_run_inflate")
    assert hasattr(cae, "_run_upstream_evaluate")
    assert cae.SCHEMA_VERSION == 1


def test_parse_report_baseline_format(cae, tmp_path: Path):
    """Parses the canonical upstream/evaluate.py report shape.

    Uses self-consistent values where the formula 100*seg + sqrt(10*pose) +
    25*rate exactly reproduces the reported final score. The historical
    "0.9001 baseline" run_log entry actually has a 0.108-gap between
    reported and recomputed — that's a separate baseline-provenance bug
    being investigated, NOT a parser bug. Council R3 #6 hardening
    correctly surfaces such formula divergences.
    """
    # Self-consistent: 100*0.002 + sqrt(10*0.0107) + 25*0.009 = 0.200 + 0.327 + 0.225 = 0.752
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.01070000
  Average SegNet Distortion: 0.00200000
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00900
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.7521
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    result = cae._parse_report(rp, archive_size=337748)

    assert result["schema_version"] == 1
    assert result["final_score"] == pytest.approx(0.7521, abs=1e-4)
    assert result["avg_posenet_dist"] == pytest.approx(0.0107, abs=1e-4)
    assert result["avg_segnet_dist"] == pytest.approx(0.0020, abs=1e-4)
    assert result["rate_unscaled"] == pytest.approx(0.009, abs=1e-4)
    assert result["score_seg_contribution"] == pytest.approx(0.20, abs=1e-3)
    assert result["score_pose_contribution"] == pytest.approx(0.327, abs=1e-3)
    assert result["score_rate_contribution"] == pytest.approx(0.225, abs=1e-3)
    assert result["n_samples"] == 600
    assert result["archive_size_bytes"] == 337748


def test_parse_report_rejects_score_formula_divergence(cae, tmp_path: Path):
    """Council R3 #6: if the reported final doesn't match the recomputed
    score within 0.01, raise loud — protects against upstream changing
    the formula weights (which would silently shift our scores)."""
    # 100*0.0024 + sqrt(10*0.0107) + 25*0.00899 = 0.24 + 0.327 + 0.2248 = 0.792
    # but reported as 0.9001 — formula divergence of 0.108 must raise.
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.0107
  Average SegNet Distortion: 0.0024
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00899
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.9001
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    with pytest.raises(RuntimeError, match="formula divergence"):
        cae._parse_report(rp, archive_size=337748)


def test_parse_report_rejects_nan(cae, tmp_path: Path):
    """Council R3 #5: NaN / inf parses as float() but is not a valid score."""
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: nan
  Average SegNet Distortion: 0.0024
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.009
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = nan
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    # NaN slips past the strict number regex (won't match) so the parser
    # raises "could not parse" — also acceptable. Either way: LOUD refusal.
    with pytest.raises(RuntimeError, match="non-finite|could not parse"):
        cae._parse_report(rp, archive_size=337748)


def test_parse_report_rejects_wrong_n_samples(cae, tmp_path: Path):
    """Council R3 #3 backstop: if the .raw byte-count check missed a partial
    inflate, the n_samples mismatch in the report catches it."""
    # 599 samples instead of 600 — partial inflate slipped through
    text = """=== Evaluation results over 599 samples ===
  Average PoseNet Distortion: 0.01
  Average SegNet Distortion: 0.002
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.009
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.74
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    with pytest.raises(RuntimeError, match="expected 600 samples"):
        cae._parse_report(rp, archive_size=337748)


def test_parse_report_score_recomputation_consistent(cae, tmp_path: Path):
    """The recomputed score from components should be CLOSE to the reported
    final (within a small tolerance for rounding in the source report)."""
    text = """=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.0107
  Average SegNet Distortion: 0.00240
  Submission file size: 337748 bytes
  Original uncompressed size: 37545489 bytes
  Compression Rate: 0.00899
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.7919
"""
    rp = tmp_path / "report.txt"
    rp.write_text(text)
    result = cae._parse_report(rp, archive_size=337748)

    recomputed = result["score_recomputed_from_components"]
    # Within 0.01 of the reported final
    assert abs(recomputed - result["final_score"]) < 0.01, (
        f"recomputed={recomputed:.4f} reported={result['final_score']:.4f}"
    )


def test_parse_report_malformed_raises(cae, tmp_path: Path):
    """A malformed report (missing required fields) must raise loudly."""
    rp = tmp_path / "bad.txt"
    rp.write_text("garbage with no fields")
    with pytest.raises(RuntimeError, match="could not parse"):
        cae._parse_report(rp, archive_size=1000)


def test_extract_archive_zip_slip_protection(cae, tmp_path: Path):
    """A malicious archive with `../` paths must NOT escape the dest dir."""
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr("../escaped.txt", b"pwned")
    with pytest.raises(RuntimeError, match="zip-slip"):
        cae._extract_archive(bad_zip, tmp_path / "dest")


def test_extract_archive_normal(cae, tmp_path: Path):
    """Normal archive extracts cleanly."""
    src = tmp_path / "good.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("renderer.bin", b"fake renderer")
        z.writestr("masks.mkv", b"fake masks")
    members = cae._extract_archive(src, tmp_path / "dest")
    assert sorted(members) == ["masks.mkv", "renderer.bin"]
    assert (tmp_path / "dest" / "renderer.bin").exists()
    assert (tmp_path / "dest" / "masks.mkv").exists()


def test_main_refuses_missing_archive(cae, tmp_path: Path):
    """--archive must point to an existing file."""
    sys.argv = [
        "contest_auth_eval.py",
        "--archive", str(tmp_path / "nonexistent.zip"),
        "--inflate-sh", str(REPO / "submissions" / "robust_current" / "inflate.sh"),
        "--upstream-dir", str(REPO / "upstream"),
    ]
    with pytest.raises(SystemExit, match="--archive does not exist"):
        cae.main()


def test_main_refuses_missing_inflate(cae, tmp_path: Path):
    """--inflate-sh must point to an existing file."""
    fake_archive = tmp_path / "fake.zip"
    fake_archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip eocd
    sys.argv = [
        "contest_auth_eval.py",
        "--archive", str(fake_archive),
        "--inflate-sh", str(tmp_path / "nonexistent.sh"),
        "--upstream-dir", str(REPO / "upstream"),
    ]
    with pytest.raises(SystemExit, match="--inflate-sh does not exist"):
        cae.main()


def test_main_refuses_missing_upstream(cae, tmp_path: Path):
    """--upstream-dir must contain evaluate.py."""
    fake_archive = tmp_path / "fake.zip"
    fake_archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    fake_inflate = tmp_path / "inflate.sh"
    fake_inflate.write_text("#!/bin/bash\necho ok\n")
    fake_inflate.chmod(0o755)
    sys.argv = [
        "contest_auth_eval.py",
        "--archive", str(fake_archive),
        "--inflate-sh", str(fake_inflate),
        "--upstream-dir", str(tmp_path),
    ]
    with pytest.raises(SystemExit, match="missing evaluate.py"):
        cae.main()


def test_help_includes_canonical_one_liner(cae):
    """The docstring should make the canonical contest flow obvious."""
    doc = cae.__doc__
    assert doc is not None
    assert "archive.zip" in doc
    assert "inflate.sh" in doc
    assert "upstream/evaluate.py" in doc
    assert "CANONICAL" in doc
