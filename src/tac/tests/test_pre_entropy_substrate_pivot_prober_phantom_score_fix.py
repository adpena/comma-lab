# SPDX-License-Identifier: MIT
"""Catalog #321 + Q4 HALT 2026-05-17: phantom-score-from-research-sidecar
extinction tests for `tools/pre_entropy_substrate_pivot_prober.py`.

Covers:
* `_validate_substrate_bytes_ship_in_contest_archive` correctness on
  research-sidecar paths (.pt / .npy / .npz / .pth / .pkl / .bin)
  → REJECTED with reason.
* `_validate_substrate_bytes_ship_in_contest_archive` correctness on
  real contest archive.zip paths under `submissions/` or
  `experiments/results/` → VALIDATED.
* `probe_substrate_archive_member` correctness on synthetic archive.zip
  fixture: extracts the specific member, runs compression probe ON THE
  MEMBER (the apples-to-apples level of analysis per Q4 Option B).
* Refactored `probe_substrate` emits REJECTED_RESEARCH_SIDECAR for the
  empirical pr101_decoder_state_dict.pt regression anchor.
* Refactored `probe_substrate` accepts `skip_contest_member_validation=
  True` opt-in for synthetic test fixtures.
* Sister: contest archive.zip under canonical root is VALIDATED.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import pre_entropy_substrate_pivot_prober as pivot  # noqa: E402


# ──────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                  #
# ──────────────────────────────────────────────────────────────────────── #


def _synthetic_fp32_weights(num_floats: int = 2048, seed: int = 1) -> bytes:
    import random

    rng = random.Random(seed)
    vocab = [0.0, 0.0, 0.0, 0.01, -0.01, 0.05]
    return b"".join(struct.pack("<f", rng.choice(vocab)) for _ in range(num_floats))


def _create_canonical_root_zip(repo_root: Path, sub_root: str, name: str, members: dict[str, bytes]) -> Path:
    """Create a .zip under a canonical contest-shipping root for validator
    acceptance tests. `sub_root` is e.g. ``'experiments/results/test_q4_fixture'``.
    """
    out_dir = repo_root / sub_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    with zipfile.ZipFile(out, "w") as zf:
        for mn, mb in members.items():
            zf.writestr(mn, mb)
    return out


# ──────────────────────────────────────────────────────────────────────── #
# Test: validator correctness on research-sidecar extensions                #
# ──────────────────────────────────────────────────────────────────────── #


def test_validator_rejects_pt_extension_anywhere(tmp_path: Path) -> None:
    """A .pt file at any path is REJECTED — torch sidecars never ship as
    standalone contest archive members."""
    f = tmp_path / "weights.pt"
    f.write_bytes(_synthetic_fp32_weights(num_floats=512))
    is_valid, reason = pivot._validate_substrate_bytes_ship_in_contest_archive("test_sub", str(f))
    assert is_valid is False
    assert reason is not None
    assert ".pt" in reason or "sidecar" in reason


@pytest.mark.parametrize("ext", [".pt", ".npy", ".npz", ".pth", ".pkl", ".bin"])
def test_validator_rejects_all_research_sidecar_extensions(tmp_path: Path, ext: str) -> None:
    f = tmp_path / f"sidecar{ext}"
    f.write_bytes(b"\x00" * 64)
    is_valid, reason = pivot._validate_substrate_bytes_ship_in_contest_archive("test_sub", str(f))
    assert is_valid is False
    assert ext in reason  # type: ignore[operator]


def test_validator_rejects_off_root_zip(tmp_path: Path) -> None:
    """A .zip OUTSIDE submissions/ or experiments/results/ is rejected
    even if it's a valid zipfile — contest archives must be under canonical
    roots."""
    f = tmp_path / "off_root.zip"
    with zipfile.ZipFile(f, "w") as zf:
        zf.writestr("x", b"hello")
    is_valid, reason = pivot._validate_substrate_bytes_ship_in_contest_archive("test_sub", str(f))
    assert is_valid is False
    assert "root" in (reason or "").lower()


def test_validator_accepts_canonical_root_zip() -> None:
    """A real contest archive.zip under submissions/ or experiments/results/
    is VALIDATED."""
    # Use the FEC6 frontier archive (real, in-repo)
    path = "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
    if not Path(path).exists():
        pytest.skip(f"live archive missing: {path}")
    is_valid, reason = pivot._validate_substrate_bytes_ship_in_contest_archive("pr101_fec6", path)
    assert is_valid is True
    assert reason is None


def test_validator_accepts_absolute_canonical_root_zip() -> None:
    """Absolute paths under the repo's canonical archive roots are VALIDATED.

    Regression: the Catalog #321 validator accepted the relative FEC6 archive
    path but rejected the same file after ``Path.resolve()``, which made the
    custody verdict depend on CLI call-site path style.
    """
    path = Path(
        "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
    )
    if not path.exists():
        pytest.skip(f"live archive missing: {path}")
    is_valid, reason = pivot._validate_substrate_bytes_ship_in_contest_archive(
        "pr101_fec6", str(path.resolve())
    )
    assert is_valid is True
    assert reason is None


def test_validator_rejects_nonexistent_path(tmp_path: Path) -> None:
    is_valid, reason = pivot._validate_substrate_bytes_ship_in_contest_archive(
        "missing_sub", str(tmp_path / "nope.pt")
    )
    assert is_valid is False
    assert "does not exist" in (reason or "")


def test_validator_rejects_empty_substrate_name() -> None:
    is_valid, reason = pivot._validate_substrate_bytes_ship_in_contest_archive("", "/some/path")
    assert is_valid is False
    assert "empty" in (reason or "")


def test_validator_rejects_corrupt_zip_under_canonical_root(tmp_path: Path) -> None:
    """A .zip under canonical root that fails zipfile.is_zipfile validation
    is rejected (truncated / corrupt)."""
    # Use a temp dir mounted as if under experiments/results/
    root_dir = REPO_ROOT / "experiments" / "results" / "test_corrupt_zip_fixture_321"
    root_dir.mkdir(parents=True, exist_ok=True)
    try:
        corrupt = root_dir / "archive.zip"
        corrupt.write_bytes(b"\x00\x01\x02\x03 not a zip")
        is_valid, reason = pivot._validate_substrate_bytes_ship_in_contest_archive(
            "corrupt_test", str(corrupt.relative_to(REPO_ROOT))
        )
        assert is_valid is False
        assert "is_zipfile" in (reason or "") or "zipfile" in (reason or "")
    finally:
        # cleanup
        if (root_dir / "archive.zip").exists():
            (root_dir / "archive.zip").unlink()
        if root_dir.exists():
            root_dir.rmdir()


# ──────────────────────────────────────────────────────────────────────── #
# Test: probe_substrate_archive_member (Catalog #321 Option B method)       #
# ──────────────────────────────────────────────────────────────────────── #


def test_probe_archive_member_synthetic_pre_entropy() -> None:
    """Create a contest-shaped archive.zip under experiments/results/ with a
    compressible member; probe_substrate_archive_member returns VALIDATED
    + non-zero deliverable."""
    fixture_dir = REPO_ROOT / "experiments" / "results" / "test_321_synth_pre_entropy_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    archive_path = fixture_dir / "archive.zip"
    try:
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("x", _synthetic_fp32_weights(num_floats=8192))

        rel = str(archive_path.relative_to(REPO_ROOT))
        result = pivot.probe_substrate_archive_member(
            substrate_name="synth_pre",
            archive_zip_path=rel,
            member_name="x",
            substrate_class="raw_float_weights",
        )
        assert result.validation_status == "VALIDATED_CONTEST_MEMBER"
        assert result.member_count == 1
        assert result.member_results[0].member_name == "x"
        # Pre-entropy synthetic weights compress, so deliverable > 0
        assert result.deliverable_score_savings_estimate > 0
        assert result.archive_path.endswith("#x")
    finally:
        if archive_path.exists():
            archive_path.unlink()
        if fixture_dir.exists():
            fixture_dir.rmdir()


def test_probe_archive_member_missing_member() -> None:
    """Member name not present in the archive → error field set, but the
    validation_status remains VALIDATED (the archive itself is valid)."""
    fixture_dir = REPO_ROOT / "experiments" / "results" / "test_321_missing_member_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    archive_path = fixture_dir / "archive.zip"
    try:
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("a", b"\x00" * 4096)
        rel = str(archive_path.relative_to(REPO_ROOT))
        result = pivot.probe_substrate_archive_member(
            substrate_name="synth_missing",
            archive_zip_path=rel,
            member_name="does_not_exist",
            substrate_class="raw_float_weights",
        )
        assert result.validation_status == "VALIDATED_CONTEST_MEMBER"
        assert result.error is not None
        assert "not found" in result.error
        assert result.deliverable_score_savings_estimate == 0.0
    finally:
        if archive_path.exists():
            archive_path.unlink()
        if fixture_dir.exists():
            fixture_dir.rmdir()


def test_probe_archive_member_rejects_research_sidecar_path(tmp_path: Path) -> None:
    """Passing a .pt path to probe_substrate_archive_member → REJECTED."""
    f = tmp_path / "x.pt"
    f.write_bytes(b"\x00" * 4096)
    result = pivot.probe_substrate_archive_member(
        substrate_name="bad",
        archive_zip_path=str(f),
        member_name="anything",
        substrate_class="raw_float_weights",
    )
    assert result.validation_status == "REJECTED_RESEARCH_SIDECAR"
    assert result.evidence_grade_per_row == "invalid_target"
    assert result.deliverable_score_savings_estimate == 0.0


def test_probe_archive_member_live_fec6_x_at_floor() -> None:
    """The FEC6 archive's `x` member is at Shannon entropy floor per the
    sister prober — probe_substrate_archive_member returns VALIDATED with
    deliverable=0.0 (compression ratio ~1.0)."""
    path = "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
    if not Path(path).exists():
        pytest.skip(f"live archive missing: {path}")
    result = pivot.probe_substrate_archive_member(
        substrate_name="pr101_fec6_member_x",
        archive_zip_path=path,
        member_name="x",
        substrate_class="post_entropy_contest_archive",
    )
    assert result.validation_status == "VALIDATED_CONTEST_MEMBER"
    assert result.deliverable_score_savings_estimate == 0.0


def test_probe_archive_member_accepts_absolute_fec6_path() -> None:
    """Absolute canonical archive paths probe the same member-level target."""
    path = Path(
        "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
    )
    if not path.exists():
        pytest.skip(f"live archive missing: {path}")
    result = pivot.probe_substrate_archive_member(
        substrate_name="pr101_fec6_member_x_abs",
        archive_zip_path=str(path.resolve()),
        member_name="x",
        substrate_class="post_entropy_contest_archive",
    )
    assert result.validation_status == "VALIDATED_CONTEST_MEMBER"
    assert result.deliverable_score_savings_estimate == 0.0


def test_probe_substrate_accepts_absolute_canonical_archive_path() -> None:
    """Whole-archive probes also preserve absolute canonical path custody."""
    path = Path(
        "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
    )
    if not path.exists():
        pytest.skip(f"live archive missing: {path}")
    result = pivot.probe_substrate(
        substrate_name="pr101_fec6_abs",
        archive_path_str=str(path.resolve()),
        substrate_class="post_entropy_contest_archive",
    )
    assert result.validation_status == "VALIDATED_CONTEST_MEMBER"
    assert result.archive_exists is True


# ──────────────────────────────────────────────────────────────────────── #
# Test: probe_substrate REJECTS the empirical pr101_decoder_state_dict.pt   #
# ──────────────────────────────────────────────────────────────────────── #


def test_probe_substrate_rejects_pr101_decoder_state_dict_regression_anchor() -> None:
    """REGRESSION (Catalog #321 / Q4 HALT 2026-05-17): the empirical anchor
    file pr101_decoder_state_dict.pt is a 903 KB standalone torch sidecar
    NOT in any contest archive.zip. The Q4 BUILD subagent's HALT triggered
    on this file. The fixed prober MUST emit REJECTED_RESEARCH_SIDECAR with
    deliverable=0.0 — pre-fix it returned phantom 0.477 savings."""
    path = "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
    if not Path(path).exists():
        pytest.skip(f"live anchor missing: {path}")
    result = pivot.probe_substrate(
        substrate_name="pr101_state_dict",
        archive_path_str=path,
        substrate_class="raw_float_weights",
    )
    assert result.validation_status == "REJECTED_RESEARCH_SIDECAR"
    assert result.evidence_grade_per_row == "invalid_target"
    assert result.deliverable_score_savings_estimate == 0.0
    assert "sidecar" in (result.validation_reason or "").lower()
    # SHA + size are still preserved for provenance
    assert result.archive_sha256 is not None
    assert result.archive_bytes_total > 0


def test_probe_substrate_skip_validation_diagnostic_path(tmp_path: Path) -> None:
    """Synthetic fixture with skip_contest_member_validation=True flows
    through the regular probe path — validation_status remains
    UNVALIDATED (no validator ran), but deliverable is computed."""
    weights_path = tmp_path / "synth.pt"
    weights_path.write_bytes(_synthetic_fp32_weights(num_floats=2048))
    result = pivot.probe_substrate(
        substrate_name="synth_diag",
        archive_path_str=str(weights_path),
        substrate_class="raw_float_weights",
        skip_contest_member_validation=True,
    )
    assert result.validation_status == "UNVALIDATED"
    assert result.evidence_grade_per_row == "predicted"
    assert result.pre_entropy_bytes > 0
    assert result.deliverable_score_savings_estimate > 0


# ──────────────────────────────────────────────────────────────────────── #
# Test: end-to-end via run_pivot_probe with mixed canonical / sidecar       #
# ──────────────────────────────────────────────────────────────────────── #


def test_run_pivot_probe_mixed_canonical_and_sidecar(tmp_path: Path) -> None:
    """End-to-end: run pivot probe with one valid contest archive and one
    research sidecar — verify the manifest's recommended_q4_target_substrate
    is NOT the phantom-scoring sidecar."""
    # Real contest archive
    fec6 = "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
    if not Path(fec6).exists():
        pytest.skip("live FEC6 archive missing")
    sidecar = tmp_path / "phantom_sidecar.pt"
    sidecar.write_bytes(_synthetic_fp32_weights(num_floats=20000))

    candidates = {
        "fec6_valid": (fec6, "post_entropy_contest_archive"),
        "phantom_sidecar": (str(sidecar), "raw_float_weights"),
    }
    output_path = tmp_path / "manifest.json"
    results, persisted = pivot.run_pivot_probe(
        candidate_substrates=candidates,
        output_path=output_path,
        persist=True,
    )
    payload = json.loads(persisted.read_text())
    # The sidecar must be REJECTED, the FEC6 VALIDATED
    sidecar_row = payload["per_substrate_results"]["phantom_sidecar"]
    fec6_row = payload["per_substrate_results"]["fec6_valid"]
    assert sidecar_row.get("validation_status") == "REJECTED_RESEARCH_SIDECAR"
    assert sidecar_row["deliverable_score_savings_estimate"] == 0.0
    assert fec6_row.get("validation_status") == "VALIDATED_CONTEST_MEMBER"
    # The Q4 recommendation must NOT pick the phantom sidecar
    assert payload["recommended_q4_target_substrate"] != "phantom_sidecar"
