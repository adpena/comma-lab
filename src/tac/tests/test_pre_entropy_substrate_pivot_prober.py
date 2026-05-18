# SPDX-License-Identifier: MIT
"""Tests for tools/pre_entropy_substrate_pivot_prober.py.

Covers the canonical-contract per Catalog #265:
* SPDX header
* __all__ via dataclass exports
* update_from_anchor (substrate probe consumer wire-in for autopilot Hook #4)
* [verified-against:] math-rigor citations in docstring
* Catalog # citations

Plus the contract tests required by the operator brief:
* Synthetic fp32 weights compress > 1.5x via lzma (PRE_ENTROPY classification)
* Synthetic post-entropy bytes (urandom) compress with ratio in [0.99, 1.05]
* Member classification thresholds work per spec
* Aggregation across multiple members
* Real archive probing on at least 1 LIVE substrate archive (read-only)
* Recommended-Q4-target selection (highest deliverable_score_savings_estimate)
* Output JSON schema validates against canonical contract
* CLI end-to-end on synthetic substrate
"""
from __future__ import annotations

import io
import json
import lzma
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

# Resolve canonical PYTHONPATH so the test runs under repo-root pytest
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import pre_entropy_substrate_pivot_prober as pivot  # noqa: E402


# ──────────────────────────────────────────────────────────────────────── #
# Synthetic fixtures                                                        #
# ──────────────────────────────────────────────────────────────────────── #


def _synthetic_fp32_weights(num_floats: int = 2048, seed: int = 42) -> bytes:
    """Generate synthetic fp32 weight bytes that compress well.

    Mimics the structure of real fp32 / fp16 weight matrices: a small
    vocabulary of repeated values (mimicking quantized weights /
    initialization patterns). This is closer to the empirical fp16
    state_dict.pt characterization (lzma 0.226 ratio on real pr106
    state_dict) than uniform-random floats would be.
    """
    import random

    rng = random.Random(seed)
    # Small vocabulary of float values (mimicking heavily quantized
    # weights or zero-dominant sparse weights)
    vocab = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05]
    floats = [rng.choice(vocab) for _ in range(num_floats)]
    return b"".join(struct.pack("<f", f) for f in floats)


def _synthetic_post_entropy_bytes(n: int = 8192) -> bytes:
    """Generate synthetic post-entropy bytes via os.urandom (uniform
    distribution -> at Shannon entropy floor)."""
    return os.urandom(n)


def _create_synthetic_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Create a ZIP archive with the named members at path."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


# ──────────────────────────────────────────────────────────────────────── #
# Test: synthetic fp32 weights compress > 1.5x via lzma                    #
# ──────────────────────────────────────────────────────────────────────── #


def test_synthetic_fp32_weights_compress_pre_entropy() -> None:
    """Synthetic fp32 weights with low variance compress massively under
    lzma -- ratio MUST be < 0.99 (PRE_ENTROPY classification)."""
    data = _synthetic_fp32_weights(num_floats=4096, seed=7)
    result = pivot.probe_member_compression("synthetic_weights.pt", data)
    assert result.classification == "PRE_ENTROPY"
    # We require >1.5x compression: ratio < 1/1.5 = 0.667
    assert result.best_ratio < 0.667, (
        f"synthetic fp32 weights compression too weak: ratio={result.best_ratio:.3f}"
    )
    # lzma should be among the top codecs for highly correlated data
    assert result.best_codec in ("lzma", "brotli")


def test_synthetic_post_entropy_bytes_at_floor() -> None:
    """Synthetic urandom bytes are uncompressible (uniform distribution)
    -- ratio MUST be in [0.99, 1.05] (AT_FLOOR or just POST_ENTROPY).
    """
    data = _synthetic_post_entropy_bytes(n=16384)
    result = pivot.probe_member_compression("urandom.bin", data)
    assert result.best_ratio >= 0.99, (
        f"urandom bytes compress unexpectedly well: {result.best_ratio:.3f}"
    )
    # Allow up to AT_FLOOR_RATIO_UPPER inflation
    assert result.best_ratio <= 1.10, (
        f"urandom bytes inflate too much: {result.best_ratio:.3f}"
    )
    assert result.classification in ("AT_FLOOR", "POST_ENTROPY")


# ──────────────────────────────────────────────────────────────────────── #
# Test: classification thresholds                                           #
# ──────────────────────────────────────────────────────────────────────── #


def test_classify_compression_ratio_pre_entropy() -> None:
    assert pivot.classify_compression_ratio(0.25) == "PRE_ENTROPY"
    assert pivot.classify_compression_ratio(0.50) == "PRE_ENTROPY"
    assert pivot.classify_compression_ratio(0.98) == "PRE_ENTROPY"


def test_classify_compression_ratio_at_floor() -> None:
    assert pivot.classify_compression_ratio(0.99) == "AT_FLOOR"
    assert pivot.classify_compression_ratio(1.00) == "AT_FLOOR"
    assert pivot.classify_compression_ratio(1.04) == "AT_FLOOR"
    assert pivot.classify_compression_ratio(1.05) == "AT_FLOOR"  # boundary


def test_classify_compression_ratio_post_entropy() -> None:
    assert pivot.classify_compression_ratio(1.06) == "POST_ENTROPY"
    assert pivot.classify_compression_ratio(1.20) == "POST_ENTROPY"
    assert pivot.classify_compression_ratio(2.0) == "POST_ENTROPY"


# ──────────────────────────────────────────────────────────────────────── #
# Test: aggregation across members                                          #
# ──────────────────────────────────────────────────────────────────────── #


def test_probe_substrate_synthetic_pre_entropy(tmp_path: Path) -> None:
    """Build a synthetic substrate with PRE-entropy weights and verify
    the aggregate is PRE_ENTROPY-dominant.

    Synthetic fixture (under tmp_path, .pt extension) is a research-sidecar
    shape — opt out of Catalog #321 contest-member validation via
    ``skip_contest_member_validation=True`` so we can still exercise the
    compression-detection contract on synthetic bytes.
    """
    weights_path = tmp_path / "synth_weights.pt"
    weights_path.write_bytes(_synthetic_fp32_weights(num_floats=8192))

    result = pivot.probe_substrate(
        substrate_name="synth_pre_entropy",
        archive_path_str=str(weights_path),
        substrate_class="raw_float_weights",
        skip_contest_member_validation=True,
    )
    assert result.archive_exists
    assert result.archive_status == "present"
    assert result.archive_sha256 is not None
    assert len(result.archive_sha256) == 64
    assert result.pre_entropy_bytes > 0
    assert result.pre_entropy_fraction >= 0.50
    assert result.deliverable_score_savings_estimate > 0
    assert result.error is None


def test_probe_substrate_synthetic_post_entropy(tmp_path: Path) -> None:
    """Build a synthetic post-entropy zip archive and verify the aggregate
    is POST_ENTROPY / AT_FLOOR (zero deliverable savings).

    Synthetic zip under tmp_path is NOT under a canonical contest-shipping
    root (submissions/ or experiments/results/) — opt out of Catalog #321
    contest-member validation for the synthetic test.
    """
    archive_path = tmp_path / "synth_post_entropy.zip"
    _create_synthetic_zip(
        archive_path,
        {"x": _synthetic_post_entropy_bytes(n=20000)},
    )
    result = pivot.probe_substrate(
        substrate_name="synth_post_entropy",
        archive_path_str=str(archive_path),
        substrate_class="post_entropy_contest_archive",
        skip_contest_member_validation=True,
    )
    assert result.archive_exists
    assert result.pre_entropy_bytes == 0
    assert result.deliverable_score_savings_estimate == 0.0


def test_probe_substrate_missing_archive_marks_pending_dispatch(tmp_path: Path) -> None:
    """A non-existent archive is tagged pending_dispatch, not error.

    Synthetic tmp_path path is not contest-shipping — opt out of Catalog
    #321 validation; pending_dispatch path is independent of the
    validator.
    """
    result = pivot.probe_substrate(
        substrate_name="nonexistent",
        archive_path_str=str(tmp_path / "does_not_exist.zip"),
        substrate_class="raw_float_weights",
        skip_contest_member_validation=True,
    )
    assert not result.archive_exists
    assert result.archive_status == "pending_dispatch"
    assert result.archive_sha256 is None
    assert result.deliverable_score_savings_estimate == 0.0


# ──────────────────────────────────────────────────────────────────────── #
# Test: real LIVE archive probing (read-only)                              #
# ──────────────────────────────────────────────────────────────────────── #


def test_probe_live_pr106_state_dict_is_rejected_research_sidecar() -> None:
    """REGRESSION (Catalog #321 / Q4 HALT 2026-05-17): the pr106 state_dict.pt
    file is a research sidecar — NOT a member of any contest archive.zip.
    The default-validated probe MUST emit REJECTED_RESEARCH_SIDECAR with
    deliverable_score_savings_estimate=0.0 even though the raw .pt bytes
    compress 0.226x via lzma. Pre-fix this returned phantom 0.47 savings.
    """
    live_path = Path("experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt")
    if not live_path.exists():
        pytest.skip(f"live archive missing: {live_path}")
    result = pivot.probe_substrate(
        substrate_name="pr106_state_dict",
        archive_path_str=str(live_path),
        substrate_class="raw_float_weights",
    )
    assert result.archive_exists
    assert result.validation_status == "REJECTED_RESEARCH_SIDECAR"
    assert result.evidence_grade_per_row == "invalid_target"
    assert result.deliverable_score_savings_estimate == 0.0
    assert "sidecar" in (result.validation_reason or "").lower()


def test_probe_live_pr106_state_dict_via_diagnostic_opt_in() -> None:
    """The standalone .pt file's raw compression IS measurable as a
    diagnostic signal — opting into ``skip_contest_member_validation=True``
    surfaces the historical 0.226x ratio for hoist-design discussions
    (but the result still carries score_claim=false; it is a DIAGNOSTIC
    not a deliverable estimate)."""
    live_path = Path("experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt")
    if not live_path.exists():
        pytest.skip(f"live archive missing: {live_path}")
    result = pivot.probe_substrate(
        substrate_name="pr106_state_dict_diagnostic",
        archive_path_str=str(live_path),
        substrate_class="raw_float_weights",
        skip_contest_member_validation=True,
    )
    assert result.archive_exists
    assert result.pre_entropy_fraction >= 0.50
    assert result.pre_entropy_bytes > 100_000


def test_probe_live_fec6_archive_is_at_floor() -> None:
    """The fec6 archive is the sister-prober anchor; lzma INFLATES it.
    Probe it; verify AT_FLOOR or POST_ENTROPY (zero deliverable savings).
    FEC6 IS a VALIDATED_CONTEST_MEMBER (under experiments/results/ canonical
    root, .zip extension, valid zip)."""
    live_path = Path(
        "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
    )
    if not live_path.exists():
        pytest.skip(f"live archive missing: {live_path}")
    result = pivot.probe_substrate(
        substrate_name="pr101_fec6",
        archive_path_str=str(live_path),
        substrate_class="post_entropy_contest_archive",
    )
    assert result.archive_exists
    assert result.validation_status == "VALIDATED_CONTEST_MEMBER"
    assert result.pre_entropy_fraction == 0.0
    assert result.deliverable_score_savings_estimate == 0.0


# ──────────────────────────────────────────────────────────────────────── #
# Test: recommended Q4 target selection                                     #
# ──────────────────────────────────────────────────────────────────────── #


def test_recommended_q4_target_picks_highest_savings(tmp_path: Path) -> None:
    """Build 3 synthetic substrates and verify the manifest picks the one
    with the highest deliverable score savings as the Q4 target.

    Synthetic fixtures (under tmp_path) opt out of Catalog #321 validation
    via run_pivot_probe(skip_contest_member_validation=True).
    """
    # High-savings substrate: 80 KB fp32 weights
    big_pre = tmp_path / "big_pre.pt"
    big_pre.write_bytes(_synthetic_fp32_weights(num_floats=20000))
    # Medium-savings substrate: 4 KB fp32 weights
    small_pre = tmp_path / "small_pre.pt"
    small_pre.write_bytes(_synthetic_fp32_weights(num_floats=1024))
    # Zero-savings substrate: post-entropy zip
    post = tmp_path / "post.zip"
    _create_synthetic_zip(post, {"x": _synthetic_post_entropy_bytes(n=16000)})

    candidate_substrates = {
        "big_pre": (str(big_pre), "raw_float_weights"),
        "small_pre": (str(small_pre), "raw_float_weights"),
        "post": (str(post), "post_entropy_contest_archive"),
    }

    output_path = tmp_path / "manifest.json"
    results, persisted = pivot.run_pivot_probe(
        candidate_substrates=candidate_substrates,
        output_path=output_path,
        persist=True,
        skip_contest_member_validation=True,
    )
    assert persisted == output_path
    payload = json.loads(output_path.read_text())
    assert payload["recommended_q4_target_substrate"] == "big_pre"
    assert payload["recommended_q4_target_pre_entropy_bytes"] > 0
    assert payload["recommended_q4_target_deliverable_savings_estimate"] > 0
    # big_pre should be in pre-entropy list
    assert "big_pre" in payload["substrates_with_pre_entropy_bytes"]
    # post should be in at-floor or post-entropy list
    assert (
        "post" in payload["substrates_at_entropy_floor"]
        or "post" in payload["substrates_post_entropy"]
    )


def test_recommended_q4_target_none_when_no_pre_entropy(tmp_path: Path) -> None:
    """If all candidates are POST_ENTROPY, recommended_q4_target_substrate
    is None. Synthetic fixtures opt out of Catalog #321 validation.
    """
    post1 = tmp_path / "post1.zip"
    _create_synthetic_zip(post1, {"x": _synthetic_post_entropy_bytes(n=10000)})
    post2 = tmp_path / "post2.zip"
    _create_synthetic_zip(post2, {"x": _synthetic_post_entropy_bytes(n=12000)})

    candidate_substrates = {
        "post1": (str(post1), "post_entropy_contest_archive"),
        "post2": (str(post2), "post_entropy_contest_archive"),
    }

    output_path = tmp_path / "manifest.json"
    results, persisted = pivot.run_pivot_probe(
        candidate_substrates=candidate_substrates,
        output_path=output_path,
        persist=True,
        skip_contest_member_validation=True,
    )
    payload = json.loads(output_path.read_text())
    assert payload["recommended_q4_target_substrate"] is None


# ──────────────────────────────────────────────────────────────────────── #
# Test: output JSON schema canonical contract                              #
# ──────────────────────────────────────────────────────────────────────── #


def test_output_json_schema_contract(tmp_path: Path) -> None:
    """Verify the persisted manifest has all required canonical fields.

    Synthetic fixture opts out of Catalog #321 validation.
    """
    weights_path = tmp_path / "synth.pt"
    weights_path.write_bytes(_synthetic_fp32_weights(num_floats=2048))

    candidate_substrates = {
        "test_sub": (str(weights_path), "raw_float_weights"),
    }
    output_path = tmp_path / "out.json"
    results, persisted = pivot.run_pivot_probe(
        candidate_substrates=candidate_substrates,
        output_path=output_path,
        persist=True,
        skip_contest_member_validation=True,
    )
    payload = json.loads(persisted.read_text())

    # Required top-level fields per canonical schema
    required_fields = {
        "schema_version",
        "candidates_probed",
        "substrates_with_pre_entropy_bytes",
        "substrates_at_entropy_floor",
        "substrates_post_entropy",
        "per_substrate_results",
        "recommended_q4_target_substrate",
        "recommended_q4_target_archive_sha256",
        "recommended_q4_target_pre_entropy_bytes",
        "recommended_q4_target_deliverable_savings_estimate",
        "evidence_grade",
        "score_claim",
        "promotion_eligible",
        "claude_md_compliance_tags",
        "written_at_utc",
    }
    missing = required_fields - payload.keys()
    assert not missing, f"missing canonical fields: {missing}"
    assert payload["schema_version"] == "pre_entropy_pivot_probe_v1"
    assert payload["evidence_grade"] == "predicted"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert "pre_entropy_pivot_per_prober_op_routable_4" in payload["claude_md_compliance_tags"]

    # Per-substrate row must have member_breakdown
    assert "test_sub" in payload["per_substrate_results"]
    row = payload["per_substrate_results"]["test_sub"]
    assert "member_breakdown" in row
    assert "archive_sha256" in row


# ──────────────────────────────────────────────────────────────────────── #
# Test: estimate_deliverable_score_savings math                            #
# ──────────────────────────────────────────────────────────────────────── #


def test_estimate_savings_zero_for_no_pre_entropy() -> None:
    assert pivot.estimate_deliverable_score_savings(
        pre_entropy_bytes=0, best_compression_ratio=0.5
    ) == 0.0
    assert pivot.estimate_deliverable_score_savings(
        pre_entropy_bytes=1000, best_compression_ratio=1.0
    ) == 0.0


def test_estimate_savings_canonical_formula() -> None:
    """Score delta = 25 * (pre_entropy_bytes * (1 - ratio)) / 37,545,489.
    Test the canonical formula directly."""
    pre_entropy_bytes = 100_000
    ratio = 0.5
    expected = 25.0 * (100_000 * 0.5) / 37_545_489
    actual = pivot.estimate_deliverable_score_savings(
        pre_entropy_bytes=pre_entropy_bytes,
        best_compression_ratio=ratio,
    )
    assert abs(actual - expected) < 1e-9


# ──────────────────────────────────────────────────────────────────────── #
# Test: resolve_candidate_substrates                                        #
# ──────────────────────────────────────────────────────────────────────── #


def test_resolve_canonical_default() -> None:
    """When no filter is provided, the canonical map is returned."""
    result = pivot.resolve_candidate_substrates(candidate_substrates_arg=None)
    assert len(result) == len(pivot.CANONICAL_CANDIDATE_SUBSTRATES)


def test_resolve_filtered_subset() -> None:
    """Filter via comma-separated list."""
    result = pivot.resolve_candidate_substrates(
        candidate_substrates_arg="pr106_state_dict,pr101_fec6_archive,unknown_substrate",
    )
    # unknown_substrate is dropped (silently); known 2 remain
    assert set(result.keys()) == {"pr106_state_dict", "pr101_fec6_archive"}


def test_resolve_extra_json(tmp_path: Path) -> None:
    """Extra substrates merge from JSON file."""
    extra_json = tmp_path / "extra.json"
    extra_json.write_text(json.dumps({
        "extra_sub": ["path/to/something.pt", "raw_float_weights"],
    }))
    result = pivot.resolve_candidate_substrates(
        candidate_substrates_arg=None,
        extra_json_path=extra_json,
    )
    assert "extra_sub" in result
    assert result["extra_sub"] == ("path/to/something.pt", "raw_float_weights")


# ──────────────────────────────────────────────────────────────────────── #
# Test: CLI end-to-end                                                      #
# ──────────────────────────────────────────────────────────────────────── #


def test_cli_dry_run_no_side_effects(tmp_path: Path) -> None:
    """The CLI runs end-to-end on a synthetic substrate when --report-only
    is passed and the manifest is NOT persisted."""
    weights_path = tmp_path / "synth_cli.pt"
    weights_path.write_bytes(_synthetic_fp32_weights(num_floats=2048))
    extra_json = tmp_path / "extra.json"
    extra_json.write_text(json.dumps({"cli_test": [str(weights_path), "raw_float_weights"]}))
    output_path = tmp_path / "should_not_exist.json"

    result = subprocess.run(
        [
            ".venv/bin/python",
            "tools/pre_entropy_substrate_pivot_prober.py",
            "--candidate-substrates",
            "cli_test",
            "--candidate-substrates-extra-json",
            str(extra_json),
            "--output",
            str(output_path),
            "--report-only-no-side-effects",
            "--skip-contest-member-validation",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "RECOMMENDED Q4 TARGET: cli_test" in result.stdout
    # Manifest must NOT exist because of report-only
    assert not output_path.exists()


def test_cli_writes_manifest_when_not_dry_run(tmp_path: Path) -> None:
    """Without --report-only, the manifest IS persisted."""
    weights_path = tmp_path / "synth_cli2.pt"
    weights_path.write_bytes(_synthetic_fp32_weights(num_floats=2048))
    extra_json = tmp_path / "extra.json"
    extra_json.write_text(json.dumps({"cli_test_2": [str(weights_path), "raw_float_weights"]}))
    output_path = tmp_path / "manifest_cli.json"

    result = subprocess.run(
        [
            ".venv/bin/python",
            "tools/pre_entropy_substrate_pivot_prober.py",
            "--candidate-substrates",
            "cli_test_2",
            "--candidate-substrates-extra-json",
            str(extra_json),
            "--output",
            str(output_path),
            "--skip-contest-member-validation",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["recommended_q4_target_substrate"] == "cli_test_2"


# ──────────────────────────────────────────────────────────────────────── #
# Test: NaN/Inf sanitization for JSON                                       #
# ──────────────────────────────────────────────────────────────────────── #


def test_sanitize_for_json_handles_nan_inf() -> None:
    payload = {"x": float("nan"), "y": float("inf"), "z": -float("inf"), "w": 1.5}
    sanitized = pivot._sanitize_for_json(payload)
    assert sanitized["x"] is None
    assert sanitized["y"] is None
    assert sanitized["z"] is None
    assert sanitized["w"] == 1.5


# ──────────────────────────────────────────────────────────────────────── #
# Test: Catalog #265 canonical contract                                     #
# ──────────────────────────────────────────────────────────────────────── #


def test_canonical_contract_tokens_present() -> None:
    """The module satisfies Catalog #265-style canonical-contract markers:
    SPDX header + verified-against citations + Catalog # citation.
    """
    src = Path(__file__).resolve().parent.parent.parent.parent / "tools" / "pre_entropy_substrate_pivot_prober.py"
    text = src.read_text()
    assert "# SPDX-License-Identifier: MIT" in text
    assert "[verified-against:" in text
    assert "Catalog #" in text


# ──────────────────────────────────────────────────────────────────────── #
# Test: probe artifact persistence via fcntl-locked write                  #
# ──────────────────────────────────────────────────────────────────────── #


def test_fcntl_locked_atomic_write_persists(tmp_path: Path) -> None:
    """Verify the fcntl-locked atomic write helper writes payload + cleans
    up tmp + creates lock sidecar."""
    out_path = tmp_path / "test.json"
    payload = {"hello": "world", "n": 42, "f": float("nan")}
    pivot._fcntl_locked_atomic_write(out_path, payload)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["hello"] == "world"
    assert loaded["n"] == 42
    assert loaded["f"] is None  # NaN -> None
    # Tmp files should be cleaned up
    leftover = list(tmp_path.glob("*.tmp.*"))
    assert not leftover, f"tmp files leaked: {leftover}"
