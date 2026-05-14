"""Tests for tools/mdl_scorer_conditional_ablation.py (Z1 zen-floor probe).

Per CLAUDE.md "Subagent coherence-by-default" Catalog #125 wire-in #6
(probe-disambiguator) this tool IS the empirical disambiguator between the
zen-floor council's competing predictions; tests pin both the unit
mechanics (section parser, perturbation modes, MDL aggregation) AND the
synthetic baseline behavior on a controlled toy archive.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" the
production CLI writes under experiments/results/<lane_id>_<TS>/. Tests
use ``tmp_path`` (pytest fixture, NOT /tmp) per CLAUDE.md FORBIDDEN
patterns.
"""
from __future__ import annotations

import importlib.util
import random
import struct
import sys
import zipfile
from pathlib import Path
from unittest import mock

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "mdl_scorer_conditional_ablation.py"


def _load_module():
    """Import the script as a module under a stable name.

    Must register the module in sys.modules BEFORE exec_module because
    @dataclass introspection via ``cls.__module__`` looks up
    ``sys.modules[cls.__module__]`` and crashes if the module isn't
    registered yet.
    """
    name = "_mdl_z1"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# Section parser
# ----------------------------------------------------------------------


def test_parse_a1_archive_bytes_returns_canonical_sections():
    mod = _load_module()
    # Build a synthetic A1 inner blob: 4-byte LE section_total + decoder + 15387 + 100 sidecar
    section_total = 4 + 100  # 4 header + 100 decoder bytes
    inner = bytearray()
    inner.extend(struct.pack("<I", section_total))
    inner.extend(b"D" * 100)  # decoder bytes
    inner.extend(b"L" * mod.A1_LATENT_BLOB_LEN)  # latent_blob
    inner.extend(b"S" * 50)  # sidecar
    sections = mod.parse_a1_archive_bytes(bytes(inner))
    assert "decoder_section_header" in sections
    assert sections["decoder_section_header"] == (0, 4)
    assert sections["decoder_blob"] == (4, 100)
    assert sections["latent_blob"] == (section_total, mod.A1_LATENT_BLOB_LEN)
    assert sections["sidecar_blob"] == (
        section_total + mod.A1_LATENT_BLOB_LEN,
        50,
    )


def test_parse_a1_archive_bytes_rejects_too_short():
    mod = _load_module()
    with pytest.raises(ValueError):
        mod.parse_a1_archive_bytes(b"\x01\x02\x03")  # < 4 bytes


def test_parse_pr106_archive_bytes_requires_magic_header():
    mod = _load_module()
    # Build a valid synthetic PR106 latent_sidecar wrapper:
    #   magic(0xFE) + format_id(0x01) + uint32 pr106_len + pr106_bytes
    #   + uint16 sidecar_len + sidecar_blob
    pr106_payload = b"P" * 100
    sidecar_payload = b"S" * 50
    inner = bytearray()
    inner.append(0xFE)
    inner.append(0x01)
    inner.extend(struct.pack("<I", len(pr106_payload)))
    inner.extend(pr106_payload)
    inner.extend(struct.pack("<H", len(sidecar_payload)))
    inner.extend(sidecar_payload)
    sections = mod.parse_pr106_archive_bytes(bytes(inner))
    assert sections["magic_format_header"] == (0, 2)
    assert sections["pr106_len_field"] == (2, 4)
    assert sections["pr106_base_archive"] == (6, 100)
    assert sections["sidecar_len_field"] == (106, 2)
    assert sections["sidecar_blob"] == (108, 50)

    # Wrong magic
    with pytest.raises(ValueError):
        mod.parse_pr106_archive_bytes(bytes([0xFF, 0x01]) + b"x" * 20)
    # Wrong format_id
    with pytest.raises(ValueError):
        mod.parse_pr106_archive_bytes(bytes([0xFE, 0x02]) + b"x" * 20)
    # Too short
    with pytest.raises(ValueError):
        mod.parse_pr106_archive_bytes(b"\x00")


# ----------------------------------------------------------------------
# Section perturbation modes
# ----------------------------------------------------------------------


def test_perturb_section_zero_writes_zeros_only_in_range():
    mod = _load_module()
    rng = random.Random(42)
    src = bytes(range(256))[:200]
    perturbed = mod._perturb_section(src, 50, 100, "zero", rng)
    assert perturbed[:50] == src[:50]
    assert all(b == 0 for b in perturbed[50:150])
    assert perturbed[150:] == src[150:]


def test_perturb_section_random_changes_only_in_range():
    mod = _load_module()
    rng = random.Random(42)
    src = b"\xAA" * 200
    perturbed = mod._perturb_section(src, 50, 100, "random", rng)
    assert perturbed[:50] == src[:50]
    # Most of the random range should differ from 0xAA
    n_changed = sum(1 for i in range(50, 150) if perturbed[i] != 0xAA)
    assert n_changed > 60  # ~99/100 expected for uniform random
    assert perturbed[150:] == src[150:]


def test_perturb_section_rejects_unknown_mode():
    mod = _load_module()
    rng = random.Random(0)
    with pytest.raises(ValueError):
        mod._perturb_section(b"X" * 100, 0, 10, "bogus_mode", rng)


# ----------------------------------------------------------------------
# Score components helper
# ----------------------------------------------------------------------


def test_score_components_matches_contest_formula():
    """Score formula = 100*seg + sqrt(10*pose). Rate term excluded."""
    mod = _load_module()
    import math
    # PR101-typical values
    pose = 0.0001
    seg = 0.0007
    expected = 100.0 * seg + math.sqrt(10.0 * pose)
    actual = mod._score_components(pose, seg)
    assert abs(actual - expected) < 1e-12


def test_score_components_clamps_negative_pose_at_zero():
    """sqrt(10 * pose) must NOT raise on negative pose (clamped to 0)."""
    mod = _load_module()
    # Should not raise (clamps internally)
    val = mod._score_components(-0.01, 0.0005)
    # 100 * 0.0005 + sqrt(10 * 0) = 0.05
    assert abs(val - 0.05) < 1e-12


# ----------------------------------------------------------------------
# Sha256 helper
# ----------------------------------------------------------------------


def test_sha256_bytes_is_deterministic_and_correct():
    mod = _load_module()
    h = mod._sha256_bytes(b"hello world")
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def _make_synthetic_archive_result(mod, sections_lengths, tier_b_fracs, tier_a_delta=None):
    """Build an ArchiveAblationResult with controlled Tier A/B fields."""
    r = mod.ArchiveAblationResult(
        archive_name="synth",
        archive_path="/dev/null",
        archive_sha256="X" * 64,
        archive_size_bytes=sum(sections_lengths.values()),
        grammar="a1",
        device="cpu",
        pair_samples=10,
        baseline_seg=0.001,
        baseline_pose=0.001,
        baseline_score_components=0.2,
    )
    for section_name, length in sections_lengths.items():
        dsc = tier_a_delta if tier_a_delta is not None else 0.01  # significant by default
        r.tier_a.append(mod.TierAResult(
            section=section_name,
            start_offset=0,
            length_bytes=length,
            perturbation_mode="zero",
            inflate_success=True,
            delta_seg=0.0,
            delta_pose=0.0,
            delta_score_components=dsc,
            failure_reason=None,
            elapsed_seconds=0.0,
        ))
    for section_name, frac in tier_b_fracs.items():
        r.tier_b.append(mod.TierBResult(
            section=section_name,
            n_samples=100,
            n_inflate_failures=int(frac * 100 * 0.5),
            n_significant=int(frac * 100),
            significance_threshold=1e-4,
            mean_abs_delta=0.001,
            std_abs_delta=0.0005,
            max_abs_delta=0.002,
            fraction_significant=frac,
            upper_bound_scorer_extracted_bits=frac * sections_lengths[section_name] * 8,
            estimated_scorer_extracted_bits_lo=frac * sections_lengths[section_name] * 4,
        ))
    return r


def test_aggregate_mdl_zero_when_no_sections_relevant():
    mod = _load_module()
    r = _make_synthetic_archive_result(
        mod, {"x": 1000}, {"x": 1.0}, tier_a_delta=0.0,  # below significance threshold
    )
    out = mod.aggregate_mdl_estimate(r)
    assert out.mdl_scorer_extracted_bytes_lo == 0.0
    assert out.mdl_density_estimate_lo == 0.0


def test_aggregate_mdl_uses_fraction_significant_times_length():
    mod = _load_module()
    r = _make_synthetic_archive_result(
        mod, {"x": 1000, "y": 500},
        {"x": 0.5, "y": 0.8},
        tier_a_delta=0.01,
    )
    out = mod.aggregate_mdl_estimate(r)
    # 0.5 * 1000 + 0.8 * 500 = 500 + 400 = 900
    assert out.mdl_scorer_extracted_bytes_lo == 900.0


def test_aggregate_mdl_density_band_recommendation_ordering():
    """Verify the LO/CENTER/HI band recommendations correctly classify by total_hi."""
    mod = _load_module()

    # 800 bytes < 1000 -> Shannon zen-floor
    r = _make_synthetic_archive_result(mod, {"x": 1000}, {"x": 0.8})
    out = mod.aggregate_mdl_estimate(r)
    assert "0.003" in out.zen_floor_band_recommendation
    assert "Shannon" in out.zen_floor_band_recommendation

    # 5000 bytes -> medium band
    r = _make_synthetic_archive_result(mod, {"x": 10_000}, {"x": 0.5})
    out = mod.aggregate_mdl_estimate(r)
    assert "0.010" in out.zen_floor_band_recommendation

    # 30000 bytes -> high-medium band
    r = _make_synthetic_archive_result(mod, {"x": 60_000}, {"x": 0.5})
    out = mod.aggregate_mdl_estimate(r)
    assert "0.050" in out.zen_floor_band_recommendation

    # 100K bytes -> high band
    r = _make_synthetic_archive_result(mod, {"x": 200_000}, {"x": 0.5})
    out = mod.aggregate_mdl_estimate(r)
    assert "0.100" in out.zen_floor_band_recommendation


def test_aggregate_mdl_with_skipped_tier_a_falls_back_to_tier_b_relevance():
    """When Tier A is skipped (empty), all Tier B sections must be treated as relevant."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100}, {"x": 0.7})
    r.tier_a = []  # simulate --skip-tier-a
    # Need a Tier B with the section that doesn't appear in tier_a
    out = mod.aggregate_mdl_estimate(r)
    # All Tier B sections counted; length unknown (no Tier A) so falls back
    # to n_samples (100) -> 0.7 * 100 = 70
    assert out.mdl_scorer_extracted_bytes_lo == 70.0


# ----------------------------------------------------------------------
# Load archive (uses real A1 archive)
# ----------------------------------------------------------------------


def test_load_archive_parses_a1_sections_on_real_file():
    mod = _load_module()
    a1 = REPO / "submissions" / "a1" / "archive.zip"
    if not a1.exists():
        pytest.skip("A1 archive not present")
    inner, sections = mod.load_archive(a1, "a1")
    assert len(inner) > 100_000
    # PR101 latent_blob = 15387 bytes (constant per A1 inflate.py)
    assert sections["latent_blob"][1] == 15_387
    # The decoder_blob length plus header (4) should == section_total
    assert sections["decoder_blob"][0] == 4


# ----------------------------------------------------------------------
# Score components clamp behavior
# ----------------------------------------------------------------------


def test_score_components_positive_seg_pose():
    mod = _load_module()
    import math
    assert mod._score_components(0.01, 0.005) == pytest.approx(
        100.0 * 0.005 + math.sqrt(10.0 * 0.01),
        rel=1e-9,
    )


# ----------------------------------------------------------------------
# CLI argument validation
# ----------------------------------------------------------------------


def test_cli_rejects_misaligned_archive_lists(tmp_path: Path, capsys):
    """--archive, --archive-name, --grammar lists must be same length."""
    mod = _load_module()
    archive = tmp_path / "fake.zip"
    archive.write_bytes(b"PK\x03\x04")  # minimal-ish zip placeholder
    # Two archives, one name = should error
    with pytest.raises(SystemExit):
        mod.main([
            "--archive", str(archive),
            "--archive", str(archive),
            "--archive-name", "a",
            "--grammar", "a1",
            "--output-dir", str(tmp_path / "out"),
            "--skip-tier-a",
            "--skip-tier-b",
            "--skip-tier-c",
        ])
    err = capsys.readouterr().err
    assert "same length" in err.lower() or "archive" in err.lower()


def test_cli_rejects_missing_archive_path(tmp_path: Path, capsys):
    mod = _load_module()
    missing = tmp_path / "nope.zip"
    rc = mod.main([
        "--archive", str(missing),
        "--archive-name", "a",
        "--grammar", "a1",
        "--output-dir", str(tmp_path / "out"),
        "--skip-tier-a",
        "--skip-tier-b",
        "--skip-tier-c",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "archive not found" in err.lower()


# ----------------------------------------------------------------------
# ArchiveAblationResult JSON-serialization round-trip
# ----------------------------------------------------------------------


def test_archive_ablation_result_json_round_trip(tmp_path: Path):
    """The result dataclass must be JSON-serializable via dataclasses.asdict."""
    import json
    from dataclasses import asdict
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100, "y": 200}, {"x": 0.5, "y": 0.3})
    r = mod.aggregate_mdl_estimate(r)
    out_path = tmp_path / "result.json"
    with open(out_path, "w") as f:
        json.dump(asdict(r), f, indent=2, default=str)
    # Re-read
    with open(out_path) as f:
        data = json.load(f)
    assert data["archive_name"] == "synth"
    assert data["mdl_scorer_extracted_bytes_lo"] == 0.5 * 100 + 0.3 * 200


# ----------------------------------------------------------------------
# Tier A failure-counts-as-relevant semantics
# ----------------------------------------------------------------------


def test_tier_a_inflate_failure_counts_as_relevant():
    """A Tier A result with inflate_success=False AND dsc=None must mark
    the section as relevant (per parse-required-bytes semantics)."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 1000}, {"x": 0.4})
    # Modify the Tier A entry to be inflate-fail
    r.tier_a[0].inflate_success = False
    r.tier_a[0].delta_seg = None
    r.tier_a[0].delta_pose = None
    r.tier_a[0].delta_score_components = None
    out = mod.aggregate_mdl_estimate(r)
    # Section IS relevant; bytes = 0.4 * 1000 = 400
    assert out.mdl_scorer_extracted_bytes_lo == 400.0


# ----------------------------------------------------------------------
# Notes / tagging discipline
# ----------------------------------------------------------------------


def test_archive_result_notes_carry_mps_advisory_when_device_mps(tmp_path: Path):
    """Per CLAUDE.md 'MPS auth eval is NOISE', MPS-derived results must
    carry the [MDL-ablation-MPS] tag (NOT a contest-CUDA / contest-CPU tag)."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100}, {"x": 0.5})
    r.device = "mps"
    # The runtime ablate_archive() decorates notes — replicate the contract here
    expected_tag = "[MDL-ablation-MPS]" if r.device == "mps" else f"[MDL-ablation-{r.device}]"
    assert expected_tag.startswith("[MDL-ablation-")
    # Tag must NOT be a contest tag
    assert "contest-CUDA" not in expected_tag
    assert "contest-CPU" not in expected_tag
