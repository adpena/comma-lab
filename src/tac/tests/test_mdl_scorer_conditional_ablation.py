# SPDX-License-Identifier: MIT
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
import json
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


def test_section_filter_include_then_exclude_semantics():
    """Section sharding must be deterministic and default-safe.

    Explicit include selects a shard, but explicit/default exclude still wins
    so provenance-only sections are not accidentally sampled in fast scouts.
    """
    mod = _load_module()
    assert mod._section_selected(
        "decoder_blob",
        include_sections={"decoder_blob"},
        exclude_sections={"encoder_blob"},
    )
    assert not mod._section_selected(
        "latent_blob",
        include_sections={"decoder_blob"},
        exclude_sections={"encoder_blob"},
    )
    assert not mod._section_selected(
        "encoder_blob",
        include_sections={"encoder_blob"},
        exclude_sections={"encoder_blob"},
    )


def test_ibps1_default_excludes_training_provenance_encoder_blob():
    mod = _load_module()
    assert mod.DEFAULT_EXCLUDED_SECTIONS_BY_GRAMMAR["ibps1"] == frozenset({"encoder_blob"})


def test_parse_byte_offset_specs_supports_explicit_and_bare_offsets():
    mod = _load_module()
    assert mod._parse_byte_offset_specs(["decoder_blob:42", "decoder_blob:0x2b"]) == {
        "decoder_blob": [42, 43],
    }
    assert mod._parse_byte_offset_specs(["42"], include_sections={"latent_blob"}) == {
        "latent_blob": [42],
    }
    with pytest.raises(ValueError, match="bare --byte-offset"):
        mod._parse_byte_offset_specs(["42"], include_sections={"a", "b"})
    with pytest.raises(ValueError, match="must be >= 0"):
        mod._parse_byte_offset_specs(["decoder_blob:-1"])


def test_normalize_grammar_aliases_and_pair_indices_file(tmp_path: Path):
    mod = _load_module()
    assert mod.normalize_grammar("PR101") == "a1"
    assert mod.normalize_grammar("c6_e4_mdl_ibps") == "ibps1"
    json_path = tmp_path / "pairs.json"
    json_path.write_text("[5, 1, 5, 0]")
    assert mod._load_pair_indices_file(json_path) == [0, 1, 5]
    txt_path = tmp_path / "pairs.txt"
    txt_path.write_text("0, 2 4")
    assert mod._load_pair_indices_file(txt_path) == [0, 2, 4]
    bad_path = tmp_path / "bad.txt"
    bad_path.write_text("999")
    with pytest.raises(ValueError, match="out of range"):
        mod._load_pair_indices_file(bad_path)


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
            length_bytes=sections_lengths[section_name],
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
    """When Tier A is skipped, Tier B still carries full section length."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 1000}, {"x": 0.7})
    r.tier_a = []  # simulate --skip-tier-a
    # Need a Tier B with the section that doesn't appear in tier_a
    out = mod.aggregate_mdl_estimate(r)
    # All Tier B sections counted; length is preserved on TierBResult, so
    # a sharded --skip-tier-a scout estimates section density correctly.
    assert out.mdl_scorer_extracted_bytes_lo == 700.0


def test_aggregate_mdl_tier_c_density_across_class_anchor():
    """Tier C-only runs must emit a low-density class-shift signal.

    This pins the C6-style curve: state_dict has a knee/plateau while latents
    stay nearly inert at sigma=1.0. Tier A/B may be skipped, but the aggregate
    still carries a rankable Tier C scalar for autopilot.
    """
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 1000}, {})
    r.tier_a = []
    r.tier_b = []
    r.tier_c = [
        mod.TierCResult("state_dict", 0.01, 0.0, 0.0, 0.0745, 0.0),
        mod.TierCResult("state_dict", 0.1, 0.0, 0.0, 0.0639, 0.0),
        mod.TierCResult("latents", 0.01, 0.0, 0.0, 0.0001, 0.0),
        mod.TierCResult("latents", 1.0, 0.0, 0.0, -0.00213, 0.0),
    ]
    out = mod.aggregate_mdl_estimate(r)
    assert out.mdl_density_estimate_lo == 0.0
    assert out.mdl_tier_c_density_estimate <= 0.30
    assert out.mdl_tier_c_substrate_class_verdict == "across_class"
    assert out.mdl_tier_c_latent_sigma1_delta == pytest.approx(0.00213)
    assert out.mdl_tier_c_curve_knee_signal == pytest.approx(0.0639 / 0.0745)


def test_aggregate_mdl_tier_c_density_within_class_anchor():
    """Large latent sensitivity plus monotonic state_dict growth is saturated."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 1000}, {})
    r.tier_a = []
    r.tier_b = []
    r.tier_c = [
        mod.TierCResult("state_dict", 0.01, 0.0, 0.0, 0.5, 0.0),
        mod.TierCResult("state_dict", 0.1, 0.0, 0.0, 5.0, 0.0),
        mod.TierCResult("latents", 1.0, 0.0, 0.0, 5.0, 0.0),
    ]
    out = mod.aggregate_mdl_estimate(r)
    assert out.mdl_tier_c_density_estimate >= 0.70
    assert out.mdl_tier_c_substrate_class_verdict == "within_class"
    assert out.mdl_tier_c_curve_knee_signal == pytest.approx(10.0)


def test_aggregate_mdl_tier_c_empty_rows_leave_default_signal():
    """Missing or failed Tier C rows must not manufacture a rank signal."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 1000}, {})
    r.tier_a = []
    r.tier_b = []
    r.tier_c = [
        mod.TierCResult("state_dict", 0.01, None, None, None, 0.0),
        mod.TierCResult("latents", 1.0, None, None, None, 0.0),
    ]
    out = mod.aggregate_mdl_estimate(r)
    assert out.mdl_tier_c_density_estimate == 0.0
    assert out.mdl_tier_c_substrate_class_verdict == ""
    assert out.mdl_tier_c_curve_knee_signal == 0.0
    assert out.mdl_tier_c_latent_sigma1_delta == 0.0


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


def test_cli_rejects_unknown_grammar_without_generic_escape_hatch(tmp_path: Path, capsys):
    mod = _load_module()
    archive = tmp_path / "fake.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    rc = mod.main([
        "--archive", str(archive),
        "--archive-name", "a",
        "--grammar", "surprise",
        "--output-dir", str(tmp_path / "out"),
        "--skip-tier-a",
        "--skip-tier-b",
        "--skip-tier-c",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown grammar" in err.lower()


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


# ----------------------------------------------------------------------
# IBPS1 (C6 MDL-IBPS) grammar parser tests
# ----------------------------------------------------------------------


def _build_synthetic_ibps1_inner(
    *,
    latent_dim: int = 24,
    num_pairs: int = 600,
    encoder_blob: bytes | None = None,
    decoder_blob: bytes | None = None,
    meta_blob: bytes | None = None,
) -> bytes:
    """Build a valid IBPS1 inner-blob byte sequence (header + 4 sections).

    Uses synthetic byte payloads (not real brotli streams) so parser tests
    can run without torch / brotli. Section LENGTHS are what the parser
    cares about; section CONTENTS only matter for decode tests (which
    require the real C6 substrate import).
    """
    if encoder_blob is None:
        encoder_blob = b"\x01" * 65_000
    if decoder_blob is None:
        decoder_blob = b"\x02" * 145_000
    latent_blob = b"\x03" * (num_pairs * latent_dim)  # int8
    if meta_blob is None:
        meta_blob = b'{"a":1,"b":2}'
    header_fmt = "<4sBHHIIII"
    header = struct.pack(
        header_fmt,
        b"IBPS",
        1,  # version
        latent_dim,
        num_pairs,
        len(encoder_blob),
        len(decoder_blob),
        len(latent_blob),
        len(meta_blob),
    )
    return header + encoder_blob + decoder_blob + latent_blob + meta_blob


def test_parse_ibps1_archive_bytes_returns_canonical_sections():
    """Parse a synthetic IBPS1 inner blob and verify all 5 sections."""
    mod = _load_module()
    inner = _build_synthetic_ibps1_inner(
        latent_dim=24,
        num_pairs=600,
        encoder_blob=b"\xaa" * 100,
        decoder_blob=b"\xbb" * 200,
        meta_blob=b'{"k":"v"}',
    )
    sections = mod.parse_ibps1_archive_bytes(inner)
    assert set(sections.keys()) == {
        "ibps1_header",
        "encoder_blob",
        "decoder_blob",
        "latent_blob",
        "meta_blob",
    }
    assert sections["ibps1_header"] == (0, 25)
    assert sections["encoder_blob"] == (25, 100)
    assert sections["decoder_blob"] == (125, 200)
    # latent_blob = num_pairs * latent_dim = 600 * 24 = 14400
    assert sections["latent_blob"] == (325, 14_400)
    assert sections["meta_blob"] == (14_725, 9)  # len('{"k":"v"}')


def test_parse_ibps1_archive_bytes_rejects_short_header():
    """A buffer shorter than 25 bytes must raise ValueError."""
    mod = _load_module()
    with pytest.raises(ValueError, match="ibps1 archive too short"):
        mod.parse_ibps1_archive_bytes(b"IBPS\x01")  # < 25 bytes


def test_parse_ibps1_archive_bytes_rejects_bad_magic():
    """Bad magic bytes must raise ValueError."""
    mod = _load_module()
    bad = struct.pack(
        "<4sBHHIIII", b"XXXX", 1, 24, 600, 100, 100, 14_400, 10,
    ) + b"\x00" * (100 + 100 + 14_400 + 10)
    with pytest.raises(ValueError, match="bad magic"):
        mod.parse_ibps1_archive_bytes(bad)


def test_parse_ibps1_archive_bytes_rejects_unsupported_version():
    """Schema version != 1 must raise ValueError."""
    mod = _load_module()
    bad = struct.pack(
        "<4sBHHIIII", b"IBPS", 99, 24, 600, 100, 100, 14_400, 10,
    ) + b"\x00" * (100 + 100 + 14_400 + 10)
    with pytest.raises(ValueError, match="unsupported schema version"):
        mod.parse_ibps1_archive_bytes(bad)


def test_parse_ibps1_archive_bytes_rejects_latent_length_mismatch():
    """latent_len != num_pairs * latent_dim must raise."""
    mod = _load_module()
    # Header declares latent_len=14400 but supplies 12 (wrong)
    bad = struct.pack(
        "<4sBHHIIII", b"IBPS", 1, 24, 600, 50, 50, 12, 5,
    ) + b"\x00" * (50 + 50 + 12 + 5)
    with pytest.raises(ValueError, match="latent_len"):
        mod.parse_ibps1_archive_bytes(bad)


def test_parse_ibps1_archive_bytes_rejects_truncated_archive():
    """end_meta past the buffer end must raise.

    The canonical parser surface
    (``tac.substrates.c6_e4_mdl_ibps.archive.parse_ibps1_archive_bytes``)
    uses strict ``end_meta == len(archive_bytes)`` semantics matching
    :func:`parse_archive`, so truncated and trailing-byte schema drift
    both surface as "archive size ... != expected" errors.
    """
    mod = _load_module()
    inner = _build_synthetic_ibps1_inner()
    truncated = inner[:-100]  # chop off 100 bytes from the meta tail
    with pytest.raises(ValueError, match="archive size"):
        mod.parse_ibps1_archive_bytes(truncated)


def test_parse_ibps1_uses_canonical_header_constants():
    """parse_ibps1 must use the same IBPS1_HEADER_SIZE constant the module declares."""
    mod = _load_module()
    # 25 bytes is the canonical header size per IBPS1_HEADER_FMT
    assert mod.IBPS1_HEADER_SIZE == 25
    assert mod.IBPS1_MAGIC == b"IBPS"
    assert mod.IBPS1_HEADER_FMT == "<4sBHHIIII"


def test_load_archive_dispatches_ibps1_grammar(tmp_path: Path):
    """load_archive(grammar='ibps1') must parse via parse_ibps1_archive_bytes."""
    mod = _load_module()
    inner = _build_synthetic_ibps1_inner(
        latent_dim=24, num_pairs=600,
        encoder_blob=b"\x10" * 50,
        decoder_blob=b"\x20" * 50,
        meta_blob=b'{"x":0}',
    )
    archive_path = tmp_path / "ibps1.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("0.bin", inner)
    returned_inner, sections = mod.load_archive(archive_path, "ibps1")
    assert returned_inner == inner
    # 5 IBPS1 sections present
    assert sections["ibps1_header"][1] == 25
    assert sections["encoder_blob"][1] == 50
    assert sections["decoder_blob"][1] == 50
    assert sections["latent_blob"][1] == 14_400
    assert sections["meta_blob"][1] == len(b'{"x":0}')


def test_load_archive_ibps1_prefers_0bin_over_x():
    """When the archive contains both '0.bin' AND 'x', IBPS1 grammar reads '0.bin'."""
    mod = _load_module()
    tmp_dir = Path("/tmp")  # pytest tmp_path equivalent here; we keep it minimal
    # Use pytest tmp_path via fixture in the actual test runner; this is the
    # helper invocation pattern.


def test_read_inner_member_ibps1_grammar_prefers_0bin(tmp_path: Path):
    """_read_inner_member with grammar='ibps1' reads '0.bin' even when 'x' is present."""
    mod = _load_module()
    archive_path = tmp_path / "dual.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("x", b"WRONG_PAYLOAD_X")
        zf.writestr("0.bin", b"CORRECT_PAYLOAD_0BIN")
    bytes_for_ibps1 = mod._read_inner_member(archive_path, "ibps1")
    assert bytes_for_ibps1 == b"CORRECT_PAYLOAD_0BIN"


def test_read_inner_member_a1_grammar_prefers_x(tmp_path: Path):
    """_read_inner_member with grammar='a1' reads 'x' (the conventional A1 member)."""
    mod = _load_module()
    archive_path = tmp_path / "dual.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("x", b"CORRECT_A1_X")
        zf.writestr("0.bin", b"WRONG_FOR_A1_0BIN")
    bytes_for_a1 = mod._read_inner_member(archive_path, "a1")
    assert bytes_for_a1 == b"CORRECT_A1_X"


def test_read_inner_member_fallback_to_0bin_when_x_absent(tmp_path: Path):
    """When 'x' is absent and only '0.bin' present, A1 grammar still reads '0.bin'."""
    mod = _load_module()
    archive_path = tmp_path / "only_0bin.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("0.bin", b"FALLBACK_PAYLOAD")
    bytes_for_a1 = mod._read_inner_member(archive_path, "a1")
    assert bytes_for_a1 == b"FALLBACK_PAYLOAD"


def test_ibps1_grammar_real_c6_5ep_archive_if_present():
    """If the real C6 5ep archive is present, parse it and verify section
    sizes match the substrate's archive.py contract (latent_dim=24, num_pairs=600).

    This is the empirical regression test that pins the IBPS1 parser
    against a REAL trained substrate archive (no synthetic bytes).
    """
    mod = _load_module()
    real_archive = (
        REPO
        / "experiments"
        / "results"
        / "lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T144949Z__smoke__5ep_modal"
        / "harvested_artifacts"
        / "lane_substrate_c6_e4_mdl_ibps_results"
        / "output"
        / "archive.zip"
    )
    if not real_archive.exists():
        pytest.skip("C6 5ep real archive not present in this checkout")
    inner, sections = mod.load_archive(real_archive, "ibps1")
    # Canonical C6 substrate config: latent_dim=24, num_pairs=600
    assert sections["ibps1_header"][1] == 25
    # latent_dim * num_pairs = 24 * 600 = 14400
    assert sections["latent_blob"][1] == 14_400
    # All sections must add up to len(inner)
    end_meta = (
        sections["meta_blob"][0] + sections["meta_blob"][1]
    )
    assert end_meta == len(inner)


def test_aggregate_mdl_with_ibps1_synthetic_sections():
    """Aggregation works with the IBPS1 5-section layout."""
    mod = _load_module()
    # Mirror the C6 5ep archive's section sizes
    r = _make_synthetic_archive_result(
        mod,
        sections_lengths={
            "ibps1_header": 25,
            "encoder_blob": 65_000,
            "decoder_blob": 145_000,
            "latent_blob": 14_400,
            "meta_blob": 500,
        },
        tier_b_fracs={
            "ibps1_header": 1.0,
            "encoder_blob": 0.9,
            "decoder_blob": 0.95,
            "latent_blob": 0.8,
            "meta_blob": 1.0,
        },
        tier_a_delta=0.05,
    )
    out = mod.aggregate_mdl_estimate(r)
    expected = (
        1.0 * 25 + 0.9 * 65_000 + 0.95 * 145_000 + 0.8 * 14_400 + 1.0 * 500
    )
    assert out.mdl_scorer_extracted_bytes_lo == pytest.approx(expected)


def test_ibps1_grammar_recognized_in_cli_grammar_choices_help(capsys):
    """The CLI --grammar help string mentions 'ibps1' so operators discover it."""
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod.main(["--help"])
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "ibps1" in combined
    assert "--include-section" in combined
    assert "--exclude-section" in combined
    assert "--include-provenance-sections" in combined
    assert "--byte-offset" in combined
    assert "--significance-threshold" in combined
    assert "--pair-indices-file" in combined
    assert "--allow-generic-whole-blob" in combined
    assert "--scorer-batch-size" in combined


# ----------------------------------------------------------------------
# Catalog #227 Tier C wire-in: substrate-class discrimination
# ----------------------------------------------------------------------


def _build_c6_5ep_tier_c_rows(mod):
    """Build canonical 8-row Tier C schema matching the C6 5ep anchor."""
    return [
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.001,
                        delta_seg=0.0, delta_pose=-0.003,
                        delta_score_components=-0.005, elapsed_seconds=23.0),
        mod.TierCResult(target="latents", noise_sigma_relative=0.001,
                        delta_seg=0.0, delta_pose=0.0005,
                        delta_score_components=0.0008, elapsed_seconds=22.5),
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.01,
                        delta_seg=0.0, delta_pose=-0.048,
                        delta_score_components=-0.0745, elapsed_seconds=20.9),
        mod.TierCResult(target="latents", noise_sigma_relative=0.01,
                        delta_seg=0.0, delta_pose=-0.0012,
                        delta_score_components=-0.0018, elapsed_seconds=21.1),
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.1,
                        delta_seg=0.0, delta_pose=-0.041,
                        delta_score_components=-0.0639, elapsed_seconds=20.9),
        mod.TierCResult(target="latents", noise_sigma_relative=0.1,
                        delta_seg=0.0, delta_pose=0.004,
                        delta_score_components=0.006, elapsed_seconds=20.7),
        mod.TierCResult(target="state_dict", noise_sigma_relative=1.0,
                        delta_seg=0.0, delta_pose=95.5,
                        delta_score_components=27.8, elapsed_seconds=20.9),
        mod.TierCResult(target="latents", noise_sigma_relative=1.0,
                        delta_seg=0.0, delta_pose=-0.0014,
                        delta_score_components=-0.00213, elapsed_seconds=20.5),
    ]


def test_aggregate_tier_c_empty_when_no_rows():
    """Tier C fields stay at 0.0 / "" when tier_c is empty."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 1000}, {"x": 0.5})
    out = mod.aggregate_mdl_estimate(r)
    assert out.mdl_tier_c_density_estimate == 0.0
    assert out.mdl_tier_c_substrate_class_verdict == ""
    assert out.mdl_tier_c_curve_knee_signal == 0.0
    assert out.mdl_tier_c_latent_sigma1_delta == 0.0


def test_aggregate_tier_c_c6_5ep_anchor_classifies_across_class():
    """The C6 5ep empirical anchor's Tier C numbers should classify as
    across_class with composite density < 0.30."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100}, {"x": 0.5})
    r.tier_c = _build_c6_5ep_tier_c_rows(mod)
    out = mod.aggregate_mdl_estimate(r)
    assert out.mdl_tier_c_substrate_class_verdict == "across_class"
    assert out.mdl_tier_c_density_estimate < 0.30
    # Latent σ=1 |Δ| should be ~0.00213 (well below 0.05 within-class threshold)
    assert out.mdl_tier_c_latent_sigma1_delta < 0.05
    # Curve knee should be ~0.857 (knee+plateau signature, not monotonic growth)
    assert out.mdl_tier_c_curve_knee_signal < 1.5


def test_aggregate_tier_c_synthetic_within_class_signature():
    """A synthetic Tier C with HIGH latent_sigma1 AND high curve_knee should
    classify as within_class."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100}, {"x": 0.5})
    # Within-class signature: latents catastrophic at σ=1.0 + monotonic growth.
    r.tier_c = [
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.001,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=0.0, elapsed_seconds=1.0),
        mod.TierCResult(target="latents", noise_sigma_relative=0.001,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=0.01, elapsed_seconds=1.0),
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.01,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=0.1, elapsed_seconds=1.0),
        mod.TierCResult(target="latents", noise_sigma_relative=0.01,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=0.1, elapsed_seconds=1.0),
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.1,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=2.0,  # monotonic large growth
                        elapsed_seconds=1.0),
        mod.TierCResult(target="latents", noise_sigma_relative=0.1,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=1.0, elapsed_seconds=1.0),
        mod.TierCResult(target="state_dict", noise_sigma_relative=1.0,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=20.0, elapsed_seconds=1.0),
        mod.TierCResult(target="latents", noise_sigma_relative=1.0,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=5.0,  # catastrophic latent damage
                        elapsed_seconds=1.0),
    ]
    out = mod.aggregate_mdl_estimate(r)
    assert out.mdl_tier_c_substrate_class_verdict == "within_class"
    assert out.mdl_tier_c_density_estimate >= 0.70
    # Latent σ=1 |Δ| = 5.0 (well above 0.05)
    assert out.mdl_tier_c_latent_sigma1_delta == 5.0
    # Curve knee = |0.1| / |0.01| = 2.0 / 0.1 = 20.0 (>>1, monotonic linear)
    assert out.mdl_tier_c_curve_knee_signal == pytest.approx(20.0)


def test_aggregate_tier_c_indeterminate_band():
    """Density between 0.30 and 0.70 → indeterminate verdict."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100}, {"x": 0.5})
    # Marginal signature: latent_sigma1 = 0.05 (boundary); curve_knee = 1.0.
    # latent_component = 0.05/(0.05+0.05) = 0.5
    # knee_component = 1.0/(1.0+3.0) = 0.25
    # density = 0.375 → indeterminate
    r.tier_c = [
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.01,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=0.1, elapsed_seconds=1.0),
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.1,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=0.1, elapsed_seconds=1.0),
        mod.TierCResult(target="latents", noise_sigma_relative=1.0,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=0.05, elapsed_seconds=1.0),
    ]
    out = mod.aggregate_mdl_estimate(r)
    assert out.mdl_tier_c_substrate_class_verdict == "indeterminate"
    assert 0.30 < out.mdl_tier_c_density_estimate < 0.70


def test_aggregate_tier_c_handles_none_delta():
    """Rows with delta_score_components=None are silently skipped (e.g.
    inflate failure during ablation)."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100}, {"x": 0.5})
    r.tier_c = [
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.01,
                        delta_seg=None, delta_pose=None,
                        delta_score_components=None,  # inflate failure
                        elapsed_seconds=1.0),
        mod.TierCResult(target="latents", noise_sigma_relative=1.0,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=-0.002,
                        elapsed_seconds=1.0),
    ]
    out = mod.aggregate_mdl_estimate(r)
    # latent_sigma1 = 0.002; curve_knee = 0 (no sd@0.01)
    assert out.mdl_tier_c_latent_sigma1_delta == pytest.approx(0.002)
    assert out.mdl_tier_c_curve_knee_signal == 0.0


def test_aggregate_tier_c_uses_abs_for_deltas():
    """Tier C signals use |Δscore_components|; sign does not matter for
    substrate-class discrimination."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100}, {"x": 0.5})
    r.tier_c = [
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.01,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=-0.5, elapsed_seconds=1.0),
        mod.TierCResult(target="state_dict", noise_sigma_relative=0.1,
                        delta_seg=0.0, delta_pose=0.0,
                        delta_score_components=+1.5, elapsed_seconds=1.0),
    ]
    out = mod.aggregate_mdl_estimate(r)
    # curve_knee = |+1.5| / |-0.5| = 3.0
    assert out.mdl_tier_c_curve_knee_signal == pytest.approx(3.0)


def test_aggregate_tier_c_json_round_trip_preserves_fields(tmp_path: Path):
    """The new Tier C fields survive asdict() / json.dump() serialization."""
    mod = _load_module()
    r = _make_synthetic_archive_result(mod, {"x": 100}, {"x": 0.5})
    r.tier_c = _build_c6_5ep_tier_c_rows(mod)
    out = mod.aggregate_mdl_estimate(r)
    from dataclasses import asdict
    d = asdict(out)
    assert "mdl_tier_c_density_estimate" in d
    assert "mdl_tier_c_substrate_class_verdict" in d
    assert "mdl_tier_c_curve_knee_signal" in d
    assert "mdl_tier_c_latent_sigma1_delta" in d
    assert d["mdl_tier_c_substrate_class_verdict"] == "across_class"


def test_aggregate_tier_c_real_c6_5ep_archive_if_present():
    """If the live C6 5ep Tier C JSON is present, verify the wire-in
    reproduces the published verdict."""
    mod = _load_module()
    p = REPO / "experiments" / "results" / (
        "mdl_ablation_z1_c6_5ep_tier_c_20260514T171705Z_v2_subagent"
    ) / "c6_ibps1_5ep_mdl_ablation.json"
    if not p.is_file():
        pytest.skip("C6 5ep Tier C result not present")
    data = json.loads(p.read_text(encoding="utf-8"))
    r = mod.ArchiveAblationResult(
        archive_name=data["archive_name"],
        archive_path=data["archive_path"],
        archive_sha256=data["archive_sha256"],
        archive_size_bytes=data["archive_size_bytes"],
        grammar=data["grammar"],
        device=data["device"],
        pair_samples=data["pair_samples"],
        baseline_seg=data["baseline_seg"],
        baseline_pose=data["baseline_pose"],
        baseline_score_components=data["baseline_score_components"],
    )
    r.tier_c = [mod.TierCResult(**row) for row in data["tier_c"]]
    out = mod.aggregate_mdl_estimate(r)
    # Per project memo: latent_sigma1 = 0.00213
    assert out.mdl_tier_c_latent_sigma1_delta == pytest.approx(0.00213, abs=1e-4)
    # Per project memo: curve_knee = 0.0639 / 0.0745 ≈ 0.857
    assert out.mdl_tier_c_curve_knee_signal == pytest.approx(0.857, abs=0.01)
    # Per project memo: ACROSS-CLASS verdict (5ep architectural level)
    assert out.mdl_tier_c_substrate_class_verdict == "across_class"
