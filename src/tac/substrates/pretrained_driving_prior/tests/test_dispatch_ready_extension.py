# SPDX-License-Identifier: MIT
"""WAVE-3-DP1-DISPATCH-READY-EXTENSION test suite.

Per task description Top-3 OP-ROUTABLES #2 + #3 of
``feedback_dp1_paired_smoke_dispatch_pre_authorization_checklist_landed_20260520.md``.

Tests cover:

* Trainer argparse + Tier 1 manifest wiring (7 new flags).
* ``_resolve_procedural_seed_bytes`` precedence (null-exploit / hex / default).
* ``_apply_procedural_codebook_replacement`` invariants (bytes saved >0,
  predicted ΔS matches canonical equation #26, provenance manifest emit).
* ``derive_dashcam_codebook_from_seed`` canonical shapes + determinism +
  byte-mutation distinguishing-feature (Catalog #272).
* ``parse_archive_procedural_aware`` routing (canonical fallthrough +
  procedural variant detection + per-pair residual preserved).
* Full encode→decode roundtrip: pack_archive → compose_with_procedural_codebook
  → parse_archive_procedural_aware → DashcamCodebook canonical contract.

Catalog discipline: Catalog #220 + #272 + #287 + #295 + #305 + #323 + #324
+ #344 cross-referenced; tests carry zero placeholder rationales.

6-hook wire-in declaration per Catalog #125:

* hook #1 sensitivity-map = N/A (test suite)
* hook #2 Pareto constraint = N/A
* hook #3 bit-allocator = N/A
* hook #4 cathedral autopilot dispatch = N/A (test suite)
* hook #5 continual-learning posterior = N/A
* hook #6 probe-disambiguator = ACTIVE (the 3-recipe contrast tests IS the
  probe disambiguator at the test surface)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Sister of src/tac/substrates/pretrained_driving_prior/tests/test_trainer_dataset_args.py
# import pattern — use path-injection so both test files share the same module
# identity for the substrate_registry @register_substrate decorator.
sys.path.insert(0, "experiments")
import train_substrate_pretrained_driving_prior as _trainer  # noqa: E402

TIER_1_OPERATOR_REQUIRED_FLAGS = _trainer.TIER_1_OPERATOR_REQUIRED_FLAGS
_apply_procedural_codebook_replacement = (
    _trainer._apply_procedural_codebook_replacement
)
_resolve_procedural_seed_bytes = _trainer._resolve_procedural_seed_bytes
_write_runtime = _trainer._write_runtime
build_argparser = _trainer.build_argparser
from tac.substrates.pretrained_driving_prior.archive import pack_archive
from tac.substrates.pretrained_driving_prior.codebook import (
    LANE_CURVATURE_PCA_SHAPE,
    ROAD_PLANE_BASIS_SHAPE,
    SKY_HORIZON_PROFILE_SHAPE,
    VEHICLE_APPEARANCE_BASIS_SHAPE,
    DashcamCodebook,
    validate_codebook,
)
from tac.substrates.pretrained_driving_prior.distillation_procedural_variant import (
    compose_with_procedural_codebook,
)
from tac.substrates.pretrained_driving_prior.procedural_codebook_inflate import (
    PROCEDURAL_CODEBOOK_META_FLAG,
    derive_dashcam_codebook_from_seed,
    is_procedural_codebook_variant_archive,
    parse_archive_procedural_aware,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_canonical_book() -> DashcamCodebook:
    """Construct a canonical DashcamCodebook with valid shapes/dtypes."""
    rng = np.random.default_rng(42)
    return DashcamCodebook(
        road_plane_basis=rng.integers(
            -128, 127, ROAD_PLANE_BASIS_SHAPE, dtype=np.int8
        ),
        sky_horizon_profile=rng.integers(
            -128, 127, SKY_HORIZON_PROFILE_SHAPE, dtype=np.int8
        ),
        lane_curvature_pca=rng.standard_normal(LANE_CURVATURE_PCA_SHAPE).astype(
            np.float16
        ),
        vehicle_appearance_basis=rng.integers(
            -128, 127, VEHICLE_APPEARANCE_BASIS_SHAPE, dtype=np.int8
        ),
        metadata={
            "road_plane_scale": 0.5,
            "sky_horizon_scale": 0.3,
            "vehicle_scale": 0.4,
            "dataset_provenance": "comma2k19_test",
            "distillation_version": "test_v1",
            "license_tags": ["MIT"],
        },
    )


def _make_minimal_renderer_state_dict() -> dict[str, torch.Tensor]:
    """Construct a minimal renderer state_dict for archive packing."""
    return {
        "net.0.linear.weight": torch.zeros(64, 4),
        "net.0.linear.bias": torch.zeros(64),
        "net.1.linear.weight": torch.zeros(64, 64),
        "net.1.linear.bias": torch.zeros(64),
        "net.2.linear.weight": torch.zeros(3, 64),
        "net.2.linear.bias": torch.zeros(3),
    }


# ---------------------------------------------------------------------------
# Trainer flag wiring tests
# ---------------------------------------------------------------------------


def test_tier1_manifest_carries_7_procedural_flags() -> None:
    """Catalog #151: all 7 new procedural flags must appear in the Tier 1 manifest."""
    expected = {
        "--enable-procedural-codebook-replacement",
        "--procedural-codebook-seed-hex",
        "--procedural-codebook-generator-kind",
        "--procedural-codebook-null-exploit-control",
        "--procedural-codebook-validate-domain",
        "--procedural-variant-provenance-path",
        "--procedural-variant-distillation-skip",
    }
    actual = set(TIER_1_OPERATOR_REQUIRED_FLAGS.keys())
    missing = expected - actual
    assert not missing, f"Tier 1 manifest missing procedural flags: {missing}"


def test_tier1_manifest_env_vars_canonical() -> None:
    """Catalog #151 + design memo §4: env-var names must match the canonical recipe."""
    expected_envs = {
        "--enable-procedural-codebook-replacement": "DPP_PROCEDURAL_CODEBOOK_REPLACEMENT",
        "--procedural-codebook-seed-hex": "DPP_PROCEDURAL_CODEBOOK_SEED_HEX",
        "--procedural-codebook-generator-kind": "DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND",
        "--procedural-codebook-null-exploit-control": (
            "DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL"
        ),
        "--procedural-codebook-validate-domain": (
            "DPP_PROCEDURAL_CODEBOOK_VALIDATE_DOMAIN"
        ),
        "--procedural-variant-provenance-path": (
            "DPP_PROCEDURAL_VARIANT_PROVENANCE_PATH"
        ),
        "--procedural-variant-distillation-skip": (
            "DPP_PROCEDURAL_VARIANT_DISTILLATION_SKIP"
        ),
    }
    for flag, expected_env in expected_envs.items():
        actual_env = TIER_1_OPERATOR_REQUIRED_FLAGS[flag]["env"]
        assert actual_env == expected_env, (
            f"{flag} env={actual_env!r} != expected {expected_env!r}"
        )


def test_argparser_accepts_all_procedural_flags() -> None:
    """Catalog #12 preflight_arity sister: argparse must wire every Tier-1 flag."""
    parser = build_argparser()
    args = parser.parse_args(
        [
            "--enable-procedural-codebook-replacement",
            "--procedural-codebook-seed-hex",
            "ab" * 32,
            "--procedural-codebook-generator-kind",
            "pcg64",
            "--procedural-variant-distillation-skip",
        ]
    )
    assert args.enable_procedural_codebook_replacement is True
    assert args.procedural_codebook_seed_hex == "ab" * 32
    assert args.procedural_codebook_generator_kind == "pcg64"
    assert args.procedural_variant_distillation_skip is True
    # default-True flag remains True without explicit flag.
    assert args.procedural_codebook_validate_domain is True


# ---------------------------------------------------------------------------
# _resolve_procedural_seed_bytes precedence tests
# ---------------------------------------------------------------------------


def _make_namespace(**overrides) -> argparse.Namespace:
    """Construct an argparse Namespace with sensible defaults for these tests."""
    defaults = dict(
        seed=0xDA5C,
        enable_procedural_codebook_replacement=True,
        procedural_codebook_seed_hex="",
        procedural_codebook_generator_kind="pcg64",
        procedural_codebook_null_exploit_control=False,
        procedural_codebook_validate_domain=True,
        procedural_variant_provenance_path="",
        procedural_variant_distillation_skip=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_seed_null_exploit_returns_32_zero_bytes() -> None:
    args = _make_namespace(procedural_codebook_null_exploit_control=True)
    seed = _resolve_procedural_seed_bytes(args)
    assert seed == b"\x00" * 32


def test_resolve_seed_operator_supplied_hex_parses() -> None:
    args = _make_namespace(procedural_codebook_seed_hex="ab" * 32)
    seed = _resolve_procedural_seed_bytes(args)
    assert seed == b"\xab" * 32


def test_resolve_seed_default_derives_from_seed_arg_deterministically() -> None:
    args = _make_namespace(seed=42)
    seed_a = _resolve_procedural_seed_bytes(args)
    seed_b = _resolve_procedural_seed_bytes(args)
    assert seed_a == seed_b
    assert len(seed_a) == 32
    # Different seeds produce different bytes.
    args2 = _make_namespace(seed=43)
    seed_c = _resolve_procedural_seed_bytes(args2)
    assert seed_c != seed_a


def test_resolve_seed_rejects_odd_hex_length() -> None:
    args = _make_namespace(procedural_codebook_seed_hex="abc")
    with pytest.raises(SystemExit, match="domain-of-validity"):
        _resolve_procedural_seed_bytes(args)


def test_resolve_seed_rejects_non_hex_chars() -> None:
    args = _make_namespace(procedural_codebook_seed_hex="zzzzzzzz" * 8)
    with pytest.raises(SystemExit, match="non-hex"):
        _resolve_procedural_seed_bytes(args)


def test_resolve_seed_null_exploit_overrides_hex() -> None:
    """Null exploit takes precedence over operator-supplied hex (recipe #3 semantics)."""
    args = _make_namespace(
        procedural_codebook_null_exploit_control=True,
        procedural_codebook_seed_hex="ab" * 32,
    )
    seed = _resolve_procedural_seed_bytes(args)
    assert seed == b"\x00" * 32


# ---------------------------------------------------------------------------
# _apply_procedural_codebook_replacement invariants
# ---------------------------------------------------------------------------


def _build_canonical_archive_for_test() -> bytes:
    book = _make_canonical_book()
    sd = _make_minimal_renderer_state_dict()
    residual = bytes(4 * 12)
    return pack_archive(
        book,
        sd,
        residual,
        {"residual_int8_scale": 64.0},
        num_pairs=4,
        output_height=384,
        output_width=512,
        per_pair_bytes=12,
    )


def test_apply_procedural_replacement_reduces_archive_bytes(tmp_path: Path) -> None:
    canonical = _build_canonical_archive_for_test()
    args = _make_namespace()
    seed = _resolve_procedural_seed_bytes(args)
    new_archive = _apply_procedural_codebook_replacement(
        args=args,
        canonical_archive_bytes=canonical,
        seed_bytes=seed,
        output_dir=tmp_path,
    )
    assert len(new_archive) < len(canonical)
    # Expected savings ≈ 4-10KB.
    assert (len(canonical) - len(new_archive)) > 1000


def test_apply_procedural_replacement_writes_provenance_json(tmp_path: Path) -> None:
    canonical = _build_canonical_archive_for_test()
    args = _make_namespace()
    seed = _resolve_procedural_seed_bytes(args)
    _apply_procedural_codebook_replacement(
        args=args,
        canonical_archive_bytes=canonical,
        seed_bytes=seed,
        output_dir=tmp_path,
    )
    provenance_path = tmp_path / "procedural_variant_provenance.json"
    assert provenance_path.exists()
    payload = json.loads(provenance_path.read_text())
    # Catalog #323 canonical Provenance: non-promotable markers MUST be set.
    assert payload["score_claim"] is False
    assert payload["score_claim_valid"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["axis_tag"] == "[predicted]"
    assert payload["evidence_grade"] == "[predicted]"
    # Catalog #324 predicted_band_validation_status: pending_post_training.
    assert payload["predicted_band_validation_status"] == "pending_post_training"
    # Canonical equation #26 cross-reference.
    assert payload["canonical_equation_id"] == (
        "procedural_codebook_from_seed_compression_savings_v1"
    )
    assert payload["seed_size_bytes"] == 32
    assert payload["seed_sha256"] == hashlib.sha256(seed).hexdigest()


def test_procedural_provenance_uses_recipe_lane_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _build_canonical_archive_for_test()
    args = _make_namespace()
    seed = _resolve_procedural_seed_bytes(args)
    monkeypatch.setenv("DPP_LANE_ID", "lane_dp1_recipe_test")
    _apply_procedural_codebook_replacement(
        args=args,
        canonical_archive_bytes=canonical,
        seed_bytes=seed,
        output_dir=tmp_path,
    )
    payload = json.loads((tmp_path / "procedural_variant_provenance.json").read_text())
    assert payload["lane_id"] == "lane_dp1_recipe_test"
    assert payload["generator_kind"] == "pcg64"
    assert payload["null_exploit_control"] is False


def test_apply_procedural_replacement_predicted_delta_s_matches_canonical_equation_26(
    tmp_path: Path,
) -> None:
    canonical = _build_canonical_archive_for_test()
    args = _make_namespace()
    seed = _resolve_procedural_seed_bytes(args)
    _apply_procedural_codebook_replacement(
        args=args,
        canonical_archive_bytes=canonical,
        seed_bytes=seed,
        output_dir=tmp_path,
    )
    payload = json.loads(
        (tmp_path / "procedural_variant_provenance.json").read_text()
    )
    bytes_saved = payload["archive_bytes_saved"]
    expected_delta_s = -25.0 * bytes_saved / 37_545_489.0
    assert abs(payload["predicted_delta_s_contest_rate"] - expected_delta_s) < 1e-9


def test_apply_procedural_replacement_custom_provenance_path(tmp_path: Path) -> None:
    canonical = _build_canonical_archive_for_test()
    custom_path = tmp_path / "custom_dir" / "my_provenance.json"
    args = _make_namespace(procedural_variant_provenance_path=str(custom_path))
    seed = _resolve_procedural_seed_bytes(args)
    _apply_procedural_codebook_replacement(
        args=args,
        canonical_archive_bytes=canonical,
        seed_bytes=seed,
        output_dir=tmp_path,
    )
    assert custom_path.exists()


# ---------------------------------------------------------------------------
# derive_dashcam_codebook_from_seed canonical shapes + determinism + Catalog #272
# ---------------------------------------------------------------------------


def test_derive_codebook_canonical_shapes_and_dtypes() -> None:
    seed = b"\xab" * 32
    book = derive_dashcam_codebook_from_seed(seed)
    assert isinstance(book, DashcamCodebook)
    assert book.road_plane_basis.shape == ROAD_PLANE_BASIS_SHAPE
    assert book.road_plane_basis.dtype == np.int8
    assert book.sky_horizon_profile.shape == SKY_HORIZON_PROFILE_SHAPE
    assert book.sky_horizon_profile.dtype == np.int8
    assert book.lane_curvature_pca.shape == LANE_CURVATURE_PCA_SHAPE
    assert book.lane_curvature_pca.dtype == np.float16
    assert book.vehicle_appearance_basis.shape == VEHICLE_APPEARANCE_BASIS_SHAPE
    assert book.vehicle_appearance_basis.dtype == np.int8
    # validate_codebook MUST accept the derived codebook.
    validate_codebook(book)


def test_derive_codebook_deterministic() -> None:
    seed = b"\xcd" * 32
    a = derive_dashcam_codebook_from_seed(seed)
    b = derive_dashcam_codebook_from_seed(seed)
    assert np.array_equal(a.road_plane_basis, b.road_plane_basis)
    assert np.array_equal(a.sky_horizon_profile, b.sky_horizon_profile)
    assert a.lane_curvature_pca.tobytes() == b.lane_curvature_pca.tobytes()
    assert np.array_equal(a.vehicle_appearance_basis, b.vehicle_appearance_basis)


def test_derive_codebook_byte_mutation_distinguishing_feature_catalog_272() -> None:
    """Catalog #272: flipping any seed byte MUST change the derived codebook."""
    seed = bytes(range(32))
    book_orig = derive_dashcam_codebook_from_seed(seed)
    mutated = bytearray(seed)
    mutated[5] = (mutated[5] + 1) & 0xFF
    book_mut = derive_dashcam_codebook_from_seed(bytes(mutated))
    # All 4 sections must differ (sha256 cascade ensures this).
    assert not np.array_equal(book_orig.road_plane_basis, book_mut.road_plane_basis)
    assert not np.array_equal(
        book_orig.sky_horizon_profile, book_mut.sky_horizon_profile
    )
    assert (
        book_orig.lane_curvature_pca.tobytes()
        != book_mut.lane_curvature_pca.tobytes()
    )
    assert not np.array_equal(
        book_orig.vehicle_appearance_basis, book_mut.vehicle_appearance_basis
    )


def test_derive_codebook_pcg64_default_matches_explicit() -> None:
    seed = b"\xef" * 32
    default_book = derive_dashcam_codebook_from_seed(seed)
    explicit_book = derive_dashcam_codebook_from_seed(seed, generator_kind="pcg64")
    assert np.array_equal(
        default_book.road_plane_basis, explicit_book.road_plane_basis
    )


def test_derive_codebook_different_generators_produce_different_arrays() -> None:
    seed = b"\xff" * 32
    pcg64 = derive_dashcam_codebook_from_seed(seed, generator_kind="pcg64")
    xorshift = derive_dashcam_codebook_from_seed(seed, generator_kind="xorshift")
    assert not np.array_equal(
        pcg64.road_plane_basis, xorshift.road_plane_basis
    )


# ---------------------------------------------------------------------------
# parse_archive_procedural_aware roundtrip + detection
# ---------------------------------------------------------------------------


def test_is_procedural_codebook_variant_archive_detector() -> None:
    assert is_procedural_codebook_variant_archive(
        {PROCEDURAL_CODEBOOK_META_FLAG: True}
    )
    assert not is_procedural_codebook_variant_archive(
        {PROCEDURAL_CODEBOOK_META_FLAG: False}
    )
    assert not is_procedural_codebook_variant_archive({})
    assert not is_procedural_codebook_variant_archive({"other_field": True})


def test_parse_canonical_archive_falls_through(tmp_path: Path) -> None:
    """When meta does NOT declare procedural variant, parse_archive is called."""
    book = _make_canonical_book()
    canonical = pack_archive(
        book,
        _make_minimal_renderer_state_dict(),
        bytes(4 * 12),
        {"a": 1},
        num_pairs=4,
        output_height=384,
        output_width=512,
        per_pair_bytes=12,
    )
    parsed = parse_archive_procedural_aware(canonical)
    assert not is_procedural_codebook_variant_archive(parsed.meta)
    # Original codebook bytes recovered.
    assert np.array_equal(parsed.codebook.road_plane_basis, book.road_plane_basis)
    assert np.array_equal(
        parsed.codebook.sky_horizon_profile, book.sky_horizon_profile
    )
    assert (
        parsed.codebook.lane_curvature_pca.tobytes()
        == book.lane_curvature_pca.tobytes()
    )
    assert np.array_equal(
        parsed.codebook.vehicle_appearance_basis, book.vehicle_appearance_basis
    )


def test_parse_procedural_archive_re_derives_codebook_from_seed(
    tmp_path: Path,
) -> None:
    """Full encode→decode roundtrip via compose_with_procedural_codebook."""
    book = _make_canonical_book()
    seed = b"\xab" * 32
    meta = {
        "residual_int8_scale": 64.0,
        PROCEDURAL_CODEBOOK_META_FLAG: True,
        "procedural_codebook_seed_hex": seed.hex(),
        "procedural_codebook_generator_kind": "pcg64",
    }
    canonical = pack_archive(
        book,
        _make_minimal_renderer_state_dict(),
        bytes(4 * 12),
        meta,
        num_pairs=4,
        output_height=384,
        output_width=512,
        per_pair_bytes=12,
    )
    procedural_archive = compose_with_procedural_codebook(canonical, seed)
    parsed = parse_archive_procedural_aware(procedural_archive)
    assert is_procedural_codebook_variant_archive(parsed.meta)
    # Codebook re-derived deterministically; matches direct call.
    expected_book = derive_dashcam_codebook_from_seed(seed, generator_kind="pcg64")
    assert np.array_equal(
        parsed.codebook.road_plane_basis, expected_book.road_plane_basis
    )
    assert np.array_equal(
        parsed.codebook.sky_horizon_profile, expected_book.sky_horizon_profile
    )
    # Per-pair residual preserved through composition.
    assert parsed.per_pair_residual == bytes(4 * 12)
    # Header fields preserved.
    assert parsed.num_pairs == 4
    assert parsed.output_height == 384
    assert parsed.output_width == 512


def test_parse_procedural_archive_rejects_missing_seed_hex(tmp_path: Path) -> None:
    """Defensive: procedural flag set but seed_hex missing must raise."""
    book = _make_canonical_book()
    meta = {PROCEDURAL_CODEBOOK_META_FLAG: True}  # seed_hex missing
    canonical = pack_archive(
        book,
        _make_minimal_renderer_state_dict(),
        bytes(4 * 12),
        meta,
        num_pairs=4,
        output_height=384,
        output_width=512,
        per_pair_bytes=12,
    )
    # compose_with_procedural_codebook would swap codebook; for this test
    # we directly parse a hand-crafted canonical archive whose meta lies.
    with pytest.raises(ValueError, match="seed_hex"):
        parse_archive_procedural_aware(canonical)


def test_inflate_py_routes_through_procedural_aware_parser() -> None:
    """Source-level smoke: inflate.py imports parse_archive_procedural_aware."""
    inflate_src = (
        Path(__file__).parent.parent / "inflate.py"
    ).read_text(encoding="utf-8")
    assert "parse_archive_procedural_aware" in inflate_src
    assert "from tac.substrates.pretrained_driving_prior.procedural_codebook_inflate" in (
        inflate_src
    )


def test_write_runtime_vendors_procedural_inflate_dependencies(tmp_path: Path) -> None:
    submission_dir = tmp_path / "submission"
    _write_runtime(submission_dir)

    expected = (
        "src/tac/substrates/pretrained_driving_prior/prior_application.py",
        "src/tac/substrates/pretrained_driving_prior/procedural_codebook_inflate.py",
        "src/tac/procedural_codebook_generator/__init__.py",
        "src/tac/procedural_codebook_generator/seed_derived_codebook.py",
    )
    for relpath in expected:
        assert (submission_dir / relpath).is_file(), relpath

    env = dict(os.environ)
    env["PYTHONPATH"] = str(submission_dir / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tac.substrates.pretrained_driving_prior.inflate import "
                "inflate_one_video; "
                "from tac.substrates.pretrained_driving_prior."
                "procedural_codebook_inflate import parse_archive_procedural_aware; "
                "from tac.procedural_codebook_generator.seed_derived_codebook "
                "import derive_codebook_from_seed"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
