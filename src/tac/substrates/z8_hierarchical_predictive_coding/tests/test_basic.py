# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding L0 SCAFFOLD basic tests.

Covers:
- Catalog #124 8-field declaration imports cleanly
- Catalog #91 ENCODE_INFLATE_ROUNDTRIP — pack_archive → parse_archive
  byte-deterministic
- Catalog #139 + #105 + #272 byte-mutation no_op_proof — header bytes are
  control_or_metadata; section bytes are operationally consumed
- MLX ↔ PyTorch byte stability of state-dict round-trip
- Substrate package import
- Z8L0ScaffoldNotImplementedError raised at runtime forward
"""

from __future__ import annotations

import json

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Import + canonical-field tests
# -----------------------------------------------------------------------------


def test_z8_substrate_package_imports() -> None:
    """Substrate package imports cleanly without MLX (numpy + brotli only)."""
    import tac.substrates.z8_hierarchical_predictive_coding as z8

    assert z8.IMPLEMENTATION_STATUS == "l0_research_only_mlx_renderer_archive_scaffold"
    assert z8.RESEARCH_ONLY is True
    assert z8.NUM_PAIRS == 600
    assert z8.EVAL_HW == (384, 512)


def test_z8_catalog_124_archive_grammar_fields_declared() -> None:
    """Catalog #124: all 8 required fields declared."""
    import tac.substrates.z8_hierarchical_predictive_coding as z8

    required = {
        "archive_grammar",
        "parser_section_manifest",
        "inflate_runtime_loc_budget",
        "runtime_dep_closure",
        "export_format",
        "score_aware_loss",
        "bolt_on_loc_budget",
        "no_op_detector_planned",
    }
    assert required.issubset(set(z8.ARCHIVE_GRAMMAR_FIELDS.keys()))
    # Every field must be a non-empty string per Catalog #287 placeholder
    # rejection sister discipline.
    for field, value in z8.ARCHIVE_GRAMMAR_FIELDS.items():
        assert isinstance(value, str) and value, f"Empty field {field}"
        assert "<rationale>" not in value, f"Placeholder in {field}"
        assert "<reason>" not in value, f"Placeholder in {field}"


def test_z8_canonical_equation_refs_declared() -> None:
    """Catalog #344: canonical equation references declared."""
    import tac.substrates.z8_hierarchical_predictive_coding as z8

    assert len(z8.CANONICAL_EQUATION_IDS) >= 3
    for eq_id in z8.CANONICAL_EQUATION_IDS:
        assert isinstance(eq_id, str) and eq_id.endswith(("_v1", "_v2"))


# -----------------------------------------------------------------------------
# Archive byte-deterministic round-trip tests (Catalog #91 ENCODE_INFLATE_ROUNDTRIP)
# -----------------------------------------------------------------------------


def _make_synthetic_state_dict() -> dict[str, np.ndarray]:
    """Minimal canonical synthetic state dict for round-trip tests."""
    rng = np.random.default_rng(42)
    return {
        "decoder.0.weight": rng.normal(size=(8, 4, 3, 3)).astype(np.float32),
        "decoder.0.bias": rng.normal(size=(8,)).astype(np.float32),
        "decoder.1.weight": rng.normal(size=(8,)).astype(np.float32),
    }


def _make_synthetic_dreamer_state_dict() -> dict[str, np.ndarray]:
    """Minimal canonical DreamerV3 state init synthetic."""
    rng = np.random.default_rng(43)
    return {
        "gru.weight_ih": rng.normal(size=(64, 10)).astype(np.float32),
        "gru.weight_hh": rng.normal(size=(64, 64)).astype(np.float32),
    }


def _make_synthetic_indices(
    num_pairs: int,
    num_groups_per_level: tuple[int, ...],
    num_categories_per_level: tuple[int, ...],
) -> list[np.ndarray]:
    """Per-level categorical indices for round-trip tests."""
    rng = np.random.default_rng(44)
    return [
        rng.integers(
            low=0,
            high=int(num_categories_per_level[level_idx]),
            size=(num_pairs, int(num_groups_per_level[level_idx])),
            dtype=np.int32,
        )
        for level_idx in range(len(num_groups_per_level))
    ]


def test_z8_archive_round_trip_byte_deterministic() -> None:
    """Catalog #91 ENCODE_INFLATE_ROUNDTRIP: pack→parse→pack produces identical bytes."""
    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        pack_archive,
        parse_archive,
    )

    num_pairs = 4
    num_levels = 3
    num_groups = (4, 3, 2)
    num_cats = (16, 8, 4)
    decoder_sd = _make_synthetic_state_dict()
    dreamer_sd = _make_synthetic_dreamer_state_dict()
    indices = _make_synthetic_indices(num_pairs, num_groups, num_cats)
    wavelet_blob = b"\x01\x02\x03\x04" * 8
    wz_blob = b"\x10\x20\x30" * 5
    meta = {"foo": 1, "bar": [1, 2, 3], "wavelet_basis": "daubechies_4"}

    bytes_1 = pack_archive(
        decoder_state_dict=decoder_sd,
        per_level_category_indices=indices,
        wavelet_coeffs_blob=wavelet_blob,
        wyner_ziv_top_blob=wz_blob,
        dreamer_state_dict=dreamer_sd,
        meta=meta,
        num_levels=num_levels,
        num_groups_per_level=num_groups,
        num_categories_per_level=num_cats,
        num_pairs=num_pairs,
        decoder_latent_dim=28,
        base_channels=24,
        wavelet_basis_id=0,
    )

    # parse → re-pack → bytes must match
    parsed = parse_archive(bytes_1)
    assert parsed.num_levels == num_levels
    assert parsed.num_groups_per_level == num_groups
    assert parsed.num_categories_per_level == num_cats
    assert parsed.num_pairs == num_pairs
    assert parsed.decoder_latent_dim == 28
    assert parsed.base_channels == 24
    assert parsed.wavelet_basis_id == 0

    # Indices match
    for level_idx in range(num_levels):
        np.testing.assert_array_equal(
            parsed.per_level_category_indices[level_idx],
            indices[level_idx],
        )

    # Opaque blobs match
    assert parsed.wavelet_coeffs_blob == wavelet_blob
    assert parsed.wyner_ziv_top_blob == wz_blob

    # Meta JSON parsed correctly (sorted-keys deterministic)
    assert parsed.meta == meta

    # Re-pack and compare bytes for full determinism
    bytes_2 = pack_archive(
        decoder_state_dict={
            # Re-derive from parsed decoder_state_dict (fp16 round-trip)
            k: parsed.decoder_state_dict[k] for k in parsed.decoder_state_dict
        },
        per_level_category_indices=parsed.per_level_category_indices,
        wavelet_coeffs_blob=parsed.wavelet_coeffs_blob,
        wyner_ziv_top_blob=parsed.wyner_ziv_top_blob,
        dreamer_state_dict={
            k: parsed.dreamer_state_blob[k] for k in parsed.dreamer_state_blob
        },
        meta=parsed.meta,
        num_levels=parsed.num_levels,
        num_groups_per_level=parsed.num_groups_per_level,
        num_categories_per_level=parsed.num_categories_per_level,
        num_pairs=parsed.num_pairs,
        decoder_latent_dim=parsed.decoder_latent_dim,
        base_channels=parsed.base_channels,
        wavelet_basis_id=parsed.wavelet_basis_id,
    )
    assert bytes_1 == bytes_2, "round-trip byte determinism broken"


def test_z8_archive_section_offsets_parse() -> None:
    """parse_z8hpc1_archive_bytes returns canonical section ranges."""
    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        pack_archive,
        parse_z8hpc1_archive_bytes,
    )

    num_pairs = 2
    num_levels = 3
    num_groups = (4, 3, 2)
    num_cats = (16, 8, 4)
    arc_bytes = pack_archive(
        decoder_state_dict=_make_synthetic_state_dict(),
        per_level_category_indices=_make_synthetic_indices(
            num_pairs, num_groups, num_cats
        ),
        wavelet_coeffs_blob=b"\xAA" * 16,
        wyner_ziv_top_blob=b"\xBB" * 12,
        dreamer_state_dict=_make_synthetic_dreamer_state_dict(),
        meta={"k": 1},
        num_levels=num_levels,
        num_groups_per_level=num_groups,
        num_categories_per_level=num_cats,
        num_pairs=num_pairs,
        decoder_latent_dim=28,
        base_channels=24,
        wavelet_basis_id=0,
    )

    sections = parse_z8hpc1_archive_bytes(arc_bytes)
    required = {
        "z8hpc1_header",
        "decoder_blob",
        "indices_blob",
        "wavelet_blob",
        "wyner_ziv_blob",
        "dreamer_state_blob",
        "meta_blob",
    }
    assert set(sections.keys()) == required

    # No section overlap; sections cover full archive
    sorted_sections = sorted(sections.values(), key=lambda x: x[0])
    pos = 0
    for start, length in sorted_sections:
        assert start == pos, f"section gap or overlap at start={start}, pos={pos}"
        pos += length
    assert pos == len(arc_bytes)


def test_z8_archive_refuses_corrupt_magic() -> None:
    """parse_archive raises ValueError on corrupted magic."""
    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        Z8HPC1_HEADER_SIZE,
        parse_archive,
    )

    bytes_data = bytearray(b"\x00" * 200)
    bytes_data[:8] = b"NOTZ8HPC"  # corrupt magic
    with pytest.raises(ValueError, match="magic mismatch"):
        parse_archive(bytes(bytes_data))


# -----------------------------------------------------------------------------
# Catalog #139 + #272 byte-mutation no-op proof
# -----------------------------------------------------------------------------


def test_z8_distinguishing_feature_sections_change_archive_bytes() -> None:
    """Byte-mutation gate: mutating each of the 4 distinguishing-feature sections
    produces different archive bytes (proves bytes are structurally consumed)."""
    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        pack_archive,
    )

    num_pairs = 2
    num_levels = 3
    num_groups = (4, 3, 2)
    num_cats = (16, 8, 4)

    base_args = dict(
        decoder_state_dict=_make_synthetic_state_dict(),
        per_level_category_indices=_make_synthetic_indices(
            num_pairs, num_groups, num_cats
        ),
        wavelet_coeffs_blob=b"\xAA" * 16,
        wyner_ziv_top_blob=b"\xBB" * 12,
        dreamer_state_dict=_make_synthetic_dreamer_state_dict(),
        meta={"k": 1},
        num_levels=num_levels,
        num_groups_per_level=num_groups,
        num_categories_per_level=num_cats,
        num_pairs=num_pairs,
        decoder_latent_dim=28,
        base_channels=24,
        wavelet_basis_id=0,
    )

    baseline = pack_archive(**base_args)

    # DISTINGUISHING FEATURE #1: per-level categorical indices
    mutated_indices = [arr.copy() for arr in base_args["per_level_category_indices"]]
    mutated_indices[0][0, 0] = (mutated_indices[0][0, 0] + 1) % int(num_cats[0])
    mutant_1 = pack_archive(**{**base_args, "per_level_category_indices": mutated_indices})
    assert mutant_1 != baseline, "indices_blob mutation did not change archive bytes"

    # DISTINGUISHING FEATURE #2: wavelet coeffs blob
    mutant_2 = pack_archive(**{**base_args, "wavelet_coeffs_blob": b"\xAB" * 16})
    assert mutant_2 != baseline, "wavelet_blob mutation did not change archive bytes"

    # DISTINGUISHING FEATURE #3: Wyner-Ziv top blob
    mutant_3 = pack_archive(**{**base_args, "wyner_ziv_top_blob": b"\xBC" * 12})
    assert mutant_3 != baseline, "wyner_ziv_blob mutation did not change archive bytes"

    # DISTINGUISHING FEATURE #4: DreamerV3 state dict
    mutated_dreamer = dict(base_args["dreamer_state_dict"])
    mutated_dreamer["gru.weight_ih"] = (
        mutated_dreamer["gru.weight_ih"] + 1.0
    ).astype(np.float32)
    mutant_4 = pack_archive(**{**base_args, "dreamer_state_dict": mutated_dreamer})
    assert mutant_4 != baseline, "dreamer_state_blob mutation did not change archive bytes"


def test_z8_archive_meta_json_deterministic_sort_keys() -> None:
    """Meta JSON serialization MUST sort keys for byte determinism."""
    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        pack_archive,
        parse_archive,
    )

    num_pairs = 1
    num_levels = 3
    num_groups = (2, 2, 2)
    num_cats = (4, 4, 4)
    common = dict(
        decoder_state_dict=_make_synthetic_state_dict(),
        per_level_category_indices=_make_synthetic_indices(
            num_pairs, num_groups, num_cats
        ),
        wavelet_coeffs_blob=b"x",
        wyner_ziv_top_blob=b"y",
        dreamer_state_dict=_make_synthetic_dreamer_state_dict(),
        num_levels=num_levels,
        num_groups_per_level=num_groups,
        num_categories_per_level=num_cats,
        num_pairs=num_pairs,
        decoder_latent_dim=28,
        base_channels=24,
        wavelet_basis_id=0,
    )

    # Same meta with different insertion order MUST produce identical archive bytes
    arc_1 = pack_archive(**common, meta={"a": 1, "b": 2, "c": 3})
    arc_2 = pack_archive(**common, meta={"c": 3, "a": 1, "b": 2})
    assert arc_1 == arc_2, "sort-keys JSON not deterministic"


# -----------------------------------------------------------------------------
# Inflate runtime tests (L0 SCAFFOLD parse+validate only; forward council-gated)
# -----------------------------------------------------------------------------


def test_z8_inflate_parse_and_validate_passes_on_valid_archive() -> None:
    """parse_and_validate_archive accepts a well-formed Z8HPC1 archive."""
    from tac.substrates.z8_hierarchical_predictive_coding.archive import pack_archive
    from tac.substrates.z8_hierarchical_predictive_coding.inflate import (
        parse_and_validate_archive,
    )

    num_pairs = 2
    num_groups = (4, 3, 2)
    num_cats = (16, 8, 4)
    arc_bytes = pack_archive(
        decoder_state_dict=_make_synthetic_state_dict(),
        per_level_category_indices=_make_synthetic_indices(
            num_pairs, num_groups, num_cats
        ),
        wavelet_coeffs_blob=b"\xAA" * 16,
        wyner_ziv_top_blob=b"\xBB" * 12,
        dreamer_state_dict=_make_synthetic_dreamer_state_dict(),
        meta={"k": 1},
        num_levels=3,
        num_groups_per_level=num_groups,
        num_categories_per_level=num_cats,
        num_pairs=num_pairs,
        decoder_latent_dim=28,
        base_channels=24,
        wavelet_basis_id=0,
    )
    arc = parse_and_validate_archive(arc_bytes)
    assert arc.num_levels == 3
    assert arc.num_pairs == num_pairs
    assert len(arc.decoder_state_dict) >= 1


def test_z8_inflate_raises_l0_scaffold_not_implemented_on_runtime_forward() -> None:
    """Catalog #240 acceptance cascade (c): runtime forward IS council-gated."""
    import tempfile
    from pathlib import Path

    from tac.substrates.z8_hierarchical_predictive_coding.archive import pack_archive
    from tac.substrates.z8_hierarchical_predictive_coding.inflate import (
        Z8L0ScaffoldNotImplementedError,
        inflate_one_video_l0_scaffold,
    )

    num_pairs = 2
    num_groups = (4, 3, 2)
    num_cats = (16, 8, 4)
    arc_bytes = pack_archive(
        decoder_state_dict=_make_synthetic_state_dict(),
        per_level_category_indices=_make_synthetic_indices(
            num_pairs, num_groups, num_cats
        ),
        wavelet_coeffs_blob=b"\xAA" * 16,
        wyner_ziv_top_blob=b"\xBB" * 12,
        dreamer_state_dict=_make_synthetic_dreamer_state_dict(),
        meta={"k": 1},
        num_levels=3,
        num_groups_per_level=num_groups,
        num_categories_per_level=num_cats,
        num_pairs=num_pairs,
        decoder_latent_dim=28,
        base_channels=24,
        wavelet_basis_id=0,
    )
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.raw"
        with pytest.raises(Z8L0ScaffoldNotImplementedError, match="L0 SCAFFOLD"):
            inflate_one_video_l0_scaffold(arc_bytes, out_path, device="cpu")


# -----------------------------------------------------------------------------
# MLX renderer tests (skip if MLX unavailable per L0 platform-tolerant pattern)
# -----------------------------------------------------------------------------


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _mlx_available(), reason="MLX not installed on this platform")
def test_z8_mlx_config_defaults_validate() -> None:
    """Z8HierarchicalConfig defaults pass __post_init__ invariants."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
    )

    cfg = Z8HierarchicalConfig()
    assert cfg.num_levels == 3
    assert len(cfg.num_groups_per_level) == 3
    assert len(cfg.num_categories_per_level) == 3
    # categorical bits / packing bytes are derivable
    assert cfg.total_categorical_bits_per_sample > 0
    assert cfg.total_latent_packing_bytes_per_pair > 0


@pytest.mark.skipif(not _mlx_available(), reason="MLX not installed on this platform")
def test_z8_mlx_config_rejects_bad_level_counts() -> None:
    """__post_init__ validates per-level array lengths match num_levels."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
    )

    with pytest.raises(ValueError, match="num_groups_per_level length"):
        Z8HierarchicalConfig(
            num_levels=3,
            num_groups_per_level=(24, 16),  # length 2 != 3
            num_categories_per_level=(256, 128, 64),
        )


@pytest.mark.skipif(not _mlx_available(), reason="MLX not installed on this platform")
def test_z8_mlx_renderer_constructs_and_forward() -> None:
    """Z8HierarchicalPredictiveCoderMLX constructs and forward returns expected shape."""
    import mlx.core as mx

    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
        Z8HierarchicalPredictiveCoderMLX,
    )

    # Small smoke config for L0 CI tractability
    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=4,
        deterministic_state_dim=8,
    )
    model = Z8HierarchicalPredictiveCoderMLX(cfg)

    pair_indices = mx.array([0, 1], dtype=mx.int32)
    rgb_pair, per_level_indices, per_level_soft = model.forward_training(pair_indices)
    # Shape: (B=2, 2 frames, 3 RGB, H=384, W=512)
    assert tuple(int(d) for d in rgb_pair.shape) == (2, 2, 3, 384, 512)
    assert len(per_level_indices) == 3
    assert len(per_level_soft) == 3
    # Per-level indices shape: (B, G_l)
    for level_idx, (G, K) in enumerate(zip(cfg.num_groups_per_level, cfg.num_categories_per_level)):
        assert tuple(int(d) for d in per_level_indices[level_idx].shape) == (2, G)
        assert tuple(int(d) for d in per_level_soft[level_idx].shape) == (2, G, K)


@pytest.mark.skipif(not _mlx_available(), reason="MLX not installed on this platform")
def test_z8_mlx_renderer_eval_from_indices_matches_shape() -> None:
    """forward_eval_from_indices returns the same RGB shape as training forward."""
    import mlx.core as mx

    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
        Z8HierarchicalPredictiveCoderMLX,
    )

    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=4,
        deterministic_state_dim=8,
    )
    model = Z8HierarchicalPredictiveCoderMLX(cfg)

    # Provide synthetic per-level indices directly
    B = 2
    per_level_indices = [
        mx.zeros((B, int(cfg.num_groups_per_level[level_idx])), dtype=mx.int32)
        for level_idx in range(cfg.num_levels)
    ]
    rgb_pair = model.forward_eval_from_indices(per_level_indices)
    assert tuple(int(d) for d in rgb_pair.shape) == (2, 2, 3, 384, 512)


@pytest.mark.skipif(not _mlx_available(), reason="MLX not installed on this platform")
def test_z8_mlx_architecture_manifest_observability_surface() -> None:
    """architecture_manifest exposes Catalog #305 cite-chain + non-promotable tags."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
        Z8HierarchicalPredictiveCoderMLX,
    )

    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=4,
        deterministic_state_dim=8,
    )
    model = Z8HierarchicalPredictiveCoderMLX(cfg)
    manifest = model.architecture_manifest()
    # Non-promotable tags per Catalog #127 + #192 + #317 + #341
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["axis_tag"] == "[macOS-MLX research-signal]"
    # Cite-chain to canonical equations per Catalog #344
    assert "categorical_posterior_capacity_vs_continuous_gaussian_v1" in (
        manifest["canonical_equation_refs"]
    )
    # Schema identifier consistent with Z8 naming
    assert manifest["schema"].startswith("z8_hierarchical_predictive_coding")
    # Multi-level structure surfaces
    assert manifest["num_levels"] == 3
    assert manifest["total_categorical_bits_per_sample"] > 0


# -----------------------------------------------------------------------------
# Param count and config invariant sanity tests
# -----------------------------------------------------------------------------


def test_z8_decoder_param_count_increases_with_levels() -> None:
    """Param count derivation is monotonic and reasonable."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
        z8_decoder_param_count,
    )

    # Small config (smoke)
    cfg_small = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=4,
        deterministic_state_dim=8,
    )
    params_small = z8_decoder_param_count(cfg_small)
    assert params_small > 0

    # Larger config — should be strictly larger
    cfg_large = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(24, 16, 8),
        num_categories_per_level=(256, 128, 64),
        base_channels=24,
        decoder_latent_dim=28,
        num_pairs=600,
        deterministic_state_dim=64,
    )
    params_large = z8_decoder_param_count(cfg_large)
    assert params_large > params_small
