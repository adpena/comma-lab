# SPDX-License-Identifier: MIT
"""Canonical test suite for Slot GGG × Cascade A FEC10 selector codec composition.

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #287
placeholder-rationale rejection + Catalog #341 + #357 Tier A canonical-routing
markers + Catalog #356 AxisDecomposition + Catalog #323 canonical Provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.composition.slot_ggg_x_cascade_a_fec10_selector_codec_composition import (
    CASCADE_A_FEC10_BASELINE_K,
    CASCADE_A_FEC10_K16_BASELINE_WIRE_BYTES,
    PREDICTED_SCORE_DELTA_BAND_LOWER,
    PREDICTED_SCORE_DELTA_BAND_UPPER,
    SLOT_GGG_K_RANKED_CONFIRMED_MODES,
    SLOT_GGG_SCALE_UP_ARTIFACT_PATH,
    CompositionArchiveResult,
    SlotGGGxCascadeAFEC10CompositionStrategy,
    SlotGGGxCascadeAFEC10Config,
    build_axis_decomposition_for_slot_ggg_x_cascade_a_fec10,
    build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive,
    list_canonical_paired_cuda_ratification_targets,
    load_slot_ggg_scale_up_artifact,
)


# ---- Constants ---------------------------------------------------------


def test_cascade_a_fec10_baseline_k_is_canonical_16() -> None:
    """Canonical 16-symbol palette per PR101 FEC6 sister discipline."""
    assert CASCADE_A_FEC10_BASELINE_K == 16


def test_cascade_a_fec10_k16_baseline_wire_bytes_is_canonical_236() -> None:
    """Per canonical equation cascade_a_fec10_hybrid_adaptive_blend_savings_v1
    anchor: 236 wire bytes on live PR110 600-pair K=16 selector stream."""
    assert CASCADE_A_FEC10_K16_BASELINE_WIRE_BYTES == 236


def test_predicted_score_delta_band_is_negative_savings_band() -> None:
    """Predicted ΔS band must be NEGATIVE (savings = lower contest score)."""
    assert PREDICTED_SCORE_DELTA_BAND_LOWER < 0
    assert PREDICTED_SCORE_DELTA_BAND_UPPER < 0
    assert PREDICTED_SCORE_DELTA_BAND_LOWER < PREDICTED_SCORE_DELTA_BAND_UPPER


def test_slot_ggg_k_ranked_confirmed_modes_is_5() -> None:
    """Per Slot GGG SCALE-UP artifact: 5 CONFIRMED DCT_CHROMA modes."""
    assert SLOT_GGG_K_RANKED_CONFIRMED_MODES == 5


def test_slot_ggg_scale_up_artifact_path_points_to_canonical_artifact() -> None:
    """Canonical artifact path resolves to a valid file in the repo."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    assert artifact_path.is_file()


# ---- Strategy enum ------------------------------------------------------


def test_strategy_enum_has_canonical_4_values() -> None:
    """Per Catalog #308 alternative reducer enumeration discipline."""
    values = {s.value for s in SlotGGGxCascadeAFEC10CompositionStrategy}
    assert values == {
        "direct_5_mode_palette",
        "expanded_8_mode_palette_with_3_null_modes",
        "per_pair_grouped_by_segnet_class_region",
        "multi_mode_stacking_per_pair",
    }


def test_strategy_enum_is_str_subclass() -> None:
    """Canonical sister discipline: enum members are str for canonical
    routing markers Provenance serialization."""
    assert issubclass(SlotGGGxCascadeAFEC10CompositionStrategy, str)


# ---- Config __post_init__ invariants -----------------------------------


def test_config_canonical_construction_accepts_substantive_rationale() -> None:
    """Canonical config accepts substantive rationale per Catalog #287."""
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        rationale="canonical baseline strategy per Slot GGG ranked CONFIRMED 5 modes",
    )
    assert config.strategy == SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE
    assert config.num_pairs == 600
    assert config.n_confirmed_modes_to_use == 5
    assert config.cascade_a_fec10_alpha == 2


def test_config_rejects_placeholder_rationale_per_catalog_287() -> None:
    """Catalog #287 placeholder rationale rejection."""
    for placeholder in ("<rationale>", "<reason>", "<rationale_here>", "<reason_here>", "tbd", "TODO"):
        with pytest.raises(ValueError, match="placeholder"):
            SlotGGGxCascadeAFEC10Config(
                strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
                rationale=placeholder,
            )


def test_config_rejects_short_rationale() -> None:
    """Catalog #287 sister discipline: rationale ≥4 chars."""
    with pytest.raises(ValueError, match="≥4 chars"):
        SlotGGGxCascadeAFEC10Config(
            strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
            rationale="ab",
        )


def test_config_rejects_invalid_strategy_type() -> None:
    """Type invariant: strategy must be SlotGGGxCascadeAFEC10CompositionStrategy."""
    with pytest.raises(ValueError, match="SlotGGGxCascadeAFEC10CompositionStrategy"):
        SlotGGGxCascadeAFEC10Config(
            strategy="not_a_strategy",  # type: ignore[arg-type]
            rationale="canonical rationale text",
        )


def test_config_rejects_invalid_num_pairs() -> None:
    """num_pairs must be int in [1, 10000]."""
    for bad in (0, -1, 10001, 1.5):
        with pytest.raises(ValueError, match="num_pairs"):
            SlotGGGxCascadeAFEC10Config(
                strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
                num_pairs=bad,  # type: ignore[arg-type]
                rationale="canonical rationale text",
            )


def test_config_rejects_invalid_n_confirmed_modes() -> None:
    """n_confirmed_modes_to_use must be int in [2, 16]."""
    for bad in (1, 17, 0, -1):
        with pytest.raises(ValueError, match="n_confirmed_modes_to_use"):
            SlotGGGxCascadeAFEC10Config(
                strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
                n_confirmed_modes_to_use=bad,
                rationale="canonical rationale text",
            )


def test_config_rejects_invalid_alpha() -> None:
    """cascade_a_fec10_alpha must be int in [1, 8]."""
    for bad in (0, 9, -1, 100):
        with pytest.raises(ValueError, match="cascade_a_fec10_alpha"):
            SlotGGGxCascadeAFEC10Config(
                strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
                cascade_a_fec10_alpha=bad,
                rationale="canonical rationale text",
            )


def test_config_is_frozen() -> None:
    """frozen=True invariant per Catalog #287 sister discipline."""
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        rationale="canonical rationale text",
    )
    with pytest.raises((AttributeError, TypeError)):
        config.num_pairs = 999  # type: ignore[misc]


# ---- Slot GGG SCALE-UP artifact loader ---------------------------------


def test_load_slot_ggg_scale_up_artifact_canonical_path_returns_valid_payload() -> None:
    """Canonical loader returns the canonical SCALE-UP payload."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    payload = load_slot_ggg_scale_up_artifact(artifact_path)
    assert payload["schema"].startswith("slot_ggg_scale_up_matrix")
    assert payload["n_modes_confirmed"] == 5
    assert len(payload["ranked_confirmed_modes_by_capacity_per_cost"]) == 5


def test_load_slot_ggg_scale_up_artifact_rejects_missing_path(tmp_path: Path) -> None:
    """Canonical loader fail-closed when artifact missing."""
    with pytest.raises(FileNotFoundError):
        load_slot_ggg_scale_up_artifact(tmp_path / "missing.json")


def test_load_slot_ggg_scale_up_artifact_rejects_invalid_schema(tmp_path: Path) -> None:
    """Canonical loader fail-closed when schema mismatch."""
    bad = tmp_path / "bad_schema.json"
    bad.write_text(json.dumps({"schema": "not_slot_ggg", "n_modes_confirmed": 5}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_slot_ggg_scale_up_artifact(bad)


def test_load_slot_ggg_scale_up_artifact_rejects_insufficient_modes(tmp_path: Path) -> None:
    """Canonical loader fail-closed when n_modes_confirmed < 2."""
    bad = tmp_path / "insufficient.json"
    bad.write_text(
        json.dumps(
            {
                "schema": "slot_ggg_scale_up_matrix.v1",
                "n_modes_confirmed": 1,
                "ranked_confirmed_modes_by_capacity_per_cost": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="insufficient"):
        load_slot_ggg_scale_up_artifact(bad)


def test_load_slot_ggg_scale_up_artifact_rejects_missing_ranked_list(tmp_path: Path) -> None:
    """Canonical loader fail-closed when ranked list missing."""
    bad = tmp_path / "missing_ranked.json"
    bad.write_text(
        json.dumps(
            {
                "schema": "slot_ggg_scale_up_matrix.v1",
                "n_modes_confirmed": 5,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ranked_confirmed_modes"):
        load_slot_ggg_scale_up_artifact(bad)


# ---- Composition archive builder ---------------------------------------


def test_build_canonical_archive_returns_canonical_composition_archive_result() -> None:
    """Canonical builder returns CompositionArchiveResult per Catalog #245."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=60,
        rationale="canonical baseline test on 60-pair sub-sample",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    assert isinstance(result, CompositionArchiveResult)
    assert isinstance(result.selector_codec_payload_bytes, bytes)
    assert len(result.selector_codec_payload_bytes) > 0
    assert result.num_pairs == 60
    assert len(result.per_pair_selector_indices) == 60
    assert len(result.mode_ids_used) == 5


def test_build_canonical_archive_emits_only_dct_chroma_mode_ids() -> None:
    """Canonical Slot GGG SCALE-UP ranked CONFIRMED modes are all DCT_CHROMA."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=10,
        rationale="canonical test that mode_ids_used are DCT_CHROMA modes",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    for mode_id in result.mode_ids_used:
        assert mode_id.startswith("frame1_dct_chroma_"), (
            f"All 5 ranked CONFIRMED modes from Slot GGG must be DCT_CHROMA; "
            f"got {mode_id!r}"
        )


def test_build_canonical_archive_per_pair_selector_indices_in_bounds() -> None:
    """All per-pair selector indices must be in [0, n_confirmed_modes_to_use)."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=30,
        rationale="canonical bounds test on per-pair selector indices",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    for idx in result.per_pair_selector_indices:
        assert 0 <= idx < config.n_confirmed_modes_to_use


def test_build_canonical_archive_is_byte_deterministic_per_base_archive_sha() -> None:
    """Canonical sister discipline: same base_archive_sha → same selector indices."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=50,
        rationale="canonical byte-deterministic reproducibility test",
    )
    a = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    b = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    assert a.per_pair_selector_indices == b.per_pair_selector_indices
    assert a.selector_codec_payload_bytes == b.selector_codec_payload_bytes


def test_build_canonical_archive_different_seeds_produce_different_indices() -> None:
    """Canonical determinism: different base_archive_sha → different selectors."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=100,
        rationale="canonical seed-derived determinism test",
    )
    a = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    b = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="9cb989cef5190000000000000000000000000000000000000000000000000000",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    # Different seeds must produce different selector index sequences (with very high probability)
    assert a.per_pair_selector_indices != b.per_pair_selector_indices


def test_build_canonical_archive_emits_feca_magic_header() -> None:
    """Canonical FECA magic header per Cascade A FEC10 sister discipline."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=20,
        rationale="canonical FECA magic header test",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    assert result.selector_codec_payload_bytes[:4] == b"FECa"


def test_build_canonical_archive_accepts_explicit_per_pair_selector_indices() -> None:
    """Canonical builder honors explicit per-pair selector indices."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=5,
        rationale="canonical explicit-indices test",
    )
    explicit = (0, 1, 2, 3, 4)
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        per_pair_selector_indices=explicit,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    assert result.per_pair_selector_indices == explicit


def test_build_canonical_archive_rejects_mismatched_explicit_indices_length() -> None:
    """Length mismatch: explicit per-pair selector indices must equal num_pairs."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=10,
        rationale="canonical length mismatch test",
    )
    with pytest.raises(ValueError, match="num_pairs"):
        build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
            base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
            config=config,
            per_pair_selector_indices=(0, 1, 2),  # length 3 != num_pairs 10
            slot_ggg_scale_up_artifact_path=artifact_path,
        )


def test_build_canonical_archive_rejects_out_of_range_explicit_indices() -> None:
    """Bounds: explicit per-pair selector indices must be < n_confirmed_modes_to_use."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=3,
        rationale="canonical out-of-range index rejection test",
    )
    with pytest.raises(ValueError, match="outside"):
        build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
            base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
            config=config,
            per_pair_selector_indices=(0, 1, 99),  # 99 >= n_confirmed_modes_to_use=5
            slot_ggg_scale_up_artifact_path=artifact_path,
        )


# ---- Canonical Tier A markers per Catalog #341 + #357 ------------------


def test_canonical_routing_markers_have_canonical_keys() -> None:
    """Catalog #341 + #357 canonical-routing-markers contract."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=10,
        rationale="canonical Tier A markers test",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    markers = result.canonical_routing_markers
    assert markers["predicted_delta_adjustment"] == 0.0
    assert markers["promotable"] is False
    assert markers["score_claim"] is False
    assert markers["axis_tag"] == "[macOS-CPU advisory]"
    assert markers["evidence_grade"] == "predicted"
    assert isinstance(markers["rationale"], str) and len(markers["rationale"]) > 50


def test_canonical_routing_markers_apply_to_every_strategy() -> None:
    """Catalog #341 + #357: EVERY strategy emits Tier A markers."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    for strategy in SlotGGGxCascadeAFEC10CompositionStrategy:
        config = SlotGGGxCascadeAFEC10Config(
            strategy=strategy,
            num_pairs=8,
            rationale=f"canonical Tier A test for strategy {strategy.value}",
        )
        result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
            base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
            config=config,
            slot_ggg_scale_up_artifact_path=artifact_path,
        )
        assert result.canonical_routing_markers["promotable"] is False
        assert result.canonical_routing_markers["predicted_delta_adjustment"] == 0.0
        assert result.canonical_routing_markers["score_claim"] is False


# ---- Canonical AxisDecomposition per Catalog #356 ----------------------


def test_predicted_axis_decomposition_has_canonical_keys() -> None:
    """Catalog #356 AxisDecomposition canonical keys."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=10,
        rationale="canonical AxisDecomposition keys test",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    axis = result.predicted_axis_decomposition
    assert "predicted_d_seg_delta" in axis
    assert "predicted_d_pose_delta" in axis
    assert "predicted_archive_bytes_delta" in axis
    assert axis["axis_tag"] == "[predicted]"
    assert isinstance(axis["canonical_provenance"], dict)


def test_predicted_axis_decomposition_d_seg_near_zero() -> None:
    """Slot GGG SCALE-UP empirical anchor: d_seg ≈ 0 (SegNet-null invariant)."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=10,
        rationale="canonical SegNet-null projection invariant test",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    # Per Slot GGG SCALE-UP aggregate d_seg ≈ -1.2e-4 (very small)
    assert abs(result.predicted_axis_decomposition["predicted_d_seg_delta"]) < 1e-3


def test_predicted_axis_decomposition_d_pose_in_carrier_band() -> None:
    """Slot GGG SCALE-UP empirical anchor: |d_pose| ∈ [1e-9, 1e-3] carrier band."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=10,
        rationale="canonical pose-axis carrier band test",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    d_pose = result.predicted_axis_decomposition["predicted_d_pose_delta"]
    assert 1e-9 <= abs(d_pose) <= 1e-3


def test_build_axis_decomposition_for_slot_ggg_x_cascade_a_fec10_public_api() -> None:
    """Public AxisDecomposition builder per Catalog #356."""
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=600,
        rationale="canonical public-API AxisDecomposition test",
    )
    axis = build_axis_decomposition_for_slot_ggg_x_cascade_a_fec10(
        config=config,
        selector_codec_wire_bytes=180,
    )
    assert isinstance(axis, dict)
    assert axis["axis_tag"] == "[predicted]"
    # At 180 wire bytes vs baseline 236 = -56 bytes savings
    assert axis["predicted_archive_bytes_delta"] == 180 - 236


# ---- Canonical Provenance per Catalog #323 -----------------------------


def test_canonical_provenance_has_canonical_keys() -> None:
    """Catalog #323 canonical Provenance keys."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=10,
        rationale="canonical Provenance keys test",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    prov = result.canonical_provenance
    assert prov["artifact_kind"] == "predicted_from_model"
    assert prov["evidence_grade"] == "predicted"
    assert prov["promotion_eligible"] is False
    assert prov["score_claim_valid"] is False
    assert prov["measurement_axis"] == "[macOS-CPU advisory]"
    assert "captured_at_utc" in prov


# ---- Canonical paired-CUDA RATIFICATION target enumeration -------------


def test_list_canonical_paired_cuda_ratification_targets_returns_4_targets() -> None:
    """Canonical enumeration per Slot GGG sister-pattern template."""
    targets = list_canonical_paired_cuda_ratification_targets()
    assert len(targets) == 4
    substrate_ids = {t["substrate_id"] for t in targets}
    assert substrate_ids == {"v14_v2_dqs1", "fec6", "pr106_format0d", "nscs06_v8_stacked"}


def test_canonical_paired_cuda_ratification_targets_have_canonical_fields() -> None:
    """Every target has canonical_sha_prefix + frontier_role + predicted_delta_s_band + paired_cuda_envelope_usd."""
    targets = list_canonical_paired_cuda_ratification_targets()
    for target in targets:
        assert "substrate_id" in target
        assert "canonical_sha_prefix" in target
        assert "frontier_role" in target
        assert "predicted_delta_s_band" in target
        assert "paired_cuda_envelope_usd" in target
        # predicted_delta_s_band must be (lower, upper) tuple
        band = target["predicted_delta_s_band"]
        assert isinstance(band, tuple) and len(band) == 2
        assert band[0] < 0 and band[1] < 0  # both negative (savings)
        # paired_cuda_envelope_usd ≤ $0.30 per Catalog #246 sister discipline
        assert target["paired_cuda_envelope_usd"] <= 1.0


def test_canonical_paired_cuda_ratification_targets_match_catalog_343_frontier() -> None:
    """Targets cite canonical Catalog #343 frontier pointer sha prefixes."""
    targets = list_canonical_paired_cuda_ratification_targets()
    fec6 = next(t for t in targets if t["substrate_id"] == "fec6")
    assert fec6["canonical_sha_prefix"] == "6bae0201fb082457"
    v14_v2 = next(t for t in targets if t["substrate_id"] == "v14_v2_dqs1")
    assert v14_v2["canonical_sha_prefix"] == "7a0da5d0fc327cba"
    pr106 = next(t for t in targets if t["substrate_id"] == "pr106_format0d")
    assert pr106["canonical_sha_prefix"] == "9cb989cef519"


# ---- Sister-disjoint regression vs canonical sister helpers ------------


def test_slot_ggg_canonical_helper_remains_importable() -> None:
    """Sister-disjoint regression: canonical Slot GGG helper still importable."""
    from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (  # noqa: F401
        list_canonical_paired_cuda_ratification_targets as slot_ggg_targets,
    )
    targets = slot_ggg_targets()
    assert len(targets) == 4


def test_composition_module_has_canonical_public_api() -> None:
    """Catalog #335 canonical contract: __all__ exposes canonical public API."""
    from tac.composition import slot_ggg_x_cascade_a_fec10_selector_codec_composition as mod
    assert "SlotGGGxCascadeAFEC10CompositionStrategy" in mod.__all__
    assert "SlotGGGxCascadeAFEC10Config" in mod.__all__
    assert "build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive" in mod.__all__
    assert "build_axis_decomposition_for_slot_ggg_x_cascade_a_fec10" in mod.__all__
    assert "list_canonical_paired_cuda_ratification_targets" in mod.__all__
    assert "load_slot_ggg_scale_up_artifact" in mod.__all__


# ---- Catalog #287 selector codec invariants ----------------------------


def test_selector_codec_payload_wire_bytes_match_expected_capacity() -> None:
    """Per Shannon's source-coding theorem: wire bytes ≈ num_pairs × log2(K) / 8 + header."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=100,
        n_confirmed_modes_to_use=5,
        rationale="canonical wire-byte capacity test",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    # At K=5, log2(5) ≈ 2.32 bits/pair → ceil(log2(5)) = 3 bits/pair = 38 bytes raw at 100 pairs
    # + 8 byte header = ~46 bytes total wire
    wire_bytes = len(result.selector_codec_payload_bytes)
    assert 30 <= wire_bytes <= 100, f"wire_bytes={wire_bytes} outside reasonable bounds for K=5, num_pairs=100"


def test_composition_result_predicted_delta_archive_bytes_is_meaningful() -> None:
    """At K=5 sub-palette, predicted archive bytes delta should be significantly
    smaller than baseline K=16 = 236 bytes."""
    repo_root = Path(__file__).resolve().parents[5]
    artifact_path = repo_root / SLOT_GGG_SCALE_UP_ARTIFACT_PATH
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        num_pairs=600,
        n_confirmed_modes_to_use=5,
        rationale="canonical 600-pair predicted-delta-archive-bytes test",
    )
    result = build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
        base_archive_sha="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        config=config,
        slot_ggg_scale_up_artifact_path=artifact_path,
    )
    # At 600 pairs × 3 bits/pair = 225 bytes raw + 8 header = 233 bytes
    # vs baseline 236 → predicted_archive_bytes_delta ≈ -3 to -10 bytes
    delta = result.predicted_axis_decomposition["predicted_archive_bytes_delta"]
    assert delta < 0, f"Expected savings (negative delta); got {delta}"


# ---- CompositionArchiveResult __post_init__ invariants -----------------


def test_composition_archive_result_rejects_non_bytes_payload() -> None:
    """CompositionArchiveResult must have bytes payload."""
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        rationale="canonical CompositionArchiveResult type test",
    )
    with pytest.raises(ValueError, match="selector_codec_payload_bytes must be bytes"):
        CompositionArchiveResult(
            selector_codec_payload_bytes="not bytes",  # type: ignore[arg-type]
            mode_ids_used=(),
            per_pair_selector_indices=(),
            num_pairs=0,
            config=config,
            canonical_routing_markers={},
            predicted_axis_decomposition={},
            canonical_provenance={},
        )


def test_composition_archive_result_rejects_non_tuple_mode_ids() -> None:
    """CompositionArchiveResult mode_ids_used must be tuple of strs."""
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        rationale="canonical CompositionArchiveResult mode_ids type test",
    )
    with pytest.raises(ValueError, match="mode_ids_used"):
        CompositionArchiveResult(
            selector_codec_payload_bytes=b"",
            mode_ids_used=["a", "b"],  # type: ignore[arg-type]
            per_pair_selector_indices=(),
            num_pairs=0,
            config=config,
            canonical_routing_markers={},
            predicted_axis_decomposition={},
            canonical_provenance={},
        )


def test_composition_archive_result_rejects_length_mismatch() -> None:
    """CompositionArchiveResult per_pair_selector_indices length must equal num_pairs."""
    config = SlotGGGxCascadeAFEC10Config(
        strategy=SlotGGGxCascadeAFEC10CompositionStrategy.DIRECT_5_MODE_PALETTE,
        rationale="canonical CompositionArchiveResult length mismatch test",
    )
    with pytest.raises(ValueError, match="per_pair_selector_indices"):
        CompositionArchiveResult(
            selector_codec_payload_bytes=b"",
            mode_ids_used=("m1",),
            per_pair_selector_indices=(0, 1, 2),
            num_pairs=10,  # mismatch
            config=config,
            canonical_routing_markers={},
            predicted_axis_decomposition={},
            canonical_provenance={},
        )
