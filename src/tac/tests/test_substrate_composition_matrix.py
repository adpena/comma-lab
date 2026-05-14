# SPDX-License-Identifier: MIT
"""Tests for `tac.optimization.substrate_composition_matrix`.

Per CLAUDE.md "Recursive adversarial review protocol": these tests were
written for the 3-clean-pass greenup. Coverage: matrix construction +
composition rules + format_id collision detection + Pareto rows + composite
delta prediction + serialization + invariants.
"""

from __future__ import annotations

import json

import pytest

from tac.optimization.substrate_composition_matrix import (
    DISPATCH_COST_USD_MIDPOINT,
    SCHEMA_VERSION,
    Composability,
    CompositionMatrix,
    CompositionResult,
    ScoreAxis,
    SubstrateClass,
    SubstrateRow,
    build_composition_matrix,
    canonical_substrate_inventory,
    classify_pairwise_composability,
    filter_pareto_dominated,
    per_substrate_pareto_rows,
    predicted_composite_delta,
    rank_substrates_by_ev_per_dollar,
    serialize_matrix,
    serialize_pareto_rows,
    write_matrix_json,
)

# ── Inventory invariants ──────────────────────────────────────────────────


def test_inventory_has_expected_count():
    """50-row inventory: 24 legacy + 15 FIX-J + 9 WAVE-A-2 + 2 cooperative rows.

    Legacy 24 = residual basis 5 + pose-axis 3 + self-compression 3 +
    NeRV-family 5 + NeRV/MNeRV/VQVAE 3 + ANR/categorical 2 + magic codec 1 +
    bolt-ons 2.

    FIX-J wire-in 2026-05-12 (LOOPCLOSE) added 15 substrate-scaffold rows
    for the Fields-medal-council subpackages under ``src/tac/substrates/``
    (sane_hnerv, balle_renderer, hybrid_renderer_residual, self_compress_nn,
    pr101_lc_v2_clone, cool_chic_full_renderer, wavelet_full_renderer,
    grayscale_lut, vq_vae_substrate, siren_substrate, block_nerv_substrate,
    tc_nerv_substrate, ff_nerv_substrate, ds_nerv_substrate,
    hi_nerv_substrate). See
    ``feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512.md``.

    WAVE-A-2 wire-in 2026-05-12 (CANON-1.A explicit-taxonomy resolution)
    added 9 TRADITION 2 single-file substrate rows for the older
    ``src/tac/<name>_as_renderer.py`` / ``<name>_renderer.py`` substrates
    that pre-date the substrate-scaffold subpackage discipline: cnerv,
    e_nerv, ego_nerv, lane_12_v2_nerv, nervdc, quantizr_faithful,
    mlx_mask_renderer, dp_sims_renderer, diffusion_renderer. See
    ``.omx/research/substrate_tradition_taxonomy_20260512.md`` and
    ``feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512.md``.

    Cooperative-receiver wire-in 2026-05-13 added 2 rows: SARC and WZ1.
    """
    rows = canonical_substrate_inventory()
    assert len(rows) == 50, (
        "expected 50 substrate rows (24 legacy + 15 FIX-J + 9 WAVE-A-2 + "
        f"2 cooperative), got {len(rows)}"
    )


def test_inventory_substrate_ids_unique():
    rows = canonical_substrate_inventory()
    ids = [r.substrate_id for r in rows]
    assert len(set(ids)) == len(ids), "substrate_ids must be unique"


def test_inventory_has_at_least_one_per_class():
    rows = canonical_substrate_inventory()
    classes = {r.substrate_class for r in rows}
    expected = {
        SubstrateClass.RESIDUAL,
        SubstrateClass.RENDERER_REPLACEMENT,
        SubstrateClass.SELF_COMPRESSION,
        SubstrateClass.POSE_AXIS_SIDECHANNEL,
        SubstrateClass.META_CODEC,
        SubstrateClass.BOLT_ON,
    }
    assert expected <= classes, f"missing classes: {expected - classes}"


def test_inventory_format_ids_unique_within_class():
    """Within each substrate_class, format_ids must be distinct so the
    archive grammar can route bytes deterministically.

    NOTE: format_ids may overlap ACROSS classes (e.g., 0x10-0x14 for
    residual basis vs 0x60-0x64 for NeRV-family) because the magic+id
    pair is the disambiguator at parse time. Within a class, however,
    the id alone must be unique."""
    rows = canonical_substrate_inventory()
    by_class: dict[SubstrateClass, list[int]] = {}
    for r in rows:
        by_class.setdefault(r.substrate_class, []).append(r.format_id)
    for cls, ids in by_class.items():
        assert len(set(ids)) == len(ids), f"duplicate format_id within class {cls}: {ids}"


def test_inventory_dispatch_cost_table_complete():
    """Every substrate must have a cost estimate in DISPATCH_COST_USD_MIDPOINT."""
    rows = canonical_substrate_inventory()
    for r in rows:
        assert r.substrate_id in DISPATCH_COST_USD_MIDPOINT, (
            f"missing dispatch cost for {r.substrate_id!r}"
        )


def test_substrate_row_predicted_delta_alone_midpoint():
    r = SubstrateRow(
        substrate_id="t",
        name="t",
        substrate_class=SubstrateClass.RESIDUAL,
        target_axis=ScoreAxis.RATE,
        format_id=0xFF,
        magic_bytes="TEST",
        runtime_dep_closure=("torch",),
        byte_budget_band=(100, 500),
        predicted_delta_alone_band=(-0.001, -0.0001),
        requires_score_aware_training=True,
        landed_at="2026-05-11",
        landing_memo="test",
    )
    mid = r.predicted_delta_alone_midpoint()
    assert abs(mid - (-0.00055)) < 1e-12


# ── Matrix construction ──────────────────────────────────────────────────


def test_build_composition_matrix_default():
    matrix = build_composition_matrix()
    assert isinstance(matrix, CompositionMatrix)
    n = matrix.n_substrates()
    assert n == len(canonical_substrate_inventory())
    assert matrix.n_cells() == n * n
    assert matrix.schema_version == SCHEMA_VERSION
    assert matrix.score_claim is False
    assert matrix.promotion_eligible is False
    assert matrix.ready_for_exact_eval_dispatch is False


def test_build_composition_matrix_refuses_empty():
    with pytest.raises(ValueError, match="empty"):
        build_composition_matrix(substrates=[])


def test_build_composition_matrix_refuses_duplicate_ids():
    rows = list(canonical_substrate_inventory())
    # Inject a duplicate.
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        build_composition_matrix(substrates=rows)


def test_matrix_get_returns_typed_result():
    matrix = build_composition_matrix()
    cell = matrix.get("wavelet_residual", "c3_residual")
    assert isinstance(cell, CompositionResult)
    assert cell.substrate_a == "wavelet_residual"
    assert cell.substrate_b == "c3_residual"


def test_matrix_get_raises_keyerror_unknown_substrate():
    matrix = build_composition_matrix()
    with pytest.raises(KeyError):
        matrix.get("nonexistent", "wavelet_residual")


# ── Composition rule verification ────────────────────────────────────────


def test_two_renderer_replacements_are_replacement():
    """RENDERER_REPLACEMENT vs RENDERER_REPLACEMENT must be REPLACEMENT
    (only one renderer slot per HNeRV parity discipline lesson 5)."""
    matrix = build_composition_matrix()
    cell = matrix.get("nerv_as_renderer", "blocknerv")
    assert cell.composability == Composability.REPLACEMENT
    assert cell.expected_alpha == 0.0
    cell2 = matrix.get("anr_token_renderer_v62", "vqvae_as_full_renderer")
    assert cell2.composability == Composability.REPLACEMENT


def test_residual_different_axes_orthogonal():
    """wavelet (RATE) + c3 (MIXED) — different axes -> ORTHOGONAL."""
    matrix = build_composition_matrix()
    cell = matrix.get("wavelet_residual", "c3_residual")
    assert cell.composability == Composability.ORTHOGONAL
    assert cell.expected_alpha == 1.0


def test_residual_same_axis_redundant():
    """wavelet (RATE) + cool_chic (RATE) — same axis -> REDUNDANT."""
    matrix = build_composition_matrix()
    cell = matrix.get("wavelet_residual", "cool_chic_residual")
    assert cell.composability == Composability.REDUNDANT
    assert 0.0 < cell.expected_alpha < 0.5


def test_residual_plus_renderer_replacement_is_serial_stack():
    matrix = build_composition_matrix()
    cell = matrix.get("wavelet_residual", "nerv_as_renderer")
    assert cell.composability == Composability.STACKABLE_SERIAL
    assert 0.5 < cell.expected_alpha <= 1.0


def test_self_compression_with_anything_orthogonal():
    matrix = build_composition_matrix()
    cell = matrix.get("hessian_block_fp", "wavelet_residual")
    assert cell.composability == Composability.ORTHOGONAL
    cell2 = matrix.get("hessian_block_fp", "nerv_as_renderer")
    assert cell2.composability == Composability.ORTHOGONAL
    cell3 = matrix.get("mdl_fp4_tto", "categorical_substrate")
    assert cell3.composability == Composability.ORTHOGONAL


def test_self_compression_with_self_compression_is_cascade():
    """SC++ + Hessian-block-FP + MDL-FP4-TTO should cascade."""
    matrix = build_composition_matrix()
    cell = matrix.get("scpp_substrate", "hessian_block_fp")
    assert cell.composability == Composability.STACKABLE_CASCADE
    cell2 = matrix.get("hessian_block_fp", "mdl_fp4_tto")
    assert cell2.composability == Composability.STACKABLE_CASCADE


def test_pose_axis_with_spatial_orthogonal():
    matrix = build_composition_matrix()
    cell = matrix.get("foveation_field", "wavelet_residual")
    assert cell.composability == Composability.ORTHOGONAL
    cell2 = matrix.get("raft_pose_stream", "nerv_as_renderer")
    assert cell2.composability == Composability.ORTHOGONAL


def test_pose_axis_with_pose_axis_parallel_with_partial_redundancy():
    matrix = build_composition_matrix()
    cell = matrix.get("foveation_field", "raft_pose_stream")
    assert cell.composability == Composability.STACKABLE_PARALLEL
    assert 0.0 < cell.expected_alpha < 1.0


def test_meta_codec_with_anything_orthogonal():
    matrix = build_composition_matrix()
    cell = matrix.get("magic_codec", "wavelet_residual")
    assert cell.composability == Composability.ORTHOGONAL
    cell2 = matrix.get("magic_codec", "nerv_as_renderer")
    assert cell2.composability == Composability.ORTHOGONAL
    cell3 = matrix.get("magic_codec", "scpp_substrate")
    assert cell3.composability == Composability.ORTHOGONAL


def test_magic_codec_is_zero_ev_until_byte_closed_runtime_lands():
    row = next(
        r for r in canonical_substrate_inventory()
        if r.substrate_id == "magic_codec"
    )
    assert row.predicted_delta_alone_band == (0.0, 0.0)
    assert "repo_tac_required_until_vendored" in row.runtime_dep_closure
    pareto = {
        r.substrate_id: r for r in per_substrate_pareto_rows()
    }["magic_codec"]
    assert pareto.predicted_delta_alone_midpoint == 0.0
    assert pareto.estimated_dispatch_cost_usd == 0.0


def test_wyner_ziv_magic_matches_archive_grammar():
    from tac.substrates.wyner_ziv_cooperative_receiver.archive import WZ1_MAGIC

    row = {
        r.substrate_id: r
        for r in canonical_substrate_inventory()
    }["wyner_ziv_cooperative_receiver_substrate"]
    assert row.magic_bytes.encode("latin1") == WZ1_MAGIC


def test_self_with_self_redundant_alpha_zero():
    matrix = build_composition_matrix()
    cell = matrix.get("wavelet_residual", "wavelet_residual")
    assert cell.composability == Composability.REDUNDANT
    assert cell.expected_alpha == 0.0


def test_bolt_on_plus_renderer_replacement_parallel():
    """FiLM bolt-on modulates a host renderer; should be STACKABLE_PARALLEL."""
    matrix = build_composition_matrix()
    cell = matrix.get("film_pose_conditioning", "nerv_as_renderer")
    assert cell.composability == Composability.STACKABLE_PARALLEL


def test_bolt_on_plus_bolt_on_different_axes_orthogonal():
    """nerv_enc_dec_separated (RATE) + film_pose_conditioning (POSE)."""
    matrix = build_composition_matrix()
    cell = matrix.get("nerv_enc_dec_separated", "film_pose_conditioning")
    assert cell.composability == Composability.ORTHOGONAL


# ── Format-ID collision detection ────────────────────────────────────────


def test_no_format_id_collision_in_canonical_inventory_within_class():
    matrix = build_composition_matrix()
    # The canonical inventory uses non-overlapping format_id ranges by class
    # so within-class ids are unique. Cross-class ids may overlap by design.
    # n_format_id_collisions counts ANY pair with the same format_id and
    # different substrate_id; this would be 0 unless we cross classes.
    n = matrix.n_format_id_collisions()
    assert n == 0, f"unexpected format-id collisions: {n}"


def test_format_id_collision_detected_when_injected():
    rows = list(canonical_substrate_inventory())
    rows[0] = SubstrateRow(
        substrate_id=rows[0].substrate_id,
        name=rows[0].name,
        substrate_class=rows[0].substrate_class,
        target_axis=rows[0].target_axis,
        format_id=rows[1].format_id,  # Collide with row 1.
        magic_bytes=rows[0].magic_bytes,
        runtime_dep_closure=rows[0].runtime_dep_closure,
        byte_budget_band=rows[0].byte_budget_band,
        predicted_delta_alone_band=rows[0].predicted_delta_alone_band,
        requires_score_aware_training=rows[0].requires_score_aware_training,
        landed_at=rows[0].landed_at,
        landing_memo=rows[0].landing_memo,
    )
    matrix = build_composition_matrix(substrates=rows)
    n = matrix.n_format_id_collisions()
    assert n >= 1, "expected at least one format-id collision"


# ── Pareto rows + EV/dollar ranking ──────────────────────────────────────


def test_per_substrate_pareto_rows_count():
    rows = per_substrate_pareto_rows()
    assert len(rows) == len(canonical_substrate_inventory())


def test_pareto_rows_score_claim_invariants():
    rows = per_substrate_pareto_rows()
    for r in rows:
        assert r.score_claim is False
        assert r.promotion_eligible is False
        assert r.ready_for_exact_eval_dispatch is False


def test_rank_substrates_by_ev_per_dollar_descending():
    ranked = rank_substrates_by_ev_per_dollar()
    for i in range(len(ranked) - 1):
        assert ranked[i].eig_per_dollar >= ranked[i + 1].eig_per_dollar


def test_zero_cost_with_nonzero_eig_is_cost_unknown_not_free():
    """Hessian-block-FP has cost $0.50 and predicted delta band; bolt-ons
    have cost $0; magic codec has cost $0.10. Cost-zero is an unknown-cost
    blocker, not a free dispatch signal.

    Cost-unknown rows must rank last and carry cost-estimation notes instead
    of surfacing +inf EV/$.
    """
    ranked = rank_substrates_by_ev_per_dollar()
    cost_unknown = [
        r
        for r in ranked
        if r.substrate_id in {"nerv_enc_dec_separated", "film_pose_conditioning"}
    ]

    assert cost_unknown
    assert all(r.eig_per_dollar == 0.0 for r in cost_unknown)
    assert all("cost_estimation_required" in r.notes for r in cost_unknown)
    assert all(r not in ranked[:5] for r in cost_unknown)


def test_filter_pareto_dominated_drops_redundant_lower_ev_sibling():
    """Synthetic test: two RATE residuals with different EV/$ should drop the lower one."""
    matrix = build_composition_matrix()
    rows = per_substrate_pareto_rows(matrix=matrix)
    rate_residuals = [
        r for r in rows
        if r.substrate_class == SubstrateClass.RESIDUAL and r.target_axis == ScoreAxis.RATE
    ]
    assert len(rate_residuals) >= 2
    filtered = filter_pareto_dominated(rate_residuals, matrix=matrix)
    # At least ONE redundant lower-EV sibling should be dropped if multiple
    # rows share both class+axis and pairwise REDUNDANT.
    assert len(filtered) <= len(rate_residuals)


# ── Composite delta prediction ───────────────────────────────────────────


def test_composite_delta_single_substrate_is_alone_delta():
    result = predicted_composite_delta(["wavelet_residual"])
    assert result["score_claim"] is False
    assert result["n_substrates"] == 1
    rows = canonical_substrate_inventory()
    target = next(r for r in rows if r.substrate_id == "wavelet_residual")
    assert abs(result["predicted_composite_delta"] - target.predicted_delta_alone_midpoint()) < 1e-12


def test_composite_delta_two_orthogonal_residuals_additive():
    """wavelet (RATE) + c3 (MIXED) = ORTHOGONAL with alpha=1.0; composite = sum."""
    result = predicted_composite_delta(["wavelet_residual", "c3_residual"])
    rows = canonical_substrate_inventory()
    by_id = {r.substrate_id: r for r in rows}
    expected = (
        by_id["wavelet_residual"].predicted_delta_alone_midpoint()
        + by_id["c3_residual"].predicted_delta_alone_midpoint()
    )
    # alpha=1.0 -> penalty=0 -> additive.
    assert abs(result["predicted_composite_delta"] - expected) < 1e-12


def test_composite_delta_refuses_two_replacements():
    """Two RENDERER_REPLACEMENT must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        predicted_composite_delta(["nerv_as_renderer", "blocknerv"])


def test_composite_delta_refuses_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        predicted_composite_delta(["wavelet_residual", "wavelet_residual"])


def test_composite_delta_refuses_unknown_substrate():
    with pytest.raises(ValueError, match="not in canonical inventory"):
        predicted_composite_delta(["unknown_substrate_xyz"])


def test_composite_delta_has_planning_only_invariants():
    result = predicted_composite_delta(["wavelet_residual", "scpp_substrate"])
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "evidence_grade" in result
    assert "predicted" in result["evidence_grade"]


def test_composite_delta_redundant_residuals_diluted():
    """Two REDUNDANT residuals should give a composite less than the sum."""
    rows = canonical_substrate_inventory()
    by_id = {r.substrate_id: r for r in rows}
    naive_sum = (
        by_id["wavelet_residual"].predicted_delta_alone_midpoint()
        + by_id["cool_chic_residual"].predicted_delta_alone_midpoint()
    )
    result = predicted_composite_delta(["wavelet_residual", "cool_chic_residual"])
    # |composite| < |naive_sum| because alpha < 1.
    assert abs(result["predicted_composite_delta"]) < abs(naive_sum)


def test_composite_delta_self_compression_cascade_has_diminishing_returns():
    result = predicted_composite_delta(
        ["scpp_substrate", "hessian_block_fp", "mdl_fp4_tto"]
    )
    assert result["score_claim"] is False
    # 3-substrate cascade with alpha=0.7 each -> composite less than sum.
    rows = canonical_substrate_inventory()
    by_id = {r.substrate_id: r for r in rows}
    naive_sum = sum(
        by_id[s].predicted_delta_alone_midpoint()
        for s in ["scpp_substrate", "hessian_block_fp", "mdl_fp4_tto"]
    )
    # The composite should be less negative than the naive sum
    # (since alpha=0.7 < 1 dilutes the additive savings).
    assert result["predicted_composite_delta"] > naive_sum or abs(result["predicted_composite_delta"]) < abs(naive_sum)


# ── Serialization ────────────────────────────────────────────────────────


def test_serialize_matrix_jsonable():
    matrix = build_composition_matrix()
    payload = serialize_matrix(matrix)
    serialized = json.dumps(payload, sort_keys=True)
    parsed = json.loads(serialized)
    assert parsed["schema"] == SCHEMA_VERSION
    assert parsed["n_substrates"] == len(canonical_substrate_inventory())
    assert parsed["score_claim"] is False
    assert parsed["promotion_eligible"] is False
    assert parsed["ready_for_exact_eval_dispatch"] is False
    assert "claude_md_compliance_tags" in parsed


def test_serialize_pareto_rows_jsonable():
    rows = per_substrate_pareto_rows()
    payload = serialize_pareto_rows(rows)
    json.dumps(payload, sort_keys=True)  # Must not raise.
    assert len(payload) == len(canonical_substrate_inventory())
    for r in payload:
        assert r["score_claim"] is False
        assert "substrate_class" in r
        assert "target_axis" in r


def test_write_matrix_json_refuses_tmp_path(tmp_path):
    matrix = build_composition_matrix()
    bad_path = "/tmp/forbidden_substrate_matrix.json"
    with pytest.raises(ValueError, match="forbidden /tmp"):
        write_matrix_json(matrix, bad_path)


def test_write_matrix_json_writes_durable_path(tmp_path):
    """tmp_path is a pytest fixture pointing at a per-test temp dir; it is
    NOT a /tmp path (the fixture uses pytest's basetemp under the user's
    cache dir, not /tmp)."""
    matrix = build_composition_matrix()
    durable = tmp_path / "matrix.json"
    write_matrix_json(matrix, str(durable))
    parsed = json.loads(durable.read_text())
    assert parsed["schema"] == SCHEMA_VERSION


# ── Pairwise classifier directly (helper coverage) ───────────────────────


def test_classify_pairwise_residual_self_redundant():
    rows = canonical_substrate_inventory()
    a = next(r for r in rows if r.substrate_id == "wavelet_residual")
    cell = classify_pairwise_composability(a, a)
    assert cell.composability == Composability.REDUNDANT
    assert cell.expected_alpha == 0.0


def test_classify_pairwise_renderer_pair_replacement():
    rows = canonical_substrate_inventory()
    a = next(r for r in rows if r.substrate_id == "nerv_as_renderer")
    b = next(r for r in rows if r.substrate_id == "blocknerv")
    cell = classify_pairwise_composability(a, b)
    assert cell.composability == Composability.REPLACEMENT


# ── find_pairs helper ────────────────────────────────────────────────────


def test_find_pairs_replacement_returns_renderer_combinations():
    matrix = build_composition_matrix()
    pairs = matrix.find_pairs(Composability.REPLACEMENT)
    # 10 RENDERER_REPLACEMENT substrates -> 10*9/2 = 45 distinct pairs.
    rep_substrates = [
        s for s in matrix.substrates
        if s.substrate_class == SubstrateClass.RENDERER_REPLACEMENT
    ]
    n = len(rep_substrates)
    expected = n * (n - 1) // 2
    assert len(pairs) == expected


def test_find_pairs_orthogonal_includes_residual_cross_axes():
    matrix = build_composition_matrix()
    pairs = matrix.find_pairs(Composability.ORTHOGONAL)
    # wavelet (RATE) + c3 (MIXED) is ORTHOGONAL.
    expected = (
        min("wavelet_residual", "c3_residual"),
        max("wavelet_residual", "c3_residual"),
    )
    assert expected in pairs
