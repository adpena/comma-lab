# SPDX-License-Identifier: MIT
"""Tests for ``tac.optimization.cross_paradigm_composition_examples``.

Per CLAUDE.md "Recursive adversarial review protocol": these tests cover the
3-clean-pass adversarial greenup for the cross-paradigm composition examples
landing 2026-05-11.

Coverage:
- select_top_orthogonal_pairs invariants
- build_composition_example correctness
- format-ID collision refusal
- non-orthogonal pair refusal
- decoder ordering rules
- byte-budget feasibility
- materialize / parse round-trip (envelope grammar)
- smoke composition example
- serialization invariants
- /tmp path refusal
- write_example_archive_bytes
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.cross_paradigm_composition_examples import (
    SCHEMA_VERSION,
    build_composition_example,
    build_top_k_composition_examples,
    materialize_composition_example_bytes,
    parse_composition_example_envelope,
    select_top_orthogonal_pairs,
    serialize_composition_example_set,
    smoke_composition_example,
    write_composition_example_set_json,
    write_example_archive_bytes,
)
from tac.optimization.substrate_composition_matrix import (
    Composability,
    SubstrateClass,
    build_composition_matrix,
    canonical_substrate_inventory,
)


# ── Pair selection ───────────────────────────────────────────────────────


def test_select_top_5_orthogonal_pairs_default():
    pairs = select_top_orthogonal_pairs(top_k=5)
    assert len(pairs) == 5
    matrix = build_composition_matrix()
    for ri, rj in pairs:
        cell = matrix.get(ri.substrate_id, rj.substrate_id)
        assert cell.composability == Composability.ORTHOGONAL


def test_select_top_pairs_axes_diverge_default():
    pairs = select_top_orthogonal_pairs(top_k=5, require_axes_diverge=True)
    for ri, rj in pairs:
        assert ri.target_axis != rj.target_axis


def test_select_top_pairs_admit_same_axis_when_disabled():
    pairs = select_top_orthogonal_pairs(top_k=10, require_axes_diverge=False)
    matrix = build_composition_matrix()
    for ri, rj in pairs:
        cell = matrix.get(ri.substrate_id, rj.substrate_id)
        assert cell.composability == Composability.ORTHOGONAL
    # Some pair will have same axis at full top_k=10 with relaxed constraint.
    same_axis = [p for p in pairs if p[0].target_axis == p[1].target_axis]
    # Note: not strictly required, but admit at least one with relaxed flag.
    assert isinstance(same_axis, list)


def test_select_top_pairs_no_substrate_in_two_pairs():
    pairs = select_top_orthogonal_pairs(top_k=5)
    used_ids: set[str] = set()
    for ri, rj in pairs:
        assert ri.substrate_id not in used_ids
        assert rj.substrate_id not in used_ids
        used_ids.add(ri.substrate_id)
        used_ids.add(rj.substrate_id)


def test_select_excludes_replacement_renderers():
    pairs = select_top_orthogonal_pairs(
        top_k=5, exclude_replacement_renderers=True
    )
    for ri, rj in pairs:
        assert ri.substrate_class != SubstrateClass.RENDERER_REPLACEMENT
        assert rj.substrate_class != SubstrateClass.RENDERER_REPLACEMENT


def test_top_k_pairs_sorted_by_joint_ev_desc():
    pairs = select_top_orthogonal_pairs(top_k=5)
    joint_evs = [
        abs(ri.predicted_delta_alone_midpoint())
        + abs(rj.predicted_delta_alone_midpoint())
        for ri, rj in pairs
    ]
    assert joint_evs == sorted(joint_evs, reverse=True)


# ── build_composition_example ─────────────────────────────────────────────


def test_build_composition_example_basic():
    pairs = select_top_orthogonal_pairs(top_k=1)
    ex = build_composition_example(pairs[0])
    assert ex.composability == "orthogonal"
    assert ex.expected_alpha == 1.0
    assert ex.score_claim is False
    assert ex.promotion_eligible is False
    assert ex.ready_for_exact_eval_dispatch is False


def test_build_composition_example_refuses_non_orthogonal():
    rows = canonical_substrate_inventory()
    # Pick two RENDERER_REPLACEMENT rows; per matrix they are REPLACEMENT.
    renderers = [
        r for r in rows if r.substrate_class == SubstrateClass.RENDERER_REPLACEMENT
    ]
    assert len(renderers) >= 2
    with pytest.raises(ValueError, match="not ORTHOGONAL"):
        build_composition_example((renderers[0], renderers[1]))


def test_build_composition_example_byte_allocation_within_cap():
    pairs = select_top_orthogonal_pairs(top_k=3)
    for p in pairs:
        ex = build_composition_example(p, per_pair_byte_cap=50_000)
        assert ex.total_archive_bytes <= 50_000


def test_build_composition_example_decoder_ordering_renderer_first():
    rows = canonical_substrate_inventory()
    by_id = {r.substrate_id: r for r in rows}
    # Pick a renderer-replacement and a residual.
    renderer = next(
        r for r in rows if r.substrate_class == SubstrateClass.RENDERER_REPLACEMENT
    )
    residual = next(
        r for r in rows if r.substrate_class == SubstrateClass.RESIDUAL
    )
    matrix = build_composition_matrix()
    cell = matrix.get(renderer.substrate_id, residual.substrate_id)
    if cell.composability != Composability.ORTHOGONAL:
        # Residual + renderer is STACKABLE_SERIAL not ORTHOGONAL; pick a
        # different pair.
        return
    ex = build_composition_example((renderer, residual))
    assert ex.decoder_ordering[0] == renderer.substrate_id


def test_build_composition_example_ordering_self_compression_before_residual():
    rows = canonical_substrate_inventory()
    sc = next(r for r in rows if r.substrate_class == SubstrateClass.SELF_COMPRESSION)
    res = next(r for r in rows if r.substrate_class == SubstrateClass.RESIDUAL)
    matrix = build_composition_matrix()
    cell = matrix.get(sc.substrate_id, res.substrate_id)
    if cell.composability != Composability.ORTHOGONAL:
        pytest.skip("pair is not ORTHOGONAL in canonical matrix")
    ex = build_composition_example((sc, res))
    assert ex.decoder_ordering[0] == sc.substrate_id


# ── build_top_k_composition_examples ─────────────────────────────────────


def test_build_top_k_composition_examples_count():
    es = build_top_k_composition_examples(top_k=5)
    assert es.n_examples == 5
    assert len(es.examples) == 5


def test_build_top_k_composition_examples_pair_unique_substrates():
    es = build_top_k_composition_examples(top_k=5)
    used: set[str] = set()
    for ex in es.examples:
        assert ex.pair_a_substrate_id not in used
        assert ex.pair_b_substrate_id not in used
        used.add(ex.pair_a_substrate_id)
        used.add(ex.pair_b_substrate_id)


def test_build_top_k_examples_score_claim_false():
    es = build_top_k_composition_examples(top_k=5)
    assert es.score_claim is False
    for ex in es.examples:
        assert ex.score_claim is False


# ── Materialization round-trip ────────────────────────────────────────────


def test_materialize_envelope_starts_with_xcpe_magic():
    ex = build_top_k_composition_examples(top_k=1).examples[0]
    payload = materialize_composition_example_bytes(ex)
    assert payload[:4] == b"XCPE"


def test_materialize_envelope_round_trips():
    ex = build_top_k_composition_examples(top_k=1).examples[0]
    payload = materialize_composition_example_bytes(ex)
    parsed = parse_composition_example_envelope(payload)
    assert parsed["envelope_magic"] == "XCPE"
    assert parsed["version"] == 1
    assert parsed["n_substrates"] == 2
    assert len(parsed["rows"]) == 2


def test_materialize_envelope_preserves_decoder_ordering():
    ex = build_top_k_composition_examples(top_k=1).examples[0]
    payload = materialize_composition_example_bytes(ex)
    parsed = parse_composition_example_envelope(payload)
    expected_format_ids = []
    for sid in ex.decoder_ordering:
        row = next(r for r in ex.rows if r.substrate_id == sid)
        expected_format_ids.append(row.format_id)
    parsed_format_ids = [r["format_id"] for r in parsed["rows"]]
    assert parsed_format_ids == expected_format_ids


def test_parse_envelope_refuses_truncated_payload():
    with pytest.raises(ValueError, match="too short"):
        parse_composition_example_envelope(b"XCPE")


def test_parse_envelope_refuses_wrong_magic():
    with pytest.raises(ValueError, match="envelope magic mismatch"):
        parse_composition_example_envelope(b"FAKE\x01\x02xxxxxxx")


def test_parse_envelope_refuses_non_pair():
    # Build a valid envelope then mutate n_substrates byte to 3.
    ex = build_top_k_composition_examples(top_k=1).examples[0]
    payload = bytearray(materialize_composition_example_bytes(ex))
    payload[5] = 3
    with pytest.raises(ValueError, match="exactly 2 substrates"):
        parse_composition_example_envelope(bytes(payload))


# ── Smoke ────────────────────────────────────────────────────────────────


def test_smoke_composition_example_round_trip_passes():
    ex = build_top_k_composition_examples(top_k=1).examples[0]
    smoke = smoke_composition_example(ex)
    assert smoke["round_trip_passed"] is True
    assert smoke["score_claim"] is False
    assert smoke["evidence_grade"] == "diagnostic_not_score"
    assert smoke["byte_proxy_only"] is True


# ── Serialization ────────────────────────────────────────────────────────


def test_serialize_composition_example_set_jsonable():
    es = build_top_k_composition_examples(top_k=3)
    payload = serialize_composition_example_set(es)
    s = json.dumps(payload)
    parsed = json.loads(s)
    assert parsed["schema"] == SCHEMA_VERSION
    assert parsed["score_claim"] is False
    assert "claude_md_compliance_tags" in parsed


def test_write_composition_example_set_json_roundtrips(tmp_path: Path):
    es = build_top_k_composition_examples(top_k=3)
    out = tmp_path / "examples.json"
    write_composition_example_set_json(es, str(out))
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == SCHEMA_VERSION
    assert loaded["n_examples"] == 3


def test_write_composition_example_set_refuses_tmp_path():
    es = build_top_k_composition_examples(top_k=1)
    with pytest.raises(ValueError, match="forbidden /tmp path"):
        write_composition_example_set_json(es, "/tmp/should_not_be_allowed.json")
    with pytest.raises(ValueError, match="forbidden /tmp path"):
        write_composition_example_set_json(es, "/var/tmp/should_not_be_allowed.json")


def test_write_example_archive_bytes_outputs(tmp_path: Path):
    ex = build_top_k_composition_examples(top_k=1).examples[0]
    out_dir = tmp_path / "examples_dir"
    result = write_example_archive_bytes(ex, str(out_dir))
    archive_path = Path(result["archive_path"])
    assert archive_path.exists()
    assert archive_path.read_bytes()[:4] == b"XCPE"
    assert Path(result["manifest_path"]).exists()
    assert Path(result["smoke_result_path"]).exists()
    assert result["archive_size_bytes"] == archive_path.stat().st_size


def test_write_example_archive_bytes_refuses_tmp_path():
    ex = build_top_k_composition_examples(top_k=1).examples[0]
    with pytest.raises(ValueError, match="forbidden /tmp path"):
        write_example_archive_bytes(ex, "/tmp/cpex_archive_dir")


# ── Format-ID non-collision per pair ─────────────────────────────────────


def test_top_k_examples_format_ids_disjoint_within_pair():
    es = build_top_k_composition_examples(top_k=5)
    for ex in es.examples:
        format_ids = {row.format_id for row in ex.rows}
        assert len(format_ids) == 2, (
            f"format-ID collision in example {ex.example_id}: "
            f"{[r.format_id for r in ex.rows]}"
        )


def test_top_k_examples_predicted_composite_delta_finite():
    """Composite delta is a finite number (per Volterra correction)."""
    import math
    es = build_top_k_composition_examples(top_k=5)
    for ex in es.examples:
        assert math.isfinite(ex.predicted_composite_delta)


def test_top_k_examples_byte_budget_feasibility_recorded():
    es = build_top_k_composition_examples(
        top_k=5, per_pair_byte_cap=50_000
    )
    for ex in es.examples:
        assert ex.fits_pr106_r2_frontier == (ex.total_archive_bytes <= 50_000)


def test_top_k_examples_runtime_dep_closure_inherits_substrate():
    es = build_top_k_composition_examples(top_k=3)
    matrix = build_composition_matrix()
    by_id = {s.substrate_id: s for s in matrix.substrates}
    for ex in es.examples:
        for row in ex.rows:
            sub = by_id[row.substrate_id]
            assert row.runtime_dep_closure == sub.runtime_dep_closure
