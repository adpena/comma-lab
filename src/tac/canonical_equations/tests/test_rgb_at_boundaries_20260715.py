"""NO-FAKE tests for the rgb_chroma_necessity_per_boundary_pair_v1 law (n600 measured payload)."""

import json

from tac.canonical_equations.rgb_at_boundaries_20260715 import (
    DSEG_EQUIV_DESAT_ANNULUS,
    DSEG_EQUIV_DESAT_FULL,
    DSEG_EQUIV_KEEP_ANNULUS,
    EQUATION_ID,
    FLIP_DESAT_FULL_BY_PAIR,
    build_rgb_chroma_necessity_per_boundary_pair_v1,
    populate_rgb_chroma_necessity_equation,
)


def test_builds_and_validates():
    eq = build_rgb_chroma_necessity_per_boundary_pair_v1()
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 1
    anchor = eq.empirical_anchors[0]
    assert anchor.empirical_verification_status == "VERIFIED_VIA_EMPIRICAL_ANCHOR"
    assert anchor.empirical_output["keep_annulus_worse_than_desat_full"] is True


def test_region_consistency_dominance_ordering():
    # the load-bearing structural inversion: keep_annulus > desat_full > desat_annulus
    assert DSEG_EQUIV_KEEP_ANNULUS > DSEG_EQUIV_DESAT_FULL > DSEG_EQUIV_DESAT_ANNULUS


def test_pair_ranking_movable_edges_most_chroma_decided():
    ranked = sorted(FLIP_DESAT_FULL_BY_PAIR, key=FLIP_DESAT_FULL_BY_PAIR.get, reverse=True)
    assert ranked[0] == "Undrivable|Movable"
    assert FLIP_DESAT_FULL_BY_PAIR["Road|Lane"] < FLIP_DESAT_FULL_BY_PAIR["Road|Movable"]


def test_payload_matches_measured_summary_artifact_when_present(tmp_path):
    # the constants must match the on-disk n600 summary when the artifact exists (ignored dir; may be absent
    # on fresh checkouts — the registry row is the durable copy).
    from pathlib import Path

    summary = Path("experiments/results/rgb_at_boundaries_chroma_jacobian_20260715/summary.json")
    if not summary.exists():
        return
    s = json.loads(summary.read_text())
    assert abs(s["global"]["d_seg_equiv_desat_full"] - DSEG_EQUIV_DESAT_FULL) < 5e-7
    assert abs(s["global"]["d_seg_equiv_desat_annulus"] - DSEG_EQUIV_DESAT_ANNULUS) < 5e-7
    assert abs(s["global"]["d_seg_equiv_keep_annulus"] - DSEG_EQUIV_KEEP_ANNULUS) < 5e-7
    assert s["n_pairs"] == 600


def test_populate_is_append_only_and_idempotent(tmp_path):
    path = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.lock"
    populate_rgb_chroma_necessity_equation(path=path, lock_path=lock, agent="test")
    populate_rgb_chroma_necessity_equation(path=path, lock_path=lock, agent="test")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(r["equation_id"] == EQUATION_ID for r in rows)
    assert len(rows) == 2  # append-only, latest-row-wins
