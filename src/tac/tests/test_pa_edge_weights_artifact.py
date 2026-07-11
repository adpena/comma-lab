"""Tests for the FEED-PA edge-weight matrix W_e (P0 FORCE 3 tie-locus displacement; task #360).

Covers the SHIPPED artifact ``reports/pa_edge_weights.json`` (structural + FEED-PA cross-check
properties) AND the deterministic builder ``tools/build_pa_edge_weights.py`` (reproducibility,
mean-normalization, symmetry, Road-hub structure). All $0 / cache-only / no scorer forward.

Closes the last #360 FORCE-3 gap: without this artifact the DSL default ``edge_weight_source=
pa_flipmass`` silently downgrades to uniform (trainer L5786+), so the highest-EV precision force
never fires its designed flip-density weighting. Pointer UNMOVED — the artifact is MEANS.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "reports" / "pa_edge_weights.json"
TOOL = REPO / "tools" / "build_pa_edge_weights.py"
GT_N96 = REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n96.npz"
CLASS_ORDER = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_pa_edge_weights", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip("reports/pa_edge_weights.json not built (run tools/build_pa_edge_weights.py)")
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def W_e(artifact) -> np.ndarray:
    return np.asarray(artifact["W_e"], dtype=np.float64)


# ---- shipped-artifact structural properties -------------------------------------------------

def test_artifact_exists_and_valid_json(artifact):
    assert isinstance(artifact, dict)
    assert "W_e" in artifact and "provenance" in artifact and "feed_pa_crosscheck" in artifact


def test_W_e_is_5x5(W_e):
    assert W_e.shape == (5, 5), f"trainer requires (5,5), got {W_e.shape}"


def test_W_e_symmetric(W_e):
    # the edge is unordered {c_a,c_b} — the provider looks up either orientation.
    assert np.allclose(W_e, W_e.T, atol=1e-6)


def test_W_e_diagonal_zero(W_e):
    # a straddle requires differing classes -> no same-class edge ever contributes.
    assert np.allclose(np.diag(W_e), 0.0)


def test_W_e_mean_over_populated_offdiag_is_one(artifact, W_e):
    # "populated" = edges with a real straddle (count>0); absent edges carry the 0.05 floor and are
    # excluded. Use the exact count matrix from stats (not a magnitude threshold, which would drop
    # small-but-real edges like Lane<->Undriv ~0.009 and inflate the mean).
    count = np.asarray(artifact["stats"]["raw_edge_count"], dtype=np.int64)
    off = ~np.eye(5, dtype=bool)
    populated = off & (count > 0)
    assert populated.any()
    # mean over the real (measured) edges == 1.0 by construction so W_e REWEIGHTS (not rescales) the loss.
    assert abs(float(W_e[populated].mean()) - 1.0) <= 1e-3


def test_W_e_all_nonnegative(W_e):
    assert (W_e >= 0.0).all()


def test_W_e_road_is_the_hub(W_e):
    # FEED-PA: every class flips ONLY at its Road separatrix -> Road's row carries the most edge mass.
    row_sums = W_e.sum(axis=1)
    assert int(np.argmax(row_sums)) == 0, "Road (idx 0) must be the heaviest hub"


def test_W_e_road_lane_is_single_heaviest_edge(W_e):
    # derivation §3.2 + FEED-PA: Road<->Lane heaviest (41% of Road's flips).
    off = ~np.eye(5, dtype=bool)
    masked = np.where(off, W_e, -np.inf)
    ij = np.unravel_index(int(np.argmax(masked)), W_e.shape)
    assert set(ij) == {0, 1}, f"heaviest edge must be Road<->Lane, got {ij}"


def test_W_e_non_road_edges_light(W_e):
    # Lane<->Undriv, Lane<->Movable, Undriv<->MyCar etc. are near the floor (Road-hubbed graph).
    non_road = [(1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    for a, b in non_road:
        assert W_e[a, b] < W_e[0, 1], f"non-Road edge ({a},{b}) must be lighter than Road<->Lane"


def test_crosscheck_ranking_agrees(artifact):
    cc = artifact["feed_pa_crosscheck"]
    assert cc["ranking_agrees"] is True
    assert cc["road_is_hub_built"] and cc["road_is_hub_feed_pa"]


def test_crosscheck_pearson_strong(artifact):
    # the built GT-fragility density must correlate with the FEED-PA witness-confusion flip mass.
    assert artifact["feed_pa_crosscheck"]["pearson_r_built_vs_feed_pa"] >= 0.80


def test_class_order_canonical(artifact):
    assert artifact["class_order"] == CLASS_ORDER


def test_provenance_stamped_non_promotable(artifact):
    prov = artifact["provenance"]
    assert prov["tool"] == "tools/build_pa_edge_weights.py"
    assert "p0_forces_derivation" in prov["derivation"]
    assert "FEED-PA" in prov["dag_anchor"]
    assert "NON-PROMOTABLE" in prov["axis"]
    assert prov["gt_cache"].endswith(".npz")


def test_artifact_built_at_n600(artifact):
    # the shipped artifact must be n600-scale (allergic-to-toys discipline), never an n96 subset.
    assert artifact["stats"]["n_pairs"] == 600


# ---- builder reproducibility / mechanism (fast, on the n96 cache) ---------------------------

@pytest.mark.skipif(not GT_N96.is_file(), reason="gt_n96 cache absent")
def test_builder_deterministic_and_road_hub_on_n96():
    mod = _load_tool()
    W1, s1 = mod.build_edge_weights(GT_N96, v_band=1.0)
    W2, s2 = mod.build_edge_weights(GT_N96, v_band=1.0)
    assert np.array_equal(W1, W2), "builder must be deterministic"
    assert W1.shape == (5, 5)
    assert np.allclose(W1, W1.T)               # symmetric
    assert np.allclose(np.diag(W1), 0.0)       # zero diagonal
    assert int(np.argmax(W1.sum(axis=1))) == 0  # Road hub on n96 too
    off = ~np.eye(5, dtype=bool)
    masked = np.where(off, W1, -np.inf)
    ij = np.unravel_index(int(np.argmax(masked)), W1.shape)
    assert set(ij) == {0, 1}                    # Road<->Lane heaviest on n96 too


@pytest.mark.skipif(not GT_N96.is_file(), reason="gt_n96 cache absent")
def test_builder_floor_applied_to_absent_edges():
    mod = _load_tool()
    W, _ = mod.build_edge_weights(GT_N96, v_band=1.0, floor=0.05)
    off = ~np.eye(5, dtype=bool)
    # every off-diagonal cell is > 0 (populated edges normalized, absent edges get the floor).
    assert (W[off] > 0.0).all()


@pytest.mark.skipif(not GT_N96.is_file(), reason="gt_n96 cache absent")
def test_builder_crosscheck_helper_reports_road_hub():
    mod = _load_tool()
    _, stats = mod.build_edge_weights(GT_N96, v_band=1.0)
    cc = mod._crosscheck_ranking(stats["per_class_share_measured"])
    assert cc["road_is_hub_built"] is True
    assert cc["road_is_hub_feed_pa"] is True


# ---- trainer-consumption contract (the schema the loss path reads) --------------------------

def test_trainer_schema_contract(W_e):
    # the trainer reads json[...]["W_e"] as np.float32 and asserts shape (5,5) (L5801-5804).
    arr = W_e.astype(np.float32)
    assert arr.shape == (5, 5)
    assert np.isfinite(arr).all()
