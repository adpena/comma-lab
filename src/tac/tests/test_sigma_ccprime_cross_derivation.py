"""sigma_cc' SECOND-DERIVATION (fragility) + GENERALIZATION-classification tests
(task #382, FEED-sigma-ccprime, 2026-07-09).

LOAD-BEARING contract under test:
1. The fragility law is a DETERMINISTIC reduction of the measured junction artifact and
   reproduces the hardcoded ``fragility-20260709`` preset BIT-FOR-BIT (no drift).
2. geometric-mean-1 gauge holds over the OBSERVED off-diagonal pairs; unobserved pairs are the
   1.0 null; the matrix is symmetric and fail-closed-valid.
3. The CROSS-DERIVATION FINDING: Young's-angle and fragility DISAGREE on Road-Lane (0.377 vs
   1.029) but AGREE that the thin Lane-Undrivable sliver is lowered (<1) — the load-bearing
   result that the angle-based Herring derivation is not interchangeable with an abundance proxy.
4. The GENERALIZATION classification (sigma==1 -> byte-identical control path) is preserved: the
   new preset does NOT disturb the 'all-ones' -> None byte-identity, and the DSL LengthSigma
   lever routes the new preset with no invented flag.
5. The FORMALIZATION_PENDING equation module builds and carries the disagreement anchor.

Adversarial note (would these pass if broken?): the bit-consistency test fails if the preset or
the reduction drifts; the disagreement test fails if the fragility law is silently replaced by
the Young's fit; the byte-identity test fails if the new preset leaks into the control path.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tac.boundary_math.length_sigma import (
    FRAGILITY_20260709_K,
    FRAGILITY_20260709_MATRIX,
    N_CLASSES,
    PRESET_ALL_ONES,
    PRESET_FRAGILITY_20260709,
    PRESETS,
    derive_fragility_sigma_from_junction_fit,
    describe_length_sigma,
    resolve_length_sigma_matrix,
    validate_sigma_matrix,
)

_FIT_JSON = "experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json"
# Canonical class order Road0 / Lane1 / Undrivable2 / Movable3 / MyCar4.
_RL, _LU, _RU = (0, 1), (1, 2), (0, 2)
_UM_UNOBSERVED = (2, 4)  # Undrivable-MyCar: 0 junctions -> 1.0 null


def _synthetic_fit(per_triple: dict) -> dict:
    return {"fit": {"per_triple": per_triple}}


# --- 1. preset registration + resolver routing ------------------------------------------------
def test_fragility_preset_is_registered_in_presets_tuple():
    assert PRESET_FRAGILITY_20260709 in PRESETS
    assert PRESET_FRAGILITY_20260709 == "fragility-20260709"


def test_fragility_preset_resolves_valid_symmetric_matrix():
    a = resolve_length_sigma_matrix(PRESET_FRAGILITY_20260709)
    assert a is not None
    assert a.shape == (N_CLASSES, N_CLASSES)
    validate_sigma_matrix(a, source="test")  # symmetric / positive / finite off-diagonal
    assert np.array_equal(a, a.T)


# --- 2. the control path (generalization) is UNDISTURBED --------------------------------------
def test_all_ones_still_none_byte_identity_preserved():
    # The generalization classification hinges on this: sigma==1 -> pre-existing branch.
    assert resolve_length_sigma_matrix(PRESET_ALL_ONES) is None


def test_fragility_preset_is_not_all_ones():
    a = resolve_length_sigma_matrix(PRESET_FRAGILITY_20260709)
    off = ~np.eye(N_CLASSES, dtype=bool)
    assert not np.allclose(a[off], 1.0)  # it is a genuine treatment, not a disguised control


# --- 3. deterministic reduction reproduces the preset bit-for-bit ------------------------------
def test_derivation_reproduces_hardcoded_preset_bitwise():
    derived = derive_fragility_sigma_from_junction_fit(_FIT_JSON)
    preset = np.asarray(FRAGILITY_20260709_MATRIX, dtype=np.float64)
    assert np.array_equal(derived, preset), "preset drifted from the derivation producer"


def test_derivation_accepts_dict_and_path_identically():
    obj = json.loads(open(_FIT_JSON).read())
    from_path = derive_fragility_sigma_from_junction_fit(_FIT_JSON)
    from_dict = derive_fragility_sigma_from_junction_fit(obj)
    assert np.array_equal(from_path, from_dict)


# --- 4. gauge + geometry properties -----------------------------------------------------------
def test_geometric_mean_one_over_observed_offdiagonal():
    a = resolve_length_sigma_matrix(PRESET_FRAGILITY_20260709)
    off = ~np.eye(N_CLASSES, dtype=bool)
    obs = off.copy()
    obs[_UM_UNOBSERVED] = obs[_UM_UNOBSERVED[::-1]] = False  # exclude the 1.0-null unobserved pair
    gm = np.exp(np.mean(np.log(a[obs])))
    assert gm == pytest.approx(1.0, abs=1e-12)


def test_unobserved_pair_is_null_one():
    a = resolve_length_sigma_matrix(PRESET_FRAGILITY_20260709)
    assert a[_UM_UNOBSERVED] == 1.0  # Undrivable-MyCar has 0 junctions


def test_diagonal_is_one():
    a = resolve_length_sigma_matrix(PRESET_FRAGILITY_20260709)
    assert np.array_equal(np.diag(a), np.ones(N_CLASSES))


# --- 5. THE CROSS-DERIVATION FINDING (the load-bearing result) --------------------------------
def test_cross_derivation_disagrees_on_road_lane():
    # Young's-angle LOWERS Road-Lane to 0.377; fragility does NOT (abundance dilutes the drops).
    a = resolve_length_sigma_matrix(PRESET_FRAGILITY_20260709)
    assert a[_RL] > 1.0  # fragility Road-Lane ~ 1.029, NOT lowered
    assert abs(a[_RL] - 0.377) > 0.5  # far from the Young's-angle value -> the laws disagree


def test_cross_derivation_agrees_lane_undrivable_sliver_is_lowered():
    # Both laws LOWER the thin Lane-Undrivable sliver (Young's 0.738; fragility ~0.710).
    a = resolve_length_sigma_matrix(PRESET_FRAGILITY_20260709)
    assert a[_LU] < 1.0
    assert a[_LU] < a[_RU]  # sliver interface stiffer-relieved than the bulk Road-Undrivable


# --- 6. k gauge monotonicity (the derivation LAW behaves) -------------------------------------
def test_k_gauge_monotonic_spread():
    # Larger k => sigma spreads further from 1 (more fragility weighting), gauge still geomean-1.
    lo = derive_fragility_sigma_from_junction_fit(_FIT_JSON, k=0.5)
    hi = derive_fragility_sigma_from_junction_fit(_FIT_JSON, k=2.0)
    off = ~np.eye(N_CLASSES, dtype=bool)
    spread_lo = np.std(np.log(lo[off & (lo != 1.0)]))
    spread_hi = np.std(np.log(hi[off & (hi != 1.0)]))
    assert spread_hi > spread_lo
    assert FRAGILITY_20260709_K == 1.0  # the preset's DERIVED gauge


# --- 7. fail-closed on malformed inputs -------------------------------------------------------
def test_derivation_refuses_empty_per_triple():
    with pytest.raises(ValueError, match="per_triple"):
        derive_fragility_sigma_from_junction_fit(_synthetic_fit({}))


def test_derivation_refuses_bad_class_ids():
    bad = _synthetic_fit({"T": {"class_ids": [0, 1], "n_junctions": 5, "n_dropped_arc_ge_180": 1}})
    with pytest.raises(ValueError, match="class_ids"):
        derive_fragility_sigma_from_junction_fit(bad)


def test_derivation_refuses_nonpositive_k():
    with pytest.raises(ValueError, match="k must be"):
        derive_fragility_sigma_from_junction_fit(_FIT_JSON, k=0.0)


def test_derivation_refuses_non_dict_non_path():
    with pytest.raises(ValueError, match="dict or JSON path"):
        derive_fragility_sigma_from_junction_fit(1234)  # type: ignore[arg-type]


def test_derivation_refuses_missing_file():
    with pytest.raises(ValueError, match="not an existing"):
        derive_fragility_sigma_from_junction_fit("does/not/exist_sigma.json")


# --- 8. synthetic sanity: a fully-fragile pair gets sigma < a robust pair ----------------------
def test_synthetic_fragile_pair_gets_lower_sigma():
    # triple {0,1,2}: pair(0,1) all-dropped (f=1) vs pair(1,2)/(0,2) robust (f=0).
    per = {"T": {"class_ids": [0, 1, 2], "n_junctions": 0, "n_dropped_arc_ge_180": 100},
           "U": {"class_ids": [0, 2, 3], "n_junctions": 100, "n_dropped_arc_ge_180": 0}}
    a = derive_fragility_sigma_from_junction_fit(_synthetic_fit(per))
    # pair (0,1): only appears in T (f=1) -> exp(-1); pair (2,3): only in U (f=0) -> exp(0).
    assert a[0, 1] < a[2, 3]


# --- 9. describe() provenance ------------------------------------------------------------------
def test_describe_fragility_records_formalization_pending_provenance():
    d = describe_length_sigma(PRESET_FRAGILITY_20260709)
    assert d["active"] is True
    assert "FORMALIZATION_PENDING" in d["provenance"]
    assert d["offdiag_geomean"] == pytest.approx(1.0, abs=1e-9)


# --- 10. DSL lever routes the new preset (no invented flag) ------------------------------------
def test_dsl_length_sigma_lever_routes_fragility_preset():
    from tac.witness_dsl.curriculum_dsl import LengthSigma

    lev = LengthSigma(PRESET_FRAGILITY_20260709)
    assert lev.overrides == {"--length-sigma-matrix": PRESET_FRAGILITY_20260709}


# --- 11. equations leg (FORMALIZATION_PENDING) builds -----------------------------------------
def test_sigma_ccprime_equations_build_and_carry_disagreement():
    from tac.canonical_equations.sigma_ccprime_generalization_20260709 import (
        build_sigma_ccprime_fragility_cross_derivation_v1,
        build_sigma_ccprime_length_generalization_v1,
    )

    gen = build_sigma_ccprime_length_generalization_v1()
    cross = build_sigma_ccprime_fragility_cross_derivation_v1()
    assert gen.equation_id == "sigma_ccprime_length_generalization_v1"
    assert cross.equation_id == "sigma_ccprime_fragility_cross_derivation_v1"
    disagree = list(cross.predicted_vs_empirical_residual.values())[0]
    assert disagree == pytest.approx(0.6517, abs=1e-3)  # |0.377 - 1.029|
