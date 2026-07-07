"""sigma_ij per-class-pair LENGTH-WEIGHT lever tests (junction_young_angle_sigma_fit_v1
consumption path, 2026-07-07).

LOAD-BEARING contract under test:
1. Default ("all-ones") is BYTE-IDENTICAL — both by code path (resolver -> None => the
   pre-existing unweighted branch) AND bitwise (an explicit all-ones matrix through the sigma
   branch reproduces the default length/eik values bit-for-bit on MLX CPU).
2. The fitted-20260707 preset carries the EXACT full-precision measured values (fit JSON,
   commit 3571e5b65), NaN-unobserved pairs filled with the 1.0 null, consistent with the
   registered equation constants.
3. JSON round-trip (raw 5x5 + the fit tool's own JSON shape).
4. Fail-closed validation: wrong shape / non-symmetric / non-positive / non-finite refused;
   the DSL LengthSigma factory refuses 'all-ones' (silent-no-op lever) and malformed specs.

Adversarial note (would these pass if broken?): test 1 fails if the sigma branch perturbs the
numerics; test_wrong_shape fails if the runtime shape guard is dropped; the refusal tests fail
if validation silently accepts malformed input.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)  # bit-identical CPU-locked proofs (MLX-GPU is not crossproc-stable)

from tac.boundary_math.length_sigma import (  # noqa: E402
    FITTED_20260707_MATRIX,
    N_CLASSES,
    PRESET_ALL_ONES,
    PRESET_FITTED_20260707,
    describe_length_sigma,
    resolve_length_sigma_matrix,
    validate_sigma_matrix,
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_TRAINER_PATH = os.path.join(_REPO, "experiments", "train_levelset_witness_realized_through_R_mlx.py")
_spec = importlib.util.spec_from_file_location("_lvl_trainer_for_sigma_test", _TRAINER_PATH)
_lvl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lvl)
_EIKONAL_LENGTH = _lvl._eikonal_length_mlx

_FIT_JSON = os.path.join(
    _REPO, "experiments", "results", "solver_pack_20260707", "junction_sigma",
    "junction_sigma_fit.json")

# the 7 MEASURED fitted pairs (full precision, from the fit JSON; commit 3571e5b65)
_EXPECTED_PAIRS = {
    (0, 1): 0.3771195466360733,   # Road-Lane (the headline; excludes 1.0)
    (0, 2): 1.0848087450168646,   # Road-Undrivable
    (0, 3): 1.0062627915225708,   # Road-Movable
    (0, 4): 1.7791690170773755,   # Road-MyCar (excludes 1.0)
    (1, 2): 0.7381986449045815,   # Lane-Undrivable (excludes 1.0)
    (1, 4): 1.764344211480968,    # Lane-MyCar (excludes 1.0)
    (2, 3): 1.0482927871960461,   # Undrivable-Movable
}
_UNOBSERVED_PAIRS = ((1, 3), (2, 4), (3, 4))  # Lane-Movable, Undrivable-MyCar, Movable-MyCar


def _phi(h=8, w=10, k=5, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((h * w, k)).astype(np.float32))


# ---------------------------------------------------------------------------
# 1. bitwise identity at all-ones (LOAD-BEARING)
# ---------------------------------------------------------------------------
def test_resolver_all_ones_is_none_code_path():
    assert resolve_length_sigma_matrix(PRESET_ALL_ONES) is None
    assert resolve_length_sigma_matrix(" all-ones ") is None  # whitespace-tolerant


def test_all_ones_matrix_bitwise_identical_to_default():
    """Explicit all-ones matrix through the sigma branch == the default unweighted branch,
    BITWISE, for length AND eikonal, on multiple seeds/shapes."""
    for seed, (h, w) in [(0, (8, 10)), (1, (12, 7)), (2, (5, 5))]:
        phi = _phi(h, w, seed=seed)
        e0, l0, g0 = _EIKONAL_LENGTH(phi, h, w)
        e1, l1, g1 = _EIKONAL_LENGTH(phi, h, w, sigma_matrix=mx.ones((5, 5)))
        mx.eval(e0, l0, g0, e1, l1, g1)
        assert np.array(l0).tobytes() == np.array(l1).tobytes(), f"length bitwise diff seed={seed}"
        assert np.array(e0).tobytes() == np.array(e1).tobytes(), f"eik bitwise diff seed={seed}"
        assert np.array(g0).tobytes() == np.array(g1).tobytes(), f"g bitwise diff seed={seed}"


def test_sigma_none_kwarg_identical_to_positional_default():
    phi = _phi()
    e0, l0, _ = _EIKONAL_LENGTH(phi, 8, 10)
    e1, l1, _ = _EIKONAL_LENGTH(phi, 8, 10, sigma_matrix=None)
    mx.eval(e0, l0, e1, l1)
    assert np.array(l0).tobytes() == np.array(l1).tobytes()
    assert np.array(e0).tobytes() == np.array(e1).tobytes()


# ---------------------------------------------------------------------------
# 2. fitted preset exact values + equation cross-consistency
# ---------------------------------------------------------------------------
def test_fitted_preset_exact_values():
    a = resolve_length_sigma_matrix(PRESET_FITTED_20260707)
    assert a is not None and a.shape == (N_CLASSES, N_CLASSES)
    for (i, j), v in _EXPECTED_PAIRS.items():
        assert a[i, j] == v, f"pair ({i},{j}): {a[i, j]!r} != {v!r}"
        assert a[j, i] == v, "symmetry"
    for (i, j) in _UNOBSERVED_PAIRS:
        assert a[i, j] == 1.0 and a[j, i] == 1.0, "unobserved pair must be the all-ones null"
    assert np.array_equal(a, a.T)
    assert np.array_equal(a, np.asarray(FITTED_20260707_MATRIX, dtype=np.float64))


def test_fitted_preset_consistent_with_registered_equation():
    """Full-precision preset agrees with the (rounded) registered equation constants."""
    from tac.canonical_equations.junction_young_sigma_and_powerlaw_exit_20260707 import (
        FITTED_SIGMA_MATRIX,
    )
    a = resolve_length_sigma_matrix(PRESET_FITTED_20260707)
    eq = np.asarray(FITTED_SIGMA_MATRIX, dtype=np.float64)
    off = ~np.eye(N_CLASSES, dtype=bool)
    fitted_mask = off & np.isfinite(eq)
    assert np.all(np.abs(a[fitted_mask] - eq[fitted_mask]) < 1e-3), (
        "preset drifted from the registered equation constants")
    # unobserved (NaN in the equation) => the 1.0 null in the preset
    assert np.all(a[off & ~np.isfinite(eq)] == 1.0)


# ---------------------------------------------------------------------------
# 3. JSON round-trip
# ---------------------------------------------------------------------------
def test_json_round_trip_raw_list(tmp_path):
    a = resolve_length_sigma_matrix(PRESET_FITTED_20260707)
    p = tmp_path / "sigma.json"
    p.write_text(json.dumps(a.tolist()))
    b = resolve_length_sigma_matrix(str(p))
    assert np.array_equal(a, b)


def test_json_dict_form_with_nan_filled(tmp_path):
    mat = [[0.0 if i == j else (float("nan") if (i, j) in ((0, 1), (1, 0)) else 1.5)
            for j in range(5)] for i in range(5)]
    p = tmp_path / "sigma_dict.json"
    p.write_text(json.dumps({"sigma_matrix_5x5": mat}))
    b = resolve_length_sigma_matrix(str(p))
    assert b[0, 1] == 1.0 and b[1, 0] == 1.0  # NaN -> the null
    assert b[2, 3] == 1.5


def test_actual_fit_json_passes_directly():
    """The primary measured artifact resolves as-is (NaN pairs filled, symmetric)."""
    if not os.path.isfile(_FIT_JSON):
        pytest.skip("fit JSON artifact not present")
    b = resolve_length_sigma_matrix(_FIT_JSON)
    assert b[0, 1] == _EXPECTED_PAIRS[(0, 1)]
    for (i, j) in _UNOBSERVED_PAIRS:
        assert b[i, j] == 1.0
    assert np.array_equal(b, b.T)


# ---------------------------------------------------------------------------
# 4. fail-closed validation
# ---------------------------------------------------------------------------
def test_validate_refuses_wrong_shape():
    with pytest.raises(ValueError, match="5x5"):
        validate_sigma_matrix(np.ones((4, 4)))
    with pytest.raises(ValueError, match="5x5"):
        validate_sigma_matrix(np.ones((5,)))


def test_validate_refuses_non_symmetric():
    a = np.ones((5, 5))
    a[0, 1] = 0.5
    with pytest.raises(ValueError, match="not symmetric"):
        validate_sigma_matrix(a)


def test_validate_refuses_non_positive():
    a = np.ones((5, 5))
    a[0, 1] = a[1, 0] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_sigma_matrix(a)
    a[0, 1] = a[1, 0] = -0.3
    with pytest.raises(ValueError, match="non-positive"):
        validate_sigma_matrix(a)


def test_validate_refuses_non_finite():
    a = np.ones((5, 5))
    a[0, 1] = a[1, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_sigma_matrix(a)


def test_resolver_refuses_missing_path_and_bad_json(tmp_path):
    with pytest.raises(ValueError, match="neither a preset"):
        resolve_length_sigma_matrix("no-such-preset-or-file")
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        resolve_length_sigma_matrix(str(p))
    p2 = tmp_path / "nokey.json"
    p2.write_text(json.dumps({"something_else": 1}))
    with pytest.raises(ValueError, match="sigma_matrix_5x5"):
        resolve_length_sigma_matrix(str(p2))


def test_term_refuses_wrong_shape_sigma():
    phi = _phi()
    with pytest.raises(ValueError, match="length-sigma-matrix shape"):
        _EIKONAL_LENGTH(phi, 8, 10, sigma_matrix=mx.ones((4, 4)))


# ---------------------------------------------------------------------------
# 5. the treatment actually acts + gradients flow
# ---------------------------------------------------------------------------
def test_fitted_sigma_changes_length_only_and_grads_flow():
    phi = _phi()
    sig = mx.array(resolve_length_sigma_matrix(PRESET_FITTED_20260707).astype(np.float32))
    e0, l0, _ = _EIKONAL_LENGTH(phi, 8, 10)
    e1, l1, _ = _EIKONAL_LENGTH(phi, 8, 10, sigma_matrix=sig)
    mx.eval(e0, l0, e1, l1)
    assert float(l1) != float(l0), "fitted sigma must change the length term"
    assert np.array(e1).tobytes() == np.array(e0).tobytes(), "eikonal must be untouched by sigma"

    def loss(p):
        _, ln, _ = _EIKONAL_LENGTH(p, 8, 10, sigma_matrix=sig)
        return ln

    g = mx.grad(loss)(phi)
    mx.eval(g)
    assert bool(mx.all(mx.isfinite(g)))
    assert float(mx.sum(mx.abs(g))) > 0.0


def test_sigma_gather_matches_independent_numpy_reference():
    """Independent numpy fp32 mirror of the documented math (forward diffs, same slices,
    per-pixel sigma[top1,top2] gather) — catches gather misalignment / flat-index bugs that
    the all-ones bitwise test is insensitive to. Distinct off-diagonal values per pair so a
    wrong pair lookup CANNOT agree."""
    h, w, k = 9, 11, 5
    rng = np.random.default_rng(7)
    phi_np = rng.standard_normal((h * w, k)).astype(np.float32)
    # distinct symmetric sigma: sigma[i,j] = 1 + 0.1*(i+j) + 0.01*i*j (off-diag all distinct)
    sig_np = np.ones((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            if i != j:
                sig_np[i, j] = 1.0 + 0.1 * (i + j) + 0.01 * i * j
    e_mx, l_mx, _ = _EIKONAL_LENGTH(mx.array(phi_np), h, w, sigma_matrix=mx.array(sig_np))
    mx.eval(e_mx, l_mx)

    # --- independent numpy mirror (fp32) ---
    phi = phi_np.reshape(h, w, k)
    srt = np.sort(phi, axis=-1)
    m = srt[..., -1] - srt[..., -2]
    gy = m[1:, :] - m[:-1, :]
    gx = m[:, 1:] - m[:, :-1]
    gmag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2 + np.float32(1e-8))
    mc = m[:-1, :-1]
    len_eps = np.float32(1.0)
    delta = (len_eps / np.float32(np.pi)) / (len_eps * len_eps + mc * mc)
    order = np.argsort(phi, axis=-1)
    i1 = order[..., -1][:-1, :-1]
    i2 = order[..., -2][:-1, :-1]
    sig_px = sig_np[i1, i2]
    length_ref = float(np.mean(sig_px * (delta * gmag)))
    assert abs(float(l_mx) - length_ref) < 1e-6, (float(l_mx), length_ref)


def test_describe_provenance_record():
    d0 = describe_length_sigma(PRESET_ALL_ONES)
    assert d0["active"] is False
    d1 = describe_length_sigma(PRESET_FITTED_20260707)
    assert d1["active"] is True
    assert d1["sigma_road_lane"] == _EXPECTED_PAIRS[(0, 1)]
    assert math.isfinite(d1["offdiag_geomean"])


# ---------------------------------------------------------------------------
# 6. DSL LengthSigma Lever factory
# ---------------------------------------------------------------------------
def test_length_sigma_lever_default_is_fitted_treatment():
    from tac.witness_dsl.curriculum_dsl import LengthSigma, Lever

    lv = LengthSigma()
    assert isinstance(lv, Lever)
    assert lv.name == "FEED_08a_length_sigma"
    assert lv.overrides == {"--length-sigma-matrix": PRESET_FITTED_20260707}
    assert lv.epochs_delta == 0  # loss-geometry config change, no epoch budget


def test_length_sigma_lever_refuses_all_ones_silent_noop():
    from tac.witness_dsl.curriculum_dsl import LengthSigma

    with pytest.raises(ValueError, match="silent no-op"):
        LengthSigma(PRESET_ALL_ONES)


def test_length_sigma_lever_refuses_malformed(tmp_path):
    from tac.witness_dsl.curriculum_dsl import LengthSigma

    with pytest.raises(ValueError, match="LengthSigma"):
        LengthSigma("bogus-preset")
    bad = tmp_path / "asym.json"
    a = np.ones((5, 5)).tolist()
    a[0][1] = 0.5
    bad.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="not symmetric"):
        LengthSigma(str(bad))


def test_length_sigma_lever_accepts_valid_json_path(tmp_path):
    from tac.witness_dsl.curriculum_dsl import LengthSigma

    p = tmp_path / "ok.json"
    a = np.full((5, 5), 1.2)
    np.fill_diagonal(a, 1.0)
    p.write_text(json.dumps(a.tolist()))
    lv = LengthSigma(str(p))
    assert lv.overrides["--length-sigma-matrix"] == str(p)


def test_trainer_argparse_has_the_flag():
    """never-invent-flags: the flag the lever emits exists in the trainer's argparse."""
    src = open(_TRAINER_PATH).read()
    assert '"--length-sigma-matrix"' in src


def test_lever_registry_discovers_length_sigma():
    from tac.witness_dsl.lever_registry import lever_factories

    assert "LengthSigma" in lever_factories()
