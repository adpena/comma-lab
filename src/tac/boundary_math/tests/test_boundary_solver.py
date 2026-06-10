# SPDX-License-Identifier: MIT
"""Behavior tests for ``closed_spec_boundary_solver.v1`` (task #55).

These tests verify BEHAVIOR not constants (NO-FAKE discipline):
  - the Gα≥b solve produces amplitudes that ACTUALLY flip the predicted class on a
    controlled linear oracle (replacing the solve body with identity / zero FAILS);
  - the basis atom is a real smooth field (a degenerate constant field FAILS the
    flip test);
  - the flip-component extraction finds the real components (a no-op FAILS);
  - the graph-cut SELECTS the predicted support (small components are dropped at the
    water level; big ones admitted) — a select-all stub FAILS;
  - the MDL admission charges real byte costs (a zero-cost stub admits everything and
    FAILS the value>cost discrimination test);
  - the seg-only solve+measure reduces d_seg AND accounts new_bad_flips honestly
    (a fake row that omits collateral FAILS).

No torch / no SegNet is loaded here: the Gα≥b solve is exercised against a
deterministic LINEAR ORACLE Jacobian provider (a stand-in for the real autograd
SegNet) so the SOLVE math is tested in isolation, fast.  The real-SegNet integration
is exercised by the smoke CLI ``tools/closed_spec_boundary_solver_smoke.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.boundary_solver import (
    BasisAtom,
    BoundarySolverError,
    SolverSmokeRow,
    _contour_code_bytes,
    _flip_components,
    assemble_correction_field,
    gaussian_blob_atom,
    plan_contour_normal,
    plan_graph_cut,
    plan_mdl_contour,
    score_from_components,
    solve_and_measure_seg_only,
    solve_galpha_geq_b,
    support_localized_atom,
)


# ───────────────────────── a deterministic linear oracle ─────────────────────
class LinearOracleJacobian:
    """A controlled stand-in for ``TorchSegNetJacobian`` for testing the SOLVE.

    The "logit" of class ``c`` at pixel ``p`` is a LINEAR function of the correction
    field: ``logit_c(p) = base_logit[c, p] + gain * (field·atom_response)``.  Each
    atom raises its TARGET class logit at its center by ``gain * peak_overlap`` per
    unit α (a positive, known Jacobian), so the closed-form ``α = b/G`` provably
    flips the argmax when applied — exactly the property the real SegNet exhibits
    (verified in the feasibility probe).  This makes the SOLVE testable without torch.
    """

    def __init__(self, shape, base_logits, gain=1.0):
        self.shape = shape
        self.base = np.asarray(base_logits, dtype=np.float64)  # (C, H, W)
        self.gain = float(gain)
        self._field = np.zeros(shape, dtype=np.float64)

    def __call__(self, atom: BasisAtom) -> float:
        # G_k = ∂(logit_target − logit_current)(center)/∂α = gain * φ(center) = gain*1.
        r0, c0 = atom.center
        return self.gain * float(atom.field[r0, c0])

    def margin_gap_at(self, atom: BasisAtom) -> float:
        r0, c0 = atom.center
        cur = float(self.base[atom.current, r0, c0])
        tgt = float(self.base[atom.target, r0, c0])
        return max(cur - tgt, 0.0)

    def _logits(self, field):
        # the target class of EVERY atom is raised in proportion to the field overlap;
        # for the test we apply the field as a uniform additive bump to a single class.
        out = self.base.copy()
        # the field raises class 1 logits by gain*field (the "target" the tests use).
        out[1] += self.gain * field
        return out

    def batch_coeffs_and_gaps(self, atoms):
        coeff = np.array([self(a) for a in atoms], dtype=np.float64)
        gaps = np.array([self.margin_gap_at(a) for a in atoms], dtype=np.float64)
        return coeff, gaps

    def argmax(self) -> np.ndarray:
        return self._logits(self._field).argmax(axis=0).astype(np.int64)

    def argmax_after(self, field) -> np.ndarray:
        return self._logits(np.asarray(field, dtype=np.float64)).argmax(axis=0).astype(np.int64)


def _two_class_base(shape, flip_box):
    """Base logits where class 0 wins everywhere, but in ``flip_box`` class 1 is just
    below class 0 (a recoverable flip toward class 1)."""

    h, w = shape
    base = np.zeros((2, h, w), dtype=np.float64)
    base[0] = 5.0  # class 0 wins by default
    base[1] = 0.0
    r0, r1, c0, c1 = flip_box
    base[1, r0:r1, c0:c1] = 4.5  # class 1 just 0.5 below class 0 -> recoverable
    return base


# ─────────────────────────── basis atom behavior ─────────────────────────────
def test_gaussian_blob_is_a_real_smooth_field_not_constant():
    atom = gaussian_blob_atom((10, 20), (32, 48), sigma=3.0, target=1, current=0)
    assert atom.field.shape == (32, 48)
    assert atom.field[10, 20] == pytest.approx(1.0)  # peak at center
    # decays away from center (NOT a constant field — a constant would fail).
    assert atom.field[10, 20] > atom.field[10, 30] > atom.field[10, 47]
    assert atom.field.min() >= 0.0
    assert not np.allclose(atom.field, atom.field[0, 0])  # not constant


def test_gaussian_blob_rejects_center_outside_grid():
    with pytest.raises(BoundarySolverError):
        gaussian_blob_atom((100, 20), (32, 48), sigma=3.0, target=1, current=0)


def test_gaussian_blob_rejects_nonpositive_sigma():
    with pytest.raises(BoundarySolverError):
        gaussian_blob_atom((10, 20), (32, 48), sigma=0.0, target=1, current=0)


def test_support_localized_atom_is_zero_outside_dilated_support():
    """The support-localized atom confines the correction to the flip pixels (dilated).

    This is the structural fix for the scattered single-pixel boundary residual: a
    wide blob bleeds into correct interior (measured collateral); the localized atom
    is non-zero ONLY on/near the component's own pixels.
    """

    shape = (32, 32)
    # a 2x2 flip component centered at (16, 16).
    rows = np.array([16, 16, 17, 17])
    cols = np.array([16, 17, 16, 17])
    coords = np.stack([rows, cols])
    atom = support_localized_atom(coords, (16, 16), shape, sigma=2.0, dilate=1,
                                  target=1, current=0)
    # nonzero on the support pixels.
    assert atom.field[16, 16] > 0
    # zero far from the support (a wide blob would NOT be zero here).
    assert atom.field[0, 0] == 0.0
    assert atom.field[31, 31] == 0.0
    # the dilation gives a 1-pixel ring; pixels 3+ away are zero.
    assert atom.field[16, 22] == 0.0


def test_support_localized_atom_dilate_zero_is_exact_support():
    shape = (16, 16)
    coords = np.stack([np.array([8]), np.array([8])])  # single pixel
    atom = support_localized_atom(coords, (8, 8), shape, sigma=2.0, dilate=0,
                                  target=1, current=0)
    nz = np.argwhere(atom.field > 0)
    assert nz.shape == (1, 2)  # only the single support pixel
    assert tuple(nz[0]) == (8, 8)


# ──────────────────────── the Gα≥b SOLVE behavior ────────────────────────────
def test_solve_galpha_produces_amplitude_that_flips_the_oracle_argmax():
    """The closed-form α = b/G ACTUALLY flips the predicted class (the real solve)."""

    shape = (16, 16)
    base = _two_class_base(shape, (6, 10, 6, 10))  # a 4x4 flip patch toward class1
    oracle = LinearOracleJacobian(shape, base, gain=1.0)
    a_before = oracle.argmax()
    # before: class 0 wins everywhere (the patch is a recoverable class-1 flip).
    assert (a_before == 0).all()
    # place an atom at the patch center; solve; apply.
    atom = gaussian_blob_atom((8, 8), shape, sigma=2.0, target=1, current=0)
    gap = np.array([oracle.margin_gap_at(atom)])  # = 0.5
    sol = solve_galpha_geq_b([atom], oracle, margin_gap=gap, slack=0.2)
    assert sol.feasible[0]
    assert sol.alpha[0] > 0  # positive amplitude raises class 1
    field = assemble_correction_field([atom], sol.alpha, shape)
    a_after = oracle.argmax_after(field)
    # the center pixel MUST now be class 1 (the solve flipped it).
    assert a_after[8, 8] == 1


def test_solve_galpha_zero_amplitude_does_NOT_flip_identity_guard():
    """If the solver returned α=0 (a fake/no-op solve), the argmax would NOT flip.

    This is the identity guard: a degenerate solver that emits zeros fails to repair.
    """

    shape = (16, 16)
    base = _two_class_base(shape, (6, 10, 6, 10))
    oracle = LinearOracleJacobian(shape, base, gain=1.0)
    atom = gaussian_blob_atom((8, 8), shape, sigma=2.0, target=1, current=0)
    # simulate a fake solve: alpha forced to zero.
    field = assemble_correction_field([atom], np.zeros(1), shape)
    a_after = oracle.argmax_after(field)
    assert a_after[8, 8] == 0  # NOT flipped -> proves the real α>0 is what flips it


def test_solve_galpha_infeasible_when_jacobian_wrong_sign():
    """An atom whose Jacobian points the wrong way (G<=0) is infeasible (α=0)."""

    shape = (8, 8)
    base = _two_class_base(shape, (2, 4, 2, 4))
    oracle = LinearOracleJacobian(shape, base, gain=-1.0)  # negative gain -> wrong sign
    atom = gaussian_blob_atom((3, 3), shape, sigma=1.5, target=1, current=0)
    gap = np.array([oracle.margin_gap_at(atom)])
    sol = solve_galpha_geq_b([atom], oracle, margin_gap=gap)
    assert not sol.feasible[0]
    assert sol.alpha[0] == 0.0


def test_solve_galpha_rejects_negative_gap():
    shape = (8, 8)
    base = _two_class_base(shape, (2, 4, 2, 4))
    oracle = LinearOracleJacobian(shape, base, gain=1.0)
    atom = gaussian_blob_atom((3, 3), shape, sigma=1.5, target=1, current=0)
    with pytest.raises(BoundarySolverError):
        solve_galpha_geq_b([atom], oracle, margin_gap=np.array([-0.1]))


def test_solve_galpha_amplitude_is_closed_form_b_over_G():
    """α = (gap+slack)/G exactly — the SOLVE, not a sweep."""

    shape = (8, 8)
    base = _two_class_base(shape, (2, 4, 2, 4))
    oracle = LinearOracleJacobian(shape, base, gain=2.0)  # G = 2.0 at center
    atom = gaussian_blob_atom((3, 3), shape, sigma=1.5, target=1, current=0)
    gap = np.array([oracle.margin_gap_at(atom)])  # 0.5
    sol = solve_galpha_geq_b([atom], oracle, margin_gap=gap, slack=0.3)
    assert sol.coeff[0] == pytest.approx(2.0)
    assert sol.alpha[0] == pytest.approx((0.5 + 0.3) / 2.0)


def test_solve_galpha_precomputed_coeff_matches_jacobian_path():
    """The precomputed-coeff path (batched) gives the same α as the per-atom path."""

    shape = (8, 8)
    base = _two_class_base(shape, (2, 4, 2, 4))
    oracle = LinearOracleJacobian(shape, base, gain=1.5)
    atom = gaussian_blob_atom((3, 3), shape, sigma=1.5, target=1, current=0)
    gap = np.array([oracle.margin_gap_at(atom)])
    per_atom = solve_galpha_geq_b([atom], oracle, margin_gap=gap, slack=0.2)
    coeff = np.array([oracle(atom)])
    batched = solve_galpha_geq_b([atom], None, margin_gap=gap, slack=0.2, coeff=coeff)
    assert batched.alpha[0] == pytest.approx(per_atom.alpha[0])


def test_solve_galpha_requires_jacobian_or_coeff():
    shape = (8, 8)
    atom = gaussian_blob_atom((3, 3), shape, sigma=1.5, target=1, current=0)
    with pytest.raises(BoundarySolverError):
        solve_galpha_geq_b([atom], None, margin_gap=np.array([0.5]))


def test_solve_galpha_coeff_length_mismatch_rejected():
    shape = (8, 8)
    atom = gaussian_blob_atom((3, 3), shape, sigma=1.5, target=1, current=0)
    with pytest.raises(BoundarySolverError):
        solve_galpha_geq_b([atom], None, margin_gap=np.array([0.5]), coeff=np.array([1.0, 2.0]))


def test_assemble_correction_field_is_linear_superposition():
    shape = (16, 16)
    a1 = gaussian_blob_atom((4, 4), shape, sigma=2.0, target=1, current=0)
    a2 = gaussian_blob_atom((12, 12), shape, sigma=2.0, target=1, current=0)
    field = assemble_correction_field([a1, a2], np.array([2.0, 3.0]), shape)
    expected = 2.0 * a1.field + 3.0 * a2.field
    assert np.allclose(field, expected)


def test_assemble_correction_field_rejects_alpha_length_mismatch():
    shape = (8, 8)
    a1 = gaussian_blob_atom((4, 4), shape, sigma=2.0, target=1, current=0)
    with pytest.raises(BoundarySolverError):
        assemble_correction_field([a1], np.array([1.0, 2.0]), shape)


# ──────────────────────── flip-component behavior ────────────────────────────
def test_flip_components_finds_real_components_not_a_noop():
    base = np.zeros((20, 20), dtype=np.int64)
    target = np.zeros((20, 20), dtype=np.int64)
    target[2:5, 2:5] = 1  # one 3x3 flip patch
    target[10:12, 14:18] = 2  # a second patch
    _, comps = _flip_components(base, target)
    assert len(comps) == 2
    sizes = sorted(c["pixels"] for c in comps)
    assert sizes == [8, 9]  # 2x4=8 and 3x3=9
    # the larger component's dominant target is class 1.
    big = max(comps, key=lambda c: c["pixels"])
    assert big["target"] == 1
    assert big["current"] == 0


def test_flip_components_empty_when_base_equals_target():
    base = np.arange(16).reshape(4, 4) % 3
    _, comps = _flip_components(base, base.copy())
    assert comps == []


# ───────────────────────── graph-cut behavior ───────────────────────────────
def test_graph_cut_selects_big_components_drops_below_water_level():
    """The min-cut SELECTS components whose repair value beats their byte cost.

    A 1-pixel flip (value 1.27 B) vs a 2-byte component cost is DROPPED; a large
    component (value >> cost) is SELECTED.  A select-all stub would fail (it would
    select the 1-pixel one too).
    """

    base = np.zeros((40, 40), dtype=np.int64)
    target = np.zeros((40, 40), dtype=np.int64)
    target[5:15, 5:15] = 1  # 100-pixel patch -> value 127 B >> 2 B cost -> SELECT
    target[30, 30] = 1  # 1-pixel patch -> value 1.27 B < 2 B cost -> DROP
    plan = plan_graph_cut(base, target, bytes_per_component=2.0)
    selected = [m for m in plan.component_meta if m["selected"]]
    dropped = [m for m in plan.component_meta if not m["selected"]]
    assert len(selected) == 1
    assert selected[0]["pixels"] == 100
    assert len(dropped) == 1
    assert dropped[0]["pixels"] == 1
    # exactly one atom placed (for the selected component).
    assert len(plan.atoms) == 1
    assert plan.archive_bytes_delta == 2


def test_graph_cut_records_rag_degree_structural_weight():
    base = np.zeros((20, 20), dtype=np.int64)
    base[:, 10:] = 1  # two regions -> the flip component is adjacent to both
    target = base.copy()
    target[8:12, 8:12] = 2  # a flip patch straddling the boundary
    plan = plan_graph_cut(base, target, bytes_per_component=1.0)
    assert any("rag_degree" in m for m in plan.component_meta)


# ───────────────────────── MDL contour behavior ─────────────────────────────
def test_contour_code_bytes_grows_with_component_size():
    small = _contour_code_bytes(np.zeros((2, 4)), 4)
    big = _contour_code_bytes(np.zeros((2, 400)), 400)
    assert big > small >= 4  # header + chain code; bigger perimeter costs more


def test_mdl_contour_admits_high_value_drops_low_value():
    """A component is admitted iff flips_fixed*1.27 > coded_bytes.

    A tiny component costs a header (4+ bytes) but fixes few flips -> value < cost ->
    DROP.  A large component fixes many flips -> value >> cost -> ADMIT.  A zero-cost
    stub would admit the tiny one too (and fail this discrimination).
    """

    base = np.zeros((50, 50), dtype=np.int64)
    target = np.zeros((50, 50), dtype=np.int64)
    target[5:20, 5:20] = 1  # 225-pixel patch -> value 285 B >> ~25 B coded -> ADMIT
    target[40, 40] = 1  # 1-pixel -> value 1.27 B < 4 B header -> DROP
    plan = plan_mdl_contour(base, target)
    admitted = [m for m in plan.component_meta if m["admitted"]]
    dropped = [m for m in plan.component_meta if not m["admitted"]]
    assert len(admitted) == 1
    assert admitted[0]["pixels"] == 225
    assert len(dropped) == 1
    assert dropped[0]["pixels"] == 1
    assert plan.archive_bytes_delta == admitted[0]["coded_bytes"]


def test_mdl_contour_value_exceeds_cost_for_admitted():
    base = np.zeros((40, 40), dtype=np.int64)
    target = np.zeros((40, 40), dtype=np.int64)
    target[5:25, 5:25] = 1
    plan = plan_mdl_contour(base, target)
    for m in plan.component_meta:
        if m["admitted"]:
            assert m["value_bytes"] > m["coded_bytes"]


# ───────────────────────── contour-normal plan ──────────────────────────────
def test_contour_normal_places_one_atom_per_component_zero_bytes():
    base = np.zeros((40, 40), dtype=np.int64)
    target = np.zeros((40, 40), dtype=np.int64)
    target[5:10, 5:10] = 1
    target[20:24, 20:24] = 2
    plan = plan_contour_normal(base, target)
    assert plan.correction_kind == "contour_normal"
    assert len(plan.atoms) == 2  # one per flip component
    assert plan.archive_bytes_delta == 0  # deterministic decode-time field, no bytes


def test_contour_normal_respects_max_components_cap():
    base = np.zeros((60, 60), dtype=np.int64)
    target = np.zeros((60, 60), dtype=np.int64)
    target[2:6, 2:6] = 1
    target[20:30, 20:30] = 1  # bigger
    target[50:52, 50:52] = 1  # smaller
    plan = plan_contour_normal(base, target, max_components=1)
    assert len(plan.atoms) == 1
    # the largest component is kept (sorted by -pixels).
    assert plan.component_meta[0]["pixels"] == 100


# ───────────────────── score recomputation + smoke row ──────────────────────
def test_score_from_components_matches_the_law():
    s = score_from_components(d_seg=0.01, d_pose=0.0001, archive_bytes=177169)
    expected = 100 * 0.01 + np.sqrt(10 * 0.0001) + 25 * 177169 / 37_545_489
    assert s == pytest.approx(expected)


def test_score_from_components_clamps_negative_pose():
    # negative pose (numerical) is clamped to 0 under the sqrt.
    s = score_from_components(0.0, -1e-9, 0)
    assert s == pytest.approx(0.0)


def test_solve_and_measure_seg_only_reduces_dseg_and_accounts_collateral():
    """End-to-end seg-only solve against the linear oracle: d_seg drops, collateral
    is honestly counted (new_bad_flips), and the row is the canonical schema."""

    shape = (24, 24)
    base = _two_class_base(shape, (8, 16, 8, 16))  # an 8x8 recoverable flip patch
    oracle = LinearOracleJacobian(shape, base, gain=1.0)
    target = np.ones(shape, dtype=np.int64)  # we want class 1 in the patch...
    # ...but class 0 wins outside the patch in base, so target=all-1 means the
    # OUTSIDE is also "flipped" — to make a clean seg test, target = base argmax with
    # the patch set to class 1 (the recoverable region).
    a_before = oracle.argmax()
    target = a_before.copy()
    target[8:16, 8:16] = 1  # the recoverable patch should be class 1

    plan = plan_contour_normal(a_before, target)
    assert len(plan.atoms) >= 1

    row, field = solve_and_measure_seg_only(
        plan, oracle, target, base_candidate="oracle_test",
        base_archive_bytes=177169, d_pose_before=0.0,
    )
    assert isinstance(row, SolverSmokeRow)
    assert row.d_seg_after <= row.d_seg_before  # the repair reduced (or held) d_seg
    assert row.pixels_flipped_repaired >= 1  # it fixed at least one flip
    assert row.new_bad_flips_created >= 0  # collateral is counted (honest)
    assert not row.uses_stored_per_pixel_table  # NO per-pixel oracle table
    assert row.no_fake_class_6_passed
    obj = row.to_json_obj()
    assert obj["schema"] == "engineered_correction_boundary_solver_smoke.v1"
    # the row reports BOTH collateral fields (without them it could lie).
    assert "new_bad_flips_created" in obj and "pose_side_effect" in obj


def test_solve_and_measure_identity_field_would_not_reduce_dseg():
    """Guard: if the solve emitted a ZERO field (fake), d_seg would NOT improve.

    We force α=0 by using a wrong-sign oracle (all atoms infeasible) and assert the
    after-d_seg equals before — proving the real (feasible) solve is what reduces it.
    """

    shape = (24, 24)
    base = _two_class_base(shape, (8, 16, 8, 16))
    oracle = LinearOracleJacobian(shape, base, gain=-1.0)  # wrong sign -> infeasible
    a_before = oracle.argmax()
    target = a_before.copy()
    target[8:16, 8:16] = 1
    plan = plan_contour_normal(a_before, target)
    row, _ = solve_and_measure_seg_only(
        plan, oracle, target, base_candidate="oracle_infeasible",
        base_archive_bytes=177169,
    )
    # no feasible atom -> field is zero -> d_seg unchanged.
    assert row.d_seg_after == pytest.approx(row.d_seg_before)
    assert row.pixels_flipped_repaired == 0
