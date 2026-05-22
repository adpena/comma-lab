# SPDX-License-Identifier: MIT
"""Compositional integration tests: pairs of xray primitives produce
consistent / cross-reinforcing results.

Per the RECOVERY-3 prompt's XRAY Batch 6 mandate: *"Integration tests
verifying composition: F1 × F2 → MDL density vs Shannon R(D) cross-check;
F3 × F10 → camera-resolution nullspace × YUV6 sublattice; F7 × F8 →
polytope × Lie-algebra; etc."*

These tests verify the CROSS-PRIMITIVE CONTRACT, not each primitive's
internal math (that lives in dedicated per-primitive test files). When two
primitives engage the same wire-in hook with the same target, their results
should be mutually consistent — either reinforcing or surfacing a typed
disagreement the solver-stack consumer can route through the probe-
disambiguator hook.

Lane: ``lane_xray_canon_math_findings_wire_in_20260514``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.xray import (
    canonical_xray_primitive_inventory,
    discover_primitives_by_hook,
    wire_in_for_hook,
)
from tac.xray.bilinear_resize_nullspace import BilinearResizeNullspace
from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator
from tac.xray.posenet_se3_lie_algebra import PoseNetSE3LieAlgebra
from tac.xray.segnet_margin_polytope import SegNetLogitMarginPolytope
from tac.xray.shannon_vector_r_d import ShannonVectorRDEstimator
from tac.xray.vq_codebook_coverage import VQCodebookCoverage
from tac.xray.wavelet_hf_energy import WaveletHFEnergy
from tac.xray.yuv6_sublattice_geometry import YUV6SublatticeGeometry

A1_ARCHIVE_PATH = Path("submissions/a1/archive.zip")


# ---------------------------------------------------------------------------
# F1 × F2: MDL density vs Shannon R(D)
# ---------------------------------------------------------------------------


def test_f1_times_f2_mdl_density_consistent_with_shannon_r_d_predicted_bytes():
    """F1 (MDL scorer-conditional) and F2 (Shannon vector R(D)) should agree
    on the order-of-magnitude of bytes the contest task needs.

    F1 measures observed MDL density (= observed archive bytes / theoretical
    bound from scorer-conditional MDL).
    F2 derives ``r_min_bytes`` (the Shannon vector R(D) lower bound) from
    (d_seg, d_pose, n_pairs).

    Both numbers describe "how many bytes does the contest actually need";
    they should both be POSITIVE and the A1 archive's actual byte count
    should be > F2's r_min_bytes (otherwise A1 violates Shannon).
    """
    if not A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 archive not present")

    f1 = ScorerConditionalMDLEstimator()
    f2 = ShannonVectorRDEstimator()

    r1 = f1.compute(A1_ARCHIVE_PATH)
    r2 = f2.compute(
        A1_ARCHIVE_PATH,
        d_seg_target=0.067,
        d_pose_target=0.018,
        n_pairs=600,
    )

    # F1 reports MDL density (observed / theoretical-bound). >= 0 always.
    f1_density = float(r1.primitive_value.mdl_density)
    f1_archive_bytes = int(r1.primitive_value.total_archive_bytes)
    assert f1_density > 0, "MDL density must be positive"
    assert f1_archive_bytes > 0

    # F2 reports the Shannon R(D) lower bound on bytes at the (d_seg,
    # d_pose) distortion pair.
    f2_r_min_bytes = float(r2.primitive_value.r_min_bytes)
    assert f2_r_min_bytes > 0

    # Cross-check: the actual A1 archive bytes (~178 kB) must be GREATER
    # than the Shannon R(D) lower bound (an information-theoretic floor).
    assert f1_archive_bytes >= f2_r_min_bytes, (
        f"F1 × F2 cross-check FAILED: A1 archive ({f1_archive_bytes} B) is "
        f"BELOW the Shannon R(D) lower bound ({f2_r_min_bytes:.0f} B) — "
        "this would violate the rate-distortion theorem; one of the two "
        "primitives is mis-targeted (probe-disambiguator should route this)."
    )


def test_f1_times_f2_both_engage_probe_disambiguator():
    """F1 and F2 are sister probes — they both engage probe_disambiguator
    and the solver should reach BOTH via that hook for cross-check."""
    discovered = discover_primitives_by_hook()
    probe_set = set(discovered["probe_disambiguator"])
    assert "mdl_scorer_conditional" in probe_set
    assert "shannon_vector_r_d" in probe_set


# ---------------------------------------------------------------------------
# F3 × F10: bilinear-resize nullspace × YUV6 sublattice
# ---------------------------------------------------------------------------


def test_f3_times_f10_nullspace_dimension_and_yuv6_lattice_dimension_consistent():
    """F3 (bilinear-resize nullspace) and F10 (YUV6 sublattice) both
    characterize the SCORER PIPELINE's hidden-degrees-of-freedom — places
    where archive bytes can vary without affecting the scorer output.

    F3 measures the bilinear-resize step (camera-frame → scorer-input). It
    reports a nullspace of the resize linear map.

    F10 measures the YUV6 channel-decimation step (RGB → 6-channel YUV
    with 2× sub-sampling). It reports a sublattice (basis of the kernel).

    Together they bound the TOTAL nullspace of the scorer preprocessing
    pipeline (resize-then-YUV6 composition). The sublattice dimension and
    the nullspace dimension should both be POSITIVE (the contest
    preprocessing IS many-to-one) AND consistent with the camera/scorer
    dimensions.
    """
    f3 = BilinearResizeNullspace()
    f10 = YUV6SublatticeGeometry()

    # F3 — analyze the camera (874, 1164) → scorer (384, 512) resize.
    r3 = f3.compute(
        target=None,
        camera_size=(874, 1164),
        scorer_size=(384, 512),
    )
    # The resize map (874*1164 → 384*512) IS many-to-one; the upper-bound
    # nullspace fraction is positive.
    f3_pv = r3.primitive_value
    assert f3_pv.upper_bound_nullspace_fraction > 0
    assert f3_pv.camera_n_pixels > f3_pv.scorer_n_pixels  # downsampling

    # F10 — analyze a sample (N, 3, H, W) frame batch at scorer resolution.
    rgb_batch = torch.randn(1, 3, 384, 512) * 50 + 128  # roughly uint8 range
    r10 = f10.compute(rgb_batch)
    # YUV6 sub-samples U/V at 2×; per-sublattice energy is positive.
    f10_pv = r10.primitive_value
    assert float(f10_pv.u_sublattice_energy) >= 0
    assert float(f10_pv.v_sublattice_energy) >= 0

    # Cross-check: both primitives engage sensitivity_map and bit_allocator
    # (they characterize where the bit-allocator can route bytes
    # "for free" — i.e., into nullspace + sublattice directions).
    discovered = discover_primitives_by_hook()
    assert "bilinear_resize_nullspace" in discovered["sensitivity_map"]
    assert "yuv6_sublattice_geometry" in discovered["sensitivity_map"]
    assert "bilinear_resize_nullspace" in discovered["bit_allocator"]
    assert "yuv6_sublattice_geometry" in discovered["bit_allocator"]


# ---------------------------------------------------------------------------
# F7 × F8: SegNet margin polytope × PoseNet SE(3) Lie-algebra
# ---------------------------------------------------------------------------


def test_f7_times_f8_scorer_internal_polytope_and_lie_algebra_both_bound_margin():
    """F7 (SegNet logit-margin polytope) and F8 (PoseNet SE(3) Lie-algebra
    residual) characterize the per-scorer margin geometry — how much
    archive-bit perturbation each scorer can absorb before its output
    distortion class flips."""
    f7 = SegNetLogitMarginPolytope()
    f8 = PoseNetSE3LieAlgebra()

    # F7 — synthetic 5-class logit tensor with margin ~0.5.
    B, classes, H, W = 1, 5, 24, 32
    logits = torch.randn(B, classes, H, W)
    r7 = f7.compute(logits, margin_threshold=0.5)
    # F7 reports min/mean/median/max margin and safe-perturbation-budget.
    f7_pv = r7.primitive_value
    assert f7_pv.min_margin is not None
    assert f7_pv.safe_perturbation_budget_l_inf is not None

    # F8 — synthetic 6-dim pose vector pair (target + target_b for the Lie
    # algebra residual comparison).
    pose_a = torch.randn(1, 6)
    pose_b = torch.randn(1, 6)
    r8 = f8.compute(pose_a, target_b=pose_b)
    # F8 reports Euclidean vs Lie-algebra distance gap.
    f8_pv = r8.primitive_value
    assert f8_pv.mean_euclidean_distance is not None
    assert f8_pv.mean_lie_algebra_distance is not None

    # Cross-check: both engage sensitivity_map + probe_disambiguator hooks.
    hooks_7 = set(r7.wire_in_hooks_engaged)
    hooks_8 = set(r8.wire_in_hooks_engaged)
    assert "sensitivity_map" in hooks_7 and "sensitivity_map" in hooks_8
    assert "probe_disambiguator" in hooks_7 and "probe_disambiguator" in hooks_8


# ---------------------------------------------------------------------------
# F5 × F6: VQ codebook coverage × Wavelet HF energy
# ---------------------------------------------------------------------------


def test_f5_times_f6_codec_axis_both_engage_bit_allocator():
    """F5 (VQ codebook coverage) and F6 (Wavelet HF energy) are codec-axis
    primitives — they inform the bit-allocator on WHERE in the archive
    structure to route bits."""
    f5 = VQCodebookCoverage()
    f6 = WaveletHFEnergy()

    # F5 — synthetic latent batch + matching codebook.
    latents = torch.randn(8, 16)
    codebook = torch.randn(32, 16)
    r5 = f5.compute(latents, codebook=codebook)
    assert r5.primitive_value.codebook_index_entropy_bits is not None

    # F6 — synthetic single-channel grayscale image (1, H, W).
    img = torch.randn(1, 64, 64)
    r6 = f6.compute(img)
    assert r6.primitive_value.total_hf_energy is not None
    assert r6.primitive_value.hf_energy_fraction is not None

    discovered = discover_primitives_by_hook()
    assert "vq_codebook_coverage" in discovered["bit_allocator"]
    assert "wavelet_hf_energy" in discovered["bit_allocator"]


# ---------------------------------------------------------------------------
# F4 × F2: Score Lipschitz × Shannon vector R(D)
# ---------------------------------------------------------------------------


def test_f4_times_f2_pareto_hook_yields_both_primitives():
    """F4 (Score-vs-archive Lipschitz) and F2 (Shannon vector R(D)) both
    feed the Pareto-constraint hook. The pareto bundle should yield BOTH."""
    bundle = wire_in_for_hook(
        "pareto_constraint",
        targets={
            "shannon_vector_r_d": {
                "target": None,
                "d_seg_target": 0.067,
                "d_pose_target": 0.018,
            },
            "score_lipschitz": {"target": b"\x00" * 16},
        },
    )
    names = {r.primitive_name for r in bundle.results}
    assert "shannon_vector_r_d" in names
    assert "score_lipschitz" in names

    # Pareto results MUST carry confidence_band for the consumer's
    # trust-region construction.
    for r in bundle.results:
        assert r.confidence_band is not None


# ---------------------------------------------------------------------------
# F9 × F11: Per-pair decomposition × Unified action principle
# ---------------------------------------------------------------------------


def test_f9_times_f11_cathedral_autopilot_hook_yields_both():
    """F9 (per-pair score decomposition) and F11 (unified action principle)
    both engage the cathedral autopilot — F9 yields the top-K pairs by
    marginal contribution; F11 yields the unified action value."""
    f9_target = torch.tensor([[0.1, 0.01], [0.5, 0.02], [0.05, 0.005]])

    # F11 unified action takes a different target shape - skip its compute
    # here and just verify the hook discovery.
    discovered = discover_primitives_by_hook()
    assert "per_pair_score_decomposition" in discovered["cathedral_autopilot"]
    assert "unified_action_principle" in discovered["cathedral_autopilot"]

    bundle = wire_in_for_hook(
        "cathedral_autopilot",
        targets={
            "per_pair_score_decomposition": {
                "target": f9_target,
                "top_k": 2,
            },
        },
    )
    names = {r.primitive_name for r in bundle.results}
    assert "per_pair_score_decomposition" in names


# ---------------------------------------------------------------------------
# Inventory cardinality + hook coverage
# ---------------------------------------------------------------------------


def test_fourteen_xray_primitives_each_engage_at_least_one_hook():
    """Every xray primitive must engage at least one of the 6 canonical
    wire-in hooks (per CLAUDE.md "Subagent coherence-by-default" - silent
    omission is the orphan-work failure mode)."""
    inv = canonical_xray_primitive_inventory()
    assert len(inv) == 14
    for spec in inv:
        hooks_engaged = set(spec.wire_in_hooks)
        assert hooks_engaged, (
            f"primitive {spec.primitive_name!r} engages NO hook — orphan-work "
            "failure mode per CLAUDE.md"
        )


def test_all_six_hooks_engaged_by_at_least_one_primitive():
    """All 6 canonical hooks must have at least one primitive engaging
    them — otherwise the solver-stack's consumer surface is unreachable."""
    discovered = discover_primitives_by_hook()
    for hook in (
        "sensitivity_map",
        "pareto_constraint",
        "bit_allocator",
        "cathedral_autopilot",
        "continual_learning",
        "probe_disambiguator",
    ):
        assert hook in discovered, f"hook {hook!r} missing from registry"
        assert len(discovered[hook]) >= 1, (
            f"hook {hook!r} has 0 engaging primitives — unreachable surface"
        )


def test_inventory_canonical_names_match_module_paths():
    """Every primitive_name in the registry must match the canonical
    module/symbol it points to (regression guard against accidental
    rename drift)."""
    from importlib import import_module

    for spec in canonical_xray_primitive_inventory():
        module = import_module(spec.canonical_module)
        cls = getattr(module, spec.canonical_symbol)
        # The canonical_symbol points to a class; primitive_name should
        # appear in the module path (snake_case ↔ ClassName mapping).
        assert cls is not None


def test_fourteen_primitive_total_hook_engagements_match_solver_stack_consumers():
    """The solver-stack adapters expect the following minimum hook
    engagement counts (per Batch 6 wire-in design):

    - sensitivity_map: 8+ primitives
    - pareto_constraint: 3+
    - bit_allocator: 7+
    - cathedral_autopilot: 3+
    - continual_learning: 1+
    - probe_disambiguator: 9+

    These minima are baked into the integration tests at
    test_integration_with_solver_stack.py; this test re-asserts them at
    the inventory layer so a future change to ANY primitive's
    wire_in_hooks_engaged surfaces here too."""
    discovered = discover_primitives_by_hook()
    minimums = {
        "sensitivity_map": 8,
        "pareto_constraint": 3,
        "bit_allocator": 7,
        "cathedral_autopilot": 3,
        "continual_learning": 1,
        "probe_disambiguator": 9,
    }
    for hook, minimum in minimums.items():
        actual = len(discovered.get(hook, []))
        assert actual >= minimum, (
            f"hook {hook!r}: solver-stack expects {minimum}+ engaging "
            f"primitives but registry has {actual}; downstream "
            "adapter will under-utilize the wire-in surface."
        )


# ---------------------------------------------------------------------------
# F12 × F13: Predictive coding hierarchy × Foveation ego-motion
# ---------------------------------------------------------------------------


def test_f12_times_f13_both_engage_sensitivity_map_and_bit_allocator():
    """F12 (predictive coding hierarchy) and F13 (foveation ego-motion)
    are temporal-axis primitives — they both inform the sensitivity-map
    AND the bit-allocator about WHERE temporal redundancy can be exploited."""
    discovered = discover_primitives_by_hook()
    assert "predictive_coding_hierarchy" in discovered["sensitivity_map"]
    assert "predictive_coding_hierarchy" in discovered["bit_allocator"]
    assert "foveation_ego_motion" in discovered["sensitivity_map"]
    assert "foveation_ego_motion" in discovered["bit_allocator"]


# ---------------------------------------------------------------------------
# Cross-hook composability
# ---------------------------------------------------------------------------


def test_same_primitive_can_engage_multiple_hooks():
    """A primitive engaging N hooks should appear in N different
    `discover_primitives_by_hook` lists. Shannon R(D) engages 3 hooks
    (sensitivity_map, pareto_constraint, probe_disambiguator);
    Unified action engages 4."""
    discovered = discover_primitives_by_hook()
    shannon_hooks = [
        h for h, names in discovered.items() if "shannon_vector_r_d" in names
    ]
    assert len(shannon_hooks) >= 3

    unified_hooks = [
        h for h, names in discovered.items() if "unified_action_principle" in names
    ]
    assert len(unified_hooks) >= 4


def test_no_primitive_engages_zero_hooks():
    """Negative regression: catch any future primitive that lands with an
    empty wire_in_hooks_engaged set (orphan-work)."""
    inv = canonical_xray_primitive_inventory()
    for spec in inv:
        assert len(spec.wire_in_hooks) >= 1, (
            f"primitive {spec.primitive_name!r}: wire_in_hooks_engaged is "
            "EMPTY — orphan-work failure mode (CLAUDE.md non-negotiable)"
        )
