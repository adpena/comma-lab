# SPDX-License-Identifier: MIT
"""Regression tests for the FULL-RANK multi-op problem-spec fix.

Closes the rank-1 operator-gradient tautology exposed by the rigor review
``.omx/research/master_gradient_analysis_rigor_signal_review_20260527T151020Z.md``
(commit ``21014faa7``): the original ``_build_multiop_problem_spec`` built every
operator gradient as ``(seg_aggregate, pose_aggregate, rate_aggregate) ×
scalar_leverage`` — a RANK-1 basis (all operators scalar multiples of one
vector) whose feasible polytope has ZERO synergy-axis volume for ANY input, so
the Dykstra "synergy ≈ 0" was a TAUTOLOGY, not a measurement.

The FIX derives each operator's per-axis gradient direction from its OWN
per-pair footprint via its best candidate's Catalog #356 ``AxisDecomposition``.
Different operators touch different pairs/bytes/regions, so their aggregate axis
DIRECTIONS are genuinely distinct → full-rank (rank > 1) operator basis → the
synergy term becomes a genuine MEASUREMENT.

The CRITICAL assertion (so the rank-1 tautology cannot silently return): when
operators have distinct footprints, the FULL-RANK spec's operator-gradient
matrix rank MUST be > 1.

All $0 — MLX-local / numpy / CPU only. NON-PROMOTABLE per Catalog #192/#127/#323.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.canvas_multiop_composition_closed_form_prediction_sweep import (
    OperatorCandidateSummary,
    _build_multiop_problem_spec,
    _operator_axis_gradient_from_decomposition,
    operator_gradient_matrix_rank,
)


def _axis_dec(seg: float, pose: float, rate_bytes: int) -> dict:
    """Build a minimal Catalog #356 AxisDecomposition dict footprint."""
    return {
        "predicted_d_seg_delta": seg,
        "predicted_d_pose_delta": pose,
        "predicted_archive_bytes_delta": rate_bytes,
        "axis_tag": "[predicted]",
        "canonical_provenance": {"evidence_grade": "predicted"},
    }


class _FakeCanvas:
    """Minimal canvas exposing the ``_cells`` sparse store the spec builder reads
    for the shared-aggregate fallback path."""

    def __init__(self, cells: dict | None = None) -> None:
        self._cells = cells or {}


def _summary(op: str, seg: float, pose: float, rate_bytes: int, *, delta: float = -0.01) -> OperatorCandidateSummary:
    return OperatorCandidateSummary(
        operation=op,
        n_candidates=8,
        best_predicted_delta_score=delta,
        best_predicted_byte_cost=rate_bytes,
        best_axis_decomposition=_axis_dec(seg, pose, rate_bytes),
    )


# ---------------------------------------------------------------------------
# CORE REGRESSION: full-rank spec produces rank > 1 with distinct footprints.
# ---------------------------------------------------------------------------


def test_full_rank_spec_rank_exceeds_one_with_distinct_footprints():
    """THE canonical anti-tautology assertion.

    Operators with genuinely distinct per-axis footprints (different
    seg/pose ratios + a rate-axis-touching operator) MUST yield a
    full-rank (rank > 1) operator-gradient basis.
    """
    summaries = [
        # seg-heavy, no rate
        _summary("replace_many", seg=1.66e-05, pose=4.36e-03, rate_bytes=0),
        # pose-only-ish, no rate (different ratio)
        _summary("merge_pair", seg=2.46e-06, pose=1.63e-03, rate_bytes=0),
        # rate-axis-touching (breaks rank-1 collapse)
        _summary("reorder_pair", seg=1.52e-05, pose=2.36e-03, rate_bytes=-8),
        _summary("temporal_coherence", seg=4.91e-06, pose=3.25e-03, rate_bytes=0),
    ]
    canvas = _FakeCanvas()
    spec = _build_multiop_problem_spec(canvas, summaries, "deadbeefcafe1234")
    rank = operator_gradient_matrix_rank(spec)
    assert rank > 1, (
        f"FULL-RANK spec must produce operator-gradient rank > 1 with distinct "
        f"footprints; got rank {rank}. A rank-1 basis is the synergy-tautology "
        f"the rigor review exposed."
    )
    # With a rate-axis-touching operator + distinct seg/pose ratios, the basis
    # spans all 3 axes.
    assert rank == 3


def test_legacy_rank1_spec_is_rank_one_tautology():
    """The legacy path reproduces the rank-1 tautology (so the before/after
    comparison is faithful)."""
    # Build a canvas whose _cells store yields nonzero shared aggregates so the
    # legacy (seg,pose,rate)×scalar path has a nonzero shared direction.
    from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
        ScorerAxis,
    )

    class _Cell:
        def __init__(self, axis, val):
            self.scorer_axis = axis
            self.predicted_delta_score = val

    cells = {
        "seg": _Cell(ScorerAxis.SEGNET_5CLASS, -1e-3),
        "pose": _Cell(ScorerAxis.POSENET_6D, -2e-3),
        "rate": _Cell(ScorerAxis.RATE_TERM, 0.0),
    }
    canvas = _FakeCanvas(cells)
    summaries = [
        _summary("replace_many", seg=1.66e-05, pose=4.36e-03, rate_bytes=0, delta=-0.02),
        _summary("merge_pair", seg=2.46e-06, pose=1.63e-03, rate_bytes=0, delta=-0.005),
        _summary("reorder_pair", seg=1.52e-05, pose=2.36e-03, rate_bytes=-8, delta=-0.009),
    ]
    spec = _build_multiop_problem_spec(
        canvas, summaries, "deadbeefcafe1234", rank1_legacy=True
    )
    rank = operator_gradient_matrix_rank(spec)
    assert rank == 1, (
        f"legacy rank1 path must be rank 1 (the documented tautology); got {rank}"
    )


def test_full_rank_strictly_greater_than_legacy_on_same_summaries():
    """The fix's whole point: full-rank > legacy-rank on the SAME operators."""
    from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
        ScorerAxis,
    )

    class _Cell:
        def __init__(self, axis, val):
            self.scorer_axis = axis
            self.predicted_delta_score = val

    cells = {
        "seg": _Cell(ScorerAxis.SEGNET_5CLASS, -1e-3),
        "pose": _Cell(ScorerAxis.POSENET_6D, -2e-3),
        "rate": _Cell(ScorerAxis.RATE_TERM, 0.0),
    }
    canvas = _FakeCanvas(cells)
    summaries = [
        _summary("replace_many", seg=1.66e-05, pose=4.36e-03, rate_bytes=0, delta=-0.02),
        _summary("merge_pair", seg=2.46e-06, pose=1.63e-03, rate_bytes=0, delta=-0.005),
        _summary("reorder_pair", seg=1.52e-05, pose=2.36e-03, rate_bytes=-8, delta=-0.009),
    ]
    legacy_rank = operator_gradient_matrix_rank(
        _build_multiop_problem_spec(canvas, summaries, "x" * 16, rank1_legacy=True)
    )
    full_rank = operator_gradient_matrix_rank(
        _build_multiop_problem_spec(canvas, summaries, "x" * 16, rank1_legacy=False)
    )
    assert legacy_rank == 1
    assert full_rank > legacy_rank


# ---------------------------------------------------------------------------
# Helper unit tests.
# ---------------------------------------------------------------------------


def test_axis_gradient_helper_returns_magnitudes():
    s = _summary("replace_many", seg=-1.66e-05, pose=-4.36e-03, rate_bytes=-8)
    got = _operator_axis_gradient_from_decomposition(s)
    assert got is not None
    assert got == (abs(-1.66e-05), abs(-4.36e-03), abs(-8))


def test_axis_gradient_helper_none_when_no_decomposition():
    s = OperatorCandidateSummary(
        operation="x", n_candidates=8, best_predicted_delta_score=-0.01,
        best_predicted_byte_cost=0, best_axis_decomposition=None,
    )
    assert _operator_axis_gradient_from_decomposition(s) is None


def test_axis_gradient_helper_none_when_all_zero():
    s = _summary("x", seg=0.0, pose=0.0, rate_bytes=0)
    assert _operator_axis_gradient_from_decomposition(s) is None


def test_full_rank_falls_back_to_aggregate_when_decomposition_absent():
    """An operator without an axis decomposition must NOT crash the spec; it
    falls back to the shared-aggregate × leverage value."""
    from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
        ScorerAxis,
    )

    class _Cell:
        def __init__(self, axis, val):
            self.scorer_axis = axis
            self.predicted_delta_score = val

    cells = {
        "seg": _Cell(ScorerAxis.SEGNET_5CLASS, -1e-3),
        "pose": _Cell(ScorerAxis.POSENET_6D, -2e-3),
        "rate": _Cell(ScorerAxis.RATE_TERM, 0.0),
    }
    canvas = _FakeCanvas(cells)
    summaries = [
        _summary("replace_many", seg=1.66e-05, pose=4.36e-03, rate_bytes=-8),
        OperatorCandidateSummary(
            operation="no_dec", n_candidates=4, best_predicted_delta_score=-0.01,
            best_predicted_byte_cost=0, best_axis_decomposition=None,
        ),
    ]
    spec = _build_multiop_problem_spec(canvas, summaries, "x" * 16)
    # No crash; both operators get a gradient row.
    assert len(spec.per_axis_gradient_l2_norms) == 2
    # The fallback row uses the shared aggregate × leverage (nonzero seg/pose).
    fallback_row = spec.per_axis_gradient_l2_norms[1]
    assert fallback_row[0] > 0 or fallback_row[1] > 0


def test_empty_spec_rank_zero():
    spec = _build_multiop_problem_spec(
        _FakeCanvas(),
        [
            OperatorCandidateSummary(
                operation="x::ERROR", n_candidates=0, best_predicted_delta_score=0.0,
                best_predicted_byte_cost=0, best_axis_decomposition=None,
            )
        ],
        "x" * 16,
    )
    # productive falls back to summaries[:1]; the single ERROR op has no
    # decomposition and zero aggregate (empty canvas) → all-zero gradient.
    rank = operator_gradient_matrix_rank(spec)
    assert rank in (0, 1)


# ---------------------------------------------------------------------------
# Live-artifact regression guard (skipped if the full-600 artifact is absent).
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "artifact_name",
    [
        "master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy",
        "master_gradient_fec6_frontier_mlx_per_pair_64pair_20260527.npy",
    ],
)
def test_live_mlx_artifact_full_rank_exceeds_one(artifact_name: str):
    """On the live MLX per-pair artifacts, the full-rank spec MUST produce
    rank > 1 (genuine synergy-axis volume) while the legacy path produces
    rank 1 (the tautology). Skipped if the artifact is absent."""
    root = _repo_root()
    npy = root / ".omx" / "state" / artifact_name
    if not npy.exists():
        pytest.skip(f"artifact {artifact_name} not present")

    from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (
        populate_per_pair_cells_from_gradient_array,
    )
    from tools.canvas_multiop_composition_closed_form_prediction_sweep import (
        _run_all_12_operators,
    )

    manifest = populate_per_pair_cells_from_gradient_array(
        str(npy), repo_root=root, write_sidecar=False, max_pairs=None
    )
    canvas = manifest.canvas
    summaries = _run_all_12_operators(canvas, top_n=32)

    legacy = _build_multiop_problem_spec(
        canvas, summaries, manifest.archive_sha256, rank1_legacy=True
    )
    full = _build_multiop_problem_spec(
        canvas, summaries, manifest.archive_sha256, rank1_legacy=False
    )
    assert operator_gradient_matrix_rank(legacy) == 1, (
        "legacy path must reproduce the rank-1 tautology"
    )
    assert operator_gradient_matrix_rank(full) > 1, (
        "full-rank path must produce rank > 1 on the live MLX per-pair artifact"
    )
