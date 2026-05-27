# SPDX-License-Identifier: MIT
"""Tests for the per-pair MLX heuristic-prior 5D canvas populator path.

Per the PARADOX-CLOSER Half-2 re-point (ANALYSIS sister hand-off from
`.omx/research/mlx_per_pair_master_gradient_authoritative_artifacts_landed_20260527.md`):
`populate_per_pair_cells_from_gradient_array` consumes the
`(N_bytes, N_pairs, 3)` MLX per-pair master-gradient HEURISTIC PRIOR and
builds ONE coordinate per distinct `pair_idx` — the >=2-distinct-coordinate
structure the 12-operator multi-op composition sweep needs.

NON-PROMOTABLE macOS-MLX research-signal per CLAUDE.md "MLX portable-local-
substrate authority" + Catalog #192/#127/#323. Every cell carries canonical
macOS-CPU advisory Provenance with promotion_eligible=false /
score_claim_valid=false / heuristic_prior=true.

Test coverage:

1. Shape contract (wrong ndim / wrong axis-2 / empty axes fail-closed).
2. Per-pair cell mapping (N_pairs * 3 cells; distinct pair_idx; per-axis
   verbatim aggregate; rate column preserved verbatim).
3. Provenance contract (canonical macos_cpu_advisory grade; non-promotable
   markers; heuristic_prior diagnostic + violation reason).
4. Sidecar write + read roundtrip via canonical writer/reader.
5. max_pairs cap.
6. Schema-mismatch sidecar fail-closed.
7. Operator productivity: per-pair canvas produces multi-op candidates
   (the >=2-distinct-coordinate Half-2 unblock).
8. Missing artifact fail-closed; missing/absent sha fail-closed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (  # noqa: E402
    CpuCudaAxis,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (  # noqa: E402
    PopulatedCanvasManifest,
    PopulatorError,
    load_empirical_lattice,
    populate_per_pair_cells_from_gradient_array,
)

# A valid 64-char lowercase hex sha for the synthetic fixtures (the fec6
# frontier sha; canonical Provenance requires 64-char hex source_sha256).
_FRONTIER_SHA = "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
_MLX_SCHEMA = "mlx_tensor_fd_gradient_heuristic_v1_20260527"


def _write_synthetic_artifact(
    tmp_path: Path,
    arr: np.ndarray,
    *,
    sha: str = _FRONTIER_SHA,
    schema: str | None = _MLX_SCHEMA,
    write_meta: bool = True,
) -> Path:
    """Write a synthetic .npy + sidecar meta mirroring the MLX producer."""
    npy = tmp_path / "synthetic_per_pair.npy"
    np.save(npy, arr)
    if write_meta:
        meta = {
            "archive_sha256": sha,
            "npy_path": str(npy),
            "npy_sha256": "deadbeef" * 8,
            "captured_at_utc": "2026-05-27T14:00:00.000000+00:00",
            "n_pairs_total": 600,
            "n_pairs_used": arr.shape[1],
            "gradient_tensor_kind": "tensor_fd_uniform_decompressed_projection_heuristic_v1",
            "gradient_byte_domain": "decompressed_decoder_mantissa_span_uniform_attribution",
            "master_gradient_anchor_written": False,
            "master_gradient_anchor_blockers": [
                "source_runtime_full_frame_parity_missing",
            ],
            "hardware_substrate": "darwin_arm64_m5_max_macos_mlx_advisory",
            "measurement_method": "per_tensor_central_fd_via_mlx_scorer_oracle",
        }
        if schema is not None:
            meta["schema_version"] = schema
        (npy.with_suffix(npy.suffix + ".meta.json")).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return npy


def _signed_per_pair_array(n_pairs: int = 8, n_bytes: int = 100) -> np.ndarray:
    """Synthetic (N_bytes, N_pairs, 3) with mixed-sign per-pair aggregates.

    Even pairs aggregate NEGATIVE (score improvement); odd pairs POSITIVE.
    Rate column all-zero (matches the real MLX artifact).
    """
    rng = np.random.default_rng(20260527)
    arr = rng.normal(0.0, 1e-5, size=(n_bytes, n_pairs, 3))
    arr[:, :, 2] = 0.0  # rate column zero
    # Bias even pairs negative, odd pairs positive so >=1 pair composes.
    for p in range(n_pairs):
        bias = -1e-4 if p % 2 == 0 else 1e-4
        arr[:, p, 0] += bias / n_bytes
        arr[:, p, 1] += bias / n_bytes
    return arr


# ---------------------------------------------------------------------------
# 1. Shape contract.
# ---------------------------------------------------------------------------


def test_wrong_ndim_fails_closed(tmp_path):
    arr = np.zeros((100, 8), dtype=np.float64)  # 2D, not 3D
    npy = _write_synthetic_artifact(tmp_path, arr)
    with pytest.raises(PopulatorError, match=r"shape \(N_bytes, N_pairs, 3\)"):
        populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)


def test_wrong_axis2_fails_closed(tmp_path):
    arr = np.zeros((100, 8, 4), dtype=np.float64)  # 4 axes, not 3
    npy = _write_synthetic_artifact(tmp_path, arr)
    with pytest.raises(PopulatorError, match=r"shape \(N_bytes, N_pairs, 3\)"):
        populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)


def test_empty_pairs_fails_closed(tmp_path):
    arr = np.zeros((100, 0, 3), dtype=np.float64)
    npy = _write_synthetic_artifact(tmp_path, arr)
    with pytest.raises(PopulatorError, match="empty axes"):
        populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)


# ---------------------------------------------------------------------------
# 2. Per-pair cell mapping.
# ---------------------------------------------------------------------------


def test_per_pair_cell_count_and_distinct_coords(tmp_path):
    arr = _signed_per_pair_array(n_pairs=8, n_bytes=50)
    npy = _write_synthetic_artifact(tmp_path, arr)
    m = populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)
    assert isinstance(m, PopulatedCanvasManifest)
    # 8 pairs * 3 scorer axes = 24 cells.
    assert m.cells_populated == 24
    assert m.anchors_consumed == 8
    pairs = sorted({c.pair_idx for c in m.canvas._cells.values()})
    assert pairs == list(range(8))  # 8 distinct pair coordinates (>=2 structure)


def test_per_pair_axis_values_verbatim(tmp_path):
    arr = _signed_per_pair_array(n_pairs=4, n_bytes=30)
    expected = arr.sum(axis=0)  # (4, 3) seg,pose,rate
    npy = _write_synthetic_artifact(tmp_path, arr)
    m = populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)
    for p in range(4):
        seg_cell = m.canvas.query_cell(
            p, min(p * 2, 1199), ScorerAxis.SEGNET_5CLASS,
            ReceiverRuntime.RAW_RESIDUAL, CpuCudaAxis.CONTEST_CPU,
        )
        pose_cell = m.canvas.query_cell(
            p, min(p * 2, 1199), ScorerAxis.POSENET_6D,
            ReceiverRuntime.RAW_RESIDUAL, CpuCudaAxis.CONTEST_CPU,
        )
        assert seg_cell is not None and pose_cell is not None
        assert seg_cell.predicted_delta_score == pytest.approx(expected[p, 0])
        assert pose_cell.predicted_delta_score == pytest.approx(expected[p, 1])


def test_rate_column_preserved_verbatim(tmp_path):
    arr = _signed_per_pair_array(n_pairs=4, n_bytes=30)
    arr[:, :, 2] = 0.0  # explicit zero rate
    npy = _write_synthetic_artifact(tmp_path, arr)
    m = populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)
    for p in range(4):
        rate_cell = m.canvas.query_cell(
            p, min(p * 2, 1199), ScorerAxis.RATE_TERM,
            ReceiverRuntime.RAW_RESIDUAL, CpuCudaAxis.CONTEST_CPU,
        )
        assert rate_cell is not None
        assert rate_cell.predicted_delta_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Provenance contract (NON-PROMOTABLE heuristic prior).
# ---------------------------------------------------------------------------


def test_provenance_non_promotable_macos_advisory(tmp_path):
    arr = _signed_per_pair_array(n_pairs=4)
    npy = _write_synthetic_artifact(tmp_path, arr)
    m = populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)
    prov = m.catalog_323_provenance
    assert prov["evidence_grade"] == "macos_cpu_advisory"
    assert prov["promotion_eligible"] is False
    assert prov["score_claim_valid"] is False
    assert prov["artifact_kind"] == "advisory_non_promotable"
    assert prov["heuristic_prior"] is True
    assert prov["is_authoritative_contest_axis"] is False
    assert "not_authoritative" in prov["contest_axis_authority_violation_reason"]
    # Per-cell Provenance carries the same non-promotable markers.
    for cell in m.canvas._cells.values():
        cp = cell.catalog_323_provenance
        assert cp["promotion_eligible"] is False
        assert cp["heuristic_prior"] is True


# ---------------------------------------------------------------------------
# 4. Sidecar write + read roundtrip.
# ---------------------------------------------------------------------------


def test_sidecar_write_and_read_roundtrip(tmp_path):
    arr = _signed_per_pair_array(n_pairs=4)
    npy = _write_synthetic_artifact(tmp_path, arr)
    out = tmp_path / "perpair_sidecar.json"
    m = populate_per_pair_cells_from_gradient_array(
        npy, write_sidecar=True, output_path=out
    )
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["source_kind"] == "mlx_per_pair_heuristic_prior"
    assert payload["promotable"] is False
    assert payload["axis_tag"] == "[predicted]"
    assert payload["pairs_populated"] == 4
    # Read back via canonical reader (writes to canonical sidecar dir form).
    out2 = tmp_path / f"{_FRONTIER_SHA[:12]}_perpair_RT.json"
    populate_per_pair_cells_from_gradient_array(
        npy, write_sidecar=True, output_path=out2
    )
    loaded = load_empirical_lattice(
        _FRONTIER_SHA, sidecar_dir=tmp_path
    )
    assert loaded.cell_count() == m.cells_populated


# ---------------------------------------------------------------------------
# 5. max_pairs cap.
# ---------------------------------------------------------------------------


def test_max_pairs_cap(tmp_path):
    arr = _signed_per_pair_array(n_pairs=8)
    npy = _write_synthetic_artifact(tmp_path, arr)
    m = populate_per_pair_cells_from_gradient_array(
        npy, write_sidecar=False, max_pairs=3
    )
    assert m.anchors_consumed == 3
    assert m.cells_populated == 9  # 3 pairs * 3 axes
    pairs = sorted({c.pair_idx for c in m.canvas._cells.values()})
    assert pairs == [0, 1, 2]


# ---------------------------------------------------------------------------
# 6. Schema-mismatch sidecar fail-closed.
# ---------------------------------------------------------------------------


def test_schema_mismatch_fails_closed(tmp_path):
    arr = _signed_per_pair_array(n_pairs=4)
    npy = _write_synthetic_artifact(
        tmp_path, arr, schema="some_other_schema_v9"
    )
    with pytest.raises(PopulatorError, match="schema mismatch"):
        populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)


def test_corrupt_sidecar_fails_closed(tmp_path):
    arr = _signed_per_pair_array(n_pairs=4)
    npy = _write_synthetic_artifact(tmp_path, arr, write_meta=False)
    (npy.with_suffix(npy.suffix + ".meta.json")).write_text(
        "{not valid json", encoding="utf-8"
    )
    with pytest.raises(PopulatorError, match="corrupt JSON"):
        populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)


# ---------------------------------------------------------------------------
# 7. Operator productivity (the Half-2 >=2-distinct-coordinate unblock).
# ---------------------------------------------------------------------------


def test_per_pair_canvas_produces_multiop_candidates(tmp_path):
    arr = _signed_per_pair_array(n_pairs=8, n_bytes=50)
    npy = _write_synthetic_artifact(tmp_path, arr)
    m = populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)
    # FULL_DROP on RAW_RESIDUAL: per-pair groups with total_delta<0 emit.
    fd = m.canvas.generate_full_drop_starts(
        top_n=32, receiver_runtime=ReceiverRuntime.RAW_RESIDUAL
    )
    # The 4 even pairs aggregate negative => >=1 candidate (the resolution
    # unblock vs the 0-candidate archive-aggregate canvas).
    assert len(fd) >= 1
    # Every candidate carries non-promotable canonical-routing markers.
    for cand in fd:
        assert cand.canonical_routing_markers["promotable"] is False
        assert cand.canonical_routing_markers["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# 8. Missing artifact / absent sha fail-closed.
# ---------------------------------------------------------------------------


def test_missing_artifact_fails_closed(tmp_path):
    with pytest.raises(PopulatorError, match="not found"):
        populate_per_pair_cells_from_gradient_array(
            tmp_path / "nonexistent.npy", write_sidecar=False
        )


def test_absent_sha_fails_closed(tmp_path):
    arr = _signed_per_pair_array(n_pairs=4)
    # Write artifact with NO meta + NO explicit sha => cannot key.
    npy = _write_synthetic_artifact(tmp_path, arr, write_meta=False)
    with pytest.raises(PopulatorError, match="archive_sha256 not provided"):
        populate_per_pair_cells_from_gradient_array(npy, write_sidecar=False)


def test_explicit_sha_override(tmp_path):
    arr = _signed_per_pair_array(n_pairs=4)
    npy = _write_synthetic_artifact(tmp_path, arr, write_meta=False)
    m = populate_per_pair_cells_from_gradient_array(
        npy, write_sidecar=False, archive_sha256=_FRONTIER_SHA
    )
    assert m.archive_sha256 == _FRONTIER_SHA


def test_live_mlx_artifact_if_present():
    """Live-repo regression: the real 64-pair MLX artifact populates 192 cells."""
    art = (
        _REPO_ROOT
        / ".omx/state/master_gradient_fec6_frontier_mlx_per_pair_64pair_20260527.npy"
    )
    if not art.exists():
        pytest.skip("live MLX per-pair artifact not present")
    m = populate_per_pair_cells_from_gradient_array(art, write_sidecar=False)
    assert m.cells_populated == 192  # 64 pairs * 3 axes
    assert m.anchors_consumed == 64
    assert m.catalog_323_provenance["heuristic_prior"] is True
    assert m.catalog_323_provenance["promotion_eligible"] is False
