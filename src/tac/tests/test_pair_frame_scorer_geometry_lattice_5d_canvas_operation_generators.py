# SPDX-License-Identifier: MIT
"""Tests for BUILD-2 + BUILD-3 5D canvas operation generators.

Per `.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md`
PRIORITY 2 + the BUILD-2 + BUILD-3 sister subagent landing
2026-05-26. Covers the 4 canonical operations (full-drop / repair /
masked / feathered) + Catalog #356 AxisDecomposition wire-in +
canonical Provenance per Catalog #323 + Tier A canonical-routing
markers per Catalog #341 + CLI surface.

Test scope:

1. Per-operation correctness (synthetic 5D canvas -> operation ->
   expected output)
2. Catalog #356 AxisDecomposition emission verification
3. Catalog #323 canonical Provenance threading
4. Catalog #341 Tier A canonical-routing markers
5. CLI exit codes (0 clean / 1 op-failed / 2 canvas-invalid / 3 cli)
6. Live-repo regression guard (module imports + class signatures)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CANVAS_SCHEMA,
    CanonicalOperation,
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
    generate_queue_executable_start,
    query_receiver_runtime_feasibility,
    sha256_hex,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (
    POPULATOR_SCHEMA_VERSION,
    load_empirical_lattice,
    populate_5d_canvas_from_master_gradient_anchors,
)
from tac.tests.test_pair_frame_scorer_geometry_lattice_5d_canvas_populator import (
    _synthetic_anchor,
    _write_synthetic_ledger,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CLI_PATH = _REPO_ROOT / "tools" / "apply_operation_to_5d_canvas_cli.py"


def _make_canvas_with_cells(
    archive_sha256: str = "a" * 64,
    cells: list[PairFrameScorerGeometryCell] | None = None,
) -> PairFrameScorerGeometryLattice:
    canvas = PairFrameScorerGeometryLattice(archive_sha256)
    for cell in cells or []:
        canvas._cells[cell.coordinate] = cell
    return canvas


def _make_cell(
    *,
    pair_idx: int = 100,
    frame_idx: int = 200,
    scorer_axis: ScorerAxis = ScorerAxis.SEGNET_5CLASS,
    receiver_runtime: ReceiverRuntime = ReceiverRuntime.FULL_DROP,
    cpu_cuda_axis: CpuCudaAxis = CpuCudaAxis.CONTEST_CPU,
    predicted_delta_score: float = -1e-4,
    predicted_byte_cost: int = -100,
    receiver_feasibility: bool = True,
) -> PairFrameScorerGeometryCell:
    return PairFrameScorerGeometryCell(
        pair_idx=pair_idx,
        frame_idx=frame_idx,
        scorer_axis=scorer_axis,
        receiver_runtime=receiver_runtime,
        cpu_cuda_axis=cpu_cuda_axis,
        predicted_delta_score=predicted_delta_score,
        predicted_byte_cost=predicted_byte_cost,
        receiver_feasibility=receiver_feasibility,
    )


# ---------------------------------------------------------------------------
# Tier 1: Per-operation correctness (4 operations × happy path)
# ---------------------------------------------------------------------------


class TestPerOperationHappyPath:
    def test_full_drop_emits_candidate_for_score_improving_pair(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(scorer_axis=ScorerAxis.SEGNET_5CLASS, predicted_delta_score=-2e-4, predicted_byte_cost=0),
                _make_cell(scorer_axis=ScorerAxis.POSENET_6D, predicted_delta_score=-1e-4, predicted_byte_cost=0),
                _make_cell(scorer_axis=ScorerAxis.RATE_TERM, predicted_delta_score=0.0, predicted_byte_cost=-200),
            ]
        )
        candidates = canvas.generate_full_drop_starts(top_n=5)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.operation is CanonicalOperation.FULL_DROP
        # Score delta = -2e-4 + -1e-4 + 0.0 = -3e-4
        assert abs(c.predicted_delta_score - (-3e-4)) < 1e-9
        # Byte cost = 0 + 0 + (-200) = -200 (sum across all cells per group)
        assert c.predicted_byte_cost == -200

    def test_repair_emits_candidate_with_smoothed_residual_default(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(
                    scorer_axis=ScorerAxis.SEGNET_5CLASS,
                    receiver_runtime=ReceiverRuntime.SMOOTHED_RESIDUAL,
                    predicted_delta_score=-5e-4,
                )
            ]
        )
        candidates = canvas.generate_repair_starts(top_n=5)
        assert len(candidates) == 1
        assert candidates[0].operation is CanonicalOperation.REPAIR

    def test_repair_refuses_raw_residual_receiver(self) -> None:
        canvas = _make_canvas_with_cells()
        with pytest.raises(ValueError, match="REPAIR refuses RAW_RESIDUAL"):
            canvas.generate_repair_starts(
                top_n=5, receiver_runtime=ReceiverRuntime.RAW_RESIDUAL
            )

    def test_masked_emits_candidate_with_masked_default(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(
                    scorer_axis=ScorerAxis.SEGNET_5CLASS,
                    receiver_runtime=ReceiverRuntime.MASKED,
                    predicted_delta_score=-3e-4,
                )
            ]
        )
        candidates = canvas.generate_masked_starts(top_n=5)
        assert len(candidates) == 1
        assert candidates[0].operation is CanonicalOperation.MASKED

    def test_feathered_emits_candidate_with_feathered_default(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(
                    scorer_axis=ScorerAxis.SEGNET_5CLASS,
                    receiver_runtime=ReceiverRuntime.FEATHERED,
                    predicted_delta_score=-4e-4,
                )
            ]
        )
        candidates = canvas.generate_feathered_starts(top_n=5)
        assert len(candidates) == 1
        assert candidates[0].operation is CanonicalOperation.FEATHERED


class TestOperationGeneratorFiltering:
    def test_skips_score_regression_cells(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[_make_cell(predicted_delta_score=+1e-4)]
        )
        # Positive delta = score regression; should be filtered out.
        assert canvas.generate_full_drop_starts(top_n=5) == []

    def test_skips_infeasible_cells(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(
                    receiver_feasibility=False, predicted_delta_score=-1e-3
                )
            ]
        )
        assert canvas.generate_full_drop_starts(top_n=5) == []

    def test_groups_by_pair_for_pair_level_operations(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(pair_idx=50, scorer_axis=ScorerAxis.SEGNET_5CLASS, predicted_delta_score=-1e-4),
                _make_cell(pair_idx=50, scorer_axis=ScorerAxis.POSENET_6D, predicted_delta_score=-1e-4),
                _make_cell(pair_idx=75, scorer_axis=ScorerAxis.SEGNET_5CLASS, predicted_delta_score=-2e-4),
            ]
        )
        candidates = canvas.generate_full_drop_starts(top_n=10)
        # 2 distinct pair_idx groups
        assert len(candidates) == 2
        # Sorted ascending by score: pair 75 first (-2e-4 < -2e-4 sum)
        assert candidates[0].predicted_delta_score == -2e-4

    def test_groups_by_frame_for_frame_level_operations(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(
                    pair_idx=100, frame_idx=200,
                    scorer_axis=ScorerAxis.SEGNET_5CLASS,
                    receiver_runtime=ReceiverRuntime.MASKED,
                    predicted_delta_score=-1e-4,
                ),
                _make_cell(
                    pair_idx=101, frame_idx=200,  # same frame, different pair
                    scorer_axis=ScorerAxis.POSENET_6D,
                    receiver_runtime=ReceiverRuntime.MASKED,
                    predicted_delta_score=-1e-4,
                ),
                _make_cell(
                    pair_idx=102, frame_idx=400,  # different frame
                    scorer_axis=ScorerAxis.SEGNET_5CLASS,
                    receiver_runtime=ReceiverRuntime.MASKED,
                    predicted_delta_score=-2e-4,
                ),
            ]
        )
        candidates = canvas.generate_masked_starts(top_n=10)
        # 2 distinct frame_idx groups: 200 (combined -2e-4) and 400 (-2e-4)
        assert len(candidates) == 2

    def test_top_n_limit_respected(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(pair_idx=p, predicted_delta_score=-1e-4)
                for p in range(50, 60)
            ]
        )
        candidates = canvas.generate_full_drop_starts(top_n=3)
        assert len(candidates) == 3


class TestOperationGeneratorInputValidation:
    def test_full_drop_rejects_zero_top_n(self) -> None:
        canvas = _make_canvas_with_cells()
        with pytest.raises(ValueError, match="top_n must be positive"):
            canvas.generate_full_drop_starts(top_n=0)

    def test_full_drop_rejects_non_int_top_n(self) -> None:
        canvas = _make_canvas_with_cells()
        with pytest.raises(ValueError, match="top_n must be positive"):
            canvas.generate_full_drop_starts(top_n="five")  # type: ignore[arg-type]

    def test_repair_rejects_non_enum_receiver(self) -> None:
        canvas = _make_canvas_with_cells()
        with pytest.raises(ValueError, match="receiver_runtime must be ReceiverRuntime"):
            canvas.generate_repair_starts(
                top_n=5, receiver_runtime="smoothed"  # type: ignore[arg-type]
            )

    def test_masked_rejects_negative_top_n(self) -> None:
        canvas = _make_canvas_with_cells()
        with pytest.raises(ValueError, match="top_n must be positive"):
            canvas.generate_masked_starts(top_n=-1)

    def test_feathered_rejects_negative_top_n(self) -> None:
        canvas = _make_canvas_with_cells()
        with pytest.raises(ValueError, match="top_n must be positive"):
            canvas.generate_feathered_starts(top_n=-5)


# ---------------------------------------------------------------------------
# Tier 2: Catalog #356 AxisDecomposition verification
# ---------------------------------------------------------------------------


class TestCatalog356AxisDecompositionWireIn:
    def test_every_candidate_carries_axis_decomposition(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(scorer_axis=ScorerAxis.SEGNET_5CLASS, predicted_delta_score=-1e-3),
                _make_cell(scorer_axis=ScorerAxis.POSENET_6D, predicted_delta_score=-2e-4),
                _make_cell(scorer_axis=ScorerAxis.RATE_TERM, predicted_delta_score=0.0, predicted_byte_cost=-100),
            ]
        )
        candidates = canvas.generate_full_drop_starts(top_n=5)
        assert len(candidates) == 1
        ad = candidates[0].predicted_axis_decomposition
        assert ad is not None
        assert isinstance(ad, AxisDecomposition)

    def test_axis_decomposition_inverts_canonical_formula(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(scorer_axis=ScorerAxis.SEGNET_5CLASS, predicted_delta_score=-0.001, predicted_byte_cost=0),
                _make_cell(scorer_axis=ScorerAxis.POSENET_6D, predicted_delta_score=-0.0001, predicted_byte_cost=0),
                _make_cell(scorer_axis=ScorerAxis.RATE_TERM, predicted_delta_score=0.0, predicted_byte_cost=-500),
            ]
        )
        candidates = canvas.generate_full_drop_starts(top_n=5)
        ad = candidates[0].predicted_axis_decomposition
        # SegNet: score_delta -0.001 / 100 = -1e-5
        assert abs(ad.predicted_d_seg_delta - (-1e-5)) < 1e-12
        # Bytes: 0 + 0 + (-500) = -500
        assert ad.predicted_archive_bytes_delta == -500

    def test_axis_tag_is_predicted_per_catalog_341(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[_make_cell(predicted_delta_score=-1e-4)]
        )
        candidates = canvas.generate_full_drop_starts(top_n=5)
        assert candidates[0].predicted_axis_decomposition.axis_tag == "[predicted]"


# ---------------------------------------------------------------------------
# Tier 3: Catalog #323 canonical Provenance threading
# ---------------------------------------------------------------------------


class TestCatalog323CanonicalProvenanceThreading:
    def test_candidate_carries_provenance_dict(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[_make_cell(predicted_delta_score=-1e-4)]
        )
        candidates = canvas.generate_full_drop_starts(top_n=5)
        prov = candidates[0].catalog_323_provenance
        assert isinstance(prov, dict)
        assert prov.get("artifact_kind") == "predicted_from_model"

    def test_axis_decomposition_provenance_matches_candidate(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[_make_cell(predicted_delta_score=-1e-4)]
        )
        candidates = canvas.generate_full_drop_starts(top_n=5)
        ad_prov = candidates[0].predicted_axis_decomposition.canonical_provenance
        c_prov = candidates[0].catalog_323_provenance
        assert dict(ad_prov) == dict(c_prov)

    def test_provenance_model_id_distinct_per_operation(self) -> None:
        # All 4 operations should produce distinct model_ids in source_path.
        cells_kwargs = [
            {"receiver_runtime": ReceiverRuntime.FULL_DROP},
            {"receiver_runtime": ReceiverRuntime.SMOOTHED_RESIDUAL},
            {"receiver_runtime": ReceiverRuntime.MASKED},
            {"receiver_runtime": ReceiverRuntime.FEATHERED},
        ]
        model_ids = set()
        for kw in cells_kwargs:
            canvas = _make_canvas_with_cells(
                cells=[_make_cell(predicted_delta_score=-1e-4, **kw)]
            )
            if kw["receiver_runtime"] == ReceiverRuntime.FULL_DROP:
                cands = canvas.generate_full_drop_starts(top_n=5)
            elif kw["receiver_runtime"] == ReceiverRuntime.SMOOTHED_RESIDUAL:
                cands = canvas.generate_repair_starts(top_n=5)
            elif kw["receiver_runtime"] == ReceiverRuntime.MASKED:
                cands = canvas.generate_masked_starts(top_n=5)
            else:
                cands = canvas.generate_feathered_starts(top_n=5)
            assert len(cands) == 1
            model_ids.add(cands[0].catalog_323_provenance.get("source_path"))
        # All 4 model_ids should be distinct
        assert len(model_ids) == 4


# ---------------------------------------------------------------------------
# Tier 4: Catalog #341 Tier A canonical-routing markers
# ---------------------------------------------------------------------------


class TestCatalog341TierACanonicalRoutingMarkers:
    def test_candidate_carries_tier_a_markers_default(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[_make_cell(predicted_delta_score=-1e-4)]
        )
        candidates = canvas.generate_full_drop_starts(top_n=5)
        markers = dict(candidates[0].canonical_routing_markers)
        assert markers.get("promotable") is False
        assert markers.get("axis_tag") == "[predicted]"
        assert markers.get("predicted_delta_adjustment") == 0.0


class TestModuleLevelQueueStart:
    def test_generate_queue_executable_start_emits_false_authority_candidate(self) -> None:
        canvas = _make_canvas_with_cells(
            cells=[
                _make_cell(
                    pair_idx=123,
                    frame_idx=246,
                    scorer_axis=ScorerAxis.SEGNET_5CLASS,
                    receiver_runtime=ReceiverRuntime.FULL_DROP,
                    cpu_cuda_axis=CpuCudaAxis.CONTEST_CPU,
                    predicted_delta_score=-2e-4,
                    predicted_byte_cost=-12,
                )
            ]
        )
        candidate = generate_queue_executable_start(
            CanonicalOperation.FULL_DROP,
            pair_idxs=[123],
            frame_idxs=[246],
            receiver_runtime=ReceiverRuntime.FULL_DROP,
            cpu_cuda_axis=CpuCudaAxis.CONTEST_CPU,
            canvas=canvas,
        )
        assert candidate.operation is CanonicalOperation.FULL_DROP
        assert candidate.predicted_delta_score == pytest.approx(-2e-4)
        assert candidate.predicted_byte_cost == -12
        assert candidate.canonical_routing_markers["promotable"] is False
        assert candidate.canonical_routing_markers["axis_tag"] == "[predicted]"
        assert candidate.predicted_axis_decomposition is not None

    def test_generate_queue_executable_start_refuses_missing_coordinate(self) -> None:
        canvas = _make_canvas_with_cells(cells=[])
        with pytest.raises(ValueError, match="no feasible cells"):
            generate_queue_executable_start(
                CanonicalOperation.FULL_DROP,
                pair_idxs=[123],
                frame_idxs=[246],
                receiver_runtime=ReceiverRuntime.FULL_DROP,
                cpu_cuda_axis=CpuCudaAxis.CONTEST_CPU,
                canvas=canvas,
            )


class TestBuild1EntryPointsNoLongerScaffoldOnly:
    def test_class_build_lattice_delegates_to_populator(
        self, tmp_path: pathlib.Path
    ) -> None:
        archive_bytes = b"synthetic archive bytes for 5d canvas"
        archive_path = tmp_path / "archive.zip"
        archive_path.write_bytes(archive_bytes)
        archive_sha = hashlib.sha256(archive_bytes).hexdigest()
        _write_synthetic_ledger(tmp_path, [_synthetic_anchor(archive_sha=archive_sha)])

        canvas = PairFrameScorerGeometryLattice.build_lattice(archive_path)

        assert canvas.archive_sha256 == archive_sha
        assert canvas.cell_count() == 3

    def test_class_load_empirical_lattice_delegates_to_reader(
        self, tmp_path: pathlib.Path
    ) -> None:
        archive_sha = "b" * 64
        _write_synthetic_ledger(tmp_path, [_synthetic_anchor(archive_sha=archive_sha)])
        populate_5d_canvas_from_master_gradient_anchors(
            archive_sha256=archive_sha,
            write_sidecar=True,
            repo_root=tmp_path,
        )

        loaded = PairFrameScorerGeometryLattice.load_empirical_lattice(
            archive_sha,
            repo_root=tmp_path,
        )
        direct = load_empirical_lattice(archive_sha, repo_root=tmp_path)

        assert loaded.cell_count() == direct.cell_count() == 3

    def test_receiver_runtime_feasibility_is_deterministic_false_authority(
        self, tmp_path: pathlib.Path
    ) -> None:
        archive_path = tmp_path / "archive.zip"
        archive_path.write_bytes(b"unused for heuristic feasibility")

        full_drop = query_receiver_runtime_feasibility(
            0,
            0,
            ReceiverRuntime.FULL_DROP,
            archive_path,
        )
        masked = query_receiver_runtime_feasibility(
            0,
            0,
            ReceiverRuntime.MASKED,
            archive_path,
        )

        assert full_drop[ScorerAxis.RATE_TERM] is True
        assert full_drop[ScorerAxis.SEGNET_5CLASS] is False
        assert masked[ScorerAxis.SEGNET_5CLASS] is True
        assert masked[ScorerAxis.POSENET_6D] is False


# ---------------------------------------------------------------------------
# Tier 5: CLI exit codes + behavior
# ---------------------------------------------------------------------------


def _write_canvas_json(
    path: pathlib.Path,
    cells: list[PairFrameScorerGeometryCell],
    archive_sha256: str = "c" * 64,
) -> pathlib.Path:
    payload = {
        "schema": CANVAS_SCHEMA,
        "archive_sha256": archive_sha256,
        "cells": [c.as_dict() for c in cells],
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))
    return path


@pytest.fixture
def synthetic_canvas_json(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_canvas_json(
        tmp_path / "canvas.json",
        cells=[
            _make_cell(scorer_axis=ScorerAxis.SEGNET_5CLASS, predicted_delta_score=-1e-4),
            _make_cell(scorer_axis=ScorerAxis.RATE_TERM, predicted_delta_score=0.0, predicted_byte_cost=-200),
        ],
    )


class TestCLIExitCodes:
    def test_cli_clean_exit_zero(
        self, synthetic_canvas_json: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        out_path = tmp_path / "candidates.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--canvas-input", str(synthetic_canvas_json),
                "--operation", "full-drop",
                "--output-archive", str(out_path),
                "--top-n", "5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr={result.stderr}"
        assert out_path.exists()
        payload = json.loads(out_path.read_text())
        assert payload["operation"] == "full_drop"
        assert payload["candidate_count"] == 1

    def test_cli_accepts_populator_manifest_schema(
        self, synthetic_canvas_json: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        payload = json.loads(synthetic_canvas_json.read_text())
        payload["schema"] = POPULATOR_SCHEMA_VERSION
        populated_path = tmp_path / "populated_canvas.json"
        populated_path.write_text(json.dumps(payload, sort_keys=True, indent=2))
        out_path = tmp_path / "candidates_from_populated.json"

        result = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--canvas-input",
                str(populated_path),
                "--operation",
                "full-drop",
                "--output-archive",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"stderr={result.stderr}"
        assert json.loads(out_path.read_text())["candidate_count"] == 1

    def test_cli_canvas_invalid_exit_two(self, tmp_path: pathlib.Path) -> None:
        bad_path = tmp_path / "missing.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--canvas-input", str(bad_path),
                "--operation", "full-drop",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "CANVAS-INVALID" in result.stderr

    def test_cli_canvas_malformed_exit_two(self, tmp_path: pathlib.Path) -> None:
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not json at all {")
        result = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--canvas-input", str(bad_path),
                "--operation", "full-drop",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_cli_operation_failed_exit_one(self, tmp_path: pathlib.Path) -> None:
        # Build canvas with RAW_RESIDUAL cells; REPAIR refuses RAW_RESIDUAL.
        canvas_path = _write_canvas_json(
            tmp_path / "canvas_raw.json",
            cells=[
                _make_cell(
                    receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
                    predicted_delta_score=-1e-4,
                )
            ],
        )
        result = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--canvas-input", str(canvas_path),
                "--operation", "repair",
                "--receiver-runtime", "raw_residual",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "OPERATION-FAILED" in result.stderr

    def test_cli_json_output_to_stdout(
        self, synthetic_canvas_json: pathlib.Path
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--canvas-input", str(synthetic_canvas_json),
                "--operation", "full-drop",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["operation"] == "full_drop"

    def test_cli_supports_all_four_operations(
        self, synthetic_canvas_json: pathlib.Path
    ) -> None:
        # Build canvas with cells in all 4 receiver modes
        cells = []
        for receiver, _op_name in [
            (ReceiverRuntime.FULL_DROP, "full-drop"),
            (ReceiverRuntime.SMOOTHED_RESIDUAL, "repair"),
            (ReceiverRuntime.MASKED, "masked"),
            (ReceiverRuntime.FEATHERED, "feathered"),
        ]:
            cells.append(_make_cell(
                receiver_runtime=receiver,
                predicted_delta_score=-2e-4,
                pair_idx=100 + len(cells),
            ))
        canvas_path = _write_canvas_json(
            synthetic_canvas_json.parent / "multi_canvas.json",
            cells=cells,
        )
        for op in ["full-drop", "repair", "masked", "feathered"]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(_CLI_PATH),
                    "--canvas-input", str(canvas_path),
                    "--operation", op,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"op={op} stderr={result.stderr}"


# ---------------------------------------------------------------------------
# Tier 6: Live-repo regression guards
# ---------------------------------------------------------------------------


class TestLiveRepoRegressionGuard:
    def test_canvas_module_imports_cleanly(self) -> None:
        # Sanity check: the module imports + provides the canonical surface
        from tac.optimization import pair_frame_scorer_geometry_lattice_5d_canvas as mod
        assert hasattr(mod, "PairFrameScorerGeometryLattice")
        assert hasattr(mod, "ExecutableCandidate")
        assert hasattr(mod, "sha256_hex")

    def test_all_four_methods_are_callable_not_notimplemented(self) -> None:
        # Regression guard: confirm BUILD-2 wired the 4 generators (no
        # NotImplementedError under happy-path inputs).
        canvas = _make_canvas_with_cells()
        # Empty canvas returns empty list, NOT raises.
        assert canvas.generate_full_drop_starts(top_n=1) == []
        assert canvas.generate_repair_starts(top_n=1) == []
        assert canvas.generate_masked_starts(top_n=1) == []
        assert canvas.generate_feathered_starts(top_n=1) == []

    def test_sha256_hex_deterministic(self) -> None:
        assert sha256_hex("abc") == sha256_hex("abc")
        assert len(sha256_hex("xyz")) == 64

    def test_cli_entry_point_exists(self) -> None:
        assert _CLI_PATH.exists()
        assert _CLI_PATH.is_file()
