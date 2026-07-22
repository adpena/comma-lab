# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from tac.analysis.hprc_synthesis_adjoint import bilinear_resize_forward
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.solve_diff_operator_mining import (
    AXIS,
    CLASS_ORDER,
    INNER_JACOBIAN_BLOCKER,
    POINTER,
    ProductionInputContext,
    SolveDiffCostateRowV1,
    SolveDiffMiningConfigV1,
    SolveDiffMiningError,
    _load_production_inputs,
    _load_start_receipt_rows,
    canonical_json_bytes,
    canonical_jsonl_line,
    cheapest_target_hyperplane,
    coded_byte_counts,
    compact_parabolic_shearlet_coefficients,
    compact_temporal_features,
    derived_tolerance_ladder,
    exact_resize_adjoint_pullback,
    flip_distance_histogram,
    iter_costate_rows,
    leave_one_window_out_transport,
    load_checked_uint8_chunk,
    partition_movable_innovations,
    range_kernel_energy_split,
    rank4_head_accounting,
    realize_solve_camera,
    run_mining_pass,
    uint8_reachability_accounting,
    write_once_stage_checkpoint,
    xi_autocorrelation_rows,
    xi_features_from_twists,
)


def _fixture_config(tmp_path: Path, **updates: object) -> SolveDiffMiningConfigV1:
    payload: dict[str, object] = {
        "schema": "SolveDiffMiningConfigV1",
        "run_id": "solve-diff-fixture",
        "input_mode": "synthetic_fixture",
        "seed": 1234,
        "pair_start": 0,
        "pair_count": 6,
        "chunk_size": 3,
        "solved_planes_receipt_path": "unused-solved.json",
        "solved_planes_receipt_sha256": "0" * 64,
        "predictor_archive_path": "unused-predictor.receipt-bytes",
        "predictor_archive_sha256": "1" * 64,
        "start_receipt_path": "unused-start.json",
        "start_receipt_sha256": "2" * 64,
        "gt_cache_path": "unused-gt.npz",
        "gt_cache_sha256": "3" * 64,
        "class_order": list(CLASS_ORDER),
        "margin_thresholds": [0.25, 1.0, 4.0],
        "tolerance_retained_energy": [1.0, 0.5, 0.0],
        "temporal_window": 2,
        "ridge": 1e-6,
        "support_threshold": 0.5,
        "candidate_operators": [
            "xi_transport",
            "rank4_head_chart",
            "compact_parabolic_shearlet",
            "irreducible_residual",
        ],
        "spatial_chart": "compact_parabolic_shearlet",
        "coder_policy": "min_bytes_tie_zlib",
        "storage_roots": [str(tmp_path / "missing-bulk-tier")],
        "required_free_bytes": 1,
        "hard_pair_panels": 3,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
    }
    payload.update(updates)
    return SolveDiffMiningConfigV1.model_validate(payload)


def test_strict_config_rejects_authority_shape_and_unknown_keys(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    base = config.model_dump(mode="python", by_alias=True)
    for update in (
        {"unknown": 1},
        {"class_order": [*CLASS_ORDER[:-1], "Unknown"]},
        {"execution_allowed": True},
        {"score_claim": True},
        {"chunk_size": 13},
        {"predictor_archive_sha256": "BAD"},
        {"predictor_archive_path": ""},
    ):
        with pytest.raises((SolveDiffMiningError, ValidationError)):
            SolveDiffMiningConfigV1.model_validate({**base, **update})


def test_canonical_json_round_trip_and_nonfinite_refusal() -> None:
    payload = {"z": [3, 2, 1], "a": {"finite": 1.25}}
    encoded = canonical_json_bytes(payload)
    assert encoded == b'{"a":{"finite":1.25},"z":[3,2,1]}'
    assert json.loads(canonical_jsonl_line(payload)) == payload
    for value in (float("nan"), float("inf"), np.asarray([1.0, np.nan])):
        with pytest.raises(SolveDiffMiningError):
            canonical_json_bytes({"bad": value})


def test_range_kernel_split_and_uint8_reachability_are_separate() -> None:
    kernel = FullResizeKernel.build(camera_h=8, camera_w=10, scorer_h=4, scorer_w=5)
    rng = np.random.default_rng(9)
    delta = rng.normal(size=(8, 10, 3)).astype(np.float64)
    split = range_kernel_energy_split(delta, kernel)
    np.testing.assert_allclose(split["range"] + split["ker"], delta, atol=1e-12)
    scale = max(1.0, split["range_energy"] * split["ker_energy"])
    assert split["orthogonality_abs"] / np.sqrt(scale) < 1e-11
    reachable = uint8_reachability_accounting(np.full((8, 10, 3), 127, dtype=np.uint8), kernel)
    assert "lower_bound" in reachable["semantics"]
    assert reachable["feasible_basis_directions_lower_bound"] <= reachable["full_basis_directions"]


def test_resize_adjoint_exact_dot_product() -> None:
    rng = np.random.default_rng(11)
    camera = rng.normal(size=(1, 8, 10, 5))
    scorer_costate = rng.normal(size=(4, 5, 5))
    forward = bilinear_resize_forward(camera, 4, 5)[0]
    pulled = exact_resize_adjoint_pullback(scorer_costate, 8, 10)
    np.testing.assert_allclose(np.vdot(forward, scorer_costate), np.vdot(camera[0], pulled), rtol=1e-12, atol=1e-12)


def test_rank4_gauge_null_and_costate_blocker_round_trip(tmp_path: Path) -> None:
    result = rank4_head_accounting(np.zeros((2, 3, 5), dtype=np.float64))
    assert result["rank"] == 4
    assert result["gauge_null_linf"] < 1e-10
    row = SolveDiffCostateRowV1(
        pair_id=0,
        class_pair="Road-Lane",
        coefficient_family="rank4_head_chart",
        coefficient_index=0,
        value=1.0,
        head_linearization="EXACT_RANK4_5CLASS_QUOTIENT",
        resize_adjoint="EXACT_BILINEAR_RESIZE_TRANSPOSE",
        inner_encoder_jacobian="ABSENT",
        blocker_status=INNER_JACOBIAN_BLOCKER,
        source_coordinates="CACHED_TARGET_LABEL_MARGIN_AND_ENDPOINT_RESIDUAL",
    )
    path = tmp_path / "costate.jsonl"
    path.write_bytes(canonical_jsonl_line(row))
    assert list(iter_costate_rows(path)) == [row]
    assert row.exact_frozen_segnet_input_gradient is False


def test_real_coders_are_deterministic_and_chart_is_not_fourier() -> None:
    payload = np.arange(128, dtype=np.int16)
    assert coded_byte_counts(payload) == coded_byte_counts(payload)
    coded = coded_byte_counts(payload)
    assert coded["selected_bytes"] in (coded["zlib"], coded["lzma"])
    chart = compact_parabolic_shearlet_coefficients(np.arange(64, dtype=np.float64).reshape(8, 8), block=4)
    assert chart.shape == (4, 5)
    assert "fourier" not in compact_parabolic_shearlet_coefficients.__name__.lower()


def test_xi_transport_is_deterministic_and_held_out() -> None:
    twists = np.zeros((7, 6), dtype=np.float64)
    twists[:, 0] = np.arange(7) * 0.1
    twists[:, 5] = np.arange(7) * 0.01
    features = xi_features_from_twists(twists)
    np.testing.assert_allclose(features, xi_features_from_twists(twists))
    targets = features @ np.arange(12, dtype=np.float64).reshape(6, 2)
    rows = leave_one_window_out_transport(features, targets, window=2, ridge=1e-6)
    assert len(rows) == 3
    assert all(row["window_stop"] - row["window_start"] <= 2 for row in rows)
    assert all(np.isfinite(row["heldout_explained_squared_energy"]) for row in rows)
    autocorr = xi_autocorrelation_rows(features, max_lag=2)
    assert len(autocorr) == 12
    assert all(-1.0 <= row["autocorrelation"] <= 1.0 for row in autocorr)


def test_movable_birth_post_birth_and_residual_partition() -> None:
    actual = np.asarray([[0, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
    predicted = np.asarray([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
    result = partition_movable_innovations(actual, predicted)
    assert result == {
        "birth_frame_innovation_energy": 2.0,
        "movable_post_birth_predictable_energy": 1.0,
        "per_frame_residual_energy": 1.0,
    }
    carried = partition_movable_innovations(
        actual[1:2],
        predicted[1:2],
        previous_support=np.asarray([1, 0, 0], dtype=np.float64),
    )
    assert carried["birth_frame_innovation_energy"] == 0.0
    assert carried["per_frame_residual_energy"] == 1.0


def test_compact_movable_features_preserve_small_island_presence() -> None:
    delta = np.zeros((8, 8, 3), dtype=np.float64)
    movable = np.zeros((8, 8), dtype=bool)
    movable[0, 0] = True
    _endpoint, support = compact_temporal_features(delta, movable)
    assert support.shape == (16,)
    assert np.count_nonzero(support) == 1
    assert set(support) == {0.0, 1.0}


def test_tolerance_ladder_is_monotone_and_derived() -> None:
    delta = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    rows = derived_tolerance_ladder(delta, (1.0, 0.75, 0.25, 0.0))
    energies = [row["derived_energy"] for row in rows]
    assert energies == sorted(energies, reverse=True)
    assert all(row["label"] == "DERIVED_TOLERANCE_LADDER" for row in rows)
    assert not any(row["evaluator_measurement"] for row in rows)


def test_flip_distance_uses_registered_cheapest_target_hyperplane() -> None:
    rival, norm = cheapest_target_hyperplane(CLASS_ORDER.index("Lane"))
    assert rival == "Movable"
    assert norm == pytest.approx(4.007)
    histogram = flip_distance_histogram(np.asarray([0.0, 0.024, 0.025, 0.2, 2.0]))
    assert sum(histogram.values()) == 5
    assert histogram["[0,0.025)"] == 2
    assert histogram["[0.025,0.05)"] == 1


def test_production_chunk_keeps_both_endpoints_as_scorer_planes(tmp_path: Path) -> None:
    kernel = FullResizeKernel.build(camera_h=8, camera_w=10, scorer_h=4, scorer_w=5)
    ids = (0, 1)
    y0 = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    y1 = np.flip(y0, axis=2).copy()
    y0_path = tmp_path / "y0.bin"
    y1_path = tmp_path / "y1.bin"
    y0_path.write_bytes(y0.tobytes())
    y1_path.write_bytes(y1.tobytes())

    class Receiver:
        predictor = SimpleNamespace(source_pair_start=0, z=SimpleNamespace(n_pairs=2))

        @staticmethod
        def render_pairs(pair_ids: tuple[int, ...]) -> np.ndarray:
            assert pair_ids == ids
            return np.full((2, 2, 4, 5, 3), 127, dtype=np.uint8)

    receipt = {
        "chunks": [
            {
                "chunk_index": 0,
                "pair_ids": list(ids),
                "y0": {"path": str(y0_path), "sha256": sha256(y0.tobytes()).hexdigest()},
                "y1": {"path": str(y1_path), "sha256": sha256(y1.tobytes()).hexdigest()},
            }
        ]
    }
    labels = np.zeros((2, 4, 5), dtype=np.int64)
    context = ProductionInputContext(
        receipt,
        Receiver(),
        labels,
        np.ones_like(labels, dtype=np.float32),
        np.zeros((2, 6), dtype=np.float64),
    )
    chunk = _load_production_inputs(context, _fixture_config(tmp_path), ids, kernel)
    assert chunk.solved_planes.shape == (2, 2, 4, 5, 3)
    assert chunk.predictor_planes.shape == (2, 2, 4, 5, 3)
    solved_camera = realize_solve_camera(chunk.solved_planes[0, 0], kernel)
    predictor_camera = realize_solve_camera(chunk.predictor_planes[0, 0], kernel)
    assert solved_camera.shape == predictor_camera.shape == (8, 10, 3)


def test_start_receipt_sha_and_measured_ladder_persist(tmp_path: Path) -> None:
    base = _fixture_config(tmp_path).model_dump(mode="python", by_alias=True)
    predictor_sha = "a" * 64
    receipt = {
        "schema": "direct_description_v12_obligation_drain_receipt.v1",
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
        "ladder": [
            {
                "effective_added_budget_bytes": 0,
                "archive": {
                    "bytes": 102105,
                    "sha256": predictor_sha,
                    "receiver_closed": True,
                },
                "bridge": {
                    "pose": {"d_pose": "2.5"},
                    "segmentation": {
                        "d_seg": "0.25",
                        "strata": {"target_class": {"Road": {"d_seg": "0.1"}}},
                    },
                },
            }
        ],
    }
    receipt_path = tmp_path / "start.json"
    receipt_bytes = canonical_json_bytes(receipt)
    receipt_path.write_bytes(receipt_bytes)
    config = SolveDiffMiningConfigV1.model_validate(
        {
            **base,
            "input_mode": "production",
            "pair_count": 600,
            "chunk_size": 12,
            "predictor_archive_sha256": predictor_sha,
            "start_receipt_path": str(receipt_path),
            "start_receipt_sha256": sha256(receipt_bytes).hexdigest(),
        }
    )
    rows, _ = _load_start_receipt_rows(config)
    assert [row.attribution_status for row in rows] == [
        "MEASURED_RECEIPT_GLOBAL",
        "MEASURED_RECEIPT_AGGREGATE_STRATUM",
    ]
    receipt_path.write_bytes(receipt_bytes + b"\n")
    with pytest.raises(SolveDiffMiningError, match="SHA-256 mismatch"):
        _load_start_receipt_rows(config)


def test_chunk_sha_write_once_and_resume_revalidation(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk.bin"
    raw = np.arange(2 * 2 * 3, dtype=np.uint8).tobytes()
    chunk.write_bytes(raw)
    array = load_checked_uint8_chunk(chunk, sha256(raw).hexdigest(), pair_count=1, height=2, width=2)
    assert array.shape == (1, 2, 2, 3)
    with pytest.raises(SolveDiffMiningError):
        load_checked_uint8_chunk(chunk, "0" * 64, pair_count=1, height=2, width=2)
    checkpoint = tmp_path / "checkpoint.json"
    assert write_once_stage_checkpoint(checkpoint, {"a": 1}) == "WRITTEN"
    assert write_once_stage_checkpoint(checkpoint, {"a": 1}) == "RESUME_SKIP"
    with pytest.raises(SolveDiffMiningError):
        write_once_stage_checkpoint(checkpoint, {"a": 2})

    output = tmp_path / "run"
    config = _fixture_config(tmp_path)
    first = run_mining_pass(config, output, pair_limit=6, argv=["fixture-smoke"])
    assert first.completed_stages == 2
    assert first.processed_pair_ids == tuple(range(6))
    assert not list(output.rglob("*.zip"))
    resumed = run_mining_pass(config, output, pair_limit=6, resume=True, argv=["fixture-smoke", "--resume"])
    assert resumed.model_dump() == first.model_dump()
    checkpoint_payload = json.loads((output / "stages" / "pairs_0000_0003" / "checkpoint.json").read_text())
    assert len(checkpoint_payload["instrument_module_sha256"]) == 64
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage_producer_module_sha256"] == receipt["finalizer_module_sha256"]
    assert receipt["stage_module_override_used"] is False
    with pytest.raises(SolveDiffMiningError, match="custody mismatch"):
        run_mining_pass(
            config,
            output,
            pair_limit=6,
            resume=True,
            resume_stage_module_sha256="0" * 64,
        )
    stage_member = output / "stages" / "pairs_0000_0003" / "pair_rows.jsonl"
    stage_member.write_bytes(stage_member.read_bytes() + b"{}\n")
    with pytest.raises(SolveDiffMiningError, match="digest mismatch"):
        run_mining_pass(config, output, pair_limit=6, resume=True)


def test_fixture_cli_receipt_typed_outputs_charts_and_false_authority(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, pair_count=4, chunk_size=2)
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_json_bytes(config) + b"\n")
    output = tmp_path / "cli-output"
    repo = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        str(repo / "tools" / "measure_ddm_solve_diff_operator.py"),
        "--config",
        str(config_path),
        "--output-root",
        str(output),
        "--pair-limit",
        "4",
    ]
    completed = subprocess.run(command, cwd=repo, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", receipt["git_sha"])
    for key, expected in {
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
    }.items():
        assert receipt[key] == expected
    for name in (
        "pair_rows.jsonl",
        "stratum_rows.jsonl",
        "window_rows.jsonl",
        "costate_rows.jsonl",
        "tolerance_rows.jsonl",
        "start_receipt_rows.jsonl",
        "temporal_features.jsonl",
    ):
        assert (output / name).is_file()
    families = {
        json.loads(line)["coefficient_family"] for line in (output / "costate_rows.jsonl").read_text().splitlines()
    }
    assert families == {"rank4_head_chart", "compact_parabolic_shearlet"}
    pngs = sorted((output / "charts").glob("*.png"))
    htmls = sorted((output / "charts").glob("*.html"))
    assert len(pngs) == len(htmls) == 5
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in pngs)
    hard = next(path for path in htmls if path.name == "hard_pair_panels.html")
    assert hard.read_text().count('"panel_index"') == 3
    hard_png = next(path for path in pngs if path.name == "hard_pair_panels.png").read_bytes()
    assert struct.unpack(">II", hard_png[16:24]) == (392, 128)


def test_import_has_no_scorer_gpu_or_filesystem_side_effect(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[4]
    script = """
import json, pathlib, sys
before = sorted(p.name for p in pathlib.Path('.').iterdir())
import tac.optimization.solve_diff_operator_mining  # noqa: F401
after = sorted(p.name for p in pathlib.Path('.').iterdir())
forbidden = [name for name in sys.modules if name == 'mlx' or name.startswith('mlx.') or name == 'upstream.modules']
print(json.dumps({'before': before, 'after': after, 'forbidden': forbidden}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={"PYTHONPATH": str(repo / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result == {"before": [], "after": [], "forbidden": []}
