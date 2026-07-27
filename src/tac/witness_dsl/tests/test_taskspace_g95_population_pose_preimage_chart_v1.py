from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.taskspace_g95_population_pose_preimage_chart_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MISS_VERDICT_SCOPE,
    ONE_STATE_MISS_AXIS,
    OUTER_ZIP_SCORE_ADMISSION,
    POPULATION_TRANSFER_REQUEST,
    REACHABILITY_COORDINATE_SCOPE,
    G95ControlModeV1,
    ParsedPopulationPosePreimageBasisV1,
    ParsedPopulationPosePreimageCoefficientChunkV1,
    PopulationPosePreimageChartBatchResultV1,
    PopulationPosePreimageChartError,
    PopulationPosePreimageChartReceiverV1,
    PopulationPosePreimageChartWireSetV1,
    bilinear_resize_align_corners_false_numpy,
    encode_population_pose_preimage_basis,
    encode_population_pose_preimage_coefficient_chunk,
    parse_population_pose_preimage_basis,
    parse_population_pose_preimage_coefficient_chunk,
    population_state_key,
    richer_control_request_for_miss,
    validate_population_chunk_coverage,
)
from tools.measure_taskspace_g95_population_pose_preimage_chart import (
    DEFAULT_CONFIG,
    EXPECTED_CONDITIONING_SHA256,
    EXPECTED_G94_PRODUCT_SHA256,
    REACHABILITY_THRESHOLD,
    G95RunnerError,
    _live_competitive_target_snapshot,
    _load_config,
    _load_or_build_basis,
    _resume_state_array,
    _resume_state_key,
    _sha256_array,
    _validate_stage00_reuse_state,
    build_noop_control_rows,
)

PRODUCT_SHA = "a" * 64
CONDITIONING_SHA = "b" * 64
WHOLE_PRECONDITIONAL_SHA = "c" * 64
TARGET_TABLE_SHA = "d" * 64
POSENET_WEIGHTS_SHA = "e" * 64
SELECTED_TARGET_SHA = "f" * 64


def _preconditional(
    *,
    pair_count: int = 1,
    y0_value: int = 10,
    y1_value: int = 100,
) -> np.ndarray:
    result = np.empty(
        (pair_count, 2, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8,
    )
    result[:, 0] = y0_value
    result[:, 1] = y1_value
    return result


def _basis(
    *,
    basis_q: np.ndarray | None = None,
    basis_scales: np.ndarray | None = None,
    whole_preconditional_camera_sha256: str = WHOLE_PRECONDITIONAL_SHA,
) -> tuple[bytes, ParsedPopulationPosePreimageBasisV1]:
    if basis_q is None:
        basis_q = np.asarray([[[[1, 0, 0]]]], dtype=np.int8)
    rank = int(basis_q.shape[0])
    if basis_scales is None:
        basis_scales = np.ones(rank, dtype=np.float32)
    object_bytes = encode_population_pose_preimage_basis(
        g94_product_member_sha256=PRODUCT_SHA,
        g94_conditioning_state_sha256=CONDITIONING_SHA,
        whole_preconditional_camera_sha256=whole_preconditional_camera_sha256,
        selected_target_table_sha256=TARGET_TABLE_SHA,
        posenet_weights_sha256=POSENET_WEIGHTS_SHA,
        basis_q=basis_q,
        basis_scales=basis_scales,
    )
    return object_bytes, parse_population_pose_preimage_basis(
        object_bytes,
        expected_object_sha256=hashlib.sha256(object_bytes).hexdigest(),
    )


def _chunk(
    basis: ParsedPopulationPosePreimageBasisV1,
    *,
    source_pair_ids: tuple[int, ...] = (0,),
    preconditional_camera_pairs: np.ndarray | None = None,
    selected_target_sha256: str = SELECTED_TARGET_SHA,
    rank: int | None = None,
    coefficients_q: np.ndarray | None = None,
    coefficient_scales: np.ndarray | None = None,
) -> tuple[bytes, ParsedPopulationPosePreimageCoefficientChunkV1]:
    if preconditional_camera_pairs is None:
        preconditional_camera_pairs = _preconditional(pair_count=len(source_pair_ids))
    chunk_rank = basis.rank if rank is None else rank
    if coefficients_q is None:
        coefficients_q = np.full(
            (len(source_pair_ids), chunk_rank),
            2,
            dtype=np.int16,
        )
    if coefficient_scales is None:
        coefficient_scales = np.ones(chunk_rank, dtype=np.float32)
    object_bytes = encode_population_pose_preimage_coefficient_chunk(
        basis_object_sha256=basis.object_sha256,
        population_state_key_sha256=basis.population_state_key,
        preconditional_camera_sha256=_sha256_array(preconditional_camera_pairs),
        selected_target_sha256=selected_target_sha256,
        source_pair_ids=source_pair_ids,
        rank=chunk_rank,
        coefficients_q=coefficients_q,
        coefficient_scales=coefficient_scales,
    )
    return object_bytes, parse_population_pose_preimage_coefficient_chunk(
        object_bytes,
        expected_object_sha256=hashlib.sha256(object_bytes).hexdigest(),
    )


@pytest.fixture(scope="module")
def decoded_case() -> tuple[
    PopulationPosePreimageChartWireSetV1,
    np.ndarray,
    PopulationPosePreimageChartBatchResultV1,
]:
    _basis_bytes, basis = _basis()
    preconditional = _preconditional()
    _chunk_bytes, chunk = _chunk(
        basis,
        preconditional_camera_pairs=preconditional,
    )
    wire_set = PopulationPosePreimageChartWireSetV1(basis=basis, chunk=chunk)
    receiver = PopulationPosePreimageChartReceiverV1.open(
        basis,
        expected_g94_product_member_sha256=PRODUCT_SHA,
        expected_g94_conditioning_state_sha256=CONDITIONING_SHA,
        expected_whole_preconditional_camera_sha256=WHOLE_PRECONDITIONAL_SHA,
        expected_selected_target_table_sha256=TARGET_TABLE_SHA,
        expected_posenet_weights_sha256=POSENET_WEIGHTS_SHA,
    )
    result = receiver.decode_preconditional_chunk(chunk, preconditional)
    return wire_set, preconditional, result


def test_p_once_basis_and_chunk_are_strict_separate_counted_objects(
    decoded_case: tuple[
        PopulationPosePreimageChartWireSetV1,
        np.ndarray,
        PopulationPosePreimageChartBatchResultV1,
    ],
) -> None:
    wire_set, _preconditional_batch, _result = decoded_case
    assert wire_set.basis.basis_bytes in wire_set.basis.object_bytes
    assert not hasattr(wire_set.chunk, "basis_bytes")
    assert not hasattr(wire_set.chunk, "basis_q")
    assert wire_set.basis.counted_bytes + wire_set.chunk.counted_bytes == wire_set.total_counted_bytes
    assert wire_set.chunk.basis_object_sha256 == wire_set.basis.object_sha256
    assert wire_set.chunk.population_state_key == wire_set.basis.population_state_key
    assert wire_set.basis.population_state_key == population_state_key(
        g94_product_member_sha256=PRODUCT_SHA,
        g94_conditioning_state_sha256=CONDITIONING_SHA,
        whole_preconditional_camera_sha256=WHOLE_PRECONDITIONAL_SHA,
        selected_target_table_sha256=TARGET_TABLE_SHA,
        posenet_weights_sha256=POSENET_WEIGHTS_SHA,
    )
    assert wire_set.basis.outer_zip_score_admission == OUTER_ZIP_SCORE_ADMISSION
    assert wire_set.chunk.outer_zip_score_admission == OUTER_ZIP_SCORE_ADMISSION


def test_600_rows_are_covered_by_38_chunks_without_basis_duplication() -> None:
    basis_bytes, basis = _basis()
    chunks: list[ParsedPopulationPosePreimageCoefficientChunkV1] = []
    chunk_bytes = 0
    for start in range(0, 600, 16):
        ids = tuple(range(start, min(start + 16, 600)))
        raw, chunk = _chunk(
            basis,
            source_pair_ids=ids,
            preconditional_camera_pairs=_preconditional(pair_count=len(ids)),
        )
        assert not hasattr(chunk, "basis_bytes")
        chunks.append(chunk)
        chunk_bytes += len(raw)
    assert len(chunks) == 38
    assert validate_population_chunk_coverage(basis, chunks) == 600
    p_once_total = len(basis_bytes) + chunk_bytes
    duplicated_total = len(basis_bytes) * len(chunks) + chunk_bytes
    assert p_once_total < duplicated_total
    assert sum(chunk.counted_bytes for chunk in chunks) == chunk_bytes


@pytest.mark.parametrize("mutation", ["basis", "state", "rank", "gap", "overlap"])
def test_population_stream_ref_state_rank_and_coverage_mismatch_refused(
    mutation: str,
) -> None:
    _basis_bytes, basis = _basis()
    chunks = [
        _chunk(
            basis,
            source_pair_ids=tuple(range(start, min(start + 16, 600))),
            preconditional_camera_pairs=_preconditional(pair_count=min(16, 600 - start)),
        )[1]
        for start in range(0, 600, 16)
    ]
    if mutation == "basis":
        _foreign_bytes, foreign_basis = _basis(basis_q=np.asarray([[[[0, 1, 0]]]], dtype=np.int8))
        _raw, chunks[0] = _chunk(
            foreign_basis,
            source_pair_ids=chunks[0].source_pair_ids,
            preconditional_camera_pairs=_preconditional(pair_count=16),
        )
        expected = "basis reference"
    elif mutation == "state":
        _foreign_bytes, foreign_basis = _basis(whole_preconditional_camera_sha256="1" * 64)
        ids = chunks[0].source_pair_ids
        preconditional = _preconditional(pair_count=16)
        raw = encode_population_pose_preimage_coefficient_chunk(
            basis_object_sha256=basis.object_sha256,
            population_state_key_sha256=foreign_basis.population_state_key,
            preconditional_camera_sha256=_sha256_array(preconditional),
            selected_target_sha256=SELECTED_TARGET_SHA,
            source_pair_ids=ids,
            rank=basis.rank,
            coefficients_q=np.full((16, basis.rank), 2, dtype=np.int16),
            coefficient_scales=np.ones(basis.rank, dtype=np.float32),
        )
        chunks[0] = parse_population_pose_preimage_coefficient_chunk(
            raw,
            expected_object_sha256=hashlib.sha256(raw).hexdigest(),
        )
        expected = "whole-state"
    elif mutation == "rank":
        _raw, chunks[0] = _chunk(
            basis,
            source_pair_ids=chunks[0].source_pair_ids,
            preconditional_camera_pairs=_preconditional(pair_count=16),
            rank=2,
        )
        expected = "rank differs"
    elif mutation == "gap":
        chunks.pop(1)
        expected = "gap"
    else:
        chunks.insert(1, chunks[0])
        expected = "gap"
    with pytest.raises(PopulationPosePreimageChartError, match=expected):
        validate_population_chunk_coverage(basis, chunks)


def test_crc_eof_and_section_digest_rejection() -> None:
    basis_bytes, basis = _basis()
    chunk_bytes, _chunk_object = _chunk(basis)
    for payload, parser in (
        (basis_bytes, parse_population_pose_preimage_basis),
        (chunk_bytes, parse_population_pose_preimage_coefficient_chunk),
    ):
        with pytest.raises(PopulationPosePreimageChartError, match="CRC32"):
            parser(payload[:-1] + bytes((payload[-1] ^ 1,)))
        with pytest.raises(PopulationPosePreimageChartError, match=r"CRC32|EOF"):
            parser(payload + b"\x00")
        tampered = bytearray(payload)
        tampered[-5] ^= 1
        prefix = bytes(tampered[:-4])
        tampered[-4:] = int(__import__("zlib").crc32(prefix) & 0xFFFFFFFF).to_bytes(
            4,
            "big",
        )
        with pytest.raises(PopulationPosePreimageChartError, match="SHA-256"):
            parser(bytes(tampered))


def test_receiver_refuses_wrong_state_chunk_and_preconditional(
    decoded_case: tuple[
        PopulationPosePreimageChartWireSetV1,
        np.ndarray,
        PopulationPosePreimageChartBatchResultV1,
    ],
) -> None:
    wire_set, preconditional, _result = decoded_case
    receiver = PopulationPosePreimageChartReceiverV1.open(
        wire_set.basis,
        expected_g94_product_member_sha256=PRODUCT_SHA,
        expected_g94_conditioning_state_sha256=CONDITIONING_SHA,
        expected_whole_preconditional_camera_sha256=WHOLE_PRECONDITIONAL_SHA,
        expected_selected_target_table_sha256=TARGET_TABLE_SHA,
        expected_posenet_weights_sha256=POSENET_WEIGHTS_SHA,
    )
    wrong = preconditional.copy()
    wrong[0, 0, 0, 0, 0] ^= 1
    with pytest.raises(PopulationPosePreimageChartError, match="chunk-bound state"):
        receiver.decode_preconditional_chunk(wire_set.chunk, wrong)
    foreign_basis_bytes, foreign_basis = _basis(basis_q=np.asarray([[[[0, 1, 0]]]], dtype=np.int8))
    del foreign_basis_bytes
    _raw, foreign_chunk = _chunk(
        foreign_basis,
        preconditional_camera_pairs=preconditional,
    )
    with pytest.raises(PopulationPosePreimageChartError, match="basis reference"):
        receiver.decode_preconditional_chunk(foreign_chunk, preconditional)
    with pytest.raises(PopulationPosePreimageChartError, match="whole-state foreign key"):
        PopulationPosePreimageChartReceiverV1.open(
            wire_set.basis,
            expected_g94_product_member_sha256=PRODUCT_SHA,
            expected_g94_conditioning_state_sha256=CONDITIONING_SHA,
            expected_whole_preconditional_camera_sha256="1" * 64,
            expected_selected_target_table_sha256=TARGET_TABLE_SHA,
            expected_posenet_weights_sha256=POSENET_WEIGHTS_SHA,
        )


def test_numpy_receiver_is_deterministic_preserves_y1_and_changes_y0(
    decoded_case: tuple[
        PopulationPosePreimageChartWireSetV1,
        np.ndarray,
        PopulationPosePreimageChartBatchResultV1,
    ],
) -> None:
    wire_set, preconditional, first = decoded_case
    receiver = PopulationPosePreimageChartReceiverV1.open(
        wire_set.basis,
        expected_g94_product_member_sha256=PRODUCT_SHA,
        expected_g94_conditioning_state_sha256=CONDITIONING_SHA,
        expected_whole_preconditional_camera_sha256=WHOLE_PRECONDITIONAL_SHA,
        expected_selected_target_table_sha256=TARGET_TABLE_SHA,
        expected_posenet_weights_sha256=POSENET_WEIGHTS_SHA,
    )
    second = receiver.decode_preconditional_chunk(wire_set.chunk, preconditional)
    assert first.camera_sha256 == second.camera_sha256
    assert np.array_equal(first.camera_pairs[:, 1], preconditional[:, 1])
    assert np.all(first.camera_pairs[:, 0, :, :, 0] == 102)
    assert np.all(first.camera_pairs[:, 0, :, :, 1:] == 100)
    assert first.changed_y0_values > 0
    assert first.score_claim is False


def test_round_to_nearest_even_is_observable_at_half_ties() -> None:
    preconditional = _preconditional()
    preconditional[:, 1, :, :, 1] = 101
    _basis_bytes, basis = _basis(
        basis_q=np.asarray([[[[1, 1, 0]]]], dtype=np.int8),
        basis_scales=np.asarray([0.5], dtype=np.float32),
    )
    _chunk_bytes, chunk = _chunk(
        basis,
        preconditional_camera_pairs=preconditional,
        coefficients_q=np.asarray([[1]], dtype=np.int16),
    )
    receiver = PopulationPosePreimageChartReceiverV1.open(
        basis,
        expected_g94_product_member_sha256=PRODUCT_SHA,
        expected_g94_conditioning_state_sha256=CONDITIONING_SHA,
        expected_whole_preconditional_camera_sha256=WHOLE_PRECONDITIONAL_SHA,
        expected_selected_target_table_sha256=TARGET_TABLE_SHA,
        expected_posenet_weights_sha256=POSENET_WEIGHTS_SHA,
    )
    result = receiver.decode_preconditional_chunk(chunk, preconditional)
    assert int(result.camera_pairs[0, 0, 0, 0, 0]) == 100
    assert int(result.camera_pairs[0, 0, 0, 0, 1]) == 102


def test_numpy_bilinear_reference_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    source = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3) / np.float32(7)
    actual = bilinear_resize_align_corners_false_numpy(
        source,
        output_height=5,
        output_width=7,
    )
    expected = (
        torch.nn.functional.interpolate(
            torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0),
            size=(5, 7),
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-7)


class _TinyPoseNet:
    def preprocess_input(self, pairs):
        return pairs

    def __call__(self, pairs):
        means = pairs.mean(dim=(2, 3, 4))
        pose = means.new_empty((pairs.shape[0], 6))
        pose[:, 0] = means[:, 0]
        pose[:, 1] = means[:, 1]
        pose[:, 2] = means[:, 1] - means[:, 0]
        pose[:, 3] = means[:, 0] + means[:, 1]
        pose[:, 4] = means[:, 0] * 0.5
        pose[:, 5] = means[:, 1] * 0.5
        return {"pose": pose}


def test_noop_and_copy_controls_remain_explicit_real_replays() -> None:
    preconditional = _preconditional()
    pass_row, copy_row, pass_prediction, copy_prediction = build_noop_control_rows(
        posenet=_TinyPoseNet(),
        preconditional_camera_pairs=preconditional,
        targets=np.zeros((1, 6), dtype=np.float64),
    )
    assert pass_row["mode"] == G95ControlModeV1.PASS_PRECONDITIONAL_Y0.name
    assert copy_row["mode"] == G95ControlModeV1.COPY_EXACT_CONDITIONAL_Y1.name
    assert pass_row["packet_bytes"] == copy_row["packet_bytes"] == 0
    assert not np.array_equal(pass_prediction, copy_prediction)


def test_richer_request_is_static_basis_scoped_not_family_kill() -> None:
    request = richer_control_request_for_miss(
        g94_product_member_sha256=EXPECTED_G94_PRODUCT_SHA256,
        g94_conditioning_state_sha256=EXPECTED_CONDITIONING_SHA256,
        source_pair_ids=(0,),
        attempted_rank=24,
        attempted_grid_height=48,
        attempted_grid_width=64,
        exact_d_pose=0.001,
        reachability_threshold=REACHABILITY_THRESHOLD,
        requested_minimum_rank=48,
        requested_minimum_grid_height=64,
        requested_minimum_grid_width=96,
    )
    payload = request.to_dict()
    assert payload["one_state_miss_axis"] == ONE_STATE_MISS_AXIS
    assert payload["population_transfer_request"] == POPULATION_TRANSFER_REQUEST
    assert payload["family_dead_claim"] is False
    assert REACHABILITY_COORDINATE_SCOPE.startswith("SUFFICIENT_POSE_COORDINATE")
    assert MISS_VERDICT_SCOPE == "STATIC_SHARED_BASIS_AT_EXACT_G94_CONDITIONING_STATE"
    with pytest.raises(PopulationPosePreimageChartError, match="permissive"):
        replace(request, score_claim=True)


def test_resume_state_binds_whole_and_selected_state_and_code(
    tmp_path: Path,
) -> None:
    state_inputs = {
        "preconditional_camera_sha256": "1" * 64,
        "whole_preconditional_camera_sha256": "2" * 64,
        "selected_target_sha256": "3" * 64,
        "selected_target_table_sha256": "4" * 64,
        "posenet_weights_sha256": "5" * 64,
        "g94_product_member_sha256": EXPECTED_G94_PRODUCT_SHA256,
        "g94_conditioning_state_sha256": EXPECTED_CONDITIONING_SHA256,
        "config_sha256": "6" * 64,
        "receiver_module_sha256": "7" * 64,
        "measurement_tool_sha256": "8" * 64,
    }
    expected_key = _resume_state_key(source_pair_ids=(0,), **state_inputs)
    foreign_inputs = dict(state_inputs)
    foreign_inputs["whole_preconditional_camera_sha256"] = "9" * 64
    foreign_key = _resume_state_key(source_pair_ids=(0,), **foreign_inputs)
    assert foreign_key != expected_key

    stage = tmp_path / "stage"
    stage.mkdir()
    basis_q = np.zeros((1, 48, 64, 3), dtype=np.int8)
    basis_scales = np.ones(1, dtype=np.float32)
    singular_values = np.ones(1, dtype=np.float64)
    np.savez(
        stage / "basis_checkpoint.npz",
        basis_q=basis_q,
        basis_scales=basis_scales,
        singular_values=singular_values,
        resume_state_key=_resume_state_array(foreign_key),
    )
    (stage / "basis_receipt.json").write_text(
        json.dumps(
            {
                "resume_state_key": foreign_key,
                "basis_q_sha256": _sha256_array(basis_q),
                "basis_scales_sha256": _sha256_array(basis_scales),
                "selected_singular_values_sha256": _sha256_array(singular_values),
                "g94_conditioning_state_sha256": EXPECTED_CONDITIONING_SHA256,
                "costate_rows": [],
                "basis_metadata": {},
            }
        )
    )
    with pytest.raises(G95RunnerError, match="resume-state key differs"):
        _load_or_build_basis(
            stage=stage,
            posenet=None,
            exact_y1_uint8=np.empty((0,), dtype=np.uint8),
            targets=np.empty((0, 6), dtype=np.float64),
            rank=1,
            config={},
            resume_state_key=expected_key,
        )


@pytest.mark.parametrize("identity_name", ["config", "receiver_module", "measurement_tool"])
def test_stage00_rejects_different_receiver_tool_or_config(identity_name: str) -> None:
    current = {
        "resume_state_sources_hashed_before_scorer_load": {
            name: {
                "path": f"/repo/{name}",
                "bytes": 10,
                "sha256": character * 64,
            }
            for name, character in (
                ("config", "1"),
                ("receiver_module", "2"),
                ("measurement_tool", "3"),
            )
        }
    }
    preserved = json.loads(json.dumps(current))
    preserved["resume_state_sources_hashed_before_scorer_load"][identity_name]["sha256"] = "f" * 64
    with pytest.raises(G95RunnerError, match="receiver/tool/config"):
        _validate_stage00_reuse_state(preserved, current)


def test_receiver_has_no_scorer_or_torch_dependency_and_dataclass_fields_unique() -> None:
    module_path = Path(__file__).resolve().parents[1] / "taskspace_g95_population_pose_preimage_chart_v1.py"
    source = module_path.read_text()
    assert "import torch" not in source
    assert "PoseNet" not in source
    assert "gt_poses" not in source
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
            or (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "dataclass"
            )
            for decorator in node.decorator_list
        ):
            continue
        names = [
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        ]
        assert len(names) == len(set(names)), node.name


def test_frozen_config_binds_exact_g94_and_research_boundary() -> None:
    config = _load_config(DEFAULT_CONFIG)
    assert config["g94"]["parent_git_commit"] == "9e84c69b8a389337270b70fd4023a4174ef3c552"
    assert config["g94"]["product_member_sha256"] == EXPECTED_G94_PRODUCT_SHA256
    assert config["g94"]["conditioning_state_sha256"] == EXPECTED_CONDITIONING_SHA256
    assert config["pair_start"] == 0
    assert config["pair_count"] == 1
    assert config["scorer_batch_pairs"] <= 16
    assert config["rank_ladder"] == [6, 12, 24]
    assert config["reachability_threshold_d_pose"] == 0.00047366
    assert config["research_only"] is True
    assert config["score_claim"] is False


def test_live_competitive_target_is_pointer_derived_and_custodied() -> None:
    snapshot = _live_competitive_target_snapshot()
    assert snapshot["score_to_beat"] == snapshot["effective_frontier"]["score"]
    assert snapshot["selection_rule"] == (
        "min(our_local_frontier_contest_cpu, our_local_frontier_contest_cuda, upstream_official_leaderboard.best_entry)"
    )
    assert snapshot["role"] == "REPORTING_CUSTODY_ONLY_NOT_G95_FIT_OR_DECODE_STATE"
    assert len(snapshot["sha256"]) == 64
