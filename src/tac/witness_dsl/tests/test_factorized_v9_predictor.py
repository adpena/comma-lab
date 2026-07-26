# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import lzma
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_entropy_priced_member import (
    StructuredS4SourcesV1,
)
from tac.optimization.direct_description_entropy_streams import (
    STREAM_ORDER,
    compile_entropy_chart_archive,
)
from tac.optimization.direct_description_measurement_ladder import (
    _ANCHOR_RECORD,
    _GRADIENT_RECORD,
    _POSE_RECORD,
    _RESIDUAL_RECORD,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
)
from tac.witness_dsl import factorized_v9_predictor as factorized_v9_predictor_module
from tac.witness_dsl.factorized_v9_predictor import (
    PREDICTOR_CONTRACT_ID,
    RENDERER_SOURCE_PATHS,
    FactorizedV9PredictorError,
    compile_factorized_v9_predictor,
    load_factorized_v9_predictor,
    receive_factorized_v9_predictor,
    renderer_source_manifest,
    renderer_source_sha256,
)
from tac.witness_dsl.predictor_bound_residual import (
    apply_predictor_bound_partition_residual,
    packet_accounting,
)
from tac.witness_dsl.progressive_geometry_residual import (
    build_progressive_geometry_residual,
)
from tac.witness_dsl.progressive_v9_entropy_measurement import (
    apply_progressive_v9_entropy_measurement,
)


def _chart_program() -> DirectDescriptionChartZV1:
    bodies = {name: bytearray() for name in STREAM_ORDER}
    for plane_id in range(2):
        bodies["global_chart_anchors"].extend(_ANCHOR_RECORD.pack(0, plane_id, 96, 112, 128))
        bodies["axial_chart_gradients"].extend(_GRADIENT_RECORD.pack(0, plane_id, 0, 0, 0, 0, 0, 0))
        for stratum, stream_name in enumerate(STREAM_ORDER[2:5]):
            for chart_id in range(stratum * 64, (stratum + 1) * 64):
                bodies[stream_name].extend(_RESIDUAL_RECORD.pack(0, plane_id, chart_id, 0, 0, 0))
    bodies["pose6_pair_codes"].extend(_POSE_RECORD.pack(0, 20, 21, 22, 23, 24, 25))
    return DirectDescriptionChartZV1(
        n_pairs=1,
        **{name: CountedChartStreamV1(payload=bytes(bodies[name])) for name in STREAM_ORDER},
    )


def _lane_payload() -> tuple[bytes, tuple[tuple[np.ndarray, ...], ...], dict[str, object]]:
    vector = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 50.0])
    lines = tuple((vector.copy(),) for _ in range(600))
    header: dict[str, object] = {
        "cx": 256.0,
        "dash_forward_max_m": 50.0,
        "dash_gate": True,
        "rd": {"K": 1, "base_steps": [0.25] * 11, "d_slot": 11, "n_pairs": 600},
        "softness": 1.0,
        "v_h": 150.0,
    }
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    presence = np.packbits(np.ones((600, 1), dtype=np.uint8)).tobytes()
    quantized = np.rint(np.stack([vector] * 600) / 0.25).astype(np.int64)
    delta = np.diff(quantized, axis=0, prepend=np.zeros((1, 11), dtype=np.int64))
    zigzag = ((delta << 1) ^ (delta >> 63)).astype("<u4")
    raw = (
        b"LBND2\0"
        + struct.pack("<I", len(raw_header))
        + raw_header
        + struct.pack("<I", len(presence))
        + presence
        + zigzag.tobytes(order="C")
    )
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]
    return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filters), lines, header


def _program() -> bytes:
    empty = tuple(() for _ in range(600))
    classes: list[tuple[tuple[np.ndarray, ...], ...]] = []
    for class_id in range(5):
        rows = list(empty)
        rows[0] = (np.asarray([class_id * 8, class_id * 8 + 1], dtype=np.int64),)
        classes.append(tuple(rows))
    road = np.zeros((384, 512), dtype=bool)
    road[200:220, 100:120] = True
    hood = np.zeros((384, 512), dtype=bool)
    hood[360:, 220:292] = True
    lane_encoded, lane_lines, lane_header = _lane_payload()
    sources = StructuredS4SourcesV1(
        pair_count=600,
        palette=np.asarray(
            [[153, 255, 51], [51, 255, 204], [0, 153, 0], [102, 204, 51], [0, 255, 153]],
            dtype=np.uint8,
        ),
        camera={"height_m": 1.2, "fx_scorer": 400.3, "fy_scorer": 399.5},
        static_masks={
            "Road": road,
            "Undrivable": np.zeros((384, 512), dtype=bool),
            "MyCar": hood,
        },
        lane_encoded=lane_encoded,
        lane_lines=lane_lines,
        lane_header=lane_header,
        events=tuple(classes),
        components=tuple(empty for _ in range(5)),
        custody={"fixture": True},
        role_class_ids={"Road": 0, "Lane": 1, "UndrivableBoundary": 2, "Movable": 3, "MyCar": 4},
        role_rgb_u8={
            "Road": (153, 255, 51),
            "Lane": (51, 255, 204),
            "UndrivableBoundary": (0, 153, 0),
            "Movable": (102, 204, 51),
            "MyCar": (0, 255, 153),
        },
        routing_custody={"fixture": True},
    )
    baseline = compile_entropy_chart_archive(_chart_program()).archive
    return compile_factorized_v9_predictor(baseline, sources, pair_start=0).program


@pytest.fixture(scope="module")
def program() -> bytes:
    return _program()


def test_real_v9_archive_grammar_decodes_all_five_factor_roles(program: bytes) -> None:
    receiver = receive_factorized_v9_predictor(program)
    labels = receiver.decode_all_semantics(batch_size=1)
    assert labels.dtype == np.uint8
    assert labels.shape == (1, 384, 512)
    assert set(np.unique(labels).tolist()) == set(range(5))
    assert labels[0, 205, 105] == 0  # static Road bulk
    assert labels[0, 100, 100] == 2  # Undrivable complement
    assert labels[0, 0, 24] == 3  # decoded Movable event site
    assert labels[0, 370, 250] == 4  # static MyCar component
    assert np.count_nonzero(labels == 1) > 0  # decoded Lane chart

    binding = receiver.semantic_binding(labels)
    assert binding["all_five_factor_roles_consumed"] is True
    assert binding["predictor_program_sha256"] == hashlib.sha256(program).hexdigest()
    assert binding["target_table_bytes"] == 0
    assert binding["decode_scorer_dependency"] is False
    assert binding["score_claim"] is False
    assert binding["source_pair_start"] == 0
    assert binding["source_pair_stop_exclusive"] == 1
    assert receiver.source_pair_ids == (0,)
    assert binding["temporal_pose6_shape"] == [1, 6]
    assert len(binding["temporal_pose6_sha256"]) == 64
    assert binding["counted_temporal_pose6_bound"] is True


def test_pbr1_recomputes_against_fresh_program_decode(program: bytes) -> None:
    receiver = receive_factorized_v9_predictor(program)
    predictor = receiver.decode_all_semantics()
    target = predictor.copy()
    target[0, 100, 100] = 0
    packet = receiver.build_pbr1(target)
    recovered = apply_predictor_bound_partition_residual(
        packet,
        predictor_program=program,
        predictor_contract_id=PREDICTOR_CONTRACT_ID,
        predictor_renderer_sha256=receiver.source_manifest_sha256,
        predictor_labels=receiver.decode_all_semantics(),
    )
    assert np.array_equal(recovered, target)
    accounting = packet_accounting(packet)
    assert accounting["event_count"] == 1
    assert accounting["separate_dense_target_table_section_bytes"] == 0
    assert accounting["candidate_payload_allowed"] is False


def test_source_closure_is_content_addressed_and_deterministic() -> None:
    first = renderer_source_manifest()
    second = renderer_source_manifest()
    assert first == second
    paths = [row["path"] for row in first["sources"]]
    assert paths == list(RENDERER_SOURCE_PATHS)
    assert "src/tac/optimization/direct_description_minimizer.py" in paths
    assert "src/tac/witness_dsl/predictor_bound_residual.py" in paths
    assert first["source_identity_scope"] == "declared_semantic_interpreter_and_wire_modules.v1"
    assert renderer_source_sha256() == renderer_source_sha256()
    assert len(renderer_source_sha256()) == 64


def test_file_loader_uses_exact_single_read_custody(tmp_path: Path, program: bytes) -> None:
    path = tmp_path / "v9_program.zip"
    path.write_bytes(program)
    digest = hashlib.sha256(program).hexdigest()
    receiver = load_factorized_v9_predictor(path, expected_sha256=digest)
    assert receiver.program == program
    with pytest.raises(FactorizedV9PredictorError, match="custody mismatch"):
        load_factorized_v9_predictor(path, expected_sha256="0" * 64)


def test_file_loader_reads_program_larger_than_one_mib_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(256)) * 8193
    path = tmp_path / "large_v9_program.zip"
    path.write_bytes(payload)

    def return_exact_program(
        program: bytes,
        *,
        repository_root: Path | None = None,
    ) -> bytes:
        assert repository_root is None
        return program

    monkeypatch.setattr(
        factorized_v9_predictor_module,
        "receive_factorized_v9_predictor",
        return_exact_program,
    )
    assert (
        factorized_v9_predictor_module.load_factorized_v9_predictor(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
        == payload
    )


def test_invalid_program_and_caller_attested_semantics_refuse(program: bytes) -> None:
    with pytest.raises(FactorizedV9PredictorError, match="receiver-closed"):
        receive_factorized_v9_predictor(program[:-1])
    receiver = receive_factorized_v9_predictor(program)
    labels = receiver.decode_all_semantics()
    labels[0, 10, 10] = (int(labels[0, 10, 10]) + 1) % 5
    with pytest.raises(FactorizedV9PredictorError, match="fresh program decode"):
        receiver.semantic_binding(labels)


def test_identity_decode_reopens_program_not_mutable_cached_receiver(program: bytes) -> None:
    receiver = receive_factorized_v9_predictor(program)
    road = next(layer for layer in receiver.receiver.layers if layer.role == "Road")
    assert road.static_mask is not None
    road.static_mask[205, 105] = False
    assert receiver.decode_all_semantics(batch_size=1)[0, 205, 105] == 0

    source_manifest = receiver.source_manifest
    assert isinstance(source_manifest, dict)
    source_manifest["schema"] = "mutated"
    with pytest.raises(FactorizedV9PredictorError, match="source manifest changed"):
        receiver.decode_all_semantics(batch_size=1)


def test_progressive_measurement_rederives_v9_semantics_in_fresh_process(
    tmp_path: Path,
    program: bytes,
) -> None:
    receiver = receive_factorized_v9_predictor(program)
    target = receiver.decode_all_semantics()
    target[0, 100, 100] = np.uint8((int(target[0, 100, 100]) + 1) % 5)
    residual = build_progressive_geometry_residual(
        predictor_program=program,
        predictor_contract_id=PREDICTOR_CONTRACT_ID,
        predictor_renderer_sha256=receiver.source_manifest_sha256,
        predictor_labels=receiver.decode_all_semantics(),
        target_labels=target,
        source_pair_ids=receiver.source_pair_ids,
        target_semantic_lineage="synthetic_fixture",
    )
    assert np.array_equal(
        apply_progressive_v9_entropy_measurement(program, residual),
        target,
    )

    program_path = tmp_path / "predictor.zip"
    residual_path = tmp_path / "residual.pbr2"
    program_path.write_bytes(program)
    residual_path.write_bytes(residual)
    script = """
import hashlib
import sys
from pathlib import Path
from tac.witness_dsl.progressive_v9_entropy_measurement import apply_progressive_v9_entropy_measurement

program = Path(sys.argv[1]).read_bytes()
residual = Path(sys.argv[2]).read_bytes()
output = apply_progressive_v9_entropy_measurement(program, residual)
print(hashlib.sha256(output.tobytes(order="C")).hexdigest())
"""
    root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(program_path), str(residual_path)],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == hashlib.sha256(target.tobytes(order="C")).hexdigest()
