from __future__ import annotations

import base64
import json

import pytest

from tac.optimization.direct_description_measurement_ladder import (
    _ANCHOR_RECORD,
    _GRADIENT_RECORD,
    _POSE_RECORD,
    _RESIDUAL_RECORD,
    CHARTS_PER_PLANE,
    MEMBER_BY_STREAM,
    STREAM_ORDER,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
    DirectDescriptionMeasurementLadderCheckpointV1,
    DirectDescriptionMeasurementLadderConfigV1,
    compile_chart_archive,
    parse_chart_archive,
    prove_sampled_noop_honesty,
    receive_chart_archive,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError, _sha256


def _synthetic_z(n_pairs: int = 2) -> DirectDescriptionChartZV1:
    bodies = {name: bytearray() for name in STREAM_ORDER}
    for pair_id in range(n_pairs):
        for plane_id in range(2):
            bodies["global_chart_anchors"].extend(
                _ANCHOR_RECORD.pack(pair_id, plane_id, 96 + pair_id, 112 + plane_id, 128)
            )
            bodies["axial_chart_gradients"].extend(_GRADIENT_RECORD.pack(pair_id, plane_id, 3, -2, 1, -4, 2, -1))
            for stratum_index, stream_name in enumerate(STREAM_ORDER[2:5]):
                for chart_id in range(stratum_index * 64, (stratum_index + 1) * 64):
                    bodies[stream_name].extend(
                        _RESIDUAL_RECORD.pack(
                            pair_id,
                            plane_id,
                            chart_id,
                            chart_id % 5,
                            -(chart_id % 3),
                            chart_id % 2,
                        )
                    )
        bodies["pose6_pair_codes"].extend(_POSE_RECORD.pack(pair_id, *(pair_id + value for value in range(6))))
    return DirectDescriptionChartZV1(
        n_pairs=n_pairs,
        **{name: CountedChartStreamV1(payload=bytes(bodies[name])) for name in STREAM_ORDER},
    )


def test_chart_archive_is_deterministic_parse_exact_and_full_resolution() -> None:
    z = _synthetic_z()
    first = compile_chart_archive(z)
    second = compile_chart_archive(z)
    assert first.archive == second.archive
    assert parse_chart_archive(first.archive).archive == first.archive
    receiver = receive_chart_archive(first.archive)
    assert receiver.render_pairs((0, 1)).shape == (2, 2, 384, 512, 3)
    assert receiver.render_pairs((0,)).dtype.name == "uint8"
    assert receiver.custody["all_archive_bytes_have_one_home"] is True
    assert receiver.custody["unique_home_coverage_bytes"] == len(first.archive)
    assert [row["member"] for row in receiver.custody["stream_ledger"]] == [
        MEMBER_BY_STREAM[name] for name in STREAM_ORDER
    ]


def test_all_chart_records_are_owned_per_chart_not_per_pixel() -> None:
    z = _synthetic_z(1)
    receiver = receive_chart_archive(compile_chart_archive(z).archive)
    assert receiver.residuals.shape == (1, 2, 12, 16, 3)
    assert CHARTS_PER_PLANE == 192
    assert {row["ownership"] for row in receiver.custody["stream_ledger"]} == {"per_chart_or_per_stratum_not_per_pixel"}


def test_sampled_noop_honesty_covers_six_streams_and_archive_homes() -> None:
    proof = prove_sampled_noop_honesty(_synthetic_z(1))
    assert proof["semantic_samples"] == 14
    assert proof["all_six_streams_sampled"] is True
    assert proof["all_semantic_samples_changed_receiver_output"] is True
    assert proof["archive_home_samples"] == proof["archive_home_samples_refused"]


def test_archive_rejects_noncanonical_mutation() -> None:
    archive = bytearray(compile_chart_archive(_synthetic_z(1)).archive)
    archive[len(archive) // 2] ^= 1
    with pytest.raises(DirectDescriptionError):
        receive_chart_archive(bytes(archive))


def test_config_requires_rung2_at_least_256() -> None:
    row = {
        "schema": "DirectDescriptionMeasurementLadderConfigV1",
        "rung2_pairs": 255,
        "target_receipt_path": "target.json",
        "target_receipt_sha256": "0" * 64,
    }
    with pytest.raises(ValueError):
        DirectDescriptionMeasurementLadderConfigV1.model_validate(row)


def test_checkpoint_envelope_is_canonical_and_tamper_evident() -> None:
    z = _synthetic_z(64)
    archive = compile_chart_archive(z).archive
    config = DirectDescriptionMeasurementLadderConfigV1(
        rung2_pairs=256,
        target_receipt_path="target.json",
        target_receipt_sha256="0" * 64,
    )
    argv = ("python3", "tools/run_direct_description_measurement_ladder.py")
    history = (
        {
            "stage_index": 0,
            "stage_name": "rung1_n64_full_resolution",
            "archive_sha256": _sha256(archive),
        },
    )
    checkpoint = DirectDescriptionMeasurementLadderCheckpointV1(
        config=config.model_dump(mode="json", by_alias=True),
        config_sha256=config.typed_config_hash(),
        dsl_compile_hash=config.dsl_compile_hash(),
        semantic_argv=argv,
        semantic_argv_sha256=_sha256("\0".join(argv).encode()),
        target_receipt_sha256=config.target_receipt_sha256,
        completed_stage_index=0,
        completed_stage_name="rung1_n64_full_resolution",
        next_stage_index=1,
        described_pairs=64,
        current_archive_b64=base64.b64encode(archive).decode(),
        current_archive_sha256=_sha256(archive),
        current_archive_bytes=len(archive),
        quantity_bridge={"archive_sha256": _sha256(archive)},
        stage_history=history,
    )
    payload = checkpoint.to_bytes()
    assert DirectDescriptionMeasurementLadderCheckpointV1.from_bytes(payload) == checkpoint
    envelope = json.loads(payload)
    envelope["body"]["described_pairs"] = 65
    with pytest.raises((DirectDescriptionError, ValueError)):
        DirectDescriptionMeasurementLadderCheckpointV1.from_bytes(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        )
