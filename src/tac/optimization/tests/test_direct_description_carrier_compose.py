# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import lzma
import struct

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    DirectDescriptionV9CarrierComposeConfigV1,
    compile_carrier_compose_archive,
    parse_carrier_compose_archive,
    prove_carrier_archive_fail_closed,
    receive_carrier_compose_archive,
    recursive_carrier_byte_rows,
)
from tac.optimization.direct_description_entropy_priced_member import (
    StructuredS4SourcesV1,
    compile_composed_structured_member_archive,
)
from tac.optimization.direct_description_entropy_streams import STREAM_ORDER, compile_entropy_chart_archive
from tac.optimization.direct_description_measurement_ladder import (
    _ANCHOR_RECORD,
    _GRADIENT_RECORD,
    _POSE_RECORD,
    _RESIDUAL_RECORD,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.predictor_upgrade_xi_chart import LaneCoefficientDelta


def _z() -> DirectDescriptionChartZV1:
    bodies = {name: bytearray() for name in STREAM_ORDER}
    for pair_id in range(64):
        for plane_id in range(2):
            bodies["global_chart_anchors"].extend(_ANCHOR_RECORD.pack(pair_id, plane_id, 96, 112, 128))
            bodies["axial_chart_gradients"].extend(_GRADIENT_RECORD.pack(pair_id, plane_id, 0, 0, 0, 0, 0, 0))
            for stratum, stream_name in enumerate(STREAM_ORDER[2:5]):
                for chart_id in range(stratum * 64, (stratum + 1) * 64):
                    bodies[stream_name].extend(_RESIDUAL_RECORD.pack(pair_id, plane_id, chart_id, 0, 0, 0))
        bodies["pose6_pair_codes"].extend(_POSE_RECORD.pack(pair_id, 20, 21, 22, 23, 24, 25))
    return DirectDescriptionChartZV1(
        n_pairs=64,
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


def _predictor() -> bytes:
    empty = tuple(() for _ in range(64))
    classes: list[tuple[tuple[np.ndarray, ...], ...]] = [empty for _ in range(5)]
    for class_id in range(5):
        rows = list(empty)
        rows[0] = (np.asarray([class_id * 8, class_id * 8 + 1], dtype=np.int64),)
        classes[class_id] = tuple(rows)
    road = np.zeros((384, 512), dtype=bool)
    road[200:220, 100:120] = True
    hood = np.zeros((384, 512), dtype=bool)
    hood[360:, 220:292] = True
    encoded, lines, header = _lane_payload()
    sources = StructuredS4SourcesV1(
        pair_count=64,
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
        lane_encoded=encoded,
        lane_lines=lines,
        lane_header=header,
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
    baseline = compile_entropy_chart_archive(_z()).archive
    return compile_composed_structured_member_archive(baseline, sources, pair_start=0)[0]


def test_v9_config_is_strict_local_only_and_window_bounded() -> None:
    config = DirectDescriptionV9CarrierComposeConfigV1(
        pair_start=448,
        pair_count=64,
        v6_receipt_path="v6.json",
        v6_receipt_sha256="1" * 64,
        predictor_archive_path="predictor.zip",
        predictor_archive_sha256="2" * 64,
        upstream_root="/absolute/upstream",
        scorer_threads=1,
    )
    assert config.execution_allowed is False
    assert config.score_claim is False
    with pytest.raises(ValueError, match="bridge windows"):
        DirectDescriptionV9CarrierComposeConfigV1(
            pair_start=344,
            pair_count=64,
            v6_receipt_path="v6.json",
            v6_receipt_sha256="1" * 64,
            predictor_archive_path="predictor.zip",
            predictor_archive_sha256="2" * 64,
            upstream_root="/absolute/upstream",
            scorer_threads=1,
        )


def test_v9_archive_roundtrip_unique_homes_and_six_nonduplicated_strata() -> None:
    archive, homes = compile_carrier_compose_archive(_predictor())
    members, replay_homes = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    rows = recursive_carrier_byte_rows(archive)
    assert set(members) == {"manifest.json", "predictor.zip"}
    assert homes == replay_homes
    assert sum(row["zip_home_bytes"] for row in homes) == len(archive)
    assert [row["stratum"] for row in rows[:6]] == [
        "Road",
        "Lane",
        "Undrivable",
        "Movable",
        "MyCar",
        "xi/Pose6",
    ]
    assert all(row["nested_unique_home_bytes"] > 0 for row in rows[:6])
    assert receiver.custody["nested_pose6_owner_reused"] is True
    assert receiver.custody["pixel_coordinate_or_rgb_patch_present"] is False
    assert np.array_equal(
        receiver.render_pairs((0, 63)), receive_carrier_compose_archive(archive).render_pairs((0, 63))
    )


def test_g2cs1_refinement_is_consumed_before_region_coherent_rasterization() -> None:
    predictor = _predictor()
    base = receive_carrier_compose_archive(compile_carrier_compose_archive(predictor)[0])
    symbol = LaneCoefficientDelta(pair_index=0, line_index=0, coefficient_index=3, coefficient_delta=0.5)
    archive, _homes = compile_carrier_compose_archive(predictor, (symbol,))
    corrected = receive_carrier_compose_archive(archive)
    assert corrected.symbols == (symbol,)
    assert not np.array_equal(base.render_pairs((0,)), corrected.render_pairs((0,)))
    assert corrected.custody["region_coherent_chart_rerasterization"] is True
    rows = recursive_carrier_byte_rows(archive)
    assert rows[-1]["nested_unique_home_bytes"] > 0


def test_v9_refuses_pixel_or_absent_chart_addresses_and_sampled_mutations() -> None:
    predictor = _predictor()
    with pytest.raises(DirectDescriptionError, match="source window"):
        compile_carrier_compose_archive(
            predictor,
            (LaneCoefficientDelta(64, 0, 3, 0.5),),
        )
    with pytest.raises(DirectDescriptionError, match="centerline"):
        receive_carrier_compose_archive(
            compile_carrier_compose_archive(
                predictor,
                (LaneCoefficientDelta(0, 0, 4, 0.5),),
            )[0]
        )
    archive = compile_carrier_compose_archive(predictor)[0]
    proof = prove_carrier_archive_fail_closed(archive)
    assert proof["all_samples_refused_or_changed_decode"] is True
