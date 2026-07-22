# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import lzma
import struct
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_entropy_priced_member import (
    TOLERANCE_LADDER,
    DirectDescriptionDsegBridgeAmortizeConfigV1,
    DirectDescriptionDsegBridgeAmortizeProgramV1,
    DirectDescriptionEntropyCandidateCheckpointV1,
    DirectDescriptionEntropyPricedMemberConfigV1,
    DirectDescriptionEntropyPricedMemberProgramV1,
    DirectDescriptionRouteFixComposeConfigV1,
    DirectDescriptionStratumStructuredMemberConfigV1,
    StructuredS4SourcesV1,
    _amortize_chart_z,
    _amortize_structured_sources,
    _decode_site_records,
    _derived_membership_proxy,
    _encode_site_records,
    _xi_pose6_keyframes,
    build_entropy_candidate_z,
    compile_composed_structured_member_archive,
    compile_structured_member_archive,
    parse_structured_member_archive,
    receive_structured_member_archive,
    run_entropy_candidate_stages,
    run_entropy_rung_stages,
    run_structured_candidate_stages,
    select_role_paint_values,
)
from tac.optimization.direct_description_entropy_streams import (
    CODER_AQC1,
    CODER_HUFFMAN_RANK16,
    STREAM_ORDER,
    _decode_generic,
    _encode_aqc1,
    _encode_huffman_rank16,
    _transform_candidates,
    compile_entropy_chart_archive,
    parse_entropy_chart_archive,
    parse_entropy_stream,
    prove_entropy_home_fail_closed,
    receive_entropy_chart_archive,
)
from tac.optimization.direct_description_measurement_ladder import (
    _ANCHOR_RECORD,
    _GRADIENT_RECORD,
    _POSE_RECORD,
    _RESIDUAL_RECORD,
    MEMBER_BY_STREAM,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
    compile_chart_archive,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError


def _fixture_z(n_pairs: int = 2) -> DirectDescriptionChartZV1:
    bodies = {name: bytearray() for name in STREAM_ORDER}
    for pair_id in range(n_pairs):
        for plane_id in range(2):
            bodies["global_chart_anchors"].extend(
                _ANCHOR_RECORD.pack(pair_id, plane_id, 96 + pair_id, 112 + pair_id, 128 + pair_id)
            )
            bodies["axial_chart_gradients"].extend(
                _GRADIENT_RECORD.pack(
                    pair_id,
                    plane_id,
                    pair_id,
                    -pair_id,
                    plane_id,
                    pair_id + plane_id,
                    0,
                    -plane_id,
                )
            )
            for stratum, stream_name in enumerate(STREAM_ORDER[2:5]):
                for chart_id in range(stratum * 64, (stratum + 1) * 64):
                    residual = (pair_id + stratum, (chart_id % 5) - 2, -pair_id)
                    bodies[stream_name].extend(_RESIDUAL_RECORD.pack(pair_id, plane_id, chart_id, *residual))
        bodies["pose6_pair_codes"].extend(_POSE_RECORD.pack(pair_id, *(20 + pair_id + channel for channel in range(6))))
    return DirectDescriptionChartZV1(
        n_pairs=n_pairs,
        **{name: CountedChartStreamV1(payload=bytes(bodies[name])) for name in STREAM_ORDER},
    )


def _structured_fixture_z() -> DirectDescriptionChartZV1:
    bodies = {name: bytearray() for name in STREAM_ORDER}
    for pair_id in range(64):
        for plane_id in range(2):
            bodies["global_chart_anchors"].extend(_ANCHOR_RECORD.pack(pair_id, plane_id, 96, 112, 128))
            bodies["axial_chart_gradients"].extend(_GRADIENT_RECORD.pack(pair_id, plane_id, 0, 0, 0, 0, 0, 0))
            for stratum, stream_name in enumerate(STREAM_ORDER[2:5]):
                for chart_id in range(stratum * 64, (stratum + 1) * 64):
                    bodies[stream_name].extend(_RESIDUAL_RECORD.pack(pair_id, plane_id, chart_id, 0, 0, 0))
        bodies["pose6_pair_codes"].extend(_POSE_RECORD.pack(pair_id, *(20 + channel for channel in range(6))))
    return DirectDescriptionChartZV1(
        n_pairs=64,
        **{name: CountedChartStreamV1(payload=bytes(bodies[name])) for name in STREAM_ORDER},
    )


def _config() -> DirectDescriptionEntropyPricedMemberConfigV1:
    return DirectDescriptionEntropyPricedMemberConfigV1(
        target_receipt_path="target.json",
        target_receipt_sha256="0" * 64,
        upstream_root="/absolute/upstream",
        scorer_threads=1,
    )


def _membership(receiver: object) -> dict[str, object]:
    n_pairs = receiver.z.n_pairs  # type: ignore[attr-defined]
    return {
        "same_c1_argmax_cell_fraction": "1.000000000000",
        "per_pair": [{"pair_id": pair_id} for pair_id in range(n_pairs)],
        "strata": {
            "overall": {"all": {"argmax_cell_escape_fraction": "0.000000000000"}},
            "target_class": {"Road": {"argmax_cell_escape_fraction": "0.000000000000"}},
        },
    }


def _zero_lane_payload() -> bytes:
    header = {
        "cx": 256.0,
        "dash_forward_max_m": 50.0,
        "dash_gate": True,
        "rd": {"K": 0, "base_steps": [1.0] * 11, "d_slot": 11, "n_pairs": 600},
        "softness": 1.0,
        "v_h": 150.0,
    }
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    raw = b"LBND2\0" + struct.pack("<I", len(raw_header)) + raw_header + struct.pack("<I", 0)
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]
    return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filters)


def _structured_sources() -> StructuredS4SourcesV1:
    empty = tuple(() for _ in range(64))
    classes: list[tuple[tuple[np.ndarray, ...], ...]] = [empty for _ in range(5)]
    for class_id in range(5):
        rows = list(empty)
        rows[0] = (np.asarray([class_id * 8 + 0, class_id * 8 + 1, class_id * 8 + 2], dtype=np.int64),)
        classes[class_id] = tuple(rows)
    static_road = np.zeros((384, 512), dtype=bool)
    static_road[100:104, 200:204] = True
    hood = np.zeros((384, 512), dtype=bool)
    hood[360:, 220:292] = True
    return StructuredS4SourcesV1(
        pair_count=64,
        palette=np.asarray(
            [[153, 255, 51], [51, 255, 204], [0, 153, 0], [102, 204, 51], [0, 255, 153]],
            dtype=np.uint8,
        ),
        camera={"height_m": 1.2, "fx_scorer": 400.3, "fy_scorer": 399.5},
        static_masks={
            "Road": static_road,
            "Undrivable": np.zeros((384, 512), dtype=bool),
            "MyCar": hood,
        },
        lane_encoded=_zero_lane_payload(),
        lane_lines=tuple(() for _ in range(600)),
        lane_header={
            "cx": 256.0,
            "dash_forward_max_m": 50.0,
            "dash_gate": True,
            "rd": {"K": 0, "base_steps": [1.0] * 11, "d_slot": 11, "n_pairs": 600},
            "softness": 1.0,
            "v_h": 150.0,
        },
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


def _structured_config() -> DirectDescriptionStratumStructuredMemberConfigV1:
    return DirectDescriptionStratumStructuredMemberConfigV1(
        target_receipt_path="target.json",
        target_receipt_sha256="0" * 64,
        upstream_root="/absolute/upstream",
        scorer_threads=1,
        s4_container_path="/absolute/s4/0.bin",
        s4_container_sha256="1" * 64,
        s4_runtime_path="/absolute/s4/runtime/inflate.py",
        s4_runtime_sha256="2" * 64,
    )


def _structured_membership(receiver: object) -> dict[str, object]:
    n_pairs = receiver.z.n_pairs  # type: ignore[attr-defined]
    classes = {
        name: {
            "same_c1_argmax_cell_fraction": "0.500000000000" if name == "MyCar" else "0.250000000000",
            "argmax_cell_escape_fraction": "0.500000000000",
        }
        for name in ("Road", "Lane", "Undrivable", "Movable", "MyCar")
    }
    return {
        "same_c1_argmax_cell_fraction": "0.400000000000",
        "per_pair": [{"pair_id": pair_id} for pair_id in range(n_pairs)],
        "strata": {
            "overall": {"all": {"argmax_cell_escape_fraction": "0.600000000000"}},
            "target_class": classes,
        },
    }


def test_entropy_archive_roundtrips_exact_semantic_payloads() -> None:
    z = _fixture_z()
    compiled = compile_entropy_chart_archive(z)
    parsed = parse_entropy_chart_archive(compiled.archive)
    assert parsed.archive == compiled.archive
    assert parsed.z == z


def test_entropy_archive_is_deterministic_and_smaller_than_fixed_fixture() -> None:
    z = _fixture_z(4)
    first = compile_entropy_chart_archive(z)
    second = compile_entropy_chart_archive(z)
    assert first.archive == second.archive
    assert len(first.archive) < len(compile_chart_archive(z).archive)


def test_every_stream_reports_a_unique_home_and_measured_tournament() -> None:
    rows = compile_entropy_chart_archive(_fixture_z()).stream_byte_rows()
    assert [row["stream"] for row in rows] == list(STREAM_ORDER)
    assert all(row["unique_final_zip_home_bytes"] > row["coded_payload_bytes"] for row in rows)
    assert all(len(row["candidate_rows"]) >= 5 for row in rows)
    assert all(row["exact_semantic_roundtrip"] for row in rows)


@pytest.mark.parametrize("stream_name", STREAM_ORDER)
def test_all_transform_variants_reconstruct_semantic_stream(stream_name: str) -> None:
    z = _fixture_z()
    payload = getattr(z, stream_name).payload
    candidates = _transform_candidates(stream_name, payload, z.n_pairs)
    assert candidates
    from tac.optimization.direct_description_entropy_streams import _decode_transform

    assert all(
        _decode_transform(stream_name, z.n_pairs, item.transform_id, item.canonical_bytes()) == payload
        for item in candidates
    )


def test_aqc1_adapter_roundtrips_and_rejects_trailing_bytes() -> None:
    payload = bytes([0, 1, 1, 2, 2, 2]) * 32
    encoded = _encode_aqc1(payload)
    assert _decode_generic(CODER_AQC1, encoded, len(payload)) == payload
    with pytest.raises(DirectDescriptionError):
        _decode_generic(CODER_AQC1, encoded + b"\0", len(payload))


def test_ranked_huffman_adapter_roundtrips_and_rejects_nonzero_padding() -> None:
    payload = bytes([2, 3, 5, 7, 11, 13]) * 11
    encoded = _encode_huffman_rank16(payload)
    assert _decode_generic(CODER_HUFFMAN_RANK16, encoded, len(payload)) == payload
    mutated = bytearray(encoded)
    mutated[-1] |= 1
    with pytest.raises(DirectDescriptionError):
        _decode_generic(CODER_HUFFMAN_RANK16, bytes(mutated), len(payload))


def test_ranked_huffman_refuses_more_than_sixteen_symbols() -> None:
    with pytest.raises(DirectDescriptionError, match=r"1\.\.16"):
        _encode_huffman_rank16(bytes(range(17)))


def test_entropy_stream_rejects_truncation() -> None:
    compiled = compile_entropy_chart_archive(_fixture_z())
    member = MEMBER_BY_STREAM["pose6_pair_codes"]
    with pytest.raises(DirectDescriptionError):
        parse_entropy_stream("pose6_pair_codes", compiled.framed_members[member][:-1])


def test_entropy_archive_rejects_appended_bytes() -> None:
    archive = compile_entropy_chart_archive(_fixture_z()).archive
    with pytest.raises(DirectDescriptionError):
        parse_entropy_chart_archive(archive + b"\0")


def test_entropy_receiver_preserves_pose_and_render_geometry() -> None:
    z = _fixture_z()
    receiver = receive_entropy_chart_archive(compile_entropy_chart_archive(z).archive)
    assert receiver.render_pairs((0,)).shape == (1, 2, 384, 512, 3)
    assert receiver.pose6_codes[0].tolist() == [20, 21, 22, 23, 24, 25]
    assert receiver.custody["all_coded_sections_consumed_exactly"] is True


def test_entropy_home_mutations_are_effective_or_refused() -> None:
    proof = prove_entropy_home_fail_closed(_fixture_z())
    assert proof["sampled_positions"] > 12
    assert proof["all_samples_effective_or_refused"] is True


def test_program_compiles_only_the_typed_local_argv() -> None:
    program = DirectDescriptionEntropyPricedMemberProgramV1(config_path="cfg.json", output_directory="out")
    assert program.compile_consumer_argv() == (
        "/usr/bin/env",
        "python3",
        "tools/run_direct_description_entropy_priced_member.py",
        "--config",
        "cfg.json",
        "--output-dir",
        "out",
        "--execution-allowed",
        "false",
    )


def test_config_rejects_a_different_tolerance_ladder() -> None:
    with pytest.raises(ValueError, match="tolerance_ladder"):
        DirectDescriptionEntropyPricedMemberConfigV1(
            target_receipt_path="target.json",
            target_receipt_sha256="0" * 64,
            upstream_root="/absolute/upstream",
            scorer_threads=1,
            tolerance_ladder=("0.000000",),
        )


def test_candidate_subset_changes_only_named_residual_streams() -> None:
    z = _fixture_z()
    changed = build_entropy_candidate_z(z, 0b101)
    assert changed.low_variation_chart_residuals != z.low_variation_chart_residuals
    assert changed.mid_variation_chart_residuals == z.mid_variation_chart_residuals
    assert changed.high_variation_chart_residuals != z.high_variation_chart_residuals
    assert changed.pose6_pair_codes == z.pose6_pair_codes


def test_candidate_checkpoint_roundtrip_and_hash_refusal() -> None:
    config = _config()
    argv = ("python3", "tool.py")
    checkpoint = DirectDescriptionEntropyCandidateCheckpointV1(
        config=config.model_dump(mode="json", by_alias=True),
        config_sha256=config.typed_config_hash(),
        dsl_compile_hash=config.dsl_compile_hash(),
        semantic_argv=argv,
        semantic_argv_sha256=__import__("hashlib").sha256("\0".join(argv).encode()).hexdigest(),
        completed_subset_mask=0,
        next_subset_mask=1,
        candidates=({"subset_mask": 0},),
    )
    payload = checkpoint.to_bytes()
    assert DirectDescriptionEntropyCandidateCheckpointV1.from_bytes(payload) == checkpoint
    mutated = bytearray(payload)
    mutated[-2] ^= 1
    with pytest.raises((DirectDescriptionError, ValueError)):
        DirectDescriptionEntropyCandidateCheckpointV1.from_bytes(bytes(mutated))


def test_candidate_and_rung_stages_resume_without_losing_checkpoints(tmp_path: Path) -> None:
    z = _fixture_z(64)
    pose = np.zeros((600, 6), dtype=np.uint8)
    pose[:64] = receive_entropy_chart_archive(compile_entropy_chart_archive(z).archive).pose6_codes
    config = _config()
    argv = ("python3", "tool.py")
    candidate_dir = tmp_path / "candidates"
    partial = run_entropy_candidate_stages(
        config,
        baseline_z=z,
        target_pose_codes=pose,
        membership_measure=_membership,
        semantic_argv=argv,
        checkpoint_directory=candidate_dir,
        stop_after_subset_mask=2,
    )
    assert not partial.complete
    resumed = run_entropy_candidate_stages(
        config,
        baseline_z=z,
        target_pose_codes=pose,
        membership_measure=_membership,
        semantic_argv=argv,
        checkpoint_directory=candidate_dir,
        resume_from=partial.checkpoint_paths[-1],
    )
    assert resumed.complete
    assert len(tuple(candidate_dir.iterdir())) == 8
    rung_dir = tmp_path / "rungs"
    rung_partial = run_entropy_rung_stages(
        config,
        baseline_z=z,
        candidates=resumed.candidates,
        semantic_argv=argv,
        checkpoint_directory=rung_dir,
        stop_after_rung_index=1,
    )
    assert not rung_partial.complete
    rung_resumed = run_entropy_rung_stages(
        config,
        baseline_z=z,
        candidates=resumed.candidates,
        semantic_argv=argv,
        checkpoint_directory=rung_dir,
        resume_from=rung_partial.checkpoint_paths[-1],
    )
    assert rung_resumed.complete
    assert len(tuple(rung_dir.iterdir())) == len(TOLERANCE_LADDER)
    minimum = min(int(row["archive_bytes"]) for row in resumed.candidates)
    assert all(row["selected_archive_bytes"] == minimum for row in rung_resumed.curve)
    assert all(row["rung_feasible"] for row in rung_resumed.curve)


def test_structured_site_stream_roundtrips_and_rejects_trailing_bytes() -> None:
    rows = [() for _ in range(4)]
    rows[1] = (np.asarray([3, 8, 21], dtype=np.int64),)
    payload = _encode_site_records(rows, class_id=0, source_id=1, coder="lzma1_raw_1MiB")
    decoded = _decode_site_records(payload, expected_class=0, expected_source=1)
    assert decoded[1][0].tolist() == [3, 8, 21]
    with pytest.raises(DirectDescriptionError):
        _decode_site_records(payload + b"x", expected_class=0, expected_source=1)


@pytest.mark.parametrize("role", ("baseline", "Road", "Lane", "MyCar", "UndrivableBoundary", "Movable"))
def test_structured_archive_roundtrips_and_receiver_preserves_pose(role: str) -> None:
    z = _structured_fixture_z()
    baseline = compile_entropy_chart_archive(z).archive
    first, homes = compile_structured_member_archive(baseline, _structured_sources(), role)
    second, _ = compile_structured_member_archive(baseline, _structured_sources(), role)
    assert first == second
    members, parsed_homes = parse_structured_member_archive(first)
    assert members["chart.zip"] == baseline
    assert sum(row["zip_home_bytes"] for row in homes) == len(first)
    assert homes == parsed_homes
    receiver = receive_structured_member_archive(first)
    assert receiver.pose6_codes[0].tolist() == [20, 21, 22, 23, 24, 25]
    assert receiver.render_pairs((0,)).shape == (1, 2, 384, 512, 3)
    assert receiver.custody["all_archive_bytes_have_one_home"] is True


def test_structured_archive_refuses_appended_bytes() -> None:
    baseline = compile_entropy_chart_archive(_structured_fixture_z()).archive
    archive, _ = compile_structured_member_archive(baseline, _structured_sources(), "MyCar")
    with pytest.raises(DirectDescriptionError, match="byte-canonical"):
        parse_structured_member_archive(archive + b"x")


def test_mycar_structured_receiver_changes_only_static_hood() -> None:
    z = _structured_fixture_z()
    baseline_archive = compile_entropy_chart_archive(z).archive
    sources = _structured_sources()
    archive, _ = compile_structured_member_archive(baseline_archive, sources, "MyCar")
    structured = receive_structured_member_archive(archive).render_pairs((0,))
    baseline = receive_entropy_chart_archive(baseline_archive).render_pairs((0,))
    mask = sources.static_masks["MyCar"]
    assert np.all(structured[0, 1, mask] == sources.palette[4])
    assert np.array_equal(structured[0, 1, ~mask], baseline[0, 1, ~mask])


def test_role_value_selector_maximizes_each_self_detected_role_without_fixed_indices() -> None:
    role_ids = {"Road": 3, "Lane": 4, "MyCar": 0, "UndrivableBoundary": 1, "Movable": 2}
    winning_candidates = {"Road": 3, "Lane": 5, "MyCar": 0, "UndrivableBoundary": 1, "Movable": 2}
    score_rows = {}
    for role_index, (role, target_class_id) in enumerate(role_ids.items()):
        score_rows[role] = [
            {
                "candidate_index": candidate_index,
                "target_class_id": target_class_id,
                "own_class_matches": 100
                + candidate_index
                + (1000 if candidate_index == winning_candidates[role] else 0),
                "rgb_u8": [candidate_index, role_index, target_class_id],
            }
            for candidate_index in range(6)
        ]
    selected = select_role_paint_values(role_ids, score_rows)
    for role in role_ids:
        assert selected[role]["candidate_index"] == winning_candidates[role]
        assert selected[role]["own_class_matches"] == max(row["own_class_matches"] for row in score_rows[role])


def test_composed_member_consumes_every_role_and_keeps_pose_in_one_receiver() -> None:
    z = _structured_fixture_z()
    baseline_archive = compile_entropy_chart_archive(z).archive
    sources = _structured_sources()
    archive, homes = compile_composed_structured_member_archive(
        baseline_archive,
        sources,
        pair_start=0,
    )
    receiver = receive_structured_member_archive(archive)
    assert [layer.role for layer in receiver.layers] == [
        "UndrivableBoundary",
        "Road",
        "Lane",
        "MyCar",
        "Movable",
    ]
    assert receiver.pose6_codes[0].tolist() == [20, 21, 22, 23, 24, 25]
    assert receiver.render_pairs((0,)).shape == (1, 2, 384, 512, 3)
    assert receiver.custody["all_five_roles_consumed"] is True
    assert sum(row["zip_home_bytes"] for row in homes) == len(archive)


def test_v5_config_requires_probe_inside_n64_or_n256_state_window() -> None:
    common = {
        "pair_start": 448,
        "pair_count": 64,
        "target_receipt_path": "target.json",
        "target_receipt_sha256": "0" * 64,
        "upstream_root": "/absolute/upstream",
        "scorer_threads": 1,
        "s4_container_path": "/absolute/s4/0.bin",
        "s4_container_sha256": "1" * 64,
        "s4_runtime_path": "/absolute/s4/runtime/inflate.py",
        "s4_runtime_sha256": "2" * 64,
    }
    config = DirectDescriptionRouteFixComposeConfigV1(**common)
    assert config.pair_start == 448 and config.pair_count == 64
    with pytest.raises(ValueError, match="n64 or n256"):
        DirectDescriptionRouteFixComposeConfigV1(**{**common, "pair_count": 128})
    with pytest.raises(ValueError, match="contained"):
        DirectDescriptionRouteFixComposeConfigV1(**{**common, "pair_start": 384})


def test_v6_config_and_program_are_typed_local_only() -> None:
    config = DirectDescriptionDsegBridgeAmortizeConfigV1(
        pair_start=448,
        pair_count=64,
        v5_receipt_path="v5.json",
        v5_receipt_sha256="1" * 64,
        v5_archive_path="v5.zip",
        v5_archive_sha256="2" * 64,
        scorer_threads=1,
    )
    assert config.candidate_modes == (
        "v5_exact",
        "fixed_ar1_hold24",
        "xi_pose6_ar1_hold24",
        "residual_zero_static_once",
    )
    program = DirectDescriptionDsegBridgeAmortizeProgramV1(config_path="v6.json", output_directory="out")
    assert program.compile_consumer_argv()[-2:] == ("--execution-allowed", "false")
    with pytest.raises(ValueError, match="n64 or n256"):
        DirectDescriptionDsegBridgeAmortizeConfigV1(
            pair_start=448,
            pair_count=128,
            v5_receipt_path="v5.json",
            v5_receipt_sha256="1" * 64,
            v5_archive_path="v5.zip",
            v5_archive_sha256="2" * 64,
            scorer_threads=1,
        )


def test_v6_key_hold_preserves_pose_and_rewrites_pair_identity() -> None:
    baseline = _fixture_z(64)
    held = _amortize_chart_z(baseline, keyframes=(0, 24, 48))
    assert held.pose6_pair_codes == baseline.pose6_pair_codes
    original = receive_entropy_chart_archive(compile_entropy_chart_archive(baseline).archive)
    receiver = receive_entropy_chart_archive(compile_entropy_chart_archive(held).archive)
    assert np.array_equal(receiver.anchors[23], original.anchors[0])
    assert np.array_equal(receiver.gradients[47], original.gradients[24])
    assert np.array_equal(receiver.residuals[63], original.residuals[48])
    assert np.array_equal(receiver.pose6_codes, original.pose6_codes)


def test_v6_xi_schedule_is_counted_pose_derived_and_gap_bounded() -> None:
    receiver = receive_entropy_chart_archive(compile_entropy_chart_archive(_fixture_z(64)).archive)
    first, receipt = _xi_pose6_keyframes(receiver.pose6_codes, max_gap=24)
    second, _ = _xi_pose6_keyframes(receiver.pose6_codes, max_gap=24)
    assert first == second
    assert first[0] == 0
    assert max(right - left for left, right in zip(first, (*first[1:], 64), strict=True)) <= 24
    assert receipt["unmeasured_motion_threshold_invented"] is False


def test_v6_membership_proxy_is_an_explicit_triangle_bound() -> None:
    proxy = _derived_membership_proxy("0.044353087743", measured_control="0.955627997716")
    assert proxy["status"] == "MEASURED_CONTROL_PLUS_DERIVED_BOUND"
    assert Decimal(proxy["same_c1_argmax_cell_fraction_lower"]) <= Decimal("0.955627997716")
    assert Decimal(proxy["same_c1_argmax_cell_fraction_upper"]) >= Decimal("0.955627997716")
    assert proxy["score_claim"] is False


def test_v6_structured_hold_reuses_keyed_events_but_keeps_static_once() -> None:
    sources = _structured_sources()
    held = _amortize_structured_sources(
        sources,
        pair_start=0,
        pair_count=64,
        keyframes=(0, 24, 48),
    )
    assert held.static_masks is sources.static_masks
    assert held.lane_encoded is sources.lane_encoded
    assert np.array_equal(held.events[0][23][0], sources.events[0][0][0])
    assert held.events[0][24] == sources.events[0][24]


def test_structured_candidate_stages_resume_and_preserve_every_archive(tmp_path: Path) -> None:
    z = _structured_fixture_z()
    pose = np.zeros((600, 6), dtype=np.uint8)
    pose[:64] = receive_entropy_chart_archive(compile_entropy_chart_archive(z).archive).pose6_codes
    config = _structured_config()
    partial = run_structured_candidate_stages(
        config,
        baseline_archive=compile_entropy_chart_archive(z).archive,
        sources=_structured_sources(),
        target_pose_codes=pose,
        membership_measure=_structured_membership,
        semantic_argv=("python3", "tool.py"),
        output_directory=tmp_path,
        stop_after_candidate_index=2,
    )
    assert not partial.complete
    resumed = run_structured_candidate_stages(
        config,
        baseline_archive=compile_entropy_chart_archive(z).archive,
        sources=_structured_sources(),
        target_pose_codes=pose,
        membership_measure=_structured_membership,
        semantic_argv=("python3", "tool.py"),
        output_directory=tmp_path,
        resume_from=partial.checkpoint_paths[-1],
    )
    assert resumed.complete
    assert [row["role"] for row in resumed.candidates] == [
        "baseline",
        "Road",
        "Lane",
        "MyCar",
        "UndrivableBoundary",
        "Movable",
    ]
    assert all(row["pose_completeness"] == "1.000000000000" for row in resumed.candidates)
    assert len(tuple((tmp_path / "candidate_receipts").iterdir())) == 6
