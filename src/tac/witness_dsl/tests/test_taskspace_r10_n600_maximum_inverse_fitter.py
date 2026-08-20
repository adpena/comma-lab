from __future__ import annotations

import dataclasses
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl import taskspace_r10_n600_maximum_inverse_fitter as fitter_module
from tac.witness_dsl.taskspace_r10_feature_texture_relay import (
    decode_r10_packet,
    pair_population_sha256,
    parse_r10_packet,
    realization_sha256,
    serialize_r10_packet,
)
from tac.witness_dsl.taskspace_r10_n600_maximum_inverse_fitter import (
    G32_SECTION_ORDER,
    R10BoundedInverseConfigV1,
    R10BoundedInverseError,
    R10VideoSpecificClaimV1,
    apply_generic_bounded_repair,
    audit_counted_state_manifest,
    compile_r10_bounded_inverse,
    result_receipt_dict,
    streaming_realization_sha256,
    strict_json_loads,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import G17PhysicalCodingGroupV1


def _fixture() -> tuple[np.ndarray, np.ndarray, R10BoundedInverseConfigV1, str]:
    rng = np.random.default_rng(310)
    base = rng.integers(20, 230, size=(2, 2, 16, 20, 3), dtype=np.uint8)
    source = base.copy()
    source[0, :, 4:11, 6:14] = np.clip(
        source[0, :, 4:11, 6:14].astype(np.int16) + np.array((6, 3, -2)),
        0,
        255,
    ).astype(np.uint8)
    source[1, :, 6:13, 4:12] = np.clip(
        source[1, :, 6:13, 4:12].astype(np.int16) + np.array((-3, 4, 7)),
        0,
        255,
    ).astype(np.uint8)
    config = R10BoundedInverseConfigV1(
        seed=17,
        pair_count=2,
        height=16,
        width=20,
        chunk_pairs=1,
        sample_stride=4,
        pitch_candidates_q20=(0,),
        xi_steps=(0.001,),
        texture_frequencies_q8=(64,),
        texture_phases_q10=(0, 512),
        xi_q_levels=128,
        support_radius_px=3,
        flow_translation_q8=(0,),
    )
    return base, source, config, "3" * 64


@pytest.fixture(scope="module")
def fitted():
    base, source, config, source_sha256 = _fixture()
    states: dict[str, dict[str, object]] = {}
    result = compile_r10_bounded_inverse(
        base,
        source,
        source_sha256=source_sha256,
        config=config,
        stage_callback=lambda stage, payload: states.__setitem__(stage, dict(payload)),
    )
    return base, source, config, source_sha256, result, states


def test_real_fitter_emits_all_nine_sections_and_executes_frozen_receiver(fitted) -> None:
    base, _source, config, source_sha256, result, _states = fitted
    packet = parse_r10_packet(result.packet_bytes)
    assert serialize_r10_packet(packet) == result.packet_bytes
    assert tuple(span.section for span in result.counted_state_manifest.physical_sections) == G32_SECTION_ORDER
    assert packet.n_pairs == config.pair_count
    assert packet.shooting_knots
    assert packet.pullback_polygons
    assert packet.stratified_flows
    population = pair_population_sha256(source_sha256, tuple(range(config.pair_count)))
    replay = decode_r10_packet(
        result.packet_bytes,
        base,
        expected_source_sha256=source_sha256,
        expected_pair_population_sha256=population,
    )
    assert replay.dtype == np.uint8
    assert replay.shape == base.shape
    assert np.array_equal(replay, result.bounded_output)
    assert np.count_nonzero(replay != base) > 0


def test_packet_has_exact_counted_coverage_and_real_g23_zip_custody(fitted) -> None:
    _base, _source, _config, _source_sha256, result, _states = fitted
    audit_counted_state_manifest(result.counted_state_manifest, result.packet_bytes)
    adapter = result.physical_group_adapter
    assert type(adapter.physical_group) is G17PhysicalCodingGroupV1
    assert adapter.physical_group.exact_member_bytes == result.packet_bytes
    with zipfile.ZipFile(io.BytesIO(adapter.physical_group.exact_archive_bytes)) as archive:
        assert archive.read("r10.packet") == result.packet_bytes
    for span in adapter.section_spans:
        start = span.archive_offset
        stop = start + span.byte_length
        assert (
            result.wrapper_zip_bytes[start:stop]
            == result.packet_bytes[span.packet_offset : span.packet_offset + span.byte_length]
        )
    assert {row["constraint"] for row in adapter.constraint_operand_spans} == {
        "AMPLITUDE",
        "FREQUENCY",
        "PHASE",
        "CONTRAST",
        "CHANNEL_ENERGY",
        "TEXTURE",
        "MULTIPLE_SHOOTING",
        "FROZEN_FEATURE",
    }
    assert adapter.continuation_equivalence_sha256 == result.continuation_equivalence.identity_sha256
    assert result.continuation_equivalence.base_realization_sha256 == realization_sha256(_base)


@pytest.mark.parametrize(
    "attack",
    ("uncounted", "drift", "hidden_threshold", "hidden_exception", "packaged_code"),
)
def test_hidden_code_as_data_attacks_fail_closed(fitted, attack: str) -> None:
    _base, _source, _config, _source_sha256, result, _states = fitted
    manifest = result.counted_state_manifest
    if attack == "uncounted":
        bad_claim = dataclasses.replace(manifest.claims[0], byte_length=0)
        bad = dataclasses.replace(manifest, claims=(bad_claim, *manifest.claims[1:]))
    elif attack == "drift":
        bad_claim = dataclasses.replace(manifest.claims[1], span_sha256="0" * 64)
        bad = dataclasses.replace(manifest, claims=(manifest.claims[0], bad_claim, *manifest.claims[2:]))
    elif attack == "hidden_threshold":
        bad = dataclasses.replace(
            manifest,
            generic_decoder_mechanisms=(*manifest.generic_decoder_mechanisms, "source_selected_threshold=11"),
        )
    elif attack == "hidden_exception":
        claim = R10VideoSpecificClaimV1(
            claim_id="per_pair_exception",
            section="PAIR_INDEX",
            packet_offset=0,
            byte_length=1,
            span_sha256="0" * 64,
            active=True,
            video_specific=False,
            derivation="generic",
        )
        bad = dataclasses.replace(manifest, claims=(*manifest.claims, claim))
    else:
        bad = dataclasses.replace(
            manifest,
            generic_decoder_mechanisms=(*manifest.generic_decoder_mechanisms, "packaged video_specific weights"),
        )
    with pytest.raises(R10BoundedInverseError):
        audit_counted_state_manifest(bad, result.packet_bytes)


def test_repair_abi_is_bounded_inactive_and_does_not_invent_a_byte_win(fitted) -> None:
    _base, _source, _config, _source_sha256, result, _states = fitted
    repaired, receipt = apply_generic_bounded_repair(result.bounded_output, result.packet_bytes, 0)
    assert np.array_equal(repaired, result.bounded_output)
    assert receipt["active"] is False
    assert receipt["changed_values"] == 0
    with pytest.raises(R10BoundedInverseError):
        apply_generic_bounded_repair(result.bounded_output, result.packet_bytes, 1)
    assert all(row.verdict == "STORED_COUNTED_STATE_REQUIRED" for row in result.correction_placement)
    assert all(not row.exact_generic_regenerator_available for row in result.correction_placement)


def test_resume_state_skips_fits_and_reproduces_packet_exactly(
    fitted,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, source, config, source_sha256, result, states = fitted
    replayed_states: dict[str, dict[str, object]] = {}
    for name in (
        "_fit_base_feature_record",
        "_fit_texture_record",
        "_fit_flow_record",
        "_refit_joint_flow_record",
    ):
        monkeypatch.setattr(
            fitter_module,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(f"full-stage resume re-entered {_name}"),
        )
    resumed = compile_r10_bounded_inverse(
        base,
        source,
        source_sha256=source_sha256,
        config=config,
        resume_stages=states,
        stage_callback=lambda stage, payload: replayed_states.__setitem__(stage, dict(payload)),
    )
    assert resumed.packet_bytes == result.packet_bytes
    assert resumed.wrapper_zip_bytes == result.wrapper_zip_bytes
    assert replayed_states == states
    joint = states["110_joint_refit"]
    assert joint["interaction_passes"] == 1
    assert joint["trace"]["corner_selection"]["selected"] in {
        "INITIAL_SEQUENTIAL",
        "POST_INTERACTION_REFIT",
    }
    objectives = [row["objective"] for row in joint["trace"]["corner_selection"]["corners"]]
    assert joint["trace"]["corner_selection"]["selected_objective"] == min(objectives)


def test_geometry_only_checkpoint_legally_restarts_xip2(fitted) -> None:
    base, source, config, source_sha256, result, states = fitted
    resumed = compile_r10_bounded_inverse(
        base,
        source,
        source_sha256=source_sha256,
        config=config,
        resume_stages={"030_geometry": states["030_geometry"]},
    )
    assert resumed.packet_bytes == result.packet_bytes
    assert np.array_equal(resumed.fitted_xi, result.fitted_xi)


def test_complete_fit_range_prefix_resumes_without_resolving_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, source, config, source_sha256 = _fixture()
    states: dict[str, dict[str, object]] = {}
    ranges: dict[str, list[dict[str, object]]] = {}

    def keep_range(stage, first_pair, stop_pair, payload):
        ranges.setdefault(stage, []).append(
            {"first_pair": first_pair, "stop_pair": stop_pair, "payload": dict(payload)}
        )

    original = compile_r10_bounded_inverse(
        base,
        source,
        source_sha256=source_sha256,
        config=config,
        stage_callback=lambda stage, payload: states.__setitem__(stage, dict(payload)),
        range_callback=keep_range,
    )
    for name in (
        "_fit_base_feature_record",
        "_fit_texture_record",
        "_fit_flow_record",
        "_refit_joint_flow_record",
    ):
        monkeypatch.setattr(
            fitter_module,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(f"range resume re-entered {_name}"),
        )
    resumed = compile_r10_bounded_inverse(
        base,
        source,
        source_sha256=source_sha256,
        config=config,
        resume_stages={"030_geometry": states["030_geometry"]},
        resume_ranges=ranges,
    )
    assert resumed.packet_bytes == original.packet_bytes


def test_xip2_uses_native_geometry_then_samples_and_full_resolution_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 8, 10
    base = np.zeros((1, 2, height, width, 3), dtype=np.uint8)
    source = base.copy()
    source[0, 0, 0, 0] = 10
    config = R10BoundedInverseConfigV1(
        seed=0,
        pair_count=1,
        height=height,
        width=width,
        sample_stride=2,
        pitch_candidates_q20=(0,),
        xi_steps=(0.1,),
        texture_frequencies_q8=(64,),
        texture_phases_q10=(0,),
        support_radius_px=2,
        flow_translation_q8=(0,),
    )
    geometry_calls: list[tuple[int, int]] = []

    class FakeGeometry:
        @classmethod
        def eon(cls, *, native_hw, pitch):
            assert pitch == 0.0
            geometry_calls.append(tuple(native_hw))
            return cls()

    def fake_warp(frame, xi, _geometry):
        out = np.array(frame, copy=True)
        if float(xi[0]) > 0.0:
            out[0, 0] = 10
            out[1::2, 1::2] = 100
        return out

    monkeypatch.setattr(fitter_module, "GroundHomographyGeom", FakeGeometry)
    monkeypatch.setattr(fitter_module, "warp_frame0_uint8_numpy", fake_warp)
    _pitch, decoded, trace = fitter_module._fit_pitch_and_xi(base, source, config)
    assert geometry_calls
    assert set(geometry_calls) == {(height, width)}
    assert np.array_equal(decoded, np.zeros((1, 6), dtype=np.float64))
    assert trace["rejected_pairs_full_resolution"] == [0]


def test_streaming_realization_hash_matches_frozen_r10_hash(tmp_path: Path) -> None:
    base, _source, config, _source_sha256 = _fixture()
    path = tmp_path / "pairs.raw"
    path.write_bytes(base.tobytes(order="C"))
    assert streaming_realization_sha256(
        path,
        pair_count=config.pair_count,
        height=config.height,
        width=config.width,
        read_bytes=17,
    ) == realization_sha256(base)


def test_receipt_keeps_every_authority_field_null(fitted) -> None:
    _base, _source, _config, _source_sha256, result, _states = fitted
    receipt = result_receipt_dict(result)
    assert receipt["authority"]["research_only"] is True
    assert receipt["authority"]["d_seg"] is None
    assert receipt["authority"]["d_pose"] is None
    assert receipt["authority"]["complete_candidate_zip_bytes"] is None
    assert receipt["authority"]["contest_score"] is None
    assert receipt["pointer_delta"] is False
    assert receipt["continuation_equivalence"]["identity_sha256"] == (result.continuation_equivalence.identity_sha256)
    assert receipt["continuation_equivalence"]["bounded_output_realization_sha256"] == realization_sha256(
        result.bounded_output
    )
    assert result.residual_inventory.learned_residual_admitted is False
    assert result.residual_inventory.bounded_action_family_exhausted is False
    assert result.residual_inventory.unenumerated_action_families
    assert result.residual_inventory.full_n600_receiver_measured_exhaustion is False
    assert all(row.objective_before is None and row.wall_seconds is None for row in result.section_fits)
    receiver_receipt = receipt["bounded_decode_receipt"]
    assert "bounded_inverse_family_hook" in receiver_receipt
    encoded_receiver_receipt = fitter_module._canonical_json(receiver_receipt).lower()
    assert b"maximum_inverse" not in encoded_receiver_receipt
    assert b"unbounded_encode" not in encoded_receiver_receipt


def test_config_and_strict_json_gates() -> None:
    with pytest.raises(R10BoundedInverseError, match="n600 requires"):
        R10BoundedInverseConfigV1(seed=0, pair_count=600, height=874, width=1164)
    with pytest.raises(R10BoundedInverseError, match="repair inactive"):
        R10BoundedInverseConfigV1(
            seed=0,
            pair_count=1,
            height=16,
            width=20,
            support_radius_px=3,
            decoder_iteration_budget=1,
        )
    with pytest.raises(R10BoundedInverseError, match="duplicate JSON key"):
        strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(R10BoundedInverseError, match="non-finite"):
        strict_json_loads('{"a":NaN}')


def test_n600_compile_requires_live_governance_callback() -> None:
    config = R10BoundedInverseConfigV1(
        seed=0,
        pair_count=600,
        height=8,
        width=8,
        sample_stride=4,
        pitch_candidates_q20=(0,),
        xi_steps=(0.1,),
        texture_frequencies_q8=(64,),
        texture_phases_q10=(0,),
        support_radius_px=2,
        flow_translation_q8=(0,),
        execute_reviewed=True,
    )
    base = np.zeros((600, 2, 8, 8, 3), dtype=np.uint8)
    source = np.ones_like(base)
    with pytest.raises(R10BoundedInverseError, match="live canonical governance callback"):
        compile_r10_bounded_inverse(
            base,
            source,
            source_sha256="4" * 64,
            config=config,
            execute_bounded_decode=False,
        )
