# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_entropy_priced_member import (
    TOLERANCE_LADDER,
    DirectDescriptionEntropyCandidateCheckpointV1,
    DirectDescriptionEntropyPricedMemberConfigV1,
    DirectDescriptionEntropyPricedMemberProgramV1,
    build_entropy_candidate_z,
    run_entropy_candidate_stages,
    run_entropy_rung_stages,
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
