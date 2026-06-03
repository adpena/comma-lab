# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

from tac.analysis.nerv_receiver_closed_ladder_row_harvest import (
    build_nerv_receiver_closed_ladder_row_harvest,
)
from tac.analysis.nerv_receiver_closed_modelsize_ladder import (
    build_nerv_receiver_closed_modelsize_ladder,
)
from tac.analysis.nerv_trained_ladder_row_emitter import (
    SCHEMA,
    build_nerv_trained_ladder_row_payload,
)


def test_emitted_snerv_rows_feed_harvest_and_modelsize_ladder(tmp_path: Path) -> None:
    first = _ready_snerv_payload(
        tmp_path,
        name="tiny.zip",
        modelsize=0.04,
        fc_dim=16,
        d_seg=0.004,
        d_pose=0.002,
    )
    second = _ready_snerv_payload(
        tmp_path,
        name="small.zip",
        modelsize=0.08,
        fc_dim=24,
        d_seg=0.003,
        d_pose=0.002,
    )

    assert first["schema"] == SCHEMA
    assert first["status"] == "trained_ladder_row_ready"
    assert first["ready_for_receiver_closed_ladder_harvest"] is True
    assert first["score_claim"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    row = first["rows"][0]
    assert row["archive_bytes"] == len(b"archive-tiny.zip")
    assert row["archive_sha256"] == hashlib.sha256(b"archive-tiny.zip").hexdigest()
    assert row["receiver_proof_identity_bound"] is True
    assert row["byte_closed_receiver_proof"] is True
    assert row["receiver_proof_passed"] is True
    assert row["accepted"] is True

    harvest = build_nerv_receiver_closed_ladder_row_harvest(
        [first, second],
        carrier_id="snerv",
    )
    assert harvest["status"] == "receiver_closed_ladder_rows_ready"
    assert harvest["ladder_candidate_row_count"] == 2
    assert harvest["harvested_rows"][0]["axis_tag"].startswith("[contest-CPU")
    assert harvest["harvested_rows"][0]["source_axis_receiver_closed_authority"] is True

    ladder = build_nerv_receiver_closed_modelsize_ladder(
        harvest["harvested_rows"],
        carrier_id="snerv",
        repo_root=tmp_path,
    )
    assert ladder["status"] == "receiver_closed_modelsize_ladder_ready"
    assert ladder["receiver_closed_row_count"] == 2
    assert ladder["ready_for_exact_eval_dispatch"] is False


def test_archive_metadata_mismatch_blocks_row_acceptance(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"real-bytes")
    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "archive_bytes": 999,
            "archive_sha256": "a" * 64,
            "n_pairs": 600,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "official_controls": _snerv_controls(),
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof=_receiver_proof(tmp_path, "mismatch", archive=archive),
        scorer_eval={"avg_segnet_dist": 0.004, "avg_posenet_dist": 0.002},
        repo_root=tmp_path,
    )

    assert payload["status"] == "trained_ladder_row_blocked"
    assert "archive_bytes_metadata_mismatch" in payload["blockers"]
    assert "archive_sha256_metadata_mismatch" in payload["blockers"]
    assert payload["rows"][0]["accepted"] is False


def test_snerv_rows_with_false_mfu_hfr_remain_source_parity_blocked(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"receiver-closed")
    controls = _snerv_controls()
    controls["mfu_enabled"] = False
    controls["hfr_enabled"] = False

    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "n_pairs": 600,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "official_controls": controls,
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof=_receiver_proof(tmp_path, "false-mfu-hfr", archive=archive),
        scorer_eval={"avg_segnet_dist": 0.004, "avg_posenet_dist": 0.002},
        repo_root=tmp_path,
    )

    assert payload["status"] == "trained_ladder_row_blocked"
    assert "required_emission_field_false:official_controls.mfu_enabled" in payload[
        "blockers"
    ]
    assert "required_emission_field_false:official_controls.hfr_enabled" in payload[
        "blockers"
    ]
    assert payload["rows"][0]["accepted"] is False


def test_snerv_rows_with_false_source_faithful_stack_remain_blocked(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"receiver-closed")
    controls = _snerv_controls()
    controls["source_faithful_stack"] = False

    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "n_pairs": 600,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "official_controls": controls,
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof=_receiver_proof(tmp_path, "false-source", archive=archive),
        scorer_eval={"avg_segnet_dist": 0.004, "avg_posenet_dist": 0.002},
        repo_root=tmp_path,
    )

    assert payload["status"] == "trained_ladder_row_blocked"
    assert (
        "required_emission_field_false:official_controls.source_faithful_stack"
        in payload["blockers"]
    )
    assert payload["rows"][0]["accepted"] is False


def test_hinerv_alias_requires_official_controls_and_receiver_codec(tmp_path: Path) -> None:
    archive = tmp_path / "hinerv.zip"
    archive.write_bytes(b"hinerv")
    payload = build_nerv_trained_ladder_row_payload(
        family="hi_nerv",
        archive_path=archive,
        trainer_metadata={"n_pairs": 600, "fc_dim": 64},
        receiver_proof=_receiver_proof(tmp_path, "hinerv", archive=archive),
        scorer_eval={
            "axis_tag": "[contest-CPU unit-test receiver-closed]",
            "d_seg": 0.003,
            "d_pose": 0.002,
        },
        repo_root=tmp_path,
    )

    assert payload["family"] == "hinerv"
    assert payload["status"] == "trained_ladder_row_blocked"
    assert payload["rows"][0]["accepted"] is False
    assert "required_emission_field_missing:official_controls.config_name" in payload[
        "blockers"
    ]
    assert "required_emission_field_missing:bitstream_codec" in payload["blockers"]


def test_receiver_replay_boolean_without_proof_identity_blocks_row(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"receiver-closed")
    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "n_pairs": 600,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "official_controls": _snerv_controls(),
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof={"receiver_archive_replay_verified": True},
        scorer_eval={"avg_segnet_dist": 0.004, "avg_posenet_dist": 0.002},
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert payload["status"] == "trained_ladder_row_blocked"
    assert "receiver_proof_identity_missing" in payload["blockers"]
    assert row["receiver_archive_replay_verified"] is True
    assert row["receiver_proof_identity_bound"] is False
    assert row["receiver_proof_identity"]["bound"] is False
    assert "receiver_proof_path_missing" in row["receiver_proof_identity"]["blockers"]
    assert "receiver_proof_sha256_missing_or_invalid" in row["receiver_proof_identity"][
        "blockers"
    ]
    assert row["byte_closed_receiver_proof"] is False
    assert row["receiver_proof_passed"] is False


def test_receiver_proof_path_with_bad_sha_blocks_row(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"receiver-closed")
    proof = tmp_path / "bad-sha.receiver_proof.json"
    proof.write_text(
        (
            '{"schema":"snerv_inverse_steg_generated_receiver_proof.v1",'
            '"receiver_contract_satisfied":true,'
            '"runtime_consumption_proof_passed":true,'
            '"archive_bytes":1,'
            f'"archive_sha256":"{hashlib.sha256(b"other").hexdigest()}",'
            '"blockers":[]}\n'
        ),
        encoding="utf-8",
    )

    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "n_pairs": 600,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "official_controls": _snerv_controls(),
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof={
            "receiver_archive_replay_verified": True,
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_ready": True,
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_sha256": "0" * 64,
        },
        scorer_eval={
            "axis_tag": "[contest-CPU unit-test receiver-closed]",
            "avg_segnet_dist": 0.004,
            "avg_posenet_dist": 0.002,
        },
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert payload["status"] == "trained_ladder_row_blocked"
    assert "receiver_proof_identity_missing" in payload["blockers"]
    assert "receiver_proof_sha256_mismatch" in payload["blockers"]
    assert row["receiver_proof_identity_bound"] is False
    assert row["receiver_proof_identity"]["proof_path"] == proof.as_posix()
    assert row["receiver_proof_identity"]["proof_sha256"] == hashlib.sha256(
        proof.read_bytes()
    ).hexdigest()
    assert row["byte_closed_receiver_proof"] is False


def test_advisory_axis_with_file_backed_receiver_proof_blocks_row(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"receiver-closed")
    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "n_pairs": 600,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "official_controls": _snerv_controls(),
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof=_receiver_proof(tmp_path, "advisory", archive=archive),
        scorer_eval={
            "axis_tag": "[macOS-CPU advisory]",
            "avg_segnet_dist": 0.004,
            "avg_posenet_dist": 0.002,
        },
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert payload["status"] == "trained_ladder_row_blocked"
    assert row["receiver_proof_identity_bound"] is True
    assert row["byte_closed_receiver_proof"] is True
    assert row["receiver_proof_passed"] is False
    assert "source_axis_not_receiver_closed_contest_authority" in payload["blockers"]


def test_arbitrary_file_backed_receiver_proof_content_blocks_row(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"receiver-closed")
    proof = tmp_path / "not-a-proof.json"
    proof.write_text('{"note":"not a receiver proof"}\n', encoding="utf-8")
    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "n_pairs": 600,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "official_controls": _snerv_controls(),
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof={
            "receiver_archive_replay_verified": True,
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        },
        scorer_eval={
            "axis_tag": "[contest-CPU unit-test receiver-closed]",
            "avg_segnet_dist": 0.004,
            "avg_posenet_dist": 0.002,
        },
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert payload["status"] == "trained_ladder_row_blocked"
    assert row["receiver_proof_identity_bound"] is False
    assert "receiver_proof_schema_missing" in payload["blockers"]
    assert "receiver_proof_payload_pass_flag_missing" in payload["blockers"]


def test_suffix_only_receiver_proof_schema_does_not_bind_row(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"receiver-closed")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = tmp_path / "fake-generated-proof.json"
    proof.write_text(
        (
            '{"schema":"invented_generated_receiver_proof.v1",'
            '"receiver_contract_satisfied":true,'
            '"runtime_consumption_proof_ready":true,'
            f'"archive_bytes":{archive.stat().st_size},'
            f'"archive_sha256":"{archive_sha}",'
            '"blockers":[]}\n'
        ),
        encoding="utf-8",
    )

    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "n_pairs": 600,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "official_controls": _snerv_controls(),
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof={
            "receiver_archive_replay_verified": True,
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        },
        scorer_eval={
            "axis_tag": "[contest-CPU unit-test receiver-closed]",
            "avg_segnet_dist": 0.004,
            "avg_posenet_dist": 0.002,
        },
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert payload["status"] == "trained_ladder_row_blocked"
    assert row["receiver_proof_identity_bound"] is False
    assert (
        "receiver_proof_schema_unrecognized:invented_generated_receiver_proof.v1"
        in payload["blockers"]
    )


def _ready_snerv_payload(
    tmp_path: Path,
    *,
    name: str,
    modelsize: float,
    fc_dim: int,
    d_seg: float,
    d_pose: float,
) -> dict[str, object]:
    archive = tmp_path / name
    archive.write_bytes(f"archive-{name}".encode("ascii"))
    return build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive,
        trainer_metadata={
            "n_pairs": 600,
            "modelsize_mparams": modelsize,
            "fc_dim": fc_dim,
            "official_controls": _snerv_controls(modelsize=modelsize, fc_dim=fc_dim),
            "receiver_codec_mode": "snar1",
            "lf_payload_codec": "portfolio_auto",
            "decoder_precision_mode": "mixed_magnitude_symmetric",
            "step_map_codec": "waterfill",
            "target_bits_per_coeff": 6,
            "qat_bits": 4,
        },
        receiver_proof=_receiver_proof(tmp_path, name, archive=archive),
        scorer_eval={
            "axis_tag": "[contest-CPU unit-test receiver-closed]",
            "avg_segnet_dist": d_seg,
            "avg_posenet_dist": d_pose,
        },
        repo_root=tmp_path,
    )


def _snerv_controls(
    *,
    modelsize: float = 0.04,
    fc_dim: int = 16,
) -> dict[str, object]:
    return {
        "--modelsize": modelsize,
        "fc_dim": fc_dim,
        "emb_size": 8,
        "wavelet": "haar",
        "levels": 1,
        "source_faithful_stack": True,
        "mfu_enabled": True,
        "hfr_enabled": True,
        "snerv_t_enabled": False,
    }


def _receiver_proof(tmp_path: Path, name: str, *, archive: Path) -> dict[str, object]:
    proof = tmp_path / f"{name}.receiver_proof.json"
    proof.write_text(
        (
            '{"schema":"snerv_inverse_steg_generated_receiver_proof.v1",'
            '"receiver_contract_satisfied":true,'
            '"runtime_consumption_proof_ready":true,'
            '"runtime_consumption_proof_passed":true,'
            f'"archive_bytes":{archive.stat().st_size},'
            f'"archive_sha256":"{hashlib.sha256(archive.read_bytes()).hexdigest()}",'
            '"receiver_output_bytes":123,'
            '"expected_receiver_output_bytes":123,'
            '"blockers":[]}\n'
        ),
        encoding="utf-8",
    )
    return {
        "receiver_archive_replay_verified": True,
        "receiver_contract_satisfied": True,
        "runtime_consumption_proof_ready": True,
        "receiver_proof_path": proof.as_posix(),
        "receiver_proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
    }
