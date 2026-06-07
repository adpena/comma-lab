# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import tac.analysis.snerv_source_forward_producer as source_forward_producer
import tools.source_forward_witness as witness_tool
from tac.analysis.snerv_source_forward_proof import (
    SOURCE_FORWARD_SURFACES,
    SOURCE_FORWARD_TENSOR_NAMES,
)
from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    build_snerv_archive_payload_bitflip_falsification_matrix,
    build_snerv_official_tub_input_prune_report,
    build_snerv_official_tub_input_pruned_packet,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    encode_official_mfu_hfr_tub_decoder_payload,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    OfficialConv2dNchw,
    OfficialHfrConvBlock,
    OfficialHfrHeads,
)
from tac.substrates.snerv_inverse_steg_carrier.official_mfu import (
    OfficialConvTranspose2dNchw,
    OfficialResidualBlocksWithInputConv,
    OfficialSnervMfu,
    OfficialSnervMfuSpec,
)
from tools.source_forward_witness import (
    build_source_forward_witness_payload,
    main,
    resolve_checkpoint_export_report_witness_inputs,
)


def test_source_forward_witness_cli_writes_fail_closed_artifact(
    tmp_path: Path,
) -> None:
    packet = _legacy_packet()
    packet_path = tmp_path / "legacy.snar"
    out = tmp_path / "witness.json"
    proof_rows = tmp_path / "proof_rows.jsonl"
    packet_path.write_bytes(packet)

    assert (
        main(
            [
                "--packet",
                str(packet_path),
                "--out",
                str(out),
                "--proof-row-jsonl",
                str(proof_rows),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["schema"] == "snerv_source_forward_witness_cli.v1"
    assert payload["passed"] is False
    assert payload["launch_gate_clearable"] is False
    assert payload["score_claim"] is False
    assert payload["source_forward_proof_action_effect"] is None
    assert proof_rows.read_text(encoding="utf-8") == ""
    assert any(
        blocker.startswith("snerv_source_forward_witness_build_failed:")
        for blocker in payload["blockers"]
    )
    assert (
        main(
            [
                "--packet",
                str(packet_path),
                "--out",
                str(out),
                "--allow-overwrite",
                "--expected-output-sha256",
                payload_file_sha256(out),
                "--fail-on-blockers",
            ]
        )
        == 2
    )


def test_source_forward_witness_payload_names_official_packet_blockers(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "official.snar"
    packet_path.write_bytes(_official_packet())

    payload = build_source_forward_witness_payload(
        packet_path=packet_path,
        pair_ids=[0],
        capture_official_torch_from_archive=True,
        generated_utc="2026-06-07T00:00:00Z",
    )

    assert payload["source_forward_proof_action_effect"] is not None
    assert payload["passed"] is False
    assert payload["launch_gate_clearable"] is False
    assert payload["output2_verdict"] in {
        "DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS",
        "REPARAMETERIZED_RENAME_REQUIRED",
    }
    assert payload["first_failed_tensor"] is not None
    assert any(
        "snerv_output2_boundary_not_source_identical" in blocker
        for blocker in payload["blockers"]
    )


def test_official_packet_bitflip_matrix_covers_logical_source_basis() -> None:
    matrix = build_snerv_archive_payload_bitflip_falsification_matrix(
        _official_packet()
    )

    assert matrix["schema"] == "snerv_payload_bitflip_falsification_matrix.v1"
    assert matrix["required_sections"] == [
        "lf_payload",
        "decoder_payload.mfu",
        "decoder_payload.hfr",
        "decoder_payload.tub",
        "decoder_payload.output_2",
    ]
    assert set(matrix["section_proofs"]) == set(matrix["required_sections"])
    for section, proof in matrix["section_proofs"].items():
        assert proof["bitflip_section"] == section
        assert proof["baseline_section_sha256"] != proof["mutated_section_sha256"]


def test_official_packet_tub_prune_report_marks_noncausal_tub_dominated() -> None:
    packet = _official_packet()
    report = build_snerv_official_tub_input_prune_report(packet)

    assert json.dumps(report, sort_keys=True)
    assert report["schema"] == "snerv_official_tub_input_prune_report.v1"
    assert report["passed"] is True
    assert report["verdict"] == "DROP_OR_REIFY"
    assert report["section"] == "decoder_payload.tub"
    assert report["causality"] == "noncausal"
    assert report["byte_value"] == 0
    assert report["receiver_rgb_equal"] is True
    assert report["max_abs_receiver_delta"] == 0.0
    assert report["bytes_saved"] > 0
    assert report["delta_bytes"] < 0
    assert report["rate_delta_score"] < 0.0
    assert report["raw_tub_input_bytes_saved"] > 0
    assert report["source_tub_input_storage"]["codec"] == "full_float64"
    assert (
        report["candidate_tub_input_storage"]["codec"]
        == "unused_synthetic_float64"
    )
    assert report["tub_bitflip_causality"]["noncausal"] is True
    assert report["required_next_test"] == "tub_operator_reification_bitflip"
    assert report["launch_gate_clearable"] is False
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False

    pruned = build_snerv_official_tub_input_pruned_packet(packet)
    assert pruned.total_bytes == report["candidate_packet_bytes"]
    source_decoded = unpack_snerv_archive(packet)
    assert pruned.section_bytes["decoder_payload"] < len(
        source_decoded.sections["decoder_payload"]
    )

    source_frames = source_decoded.decode_frames(clip_to_uint8_range=True)
    pruned_frames = unpack_snerv_archive(pruned.packet).decode_frames(
        clip_to_uint8_range=True
    )
    assert np.array_equal(
        np.rint(source_frames).astype(np.uint8),
        np.rint(pruned_frames).astype(np.uint8),
    )


def test_source_forward_witness_build_failure_preserves_input_resolution_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path = tmp_path / "official.snar"
    packet_path.write_bytes(_official_packet())

    def fail_build(**_kwargs):
        raise RuntimeError("strict source graph inputs unavailable")

    monkeypatch.setattr(
        witness_tool,
        "build_snerv_source_forward_proof_from_archive_packet",
        fail_build,
    )
    payload = build_source_forward_witness_payload(
        packet_path=packet_path,
        pair_ids=[0],
        capture_official_torch_from_upstream_source_graph=True,
        checkpoint_export_report_resolution={
            "schema": "snerv_source_forward_witness_input_resolution.v1",
            "checkpoint_export_report_requested": True,
            "blockers": [
                "snerv_source_forward_witness_report_source_config_path_missing",
                "snerv_source_forward_witness_report_source_frame_triplets_missing",
            ],
        },
        generated_utc="2026-06-07T00:00:00Z",
    )

    assert payload["source_forward_proof_action_effect"] is None
    assert payload["passed"] is False
    assert payload["launch_gate_clearable"] is False
    assert "snerv_source_forward_witness_report_source_config_path_missing" in (
        payload["blockers"]
    )
    assert "snerv_source_forward_witness_report_source_frame_triplets_missing" in (
        payload["blockers"]
    )
    assert any(
        blocker.startswith("snerv_source_forward_witness_build_failed:")
        for blocker in payload["blockers"]
    )


def test_source_forward_witness_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path = tmp_path / "official.snar"
    packet_path.write_bytes(_official_packet())

    def slow_build(**_kwargs):
        time.sleep(2)
        raise AssertionError("timeout did not fire")

    monkeypatch.setattr(
        witness_tool,
        "build_snerv_source_forward_proof_from_archive_packet",
        slow_build,
    )

    payload = build_source_forward_witness_payload(
        packet_path=packet_path,
        pair_ids=[0],
        capture_official_torch_from_upstream_source_graph=True,
        max_build_seconds=1,
        generated_utc="2026-06-07T00:00:00Z",
    )

    assert payload["source_forward_proof_action_effect"] is None
    assert payload["passed"] is False
    assert payload["launch_gate_clearable"] is False
    assert payload["capture_modes"]["max_build_seconds"] == 1
    assert payload["build_exception"]["type"] == "TimeoutError"
    assert payload["build_exception"]["message"] == (
        "snerv_source_forward_witness_build_timeout:1s"
    )
    assert "snerv_source_forward_witness_build_failed:TimeoutError" in payload[
        "blockers"
    ]


def test_source_forward_witness_missing_packet_is_fail_closed_payload(
    tmp_path: Path,
) -> None:
    missing_packet = tmp_path / "missing.snar"

    payload = build_source_forward_witness_payload(
        packet_path=missing_packet,
        pair_ids=[0],
        checkpoint_export_report_resolution={
            "schema": "snerv_source_forward_witness_input_resolution.v1",
            "checkpoint_export_report_requested": True,
            "blockers": [
                "snerv_source_forward_witness_report_packet_path_missing_on_disk"
            ],
        },
        generated_utc="2026-06-07T00:00:00Z",
    )

    assert payload["source_forward_proof_action_effect"] is None
    assert payload["packet_sha256"] is None
    assert payload["packet_bytes"] is None
    assert payload["passed"] is False
    assert payload["launch_gate_clearable"] is False
    assert "snerv_source_forward_witness_report_packet_path_missing_on_disk" in (
        payload["blockers"]
    )
    assert "snerv_source_forward_witness_packet_path_missing_on_disk" in (
        payload["blockers"]
    )


def test_source_forward_witness_cli_writes_proof_row_jsonl(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "official.snar"
    out = tmp_path / "witness.json"
    proof_rows = tmp_path / "proof_rows.jsonl"
    packet_path.write_bytes(_official_packet())

    assert (
        main(
            [
                "--packet",
                str(packet_path),
                "--out",
                str(out),
                "--proof-row-jsonl",
                str(proof_rows),
                "--capture-official-torch-from-archive-diagnostic",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in proof_rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    assert rows[0] == payload["source_forward_proof_action_effect"]
    assert rows[0]["schema"] == "snerv_source_forward_proof_action_effect.v1"
    assert rows[0]["score_claim"] is False
    assert rows[0]["promotion_eligible"] is False


def test_source_forward_witness_upstream_pair_mismatch_is_graph_unproven(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path = tmp_path / "official.snar"
    packet_path.write_bytes(_official_packet())

    def fake_upstream_capture(**_kwargs) -> dict[str, object]:
        return {
            "schema": "snerv_official_tub_source_forward_tensor_bundle.v1",
            "pair_ids": [1],
            "tensors": {},
            "model_source_sha256": "8" * 64,
            "checkpoint_sha256": "6" * 64,
            "state_dict_sha256": "7" * 64,
            "decoder_len": 7,
            "source_scope": "official_trained_checkpoint",
        }

    monkeypatch.setattr(
        source_forward_producer,
        "build_official_torch_upstream_fixture_tensors",
        fake_upstream_capture,
    )
    payload = build_source_forward_witness_payload(
        packet_path=packet_path,
        pair_ids=[0],
        capture_official_torch_from_upstream_fixture=True,
        generated_utc="2026-06-07T00:00:00Z",
    )
    status = payload["source_forward_proof_action_effect"]["producer_status"][
        "official_torch_upstream_capture_status"
    ]

    assert payload["passed"] is False
    assert payload["launch_gate_clearable"] is False
    assert status["verdict"] == "SOURCE_GRAPH_UNPROVEN"
    assert status["source_graph_unproven"] is True
    assert "snerv_upstream_source_capture_pair_ids_mismatch" in status["blockers"]
    assert "snerv_upstream_source_graph_unproven" in payload["blockers"]
    assert "snerv_upstream_source_capture_pair_ids_mismatch" in payload["blockers"]


def test_source_forward_witness_upstream_fixture_scope_status_is_graph_unproven(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path = tmp_path / "official.snar"
    packet_path.write_bytes(_official_packet())

    def fake_upstream_fixture_capture(**_kwargs) -> dict[str, object]:
        return {
            "schema": "snerv_official_tub_source_forward_tensor_bundle.v1",
            "pair_ids": [0],
            "tensors": {},
            "model_source_sha256": "8" * 64,
            "checkpoint_sha256": "6" * 64,
            "state_dict_sha256": "7" * 64,
            "decoder_len": 7,
            "source_scope": "official_source_fixture_state",
        }

    monkeypatch.setattr(
        source_forward_producer,
        "build_official_torch_upstream_fixture_tensors",
        fake_upstream_fixture_capture,
    )
    payload = build_source_forward_witness_payload(
        packet_path=packet_path,
        pair_ids=[0],
        capture_official_torch_from_upstream_fixture=True,
        generated_utc="2026-06-07T00:00:00Z",
    )
    status = payload["source_forward_proof_action_effect"]["producer_status"][
        "official_torch_upstream_capture_status"
    ]

    assert payload["passed"] is False
    assert payload["launch_gate_clearable"] is False
    assert status["verdict"] == "SOURCE_GRAPH_UNPROVEN"
    assert status["source_graph_unproven"] is True
    assert "snerv_upstream_source_graph_unproven" in status["blockers"]
    assert "snerv_official_torch_source_graph_unproven" in status["blockers"]
    assert (
        "snerv_official_torch_trained_checkpoint_source_scope_missing"
        in status["blockers"]
    )


def test_source_forward_witness_cli_threads_official_source_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path = tmp_path / "official.snar"
    out = tmp_path / "witness.json"
    source_config_path = tmp_path / "args.json"
    packet_path.write_bytes(_official_packet())
    source_config_path.write_text('{"fc_dim": 1152}\n', encoding="utf-8")
    captured_kwargs: dict[str, object] = {}

    def fake_upstream_fixture_capture(**kwargs) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {
            "schema": "snerv_official_tub_source_forward_tensor_bundle.v1",
            "pair_ids": [0],
            "tensors": {},
            "model_source_sha256": "8" * 64,
            "checkpoint_sha256": "6" * 64,
            "state_dict_sha256": "7" * 64,
            "decoder_len": 7,
            "source_scope": "official_trained_checkpoint",
            "source_config_lineage": kwargs["official_trained_source_config_kind"],
            "source_config_sha256": "9" * 64,
            "source_config_kind": "official_snerv_t_train_config",
            "source_config_source": kwargs["official_trained_source_config_path"],
            "source_config_is_fixture": False,
        }

    monkeypatch.setattr(
        source_forward_producer,
        "build_official_torch_upstream_fixture_tensors",
        fake_upstream_fixture_capture,
    )

    assert (
        main(
            [
                "--packet",
                str(packet_path),
                "--out",
                str(out),
                "--capture-official-torch-from-upstream-fixture",
                "--official-torch-source-config",
                str(source_config_path),
                "--official-torch-source-config-kind",
                "checkpoint_export_official_trained_run_config",
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    row = payload["source_forward_proof_action_effect"]
    manifest = row["producer_status"]["official_torch_upstream_capture_manifest"]
    status = row["producer_status"]["official_torch_upstream_capture_manifest_status"]
    assert captured_kwargs["official_trained_source_config_path"] == (
        source_config_path.as_posix()
    )
    assert (
        captured_kwargs["official_trained_source_config_kind"]
        == "checkpoint_export_official_trained_run_config"
    )
    assert payload["capture_modes"]["official_torch_source_config_requested"] is True
    assert manifest["source_config_source"] == source_config_path.as_posix()
    assert (
        manifest["source_config_lineage"]
        == "checkpoint_export_official_trained_run_config"
    )
    assert manifest["source_config_is_fixture"] is False
    assert "snerv_official_torch_source_config_sha256_invalid" not in status["blockers"]
    assert payload["launch_gate_clearable"] is False
    assert payload["score_claim"] is False


def test_source_forward_witness_cli_threads_strict_source_graph_triplets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path = tmp_path / "official.snar"
    out = tmp_path / "witness.json"
    source_config_path = tmp_path / "args.json"
    triplets_path = tmp_path / "triplets.npy"
    packet_path.write_bytes(_official_packet())
    source_config_path.write_text('{"fc_dim": 1152}\n', encoding="utf-8")
    np.save(triplets_path, np.zeros((1, 3, 3, 4, 4), dtype=np.float32))
    captured_kwargs: dict[str, object] = {}

    def fake_upstream_source_graph_capture(**kwargs) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        triplets = np.asarray(kwargs["source_frame_triplets_nchw255"])
        return {
            "schema": "snerv_official_tub_strict_source_graph_capture.v1",
            "pair_ids": [0],
            "tensors": {},
            "model_source_sha256": "8" * 64,
            "checkpoint_sha256": "6" * 64,
            "state_dict_sha256": "7" * 64,
            "decoder_len": 7,
            "source_scope": "official_trained_checkpoint",
            "source_config_lineage": kwargs["official_trained_source_config_kind"],
            "source_config_sha256": "9" * 64,
            "source_config_kind": "official_snerv_t_train_config",
            "source_config_source": kwargs["official_trained_source_config_path"],
            "source_config_is_fixture": False,
            "source_graph_unproven": False,
            "source_frame_triplets_shape": list(triplets.shape),
        }

    monkeypatch.setattr(
        source_forward_producer,
        "build_official_torch_upstream_source_graph_tensors",
        fake_upstream_source_graph_capture,
    )

    assert (
        main(
            [
                "--packet",
                str(packet_path),
                "--out",
                str(out),
                "--capture-official-torch-from-upstream-source-graph",
                "--official-torch-checkpoint-state-dict",
                str(tmp_path / "state.npz"),
                "--official-torch-source-config",
                str(source_config_path),
                "--official-torch-source-frame-triplets-npy",
                str(triplets_path),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    row = payload["source_forward_proof_action_effect"]
    status = row["producer_status"]["official_torch_upstream_capture_status"]
    assert captured_kwargs["official_trained_source_config_path"] == (
        source_config_path.as_posix()
    )
    assert captured_kwargs["official_trained_checkpoint_state_dict_path"] == (
        (tmp_path / "state.npz").as_posix()
    )
    assert np.asarray(captured_kwargs["source_frame_triplets_nchw255"]).shape == (
        1,
        3,
        3,
        4,
        4,
    )
    assert (
        payload["capture_modes"]["official_torch_from_upstream_source_graph"] is True
    )
    assert payload["capture_modes"]["official_torch_source_frame_triplets_requested"] is True
    assert (
        row["producer_status"]["official_torch_captured_from_upstream_source_graph"]
        is True
    )
    assert status["source_graph_unproven"] is True
    assert "snerv_official_torch_source_graph_unproven" in status["blockers"]
    assert payload["launch_gate_clearable"] is False
    assert payload["score_claim"] is False


def test_source_forward_witness_checkpoint_export_report_resolves_strict_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_path = tmp_path / "official.snar"
    state_slice_path = tmp_path / "official_state_slice.npz"
    source_config_path = tmp_path / "args.json"
    triplets_path = tmp_path / "triplets.npy"
    report_path = tmp_path / "snerv_checkpoint_archive_export.json"
    out = tmp_path / "witness.json"
    packet_path.write_bytes(_official_packet())
    state_slice_path.write_bytes(b"state slice placeholder")
    source_config_path.write_text('{"fc_dim": 1152}\n', encoding="utf-8")
    np.save(triplets_path, np.zeros((1, 3, 3, 4, 4), dtype=np.float32))
    report_path.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "packet_path": packet_path.name,
                "startup_json_path": "startup_marker_is_not_source_config.json",
                "official_checkpoint_export_binding": {
                    "official_trained_checkpoint_state_dict_slice_path": (
                        state_slice_path.name
                    ),
                    "official_trained_checkpoint_state_dict_slice_present": True,
                },
            }
        ),
        encoding="utf-8",
    )
    captured_kwargs: dict[str, object] = {}

    def fake_upstream_source_graph_capture(**kwargs) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {
            "schema": "snerv_official_tub_strict_source_graph_capture.v1",
            "pair_ids": [0],
            "tensors": {},
            "model_source_sha256": "8" * 64,
            "checkpoint_sha256": "6" * 64,
            "state_dict_sha256": "7" * 64,
            "decoder_len": 7,
            "source_scope": "official_trained_checkpoint",
            "source_config_lineage": kwargs["official_trained_source_config_kind"],
            "source_config_sha256": "9" * 64,
            "source_config_kind": "official_snerv_t_train_config",
            "source_config_source": kwargs["official_trained_source_config_path"],
            "source_config_is_fixture": False,
            "source_graph_unproven": False,
        }

    monkeypatch.setattr(
        source_forward_producer,
        "build_official_torch_upstream_source_graph_tensors",
        fake_upstream_source_graph_capture,
    )

    assert (
        main(
            [
                "--checkpoint-export-report",
                str(report_path),
                "--out",
                str(out),
                "--capture-official-torch-from-upstream-source-graph",
                "--official-torch-source-config",
                str(source_config_path),
                "--official-torch-source-config-kind",
                "checkpoint_export_official_trained_run_config",
                "--official-torch-source-frame-triplets-npy",
                str(triplets_path),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    resolution = payload["checkpoint_export_report_resolution"]
    assert payload["packet_path"] == packet_path.resolve(strict=False).as_posix()
    assert payload["capture_modes"]["checkpoint_export_report_requested"] is True
    assert resolution["official_torch_checkpoint_state_dict_source"] == (
        "checkpoint_export_official_state_dict_slice"
    )
    assert resolution["startup_json_path_not_source_authority"] == (
        "startup_marker_is_not_source_config.json"
    )
    assert captured_kwargs["official_trained_checkpoint_state_dict_path"] == (
        state_slice_path.resolve(strict=False).as_posix()
    )
    assert captured_kwargs["official_trained_source_config_path"] == (
        source_config_path.resolve(strict=False).as_posix()
    )
    assert np.asarray(captured_kwargs["source_frame_triplets_nchw255"]).shape == (
        1,
        3,
        3,
        4,
        4,
    )
    assert "snerv_source_forward_witness_report_source_config_path_missing" not in (
        payload["blockers"]
    )
    assert "snerv_source_forward_witness_report_source_frame_triplets_missing" not in (
        payload["blockers"]
    )
    assert payload["score_claim"] is False


def test_source_forward_producer_refreshes_strict_capture_status_after_scorer_tensors(
    monkeypatch,
) -> None:
    packet = _official_packet()
    archive_tensors = unpack_snerv_archive(packet).source_forward_receiver_tensor_surfaces(
        [0]
    )["surface_tensors"]["archive_parseback"]
    scorer_tensors = {
        "segnet_input": np.zeros((1, 3, 2, 2), dtype=np.float32),
        "posenet_input": np.zeros((1, 6, 2, 2), dtype=np.float32),
        "segnet_logits": np.zeros((1, 4, 2, 2), dtype=np.float32),
        "segnet_argmax": np.zeros((1, 2, 2), dtype=np.int64),
        "posenet_output": np.zeros((1, 12), dtype=np.float32),
    }

    def fake_upstream_source_graph_capture(**kwargs) -> dict[str, object]:
        return {
            "schema": "snerv_official_tub_strict_source_graph_capture.v1",
            "pair_ids": [0],
            "tensors": dict(archive_tensors),
            "model_source_sha256": "8" * 64,
            "checkpoint_sha256": "6" * 64,
            "state_dict_sha256": "7" * 64,
            "decoder_len": 7,
            "source_scope": "official_trained_checkpoint",
            "source_config_lineage": kwargs["official_trained_source_config_kind"],
            "source_config_sha256": "9" * 64,
            "source_config_kind": "official_snerv_t_train_config",
            "source_config_source": kwargs["official_trained_source_config_path"],
            "source_config_is_fixture": False,
            "source_graph_unproven": False,
        }

    monkeypatch.setattr(
        source_forward_producer,
        "build_official_torch_upstream_source_graph_tensors",
        fake_upstream_source_graph_capture,
    )

    row = source_forward_producer.build_snerv_source_forward_proof_from_archive_packet(
        action_id="strict-capture-status-refresh",
        archive_packet=packet,
        pair_ids=[0],
        capture_official_torch_from_upstream_source_graph=True,
        official_torch_checkpoint_state_dict_path="state.npz",
        official_torch_source_config_path="args.json",
        official_torch_source_config_kind=(
            "checkpoint_export_official_trained_run_config"
        ),
        official_torch_source_frame_triplets_nchw255=np.zeros(
            (1, 3, 3, 4, 4),
            dtype=np.float32,
        ),
        pact_mlx_tensors=dict(archive_tensors),
        scorer_tensors_by_surface={
            surface: dict(scorer_tensors) for surface in SOURCE_FORWARD_SURFACES
        },
        scorer_deltas={
            "d_seg": 0.0,
            "d_pose": 0.0,
            "delta_score_nonrate": 0.0,
            "by_surface": {
                surface: {"d_seg": 0.0, "d_pose": 0.0}
                for surface in SOURCE_FORWARD_SURFACES
            },
        },
    )
    status = row["producer_status"]["official_torch_upstream_capture_status"]

    assert status["verdict"] == "SOURCE_GRAPH_CAPTURED"
    assert status["source_graph_unproven"] is False
    assert status["blockers"] == []
    assert row["producer_status"]["official_torch_upstream_capture_manifest_passed"] is True
    assert row["producer_status"]["official_torch_captured_from_upstream_source_graph"] is True
    assert row["output2_boundary_verdict"]["verdict"] == "SOURCE_IDENTICAL"
    assert not any(
        blocker.startswith(
            "snerv_official_torch_manifest_missing_required_tensors:"
        )
        for blocker in row["blockers"]
    )


def test_source_forward_witness_report_resolution_refuses_startup_as_config(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "official.snar"
    state_slice_path = tmp_path / "official_state_slice.npz"
    report_path = tmp_path / "snerv_checkpoint_archive_export.json"
    packet_path.write_bytes(_official_packet())
    state_slice_path.write_bytes(b"state slice placeholder")
    report_path.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "packet_path": packet_path.name,
                "startup_json_path": "startup_marker_is_not_source_config.json",
                "official_checkpoint_export_binding": {
                    "official_trained_checkpoint_state_dict_slice_path": (
                        state_slice_path.name
                    ),
                    "official_trained_checkpoint_state_dict_slice_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    resolution = resolve_checkpoint_export_report_witness_inputs(report_path)

    assert resolution["packet_path"] == packet_path.resolve(strict=False).as_posix()
    assert resolution["official_torch_checkpoint_state_dict_path"] == (
        state_slice_path.resolve(strict=False).as_posix()
    )
    assert resolution["official_torch_source_config_path"] is None
    assert resolution["startup_json_path_not_source_authority"] == (
        "startup_marker_is_not_source_config.json"
    )
    assert "snerv_source_forward_witness_report_source_config_path_missing" in (
        resolution["blockers"]
    )
    assert "snerv_source_forward_witness_report_source_frame_triplets_missing" in (
        resolution["blockers"]
    )


def test_source_forward_witness_report_resolution_marks_stale_paths(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "snerv_checkpoint_archive_export.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "packet_path": "stale_packet.snar",
                "official_torch_source_config_path": "stale_args.json",
                "source_frame_triplets_npy": "stale_triplets.npy",
                "official_checkpoint_export_binding": {
                    "official_trained_checkpoint_state_dict_slice_path": (
                        "stale_state.npz"
                    ),
                    "official_trained_checkpoint_state_dict_slice_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    resolution = resolve_checkpoint_export_report_witness_inputs(report_path)

    assert resolution["packet_path"] == (
        tmp_path / "stale_packet.snar"
    ).resolve(strict=False).as_posix()
    assert "snerv_source_forward_witness_report_packet_path_missing_on_disk" in (
        resolution["blockers"]
    )
    assert (
        "snerv_source_forward_witness_report_checkpoint_state_dict_path_missing_on_disk"
        in resolution["blockers"]
    )
    assert (
        "snerv_source_forward_witness_report_source_config_path_missing_on_disk"
        in resolution["blockers"]
    )
    assert (
        "snerv_source_forward_witness_report_source_frame_triplets_missing_on_disk"
        in resolution["blockers"]
    )


def test_source_forward_producer_refreshes_manifest_after_supplied_scorer_tensors() -> None:
    scorer_tensor_names = {
        "segnet_input",
        "posenet_input",
        "segnet_logits",
        "segnet_argmax",
        "posenet_output",
    }
    official_tensors = {
        name: np.zeros((1,), dtype=np.float32)
        for name in SOURCE_FORWARD_TENSOR_NAMES
        if name not in scorer_tensor_names
    }
    manifest = source_forward_producer.build_snerv_official_torch_upstream_capture_manifest(
        pair_ids=[0],
        tensor_names=official_tensors.keys(),
        model_source_sha256="8" * 64,
        checkpoint_sha256="6" * 64,
        state_dict_sha256="7" * 64,
        source_config_lineage="official_trained_run_config",
        source_config_sha256="9" * 64,
        source_config_kind="official_snerv_t_train_config",
        source_config_source="unit_test_exact_trained_config",
        source_config_is_fixture=False,
        decoder_len=7,
        source_scope="official_trained_checkpoint",
        trained_checkpoint_lineage="official_trained_checkpoint_state_dict",
        capture_origin="official_upstream_trained_checkpoint",
    )
    scorer_tensors = {
        "segnet_input": np.zeros((1, 3, 2, 2), dtype=np.float32),
        "posenet_input": np.zeros((1, 6, 2, 2), dtype=np.float32),
        "segnet_logits": np.zeros((1, 4, 2, 2), dtype=np.float32),
        "segnet_argmax": np.zeros((1, 2, 2), dtype=np.int64),
        "posenet_output": np.zeros((1, 6), dtype=np.float32),
    }
    scorer_deltas = {
        "d_seg": 0.0,
        "d_pose": 0.0,
        "delta_score_nonrate": 0.0,
        "by_surface": {
            surface: {"d_seg": 0.0, "d_pose": 0.0}
            for surface in SOURCE_FORWARD_SURFACES
        },
    }

    row = source_forward_producer.build_snerv_source_forward_proof_from_archive_packet(
        action_id="refresh_manifest",
        archive_packet=_official_packet(),
        pair_ids=[0],
        official_torch_tensors=official_tensors,
        official_torch_capture_manifest=manifest,
        scorer_tensors_by_surface={"official_torch": scorer_tensors},
        scorer_deltas=scorer_deltas,
        generated_utc="2026-06-07T00:00:00Z",
    )

    refreshed = row["producer_status"]["official_torch_upstream_capture_manifest"]
    status = row["producer_status"]["official_torch_upstream_capture_manifest_status"]
    assert status["passed"] is True
    assert refreshed["source_graph_unproven"] is False
    assert refreshed["missing_required_tensor_names"] == []
    assert row["surface_provenance"]["official_torch"][
        "tensor_capture_authority"
    ] == "upstream_snerv_t_forward_source_graph"


def _legacy_packet() -> bytes:
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[0.0]),
        lf_payload=encode_lf_quant_payload([np.zeros((1, 1), dtype=np.int64)]),
        decoder_payload=encode_decoder_payload(HfGenerationDecoder.zeros(levels=1)),
        step_map_packet=encode_step_maps(
            [np.ones((1, 1), dtype=np.float32)],
            bins=4,
        ).packet,
        metadata={"lf_plane_count": 1, "levels": 1, "wavelet": "haar"},
    )
    return archive.packet


def _official_packet() -> bytes:
    bundle = _official_payload_fixture()
    bundle["low"] = np.concatenate(
        [bundle["low"], np.asarray(bundle["low"]) + 0.125],
        axis=0,
    )
    bundle["skip_mid"] = np.concatenate(
        [bundle["skip_mid"], np.asarray(bundle["skip_mid"]) - 0.125],
        axis=0,
    )
    bundle["skip_high"] = np.concatenate(
        [bundle["skip_high"], np.asarray(bundle["skip_high"]) + 0.25],
        axis=0,
    )
    bundle["temporal_encoder_output_shape"] = (1, 6, 8, 8)
    bundle["output2_decoder_output_shape"] = (2, 12, 8, 8)
    official_payload = encode_official_mfu_hfr_tub_decoder_payload(
        **bundle,
        tub_temporal_encoder_concat=np.linspace(
            0.0,
            1.0,
            1 * 6 * 8 * 8,
            dtype=np.float64,
        ).reshape(1, 6, 8, 8),
        tub_output2_raw=np.full((2, 12, 8, 8), 0.125, dtype=np.float64),
        store_tub_output2_for_receiver_proof=True,
    )
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[0.0]),
        lf_payload=encode_lf_quant_payload([np.zeros((1, 1), dtype=np.int64)]),
        decoder_payload=official_payload,
        step_map_packet=encode_step_maps(
            [np.ones((1, 1), dtype=np.float32)],
            bins=4,
        ).packet,
        metadata={
            "lf_plane_count": 1,
            "levels": 1,
            "wavelet": "haar",
            "orig_hw": [16, 16],
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 3,
        },
    )
    return archive.packet


def _official_payload_fixture(seed: int = 17) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    spec = OfficialSnervMfuSpec(
        low_channels=1,
        mid_channels=1,
        high_channels=1,
        mid_stride=2,
        high_stride=2,
        num_blocks=0,
    )
    mfu = OfficialSnervMfu(
        spec=spec,
        upsample_mid=OfficialConvTranspose2dNchw(
            rng.standard_normal((1, 1, 2, 2)) * 0.04,
            rng.standard_normal(1) * 0.01,
            stride=2,
        ),
        rb_mid=OfficialResidualBlocksWithInputConv(
            input_conv=OfficialConv2dNchw(
                rng.standard_normal((1, 2, 3, 3)) * 0.04,
                rng.standard_normal(1) * 0.01,
                padding=1,
            ),
            residual_blocks=(),
        ),
        upsample_high=OfficialConvTranspose2dNchw(
            rng.standard_normal((1, 1, 2, 2)) * 0.04,
            rng.standard_normal(1) * 0.01,
            stride=2,
        ),
        rb_high=OfficialResidualBlocksWithInputConv(
            input_conv=OfficialConv2dNchw(
                rng.standard_normal((1, 2, 3, 3)) * 0.04,
                rng.standard_normal(1) * 0.01,
                padding=1,
            ),
            residual_blocks=(),
        ),
    )
    yy, xx = np.mgrid[0:8, 0:8].astype(np.float64)
    return {
        "mfu": mfu,
        "hfr_heads": OfficialHfrHeads(
            lh_head=_official_hfr_head(rng),
            hl_head=_official_hfr_head(rng),
            hh_head=_official_hfr_head(rng),
        ),
        "low": rng.standard_normal((1, 1, 2, 2)) * 0.2,
        "skip_mid": rng.standard_normal((1, 1, 4, 4)) * 0.2,
        "skip_high": rng.standard_normal((1, 1, 8, 8)) * 0.2,
        "tub_current": np.stack([np.sin(xx / 3.0) + np.cos(yy / 4.0)], axis=0),
        "tub_previous": np.stack(
            [np.sin((xx - 1.0) / 3.0) + np.cos(yy / 4.0)],
            axis=0,
        ),
        "tub_next_frame": np.stack(
            [np.sin((xx + 1.0) / 3.0) + np.cos(yy / 4.0)],
            axis=0,
        ),
        "temporal_encoder_output_shape": (1, 4, 4, 4),
        "fc_hw": (2, 2),
        "output2_decoder_output_shape": (2, 8, 4, 4),
    }


def _official_hfr_head(rng: np.random.Generator) -> OfficialHfrConvBlock:
    return OfficialHfrConvBlock(
        conv1=OfficialConv2dNchw(
            rng.standard_normal((2, 1, 1, 1)) * 0.04,
            rng.standard_normal(2) * 0.01,
        ),
        conv2=OfficialConv2dNchw(
            rng.standard_normal((3, 2, 3, 3)) * 0.04,
            rng.standard_normal(3) * 0.01,
            padding=1,
        ),
    )


def payload_file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
