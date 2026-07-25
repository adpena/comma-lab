# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import json
import lzma
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.optimization import ddm_runtime_exporter as exporter
from tac.optimization import ddm_runtime_receiver as receiver
from tac.optimization.ddm_e5a_midcampaign_adapter import (
    DDME5AMidcampaignCheckpointAdapterConfigV1,
    DDME5ASolveMemberAdapterConfigV1,
    compile_solve_member_bundle,
)
from tac.optimization.ddm_pc1_pose_stream import PC1PosePacketV1
from tac.optimization.ddm_rg4_g3_blocks_and_active_tube import (
    build_source_local_composition_archive,
)
from tools import rehearse_ddm_runtime_upstream as harness

REPO = Path(__file__).resolve().parents[4]
W_SEG_CONFIG = REPO / ".omx/research/configs/ddm_e5_e4_ws1_wseg_brotli_20260724.json"
IC1_CONFIG = REPO / ".omx/research/configs/ddm_ic1_incumbent_compose_and_buy_row_20260724.json"
IC2_CONFIG = REPO / ".omx/research/configs/ddm_ic2_optimal_incumbent_pose_typed_20260724.json"
E5A_STEP50_STATE = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_ws4_pose_null_projected_seg_start_20260725T112500Z/01_archives/"
    "W_joint_step50_live.zip.receipt-bytes"
)
KS1_ADAPTER_CONFIG = (
    REPO / ".omx/research/configs/ddm_ks1_knee_member_adapter_20260725.json"
)
KS1_EXPORT_CONFIG = (
    REPO / ".omx/research/configs/ddm_ks1_knee_member_e5a_export_20260725.json"
)


def test_e5a_adapter_config_refuses_non_ssd_checkpoint_custody() -> None:
    with pytest.raises(ValueError, match="governed SSD"):
        DDME5AMidcampaignCheckpointAdapterConfigV1(
            ticket_path=".omx/research/configs/ticket.json",
            ticket_sha256="a" * 64,
            checkpoint_path="/tmp/checkpoint.npz",
            checkpoint_bytes=1,
            checkpoint_sha256="b" * 64,
            expected_stage_id="stage",
            expected_global_step=50,
            expected_lane_programs_materialized=False,
            expected_state_bytes=1,
            expected_state_sha256="c" * 64,
            output_state_path=(
                "/Volumes/VertigoDataTier/pact/e5a/state.receipt-bytes"
            ),
            output_receipt_path=(
                "/Volumes/VertigoDataTier/pact/e5a/receipt.json"
            ),
        )


def test_e5a_solve_member_adapter_reconstructs_rd1_knee_exactly() -> None:
    config = DDME5ASolveMemberAdapterConfigV1.model_validate_json(
        KS1_ADAPTER_CONFIG.read_bytes(),
        strict=True,
    )
    state, proof = compile_solve_member_bundle(config)
    assert len(state) == 138801
    assert hashlib.sha256(state).hexdigest() == (
        "5aa45850ab05d47f411583fd7582e27644c5bf289cd6d5bc32c05a52706c433e"
    )
    assert proof["candidate_id"] == "statistics_hard_analytic_composed_frame1"


def test_e5a_solve_member_export_config_uses_existing_e5a_route() -> None:
    config = exporter.load_config(KS1_EXPORT_CONFIG)
    assert isinstance(config, exporter.DDME5AMidcampaignRuntimeExporterConfigV1)
    assert config.run_id == "ddm_ks1_knee_member_realization_20260725"
    assert config.candidate == "W_joint"


def test_blob_frame_roundtrip_and_terminal_tamper_refusal() -> None:
    raw = bytes(range(251)) * 7
    framed = exporter._frame_blob(raw, kind=1, dimensions=(7, 251))
    decoded, shape = receiver._parse_blob(framed, expected_kind=1, label="synthetic")
    assert decoded == raw
    assert shape == (7, 251)

    tampered = bytearray(framed)
    tampered[-1] ^= 1
    with pytest.raises(receiver.ReceiverError):
        receiver._parse_blob(bytes(tampered), expected_kind=1, label="synthetic")


def test_e4_missing_brotli_uses_exact_e3_lzma1_bytes(monkeypatch) -> None:
    raw = bytes(range(251)) * 11
    monkeypatch.setattr(exporter, "brotli", None)
    assert exporter._e4_coder() == exporter.E3_LZMA1_CODER
    framed = exporter._frame_blob(
        raw,
        kind=1,
        dimensions=(11, 251),
        coder=exporter._e4_coder(),
    )
    coded = lzma.compress(
        raw,
        format=lzma.FORMAT_RAW,
        filters=receiver.LZMA_FILTERS,
    )
    expected = (
        exporter.BLOB_HEADER.pack(
            exporter.BLOB_MAGIC,
            1,
            2,
            1,
            2,
            len(raw),
            len(coded),
            hashlib.sha256(raw).digest(),
        )
        + struct.pack(">2I", 11, 251)
        + coded
    )
    assert framed == expected
    assert hashlib.sha256(framed).hexdigest() == ("7ed8d0bcfcb1833c15291d72cd2266dc81e168db1ebd8a3559490739d26be7e0")

    monkeypatch.setattr(receiver, "brotli", None)
    decoded, shape = receiver._parse_blob(
        framed,
        expected_kind=1,
        label="e4-fallback",
    )
    assert decoded == raw
    assert shape == (11, 251)


def test_ws1_refuses_false_brotli_absent_fallback(monkeypatch) -> None:
    monkeypatch.setattr(exporter, "brotli", None)
    with pytest.raises(
        exporter.ExporterError,
        match="source grammar itself contains Brotli-coded streams",
    ):
        exporter._ws1_e4_coder()


def test_e4_does_not_mask_brotli_coder_failure(monkeypatch) -> None:
    class BrokenBrotli:
        @staticmethod
        def compress(raw: bytes, *, quality: int) -> bytes:
            raise RuntimeError(f"brotli failure at q{quality} for {len(raw)} bytes")

    monkeypatch.setattr(exporter, "brotli", BrokenBrotli())
    assert exporter._e4_coder() == exporter.BROTLI_Q11_CODER
    with pytest.raises(RuntimeError, match="brotli failure"):
        exporter._frame_blob(
            b"must-not-fallback",
            kind=0,
            coder=exporter._e4_coder(),
        )


def test_e4_brotli_frame_is_consumed_and_terminal_tamper_refused() -> None:
    if exporter.brotli is None or receiver.brotli is None:
        pytest.skip("Brotli-present arm requires the declared dependency")
    raw = bytes(range(251)) * 13
    framed = exporter._frame_blob(
        raw,
        kind=1,
        dimensions=(13, 251),
        coder=exporter.BROTLI_Q11_CODER,
    )
    decoded, shape = receiver._parse_blob(
        framed,
        expected_kind=1,
        label="e4-brotli-consumption",
    )
    assert decoded == raw
    assert shape == (13, 251)

    tampered = bytearray(framed)
    tampered[-1] ^= 1
    with pytest.raises(receiver.ReceiverError):
        receiver._parse_blob(
            bytes(tampered),
            expected_kind=1,
            label="e4-brotli-consumption-tampered",
        )


def test_ws1_grammar_route_is_explicit_and_legacy_literals_remain_sealed() -> None:
    config = exporter.load_config(W_SEG_CONFIG)
    assert isinstance(config, exporter.DDME4WS1RuntimeExporterConfigV1)
    source = Path(config.source_archive_path).read_bytes()
    _received, admission, dofs = exporter._ws1_grammar_state(source, config)
    assert admission.archive_sha256 == config.source_archive_sha256
    assert [row.name for row in admission.streams] == [
        "nested_preuint8_archive",
        "warm_start_payload",
    ]
    assert sum(row.bytes for row in admission.streams) == len(source)
    assert dofs["total"] == 368

    with pytest.raises(ValueError):
        exporter.DDME1RuntimeExporterConfigV1(
            schema="DDME4RuntimeExporterConfigV1",
            run_id="ddm_e4_brotli_declared_dep_20260724",
            source_archive_path="state.zip",
            source_archive_bytes=len(source),
            source_archive_sha256=hashlib.sha256(source).hexdigest(),
            state_archive_bytes=len(source),
            state_archive_sha256=hashlib.sha256(source).hexdigest(),
            state_name="v15_j2_lane_seed_theta0",
            output_directory="packet",
            proof_root="/Volumes/VertigoDataTier/pact/evidence/test",
            minimum_free_bytes=8 * 1024 * 1024 * 1024,
        )


def test_ws1_packet_reconstructs_source_and_refuses_stream_or_coder_tamper() -> None:
    config = exporter.load_config(W_SEG_CONFIG)
    assert isinstance(config, exporter.DDME4WS1RuntimeExporterConfigV1)
    source = Path(config.source_archive_path).read_bytes()
    _received, admission, _dofs = exporter._ws1_grammar_state(source, config)
    framed = exporter._frame_blob(
        source,
        kind=0,
        coder=exporter.BROTLI_Q11_CODER,
    )
    members = {
        "manifest.json": b"",
        "state/ws1.ddj5": framed,
    }
    manifest = {
        "dependencies": ["numpy", "scipy", "torch", "brotli"],
        "false_authority": {
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "research_only": True,
            "score_claim": False,
        },
        "geometry": {
            "camera_hw": [874, 1164],
            "channels": 3,
            "frames_per_pair": 2,
            "pair_count": 600,
            "scorer_hw": [384, 512],
        },
        "grammar_admission": admission.to_dict(),
        "output": {
            "bytes": 600 * 2 * 874 * 1164 * 3,
            "sha256": "a" * 64,
        },
        "schema": exporter.E4_WS1_SCHEMA,
        "sections": [
            {
                "bytes": len(framed),
                "member": "state/ws1.ddj5",
                "sha256": hashlib.sha256(framed).hexdigest(),
                "typed_stream_tag": {
                    "schema": "ddm_typed_stream_tag.v1",
                    "type": "SKELETON",
                    "layer_home": "L1_program",
                    "evaluate_py_recursion_level_cited": ("L1_program -> L3_raster -> L4_scorer_feature -> L5_verdict"),
                    "counted_bytes": len(framed),
                    "free_receiver_code": True,
                },
            }
        ],
        "state": {
            "batch_pairs": 32,
            "name": config.state_name,
            "receiver_effective_dofs": 368,
        },
    }
    members["manifest.json"] = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    validated = receiver._validate_ws1_manifest(manifest, members)
    reconstructed, streams = receiver._reconstruct_ws1_state(
        validated,
        members,
    )
    assert reconstructed == source
    assert len(streams) == 2

    malformed = json.loads(json.dumps(manifest))
    malformed["grammar_admission"]["streams"][1]["sha256"] = "0" * 64
    with pytest.raises(receiver.ReceiverError, match="stream custody"):
        receiver._reconstruct_ws1_state(malformed, members)
    tampered = bytearray(framed)
    tampered[-1] ^= 1
    with pytest.raises(receiver.ReceiverError):
        receiver._parse_blob(
            bytes(tampered),
            expected_kind=0,
            label="state/ws1.ddj5",
        )


def test_cb1_rg4_packet_reconstructs_source_local_state_through_e4_frame() -> None:
    config = exporter.load_config(W_SEG_CONFIG)
    assert isinstance(config, exporter.DDME4WS1RuntimeExporterConfigV1)
    parent = Path(config.source_archive_path).read_bytes()
    packet = PC1PosePacketV1(
        active=False,
        pair_count=600,
        xi_scales=(1.0,) * 6,
        residual_scale=1.0,
        q_xi=np.zeros((2, 6), dtype=np.int16),
        q_luma_phase=np.zeros((2, 4), dtype=np.int8),
    )
    source = build_source_local_composition_archive(
        parent_archive=parent,
        parent_sha256=hashlib.sha256(parent).hexdigest(),
        packet=packet,
    )
    archive, receipt = exporter.compile_cb1_rg4_runtime_packet(
        source,
        state_name="test_cb1_inactive_source_local",
        output_bytes=600 * 2 * 874 * 1164 * 3,
        output_sha256="a" * 64,
    )
    assert receipt["packet_parseback_source_byte_identical"] is True
    assert receipt["coder"] in {
        exporter.BROTLI_Q11_CODER,
        exporter.E3_LZMA1_CODER,
    }
    with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
        members = {
            name: handle.read(name)
            for name in exporter.EXPECTED_CB1_MEMBERS
        }
    manifest = receiver._validate_cb1_manifest(
        json.loads(members["manifest.json"]),
        members,
    )
    reconstructed, dimensions = receiver._parse_blob(
        members["state/rg4.ddr4"],
        expected_kind=0,
        label="state/rg4.ddr4",
    )
    assert manifest["schema"] == exporter.CB1_SCHEMA
    assert reconstructed == source
    assert dimensions == ()


def test_e5a_la1_bundle_reconstructs_step50_state_and_refuses_tamper() -> None:
    if not E5A_STEP50_STATE.is_file():
        pytest.skip("WS4 step-50 receiver state is absent")
    source = E5A_STEP50_STATE.read_bytes()
    bundle, rows = exporter._compile_e5a_la1_bundle(source)
    reconstructed, consumed = receiver._reconstruct_e5a_la1_bundle(bundle)
    assert reconstructed == source
    assert len(bundle) == 128001
    assert sum(row["selected_framed_bytes"] for row in rows) == 127951
    assert len(rows) == len(consumed) == 7

    tampered = bytearray(bundle)
    tampered[-1] ^= 1
    with pytest.raises(receiver.ReceiverError):
        receiver._reconstruct_e5a_la1_bundle(bytes(tampered))


def test_ic1_is_a_separate_w_joint_then_pa1_typed_route() -> None:
    legacy = exporter.load_config(W_SEG_CONFIG)
    config = exporter.load_config(IC1_CONFIG)
    assert isinstance(legacy, exporter.DDME4WS1RuntimeExporterConfigV1)
    assert not isinstance(legacy, exporter.DDMIC1RuntimeExporterConfigV1)
    assert "amplitude_transform" not in legacy.model_dump(mode="json", by_alias=True)
    assert isinstance(config, exporter.DDMIC1RuntimeExporterConfigV1)
    assert config.candidate == "W_joint"
    assert config.batch_pairs == 16

    source = Path(config.source_archive_path).read_bytes()
    _received, admission, _dofs = exporter._ws1_grammar_state(source, config)
    framed = exporter._frame_blob(
        source,
        kind=0,
        coder=exporter.BROTLI_Q11_CODER,
    )
    members = {"manifest.json": b"", "state/ws1.ddj5": framed}
    manifest = {
        "amplitude_transform": {
            "application_frame": 0,
            "composition_order": "W_joint_then_PA1_frame0",
            "payload_bytes": 0,
            "rate_class": "FREE",
            "target_derivation": ("frozen_posenet_first_stem_conv_and_bn_only_video_independent"),
            "transform_id": receiver.PA1_TRANSFORM_ID,
        },
        "dependencies": ["numpy", "scipy", "torch", "brotli"],
        "false_authority": {
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "research_only": True,
            "score_claim": False,
        },
        "geometry": {
            "camera_hw": [874, 1164],
            "channels": 3,
            "frames_per_pair": 2,
            "pair_count": 600,
            "scorer_hw": [384, 512],
        },
        "grammar_admission": admission.to_dict(),
        "output": {
            "bytes": 600 * 2 * 874 * 1164 * 3,
            "sha256": "a" * 64,
        },
        "schema": exporter.IC1_SCHEMA,
        "sections": [
            {
                "bytes": len(framed),
                "member": "state/ws1.ddj5",
                "sha256": hashlib.sha256(framed).hexdigest(),
                "typed_stream_tag": {
                    "schema": "ddm_typed_stream_tag.v1",
                    "type": "SKELETON",
                    "layer_home": "L1_program",
                    "evaluate_py_recursion_level_cited": ("L1_program -> L3_raster -> L4_scorer_feature -> L5_verdict"),
                    "counted_bytes": len(framed),
                    "free_receiver_code": True,
                },
            }
        ],
        "state": {
            "batch_pairs": 16,
            "name": config.state_name,
            "receiver_effective_dofs": 368,
        },
    }
    members["manifest.json"] = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert receiver._validate_ws1_manifest(manifest, members) is manifest

    missing_transform = dict(manifest)
    del missing_transform["amplitude_transform"]
    with pytest.raises(receiver.ReceiverError, match="sealed schema"):
        receiver._validate_ws1_manifest(missing_transform, members)
    widened_legacy = {**manifest, "schema": exporter.E4_WS1_SCHEMA}
    with pytest.raises(receiver.ReceiverError, match="sealed schema"):
        receiver._validate_ws1_manifest(widened_legacy, members)
    wrong_order = json.loads(json.dumps(manifest))
    wrong_order["amplitude_transform"]["composition_order"] = "PA1_then_W_joint"
    with pytest.raises(receiver.ReceiverError, match="composition contract"):
        receiver._validate_ws1_manifest(wrong_order, members)


def test_ic2_is_a_separate_w_seg_then_pa1_declared_cv2_route() -> None:
    legacy = exporter.load_config(W_SEG_CONFIG)
    config = exporter.load_config(IC2_CONFIG)
    assert isinstance(legacy, exporter.DDME4WS1RuntimeExporterConfigV1)
    assert not isinstance(legacy, exporter.DDMIC2RuntimeExporterConfigV1)
    assert isinstance(config, exporter.DDMIC2RuntimeExporterConfigV1)
    assert config.candidate == "W_seg"
    assert config.composition_order == "W_seg_then_PA1_frame0"
    assert config.batch_pairs == 16

    source = Path(config.source_archive_path).read_bytes()
    _received, admission, _dofs = exporter._ws1_grammar_state(source, config)
    framed = exporter._frame_blob(
        source,
        kind=0,
        coder=exporter.BROTLI_Q11_CODER,
    )
    members = {"manifest.json": b"", "state/ws1.ddj5": framed}
    manifest = {
        "amplitude_transform": {
            "application_frame": 0,
            "composition_order": "W_seg_then_PA1_frame0",
            "payload_bytes": 0,
            "rate_class": "FREE",
            "target_derivation": ("frozen_posenet_first_stem_conv_and_bn_only_video_independent"),
            "transform_id": receiver.PA1_TRANSFORM_ID,
        },
        "dependencies": ["numpy", "torch", "brotli", "cv2"],
        "false_authority": {
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "research_only": True,
            "score_claim": False,
        },
        "geometry": {
            "camera_hw": [874, 1164],
            "channels": 3,
            "frames_per_pair": 2,
            "pair_count": 600,
            "scorer_hw": [384, 512],
        },
        "grammar_admission": admission.to_dict(),
        "output": {
            "bytes": 600 * 2 * 874 * 1164 * 3,
            "sha256": "a" * 64,
        },
        "schema": exporter.IC2_SCHEMA,
        "sections": [
            {
                "bytes": len(framed),
                "member": "state/ws1.ddj5",
                "sha256": hashlib.sha256(framed).hexdigest(),
                "typed_stream_tag": {
                    "schema": "ddm_typed_stream_tag.v1",
                    "type": "SKELETON",
                    "layer_home": "L1_program",
                    "evaluate_py_recursion_level_cited": (
                        "L1_program -> L3_raster -> L4_scorer_feature -> L5_verdict"
                    ),
                    "counted_bytes": len(framed),
                    "free_receiver_code": True,
                },
            }
        ],
        "state": {
            "batch_pairs": 16,
            "name": config.state_name,
            "receiver_effective_dofs": 368,
        },
    }
    members["manifest.json"] = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert receiver._validate_ws1_manifest(manifest, members) is manifest

    wrong_dependencies = json.loads(json.dumps(manifest))
    wrong_dependencies["dependencies"] = ["numpy", "scipy", "torch", "brotli"]
    with pytest.raises(receiver.ReceiverError, match="dependency contract"):
        receiver._validate_ws1_manifest(wrong_dependencies, members)
    widened_legacy = {**manifest, "schema": exporter.E4_WS1_SCHEMA}
    with pytest.raises(receiver.ReceiverError, match="sealed schema"):
        receiver._validate_ws1_manifest(widened_legacy, members)
    script = exporter._inflate_sh(bootstrap_opencv=True)
    assert b"opencv-python-headless==4.11.0.86" in script
    assert b"Brotli==1.2.0" in script
    assert b"opencv-python-headless" not in exporter._inflate_sh()


def test_ws1_runtime_payload_contains_only_a_sealed_generic_source_bundle() -> None:
    payload = exporter._ws1_runtime_payload()
    row = exporter._runtime_cleanliness(payload)
    assert row["status"] == "PASS"
    assert "embedded:tac.optimization.ddm_ws1_warm_start" in row["allowed_dependency_roots"]
    assert b'WS1_SOURCE_BUNDLE_B85 = b""' not in payload
    assert len(exporter._ws1_runtime_source_bundle()) > 0


def test_deterministic_zip_has_bijective_byte_homes() -> None:
    members = {
        "manifest.json": b"{}",
        "base/chart.ddb": exporter._frame_blob(b"chart", kind=0),
        "semantic/composed.dds": exporter._frame_blob(b"\0" * 8, kind=1, dimensions=(2, 2, 2)),
    }
    first = exporter._deterministic_zip(members)
    second = exporter._deterministic_zip(members)
    homes = exporter._zip_home_ledger(first)
    assert first == second
    assert sum(int(row["home_bytes"]) for row in homes) == len(first)


def test_public_exact_runtime_marginal_price_includes_container_and_parseback() -> None:
    control = {
        "manifest.json": b"{}",
        "base/chart.ddb": exporter._frame_blob(b"chart", kind=0),
        "semantic/composed.dds": exporter._frame_blob(
            b"\0" * 8,
            kind=1,
            dimensions=(2, 2, 2),
        ),
    }
    candidate = dict(control)
    candidate["base/chart.ddb"] = exporter._frame_blob(b"chart!", kind=0)

    def parseback(archive: bytes) -> bool:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
            return tuple(handle.namelist()) == exporter.EXPECTED_MEMBERS

    row = exporter.price_exact_runtime_marginal(
        control,
        candidate,
        parseback=parseback,
    )
    assert row.parseback_verified is True
    assert row.delta_archive_bytes == (row.candidate.archive_bytes - row.control.archive_bytes)
    assert row.control.archive_bytes == (row.control.member_payload_bytes + row.control.container_bytes)
    assert sum(int(home["home_bytes"]) for home in row.candidate.byte_home_ledger) == (row.candidate.archive_bytes)
    with pytest.raises(exporter.ExporterError, match="parse-back"):
        exporter.price_exact_runtime_packet(control, parseback=lambda _archive: False)


def test_runtime_cleanliness_allows_only_generic_dependencies() -> None:
    runtime = exporter.RUNTIME_SOURCE.read_bytes()
    row = exporter._runtime_cleanliness(runtime)
    assert row["status"] == "PASS"
    assert row["allowed_dependency_roots"] == ["torch", "brotli"]
    assert row["runtime_sha256"] == hashlib.sha256(runtime).hexdigest()
    assert b"import brotli" in runtime
    assert b"torch.cuda.is_available" not in runtime


def test_typed_extension_slots_and_joint_cycle_are_explicit_and_inactive() -> None:
    assert [row["block"] for row in exporter.EXTENSION_SLOTS] == [
        "D1",
        "D2",
        "D5",
        "D4",
        "D6",
    ]
    assert all(row["active_member"] is None for row in exporter.EXTENSION_SLOTS)
    assert all(row["rate_custody"] == "independent_member_bytes_sha256" for row in exporter.EXTENSION_SLOTS)
    assert list(exporter.EXTENSION_SLOTS) == receiver.EXPECTED_EXTENSION_SLOTS
    assert exporter.REFINEMENT_CONTRACT == receiver.EXPECTED_REFINEMENT
    assert exporter.REFINEMENT_CONTRACT["block_order"] == [
        "L",
        "D2",
        "D1",
        "D4",
        "D6",
        "D5",
    ]
    assert exporter.LANGUAGE_VERSION == receiver.LANGUAGE_VERSION


def test_block_version_stamps_are_verified_at_consumption_time() -> None:
    members = {
        "manifest.json": b"not-self-hashed",
        "base/chart.ddb": b"chart",
        "semantic/composed.dds": b"semantic",
    }
    rows = exporter._block_versions(
        chart_sha256=hashlib.sha256(members["base/chart.ddb"]).hexdigest(),
        semantic_sha256=hashlib.sha256(members["semantic/composed.dds"]).hexdigest(),
    )
    receiver._validate_block_versions(rows, members, amplitude_enabled=False)
    rows[0]["input_members"][0]["sha256"] = "0" * 64
    with pytest.raises(receiver.ReceiverError, match="stale"):
        receiver._validate_block_versions(rows, members, amplitude_enabled=False)
    with pytest.raises(receiver.ReceiverError, match="order"):
        receiver._validate_block_versions(["malformed"], members, amplitude_enabled=False)
    amplitude_rows = exporter._block_versions(
        chart_sha256=hashlib.sha256(members["base/chart.ddb"]).hexdigest(),
        semantic_sha256=hashlib.sha256(members["semantic/composed.dds"]).hexdigest(),
        amplitude_enabled=True,
    )
    receiver._validate_block_versions(amplitude_rows, members, amplitude_enabled=True)
    assert amplitude_rows[2]["status"] == "active"
    assert amplitude_rows[2]["version"] == "ddm_D1_pa1_scorer_stat_affine_free.v1"


def test_upstream_harness_pass_requires_archive_raw_exit_and_budget() -> None:
    parsed = {
        "archive_bytes": 10,
        "d_pose": "1.0",
        "d_seg": "0.1",
        "score_rounded": "4.0",
    }
    assert (
        harness._failure_reasons(
            timeout_hit=False,
            returncode=0,
            parse_error=None,
            parsed=parsed,
            expected_archive_bytes=10,
            raw_identity=(20, "a" * 64),
            expected_raw=(20, "a" * 64),
            wallclock=100.0,
            timeout_seconds=1800,
        )
        == []
    )
    assert harness._failure_reasons(
        timeout_hit=True,
        returncode=124,
        parse_error="missing report",
        parsed={},
        expected_archive_bytes=10,
        raw_identity=(0, ""),
        expected_raw=(20, "a" * 64),
        wallclock=1800.0,
        timeout_seconds=1800,
    ) == [
        "harness_timeout",
        "exit_code_124",
        "missing report",
        "upstream_raw_identity_mismatch",
        "wallclock_budget_exceeded",
    ]


def test_manifest_nested_state_and_section_types_fail_closed() -> None:
    members = {
        "manifest.json": b"not-self-hashed",
        "base/chart.ddb": exporter._frame_blob(b"chart", kind=0),
        "semantic/composed.dds": exporter._frame_blob(
            b"\0" * 8,
            kind=1,
            dimensions=(2, 2, 2),
        ),
    }
    manifest = {
        "archive": {
            "source_bytes": 133_941,
            "source_sha256": "a" * 64,
            "state_bytes": 134_211,
            "state_sha256": "b" * 64,
        },
        "block_versions": exporter._block_versions(
            chart_sha256=hashlib.sha256(members["base/chart.ddb"]).hexdigest(),
            semantic_sha256=hashlib.sha256(members["semantic/composed.dds"]).hexdigest(),
        ),
        "chart": {},
        "dependencies": ["torch"],
        "extension_slots": [dict(row) for row in exporter.EXTENSION_SLOTS],
        "false_authority": {
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "research_only": True,
            "score_claim": False,
        },
        "geometry": {
            "camera_hw": [874, 1164],
            "chart_grid_hw": [12, 16],
            "chart_hw": [32, 32],
            "channels": 3,
            "frames_per_pair": 2,
            "pair_count": 600,
            "scorer_hw": [384, 512],
        },
        "language_version": exporter.LANGUAGE_VERSION,
        "output": {},
        "refinement": dict(exporter.REFINEMENT_CONTRACT),
        "schema": exporter.SCHEMA,
        "sections": [
            {
                "bytes": len(members[name]),
                "member": name,
                "sha256": hashlib.sha256(members[name]).hexdigest(),
                "typed_stream_tag": {
                    "schema": "ddm_typed_stream_tag.v1",
                    "type": ("FIBER" if name == "base/chart.ddb" else "SKELETON"),
                    "layer_home": ("L2_chart" if name == "base/chart.ddb" else "L1_program"),
                    "evaluate_py_recursion_level_cited": ("L2_chart -> L3_raster -> L5_verdict"),
                    "counted_bytes": len(members[name]),
                    "free_receiver_code": True,
                },
            }
            for name in exporter.EXPECTED_MEMBERS[1:]
        ],
        "state": {
            "batch_pairs": 16,
            "name": "v15_j2_lane_seed_theta0",
            "receiver_effective_dofs": {
                "island_translation_dofs": 326,
                "lane_program_dofs": 24,
                "shared_template_dofs": 18,
                "total": 368,
            },
        },
    }
    assert receiver._validate_manifest(manifest, members) is manifest
    malformed = dict(manifest)
    malformed["sections"] = ["not-a-section"]
    with pytest.raises(receiver.ReceiverError, match="section order"):
        receiver._validate_manifest(malformed, members)
    malformed = dict(manifest)
    malformed["state"] = {**manifest["state"], "untracked": True}
    with pytest.raises(receiver.ReceiverError, match="state changed"):
        receiver._validate_manifest(malformed, members)
    malformed = json.loads(json.dumps(manifest))
    malformed["sections"][0]["typed_stream_tag"]["counted_bytes"] += 1
    with pytest.raises(receiver.ReceiverError, match="typed-stream custody"):
        receiver._validate_manifest(malformed, members)
    malformed = json.loads(json.dumps(manifest))
    del malformed["sections"][1]["typed_stream_tag"]
    with pytest.raises(receiver.ReceiverError, match="section keys"):
        receiver._validate_manifest(malformed, members)


def test_preserved_stage_is_write_once_and_adoptable(tmp_path) -> None:
    stage = tmp_path / "stage.raw"
    state = tmp_path / "stage.json"
    payload = receiver.torch.tensor(list(b"receiver-stage"), dtype=receiver.torch.uint8)
    first = receiver._write_or_adopt_rendered_stage(
        stage_path=stage,
        state_path=state,
        rendered=payload,
        manifest_sha256="a" * 64,
        start=0,
        stop=1,
    )
    second = receiver._load_preserved_stage(
        stage_path=stage,
        state_path=state,
        manifest_sha256="a" * 64,
        start=0,
        stop=1,
        expected_bytes=len(b"receiver-stage"),
    )
    assert first == second
    assert stage.read_bytes() == b"receiver-stage"

    with pytest.raises(receiver.ReceiverError):
        receiver._load_preserved_stage(
            stage_path=stage,
            state_path=state,
            manifest_sha256="a" * 64,
            start=0,
            stop=1,
            expected_bytes=len(b"receiver-stage") + 1,
        )


def test_config_refuses_non_ssd_proof_root() -> None:
    with pytest.raises(ValueError):
        exporter.DDME1RuntimeExporterConfigV1(
            source_archive_path="source.zip",
            output_directory="packet",
            proof_root="/tmp/proof",
            minimum_free_bytes=8 * 1024 * 1024 * 1024,
        )


def test_four_clause_audit_requires_every_stream_and_ordered_pair() -> None:
    chart_raw = bytes(range(64)) * 4
    semantic_raw = b"\0\1\1\0" * 128
    audit = exporter._rate_doctrine_manifest(
        chart_raw=chart_raw,
        chart_member=exporter._frame_blob(chart_raw, kind=0),
        semantic_raw=semantic_raw,
        semantic_member=exporter._frame_blob(
            semantic_raw,
            kind=1,
            dimensions=(8, 8, 8),
        ),
    )
    assert [row["member"] for row in audit["streams"]] == list(exporter.EXPECTED_MEMBERS[1:])
    assert {(row["conditioner"], row["stream"]) for row in audit["ordered_redundancy_matrix"]} == {
        ("base/chart.ddb", "semantic/composed.dds"),
        ("semantic/composed.dds", "base/chart.ddb"),
    }
    assert all(row["first_rung"] for row in audit["streams"])
    malformed = {**audit, "streams": audit["streams"][:-1]}
    with pytest.raises(exporter.ExporterError, match="missing"):
        exporter._validate_rate_doctrine_manifest(malformed)


def test_e2_frame0_is_structurally_seg_free() -> None:
    anchors = torch.zeros((1, 2, 3), dtype=torch.int16)
    gradients = torch.zeros((1, 2, 2, 3), dtype=torch.int16)
    residuals = torch.zeros((1, 2, 12, 16, 3), dtype=torch.int16)
    labels = torch.ones((1, 384, 512), dtype=torch.uint8)
    palette = torch.tensor([[0, 0, 0], [17, 23, 31]], dtype=torch.uint8)
    rows = torch.div(
        torch.arange(874, dtype=torch.int64) * 384,
        874,
        rounding_mode="floor",
    )
    columns = torch.div(
        torch.arange(1164, dtype=torch.int64) * 512,
        1164,
        rounding_mode="floor",
    )
    rendered = receiver._render_batch(
        start=0,
        stop=1,
        anchors=anchors,
        gradients=gradients,
        residuals=residuals,
        labels=labels,
        palette=palette,
        camera_rows=rows,
        camera_columns=columns,
        semantic_frame_policy="frame1_only_seg_free_frame0",
    )
    assert torch.count_nonzero(rendered[0, 0]) == 0
    assert torch.all(rendered[0, 1] == palette[1])


def test_pa1_decode_derived_affine_reproduces_sealed_arm() -> None:
    count = 29_491_200
    normalized_mean = [
        -1.6709670965020407,
        -1.6709704808928285,
        -1.6709386120242815,
        -1.6709420435556934,
        -0.005337133002923984,
        0.06640471001495353,
        -1.5689619462840858,
        -1.5678262510292515,
        -1.5689084015928298,
        -1.5677425069496225,
        0.0052695566267342495,
        0.044995676322040636,
    ]
    normalized_variance = [
        0.0859960231028586,
        0.08601678941486642,
        0.08599954319058083,
        0.08602016523027242,
        0.0021268988694989914,
        0.0026202825121244976,
        0.18712484740348634,
        0.18701954331976434,
        0.1870017317225781,
        0.18691449772988256,
        0.006838288122042584,
        0.01726789117444064,
    ]
    raw_mean = [value * 63.75 + 127.5 for value in normalized_mean]
    raw_variance = [value * 63.75**2 for value in normalized_variance]
    moments = {
        "count": count,
        "sum": [value * count for value in raw_mean],
        "sum_sq": [(raw_variance[index] + raw_mean[index] ** 2) * count for index in range(12)],
    }
    gain, bias = receiver._derive_pa1_affine(moments)
    assert gain.tolist() == pytest.approx(
        [
            0.03410050645470619,
            4.403048038482666,
            2.40238618850708,
            3.587740182876587,
            4.5472588539123535,
            0.4650801122188568,
            0.023117149248719215,
            0.02312365546822548,
            0.023124758154153824,
            0.023130152374505997,
            2.1682441234588623,
            0.07609924674034119,
        ],
        abs=2e-6,
    )
    assert bias.tolist() == pytest.approx(
        [
            80.24357604980469,
            -11.028668403625488,
            31.031280517578125,
            5.6558756828308105,
            -446.63177490234375,
            62.7425422668457,
            80.5193099975586,
            80.5704574584961,
            80.86619567871094,
            80.27690124511719,
            -145.78759765625,
            114.24369812011719,
        ],
        abs=2e-4,
    )


def test_pa1_frame0_realizer_cannot_change_frame1() -> None:
    generator = torch.Generator().manual_seed(7)
    camera = torch.randint(
        0,
        256,
        (1, 2, 874, 1164, 3),
        dtype=torch.uint8,
        generator=generator,
    )
    gain = torch.tensor(
        [
            0.9,
            1.1,
            0.8,
            1.2,
            0.7,
            1.3,
            0.9,
            1.1,
            0.8,
            1.2,
            0.7,
            1.3,
        ],
        dtype=torch.float32,
    )
    bias = torch.arange(12, dtype=torch.float32) - 6.0
    corrected = receiver._apply_pa1_frame0_affine(camera, gain, bias)
    assert torch.equal(corrected[:, 1], camera[:, 1])
