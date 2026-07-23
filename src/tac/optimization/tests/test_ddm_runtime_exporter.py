# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import pytest

from tac.optimization import ddm_runtime_exporter as exporter
from tac.optimization import ddm_runtime_receiver as receiver
from tools import rehearse_ddm_runtime_upstream as harness


def test_blob_frame_roundtrip_and_terminal_tamper_refusal() -> None:
    raw = bytes(range(251)) * 7
    framed = exporter._frame_blob(raw, kind=1, dimensions=(7, 251))
    decoded, shape = receiver._parse_blob(
        framed, expected_kind=1, label="synthetic"
    )
    assert decoded == raw
    assert shape == (7, 251)

    tampered = bytearray(framed)
    tampered[-1] ^= 1
    with pytest.raises(receiver.ReceiverError):
        receiver._parse_blob(
            bytes(tampered), expected_kind=1, label="synthetic"
        )


def test_deterministic_zip_has_bijective_byte_homes() -> None:
    members = {
        "manifest.json": b"{}",
        "base/chart.ddb": exporter._frame_blob(b"chart", kind=0),
        "semantic/composed.dds": exporter._frame_blob(
            b"\0" * 8, kind=1, dimensions=(2, 2, 2)
        ),
    }
    first = exporter._deterministic_zip(members)
    second = exporter._deterministic_zip(members)
    homes = exporter._zip_home_ledger(first)
    assert first == second
    assert sum(int(row["home_bytes"]) for row in homes) == len(first)


def test_runtime_cleanliness_allows_only_generic_dependencies() -> None:
    runtime = exporter.RUNTIME_SOURCE.read_bytes()
    row = exporter._runtime_cleanliness(runtime)
    assert row["status"] == "PASS"
    assert row["allowed_dependency_roots"] == ["torch", "brotli"]
    assert row["runtime_sha256"] == hashlib.sha256(runtime).hexdigest()
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
    assert all(
        row["rate_custody"] == "independent_member_bytes_sha256"
        for row in exporter.EXTENSION_SLOTS
    )
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
        semantic_sha256=hashlib.sha256(
            members["semantic/composed.dds"]
        ).hexdigest(),
    )
    receiver._validate_block_versions(rows, members)
    rows[0]["input_members"][0]["sha256"] = "0" * 64
    with pytest.raises(receiver.ReceiverError, match="stale"):
        receiver._validate_block_versions(rows, members)
    with pytest.raises(receiver.ReceiverError, match="order"):
        receiver._validate_block_versions(["malformed"], members)


def test_upstream_harness_pass_requires_archive_raw_exit_and_budget() -> None:
    parsed = {
        "archive_bytes": 10,
        "d_pose": "1.0",
        "d_seg": "0.1",
        "score_rounded": "4.0",
    }
    assert harness._failure_reasons(
        timeout_hit=False,
        returncode=0,
        parse_error=None,
        parsed=parsed,
        expected_archive_bytes=10,
        raw_identity=(20, "a" * 64),
        expected_raw=(20, "a" * 64),
        wallclock=100.0,
        timeout_seconds=1800,
    ) == []
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
        "base/chart.ddb": b"chart",
        "semantic/composed.dds": b"semantic",
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
            semantic_sha256=hashlib.sha256(
                members["semantic/composed.dds"]
            ).hexdigest(),
        ),
        "chart": {},
        "dependencies": ["torch", "brotli"],
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


def test_preserved_stage_is_write_once_and_adoptable(tmp_path) -> None:
    stage = tmp_path / "stage.raw"
    state = tmp_path / "stage.json"
    payload = receiver.torch.tensor(
        list(b"receiver-stage"), dtype=receiver.torch.uint8
    )
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
