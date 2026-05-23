# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.hnerv_lowlevel_packer import (
    read_strict_single_member_zip,
    write_stored_single_member_zip,
)
from tac.optimization.inverse_scorer_cell_materializer import (
    CANDIDATE_SCHEMA,
    DESCRIPTOR_SCHEMA,
    IAS1_MAGIC,
    MATERIALIZER_ID,
    RECEIVER_CONTRACT_ID,
    RECEIVER_PROOF_SCHEMA,
    RUNTIME_ADAPTER_SCHEMA,
    TARGET_KIND,
    build_inverse_scorer_cell_candidate_plan,
    build_inverse_scorer_cell_receiver_proof,
    materialize_inverse_scorer_cell_candidate,
    unpack_inverse_scorer_cell_descriptor,
    verify_inverse_scorer_cell_receiver_contract,
)
from tac.repo_io import sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = REPO / "tools" / "materialize_inverse_scorer_cell_candidate.py"
PROOF_SCRIPT = REPO / "tools" / "build_inverse_scorer_cell_receiver_proof.py"


def _action_functional() -> dict[str, object]:
    return {
        "schema": "inverse_steganalysis_discrete_action_functional.v1",
        "tool": "tac.optimization.inverse_steganalysis_acquisition",
        "math_model": {
            "representation": "discrete_riemann_sum_with_second_order_interactions",
            "stationarity_rule": "select positive euler_lagrange_residual cells",
        },
        "integral_totals": {"cell_count": 1, "blocked_cell_count": 0},
        "water_bucket": {
            "schema": "inverse_steganalysis_water_bucket_plan.v1",
            "selected_count": 1,
            "selected_cells": [
                {
                    "atom_id": "inverse_surface_pair0007",
                    "candidate_id": "candidate_pair0007",
                    "scope_axis": "pairs",
                    "component": "posenet",
                    "water_fill_cost_bytes": 32,
                    "expected_score_gain": 0.0004,
                    "euler_lagrange_residual": 0.00039,
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "cells": [],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_template_archive(path: Path) -> None:
    write_stored_single_member_zip(path, member_name="x", payload=b"base-payload")


def test_inverse_scorer_cell_plan_requires_receiver_proof(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    _write_template_archive(template)
    write_json(action, _action_functional())

    plan = build_inverse_scorer_cell_candidate_plan(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        repo_root=tmp_path,
    )

    assert plan["schema"] == "inverse_scorer_cell_candidate_plan_v1"
    assert plan["materializer_id"] == MATERIALIZER_ID
    assert plan["target_kind"] == TARGET_KIND
    assert plan["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_missing" in plan["readiness_blockers"]
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False


def test_materializer_appends_ias1_descriptor_and_keeps_false_authority(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    _write_template_archive(template)
    write_json(action, _action_functional())

    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )

    assert manifest["schema"] == CANDIDATE_SCHEMA
    assert manifest["byte_closed_candidate_emitted"] is True
    assert manifest["candidate_archive"]["sha256"] == sha256_file(output)
    assert manifest["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_missing" in manifest["readiness_blockers"]
    assert "inverse_scorer_cell_receiver_contract_not_satisfied" in manifest[
        "readiness_blockers"
    ]
    assert manifest["archive_diff_manifest"]["candidate_non_noop"] is True
    descriptor_record = manifest["inverse_scorer_cell_descriptor"]
    assert descriptor_record["schema"] == DESCRIPTOR_SCHEMA
    assert descriptor_record["magic"] == "IAS1"
    assert descriptor_record["selected_atom_ids"] == ["inverse_surface_pair0007"]

    materialized = read_strict_single_member_zip(output)
    assert materialized.payload.startswith(b"base-payload")
    packet = materialized.payload[len(b"base-payload") :]
    assert packet.startswith(IAS1_MAGIC)
    descriptor = unpack_inverse_scorer_cell_descriptor(packet)
    assert descriptor["schema"] == DESCRIPTOR_SCHEMA
    assert descriptor["raw_contest_video_digest"] == "a" * 64
    assert descriptor["selected_cells"][0]["atom_id"] == "inverse_surface_pair0007"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False


def test_receiver_contract_accepts_strict_descriptor_consumption_proof(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    _write_template_archive(template)
    write_json(action, _action_functional())
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    proof = {
        "schema": RECEIVER_PROOF_SCHEMA,
        "ready_for_exact_eval_runtime": True,
        "candidate_archive_sha256": manifest["candidate_archive"]["sha256"],
        "candidate_member_sha256": manifest["candidate_archive"]["member_sha256"],
        "descriptor_packet_sha256": manifest["inverse_scorer_cell_descriptor"][
            "packet_sha256"
        ],
        "raw_contest_video_digest": "a" * 64,
        "selected_atom_ids": ["inverse_surface_pair0007"],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }

    verification = verify_inverse_scorer_cell_receiver_contract(
        runtime_consumption_proof=proof,
        required_candidate_archive_sha256=manifest["candidate_archive"]["sha256"],
        required_candidate_member_sha256=manifest["candidate_archive"]["member_sha256"],
        required_descriptor_packet_sha256=manifest["inverse_scorer_cell_descriptor"][
            "packet_sha256"
        ],
        required_raw_contest_video_digest="a" * 64,
        required_selected_atom_ids=["inverse_surface_pair0007"],
    )

    assert verification["receiver_contract_id"] == RECEIVER_CONTRACT_ID
    assert verification["receiver_contract_satisfied"] is True
    assert verification["blockers"] == []
    assert verification["ready_for_exact_eval_dispatch"] is False


def test_receiver_proof_builder_transcodes_runtime_adapter_manifest(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    adapter_manifest = tmp_path / "runtime_adapter.json"
    _write_template_archive(template)
    write_json(action, _action_functional())
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    write_json(candidate_manifest, manifest)
    write_json(
        adapter_manifest,
        {
            "schema": RUNTIME_ADAPTER_SCHEMA,
            "candidate_manifest": {
                "path": candidate_manifest.name,
                "sha256": sha256_file(candidate_manifest),
            },
            "candidate_archive": {
                "sha256": manifest["candidate_archive"]["sha256"],
            },
            "runtime_tree_sha256": "b" * 64,
            "runtime_consumption_probe": {"passed": True},
            "descriptor_consumption": {
                "passed": True,
                "descriptor_packet_sha256": manifest[
                    "inverse_scorer_cell_descriptor"
                ]["packet_sha256"],
            },
            "score_claim": False,
            "dispatch_attempted": False,
        },
    )

    proof = build_inverse_scorer_cell_receiver_proof(
        runtime_adapter_manifest=adapter_manifest,
        repo_root=tmp_path,
    )

    assert proof["schema"] == RECEIVER_PROOF_SCHEMA
    assert proof["ready_for_exact_eval_runtime"] is True
    assert proof["candidate_archive_sha256"] == manifest["candidate_archive"]["sha256"]
    assert proof["descriptor_packet_sha256"] == manifest[
        "inverse_scorer_cell_descriptor"
    ]["packet_sha256"]
    assert proof["selected_atom_ids"] == ["inverse_surface_pair0007"]
    assert proof["blockers"] == []


def test_inverse_scorer_cell_materializer_cli_writes_manifest(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    manifest_out = tmp_path / "manifest.json"
    _write_template_archive(template)
    write_json(action, _action_functional())

    proc = subprocess.run(
        [
            sys.executable,
            str(MATERIALIZE_SCRIPT),
            "--candidate-archive-template",
            str(template),
            "--inverse-action-functional",
            str(action),
            "--raw-contest-video-digest",
            "a" * 64,
            "--output-archive",
            str(output),
            "--manifest-out",
            str(manifest_out),
            "--repo-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["schema"] == CANDIDATE_SCHEMA
    assert output.is_file()
    assert "score_claim" in proc.stdout


def test_inverse_scorer_cell_receiver_proof_cli_writes_json(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    adapter_manifest = tmp_path / "runtime_adapter.json"
    proof_out = tmp_path / "proof.json"
    _write_template_archive(template)
    write_json(action, _action_functional())
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    write_json(candidate_manifest, manifest)
    write_json(
        adapter_manifest,
        {
            "schema": RUNTIME_ADAPTER_SCHEMA,
            "candidate_manifest": {"path": candidate_manifest.name},
            "candidate_archive": {
                "sha256": manifest["candidate_archive"]["sha256"],
            },
            "runtime_consumption_probe": {"passed": True},
            "descriptor_consumption": {
                "passed": True,
                "descriptor_packet_sha256": manifest[
                    "inverse_scorer_cell_descriptor"
                ]["packet_sha256"],
            },
            "score_claim": False,
            "dispatch_attempted": False,
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(PROOF_SCRIPT),
            "--runtime-adapter-manifest",
            str(adapter_manifest),
            "--json-out",
            str(proof_out),
            "--repo-root",
            str(tmp_path),
            "--fail-if-not-ready",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    proof = json.loads(proof_out.read_text(encoding="utf-8"))
    assert proof["schema"] == RECEIVER_PROOF_SCHEMA
    assert proof["ready_for_exact_eval_runtime"] is True
