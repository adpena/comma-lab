# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.hnerv_lowlevel_packer import (
    read_strict_single_member_zip,
    write_stored_single_member_zip,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_SCHEMA,
    build_inverse_scorer_cell_candidate_chain,
)
from tac.optimization.inverse_scorer_cell_inflate_parity import (
    INFLATE_PARITY_PROBE_SCHEMA,
    build_inverse_scorer_cell_inflate_parity_probe,
    build_inverse_scorer_cell_inflate_parity_probe_from_archives,
    verify_inverse_scorer_cell_inflate_parity_probe,
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
    build_inverse_scorer_cell_runtime_adapter_manifest,
    materialize_inverse_scorer_cell_candidate,
    unpack_inverse_scorer_cell_descriptor,
    verify_inverse_scorer_cell_candidate_manifest,
    verify_inverse_scorer_cell_receiver_contract,
)
from tac.repo_io import sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = REPO / "tools" / "materialize_inverse_scorer_cell_candidate.py"
PROOF_SCRIPT = REPO / "tools" / "build_inverse_scorer_cell_receiver_proof.py"
ADAPTER_SCRIPT = REPO / "tools" / "build_inverse_scorer_cell_runtime_adapter_manifest.py"
PARITY_SCRIPT = REPO / "tools" / "build_inverse_scorer_cell_inflate_parity_probe.py"
CHAIN_SCRIPT = REPO / "tools" / "run_inverse_scorer_cell_candidate_chain.py"


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


def _write_output_tree(path: Path, payload: bytes = b"frame-bytes") -> None:
    (path / "frames").mkdir(parents=True)
    (path / "frames" / "000000.raw").write_bytes(payload)


def _write_constant_inflate_runtime(path: Path) -> None:
    path.mkdir(parents=True)
    inflate = path / "inflate.sh"
    inflate.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "OUT=\"$2\"",
                "mkdir -p \"$OUT/frames\"",
                "printf 'frame-bytes' > \"$OUT/frames/000000.raw\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    inflate.chmod(0o755)


def _write_fake_inflate_runtime(path: Path) -> None:
    path.mkdir(parents=True)
    inflate = path / "inflate.sh"
    inflate.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "data_dir=\"$1\"",
                "out_dir=\"$2\"",
                "file_list=\"$3\"",
                "mkdir -p \"$out_dir/frames\"",
                "test -f \"$data_dir/x\"",
                "test -s \"$file_list\"",
                "printf frame-bytes > \"$out_dir/frames/000000.raw\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    inflate.chmod(0o755)


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
    assert "inverse_scorer_cell_receiver_contract_not_satisfied" in manifest["readiness_blockers"]
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
        "ready_for_receiver_verification": True,
        "descriptor_receiver_contract_satisfied": True,
        "ready_for_exact_eval_runtime": False,
        "candidate_archive_sha256": manifest["candidate_archive"]["sha256"],
        "candidate_member_sha256": manifest["candidate_archive"]["member_sha256"],
        "descriptor_packet_sha256": manifest["inverse_scorer_cell_descriptor"]["packet_sha256"],
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
        required_descriptor_packet_sha256=manifest["inverse_scorer_cell_descriptor"]["packet_sha256"],
        required_raw_contest_video_digest="a" * 64,
        required_selected_atom_ids=["inverse_surface_pair0007"],
    )

    assert verification["receiver_contract_id"] == RECEIVER_CONTRACT_ID
    assert verification["receiver_contract_satisfied"] is True
    assert verification["blockers"] == []
    assert verification["ready_for_exact_eval_dispatch"] is False


def test_receiver_contract_rejects_truthy_authority_fields(
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
        "ready_for_receiver_verification": True,
        "descriptor_receiver_contract_satisfied": True,
        "ready_for_exact_eval_runtime": False,
        "ready_for_exact_eval_dispatch": True,
        "score_claim_valid": True,
        "candidate_archive_sha256": manifest["candidate_archive"]["sha256"],
        "candidate_member_sha256": manifest["candidate_archive"]["member_sha256"],
        "descriptor_packet_sha256": manifest["inverse_scorer_cell_descriptor"]["packet_sha256"],
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
        required_descriptor_packet_sha256=manifest["inverse_scorer_cell_descriptor"]["packet_sha256"],
        required_raw_contest_video_digest="a" * 64,
        required_selected_atom_ids=["inverse_surface_pair0007"],
    )

    assert verification["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_has_truthy_authority_field" in verification["blockers"]


def test_raw_digest_mapping_without_sha_fails_closed(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    _write_template_archive(template)
    write_json(action, _action_functional())

    with pytest.raises(ValueError, match="raw_contest_video_digest"):
        materialize_inverse_scorer_cell_candidate(
            raw_contest_video_digest={"path": "upstream/videos/0.mkv"},
            candidate_archive_template=template,
            inverse_action_functional=action,
            output_archive=output,
            repo_root=tmp_path,
        )


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
    adapter = build_inverse_scorer_cell_runtime_adapter_manifest(
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
        runtime_source_paths=[__file__],
    )
    write_json(adapter_manifest, adapter)

    proof = build_inverse_scorer_cell_receiver_proof(
        runtime_adapter_manifest=adapter_manifest,
        repo_root=tmp_path,
    )

    assert proof["schema"] == RECEIVER_PROOF_SCHEMA
    assert proof["ready_for_receiver_verification"] is True
    assert proof["descriptor_receiver_contract_satisfied"] is True
    assert proof["ready_for_exact_eval_runtime"] is False
    assert proof["candidate_archive_sha256"] == manifest["candidate_archive"]["sha256"]
    assert proof["descriptor_packet_sha256"] == manifest["inverse_scorer_cell_descriptor"]["packet_sha256"]
    assert proof["selected_atom_ids"] == ["inverse_surface_pair0007"]
    assert proof["blockers"] == []


def test_runtime_adapter_manifest_consumes_ias1_packet_and_verifies_candidate(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
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

    adapter = build_inverse_scorer_cell_runtime_adapter_manifest(
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
        runtime_source_paths=[__file__],
    )
    proof = build_inverse_scorer_cell_receiver_proof(
        runtime_adapter_manifest=adapter,
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
    )
    verified = verify_inverse_scorer_cell_candidate_manifest(
        candidate_manifest=candidate_manifest,
        runtime_consumption_proof=proof,
        repo_root=tmp_path,
    )

    assert adapter["schema"] == RUNTIME_ADAPTER_SCHEMA
    assert adapter["runtime_consumption_probe"]["passed"] is True
    assert adapter["descriptor_consumption"]["passed"] is True
    assert adapter["readiness_blockers"] == []
    assert adapter["ready_for_descriptor_receiver"] is True
    assert adapter["ready_for_exact_eval_runtime"] is False
    assert proof["ready_for_receiver_verification"] is True
    assert proof["ready_for_exact_eval_runtime"] is False
    assert verified["receiver_contract_satisfied"] is True
    assert "runtime_consumption_proof_missing" not in verified["readiness_blockers"]
    assert "inverse_scorer_cell_receiver_contract_not_satisfied" not in verified["readiness_blockers"]
    assert "candidate_inflate_output_parity_missing" in verified["readiness_blockers"]
    assert verified["score_claim"] is False
    assert verified["ready_for_exact_eval_dispatch"] is False


def test_inflate_parity_probe_clears_only_candidate_parity_blocker(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    source_out = tmp_path / "source_out"
    candidate_out = tmp_path / "candidate_out"
    _write_template_archive(template)
    write_json(action, _action_functional())
    _write_output_tree(source_out)
    _write_output_tree(candidate_out)
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    write_json(candidate_manifest, manifest)
    adapter = build_inverse_scorer_cell_runtime_adapter_manifest(
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
        runtime_source_paths=[__file__],
    )
    proof = build_inverse_scorer_cell_receiver_proof(
        runtime_adapter_manifest=adapter,
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
    )
    parity = build_inverse_scorer_cell_inflate_parity_probe(
        candidate_manifest=candidate_manifest,
        source_output_dir=source_out,
        candidate_output_dir=candidate_out,
        repo_root=tmp_path,
    )
    verified = verify_inverse_scorer_cell_candidate_manifest(
        candidate_manifest=candidate_manifest,
        runtime_consumption_proof=proof,
        inflate_parity_probe=parity,
        repo_root=tmp_path,
    )

    assert parity["schema"] == INFLATE_PARITY_PROBE_SCHEMA
    assert parity["full_frame_inflate_output_parity_claim"] is True
    assert parity["inverse_scorer_cell_descriptor"]["packet_sha256"] == (
        manifest["inverse_scorer_cell_descriptor"]["packet_sha256"]
    )
    assert verified["inflate_parity_satisfied"] is True
    assert "candidate_inflate_output_parity_missing" not in verified["readiness_blockers"]
    assert "exact_auth_eval_required_before_score_claim" in verified["readiness_blockers"]
    assert verified["score_claim"] is False
    assert verified["promotion_eligible"] is False
    assert verified["rank_or_kill_eligible"] is False
    assert verified["ready_for_exact_eval_dispatch"] is False


def test_inflate_parity_probe_from_archives_runs_actual_inflate(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    runtime_dir = tmp_path / "runtime"
    _write_template_archive(template)
    _write_fake_inflate_runtime(runtime_dir)
    write_json(action, _action_functional())
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    write_json(candidate_manifest, manifest)

    parity = build_inverse_scorer_cell_inflate_parity_probe_from_archives(
        candidate_manifest=candidate_manifest,
        inflate_runtime_dir=runtime_dir,
        repo_root=tmp_path,
        timeout_seconds=30,
    )
    verification = verify_inverse_scorer_cell_inflate_parity_probe(
        candidate_manifest=candidate_manifest,
        inflate_parity_probe=parity,
        repo_root=tmp_path,
    )

    assert parity["schema"] == INFLATE_PARITY_PROBE_SCHEMA
    assert parity["source_inflate_run"]["returncode"] == 0
    assert parity["candidate_inflate_run"]["returncode"] == 0
    assert parity["source_inflate_run"]["full_frame_file_list_claim"] is True
    assert parity["full_frame_inflate_output_parity_claim"] is True
    assert parity["work_dir_retained"] is False
    assert verification["inflate_parity_satisfied"] is True
    assert parity["score_claim"] is False
    assert parity["ready_for_exact_eval_dispatch"] is False


def test_inflate_parity_probe_fails_closed_on_output_mismatch(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    source_out = tmp_path / "source_out"
    candidate_out = tmp_path / "candidate_out"
    _write_template_archive(template)
    write_json(action, _action_functional())
    _write_output_tree(source_out, b"source")
    _write_output_tree(candidate_out, b"candidate")
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    write_json(candidate_manifest, manifest)

    parity = build_inverse_scorer_cell_inflate_parity_probe(
        candidate_manifest=candidate_manifest,
        source_output_dir=source_out,
        candidate_output_dir=candidate_out,
        repo_root=tmp_path,
    )
    verification = verify_inverse_scorer_cell_inflate_parity_probe(
        candidate_manifest=candidate_manifest,
        inflate_parity_probe=parity,
        repo_root=tmp_path,
    )

    assert parity["full_frame_inflate_output_parity_claim"] is False
    assert "inflate_output_bytes_not_identical" in parity["blockers"]
    assert verification["inflate_parity_satisfied"] is False
    assert "candidate_inflate_output_parity_missing" in verification["blockers"]


def test_inflate_parity_probe_rejects_descriptor_sha_mismatch_and_truthy_authority(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    source_out = tmp_path / "source_out"
    candidate_out = tmp_path / "candidate_out"
    _write_template_archive(template)
    write_json(action, _action_functional())
    _write_output_tree(source_out)
    _write_output_tree(candidate_out)
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    write_json(candidate_manifest, manifest)
    parity = build_inverse_scorer_cell_inflate_parity_probe(
        candidate_manifest=candidate_manifest,
        source_output_dir=source_out,
        candidate_output_dir=candidate_out,
        repo_root=tmp_path,
    )
    parity["inverse_scorer_cell_descriptor"]["packet_sha256"] = "0" * 64
    parity["nested"] = {"ready_for_exact_eval_dispatch": True}

    verification = verify_inverse_scorer_cell_inflate_parity_probe(
        candidate_manifest=candidate_manifest,
        inflate_parity_probe=parity,
        repo_root=tmp_path,
    )

    assert verification["inflate_parity_satisfied"] is False
    assert "inflate_parity_descriptor_packet_sha_mismatch" in verification["blockers"]
    assert "inflate_parity_probe_has_truthy_authority_field" in verification["blockers"]


def test_inflate_parity_probe_records_missing_artifact_blockers(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
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

    parity = build_inverse_scorer_cell_inflate_parity_probe(
        candidate_manifest=candidate_manifest,
        source_output_dir=tmp_path / "missing_source_out",
        candidate_output_dir=tmp_path / "missing_candidate_out",
        repo_root=tmp_path,
    )

    assert parity["full_frame_inflate_output_parity_claim"] is False
    assert "source_inflate_output_dir_missing" in parity["blockers"]
    assert "candidate_inflate_output_dir_missing" in parity["blockers"]


def test_inflate_parity_probe_from_archives_runs_shell_runtime(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    runtime_dir = tmp_path / "runtime"
    work_dir = tmp_path / "parity_work"
    _write_template_archive(template)
    _write_constant_inflate_runtime(runtime_dir)
    write_json(action, _action_functional())
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    write_json(candidate_manifest, manifest)

    parity = build_inverse_scorer_cell_inflate_parity_probe_from_archives(
        candidate_manifest=candidate_manifest,
        inflate_runtime_dir=runtime_dir,
        source_archive=template,
        candidate_archive=output,
        repo_root=tmp_path,
        work_dir=work_dir,
    )

    assert parity["full_frame_inflate_output_parity_claim"] is True
    assert parity["source_inflate_run"]["returncode"] == 0
    assert parity["candidate_inflate_run"]["returncode"] == 0
    assert parity["inflate_runtime"]["inflate_sh_sha256"]
    assert parity["work_dir_retained"] is False
    assert not work_dir.exists()


def test_runtime_adapter_manifest_fails_closed_on_descriptor_sha_mismatch(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_template_archive(template)
    write_json(action, _action_functional())
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    manifest["inverse_scorer_cell_descriptor"]["packet_sha256"] = "0" * 64
    write_json(candidate_manifest, manifest)

    adapter = build_inverse_scorer_cell_runtime_adapter_manifest(
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
        runtime_source_paths=[__file__],
    )
    proof = build_inverse_scorer_cell_receiver_proof(
        runtime_adapter_manifest=adapter,
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
    )

    assert adapter["ready_for_exact_eval_runtime"] is False
    assert adapter["runtime_consumption_probe"]["passed"] is False
    assert adapter["descriptor_consumption"]["passed"] is False
    assert "descriptor_packet_sha_mismatch" in adapter["readiness_blockers"]
    assert proof["ready_for_exact_eval_runtime"] is False
    assert "runtime_adapter_manifest_has_blockers" in proof["blockers"]
    assert proof["score_claim"] is False


def test_runtime_adapter_manifest_fails_closed_on_invalid_descriptor_count(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_template_archive(template)
    write_json(action, _action_functional())
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    manifest["inverse_scorer_cell_descriptor"]["selected_cell_count"] = "not-an-int"
    write_json(candidate_manifest, manifest)

    adapter = build_inverse_scorer_cell_runtime_adapter_manifest(
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
        runtime_source_paths=[__file__],
    )

    assert adapter["ready_for_exact_eval_runtime"] is False
    assert adapter["runtime_consumption_probe"]["passed"] is False
    assert "descriptor_selected_cell_count_invalid" in adapter["readiness_blockers"]
    assert adapter["score_claim"] is False


def test_runtime_adapter_manifest_resolves_archive_relative_to_manifest(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    manifest_dir = tmp_path / "manifest_dir"
    repo.mkdir()
    manifest_dir.mkdir()
    template = repo / "template.zip"
    action = repo / "action.json"
    output = manifest_dir / "candidate.zip"
    candidate_manifest = manifest_dir / "candidate_manifest.json"
    _write_template_archive(template)
    write_json(action, _action_functional())
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=repo,
    )
    manifest["candidate_archive"]["path"] = "candidate.zip"
    write_json(candidate_manifest, manifest)

    adapter = build_inverse_scorer_cell_runtime_adapter_manifest(
        candidate_manifest=candidate_manifest,
        repo_root=repo,
        runtime_source_paths=[__file__],
    )

    assert adapter["runtime_consumption_probe"]["passed"] is True
    assert adapter["descriptor_consumption"]["passed"] is True
    assert adapter["candidate_archive"]["member_sha256"] == manifest["candidate_archive"]["member_sha256"]


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


def test_inverse_scorer_cell_runtime_adapter_cli_writes_manifest(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    adapter_out = tmp_path / "runtime_adapter.json"
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

    proc = subprocess.run(
        [
            sys.executable,
            str(ADAPTER_SCRIPT),
            "--candidate-manifest",
            str(candidate_manifest),
            "--json-out",
            str(adapter_out),
            "--repo-root",
            str(tmp_path),
            "--runtime-source-path",
            str(__file__),
            "--fail-if-blocked",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    adapter = json.loads(adapter_out.read_text(encoding="utf-8"))
    assert adapter["schema"] == RUNTIME_ADAPTER_SCHEMA
    assert adapter["runtime_consumption_probe"]["passed"] is True
    assert (
        adapter["descriptor_consumption"]["descriptor_packet_sha256"]
        == manifest["inverse_scorer_cell_descriptor"]["packet_sha256"]
    )


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
    adapter = build_inverse_scorer_cell_runtime_adapter_manifest(
        candidate_manifest=candidate_manifest,
        repo_root=tmp_path,
        runtime_source_paths=[__file__],
    )
    write_json(adapter_manifest, adapter)

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
    assert proof["ready_for_receiver_verification"] is True
    assert proof["ready_for_exact_eval_runtime"] is False


def test_inverse_scorer_cell_inflate_parity_cli_writes_json(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output = tmp_path / "candidate.zip"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    source_out = tmp_path / "source_out"
    candidate_out = tmp_path / "candidate_out"
    parity_out = tmp_path / "inflate_parity.json"
    _write_template_archive(template)
    write_json(action, _action_functional())
    _write_output_tree(source_out)
    _write_output_tree(candidate_out)
    manifest = materialize_inverse_scorer_cell_candidate(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_archive=output,
        repo_root=tmp_path,
    )
    write_json(candidate_manifest, manifest)

    proc = subprocess.run(
        [
            sys.executable,
            str(PARITY_SCRIPT),
            "--candidate-manifest",
            str(candidate_manifest),
            "--source-output-dir",
            str(source_out),
            "--candidate-output-dir",
            str(candidate_out),
            "--json-out",
            str(parity_out),
            "--repo-root",
            str(tmp_path),
            "--fail-if-blocked",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    parity = json.loads(parity_out.read_text(encoding="utf-8"))
    assert parity["schema"] == INFLATE_PARITY_PROBE_SCHEMA
    assert parity["full_frame_inflate_output_parity_claim"] is True
    assert parity["score_claim"] is False


def test_inverse_scorer_cell_chain_clears_receiver_blockers_only(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    _write_template_archive(template)
    write_json(action, _action_functional())

    chain = build_inverse_scorer_cell_candidate_chain(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_dir=tmp_path / "chain",
        selected_limit=1,
        repo_root=tmp_path,
    )

    assert chain["schema"] == CHAIN_SCHEMA
    assert chain["byte_closed_candidate_emitted"] is True
    assert chain["runtime_adapter_ready"] is True
    assert chain["receiver_proof_ready"] is True
    assert chain["receiver_contract_satisfied"] is True
    assert chain["candidate_runtime_adapter_blocker_cleared"] is True
    assert "runtime_consumption_proof_missing" not in chain["readiness_blockers"]
    assert "inverse_scorer_cell_receiver_contract_not_satisfied" not in chain["readiness_blockers"]
    assert "candidate_inflate_output_parity_missing" in chain["readiness_blockers"]
    assert "exact_auth_eval_required_before_score_claim" in chain["readiness_blockers"]
    assert chain["score_claim"] is False
    assert chain["ready_for_exact_eval_dispatch"] is False


def test_inverse_scorer_cell_chain_can_attach_inflate_parity_probe(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    source_out = tmp_path / "source_out"
    candidate_out = tmp_path / "candidate_out"
    _write_template_archive(template)
    write_json(action, _action_functional())
    _write_output_tree(source_out)
    _write_output_tree(candidate_out)

    chain = build_inverse_scorer_cell_candidate_chain(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_dir=tmp_path / "chain_with_parity",
        selected_limit=1,
        repo_root=tmp_path,
        source_inflate_output_dir=source_out,
        candidate_inflate_output_dir=candidate_out,
    )

    assert chain["receiver_contract_satisfied"] is True
    assert chain["inflate_parity_satisfied"] is True
    assert "inflate_parity_probe" in chain["artifacts"]
    assert "candidate_inflate_output_parity_missing" not in chain["readiness_blockers"]
    assert "exact_auth_eval_required_before_score_claim" in chain["readiness_blockers"]
    assert any(
        step["step_id"] == "build_inflate_parity_probe"
        and step["full_frame_inflate_output_parity_claim"] is True
        for step in chain["chain_steps"]
    )
    assert chain["ready_for_exact_eval_dispatch"] is False


def test_inverse_scorer_cell_chain_can_run_inflate_runtime_parity(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    runtime_dir = tmp_path / "runtime"
    _write_template_archive(template)
    _write_fake_inflate_runtime(runtime_dir)
    write_json(action, _action_functional())

    chain = build_inverse_scorer_cell_candidate_chain(
        raw_contest_video_digest="a" * 64,
        candidate_archive_template=template,
        inverse_action_functional=action,
        output_dir=tmp_path / "chain_with_runtime_parity",
        selected_limit=1,
        repo_root=tmp_path,
        inflate_runtime_dir=runtime_dir,
        inflate_timeout_seconds=30,
    )

    assert chain["receiver_contract_satisfied"] is True
    assert chain["inflate_parity_satisfied"] is True
    assert "candidate_inflate_output_parity_missing" not in chain["readiness_blockers"]
    assert "exact_auth_eval_required_before_score_claim" in chain["readiness_blockers"]
    assert chain["artifacts"]["inflate_parity_probe"]["path"].endswith("inflate_parity_probe.json")
    assert chain["ready_for_exact_eval_dispatch"] is False


def test_inverse_scorer_cell_chain_writes_failure_manifest_on_partial_failure(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output_dir = tmp_path / "chain_failure"
    _write_template_archive(template)
    write_json(action, _action_functional())

    with pytest.raises(ValueError):
        build_inverse_scorer_cell_candidate_chain(
            raw_contest_video_digest="a" * 64,
            candidate_archive_template=template,
            inverse_action_functional=action,
            output_dir=output_dir,
            selected_limit=1,
            repo_root=tmp_path,
            min_free_bytes=10**30,
        )

    failure = json.loads((output_dir / "inverse_scorer_cell_candidate_chain_manifest.json").read_text(encoding="utf-8"))
    assert failure["schema"] == CHAIN_SCHEMA
    assert failure["status"] == "failed"
    assert "inverse_scorer_cell_candidate_chain_failed" in failure["readiness_blockers"]
    assert failure["score_claim"] is False


def test_inverse_scorer_cell_chain_cli_writes_tool_manifest(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zip"
    action = tmp_path / "action.json"
    output_dir = tmp_path / "chain_cli"
    _write_template_archive(template)
    write_json(action, _action_functional())

    proc = subprocess.run(
        [
            sys.executable,
            str(CHAIN_SCRIPT),
            "--candidate-archive-template",
            str(template),
            "--inverse-action-functional",
            str(action),
            "--raw-contest-video-digest",
            "a" * 64,
            "--output-dir",
            str(output_dir),
            "--selected-limit",
            "1",
            "--fail-if-receiver-blocked",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    chain = json.loads((output_dir / "inverse_scorer_cell_candidate_chain_manifest.json").read_text(encoding="utf-8"))
    assert chain["schema"] == CHAIN_SCHEMA
    assert chain["receiver_contract_satisfied"] is True
    assert chain["tool_run_manifest"]["ready_for_exact_eval_dispatch"] is False
