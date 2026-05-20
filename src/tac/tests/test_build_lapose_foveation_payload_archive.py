# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

from tac.analysis.lapose_foveation_atoms import build_foveation_transport_atom_manifest
from tac.analysis.lapose_foveation_payload import (
    PAYLOAD_MEMBER,
    build_lapose_foveation_tuple_payload_artifact,
    pack_lapose_foveation_tuple_payload,
)
from tac.lapose_foveation_payload_candidate import (
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    MEMBER_ORDER,
    RUNTIME_LOADER_PARITY_CONTRACT,
    RUNTIME_PROOF_SKELETON_CONTRACT,
    audit_lapose_foveation_payload_candidate,
    build_lapose_foveation_payload_archive_candidate,
)
from tac.lapose_foveation_runtime_skeleton import (
    FOVEATION_PARAMS_MEMBER,
    LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT,
    PROOF_MEMBER,
    RUNTIME_EFFECT_CONTROLS_CONTRACT,
    RUNTIME_SCORER_VISIBLE_BRIDGE_CONTRACT,
    apply_lfv1_to_rgb_frames,
    build_inflate_adapter_byte_output_control_report,
    build_lfv1_foveation_params_bridge_report,
    build_runtime_effect_control_report,
    build_scorer_visible_byte_output_control_report,
    build_scorer_visible_frame_warp_control_report,
    load_foveation_params,
    rgb_frames_to_rgb24_bytes,
)
from tac.repo_io import read_json, sha256_bytes, sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]


def test_build_lapose_foveation_payload_archive_is_byte_closed_and_fail_closed(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "lapose_foveation_tuples.lfv1"
    source_readiness_path = tmp_path / "lfv1_readiness.json"
    source_readiness = build_lapose_foveation_tuple_payload_artifact(
        _manifest(),
        payload_path=payload_path,
        repo_root=tmp_path,
        max_atoms=2,
    )
    write_json(source_readiness_path, source_readiness)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    for out_dir in (out_a, out_b):
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "build_lapose_foveation_payload_archive.py"),
                "--out-dir",
                str(out_dir),
                "--lfv1-payload",
                str(payload_path),
                "--source-readiness-json",
                str(source_readiness_path),
            ],
            check=True,
            cwd=REPO,
            text=True,
        )

    archive_a = out_a / "archive.zip"
    archive_b = out_b / "archive.zip"
    assert sha256_file(archive_a) == sha256_file(archive_b)

    candidate = read_json(out_a / "candidate.json")
    readiness = audit_lapose_foveation_payload_candidate(
        candidate,
        repo_root=REPO,
        manifest_dir=out_a,
    )
    readiness_file = read_json(out_a / "readiness.json")
    summary = read_json(out_a / "summary.json")

    assert candidate["candidate_manifest_contract"] == CANDIDATE_MANIFEST_CONTRACT
    assert candidate["score_claim"] is False
    assert candidate["dispatch_attempted"] is False
    assert candidate["ready_for_exact_eval_dispatch"] is False
    assert candidate["lfv1_payload"]["member"] == PAYLOAD_MEMBER
    assert candidate["lfv1_payload"]["bytes"] == payload_path.stat().st_size
    assert candidate["lfv1_payload"]["sha256"] == sha256_file(payload_path)
    assert candidate["runtime_loader_parity"]["runtime_loader_parity_contract"] == (
        RUNTIME_LOADER_PARITY_CONTRACT
    )
    assert readiness["payload_member_proven"] is True
    assert readiness["byte_closed_local_archive"] is True
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["candidate_archive"]["untracked_members"] == []
    assert readiness["candidate_archive"]["member_order_matches_manifest"] is True
    assert readiness["candidate_archive"]["zip_determinism_contract"]["passed"] is True
    assert readiness["archive_member_manifest"]["member_order_matches_manifest"] is True
    assert readiness["runtime_loader_parity"]["sidecar_free"] is True
    assert readiness["runtime_loader_parity"]["accepted"] is False
    assert readiness["lfv1_payload_decode"]["accepted"] is True
    assert readiness["runtime_effect_controls"]["accepted"] is True
    assert readiness["runtime_effect_controls"]["contract"] == RUNTIME_EFFECT_CONTROLS_CONTRACT
    frame_warp_control = readiness["runtime_effect_controls"][
        "scorer_visible_frame_warp_control"
    ]
    assert frame_warp_control["contract"] == (
        "lapose_foveation_scorer_visible_frame_warp_control_v1"
    )
    assert frame_warp_control["passed"] is True
    assert frame_warp_control["identity_control"]["passed"] is True
    assert frame_warp_control["nonidentity_control"]["passed"] is True
    assert frame_warp_control["nonidentity_control"]["output_changed"] is True
    byte_output_control = readiness["runtime_effect_controls"][
        "scorer_visible_byte_output_control"
    ]
    assert byte_output_control["contract"] == (
        "lapose_foveation_scorer_visible_byte_output_control_v1"
    )
    assert byte_output_control["passed"] is True
    assert byte_output_control["identity_byte_output_control"]["byte_exact"] is True
    assert byte_output_control["nonidentity_byte_output_control"]["output_changed"] is True
    inflate_adapter_control = readiness["runtime_effect_controls"][
        "inflate_adapter_byte_output_control"
    ]
    assert inflate_adapter_control["contract"] == (
        "lapose_foveation_local_rgb24_inflate_adapter_control_v1"
    )
    assert inflate_adapter_control["passed"] is True
    assert inflate_adapter_control["identity_adapter_control"]["byte_exact"] is True
    assert inflate_adapter_control["nonidentity_adapter_control"]["output_changed"] is True
    assert readiness["runtime_effect_controls"]["scored_runtime_output_parity"][
        "local_frame_warp_control_passed"
    ] is True
    assert readiness["runtime_effect_controls"]["scored_runtime_output_parity"][
        "local_byte_output_control_passed"
    ] is True
    assert readiness["runtime_effect_controls"]["scored_runtime_output_parity"][
        "local_inflate_adapter_control_passed"
    ] is True
    assert candidate["lfv1_foveation_params_bridge"]["contract"] == (
        LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT
    )
    assert candidate["lfv1_foveation_params_bridge"]["passed"] is True
    assert candidate["lfv1_foveation_params_bridge"]["target_member"] == FOVEATION_PARAMS_MEMBER
    decoded_rows = candidate["lfv1_payload"]["decoded"]["rows"]
    expected_target_frame_count = 2 * max(row["pair_index"] for row in decoded_rows) + 2
    assert candidate["lfv1_foveation_params_bridge"]["target_frame_count"] == (
        expected_target_frame_count
    )
    assert candidate["lfv1_foveation_params_bridge"]["pair_to_frame_policy"] == (
        "contest_pair_maps_to_frames_2k_and_2k_plus_1"
    )
    assert [
        row["target_frame_indices"]
        for row in candidate["lfv1_foveation_params_bridge"]["applied_rows"]
    ] == [[2 * row["pair_index"], 2 * row["pair_index"] + 1] for row in decoded_rows]
    assert readiness["lfv1_foveation_params_bridge"]["accepted"] is True
    assert readiness["lfv1_foveation_params_bridge"]["passed"] is True
    assert readiness["lfv1_foveation_params_bridge"]["derived_bytes_match"] is True
    assert candidate["scorer_visible_bridge"]["contract"] == RUNTIME_SCORER_VISIBLE_BRIDGE_CONTRACT
    assert candidate["scorer_visible_bridge"]["bridge_path_present"] is True
    assert readiness["scorer_visible_bridge"]["accepted"] is True
    assert readiness["scorer_visible_bridge"]["bridge_path_present"] is True
    assert readiness["scorer_visible_bridge"]["archive_member_groups"][
        "mask_or_segmentation_stream"
    ]["present"] is False
    assert readiness["scorer_visible_bridge"]["archive_member_groups"][
        "renderer_or_segmap_runtime"
    ]["present"] is False
    assert readiness["scorer_visible_bridge"]["archive_member_groups"][
        "pose_or_geometry_stream"
    ]["present"] is True
    assert readiness["runtime_consumption_audit"]["structural_runtime_consumption"]["passed"] is True
    assert readiness["runtime_consumption_audit"]["scored_runtime_output_parity"]["passed"] is False
    assert readiness["runtime_consumption_audit"]["scorer_visible_bridge"][
        "bridge_path_present"
    ] is True
    assert readiness_file["tool_run_manifest"]["tool"] == (
        "tools/build_lapose_foveation_payload_archive.py"
    )
    assert readiness_file["ready_for_exact_eval_dispatch"] is False

    blockers = set(readiness["dispatch_blockers"])
    assert "runtime_loader_parity_not_passed" in blockers
    assert "lapose_foveation_scorer_visible_output_parity_not_proven" in blockers
    assert "lapose_foveation_runtime_output_parity_not_proven" in blockers
    assert "exact_cuda_auth_eval_missing" in blockers
    assert not any(blocker.startswith("no_op_control_not_passed:") for blocker in blockers)
    assert not any(blocker.startswith("runtime_effect_controls_") for blocker in blockers)
    assert not any(blocker.startswith("scorer_visible_bridge_") for blocker in blockers)
    assert readiness["no_op_controls"]["failed_controls"] == []
    assert set(readiness["no_op_controls"]["passed_controls"]) == {
        "lfv1_identity_decode_control",
        "lfv1_tuple_mutation_runtime_output_control",
        "charged_member_presence_control",
        "runtime_consumes_foveation_tuple_control",
        "scorer_visible_frame_warp_control",
        "scorer_visible_byte_output_control",
        "inflate_adapter_byte_output_control",
    }

    archive_member_manifest = read_json(out_a / "archive_member_manifest.json")
    assert archive_member_manifest["archive_member_manifest_contract"] == (
        ARCHIVE_MEMBER_MANIFEST_CONTRACT
    )
    assert archive_member_manifest["member_order"] == list(MEMBER_ORDER)
    assert archive_member_manifest["member_count"] == len(MEMBER_ORDER)
    payload_proof = next(
        proof
        for proof in readiness["archive_member_manifest"]["member_sha256_proofs"]
        if proof["name"] == PAYLOAD_MEMBER
    )
    assert payload_proof["manifest_bytes"] == payload_path.stat().st_size
    assert payload_proof["actual_sha256"] == sha256_file(payload_path)
    assert payload_proof["sha256_match"] is True

    with zipfile.ZipFile(archive_a) as archive:
        assert archive.namelist() == list(MEMBER_ORDER)
        foveation_params = archive.read(FOVEATION_PARAMS_MEMBER)
        bridge = build_lfv1_foveation_params_bridge_report(
            archive.read(PAYLOAD_MEMBER),
            foveation_params,
        )
        proof_skeleton = json.loads(archive.read(PROOF_MEMBER).decode("utf-8"))
    assert foveation_params[:4] == b"HFV1"
    magic, n_frames, height, width = struct.Struct("<4sIII").unpack(foveation_params[:16])
    assert magic == b"HFV1"
    assert n_frames == expected_target_frame_count
    assert (height, width) == (200, 320)
    assert bridge["passed"] is True
    assert bridge["pair_to_frame_policy"] == "contest_pair_maps_to_frames_2k_and_2k_plus_1"
    assert proof_skeleton["runtime_consumer_proof_skeleton_contract"] == (
        RUNTIME_PROOF_SKELETON_CONTRACT
    )
    assert proof_skeleton["ready_for_exact_eval_dispatch"] is False
    assert proof_skeleton["proof_status"]["archive_contains_payload_and_runtime"] is True
    assert proof_skeleton["proof_status"]["runtime_output_parity"] is False
    assert proof_skeleton["proof_status"]["structural_runtime_consumption"] is True
    assert proof_skeleton["proof_status"]["scorer_visible_frame_warp_control"] is True
    assert proof_skeleton["proof_status"]["scorer_visible_byte_output_control"] is True
    assert proof_skeleton["proof_status"]["local_rgb24_inflate_adapter_control"] is True
    assert proof_skeleton["proof_status"]["lfv1_to_foveation_params_bridge"] is True
    assert proof_skeleton["proof_status"]["scorer_visible_output_bridge"] is True
    assert proof_skeleton["proof_status"]["scored_runtime_output_parity"] is False
    assert proof_skeleton["proof_status"]["noop_controls"] is True

    assert summary["kind"] == "lapose_foveation_byte_closed_local_candidate_build"
    assert summary["archive_sha256"] == sha256_file(archive_a)
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["tool_run_manifest"]["tool"] == (
        "tools/build_lapose_foveation_payload_archive.py"
    )
    assert "runtime_loader_parity_not_passed" in summary["readiness_blockers"]


def test_lapose_foveation_runtime_skeleton_verifies_members_then_exits_fail_closed(
    tmp_path: Path,
) -> None:
    payload = _lfv1_payload()
    result = build_lapose_foveation_payload_archive_candidate(
        out_dir=tmp_path / "candidate",
        lfv1_payload=payload,
        payload_source={
            "kind": "fixture_lfv1_payload",
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
        },
        repo_root=REPO,
    )
    extract_dir = tmp_path / "extract"
    with zipfile.ZipFile(tmp_path / "candidate" / "archive.zip") as archive:
        archive.extractall(extract_dir)

    proc = subprocess.run(
        [
            sys.executable,
            str(extract_dir / "runtime_consumer.py"),
            "--archive-root",
            str(extract_dir),
        ],
        check=False,
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    report = json.loads(proc.stdout)
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["lfv1_payload_decode"] == result["candidate_manifest"]["lfv1_payload"]["decoded"]
    assert report["lfv1_foveation_params_bridge"]["contract"] == (
        LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT
    )
    assert report["lfv1_foveation_params_bridge"]["passed"] is True
    assert report["lfv1_to_foveation_params_bridge_proven"] is True
    assert report["runtime_effect_controls"]["passed"] is True
    assert report["runtime_effect_controls"]["runtime_effect_controls_contract"] == (
        RUNTIME_EFFECT_CONTROLS_CONTRACT
    )
    assert report["scorer_visible_bridge"]["contract"] == RUNTIME_SCORER_VISIBLE_BRIDGE_CONTRACT
    assert report["scorer_visible_bridge"]["bridge_path_present"] is True
    assert report["structural_runtime_consumption_proven"] is True
    assert report["runtime_output_parity_proven"] is False
    assert report["scored_runtime_output_parity_proven"] is False
    assert report["noop_controls_proven"] is True
    assert "exact_cuda_auth_eval_missing" in report["dispatch_blockers"]
    assert "lapose_foveation_scored_runtime_output_parity_missing" in report["dispatch_blockers"]
    assert "lapose_foveation_scorer_visible_output_parity_not_proven" in report["dispatch_blockers"]


def test_lfv1_runtime_effect_controls_identity_and_structural_mutation() -> None:
    payload = _lfv1_payload()

    report = build_runtime_effect_control_report(payload)

    identity = report["lfv1_identity_decode_control"]
    mutation = report["lfv1_tuple_mutation_runtime_output_control"]
    runtime_consumes = report["runtime_consumes_foveation_tuple_control"]
    frame_warp = report["scorer_visible_frame_warp_control"]
    byte_output = report["scorer_visible_byte_output_control"]
    inflate_adapter = report["inflate_adapter_byte_output_control"]
    assert report["passed"] is True
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert identity["passed"] is True
    assert identity["byte_exact"] is True
    assert identity["source_payload_sha256"] == identity["reencoded_payload_sha256"]
    assert mutation["passed"] is True
    assert mutation["structural_output_changed"] is True
    assert (
        mutation["source_structural_output_sha256"]
        != mutation["mutated_structural_output_sha256"]
    )
    assert runtime_consumes["passed"] is True
    assert runtime_consumes["source_structural_output"]["route_count"] == 2
    assert report["structural_runtime_consumption"]["passed"] is True
    assert frame_warp["passed"] is True
    assert frame_warp["identity_control"]["passed"] is True
    assert frame_warp["nonidentity_control"]["passed"] is True
    assert frame_warp["nonidentity_control"]["output_changed"] is True
    assert byte_output["passed"] is True
    assert byte_output["identity_byte_output_control"]["byte_exact"] is True
    assert byte_output["nonidentity_byte_output_control"]["output_changed"] is True
    assert inflate_adapter["passed"] is True
    assert inflate_adapter["identity_adapter_control"]["byte_exact"] is True
    assert inflate_adapter["nonidentity_adapter_control"]["output_changed"] is True
    assert report["scored_runtime_output_parity"]["passed"] is False
    assert report["scored_runtime_output_parity"]["local_frame_warp_control_passed"] is True
    assert report["scored_runtime_output_parity"]["local_byte_output_control_passed"] is True
    assert report["scored_runtime_output_parity"]["local_inflate_adapter_control_passed"] is True


def test_lfv1_foveation_params_drive_scorer_visible_rgb_warp(tmp_path: Path) -> None:
    payload = _lfv1_payload()
    out_dir = tmp_path / "direct"
    result = build_lapose_foveation_payload_archive_candidate(
        out_dir=out_dir,
        lfv1_payload=payload,
        payload_source={
            "kind": "fixture_lfv1_payload",
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
        },
        repo_root=REPO,
    )
    foveation_record = next(
        record
        for record in result["archive_member_manifest"]["members"]
        if record["name"] == FOVEATION_PARAMS_MEMBER
    )
    assert result["summary"]["archive_sha256"] == sha256_file(out_dir / "archive.zip")
    with zipfile.ZipFile(out_dir / "archive.zip") as archive:
        foveation_params = archive.read(FOVEATION_PARAMS_MEMBER)

    assert foveation_record["sha256"] == sha256_bytes(foveation_params)
    control = build_scorer_visible_frame_warp_control_report(foveation_params)
    assert control["passed"] is True
    params = load_foveation_params(foveation_params)
    frame_index = control["probe_frame_indices"][0]
    image_size = control["hf_v1_decode"]["image_size"]
    sample = torch.linspace(
        0.0,
        1.0,
        steps=3 * image_size["height"] * image_size["width"],
        dtype=torch.float32,
    ).view(1, 3, image_size["height"], image_size["width"])
    warped = apply_lfv1_to_rgb_frames(
        sample,
        params,
        frame_indices=[frame_index],
    )
    assert tuple(warped.shape) == tuple(sample.shape)
    assert not torch.allclose(warped, sample, atol=1e-5, rtol=1e-5)


def test_lfv1_runtime_rgb24_inflate_adapter_writes_changed_bytes(tmp_path: Path) -> None:
    payload = _lfv1_payload()
    out_dir = tmp_path / "adapter"
    result = build_lapose_foveation_payload_archive_candidate(
        out_dir=out_dir,
        lfv1_payload=payload,
        payload_source={
            "kind": "fixture_lfv1_payload",
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
        },
        repo_root=REPO,
    )
    extract_dir = tmp_path / "extract_adapter"
    with zipfile.ZipFile(out_dir / "archive.zip") as archive:
        archive.extractall(extract_dir)
        foveation_params = archive.read(FOVEATION_PARAMS_MEMBER)

    byte_control = build_scorer_visible_byte_output_control_report(foveation_params)
    adapter_control = build_inflate_adapter_byte_output_control_report(foveation_params)
    assert byte_control["passed"] is True
    assert adapter_control["passed"] is True
    frame_index = adapter_control["probe_frame_indices"][0]
    image_size = adapter_control["hf_v1_decode"]["image_size"]
    sample = torch.zeros(1, 3, image_size["height"], image_size["width"], dtype=torch.float32)
    sample[:, 0] = 1.0
    sample[:, 1, :, ::2] = 1.0
    sample[:, 2, ::2, :] = 1.0
    input_raw = rgb_frames_to_rgb24_bytes(sample)
    input_path = tmp_path / "input.rgb24"
    output_path = tmp_path / "output.rgb24"
    input_path.write_bytes(input_raw)

    proc = subprocess.run(
        [
            sys.executable,
            str(extract_dir / "runtime_consumer.py"),
            "--archive-root",
            str(extract_dir),
            "--input-rgb24",
            str(input_path),
            "--output-rgb24",
            str(output_path),
            "--frame-count",
            "1",
            "--height",
            str(image_size["height"]),
            "--width",
            str(image_size["width"]),
            "--frame-indices",
            str(frame_index),
        ],
        check=False,
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert output_path.is_file()
    output_raw = output_path.read_bytes()
    assert len(output_raw) == len(input_raw)
    assert output_raw != input_raw
    report = json.loads(proc.stdout)
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["runtime_adapter_executed"] is True
    adapter_run = report["local_rgb24_inflate_adapter_run"]
    assert adapter_run["adapter_scope"] == "local_rgb24_fixture_not_full_contest_inflate"
    assert adapter_run["output_changed"] is True
    assert adapter_run["output"]["sha256"] == sha256_bytes(output_raw)
    assert report["lfv1_payload_decode"] == result["candidate_manifest"]["lfv1_payload"]["decoded"]


def test_lfv1_inflate_sh_official_signature_facade_uses_base_raw_dir(tmp_path: Path) -> None:
    payload = _small_pair0_lfv1_payload()
    out_dir = tmp_path / "official_candidate"
    build_lapose_foveation_payload_archive_candidate(
        out_dir=out_dir,
        lfv1_payload=payload,
        payload_source={
            "kind": "fixture_pair0_lfv1_payload",
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
        },
        repo_root=REPO,
    )
    extract_dir = tmp_path / "official_extract"
    with zipfile.ZipFile(out_dir / "archive.zip") as archive:
        archive.extractall(extract_dir)
        foveation_params = archive.read(FOVEATION_PARAMS_MEMBER)
    adapter_control = build_inflate_adapter_byte_output_control_report(foveation_params)
    image_size = adapter_control["hf_v1_decode"]["image_size"]
    frame_count = max(adapter_control["probe_frame_indices"]) + 1
    base_dir = tmp_path / "base_raw"
    inflated_dir = tmp_path / "inflated"
    file_list = tmp_path / "file_list.txt"
    base_dir.mkdir()
    file_list.write_text("0.mkv\n", encoding="utf-8")
    sample = torch.zeros(
        frame_count,
        3,
        image_size["height"],
        image_size["width"],
        dtype=torch.float32,
    )
    for frame_index in range(frame_count):
        sample[frame_index, 0] = 1.0 if frame_index % 2 == 0 else 0.0
        sample[frame_index, 1, :, ::2] = 1.0
        sample[frame_index, 2, ::2, :] = 1.0
    input_raw = rgb_frames_to_rgb24_bytes(sample)
    (base_dir / "0.raw").write_bytes(input_raw)
    env = os.environ.copy()
    env["PACT_PYTHON_BIN"] = sys.executable
    env["LFV1_BASE_RAW_DIR"] = str(base_dir)
    env["LFV1_CHUNK_FRAMES"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(extract_dir / "inflate.sh"),
            str(extract_dir),
            str(inflated_dir),
            str(file_list),
        ],
        check=False,
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    output_path = inflated_dir / "0.raw"
    assert output_path.is_file()
    output_raw = output_path.read_bytes()
    assert len(output_raw) == len(input_raw)
    assert output_raw != input_raw
    report = json.loads(proc.stderr)
    facade = report["official_signature_local_facade_run"]
    assert facade["adapter_scope"] == (
        "official_signature_local_base_raw_facade_not_self_sufficient_contest_decoder"
    )
    assert facade["hardware_contract"]["chunked_streaming"] is True
    assert facade["hardware_contract"]["chunk_frames"] == 1
    assert facade["research_fidelity_contract"]["external_teacher_models_at_inflate"] is False
    assert facade["research_fidelity_contract"]["posenet_segnet_sensitivity_collapsed"] is False
    assert facade["all_outputs_written"] is True
    assert facade["any_output_changed"] is True
    assert facade["records"][0]["output"]["sha256"] == sha256_bytes(output_raw)
    assert facade["records"][0]["pair_sensitivity_route_summary"][
        "posenet_segnet_sensitivity_collapsed"
    ] is False


def test_lapose_foveation_candidate_audit_fails_closed_on_payload_member_mismatch(
    tmp_path: Path,
) -> None:
    payload = _lfv1_payload()
    out = tmp_path / "candidate"
    result = build_lapose_foveation_payload_archive_candidate(
        out_dir=out,
        lfv1_payload=payload,
        payload_source={
            "kind": "fixture_lfv1_payload",
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
        },
        repo_root=REPO,
    )
    archive_path = out / "archive.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    mutated = bytearray(members[PAYLOAD_MEMBER])
    mutated[-1] ^= 1
    members[PAYLOAD_MEMBER] = bytes(mutated)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in MEMBER_ORDER:
            archive.writestr(name, members[name], compress_type=zipfile.ZIP_STORED)

    readiness = audit_lapose_foveation_payload_candidate(
        result["candidate_manifest"],
        repo_root=REPO,
        manifest_dir=out,
    )

    blockers = set(readiness["dispatch_blockers"])
    assert readiness["payload_member_proven"] is False
    assert readiness["byte_closed_local_archive"] is False
    assert "candidate_archive_sha256_mismatch" in blockers
    assert f"archive_member_manifest_member_sha256_mismatch:{PAYLOAD_MEMBER}" in blockers
    assert "lfv1_payload_member_sha256_mismatch" in blockers
    assert "lfv1_payload_decoded_preview_mismatch" in blockers
    assert "lfv1_foveation_params_bridge_report_mismatch" in blockers


def _lfv1_payload() -> bytes:
    payload, _pack = pack_lapose_foveation_tuple_payload(_manifest(), max_atoms=2)
    return payload


def _small_pair0_lfv1_payload() -> bytes:
    header = struct.Struct("<4sHHHH").pack(b"LFV1", 1, 1, 32, 24)
    row = struct.Struct("<BHHHHHH").pack(1, 0, 32768, 32768, 32768, 32768, 32768)
    return header + row


def _manifest() -> dict:
    return build_foveation_transport_atom_manifest(
        _records(),
        base_pose_dist=0.02,
        source="fixture",
        frame_width=320,
        frame_height=200,
        foveal_center=(160.0, 90.0),
        center_gain=(12.0, 8.0),
    )


def _records() -> list[dict]:
    return [
        {
            "pair_index": 10,
            "latent_action": [-0.3, 0.0, 1.0, 0.1, -0.2, 0.3, 0.02, -0.03],
            "expected_seg_dist_delta": -0.00001,
            "expected_pose_dist_delta": -0.00002,
            "confidence": 0.6,
            "class_support": [1],
            "pair_support": [10],
            "geometry_priors": ["lane_boundary"],
        },
        {
            "pair_index": 75,
            "hard_pair_rank": 0,
            "latent_action": [0.0, 1.0, 0.0, 1.2, 0.8, 1.5, 0.4, 0.2],
            "expected_seg_dist_delta": -0.0002,
            "expected_pose_dist_delta": -0.00005,
            "confidence": 0.8,
            "hard_pair_score": 4.2,
            "pair_support": [75],
            "hard_pair_support": [75],
            "class_support": [2, 3],
            "geometry_priors": ["lane_boundary"],
            "openpilot_priors": ["ego_motion"],
        },
        {
            "pair_index": 127,
            "latent_action": [0.4, 0.5, -0.5, -0.3, 0.9, 0.5, -0.2, 0.4],
            "expected_seg_dist_delta": -0.00005,
            "expected_pose_dist_delta": -0.00001,
            "confidence": 0.7,
            "class_support": [3],
            "openpilot_priors": ["yaw_rate"],
            "evidence_grade": "planning_lapose_foveation_transport",
        },
    ]
