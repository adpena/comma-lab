from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import zipfile
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "build_contest_submission_packet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_contest_submission_packet", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rpk1_payload_with_members(members: list[tuple[str, bytes]]) -> bytes:
    header = {
        "schema": "renderer_payload_v1",
        "members": [
            {
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "codec": "raw",
            }
            for name, payload in members
        ],
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return b"RPK1" + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(
        payload for _name, payload in members
    )


def _write_artifact(
    root: Path,
    *,
    device: str = "cuda",
    gpu_t4_match: bool | None = True,
    n_samples: int = 600,
    include_adjudication: bool = True,
    trace_all_match: bool = True,
    include_sjkl_payload: bool = False,
    include_cdo1_payload: bool = False,
    include_amr1_payload: bool = False,
    include_packed_cdo1_payload: bool = False,
    sjkl_auth_log: str | None = None,
    extra_auth_log: str = "",
) -> dict[str, object]:
    root.mkdir(parents=True)
    archive = root / "archive.zip"
    if include_sjkl_payload or include_cdo1_payload or include_amr1_payload or include_packed_cdo1_payload:
        with zipfile.ZipFile(archive, "w") as zf:
            members = []
            if include_packed_cdo1_payload:
                import brotli  # type: ignore

                packed = _rpk1_payload_with_members(
                    [
                        ("renderer.bin", b"renderer"),
                        ("masks.mkv", b"masks"),
                        ("masks.cdo1.xz", b"charged packed overlay"),
                        ("optimized_poses.bin", b"poses"),
                    ]
                )
                members.append(("p", brotli.compress(packed, quality=11)))
            else:
                members.append(("p", b"deterministic archive bytes"))
            if include_sjkl_payload:
                members.append(("sjkl.bin", b"charged residual"))
            if include_cdo1_payload:
                members.append(("masks.cdo1.xz", b"charged overlay"))
            if include_amr1_payload:
                members.append(("alpha4_residual_repair.amr1.xz", b"charged repair"))
            for name, payload in members:
                info = zipfile.ZipInfo(name)
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.external_attr = 0o644 << 16
                zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    else:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            info = zipfile.ZipInfo("x")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = 0o644 << 16
            zf.writestr(info, b"deterministic archive bytes")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_bytes = archive.stat().st_size
    avg_segnet = 0.00061244
    avg_posenet = 0.00049637
    score = 100 * avg_segnet + (10 * avg_posenet) ** 0.5 + 25 * archive_bytes / 37_545_489
    provenance = {
        "schema_version": 1,
        "started_at_utc": "2026-05-02T03:45:50Z",
        "tool": "experiments/contest_auth_eval.py",
        "archive_path": "/remote/pact/archive.zip",
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "device": device,
        "gpu_model": "Tesla T4" if gpu_t4_match else "L40S",
        "gpu_t4_match": gpu_t4_match,
        "cuda_available": device == "cuda",
        "cuda_device_count": 1 if device == "cuda" else 0,
        "inflate_timeout_seconds": 1800,
        "evaluate_timeout_seconds": 1800,
        "sys_argv": ["experiments/contest_auth_eval.py", "--device", device],
        "upstream_commit": "11ad728f563d8970929e8947a1cf6124ee6303e4",
    }
    contest = {
        "schema_version": 1,
        "final_score": round(score, 2),
        "avg_posenet_dist": avg_posenet,
        "avg_segnet_dist": avg_segnet,
        "score_recomputed_from_components": score,
        "score_pose_contribution": (10 * avg_posenet) ** 0.5,
        "score_seg_contribution": 100 * avg_segnet,
        "score_rate_contribution": 25 * archive_bytes / 37_545_489,
        "archive_size_bytes": archive_bytes,
        "n_samples": n_samples,
        "inflate_elapsed_seconds": 10.0,
        "evaluate_elapsed_seconds": 20.0,
        "contest_auth_eval_elapsed_seconds": 30.0,
        "provenance": provenance,
    }
    (root / "contest_auth_eval.json").write_text(json.dumps(contest, indent=2, sort_keys=True) + "\n")
    (root / "eval_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (root / "report.txt").write_text("contest report\n")
    if sjkl_auth_log is not None:
        (root / "auth_eval.log").write_text(sjkl_auth_log + extra_auth_log)
    elif extra_auth_log:
        (root / "auth_eval.log").write_text(extra_auth_log)
    trace = {
        "schema_version": 1,
        "evidence_grade": "diagnostic_component_trace",
        "score_claim": False,
        "archive_size_bytes": archive_bytes,
        "n_samples": n_samples,
        "contest_auth_eval_cross_check": {
            "all_match": trace_all_match,
            "contest_auth_eval_json_sha256": hashlib.sha256(
                (root / "contest_auth_eval.json").read_bytes()
            ).hexdigest(),
        },
    }
    (root / "component_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    if include_adjudication:
        (root / "contest_auth_eval.adjudicated.json").write_text(
            json.dumps(contest, indent=2, sort_keys=True) + "\n"
        )
        (root / "adjudication_provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "passed",
                    "result_copy": "/remote/pact/contest_auth_eval.adjudicated.json",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return {"archive_sha": archive_sha, "archive_bytes": archive_bytes, "score": score}


def _attach_runtime_manifest(artifact: Path, runtime: Path) -> dict[str, object]:
    rows = []
    for rel in ("inflate.py", "inflate.sh"):
        path = runtime / rel
        rows.append(
            {
                "relative_path": rel,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = json.loads((artifact / "contest_auth_eval.json").read_text())
    payload["provenance"]["inflate_runtime_manifest"] = {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_tree_sha256": "b" * 64,
        "files": rows,
    }
    (artifact / "contest_auth_eval.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    trace = json.loads((artifact / "component_trace.json").read_text())
    trace["contest_auth_eval_cross_check"]["contest_auth_eval_json_sha256"] = hashlib.sha256(
        (artifact / "contest_auth_eval.json").read_bytes()
    ).hexdigest()
    (artifact / "component_trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n"
    )
    return {"runtime_tree_sha256": "b" * 64, "files": rows}


def test_build_packet_writes_deterministic_manifest_and_checklist(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    expected = _write_artifact(artifact)
    output = tmp_path / "packet"

    manifest = mod.build_packet(
        artifact,
        output,
        repo_root=tmp_path,
        expected_archive_sha256=str(expected["archive_sha"]),
        expected_archive_size_bytes=int(expected["archive_bytes"]),
        expected_samples=600,
    )
    manifest_path = output / "submission_packet_manifest.json"
    checklist_path = output / "submission_packet_checklist.md"
    first_manifest = manifest_path.read_text()
    first_checklist = checklist_path.read_text()

    manifest_again = mod.build_packet(
        artifact,
        output,
        repo_root=tmp_path,
        expected_archive_sha256=str(expected["archive_sha"]),
        expected_archive_size_bytes=int(expected["archive_bytes"]),
        expected_samples=600,
    )

    assert manifest == manifest_again
    assert manifest_path.read_text() == first_manifest
    assert checklist_path.read_text() == first_checklist
    assert manifest["metadata_only"] is True
    assert manifest["raw_frames_copied"] is False
    assert manifest["archive_copied"] is False
    assert manifest["claim_policy"]["score_claim"] is False
    assert manifest["archive"]["sha256"] == expected["archive_sha"]
    assert manifest["archive"]["size_bytes"] == expected["archive_bytes"]
    assert manifest["evidence_grade"]["grade"] == "A++"
    assert manifest["frontier_summary"]["score_claim"] is False
    assert manifest["frontier_summary"]["archive_sha256"] == expected["archive_sha"]
    assert manifest["validation"]["status"] == "passed"
    assert manifest["optional_artifacts"]["component_trace.json"]["present"] is True
    assert "Field-supported grade: `A++`" in first_checklist


def test_build_packet_copies_selected_runtime_archive_report_with_release_modes(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    expected = _write_artifact(artifact)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.py").write_text("print('runtime')\n")
    inflate = runtime / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\npython inflate.py\n")
    inflate.chmod(0o755)
    (runtime / "README.md").write_text("not selected by auth runtime manifest\n")
    _attach_runtime_manifest(artifact, runtime)

    manifest = mod.build_packet(
        artifact,
        tmp_path / "packet",
        repo_root=tmp_path,
        runtime_dir=runtime,
        expected_archive_sha256=str(expected["archive_sha"]),
        expected_archive_size_bytes=int(expected["archive_bytes"]),
        expected_samples=600,
        expected_lane_id="test_lane",
        expected_job_id="test_job",
    )

    submission = tmp_path / "packet" / "submission"
    assert manifest["metadata_only"] is False
    assert manifest["archive_copied"] is True
    assert manifest["raw_frames_copied"] is False
    assert (submission / "archive.zip").read_bytes() == (artifact / "archive.zip").read_bytes()
    report_text = (submission / "report.txt").read_text()
    assert report_text.startswith("contest report\n")
    assert str(expected["archive_sha"]) in report_text
    assert str(expected["archive_bytes"]) in report_text
    assert "test_lane" in report_text
    assert "test_job" in report_text
    assert (submission / "inflate.py").read_text() == "print('runtime')\n"
    assert (submission / "archive_manifest.json").is_file()
    assert not (submission / "README.md").exists()
    assert os.access(submission / "inflate.sh", os.X_OK)
    copied = {
        row["submission_relative_path"]: row
        for row in manifest["submission"]["runtime_files"]
    }
    assert copied["inflate.sh"]["mode_octal"] == "0o755"
    assert copied["inflate.py"]["sha256"] == hashlib.sha256(
        (runtime / "inflate.py").read_bytes()
    ).hexdigest()
    assert (submission / "archive.zip").stat().st_mode & 0o777 == 0o644
    assert (submission / "report.txt").stat().st_mode & 0o777 == 0o644
    archive_manifest = json.loads((submission / "archive_manifest.json").read_text())
    assert archive_manifest["archive_sha256"] == expected["archive_sha"]
    assert archive_manifest["archive_size_bytes"] == expected["archive_bytes"]
    assert archive_manifest["members"][0]["name"] == "x"
    checklist = (tmp_path / "packet" / "submission_packet_checklist.md").read_text()
    assert "## Copied Submission" in checklist


def test_build_packet_normalizes_inflate_sh_executable_even_if_source_is_not(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    expected = _write_artifact(artifact)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.py").write_text("print('runtime')\n")
    inflate = runtime / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\npython inflate.py\n")
    inflate.chmod(0o644)
    _attach_runtime_manifest(artifact, runtime)

    manifest = mod.build_packet(
        artifact,
        tmp_path / "packet",
        repo_root=tmp_path,
        runtime_dir=runtime,
        expected_archive_sha256=str(expected["archive_sha"]),
        expected_archive_size_bytes=int(expected["archive_bytes"]),
        expected_samples=600,
    )

    copied = {
        row["submission_relative_path"]: row
        for row in manifest["submission"]["runtime_files"]
    }
    assert copied["inflate.sh"]["mode_octal"] == "0o755"
    assert os.access(tmp_path / "packet" / "submission" / "inflate.sh", os.X_OK)


def test_build_packet_rejects_runtime_file_hash_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.py").write_text("print('runtime')\n")
    inflate = runtime / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\npython inflate.py\n")
    inflate.chmod(0o755)
    _attach_runtime_manifest(artifact, runtime)
    (runtime / "inflate.py").write_text("print('changed')\n")

    with pytest.raises(mod.PacketError, match="runtime file does not match auth eval manifest"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path, runtime_dir=runtime)


def test_build_packet_rejects_archive_sha_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    payload = json.loads((artifact / "contest_auth_eval.json").read_text())
    payload["provenance"]["archive_sha256"] = "0" * 64
    (artifact / "contest_auth_eval.json").write_text(json.dumps(payload) + "\n")

    with pytest.raises(mod.PacketError, match="archive_sha256_matches_contest_auth_eval"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)


def test_build_packet_rejects_component_trace_cross_check_failure(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, trace_all_match=False)

    with pytest.raises(mod.PacketError, match="component_trace_cross_check"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)


def test_build_packet_classifies_cuda_non_t4_without_adjudication_as_a(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, gpu_t4_match=False, include_adjudication=False)

    manifest = mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)

    assert manifest["evidence_grade"]["grade"] == "A"
    assert manifest["evidence_grade"]["basis"] == "cuda_full_sample_fields"
    assert manifest["optional_artifacts"]["contest_auth_eval.adjudicated.json"]["present"] is False


def test_build_packet_records_non_score_supporting_artifacts(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    expected = _write_artifact(artifact)
    adjudicated = json.loads((artifact / "contest_auth_eval.adjudicated.json").read_text())
    adjudicated["adjudication_status"] = "passed"
    (artifact / "contest_auth_eval.adjudicated.json").write_text(
        json.dumps(adjudicated, indent=2, sort_keys=True) + "\n"
    )
    planner = tmp_path / "plans" / "planner.json"
    visualization = tmp_path / "reports" / "target_gap.svg"
    next_actions = tmp_path / "docs" / "next_actions.md"
    planner.parent.mkdir(parents=True)
    visualization.parent.mkdir(parents=True)
    next_actions.parent.mkdir(parents=True)
    planner.write_text('{"schema": "planner-ledger", "score_claim": false}\n')
    visualization.write_text("<svg><title>target gap</title></svg>\n")
    next_actions.write_text("# Next Action Tranche\n\nLocal byte win before exact eval.\n")

    manifest = mod.build_packet(
        artifact,
        tmp_path / "packet",
        repo_root=tmp_path,
        score_authority="contest_auth_eval.adjudicated.json",
        expected_archive_sha256=str(expected["archive_sha"]),
        expected_archive_size_bytes=int(expected["archive_bytes"]),
        expected_samples=600,
        planner_ledgers=[planner],
        visualizations=[visualization],
        next_action_tranches=[next_actions],
    )
    checklist = (tmp_path / "packet" / "submission_packet_checklist.md").read_text()

    assert manifest["claim_policy"]["score_authority"] == "contest_auth_eval.adjudicated.json"
    assert manifest["contest_auth_eval"]["artifact_relative_path"] == "contest_auth_eval.adjudicated.json"
    trace_cross_check = manifest["optional_artifacts"]["component_trace.json"]["validated_fields"][
        "contest_auth_eval_cross_check"
    ]
    assert manifest["contest_auth_eval"]["sha256"] != trace_cross_check["contest_auth_eval_json_sha256"]
    assert manifest["frontier_summary"]["score_recomputed_from_components"] == expected["score"]
    assert manifest["supporting_artifacts"]["claim_policy"] == {
        "score_claim": False,
        "ranking_claim": False,
        "promotion_claim": False,
        "supporting_artifacts_are_not_score_authorities": True,
    }
    assert manifest["supporting_artifacts"]["planner_ledgers"][0]["evidence_use"] == (
        "planning_or_proxy_only"
    )
    assert manifest["supporting_artifacts"]["planner_ledgers"][0]["artifact_relative_path"] is None
    assert manifest["supporting_artifacts"]["planner_ledgers"][0]["score_claim"] is False
    assert manifest["supporting_artifacts"]["visualizations"][0]["evidence_use"] == "visual_audit_only"
    assert manifest["supporting_artifacts"]["next_action_tranches"][0]["evidence_use"] == (
        "roadmap_only"
    )
    assert "## Non-Score Supporting Artifacts" in checklist
    assert "`contest_auth_eval.adjudicated.json` samples: `600`" in checklist


def test_build_packet_rejects_missing_supporting_artifact(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)

    with pytest.raises(mod.PacketError, match="missing planner ledger"):
        mod.build_packet(
            artifact,
            tmp_path / "packet",
            repo_root=tmp_path,
            planner_ledgers=[tmp_path / "missing-planner.json"],
        )


def test_build_packet_rejects_charged_sjkl_without_apply_proof(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, include_sjkl_payload=True)

    with pytest.raises(mod.PacketError, match="sjkl_auth_eval_log_present"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)


def test_build_packet_accepts_charged_sjkl_with_strict_apply_proof(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(
        artifact,
        include_sjkl_payload=True,
        sjkl_auth_log=(
            "Loaded SJ-KL residual payload: 16 pairs, k=1, target=384x512, "
            "alpha_bits=3 (250 charged bytes)\n"
            "Applying SJ-KL residuals to JointFrameGenerator fake1 "
            "(384x512, device=cuda:0)\n"
            "SJ-KL strict contract passed: applied to 16 pair(s).\n"
        ),
    )

    manifest = mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)

    sjkl = manifest["archive_payload_contracts"]["sjkl"]
    assert sjkl["present"] is True
    assert sjkl["loaded_payload_log_present"] is True
    assert sjkl["apply_log_present"] is True
    assert sjkl["strict_contract_passed_log_present"] is True


def test_build_packet_rejects_charged_cdo1_without_apply_proof(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, include_cdo1_payload=True)

    with pytest.raises(mod.PacketError, match="cdo1_auth_eval_log_present"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)


def test_build_packet_accepts_charged_cdo1_with_apply_proof(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(
        artifact,
        include_cdo1_payload=True,
        extra_auth_log="  Applied CDO1 decoded-mask overlay masks.cdo1.xz: 1,024 raw bytes\n",
    )

    manifest = mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)

    cdo1 = manifest["archive_payload_contracts"]["cdo1"]
    assert cdo1["present"] is True
    assert cdo1["present_members"] == ["masks.cdo1.xz"]
    assert cdo1["apply_log_present"] is True


def test_build_packet_detects_packed_charged_cdo1_member(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(
        artifact,
        include_packed_cdo1_payload=True,
        extra_auth_log="  Applied CDO1 decoded-mask overlay masks.cdo1.xz: 2,048 raw bytes\n",
    )

    manifest = mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)

    cdo1 = manifest["archive_payload_contracts"]["cdo1"]
    assert cdo1["present"] is True
    assert cdo1["present_members"] == ["masks.cdo1.xz"]
    assert cdo1["apply_log_present"] is True


def test_build_packet_rejects_charged_amr1_without_apply_proof(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, include_amr1_payload=True)

    with pytest.raises(mod.PacketError, match="amr1_auth_eval_log_present"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)


def test_build_packet_accepts_charged_amr1_with_apply_proof(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(
        artifact,
        include_amr1_payload=True,
        extra_auth_log=(
            "  Applied Alpha residual repair alpha4_residual_repair.amr1.xz: "
            "16,384 raw AMR1 bytes\n"
        ),
    )

    manifest = mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)

    amr1 = manifest["archive_payload_contracts"]["amr1"]
    assert amr1["present"] is True
    assert amr1["present_members"] == ["alpha4_residual_repair.amr1.xz"]
    assert amr1["apply_log_present"] is True


def test_build_packet_rejects_duplicate_zip_member(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    archive = artifact / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"a")
        zf.writestr("x", b"b")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_bytes = archive.stat().st_size
    payload = json.loads((artifact / "contest_auth_eval.json").read_text())
    payload["archive_size_bytes"] = archive_bytes
    payload["score_rate_contribution"] = 25 * archive_bytes / 37_545_489
    payload["score_recomputed_from_components"] = (
        payload["score_seg_contribution"]
        + payload["score_pose_contribution"]
        + payload["score_rate_contribution"]
    )
    payload["provenance"]["archive_sha256"] = archive_sha
    payload["provenance"]["archive_size_bytes"] = archive_bytes
    (artifact / "contest_auth_eval.json").write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(mod.PacketError, match="archive_no_duplicate_members"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)


def test_build_packet_rejects_multiple_packed_payload_containers(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    archive = artifact / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", b"a")
        zf.writestr("renderer_payload.bin.br", b"b")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_bytes = archive.stat().st_size
    payload = json.loads((artifact / "contest_auth_eval.json").read_text())
    payload["archive_size_bytes"] = archive_bytes
    payload["score_rate_contribution"] = 25 * archive_bytes / 37_545_489
    payload["score_recomputed_from_components"] = (
        payload["score_seg_contribution"]
        + payload["score_pose_contribution"]
        + payload["score_rate_contribution"]
    )
    payload["provenance"]["archive_sha256"] = archive_sha
    payload["provenance"]["archive_size_bytes"] = archive_bytes
    (artifact / "contest_auth_eval.json").write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(mod.PacketError, match="archive_packed_payload_singleton"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)
