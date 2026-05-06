from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import stat
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(seg: float, pose: float, archive_bytes: int) -> float:
    return 100 * seg + math.sqrt(10 * pose) + 25 * archive_bytes / 37_545_489


def _stored_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in members:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload)


def _rpk1_payload_with_members(members: list[tuple[str, bytes]]) -> bytes:
    offset = 0
    rows = []
    body = bytearray()
    for name, payload in members:
        rows.append(
            {
                "name": name,
                "offset": offset,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        body.extend(payload)
        offset += len(payload)
    header = json.dumps({"members": rows}, sort_keys=True).encode("utf-8")
    return b"RPK1" + struct.pack("<I", len(header)) + header + bytes(body)


def _write_auth_eval(artifact: Path, *, device: str, gpu_t4_match: bool, n_samples: int) -> dict[str, object]:
    archive = artifact / "archive.zip"
    archive_sha = _sha(archive)
    archive_bytes = archive.stat().st_size
    avg_segnet = 0.00061244
    avg_posenet = 0.00049637
    score = _score(avg_segnet, avg_posenet, archive_bytes)
    provenance = {
        "tool": "experiments/contest_auth_eval.py",
        "archive_path": "/remote/pact/archive.zip",
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "device": device,
        "gpu_model": "Tesla T4" if gpu_t4_match else "L40S",
        "gpu_t4_match": gpu_t4_match,
        "cuda_available": device == "cuda",
        "cuda_device_count": 1 if device == "cuda" else 0,
        "upstream_commit": "11ad728f563d8970929e8947a1cf6124ee6303e4",
    }
    contest = {
        "final_score": round(score, 2),
        "avg_posenet_dist": avg_posenet,
        "avg_segnet_dist": avg_segnet,
        "score_recomputed_from_components": score,
        "score_pose_contribution": math.sqrt(10 * avg_posenet),
        "score_seg_contribution": 100 * avg_segnet,
        "score_rate_contribution": 25 * archive_bytes / 37_545_489,
        "archive_size_bytes": archive_bytes,
        "n_samples": n_samples,
        "inflate_elapsed_seconds": 10,
        "evaluate_elapsed_seconds": 20,
        "contest_auth_eval_elapsed_seconds": 30,
        "provenance": provenance,
    }
    (artifact / "contest_auth_eval.json").write_text(
        json.dumps(contest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact / "eval_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"archive_sha": archive_sha, "archive_bytes": archive_bytes, "score": score}


def _write_component_trace(artifact: Path, *, all_match: bool = True) -> None:
    contest_sha = _sha(artifact / "contest_auth_eval.json")
    contest = json.loads((artifact / "contest_auth_eval.json").read_text(encoding="utf-8"))
    trace = {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "diagnostic_component_trace",
        "n_samples": contest["n_samples"],
        "archive_size_bytes": contest["archive_size_bytes"],
        "contest_auth_eval_cross_check": {
            "all_match": all_match,
            "contest_auth_eval_json_sha256": contest_sha,
        },
    }
    (artifact / "component_trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_artifact(
    root: Path,
    *,
    members: list[tuple[str, bytes]] | None = None,
    device: str = "cuda",
    gpu_t4_match: bool = True,
    n_samples: int = 600,
    trace_all_match: bool = True,
    auth_log: str = "",
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    _stored_archive(root / "archive.zip", members or [("x", b"deterministic archive bytes")])
    result = _write_auth_eval(root, device=device, gpu_t4_match=gpu_t4_match, n_samples=n_samples)
    _write_component_trace(root, all_match=trace_all_match)
    (root / "report.txt").write_text("contest report\n", encoding="utf-8")
    if auth_log:
        (root / "auth_eval.log").write_text(auth_log, encoding="utf-8")
    return result


def _attach_runtime_manifest(artifact: Path, runtime: Path) -> None:
    rows = []
    for rel in ("inflate.py", "inflate.sh"):
        path = runtime / rel
        rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": _sha(path)})
    payload = json.loads((artifact / "contest_auth_eval.json").read_text(encoding="utf-8"))
    payload["provenance"]["inflate_runtime_manifest"] = {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_tree_sha256": "b" * 64,
        "files": rows,
    }
    (artifact / "contest_auth_eval.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_component_trace(artifact, all_match=True)


def _rewrite_auth_after_archive_change(artifact: Path) -> None:
    contest = json.loads((artifact / "contest_auth_eval.json").read_text(encoding="utf-8"))
    archive = artifact / "archive.zip"
    archive_bytes = archive.stat().st_size
    contest["archive_size_bytes"] = archive_bytes
    contest["score_rate_contribution"] = 25 * archive_bytes / 37_545_489
    contest["score_recomputed_from_components"] = (
        contest["score_seg_contribution"] + contest["score_pose_contribution"] + contest["score_rate_contribution"]
    )
    contest["provenance"]["archive_sha256"] = _sha(archive)
    contest["provenance"]["archive_size_bytes"] = archive_bytes
    (artifact / "contest_auth_eval.json").write_text(
        json.dumps(contest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_component_trace(artifact, all_match=True)


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
    first_manifest = (output / "submission_packet_manifest.json").read_text(encoding="utf-8")
    first_checklist = (output / "submission_packet_checklist.md").read_text(encoding="utf-8")
    repeat = mod.build_packet(
        artifact,
        output,
        repo_root=tmp_path,
        expected_archive_sha256=str(expected["archive_sha"]),
        expected_archive_size_bytes=int(expected["archive_bytes"]),
        expected_samples=600,
    )

    assert manifest == repeat
    assert first_manifest == (output / "submission_packet_manifest.json").read_text(encoding="utf-8")
    assert first_checklist == (output / "submission_packet_checklist.md").read_text(encoding="utf-8")
    assert manifest["metadata_only"] is True
    assert manifest["claim_policy"]["score_claim"] is False
    assert "Score claim: `false`" in first_checklist


def test_build_packet_copies_selected_runtime_archive_report_with_modes(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    expected = _write_artifact(artifact)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.py").write_text("print('runtime')\n", encoding="utf-8")
    inflate = runtime / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\npython inflate.py\n", encoding="utf-8")
    inflate.chmod(0o644)
    (runtime / "README.md").write_text("not selected\n", encoding="utf-8")
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
    assert (submission / "archive.zip").read_bytes() == (artifact / "archive.zip").read_bytes()
    assert "dispatch_lane_id: test_lane" in (submission / "report.txt").read_text(encoding="utf-8")
    assert "dispatch_job_id: test_job" in (submission / "report.txt").read_text(encoding="utf-8")
    assert stat.S_IMODE((submission / "inflate.sh").stat().st_mode) == 0o755
    assert not (submission / "README.md").exists()
    assert manifest["submission"]["raw_frames_copied"] is False


def test_build_packet_rejects_runtime_file_hash_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.py").write_text("print('runtime')\n", encoding="utf-8")
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py\n", encoding="utf-8")
    _attach_runtime_manifest(artifact, runtime)
    (runtime / "inflate.py").write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(mod.PacketError, match="runtime file does not match auth eval manifest"):
        mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path, runtime_dir=runtime)


def test_build_packet_rejects_archive_sha_and_component_trace_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    payload = json.loads((artifact / "contest_auth_eval.json").read_text(encoding="utf-8"))
    payload["provenance"]["archive_sha256"] = "0" * 64
    (artifact / "contest_auth_eval.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(mod.PacketError, match="archive_sha256_matches_contest_auth_eval"):
        mod.build_packet(artifact, tmp_path / "packet_sha", repo_root=tmp_path)

    artifact2 = tmp_path / "artifact2"
    _write_artifact(artifact2, trace_all_match=False)
    with pytest.raises(mod.PacketError, match="component_trace_cross_check"):
        mod.build_packet(artifact2, tmp_path / "packet_trace", repo_root=tmp_path)


def test_build_packet_records_non_score_supporting_artifacts(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    planner = tmp_path / "plans" / "planner.json"
    visualization = tmp_path / "reports" / "target_gap.svg"
    next_actions = tmp_path / "docs" / "next_actions.md"
    for path, text in (
        (planner, '{"schema": "planner-ledger", "score_claim": false}\n'),
        (visualization, "<svg><title>target gap</title></svg>\n"),
        (next_actions, "# Next Action Tranche\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    manifest = mod.build_packet(
        artifact,
        tmp_path / "packet",
        repo_root=tmp_path,
        planner_ledgers=[planner],
        visualizations=[visualization],
        next_action_tranches=[next_actions],
    )

    assert manifest["supporting_artifacts"]["claim_policy"]["supporting_artifacts_are_not_score_authorities"] is True
    for section in ("planner_ledgers", "visualizations", "next_action_tranches"):
        record = manifest["supporting_artifacts"][section][0]
        assert record["score_claim"] is False
        assert record["ranking_claim"] is False


def test_build_packet_rejects_charged_payloads_without_apply_proof(tmp_path: Path) -> None:
    mod = _load_module()
    cases = [
        ("sjkl", [("p", b"payload"), ("sjkl.bin", b"charged residual")], "sjkl_auth_eval_log_present"),
        ("cdo1", [("p", b"payload"), ("masks.cdo1.xz", b"charged overlay")], "cdo1_auth_eval_log_present"),
        ("amr1", [("p", b"payload"), ("alpha4_residual_repair.amr1.xz", b"charged repair")], "amr1_auth_eval_log_present"),
    ]
    for label, members, pattern in cases:
        artifact = tmp_path / label
        _write_artifact(artifact, members=members)
        with pytest.raises(mod.PacketError, match=pattern):
            mod.build_packet(artifact, tmp_path / f"packet_{label}", repo_root=tmp_path)


def test_build_packet_accepts_charged_payloads_with_apply_proof(tmp_path: Path) -> None:
    mod = _load_module()
    cases = [
        (
            "sjkl",
            [("p", b"payload"), ("sjkl.bin", b"charged residual")],
            "Loaded SJ-KL residual payload: 16 pairs\n"
            "Applying SJ-KL residuals to JointFrameGenerator fake1\n"
            "SJ-KL strict contract passed: applied\n",
        ),
        (
            "cdo1",
            [("p", b"payload"), ("masks.cdo1.xz", b"charged overlay")],
            "Applied CDO1 decoded-mask overlay masks.cdo1.xz: 1024 raw bytes\n",
        ),
        (
            "amr1",
            [("p", b"payload"), ("alpha4_residual_repair.amr1.xz", b"charged repair")],
            "Applied Alpha residual repair alpha4_residual_repair.amr1.xz: 16384 raw bytes\n",
        ),
    ]
    for label, members, auth_log in cases:
        artifact = tmp_path / label
        _write_artifact(artifact, members=members, auth_log=auth_log)
        manifest = mod.build_packet(artifact, tmp_path / f"packet_{label}", repo_root=tmp_path)
        assert manifest["archive_payload_contracts"][label]["present"] is True


def test_build_packet_detects_packed_charged_cdo1_member(tmp_path: Path) -> None:
    brotli = pytest.importorskip("brotli")
    mod = _load_module()
    packed = _rpk1_payload_with_members(
        [
            ("renderer.bin", b"renderer"),
            ("masks.cdo1.xz", b"charged packed overlay"),
            ("optimized_poses.bin", b"poses"),
        ]
    )
    artifact = tmp_path / "packed"
    _write_artifact(
        artifact,
        members=[("p", brotli.compress(packed, quality=11))],
        auth_log="Applied CDO1 decoded-mask overlay masks.cdo1.xz: 2048 raw bytes\n",
    )

    manifest = mod.build_packet(artifact, tmp_path / "packet", repo_root=tmp_path)

    assert manifest["archive_payload_contracts"]["cdo1"]["present_members"] == ["masks.cdo1.xz"]


def test_build_packet_rejects_duplicate_members_and_multiple_packed_payloads(tmp_path: Path) -> None:
    mod = _load_module()
    duplicate = tmp_path / "duplicate"
    _write_artifact(duplicate)
    with zipfile.ZipFile(duplicate / "archive.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"a")
        with pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr("x", b"b")
    _rewrite_auth_after_archive_change(duplicate)
    with pytest.raises(mod.PacketError, match="archive_no_duplicate_members"):
        mod.build_packet(duplicate, tmp_path / "packet_duplicate", repo_root=tmp_path)

    multi = tmp_path / "multi"
    _write_artifact(multi)
    _stored_archive(multi / "archive.zip", [("p", b"a"), ("renderer_payload.bin.br", b"b")])
    _rewrite_auth_after_archive_change(multi)
    with pytest.raises(mod.PacketError, match="archive_packed_payload_singleton"):
        mod.build_packet(multi, tmp_path / "packet_multi", repo_root=tmp_path)
