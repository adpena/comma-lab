# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import tac.optimization.einstein_kolmogorov_frontier as frontier
from tac.optimization.einstein_kolmogorov_frontier import (
    CANONICAL_POINTER_SOURCE,
    CURRENT_SCOPE,
    R3_DESCRIPTION_BYTES,
    TARGET_TOLERANCES,
    FrontierRefusal,
    compile_frontier,
    preflight_u3,
    validate_receipt,
    write_checkpoint,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _archive(path: Path, data_bytes: int) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr(info, b"x" * data_bytes)


def _candidate(
    path: Path,
    *,
    candidate_id: str = "candidate",
    data_bytes: int = 64,
    container_overhead_bytes: int = 0,
    runtime_overhead_bytes: int = 0,
    d_seg: float = 0.0001,
    **changes: object,
) -> dict[str, object]:
    _archive(path, data_bytes)
    artifact = _ref(path)
    payload_bytes = int(artifact["bytes"]) - container_overhead_bytes
    evidence = path.parent / f"{path.stem}_evidence"
    evidence.mkdir()

    stream_a = evidence / "stream_a.raw"
    stream_b = evidence / "stream_b.raw"
    stream_a.write_bytes(b"receiver-stream")
    stream_b.write_bytes(b"receiver-stream")
    runtime = evidence / "runtime_manifest.json"
    runtime_tree_sha = hashlib.sha256(b"runtime-tree").hexdigest()
    _json(runtime, {"runtime_tree_sha256": runtime_tree_sha})
    evaluator = evidence / "evaluate.py"
    evaluator.write_text("# frozen evaluator fixture\n", encoding="utf-8")
    source = evidence / "source.json"
    config = evidence / "config.json"
    gt = evidence / "gt.json"
    _json(source, {"source": "fixture"})
    _json(config, {"seed": 7})
    _json(gt, {"sample_count": 600})
    report = evidence / "official_report.txt"
    report.write_text(
        "=== Evaluation results over 600 samples ===\n"
        "  Average PoseNet Distortion: 0.0002\n"
        f"  Average SegNet Distortion: {d_seg}\n"
        f"  Submission file size: {artifact['bytes']:,} bytes\n",
        encoding="utf-8",
    )
    parseback = evidence / "parseback.json"
    parseback_payload = {
        "schema": "einstein_kolmogorov_parseback.v1",
        "archive": artifact,
        "first_stream": _ref(stream_a),
        "second_stream": _ref(stream_b),
        "runtime": _ref(runtime),
        "receiver_closed": True,
        "double_decode_identical": True,
        "argv": ["inflate.sh"],
    }
    _json(parseback, parseback_payload)
    evaluation = evidence / "evaluation.json"
    eval_argv = ["python3", "upstream/evaluate.py", "--device", "cpu"]
    evaluation_payload = {
        "schema": "einstein_kolmogorov_official_eval.v1",
        "archive": artifact,
        "realized_stream": _ref(stream_a),
        "official_report": _ref(report),
        "evaluator": _ref(evaluator),
        "source_manifest": _ref(source),
        "config_manifest": _ref(config),
        "gt_manifest": _ref(gt),
        "sample_count": 600,
        "d_seg": d_seg,
        "d_pose": 0.0002,
        "hard_score": True,
        "evidence_axis": "[contest-CPU]",
        "argv": eval_argv,
        "seed": 7,
    }
    _json(evaluation, evaluation_payload)
    candidate: dict[str, object] = {
        "candidate_id": candidate_id,
        "artifact": artifact,
        "counted_payload_bytes": payload_bytes,
        "target_tolerance": 0.000152,
        "container_overhead_bytes": container_overhead_bytes,
        "runtime_overhead_bytes": runtime_overhead_bytes,
        "levels": [
            {"level": "chart", "bytes": payload_bytes - 1, "miss_mass": 1.0},
            {"level": "pixel", "bytes": 1, "miss_mass": 0.0},
        ],
        "receiver_closed": True,
        "parseback_double_decode_identical": True,
        "score": {
            "d_seg": d_seg,
            "d_pose": 0.0002,
            "artifact_sha256": artifact["sha256"],
            "sample_count": 600,
            "hard_score": True,
            "realized_stream_hash": _sha(stream_a),
        },
        "evidence_axis": "[contest-CPU]",
        "source_evidence_axis": "[contest-CPU]",
        "quarantined_identifiers": [],
        "provenance": {
            "source_hash": _sha(source),
            "runtime_hash": _sha(runtime),
            "evaluator_hash": _sha(evaluator),
            "config_hash": _sha(config),
            "seed": 7,
            "gt_hash": _sha(gt),
            "argv": eval_argv,
        },
        "authority_bundle": {
            "parseback_receipt": _ref(parseback),
            "evaluation_receipt": _ref(evaluation),
        },
    }
    candidate.update(changes)
    return candidate


def _compile(rows: list[dict[str, object]], **kwargs: object):
    pointer_path = frontier.REPO_ROOT / CANONICAL_POINTER_SOURCE
    pointer_ref = _ref(pointer_path)
    pointer_ref["path"] = CANONICAL_POINTER_SOURCE
    return compile_frontier(rows, pointer_source=pointer_ref, **kwargs)


def _trust_canonical_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the selector after a separately validated canonical adapter.

    This is an explicit unit-test double, not evaluator evidence.  Separate
    tests below prove that the self-authored on-disk fixture is rejected by the
    real authority gate.
    """

    def trusted(raw: dict[str, object], artifact: dict[str, object]) -> frontier.AuthorityEvidence:
        score = raw["score"]
        provenance = raw["provenance"]
        assert isinstance(score, dict) and isinstance(provenance, dict)
        return frontier.AuthorityEvidence(
            d_seg=float(score["d_seg"]),
            d_pose=float(score["d_pose"]),
            sample_count=int(score["sample_count"]),
            archive_bytes=int(artifact["bytes"]),
            archive_sha256=str(artifact["sha256"]),
            stream_sha256=str(score["realized_stream_hash"]),
            evidence_axis=str(raw["evidence_axis"]),
            runtime_sha256=str(provenance["runtime_hash"]),
            runtime_tree_sha256="1" * 64,
            interpreter_path="/usr/bin/python3",
            interpreter_sha256="2" * 64,
            interpreter_version="3.test",
            evaluator_sha256=str(provenance["evaluator_hash"]),
            source_sha256=str(provenance["source_hash"]),
            config_sha256=str(provenance["config_hash"]),
            gt_sha256=str(provenance["gt_hash"]),
            gt_source_path="/fixture/gt",
            gt_source_sha256="3" * 64,
            seed=int(provenance["seed"]),
            argv=tuple(str(item) for item in provenance["argv"]),
            receipt_paths=("unit-test-double",),
        )

    monkeypatch.setattr(frontier, "_validate_authority_bundle", trusted)


def test_selector_after_canonical_adapter_chooses_least_archive_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _trust_canonical_adapter(monkeypatch)
    expensive = _candidate(tmp_path / "expensive.zip", candidate_id="expensive", data_bytes=80)
    cheap = _candidate(tmp_path / "cheap.zip", candidate_id="cheap", data_bytes=40)

    receipt = _compile([expensive, cheap])

    assert tuple(row.tolerance for row in receipt.tolerance_rows) == TARGET_TOLERANCES
    assert all(row.status == "MEASURED_FRONTIER_POINT" for row in receipt.tolerance_rows)
    assert {row.candidate_id for row in receipt.tolerance_rows} == {"cheap"}
    assert {row.evidence_axis for row in receipt.tolerance_rows} == {"[contest-CPU]"}
    assert receipt.dominated_measured_candidates


def test_frontier_compares_full_archive_not_payload_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _trust_canonical_adapter(monkeypatch)
    larger = _candidate(tmp_path / "larger.zip", candidate_id="larger", data_bytes=100)
    larger_bytes = int(larger["artifact"]["bytes"])  # type: ignore[index]
    smaller = _candidate(tmp_path / "smaller.zip", candidate_id="smaller", data_bytes=20)
    smaller_bytes = int(smaller["artifact"]["bytes"])  # type: ignore[index]
    larger["container_overhead_bytes"] = larger_bytes - 1
    larger["counted_payload_bytes"] = 1
    larger["levels"] = [{"level": "chart", "bytes": 1, "miss_mass": 1.0}]

    receipt = _compile([larger, smaller])

    assert larger_bytes > smaller_bytes
    assert {row.candidate_id for row in receipt.tolerance_rows} == {"smaller"}
    assert {row.counted_archive_bytes for row in receipt.tolerance_rows} == {smaller_bytes}


def test_full_counted_archive_accounting_must_close(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "under-counted.zip")
    candidate["container_overhead_bytes"] = 1
    receipt = _compile([candidate])
    assert "BLOCKED_FULL_COUNTED_ARCHIVE_BYTE_MISMATCH" in receipt.candidate_rejections["candidate"]


def test_generic_runtime_is_hash_bound_but_never_rate_priced(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "runtime-priced.zip", runtime_overhead_bytes=17)
    receipt = _compile([candidate])
    assert "BLOCKED_VIDEO_AGNOSTIC_RUNTIME_MUST_BE_FREE" in receipt.candidate_rejections["candidate"]
    assert "BLOCKED_FULL_COUNTED_ARCHIVE_BYTE_MISMATCH" not in receipt.candidate_rejections["candidate"]


def test_description_only_r3_refuses_without_execution_plan() -> None:
    report = preflight_u3({"description_only": True, "counted_bytes": R3_DESCRIPTION_BYTES})
    assert report.status == "BLOCKED_DESCRIPTION_ROW_NOT_RECEIVER_TUPLE"
    assert report.execution_plan is None
    assert any("451191" in blocker for blocker in report.blockers)


def test_missing_and_explicit_false_custody_are_distinct(tmp_path: Path) -> None:
    missing = _candidate(tmp_path / "missing.zip", candidate_id="missing")
    missing.pop("receiver_closed")
    false = _candidate(tmp_path / "false.zip", candidate_id="false", receiver_closed=False)
    receipt = _compile([missing, false])
    assert "BLOCKED_MISSING_RECEIVER_CLOSED" in receipt.candidate_rejections["missing"]
    assert "BLOCKED_RECEIVER_NOT_CLOSED" in receipt.candidate_rejections["false"]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"quarantined_identifiers": ["retired-artifact"]}, "BLOCKED_QUARANTINED_IDENTIFIER"),
        ({"authority_bundle": {}}, "BLOCKED_MISSING_AUTHORITY_BUNDLE"),
        ({"evidence_axis": "invented-axis"}, "BLOCKED_INVALID_EVIDENCE_AXIS"),
    ],
)
def test_inadmissible_authority_classes_refuse(tmp_path: Path, change: dict[str, object], expected: str) -> None:
    candidate = _candidate(tmp_path / f"{expected}.zip", candidate_id=expected)
    candidate.update(change)
    receipt = _compile([candidate])
    assert expected in receipt.candidate_rejections[expected]
    assert all(row.status == "NO_FEASIBLE_CANDIDATE" for row in receipt.tolerance_rows)
    assert all(row.verdict_scope == CURRENT_SCOPE for row in receipt.tolerance_rows)


def test_marker_only_fake_measurement_is_never_admitted(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "marker-only.zip")
    candidate.pop("authority_bundle")
    receipt = _compile([candidate])
    assert "BLOCKED_MISSING_AUTHORITY_BUNDLE" in receipt.candidate_rejections["candidate"]
    assert not receipt.measured_frontier


def test_self_authored_advisory_evidence_is_never_admitted(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "self-authored-advisory.zip")
    candidate["evidence_axis"] = "[macOS-CPU advisory]"
    candidate["source_evidence_axis"] = "[macOS-CPU advisory]"
    receipt = _compile([candidate])
    assert "BLOCKED_INVALID_EVIDENCE_AXIS" in receipt.candidate_rejections["candidate"]
    assert not receipt.measured_frontier


def test_canonical_evaluator_sidecar_cannot_mask_executed_script(tmp_path: Path) -> None:
    canonical_sidecar = tmp_path / "evaluate.py"
    canonical_sidecar.write_text("canonical fixture", encoding="utf-8")
    malicious = tmp_path / "self_authored_evaluator.py"
    malicious.write_text("self-authored fixture", encoding="utf-8")
    argv = ("/usr/bin/python3", str(malicious), "--unused", str(canonical_sidecar))
    with pytest.raises(FrontierRefusal, match="BLOCKED_EXECUTED_EVALUATOR_PATH_MISMATCH"):
        frontier._require_executed_evaluator(argv, canonical_sidecar)

    claimed_argv = ("/usr/bin/python3", str(canonical_sidecar), "--device", "cpu")
    malicious_command = f"/usr/bin/python3 {malicious} --device cpu --unused {canonical_sidecar}"
    with pytest.raises(FrontierRefusal, match="BLOCKED_EXACT_EVAL_COMMAND_ARGV_MISMATCH"):
        frontier._require_exact_eval_command_binding(
            malicious_command,
            claimed_argv,
            canonical_sidecar,
        )


def test_artifact_hash_mismatch_refuses(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "mutated.zip")
    Path(candidate["artifact"]["path"]).write_bytes(b"mutated")  # type: ignore[index]
    receipt = _compile([candidate])
    assert "BLOCKED_ARTIFACT_HASH_MISMATCH" in receipt.candidate_rejections["candidate"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda row: row["artifact"].__setitem__("bytes", True), "INVALID_artifact_bytes"),  # type: ignore[union-attr]
        (lambda row: row.__setitem__("counted_payload_bytes", "100"), "INVALID_candidate_counted_payload_bytes"),
        (lambda row: row["levels"][0].__setitem__("bytes", 1.5), "INVALID_level_bytes"),  # type: ignore[union-attr,index]
        (lambda row: row["score"].__setitem__("sample_count", "600"), "INVALID_score_sample_count"),  # type: ignore[union-attr]
    ],
)
def test_numeric_custody_rejects_silent_coercion(tmp_path: Path, mutate: object, expected: str) -> None:
    candidate = _candidate(tmp_path / hashlib.sha256(expected.encode()).hexdigest()[:8])
    mutate(candidate)  # type: ignore[operator]
    receipt = _compile([candidate])
    assert receipt.candidate_rejections["candidate"] == (expected,)


def test_fake_u3_markers_cannot_create_ready_tuple(tmp_path: Path) -> None:
    archive = tmp_path / "fake.zip"
    _archive(archive, 8)
    row = {
        "archive": _ref(archive),
        "counted_bytes": archive.stat().st_size,
        "counted_payload_bytes": archive.stat().st_size,
        "container_overhead_bytes": 0,
        "runtime_overhead_bytes": 0,
        "receiver_closed": True,
        "parseback_double_decode_identical": True,
    }
    report = preflight_u3(row)
    assert report.status == "BLOCKED_U3_RECEIVER_TUPLE"
    assert report.execution_plan is None
    assert not report.predicate_table["authority_receipts_rederived"]


def test_settled_s4_guard_uses_full_identity_not_byte_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "same-size-only.zip"
    _archive(archive, 8)
    with zipfile.ZipFile(archive) as handle:
        member_sha = hashlib.sha256(handle.read("0.bin")).hexdigest()
    monkeypatch.setattr(frontier, "S4_SETTLED_BYTES", archive.stat().st_size)
    unrelated = preflight_u3({"archive": _ref(archive), "counted_bytes": archive.stat().st_size})
    assert unrelated.status != "SETTLED_S4_REUSE_RECEIPT_ONLY_NO_RERUN"

    monkeypatch.setattr(frontier, "S4_SETTLED_SHA256", _sha(archive))
    monkeypatch.setattr(frontier, "S4_SETTLED_MEMBER_SHA256", member_sha)
    monkeypatch.setattr(frontier, "S4_SETTLED_RUNTIME_SHA256", "r" * 64)
    monkeypatch.setattr(frontier, "S4_SETTLED_STREAM_SHA256", "s" * 64)
    settled = preflight_u3(
        {
            "archive": _ref(archive),
            "counted_bytes": archive.stat().st_size,
            "runtime_hash": "r" * 64,
            "realized_stream_hash": "s" * 64,
        }
    )
    assert settled.status == "SETTLED_S4_REUSE_RECEIPT_ONLY_NO_RERUN"
    assert settled.execution_plan is None


def test_receipt_validation_rederives_sources_and_resume_is_immutable(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "candidate.zip")
    first = _compile([candidate])
    second = _compile([candidate])
    assert first.receipt_sha256 == second.receipt_sha256
    payload = json.loads(json.dumps(first.as_dict()))
    validate_receipt(payload)
    output = tmp_path / "receipt.json"
    write_checkpoint(output, first)
    write_checkpoint(output, first)
    with pytest.raises(FrontierRefusal, match="INCOMPATIBLE_RESUME_OUTPUT"):
        write_checkpoint(output, replace(second, authority_labels={"changed": True}).with_hash())

    payload["tolerance_rows"][0]["status"] = "fabricated"
    payload["receipt_sha256"] = hashlib.sha256(frontier.canonical_json({**payload, "receipt_sha256": ""})).hexdigest()
    with pytest.raises(FrontierRefusal, match="RECEIPT_REDERIVATION_MISMATCH"):
        validate_receipt(payload)


def test_receipt_hash_binds_exact_input_manifest_bytes(tmp_path: Path) -> None:
    manifest = tmp_path / "candidates.json"
    _json(manifest, [])
    receipt = _compile([], input_manifests=[_ref(manifest)])
    assert receipt.source_input_manifests == ({**_ref(manifest), "custody_rederived": True},)

    manifest.write_text("[]\n", encoding="utf-8")
    with pytest.raises(FrontierRefusal, match="MISMATCH_input_manifest_0_FILE_CUSTODY"):
        validate_receipt(json.loads(json.dumps(receipt.as_dict())))


def test_checkpoint_can_never_exceed_its_validator_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _compile([])
    monkeypatch.setattr(frontier, "MAX_RECEIPT_BYTES", 8)
    with pytest.raises(FrontierRefusal, match="FRONTIER_RECEIPT_TOO_LARGE"):
        write_checkpoint(tmp_path / "oversized.json", receipt)


def test_runtime_tree_hash_is_rederived_from_actual_files(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    entrypoint = runtime_root / "inflate.py"
    entrypoint.write_text("print('receiver')\n", encoding="utf-8")
    declared = frontier.tree_sha256(runtime_root)

    root, actual = frontier._runtime_tree_custody({"runtime_root": str(runtime_root), "runtime_tree_sha256": declared})
    assert root == runtime_root.resolve()
    assert actual == declared

    entrypoint.write_text("print('mutated')\n", encoding="utf-8")
    with pytest.raises(FrontierRefusal, match="BLOCKED_RUNTIME_TREE_SHA_NOT_REDERIVED"):
        frontier._runtime_tree_custody({"runtime_root": str(runtime_root), "runtime_tree_sha256": declared})


def test_realized_output_manifest_hashes_actual_scored_stream(tmp_path: Path) -> None:
    output_root = tmp_path / "inflated"
    output_root.mkdir()
    realized = output_root / "0.raw"
    realized.write_bytes(b"scored receiver bytes")
    rows = [
        {
            "relative_path": "0.raw",
            "bytes": realized.stat().st_size,
            "sha256": _sha(realized),
        }
    ]
    aggregate = hashlib.sha256(frontier.canonical_json({"files": rows})).hexdigest()
    manifest = tmp_path / "inflated_outputs_manifest.json"
    _json(
        manifest,
        {
            "schema": "contest_auth_eval_inflated_output_manifest_v1",
            "inflated_dir": str(output_root),
            "raw_file_count": 1,
            "total_bytes": realized.stat().st_size,
            "files": [{**rows[0], "exists": True, "video_name": "0.mkv"}],
            "aggregate_sha256": aggregate,
        },
    )

    assert (
        frontier._validate_realized_output_manifest(
            manifest,
            realized_path=realized,
            realized_bytes=realized.stat().st_size,
            realized_sha256=_sha(realized),
            exact_raw_aggregate_sha256=aggregate,
        )
        == aggregate
    )
    realized.write_bytes(b"different")
    with pytest.raises(FrontierRefusal, match="BLOCKED_INFLATED_OUTPUT_FILE_CUSTODY"):
        frontier._validate_realized_output_manifest(
            manifest,
            realized_path=realized,
            realized_bytes=len(b"scored receiver bytes"),
            realized_sha256=rows[0]["sha256"],
            exact_raw_aggregate_sha256=aggregate,
        )


def test_local_external_attestation_is_never_execution_authority(tmp_path: Path) -> None:
    archive_sha = "a" * 64
    runtime_sha = "b" * 64
    manifest_sha = "c" * 64
    aggregate_sha = "d" * 64
    evaluator_sha = "e" * 64
    argv = ("/usr/bin/python3", "/repo/upstream/evaluate.py", "--device", "cpu")
    attestation = tmp_path / "external.json"
    _json(
        attestation,
        {
            "schema": frontier.EXTERNAL_EXECUTION_SCHEMA,
            "archive_sha256": archive_sha,
            "archive_bytes": 123,
            "runtime_tree_sha256": runtime_sha,
            "output_manifest_sha256": manifest_sha,
            "output_aggregate_sha256": aggregate_sha,
            "evaluator_sha256": evaluator_sha,
            "argv": list(argv),
            "score_claim_valid": True,
            "terminal": True,
            "provider": "modal",
            "provider_job_id": "fake-local-job",
            "hardware": "Linux x86_64 CPU",
        },
    )
    with pytest.raises(FrontierRefusal, match="BLOCKED_EXTERNAL_EXECUTION_ATTESTATION_UNVERIFIED"):
        frontier._require_external_execution_attestation(
            _ref(attestation),
            artifact={"sha256": archive_sha, "bytes": 123},
            runtime_tree_sha256=runtime_sha,
            output_manifest_sha256=manifest_sha,
            output_aggregate_sha256=aggregate_sha,
            evaluator_sha256=evaluator_sha,
            argv=argv,
        )


def test_checkpoint_race_never_overwrites_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "receipt.json"
    receipt = _compile([])

    def racing_writer(path: Path, payload: bytes, **_: object) -> None:
        path.write_bytes(b"racing-writer-won")
        raise frontier.ArtifactWriteError("simulated final-name race")

    monkeypatch.setattr(frontier, "write_bytes_artifact", racing_writer)
    with pytest.raises(FrontierRefusal, match="INCOMPATIBLE_RESUME_OUTPUT"):
        write_checkpoint(output, receipt)
    assert output.read_bytes() == b"racing-writer-won"
