from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments import ddm_vr3_certified_raw_reclaim as vr3


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_target_allowlist_and_live_pointer_refusal() -> None:
    admitted = vr3.AP1_ROOT / "carrier_l1/work/inflated/0.raw"
    typo = vr3.AP1_ROOT / "carrier_l1/work/inflated/1.raw"
    protected = vr3.VERTIGO_ROOT / "ddm_sj1_multipass_token_predistortion/candidate/parseback/0.raw"

    assert vr3._selected_family(admitted) == ("AP1_TERMINAL_ADVISORY", "carrier_l1")
    assert vr3._selected_family(typo) is None
    assert vr3._forbidden_target_reason(protected) == "LIVE_POINTER_TREE_PROTECTED"


def test_certificate_verifies_archive_runtime_and_raw_receipt(tmp_path: Path) -> None:
    work = tmp_path / "candidate/work"
    raw = work / "inflated/0.raw"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"decoded raw")
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime_file = runtime / "inflate.sh"
    runtime_file.write_bytes(b"#!/bin/sh\n")
    runtime_sha = hashlib.sha256(runtime_file.read_bytes()).hexdigest()
    archive = runtime / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    _write_json(
        work / "inflated_outputs_manifest.json",
        {"inflated_dir": str(raw.parent), "files": [{"relative_path": "0.raw", "sha256": raw_sha}]},
    )
    _write_json(
        work / "provenance.json",
        {
            "archive_path": str(archive),
            "archive_sha256": archive_sha,
            "effective_inflate_python": "/usr/bin/python3",
            "inflate_script": str(runtime_file),
            "inflate_script_sha256": runtime_sha,
            "upstream_snapshot_sha256": "d" * 64,
            "sys_argv": [
                "experiments/contest_auth_eval.py",
                "--archive",
                str(archive),
                "--inflate-sh",
                str(runtime_file),
            ],
            "inflate_runtime_manifest": {
                "runtime_root": str(runtime),
                "runtime_tree_sha256": "a" * 64,
                "runtime_content_tree_sha256": "b" * 64,
                "runtime_files_sha256": "c" * 64,
                "files": [
                    {
                        "relative_path": "inflate.sh",
                        "bytes": runtime_file.stat().st_size,
                        "sha256": runtime_sha,
                    }
                ],
            },
        },
    )
    _write_json(work / "contest_auth_eval.json", {"score_claim": False})

    certificate = vr3.certify_selected(raw, raw_sha)
    assert certificate["archive_sha256_verified_current"] is True
    assert certificate["runtime_file_manifest_verified_current"] is True
    assert certificate["decode_receipt"]["equals_file_sha256"] is True

    (runtime / "unrecorded.py").write_bytes(b"drift")
    with pytest.raises(vr3.CertifyError, match="runtime path-set drift"):
        vr3.certify_selected(raw, raw_sha)


def test_stat_identity_detects_content_identity_metadata_drift(tmp_path: Path) -> None:
    raw = tmp_path / "0.raw"
    raw.write_bytes(b"stable")
    stat = raw.stat()
    row = {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    assert vr3._stat_identity_blockers(raw, row) == []
    row["mtime_ns"] += 1
    assert vr3._stat_identity_blockers(raw, row) == ["RAW_MTIME_NS_DRIFT"]


def test_runtime_manifest_rejects_duplicate_relative_paths(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload = runtime / "inflate.sh"
    payload.write_bytes(b"#!/bin/sh\n")
    record = {
        "relative_path": "inflate.sh",
        "bytes": payload.stat().st_size,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    manifest = {
        "runtime_root": str(runtime),
        "runtime_tree_sha256": "a" * 64,
        "runtime_content_tree_sha256": "b" * 64,
        "runtime_files_sha256": "c" * 64,
        "files": [record, record],
    }

    with pytest.raises(vr3.CertifyError, match="duplicate runtime relative path"):
        vr3._verify_runtime_manifest(manifest)


def test_forbidden_target_rejects_resolved_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "vertigo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    raw = outside / "0.raw"
    raw.write_bytes(b"raw")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(vr3, "VERTIGO_ROOT", root)

    assert vr3._forbidden_target_reason(root / "escape/0.raw") == "RESOLVED_TARGET_OUTSIDE_VERTIGO_ROOT"


def test_reference_scan_ignores_only_this_arms_observation_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vr3, "VERTIGO_ROOT", tmp_path / "absent")
    monkeypatch.setattr(vr3, "AP_ROOT", tmp_path / "absent_ap")
    monkeypatch.setattr(vr3, "LIVE_POINTER_REFERENCE_ROOT", tmp_path / "absent_pointer")
    alias = "/Volumes/VertigoDataTier/pact/candidate/work/inflated/0.raw"
    observation = tmp_path / ".omx/tmp/codex_runs/ddm_vr3_both_ssds_full_certify_or_block_reclaim.log"
    observation.parent.mkdir(parents=True)
    observation.write_text(f"observed {alias}\n", encoding="utf-8")

    assert vr3.repo_reference_hits({alias: [alias]}, tmp_path)[alias] == []

    consumer = tmp_path / ".omx/research/consumer.md"
    consumer.parent.mkdir(parents=True)
    consumer.write_text(f"consume {alias}\n", encoding="utf-8")

    hits = vr3.repo_reference_hits({alias: [alias]}, tmp_path)[alias]
    assert len(hits) == 1
    assert "consumer.md" in hits[0]
