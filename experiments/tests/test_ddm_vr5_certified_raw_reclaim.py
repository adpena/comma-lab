from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from experiments import ddm_vr3_certified_raw_reclaim as vr3
from experiments.tests.test_ddm_vr3_certified_raw_reclaim import _write_json


def fixture_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vr3, "AP_ROOT", tmp_path / "absent_ap")
    monkeypatch.setattr(vr3, "LIVE_POINTER_REFERENCE_ROOT", tmp_path / "absent_pointer")
    work = tmp_path / "ddm_test/work"
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

    monkeypatch.setattr(vr3, "VERTIGO_ROOT", tmp_path)
    stat = raw.stat()
    source = {
        "path": str(raw),
        "owner": "MAIN / ddm_test",
        "certificate_status": vr3.RETAINED_STATUS,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "historical_sha256": raw_sha,
        "reproducer": vr3.certify_selected(raw, raw_sha),
    }
    rehash = {"path": str(raw), "sha256": raw_sha, "bytes": stat.st_size, "present": True}
    memo = tmp_path / "closure.md"
    memo.write_text("Terminal advisory instance refused after completed measurement.")
    closure = {
        "family": "ddm_test",
        "disposition": "CLOSED_ADVISORY_INSTANCE",
        "memo": vr3.pinned_file(memo),
        "verdict_quote": memo.read_text(),
    }
    return raw, source, rehash, closure


def test_retained_admission_and_original_two_families(tmp_path, monkeypatch):
    raw, source, rehash, closure = fixture_chain(tmp_path, monkeypatch)
    certificate = vr3.certify_retained(source, rehash, closure)
    assert certificate["archive_sha256"] == source["reproducer"]["archive_sha256"]
    assert vr3._selected_family(vr3.AP1_ROOT / "carrier_l1/work/inflated/0.raw")
    assert vr3._selected_family(vr3.JF2_ROOT / "null_r2/work/inflated/0.raw")
    assert raw.read_bytes() == b"decoded raw"


@pytest.mark.parametrize("drift", ["historical", "archive", "runtime", "reproducer", "closure", "rehash"])
def test_admission_rejects_broken_chain(tmp_path, monkeypatch, drift):
    raw, source, rehash, closure = fixture_chain(tmp_path, monkeypatch)
    if drift == "historical":
        source["historical_sha256"] = "f" * 64
    elif drift == "archive":
        Path(source["reproducer"]["archive_path"]).write_bytes(b"drift")
    elif drift == "runtime":
        Path(source["reproducer"]["inflate_script"]).write_bytes(b"drift")
    elif drift == "reproducer":
        source["reproducer"] = None
    elif drift == "closure":
        closure["disposition"] = "QUEUED"
    else:
        rehash["present"] = False
    with pytest.raises(vr3.CertifyError):
        vr3.certify_retained(source, rehash, closure)
    assert raw.exists()


@pytest.mark.parametrize(
    "name", ["ddm_sj1_x", "ddm_rp1_x", "ddm_bnd1_x", "ddm_gb2_x", "ddm_cmp2_compose", "ddm_rp1", "ddm_gb2"]
)
def test_live_tree_refused_before_certificate(tmp_path, monkeypatch, name):
    monkeypatch.setattr(vr3, "VERTIGO_ROOT", tmp_path)
    assert vr3._forbidden_target_reason(tmp_path / name / "work/inflated/0.raw") == "LIVE_POINTER_TREE_PROTECTED"


@pytest.mark.parametrize("mode", ["ps_denied", "pgrep_denied", "blind", "live", "lsof_warning"])
def test_process_visibility_fail_closed(monkeypatch, mode):
    def run(command, **kwargs):
        if command[0] == "lsof":
            return subprocess.CompletedProcess(command, 1, "", "cannot stat mount")
        if command[0] == "ps":
            if mode == "ps_denied":
                raise PermissionError("sandbox")
            text = f"1 0 launchd\n{os.getpid()} 1 pytest\n"
            return subprocess.CompletedProcess(command, 0, "" if mode == "blind" else text, "")
        return subprocess.CompletedProcess(
            command,
            3 if mode == "pgrep_denied" else 0 if mode == "live" else 1,
            "123 ddm_test worker" if mode == "live" else "",
            "Cannot get process list" if mode == "pgrep_denied" else "",
        )

    monkeypatch.setattr(vr3.subprocess, "run", run)
    with pytest.raises(vr3.CertifyError):
        if mode == "lsof_warning":
            vr3.lsof_plus_d(Path("."))
        else:
            vr3.process_gate("ddm_test")


def test_visible_idle_host_passes(monkeypatch):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0 if command[0] == "ps" else 1,
            f"1 0 launchd\n{os.getpid()} 1 pytest\n" if command[0] == "ps" else "",
            "",
        )

    monkeypatch.setattr(vr3.subprocess, "run", run)
    assert vr3.process_gate("ddm_test")["visible"]


def make_plan(tmp_path, monkeypatch):
    raw, source, rehash, closure = fixture_chain(tmp_path, monkeypatch)
    (tmp_path / "experiments").mkdir()
    source_file = tmp_path / "source.jsonl"
    hashes_file = tmp_path / "hashes.jsonl"
    vr3.atomic_jsonl(source_file, [source])
    vr3.atomic_jsonl(hashes_file, [rehash])
    policy = tmp_path / "policy.json"
    _write_json(policy, {"closures": {"ddm_test": closure}, "observation_files": []})
    ledger = tmp_path / "plan.jsonl"
    args = argparse.Namespace(
        source_ledger=source_file, rehash_ledger=hashes_file, closures=policy, output_ledger=ledger, repo_root=tmp_path
    )
    monkeypatch.setattr(
        vr3, "combined_df", lambda roots: {"available_bytes": 1000 + (0 if raw.exists() else len(b"decoded raw"))}
    )
    vr3.plan_retained(args)
    apply_args = argparse.Namespace(
        ledger=ledger,
        expected_ledger_sha256=vr3.sha256_file(ledger),
        journal=tmp_path / "journal.jsonl",
        target_bytes=source["bytes"],
        repo_root=tmp_path,
    )
    return raw, args, apply_args


@pytest.mark.parametrize("fault", ["none", "blind", "raw_drift", "consumer", "archive_drift"])
def test_plan_apply_journals_or_preserves_fixture(tmp_path, monkeypatch, fault):
    raw, plan_args, args = make_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(vr3, "process_gate", lambda owner: {"visible": True})
    monkeypatch.setattr(vr3, "lsof_plus_d", lambda path: {"open_descriptor_rows": []})
    if fault == "blind":

        def blind(owner):
            raise vr3.CertifyError("PROCESS_VISIBILITY_UNAVAILABLE")

        monkeypatch.setattr(vr3, "process_gate", blind)
        with pytest.raises(vr3.CertifyError, match="PROCESS_VISIBILITY_UNAVAILABLE"):
            vr3.apply(args)
        assert raw.exists() and not args.journal.exists()
        return
    if fault == "raw_drift":
        raw.write_bytes(b"changed raw")
    if fault == "consumer":
        (tmp_path / "experiments/consumer.py").write_text(f"RAW = {str(raw)!r}")
    if fault == "archive_drift":
        (tmp_path / "runtime/archive.zip").write_bytes(b"drift")
    result = vr3.apply(args)
    phases = [json.loads(line)["phase"] for line in args.journal.read_text().splitlines()]
    if fault == "none":
        assert result == 0 and not raw.exists()
        assert phases == ["APPLY_START", "PRE_DELETE", "DELETED", "APPLY_COMPLETE"]
        assert json.loads(args.ledger.read_text())["verdict"] == "DELETED"
    else:
        assert result == 3 and raw.exists()
        assert "DELETED" not in phases
        assert json.loads(args.ledger.read_text())["verdict"].startswith("BLOCKED:")
    assert (tmp_path / "runtime/archive.zip").exists()
    assert (tmp_path / "runtime/inflate.sh").exists()


def test_observation_pin_cannot_hide_changed_consumer(tmp_path):
    observation = tmp_path / "observation.json"
    observation.write_text("inventory only")
    pin = vr3.pinned_file(observation)
    assert vr3.observation_exclusions([pin], tmp_path) == ["observation.json"]
    observation.write_text("live consumer added")
    with pytest.raises(vr3.CertifyError, match="PIN_SHA256_DRIFT"):
        vr3.observation_exclusions([pin], tmp_path)


def test_plan_requires_complete_unique_main_rehash(tmp_path, monkeypatch):
    raw, args, _ = make_plan(tmp_path, monkeypatch)
    args.rehash_ledger.write_text("")
    with pytest.raises(vr3.CertifyError, match="empty or duplicate"):
        vr3.plan_retained(args)
    assert raw.exists()


def test_ap_root_admits_same_certificate_without_new_family_constant(tmp_path, monkeypatch):
    raw, source, rehash, closure = fixture_chain(tmp_path, monkeypatch)
    monkeypatch.setattr(vr3, "AP_ROOT", tmp_path)
    monkeypatch.setattr(vr3, "VERTIGO_ROOT", tmp_path / "absent_vertigo")
    assert vr3.storage_root(raw) == tmp_path
    assert vr3.certify_retained(source, rehash, closure)["decode_receipt"]["equals_file_sha256"]


def test_raw_hash_drift_even_when_metadata_restored(tmp_path, monkeypatch):
    raw, _, args = make_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(vr3, "process_gate", lambda owner: {"visible": True})
    monkeypatch.setattr(vr3, "lsof_plus_d", lambda path: {"open_descriptor_rows": []})
    stat = raw.stat()
    raw.write_bytes(b"changed raw")
    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert vr3.apply(args) == 3
    row = json.loads(args.ledger.read_text())
    assert "RAW_SHA256_DRIFT_AT_APPLY" in row["verdict"]
    assert raw.exists()


def test_lsof_unavailable_refuses(monkeypatch):
    def denied(*args, **kwargs):
        raise FileNotFoundError("lsof")

    monkeypatch.setattr(vr3.subprocess, "run", denied)
    with pytest.raises(vr3.CertifyError, match="lsof unavailable"):
        vr3.lsof_plus_d(Path("."))


def test_legacy_apply_still_deletes_with_original_certificate(tmp_path, monkeypatch):
    raw, _, args = make_plan(tmp_path, monkeypatch)
    row = json.loads(args.ledger.read_text())
    row.pop("retained_admission")
    row["family"] = "AP1_TERMINAL_ADVISORY"
    vr3.atomic_jsonl(args.ledger, [row])
    args.expected_ledger_sha256 = vr3.sha256_file(args.ledger)
    monkeypatch.setattr(vr3, "TARGET_BYTES", row["bytes"])
    monkeypatch.setattr(vr3, "_selected_family", lambda path: ("AP1_TERMINAL_ADVISORY", "carrier_l1"))
    monkeypatch.setattr(vr3, "process_gate", lambda owner: {"visible": True})
    monkeypatch.setattr(vr3, "lsof_plus_d", lambda path: {"open_descriptor_rows": []})
    assert vr3.apply(args) == 0
    assert not raw.exists()


def test_ignored_live_pointer_still_blocks_apply(tmp_path, monkeypatch):
    raw, _, args = make_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(vr3, "process_gate", lambda owner: {"visible": True})
    monkeypatch.setattr(vr3, "lsof_plus_d", lambda path: {"open_descriptor_rows": []})
    (tmp_path / ".gitignore").write_text(".omx/state/*.json\n")
    _write_json(tmp_path / ".omx/state/canonical_frontier_pointer.json", {"raw": str(raw)})
    assert vr3.apply(args) == 3
    assert "REPOSITORY_REFERENCE_HIT_AT_APPLY" in json.loads(args.ledger.read_text())["verdict"]
    assert raw.exists()
