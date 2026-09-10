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


@pytest.mark.parametrize(
    "mode", ["ps_denied", "pgrep_denied", "lsof_denied", "blind", "live_command", "live_cwd", "lsof_warning"]
)
def test_process_visibility_fail_closed(monkeypatch, mode):
    def run(command, **kwargs):
        if command[0] == "lsof":
            if mode in {"lsof_denied", "lsof_warning"}:
                return subprocess.CompletedProcess(command, 1, "", "cannot stat mount")
            cwd = "/work/ddm_test_run" if mode == "live_cwd" else "/work/idle"
            return subprocess.CompletedProcess(command, 0, f"p1\nclaunchd\nn/\np123\ncpython3\nn{cwd}\np{os.getpid()}\ncpytest\nn/work/repo\n", "")
        if command[0] == "ps":
            if mode == "ps_denied":
                raise PermissionError("sandbox")
            owner_command = "/usr/local/bin/ddm_test_worker" if mode == "live_command" else "/usr/bin/python3"
            text = f"1 0 /sbin/launchd\n123 1 {owner_command}\n{os.getpid()} 1 /usr/bin/pytest\n"
            return subprocess.CompletedProcess(command, 0, "" if mode == "blind" else text, "")
        return subprocess.CompletedProcess(
            command,
            3 if mode == "pgrep_denied" else 0 if mode == "live_command" else 1,
            "123\n" if mode == "live_command" else "",
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
        if command[0] == "lsof":
            return subprocess.CompletedProcess(command, 0, f"p1\nclaunchd\nn/\np{os.getpid()}\ncpytest\nn/work/repo\n", "")
        return subprocess.CompletedProcess(
            command,
            0 if command[0] == "ps" else 1,
            f"1 0 /sbin/launchd\n{os.getpid()} 1 /usr/bin/pytest\n" if command[0] == "ps" else "",
            "",
        )

    monkeypatch.setattr(vr3.subprocess, "run", run)
    assert vr3.process_gate("ddm_test")["visible"]


def test_owner_token_only_in_argv_does_not_trip_process_gate(monkeypatch):
    def run(command, **kwargs):
        if command[0] == "ps":
            assert command == ["ps", "-axo", "pid=,ppid=,comm="]
            return subprocess.CompletedProcess(
                command,
                0,
                f"1 0 /sbin/launchd\n123 1 /usr/bin/ruff\n{os.getpid()} 1 /usr/bin/pytest\n",
                "",
            )
        if command[0] == "pgrep":
            assert command == ["pgrep", "-x", "ddm_test"]
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(
            command, 0, f"p1\nclaunchd\nn/\np123\ncpython3\nn/work/review\np{os.getpid()}\ncpytest\nn/work/repo\n", ""
        )

    monkeypatch.setattr(vr3.subprocess, "run", run)
    assert vr3.process_gate("ddm_test")["visible"]


@pytest.mark.parametrize(
    "cwd_output",
    [
        "",  # Successful but empty lsof is not visibility.
        "self_only",  # A hidden Python consumer could have the owner's cwd.
        "p1\nclaunchd\nn/\np123\n",  # Missing name on the last process.
        "p1\nclaunchd\nn/\np123\ncpython3\nnrelative\n",
        "p1\nclaunchd\nn/\np123\ncpython3\nn/work/idle\nn/work/other\n",
        "p1\nclaunchd\nn/\np123\ncpython3\nn/work/idle\np123\ncpython3\nn/work/other\n",
        "p1\nclaunchd\nn/\np123\ncpython3\nn/work/idle\nERROR hidden mount\n",
        "p1\nclaunchd\nn/\np999\nn/work/new-process\n",
    ],
)
def test_incomplete_or_malformed_cwd_census_refused(monkeypatch, cwd_output):
    def run(command, **kwargs):
        if command[0] == "ps":
            output = f"1 0 /sbin/launchd\n123 1 /usr/bin/python3\n{os.getpid()} 1 /usr/bin/pytest\n"
        elif command[0] == "pgrep":
            return subprocess.CompletedProcess(command, 1, "", "")
        else:
            output = f"p{os.getpid()}\ncpytest\nn/work/repo\n" if cwd_output == "self_only" else cwd_output
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(vr3.subprocess, "run", run)
    with pytest.raises(vr3.CertifyError, match="PROCESS_VISIBILITY_UNAVAILABLE"):
        vr3.process_gate("ddm_test")


@pytest.mark.parametrize(
    ("comm", "cwd", "refuse"),
    [
        ("/usr/bin/ddm_test_worker", "/work/idle", True),
        ("/usr/bin/python3", "/work/ddm_test-run", True),
        ("/usr/bin/notddm_test", "/work/notddm_test", False),
        ("/usr/bin/ddm_test2", "/work/ddm_test2", False),
        ("/usr/bin/ruff", "/work/review", False),
    ],
)
def test_owner_token_boundary_on_command_and_cwd(monkeypatch, comm, cwd, refuse):
    def run(command, **kwargs):
        if command[0] == "ps":
            output = f"1 0 /sbin/launchd\n123 1 {comm}\n{os.getpid()} 1 /usr/bin/pytest\n"
        elif command[0] == "pgrep":
            return subprocess.CompletedProcess(command, 1, "", "")
        else:
            output = f"p1\nclaunchd\nn/\np123\ncpython3\nfcwd\nn{cwd}\np{os.getpid()}\ncpytest\nn/work/repo\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(vr3.subprocess, "run", run)
    if refuse:
        with pytest.raises(vr3.CertifyError, match="LIVE_OWNER_PROCESS"):
            vr3.process_gate("ddm_test")
    else:
        assert vr3.process_gate("ddm_test")["visible"]


@pytest.mark.parametrize("extra_command", ["lsof", "ddm_test_worker"])
def test_observer_lifecycle_and_new_process_command(monkeypatch, extra_command):
    def run(command, **kwargs):
        if command[0] == "ps":
            output = f"1 0 launchd\n{os.getpid()} 1 pytest\n321 {os.getpid()} /bin/ps\n"
        elif command[0] == "pgrep":
            return subprocess.CompletedProcess(command, 1, "", "")
        else:
            assert command[-1] == "pcn"
            output = f"p1\nclaunchd\nn/\np{os.getpid()}\ncpytest\nn/work\np456\nc{extra_command}\nn/work\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(vr3.subprocess, "run", run)
    if extra_command == "lsof":
        assert vr3.process_gate("ddm_test")["visible"]
    else:
        with pytest.raises(vr3.CertifyError, match="LIVE_OWNER_PROCESS"):
            vr3.process_gate("ddm_test")


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


def test_reclaim_custody_artifacts_are_not_consumer_references(tmp_path):
    # MAIN 2026-09-10: the first vr5 apply refused all 17 rows because the raw's path
    # appeared inside vr5's OWN serializer format-patch (a copy of the plan ledger).
    # Custody artifacts of the reclaim family name the path in order to certify it.
    mod = vr3
    repo = tmp_path / "repo"
    (repo / ".omx" / "research" / "ddm_vr9_20260910" / "serializer").mkdir(parents=True)
    (repo / "experiments").mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    raw = "/Volumes/APDataStore/pact/ddm_zz9/advisory/work/inflated/0.raw"
    custody = repo / ".omx" / "research" / "ddm_vr9_20260910" / "serializer" / "intended-commit.format-patch"
    custody.write_text('+{"path": "' + raw + '", "verdict": "DELETABLE"}\n')
    (repo / ".omx" / "research" / "ddm_vr9_reclaim_plan_20260910.jsonl").write_text('{"path": "' + raw + '"}\n')
    hits = mod.repo_reference_hits({raw: [raw]}, repo)[raw]
    assert hits == [], hits
    consumer = repo / "experiments" / "consumer.py"
    consumer.write_text('RAW = "' + raw + '"\n')
    hits = mod.repo_reference_hits({raw: [raw]}, repo)[raw]
    assert any("experiments/consumer.py" in h for h in hits), hits


@pytest.mark.parametrize("shape", ["omitted", "command_only", "last_command_only"])
def test_visible_host_records_unreadable_cwd(monkeypatch, shape):
    self_pid = os.getpid()
    missing = "p123\ncpython3\n"
    output = f"p414\ncloginwindow\nfcwd\nn/\np{self_pid}\ncpytest\nfcwd\nn/work/repo\n"
    if shape == "command_only":
        output = missing + output
    elif shape == "last_command_only":
        output += missing

    def run(command, **kwargs):
        if command[0] == "ps":
            return subprocess.CompletedProcess(command, 0, f"1 0 launchd\n414 1 loginwindow\n123 1 python3\n{self_pid} 1 pytest\n", "")
        if command[0] == "pgrep":
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(vr3.subprocess, "run", run)
    result = vr3.process_gate("ddm_bz2d")
    assert result["status"] == "COMPLETE"
    assert result["cwd_unreadable"] == [
        {"pid": 1, "command": "launchd", "cwd_unreadable": True},
        {"pid": 123, "command": "python3", "cwd_unreadable": True},
    ]


@pytest.mark.parametrize("source", ["ps", "lsof"])
def test_unreadable_owner_command_refuses(monkeypatch, source):
    def run(command, **kwargs):
        if command[0] == "ps":
            owner = "321 1 /work/ddm_bz2d_worker\n" if source == "ps" else ""
            return subprocess.CompletedProcess(command, 0, f"1 0 launchd\n{os.getpid()} 1 pytest\n{owner}", "")
        if command[0] == "pgrep":
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, 0, f"p1\nclaunchd\nn/\np{os.getpid()}\ncpytest\nn/work/repo\np321\ncddm_bz2d_worker\n", "")

    monkeypatch.setattr(vr3.subprocess, "run", run)
    with pytest.raises(vr3.CertifyError, match="LIVE_OWNER_PROCESS"):
        vr3.process_gate("ddm_bz2d")


def test_captured_host_lsof_without_init_with_controlled_ps(monkeypatch):
    # lsof bytes are real host input; ps/self identity below are explicit controls.
    capture = json.loads((Path(__file__).parent / "fixtures/ddm_vg1_sandbox_process_capture.json").read_text())
    observed = capture[2]
    pids = [int(line[1:]) for line in observed["stdout"].splitlines() if line.startswith("p")]
    assert 1 not in pids
    monkeypatch.setattr(vr3.os, "getpid", lambda: pids[0])

    def run(command, **kwargs):
        if command[0] == "ps":
            output = "1 0 launchd\n" + "".join(f"{pid} 1 controlled_command\n" for pid in pids)
            return subprocess.CompletedProcess(command, 0, output, "")
        if command[0] == "pgrep":
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, observed["returncode"], observed["stdout"], observed["stderr"])

    monkeypatch.setattr(vr3.subprocess, "run", run)
    result = vr3.process_gate("ddm_vg1fixture")
    assert result["cwd_readable_count"] == len(pids)
    assert result["cwd_unreadable"] == [{"pid": 1, "command": "launchd", "cwd_unreadable": True}]


def test_captured_sandbox_ps_denial_remains_closed(monkeypatch):
    capture = json.loads((Path(__file__).parent / "fixtures/ddm_vg1_sandbox_process_capture.json").read_text())
    assert "Operation not permitted" in capture[0]["error"]

    def run(command, **kwargs):
        assert command[0] == "ps"
        raise PermissionError(capture[0]["error"])

    monkeypatch.setattr(vr3.subprocess, "run", run)
    with pytest.raises(vr3.CertifyError, match="PROCESS_VISIBILITY_UNAVAILABLE"):
        vr3.process_gate("ddm_bz2d")


@pytest.mark.parametrize("role", ["owner", "reviewer"])
def test_live_named_process_and_argv_only_reviewer(tmp_path, role):
    # Read-only gate integration; no raw, archive, apply, or reclaim is involved.
    try:
        visibility = subprocess.run(["ps", "-axo", "pid=,ppid=,comm="], capture_output=True, text=True, check=False)
    except OSError as exc:
        pytest.skip(f"live ps visibility unavailable: {exc}")
    if visibility.returncode or visibility.stderr:
        pytest.skip(f"live ps visibility unavailable: {visibility.stderr}")
    import shutil

    family = "ddm_bz2d"
    work = tmp_path / (family if role == "owner" else "review")
    work.mkdir()
    executable = work / (family if role == "owner" else "reviewer")
    shutil.copyfile("/bin/sleep" if role == "owner" else "/bin/sh", executable)
    executable.chmod(0o700)
    argv = [str(executable), "30"] if role == "owner" else [str(executable), "-c", "read answer", family]
    child = subprocess.Popen(argv, cwd=work, stdin=subprocess.PIPE)
    try:
        assert child.poll() is None
        if role == "owner":
            with pytest.raises(vr3.CertifyError, match="LIVE_OWNER_PROCESS"):
                vr3.process_gate(family)
        else:
            assert vr3.process_gate(family)["visible"]
    finally:
        child.terminate()
        child.communicate(timeout=5)


@pytest.mark.parametrize("mode", ["observer_only", "warning", "duplicate_unreadable"])
def test_partial_census_cannot_claim_visibility(monkeypatch, mode):
    def run(command, **kwargs):
        if command[0] == "ps":
            return subprocess.CompletedProcess(command, 0, f"1 0 launchd\n123 1 python3\n{os.getpid()} 1 pytest\n", "")
        if command[0] == "pgrep":
            return subprocess.CompletedProcess(command, 1, "", "")
        output = f"p{os.getpid()}\ncpytest\nn/work/repo\np456\nclsof\nn/work/repo\n"
        if mode == "duplicate_unreadable":
            output += "p123\ncpython3\np123\ncpython3\n"
        return subprocess.CompletedProcess(command, 0, output, "permission warning" if mode == "warning" else "")

    monkeypatch.setattr(vr3.subprocess, "run", run)
    with pytest.raises(vr3.CertifyError, match="PROCESS_VISIBILITY_UNAVAILABLE"):
        vr3.process_gate("ddm_bz2d")
