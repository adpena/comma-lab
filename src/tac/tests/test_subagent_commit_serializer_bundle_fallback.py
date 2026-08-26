# SPDX-License-Identifier: MIT
"""Executed controls for #1293's git-object-denial bundle fallback.

The positive control makes ``.git/objects`` read-only only inside a throwaway
repository.  The live repository is never chmod'd or otherwise simulated
against.  The negative control proves an ordinary serializer commit keeps its
existing rc=0 behavior and creates no fallback artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
SERIALIZER = REPO / "tools" / "subagent_commit_serializer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_serializer_bundle_fallback", SERIALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=check
    )


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.name", "Serializer Control")
    _git(path, "config", "user.email", "serializer-control@example.invalid")
    _git(path, "config", "commit.gpgsign", "false")
    (path / ".omx" / "state").mkdir(parents=True)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(path, "add", "seed.txt")
    _git(path, "commit", "-m", "seed")
    return path


def _run_serializer(repo: Path, receipt_dir: Path, target: str) -> subprocess.CompletedProcess[str]:
    payload = repo / target
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    env = dict(os.environ)
    env["REVIEW_GATE_OVERRIDE"] = "1"  # control commits a text fixture, never Python
    return subprocess.run(
        [
            sys.executable,
            str(SERIALIZER),
            "--repo-root",
            str(repo),
            "--message",
            "serializer fallback control [no-triality] [p0-ledger-ok]",
            "--files",
            target,
            "--expected-content-sha256",
            f"{target}={digest}",
            "--fallback-receipt-dir",
            str(receipt_dir),
            "--no-sister-checkpoint-check",
            "--label",
            "ddm_hd1_control",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _set_object_directories_writable(objects: Path, writable: bool) -> None:
    mode = 0o755 if writable else 0o555
    for directory in [objects, *(path for path in objects.rglob("*") if path.is_dir())]:
        directory.chmod(mode)


def test_denial_signature_is_specific_to_git_object_store_writes() -> None:
    module = _load_module()
    assert module._is_git_object_write_denial(
        "error: unable to create temporary file: Operation not permitted\n"
        "error: payload.bin: failed to insert into database"
    )
    assert module._is_git_object_write_denial(
        "error: insufficient permission for adding an object to repository database .git/objects"
    )
    assert not module._is_git_object_write_denial("Operation not permitted: unrelated hook output")
    assert not module._is_git_object_write_denial("failed to insert into database")


def test_positive_read_only_objects_emits_verified_bundle_and_typed_receipt(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path / "denied")
    target = repo / "intended.txt"
    target.write_text("the exact intended content\n", encoding="utf-8")
    base_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    objects = repo / ".git" / "objects"
    _set_object_directories_writable(objects, writable=False)
    try:
        proc = _run_serializer(repo, tmp_path / "receipts", target.name)
    finally:
        _set_object_directories_writable(objects, writable=True)

    assert proc.returncode == 17, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "BUNDLE_FALLBACK rc=17 phase=git_add" in proc.stdout
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == base_head
    assert _git(repo, "status", "--short", "--", target.name).stdout.startswith("??")

    receipts = list((tmp_path / "receipts").rglob("receipts.jsonl"))
    assert len(receipts) == 1
    row = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert row["schema"] == "subagent_commit_bundle_fallback.v1"
    assert row["status"] == "BUNDLE_READY_MAIN_MUST_LAND"
    assert row["construction_mode"] == "isolated_git_plumbing_no_checkout"
    assert row["projected_artifact_bytes"] <= row["artifact_cap_bytes"]
    assert row["storage_reserve"]["selected"]["eligible"] is True
    assert row["failure"]["serializer_rc"] != 0
    assert row["environment"]["cwd"] == str(repo)
    assert row["environment"]["uid"] == os.getuid()
    assert row["files"] == [
        {
            "path": target.name,
            "content_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    ]

    bundle = Path(row["bundle_path"])
    assert bundle.is_file()
    assert bundle.stat().st_size == row["bundle_bytes"]
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == row["bundle_sha256"]
    verify = _git(repo, "bundle", "verify", str(bundle), check=False)
    assert verify.returncode == 0, verify.stderr

    harvest = tmp_path / "harvest"
    _git(tmp_path, "clone", "--quiet", "--shared", str(repo), str(harvest))
    fetched = _git(
        harvest,
        "fetch",
        str(bundle),
        "refs/heads/serializer-fallback",
        check=False,
    )
    assert fetched.returncode == 0, fetched.stderr
    landed = _git(harvest, "show", f"FETCH_HEAD:{target.name}").stdout
    assert landed == "the exact intended content\n"


def test_storage_reserve_refuses_before_creating_ssd_receipt_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo(tmp_path / "reserve-refusal")
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", repo)
    reserve = module._canonical_storage_reserve_bytes()
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=reserve * 2, used=reserve, free=reserve - 1),
    )
    requested_parent = tmp_path / "ssd-receipts"

    with pytest.raises(module.BundleFallbackStorageRefusal) as caught:
        module._author_bundle_fallback(
            files=["intended.txt"],
            final_message="reserve refusal control",
            label="ddm_fc1x_control",
            original_rc=128,
            original_output="Operation not permitted: failed to insert into database",
            fallback_receipt_dir=str(requested_parent),
            intended_snapshot={"intended.txt": (b"bounded intent\n", "100644")},
            patch_bytes=None,
            allow_empty=False,
        )

    assert not requested_parent.exists()
    receipt = caught.value.receipt_path
    assert receipt.is_file()
    assert str(receipt).startswith(str(repo / ".omx" / "state"))
    row = json.loads(receipt.read_text(encoding="utf-8"))
    assert row["status"] == "BUNDLE_FALLBACK_STORAGE_REFUSED"
    assert row["serializer_rc"] == 19
    assert row["storage_candidates"][0]["eligible"] is False


def test_main_returns_loud_typed_rc19_for_storage_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _make_repo(tmp_path / "typed-refusal")
    target = repo / "intended.txt"
    target.write_text("typed refusal\n", encoding="utf-8")
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(module, "LOCK_PATH", repo / ".omx" / "state" / ".commit-lock")
    monkeypatch.setattr(module, "LOG_PATH", repo / ".omx" / "state" / "commit-serializer.log")
    reserve = module._canonical_storage_reserve_bytes()
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=reserve * 2, used=reserve, free=reserve - 1),
    )
    monkeypatch.setenv("REVIEW_GATE_OVERRIDE", "1")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subagent_commit_serializer.py",
            "--message",
            "typed reserve refusal [no-triality] [p0-ledger-ok]",
            "--files",
            target.name,
            "--expected-content-sha256",
            f"{target.name}={hashlib.sha256(target.read_bytes()).hexdigest()}",
            "--fallback-receipt-dir",
            str(tmp_path / "typed-ssd-receipts"),
            "--no-sister-checkpoint-check",
            "--label",
            "ddm_fc1x_typed_control",
        ],
    )
    objects = repo / ".git" / "objects"
    _set_object_directories_writable(objects, writable=False)
    try:
        rc = module.main()
    finally:
        _set_object_directories_writable(objects, writable=True)

    captured = capsys.readouterr()
    assert rc == 19
    assert "BUNDLE_FALLBACK_STORAGE_REFUSED rc=19 phase=git_add" in captured.err
    assert not (tmp_path / "typed-ssd-receipts").exists()


def test_patch_intent_uses_same_clone_free_bundle_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo(tmp_path / "patch-intent")
    (repo / "seed.txt").write_text("patched through exact intent\n", encoding="utf-8")
    patch_bytes = _git(repo, "diff", "--binary", "--", "seed.txt").stdout.encode()
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", repo)

    row = module._author_bundle_fallback(
        files=["seed.txt"],
        final_message="patch intent control",
        label="ddm_fc1x_patch_control",
        original_rc=128,
        original_output="Operation not permitted: failed to insert into database",
        fallback_receipt_dir=str(tmp_path / "patch-receipts"),
        intended_snapshot=None,
        patch_bytes=patch_bytes,
        allow_empty=False,
    )

    assert row["construction_mode"] == "isolated_git_plumbing_no_checkout"
    harvest = tmp_path / "patch-harvest"
    _git(tmp_path, "clone", "--quiet", "--shared", str(repo), str(harvest))
    fetched = _git(
        harvest,
        "fetch",
        str(row["bundle_path"]),
        "refs/heads/serializer-fallback",
        check=False,
    )
    assert fetched.returncode == 0, fetched.stderr
    assert _git(harvest, "show", "FETCH_HEAD:seed.txt").stdout == "patched through exact intent\n"


def test_negative_normal_commit_is_unchanged_and_emits_no_fallback(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "normal")
    target = repo / "ordinary.txt"
    target.write_text("ordinary commit\n", encoding="utf-8")
    receipt_dir = tmp_path / "should-not-exist"

    proc = _run_serializer(repo, receipt_dir, target.name)

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "BUNDLE_FALLBACK" not in proc.stdout + proc.stderr
    assert not receipt_dir.exists()
    assert _git(repo, "show", f"HEAD:{target.name}").stdout == "ordinary commit\n"
