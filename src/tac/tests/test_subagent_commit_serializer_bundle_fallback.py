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
