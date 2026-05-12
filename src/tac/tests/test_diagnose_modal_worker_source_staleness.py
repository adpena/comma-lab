"""Tests for ``tools/diagnose_modal_worker_source_staleness.py``.

Coverage targets:

- _classify_divergence emits HOK when worker matches HEAD blob hashes
- H4 fired when sentinel file MISSING on worker
- H5 fired when worker tac.__file__ NOT under /workspace/pact
- H3 fired when env-injected SHA != worker-side git SHA
- H2 fired when worker hash matches local disk but not HEAD blob
- H1 fired when worker hash differs from HEAD blob (and disk too)
- _local_file_sha256 / _local_head_blob_sha256 round-trip
- offline-fixture mode: skip Modal spawn, classify against fixture JSON
- main() returns 0 when verdict=HOK, 4 otherwise
- main() writes structured report when --output-json
- _build_remote_probe_payload yields executable Python with the sentinel list
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from diagnose_modal_worker_source_staleness import (
    SENTINEL_FILES,
    _build_remote_probe_payload,
    _classify_divergence,
    _local_file_sha256,
    _local_head_blob_sha256,
    main,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_classify_hok_when_worker_matches_head() -> None:
    head_blobs = {rel: _sha(f"head-{rel}") for rel in SENTINEL_FILES}
    worker = dict(head_blobs)
    worker["__head_env__"] = "abc123def4567"
    worker["__head_git__"] = "abc123def4567"
    worker["__tac_file__"] = "/workspace/pact/src/tac/__init__.py"
    code, text = _classify_divergence(
        worker_head_env="abc123def4567",
        worker_head_git="abc123def4567",
        local_head="abc123def4567",
        worker_hashes=worker,
        local_head_blob_hashes=head_blobs,
        local_disk_hashes=head_blobs,
    )
    assert code == "HOK"


def test_h4_fired_when_sentinel_missing() -> None:
    head_blobs = {rel: _sha(f"head-{rel}") for rel in SENTINEL_FILES}
    worker = dict(head_blobs)
    worker[SENTINEL_FILES[0]] = "MISSING"
    worker["__head_env__"] = "abc"
    worker["__head_git__"] = "abc"
    worker["__tac_file__"] = "/workspace/pact/src/tac/__init__.py"
    code, _ = _classify_divergence(
        worker_head_env="abc",
        worker_head_git="abc",
        local_head="abc",
        worker_hashes=worker,
        local_head_blob_hashes=head_blobs,
        local_disk_hashes=head_blobs,
    )
    assert code == "H4"


def test_h5_fired_when_tac_file_not_under_workspace() -> None:
    head_blobs = {rel: _sha(f"head-{rel}") for rel in SENTINEL_FILES}
    worker = dict(head_blobs)
    worker["__head_env__"] = "abc"
    worker["__head_git__"] = "abc"
    worker["__tac_file__"] = "/usr/local/lib/python3.11/site-packages/tac/__init__.py"
    code, _ = _classify_divergence(
        worker_head_env="abc",
        worker_head_git="abc",
        local_head="abc",
        worker_hashes=worker,
        local_head_blob_hashes=head_blobs,
        local_disk_hashes=head_blobs,
    )
    assert code == "H5"


def test_h3_fired_when_env_head_diverges_from_git_head() -> None:
    head_blobs = {rel: _sha(f"head-{rel}") for rel in SENTINEL_FILES}
    worker = dict(head_blobs)
    worker["__head_env__"] = "newSha1234567"
    worker["__head_git__"] = "oldSha7654321"
    worker["__tac_file__"] = "/workspace/pact/src/tac/__init__.py"
    code, _ = _classify_divergence(
        worker_head_env="newSha1234567",
        worker_head_git="oldSha7654321",
        local_head="newSha1234567",
        worker_hashes=worker,
        local_head_blob_hashes=head_blobs,
        local_disk_hashes=head_blobs,
    )
    assert code == "H3"


def test_h2_fired_when_worker_matches_disk_but_not_head() -> None:
    head_blobs = {rel: _sha(f"head-{rel}") for rel in SENTINEL_FILES}
    disk_hashes = {rel: _sha(f"disk-{rel}") for rel in SENTINEL_FILES}
    worker = dict(disk_hashes)
    worker["__head_env__"] = "abc"
    worker["__head_git__"] = "abc"
    worker["__tac_file__"] = "/workspace/pact/src/tac/__init__.py"
    code, _ = _classify_divergence(
        worker_head_env="abc",
        worker_head_git="abc",
        local_head="abc",
        worker_hashes=worker,
        local_head_blob_hashes=head_blobs,
        local_disk_hashes=disk_hashes,
    )
    assert code == "H2"


def test_h1_fired_when_worker_differs_from_disk_and_head() -> None:
    head_blobs = {rel: _sha(f"head-{rel}") for rel in SENTINEL_FILES}
    disk_hashes = {rel: _sha(f"disk-{rel}") for rel in SENTINEL_FILES}
    worker = {rel: _sha(f"image-{rel}") for rel in SENTINEL_FILES}
    worker["__head_env__"] = "abc"
    worker["__head_git__"] = "abc"
    worker["__tac_file__"] = "/workspace/pact/src/tac/__init__.py"
    code, _ = _classify_divergence(
        worker_head_env="abc",
        worker_head_git="abc",
        local_head="abc",
        worker_hashes=worker,
        local_head_blob_hashes=head_blobs,
        local_disk_hashes=disk_hashes,
    )
    assert code == "H1"


def test_local_file_sha256_round_trip(tmp_path: Path) -> None:
    f = tmp_path / "hello.py"
    f.write_text("hello\n")
    repo = tmp_path
    rel = "hello.py"
    expected = hashlib.sha256(b"hello\n").hexdigest()
    assert _local_file_sha256(repo, rel) == expected


def test_local_file_sha256_returns_missing_when_absent(tmp_path: Path) -> None:
    assert _local_file_sha256(tmp_path, "no-such-file.py") == "MISSING"


def test_local_head_blob_sha256_in_real_git_repo(tmp_path: Path) -> None:
    """Build a tiny git repo, commit a file, verify HEAD-blob hash."""
    repo = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "src").mkdir()
    (repo / "src" / "x.py").write_text("HEAD content\n")
    subprocess.run(["git", "add", "src/x.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )
    expected = hashlib.sha256(b"HEAD content\n").hexdigest()
    assert _local_head_blob_sha256(repo, "src/x.py") == expected


def test_local_head_blob_sha256_returns_missing_when_unknown(tmp_path: Path) -> None:
    repo = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "x.py").write_text("seed\n")
    subprocess.run(["git", "add", "x.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )
    assert _local_head_blob_sha256(repo, "no-such-file") == "MISSING"


def test_offline_fixture_mode_classifies_against_json(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    # Build minimal sentinel files so blob hashes are computable.
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    for rel in SENTINEL_FILES:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(f"# stub {rel}\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )
    # Build a worker fixture JSON that DOES match HEAD blobs.
    head_blobs = {
        rel: hashlib.sha256((f"# stub {rel}\n").encode()).hexdigest()
        for rel in SENTINEL_FILES
    }
    worker = dict(head_blobs)
    worker["__head_env__"] = "abc"
    worker["__head_git__"] = "abc"
    worker["__tac_file__"] = "/workspace/pact/src/tac/__init__.py"
    fixture = tmp_path / "worker_fixture.json"
    fixture.write_text(json.dumps(worker))
    out_json = tmp_path / "report.json"
    rc = main([
        "--offline-fixture", str(fixture),
        "--output-json", str(out_json),
        "--repo-root", str(repo),
        "--quiet",
    ])
    assert rc == 0
    report = json.loads(out_json.read_text())
    assert report["verdict_code"] == "HOK"


def test_offline_fixture_h4_returns_rc4(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    for rel in SENTINEL_FILES:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(f"# stub {rel}\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )
    worker = dict.fromkeys(SENTINEL_FILES, "MISSING")
    worker["__head_env__"] = "abc"
    worker["__head_git__"] = "abc"
    worker["__tac_file__"] = "/workspace/pact/src/tac/__init__.py"
    fixture = tmp_path / "worker_fixture.json"
    fixture.write_text(json.dumps(worker))
    rc = main([
        "--offline-fixture", str(fixture),
        "--repo-root", str(repo),
        "--quiet",
    ])
    assert rc == 4


def test_remote_probe_payload_includes_sentinel_list_and_imports() -> None:
    payload = _build_remote_probe_payload()
    # Sentinel paths must be embedded
    for rel in SENTINEL_FILES:
        assert rel in payload
    # Must populate the result dict via the worker-side variables
    assert "import tac" in payload
    assert "result" in payload
    assert "__tac_file__" in payload
    assert "__head_env__" in payload
    assert "__head_git__" in payload


def test_main_returns_2_when_offline_fixture_missing(tmp_path: Path) -> None:
    rc = main([
        "--offline-fixture", str(tmp_path / "no-such-fixture.json"),
        "--repo-root", str(tmp_path),
        "--quiet",
    ])
    assert rc == 2


def test_main_returns_3_when_worker_exec_failed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "x.py").write_text("seed\n")
    subprocess.run(["git", "add", "x.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )
    fixture = tmp_path / "broken_worker.json"
    fixture.write_text(json.dumps({"__exec_error__": "RuntimeError: boom"}))
    rc = main([
        "--offline-fixture", str(fixture),
        "--repo-root", str(repo),
        "--quiet",
    ])
    assert rc == 3
