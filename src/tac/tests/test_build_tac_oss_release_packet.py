# SPDX-License-Identifier: MIT
"""Tests for tools/build_tac_oss_release_packet.py — sanitization gate exercises.

Covers per CLAUDE.md "Public Disclosure Hygiene":
  - sanitization gate refuses on local absolute path leak
  - sanitization gate refuses on private-key material
  - sanitization gate refuses on API token leak
  - sanitization gate refuses on Vast SSH endpoint leak
  - sanitization gate refuses on operator-supplied README leak
  - clean surface produces a release packet with manifest + receipt
  - manifest schema-version stamped + per-file SHA-256 + total bytes
  - overwrite=False raises FileExistsError
  - overwrite=True rebuilds in place
  - empty version refused
  - missing readme path refused
  - hard-excluded paths never appear in the packet
  - included extensions enforced
  - manifest paths are all relative (NEVER /tmp or absolute)
  - generated README placeholder used when no readme supplied
  - main() returns 0 on success, non-zero on failure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure tools/ is importable
TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_tac_oss_release_packet as oss_pkt  # noqa: E402


# ── Fixture helpers ────────────────────────────────────────────────────────


def _make_clean_repo(root: Path) -> None:
    """Build a synthetic mini-repo with the canonical layout but only clean files."""
    (root / "src" / "tac").mkdir(parents=True)
    (root / "src" / "tac" / "__init__.py").write_text(
        '"""tac — public release."""\nVERSION = "0.5.0"\n', encoding="utf-8"
    )
    (root / "src" / "tac" / "core.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (root / "src" / "tac" / "tests").mkdir()
    (root / "src" / "tac" / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "tac" / "tests" / "test_core.py").write_text(
        "from tac.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (root / "docs" / "paper").mkdir(parents=True)
    (root / "docs" / "paper" / "method.md").write_text(
        "# Method\nClean public docs.\n", encoding="utf-8"
    )
    (root / "LICENSE").write_text("Apache 2.0 placeholder\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "tac"\nversion = "0.5.0"\n', encoding="utf-8"
    )


def _inject_leak(root: Path, rel: str, leak_text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(leak_text, encoding="utf-8")


# ── Sanitization gate refusals ─────────────────────────────────────────────


def test_clean_surface_builds_packet(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    manifest = oss_pkt.build_packet(out, "0.5.0", repo_root=repo)
    assert manifest["schema"] == oss_pkt.RELEASE_PACKET_SCHEMA
    assert manifest["version"] == "0.5.0"
    assert (out / "release_manifest.json").is_file()
    assert (out / "sanitization_receipt.json").is_file()


def test_local_absolute_operator_path_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(repo, "src/tac/leaky.py", "PATH = '/Users/operator/private/dir'\n")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(out, "0.5.0", repo_root=repo)
    assert not out.exists() or not (out / "release_manifest.json").exists()


def test_private_key_material_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(
        repo, "src/tac/keys.py",
        "KEY = '''-----BEGIN OPENSSH PRIVATE KEY-----\\nfake\\n-----END-----'''\n"
    )
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(tmp_path / "out", "0.5.0", repo_root=repo)


def test_openai_token_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(
        repo, "src/tac/secrets.py",
        "TOKEN = 'sk-proj-abcdefghij1234567890XYZ'\n"
    )
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(tmp_path / "out", "0.5.0", repo_root=repo)


def test_github_pat_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(
        repo, "src/tac/gh.py",
        "TOKEN = 'ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAA'\n"
    )
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(tmp_path / "out", "0.5.0", repo_root=repo)


def test_huggingface_token_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(
        repo, "src/tac/hf.py",
        "TOKEN = 'hf_AAAAAAAAAAAAAAAAAAAAAAAA'\n"
    )
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(tmp_path / "out", "0.5.0", repo_root=repo)


def test_vast_ssh_endpoint_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(
        repo, "src/tac/vast.py",
        "URL = 'ssh5.vast.ai:12345'\n"
    )
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(tmp_path / "out", "0.5.0", repo_root=repo)


def test_lightning_studio_link_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(
        repo, "src/tac/light.py",
        "URL = 'https://lightning.ai/operator/comma/studios/private'\n"
    )
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(tmp_path / "out", "0.5.0", repo_root=repo)


def test_explicit_secret_env_assignment_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(
        repo, "src/tac/env.py",
        "VAST_API_KEY = 'real-secret-here'\n"
    )
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(tmp_path / "out", "0.5.0", repo_root=repo)


def test_modal_call_id_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(
        repo, "src/tac/modal.py",
        "CALL = 'fc-AAAAAAAAAAAAAAAAAAAAAAAAA'\n"
    )
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(tmp_path / "out", "0.5.0", repo_root=repo)


def test_operator_readme_with_leak_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    leaky_readme = tmp_path / "leaky_readme.md"
    leaky_readme.write_text(
        "# tac\nSee /Users/operator/private/notes for details\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="README failed sanitization"):
        oss_pkt.build_packet(
            tmp_path / "out", "0.5.0",
            readme_source=leaky_readme, repo_root=repo,
        )


# ── Manifest contents ──────────────────────────────────────────────────────


def test_manifest_records_per_file_sha256(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    manifest = oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    files = manifest["files"]
    assert len(files) > 0
    for entry in files:
        assert "sha256" in entry
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] >= 0  # zero-byte __init__.py is legitimate
        assert "path" in entry


def test_manifest_paths_all_relative_never_absolute(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    manifest = oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    for entry in manifest["files"]:
        assert not entry["path"].startswith("/")
        assert "/tmp/" not in entry["path"]
        assert not Path(entry["path"]).is_absolute()


def test_manifest_total_bytes_matches_per_file_sum(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    manifest = oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    expected = sum(r["bytes"] for r in manifest["files"])
    assert manifest["total_bytes"] == expected


def test_sanitization_receipt_passed_status(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    receipt = json.loads((out / "sanitization_receipt.json").read_text())
    assert receipt["scan_status"] == "PASSED"
    assert receipt["n_leaks_found"] == 0
    assert "patterns_checked" in receipt
    assert len(receipt["patterns_checked"]) >= 5


def test_compliance_tags_in_manifest(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    manifest = oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    tags = manifest["claude_md_compliance_tags"]
    assert "public_disclosure_hygiene_gate_passed" in tags
    assert "tac_stays_clean_canonical_surface_only" in tags
    assert "no_tmp_paths_in_manifest" in tags


# ── Surface enforcement ────────────────────────────────────────────────────


def test_omx_state_paths_excluded(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    (repo / ".omx" / "state").mkdir(parents=True)
    (repo / ".omx" / "state" / "private.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    manifest = oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    paths = {f["path"] for f in manifest["files"]}
    assert not any(".omx/" in p for p in paths)


def test_pyc_files_excluded(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    (repo / "src" / "tac" / "__pycache__").mkdir()
    (repo / "src" / "tac" / "__pycache__" / "core.cpython-313.pyc").write_text(
        "binary stub", encoding="utf-8"
    )
    out = tmp_path / "out"
    manifest = oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    paths = {f["path"] for f in manifest["files"]}
    assert not any(".pyc" in p for p in paths)


def test_unincluded_extension_excluded(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    (repo / "src" / "tac" / "weights.bin").write_bytes(b"\x00" * 100)
    out = tmp_path / "out"
    manifest = oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    paths = {f["path"] for f in manifest["files"]}
    assert "src/tac/weights.bin" not in paths


# ── Idempotency / overwrite policy ─────────────────────────────────────────


def test_overwrite_false_raises_on_existing(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    with pytest.raises(FileExistsError):
        oss_pkt.build_packet(out, "1.0.1", repo_root=repo)


def test_overwrite_true_rebuilds(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    manifest2 = oss_pkt.build_packet(out, "1.0.1", repo_root=repo, overwrite=True)
    assert manifest2["version"] == "1.0.1"


def test_overwrite_true_preserves_prior_packet_on_sanitization_failure(tmp_path):
    """Per round-2 review: sanitization gate must run BEFORE destructive rmtree."""
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    oss_pkt.build_packet(out, "1.0.0", repo_root=repo)
    # First build succeeded. Now inject a leak and try to overwrite.
    _inject_leak(repo, "src/tac/oops.py", "PATH = '/Users/leak/x'\n")
    with pytest.raises(RuntimeError, match="sanitization gate REFUSED"):
        oss_pkt.build_packet(out, "1.0.1", repo_root=repo, overwrite=True)
    # Prior packet must still exist.
    assert (out / "release_manifest.json").is_file()
    prior_manifest = json.loads((out / "release_manifest.json").read_text())
    assert prior_manifest["version"] == "1.0.0"


def test_empty_version_refused(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    with pytest.raises(ValueError, match="version must be a non-empty"):
        oss_pkt.build_packet(tmp_path / "out", "  ", repo_root=repo)


def test_missing_readme_path_raises(tmp_path):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        oss_pkt.build_packet(
            out, "1.0.0",
            readme_source=tmp_path / "no_such_readme.md",
            repo_root=repo,
        )


# ── README handling ────────────────────────────────────────────────────────


def test_generated_readme_placeholder_when_none_supplied(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    oss_pkt.build_packet(out, "1.2.3", repo_root=repo)
    readme_text = (out / "README.md").read_text()
    assert "1.2.3" in readme_text
    assert "release_manifest.json" in readme_text


def test_operator_readme_used_when_supplied(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    src_readme = tmp_path / "release_readme.md"
    src_readme.write_text("# Custom OSS Release\nDetails here.\n", encoding="utf-8")
    oss_pkt.build_packet(out, "1.2.3", readme_source=src_readme, repo_root=repo)
    assert "Custom OSS Release" in (out / "README.md").read_text()


# ── CLI smoke ──────────────────────────────────────────────────────────────


def test_main_returns_2_on_existing_dir(tmp_path, capsys, monkeypatch):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    out.mkdir()  # exists already
    monkeypatch.setattr(oss_pkt, "REPO_ROOT", repo)
    rc = oss_pkt.main([
        "--out-dir", str(out),
        "--version", "1.0.0",
    ])
    assert rc == 2


def test_main_returns_3_on_leak(tmp_path, capsys, monkeypatch):
    repo = tmp_path / "repo"
    _make_clean_repo(repo)
    _inject_leak(repo, "src/tac/oops.py", "X = '/Users/leak/here'\n")
    monkeypatch.setattr(oss_pkt, "REPO_ROOT", repo)
    rc = oss_pkt.main([
        "--out-dir", str(tmp_path / "out"),
        "--version", "1.0.0",
    ])
    assert rc == 3


def test_main_returns_0_on_clean_packet(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _make_clean_repo(repo)
    monkeypatch.setattr(oss_pkt, "REPO_ROOT", repo)
    rc = oss_pkt.main([
        "--out-dir", str(out),
        "--version", "0.9.0",
    ])
    assert rc == 0
    assert (out / "release_manifest.json").is_file()


# ── Hard-exclusion helper ──────────────────────────────────────────────────


def test_is_excluded_recognizes_omx():
    assert oss_pkt._is_excluded(".omx/state/file.json")
    assert oss_pkt._is_excluded("path/to/.omx/foo")


def test_is_excluded_recognizes_pycache():
    assert oss_pkt._is_excluded("src/__pycache__/x.pyc")


def test_is_excluded_recognizes_results():
    assert oss_pkt._is_excluded("experiments/results/lane_x/build_manifest.json")


def test_is_excluded_passes_clean_path():
    assert not oss_pkt._is_excluded("src/tac/core.py")
    assert not oss_pkt._is_excluded("docs/paper/method.md")
