from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from tac.landing_diff_manifest import (
    DispositionDeclaration,
    LandingDiffManifest,
    LandingDiffManifestError,
    PathDisposition,
    build_manifest,
    load_manifest,
    load_manifest_bytes,
    main,
    parse_declarations,
    verify_manifest,
    write_manifest,
)


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Landing Test",
        "GIT_AUTHOR_EMAIL": "landing@example.invalid",
        "GIT_COMMITTER_NAME": "Landing Test",
        "GIT_COMMITTER_EMAIL": "landing@example.invalid",
    }
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    base = _commit(root, "base")
    return root, base


def _merged() -> DispositionDeclaration:
    return DispositionDeclaration(PathDisposition.MERGED)


def test_fully_merged_diff_is_complete(repo):
    root, base = repo
    (root / "added.txt").write_text("value\n", encoding="utf-8")
    head = _commit(root, "add")
    manifest = build_manifest(root, base, head, {"added.txt": _merged()})
    assert manifest.complete
    assert manifest.blockers == ()
    assert manifest.paths[0].disposition is PathDisposition.MERGED
    assert manifest.paths[0].head_sha256 == hashlib.sha256(b"value\n").hexdigest()


def test_changed_path_defaults_unaccounted(repo):
    root, base = repo
    (root / "seed.txt").write_text("changed\n", encoding="utf-8")
    head = _commit(root, "modify")
    manifest = build_manifest(root, base, head)
    assert not manifest.complete
    assert [blocker.code for blocker in manifest.blockers] == ["unaccounted_path"]


def test_dropped_with_real_reason_is_complete(repo):
    root, base = repo
    (root / "drop.txt").write_text("drop\n", encoding="utf-8")
    head = _commit(root, "drop candidate")
    declaration = DispositionDeclaration(PathDisposition.INTENTIONALLY_DROPPED, reason="superseded by task#560")
    assert build_manifest(root, base, head, {"drop.txt": declaration}).complete


@pytest.mark.parametrize("reason", [None, "", "todo", "<reason>"])
def test_dropped_missing_or_placeholder_reason_blocks(repo, reason):
    root, base = repo
    (root / "drop.txt").write_text("drop\n", encoding="utf-8")
    head = _commit(root, "drop candidate")
    declaration = DispositionDeclaration(PathDisposition.INTENTIONALLY_DROPPED, reason=reason)
    manifest = build_manifest(root, base, head, {"drop.txt": declaration})
    assert "dropped_reason_missing" in [blocker.code for blocker in manifest.blockers]


def test_deferred_with_named_consumer_is_complete(repo):
    root, base = repo
    (root / "later.txt").write_text("later\n", encoding="utf-8")
    head = _commit(root, "defer")
    declaration = DispositionDeclaration(PathDisposition.DEFERRED, named_consumer="task#561")
    assert build_manifest(root, base, head, {"later.txt": declaration}).complete


@pytest.mark.parametrize("consumer", [None, "", "tbd", "<consumer>"])
def test_deferred_missing_or_placeholder_consumer_blocks(repo, consumer):
    root, base = repo
    (root / "later.txt").write_text("later\n", encoding="utf-8")
    head = _commit(root, "defer")
    declaration = DispositionDeclaration(PathDisposition.DEFERRED, named_consumer=consumer)
    manifest = build_manifest(root, base, head, {"later.txt": declaration})
    assert "deferred_consumer_missing" in [blocker.code for blocker in manifest.blockers]


def test_findings_memo_with_per_path_consumer_is_complete(repo):
    root, base = repo
    memo = root / ".omx" / "research" / "codex_findings_arm.md"
    memo.parent.mkdir(parents=True)
    memo.write_text("finding\n", encoding="utf-8")
    head = _commit(root, "memo")
    declaration = DispositionDeclaration(PathDisposition.MERGED, named_consumer="task#555")
    manifest = build_manifest(root, base, head, {memo.relative_to(root).as_posix(): declaration})
    assert manifest.complete
    assert manifest.paths[0].findings_or_memo


def test_consumerless_findings_memo_blocks(repo):
    root, base = repo
    memo = root / ".omx" / "research" / "arm.md"
    memo.parent.mkdir(parents=True)
    memo.write_text("finding\n", encoding="utf-8")
    head = _commit(root, "memo")
    manifest = build_manifest(root, base, head, {memo.relative_to(root).as_posix(): _merged()})
    assert "findings_consumer_missing" in [blocker.code for blocker in manifest.blockers]


def test_global_consumer_is_copied_into_memo_record(repo):
    root, base = repo
    memo = root / ".omx" / "research" / "arm.md"
    memo.parent.mkdir(parents=True)
    memo.write_text("finding\n", encoding="utf-8")
    head = _commit(root, "memo")
    manifest = build_manifest(
        root, base, head, {memo.relative_to(root).as_posix(): _merged()}, global_consumer="task#555"
    )
    assert manifest.complete
    assert manifest.paths[0].named_consumer == "task#555"


def test_rename_preserves_old_path_and_both_hashes(repo):
    root, base = repo
    _git(root, "mv", "seed.txt", "renamed.txt")
    head = _commit(root, "rename")
    manifest = build_manifest(root, base, head, {"renamed.txt": _merged()})
    record = manifest.paths[0]
    assert record.git_status.startswith("R")
    assert record.old_path == "seed.txt"
    assert record.base_sha256 == record.head_sha256 == hashlib.sha256(b"seed\n").hexdigest()


def test_delete_has_base_hash_and_no_head_hash(repo):
    root, base = repo
    (root / "seed.txt").unlink()
    head = _commit(root, "delete")
    manifest = build_manifest(root, base, head, {"seed.txt": _merged()})
    record = manifest.paths[0]
    assert record.git_status == "D"
    assert record.base_sha256 == hashlib.sha256(b"seed\n").hexdigest()
    assert record.head_sha256 is None


def test_forced_add_gitignored_path_is_detected_and_blocked(repo):
    root, base = repo
    (root / ".gitignore").write_text("ignored.bin\n", encoding="utf-8")
    (root / "ignored.bin").write_bytes(b"payload")
    _git(root, "add", ".gitignore")
    _git(root, "add", "-f", "ignored.bin")
    _git(root, "commit", "-m", "forced ignored")
    head = _git(root, "rev-parse", "HEAD")
    declarations = {".gitignore": _merged(), "ignored.bin": _merged()}
    manifest = build_manifest(root, base, head, declarations)
    ignored = next(record for record in manifest.paths if record.path == "ignored.bin")
    assert ignored.gitignored_at_head
    assert "gitignored_changed_path" in [blocker.code for blocker in manifest.blockers]
    # Later worktree rules are not authority for this historical BASE..HEAD.
    (root / ".gitignore").write_text("different.bin\n", encoding="utf-8")
    rebuilt = build_manifest(root, base, head, declarations)
    assert rebuilt.to_json_bytes() == manifest.to_json_bytes()


def test_empty_diff_is_complete(repo):
    root, base = repo
    manifest = build_manifest(root, base, base)
    assert manifest.complete
    assert manifest.paths == ()
    assert manifest.blockers == ()


def test_declaration_outside_diff_is_rejected(repo):
    root, base = repo
    with pytest.raises(LandingDiffManifestError, match=r"outside BASE\.\.HEAD"):
        build_manifest(root, base, base, {"not-changed.txt": _merged()})


def test_receipt_rebuild_and_json_are_deterministic(repo, tmp_path):
    root, base = repo
    (root / "added.txt").write_text("value\n", encoding="utf-8")
    head = _commit(root, "add")
    first = build_manifest(root, base, head, {"added.txt": _merged()})
    second = build_manifest(root, base, head, {"added.txt": _merged()})
    assert first.to_json_bytes() == second.to_json_bytes()
    output = tmp_path / "receipt.json"
    write_manifest(output, first)
    assert load_manifest(output) == first
    assert verify_manifest(root, first) == ()


def test_unknown_schema_is_rejected(repo):
    root, base = repo
    raw = build_manifest(root, base, base).to_dict()
    raw["schema"] = "unknown.v9"
    with pytest.raises(LandingDiffManifestError, match="unsupported manifest schema"):
        LandingDiffManifest.from_dict(raw)


def test_duplicate_json_key_and_unknown_field_are_rejected(repo):
    root, base = repo
    manifest = build_manifest(root, base, base)
    duplicated = manifest.to_json_bytes().replace(
        b'"schema": "pact.landing_diff_manifest.v1",',
        b'"schema": "pact.landing_diff_manifest.v1",\n  "schema": "pact.landing_diff_manifest.v1",',
    )
    with pytest.raises(LandingDiffManifestError, match="duplicate JSON key"):
        load_manifest_bytes(duplicated)
    raw = manifest.to_dict()
    raw["unexpected"] = True
    with pytest.raises(LandingDiffManifestError, match="unknown fields"):
        LandingDiffManifest.from_dict(raw)


def test_tampered_content_hash_is_detected_against_git(repo):
    root, base = repo
    (root / "added.txt").write_text("value\n", encoding="utf-8")
    head = _commit(root, "add")
    raw = build_manifest(root, base, head, {"added.txt": _merged()}).to_dict()
    raw["paths"][0]["head_sha256"] = "0" * 64
    tampered = LandingDiffManifest.from_dict(raw)
    assert [blocker.code for blocker in verify_manifest(root, tampered)] == ["receipt_git_mismatch"]


def test_tampered_path_count_and_complete_claim_are_rejected(repo):
    root, base = repo
    (root / "added.txt").write_text("value\n", encoding="utf-8")
    head = _commit(root, "add")
    raw = build_manifest(root, base, head).to_dict()
    raw["path_count"] = 2
    with pytest.raises(LandingDiffManifestError, match="path_count"):
        LandingDiffManifest.from_dict(raw)
    raw = build_manifest(root, base, head).to_dict()
    raw["complete"] = True
    with pytest.raises(LandingDiffManifestError, match="complete claim"):
        LandingDiffManifest.from_dict(raw)


def test_parse_declarations_accepts_wrapper_and_rejects_escape():
    parsed = parse_declarations({"dispositions": {"a.txt": "merged"}})
    assert parsed["a.txt"].disposition is PathDisposition.MERGED
    with pytest.raises(LandingDiffManifestError, match="escapes"):
        parse_declarations({"../outside": "merged"})


def test_cli_writes_incomplete_receipt_and_returns_three(repo, tmp_path):
    root, base = repo
    (root / "added.txt").write_text("value\n", encoding="utf-8")
    head = _commit(root, "add")
    output = tmp_path / "receipt.json"
    assert main(["--repo", str(root), "--base", base, "--head", head, "--output", str(output)]) == 3
    assert output.is_file()
    assert not load_manifest(output).complete
