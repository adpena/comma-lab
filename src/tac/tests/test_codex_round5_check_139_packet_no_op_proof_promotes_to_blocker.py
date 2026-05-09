"""Tests for codex round-5 catalog #139 — packet-compiler no_op_proof must promote to blocker.

Bug class (codex round-5 HIGH 1, 2026-05-09): the previous
``tac.phase1_packet_compiler._finalize_packet_result`` computed
``no_op_proof.runtime_consumes_bytes`` and ``no_op_proof.no_op_detector_passed``
but never converted ``False`` results into entries in ``blockers``. The CLI
exit gate read only ``result.blockers`` so a no-op inflate.py exited 0 and
burned eval spend.

The fix:

1. ``_finalize_packet_result`` now appends
   ``inflate_does_not_consume_archive_bytes:...`` and
   ``no_op_detector_failed:...`` to ``blockers`` when the corresponding
   no_op_proof field is ``False``.
2. A new ``_verify_runtime_consumes_payload_bytes_executable`` helper
   actually mutates one byte of the archive and observes whether downstream
   inflate output changes; if not, the static-detector verdict is
   downgraded.
3. STRICT preflight gate #139 refuses any future packet-compiler-style
   finalize that calls ``_build_no_op_proof(...)`` and mutates ``blockers``
   without appending the canonical promotion tags.

Memory: feedback_codex_round5_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_packet_compiler_no_op_proof_promotes_to_blocker,
)


def _make_repo(tmp_path: Path) -> Path:
    """Lay down the minimum directory shape the gate scans."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    return tmp_path


# ── Live-repo sanity ────────────────────────────────────────────────────


def test_139_live_repo_clean():
    """Live-repo: catalog #139 must land at 0 violations after fix.

    The HIGH 1 fix in ``tac.phase1_packet_compiler._finalize_packet_result``
    promotes ``runtime_consumes_bytes=False`` /
    ``no_op_detector_passed=False`` to first-class blockers; any future
    regression (re-introducing the advisory-only failure mode) will fail
    this test loud-and-early.
    """
    v = check_packet_compiler_no_op_proof_promotes_to_blocker(
        strict=False, verbose=False
    )
    assert v == [], (
        f"Catalog #139 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )


# ── Catch the canonical no-op-proof-without-promotion bug ──────────────


def test_139_catches_finalize_without_blocker_promotion(tmp_path):
    """A function that calls _build_no_op_proof + mutates blockers without
    appending the canonical promotion tags MUST be flagged."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_packet_compiler.py"
    target.write_text(
        "def _finalize_packet(out_dir, archive_sha):\n"
        "    blockers = []\n"
        "    runtime_consumes = True\n"
        "    no_op_proof = _build_no_op_proof(\n"
        "        new_archive_sha256=archive_sha,\n"
        "        runtime_consumes_bytes=runtime_consumes,\n"
        "    )\n"
        "    blockers.append('something_unrelated')\n"
        "    return blockers, no_op_proof\n"
    )
    v = check_packet_compiler_no_op_proof_promotes_to_blocker(
        repo_root=root, strict=False, verbose=False
    )
    assert any("Check 139" in x and "_finalize_packet" in x for x in v), (
        f"expected Check 139 hit on _finalize_packet; got: {v}"
    )


def test_139_accepts_finalize_with_canonical_promotion(tmp_path):
    """A function that promotes the no-op proof failure correctly is OK."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_packet_compiler.py"
    target.write_text(
        "def _finalize_packet(out_dir, archive_sha):\n"
        "    blockers = []\n"
        "    runtime_consumes = False\n"
        "    no_op_proof = _build_no_op_proof(\n"
        "        new_archive_sha256=archive_sha,\n"
        "        runtime_consumes_bytes=runtime_consumes,\n"
        "    )\n"
        "    if not runtime_consumes:\n"
        "        blockers.append('inflate_does_not_consume_archive_bytes')\n"
        "    return blockers, no_op_proof\n"
    )
    v = check_packet_compiler_no_op_proof_promotes_to_blocker(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"expected 0 violations; got: {v}"


def test_139_accepts_no_op_detector_failed_promotion(tmp_path):
    """The sister blocker tag `no_op_detector_failed` is also accepted."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_packet_compiler.py"
    target.write_text(
        "def _finalize_packet(out_dir, archive_sha):\n"
        "    blockers = []\n"
        "    no_op_proof = _build_no_op_proof(\n"
        "        new_archive_sha256=archive_sha,\n"
        "        runtime_consumes_bytes=True,\n"
        "    )\n"
        "    if no_op_proof.get('no_op_detector_passed') is False:\n"
        "        blockers.append('no_op_detector_failed:...')\n"
        "    return blockers, no_op_proof\n"
    )
    v = check_packet_compiler_no_op_proof_promotes_to_blocker(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"expected 0 violations; got: {v}"


def test_139_respects_same_line_waiver(tmp_path):
    """Same-line `# NO_OP_PROOF_ADVISORY_OK:...` waives the gate."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_packet_compiler.py"
    target.write_text(
        "def _probe_only_finalize(out_dir, archive_sha):  # NO_OP_PROOF_ADVISORY_OK:probe-only-no-runnable-inflate\n"
        "    blockers = []\n"
        "    no_op_proof = _build_no_op_proof(\n"
        "        new_archive_sha256=archive_sha,\n"
        "        runtime_consumes_bytes=False,\n"
        "    )\n"
        "    blockers.append('something')\n"
        "    return blockers, no_op_proof\n"
    )
    v = check_packet_compiler_no_op_proof_promotes_to_blocker(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"expected 0 violations after waiver; got: {v}"


def test_139_strict_mode_raises(tmp_path):
    """strict=True raises PreflightError when violations are present."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_packet_compiler.py"
    target.write_text(
        "def _finalize_packet(out_dir, archive_sha):\n"
        "    blockers = []\n"
        "    no_op_proof = _build_no_op_proof(\n"
        "        new_archive_sha256=archive_sha,\n"
        "        runtime_consumes_bytes=False,\n"
        "    )\n"
        "    blockers.append('unrelated')\n"
        "    return blockers, no_op_proof\n"
    )
    with pytest.raises(PreflightError, match="Check 139|no-op-proof"):
        check_packet_compiler_no_op_proof_promotes_to_blocker(
            repo_root=root, strict=True, verbose=False
        )


def test_139_alternate_tag_scheme_modules_out_of_scope(tmp_path):
    """Modules using their OWN tag scheme (not _build_no_op_proof) are
    out-of-scope. e.g. monolithic_packet_closure_gate's
    `_runtime_consumption_summary` uses `runtime_proof_*` tags.
    """
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "alternate_runtime_proof.py"
    target.write_text(
        "def _runtime_consumption_summary(proof, candidate_manifest, changed_sections):\n"
        "    blockers = []\n"
        "    if not isinstance(proof, dict):\n"
        "        blockers.append('runtime_consumption_proof_missing')\n"
        "    if proof.get('runtime_consumes_bytes') is False:\n"
        "        blockers.append('runtime_proof_runtime_consumes_bytes_false')\n"
        "    return {'summary': True}, blockers\n"
    )
    v = check_packet_compiler_no_op_proof_promotes_to_blocker(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], (
        f"alternate-tag-scheme modules should be out of scope; got: {v}"
    )


def test_139_test_files_excluded(tmp_path):
    """Test files (test_*.py / under tests/) must not be scanned."""
    root = _make_repo(tmp_path)
    tests_dir = root / "src" / "tac" / "tests"
    tests_dir.mkdir(parents=True)
    target = tests_dir / "test_fake.py"
    target.write_text(
        "def test_finalize_packet():\n"
        "    blockers = []\n"
        "    no_op_proof = _build_no_op_proof(runtime_consumes_bytes=False)\n"
        "    blockers.append('unrelated')\n"
    )
    v = check_packet_compiler_no_op_proof_promotes_to_blocker(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"test files should be excluded; got: {v}"


def test_139_public_pr_intake_excluded(tmp_path):
    """Public-PR intake clones must be out-of-scope."""
    root = _make_repo(tmp_path)
    intake = (
        root
        / "experiments"
        / "results"
        / "public_pr107_intake_20260508_codex"
        / "source"
    )
    intake.mkdir(parents=True)
    target = intake / "fake_pkc.py"
    target.write_text(
        "def _finalize_packet(out_dir, archive_sha):\n"
        "    blockers = []\n"
        "    no_op_proof = _build_no_op_proof(runtime_consumes_bytes=False)\n"
        "    blockers.append('unrelated')\n"
    )
    v = check_packet_compiler_no_op_proof_promotes_to_blocker(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"public PR intake should be excluded; got: {v}"


# ── Behavioural test of the actual fix in phase1_packet_compiler ────────


def test_finalize_promotes_runtime_not_consumed_to_blocker(tmp_path):
    """End-to-end: a packet whose inflate.py does NOT read the archive
    must produce a blocker entry, not a clean compile."""
    from tac.phase1_packet_compiler import _finalize_packet_result

    import zipfile

    out_dir = tmp_path / "phase1_out"
    out_dir.mkdir()
    # Create a real ZIP archive (not just bytes) so _read_archive_members
    # doesn't raise BadZipFile during finalize.
    archive = out_dir / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("payload.bin", b"x" * 1024)
    (out_dir / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\necho noop\n"
    )
    # Inflate.py with NO byte-read pattern → should fail the static check
    # AND the executable smoke. A no-op script that ignores its args.
    (out_dir / "inflate.py").write_text(
        "import sys\n"
        "def main():\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    src_dir = out_dir / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("")

    result = _finalize_packet_result(
        output_dir=out_dir,
        mode="identity",
        target_mode="contest_one_video_replay",
        runtime_dep_closure=["torch"],
        export_format="test_format",
        bolt_on_loc_budget=400,
        baseline_archive_sha256=None,
        baseline_archive_size_bytes=None,
        score_affecting_payload_changed=False,
        declared_files=["inflate.sh", "inflate.py", "src/__init__.py"],
    )
    has_no_op_blocker = any(
        b.startswith("inflate_does_not_consume_archive_bytes")
        for b in result.blockers
    )
    assert has_no_op_blocker, (
        "expected `inflate_does_not_consume_archive_bytes` in blockers; "
        f"got: {list(result.blockers)}"
    )


def test_executable_smoke_function_returns_pair(tmp_path):
    """The new ``_verify_runtime_consumes_payload_bytes_executable`` returns
    (bool, str_reason) and writes its scratch under packet_dir, NOT /tmp."""
    from tac.phase1_packet_compiler import (
        _verify_runtime_consumes_payload_bytes_executable,
    )

    out_dir = tmp_path / "phase1_out"
    out_dir.mkdir()
    (out_dir / "inflate.py").write_text(
        "import sys\nsys.exit(0)\n"
    )
    archive = out_dir / "archive.zip"
    archive.write_bytes(b"x" * 256)

    passed, reason = _verify_runtime_consumes_payload_bytes_executable(
        packet_dir=out_dir,
        archive_path=archive,
    )
    assert isinstance(passed, bool)
    assert isinstance(reason, str)
    # Must NOT leave /tmp paths behind — scratch is _no_op_smoke under packet_dir
    assert not (out_dir / "_no_op_smoke").exists(), (
        "smoke scratch should be cleaned up after smoke run"
    )


def test_executable_smoke_handles_missing_archive(tmp_path):
    """Missing archive returns (False, 'archive_missing') without raising."""
    from tac.phase1_packet_compiler import (
        _verify_runtime_consumes_payload_bytes_executable,
    )

    out_dir = tmp_path / "phase1_out"
    out_dir.mkdir()
    (out_dir / "inflate.py").write_text("import sys\nsys.exit(0)\n")

    passed, reason = _verify_runtime_consumes_payload_bytes_executable(
        packet_dir=out_dir,
        archive_path=out_dir / "nonexistent.zip",
    )
    assert passed is False
    assert reason == "archive_missing"


def test_executable_smoke_no_tmp_paths(tmp_path):
    """Per CLAUDE.md FORBIDDEN /tmp paths: smoke scratch MUST NOT live in /tmp."""
    import inspect

    from tac.phase1_packet_compiler import (
        _verify_runtime_consumes_payload_bytes_executable,
    )

    src = inspect.getsource(_verify_runtime_consumes_payload_bytes_executable)
    # No `/tmp/` literal in the helper body
    assert "/tmp/" not in src, "smoke helper must not write to /tmp"
