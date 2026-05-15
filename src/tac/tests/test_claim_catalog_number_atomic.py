# SPDX-License-Identifier: MIT
"""Unit tests for FIX-2 atomic catalog-number claim (META-META 2026-05-08).

Verifies that ``tools/claim_catalog_number.py`` serializes concurrent claims
through fcntl.flock(LOCK_EX) on the state file. Each claimant gets a
monotonically-increasing unique number; concurrent processes never receive
the same number.

Bug class: 2026-05-08 dual-#114 collision (FIX-A-CUSTODY ``fa604f72`` and
FIX-A-SYNTH ``c80162e7`` both grabbed Catalog #114). Sister fork ``000089d1``
manually renumbered. The lock-on-state-file pattern makes that impossible.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_module(state_path: Path, log_path: Path):
    """Load claim_catalog_number with state/log paths redirected to tmp_path."""
    spec_path = REPO / "tools" / "claim_catalog_number.py"
    spec = importlib.util.spec_from_file_location(
        f"_claim_catalog_number_{state_path.parent.name}", spec_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Redirect global paths to tmp_path so tests don't touch real .omx state.
    module.STATE_PATH = state_path
    module.LOG_PATH = log_path
    return module


def test_claim_returns_initial_value_then_increments(tmp_path: Path) -> None:
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)

    # First claim: returns DEFAULT_INITIAL_VALUE; state file then holds N+1.
    n1 = mod.claim_one()
    assert n1 == mod.DEFAULT_INITIAL_VALUE
    assert state.read_text().strip() == str(n1 + 1)

    # Second claim: monotonic increment.
    n2 = mod.claim_one()
    assert n2 == n1 + 1
    assert state.read_text().strip() == str(n2 + 1)


def test_peek_does_not_increment(tmp_path: Path) -> None:
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)

    n_before = mod.peek()
    n_again = mod.peek()
    assert n_before == n_again
    # Now claim and verify peek shows next-after-claim.
    claimed = mod.claim_one()
    assert claimed == n_before
    assert mod.peek() == n_before + 1


def test_set_value_overrides_counter(tmp_path: Path) -> None:
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)

    mod.claim_one()  # Initialize and consume one.
    mod.set_value(500)
    assert mod.peek() == 500
    n = mod.claim_one()
    assert n == 500
    assert mod.peek() == 501


def _claim_in_subprocess(state_path: str, log_path: str, queue: mp.Queue) -> None:
    """Helper run in subprocess to claim one catalog number."""
    spec_path = str(REPO / "tools" / "claim_catalog_number.py")
    spec = importlib.util.spec_from_file_location("_subproc_claim", spec_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STATE_PATH = Path(state_path)
    module.LOG_PATH = Path(log_path)
    n = module.claim_one()
    queue.put(n)


def test_concurrent_claims_serialize_no_duplicates(tmp_path: Path) -> None:
    """Spawn N concurrent subprocesses; verify each gets a unique number."""
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    # Initialize.
    mod = _load_module(state, log)
    mod.set_value(1000)

    n_workers = 8
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_claim_in_subprocess,
            args=(str(state), str(log), queue),
        )
        for _ in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=15)
        assert p.exitcode == 0, f"subprocess died with exitcode={p.exitcode}"

    # Drain queue.
    claimed = []
    while not queue.empty():
        claimed.append(queue.get_nowait())

    # Every claim must be unique.
    assert len(claimed) == n_workers
    assert len(set(claimed)) == n_workers, f"duplicates in {claimed!r}"
    # The set must be a contiguous range starting at 1000.
    assert sorted(claimed) == list(range(1000, 1000 + n_workers))
    # Final state file should hold next-available = 1000 + n_workers.
    assert state.read_text().strip() == str(1000 + n_workers)


def test_state_file_initialized_when_absent(tmp_path: Path) -> None:
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    assert not state.exists()
    n = mod.claim_one()
    assert state.exists()
    assert n == mod.DEFAULT_INITIAL_VALUE


def test_log_records_each_claim(tmp_path: Path) -> None:
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    mod.claim_one()
    mod.claim_one()
    assert log.exists()
    lines = [line for line in log.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    # Each line must be valid JSON with the expected schema.
    import json
    for line in lines:
        rec = json.loads(line)
        assert "ts" in rec
        assert "claimed" in rec
        assert "next_will_be" in rec
        assert rec["next_will_be"] == rec["claimed"] + 1


# -----------------------------------------------------------------------------
# CANON-1.E hardening - tests for --commit-via-serializer git-transactional
# claim mode (LANDED 2026-05-12).
# -----------------------------------------------------------------------------


def test_canon_1_e_cli_rejects_commit_without_reason() -> None:
    """--commit-via-serializer requires --reason; bare invocation rc=2."""
    import subprocess
    spec_path = REPO / "tools" / "claim_catalog_number.py"
    proc = subprocess.run(
        [sys.executable, str(spec_path), "claim", "--commit-via-serializer"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--reason" in proc.stderr


def test_canon_1_e_module_exports_commit_helper(tmp_path: Path) -> None:
    """The new _commit_state_via_serializer helper is exposed at module scope."""
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    assert hasattr(mod, "_commit_state_via_serializer")
    assert callable(mod._commit_state_via_serializer)


def test_canon_1_e_serializer_path_constant_present(tmp_path: Path) -> None:
    """Module exposes SERIALIZER_PATH pointing at the canonical helper."""
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    assert hasattr(mod, "SERIALIZER_PATH")
    assert mod.SERIALIZER_PATH.name == "subagent_commit_serializer.py"
    assert mod.SERIALIZER_PATH.parent.name == "tools"


def test_canon_1_e_commit_raises_when_serializer_missing(tmp_path: Path) -> None:
    """If the serializer is absent, _commit_state_via_serializer raises."""
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    # Initialize the state file first; the helper reads it for the sha.
    mod.claim_one()
    # Point SERIALIZER_PATH at a non-existent file.
    mod.SERIALIZER_PATH = tmp_path / "nonexistent_serializer.py"
    import pytest
    with pytest.raises(RuntimeError, match="serializer not found"):
        mod._commit_state_via_serializer(claimed_n=999, reason="test")


# -----------------------------------------------------------------------------
# OP-6 atomic-write hardening tests (codex chunk 8 HIGH, 2026-05-15).
#
# Pre-OP-6, ``claim_one()`` used a truncate+rewrite pattern (seek(0); truncate();
# write(); fsync()) that left the canonical file empty for a window. A process
# crash inside that window — empirically observed during the 2026-05-09 catalog
# #183/#184 dual-claim incident — left the file empty; subsequent claimants
# silently fell back to ``DEFAULT_INITIAL_VALUE`` (116) and reissued already-
# claimed numbers.
#
# OP-6 fix: (1) atomic temp+fsync+rename via ``_atomic_write_state``; (2)
# fail-closed strict read via ``_read_state_strict`` so an empty/malformed
# state file raises ``CatalogStateCorruptError`` instead of returning a
# fallback value.
# -----------------------------------------------------------------------------


def test_op6_atomic_write_uses_temp_then_replace(tmp_path: Path, monkeypatch) -> None:
    """``_atomic_write_state`` writes via a unique tmp suffix then os.replace.

    Patches ``os.replace`` to capture the (src, dst) pair so we can assert
    the source is a tmp-suffixed sibling and the dst is the canonical
    state path.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)

    captured: dict = {}
    real_replace = os.replace

    def _captured_replace(src, dst):
        captured["src"] = str(src)
        captured["dst"] = str(dst)
        real_replace(src, dst)

    monkeypatch.setattr(mod.os, "replace", _captured_replace)
    mod._atomic_write_state(777)

    assert state.read_text().strip() == "777"
    assert captured["dst"] == str(state)
    # The source must be a tmp-suffixed sibling (not the canonical file).
    assert ".tmp." in captured["src"]
    assert captured["src"].startswith(str(state) + ".tmp.")
    # No tmp leakage on the success path.
    leftovers = list(tmp_path.glob("next_catalog_number.txt.tmp.*"))
    assert leftovers == [], f"tmp leakage: {leftovers!r}"


def test_op6_atomic_write_fsyncs_temp_before_rename(tmp_path: Path, monkeypatch) -> None:
    """``_atomic_write_state`` calls os.fsync on the temp fd BEFORE os.replace.

    Captures the call sequence so we can assert fsync precedes replace.
    Without fsync-before-rename, a power-loss between rename and the
    eventual disk flush could leave the canonical file with stale or
    zero-byte content.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)

    sequence: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def _captured_fsync(fd):
        sequence.append("fsync")
        real_fsync(fd)

    def _captured_replace(src, dst):
        sequence.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(mod.os, "fsync", _captured_fsync)
    monkeypatch.setattr(mod.os, "replace", _captured_replace)
    mod._atomic_write_state(888)

    # fsync must precede replace so the new bytes are durable on disk
    # before the rename publishes them.
    assert sequence == ["fsync", "replace"], (
        f"expected fsync-before-replace; got {sequence!r}"
    )


def test_op6_no_tmp_leakage_on_write_failure(tmp_path: Path, monkeypatch) -> None:
    """If ``os.replace`` raises, the temp file is cleaned up by the finally clause."""
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)

    def _raising_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(mod.os, "replace", _raising_replace)
    import pytest
    with pytest.raises(OSError, match="simulated rename failure"):
        mod._atomic_write_state(999)

    # No tmp file should remain on disk after the failed write.
    leftovers = list(tmp_path.glob("next_catalog_number.txt.tmp.*"))
    assert leftovers == [], f"tmp leakage on failure: {leftovers!r}"
    # Canonical file should not have been created (it never existed pre-test).
    assert not state.exists()


def test_op6_read_strict_raises_on_empty_file(tmp_path: Path) -> None:
    """``_read_state_strict`` raises CatalogStateCorruptError on empty file.

    This is the fail-closed contract: pre-OP-6, an empty state file (the
    crash-window symptom) silently returned ``DEFAULT_INITIAL_VALUE`` and
    reissued claimed numbers. Post-OP-6, the operator gets a loud error
    with recovery instructions.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    # Simulate the crash-window symptom: existing file with zero bytes.
    state.write_text("")
    import pytest
    with pytest.raises(mod.CatalogStateCorruptError, match="empty"):
        mod._read_state_strict()


def test_op6_read_strict_raises_on_whitespace_only(tmp_path: Path) -> None:
    """``_read_state_strict`` treats whitespace-only as empty (fail-closed)."""
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    state.write_text("   \n  \t  \n")
    import pytest
    with pytest.raises(mod.CatalogStateCorruptError, match="empty"):
        mod._read_state_strict()


def test_op6_read_strict_raises_on_non_integer(tmp_path: Path) -> None:
    """``_read_state_strict`` raises on non-integer content."""
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    state.write_text("not-an-integer\n")
    import pytest
    with pytest.raises(mod.CatalogStateCorruptError, match="non-integer"):
        mod._read_state_strict()


def test_op6_read_strict_returns_default_when_file_absent(tmp_path: Path) -> None:
    """When the file does not exist (bootstrap case), strict-read returns DEFAULT.

    The bootstrap-no-file case is normal (first ever claim on a fresh repo)
    and is intentionally distinct from the empty-file case (which signals
    the crash-window or external truncation). Only the latter raises.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    assert not state.exists()
    n = mod._read_state_strict()
    assert n == mod.DEFAULT_INITIAL_VALUE


def test_op6_claim_one_raises_on_empty_state_no_silent_fallback(tmp_path: Path) -> None:
    """``claim_one()`` refuses to silently fall back to DEFAULT on empty state.

    This is the headline OP-6 protection: pre-fix, the dual-claim
    incident on 2026-05-09 saw catalog #183/#184 reissued because the
    state file had been truncated mid-write. Post-fix, the same scenario
    raises ``CatalogStateCorruptError`` so the operator gets a loud
    signal instead of silent number reuse.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    # Simulate the crash-window symptom.
    state.write_text("")
    import pytest
    with pytest.raises(mod.CatalogStateCorruptError):
        mod.claim_one()


def test_op6_lock_uses_sibling_lockfile_not_state_file(tmp_path: Path) -> None:
    """``_acquire_lock`` opens a sibling ``.lock`` file, NOT the canonical state file.

    The sibling-lockfile pattern is required for ``os.replace``-based
    atomic writes to work correctly: an fd holding a lock on the OLD
    inode (about to be replaced) does not arbitrate against writers
    locking the NEW inode (the temp file post-rename). Lockers must
    contend on a stable inode that is never renamed.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)

    fh = mod._acquire_lock(timeout_seconds=5)
    try:
        # The lock fd points at the sibling .lock file, NOT at the
        # canonical state file. Verify by inspecting the open fd's name.
        lock_path = mod._lock_path()
        assert lock_path.exists()
        assert lock_path.name == "next_catalog_number.txt.lock"
        # Resolve symlinks so /private/var/folders ↔ /var/folders comparisons match.
        assert Path(fh.name).resolve() == lock_path.resolve()
    finally:
        mod._release_lock(fh)


def _atomic_claim_in_subprocess(state_path: str, log_path: str, n_claims: int, queue: mp.Queue) -> None:
    """Helper: claim N catalog numbers in subprocess and report each.

    Used by ``test_op6_atomic_concurrent_claim_4proc_5each`` to stress
    the OP-6 atomic-write + sibling-lockfile combo across 4 procs × 5
    claims each, asserting all 20 claims are unique.
    """
    spec_path = str(REPO / "tools" / "claim_catalog_number.py")
    spec = importlib.util.spec_from_file_location("_subproc_atomic_claim", spec_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STATE_PATH = Path(state_path)
    module.LOG_PATH = Path(log_path)
    claimed_locally: list[int] = []
    for _ in range(n_claims):
        n = module.claim_one()
        claimed_locally.append(n)
    queue.put(claimed_locally)


def test_op6_atomic_concurrent_claim_4proc_5each(tmp_path: Path) -> None:
    """4-proc spawn pool × 5 claims each = 20 unique contiguous claims.

    Stress-test the OP-6 atomic-write + sibling-lockfile combo across
    multiprocessing. Pre-fix, the truncate-mid-write window could (under
    a power-loss simulation) leave the file empty between two claimants;
    post-fix, the atomic ``os.replace`` is observed by every reader as
    a single transition from old→new with NO empty intermediate state.

    The contract: 20 claims arrive, all distinct, all in the contiguous
    range ``[start, start+20)``, and the final state file holds
    ``start + 20``.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    start = 5000
    mod.set_value(start)

    n_workers = 4
    n_claims_per_worker = 5
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_atomic_claim_in_subprocess,
            args=(str(state), str(log), n_claims_per_worker, queue),
        )
        for _ in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"subprocess died with exitcode={p.exitcode}"

    # Drain queue; each subprocess pushes a list of n_claims_per_worker
    # integers it claimed.
    all_claimed: list[int] = []
    while not queue.empty():
        all_claimed.extend(queue.get_nowait())

    assert len(all_claimed) == n_workers * n_claims_per_worker, (
        f"expected {n_workers * n_claims_per_worker} claims, got {len(all_claimed)}"
    )
    assert len(set(all_claimed)) == len(all_claimed), (
        f"duplicate claims detected: {sorted(all_claimed)!r}"
    )
    assert sorted(all_claimed) == list(
        range(start, start + n_workers * n_claims_per_worker)
    )
    # No tmp file leakage anywhere across the concurrent run.
    leftovers = list(tmp_path.glob("next_catalog_number.txt.tmp.*"))
    assert leftovers == [], f"tmp leakage post-stress: {leftovers!r}"


def test_op6_set_value_uses_atomic_write(tmp_path: Path, monkeypatch) -> None:
    """``set_value`` also routes through the atomic temp+fsync+rename path.

    The pre-fix ``set_value`` had the same truncate+rewrite vulnerability
    as ``claim_one``; the OP-6 fix routes both through ``_atomic_write_state``.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)

    # Initialize with a real claim so the file exists.
    mod.claim_one()

    captured: dict = {}
    real_replace = os.replace

    def _captured_replace(src, dst):
        captured["src"] = str(src)
        captured["dst"] = str(dst)
        real_replace(src, dst)

    monkeypatch.setattr(mod.os, "replace", _captured_replace)
    mod.set_value(424242)

    assert state.read_text().strip() == "424242"
    assert ".tmp." in captured["src"]
    assert captured["dst"] == str(state)


def test_op6_canonical_cli_signature_preserved() -> None:
    """The CLI surface is unchanged post-OP-6; sister subagents are not broken.

    Per the OP-6 prompt's CRITICAL note: ``claim_catalog_number.py`` is
    used by every sister subagent claiming catalog #s. The OP-6 fix
    must NOT change the CLI signature. This test pins the canonical
    subcommand surface (``claim`` / ``peek`` / ``set``) and the
    ``--commit-via-serializer`` / ``--reason`` flags so a future
    refactor that drops a flag fails the test loudly.
    """
    import subprocess
    spec_path = REPO / "tools" / "claim_catalog_number.py"
    proc = subprocess.run(
        [sys.executable, str(spec_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    out = proc.stdout
    assert "claim" in out
    assert "peek" in out
    assert "set" in out
    # Verify --commit-via-serializer + --reason still in the claim subparser.
    proc = subprocess.run(
        [sys.executable, str(spec_path), "claim", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--commit-via-serializer" in proc.stdout
    assert "--reason" in proc.stdout


def test_canon_1_e_log_marks_committed_when_serializer_succeeds(tmp_path: Path) -> None:
    """When the serializer commit succeeds, the log records committed_via_serializer=True.

    Uses a fake serializer script that exits 0 to verify the success path
    appends the expected log entry without depending on a real git commit.
    """
    state = tmp_path / "next_catalog_number.txt"
    log = tmp_path / "catalog-claim.log"
    mod = _load_module(state, log)
    # Initialize the state file with one real claim to populate it.
    mod.claim_one()
    # Point SERIALIZER_PATH at a fake script that always exits 0.
    fake = tmp_path / "fake_serializer.py"
    fake.write_text("import sys; sys.exit(0)\n")
    mod.SERIALIZER_PATH = fake
    # Also redirect REPO_ROOT so subprocess cwd doesn't escape tmp.
    mod.REPO_ROOT = tmp_path
    # Ensure relative-path computation works: state must be under REPO_ROOT.
    state_in_repo = tmp_path / ".omx" / "state" / "next_catalog_number.txt"
    state_in_repo.parent.mkdir(parents=True, exist_ok=True)
    state_in_repo.write_text("999\n")
    mod.STATE_PATH = state_in_repo
    mod._commit_state_via_serializer(claimed_n=998, reason="test reason")
    import json
    entries = [
        json.loads(line)
        for line in log.read_text().splitlines()
        if line.strip()
    ]
    last = entries[-1]
    assert last.get("committed_via_serializer") is True
    assert last.get("reason") == "test reason"
    assert last.get("claimed") == 998
