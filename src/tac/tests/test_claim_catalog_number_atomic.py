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
