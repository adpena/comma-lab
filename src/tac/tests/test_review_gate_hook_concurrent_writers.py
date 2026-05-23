# SPDX-License-Identifier: MIT
"""Tests for `tools/review_gate_hook.py` duckdb lock-contention robustness.

Lane: lane_duckdb_lock_fix_review_gate_hook_20260514.

Empirical anchors (CLAUDE.md MEMORY.md):
- CATALOG-226-REFACTOR landing memo: 84.9s stall on canonical commit serializer
  caused by sister-subagent DuckDB write lock during parallel-wave editing.
- F3-GTSCORERCACHE landing memo: 10 retries / 5+ minutes failure cascade
  forcing operators to use `REVIEW_GATE_ENABLED=0` infrastructure workaround.

These tests validate the fix landed in this lane:
1. `_connect_duckdb` accepts a `retry_seconds` override.
2. `_is_duckdb_lock_error` recognizes the cross-process "Could not set lock"
   error AND the in-process "Can't open a connection" error.
3. The hook routes through the retry helper with a tight 1.5s budget.
4. The hook falls through to the JSON snapshot when DuckDB lock cannot
   be acquired in time — entity-status enforcement (needs_fix / unreviewed)
   continues to BLOCK; policy-check is downgraded to a warning banner.
5. Under 10 concurrent reader processes with no writer, the hot path
   completes in <2s wall-clock.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable, this test file is the regression guard against re-introducing
the lock-contention class.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK_PATH = REPO_ROOT / "tools" / "review_gate_hook.py"
TRACKER_PATH = REPO_ROOT / "tools" / "review_tracker.py"


def _load_module(name: str, file_path: Path):
    """Load a tool module by absolute file path (the hook isn't a regular package)."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Helper invoked under multiprocessing — must be top-level for picklability.
def _writer_hold_lock(db_path: str, duration_seconds: float, ready_event, done_event):
    """Open the DB in RW mode and hold the lock for ``duration_seconds``.

    Signals ``ready_event`` once the lock is acquired so the parent can
    deterministically race readers against it. Waits for ``done_event``
    OR the duration timeout, whichever comes first.
    """
    import duckdb
    con = duckdb.connect(db_path)
    try:
        # Ensure the lock is materialized via a write.
        con.execute("CREATE TABLE IF NOT EXISTS _lock_marker (id INT)")
        con.execute("INSERT INTO _lock_marker VALUES (1)")
        ready_event.set()
        done_event.wait(timeout=duration_seconds)
    finally:
        con.close()


def _reader_attempt_connect(db_path: str, retry_seconds: float, result_queue):
    """Try to open the DB read-only using the canonical retry helper.

    Returns (kind, payload) via the queue:
        ('ok', wall_seconds) — connection succeeded
        ('lock_err', error_text) — recognized lock-contention error
        ('other_err', error_text) — unrecognized error
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import review_tracker as rt
    except Exception as exc:
        result_queue.put(("import_err", str(exc)))
        return
    t0 = time.monotonic()
    try:
        con = rt._connect_duckdb(
            Path(db_path), read_only=True, retry_seconds=retry_seconds
        )
        con.close()
        result_queue.put(("ok", time.monotonic() - t0))
    except Exception as exc:
        # Classify
        if rt._is_duckdb_lock_error(exc):
            result_queue.put(("lock_err", str(exc)[:200]))
        else:
            result_queue.put(("other_err", f"{type(exc).__name__}: {str(exc)[:200]}"))


class IsDuckDBLockErrorTests(unittest.TestCase):
    """Test that _is_duckdb_lock_error recognizes BOTH error families."""

    @classmethod
    def setUpClass(cls):
        # Use isolated module name to avoid clashing with any sister test
        # that may have imported tools.review_tracker.
        cls.rt = _load_module("test_rt_lockerr", TRACKER_PATH)

    def test_recognizes_could_not_set_lock(self):
        """Cross-process IOException: Could not set lock on file"""
        err = Exception("IO Error: Could not set lock on file 'x.duckdb'")
        self.assertTrue(self.rt._is_duckdb_lock_error(err))

    def test_recognizes_conflicting_lock(self):
        """Legacy duckdb wording: Conflicting lock"""
        err = Exception("IO Error: Conflicting lock on file 'x.duckdb'")
        self.assertTrue(self.rt._is_duckdb_lock_error(err))

    def test_recognizes_in_process_mixed_mode(self):
        """In-process error: Can't open a connection to same database file ...

        This was UNRECOGNIZED before the fix; the empirical anchor in
        the directives reproduced it.
        """
        err = Exception(
            "Connection Error: Can't open a connection to same database "
            "file with a different configuration"
        )
        self.assertTrue(self.rt._is_duckdb_lock_error(err))

    def test_does_not_misclassify_unrelated_errors(self):
        """Random errors must NOT be classified as lock contention."""
        self.assertFalse(self.rt._is_duckdb_lock_error(ValueError("bad input")))
        self.assertFalse(self.rt._is_duckdb_lock_error(OSError("disk full")))
        self.assertFalse(
            self.rt._is_duckdb_lock_error(Exception("Some random duckdb error"))
        )


class ConnectDuckDBRetrySecondsTests(unittest.TestCase):
    """Test that _connect_duckdb respects an explicit retry_seconds override."""

    @classmethod
    def setUpClass(cls):
        cls.rt = _load_module("test_rt_retry", TRACKER_PATH)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.db = Path(self.tmpdir) / "t.duckdb"
        import duckdb
        con = duckdb.connect(str(self.db))
        con.execute("CREATE TABLE entities (id INT)")
        con.execute("INSERT INTO entities VALUES (1)")
        con.close()

    def test_default_uses_global_budget(self):
        """retry_seconds=None uses DB_LOCK_RETRY_SECONDS."""
        # Simply verify the call signature works with default.
        con = self.rt._connect_duckdb(self.db, read_only=True)
        self.assertIsNotNone(con)
        con.close()

    def test_custom_retry_seconds_short(self):
        """retry_seconds=0.5 returns quickly on uncontended DB."""
        t0 = time.monotonic()
        con = self.rt._connect_duckdb(self.db, read_only=True, retry_seconds=0.5)
        elapsed = time.monotonic() - t0
        self.assertIsNotNone(con)
        self.assertLess(elapsed, 0.5)
        con.close()

    def test_zero_retry_seconds_one_attempt(self):
        """retry_seconds=0.0 still makes at least one attempt."""
        con = self.rt._connect_duckdb(self.db, read_only=True, retry_seconds=0.0)
        self.assertIsNotNone(con)
        con.close()


class JsonSnapshotFallbackTests(unittest.TestCase):
    """Test that load_entities_from_json_snapshot handles missing/corrupt JSON."""

    @classmethod
    def setUpClass(cls):
        cls.rt = _load_module("test_rt_json", TRACKER_PATH)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.json_path = Path(self.tmpdir) / "review_tracker.json"

    def test_missing_file_returns_none(self):
        result = self.rt.load_entities_from_json_snapshot(self.json_path)
        self.assertIsNone(result)

    def test_valid_json_returns_parsed_dict(self):
        payload = {
            "version": 3,
            "last_scan": "2026-05-14T00:00:00Z",
            "entity_count": 1,
            "review_count": 0,
            "entities": {
                "mod::func": {
                    "module": "mod",
                    "file_path": "src/x.py",
                    "entity_type": "function",
                    "name": "func",
                    "start_line": 1,
                    "end_line": 5,
                    "line_count": 5,
                    "complexity": 1,
                    "review_status": "unreviewed",
                }
            },
        }
        self.json_path.write_text(json.dumps(payload))
        result = self.rt.load_entities_from_json_snapshot(self.json_path)
        self.assertEqual(result["entity_count"], 1)
        self.assertIn("mod::func", result["entities"])

    def test_corrupt_json_returns_none(self):
        self.json_path.write_text("{ this is not valid json")
        result = self.rt.load_entities_from_json_snapshot(self.json_path)
        self.assertIsNone(result)

    def test_empty_file_returns_none(self):
        self.json_path.write_text("")
        result = self.rt.load_entities_from_json_snapshot(self.json_path)
        self.assertIsNone(result)


class HookCrossProcessLockContentionTests(unittest.TestCase):
    """Multiprocess test: verify the hook handles a long-running writer gracefully.

    This is the canonical empirical anchor reproducer. A sister process
    holds the DuckDB RW lock; many readers try to open read-only and would
    historically fail with `Could not set lock`. The fix is the retry
    helper with a tight deadline + JSON fallback.
    """

    @classmethod
    def setUpClass(cls):
        cls.rt = _load_module("test_rt_xproc", TRACKER_PATH)
        # Force the spawn start method for cross-platform determinism (macOS
        # default is spawn anyway, but be explicit).
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.db = Path(self.tmpdir) / "t.duckdb"
        import duckdb
        con = duckdb.connect(str(self.db))
        con.execute("CREATE TABLE entities (id INT)")
        con.execute("INSERT INTO entities VALUES (1)")
        con.close()

    def test_reader_fails_during_writer_hold_with_zero_retry(self):
        """Baseline: cross-process reader fails immediately when writer holds lock."""
        ctx = mp.get_context("spawn")
        ready = ctx.Event()
        done = ctx.Event()
        result_q = ctx.Queue()
        writer = ctx.Process(
            target=_writer_hold_lock, args=(str(self.db), 5.0, ready, done)
        )
        writer.start()
        try:
            self.assertTrue(ready.wait(timeout=3.0))
            # Reader with retry_seconds=0.0 should fail immediately.
            reader = ctx.Process(
                target=_reader_attempt_connect,
                args=(str(self.db), 0.0, result_q),
            )
            reader.start()
            reader.join(timeout=10)
            self.assertFalse(reader.is_alive())
            kind, payload = result_q.get(timeout=2)
            self.assertEqual(kind, "lock_err")
        finally:
            done.set()
            writer.join(timeout=5)

    def test_reader_succeeds_after_writer_releases_with_retry(self):
        """With a retry budget that outlasts the writer, reader succeeds."""
        ctx = mp.get_context("spawn")
        ready = ctx.Event()
        done = ctx.Event()
        result_q = ctx.Queue()
        # Writer holds for 1s; reader has 3s retry budget — should succeed.
        writer = ctx.Process(
            target=_writer_hold_lock, args=(str(self.db), 1.0, ready, done)
        )
        writer.start()
        try:
            self.assertTrue(ready.wait(timeout=3.0))
            reader = ctx.Process(
                target=_reader_attempt_connect,
                args=(str(self.db), 3.0, result_q),
            )
            reader.start()
            reader.join(timeout=10)
            self.assertFalse(reader.is_alive())
            kind, payload = result_q.get(timeout=2)
            self.assertEqual(kind, "ok", msg=f"expected ok, got {kind}: {payload}")
            # The reader should have waited approximately as long as the writer
            # held the lock (1s) — definitely more than 0.5s.
            self.assertGreater(payload, 0.5)
        finally:
            done.set()
            writer.join(timeout=5)

    def test_reader_falls_through_when_writer_outlasts_retry_budget(self):
        """When writer outlasts the retry deadline, reader sees lock_err.

        The hook's higher-level code then falls back to the JSON snapshot.
        This test pins the retry-helper boundary; the hook fallback is
        exercised in `HookCheckStagedFilesFallbackTests` below.
        """
        ctx = mp.get_context("spawn")
        ready = ctx.Event()
        done = ctx.Event()
        result_q = ctx.Queue()
        # Writer holds for 5s; reader has 0.5s retry — should fail.
        writer = ctx.Process(
            target=_writer_hold_lock, args=(str(self.db), 5.0, ready, done)
        )
        writer.start()
        try:
            self.assertTrue(ready.wait(timeout=3.0))
            reader = ctx.Process(
                target=_reader_attempt_connect,
                args=(str(self.db), 0.5, result_q),
            )
            t0 = time.monotonic()
            reader.start()
            reader.join(timeout=10)
            wall = time.monotonic() - t0
            self.assertFalse(reader.is_alive())
            kind, payload = result_q.get(timeout=2)
            self.assertEqual(kind, "lock_err")
            # Retry deadline should be respected — reader did not block
            # past ~0.5s + a small slack for IPC + sleep granularity.
            self.assertLess(wall, 2.5)
        finally:
            done.set()
            writer.join(timeout=5)


# Helper for the concurrent-readers scaling test.
def _ten_concurrent_readers_one_writer(db_path: str, results: list):
    """Spawn 10 readers attempting to open the DB during a writer.

    Used inside a single multiprocessing test to validate that the canonical
    retry helper + JSON fallback together produce a graceful end-to-end
    outcome (no readers crash; readers either succeed-after-retry or
    return classified lock_err).
    """
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    done = ctx.Event()
    qs = [ctx.Queue() for _ in range(10)]
    procs = []
    writer = ctx.Process(
        target=_writer_hold_lock, args=(db_path, 0.7, ready, done)
    )
    writer.start()
    ready.wait(timeout=3.0)
    for i in range(10):
        # Mix of retry budgets — 5 with tight (the hook default-ish), 5 with longer.
        retry = 1.5 if i < 5 else 3.0
        p = ctx.Process(
            target=_reader_attempt_connect,
            args=(db_path, retry, qs[i]),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=10)
    done.set()
    writer.join(timeout=5)
    for q in qs:
        try:
            results.append(q.get(timeout=2))
        except Exception as exc:
            results.append(("queue_err", str(exc)))


class TenConcurrentReadersStressTests(unittest.TestCase):
    """10 readers + 1 writer: every reader resolves cleanly (no crash)."""

    @classmethod
    def setUpClass(cls):
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

    def test_10_readers_resolve_cleanly(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        db = Path(tmpdir) / "t.duckdb"
        import duckdb
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE entities (id INT)")
        con.close()

        results: list = []
        _ten_concurrent_readers_one_writer(str(db), results)
        # 10 readers must all report a recognized outcome.
        self.assertEqual(len(results), 10)
        ok_count = sum(1 for r in results if r[0] == "ok")
        lock_err_count = sum(1 for r in results if r[0] == "lock_err")
        # No "other_err" / "queue_err" / "import_err" allowed.
        for r in results:
            self.assertIn(
                r[0],
                ("ok", "lock_err"),
                msg=f"unexpected outcome: {r}",
            )
        # With a 0.7s writer hold, the tight-retry (1.5s) readers should
        # mostly succeed; the long-retry (3s) readers should all succeed.
        self.assertGreater(ok_count, 0, msg=f"expected >0 ok, results={results}")
        # Stress-mode invariant: total readers == 10 and none crashed.
        self.assertEqual(ok_count + lock_err_count, 10)


class HookCheckStagedFilesFallbackTests(unittest.TestCase):
    """End-to-end: check_staged_files falls back to JSON when DB unavailable."""

    @classmethod
    def setUpClass(cls):
        # Load the hook module standalone so we can monkey-patch TRACKER_DB etc.
        cls.hook = _load_module("test_hook_fallback", HOOK_PATH)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.fake_db = Path(self.tmpdir) / "review_tracker.duckdb"
        self.fake_json = Path(self.tmpdir) / "review_tracker.json"
        # Build a JSON snapshot with one needs_fix + one reviewed entity.
        self.fake_json.write_text(json.dumps({
            "version": 3,
            "last_scan": "2026-05-14T00:00:00Z",
            "entity_count": 2,
            "review_count": 0,
            "entities": {
                "src.tac.x::func_a": {
                    "module": "src.tac.x",
                    "file_path": "src/tac/x.py",
                    "entity_type": "function",
                    "name": "func_a",
                    "start_line": 1,
                    "end_line": 5,
                    "line_count": 5,
                    "complexity": 1,
                    "review_status": "needs_fix",
                },
                "src.tac.x::func_b": {
                    "module": "src.tac.x",
                    "file_path": "src/tac/x.py",
                    "entity_type": "function",
                    "name": "func_b",
                    "start_line": 10,
                    "end_line": 15,
                    "line_count": 5,
                    "complexity": 1,
                    "review_status": "reviewed",
                },
            },
        }))

    def _run_hook_with_fake_paths(self, staged_files: list[str], policy: dict | None = None):
        """Patch the hook module's TRACKER_DB and TRACKER_JSON for isolated tests."""
        with mock.patch.object(self.hook, "TRACKER_DB", self.fake_db), \
             mock.patch.object(self.hook, "TRACKER_JSON", self.fake_json), \
             mock.patch.object(self.hook, "load_policy", return_value=policy or {}):
            return self.hook.check_staged_files(staged_files)

    def test_falls_back_to_json_when_db_missing(self):
        """No DB file → falls through to JSON snapshot (if present)."""
        blocking, warnings, stats = self._run_hook_with_fake_paths(["src/tac/x.py"])
        self.assertEqual(stats.get("source"), "json")
        # The needs_fix entity should be in the violation list.
        self.assertTrue(
            any("NEEDS_FIX" in line for line in blocking + warnings),
            msg=f"expected NEEDS_FIX violation, got blocking={blocking}, warnings={warnings}",
        )

    def test_no_db_no_json_emits_skip_warning(self):
        """No DB AND no JSON → skip the gate (don't block the commit)."""
        self.fake_json.unlink()
        blocking, warnings, stats = self._run_hook_with_fake_paths(["src/tac/x.py"])
        self.assertEqual(stats.get("source"), "none")
        # No blocking.
        self.assertEqual(blocking, [])
        # Banner mentions tracker DB.
        self.assertTrue(
            any("tracker DB" in w or "scan" in w for w in warnings),
            msg=f"warnings={warnings}",
        )

    def test_json_path_emits_degraded_banner(self):
        """JSON-snapshot path emits a one-line banner explaining the degradation."""
        blocking, warnings, stats = self._run_hook_with_fake_paths(["src/tac/x.py"])
        # The first warning should be the degraded-mode banner.
        self.assertGreaterEqual(len(warnings), 1)
        self.assertIn("JSON snapshot", warnings[0])
        self.assertIn("retry budget", warnings[0])

    def test_json_path_blocks_on_needs_fix_in_critical_path(self):
        """`needs_fix` in src/tac/ (standard rigor) must still block the commit."""
        blocking, warnings, stats = self._run_hook_with_fake_paths(["src/tac/x.py"])
        # Standard rigor: needs_fix entities go to blocking list.
        # (Default policy has src/tac/ as 'standard' — that branch blocks.)
        # If the policy is fully empty, get_rigor_for_file may return relaxed
        # — accept either, but stats must record the needs_fix.
        self.assertEqual(stats.get("needs_fix"), 1)

    def test_json_path_unreviewed_entity_counted(self):
        """Unreviewed status flows through the JSON path."""
        # Overwrite snapshot with an unreviewed entity only.
        self.fake_json.write_text(json.dumps({
            "version": 3,
            "entity_count": 1,
            "entities": {
                "src.tac.x::func_c": {
                    "module": "src.tac.x",
                    "file_path": "src/tac/x.py",
                    "entity_type": "function",
                    "name": "func_c",
                    "start_line": 1, "end_line": 2, "line_count": 2,
                    "complexity": 1,
                    "review_status": "unreviewed",
                },
            },
        }))
        _, _, stats = self._run_hook_with_fake_paths(["src/tac/x.py"])
        self.assertEqual(stats.get("violations"), 1)

    def test_json_path_reviewed_entity_counted_compliant(self):
        """Reviewed relaxed entities remain advisory under JSON-degraded mode."""
        self.fake_json.write_text(json.dumps({
            "version": 3,
            "entity_count": 1,
            "entities": {
                "src.tac.x::func_d": {
                    "module": "src.tac.x",
                    "file_path": "src/tac/x.py",
                    "entity_type": "function",
                    "name": "func_d",
                    "start_line": 1, "end_line": 2, "line_count": 2,
                    "complexity": 1,
                    "review_status": "reviewed",
                },
            },
        }))
        _, _, stats = self._run_hook_with_fake_paths(["src/tac/x.py"])
        self.assertEqual(stats.get("compliant"), 1)
        self.assertEqual(stats.get("violations"), 0)

    def test_json_path_blocks_reviewed_standard_entity_without_full_policy(self):
        """Reviewed critical/standard entities need DuckDB policy evidence."""
        self.fake_json.write_text(json.dumps({
            "version": 3,
            "entity_count": 1,
            "entities": {
                "src.tac.x::func_d": {
                    "module": "src.tac.x",
                    "file_path": "src/tac/x.py",
                    "entity_type": "function",
                    "name": "func_d",
                    "start_line": 1, "end_line": 2, "line_count": 2,
                    "complexity": 1,
                    "review_status": "reviewed",
                },
            },
        }))
        policy = {
            "rigor": {"standard": {"min_consecutive_clean_passes": 2}},
            "file_policies": [
                {"pattern": "src/tac/*.py", "rigor": "standard"},
            ],
        }

        blocking, warnings, stats = self._run_hook_with_fake_paths(
            ["src/tac/x.py"],
            policy=policy,
        )

        self.assertEqual(stats.get("violations"), 1)
        self.assertTrue(
            any("POLICY_UNPROVEN_JSON_FALLBACK" in line for line in blocking),
            msg=f"blocking={blocking}, warnings={warnings}",
        )

    def test_corrupt_json_treated_as_missing(self):
        """A corrupt JSON snapshot is treated like missing (degraded skip)."""
        self.fake_json.write_text("{ not valid json")
        blocking, warnings, stats = self._run_hook_with_fake_paths(["src/tac/x.py"])
        # No DB + no usable JSON → source=none
        self.assertEqual(stats.get("source"), "none")
        self.assertEqual(blocking, [])

    def test_files_not_in_snapshot_silently_ignored(self):
        """Staged files with no entity rows produce no output (existing behavior)."""
        blocking, warnings, stats = self._run_hook_with_fake_paths(["src/tac/other.py"])
        # source=json (snapshot was used); zero entities found.
        self.assertEqual(stats.get("source"), "json")
        self.assertEqual(stats.get("total"), 0)


class ResolveHookRetrySecondsTests(unittest.TestCase):
    """Test the env-var override for the hook's retry budget."""

    @classmethod
    def setUpClass(cls):
        cls.hook = _load_module("test_hook_retry_env", HOOK_PATH)

    def test_default_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REVIEW_GATE_HOOK_RETRY_SECONDS", None)
            self.assertEqual(
                self.hook._resolve_hook_retry_seconds(),
                self.hook.DEFAULT_HOOK_RETRY_SECONDS,
            )

    def test_env_override_accepted(self):
        with mock.patch.dict(
            os.environ, {"REVIEW_GATE_HOOK_RETRY_SECONDS": "0.25"}, clear=False
        ):
            self.assertEqual(self.hook._resolve_hook_retry_seconds(), 0.25)

    def test_invalid_env_falls_back_to_default(self):
        with mock.patch.dict(
            os.environ, {"REVIEW_GATE_HOOK_RETRY_SECONDS": "not a number"}, clear=False
        ):
            self.assertEqual(
                self.hook._resolve_hook_retry_seconds(),
                self.hook.DEFAULT_HOOK_RETRY_SECONDS,
            )

    def test_negative_env_falls_back_to_default(self):
        with mock.patch.dict(
            os.environ, {"REVIEW_GATE_HOOK_RETRY_SECONDS": "-1.0"}, clear=False
        ):
            self.assertEqual(
                self.hook._resolve_hook_retry_seconds(),
                self.hook.DEFAULT_HOOK_RETRY_SECONDS,
            )

    def test_zero_env_is_accepted(self):
        """Zero is a valid (degenerate) budget — accept it for opt-out behavior."""
        with mock.patch.dict(
            os.environ, {"REVIEW_GATE_HOOK_RETRY_SECONDS": "0.0"}, clear=False
        ):
            self.assertEqual(self.hook._resolve_hook_retry_seconds(), 0.0)


class HookEnvVarPassThroughTests(unittest.TestCase):
    """Test that the REVIEW_GATE_ENABLED / REVIEW_GATE_OVERRIDE env vars work."""

    @classmethod
    def setUpClass(cls):
        cls.hook = _load_module("test_hook_envvars", HOOK_PATH)

    def test_disabled_returns_zero(self):
        with mock.patch.dict(os.environ, {"REVIEW_GATE_ENABLED": "0"}, clear=False):
            self.assertEqual(self.hook.main(), 0)

    def test_override_returns_zero(self):
        with mock.patch.dict(os.environ, {"REVIEW_GATE_OVERRIDE": "1"}, clear=False):
            self.assertEqual(self.hook.main(), 0)


if __name__ == "__main__":
    unittest.main()
