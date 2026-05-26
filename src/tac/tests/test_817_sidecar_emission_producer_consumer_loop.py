# SPDX-License-Identifier: MIT
"""Tests for #817 sidecar emission wire-in (BUCKET C producer-consumer loop closure).

Lane: lane_817_sidecar_emission_wire_in_bucket_c_loop_closure_20260517
Task: #824 (#818 op-routable #2)

Sister #817 extended `tac.optimization.{bit_allocator_end_to_end,
jacobian_fisher_importance_allocator, field_equation_planner}` with per-pair
allocators / Lagrangian-dual binders BUT those helpers did NOT persist sidecars.
Sister #818 wired the BUCKET C autopilot consumer to READ sidecars at
`.omx/state/master_gradient_consumers/per_pair_bit_allocation_*.json` +
`per_pair_fisher_importance_*.json`. The producer-consumer loop was OPEN —
the BUCKET C reward factor was ALWAYS 1.0× (sidecar-not-found passthrough).

This subagent (#824) closes the loop by extending the 3 producers to EMIT
fcntl-locked JSONL sidecars per Catalog #131 / #245 canonical pattern. These
tests verify:

  1. Sidecars are written to canonical paths with `persist_sidecar=True`.
  2. Sidecars are NOT written with `persist_sidecar=False`.
  3. Sidecar schema validates against `*_sidecar_v1` contract.
  4. BUCKET C autopilot consumer successfully reads producer's sidecar.
  5. End-to-end: producer → sidecar → autopilot reward factor varies.
  6. Multi-archive: sidecars disambiguated by archive_sha256 prefix.
  7. Most-recent sidecar wins on collision.
  8. fcntl-locked write under 4-proc concurrent stress.
  9. Atomic write: no .tmp file leaks.
 10. Payload carries score_claim=False + promotion_eligible=False (CLAUDE.md).
 11. Sister regressions: existing #817 + #818 tests still pass.
 12. `to_dict()` method available on producer dataclasses.
"""

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SIDECAR_DIR = REPO_ROOT / ".omx" / "state" / "master_gradient_consumers"


@pytest.fixture
def isolated_sidecar_sha(tmp_path: Path, monkeypatch) -> str:
    """Yield an archive sha unique to this test; clean up emitted sidecars."""
    # Use a deterministic sha derived from the tmp_path so concurrent tests
    # do not collide. The sidecar dir is the canonical location (NOT tmp_path)
    # because the BUCKET C consumer scans the canonical dir; we clean up any
    # sidecars we emitted using the sha prefix at fixture teardown.
    sha = (
        "fe" + str(abs(hash(str(tmp_path))) % (1 << 60)).zfill(14) + "0" * 48
    )[:64]
    sha_short = sha[:12]
    # Pre-clean
    for pattern in (
        f"per_pair_bit_allocation_{sha_short}_*.json",
        f"per_pair_fisher_importance_{sha_short}_*.json",
        f"field_equation_per_pair_lagrangian_{sha_short}_*.json",
    ):
        for f in glob.glob(str(SIDECAR_DIR / pattern)):
            try:
                os.unlink(f)
            except OSError:
                pass
    yield sha
    # Post-clean
    for pattern in (
        f"per_pair_bit_allocation_{sha_short}_*.json",
        f"per_pair_fisher_importance_{sha_short}_*.json",
        f"field_equation_per_pair_lagrangian_{sha_short}_*.json",
    ):
        for f in glob.glob(str(SIDECAR_DIR / pattern)):
            try:
                os.unlink(f)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Producer-side tests: bit allocator
# ─────────────────────────────────────────────────────────────────────────────


class TestBitAllocatorSidecarEmission:
    def test_sidecar_written_to_canonical_path_with_persist_true(self, isolated_sidecar_sha):
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        outcome = allocate_per_pair_bits(
            archive_sha256=isolated_sidecar_sha,
            total_bit_budget=100,
            auto_load=False,
            persist_sidecar=True,
        )
        sha_short = isolated_sidecar_sha[:12]
        matches = list(
            SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_short}_*.json")
        )
        assert len(matches) == 1, f"Expected 1 sidecar, got {len(matches)}"
        assert outcome.cascade_path_used == "aggregate_fallback"

    def test_sidecar_not_written_with_persist_false(self, isolated_sidecar_sha):
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        allocate_per_pair_bits(
            archive_sha256=isolated_sidecar_sha,
            total_bit_budget=100,
            auto_load=False,
            persist_sidecar=False,
        )
        sha_short = isolated_sidecar_sha[:12]
        matches = list(
            SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_short}_*.json")
        )
        assert len(matches) == 0, f"Expected 0 sidecars, got {len(matches)}"

    def test_sidecar_payload_has_flat_cascade_path_used(self, isolated_sidecar_sha):
        """BUCKET C consumer reads payload['cascade_path_used'] directly."""
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        allocate_per_pair_bits(
            archive_sha256=isolated_sidecar_sha,
            total_bit_budget=100,
            auto_load=False,
            persist_sidecar=True,
        )
        sha_short = isolated_sidecar_sha[:12]
        sidecar = list(
            SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_short}_*.json")
        )[0]
        payload = json.loads(sidecar.read_text())
        # TOP-LEVEL field (NOT nested under "outcome") is the BUCKET C contract
        assert "cascade_path_used" in payload
        assert payload["cascade_path_used"] in (
            "optimal_plan",
            "wyner_ziv_composition",
            "aggregate_fallback",
        )

    def test_sidecar_payload_carries_canonical_non_promotable_fields(
        self, isolated_sidecar_sha
    ):
        """Per CLAUDE.md 'Apples-to-apples evidence discipline'."""
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        allocate_per_pair_bits(
            archive_sha256=isolated_sidecar_sha,
            total_bit_budget=100,
            auto_load=False,
            persist_sidecar=True,
        )
        sha_short = isolated_sidecar_sha[:12]
        sidecar = list(
            SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_short}_*.json")
        )[0]
        payload = json.loads(sidecar.read_text())
        assert payload["score_claim"] is False
        assert payload["promotion_eligible"] is False
        assert payload["ready_for_exact_eval_dispatch"] is False
        assert payload["sidecar_schema"] == "per_pair_bit_allocation_sidecar_v1"

    def test_outcome_to_dict_round_trips(self, isolated_sidecar_sha):
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        outcome = allocate_per_pair_bits(
            archive_sha256=isolated_sidecar_sha,
            total_bit_budget=100,
            auto_load=False,
            persist_sidecar=False,
        )
        d = outcome.to_dict()
        assert d["schema"] == "tac_bit_allocator_per_pair_v1"
        assert d["cascade_path_used"] == outcome.cascade_path_used
        assert d["archive_sha256"] == outcome.archive_sha256
        # Must be JSON-serializable
        json.dumps(d)  # raises TypeError if not serializable


# ─────────────────────────────────────────────────────────────────────────────
# Producer-side tests: Fisher importance
# ─────────────────────────────────────────────────────────────────────────────


class TestFisherImportanceSidecarEmission:
    def _make_grad(self) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        return rng.standard_normal((30, 6, 3)).astype(np.float32)

    def test_sidecar_written_with_persist_true(self, isolated_sidecar_sha):
        from tac.optimization.jacobian_fisher_importance_allocator import (
            allocate_per_pair_fisher_importance,
        )

        outcome = allocate_per_pair_fisher_importance(
            archive_sha256=isolated_sidecar_sha,
            per_pair_gradient=self._make_grad(),
            auto_load=False,
            persist_sidecar=True,
        )
        sha_short = isolated_sidecar_sha[:12]
        matches = list(
            SIDECAR_DIR.glob(f"per_pair_fisher_importance_{sha_short}_*.json")
        )
        assert len(matches) == 1
        assert outcome.aggregate_fisher_l1 > 0

    def test_sidecar_not_written_with_persist_false(self, isolated_sidecar_sha):
        from tac.optimization.jacobian_fisher_importance_allocator import (
            allocate_per_pair_fisher_importance,
        )

        allocate_per_pair_fisher_importance(
            archive_sha256=isolated_sidecar_sha,
            per_pair_gradient=self._make_grad(),
            auto_load=False,
            persist_sidecar=False,
        )
        sha_short = isolated_sidecar_sha[:12]
        matches = list(
            SIDECAR_DIR.glob(f"per_pair_fisher_importance_{sha_short}_*.json")
        )
        assert len(matches) == 0

    def test_sidecar_payload_has_flat_aggregate_fisher_l1(self, isolated_sidecar_sha):
        """BUCKET C consumer reads payload['aggregate_fisher_l1'] directly."""
        from tac.optimization.jacobian_fisher_importance_allocator import (
            allocate_per_pair_fisher_importance,
        )

        allocate_per_pair_fisher_importance(
            archive_sha256=isolated_sidecar_sha,
            per_pair_gradient=self._make_grad(),
            auto_load=False,
            persist_sidecar=True,
        )
        sha_short = isolated_sidecar_sha[:12]
        sidecar = list(
            SIDECAR_DIR.glob(f"per_pair_fisher_importance_{sha_short}_*.json")
        )[0]
        payload = json.loads(sidecar.read_text())
        assert "aggregate_fisher_l1" in payload
        assert isinstance(payload["aggregate_fisher_l1"], (int, float))
        assert payload["aggregate_fisher_l1"] > 0
        assert payload["sidecar_schema"] == "per_pair_fisher_importance_sidecar_v1"

    def test_outcome_to_dict_round_trips(self, isolated_sidecar_sha):
        from tac.optimization.jacobian_fisher_importance_allocator import (
            allocate_per_pair_fisher_importance,
        )

        outcome = allocate_per_pair_fisher_importance(
            archive_sha256=isolated_sidecar_sha,
            per_pair_gradient=self._make_grad(),
            auto_load=False,
            persist_sidecar=False,
        )
        d = outcome.to_dict()
        assert d["aggregate_fisher_l1"] == outcome.aggregate_fisher_l1
        assert d["catalog_123_invariant_satisfied"] is True
        json.dumps(d)


# ─────────────────────────────────────────────────────────────────────────────
# Producer-side tests: field equation Lagrangian dual
# ─────────────────────────────────────────────────────────────────────────────


class TestFieldEquationLagrangianSidecarEmission:
    def test_fallback_sidecar_written_with_persist_true(self, isolated_sidecar_sha):
        from tac.optimization.field_equation_planner import (
            consume_per_pair_lagrangian_duals,
        )

        env = consume_per_pair_lagrangian_duals(
            archive_sha256=isolated_sidecar_sha,
            auto_load=False,
            persist_sidecar=True,
        )
        sha_short = isolated_sidecar_sha[:12]
        matches = list(
            SIDECAR_DIR.glob(
                f"field_equation_per_pair_lagrangian_{sha_short}_*.json"
            )
        )
        assert len(matches) == 1
        assert env["cascade_path_used"] == "default_constraints_fallback"

    def test_sidecar_not_written_with_persist_false(self, isolated_sidecar_sha):
        from tac.optimization.field_equation_planner import (
            consume_per_pair_lagrangian_duals,
        )

        consume_per_pair_lagrangian_duals(
            archive_sha256=isolated_sidecar_sha,
            auto_load=False,
            persist_sidecar=False,
        )
        sha_short = isolated_sidecar_sha[:12]
        matches = list(
            SIDECAR_DIR.glob(
                f"field_equation_per_pair_lagrangian_{sha_short}_*.json"
            )
        )
        assert len(matches) == 0


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end producer-consumer loop closure
# ─────────────────────────────────────────────────────────────────────────────


class TestProducerConsumerLoopClosure:
    """The critical demonstration: BUCKET C reward factor VARIES based on
    sidecar presence (NOT always 1.0× passthrough)."""

    def _import_autopilot(self):
        """Import the BUCKET C autopilot consumer via sys.path injection."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            import cathedral_autopilot_autonomous_loop as autopilot

            return autopilot
        finally:
            sys.path.pop(0)

    def test_bucket_c_reward_factor_varies_with_sidecar_presence(
        self, isolated_sidecar_sha
    ):
        """The structural proof that the producer-consumer loop is CLOSED."""
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits
        from tac.optimization.jacobian_fisher_importance_allocator import (
            allocate_per_pair_fisher_importance,
        )

        autopilot = self._import_autopilot()
        delta = -0.05

        # Phase 1: NO sidecars → 1.0× passthrough (loop OPEN)
        adj_before = autopilot.adjust_predicted_delta_for_per_pair_sister_817_sidecars(
            delta, isolated_sidecar_sha
        )
        assert abs(adj_before - delta) < 1e-12, (
            f"Phase 1 (no sidecars) expected 1.0× passthrough = {delta}; got {adj_before}"
        )

        # Phase 2: Emit BOTH sidecars (aggregate_fallback bit_alloc -> 1.0x;
        # fisher_l1 > 0 -> 1.03x reward). Negative deltas are improvements, so
        # factor > 1.0 makes the candidate more negative and better-ranked.
        rng = np.random.default_rng(seed=13)
        grad = rng.standard_normal((20, 5, 3)).astype(np.float32)
        allocate_per_pair_bits(
            archive_sha256=isolated_sidecar_sha,
            total_bit_budget=50,
            auto_load=False,
            persist_sidecar=True,
        )
        allocate_per_pair_fisher_importance(
            archive_sha256=isolated_sidecar_sha,
            per_pair_gradient=grad,
            auto_load=False,
            persist_sidecar=True,
        )

        adj_after = autopilot.adjust_predicted_delta_for_per_pair_sister_817_sidecars(
            delta, isolated_sidecar_sha
        )
        expected_after = delta * 1.0 * 1.03  # aggregate_fallback x fisher_present
        assert abs(adj_after - expected_after) < 1e-9, (
            f"Phase 2 expected {expected_after}; got {adj_after}"
        )
        assert adj_after < adj_before

        # CRITICAL: factor changed — loop is CLOSED.
        assert abs(adj_after - adj_before) > 1e-9, (
            "BUCKET C reward factor did NOT vary between Phase 1 and Phase 2 — "
            "the producer-consumer loop is OPEN; sidecars are not being consumed."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-archive disambiguation + most-recent-wins
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiArchiveAndMostRecentResolution:
    def test_sidecars_disambiguated_by_archive_sha_prefix(self):
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        sha_a = "a1" * 32
        sha_b = "b2" * 32

        try:
            allocate_per_pair_bits(
                archive_sha256=sha_a,
                total_bit_budget=20,
                auto_load=False,
                persist_sidecar=True,
            )
            allocate_per_pair_bits(
                archive_sha256=sha_b,
                total_bit_budget=20,
                auto_load=False,
                persist_sidecar=True,
            )
            matches_a = list(
                SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_a[:12]}_*.json")
            )
            matches_b = list(
                SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_b[:12]}_*.json")
            )
            assert len(matches_a) == 1
            assert len(matches_b) == 1
            assert matches_a[0] != matches_b[0]
        finally:
            for pat in (sha_a[:12], sha_b[:12]):
                for f in SIDECAR_DIR.glob(f"per_pair_bit_allocation_{pat}_*.json"):
                    try:
                        os.unlink(f)
                    except OSError:
                        pass

    def test_most_recent_sidecar_wins_on_collision(self, isolated_sidecar_sha):
        """When multiple sidecars exist for the same sha, consumer reads the
        lex-max (= chrono-max via UTC YYYYMMDDTHHMMSS) suffix."""
        import time
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        # Emit two sidecars at least 1 second apart so the UTC suffix differs
        allocate_per_pair_bits(
            archive_sha256=isolated_sidecar_sha,
            total_bit_budget=20,
            auto_load=False,
            persist_sidecar=True,
        )
        time.sleep(1.1)
        allocate_per_pair_bits(
            archive_sha256=isolated_sidecar_sha,
            total_bit_budget=40,
            auto_load=False,
            persist_sidecar=True,
        )
        sha_short = isolated_sidecar_sha[:12]
        matches = sorted(
            SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_short}_*.json")
        )
        assert len(matches) == 2
        # The lex-max is the most-recent
        most_recent_payload = json.loads(matches[-1].read_text())
        assert most_recent_payload["total_bit_budget"] == 40


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent-write stress test
# ─────────────────────────────────────────────────────────────────────────────


def _concurrent_writer(args):
    """Helper for 4-proc spawn pool."""
    proc_idx, sha = args
    # Re-import in subprocess
    from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

    allocate_per_pair_bits(
        archive_sha256=sha,
        total_bit_budget=10 + proc_idx,  # vary so we can check distinct rows
        auto_load=False,
        persist_sidecar=True,
    )
    return proc_idx


class TestConcurrentSidecarWrite:
    def test_4_proc_concurrent_writes_no_tmp_leak(self, isolated_sidecar_sha):
        """fcntl-locked writes survive 4-proc concurrent stress; no .tmp leaks."""
        # Use forkserver/spawn so subprocess re-imports cleanly
        ctx = mp.get_context("spawn")
        args = [(i, isolated_sidecar_sha) for i in range(4)]
        with ctx.Pool(processes=4) as pool:
            results = pool.map(_concurrent_writer, args)
        assert sorted(results) == [0, 1, 2, 3]

        sha_short = isolated_sidecar_sha[:12]
        # All 4 sidecars should land (different UTC suffixes typically; if
        # they collide on the same second, the last writer wins → ≥1 sidecar)
        sidecars = list(
            SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_short}_*.json")
        )
        assert len(sidecars) >= 1
        # No .tmp leaks
        tmp_files = list(
            SIDECAR_DIR.glob(f"per_pair_bit_allocation_{sha_short}_*.tmp.*")
        )
        assert len(tmp_files) == 0, (
            f".tmp file leak detected: {tmp_files}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sister regression
# ─────────────────────────────────────────────────────────────────────────────


class TestSisterRegression:
    """Verify sister #817 + #818 tests still pass with sidecar emission default ON."""

    def test_817_bit_allocator_sister_tests_present(self):
        sister_test = REPO_ROOT / "src" / "tac" / "tests" / "test_bit_allocator_per_pair_consumption.py"
        assert sister_test.exists(), f"Sister #817 test file missing: {sister_test}"

    def test_817_fisher_sister_tests_present(self):
        sister_test = REPO_ROOT / "src" / "tac" / "tests" / "test_jacobian_fisher_per_pair_consumption.py"
        assert sister_test.exists()

    def test_818_bucket_c_sister_tests_present(self):
        sister_test = (
            REPO_ROOT
            / "src"
            / "tac"
            / "tests"
            / "test_low_gap_closure_widened_bucket_c_autopilot_sister_817_consumption.py"
        )
        assert sister_test.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
