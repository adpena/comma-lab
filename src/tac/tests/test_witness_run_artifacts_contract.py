# SPDX-License-Identifier: MIT
"""Contract + drift + anti-hardcoding guard for :mod:`tac.witness_run_artifacts`.

Three protections (the self-protecting half of the two-landing fix):

1. **Drift test** — parse the PRODUCER sources (trainer + checkpoint-retention) for
   ``levelset_*.{npz,json}`` literals and assert every one is declared in the
   contract. Add/rename a producer artifact and this fails until the contract is
   updated — the coupling can never silently drift again.
2. **Liveness regression** — a run dir with ONLY a streaming ``daemon.log`` (no
   checkpoint yet) is NOT stale (the 2026-07-10 false-RED bug); ``observer.log``
   alone does NOT count as trainer liveness (would mask a hung trainer).
3. **Anti-hardcoding ratchet** — no NEW file may spell a run-artifact literal
   inline; the offender set may only shrink from the migration baseline.
"""
from __future__ import annotations

import re
from pathlib import Path

from tac import witness_run_artifacts as wra

_REPO = Path(__file__).resolve().parents[3]

# Producers that WRITE the run artifacts (drift test parses these).
_PRODUCER_SOURCES = (
    _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py",
    _REPO / "src" / "tac" / "checkpoint_retention.py",
)

_ARTIFACT_LITERAL = re.compile(r"levelset_[A-Za-z0-9_]+\.(?:npz|json)")


# --------------------------------------------------------------- drift --------
def test_contract_covers_every_producer_emission():
    """Every ``levelset_*.{npz,json}`` literal a producer writes is in the contract."""
    declared = set(wra.TRAINER_ARTIFACTS)
    for src in _PRODUCER_SOURCES:
        assert src.exists(), f"producer source missing: {src}"
        found = set(_ARTIFACT_LITERAL.findall(src.read_text()))
        missing = found - declared
        assert not missing, (
            f"{src.name} emits run-artifact literal(s) not in the contract: "
            f"{sorted(missing)} — add them to tac.witness_run_artifacts.TRAINER_ARTIFACTS"
        )


def test_contract_names_are_distinct_and_nonempty():
    assert len(set(wra.TRAINER_ARTIFACTS)) == len(wra.TRAINER_ARTIFACTS)
    assert all(n and n.endswith((".npz", ".json")) for n in wra.TRAINER_ARTIFACTS)
    assert wra.COSTATE_JSONL.endswith(".jsonl")


# ------------------------------------------------------------- liveness -------
def test_streaming_log_alone_is_live_not_stale(tmp_path):
    """A run streaming daemon.log with NO checkpoint yet is live (the false-RED fix)."""
    run = tmp_path / "levelset_v999_smoke_20260710T000000Z"
    run.mkdir()
    log = run / "daemon.log"
    log.write_text('{"stage":"loss_terms","ep":3}\n')
    now = log.stat().st_mtime + 5.0
    age = wra.freshest_signal_age_s(run, now)
    assert age is not None and age < 3600.0, "fresh trainer log must count as liveness"
    assert log in wra.liveness_paths(run)


def test_observer_log_excluded_from_liveness(tmp_path):
    """observer.log must NOT count — it can stay fresh while the trainer is hung."""
    run = tmp_path / "levelset_v999_smoke_20260710T000001Z"
    run.mkdir()
    (run / "observer.log").write_text("heartbeat\n")
    assert wra.progress_log_paths(run) == []
    assert wra.freshest_signal_age_s(run, 1e12) is None  # no trainer-owned signal


def test_named_signal_paths_and_newest_run_dir(tmp_path):
    a = tmp_path / "levelset_a_20260101T000000Z"
    b = tmp_path / "levelset_b_20260102T000000Z"
    a.mkdir()
    b.mkdir()
    (a / wra.EMA_NPZ).write_bytes(b"x")
    (b / wra.RESUME_NPZ).write_bytes(b"y")
    import os

    os.utime(b / wra.RESUME_NPZ, (2e9, 2e9))  # make b newest
    assert wra.newest_run_dir(tmp_path) == b
    assert (a / wra.EMA_NPZ) in wra.signal_paths(a)


# ---------------------------------------------------- anti-hardcoding ratchet -
# Files that still spell a run-artifact literal inline, pending migration onto the
# contract. This set may only SHRINK — a new file with a literal fails the guard.
# canonical_equations/*.py are EXEMPT (historical-provenance docstrings, not live
# path couplings). Remove a file here when it is migrated.
_MIGRATION_BASELINE = frozenset({
    # migrated (live couplings on the contract); remain only for doc-string /
    # comment / log-message mentions of the filename (never a path construction):
    "src/tac/witness_control/__init__.py",
    "src/tac/witness_control/dynamics_analyzer.py",
    "src/tac/witness_control/shadow_controller.py",
    "src/tac/witness_control/trace_probes.py",
    "src/tac/witness_dsl/campaign.py",
    "tools/build_witness_showcase.py",
    "tools/costate_digest.py",
    "tools/costate_observer_loop.py",
    "tools/costate_shadow_report.py",
    "tools/dashboard_flow_sequence.py",
    "tools/dashboard_server.py",
    "tools/launch_witness_run.py",
    "tools/levelset_byte_close_and_eval.py",
    "tools/levelset_torch_inflate_parity.py",
    "tools/mlx_gpu_determinism_probe.py",
    "tools/render_levelset_dashboard.py",
    "tools/render_witness_morse_smale_viz.py",
    "tools/tau_crossover_trainflow_probe_n600.py",
    "tools/witness_annulus_convergence.py",
    "tools/witness_annulus_live_monitor.py",
    "tools/witness_dynamics_analyzer.py",
    "tools/witness_observer_replay.py",
    "tools/witness_per_stage_annulus_attribution.py",
    "tools/witness_run_introspect.py",
    "tools/witness_tau_mq_confirm.py",
})

# The full literal set the guard forbids (contract members + costate jsonl).
_GUARDED = re.compile(
    "|".join(re.escape(n) for n in (*wra.TRAINER_ARTIFACTS, wra.COSTATE_JSONL))
)
_SCAN_ROOTS = ("tools", "src/tac")
_EXEMPT_SUBSTR = ("/tests/", "test_", "witness_run_artifacts.py",
                  "witness_checkin.py", "/canonical_equations/")


def _current_offenders() -> set[str]:
    offenders: set[str] = set()
    for root in _SCAN_ROOTS:
        for py in (_REPO / root).rglob("*.py"):
            rel = py.relative_to(_REPO).as_posix()
            if any(s in rel for s in _EXEMPT_SUBSTR):
                continue
            if _GUARDED.search(py.read_text()):
                offenders.add(rel)
    return offenders


def test_no_new_hardcoded_run_artifact_literals():
    """RATCHET: the set of files hardcoding a run-artifact literal may only shrink."""
    offenders = _current_offenders()
    new = offenders - _MIGRATION_BASELINE
    assert not new, (
        "NEW hardcoded run-artifact literal(s) — import them from "
        f"tac.witness_run_artifacts instead: {sorted(new)}"
    )
