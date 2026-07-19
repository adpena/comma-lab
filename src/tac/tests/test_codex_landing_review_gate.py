"""Unit tests for the codex landing review gate (tools/codex_landing_review_gate.py).

Pure-logic tests over ``compute_pending`` + disposition semantics — no live fleet,
no pgrep dependency (alive_fn is injected). Mirrors the triality-drift-detector
testing style: the load-bearing classification lives in module functions.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "codex_landing_review_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("codex_landing_review_gate", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["codex_landing_review_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load()


def _deleg(label, stamp, *, done=True, tmp=None):
    """A delegation dict; ``done`` writes a real .done marker so _landing_done sees it."""
    d = {"label": label, "stamp": stamp}
    if done and tmp is not None:
        marker = tmp / f"{label}_{stamp}.done"
        marker.write_text("rc=0\n")
        d["done_marker"] = str(marker)
        d["last"] = str(marker)  # reuse for mtime
    return d


# ---- compute_pending: the core classifier ----------------------------------
def test_landed_dead_undispositioned_is_pending(tmp_path):
    d = _deleg("arm_a", "S1", tmp=tmp_path)
    pending = G.compute_pending([d], {}, alive_fn=lambda arm, s: False)
    assert len(pending) == 1
    assert pending[0]["label"] == "arm_a"
    assert pending[0]["reason"] == "undispositioned"


def test_landed_but_alive_is_not_pending(tmp_path):
    d = _deleg("arm_a", "S1", tmp=tmp_path)
    # process still alive ⇒ held-by-design, NOT a landing to review
    pending = G.compute_pending([d], {}, alive_fn=lambda arm, s: True)
    assert pending == []


def test_not_done_is_not_pending(tmp_path):
    d = {"label": "arm_a", "stamp": "S1"}  # no marker ⇒ not landed
    pending = G.compute_pending([d], {}, alive_fn=lambda arm, s: False)
    assert pending == []


@pytest.mark.parametrize("status", ["reviewed_committed", "respawned", "closed"])
def test_terminal_disposition_clears(tmp_path, status):
    d = _deleg("arm_a", "S1", tmp=tmp_path)
    disp = {("arm_a", "S1"): {"label": "arm_a", "stamp": "S1", "status": status}}
    pending = G.compute_pending([d], disp, alive_fn=lambda arm, s: False)
    assert pending == []


def test_held_entangled_with_live_blocker_is_not_pending(tmp_path):
    landed = _deleg("arm_a", "S1", tmp=tmp_path)
    blocker = {"label": "live_b", "stamp": "S2"}  # a delegation for the blocker
    disp = {("arm_a", "S1"): {
        "label": "arm_a", "stamp": "S1", "status": "held_entangled",
        "blocked_by": ["live_b"]}}
    # live_b is alive ⇒ legitimately held
    pending = G.compute_pending(
        [landed, blocker], disp, alive_fn=lambda arm, s: arm == "live_b")
    assert pending == []


def test_held_entangled_resurfaces_when_blockers_die(tmp_path):
    landed = _deleg("arm_a", "S1", tmp=tmp_path)
    blocker = {"label": "live_b", "stamp": "S2"}
    disp = {("arm_a", "S1"): {
        "label": "arm_a", "stamp": "S1", "status": "held_entangled",
        "blocked_by": ["live_b"]}}
    # everything dead ⇒ the held landing MUST resurface (held is not a grave)
    pending = G.compute_pending([landed, blocker], disp, alive_fn=lambda arm, s: False)
    assert len(pending) == 1
    assert pending[0]["reason"] == "held_entangled_unblocked"


def test_unknown_status_is_pending_failsafe(tmp_path):
    d = _deleg("arm_a", "S1", tmp=tmp_path)
    disp = {("arm_a", "S1"): {"label": "arm_a", "stamp": "S1", "status": "weird"}}
    pending = G.compute_pending([d], disp, alive_fn=lambda arm, s: False)
    assert len(pending) == 1  # fail-safe toward review


# ---- pending_signature: loop-safety key ------------------------------------
def test_signature_stable_and_order_independent():
    a = [{"label": "x", "stamp": "1"}, {"label": "y", "stamp": "2"}]
    b = [{"label": "y", "stamp": "2"}, {"label": "x", "stamp": "1"}]
    assert G.pending_signature(a) == G.pending_signature(b)
    c = [{"label": "x", "stamp": "1"}]
    assert G.pending_signature(a) != G.pending_signature(c)


def test_empty_signature_is_stable():
    assert G.pending_signature([]) == G.pending_signature([])


# ---- latest_dispositions: latest-row-wins ----------------------------------
def test_latest_row_wins():
    rows = [
        {"label": "a", "stamp": "1", "status": "held_entangled"},
        {"label": "a", "stamp": "1", "status": "reviewed_committed"},
    ]
    latest = G.latest_dispositions(rows)
    assert latest[("a", "1")]["status"] == "reviewed_committed"


def test_rows_missing_keys_skipped():
    rows = [{"status": "closed"}, {"label": "a", "status": "closed"},
            {"label": "a", "stamp": "1", "status": "closed"}]
    latest = G.latest_dispositions(rows)
    assert list(latest.keys()) == [("a", "1")]


# ---- constants sanity ------------------------------------------------------
def test_state_partitions():
    assert {"reviewed_committed", "respawned", "closed"} == G.TERMINAL_STATES
    assert "held_entangled" in G.NONTERMINAL_STATES
    assert G.VALID_STATES == G.TERMINAL_STATES | G.NONTERMINAL_STATES


# ---- --consumed-by enforcement (DISPOSITION != CONSUMPTION, 2026-07-17) ----
# Terminal review dispositions (reviewed_committed / closed) are CUSTODY states;
# they must ALSO name the artifact that absorbed (or knowingly rejected, with a
# recorded decision) the arm's findings. Proven orphan class: the 4-arm
# curvelet/Fourier basis cluster (rc=0 REVIEWED, findings unconsumed).


def _disposition_args(
    status,
    *,
    consumed_by=None,
    blocked_by=None,
    reason="r",
    landing_manifest=None,
    landing_manifest_strict=False,
    landing_manifest_waiver=None,
):
    import argparse

    return argparse.Namespace(
        label="arm_x",
        stamp="S1",
        status=status,
        commit=None,
        respawn=None,
        reason=reason,
        blocked_by=blocked_by,
        consumed_by=consumed_by,
        landing_manifest=landing_manifest,
        landing_manifest_strict=landing_manifest_strict,
        landing_manifest_waiver=landing_manifest_waiver,
    )


@pytest.fixture()
def _ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(G, "LEDGER", ledger)
    return ledger


@pytest.mark.parametrize("status", ["reviewed_committed", "closed"])
def test_terminal_disposition_without_consumed_by_refused(_ledger, status):
    rc = G._cmd_disposition(_disposition_args(status))
    assert rc == 2
    assert not _ledger.exists()  # refused BEFORE any ledger write


@pytest.mark.parametrize("placeholder", ["none", "None", "N/A", "tbd", "<reason>"])
def test_placeholder_consumed_by_refused(_ledger, placeholder):
    rc = G._cmd_disposition(
        _disposition_args("reviewed_committed", consumed_by=placeholder))
    assert rc == 2
    assert not _ledger.exists()


def test_consumed_by_recorded_in_ledger_row(_ledger):
    rc = G._cmd_disposition(_disposition_args(
        "reviewed_committed",
        consumed_by=".omx/research/arm_x_DAG_FEED_20260717.md"))
    assert rc == 0
    rows = G._read_jsonl(_ledger)
    assert len(rows) == 1
    assert rows[0]["consumed_by"] == ".omx/research/arm_x_DAG_FEED_20260717.md"
    assert rows[0]["status"] == "reviewed_committed"


def test_none_with_reason_escape_accepted(_ledger):
    rc = G._cmd_disposition(_disposition_args(
        "closed", consumed_by="none:pure-mechanical arm; no findings emitted"))
    assert rc == 0
    rows = G._read_jsonl(_ledger)
    assert rows[0]["consumed_by"].startswith("none:")


def test_respawned_exempt_from_consumed_by(_ledger):
    rc = G._cmd_disposition(_disposition_args("respawned"))
    assert rc == 0
    assert G._read_jsonl(_ledger)[0]["status"] == "respawned"


def test_held_entangled_exempt_from_consumed_by(_ledger):
    rc = G._cmd_disposition(
        _disposition_args("held_entangled", blocked_by="live_arm"))
    assert rc == 0
    row = G._read_jsonl(_ledger)[0]
    assert row["status"] == "held_entangled"
    assert row["blocked_by"] == ["live_arm"]


def test_old_rows_without_consumed_by_still_terminal(tmp_path):
    # Backwards compatibility: pre-2026-07-17 ledger rows lack consumed_by and
    # MUST still clear pending (enforcement is at the CLI write path only).
    d = _deleg("arm_old", "S0", tmp=tmp_path)
    disp = {("arm_old", "S0"): {
        "label": "arm_old", "stamp": "S0", "status": "reviewed_committed"}}
    assert G.compute_pending([d], disp, alive_fn=lambda a, s: False) == []


def test_cli_parser_accepts_consumed_by(_ledger):
    rc = G.main([
        "disposition", "--label", "arm_x", "--stamp", "S1",
        "--status", "reviewed_committed", "--commit", "abc123",
        "--consumed-by", "task#999"])
    assert rc == 0
    assert G._read_jsonl(_ledger)[0]["consumed_by"] == "task#999"


# ---- typed BASE..HEAD landing manifest integration (task #555) -------------
def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Landing Gate Test",
        "GIT_AUTHOR_EMAIL": "gate@example.invalid",
        "GIT_COMMITTER_NAME": "Landing Gate Test",
        "GIT_COMMITTER_EMAIL": "gate@example.invalid",
    }
    proc = subprocess.run(["git", "-C", str(repo), *args], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.fixture()
def _manifest_repo(tmp_path, monkeypatch):
    from tac.landing_diff_manifest import build_manifest, write_manifest

    repo = tmp_path / "manifest_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "changed.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "changed.txt")
    _git(repo, "commit", "-m", "head")
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(G, "REPO", repo)

    def make(declarations, name="manifest.json"):
        manifest = build_manifest(repo, base, head, declarations)
        path = tmp_path / name
        write_manifest(path, manifest)
        return path, manifest

    return make


def test_terminal_missing_manifest_warns_and_records_migration_debt(_ledger, capsys):
    rc = G._cmd_disposition(_disposition_args("reviewed_committed", consumed_by="task#555"))
    assert rc == 0
    row = G._read_jsonl(_ledger)[0]
    assert row["landing_diff_manifest"]["mode"] == "warn"
    assert row["landing_diff_manifest"]["blockers"] == ["landing_diff_manifest_missing"]
    assert "WARN-ONLY" in capsys.readouterr().err


def test_terminal_missing_manifest_strict_refuses_before_append(_ledger):
    rc = G._cmd_disposition(
        _disposition_args("reviewed_committed", consumed_by="task#555", landing_manifest_strict=True)
    )
    assert rc == 2
    assert not _ledger.exists()


def test_incomplete_manifest_warn_only_appends_blocker_summary(_ledger, _manifest_repo, capsys):
    path, manifest = _manifest_repo({})
    assert not manifest.complete
    rc = G._cmd_disposition(_disposition_args("closed", consumed_by="task#555", landing_manifest=str(path)))
    assert rc == 0
    row = G._read_jsonl(_ledger)[0]
    assert row["landing_diff_manifest"]["path_count"] == 1
    assert row["landing_diff_manifest"]["blockers"] == ["unaccounted_path"]
    assert row["landing_diff_manifest"]["blocker_details"][0]["path"] == "changed.txt"
    warning = capsys.readouterr().err
    assert "WARN-ONLY" in warning
    assert "unaccounted_path:changed.txt" in warning


def test_incomplete_manifest_strict_refuses(_ledger, _manifest_repo):
    path, _ = _manifest_repo({})
    rc = G._cmd_disposition(
        _disposition_args("closed", consumed_by="task#555", landing_manifest=str(path), landing_manifest_strict=True)
    )
    assert rc == 2
    assert not _ledger.exists()


def test_real_same_line_waiver_allows_strict_and_is_recorded(_ledger, _manifest_repo):
    path, _ = _manifest_repo({})
    rationale = "historical arm predates typed receipts; task#555 retains the measured debt"
    rc = G._cmd_disposition(
        _disposition_args(
            "closed",
            consumed_by="task#555",
            landing_manifest=str(path),
            landing_manifest_strict=True,
            landing_manifest_waiver=rationale,
        )
    )
    assert rc == 0
    evidence = G._read_jsonl(_ledger)[0]["landing_diff_manifest"]
    assert evidence["waiver"] == rationale
    assert evidence["blockers"] == ["unaccounted_path"]


@pytest.mark.parametrize("waiver", ["todo", "<rationale>", "n/a"])
def test_placeholder_manifest_waiver_refuses(_ledger, waiver):
    rc = G._cmd_disposition(
        _disposition_args(
            "closed", consumed_by="task#555", landing_manifest_strict=True, landing_manifest_waiver=waiver
        )
    )
    assert rc == 2
    assert not _ledger.exists()


def test_complete_verified_manifest_passes_strict(_ledger, _manifest_repo):
    path, manifest = _manifest_repo({"changed.txt": "merged"})
    assert manifest.complete
    rc = G._cmd_disposition(
        _disposition_args(
            "reviewed_committed", consumed_by="task#555", landing_manifest=str(path), landing_manifest_strict=True
        )
    )
    assert rc == 0
    evidence = G._read_jsonl(_ledger)[0]["landing_diff_manifest"]
    assert evidence["valid"] is True
    assert evidence["complete"] is True
    assert evidence["blockers"] == []
    assert len(evidence["receipt_sha256"]) == 64
    assert evidence["tracked_diff_sha256"] == manifest.tracked_diff_sha256


def test_respawned_remains_manifest_exempt(_ledger):
    rc = G._cmd_disposition(_disposition_args("respawned"))
    assert rc == 0
    assert G._read_jsonl(_ledger)[0]["landing_diff_manifest"] is None
