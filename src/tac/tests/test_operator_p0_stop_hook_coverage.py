"""CLASS 3 (bug-class sweep 2026-07-17): the operator-P0 stop hook must not demand re-registration
of an ALREADY-REGISTERED P0. Fixtures are today's two real false-positive designations."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

hook = importlib.import_module("tools.operator_p0_stop_hook")

# Real ledger rows (2026-07-17) that the hook re-demanded despite being registered.
_ROWS = [
    {"p0_id": "p0_v10_capstone_cold_start_seeded_20260717",
     "verbatim_ask": "We should advance to v10 as our capstone frontier and clear up naming "
                     "resolutions and be very rigorous about the seeded cold start"},
    {"p0_id": "p0_bug_class_sweep_20260717",
     "verbatim_ask": "Fix bug classes and meta bugs everywhere"},
    {"p0_id": "p0_resume_warmup_geometry_20260717",
     "verbatim_ask": "Should we be warming upon resume for optimal dynamics and convergence "
                     "against deep math"},
]


def test_real_false_positive_v10_is_covered():
    d = "We should advance to v10 as our capstone frontier and clear up naming resolutions"
    assert hook.designation_already_covered(d, _ROWS, use_fm=False) is True


def test_real_false_positive_bug_sweep_is_covered():
    d = "Fix bug classes and meta bugs everywhere"
    assert hook.designation_already_covered(d, _ROWS, use_fm=False) is True


def test_genuinely_new_designation_still_demanded():
    d = "pursue a brand new chroma pose carrier as P0 immediately"
    assert hook.designation_already_covered(d, _ROWS, use_fm=False) is False


def test_too_few_tokens_fails_closed():
    # <3 meaningful tokens -> cannot judge similarity -> still demand (fail-closed).
    assert hook.designation_already_covered("fix now", _ROWS, use_fm=False) is False


def test_empty_rows_never_covered():
    assert hook.designation_already_covered("Fix bug classes everywhere", [], use_fm=False) is False


def test_containment_directional():
    d = hook._cover_tokens("advance v10 capstone frontier")
    ask = hook._cover_tokens("advance v10 capstone frontier clear naming resolutions rigorous")
    assert hook._containment(d, ask) == 1.0  # all designation tokens present in ask
    assert hook._containment(ask, d) < 1.0   # ask has extra tokens


def test_fm_never_sole_gate_below_gray_band():
    # A directive with near-zero deterministic overlap must NOT be dropped even if FM would say
    # covered — the FM is only consulted in the gray band with a nonzero deterministic corroboration.
    called = {"fm": False}

    def _fake_fm(directive, ask, timeout=15):
        called["fm"] = True
        return True

    orig = hook._fm_covered
    hook._fm_covered = _fake_fm
    try:
        assert hook.designation_already_covered(
            "completely unrelated request about tunnels", _ROWS, use_fm=True) is False
        assert called["fm"] is False  # never consulted below the gray band
    finally:
        hook._fm_covered = orig


# ─────────────────────── CLASS 4: zero-work arm detector ───────────────────────
import datetime as _dt  # noqa: E402

liveness = importlib.import_module("tools.subagent_liveness")


def _mk_progress(tmp_path, rows):
    p = tmp_path / "subagent_progress.jsonl"
    import json as _j
    p.write_text("\n".join(_j.dumps(r) for r in rows) + "\n")
    return p


def test_zero_work_arm_flagged(tmp_path):
    now = _dt.datetime(2026, 7, 17, 12, 0, tzinfo=_dt.UTC)
    stale = (now - _dt.timedelta(minutes=40)).isoformat()
    p = _mk_progress(tmp_path, [
        {"subagent_id": "dead_arm", "step": 0, "status": "in_progress",
         "written_at_utc": stale, "next_action": "start"},
    ])
    arms = liveness.stale_zero_work_arms(p, stale_minutes=20, max_age_hours=24, now=now)
    assert [a["subagent_id"] for a in arms] == ["dead_arm"]


def test_advanced_arm_not_flagged(tmp_path):
    now = _dt.datetime(2026, 7, 17, 12, 0, tzinfo=_dt.UTC)
    stale = (now - _dt.timedelta(minutes=40)).isoformat()
    p = _mk_progress(tmp_path, [
        {"subagent_id": "busy", "step": 0, "status": "in_progress", "written_at_utc": stale},
        {"subagent_id": "busy", "step": 3, "status": "in_progress", "written_at_utc": stale},
    ])
    assert liveness.stale_zero_work_arms(p, now=now) == []


def test_complete_arm_not_flagged(tmp_path):
    now = _dt.datetime(2026, 7, 17, 12, 0, tzinfo=_dt.UTC)
    stale = (now - _dt.timedelta(minutes=40)).isoformat()
    p = _mk_progress(tmp_path, [
        {"subagent_id": "done", "step": "complete", "status": "complete", "written_at_utc": stale},
    ])
    assert liveness.stale_zero_work_arms(p, now=now) == []


def test_fresh_arm_not_yet_flagged(tmp_path):
    now = _dt.datetime(2026, 7, 17, 12, 0, tzinfo=_dt.UTC)
    fresh = (now - _dt.timedelta(minutes=5)).isoformat()
    p = _mk_progress(tmp_path, [
        {"subagent_id": "new", "step": 0, "status": "in_progress", "written_at_utc": fresh},
    ])
    assert liveness.stale_zero_work_arms(p, stale_minutes=20, now=now) == []


def test_ancient_arm_excluded(tmp_path):
    now = _dt.datetime(2026, 7, 17, 12, 0, tzinfo=_dt.UTC)
    old = (now - _dt.timedelta(hours=48)).isoformat()
    p = _mk_progress(tmp_path, [
        {"subagent_id": "ancient", "step": 0, "status": "in_progress", "written_at_utc": old},
    ])
    assert liveness.stale_zero_work_arms(p, max_age_hours=24, now=now) == []
