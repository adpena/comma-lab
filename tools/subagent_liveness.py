"""Zero-work arm death detector (CLASS 4, bug-class sweep 2026-07-17). DETECT-ONLY.

WHY (2026-07-17: a SPEC_v10 arm died at ~15 tokens / 2 tool-uses having done NO work; only a human
notification surfaced it): a subagent that registers its step-0 checkpoint and then goes silent
without ever advancing is invisible to the apparatus. This reads the canonical
``.omx/state/subagent_progress.jsonl`` (fcntl-appended by ``tools/subagent_checkpoint.py``),
aggregates the latest state per ``subagent_id``, and flags arms that:

  (a) never advanced past ``min_step`` (default 0 — only the registration checkpoint), AND
  (b) last wrote between ``stale_minutes`` and ``max_age_hours`` ago (recently-started-then-silent,
      not ancient history), AND
  (c) are still ``in_progress`` (not ``complete`` / ``blocked``).

It writes nothing, kills nothing — it only returns candidates for a digest line. Sister of the
witness_chain_watchdog (verdict-only liveness) at the SUBAGENT surface."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PROGRESS = _REPO / ".omx" / "state" / "subagent_progress.jsonl"


def _parse_utc(ts: str) -> _dt.datetime | None:
    try:
        d = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.UTC)
    except Exception:
        return None


def _latest_by_arm(path: Path) -> dict[str, dict]:
    """Latest record + max numeric step per subagent_id (streamed; last-write-wins)."""
    agg: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                sid = r.get("subagent_id")
                if not sid:
                    continue
                cur = agg.get(sid)
                if cur is None:
                    agg[sid] = {"last": r, "max_step": -1}
                    cur = agg[sid]
                cur["last"] = r  # JSONL is append-order, so the last seen is the latest
                step = r.get("step")
                if isinstance(step, int) and step > cur["max_step"]:
                    cur["max_step"] = step
    except FileNotFoundError:
        return {}
    return agg


def stale_zero_work_arms(
    path: Path | str | None = None,
    *,
    stale_minutes: float = 20.0,
    max_age_hours: float = 24.0,
    min_step: int = 0,
    now: _dt.datetime | None = None,
) -> list[dict]:
    """Return arms that registered but never did real work and then went silent (see module docstring).

    Each entry: ``{subagent_id, max_step, age_minutes, last_status, last_next_action}``.
    DETECT-ONLY — pure read."""
    p = Path(path) if path else _PROGRESS
    now = now or _dt.datetime.now(_dt.UTC)
    out: list[dict] = []
    for sid, info in _latest_by_arm(p).items():
        last = info["last"]
        if last.get("status") != "in_progress":
            continue
        if info["max_step"] > min_step:
            continue
        wa = _parse_utc(last.get("written_at_utc", ""))
        if wa is None:
            continue
        age_min = (now - wa).total_seconds() / 60.0
        if age_min < stale_minutes or age_min > max_age_hours * 60.0:
            continue
        out.append({
            "subagent_id": sid,
            "max_step": info["max_step"],
            "age_minutes": round(age_min, 1),
            "last_status": last.get("status"),
            "last_next_action": (last.get("next_action") or "")[:80],
        })
    out.sort(key=lambda r: r["age_minutes"])
    return out


def digest_line(stale_minutes: float = 20.0, max_age_hours: float = 24.0) -> tuple[str | None, list[dict]]:
    """One-line digest for costate_digest / witness_checkin. None when no zero-work arms."""
    arms = stale_zero_work_arms(stale_minutes=stale_minutes, max_age_hours=max_age_hours)
    if not arms:
        return None, []
    ids = ", ".join(f"{a['subagent_id']}(+{a['age_minutes']:.0f}m,step{a['max_step']})" for a in arms[:4])
    more = f" +{len(arms) - 4} more" if len(arms) > 4 else ""
    return (f"zero-work arms (registered, never advanced, silent {stale_minutes:.0f}m–"
            f"{max_age_hours:.0f}h): {len(arms)} — {ids}{more}"), arms


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Detect zero-work subagent arms (detect-only).")
    ap.add_argument("--stale-minutes", type=float, default=20.0)
    ap.add_argument("--max-age-hours", type=float, default=24.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    line, arms = digest_line(a.stale_minutes, a.max_age_hours)
    if a.json:
        print(json.dumps(arms, indent=2))
    else:
        print(line or "zero-work arms: none")
