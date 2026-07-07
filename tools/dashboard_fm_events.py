#!/usr/bin/env python3
"""dashboard_fm_events — on-device FM classification of run-log EVENTS (the fmtools layer).

The trainer's run.log carries occasional NOTABLE one-shot events between the
high-frequency telemetry (structured_init, muon_finisher_switch, palette_anchor,
spike/guard alarms, stage-transition checkpoints, ...). This module classifies each
one ON-DEVICE (Apple Foundation Models via the operator's fmtools toolkit,
~/Projects/fmtools) into {category, one-line summary, known-failure-class match} and
the dashboard renders them as a conditional EVENT FEED panel.

ISOLATION (the established detached-subprocess pattern — oracle/whyhow/flowseq):
FM inference NEVER runs inside the dashboard daemon. The server spawns THIS module
as a detached governed subprocess under the interpreter named by ``DASH_FM_PYTHON``
(a venv that has fmtools + apple_fm_sdk, e.g. ~/Projects/fmtools/.venv/bin/python).
The pact venv gains ZERO new dependencies. fmtools absent / model unavailable /
DASH_FM_PYTHON unset -> the feature is ABSENT (conditional rendering), never a
value-shaped stub.

HONESTY: classifications are LABELLED provenance ("on-device FM · advisory") — the
raw event line rides each row's tooltip; the FM never invents numbers (the summary
restates the event; category is a closed enum). Failure-class matching is against
the KNOWN class ids from the harness failure ledger (the costate digest's glob
mechanism) — "none" when no known class matches. Score-neutral, read-only.

CLI (run under the fmtools venv):
  <fm-python> tools/dashboard_fm_events.py --log <run.log> --out <stem>.json \
      --done <stem>.done [--cache <cache.json>] [--limit 12] [--fake]

``--fake`` = deterministic stub classifier (no fmtools import) for tests + dev.
"""
from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

#: high-frequency telemetry stages that are NOT events (charted/carded elsewhere).
HIGH_FREQ_STAGES = frozenset({
    "loss_terms", "mem_probe", "verdict", "verdict_async_done", "reorient",
    "closed_loop", "annulus", "annulus_live",
})

#: checkpoint lines are high-frequency EXCEPT the milestone kinds.
NOTABLE_CHECKPOINT_KINDS = frozenset({"stage_transition", "final"})


def extract_notable_events(lines, limit: int = 12) -> list[dict]:
    """The last ``limit`` NOTABLE one-shot ``{"stage": ...}`` events from run-log lines
    (pure; tolerant of malformed lines). High-frequency telemetry stages are excluded;
    checkpoint lines count only at milestone kinds (stage_transition / final)."""
    out: list[dict] = []
    for raw in lines:
        s = raw.strip()
        if '"stage"' not in s:
            continue
        try:
            d = json.loads(s)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        st = d.get("stage")
        if not isinstance(st, str) or st in HIGH_FREQ_STAGES:
            continue
        if st == "checkpoint" and d.get("kind") not in NOTABLE_CHECKPOINT_KINDS:
            continue
        out.append({"stage": st, "epoch": d.get("epoch"), "line": s})
    return out[-int(limit):]


def event_key(ev: dict) -> str:
    """Stable cache key for one event (sha256 of the raw line)."""
    return hashlib.sha256(ev["line"].encode("utf-8", "replace")).hexdigest()


def events_digest(events: list[dict]) -> str:
    """One digest over the whole notable-event window (the server's spawn key)."""
    h = hashlib.sha256()
    for ev in events:
        h.update(event_key(ev).encode())
    return h.hexdigest()[:16]


def known_failure_classes(repo: Path | None = None) -> list[str]:
    """Known failure-class ids from the harness failure ledger(s) — the same glob
    mechanism tools/costate_digest.py uses (REUSED, not reinvented). Fail-open []."""
    try:
        hits = sorted(_glob.glob(str((repo or _REPO) / ".omx" / "state" / "*failure*ledger*.jsonl")))
        ids: set[str] = set()
        for p in hits:
            for ln in Path(p).read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                if isinstance(row, dict) and row.get("failure_id"):
                    ids.add(str(row["failure_id"]))
        return sorted(ids)
    except Exception:
        return []


# ─────────────────────────── cache (tiny json; bounded) ───────────────────────────
def load_cache(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(errors="replace"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_cache(path: Path, cache: dict, max_entries: int = 400) -> None:
    """Atomic write; oldest-first eviction keeps the cache bounded."""
    try:
        if len(cache) > max_entries:
            items = sorted(cache.items(), key=lambda kv: kv[1].get("ts", 0.0))
            cache = dict(items[-max_entries:])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, path)
    except Exception:
        pass  # a cache write must never fail the classification run


# ─────────────────────────── classifiers ───────────────────────────
def fake_classify(events: list[dict], failure_ids: list[str]) -> list[dict]:
    """Deterministic stub (no fmtools import) for tests + dev loops. LABELLED fake."""
    out = []
    for ev in events:
        st = ev["stage"]
        cat = ("failure" if any(t in st for t in ("oom", "error", "alarm", "refused"))
               else "warning" if any(t in st for t in ("guard", "spike", "stale"))
               else "milestone" if any(t in st for t in ("switch", "transition", "final",
                                                         "structured_init", "gt"))
               else "info")
        out.append({**{k: ev[k] for k in ("stage", "epoch", "line")},
                    "category": cat, "summary": f"{st} event (fake classifier)",
                    "failure_class": "none", "classifier": "fake"})
    return out


def fm_classify(events: list[dict], failure_ids: list[str]) -> list[dict]:
    """On-device Apple FM classification via fmtools @local_extract (structured
    generation against a closed schema). Raises on setup/model unavailability —
    the CLI turns that into an honest {"ok": False} payload."""
    import asyncio

    import apple_fm_sdk as fm
    from fmtools import local_extract

    @fm.generable()
    class RunEvent:
        category: str = fm.guide(anyOf=["milestone", "info", "warning", "failure"])
        summary: str = fm.guide(
            description="One short plain sentence restating what this event says. "
                        "Use only facts present in the event; never invent numbers.")
        failure_class: str = fm.guide(
            description="The EXACT known failure-class id this event matches, "
                        "or 'none' if it matches no known class.")

    known = ", ".join(failure_ids) if failure_ids else "(none registered)"
    # known classes ride the INSTRUCTIONS, never the input text: injected as an input
    # kwarg they leaked into summaries ("dashboard reported two failure messages" —
    # confabulated from the class-id list; measured on the first live batch).
    _instructions = (
        "You label ONE telemetry event from a long machine-learning training run. "
        "The event describes something that HAPPENED AS DESIGNED unless its own text "
        "says an error occurred. Rules: category is 'failure' ONLY if the event text "
        "itself reports an error/OOM/crash/kill/refusal; 'warning' for guard, spike, "
        "alarm, stale or clip signals; 'milestone' for stage or optimizer transitions, "
        "initialization, and checkpoints being written; 'info' otherwise. summary: one "
        "short plain sentence restating ONLY what the event text says — a checkpoint "
        "being written is a save, never a failure; never invent outcomes or numbers. "
        f"failure_class: exactly one id from this list [{known}] ONLY if the event "
        "text clearly matches it, else 'none'.")

    @local_extract(RunEvent, retries=2, instructions=_instructions)
    async def _classify(event_text: str) -> RunEvent:
        """(instructions provided explicitly above)"""

    def _prose(ev: dict) -> str:
        """Prose framing: mostly-numeric raw JSON can trip the FM language guardrail
        ('Unsupported language or locale'); a k-is-v English sentence does not."""
        try:
            d = json.loads(ev["line"])
            body = "; ".join(f"{k} is {v}" for k, v in list(d.items())[:12])
        except Exception:
            body = ev["line"][:400]
        return f"The training run logged this event: {body}"

    async def _run() -> list[dict]:
        out = []
        for ev in events:
            base = {k: ev[k] for k in ("stage", "epoch", "line")}
            try:
                r = await _classify(_prose(ev))
            except Exception as exc:
                # PER-EVENT fault tolerance: one guardrail-tripping line (e.g. a dense
                # numeric dump) must never fail the whole batch. Honest non-value row.
                out.append({**base, "category": "info",
                            "summary": f"unclassified ({type(exc).__name__}: FM guardrail/setup)",
                            "failure_class": "none", "classifier": "apple-fm-skip"})
                continue
            fc = str(getattr(r, "failure_class", "none") or "none")
            if fc != "none" and fc not in failure_ids:
                fc = "none"  # the model may not invent a class id
            out.append({**base,
                        "category": str(getattr(r, "category", "info")),
                        "summary": str(getattr(r, "summary", ""))[:220],
                        "failure_class": fc, "classifier": "apple-fm-on-device"})
        return out

    return asyncio.run(_run())


# ─────────────────────────── CLI ───────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, help="run log to extract notable events from")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--done", required=True, help="done-marker JSON path (written LAST)")
    ap.add_argument("--cache", default=str(_REPO / ".omx" / "tmp" / "dash_fm_events" / "cache.json"))
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--fake", action="store_true",
                    help="deterministic stub classifier (no fmtools; tests/dev)")
    a = ap.parse_args(argv)

    t0 = time.time()
    try:
        lines = Path(a.log).read_text(errors="replace").splitlines()
    except Exception as exc:
        payload = {"ok": False, "reason": f"log unreadable: {exc}"}
        Path(a.out).write_text(json.dumps(payload))
        Path(a.done).write_text(json.dumps(payload))
        return 2
    events = extract_notable_events(lines, limit=a.limit)
    failure_ids = known_failure_classes()
    cache_path = Path(a.cache)
    cache = load_cache(cache_path)

    fresh = [ev for ev in events if event_key(ev) not in cache]
    classified_fresh: list[dict] = []
    reason = None
    if fresh:
        try:
            classified_fresh = (fake_classify if a.fake else fm_classify)(fresh, failure_ids)
        except Exception as exc:  # setup/model unavailability -> honest failure payload
            reason = f"{type(exc).__name__}: {exc}"
    for row in classified_fresh:
        cache[event_key(row)] = {**row, "ts": time.time()}
    if classified_fresh:
        save_cache(cache_path, cache)

    # ok == we can render SOMETHING classified (cached rows survive a fresh-batch
    # failure); ``reason`` carries any classification failure honestly alongside.
    rows = [cache[event_key(ev)] for ev in events if event_key(ev) in cache]
    payload = {
        "ok": bool(rows), "reason": reason,
        "events": [{k: r.get(k) for k in ("stage", "epoch", "line", "category",
                                          "summary", "failure_class", "classifier")}
                   for r in rows],
        "n_notable": len(events), "n_classified": len(rows),
        "known_failure_classes": len(failure_ids),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 2),
        "authority": "on-device FM · advisory · read-only · NON-PROMOTABLE",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload))
    Path(a.done).write_text(json.dumps({"ok": payload["ok"], "reason": reason,
                                        "built_at_utc": payload["built_at_utc"],
                                        "n_classified": len(rows)}))
    return 0 if payload["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
