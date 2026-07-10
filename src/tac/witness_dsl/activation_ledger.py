"""Lever ACTIVATION ledger — the "off is a tracked queue, never a forgotten default" apparatus.

Per the CLAUDE.md NON-NEGOTIABLE "'Off' is a tracked queue" (operator 2026-07-06 "Default off is a
source of orphaned signal ... never orphaned or rediscovery or counting on me to remember"): a
score-affecting lever may default OFF (safety) ONLY if "off" is a TRACKED queue state the controller
drains — never a silent default that relies on operator memory.

``lever_registry`` already answers COVERAGE ("is this trainer flag held by a DSL ``Lever`` factory?").
This module adds the missing ACTIVATION dimension: "has this DSL lever ever FIRED (been launched as an
A/B arm) and been MEASURED?" A lever that is held-by-the-DSL yet NEVER-FIRED is still orphaned signal
until this ledger tracks it.

State machine per lever:  never-fired  ->fired->  fired-unmeasured  ->measured->  measured
                          (and ->retired, with a recorded reason, from any state; terminal).

The ledger is a canonical, APPEND-ONLY, fcntl-locked JSONL at ``.omx/state/lever_activation_ledger.jsonl``
(mirrors the other ``.omx/state/*.jsonl`` canonical stores). It is POPULATED by REAL events only
(NO-FAKE): a "fired" event is written by ``tools/launch_witness_run.py`` when a run launches with
``--dsl-lever NAME`` (the canonical DSL-launcher path), a "measured" event when that run's byte-closed
verdict lands. It starts EMPTY — so ``never_fired()`` honestly returns EVERY DSL lever until one is
actually launched through the DSL path. (A raw-flag launch that bypasses the DSL launcher is NOT
recorded — which is itself the config-orphan the DSL exists to extinct; such launches SHOULD go through
the DSL so the campaign can track them.)

The #247 costate SENSE layer consumes ``duty_to_measure()`` to rank never-fired high-value levers into
its DECIDE queue: the CONTROLLER remembers and surfaces; the operator never has to.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tac.jsonl_store import append_locked_jsonl
from tac.witness_dsl.lever_registry import lever_factories

_REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_PATH = _REPO_ROOT / ".omx" / "state" / "lever_activation_ledger.jsonl"

# ── RELATIVE-SIGNIFICANCE store (the missing ΔS value axis) ──────────────────────────────────────
# Per re-audit `.omx/research/relative_significance_reaudit_20260708.md` + CLAUDE.md
# "Results must become system intelligence" + the recurring "relative-not-absolute-significance-near-
# goal-dont-orphan-small-deltaS" lesson: a lever carries NO ΔS estimate anywhere, so the duty-to-measure
# queue falls back to state-then-alphabetical ordering, which lets the operator's eyeball anchor on
# ABSOLUTE ΔS and orphan a small-but-near-goal-significant lever. This store is that missing field.
# Canonical, APPEND-ONLY, fcntl-locked JSONL; latest-row-wins per lever (mirrors the .omx/state pattern).
SIGNIFICANCE_PATH = _REPO_ROOT / ".omx" / "state" / "lever_relative_significance.jsonl"
_POINTER_PATH = _REPO_ROOT / ".omx" / "state" / "canonical_frontier_pointer.json"

# THE GOAL (CLAUDE.md §THE GOAL — SUB-0.15). s_target is parameterized everywhere; this is only the
# default the goal ladder pins. relative significance = est_delta_s / (s_current − s_target) = the
# fraction of the REMAINING descent to sub-0.15 a lever buys (pointer-anchored, NOT hardcoded s_current).
S_TARGET_DEFAULT = 0.15
# operator trigger-framing denominator (re-audit): frontier seg-term d_seg 0.00056; a competitive
# witness needs ~0.0009-class d_seg → Δd_seg/target_d_seg is the readable d_seg-axis fraction.
TARGET_D_SEG = 0.0009
# S is LINEAR in d_seg with coefficient 100 (S = 100·d_seg + √(10·d_pose) + 25·bytes/37_545_489), so a
# d_seg-axis ΔS maps to Δd_seg = ΔS / 100.
_S_PER_DSEG = 100.0

SIG_LABEL_MEASURED = "MEASURED"
SIG_LABEL_ESTIMATED = "ESTIMATED"
SIG_LABEL_UNMEASURED = "UNMEASURED"  # a registered duty-to-ESTIMATE row (an un-estimated lever is itself orphaned signal)
VALID_SIG_LABELS = frozenset({SIG_LABEL_MEASURED, SIG_LABEL_ESTIMATED, SIG_LABEL_UNMEASURED})
VALID_SIG_AXES = frozenset({"d_seg", "d_pose", "rate"})

# ── SIGNIFICANCE-KEY CANONICALIZATION (the built-but-mislabeled-unbuilt fix; #377 build-wave) ──────
# A significance-store row is keyed by a lever NAME. When a finding is FIRST recorded (before it becomes a
# held DSL ``Lever`` factory) it is keyed by a human/task-# name (e.g. ``d_seg_aware_taper_121``). Once the
# lever is BUILT + HELD, its canonical name is the ``lever_registry`` factory name (e.g. ``DsegAwareTaper``).
# If the legacy significance key is never reconciled, ``duty_to_measure_ranked`` computes
# ``registered = (key in factory_names)`` == False and the digest FALSELY marks a built+held+wired lever
# ``~=unbuilt`` (a duty-to-BUILD) instead of ``*=never-fired`` (a duty-to-MEASURE) — orphaned signal per
# CLAUDE.md "'Off' is a tracked queue" + "Results must become system intelligence". This map reconciles
# legacy significance keys onto the canonical factory name at READ time (the APPEND-ONLY store is NOT
# rewritten; history is preserved). An alias is applied ONLY when its TARGET is a real held factory — so if
# the factory is later renamed/removed the row correctly reverts to a build gap. VERIFIED (source
# inspection 2026-07-09): ``DsegAwareTaper`` (curriculum_dsl.py:1944, #121) + ``HorizonWeightedMargin``
# (curriculum_dsl.py:3192, #169) are both held factories with all flags mapped (completeness) + wired into
# the levelset trainer argparse + loss (byte-identical default-OFF) + 58 passing tests. They are owed a
# MEASUREMENT, not a build. ``latent_table_truncate_d18_k90`` is intentionally NOT aliased — it is a
# byte-close-tool lever (not a DSL factory), correctly still a finding (see deferral ledger D18).
_SIGNIFICANCE_LEVER_ALIASES: dict[str, str] = {
    "d_seg_aware_taper_121": "DsegAwareTaper",
    "horizon_weighted_margin_169": "HorizonWeightedMargin",
}

# canonical event vocabulary
EVENT_FIRED = "fired"        # a run launched with this lever (an A/B arm was actually run)
EVENT_MEASURED = "measured"  # a byte-closed verdict landed for a run using this lever
EVENT_RETIRED = "retired"    # the lever was retired with a recorded reason (dormant-with-reactivation)
VALID_EVENTS = frozenset({EVENT_FIRED, EVENT_MEASURED, EVENT_RETIRED})

# canonical states (latest-event-wins per lever, with fired/measured monotonic)
STATE_NEVER_FIRED = "never-fired"
STATE_FIRED_UNMEASURED = "fired-unmeasured"
STATE_MEASURED = "measured"
STATE_RETIRED = "retired"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_activation(
    lever: str,
    event: str,
    *,
    run_ref: str | None = None,
    verdict_ref: str | None = None,
    reason: str = "",
    agent: str | None = None,
    path: Path | None = None,
) -> dict:
    """Append ONE activation event (fcntl-locked, APPEND-ONLY). Returns the written row.

    ``event`` must be one of :data:`VALID_EVENTS`. ``run_ref`` is the run out-dir / id; ``verdict_ref``
    the byte-closed verdict artifact (required-ish for a "measured" event to be trustworthy, but not
    hard-enforced here — the reader flags a measured-without-verdict as weak). NEVER a score claim.
    """
    if event not in VALID_EVENTS:
        raise ValueError(f"invalid activation event {event!r}; must be one of {sorted(VALID_EVENTS)}")
    if not lever or not isinstance(lever, str):
        raise ValueError(f"lever must be a non-empty str, got {lever!r}")
    row = {
        "lever": lever,
        "event": event,
        "run_ref": run_ref,
        "verdict_ref": verdict_ref,
        "reason": reason,
        "agent": agent,
        "ts": _utc(),
    }
    p = Path(path) if path is not None else LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    # fcntl-locked append (canonical .omx/state pattern). Best-effort on platforms without fcntl.
    try:
        import fcntl
        with open(p, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - non-POSIX fallback
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
    return row


def _read_events(path: Path | None = None) -> list[dict]:
    """Read all events, lenient (skips corrupt lines rather than failing the whole read)."""
    p = Path(path) if path is not None else LEDGER_PATH
    if not p.exists():
        return []
    out: list[dict] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("lever") and row.get("event") in VALID_EVENTS:
            out.append(row)
    return out


@dataclass(frozen=True)
class ActivationStatus:
    lever: str
    ever_fired: bool
    ever_measured: bool
    retired: bool
    state: str
    last_event: str | None
    last_ts: str | None
    n_fired: int
    n_measured: int


def activation_status(lever: str, path: Path | None = None) -> ActivationStatus:
    """The activation state of ONE lever (from its recorded events)."""
    evs = [e for e in _read_events(path) if e.get("lever") == lever]
    fired = [e for e in evs if e["event"] == EVENT_FIRED]
    measured = [e for e in evs if e["event"] == EVENT_MEASURED]
    retired = any(e["event"] == EVENT_RETIRED for e in evs)
    last = evs[-1] if evs else None
    # retired is terminal; else measured > fired > never.
    if retired:
        state = STATE_RETIRED
    elif measured:
        state = STATE_MEASURED
    elif fired:
        state = STATE_FIRED_UNMEASURED
    else:
        state = STATE_NEVER_FIRED
    return ActivationStatus(
        lever=lever,
        ever_fired=bool(fired),
        ever_measured=bool(measured),
        retired=retired,
        state=state,
        last_event=last["event"] if last else None,
        last_ts=last.get("ts") if last else None,
        n_fired=len(fired),
        n_measured=len(measured),
    )


def known_levers() -> tuple[str, ...]:
    """The canonical set of DSL ``Lever`` factory names (from ``lever_registry.lever_factories``)."""
    return tuple(sorted(lever_factories().keys()))


def never_fired(known: tuple[str, ...] | None = None, path: Path | None = None) -> tuple[str, ...]:
    """DSL levers that have NEVER fired (no ``fired`` event) and are not retired — the orphan surface.

    An empty ledger honestly returns every DSL lever: none has been launched through the DSL path yet.
    """
    names = known if known is not None else known_levers()
    out = []
    for name in names:
        st = activation_status(name, path)
        if not st.ever_fired and not st.retired:
            out.append(name)
    return tuple(out)


def duty_to_measure(known: tuple[str, ...] | None = None, path: Path | None = None) -> tuple[str, ...]:
    """Levers OWED a measurement: never-fired OR fired-but-never-measured (not retired). The costate
    SENSE ranks these into its DECIDE queue (the duty-to-measure the discipline mandates)."""
    names = known if known is not None else known_levers()
    out = []
    for name in names:
        st = activation_status(name, path)
        if st.retired:
            continue
        if not st.ever_fired or not st.ever_measured:
            out.append(name)
    return tuple(out)


def _run_ref_matches(fired_ref: str | None, query_ref: str) -> bool:
    """A fired ``run_ref`` (the launch out-dir) matches a CLOSE ``query_ref`` (a byte-close ckpt-dir)
    when the two run dirs are the same or one contains the other (ckpt-dir is often ``<out_dir>`` or a
    subdir of it). Path-component containment, not a raw substring, so ``.../run_2`` never matches
    ``.../run_20``."""
    if not fired_ref:
        return False
    a = os.path.normpath(os.path.abspath(str(fired_ref)))
    b = os.path.normpath(os.path.abspath(str(query_ref)))
    if a == b:
        return True
    sep = os.sep
    return a.startswith(b + sep) or b.startswith(a + sep)


def levers_fired_for_run(run_ref: str, path: Path | None = None) -> tuple[str, ...]:
    """DSL levers that recorded a ``fired`` event for this run (matched by run-dir containment).
    The CLOSE side reads these back so it records ``measured`` ONLY for levers that genuinely fired
    for the run (NO-FAKE: never a lever the run did not use)."""
    fired: list[str] = []
    seen: set[str] = set()
    for e in _read_events(path):
        if e["event"] == EVENT_FIRED and e["lever"] not in seen and _run_ref_matches(e.get("run_ref"), run_ref):
            fired.append(e["lever"])
            seen.add(e["lever"])
    return tuple(fired)


def record_measured_for_run(
    run_ref: str,
    *,
    verdict_ref: str | None = None,
    reason: str = "",
    agent: str | None = None,
    path: Path | None = None,
) -> tuple[dict, ...]:
    """CLOSE the loop: record a ``measured`` event for every lever that FIRED for ``run_ref`` and is
    not yet measured (draining it from :func:`duty_to_measure`). Returns the rows written (empty if no
    fired-unmeasured lever matched). Idempotent-ish: a lever already measured for this run is skipped."""
    # (review-fix HIGH) the skip must be PER-RUN, not global. The prior `st.ever_measured` is the
    # all-runs status, so a lever measured in run A would silently drop its GENUINE, distinct
    # measurement in run B (a later A/B arm) — a real event lost + duty_to_measure permanently
    # considering it satisfied. Filter by THIS run's own `measured` events (retired stays terminal-global).
    rows: list[dict] = []
    measured_this_run: set[str] = {
        e["lever"] for e in _read_events(path)
        if e["event"] == EVENT_MEASURED and _run_ref_matches(e.get("run_ref"), run_ref)
    }
    for lever in levers_fired_for_run(run_ref, path):
        st = activation_status(lever, path)
        if lever in measured_this_run or st.retired:
            continue
        rows.append(record_activation(
            lever, EVENT_MEASURED, run_ref=run_ref, verdict_ref=verdict_ref,
            reason=reason, agent=agent, path=path,
        ))
    return tuple(rows)


def activation_report(known: tuple[str, ...] | None = None, path: Path | None = None) -> list[dict]:
    """The operator-facing 'what is registered but never fired?' surface — one row per DSL lever
    (default OFF by construction for score-affecting levers), with its activation state + reason-to-be
    (that the controller, not the operator, holds the queue)."""
    names = known if known is not None else known_levers()
    rows = []
    for name in names:
        st = activation_status(name, path)
        rows.append({
            "lever": name,
            "default": "off",  # score-affecting levers default off by construction (safety)
            "state": st.state,
            "ever_fired": st.ever_fired,
            "ever_measured": st.ever_measured,
            "n_fired": st.n_fired,
            "n_measured": st.n_measured,
            "last_event": st.last_event,
            "last_ts": st.last_ts,
        })
    # surface never-fired / fired-unmeasured first (the queue the controller drains)
    _order = {STATE_NEVER_FIRED: 0, STATE_FIRED_UNMEASURED: 1, STATE_MEASURED: 2, STATE_RETIRED: 3}
    rows.sort(key=lambda r: (_order.get(r["state"], 9), r["lever"]))
    return rows


# ── RELATIVE-SIGNIFICANCE: store + metric + value-ranked duty-to-measure ─────────────────────────
# _append_locked_jsonl canonicalized to tac.jsonl_store.append_locked_jsonl (audit finding #4,
# .omx/research/hardcode_duplication_audit_witness_stack_20260710.md) — was byte-identical to the
# copy in curriculum_candidate_pool.py; now a single shared helper.


def record_relative_significance(
    lever: str,
    est_delta_s: float | None,
    *,
    label: str,
    source_anchor: str,
    axis: str,
    notes: str = "",
    agent: str | None = None,
    path: Path | None = None,
) -> dict:
    """Append ONE relative-significance row (APPEND-ONLY, fcntl-locked). Returns the written row.

    ``est_delta_s`` is the POSITIVE magnitude of the ΔS the lever buys (may be ``None`` for a registered
    duty-to-ESTIMATE marker). ``label`` ∈ {MEASURED, ESTIMATED, UNMEASURED}; ``axis`` ∈ {d_seg, d_pose,
    rate}; ``source_anchor`` is the commit/memo/DAG-FEED that measured/estimated it (NO-FAKE: never a
    guessed number without a source). Latest row wins per lever on read.
    """
    if not lever or not isinstance(lever, str):
        raise ValueError(f"lever must be a non-empty str, got {lever!r}")
    if label not in VALID_SIG_LABELS:
        raise ValueError(f"invalid label {label!r}; must be one of {sorted(VALID_SIG_LABELS)}")
    if axis not in VALID_SIG_AXES:
        raise ValueError(f"invalid axis {axis!r}; must be one of {sorted(VALID_SIG_AXES)}")
    if est_delta_s is not None:
        est_delta_s = float(est_delta_s)
        if est_delta_s < 0:
            raise ValueError(f"est_delta_s must be a positive ΔS magnitude or None, got {est_delta_s!r}")
    if label != SIG_LABEL_UNMEASURED and est_delta_s is None:
        raise ValueError(f"label {label!r} requires a numeric est_delta_s (only UNMEASURED may be None)")
    if not source_anchor:
        raise ValueError("source_anchor is required (NO-FAKE: every ΔS row cites its measurement/estimate)")
    row = {
        "lever": lever,
        "est_delta_s": est_delta_s,
        "delta_s_label": label,
        "source_anchor": source_anchor,
        "axis": axis,
        "notes": notes,
        "agent": agent,
        "ts": _utc(),
    }
    append_locked_jsonl(Path(path) if path is not None else SIGNIFICANCE_PATH, row)
    return row


def _read_significance(path: Path | None = None) -> dict[str, dict]:
    """Latest-row-wins map ``lever -> row`` from the significance store (lenient; skips corrupt lines)."""
    p = Path(path) if path is not None else SIGNIFICANCE_PATH
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("lever") and row.get("delta_s_label") in VALID_SIG_LABELS:
            out[row["lever"]] = row  # later row overwrites earlier -> latest-wins
    return out


def canonicalize_significance_keys(
    sig: dict[str, dict], factory_names: set[str], aliases: dict[str, str] | None = None
) -> dict[str, dict]:
    """Re-key legacy significance rows onto their held DSL-factory name (read-time; store unchanged).

    For each ``legacy -> canonical`` in :data:`_SIGNIFICANCE_LEVER_ALIASES`: if ``legacy`` has a row, the
    ``canonical`` name IS a real held factory, and no explicit ``canonical`` row already exists (latest-wins
    is preserved for an explicit canonical row), move the legacy row onto ``canonical`` (stamping
    ``_alias_from`` provenance). Idempotent; pure; returns a NEW dict (input not mutated). This is what
    flips a built+held+wired lever from ``~=unbuilt`` (duty-to-BUILD) to ``*=never-fired`` (duty-to-MEASURE)
    in the digest.
    """
    amap = _SIGNIFICANCE_LEVER_ALIASES if aliases is None else aliases
    out = dict(sig)
    for legacy, canonical in amap.items():
        if legacy in out and canonical in factory_names and canonical not in out:
            row = dict(out.pop(legacy))
            row["lever"] = canonical
            row["_alias_from"] = legacy
            out[canonical] = row
    return out


def read_pointer_s(path: Path | None = None) -> float | None:
    """Read the LIVE exact contest-CPU frontier score from the canonical pointer (NEVER hardcoded).

    Returns ``our_local_frontier_contest_cpu.score`` or ``None`` if the pointer is unreadable.
    """
    p = Path(path) if path is not None else _POINTER_PATH
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return float(d["our_local_frontier_contest_cpu"]["score"])
    except Exception:
        return None


def relative_significance(
    est_delta_s: float | None, s_current: float | None, s_target: float = S_TARGET_DEFAULT
) -> float | None:
    """rel-sig(S) = est_delta_s / (s_current − s_target) = fraction of the REMAINING descent to the goal.

    Returns ``None`` when it cannot be computed (no estimate, no pointer, or no remaining gap — at/below
    the goal the "fraction of remaining descent" is undefined). Higher = more urgent. The key property
    (the recurring-bug correction): for a FIXED est_delta_s, rel-sig RISES as ``s_current`` approaches
    ``s_target`` — a small absolute ΔS becomes MORE significant near the goal, not negligible.
    """
    if est_delta_s is None or s_current is None:
        return None
    gap = float(s_current) - float(s_target)
    if gap <= 0:
        return None
    return float(est_delta_s) / gap


def duty_to_measure_ranked(
    s_current: float | None = None,
    s_target: float = S_TARGET_DEFAULT,
    *,
    known: tuple[str, ...] | None = None,
    path: Path | None = None,
    sig_path: Path | None = None,
    pointer_path: Path | None = None,
    term_current: "dict[str, float] | None" = None,
    term_floors: "dict[str, object] | None" = None,
    floor_aware: bool = True,
) -> list[dict]:
    """The duty-to-measure queue RANKED by relative significance (fraction of remaining descent).

    Joins :func:`duty_to_measure` (registered levers owed a measurement) with the relative-significance
    store, computes ``rel_sig = est_delta_s / (s_current − s_target)`` (reading ``s_current`` from the
    canonical pointer when not passed — NEVER hardcoded), and returns rows sorted by ``rel_sig``
    DESCENDING. Store findings that are not (yet) registered levers are INCLUDED (an orphaned lever is
    often a *missing wire* — a duty-to-BUILD); registered owed levers with no ΔS row are surfaced as a
    duty-to-ESTIMATE queue (an un-estimated lever is itself orphaned signal). Ties / unknowns break by
    est_delta_s then name — the eyeball is removed from the loop.

    **FLOOR-AWARE (P8 FLOOR-FIRST; design_philosophies_eightfold_20260709).** When ``floor_aware``
    (default), each row consults its target TERM's MEASURED floor (``term_floors`` overrides
    :func:`tac.witness_dsl.term_floors.resolve_term_floors`) with value-provenance: a lever whose target
    term is AT its floor ranks ~0 (``floor_status=AT_FLOOR``, rel_sig 0); a lever whose est exceeds the
    S-headroom to its floor is CAPPED to that headroom. This ONLY changes rel_sig when a measured
    ``term_current`` value (axis -> current term value, e.g. the live run's d_seg) makes the at-floor
    judgement possible — so with no ``term_current`` (the default) rel_sig + ordering are UNCHANGED and
    every row simply gains surfaced floor metadata (``floor_status=FLOOR_UNMEASURED`` /
    ``FLOOR_KNOWN_CURRENT_UNKNOWN``). NO-FAKE: only a clean numeric MEASURED floor caps; regime-dependent
    / LOOSE / owed floors are surfaced but never change the ranking (never a guessed floor).
    """
    if s_current is None:
        s_current = read_pointer_s(pointer_path)
    owed = set(duty_to_measure(known, path))
    known_set = set(known) if known is not None else set(known_levers())
    # reconcile legacy significance keys onto their now-held factory names so a built+held+wired lever
    # is correctly surfaced as duty-to-MEASURE (never-fired) rather than duty-to-BUILD (unregistered).
    sig = canonicalize_significance_keys(_read_significance(sig_path), known_set)

    floors: dict[str, object] = {}
    if floor_aware:
        if term_floors is not None:
            floors = dict(term_floors)
        else:
            try:
                from tac.witness_dsl.term_floors import resolve_term_floors
                floors = resolve_term_floors()
            except Exception:
                floors = {}

    rows: list[dict] = []
    for lever in sorted(owed | set(sig.keys())):
        srow = sig.get(lever)
        est = srow.get("est_delta_s") if srow else None
        axis = srow.get("axis") if srow else None
        registered = lever in known_set
        # activation state only meaningful for registered levers; findings not-yet-a-lever = a build gap.
        state = activation_status(lever, path).state if registered else "not-registered"

        # FLOOR-FIRST consult: cap the achievable ΔS by the term's headroom-to-floor (P8). eff_est == est
        # whenever no measured floor + measured current value fire, so the default path is unchanged.
        eff_est = est
        floor_status = None
        floor_meta: dict | None = None
        headroom_s = None
        if floor_aware:
            floor = floors.get(axis) if axis else None
            cur = term_current.get(axis) if (term_current and axis) else None
            try:
                from tac.witness_dsl.term_floors import apply_term_floor
                app = apply_term_floor(axis, est, cur, floor)
                floor_status = app.floor_status
                headroom_s = app.headroom_s
                eff_est = app.capped_est if app.capped_est is not None else est
                floor_meta = app.floor.to_dict() if app.floor is not None else None
            except Exception:  # fail-open: floor machinery must NEVER break the ranking
                floor_status, floor_meta, headroom_s, eff_est = None, None, None, est

        rel = relative_significance(eff_est, s_current, s_target)
        rel_dseg = None
        if eff_est is not None and axis == "d_seg":
            rel_dseg = (float(eff_est) / _S_PER_DSEG) / TARGET_D_SEG
        rows.append({
            "lever": lever,
            "est_delta_s": est,                 # the RAW estimate (provenance preserved)
            "est_floor_capped": eff_est,        # est after floor cap (== est when no cap)
            "delta_s_label": (srow.get("delta_s_label") if srow else None),
            "source_anchor": (srow.get("source_anchor") if srow else None),
            "axis": axis,
            "notes": (srow.get("notes") if srow else ""),
            "rel_sig": rel,
            "rel_sig_pct": (round(100.0 * rel, 1) if rel is not None else None),
            "rel_sig_dseg": rel_dseg,
            "in_duty_queue": lever in owed,
            "registered": registered,
            "activation_state": state,
            "floor_status": floor_status,
            "floor": floor_meta,
            "floor_headroom_s": headroom_s,
            "s_current": s_current,
            "s_target": s_target,
            "gap": (None if s_current is None else float(s_current) - float(s_target)),
        })

    # rank: rel_sig desc (None last) -> est_delta_s desc (None last) -> name asc.
    def _key(r: dict) -> tuple:
        rel = r["rel_sig"]
        est = r["est_delta_s"]
        return (
            0 if rel is not None else 1, -(rel or 0.0),
            0 if est is not None else 1, -(est or 0.0),
            r["lever"],
        )

    rows.sort(key=_key)
    return rows


__all__ = [
    "EVENT_FIRED",
    "EVENT_MEASURED",
    "EVENT_RETIRED",
    "LEDGER_PATH",
    "SIGNIFICANCE_PATH",
    "SIG_LABEL_ESTIMATED",
    "SIG_LABEL_MEASURED",
    "SIG_LABEL_UNMEASURED",
    "STATE_FIRED_UNMEASURED",
    "STATE_MEASURED",
    "STATE_NEVER_FIRED",
    "STATE_RETIRED",
    "S_TARGET_DEFAULT",
    "TARGET_D_SEG",
    "VALID_EVENTS",
    "VALID_SIG_AXES",
    "VALID_SIG_LABELS",
    "ActivationStatus",
    "activation_report",
    "activation_status",
    "duty_to_measure",
    "duty_to_measure_ranked",
    "known_levers",
    "levers_fired_for_run",
    "never_fired",
    "read_pointer_s",
    "record_activation",
    "record_measured_for_run",
    "record_relative_significance",
    "relative_significance",
]
