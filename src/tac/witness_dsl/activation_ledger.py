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
    # fcntl-locked append via the canonical .omx/state helper (tac.jsonl_store); see
    # .omx/research/fcntl_lock_canonicalization_plan_20260710.md Batch 1.
    append_locked_jsonl(p, row)
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


def curriculum_dsl_known_levers() -> tuple[str, ...]:
    """ONLY the ``Lever`` factories declared in ``curriculum_dsl.py`` (``lever_registry.lever_factories``).

    A DELIBERATELY NARROW surface, named so a reader cannot mistake it for the campaign's lever
    universe. It exists for the one legitimate question "what does the levelset trainer's own DSL
    module hold?" — never for orphan/duty accounting, which must see the whole package
    (:func:`known_levers`). ``check_no_legacy_single_module_lever_surface_consumers``
    (``tac.confound_gates``) refuses NEW consumers of this surface outside its allowlist.
    """
    return tuple(sorted(lever_factories().keys()))


def known_levers() -> tuple[str, ...]:
    """The canonical set of DSL ``Lever`` factory names — EVERY module in ``tac.witness_dsl``.

    ddm_rg5 (task #825), 2026-07-31 — **the sense organ was blind to its own stubs.** This
    function used to return ``lever_factories()``, which ASTs ONE file (``curriculum_dsl.py``).
    ``never_fired()`` and ``duty_to_measure()`` BOTH default to it, and those two ARE the
    duty-to-measure queue CLAUDE.md's "'Off' is a tracked queue, never a forgotten default"
    non-negotiable mandates. ddm_sb2 repaired the registry by ADDING
    :func:`package_known_levers` beside it but PRESERVING this default, so nothing opted in: the
    honest superset had exactly one grep hit outside its own definition, and that hit was a
    docstring.

    MEASURED (ddm_rg5, n=live registry + ledger): legacy 116 factories vs package 177 — the
    legacy set is a strict SUBSET (0 extra), so **61 factories were structurally INELIGIBLE for
    the duty queue**, including **9 of the 10 DESIGNED-STUBS** (fh1 x5, ph3_s10 x2, ax1 x1,
    constants_telemetry x1). Those stubs are the exact NO-FAKE forbidden-class-#1 surface the
    same-day binding memory
    (``designed_stub_is_orphan_signal_and_a_no_fake_violation_20260731``) names — invisible to
    their own tracker on the day the rule was written.

    HONEST BOUND on what this repairs (ddm_rg5 ran the diff before claiming it): the RANKED HEAD
    of the queue is UNCHANGED. All 61 newly-visible factories carry no ``est_delta_s`` row, so
    they sort into the alphabetical duty-to-ESTIMATE tail below the 5 significance-carrying rows,
    and the digest prints only the top-N. What the blindness cost is COUNT (owed 115 -> 176),
    ELIGIBILITY (a lever absent from ``known_levers`` can never be nagged, estimated, or ranked
    at all), and the ORPHAN LIST ITSELF (:func:`never_fired`). It did NOT corrupt the ranking —
    any claim that it did is unsupported by the measured diff.

    Cost is not a reason to stay narrow: ``package_lever_factories`` is cached
    (MEASURED cold 1340 ms, warm 1.2 ms — a slow gate is a disabled gate, which is how the
    vacuity survived; ddm_sb2's lesson, honored).
    """
    return package_known_levers()


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
    term_current: dict[str, float] | None = None,
    term_floors: dict[str, object] | None = None,
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


# ══ BUILD COMPLETENESS — the dimension this ledger could not express ═══════════════════════════
# ddm_sb2 (task #819), operator 2026-07-31: *"Everything that is designed to stub ... needs to be
# fully built out. No orphan signal is a very important principle"* + *"Everything that we need for
# our truly optimal from start run or continued warm start and fresh tuning needs to be fully built."*
#
# THE BUG THIS CLOSES. The schema above is ``{default, ever_fired, last_verdict, state}``. It can
# distinguish "off but real" from "fired" — it CANNOT distinguish "off but real" from "off and
# HOLLOW". A DESIGNED-STUB (a ``Lever`` factory whose trainer flag does not exist) reports
# ``state=never-fired``, byte-identical to a fully-built default-off lever, to every consumer: the
# duty queue, the costate SENSE layer, launch tickets, council deliberations. That is NO-FAKE
# forbidden class #1 (marker-without-mechanism) at the registry layer, and it already corrupted a
# strategy decision (a fresh-run argument rested on "the full force stack has never run from ep0",
# when 5 of 6 of those forces were stubs). Build state is now a FIRST-CLASS axis.
#
# FOUR grades, ordered by how much is missing:
#   built-and-fired    — mechanism exists, wired, and has actually run
#   built-never-fired  — mechanism exists + wired, never executed (the classic orphan)
#   designed-stub      — a Lever/flag exists, the MECHANISM does not (derived: emitted flag absent
#                        from the module's own trainer argparse)
#   not-even-designed  — a component an optimal config REQUIRES with no Lever, no flag, nothing.
#
# ``not-even-designed`` is STRUCTURALLY INVISIBLE to any AST/registry sweep — there is no artifact
# to detect — so it cannot be derived; it must be DECLARED against the config that needs it. That is
# what :func:`record_required_component` is for, and it is the deliberate reason this ledger gains a
# second store rather than another boolean. A required component that is never declared is the grade
# that kills a run at fire time while every gate stays green.
BUILD_FIRED = "built-and-fired"
BUILD_NEVER_FIRED = "built-never-fired"
BUILD_DESIGNED_STUB = "designed-stub"
BUILD_NOT_DESIGNED = "not-even-designed"
# The BUILD axis's own dormant-with-reactivation state, mirroring STATE_RETIRED on the
# ACTIVATION axis (which has had it since the ledger landed). Without it the required-component
# queue can ONLY be drained by BUILDING, so a component that is architecturally absent on the
# live vehicle -- no recipient mechanism can exist, so building one is wrong, not merely
# deferred -- has no representation and nags forever. CLAUDE.md's "'Off' is a tracked queue"
# names this exact state ("retired-with-reason") as mandatory. NO-FAKE: a retired row is NOT a
# capability and NOT a kill; ``notes`` MUST carry the reactivation trigger (enforced by refusal
# in ``record_required_component``), per CLAUDE.md's forbidden-premature-KILL rule.
BUILD_RETIRED = "retired-with-reason"
VALID_BUILD_GRADES = (BUILD_FIRED, BUILD_NEVER_FIRED, BUILD_DESIGNED_STUB, BUILD_NOT_DESIGNED,
                      BUILD_RETIRED)
# Worst-debt-first read order, exported so no consumer hand-copies it. A duplicated order map is
# a drift generator: the copy in the test suite silently went stale the moment a 5th grade landed.
BUILD_GRADE_ORDER: dict[str, int] = {
    BUILD_NOT_DESIGNED: 0, BUILD_DESIGNED_STUB: 1, BUILD_NEVER_FIRED: 2, BUILD_FIRED: 3,
    BUILD_RETIRED: 4,  # adjudicated closed -- never above live debt
}

# Canonical APPEND-ONLY fcntl-locked store for DECLARED required components (grade 4). Keyed by
# (component, needed_by); latest row wins on read.
REQUIRED_COMPONENT_PATH = _REPO_ROOT / ".omx" / "state" / "required_component_ledger.jsonl"


def _build_index() -> dict[str, object]:
    """{factory_name: FactoryBuild} across the WHOLE witness_dsl package (per-module trainer)."""
    from tac.witness_dsl.lever_registry import package_lever_factories

    return {f.factory: f for f in package_lever_factories()}


def package_known_levers() -> tuple[str, ...]:
    """Every lever factory in the package — the honest lever universe.

    Landed by ddm_sb2 (#819) as a superset BESIDE ``known_levers``; ddm_rg5 (#825) made
    :func:`known_levers` RETURN it, because a correct surface nobody opts into is not a repair.
    The narrow single-module view survives, explicitly named, as
    :func:`curriculum_dsl_known_levers`. This function is retained as the intention-revealing
    name (callers that specifically mean "the whole package" should keep saying so).
    """
    return tuple(sorted(_build_index()))


def build_grade(
    lever: str,
    *,
    path: Path | None = None,
    index: dict[str, object] | None = None,
) -> str:
    """The BUILD grade of one lever — derived from the registry + this ledger, never remembered.

    A lever absent from the package index is ``not-even-designed`` (nothing implements it); a lever
    whose factory emits a flag its trainer does not declare is ``designed-stub``; otherwise the
    activation ledger decides fired vs never-fired.
    """
    idx = _build_index() if index is None else index
    fb = idx.get(lever)
    if fb is None:
        return BUILD_NOT_DESIGNED
    if getattr(fb, "is_stub", False):
        return BUILD_DESIGNED_STUB
    return BUILD_FIRED if activation_status(lever, path).ever_fired else BUILD_NEVER_FIRED


def record_required_component(
    component: str,
    *,
    needed_by: str,
    missing_mechanism: str,
    owner: str,
    fire_order: int,
    consumer: str,
    grade: str = BUILD_NOT_DESIGNED,
    notes: str = "",
    agent: str | None = None,
    path: Path | None = None,
) -> dict:
    """DECLARE a component an optimal config requires — the only way grade 4 becomes visible.

    Every field is mandatory-by-refusal because a charter missing any of them is the same orphan in
    a new coat: ``owner`` (who builds it), ``missing_mechanism`` (what exactly does not exist),
    ``fire_order`` (when it must be ready), ``consumer`` (what reads it), ``needed_by`` (the config
    that blocks without it). NO-FAKE: this records a DEBT, never a capability — declaring a component
    does not build it, and :func:`not_even_designed` keeps reporting it until a factory exists.
    """
    if not component or not isinstance(component, str):
        raise ValueError(f"component must be a non-empty str, got {component!r}")
    if grade not in VALID_BUILD_GRADES:
        raise ValueError(f"invalid grade {grade!r}; must be one of {list(VALID_BUILD_GRADES)}")
    for name, val in (("needed_by", needed_by), ("missing_mechanism", missing_mechanism),
                      ("owner", owner), ("consumer", consumer)):
        if not isinstance(val, str) or len(val.strip()) < 3:
            raise ValueError(
                f"{name} is required and must be substantive (>=3 chars); got {val!r} — a charter "
                "without it cannot be actioned, which is the orphan this ledger exists to extinct")
    if not isinstance(fire_order, int) or fire_order < 0:
        raise ValueError(f"fire_order must be a non-negative int, got {fire_order!r}")
    if grade == BUILD_RETIRED and (not isinstance(notes, str) or len(notes.strip()) < 3):
        raise ValueError(
            "grade=retired-with-reason requires substantive notes carrying the REACTIVATION "
            "TRIGGER (what measured fact would re-open it); a retirement without one is a KILL, "
            "which CLAUDE.md forbids as a resting state")
    row = {
        "component": component, "needed_by": needed_by, "grade": grade,
        "missing_mechanism": missing_mechanism, "owner": owner, "fire_order": fire_order,
        "consumer": consumer, "notes": notes, "agent": agent, "ts": _utc(),
    }
    append_locked_jsonl(Path(path) if path is not None else REQUIRED_COMPONENT_PATH, row)
    return row


def read_required_components(path: Path | None = None) -> list[dict]:
    """Latest-row-wins declared required components, sorted by (fire_order, component)."""
    p = Path(path) if path is not None else REQUIRED_COMPONENT_PATH
    if not p.exists():
        return []
    latest: dict[tuple[str, str], dict] = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("component") and row.get("grade") in VALID_BUILD_GRADES:
            latest[(row["component"], row.get("needed_by", ""))] = row
    return sorted(latest.values(), key=lambda r: (r.get("fire_order", 0), r["component"]))


def not_even_designed(path: Path | None = None) -> tuple[dict, ...]:
    """Declared required components that STILL have no lever factory — the live grade-4 debt.

    A declared component whose factory has since landed drops off automatically (the registry, not
    a human, decides), so this list is drained by BUILDING, never by editing a memo.

    The ONE other exit is an explicit ``BUILD_RETIRED`` row (``retired-with-reason``), which
    requires a recorded reactivation trigger. It exists because "build it" is the wrong verdict
    for a component whose recipient mechanism cannot exist on the live vehicle; without this exit
    such a row nags forever and the queue trains its readers to ignore it.
    """
    idx = _build_index()
    return tuple(r for r in read_required_components(path)
                 if r["component"] not in idx and r.get("grade") != BUILD_RETIRED)


def build_completeness_report(path: Path | None = None) -> list[dict]:
    """One row per lever factory AND per declared required component, five-graded.

    This is the operator-facing "what is hollow?" surface. Ordered worst-grade-first so the debt
    is the first thing read, never a footnote under a wall of built levers.
    """
    idx = _build_index()
    rows: list[dict] = []
    for name in sorted(idx):
        fb = idx[name]
        rows.append({
            "component": name,
            "grade": build_grade(name, path=path, index=idx),
            "module": getattr(fb, "module", None),
            "trainer": getattr(fb, "trainer", None),
            "missing_flags": list(getattr(fb, "missing_flags", ())),
            "label_drift": getattr(fb, "label_drift", False),
            "activation_state": activation_status(name, path).state,
        })
    for r in read_required_components(path):
        if r["component"] in idx:
            continue  # it got built; the registry decides, not the declaration
        # A retired row keeps its OWN grade: reporting it as not-even-designed would re-assert the
        # debt this adjudication closed, which is how a retirement silently becomes a build order.
        rows.append({
            "component": r["component"],
            "grade": BUILD_RETIRED if r.get("grade") == BUILD_RETIRED else BUILD_NOT_DESIGNED,
            "module": None,
            "trainer": None, "missing_flags": [], "label_drift": False,
            "activation_state": "not-registered", "needed_by": r.get("needed_by"),
            "owner": r.get("owner"), "fire_order": r.get("fire_order"),
            "missing_mechanism": r.get("missing_mechanism"), "consumer": r.get("consumer"),
            "notes": r.get("notes"),
        })
    rows.sort(key=lambda r: (BUILD_GRADE_ORDER.get(r["grade"], 9), r.get("fire_order", 0),
                             r["component"]))
    return rows


__all__ = [
    "BUILD_DESIGNED_STUB",
    "BUILD_FIRED",
    "BUILD_GRADE_ORDER",
    "BUILD_NEVER_FIRED",
    "BUILD_NOT_DESIGNED",
    "BUILD_RETIRED",
    "EVENT_FIRED",
    "EVENT_MEASURED",
    "EVENT_RETIRED",
    "LEDGER_PATH",
    "REQUIRED_COMPONENT_PATH",
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
    "VALID_BUILD_GRADES",
    "VALID_EVENTS",
    "VALID_SIG_AXES",
    "VALID_SIG_LABELS",
    "ActivationStatus",
    "activation_report",
    "activation_status",
    "build_completeness_report",
    "build_grade",
    "curriculum_dsl_known_levers",
    "duty_to_measure",
    "duty_to_measure_ranked",
    "known_levers",
    "levers_fired_for_run",
    "never_fired",
    "not_even_designed",
    "package_known_levers",
    "read_pointer_s",
    "read_required_components",
    "record_activation",
    "record_measured_for_run",
    "record_relative_significance",
    "record_required_component",
    "relative_significance",
]
