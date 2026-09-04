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

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

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
    "seg_chroma_boundary_276": "SegChromaBoundary",  # 2026-09-04 MAIN: task-#-keyed significance row reconciled onto its held factory (P1 orphan cure)
}

# canonical event vocabulary
EVENT_FIRED = "fired"        # a run launched with this lever (an A/B arm was actually run)
EVENT_MEASURED = "measured"  # a byte-closed verdict landed for a run using this lever
EVENT_RETIRED = "retired"    # the lever was retired with a recorded reason (dormant-with-reactivation)
EVENT_FOLDED = "folded"      # terminal config explicitly folds the lever into another treatment
EVENT_QUEUED = "queued"      # terminal config explicitly queues the lever with a fire trigger
VALID_EVENTS = frozenset({
    EVENT_FIRED,
    EVENT_MEASURED,
    EVENT_RETIRED,
    EVENT_FOLDED,
    EVENT_QUEUED,
})

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
    if event in {EVENT_FOLDED, EVENT_QUEUED, EVENT_RETIRED} and not (
        isinstance(reason, str) and reason.strip()
    ):
        raise ValueError(f"{event} activation event requires a non-empty reason")
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
    # EVERY row cites the store its state was read from (ddm_lr2, 2026-08-03). MEASURED
    # cross-store contradiction: four levers are labelled never-fired by one 08-02 survey while
    # the deferral-queue ledger records them FIRED — two of them fired-and-LOST. Neither survey
    # was careless; they consulted DIFFERENT STORES and nothing joins them. A fired-and-lost
    # lever re-listed as never-fired invites a wasted run; a never-fired lever mislabelled fired
    # stays orphaned forever. Same genus as the harness-TaskList-vs-repo-task-ledger split, whose
    # cure is the same: cite CONTENT and its SOURCE, never a bare label.
    store = str(Path(path) if path is not None else LEDGER_PATH)
    rows = []
    for name in names:
        st = activation_status(name, path)
        rows.append({
            "lever": name,
            "state_store": store,
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


class TerminalJoinDisposition(StrEnum):
    FIRED = "FIRED"
    FOLDED = "FOLDED"
    QUEUED = "queued"


class TerminalJoinStatus(StrEnum):
    PASS = "PASS"
    REFUSE = "REFUSE"


@dataclass(frozen=True, slots=True)
class TerminalActivationJoinRow:
    lever: str
    disposition: TerminalJoinDisposition
    ledger_event: str
    reason: str
    run_ref: str | None
    verdict_ref: str | None
    ts: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lever": self.lever,
            "disposition": self.disposition.value,
            "ledger_event": self.ledger_event,
            "reason": self.reason,
            "run_ref": self.run_ref,
            "verdict_ref": self.verdict_ref,
            "ts": self.ts,
        }


@dataclass(frozen=True, slots=True)
class TerminalActivationJoinReceipt:
    compiled_config_sha256: str
    ledger_path: str
    ledger_sha256: str | None
    non_default_levers: tuple[str, ...]
    rows: tuple[TerminalActivationJoinRow, ...]
    missing_levers: tuple[str, ...]

    @property
    def status(self) -> TerminalJoinStatus:
        return TerminalJoinStatus.REFUSE if self.missing_levers else TerminalJoinStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "terminal_activation_join.v1",
            "status": self.status.value,
            "compiled_config_sha256": self.compiled_config_sha256,
            "ledger_path": self.ledger_path,
            "ledger_sha256": self.ledger_sha256,
            "non_default_levers": list(self.non_default_levers),
            "rows": [row.to_dict() for row in self.rows],
            "missing_levers": list(self.missing_levers),
            "score_claim": False,
            "execution_allowed": False,
        }


def _canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("compiled config must be canonical-JSON serializable") from exc


def _lever_names_from_sequence(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("compiled config lever list must be a sequence")
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            name = item
            non_default = True
        elif isinstance(item, Mapping):
            name = item.get("name")
            default = item.get("default", False)
            if type(default) is not bool:
                raise ValueError("lever default marker must be boolean")
            non_default = item.get("non_default", not default)
            if type(non_default) is not bool:
                raise ValueError("lever non_default/default marker must be boolean")
        else:
            raise ValueError("compiled config lever entries must be names or mappings")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("compiled config lever name must be a non-empty string")
        if non_default:
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("compiled config contains duplicate non-default lever names")
    return tuple(names)


def compiled_non_default_levers(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract explicitly active/non-default levers from a compiled DSL JSON object."""

    if not isinstance(config, Mapping):
        raise ValueError("compiled config must be a mapping")
    manifest = config.get("dsl_program_manifest")
    if isinstance(manifest, Mapping) and "expected_active_levers" in manifest:
        return _lever_names_from_sequence(manifest["expected_active_levers"])
    if "expected_active_levers" in config:
        return _lever_names_from_sequence(config["expected_active_levers"])
    typed = config.get("typed_config")
    if isinstance(typed, Mapping) and "levers" in typed:
        return _lever_names_from_sequence(typed["levers"])
    if "levers" in config:
        return _lever_names_from_sequence(config["levers"])
    raise ValueError(
        "compiled config must expose expected_active_levers or typed_config.levers"
    )


def terminal_activation_join(
    compiled_config: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> TerminalActivationJoinReceipt:
    """Join every non-default compiled lever to explicit terminal ledger evidence."""

    config_bytes = _canonical_config_bytes(compiled_config)
    names = compiled_non_default_levers(compiled_config)
    ledger = Path(path) if path is not None else LEDGER_PATH
    ledger_sha256 = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.is_file() else None
    events = _read_events(ledger)
    rows: list[TerminalActivationJoinRow] = []
    missing: list[str] = []
    for name in names:
        evidence = [row for row in events if row.get("lever") == name]
        selected: dict[str, Any] | None = None
        disposition: TerminalJoinDisposition | None = None
        for event in reversed(evidence):
            kind = event.get("event")
            if kind in {EVENT_FIRED, EVENT_MEASURED}:
                selected, disposition = event, TerminalJoinDisposition.FIRED
                break
            if kind in {EVENT_FOLDED, EVENT_RETIRED}:
                selected, disposition = event, TerminalJoinDisposition.FOLDED
                break
            if kind == EVENT_QUEUED:
                selected, disposition = event, TerminalJoinDisposition.QUEUED
                break
        if selected is None or disposition is None:
            missing.append(name)
            continue
        rows.append(TerminalActivationJoinRow(
            lever=name,
            disposition=disposition,
            ledger_event=str(selected["event"]),
            reason=str(selected.get("reason") or ""),
            run_ref=selected.get("run_ref"),
            verdict_ref=selected.get("verdict_ref"),
            ts=selected.get("ts"),
        ))
    return TerminalActivationJoinReceipt(
        compiled_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        ledger_path=str(ledger),
        ledger_sha256=ledger_sha256,
        non_default_levers=names,
        rows=tuple(rows),
        missing_levers=tuple(missing),
    )


# ── VACUITY SELF-REPORT: the ledger must state its own denominator (ddm_lr2, 2026-08-03) ────────
# THE INCIDENT, MEASURED. ``never_fired()`` returned 178 of 180 levers and that number was read as
# evidence of a huge orphan backlog. It was an INSTRUMENT ARTIFACT. Three measurements, each
# independently sufficient to void it:
#
#   1. The ledger's ONLY non-test writer is ``tools/launch_witness_run.py`` — the launcher of the
#      RETIRED vehicle. ``tools/launch_tr1_run.py``, the governed launcher of the vehicle we
#      actually ship, never imports ``record_activation``.
#   2. The ledger's last write is 2026-07-27T21:17:34Z. ``launch_tr1_run.py`` was added
#      2026-07-28. The ledger did not go stale from disuse — it stopped on the day the vehicle
#      changed launchers, which is the SAME bug class as the lever modules that kept binding to
#      the retired trainer (see ``lever_registry.module_declares_trainer``).
#   3. 31 governed TR1 launch receipts exist on the SSD tier and produced ZERO ledger rows.
#
# So "178 never-fired" says nothing about levers; it says the writer is bound to a dead vehicle.
# An empty scope emitting the same symbol as a clean full scope is the vacuity genus this campaign
# keeps paying for, and the cure is always the same: REPORT THE DENOMINATOR. These helpers do not
# change what ``never_fired()`` returns (its consumers depend on that contract) — they make it
# impossible to quote the number without its basis.
_TR1_RECEIPT_SCHEMA = "ddm_lv1_tr1_governed_launch_receipt.v1"
# SSD-first per CLAUDE.md storage discipline; run bulk lives off local disk, so the receipts do
# too. A root that is not mounted is reported as UNSCANNED, never as "found nothing" — an
# unmounted volume returning 0 would otherwise read exactly like a clean tree.
LIVE_LAUNCH_ROOTS: tuple[str, ...] = (
    "/Volumes/VertigoDataTier/pact",
    "/Volumes/APDataStore/pact",
    "experiments/results",
)


@dataclass(frozen=True)
class LedgerCoverage:
    """What the activation ledger can and cannot see — its basis, stated with the answer."""

    known_levers: int
    levers_with_any_row: int
    rows: int
    last_write_utc: str | None
    live_launch_receipts: int
    live_receipts_joined_to_ledger: int
    roots_scanned: tuple[str, ...]
    roots_unavailable: tuple[str, ...]
    is_vacuous: bool
    vacuity_reason: str

    @property
    def lever_coverage_fraction(self) -> float:
        """Fraction of known levers the ledger has ANY row for. 0.0 on an empty known set —
        never 1.0, because "nothing known" must not read as "everything covered"."""
        return (self.levers_with_any_row / self.known_levers) if self.known_levers else 0.0

    def to_dict(self) -> dict:
        return {
            "known_levers": self.known_levers,
            "levers_with_any_row": self.levers_with_any_row,
            "lever_coverage_fraction": round(self.lever_coverage_fraction, 4),
            "rows": self.rows,
            "last_write_utc": self.last_write_utc,
            "live_launch_receipts": self.live_launch_receipts,
            "live_receipts_joined_to_ledger": self.live_receipts_joined_to_ledger,
            "roots_scanned": list(self.roots_scanned),
            "roots_unavailable": list(self.roots_unavailable),
            "is_vacuous": self.is_vacuous,
            "vacuity_reason": self.vacuity_reason,
        }


def live_launch_receipts(roots: tuple[str, ...] | None = None) -> tuple[list[dict], tuple[str, ...], tuple[str, ...]]:
    """Governed LIVE-vehicle launch receipts, plus which roots were actually readable.

    Returns ``(receipts, roots_scanned, roots_unavailable)``. The third element is the honest
    part: an unmounted SSD must never be silently indistinguishable from a tier with no runs.
    """
    scanned: list[str] = []
    unavailable: list[str] = []
    out: list[dict] = []
    for root in (roots if roots is not None else LIVE_LAUNCH_ROOTS):
        base = Path(root)
        if not base.is_absolute():
            base = _REPO_ROOT / base
        if not base.is_dir():
            unavailable.append(str(root))
            continue
        scanned.append(str(root))
        for depth in ("*/launch_receipt.json", "*/*/launch_receipt.json", "*/*/*/launch_receipt.json"):
            for p in base.glob(depth):
                try:
                    row = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if isinstance(row, dict) and row.get("schema") == _TR1_RECEIPT_SCHEMA:
                    row = dict(row)
                    row["_receipt_path"] = str(p)
                    row["_run_dir"] = str(p.parent)
                    out.append(row)
    out.sort(key=lambda r: str(r.get("_receipt_path")))
    return out, tuple(scanned), tuple(unavailable)


def ledger_coverage(path: Path | None = None,
                    roots: tuple[str, ...] | None = None) -> LedgerCoverage:
    """The ledger's own basis — call this BEFORE quoting :func:`never_fired`.

    ``is_vacuous`` is True when the ledger cannot see the live vehicle at all: governed live
    launches exist on a readable tier and NONE of them has a ledger row. In that state
    ``never_fired()`` is measuring the writer, not the levers, and any orphan count read off it
    is unanchored.
    """
    events = _read_events(path)
    with_rows = {e["lever"] for e in events}
    last = max((e.get("ts") or "" for e in events), default="") or None
    receipts, scanned, unavailable = live_launch_receipts(roots)
    fired_refs = [str(e.get("run_ref") or "") for e in events if e.get("run_ref")]
    joined = sum(
        1 for r in receipts
        if any(_run_ref_matches(ref, str(r.get("_run_dir"))) for ref in fired_refs)
    )
    if not scanned:
        vacuous, reason = True, (
            "no launch-provenance root was readable "
            f"(unavailable: {', '.join(unavailable) or 'none configured'}) — coverage UNKNOWN, "
            "not zero"
        )
    elif receipts and joined == 0:
        vacuous, reason = True, (
            f"{len(receipts)} governed live-vehicle launch receipt(s) exist and NONE has a "
            "ledger row: the ledger's only writer is the RETIRED vehicle's launcher "
            "(tools/launch_witness_run.py); tools/launch_tr1_run.py does not record activation. "
            "never_fired() is measuring the writer, not the levers."
        )
    elif not events:
        vacuous, reason = True, "ledger is empty: every lever reads never-fired by construction"
    else:
        vacuous, reason = False, ""
    return LedgerCoverage(
        known_levers=len(known_levers()),
        levers_with_any_row=len(with_rows),
        rows=len(events),
        last_write_utc=last,
        live_launch_receipts=len(receipts),
        live_receipts_joined_to_ledger=joined,
        roots_scanned=scanned,
        roots_unavailable=unavailable,
        is_vacuous=vacuous,
        vacuity_reason=reason,
    )


def live_launch_lever_names(roots: tuple[str, ...] | None = None) -> tuple[dict[str, int], dict]:
    """Lever names the LIVE vehicle actually launched, read from governed receipts' sealed tickets.

    This is the live-vehicle answer the ledger cannot give, and it also MEASURES the join defect
    that makes naively wiring ``record_activation`` into the live launcher insufficient:

    the DSL's lever universe (:func:`known_levers`) is keyed by FACTORY name, while a ticket
    records the constructed ``Lever.name``, and the live spec builds those names with f-strings
    parameterised by the factory's own arguments (``Lever(name=f"tr1_token_grid_D{downsample}_
    c{code_width}")``). The instance-name space is therefore not statically enumerable at all,
    and MEASURED overlap with ``known_levers()`` is 0 of 20 on a live ticket. Recording ticket
    names into the ledger as-is would leave ``never_fired()`` reporting every factory as
    never-fired forever, while the rows piled up under keys nothing joins to — the cross-store
    defect, newly manufactured. Returns ``(name -> launch count, diagnostics)``.
    """
    receipts, scanned, unavailable = live_launch_receipts(roots)
    counts: dict[str, int] = {}
    tickets_read = tickets_missing = 0
    for r in receipts:
        tp = r.get("ticket_path")
        p = Path(str(tp)) if tp else None
        if p is None or not p.is_file():
            tickets_missing += 1
            continue
        try:
            ticket = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            tickets_missing += 1
            continue
        tickets_read += 1
        for lev in ticket.get("levers") or []:
            nm = (lev or {}).get("name")
            if nm:
                counts[str(nm)] = counts.get(str(nm), 0) + 1
    known = set(known_levers())
    joinable = sorted(n for n in counts if n in known)
    diagnostics = {
        "receipts": len(receipts),
        "tickets_read": tickets_read,
        "tickets_missing": tickets_missing,
        "distinct_lever_instance_names": len(counts),
        "joinable_to_known_levers": len(joinable),
        "unjoinable_to_known_levers": len(counts) - len(joinable),
        "roots_scanned": list(scanned),
        "roots_unavailable": list(unavailable),
        "namespace_join_is_broken": bool(counts) and not joinable,
    }
    return dict(sorted(counts.items())), diagnostics


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
# SIX grades. The first four are ordered by how much is missing on THIS vehicle; the last two are
# adjudications that only a human/agent can make, so both are DECLARED, never derived:
#   built-and-fired         — mechanism exists, wired, and has actually run
#   built-never-fired       — mechanism exists + wired, never executed (the classic orphan)
#   designed-stub           — a Lever/flag exists, the MECHANISM does not (derived: emitted flag
#                             absent from the module's own trainer argparse)
#   not-even-designed       — a component an optimal config REQUIRES with no Lever, no flag, nothing
#   retired-with-reason     — adjudicated dormant; a recipient cannot exist here (ddm_wr2, #864)
#   built-elsewhere-unwired — exists, tested, HAS RUN elsewhere; the live path has no call site and
#                             is running a measured-worse alternative (the #864 P0 class)
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
# The grade the operator designated P0 on 2026-08-01 ("All built-elsewhere-unwired is p0"), and
# the one this axis could not express. The four grades above all describe a component by what is
# MISSING on THIS vehicle. This one describes a component that is missing NOTHING: it exists, it
# is tested, and it HAS RUN -- elsewhere -- while the live path has no call site for it.
#
# WHY IT IS NOT ``built-never-fired``. That grade means "exists on THIS vehicle, has not run": a
# dormant capability, harmless. This one means "exists AND has run, elsewhere, and the live path
# is meanwhile running a measured-WORSE alternative". Collapsing the two loses exactly the
# distinction the P0 is about, and it loses it in the harmful direction -- the class reads as
# ordinary dormant debt while it costs realized score continuously and silently.
#
# WHY IT MUST BE DECLARED AND NOT DERIVED. ddm_gd5 (#864) built the auto-derived detector, measured
# it against real controls, and refuted three formulations: the P0's literal import-reachability
# predicate fires on 1229 of 3251 modules; the "measured-better successor" relation that makes the
# class harmful at all exists only in memos and receipts, with NO representation in code. So there
# is no artifact for an AST sweep to walk -- structurally the same reason ``not-even-designed`` is
# declared rather than derived.
#
# THE HARM CLAUSE IS MANDATORY-BY-REFUSAL. ddm_wr2 (#864) adjudicated the 7 unowned candidate rows
# and MEASURED that 4 of 7 satisfy the literal predicate while 0 of 7 satisfy the harm clause --
# because none has a RECIPIENT (a live mechanism it would replace) or a MEASURED comparison. A
# grade that accepts recipient-less rows would immediately re-admit that error and turn an
# adjudicated negative-control set back into a wiring backlog. So ``record_required_component``
# REFUSES this grade without both ``live_recipient`` and ``measured_comparison``. NO-FAKE: those
# fields record an already-measured fact, never manufacture one.
#
# THE HARM CLAUSE MUST ALSO BE SIGNED (ddm_wd2, #864/#861). The refusal above checked that a
# comparison was PRESENT, never that it pointed the right way -- ``measured_comparison`` was free
# text validated only by ``len(...) >= 3``, while the refusal message promised "a MEASURED
# COMPARISON showing the component BEATS it". That is a comment-only contract, which CLAUDE.md
# forbids, and it sat on the rank-0 grade.
#
# IT IS NOT HYPOTHETICAL: it admitted the P0's own headline instance. ``p0_864`` records the pose
# pair as "~39x better, ~38x cheaper, RACED"; ddm_wd1 then MEASURED that family plateauing at
# d_pose ~29-30 against a live realized 0.00858133 -- roughly 3,400x WORSE, an unwired far-worse
# PREDECESSOR whose wiring would be a REGRESSION. Under the length-only clause that row was
# recordable verbatim, and ``BUILD_GRADE_ORDER`` would have sorted it above every live debt row.
#
# So the numbers themselves are now mandatory-by-refusal and the direction is CHECKED: the caller
# supplies the two ALREADY-MEASURED scalars plus which way the metric runs, and the declaration is
# refused unless the candidate STRICTLY beats the live recipient. NO-FAKE: this records two
# measured facts and derives only their comparison -- it cannot manufacture a win, and a strict
# inequality means a tie is not "beats" either.
METRIC_LOWER_BETTER = "lower-is-better"
METRIC_HIGHER_BETTER = "higher-is-better"
VALID_METRIC_DIRECTIONS = (METRIC_LOWER_BETTER, METRIC_HIGHER_BETTER)
BUILD_ELSEWHERE_UNWIRED = "built-elsewhere-unwired"
VALID_BUILD_GRADES = (BUILD_FIRED, BUILD_NEVER_FIRED, BUILD_DESIGNED_STUB, BUILD_NOT_DESIGNED,
                      BUILD_RETIRED, BUILD_ELSEWHERE_UNWIRED)
# Worst-debt-first read order, exported so no consumer hand-copies it. A duplicated order map is
# a drift generator: the copy in the test suite silently went stale the moment a 5th grade landed.
#
# ``built-elsewhere-unwired`` sorts FIRST, and that rank is DERIVED rather than chosen: it is the
# only grade whose declaration is refused without a MEASURED comparison proving the live path is
# currently worse off. Every other grade describes an unquantified future cost (a capability we do
# not yet have); this one carries a measured present loss at declaration time.
BUILD_GRADE_ORDER: dict[str, int] = {
    BUILD_ELSEWHERE_UNWIRED: 0,  # measured present loss -- the only grade that quantifies its harm
    BUILD_NOT_DESIGNED: 1, BUILD_DESIGNED_STUB: 2, BUILD_NEVER_FIRED: 3, BUILD_FIRED: 4,
    BUILD_RETIRED: 5,  # adjudicated closed -- never above live debt
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
    live_recipient: str = "",
    measured_comparison: str = "",
    live_measured: float | None = None,
    candidate_measured: float | None = None,
    metric_direction: str = "",
    agent: str | None = None,
    path: Path | None = None,
) -> dict:
    """DECLARE a component an optimal config requires — the only way grade 4 becomes visible.

    Every field is mandatory-by-refusal because a charter missing any of them is the same orphan in
    a new coat: ``owner`` (who builds it), ``missing_mechanism`` (what exactly does not exist),
    ``fire_order`` (when it must be ready), ``consumer`` (what reads it), ``needed_by`` (the config
    that blocks without it). NO-FAKE: this records a DEBT, never a capability — declaring a component
    does not build it, and :func:`not_even_designed` keeps reporting it until a factory exists.

    ``grade=built-elsewhere-unwired`` additionally requires ``live_recipient`` (the live mechanism
    it would replace), ``measured_comparison`` (the human-readable citation for where the numbers
    came from), and the numbers themselves — ``live_measured``, ``candidate_measured``,
    ``metric_direction`` — because that grade's rank-0 position asserts a MEASURED PRESENT LOSS.
    The declaration is REFUSED unless the candidate strictly beats the recipient in the declared
    direction. NO-FAKE: the caller supplies two already-measured scalars; only their comparison is
    derived, so this surface cannot manufacture a win.
    """
    harm_advantage = _validate_required_component(
        component, needed_by=needed_by, missing_mechanism=missing_mechanism, owner=owner,
        fire_order=fire_order, consumer=consumer, grade=grade, notes=notes,
        live_recipient=live_recipient, measured_comparison=measured_comparison,
        live_measured=live_measured, candidate_measured=candidate_measured,
        metric_direction=metric_direction)
    row = {
        "component": component, "needed_by": needed_by, "grade": grade,
        "missing_mechanism": missing_mechanism, "owner": owner, "fire_order": fire_order,
        "consumer": consumer, "notes": notes, "live_recipient": live_recipient,
        "measured_comparison": measured_comparison, "live_measured": live_measured,
        "candidate_measured": candidate_measured, "metric_direction": metric_direction,
        "harm_advantage": harm_advantage, "agent": agent, "ts": _utc(),
    }
    append_locked_jsonl(Path(path) if path is not None else REQUIRED_COMPONENT_PATH, row)
    return row


def _validate_required_component(
    component: object,
    *,
    needed_by: object,
    missing_mechanism: object,
    owner: object,
    fire_order: object,
    consumer: object,
    grade: object,
    notes: object = "",
    live_recipient: object = "",
    measured_comparison: object = "",
    live_measured: object = None,
    candidate_measured: object = None,
    metric_direction: object = "",
) -> float | None:
    """THE single admission predicate for a required-component row. Raises; returns harm_advantage.

    Extracted by ddm_ri1 (#899) so the WRITE path and the READ path decide admissibility with the
    SAME code. Before this, only :func:`record_required_component` enforced the charter and the harm
    clause, while :func:`read_required_components` admitted any JSON line carrying a ``component``
    and a known ``grade``. MEASURED against HEAD before the change: a hand-appended grade-5 row with
    no ``live_measured``/``candidate_measured``/``metric_direction`` at all — a shape the write path
    REFUSES — was read back, entered :func:`built_elsewhere_unwired`, and sorted to
    ``build_completeness_report()[0]``, ABOVE a genuinely measured row. A gate whose read path does
    not re-run its own write gate is not a gate; per ``ddm_gd5`` deleting the grade-5 detector,
    DECLARATION is currently the only route into this grade, so the read path is the only remaining
    check on it.

    A duplicated predicate would be a drift generator (this module already says so about
    ``BUILD_GRADE_ORDER``), so there is exactly one and both callers share it.
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
    if grade == BUILD_ELSEWHERE_UNWIRED:
        # The HARM CLAUSE, enforced structurally. ddm_wr2 MEASURED that 4 of 7 candidate rows meet
        # the literal predicate while 0 of 7 meet this clause; without refusal here the grade
        # re-admits recipient-less rows and the adjudicated negative-control set silently becomes a
        # wiring backlog again. Both facts must already be MEASURED -- this records, never derives.
        for name, val in (("live_recipient", live_recipient),
                          ("measured_comparison", measured_comparison)):
            if not isinstance(val, str) or len(val.strip()) < 3:
                raise ValueError(
                    f"grade=built-elsewhere-unwired requires a substantive {name}; got {val!r}. "
                    "This grade's HARM CLAUSE needs BOTH a RECIPIENT (the live mechanism this "
                    "component would replace) and a MEASURED COMPARISON showing the component "
                    "beats it. Without both it is indistinguishable from built-never-fired, which "
                    "is dormant and harmless -- record that grade instead.")
        # ...and the comparison must POINT THE RIGHT WAY. Prose cannot be checked, so the two
        # measured scalars and the metric's direction are required and the sign is DERIVED.
        if metric_direction not in VALID_METRIC_DIRECTIONS:
            raise ValueError(
                f"grade=built-elsewhere-unwired requires metric_direction in "
                f"{list(VALID_METRIC_DIRECTIONS)}; got {metric_direction!r}. The HARM CLAUSE is "
                "directional: without knowing which way the metric runs, 'beats' is undecidable.")
        for name, val in (("live_measured", live_measured),
                          ("candidate_measured", candidate_measured)):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"grade=built-elsewhere-unwired requires a numeric {name}; got {val!r}. The "
                    "HARM CLAUSE needs the ALREADY-MEASURED value, not prose: free text can assert "
                    "a win that the numbers contradict, which is how this grade admitted a row "
                    "that was measured ~3,400x WORSE than the live path (ddm_wd1, #861).")
            if not math.isfinite(float(val)):
                raise ValueError(
                    f"grade=built-elsewhere-unwired requires a FINITE {name}; got {val!r}. A "
                    "non-finite measurement is a failed measurement, never a comparison.")
        live_f, cand_f = float(live_measured), float(candidate_measured)
        beats = cand_f < live_f if metric_direction == METRIC_LOWER_BETTER else cand_f > live_f
        if not beats:
            raise ValueError(
                f"grade=built-elsewhere-unwired REFUSED: candidate_measured={cand_f!r} does not "
                f"beat live_recipient {live_recipient!r} at live_measured={live_f!r} under "
                f"{metric_direction!r}. This grade's whole justification -- and its rank-0 position "
                "in BUILD_GRADE_ORDER -- is a MEASURED PRESENT LOSS on the live path. A component "
                "that ties or loses is an unwired EQUAL or an unwired WORSE PREDECESSOR, and "
                "wiring it would be a REGRESSION, not a repair. Record built-never-fired (dormant, "
                "harmless) or retired-with-reason (adjudicated, needs a reactivation trigger).")
        # Dimensionless relative advantage, recorded as EVIDENCE for a future ranker. Defined only
        # when both magnitudes are strictly positive; a ratio across a zero or a signed quantity is
        # meaningless, so it is None rather than a fabricated number. The strict inequality above,
        # not this ratio, is the gate.
        harm_advantage = None
        if live_f > 0.0 and cand_f > 0.0:
            harm_advantage = (live_f / cand_f if metric_direction == METRIC_LOWER_BETTER
                              else cand_f / live_f)
    else:
        harm_advantage = None
    return harm_advantage


# ── RECORD INTEGRITY (ddm_ri1, #899) ─────────────────────────────────────────────────────────────
# What a reader of a stored row is allowed to believe. A row that was WRITTEN through
# record_required_component passed the charter + harm clause; a row that appeared in the file some
# other way (hand edit, partial write, a schema change that predates a clause) did not. Before this
# typing the two were INDISTINGUISHABLE on read, which is the vacuity genus at the record level:
# "declared" returned the same shape as "verified", so a consumer could not tell an asserted harm
# from a measured one.
RECORD_VERIFIED = "verified"              # re-passes the write-path predicate NOW
RECORD_DECLARED_UNVERIFIED = "declared-unverified"  # present and readable, but fails that predicate
RECORD_MALFORMED = "malformed"            # not parseable / not a row at all -- counted, never silent
VALID_RECORD_INTEGRITY = (RECORD_VERIFIED, RECORD_DECLARED_UNVERIFIED, RECORD_MALFORMED)


def verify_required_component_row(row: dict) -> tuple[str, str]:
    """Re-run the WRITE-path predicate on a STORED row → ``(integrity, reason)``.

    ``reason`` is the verbatim refusal text when the row does not verify, so the operator sees WHY a
    stored claim is not believable rather than a bare flag. NO-FAKE: this decides nothing new — it
    replays the same admission check the writer applied, so a row can never be "verified" here on
    weaker evidence than it would have needed to be written.
    """
    if not isinstance(row, dict):
        return RECORD_MALFORMED, f"not a JSON object: {type(row).__name__}"
    try:
        _validate_required_component(
            row.get("component"), needed_by=row.get("needed_by"),
            missing_mechanism=row.get("missing_mechanism"), owner=row.get("owner"),
            fire_order=row.get("fire_order"), consumer=row.get("consumer"),
            grade=row.get("grade"), notes=row.get("notes", ""),
            live_recipient=row.get("live_recipient", ""),
            measured_comparison=row.get("measured_comparison", ""),
            live_measured=row.get("live_measured"),
            candidate_measured=row.get("candidate_measured"),
            metric_direction=row.get("metric_direction", ""))
    except (ValueError, TypeError) as exc:
        # TypeError too: a stored value of an unexpected TYPE must type the ROW, never crash the
        # read of the whole ledger. One bad row taking down the reader is how the recall layer
        # lost 100% of its corpus to 0.25% of it on 2026-08-01.
        return RECORD_DECLARED_UNVERIFIED, str(exc)
    return RECORD_VERIFIED, ""


def read_required_components(path: Path | None = None) -> list[dict]:
    """Latest-row-wins declared required components, sorted by (fire_order, component).

    Every returned row carries ``record_integrity`` (see :func:`verify_required_component_row`) and
    ``record_integrity_reason``. Rows are NEVER dropped for failing to verify — dropping would be
    the signal loss this ledger exists to prevent — but they are TYPED, so no consumer can mistake a
    declaration for a measurement. Unparseable lines are counted by
    :func:`required_component_integrity_summary`, which reports the DENOMINATOR: a silently skipped
    line and a file with nothing in it used to be indistinguishable.
    """
    rows, _ = _read_required_components_with_defects(path)
    return rows


def _read_required_components_with_defects(
    path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """(typed rows, malformed line records). Internal so the public read stays a plain list."""
    p = Path(path) if path is not None else REQUIRED_COMPONENT_PATH
    if not p.exists():
        return [], []
    latest: dict[tuple[str, str], dict] = {}
    malformed: list[dict] = []
    for line_no, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except (json.JSONDecodeError, ValueError) as exc:
            malformed.append({"line_no": line_no, "reason": f"unparseable JSON: {exc}",
                              "excerpt": ln[:120]})
            continue
        if not (isinstance(row, dict) and row.get("component")
                and row.get("grade") in VALID_BUILD_GRADES):
            malformed.append({"line_no": line_no,
                              "reason": "missing component or grade outside VALID_BUILD_GRADES",
                              "excerpt": ln[:120]})
            continue
        integrity, reason = verify_required_component_row(row)
        row["record_integrity"] = integrity
        row["record_integrity_reason"] = reason
        latest[(row["component"], row.get("needed_by", ""))] = row
    ordered = sorted(latest.values(), key=lambda r: (_fire_order_key(r), r["component"]))
    return ordered, malformed


def _fire_order_key(row: dict) -> int:
    """Sort key that a malformed ``fire_order`` cannot crash.

    Found in my own round-2 review: a stored ``fire_order`` of the wrong TYPE (a string, a null)
    made ``sorted`` raise and took down the read of the WHOLE store — one bad row costing the
    entire corpus, the failure the recall layer hit on 2026-08-01. Such a row is already typed
    ``declared-unverified``; it must not also be able to silence its 21 healthy neighbours.
    """
    val = row.get("fire_order", 0)
    return val if isinstance(val, int) and not isinstance(val, bool) else 0


def required_component_integrity_summary(path: Path | None = None) -> dict:
    """The DENOMINATOR for the required-component store: how much of it is believable.

    Exists because an empty scope and a clean scope emit the same symbol unless the counts are
    reported (memory ``vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801``). A
    caller that only ever sees the row list cannot tell "no unverified claims" from "the reader
    skipped them".
    """
    rows, malformed = _read_required_components_with_defects(path)
    by = {k: [r["component"] for r in rows if r.get("record_integrity") == k]
          for k in (RECORD_VERIFIED, RECORD_DECLARED_UNVERIFIED)}
    return {
        "rows_read": len(rows),
        "verified": len(by[RECORD_VERIFIED]),
        "declared_unverified": len(by[RECORD_DECLARED_UNVERIFIED]),
        "malformed_lines": len(malformed),
        "declared_unverified_components": by[RECORD_DECLARED_UNVERIFIED],
        "malformed_detail": malformed,
    }


def not_even_designed(path: Path | None = None) -> tuple[dict, ...]:
    """Declared required components that STILL have no lever factory — the live grade-4 debt.

    A declared component whose factory has since landed drops off automatically (the registry, not
    a human, decides), so this list is drained by BUILDING, never by editing a memo.

    TWO declared grades exit this queue, and neither is a kill:

    * ``BUILD_RETIRED`` (``retired-with-reason``) — requires a recorded reactivation trigger. "Build
      it" is the wrong verdict for a component whose recipient mechanism cannot exist on the live
      vehicle; without this exit such a row nags forever and trains its readers to ignore the queue.
    * ``BUILD_ELSEWHERE_UNWIRED`` — the component is already BUILT, so "not even designed" is simply
      false about it. Its debt is real but it is a WIRING debt, drained by :func:`built_elsewhere_unwired`.
      Leaving it here would mis-state the work as a build and hide the measured harm.

    THE RETIREMENT EXIT IS EVIDENCE-GATED (ddm_qd1, #899 residual, 2026-08-03). ``BUILD_RETIRED``
    is the only exit that is a true DRAIN: a retired row leaves this queue and appears in no other
    one. MEASURED before this fix — a ``retired-with-reason`` row whose mandatory reactivation
    trigger was MISSING (a shape :func:`record_required_component` REFUSES on write) still exited,
    so a hand-appended unverified retirement silently drained real debt. The exit now requires
    ``record_integrity == RECORD_VERIFIED``, i.e. the row must re-pass the write-path predicate NOW.

    This is the same fail-closed principle as :func:`_effective_grade_rank`, applied to the other
    thing evidence buys: there it gates a RANK, here it gates an EXIT. The grade-5 exit deliberately
    stays unconditional because it is not a drain — an unverified grade-5 row remains visible in
    :func:`built_elsewhere_unwired` (sorted last, typed by ``record_integrity``), which is MEASURED,
    so gating it here would only hide it twice rather than once.
    """
    idx = _build_index()
    return tuple(r for r in read_required_components(path)
                 if r["component"] not in idx
                 and not _exits_debt_queue(r))


def _exits_debt_queue(row: dict) -> bool:
    """Whether a declared row has EARNED its exit from the grade-4 debt queue.

    Split out of :func:`not_even_designed` so the two exits can be reasoned about separately: one
    is a hand-off to a sibling queue, the other is a drain, and only a drain needs evidence.
    """
    grade = row.get("grade")
    if grade == BUILD_ELSEWHERE_UNWIRED:
        # Hand-off, not a drain: the row stays visible in built_elsewhere_unwired().
        return True
    if grade == BUILD_RETIRED:
        # Drain: it leaves every queue, so it must still re-pass the write-path predicate.
        return row.get("record_integrity") == RECORD_VERIFIED
    return False


def built_elsewhere_unwired(path: Path | None = None) -> tuple[dict, ...]:
    """Declared components that EXIST and have run elsewhere but have no live call site (grade 5).

    The #864 P0 queue. Separate from :func:`not_even_designed` because the work is different in
    kind: these are drained by WIRING, not by building, and every row carries the two measured
    scalars and a DERIVED, SIGN-CHECKED advantage over its live recipient (refused otherwise).

    ORDER IS ``(fire_order, component)`` — NOT harm. An earlier version of this docstring claimed
    the queue was "ranked by quantified harm"; it never was, and ranking it that way would be a
    guess wearing a number. ``harm_advantage`` is dimensionless and therefore NOT comparable across
    rows measured on different axes: a 2x win on an axis worth 0.29 S beats a 100x win on an axis
    worth 1e-9 S. Ranking by realized harm needs each row's advantage converted into S units, which
    is a per-row measurement no caller supplies today. The inputs are recorded so a future ranker
    can do that honestly; until then the caller orders the work with ``fire_order``.

    A row drops off automatically once the component acquires a lever factory — the registry
    decides, not a memo — exactly as the grade-4 queue behaves.

    VERIFIED ROWS COME FIRST (ddm_ri1, #899). A row that does not re-pass the write-path harm clause
    is still returned — dropping it would be signal loss, and it may be a real wiring debt someone
    simply has not measured yet — but it cannot lead a queue whose whole premise is measured present
    loss. It is typed ``record_integrity`` so a consumer reads a declaration as a declaration.
    """
    idx = _build_index()
    rows = [r for r in read_required_components(path)
            if r.get("grade") == BUILD_ELSEWHERE_UNWIRED and r["component"] not in idx]
    rows.sort(key=lambda r: (0 if r.get("record_integrity") == RECORD_VERIFIED else 1,
                             _fire_order_key(r), r["component"]))
    return tuple(rows)


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
        # The two DECLARED-with-evidence grades keep their OWN grade; everything else is coerced to
        # not-even-designed because a component with no factory cannot honestly report as built.
        # Reporting a retired row as not-even-designed would re-assert the debt its adjudication
        # closed (a retirement silently becoming a build order); reporting a grade-5 row that way
        # would mis-state a WIRING debt as a BUILD debt and drop its measured harm on the floor.
        declared = r.get("grade")
        rows.append({
            "component": r["component"],
            "grade": (declared if declared in (BUILD_RETIRED, BUILD_ELSEWHERE_UNWIRED)
                      else BUILD_NOT_DESIGNED),
            "module": None,
            "trainer": None, "missing_flags": [], "label_drift": False,
            "activation_state": "not-registered", "needed_by": r.get("needed_by"),
            "owner": r.get("owner"), "fire_order": r.get("fire_order"),
            "missing_mechanism": r.get("missing_mechanism"), "consumer": r.get("consumer"),
            "notes": r.get("notes"), "live_recipient": r.get("live_recipient"),
            "measured_comparison": r.get("measured_comparison"),
            # The signed evidence travels WITH the row. Recording a measurement that the
            # operator-facing surface drops would re-create, inside this very module, the
            # built-but-unsurfaced class the grade exists to name.
            "live_measured": r.get("live_measured"),
            "candidate_measured": r.get("candidate_measured"),
            "metric_direction": r.get("metric_direction"),
            "harm_advantage": r.get("harm_advantage"),
            # What the reader is allowed to believe about this stored row (ddm_ri1, #899).
            "record_integrity": r.get("record_integrity"),
            "record_integrity_reason": r.get("record_integrity_reason"),
        })
    for row in rows:
        row["sort_rank"] = _effective_grade_rank(row)
    rows.sort(key=lambda r: (r["sort_rank"], _fire_order_key(r), r["component"]))
    return rows


def _effective_grade_rank(row: dict) -> int:
    """Read order for one report row — the declared rank, DEMOTED when its evidence does not hold.

    ``built-elsewhere-unwired`` holds rank 0 for exactly one reason, stated in ``BUILD_GRADE_ORDER``
    and in the write path's own refusal text: it is the only grade whose declaration is refused
    without a MEASURED present loss. A stored row that does not re-pass that check has not earned
    that reason, so it must not lead the operator-facing report above rows carrying real debt.

    The demotion TARGET is derived, not chosen: the refusal text says such a row "is
    indistinguishable from built-never-fired, which is dormant and harmless", so it reads at
    built-never-fired's rank. The row KEEPS its declared grade label — silently relabelling it would
    swap one false record for another; only the read ORDER changes, and ``record_integrity`` says
    why. Rows of every other grade keep their rank: their rank is not evidence-justified, so an
    unverified charter there is a debt with a thin description, not an unproven harm claim.
    """
    declared = BUILD_GRADE_ORDER.get(row.get("grade"), 9)
    # FAIL-CLOSED: rank 0 requires POSITIVE evidence of verification, so an absent
    # ``record_integrity`` demotes exactly like a failed one. Factory-derived rows never carry this
    # grade (``build_grade`` cannot return it), so the strict form cannot demote a real build row.
    if (row.get("grade") == BUILD_ELSEWHERE_UNWIRED
            and row.get("record_integrity") != RECORD_VERIFIED):
        return BUILD_GRADE_ORDER[BUILD_NEVER_FIRED]
    return declared


__all__ = [
    "BUILD_DESIGNED_STUB",
    "BUILD_ELSEWHERE_UNWIRED",
    "BUILD_FIRED",
    "BUILD_GRADE_ORDER",
    "BUILD_NEVER_FIRED",
    "BUILD_NOT_DESIGNED",
    "BUILD_RETIRED",
    "EVENT_FIRED",
    "EVENT_FOLDED",
    "EVENT_MEASURED",
    "EVENT_QUEUED",
    "EVENT_RETIRED",
    "LEDGER_PATH",
    "RECORD_DECLARED_UNVERIFIED",
    "RECORD_MALFORMED",
    "RECORD_VERIFIED",
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
    "VALID_RECORD_INTEGRITY",
    "VALID_SIG_AXES",
    "VALID_SIG_LABELS",
    "ActivationStatus",
    "TerminalActivationJoinReceipt",
    "TerminalActivationJoinRow",
    "TerminalJoinDisposition",
    "TerminalJoinStatus",
    "activation_report",
    "activation_status",
    "build_completeness_report",
    "build_grade",
    "built_elsewhere_unwired",
    "compiled_non_default_levers",
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
    "required_component_integrity_summary",
    "terminal_activation_join",
    "verify_required_component_row",
]
