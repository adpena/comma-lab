#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Schedule-provenance launch gate — every hardcoded-epoch schedule TRIGGER in an
emitted launch config must be EVENT-based, LawRef-DERIVED, or a TAGGED fail-safe CAP,
or the launch is REFUSED.

Operator directive 2026-07-09 (verbatim, fury, permanent): *"Fuck pr95... Never do it
again. Add a gate and hook or whatever as appropriate. I have been desperately pushing you
to move from hardcoded epochs to event based and deep math governed and costate
controller."* Sister memory:
``elementwise_audits_launder_structural_cargocult_pr95_skeleton_20260709.md`` (the
mechanism: element-wise cargo-cult audits give every brick an honest disposition while the
PR95 discrete-stage BLUEPRINT — CE→tau_softplus→Muon at proportional boundaries — passes
unexamined). This gate makes the prohibition STRUCTURAL at the ONE surface that launches.

THE PROHIBITION. A ``--*-start-epoch`` token in the emitted launch config gates WHEN a
curriculum stage / loss-form / optimizer-phase / lever BEGINS. When its value is a
hardcoded POSITIVE epoch it is a **NAKED PRIMARY EPOCH** — the exact PR95-skeleton regression
the operator prohibits — UNLESS it falls in exactly one of three legal classes:

  EVENT_TRIGGERED — a NAMED, co-emitted event-condition SENSOR governs the transition
                    (one of the recognised sensors: ``--curriculum-event-triggered`` /
                    ``--curriculum-nucleus-guard`` / ``--plateau-trigger`` /
                    ``--closed-loop-control``), DECLARED in the config's
                    ``schedule_governance`` surface. Co-emission of a sensor flag is NOT
                    enough — the binding must be DECLARED, so a stray ``--curriculum-event-
                    triggered`` cannot silently launder every hardcoded stage boundary
                    (the exact laundering the operator caught: crucible_v6 emits BOTH
                    ``--tau-softplus-start-epoch 300`` AND ``--curriculum-event-triggered``
                    and the 300 is still a naked PR95-position boundary).
  DERIVED         — the token's value is carried in ``constants_manifest.json`` with a
                    LawRef ``equation_id`` (the #351 constant-compiler surface): the value
                    is a law of measured state, not a hand-typed epoch.
  FAIL_SAFE_CAP   — the token is an explicitly TAGGED requirement-B secondary fail-safe cap
                    (``schedule_governance`` class=cap + the governing event sensor it backs
                    up + a real rationale). A fixed epoch may exist ONLY as such a cap.

Anything else REFUSES the launch, naming the exact token + the three legal paths.

DESIGN NOTES (all NON-NEGOTIABLE for a launch gate).
  * DATA-DRIVEN registry — the set of schedule-WHEN flags is parsed from the trainer's REAL
    argparse (``--*-start-epoch`` names), never a hand-typed list that rots.
  * VALUE-AWARE — a ``-start-epoch`` emitted at ``<= 0`` (or ``None``) means "from ep1 /
    disabled" = ALWAYS-ON, not a schedule transition; only a POSITIVE epoch is a candidate
    primary trigger. This precisely isolates genuine hardcoded WHEN-boundaries.
  * CAPS ARE TAGGED, NOT INFERRED — a fail-safe cap is legal only via an explicit
    ``schedule_governance`` declaration naming its governing event sensor + rationale; a
    prose provenance comment is NOT a declaration (the incumbent's ``muon_start_epoch``
    ProvenancedValue says "fail-safe cap" in prose but carries no structured tag → it FAILS,
    correctly, until the restart tags it).

means != ends: this gates a MEANS (a launch config). Only a byte-closed n600 exact row
< 0.19110 from ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── the recognised event-condition SENSORS (the named sensors an EVENT/CAP declaration may
#    cite; each must be a real trainer flag AND co-emitted for the binding to be valid).
#    Sourced from the levelset trainer's event surface (train_levelset_witness_realized_
#    through_R_mlx.py): ep_loss-plateau convergence + per-class critical-nucleus guard +
#    d_seg-slope plateau + d_seg-trend closed-loop. `forfeit`/`meat-exit`/per-class-λ are
#    CODE sensors with no CLI flag (they cannot be a launch token's declared sensor).
RECOGNISED_EVENT_SENSORS: frozenset[str] = frozenset({
    "--curriculum-event-triggered",   # ep_loss-plateau convergence (fixed epochs become caps)
    "--curriculum-nucleus-guard",     # per-class critical-nucleus sensor gating CE->tau fire
    "--plateau-trigger",              # d_seg descent-slope plateau (base trainer)
    "--closed-loop-control",          # d_seg-trend monitor -> bounded lever control
    # operator override 2026-07-08 (S4 R1): the three per-transition sensor->start WIRING flags. Each
    # is the CLI surface of a wired code/telemetry sensor (powerlaw_meat / lane_nucleus /
    # annulus_plateau) that fires its transition; the paired --<x>-start-epoch is its backstop cap.
    # An EVENT declaration names its own start-event flag as `sensor`; a CAP declaration names the
    # paired start-event flag it backs up.
    "--muon-start-event",             # muon entry <- powerlaw_meat exit (+ REV-B nucleation positive control)
    "--lane-band-start-event",        # lane-band start <- lane-class critical nucleus (born + formed)
    "--seg-chroma-boundary-start-event",  # seg-chroma start <- annulus_frac plateau (formed boundary)
})

# ── schedule-class labels (a token lands in exactly one).
CLASS_EVENT = "EVENT_TRIGGERED"
CLASS_DERIVED = "DERIVED"
CLASS_CAP = "FAIL_SAFE_CAP"
CLASS_NAKED = "NAKED_PRIMARY_EPOCH"

# ── the data-driven registry pattern: a schedule-WHEN TRIGGER flag ends in ``-start-epoch``
#    (tau-softplus-start-epoch / l7-start-epoch / muon-start-epoch / lane-band-start-epoch /
#    seg-chroma-boundary-start-epoch / ...). Duration/shape/dwell params (--anneal-epochs,
#    --*-warmup-epochs, --*-hold-frac, --curriculum-min-stage-epochs, --stage-transition-
#    rewarmup-epochs) are NOT WHEN-triggers — they govern shape/length (a req-T value-
#    provenance concern), not WHEN a stage begins, so they are out of this gate's scope.
_ADD_ARG_RE = re.compile(r'add_argument\(\s*"(--[a-z0-9-]+)"')
_START_EPOCH_RE = re.compile(r"-start-epoch$")
# operator override 2026-07-08 (S4 R1): a per-transition sensor->start WIRING flag ends in
# ``-start-event`` (--muon-start-event / --lane-band-start-event / --seg-chroma-boundary-start-event).
# Its VALUE is a sensor NAME (not an epoch); when co-emitted with a class=event/role=fires governance
# declaration the transition is EVENT_TRIGGERED (wired) and the paired --*-start-epoch is its backstop.
_START_EVENT_RE = re.compile(r"-start-event$")

# placeholder rationales are rejected in a cap declaration (per Catalog #287 discipline).
_PLACEHOLDER_RATIONALE = re.compile(
    r"^\s*(?:<[^>]*>|todo|tbd|reason|rationale|xxx|n/a|placeholder)\.?\s*$", re.IGNORECASE)

LEGAL_PATHS_MSG = (
    "Schedule-provenance gate (operator 2026-07-09 'move from hardcoded epochs to event "
    "based and deep math governed'): each --*-start-epoch schedule TRIGGER must take ONE of "
    "three legal forms —\n"
    "  (1) EVENT-TRIGGERED — declare it in the config's schedule_governance surface as "
    "{class: event, sensor: <--curriculum-event-triggered | --curriculum-nucleus-guard | "
    "--plateau-trigger | --closed-loop-control>} AND co-emit that sensor (a named sensor "
    "governs the transition; co-emission alone does NOT launder a hardcoded boundary);\n"
    "  (2) DERIVED — compile the value from a LawRef (equation_id) so it lands in "
    "constants_manifest.json (the #351 constant-compiler; the value is a law of measured "
    "state, not a hand-typed epoch);\n"
    "  (3) FAIL-SAFE CAP — declare it in schedule_governance as {class: cap, sensor: "
    "<governing event sensor>, rationale: <why the fixed epoch is only a req-B backstop the "
    "event fires before>}.\n"
    "A hardcoded positive epoch in NONE of the three = the PR95-skeleton regression this "
    "gate refuses."
)


# ────────────────────────────── data-driven registry ──────────────────────────────
def schedule_when_flags(trainer_text: str) -> frozenset[str]:
    """The registry: every ``--*-start-epoch`` flag name in the trainer's argparse text.

    Data-driven (parsed from the REAL ``add_argument(...)`` calls) so it never rots vs a
    hand-typed list. These are the WHEN-a-stage/loss-form/optimizer-phase/lever-BEGINS
    triggers the operator's prohibition targets."""
    return frozenset(
        f for f in _ADD_ARG_RE.findall(str(trainer_text or ""))
        if _START_EPOCH_RE.search(f))


def event_start_flags(trainer_text: str) -> frozenset[str]:
    """The sister registry: every ``--*-start-event`` sensor-wiring flag in the trainer's argparse.

    Data-driven (parsed from the REAL ``add_argument(...)`` calls) so it never rots. These are the
    operator-override 2026-07-08 per-transition wiring flags whose VALUE is a sensor NAME; a co-emitted
    one carrying a class=event/role=fires governance declaration makes its transition EVENT_TRIGGERED
    (its paired ``--*-start-epoch`` becomes the FAIL_SAFE_CAP backstop)."""
    return frozenset(
        f for f in _ADD_ARG_RE.findall(str(trainer_text or ""))
        if _START_EVENT_RE.search(f))


def flag_to_manifest_key(flag: str) -> str:
    """``--tau-softplus-start-epoch`` -> ``tau_softplus_start_epoch`` (the autoconfig delta /
    constants_manifest key convention: leading dashes stripped, ``-`` -> ``_``)."""
    return str(flag or "").lstrip("-").replace("-", "_")


def parse_epoch_value(value: object) -> int | None:
    """The POSITIVE integer epoch a token carries, else ``None`` (``None`` / non-int /
    ``<= 0`` = "from ep1 / disabled" = always-on, NOT a schedule transition → not a
    candidate primary trigger)."""
    if value is None:
        return None
    try:
        iv = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return iv if iv > 0 else None


def extra_flag_pairs(tokens: list[str]) -> list[tuple[str, object]]:
    """Parse a shell-split ``--extra-trainer-flags`` token list into ``(flag, value)`` pairs
    (``--foo bar`` / ``--foo=bar`` / bare ``--foo`` -> value ``None``), matching how the
    launcher composes the final emitted argv. Non-flag tokens are treated as the prior
    flag's value."""
    pairs: list[tuple[str, object]] = []
    i = 0
    toks = list(tokens or [])
    while i < len(toks):
        t = str(toks[i])
        if t.startswith("--"):
            if "=" in t:
                k, v = t.split("=", 1)
                pairs.append((k, v))
                i += 1
                continue
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if nxt is not None and not str(nxt).startswith("--"):
                pairs.append((t, nxt))
                i += 2
            else:
                pairs.append((t, None))
                i += 1
        else:
            i += 1
    return pairs


# ────────────────────────────── governance-declaration schema ──────────────────────────────
def validate_governance_entry(flag: str, entry: object, emitted_flags: set[str]
                              ) -> tuple[bool, str]:
    """Validate ONE ``schedule_governance[flag]`` declaration. Returns ``(ok, detail)``.

    A valid entry is a mapping ``{class: event|cap, sensor: <recognised sensor>, [rationale]}``
    whose ``sensor`` is a recognised event sensor AND is co-emitted; a ``cap`` additionally
    needs a real (non-placeholder, >= 8 char) rationale (a fixed epoch is legal ONLY as a
    tagged backstop, and the tag must say what event backs it up + why)."""
    if not isinstance(entry, dict):
        return False, "declaration must be a mapping {class, sensor, [rationale]}"
    cls = str(entry.get("class", "")).strip().lower()
    if cls not in ("event", "cap"):
        return False, f"class={entry.get('class')!r} must be 'event' or 'cap'"
    sensor = str(entry.get("sensor", "")).strip()
    if sensor not in RECOGNISED_EVENT_SENSORS:
        return False, (f"sensor={entry.get('sensor')!r} is not a recognised event sensor "
                       f"{sorted(RECOGNISED_EVENT_SENSORS)}")
    if sensor not in emitted_flags:
        return False, (f"declared sensor {sensor} is NOT co-emitted in this launch "
                       "(the governing sensor must actually fire)")
    # S4 R1 (2026-07-08): the event/backstop ROLE discriminator. Optional (defaults from class), but
    # when present it must AGREE with the class so a CAP's sensor cannot be misread as a firing claim.
    role = str(entry.get("role", "")).strip().lower()
    if role:
        if role not in ("fires", "backstops"):
            return False, f"role={entry.get('role')!r} must be 'fires' or 'backstops' (S4 R1)"
        if cls == "event" and role != "fires":
            return False, "class=event requires role=fires (S4 R1: an event IS the sensor-fired trigger)"
        if cls == "cap" and role != "backstops":
            return False, "class=cap requires role=backstops (S4 R1: a cap backs up an event)"
    if cls == "cap":
        rationale = str(entry.get("rationale", "")).strip()
        if len(rationale) < 8 or _PLACEHOLDER_RATIONALE.match(rationale):
            return False, ("cap declaration needs a real rationale (>= 8 chars, "
                           "non-placeholder) naming why the fixed epoch is only a backstop")
    eff_role = role or ("fires" if cls == "event" else "backstops")
    return True, f"class={cls} role={eff_role} governed by {sensor}"


# ────────────────────────────── per-token classification ──────────────────────────────
@dataclass(frozen=True)
class TokenVerdict:
    """Classification of one emitted schedule-WHEN token."""
    flag: str
    value: object
    cls: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.cls != CLASS_NAKED


def classify_token(flag: str, value: object, *, emitted_flags: set[str],
                   manifest_keys: set[str], governance: dict) -> TokenVerdict:
    """Classify ONE emitted schedule-WHEN token into EVENT / DERIVED / CAP / NAKED.

    Precedence: DERIVED (a LawRef in constants_manifest) → a valid schedule_governance tag
    (EVENT or CAP) → NAKED. A hardcoded positive epoch with none of the three is the PR95-
    skeleton regression the operator prohibits."""
    key = flag_to_manifest_key(flag)
    if key in manifest_keys:
        return TokenVerdict(flag, value, CLASS_DERIVED,
                            f"value compiled from a LawRef (constants_manifest[{key!r}])")
    entry = (governance or {}).get(flag)
    if entry is not None:
        ok, detail = validate_governance_entry(flag, entry, emitted_flags)
        if ok:
            cls = CLASS_EVENT if str(entry.get("class")).lower() == "event" else CLASS_CAP
            return TokenVerdict(flag, value, cls, detail)
        return TokenVerdict(flag, value, CLASS_NAKED,
                            f"invalid schedule_governance declaration: {detail}")
    return TokenVerdict(
        flag, value, CLASS_NAKED,
        "hardcoded positive epoch with no LawRef in constants_manifest, no event-sensor "
        "binding, and no fail-safe-cap tag (naked PR95-style primary schedule trigger)")


def classify_launch(emitted_pairs: list[tuple[str, object]], *, registry: frozenset[str],
                    manifest_keys: set[str], governance: dict,
                    event_registry: "frozenset[str] | None" = None) -> list[TokenVerdict]:
    """Classify every EMITTED schedule-WHEN token with a POSITIVE epoch value.

    ``emitted_pairs`` = the final composed argv as ``(flag, value)`` (derived-config tokens +
    parsed extra-trainer-flags). Only tokens whose flag is in the data-driven ``registry``
    AND whose value parses to a POSITIVE epoch are candidate triggers (a ``<= 0`` / ``None``
    start-epoch is always-on/disabled, never a schedule transition).

    ``event_registry`` (operator override 2026-07-08 / S4 R1): the ``--*-start-event`` sensor-wiring
    flags (from :func:`event_start_flags`). A co-emitted one carrying a class=event/role=fires
    governance declaration is ALSO classified (as EVENT_TRIGGERED) so the gate report surfaces the
    wired transition alongside its FAIL_SAFE_CAP backstop. Its value is a sensor NAME (not an epoch),
    so it is NOT epoch-filtered; an undeclared start-event flag is skipped (it is not a naked epoch)."""
    emitted_flags = {str(f) for f, _ in emitted_pairs}
    verdicts: list[TokenVerdict] = []
    for flag, value in emitted_pairs:
        if str(flag) not in registry:
            continue
        if parse_epoch_value(value) is None:
            continue
        verdicts.append(classify_token(
            str(flag), value, emitted_flags=emitted_flags,
            manifest_keys=set(manifest_keys or ()), governance=governance or {}))
    # S4 R1: surface the co-emitted, DECLARED start-event wirings as EVENT_TRIGGERED transitions.
    ereg = event_registry or frozenset()
    gov = governance or {}
    for flag, value in emitted_pairs:
        if str(flag) not in ereg or gov.get(str(flag)) is None:
            continue
        verdicts.append(classify_token(
            str(flag), value, emitted_flags=emitted_flags,
            manifest_keys=set(manifest_keys or ()), governance=gov))
    return verdicts


def gate_report(verdicts: list[TokenVerdict]) -> tuple[bool, list[TokenVerdict], str]:
    """``(ok, violations, table)`` for the launcher. ``ok`` is False iff any token is NAKED;
    ``table`` is a human-readable classification block (every classified token + its class +
    detail) — the operator-facing audit surface + the restart's to-fix spec."""
    violations = [v for v in verdicts if not v.ok]
    lines = ["# schedule-provenance gate: classifying emitted --*-start-epoch triggers "
             "(EVENT / DERIVED / CAP required; naked hardcoded epoch REFUSED)"]
    if not verdicts:
        lines.append("#   (no positive-epoch schedule triggers emitted — nothing to gate)")
    for v in verdicts:
        mark = "OK  " if v.ok else "FAIL"
        lines.append(f"#   [{mark}] {v.flag} {v.value}  [{v.cls}] — {v.detail}")
    if violations:
        lines.append(f"#   => {len(violations)} NAKED primary-epoch trigger(s) — REFUSE "
                     "(this list IS the restart's to-fix spec)")
    return (not violations, violations, "\n".join(lines))
