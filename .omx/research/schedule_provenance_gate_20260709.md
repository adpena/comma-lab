# Schedule-provenance enforcement — LAUNCHER STRICT gate + drift-detector leg (2026-07-09)

STORES CONSULTED: memory `elementwise_audits_launder_structural_cargocult_pr95_skeleton_20260709`
(the spec + escalation) · `.omx/research/t5_crucible/ORCHESTRATION_LEDGER.md` (req B fail-safe caps,
req T value-provenance ladder, the seal-round-2 BLOCKER on family-scaled epochs, the LawRef/#351
constants_manifest surface) · `tools/launch_witness_run.py` (existing gate wiring: mem-preflight b1,
throughput, system-admission, C13 dup-flag) · `tools/triality_drift_detector.py` (verdict-scope leg
= the leg template) · `src/tac/witness_autoconfig.py` (`_CRUCIBLE_V6_DELTAS`, `to_trainer_flags`,
`constants_manifest`) · trainer argparse `experiments/train_levelset_witness_realized_through_R_mlx.py`
(the `--*-start-epoch` schedule family + the event sensors).

[no-triality] — apparatus/enforcement landing; the code commits carry normal triality treatment.

## Source
Operator 2026-07-09, verbatim fury (3rd recurrence of the class): *"Fuck pr95... Never do it again.
Add a gate and hook or whatever as appropriate. I have been desperately pushing you to move from
hardcoded epochs to event based and deep math governed and costate controller."* Both prohibitions
made STRUCTURAL: (A) PR95 schedule/curriculum inheritance; (B) hardcoded epochs as PRIMARY schedule
triggers. The event machinery (#315 event-triggered curriculum, #302 nucleus guard, costate) already
existed — the failure was launch configs silently regressing to epoch scripts. The gate makes the
built machinery MANDATORY at the only surface that matters (emission).

## Layer 1 — LAUNCHER-PATH STRICT gate (`tools/schedule_provenance_gate.py`)
Data-driven registry: `schedule_when_flags()` parses the trainer argparse for `--*-start-epoch`
names (the WHEN-a-stage/loss-form/optimizer-phase/lever-BEGINS triggers). Duration/dwell/shape params
(`--anneal-epochs`, `--*-warmup-epochs`, `--*-hold-frac`, `--curriculum-min-stage-epochs`) are NOT
WHEN-triggers → out of scope (req-T value concerns, not the prohibition). VALUE-AWARE: a `-start-epoch`
emitted at `<= 0`/`None` is always-on/disabled → not a trigger; only a POSITIVE epoch is classified.

Each emitted positive-epoch trigger is classified into exactly one of THREE legal classes, or REFUSE:
- **EVENT_TRIGGERED** — a NAMED, co-emitted sensor governs it, DECLARED in the config's
  `schedule_governance` surface `{class: event, sensor: <one of --curriculum-event-triggered /
  --curriculum-nucleus-guard / --plateau-trigger / --closed-loop-control>}`. **Co-emission is NOT
  enough** — the binding must be DECLARED (this is the exact laundering the operator caught:
  crucible emits BOTH `--tau-softplus-start-epoch 300` AND `--curriculum-event-triggered`, and the
  300 is still a naked PR95-position boundary).
- **DERIVED** — the value is carried in `constants_manifest.json` with a LawRef `equation_id`
  (the #351 constant-compiler): a law of measured state, not a hand-typed epoch.
- **FAIL_SAFE_CAP** — an explicitly TAGGED req-B secondary cap: `schedule_governance {class: cap,
  sensor: <governing event>, rationale: <why the fixed epoch is only a backstop the event fires
  before>}` (rationale ≥ 8 chars, non-placeholder). A fixed epoch is legal ONLY as such a tagged cap.

Wiring (`tools/launch_witness_run.py`, step b0.5, after `constants_manifest.json` write, before the
mem/admission gates): classify the composed argv (`cfg.to_trainer_flags` + parsed extra-trainer-flags);
print the classification TABLE always; **REFUSE (rc=6)** on any NAKED trigger on a REAL launch; ADVISORY
(print table, proceed) on `--dry-run` and under the new `--skip-schedule-provenance-gate`. Fail-OPEN on
infra/import error (a gate crash must never wedge the ONE launch path). The live run-1 resumes from its
FROZEN `launch.sh` (`bash launch.sh`), NOT through the launcher → the gate does not disturb it.

## The incumbent crucible_v6 violations = THE RESTART's to-fix spec (regression fixture)
Classifying the live `derive_crucible_v6_config(n600, 3000ep)` → **5 NAKED primary-epoch triggers**
(pinned in `test_incumbent_crucible_v6_naked_epoch_set_is_the_restart_to_fix_spec`). The incumbent
carries NO `schedule_governance` surface and its manifest LawRefs are {softmax_temp_end, hosc_beta_end,
lr_anneal_epochs, lr_hold_frac} — none a start-epoch → every start-epoch is naked:

| Token | Value | What it is | Restart fix |
|---|---|---|---|
| `--tau-softplus-start-epoch` | 300 | CE→tau loss-form boundary (PR95's 10% position — operator's named example) | event-govern via nucleus-guard, or derive from a CE-plateau LawRef |
| `--l7-start-epoch` | 3000 | tau→l7 loss-form boundary (l7 demoted; 3000 = never fires) | drop it, or event-govern |
| `--muon-start-epoch` | 726 | AdamW→Muon optimizer-phase switch (a genuine fail-safe cap) | TAG as `{class: cap, sensor: --plateau-trigger, rationale: ...}` |
| `--lane-band-start-epoch` | 350 | analytic lane-band lever-engage | make event-relative (`--curriculum-reanchor-levers`) + declare, or cap-tag |
| `--seg-chroma-boundary-start-epoch` | 300 | chroma-boundary lever-engage (not in the reanchor set) | event-relative + declare, or cap-tag |

The schedule-derivation sibling consumes this table; the restart passes the gate by EVENT-governing /
LawRef-deriving / cap-tagging each of the five (or dropping l7).

## Layer 2 — drift-detector hook leg (`tools/triality_drift_detector.py`, Stop hook)
Scope: commits touching `src/tac/witness_autoconfig.py` / `src/tac/witness_dsl/` / launch-config
recipes. Flags, on ADDED diff lines: (a) a NEW naked hardcoded-epoch schedule param (a `*_start_epoch`
delta assignment or a `--*-start-epoch` argparse default, POSITIVE int) with NO co-added event/derived/
cap provenance token in the window; (b) a NEW PR95-named stage SEQUENCE (a stage/curriculum/schedule
line naming ≥ 2 distinct PR95 stages). Window-granular (matches the other legs), fail-open, same-line
waiver `# SCHEDULE_PROVENANCE_OK:<real rationale>`. Block message names the three legal paths. Sister of
the launcher gate: the launcher refuses EMITTING a naked epoch; the leg flags AUTHORING one.

## Tests + status
`src/tac/tests/test_schedule_provenance_gate.py` — **29 tests** (registry/value/mapping · each of the
3 classes passes · naked REFUSED with named token · sensor-not-co-emitted / missing-rationale /
placeholder / unrecognised-sensor all NAKED · value≤0 & non-schedule flags never classified · the
incumbent 5-violation fixture · leg: file-scope, naked-additions, argparse-default, waiver, gov-cite
suppression, PR95-sequence, opt-out, fail-open, base-classify untouched). All green; ruff F clean on all
touched files; the existing launcher + drift suites (95 tests) still pass.

**means != ends**: pointer contest-CPU 0.19110 UNMOVED — this gate is a MEANS (it forces the schedule
to be event/law-governed); only a byte-closed n600 exact row from `upstream/evaluate.py` moves the pointer.
