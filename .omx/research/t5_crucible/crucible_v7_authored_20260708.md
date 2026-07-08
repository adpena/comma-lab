# crucible_v7 AUTHORED — the FIRST requirement-V-native launch config (2026-07-08)

status: pre-registered-only (NOT reviewed; T3 council + 3×3+structure seal OWED per DRAFT §6).
review_status: pre-registered-only.
author: CRUCIBLE_V7 CONFIG AUTHOR (Opus, build-class).
[no-triality] — this is a research memo; the triality treatment is the CODE landing
(`derive_crucible_v7_config` in `witness_autoconfig.py` + `test_crucible_v7_config.py`),
composing the already-landed DSL Lever factories (SegFormUnifyTau · TailCycles ·
LadderIslandHomotopy) + the typed schema; no new DAG/equation leg is claimed here.
Pointer 0.19110 [contest-CPU] UNMOVED — this config is a MEANS until its byte-closed n600 exact row.

## STORES CONSULTED
`DRAFT_v7_restart_config_synthesis_20260708.md` (§1 violation-resolution · §2 spine · §3 pose ·
§4 A/B · §5 gate chain · §6 open council knobs) · `ORCHESTRATION_LEDGER.md` (last 6 folds =
the landed v7 pieces: UNIFY-TAU · req-V TypedWitnessConfig · schedule-provenance gate · dashboard ·
TAIL_k · LADDER #323) · `src/tac/witness_dsl/typed_config.py` (the TypedWitnessConfig schema;
crucible_v6's `_attach_dsl_program_manifest` = the migration template) ·
`tools/schedule_provenance_gate.py` (the 0-naked classifier + RECOGNISED_EVENT_SENSORS) ·
the levelset trainer argparse (sensor-wiring reality check for the wiring-gap list) ·
`curriculum_dsl.{TailCycles,LadderIslandHomotopy,SegFormUnifyTau}` (the composed Lever factories).

## WHAT LANDED
- `derive_crucible_v7_config(...)` → a `TypedWitnessConfig` (the requirement-V-native artifact:
  authored AS a typed DSL program, argv emitted by `WitnessProgram.compile_trainer_argv`, NO hand
  argv, NO parallel dict assembly). Fail-closed via `.validate_program()`.
- `compile_crucible_v7_config(...)` → `CrucibleV7Compiled{typed, argv, emitted_pairs,
  constants_manifest, dsl_program_manifest, schedule_governance, v6_flags}` — the full compiled
  artifact for the launcher gate chain + the council review surface.
- `diff_crucible_v6_to_v7(...)` + `crucible_v7_wiring_gaps()` — pure helpers (the review table + the
  honest wiring-gap list).
- `test_crucible_v7_config.py` — 25 tests (config construction · 0-naked gate · manifest ·
  mutual-exclusion · pose-verbatim · diff-table stability · unchanged-flag byte-identity · wiring gaps).
- VERIFY: 25/25 v7 + 134 sibling (migration/typed-schema/schedule-gate/autoconfig) + 21 unify-tau
  green; ruff F clean on both touched files.

## AUTHORING FORM (requirement V — how the "no parallel dict assembly" law is honored)
The substrate `base` is the SEALED crucible_v6 emitted flag set, REUSED not retyped (the same reuse
law by which v6 reuses `store_nothing_205`), transformed by the five §1 resolutions. Bare-boolean
`(flag, None)` → `True` (the DSL emitter's bare convention). The emitter is the DSL
`WitnessProgram`, never a parallel `WitnessConfig.to_trainer_flags`. That is the v7-vs-v6 distinction:
v6 emits via the autoconfig dataclass with a typed manifest ATTACHED; v7 IS the typed config.

## THE DIFF-vs-v6 TABLE (the council's review surface; 104 → 126 semantic flags)

### REMOVED (3) — DRAFT §1 resolutions 1+2 + the cosine-only knob
| flag | v6 value | why removed |
|---|---|---|
| `--tau-softplus-start-epoch` | 300 | dissolved into continuous L_τ by `--seg-form-unify-tau` (trainer `validate_seg_form_unify_tau_config` REFUSES both) |
| `--l7-start-epoch` | 3000 | l7 = measured DEFECT; inert under unify (trainer default 800 is bypassed by the unified loss) |
| `--tau-hold-frac` | 0.2 | cosine_hold-only; the geometric anneal has no hold segment |

### CHANGED (3) — the schedule spine + two event caps
| flag | v6 → v7 | class |
|---|---|---|
| `--tau-anneal-shape` | cosine_hold → **geometric** | spine (DRAFT §2 continuous form) |
| `--lane-band-start-epoch` | 350 → **500** | tagged fail-safe CAP |
| `--seg-chroma-boundary-start-epoch` | 300 → **450** | tagged fail-safe CAP |

### ADDED (25) — the three composable v7 Lever factories
- `--seg-form-unify-tau` (1) — continuous L_τ = τ·logsumexp(φ/τ)−φ_y; removes the last PR95 stage bone.
- `--tail-*` (7): cycles-max **2** [council_pending] · start-epoch 0 (→ always-on, NOT a gated trigger) ·
  cycle-floor-epochs 387.09 · dwell-min 237 · tau-halving 0.5 · lr-prop-tau 1.0 · stop-marginal-s 1e-4.
- `--ladder-*` (17): island-homotopy ON + movable dilation-GO (r0/birth/hold/anneal/λ-gate) + lane
  curve-prior (r0/birth/hold/anneal/λ-gate/dash-gate) + gate-softness/release-coeff/sigma-eff/
  max-step-px/refresh-every [council_pending on gate thresholds per DRAFT §6.3].

### UNCHANGED — everything else (β-end 10.0, LR pin 1000/1.0, fused-R, reanchor, min-stage 250,
structured init, seed-islands eased, persistence/amplify, weight-entropy 15, verdict-batch 32,
ckpt/stage-checkpoints, epochs 3000, chroma weight/margin-band, `--muon-*`, `--curriculum-*`)
carries over BYTE-IDENTICAL (asserted per-flag in `test_unchanged_flags_are_byte_identical_to_v6`).
Pose block VERBATIM: `--w-pose 1.0 --pose-carrier --pose-carrier-source generated
--pose-carrier-residual-mode table`.

## SCHEDULE-PROVENANCE GATE OUTCOME: 0 NAKED
Three emitted positive `--*-start-epoch` triggers, all classified **FAIL_SAFE_CAP** (the other two
v6 nakeds — tau-softplus/l7 — DELETED; `--tail-start-epoch 0` is always-on, not gated):

| trigger | value | class | governing wired sensor |
|---|---|---|---|
| `--muon-start-epoch` | 726 | FAIL_SAFE_CAP | `--curriculum-event-triggered` |
| `--lane-band-start-epoch` | 500 | FAIL_SAFE_CAP | `--curriculum-nucleus-guard` |
| `--seg-chroma-boundary-start-epoch` | 450 | FAIL_SAFE_CAP | `--curriculum-event-triggered` |

## HONEST CLASSIFICATION + WIRING-GAP LIST (council input, NOT a failure)
The draft frames these as "EVENT + CAP". VERIFIED against the live trainer: NONE of the three is
runtime-fired by a sensor — each engages at a FIXED epoch gate. The wired event sensors
(`--curriculum-event-triggered` / `--curriculum-nucleus-guard`) govern the CE→tau readiness hand-off,
and #333 annulus telemetry is OBSERVABILITY-ONLY. So all three are honestly TAGGED FAIL-SAFE CAPS
citing the co-emitted wired controller they back up; the DRAFT-intended specific sensor→start wiring
is an OWED build:
1. **muon**: `powerlaw_meat` exit is a CODE sensor with no CLI flag AND does not move
   `muon_start_epoch` (fixed gate, trainer ~L3216). OWED: a powerlaw_meat→muon-entry event trigger.
2. **lane-band**: `--curriculum-nucleus-guard` governs CE→tau readiness (~L1948/L4951), NOT
   `lane_band_start_epoch` (fixed gate ~L3302). OWED: nucleus→lane-band-start wiring.
3. **chroma**: the #333 `annulus_frac` telemetry is never read into training (~L4956-4959), so it
   cannot fire `seg_chroma_boundary_start` (fixed gate ~L3672). OWED: an annulus-plateau→chroma-start
   event trigger + a recognised CLI sensor for it.

## COUNCIL_PENDING KNOBS (DRAFT §6 — where I picked the draft's proposal + tagged it)
1. The three event-sensor choices + cap values (§6.1) — I use `--curriculum-event-triggered` (muon,
   chroma) + `--curriculum-nucleus-guard` (lane-band); caps 726 / 500 / 450. **Council to confirm.**
2. TAIL `k_max` (§6.2) — proposed **2** (`_CRUCIBLE_V7_TAIL_CYCLES_MAX`); stop-marginal-s 1e-4 (default).
3. LADDER gate thresholds (§6.3) — builder defaults (λ-gates OPEN 0.0, release-coeff 0.95, sigma-eff
   1.5). **Council to accept or recalibrate from run-1's per-class λ trace.**
4. STRUCTURE ROUND (§6.4, binding, blinded) — OWED to the council; not performed here.
5. run-1 stop point (§6.5) — OWED to the council.

## CONSTANTS MANIFEST
Inherited from v6 (v7's base reuses the SAME LawRef-resolved constants): `softmax_temp_end` 0.31 ·
`hosc_beta_end` 10.0 · `lr_anneal_epochs` 1000 · `lr_hold_frac` 1.0. Re-derivation triggers unchanged
from v6. These are req-T value-provenance constants (τ_end MEASURED-ANCHOR; β/LR DERIVED-AT-CONFIG),
orthogonal to the schedule-provenance gate (which classifies `--*-start-epoch` WHEN-triggers only).

## PROVENANCE / STATUS
Pre-registered, NOT reviewed. Gate chain at relaunch (DRAFT §5): DSL program manifest (req V, rc=7) →
schedule-provenance gate (0 naked, rc=6) → memory preflight → admission governor → dashboard renders
the new stage kinds additively. NO launch performed (run-1 pid 63069 untouched; run dirs read-only).
means != ends: only a byte-closed `upstream/evaluate.py` n600 row < 0.19110 moves the pointer.
