# Duty-ticket revision — real ep725-fork compiles (task #563, 2026-07-19)

Lane: `duty_queue_fire_tickets_20260719` (revision packages appended INSIDE the composer's
sealed ticket dirs as `revision_claude_20260719/`; nothing of the composer's is mutated).

**Pointer `0.1910828242 [contest-CPU]` UNMOVED.** Everything below is MEANS: pure/$0
compiles, static checks, and adjudications. NOTHING was launched — no trainer, no governed
dry-run, no paid dispatch. Per `docs/operating_manual_craft_handoff.md` §5, every number
below is labeled MEASURED / DERIVED / ADJUDICATED.

## Verdict table

| ticket | duty | verdict | full_dsl_compile_hash (MEASURED at worktree HEAD, see summary json) |
|---|---:|---|---|
| 01 `DsegAwareTaper` | 78.9% | **CANNOT-RESOLVE(ep725-fork) → RE-SCOPED-TO-FRESH-RUN**; the fresh ISO pair's full compile is itself **BLOCKED by a pre-existing V9-432 LawRef defect** (below) | control `v9_cgauge_432` typed `bfaa639a6af1e8a2…`, treatment `v9_cgauge_432_taper_off` typed `0ea55dfa63f9f713…`; full hashes **null (honest)** — `LawRef recompute hosc_beta_end 10.0 != 3.177` refuses BOTH arms |
| 02 `HorizonWeightedMargin` | 47.3% | **READY-FOR-GAUNTLET** (ep725 fork pair compiled) | OFF `c49087ce7077c9abe3f8c06dafe85a909b6867da3319ac5c33a4d2ad24fbcb1d` · ON `8ac7b6cd816f4d21d42c477fa9abd0c21904425cbf14dccd5ac627c150565137` |
| 03 `StepNativeActivation` | 34.2% | **READY-FOR-GAUNTLET** (ep725 fork pair compiled; estimand re-scoped 4.0→8.0, see below) | OFF `8c0e962064e4587741f6b24851b0d9f7ebcfd0c38553a73604260ca05dfdaad8` · ON `028bd07d738ab4f21a68d8c38abcbcec9f611228507e2e13a2a4b78a6aef4b56` |
| 04 `#497 curvelet matched-bytes` | n/a | **READY-FOR-GAUNTLET(fresh-arms; wrapper fail-closed repairs landed)** with named OWED items | unchanged pure-compile pair from the composer memo remains valid (`be96e749…` / `7ed49820…`); wrapper repaired, sha in adjudication.json |

Typed config hashes (MEASURED): hwm off `9169f422bb6c9018…` / on `77cabaa0d796765f…`;
step off `551e23903e719e40…` / on `12548534eac4942e…`. Full resolved argv per arm:
`<ticket>/revision_claude_20260719/compiled_pair.json`. Schedule-provenance gate: **rc0,
0 violations, all four arms** (MEASURED via `schedule_provenance_gate.classify_launch`
against the real trainer parser).

## The fork programs (new DSL spec — the levers live in the DSL, never hand flags)

`src/tac/witness_dsl/spec_v9c3_duty_ab_20260719.py::compile_v9c3_duty_ab_config(ticket, arm)`
— four programs `v9c3_duty_{hwm,step}_{off,on}` (v9c3 = the operator's naming lock for
ep725 restarts; `v9c3_dev_*` out-dirs are maturity-DEV, never pointer-promotable):

- **Base** = the v9c2 run's own launch.sh flag-for-flag (config of record, sha
  `eecbb745e7f7…`, MEASURED read-only from the sacred run dir).
- **Fork** = weights-only from the BANKED ep725 BEST EMA (custody re-verified byte-exact:
  460,448 B, sha `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef`,
  sidecar epoch 725 / d_seg 0.003457972208658854) + `--warm-start-weights-only` +
  explicit `--warm-start-epoch 726`. The bank npz is a 59-key EMA deploy snapshot with NO
  optimizer/RNG/event state (campaign meta-review 2026-07-18) — a fresh-optimizer fork is
  the only honest resume form; no banked-control exception claimed.
- **#518 recipe**: `ResumeLRWarmup()` (27 ep, DERIVED `ceil(2/(1−0.999)/75)` via
  `adam_v_variance_warmup_length_v1`, overrides the config-of-record 8) +
  `ForkEmaClearance()` + automatic #517 tau/beta/seg_form positioning pre-v0 + item-9
  ramp-end reorient. `ForkHeadSolve`/`MarginStepCap` OFF in BOTH arms (never-fired; a
  measured cap does not exist — named OPEN with its unlock, not silently defaulted).
- **Events re-anchored to resume epoch + geometry** (the #518/confound-engine law):
  **Muon OMITTED entirely** (donor's absolute ep726 Muon = the MEASURED confound engine;
  pure-AdamW measurement plant; #217 stays OPEN); **pose-blind** (`--w-pose 0`, compute
  gates kept, conditioning gate not composed — reproduces the trunk's realized ep700-725
  loss physics AND removes arm-divergent sigma_min_plateau engagement); c2 surgical loss
  terms (phase/satisfice/subpix @700) kept at record values in BOTH arms; l7 parked at
  1001 (record's own form); `--anneal-epochs 1000` pins schedule continuity; run ends AT
  ep1000 (the linear hosc-beta anneal has NO clamp past the plant — DERIVED from
  `_hosc_beta_for_epoch`; ending at 1000 keeps every schedule inside its designed range).
- **Geometry (all DERIVED)**: resume 726 → rewarmup 27 → treatment boundary 753 →
  primary window ep775-850 (K=4 verdict points @ cadence 25) → secondary ep775-1000
  (K=10). ~274 trained epochs/arm; arms strictly sequential (projected peak 67.6 GiB
  each, memory-preflight SAFE ≤ 89.6 GiB ceiling — static projection recorded per arm;
  the launcher's dry-start bench is the MEASURED authority and is an open blocker).

**Signed one-lever deltas (MEASURED on the canonicalized resolved argv):**
- step OFF→ON: `{"--hosc-beta-end": ["4.0", "8.0"]}` — exactly one flag.
- hwm OFF→ON: the 8 `--seg-horizon-margin-*`/row flags incl. `-derived-live` (one Lever).
- hwm_off vs step_off: physics-identical (out-dir custody only).

## Blocker → resolution mapping (the composer's 4)

1. **Taper control-comparability** → ADJUDICATED INVALID as an ep725 contrast: the lever
   is STRUCTURAL epoch-0 (its factory docstring + the trainer F2 resume-divergence
   basis-change class), and the ep725 trunk was trained WITHOUT it, so the canonical ON
   control does not exist at ep725; adding it at a fork measures add-shock, not lever
   value (and `--warm-start-weights-only` would auto-allow the drift → silent confound).
   Charter's honest path taken: re-scope to the fresh ISO pair (contrast custody: iso
   one_lever_delta=True, argv_diff = the 4 taper flags). NEW HONEST FINDING: both
   432-family arms REFUSE the #406 full compile at current HEAD —
   `V9_432_HOSC_BETA_END_LAWREF_RECOMPUTE_DEFECT` (equation recomputes 10.0 vs emitted
   3.177; the CLAUDE.md 2026-07-15 reconciliation's OWED custody debt surfaced live).
   Ticket 1 cannot carry a real full compile hash until that upstream row is repaired.
2. **HWM 0.15 share + MarginStepCap from an ep725 boundary receipt** → the 0.15 stays a
   DERIVED-LIVE **request**; the trainer resolves the realized weight at the ep753 frozen
   all-P boundary scan via `w_h=(0.15/0.85)·L_other/max(L_h,eps)` and EMITS the receipt
   `<out_dir>/horizon_margin_boundary_receipt.json` (schema
   `hwm_v9_stage_share_boundary.v1`) — the receipt path is now pinned in the ticket; the
   boundary was moved 726→753 (resume + LawRef rewarmup window) so the scan sees
   re-conditioned losses, provenance in the constants row. MarginStepCap: NO measured cap
   exists → OFF in BOTH arms (arm-identical ⇒ no confound), with the named unlock
   (derive from the OFF twin's rewarmup-window update-norm/margin telemetry). Thresholds:
   pre-registered from MEASURED noise (below), no round numbers.
3. **Step #518 reanchor + v0-position + response-window receipt** → composed: annealed
   path only (`StepNativeActivation` guard refuses fixed beta), one-flag delta, #517
   positions beta pre-v0 (control 3.1772 = `1+3·725/999`; treatment 6.0801 =
   `1+7·725/999` — DERIVED from the trainer's own formula; note 3.1772 IS the composer's
   "3.177 sealed endpoint": it is the schedule value at the fork epoch). The v0 verdict +
   the 726-753 transient rows are the response receipt; verdict windows start
   post-rewarmup. ESTIMAND RE-SCOPE (honest): the declared "3.177→8.0" is the V9-mod19
   custody surface; on this vehicle the realized contrast is beta-END 4.0→8.0 (jump
   +2.90 at 726 is disclosed as part of the treatment). The v9c2 launch.sh header's own
   adjudication ("step-native is the alternative arm for the NEXT fresh arm") is quoted
   in the package (AF2): the fork measures sharpen-a-trained-trunk, the fresh arm
   measures train-under-the-endpoint — recorded as different estimands, neither
   substituted for the other.
4. **#497 wrapper repair** → ep725-fork framing ADJUDICATED not-applicable (basis lever =
   structural class; fresh arms CORRECT, so "no resume/#518 input" is not a defect).
   Wrapper repairs LANDED (fail-closed): absent c2 run dir no longer counts as
   quiescent; liveness-inspection failure (any exception, incl. PermissionError) is now
   fail-closed; `--skip-c2-gate` requires `--operator-go`; arm mutual-exclusion refusal
   (rc4) before dry/fire. Smoke-tested refusal paths: rc5 (skip without GO), rc3
   (not-quiescent) — no launcher invocation occurred. One-factor: the contrast is
   one-LEVER; the composite bundle (basis+orient+AA) bounds the verdict scope — recorded.
   OWED (named, not silently deferred): enforced equal-byte/finalize chain; term-share
   engagement telemetry beyond front_end; current dry receipts.

## Pre-registered thresholds (derived, not round)

Noise floor MEASURED from the donor's own n600 advisory verdict series (ep750-1075, 14
points, read-only from the sacred run.log), second-difference estimator
(`sigma = sqrt(mean(d2²)/6)`, robust variant `median|d2|/(0.6745·√6)`, larger used):
`sigma ≈ 1.88e-5` d_seg/point. Paired 95% half-widths (`1.959963985·√(2/K)·sigma`;
√2 = independence upper bound — paired same-seed arms can only shrink it):
**h95(K=4) = 2.600e-5 d_seg** (primary, ep775-850) · **h95(K=10) = 1.645e-5** (secondary,
ep775-1000). Rules: FIRED-PAYS `mean Δ ≤ −h95(4)` with same-sign secondary confirmation;
FIRED-HURTS mirrored; FIRED-NEUTRAL `|Δ| < h95` on both; sign-disagreement ⇒
INDETERMINATE, no verdict. Power context: the HWM adverse prior's own ceiling
(1.2e-4–2.4e-4 d_seg) is 4.7-9.4× h95(4). Admissibility preconditions + positive-control
sentinels (resume_lr_rewarmup row @726/27, baseline_v0_schedule_positioned with the
predicted beta, lever_engage fired rows, HWM boundary receipt resolved_weight>0, ep_loss>0,
no confound alarms, ema_warmup=false in windows, argv-diff==signed delta, sequential arms,
resumable + per-stage checkpoints) are in every `compiled_pair.json`. Disclosed asymmetry:
the ON-arm lever engagement re-treats the spike guard at 753 (standard engagement
discipline — part of the treatment mechanism, stated for the gauntlet).

## Adverse findings surfaced pre-fire (the launch-must-surface law)

AF1 HWM "measured WEAK" header quote; AF2 step fresh-arm disposition quote; AF3 donor
Muon confound (this fork omits Muon); AF4 estimand re-scope 3.177/4.0→8.0; AF5 the donor
FROZE beta at 3.1772 from ep726 (FEED-fm Muon freeze) and never realized the 3.177→4.0
tail — the Muon-free arms anneal it live, shared by both arms. All embedded in the spec
manifest (`adverse_findings_surfaced`) so they travel WITH the config.

## What the gauntlet still owes before any GO (open blockers, fail-closed)

- launcher `--config v9c3_duty_*` registration (3-line c2 pattern) + hash-matched GREEN
  `--dry-start` receipt per arm (proves warm-fork weight-shape load via resume_ok +
  measures sec/ep + peak RSS) — `V9C3_COMPOSED_BENCH_NOT_MEASURED` rows.
- bank checkpoint byte-custody re-verification at fire time on main.
- operator GO per arm (CONTAINMENT).

## Artifacts

- `src/tac/witness_dsl/spec_v9c3_duty_ab_20260719.py` (the four programs; DSL-held).
- `tools/revise_duty_tickets_ep725_fork_20260719.py` (deterministic materializer).
- `tools/fire_curvelet_matched_bytes_ab_p0_497.py` (fail-closed repairs).
- `.omx/research/duty_queue_fire_tickets_20260719/{01..04}/revision_claude_20260719/`
  (compiled_pair.json / adjudication.json + verdict_card.md + revision_manifest.json)
  + `revision_claude_20260719_summary.json` at the ticket root.

## Triality

- **DSL**: the fork programs are compiled `WitnessProgram`s (never-invent-flags validated
  against the real parser; lever-owned constants; #406 full-compile custody per arm).
- **DAG/apparatus**: this memo + the ticket packages are the FEED-ready record; the DAG
  append rides the MAIN landing (worktree DAG would conflict — same convention as #518 R7).
- **equations**: no new law registered — the revision CONSUMES
  `adam_v_variance_warmup_length_v1`, `warm_start_schedule_reconstruction_v1`,
  `horizon_weighted_margin_hinge_v1`, `step_native_activation_edge_optimality_v1`; the
  threshold derivation is an application of measured noise, PROVISIONAL until the paired
  run measures true paired noise (its own residual, named).

## STORES CONSULTED

Charter prompt; composer memo + all 4 sealed ticket dirs + `main_source_custody.json`;
`src/tac/witness_dsl/duty_queue_fire_tickets_20260719.py`; #518 build memo
`p0_resume_warmup_geometry_build_20260717.md`; memories
`vehicle_naming_v9c_warm_lineage_v10_reserved_capstone_20260718`,
`warm_start_resume_must_adapt_events_to_resume_epoch_and_geometry_20260718`;
`spec_c2_surgical_20260716.py` (pattern of record); `spec_v9_cgauge.py` ISO wrappers;
the trainer argparse + consumers at the cited lines (`_hosc_beta_for_epoch`, HWM
derived-live block + receipt writer, #518 flag block, resume block); the sacred run dir
(read-only: launch.sh, run.log verdict series, levelset_best.json); bank custody
(byte-exact sha re-verified); `CLAUDE.md`; `docs/operating_manual_craft_handoff.md`.

## Self-review (rounds used: 2 of 5)

Round 1 findings (fixed pre-commit): LawRef double-ownership of the HWM start-epoch
(spec row removed; lever custody is the single owner + `_merge_lever_constant_manifests`
folds it for the schedule gate); `--warm-start-epoch` NAKED at the schedule gate (custody
row added, evaluator-legal mode); run length 1028 extrapolated the unclamped beta anneal
past its endpoint (run ends at the plant end 1000); a typo'd sha in a custody note
(corrected to the verified literal). Round 2: clean on the checks run (compile x4 rc0
gates, delta verification, wrapper refusal rcs, ruff F, py_compile).
