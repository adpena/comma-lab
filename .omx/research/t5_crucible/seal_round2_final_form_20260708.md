---
doc_type: t5_crucible_seal_round2_final_form_verdict
role: SEAL ROUND 2 of 3 (counter was 1/3 after round-1 CLEAN on the final form) — a FRESH Opus pass
  over the FINAL LAUNCHING FORM (v6.4 draft + §14.5 LawRef migration + LawRef-consuming crucible_v6 +
  launcher constants-manifest + the LawRef mechanism). Requirement U: THREE co-equal lenses
  (A bugs · B deep-math · C confound). Fresh angles (distinct from round 1): end-to-end timeline
  simulation on the final constants · the B1 V=5 co-predicate AS-BUILT · run-1 operational readiness
  (P7 preview) · Lens-C fresh cuts (EMA/liveness · inter-instrument τ_end source chain · negative space).
date: 2026-07-08
target: DRAFT_OPTIMAL_STACK_v6_20260708.md @ v6.4 + §14.5 + the CODE at HEAD (working tree CLEAN on the
  five touched files ⇒ working == HEAD).
verdict: CLEAN — 0 BLOCKER + 0 MAJOR + 0 MINOR + 2 nits (both INSTANCE characterizations changing no
  decision, consumer-read number, or build item). Counter advances 1/3 → 2/3.
axis: all numbers [macOS-CPU/MLX advisory]; pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
review_status: fresh-eyes-reviewed(2, final-form) — this verifier authored none of v1..v6.4, the LawRef
  migration, the probes, or the prior verdicts. Every load-bearing claim is [re-executed] (live launcher
  dry-run; full-precision numeric timeline replica of the three trainer schedule laws; artifact sha;
  targeted pytest; source read of the event-trigger + co-predicate + spike-guard + liveness code) or
  [verified-on-disk].
verdict_scope: no NEW negative asserted; the two nits are INSTANCE-level characterizations.
  # VERDICT_SCOPE_OK:no-new-negatives-only-characterizations
---
`[no-triality]`

STORES CONSULTED: seal_round1_final_form_20260708.md (FULL — the regression spec + its disclosed
meta-confound note, both re-verified) · DRAFT_OPTIMAL_STACK_v6_20260708.md (FULL — §0.1–0.4, §1.0/1.1/
1.4a, §2.2f/g/2.5, §3.4, §4c, §5, §7c, §9, §10, §14/14.3/14.4/14.5) · seal_round1_v62_verdict (the six
v6.2 findings) · lawref_migration + #351 memos (via §14.5) · LIVE SOURCE this session:
src/tac/witness_autoconfig.py (derive_crucible_v6_config L1438–1617, value-identity guard L1521–1528,
_CRUCIBLE_V6_DELTAS L1346–1413) · src/tac/witness_dsl/lawref_builtins.py (FULL — 4 CONSUMED + 6 LIBRARY)
· tools/launch_witness_run.py (crucible route L597–601, write_constants_manifest L457–485, mem-preflight
+ system-admission L1018–1097, shadow-observer auto-start L742–767/1168–1175) ·
experiments/train_levelset_witness_realized_through_R_mlx.py (_softmax_temp_for_epoch L2341 ·
_hosc_beta_for_epoch L2318 · _lr_scheduled_for_epoch L2389 · _stage_converged L1798 ·
_evt_resolve_seg_form L1854 · spike_guard_mode default L7548 · liveness stamping L2482/5079) ·
src/tac/witness_control/trace_probes.py (copredicate_backtest / forfeit_matched_backtest) ·
tools/witness_trace_probes.py (the manual advisory CLI) · shadow_controller.py (co-predicate consumption
= 0). EXECUTED $0: ONE real launcher --dry-run (n600/3000ep, --config crucible_v6) · a full-precision
NUMERIC timeline replica of τ/β/LR from the trainer source · artifact sha (9898d8d7…) · pytest 54/54
(crucible+lawref subset) · ruff F clean (5 files). NO launches, NO training; run dirs read-only.

# SEAL ROUND 2 (FINAL FORM v6.4 + LawRef) — CLEAN. Counter 1/3 → 2/3.

## §0 REGRESSION — round-1 CLEAN + all v6.2/6.3/6.4 fixes + the meta-confound note — ALL HOLD [re-executed]

Dry-run n600/3000ep: **106/106 flags**, NO C13 refusal, NO duplicate long flag, mem-preflight **67.61
GiB PASS**, system-admission **ADMIT**, constants_manifest.json written (4 LawRef constants). Emitted
tokens verified present: τ block (`--softmax-temp-end 0.31 --softmax-temp-start 1.0 --tau-anneal-shape
cosine_hold --anneal-epochs 3000 --tau-hold-frac 0.2`) · β (`--hosc-beta 1.0 --hosc-beta-anneal linear
--hosc-beta-end 10.0`) · LR (`--lr 1e-3 --lr-end 1e-4 --lr-anneal-epochs 1000 --lr-hold-frac 1.0`) ·
anchors (`--muon-start-epoch 726 --tau-softplus-start-epoch 300 --l7-start-epoch 3000 --curriculum-min-
stage-epochs 250`) · pose block MAJOR-A2 (`--w-pose 1.0 --pose-carrier --pose-carrier-residual-mode
table --pose-carrier-source generated`) · MAJOR-1 (`--curriculum-plateau-windows` ABSENT, grep 0) ·
`--per-group-grad-clip` injected · `--fused-r-kernel --mlx-device gpu`. The round-1 meta-confound note
(value-identity guard certifies IDENTITY not CORRECTNESS) STANDS — and the correctness it does not cover
was independently re-derived THIS round (§1 timeline). 54/54 tests green; ruff F clean.

## §1 LENS B — deep-math meat hunt + THE END-TO-END TIMELINE SIMULATION. 0 findings.

Full-precision numeric replica of the three trainer schedule laws at the FINAL constants (start τ=1.0):

| ep | τ (cosine_hold, ae3000, hold0.2) | β (linear, frozen@726) | AdamW-LR (own den1000) | event |
|---:|---:|---:|---:|---|
| 1 | 1.0000000 | 1.00000 | 1.000e-3 | run start |
| 250 | 0.7458915 | 1.74725 | 8.690e-4 | CE min-stage floor |
| 300 | 0.6566263 | 1.89730 | 8.153e-4 | CE→tau CAP (tau_softplus start) |
| 350 | 0.5672502 | 2.04735 | 7.551e-4 | lane band start |
| 600 | **0.3100030** | 2.79760 | 4.115e-4 | τ descent COMPLETE |
| 601 | **0.3100000** | 2.80060 | 4.101e-4 | τ HELD 0.31 |
| 650 | **0.3100000** | 2.94765 | 3.462e-4 | ctrl EMA-best anchor |
| 675 | **0.3100000** | 3.02267 | 3.153e-4 | forfeit fire band (advisory) |
| 726 | **0.3100000** | **3.175725** (freeze) | (Muon) | Muon swap CAP + β freeze |
| 3000 | 0.3100000 | 3.175725 | (Muon) | run end |

COHERENCE (all TRUE): τ descent-complete ep600 ≤ fire-band-lo 670 ≤ fire ep675 < Muon cap 726 — the
anneal-complete precondition + ep675 fire band + cap 726 COHERE (the forfeit arm's τ-anneal precondition
is satisfied at ep600; β's separate-axis incompleteness is the disclosed M2 finding, not a gate) ·
τ(675)=τ(726)=0.31 EXACTLY · β(726)=3.175725 (≈control 3.177, ≤0.1%) · LR(725)=2.580e-4 (≈control
2.57e-4, bit-identical pin) · settle 3/ν=237.10, tail-cycle 387.10, k_max gross floor(2350/387.1)=**6** /
net floor((2350−250)/387.1)=**5** · CE settle 3/ν_CE=150.3 < min-stage 250 ⇒ the 250 floor binds the
CE→tau earliest fire (the L1590 prov "may fire earlier (150.3)" is the raw settle, superseded by the
min-stage floor — no decision, the CAP is 300) · dwell τ_d=ln(1.275)/0.012653=19.2 ep, min-stage 250 =
13.0× margin. Crossing arithmetic unmoved by the migration (§0.2 reproduces). NO contradiction between
any window / trigger / cap. The LIBRARY LawRefs (settle 237.09 / tail 387.09 / Conley 1.7505/1.3018 /
adaptive-ε 0.7) are NOT in CRUCIBLE_V6_CONSUMED_LAWREFS ⇒ absent from launch.sh — cannot corrupt the config.

## §2 LENS A — recursive adversarial (bugs) + B1 CO-PREDICATE AS-BUILT + OPERATIONAL READINESS. 0 findings; 1 nit.

- **B1 V=5 co-predicate AS-BUILT [source-read].** The co-predicate lives in `trace_probes.copredicate_
  backtest(rows, v_window=4)` (parameterized; V=5 achievable) but its SOLE caller is the MANUAL CLI
  `tools/witness_trace_probes.py`, which invokes it at the DEFAULT **v_window=4** (the shipped-anchor
  form) with a stale `--s-star` default 1.4154e-5 (the OLD ν; HARMLESS — the draft §2.2f proves the fire
  epoch is INVARIANT for any s* ∈ [6.9e-6, 1.42e-5] ⇒ same ep675). The `#247` shadow observer does NOT
  consume the co-predicate (grep count 0). So **V=5 is an unrun DESIGN SPEC** (a margin over the V=4
  baseline), which is EXACTLY what the draft claims ("V=5 binds ONLY the B1 spec; no trainer flag
  exists") — and the forfeit LIVE firing is `B-INJ` (pre-GO, unbuilt). In run-1 the operative TAU→FIN
  transition is the HARD cap `--muon-start-epoch 726` (the live event-trigger `_evt_resolve_seg_form`
  does CE→tau→l7 via EP_LOSS plateau windows=4, NOT a forfeit exit; l7-start 3000 ⇒ never). Run-1's GO
  does NOT depend on any co-predicate firing. **NIT-1 (characterization):** the config docstring
  L1468–1469 lists "cap 726 + the event-triggered co-predicate (V=5)" as the armed-with-fallback,
  which conflates (a) the live EP_LOSS event-trigger (windows=4, CE→tau only) with (b) the manual d_seg
  V=5 advisory (run at V=4), and no explicit run-1 operational-plan line names WHO runs the advisory at
  WHAT cadence (implicitly: manual `witness_trace_probes.py` at checkpoints). The OPERATIVE fallback —
  the hard cap 726 — is emitted and correct; changes no decision, emitted flag, or number.
- **Run-1 operational readiness [re-executed].** Launcher: 106/106, mem-preflight PASS, system-admission
  ADMIT, constants_manifest written, shadow observer AUTO-STARTS (L1173), dashboard AUTO-TRACKS (no
  manual repoint). B-DET n600 composite determinism check = **owed pre-GO, NOT in the launcher preflight
  chain** (grep 0) — disclosed non-blocking (§7c/§10). Resume: `derive_crucible_v6_config` runs at LAUNCH
  time only; launch.sh carries the FROZEN literals (zero LawRef tokens), so crash-resume re-runs the same
  frozen argv WITHOUT re-resolving LawRefs — bit-faithful and CORRECT (re-resolution would risk artifact
  drift; the frozen argv avoids it). 5 signal-ledger generators spot-checked (distinct from round-1's
  req-P trace): SC-3 artifact `tau_knee_ptau2` sha 9898d8d7 launch_tau 0.31 + within-band field present ✓
  · SC-16 `pdz_deadzone_census.json`(12689B)+`.gi_hists.npz`(3680B) ✓ · SC-17 `pcon_conley_backtest.json`
  +`.ledger.npz` ✓ · SC-21/B-DET = pre-GO (owed) · F26/SC-3-live = ZERO trainer callers of
  `flip_margin_quantiles` (⇒ run-1 CHECKPOINT-granularity via the offline `witness_tau_mq_confirm.py`) —
  both disclosed. No invented flag, no dup flag.

## §3 LENS C — CONFOUND HUNT (req U). 0 findings; 1 nit.

- **(a) EMA-vs-live / spike-guard-freeze regression [source-read].** The emitted config does NOT set
  `--spike-guard-mode` ⇒ trainer default **`rollback`** (L7548), NOT the legacy median-freeze deadlock
  mode — the confound-gate #397 class + the L5 `spike-guard-median-freeze` memory are GUARDED. Liveness
  fields `accepted_frac` + `weights_stepped` are stamped on every verdict row (L2482/5079) and a
  below-floor accepted-frac classifies FROZEN → action STOP (L5248) — no consumer of run-1 telemetry can
  misread a frozen/EMA state as live descent. (EMA-shadow verdict + liveness are base-inherited from
  store_nothing_205, not a v6 change — regression-verified, in scope.)
- **(b) Inter-instrument consistency of τ_end 0.31 [re-executed].** THREE copies — manifest
  `softmax_temp_end.value` + `inputs[0].launch_tau` = 0.31 (artifact sha 9898d8d7) · launch.sh
  `--softmax-temp-end 0.31` · draft §1.4a — all trace to ONE source chain: artifact `launch_tau` →
  `resolve_flag_dict_constants` → value-identity guard asserts `== _CRUCIBLE_V6_DELTAS[0.31]` (fail-CLOSED
  on drift, L1523) → pb/alb pin → emitted token. Same for β 10.0 / LR 1000 / hold 1.0 (literal-in-LawRef
  == delta). No third divergent copy; the guard makes divergence a RAISE, not a silent emit.
- **(c) Negative space (what run-1 does NOT measure that §8/ILC assumes) [re-walked].** F26 (SC-3 live
  m_q per verdict cadence) is UNBUILT ⇒ run-1 emits m_q at CHECKPOINT granularity, not verdict cadence;
  the §4c SC-3 "per verdict cadence" wording is a run-2 capability (its consumers — τ_end live-law
  promotion / P-TAU2 live f_target / TAIL τ*_k live — are ALL run-2; run-1 uses the fail-safe CONSTANT
  0.31), disclosed in §10 F26. Run-1 EVSI is pose-dominated (~0.044 of ~0.05), so d_seg-telemetry
  granularity is not decision-critical for run-1's instrument value. **NIT-2 = round-1's NIT-2 restated**
  (value-identity guard = identity not correctness) — STANDS, and the correctness was re-derived in §1.

## §4 VERDICT + COUNTER

**CLEAN — 0 BLOCKER + 0 MAJOR + 0 MINOR + 2 nits (NIT-1: the config-docstring "V=5 co-predicate"
fallback phrasing conflates the live EP_LOSS event-trigger with the manual d_seg V=5 advisory and omits
an explicit advisory-cadence line — the OPERATIVE fallback cap 726 is emitted+correct, run-1's GO does
not depend on the advisory; NIT-2: round-1's identity-not-correctness note, correctness re-derived here).
Neither nit changes a decision, consumer-read number, or build item. Counter 1/3 → 2/3.** Honesty both
directions: the end-to-end timeline is fully coherent at the final constants (τ/β/LR reproduce at every
anchor epoch; anneal-complete ep600 < fire ep675 < cap 726; TAIL k_max 6/5; CE-floor binds at min-stage
250; dwell 13× margin); the B1 V=5 co-predicate is honestly an unrun SPEC with B-INJ pre-GO and cap 726
the operative fallback; the τ_end source chain is single-sourced + fail-closed; the spike-guard-freeze
confound is guarded (rollback default + liveness stamping); every load-bearing number reproduces at full
precision and the LawRef migration moves ZERO emitted values. One characterization worth a docstring
tighten (NIT-1) — not launch-blocking. Nothing to fix.

Pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
