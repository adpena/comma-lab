# DEFERRAL LEDGER — every open deferral, with owner + trigger. The apparatus remembers.
<!-- Surfaced by costate_digest at session start (wiring: deferral-ledger digest line).
     Rule: no deferral exists outside this file + a task with a named trigger.
     Created 2026-07-08 per operator: "You deferred too much and now it's orphaned." -->

STORES CONSULTED: ORCHESTRATION_LEDGER folds this arc · task list #24–#355 · council
positions/synthesis · builder final reports (wirings/GPU-verdict/tail/ladder/unify-tau).

| # | Deferral | Trigger (named, not "later") | Owner/task | Status 2026-07-09 |
|---|---|---|---|---|
| D1 | GPU-vs-CPU verdict agreement probe (n600, mod32cap ep650) | ↻#393-D4: **#385 chosen-chain (v7.5.2\|v8) PRE-LAUNCH** — runs BEFORE the chosen chain launches so it carries the verdict default the evidence supports (was "run-1 GOVERNED STOP → relaunch"; run-1/v7-relaunch SUPERSEDED by the dual-chain) | #355-adjacent; fold into #385 pre-launch checklist — **OWNER STRING ALSO STALE, reassign with the trigger** | ⚠ **ARMED-BUT-TRIGGER-STALE** (flagged `ddm_iv1` 2026-08-03): the trigger "#385 chosen-chain (v7.5.2\|v8) PRE-LAUNCH" was itself SUPERSEDED by the TR1 pivot, so this row **can never fire as written** — armed rows with dead triggers are silent orphans (`m37` staleness-as-named-confound). Substrate intact: ckpt `levelset_witness_ema_BEST.npz` (458,622 B), `--verdict-device {cpu,gpu}` wired at `train_levelset_witness_realized_through_R_mlx.py:18419` (11 consuming sites). **$0 code read RUN 2026-08-03 (`ddm_iv1`) — ANSWER: the TR1 line BYPASSES this lever entirely.** `verdict_device`/`--verdict-device` occurs in exactly **10 of 10,683** tracked `.py`, **all** in the levelset/witness family (`train_levelset_witness_realized_through_R_mlx.py`, `train_witness_realized_through_R_mlx.py`, `witness_autoconfig.py`, `witness_control/gpu_verdict.py`, `witness_dsl/{curriculum_dsl,typed_config}.py`, + 4 tests). All **12** TR1-line builders/receivers/probes (`ddm_v4d_*`, `inflate_runner_v4{b,c,d}*`, `ddm_pu2_*`, `ddm_dc1_*`, `ddm_gr1_*`) score **0 hits** for `gpu_verdict`/`witness_control`/`verdict_device`. ⇒ **DISPOSITION: RETIRE-WITH-REACTIVATION.** Fire-condition: a levelset-family trainer re-enters the live line. The separate *throughput-on-the-contended-evaluator-slot* need is REAL but needs a **fresh TR1-native row** — it is not this one. |
| D2 | S6-R1 knee re-derive (m_q90 → τ_end on eikonal-0.01 ckpt) | ↻#393-D4: **#385 chosen-chain first converged ckpt** — PRE-SEAL gate item (was "run-1 stop converged ckpt"; dual-chain superseded) | synthesis §C | ARMED (re-pointed) |
| D3 | Event-mode resume determinism for the 3 transition wirings | ~~retired~~ CLOSED by the canonical resume registry (2b7332f4b/8d349088d): all 3 gates round-trip (fire-state + chroma detector window), static gate-coverage test, crash-resume bit-identity test | resume-registry builder | **CLOSED-CONFIRMED 2026-07-09** (#393-D4; 2026-07-08 close verified, no reopen) |
| D4 | Serializer hardening | ~~done~~ LANDED a8ebcd12a/c35979b97 (rc=7 post-commit blob check + --patch-file exact-intent staging + contract #405) | #354 ✅ | **CLOSED-CONFIRMED 2026-07-09** (#393-D4; 2026-07-08 close verified, no reopen) |
| D5 | fp16 cf-feats cache (bank-6 memory-for-speed) | #355 compute audit assesses envelope; v7.1 arm with measured gate chain | #296 | QUEUED-W-TRIGGER |
| D6 | Async-verdict memory reclaim (subprocess/killpg) | v7 relaunch config decision — assess WITH #355 audit (same envelope math) | #330 | FOLDED→#355 review |
| D7 | #314 pose-carrier-source inheritance bug | VERIFY whether req-V typed migration structurally fixed it — check at v7.2 compile (pose flags asserted verbatim, so v7 SAFE either way; the BUG remains for other families) | #314 + v7.2 compile check | CHECK-AT-COMPILE |
| D8 | Attribution-tool consolidation (#255 "during the run") | ↻#393-D4: **chosen-chain launch +1 day** (the tools get used in anger then; was "v7 relaunch +1d"; dual-chain superseded) | #255 | TRIGGER-SET (re-pointed) |
| D9 | GPU-verdict default promotion | D1 evidence table | #355 | GATED-ON-D1 |
| D10 | marimo/molab contest #347 | ✅ RESOLVED-VERIFIED 2026-07-10 (`marimo_linkfix_20260710.md` §RESOLUTION) | operator / sibling Marimo viz agent / marimo-linkfix | **RESOLVED: entry = "The Witness Machine" (`adpena/witness-machine` → molab github-proxy URL, now landed in `paper/README.md`). Published PRE-deadline (f58568d 21:51 PST + be424ad 23:08 PST 07-09); molab-runtime bootstrap was broken (repo-checkout assumption) → repaired in place POST-deadline (f111248, 05:22 PST 07-10, same URL). E2E VERIFIED via agent-browser: URL renders, console clean, static preview shows 42/42 cached outputs 0 errors, bundle asset byte-exact, molab-faithful cold-bootstrap run-all exit 0 / 0 errors. ONLY blocked step: server run-all needs molab sign-in (anonymous forbidden) — not attempted per no-credentials rule. Evidence: `marimo_linkfix_evidence_20260710/`.** |
| D11 | Dashboard WHY/HOW copy rework + TRIALITY redesign (hidden tabs) | post-relaunch polish window | #343 / #267 | PARKED-W-OWNER |
| D12 | Hook quote-exemption tuning + activation-ledger PosixPath wart | next apparatus-maintenance batch (bundle) | new task on next batch | BUNDLED |
| D13 | S5 staged-arms attribution fallback | fires ONLY if v7 trajectory anomalous + attribution fails | synthesis §ARBITRATION | CONDITIONAL |
| D14 | v7.1: self-triggered τ *pulled into v7 already* — residual = octave-LR coupling refinements if builder chose clock-LR | τ-advance builder memo | in flight | PENDING-BUILDER |
| D15 | micro-batch-pairs 2-4x (route/drop logit-adjust conflict + n600 d_seg A/B under RSS waterfill) | ↻#393-D4: **chosen-chain .1 arm AFTER its baseline trajectory exists** (A/B needs the comparator; was "v7.1 after v7 baseline"; dual-chain superseded) | #313-followup; seal reviews | QUEUED-W-TRIGGER (re-pointed) |
| D16 | Metal #212 unbuilt fused kernels (persistence-pool, margin-map, curvelet) | ↻#393-D4: next compute-facet batch (#252 standing program) — rank by **chosen-chain** per-term cost after ep100 telemetry (was "v7 ep100"; dual-chain superseded) | #252 | QUEUED-W-TRIGGER (re-pointed) |
| D17 | safe-compile certified regions (our deterministic mx.compile) as a training lever | ↻#393-D4: **chosen-chain .1 arm alongside D15** (needs the chosen-chain baseline comparator; was "v7.1/v7 baseline"; dual-chain superseded); certified-region manifest = the evidence gate | #252 | **v2-WIRED** (mlx_safe_compile v2, tests 63): auto-discovery (AST+trace) sweep = 9 SAFE elementwise + 1 reduction; HOT LOOP WIRED into levelset `_act` (hosc, flag-flip, default-OFF BYTE-IDENTICAL verified 0.0); per-chip fingerprint {chip,build,mlx} REFUSES stale/wrong-device certs (launcher b2 + resolve); failed-cert→kernel-candidate pipeline LIVE. **v1 harness device-bug FIXED** (bit-eq ran on default device, not `device=`). MEASURED CPU: 8/9 SAFE regions bit-eq=0, **hosc fp-contracts 6e-8 on CPU** (auto kernel-candidate rank0); GPU hosc=0@n=32 (v1) but coverage-open. **DEFERRED to the chosen-chain first GOVERNED STOP** (↻#393-D4: was "run-1 stop"; dual-chain superseded): GPU cross-process re-cert WITH fingerprint (canonical manifest) + whole-step B=8 bench (governor: no GPU bench beside the live chosen-chain run). |
| D18 | Latent-table TRUNCATE-at-export byte-close A/B (truncate `code` to measured k90 columns → real Δbytes vs Δd_seg/Δd_pose) | v7 stop → byte-close truncation A/B on the FINAL ckpt, consuming the `{stage:mod_dim_dynamics}` k90 series (the free-rate lever the mod-32 autopsy suggested; the row already emits `k90_truncate_bytes_estimate` as the ESTIMATE, this measures it) | mod-dim-dynamics telemetry (this batch) → #157/#336 waterfill consumer | ARMED (sensor LANDED: `--mod-dim-dynamics` default-ON emits k90 + per-dim bit-allocation hint per verdict; #299 mod-dim A/B reads the same series). **#377 build-wave 2026-07-09 disposition: STAYS DEFERRED — named blocker = NO v7 FINAL CKPT (run `dry_start`, `best=NONE`, no `levelset_best.json`); the A/B consumes the FINAL ckpt `code` table + the `{stage:mod_dim_dynamics}` k90 series, neither exists yet. NOT an unbuilt lever: the truncation-at-export MACHINERY already exists in `tools/witness_code_pca_byteclose.py` (PCA-K sweep `--ks`, reconstructs codes from DEQUANTIZED PCA rep, measures real Δbytes vs realized-d_seg Pareto via the deploy-faithful realized verdict) + the k90 sensor `src/tac/boundary_math/mod_dim_dynamics.py` (`k_energy_cutoff`→k90, `truncate_bytes_estimate`). Only-missing-wire (deferred, premature to build now): auto-feed measured k90 from telemetry into `--ks` (the sweep already covers any K incl. k90) — trivial, and un-validatable without the ckpt.** |
| D19 | v7.1 SPEED-LEVER bundle (micro-batch A/B · safe-compile hosc flip · D16 pool · megakernel bench) — consolidates D15+D17 execution | ↻#393-D4: stop-time benches at the **chosen-chain first GOVERNED STOP**; trajectory A/Bs after the chosen-chain baseline exists (was "run-1 GOVERNED STOP / v7 baseline"; dual-chain superseded) | task #357 (execution owner) | OPEN (re-pointed 2026-07-09) |
| D20 | Non-gate controllers under the resume registry's static gate | ~~open~~ CLOSED (51ae8ea8d/7834cda31): all 4 folded as FunctionResumable, write single-sourced, keys byte-identical (tested), legacy-compat MEASURED against run-1's REAL sidecar (legacy=True, 0 warnings) | task #358 ✅ | **CLOSED-CONFIRMED 2026-07-09** (#393-D4; 2026-07-08 close verified, no reopen) |
| D21 | Blind-coordinate generic fill in the byte-close receiver (230,904 blind camera px/frame read by NO scorer resize; a video-derived value there earns ZERO score → fill with a deterministic generic rule, rule-118 free) | **post-launch byte-close of the chosen chain** — wire the generic fill into `tools/levelset_byte_close_and_eval.py` (eq `resize_exploit_flip_fix_frontier_v1` anchor `evaluator_resize_blind_coordinate_law_20260710`) | #401 | ARMED (2026-07-10 advisory-fold) |
| D22 | Receiver fail-closed hardening (exact 1,200-frame / 3,662,409,600-byte raw cardinality REFUSE before scoring; fresh-process parse-back byte conservation; no counted-then-ignored bank) | **BEFORE** #399 borrowed-bank dispatch closure AND any v7.5.3/v8 byte-close (short raw = NO-FAKE/compliance failure) | #402 | ARMED (2026-07-10 advisory-fold) |
| D23 | v8 decoded-composite interaction check (per-carrier d_seg gains are NOT additive — SegNet SqueezeExcite + deep-U-Net global path; E-edge cycle-integrability g_ij+g_jk+g_ki=0 OR K-potential decoder) | **BEFORE any v8 increment-1a E-EVENT** (gate input; 6,703 triple-junction blocks make this a live decode constraint) | v8 chain / #385 | GATED (2026-07-10 advisory-fold) |
| D24 | SegNet margin-gradient-tail reproduction (recover the 9.15/3.56/2.21% radii-64/128/192 tails + the J(edge e←region e') same/adjacent/remote block matrix into a hashed receipt) | **next measurement window; BEFORE any edge-locality / no-factorization CLAIM** — this is the `# FORMALIZATION_PENDING` receipt for the segnet-global-dependence law (advisory (d); raw receipt currently ABSENT) | v8 / scorer-geometry owner | ARMED (2026-07-10 advisory-fold) |
| D25 | v7.5.2 amber realization/waiver (advisory P0-5; `witness_stability_amber` preset vs inherited `--grad-clip 1.0`) | **pending operator** decision on the live pilot's admission semantics (`levelset_v752_pilot_20260710T154100Z`) — realize explicitly + verify startup telemetry, OR record an authority-bearing waiver/amendment | launch executor / operator | PENDING-OPERATOR (2026-07-10 advisory-fold) |
| D26 | v7.5.3 exact-D texture home build (6-atom Pose-preprocess kernel; frame1-only, project AFTER last nonlinearity, camera-grid preimage, first-6-Pose bit-stable after fresh raw reload) | v7.5.3 DESIGN/BUILD — gated on the 3 v753 P0s closing (MLX↔NumPy↔inflate one forward; counted bank exclusion; frame1-only home) (eq `posenet_luma_chroma_sensitivity_asymmetry_v1` anchor `pose_preprocess_exact_6atom_null_kernel_20260710`) | v753 build owner | GATED (2026-07-10 advisory-fold) |

---

## D4 HYGIENE PASS (2026-07-09, task #393)

**Why.** The ledger's live thread moved: run-1 (#205, mod32cap) is no longer the in-flight run — the
plan pivoted to the **v7.5.2 ↔ v8 dual-chain** (both sealed; #385 which-to-run GO pending → the CHOSEN
chain launches). Triggers armed on the stale "run-1 GOVERNED STOP / v7 relaunch / v7.1 arm / v7 baseline"
condition would never fire (there is no run-1 to stop). Re-pointed to the live state.

- **Triggers re-pointed to the dual-chain (7 rows):** D1, D2 (→ #385 chosen-chain PRE-LAUNCH),
  D8, D15, D16, D17, D19 (→ chosen-chain launch / first governed stop / baseline). Each cell carries the
  ↻#393-D4 marker + the original condition in parentheses (no signal lost).
- **Same-condition catch-all (not individually rewritten):** D5 ("#355 audit → v7.1 arm") and D18
  ("v7 stop → byte-close FINAL ckpt") also reference v7.1/v7-stop; they inherit the SAME re-point — the
  ".1 arm" / "FINAL ckpt" is the CHOSEN chain's, not v7's. D18's named blocker (NO FINAL CKPT yet) stands
  identically for the chosen chain.
- **3 CLOSED rows formalized:** D3, D4, D20 marked **CLOSED-CONFIRMED 2026-07-09** (2026-07-08 closes
  verified, no reopen).
- **D10 marimo** re-statused: a **parallel Marimo viz agent (sibling)** is finishing it — NOT
  lapsed-by-default. (Deadline TODAY 2026-07-09; the sibling pulls context from origin/main → the
  push-main-regularly discipline is why this file + siblings' work must land promptly.)

**Note routed here from #393-D1(b) (code edits skipped — sibling may own inc1a):** two v8 bare-literal
provenance NOTEs remain owed to SPEC_v8.1 (NOT edited in code this pass): `param_tolerance=0.05`
(`inc1a_harness/decoupling_screen.py:78` — un-derived matching-rule, no rung) + `_SEG_H/_SEG_W=384/512`
(`boundary_math/road_undriv_bulk_field.py` — MEASURED-ANCHOR by value = grid-pin, but bare-literal, no
inline provenance comment). Both are `philosophy_pass_v8_20260709.md` §6 NOTEs → SPEC_v8.1 derive-or-waiver.

Pointer **0.19110 UNMOVED** — hygiene only.


---

## BURN-DOWN PASS (2026-07-10, debt-burn-down executor) — pay-all-owed, no orphaned debt

**Why.** Operator: "Pay all owed, burn it down and don't accumulate debt." Full sweep of `.omx/research`
07-08..10 (fresh-eyes advisories v752/v753/v8 · philosophy_pass_v752/v8 · fullstack_fractal_optimal_synthesis ·
reactivation_campaign_397 · v8_unlock_398a · owed16_bounded_ab_and_drystart) + this file (D1-D20) + the
harness_failure_ledger. **PAID $0 this pass:** terminology gate (18 stale `adpena/tac` URLs → clean, commit
ffccb2725). Everything else is GATED (machine/spec/byte-close) or IN-FLIGHT (builder-owned) — each now carries
a named trigger + owner below so nothing lives only in a memo body. Pointer **0.19110 UNMOVED** — hygiene + inventory only.

**owed-16 re-status (was OWED-BLOCKED, now MEASURED).** `owed16_bounded_ab_and_drystart_20260710.md` §MEASURED
VERDICT: both arms RAN (n600 through-R, `[macOS-CPU advisory · NON-PROMOTABLE]`); realized directional-basis
contribution ≈ **ZERO** (|Δ|≤1.4% of ON at every matched cell, >70× separation from the −48% direct-partition
advisory → verdict robust to any plausible noise floor). Scope **FORMULATION** (bounded warm-start from a
self-orient-ON parent; from-scratch NOT covered). The OWED-BLOCKED cell is SETTLED except the single ep700 ON
cell → D21.

| # | Owed item | Trigger (named) | Owner | Bin | Status |
|---|---|---|---|---|---|
| D21 | owed-16 ep700 ON matched cell (only un-measured cell; verdict already robust at ep675) | ~15 GiB baseline freed → sequential admit when owed16v2 + v752 release RAM (6× governed REFUSE at 119 GiB) | owed-16 A/B (BUILT · queue-ready) | GATED (RAM) | LOW-PRI — re-enters pool only after owed16v2 lands per fullstack FIRE-1; verdict NOT blocked on it |
| D22 | Dual-chain DOC-gaps: P3 end-to-end tolerance ledger (fit→quantize→R→uint8→scorer, per-stage allocations) + P12 5-lever composition-sign matrix (basis×taper×AA-ipe×counter-force×temporal-screw) | SPEC_v8.1 authoring + v752/v8 **byte-close** (shared owed-gate, both chains) | SPEC_v8.1 author + #385 brief owner | GATED (spec/byte-close) | owed on BOTH chains per philosophy_pass_v752 §P3/§P12 + philosophy_pass_v8 §Dual-chain note; needs lever-domain judgment — NOT doc-closable by debt-executor |
| D23 | v8 bare-literal code provenance NOTEs: `param_tolerance=0.05` (decoupling_screen.py:78, un-derived matching-rule) · `_SEG_H/_SEG_W=384/512` (road_undriv_bulk_field.py, grid-pin, no inline comment) · `dilate=2` (movable_deshare.py:132, untagged proxy that generated de-share 0.0044) | SPEC_v8.1 derive-or-waiver + inc1a code pass | v8 decoupled-field build / inc1a sibling | IN-FLIGHT (code) | formalizes the #393-D4 note; philosophy_pass_v8 §6 |
| D24 | v8 decoupled-field trainer BUILD + `--head` Lever fold + governed decoupled/control EVENTS (Blocker 1) + owed-9 lateral 3-curve carrier is 1a-BLOCKING | next v8 build unit (owed-9 carrier already BUILT at $0 per v8_unlock_398a) | v8 decoupled-field build agent | IN-FLIGHT (trainer + witness_dsl) | v8_unlock_398a §PINNED(owed) + DUAL_CHAIN_BRIEF rows 6/7/11/build |
| D25 | Harness residual: serializer `--base-content-sha256` hot-file MANDATE/default promotion (class-fixed opt-in 56fc64e19; mandate owed) | next apparatus-maintenance batch AFTER the live multi-agent commit session quiesces (promoting a default mid-session risks sibling commits) | serializer maintenance (#354 lineage) | GATED (session-quiesce) | harness_failure_ledger `serializer_whole_file_staging_absorbs_sibling_hunks` (worked-around + class-fixed; residual owed) |
| D26 | Harness residual: false-dead liveness process-tree-walk gate (worked-around 2026-07-10, no gate landed) — refuse a DEAD verdict from a grep pipeline; require ps child-pid walk | next apparatus-maintenance batch | liveness-tooling maintenance | GATED (apparatus batch) | harness_failure_ledger `false_dead_diagnosis_incomplete_process_tree_walk` (2026-07-10T12:56Z) |
| D27 | reactivation_campaign_397 machine-bound queue (13 pinned governed commands) — home pointer so it doesn't live only in the campaign memo | machine free (owed16v2 + v752 release) → drain per campaign §3; heavy = operator-GO | reactivation campaign / operator | GATED (machine) | campaign §3 class-(b); (d) 9 trigger-not-met already map to D1/D2/D9/D15-D19 |
| D28 | v752 config bare-literal NOTEs: taper knobs `strength 1.0 / scale 0.0 / floor 0.05` (`_CRUCIBLE_V752_LAUNCH1_DELTA`, no provenance rung) + OI-2 λ_lane/λ_movable computing-FEED cite | v752 DSL/config edit at #383-frees-curriculum_dsl OR byte-close grade | launch executor (witness_autoconfig/witness_dsl) | IN-FLIGHT (config) | philosophy_pass_v752 §CONSTANT-scale + P1 NOTE (activation-ledger key resolution for `--dseg-aware-taper*`) |

**Existing D1-D20 re-verified** (reactivation_campaign_397 FIRE-3 checked every trigger against live state
2026-07-10): D1/D2/D9/D15/D16/D17/D18/D19 = TRIGGER-NOT-MET (chosen-chain GOVERNED STOP / FINAL ckpt does not
exist yet — correctly ARMED, no new fire); D5/D6/D12/D13/D14 = QUEUED/CONDITIONAL, no change; D7 = CHECK-AT-COMPILE
(re-pointed to chosen-chain compile, IN-FLIGHT); D3/D4/D10/D20 = CLOSED-CONFIRMED. **No stale trigger, no orphan.**
| D27b | TERMINAL SOLVE-UPON-BASIN STACK on the live v7.5.2 run — fire in order on the terminal-band checkpoints/byte-close (NEVER mid-run): (1) #341 quadratic-head GN/CG solve (full-P only; subset NO-GO +5.1% overfit, LM ρ 0.847/0.868 basin-confirmed) · (2) #396 MC exact-metric finisher (built; v7.5.3 Δ5 TOOL surface) · (3) #400 click-polish pair-local diagonal (code d_seg + ξ pose, rollback vs 0.001610) — **BUILT 2026-07-10**: `tac.through_r.mc_finisher.PairLocalDiagonalFinisher` (+ `DiagonalProblem`, locality guard `require_locality`, `make_through_r_code_measure` for the d_seg/code axis, `make_byte_close_xi_pose_measure` + `load_banked_r1_dxi_dpose_floor` for the 4c′ ξ/pose axis); pair-local tier of the SDF-waterfill contract; 28 fixture tests green; MEASUREMENT owed at terminal band (this deferral) · (4) HeadOffsetSolver never-fired rung. All exact-gated + byte-close-verified + rollback-guarded; each solve's anchor mints at its measured row. | TRIGGER (machine-defined 2026-07-10 per #404): `tools/witness_telemetry_audit.py --run-dir <run> --section terminal_band --json` → `.d27b_ready` = muon fired AND trailing d_seg rel-slope < 5e-3 (terminal_band adds TAIL stop/Polyak arm); costate digest surfaces it each SessionStart → fire on checkpoint, run untouched | owner: terminal-solve stack (#341/#396/#400=`src/tac/through_r/mc_finisher.py::PairLocalDiagonalFinisher` + v753 ladder) | ARMED 2026-07-10 |

## rate_law_ladder_v1 owed measurables (2026-07-13; eq rate_law_ladder_20260713.OWED_MEASURABLES)
| # | Deferral | Trigger (named, not "later") | Owner/task | Status 2026-07-13 |
|---|---|---|---|---|
| D36 | fiber_completeness_gap_n600 — conditional-codelength estimate of H(q_G(W)\|U(W)) on n600 real states (the un-captured-invariance rate; ranks all future invariance capture) | next $0-probe slot OR before any new invariance-capture arm is prioritized | rate_law_ladder rung 2 (infdesc_foundations_dig_20260713.md) | CLOSED-MEASURED: MEASURED: fiber-completeness gap H(q_G|U) = 147,616 bits = 22.12% of archive rate term; conditional-codelength saving 1,903B does NOT survive 15,256B predictor charge → no codec lever (headline budget number banked) |
| D37 | flip_conditional_mi I(F;C\|M,ξ) — nested n600 cross-fitted conditional-codelength probe; if ~0 the margin field is a SUFFICIENT STATISTIC (every waterfill simplifies) | next $0-probe slot (the #468 dig's named highest-value actionable) | rate_law_ladder rung 4 (condprob_homotopy_lie_dig_20260713.md) | CLOSED-MEASURED: MEASURED: I(F;C|M,ξ) POSITIVE — class conditioning admitted, net 318,586 bits (95% CI [306,950,329,650]); phase-aware flat table loses 44,438 bits after overhead |
| D38 | H² obstruction after TYPING the H_cov extension — split vs obstructed; if obstructed the cocycle = R_twist^ideal (named irreducible rate term) | next math-ladder arm OR CGauge payload-accounting revision | rate_law_ladder rung 3 (garrett_algebra_dig_20260713.md) | CLOSED-MEASURED: DERIVED: local strict extension SPLIT, R_twist^ideal=0 (neutral Schreier class); global gluing maps NOT-TYPED (remains owed) |
| D39 | event-MARKS telemetry — upgrade event counts → MARKED events (mark = prediction-break family: topology \| receiver-phase \| branch-residual) per the rung-4 chain rule; score-neutral ⇒ defaults ON | next trainer telemetry batch (#408 rides the same resume boundary) | rate_law_ladder rung 4 (condprob_homotopy_lie_dig_20260713.md) | CLOSED-MEASURED: SPEC'D: marked-event manifest increment spec + implementation ticket landed; causal_manifest.py untouched (implementation owed) |
| D40 | organ causal-OPE identification: log exploration/randomization (or explicit ε-greedy jitter) in schedule-arm decisions so FORE/DR off-policy evaluation of unlogged arms becomes identifiable (current deterministic logs = walk-forward only) | next costate-organ arm launch or #432-family relaunch | main | OPEN |

## Throughput / wall-clock / convergence optimal-form reactivation queue (2026-07-14)

These rows supersede binary readings of naive instances. A trigger reactivates only the named optimal form; it never authorizes re-running the settled naive instance as family evidence. Pointer unchanged.

| # | Deferral / recovered object | Trigger and required receipt | Owner | Status |
|---|---|---|---|---|
| D41 | Margin-adaptive per-channel/per-layer native-width SegNet forward | Full real 0..599 winner/rival/tie certificate, physical int8/int16/int32 widths with exact int64 accumulator proof, then same-map forced-int32 versus native-width Metal placement/residency/wall receipt | margin_adaptive_perlayer_followon | FORMULATION-DEAD-ON-HEADROOM (D41 reopen $0 measured 2026-07-14, cert `d41_margin_waterfill_reopen_certificate_DAG_FEED_20260714.md`): NO fixed-point QDQ arm reaches argmax-exact at n600 — binding residual = fp32 argmax TIES (min-margin 0..5e-7, below fp32 reduction-order noise), bit-ALLOCATION-INVARIANT (w25→w26: 13→3 flips, exactness demands MORE bits toward fp32, not fewer). Per-CHANNEL scales formally unmeasured but ruled out on headroom. Reopen ONLY via per-channel×bit ablation sweep (NOT $0) AND a tie-tolerant formulation (certified-interval bound) — the cheaper-forward EV moved to the tie-tolerant / int64-determinism (D51/L70) paths. |
| D42 | Whole-teacher decision-quotient student at K32/64/128 | Reconstruct content-bound n600 centered logits/quotients/full input VJPs and scorer/R/source hashes; no-training fidelity plus charged-economics receipt before fitting or replacement | surrogate_vjp_fidelity_metric | OPEN-CUSTODY; n=0 receipt is not a negative |

> D42 MEASURED-2026-07-14 (`[macOS-CPU advisory]`): the binary intake transfer `||Δlogit||≤√(δ/p_w)` is FALSIFIED on real SegNet K=5 logits (n96, 18.87M px; argmax match 1.000000) — Spearman rank corr vs correct directional `|t|=√(8δ_kl/C_wr)` = **−0.9601** (opposite ordering); ratio median 16.3×/worst 1025×; binary over-admit 0. Surrogate correct-locus RE-ADMISSION is a BLOCKER (probabilities/logits/Jacobians NOT retained per store-nothing) ⇒ D42 stays OPEN-CUSTODY, reopen = RE-CAPTURE not $0 recompute. memo `.omx/research/ripo_categorical_fisher_binary_vs_directional_MEASURED_20260714.md`. MPS-DECISION-WAIVED: within-instrument rank-ordering falsification (both candidate transfer formulas evaluated on the SAME advisory logits — hardware drift cancels in the comparison); no score/promotion claim; D42 custody stays OPEN.
| D43 | Custom sparse-adjoint execution | M5 Metal host parity and actual wall versus 2.2086x DERIVED ceiling, followed by an n600-admitted provider; dense arithmetic timing is inadmissible | future costate execution feed | BUILT-DEFAULT-OFF / METAL-WALL-OWED |
| D44 | Converged local margin-saliency/taper | Same-checkpoint real-n600 ON/OFF after convergence with identical EMA and exact non-treatment custody; report local annulus form separately from settled global under-converged instance | HOLD/main or existing taper owner; perclass observer consumes only | OPEN; no fifth campaign arm |
| D45 | AdamW open optimality and separate MLX semantics gap | If still decision-relevant, run the Torch treatment at matched n600 finishing boundary; independently type MLX bias correction and calibrated module groups with resume/EMA custody | existing Muon owner | OPEN OPTIMALITY GAP; the old n16 Torch memo made no kill, and the MLX gap is separate/unmeasured |
| D46 | SPS after real temporal engagement | Screw/phase-engaged n600 cosine, norm, sign-conflict and class/tail telemetry; scalarization/stratified batching tested before PCGrad or duplication | perclass_convergence_ab observer | OPEN-ENGAGEMENT; ep275 row uninformative |
| D47 | Transported/event-triggered costate reuse | New provider with full scalar/quotient parity, exact Pose/non-scorer terms, stage/event refresh and n600 admission; only then time K2 | future costate execution feed | OPEN; raw ZOH K2 remains default-off |
| D48a | YOPO optimal validation cadence | Cheap drift proxy, sparse exact audits, alternate split/cadence and real-n600 charged-cycle receipt | future costate execution feed | OPEN; do not rerun exact validation every n1 step |
| D48b | Feature-ball optimal suffix certification | Exact/interval suffix bound or learned risk proxy with sparse exact audit and real-n600 admission | future costate execution feed | OPEN; n58 first cut is INSTANCE only |
| D48c | INSTANT native projected execution | Native low-rank/custom backward primitive, broader calibration and real-n600 charged-cycle receipt | future costate execution feed | OPEN; three-state/no-Metal instance cannot close the family |
| D49 | Current-V9 optimal micro-batch | Same-SHA uncontended B1/B2 ABBA with every active semantic routed, memory custody, full-step timing and descent parity; or batch-invariant scorer kernel | main throughput HOLD | HOLD behind D41; sealed drop-in negative stands |
| D50 | ANE/CoreML advisory and native placement | Prove device residency, localize precision offenders, apply per-op precision/pre-scaling, then real-n600 worst-pair fidelity and net wall economics | throughput authority / margin owner | OPEN; arbitrary 10x discard superseded |
| D51 | Exact-integer / explicit-order megakernel | Bound every reordered reduction/nonlinearity in claimed region, retain NumPy-fp32 parity, and measure full-real closure; float mx.compile instance is not re-run | throughput authority | HOLD after D41; Q15 R-adjoint is one positive instance |
| D52a | Median-freeze convergence-confound cleanup | Liveness-proven clean checkpoint A/B with emitted update counts and stage custody | main | **COMPLETED 2026-08-03** (`ddm_qd1`, `0bfeb8733b`, test_status green) — fixed and structurally gated by #397 `check_no_spike_guard_defaults_to_deadlock_mode` + #398 `check_reject_filter_updates_reference_from_accepted_only_has_rearm`, both VERIFIED BY EXECUTION (real denominator "2 trainer(s) scanned"), not by reading a landed-claim. Cell marked stale-OPEN until `ddm_iv1` joined it 2026-08-03. |
| D52b | Cured HOSC activation | SIREN initialization plus beta 1-to-4 anneal with trajectory custody | activation owner | OPEN; fixed-beta/no-init instance only |
| D52c | FreSh governed execution | Governed n8 to n64 to n600 fixed-quality A/B with one-time cost charged | FreSh/main | OPEN; host/governor refusal is NO-VERDICT |
| D53 | Provenance for transient task #495 | Register or relay exact canonical task/source memo and object identity before any verdict or routing | repoint_dismissed_intake | BLOCKED-IDENTITY; no technical conclusion |

## FIX-ALL / ANTI-DEFER SUPERSEDING AUDIT — 2026-07-15 (Sweep Arm C)

This table is the current disposition overlay for all 57 ledger rows. Duplicate historical IDs are
qualified `a` (original table) and `b` (burn-down table). `ROUTED` means the owner, trigger, and executable
gate are named; it does not mean a run occurred. The current vehicle is V9 CGauge per the 2026-07-14
operator binding. Old v7.5/v8 triggers are either re-pointed to V9 or closed with explicitly vehicle-scoped
supersession; no family verdict is inferred. Pointer `0.1910828242 [contest-CPU]` / borrowed defensive bank
`0.1880443980` UNMOVED.

| Key | Disposition now | Verified trigger / terminal route |
|---|---|---|
| D1 | NOT-MET / ROUTED | Governor-blocked, no agreement data. Owner `#355`; exact resumable command remains the one in `d1_gpu_verdict_agreement_probe_20260708.md`: `tools/safe_run.py --label d1_gpu_verdict_probe --projected-gib 8 --rss-mb 9500 --timeout 540 -- .venv/bin/python tools/d1_gpu_verdict_agreement_probe_n600.py --chunk-seconds 460`. Fire only after system admission; formulation remains open. |
| D2 | TRIGGER-FIXED / ROUTED | Old chosen-chain wording is stale. Owner V9 sweep/provenance: re-derive the knee from the first clean, converged V9 C0 checkpoint before its seal; no such C0 exists yet. |
| D3 | CLOSED-CITED | Canonical resume registry commits `2b7332f4b`/`8d349088d`; no reopen evidence. |
| D4 | CLOSED-CITED | Serializer hardening commits `a8ebcd12a`/`c35979b97`; Python-override recurrence additionally gate-landed in `279d801b09`. |
| D5 | TRIGGER-FIXED / ROUTED | Owner `#296`; assess fp16 cache in the mandatory real-shape V9 C0 timing/RSS smoke, not a nonexistent v7.1 arm. |
| D6 | TRIGGER-FIXED / ROUTED | Owner `#330`; measure subprocess/killpg reclaim in the same V9 C0 envelope smoke. No clean C0 receipt yet. |
| D7 | TRIGGER-FIXED / OWNER-ROUTED | Exclusive V9 provenance owner must assert pose-source flags and consumer receipt in the next strict V9 compile. Do not touch provenance hot surfaces from this arm. |
| D8 | NOT-MET / ROUTED | Owner `#255`; consolidate attribution tools after the first clean V9 C0 run has one day of real use. C0 has not launched. |
| D9 | NOT-MET / ROUTED | Owner `#355`; promotion remains gated by D1 measured agreement. CPU remains authority. |
| D10 | CLOSED-CITED | Witness Machine link/runtime resolution and evidence remain cited in the original row. |
| D11 | TRIGGER-FIXED / ROUTED | Owners `#343/#267`; polish after first clean V9 C0 harvest, not “post-relaunch.” No C0 harvest exists. |
| D12 | MET / OWNER-ROUTED | Apparatus batch is active. Owner activation/provenance lane: fix the PosixPath alias through the canonical activation writer and add a hook quote-exemption regression; `src/tac/witness_dsl` remains Arm-B-owned. |
| D13 | NOT-MET / ROUTED | Owner synthesis arbitration; fire only if a clean V9 trajectory is anomalous and primary attribution fails. Neither predicate is evidenced. |
| D14 | TRIGGER-FIXED / OWNER-ROUTED | Exclusive V9 schedule/provenance owner must decide octave-LR coupling from the compiled event-native schedule receipt; old “builder chose clock-LR” trigger is stale. |
| D15 | NOT-MET / ROUTED | Owner V9 throughput; exact trigger is fresh C0 baseline plus same-SHA n600 B1/B2 ABBA with all semantics routed. No baseline. |
| D16 | NOT-MET / ROUTED | Owner `#252`; rank kernels only after V9 exact component timers produce ep100 term costs. Sweep spec names that producer as missing. |
| D17 | NOT-MET / ROUTED | Owner `#252`; GPU fingerprint re-certification plus whole-step B8 bench at first governed V9 stop. No stop exists. |
| D18 | NOT-MET / ROUTED | Owner `#157/#336`; final V9 checkpoint plus k90 telemetry is absent. Then run `tools/witness_code_pca_byteclose.py --ks <measured-k90>` through exact receiver/score selection. |
| D19 | NOT-MET / ROUTED | Owner V9 throughput; consolidated speed bundle fires only after fresh C0 stop and comparator custody. |
| D20 | CLOSED-CITED | Resume-registry commits `51ae8ea8d`/`7834cda31`; no reopen evidence. |
| D21a | MET / OWNER-ROUTED | Blind-fill module and proof exist, but `levelset_byte_close_and_eval.py` does not consume them. Owner byte-close receiver: insert `tac.through_r.blind_coordinate.apply_blind_fill` before raw/archive selection, require the n600 bit-identity receipt, and reject missing receipt. |
| D22a | CLOSED-VERIFIED | Receiver now computes exact expected raw cardinality and refuses any mismatch before scoring in both byte-close tools (`levelset_byte_close_and_eval.py:2146-2159`, `witness_byte_close_and_eval.py:468-477`); receiver hardening/byte-close repro suites passed `52/52`. |
| D23a | NOT-MET / ROUTED | Owner V9 carrier composition; require decoded-composite interaction and cycle-integrability receipt before any multi-carrier V9 event. No current multi-carrier admission. |
| D24a | MET / OWNER-ROUTED | Raw 9.15/3.56/2.21% tail receipt remains absent. Owner scorer geometry: run the registered n600 radius/block-Jacobian probe before any locality/factorization claim; output must bind scorer/source/cache hashes. |
| D25a | CLOSED-V9-SCOPED | Current V9 config explicitly imports the AMBER stability preset and emits its grad clip (`spec_v9_cgauge.py:631-669`). This closes the current-vehicle decision, not historical pilot semantics. |
| D26a | SUPERSEDED-V753 / ROUTED-IF-REOPENED | V7.5.3 exact-D home is not a current V9 launch item. Reopen only as a V9 typed carrier with MLX/NumPy/inflate parity, counted-bank exclusion, and frame1-only receipt; no family negative. |
| D21b | CLOSED-MEASURED | Ep700 ON was measured: `owed16v2_verdict_20260710.json` / `init_levers_fresh_metainit_20260712.md` report `+3.2e-05 d_seg`, marginally worse. Scope is bounded warm-start only. |
| D22b | CLOSED-DESIGN / BUILD-GATED | V9 sweep spec now supplies end-to-end custody and explicit single-arm/interaction matrix, including HORIZONxSTEP. Execution stays blocked by typed-variant provenance and GO. |
| D23b | SUPERSEDED-V8 | The three V8 literals are not authority for current V9. Current rule is one LawRef/provenance bijection; reopen only if the V8 vehicle is explicitly restored. |
| D24b | SUPERSEDED-V8 | Standalone V8 trainer is not the bound vehicle. Any decoupled field must become a V9 typed Lever/LawRef/consumer/receipt; owner is the exclusive provenance/trainer lane. |
| D25b | MET / ROUTED | Owner serializer maintenance. Make `--base-content-sha256` mandatory for declared hot/shared paths at pre-lock and post-lock, add refusal tests for omitted/mismatched bases, then preserve isolated-worktree merge review. |
| D26b | CLOSED-VERIFIED | Process-tree liveness gate landed in `279d801b09`; live root-tree repro plus regression tests passed. |
| D27 | TRIGGER-FIXED / OPERATOR-ROUTED | Old 13-command v7 campaign must not fire verbatim. Each survivor requires recompilation as a V9 typed variant, lane claim, storage/governor admission, and explicit operator GO. Owner campaign/operator. |
| D28b | OWNER-ROUTED | V7.5.2 taper literals remain historical. Exclusive V9 provenance owner must resolve any reused values through LawRef; do not copy literals into V9. |
| D27b | NOT-MET / ROUTED | No custodied `d27b_ready=true` receipt exists here. Trigger remains exact: `tools/witness_telemetry_audit.py --run-dir <run> --section terminal_band --json`; fire ordered solvers only when Muon fired and trailing relative d_seg slope is `<5e-3`. |
| D36 | CLOSED-MEASURED | Fiber gap and charged predictor economics are recorded; no codec lever survived overhead. |
| D37 | CLOSED-MEASURED | Positive conditional MI and confidence interval recorded; class conditioning admitted. |
| D38 | PARTIAL-CLOSED / TRIGGER-FIXED | Local strict extension split is derived. Global gluing remains owned by the rate-law math lane and is required before any global twist-rate claim. |
| D39 | MET / OWNER-ROUTED | Spec exists but producer is owed. Owner V9 telemetry: implement marked-event rows in `causal_manifest.py`/trainer with resume-safe append and schema regression before next V9 launch. |
| D40 | NOT-MET / ROUTED | Owner costate organ; next V9 schedule-arm launch must log randomized/exploratory propensity or explicit epsilon-greedy jitter before FORE/DR OPE. |
| D41 | MET / OWNER-ROUTED | Latest directive supersedes the headroom-only row: native-width apparatus is built/reviewed but not V9 live-wired. Exclusive provenance owner must add one V9 DSL/LawRef/async-verdict consumer and accept only full-n600 identity + ten-process + same-map A32/AN>1 Metal receipt. |
| D42 | MET / ROUTED | Owner surrogate/VJP lane; recapture content-bound n600 logits, quotients, full input VJPs, scorer/R/source hashes, then issue no-training fidelity and charged-economics receipt before fitting. |
| D43 | NOT-MET / ROUTED | Owner costate execution; run M5 Metal parity and actual wall, then n600 provider admission. Dense timing is inadmissible. |
| D44 | RUN-GATED / ROUTED | Owner taper/perclass observer; matched converged n600 ON/OFF with identical EMA and non-treatment custody. Operator GO required. |
| D45 | RUN-GATED / ROUTED | Existing Muon owner; matched n600 finishing-boundary Torch treatment plus separately typed MLX semantics/resume custody. |
| D46 | NOT-MET / ROUTED | Perclass observer; wait for real screw/phase engagement, then measure n600 cosine/norm/conflict/class-tail telemetry before SPS. |
| D47 | BUILD-GATED / ROUTED | Future costate execution owner; full scalar/quotient/Pose parity, event refresh, and n600 admission precede any K2 timing. |
| D48a | BUILD-GATED / ROUTED | Future costate owner; cheap drift proxy + sparse exact audits + alternate cadence + charged n600 cycle receipt. |
| D48b | BUILD-GATED / ROUTED | Future costate owner; exact/interval suffix bound or audited learned risk proxy plus n600 admission. N58 remains instance-only. |
| D48c | BUILD-GATED / ROUTED | Future costate owner; native low-rank/custom backward primitive, broader calibration, and n600 charged-cycle receipt. |
| D49 | NOT-MET / ROUTED | Main throughput owner; remains behind D41 V9 integration and requires same-SHA uncontended B1/B2 ABBA or batch-invariant scorer. |
| D50 | BUILD-GATED / ROUTED | Throughput authority; prove ANE residency, localize precision offenders, apply per-op precision/pre-scaling, then real-n600 worst-pair fidelity and net wall. |
| D51 | NOT-MET / ROUTED | Throughput authority; after D41, bound every reordered op, retain NumPy-fp32 parity, and measure full-real closure. |
| D52a | **COMPLETED / DRAINED** | Main; drained 2026-08-03 by `ddm_qd1` (`0bfeb8733b`). The A/B is no longer the reactivation receipt: the median-freeze / spike-guard confound class is structurally REFUSED by #397/#398, verified by running both gates. |
| D52b | RUN-GATED / ROUTED | Activation owner; SIREN initialization plus beta `1->4` anneal with full trajectory custody. |
| D52c | OPERATOR-GO / ROUTED | FreSh/main; governed n8→n64→n600 fixed-quality A/B, one-time cost charged. Host/governor refusal remains no-verdict. |
| D53 | BLOCKED-IDENTITY / ROUTED | `repoint_dismissed_intake` must register or relay exact canonical task/source memo/object identity; no technical conclusion before that record. |
