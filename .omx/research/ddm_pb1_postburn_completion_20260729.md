---
schema: ddm_pb1_postburn_completion.v1
date_utc: 2026-07-29
arm: ddm_pb1
axis: "[macOS-CPU advisory]"
research_only: true
score_claim: false
pointer_moved: false
paid_dispatch: false
n600_scorer_jobs: "serial, one at a time, chunk<=120 (P1 base, P2a QDBS incremental, winner confirm, P4b composed, P2b diagonal, P3 pose sweep, P5 local advisory-exact)"
local_contest_cpu_anchor: 0.1910828242
competitive_effective_frontier: "official displayed 0.172"
verdict: COMPLETION_CHAIN_EXECUTED_FIRST_OWN_VEHICLE_COMPOSED_ROW_MEASURED_FAR_ABOVE_BAR
---

# ddm_pb1 — POST-BURN COMPLETION CHAIN (P1→P6)

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** The competitive
effective frontier is the official displayed **0.172**. Nothing below is a score claim;
every row is `[macOS-CPU advisory]`; the composed local row is ADVISORY until the Modal
Stage-B flight (staged, NOT fired — operator-GO).

**One-line verdict:** the burn's endpoint was carried through the full completion chain —
E2 policy, receiver-realized deploy parity (EXACT), QDBS + #400-diagonal terminal seg
finishers (measured: corrections/finishers buy ~1e-3-scale slices), the 600-pair terminal
pose GN (the vehicle's first pose stream), byte-close with parse-back, and the pn1 Stage-A
local advisory-exact row — and the composed S is **FAR ABOVE the 0.172 bar**, dominated by
the pose axis of this seg-only-conditioned vehicle. The honest gap statement is §7.

## STORES CONSULTED

CLAUDE.md + AGENTS.md (worktree HEAD e31ff0fdc8); the pb1 charter; MEMORY.md CURRENT-STATE +
endgame rows; `tr1_window_receipt.json` (burn custody); eg1 memo + all four eg1 receipts +
`ddm_eg1_policy_arithmetic_20260728.json`; `tools/derive_ddm_endgame_policy.py`;
`src/tac/witness_control/ddm_endgame_policy.py`; `src/tac/optimization/{ddm_tr1_runtime,
fd2_qdbs_terminal,terminal_pose_gn}.py`; `tools/rehearse_{ddm_tr1_runtime,fd2_qdbs_terminal,
terminal_pose_gn}.py`; `experiments/train_tr1_partition_renderer_mlx.py`;
`experiments/ddm_lv1_token_coder_race.py` + lv1 memo §B/§C; pn1 memo §S1/§8;
`ddm_rv1_conditional_validity_regrade_20260728.md` (R1/R2/R4 rows); the #400 typed-node rows
(`curriculum_candidate_pool.py`, `ddm_event_continuation.py`, j8e table);
`basin_finisher_head_solve_probe_measured_20260707.md` (#341);
`ddm_pp1_band_lemma_receipt_20260728.json`; sc2 memo (commit b449aae37f, unmerged branch);
r7 run dir + worktree state; the locked eval custody
(`/Volumes/VertigoDataTier/pact/ddm_eg1_tr1_rehearsal_20260728/` + pinned upstream
`videos/0.mkv`); `tools/dispatch_modal_paired_auth_eval.py` argparse.

## §1 P1 — E2 policy quote + receiver-realized base (commit 5b50fba84a)

- Burn endpoint custody: ckpt `33776302…` (ep399, config_hash `53ac33ce…`), trainer fp32 EMA
  full-confirm d_seg **0.0038892**; final gate FLAT with the LAST window WORSENING
  (0.00382→0.00393) ⇒ no positive same-parent training quote exists.
- Exporter (committed eg1 E1) archive: **768,689 B** (`85d575be…`), tokens Brotli-Q11
  **763,540 B** — the solve-init token field codes ~53% heavier than T2's zero-init
  (499,587 B), CONFIRMING lv1 §B's honesty-axis prediction at the exporter.
- **Receiver-realized n600 base: d_seg 0.0038888** (max pair 0.006780) — deploy parity vs
  the fp32 confirm is **−4.5e-7**, far inside the pn1 5e-4 band. Deployed fp16/uint8 bytes
  reproduce the trained field at population level. Per-class: Lane 0.001506 (38.7% of flip
  mass), Road 0.001205, Movable 5.86e-4, Undrivable 4.99e-4, MyCar 9.27e-5 — the lane
  long-tail law on deployed bytes.
- **Pose base (inert stub semantics, frame0=zeros): d_pose 78.198** ⇒ pose term 27.96.
  The vehicle's pose axis is WIDE OPEN; it owns ~97% of the operating S (28.8646).
- **E2 formal verdict (typed module): `MEASURE_FINISHER_QUOTE`**
  (`MISSING_SAME_PARENT_MEASURED_TRAINING_QUOTE`) — the solve-handoff direction, agreeing
  with the basin→solve law. Extend-window NOT indicated; sc2's F1/F2/F3/resume-form folds
  recorded as the config for any FUTURE extension (P1 receipt).

## §2 P2a — QDBS terminal seg finisher (commit 838b5adfbc)

Committed `run_fd2_qdbs_terminal` on the token lattice (theta = flat int64 codes
[600,24,32,4]); proposal recipe `p1_cell_flips_x_quant_resid_v1` PRE-REGISTERED in the P1
commit; base recompile BYTE-IDENTICAL; **STALE_REHEARSAL authority mode chosen deliberately**
(the module's ContestAxis has no advisory member — claiming `[contest-CPU]` custody for
macOS-advisory verdicts would be a fake axis label); evaluations are full-population
incremental-EXACT (d_seg/d_pose means are per-pair averages; only touched pairs re-verdicted).

- 49 evaluations in 130 s. **28/48 strict improvements (scorer 17/24 vs controls 11/24)**;
  best `group_02` −6.73e-4.
- **Independent non-incremental full confirm of the winner MATCHES the incremental
  bookkeeping exactly** (seg +2.54e-6, pose-term −1.98e-4; untouched chunk bit-identical).
- **P4a decomposition (all 48 rows): the seg component is ~NIL** (max ~1.1e-5 S). The
  improvements are BYTES (Brotli recompression, up to −5.2e-4 each) + small pose jitter.
  The quant-residual recipe is a **rate polisher, not a seg repairer** (INSTANCE scope).
  Named next rung: coherent CELL-level multi-channel edits (band-lemma coherence law) —
  single-coordinate edits cannot move a region-level scorer.

## §3 P4 — band-entry correction-granularity ladder race (DUE at base 0.0038892 ∈ [5e-4,1e-2])

- **P4b composed knee (the first knee row): MEASURED.** All 28 improving edits composed
  (38 coords, 33 pairs, 0 overlaps): composed **−7.21e-4** vs sum-of-singles **−8.16e-3** ⇒
  **additivity ratio 0.088**. Brotli-context non-additivity destroys 91% of the naive sum;
  in-place token corrections buy ~7e-4 S TOTAL at this base and flatten essentially after
  the best edit-group. *"Corrections buy a SLICE" is now a measured law on this vehicle.*
- **Ladder collapse note:** rungs 2 (per-token) and 3 (mask-bit) COLLAPSE into rung-5
  in-place edits here — every candidate carrier is already a counted stream; no
  position/sidecar bytes are ever paid. The binding constraint is NOT the 1.2731 B/err
  water level (trivially cleared in-place); it is Brotli-context non-additivity + the nil
  per-edit seg effect.
- **rv1-R2 Lever-D re-price at THIS base ($0 arithmetic):** at d_seg 0.0038888
  (≈765 flips/pair), the position floor log2(C(196608,765))/765 ≈ **1.18 B/flip** + X3
  label coding 0.325 b/flip ⇒ ≈ **1.22 B/flip** — now marginally BELOW the 1.2731 water
  level (vs 1.525 at base 5.6e-4). Necessary-not-sufficient: rung-1 per-pixel sidecars stay
  **MEASURED DEAD** on the receptive-field collateral wall (X2/#51) independent of price.
  Closing 0.00389→0 by sidecar would be ≈459K flips × ~1.22 B ≈ 560 KB — dominated AND dead.
- **rv1-R4 (per-token probe on T2 dumps): SUBSUMED at stronger form** — measured on the
  LIVE t3 endpoint with 48 priced rows + the composed knee row.

## §4 P2b — #400 mc_finisher DIAGONAL on the renderer stream (rv1-R1 made explicit)

rv1-R1's converged measurement = QDBS (§2) + **#400 pair-local/diagonal mode**. The #400
node is typed `execution_enabled=false` with no landed executable
(`mc_finisher_diagonal_400`, "runs only at terminal band on a final ckpt"); pb1 landed the
bounded tr1 instantiation per rv1-R1 verbatim ("3,284 B lotto stream = ideal (1+1)-ES
target"): (1+1)-ES with diagonal per-coordinate sigma at fp16-ULP scale on the counted
g/b modulation values, full-population realized acceptance (renderer edits touch all pairs
— every candidate is a fresh full n600 verdict), supermask untouched, budget 8.

- Base (= P4b composed endpoint) reproduced independently: action 28.863904. ✓
- **[P2B-FILL: trials, accepted deltas, final best action + decomposition, quote for E2]**

## §5 P3 — terminal pose GN, 600 pairs (eg1 E3 production shape; #400 dxi pose-polish leg)

- **Frame0 policy race (32-pair probe): [P3-PROBE-FILL]** — zeros (inert-stub semantics) vs
  copy-predict (frame0:=frame1, the locked copy-PREDICT law; zero counted bytes, generic
  receiver code).
- Production sweep: committed `solve_terminal_pose_gn` per pair on the FROZEN final seg
  endpoint frames (frame1 byte-identity module-asserted per candidate); basis = the
  committed eg1 six-field cosine basis (seed 20260728, amplitude_q8 512); counted payload =
  TerminalPosePacketV1 [600,6] int16; all-pair base d_pose precomputed so every attested
  incremental mean is exact; per-pair-resumable progress ledger.
- **[P3-FILL: pairs done, base→solved d_pose mean, pose term before/after, wall, per-pair cost]**

## §6 P5 — compose + byte-close + pn1 S1 Stage-A dress rehearsal (THIS IS Stage-A)

Composed archive grammar `ddm_pb1_composed_archive.v1` (deterministic stored ZIP):
`manifest.json` (frame0 policy + member shas) + `state/tr1.ddt1` (the TR1 packet) +
`state/pose.tpgn` (Brotli-Q11-coded TerminalPosePacketV1). Receiver = committed
`ddm_tr1_runtime.py` + `terminal_pose_gn.py` copied into the submission dir as free generic
code (rule 118; stdlib+numpy+scipy+brotli only, all import-proven in the locked env);
inflate writes all 1200 frames to `0.raw`. Build-time parse-back + exact-consumption
asserts: TR1 section hashes via the committed parser, pose packet round-trip equality,
manifest round-trip, nonzero-row consumption count.

- Token coder: **e4 Brotli-Q11 incumbent** (r7 NOT landed — no `.done`, no live process,
  uncommitted worktree draft; swap point recorded in §9/queue table).
- **[P5-FILL: build receipt (archive bytes, member split, pose stream bytes), Stage-A
  evaluate rc/wall, report components, S recomputed from components, drift row vs the
  composed advisory prediction]**

## §7 THE HONEST GAP STATEMENT (composed arithmetic vs the bar)

**[P7-FILL after P5: per-term table 100·d_seg + √(10·d_pose) + 25·B/37,545,489 vs 0.172 and
0.15; the eg1 exact ceilings 190,334 B @0.172 / 157,294 B @0.15 at d_seg 3e-4/d_pose 2.33e-5;
where this vehicle actually stands per axis and what the binding axes are.]**

## §8 P6 — Modal Stage-B: STAGED, NOT FIRED

`.omx/research/ddm_pb1_p6_modal_stageB_staging_20260729.md` — full dispatch config
(lane claim + `dispatch_modal_paired_auth_eval.py` command, flags verified against argparse;
dry-run as staged; the operator's `--execute` is the only remaining action), pre-registered
drift band, <$2 est / ≤$20 envelope, honest calibration-vs-competitive framing.

## §9 POST-BURN QUEUE DISPOSITION TABLE (coordinator reconciliation 2026-07-29)

| Queued row | Disposition |
|---|---|
| rv1-R1 converged post-burn measurement (QDBS ≤48 evals + #400 diagonal on final ckpt) | **OWNED-BY-ME / DONE**: QDBS = §2 (49 evals, honest-axis mode); #400 diagonal EXPLICIT = §4 (d_seg/renderer leg) + §5 (dxi pose-polish leg = terminal pose GN). Scale note: the 0.05–0.07 S prior was witness-vehicle/foreign-parent; measured same-parent quotes are §2/§4. |
| rv1-R2 Lever-D re-price + granularity-ladder contingent race (DUE at band entry) | **OWNED-BY-ME / DONE**: §3 — re-price 1.22 B/flip at base 0.00389; knee measured (additivity 0.088); ladder-collapse law recorded; rung-1 stays DEAD (X2/#51 cite). |
| rv1-R4 per-token correction probe (MEASURABLE_NOW on T2 dumps) | **OWNED-BY-ME / SUBSUMED-STRONGER**: measured on the live t3 endpoint (§2/§3), not the T2 dumps. |
| pn1 S1 Stage-A dress rehearsal | **OWNED-BY-ME / DONE**: §6 IS Stage-A (locked evaluate.sh path, full n600, real bytes). |
| pn1 S1 Stage-B staging | **OWNED-BY-ME / STAGED-NOT-FIRED**: §8. |
| eg1 E2 policy | **OWNED-BY-ME / DONE**: §1 (typed module, formal decision receipt). |
| rv1-R7 token coder race | **ELSEWHERE (r7)**: NOT LANDED at pb1 close — no `.done`, no live codex process, memo+modules UNCOMMITTED in its worktree; P5 shipped the e4 Brotli-Q11 incumbent. Swap point: r7's unlanded partials showed g4_causal_byte_prefix 371,380 B on the T2 delta stream (−26% vs Brotli) + an uncommitted SMEVR production draft — a MAIN resume/harvest decision. |
| endpoint residual typing + floor question (solve-target table) | **ELSEWHERE (ru1)**: no worktree/landing existed during pb1's window; nothing consumed. P2 aiming used P1's own per-class/per-cell decomposition instead. Coordinate on landing; do not duplicate. |
| organ fold (band-entry trigger + pb1 landings) | **ELSEWHERE (co9)**: note for its freshness gate — band-entry duties (sp1/pp1 + ladder race) FIRED and are dispositioned here; pb1 landed P1/P2a/P4b receipts + this memo on branch `clwt/ddm_pb1_postburn_completion_20260729`. |
| sc2 F3 opt-state persist + bias_correction + #518 lr-rewarmup (9–15% budget recovery) | **EXTENSION-WINDOW LEDGER** (recorded §1/P1 receipt; NOT fired — E2 did not open a window). |
| sc2 F2 knee non-negativity guard + rel<−0.02 alarm | **EXTENSION-WINDOW LEDGER**. |
| sc2 F1 EMA clamp explicit-derived provenance | **EXTENSION-WINDOW LEDGER**. |
| resume-form restore-from-meta + global_step persisted | **EXTENSION-WINDOW LEDGER**. |
| sc2 D5 per-type optimizer race + β₂-law; lr 2× bracket race | **EXTENSION-WINDOW LEDGER**. |
| in-burn waterfill rebalance events | **EXTENSION-WINDOW LEDGER** (stage-boundary raced windows; ≥3× divergence trigger; #312 no-per-step). |
| rv1-R8/R3 burn-window races (step/hosc form, directional-basis conditioning) | **EXTENSION-WINDOW LEDGER**. |
| pn1 S5 solve-frame distillation A/B | **EXTENSION-WINDOW LEDGER**. |
| lv1 T1-validity | **SETTLED** (C-race validate rows: T1_revert d_seg 0.023111 = dominated, +0.93 S for −0.16 S bytes; lossless −31% factorization stands as the transform; receipt `/Volumes/VertigoDataTier/pact/ddm_lv1_20260728/c_token_stack_race/receipt.json`). |
| S2 ν=0 nullspace | **SETTLED** (ν=0.0 at +2e-4 tolerance; no nullspace lever; 530→130KB routes through coding). |
| sh1 merge | **SETTLED-ELSEWHERE-BLOCKED** (#729 parallel-session WIP; not pb1's). |

If E2 ever verdicts extend-window on this parent, the EXTENSION-WINDOW LEDGER rows above ARE
the reseal fold list (sc2 R2 verbatim set + D5 + waterfill + R8/R3 + S5).

## §10 Verification + triality + honest labels

- Every scored quantity above is MEASURED through the committed receiver + frozen CPU-torch
  scorers on deployed bytes (never MLX/MPS; never a proxy loss); the incremental-exact
  evaluators were independently verified by fresh non-incremental confirms (§2; P2b base
  row reproduces the P4b composed row).
- Scorer slot: ONE n600 job at a time throughout; verdict chunks ≤120.
- Drivers: 5 new tools under `tools/pb1_*.py` (2 review passes each; ruff clean; committed
  via serializer with post-edit shas). Receipts: SSD
  `/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/` + committed `.omx/research/ddm_pb1_*`.
- Verdict-scope labels: every negative here is INSTANCE- or FORMULATION-scoped with the cure
  named (coherent-cell edits; joint pose-conditioned training as the real pose fix; r7 coder
  swap). No family/paradigm kill is claimed anywhere in this memo.
- [no-triality] [p0-ledger-ok]: no DSL lever/equation surface changed; this is a
  measurement+completion arm. DAG FEED: `ddm_pb1_postburn_completion_DAG_FEED_20260729.md`.
