# DAG FEED — ddm_tr1 trained partition→pixel renderer SPEC (2026-07-28)

**Arm:** ddm_tr1 (DESIGN-ONLY; no training, no scorer jobs, no launches). Pointer **0.1910828242
[contest-CPU] UNMOVED** (submittable original-work frontier; effective_frontier target 0.172 PR130;
non-submission bank 0.18804). Isolated worktree off main @ 65325a34c6 (ddm_co6 merged, verified).
Deliverable: `.omx/research/SPEC_tr1_trained_partition_renderer_20260728.md` (all six charter sections).

## FEED-tr1a — the OBJECT is fd1's own Rung-2, derived from OUR laws (PR130 lessons-only)
- **Object:** token-grid latent field + small trained partition→pixel conv renderer (SPADE/CLADE family,
  ~3 conv per RF-7), decoded scorer-in-loop. Two COUNTED payloads (token grid, renderer weights);
  forward-pass code + token interpreter GENERIC ⇒ FREE (rule-118).
- **DERIVED geometry (never PR130 constants):** token grid = lattice over the 512×384 scorer plane at
  downsample D∈{8,12,16} (ERF-bounded: pitch ≲ r50 85px / k), per-cell code width c∈{2,4,6}, temporally
  context-coded (1.246% frame-to-frame coherence). Renderer capacity CONCENTRATED on the Lane/boundary
  manifold (pp1: Lane = 36% of partition cost) via self-detecting lane_sdf/hood_static components.
- **Why (MEASURED convergence):** pp1 R1 — explicit context-arith partition 173.6 KB → composed ~0.189
  (ABOVE bar, +57 KB vs learned tokens). Token+renderer absorbs the shared structure → token leg lands at
  the ~117 KB learned end. ee1 C10 convergence theorem + fd1 Rung-2 routing: this is "the plausible ≤5e-4
  route" because it trains ALL pairs simultaneously (dissolves fd1's measured primary binding dim —
  cross-pair transfer — by construction).

## FEED-tr1b — training physics: descend THROUGH the uint8 quantum (the fd2 lesson)
- **fd2 disambiguation (MEASURED, receipt on SSD):** canary PASS; ×1.0 BLOCK_MOVED_NOT_IMPROVED (489
  flips, d_seg WORSE); ×0.5/×0.25 REALIZATION_GAP (`description_changed=true`, ZERO flips, delta EXACTLY
  0.0) ⇒ **GN validity radius < uint8 quantum.** A propose-continuous-then-quantize optimizer cannot cross
  it. The renderer's **uint8-STE + full R in-loop at BOTH weight AND description level** optimizes THROUGH
  the quantization — the sole measured wall-crossing mechanism (eval_roundtrip law; R1 pose descent; mc1
  admitted −0.0516).
- **Pose = TERMINAL (#383), the DECISIVE lever:** seg trunk first (`pose_objective_weight=0`), pose 6-eq
  GN solve LAST on frozen composed frames; watch collateral, never price mid-descent. Composed-S swing
  0.11: terminal d_pose ~2.33e-5 → contrib 0.0153 vs banked R1 dxi (7.2 KB) → contrib 0.1269.
- **fd1r wall-clock law (MEASURED, fd1 memo):** step ≈1,547 s, GN propose ~6 s, **99.6% is the realized-
  through-R verdict** (3 chunked n600 CPU verdicts ≈514 s each) ⇒ design the VERDICT CADENCE: cheap
  g3 hard-subset inner gates + sparse full-n600 confirms at stage boundaries; admit on strict full-n600.

## FEED-tr1c — pre-registered gates + composed arithmetic (DERIVED; ONLY a byte-closed evaluate.py row is authority)
- G1 native d_seg ≤5e-4 push ≤3e-4 (band lemma ρ_c=5.02e-4 REGISTERED; falsifier ≤1e-3 @≤64KB) · G2 no
  correction stream if G1≤ρ_c · G3 renderer ≤64 KB, bit-depth int4/int5/int8 RACED (never cargo-cult int8)
  · G4 token stream ≤130 KB target ~117 KB learned end · G5 pose terminal ~2 KB · G6 total ≤187,727 B
  (→0.172) / ≤154,522 B (→sub-0.15) [fc1 budgets MEASURED].
- **Composed S (DERIVED, arithmetic shown):** A optimistic corner (149 KB, 2.97e-4, terminal pose) =
  **0.144** (sub-0.15) · B spec mid (196 KB, 3e-4) = **0.176** (matches ee1 R6) · B' at d_seg 5e-4 =
  **0.196** (⇒ G1 must push to ~3e-4) · C banked-pose fallback = **0.311** (FAILS ⇒ pose MUST be terminal).
  Decisive levers: pose-terminal (0.11) ≫ total bytes (0.03) > native d_seg (0.02).

## FEED-tr1d — P0 launch contract + receiver/R6 chain
- **P0:** per-stage checkpoints (EMA shadow, decay via ema_decay_run_geometry_v1 not flat 0.997) +
  `--resume-from`; DSL-compiled config (WitnessProgram + Lever factories, `.validate()` fail-closed — skeleton
  in SPEC §S4.1); governed launcher `tools/launch_witness_run.py`; memory preflight rc=4 at the REAL config
  (verdict-batch 32 chunking); sched1 derived event-driven schedule (l7/smooth demoted).
- **R6:** decode through the proven E4/WS1 grammar (`DDME4WS1RuntimeExporterConfigV1`, per-stream named
  receiver consumer + SHA + Brotli-Q11); exact-consumption bijection = **#417** (charter's #402 is a
  mis-cite = telemetry-liveness — reported); rule-118 boundary stated (weights COUNTED, code FREE);
  byte-close → sh1 local-exact protocol → staged Modal contest-CPU row (the <$5 exact row).

## FORK-CONDITIONAL (fd2 FORMAL verdict PENDING; NO build this arm)
- (a) REALIZATION_GAP (expected, ×0.5/×0.25 rows already read this) → renderer IS the cure, BUILD fires on
  MAIN's GO, HIGH priority. (b) MIXED/locality → renderer with all-pairs in-loop still the route; add ee1-R7
  control-token re-solve as first-class. (c) late rung-1 surprise → renderer becomes capacity rung, SPEC
  unchanged, priority drops. Object + §S1–S4 physics unchanged across all branches.

## DISCREPANCIES REPORTED (recall-first)
- charter "706/344-param lift" = TWO objects (fd1 description space 706; fd2 GN lift active 344).
- charter "#402" exact-consumption = mis-cite; correct = #417 receiver-consumption bijection.
- charter/fc1 "212 KB" budget NOT located in fc1 this session (verified: 187,727 B @0.172, 154,522 B @0.15).

## NEXT (not this arm)
- MAIN lands fd2's FORMAL typed verdict → cites it at the BUILD gate.
- BUILD arm (on GO): land the tr1 Lever factories in `src/tac/witness_dsl/`, wire the description-level
  eval_roundtrip + terminal-pose stage into the levelset trainer, run the token-grid/bit-depth race under
  the governed launcher, byte-close → sh1 local-exact → Modal contest-CPU exact row.
