# ddm_tq1 pre-empted by rt1 measured negative + sl2 terminal-pose composition arithmetic — 2026-08-05

Axis: [macOS-CPU frozen-scorer advisory] throughout. score_claim=false, promotion_eligible=false.
Frontier at writing: S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] (qo1 sub_auto_pairbit).

## 1. tq1 scorer leg KILLED — replay of a measured negative (staleness-at-consumption instance)

The tz1 READY manifest (#869, 2026-08-04) queued two scorer legs "the instant the scorer frees."
**sb1 had ALREADY run leg #1** the same evening (`ddm_sb1_20260804/B_rt1_margin_16_12_8_4_n600_eval_receipt.json`):

- rt1 margin-coupled [16,12,8,4] on sub_final, n600: **d_seg 0.00515854** (Δ +8.4675e-4 vs
  0.00431179 — OVER the pre-registered 7.56e-4 bound), **d_pose 0.16815221** (pose bank
  destroyed; baseline 0.00071459), **S 1.9753** @ 244,436 B. Per-class receipt: Road-dominated.
- qo1's pairbit is a pure container change (sb1 R8 guard: seg/pose EXACTLY unchanged), so the
  snap on qo1 tokens is byte-for-byte the same decoded transform → tq1's margin eval could only
  reproduce ≈1.975. Killed at ~7 min elapsed (pgid 41884). Compose artifacts retained at
  /Volumes/VertigoDataTier/pact/ddm_tq1_20260805/sub_tq1_margin (244,188 B, sha 0f… in
  compose_receipt.json) as byte-leg custody only.

**Failure class (m37/m44 worked example):** the READY row was consumed WITHOUT re-validating
against negatives accumulated AFTER its registration — the rt1 kill was sitting in the very
hot-state POINTER_LINE block being edited. Cure at consumption: any READY/queued measurement
older than the latest fresh-negatives update MUST be re-checked against the negatives register
(hot-state + receipt dirs, same transform signature) before fire. Memory appended to m37.

## 2. tz1 READY #1/#2 dispositions (supersession, append-only; tz1 manifest NOT edited)

- **#1 margin (−113,648 B): CLOSED — PRE-EMPTED_BY_MEASURED_NEGATIVE (rt1).** The seg overage
  alone is fatal even with pose fully re-solved: +100·8.47e-4 = +0.0847 S vs rate −0.0757 S.
- **#2 derived (−62,502 B): NOT FIRED — family-closed at this base.** Same tier fractions,
  TIGHTER bound (4.16e-4 = half the margin variant's measured overage), and the same
  pose-conditioning break (below). Composed archive banked at
  ddm_tq1_20260805/… and sb1 B_rt1_derived_activity_16_12_8_4_sub_final/ (byte custody only).
  Reactivation criterion (honest, not a KILL): a SHALLOW ladder confined to the lowest-flip-mass
  tier only, composed WITH a pose re-solve against the snapped field, priced as one pfs1 pass.

**Mechanistic law (register candidate):** the pose sections (pose_warp PFS1WPD1 + F0PR1
frame0_pose_repair) are SOLVED AGAINST the exact token field. Any post-hoc token mutation
invalidates them — d_pose 0.168 is the conditioning break, not "coarse tokens are bad."
Consequence: every rate attack that mutates tokens must RE-FIRE the downstream solved legs
(pose re-solve) inside its compose. This is the "recursion from solved states" doctrine (m22)
surfacing on the rate axis.

## 3. sl2 terminal-pose FINAL (32/32 done, rc=0) — recovery real, composition still net-negative

`ddm_sl2_20260805/sl2_composed_terminal_pose_n32.json` (STALE_REHEARSAL authority mode;
incremental-exact per-pair, not byte-closed):

- seg solved subset mean 0.004300 → **0.0010015** (η pooled 0.885, ZERO cap-bound rows — the
  sq2 uncapping validated end-to-end).
- pose erosion composed-effective 0.05807 → **0.0082654** (7.0× recovery) via the 6-eq GN
  terminal re-solve; terminal packet **479 B / 32 pairs (~15 B/pair)**; relinearizations=2.

n600 composition (advisory; assumes subset baseline pose ≈ population mean 7.1459e-4 —
subset-exact baseline not in this receipt):
- seg: (32/600)·(0.0010015−0.004300) = −1.759e-4 d_seg → **−0.01759 S**
- pose: (32/600)·(0.0082654−0.00071459) = +4.027e-4 d_pose → term 0.084533→0.105696 → **+0.02116 S**
- net **+0.0036 S LOSS** before any carriage bytes. Break-even needs composed subset pose
  < ~0.00687 (we are 20% above it).
- Carriage remains dominant and unpriced here: n32 solved-frame description at od-line measured
  prices ≈ 65 KB ≈ +0.043 S (2.4× the seg win). Doctrine confirmed: **seg is solvable; the gap
  is CARRIAGE + pose-carrying base** — the row comes via jd1 (#366) or grammar carriage, not by
  shipping solves.
- Named residual lever (censored-cap genus, #850's sibling): the terminal GN ran relins=2 with
  no convergence adjudication — uncapping plausibly crosses the pose break-even but does NOT
  cure carriage; queue behind jd1, do not burn the slot on it now.

## 4. Live state after this note

- w3 ep1189 (gate d_seg 0.003884, COUPLED_DESCENT, counted bytes falling 259,159→258,528),
  boundary ep1224 ≈ 95 min out; canonical adjudicate_tail_slope by hand at the boundary →
  extend w4 | converged → jd1 (#366) regenerates against the winner.
- Scorer slot FREE (tq1 killed). Next scorer use decided at the w3 boundary; margin-weight A/B
  (#925, bo1 §4) queues behind the jd1 decision at that boundary.
