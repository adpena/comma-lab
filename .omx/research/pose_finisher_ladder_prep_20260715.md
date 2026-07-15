# POSE-FINISHER LADDER PREP (#248/#366) — finisher-phase plan, config delta, pre-registration (2026-07-15)

**Axis:** design + $0 prep ONLY. NO launch (C0 `levelset_n600_witness_20260715T095030Z` untouched).
All numbers advisory / `[macOS-CPU advisory] NON-PROMOTABLE` unless cited otherwise.
**Pointer contest-CPU 0.19108 UNMOVED — this is MEANS.**

**OPERATOR REFRAME (2026-07-15, supersedes the standalone-run framing):** pose is the FINISHER, not a
parallel co-equal run — *"R1 was optimal and pose was a finisher we would run at the end or at the
optimal time once dseg was properly converged."* The R1 two-phase architecture already sealed into the
config (pose-blind trunk → `pose_finish` engages on the #383 `sigma_min_plateau` gate / ep726 muon
backstop → terminal joint descent) IS the vehicle. This memo delivers the finisher-PHASE prep: the rung
the window executes, the DSL config delta, the pre-registered trajectory + stop criteria, and the honest
landing zone. Reusable part kept from the standalone framing: the #366 prep
(`joint_descent_p0_launch_prep_20260708.md`) memory-preflight + warm-start custody table remains valid
for any warm-start finisher A/B (v2_attrclean mod-26 ckpts: CE ep299 / Tau ep599 / MuonStart ep726 /
L7 ep1000 + `levelset_witness_ema_mlx.npz`, standalone 67.6 GiB SAFE).

STORES CONSULTED: CLAUDE.md §Pose-is-SOLVED + 2026-07-10 CLARIFICATION · [[L68]] ·
council_pose_carrier_optimal_form_symposium_20260703 (#250, §6 L1 gate, §8 L3 elevation) ·
DAG FEED-poseladder / FEED-posehard / FEED-poseresearch / FEED-posesegdynamics / FEED-auditC /
FEED-jointdescentprep / FEED-shippable / FEED-238resolved / FEED-C0H2 (C-H2-1) ·
pose_mladder_depthwarp_measured_20260708 · pose_l2_truedepth_probe_measured_20260708 ·
joint_descent_p0_launch_prep_20260708 · graph_memory_recall "pose carrier ladder P-B FiLM read-back".

---

## 1. The arithmetic (VERIFIED — every constant cited; the ~0.11 S claim holds)

`S = 100·d_seg + √(10·d_pose) + 25·|archive.zip|/37_545_489` (upstream/evaluate.py).

- **Banked pose (authority-scale, byte-closed):** d_pose **0.001610** n600 through the real inflate ⇒
  contribution √(10·0.001610) = **0.1269** ≈ 0.127; dxi table 7,195 B ⇒ rate +0.004791
  (FEED-238resolved, `r1_dxi_shippability`).
- **The 0.018-class target:** d_pose = 0.018²/10 = **3.24e-5**. Gap from banked: 0.001610/3.24e-5 =
  **49.7× ≈ 50×** in d_pose. (0.018 is the ANCESTOR-borrowed contribution class — per L18 it does NOT
  transfer as a number; it is used here only as the budget-relaxation threshold.)
- **Budget consequence (rate basis 0.030, a lean shipped witness; R1's advisory rate was 0.060):**
  - pose at 0.127: d_seg budget for sub-0.191 = (0.191−0.127−0.030)/100 = **3.4e-4** — 2.6–4.3× BELOW
    the L85 appearance-phase d_seg target band (0.0008–0.0012) and ~15.6× below the GT-side flicker
    floor 0.005318 without appearance-phase ⇒ **infeasible near-term**. (At rate 0.060 the budget is
    4e-5 — worse.)
  - pose at 0.018: d_seg budget = (0.191−0.018−0.030)/100 = **1.43e-3** — ABOVE the appearance-phase
    band ⇒ **feasible-class**.
  - **Worth of the full ladder: 0.127 − 0.018 = 0.109 ≈ ~0.11 S** — as much as the whole d_seg
    campaign. VERIFIED.
  - Intermediate landing (d_pose 2e-4): contribution √0.002 = **0.0447** ⇒ worth ≈ **0.08 S**. VERIFIED.

**Pose-budget law (equations leg; # FORMALIZATION_PENDING: registration in tac.canonical_equations is
outside this task's file boundary — owed as a follow-on):**
`d_seg_budget(S_t, c_pose, c_rate) = (S_t − c_pose − c_rate)/100` and
`pose_ladder_worth(d1→d2) = √(10·d1) − √(10·d2)`, with EmpiricalAnchors (0.001610→0.127 measured;
3.24e-5→0.018 derived-target). The √-shape means the LAST 50× of d_pose buys only 0.109 of the 0.127 —
diminishing below ~1e-5 (symposium §1 "sweet spot ~1e-5").

## 2. The measured ladder state (what is settled — do not re-derive)

| rung | state | number | cite |
|---|---|---|---|
| P-A / L0 store-nothing ξ + joint-descent dxi TABLE | **BANKED** (n600 byte-close) | d_pose 0.001610 → 0.127, 7,195 B | FEED-238resolved |
| P-B FiLM read-back (original stored-target form) | **FIRED + SUPERSEDED** — H-TARGET verdict; post-hoc/stored family MEASURED DEAD (5 formulations, photometric wall) | carrier-only cap ~2.5 | FEED-posehard, CLAUDE.md 07-10 CLARIFICATION |
| Depth-warp (Option A / stratified / true-depth) | **FALSIFIED (formulation)** — wall is APPEARANCE not flow | true-depth 1.296 ≥ homography 0.878 (real); witness luma pose-blind ~167 | pose_l2_truedepth / pose_mladder |
| 6/12-DOF pose-space solves (A2/A2+) | **REFUTED** (≤12-DOF warp can't break ~1.2 on unconverged trunk) | A2 1.486 / A2+ 1.223 | pose_mladder |
| Joint descent from CONVERGED trunk (R1) | **THE measured cure** | 97→0.0011 in ~108 ep, plateau ep1074/1093 ~0.00108; d_seg held | FEED-poseladder |
| P-E free-frame0 inverse solve | existence proof ONLY (rate-prohibitive as pixels) | d_pose ~2.7e-7 (n3) / ~1e-8 class | #249 + symposium §1 |
| **L1 Jacobian-coefficient read-back (#250 §6)** | **UNMEASURED — the pre-registered decisive $0 gate, NEVER FIRED** | predicted →~0 at rate ~0.010–0.020 | symposium §6 |

The #250 decision tree said: L3 (in-training) primary, **L1 = the $0 fallback "if store-nothing floors
too high."** That trigger condition is now MEASURED MET: the joint-descent family floors at ~1.0–1.6e-3,
50× above the 3.24e-5 need. L1's gate is therefore the owed decisive action of the ladder.

## 3. What the pose_finish phase EXECUTES (the rung choice)

When the #383 gate engages on the next converged trunk, the finisher window executes THREE nested
things, in order:

1. **The R1 joint descent (incumbent, already sealed in config)** — store-nothing carrier, table dxi,
   w_pose 1.0, engage on `sigma_min_plateau` (backstop ep726). No delta. This produces THIS trunk's own
   dxi (FEED-shippable: the final witness needs its OWN dxi; R1's number does not transfer across
   trunks).
2. **The P-B FiLM read-back arm (the DSL delta, A/B vs table)** — `PoseFinisherFilmReadbackArm`
   (NEW, curriculum_dsl): dxi READ BACK from the already-shipped per-pair codes through a shared FiLM
   MLP (code[mod_dim]→32→6), trained JOINTLY in the window (never post-hoc). HONEST prior: film is a
   constrained reparameterization of the same 6-DOF twist residual — it cannot beat table on d_pose;
   its win is RATE: ~1.7–2.2 KB shared MLP vs 7,195 B table ⇒ −~0.0035 rate at held d_pose.
3. **The L1 Jacobian-coefficient $0 read-back gate — THE DECISIVE RUNG — fires at the finisher
   plateau, on the finished checkpoint, decode-side (no training):** store only K≈6–18 coefficients of
   the pose residual r; at decode recompute the rank-6 `J = ∂PoseNet6/∂frame0` basis from (frozen
   PoseNet, frame1, frame0_base = the finisher's warp output) — rule-118 FREE — apply k Gauss-Newton
   steps; measure d_pose through the frozen CPU-torch authority, all 600 pairs (#250 §6 verbatim,
   tooling seam: `tools/pose_frame0_inverse_solve_probe.py` #249 machinery). It COMPOSES with the
   finisher: the descent shrinks r, making the K-coefficient truncation easier. Why decisive: it is the
   ONLY rung with a measured existence proof below 1e-3 (P-E ~1e-8-class), and its outcome alone
   decides whether the ladder's last 50× is closable at all. FIREWALL (NO-FAKE #6/#8): coefficients =
   counted video-derived payload; basis+GN = generic decode algorithm; a stored optimized frame0
   image-as-"code" is the eval-hack fake.

## 4. The finisher-phase config delta (DSL leg — LANDED + dry-run VERIFIED)

Two zero-required-arg composable levers in `tac.witness_dsl.curriculum_dsl` (+9 tests in
`src/tac/witness_dsl/tests/test_pose_finisher_ladder_20260715.py`, all green; sister suites 104+16
green):

- **`PoseFinisherFilmReadbackArm()`** → `--pose-carrier-residual-mode film` (exactly ONE flag delta
  over any pose-carrier-active base; inert on a carrier-less base).
- **`PoseFinisherLiveGap()`** → `--verdict-live-gap-every 4`, cadence DERIVED by
  `pose_finisher_live_gap_cadence()` (= max(1, ceil(ema_warmup_updates(0.997)/ceil(600/8))//2) = 4;
  ≥2 live-vs-EMA samples per EMA-lag window). Closes confound C-H2-1 IN the finisher window (the
  trainer's `-1` auto mode covers only run-start warmup — structurally silent at ep726-class engage,
  exactly where the fast d_pose descent re-opens the shadow-vs-live gap). Score-neutral telemetry;
  without it the stop criterion below is not readable.

**Launch-ready invocation (rides the NEXT trunk compile automatically; engage criteria unchanged =
#383):**

```bash
.venv/bin/python tools/launch_witness_run.py \
  --config v9_cgauge_ideal_mod19 \
  --dsl-lever PoseFinisherFilmReadbackArm --dsl-lever PoseFinisherLiveGap \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --dry-run   # drop --dry-run only on operator-GO when the box frees
```

**Dry-run: PASS (2026-07-15T141804Z).** 225/225 flags validated; expected-active-lever manifest OK
(18 pinned + the 2 composed appended); schedule-provenance gate OK (pose-finish 726 = FAIL_SAFE_CAP
governed by --muon-start-event); mem-preflight 24.48 GiB << 108.8 ceiling; system-admission ADMIT
even WITH C0 live. Emitted `launch.sh` line 28 `--pose-carrier-residual-mode film`, line 234
`--verdict-live-gap-every 4`. Artifact (moved OUT of experiments/results so the dashboard latest-run
resolver cannot repoint off the live C0 run — `is_run_dir` matches the levelset_* glob without run.log):
`.omx/research/pose_finisher_ladder_prep_20260715_dryrun/{launch.sh,constants_manifest.json}`
(deterministically rebuildable by the command above).

**Table control arm** = the same command WITHOUT `PoseFinisherFilmReadbackArm` (the sealed incumbent).
**Warm-start finisher A/B variant** (if the operator wants the A/B without a fresh 3000-ep run): the
#366 prep argv (`joint_descent_p0_launch_prep_20260708.md` §1) + the film lever, warm-start
v2_attrclean mod-26, standalone 67.6 GiB SAFE — resumable, per-stage-checkpointed, `--eval-every 5`.

## 5. Pre-registration (BINDING before any finisher window is read)

Let E0 = the epoch the pose_finish gate engages (event or cap). Reference shape = R1 measured
(97→0.0011 in ~108 warm-started epochs, monotone; FEED-poseladder / FEED-posesegdynamics).

- **Expected trajectory (table arm):** d_pose ≤ 1.0 by E0+30 · ≤ 0.01 by E0+70 · ≤ 0.003 by E0+110.
- **CONTINUE** while rolling-20-ep relative d_pose improvement ≥ 3% (read on live-gap-corrected rows
  ONLY — the L3 clearance from FEED-C0H2 binds: no early-window pose verdict off an EMA-lagged row).
- **STOP-SHIP:** plateau (<3%/20 ep) AND ≥9 ep past plateau onset (EMA-settle = derived warmup_epochs)
  ⇒ ship dxi via the #238 connector (`--pose-carrier-xi-from-ckpt` / `--pose-carrier-dxi-scale`),
  byte-close, re-measure n600 through the real inflate. Then fire the L1 gate (§3.3) on the finished
  checkpoint.
- **HARD-STOP / INVESTIGATE:** d_pose > 0.05 at E0+120 (an order above R1's shape at matched window)
  ⇒ finisher defective — diagnose, do not burn epochs.
- **d_seg GUARD (apparatus-validity, not a pose verdict):** d_seg must hold within +2% of its E0 value
  (seg⊥pose 99.95% null, #206; R1 held ~0.0046). Breach ⇒ confound alarm.
- **FILM ARM KILL:** film d_pose > 1.5× table at matched E0+N ⇒ table ships; film registered
  formulation-negative (rate arm dead, verdict_scope: formulation).
- **L1 GATE GO/NO-GO (pre-registered per #250 §6):** GO iff K≤18 coefficients + k≤5 GN reach
  d_pose < 3e-4 at n600 AND projected decode wall-clock (basis recompute + GN, ×600) ≤ 10 min of the
  30-min budget AND applying δframe0 costs ~0 d_seg (structurally satisfied — SegNet reads only
  frame1, `modules.py:108`; frame0 d_seg obligation 8.5e-9, Unit C). NO-GO ⇒ pose floor = the
  finisher's byte-closed number; the ladder honest-closes there.

## 6. HONEST landing-zone estimate (0.127 → ?)

- **Finisher training alone (table or film): contribution 0.10–0.127** (d_pose ~1.0–1.6e-3). Cites:
  R1 plateau ~1.05e-3 training-side (ep1074/1093, FEED-poseladder); n600 byte-close realized 1.6×
  training-side (FEED-238resolved); no store-nothing joint-descent measurement has EVER gone below
  1.0e-3; the ≤12-DOF warp family + photometric wall bound the mechanism (pose_mladder,
  pose_l2_truedepth). **The training rungs do NOT plausibly close the 50× alone** — expected recovery
  ~0–0.03 S (mostly EMA-settle + film's −0.0035 rate).
- **+ L1 gate GO: pose line-item ~0.02–0.03 total** (contribution →~0 + coefficient rate 0.010–0.020)
  ⇒ **recovers ~0.10–0.11 of the ~0.11 S** — the ladder's full worth rides on THIS unmeasured rung.
  Risks (symposium, preserved): coefficient-store is CARGO-CULTED-UNTIL-MEASURED (Contrarian:
  "another linearized-Jacobian argument — measure before believing"); decode-compute is the real
  budget risk.
- **Intermediate case (L1 partial, d_pose ~2e-4 at rate ~0.010):** line-item ~0.055 ⇒ worth ~0.07 S —
  still budget-relevant (d_seg budget 1.26e-3-class at rate 0.030+0.010).

Plan-of-record until the L1 gate returns: **pose = 0.127 is the conservative floor** (symposium §7
discipline); d_seg remains THE blocker; nothing here moves the pointer.

## 7. Triality

- **DSL leg:** `PoseFinisherFilmReadbackArm` + `PoseFinisherLiveGap` + `pose_finisher_live_gap_cadence`
  (curriculum_dsl; registry-discovered composable; tests green). No new trainer flag invented — both
  flags pre-existed and were already DSL-held (`StoreNothingPoseCarrier`/`VerdictLiveGap`); the new
  levers are the finisher-window COMPOSITIONS.
- **DAG leg:** FEED-posefinisher (2026-07-15) appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **Equations leg:** pose-budget law stated in §1; canonical registration
  `# FORMALIZATION_PENDING: pose_budget_law registration owed (task-boundary: src/tac/canonical_equations not in this task's mutation set)`.
