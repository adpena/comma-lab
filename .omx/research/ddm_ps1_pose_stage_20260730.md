# ddm_ps1 — THE POSE STAGE on the B-control parent (task #791, fu1 rank-1) — 2026-07-30

**POINTER HONESTY FIRST:** submittable **0.1910828242 [contest-CPU] UNMOVED**. Every number in this
memo is **[macOS-CPU frozen-PoseNet advisory]**, `score_claim=false`, `research_only`. The frozen
CPU-torch PoseNet ran (this arm consumed the scorer slot); no Metal, no paid dispatch, no contest
evaluator, no pointer move. verdict_scope tags are the narrowest the receipts support. No AI
attribution; commits are the operator's alone. `[no-triality]` (pure build+measure apparatus + this
graph leg) `[p0-ledger-ok]`.

## VERDICT (one line)
On the NEW-BEST-SEG B-control parent (a **pure seg trunk**, d_seg 0.005114 realized), the post-hoc
geometric pose program (su2 stage 2: frame_0 carried-ξ warp + terminal 6-DOF GN solve) **hits the L68
photometric wall** — the warp improves the zeros stub ~5.8× (d_pose 160.1 → 27.8) but the terminal
solve **floors at d_pose 20.41 n600** (pose term 14.29), ~2400× worse than v4d (0.0086) and ~120× worse
than the same solver on a pose-legible base (pb1 control 0.172). The mechanism is proven correct (control below); the seg-native f1 simply carries no
pose-legible photometric signal. **Pose cannot be closed post-hoc on this parent; it requires joint
photometric conditioning in the burn (QA80 pose-legibility-from-birth / ea1 N3 / L68 "only JOINT descent
crosses the wall").** su2 stage-2 falsifier "non-improving archive-level pose+rate action" fires at the
INSTANCE level (B-control seg-native parent × geometric-warp family); the geometric-warp FAMILY is NOT
dead (it works on conditioned bases — pb1 control d_pose 0.172).

## STORES-CONSULTED (recall receipts; multi-pass grep; path[+sha])
- **su2 program** `.omx/research/ddm_su2_pose_endgame_program_20260730.md` (merged 6813679636): the
  ordered pose program (stages 2–5); §4 arithmetic (every non-ideal pose term > 0.172141); the QA43
  tail solver `experiments/ddm_su2_qa43_tail_solver.py` (built, 13 tests, NEVER RUN).
- **Parent B-control** `/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/control/checkpoints/stage_seg_trunk_tau_final.npz`
  (sha **feba3b7f1fa34b52…**, ep440); dw1 verdict B_control endpoint_dseg_n600 **0.005114661**,
  endpoint_bytes 259,407 (SMEVR), S 0.6842 (= seg+rate FLOOR, pose assumed 0).
- **Compiled archive** `/Volumes/VertigoDataTier/pact/ddm_zb1_s1_dress_rehearsal_20260730/B_compiled_archive.bin`
  (sha **438bc022fcd835ab…**, 360,735 B research packet); zb1 §4 realized d_seg **0.005114475**, deploy
  parity 1.86e-7 (reused verbatim — no re-compile).
- **ax1 §4b/§5/§6** `.omx/research/ddm_ax1_all_axes_derivation_20260730.md`: frame_0 carried-ξ warp
  (rule-118 free, 0 tokens); falsifier (b) warp WORSE than zeros ⇒ instance-close (did NOT fire);
  §6 QA80 pose-legibility, L68 dissolution requires JOINT.
- **pfs1 solver** `experiments/ddm_pfs1_ep_warp_pose_solve.py`: `WarpPoseOracle` + `solve_pair_gn`
  (damped GN over warp_pose(6), FD Jacobian, f16-shipped acceptance); D1 s_t init
  `/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/d1_warp_solve.partial.jsonl`.
- **p3v2** `experiments/ddm_p3v2_optimal_form_pose_resolve.py`: frozen PoseNet loader + d_pose_u8/pose6_u8.
- **pose targets** `/Volumes/VertigoDataTier/pact/ddm_ms4_metric_producers_and_measurement_20260724T042005Z/pose_metric_n600_batch32.json`
  (schema ddm_pose_metric_custody.v1; rows[i].center 6-vec; MSE surface; sha g3 registry 0c9ce6d0…).
- **ck1 v4a receiver** `experiments/ddm_ck1_build_composed_archive.py`: the BUILT frame_0 warp receiver
  (f0 = warp(f1, H(p_solved; s_t, s_r))) — the §4b mechanism, grammar v4a.
- **pb1 p5 receipt** `.omx/research/ddm_pb1_p5_eval_receipt_20260729.json`: real evaluator, zeros stub
  d_pose 38.06 on the pb1 parent (chain reference; ~19 min n600 CPU).
- **QA80 field** `/Volumes/VertigoDataTier/pact/ddm_zb1_qa80_field_20260730/` (q50 flip-dist 1.8181) —
  the band-lemma budget for any seg-touching photometric touch (S4 precondition).

## THE LADDER (receiver-realized d_pose, frozen CPU-torch PoseNet, banked GT targets, MSE surface)
All three stages measured through the EXACT pfs1_warp_receiver warp path (byte-identical to the engine
per the D1 positive control), on the B-control parent (sha 438bc022).

| Stage | mechanism | n | d_pose mean | pose term √(10·d_pose) | note |
|---|---|---:|---:|---:|---|
| **S0 stub** | f0 = zeros (inert executable meaning) | 600 | **160.0998** | **40.012** | MEASURED n600 (max 201.8, med 158.7); matches QA24 lineage ~160 |
| **S1 warp base** | f0 = warp(f1, H(p0; s_t)), p0 = tp-trans, rot=0 (§4b) | 600 | **27.8185** | **16.679** | MEASURED n600 (med 10.63); 5.8× better than stub → §4b falsifier (b) does NOT fire |
| **S2 terminal solve** | f0 = warp(f1, H(p*; s_t)), p* = damped-GN 6-DOF | 600 | **20.4075** | **14.285** | MEASURED n600 (med 7.55, max 156.55); 502/600 improve vs warp; FLOORED by photometric wall |

### The decisive CONTROL (proves the harness + isolates the cause)
Same solver, **pb1 parent** (p2c_aimed, f1 mean 51, pose-legible), NO swap, pairs 0–5:
warp **0.2474** → solved **0.1720** — reproduces pfs1 D1 (d_pose 0.221) exactly. Same solver,
**B-control** (f1 mean 107, seg-native): warp 7–138 → solved 4–93. The harness is CORRECT; the 30–100×
blowup is PURELY the parent's f1 photometric legibility. **This is L68 confirmed on THIS parent for the
first time.** verdict_scope: instance (B-control seg-native parent) — geometric-warp family intact.

## COMPOSED-S ARITHMETIC (from measured components; no evaluator needed for the verdict)
`S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489`, d_seg = 0.005114475 (zb1 realized),
κ = 6.658589531e-7 /B. seg floor = 0.511448. SMEVR counted rate (259,407 B) = 0.1727285; pose_warp.stp
member = **6,633 B** MEASURED (tp 6,470 f16-Brotli-q11 + s_t 143 + 20 header) → composed rate 0.177142.

| pose outcome | d_pose | pose term | composed S (SMEVR basis + 6,633 B) | vs bars |
|---|---:|---:|---:|---|
| ideal (perfect pose) | 0 | 0 | **0.6886** | > 0.172141 bar; ≈ dw1 S 0.6842 seg+rate floor |
| **S2 terminal solve (realized)** | 20.4075 | 14.285 | **14.9741** | ≫ all bars (pose-dominated) |
| S1 warp base | 27.8185 | 16.679 | 17.3675 | ≫ all bars |
| S0 stub | 160.10 | 40.012 | 40.7011 | catastrophic |

(research-packet basis 360,735 B: S2 realized 15.0415; ideal 0.7561 — same story.)

**Even at the geometric-solve floor the composed S is pose-dominated (14.97).** The seg+rate floor
(0.6886) is itself already above the 0.172141 official bar and the 0.15 target (the seg axis alone,
0.5114, exceeds them) — so B-control is a seg/rate anchor, NOT a target-crossing parent regardless of
pose. Bars for context: 2.2566 REF / 0.9640 v4d line / 0.172141 official.

## STAGE DISPOSITIONS (su2 program 2–5, on THIS parent)
- **S1/S2 (geometric warp + terminal solve): RUN, MEASURED n600.** Verdict above.
- **S3 (QA43 warp-tail, `ddm_su2_qa43_tail_solver.py`): NOT RUN — precondition unmet.** The tail solver
  refines the hardest-residual pairs on a base with a *solvable* residual; on a photometric-walled base
  the residual is not geometric (the whole field is ~5–100, not a tail). su2's >600 B/admitted-pair
  falsifier is moot — no pair is admissibly close. verdict_scope: instance (B-control base fails the
  photometric precondition); the QA43 tail FAMILY stays live for a conditioned base. Reopen when a
  pose-conditioned parent exists.
- **S4 (QA66 photometric refit, OFF-per-pair; QA80 budget): NOT RUN — dominated.** QA66's measured reach
  is ±0.0134 S at +150 B; against a pose term 14.29 the refit is negligible. The QA80 band budget
  (q50 flip-dist 1.8181) is real but a d_seg-safe frame_1 luma touch cannot manufacture pose legibility
  that the base lacks. Reopen post-conditioning.
- **S5 (compose + byte-close): arithmetic complete (above).** The v4a receiver (ck1) realizes the frame_0
  warp end-to-end; the composed S is fully determined by the measured components. The full upstream
  evaluator (~19 min) would confirm S ≈ 14.97 (pose-dominated) — not run, as it changes no verdict.

## WIRE-IN / ROUTING (results → system intelligence)
The pose program's terminal solve is **blocked at the base, not at the solver.** The one directive: pose
must be born pose-legible in the burn — **QA80 `qa80_margin_bounded_photometric` ON + ea1 N3
pose-legibility-from-birth + JOINT descent** (L68). Post-hoc warp+solve on a pure seg trunk is
structurally walled. This corroborates ax1 §6/§10 (Pool C composes ONLY on a conditioned base) and
closes the "why has the pose program never moved S" question: the parent was never pose-conditioned.
Ledger: QA43 → precondition-unmet (instance); QA66 → dominated (instance); §4b warp → MECHANISM
CONFIRMED (falsifier (b) did not fire, warp ≫ stub).

## BUILT + TESTED
- `experiments/ddm_ps1_pose_stage.py` (modes stub/solve/agg): reuses the proven pfs1 oracle + solver by
  swapping `oracle.packet` to B-control (no shared-tool edit); ruff clean; 3-pair smoke + n600 run.
- Deliverable custody `/Volumes/VertigoDataTier/pact/ddm_ps1_20260730/`: ps1_S0_stub_summary.json +
  ps1_S0_stub_dposes.npy; ps1_ladder.partial.jsonl (600 rows) + ps1_ladder_summary.json; logs.

pointer 0.1910828242 [contest-CPU] UNMOVED  ·  [no-triality] [p0-ledger-ok]
