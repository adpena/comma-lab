# ddm_pj1 — PROJECTION PROBE: renderer-class capacity-floor split of the 25.58× gap (2026-07-30)

**Task #788 · gc9 row-2 FORK DISCRIMINATOR.** Pointer honesty FIRST: submittable
**0.1910828242 [contest-CPU] UNMOVED**. Every number below is **[macOS-CPU advisory]**,
`score_claim=false`, `research_only`. verdict_scope tags are the narrowest that the receipts support.

---

## ANSWER (lead)

**f_photometric = realized d_seg 0.504824 (n600, max 0.519241 @ pair 133).** But this number is
**CONFOUNDED and is NOT a valid capacity floor** — MEASURED, not asserted. The photometric
fit-to-solve-frames objective is mis-specified for the (TR1-renderer, C1-solve) pair because the two
live in **disjoint photometric regimes**: the fit dragged the frozen bright renderer toward the dark
solve frames it structurally cannot reach, DESTROYING the argmax (96× worse than the endpoint's
0.00528). **The projection probe as specified does NOT adjudicate the burn-3-vs-class-change fork.**

**Fork verdict for gc9 §4:** row-2 (projection probe) returns **NOT-ADJUDICABLE — photometric route
CLOSED**. The naive gc9 f-table (f ≥ 2.6e-3 ⇒ "capacity wall") would MISREAD this confounded 0.505 as
a capacity wall; it must not. The clean capacity-vs-objective arbiter remains the **QA75 distill-WINDOW
probe (§3 row 5, scorer-in-loop, burn-adjacent)**. verdict_scope: **FORMULATION** (the photometric
projection-probe formulation is falsified as a capacity measurement) — NOT the capacity question
(OPEN), NOT the burn-3 family, NOT the renderer paradigm.

**Collateral finding (decisive, own-its-own):** the gc9 "existence proof" (the C1 solve realizes d_seg
1.52e-4) belongs to a **DIFFERENT VEHICLE** (v10 receiver / ms2r_r3 dark class-field), whose output
manifold is disjoint from TR1's (bright rgb-head). So the **25.58× "amortization gap" is a CROSS-VEHICLE
gap, not measured within-TR1 headroom** — a borrowed-number / cross-vehicle-transfer caution (operating
manual §8 #5). This SUPPORTS the gc9 Assumption-Adversary's CARGO-CULTED classification of "the 25.58×
is attackable by training" — the 1.52e-4 was never a TR1 reachability target.

**The valid TR1 capacity reading that IS available (MEASURED):** TR1's own d_seg-aligned floor =
**0.00528** (endpoint E2, verified below) — an achievable, GT-CE-aligned upper bound on TR1's true
capacity floor at this granularity/rate.

---

## §1 What was measured (receipts, all [macOS-CPU advisory])

| # | quantity | value | custody |
|---|---|---|---|
| R1 | C1 solve frame1 pixel stats (dark class-field) | mean 20.7 / 15.3, std 22.9 / 19.4 (pairs 0, 300) | b2p `qa75_solve_frames` (SolveFrameTargets) |
| R2 | TR1 endpoint render pixel stats (bright gray-ramp) | mean 103.4, std 40.8 (pair 0, camera) | compile of endpoint ckpt |
| R3 | camera L2(endpoint render, solve frame1) pair0 | **93.98 px RMS** | direct |
| R4 | frozen-renderer DARK-reachability floor (120-step token dark-fit) | render mean floors at **67.95** (min 0.61, max 243.8) | reachability smoke |
| R5 | warm_l2 photometric fit plateau (ep75, LR still ~0.01) | **55.50 px RMS** (from 62.1 @ ep1) | `warm_l2/fit_state.npz` |
| R6 | warm_l2 fitted archive bytes | **318,401 B** (−39,808 vs endpoint) sha `f2c2c37779d4c371` | `warm_l2/archive.zip` |
| R7 | **f_photometric = gate realized d_seg (n600)** | **mean 0.504824, max 0.519241 @ pair 133** | `warm_l2/gate/p1_receiver_realized_verdict.json` |
| R8 | per-class d_seg [Road,Lane,Undriv,Movable,MyCar] | **[0.2323, 0.00585, 0.0, 0.01238, 0.25426]** | same receipt |
| R9 | endpoint E2 reference (verified) | d_seg **0.0052766**, archive 358,209 B sha `e7640dee9c3cf41d` | `ddm_782_qa24_endpoint.../p1_...json`; re-derived by my compile |

**The confound chain (airtight):** R1+R2+R3 establish the disjoint regimes. R4 proves it is a **RANGE
wall**, not a fit-difficulty: the frozen renderer cannot produce mean-20 frames even when tokens are
maximally driven to black. R5 shows the fit plateaus at a poor 55.5 px match (consistent with R4). R7+R8
show the consequence: dragging the render dark **collapses the large BRIGHT classes** — Road (0.232,
22.9% area) and MyCar/hood (0.254, 25.6% area) dominate the 0.505, exactly the classes whose argmax
dies when brightness structure is destroyed. Class-2 (Undrivable/sky, already dark) survives at 0.0.
This is argmax destruction from an unreachable-target objective, not a capacity limit.

## §2 Why the photometric probe is invalid here (the reframe the operator needs)

The gc9 §2 probe premise is: *if the renderer can reproduce the solve frames, it realizes their d_seg.*
The premise is UNusable for this pair because the C1 solve is a **different vehicle's output**: the v10
production receiver decode of the ms2r_r3 box-tolerance solve produces a DARK class-field/luma frame
(R1), whereas the TR1 renderer's rgb head is `sigmoid(·)*255` — a bright gray-ramp lift (R2), whose
reachable output set floors at mean ~68 (R4). d_seg depends only on the SegNet **argmax**, and the two
vehicles reach GOOD argmax via completely different pixel patterns (TR1-bright → 0.00528; solve-dark →
1.52e-4). Forcing TR1's pixels toward the solve's pixels therefore pulls its argmax OFF its
argmax-preserving manifold and into garbage (R7). The photometric objective measures the **manifold gap
between two vehicles**, not TR1's own d_seg-expressivity.

Corollary: the QA74 "96.1% attackable / 25.58× over the teacher" typing classified flips by the
**TEACHER's** margin membership — the teacher being this different-vehicle solve. It is TEACHER-relative,
not TR1-relative. The 1.52e-4 was never a TR1 reachability point; it is a *different vehicle's* score.

## §3 What TR1's capacity floor actually is, and how to measure the fork

- **TR1 measured d_seg-aligned floor = 0.00528** (E2 endpoint, R9): TR1 optimized against GT via the
  tau_softplus seg loss converges here at this granularity (24×32 grid, 384/768 kept cells, code_width 4,
  16-level quant). This is an ACHIEVABLE point ⇒ an UPPER bound on TR1's true capacity floor.
- **Whether TR1 has headroom below 0.00528 is OPEN.** It cannot be tested scorer-free by matching the
  cross-vehicle solve pixels (this probe). It requires the d_seg-aligned **QA75 distill-WINDOW probe**
  (gc9 §3 row 5, ph3 §10): resume from E2 with the QA75 lever (target the feasible teacher's SOFT
  SegNet logits/margins, not GT hard labels), 30–60 ep, slope ratio vs plain continuation at matched
  steps. That is scorer-in-loop and burn-adjacent — the honest arbiter of capacity-vs-objective. The
  b2b harness (`tools/ddm_b2b_segnet_field_pass.py`) is the named producer of the solve-frame
  logit/margin distill FIELD that probe consumes (a scorer pass, currently OWED post-burn).
- **Byte-side note:** the photometric-fit tokens compress to 318,401 B (R6), −39,808 vs the endpoint's
  358,209 — the dark-target fit lowered token entropy. Directionally interesting for rate (a
  lower-entropy token field is cheaper) but coupled to the d_seg collapse; not separable here.

## §4 Honesty labels + verdict_scope

- f_photometric 0.505: **MEASURED** (n600 receiver-realized, frozen CPU-torch SegNet authority).
- "photometric probe is confounded / invalid capacity floor": **MEASURED** (R1–R8 chain).
- "TR1 capacity floor = 0.00528": **MEASURED** (E2), labeled UPPER bound on the true floor.
- "25.58× is cross-vehicle, not TR1 headroom": **DERIVED** from R1–R4 + the vehicle provenance.
- "QA75 distill-window is the clean arbiter": **DERIVED** (the only d_seg-aligned route left).
- verdict_scope of the negative: **FORMULATION** — the photometric-projection-probe formulation is
  falsified as a capacity measurement for the (TR1, cross-vehicle-solve) pair. The capacity QUESTION,
  the burn-3 FAMILY, and the renderer PARADIGM are all untouched.

## §5 Custody (paths + shas; certify-or-block, no /tmp)

- Endpoint ckpt: `/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out/checkpoints/stage_seg_trunk_tau_final.npz`
  sha256 `e51178c01d3d8062ac3aa91a2ef064420020cf2338b9956de4ff5178430c685b`, cfg_hash
  `7bee7aedf3b1385b…`, epoch 400 (ema:: = deploy weights).
- Solve targets: `/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa75_solve_frames` (b2p #783;
  `SolveFrameTargets`, 600 pairs, (2,874,1164,3) uint8).
- Fit code: `experiments/ddm_pj1_token_projection_fit.py` (seeded, resumable, atomic ckpts, telemetry).
- Run dir: `/Volumes/VertigoDataTier/pact/ddm_pj1_20260730/warm_l2/` — `fit_state.npz` (ep75 tokens),
  `fitted_checkpoint.npz`, `archive.zip` (318,401 B sha `f2c2c37779d4c37167797969d6ce8b6b35e50e0d6879b46efe2e23bb75358000`),
  `gate/p1_receiver_realized_verdict.json` (f), `fit_telemetry.jsonl`.
- Targets cache (rebuildable from b2p): `/Volumes/VertigoDataTier/pact/ddm_pj1_20260730/targets_scorer_input_f16_n600.npy`
  (675 MB fp16, sha `657d6a1727ddb186…`) — bilinear-down of solve frame1 to scorer-input (384,512);
  REBUILDABLE by re-running the fit (build_targets) ⇒ safe to cold-store/delete.
- gt cache: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (lstars/margins). Slot: fired ONLY
  after confirming wr1 Gate-B gone (pgrep empty); gate wall 143 s (5×120 chunks, seg-batch 12).

## §6 LIVE-HYPOTHESES / DEAD-ENDS / NEXT-IF-RESUMED

- **DEAD-END** — verdict_scope: formulation — scorer-free photometric fit to the cross-vehicle solve frames as a TR1
  capacity probe. The frozen rgb-head range wall (R4) makes it structurally invalid; do not re-run cold
  or margin-weighted variants expecting a different verdict (the range wall is init- and
  metric-independent — margin-weighting cannot manufacture dark output the head cannot produce).
- **LIVE-HYPOTHESIS:** TR1's true capacity floor may sit below its GT-CE 0.00528 — testable ONLY by the
  d_seg-aligned QA75 distill-window (soft teacher margins in the loss), which needs the solve-frame
  SegNet logit/margin field (b2b harness, one scorer pass, OWED).
- **NEXT-IF-RESUMED:** (1) run the b2b SegNet field pass on the solve frames → distill target; (2) QA75
  distill-window: resume E2 30–60 ep with the logit/margin distill loss, slope-ratio vs plain
  continuation → the real capacity-vs-objective split; (3) feed the result into the gc9 §4 decision
  table in place of the (now-retired) photometric row-2.

Pointer delta: **UNMOVED (0.1910828242 [contest-CPU])**. This unit is means, not end.
