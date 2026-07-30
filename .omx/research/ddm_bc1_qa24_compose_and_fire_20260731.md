---
schema: ddm_bc1_qa24_compose_and_fire.v1
date_utc: 2026-07-30
arm: ddm_bc1 (QA24 5-piece composed seg re-burn — compose + validate + fire)
lane_id: "lane_ddm_bc1_qa24_reburn_20260731"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory — trainer builds validated by n8 smoke + 11 unit tests; §3.5 solver plateau MEASURED on GT frames through the frozen CPU PoseNet; NO paid dispatch, NO scorer promotion, NO pointer mutation]"
consumes: [ddm_sg1_segnet_typing_and_reburn_20260731 (§3 the 5 pieces), ddm_ph3_realization_hybrid_adaptive_convocation_20260731 (§8-9),
  ddm_gr1_granularity_rerace_20260730 (cell_drop50 grid), SPEC_tr1 (spec_tr1_renderer_20260728 DSL), tools/launch_tr1_run.py (governed launcher),
  ddm_tt1_twin (differentiable warp), ddm_p3v2_optimal_form_pose_resolve (CPU PoseNet + targets), ddm_eg1_endgame_chain_20260728 (E3 GN), MAIN QA66 pose-tail signal, MAIN v4d gate (S 0.9639878179)]
consumers: [MAIN (§3.5 fire decision + the post-burn re-solve chain), QA24 burn (once §3.5 resolved), v4e/v5 composition]
tokens: [p0-ledger-ok]
---

# ddm_bc1 — QA24 re-burn: ALL 5 pieces BUILT+TESTED+COMMITTED · §3.5 MEASURED-REFORMULATED (delta) · FIRED

## §0 POINTER HONESTY FIRST (the END first — means/ends firewall)

**The exact frontier did NOT move. `0.1910828242 [contest-CPU]` is UNMOVED.** The QA24 heavy re-burn
**FIRED** (governed detached, pid 68621, standing GO) after the full 5-piece atomic build — but the
pointer moves ONLY through a byte-closed `evaluate.py` n600 row, which does not exist yet (the burn is
the MEANS; the endpoint re-solve + exact eval is MAIN's post-burn charter). This unit BUILT all 5
atomic pieces (real, tested, committed), the 5 DSL levers, the launcher venv gate0, RACED QA79
(bicubic), and — after MEASURING that §3.5's absolute bounded-solve form is INSTANCE-DEAD on this
vehicle — REFORMULATED §3.5 (MAIN Option A GO) as the DEGRADED directional-delta and FIRED. Every
number below is `[macOS-CPU advisory]`; `score_claim=false`.

**LAUNCH RECEIPT:** pid 68621 · ticket_hash 81e9f26c239bcc5c · git e0a37e82f4 · out-dir
`/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out` · gates ALL PASS (venv_gate0 · seal_freshness
· import_custody · mem 104.3 GiB vs 25.6 floor · scorer_slot FREE) · variant=lotto D16/c4 w24 400ep/
480min · Metal grouped-backward ACTIVE (~17×) · solve_project init ✓ · ep0 ep_loss 60.3 · per-stage EMA
checkpoints (stage_solve_init_pretrain.npz saved) · ETA ~4h (wall-cap 8h, resumable). Watch: telemetry
row-growth + tr1_window_receipt.json (composed_s_verdict = the directional-delta) in the out-dir.

## §1 WHAT LANDED (real, optimal-form, committed 1dcc71d2d5 + c2082f701d)

The 5-piece QA24 config was composed ATOMICALLY into the DSL + trainer (never a hand-added flag). All
four tractable pieces are REAL + UNIT-TESTED (11 tests pass) + validated by an n8 trainer smoke
(ep_loss descends 244→220; A1 gate fires; coarse grid cuts token bytes):

| piece | surface | status | test |
|---|---|---|---|
| §3.1 coarse-grid cell-mask (384-cell keep-set, from birth) | `TR1Config.token_cell_mask` + `build_module` (token-zeroing, gradient-vanishing) + `token_stream_bytes`/`counted_bytes_ledger` (byte-close exclusion) | BUILT | mask zeros inactive cells · fail-closed wrong-shape · byte-close excludes inactive · ledger < uniform |
| §3.2 margin-weighted loss | `make_loss_fn(margin_weighted=True, margin_weight_temp)` wire | BUILT | (the witness make_loss_fn already implements it; flag wired) |
| §3.3 lattice-anneal (STE@knee) | `quantized_tokens` respects `_quant_engaged`; engaged at the CE→tau knee event + F2 fallback + resume-past-knee | BUILT | disengaged→float · engaged→L16 lattice · off→engaged-from-birth |
| §3.4 rate-in-loss soft-entropy | `_soft_hist_entropy_bits` + `token_rate_term` (entropy / smevr_surrogate) added to `batch_loss` | BUILT | lower for clumped · differentiable |
| 5 DSL levers + qa24 builder | `spec_tr1_renderer_20260728`: `lever_token_cell_mask/seg_margin_weight/rate_in_loss/token_quant_anneal/composed_s_verdict` + `qa24_composed_burn_program` | BUILT | levers valid · composed program validates (fail-closed never-invent-flags) · deterministic ticket hash |
| launcher venv gate0 | `tools/launch_tr1_run.py::venv_custody_gate0` (sg1 §10 owed guard): calls `tools/check_venv_src_custody.py` if present, else inline `.pth`/`__editable__` codex_worktrees scan; REFUSE fail-closed | BUILT | `--dry-run` gate0 PASS |
| QA66 pose-tail hook (MAIN steer) | `--composed-s-subset-ids` (npy of pair indices) → §3.5 runs on the pose-mass TAIL (top-17 = 74.3%) not head | BUILT | DSL wires the flag |

**Governed chain READY:** `launch_tr1_run.py --ticket <qa24 sealed> --out-dir <SSD> --dry-run` → ALL
gates GREEN: `venv_custody_gate0 PASS · seal_freshness PASS · import_custody OK · memory 107.3 GiB free
vs 25.6 floor · scorer_slot FREE`. Sealed ticket hash `aa5046cd292165f8` (16 levers).

## §2 §3.5 COMPOSED-S — THE MEASURED FINDING (decisive across 4 solvers)

**MAIN's recall steer was correct** that a composed-S VERDICT needs only SOLVED d_pose VALUES (no
gradients through a receiver) → the v4c decode-grammar coupling is irrelevant to the OBJECTIVE, and the
razor-sharp landscape needs a damped GN not first-order Adam. I built the proven analytic-Jacobian
LM-GN (STE autograd, not FD — the FD Jacobian is uint8-quantization-noise-limited). **But the decisive
measurement lands on the DEGRADED-fallback branch of MAIN's own decision rule:**

| solver | parameterization | plateau d_pose (GT frames, bounded budget) | pose_contrib √(10·d) |
|---|---|---:|---:|
| Adam | warp-pose6 (STE grad, 1st-order) | ~10 (no converge in 60 iters) | ~10 |
| FD-LM-GN | warp-pose6 (FD Jacobian, 2nd-order) | ~30 (uint8-FD-noise floor) | ~17 |
| analytic-LM-GN | warp-pose6 (STE autograd Jacobian, 2nd-order) | ~29 (200→48 in 1 relin, then STALLS) | ~17 |
| **p3v2 s0_cosine6_solve** (MAIN's recommended production solver) | f0 over rank-6 cosine basis | ~11-16 (p3v2's own memo: 38.06 mean @ ~2 relins) | ~10-13 |

**ALL FOUR plateau at d_pose ~10-38 vs the trustworthy post-burn ~0.0016.** This is FUNDAMENTAL, not
solver quality — the diagnostic proof: `d_pose(GT_f0, GT_f1 vs target) = 9.7e-12` (the target IS
reachable with the REAL f0), but warp/basis pose-recovery from a **single seg-optimized frame_1**
plateaus at ~10-38. The trustworthy d_pose only exists after the **FULL jointly-optimized re-solve**
(pose + photometric + TTO on the fixed f1 — MAIN's ~3.5 h post-burn charter, which the v4d gate proves:
S 0.9640, pose 0.322 achieved on a seg-refined f1 via the full re-solve). A bounded stage-exit ABSOLUTE
solve cannot reach it; `absolute_solve_trustworthy=False` is stamped on the verdict.

Secondary bug found + noted (does NOT cross the plateau): I had `s_t=1.0`; ST_GRID is 0.005–0.24 (the
per-pair v4c translation scale). Fixing it would need the v4c per-pair `st_vals/sel` reference table
(the freshest = MAIN's v4d archive) and still would not cross the warp-of-single-frame plateau.

**Consequence (MAIN's decision rule):** the correct in-burn instrument is the DEGRADED DIRECTIONAL
DELTA — d_pose at a fixed reference, dropped-grid f1 vs the un-dropped/v4d baseline (the Knee-A
externality SIGN, which cancels the ~10-38 absolute offset). It requires MAIN's explicit GO and weakens
the all-or-nothing contract. The `ComposedSVerdict` module is the correct machinery for BOTH MAIN's
joint re-solve AND the delta instrument.

## §3 [SUPERSEDED by §6 — the pre-decision snapshot] WHY NOT-YET-FIRED (the all-or-nothing contract)

> This section recorded the honest NOT-YET-FIRED state at the moment the §3.5 blocker was handed to
> MAIN. MAIN GO'd Option A (the directional-delta) — §6 records the reformulation + QA79 + the FIRE.
> Preserved as the decision-trigger context (APPEND-ONLY).

The 5 pieces are ATOMIC and the §2 pose caveat makes §3.5 REQUIRED for the grid (dropping sky/hood
prices pose on the co9 Knee-A axis). §3.5's trustworthy absolute form is MEASURED-not-a-bounded-solve,
and the DEGRADED fallback requires operator GO. Firing §3.1–§3.4 with §3.5 disabled is firing a partial
config = the dispatch-at-lifted-form trap the OPTIMAL-FORM non-negotiable extincts + the arm's binding
"never launch a weaker state / do NOT fire partial." So the honest action: land the 4 tractable pieces
real+tested, land the §3.5 machinery + the measured finding, and hand MAIN the typed decision.

**Three options handed to MAIN:** (A) GO on the DEGRADED directional-delta §3.5 → wire it (cheap; needs
the v4d per-pair d_pose baseline table) → reseal → fire under standing GO; (B) drop §3.5 from the
in-burn config (`composed_s_gate_subset=0`), fire §3.1–§3.4 now, price Knee-A via the post-burn
composed-S accept gate (the joint re-solve measures it trustworthily regardless); (C) another solver
cycle (uncertain — the plateau looks fundamental). Recommended B (fastest to the critical-path exact
row) or A (in-burn early-warning).

## §6 §3.5 REFORMULATION (MAIN Option A GO) + QA79 + THE FIRE

**MAIN reframed the §3.5 finding, binding (record it):** this is NOT a weaker-state downgrade of the
all-or-nothing contract — it is a **MEASURED REFORMULATION of piece 5**. The absolute bounded stage-exit
solve is INSTANCE-measured impossible pre-joint-re-solve on this vehicle (§2). Therefore:
**LITE-absolute = INSTANCE-DEAD** (this vehicle, pre-joint-re-solve); **LITE-delta = ADOPTED** (the
strongest achievable piece 5); **FULL bilevel (v6, differentiates through the JOINT solve) = unaffected.**

**The ADOPTED directional-delta instrument** (`ddm_composed_s_verdict.delta_verdict` +
`ddm_bc1_delta_baseline.py`): per tail-subset pair, `delta[i] = d_pose(GT_f0, burn_f1) − baseline[i]`,
baseline = `d_pose(GT_f0, GT_f1) ~1e-11` (the un-dropped ideal). Self-contained (GT cache; NO v4c/v4d
archive coupling — v4d's pose_warp magic diverges + the ideal frame_0 isolates frame_1's pose cost
directly). **VALIDATED:** noise floor EXACTLY 0.0 (unchanged GT_f1 → delta 0.0); sky/hood-frozen GT_f1
→ delta 0.14215 = the knee_sensitivity (correct Knee-A pricing). n600 precompute: baseline mean 3e-12,
knee-A sensitivity mean 0.013 / max 0.16, tail-17 = 25% of knee-A mass (9× concentration).

**Four pre-registered interpretation rules (binding, MAIN):** (1) delta is DIRECTIONAL ONLY (sign +
trend of pose-recoverability); NEVER composed into an absolute pose_contrib or endpoint S. (2) stage
decisions use seg descent + delta-TREND; a monotone-worsening delta across 2+ consecutive stage exits =
surface to MAIN mid-burn (advisory, no auto-stop). (3) INSTRUMENT-NOISE FALSIFIER: if the in-burn |delta|
< the measured noise floor (0.0 here → always informative), degrade to Option B silently
(composed_s_gate_subset=0 equivalent) + note; do NOT stop. (4) tail subset = QA66 knee-A top-K.
**Scope-honesty (MAIN, keep it):** the delta does NOT touch the pose-recovery plateau — that is the L68
photometric wall (seg-only frames carry no pose-legible signal). Bicubic/the delta are realized-quality/
early-warning instruments, NOT a cure for the absolute verdict.

**QA79 (bicubic, operator pointer, raced INSIDE before reseal — the eval_roundtrip law):** ENUMERATED
every OUR-code interpolation site: (a) the burn's through-R **render→camera up-lift is ALREADY bicubic**
in BOTH train-R (`apply_contest_faithful_roundtrip_nhwc`) AND the decode-R authority
(`_torch_R_to_camera_uint8`, mode="bicubic") — consistent + deliberate (anti-Gibbs); the camera→scorer
down is FIXED upstream (not ours). (b) the two-plane **warp resampling (`warp_rgb`) is bilinear** — but
it is in the POST-BURN pose decode (frame_0), NOT the burn's seg-only R (d_seg-INVARIANT to it),
correctly deferred to MAIN's post-burn chain. (c) the token→pixel nearest x2 upsample is a learned-
renderer lever (out of the zero-byte resize scope). **RACED** the seg R up-lift bilinear-vs-bicubic on a
frozen GT base (n9, seg-boundary-sensitive): **bicubic WINS** d_seg 0.000960 vs bilinear 0.001044
(−8.4e-5). So the burn's R interpolation is at the raced-optimal (bicubic) already; NO config change; the
dimension is closed. Zero counted bytes.

**THE FIRE (governed, standing GO):** reseal → `--dry-run` all gates GREEN → FIRED via
`launch_tr1_run.py` (G5 detached, `start_new_session=True`/setsid → ppid-1, reaper-surviving). Launch
receipt in §0. The burn optimizes SEG-ONLY (the composed-S delta is verdict-level; it changes NO trained
token/weight/byte). Per-stage EMA checkpoints under distinct stage-encoded names + intra-stage saves +
`--resume-from` = the endpoint artifacts (final EMA checkpoint + stage checkpoints + token payload) will
be in place for MAIN's post-burn re-solve chain (pose re-solve via tt1 twin ~3.5h + photometric ~35min +
TTO terminal + composed gate + exact eval — MAIN's charter; bc1 does NOT run it).

## §7 ph3 §10 OPTIONAL MENU (10th convocation) — cost-evaluated → ALL to BURN-2 (timebox honored)

MAIN's ph3 §10 offered a bounded OPTIONAL-IF-CHEAP menu (QA75-lite / QA81 / QA80) with a HARD TIMEBOX
(≤ ~half a day, EV order, CUT-to-burn-2 when at risk) + a per-piece "if not trivially available → burn-2"
condition. TRUE cost evaluated by reading the surfaces (not guessed) — **all three hit their burn-2
condition:**
- **QA75-lite** (EXACT C1 solve frames as an opening-stage regression target): the C1 solve exists only
  as ARCHIVE-FORM candidates (`ddm_ms2r_r3_*box_tolerance_solve_*/stage_checkpoints/04_candidate`,
  277.7–409 MB) — the per-pixel target FRAMES are NOT materialized as arrays (sg1 §5 named this the
  MATERIALIZATION BLOCKER: "BLOCKED on inflating the exact-solve archive → per-pixel target frames"). Not
  trivially cached → **burn-2 headline #1** (MAIN's own routing).
- **QA81** (composite the cb1 Lane band carrier into the render during training): cb1's byte-closed
  carrier lives in an OFF-MAIN worktree (`ddm_cb1_perclass_carrier_byteclose_20260725T203310Z`), not a
  clean on-main drop-in. → **burn-2 headline #2** (MAIN: "if cb1's carrier doesn't drop in cleanly").
- **QA80** (photometric-consistency term bounded per-pixel below the flip distance d=|m|/‖Δw‖): the
  margin-budget plumbing (‖Δw‖ from the rank-4 SegNet head × the margin map) is NOT a ready surface in
  `src/tac`/`experiments` (grep-empty). → **burn-2 headline #3** (MAIN: "if not trivially available").

**Reseal note (which pieces made the config):** the fired burn carries the 5 QA24 pieces + QA79-bicubic
(the raced-optimal R). QA75-lite / QA81 / QA80 → BURN-2 (cheap by construction: per-stage EMA checkpoints
+ warm-start ride the endpoint; QA75-lite front-loading takes a fresh burn-2). Timebox spent ZERO build on
blocked pieces (cost-evaluated + cut, per the rule) — the burn was not delayed.

## §4 TRIALITY / verdict scope / STORES CONSULTED

- **DAG:** this memo + the committed code (1dcc71d2d5 the 4 pieces + 5 levers + gate0 + solver machinery;
  c2082f701d the §3.5 finalize) + the DAG FEED. **DSL:** the 5 QA24 levers + `qa24_composed_burn_program`
  landed in `spec_tr1_renderer_20260728` (SoT for the composed config). **equations:** no canonical-equation
  surface changed (the §3.5 finding is a MEASURED property of bounded pose-recovery, not a new law) —
  `[p0-ledger-ok]`; the DSL leg IS touched so this is NOT `[no-triality]`.
- **verdict_scope: FORMULATION** — §3.5's absolute-solve plateau is MEASURED across 4 solver
  formulations on the tr1/GT frames (n=2-3 pairs, bounded budget); the direction (bounded solve ≪
  trustworthy) is robust + diagnostic-proven (GT_f0→9.7e-12). §3.1–§3.4 are BUILT + unit-tested +
  n8-smoke-validated (not yet n600-measured through the burn — that is the fire, gated on §3.5).
- **STORES CONSULTED:** CLAUDE.md (NO-FAKE supreme rule, OPTIMAL-FORM, never-launch-weaker-state,
  measured-runnability axis #9, DSL-holds-every-lever, review-gate, serializer); docs/operating_manual;
  sg1 §3 (the 5 pieces) + §10 (venv hijack owed guard); ph3 §8-9; gr1 (the grid); SPEC_tr1 DSL + the
  governed launcher; tt1 (differentiable warp + the razor-sharp FD finding); p3v2 (CPU PoseNet + targets
  + the cosine6 solver + its 38.06 plateau); eg1 E3 (the FD-Jacobian coefficient GN); MAIN's QA66
  pose-tail signal + v4d gate (S 0.9640) + the recall steer + the decision rule.

## §5 APPARATUS NOTES (surfaced to MAIN)

- `tools/check_venv_src_custody.py` is STILL not present (MAIN was landing it). The launcher gate0
  falls back to the inline `.pth`/`__editable__` codex_worktrees scan (fail-closed) and will prefer the
  canonical tool the moment it lands. Custody verified clean this session (`tac.__file__` under
  `src/tac/`; dry-run gate0 PASS).
- The `ComposedSVerdict` geometry depends on the SSD pfs1 warp-receiver tree
  (`ddm_pfs1_20260729/.../submissions/pfs1`) — it FAILS GRACEFULLY (available=False + reason) if absent,
  so the burn never crashes on it. A canonical in-repo copy of `intrinsics_native`/`pose_to_homography`/
  `warp_rgb` (they are fixed EON geometry, 0-byte generic code) would remove the SSD coupling — an OWED
  hygiene item for whichever arm completes §3.5.
