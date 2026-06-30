# CANONICAL RESEARCH INDEX (merged) — the consult-FIRST surface (2026-06-29)

**Authority** `[$0 CPU research-consolidation / advisory]` · **score_claim** false · **promotable** false ·
**ready_for_exact_eval_dispatch** false. This is a CONSOLIDATION (a MEANS). **Pointer UNMOVED: contest-CPU
0.19109982 · contest-CUDA 0.20533003** (`.omx/state/canonical_frontier_pointer.json`).

## PURPOSE — read this BEFORE you design / conclude / kill / launch

Operator binding 2026-06-29: *"worried about signal loss + rediscovery + STARTING LESS OPTIMAL than we could
with perfect recollection."* This file is the single deduplicated, calibration-tagged marshaling of EVERYTHING
we have MEASURED / BUILT / SOLVED / DEFERRED across the five research axes, so the next launch starts at TRUE
optimal form with nothing left on the table. It MERGES five slice indices (now superseded by this file):
`canonical_research_index_dseg_20260629.md` (73e39c402), `…_rate_…` (511a8fb75), `…_vehicle_warp_…` (938db5387),
`…_pose_curriculum_…` (e537dc2f2), `…_infra_floors_…` (860f8e796 + ded04af13).

**Sister non-negotiable:** `[[proactive-recall-consult-own-research-before-concluding-20260630]]` — before
concluding/designing/awfulizing/KILLing/wall-claiming on ANY axis, grep THIS index first. The operator holds the
VISION; this index holds the MEMORY of what we measured.

### How to query it
- **"Is lever X measured? at what value? trustworthy?"** → §3 (the per-axis index tables); read the **status**
  + **calibration** columns (a `direct`/`pre-R` advisory number is NOT a `through-R` or `EXACT` number).
- **"What is the real launch config?"** → §4 OPTIMAL LAUNCH CONFIG (the config pass, real trainer flags).
- **"Is this claim still true / was it retracted?"** → §6 CONFLICTS/SUPERSEDED (cites BOTH sides + the latest verdict).
- **"What is actually a score?"** → §2 MEASURED EXACT ROWS (only `upstream/evaluate.py` n600 byte-closed rows).
- **"What is the highest-EV open thing?"** → §5 TOP OPEN-HEADROOM.

### CALIBRATION LEGEND (binding axis discipline)
- **EXACT** = `upstream/evaluate.py` n600 byte-closed (contest-CPU / contest-CUDA). The ONLY score. ·
  **CPU-adv** = frozen CPU-torch SegNet/PoseNet argmax on cached `lstars` (advisory, NON-PROMOTABLE) ·
  **MLX-rs** = `[macOS-MLX research-signal]` (advisory, NON-PROMOTABLE) · **MPS-NEVER** (PoseNet 23× drift) ·
  **DERIVED / theory-bound** = math, not a measured row · **GROUNDED** = a measured artifact, but advisory axis.
- **direct** = witness logits→argmax vs L* (symbolic partition). **through-R** = realized through the contest R
  operator (bicubic↑384→874 → uint8-STE → bilinear↓512×384) + frozen CPU-torch scorer. **pre-R** = label-space
  numpy probe, NO R (a LOWER bound). **THE CRITICAL GAP: direct ≪ through-R** (direct 0.0022 → realized 0.0064 on
  the exact-L* store).
- **n** = pairs scored (n6/n24/n96/n200/n600). Per FEED-kn, n CHANGES the outcome (driving clip), not just the CI.
- **THE SUB-0.15 GATE (the need):** at the L13 ~72 KB witness rate, direct d_seg < **9.2e-4 → ~0.162 (sub-0.19)**,
  < **3.2e-4 → ~0.110 (sub-0.15)**; realized-axis (L13 0.0481 + stored-pose floor): realized d_seg < **1.25e-3 →
  sub-0.19**, < **8.3e-4 → sub-0.15**. Vehicle-slice F4 pass-line: d_seg ≤ **1.23e-3** at the v2 byte budget.

---

## §1. THE ONE-PARAGRAPH STATE (lead)

**RATE is de-risked CHEAP on the witness and EXHAUSTED on the borrowed frontier; the binding sub-0.15 lever is
the d_seg axis — specifically the S4 trained-through-R Lane-survival residual.** The 0.19110 frontier is a
BORROWED PR101/PR110 entropy-recode (NO-FAKE #7), already at its lossless floor (`byte_delta=0`). Our only
OUR-original byte-closed row is g3 bc20 at S 0.378/0.392 (rate cheap 0.059, d_seg/pose distortion dominate). The
seg axis alone caps at best-case S≈0.184 (label-noise flip-floor ΔS≈0.012), so **sub-0.15 REQUIRES a smaller
representation** (the v2-done-right task-space witness) whose cheap rate frees the budget for d_seg. The whole
program now converges on ONE binding GPU measurement: does the trained amortized level-set generator (openpilot-
seeded, through-R) emit the ragged ±1px Road↔Lane contour at d_seg ≤ ~1.2e-3 (predicted Muon witness ~6–9e-4) at
a small conditioned code? That is what §4's launch config measures.

---

## §2. MEASURED EXACT ROWS (the byte-closed truth table — the ONLY scores)

Source: `.omx/state/active_lane_dispatch_claims.md` + `canonical_frontier_pointer.json`. `score` =
recomputed-from-components (NOT the rounded `final_score`). CPU→CUDA drift is ~+0.034 (pose-dominated); the
CPU-best archive ≠ the CUDA-best archive — never infer one axis from the other.

| Row | Axis | Score | d_seg | d_pose | bytes | sha (prefix) | Note |
|---|---|---|---|---|---|---|---|
| recoded-R3 (pr110 entropy-recode) — **CPU FRONTIER** | [contest-CPU] | **0.19109982** | 0.00056 | 0.0000294 | 177169 | `b46897267…` | Borrowed PR101/PR110 recode (NO-FAKE #7, defensive bank). Lossless rate EXHAUSTED. |
| recoded-R3 (same bytes) | [contest-CUDA] | 0.22528084 | — | — | 177169 | `b46897267…` | +0.034 vs CPU. |
| PR106 format0d — **CUDA FRONTIER** | [contest-CUDA] | **0.20533003** | — | — | 186876 | `9cb989cef…` | CUDA-best ≠ CPU-best archive. |
| **g3 torch_vehicle bc20 — OUR-ORIGINAL dual** | [contest-CPU] | **0.37797132** | 0.00260094 | 0.00034168 | 89244 | `856e3bf07…` | The only OUR-original byte-closed dual row. advisory==contest-CPU to 0.001%. |
| g3 bc20 (same bytes) | [contest-CUDA] | 0.39153009 | 0.00262703 | 0.00048168 | 89244 | `856e3bf07…` | CPU→CUDA d_pose +41%. |
| frontier-waterfill baseline (pre-recode) | [contest-CPU] | 0.19198534 | — | — | 178493 | `b7106c9bd…` | == archived 0.19199. |
| DQS1 selective-decoderq floor | [contest-CPU] | 0.19202828 | — | — | 178559 | `7a0da5d0f…` | ~30 pairset-drop variants all ≈0.1920293 (no movement). |
| pr110pp R3 per-pair pose table | [contest-CUDA] | 0.22616377 | — | — | 178495 | `1ccae18d8…` | **NO TRANSFER** — macOS-CPU selector ordering did NOT survive to Linux CPU → KILL (paradigm intact). |
| frontier-decoder-quant ladder c1/c2/c3 | [contest-CPU] | 0.26287/0.28219/0.35675 | — | — | 169185/167498/159936 | — | Rate-saved, distortion-swamped — all above frontier. |

**Frontier 0.19110 component split:** `d_seg 0.00056 →0.05598 (29.3%) + d_pose 2.94e-5 →0.01715 (9.0%) + rate
0.11797 (61.7%, BINDING) = 0.19110`. → rate is the binding term on the frontier; the seg term (0.056) is
recoverable headroom.

**NOT-a-row (flag):** triple-wave-N6 **0.156006** appears in the ledger only at `active_modal_*_spawning` status —
a PREDICTION, never completed an exact eval. Do NOT cite as a sub-0.16 achievement.

---

## §3. THE MERGED INDEX (deduplicated, calibrated; latest verdict per finding)

### 3A. d_seg LEVERS (the binding axis)

| # | Lever | Status | Magnitude (calibration) | Pointer |
|---|---|---|---|---|
| D1 | **ALL-CLASS DIRECTIONAL (curvelet/anisotropic) Fourier basis** — orient feats to the all-class boundary tangent | **MEASURED — THE decisive lever, ~0 byte** | **−48% all-class** vs −8% lane-only (MLX-rs, direct, n96); n600 −31% alone. ⚠️ circular (built from `gt.lstars`) → self-orientation fixed-point (`--self-orient`) is the byte-closeable form; realized −48% UNVERIFIED. = DM2 (Candès-Donoho cartoon-optimal). | dseg-L1; vehicle-G5; FEED-25t/bi |
| D2 | **KKT capacity-routing (waterfill on margin-saliency)** — `boundary_routing.BoundaryFiLM` | **MEASURED — dominant FLOOR lever, pays ONLY after basis-match** | basis+cap n600 best **0.002447 (−70%)**; capacity-ALONE on isotropic HURTS +6% → STRICT basis-before-capacity. | dseg-L2; FEED-25u |
| D3 | **Margin-hinge seg loss (lensA)** — grad 1.0 on flip set | **MEASURED — saturates the loss-reweight axis** | grad 1.0 on confident flip vs soft-cosine 1.9e-22; realized −16–36% vs CE; anneal target 1.0→0.5. = the witness hard-pixel routing. | dseg-L3 |
| D4 | **KD soft-logit aux (CE-anchor, kd_w=0.3 T=2.0)** | **MEASURED — small, genuinely distinct** | c1 −1.0%; long900 −10.2% (did NOT saturate). NOT redundant with margin-hinge; pure-KD DIVERGES. | dseg-L4 |
| D5 | **Longer curriculum (900ep)** | **MEASURED — cheapest real floor-mover** | **0.002176** (long900 ep800), best DIRECT d_seg on record, still descending. | dseg-L5 |
| D6 | **SDF level-set witness + hosc (the chart itself)** | **MEASURED — BEST witness chart on record** | **n96 hosc 0.00124 @ep950** (through-R surrogate); R-survival transfers (r_added ~9.5e-5). Converged-n600 TARGET ~5.2–6.5e-4 (projected). | dseg-L6; vehicle-C1 |
| D7 | **step_basis activation (learnable slopes gₖ)** | **MEASURED — speed/bandwidth knob at capacity limit, NOT a floor lever** | FINER −18.7% n100→−4.5% n600 (capacity decay); **hosc β-fixed FAILS standalone** (optimizer saturation); step_basis carries the step-native paradigm. | dseg-L7 |
| D8 | **UNIWARD texture down-weight (Fridrich, β=4.0)** — `sal/=(1+β·tex)` | **BUILT — smoke-verified, convergence A/B DEFERRED** | `--margin-saliency-uniward[-beta 4.0]`; add as LATE-STAGE (l7/Muon) lever. | dseg-L8; FEED-lo |
| D9 | **Chroma (SegNet RGB-slack argmax-flip lever)** | **OPEN — UNMEASURED on realized axis; load-bearing seg+pose** | SegNet argmax on RGB ⇒ chroma carries argmax signal at the annulus; a TRAIN lever (frontier decode-side perturb HURT = trained optimum). Every pre-chroma verdict PROVISIONAL. Baked `--chroma` in BASELINE. | dseg-L9 |
| D10 | **Openpilot lane-prior φ1 / structured-init (road-plane SDF)** | **MEASURED + BUILT — 0-byte train-time prior (rule-118 free)** | road-plane SDF lane-attributable **0.000439** vs image-coords 0.000858; separatrix residual 1.9e-5; ships 0 bytes (self-detect roles, NEVER luma-hardcode). | dseg-L10; vehicle-L1; FEED-fs |
| D11 | **Polynomial-fill lane geometry (deg-3 ground-frame centerline)** | **MEASURED — captures lane SHAPE; residual = DASH** | recon false-NEG 0.00046 < target; false-POS 0.00396 = 90% of recon d_seg; lane ≈ 35 floats/frame → ~1-2KB. | dseg-L11; vehicle-L2 |
| D12 | **Ego-hood static-clamp (#139)** | **MEASURED — FREE 0-byte, negligible standalone** | MyCar IoU 0.994 static; value = frees capacity for the boundary. | dseg-L12; vehicle (static core) |
| D13 | **Sub-pixel boundary placement (#149) / oriented 1-Lipschitz SDF ramp = area-coverage AA** | **MEASURED (advisory) — the binding R-survival cure; NOT yet built on the realized trainer** | the (C) wall = sub-pixel COLOR-MIXING in eval bilinear downsample (~24% boundary-band flip); #149 measured 12× collapse advisory. Cure = placement, NOT texture. | dseg-L13; vehicle-C3/C4 |
| D14 | **Round-trip-in-loop survival (R_surv) / train-through-R** | **MEASURED (decomposition)** | flips ∈ R_cap (routing fixes) XOR R_surv (only round-trip/sub-pixel fixes); texture-survival wall ~16% boundary px (sine-only; step-native UNVERIFIED). | dseg-L14 |
| D15 | **NCA continuous-texture witness (#146)** | **MEASURED — AMBER, dominated by SDF** | realized 0.00337 (1.31× frontier), boundary_band_flip 0.079 (HALF polynomial wall); dominated by SDF 0.00124. | dseg-L15; vehicle-M2 |
| D16 | **Wide-SDF ramp σ=1.0 (FREE d_seg cure, 0 bytes)** | **MEASURED-WIN** | R0 flat 0.0273/0.0242 → R1 ramp 0.0230/0.0185 = −16%/−24% at ZERO bytes; texture HURTS full d_seg (it is boundary PLACEMENT). | vehicle-C3 |

**d_seg FLOORS / CAPS (don't re-derive):**
- **Label-noise confident-GT cap ΔS ≈ 0.012** (EXACT contest-CPU advisory, n600): 93.9% of flips at GT-margin
  <0.5 → even a perfect confident-GT fool caps ~0.012. **seg-only best-case S ≈ 0.184 (> 0.15)** → sub-0.15 needs rate.
- **d_seg AXIS headroom reachable ~0.00016–0.0003** (EXACT, frontier existence: frontier hits ~0.0003 @177KB) →
  the axis is reducible 13× below our 0.0021. (RECONCILES with the cap: OUR decoder near its flip-floor; the AXIS has headroom.)
- **Deterministic-render floor (R1, k=0, no trained generator) ≈ 0.0185 bulk / 0.023 full** (CPU-adv through-R) =
  15–40× the budget → the trained amortized-residual generator is REQUIRED. (NO-FAKE: k=0 == 0.0185 exactly.)
- **Realized exact-L* store through R = 0.0064** (the realization gap that kills the pure-symbolic route).
- **Best DIRECT witness 0.002176 (long900) / 0.00124 (SDF hosc n96); best byte-closeable RGB-witness 0.004445** (iso fallback).
- **Manifold ~8-dim NONLINEAR lane-orbit** (AE-knee 8 / MLE 13); linear "store-the-flips" sidecar NO-GO ×3 (rank 53/60) — compressibility is NONLINEAR.
- **Flip-mass: 50% Road / 19% Lane / 13% Undriv** → orient capacity to ALL boundaries. Class order canonical `[Road,Lane,Undriv,Movable,MyCar]`.

### 3B. RATE AXIS (de-risked CHEAP)

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| R1 | **0.19110 frontier rate = 0.1185** (177,169 B), the BORROWED recode | live frontier | EXACT byte-closed n600 | rate-R1 |
| R2 | **Lossless rate on the frontier is EXHAUSTED** — IS already the L21–L32/PR112-L30 recode; every section at entropy floor; finishing-kit `byte_delta=0` | settled | EXACT (arithmetic proof) | rate-R2; FEED-lb |
| R3 | **finishing-kit "−0.005..−0.008 near-certain sub-0.19" = DOUBLE-COUNTED** (cited already-spent R1/R2 bytes + over-counted S12=0 on render substrate) | RETRACTED (NO-FAKE catch) | EXACT | rate-R3 |
| R4 | **L13 non-RGB witness format = −59% rate** (72,217 vs 177,169 B), lossless-parity-proven; pose-carrier ~22.5KB d_pose→0.006 | format PROVEN (rate half) | GROUNDED, 8-pair parity | rate-R4 |
| R5 | **L13 "72KB sub-0.15" was an OVER-CLAIM** — L13-the-vehicle is S≈0.79 (pose closed by format; d_seg=0.0068 NOT closed). The format packages d_seg cheaply; it does not LOWER d_seg. | SUPERSEDED | GROUNDED | rate-R5 |
| R6 | **bc20/G3 = 89,244 B, rate 0.0594** — cheapest real byte-closeable witness vehicle | PROVEN byte-close | **DUAL EXACT [CPU] 0.378 / [CUDA] 0.392** | rate-R6; §2 |
| R7 | **bc20 honest gap:** rate cheap, d_seg-undercapacity dominates → need d_seg ≤ ~0.00087. Rate is NOT the bc20 blocker; d_seg is. | settled | EXACT arithmetic | rate-R7 |
| R8 | **Deterministic backbone rate = 0.0060** (13 partition keyframes) + pose sidecar ~875B (0.0006) = **0.0066** | GREEN on rate | GROUNDED through-R, n96, ONE 10s window | rate-R8; vehicle-W10 |
| R9 | **Store-everything partition rate-WALL = 0.277** (what R8 decisively beats) | settled | GROUNDED | rate-R9 |
| R10 | **BUT the deterministic arm is d_seg-DEAD** (R1 floor 0.0185 = 30–40× budget, S≳2). Reach carried by partition STABILITY, NOT warp. | settled | GROUNDED, NO-FAKE | rate-R10; vehicle-W11 |
| R11 | **C4 (bulk-jitter) explicit store = rate 0.1185 → S≈0.26 alone** → MUST fold into the trained generator C7 at training time (PR95 pattern). The #1 rate decision. | settled (training-time) | GROUNDED | rate-R11 |
| R12 | **PR95 post-hoc coder stack = the FINISHING KIT, all in-tree** (L30 range/arith + L31 colex + L25 temporal-delta + L21/22/23/24/26/29 brotli-friendliness + L32 q11). ≈ −0.005..−0.008. NOT the breakthrough. | tooling ready; 1 build gap | GROUNDED band | rate-R12 |
| R13 | **The ONLY rate build gap = a v2-grammar materializer (~half-day, $0)** to lift in-tree coders onto the witness container | OPEN (low effort) | engineering | rate-R13 |
| R15 | **WRQ score-aware per-tensor weight requant on C7** (decoder ≈91% of an NN archive → largest post-T1 lever) | OPEN (high ceiling) | UNGROUNDED magnitude | rate-R15 |
| R16 | **GR rate architecture** (v2-done-right): context-tree contour-code the 8-dim lane descriptor → FREE eikonal-SDF generator → posterior minimal residual → integer decode | designed | advisory | rate-R16 |
| R17 | **DM3′ low-rank-GLOBAL additive SDF-head** (rank≈16, ~4–10KB/600, rate ~0.003–0.007); $0-test FALSIFIED the per-position SPATIAL GRID (ego-motion → rank-8=95.6%; grid ~8× worse). ⚠️ SUPERSEDED for the BULK by the deterministic stratified warp (vehicle-G7); may still apply to the residual long-tail (OPEN). | designed, alt falsified, bulk-superseded | GROUNDED ($0 rank test) | rate-R17; vehicle-G6/G7 |
| R18 | **Modulation-split FREE-hypernet → INR weights** (COIN++/functa/D'OH): seed-derived hypernet = ZERO counted bytes, only the low-dim latent counted → smallest-rep by construction (FiLM alternative) | designed | advisory | rate-R18 |
| R19 | **rule-118 FREE/COUNTED boundary (the rate law):** generator ALGORITHM + deterministically-generated tables = FREE in inflate.py; LEARNED weights + video-derived payload = COUNTED; hide-data-in-code = FORBIDDEN | binding law | compliance | rate-R19 |
| R20 | **molt compile-the-free-generator** (008 addendum on collab/pact/ main): WASM+WebGPU decode chain (se3/camera/lane_sdf/levelset/range/coord-INR), bit-exact-vs-numpy-fp32 + 30-min contracts | live two-way channel | advisory | rate-R20; infra-§5 |
| R21 | **Pose rate solved, near-free:** stored sidecar 6 scalars × 600 = 7,200B raw / <5KB zlib / ~hundreds B low-rank rank-4 (#140) → d_pose≈0 | solved | GROUNDED | rate-R21; pose-P1/P4 |
| R22 | **Emergent low-dim collapse = design FOR smallest-rep:** induce rank/spectral collapse to intrinsic dim (#110/A6/code-spectral-entropy); intrinsic dims rank-8 lane / 4.07 coarse / pose rank-2 / FiLM ~1.2-of-768; rank FLOOR guard (lane≥8) | design principle | GROUNDED (measured dims) | rate-R22; vehicle-F3 |

### 3C. VEHICLE / WARP / GEOMETRY (the v2 task-space witness)

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| W1 | **Screw/twist SE(3) warp (Chasles) = ~0-byte WIN on physical classes** (reuses stored 6-DOF pose; Road reproduced exactly; hood→identity, sky→rotation-only KRK⁻¹) | MEASURED-WIN | pre-R advisory n96≈n200 | veh-W1; `src/tac/se3.py` |
| W2 | **Stratified per-class warp (the correct model):** Road=ground-homography(pose) [+15–17% d_seg, calib CLOSES via EON fx=fy=910 cx=582 cy=437 h=1.22m] · hood/MyCar=IDENTITY (#139) · sky/Undriv=rotation-only · Lane/Movables=learned residual. A SINGLE global homography is WRONG. | MEASURED (depth×rigidity gradient) | advisory/pre-R n96≈n200 | veh-W2; FEED-iz/ja |
| W4 | **Screw-warp THROUGH R: bulk NOT free-via-warp** — warping a neighbor inherits the SegNet boundary-JITTER floor (~0.008); bulk through-R 0.0048/0.0051 ≈ 4× budget | MEASURED-NEGATIVE (robust) | through-R advisory n96+n200 | veh-W4 |
| W6 | **EXACT-pose (comma2k19 GT) A9 overturn = negative CONFIRMED ROBUST** — exact poses don't beat proxy (0.00251≈0.00256); static-hood (warp-free) = 32% of bulk flips → floor is INTRINSIC per-frame SegNet jitter, NOT warp error | MEASURED-NEGATIVE (GT validated rel_err 1e-4) | through-R advisory n96 | veh-W6 |
| W7 | **Warp-carries-POSE: d_pose 190 was a UNITS BUG (zero-motion null), NOT a wall** — real homography warp carries pose 190→12.6 (−93%) at d_pose-optimal calib | MEASURED (bug fixed) | through-R advisory n6≈n24 | veh-W7; FEED-lj |
| W8 | **DEEP CRUX: d_seg & d_pose demand OPPOSITE warp scales → lossy dual-use REFUTED** — d_seg-optimal s_t≈−0.0014 near-identity; d_pose-optimal s_t≈+0.16 WRECKS d_seg 7×. → **pose stays on the STORED sidecar**; warp's job = a RESIDUAL PREDICTOR (calibrate to MINIMIZE residual ≈ geometric scale). | MEASURED-VERDICT | through-R advisory n6≈n24 | veh-W8; FEED-lj |
| W9 | **Screw-REACH: the BULK partition is intrinsically STABLE for 47+ pairs** (persist NO-warp bulk d_seg ~0.006→0.022 across the window) | MEASURED | through-R advisory n96, ONE ~10s window | veh-W9; FEED-ll |
| W10 | **RATE consequence of W9: partition store GREEN** — k*=47 → 13 keyframes → rate 0.0060 (+pose 0.0066); 10× conservative → 0.060 ≪ 0.277 ≪ 0.191 | MEASURED-WIN (rate) | through-R advisory n96 | veh-W10 |
| W11 | **BUT the binding wall is the deterministic-render d_seg FLOOR, not rate** (R1 ≈0.0185 = 30–40× budget) → REDIRECT to the TRAINED amortized-residual generator composing with the rate-de-risked deterministic substrate = the HYBRID | MEASURED-VERDICT (NO-FAKE) | through-R advisory n96 | veh-W11 |
| C1 | **Single-SDF carrier VALIDATED through R** (1-Lipschitz ramp: lane d_seg 5.9e-4 @192 / 1e-5 @320 — CLEARS ≤1.23e-3) | MEASURED-WIN | through-R advisory n96 | veh-C1 |
| C2 | **MSDF (multi-channel SDF) FALSIFIED/dominated** (~3.6×@192 / ~76×@320 worse; lane is thin/sub-Nyquist, not a corner problem) | MEASURED-FALSIFIED | through-R advisory n96 | veh-C2 |
| L1 | **Road↔Lane (lane-marking) boundary IS the binding sub-0.15 residual (98–99% of flip mass);** openpilot deg-3 centerline IS the separatrix (residual 1.9e-5) → FREE φ1 prior | MEASURED | static-GT advisory n96 | veh-L1 |
| L3 | **openpilot lane HEAD-START built + HONEST rate correction** — from-scratch 0.00586 → conditioned 0.00207 (64.7% recovered); ⚠️ base ~65 KB/600 IMAGE-space iid (rate 0.043), NOT 0.5–5KB (adjacent-frame lane IoU 0.284) → wants the GROUND-FRAME + screw-warp home | BUILT + MEASURED-NEGATIVE (rate) | $0 CPU n96+full-600 | veh-L3 |
| G1 | **The contest = ONE variational action S_τ = 100·d_seg + √(10·d_pose) + 25·rate**, stationarity in the FIXED frozen-scorer Fisher metric (QFT-on-fixed-background, NOT full GR — well-posed + measurable) | SOLVED (framework, formalized in `tac.canonical_equations`) | derivation + 3 theorems | veh-G1; infra-§6 |
| G2 | **Co-location CONFIRMED ×3:** Fisher curvature ↔ (−margin) Pearson 0.978; boundary anisotropy 9.56:1; 96.8% flip-mass in a 2px band → the cheap top1−top2 MARGIN field is a byte-faithful Fisher surrogate | MEASURED | byte-faithful | veh-G2 |
| G4 | **DM1 (per-pair FiLM rank / Stiefel) DEMOTED to SECOND-ORDER** — EXACT $0: PR(M) collapses 2.6× WHILE d_seg IMPROVES 1.9×; per-pair FiLM only re-weights ≤192 FIXED channel patterns → can't localize the moving annulus. The DM1 decisive smoke is MOOT. | MEASURED-VERDICT | EXACT $0 per-stage ckpts | veh-G4; FEED-ip |
| G5 | **v2 conditioning axis = DM2 (oriented byte-free curvelet/WIRE basis, the −48% lever) + a low-rank GLOBAL additive code, NOT DM1, NOT a spatial grid** | MEASURED-DESIGN | EXACT $0 | veh-G5 |
| G7 | **The per-pair conditioning for the BULK = the DETERMINISTIC STRATIFIED POSE-WARP** (W2, grok-confirmed); GAP3 SETTLED: bulk needs NO trained INR → the trained INR shrinks to Lane-survival + small movables | MEASURED-VERDICT | $0 grok-test | veh-G7; FEED-ja |
| F1 | **FiLM rank-1.2 collapse = the MEASURED (2×) d_seg-plateau cause** (multiplicative resonance; PR(M)=1.19) → NEVER vanilla FiLM | MEASURED | advisory M2 | veh-F1; FEED-lg |
| V1 | **Gauge meta-layer BUILT/TESTED/WIRED** (`src/tac/witness_dsl/gauge.py`, 30 tests) — `CANONICAL_GAUGE = SCREW_TWIST · SINGLE_SDF · CONDITIONAL_ON_LANE_PRIOR · RANGE_DELTA · STORE · DETERMINISTIC_FREE` | BUILT | $0 infra | veh-V1 |
| V2 | **The gauge layer names the ONE remaining binding probe = residual DIRECT_LEARNED** (the trained-through-R lane residual = THE GPU run) | BUILT (pointer-mover named) | $0 | veh-V2 |
| V5 | **Movables (GAP1): multi-body = STORE-not-PREDICT, ~0.0008 d_seg, ~750B / store 2700B** | MEASURED-BOUNDED | advisory n96 | veh-V5 |
| O1/O2 | **Originality (NO-FAKE #7):** v2 = the UNOCCUPIED INTERSECTION of driving-scene-recon-with-warp (PSNR) + codecs-for-machines (black-box). HONEST CLAIM = "a novel COMPOSITION of known prior art, NOT a new primitive." 5 genuinely-OURS elements (exact-oracle CELL distortion · physical SE(3) screw warp · SDF-validated-by-SURVIVAL · warp→SDF→openpilot-WZ chain · gauge-canonicalize-for-MDL). | RECORDED (provisional, MEANS≠ends) | $0 lit sweep | veh-O1/O2 |

### 3D. POSE + CURRICULUM / OPTIMIZER / DYNAMICS

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| P1 | **STORED-TARGET pose sidecar: store 6 PoseNet scalars/pair, d_pose≈0, 7,200B raw/<5KB zlib** — the canonical pose SOLVE | GROUNDED, deployed | contest-fact | pose-P1; `scorer_targets.py` |
| P2 | **Pose FROZEN at inference** (trainable stored pose drifts off the exact target) | GROUNDED | quantizr-source | pose-P2 |
| P3 | **`--w-pose 0` in the witness** — pose rides the sidecar; witness's only job = d_seg | GROUNDED, deployed | deepmath | pose-P3 |
| P4 | **low-rank pose codec = rank-4/511 Pareto-dominant** (2,563B, MSE 2.7e-5, −0.0004 rate); naive rank-2/254 net-NEGATIVE | MEASURED torch-CPU advisory | advisory | pose-P4; #140 |
| P6 | **"pose collapse" (d_pose 0.06–0.34) = content-free-latent RENDERED carrier, NOT the stored sidecar** — do NOT re-treat pose as open | GROUNDED (reconcile) | quantizr-source | pose-P6 |
| P11 | **Pose DESCENDS for free with training** (ep50 0.0072→ep488 0.0002) → ruled out as a binding lever when rendered | MEASURED MLX | advisory | pose-P11 |
| C1c | **Witness SHORT curriculum** S0 seed→S1 CE→S2 tau_softplus(0.3)→S3 l7→S4 Muon; SKIP smooth+QAT/C1a/λ/σ; ~1100–2100 ep (vs PR95's 29,650) | GROUNDED, deploy design | deepmath + MLX-port | cur-C1; §4 |
| C2c | **Per-stage measured d_seg dirs:** CE 0.01045→0.00643↓ · tau_softplus →**0.00396 (THE primary drop)** · smooth →0.00423↑ (DROP IT) · l7 →0.00369 · Muon = THE drop (witness PREDICTED ~6–9e-4, unmeasured) | MEASURED MLX-port n600 | advisory | cur-C2 |
| C4c | **MUON_BITES_FROM_STAGE4:** Muon descends d_seg ~32% MORE than AdamW (gap widens monotone); AdamW grad-norm collapses on κ~19 Hessian → jump-to-Muon-early viable | MEASURED contest-CPU advisory | advisory | cur-C4 |
| C5c | **muon-lr = 2e-3** for the witness flat finisher (band 1e-3..2e-3, ceiling 5e-3); NOT 0.03 (6× too hot, from-scratch nanogpt convention) | GROUNDED (witness) | deepmath band | cur-C5 |
| C7c | **τ is TWO temps:** `--tau-softplus-tau`=0.3 (SEG-SURROGATE = reachability floor Δ_min≈0.3) vs render `--softmax-temp` 1.0→0.05 (frozen 0.05 for Muon) | GROUNDED | deepmath + anneal | cur-C7 |
| C9c | **REHEAT at every transition** = rewarmup floor 0.1×/8ep + reset-moments; PARTIAL restart (1.0× full restart re-destabilizes) | MEASURED | advisory | cur-C9; FEED-bu/fz |
| C11c | **EMA decay 0.997, save SHADOW (not live), apply at eval w/ snapshot+restore** — EMA-shadow-lag up to 78× (the "0.505 wall" was an export artifact) | GROUNDED | CLAUDE.md non-neg | cur-C11 |
| C12c | **NCA stabilizers:** grad-clip 1.0 + spike-factor 5.0 (5×-median) + per-boundary reset; n_restarts≥2 keep-best at CAMPAIGN level (ROLLBACK_BRANCH, not a flag) | GROUNDED | deepmath | cur-C12 |
| C14c | **structured-init S0 seed** = `--structured-init` (static-core SDFs, SELF-DETECT roles) + `--lane-prior-phi1` (openpilot deg-3 centerline SDF, FREE 0 bytes); seed → low-freq free → jump to high-freq annulus (NTK) | GROUNDED | FEED-fs | cur-C14 |
| C16c | **Adaptive stacking (#188):** PURE `decide_next_stage` → EXTEND/ADVANCE/RERUN_NEW_CONFIG/ROLLBACK_BRANCH; thresholds slope 1e-6/-1e-5/1e-5, window 300; emit-only, deterministic | BUILT | code | cur-C16; `campaign.py` |
| C18c | **DM1 Stiefel-W + code-spectral-entropy: byte-free; DEMOTED 2nd-order** (PR collapses WHILE d_seg improves → not the binding cause); compose adaptively only if FiLM collapse becomes binding | GROUNDED-but-demoted | deepmath | cur-C18; veh-G4 |
| C19c | **PR95 8-stage forensic** (29,650 ep = 3000 CE + 5650 tau + 1500 smooth + 500 QAT + 9000 C1a-L7 + 2000 λ + 3000 σ + 5000 Muon) — the parent we SUBSET | GROUNDED forensic | intake | cur-C19 |
| C20c | **DSL `curriculum_dsl.openpilot_seeded_opening` validates clean; never-invent** (validate() refuses any flag not in real argparse); 294 tests green | BUILT | code | cur-C20; §4 |

### 3E. INFRA / FLOORS / MOLT / DSL

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| I1 | **Byte-close → DUAL-EXACT pipeline:** `tools/witness_byte_close_and_eval.py` (trained ckpt → int8+brotli archive → numpy+torch inflate → realized d_seg/d_pose → staged contest-CPU cmd); `experiments/contest_auth_eval.py` (refuses n≠600); paired Modal `tools/dispatch_modal_paired_auth_eval.py` | BUILT | infra | infra-§2A |
| I2 | **FREE small-n exact-eval loop:** use **`--video-names-file`** with the first n names (NOT `--batch-size n`); distortion REAL-subset through the actual scorer, BUT rate uses the FULL 37,545,489 denominator → S NOT 600-comparable (distortion go/no-go ONLY) | BUILT (mechanism corrected) | $0 CPU | infra-§2B |
| I3 | **Determinism spine:** numpy-fp32 = bit-identical authority (torch/MLX parity ≥0.9997); `device_or_die` (cuda default, cpu w/ `--smoke`, **mps FORBIDDEN**); MPS training-gradient patch `torch_mps_compat.py` (~104× faster fp32, NEVER authority); seeded everywhere; resumable + per-stage ckpt + EMA-shadow | BUILT (non-negotiable) | infra | infra-§2C |
| I4 | **Scale/containment:** `tools/memory_guard.py` (3-layer; ⚠️ CODE default `DEFAULT_MIN_FREE_GB=30.0` is STALE vs operator ≥10GB → pass `--min-free-gb 10`) + `tools/safe_run.py` (`start_new_session=True`, control-plane-safe). Guard NEVER kills control-plane (custody+identity-gated). | BUILT (telemetry flag) | infra | infra-§2E |
| I5 | **molt FREE-generator path (#187):** owned Python→WASM+WebGPU compiler; LIVE two-way channel `collab/pact/` on molt **main** (pact-collab MERGED+DELETED; our 008 pushed additively to main); report 007 = molt reply (0 missing numpy/scipy symbols, milestone Kernel-A WASM parity). OPEN-Q P1 = is WebGPU/WASM available in the headless `inflate.sh` runner? | live channel | advisory | infra-§5 |
| I6 | **DSL/TRIALITY/QUADRALITY (#189):** `equations (S_τ) → GAUGE (chart choice + cost) → DSL (program) → DAG (work-graph)`; campaign engine = cyclic-recursion + harvest + compose θ*; 294 tests. Registry IDs are descriptive **`*_v1` strings, NOT E0–E12** (E0–E12 is ONLY the gauge.py action-equation abstraction); registry 424 records; `tools/list_canonical_equations.py` | BUILT | infra | infra-§6 |
| I7 | **Contest source = comma2k19 RAV4 segment;** comma10k MEMBERSHIP=NO exact contest frames (0) but SAME RAV4/device IS present (30 frames/26 drives) → SegNet did NOT memorize the clip; 4.67% lane-edge residual GENUINE | GROUNDED | source-fact | infra-§4 |

**INFORMATION-THEORETIC FLOORS [theory-bound]:**
- **S_floor (rate-only) = 0.11797** (25·177169/37,545,489) — LOOSE / REFUTED-as-realizable (assumes d_seg→0 byte-cheaply; the pincer falsified it).
- **T_floor: NO nontrivial PROVEN floor below the few-KB regime** (B_min = uncomputable conditional Kolmogorov complexity); ESTIMATE band ≈ 0.07–0.13, rate-dominated. **KEY: sub-0.15 is a DISTORTION threshold, NOT a byte threshold** — at d_seg=d_pose=0 the 177KB frontier already scores 0.135 < 0.15; at exactly-zero distortion S=0.11797.
- **Task-RD floor S\* strictly inside (0.118, 0.191)** — reachable ONLY by a task-space (feature-space) rep never built; every vehicle sits on the DOMINATED reconstruct-RGB rung (`R_X(D_Y) < R_Ỹ^{E&C}` theorem, magnitude unmeasured "without an exact row").
- **Frozen-instance exploit:** ours is ONE frozen instance → compute the EXACT per-instance optimum, provably ≤ any learned-general codec. Decisive $0 go/no-go = **P-SUFF** (reconstructed-RGB bits the frozen scorer is invariant to).

---

## §4. ⭐ OPTIMAL LAUNCH CONFIG (from-scratch openpilot-seeded witness) — the CONFIG PASS

The marshaled deploy: every measured lever at its own optimum, S0–S5 vehicle composition + the short d_seg-only
curriculum. This IS gate-3 (config pass). **DESIGN at optimal form; NOT a score — the binding S4 GPU measurement
is the only unmeasured cell on the canonical path (veh-V2). Containment: emit-only here; do NOT launch a GPU run.**

### 4a. THE VEHICLE (S0–S5 composition — the gauge V1 made concrete)
- **S0 — calibration header (FREE/tiny).** EON intrinsics fx=fy=910, cx=582, cy=437, h=1.22m (`src/tac/camera.py`); generic algorithm in inflate.py; per-clip globals counted-tiny.
- **S1 — ONE canonical static scene partition (~8–25 KB counted).** SINGLE-SDF carrier (veh-C1 WIN) + **wide-SDF ramp σ=1.0** (veh-C3, FREE −24% bulk, 0 bytes) — NOT MSDF (veh-C2 falsified). Rendered in the **DM2 oriented curvelet basis** (`--self-orient` + bank/freq, NOT `--use-dir` which does not exist) + eikonal-SDF L13 −59% format.
- **S2 — ego-pose on the STORED SIDECAR (~875 B, d_pose≈0).** Store 6 PoseNet scalars/pair (`src/tac/scorer_targets.py`), frozen, low-rank rank-4/511 codec opt-in. `--w-pose 0` (pose NOT rendered; warp dual-use lossy-REFUTED, veh-W8). RANGE_DELTA gauge cell. (Still FREE dual-use: the stored pose drives the deterministic warp at decode at 0 extra bytes.)
- **S3 — per-class warp-type mask + STRATIFIED screw-warp (~0-byte + ~0.2–1 KB).** Road=ground-homography(pose), hood/MyCar=IDENTITY (#139), sky/Undriv=rotation-only KRK⁻¹ (`src/tac/se3.py`); calibrate to MINIMIZE residual (≈ geometric scale, NOT d_seg). Reach carried by partition STABILITY (veh-W9) → ~13 keyframes → rate 0.0060.
- **S4 — the Lane-survival residual through R (THE BINDING LEARNED TERM, ~6–20 KB).** The ONLY real trained payload = the amortized level-set residual generator emitting the ragged ±1px Road↔Lane contour (smooth base 0.00207 → target ≤1.23e-3). Conditioning = **spatial-warp + DM1 Stiefel-W (byte-free) + curvelet basis; MINIMIZE vanilla FiLM** (veh-F1 collapse PR(M)=1.19). FOLD C4 bulk-jitter in here (rate-R11), do NOT store explicitly.
- **S5 — movables residual (STORE-not-predict, ~0.5–2 KB, d_seg ~0.0008; veh-V5).**
- **Decode** = deterministic integer ANS/Ballé, bit-identical CPU/CUDA, NO scorer weights in archive.

### 4b. THE CURRICULUM (short, d_seg-only; ~1100–2100 ep vs PR95's 29,650)
Canonical source = `tac.witness_dsl.curriculum_dsl.openpilot_seeded_opening` (FIXED opening) + `campaign.plan_adaptive_step` (adaptive l7/Muon stacking). Trainer = `experiments/train_levelset_witness_realized_through_R_mlx.py`.

| stage | setting (OPTIMAL) | measured d_seg dir | source |
|---|---|---|---|
| **S0 seed** (FREE, 0 bytes) | `--structured-init --structured-init-include-lane --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate` | seed = Road↔Lane separatrix, residual 1.9e-5 | cur-C14 |
| **S1 CE** (~ep1–300, `ce_to=300`) | full CE (short confidence-calibration), reheat off (opening) | 0.01045→0.00643 (↓) | cur-C2 |
| **S2 tau_softplus** [REHEAT] (~ep300–600) | `--tau-softplus-tau 0.3 --tau-softplus-start-epoch 300` | →**0.00396 (THE primary drop)** | cur-C2/C7 |
| **S3 l7+margin** [REHEAT, adaptive] | `--l7-start-epoch <resume>` (PARKED at `epochs` in opening; engaged on plateau via warm-start); `--margin-saliency-start-epoch`=l7 boundary | →0.00369 | cur-C13/C16 |
| **S4 Muon finisher** [REHEAT, adaptive] | `--muon-start-epoch <resume> --muon-lr 2e-3`, tau + render-temp FROZEN 0.05, reset-moments, muon-lr-floor-fix ON | THE conditioning drop (PREDICTED ~6–9e-4) | cur-C4/C5/C7 |
| **SKIP** | smooth (RAISES d_seg +6.8%), QAT/C1a/λ/σ (rate machinery — STRUCTURAL skip, not a flag) | — | cur-C2 |

**Cross-stage knobs (baked into BASELINE / opening):** `--self-orient --reorient-every 50 --freq-across 32 --n-dir-freqs 2 --freq-along 4 --max-bank-freq 64` (DM2 directional/curvelet, warm-start-safe value sweep for coarse→fine) · `--chroma` (d_seg+pose lever, every pre-chroma verdict provisional) · `--ema-decay 0.997` (SHADOW saved, eval-only snapshot+restore) · `--grad-clip 1.0` · `--accum-pairs 8` · `--eikonal-weight 0.01 --length-weight 0.001` (live derivative/integral regularizers) · `--stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 --stage-transition-rewarmup-shape linear --stage-transition-reset-moments` (REHEAT 0.1×/8ep) · `--w-seg 100` · `--film-stiefel` + `--code-spectral-entropy-weight <β>` (the byte-FREE A3 rank lever — NOT `--film-per-layer`/`--film-concat-code` capacity; M2: PR(M)=1.19 needs Stiefel) · UNIWARD `--margin-saliency-uniward --margin-saliency-uniward-beta 4.0` as a LATE-STAGE (l7/Muon) lever.

**Adaptive stacking (#188, deterministic, emit-only):** `campaign.decide_next_stage` (window 300) → EXTEND (slope ≤ −1e-5, resume final) / ADVANCE (|slope| < 1e-6 plateau, stack next + reheat) / RERUN_NEW_CONFIG (plateau above rerun_floor → sharper config, e.g. tau 0.3→0.2) / ROLLBACK_BRANCH (final − best > 1e-5 → resume BEST + skip). Curvelet scale climb via warm-safe `--max-bank-freq` sweep (16→32→64); shape-changing flags (`--bank-n-scales/--hidden-dim/--mod-dim`) force a FRESH arm.

**Determinism / containment / launch hygiene:** single recorded `--seed`; per-stage + ≤25-ep checkpoints (`--ckpt-every 25`); EMA-shadow; `--resume-from`-compatible; FROM SCRATCH (`resume_from=None` — structured-init IS the seed, NOT the PR95-curriculum ckpts). Daemon launch carries `--min-free-gb 10` (the DSL Contain default; the memory_guard CODE default 30 is stale — pass it explicitly). spawn_durable_daemon, one GPU, no autonomous heavy launch.

### 4c. BYTE-CLOSE → DUAL-EXACT (what turns S4 into a score)
`tools/witness_byte_close_and_eval.py` (ckpt → int8+brotli archive whose `st_size` IS the rate term → numpy+torch inflate → realized d_seg/d_pose → staged contest-CPU cmd) → `experiments/contest_auth_eval.py --device {cpu,cuda}` (refuses n≠600; recomputes score from components, refuses >0.01 formula divergence) → paired `tools/dispatch_modal_paired_auth_eval.py --execute` (CPU Linux x86_64 + CUDA T4 on the EXACT same bytes). numpy-fp32 = authority; MPS NEVER. Free small-n distortion go/no-go via `--video-names-file` (first n names; rate NOT 600-comparable).

---

## §5. TOP OPEN-HEADROOM (ranked by sub-0.15 leverage)

1. **[THE binding gate] S4 — the trained-through-R Lane-survival residual = THE GPU run** (veh-V2). Does the amortized generator emit the ragged ±1px contour at d_seg ≤ 1.23e-3 (predicted Muon ~6–9e-4) at a cheap conditioned code beating PR95's rate? UNTESTED. §4 IS the config that measures it.
2. **[$0 PRECURSOR] Is the per-frame SegNet JITTER content-predictable or white-noise?** (veh-W6 open door). If predictable → a conditioned generator emits it cheaply (door OPEN); if ~white decision-noise → even a trained generator can't (door mostly CLOSED). The cheap S4 de-risk. = the **P-SUFF** measurement (how many reconstructed-RGB bits the frozen scorer is invariant to).
3. **Realized-axis verdict for the FULL lever stack** (directional+step_basis+chroma+sub-pixel TOGETHER) — every big d_seg number is DIRECT/MLX-rs; the R-survival gate (0.0024 direct → realized) is UNMEASURED for the composed stack.
4. **Sub-pixel boundary placement built on the realized trainer (D13)** — the #1 R-survival lever, measured 12× collapse advisory but NOT yet in `train_*_through_R_mlx.py`.
5. **Byte-closeable directional basis (self-orientation fixed-point)** — the −48% lever is circular-GT; resolve iso→own-argmax→tangent→dir and RE-MEASURE realized −48%.
6. **GROUND-FRAME lane coding rate (veh-L3)** — does static-lane + screw-warp hit 0.5–5KB vs the 65KB image-space iid? The rate-half closure for the lane.
7. **v2-grammar materializer (~half-day, $0, rate-R13)** — the ONLY rate build gap to harvest the entire finishing kit; pays the moment a real C7 exists.
8. **WRQ score-aware weight requant on C7 (rate-R15)** — UNGROUNDED magnitude, highest rate CEILING (C7 ≈ 91% of bytes); needs its own exact sweep once C7 has descended.
9. **CUDA-axis pose drift headroom (infra-§7)** — every recode is ~0.034 worse on CUDA; a CUDA-targeted recode on the PR106 0.20533 archive is unexplored.

---

## §6. CONFLICTS / SUPERSEDED (cite BOTH sides + the LATEST verdict — no signal loss)

- **"Rate EXHAUSTED on the frontier" (rate-R2) vs "rate CHEAP on the witness" (R4/R6/R8)** → NOT contradictory, **DIFFERENT OBJECTS.** EXHAUSTED = the 0.19110 BORROWED PR-recode (entropy floor, `byte_delta=0`). The witness is a SMALLER representation with its own cheap rate (backbone 0.0066 + SDF head + C7). The way to cut rate now = a SMALLER REPRESENTATION (v2-done-right), NOT a coder on the frontier.
- **Witness RD-curve: B*~122 KB→S0.134 (FEED-cc) vs B*≈150 KB→S0.166–0.186 (later anchored)** → RECONCILE: sub-0.15 at B* ONLY in the optimistic + directional-ON corner; the bare-witness anchored optimum clears sub-0.19 but NOT sub-0.15. Both AGREE the 89 KB single point (S0.216) was deep in the cliff, NOT the optimum → **right-size UP; map the curve, not a point.**
- **"finishing-kit −0.005..−0.008 near-certain sub-0.19" (rate-R3) → RETRACTED** — double-counted already-spent R1/R2 bytes + over-counted S12=0 on a render substrate. NO-FAKE caught it.
- **"rate IMPROVES at n600 via amortization" (FEED-kn) → RETRACTED** — for a forward-driving ~60s clip (~15 scene-turns), keyframe cost GROWS with n; n96 is the artificially-cheap low-turnover regime.
- **"free dual-use warp lowers BOTH d_seg and d_pose" (grok G9) → REFUTED for the lossy arm (veh-W8)** — d_seg & d_pose want OPPOSITE homography scales; pose stays on the STORED sidecar; warp = residual predictor. (Survives in the FREE sense: stored pose drives the deterministic warp at 0 bytes.)
- **DM1 (per-pair FiLM rank / Stiefel) as the binding d_seg lever → DEMOTED to second-order (veh-G4)** — PR collapses WHILE d_seg improves; per-pair FiLM can't localize the moving annulus. DM1 smoke MOOT. (Stiefel-W still useful as a byte-free rank lever; vanilla FiLM NEVER.)
- **DM3 per-position SPATIAL LATENT GRID → FALSIFIED (veh-G6)** — variation globally low-rank (rank-8=95.6%); grid ~8× worse + ~100× bytes. Refined to DM2 + low-rank-global, then SUPERSEDED for the BULK by the deterministic stratified warp (veh-G7); may still apply to the residual long-tail (OPEN).
- **MSDF carrier → FALSIFIED/dominated by single-SDF (veh-C2/C1).**
- **"Free deterministic bulk via warp" → REFUTED ROBUST (W4→W6)** — the bulk carries the same per-frame texture-jitter wall (~2–4× budget) as the lane. Survivor = warp/SDF/openpilot PRIOR conditioning a TRAINED generator (the hybrid).
- **"Movable medial-axis is the IRREDUCIBLE chart gap" → OVER-LABELED (veh-M3)** — the binding residual is the Road↔Lane separatrix (98–99%), not Movable; FIX-B eikonal-relax DE-CONFIRMED.
- **"d_seg IRREDUCIBLE (label-noise)" vs "d_seg CAPACITY-LIMITED (13× headroom)" → RECONCILED (framing):** OUR decoder near its label-noise flip-floor; the AXIS is reachable to ~13× lower (frontier 0.0003). Lead both with the SAME composite verdict.
- **"0-byte decode-side levers give a free sub-0.19" → NO 0-byte sub-0.19 row** — the frontier decoder is trained-through-R (a trained optimum) → generic decode perturbations move AWAY (all measured WORSE). Levers are TRAIN-time on a fresh witness.
- **"store-the-flips linear sparse sidecar" → NO-GO ×3** (rank 53/60; compressibility is NONLINEAR).
- **muon-lr conflict (0.03 vs 2e-3 vs 2e-4 vs 3e-3)** → **witness finisher = 2e-3** (FEED-fi band). 0.03 = from-scratch nanogpt convention (wrong regime); 2e-4 = PR95's 229K HNeRV (different model); 3e-3 = A/B contrast only.
- **low-rank pose codec (rank-2 "2.7× smaller" vs rank-4/511)** → **rank-4/511 Pareto-dominant** (smaller AND lower MSE); rank-2/254 is net-NEGATIVE.
- **τ single vs two temps** → TWO: seg-surrogate `--tau-softplus-tau`=0.3 (= Δ_min) vs render `--softmax-temp` 1.0→0.05 (frozen 0.05 for Muon). No contradiction with the anneal-memo.
- **MD-Decoupling "wired" vs "trainer-gap"** → both true: WIRED in `train_witness_realized_through_R_mlx.py` (`--optimizer md`), trainer-GAP in the level-set trainer (the reheat IS the stable-transition mechanism there). PARALLEL arm only.
- **"L13 72KB lossless-parity sub-0.15" (rate-R5) → SUPERSEDED** — L13-the-vehicle is S≈0.79 (pose closed, d_seg open); the −59% rate WIN stands.
- **Frontier score literals** — the ONLY canonical numbers are contest-CPU **0.19109982** + contest-CUDA **0.20533003**. SUPERSEDED: "0.19199"/"0.192028"/"0.19205" (older DQS1/PR-body) + CUDA "0.2262100217" (older PR-body ≠ 0.20533). triple-wave-N6 "0.156006" = PREDICTION only.

### Telemetry-accuracy corrections (code-vs-memory drift, OWNED this sweep)
1. **memory_guard floor:** operator binding ≥10GB but `tools/memory_guard.py:101 DEFAULT_MIN_FREE_GB = 30.0` (vendored-from-molt, STALE) → pass `--min-free-gb 10` explicitly (the DSL Contain default already does).
2. **free small-n eval:** the shorthand "`--batch-size n` scores first n pairs" is IMPRECISE — `evaluate.py` has no `--num-samples`, `--batch-size` is the dataloader batch (default 16); the real subset knob is **`--video-names-file`**. Distortion subset-real, rate full-denominator, NOT 600-comparable.
3. **range coder path** = `src/tac/lossless/range_coder.py` (NOT top-level `src/tac/range_coder*`).
4. **canonical-equations registry IDs** are descriptive **`*_v1` strings, NOT E0–E12** (E0–E12 is ONLY the gauge.py action-equation abstraction).
5. **molt collab** = `collab/pact/` on molt **main** (pact-collab branch MERGED+DELETED; our 008 is additive on main).
6. **`--code-spectral-entropy` flag** is actually `--code-spectral-entropy-weight` in the trainer; `--use-dir` does NOT exist (directional = `--self-orient` + bank/freq).

---

**NO-FAKE ledger:** every value carries its calibration (EXACT / CPU-adv / MLX-rs / GROUNDED / DERIVED; direct vs
through-R vs pre-R; n). No score moved by this index — it is a MEANS (the marshaled toolbox); the END is a
byte-closed n600 `upstream/evaluate.py` row below 0.19110 from the §4 witness. Levers with no measured row are
tagged OPEN, not MEASURED. **Pointer UNMOVED contest-CPU 0.19109982.**
