---
schema: ddm_pfs1_posefield_and_recompose.v1
date_utc: 2026-07-29
arm: ddm_pfs1
axis: "[macOS-CPU advisory — real evaluator, real bytes]"
pointer: "0.1910828242 [contest-CPU] UNMOVED"
score_claim: false
promotable: false
pointer_moved: false
research_only: true
council_predicted_mission_contribution: frontier_protecting
verdict_scope: FORMULATION
consumes: [ddm_p3v2_optimal_form_pose_resolve_20260729, ddm_pb1_postburn_completion_20260729,
  ddm_sc1_seeded_scene_carrier_20260728, "741_ep_xi_dual_use_annotation", ddm_deferral_queue_ledger_20260729]
consumers: [ddm_pb1_p6_modal_stageB_staging_20260729, v10_SPEC_row12_pose_in_burn, QA25,
  ddm_deferral_queue_ledger_20260729]
---

# ddm_pfs1 — P5-v2 warp-base recompose (D1) + the e_p pose-field production solve (D2)

## §0 HEADLINE (pointer honesty first)

**Pointer `0.1910828242 [contest-CPU]` UNMOVED.** Everything below is
`[macOS-CPU advisory]`; the composed row is real-evaluator/real-bytes but NOT contest hardware.

- **D1 (MEASURED local row):** the pb1 composed archive re-shipped with the warp-base pose
  carrier, solved on the archive's OWN shipped f1 (stale-frame confound cured, §1). Locked-env
  full-n600 evaluate row: **S = 2.256641** (pose 0.22144216 → 1.488093 + seg 0.00389011 →
  0.389011 + rate 569,996 B → 0.379537), rc=0 — **ΔS = −18.018 vs the pb1 Stage-A 20.2746**
  at +5,116 bytes. Stage-B staging REPOINTED at this archive (sha `624ffe57…`).
- **D2 (MEASURED ladder + falsifier):** the e_p warp-pose-field GN solve reaches
  6-DOF contribution **1.2630** (rank-1 int8: 1.4383 @ **702 B**; rank-4 int8: 1.3159 @
  2,004 B). **The pre-registered falsifier FIRED** (no ≤4KB point beats 0.5) →
  **pose-in-burn returns to REQUIRED in the v10 SPEC**; the residual is typed §5 (tail-
  concentrated, off-homography). The rank rungs stand as rate levers (−6.1KB at ~equal pose).

## §1 The stale-frame confound (caught, cured — why p3v2's 0.3931 did NOT transfer)

p3v2's s3 warp point (n600 mean d_pose 0.3931 → contribution 1.9827) was solved on the ct1
FRAME_ROOT frames (state `2a2c0367…`, the 07-25 e5a identity render). The pb1 archive's own
token-render frame_1 differs from those frames at **max_abs 255** (pairs 0/200/599 probed) —
a different vehicle render entirely. Composing the s3 s_t stream verbatim onto the pb1 archive
would have shipped s_t indices fit to frames the receiver never produces (the staleness
confound, `staleness_is_a_named_confound_class`: freshness must hold at CONSUMPTION). CURE:
D1 re-solved the s_t grid on the archive's OWN shipped f1 (n600 chunked resumable), with the
SHIPPED float16 targets, so the solve point and the receiver reconstruction are the same
object by construction. Receipt: `d1_solve_receipt.json` — n600 mean d_pose **0.22155**
(median 0.0536, max 4.97, min 1.5e-4), contribution **1.4884**; the endpoint's cleaner frames
warp BETTER than the stale ct1 render (0.2725 n8 ballpark confirmed at population scale).
s_t stream: 4 active symbols {0.06:22, 0.08:364, 0.12:156, 0.16:58} — highly compressible.

## §2 D1 — grammar v3 (`ddm_pfs1_composed_archive.v3_warp`)

Only the pose member changes vs pb1 v2: `state/pose.tpgn` (7,295 B raw / 1,876 B Brotli
six-cosine) is REPLACED by `state/pose_warp.stp` = t_p (600×6 float16, Brotli-Q11) + s_t
(per-pair 11-level index, r7 SMEVR frame). Receiver: vendored `pfs1_warp_receiver.py` —
pure-numpy deterministic ground-homography warp (EON intrinsics 910/582/437, H=1.22 m —
the documented literals `tac.clip_profile` reproduces bit-identically on 0.mkv), a
byte-faithful copy of the `measure_pose_warp_dseg` / `measure_screw_warp_through_R` engine
p3v2 used. frame_0 = warp(f1, H(t_p; s_t)) at camera res; NO scorer at inflate (rule 118:
the warp is generic deterministic code; t_p + s_t are the only video-derived payload).

Custody asserts (build receipt `d1_build_receipt.json`, MEASURED): DR7T roundtrip exact;
receiver-reconstructed TR1 packet BYTE-IDENTICAL to the frozen endpoint packet; pose_warp
member roundtrip exact; **vendored warp vs the p3v2 engine max_abs_pixel = 0** on 24
shipped-f1 sample pairs (the byte-identity positive control).

Composed archive: **569,996 B** (sha `624ffe57000c6fe4a6802a6d8b9a5d6002617f29b0bbb9e186d1273fa996600c`):
tokens.dr7t 557,253 + renderer.sec 3,341 + selector.sec 535 + pose_stub.sec 83 +
**pose_warp.stp 6,864** (tp f16 Brotli 6,655 + s_t SMEVR 189 + 20 header) + manifest 1,234.
vs pb1 v2: +5,116 B (pose_warp 6,864 replaces pose.tpgn 1,876) for pose 19.51 → 1.49 —
the f16+Brotli t_p stream codes POORLY vs sc1's AR-int5 ~2,039 B proxy; the D2 rank rungs
attack exactly this member. Instrument prediction: S_pred = 0.38901 (seg, pb1 endpoint) +
1.48844 (pose, D1 solve) + 0.37954 (rate) = **2.2570**.

## §3 D1 — the measured local row (locked-env evaluate.sh, full n600)

**MEASURED (rc=0, full 600 samples, locked env, archive sha `624ffe57…`):**
`Average PoseNet Distortion 0.22144216 · Average SegNet Distortion 0.00389011 ·
Submission file size 569,996 bytes · Compression Rate 0.01518148 · Final score 2.26`.
S recomputed from components (never the rounded field):
0.389011 + √(10·0.22144216) + 25·569,996/37,545,489 = 0.389011 + 1.488093 + 0.379537 =
**2.256641 [macOS-CPU advisory — real evaluator, real bytes]**.

- **vs the pb1 Stage-A row 20.274647: ΔS = −18.018** — the entire move is the pose axis
  (19.5095 → 1.4881); rate +0.00345 (the 5,116-B heavier pose member); seg EXACTLY preserved.
- **Drift rows (instruments vs live evaluator):** d_seg 0.00389011 = the pb1 evaluator value
  BIT-EQUAL (frame_1 untouched — the frame_0-seg-free law confirmed on deployed bytes);
  d_pose 0.22154653 (solve instrument, banked targets) vs 0.22144216 (live GT PoseNet) →
  |Δ| ≈ 1.04e-4 (rel 4.7e-4) — the banked-target instrument agrees with the live evaluator
  at the pb1-calibration class (1.7e-5 there, 1.0e-4 here; both ≪ any decision threshold).
- Receipts: `d1_eval_receipt.json` (+ stdout/stderr) at the SSD custody; eval wall ~19 min.

## §4 D2 — the realization: the shipped warp pose is a FREE control; δ = p* − t_p IS e_p

The evaluator scores PoseNet(f0,f1) against LIVE GT; nothing forces the shipped 6-vector to
equal t_p. So the warp pose is a free per-pair control solved through the REAL receiver path
(frozen CPU-torch PoseNet6, STE-uint8 camera-res): damped Gauss-Newton over θ=[pose6, s_t]
(7 DOF, forward-difference Jacobian, realized-acceptance line search), objective = MSE6 vs
the true banked target. The solved field δ_i = p*_i − t_p_i is exactly sc1's e_p on THIS
vehicle — and #741's structure verdict (e_p temporally near-WHITE, lag-1 0.086 → per-pair
coefficients REQUIRED, no smooth ξ-curve absorbs it) is the design here: per-pair values,
coded low-rank, priced at shipped quantization.

**ROTATION ACTIVATION (new receiver DOF, found in review):** the D1 receiver ships s_r=0
(R=I) — pose dims 3–5 are INERT; the warp is translation-only. D2 solves at s_r=1.0 so all
6 dims act through the full plane-induced homography H = K(R − t·nᵀ/d)K⁻¹ (the s_r=0 family
is contained: rotation dims → 0). D2 rungs are therefore priced for a one-line grammar-v4
receiver amendment (generic code), NOT built/eval'd in this arm.

**Three measured solver findings (each cost one aborted chunk, all receipts in the D2 jsonl
lineage):**
1. **The s_t·dim0 scale ridge is real:** co-optimizing s_t with the pose let GN walk the
   degenerate ridge to points where f16 rounding exploded d_pose (pair 0: f64-solved 0.116 →
   f16-shipped 10.22). Cure: s_t FIXED at the D1 grid value (the s_t direction is contained
   in uniform translation scaling anyway); acceptance moved to the SHIPPED f16 lattice.
2. **Raw t_p rotation dims are POISON as expmap angles:** starting the s_r=1 solve at t_p's
   dims 3–5 lands d_pose 10.82 vs 0.146 translation-only (74×) — PoseNet's output dims 3–5
   are not raw metric rotations in this frame convention. Cure: rotation starts at ZERO
   (expmap(0)=I = exactly the D1 point) and enters only as GN-grown capacity.
3. **The objective is f16-lattice-bound along dim0:** at |dim0|≈34 the f16 spacing is
   0.03125, and a 0.011 rounding moved d_pose by +10 at a ridge point — the pose-input
   sensitivity is razor-sharp along the forward-translation direction. Quantized-acceptance
   GN (accept only f16-REPRESENTABLE improvements) is the honest solver under this lattice.

## §5 D2 — the (contribution, bytes) ladder (MEASURED n600, realized at shipped quantization)

Every rung is a REALIZED point (fresh frozen-PoseNet forwards on the quantized reconstruction
through the exact receiver path), never a ceiling. Stream bytes = the full pose member
(coeffs/values coded + s_t SMEVR 189 B + f16 dirs/mean/scales where applicable). Composed S =
seg 0.389011 + contribution + rate at the recomposed archive size.

| rung | d_pose mean | contribution | pose-stream B | composed S |
|---|---:|---:|---:|---:|
| warp-only s_r=0 (D1 SHIPPED, evaluator-measured) | 0.221442 | 1.4881 | 6,844 | **2.256641 (MEASURED)** |
| warp + e_p 6dof f16 (s_r=1) | 0.159509 | **1.2630** | 6,824 | 2.031503 |
| warp + e_p rank-1 int8 | 0.206878 | 1.4383 | **702** | 2.202784 |
| warp + e_p rank-2 int8 | 0.189858 | 1.3779 | 1,164 | 2.142656 |
| warp + e_p rank-4 int8 | 0.173152 | 1.3159 | 2,004 | 2.081198 |
| free-frame_0 floor (p3v2 §0, CITED) | ~9.1e-5 | 0.030 | unpriced | — |

(int16 variants measured IDENTICAL to int8 to 4 decimals at ~2× bytes — int8 coeff
quantization is FREE on this field. p_star SVD energy [0.970, 0.024, 0.006, …];
e_p delta field SVD [0.906, 0.074, 0.018, …] — rank-~2, matching sc1's rank-1 law on a
different base. δ per-dim std [0.82, 0.12, 0.24, 0.010, 0.007, 0.029] — translation-dominant,
rotation dims small-but-used.)

**FALSIFIER (pre-registered: warp+e_p ≤ 4,096 B must beat contribution 0.5): FIRED.**
The best ≤4KB rung (rank-4 int8, 2,004 B) lands 1.3159; even the unpriced 6-DOF reach is
1.2630. No warp-pose e_p point approaches 0.5.

**The residual, typed (verdict_scope FORMULATION — warp-pose-space e_p on this seg-only
vehicle):** the failure is TAIL-CONCENTRATED, not uniform. Median pair solves to d_pose
0.0027 (essentially closed); but 9.3% of pairs (>0.5) carry 74.6% of the mean and 4.3%
carry 51.8% — the 71–90 turn/dynamic cluster (worst: pair 77 at 4.83). On those pairs the
FULL 6-DOF plane-induced homography family (rotation ACTIVE, run to GN convergence at the
shipped lattice) cannot produce the frame_0 photometric structure PoseNet reads — the
residual is OFF the ground-homography manifold (parallax/non-planar/photometric detail),
the same wall class p3v2 measured as basis-adversarial for cheap pixel carriers. What the
e_p field CAN do post-hoc it does: −28% mean d_pose at ~0 marginal bytes (6dof) or −6.6%
at −6,142 B (rank-1). What it CANNOT do is carry the tail — that needs pose-legible frames,
i.e. pose in the TRAINING loop.

## §6 Routing (the falsifier consequences, per charter + gc8 op-routable 3)

1. **v10 SPEC row-12: pose-in-burn returns to REQUIRED.** The p3v2 "OPTIMIZATION choice"
   framing is superseded by this measured ladder: post-hoc warp+e_p saturates at
   contribution ~1.26 (mean-tail-bound), 25× above the ≤0.05-class gc8 drop-condition.
   The gc8 extension-window pose axis STAYS (its drop-condition "pfs1 closes pose to
   ≤0.05-class post-hoc" is NOT met).
2. **QA25 gets the measured optimization curve anyway** (the ladder above) — the pose-route
   decision point is now MEASURED, not conjectured: cheap post-hoc = 1.26–1.49 class;
   in-burn conditioning is the only named path below it (sc1 e_p ~2KB MEASURED-CLOSED lives
   on a CONDITIONED base; this vehicle is not one).
3. **Rate-side consumers (live now):** rank-1 int8 carries the pose member at 702 B — a
   −6,142 B / −0.0041 S rate lever at −6.6% d_pose vs the D1 shipped member; the 6dof
   f16 stream is the best-S point (2.0315). Both are grammar-v4 (s_r=1 one-line receiver
   amendment). Named consumer: the next recompose (P5-v3) if/when a composed row is worth
   another local eval; NOT re-evaluated this arm (one 19-min slot spent on D1, by design).
4. **Verdict-scope:** FORMULATION for the warp-pose e_p family on this vehicle. NOT a
   paradigm kill: on a pose-conditioned base (v10 in-burn), the terminal e_p solve remains
   the banked closer (sc1/#741 lineage).

## §7 Wire-in (#125) + labels + custody

- sensitivity-map N/A · Pareto: the §5 ladder rows are new advisory (d_pose, bytes) Pareto
  points · bit-allocator N/A · cathedral N/A · continual-learning: this memo + DAG FEED +
  ledger row flips · probe-disambiguator: the pre-registered falsifier IS the disambiguator.
- [no-triality] [p0-ledger-ok] — measurement/composition arm; no DSL lever or canonical
  equation surface changed.
- Receipts (SSD, certify-or-block): `/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/{d1,d2}/`.
- Tools: `tools/pfs1_recompose_warp_base_and_eval.py` (c3fc3f6274) +
  `experiments/ddm_pfs1_ep_warp_pose_solve.py` (8eb3d14594); both ruff-clean, 2 review passes.
- Every negative is INSTANCE- or FORMULATION-scoped with the cure named; no family/paradigm
  kill anywhere. One n600 scorer job at a time throughout (solve → eval → D2, serial).
