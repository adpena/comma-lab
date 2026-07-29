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

- **D1 (measured):** the pb1 composed archive re-shipped with the warp-base pose carrier.
  n600 warp solve on the archive's OWN shipped f1: mean d_pose **0.22155** → pose
  contribution **1.4884** (better than p3v2's stale-frame 1.9827). Local locked-env
  evaluate row: [D1-EVAL-FILL].
- **D2:** [D2-FILL — ladder + falsifier side].

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

[D1-EVAL-ROW-FILL]

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

## §5 D2 — the (contribution, bytes) ladder [MEASURED, realized at shipped quantization]

[D2-LADDER-FILL]

Falsifier (pre-registered): warp+e_p ≤ 4,096 B must beat contribution 0.5. [D2-VERDICT-FILL]

## §6 Routing

[D2-ROUTING-FILL]

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
