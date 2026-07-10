# Kalman innovation-coding rate lever on the temporal carriers — MEASURED (marginal pose / negative lane) — 2026-07-10

**Task:** #kalman (operator 2026-07-10, via johndcook.com/blog/2016/05/24/kalman-filters-and-bottom-up-learning +
"openpilot uses kalman filters"). $0 CPU, existing banked artifacts, NO training, NO heavy launch. Does replacing
our temporal-delta carrier coding with a Kalman **innovation** stream (store the whitened one-step prediction
residuals of a smooth-dynamics process model — the classical optimal predictive coder) buy rate at matched
reconstruction?

**One-line verdict:** NO meaningful win. Our existing **temporal-delta coding already IS the optimal
random-walk innovation coder** for both temporal carriers. A higher-order (constant-velocity / constant-accel)
Kalman process model gives **pose −1.5%** (marginal) and **lane +14%** (actively WORSE). `[macOS-CPU advisory]
NON-PROMOTABLE`. **Pointer contest-CPU 0.19110 UNMOVED.**

---

## The matched-error protocol (rigorous — bit-identical reconstruction)

Both the incumbent temporal-delta and the Kalman-innovation coder are **lossless predictive coders over the
IDENTICAL per-channel quantized grid**. Temporal-delta = a predictive coder whose forecast is "previous sample"
(the innovation of a **random-walk / Brownian** process model). Kalman-CV/CA = a predictive coder whose forecast
is the process-model forecast `x̂+v(+½a)`. Both code integer residuals over the SAME quantized signal in
closed-loop DPCM (decoder replays the same forecast on reconstructed values), so:

- **Reconstruction is bit-identical between the two coders** → the realized distortion (d_pose for the ξ carrier,
  induced lateral RMS for the lane carrier) is **identical by construction**, set solely by the shared
  quantization step. Matched error is not approximate — it is exact.
- For the pose ξ carrier this means d_pose = the ALREADY-MEASURED shipped value **0.001127 (n24) / 0.001610
  (n600)** (`r1_dxi_shippability_byteclose_20260708.md`) regardless of which predictor codes it. No pose re-decode
  needed — the reconstructed ξ_eff is byte-identical to the shipped one.
- For the lane carrier both coders reconstruct the same coherent-slot quantized matrix → **induced_lateral_rms = 0
  for BOTH** (unlike the prior lossy RTS-smoother row @ 9.5 m in `wave_f_lane_tracking_rate_n600_RESULT.json`,
  which threw away real geometry to get its −34%).

The comparison is therefore a **pure lossless-recoding** contest: same signal, same distortion, only the predictor
differs. Ratio = bytes(Kalman-innovation) / bytes(raw-delta). Same entropy backend both sides (the byte-close
range-coder `tac.lossless.range_coder.encode_static_symbols` + brotli-q11 counts model for ξ; brotli-q11 over the
zigzag-delta matrix for lanes).

## Artifact 1 — the R1 pose ξ sidecar (`pose_carrier.xi_stored + dxi`, 600×6 se(3) twist)

Source: R1 checkpoint `levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z/levelset_witness_ema_mlx.npz`
(the banked 20×-refinement ξ_eff, ~7.2 KB coded section). Quantized on the shipped grid q_levels=4096.

| predictor (process model) | arith bytes (6 ch, 600 pairs) | order-0 entropy | ratio vs raw-delta |
|---|---|---|---|
| **raw temporal-delta** (random-walk) | **6602 B** | 4037 B | 1.0000 |
| Kalman constant-velocity (α-β) | 6504 B | 4013 B | **0.9852 (−1.5%)** |
| Kalman constant-accel (α-β-γ) | 6506 B | 4015 B | 0.9855 (−1.5%) |

Structure validated: raw-delta replicate **6602 B** ≈ the memo's reported byte-close ξ coded **6634 B**. Per-channel
the CV forecast DOES shrink the residual magnitude ~15–19% (e.g. ch0 |Δ| 1194 → |innov| 966 q-units) — but the
entropy coder barely moves (~1.5%) because the residual is a broad, near-Laplacian jitter: a 20% magnitude drop is
≈0.3 bit/sample, ≈100 B over the whole table. **Interpretation:** the trained `dxi` is high-frequency,
near-white refinement — the very thing PoseNet responds to (it is signal, not smooth drift). There is little
smooth-dynamics redundancy left to whiten; the α-selection floored at the lowest velocity gain, i.e. the signal is
near a random walk and "previous-sample" is already near-optimal. This is the quantitative confirmation of the
#238 memo's aside — *"the coder barely helps because the per-pair dxi adds high-freq jitter that kills the
temporal-delta smoothness."*

## Artifact 2 — the #234 coherent-slot lane-coefficient tracks (600×66 poly-coeff matrix, K=6 slots)

Source: regenerated at n600 from `gt_n600.npz['lstars']` via `build_lane_band_pairs_from_lstars` →
`coherent_slot_pack` (Hungarian slot-tracking, the #234 correspondence-first carrier; ~23 s fit, labeled
regenerated-not-persisted). Quantized on the LBND2 principled per-dim steps.

| predictor | brotli-q11 bytes (matrix+presence) | order-0 | |res|mean | ratio |
|---|---|---|---|---|
| **raw temporal-delta** (random-walk) | **40934 B** | 32995 B | 190.8 | 1.0000 |
| Kalman constant-velocity (α-β) | 46566 B | 36432 B | 226.3 | **1.1376 (+14% WORSE)** |

Structure validated: raw-delta replicate **40934 B** ≈ the prior `coherent_slot_none` **41303 B**. The velocity
model makes residuals **LARGER** (190.8 → 226.3). **Mechanism:** even after coherent packing, the lane matrix
carries carry-forward **holds** (absent slots) and **births**; a constant-velocity forecast projects stale
velocity across a hold or into a birth → an overshoot innovation exactly where the signal is discontinuous. This
is the same index-permutation/discontinuity failure the survey's swap-theorem predicted and that killed the
LBND3 ego-DPCM predictor (measured WORSE). Innovation coding with a momentum state is **actively harmful** on a
carrier with birth/hold discontinuities.

## The deep-math synthesis (folds into the #318 DE-derivation lineage)

Cook's engineering-first framing (johndcook.com/blog/2016/05/24) is the right lens: the Kalman filter is the
recursive least-squares predictor of a linear-Gaussian state; **innovation coding is optimal predictive coding
IFF the process model matches the signal's dynamics.** Both our temporal carriers are best modeled as **order-1
random walks**, and temporal-delta is *exactly* that model's innovation coder — so we are already at the optimum.
Adding a velocity state (the CV/CA lever) pays only with genuine smooth momentum, which neither carrier has:

- **Pose ξ:** near-white trained jitter (the dxi is irreducible signal PoseNet reads) → CV wins a marginal 1.5%,
  not worth the decoder complexity + the loss of the delta path's simplicity.
- **Lane coeffs:** birth/hold discontinuities → a momentum predictor overshoots → +14% (dominated), reproducing
  the survey's CORRESPONDENCE-FIRST theorem at the innovation-coding layer. The lane rate lever remains
  correspondence-first + **edge-preserving** (batch, discontinuity-aware) denoise — NOT a causal momentum
  predictor. (openpilot/rednose conventions: the ego Kalman there estimates a *smooth continuous* ego state — the
  regime where innovation coding wins; our carriers are the residual jitter + discontinuous label series, the
  regime where it does not.)

**Net for the campaign:** this FORECLOSES "add Kalman innovation coding to the temporal sections" as a rate lever
for both v7.5.x and v8 temporal carriers — a measured NO, not a guess. It is system intelligence (the direction
is closed with receipts), not a pointer move. The existing temporal-delta backend is confirmed near-optimal.

## Detector note (design-only) — #383 pose-gate rolling-slope + plateau classifiers vs NIS

The Kalman literature's convergence/health test is the **normalized innovation squared (NIS)** — `εₜ = νₜᵀ Sₜ⁻¹ νₜ`,
χ²-distributed with dim(ν) dof under a correct filter; a windowed Σεₜ outside the χ² band flags divergence/stall.
Our `#383` pose-gate rolling-slope detector (`tools/levelset_pose_gate.py`) and the power-law plateau classifier
(`tools/fit_powerlaw_plateau_detector.py` / `witness_control/sigma_min_plateau.py`) test a **rolling slope of a
scalar metric (d_pose / loss) crossing ~0** to declare a plateau. Honest mapping: NIS and rolling-slope answer
**different questions** — NIS tests *"is my process/measurement model self-consistent right now"* (a per-step
whiteness test), whereas the rolling-slope tests *"has the optimization stopped improving"* (a trajectory-trend
test). NIS would be a genuine upgrade ONLY if we had an explicit dynamics model of the training-metric sequence
(we don't — the loss trajectory is not a linear-Gaussian state), and it would add a scale-free χ² threshold in
place of the hand-tuned slope band. But the plateau signal we actually want is trend-of-the-mean, which the
rolling slope already captures directly; recasting it as an NIS test would import a filter we'd have to fabricate
and does **not** improve on the current band. **Verdict: no adoption; the analogy is real but the current
detectors are already the right tool for the trend question.** (One concrete borrow worth a future probe: a
whiteness/χ² test on the *verdict-d_pose residual vs the EMA-shadow forecast* could formalize the "EMA-lag vs true
rise" disambiguation noted in the DAG — but that is a diagnostic, not a rate lever, and out of this task's scope.)

## Provenance
- Pose: `reports/kalman_innovation_20260710/pose_dxi.txt`; input R1 npz keys `pose_carrier.{xi_stored,dxi}` (600×6);
  backend `tac.lossless.range_coder.encode_static_symbols` + brotli-q11; q_levels=4096 (shipped).
- Lane: `reports/kalman_innovation_20260710/lane_coeff.txt`; regenerated coherent-slot matrix (600×66, K=6) from
  `gt_n600.npz`; backend brotli-q11 over zigzag temporal-delta (the LBND2 wire path).
- Reads-on (proactive recall, did NOT re-derive): `r1_dxi_shippability_byteclose_20260708.md` (ξ carrier + shipped
  d_pose), `wave_f_lane_tracking_rate_n600_RESULT.json` (LBND2 / coherent-slot / lossy-RTS rows),
  `lane_coeff_tracking_denoising_optimal_survey_20260702.md` (the CORRESPONDENCE-FIRST swap-theorem this confirms).
- Papers-checked (methodological, not a result): johndcook.com/blog/2016/05/24 "Kalman filters and bottom-up
  learning" — Kalman = recursive least-squares / bottom-up state estimator; innovation coding is optimal predictive
  coding conditional on model-signal match. openpilot rednose EKF (ego state) cited as the *smooth-continuous*
  regime where innovation coding wins (contrast to our jitter/discontinuous carriers).

**Verdict:** `tac.verdicts` FORMULATION-scope negative-lever row (`reports/kalman_innovation_20260710/verdict.json`).
No canonical equation minted — the finding is a lever-**foreclosure** verdict, not a predictive law with a callable
(minting a hollow-callable equation would be a fake-implementation smell). **Pointer 0.19110 UNMOVED.**
