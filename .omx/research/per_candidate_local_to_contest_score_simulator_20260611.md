# Per-candidate local→contest score simulator + the axis-choice exploit

**Date:** 2026-06-11
**Author:** per-candidate-score-simulator subagent (`per-candidate-score-simulator-20260611`)
**Operator ask (2026-06-11):** *"if we understand WHY cuda and linux-x86_64 score differently than our
local cpu, and if the mechanisms are deterministic for certain candidates, can't we solve for / simulate /
calculate the adjusted auth-eval score"* + the reframe: *"it depends by substrate and archive and full
stack and synergies and antagonisms — some PRs scored BETTER on gpu."* Determine whether we can build a
DETERMINISTIC, PER-CANDIDATE simulator that computes the contest auth-eval score (Linux-x86_64-CPU
leaderboard AND CUDA-T4) from our LOCAL score + full-stack features, WITHOUT spending the paid contest eval;
plus the axis-choice exploit (submit on the axis our substrate favors).

**Evidence grade.** Every number below is harvested from EXISTING measured anchors (contest CI bot
comments + our paired Modal CPU / Vast-T4 exact evals). macOS (torch/MLX) is `[macOS-CPU advisory]` /
`[macOS-MLX research-signal]`, NON-PROMOTABLE. Linux x86_64 = `[contest-CPU]` (leaderboard authority);
NVIDIA T4 = `[contest-CUDA]`. No MPS anywhere. The simulator's OUTPUT is a PREDICTION (research-signal)
until validated against a real paired eval; the contest exact eval remains the only arbiter.
**Did the exact frontier pointer move?** No. This is a simulator DESIGN + a harvested-dataset analysis +
an exploit thesis + an estimate-first validation plan — not an exact-eval row.

**Scope discipline.** Did NOT touch the running daemons (capstone, atlas) or the MLX-scorer files
(`mlx_scorer_adapters.py` / `capstone_trainer.py` — other live subagents own those). No paid dispatch
fired (estimate-first; see §5). Sister of the drift-ladder memo
`local_to_contest_scorer_drift_ladder_and_correction_20260611.md` — THIS memo REFRAMES that memo's RUNG C
(+0.033) from a class constant to a **candidate-dependent, operating-point-driven function** per the
operator's correction.

---

## 0. TL;DR — the operator's claim is CONFIRMED, and the mechanism is simulable

1. **The CUDA−CPU gap is candidate-dependent in SIGN, not a constant.** Harvested paired data shows it
   ranges from **+0.0335 (CPU-favoring)** to **−0.0215 (CUDA-favoring)** across substrates. The operator is
   right: SOME archives score better on CUDA.
2. **The sign flip is driven by the PoseNet OPERATING POINT, not the substrate family.** In every paired
   anchor, one device sits at `d_pose ≈ 3.2e-5` (low) and the other at `d_pose ≈ 1.65e-4` (~5× higher).
   **Whichever device lands at the HIGH pose value loses ~0.0227 score on the pose term.** SegNet is
   *always* ~1.17× worse on CUDA (a small, stable +0.010 seg-term penalty). The net gap = pose sign
   (±0.0227) + seg (+0.010 toward CPU).
3. **Mechanism is two deterministic pieces + the candidate's pose margin geometry:**
   (a) the **GT-decode path** (CUDA=DALI/NVDEC, CPU=PyAV/libav) — deterministic, locally replicable;
   (b) the **inflate-device render** (the candidate's `inflate.py` runs on the eval device, so CUDA-render
   ≠ CPU-render RGB) — deterministic per device; (c) the **scorer kernel numerics** (fp reduction-order) —
   deterministic given inputs, but the thing that decides which side of the pose operating point a
   candidate falls on. The +0.033 HNeRV constant held to σ=4e-4 over 5 independent archives ⇒ the
   per-candidate transform is a **stable function**, not noise.
4. **The simulator is FEASIBLE for the CPU axis (near-exact) and STATISTICALLY feasible for the CUDA axis
   (sign + magnitude prediction with a residual band), conditioned on the candidate's pose margin
   distribution.** The hard part is not arithmetic — it is reproducing DALI/NVDEC locally (we can't on
   macOS), so the CUDA pose operating point must be *modeled*, not computed exactly, until one paired
   anchor calibrates the class.
5. **The EXPLOIT is already partly in play and should be made deliberate.** Our CPU frontier (0.19110) is an
   HNeRV substrate (CPU-favoring); our CUDA frontier (0.20533) is a PR106 latent-sidecar (CUDA-favoring).
   For an **HNeRV-class capstone, SUBMIT FOR THE CPU LEADERBOARD** (it is the favorable axis, and the
   contest leaderboard ranks on CPU). We can also *engineer* the carrier toward whichever axis we submit.

---

## 1. The harvested per-candidate drift dataset (the operator's claim, tested)

**Sources (all contest-authority paired anchors, same archive bytes on both axes):**
`reports/public_pr100_108_cpu_cuda_drift_analysis_20260508.json` (5 HNeRV CI-bot pairs);
`.omx/research/device_axis_paired_anchor_matrix_20260511.md` (our paired Modal-CPU / Vast-T4 exact evals);
`.omx/research/cpu_cuda_drift_exact_pr103_pr106_20260511_codex.md` (PR103-on-PR106, exact);
`.omx/research/pr101_fec6_paired_cpu_cuda_axis_xray_20260515_codex.md` (our PR101 fec6 capstone-class).

| Candidate (substrate) | Bytes | CPU score | CUDA score | Δ (CUDA−CPU) | CPU d_pose | CUDA d_pose | pose ratio CUDA/CPU | seg ratio CUDA/CPU | **Favorable axis** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PR100 hnerv_lc_v2 | 178,981 | 0.19539 | 0.22827 | **+0.0329** | 3.44e-5 | 1.72e-4 | 5.00× | 1.17× | **CPU** |
| PR101 hnerv_ft_microcodec | 178,258 | 0.19285 | 0.22635 | **+0.0335** | 3.29e-5 | 1.71e-4 | 5.20× | 1.18× | **CPU** |
| PR102 hnerv_lc_v2_scale095 | 178,981 | 0.19538 | 0.22839 | **+0.0330** | 3.46e-5 | 1.73e-4 | 5.01× | 1.17× | **CPU** |
| PR103 hnerv_lc_ac | 178,223 | 0.19488 | 0.22776 | **+0.0329** | 3.44e-5 | 1.72e-4 | 5.00× | 1.17× | **CPU** |
| PR105 kitchen_sink | 177,857 | 0.19797 | 0.23044 | **+0.0325** | 3.47e-5 | 1.73e-4 | 4.97× | 1.16× | **CPU** |
| **A1** (our PR101-derived, score-gradient) | 178,262 | 0.19285 | 0.22635 | **+0.0335** | 3.30e-5 | 1.71e-4 | 5.18× | 1.18× | **CPU** |
| **PR101 fec6 K16** (our capstone-class) | 178,517 | 0.19205 | 0.22621 | **+0.0342** | ~3.4e-5 | ~1.7e-4 | ~5.0× | ~1.18× | **CPU** |
| **PR103-on-PR106 AC repack** | 185,578 | 0.22966 | 0.20898 | **−0.0207** | **1.64e-4** | **3.36e-5** | **0.205×** | 1.02× | **CUDA** |
| **PR106 latent sidecar r1** | 186,808 | 0.22868 | 0.20739 | **−0.0213** | **1.64e-4** | **3.30e-5** | **0.20×** | 1.02× | **CUDA** |
| **PR106 latent sidecar r2** | 186,822 | 0.22809 | 0.20665 | **−0.0214** | **1.64e-4** | **3.20e-5** | **0.197×** | 1.017× | **CUDA** |

**Headline (operator's claim CONFIRMED):** the gap sign is candidate-dependent. The HNeRV medal cluster
(PR100–105, A1, our fec6) is uniformly **CPU-favoring (+0.033)**; the **PR106 latent-sidecar family is
uniformly CUDA-favoring (−0.021)**. The 5.1× pose ratio has the SAME magnitude in both families but the
**SIGN flips** — exactly the synergy/antagonism the operator described.

### 1.1 What feature predicts the sign? — the PoseNet operating point, not the family name
In EVERY row, one device sits at `d_pose ≈ 3.2–3.5e-5` (the "low" operating point) and the other at
`d_pose ≈ 1.6–1.7e-4` (the "high" operating point, ~5× larger):
- HNeRV cluster: **CPU is the low-pose axis, CUDA is the high-pose axis** ⇒ CUDA pays the pose penalty ⇒
  CPU-favoring.
- PR106 sidecar: **CUDA is the low-pose axis, CPU is the high-pose axis** ⇒ CPU pays the pose penalty ⇒
  CUDA-favoring.

Numerically (verified): `sqrt(10·1.65e-4) − sqrt(10·3.2e-5) = 0.0405 − 0.0179 = +0.0227` is the pose-term
swing. The seg term is *always* +0.010 toward CPU (CUDA argmax flips ~1.17× more pixels — a small, stable,
device-fixed bias). So:

```
Δ_score(CUDA − CPU) ≈ +0.010 (seg, always CUDA-worse)
                      ± 0.0227 (pose: + if CUDA is the high-pose axis, − if CPU is)
```

This is a TWO-STATE model: the candidate × device combination lands the pose at either the low or high
operating point, and that binary choice (times the substrate's specific pose margins) sets the sign. The
PR106 latent-sidecar's per-pair latent *delta* perturbs the rendered frame so that the DALI/NVDEC-decoded
GT + CUDA-render pairing produces the *low* pose, whereas the HNeRV score-gradient-trained latent produces
the low pose only under the PyAV+CPU pairing. **The "synergy" is between the candidate's residual structure
and the eval device's GT-decode path.**

---

## 2. The mechanism model — is it deterministic + simulable?

`evaluate.py --device {cpu,cuda}` switches THREE things at once; decompose them:

| Mechanism | What it is | Deterministic? | Locally simulable? | Verdict |
|---|---|---|---|---|
| **(a) GT-decode path** | CUDA→DALI/NVDEC; CPU→PyAV/libav. The ground-truth video is decoded differently per device (CLAUDE.md: PyAV rgb24 manufactures ~100× phantom pose; the canonical GT is `frame_utils.yuv420_to_rgb`). | **YES** — both are deterministic libraries given fixed input. | **PyAV: YES locally. DALI/NVDEC: NO on macOS** (no CUDA runtime). | **Deterministic; CPU-side simulable now, CUDA-side needs one calibration anchor.** |
| **(b) inflate-device render** | The candidate's `inflate.py` runs on the eval device → CUDA-rendered RGB ≠ CPU-rendered RGB (raw-output aggregate hashes DIFFER across axes — proven on PR101 fec6: CPU `10c68e42…` vs CUDA `6fe2b194…`). | **YES** per device (fp reduction order is deterministic for a fixed kernel + input). | **CPU render: YES. CUDA render: NO on macOS** (and ExecuTorch-MLX-GPU is FP32-exact for PoseNet but is a different kernel than T4). | **Deterministic; the CUDA render is the part we cannot reproduce bit-exactly without a T4.** |
| **(c) scorer kernel numerics** | CPU vs CUDA matmul/conv reduction order in EfficientNet-B2 (SegNet) + FastViT-T12 (PoseNet). | **YES** given fixed inputs (CUDA is deterministic for fixed kernels; no atomics in these forwards). | CPU: YES. CUDA: NO on macOS. | **Deterministic; the 5× pose ratio is the `(1+ε)^L` accumulation through ~50 conv/attn ops — a stable per-architecture amplification.** |
| **(d) macOS↔Linux x86_64 CPU** (RUNG B) | fp rounding at SegNet decision boundaries between macOS-arm and Linux-x86_64. | **YES** (deterministic per platform). | **YES** — it is a SegNet-only +1.05e-5 bias, entirely boundary-pixel argmax flips; PoseNet + rate are bit-identical local↔contest. | **Deterministic + already-calibrated** (5 anchors, σ=8.3e-7). |

**Determinism conclusion.** None of these is stochastic noise. The CUDA eval is *repeatable* (the +0.033
held to σ=4e-4 across 5 independently-engineered HNeRV archives — that tight spread is itself proof the
per-candidate transform is a stable function of the archive, not run-to-run jitter). **So the question is
not "is it deterministic" (it is) but "can we reproduce the CUDA-only pieces (a)+(b)+(c) without a T4"
(we cannot bit-exactly on macOS) — therefore the CUDA axis must be MODELED from a calibration anchor, while
the CPU axis can be computed near-exactly.**

---

## 3. The simulator DESIGN

Two distinct prediction targets; never mix them (per the apples-to-apples discipline).

### 3.1 CPU-leaderboard simulator (near-exact; the supported path)
The Linux-x86_64-CPU contest score from a LOCAL macOS-CPU render is **already a near-solved problem** and
the existing `tac.optimization.local_cpu_contest_drift` module implements it:
```
S_contest_CPU  ≈  S_local_macOS_CPU  −  bias_B          (bias_B = +1.05e-5, SegNet-only)
conservative   =  S_local_macOS_CPU  −  bias_B + guard  (guard = 3e-6)
```
- Mechanism (a) PyAV decode + (b) CPU render + (c) CPU scorer are ALL locally reproducible. The only
  residual is mechanism (d) macOS-arm↔Linux-x86_64 SegNet boundary rounding = +1.05e-5.
- **Expected accuracy: ~1e-5 score** (the calibrated bias σ is 8.3e-7; the 5-anchor band is [+1.0e-5,
  +1.2e-5]). This is essentially exact relative to the T_1→T_3 distance (0.04).
- **Trust region: HNeRV-medal-band, same-archive.** Out-of-class older rows show +1.36e-4 / +2.88e-4 →
  must recalibrate per substrate class. (PR110/HNeRV capstone is in-class.)

### 3.2 CUDA-axis simulator (statistical; sign + magnitude with a residual band)
We CANNOT compute the CUDA score exactly without a T4 (DALI/NVDEC + CUDA render + CUDA scorer are not
reproducible on macOS). But we CAN predict it with a **two-state operating-point model** keyed on the
candidate's pose-margin geometry, calibrated by ONE paired anchor per substrate class:

```
S_contest_CUDA  ≈  S_contest_CPU
                   + 0.010                          (seg: CUDA always ~1.17× worse — stable class constant)
                   + sign · 0.0227                  (pose-term swing)
   where sign = +1 if the candidate renders the LOW-pose operating point on CPU (CUDA pays the pose)   [HNeRV class]
               −1 if the candidate renders the LOW-pose operating point on CUDA (CPU pays the pose)     [PR106 sidecar class]
```
- The `0.0227` magnitude and the `0.010` seg constant are HNeRV-medal-band-bounded; recalibrate for a new
  class with ONE paired eval.
- **The sign predictor** is the open modeling problem. Best available feature: the substrate's residual
  structure (score-gradient-trained latent → CPU-low; per-pair latent-delta sidecar → CUDA-low). A
  principled local proxy: render the candidate locally, decode the GT via BOTH PyAV and the
  `frame_utils.yuv420_to_rgb` canonical path, and measure which pairing puts `d_pose` near the 3.2e-5 floor
  vs the 1.65e-4 high state. (DALI/NVDEC ≈ the canonical YUV path more than PyAV rgb24; this is the lever
  the CLAUDE.md "PyAV manufactures ~100× phantom pose" note points at.)
- **Expected accuracy: SIGN reliably (the two classes are cleanly separated), MAGNITUDE to ±0.002** after a
  one-anchor calibration of the class. WITHOUT a class anchor: sign + a wide ±0.0227 band (still useful for
  axis-choice triage, never for a claim).

### 3.3 The simulator as a typed surface (wire-in target)
Extend `tac.optimization.local_cpu_contest_drift` (which already has `DriftCalibration` +
`conservative_projected_contest_score`) with a sister `DeviceAxisProjection` carrying:
`{class_id, seg_cuda_minus_cpu_const, pose_term_swing, pose_low_state, pose_high_state, sign_predictor,
calibration_anchor_sha, residual_band}`. It emits a `device_axis_projection.v1` row with FALSE-AUTHORITY
fields (`score_claim=false`, `promotion_eligible=false`) — a spend-triage prior + axis-choice signal, never
a conversion. This feeds the cathedral autopilot's axis-choice decision (§4) and reseeds on each paired
anchor (continual-learning hook).

---

## 4. The EXPLOIT — pick (and engineer for) the favorable axis

### 4.1 Which axis does our capstone favor?
Our HNeRV-class capstone (PR110 payload-entropy-recode lineage, the CPU frontier substrate) is squarely in
the **CPU-favoring** cluster (A1, PR101 fec6, PR100–105 all +0.033 CPU-favoring; our CPU frontier 0.19110 is
this class). So:
- **For an HNeRV-class capstone, the CPU leaderboard is the favorable axis** — and it is also the axis the
  **contest leaderboard ranks on** (CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": *"The contest
  leaderboard ranks by the CPU eval, not the CUDA eval"*; PR102's third prize was awarded on its 0.19538
  CPU score). **Double-favorable: the axis we win on is the axis that decides the medal.**
- Our PR106 format0d CUDA frontier (0.20533) is the CUDA-favoring class — it would be a *worse* leaderboard
  (CPU) entry (~0.227 CPU) than our HNeRV CPU frontier. **Confirms the exploit is already implicit in why we
  hold two different substrates on two different axes.**

### 4.2 Is the medal CPU, CUDA, or better-of?
Per the harvested PR102 third-prize evidence + CLAUDE.md: **the leaderboard ranks on the CPU eval.** The bot
posts both, but the prize was awarded against the CPU score. ⇒ **The exploit is unambiguous: for the medal,
optimize and submit the CPU-favorable substrate, and report the CPU score as the headline.** (Keep the
paired CUDA eval for the dual-axis submission packet per the non-negotiable, but the medal axis is CPU.)

### 4.3 Can we engineer the carrier toward the favorable axis?
Yes — and the PR106 sidecar is the existence proof. The per-pair latent-delta sidecar moved the substrate
from CPU-low to CUDA-low (flipped the favorable axis). For an HNeRV capstone aiming at the CPU leaderboard,
the design rule is: **keep the carrier "decode-friendly" for the PyAV/CPU GT path** (the path that puts our
pose at the 3.2e-5 floor on CPU) — i.e. do NOT add a residual whose structure only nulls pose under the
DALI/NVDEC pairing. This is a *negative* design constraint (don't accidentally flip to CUDA-low) more than a
new lever, because our class already favors CPU. The positive lever — deliberately pushing pose to the low
floor on the CPU axis — is the standard pose-axis attack (RAFT/LAPose sidecars) the roadmap already ranks.

---

## 5. Validation plan (estimate-first)

**Goal:** validate the simulator on ONE of OUR current archives — confirm (i) the CPU-axis projection is
accurate to ~1e-5, and (ii) the CUDA-axis two-state model predicts the right sign + magnitude for the
PR110/HNeRV class.

### 5.1 Smallest paired eval — ESTIMATE
- **What:** paired exact eval on our CPU-frontier archive `b46897267ded…` (PR110, 177,169 B):
  `[contest-CPU]` Modal CPU Linux-x86_64 (~$0.12, 60–120 min) + `[contest-CUDA]` Vast/Lightning T4 (~$0.30,
  ~25 min). Reuse `experiments/modal_auth_eval_cpu.py` + `tools/plan_dual_device_auth_eval.py`.
- **Cost: ≈ $0.42 combined** (per `docs/findings/cuda_cpu_auth_eval_split_20260508.md` §11). The CPU leg
  alone (~$0.12) validates §3.1; both legs validate §3.2's sign + magnitude.
- **Within <$5 budget? YES. Trivially <~$1? The CPU leg alone (~$0.12) is; the paired (~$0.42) is >$0.30.**
- **Decision: I did NOT auto-run it.** (a) estimate-first directive; (b) the CPU-axis simulator is already
  calibrated to σ=8.3e-7 over 5 anchors and our CPU frontier is *already a measured `[contest-CPU]` row*
  (0.19110) — so §3.1 is effectively pre-validated; (c) the CUDA-axis leg is >$0.30 and needs a lane-claim
  (`tools/claim_lane_dispatch.py`) + HARVEST-OR-LOSE — a real dispatch the operator should greenlight, not
  fire unattended. **Recommendation: GREENLIT-ON-REQUEST.** The single highest-value cheap action is the
  **CPU leg on whatever NEW capstone candidate beats the frontier** (~$0.12, validates the simulator AND
  buys an exact-eval row simultaneously — the means and the end coincide). The CUDA leg (~$0.30) is a
  *nice-to-have* class-calibration of the §3.2 magnitude; PROPOSE, don't auto-run.
- Command sketch (verify argparse before running):
  ```bash
  # lane-claim FIRST (non-negotiable)
  .venv/bin/python tools/claim_lane_dispatch.py claim --lane-id lane_per_candidate_simulator_validate_20260611 \
      --instance modal_cpu --status active --notes "validate per-candidate CPU+CUDA simulator on PR110 frontier"
  # paired plan/execute (the canonical dual-axis tool):
  .venv/bin/python tools/plan_dual_device_auth_eval.py --archive <archive.zip> --device-pair cpu cuda
  # harvest within 24h (HARVEST-OR-LOSE); diff predicted vs measured; reseed device_axis_projection.v1.
  ```

### 5.2 What would PROMOTE the simulator from "designed" to "validated"
One paired anchor on the PR110/HNeRV capstone class (§5.1) closes the §3.2 magnitude calibration for our
*current* class and confirms the §1.1 sign predictor on a substrate we actually ship. Until then the CUDA
projection is a sign-reliable, ±0.0227-band research-signal prior — fine for axis-choice triage, never a
claim.

---

## 6. Solver / system wire-in (results become intelligence)

Per CLAUDE.md "Results must become system intelligence" — the 6 unified-Lagrangian hooks:
1. **Sensitivity-map** — ACTIVE (prior). The cross-hardware per-byte leverage drift (6.4% macOS-CPU vs
   11.1% contest-CUDA on the same archive) is the byte-level shadow of this same device-axis phenomenon; the
   simulator formalizes the *axis* on which sensitivity is read.
2. **Pareto constraint** — N/A (this sets an axis-projection prior + axis-choice, not a rate/seg/pose
   polytope constraint).
3. **Bit-allocator hook** — CONDITIONAL: the §4.3 "decode-friendly carrier" design rule is a *negative*
   constraint on residual structure (don't flip the favorable axis); it does not change per-tensor
   importance, so no allocator change lands here.
4. **Cathedral autopilot dispatch hook** — ACTIVE target: the `device_axis_projection.v1` row (§3.3) feeds
   the autopilot's **axis-choice** decision (which axis to spend the paired eval on / which axis to submit).
   No code landed (research+design only); the surface is the proposed extension of
   `tac.optimization.local_cpu_contest_drift`.
5. **Continual-learning posterior** — ACTIVE on dispatch: the §5.1 paired anchor reseeds both the RUNG-B CPU
   bias AND the §3.2 CUDA two-state class calibration (seg const + pose swing + sign predictor).
6. **Probe-disambiguator** — the §5.1 paired eval IS the disambiguator between "§3.2 magnitude is +0.033
   HNeRV-class (assumed)" vs "measured for the PR110 capstone class"; the GT-decode-vs-kernel mechanism
   split (§2 a/b/c) is the separate shared-tensor 2×2 program (out of scope here; does not change the
   simulator's usability as a class-bounded prior).

**`research_only=true`** for this memo: no code landed, no exact-eval row, no promotion. Integration blocker
for promoting the CUDA-axis simulator from "designed" to "validated": the ~$0.42 paired Modal-CPU + T4
dispatch in §5.1 (operator-greenlight-on-request).

---

## 7. NO-FAKE / authority notes
- All local (macOS torch + MLX) numbers are `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`,
  NON-PROMOTABLE. Only Linux-x86_64 = `[contest-CPU]`; only NVIDIA T4 = `[contest-CUDA]`. No MPS anywhere.
- The +0.010 seg const, the ±0.0227 pose swing, and the two-state sign model are HNeRV/PR106-class
  *projection priors / axis-choice triage signals*, NEVER score claims, conversions, promotions, ranks, or
  kills. The contest exact eval is the only arbiter.
- The PR106-CUDA-favoring rows directly FALSIFY the broad "CPU always wins" reading of the drift-ladder
  memo's RUNG C; this memo supersedes that constant with the candidate-dependent function per the operator's
  correction (APPEND-ONLY: the drift-ladder memo's RUNG-C +0.033 remains valid AS the HNeRV-class case).
- The simulator's CUDA output is a PREDICTION until validated by §5.1; honestly flagged as such, not
  presented as measured.

## 8. Reproduce / sources
- Paired dataset: `reports/public_pr100_108_cpu_cuda_drift_analysis_20260508.json`,
  `.omx/research/device_axis_paired_anchor_matrix_20260511.md`,
  `.omx/research/cpu_cuda_drift_exact_pr103_pr106_20260511_codex.md`,
  `.omx/research/pr101_fec6_paired_cpu_cuda_axis_xray_20260515_codex.md`,
  `.omx/research/cpu_cuda_xray_synthesis_20260511.md`.
- Existing simulator-adjacent code: `src/tac/optimization/local_cpu_contest_drift.py`
  (`DriftCalibration.conservative_projected_contest_score`),
  `tools/calibrate_local_cpu_contest_drift.py`, `tools/xray_paired_cpu_cuda_axis_delta.py`,
  `tools/plan_dual_device_auth_eval.py`, `experiments/modal_auth_eval_cpu.py`.
- Mechanism tools (CUDA cells pending operator gate): `tools/cpu_cuda_xray_loader_drift.py`,
  `tools/cpu_cuda_xray_segnet_layer_drift.py`, `tools/cpu_cuda_xray_posenet_layer_drift.py`.
- Sister memo: `.omx/research/local_to_contest_scorer_drift_ladder_and_correction_20260611.md` (the
  3-rung ladder this memo reframes RUNG C of).
- Frontier: `.omx/state/canonical_frontier_pointer.json` (CPU 0.19109982 PR110 `b4689726…`;
  CUDA 0.20533003 PR106-format0d `9cb989ce…`).
