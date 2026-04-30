# Council Design Review — Lane RAFT/radial pose

**Status:** Phase A council review for Level 0 → Level 1 graduation.
**Anchor:** Lane G v3 = 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL).
**Predicted band [prediction]:** Two scenarios:
1. **Inflate-time RAFT (zero pose bytes shipped):** -0.000 to -0.005 score IF disagreement vs
   contest pose ≤ 0.001 average; -0.05 to -0.20 score REGRESSION if disagreement is high.
   Net byte savings: ~50–100 KB (the entire pose stream eliminated).
2. **Compress-time RAFT prior (pose stream still shipped):** -0.001 to -0.003 score via
   improved baseline pose initialization; ~0 byte change.
**Cost estimate:** $0.30–$0.50 (RAFT inference at compress time + Lane G v3 anchor reuse).
**Dependencies:** existing `src/tac/raft_pose.py` (Lane FL — single-DOF radial dim 0); needs
extension to full 6-DoF radial-basis decomposition.

## 1. Existing scaffold audit

`src/tac/raft_pose.py` (143 LOC, 1 test in test_raft_pose.py):

- `compute_raft_flow(video_path, n_frames, device='cuda')` — RAFT-Large from torchvision
- `flow_to_pose_dim0(flow, fx, road_region)` — converts road-region horizontal flow → pose dim 0
- `calibrate_pose_dim0(raw_dim0, baseline_dim0)` — least-squares affine fit
- `build_pose_tensor_from_flow(flow_dim0, baseline_poses)` — assemble 6-DoF tensor (other
  dims zeroed)

**Limitations of current scaffold:**
- ONLY computes pose dim 0 (the longitudinal/forward translation along the focal axis)
- Other 5 DoF (lateral, vertical, pitch, yaw, roll) are zeroed → contest pose distortion
  collapses for the unzeroed dimensions
- No radial-basis decomposition; just averaged road-strip flow
- No inflate-time integration

**What Lane RAFT/radial extends:**
1. Full 6-DoF estimation from RAFT flow via radial-basis function decomposition
2. Per-frame agreement vs contest pose stream (tag fragility)
3. Two-mode operation: compress-time prior (low risk) OR inflate-time recompute (high reward)
4. Mallat-style multi-scale flow basis (wavelet-grade radial decomposition)

## 2. Math foundation

### 2.1 Optical flow → ego-motion (classical structure-from-motion)

For a calibrated camera with focal length `f`, principal point `(c_x, c_y)`, and motion
between two frames consisting of translation `t = (t_x, t_y, t_z)` and rotation `R`
(small-angle: `(ω_x, ω_y, ω_z)`), the flow at image point `(x, y)` is:

    u(x, y) = -f·t_x/Z + (x·t_z)/Z + ω_y·f - ω_z·y + ω_x·x·y/f - ω_y·x²/f
    v(x, y) = -f·t_y/Z + (y·t_z)/Z - ω_x·f + ω_z·x + ω_y·x·y/f - ω_x·y²/f

where `Z` = depth at that pixel (Longuet-Higgins 1981). The flow is the SUM of
**translation flow** (depth-dependent, radial from the FOE) and **rotation flow**
(depth-INDEPENDENT, polynomial in image coords).

Key insight: the rotation contribution is a depth-INDEPENDENT polynomial basis. The
translation contribution requires a depth estimate. For driving scenes:
- Road region has approximately known Z (road plane geometry)
- Sky region has Z → ∞ (pure rotation flow)

### 2.2 Radial basis decomposition (Mallat-style)

Decompose flow field `F(x, y) ∈ ℝ²` into a basis:

    F(x, y) = Σ_k α_k · B_k(x, y)

where `{B_k}` are **dimensional-physics basis functions**:

- B_0(x, y) = (1, 0)            — pure horizontal translation
- B_1(x, y) = (0, 1)            — pure vertical translation
- B_2(x, y) = (x, y)             — radial (FOE-radiating, t_z component)
- B_3(x, y) = (-y, x)            — pure roll (ω_z)
- B_4(x, y) = (-y², x·y)         — pitch (ω_x)
- B_5(x, y) = (x·y, -x²)         — yaw (ω_y)
- + higher-order radial wavelets for residuals (Mallat scattering transform analog)

Solving `α = (BᵀB)⁻¹ Bᵀ F` is a linear LSQ. The 6 coefficients map to the 6-DoF pose under
the calibration (but the contest pose is uncalibrated — we use empirical affine
calibration `pose_contest = A @ α + b` fit on a held-out segment).

### 2.3 Calibration to contest pose

Following Lane FL (existing scaffold) approach:
- Use first 200 frames as calibration set
- Fit `(A, b) ∈ ℝ^{6×6} × ℝ^6` via least-squares against contest pose
- Apply to remaining 400 frames
- Cross-validate: evaluate disagreement on held-out frames

### 2.4 The strict-scorer-rule question

CLAUDE.md non-negotiable: "NO loading PoseNet or SegNet at inflate time."

**Critical clarification:**
- RAFT is **NOT** a contest scorer (it's the torchvision Raft_Large optical flow network)
- BUT it loads a ~5–20 MB neural network at inflate time
- Per CLAUDE.md "non-compliant, requires compliance ruling": this requires explicit
  human approval before being labeled `[contest-compliant]`
- Until that ruling, Lane RAFT inflate-time mode must be tagged `[non-compliant, requires
  compliance ruling]` and disabled by default (env-gate `INFLATE_RAFT=0`)

**Two safe operating modes (for now):**
- **Mode A (compress-time prior, default):** RAFT runs at compress time only. Produces a
  6-DoF pose initialization that the existing pose-TTO loop refines. Pose stream still
  shipped (~50 KB). Net effect: faster TTO convergence, slightly better optimum, ~0 byte
  change.
- **Mode B (inflate-time recompute, env-gated):** RAFT runs at inflate time. Pose stream
  is NOT shipped (or only a small calibration delta is). Eliminates ~50 KB. Requires
  explicit compliance ruling before any contest submission.

## 3. Council deliberation

### Hotz (LEAD — raw engineering)

> "Wait. RAFT-Large is ~5MB on disk. That's BIGGER than the entire pose stream you're trying
> to eliminate. It's only a win at inflate time if RAFT is ALREADY available on the contest
> scorer (torchvision-pretrained, magnetized). If RAFT must ship in archive.zip, this lane
> is DEAD on arrival.
>
> Check the contest scorer environment. If torchvision raft_large is preinstalled (it
> usually is in modern PyTorch envs), Mode B is alive. Otherwise it's dead.
>
> ALSO: RAFT inference is 2.4s per frame on T4 × 600 frames = 24 min. That's most of the
> 30-min inflate budget. Not workable if SegNet inflate must also run."
>
> **Verdict:** YELLOW. Mode A is alive unconditionally. Mode B requires (a) torchvision
> RAFT preinstalled AND (b) <10min RAFT inference budget. Both are checkable.

### Yousfi (channel — challenge creator + steganalysis lead)

> "The pose stream in our archive is ~50 KB at FP16; ~25 KB if delta-coded; ~10 KB if
> Lane PD-V2 lands. Eliminating it eliminates a ROUNDED-DOWN 0.04 contribution to the
> rate term (60KB → 0.04 score per CLAUDE.md arithmetic). But the pose distortion gets
> worse if RAFT-radial doesn't match the contest pose ground truth.
>
> Concretely: contest PoseNet distortion at Lane G v3 baseline is ~0.0030. If RAFT-radial
> agrees within 0.0001, score moves +0.0007 per 0.0001 increase × 7 = +0.005 score = bad
> trade. If it agrees within 0.001, score moves +0.07 = catastrophic.
>
> The viability gate: average per-frame RAFT-radial vs contest pose disagreement < 1e-4.
> Below that, Mode B is a small win (-0.04 + 0.005 = -0.035). Above that, dead."
>
> **Verdict:** GREEN on Mode A. Mode B contingent on empirical disagreement measurement.

### Fridrich (channeled — Yousfi's PhD advisor, steganalysis grandmaster)

> "Steganalysis perspective: the contest pose ground truth IS NOT TRUTH. It is what
> upstream/contest_scorer/PoseNet outputs on the original raw frames. RAFT-radial computes
> a DIFFERENT estimate of ego-motion using a DIFFERENT network. The two will disagree on
> systematic biases (camera intrinsics assumptions, depth priors, road-plane assumptions).
>
> The right empirical experiment: run RAFT-radial on Lane G v3 frames, compare against the
> shipped pose stream, plot the per-frame disagreement histogram. If it's bimodal (some
> frames agree perfectly, others diverge), DELTA-CODE the disagreement: ship RAFT-radial
> + small per-frame correction stream. That's a hybrid Mode A/B that captures most savings
> without catastrophic risk."
>
> **Verdict:** GREEN with hybrid recommendation (Mode A/B blend).

### Mallat (channeled — wavelet/scattering grandmaster)

> "The radial-basis decomposition I proposed above is the FIRST ORDER. The full Mallat
> approach is a SCATTERING TRANSFORM on the flow field — multi-scale wavelet decomposition
> of the residual after the 6-DoF projection. The residual encodes scene-dependent depth
> + dynamic objects + camera shake. 95% of energy is in the 6-DoF projection; the residual
> is sparse and compresses to <1KB per frame in wavelet basis.
>
> So: Lane RAFT/radial Phase B produces (6-DoF coeffs, wavelet residual). The wavelet
> residual is itself optional — Mode A1 ships only the 6-DoF + zero residual; Mode A2
> ships the wavelet residual too at small marginal cost.
>
> This is the natural Lane 11 (Wavelet) and Lane 18 (RAFT) intersection."
>
> **Verdict:** GREEN. Architecture plays well with Lane 11.

### Karpathy (advisory — let compute speak)

> "Run the empirical disagreement measurement FIRST. If RAFT-radial and contest pose
> agree at 1e-4 average, this lane is a slam-dunk worth $0.50. If they disagree at 1e-3,
> kill or hybrid. Stop debating; start measuring."
>
> **Verdict:** GREEN on Phase B empirical step.

### Quantizr (adversarial — leader at 0.33)

> "Quantizr ships a fully shipped pose stream (~10–15 KB at FP4 + Brotli). I'm not eliminating
> mine. If your RAFT-radial saves 50 KB net (after accounting for the calibration delta you
> still need to ship), you have -0.04 score over me. That's a real lane.
>
> But Hotz is right: if RAFT must ship in archive.zip, you're dead. Make sure torchvision
> raft_large is in the contest scorer's torch environment. CHECK THIS FIRST."
>
> **Verdict:** YELLOW. Mode B alive only if scorer env supports it.

### Selfcomp (block-FP author)

> "Pose stream in my archive is also small (~10 KB). I haven't optimized it because the
> renderer.bin and masks.mkv dominate. RAFT-radial is interesting because it eliminates a
> dimension my codec doesn't address."
>
> **Verdict:** GREEN.

### Contrarian

> "Strong objection to inflate-time RAFT under the strict-scorer-rule. CLAUDE.md is
> NON-NEGOTIABLE: 'NEVER claim a contest-compliant score that depends on inflate-time
> scorer access.' Even if RAFT is technically not the contest scorer, it IS a learned
> neural network loaded at inflate time. The HUMAN must approve before this is shipped.
>
> Mode A (compress-time prior) is unconditionally fine. Mode B is gated on EXPLICIT human
> approval per CLAUDE.md."
>
> **Verdict:** GREEN on Mode A; **VETO on Mode B without explicit human approval**.

## 4. Decision

**Adopt:** Two-mode implementation. Mode A (compress-time prior) is the default, unconditionally
shippable. Mode B (inflate-time recompute) is implemented as code but env-gated AND
flagged `[non-compliant, requires compliance ruling]` until explicit human approval.

**Architecture:**
- `RaftRadialDecomposition(n_basis_functions=6)` — fits 6-DoF physics basis to flow
- `compute_radial_basis_from_flow(flow_field)` → returns (α_6dof, residual_field)
- `calibrate_to_contest_pose(α, contest_pose, calibration_frames)` → returns (A, b) affine
- `apply_calibration(α, A, b)` → 6-DoF pose tensor
- `evaluate_disagreement(pose_estimated, pose_contest)` → per-frame MSE histogram
- `RaftRadialPoseConfig` dataclass — operating mode (A/B), calibration window, basis count

**Wire format (Mode B only, env-gated):**
- No pose stream in archive.zip
- Optional small calibration delta (`pose_calibration.bin`, ~200 B) holds the (A, b) affine
- Inflate calls `compute_radial_basis_from_flow` + `apply_calibration`

**Strict-scorer-rule compliance:**
- Mode A: full compliance (compress-time only).
- Mode B: env-gate `INFLATE_RAFT=0` default; runtime banner
  `print('[strict-scorer-rule] Lane RAFT loads torchvision RAFT-Large at inflate time')`
  when enabled. Human approval REQUIRED before any contest submission with this active.

**Kill criteria:**
- If empirical RAFT-radial vs contest pose disagreement > 1e-3 average → Mode B dead;
  Mode A still alive as a TTO initialization speedup only.
- If torchvision RAFT not preinstalled in contest scorer env → Mode B dead.
- If RAFT inflate inference > 10 min on T4 → Mode B dead (squeezes 30-min budget).

## 5. Phase ordering (operational)

1. **Phase A** (this doc) — DONE
2. **Phase B (Level 1)** — `src/tac/raft_radial_pose.py` skeleton + 7-10 synthetic tests
3. **Phase C (Level 2 prep)** — empirical RAFT-vs-contest-pose disagreement on Lane G v3
   anchor; tag `[empirical:reports/raft_radial_disagreement.json]`
4. **Phase D (Level 2)** — wire Mode A into `compress.sh` as TTO initializer
5. **Phase E (Mode B prep)** — implement Mode B BEHIND env-gate; runtime banner; HUMAN
   APPROVAL gate
6. **Phase F (Level 3 path)** — STRICT preflight Check XX (no INFLATE_RAFT=1 without
   approval marker); 3-clean-pass adversarial review

## 6. Cross-references

- CLAUDE.md "Strict scorer rule" (the binding gate on Mode B)
- CLAUDE.md "Auth eval EVERYWHERE"
- `feedback_production_hardened_standard_definition_20260430.md`
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 18 RAFT/radial pose"
- `src/tac/raft_pose.py` (existing single-DOF Lane FL scaffold)
- `src/tac/depth_motion.py` (sibling lane — depth-flow joint motion)
- Teed & Deng 2020 — RAFT (arXiv 2003.12039)
- Mallat 2009 — *A Wavelet Tour of Signal Processing* Ch. 6 (radial wavelets)
- Longuet-Higgins 1981 — "A computer algorithm for reconstructing a scene from two
  projections" (the flow-to-ego-motion equations)
