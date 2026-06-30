# RL lab for the level-set witness — scoping + $0 env-step timing smoke + go/no-go

**Tag** `[$0 CPU scoping + GPU-free timing smoke / advisory / DESIGN]` · 2026-06-30T16:01Z ·
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false.
This is a MEANS (a feasibility scope + a timing measurement + a decisive-smoke plan), NOT a score.
**Pointer UNMOVED: contest-CPU 0.19109982 / contest-CUDA 0.20533003.**

**Source:** operator 2026-06-30 — *"our new level set system together with dynamical and cooperative
receiver … make for a possibly good target and reward for reinforcement learning, what about pufferlib?
… set up a RL lab."* Spec = `project_rl_lab_levelset_exact_dseg_reward_direction_20260630`. Coordinator
refinement 2026-06-30: torch off-the-shelf (PufferLib/CleanRL/torch-on-MPS) is acceptable for the
policy/gradient side — recommend on merits (velocity-to-annulus-unlock NOW vs world-class-MLX trajectory).
**HARD INVARIANT (non-negotiable):** the d_seg AND d_pose REWARD/VERDICT is the numpy-fp32 CPU/CUDA
AUTHORITY path (real through-R argmax + frozen CPU scorer); MPS (incl. torch-on-MPS) is a
gradient/training/timing device ONLY — NEVER a reward, NEVER a score.

**Evidence-grade legend:** `MEASURED-BY-US` (timed today) · `DERIVED` (math from our anchors) ·
`CITED` (our own prior anchors / external lit) · `ASSERTED` (plausible, flagged).

---

## TL;DR + GO/NO-GO (lead)

- **GO — RUN the decisive annulus-unlock micro-loop** ($0, GPU-free, ~3–35 min CPU-authority). It is the
  make-or-break and it is cheap. It runs CPU-only (does NOT contend the GPU the live Muon arm owns).
- **CONDITIONAL — DEFER building the full RL lab** until the micro-loop confirms RL moves a *stalled*
  annulus residual the smooth-surrogate gradient cannot. Do NOT build the lab first (means≠ends).
- **NO-GO — do NOT adopt PufferLib.** Its throughput advantage (~1e6 env-steps/s vectorization) is **moot
  for our env**: our env is **reward-bound at ~5 exact-rewards/s** on the CPU-authority scorer, not
  policy-bound. Draw its *ideas* (vectorized envs + batched reward + obs/action emulation); adopt a minimal
  single-file PPO (MLX-native preferred for the world-class-MLX trajectory; CleanRL-torch acceptable for
  fastest stand-up — the choice barely matters because the policy is tiny and the reward dominates).

**The binding number (corrected for the invariant):** the EXACT reward = the frozen **CPU-authority**
SegNet argmax through R. The 104× Apple-GPU scorer speedup is **forbidden for the reward** (MPS≠authority),
so the local reward floor is **~180–190 ms/exact-reward at batch≈8 → ~5 exact-rewards/s** (`MEASURED-BY-US`).
That makes the **micro-loop (1e3–1e4 exact rewards) tractable (3–35 min)** but a **classic 1e5–1e6-exact-step
PPO NOT tractable on local CPU-authority alone** (5.5–55 h) — the lab must therefore decouple the cheap dense
shaping from the expensive exact reward (see §3) or dispatch the reward to CUDA.

---

## 1. The thesis (why RL is a real fit, not RL-for-its-own-sake)

`DERIVED` from the convergence deep-math (`pr95_dseg_30k_convergence_deepmath_20260630`,
`project_dseg_islands_8dim_manifold`): d_seg is a **0-1 argmax loss whose gradient is a Dirac on a
measure-zero codim-1 contour.** We train a smooth surrogate (CE→tau→l7) whose gradient mass on the boundary
**vanishes as τ anneals** → the long flat tail (PR95's ~14k boundary epochs; our l7/Muon flat tails). The
live Muon arm sits in exactly this regime: **BEST realized d_seg = 0.004250 @ep850** (`levelset_best.json`,
`MEASURED-BY-US`), descending slowly. The ~8-dim **nonlinear lane-orbit manifold** is the binding residual.

**RL optimizes a non-differentiable reward DIRECTLY.** A policy acting on φ, rewarded by the EXACT through-R
SegNet d_seg, can target the annulus flips where the surrogate gradient ≈ 0 by construction. This is the one
mechanism that is *categorically* different from more gradient epochs: it does not need a usable gradient on
the boundary. **The dense shaping reward already exists for free**: the φ-margin (top1−top2 of the SDF
field) IS the cooperative-receiver confidence — computable from the render forward WITHOUT the scorer.

**Two distinct machines (don't conflate):** (a) **RL** = policy net optimizing φ vs the exact reward (this
memo). (b) **"local models ripping on it"** = an autonomous *research-agent* loop (LLMs proposing/testing
levers) = scaling the θ* campaign — NOT classic RL, out of scope here.

---

## 2. The MEASURED env-step timing smoke (M5 Max, CPU-authority, GPU-free)

Smoke: `scratchpad/rl_env_step_timing_smoke.py`. Decomposes ONE env step into its three real operations on
the authority path; render at 384×512 (P=196,608), witness hidden=96×4, mod=32, n_classes=5, in_feat≈160.
GT = `gt_n1.npz` (8 MB, one real pair). torch threads capped (4–6). **NO-FAKE sanity: d_seg(real GT frame)
= 0.000000 → the scorer path is the REAL frozen SegNet.** Render/R timed on the real codepaths with a
representative random param dict (timing is content-independent for these dense ops — a wall-clock
measurement, not a score).

| Operation | cost (`MEASURED-BY-US`) | notes |
|---|---|---|
| (1) render φ — `levelset_rgb_forward_numpy`, numpy fp32, P=196,608 | **~348 ms** | numpy deploy-faithful path; the TRAINER renders on MLX-GPU (the fast real path). Dense φ-margin shaping needs ONLY this (no scorer). |
| (2) R operator — `_torch_R_to_camera_uint8` (bicubic↑→874×1164 + uint8) | **~6 ms** | cheap |
| (3) **scorer — `cpu_verdict_d_seg_batch` (frozen SegNet argmax = the EXACT reward, CPU-AUTHORITY)** | **batch=1 ~444 ms; batch=8 ~180–190 ms/frame (≈5 exact-rewards/s); batch=16/32 WORSE (~330–360 ms/frame, memory-bound)** | **the binding reward cost.** Amortization plateaus at batch≈8; do not over-batch. |

**Full single-env step (CPU-authority):** dense-shaping (render-only) ~348 ms → ~2.9/s; full exact-reward
(render+R+scorer, b=1) ~798 ms → ~1.3/s; full with scorer amortized at batch 8 ~533 ms → ~1.9/s.

**Throughput verdict (the corrected, invariant-faithful numbers):**

| budget | CPU-authority exact rewards @ ~5/s (batch 8) | note |
|---|---|---|
| 1e3 exact rewards | **~3.3 min** | micro-loop scale — TRACTABLE |
| 1e4 exact rewards | **~33 min** | micro-loop / small lab — TRACTABLE |
| 1e5 exact rewards | **~5.5 h** | heavy; needs reward-decoupling or CUDA |
| 1e6 exact rewards | **~55 h** | NOT tractable locally; classic PPO step count |

**Invariant correction (important, NO-FAKE).** The smoke's "Apple-GPU projection" (~22 ms/step → 45 steps/s)
used the **104× MPS scorer factor — which is FORBIDDEN for the reward.** That projection is valid ONLY for a
*gradient/timing* device, NOT the reward. The reward floor is the CPU-authority ~5/s above (or CUDA if
dispatched: T4 SegNet batched ≈10–30 ms/frame `ASSERTED` → ~30–100/s, a paid scaling path, not local). The
render and dense-shaping CAN ride MLX-GPU (fast); only the **reward** is pinned to CPU/CUDA authority.

**Why the env is reward-bound (the key structural fact):** render φ on MLX-GPU is cheap and batches well;
the EXACT reward (CPU-authority scorer) does NOT amortize past batch≈8 and is the ~180 ms/frame wall. So the
PufferLib bet (make the env step cheap by vectorizing thousands of envs) **solves a bottleneck we don't
have** — vectorizing renders is easy; the scorer reward is the wall, and it is CPU-authority by mandate.

---

## 3. RL framing (state / action / reward / episode / algorithm) — tied to the level-set / cooperative-receiver / dynamical frame

- **Observation / state.** The per-pair conditioning **code** (the ~32-dim `mod_dim` FiLM code, or the
  reduced ~8-dim lane-orbit coordinates) + a **flip-risk map** = the φ-margin field (top1−top2) restricted to
  the annulus band (the codim-1 boundary ±1–2 px, 96.8% of flip mass; `CITED` veh-G2) + the GT `lstar`/
  `margin` context (which pixels are flip-prone). The annulus mask makes the observation small + on-task.
- **Action space (the manipulable level-set field).** Continuous deltas that **move the zero-level-set**:
  (a) **per-pair code δ** (~32-dim, the FiLM modulation that shifts the partition — the lowest-friction
  action), or (b) **lane-orbit control δ** along the ~8-dim nonlinear separatrix manifold (the *sufficient
  statistic* itself — the principled, minimal action), or (c) `out_sdf`-head local nudges. Recommend (a) for
  the micro-loop (smallest, already in the witness), (b) as the principled upgrade. = moving along the code
  manifold, NOT per-pixel logits (per veh-G4: per-pair FiLM re-weights fixed channel patterns; the policy
  *learns the descent direction* the gradient lacks).
- **Reward (the cooperative-receiver / level-set frame).**
  - **Sparse exact anchor:** `−d_seg` from the EXACT through-R frozen **CPU-authority** SegNet argmax
    (computed every K steps / at episode end; the only score-true term; NO-FAKE).
  - **Dense shaping (render-only, cheap, every step):** increase the φ-margin (top1−top2) on flip-prone
    pixels = raise the cooperative receiver's confidence on the annulus. This is the level-set "push the
    boundary so the receiver classifies the ragged contour correctly" signal. Potential-based shaping
    (Ng-Harada) so it does not bias the exact-d_seg optimum.
  - **Constraint (invariant): d_pose unharmed.** Pose rides the stored sidecar (`--w-pose 0`); φ-edits must
    not perturb the pose pair. Monitor exact d_pose (CPU-authority) as a constraint/penalty — never a proxy.
- **Episode.** One episode = refine ONE (or a small batch of) stalled annulus pair(s) from the **Muon BEST
  warm-start** toward lower d_seg over T≈50–200 steps; dense shaping each step + exact d_seg every K steps
  and at terminal; terminate on d_seg plateau or step budget. Reset = next stalled pair / re-warm-start.
- **Dynamics framing.** The φ field = level-set; actions move the zero-contour; the policy learns the
  Morse-Smale descent direction on the action manifold that the vanishing surrogate gradient cannot supply.
- **Algorithm.** Start **PPO** (robust continuous control, the PufferLib/CleanRL default; ~150 LOC: GAE +
  clipped surrogate + tiny Gaussian policy). Because the exact reward is expensive, flag **off-policy
  (SAC/TD3)** as the sample-efficiency upgrade (reuses each costly exact reward via a replay buffer) and
  **dense-shaping-dominant rollouts** (most steps use the cheap render-only φ-margin; exact d_seg is the
  periodic sparse anchor) as the way to keep exact-reward count ≪ env-step count — this is what makes a
  larger loop tractable under the ~5/s reward floor.

---

## 4. Substrate recommendation (on merits — velocity vs world-class-MLX)

**The policy is tiny (~32-dim obs → ~32-dim action, a small MLP); the bottleneck is the CPU-authority
reward env, not the policy framework.** Therefore framework choice barely moves velocity-to-unlock; pick the
lowest-integration-cost option that keeps the reward on authority.

| option | verdict | rationale |
|---|---|---|
| **PufferLib (torch)** | **draw ideas, do NOT adopt** | Built for policy-bound envs at ~1e6 steps/s; our env is **reward-bound at ~5/s** → its core advantage is moot, and it adds a torch-centric vectorization layer around a problem whose wall is the CPU-authority scorer. Keep its **vectorized-env + batched-reward + obs/action-emulation** ideas. |
| **CleanRL-style single-file PPO (torch)** | **acceptable — fastest stand-up** | ~150 LOC wrapping our existing numpy env; torch policy may ride MPS (gradient only). Lowest time-to-decisive-smoke if MLX-PPO primitives are slower to write. |
| **MLX-native tiny PPO** | **PREFERRED (world-class-MLX trajectory)** | Keeps the whole loop MLX (render is already MLX-GPU; policy on MLX). The only gap = no off-the-shelf MLX PPO/GAE — but GAE + clipped-surrogate is ~150 LOC. **MLX-improvement opportunity:** a reusable MLX `ppo`/`gae`/gaussian-policy primitive (advances the world-class-MLX goal; numpy-portable). |

**Recommendation:** for the **decisive micro-loop**, write a **~150-LOC single-file PPO** (MLX-native if it
costs ≤ a few hours more than torch; else CleanRL-torch) wrapping the existing numpy/CPU-authority env +
MLX-GPU render. **The env reward stays CPU-authority numpy-fp32 regardless of policy framework.** Defer the
PufferLib-grade vectorized harness until/unless a CUDA-reward scaling path makes the env policy-bound.

---

## 5. The decisive next-smoke PLAN (design — DO NOT RUN until greenlit / after the Muon arm)

**Question:** does RL-on-EXACT-d_seg move a *stalled* annulus residual the surrogate gradient has plateaued
on? (The make-or-break; the only thing that justifies the lab.)

**Minimal proof-of-concept:**
1. **Warm-start** from the Muon BEST ckpt: `experiments/results/levelset_thetastar_muon_arm/
   levelset_witness_ema_BEST.npz` (d_seg 0.004250 @ep850 — the flat tail). Read-only; do not touch the live run.
2. **Select a few STALLED annulus pairs** — pairs whose per-pixel d_seg on the Road↔Lane separatrix has
   plateaued in the Muon tail (identify via the per-stage annulus attribution flip-set; the surrogate-gradient
   slope there is ≈0 by construction → the clean test).
3. **RL micro-loop:** tiny PPO; action = per-pair code δ (~32-dim); T≈50–200 steps/episode; **~1e3–1e4 total
   exact-reward evals** (3–35 min CPU-authority). Dense φ-margin shaping each step (render-only) + EXACT
   through-R d_seg reward (CPU-authority) every K steps + at terminal; **exact d_pose monitored unharmed.**
4. **Success threshold:** RL lowers the EXACT through-R d_seg on the stalled pairs by **≥10–20% relative**
   on the targeted annulus (a drop the gradient could NOT achieve, since its slope there ≈0). NO-FAKE:
   measured on the CPU-authority through-R argmax; n = the few pairs (advisory **distortion go/no-go ONLY**,
   NOT a 600-row score; rate not measured here).
5. **Containment / scheduling:** $0, GPU-free (numpy + CPU-torch only). Runs CPU-only → does not contend the
   GPU; but the Muon arm's async CPU verdict shares CPU, so **schedule after the arm releases or throttle
   threads**. Resumable, seeded; per-episode checkpoints; `--min-free-gb 10`.

**Go/no-go on the FULL lab (gated by step 4):**
- **GREEN → build the lab** if the micro-loop moves a stalled annulus residual the gradient can't.
- **RED → no lab** if it cannot → the residual is the FEED-lq **aleatoric boundary-wobble floor** (content
  decision-noise, not a learnable direction) → switch to the surgical-repair toolbox (STORE the irreducible
  flips / UNIWARD-downweight / deterministic-gen, chosen by Δd_seg-per-byte under through-R survival).
  Either outcome is decisive signal.

---

## 6. GO/NO-GO with evidence grades

| decision | verdict | grade |
|---|---|---|
| RUN the decisive annulus-unlock micro-loop ($0, GPU-free, 3–35 min) | **GO** | timing `MEASURED-BY-US`; annulus-unlock hypothesis `DERIVED` (vanishing-gradient) — the micro-loop confirms |
| Build the FULL RL lab now | **DEFER (conditional-GO, gated on the micro-loop)** | `DERIVED` + means≠ends |
| Adopt PufferLib | **NO-GO (draw ideas only)** | `MEASURED-BY-US` (env is reward-bound ~5/s, not policy-bound → PufferLib's advantage moot) |
| Substrate for the micro-loop | **minimal single-file PPO (MLX-native preferred, CleanRL-torch acceptable)**; reward stays CPU-authority | `DERIVED` (policy tiny, reward dominates) |
| Classic 1e5–1e6-exact-step PPO on local CPU-authority | **NO-GO locally** (5.5–55 h) → decouple dense/exact reward or dispatch reward to CUDA | `MEASURED-BY-US` (~5 exact-rewards/s) |

---

## 7. NO-FAKE ledger

- Every reward in the design = the EXACT through-R frozen **CPU/CUDA-authority** SegNet argmax (+ d_pose
  constraint); MPS/torch-on-MPS is gradient/timing ONLY, never reward/score (CLAUDE.md "MPS never authority").
- Sanity `d_seg(real GT frame)=0.000000` (`MEASURED-BY-US`) proves the timed scorer is the real frozen SegNet.
- The "Apple-GPU 45 steps/s" projection in the smoke is **invalid for the reward** (uses the forbidden 104×
  MPS scorer) — flagged and superseded by the CPU-authority ~5 exact-rewards/s floor.
- **MEANS≠ENDS:** this memo moves no score. **Pointer UNMOVED contest-CPU 0.19109982.** The lab is justified
  ONLY by the annulus-unlock hypothesis + the cheap decisive gate; the END is a byte-closed n600
  `upstream/evaluate.py` row below 0.19110.

### Anchors
`project_rl_lab_levelset_exact_dseg_reward_direction_20260630` (spec) ·
`pr95_dseg_30k_convergence_deepmath_20260630T1542Z` (vanishing-gradient motivation) ·
`project_dseg_islands_8dim_manifold_go_generator_convergence` (8-dim lane orbit) ·
`CANONICAL_RESEARCH_INDEX_20260629` (veh-G2 annulus / veh-G4 FiLM / D-axis levers / I3 determinism) ·
`feedback_mlx_first_everything_numpy_portable_tinygrad_primitives` (world-class-MLX) ·
env primitives: `src/tac/boundary_math/lever_b_levelset_generator.py::levelset_rgb_forward_numpy` +
`experiments/train_witness_realized_through_R_mlx.py::{_torch_R_to_camera_uint8,cpu_verdict_d_seg_batch}` ·
warm-start: `experiments/results/levelset_thetastar_muon_arm/levelset_witness_ema_BEST.npz` (d_seg 0.004250) ·
smoke: `scratchpad/rl_env_step_timing_smoke.py` (GPU-free, CPU-authority).
```
