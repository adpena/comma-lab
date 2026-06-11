# Quantizr pose-implementation audit vs our capstone store-6-FiLM carrier (2026-06-11)

**Authority:** `[macOS-MLX research-signal]` / `[local CPU-torch advisory]` — NON-PROMOTABLE
per Catalog #192/#341. No contest score claimed; a score requires `upstream/evaluate.py` on
paired CUDA + Linux-x86_64 CPU. **mission = frontier_breaking_enabler.**

**Scope (pose-mechanism lane only):** `_PoseFiLM` (`vq_nerv_bundle.py`), `pose_film.py`,
the pose-loss + d_pose-measurement paths, the partition fn's FiLM routing. The curriculum
scheduler and the optimizer LR-schedule are sister-subagent lanes — UNTOUCHED here.

**Source of truth for Quantizr:**
`experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/repo/submissions/quantizr/{compress.py,inflate.py}`
(the real submission — `JointFrameGenerator`, `FiLMSepResBlock`, `pose.npy.br`).

---

## 1. Quantizr's ACTUAL pose mechanism (the empirical anchor)

Quantizr (PR#55/#56 lineage, szabolcs/selfcomp) is a **mask-conditioned** generator, NOT a
latent-NeRV. Its pose mechanism, read from the source:

- **Store** the 6-d GT PoseNet output per pair, `pose.npy.br` (fp32 → brotli q11; `compress.py:323-348`
  `extract_and_compress_poses`). The stored pose is **FROZEN** at inference — it is read from the
  archive and handed in, never optimized (`inflate.py:283-285`, `301`).
- **Render contract** (`JointFrameGenerator.forward`, `inflate.py:213-223` / `compress.py:565-580`):
  - a `SharedMaskDecoder` trunk turns the stored **mask** (5-class, `+ coord grid`) into a 56-ch feature;
  - **`frame2_head` (the static/reference frame) gets NO pose** — `Frame2StaticHead`, pure conv;
  - **`frame1_head` (the moving frame) is the ONLY pose-conditioned head.** Pose enters via
    `pose_mlp = Linear(6,48) → SiLU → Linear(48,48)` → a **`FiLMSepResBlock`** (`block1` of the
    head): the FiLM `gamma,beta = Linear(48, 2·56)` modulate the **conv output INSIDE a residual block**,
    at FULL 384×512 spatial resolution: `x = norm(conv2(conv1(x)))·(1+gamma)+beta`, then `act(residual+x)`.
- **Pose LOSS** (`compress.py:680-682, 707-708`): run PoseNet on the **rendered pair**, MSE vs PoseNet
  on the **GT pair** — `loss_pose = F.mse_loss(fake_pose, gt_pose)`, weighted `×10` (FINETUNE) / `×30`
  (JOINT). It is the contest term: difference of PoseNet(rendered) vs PoseNet(GT), NOT a fit to the
  raw stored scalars. The stored scalars are the *FiLM input*, the GT-pair PoseNet output is the *target*.
- **Optimizer:** plain `AdamW(betas=(0.9,0.99))` for the WHOLE generator (`compress.py:607`), with
  LinearLR warmup → CosineAnnealingLR, EMA 0.99, grad-clip 1.0, FP4 QAT. **No Muon anywhere.**
  → The pose MLP + FiLM are trained by AdamW in Quantizr by construction.

**Achieved d_pose (the anchor):** the #81 intake recorded Quantizr ≈ **0.00051** pose distortion at
88K params (the tube class). That number is the bar; our reads of the public archive corroborate the
mechanism (store-pose + FiLM-on-the-moving-head + AdamW).

---

## 2. Where OUR implementation diverges (fidelity gaps, ranked by likely pose impact)

| # | Quantizr | Our capstone (`_PoseFiLM` + bundle) | Divergence class |
|---|---|---|---|
| D1 | FiLM **inside a residual conv block** (`FiLMSepResBlock`), modulating a **conv output** at 384×512 | FiLM is a **single per-channel affine on the FINAL feature** right before the RGB head | **injection point + capacity** — ours is a global per-channel scale/shift with NO spatial conv between pose and pixels |
| D2 | FiLM on the **moving frame ONLY**; static frame is pose-free | FiLM on **both** frames (`film0`,`film1`), each its own head | structurally different motion factorization |
| D3 | Trunk is **mask-conditioned** — a full stored 384×512 5-class mask per pair (KILOBYTES of real geometry, AV1-coded; it IS the seg term) → frame has real scene structure; pose only adds motion | Trunk is a **content-free per-pair latent** = an **8-bit VQ index** into a shared 256-codebook (`codebook_size=256` → `ceil(log2)=8 bits/pair`, 600 bytes for ALL 600 pairs) → **ALL** image+pose structure must come through 8 bits + FiLM | **the deep one (quantified)** — our per-pair content carrier is **8 bits**; Quantizr's is a whole mask. Our FiLM has almost no scene to "move"; Quantizr's perturbs a real masked road |
| D4 | `pose_mlp` 6→48→48 **SiLU**, then `Linear(48,112)` FiLM proj | `_PoseFiLM` 6→32 **sin**, then `Linear(32,2·C)` | smaller hidden (32 vs 48), `sin` vs `SiLU`, no separate emb stage |
| D5 | Pose **FROZEN** (read-only archive scalars) | `stored_pose` is **trainable** (init = GT, drifts in-loop) | drift risk; the GT is the literal answer, fine-tuning it can only move it OFF the target the scorer measures against |
| D6 | AdamW for the FiLM/pose MLP | `pose_film*.fc{1,2}.weight` route to **Muon** (2-D + "weight") | optimizer-class mismatch (audit #3) |
| D7 (measurement) | n/a | `mean_d_pose` measured **clamp-only** (no eval_roundtrip) while loss + `exact_d_seg` DO roundtrip | reported d_pose **understated** the contest value (audit #4) |

The **`pose_film.py` `PoseFiLMDecoderMLX`** (the #84 module) has the same D1/D3 shape as the bundle
(FiLM at the final feature over a content-free latent) — so the fidelity gaps apply to BOTH
implementations; neither matches Quantizr's FiLM-in-conv-block-over-a-masked-trunk.

---

## 3. Ranked root causes of the d_pose 0.06–0.34 oscillation

1. **[STRUCTURAL — D3, the dominant one] The carrier has no geometry for pose to move.**
   Quantizr's pose FiLM perturbs a **real masked scene** (the trunk already renders road/sky/cars
   from the stored mask). Our pose FiLM modulates a **content-free 28-d latent render** — there is
   no road for the camera to move past, so the FiLM must *synthesize the entire frame0↔frame1 optical
   difference from 6 scalars through one per-channel affine*. The PoseNet reads ego-motion off that
   differential; a global per-channel scale/shift on a featureless render cannot produce a *consistent*
   optical-flow field, so PoseNet's read **wanders** epoch-to-epoch as the latent and FiLM co-adapt →
   the 0.06–0.34 bounce. This is the same class as the #80 "pixel-RMSE<3 wall" but now at the *pose*
   axis: a single-affine FiLM over a content-free latent is under-powered for the motion the scorer reads.
2. **[CAPACITY/INJECTION — D1+D4] Single per-channel affine at the final feature is too weak.**
   No spatial conv sits between pose and pixels, so pose cannot produce *spatially-varying* motion
   (ego-motion is a spatially-structured flow field). Quantizr puts FiLM *inside a conv residual block*
   so the modulation propagates through a conv → spatial structure.
3. **[OPTIMIZER — D6, audit #3] FiLM→Muon.** Newton-Schulz gives grad-norm-INDEPENDENT O(1) steps;
   for a zero-init pose MLP this is a real instability candidate. **HONEST EMPIRICAL CAVEAT (below):
   in the synthetic reachable-pose harnesses Muon-FiLM actually converged BETTER than AdamW-FiLM —
   so Muon is NOT the dominant driver there; it is a tuning fork, not the structural blocker.**
4. **[DRIFT — D5] Trainable stored pose** can move the FiLM input off the GT the scorer compares to.
5. **[MEASUREMENT — D7, audit #4] Clamp-only d_pose** under-reports the contest value (uint8 quant is
   exactly where pose drifts) — an oscillation read on a clamp-only number is itself partly an artifact.

---

## 4. What I implemented + the test result

**Implemented (real, tested, parity-checked):**

- **Fix D6 (FiLM→AdamW) as an OPT-IN hook, additive + non-breaking, DEFAULT OFF.** Added
  `force_adamw_substrings: Sequence[str]|None` (default `None` = byte-identical) to
  `apply_pr95_mlx_optimizer_step` (`pr95_hnerv_mlx.py`). When set, any param whose lowered name
  contains a needle is routed OUT of Muon INTO AdamW. The shared PR95
  `partition_pr95_mlx_parameter_names` and every other caller are **untouched** (verified: the 25-test
  `test_pr95_hnerv_mlx.py` suite passes; all 4 production callers pass 4 positional args only). The
  hook is exposed on the capstone (`CapstoneTrainConfig.force_film_to_adamw`) and the
  `pose_film_trainer` (`PoseFilmTrainerConfig.force_film_to_adamw`), **both defaulting FALSE.**
  Routing is observable via the new `forced_to_adamw_parameter_names` summary field (Catalog #305).
  **WHY DEFAULT OFF (the honest reversal):** I initially defaulted it `True` per the audit's #3
  hypothesis, but that BROKE the existing validated `test_severed_film_holds_pose_worse_no_fake_control`
  (FiLM-on 0.0279 > severed 0.0244 — AdamW-FiLM is *weaker* than Muon-FiLM, so the load-bearing-FiLM
  property fails). The synthetic A/B confirmed the same. Defaulting a refuted fix ON is exactly the
  NO-FAKE failure mode — so the hook is opt-in, default OFF, pending a real-scorer A/B.
- **Fix D7 (d_pose roundtrip), measurement-honesty.** `CapstoneTrainer.mean_d_pose` now routes through
  `bridge.exact_d_pose` (applies the SAME eval_roundtrip as the loss + `exact_d_seg`) instead of the
  bespoke clamp-only `_exact_d_pose`. Reported d_pose is now the contest-honest (uint8-roundtripped)
  value.

**Tests (NO-FAKE, behavioral):** `src/tac/capstone_vq_nerv/tests/test_film_adamw_routing.py` — 5 tests,
all pass. The decisive one (`test_forced_adamw_step_scales_with_gradient_norm_muon_does_not`) proves the
routing actually took the AdamW branch by the Muon-vs-AdamW DISCRIMINATOR: a Muon step's magnitude is
grad-norm-independent (orthogonalized), an AdamW step's is not — if the fix were a no-op, the assertion
that forced-AdamW magnitude ≠ Muon magnitude FAILS. The default-None path is asserted byte-identical to
the PR95 partition. The 25-test `test_pr95_hnerv_mlx.py` suite passes unchanged (additive default safe).

**Empirical A/B (HONEST, including the refutation):**
- *Synthetic reachable-pose harnesses* (linear `_GlobalReadPose` and a harder nonlinear differential
  read): **Muon-FiLM reached a LOWER d_pose than AdamW-FiLM** at equal budget (e.g. nonlinear seed-0
  Muon 0.0058 vs AdamW 0.0187; seed-2 Muon 0.0032 vs AdamW 0.0446). → **The simple "FiLM→Muon is THE
  oscillation cause" hypothesis is NOT supported in these harnesses.** Muon's large steps help a small
  *reachable* MLP. The FiLM→AdamW fix is correct PR95-discipline and removes a real instability class,
  but the synthetic evidence says the **dominant** driver is structural (D3/D1), not the optimizer.
- *Real FastViT PoseNet A/B (n=24, 40ep, roundtripped d_pose):* launched serially against the cached
  real GT targets (`gt_targets_n24.pt`); it is a CONFIRMING-only data point and does NOT gate the
  verdict (the synthetic refutation + D3 structural analysis are decisive and the conservative default
  OFF is correct either way). If it shows AdamW-FiLM materially beats Muon-FiLM against the real
  PoseNet, that flips the capstone default to ON via the existing hook (one-line config change + a
  reactivation row); if not, the hook stays opt-in. This is wired so the decision is a config flip,
  not a re-implementation.

---

## 5. Can a FiLM-conditioned decoder reach d_pose ~1e-4? — VERDICT

**On a content-free latent carrier (our capstone + `pose_film.py`): structurally doubtful at the tube.**
The synthetic harnesses show FiLM reaches ~3e-3 ONLY when the pose target is *linearly reachable* from a
fixed global read. The real PoseNet reads a *spatially-structured ego-motion flow* off the frame0↔frame1
differential; a single per-channel affine over a geometry-less latent render cannot synthesize a
consistent flow field, which is why d_pose wanders. **Quantizr reaches 5e-4 because its FiLM perturbs a
real mask-conditioned scene inside a conv block — the geometry it needs to "move" is already there.**

**Recommendation (the factorized carrier):** the capstone's `[VQ-index]⊕[pose-store]` over a content-free
latent is the wrong substrate for the tube. The Quantizr-faithful path is `[mask-blob]⊕[pose-store]`: a
**mask-conditioned trunk** (the SegNet argmax is already a cheap stored carrier — it's the seg term) +
**FiLM-in-a-conv-residual-block on the moving frame only** + **AdamW** + **frozen stored pose**. That is
literally Quantizr's `JointFrameGenerator`. The capstone should EITHER adopt the mask-conditioned trunk
(merging the seg-blob and pose-store into one Quantizr-shaped carrier) OR accept that the pure-latent
VQ-NeRV substrate holds pose only to ~1e-2, not the 1e-4 tube. The pose-mechanism fixes here (AdamW
routing + honest roundtrip measurement) are necessary hygiene but **not sufficient** — the carrier
geometry (D3) is the binding constraint.

---

## Canonical-vs-unique decision per layer (Catalog #290)

- `force_adamw_substrings` optimizer hook: **FORK_PRINCIPLED** of the routing only (PR95 Muon class is
  conv-hidden-weights-only; the pose path is a capstone addition PR95 lacks). Shared fn untouched =
  ADOPT_CANONICAL for every other caller.
- d_pose roundtrip: **ADOPT_CANONICAL** — reuse `bridge.exact_d_pose` (the same path the loss uses).

## 6-hook wire-in (Catalog #125)

1. sensitivity-map: N/A (no new byte-axis). 2. Pareto: N/A. 3. bit-allocator: N/A.
4. cathedral autopilot: N/A (research-signal, non-promotable). 5. continual-learning: this memo + the
A/B rows are the posterior signal for the pose-mechanism lane. 6. **probe-disambiguator: ACTIVE** — the
synthetic-vs-real A/B IS the disambiguator between "optimizer-class oscillation" (refuted) and
"structural-carrier-geometry" (supported) hypotheses.

## Cross-references

- `.omx/research/per_step_optimizer_training_poison_audit_20260611T014320Z.md` (#3 FiLM→Muon, #4 d_pose roundtrip).
- Quantizr source (the empirical anchor) — paths in §0.
- The #80 pose-crux / #81 store-pose escape memos (the lineage this audit refines at the pose axis).
