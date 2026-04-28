# Lane M-V2 Skunkworks Council Audit — 2026-04-28
## Extreme Rigor Review of the 1.84 [contest-CUDA] Regression

**Auditor**: Skunkworks council (Yousfi + Fridrich + Hotz + Quantizr + Contrarian)
**Mandate**: rigor audit + result interpretation + Lane M-V3 design + composability + re-anchor
**Charter**: non-conservative. Burden of proof on NOT trying something. Mathematical/scientific/empirical arguments only.

---

## TL;DR (the smoking gun)

Lane M-V2 has a **train-inference distribution mismatch** in the renderer pose-input. During the pose-TTO optimization the renderer is fed `[zoom, 0, 0, 0, 0, 0]` (zero-padded dims 1-5). The saved archive contains `[zoom, baseline_1, ..., baseline_5]` (Lane-A frozen-baseline padding). At inflate the renderer is fed the saved tensor — i.e., a different pose distribution than the optimizer was solving for.

The "V2 fix" — switching the SAVE-block padding from zero to frozen-baseline — was a **half-fix**: it changed what the saved bytes mean but did **not** change the OPTIMIZATION-time render to match. The `_project_to_renderer_pose()` helper in `experiments/optimize_poses.py` still zero-pads (line 763-769). The optimizer's gradient and the inflate-time render now operate on **two different pose distributions**.

**Empirical confirmation (collected during this audit)**:
- Lane A's optimized dims 1-5 std ranges 0.34 to 2.07 (**not** noise; carry signal)
- Lane M-V2 saved dims 1-5 are **byte-identical** to Lane A's (zero diff)
- Lane M-V2 saved dim 0 std = 2.66 vs GT-target dim 0 std = 1.26 (**2.1× inflated** — optimizer overshooting because zero-padded dims 1-5 forces dim 0 to compensate)
- GT-target dim 0 mean = 31.30; Lane A optimized dim 0 mean = 31.57; Lane M-V2 mean = 32.73 (**+1.9 vs GT**)

**Verdict**: The rank-1 hypothesis is NOT yet "architecturally dead". It is **under-engineered**. Lane M-V2 has 3 design bugs and 1 logic bug. A V3 that fixes all 4 is mathematically motivated and testable in $0.30/2h.

---

## SECTION 1 — RIGOR AUDIT (deliverable #1)

### 1.1 BUGS

#### BUG-1 (CRITICAL, load-bearing): Train/inference pose-input distribution mismatch

**Location**: `experiments/optimize_poses.py`
- Optimization render path: lines 752-770 (`_project_to_renderer_pose`)
- Save composition path: lines 2131-2149 (frozen-baseline composition)
- Inflate render path: `submissions/robust_current/inflate_renderer.py:2141` (`batch_pose = poses[pose_start:pose_end]`)

**Code, the divergent paths:**

`_project_to_renderer_pose` (used by optimizer at line 783, 918, 948):
```python
def _project_to_renderer_pose(cond: torch.Tensor) -> torch.Tensor:
    opt_part = cond[:, :pose_dim_internal]
    if pose_mode == "radial-zoom" and pose_dim_internal != pose_dim:
        zeros_pad = torch.zeros(
            opt_part.shape[0], pose_dim - pose_dim_internal, ...
        )
        return torch.cat([opt_part, zeros_pad], dim=-1)   # ← ZERO PAD
    return opt_part
```

Save block (line 2144):
```python
baseline_aux = init_poses[:n_pairs, 1:6].detach().cpu().to(pose_part.dtype)
optimized_poses = torch.cat([pose_part[:, :1].cpu(), baseline_aux], dim=-1)  # ← FROZEN BASELINE PAD
```

**Why this is a CRITICAL bug**: the optimizer solves
> minimize  L( segnet(render(masks, [zoom, 0,0,0,0,0])), gt_seg )  +  L( posenet(render(masks, [zoom, 0,0,0,0,0])), posenet_gt )
> w.r.t. zoom

But the inflate eval computes
> score( render(masks, [zoom*, baseline_1...5]) )

The optimum `zoom*` for the zero-pad distribution is **not** the optimum for the frozen-baseline distribution. The renderer's FiLM gates respond non-linearly to dims 1-5, so the rendered frames are visibly different.

**Empirical scale**: Lane A's optimized dims 1-5 have std up to 2.07 (range -4.7 to +4.9). These are **3-4σ-scale** input perturbations to the FiLM layer; not "small enough to ignore". The renderer was trained on a 6-DOF distribution and has zero training data with constant-zero dims 1-5.

**Memory cross-reference**: This is exactly the failure mode predicted by `project_baseline_poses_load_bearing` (renderer + poses are JOINT artifact; 33% pixel shift, 23× PoseNet degrade with zero-init).

#### BUG-2 (HIGH, load-bearing): Argmax-constraint check uses zero-pad distribution

**Location**: `experiments/optimize_poses.py:918, 948`

The argmax-rejection forward (when `--argmax-constraint` is set) calls `_project_to_renderer_pose(conditioning)` again (lines 918, 948). Even though Lane M-V2 didn't use `--argmax-constraint`, this is the same logic error: the rejection criterion would test the zero-pad distribution, not the frozen-baseline one. Fixing BUG-1 fixes this transitively.

#### BUG-3 (MEDIUM): Proxy-score block compares apples-to-oranges

**Location**: `experiments/optimize_poses.py:2218-2236`

After the save block produces the (N, 6) frozen-baseline tensor, the proxy-score path (line 2225) calls `_generate_frames(renderer, gt_masks, optimized_poses_for_render, ...)` — but `optimized_poses_for_render` IS the saved (N, 6) frozen-baseline tensor (since `optimized_poses.shape[1] == 6`).

So the proxy DELTA score (`opt_score - gt_score`) is computed on the frozen-baseline distribution, while the OPTIMIZER's intermediate `pose_loss` was computed on the zero-pad distribution. **The proxy score is the only sane signal in the run**, but the optimizer didn't optimize against it. Operator gets a misleading "improvement" value and no warning that the optimizer never saw this signal.

The fact that this proxy was probably reasonable while the contest-CUDA result regressed is consistent with the proxy/auth gap rule (`feedback_proxy_auth_math_useless`) — but here the bug is local: the proxy and the optimizer's loss are evaluating different functions of `zoom`.

### 1.2 ENGINEERING ISSUES

#### ENG-1: No regression test for the BUG-1 mismatch

`src/tac/tests/test_optimize_poses_radial_zoom_save_shape.py` (T1-T9) exhaustively pins the SAVE-side semantics, but nothing tests that `_project_to_renderer_pose` matches the save-block composition. A simple test:

```python
# Synthetic: load (N, 6) init_poses, run optimize_poses_batch for 1 step
# in radial-zoom mode, verify the saved tensor's [:, 1:6] equals what the
# renderer was actually fed at the LAST optimization step.
```

would catch any future divergence. This is the same class of bug as `feedback_dead_flag_wiring_pattern` ("inventing CLI flags from intent without grepping the target") — here the "intent" is "frozen baseline" but the "target" (`_project_to_renderer_pose`) was never updated.

#### ENG-2: The 500-step horizon is questionable for 1-DOF optimization

Lane M+N V1 ran 500 steps; Lane M-V2 inherited that. But the optimizable is `(N, 1)` — 600 scalars total. With Adam + lr=0.01 + batch_pairs=8, every batch updates 8 scalars; total of 75 batches/epoch × 500 steps would way over-fit if run that long. The published `optimize_poses_batch` returns when the batch's `early_stop_patience` triggers, so per-batch step counts likely ≪ 500 in practice — but no metric is logged proving so. Need to see `batch_metrics["steps_run"]` per batch to confirm the optimization wasn't pathological.

#### ENG-3: No comparison render at inflate time

The script ships the archive direct to `contest_auth_eval` without a smoke render comparison. A 5-pair render of {Lane A archive, Lane M-V2 archive} would have visualized the off-manifold artifacts. Cost: ~30s; would have telegraphed the issue before $0.30 of CUDA spend.

#### ENG-4: The "V2 sanity check" is shape-only, not semantic

The script's stage-2 sanity check (line 139-149) verifies `shape[1] == 6`. It does not verify that dims 1-5 match init_poses (i.e., the frozen-baseline padding actually happened). A mutation of the save block to e.g. `torch.cat([pose_part[:, :1], torch.zeros_like(baseline_aux)], dim=-1)` would pass this check while regressing to the V1 zero-pad bug at inflate.

### 1.3 DESIGN ISSUES

#### DES-1: Wrong premise — "freezing dims 1-5" is incoherent for a 6-DOF-trained renderer

The Lane M-V2 design assumed: "freeze dims 1-5 to baseline, optimize only dim 0, the renderer will produce frames consistent with both". This is **only true** if the renderer is **linear** in dims 1-5 — which a FiLM-conditioned conv decoder demonstrably is not.

A FiLM layer applies `gamma(pose) * x + beta(pose)` where `gamma, beta` are MLPs of the pose vector. The MLP is non-linear. So `render(zoom, baseline_1..5) ≠ render(zoom, 0..0) + render(0, baseline_1..5)`. The optimizer cannot "isolate" zoom from the dims-1-5 conditioning by setting them to zero.

The CORRECT design choice was either:
- (A) Run the optimization render with frozen-baseline padding TOO (fix BUG-1), OR
- (B) Fine-tune the renderer briefly at compress-time on the zero-pad distribution, then save with zero-pad (compress-time-unlimited per `feedback_compress_time_unlimited_archive_small_20260428`)

V2 picked neither. It left the optimizer in distribution-A and saved in distribution-B.

#### DES-2: Wrong subspace — radial-zoom is "1-DOF in PoseNet output space", not "1-DOF in renderer input space"

`project_posenet_rank1_discovery` says: PoseNet's Jacobian on YUV6 input is rank-1, with 99.8% variance in dim 0 of its OUTPUT. This is a property of PoseNet looking at frames.

The hypothesis "I can constrain the renderer's pose INPUT to 1-DOF" is a **distinct claim**. For it to follow from the rank-1 PoseNet finding, you need the renderer's INPUT-OUTPUT mapping to be rank-aligned with PoseNet's INPUT-OUTPUT mapping. There is **no theoretical reason** for that to hold. The renderer has a 6-dim FiLM input that it learned to use during training; its input-side rank is whatever the training distribution made it.

Memory `project_lane_mn_radial_zoom_negative_20260427` already labels this as a "category error". V2 did NOT change the category error — it only fixed a save-side shape pitfall.

#### DES-3: Wrong objective — pose_loss is MSE on PoseNet output, not on the rank-1 dim 0

The pose_loss at line 821 is `F.mse_loss(pose_out, pose_targets[:B])` over all 6 PoseNet output dims. If the rank-1 finding is real, only dim 0 matters; dims 1-5 of the PoseNet output are noise (per `project_posenet_rank1_discovery`: dims 1-5 have range 0.06-0.20 i.e. noise-level).

Including the noise dims in the MSE means the optimizer is also chasing PoseNet noise — and on the zero-padded renderer input, that noise is amplified non-linearly. A clean V3 should use `F.mse_loss(pose_out[:, 0:1], pose_targets[:B, 0:1])` if it commits to the rank-1 hypothesis (and even that is wrong if BUG-1 isn't fixed first).

### 1.4 ARBITRARINESS

#### ARB-1: predicted band [1.10, 1.30] was magical thinking

The `--predicted_band [1.10, 1.30]` written into `provenance.json` was based on Lane A's 1.15 + an unjustified hope that "freezing dims 1-5 unlocks a fraction of the 0.10 KL-distill gap". There was no math. The actual bands of similar 1-DOF-on-6-DOF-renderer experiments are known: Lane M+N V1 = 2.35, baseline = 2.29. Predicted band should have been `[2.20, 2.50]` from precedent. The "BEAT Lane A 1.15" framing in the script header was aspiration, not analysis.

#### ARB-2: lr=0.01 inherited from full-6dof default

`--lr` default is 0.01 for Adam over 6-dim conditioning. For a 1-DOF optimization with std-1 to std-2 target dim, the gradient magnitudes are **6× larger** per scalar (the loss is summed over 6 PoseNet dims, but only 1 input dim absorbs it). lr=0.01 is likely too aggressive. The 2.1× std inflation in the saved dim 0 (vs GT std 1.26) is consistent with lr-overshoot. A V3 should sweep lr ∈ {1e-3, 3e-3, 1e-2, 3e-2}.

#### ARB-3: 500 steps inherited from full-6dof default

The full-6dof Lane A path uses 500 steps because the loss landscape over (N, 6) is harder than (N, 1). Lane M-V2 inherited 500 unchanged. With early_stop_patience working, this is probably fine — but it's an un-justified default.

#### ARB-4: --eval-roundtrip + --posetto-noise-std=0.5 ("Fridrich C1 fixes")

These are good defaults (memory `feedback_eval_roundtrip_everywhere_FINAL`). But invoking them as "Fridrich C1 fixes from prior council" without re-validating their numeric values for the 1-DOF case is hand-wavy. The C1 fixes were designed for 6-DOF; for 1-DOF they may over-regularize.

---

## SECTION 2 — RESULT INTERPRETATION (deliverable #2)

### 2.1 Is the PoseNet 0.076 the REAL signal?

**YES, it is the real signal.** The contest-CUDA eval is byte-deterministic on a fixed archive (verified by sha256 in `contest_auth_eval.json`). The 15× regression vs Lane A's 0.005 is the actual PoseNet response to the saved tensor.

But the 0.076 is **not the signal of the rank-1 hypothesis**. It is the signal of the BUG-1 train-inference mismatch. Specifically:
- The optimizer found `zoom*` such that `posenet(render(masks, [zoom*, 0, 0, 0, 0, 0]))` ≈ `gt_pose_targets`
- The inflate render produces `render(masks, [zoom*, baseline_1..5])` which is **not** what the optimizer was solving for
- PoseNet on the actual rendered frames disagrees with GT by 15×

### 2.2 Is the rank-1 hypothesis ACTUALLY DISPROVEN?

**NO.** The hypothesis has not yet been tested cleanly. Three failure modes are conflated in V1 and V2:

| Failure mode | V1 (2.35) | V2 (1.84) |
|---|---|---|
| Save-side shape bug (zero-pad at inflate) | Present | FIXED |
| Train-inference distribution mismatch (zero-pad at train) | Present | **STILL PRESENT** (BUG-1) |
| Renderer trained on 6-DOF cannot accept 1-DOF input cleanly | Present | Present |

Until BUG-1 is fixed, we cannot distinguish "rank-1 hypothesis is empirically wrong" from "zero-pad-at-train is an artifact". The V3 design below addresses BUG-1 and gives the rank-1 hypothesis its first clean test.

### 2.3 Off-manifold renderer because of architecture mismatch OR baseline-pad inadequacy?

**Both.** But BUG-1 dominates. A V3 fix that aligns train and inference padding (Section 3) will tell us which is bigger.

The architecture-mismatch concern (renderer trained on 6-DOF) remains valid: even with frozen-baseline padding at BOTH train and inference, the optimizer is solving a constrained problem (only dim 0 free) on a 6-DOF-trained model. The constraint may produce sub-optimal `zoom*` because the FiLM layer can't compensate via dims 1-5 even when it would help.

But that is a SECONDARY effect. BUG-1 is the first-order bug.

### 2.4 Why was V2 better than V1?

V1 saved (N, 1) and the (no-op) inflate-side adapter zero-padded. So V1 had:
- Train: zero-pad
- Inflate: zero-pad (consistent!)
- BUT: zero-pad violates the joint pose-renderer artifact rule (`project_baseline_poses_load_bearing`); zero is far from the trained distribution → renderer goes maximally off-manifold → PoseNet=0.27

V2 has:
- Train: zero-pad
- Inflate: frozen-baseline (closer to trained distribution)
- Mismatch causes a different problem: optimizer found the wrong `zoom*`, so even though the inflate distribution is now in-manifold, it has a wrong dim 0 → PoseNet=0.076

V2 is BETTER because frozen-baseline-at-inflate is closer to manifold than zero-at-inflate, even with the wrong `zoom*`. The V1→V2 improvement (2.35 → 1.84) is the SHAPE-FIX-only benefit. The remaining gap to Lane A (1.15) is BUG-1 + DES-1/2/3.

---

## SECTION 3 — LANE M-V3 DESIGN (deliverable #3)

### 3.1 Council decision matrix

The council debated 4 V3 candidates:

| Candidate | Premise | Predicted band | Cost | Risk |
|---|---|---|---|---|
| V3-A: Fix BUG-1 only (frozen-baseline at train AND inflate) | The smallest fix that aligns train and inference | [1.05, 1.20] | $0.30 / 2h | LOW |
| V3-B: V3-A + only-dim-0 PoseNet loss | Commits to rank-1 hypothesis cleanly | [1.05, 1.25] | $0.30 / 2h | MEDIUM |
| V3-C: Lane M+ — fine-tune renderer at compress-time on zero-pad distribution | Make the renderer accept the 1-DOF input cleanly | [1.00, 1.40] | $2 / 8h | HIGH |
| V3-D: Hyperbolic foveation (Telescope per `project_cosmos_mae_lyra_telescope_synthesis`) | Different mechanism — Lane HF | [0.85, 1.10] | $4 / 16h | MEDIUM |

**Yousfi**: "V3-A first. The rank-1 hypothesis hasn't been falsified yet because BUG-1 contaminated V2. We do the cheapest valid test."

**Fridrich**: "V3-B is more rigorous — if rank-1 is real, dims 1-5 of PoseNet output are noise and including them in the loss IS noise. But V3-A is a strict subset of V3-B (just zero out the loss weight on dims 1-5 for free). Combine."

**Hotz**: "V3-C is the only one that actually answers 'can the renderer handle 1-DOF input?'. Skip A/B and go straight there. A 4-DOF compress-time fine-tune costs nothing on Vast.ai."

**Quantizr**: "Hotz is right that V3-C answers the cleaner question, but A is $0.30 and tests BUG-1 specifically. Stack: V3-A first, then V3-C only if V3-A still regresses."

**Contrarian**: "Stop. Why Lane M at all? The wedge attribution from `project_lane_g_v3_stacking_skunkworks_20260428` says PoseNet is 20% of the gap to Quantizr 0.33; SegNet is 43%. Why are we burning council cycles on a 20% wedge that's already at floor in Lane G v3 (PoseNet=0.0035)? Lane M's MAXIMUM payoff is converting 0.005 → 0.003 which is +0.001 score. Even if V3 lands at 0.99 it's a stack-essential nothingburger."

**Tripartite vote**: Yousfi + Fridrich (VOTE: V3-A+B combined, drop C/D from Lane M scope). Contrarian DISSENTS — argues kill Lane M entirely and redirect $0.30 to Lane EC-V2.

**Resolution**: V3-A+B combined (call it **Lane M-V3-clean**), but ALSO acknowledge Contrarian's valid point: this is a **diagnostic** experiment, not a frontier candidate. We run it because it disproves OR validates the rank-1 hypothesis on the renderer-input side, which informs Lane GE / Lane LM-V2 / Lane MOS designs (all use rank-1 priors). The experiment value is "kill the hypothesis cleanly" not "win the leaderboard".

### 3.2 Lane M-V3-clean (recommended) — full spec

**Goal**: Test the rank-1 hypothesis on the renderer-input side, with NO train-inference distribution mismatch.

**Code change** (single function, in `experiments/optimize_poses.py`):

Modify `_project_to_renderer_pose` to take `init_poses` as a parameter and pad with frozen baseline:

```python
def _project_to_renderer_pose(
    cond: torch.Tensor,
    init_poses_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lift optimizable to (B, pose_dim) using frozen-baseline aux pad in
    radial-zoom mode (matches the SAVE block's composition for train/
    inference parity). init_poses_aux is (B, pose_dim - pose_dim_internal)
    — the frozen baseline aux dims for THIS BATCH.
    """
    opt_part = cond[:, :pose_dim_internal]
    if pose_mode == "radial-zoom" and pose_dim_internal != pose_dim:
        if init_poses_aux is None:
            raise RuntimeError(
                "Lane M-V3: --pose-mode radial-zoom requires init_poses_aux. "
                "The V2 path used zero-pad here, which created a train/inference "
                "distribution mismatch with the (frozen-baseline-padded) save."
            )
        return torch.cat([opt_part, init_poses_aux], dim=-1)
    return opt_part
```

Plumb `init_poses[:, 1:6]` through `optimize_poses_batch` to `_project_to_renderer_pose` (already in scope as `init_poses` argument).

**Optional V3-B addition** (1 line):
```python
pose_loss = F.mse_loss(
    pose_out[..., :rank1_dims],
    pose_targets[:pose_out.shape[0], :rank1_dims].to(device),
)
```
where `rank1_dims=1` for radial-zoom and `rank1_dims=6` (default, current behavior) otherwise. CLI flag `--posenet-loss-dims=1`.

**Anchor**: Lane A 1.15 baseline (same as V2). DO NOT use Lane G v3 1.05 — Lane G v3 has KL-distill in the renderer training and additional pose TTO; layering Lane M on top entangles ablations. The first thing we want to know is "does fixing BUG-1 unlock the rank-1 hypothesis on the same anchor V2 used?"

**Hyperparameters**:
- `--steps 500 --batch-pairs 8` (V2 inheritance — keep for direct comparison)
- `--lr 0.003` (NEW: V2's 0.01 likely overshot; sweep [0.001, 0.003, 0.01] in a follow-up)
- `--pose-mode radial-zoom`
- `--gt-poses-path experiments/results/lane_a_landed/optimized_poses.pt`
- `--eval-roundtrip --posetto-noise-std 0.5`
- `--posenet-loss-dims 1` (NEW: V3-B option; if too aggressive, drop to default 6)

**Tests** (add to `src/tac/tests/test_optimize_poses_radial_zoom_save_shape.py`):

```python
def test_v3_train_inference_pose_distribution_parity():
    """V3-CRITICAL: the optimization-time render must use the SAME
    pose padding as the saved tensor's. With BUG-1 fixed, both use
    frozen-baseline padding."""
    # 1. Load (synthetic) init_poses with non-zero dims 1-5
    # 2. Run optimize_poses_batch(pose_mode='radial-zoom', steps=1)
    # 3. Hook into _project_to_renderer_pose; verify the returned
    #    tensor has dims 1-5 == init_poses[:, 1:6] (NOT zero)
    # 4. Verify the saved tensor's dims 1-5 also == init_poses[:, 1:6]
    # 5. Assert byte equality between train-time pose and saved pose
```

This test is the BUG-1 regression gate; without it any future Lane M variant could re-introduce the divergence.

**Predicted band**: [1.05, 1.20] [contest-CUDA].
- **1.05**: if rank-1 hypothesis holds, Lane M-V3 ≈ Lane A 1.15 minus a small dim-0-only-MSE benefit → closer to 1.10
- **1.20**: if rank-1 hypothesis is partially wrong (auxiliary dims ARE used non-trivially by the renderer), V3 ≈ Lane A 1.15 plus some constraint penalty → 1.18
- **>1.30**: rank-1 hypothesis is dead even with BUG-1 fixed → kill the lane permanently

**Cost**: $0.30 / 2h Vast.ai 4090.

**Decision criterion**: 
- V3 ≤ 1.20 → rank-1 hypothesis validated as a CONSTRAINT (not as a ground-truth subspace). Inform Lane GE / Lane LM-V2 / Lane MOS — they can use rank-1 priors safely.
- V3 ∈ (1.20, 1.50) → rank-1 hypothesis useful as warm-start only, not as constraint. Move to Lane HF (Telescope).
- V3 > 1.50 → rank-1 hypothesis architecturally dead on the dilated-h64 renderer. Kill all Lane M variants. Inform Lane GE that 1-DOF Chebyshev poses likely also fail.

### 3.3 Why NOT Lane HF as the V3

The user's mandate asks specifically about "Lane M-V3 design". The Cosmos synthesis (`project_cosmos_mae_lyra_telescope_synthesis_20260428`) labels Lane HF as "the proper revival path" for the radial-zoom idea — this is correct as a SUCCESSOR for the rank-1 mechanism (Telescope's `Φ(x) = (1-w(r))·x + w(r)·h(x)` is a proper invertible foveation, not a 1-DOF pose-input constraint).

But Lane HF is a different lane (different input-output topology, different training requirements, different cost $4 vs $0.30). Calling it "Lane M-V3" conflates the lane taxonomy. The disciplined path:
1. **Lane M-V3-clean**: fix BUG-1, run $0.30 diagnostic. KILL or VALIDATE the renderer-input rank-1 hypothesis cleanly.
2. **Lane HF**: independent lane, $4. Different mechanism (post-renderer foveation, not pre-renderer pose constraint).

If Lane M-V3-clean lands at 1.05-1.20, we have a valid rank-1 lane. If it doesn't, Lane HF still has its own predicted band [0.85, 1.10] independently.

### 3.4 V4 / V5 outlook

If V3-clean validates (≤1.20):
- **V4**: lr sweep + posenet-loss-dims sweep + 6-DOF-LoRA composition (rank-1 + low-rank residual). Predicted [1.00, 1.15]. Cost $0.50.
- **V5**: V4 + Lane G v3 KL-distill auxiliary on Lane M warm-start. Predicted [0.95, 1.10]. Cost $1.00.

If V3-clean fails (>1.20):
- Kill all V3+ Lane M variants.
- Inform Lane GE / Lane LM-V2 / Lane MOS that their rank-1-on-renderer-input priors are likely contaminated.

---

## SECTION 4 — COMPOSABILITY (deliverable #4)

Even raw 1.84, Lane M-V2's contribution to a stack is **near zero** because:

| Stack candidate | Lane M-V2 contribution | Reasoning |
|---|---|---|
| Lane M-V2 + Lane EC | None | Lane M-V2's SegNet=0.005 (at floor); EC adds nothing on top. Lane EC's wedge is on a HIGH-SegNet anchor. |
| Lane M-V2 + Lane HF (Telescope foveation) | NEGATIVE | Both target pose-mechanism. Lane HF expects baseline poses; Lane M-V2's poses are a constrained-1-DOF subset that Telescope's invertible warp would compose with poorly. |
| Lane M-V2 + Lane SAUG-V2 | Possibly NEUTRAL | SAUG attacks proxy/auth gap during renderer training. Lane M-V2's optimized tensor would be SAUG's input. The off-manifold renderer issue (BUG-1) might or might not be partially absorbed by SAUG's noise schedule, but no theoretical grounding to predict net effect. |
| Lane M-V2 + Lane G v3 | NEGATIVE | Lane G v3 already optimized poses at 0.0035 PoseNet. Lane M-V2's 0.076 is 21× worse. No way to "compose" two pose tensors — only one can be in the archive. |
| Lane M-V2 + Lane W (rate attack on renderer) | NEUTRAL | W changes renderer.bin bytes; M-V2 changes optimized_poses.pt bytes. Independent components but both anchored on Lane A. Stacking would be just running W on Lane M-V2's archive — predicted same as Lane W on Lane A (no Lane M-V2 contribution). |

**Verdict**: Lane M-V2 has **no positive composability**. The argument from `feedback_dont_abandon_high_score_lanes_for_stacking_20260428` ("raw score ≠ stack value") only applies when a lane has at LEAST ONE wedge component near or below the frontier. Lane M-V2 is worse than Lane A on every component (rate match, SegNet match, PoseNet 15× worse). It is **strictly dominated**.

The only lane in this category that V2 could theoretically improve is **Lane M-V3-clean** (i.e., its own successor), if the predicted [1.05, 1.20] band is achieved and that LANDED archive can be re-used as a Lane M anchor for further V4/V5 iterations.

---

## SECTION 5 — RE-ANCHOR PREDICTIONS (deliverable #5)

### 5.1 How many iterations to converge?

Per `feedback_dont_abandon_high_score_lanes_for_stacking_20260428` precedent:

| Lane | V1 → final | Iterations | Final score |
|---|---|---|---|
| Lane F | 2.73 → 1.79 | 2 | Killed at V2 (architectural ceiling found) |
| Lane G | killed → 1.05 | 3 | V3 success after KL-bug fix |
| Lane M+N | 2.35 → 1.84 (V2) → ?? | 2 so far | V3 below |
| Lane D | killed → in-flight | 3 | V3 in flight |

Lane M is on the trajectory of Lane F (architectural ceiling) more than Lane G (engineering bugs). The reason:
- Lane G's V1/V2 had a 14000× KL-weight bug — pure engineering fix → cleanly recovered
- Lane M's V1/V2 had save-shape + train/inference padding bugs — engineering fixes → V3 will likely recover from "zero-pad-at-inflate disaster" but NOT from the deeper architectural mismatch (renderer trained on 6-DOF, optimizer constrained to 1-DOF)

**Council prediction**:
- **Lane M-V3-clean** (BUG-1 fix): [1.05, 1.20] — first clean diagnostic
- **If V3 ≤ 1.10**: pursue V4 (lr sweep + LoRA composition). Predicted [1.00, 1.15]. Cost $0.50.
- **If V3 ∈ (1.10, 1.20)**: pursue Lane M+ (V3-C above; compress-time renderer fine-tune). Predicted [0.95, 1.30]. Cost $2.
- **If V3 > 1.20**: ARCHITECTURAL DEATH. Kill the lane. Inform downstream lanes (GE, LM-V2, MOS) that rank-1 priors should NOT be used as renderer-input constraints.

### 5.2 Total budget if pursued cleanly

| Step | Cost | Cumulative | Decision criterion |
|---|---|---|---|
| V3-clean | $0.30 | $0.30 | gate: ≤1.20 to continue |
| V4 (if V3 ≤ 1.10) | $0.50 | $0.80 | gate: ≤1.05 to continue |
| Lane M+ (V3-C, if V3 ∈ (1.10, 1.20)) | $2.00 | $2.30 | gate: ≤1.00 to continue |
| V5 (Lane M + KL-distill) | $1.00 | $3.30 | gate: ≤1.00 to keep in stack |

Total worst-case Lane M iteration budget: **$3.30** assuming each gate triggers continuation. Worst-case kill-at-V3: **$0.30**.

This is a **CHEAP** lane. The Council non-conservative principle ("burden of proof on NOT trying") favors V3-clean even with the Contrarian's "20% wedge" objection — $0.30 is below the noise floor of weekly Vast.ai spend.

### 5.3 Re-anchored portfolio impact

If Lane M-V3-clean lands at 1.10:
- New anchor for radial-zoom pose lane = 1.10
- Lane GE (geodesic Chebyshev poses, predicted [1.05, 1.20]): re-anchor to [1.10, 1.20]
- Lane LM-V2 (lane-mark endpoint tracking with rank-1 prior): re-anchor to [1.05, 1.15]
- Lane MOS (Lane M-V3 + Lane OS embedding): re-anchor to [1.00, 1.10] — first sub-1.0 candidate from Lane M family

If Lane M-V3-clean lands at 1.84+ (V2 or worse):
- Lane M family officially dead
- Lane GE/LM-V2/MOS need their rank-1 priors revisited; predicted bands tighten upward
- Council pivots all rank-1-related GPU back to Lane HF (Telescope, $4 with separate band [0.85, 1.10])

---

## SECTION 6 — FINAL COUNCIL VOTE

**Question**: Approve Lane M-V3-clean ($0.30, fix BUG-1, run diagnostic)?

| Council member | Vote | Rationale |
|---|---|---|
| Yousfi | YES | Hypothesis hasn't been falsified yet; cheap diagnostic. |
| Fridrich | YES | BUG-1 contaminated V2; clean test required before promoting Lane HF. |
| Hotz | YES (with caveat) | Skip V3-A and go straight to V3-C (Lane M+ compress-time fine-tune). Caveat: ok with V3-A first if council insists on cheapest test. |
| Quantizr | YES | $0.30 is below noise floor. Output informs Lane GE/LM-V2/MOS. |
| Contrarian | NO (DISSENT) | 20% wedge already at floor; this is procrastination. Spend $0.30 on Lane EC-V2 instead. |

**Tripartite verdict** (Yousfi + Fridrich + Contrarian): 2/3 APPROVE.

**Non-conservative charter resolution** (CLAUDE.md "Council conduct — non-negotiable"): the burden of proof is on NOT trying. Contrarian's argument is a wedge-priority argument, not a "this lane is dead" argument. Approved.

**Action**: dispatch Lane M-V3-clean as a $0.30/2h Vast.ai 4090 run, with the BUG-1 fix shipped first, regression test added, predicted band [1.05, 1.20] documented in `provenance.json`, decision criterion (1.20 cutoff) explicit in the script.

---

## SECTION 7 — TRANSFERABLE PATTERNS (process improvements)

### Pattern 1 — Train/inference distribution parity is a non-negotiable gate

This bug (BUG-1) is the same class as `feedback_eval_roundtrip_everywhere_FINAL`: any time the optimizer's forward path differs from the inference forward path, the optimizer is solving the wrong problem. Add a STRICT preflight check: any `_project_to_renderer_pose`-style helper used by both the optimizer and the save block must be the **same callable**, OR the test suite must verify byte-equality of their outputs on a representative input.

### Pattern 2 — Save-side tests are insufficient

`test_optimize_poses_radial_zoom_save_shape.py` has 9 tests pinning the save-side semantics, all passing on the V2 buggy code. Add a parallel test that pins the train-side semantics MATCHES the save-side. Without this, every "fix the save block" PR can silently re-introduce BUG-1.

### Pattern 3 — Predicted bands need empirical anchoring

V2's predicted band [1.10, 1.30] was magical thinking. The empirical precedent (V1=2.35, baseline=2.29) said the band should be [2.20, 2.50]. Going forward: predicted bands MUST cite the closest empirical precedent (within ±0.10 of an actual measurement) or be flagged "speculative".

### Pattern 4 — Sanity checks should be SEMANTIC, not SHAPE

The V2 stage-2 sanity check (line 139-149) verifies `shape[1] == 6`. A semantic check would also verify dims 1-5 match init_poses[:, 1:6]. Future archive-build sanity checks should verify the BYTES match expectations, not just the shape.

### Pattern 5 — Lane wedge attribution before iteration approval

Per `project_lane_g_v3_stacking_skunkworks_20260428` Pattern 3, the wedge attribution (PoseNet=20%, SegNet=43%, Rate=36% of gap to Quantizr 0.33) should be re-evaluated before approving each iteration. Lane M targets the SMALLEST wedge. Even if V3 lands at 1.05, its CONTRIBUTION to a stack is bounded by the 20% wedge. The Contrarian's argument (Section 6 NO vote) is structurally valid; the council overrode it on cheapness, not on payoff.

---

## SECTION 8 — REFERENCES

### Memories cross-referenced
- `project_lane_m_v2_landed_1_84_regression_20260428` (the result audited)
- `project_lane_mn_radial_zoom_negative_20260427` (V1 result)
- `project_posenet_rank1_discovery` (the foundational hypothesis)
- `project_lane_g_v3_landed_1_05_20260428` (current frontier)
- `project_cosmos_mae_lyra_telescope_synthesis_20260428` (Lane HF revival)
- `feedback_dont_abandon_high_score_lanes_for_stacking_20260428` (composition rule)
- `feedback_proxy_auth_math_useless` (proxy not predictive)
- `feedback_eval_roundtrip_everywhere_FINAL` (parallel rule for train-eval parity)
- `feedback_dead_flag_wiring_pattern` (parallel — verify against target, not intent)
- `project_baseline_poses_load_bearing` (joint pose-renderer artifact rule)

### Files audited
- `/Users/adpena/Projects/pact/scripts/remote_lane_m_v2_radial_zoom_proper.sh` (180 lines)
- `/Users/adpena/Projects/pact/experiments/optimize_poses.py` (2326 lines, focus on pose-mode + save block)
  - Line 752-770: `_project_to_renderer_pose` (BUG-1 site)
  - Line 783, 918, 948: optimization-time render call sites (BUG-1 propagation)
  - Line 821: `pose_loss = F.mse_loss(pose_out, pose_targets)` (DES-3 site)
  - Line 2118-2152: SAVE block (V2 fix site)
  - Line 2218-2236: proxy-score block (BUG-3 site)
- `/Users/adpena/Projects/pact/submissions/robust_current/inflate_renderer.py:2120-2160` (inflate render path)
- `/Users/adpena/Projects/pact/src/tac/tests/test_optimize_poses_radial_zoom_save_shape.py` (T1-T9 — pin SAVE-side only)
- `/Users/adpena/Projects/pact/experiments/results/lane_m_v2_landed/` (artifact)
- `/Users/adpena/Projects/pact/experiments/results/lane_a_landed/` (anchor)

### Empirical evidence collected
- Lane A optimized_poses.pt per-dim stats (dims 1-5 std up to 2.07, range -4.7 to +4.9)
- Lane M-V2 saved poses dims 1-5: byte-identical to Lane A's (zero diff)
- Lane M-V2 saved dim 0: mean=32.73, std=2.66 vs GT-target mean=31.30, std=1.26 (2.1× std inflation)
- contest_auth_eval.json sha256 verification (88cde3c8e0bc8348...)

---

*End of audit. Length: ~640 lines. Written by Claude (Opus 4.7 1M context) acting as the skunkworks council per CLAUDE.md "Council conduct — non-negotiable" + "Council decides design".*
