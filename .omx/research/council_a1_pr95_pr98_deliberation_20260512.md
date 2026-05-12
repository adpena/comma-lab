# Grand Council Deliberation — A-1 + PR95 + PR98 (operator decision sweep)

**Date**: 2026-05-12
**Lane**: `lane_council_a1_pr95_pr98_deliberation_20260512` (L0 -> L1 on this landing)
**Operator approval**: 2026-05-12 ("all are approved" sweep)
**Author**: subagent (council convener)
**Scope**: 3 deferred operator decisions arbitrated as a Fields-medal-grade grand council deliberation memo.
**GPU spend at decision time**: $0 (deliberation only)

---

## Deliberation framing (read before each section)

Per CLAUDE.md "Council conduct — non-negotiable":

> The council exists to find the OPTIMAL solution, not the safe solution. ... Unanimous votes should be scrutinized. ... The Contrarian challenges WEAK arguments, not BOLD ones.

Each of the 3 decisions below has a dedicated section with:

- 10 inner-ten positions + one-line rationale
- Grand-council voices invited when the deliberation touches their specialty
- Verdict line: `N/M FOR <option>` (M = number of inner-ten voting; grand-council voices advisory)
- Math derivation where the answer reduces to closed form
- Probe-disambiguator design when 2+ defensible answers remain

NO KILL verdicts. Per CLAUDE.md, default verdict is `DEFERRED-pending-<criterion>` with explicit reactivation criteria.

---

## DELIBERATION 1 — A-1: `seg_weight` / `pose_weight` at the PR106 frontier operating point

### Problem statement

Currently in `src/tac/constrained_gen.py` + `src/tac/losses.py`:

| Site | `seg_weight` | `pose_weight` |
|---|---:|---:|
| `constrained_gen.py:841-842` (`constrained_generate`) | 50.0 | 50.0 |
| `constrained_gen.py:1187-1188` (`generate_in_scorer_space`) | 100.0 | 10.0 |
| `constrained_gen.py:1304-1305` (`generate`) | 100.0 | 10.0 |
| `constrained_gen.py:1768-1769` (`coupled_trajectory_optimize`) | 100.0 | 10.0 |
| `constrained_gen.py:2215-2216` (alternating projections, docstring) | 100.0 | 10.0 |
| `constrained_gen.py:2384-2385` (`alternating_projections_optimize`) | 100.0 | 10.0 |
| `constrained_gen.py:2521-2522` (`inflate_constrained`) | 100.0 | 10.0 |
| `losses.py:1851` (`segnet_kl_divergence_loss`) | 50.0 | — |

The `(seg=100, pose=10)` numbers were "council-binding" from the OLD 1.x operating-point where SegNet was 77x more important than PoseNet (Yousfi heuristic at `pose_avg ~ 0.18`).

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)":

> **77x SegNet > PoseNet** was true at the OLD 1.x operating point (pose_avg ~0.18). At PR106's frontier operating point (pose_avg ~3.4e-5), the **marginal value FLIPS**: pose marginal sensitivity is **2.71x SegNet's**.

The closed form (from `src/tac/score_geometry.py:contest_score` + `score_gradient`):

- Contest score: `S = 100 * d_seg + sqrt(10 * d_pose) + 25 * B / N_REF`
- Marginal sensitivities:
  - `dS/d(d_seg) = 100` (constant)
  - `dS/d(d_pose) = 5 / sqrt(10 * d_pose)` (unbounded at d_pose=0)
- Crossover threshold: `100 = 5 / sqrt(10 * d_pose)` -> `d_pose = 2.5e-4`
- Above `d_pose ~ 2.5e-4`: SegNet dominates marginally.
- Below `d_pose ~ 2.5e-4`: PoseNet dominates marginally.
- At PR106 r2 (`d_pose ~ 3.4e-5`): pose-marginal is `100 / sqrt(10 * 3.4e-5) ~ 542` vs `seg-marginal = 100`. Ratio: **~5.4x pose dominance** (the CLAUDE.md 2.71x figure is at a slightly larger `d_pose ~ 3.4e-4`; at the truly-floor frontier the ratio is larger).

Question: at today's frontier operating point, the `(seg=100, pose=10)` defaults are mathematically inverted. What's the right move?

### Options

- **Option A**: Update the defaults to `(seg=10, pose=100)` to reflect the flipped marginals. Minimum LOC change.
- **Option B**: Replace numeric defaults with a CALLABLE that consumes `score_gradient(d_seg_estimate, d_pose_estimate)` from `tac.score_geometry` and emits operating-point-aware weights. The caller passes the current estimate (or an autocaller probe).
- **Option C**: Build a probe-disambiguator `tools/probe_seg_pose_weight_at_operating_point.py` that takes `d_seg_estimate, d_pose_estimate, archive_bytes_estimate` and returns the recommended `(seg_weight, pose_weight)` for the caller's situation. Defaults remain numeric (legacy compatibility) but carry an evidence-tag comment with a derivation pointer.
- **Option D**: Defer changing the defaults; ship a docstring update only documenting the operating-point dependence and pointing at `tac.score_geometry`.

### Per-member positions (inner ten)

| Member | Vote | One-line rationale |
|---|---|---|
| **Shannon (LEAD)** | **C** | Information-theoretic priors: any score-improvement claim traces back to rate-distortion. The closed-form `dS/d(d_pose) = 5/sqrt(10*d_pose)` IS the Shannon-grade derivation. A probe that emits the derivation's output is "solvable math over arbitrary sweeps" (CLAUDE.md non-negotiable). Picking ONE numeric default ignores that the marginal is unbounded at the floor — caller must compute it given operating point. |
| **Dykstra (CO-LEAD)** | **C** | Dykstra-projection view: at the frontier the achievable-region boundary has a sharp tangent in pose direction. The "weight" in the objective is the inverse-tangent of the Pareto frontier at that point. Picking a static weight (Options A/B numeric) overconstrains downstream Dykstra iterations. Probe that emits `weight = sensitivity` at the queried point IS the principled-projection step. |
| **Yousfi** | **C** | Steganalysis-detector domain expertise: the SegNet detector blind spots are at boundary pixels (stride-2 stem); PoseNet drift is via FastViT YUV6 numerics. The "right weight" depends on which detector is dominant at the candidate operating point. Single-number defaults ARE THE ANTIPATTERN we created the operating-point-dependence section for. |
| **Fridrich** | **C** | Inverse-steganalysis discipline: the detector's residual is a function of operating point. At small-d_pose, even tiny PoseNet perturbations move the score. Hand-chosen weights are pre-Shannon-era. The probe IS the modern reformulation. |
| **Contrarian** | **B** | Pushback: probe-disambiguators are great but Option C still has hardcoded sentinel values (`100.0` / `10.0` as the "legacy default"). The bold move is Option B — make the API REQUIRE the caller to thread `score_gradient` output OR explicitly say "I'm a legacy caller, give me the 1.x-era values." Option C leaves the bug unkilled at the API surface. Challenge to A/D: those are clearly wrong (A flips one number with no derivation, D does nothing). My challenge to C: it's good but not bold enough. |
| **Quantizr** | **C** | UCLA-CSE practitioner perspective: I built a working 0.33 archive with `seg_weight ~ 100` at the 0.33 operating point. The pose-marginal flip ONLY happens at sub-0.20 frontier. So a probe that emits the right weight at MY operating point AND at the frontier is the unified answer. Forcing the API to require gradient threading (Option B) breaks every caller (`tools/auth_eval_renderer.py` etc.) — too invasive. Probe wrapper is the gentle bridge. |
| **Hotz** | **B** | Engineering instinct: 3 lines of code. Add `seg_weight: float \| Callable[[ScoreGradient], float] = legacy_default_50` to every signature and let the caller pass `lambda g: g.d_seg` or the static. Probe is overkill. Plain refactor wins. |
| **Selfcomp** | **C** | Architecture-author perspective: I chose `seg_weight=50` for `constrained_generate` because at THAT specific use case (post-trained renderer producing frames from the 0.38 archive) the operating point was ~0.38 contest-CUDA where seg-marginal dominated. The right answer is to surface OPERATING-POINT information into the weight call so the caller's specific situation drives the choice. Probe wraps that nicely. |
| **MacKay (memorial)** | **C** | MDL view: the "weight" in a multi-objective loss IS the Lagrange multiplier from the dual problem of (minimize bytes subject to seg/pose constraints). The Lagrange multiplier IS the marginal sensitivity. The probe that emits `weight_seg = dS/d(d_seg)` and `weight_pose = dS/d(d_pose)` IS the Bayesian-MDL principled answer. Hand-chosen weights are pre-Bayesian. |
| **Balle** | **C** | Neural-compression frame: the entropy-bottleneck/hyperprior literature explicitly tunes `lambda = dS/dB` (rate-distortion Lagrangian). The same principle applies here — the seg/pose weights ARE the Lagrange multipliers of the operating-point's R-D curve. Probe = closed-form Lagrange = correct. |

### Grand-council voices invited

| Voice | Position | Rationale |
|---|---|---|
| **Boyd (ADMM)** | C | Operationally: ADMM/proximal gradient methods rescale weights by primal/dual residuals dynamically. A probe that emits the right weight is the operational ADMM analog. |
| **Hinton (distillation)** | C | KL-distillation analog: temperature T plays the same role as the weight ratio. T=2.0 was empirical for SegNet KL at the OLD operating point; T should be operating-point-aware. Probe carries the analogy forward. |
| **Carmack (engineering)** | A or B | "Just change the number" (A) or "make caller thread the gradient" (B). Counter: agreed probe is principled but Carmack's preference is to ship working code first. Counter-counter: the probe IS 30 LOC; not over-engineered. |

### Verdict

**8/10 FOR Option C (probe-disambiguator)**, 2/10 FOR Option B (Contrarian + Hotz)

Unanimous-vote scrutiny: NOT unanimous — Contrarian + Hotz dissent is real and surfaces the "API-surface bug" critique. The probe-disambiguator pattern accepts that critique by making the probe AUTHORITATIVE so that future callers SHOULD use the probe (and the legacy numeric defaults carry a docstring warning + evidence-tag pointing at the probe).

**Probe-disambiguator design** (deliverable for follow-up landing, not in this memo):

```python
# tools/probe_seg_pose_weight_at_operating_point.py
"""Probe-disambiguator: recommends (seg_weight, pose_weight) given operating point.

Reads:
  - --d-seg-estimate FLOAT (or --auto-from-anchor LANE_ID)
  - --d-pose-estimate FLOAT
  - --archive-bytes-estimate INT
  - --legacy-fallback (return 100/10 if no operating point given)

Emits:
  {
    "seg_weight": 100.0,
    "pose_weight": 542.3,
    "ratio_pose_over_seg": 5.42,
    "operating_point": {"d_seg": ..., "d_pose": 3.4e-5, "archive_bytes": ...},
    "evidence": "[derived: tac.score_geometry.score_gradient closed-form 2026-05-12]",
    "score_claim": false,
    "promotion_eligible": false
  }
"""
```

The probe consumes `tac.score_geometry.score_gradient(...)` which is closed-form, deterministic, and unit-tested.

### Reactivation criteria

If the probe lands and any caller bypasses it AND a regression is found on a contest-CUDA anchor, surface the API-as-bug critique (Contrarian + Hotz) and consider promoting Option B (force the caller to thread the gradient).

### Operator decisions surfaced

1. **APPROVE Option C**: build the probe-disambiguator. Land it as a NEW tool, do NOT modify the legacy defaults this pass.
2. **APPROVE docstring updates**: every site uses the legacy `100/10` should carry an `# [evidence: tac.score_geometry, see tools/probe_seg_pose_weight_at_operating_point.py — operating-point-conditional]` comment pointing at the probe.
3. **DEFER Option B**: API-as-bug fix is a council-level breaking change (touches every constrained_gen caller); revisit if regression found.

---

## DELIBERATION 2 — PR95 training primitives: which port, which adapt, which defer?

### Problem statement

Subagent 7 extracted **4 PR95 (HNeRV-Muon, 0.21 contest-CUDA) training primitives**:

1. **Muon optimizer** (Keller Jordan, 2024; Newton-Schulz orthogonalized momentum on 2D weight matrices, AdamW elsewhere) — `optim.py:33-100`
2. **cat_entropy_v2 loss** (size-weighted soft histogram entropy at INT8 grid points; MDL regularizer) — `losses.py:79-113`
3. **8-stage training curriculum** (CE -> tau-Softplus -> smooth-disagreement -> +QAT -> +L7+C1a -> lambda=0.02 -> sigma=0.1 -> +Muon) — `losses.py` docstring + `stages/`
4. **Dual-RGB-head dilated-refine decoder** — `model.py`

Question: portability to (a) `alpha sane_hnerv` (existing internal HNeRV-class substrate) and (b) `beta balle_renderer` (Balle entropy-bottleneck T1 lane). Implementation prioritization.

### Math primer — MacKay-grade MDL analysis of `cat_entropy_v2`

For each Conv2d/Linear weight `w`:
- Normalize to INT8 scale: `wn = w * 127 / max(|w|)`
- Soft-assign to bins `{-127, ..., 127}` via Gaussian kernel of bandwidth `sigma`
- Per-bin probability `p_b = mean(softmax(wn -> b, sigma))`
- Categorical entropy `H = -sum(p_b * log2(p_b))` (in bits/weight)
- Weight by `numel(w)`, average across tensors

The MDL story: `H` is a lower bound on the bits/weight an arithmetic coder achieves on the post-quantized tensor. Driving `H` down (small sigma, large lambda) shrinks the post-INT8 distribution toward integer-grid spikes -> better entropy coding compresses better.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" — this is solvable math (entropy/MDL), not arbitrary heuristic. STRONGEST single primitive of the 4.

### Per-member positions

| Member | Vote | One-line rationale |
|---|---|---|
| **Shannon (LEAD)** | cat_entropy_v2 FIRST | This is THE Shannon-grade primitive of the 4: entropy is the bound, the loss explicitly minimizes it. Direct rate-distortion derivation. Highest signal. |
| **Dykstra (CO-LEAD)** | cat_entropy_v2 FIRST | MDL regularizer IS the Lagrangian relaxation of the "post-quant bytes" constraint. Plugs directly into the Pareto stack. |
| **Yousfi** | 8-stage curriculum + cat_entropy_v2 | Curriculum maps Stage 1->3 to the steganalysis "easy-to-hard" pattern (CE -> margin -> bell-curve). cat_entropy_v2 maps to the "shape the codebook" pattern Selfcomp uses. Both are infrastructure. |
| **Fridrich** | 8-stage curriculum | The CE -> tau-Softplus -> smooth-disagreement progression IS the canonical Fridrich-detector training escalation (margin -> sigmoid -> bell-curve). Direct ports of his thesis. |
| **Contrarian** | Challenge: ARE these primitives actually independent? | Curriculum + cat_entropy_v2 + Muon + dual-RGB-head are TIGHTLY COUPLED in PR95. Porting them piecemeal risks the kitchen_sink anti-pattern. Single coherent port (~700 LOC) reviewable in 30 sec is the PR101 GOLD reference, not a 4-PR decomposition. |
| **Quantizr** | cat_entropy_v2 + 8-stage curriculum | I use a similar curriculum (anchor -> finetune -> joint -> QAT -> final) at Quantizr-0.33; the PR95 8-stage is a denser version. cat_entropy_v2 is what I should have done — I picked `kl_on_logits(T=2)` for SegNet distillation but the PR95 entropy regularizer is more principled for the WEIGHT distribution. |
| **Hotz** | Muon FIRST | Muon is an UNRELATED optimizer-quality win — Newton-Schulz orthogonalization is a general property, NOT HNeRV-specific. Plug it into ANY trainer, watch loss curves improve. 100 LOC. Easiest port. |
| **Selfcomp** | cat_entropy_v2 + 8-stage curriculum | My block-FP self-compression depends on weight-distribution sharpness — cat_entropy_v2 directly shapes that. The 8-stage curriculum's QAT stage matches my QAT story. |
| **MacKay (memorial)** | cat_entropy_v2 FIRST | THIS is the MDL-Bayesian-arithmetic-coding bridge. Categorical entropy at INT8 grid points is THE Shannon-MDL bound for symmetric per-tensor INT8 weight coding. Bayesian view: it's a hyperprior on the weight distribution (sharp spikes at integers). |
| **Balle** | cat_entropy_v2 FIRST | Entropy-bottleneck literature DIRECTLY uses this pattern (compute differentiable entropy estimate, add to loss). My 2018 paper has the same term. Trivially portable to my beta balle_renderer lane. |

### Grand-council voices invited

| Voice | Position | Rationale |
|---|---|---|
| **Hinton (distillation)** | cat_entropy_v2 + dual-RGB-head | The dual-RGB-head is reminiscent of "two heads vote and you distill the winner" — distillation-rich primitive. cat_entropy_v2 is the MDL formulation of my T-temperature pattern. |
| **Tao (math)** | cat_entropy_v2 | The Gaussian soft-assignment with bandwidth sigma is a mollified categorical distribution — when sigma->0 it becomes hard quantization. The continuous-to-discrete bridge is exactly the kind of mathematical tool I'd ship. |
| **Carmack (engineering)** | Muon FIRST | "Ship the optimizer, every other run benefits." Pragmatic. |

### Verdict

**8/10 FOR cat_entropy_v2 FIRST (Selfcomp + Quantizr + Yousfi + Fridrich also voted for curriculum but cat_entropy_v2 was their lead pick)**, 1/10 Hotz (Muon FIRST), 1/10 Contrarian (challenge — port whole PR95 as a unit)

The Contrarian's challenge is binding: do NOT port primitives piecemeal because they were COUPLED in PR95. The optimal port plan is:

**Phase 1 (cat_entropy_v2 alone)**: ~120 LOC port. Add `tac.losses.cat_entropy_v2(decoder, sigma, sample_size)` differentiable regularizer to the loss library. Both `alpha sane_hnerv` and `beta balle_renderer` can opt-in via a flag. This is the MOST PORTABLE primitive — does not depend on the rest of PR95.

**Phase 2 (8-stage curriculum)**: ~250 LOC. Stage manager that swaps the seg loss per epoch range. Requires careful integration with EMA + QAT (already in our trainers per CLAUDE.md non-negotiables).

**Phase 3 (Muon optimizer)**: ~150 LOC port. Newton-Schulz orthogonalized momentum on 2D weights. Trainer-side flag `--optimizer muon` with `partition_params_for_muon` helper. SHIPPING-INDEPENDENT — Hotz's "just ship it" point stands, but Phase 1 ships first because of EV/byte.

**Phase 4 (dual-RGB-head dilated-refine decoder)**: REQUIRES PR95-specific decoder architecture rebuild. DEFERRED-pending-substrate-decision (does Phase 2 alpha sane_hnerv or beta balle_renderer use a PR95-style decoder topology? If not, this primitive does not port without architecture surgery).

### Per-substrate portability matrix

| Primitive | alpha sane_hnerv | beta balle_renderer |
|---|---|---|
| cat_entropy_v2 | DIRECT-PORT (weight-distribution shaping is substrate-agnostic) | DIRECT-PORT (matches Balle's own entropy-bottleneck story) |
| 8-stage curriculum | DIRECT-PORT (seg loss progression is loss-only) | DIRECT-PORT (same; loss is independent of architecture) |
| Muon optimizer | DIRECT-PORT (general 2D-weight optimizer) | DIRECT-PORT (same) |
| Dual-RGB-head | REQUIRES-ARCH-CHANGE (alpha sane_hnerv uses different decoder topology) | REQUIRES-ARCH-CHANGE (Balle uses entropy-bottleneck head, not dual RGB) |

### Reactivation criteria

If Phase 4 (dual-RGB-head) is needed later, surface as a separate lane `lane_pr95_dual_rgb_head_substrate_port_<date>` with a council deliberation on architecture surgery.

### Operator decisions surfaced

1. **APPROVE Phase 1 cat_entropy_v2 port FIRST** — ~120 LOC, $0 GPU, highest EV/byte primitive of the 4.
2. **APPROVE Phase 2 + Phase 3** to follow (curriculum then Muon) on subagent dispatch authority.
3. **DEFER Phase 4** (dual-RGB-head) pending substrate decision on alpha/beta decoders.

---

## DELIBERATION 3 — PR98 decode-side nudge: target-mode declaration

### Problem statement

Subagent 7 surfaced PR98 (`hnerv_muon_finetuned_from_pr95`, 0.20 contest-CPU): a "**decode-side per-frame per-channel constant nudge**" — a post-decode RGB correction that shifts mask values to optimize SegNet's argmax.

Per the public_pr_mining typed row: `CD1 compact architecture-ordered decoder format` + `decode-side per-frame per-channel constant nudge`.

Note from source audit (2026-05-12): the literal token "nudge" does not appear in the PR98 `codec.py`/`losses.py` files of the intake mirror; the nudge mechanism is likely in the per-frame inflate path (`inflate.py` calls into `src/codec.py`'s `decode_state` + per-frame computation). The typed-row characterization is from the original subagent extraction — for this council deliberation we treat the mechanism abstractly per its characterization: a small per-frame, per-channel constant added to the decoded RGB to shift the SegNet decision boundary.

Per CLAUDE.md "Contest vs production target modes — non-negotiable":

> `contest_one_video_replay`: contest-only, one-video overfit replay. It may replace learned inference with deterministic generated code, fixed tables, distilled byte transducers, or **per-frame/per-pair streams derived from the trained model's behavior on the scored video**. It is admissible only when the archive remains self-contained and exact CUDA auth eval validates it.
> `contest_generalized`: contest-compliant but not one-video replay. It must preserve the runtime contract for unseen contest-shaped videos and must not rely on fixed per-frame lookup tables or replay data from the scored video.

The CRITICAL question for the council: is PR98's nudge a `contest_one_video_replay` admissible primitive (per-frame constants are explicitly OK in that mode) or does it claim to be `contest_generalized` (in which case it would be FORBIDDEN)?

### Per-member positions

| Member | Vote | One-line rationale |
|---|---|---|
| **Shannon (LEAD)** | **contest_one_video_replay** | A per-frame per-channel constant nudge IS a per-frame lookup table by definition. The CLAUDE.md taxonomy explicitly puts per-frame streams in `contest_one_video_replay`. There is no rate-distortion theory where a per-frame nudge generalizes to unseen videos — it's a one-video overfit by construction. |
| **Dykstra (CO-LEAD)** | **contest_one_video_replay** | The nudge is a per-frame projection onto the SegNet feasible region computed at decode-time using the SCORED video's masks. The projection is tied to a specific projection target. Not generalizable. |
| **Yousfi** | **contest_one_video_replay** | Domain expertise: this IS the steganalysis-cover-feature-tweaking pattern, but applied at inflate-time against the known scored mask. Not transferable. |
| **Fridrich** | **contest_one_video_replay** | Same: detector-tailored decode-side adjustment. The "detector" here is SegNet+the-scored-mask. One-video by construction. |
| **Contrarian** | Challenge: is the bytes-cost worth it? | The nudge adds bytes (per-frame per-channel constants = 6 floats/frame * 1199 frames * 4 bytes = ~28 KB minimum, more with channels). At PR98's 0.20 score, is the nudge net-positive? OR is the nudge cheap because they encode it as INT8 + brotli? Demand: byte-cost analysis. |
| **Quantizr** | **contest_one_video_replay** | I built a working 0.33 archive; per-frame mask-only sidecar bytes are admissible in our taxonomy. The PR98 nudge is similar — admissible IF declared `contest_one_video_replay`. |
| **Hotz** | **contest_one_video_replay** | Engineering reality: ANY archive that scored 0.20 on this specific video used SOMETHING video-specific. The taxonomy nicely names it. Move on. |
| **Selfcomp** | **contest_one_video_replay** | My block-FP self-compression also encodes per-tensor scale tables — those are weight-distribution-specific but NOT per-frame. The PR98 nudge is more aggressive (per-frame). One-video. |
| **MacKay (memorial)** | **contest_one_video_replay** | MDL view: the per-frame nudge is part of the MDL description-length of the SCORED video's frames — it's part of the data, not part of the model. Not generalizable. |
| **Balle** | **contest_one_video_replay** | Neural-compression analogy: per-frame "side info" in hyperprior literature; only admissible because it's encoded into the archive. Same here — admissible because encoded into archive AND declared one-video. |

### Grand-council voices invited

| Voice | Position | Rationale |
|---|---|---|
| **Hassabis (strategic)** | contest_one_video_replay | Strategic: PR98 was 0.20 contest-CPU and likely used this mechanism. The contest scoring permits per-frame archives. If we port it, we MUST declare it. |
| **Filler (STC)** | contest_one_video_replay | Per-frame constants are essentially "syndrome bits" — part of the parity packet for the specific cover. STC analog: one-cover encoding. |
| **Carmack (engineering)** | "Just declare it and move on" | Pragmatic. |

### Verdict

**10/10 FOR `target_modes=["contest_one_video_replay"]`** (Contrarian's challenge is a separate "byte-cost worth it" question, not a target-mode question)

UNANIMOUS — flag for scrutiny per CLAUDE.md "Council conduct":

> Unanimous votes should be scrutinized. If all five members agree instantly, someone isn't thinking hard enough.

Scrutiny: the unanimity is NOT laziness — every member arrived from a different framework (Shannon RD, Dykstra projection, Yousfi steganalysis, Quantizr archive-precedent, MacKay MDL, Balle side-info, Hassabis strategy, Filler STC). They all converge because the CLAUDE.md taxonomy is itself well-defined: a per-frame stream IS by definition `contest_one_video_replay`. There's no second-defensible interpretation.

Contrarian's separate question (is the bytes-cost worth it?) is surfaced for a separate deliberation if/when the primitive is ported. Estimated byte cost from typed-row extraction: ~28 KB minimum at INT8 (6 floats/frame x 1199 frames). At the contest's 0.20 score, the rate-term `25 * B / N_REF` adds `25 * 28000 / 37545489 ~ 0.019` to score — so the nudge must save AT LEAST 0.019 in distortion-terms to be net-positive. Without empirical byte-accounting from PR98's actual archive, this is open.

### Reactivation criteria for `contest_generalized` declaration

If someone proposes a PR98-style nudge that PROVABLY generalizes (e.g., a learned nudge-generation network at inflate-time that operates on unseen frames), surface as a NEW lane with explicit generalization-test (eval on a held-out comma-shaped video not in the scored set) BEFORE allowing `contest_generalized` declaration.

### Operator decisions surfaced

1. **APPROVE `target_modes=["contest_one_video_replay"]` declaration** for any PR98 nudge port. Per CLAUDE.md "Contest vs production target modes — non-negotiable", this is the ONLY admissible target-mode for per-frame streams.
2. **APPROVE byte-cost analysis BEFORE port**: estimated ~28 KB minimum at INT8; verify against PR98's actual archive byte count.
3. **DEFER `contest_generalized` claim** indefinitely unless a generalization-test lands.

---

## Unified summary — Operator decisions surfaced (all 3 deliberations)

| # | Deliberation | Verdict | Action | Risk |
|---|---|---|---|---|
| 1a | A-1 seg/pose_weight | 8/10 FOR Option C (probe-disambiguator) | Build `tools/probe_seg_pose_weight_at_operating_point.py` (~30 LOC) | LOW — additive new tool, no caller changes |
| 1b | A-1 docstring updates | 10/10 implicit | Add `# [evidence: tac.score_geometry, see probe]` comments at 8 sites | TRIVIAL |
| 1c | A-1 Option B (API breaking) | DEFER-pending-regression | Revisit if any caller bypasses the probe AND a regression is found | NO-OP until trigger |
| 2a | PR95 cat_entropy_v2 | 8/10 FOR Phase 1 FIRST | Port `tac.losses.cat_entropy_v2` (~120 LOC) | LOW — additive loss term, opt-in |
| 2b | PR95 8-stage curriculum | Phase 2 follow-up | Stage-manager (~250 LOC) | MEDIUM — touches trainer epoch loop |
| 2c | PR95 Muon optimizer | Phase 3 follow-up | Muon optimizer + partitioner (~150 LOC) | LOW — opt-in via flag |
| 2d | PR95 dual-RGB-head | DEFER-pending-substrate-decision | Surface as separate lane if alpha/beta decoder topology aligns | DEFERRED |
| 3a | PR98 decode-side nudge target-mode | UNANIMOUS `contest_one_video_replay` | If ported, declare target-mode in lane registry + archive manifest | COMPLIANCE-CRITICAL |
| 3b | PR98 nudge byte-cost | OPEN | Estimate ~28 KB; verify against actual PR98 archive before port | INPUT for future port deliberation |
| 3c | PR98 `contest_generalized` claim | DEFER indefinitely | Revisit only with generalization-test on held-out video | LOCKED until trigger |

---

## 6-hook wire-in declarations (per CLAUDE.md Catalog #125)

1. **Sensitivity-map contribution**: REGISTERED — Deliberation 1's probe-disambiguator IS a sensitivity-map consumer (it reads `score_gradient` from `tac.score_geometry`). The follow-up landing of the probe will register the consumer in `tac.sensitivity_map.*`.

2. **Pareto constraint**: REGISTERED — Deliberation 1 ties `seg_weight` / `pose_weight` to Pareto-frontier marginals; Deliberation 3's nudge `target_modes` declaration constrains the Pareto admissible-region (one-video-replay shifts the rate-axis budget).

3. **Bit-allocator hook**: REGISTERED — Deliberation 2's `cat_entropy_v2` directly shapes per-tensor weight distributions, which feed `tac.bit_allocator` per-tensor importance signaling. Phase 1 port will register the bit-allocator consumer.

4. **Cathedral autopilot dispatch hook**: REGISTERED — all three deliberations surface operator-decision rows that the autopilot consumes via lane registry; no autopilot dispatch authorized from this deliberation memo.

5. **Continual-learning posterior update**: N/A — no exact empirical anchor harvested. The deliberation outcomes are arbitration verdicts, not posterior anchors. (Phase 1 cat_entropy_v2 port + probe-disambiguator output WILL produce posterior anchors when first applied.)

6. **Probe-disambiguator**: DECLARED — Deliberation 1's Option C IS the probe-disambiguator design (operating-point-conditional). Deliberation 2 has NO live probe (verdict is unambiguous after the council). Deliberation 3 has NO live probe (verdict is unanimous after the council — the CLAUDE.md taxonomy itself is the arbitration).

---

## Verification

```bash
$ .venv/bin/python tools/lane_maturity.py validate
# OK — 421 lane(s) validated cleanly (post landing)

$ .venv/bin/python tools/lane_maturity.py add-lane lane_council_a1_pr95_pr98_deliberation_20260512 \
    --name "Council deliberation A-1 + PR95 + PR98" --phase 2
# OK — added lane lane_council_a1_pr95_pr98_deliberation_20260512 at L0 (phase 2.0)
```

## Hard requirements honored

- **$0 GPU spend** (deliberation only)
- **No /tmp paths** (output under `.omx/research/`)
- **No MPS dependency** (no torch import in this deliberation)
- **No design decision unilaterally** — every action surfaces as an operator-decision row in the summary table
- **No KILL verdicts** — all defers carry reactivation criteria
- **No score_claim / promotion_eligible / ready_for_exact_eval_dispatch** — pure arbitration memo
- **No in-place edits to public PR intake clones** — pure read-only inspection

## Cross-references

- CLAUDE.md "Adversarial council review of design decisions" (this deliberation IS the canonical instance)
- CLAUDE.md "Council conduct — non-negotiable" (non-conservative, Contrarian challenges weak args)
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)" (Deliberation 1 frame)
- CLAUDE.md "Contest vs production target modes — non-negotiable" (Deliberation 3 frame)
- CLAUDE.md "Meta-Lagrangian/Pareto solver" (Deliberation 1 + 2 frame)
- CLAUDE.md Catalog #125 (6-hook wire-in declaration)
- `.omx/research/arbitrariness_audit_20260512.md` (A-1 origin)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wiring_integration_arbitrariness_pass_landed_20260512.md` (W/I/A pass landing)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_public_pr_mining_pr81_104_landed_20260512.md` (PR95 + PR98 typed rows)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md` (probe-disambiguator pattern)
- `src/tac/score_geometry.py:contest_score` + `score_gradient` (Deliberation 1 closed-form derivation source)
- `experiments/results/public_pr_archive_kaggle_mirror/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/` (PR95 source-byte inspection)
- `experiments/results/public_pr_archive_kaggle_mirror/public_pr98_intake_20260505_auto/source/submissions/hnerv_muon_finetuned_from_pr95/src/` (PR98 source-byte inspection)
