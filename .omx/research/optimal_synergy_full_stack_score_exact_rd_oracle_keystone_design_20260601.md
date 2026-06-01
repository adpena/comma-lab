# Optimal synergy full-stack design — the score-exact differentiable RD oracle as keystone

- **Date:** 2026-06-01
- **Lane:** `lane_boundary_aware_rd_allocation_grammar_20260601`
- **horizon_class:** frontier_pursuit
- **axis_tag:** [predicted] (design); empirical anchors herein are [macOS-CPU advisory], NON-PROMOTABLE per Catalog #341/#192/#127/#323
- **Status:** v1 single-author deliberation, OPEN to T3 grand-council adversarial ratification. No score claim. Keystone feasibility VERIFIED (see §0); forward score effects are predicted/pending.

---

## 0. What is established fact (verified) vs design claim

Verified this session by `tools/verify_upstream_scorer_mirror_fidelity.py` against the **real frozen** `upstream/models/{segnet,posenet}.safetensors` on **real** `upstream/videos/0.mkv` frames (Catalog #213; NOT synthetic), committed `8173b493a`, ledger `.omx/research/upstream_scorer_anatomy_and_differentiable_mirror_audit_20260601T163112Z.md`:

- The tac differentiable mirror reproduces the upstream forward **bit-exactly on CPU**: `differentiable_rgb_to_yuv6`, SegNet logits, PoseNet pose (incl. first-6) all **max-abs-diff 0.0**; argmax disagreement 0.0.
- `s_seg` (DeepFool boundary saliency on real SegNet logits) is **boundary-peaked 20,816× over interior**.
- `s_seg` and `s_pose` are **exactly computable** against the real frozen weights today (finite, nonzero, spatially structured).
- Scorer constants (read from `upstream/`): `score = 100·d_seg + √(10·d_pose) + 25·(archive_bytes/N)`; `N = 37,545,489`; `λ = ∂S/∂byte = 25/N = 6.659e-7`; **1,501.82 bytes ↔ 0.001 score**. SegNet scores the **last frame only** (`modules.py:108`) as a 5-class argmax-flip rate at 384×512; PoseNet scores the **pair** (12-ch YUV6) as MSE on the **first 6 of 12** dims (`modules.py:84`). YUV6 = BT.601 full-range + 4:2:0.

Everything below labelled "predicted" or "design" is NOT yet measured. No claim is promoted without paired CPU+CUDA on exact archive bytes (Catalog #246).

---

## 1. Thesis — the most important part

**The keystone of a synergy-maximised, score-lowering full stack is the score-exact, eval-roundtrip-faithful, differentiable distortion oracle** — a bit-faithful mirror of `SegNet + PoseNet + frame_utils`. Its per-element gradient `∇S` is *simultaneously*:

1. the **training signal** (score-aware, kills the 2–350× proxy-auth gap),
2. the **bit-allocation saliency** (`sᵢ = ∂S/∂elementᵢ`; the reverse-waterfilling priority),
3. the **rate↔distortion coupling** (the single object both terms differentiate through), and
4. the **operational definition of the floor** (the task-aware information bottleneck: `∇S` identifies the minimal RGB content the scorer's decisions depend on).

Synergy = every component optimises ONE objective. The oracle *is* that objective made differentiable. Without it you have a pile of parts; with it, one variational problem.

## 2. The reduction (verified)

`upstream/evaluate.py:92` is a rate-distortion Lagrangian with a **fixed multiplier** `λ = 25/N` handed to us by the contest. Targets are a **decision** (`d_seg` argmax-flip, last frame) and a **6-dim regression** (`d_pose`), so the score depends on a tiny fraction of the RGB. We submit RGB, but the scored object is `Decision(SegNet(RGB))` and `Regress(PoseNet(RGB))`; the minimal sufficient statistic is the RGB content surviving into those decisions, and `∇_{RGB}S` computes exactly which content that is. This is a coding-for-machines problem; the oracle is the task bottleneck made computable.

## 3. Keystone identification — deliberation

Five candidates steelmanned; four collapse into "a consequence of `∇S`":

| Candidate | Why NOT the keystone |
|---|---|
| Renderer architecture (HNeRV/HPRC) | Lab had every arch pre-PR95, never broke 0.20; wins/loses on whether it's *bound* to the score-aware objective. Necessary, not sufficient. |
| Archive grammar / entropy coding | Saturated (PR103 AC ≈ 290 B ≈ 0.0002 over PR101). Lever is *fewer bits via decision-relevant encoding*, which needs `∇S`. Downstream. |
| Score-aware training | This *is* "follow `∇S`." A use of the oracle. |
| Boundary-aware bit allocation | The saliency *is* `∇S` per element. A use of the oracle. |
| The RD coupling | Correct framing — but its *enabler* is the differentiable score-exact oracle. |

This is the concrete instantiation of the lab's existing "Meta-Lagrangian/Pareto solver" + "unified Lagrangian action `S_total`" non-negotiables, and the mechanism behind the HNeRV-parity lesson "winners bound all ingredients simultaneously" (PR95 did it implicitly via a hand-tuned 8-stage curriculum; the keystone makes it explicit + differentiable so it is not luck).

## 4. Optimal design of the keystone — four subtleties

1. **Non-differentiable argmax-flip.** Train against a *calibrated differentiable surrogate* (DeepFool flip-risk `‖∇(z_k̂−z_k*)‖²/(z_k̂−z_k*)²`, a smooth majorizer of flip probability) — eval uses hard argmax, training uses the surrogate, surrogate validated as a majorizer so descent on it descends the real metric. **Verified computable** (GAP-1: compute from logits directly, not the soft training surrogate in `scorer_loss_terms_btchw`).
2. **Discrete rate.** Differentiable proxy `R(θ) = Σ −log₂ pₑ(symbolᵢ)` under the **same entropy model the grammar actually uses** (closes the L1↔L4 loop); proxy-vs-actual-bytes gap must be measured + bounded.
3. **Eval-roundtrip faithfulness.** Forward must simulate bilinear resize → BT.601 YUV6 → uint8 quantisation → last-frame slice → first-6 pose dims, with the differentiable `rgb_to_yuv6` (the upstream `@torch.no_grad()` + in-place `clamp_` sever grads). **Verified bit-exact** (§0); GAP-2: the eval-roundtrip sim is currently training-time, must be composed before the scorer for deployment-faithful saliency.
4. **Verification.** Bit-exact mirror cross-check (**done, §0, CPU**); GAP-4 CUDA low-order bits + GAP-5 full-video 600-pair aggregation pending; paired CPU+CUDA on exact bytes is the only score authority.
   Plus the operating-point coupling: pose weight `5/√(10·d_pose)` rises as `d_pose` falls (≈271 vs seg's 100 at frontier `d_pose≈3.4e-5`), re-scaled each iteration.

## 5. The full stack — one objective `L(θ)=100·d̃_seg+√(10·d̃_pose)+25·R/N`, seven layers

- **L0 Keystone** — the verified score-exact differentiable oracle; emits `d̃_seg, d̃_pose, R` + gradients.
- **L1 Objective** — the unified RD Lagrangian with the contest-fixed `λ=25/N`; the only objective anywhere.
- **L2 Representation** — score-aware full-RGB renderer (HPRC), per-pair latent; eval-roundtrip-aware (QAT); coder-aware-regularised; exploits the asymmetry (frame_1 latent dims carry seg+pose, frame_0 pose-only).
- **L3 Allocation** — reverse-waterfilling at the fixed water level via the existing `joint_p18_p19_waterfill` KKT-Dykstra allocator, fed the oracle gradient (replacing its static boundary mask). GAP-3: push pixel-space `∇S` into the latent/coeff domain via the substrate synthesis adjoint.
- **L4 Grammar** — entropy coder matched to the trained symbol distribution; monolithic `0.bin`, ≤100-LOC inflate, ≤2 deps; **its entropy model IS `R(θ)` in L1** (loop closed).
- **L5 Runtime** — bit-exact `inflate.sh`, CPU/CUDA-agnostic ⇒ optimiser's d_seg/d_pose == contest's.
- **L6 Truth** — paired CPU+CUDA auth-eval on exact archive bytes (Catalog #246); the only score claim.

## 6. Synergy ledger — super-additive couplings (the "maximum synergy")

1. Training × allocation — budget-aware training yields quantisation-robust + low-entropy latents (joint > sequential).
2. Decision-robustness × quantisation — margin-robust boundaries so quant noise does not flip the argmax (impossible under L2).
3. Frame/pair asymmetry × shared latent — allocate latent dims by which-frame-which-term they serve.
4. Fixed water level × task bottleneck — provably spend rate only on score-relevant info; *this is the rate↔distortion decoupling that has blocked competitiveness*.

## Canonical-vs-unique decision per layer

- L0 oracle: **ADOPT_CANONICAL** — reuse verified `differentiable_rgb_to_yuv6` + `load_differentiable_scorers`; FORK only the saliency read (compute `s_seg` from logits per GAP-1).
- L1 objective: **FORK_PRINCIPLED** — single contest-exact Lagrangian replaces per-substrate proxy losses.
- L2 renderer: **ADOPT_CANONICAL** (HPRC) for architecture; **FORK_PRINCIPLED** for the latent-dim/frame asymmetry exploitation (substrate-specific, score-driven).
- L3 allocator: **ADOPT_CANONICAL** — `joint_p18_p19_waterfill` exists and carries the exact weight formula; FORK the saliency input (oracle gradient vs static mask).
- L4 grammar: **ADOPT_CANONICAL** byte-format (near-saturated); the rate-model-equals-grammar-model closure is **FORK_PRINCIPLED**.
- L5/L6 runtime+truth: **ADOPT_CANONICAL** (contest contract + Catalog #246).

## 9-dimension success checklist evidence

UNIQUENESS — the explicit differentiable contest-exact RD objective with the contest-fixed water level is a class-shift from proxy-trained substrates. BEAUTY — one objective, seven thin layers, each a term. DISTINCTNESS — vs perceptual/L2 codecs: optimises the *decision*, not fidelity. RIGOR — keystone bit-exactness verified against real weights (§0); surrogate-majorizer + proxy-rate gaps named as gates. OPTIMIZATION-PER-TECHNIQUE — DeepFool seg saliency + PoseNet-Fisher pose saliency are the per-technique optima per the two research streams. STACK-OF-STACKS-COMPOSABILITY — every layer composes through `L`. DETERMINISTIC-REPRODUCIBILITY — bit-exact decode (L5). EXTREME-OPTIMIZATION — reverse-waterfilling at the contest-fixed λ is the RD-optimal allocation. OPTIMAL-MINIMAL-CONTEST-SCORE — the objective IS the contest score (no proxy).

## Cargo-cult audit per assumption

- "Better entropy coding lowers score" — CARGO-CULTED at the frontier (PR103 ≈ 0.0002). Unwound: the lever is fewer score-relevant bits, not better coding.
- "Reconstruct the image well" — CARGO-CULTED (perceptual fidelity). HARD-EARNED replacement: reconstruct only the scorer's *decisions* (20,816× boundary concentration confirms it empirically).
- "A proxy loss is good enough" — CARGO-CULTED (2–350× proxy-auth gap). Unwound: train on the verified score-exact oracle.
- "Saliency is one map" — CARGO-CULTED. Unwound: two structurally different fields (seg=last-frame argmax-flip; pose=pair Fisher-Jacobian), combined at exact score-derivative weights.

## Observability surface

Inspectable per layer (each layer exposes its term of `L` + its gradient). Decomposable per signal (d_seg / d_pose / rate separately; per-pixel s_seg, s_pose maps). Diff-able across runs (bit-exact mirror enables run-to-run logit/pose diff). Queryable post-hoc (saliency maps + dead-zone fraction as artifacts). Cite-able (every saliency anchored to commit + frozen-weight sha + frame index). Counterfactual-able (byte-mutation smoke per Catalog #139/#272 on the allocated bits).

## Predicted ΔS band + Dykstra-feasibility

Predicted direction: NEGATIVE (score-lowering) via rate↔distortion decoupling; magnitude PENDING full-video empirical (GAP-5). Dykstra-feasibility: the achievable RD point is the projection onto the constraint intersection {rate ≤ R, seg-protect, pose-null}, computed by the existing `joint_p18_p19_waterfill` KKT-Dykstra alternating-projections solve (Catalog #296 satisfied natively); first-principles bound = Shannon R(D) for the task bottleneck. No numeric ΔS asserted until the full-video dead-zone-fraction diagnostic + paired CPU+CUDA land. `# PREDICTED_BAND_VIBES_OK:band-deferred-to-full-video-empirical-feasibility-grounded-in-existing-KKT-Dykstra-allocator-and-Shannon-RD`

## Canonical equation reference

`# FORMALIZATION_PENDING: score_exact_rd_oracle_keystone_lagrangian — to be registered in tac.canonical_equations after the full-video dead-zone-fraction empirical anchor lands; the equation S=100·d_seg+√(10·d_pose)+25·R/N with λ=25/N=6.659e-7 is verified-constant, the achievable-point model is pending empirical.`

## 6-hook wire-in declaration (Catalog #125)

1. Sensitivity-map: ACTIVE — `∇S` per element IS the canonical sensitivity. 2. Pareto constraint: ACTIVE — the {rate,seg,pose} polytope via joint_p18_p19 KKT-Dykstra. 3. Bit-allocator: ACTIVE — reverse-waterfilling at λ=25/N. 4. Cathedral autopilot: ACTIVE — the oracle gradient feeds candidate ranking. 5. Continual-learning posterior: ACTIVE — full-video anchors recalibrate the achievable-point model. 6. Probe-disambiguator: ACTIVE — the dead-zone-fraction diagnostic disambiguates saturated-vs-headroom per substrate.

## Risks / gates

Mirror-fidelity (PASSED CPU; CUDA pending GAP-4). Surrogate-majorizer calibration (gate before dispatch). Proxy-rate gap (measure vs actual bytes). GAP-3 latent-domain adjoint (substrate-specific). No score claim without paired CPU+CUDA (Catalog #246).
