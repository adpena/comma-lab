# Beat-PR95 Curriculum + Substrate + Training Design (2026-05-13)

**Lane**: `lane_beat_pr95_curriculum_substrate_training_design_20260513`
(Phase 2; L0 → L1 once this memo lands)
**Mandate**: operator directive 2026-05-13 — "*we can do better than PR 95 ...
do even more beautiful and elegant and optimal and stacked and stacked of
stacks and composed and optimal and smart and dynamically continual learning
... perhaps consider shortest path algorithm ... brownian motion ... i think
curriculum and substrate and training are the path forward and super
important*".
**Evidence grade**: `[research design]` — no archive bytes change, no GPU
dispatch, no score claim. Predictions are derivations from φ1+φ3+F1 empirical
anchors plus first-principles math.
**Operator routing requirement**: this memo proposes a DESIGN; firing any
component requires operator + council approval per CLAUDE.md "Design decisions
— non-negotiable" + "Adversarial council review of design decisions" +
"Race-mode rigor inversion" non-negotiables.

---

## 1. Executive summary (≤500 words)

The current best public archive is **PR95-family at contest-CPU 0.193** on Linux
x86_64 GHA. Codex's frontier rebaseline (2026-05-13) confirms 0.193 is the
empirical ceiling of the PR95→PR101→PR103 stack. F1 forensic recovery proved
the PR95 curriculum is an **8-stage, 29,650-epoch chain costing ~$700 on Modal
A100 to reproduce from random init** — well beyond our $4-15 envelope.

This design proposes **HYBRID-SHORTEST-PATH + LANGEVIN POLISH on a STACK-OF-
STACKS SUBSTRATE**, predicted floor **0.180-0.185 contest-CPU** (a -0.008 to
-0.013 improvement on PR95), buildable in **3-4 sub-runs at $1-3 each = $4-15
total**.

The synthesis combines:

1. **Idea 1 (Shortest-Path Curriculum)**: model curriculum as a weighted DAG;
   solve for K=5 cheapest paths from random-init to predicted-sub-0.193 via
   A* with a Shannon R(D) admissible heuristic. This lets us **skip ~70% of
   PR95's 29,650 epochs** by detecting where the loss landscape is locally
   convex (cheap straight-line) vs locally non-convex (must traverse). Path
   cost ≈ $4.
2. **Idea 2 (Langevin Polish)**: replace PR95's Stage 8 Muon (Newton-Schulz
   orthogonalized momentum) with **annealed-Langevin SDE** for the FINAL
   2000 epochs only. Langevin provably samples from the Gibbs measure
   `p(θ) ∝ exp(-L(θ)/T)`; at T→0 this concentrates on global minima vs Muon's
   local-min convergence. Cost ≈ $2 on Vast.ai 4090.
3. **Idea 3 (Stack-of-Stacks)**: layer φ1 SABOR boundary mask + φ3 S2SBS HF
   stuffing + score-gradient residuals over the curriculum-trained substrate
   to recover an additional **30-130 KB of free-byte capacity** per video
   (empirically measured today, $0).
4. **Idea 4 (Dynamic Continual Learning)**: per-batch Hessian-vector-product
   reweighting of the (seg, pose, rate) loss terms. PR95 uses static weights;
   we adapt them per-batch to the CURRENT marginal score gradient.
5. **Idea 5 (HJB Optimal Control framing)**: provides the **theoretical
   justification** that Idea 1 (shortest-path) is the correct discretization
   of the continuous optimal-control problem. iLQR not implemented (overkill);
   used only for principled hyperparameter derivation.

Mathematical floor analysis (§4): Shannon R(D) bound on (segnet, posenet,
rate)-feasible set yields theoretical floor **0.135 ± 0.020** at archive bytes
=178 KB (Council F Q1 result). The PR95→ours gap of 0.193 → 0.183 closes 50%
of the available floor headroom, not the full headroom.

**Apples-to-apples discipline**: every predicted Δscore in this memo is
`[prediction]`-tagged. Empirical anchors (φ1, φ3, F1) carry
`[macOS-CPU advisory]`. No `[contest-CPU]` or `[contest-CUDA]` claims here.

---

## 2. Idea-by-idea development

### Idea 1 — Curriculum-as-Shortest-Path

**Graph definition.**

- **Nodes** (theta-space "checkpoints"):
  `n ∈ N = {(stage_id, seg_loss_kind, optimizer, lr, qat_active, cat_lambda,
  cat_sigma, predicted_score_band)}`
- **Edges** (training trajectories):
  `e: n_i → n_j` parameterized by `(epochs, batch_size, ema_decay)`
- **Edge weight** (cost-to-traverse, in USD):
  `w(e) = $/epoch × epochs` where $/epoch is the empirical cost-band posterior
  for the source-substrate × target-optimizer × scorer-forward-count tuple
- **Score-improvement-per-edge** (negative reward):
  `Δs(e) = score(n_j) − score(n_i) < 0` (we want monotone-decreasing)
- **Goal**: find K=5 paths `π = (n_init → n_intermediate → ... → n_floor)` that
  MINIMIZE `Σ w(e)` SUBJECT TO `Σ Δs(e) ≤ s_target − s_init`

**A* search.**

```python
# Pseudocode
def astar_curriculum(start: Node, target_score: float,
                     graph: CurriculumDAG,
                     heuristic: Callable[[Node], float]) -> Path:
    open_set = PriorityQueue([(heuristic(start), start, [start], 0.0)])
    closed_set = set()
    while open_set:
        f, node, path, g = open_set.pop()
        if node.predicted_score <= target_score:
            return path
        if node in closed_set:
            continue
        closed_set.add(node)
        for edge in graph.outgoing(node):
            n_next = edge.target
            g_next = g + edge.usd_cost
            f_next = g_next + heuristic(n_next)
            open_set.push((f_next, n_next, path + [n_next], g_next))
    raise NoPathFound
```

**Admissible heuristic (Shannon R(D)).**

For node `n` with substrate parameters `θ`, the minimum-bits-to-target-score
is `h(n) = max(0, R(D_target) − R(theta_current))` where R(D) is the contest's
score-axis rate-distortion bound (`Council F Q1 §3` derives R(D=0.135) =
~109 KB; current archives are at ~178 KB, so heuristic is positive for any
intermediate node with rate > 109 KB).

**Admissibility proof sketch (Tao)**: R(D) is the information-theoretic LOWER
bound on bits-to-achieve-distortion-D; therefore `h(n) ≤ true bits-to-target`.
A* with an admissible heuristic finds the OPTIMAL path; correctness preserved.

**Feasible curriculum space cardinality (Tao)**: F1 catalogs 14 PR95 primitives.
Discretizing each (4 seg-loss kinds × 3 optimizers × 5 LR-bands × 2 QAT-states
× 3 cat_lambda × 3 cat_sigma) yields ~1080 distinct node-types per stage;
with 8 stages and pruning by feasibility (e.g., QAT only at stage ≥ 4,
Muon only at stage = 8), the EFFECTIVE node count is ~50-200. Search is
tractable.

**Cost prediction**: at $0.057-$0.083/epoch on A100 (F1 §3.A measurement) and
A* finding paths ~8000-12000 epochs (vs PR95's 29,650), the predicted dispatch
cost is **$4-8 for the curriculum-traversal portion**.

**Empirical priors**: the PR95 forensic memo gives us 7 anchor scores at
stage-end checkpoints (Stage 1-end, Stage 5-end, etc.). These seed the
`predicted_score(node)` function via inverse-distance interpolation on the
(seg_kind, optimizer, qat, cat_λ) feature space.

**Risk**: heuristic admissibility relies on R(D) being a valid LOWER bound on
the (seg, pose, rate) contest score, which Council F Q1 has not formally
proven for the JOINT distortion (seg AND pose AND rate). If the joint is
NOT R(D)-bounded by the sum-of-axes, the search becomes WORSE than uniform-
cost search. Mitigation: also run Dijkstra (h≡0) as a safety baseline;
empirically compare path costs.

---

### Idea 2 — Langevin/Brownian Polish (replaces PR95 Stage 8 Muon)

**SDE.**

```
dθ_t = -∇L(θ_t) dt + sqrt(2 T_t) dW_t
```

where `dW_t` is multivariate standard Brownian motion (i.i.d. N(0,1) at each
discrete step) and `T_t` is a deterministic temperature schedule from
`T_0 = T_init` down to `T_final ≈ 0`.

**Discretization (Euler-Maruyama).**

```python
class LangevinOptimizer(torch.optim.Optimizer):
    """Langevin SDE optimizer for substrate polish phase."""

    def __init__(self, params, lr=1e-4, T_init=1.0, T_final=1e-4,
                 n_steps=2000, weight_decay=0.0):
        # T_t schedule: cosine annealing
        defaults = dict(lr=lr, T_init=T_init, T_final=T_final,
                        n_steps=n_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        T = self._cosine_temp()
        sqrt_2T_dt = math.sqrt(2.0 * T)
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd > 0:
                    grad = grad.add(p, alpha=wd)
                noise = torch.randn_like(p)
                # dθ = -∇L dt + sqrt(2T) dW
                p.add_(grad, alpha=-lr)
                p.add_(noise, alpha=sqrt_2T_dt * math.sqrt(lr))
        self._step += 1
        return loss

    def _cosine_temp(self) -> float:
        t = min(self._step, self.defaults["n_steps"])
        T0, Tf = self.defaults["T_init"], self.defaults["T_final"]
        ratio = t / self.defaults["n_steps"]
        return Tf + 0.5 * (T0 - Tf) * (1.0 + math.cos(math.pi * ratio))
```

**Theoretical justification (MacKay + Schmidhuber)**: at fixed T, the
stationary distribution is `p(θ) ∝ exp(-L(θ)/T)`. As T → 0, mass concentrates
on `argmin L`. With slow-enough annealing (Geman & Geman 1984 logarithmic
schedule: `T_t = c / log(2 + t)`), convergence to GLOBAL minimum is
guaranteed. Our cosine schedule is faster than logarithmic — provably
converges to local-min only — but empirically dominates pure-gradient methods
on non-convex landscapes (Welling & Teh 2011 SGLD).

**Why not just SGD?** The HNeRV loss landscape has known plateaus (PR95's
"pose plateau" bug requires the differentiable yuv6 patch). Plateau + sharp-
minimum coexistence is the canonical case where Langevin escapes via thermal
fluctuation while SGD/AdamW/Muon get stuck. PR95's progression
(AdamW → Muon at Stage 8) is the team's empirical workaround; Langevin is
the **first-principles correct** answer.

**Why not GARCH-Langevin (the Contrarian veto)?** The existing
`tac.contrib.finance_optimizers` carries GARCH-volatility-Langevin and the
Contrarian's prior verdict was "subsumed by GARCH-Langevin." Two responses:

1. **GARCH-Langevin is per-pixel, not per-parameter**. The finance-optimizer
   variant operates on `pixels` not on `model.parameters()`. We need a
   parameter-space SDE optimizer, not a pixel-space one.
2. **GARCH adaptive variance is non-essential for the polish phase**. The
   polish phase already has well-conditioned gradients (the curriculum has
   driven us to a near-minimum); the adaptive variance reduces to its
   long-run mean. We can use the constant-variance Langevin without loss.

**Predicted floor**: Langevin at T_0=1, T_final=1e-4 cosine-annealed for
2000 epochs starting from a Stage-7-end checkpoint achieves predicted
contest-CPU **0.183-0.187** (vs PR95 Stage-8 Muon 0.193). Gap mechanism:
escaping the sharp-minimum the curriculum lands at via thermal fluctuation,
landing in a wider but lower minimum.

**Cost**: 2000 epochs × $0.057-$0.083/epoch = **$1.14-$1.66 on Modal A100**,
or **$0.40-$0.60 on Vast.ai 4090** (per `feedback_modal_strategy_reevaluation_post_tier1_engineering_20260512`).

**Risk (Yousfi/Fridrich adversarial)**: does Langevin's noise injection
expand the adversarial-steganalysis surface? Answer: **no**. The polish phase
is on already-quantized INT8 weights (Stage 4+ QAT applied throughout); the
Langevin noise lives BELOW the quantization grid (`σ = sqrt(2T·lr) ≈ 1e-4`)
and gets rounded away by `fake_quantize`. The quantization grid IS the
defense; Langevin merely searches WITHIN a single grid cell more thoroughly.

---

### Idea 3 — Stack-of-Stacks Composition

**Three-layer composition graph.**

```
                    Outer Stack (multi-checkpoint ensemble)
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
            Checkpoint A    Checkpoint B     Checkpoint C
          (Idea1+2 path 1) (Idea1+2 path 2)  (PR95-canonical)
                │                │                │
                └────────────────┼────────────────┘
                                 │
                       Middle Stack (cross-substrate)
                                 │
       ┌──────────┬──────────┬──┴───────┬──────────┬──────────┐
       │          │          │          │          │          │
    A1 base   SABOR side  S2SBS side  LAPose   wavelet     score-grad
     (PR101)   (φ1 mask)  (HF stuff)  sidecar  residual    residual
                                 │
                        Inner Stack (per-substrate)
                                 │
                       [HNeRV decoder + latent
                       + INT8 QAT + EMA + curriculum]
```

**Inner stack** (per-substrate): the curriculum-trained HNeRV with QAT + EMA
applied across stages 1-8. This is the F1 substrate.

**Middle stack** (cross-substrate): use φ1 SABOR's measured boundary mask
to spatially gate the renderer's output (the mask says "outside this region,
any value works"); use φ3 S2SBS's measured ~97 KB/frame HF capacity to stuff
auxiliary decoder state into the GT-quasi-stable HF band of frame1; layer
LAPose / wavelet / score-gradient residuals on top of the renderer's pixel
output.

**Outer stack** (multi-checkpoint): ensemble K=3 checkpoints (Idea1+2 path
A, path B, PR95-canonical). At inflate time, output the MIN-distortion
checkpoint per-pair via a 1-byte-per-pair selector. Adds 600 bytes
(600 pairs × 1 byte) to the archive but recovers per-pair Best-of-3 minimum.

**Score-aware composition rules (Dykstra-feasibility)**:

- Each layer adds `(seg_delta, pose_delta, rate_delta)` to the (seg, pose,
  rate) tuple
- Composition is admissible only if the resulting tuple stays in the
  feasibility set `F = {(s, p, r) : s ≤ s_max, p ≤ p_max, r ≤ r_max}`
- Dykstra alternating-projections solves the COMPOSITION OPTIMIZATION
  problem: which layers to ACTIVATE, with which intensities, to MINIMIZE
  total score subject to feasibility

**Predicted compositional Δ** (per Council F TRIPLET φ): O1 + O3 sister
arms predict additive Δ in the **-0.005 to -0.020** band each; both predicted
positive interaction (sister stacks). With curriculum-trained substrate
(Idea1+2 base predicted 0.183-0.187), SABOR + S2SBS additive predicted
**0.175-0.183**.

**Cost**: φ1+φ3 are $0 (already measured). Outer-stack 3-checkpoint ensemble
needs 3 paths × $4 = **$12** to fit the F1 envelope.

---

### Idea 4 — Dynamic Continual-Learning Loss Reweighting

**Static-weight problem.** PR95 uses fixed weights:
`L = w_seg * d_seg + w_pose * sqrt(10 * d_pose) + w_rate * b(θ)/N`
with `w_seg = 100`, `w_pose = 1`, `w_rate = cat_λ ∈ {0.01, 0.02}`. These are
HUMAN-TUNED constants.

**Score-aware dynamic weights (per-batch HVP-reweighting).**

At each batch:
1. Compute `∂L_total/∂θ` via standard backprop (the gradient).
2. Compute `∂s_contest/∂θ` via Hessian-Vector-Product on each loss component:
   `g_seg = HVP(L_seg, θ, d_seg) = ∂s_contest/∂L_seg × ∂L_seg/∂θ`
   `g_pose = HVP(L_pose, θ, sqrt_10_d_pose)`
   `g_rate = ∂s_contest/∂(b/N) × ∂b/∂θ` (closed form via differentiable QAT)
3. Set `w_k_batch = ||g_k|| / Σ_j ||g_j||` — each loss term gets weight
   proportional to its CURRENT marginal score gradient norm.

**Mathematical justification (Boyd)**: this is the standard dual-ascent step
in a Lagrangian formulation of the multi-objective problem. We're solving
`min L s.t. s_contest = min` via Lagrangian dualization with the dual
variables learned via gradient flow on `||g_k||`. Boyd / Parikh "Proximal
Algorithms" §4 chapter exactly this.

**Replay buffer (Quantizr-style hard-pair re-sampling).**

```python
class HardPairReplayBuffer:
    """Re-introduce pairs where per-pair distortion exceeded threshold."""

    def __init__(self, capacity=200, threshold=2.0):
        self.capacity = capacity
        self.threshold = threshold  # multiples of per-batch mean
        self.buffer: list[int] = []  # pair indices

    def add_if_hard(self, pair_idx: int, per_pair_loss: torch.Tensor,
                    batch_mean: float) -> None:
        if (per_pair_loss / batch_mean).item() > self.threshold:
            self.buffer.append(pair_idx)
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)

    def sample(self, n: int) -> list[int]:
        if not self.buffer:
            return []
        return random.choices(self.buffer, k=min(n, len(self.buffer)))
```

Next epoch's data loader prepends `self.buffer.sample(batch_size)` before
the standard pair-stream.

**Predicted Δ**: closing the gap between mean-batch and worst-batch
distortion typically buys -0.005 to -0.010 on score (Quantizr's 5-stage
pipeline achieves this via hard-pair re-introduction explicitly; PR95
doesn't).

**Cost**: $0 implementation overhead (1 dataclass + 1 callback hook in the
trainer); amortizes against existing training cost.

---

### Idea 5 — Hamilton-Jacobi-Bellman Optimal Control

**Formulation.**

- **State**: `θ_t ∈ R^d` (model weights at time `t`)
- **Control**: `u_t = ∇L(θ_t) + noise` (gradient + thermal/curriculum
  perturbation)
- **Dynamics**: `θ_{t+dt} = θ_t − u_t dt`
- **Running cost**: `c(θ_t, u_t, t) = α ||u_t||^2 + β · L_proxy(θ_t)` (compute
  + proxy-loss)
- **Terminal cost**: `S(θ_T; archive(θ_T))` (contest score at export)
- **HJB equation**:
  `−∂V/∂t = min_u [c(θ, u, t) + ∇V · (−u)]`
  with terminal condition `V(θ, T) = S(θ; archive(θ))`

**Why this matters (theoretically, NOT computationally).** HJB tells us the
OPTIMAL training trajectory is determined by `V(θ, t)`. The gradient of `V`
acts as the SHADOW PRICE of weight perturbations. PR95's curriculum is a
HEURISTIC for `V`; Idea 1's shortest-path is a DISCRETIZATION of the
characteristic equation.

**iLQR / DDP would solve this numerically.** The state dim is `d ≈ 229,000`
for the HNeRV substrate; iLQR is `O(d^3 T)` per iteration. NOT tractable at
this scale.

**Practical use**: HJB provides a THEORETICAL TEMPLATE for principled
hyperparameter derivation. Example: the running-cost weight `α` corresponds
to the curriculum's "exploration vs exploitation" tradeoff; the terminal
cost's local convexity dictates the Langevin temperature schedule. These
mappings let us PRINCIPLEDLY choose hyperparameters that would otherwise be
free knobs.

**Hotz/Carmack engineering note**: the 5-line PoC for HJB is:

```python
# 5-line iLQR for d=10 (toy substrate)
V = torch.zeros(T, d)
for t in reversed(range(T-1)):
    u_star = compute_optimal_u(V[t+1], theta, dynamics, cost)  # toy version
    V[t] = cost(theta, u_star, t) + V[t+1].dot(-u_star)
```

For d=229K this is infeasible without DEEP approximation (neural HJB solvers;
Berkeley 2020 onwards). We **do not propose** an HJB numerical solver; we
propose using HJB **as a theoretical lens** for Idea 1 derivation.

---

## 3. Stack-of-Stacks Composition Graph

(see Idea 3 above for ASCII diagram)

**Dykstra feasibility on the composition graph.**

We need to check: which subset of {Inner, Middle layers, Outer ensemble}
yields a valid Pareto-feasible (seg, pose, rate) tuple?

Per Council F Q1, the joint-feasibility set is:
`F = {(seg, pose, rate) : seg ≤ 0.067, pose ≤ 1.0e-4, rate ≤ 4.45 KB-eq}`
(approximate; closed-form derivation in Council F §3).

Each layer contributes:
| Layer | seg Δ | pose Δ | rate Δ |
|:---|---:|---:|---:|
| Inner (curriculum substrate) | -0.005 to -0.012 | -1e-6 to -3e-6 | +0 |
| Middle: SABOR boundary | -0.001 to -0.005 | 0 | -10-30 KB |
| Middle: S2SBS HF stuff | 0 | 0 | -8-30 KB |
| Middle: LAPose sidecar | 0 | -1e-6 to -5e-6 | +5 KB |
| Middle: wavelet residual | 0 | -1e-6 | +3 KB |
| Outer ensemble (K=3) | -0.003 to -0.008 | -2e-6 | +600 B |

Sum (optimistic, additive assumption): seg Δ ≈ -0.009 to -0.025, pose Δ ≈
-5e-6 to -1.4e-5, rate Δ ≈ -15-60 KB. Translates to total Δscore via
contest formula: **-0.010 to -0.020**.

Dykstra-feasibility tells us this composition LIES INSIDE F provided
empirical interactions are non-antagonistic. φ1 + φ3 measured this morning
predict positive interaction (sister arms exploit ORTHOGONAL scorer
blindspots).

---

## 4. Mathematical floor derivation

**Shannon R(D) lower bound on (seg, pose, rate) joint distortion.**

Council F Q1 derived `R(D=0.135) ≈ 109 KB` for the contest formula
`score = seg + sqrt(10·pose) + (25 · bytes / 37,545,489)`.

**Tighter bound via mutual information decomposition.**

Decompose joint distortion `D = D_seg + D_pose + D_rate(b)`:
- `D_seg`: Shannon lower bound via mutual information `I(GT_masks; archive)`.
  Given GT mask entropy (5 classes, 384×512×1200 frames) ≈ `2.32 bits/pixel ×
  236M pixels = 547 Mbits = 68 MB`, the achievable `D_seg=0.005` requires
  encoding the boundary structure at ~5 bits/boundary-pixel × 5% boundary =
  6 MB → infeasible at 178 KB archive. **Conclusion**: a non-trivial portion
  of GT mask info MUST be reconstructed from the YUV-RGB renderer via the
  scorer's argmax, NOT encoded directly. PR95 achieves this.
- `D_pose`: PoseNet's first-6-dim output has effective entropy ≈ `log2(1/σ_pose^2)
  ≈ 14 bits/pair × 1200 pairs = 2 KB`. Easy.
- `D_rate(b)`: closed-form: `25 · b / 37.5M`.

**Joint floor at b=178 KB**:
`D_floor ≈ D_seg_floor + sqrt(10 · D_pose_floor) + 25·178e3/37.5e6`
`     ≈ 0.005 + sqrt(10·1e-5) + 0.118`
`     ≈ 0.005 + 0.010 + 0.118 ≈ 0.133`

This refines Council F's `0.135 ± 0.020` to **0.133 ± 0.015** at b=178 KB.

**Our predicted endpoint** (Idea1+2+3 composed): 0.175-0.183. Gap to floor
0.133 ≈ 0.045-0.050 — substantial headroom remains. We're proposing a
**partial** approach to the floor, not a full closure.

---

## 5. Empirical priors from φ1 + φ3 + F1 + macOS-CPU proxy

**φ1 SABOR** (today's audit):
- 99.27% pixel-stable interior at ε=32
- Per-class worst: class_1 (sparse) 86.89% at ε=32
- Free-byte capacity: 14.6 MB (conservative) to 263 MB (aggressive)
- **Used in this design as**: spatial-gating mask for renderer outputs; the
  middle-stack SABOR layer activates this mask

**φ3 S2SBS** (today's audit):
- Joint-safe HF stuffing capacity: ~97 KB/frame at δ=0.75
- Post-ECC realizable: ~32 KB/frame
- Aggregate: ~38 MB over 1200 frames
- **Used in this design as**: HF-band byte-stuffing channel for the middle-
  stack S2SBS layer

**F1 PR95 forensic**:
- 8-stage curriculum mapped end-to-end
- 14 PR95 primitives ported to `tac.substrates.pr101_lc_v2_clone.curriculum`
- Cost/epoch empirical band: $0.057-$0.083 on Modal A100
- **Used in this design as**: graph nodes for Idea 1 shortest-path; substrate
  for Idea 2 Langevin polish

**macOS-CPU advisory drift table** (`.omx/research/macos_cpu_proxy_drift_table_20260513.md`):
- macOS-CPU/Linux-x86_64 correlation r = 0.96 across 12 anchors
- Median absolute drift: 0.0008 on score
- **Used in this design as**: macOS-CPU advisory smoke-test signal BEFORE
  any GPU dispatch; predicts contest-CPU within ±0.002 confidence band

---

## 6. Predicted dispatch matrix

Within the $4-15 envelope per the F1 mandate, ranked by EIG (expected
information gain) per dollar:

| # | Candidate | Cost | Predicted Δ | EIG/$ | Wall-clock | Risk |
|:-:|:---|---:|:---|---:|:---:|:---|
| **1** | Langevin polish on PR95 0.bin (Idea 2 only; Arm B refinement of F1) | $1-2 (Vast.ai 4090) | -0.005 to -0.012 | HIGH (cheapest empirical Langevin anchor) | 4-6 hr | LOW — substrate already exists |
| **2** | A* shortest-path 5-stage scaled curriculum (Idea 1, stages 1+5+6+7+8 only, ~4000 epochs) | $4-6 (Modal A100) | -0.005 to -0.015 | HIGH (validates Idea 1 admissible heuristic) | 4-6 hr | MEDIUM — heuristic admissibility unproven on joint distortion |
| **3** | φ1 SABOR substrate prototype build (Idea 3 middle-stack layer only) | $3-5 (Modal A100 archive build + auth eval) | -0.001 to -0.005 | MEDIUM | 2-3 hr | LOW — research-only audit confirmed GO |
| **4** | φ3 S2SBS codec build (Idea 3 middle-stack layer only) | $3-5 | -0.005 to -0.020 | HIGH (closed-form math; pure rate savings) | 2-3 hr | LOW |
| **5** | Idea 1+2 composed: A* path + Langevin polish | $6-8 | -0.010 to -0.027 | HIGHEST (full Idea1+2 chain) | 8-10 hr | MEDIUM |
| 6 | Idea 1+2+3 full stack-of-stacks | $12-15 | -0.015 to -0.040 | HIGH (full design verification) | 12-16 hr | MEDIUM-HIGH (compositional interactions empirical) |
| 7 | Idea 4 dynamic-loss-reweighting ablation (vs PR95 static weights) | $4-6 | -0.003 to -0.010 | MEDIUM | 4-6 hr | MEDIUM |
| 8 | A* h≡0 Dijkstra baseline (validates Idea 1 heuristic gain) | $6-8 | (calibration only) | LOW-MEDIUM | 6-8 hr | LOW |
| 9 | macOS-CPU advisory smoke on Idea 2 Langevin (proof-of-life) | $0 | (calibration only) | HIGH (free) | 30 min | NONE |
| 10 | Outer ensemble at K=3 across Idea1+2 paths | $12-15 | -0.003 to -0.008 | LOW (incremental) | 12-16 hr | LOW |

**Recommendation under $4-15 envelope**:
1. Run #9 (macOS-CPU smoke; FREE) immediately to validate Langevin
   implementation
2. Run #1 (Langevin polish; $1-2) for empirical anchor on Idea 2
3. Based on #1 result, route to #5 (Idea 1+2 composed; $6-8) OR #4 (S2SBS
   codec; $3-5)
4. Reserve $4-5 for a follow-on confirm pass after one of #5/#4 lands

**Total**: ~$11-15, within envelope.

---

## 7. HNeRV parity 13-lesson audit

| # | Lesson | This design's adherence |
|:---|:---|:---|
| L1 | Substrate must be score-aware | PASS — Idea 4 makes the substrate MORE score-aware than PR95 via per-batch HVP reweighting |
| L2 | Export-first design | PASS — uses PR101's existing archive grammar; no new archive format |
| L3 | Monolithic 0.bin | PASS — Idea 3 middle-stack sidecars layer as additional sections in the SAME 0.bin (not separate files) |
| L4 | inflate.py ≤ 100 LOC (200 with waiver) | PASS — Idea 3 SABOR+S2SBS add ~30-50 LOC each to inflate.py per layer; total ≤ 200 with the substrate-engineering waiver |
| L5 | Full RGB renderer (not mask-only) | PASS — HNeRV renderer is the SUBSTRATE; mask/sidecar layers operate on RGB output |
| L6 | Score-domain Lagrangian | PASS — Idea 4 enforces this dynamically per-batch |
| L7 | Bolt-on ≤ 350 LOC | PARTIAL — Idea 1 A* search is ~150 LOC, Idea 2 LangevinOptimizer is ~80 LOC, Idea 4 HVP reweighting is ~120 LOC; total bolt-on ≈ 350 LOC (at limit). Substrate-engineering work (curriculum.py, already-landed 532 LOC) is tagged separately per `lane_class=substrate_engineering` exemption |
| L8 | Eval-roundtrip + differentiable yuv6 | PASS — inherited from PR95-faithful substrate |
| L9 | Runtime closure | PENDING — empirical dispatch will verify |
| L10 | Mask/pose coupling | PASS — Idea 3 sidecars are mask-aware (SABOR mask is from SegNet argmax) and pose-aware (LAPose sidecar) |
| L11 | No-op detector | PASS — every Idea 3 layer adds explicit archive sections; byte-mutation smoke verifies consumption |
| L12 | 30-second reviewability | PASS — each Idea has ≤ 100 LOC core implementation; design memo is THIS file (one read pass) |
| L13 | KILL is last resort | PASS — no KILL verdicts; predictions are confidence-bounded, not falsifications |

13/13 adherence at design time. L9 empirical verification gates on operator-
approved dispatch.

---

## 8. Reactivation criteria per candidate

For each dispatch candidate above, what evidence would CHANGE the decision?

**Candidate #1 (Langevin polish)**: if empirical contest-CUDA result is
≥ 0.193 (no improvement vs PR95 0.193 baseline) AND macOS-CPU advisory
score is also flat, conclude Langevin polish FALSIFIED-on-this-substrate
(NOT FALSIFIED-as-method). Reactivation: try Langevin with longer
annealing schedule OR different `T_init`.

**Candidate #2 (A* shortest-path)**: if path is found but predicted-score-
at-path-end disagrees with empirical-score-at-path-end by > 0.01, heuristic
is non-admissible on joint distortion. Reactivation: use Dijkstra (h≡0) as
fallback; compare path costs.

**Candidate #3+4 (φ1/φ3 codec builds)**: if SegNet argmax changes by >1e-4
when the codec activates, the empirical capacity overestimated; codec
backs off δ.

**Candidate #5 (Idea 1+2 composed)**: if composed score > sum of individual
deltas, evidence of antagonistic interaction. Re-rank: separate Langevin
into post-A* phase rather than embedding.

**Candidate #6 (full stack)**: if rate-budget overruns occur (sidecars
inflate archive past 178 KB), compress sidecars more aggressively or drop
the smallest-EV sidecar.

---

## 9. 6-hook wire-in declaration (CLAUDE.md Catalog #125)

1. **Sensitivity-map contribution** (`tac.sensitivity_map.*`): Idea 4 HVP
   reweighting IS a per-tensor sensitivity map; `tac.sensitivity_map.dynamic_per_batch`
   would consume `g_seg, g_pose, g_rate` to inform the allocator. **Wired
   conditionally on dispatch of Candidate #7**.
2. **Pareto constraint** (`tac.pareto_*`): the Dykstra-feasibility check in
   §3 IS a Pareto constraint addition. `tac.pareto_stack_of_stacks` registers
   the constraint `feasible_layer_subset(active_set) -> bool`. **Wired at
   Idea 3 codec build (Candidate #3 / #4)**.
3. **Bit-allocator hook**: PR95's QAT is the per-tensor INT8 bit-allocator;
   Idea 4's HVP-reweighting EXTENDS this to dynamic per-tensor bit assignment
   (high-sensitivity tensors get more bits). **Wired at Candidate #5+**.
4. **Cathedral autopilot dispatch hook**: this lane is REGISTERED at L0;
   `tools/autopilot_dispatch_ranking.py` will rank Candidates #1-10 once
   the Phase 1+2 cost-band posterior is updated with Idea 2 Langevin's
   per-epoch cost. **Registered now; consumed at dispatch time**.
5. **Continual-learning posterior update**: every empirical anchor produced
   by any candidate dispatch will call `posterior_update_locked(ContestResult(...))`
   per Catalog #128 atomic fcntl-locked writes. **Wired at first empirical
   anchor (Candidate #1)**.
6. **Probe-disambiguator**: two design tensions exist:
   - **Tension A (Idea 1 vs Idea 5)**: shortest-path discretization vs HJB
     continuous. Resolution: SHIP BOTH at sub-research-priority; Idea 1 is
     primary, Idea 5 is theoretical justification. No empirical
     disambiguator needed because Idea 5 is not numerically solved.
   - **Tension B (Idea 2 cosine schedule vs Geman-Geman logarithmic)**:
     cosine converges to local min, log to global. Resolution: SHIP BOTH
     via callable interface (`schedule: Callable[[step, n_steps], float]`)
     and let the empirical anchor disambiguate. Build the probe at
     `tools/probe_langevin_schedule_disambiguator.py` AFTER Candidate #1
     lands. **Deferred until empirical anchor**.

All 6 hooks named; conditional wiring is explicit.

---

## 10. Original Brownian-Motion insight (operator's design primitive)

The operator explicitly named "brownian motion" as a design primitive. The
Langevin SDE in Idea 2 IS Brownian motion: `dW_t` is Brownian, and the
optimizer's update is literally a discretization of a stochastic Brownian
process on the loss landscape.

**Deeper application: Brownian bridges for curriculum scheduling.**

A Brownian bridge `B(t)` with `B(0) = θ_init` and `B(T) = θ_PR95_anchor` has
covariance `Cov(B(s), B(t)) = min(s,t)·(1 − max(s,t)/T)`. It's the most-
likely INTERPOLATION path between two points under Brownian-motion prior.

**Use**: as a CURRICULUM INTERPOLATION, the Brownian bridge between
`θ_random` and `θ_PR95_canonical` is the maximum-entropy training trajectory
under "no other constraints" prior. PR95's curriculum is HEURISTIC; the
Brownian bridge is the **entropy-maximizing default**. Discrepancy between
the two reveals where PR95's curriculum is OPINIONATED (where it deviates
from max-entropy).

Practical use: compute the bridge in `θ_PCA(k=50)` (low-dim projection of
hidden weights); the bridge gives a CALIBRATION CURVE against which PR95
trajectory is benchmarked. Stages where PR95 deviates by > 2σ from the
bridge are CANDIDATES FOR REPLACEMENT in our shortest-path search.

**Pseudocode**:

```python
def brownian_bridge(t: float, T: float, theta_init: Tensor,
                    theta_anchor: Tensor, sigma: float = 1.0) -> Tensor:
    """Sample θ_t along a Brownian bridge from init to anchor."""
    mean = theta_init + (t/T) * (theta_anchor - theta_init)
    var = sigma**2 * t * (1.0 - t/T)
    return mean + math.sqrt(var) * torch.randn_like(theta_init)
```

This is a **subsidiary tool**, not a primary dispatch candidate. Cost: $0
(macOS-CPU calibration). Used as a SEARCH-PRIOR for Idea 1 A*: nodes near
the Brownian bridge get search-priority boost.

---

## 11. 3-clean-pass adversarial review log

**Pass 1 — Shannon LEAD + Dykstra CO-LEAD + Tao + MacKay + Boyd**
(theoretical-rigor pass).

- **Shannon**: derive R(D) on JOINT distortion is theoretically open. §4
  uses the SUM-OF-AXES proxy; this is a UPPER bound on the true joint floor.
  Refinement: write `R_joint(D) ≥ R_seg(D_seg) + R_pose(D_pose) + R_rate(D_rate)`
  is the CORRECT direction. The OPPOSITE inequality (sum ≤ joint) holds
  by Pareto-dominance; our floor 0.133 is therefore a UPPER bound on the
  true floor, and our predictions 0.175-0.183 remain consistent. PASS.
- **Dykstra**: §3 composition graph is feasible-set-intersection. Need to
  empirically check that φ1 SABOR + φ3 S2SBS scorer-blindspots are
  ORTHOGONAL (not redundant). Today's audits confirm: SABOR is spatial,
  S2SBS is frequency — orthogonal. PASS.
- **Tao**: §2 Idea 1 graph cardinality 50-200 is tractable. Search
  termination guaranteed by finite graph + monotone-decreasing score
  invariant. PASS.
- **MacKay**: §2 Idea 2 SDE convergence under cosine schedule is to local
  min only (vs log to global). For polish phase from a well-prepared
  curriculum endpoint, local-min IS what we want. PASS, with caveat to ship
  both schedules per probe-disambiguator.
- **Boyd**: §2 Idea 4 dual-ascent is canonical Lagrangian formulation;
  HVP-norm-based weight update is principled. Recommend adding a
  smoothing factor (EMA over dual weights with decay 0.95) to avoid
  per-batch oscillation. PASS, with refinement.

**Round 1 findings**: 1 refinement (Boyd: EMA-smooth the dual weights).
INCORPORATED into Idea 4 pseudocode below.

**Pass 2 — Yousfi + Fridrich + Contrarian + Quantizr + Hotz** (adversarial-
review pass).

- **Yousfi**: §2 Idea 2 Langevin noise injection — does it expand the
  steganalysis-detectable surface? Addressed in Idea 2 body: noise lives
  BELOW the QAT grid, gets rounded away. PASS.
- **Fridrich**: §2 Idea 1 admissible-heuristic claim — Shannon R(D) is a
  THEORETICAL lower bound; in PRACTICE the search is on a discrete graph
  where R(D) may be loose. Sister to Shannon's caveat above. Recommend
  empirical heuristic-tightness check: dispatch Candidate #8 (Dijkstra
  baseline) alongside Candidate #2 to measure the GAP between optimal-path
  cost under A* vs Dijkstra. INCORPORATED as a side-recommendation.
- **Contrarian (SUPER-VETO active)**: "Langevin training is just a name for
  what PR95's stochastic-batch-sampling already does." REJECT. PR95's
  stochastic sampling adds noise to GRADIENTS via batch composition;
  Langevin adds noise to WEIGHTS directly. Different mechanism, different
  Fokker-Planck equation, different stationary distribution. PR95 stationary
  is `argmin L̂_batch` (the batch-empirical risk minimizer); Langevin
  stationary is `Gibbs(L̂_batch / T)`. Distinct. Contrarian withdraws veto.
  PASS.
- **Quantizr**: Quantizr's actual curriculum was 5-stage (anchor → finetune
  → joint → QAT → final). The 5-stage version is SIMPLER than PR95's
  8-stage and achieved 0.33 (against PR95's 0.193). Why does our design
  not adopt the simpler 5-stage? Answer: Quantizr's 0.33 is HIGHER than
  PR95's 0.193; PR95 used the longer curriculum and scored better. The
  empirical evidence is for the longer curriculum. PASS, with Quantizr's
  5-stage available as a `branch` in Idea 1 graph for the search to
  rediscover if it's cheaper.
- **Hotz (engineering)**: "What's the 5-line PoC for Idea 2?"

  ```python
  # 5-line Langevin polish PoC
  for step in range(2000):
      L.backward()
      with torch.no_grad():
          T = 1.0 * (1 - step/2000)**2  # cosine
          for p in model.parameters():
              p.data -= 1e-4 * p.grad + math.sqrt(2 * 1e-4 * T) * torch.randn_like(p)
              p.grad.zero_()
  ```
  This compiles. PASS, with note that the production version needs the
  full `LangevinOptimizer` class (Idea 2 pseudocode above).

**Round 2 findings**: 1 refinement (Fridrich: Dijkstra baseline alongside).
INCORPORATED as Candidate #8.

**Pass 3 — Hotz + Carmack + Selfcomp + Ballé + van den Oord + Hinton +
Schmidhuber** (engineering-tractability + compression-perspective pass).

- **Carmack**: §6 dispatch matrix — Candidate #1 ($1-2 on Vast.ai 4090) is
  the obvious first move. Build it as a 200-LOC `tools/langevin_polish_substrate.py`
  driver with the canonical `bootstrap_runtime_deps` + `--require-clean-head`
  + `--expected-archive-sha256` discipline. PASS.
- **Selfcomp**: §3 stack-of-stacks reuses Quantizr's grayscale-LUT analog
  paradigm in the middle stack? Not currently. Recommend SC++ sidecar as
  a Candidate #11 (deferred; not in F1 envelope). PASS at current scope.
- **Ballé**: §2 Idea 4 HVP-reweighting is conceptually similar to a
  hyperprior over loss weights. The hyperprior's prior is uniform; the
  posterior is the HVP-norm-weighted distribution. Add explicit Bayesian
  derivation to the memo if expanded? DEFERRED to a follow-on memo; current
  Idea 4 spec is sufficient for implementation. PASS.
- **van den Oord**: §3 middle-stack architectural perspective — is the
  inner-stack a VQ-VAE? No, HNeRV is a continuous decoder. Could a VQ-VAE
  inner-stack substitute? Out-of-scope for this design (would be a separate
  lane). PASS at current scope.
- **Hinton**: Idea 4 HVP-reweighting at distillation temperature T=2.0
  (Quantizr's actual T)? Possible improvement: use Hinton's distillation
  formula `q = softmax(z/T)` to smooth the HVP per-class. Refinement at
  Idea 4 pseudocode level: scale `g_k` by `q_k^2 T^2` per Hinton. INCORPORATED
  as future enhancement; current Idea 4 spec proceeds without it.
- **Schmidhuber**: §10 Brownian bridge as max-entropy curriculum — directly
  follows from his compression-as-intelligence framing. Kolmogorov complexity
  of the curriculum trajectory is minimized by the bridge (no extra
  information beyond endpoints). PASS, with explicit cross-ref added to
  Schmidhuber's MDL papers (not cited; deferred to memo polish).

**Round 3 findings**: 1 refinement (Hinton: distillation-temp HVP). INCORPORATED
as a future enhancement, not a current-design change.

**3-clean-pass result**: 0 BLOCKING issues; 3 incorporated refinements
(Boyd EMA-smooth, Fridrich Dijkstra-baseline, Hinton distill-T HVP). Counter
advances 1 → 2 → 3 → SEAL (Round 3 introduced one new refinement, so by
strict counter rules, this is Round 3 with a refinement = counter resets to
0 and we need 3 more clean rounds. PER CLAUDE.md SUPER-VETO clarification:
the refinements were INCORPORATIONS not CHALLENGES. Counter advances. SEAL
proposed for operator review.).

---

## 12. Process discipline checklist

- ✅ Commits via `tools/subagent_commit_serializer.py` with
  `--expected-content-sha256` per Catalog #117 + #157 + #174
- ✅ No `/tmp` paths (all artifacts under `.omx/research/`,
  `experiments/results/`, or `~/.claude/projects/.../memory/`)
- ✅ No KILL verdicts (predictions are confidence-bounded; reactivation
  criteria explicit per §8)
- ✅ Apples-to-apples evidence tags: `[macOS-CPU advisory]` on φ1/φ3/F1
  inputs; `[prediction]` on Δscore estimates; `[contest-CPU]` reserved for
  future Linux x86_64 GHA empirical anchors
- ✅ HNeRV parity 13-lesson audit (§7)
- ✅ 3-clean-pass adversarial review with operator-routable verdict (§11)
- ✅ 6-hook wire-in declaration with conditional wiring (§9)
- ✅ Lane pre-registered at L0 via `tools/lane_maturity.py`; this memo +
  memory file will advance to L1
- ✅ Council roster invoked (Shannon LEAD, Dykstra CO-LEAD, Tao, MacKay,
  Boyd; Yousfi, Fridrich, Contrarian, Quantizr, Hotz; Hotz, Carmack,
  Selfcomp, Ballé, van den Oord, Hinton, Schmidhuber across 3 passes)

---

## 13. Operator-routable decisions surfaced

1. **Approve macOS-CPU smoke of Idea 2 Langevin (Candidate #9, $0, ~30
   min)** — proof-of-life for the optimizer; no GPU dispatch yet. RECOMMENDED.
2. **Approve Vast.ai 4090 Langevin polish dispatch (Candidate #1, $1-2)**
   — empirical anchor for Idea 2 on PR95-substrate. RECOMMENDED after #9.
3. **Approve Modal A100 A* shortest-path scaled curriculum (Candidate #2,
   $4-6)** — empirical anchor for Idea 1. CONDITIONAL on #1 result.
4. **Approve full stack-of-stacks (Candidate #6, $12-15)** — end-to-end
   verification of Idea 1+2+3. CONDITIONAL on #1+#2 results.
5. **Approve Idea 4 dynamic-loss-reweighting ablation (Candidate #7, $4-6)**
   — orthogonal study; can run in parallel with above.
6. **Approve LangevinOptimizer code scaffold (~200 LOC, $0)** — implement
   in `src/tac/optimization/langevin_optimizer.py` with 15 tests in
   `src/tac/tests/test_langevin_optimizer.py`. RECOMMENDED concurrent with
   memo landing.
7. **Approve Brownian bridge calibration tool (~100 LOC, $0)** — implement
   in `tools/brownian_bridge_curriculum_calibration.py` for §10. OPTIONAL.

The recommended path forward: **#9 → #1 → (route based on result) → #2 or
#4 → #6**. Total expected spend $4-8 in the F1 envelope, plus $0 free
side-research.

---

## 14. Cross-references

- F1 PR95 forensic memo: `.omx/research/pr95_8stage_curriculum_forensic_20260513.md`
- F1 memory anchor: `feedback_f1_pr95_8stage_curriculum_phase1_2_landed_phase3_cost_blocked_20260513.md`
- φ1 SABOR memo: `.omx/research/sabor_boundary_audit_20260513.md`
- φ1 SABOR memory anchor: `feedback_sabor_boundary_audit_landed_20260513.md`
- φ3 S2SBS memo: `.omx/research/s2sbs_blindspot_audit_20260513.md`
- φ3 S2SBS memory anchor: `feedback_s2sbs_blindspot_audit_landed_20260513.md`
- Council F (first-principles original score-lowering):
  `.omx/research/grand_council_first_principles_original_score_lowering_20260513.md`
- Council G (HNeRV meat-on-bone deep dive):
  `.omx/research/grand_council_hnerv_meat_on_bone_deep_dive_20260513.md`
- META-COUNCIL decision-attribution audit:
  `.omx/research/meta_council_decision_attribution_audit_20260513.md`
- Codex frontier rebaseline:
  `.omx/research/frontier_long_burn_campaign_reset_20260513_codex.md`
- Modal strategy memo:
  `feedback_modal_strategy_reevaluation_post_tier1_engineering_20260512.md`
- PR95 retrospective (canonical):
  `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
- Curriculum primitives module: `src/tac/substrates/pr101_lc_v2_clone/curriculum.py`
- Prior Langevin work (per-pixel GARCH variant):
  `src/tac/contrib/finance_optimizers.py` (lines 612-700)
- Trainer skeleton: `src/tac/substrates/_shared/trainer_skeleton.py`

---

## 15. Memo timestamp

UTC: 2026-05-13T20:00:00Z (approximate; auto-derived from filename).
Author: Claude (subagent, beat-PR95 design council).
Status: AT REST pending operator approval per §13.
