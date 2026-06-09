# comp-muon (Compositional Muon) — Applicability Research for HiNeRV MLX Stage-8

- **Date:** 2026-06-09 (operator-override research subagent)
- **Lane:** `lane_research_comp_muon_20260609` (research; no dispatch; no code edits)
- **Subject repo:** https://github.com/tilde-research/comp-muon-release
- **research_only:** `true` — all performance/quality numbers below are `[external-claim]`
  (from Tilde Research blogs/README) or `[extrapolation]`. NONE is a contest-score claim.
  Only `upstream/evaluate.py` on contest hardware produces a contest score.
- **Operator directive (verbatim):** *"spawn a research subagent, operator override, to
  comprehensive and deeply and broadly research this and its applicabilty and usefulness for
  us: https://github.com/tilde-research/comp-muon-release; we can thrash our local machine and
  keep the 128 gb saturated"*
- **Sister-agent boundary:** `B1-CAMPAIGN-V2` owns the HiNeRV runner + Muon wiring
  concurrently. This memo does NOT edit any training/runner code; recommendations are SPEC-only
  for B1 (or a follow-up) to consume.

---

## TL;DR / Recommendation

| Candidate | Regime fit for our 229K HNeRV sin-decoder | Verdict |
|---|---|---|
| **comp-muon (Compositional Muon)** | **None** — it optimizes *transformer attention* QK/OV composed matrices (`W_Q W_K^T`, `W_O W_V`). HNeRV has **no attention, no q/k/v/o projections, no heads**. Structurally inapplicable. | **DEFER (not for our architecture).** Reactivation only if we adopt an attention-bearing witness backend (e.g. a ViT/transformer-NeRV variant). |
| **Aurora (Tilde sister optimizer)** | **Weak** — it fixes Muon "neuron death" in **tall MLP up/gate projections** under activations with `φ(0)=0, φ'(0)≈0` (SwiGLU/ReLU²). HNeRV uses **sin** (`sin'(0)=1≠0`, precondition broken) and is a **229K conv decoder** (out of Aurora's tested regime). Genuinely a general-matrix drop-in though; cheaper to port than comp-muon. | **DEFER-with-reactivation.** A $0 MLX A/B vs vanilla Muon at stage-8 is the only cheap way to falsify the "sin breaks the precondition" hypothesis — see SPEC §Aurora. |
| **Vanilla Muon (Keller Jordan) — what B1 is already wiring** | **Direct fit.** Canonical PR95 L15 stage-8 continuation; **already merged into MLX core** (`mlx.optimizers.Muon`, July 2025) and already referenced in our codebase (`adapter.py:148,7738`). | **KEEP / PROCEED.** This is the OPTIMAL engineering for stage-8 per the "Canonical-vs-unique decision per layer" lens. |

**Bottom line:** comp-muon is the **wrong tool for our architecture** (attention-circuit
optimizer applied to an attention-free renderer). Do **not** wire comp-muon into B1 stage-8.
**Vanilla Muon is the correct stage-8 optimizer.** The only Tilde idea worth a cheap local
probe is **Aurora** (its non-square row-uniform polar *could* help the stem/large conv legs),
but the regime + activation evidence predicts a near-zero effect, so it is DEFER-with-
reactivation behind a $0 MLX A/B — not an adopt.

---

## What comp-muon is (algorithm summary)

**Source:** README + `src/compositional_muon.py` (cloned, 925 LOC across 5 files) +
blog "Towards Compositional Steepest Descent"
(https://blog.tilderesearch.com/blog/compositional-muon).

Vanilla Muon controls the **operator norm of each individual weight update**:
`ΔW* = −ε·msign(G_W)` (msign = Newton-Schulz orthogonalization → nearest semi-orthogonal
matrix). Compositional Muon (CM) instead controls the operator norm of the **composed**
update the loss actually sees through transformer attention — the QK product `M = W_Q W_K^T`
and the OV product `W_O W_V`. Each factor's gradient is **partner-whitened** by the inverse
Gram root of its partner before the spectral sign, then scaled by it again:

```
ΔW_Q = −(η/2)·msign(G_Q · C_K^{-1})·C_K^{-1},   C_K = (W_K^T W_K + λI)^{1/2}
ΔW_K = −(η/2)·msign(G_K · C_Q^{-1})·C_Q^{-1},   C_Q = (W_Q^T W_Q + λI)^{1/2}
```

(the "half-split" rule, ε/2 budget each; a "joint" variant stacks the two whitened factors
under one shared spectral sign). The OV pathway is analogous (V per-head, O per-matrix). When
each partner's Gram is near-isotropic, `C^{-1} ≈ c^{-1}I` collapses to a **scalar per-head
dynamic learning rate** — the "isotropic" approximation, which the blog says "recovers nearly
all of the gains" of full whitening at scalar cost.

**The public API makes the regime explicit and binding:**
```python
from compositional_muon import cm_ov, cm_qk
cm_qk(attn.q_proj.weight, attn.k_proj.weight, ..., head_dim=attn.head_dim, eta=lr)
cm_ov(attn.v_proj.weight, attn.o_proj.weight, ..., head_dim=attn.head_dim, eta=lr)
```
README, verbatim: *"CM governs only the attention QK and OV pairs; update the other parameters
with your optimizer of choice."* It **requires** `nn.Linear` q/k/v/o projection weights and a
`head_dim`. There is no general-matrix entry point.

---

## The 6 research-question answers

### RQ1 — WHAT IS IT
"comp" = **compositional** (not compute-optimal / compressed). It extends Keller Jordan's
Muon from single-matrix steepest descent to the **composed operators of transformer
attention**. Diff vs vanilla Muon: partner-whitening (inverse-Gram-root pre/post the spectral
sign) so each factor's step adapts to its partner's spectral geometry. Diff vs the
Moonlight/scaling-Muon line: the blog does not benchmark against Moonlight directly; it
positions CM as extending Muon's core principle to circuits (orthogonal axis to Moonlight's
LR/weight-decay scaling rules). `[external-claim]` benchmark numbers (LM pretraining):
- **340M / 10B tokens, QK-norm+RoPE (most realistic):** CM val loss 2.5477 vs Muon 2.5575 →
  **−0.0098 loss**.
- **1B / 70B tokens, QK-norm+RoPE:** 2.2866 vs 2.2917 → **−0.0051 loss**.
- **modded-nanoGPT track-3:** clears the 3.28 threshold at step 2875 vs 2890 → **~15 steps /
  ~0.5% faster**.
- **Downstream (ARC-C / HellaSwag / MMLU / Winogrande):** "largely comparable… no clear
  winner"; the blog explicitly *"avoids overstating pretraining gains."*

So even **in its home regime** (large transformer LM pretraining) the win is small and does
not translate downstream.

### RQ2 — IMPLEMENTATION / PORTABILITY
- **Framework:** pure **PyTorch** (`import torch`). No CUDA kernels, no Triton, no JAX. No MLX
  path. License **Apache 2.0**.
- **Core primitives:** `msign` (8-step Newton-Schulz, `polar_express` coefficients, bf16
  compute) + Gram inverse-square-root (`coupled_inv_sqrt` / `eigh_inv_sqrt`) + per-head
  reshaping + gauge-fixing. The Newton-Schulz half is identical in spirit to what **MLX
  already ships** (`mlx.optimizers.Muon._zeropower_via_newtonschulz5`). The *new* part vs MLX
  Muon is the partner-Gram inverse-root + per-head q/k/v/o reshaping.
- **MLX-port effort of the CORE algorithm:** moderate-but-pointless. The matmuls / Newton-
  Schulz / `eigh` all have MLX equivalents (`mx.linalg.eigh`, `mx.linalg.norm`), so a port is
  ~1–2 days. **But the port would have nothing to optimize in HNeRV** because there are no
  attention factor-pairs to feed `cm_qk`/`cm_ov`. Porting is only justified if a future
  witness backend is a transformer.

### RQ3 — REGIME FIT (the decisive question)
comp-muon's benefit is defined over **transformer attention QK/OV composed matrices**. Our
substrate (`src/tac/hnerv_arch_schema.py`, `src/tac/substrates/sane_hnerv/architecture.py`,
`src/tac/local_acceleration/pr95_hnerv_mlx.py`) is a **~229K-param HNeRV decoder**: a Linear
stem → **6 PixelShuffle-x2 conv stages** (channel taper `[C,C,C,.75C,.58C,.5C,.5C]`, C=36) →
refine convs → two RGB-head convs, with **sin (SIREN/NeRF) activation**. There is **no
attention, no Q/K/V/O, no multi-head structure**. The composed operator `M = W_Q W_K^T` that
CM's entire derivation rests on **does not exist** in our model. **Regime mismatch is total**,
not marginal. (Honest framing per CLAUDE.md: this is an implementation/architecture mismatch,
not a falsification of the CM idea — CM is fine *for transformers*.)

### RQ4 — 128GB-SATURATION ANGLE
comp-muon does **not** change the batch-size/throughput tradeoff or enable memory-heavy
strategies. Its extra cost is **per-parameter Gram inverse-roots**, which scale with `d_head²`
(tiny), not with batch. The blog reports no batch-size studies. So comp-muon offers **nothing**
for the operator's "saturate the 128 GB / thrash the machine" interest. The 128GB-saturation
lever for our regime is **larger MLX batch sizes + more frames/pairs per step in the existing
HNeRV trainer** (a B1/V2/V3 throughput concern), independent of which Muon variant runs in
stage-8. (Vanilla Muon's per-step orthogonalization cost is also batch-independent and `<3%`
wallclock overhead per Keller Jordan — so the optimizer is not the saturation bottleneck
either way.)

### RQ5 — SCORE-EV
For our HNeRV stage-8, comp-muon's score-EV is **≈ 0** (it cannot run — no attention factors).
`[extrapolation]` Even in a hypothetical attention-NeRV, the home-regime gains (−0.005 to
−0.01 LM val loss, no downstream win) would map to a **small, uncertain, possibly-zero** change
in `d_seg`/`d_pose`, because (a) the LM-loss → contest-cell-debt transfer is unknown, and (b)
the contest objective is dominated by `100·d_seg` argmax-flip behavior + bytes, not by the
optimizer's last-stage loss decimal. **No fabricated score is asserted.** EV does not clear the
bar to displace vanilla Muon at stage-8.

### RQ6 — RECOMMENDATION + WIRE-IN
**comp-muon → DEFER (architecture mismatch).** Do not wire into B1 stage-8. Reactivation
criterion: we adopt an **attention-bearing witness backend** (ViT-NeRV / transformer decoder)
where `cm_qk`/`cm_ov` have real factor-pairs — only then re-evaluate (and at that point also
re-check Aurora + Moonlight scaling rules). Per "Forbidden premature KILL": this is DEFER, not
KILL — the CM paradigm is intact for transformers.

**Vanilla Muon → KEEP for stage-8 (Canonical-vs-unique decision: ADOPT_CANONICAL).** It is the
canonical PR95 L15 continuation, `mlx.optimizers.Muon` is already merged + already referenced
in our codebase, and the stage-8 partition descriptor `pr95_stage8_muon_adamw_mlx`
(`pr95_hnerv_mlx.py:145`) already exists. Vanilla Muon **suffices** — comp-muon is not the
optimal engineering for an attention-free renderer; it is the optimal engineering for an
attention model we do not have.

---

## Sister candidate found during research: **Aurora** (the one actually worth a probe)

While researching comp-muon I found Tilde's **sister optimizer Aurora**
(https://github.com/tilde-research/aurora-release, **MIT**, pure-Python, 457 LOC), which is a
**much better regime match than comp-muon** and merits an explicit verdict.

- **Problem it fixes:** Muon's `polar(G)` on **non-square (tall, m>n)** matrices inherits
  **non-uniform left-singular row norms** → low-leverage rows get a self-reinforcing small
  share of update mass → **"neuron death."** `[external-claim]` *"more than one in four
  neurons are effectively dead by step 500"* at 340M.
- **Mechanism:** alternating projection onto the **intersection of the Stiefel manifold
  (`UᵀU=I`) and the row-oblique manifold (`‖U_i‖²=n/m`)** — i.e. row-normalize → polar →
  repeat (`pp_iterations` default 2). The code (`aurora.py`) is a **clean general-matrix
  drop-in**: `aurora(W, G, m, eta=lr, ...)`. For **square** matrices it explicitly *"reduces to
  the standard Muon update"*; for **wide** matrices it transposes to tall.
- **`[external-claim]` benchmarks (1.1B / ~100B tokens):** HellaSwag 67.6% vs Muon 65.1%; MMLU
  37.9% vs 27.1%; final loss 2.26 vs 2.31 (−0.05). modded-nanoGPT: 3225 steps (Aurora) /
  3175 (Aurora+Contra-Muon SoTA) vs NorMuon 3250. Overhead *"only ~6% over Muon."*
- **Why it is still only DEFER-with-reactivation for us (regime + activation evidence):**
  1. **Activation precondition is broken.** Aurora's neuron-death feedback loop is proven for
     activations with `φ(0)=0 AND φ'(0)≈0` (SwiGLU, ReLU²). HNeRV uses **sin**: `sin(0)=0` but
     `sin'(0)=cos(0)=1 ≠ 0`. The "dead-row starves the down-projection" cascade Aurora targets
     does **not** structurally arise with sin. `[extrapolation]` → predicted effect ≈ 0.
  2. **Out of tested regime.** Aurora's evidence is 340M–1.1B **transformer MLP up/gate
     projections**. We are a **229K conv decoder**; the blog says row-normalization is
     *"unnecessary or perhaps even harmful for square/wide cases"* — and our conv-flattened
     weights are mostly moderate-aspect (e.g. `blocks.0` flattens to (144, 324)), while the
     two genuinely extreme legs are `stem.weight` (very tall, but **input-adjacent → AdamW per
     Keller Jordan, not Muon**) and the **RGB heads** (very wide (3,162) → **output head →
     AdamW**, and "harmful for wide"). The matrices Muon actually touches in HNeRV are not the
     tall-MLP shape Aurora helps.
  3. **EV/uncertainty:** `[extrapolation]` plausible contest-score effect at stage-8 ≈ **0 ±
     small**; the ~6% per-step overhead is free locally (MLX, $0) but buys little.
- **Why it is worth a cheap probe anyway:** it is a **$0 local MLX A/B** (operator: "thrash the
  machine"), the port is trivial (only `polar()` is nontrivial and MLX already has Newton-
  Schulz), and it is the *one* falsifiable way to test the "sin breaks the precondition"
  hypothesis on the actual decoder. If a stage-8 Aurora run measurably beats stage-8 vanilla
  Muon on the proxy *and survives byte-closed re-eval* (`[macOS-MLX research-signal]` →
  candidate → CPU/CUDA), reactivate; otherwise the DEFER stands with evidence.

---

## Canonical-vs-unique decision per layer (stage-8 optimizer)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" falling-rule list:

| Layer | Canonical | Choice | Rationale |
|---|---|---|---|
| Stage-8 hidden-weight optimizer | `mlx.optimizers.Muon` (vanilla, PR95 L15) | **ADOPT_CANONICAL** | Obvious-fit: PR95 winner used exactly this; MLX-native; our `pr95_stage8_muon_adamw_mlx` descriptor already encodes it. |
| comp-muon as stage-8 optimizer | comp-muon | **REJECT (principled mismatch)** | comp-muon requires attention QK/OV factor-pairs that do not exist in HNeRV. Not a fork decision — a structural inapplicability. |
| Aurora as stage-8 optimizer | Aurora | **UNCLEAR → DEFER behind $0 paired smoke** | Non-square row-uniform polar *might* help, but sin activation + small-conv regime predict ≈0; the burden of proof is on proving-it-helps via a paired MLX A/B, not adopting by default. |
| Non-matrix params (latents/biases/norms/QAT/scalars) | AdamW (PR95) | **ADOPT_CANONICAL** | Keller Jordan canon: Muon only on hidden ≥2D; AdamW on embeddings/heads/scalars/vectors. Unchanged by any of the above. |

---

## Wire-in SPEC (for B1-CAMPAIGN-V2 or a follow-up — NOT implemented here)

**A) comp-muon:** *No wire-in.* It is architecturally inapplicable to HNeRV. Do not add it to
the stage-8 partition. (If/when an attention-NeRV witness backend is designed, the entry points
would be `cm_qk(q,k,...,head_dim)` + `cm_ov(v,o,...,head_dim)` on that model's attention blocks,
with vanilla Muon/AdamW on the rest — but that is a new-substrate design, not a stage-8 tweak.)

**B) Vanilla Muon (status quo — keep):** B1's existing stage-8 partition is correct:
- Muon on **matrix-like decoder weights** (ndim≥2 after conv-flatten), **excluding** `stem.*`
  (input-adjacent) and `rgb_0/rgb_1.*` (output heads) per Keller Jordan canon.
- AdamW on latents, biases, norms, entropy/QAT-quant params, scalar-schedule params, stem, and
  rgb-heads.
- Use `mlx.optimizers.Muon` (already imported at `adapter.py:148,7738`); descriptor
  `pr95_stage8_muon_adamw_mlx` (`pr95_hnerv_mlx.py:145`). No change required.

**C) Aurora (optional $0 falsification probe — DEFER-with-reactivation; do NOT promote):**
1. Port `aurora.py` (72 LOC) + `polar.py` (48 LOC) to MLX as a **research-only** optimizer
   `tac.local_acceleration.aurora_mlx` (NOT in `tac` core; it is a research probe). The only
   nontrivial primitive is the 12-step simple-quintic Newton-Schulz `polar()`, which has a
   direct MLX analogue (mirror `mlx.optimizers.Muon._zeropower_via_newtonschulz5`); the rest is
   `mx` matmuls + `norm(dim=-1)` row-normalization. MIT license → clean to vendor with
   attribution.
2. **A/B at stage-8 only:** identical PR95 stage-1..7 checkpoint, then run stage-8 twice —
   vanilla Muon vs Aurora — same epochs/LR/seed. Measure proxy `d_seg`/`d_pose` deltas, then
   **byte-close the better archive and re-score** (`[macOS-MLX research-signal]` → candidate;
   exact CPU/CUDA before any score/promote claim — MLX is never a contest axis).
3. **Reactivation criterion (per "Substrate MUST be at OPTIMAL FORM"):** reactivate Aurora into
   stage-8 only if it beats vanilla Muon by a real, byte-closed margin that survives re-eval.
   Predicted outcome `[extrapolation]`: ≈ no improvement (sin precondition broken + Muon-
   touched HNeRV legs are not the tall-MLP shape Aurora helps). If confirmed-null, record the
   DEFER with the empirical artifact and move on.

---

## 6-hook wire-in declaration (per Catalog #125; this is a research memo)

- **#1 Sensitivity-map:** N/A — research memo, no per-axis byte sensitivity produced.
- **#2 Pareto constraint:** N/A — non-binding (optimizer choice does not change the
  rate/seg/pose feasible region directly).
- **#3 Bit-allocator hook:** N/A — optimizer, not a codec primitive.
- **#4 Cathedral autopilot dispatch:** N/A — research_only; no archive-deployable artifact.
- **#5 Continual-learning posterior:** N/A — no empirical anchor landed (no run executed). If
  the optional Aurora $0 A/B (SPEC §C) is later run, *that* lands a posterior anchor.
- **#6 Probe-disambiguator:** the SPEC §C Aurora-vs-vanilla-Muon stage-8 A/B **is** the
  disambiguator for the one defensible open question (does Aurora's non-square fix help our
  conv legs under sin?). Until run, the regime+activation argument is the (provisional) verdict.

`council_predicted_mission_contribution: frontier_protecting` — this memo prevents a low-EV
detour (porting/wiring an attention-circuit optimizer into an attention-free renderer) and
redirects the one cheap probe (Aurora) behind a falsifiable $0 gate, protecting B1's vanilla-
Muon stage-8 from churn.

---

## Sources

- comp-muon repo: https://github.com/tilde-research/comp-muon-release (Apache 2.0; cloned to
  scratch, 925 LOC; `src/compositional_muon.py`, `src/msign.py`, `src/whitening.py`,
  `src/gauge.py`, `src/main.py`).
- comp-muon blog "Towards Compositional Steepest Descent":
  https://blog.tilderesearch.com/blog/compositional-muon
- Aurora repo: https://github.com/tilde-research/aurora-release (MIT; cloned, 457 LOC).
- Aurora blog: https://blog.tilderesearch.com/blog/aurora
- Aurora coverage (corroborating "neuron death" framing): MarkTechPost
  https://www.marktechpost.com/2026/05/12/tilde-research-introduces-aurora-a-leverage-aware-optimizer-that-fixes-a-hidden-neuron-death-problem-in-muon/
- Keller Jordan, vanilla Muon ("optimizer for hidden layers"; Muon on ≥2D hidden, AdamW on
  embeddings/heads/scalars): https://kellerjordan.github.io/posts/muon/ ;
  https://github.com/KellerJordan/Muon
- MLX Muon merge (PR #1914, merged 2025-07-18; `mlx.optimizers.Muon`; conv via reshape;
  split-group via separate optimizer instances): https://github.com/ml-explore/mlx/pull/1914
- Our HNeRV substrate (regime grounding): `src/tac/hnerv_arch_schema.py` (weight schema),
  `src/tac/substrates/sane_hnerv/architecture.py` (sin activation, ~229K), and
  `src/tac/local_acceleration/pr95_hnerv_mlx.py:135-145` (stage-8 = `pr95_stage8_muon_adamw_mlx`;
  `mlx.optimizers.Muon` referenced at `adapter.py:148,7738`).

---

## Scratch / disk hygiene

Both repos cloned (depth-1) to `.omx/tmp/comp_muon_research_scratch/` (local scratch, NOT
`/tmp` per AGENTS.md; SSD tiers `/Volumes/VertigoDataTier/pact` 811Gi + `/Volumes/APDataStore/pact`
1.2Ti confirmed available but unneeded — clones are <2 MB each). These are throwaway forensic
inputs (Apache-2.0 / MIT public source); safe to delete. No large artifacts produced.
