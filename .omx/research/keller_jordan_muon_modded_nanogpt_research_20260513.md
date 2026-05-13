# Keller Jordan — Muon optimizer + modded-nanogpt — research synthesis (2026-05-13)

**Lane**: `lane_keller_jordan_muon_modded_nanogpt_research_20260513` (registered L0 → working to L1)
**lane_class**: `substrate_engineering` (optimizer + training recipe; archive grammar unaffected; this lane delivers a primitive consumed by downstream substrate lanes)
**research_only**: false (Muon primitive ports to live substrate trainers and consumes contest-CUDA budget once integrated)
**Operator directive**: 2026-05-13 — "https://x.com/kellerjordan0/status/2054255672636981423 and the related repo and all information there to exploit and use"
**Sister memos**: `pr95_8stage_curriculum_forensic_20260513.md` (where Muon was already used in Stage 8), `online_research_C_optimizers_20260513.md`, `online_research_bleeding_edge_synthesis_20260513.md`, `zen_state_frontier_deep_math_research_20260513.md`, IGLT source `src/tac/optimization/iglt.py`.

Evidence grades used: `[third-party-empirical:<repo>@<sha>]`, `[third-party-empirical:<arxiv-id>]`, `[mathematical-derivation]`, `[literature-prediction]`. NO `[contest-CUDA]` or `[contest-CPU]` claims in this memo — no archive built; promotion-eligible=false per CLAUDE.md.

---

## 1. The X post (the operator's link)

**URL**: `https://x.com/kellerjordan0/status/2054255672636981423`
**Date**: 2026-05-04 (announcement by Keller Jordan)
**Author**: Keller Jordan (`@kellerjordan0`), OpenAI; CIFAR-10 / NanoGPT speedrun curator
**Content** (verbatim per public Google index):

> "Modded-NanoGPT optimization result #14 (2026/05/04): @Sam_Acqua has achieved a new record of 3150 steps (-60), by adding SOAP preconditioning before Muon orthogonalization for the MLP weights (SOAP-Muon)."

`[third-party-empirical:google-search-index]` — X is auth-walled to WebFetch; content recovered from web search citation. The previous record was 3210 steps; the new record (3150 steps, **-60 step / ~1.9% step-count reduction**) was set by Sam Acqua via **SOAP-Muon**: layer SOAP (Shampoo with Adam in eigenbasis) preconditioning IN FRONT of Muon's Newton-Schulz orthogonalization on MLP weight matrices.

### What "SOAP-Muon" means structurally

1. **SOAP** (Vyas et al. 2024, arXiv 2409.11321) — Shampoo decomposed via eigendecomposition of Shampoo's L/R preconditioners, with Adam's second-moment running average computed IN that eigenbasis. Equivalent to "Adam in a rotated coordinate system." Adds ONE hyperparameter (precondition_frequency, default 10) over Adam. Original SOAP claims: -40% iterations and -35% wall-clock vs AdamW at 360M-660M LLM scale.
2. **Muon** (Jordan 2024) — momentum-SGD update with the resulting matrix replaced by `Ortho(M) = U V^T` via 5-step Newton-Schulz quintic iteration. Spectral-norm steepest descent (Bernstein derivation).
3. **SOAP-Muon** — pipeline: gradient → SOAP eigenbasis rotation → Adam second-moment in eigenbasis → Newton-Schulz orthogonalization. SOAP handles directional + magnitude curvature (preconditioning); Muon handles spectral-norm bounded-step geometry. The combination is multiplicative on convergence rate when both signals are non-redundant — Sam Acqua's empirical result confirms they are non-redundant for transformer MLP weights at NanoGPT scale.

**Predicted applicability to our contest**: HNeRV-family decoders are conv-stacks, not transformers. The MLP-specific SOAP-Muon result does NOT immediately port because (a) our parameter count is ~80K-230K, not ~125M-660M, (b) we have no MLP layers (stem is a single Linear, hidden is Conv2d), (c) condition number on small Conv2d may already be tractable for plain Muon. BUT — the *technique class* (preconditioner-then-orthogonalize) is portable. See §6 for a contest-applicable formulation.

---

## 2. Muon repo deep-dive

**Source**: `https://github.com/KellerJordan/Muon` (HEAD `6399c65`, 2026-01-19; 2,574 stars; updated 2026-05-13)
**Canonical implementation**: `muon.py` (~200 LOC)
**Reference**: Jordan, Jin, Boza, Jiacheng, Cesista, Newhouse, Bernstein (2024). "Muon: An optimizer for hidden layers in neural networks." `https://kellerjordan.github.io/posts/muon/`

### 2.1 The canonical update rule (verbatim from `muon.py`)

```python
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:                              # conv filters → flatten to 2D
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update
```

Then apply decoupled weight decay: `p.mul_(1 - lr * weight_decay)`, then `p.add_(update.reshape(p.shape), alpha=-lr)`.

### 2.2 Newton-Schulz quintic — the canonical coefficients (CURRENT, stable)

```python
def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)              # CANONICAL
    X = G.bfloat16()                                  # BF16 native
    if G.size(-2) > G.size(-1):
        X = X.mT                                      # work on tall side
    X = X / (X.norm(dim=(-2,-1), keepdim=True) + 1e-7)  # spectral norm ≤ 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A                         # YouJiacheng quintic strategy
        X = a * X + B @ X
    return X.mT if G.size(-2) > G.size(-1) else X
```

**Coefficient derivation** `[mathematical-derivation]`: the iteration applies `φ(σ) = aσ + bσ³ + cσ⁵` elementwise to singular values. The official optimization objective is "maximize slope at 0 subject to `φ^N([0,1]) ⊂ [0.7, 1.3]` for N=5." Iteration converges to `Ortho(G) = U S' V^T` where `S'_{ii} ∼ Uniform(0.5, 1.5)` — **NOT** exactly UV^T (the original Higham/Björck NS for UV^T uses coefficients (1.5, -0.5, 0)). The relaxed-tolerance variant is the optimization-friendly form Keller found empirically.

**Refinement attempts** (`[third-party-empirical:leloykun.github.io/ponder/muon-opt-coeffs]`, Cesista 2025): gradient-descent tuning of `(a,b,c)` yields **1-2% wall-clock improvements at GPT2-medium scale**, but **NO improvement on GPT2-small** because "early-training steepness near 0 matters more than tail variance" — small models live entirely in the steepness regime. **Implication for our 80K-230K param renderers**: stick with canonical `(3.4445, -4.7750, 2.0315)`. The 1-2% paper is at the noise floor for our parameter count.

### 2.3 Hyperparameter defaults (recommendation from the README)

| Parameter | Default | Notes |
|---|---|---|
| `lr` (Muon) | `0.02` | "in units of spectral norm per update"; muP-style — should not need retuning across scales |
| `lr` (Adam aux) | `3e-4` | typical AdamW |
| `momentum` | `0.95` | Nesterov-style |
| `nesterov` | `True` | "works a bit better in every case we have tested" |
| `ns_steps` | `5` | 5 NS steps suffice; +overhead ∝ T·m/B (NanoGPT scale: 0.7%) |
| `weight_decay` | `0.01` (Muon), `0.01` (Adam) | Kimi.ai paper uses `0.1` at LLM scale |
| `betas` (Adam aux) | `(0.9, 0.95)` | tighter β2 than canonical 0.999 |
| `eps` (Adam aux) | `1e-10` | smaller than canonical 1e-8 |

**Distributed vs single-device**: repo provides `Muon` (distributed, requires `dist.init_process_group`) and `SingleDeviceMuon` (no `dist` calls; our use case). PR95's `submissions/hnerv_muon/src/optim.py` already inlines a single-device variant — exact byte-faithful port of the canonical algorithm with `wd=5e-4` (researcher #24's tweak, not in canonical PR95 but in PR95-extension Stage 8).

### 2.4 Parameter partition (`MuonWithAuxAdam`)

The canonical pattern from `Muon/muon.py`:

```python
hidden_matrix_params = [p for n,p in model.blocks.named_parameters() if p.ndim >= 2 and 'embed' not in n]
embed_params = [p for n,p in model.named_parameters() if 'embed' in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
adam_groups = [
    dict(params=head_params,   lr=0.22),
    dict(params=embed_params,  lr=0.6),
    dict(params=scalar_params, lr=0.04),
]
adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
muon_group   = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
optimizer = MuonWithAuxAdam([*adam_groups, muon_group])
```

For HNeRV-family (substrate = `HNeRVDecoder`):
- **Muon group** (2D+ hidden weights, excludes stem + RGB heads): conv weights in the 6 upsample blocks + the dilated `refine` block + 1×1 conv weights. PR95 partition: `name not in {'stem', 'rgb_0', 'rgb_1'} AND p.ndim >= 2`.
- **AdamW group**: stem Linear, both Conv RGB heads, all biases, all 1D params (BN/LN gains if any), AND the per-pair latents at 10× lr_mult.

### 2.5 Recent activity / version surface

- HEAD `6399c65` (2026-01-19) merged sherlcok314159 PR #60 — **fix the adjusted lr for conv params**. This is the very fix our local copy in `submissions/hnerv_muon/src/optim.py` already reflects: `scale = max(1.0, (g2d.size(0) / g2d.size(1)) ** 0.5)`. No additional code changes since.
- 2 stars/day growth; Kimi.ai (Moonlight) used Muon for a 3B/16B MoE pretrain (arXiv 2502.16982).
- Cesista coefficient-tuning paper (2025) — superseded discussion; canonical coefficients stable.

---

## 3. modded-nanogpt deep-dive

**Source**: `https://github.com/KellerJordan/modded-nanogpt`
**Latest record**: **Record 80, 84.4s wall-clock**, 8× H100, val_loss ≤ 3.28 (2026-01-30 baseline). Sam Acqua's **2026-05-04 SOAP-Muon record** is the operator's directive trigger.
**Cumulative speedup**: llm.c baseline 45 min → 84.4s ≈ **32× speedup** across ~80 record-iterations.

### 3.1 Technique catalog (ranked by impact on speedrun timer)

`[third-party-empirical:modded-nanogpt README + records/ logs]`

| Rank | Technique | Estimated wall-clock saved | Notes / portability to contest |
|---|---|---|---|
| 1 | **Flash Attention 3 + sliding-window** | ~540s | Transformer-only; not portable |
| 2 | **Muon optimizer** | ~300s | **HIGHLY PORTABLE** — already in PR95 Stage 8 |
| 3 | **FP8 LM-head + asymmetric rescale** | ~150s | Specific to softmax-output; partial fit (FP8 conv head?) |
| 4 | **Fused Triton kernels** (CE, linear+ReLU², softcap) | ~200s | **PARTIAL** — fused score-aware loss surface candidate |
| 5 | **U-Net skip pattern (block 0→6, 2→6, 4→6)** | ~90s | **HIGHLY PORTABLE** — already in HNeRVDecoder block design |
| 6 | **Value embeddings (untied)** | ~80s | Transformer-only |
| 7 | **YaRN window warmup** | ~45s | Transformer-only |
| 8 | **Batch size scheduling** | ~35s | **PORTABLE** — already in HNeRV-family practice (batch 8 fixed; could schedule) |
| 9 | **ReLU² activation** | ~5%+ | **PORTABLE** — drop-in replacement candidate vs `sin(x+identity)` SIREN |
| 10 | **Logit softcap (cap=15)** | ~20s | **PORTABLE** — bounds output magnitude before scorer ingest |
| 11 | **Embedding skip → every block** | — | **HIGHLY PORTABLE** — latent skip into deeper decoder blocks |
| 12 | **Paired-head attention (Muon orthogonalize Q/K per 2-head group)** | ~12s | Transformer-only |
| 13 | **Vocab padding to 128-multiple** | ~2% | N/A |
| 14 | **RMSNorm (no affine)** | ~3% | **PORTABLE** — replace any LN with affine-free RMSNorm |
| 15 | **Trapezoidal LR schedule** | ~4% | **PORTABLE** — alternative to cosine in PR95 Stage 8 |
| 16 | **WSD (Warmup-Stable-Decay) schedule** | — | **HIGHLY PORTABLE** — PR95 uses fresh cosine per stage; WSD may be superior |
| 17 | **Multi-token prediction** | — | Predict next 2-3 tokens jointly; LM-specific |
| 18 | **Aligning batch starts with EoS** | ~12s | N/A |
| 19 | **Cautious weight-decay schedule** | — | **PORTABLE** — decoupled WD that ramps with LR cooldown |
| 20 | **Async data loading** | ~40s | **PORTABLE** — pyav decode in worker thread |
| 21 | **`torch.compile()` Inductor** | — | **PORTABLE** — already opt-in flag in time-traveler trainer |
| 22 | **BF16 activation cast** | ~10s | **PORTABLE** — already discussed in Tier 1 engineering wins |

### 3.2 Optimizer schedule (record 80 reference, Muon era)

- **Warmup**: ~5% of training steps, linear LR
- **Stable/peak**: short region at peak LR
- **Cooldown**: ~45% of training at low LR (trapezoidal tail, NOT cosine to zero)
- **Final min LR**: 0.1× peak (NOT zero — cosine-to-zero is suboptimal)
- **Separate schedules per optimizer**: Muon and Adam have different curves (Muon LR can decay faster than Adam in practice)

### 3.3 Architectural "free wins" applicable to HNeRV-family

1. **U-Net skips**: block 0→6, block 2→6, block 4→6. HNeRVDecoder already has bilinear-skip per upsample block, but does NOT have cross-block long-range skips. Adding `feat_block0 → feat_block6` (after channel projection) is a few-LOC change with empirical ~1.5min/45min ≈ 3% speedup precedent.
2. **ReLU² vs sin(x+identity)**: HNeRV stages 1-7 use SIREN-style `sin(x + identity)`. Speedrun found ReLU² faster + same/better final loss in transformer setting. For SegMap-style boundary tasks the SIREN bias may matter; this is a council-grade tradeoff.
3. **Logit softcap before scorer**: PR95 already applies `.clamp(0, 255)` + STE round. Speedrun-style softcap (`30 * tanh(logits/30)`) instead of hard clamp gives differentiable gradient everywhere and prevents saturation. Worth a smoke ablation.
4. **WSD schedule replacing cosine**: PR95 Stage 8 uses cosine; WSD's flat-peak-then-trapezoidal-cooldown is the speedrun's chosen schedule. ~4% wall-clock saved precedent.
5. **BF16 throughout (not just NS)**: Muon already does NS in BF16. Speedrun extends to BF16 activations and model weights; FP32 only for optimizer state, layer norm running stats. This is the autocast-FP16 work-in-progress in time-traveler trainer (Tier 1 deferred).

---

## 4. Kimi.ai Muon scaling paper (arXiv 2502.16982v1, Moonlight)

`[third-party-empirical:arxiv-2502.16982v1]` — large-scale Muon evidence.

Two critical fixes for scale that ARE small-model-relevant:

**Fix 1 — Add explicit weight decay** (the canonical Muon as ported in PR95 used `wd=0.0`):
- Kimi recommendation: `weight_decay = 0.1`
- Mechanism: without WD, "some model weights grew too large over time"; convergence is faster initially but degrades at long horizon
- For our HNeRV-family at ~3000 epoch training: WD is the right move. PR95-extension already moved to `wd=5e-4` (researcher #24 tweak); Kimi suggests it could go higher

**Fix 2 — Per-parameter update RMS-matched to Adam**:

> `W_t = W_t-1 - η_t · (0.2 · O_t · √max(A,B) + λ · W_t-1)`

where `O_t = orthogonalize(M_t)` and `A, B` are matrix shape dims. The `0.2 · √max(A,B)` factor matches the RMS of Muon's update to Adam's update at the same LR. Without this, a single LR cannot be used across both optimizers — Muon's "spectral norm units" don't match Adam's "elementwise RMS units."

**Note**: the canonical Muon `muon.py` already includes `update *= max(1, update.size(-2)/update.size(-1))**0.5` which captures the *shape-aspect-ratio* correction. The Kimi `0.2 · √max(A,B)` is the *magnitude-matching* factor (an additional multiplier) — these are different corrections. Adding both is the canonical Moonlight recipe.

### 4.1 What this means for our 80K-230K param decoders

- WD = 0.1 is too aggressive at our scale; PR95-extension's 5e-4 is closer to right
- The `0.2 · √max(A,B)` factor is a STRONG normalization; without it, Muon at `lr=2e-4` (PR95 Stage 8 setting) and AdamW at `lr=1e-5` are NOT comparable. PR95 chose these by separate tuning sweeps — implicitly recovering some of the RMS match
- **Predicted score-lowering effect**: adding the Moonlight RMS-match + Kimi-WD gives PR95-stage-8 a more principled hyperparameter surface. Predicted Δscore: −0.001 to −0.005 [literature-prediction] — small but cheap

---

## 5. Top-10 techniques applicable to our contest

Ranked by `(predicted Δscore × probability of success) / implementation cost`. Score-axis attribution per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent": at PR106 r2 frontier (pose_avg ≈ 3.4e-5), pose marginal sensitivity is 2.71× SegNet's, so pose-axis improvements rank higher.

| # | Technique | Predicted Δscore (lit) | Impl cost | Composition |
|---|---|---|---|---|
| **1** | **Muon for hidden conv weights (replaces / supplements AdamW)** | −0.003 to −0.010 | LOW (existing port in PR95; ~50 LOC backport to time-traveler) | Composes with IGLT (sequential, end-of-training polish); composes with ternary QAT (Muon during float→8bit→4bit→2bit warmup) |
| **2** | **WSD LR schedule replacing cosine** | −0.001 to −0.003 | LOW (~20 LOC, replace `CosineAnnealingLR`) | Composes orthogonally with all optimizers |
| **3** | **Kimi.ai RMS-matched update scale (`0.2 · √max(A,B)`)** | −0.001 to −0.005 | LOW (~10 LOC patch to Muon) | Composes with Muon (it IS a Muon variant); decouples Muon LR from Adam LR |
| **4** | **Cross-block long-range skip (block 0 → block 6 in HNeRVDecoder)** | −0.002 to −0.008 | LOW-MED (~30 LOC, requires channel projection) | Composes with any decoder substrate (HNeRV, time-traveler, sane_hnerv) |
| **5** | **BF16 activations + FP32 optimizer state** | 0 (speed, not score) | MED (autocast wrap; Tier 1 backport pending) | Composes with everything; unlocks 4-6× T4 throughput per Tier 1 engineering audit |
| **6** | **Differentiable logit softcap before STE clamp** | −0.001 to −0.005 | LOW (~5 LOC; `30·tanh(x/30)` replaces `.clamp(0,255)` STE) | Composes with eval-roundtrip; potentially closer to true contest forward |
| **7** | **`torch.compile()` (Inductor)** | 0 (speed, not score) | LOW (flag already exists; just enable) | Composes with everything; ~2× speed unlock |
| **8** | **ReLU² activation in non-final conv blocks** | −0.001 to −0.005 | LOW-MED (~10 LOC; replaces `sin(x+identity)`; substrate ablation) | Council-grade tradeoff — SIREN bias for boundaries may be load-bearing |
| **9** | **SOAP preconditioning for the stem Linear (1×6×8 → C channels)** | −0.001 to −0.003 | MED (SOAP impl + integration; ~150 LOC) | Composes with Muon (SOAP for stem, Muon for hidden, AdamW for heads); the X-post-#14 technique scaled down |
| **10** | **Cautious weight-decay schedule (WD ramps with cooldown)** | −0.0005 to −0.002 | LOW (~10 LOC) | Composes with WSD; small effect alone |

Caveats:
- `[literature-prediction]` — predictions are NanoGPT-derived speedup translated to score-lowering via the rough rule "1% wall-clock saving on a fixed budget = comparable empirical-loss improvement at saturated training." Our score is not val_loss but the contest's `100·d_seg + √(10·d_pose) + 25·rate` formula. Direct mapping has 2-5× scatter. **No prediction is `[contest-CUDA]` until an empirical anchor lands.**
- All techniques composable WITH the just-landed time-traveler + IGLT + ternary QAT primitives. None compete; all stack additively at distinct frontier stages.

---

## 6. Concrete Muon integration plan for the time-traveler trainer

### 6.1 Wire-in surface

`experiments/train_substrate_time_traveler_l5_autonomy.py:245`:

```python
p.add_argument("--optimizer", choices=("adamw", "iglt", "muon", "muon+iglt"), default="adamw",
               help="adamw | iglt | muon | muon+iglt (Muon main, IGLT polish at end)")
```

`L672` becomes:

```python
if args.optimizer == "muon":
    from tac.optimization.muon import MuonWithAuxAdam, partition_params_for_muon
    muon_params, adamw_params = partition_params_for_muon(substrate)
    # per-pair latent always AdamW at 10× LR
    adamw_groups = [
        dict(params=adamw_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-10,
             weight_decay=args.weight_decay, use_muon=False),
        dict(params=[per_pair_side_info_float], lr=args.lr * 10.0,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, use_muon=False),
    ]
    muon_group = dict(params=muon_params, lr=args.muon_lr,
                      momentum=0.95, weight_decay=args.weight_decay, use_muon=True)
    optimizer = MuonWithAuxAdam([*adamw_groups, muon_group])
    scheduler = make_wsd_schedule(optimizer, args.epochs,
                                  warmup_frac=0.05, decay_frac=0.45, final_lr_frac=0.1)
```

### 6.2 Parameter partition for time-traveler substrate

Time-traveler L5 mirrors HNeRVDecoder structure (`world_model` + `decoder` + `predictive_residual_head`). The partition:

- **Muon group**: every `Conv2d.weight` with `ndim >= 2` in `world_model.layers[*]` and `decoder.upsample_blocks[*]`, EXCLUDING the stem Linear and the RGB output convs (per Keller's "first conv goes to Adam" guidance).
- **AdamW group**: stem Linear, RGB heads, all biases, all 1D params (gain/scale), and the per-pair side-info float tensor (latents at 10× lr_mult per PR95 convention).

Provide a canonical helper `tac.optimization.muon.partition_params_for_muon(model)` modeled exactly on the PR95 implementation at `submissions/hnerv_muon/src/optim.py:83-100` but with the time-traveler module names.

### 6.3 Hyperparameter starting point (PR95 + Kimi blend)

| Group | LR | momentum | wd | β2 / eps |
|---|---|---|---|---|
| Muon hidden conv | `2e-4` (PR95 Stage 8) → `5e-4` (more aggressive smoke) | `0.95` | `5e-4` | n/a |
| AdamW heads + biases | `1e-5` (PR95 Stage 8) → `3e-4` (canonical) | β1=0.9 | `0.01` | β2=0.95, eps=1e-10 |
| AdamW latents | `1e-4` (10× lr_mult) | β1=0.9 | `0.0` | β2=0.95, eps=1e-10 |

### 6.4 Composition with IGLT (the "muon+iglt" mode)

The `muon+iglt` mode runs **Muon for the first 80% of epochs** (heavy-lifting optimization on the score-aware Lagrangian), then **switches to IGLT for the final 20%** (Fisher-preconditioned Langevin polish on the float-shadow weights). This mirrors the canonical "main optimizer + Langevin polish" pattern in `src/tac/optimization/langevin_optimizer.py`'s use case discussion.

The Muon-then-IGLT composition is principled because:
1. Muon does spectral-norm steepest descent — fast convergence to a basin
2. IGLT does Fisher-preconditioned Langevin — basin shape exploration with the correct geometry
3. The two are not competing; they're sequential phases of a curriculum, exactly the pattern PR95's 8-stage curriculum uses

**Predicted Δscore for `muon+iglt`**: −0.005 to −0.015 [literature-prediction] vs adamw-only. Plain `muon` predicted −0.003 to −0.010. The IGLT polish adds another −0.002 to −0.005 on top.

### 6.5 Composition with ternary QAT (`tac.optimization.ternary_qat`)

PR95 Stage 4+ applies INT8 fake-quant during the forward; ternary QAT is the 1.58-bit extension. Both work as in-place forward-time mutations of the live weights with STE on the backward. Muon's update rule does NOT care whether the forward weight was fake-quantized — it only sees the gradient. So `muon` + `--enable-ternary-qat` compose orthogonally. **No special handling required.**

### 6.6 Composition with the just-landed mp4 codec roundtrip simulation

`tac.differentiable_eval_roundtrip.apply_mp4_codec_simulation` operates on the decoded RGB tensor before scorer ingest. Muon operates on parameter updates. They are at different layers of the pipeline; they compose with no interaction. ✓

---

## 7. Verdict per technique

| Technique | Verdict | Reactivation criteria |
|---|---|---|
| Muon for hidden conv weights | **PROMOTE** — port from PR95 to canonical `tac.optimization.muon` module; wire into time-traveler `--optimizer muon` | Empirical anchor: 1 [contest-CUDA] dispatch (~$5 Modal A100, 3000 epochs) with `muon` vs `adamw` on same substrate / same data / same seed |
| `muon+iglt` composition | **PROMOTE** — adds ~50 LOC switching logic; high EV per CLAUDE.md compose-vs-replace primitive | Same anchor + IGLT polish phase |
| Kimi RMS-matched update scale | **PROMOTE** — small patch (~10 LOC); decouples Muon LR from Adam LR | Smoke ablation showing improved LR-transferability |
| WSD schedule replacing cosine | **PROMOTE** — drop-in replacement; orthogonal to optimizer | Anchor showing comparable or better final score with smaller LR-tuning surface |
| Long-range cross-block skip | **DEFERRED-pending-council-on-substrate-edit** | Council quintet sign-off; substrate-edit-grade decision |
| BF16 activations | **PROMOTE-PENDING-TIER-1-BACKPORT** — same Tier 1 engineering wave | Tier 1 autocast backport lands (operator-gated) |
| Differentiable logit softcap | **PROMOTE-AS-SMOKE-ABLATION** — small change, easy to A/B | Same-anchor pair: `.clamp(0,255)` STE vs `30·tanh(x/30)` |
| `torch.compile()` Inductor | **PROMOTE-PENDING-TIER-2-BACKPORT** | Same Tier 1/2 engineering wave |
| ReLU² in non-final blocks | **DEFERRED-pending-council** | Council quintet sign-off (SIREN bias for boundary tasks is load-bearing) |
| SOAP-Muon (stem Linear only) | **DEFERRED-pending-Muon-anchor** | After plain Muon anchor lands; SOAP adds 1 hyperparameter + ~150 LOC, justification requires Muon to already be the standing optimizer |
| Cautious WD schedule | **DEFERRED-pending-WSD-anchor** | After WSD anchor; cautious-WD is meaningless without a non-cosine LR cooldown |

NO `KILL` verdicts. All deferrals carry explicit reactivation criteria per CLAUDE.md `KILL is LAST RESORT`.

---

## 8. Operator-routable decisions surfaced

1. **Should Muon replace AdamW for the default substrate trainer?** Recommendation: NO — Muon should be opt-in via `--optimizer muon` until an empirical anchor confirms the predicted Δscore. AdamW remains the default because (a) it's the most-tested path in the trainer skeleton, (b) PR95 used AdamW for 7 of 8 stages and only switched to Muon for Stage 8's fine-tune, (c) the canonical Muon README itself recommends Muon "for hidden layers" with AdamW for everything else — it is NOT a drop-in AdamW replacement.

2. **Authorize a `[contest-CUDA]` anchor on Muon-vs-AdamW comparison?** Recommendation: YES — cheap (~$5 Modal A100, 3000 epochs on TCNeRV or sane_hnerv substrate where both code paths are wired). Highest-EV next-step on this lane. Operator-gated.

3. **Port the canonical Muon source from `submissions/hnerv_muon/src/optim.py` to `src/tac/optimization/muon.py`?** Recommendation: YES — substrate-engineering port with the Kimi RMS-match patch already included. This unblocks every downstream substrate trainer from depending on the in-submission copy. Operator-gated; ~100 LOC + ~50 LOC tests.

4. **Build SOAP-Muon as an experimental optimizer (the operator's X post technique)?** Recommendation: NO until plain Muon has an anchor. SOAP-Muon's 1.9% step-count improvement is at NanoGPT-LLM scale (125M params, MLP weights). Our 80K-230K conv-stack regime has no published evidence. Reactivation: after plain Muon anchor + a council decision to authorize the 150 LOC SOAP integration.

5. **Tier 1 engineering wave (autocast FP16 + BF16 activations + `torch.compile`)** — already in the operator's queue per `feedback_modal_strategy_reevaluation_post_tier1_engineering_20260512.md`. Muon's BF16-native Newton-Schulz partially overlaps; the autocast wrap is still pending. Sequential with this lane, not blocking.

---

## 9. 6-hook wire-in declaration (CLAUDE.md Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default — Mandatory wire-in for every landing":

1. **Sensitivity-map contribution**: N/A — research memo, not a sensitivity-changing primitive. The downstream Muon port WILL contribute (Muon's per-param Fisher-equivalent direction is `1/√(g·gᵀ)`, distinct from Hutchinson diagonal; the per-tensor importance ranking under Muon differs from AdamW).
2. **Pareto constraint**: N/A — no archive change. The downstream Muon-trained anchor will contribute a new Pareto point on the (cost-band $/training-step × empirical_score) surface.
3. **Bit-allocator hook**: N/A — no bit allocation change.
4. **Cathedral autopilot dispatch hook**: declared — the recommended Muon-vs-AdamW dispatch becomes a candidate row in the autopilot ranking once the port lands. Estimated cost band: $5 / dispatch (Modal A100, 3000 epochs, single substrate).
5. **Continual-learning posterior update**: N/A on this memo; will trigger on every empirical anchor harvested through this lane.
6. **Probe-disambiguator**: N/A — the recommendations have a single defensible interpretation (Muon is a known well-defined optimizer); no probe needed. SOAP-Muon-vs-plain-Muon may need a probe later, but that's downstream.

---

## 10. Custody + provenance

- **Sources**: `https://github.com/KellerJordan/Muon` (HEAD `6399c65`, 2026-01-19), `https://github.com/KellerJordan/modded-nanogpt`, `https://kellerjordan.github.io/posts/muon/`, `https://arxiv.org/abs/2502.16982` (Kimi.ai Moonlight), `https://arxiv.org/abs/2409.11321` (SOAP), `https://jeremybernste.in/writing/deriving-muon`, `https://leloykun.github.io/ponder/muon-opt-coeffs/`, `https://x.com/kellerjordan0/status/2054255672636981423` (operator-directive X post, content from Google index).
- **Local Muon source in repo**: `submissions/hnerv_muon/src/optim.py` (PR95 byte-faithful port; council G provenance `896f1d79`); `data/working/upstream/submissions/hnerv_muon/src/optim.py` (intake mirror).
- **No archive built**: this memo emits zero `[contest-CUDA]` claims. All `predicted` values are `[literature-prediction]` from NanoGPT-derived speedup mapped to score via a 1:1 saturation heuristic with 2-5× scatter — non-promotable.
- **No /tmp paths**: all artifacts under `.omx/research/` and `~/.claude/projects/.../memory/` per CLAUDE.md FORBIDDEN /tmp paths.
- **No KILL verdicts**: every deferral has explicit reactivation criteria per CLAUDE.md `KILL is LAST RESORT`.

---

## 11. Next steps

1. **(Operator-gated)** Authorize the port of `submissions/hnerv_muon/src/optim.py` → `src/tac/optimization/muon.py` (canonical module). Wire into time-traveler trainer via `--optimizer muon|muon+iglt`. ~100 LOC + 50 LOC tests. Sister of `tac.optimization.iglt`.
2. **(Operator-gated)** Authorize a single Modal A100 anchor (`--epochs 3000 --optimizer muon` vs `--optimizer adamw`) on the time-traveler substrate. Cost band: $5. Predicted Δscore: −0.003 to −0.010 if technique works as advertised at our scale.
3. **(Council-grade, deferred)** SOAP-Muon for the stem Linear — requires plain Muon anchor + council quintet sign-off.
4. **(Sister lane)** Tier 1 engineering wave (autocast FP16, BF16 activations, `torch.compile`) — orthogonal but compositional with this lane.

Memo landed; lane registered. Awaiting operator decision on canonical Muon port + first anchor.
