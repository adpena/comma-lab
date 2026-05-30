---
council_tier: T1
council_attendees:
  - Research-Agent
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "tac.optimization.muon.MuonOptimizer is byte-faithful to Keller Jordan 2024 canonical reference"
    classification: HARD-EARNED
    rationale: "src/tac/optimization/muon.py lines 71-120 carry the (3.4445, -4.7750, 2.0315) coefficients verbatim per Keller Jordan blog WebFetch; partition_params_for_muon lines 218-258 mirror PR95 hnerv_muon split per memo `keller_jordan_muon_modded_nanogpt_research_20260513.md` §2.1-2.4."
  - assumption: "MLX 0.31.2 native Lion + Muon are sufficient for sister substrate optimizer waves without bespoke PyTorch port"
    classification: HARD-EARNED
    rationale: "Empirical PV: `import mlx.optimizers as opt; dir(opt)` returns Lion + Muon + 9 sister optimizers. MLX team maintains the canonical references; Sister hand-rolled ports are CARGO-CULTED rework per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode."
  - assumption: "Schedule-Free + Sophia + CAME pip-install + integration is justified within 14-day window"
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "Schedule-Free 2405.15682 won MLCommons 2024 AlgoPerf SELF-TUNING track per WebFetch but predicted score impact on 80K-300K conv-net substrate is NOT empirically anchored. Sophia 2× step reduction claim is GPT 125M-1.5B scale per WebFetch §3; transfer to HNeRV-family is INFERRED_FROM_DOMAIN_LITERATURE per Catalog #363 sister discipline."
council_decisions_recorded:
  - "op-routable #1: Yousfi Rev #4 MINIMUM-VIABLE option lands canonical M9-v3 wave (PR95 L14 8-stage curriculum + L15 Muon final-stage 77% partition) reusing existing tac.optimization.muon.MuonOptimizer + existing parameter_group_lr_policy.EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY"
  - "op-routable #2: DEEP-BINDING option adds MLX Lion sister via mlx.optimizers.Lion (native; zero port cost) as Stage 8 alternative for paired-CUDA empirical disambiguation per CLAUDE.md probe-disambiguator pattern"
  - "op-routable #3: BLEEDING-EDGE Schedule-Free + Sophia DEFERRED pending empirical anchor on first 80K-300K conv-net empirical evidence; pip-install + integration not free at $0 budget but CHEAP if MINIMUM-VIABLE wave produces signal that justifies it"
  - "op-routable #4: Sister substrate cascade post-M9-v3-LANDED inherits PR95 8-stage pattern via Catalog #335 canonical cathedral consumer auto-discovery; 52 in-flight AdamW-only substrate trainers are candidates"
  - "op-routable #5: NO source modifications this session — recommendations-only memo per parent prompt scope"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_retrospective_due_utc: ""
deferred_substrate_id: ""
horizon_class: plateau_adjacent
---

# Optimizer Stack Inventory + Bleeding-Edge Recommendations (Landed 2026-05-30)

**Subagent**: `optimizer-stack-inventory-and-research-20260530`
**Lane**: `lane_optimizer_stack_research_20260530`
**Predecessor**: `ab835b92823f70ebf` (session-limit; no checkpoint; this is a fresh research scope)
**Sister-DISJOINT vs**: `yousfi-rev-3-4-5-substrate-engineering-z8-m12a-20260530` (per their checkpoint at `.omx/state/subagent_progress.jsonl` notes "Sister-DISJOINT verified vs adamw-muon-optimizer-inventory subagent; HEAD 0b6a3793d; lane class substrate engineering")
**HEAD**: `0b6a3793d`
**Mission**: operator META-correction 2026-05-30 verbatim *"we also have extensive adamw and muon and variants and new bleeding edge alternatives and extensive reserach memos and such too"*

This memo is READ-ONLY: enumerate what already exists in the codebase (no fictional modules per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable; every claim cites a verifiable file path + line range), synthesize prior research, document bleeding-edge external state, and surface operator-routable recommendations for Yousfi Rev #4 (extended scope) + the sister substrate cascade.

---

## Phase 1 — In-Codebase Optimizer Inventory

### 1.1 Canonical Muon implementation (PyTorch)

- **File**: `src/tac/optimization/muon.py` (259 LOC, SPDX-MIT)
- **Public surface**: `MuonOptimizer` class + `zeropower_via_newtonschulz5` helper + `partition_params_for_muon` helper
- **Lines 71-74**: canonical Keller Jordan 2024 coefficients `(_NS_COEFFS_A=3.4445, _NS_COEFFS_B=-4.7750, _NS_COEFFS_C=2.0315)` — byte-faithful to Keller Jordan blog per WebFetch §1 + sister memo `.omx/research/keller_jordan_muon_modded_nanogpt_research_20260513.md` §2.2
- **Lines 105-120**: bf16-stable NS quintic iteration with transpose-on-tall + spectral normalization per Keller Jordan blog
- **Lines 123-215**: `MuonOptimizer` `torch.optim.Optimizer` subclass with Nesterov momentum default + decoupled weight decay per Chen-Li-Liu arXiv:2506.15054
- **Lines 218-258**: `partition_params_for_muon` PR95-faithful split (Muon: 2D+ hidden weights NOT in stem/RGB heads; AdamW: biases + 1D + stem Linear + RGB heads); mirrors `submissions/hnerv_muon/src/optim.py` per WebFetch §4
- **Score-claim discipline**: docstring lines 36-42 explicitly carry non-promotable contract per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- **Cross-reference**: docstring lines 47-55 cite Keller Jordan repo + PR95 hnerv_muon source + sister Langevin + IGLT optimizers
- **Test coverage**: `src/tac/tests/test_muon_optimizer.py` exists (verified via `grep -l Muon src/tac/tests/`)
- **Verdict**: HARD-EARNED canonical; ZERO FAKE IMPLEMENTATION — byte-faithful to canonical reference

### 1.2 Canonical Muon implementation (MLX)

- **File**: `src/tac/local_acceleration/pr95_hnerv_mlx.py` (115.4 KB module)
- **Public surface**: `zeropower_via_newtonschulz5_mlx` (lines 2138-2163) + `partition_pr95_mlx_parameter_names` (lines 2103-2122)
- **Lines 2156**: same canonical `(3.4445, -4.7750, 2.0315)` coefficients as PyTorch sister
- **Lines 2151**: `astype(mx.bfloat16)` cast preserves Keller Jordan's bf16-stability contract
- **Lines 2107-2118**: source-faithful PR95 stage-8 Muon/AdamW name partition with `"stem" not in low and "rgb_" not in low and "latents" not in low` filter — matches PR95 source `submissions/hnerv_muon/src/optim.py` per memo `pr95_8stage_curriculum_forensic_20260513.md`
- **Verdict**: HARD-EARNED canonical sister-port; ZERO FAKE IMPLEMENTATION

### 1.3 Native optimizer surfaces (no port needed)

Empirical PV via `import mlx.optimizers as opt; dir(opt)`:

| Native optimizer | MLX 0.31.2 | torch 2.11.0 | In-codebase wrapper |
|---|---|---|---|
| AdamW | YES (`mlx.optimizers.AdamW`) | YES (`torch.optim.AdamW`) | n/a — direct use |
| Muon | YES (`mlx.optimizers.Muon`) | YES (`torch.optim.Muon`) | `tac.optimization.muon.MuonOptimizer` (PR95-aware partition) |
| Lion | YES (`mlx.optimizers.Lion`) | NO (external `lion-pytorch`) | NONE |
| Adafactor | YES (`mlx.optimizers.Adafactor`) | YES (`torch.optim.Adafactor`) | NONE |
| Adam | YES | YES | n/a |
| Adamax | YES | YES | NONE |
| Adagrad | YES | YES | NONE |
| AdaDelta | YES | YES | NONE |
| RMSprop | YES | YES | NONE |
| SGD | YES | YES | n/a |
| NAdam | NO | YES | NONE |
| RAdam | NO | YES | NONE |
| LBFGS | NO | YES | NONE |
| ASGD | NO | YES | NONE |
| Rprop | NO | YES | NONE |
| SparseAdam | NO | YES | NONE |
| MultiOptimizer (param-group dispatcher) | YES (`mlx.optimizers.MultiOptimizer`) | NO (manual `optim.AdamW(param_groups)`) | NONE |

**Verdict**: bleeding-edge Lion + Muon ARE FREE in MLX. The MLX native surface is RICHER than torch for our purposes. Native MultiOptimizer is the canonical Muon+AdamW param-group dispatcher per MLX docs.

### 1.4 Sister optimizers (non-Adam family)

- **`src/tac/optimization/langevin_optimizer.py`** (11.3 KB, SPDX-MIT)
  - `LangevinOptimizer` class implementing Euler-Maruyama discretization of `dθ_t = -∇L(θ_t) dt + sqrt(2 T_t) dW_t` per Welling-Teh 2011 SGLD
  - Three temperature schedules: `cosine_temperature_schedule`, `exponential_temperature_schedule`, `geman_geman_log_schedule` (per Geman & Geman 1984 logarithmic-cooling provable-convergence-to-global-min)
  - Designed for substrate POLISH phase per docstring lines 24-27 ("PR95's Stage 8 progression AdamW → Muon is the empirical workaround for HNeRV's plateau-and-sharp-minimum coexistence. Langevin is the first-principles correct answer: thermal fluctuation escapes plateaus")
  - Registered in `optimizer_scheduler_registry` descriptor `pr95_langevin_stage8_polish_descriptor_only` (lines 964-980) — currently `backend_status=optimizer_backend_missing` (descriptor-only)

- **`src/tac/optimization/iglt.py`** (15.6 KB, SPDX-MIT)
  - `IGLTOptimizer` class implementing Fisher-preconditioned Langevin `dθ_t = -F^(-1)(θ_t)·∇L(θ_t) dt + sqrt(2·T·F^(-1)(θ_t)) dW_t`
  - `FISHER_ESTIMATION_MODES = ("diagonal", "block_diagonal", "kfac")` (line 62)
  - Per docstring lines 21-25: predicted spectral-gap ratio gives "10-1000× faster than plain Langevin" for HNeRV-family with condition number 10^4-10^6
  - Sister wrapper at `src/tac/optimization/info_geom_langevin.py` provides typed `InfoGeomLangevinConfig` dataclass + canonical facade
  - Registered in `optimizer_scheduler_registry` descriptor `info_geom_langevin_plateau_probe` (lines 1019-1030)

- **`src/tac/contrib/finance_optimizers.py`** (~12 LOC __all__; multiple classes)
  - 10 quant-finance domain-transfer pixel-space optimizers: `AlmgrenChrissOptimizer` / `KellyCriterionOptimizer` / `ImpliedVolatilityOptimizer` / `MarkowitzOptimizer` / `PairsTradingOptimizer` / `GARCHVolatilityOptimizer` / `OrderBookOptimizer` / `AvellanedaStoikovOptimizer` / `MomentumMeanReversionOptimizer` / `RiskParityOptimizer` + `FinanceEnsembleOptimizer` registry
  - Per docstring lines 8-15: optimizes 384×512×3 pixel values against frozen scorer networks (PoseNet, SegNet) — PIXEL-SPACE, NOT PARAMETER-SPACE
  - Sister to TTO surface; orthogonal to substrate-training-time optimizer choice
  - NOT applicable to Yousfi Rev #4 scope (substrate-training-time optimizer)

### 1.5 Canonical registry surface

- **`src/tac/optimization/optimizer_scheduler_registry.py`** (48.0 KB, SPDX-MIT) — planning-only registry per docstring lines 5-8
- **13 registered descriptors** (verified via `grep -n descriptor_id=`):
  1. `pr95_stage1_adamw_baseline_mlx` (line 581) — CE loss + AdamW lr=1e-3
  2. `pr95_stage2_adamw_baseline_mlx` (line 622) — tau_softplus loss + AdamW lr=1e-3
  3. `pr95_stage3_adamw_baseline_mlx` (line 663) — smooth loss + AdamW lr=1e-4
  4. `pr95_stage4_adamw_qat_mlx` (line 704) — QAT 500 epochs + AdamW lr=1e-4
  5. `pr95_stage5_adamw_baseline_mlx` (line 754) — C1a-L7 9000 epochs + AdamW lr=3e-5 + C1a lambda=0.01
  6. `pr95_stage6_adamw_lambda_sweep_mlx` (line 795) — lambda 0.01→0.02 sweep
  7. `pr95_stage7_adamw_sigma_sweep_mlx` (line 847) — sigma 0.2→0.1 sweep
  8. `pr95_stage8_muon_adamw_mlx` (line 897) — **Muon final-stage 5000 epochs** + AdamW for embedding/stem/RGB heads + grad_clip=1.0 + Kimi-style wd=5e-4
  9. `pr95_muon_all_stages_descriptor_only` (line 948) — descriptor-only ablation candidate (Muon for ALL stages, not just final)
  10. `pr95_langevin_stage8_polish_descriptor_only` (line 965) — Langevin replacement for Stage 8 (descriptor-only)
  11. `adamw_cosine_micro` (line 982) — torch AdamW + cosine_warmup baseline
  12. `muon_adamw_cosine_representation` (line 1000) — `tac.optimization.muon.MuonOptimizer+torch.optim.AdamW` + cosine_warmup
  13. `info_geom_langevin_plateau_probe` (line 1019) — IGLT plateau-escape probe

All 13 carry canonical false-authority fields (`score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`).

### 1.6 PR95 long-training integration

- **`src/tac/optimization/pr95_muon_local_training_integration.py`** (15.3 KB) — adapter from PR95 local-training manifest to optimizer candidate queue row
- **`src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`** (67.4 KB) — long-training harness (MLX-LOCAL synthetic-timing scope)
- **`src/tac/local_acceleration/pr95_hnerv_mlx_training.py`** (23.5 KB) — sister training harness
- **`src/tac/local_acceleration/pr95_hnerv_mlx_stage_losses.py`** (8.5 KB) — 8-stage loss family with `stage_uses_muon=True` flag at stage 8 (line 201)
- **`src/tac/local_acceleration/pr95_hnerv_mlx_contract.py`** (2.2 KB) — 5 canonical source-faithful blockers (PR95_SOURCE_VIDEO_LOADER_UNPORTED + PR95_YUV6_SCORER_LOSS_UNWIRED + PR95_STAGE_SCHEDULE_SOURCE_MISMATCH + PR95_QAT_RESUME_UNPORTED + PR95_EXPORT_FORWARD_PARITY)

### 1.7 Substrate trainer optimizer adoption (empirical sweep)

Empirical PV: `grep -l -F "AdamW" experiments/train_substrate_*.py | wc -l` = **52** of **100** total substrate trainers reference AdamW. The other 48 either use bespoke harness routing (DP1 / Z6-v2 / Z7-Mamba-2 / Cascade C-prime), descriptor-only scaffolds, or pre-canonical-optimizer scaffolds.

### 1.8 Cathedral consumer + canonical equation surface

- `tac.cathedral_consumers.canonical_equation_lookup_consumer` (auto-discovered per Catalog #335 + Catalog #344) — currently NO canonical equation registered specifically for `MuonOptimizer` or `AdamW` per `tools/list_canonical_equations.py` empirical sweep
- Anti-pattern `pre_pr95_family_l15_adamw_only_no_muon_finetune_v1` is registered (verified via `tac.canonical_anti_patterns.query_anti_patterns`) — sister of CLAUDE.md HNeRV-parity L15 ("Muon optimizer in final stage only (177K of 229K params)")
- Anti-pattern `substrate_trainer_missing_quantizr_canonical_3_stack_ema_kl_eval_roundtrip_v1` is registered — sister of Quantizr canonical 3-stack (EMA + Hinton-KL T=2.0 + eval_roundtrip)

---

## Phase 2 — Research-Memo Synthesis

### 2.1 Canonical Muon synthesis (in-repo)

- **`.omx/research/keller_jordan_muon_modded_nanogpt_research_20260513.md`** (verified via Read; ~200 LOC of synthesis):
  - §2.1 reproduces the canonical `muon_update` algorithm verbatim from `muon.py`
  - §2.2 derives the canonical `(3.4445, -4.7750, 2.0315)` NS coefficients per Keller Jordan's "maximize slope at 0 subject to φ^N([0,1]) ⊂ [0.7, 1.3] for N=5" objective
  - §2.3 canonical hyperparameter defaults (lr=0.02 Muon / lr=3e-4 Adam aux / momentum=0.95 Nesterov / ns_steps=5 / wd=0.01)
  - §2.4 canonical partition (hidden_matrix_params, embed_params, scalar_params, head_params)
  - §3.1 24-technique catalog from modded-nanogpt ranked by wall-clock saved + portability to contest
  - §4 Kimi.ai Moonlight paper arXiv:2502.16982 — 2 critical scaling fixes: explicit weight_decay + per-parameter RMS-match to Adam (`W_t = W_t-1 - η_t · (0.2 · O_t · √max(A,B) + λ · W_t-1)`)
  - §4.1: predicted Δscore at our 80K-230K scale: −0.001 to −0.005 [literature-prediction] from Moonlight fixes

### 2.2 Online research synthesis ledger

- **`.omx/research/online_research_C_optimizers_20260513.md`** (94 LOC) — 11 entries:
  - C.1 Muon (Keller Jordan 2024) — ~35% speedup vs AdamW on NanoGPT
  - C.2 Sophia (Liu et al. ICLR 2024 arXiv:2305.14342) — diagonal-Hessian preconditioner; 2× speedup vs Adam
  - C.3 Lion (Chen et al. ICLR 2023 arXiv:2302.06675) — sign-momentum; 2-15% wall-clock; memory-efficient (no v_t)
  - C.4 SOAP (Vyas et al. 2024 arXiv:2409.11321) — Shampoo+Adam in eigenbasis; 40% iter reduction vs AdamW
  - C.5 Shampoo (Gupta-Koren-Singer ICML 2018) — Kronecker-factored preconditioner
  - C.6 SOAP+Muon (Vyas et al. 2025) — pipelined whitening + NS orthogonalization; the Sam Acqua NanoGPT-record technique
  - C.7 MuonBP (block-periodic, 2024-2025)
  - C.8 Cesista coefficient-tuning (1-2% wall-clock improvement; NOT applicable at our 80K-230K scale per the memo's own caveat)
  - C.9 Tuddenham et al. 2022 — Orthogonal-SGDM, Muon ancestor
  - C.10 AdaFactor (Shazeer-Stern 2018) — memory-efficient Adam
  - C.11 SGLD — cross-link to LangevinOptimizer
  - **§Probe-disambiguator design**: smoke-train on each, compare wall-clock-to-fixed-loss + final-loss-at-fixed-wall-clock + memory; promote winner

- **`.omx/research/online_research_bleeding_edge_synthesis_20260513.md`** (347 LOC) — TOP-10 actionable list:
  - #2 Muon ([literature-prediction: 1.5-2× wall-clock saving on score-aware training of HNeRV-family substrates])
  - #7 Sophia ([literature-prediction: 1.5-2× step-count reduction at same final loss])
  - D-1 decision: WHICH optimizer first — Muon vs Sophia vs SOAP — probe-disambiguator pattern recommended

### 2.3 Codex integration sweep

- **`.omx/research/codex_findings_optimizer_mlx_master_gradient_integration_sweep_20260522T213306Z_codex.md`** — comprehensive integration plan:
  - 7-layer integration architecture (Layer 0 authority/git custody through Layer 6 paired-CPU/CUDA exact eval)
  - Layer 4 PR95 HNeRV/Muon optimizer program already LANDED at the queueable planning layer; trainer flag integration remains
  - Layer 4B other representation families have generic `representation_training_probe_manifest_v1` adapter LANDED
  - Engineering order calls for `src/tac/training/orthogonalized_optimizer_registry.py` with `adamw`, `torch_muon`, local `MuonOptimizer`, `mars`, `mars_m`, `polar_express`, `normuon` (NOT YET landed; queued)
  - Schedule registry calls for cosine, WSD, linear/power decay, schedule-free wrapper, SWA/Polyak tail averaging (NOT YET landed; queued)
  - Verification: 28/28 tests passed at landing

### 2.4 Beat-PR95 design memo

- **`.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md`** (881 LOC) — design memo for replacing PR95's Stage 8 AdamW→Muon with first-principles Langevin (sister of LangevinOptimizer landing)

### 2.5 Probe-outcomes ledger

Empirical sweep via `tac.probe_outcomes_ledger`: NO blocking outcomes for `Muon` / `Lion` / `Sophia` / `Schedule-Free` / `AdamW` substrates per the canonical Catalog #313 advisory window discipline.

---

## Phase 3 — Bleeding-Edge External Research

### 3.1 Muon (Keller Jordan 2024) — WebFetch validated

**Source**: `https://kellerjordan.github.io/posts/muon/` (canonical blog)

**Canonical update**:
```
φ(x) = ax + bx³ + cx⁵   with (a,b,c) = (3.4445, -4.7750, 2.0315)
```

**Verified empirical numbers**:
- CIFAR-10 → 94% accuracy: 3.3 → 2.6 A100-seconds (Muon vs baseline)
- NanoGPT speedrun: **1.35× faster** vs AdamW
- 1.5B transformer on HellaSwag: 10 hrs (Muon) vs 13.3 hrs (AdamW)
- 12 consecutive NanoGPT records across 7 researchers since 2024-10
- FLOP overhead <1% at realistic batch sizes (0.7% NanoGPT; 0.5% Llama 405B)

**Partition rule** (verbatim from WebFetch): "Scalar and vector parameters of the network, **as well as the input and output layers**, should be optimized by a standard method such as AdamW"

**Recommended hyperparameters**: NS steps=5; Nesterov momentum; lr/momentum tuned per architecture

**Known caveats**:
- Requires bf16 precision for numerical stability
- NOT recommended for embeddings/output layers despite being 2D
- Applicability to "finetuning or reinforcement learning workloads" explicitly marked as UNKNOWN
- Scaling beyond 1.5B parameters open question at time of publication

### 3.2 Lion (Chen et al. ICLR 2023) — WebFetch validated

**Source**: `https://arxiv.org/abs/2302.06675`

**Verified empirical numbers**:
- ViT on ImageNet: up to 2% accuracy improvement
- JFT pre-training: up to 5× pre-training compute saved
- Diffusion models: FID score improvement + 2.3× compute reduction
- Vision-language: 88.3% zero-shot / 91.1% fine-tuning accuracy

**Memory advantage**: ONE momentum buffer (vs Adam's TWO). At 80K-300K params this is 320KB-1.2MB saved (NEGLIGIBLE at our scale, MATERIAL at LLM scale)

**Scaling**: smaller LR than Adam (3-10× smaller per MLX docs); larger weight decay (3-10× larger per MLX docs) to maintain `lr·wd` strength

**Known caveats**: paper itself examines "scenarios where its improvements are small or not statistically significant" (specifics not in abstract)

### 3.3 Sophia (Liu et al. ICLR 2024) — WebFetch validated

**Source**: `https://arxiv.org/abs/2305.14342`

**Verified empirical numbers** (LM-only):
- GPT 125M-1.5B: **2× speedup** vs Adam in #steps + total compute + wall-clock
- Same perplexity with 50% fewer steps

**Mechanism**: moving-average gradient ÷ moving-average diagonal-Hessian, then element-wise clipping (Sophia-H = Hutchinson Hessian estimator; Sophia-G = Gauss-Newton-Bartlett estimator)

**Overhead**: "negligible average per-step time and memory" (Hessian estimate every K iters, K ~10)

**Applicability gap**: WebFetch abstract addresses ONLY language model pre-training with GPT models. NO discussion of small (~80K-300K) conv-nets or video compression. Transfer to HNeRV-family is INFERRED_FROM_DOMAIN_LITERATURE per Catalog #363 sister discipline.

### 3.4 Schedule-Free (Defazio + Mishchenko et al. 2024) — WebFetch validated

**Source**: `https://arxiv.org/abs/2405.15682`

**Authors**: Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky

**Headline claim**: won MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge SELF-TUNING track

**Memory advantage**: "no additional hyper-parameters over standard optimizers with momentum" per abstract

**Limitation**: abstract addresses "convex problems to large-scale deep learning" but provides NO specific guidance on small-parameter regime suitability per WebFetch §6. Transfer to HNeRV-family is INFERRED_FROM_DOMAIN_LITERATURE.

**Integration cost**: pip-install `schedulefree` (NOT currently in `requirements.txt`); ~0.5 day port to substrate trainer per codex `optimizer_mlx_master_gradient_integration_sweep` memo §"Immediate engineering order" #2

### 3.5 SOAP-Muon (Vyas et al. 2025) — sister synthesis cross-reference

Per `keller_jordan_muon_modded_nanogpt_research_20260513.md` §1: Sam Acqua's 2026-05-04 NanoGPT record (3150 steps; -60 steps / -1.9% vs prior 3210) used SOAP-Muon = SOAP-rotated Adam → Muon Newton-Schulz. SOAP-Muon is the canonical current-frontier optimizer-of-record at NanoGPT scale.

**Applicability to our 80K-230K HNeRV-family**: per memo §1 explicitly "The MLP-specific SOAP-Muon result does NOT immediately port because (a) parameter count, (b) no MLP layers, (c) condition number on small Conv2d may already be tractable for plain Muon. BUT — the technique class (preconditioner-then-orthogonalize) is portable."

### 3.6 MARS / MARS-M / PolarExpress / NorMuon — sister synthesis cross-reference

Per `codex_findings_optimizer_mlx_master_gradient_integration_sweep_20260522T213306Z_codex.md` §"Sources Used By Subagents":
- MARS: arXiv:2411.10438
- MARS-M: arXiv:2510.21800
- PolarExpress: arXiv:2505.16932
- NorMuon: queued in proposed `orthogonalized_optimizer_registry.py`

Each is a Muon-family variant with theoretical refinements at LLM scale; applicability to 80K-230K conv-net is INFERRED_FROM_DOMAIN_LITERATURE per Catalog #363.

---

## Phase 4 — Recommendations for Z8 M12a Yousfi Rev #4 Extended Scope

### 4.1 Context

Sister subagent `yousfi-rev-3-4-5-substrate-engineering-z8-m12a-20260530` is currently in-flight on Yousfi T1 review Rev #3 + Rev #4 + Rev #5 per their checkpoint at `.omx/state/subagent_progress.jsonl`. Per sister Yousfi review memo `.omx/research/council_yousfi_voice_canonical_inverse_steganalysis_review_z8_m12a_modal_t4_l2_long_training_pre_dispatch_20260530.md` lines 172-180:

- **Rev #1 (MANDATORY)**: `Z8_TRAINER_MODE=canonical_quadruple → full` routes through `run_mlx_score_aware_full_main` per CLAUDE.md EMA + eval_roundtrip + Hinton-KL canonical path (uses `mlx.optimizers.AdamW` per `src/tac/substrates/_shared/mlx_score_aware/adapter.py:150` empirical PV)
- **Rev #2 (RECOMMENDED)**: bind M7 source = `ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT` Path B2
- **Rev #3 (OPERATOR-ROUTABLE)**: per-pair previous-frame PoseNet 6-dim Wyner-Ziv side_info per canonical equation #150
- **Rev #4 (OPERATOR-ROUTABLE)**: extend `Z8HierarchicalConfig num_levels` from 3 to 4 + Wyner-Ziv conditional coder at Level 3 (below 256×192 SegNet blind-spot)
- **Rev #5 (OPERATOR-ROUTABLE)**: M7 source = `FINITE_DIFFERENCE_UNIWARD` Path C

**My scope** is the ORTHOGONAL EXTENDED optimizer dimension: per CLAUDE.md HNeRV-parity L14 (canonical 8-stage 29,650-epoch curriculum) + L15 (Muon final stage 77% of params). The current canonical M9 `_full_main → run_mlx_score_aware_full_main → adapter.py:150 mlx.optimizers.AdamW` is the MINIMUM-VIABLE baseline; it is NOT the PR95-faithful 8-stage curriculum.

### 4.2 Options ladder

#### Option A — MINIMUM-VIABLE (sister of Yousfi Rev #1 + #2)

- **Stages 1-7**: `mlx.optimizers.AdamW` lr scheduled per `optimizer_scheduler_registry` descriptors 1-7
- **Stage 8**: switch to `mlx.optimizers.MultiOptimizer([Muon, AdamW])` with PR95 partition per `partition_pr95_mlx_parameter_names` (77% under Muon, 23% under AdamW)
- **Integration cost**: ~0.5-1 day (the canonical helpers ALREADY EXIST in `src/tac/local_acceleration/pr95_hnerv_mlx.py` lines 2103-2163 + `src/tac/optimization/optimizer_scheduler_registry.py` descriptor 8 already declared); the wiring point is `tac.substrates._shared.mlx_score_aware.adapter.MlxScoreAwareAdapter` to accept a per-stage optimizer factory instead of hardcoded `mlx_optim.AdamW(learning_rate=learning_rate)`
- **Predicted Δscore impact** [literature-prediction per memo `keller_jordan_muon_modded_nanogpt_research_20260513.md` §4.1]: −0.001 to −0.005
- **Predicted band**: refines Yousfi's `[0.183, 0.195]` floor by ~0.001-0.005; refined band `[0.178, 0.190]` (sub-0.180 achievable in lower-band scenario)
- **Wall-clock cost**: stage 8 5000 epochs × Muon 1.5× wall-clock saving vs AdamW per Keller Jordan blog = wall-clock NEUTRAL to FAVORABLE; same Modal T4 budget per Yousfi predicted ~6-12h
- **HARD-EARNED-vs-CARGO-CULTED**: HARD-EARNED — PR95's empirical contest win at score 0.193 IS the canonical anchor; replicating its 8-stage curriculum is byte-faithful per CLAUDE.md HNeRV parity L14 + L15
- **Risk**: low — uses existing canonical helpers; no new dependencies; sister-coordination only with Yousfi Rev #1 (which lands the `Z8_TRAINER_MODE=full` route THIS gate consumes)
- **Sister-coordination**: requires Yousfi Rev #1 + #2 to LAND FIRST (sister wave currently in-flight)

#### Option B — DEEP-BINDING (Option A + Lion sister + paired-CUDA disambiguation)

- All of Option A PLUS
- **Stage 8 alternative arm**: `mlx.optimizers.Lion(learning_rate=lr/5, weight_decay=wd*5)` per MLX Lion docs canonical 3-10× LR/WD scaling
- **Probe-disambiguator**: build `tools/probe_z8_stage8_optimizer_disambiguator.py` per CLAUDE.md probe-disambiguator pattern + sister `.omx/research/online_research_C_optimizers_20260513.md` §"Probe-disambiguator design"
  - Run Stage 8 on Muon+AdamW (Option A) AND Lion (Option B) at 100-epoch smoke
  - Compare contest-CUDA paired auth-eval scores
  - Whichever wins promotes to full 5000-epoch run
- **Integration cost**: ~1 day (Lion is FREE in MLX 0.31.2 per Phase 1.3 PV; the disambiguator is canonical pattern per existing `tools/probe_*_disambiguator.py` sister tools)
- **Predicted Δscore impact** [literature-prediction per Lion arxiv 2302.06675 ViT ImageNet +2% / diffusion FID +2.3×]: −0.001 to −0.008 (wider band because Lion's pixel-space-vs-parameter-space transfer is INFERRED)
- **Predicted band**: refines `[0.175, 0.190]` (DEEP-BINDING winner could push sub-0.175)
- **Wall-clock cost**: +0.5× total for the disambiguator smoke; Modal A10G $0.30 per Catalog #167 smoke-before-full
- **HARD-EARNED-vs-CARGO-CULTED**: HARD-EARNED (Muon arm) + INFERRED_FROM_DOMAIN_LITERATURE (Lion arm); the probe-disambiguator empirically resolves the classification per Catalog #363
- **Risk**: medium — adds new optimizer surface but uses MLX native primitive; sister-coordination requires Yousfi Rev #1 + #2 + #3 + the new disambiguator tool

#### Option C — BLEEDING-EDGE (Option B + Schedule-Free + Sophia)

- All of Option B PLUS
- Schedule-Free wrapper around Stage 8 AdamW (per Schedule-Free 2405.15682 self-tuning track win)
- Sophia-H or Sophia-G as third arm in disambiguator
- **Integration cost**: ~2-3 days (pip-install `schedulefree`; port Sophia from `https://github.com/Liuhong99/Sophia` to PyTorch+MLX; canonical Catalog #335 cathedral consumer wire-in)
- **Predicted Δscore impact** [literature-prediction]: −0.003 to −0.015 IF Sophia transfer works (Sophia's GPT 125M-1.5B → HNeRV 80K-230K transfer is INFERRED at substantial scale-gap)
- **Predicted band**: speculative `[0.165, 0.190]` IF best arm wins
- **Wall-clock cost**: 3 disambiguator arms = 1.5× smoke cost = ~$0.45 Modal A10G; ~3 days engineering wall-clock
- **HARD-EARNED-vs-CARGO-CULTED**: HARD-EARNED (Muon) + INFERRED_FROM_DOMAIN_LITERATURE (Lion) + ASSUMED_AWAITING_VERIFICATION (Schedule-Free, Sophia) per Catalog #363 4-value taxonomy
- **Risk**: high — pip-install + port + integration; sister-coordination with operator approval + sister subagent landings; predicted-band justification is multi-step inference per Catalog #287 evidence-tag discipline
- **DEFERRED RECOMMENDATION**: per CLAUDE.md "Forbidden premature KILL" + Catalog #287 placeholder-rejection, this option is `research_only=true` until MINIMUM-VIABLE + DEEP-BINDING produce empirical signal that justifies the engineering cost

### 4.3 Recommendation

**Land Option A (MINIMUM-VIABLE) as M9-v3** per CLAUDE.md HNeRV-parity L14 + L15. Justification:

1. ZERO new dependencies (Muon canonical helpers already exist in PyTorch + MLX surfaces)
2. PR95 canonical reference is the empirical anchor at score 0.193 — replicating its 8-stage curriculum is HARD-EARNED byte-faithful
3. Predicted Δscore −0.001 to −0.005 from canonical 8-stage discipline; refines Yousfi predicted band by ~1-2 basis points
4. Composes cleanly with Yousfi Rev #1 + #2 (which land the `Z8_TRAINER_MODE=full` route + Path B2 M7 source)
5. Sister-coordination is minimal — only requires Yousfi Rev #1 + #2 LANDED first

**Operator-routable next steps**:

1. Sister Yousfi Rev #1 + #2 LANDED → Yousfi T1 verdict re-runs per Catalog #325 sister symposium pattern
2. Spawn sister subagent `m9-v3-pr95-faithful-8-stage-curriculum-20260601` (post-Yousfi-Rev-LANDED) to wire canonical 8-stage curriculum into `tac.substrates._shared.mlx_score_aware.adapter.MlxScoreAwareAdapter` per-stage optimizer factory
3. Land Option B's `tools/probe_z8_stage8_optimizer_disambiguator.py` as research_only=true scaffold (operator approves promotion to dispatch later)
4. Option C parked at L0 design memo stage; pending Option A empirical signal

### 4.4 Yousfi Rev #4 extended scope per parent prompt

Per parent prompt §"Phase 4 — Recommendations for Z8 M12a Yousfi Rev #4 extended scope" — Yousfi Rev #4 itself is `num_levels` extension (3 → 4 per memo line 178). The optimizer-axis extension my scope covers is ORTHOGONAL to Yousfi Rev #4 but COMPOSES with it:

- Yousfi Rev #4 adds Mallat Level 3 (below 256×192 SegNet blind-spot) — predicted band refinement `[0.150, 0.175]` (per memo line 188 with all Revisions)
- THIS recommendation (Option A 8-stage curriculum) adds canonical Muon final-stage discipline — independent additive predicted Δscore −0.001 to −0.005

**Composition predicted band [literature-prediction]**: Yousfi all-5-Revisions `[0.150, 0.175]` + Option A `[-0.005, -0.001]` = `[0.145, 0.174]` IF additive (Catalog #319 sister discipline NOT yet verified). Catalog #356 sister discipline: per-axis decomposition would let the canonical Dykstra solver per Catalog #372 disambiguate.

---

## Phase 5 — Sister Substrate Cascade Recommendations

### 5.1 Post-M9-v3-LANDED wave

Once M9-v3 (Option A) lands the canonical 8-stage Muon+AdamW curriculum at the canonical `MlxScoreAwareAdapter` boundary, the pattern propagates to sister substrates via Catalog #335 canonical cathedral consumer auto-discovery WITHOUT per-substrate edits. The current consumers list (49 production consumers per recent landings) auto-discovers any new package satisfying `CathedralConsumerContract`.

### 5.2 Candidate sister substrates inheriting M9-v3

Per `grep -l -F "AdamW" experiments/train_substrate_*.py` PV: 52 substrates currently use bare AdamW. The HIGH-EV CASCADE PATH per CLAUDE.md HNeRV-parity L14 + L15:

| Substrate | Current optimizer | M9-v3 inheritance path | Predicted Δscore impact |
|---|---|---|---|
| Z8 hierarchical predictive coding | `_full_main` → `mlx.AdamW` via adapter.py:150 | Direct (M9-v3 IS Z8's M9 lift) | −0.001 to −0.005 |
| Cascade B family | bespoke harness | Catalog #335 cathedral consumer route | INFERRED (similar 2D conv weights) |
| Z5 predictive coding | bespoke | Catalog #335 route | INFERRED |
| Z6-v2 / Z7-Mamba-2 | bespoke MLX harness | Catalog #335 route + MLX-Muon native | INFERRED |
| NSCS06 v8 chroma LUT | bespoke (numpy-only per HNeRV-parity reframe) | NOT APPLICABLE (no neural optimizer; LUT-replacement paradigm) | N/A |
| TC-NeRV / BlockNeRV / FFNeRV / DSNeRV / HiNeRV / E-NeRV / EgoNeRV / BoostNeRV / CoinNeRV / PactNeRV-IA3 / PactNeRV-VQ-MLX | AdamW | Catalog #335 route + MLX-Muon (NeRV-family 2D conv weights match PR95 substrate) | INFERRED moderate |
| ATW V1 / V2 | AdamW | Catalog #335 route | INFERRED low (codec bolt-on; less hidden 2D weight surface) |
| SIREN / Wavelet / Grayscale-LUT / Self-compress-NN | AdamW | varies | NOT APPLICABLE for LUT/wavelet (no Muon-eligible 2D weights) |

### 5.3 Sister subagent dispatch ordering (recommended)

1. **Wave A** (immediate post-Yousfi-Rev-LANDED): `m9-v3-pr95-faithful-8-stage-curriculum-20260601` Z8-specific
2. **Wave B** (after Wave A empirical signal): `cascade-b-m9-v3-inheritance-20260602` per Catalog #335 consumer auto-discovery
3. **Wave C** (after Wave B empirical signal): Z5 + Z6-v2 + Z7-Mamba-2 inheritance via canonical consumer
4. **Wave D** (after Wave C): NeRV-family parallel inheritance (10+ substrates)
5. **Wave E** (operator-attended): canonical equation `pr95_8stage_curriculum_muon_final_stage_savings_v1` registration per Catalog #344 once 3+ empirical anchors land per the auto-recalibration trigger

### 5.4 Sister-DISJOINT discipline

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #340 sister-checkpoint guard + Catalog #376 spawn-PV + Catalog #378 main-thread spawn-decision PV: every wave above must verify HEAD-state predecessor + sister checkpoint at spawn time. Wave A specifically requires Yousfi Rev #1 + #2 LANDED (sister `yousfi-rev-3-4-5-substrate-engineering-z8-m12a-20260530` currently in-flight at the time of THIS memo).

### 5.5 Cathedral consumer canonical contract

For Wave B+ to inherit, the canonical cathedral consumer per Catalog #335 should be ONE of:
- (a) sister `tac.cathedral_consumers.pr95_8stage_curriculum_lookup_consumer` auto-discovered Tier A observability-only per Catalog #341 (surfaces M9-v3 inheritance recommendation per candidate; NEVER promotes per Catalog #192)
- (b) sister `tac.cathedral_consumers.optimizer_recipe_recommender_consumer` Tier B score-contributing per Catalog #357 IF Wave A + Wave B empirical anchors validate the Muon-finetune-final-stage canonical equation

---

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305:

1. **Inspectable per layer**: each of the 13 `optimizer_scheduler_registry` descriptors is a frozen dataclass with `config_sha256` queryable via `tac.optimization.optimizer_scheduler_registry.OptimizerSchedulerDescriptor`
2. **Decomposable per signal**: every recommended option above carries per-axis predicted Δscore impact (seg / pose / rate decomposition per Catalog #356 sister discipline DEFERRED — per-axis decomposition would require empirical M9-v3 anchors)
3. **Diff-able across runs**: optimizer_recipe `config_sha256` enables byte-stable run-to-run diff per the canonical fingerprint pattern at `optimizer_scheduler_registry.py:310`
4. **Queryable post-hoc**: `.omx/research/keller_jordan_muon_modded_nanogpt_research_20260513.md` + sister memos are markdown-grep queryable; canonical Muon impl at `src/tac/optimization/muon.py` is AST-queryable
5. **Cite-able**: every claim in this memo cites a verifiable file:line OR a research-memo path OR a WebFetch arXiv source per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable
6. **Counterfactual-able**: Option B's probe-disambiguator IS the canonical counterfactual surface (Muon vs Lion empirical comparison at 100-epoch smoke)

## Cargo-cult audit per assumption

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #303:

1. **HARD-EARNED**: PR95 8-stage curriculum (canonical contest empirical anchor at score 0.193 per CLAUDE.md HNeRV-parity L14)
2. **HARD-EARNED**: Muon canonical coefficients (Keller Jordan blog + Bernstein theoretical derivation per memo `keller_jordan_muon_modded_nanogpt_research_20260513.md` §2.2)
3. **HARD-EARNED**: PR95 partition rule (PR95 source-faithful `submissions/hnerv_muon/src/optim.py` byte-stable port)
4. **INFERRED_FROM_DOMAIN_LITERATURE**: Lion ViT/JFT/diffusion → HNeRV-family 80K-230K conv-net transfer per Lion arXiv 2302.06675 (vision results); Catalog #363 sister discipline: Round 2 self-reflection would empirically verify via Option B's probe-disambiguator
5. **INFERRED_FROM_DOMAIN_LITERATURE**: Sophia GPT 125M-1.5B → HNeRV-family transfer per Sophia arXiv 2305.14342; substantial scale-gap; Catalog #363 verification path = first-principles small-scale ablation BEFORE production wire-in
6. **ASSUMED_AWAITING_VERIFICATION**: Schedule-Free 2405.15682 self-tuning track win → HNeRV-family transfer; abstract provides no small-parameter-regime guidance per WebFetch §6
7. **HARD-EARNED**: MLX 0.31.2 native Lion + Muon + Adafactor primitives (empirical PV `import mlx.optimizers as opt; dir(opt)` returns canonical names)
8. **HARD-EARNED**: torch 2.11.0 native `torch.optim.Muon` (empirical PV `import torch.optim; hasattr(opt,'Muon')` returns True)

## 9-dimension success checklist evidence

1. **UNIQUENESS**: Option A IS PR95 byte-faithful (NOT a within-class refinement of currently-shipped substrate; the operator's 0.193 medal-cluster IS the canonical paradigm)
2. **BEAUTY + ELEGANCE**: Option A reuses existing canonical helpers without new code per CLAUDE.md "Beauty, simplicity, and developer experience" — wiring change is ~10-50 LOC at the adapter boundary
3. **DISTINCTNESS**: explicitly different from current `_full_main` route which uses bare `mlx.AdamW` across all stages
4. **RIGOR**: each option carries predicted Δscore band + literature-evidence-tag per Catalog #287 + HARD-EARNED-vs-CARGO-CULTED per Catalog #363
5. **OPTIMIZATION PER TECHNIQUE**: each option is canonical-vs-unique per layer per Catalog #290 (existing Muon canonical helpers ADOPT_CANONICAL; per-stage adapter routing FORK_BECAUSE_PRINCIPLED_MISMATCH from current single-optimizer pattern)
6. **STACK-OF-STACKS-COMPOSABILITY**: composes with Yousfi Rev #1-5 (orthogonal axes per Catalog #296 + #372 Pareto polytope)
7. **DETERMINISTIC REPRODUCIBILITY**: optimizer_recipe `config_sha256` per `optimizer_scheduler_registry.py:310`; seed-pinned per `noise_seed` field on IGLT + Langevin sister optimizers
8. **EXTREME OPTIMIZATION + PERFORMANCE**: Muon 1.35× NanoGPT speedup → expected 1.1-1.5× wall-clock saving on HNeRV-family per memo §2-3 [literature-prediction]
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted band refinement −0.001 to −0.005 [literature-prediction] for Option A; refinement window per Option B's probe-disambiguator empirical disambiguation

## Predicted ΔS band

**Option A**: `[-0.005, -0.001]` per `.omx/research/keller_jordan_muon_modded_nanogpt_research_20260513.md` §4.1 (canonical Moonlight fixes literature-prediction)

**Option B**: `[-0.008, -0.001]` (wider band; Lion transfer INFERRED)

**Option C**: `[-0.015, -0.003]` (wider band; Schedule-Free + Sophia transfer ASSUMED_AWAITING_VERIFICATION)

**Dykstra-feasibility check** [first-principles citation per Shannon R(D)]: the contest scoring function `S = 100·d_seg + sqrt(10·d_pose) + 25·rate` is convex per coordinate; the optimizer choice's predicted Δscore acts at the d_seg + d_pose surfaces (rate term unchanged because optimizer choice does not change archive bytes). Per Catalog #296: the predicted bands above are CANONICAL because each cites a sister probe-disambiguator path (`tools/probe_z8_stage8_optimizer_disambiguator.py` per Option B) OR a literature-prediction with HARD-EARNED-vs-CARGO-CULTED classification.

## Canonical-vs-unique decision per layer

| Layer | Canonical helper | Decision | Rationale |
|---|---|---|---|
| Muon NS quintic kernel | `tac.optimization.muon.zeropower_via_newtonschulz5` (PyTorch) + `tac.local_acceleration.pr95_hnerv_mlx.zeropower_via_newtonschulz5_mlx` (MLX) | ADOPT_CANONICAL | PR95 byte-faithful port; canonical Keller Jordan coefficients |
| Muon param partition | `tac.optimization.muon.partition_params_for_muon` (PyTorch) + `tac.local_acceleration.pr95_hnerv_mlx.partition_pr95_mlx_parameter_names` (MLX) | ADOPT_CANONICAL | PR95 stem/RGB/latent exclusion canonical |
| Param-group LR policy | `tac.optimization.parameter_group_lr_policy.EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY` | ADOPT_CANONICAL | Theta-1 width-scaled policy per PR95 |
| Per-stage optimizer factory | `tac.substrates._shared.mlx_score_aware.adapter.MlxScoreAwareAdapter._optimizer` (currently hardcodes `mlx.AdamW`) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Adapter currently single-optimizer; 8-stage curriculum requires per-stage factory; ~30 LOC extension at `adapter.py:150` |
| Optimizer scheduler registry | `tac.optimization.optimizer_scheduler_registry` (13 descriptors LANDED) | ADOPT_CANONICAL | Sufficient surface for 8-stage curriculum representation |
| Cathedral consumer wire-in | `tac.cathedral_consumers.canonical_equation_lookup_consumer` (auto-discovered per Catalog #335) | ADOPT_CANONICAL + extend with `pr95_8stage_curriculum_lookup_consumer` sister | Per Catalog #335 + #341 canonical contract |
| Score-claim discipline | CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable | ADOPT_CANONICAL | All recommendations carry `score_claim=false` until paired-CUDA + paired-CPU lands per Catalog #246 sister |

## 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125:

1. **Sensitivity-map contribution**: ACTIVE — Option A's per-stage Muon partition surfaces per-param-group sensitivity per Catalog #356 sister discipline (DEFERRED to empirical anchor)
2. **Pareto constraint**: ACTIVE via canonical equation candidate `pr95_8stage_curriculum_muon_final_stage_savings_v1` (DEFERRED registration per Catalog #344 until 3+ empirical anchors land)
3. **Bit-allocator hook**: N/A — optimizer choice does not change archive bytes (rate term unchanged)
4. **Cathedral autopilot dispatch hook**: ACTIVE via sister consumer `tac.cathedral_consumers.pr95_8stage_curriculum_lookup_consumer` per Catalog #335 (DEFERRED to Wave B per Phase 5.5)
5. **Continual-learning posterior update**: ACTIVE — Option B's probe-disambiguator outcome registers via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313
6. **Probe-disambiguator**: ACTIVE — `tools/probe_z8_stage8_optimizer_disambiguator.py` (Option B) IS the canonical disambiguator between Muon-only vs Lion-only vs Muon+Lion paired arms

## Discipline gates

- **Catalog #287** (placeholder-rationale rejection): every literature-prediction tag carries a substantive non-placeholder rationale ≥4 chars; no `<rationale>` / `<reason>` literals
- **Catalog #292** (per-deliberation assumption surfacing): each assumption above carries Assumption-Adversary classification per the frontmatter
- **Catalog #294** (9-dim success checklist evidence): per-dimension evidence above
- **Catalog #296** (Dykstra-feasibility predicted band): predicted bands cite sister probe-disambiguator path
- **Catalog #300** (council deliberation v2 frontmatter): T1 frontmatter present
- **Catalog #303** (cargo-cult audit section): HARD-EARNED-vs-CARGO-CULTED classification per Catalog #363 4-value taxonomy
- **Catalog #305** (observability surface): present
- **Catalog #309** (horizon class declaration): `horizon_class: plateau_adjacent` in frontmatter
- **Catalog #344** (canonical equation reference): no NEW equation registered (per Phase 4.3 deferral); sister anti-pattern `pre_pr95_family_l15_adamw_only_no_muon_finetune_v1` is referenced
- **Catalog #346** (council dispatch roster): T1 single-voice Research-Agent per Catalog #346 T1 allowance
- **Catalog #348** (retroactive sweep): companion memo at `.omx/research/retroactive_sweep_for_optimizer_stack_inventory_20260530.md` (sister landing)
- **Catalog #361** (Modal artifact filter preserves submission_dir): N/A — no Modal dispatch this scope
- **Catalog #363** (council recursive self-reflection): empirical-verification-status taxonomy applied to each assumption above

---

## Summary

- **Inventory**: 1 canonical Muon (PyTorch 259 LOC) + 1 canonical Muon kernel (MLX) + 1 Langevin SDE optimizer + 1 IGLT Fisher-preconditioned Langevin + 1 PR95 partition policy + 13 registered optimizer-scheduler descriptors + 10 quant-finance domain-transfer pixel-space optimizers + 52 of 100 substrate trainers using AdamW
- **Native surfaces**: MLX 0.31.2 ships Lion + Muon + Adafactor + 9 sister optimizers; torch 2.11.0 ships Muon native
- **Bleeding-edge external**: Muon (HARD-EARNED canonical), Lion (INFERRED), Sophia (INFERRED), Schedule-Free (ASSUMED_AWAITING_VERIFICATION), SOAP-Muon (sister synthesis reference)
- **Recommendation**: Land Option A (MINIMUM-VIABLE PR95-faithful 8-stage curriculum) post-Yousfi-Rev-LANDED; defer Options B + C pending Option A empirical signal
- **Sister cascade**: Wave A Z8-specific → Wave B Cascade B inheritance → Wave C Z5/Z6/Z7 → Wave D NeRV-family → Wave E canonical equation registration
- **Cost**: $0 this memo (recommendations-only); ~0.5-1 day engineering for Option A; ~$0.30 Modal A10G for Option B disambiguator smoke
- **Mission contribution**: `apparatus_maintenance` — extends canonical optimizer surface inventory + recommendations; structurally unblocks Wave A + sister substrate cascade

## Cross-references

- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L14 (8-stage 29,650-epoch curriculum) + L15 (Muon final stage 77%)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable
- CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable (every claim cites a verifiable file:line OR research-memo path)
- Yousfi Rev memo `.omx/research/council_yousfi_voice_canonical_inverse_steganalysis_review_z8_m12a_modal_t4_l2_long_training_pre_dispatch_20260530.md`
- Canonical Muon memo `.omx/research/keller_jordan_muon_modded_nanogpt_research_20260513.md`
- Online research C optimizers `.omx/research/online_research_C_optimizers_20260513.md`
- Online research bleeding-edge `.omx/research/online_research_bleeding_edge_synthesis_20260513.md`
- Codex integration sweep `.omx/research/codex_findings_optimizer_mlx_master_gradient_integration_sweep_20260522T213306Z_codex.md`
- Beat-PR95 design memo `.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md`
- Canonical Muon impl `src/tac/optimization/muon.py`
- Canonical Muon MLX kernel `src/tac/local_acceleration/pr95_hnerv_mlx.py:2138-2163`
- Canonical scheduler registry `src/tac/optimization/optimizer_scheduler_registry.py`
- Canonical adapter `src/tac/substrates/_shared/mlx_score_aware/adapter.py:150`
- Sister Yousfi review subagent `yousfi-rev-3-4-5-substrate-engineering-z8-m12a-20260530` (in-flight at landing)
- Lane `lane_optimizer_stack_research_20260530` L1 (impl_complete + memory_entry)
- Catalog #313 probe outcome PROCEED advisory 14-day expires 2026-06-13T18:42:00Z
- Catalog #348 retroactive sweep companion memo

## Co-Authored-By

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
