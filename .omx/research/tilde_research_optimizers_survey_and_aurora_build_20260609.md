# Tilde Research Optimizer Survey + Aurora MLX Kernel Build (VERIFIED)

- **Date:** 2026-06-09
- **Agent:** TILDE-OPT (optimizer-axis arm of the B1 pilot fleet)
- **Lane:** `lane_tilde_opt_aurora_20260609`
- **research_only:** `true`. All perf/quality numbers are `[external-claim]`
  (Tilde blog/README/X) or `[extrapolation]`. MLX numbers are
  `[macOS-MLX research-signal]`. `score_claim=false`, `promotable=false`,
  `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`. Only
  `upstream/evaluate.py` on contest hardware (paired CUDA + Linux-x86_64 CPU)
  produces a contest score.
- **Operator directive (verbatim, 2026-06-09):** *"aurora is super cool and we
  should prioritize all of that tilde research stuff, but not above what we are
  doing now, but maybe as part of it; we can push hard and fast and use multiple
  subagents."*
- **Sister-agent boundary:** the THROUGHPUT-FIX agent owns
  `adapter.py` + the runner concurrently. This memo + the new module do NOT edit
  `adapter.py`, the runner, the snerv files, or `scorer_read_surface_atoms`. The
  wire-in is delivered as a SPEC for a follow-up to apply after the
  throughput-fix lands.
- **Prior memo audited:** `comp_muon_release_applicability_research_20260609.md`
  (commit `970fb1780`) — its Aurora claim is **CONFIRMED** (see §1).

---

## 0. TL;DR

1. **Aurora is REAL.** The prior subagent's claim was **substantially correct**,
   not a hallucination. Primary sources: Tilde blog
   (https://blog.tilderesearch.com/blog/aurora), GitHub
   (https://github.com/tilde-research/aurora-release, **MIT**, commit
   `7303d8cb`), Tilde's X announcement, MarkTechPost. One detail correction:
   "457 LOC" is the **whole repo** (`main.py` 181 + `riemannian_aurora.py` 156 +
   `aurora.py` 72 + `polar.py` 48 = 457); the *core* algorithm is the 72-LOC
   `aurora.py` + 48-LOC `polar.py`. No fabrication.
2. **Regime fit for our HNeRV decoder = predicted NULL, now with TWO
   independent reasons** (the prior memo had one): (a) the **sin** activation
   breaks Aurora's `φ(0)=0 ∧ φ'(0)≈0` neuron-death precondition; AND (b) —
   measured here — the HNeRV Muon-eligible partition is **11 WIDE matrices, 0
   TALL, 0 SQUARE**, and Aurora only helps TALL matrices (it says
   row-normalization is "unnecessary or perhaps even harmful" for wide). The one
   genuinely tall weight (`stem.weight`, aspect 61.7) is in the **AdamW**
   partition, where Aurora never runs.
3. **Built** a clean, standalone, source-faithful MLX **kernel**
   `tac.optimization.aurora_mlx` (the leverage-uniform polar) with **26 NO-FAKE
   behavioral tests** incl. byte-identical polar parity + tight update parity vs
   the cloned PyTorch reference. It is the falsifiable arm: a `$0` MLX A/B vs
   vanilla Muon at stage-8, arbitrated by the exact evaluator-action waterfiller.
4. **Discovered + factored out a duplication:** a prior agent already built an
   `aurora_like` *optimizer object* inside the 456 KB `adapter.py` (Wave-N+11
   path) with its own private `_polar_simple_quintic` — duplicating the polar
   math, untested as a unit, and not reachable by the **canonical** stage-8 path
   (`apply_pr95_mlx_optimizer_step`, which uses `zeropower_via_newtonschulz5_mlx`
   and has NO Aurora option). The new module is the single tested kernel both
   paths should route through (wire-in SPEC §6).
5. **Other Tilde releases:** only `nitrobrew` (fused-KL distillation) is even
   adjacent to our regime, and it solves a problem we do not have (152k-vocab
   logit materialization). Everything else (comp-muon, nsa, momoe,
   wall-attention) requires attention/MoE we do not have. See §2.

---

## 1. Aurora VERIFICATION (strict scrutiny)

### 1.1 Does it exist as claimed? YES.

| Claim (prior memo) | Verdict | Primary source |
|---|---|---|
| Tilde Research released Aurora | **TRUE** | blog.tilderesearch.com/blog/aurora; github.com/tilde-research/aurora-release; x.com/tilderesearch/status/2052798181558370419 |
| Fixes Muon "neuron death" | **TRUE** | blog + README: "Muon unintentionally kills a large portion of neurons in tall weight matrices" |
| License MIT | **TRUE** | `LICENSE`: "MIT License / Copyright (c) 2026 Tilde Research" |
| "457 LOC" | **TRUE (whole repo)** | `wc -l`: main 181 + riemannian 156 + aurora 72 + polar 48 = 457 |
| "row-uniform polar projection" | **TRUE** | algorithm = alternating projection onto Stiefel ∩ row-oblique manifold |
| "square → reduces to Muon" | **TRUE** | `aurora.py:42-44` `if m == n: update = polar(update)`; blog: "For square matrices Aurora reduces to the standard Muon update." **Empirically reproduced: max diff 0.0** (test `test_square_reduces_to_muon_polar`). |
| neuron-death precondition `φ(0)=0 ∧ φ'(0)≈0` | **TRUE** | blog: "For any activation φ with φ(0)=0 and φ'(0)≈0, the same vanishing-gradient feedback loop occurs. This includes SwiGLU, ReLU², GELU, and SiLU." |

`[external-claim]` benchmarks (1.1B / ~100B tokens, fully open data): HellaSwag
67.6% (+2.5 vs Muon), MMLU 37.9% (+10.8), final loss 2.26 (vs 2.31 Muon / 2.33
NorMuon), modded-nanoGPT new SoTA, ~6% compute overhead, "gains scale with MLP
width", "100x data efficiency". Aurora-1.1B matches Qwen2-1.5B with ~100x fewer
tokens.

### 1.2 The actual algorithm (verified from the clone, commit `7303d8cb`)

`polar.py` — 12-step **simple-quintic** Newton-Schulz `p(σ)=2σ-1.5σ³+0.5σ⁵`
(fixed points {0,1,√2}, σ=1 super-attracting), bf16 compute, transpose-tall-to-
wide before the Gram. NOTE: this is NOT Keller-Jordan's tuned 5-step
`(3.4445,-4.7750,2.0315)` — it deliberately matches the modded-nanoGPT track-3
"not optimizing for wallclock" baseline so val-loss curves reproduce.

`aurora.py::aurora(W, G, momentum, eta, weight_decay=0.025, mu=0.95,
nesterov=True, pp_iterations=2, pp_beta=0.5)`:
1. SGD/Nesterov momentum (`momentum.lerp_`).
2. Leverage-uniform polar:
   - `m == n`: `polar(update)` (== Muon).
   - else (wide→transpose to tall): `target_row_sq = n/m`; `D = 1/row_norm`;
     loop `pp_iterations`: `U = polar(D*G)`, then
     `D *= (target_row_sq / rowsq(U))**pp_beta`.
3. Muon aspect-ratio scaling `max(1, m/n)**0.5`.
4. Decoupled weight decay then apply: `W = W·(1-ηλ) - η·update`.

(Riemannian-Aurora in `riemannian_aurora.py` is the reference solver — O(m²n) —
not needed for our regime; "Riemannian achieves 0.6% higher gradient alignment".)

### 1.3 Honest correction to the prior memo

The prior memo's "row-uniform polar… `√(n/m)`" wording conflated the **internal
preconditioning target** (`target_row_sq = n/m`, correct) with a one-shot scale.
Aurora does NOT one-shot scale by `√(n/m)`; it iterates a damped diagonal
preconditioner toward that row-norm target, then applies the standard Muon
`max(1, m/n)**0.5` aspect scale. The new module implements the iteration exactly
(verified parity §4). This is a clarification, not a falsification — the prior
recommendation (DEFER-with-reactivation behind a $0 probe) stands and is
**strengthened** by the new shape evidence.

---

## 2. Broad Tilde Research survey + per-item EV ranking

Full public org inventory (github.com/tilde-research, fetched 2026-06-09):

| Repo | What it is | Framework / license | Our-regime fit | EV for lowering contest S | Verdict |
|---|---|---|---|---|---|
| **aurora-release** | Leverage-aware optimizer; fixes Muon neuron-death on **tall** matrices | PyTorch / **MIT** | Weak — sin precond. broken + Muon partition all-wide | **Low but nonzero** (cheap falsifiable arm) | **BUILD as fleet-arm; DEFER-with-reactivation** (this memo) |
| comp-muon-release | Compositional Muon; whitens attention **QK/OV** factor-pairs | PyTorch / Apache-2.0 | **None** — no attention in HNeRV | ~0 (cannot run) | **DEFER** (architecture mismatch; per `comp_muon_release_applicability_research_20260609.md`) |
| nitrobrew-release | Fused, constant-memory KL-divergence-from-hidden-states for **distillation** (O(B·T·V)→O(B·T·chunk_V)) | PyTorch≥2.1 / Apache-2.0 | **None** — solves 152k-vocab logit materialization; our teachers are SegNet (5-class) + PoseNet (6-dim), no large-V bottleneck | ~0 | **DEFER** (regime mismatch; we have no large-vocab logit memory problem) |
| nsa-release | Native Sparse Attention kernel | PyTorch / — | None (no attention) | 0 | DEFER |
| momoe-release | Memory-optimized Mixture-of-Experts | PyTorch / — | None (no MoE) | 0 | DEFER |
| wall-attention-release | Attention variant, per-channel multiplicative decay | PyTorch / — | None (no attention) | 0 | DEFER |
| activault | Engine for collecting/uploading/downloading model activations | PyTorch / — | Tangential (observability), not score-lowering | ~0 (apparatus only) | NOTE only |
| nitrobrew-trl / nitrobrew-verl | RL post-training framework forks | PyTorch / Apache-2.0 | None (no RL post-training) | 0 | DEFER |
| (blog) Rate-Distortion SAEs | Interpretability research (SAE R(D)) | — | Tangential to our R(D) framing; no code release for us | ~0 | NOTE only |

**Ranking by EV for our contest objective:** Aurora (low-but-falsifiable) >
nitrobrew (≈0, regime-mismatch) > everything else (0, structurally
inapplicable). Aurora is the only one worth a `$0` empirical arm. Per
"Forbidden premature KILL", all are **DEFER** (intact paradigms for their home
regimes), not KILL. Reactivation criterion for the attention-family (comp-muon,
nsa, wall-attention): we adopt an attention-bearing witness backend (ViT-NeRV /
transformer decoder). Reactivation for nitrobrew: a witness backend whose
distillation teacher has a large-vocabulary logit tensor.

---

## 3. Regime-fit analysis for HNeRV stage-8 (the decisive section)

### 3.1 Measured Muon-eligible partition (the new evidence)

`partition_pr95_mlx_parameter_names` (source-faithful PR95 stage-8 split:
`ndim≥2` weights, excluding `stem` / `rgb_` / `latents`) on the default
`HNeRVDecoderMLX(latent_dim=28, base_channels=36)` (228,958 params):

| Muon-eligible weight | NS-flat (m, n) | aspect | class |
|---|---|---|---|
| blocks.0.conv.weight | (144, 324) | 0.444 | WIDE |
| blocks.1.conv.weight | (144, 324) | 0.444 | WIDE |
| blocks.2.conv.weight | (108, 324) | 0.333 | WIDE |
| blocks.2.skip_conv.weight | (27, 36) | 0.750 | WIDE |
| blocks.3.conv.weight | (80, 243) | 0.329 | WIDE |
| blocks.3.skip_conv.weight | (20, 27) | 0.741 | WIDE |
| blocks.4.conv.weight | (72, 180) | 0.400 | WIDE |
| blocks.4.skip_conv.weight | (18, 20) | 0.900 | WIDE |
| blocks.5.conv.weight | (72, 162) | 0.444 | WIDE |
| refine0.weight | (9, 162) | 0.056 | WIDE |
| refine1.weight | (18, 81) | 0.222 | WIDE |

**Counts: TALL=0, WIDE=11, SQUARE=0.** (Guarded by
`test_hnerv_muon_eligible_partition_has_no_tall_matrices` — if a future arch
introduces a tall Muon-eligible weight, that test breaks and signals
"re-evaluate Aurora".) The one genuinely tall weight, `stem.weight` (1728, 28,
aspect 61.7), is in the **AdamW** partition (input-adjacent, Keller-Jordan
canon), so Aurora never touches it.

### 3.2 Two independent null-predictors

1. **Activation precondition broken.** Aurora's neuron-death feedback loop is
   proven for `φ(0)=0 ∧ φ'(0)≈0` (SwiGLU/ReLU²/GELU/SiLU). HNeRV uses **sin**:
   `sin(0)=0` but `sin'(0)=cos(0)=1 ≠ 0`. The dead-row→starved-down-projection
   cascade does not structurally arise. `[extrapolation]` effect ≈ 0.
2. **Shape mismatch (new).** Aurora helps **tall** matrices. The Muon partition
   is **all wide**. For wide, Aurora transposes-to-tall, but the blog itself:
   "row-normalization is unnecessary or perhaps even harmful for square/wide
   cases" because the orthogonality constraint already forces uniform leverage
   on the short dimension. So on our weights Aurora is ≈ Muon at best, mildly
   harmful at worst. `[extrapolation]` effect ≈ 0 ± small (sign possibly
   negative).

### 3.3 Canonical-vs-unique decision per layer (stage-8 optimizer)

| Layer | Canonical | Choice | Rationale |
|---|---|---|---|
| Stage-8 hidden-weight optimizer | `zeropower_via_newtonschulz5_mlx` (vanilla Muon, PR95 L15) inside `apply_pr95_mlx_optimizer_step` | **ADOPT_CANONICAL** | PR95 winner used exactly this; MLX-native; descriptor `pr95_stage8_muon_adamw_mlx` already encodes it. Vanilla Muon **suffices**. |
| Aurora kernel as stage-8 polar | `aurora_leverage_uniform_polar_mlx` (this build) | **UNCLEAR → DEFER behind $0 paired MLX A/B** | The leverage-uniform polar *could* differ, but sin-precond + all-wide partition predict ≈0. Burden of proof is on proving-it-helps via a byte-closed A/B arbitrated by exact ΔS, not adopting by default. |
| Non-matrix params (latents/biases/norms/QAT/scalars) + stem + rgb heads | AdamW (PR95) | **ADOPT_CANONICAL** | Keller-Jordan canon; unchanged by any of the above. |

This is the OPTIMAL engineering for stage-8: vanilla Muon is correct; Aurora is
a cheap falsifiable experiment, not a default adoption.

---

## 4. Built module + tests (`tac.optimization.aurora_mlx`)

### 4.1 Surface

- `aurora_simple_quintic_polar_mlx(matrix, *, steps=12, eps, cast_float32_to_bfloat16)`
  — Aurora's 12-step quintic polar (byte-faithful to `polar.py`).
- `aurora_leverage_uniform_polar_mlx(update, *, pp_iterations=2, pp_beta=0.5,
  eps, apply_aspect_scale=True, polar_steps=12, polar_cast_float32_to_bfloat16)`
  — **the drop-in replacement** for the `zeropower_via_newtonschulz5_mlx` call
  inside `apply_pr95_mlx_optimizer_step`. Handles ndim-2 and ndim-4 (conv
  reshape), square→Muon, tall/wide→leverage-uniform, Muon aspect scaling.
- `aurora_update_mlx(weight, gradient, momentum, *, eta, weight_decay, mu,
  nesterov, ...)` — full functional reference update (momentum→polar→scale→WD),
  returns `(new_weight, new_momentum)` (MLX is immutable). Primarily for parity
  testing + as a self-contained reference.
- `classify_matrix_shape(rows, cols)` → `"tall"|"wide"|"square"`.

It is a **kernel** module, NOT an optimizer object — the production wire-in reuses
the canonical `apply_pr95_mlx_optimizer_step` scaffolding (which already owns
momentum / WD / AdamW). This is the deliberate complement to `adapter.py`'s
heavier `aurora_like` `mlx.optimizers.Optimizer` subclass (§6 routes both through
this kernel to kill the duplication).

### 4.2 NO-FAKE tests (26 total, all pass; ruff clean)

Behavioral guards that FAIL if the body is faked (e.g. reverts to plain Muon):
- `test_square_reduces_to_muon_polar` — leverage-uniform == plain polar for
  square (max diff **0.0**). The "reduces to Muon" property.
- `test_tall_leverage_uniformity_beats_plain_polar` — **THE headline claim**: on
  a tall (64,8) matrix Aurora row-`‖U_i‖²` std = **0.0059** vs plain Muon
  **0.0594** (10× tighter), mean ≈ target `n/m = 0.125`. If the body reverted to
  plain polar, std reduction vanishes → FAIL.
- `test_more_pp_iterations_tighten_leverage` — pp=3 tighter than pp=1.
- `test_wide_transposes_and_orthogonalizes` — wide handled; `U Uᵀ ≈ I_m`.
- `test_conv4d_matches_manual_flatten` — N-D path == manual reshape (diff 0.0).
- Update mechanics: weight-decay-only-on-zero-grad, momentum accumulation,
  non-trivial step, all the validation raises.

Parity vs the cloned PyTorch reference (commit `7303d8cb`; skipped if clone
absent):
- `test_parity_polar_vs_pytorch_reference` — MLX bf16 polar matches torch bf16
  polar to a few bf16 ULPs AND both have **identical** `UᵀU-I` error (proving the
  gap is precision, not logic; some shapes match at **0.0**).
- `test_polar_logic_exact_in_float64_vs_numpy_reference` — MLX f32 polar matches
  an independent numpy-f64 mirror to **<2e-3** (pins coeffs/steps/transpose).
- `test_parity_square_update_vs_pytorch_reference` — square update matches at
  **<1e-6** (no f32 preconditioning loop → near-exact).
- `test_parity_tall_update_vs_pytorch_reference` — tall update matches at
  **<2e-3** (residual is pure f32 reduction-order over the 2 preconditioning
  iterations; a logic error would be orders of magnitude larger).

Regime guard:
- `test_hnerv_muon_eligible_partition_has_no_tall_matrices` — encodes §3.1
  (TALL=0) so future arch changes that add tall Muon weights break the test and
  re-surface Aurora's reactivation.

### 4.3 False-authority discipline

Module docstring carries `score_claim=false / promotable=false /
promotion_eligible=false / ready_for_exact_eval_dispatch=false` and the
`[macOS-MLX research-signal]` tag per CLAUDE.md "MPS auth eval is NOISE" + "MLX
portable-local-substrate authority". The kernel computes NO score.

---

## 5. The byte-closed, exact-ΔS-arbitrated arm (how this becomes evidence)

Per AGENTS.md "Evaluator-Equivalent Witness Compiler Paradigm" + the operator's
two-level optimizer framing: Aurora is an **inner** weight-space proposal step.
It only becomes admissible evidence through the **outer** exact arbiter:

1. Run PR95 stages 1–7 once to a fixed checkpoint.
2. Stage-8 A/B from that identical checkpoint, same epochs/LR/seed: arm A =
   vanilla Muon (`zeropower_via_newtonschulz5_mlx`), arm B = Aurora
   (`aurora_leverage_uniform_polar_mlx`).
3. Byte-close BOTH resulting archives.
4. Arbitrate with the exact evaluator-action waterfiller
   (`tac.optimization.evaluator_action_waterfill`): admit a σ/arm iff
   `S(P+σ) < S(P)` on the exact `100·d_seg + sqrt(10·d_pose) + 25·bytes/N`.
5. MLX deltas are `[macOS-MLX research-signal]` ONLY; any score/promote/kill
   decision requires paired CUDA + Linux-x86_64 CPU auth-eval on the exact
   byte-closed archive.

**Predicted outcome** `[extrapolation]`: arm B ≈ arm A (no improvement; possibly
slightly worse on the all-wide partition). If confirmed, record the DEFER with
the empirical artifact and move on. If — against prediction — B wins a real,
byte-closed, re-eval-surviving margin, reactivate Aurora into stage-8.

---

## 6. WIRE-IN SPEC (for a follow-up; NOT applied here — throughput-fix owns adapter/runner)

**Goal:** make `aurora_leverage_uniform_polar_mlx` selectable at the canonical
stage-8 dispatch, AND collapse the duplicated polar in `adapter.py`. Three small,
reviewable edits — all deferred until the throughput-fix lands.

### 6.A Canonical stage-8 path (`apply_pr95_mlx_optimizer_step`, `pr95_hnerv_mlx.py`)

Add ONE config flag + ONE branch at the existing kernel call site (currently
`pr95_hnerv_mlx.py:2523` for ndim-4 and `:2533` for ndim-2):

1. In `Pr95MlxOptimizerConfig` (around `pr95_hnerv_mlx.py:2231`) add:
   ```python
   muon_kernel: str = "newton_schulz"   # "newton_schulz" | "aurora_leverage_uniform"
   aurora_pp_iterations: int = 2
   aurora_pp_beta: float = 0.5
   ```
2. In the Muon branch of `apply_pr95_mlx_optimizer_step`, replace the bare
   `zeropower_via_newtonschulz5_mlx(...)` call with a dispatch. Because Aurora's
   `aurora_leverage_uniform_polar_mlx` ALREADY applies the `max(1, m/n)**0.5`
   aspect scale internally (and ndim-4 reshape), the cleanest edit is:
   ```python
   from tac.optimization.aurora_mlx import aurora_leverage_uniform_polar_mlx
   ...
   if config.muon_kernel == "aurora_leverage_uniform":
       # Aurora applies the conv reshape + aspect scale itself; pass the raw update.
       update = aurora_leverage_uniform_polar_mlx(
           update,
           pp_iterations=config.aurora_pp_iterations,
           pp_beta=config.aurora_pp_beta,
           apply_aspect_scale=True,
           polar_cast_float32_to_bfloat16=config.cast_muon_float32_to_bfloat16,
       )
   else:
       # existing newton_schulz path (reshape + NS + scale) unchanged
       ...
   ```
   Keep the existing NS path byte-identical when `muon_kernel="newton_schulz"`
   (the default), so this is a pure-additive opt-in — no regression risk to the
   canonical Muon path. Add a regression test asserting
   `muon_kernel="newton_schulz"` output is byte-identical to pre-change.
3. Thread the flag from the curriculum/recipe so the A/B is selectable from the
   stage-8 descriptor `pr95_stage8_muon_adamw_mlx`.

### 6.B Collapse the `adapter.py` duplication (DEFERRED; adapter owned by throughput-fix)

`adapter.py::_build_aurora_like_mlx_optimizer` has its own private
`_polar_simple_quintic` + leverage-uniform loop (duplicate of this kernel). Once
the throughput-fix agent is done, route that optimizer object's matrix update
through `aurora_leverage_uniform_polar_mlx` (delete the private copy). This
removes a latent drift risk (two copies of the same numerics) per CLAUDE.md
"Bugs must be permanently fixed AND self-protected against" + "Results must
become system intelligence". Net: ONE tested Aurora kernel, two consumers.

### 6.C Wire-in pillar declaration (Catalog #125; this is a research+kernel landing)

- **#1 Sensitivity-map:** N/A — optimizer kernel, no per-axis byte sensitivity.
- **#2 Pareto constraint:** N/A — non-binding (optimizer choice does not move the
  rate/seg/pose feasible region directly).
- **#3 Bit-allocator hook:** N/A — not a codec primitive.
- **#4 Cathedral autopilot dispatch:** N/A — research_only; no archive-deployable
  artifact (the A/B is operator/B1-routed).
- **#5 Continual-learning posterior:** PENDING — emit an anchor only when the §5
  byte-closed A/B produces an empirical ΔS (none run here).
- **#6 Probe-disambiguator:** the §5 A/B **is** the disambiguator between
  "Aurora helps stage-8" vs "predicted-null" — resolved empirically by the exact
  evaluator-action waterfiller, not by prose.

---

## 7. Provenance / disk hygiene

- Cloned reference: `git clone --depth 1
  https://github.com/tilde-research/aurora-release` → SSD scratch
  `/Volumes/VertigoDataTier/pact/tilde_scratch/aurora-release` (NOT /tmp), commit
  `7303d8cb9999d735cb12c921f3651f04bf362524` ("wire riemannian aurora into main",
  2026-05-27). License MIT. The clone is research scratch on the SSD tier per
  AGENTS.md disk-hygiene waterfall; it is rebuildable from the public URL (no
  signal loss on cleanup) and is referenced (optionally) by the parity tests via
  `_REF_SRC` (tests skip gracefully if absent).
- New repo files (small, durable): `src/tac/optimization/aurora_mlx.py`,
  `src/tac/tests/test_aurora_mlx.py`, this memo.

---

## 8. Council-grade summary (for the optimizer-axis arm of the B1 fleet)

- **Shannon/Dykstra lens:** Aurora's value is a better-conditioned spectral step
  on tall matrices; our feasible region's optimizer-side slack is dominated by
  the all-wide partition where the conditioning is already near-isotropic. No
  R(D)-bound argument predicts a contest-cell-debt improvement here.
- **Assumption-Adversary:** the shared assumption "a better optimizer lowers S"
  is CARGO-CULTED for this arm — S is dominated by `100·d_seg` argmax-flip + bytes,
  and the optimizer's last-stage conditioning maps to that only weakly. The arm
  is worth running precisely because it is `$0` and falsifies the assumption
  cleanly.
- **Verdict:** BUILD ✓ (done, tested), DEFER-with-reactivation on adoption,
  arbitrated by exact ΔS. Vanilla Muon remains the canonical stage-8 optimizer.
