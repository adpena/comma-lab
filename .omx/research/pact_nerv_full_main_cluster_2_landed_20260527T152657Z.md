# PACT-NeRV `_full_main` implementation cluster 2 — LANDED 2026-05-27

**Subagent:** `pact_nerv_full_main_impl_2` (successor of `pact_nerv_full_main_impl_1`)
**Lane (umbrella):** `lane_pact_nerv_full_main_cluster_2_20260527`
**Commits:** `6651189bd` (batch 1) + `142940946` (batch 2) + `9077d19bb` (batch 3)
**Wall-clock:** ~1h. **GPU spend:** $0 (all CPU-local validation; full training paths are CUDA-gated + paid-GPU-gated per Catalog #325).

## What this wave did

Implemented real score-aware `_full_main` training paths for the **remaining 12
PACT-NeRV substrate variants** queued by predecessor `pact_nerv_full_main_impl_1`
(`.omx/research/pact_nerv_full_main_implementation_cluster_landed_20260527.md`).
The predecessor landed the canonical shared base
`src/tac/substrates/_shared/pact_nerv_full_main.py` + the top-6 variants (ia3,
distilled_scorer, neural_codec_e2e, mamba, selector_v2, vq). This wave drains the
long-tail 12, completing the full 18-variant PACT-NeRV family.

Each variant's `_full_main` now routes through the canonical shared base
(`run_pact_nerv_score_aware_training`) + the variant-specific architecture +
score-aware loss + `pack_archive`. No `raise NotImplementedError`, no `pass`, no
relocated NotImplementedError — verified AST-level 0 in every `_full_main` body.

**CRITICAL — code complete, PAID DISPATCH still gated.** Per the Catalog #325
resolution ("implement all without firing council-gated paid paths"): every
variant's `_full_main` is now a REAL training loop, but every recipe stays
`dispatch_enabled: false` + `research_only: true` (verified all 12). The full
path is CUDA-required (`device_or_die` rejects MPS/CPU per Catalog #1) and
refuses on CPU with `SystemExit` — so no paid GPU can fire until each substrate's
per-substrate symposium clears it (Catalog #325). This is the trigger that stays
gated, not the code.

## The 12 variants landed (0 NotImplementedError verified)

| Variant | Distinguishing feature | Archive extras | Loss extra | Batch |
|---|---|---|---|---|
| `pact_nerv_selector_v3` | Rice-Golomb-coded FEC6 selector (Golomb 1966 + Rice 1971) | selector_bytes + palette_size | — | 1 |
| `pact_nerv_selector_v4` | Run-length-coded FEC6 selector | selector_bytes + palette_size | — | 1 |
| `pact_nerv_multi_modal` | pose+class-prior+odometry fusion conditioning | pose + class_prior + odometry tensors | — | 1 |
| `pact_nerv_diffusion_trajectory` | per-pair diffusion trajectory predictor over seeds | seeds + num_timesteps | — | 1 |
| `pact_nerv_ia3_multi` | multi-block IA3 γ-only + per-pair difficulty conditioning (Liu 2022) | ego_poses + difficulties + pose_dim + difficulty_dim | — | 2 |
| `pact_nerv_asymmetric_boundary` | asymmetric boundary-FiLM after all blocks | boundary_signals + boundary_signal_dim | — | 2 |
| `pact_nerv_moe` | pose-conditioned mixture-of-experts routing | latents-only | load_balance_aux (positional) | 2 |
| `pact_nerv_bayesian` | Bayesian variational latents (Blundell 1505.05424) | latents-only | kl_div (kw) | 2 |
| `pact_nerv_neural_codec_e2e_cross` | dual-branch hyperprior-gated E2E composition (Ballé 2018) | branch_a/b + hyperprior + latents_a/b | gate_values (kw) | 3 |
| `pact_nerv_cross_codec_a` | fec6 base + Pact-NeRV side-info residual composition | fec6_base_bytes + selector_bytes + composition_alpha | — | 3 |
| `pact_nerv_cross_codec_b` | PR106 base + IA3 side-info residual composition | pr106_base_bytes + ego_poses + score_index_bytes + score_table_size | — | 3 |
| `pact_nerv_diffusion_distilled` | 1-step diffusion-distilled student decoder | student_state + teacher_num_timesteps | — | 3 |

Each `_full_main` is REAL: decode real contest video (Catalog #114) → patch
yuv6 BEFORE scorer construction (eval_roundtrip non-negotiable) →
`load_differentiable_scorers` (no scorer at inflate) → variant score-domain
Lagrangian via Catalog #164 `score_pair_components_dispatch` → EMA shadow
(Quantizr 0.997) → best-EMA checkpoint → variant `pack_archive` → contest-
compliant numpy/PIL runtime (Catalog #146 + #295) → CUDA auth-eval via canonical
`gate_auth_eval_call` (Catalog #226) → `posterior_update_locked` (Catalog #128).

## Canonical-vs-unique decision per layer (Catalog #290)

Identical to the shared-base wave (the decision is uniform across the 18-variant
family because the distinguishing primitive is the architecture/loss, not the
training scaffold):

| Layer | Decision | Rationale |
|---|---|---|
| train/val loop + EMA + NaN watchdog | ADOPT_CANONICAL (`run_pact_nerv_score_aware_training`) | substrate-AGNOSTIC; identical to implemented ds_nerv/ia3/vq sisters |
| `decode_real_pairs` / `device_or_die` / scorer load / GTScorerCache | ADOPT_CANONICAL | already shared across all substrates (`trainer_skeleton`) |
| contest runtime emission + deterministic archive zip + weight-byte proxy | ADOPT_CANONICAL (`write_contest_runtime` / `build_archive_zip` / `closed_form_weight_byte_proxy`) | substrate-AGNOSTIC packet plumbing per Catalog #146/#19/#295 |
| architecture (`architecture.py`) | FORK (per-package) | the distinguishing primitive (selector coder / MoE router / Bayesian layers / hyperprior gate / cross-codec base / diffusion student / IA3-multi γ stack / boundary FiLM / multi-modal fusion / trajectory predictor) |
| archive grammar (`archive.py` `pack_archive`) | FORK (per-package) | each variant's `pack_archive` signature differs (selector_bytes / branch state dicts / base bytes + side-info / seeds / etc.) |
| numpy/PIL inflate (`inflate.py`) | FORK (per-package) | already landed per HNeRV parity L4 (≤200 LOC, no scorer imports, no MLX dep) |
| score-aware loss (`score_aware_loss.py`) | FORK (per-package) | variant-specific extra terms threaded via the `compute_loss` callback (load_balance_aux / kl_div / gate_values; standard for the rest) |

## MLX-first directive note (8th standing directive)

Per the predecessor's analysis (preserved): the PACT-NeRV family is an EXISTING
PyTorch architecture family (all 18 `architecture.py` are `torch.nn.Module`; the
canonical score-aware loss routes through PyTorch SegNet/PoseNet via
`load_differentiable_scorers` + Catalog #164). The OPTIMAL ENGINEERING for these
PyTorch substrates is the canonical PyTorch training loop (the implemented
sister ds_nerv/ia3/vq use it). The INFLATE path is already numpy/PIL-portable
(no MLX dep) per the 8th directive's portability requirement. The 8th MLX-first
directive applies to NEW substrates; migrating 18 existing architectures to MLX
is an architecture migration, not a `_full_main` implementation.

## Variants that needed a genuinely distinct architecture/loss vs the shared base

The shared base's `compute_loss` callback contract absorbed every variant
cleanly. Three variants needed a distinct extra loss-term threaded through the
callback (still inside the canonical loop — the callback is the canonical
extension point):

- **moe** — `model._last_load_balance_aux` (Shazeer-style load-balancing aux,
  passed POSITIONAL as the 6th arg to the moe loss `forward`).
- **bayesian** — `model.last_kl_div` (Blundell variational KL, passed as `kl_div=`
  kwarg).
- **neural_codec_e2e_cross** — `model.gate_values(idx)` (hyperprior gate-entropy
  term, passed as `gate_values=` kwarg).

Two variants needed a custom Config construction:
- **neural_codec_e2e_cross** — Config takes `latent_dim_a`/`latent_dim_b` (dual
  branches), NOT a single `latent_dim`; the trainer constructs both from
  `args.latent_dim`.

All others (selector_v3/v4, multi_modal, diffusion_trajectory, ia3_multi,
asymmetric_boundary, cross_codec_a/b, diffusion_distilled) use the standard
callback (the architecture's buffer-backed conditioning — ego_poses, difficulties,
boundary_signals, pose/class/odometry, seeds — is internal to
`model(pair_indices)`, so the callback is plain).

## Validation evidence

- **AST 0-NotImplementedError**: verified per-`_full_main` for all 12 (count=0).
- **171 package tests pass** across all 12 packages (`pytest` green).
- **Loss-callback wiring verified end-to-end** with REAL scorers + 1 backward
  for the 6 trickiest variants (moe positional aux / bayesian kl / ncec gate /
  asymmetric / cross_codec_a / ia3_multi) — all finite + backward_ok, matching
  the canonical loop's all-None GT-forward cache path.
- **Variant forward + extra-term accessors** verified for all 8 non-standard
  variants (`_last_load_balance_aux`, `last_kl_div`, `gate_values(idx)`).
- **Ruff clean** on all 12 trainers (3 pre-existing C416 in the ncec test file
  are sister-pre-existing, not introduced by this wave).
- **CUDA-gated**: each `test_trainer_full_main_implemented_and_cuda_gated`
  asserts `_full_main(--device cpu)` raises `SystemExit` (device_or_die) + that
  `run_pact_nerv_score_aware_training` is in the body + no `raise NotImplementedError`.
- **Catalog #325 gating intact**: all 12 recipes `dispatch_enabled: false` +
  `research_only: true` (asserted by `test_recipe_research_only_and_dispatch_disabled`).

## 6-hook wire-in declaration (Catalog #125)

Identical posture to the predecessor's 6 (hooks declared in each
`SubstrateContract` at L0 SCAFFOLD landing 2026-05-20; this wave makes hooks #4
+ #5 ACTIVE-pending-anchor by giving each `_full_main` a real archive-emitting +
posterior-updating path):

- **Hook 1 (sensitivity-map):** `not_applicable_with_rationale` — no signal until paid dispatch.
- **Hook 2 (Pareto constraint):** `rate_distortion_v1` (the score-domain Lagrangian's rate + seg + sqrt(pose) terms).
- **Hook 3 (bit-allocator):** `not_applicable_with_rationale` — fp16 brotli on combined weight blob.
- **Hook 4 (cathedral autopilot dispatch):** ACTIVE-pending-anchor — `_full_main` emits byte-stable `archive.zip`.
- **Hook 5 (continual-learning posterior):** ACTIVE-pending-anchor — `posterior_update_locked(ContestResult(...))` wired in the auth-eval tail; fires only when a `[contest-CUDA]` score returns.
- **Hook 6 (probe-disambiguator):** `None` — single mechanism per variant; the per-substrate symposium IS the empirical disambiguation (Catalog #325).

## Canonical equation (Catalog #344) — FORMALIZATION_PENDING

The PACT-NeRV family shares the canonical score-improvement-prediction equation
`predicted_ΔS = α·B(θ)/N + β·d_seg(θ) + γ·√(10·d_pose(θ))` where the
distinguishing primitive modulates `d_seg`/`d_pose`/`B(θ)` per variant.
# FORMALIZATION_PENDING: canonical equation `pact_nerv_family_score_domain_lagrangian_v1`
queued for sequential registration in `.omx/state/canonical_equations_registry.jsonl`
after the sister registry-owning subagent (master-gradient + canonical-equations
sisters were mid-write at this landing per the prompt's sister-safety note) lands;
the equation is predicted-only (no empirical anchor exists yet — paid dispatch is
gated per Catalog #325), and racing the registry write would risk a fcntl-locked
JSONL collision per Catalog #131/#344.

## Lane registry (Catalog #90 / #233)

All 12 lanes (`lane_pact_nerv_<variant>_l0_scaffold_20260520`) had their gates
unset at L0 SCAFFOLD. This wave marks `impl_complete` (the `_full_main` body now
exists, routing through the canonical base) + `memory_entry` (THIS memo). No
level change beyond L1 (L2 requires a real `[contest-CUDA]` anchor which is
paid-dispatch gated per Catalog #325). The lanes stay `research_only=true` /
`dispatch_enabled: false` — Catalog #240-compliant.

## Discipline honored

- Catalog #157/#174 canonical serializer with POST-EDIT `--expected-content-sha256`
  on every commit (3 batches: `6651189bd` / `142940946` / `9077d19bb`).
- Catalog #119 Co-Authored-By trailer auto-appended on all 3 commits.
- Catalog #206 checkpoint discipline (8 checkpoints across the wave).
- Catalog #314/#340 sister-safety — owned ONLY `*pact_nerv*` files; the dirty
  non-pact_nerv state files (master_gradient + canonical_equations + automation
  sister territory) were NEVER staged.
- Catalog #1 / "MPS auth eval is NOISE" — full path CUDA-required.
- Catalog #114 — real contest video; synthetic FORBIDDEN outside `--smoke`.
- Catalog #6 — eval_roundtrip patched before scorer construction.
- "EMA — NON-NEGOTIABLE" — EMA shadow is the inference checkpoint.
- Catalog #325 — recipes stay `dispatch_enabled: false` (paid path gated).
- Review gate — all 24 `.py` files marked reviewed; no `REVIEW_GATE_OVERRIDE=1`.

[verified-against: experiments/train_substrate_pact_nerv_vq.py::_full_main canonical PyTorch pattern]
[verified-against: src/tac/substrates/_shared/pact_nerv_full_main.py shared helper]
[verified-against: 12 variant package test suites — 171 tests pass on CPU]
[verified-against: end-to-end loss-callback wiring with real scorers — 6 trickiest variants finite + backward_ok]
