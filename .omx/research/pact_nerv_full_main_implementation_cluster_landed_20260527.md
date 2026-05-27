# PACT-NeRV `_full_main` implementation cluster — LANDED 2026-05-27

**Lane:** `lane_pact_nerv_full_main_implementation_cluster_20260527`
**Subagent:** `pact_nerv_full_main_impl_1`
**Commits:** `259292757` (helper + ia3) + `b5e6331e4` (distilled_scorer + neural_codec_e2e) + `069360c7a` (mamba + selector_v2 + vq)
**Wall-clock:** ~1.5h. **GPU spend:** $0 (all CPU-local validation; full training paths are CUDA-gated + paid-GPU-gated).

## What this wave did

Per operator IMPLEMENTATION WAVE cluster 1 directive ("design and flesh out and
implement and write and integrate all" + "search for not implemented errors
too"): extinguished the L0 SCAFFOLD `raise NotImplementedError` in the
`_full_main` of the **top-6 highest-EV PACT-NeRV substrate trainers** by
implementing the real score-aware training path. The remaining 12 PACT-NeRV
variants are queued (per-variant table below) — this wave landed the shared
base + the 6 with the clearest distinguishing feature + cleanest archive
grammar, per the prompt's explicit "shared base + top-6, queue the long tail"
scope authorization.

**CRITICAL — code complete, PAID DISPATCH still gated.** Per the Catalog #325
resolution ("implement all without firing council-gated paid paths"): every
variant's `_full_main` is now a REAL training loop (no `pass`, no relocated
`NotImplementedError`), but every recipe stays `dispatch_enabled: false` +
`research_only: true`. The full path is CUDA-required (`device_or_die` rejects
MPS per Catalog #1) and refuses on CPU with a `SystemExit` — so no paid GPU
can fire until each substrate's per-substrate symposium clears it (Catalog
#325). This is the trigger that stays gated, not the code.

## The shared base (canonical helper)

`src/tac/substrates/_shared/pact_nerv_full_main.py` (~590 LOC) provides the
substrate-AGNOSTIC training scaffold:

- `run_pact_nerv_score_aware_training(...)` — the canonical train/val loop
  (AdamW + cosine-annealing, NaN watchdog per Council D, EMA shadow update
  after every `optimizer.step` + EMA-at-eval snapshot/restore per CLAUDE.md
  "EMA — NON-NEGOTIABLE", best-EMA-checkpoint selection by validation
  Lagrangian). Variant-specific forward + loss arrives via a `compute_loss`
  callback so each substrate's UNIQUE path stays in its own package.
- `decode_pairs_for_training(...)` — thin wrapper over canonical
  `trainer_skeleton.decode_real_pairs` (real contest video; Catalog #114).
- `write_contest_runtime(...)` — parameterized contest-compliant `inflate.sh`
  + `inflate.py` emission (Catalog #146; numpy/PIL-portable, no scorer
  imports, vendored package per Catalog #295).
- `build_archive_zip(...)` — deterministic ZipInfo + fixed-timestamp archive
  (Catalog #19).
- `closed_form_weight_byte_proxy(...)` — fp16 weight-byte proxy for the rate
  term.

10 dedicated tests in `src/tac/substrates/_shared/tests/test_pact_nerv_full_main.py`
validate the loop, EMA, NaN watchdog, runtime emission, and deterministic-zip
byte-stability on CPU (no GPU).

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| train/val loop + EMA + NaN watchdog | ADOPT_CANONICAL (shared helper) | substrate-agnostic; identical to implemented `ds_nerv` sister |
| `decode_real_pairs` / `device_or_die` / scorer load | ADOPT_CANONICAL | already shared across all substrates |
| architecture (`architecture.py`) | FORK (per-package) | the distinguishing primitive |
| archive grammar (`archive.py`) | FORK (per-package) | each variant's `pack_archive` signature differs |
| numpy/PIL inflate (`inflate.py`) | FORK (per-package) | already landed per HNeRV parity L4 |
| score-aware loss (`score_aware_loss.py`) | FORK (per-package) | variant-specific extra terms via `compute_loss` callback |

## MLX-first directive note (8th standing directive)

The 8th directive mandates MLX-first for NEW substrates. The PACT-NeRV family is
an EXISTING PyTorch architecture family (all 18 `architecture.py` are
`torch.nn.Module`; the canonical score-aware loss routes through PyTorch
SegNet/PoseNet via `load_differentiable_scorers` + Catalog #164
`score_pair_components_dispatch`). Migrating 18 architectures to MLX renderers
is an architecture migration, NOT a `_full_main` implementation; the OPTIMAL
ENGINEERING for these PyTorch substrates is the canonical PyTorch training loop
(the implemented sister `ds_nerv` uses it). The INFLATE path is already
numpy/PIL-portable (no MLX dep) per the 8th directive's portability requirement.

## Per-variant implementation table (6 landed)

| Variant | Distinguishing feature | Archive grammar (UNIQUE) | _full_main LOC | Tests | recipe dispatch_enabled |
|---|---|---|---|---|---|
| `pact_nerv_ia3` | IA3 γ-only ego-pose modulation (Liu 2022) | PIA3: latents + ego_poses + pose_dim | ~215 | 12 pass | `false` ✓ |
| `pact_nerv_distilled_scorer` | Hinton KL-T=2.0 scorer surrogate (1503.02531) | PDS: latents + surrogate meta | ~190 | 20 pass | `false` ✓ |
| `pact_nerv_neural_codec_e2e` | Ballé scale-hyperprior + rate_bits term (1802.01436) | PNNC: latents + hyperprior meta | ~210 | 14 pass | `false` ✓ |
| `pact_nerv_mamba` | Mamba-2 SSM backbone ssm.A_log state (2405.21060) | PNMB: latents + ssm_state | ~200 | 13 pass | `false` ✓ |
| `pact_nerv_selector_v2` | Arithmetic-coded FEC6 selector menu (Witten 1987) | PSV2: latents + selector_bytes + palette_size | ~210 | 14 pass | `false` ✓ |
| `pact_nerv_vq` | VQ-VAE codebook + per-pair indices + commitment (van den Oord 1711.00937) | PVQ: codebook + indices | ~225 | 12 pass | `false` ✓ |

Each `_full_main` is REAL (decode real video → patch yuv6 → load
differentiable scorers → variant score-aware Lagrangian via Catalog #164
dispatch → EMA → best-checkpoint → variant archive pack → contest-compliant
runtime → CUDA auth-eval → posterior update). No `pass` stubs, no relocated
NotImplementedError. CPU smoke unchanged; full path CUDA-gated (verified each
variant refuses `--device cpu` non-smoke via `device_or_die` SystemExit).

## 6-hook wire-in declaration (Catalog #125)

The 6 hooks were already declared in each variant's `SubstrateContract`
(Catalog #241 META layer) at L0 SCAFFOLD landing 2026-05-20. This wave does
not change the hook declarations; it makes hooks #4 (cathedral autopilot
dispatch — the training path now produces an archive the autopilot can rank)
and #5 (continual-learning posterior — `posterior_update_locked` wired in the
auth-eval tail) OPERATIONAL pending a real `[contest-CUDA]` anchor (which only
lands at paid dispatch, still gated):

- **Hook 1 (sensitivity-map):** `not_applicable_with_rationale` per each
  contract — no sensitivity signal until paid dispatch produces a real anchor.
- **Hook 2 (Pareto constraint):** `rate_distortion_v1` (the score-domain
  Lagrangian's rate + seg + sqrt(pose) terms).
- **Hook 3 (bit-allocator):** `not_applicable_with_rationale` — fp16 brotli on
  combined weight blob; per-tensor allocation is the L1+ research path.
- **Hook 4 (cathedral autopilot dispatch):** ACTIVE-pending-anchor — the
  `_full_main` now emits a byte-stable `archive.zip` the autopilot ranker can
  consume once a paid anchor exists.
- **Hook 5 (continual-learning posterior):** ACTIVE-pending-anchor —
  `posterior_update_locked(ContestResult(...))` wired in the auth-eval tail;
  fires only when a `[contest-CUDA]` score returns.
- **Hook 6 (probe-disambiguator):** `None` — single mechanism per variant;
  the FiLM-vs-{IA3,VQ,Mamba,...} disambiguation IS the per-substrate
  symposium's empirical purpose (Catalog #325).

## Lane registry (Catalog #90 / #233)

All 6 lanes were already L1 with `impl_complete=✓` at L0 SCAFFOLD landing. This
wave strengthens the `impl_complete` evidence (the `_full_main` body now
exists). The `memory_entry` gate is marked by THIS memo. No level change (they
remain L1; L2 requires a real `[contest-CUDA]` anchor which is paid-dispatch
gated per Catalog #325). The lanes stay `research_only=true` /
`dispatch_enabled: false` — Catalog #240-compliant (recipe research_only +
trainer implemented is a valid state; Catalog #240 only refuses
implicit-contest-CUDA recipes with NotImplementedError trainers).

## Canonical equation (Catalog #344) — FORMALIZATION_PENDING

The PACT-NeRV family shares a canonical score-improvement-prediction equation:
`predicted_ΔS = α·B(θ)/N + β·d_seg(θ) + γ·√(10·d_pose(θ))` where the
distinguishing primitive modulates `d_seg`/`d_pose`/`B(θ)` per variant. This is
NOT yet registered in `.omx/state/canonical_equations_registry.jsonl` because
the canonical equations registry is sister-owned (subagent `a75952e0fb1416383`
mid-write on `tac.canonical_equations` auto_recalibrate at the time of this
landing) — racing the registry write would risk a fcntl-locked-JSONL collision
per Catalog #131/#344. # FORMALIZATION_PENDING: canonical equation
`pact_nerv_family_score_domain_lagrangian_v1` queued for sequential registration
after the sister registry-owning subagent lands; the equation is predicted-only
(no empirical anchor exists yet — paid dispatch is gated per Catalog #325).

## Long-tail queue (12 remaining PACT-NeRV variants)

Per the prompt's explicit subset authorization, these 12 are NOT implemented
this wave. Each follows the identical canonical pattern (route `_full_main`
through `run_pact_nerv_score_aware_training` + variant-specific `pack_archive`).
A resumable successor subagent can drain them in priority order:

| Variant | Distinguishing feature | Archive extra | Est. effort |
|---|---|---|---|
| `pact_nerv_ia3_multi` | multi-layer IA3 γ stack | ego_poses + pose_dim (PIA3-like) | LOW (ia3 clone) |
| `pact_nerv_selector_v3` | selector menu v3 | selector_bytes + palette_size (PSV2-like) | LOW (selector_v2 clone) |
| `pact_nerv_selector_v4` | selector menu v4 | selector_bytes + palette_size | LOW (selector_v2 clone) |
| `pact_nerv_moe` | mixture-of-experts routing | expert-routing indices | MED (router state extract) |
| `pact_nerv_bayesian` | Bayesian posterior latents | posterior mean/var | MED (2-tensor latent pack) |
| `pact_nerv_multi_modal` | multi-modal conditioning | modality embeddings | MED |
| `pact_nerv_asymmetric_boundary` | asymmetric boundary codec | boundary map | MED |
| `pact_nerv_cross_codec_a` | cross-codec A | cross-codec side info | MED-HIGH (cross-archive) |
| `pact_nerv_cross_codec_b` | cross-codec B | cross-codec side info | MED-HIGH |
| `pact_nerv_neural_codec_e2e_cross` | neural codec E2E cross | hyperprior + cross side info | MED-HIGH |
| `pact_nerv_diffusion_distilled` | diffusion-distilled decoder | distill trajectory meta | HIGH (distill loss) |
| `pact_nerv_diffusion_trajectory` | diffusion trajectory | trajectory tensor | HIGH |

Resume via: `tools/subagent_checkpoint.py read --subagent-id pact_nerv_full_main_impl_1`.

## Discipline honored

- Catalog #157/#174 canonical serializer with POST-EDIT `--expected-content-sha256`
  on every commit (3 batches).
- Catalog #206 checkpoint discipline (9 checkpoints across the wave).
- Catalog #1 / "MPS auth eval is NOISE" — full path CUDA-required.
- Catalog #114 — real contest video; synthetic FORBIDDEN outside `--smoke`.
- Catalog #6 — eval_roundtrip patched before scorer construction.
- "EMA — NON-NEGOTIABLE" — EMA shadow is the inference checkpoint.
- Catalog #325 — recipes stay `dispatch_enabled: false` (paid path gated).
- Catalog #340 sister-checkpoint guard — self-collision false positives
  resolved via `--no-sister-checkpoint-check` (only this subagent's own
  checkpoints touched these files; disjoint from master_gradient +
  canonical_equations sister territory).
- Review gate — all `.py` files marked reviewed; no `REVIEW_GATE_OVERRIDE=1`.

[verified-against: experiments/train_substrate_ds_nerv.py::_full_main canonical PyTorch pattern]
[verified-against: src/tac/substrates/_shared/pact_nerv_full_main.py shared helper + 10 tests]
[verified-against: 6 variant package test suites — 85 tests total pass on CPU]
