# CLASS-SHIFT-FULL-MAIN-CLUSTER landed 2026-05-27

Lane: `lane_class_shift_full_main_cluster_20260527`
Subagent: `class_shift_full_main_cluster_20260527T153322Z`
Cost: $0 (CPU-local validation only; NO paid GPU)

## Mandate

Top standing priority continuation: implement real score-aware `_full_main`
training loops for the remaining **non-PACT-NeRV** `experiments/train_substrate_*.py`
trainers whose `_full_main` still `raise NotImplementedError`. PACT-NeRV's 18
variants were landed by the prior cluster (PACT-NERV-FULL-MAIN-CLUSTER, today);
this is the class-shift / paradigm-shift remainder. Implement-the-code,
gate-the-paid-trigger per Catalog #325 (every recipe stays `dispatch_enabled:
false` + `research_only: true`; full path is CUDA-required via `device_or_die`).

## Grep enumeration (NotImplementedError in `_full_main`, excluding `*pact_nerv*`)

24 trainers matched `raise NotImplementedError`; AST-scoped to the `_full_main`
function body yielded **22 with NIE directly in `_full_main`** (2 false matches:
`time_traveler_l5_autonomy` had a clean `_full_main`; `z7_mamba2_v2` has no
`_full_main` def). Of the 22, classification:

| substrate | torch arch | archive grammar (pack_archive) | verdict |
|---|---|---|---|
| boost_nerv | yes | yes | **IMPLEMENTED** |
| coin_plus_plus | yes | yes | **IMPLEMENTED** |
| nirvana | yes | yes | **IMPLEMENTED** |
| z5_predictive_coding_world_model | yes | yes (multi-component) | **IMPLEMENTED** |
| atw_codec_v1 | yes | yes (multi-component) | **IMPLEMENTED** |
| c1_world_model_foveation | yes | yes (all-kwargs) | BLOCKER (loss API divergence) |
| tishby_ib_pure | yes | yes (sections) | BLOCKER (IB-term loss, not RGB renderer) |
| pr101_lc_v2_clone_enhanced_curriculum | yes | yes | BLOCKER (curriculum/Muon/QAT; no `_smoke_main`; far beyond shared helper) |
| pr101_with_dp1_prior_regularizer | yes (DrivingPriorRenderer) | yes (DP1) | BLOCKER (Comma2k19 codebook prior; bespoke) |
| e_nerv | yes (`tac.e_nerv_as_renderer`) | **NO pack_archive** | BLOCKER (no archive grammar) |
| ego_nerv | yes (`tac.ego_nerv_as_renderer`) | **NO pack_archive** | BLOCKER (no archive grammar) |
| nervdc | yes (`tac.nervdc_as_renderer`) | **NO pack_archive** | BLOCKER (no archive grammar) |
| atw_v2_1 | no package | n/a | BLOCKER (no substrate package) |
| nirvana_cascading_nerv_mlx | no package | n/a | BLOCKER (no substrate package) |
| time_traveler_l5_tt5l_v2 | no package | n/a | BLOCKER (no substrate package) |
| atw_v2_cooperative_receiver_v2 | **MLX-only** (`mlx_renderer.py`) | yes | BLOCKER (MLX-first; no torch arch; shared helper is PyTorch-only) |
| coin_pp_implicit_neural_representation_mlx | **MLX-only** | yes | BLOCKER (MLX-first) |
| dreamer_v3_rssm | **MLX-only** (`module.py`, mlx) | yes | BLOCKER (MLX-first) |
| faiss_ivf_pq_residual | **MLX-only** | yes | BLOCKER (MLX-first) |
| mdl_ibps_j_discrete_categorical_mine_hybrid | **MLX-only** | yes | BLOCKER (MLX-first) |
| z8_hierarchical_predictive_coding_mlx | **MLX-only** (mlx) | yes | BLOCKER (MLX-first) |

## IMPLEMENTED (5 substrates; 0 NotImplementedError verified per trainer)

All 5 route the substrate-AGNOSTIC training loop through the canonical
`tac.substrates._shared.pact_nerv_full_main.run_pact_nerv_score_aware_training`
helper (the same helper the implemented `ds_nerv` + 18 PACT-NeRV variants use).
Each substrate's UNIQUE distinguishing feature stays in its architecture +
archive + score-aware loss; the `_full_main` is a thin adapter closure +
per-substrate archive-build block. NotImplementedError extinguished; PAID
DISPATCH stays gated by `dispatch_enabled: false` + `research_only: true` on the
recipe per Catalog #325 (code complete, trigger gated).

1. **boost_nerv** — iterative residual-refinement chain (`num_boosting_rounds`);
   simple NeRV-family `model(idx) -> (rgb_0, rgb_1)`; `pack_archive(decoder_sd,
   latents, meta, num_boosting_rounds=...)`. Commit `77f102d3c`.
2. **coin_plus_plus** — per-pair modulated coordinate-MLP / INR
   (`modulation_dim`); latent key `modulations`; `pack_archive(base_sd,
   modulations, meta, modulation_dim=...)`; rate proxy includes `modulations`
   via `extra_param_names`. Commit `77f102d3c`.
3. **nirvana** — patch-wise NeRV decode + stitch (`patch_grid_h/w`,
   `patch_embed_dim`); `pack_archive(decoder_sd, latents, meta, patch_grid_h,
   patch_grid_w, patch_embed_dim)`. Commit `77f102d3c`.
4. **z5_predictive_coding_world_model** — Rao-Ballard hierarchical predictive
   coding (encoder + autoregressive ego-motion predictor + per-pair residual
   codes; Atick-Redlich cooperative-receiver). `reconstruct_pair(idx) ->
   (rgb_0, rgb_1, z)` adapter threads `m.residuals[idx]` into the loss;
   multi-component archive (encoder/decoder/predictor + latent_init + residuals
   + ego_motion) partitioned by reloading the EMA shadow into a fresh model.
   CUDA-required via `_resolve_full_device` (rejects MPS per Catalog #1;
   `--full-cpu` opt-in per Catalog #197). Z4 canary dependency preserved in the
   recipe gate. Commit `d3383b4a5`.
5. **atw_codec_v1** — Atick-Tishby-Wyner cooperative-receiver IB encoder +
   Wyner-Ziv side-information residual head; three-knob Lagrangian
   (kappa_ib/lambda_wz/lambda_pixel). `model(idx, frames_for_encoder=None,
   compute_wz_residual=True) -> (rgb_0, rgb_1, mu, logvar, z_residual,
   z_predicted)` adapter threads `z_residual`/`z_predicted` into the loss;
   multi-component archive (encoder/decoder/wz_side_info_head + latent_residual
   + scorer_class_prior_table) partitioned by EMA-reload. Commit `30ea2077d`.

### Validation per substrate (CPU-local, $0)

- Trainer imports cleanly; `_full_main` source contains 0 `raise
  NotImplementedError` AND `run_pact_nerv_score_aware_training`.
- `_full_main(args)` with `--device cpu` (no `--full-cpu`) raises `SystemExit`
  via `device_or_die` / `_resolve_full_device` — CUDA-required gate intact
  (Catalog #1 + #325).
- Architecture forward / reconstruct_pair / archive-partition shapes verified
  on tiny CPU configs.
- ruff clean (import-organization auto-fixed).
- Per-substrate dedicated tests updated from `test_full_main_raises_not_implemented`
  → `test_full_main_implemented_and_cuda_gated` (canonical pact_nerv_ia3 pattern).

### Test pass count

- boost_nerv: 11/11
- coin_plus_plus: 11/11
- nirvana: 11/11
- z5: 47/47 (substrate + parse + new trainer full_main test)
- atw_codec_v1: 50/50 (scaffold test file)
- **Total: 130/130 pass.**

### Catalog #325 / #240 gating verification

- Catalog #240 (recipe-vs-trainer-state consistency): **0 violations**.
- Catalog #325 (per-substrate symposium before dispatch): 4 pre-existing
  violations (`grayscale_lut_lut_bits_5`, `rudin_floor_interpretable_ml`,
  `z6_v2_candidate_1/4c` — all `dispatch_enabled: true`, NONE mine). My 5
  substrates do NOT appear in the #325 list — their recipes stay
  `dispatch_enabled: false`. **Gating intact.**

## HONEST BLOCKERS (17 substrates NOT implemented — precise missing piece)

Per the prompt's "honest partial > fake-complete" + CLAUDE.md "Forbidden
representation-without-archive-grammar (research-substrate trap)" + "Substrate
scaffolds MUST be COMPLETE or RESEARCH-ONLY":

### Loss / API divergence from the canonical renderer contract (4)

- **c1_world_model_foveation** — `render_pair(pair_idx)` returns `(rgb_0,
  rgb_1)` but the loss requires `residual` + `foveation_map` which render_pair
  does NOT return; the loss has NO F3-cache kwargs (`gt_pose_batch`/`gt_seg_batch`/
  `gt_seg_already_probs`), so the shared helper's `compute_loss` cache-passing
  is incompatible. `pack_archive()` is an all-kwargs world-model API. Adapting
  requires modifying the c1 loss (out of trainer scope; risks the substrate's
  own tests) OR a non-canonical loop. **Missing piece: F3-cache-compatible
  loss signature + render_pair returning residual/foveation_map.**
- **tishby_ib_pure** — loss `__call__(reconstruction_term, kl_term, rate_term)`
  operates on Information-Bottleneck terms, NOT pixel-RGB; `pack_archive(sections)`
  is a generic sections API. Genuinely not a frame-renderer; the shared
  renderer helper does not apply. **Missing piece: IB-native training harness
  (the shared helper assumes an RGB-pair renderer).**
- **pr101_lc_v2_clone_enhanced_curriculum** — curriculum-stage system
  (`CURRICULUM_STAGES`, `Muon` optimizer, QAT, ternary stage budgets,
  Comma2k19 stage-0 bootstrap); NO `_smoke_main`; forward returns
  `torch.stack([f0,f1],dim=1)` already `*255`. Scope is an entire curriculum
  trainer, far beyond a shared-helper wire-in. **Missing piece: dedicated
  enhanced-curriculum training loop (Muon + QAT + multi-stage schedule).**
- **pr101_with_dp1_prior_regularizer** — `DrivingPriorRenderer` + Comma2k19
  dashcam codebook prior (`distill_codebook`, `Comma2k19FrameIterator`,
  `compose_with`); `pack_archive(codebook, renderer_state_dict,
  per_pair_residual, meta)`. Bespoke pretrained-prior pipeline; depends on
  Comma2k19 dataset access per Catalog #213. **Missing piece: DP1-prior
  composition training loop + Comma2k19 codebook distillation wiring.**

### Missing archive grammar (research-substrate trap if forced) (3)

- **e_nerv** (`tac.e_nerv_as_renderer`), **ego_nerv** (`tac.ego_nerv_as_renderer`),
  **nervdc** (`tac.nervdc_as_renderer`) — renderer + encoder + latent-table +
  score-aware-loss modules exist, but **NO `pack_archive` / `parse_archive` /
  inflate** anywhere. Per HNeRV parity L2 + the 8th forbidden pattern, a
  substrate without an archive builder + inflate runtime is research-only by
  construction; a full training loop that cannot export scored bytes would be
  the exact research-substrate trap CLAUDE.md forbids. **Missing piece:
  archive grammar (`pack_archive` + `parse_archive` + contest inflate.py).**

### No substrate package (3)

- **atw_v2_1**, **nirvana_cascading_nerv_mlx**, **time_traveler_l5_tt5l_v2** —
  no `src/tac/substrates/<id>/` package (and no `tac.<id>_as_renderer` module).
  atw_v2_1 imports only `tac.optimization.faiss_ivf_pq_atw_channel`;
  nirvana_cascading (128 lines) + tt5l_v2 (726 lines) reference no instantiable
  architecture module. **Missing piece: the substrate package
  (architecture + archive + inflate + score_aware_loss).**

### MLX-first (no torch architecture; PyTorch-only shared helper) (6)

- **atw_v2_cooperative_receiver_v2**, **coin_pp_implicit_neural_representation_mlx**,
  **dreamer_v3_rssm**, **faiss_ivf_pq_residual**,
  **mdl_ibps_j_discrete_categorical_mine_hybrid**,
  **z8_hierarchical_predictive_coding_mlx** — these have `mlx_renderer.py` /
  `numpy_reference.py` / mlx `module.py` but NO torch `architecture.py`. The
  canonical `run_pact_nerv_score_aware_training` is PyTorch-only (`torch.optim.AdamW`
  + `tac.training.EMA` + torch autograd + torch differentiable scorers). A full
  MLX score-aware training loop would require an entirely new MLX-native
  training harness (MLX optimizer + MLX EMA + MLX↔torch scorer bridge for the
  contest SegNet/PoseNet) which does not exist. Implementing it is a large new
  infrastructure build, not a `_full_main` wire-in. **Missing piece: a canonical
  MLX score-aware training harness (`run_mlx_score_aware_training`) — the 8th
  MLX-first standing directive's inflate-side is numpy-portable, but the
  TRAINING-side MLX harness is not yet built.** These are the highest-EV
  follow-up: per the 8th directive (MLX-first training on M5 Max), building the
  canonical MLX training harness would unlock all 6 at once.

## 6-hook wire-in declaration (Catalog #125) — applies to the 5 implemented

- hook #1 sensitivity-map: N/A — score-aware substrates; rate-distortion
  captures sensitivity per the SubstrateContract `hook_not_applicable_rationale`.
- hook #2 Pareto constraint: ACTIVE via `rate_distortion_v1` (each
  SubstrateContract declares `hook_pareto_constraint`).
- hook #3 bit-allocator: N/A — fp16/brotli on weight blobs; no per-tensor allocator.
- hook #4 cathedral autopilot dispatch: GATED — recipes stay `dispatch_enabled:
  false` per Catalog #325; not dispatch-eligible until per-substrate symposium.
- hook #5 continual-learning posterior: ACTIVE — each `_full_main` calls
  `posterior_update_locked` after a real `[contest-CUDA]` auth-eval (fires only
  on a real CUDA dispatch, which is gated).
- hook #6 probe-disambiguator: N/A — single mechanism per substrate.

## Discipline honored

- Catalog #229 premise verification: read CLAUDE.md + canonical `ds_nerv` +
  `pact_nerv_ia3` template + shared `pact_nerv_full_main.py` + every target
  substrate's architecture/loss/archive API BEFORE editing.
- Catalog #117/#157/#174/#235 canonical serializer + POST-EDIT
  `--expected-content-sha256` for every commit (3 commits: `77f102d3c`,
  `d3383b4a5`, `30ea2077d`).
- Catalog #340 sister-checkpoint guard: the only overlap was my OWN in-flight
  checkpoint id; resolved via the documented paired-env self-coordination
  bypass with explicit rationale.
- Catalog #206 checkpoint discipline (4 checkpoints under my subagent id).
- Catalog #1/#114 no-MPS-fallback / real-video-not-synthetic: all full paths
  CUDA-required + decode real `upstream/videos/0.mkv` via the canonical
  `decode_pairs_for_training`.
- Catalog #290 canonical-vs-unique: each `_full_main` ADOPTS the canonical
  shared loop (serves: identical RGB-renderer score-aware training contract)
  and FORKS the per-substrate archive-build block (the distinguishing feature
  lives there) — the canonical bolt-on-vs-substrate-engineering split.
- SISTER-SAFETY: touched ONLY non-PACT-NeRV `experiments/train_substrate_*.py`
  + their substrate test files + this memo. Did NOT touch master-gradient /
  5D-canvas / paradox-sweep territory or any `*pact_nerv*` file.

## Operator-routable next steps

1. **Highest EV**: build the canonical MLX score-aware training harness
   (`run_mlx_score_aware_training` sibling of `run_pact_nerv_score_aware_training`)
   per the 8th MLX-first standing directive — unlocks all 6 MLX-first
   substrates at once (atw_v2_cr_v2, coin_pp_mlx, dreamer_v3_rssm,
   faiss_ivf_pq, mdl_ibps_j, z8_mlx).
2. Add `pack_archive`/`parse_archive`/inflate to `tac.{e,ego}_nerv_as_renderer`
   + `tac.nervdc_as_renderer` → then their `_full_main` becomes a clean
   shared-helper wire-in (same pattern as the 5 landed here).
3. c1_world_model_foveation: F3-cache-compatible loss signature + render_pair
   returning residual/foveation_map (council-grade per the substrate's design).
4. pr101_lc_v2_clone_enhanced_curriculum + pr101_with_dp1: bespoke
   curriculum/DP1 training loops (out of canonical-shared-helper scope).
