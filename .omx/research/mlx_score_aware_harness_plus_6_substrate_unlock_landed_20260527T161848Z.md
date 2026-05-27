# MLX score-aware harness + 6 MLX-first substrate unlock — LANDED 2026-05-27

Lane: `lane_mlx_score_aware_harness_plus_6_substrate_unlock_20260527`
Evidence grade: `[macOS-MLX research-signal]` (non-promotable; $0 M5 Max MLX-local)
Commits: `9635ca39a` (harness + 23 tests) + `6ebf95408` (dreamer + z8 `_full_main`)

## What landed

The canonical **MLX-first score-aware training harness** —
`src/tac/substrates/_shared/mlx_score_aware_full_main.py` — the MLX sister of
the PyTorch-only `pact_nerv_full_main.py`. It is the substrate-AGNOSTIC
`_full_main` body that the MLX-first class-shift substrates route through,
extinguishing their `_full_main` `NotImplementedError` for substrates whose
distinguishing primitive is a trainable MLX renderer.

### Harness contract

- `RendererBundle` — the substrate's UNIQUE axis: its MLX renderer (an
  `mlx.nn.Module` with `__call__(idx) -> (B,2,3,H,W)` in `[0,255]`, OR
  `reconstruct_pair(idx) -> (rgb_0, rgb_1)` NCHW in `[0,1]`), real-video
  target buffers, optional `extra_loss_terms` callback, and `distillation_weight`.
- `score_aware_loss(bundle, idx)` — the gradient-reachable MLX score-aware
  Lagrangian: reconstruction MSE (NHWC `[0,1]`) + optional **Hinton-distilled
  KL T=2.0 scorer surrogate** (canonical math from
  `tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss`; student =
  deterministic projection on the DECODED frame, teacher = stop-gradient
  projection on the TARGET frame — gradient flows KL → decoded → renderer
  params per CLAUDE.md "eval_roundtrip" + Catalog #164 sister discipline) +
  optional substrate-specific extra terms.
- `MlxScoreAwareAdapter` — generic Style-B adapter satisfying the canonical
  `tac.training.long_training_canonical.SubstrateLongTrainingAdapter` Protocol
  (combined `train_step` via `mlx.nn.value_and_grad` + AdamW). Generalizes the
  proven `Z6LongTrainingAdapter` reference. `export_state_dict` writes a
  numpy-portable MLX-native `.npz` by default (no PyTorch dep); the substrate's
  MLX→PyTorch bridge is opt-in via `export_state_dict_fn`.
- `run_mlx_score_aware_full_main(...)` — decodes real contest video
  (`decode_mlx_targets`, Catalog #114), wraps the bundle, builds the canonical
  `LongTrainingConfig`, and routes through `run_long_training` (canonical EMA
  shadow / OOM-safe step / early-stop / telemetry / Provenance / posterior
  anchor). Each substrate `_full_main` is now ~30 LOC of config + one call.
- `require_mlx_for_harness()` — **fail-closed** on a non-MLX host (NO silent
  CPU/CUDA fallback per Catalog #1 + #317 + #325). The harness is MLX-local $0
  by construction; there is no paid-dispatch leak.
- `assert_numpy_portable_inflate(inflate_py_path)` — ast-based verifier that a
  substrate's `inflate.py` imports neither `mlx` nor `torch` (the 8th
  directive's numpy/PIL-portable inflate half + HNeRV parity L4).

Tests: `src/tac/substrates/_shared/tests/test_mlx_score_aware_full_main.py`
— **23/23 pass** (numpy-portable contract runs everywhere; MLX-bound tests
skip cleanly off Apple Silicon). Includes a real end-to-end MLX training run
through `run_long_training`.

Non-promotable by construction per CLAUDE.md "MLX portable-local-substrate
authority" + Catalog #127/#192/#317/#341: every artifact `score_claim=False`,
`promotion_eligible=False`, `ready_for_exact_eval_dispatch=False`,
`evidence_grade=[macOS-MLX research-signal]`. The canonical L2 harness
auto-stamps these on the `TrainingArtifact`.

## Substrates: 2 landed REAL `_full_main` / 4 honest blockers

The prompt named 6 MLX-first targets. After verifying each renderer's
trainability via `mlx.nn.value_and_grad`, the honest outcome is:

### LANDED real `_full_main` (0 NotImplementedError; verified e2e real-video)

1. **`dreamer_v3_rssm`** (`experiments/train_substrate_dreamer_v3_rssm.py`).
   Renderer `DreamerV3RSSMSubstrateMLX` (`module.py`) is a trainable
   `mlx.nn.Module` (`__call__(idx) -> (B,2,3,384,512)`; categorical posterior +
   Gumbel-Softmax STE + HNeRV decoder). `_full_main` routes through the
   harness; e2e run confirmed (3 epochs, 4 pairs, promotable=False). `--smoke`
   preserved. Added `--video-path` / `--full-lr` / `--distillation-weight`.
   `## Canonical-vs-unique decision per layer` section added (Catalog #290).

2. **`z8_hierarchical_predictive_coding`**
   (`experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py`).
   Renderer `Z8HierarchicalPredictiveCoderMLX` (`mlx_renderer.py`) is a
   trainable `mlx.nn.Module` (multi-level RSSM + per-level Gumbel-Softmax +
   Mallat wavelet proxy + DreamerV3 deterministic state + HNeRV decoder).
   `_full_main` routes through the harness; e2e run confirmed. `--smoke`
   preserved. Added `--output-dir` / `--video-path` / `--full-lr` /
   `--distillation-weight`. Canonical-vs-unique section added.

### HONEST BLOCKERS (renderer is NOT a trainable nn.Module; per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + the no-fake-it discipline)

3. **`atw_v2_cooperative_receiver_v2`** — BLOCKER: renderer
   `ATWv2CooperativeReceiverV2MLX` (`mlx_renderer.py`) is an L0 SCAFFOLD that is
   **NOT an `mlx.nn.Module`** (`isinstance(m, nn.Module)` is `False`; no
   `.parameters()`), and its `reconstruct_pair(pair_indices, pose_delta)`
   requires a separate `pose_delta` argument the harness does not supply.
   **Missing piece**: wrap the cond-embed head + decoder + per-pair latent
   residual as a single trainable `mlx.nn.Module` exposing `__call__(idx)` (the
   `pose_delta` must be carried internally per-pair, not passed at call time).
   This is substrate-engineering (nn.Module wrap), NOT a `_full_main` wiring
   task; the renderer's own docstring marks it "L0 SCAFFOLD ~600-800 LOC".
   Reactivation criterion: land the nn.Module wrapper, then the existing
   `_full_main` NotImplementedError can be replaced with the harness call.

4. **`mdl_ibps_j_discrete_categorical_mine_hybrid`** — BLOCKER: renderer
   `MDLIBPSJRendererMLX` (`mlx_renderer.py`) holds **fixed** weights
   (`self.weights = weights.astype(mx.float32)` — loaded-weight inference, not
   trainable params; not an `mlx.nn.Module`). The substrate's existing
   `long_training_adapter.MdlIbpsJLongTrainingAdapter.__init__` already
   `raise NotImplementedError` per Catalog #240 documenting the exact L1
   follow-up: "wrap the MDLIBPSJRendererMLX primitives as a trainable
   `mlx.nn.Module` with `self.weights = ...`". **Missing piece**: the
   trainable-nn.Module wrap (FiLM proj + CoordMLP + MINE critic as nn.Module
   submodules with learnable params). Same substrate-engineering blocker as
   atw_v2; harness wiring is a drop-in once the wrap lands.

5. **`faiss_ivf_pq_residual`** — BLOCKER: `mlx_renderer.py` contains only
   numpy/MLX **PQ-decode primitives** (`mlx_pq_codebook_gather`,
   `mlx_pq_reconstruct_tile_vectors`, `mlx_tiles_to_frame_nhwc`), NOT a trainable
   renderer — it is a product-quantization codebook quantizer, not a learnable
   network. Its own `_full_main` raises NotImplementedError pending the Phase 2
   Catalog #325 symposium. **Missing piece**: this substrate's "training" is
   codebook *fitting* (k-means / OPQ rotation), structurally different from
   gradient-descent renderer training; it does not fit the gradient-reachable
   harness contract. Reactivation: a codebook-fitting trainer (not the
   gradient-descent harness) OR a learnable-residual-decoder nn.Module if the
   substrate adds one.

6. **`coin_pp_implicit_neural_representation`** — BLOCKER: `mlx_renderer.py`
   contains only config + param-count + archive-bytes-estimate helpers — there
   is **no MLX SIREN/COIN++ renderer forward implemented** (no `reconstruct`,
   no `__call__`, no trainable modulation network). Its own `_full_main` raises
   NotImplementedError. **Missing piece**: the COIN++ base-MLP (SIREN) +
   per-instance modulation network as a trainable `mlx.nn.Module`. Once that
   forward lands, harness wiring is a drop-in.

## Numpy-portable inflate confirmation (a real finding to record honestly)

The 8th standing directive requires the INFLATE path to be numpy/PIL-portable
(no `mlx`/`torch` import). The harness ships `assert_numpy_portable_inflate`
(ast verifier) for this contract. **However**: the 4 substrates with an
existing `inflate.py` (dreamer / z8 / atw_v2 / mdl_ibps_j) currently import
`torch` in their `inflate.py` (the PyTorch-runtime decode pattern) — they are
**NOT numpy-portable today**. I therefore did NOT wire `inflate_py_path`
verification into the substrate `_full_main` calls (it would fail-closed and
block training that is otherwise legitimately MLX-local). The TRAINING half of
the 8th directive is bound by this harness; the numpy-portable INFLATE half is
a pre-existing gap for these substrates and is operator-routable as a separate
inflate-portability pass (those `inflate.py` files are out of the
inflate-portability-audit sister's declared scope — that sister owns the 5
class-shift PyTorch substrates: boost_nerv / coin_plus_plus / nirvana /
z5_predictive_coding_world_model / atw_codec_v1 + PACT-NeRV inflate, NOT these
6 MLX-first substrate dirs). The harness verifier is wired + tested so the
contract is enforceable the moment a substrate's `inflate.py` is converted to
numpy/PIL-only.

## Dispatch gating (Catalog #325)

No operator-authorize recipe exists for dreamer or z8, so there is no
paid-dispatch path to gate — the harness fails closed on a non-MLX host and the
`_full_main` runs MLX-local $0 only. The recipes, if/when created, must carry
`dispatch_enabled: false` + `research_only: true` per Catalog #325 until each
substrate clears its per-substrate symposium. Per-axis SegNet/PoseNet
decomposition + MLX→PyTorch export bridge + Catalog #319 deliverability_proof +
paired [contest-CUDA]+[contest-CPU] anchor remain DEFERRED to the PyTorch
sister L2 path (the harness's `score_aware_components` returns `None`, matching
the Z6 reference adapter's deferral).

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map = N/A (training harness; the canonical `run_long_training`
  posterior anchor is the sensitivity surface downstream).
- hook #2 Pareto constraint = N/A (per-axis decomposition DEFERRED to PyTorch sister L2).
- hook #3 bit-allocator = N/A (no per-tensor importance change at this layer).
- hook #4 cathedral autopilot dispatch = N/A (non-promotable MLX-local research signal;
  no archive-deployable artifact — promotion path is the PyTorch sister).
- hook #5 continual-learning posterior = **ACTIVE** (`run_long_training` emits the
  canonical posterior anchor via the canonical posterior_emission_helper on every run).
- hook #6 probe-disambiguator = N/A (single canonical training path; no 2+ defensible
  interpretations to arbitrate at the harness layer).

## Discipline honored

Catalog #229 PV (read PyTorch sister + canonical L2 harness + Z6 reference +
all 6 renderers before writing) / #117/#157/#174 canonical serializer with
POST-EDIT `--expected-content-sha256` / #206 checkpoints / #110/#113 APPEND-ONLY
(NEW harness + memo; no mutation of forensic artifacts) / #290 canonical-vs-unique
section in harness + both trainers / #314/#340 sister-safety (owned only the NEW
`_shared/mlx_score_aware_full_main.py` + tests + the 2 wired trainers + this memo +
lane row; did NOT touch the 5 class-shift PyTorch substrates or any
`*pact_nerv*` / master-gradient files) / #287 placeholder-rationale rejection /
Review gate satisfied (both trainers 2/2 clean passes).

$0 GPU + MLX-local M5 Max. No paid dispatch fired or prepared.
