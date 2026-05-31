# Codex Findings: MLX SSD + Posterior Read-Surface Partner Review

Date: 2026-05-31T01:20:00Z
Agent: Codex
Scope:
- `src/tac/canonical_posterior_read_validator/`
- `src/tac/cathedral_consumers/phantom_score_canonical_posterior_lookup_consumer/`
- `src/tac/canonical_equations/equation.py`
- `src/tac/canonical_equations/modal_dispatch_runtime_tree_hash_parity.py`
- `src/tac/optimization/mamba2_predictor.py`
- `src/tac/substrates/time_traveler_l5_z7_mamba2/`
- `src/tac/substrates/z8_hierarchical_predictive_coding/mamba2_adapter.py`
- `experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py`

## Verdict

Proceed after repair. The partner work had real integration value but was not
landable as found: lint failed, the trainable Z7 MLX module was untracked while
a tracked experiment imported it, and the Z7 smoke manifest still claimed the
full MLX path was blocked pending the migration that now exists.

## Repairs Landed

1. Canonical posterior read-surface protection is executable:
   - Added/validated `canonical_posterior_read_validator`.
   - Added/validated the cathedral consumer that surfaces phantom-score
     posterior verdicts without mutating candidate score.
   - Preserved false authority: predicted-only provenance, no score claim, no
     promotable state.

2. Canonical equations anchor surfaces are stricter:
   - Added optional `EmpiricalAnchor.empirical_verification_status` with the
     four-value Catalog #363 taxonomy.
   - Preserved legacy byte identity by omitting the field when unset.
   - Validated the Modal runtime-tree hash parity builder against the existing
     16-test parity suite.

3. Z7/Z8 Mamba-2 SSD is a real opt-in path:
   - `Mamba2Predictor` now exposes `backend="ssd_reference"`.
   - Z7 config threads `ssd_nheads` / `ssd_headdim`.
   - Z8 adapter exposes `use_canonical_ssd=True`.
   - Tests prove the canonical helper is actually invoked and gradients flow.

4. Z7 MLX-local smoke now exercises the trainable module:
   - Tracked `Z7Mamba2MLXModule`.
   - Updated the operator smoke to instantiate the module, emit parameter
     count, emit latent shape, and remove the stale blocked-migration message.
   - Added regression tests so the smoke cannot silently fall back to the old
     native-only renderer.

## Non-Authority Boundary

All MLX outputs remain `[macOS-MLX research-signal]`. They may accelerate local
candidate generation and training, but they do not claim score, promote, rank,
kill, or dispatch exact eval without PyTorch bridge export plus paired contest
CPU/CUDA replay.

## Validation

- `ruff check` on all touched Python files: pass.
- Posterior/cathedral/canonical-equation tests: 75 passed.
- Z7/Z8 SSD rewire tests: 30 passed.
- Z7 MLX module smoke tests: 2 passed.
- Combined focused suite: 107 passed.
- Modal runtime-tree hash parity suite: 16 passed.
- Live MLX smoke: wrote
  `.omx/research/codex_z7_mamba2_mlx_module_smoke_20260531T011800Z/smoke_manifest.json`
  with module-backed `(2, 3, 384, 512)` frame tensors, `(2, 24)` latents,
  `renderer_num_parameters=751614`, and fail-closed authority fields.

## Remaining Risk

The SSD reference backend is structurally wired and gradient-tested, but no
score movement is claimed. Next work should run short MLX training probes and
posterior ingestion, then bridge any winner into byte-closed PyTorch/runtime
custody before exact-axis spend.
