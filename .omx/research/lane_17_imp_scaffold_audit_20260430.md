---
name: Lane 17 (IMP) — Scaffold Audit (Phase A of Level 1 → Level 3 push)
description: 2026-04-30 audit of every Lane 17 IMP artifact in the repo before the Level-3 push. Inventory + maturity per gate + concrete gap list.
type: research
authoritative_for: lane_17_imp
---

## Inventory of Lane 17 / Lane J-IMP artifacts (as of 2026-04-30)

### Module — `src/tac/iterative_magnitude_pruning.py` (610 LOC)
Mature. Public API:
- `IMPState` dataclass with `cycle_count`, `sparsity_target`, `sparsity_increment`, `mask`, `early_epoch_weights` + `to_dict` / `from_dict`.
- `iter_prunable_parameters(model)` — Conv2d / ConvTranspose2d weights only. Excludes BN, biases, embeddings (Frankle 2020 §4.1 convention).
- `snapshot_state_dict(model)` — CPU clone for early-epoch rewinding.
- `prune_lowest_magnitude(model, sparsity_increment=0.20, current_mask=None)` — global magnitude prune, monotone (already-pruned positions stay False).
- `apply_mask_to_model(model, mask)` — in-place zero of pruned positions.
- `rewind_weights_to_early_epoch(model, snap, mask)` — Frankle 1912.05671 stabilization: survivors copied from snap, pruned positions stay zero.
- `compute_actual_sparsity(model, mask=None)` — fraction of zeroed (or False-masked) prunable weights.
- `sparse_csr_export(weights, mask)` / `sparse_csr_decode(blob)` — uint16 idx + FP4 nibble encoding. Header layout documented in module docstring. numel ≤ 65535 cap (Lane G v3 largest conv has ~16K weights → fine).
- `fp4_pack_values` / `fp4_unpack_values` — nibble pack/unpack with low-nibble first.

Sparse-CSR breakeven analysis embedded in docstring: dense-FP4 = 88K × 4 / 8 = 44KB; sparse-CSR = nnz × 2.5B; beats dense once sparsity > 80%; at 89% sparsity ~9.4K nnz → ~23.5KB → saves ~21KB ≈ -0.014 score points at the 25× rate multiplier.

### Single-cycle script — `experiments/train_imp_cycle.py` (401 LOC)
Mature, EMA-wired (Council D 2026-04-29). Per-cycle invocation pattern:
- `--cycle N` (0 = first; uses anchor `.bin`; 1+ takes prior cycle's `.pt`/mask/snapshot).
- `--target-sparsity 0.20` per cycle, `--final-sparsity-target 0.89`.
- `--epochs 200`, `--lr 1e-4`, `--batch-size 4`, `--device cuda`.
- `--ema-decay 0.997` REQUIRED per CLAUDE.md "EMA — NON-NEGOTIABLE".
- `--smoke` skips fine-tune for unit tests / CI.
- `--no-auth-eval-on-best` is the explicit opt-out the preflight `check_training_scripts_have_auth_eval` looks for; per-cycle auth eval is deferred to Stage 4 of the bash dispatcher (one auth eval on the final 89%-sparse renderer).
- Saves: `renderer.pt`, `mask.pt`, `early_epoch_snapshot.pt`, `imp_state.pt`, `stats.json`.
- EMA snapshot+restore correctly applied AFTER fine-tune AND mask is re-applied AFTER `ema.apply()` to preserve sparsity (the EMA shadow may have non-zero values at pruned positions due to initial-state averaging).

### In-process orchestrator — `experiments/imp_cycle_runner.py` (319 LOC)
Mature, deterministic, no silent defaults. `run_imp_cycles(model, num_cycles, sparsity_increment, train_step_fn, ...)` returns `ImpRunResult` with per-cycle `CycleResult` + `monotone_sparsity` invariant. CLI is intentionally a `NotImplementedError` SCAFFOLD — Check 81 STRICT compliance (no silent default model loader). The shell dispatcher / external dispatcher must call `run_imp_cycles` directly with a wired model + train_step_fn.

### Bash dispatcher — `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` (325 LOC)
Production-grade. Stages:
- Stage 0: NVDEC probe (`scripts/probe_nvdec.sh --ensure-dali`).
- Stage 0b: canonical git sync (`fetch origin main && reset --hard origin/main`).
- Pre-flight: argparse dead-flag scan against `train_imp_cycle.py` (CLAUDE.md non-negotiable — NEVER invent CLI flags).
- Stage 1: 10-cycle IMP loop. Cycle 0 reads `experiments/results/lane_g_v3_landed/iter_0/renderer.bin`; cycles 1-9 read previous cycle's `renderer.pt` / `mask.pt` / `early_epoch_snapshot.pt`.
- Stage 2: re-export final FP32 weights to ASYM/FP4 `renderer.bin` (uses `tac.renderer_export.export_asymmetric_checkpoint_fp4`).
- Stage 3: build contest archive (IMP renderer + Lane G v3 masks + Lane G v3 poses).
- Stage 3b: archive size assertion (Lane B-class disaster guard).
- Stage 4: contest_auth_eval [contest-CUDA].
- Heartbeat loop, provenance JSON, predicted_band metadata, `[contest-CUDA]` tag at completion.

### Tests
- `src/tac/tests/test_iterative_magnitude_pruning.py` (10 tests):
  - `test_prune_lowest_magnitude_removes_target_fraction` — exact pruned count ±1 (kthvalue tie tolerance).
  - `test_prune_preserves_high_magnitude_weights` — top-50 magnitudes survive.
  - `test_rewind_only_affects_unpruned_weights` — pruned stay 0; survivors == snapshot.
  - `test_compute_actual_sparsity_matches_pruning_target` — sparsity tracker honest at 10/30/50/80%.
  - `test_sparse_csr_export_byte_size` — at 95% sparsity, sparse-CSR < dense FP4.
  - `test_sparse_csr_roundtrip_preserves_values` — mask exact; values within FP4 step error.
  - `test_imp_state_serialization_roundtrip` — torch.save/load lossless.
  - `test_multi_cycle_sparsity_grows_monotonically` — pruned stay pruned + 3-cycle 48.8% match.
  - `test_fp4_pack_unpack_roundtrip` — nibble pack exact for n=0..33.
- `src/tac/tests/test_imp_cycle_runner.py` (7 tests):
  - `test_sparsity_monotone_across_cycles_synthetic` — 4-cycle 59.04% match.
  - `test_weight_count_kept_decreases_each_cycle_synthetic` — non-increasing.
  - `test_pruning_mask_deterministic_synthetic` — same seed → same mask.
  - `test_no_rewind_path_runs_synthetic` — `rewind_after_prune=False` valid path.
  - `test_no_silent_defaults_synthetic` — `model=None` / `num_cycles=0` / bad sparsity / `train_step_fn=None` all raise.
  - `test_cli_raises_not_implemented_synthetic` — Check 81 STRICT compliance.
  - `test_write_manifest_produces_valid_json_synthetic` — manifest has all expected fields.

Total: 17 tests across both suites.

### Profile — `IMP_CYCLE_DILATED_H64` in `src/tac/profiles.py:2860`
Inherits PROVEN_BASELINE; epochs=200, lr=1e-4, batch=4, loss_mode=standard, seed=89.

### Lane registry — `lane_17_imp_10cycle` (`.omx/state/lane_registry.json:246`)
- `impl_complete: true` (evidence: `src/tac/imp_cycle.py + experiments/train_imp_cycle.py`)
- `real_archive_empirical: false` ← gap
- `contest_cuda: false` ← gap
- `strict_preflight: false` ← gap
- `three_clean_review: false` ← gap
- `memory_entry: false` ← gap
- `deploy_runbook: false` ← gap (script exists but not registry-attested)

## Per-gate maturity vs the user's Level-3 standard

| Gate | Status | Evidence | Gap |
|------|--------|----------|-----|
| Implementation completion | DONE | module + script + orchestrator + bash dispatcher + EMA + Frankle-2019 rewinding + sparse-CSR codec + FP4 pack/unpack | Inflate-side sparse-CSR magic-byte handler MISSING (current dispatcher re-exports to FP4A which loses the sparsity benefit at archive time). Revert-on-regression kill-criterion not wired. |
| Real-archive empirical | NOT DONE | — | Need a CPU-only smoke that runs IMP for 1-2 cycles on the actual Lane G v3 anchor renderer.bin and measures the sparse-CSR byte savings vs the FP4 baseline. |
| Contest-CUDA | NOT DONE | dispatcher Stage 4 wired but never run | EXPENSIVE — see pre-dispatch memo. |
| STRICT preflight | NOT DONE | Check 88 (EMA wire-in) covers `train_imp_cycle.py`; Check 22 (training scripts have auth eval) is opted-out via `--no-auth-eval-on-best` flag | Need a NEW check that audits `train_imp_cycle.py` specifically: (a) EMA decay 0.997 default, (b) the deferred auth eval IS actually invoked at Stage 4, (c) revert-on-regression invocation. |
| 3-clean-pass adversarial review | NOT DONE | — | Need 3 rounds @ 0 bugs in `.omx/research/council_lane_17_imp_round{1,2,3}_*.md`. |
| Memory entry | PARTIAL | `project_phases_2_3_4_design...` describes the design only; no landed memory | Need `project_lane_17_imp_landed_20260430.md`. |
| Deploy runbook | DONE-IN-SCRIPT | bash dispatcher has heartbeat + provenance + harvest path is implicit (Stage 4 writes `auth_eval.log`) | Should be cross-referenced in runbook memory + lane registry `deploy_runbook` gate. |

## Concrete gap list (in execution order)

1. **Council design review** (Phase B) — write `.omx/research/council_lane_17_imp_design_20260430.md` with rotating perspectives (Frankle, Hotz, Quantizr, Selfcomp, Shannon).
2. **Revert-on-regression kill criterion** — wire into bash dispatcher: if cycle N's per-cycle CUDA smoke score regresses >10% from cycle N-1 baseline, REVERT to cycle N-1 mask + STOP. (Council to decide whether per-cycle smoke is required vs cycle-final-only — adds ~$0.30/cycle × 10 = $3 vs single $0.50 final.)
3. **Inflate-side sparse-CSR handler** — magic byte `IMPS` (b"IMPS") in `inflate_renderer.py:_load_renderer`. Routes to a packing/unpacking path that uses `tac.iterative_magnitude_pruning.sparse_csr_decode`.
4. **STRICT preflight Check 91** — `check_imp_cycles_use_ema_and_auth_eval` — flag any IMP cycle script lacking EMA snapshot OR per-cycle / final CUDA auth eval. STRICT @ 0 violations.
5. **Real-archive empirical (Phase F)** — short CPU smoke in tests dir that loads Lane G v3 anchor, runs 2 cycles synthetic-data fine-tune, measures sparse-CSR vs FP4A byte savings, tagged `[empirical:reports/lane_17_imp_real_archive_smoke.json]`.
6. **3-clean-pass adversarial review** — Round 1, 2, 3 with rotating perspectives.
7. **Pre-dispatch memo** — `project_lane_17_imp_pre_dispatch_20260430.md` with cost breakdown ($25 full vs $12.50 5-cycle quick variant) + predicted score band + kill criteria + alternative.
8. **STOP and ask user** — DO NOT dispatch full 10-cycle without explicit user "approved" memo response.
9. **(Post-approval) Pattern A nohup detach + heartbeat** for the GPU work.

## Key non-negotiables already satisfied

- EMA decay 0.997 with snapshot+restore-after-mask-reapply — `experiments/train_imp_cycle.py:394-396`.
- eval_roundtrip — N/A in current per-cycle stub (synthetic loss); the deploy script's Stage 4 auth eval inherits eval_roundtrip from `contest_auth_eval.py` → `inflate.sh` → `evaluate.py`.
- No silent defaults — orchestrator's CLI `main` raises `NotImplementedError` (Check 81 STRICT).
- No invented CLI flags — bash dispatcher does an argparse dead-flag scan before dispatching.
- NVDEC probe at Stage 0 — `scripts/probe_nvdec.sh --ensure-dali`.
- Provenance JSON + heartbeat — both written.
- Predicted band metadata `[0.85, 1.00]` written in provenance.
- `[contest-CUDA]` tag in dispatcher's "DONE" message.

## Cross-refs

- `feedback_production_hardened_standard_definition_20260430.md` (the Level 3 standard)
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 17"
- `feedback_check_64_smoke_proofs_resolved_AND_subagent_serializer_landed_20260429.md` (commit serializer)
- CLAUDE.md "EMA — NON-NEGOTIABLE", "Auth eval EVERYWHERE", "NEVER invent CLI flags", "Recursive adversarial review protocol"
