# BI1 Receipt — TR1 Birth Seed/Amplify Path

## Answer First

BI1 built the missing TR1-native seed/amplify BIRTH mechanism behind default-OFF runtime flags.  It does not run scorers, does not launch, does not claim a d_seg or score improvement, and does not move the contest pointer.

Status: IMPLEMENTED + SCORER-FREE SMOKE PASSED + A/B READY, with one execution caveat: the sandbox cannot import `mlx.nn` because no Metal device is available, so the MLX live-token/checkpoint-byte smoke is not measured here.  The OFF proof below is structural plus focused unit coverage, not an empirical checkpoint-byte comparison.

## Recall Evidence Used

- p4x measured the birth/existence debt and explicitly left the TR1 seed/amplify half unbuilt.  The retired island-protection family measured an ancestor/simulated-erasure effect, so BI1 reuses only the geometry (`eased_island_masks` / persistence weighting), not the retired flags or numbers.
- GC12 framed the wall branch as birth-completion ladder work; LP2 provided an external `birth_completion` producer only, not a trainer seed/amplify path.
- R1C showed the parent birth-completion key had already fired at that parent.  BI1 therefore builds mechanism readiness only and makes no plateau, hinge, or quality claim.

## Diff Summary

- `experiments/train_tr1_partition_renderer_mlx.py`
  - Added args-only flags: `--tr1-birth-seed-weight`, `--tr1-birth-seed-classes`, `--tr1-birth-seed-dilate-px`, `--tr1-birth-amplify-weight`, `--tr1-birth-amplify-persist`.
  - Added scorer-free GT-support bank builder using `lstars`, eased Lane/Movable island supports, token-lattice max pooling, and persistence weighting.
  - Added ON-only token seeding plus `tr1_birth_amplify_term` anchor in `batch_loss`.
  - Added fail-closed guards for unknown classes, negative weights, amplify-without-seed, all-zero ON targets, `--token-cell-mask`, and `--token-rowband-spec`.
- `src/tac/witness_dsl/bi1_birth_seed_levers_20260805.py`
  - Registered `lever_tr1_birth_seed_amplify()` under the TR1 trainer with provenance-bearing constants and `score_claim=False`.
- `src/tac/tests/test_ddm_tb1_tr1_renderer.py` and `src/tac/witness_dsl/tests/test_bi1_birth_seed_levers.py`
  - Added parser/default-off, pure birth-bank smoke, geometry-refusal, and DSL registry tests.

## Byte-Identity Proof

MEASURED/static-unit:
- `TR1Config` has no `birth_seed` or `birth_amplify` fields.
- Parser defaults are OFF: seed weight `0.0`, amplify weight `0.0`, classes `lane`.
- OFF helper call returns `{"active": false, "reason": "weight<=0"}` before MLX import and does not attach `_tr1_birth_seed`.
- OFF path emits no BI1 telemetry row and does not enter the loss addend (`birth_seed_amplify_weight = 0.0`).

DERIVED:
- Because checkpoint metadata is `asdict(cfg)` plus telemetry tail, and BI1 OFF does not mutate `cfg`, model params, EMA, optimizer state, or telemetry tail, checkpoint bytes should be unchanged for a runnable MLX checkpoint writer.

NOT MEASURED HERE:
- Actual checkpoint-byte equality, because `mlx.nn` import fails in this sandbox with `No Metal device available`.

## Smoke And Checks

- Focused scorer-free tests: `18 passed in 1.77s`.  The MLX package still emits a no-Metal atexit warning; no BI1 scorer or training run was executed.
- Pure BI1 bank smoke output: target shape `[2, 24, 32, 2]`, mask shape `[2, 24, 32, 1]`, Lane seeded cells `16`, nonzero target coefficients `32`, `score_claim=false`.
- DSL/p4x registry tests passed: BI1 factory emits only real TR1 flags and is filed under `experiments/train_tr1_partition_renderer_mlx.py`.
- Ruff: BI1-introduced C420 slice clean; new BI1 DSL module and test file clean.  Whole-file ruff on the historical TR1 trainer/test still reports pre-existing style findings outside BI1 scope.
- dn1 declared/read self-check: controls PASS; denominator `184/184` parsed files, `55` constructor-field declarations, `483` argparse declarations, `none_path_rows=1`.  The one row is the known positive-control `margin_targets`; no BI1-owned declared-never-read row was found in the dn1 scope.
- Review tracker: two `mark-file --status reviewed` passes recorded for all four touched Python files; status reports 100% reviewed for each.

## A/B Readiness

Ready command shape, not launched:

```bash
.venv/bin/python experiments/train_tr1_partition_renderer_mlx.py \
  --variant plain \
  --out-dir <run_dir> \
  --tr1-birth-seed-weight 0.35 \
  --tr1-birth-amplify-weight 0.05 \
  --tr1-birth-seed-classes lane
```

Boundary: this only proves mechanism availability.  A future A/B must use a scorer slot, matched seed/schedule/base, and owned byte-closed evaluation before any d_seg or S claim.

## NEXT_IF_RESUMED

```json
{
  "status": "A_B_READY_NO_SCORE_CLAIM",
  "next_actions": [
    "Run the same focused tests in an MLX environment with Metal access to close the live-token and checkpoint-byte empirical proof.",
    "If a scorer slot is assigned, claim a lane before launching a matched OFF-vs-ON BI1 A/B.",
    "Keep BI1 separated from token_cell_mask and token_rowband_spec until an explicit target projection is built."
  ],
  "not_done": [
    "No scorer forward",
    "No hinge A/B",
    "No exact eval",
    "No frontier movement"
  ],
  "score_claim": false
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
