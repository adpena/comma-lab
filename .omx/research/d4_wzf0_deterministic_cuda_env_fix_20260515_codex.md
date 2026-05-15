# D4 WZF0 Deterministic CUDA Env Fix - 2026-05-15

score_claim: `false`
promotion_eligible: `false`
dispatch_attempted: `false`

## Trigger

The harvested D4 50-epoch, 200-pair Modal T4 smoke
`fc-01KRN4AAM138RHZT6WXKXK464B` completed as a training artifact, but
`trainer.log` emitted PyTorch deterministic warnings for cuBLAS-backed CUDA
ops:

- `torch.bmm` inside `motion_model.py`
- `affine_grid_generator`
- CUDA backward paths for bicubic upsample and grid sampler

This did not invalidate the smoke result because no score claim was made, but
it is a repeatability weakness before any full D4 first-anchor dispatch.

## Fix

`scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh` now exports the CUDA
runtime determinism environment before sourcing the bootstrap helper or invoking
the trainer:

```bash
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
```

The remote provenance JSON records all three values so future harvest reviews
can verify the contract from artifacts.

## Guard

`src/tac/tests/test_d4_f1_f4_smoke_recipe.py` now asserts:

- D4 exports `CUBLAS_WORKSPACE_CONFIG`;
- D4 exports the DALI and CUDA allocator guards;
- the exports appear before the bootstrap helper and trainer invocation;
- provenance includes `cublas_workspace_config`.

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -p no:cacheprovider -q src/tac/tests/test_d4_f1_f4_smoke_recipe.py
.venv/bin/ruff check src/tac/tests/test_d4_f1_f4_smoke_recipe.py
bash -n scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh
git diff --check
```

Results:

- `5 passed`
- ruff passed
- shell syntax passed
- diff whitespace clean

## Next D4 Gate

Do not spend on a full D4 auth-eval run from this patch alone. The last smoke
emitted a `657455` byte WZF01 archive and selected epoch `0` as best validation,
so the next D4 action should first reduce proxy archive bytes or improve the
validation Lagrangian enough to justify a full 600-pair score-anchor dispatch.
