# Findings

## 2026-04-10 promoted floor

- Track B promoted floor is **1.33**.
- Variant: `dilated_h64`.
- Platform: `modal_a10g`.
- PoseNet `0.00218374`, SegNet `0.00609921`, rate `0.02301653`.
- Canonical score/report mirrors are generated from `.omx/state/promoted_result.json`.

## 2026-04-11 [discovery] Training pipeline view/reshape crash on MPS

fit_lazy() crashes at backward pass with RuntimeError: view size not compatible. AllNorm monkey-patch in _patch_scorers_for_training converts .view() to .reshape() but the error persists. Root cause: non-contiguous tensors from channels_last or permute operations hitting upstream scorer .view() calls. Need to make tensors .contiguous() before scorer forward pass.

## 2026-04-11 [discovery] Root cause: channels_last on scorers breaks MPS backward pass

The P2 optimization (channels_last on scorers) was incompatible with PyTorch MPS backend. AllNorm batch_norm backward uses .view() internally which fails with channels_last stride layout. Fix: keep channels_last on the model (postfilter) for speed but leave scorers in standard layout. Training now runs successfully on MPS.

## 2026-04-11 [discovery] Full-stack review: 133 test failures from orphaned imports, 2 broken deploys

13 test files reference deleted experiments/*.py via spec_from_file_location. 2 Modal deploy scripts point to deleted trainers. Training pipeline MPS crash fixed (channels_last on scorers). Training smoke test now running successfully on MPS (5+ min, still going).

## 2026-04-11 [discovery] Training performance regression: 6+ min per epoch on smoke profile

smoke profile uses subsample=4 (75 pairs/epoch). Old scripts used subsample=8 (9 pairs). Full 874x1164 scorer forward+backward on MPS is ~5s/pair. 75*5s = 375s = 6 min/epoch. Need subsample=8 or higher for smoke. Not a code bug — configuration issue. Training runs but too slow for interactive testing.

## 2026-04-11 [discovery] Modal deploy needs debugging — image build likely failing

Two Modal apps launched (mask_renderer_smoke + smoke). Both completed quickly with no results in tac-renderer-results volume. Likely failure: image build cant find tac source (add_local_dir path resolution from deploy script location). Need to test image build locally or check modal app logs. Precomputed data IS on the volume.

## 2026-04-11 [discovery] Modal deploy iterative debug: 4 fixes to get training launching

Fix 1: remove condition kwarg (Modal 1.4.1 incompatible). Fix 2: add_local_dir must be last. Fix 3: guard REPO_ROOT for container path depth. Fix 4: clone upstream scorer repo for PoseNet/SegNet modules. Training reached scorer loading on A10G — next attempt should start actual training.

## 2026-04-11 [discovery] MLX Phase 1 crash: gradient clipping passes optimizer instead of grads

train_renderer_mlx.py gradient clipping code calls .square() on AdamW optimizer object instead of gradient tensors. Fix: check the grad_clip implementation in the MLX training loop.

## 2026-04-11 [discovery] Modal A10G training still failing: results dirs created but empty

Modal v4 (with upstream clone) created mask_v3 and mask_v4 dirs in results volume but they are empty. Training likely crashed during scorer loading or first forward pass. Need to check Modal logs for exact error.
