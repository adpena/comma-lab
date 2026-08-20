# 08 — Tests to add

## 1. Parse-back selection test

File: `src/tac/tests/test_long_training_archive_selection.py`

```python
def test_archive_selection_prefers_parseback_replay_over_live_proxy(tmp_path):
    # fake adapter:
    # live local proxy = 1, live parseback proxy = 10
    # ema local proxy = 2, ema parseback proxy = 2.5
    # current behavior selects live; patched behavior selects ema.
```

## 2. Parse-back required fail-closed test

```python
def test_archive_selection_replay_required_fails_without_adapter_hook(tmp_path):
    config.archive_selection_replay_required = True
    adapter has no archive_replay_components
    assert selected_archive is None
    assert manifest has blocker "archive_selection_replay_required_but_unavailable"
```

## 3. HiNeRV hard-birth scoped actuator

File: `src/tac/substrates/hi_nerv/tests/test_target_region_birth.py`

Assertions:

- worst target region has before ratio 0.0.
- birth step improves frontier margin or hard ratio.
- only allowed param patterns update.
- no sidecar bytes.

## 4. Seg/Pose trust-region

File: `src/tac/substrates/_shared/mlx_score_aware/tests/test_seg_pose_trust_region.py`

Assertions:

- reject positive joint score update.
- accept negative joint score update.
- frame0 pose-only update has bounded SegNet leak.

## 5. SNeRV full TUB source-forward closure

File: `src/tac/substrates/snerv_inverse_steg_carrier/tests/test_official_tub_full_source_forward.py`

Assertions:

- no `unmapped_temporal_encoder`.
- no `unmapped_output2_decoder`.
- official Torch full TUB output equals portable NumPy/MLX.
- manifest sets full parity true only after tensor equality.
