# 04 — HiNeRV class birth: why min-ratio can stay zero while debt actuators move

## Finding

Current direct-live SegNet machinery has the right telemetry:

`score_weighted_unsolved = 100 * target_fraction * (1 - region_ratio)`.

It also computes target-region crossing loss:

`margin_c(x,y)=max_{j != c} z_j(x,y)-z_c(x,y)+1`.

But a compact renderer can reduce soft loss, global histogram loss, or target probability floors while still not creating a **hard argmax island** for the target class inside its own target region. That yields:

```text
target coverage can improve
candidate soft class mass can improve
target min-ratio remains 0.0
receiver SegNet argmax probe fails
```

This is not contradictory: a class can have probability mass below the top-1 boundary everywhere.

## Actual missing actuator

Use the existing telemetry, but bind it to a scoped HiNeRV parameter update:

```text
worst target class/region
-> direct-live SegNet VJP
-> update only head_rgb_1 + late feature_grid/proj + optional fine latent injector
-> preserve PoseNet via trust region
-> write ordinary archive-charged tensors only
```

## Patch target

`src/tac/substrates/hi_nerv/mlx_renderer.py`

Add:

```python
def fit_target_region_birth_from_segnet(
    self,
    *,
    scorer_teacher,
    target_rgb_1,
    target_argmax_1,
    pair_indices,
    max_steps: int = 64,
    learning_rate: float = 5e-4,
    target_min_region_ratio: float = 0.02,
    update_patterns: tuple[str, ...] = (
        "head_rgb_1",
        "feature_grids.*.grids",
        "feature_grids.*.proj",
        "fine_injector",
    ),
) -> dict[str, Any]:
    ...
```

Loss:

```text
L_birth = sum_c w_c [
  softplus(margin_c/tau)^2
  + lambda_p * relu(p_floor - p_c)^2
  + lambda_s * seed_margin^2
]
w_c = 100 * target_fraction_c * (1-region_ratio_c)
```

Stop-gradient metrics choose region; gradients flow through candidate RGB -> live SegNet logits -> renderer params.

## Why output-head/late-grid, not all params?

- `head_rgb_1` is the only output head seen directly by SegNet.
- Late grids/head have local spatial leverage.
- Early coarse/stem changes move many pairs and can break PoseNet geometry.
- The same method can emit gradient norms by group to prove actuation.

## Failing test

`tests/test_hinerv_target_region_birth.py`

- Build a tiny mock SegNet teacher with target class present but candidate class absent.
- Call `fit_target_region_birth_from_segnet(max_steps=4)`.
- Assert:
  - `before_region_ratio == 0.0`
  - `after_target_region_frontier_margin < before_target_region_frontier_margin`
  - updated parameter names are subset of allowed patterns
  - no sidecar bytes introduced
