# 02 — PR95 parse-back selection: minimum patch

## Finding

PR95 selected candidates after building an archive, parsing it back, and scoring the parsed decoder. The local PR95 forensic memo states: every 25 epochs it built a fresh archive from EMA decoder+latents, parsed it back, ran the parsed archive forward, and scored using `100*seg + sqrt(10*pose) + 25*rate`.

At the reviewed ref, `run_long_training` can export live and EMA archives, but `_export_live_ema_archive_selection()` still selects with:

```text
health sort key
+ local proxy score
+ archive bytes * 25/N
```

The selection manifest says `authority = local_training_proxy_false_authority`. That is honest, but it is not PR95-source-faithful.

## Minimum code change

Do **not** run full official evaluate every epoch. Do **not** slow the MLX loop.

Add one optional hook used only at archive selection:

```python
archive_replay_components(
    archive_path: Path,
    batch: Any,
    *,
    candidate_kind: str,
) -> Mapping[str, float] | None
```

Contract:

- Must decode the emitted archive or packet parse-back path, not live model tensors.
- Must return score-unit keys: `seg`, `pose`, optional `recon_aux`, optional scorer health.
- Long-run configs can require it with `archive_selection_replay_required=True`.
- If hook returns components, selection uses parse-back proxy:
  `replay_seg + replay_pose + archive_bytes * 25/N`.
- If hook absent and required, manifest records blocker and no selected archive.

## Patch targets

- `src/tac/training/long_training_canonical.py`
  - `LongTrainingConfig`: add `archive_selection_replay_required: bool = False`.
  - `SubstrateLongTrainingAdapter` docs: optional hook.
  - `_export_live_ema_archive_selection`: call replay hook after archive export.
  - selection rows: add `parseback_score_components`, `parseback_proxy_score`, `selection_authority`.
- `src/tac/substrates/_shared/mlx_score_aware/adapter.py`
  - Add method routing to `bundle.archive_replay_components_fn` if provided.
- HiNeRV/SNeRV runners:
  - Set `archive_selection_replay_required=True` only after adapter hook exists.

## Failing test

`tests/test_parseback_selection.py` creates a fake adapter:

- live local proxy: 1.0, parseback proxy: 10.0
- EMA local proxy: 2.0, parseback proxy: 2.5

Expected selected candidate after patch: `ema`.

Current behavior selects `live`.

## Passing target

```bash
pytest -q src/tac/tests/test_long_training_archive_selection.py::test_archive_selection_prefers_parseback_replay_over_live_proxy
```

## Why this preserves MLX velocity

The hook runs only at export/selection, not inside `OOMSafeStepRunner.run_step()`. It can use a small fixed replay batch for live-vs-EMA selection, then separately a full-video replay gate before exact eval.
