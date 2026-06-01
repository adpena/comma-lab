# HPRC P18/P19 Prefix Projection Materializer Fix

Timestamp: 2026-06-01T11:23:43Z
Author: Codex
Status: LANDED_AND_QUEUE_SMOKED

## Trigger

A queue-owned corrected HPRC smoke was launched after wiring native P18/P19
artifacts into both training and rate-collapse:

`/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_native_rate_p18p19_corrected_smoke_20260601T112100Z`

The queue executed:

1. `build_hprc_native_rate_residual_protection_surface`
2. `run_local_training` with `--training-backend mlx`
3. `transcode_hprc_rate_collapse`
4. `write_hprc_campaign_followup_report`

The first attempt found a real materializer bug:

```text
transcode_hprc_compact_receiver_rate_collapse failed:
P19 PoseNet-null artifact n_pairs does not match HPRC packet
```

## Bug Class

The train-time native surface builder correctly supports using full-video
P18/P19 scorer artifacts as prefix projections for smaller 32/128-pair smoke
campaigns. The rate-collapse materializer did not: it required exact pair-count
equality and rejected valid full-video artifacts when applied to smaller HPRC
packets.

That made queue-owned smoke campaigns fail even though full600 would be valid.
Worse, it encouraged either duplicated ad hoc prefix artifacts or skipping the
rate-collapse scorer surface entirely.

## Fix

`tools/transcode_hprc_compact_receiver_rate_collapse.py` now:

- accepts P18/P19 artifacts whose declared pair count is larger than the HPRC
  packet pair count;
- filters selected pairs/rows to the packet prefix;
- records explicit provenance blockers:
  - `p19_full_video_artifact_prefix_projected_to_hprc_packet`
  - `p18_full_video_artifact_prefix_projected_to_hprc_packet`
- still fails closed when P18/P19 artifacts are shorter than the HPRC packet;
- still rejects out-of-range selected pairs for same-size artifacts.

## Verification

Targeted tests:

```text
uv run ruff check tools/transcode_hprc_compact_receiver_rate_collapse.py src/tac/substrates/hprc/tests/test_rate_collapse.py
PYTHONPATH=. .venv/bin/pytest src/tac/substrates/hprc/tests/test_rate_collapse.py -q
```

Result:

```text
12 passed
```

Queue retry after rewind:

```text
success_count=2
failure_count=0
```

Smoke report:

- report:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_native_rate_p18p19_corrected_smoke_20260601T112100Z/hprc_rate_collapse/hprc_rate_collapse_report.json`
- `residual_importance_enabled=true`
- `residual_importance_selection_domain=eligible_low`
- eligible residual-cell fraction: `0.12369791666666667`
- source binding status: `video_pair_count_compatible_proxy_priors`
- best archive bytes: `84775`
- result is not promotable and not exact-ready.

The smoke root is 6.1 MiB on the SSD tier and contains no retained inflated raw
video dump.

## Next Action

This closes the queue/materializer blocker. The next meaningful HPRC campaign
is full600, because prefix smoke artifacts are intentionally advisory. Full600
should run with:

- MLX training backend;
- native P18/P19 residual protection;
- rate-collapse P18/P19 reuse;
- full local replay only after the rate gate;
- exact CPU/CUDA only if local replay wins.

The smoke selected the lossless section transcode variant over lossy residual
collapse at 32 pairs, so the next score-moving work is not another small smoke:
it is native rate-aware training plus multi-resolution pose/boundary token
allocation before full600 replay.
