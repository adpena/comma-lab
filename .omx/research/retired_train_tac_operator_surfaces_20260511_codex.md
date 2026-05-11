# Retired train-tac operator surfaces (2026-05-11)

## Scope

This cleanup removes runnable references to retired training and Modal launch
entrypoints from current operator-facing surfaces:

- `README.md`
- `src/tac/profiles.py`
- `docs/research_roadmap.md`
- `data/artifacts/job_queue.json`

The canonical paths are now:

- `experiments/pipeline.py` for profile-driven local/proxy compression and eval
- `tac.deploy.deploy_config` for provider-neutral training command flags
- `python -m tac.deploy.build_bundle` for provider-neutral cloud bundles
- lane-specific actuators such as `experiments/modal_t1_balle_endtoend.py` for
  claimed score-lowering dispatches

## Why

The retired queue and docs mixed historical MPS/proxy experiments with commands
that no longer map to the current parser surface. Leaving those as runnable
examples creates an apples-to-oranges risk: a future sweep can spend wall-clock
or GPU time on a stale training path and then compare its outputs against the
current score-lowering substrate.

## Preserved signal

The old queue rows were retained as `retired_historical_queue_entry` records
with the tag, priority, dependency, platform, and design intent still visible.
Their status is now `retired_historical`, not `running` or `queued`. They must
be ported into a canonical profile or deploy-config variant before any new run.

This is infrastructure hardening only. It is not a score claim and does not
change the active T1 Modal dispatch.
