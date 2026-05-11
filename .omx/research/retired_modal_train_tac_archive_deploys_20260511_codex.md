# Retired Modal train_tac archive deploy scripts (2026-05-11)

## Scope

This cleanup replaced four legacy executable Modal launchers with fail-closed
compatibility stubs in
`src/tac/deploy/modal/archive/`:

- `modal_h96_v2_deploy.py`
- `modal_nuclear_deploy.py`
- `modal_dilated_h96_dual_sal_deploy.py`
- `modal_dilated_kl_hardframe_deploy.py`

## Why

The previous implementations referenced the retired `experiments/train_tac.py`
entry point and passed stale flags such as `--archive`, `--gt-video`, `--saliency`,
`--models-dir`, `--upstream-dir`, and `--resume-from`. Those flags are not part
of the current canonical renderer path and would be invalid for the active
score-lowering training entry points.

Keeping those implementations executable under `src/tac/deploy/` created a concrete
provider-drift hazard: an operator or agent could launch a GPU job that fails
before producing score evidence or, worse, trains a non-comparable path.

## Preserved signal and fail-closed behavior

Useful historical patterns from those scripts remain part of the design prior:

- Bake immutable scorer/upstream assets into the provider image when doing so
  shortens wall clock without changing contest custody.
- Keep archives, checkpoints, and optional priors on explicit provider volumes.
- Print cost estimates before dispatch, but never treat estimates as evidence.
- Resume only from a path accepted by the actual training CLI.
- Avoid synthetic or uniform saliency fallbacks for promoted score work unless
  the run is explicitly marked proxy-only.

The files remain present only as tiny fail-closed stubs. The current canonical
replacement is the provider-neutral bundle path
`python -m tac.deploy.build_bundle`, the shared training flag source
`tac.deploy.deploy_config`, and the Modal T1 actuator
`experiments/modal_t1_balle_endtoend.py`.

## Evidence

- `python -m tac.deploy.build_bundle --output /tmp/tac_bundle_codex_check --no-saliency`
  builds a bundle with `train_renderer_fridrich.py`, `src/tac`, `archive.zip`,
  and upstream scorer/video assets.
- Generated `train.sh` now sets `TAC_UPSTREAM_DIR`, `TAC_MODELS_DIR`, and
  `TAC_RESULTS_DIR` and uses `deploy_config.build_flags(..., resume_from=...)`.
- Focused provider tests and all-lanes preflight passed before the preceding
  canonical bundle commit.

## Score-lowering consequence

This is infrastructure cleanup, not a score claim. It reduces dispatch waste
and preserves apples-to-apples score lowering by preventing obsolete Modal
launchers from bypassing the canonical claim/custody/config path.
