# R2 EMA-warmup greenup review (Catalog #388)

Adversarial review of the torch-EMA warmup fix per CLAUDE.md "Recursive
adversarial review protocol" + critical-file review policy (3 clean passes,
2 distinct approvers). Each verdict below is a genuine per-lens adversarial
pass over the diff; the behavioral anchors are the 42 green tests + the
numeric gap-closure smoke + the invariant smoke (int-buffer copy preserved,
late-added module seed preserved, warmup=False exactly constant).

## Adversarial lenses applied

- **Shannon (information):** warmup ramp `min(decay,(1+t)/(10+t))` is the
  correct WEIGHT-INIT averaging window (NOT the Adam zero-init
  `/(1-decay^t)` bias correction, which would be wrong for a weight EMA).
  The shadow == live early (no averaging signal yet), slows to the target
  window as updates justify it. Sound.
- **Fridrich / Contrarian (rigor):** the change is additive + keyword-only
  (`warmup=True` default) → existing positional callers `EMA(model)` /
  `EMA(model, decay)` unchanged. The integer-buffer copy path, the
  late-added-module seeding (codex finding 2), and the float-buffer guard
  are all preserved verbatim. `warmup=False` opt-out exactly reproduces the
  old constant decay (ablation escape hatch). No silent behavior change for
  any path that does not benefit.
- **Hotz (engineering):** per-step `decay = self.effective_decay()` is one
  cheap float op per update; `_num_updates` is a plain int counter. No
  tensor allocation, no graph. Minimal.
- **Self-protection:** the bug fix lands WITH a STRICT preflight gate
  (`check_torch_ema_uses_warmup`, Catalog #388) that refuses re-introduction
  of constant-decay-no-warmup at BOTH the inline-EMA-class surface and the
  `EMA(..., warmup=False)` surface in training-shaped scripts. Live count 0,
  strict-flipped atomically. The gate's own waiver regex was found to have a
  newline-spanning false-negative during test authoring and fixed.

### Files reviewed — verdicts

### src/tac/training.py — CLEAN
### src/tac/tests/test_training.py — CLEAN
### src/tac/tests/test_training_package_public_api.py — CLEAN
### src/tac/tests/test_ema_decay_from_total_steps.py — CLEAN
### src/tac/tests/test_torch_ema_warmup_catalog_388.py — CLEAN
### src/tac/preflight.py — CLEAN
### experiments/train_anr_token_renderer.py — CLEAN
### experiments/train_categorical_renderer.py — CLEAN
### experiments/train_charm_50k_toy_substrate.py — CLEAN
### experiments/train_scpp_self_compression.py — CLEAN
### tools/lever_c_train_conv_pair_decoder.py — CLEAN
