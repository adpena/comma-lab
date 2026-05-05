---
name: train_renderer.py arch drift — checkpoints unusable downstream
description: train_renderer.py builds models with one arch but saves checkpoints whose state_dict shapes don't match the profile spec. Consumers (pipeline.py, auth_eval_renderer.py) crash on load.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Severity: CRITICAL.** A full DEN training run (1.2 hours, $0.30) produced a checkpoint that is UNLOADABLE by both `pipeline.py compress` and `experiments/auth_eval_renderer.py`. The arch the training script BUILDS does not match what the profile spec describes, AND the checkpoint doesn't carry arch metadata for consumers to use.

**Concrete failure 2026-04-26:** DEN profile sets `use_zoom_flow=True` (motion head outputs 4 channels: gate(1) + residual(3)). But the saved `renderer_den_best_fp32.pt` has `motion.head.bias` shape (2,) — i.e., trained with `use_zoom_flow=False`. The profile resolver in `train_renderer.py` either didn't pick up the flag or the model builder ignores it. Additionally, `__meta__` records seed + epoch + fp4_codebook but NOT arch fields, so even if consumers wanted to recover the right arch they couldn't.

Compounding: `auth_eval_renderer.py::_load_renderer_checkpoint` instantiates AsymmetricPairGenerator with HARDCODED defaults (`base_ch=36, mid_ch=60`), not even reading from CLI. DEN trained with `base_ch=28, mid_ch=40` → completely incompatible.

**Permanent fix needed (NOT done yet — flagged for next cycle):**
1. `train_renderer.py` MUST embed the full arch dict in checkpoint `__meta__` (every flag from the profile).
2. `auth_eval_renderer.py::_load_renderer_checkpoint` MUST read arch from `__meta__` if present, falling back to CLI / defaults.
3. `pipeline.py::step_export` MUST do the same.
4. Add a preflight rule: every renderer training script asserts the saved checkpoint round-trips through `_load_renderer_checkpoint(path)` before saving — fail loud at training time, not at deploy time.

**Until fixed:**
- DON'T run pipeline.py compress on a checkpoint produced by train_renderer.py.
- Use the SHIRAZ flow (train_distill.py → distill_phase3_best.pt) which has consistent arch handling.
- Or pass full arch flags explicitly: `--base-ch 28 --mid-ch 40 --use-zoom-flow` etc — but verify against the actual checkpoint's tensor shapes first via `torch.load(...)["model_state_dict"]["motion.head.bias"].shape`.

**How to apply:**
- Future training runs: explicitly verify checkpoint round-trips before launching downstream eval.
- Treat any new training profile's first few runs as smoke tests; only commit GPU time after the load works.
- Burned 1.2h + $0.30 on this bug today — small cost per occurrence but indicates broader DX rot.
