# Wave Package Import Surface Hardening

Date: 2026-05-17
Author: codex
Authority: implementation hygiene; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`;
`dispatch_attempted=false`.

## Problem

Three untracked frontier-wave packages preserved useful operator signal while
their package-level import surfaces were racing concurrent landings:

- `src/tac/quantization_wave/__init__.py`
- `src/tac/training_curriculum/__init__.py`
- `src/tac/contest_exploits/__init__.py`

Committing an inconsistent package surface would either make `main` fail on
ordinary imports or create false API authority by exporting helpers that were
not actually present.

## Fix

The packages now expose their state explicitly:

- `IMPLEMENTED_MODULES`: importable, tested helpers currently present.
- `DEFERRED_MODULES`: no-signal-loss roadmap names that are not exported until
  the concrete module and tests land.
- `DEFERRED_RATIONALE`: one-line operator-facing reason for the split.

After the concurrent wave finished, `tac.quantization_wave`,
`tac.training_curriculum`, and `tac.contest_exploits` all have concrete modules
behind their package-level exports. The hardening test protects against future
dangling exports and preserves any genuine deferred item in `DEFERRED_MODULES`
instead of hiding it in prose.

2026-05-17 adversarial follow-up: the first verification pass before pushing
found a real import-surface regression. `tac.quantization_wave.__init__`
imported `mlx_inference_path`, and `mlx.nn` can raise `RuntimeError:
[metal::load_device] No Metal device available` when MLX is installed but the
process has no accessible Metal device. The final fix performs **no top-level
MLX import at all**: the package exposes the optional helper but only probes MLX
inside explicit helper calls. This prevents both test-collection crashes and
noisy `atexit` Metal-device failures during ordinary package imports.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_wave_package_import_surfaces.py -q
.venv/bin/python -m pytest src/tac/quantization_wave/tests -q
.venv/bin/python - <<'PY'
import tac.quantization_wave
import tac.training_curriculum
import tac.contest_exploits
print(tac.quantization_wave.MLX_AVAILABLE)
PY
```

No provider dispatch was attempted. No score claim is made.
