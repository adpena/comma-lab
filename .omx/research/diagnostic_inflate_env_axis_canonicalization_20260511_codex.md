# Diagnostic inflate-device axis canonicalization (2026-05-11)

## Scope

The PR103-on-PR106 raw-manifest pair proved that CPU/CUDA drift can originate
inside inflate math before the scorer sees frames. This pass adds a canonical,
non-promotable way to isolate that mechanism without cloning or string-mutating
runtime code.

## Code changes

- `experiments/contest_auth_eval.py`
  - adds `--inflate-device {auto,cpu,cuda}`;
  - adds `--inflate-env KEY=VALUE`;
  - applies overrides only to the `inflate.sh` subprocess, not
    `upstream/evaluate.py`;
  - records `inflate_device_policy` and overrides in provenance;
  - demotes any non-auto policy or env override to diagnostic evidence
    (`score_claim=false`, `score_axis=diagnostic_<device>`).
- `experiments/modal_auth_eval.py`
  - adds `inflate_device` and `inflate_env` to the Modal CUDA wrapper;
  - passes `inflate_device` through as `--inflate-device`;
  - passes it through as `--inflate-env` to remote `contest_auth_eval.py`;
  - labels detached metadata as `diagnostic_cuda` when an override is present.
- `submissions/pr103_pr106_final_runtime/inflate.py`
  - honors `PACT_INFLATE_DEVICE={auto,cpu,cuda}`;
  - fails closed on invalid values or unavailable forced CUDA;
  - preserves `auto` as the default contest behavior.

## Guardrail

Allowed override keys are intentionally narrow:

- `PACT_*`
- `INFLATE_*`
- `CUDA_VISIBLE_DEVICES`

`--inflate-device cpu` sets `PACT_INFLATE_DEVICE=cpu` and hides CUDA from the
inflate subprocess. This supports the immediate forced-inflate-device
diagnostic while reducing the chance that secrets or broad process mutations
become score evidence.

## Why this matters for score lowering

For HNeRV-style runtimes that select `torch.cuda.is_available()` inside
inflate, one can now run:

- default CUDA inflate + CUDA scorer;
- forced CPU inflate + CUDA scorer via `--inflate-device cpu`;
- Linux CPU inflate + CPU scorer.

The first two isolate inflate-device effects while holding scorer device fixed.
The third preserves public CPU leaderboard reproduction. Any forced-env run is
diagnostic only by construction; if it finds a better basin, the score-lowering
work is to turn the discovered math choice into contest-compliant default
runtime behavior and then re-evaluate without `--inflate-env`.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_contest_auth_eval.py \
  src/tac/tests/test_modal_auth_eval.py \
  src/tac/tests/test_pr103_pr106_final_runtime_packet.py
```

Result: `74 passed`.

```bash
.venv/bin/python -m py_compile \
  experiments/contest_auth_eval.py \
  experiments/modal_auth_eval.py \
  submissions/pr103_pr106_final_runtime/inflate.py
```

Result: passed.
