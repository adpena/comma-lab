# CPU/CUDA drift analyzer canonicalization (2026-05-11)

## Scope

This pass canonicalized the exact-pair CPU/CUDA drift analyzer on the active
score-lowering path. It is not a score claim and does not promote any lane.

## Code change

- `tools/analyze_cpu_cuda_eval_drift.py`
  - now uses `tools/tool_bootstrap.py` for repo import setup;
  - now uses `tac.repo_io.read_json`, `write_json`, and `json_text` for
    deterministic JSON custody;
  - keeps the existing split between `valid_for_pair_score_analysis` and
    `valid_for_mechanism_analysis`.

## Adversarial note

The currently available local advisory CPU artifact and recovered Modal CUDA
artifact have different raw-output aggregate SHA-256 values:

- macOS CPU advisory raw aggregate:
  `e7a4b402b0ec381616f625985984dc72cfe386fa060d80e358347701bf6351b1`
- Modal T4 CUDA raw aggregate:
  `99141b32678d60bb736fce21eac1f68fce402e627ed44e7fc9a0c9a76b44c1e7`

That is useful debugging signal, but it is not a contest CPU/CUDA mechanism
claim because the CPU side is `cpu_advisory`, not Linux x86_64 contest CPU.
The pending Modal Linux x86_64 CPU run must be harvested before attributing the
gap to runtime/inflate device behavior versus scorer/loader device behavior.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_analyze_cpu_cuda_eval_drift.py \
  src/tac/tests/test_auth_eval_records.py
```

Result: `18 passed`.

```bash
.venv/bin/python -m py_compile tools/analyze_cpu_cuda_eval_drift.py
.venv/bin/python tools/all_lanes_preflight.py
```

Result: all `29` preflight checks passed.
