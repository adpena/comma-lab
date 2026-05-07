# SJ-KL Runtime Apply Proof Hardening - 2026-05-07 Codex

## Scope

Ownership slice: Wave-Omega / SJ-KL integration readiness only.

This patch tightens the first exact-evaluable SJ-KL archive step. It does not
dispatch GPU work and does not claim score movement.

## Patch

- `experiments/build_sjkl_c067_archive.py` now fails closed unless the charged
  residual member is the default inflate-runtime name `sjkl.bin`.
- The builder records a typed `sjkl_payload` block, output runtime member names,
  and a `sjkl_top_level_runtime_contract_v1` manifest block.
- The runtime contract includes a local `sjkl_runtime_apply_proof_v1`: the
  builder imports the actual robust-current inflate runtime, decodes `sjkl.bin`,
  applies it to a probe JointFrameGenerator pair tensor at the payload target
  shape, and requires a nonzero pixel delta before emitting an archive.
- `scripts/remote_lane_sjkl_c067.sh` now invokes the builder with
  `--sjkl-member-name sjkl.bin` and forwards `SJKL_MAX_BYTES` into the typed
  manifest cap check.

## Evidence

Evidence grade: `empirical` guardrail and dispatch-readiness structure only.

Focused checks run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_sjkl_residual.py src/tac/tests/test_sjkl_cuda_scorer_wiring.py src/tac/tests/test_sjkl_basis.py src/tac/tests/test_prepare_sjkl_pair_tensors.py src/tac/tests/test_build_sjkl_c067_archive.py src/tac/tests/test_remote_lane_sjkl_c067_script.py src/tac/tests/test_inflate_renderer_sjkl_runtime.py -q
bash -n scripts/remote_lane_sjkl_c067.sh
.venv/bin/python -m py_compile experiments/build_sjkl_c067_archive.py
```

Result: 82 pytest tests passed, shell syntax passed, and Python compile passed.

## Next Exact-Evaluable Action

Produce a PR106 or C067 SJ-KL candidate archive with
`experiments/build_sjkl_c067_archive.py`, inspect the emitted
`runtime_contract.runtime_apply_proof`, then claim the lane before any remote
CUDA eval dispatch. Exact score evidence still requires
`archive.zip -> inflate.sh -> upstream/evaluate.py` with CUDA and
`SJKL_REQUIRE_APPLIED=1`.
