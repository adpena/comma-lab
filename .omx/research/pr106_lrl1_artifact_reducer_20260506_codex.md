# PR106 LRL1 Artifact Reducer - 2026-05-06

## Context

PR106 LRL1 is the third score-aware sidechannel after latent and yshift. Before
this tranche, the local builder could only emit zero-smoke bytes or fail closed
for CUDA search stubs. That left no deterministic handoff path from a remote or
offline search job into charged LRL1 archive bytes.

## Change

- Added `--search-mode artifact` to
  `experiments/build_pr106_lrl1_sidechannel.py`.
- Added `--basis-npy`, `--coeffs-npy`, and `--artifact-manifest` inputs.
- Added strict artifact custody checks:
  - `.npy` arrays load with `allow_pickle=False`
  - basis dtype/shape must be `int8` and `(K, low_h, low_w)`
  - coeff dtype/shape must be `int8` and `(n_frames, K)`
  - manifest schema must be `pr106_lrl1_artifact_manifest_v1`
  - manifest must keep `score_claim=false`, `ready_for_builder=true`, and
    dispatch flags false
  - source archive, basis, coeff SHA-256s and all shape/step parameters must
    match the builder invocation
- Extended the PR106 sidechannel dry-run guard so the LRL1 builder surface
  includes the new artifact-mode flags.

## Verification

Focused:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_pr106_lrl1_sidechannel.py \
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py -q
```

Result: `39 passed`.

Full preflight:

```text
.venv/bin/python tools/all_lanes_preflight.py --timings
```

## Promotion Status

This is not score evidence and does not dispatch CUDA work. It unlocks a
deterministic local reduction path for future LRL1 search artifacts. Any built
archive remains blocked on exact CUDA auth eval, runtime tree hash provenance,
and lane-claim discipline before dispatch.
