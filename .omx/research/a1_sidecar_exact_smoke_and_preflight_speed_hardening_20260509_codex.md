# A1 sidecar exact-smoke + preflight speed hardening (2026-05-09)

## Scope

Codex converted recursive adversarial review findings into durable guards:

- A1 sidecar runtime readiness now has an explicit exact `inflate.sh <archive_dir> <output_dir> <file_list>` smoke producer, separate from bounded import-smoke.
- Exact smoke evidence now requires a real output digest and empty evidence-local blockers; a zero-exit `inflate.sh` that writes no raw output fails readiness.
- A1 sidecar dispatch blockers are recomputed from structured facts instead of preserving stale provisional text blockers forever.
- Recheck telemetry distinguishes planned unproven rechecks from pairs actually rechecked this run, with a remaining-unproven counter.
- Developer/full preflight clean-cache hits run before filesystem/source-index setup, so clean repeat runs do not pay source-index construction.
- `check_no_mps_fallback_default` now caches zero-violation candidate scans with a strong stat fingerprint.

## Current A1 sidecar classification

The local sidecar archive remains **not dispatch-ready**. The latest completed
patched tranche reached full coverage with 436/600 searched pairs carrying
machine-readable scalar-equivalent provenance; 164 legacy pairs still require
`--recheck-unproven-pairs`. The next tranche is running locally under the
patched builder.

Current byte-different archive:

- path: `experiments/results/a1_sidecar_resumable_codex_20260509T_local/submission_dir/archive.zip`
- bytes: `178316`
- sha256: `c7f3d88e1ad23bf8cda987583e702ac57e293b64bc7bfea77902e835d19cea10`
- current blockers: missing 164 pair records, import-smoke is not exact
  `inflate.sh`, missing dispatch claim record, missing exact-eval preflight
  record

No CUDA/GHA dispatch was launched. No MPS result is promoted; MPS remains
advisory only for sweeps and configuration discovery.

## Verification

- `python -m pytest src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q`
  - 32 passed
- `python -m pytest src/tac/tests/test_preflight_all_clean_cache.py src/tac/tests/test_preflight_meta_bugs.py::TestNoMpsFallbackDefault src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q`
  - 52 passed
- `python tools/profile_preflight_latency.py --surface preflight-dev-cli --top 20 --fail-on-surface-failure`
  - passed, 10.556s while the worktree was dirty
- Post-commit repeat measurement:
  - populate run: 9.826s
  - clean-cache repeat: 1.970s

## Remaining gates

- Finish rechecking the remaining 220 unproven sidecar pairs.
- Run exact `inflate.sh` smoke only when willing to emit the full raw output.
- Add active dispatch-claim row proof and exact-eval preflight proof before any remote/GHA/CUDA dispatch.
- Keep Phase 1/T1 and Lane 12-v2 local until they emit runtime-consumed packets with no-op proof and exact custody.
