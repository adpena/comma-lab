# A1 sidecar exact-smoke + preflight speed hardening (2026-05-09)

## Scope

Codex converted recursive adversarial review findings into durable guards:

- A1 sidecar runtime readiness now has an explicit exact `inflate.sh <archive_dir> <output_dir> <file_list>` smoke producer, separate from bounded import-smoke.
- A1 sidecar dispatch blockers are recomputed from structured facts instead of preserving stale provisional text blockers forever.
- Recheck telemetry distinguishes planned unproven rechecks from pairs actually rechecked this run, with a remaining-unproven counter.
- Developer/full preflight clean-cache hits run before filesystem/source-index setup, so clean repeat runs do not pay source-index construction.
- `check_no_mps_fallback_default` now caches zero-violation candidate scans with a strong stat fingerprint.

## Current A1 sidecar classification

The local sidecar archive remains **not dispatch-ready**. The latest completed
pre-patch tranche reached full coverage but only 380/600 searched pairs had
machine-readable scalar-equivalent provenance; 220 legacy pairs still require
`--recheck-unproven-pairs`. The next tranche is running locally under the
patched builder.

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
