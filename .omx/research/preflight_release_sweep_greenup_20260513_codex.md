# Preflight release-sweep greenup (2026-05-13)

## Summary

Scope: strict release-preflight blockers surfaced while keeping Pact's exact-eval
and proxy-score custody rules intact. No GPU dispatch was launched and no score
claim was created.

## Fixes landed

- Hardened `check_inflate_sh_handles_br_centrally` so early
  `PYTHON_INFLATE` helper predicates in `inflate.sh` do not mask the real
  per-video branch dispatch after Stage 0 brotli handling.
- Hardened `check_no_proxy_metric_drives_decision` so raw provider snapshots
  under `reports/raw/` and the dispatch-claim ledger are treated as custody
  evidence, not live promotion/kill decision records.
- Hardened `check_remote_lane_argparse_arity` to resolve shell arrays used as
  reusable training argument vectors, including command arrays whose Python
  invocation starts on the same line as the array assignment.
- Added AppleDouble cleanup before auth-eval paths in 12 substrate remote
  scripts and restored executable bits for the three substrate scripts that
  were missing them.
- Backfilled local, score-neutral E2E smoke proofs for 15 substrate lane
  scripts and a plumbing-only `cool-chic-sidecar` lane-class proof in
  `.omx/state/`. These state files remain gitignored per repository policy.
- Patched five Claude memory files with explicit PCC4 Grand Council,
  internal-consistency, and reactivation sections so kill/defer memories remain
  adversarially reviewable.

## Verification

- `.venv/bin/pytest` focused preflight regression suite:
  `24 passed in 1.51s`.
- `.venv/bin/python -m py_compile` on all modified Python source/test files:
  passed.
- `.venv/bin/python tools/all_lanes_preflight.py`:
  `ALL 30 PREFLIGHT CHECKS PASSED`.
- `.venv/bin/python -m tac.preflight --scope dev`:
  `PREFLIGHT PASSED` in 8.463s wall.
- `.venv/bin/python -m tac.preflight --scope all --allow-slow-preflight`:
  `PREFLIGHT PASSED` in 62.12s wall.

## Remaining DX performance gap

The full release sweep still exceeds the operator's desired 30s ceiling. The
timing profile at
`.omx/research/artifacts/preflight_dx_profiles_20260513_codex/preflight_all_timing_after_pcc4_memory_fix.json`
shows the largest wall contributors are broad repeated scans:

- `check_ast_walker_handles_both_assign_and_annassign`: 8.47s
- `preflight_filename_contract`: 7.66s
- `check_test_files_imports_resolve`: 7.25s
- `check_no_proxy_metric_drives_decision`: 6.37s
- `check_no_compromised_lightning_supply_chain`: 6.35s

Next optimization should consolidate these into the existing source-index /
single-open-per-file path instead of adding more independent recursive scans.

## Score-lowering relevance

This was infrastructure work, not a candidate-score tranche. The score-lowering
impact is indirect but necessary: release preflight now permits claimed
exact-eval packets to move without false blockers from raw provider snapshots,
shell-array dispatch wrappers, AppleDouble artifacts, or missing local smoke
proofs.
