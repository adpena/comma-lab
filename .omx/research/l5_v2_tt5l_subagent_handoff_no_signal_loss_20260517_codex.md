# L5 v2 TT5L Subagent Handoff No-Signal-Loss Ledger - 2026-05-17

Generated at: 2026-05-17T11:13:16Z

Repo commit at capture: `c09f0161d98ffbfcc88c925660d80d8ead56da51`

Scope: preserve late-returning read-only subagent signal before main push. This
ledger makes no score claim, no promotion claim, and no provider-dispatch claim.

Authority flags:

- `planning_only=true`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `provider_spend_attempted=false`

## Subagent 019e356e-9011-7bd3-bc36-3318c8a4ab86

Verdict: read-only exploration only; no files edited. Worktree was clean on
`main`; no recent `.omx/research/*_directive_*` files were found; current L5 v2
TT5L Lightning preflight reported `10/10` cells ready for operator claiming
while remaining `ready_for_provider_dispatch=false`.

Reusable surfaces identified:

- `src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_preflight.py`
  and its JSON/Markdown artifacts already model claim templates, terminal
  templates, harvest probes, false-authority flags, and active-claim conflict
  checks.
- `src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py`
  emits the five-variant by two-axis Lightning SDK specs without provider
  submission.
- `src/tac/optimization/tt5l_sideinfo_variant_packets.py` preserves sideinfo
  packet custody and no-op/liveness checks.
- `src/tac/optimization/l5_v2_measurement_schedule.py` and
  `src/tac/optimization/l5_v2_sideinfo_effect_curve.py` are the fail-closed
  effect-curve validators.
- `src/tac/deploy/lightning/batch_jobs.py` and
  `scripts/launch_lightning_batch_job.py` are the preferred official Lightning
  path; avoid legacy SSH/tmux routing for new TT5L exact-eval execution.

Bundle hardening recommendations:

- Top-level bundle fields should include source plan path/SHA/commit, source
  variant manifest path/SHA, source dispatch plan path/SHA, runtime submission
  directory, runtime tree/content SHA, required variants, required axes, cell
  count, ready cell count, execution order, and global blockers.
- Per-cell fields should include variant, axis, axis label, role, required
  device, lane id, platform, job name, pair group, run id, archive path/bytes/SHA,
  artifact directories, expected result paths, preflight files, claim templates,
  terminal templates, harvest command, and active claim conflicts.
- Harvested evidence must be ready to carry archive SHA, runtime custody,
  hardware, inflate/eval devices, auth-eval command, sample count, component
  distances, recomputed score, formula check, inflated-output manifest, raw
  output aggregate SHA, logs, adjudication JSON, dispatch claim status, and
  sideinfo liveness/pair identity.
- Missing Dykstra feasibility artifact must remain a blocker for promotional or
  provider-ready claims.

## Subagent 019e357f-da29-75c0-b15d-6fd3b2ea82d9

Verdict: read-only adversarial review only; no files edited, no tests run.

Key invariants to preserve:

- Dry-run success is parse-only authority. It must not imply non-dry-run
  readiness because launcher dry-run skips dispatch-claim, identity,
  remote-preflight, and source-manifest submit gates.
- Top-level and per-cell flags must remain `planning_only=true`,
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`,
  `ready_for_provider_dispatch=false`, and `ready_for_non_dry_run_submit=false`
  until exact custody exists.
- The verifier must validate archive/runtime custody itself. Launcher dry-run
  parsing alone does not prove remote archive existence, local byte freshness, or
  staged runtime closure.
- All ten cells must be verified: `{zero, random_lsb, shuffled, trained,
  ablated} x {contest_cpu, contest_cuda}` with no duplicates or extras.
- CPU and CUDA axes are separate evidence spaces. CPU cells must use `--device
  cpu` and avoid CUDA markers; CUDA cells must use `--device cuda` and require
  CUDA/T4 runtime markers.
- Command custody should distinguish source spec command SHA from launcher
  submit-layer command SHA. Equality to the source spec is not the right
  invariant after launcher wrapping, but metadata must preserve the source spec
  SHA and queue command SHA separately.
- Dykstra can be absent for dry-run parsing, but must gate sideinfo proof,
  timing smoke, paired anchor, or promotion-adjacent readiness.

Concrete follow-up tests recommended:

- Fail on stdout/state JSON disagreement for command SHA, source plan, lane id,
  metadata, or spec fields.
- Fail stale harvest output whose scored archive SHA differs from the cell
  archive SHA.
- Verify CUDA cells require DALI/preflight evidence while CPU cells do not.
- Verify effect-curve handoff only when all ten exact-eval cells and pair
  identities are present.

## Subagent 019e354e-3c89-7f82-a7fe-f17ff0323569

Verdict: read-only audit only; no files edited, no tests run.

Gap found: there is no existing end-to-end post-harvest converter from the
Lightning paired-axis plan plus harvested result directories into the
`--cell-json` consumed by `tools/build_l5_v2_sideinfo_effect_curve.py`.

Recommended adapter:

- Read Lightning plan cells.
- Locate each harvested `local_artifact_dir`.
- Normalize harvested `contest_auth_eval(.adjudicated).json` through
  `exact_eval_evidence_from_auth_eval_artifact`.
- Attach `axis`, `variant`, `pair_group_id`, `run_id`,
  `source_variant_manifest`, and sideinfo liveness.
- Emit the cell JSON consumed by `tools/build_l5_v2_sideinfo_effect_curve.py`.

Evidence surfaces:

- `tools/build_l5_v2_sideinfo_effect_curve.py` only reads already-shaped cell
  JSON; it does not know about Lightning plans or result directories.
- `src/tac/optimization/l5_v2_sideinfo_effect_curve.py` validates final cells
  and preserves axis/variant/pair/run identity, but does not retain
  `source_variant_manifest`.
- `src/tac/optimization/l5_v2_probe_intake.py` already contains the reusable
  exact-eval artifact normalizer and should be reused rather than duplicated.

## Next Concrete Artifacts

1. Add a route-unblock packet builder so
   `.omx/research/l5_v2_tt5l_lightning_route_unblock_packet_20260517_codex.*`
   is generated from live artifact SHA/custody fields rather than manual edits.
2. Add a post-harvest Lightning-plan-to-effect-curve-cell adapter.
3. Keep all TT5L/L5 v2 exact-eval authority false until the full ten-cell
   contest CPU/CUDA paired evidence set is harvested with archive/runtime
   custody.
