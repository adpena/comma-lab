# ddm_pf2x_preflight_chain_burndown — cure the mirror-helper meta-gate's 10 sites, then LOOP the full-preflight chain until dry or non-mechanical

## MANDATE

The r56 full-preflight chain (receipt
`.omx/tmp/preflight_full_r56_20260826/PREFLIGHT_RESULT.json`) is RED on the OSS-export
MIRROR HELPER meta-gate: 10 functions inside `src/tac/preflight.py` rglob()
`experiments/` or `src/tac/` without calling `_is_oss_export_mirror_path` within ±15
lines — so those checks re-scan the `comma_lab_public_export/` staging copies and can
double-report canonical findings. Sites (line numbers at commit `7d7ddfc304`-era HEAD;
re-derive live): `_check_154_manifestless_cleanup_identity`:50776 ·
`_check_199_iter_candidate_files`:57729 · `_check_209_iter_target_files`:59656 ·
`_check_210_iter_target_files`:59842 · `_check_211_iter_target_files`:60026 ·
`check_modal_dispatches_register_call_id`:63566 ·
`check_slim_ranker_consumes_canonical_taylor_proxies`:67307 ·
`check_falling_rule_list_canonical_use`:67385 ·
`check_rashomon_ensemble_continual_update_locked`:67459 ·
`check_compressive_landscape_canonical_use`:67538.

1. PER-SITE adjudication (the mg1 discipline — zero blanket moves): for each site
   choose the gate's own sanctioned cure — (a) filter the rglob through
   `_is_oss_export_mirror_path` (correct when the check scans REPO SOURCE and the
   mirror copy is a duplicate of a canonically-scanned file — the expected common
   case); (b) route through `_iter_python_files`/`_iter_shell_files`; or (c)
   `# preflight-mirror-skip-ok: <substantive reason>` within ±8 lines ONLY where
   scanning the mirror is genuinely intended/harmless. Each site's disposition recorded.
   BEHAVIOR CHECK per cure: the check's own test(s) stay green; where a check has no
   test, run it strict before/after and diff the violation set (only mirror-path rows
   may disappear).
2. THEN LOOP THE CHAIN: re-run `preflight_all(wall_clock_budget_s=None)` (the r-series
   runner pattern under `.omx/tmp/preflight_full_r5*_20260826/run.py`; write r57, r58, …
   receipts). For each next RED: if the cure is MECHANICAL HYGIENE of the same genus
   (scanner scope, stale fixture, waiver-with-real-rationale, generated-file
   regeneration, historical-tag annotation — the classes already burned this session:
   Check 126 blocklist · no-mps scope/sites · this mirror gate), adjudicate + cure +
   commit + continue. STOP AND REPORT as a typed blocker (do NOT improvise) on any red
   whose cure would touch: score-relevant semantics, sealed custody/receipts, archive
   bytes, canonical equations content, launch configs, or anything needing a
   measurement. Budget: stop after ~6 gate-cure rounds or when a chain run comes back
   GREEN, whichever first; report the chain's final state either way.
3. Ledger rows via tools/canonical_task_status.py (actor ddm_pf2x); per-gate disposition
   table in the memo.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal, NO scorer, NO archive mutation. `.py` = 2 genuine
  review passes per edit batch; serializer commits w/ post-edit shas; #1293 bundle on
  git denial (memo `.omx/research/ddm_hd1_apparatus_two_landings_20260826.md`).
- NEVER weaken a gate: after each cure run the gate's POSITIVE direction where a control
  exists or can be synthesized cheaply (the mg1 bar: a synthetic violation still fires).
- Preflight runs are LONG (~10-16 min): run them via
  tools/launch_detached_process.py with a done-receipt and poll, never a foreground
  subprocess (the reaper kills ~3 min foregrounds — m139).
- Memo corrections APPEND-ONLY on HISTORICAL_PROVENANCE files; generated files
  (reports/, graph_memory synthesized) are fixed at their GENERATOR, never hand-edited.

## PRIOR NEGATIVE SIGNAL

- mg1's burn-down (memo `.omx/research/ddm_mg1_mps_gate_burndown_20260826.md`, commit
  `7d7ddfc304`): 21→0 with ZERO waivers, positive control executed — the reference form
  for per-site adjudication; #821: N sites of one copied pattern = ONE fact (fix the
  class, count honestly).
- The 5fcf9c1c7f scope-fix precedent: 97.8% of a "huge" violation list was ONE scratch
  copy — measure the population by bucket BEFORE editing.
- #842: these gates accrued violations in the dark (commit hook runs none of them) —
  expect stale populations, not fresh regressions; date the debt honestly.

## OPTIMAL FORM

- Family REFERENCE exemplars w/ provenance pins: mg1's disposition-table memo
  (`7d7ddfc304`) · the no-mps scope fix (`5fcf9c1c7f`, src/tac/preflight.py exemption
  rationale comments) · the Check-126 blocklist cure (`1f46e207d6`) · the r55/r56
  runner scripts under `.omx/tmp/preflight_full_r5*_20260826/`.
- SCOPE reductions declared: ≤6 gate-cure rounds this arm; judgment-heavy reds exit as
  typed blockers. MECHANISM reductions FORBIDDEN: no gate deletions, no blanket
  exemptions, no waivers with placeholder rationales.
- **PRIOR-LAW PREDICTION (falsifiable):** the 10 mirror sites all take cure (a)/(b)
  with zero behavior change outside mirror paths, and the chain's subsequent reds are
  predominantly the same dark-window hygiene genus, yielding ≥3 more gates cured this
  arm. FALSIFIER: a red requires semantic/custody judgment early — then the arm stops
  there with the typed blocker and the chain's burn-down cadence returns to
  MAIN-per-gate.

## DELIVERABLE

`.omx/research/ddm_pf2x_preflight_chain_burndown_20260826.md` — 10-site disposition
table + per-round chain receipts (rN status → gate → cure → controls) + final chain
state (GREEN or the typed blocker) + ledger rows + GESTALT-DELTA line. Serializer
commits (or bundles). End with the own-vehicle frontier line.
