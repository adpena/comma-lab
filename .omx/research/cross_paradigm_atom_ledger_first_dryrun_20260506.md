# Cross-paradigm atom ledger — first dryrun (claude:main 2026-05-06)

**Result artifact:** `reports/cross_paradigm_atom_ledger_20260506.json`
**Source input:** `experiments/results/hnerv_decoder_recode_pr106_20260506_codex/profile.json` only
**Council verdicts applied:** see
`grand_council_meta_lagrangian_pareto_design_decisions_20260506.md`

## Headline findings

| Metric | Value |
|---|---|
| Total atoms in ledger | **8** |
| Pareto-frontier atoms (Q4 verdict: family_group scope) | **1** |
| Atoms with closed-byte archive manifest | **0** |
| Atoms eligible for parallel dispatch (Q1 verdict requires) | **0** |
| Families represented | 1 (hnerv_rate_recode only) |

## The lone Pareto-frontier atom

```
hnerv_rate_recode:public_pr106_belt_and_suspenders:brotli_q10_current_raw
  byte_delta:               -151 bytes
  expected_total_score_delta: -0.00010 (-10 millibp on PR106)
  expected_seg_dist_delta:    0.0
  expected_pose_dist_delta:   0.0
  pareto_frontier:           True
  dispatchable:              False
  archive_manifest_attached: False
```

The 7 non-frontier atoms are all DOMINATED — they spend more bytes (+2.8KB
to +96.7KB) for proportionally weaker score deltas (+0.0019 to +0.064).
brotli_q10 wins because it actually SAVES 151 bytes for free score.

## Top-level dispatch blockers (every atom inherits these)

- `planning_only_atom_ranking`
- `requires_stack_interaction_review`
- `requires_exact_cuda_auth_eval`
- `cross_paradigm_adapter_rows_require_source_review`

## Operator decisions surfaced

Per the Grand Council Q1+Q2 verdicts (parallel-fan-out the Pareto frontier;
atom needs Pareto membership + closed-byte manifest), the actual gating
work BEFORE any dispatch is **planning-time, not GPU-time**:

1. **Per-atom source review** (Q3 verdict: verified interaction assumptions).
   The single Pareto-frontier atom needs an explicit source review confirming
   the brotli_q10 recode is bit-equivalent to the q11 baseline and produces
   a decode-roundtrip-clean archive. Code path: `tac.hnerv_decoder_recode`.

2. **Closed-byte archive manifest** for `brotli_q10_current_raw`. The
   profile artifact at
   `experiments/results/hnerv_decoder_recode_pr106_20260506_codex/profile.json`
   contains the byte counts but not a packaged candidate archive. Need
   the actual recoded archive bytes + sha256 + manifest before dispatch.

3. **Other-family inputs** to expand the Pareto frontier beyond
   single-family hnerv_rate_recode. The CLI accepts:
   - `--wr01-wavelet-plan <path>`
   - `--categorical-mask-plan <path>`
   - `--lapose-plan <path>`
   - `--foveation-plan <path>`
   Without those, the cross-paradigm advantage of the Pareto-gate is
   structurally absent (verdict Q4 default scope is `family_group`, but
   without multiple families the frontier is one-dimensional).

## What is NOT a Tier-1 GPU spend yet

- The SINGLE Pareto-frontier atom predicts a -10 millibp improvement
  (-0.00010 score). On its own, this is below the typical contest-CUDA
  measurement noise floor (~1-5 millibp). Dispatching it standalone is a
  measurement loss.
- The Council's parallel-frontier rule (Q1 verdict A) needs the frontier
  to span multiple families; right now it's 1 atom in 1 family.
- Therefore: **no GPU spend is justified by this ledger run.** The blocker
  is upstream (more family inputs + per-atom source review), not downstream
  (operator authorizing dispatch).

## Recommended next planning-time actions (no GPU)

1. **Run codex's other paradigm planners** (WR01 wavelet, categorical
   mask, lapose, foveation) to produce input plans for re-running the
   ledger. These planners exist per `cross_paradigm_atoms_20260506_codex.md`.
2. **Build closed-byte archive for brotli_q10** by running
   `tac.hnerv_decoder_recode` against the PR106 baseline.
3. **Re-run the ledger** with all 5 family inputs to discover the
   cross-family Pareto frontier (which is the actual "new floor"
   structure the user referenced).

## Cross-references

- `grand_council_meta_lagrangian_pareto_design_decisions_20260506.md` —
  Council deliberation
- `meta_lagrangian_pareto_gate_20260506_codex.md` — codex's Pareto-gate
- `cross_paradigm_atoms_20260506_codex.md` — 5-family adapter layer
- `feedback_may_4_hnerv_race_postmortem_20260505.md` — race-mode rule
  (parallel actuator only fires when frontier has dispatchable atoms)
- `tools/build_cross_paradigm_atom_ledger.py` — re-run command
