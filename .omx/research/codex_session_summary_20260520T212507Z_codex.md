# Codex Session Summary

Generated: 2026-05-20T21:25:07Z
Author: Codex

## Concrete landings

- Spawned four xhigh adversarial review workers for fast-moving landings:
  - sparse residual / scorer-gradient review
  - partner landing review
  - authority/math rigor review
  - PR110 safety/custody review
- Added scorer-gradient sparse residual tooling:
  - `src/tac/optimization/scorer_gradient_sparse_residual.py`
  - `tools/run_scorer_gradient_sparse_residual_smoke.py`
  - `src/tac/tests/test_scorer_gradient_sparse_residual.py`
- Added upstream-target decode helper:
  - `tools/decode_upstream_video_to_raw.py`
- Hardened sparse residual smoke tooling:
  - scorer-gradient candidate directories now include target-raw SHA-derived
    IDs and refuse accidental overwrite without `--overwrite-candidate`
  - sparse-residual oracle smoke received the same SHA-derived candidate ID and
    overwrite guard
  - no-visible-change advisory skips can now clean candidate raw outputs when
    `--cleanup-candidate-raw` is set
- Added reusable LL/backprop primitive:
  - `select_budgeted_gradient_residuals(...)` consumes arbitrary gradients,
    saliency masks, byte costs, and a budget limit, then water-fills by
    predicted objective gain per byte.
- Fixed xhigh-found selector correctness bug:
  - both gradient selectors now reject gradient/residual tensors whose H/W does
    not match `RawVideoShape`
  - both selectors now reject out-of-range frame indices before converting to
    global raw-pixel indices

## Empirical result

Corrected-target scorer-gradient sparse residual smoke:

- Target decode helper SHA: `bb9cb031acc7d9898d28618d49b256cc9f2e9cc92a25327acd4a9061f6565907`
- Prior ffmpeg target SHA: `4f1ca43f44f3a7c83e78162cbe5c82d845416e7b9496b6ba743fdb64ee67b23a`
- Result JSON: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_gradient_sparse_residual_smoke_20260520_codex/scorer_gradient_pose_pair508_k512_d1_upstream_target_20260520_codex.json`
- Local pair 508 PoseNet improved: `-3.2431562431156635e-06`
- Full `[macOS-CPU advisory]` score regressed:
  `0.19206142414659494 -> 0.19339092414659495`
- Delta: `+0.0013295000000000112`
- Score claim: false

Conclusion: pair-local backprop alone is insufficient. The next LL lane should
learn or fit a fuller scorer-response surrogate and allocate residual bytes by
held-out predicted full-score improvement per byte.

## New/adopted lanes

- `lane_ll_hinton_distilled_scorer_saliency_residual_20260520`
  - L0, `research_only=true`, `lane_class=optimization_tool`
  - Reactivation gate: build a local scorer-surrogate dataset from existing
    baseline/target pairs and sparse-residual failures; require held-out
    advisory correlation before widened candidate spend.

## Review findings consumed

- `codex_findings_xhigh_sparse_residual_gradient_review_20260520T2115Z_codex.md`
  - Fixed the gradient H/W and frame-index guard bug.
  - Patched future artifact overwrite/custody risk in both sparse-residual
    smoke tools.
- `codex_findings_xhigh_authority_math_rigor_review_20260520T2115Z_codex.md`
  - Identified stale Percepta mechanics language, false precision in
    advisory-score memos, stale `--archive-bin archive.zip` commands, and
    overbroad negatives that should be weakened in follow-up memo cleanup.
- `codex_findings_xhigh_pr110_safety_and_custody_review_20260520T2115Z_codex.md`
  - Live PR110 custody looks coherent; local provisional PR110 tree and local
    README/manifest mirror remain contamination risks.
- `codex_findings_xhigh_partner_landing_review_20260520T2115Z_codex.md`
  - Partner batch has transactional risks: tracked imports depending on
    untracked modules, public-surface transitive docs risk, selector V2
    authority-label drift, and stale deleted-path references.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_gradient_sparse_residual.py src/tac/tests/test_sparse_residual_oracle.py src/tac/tests/test_inflate_postprocess_surface.py`
  - `14 passed`
- `.venv/bin/python -m py_compile tools/run_sparse_residual_oracle_smoke.py tools/run_scorer_gradient_sparse_residual_smoke.py tools/decode_upstream_video_to_raw.py src/tac/optimization/scorer_gradient_sparse_residual.py src/tac/optimization/sparse_residual_oracle.py`
  - passed
- `git diff --check` on touched files
  - passed
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1078 lane(s) validated cleanly`

## Immediate next actions

1. Patch authority memo language flagged by the xhigh authority review:
   supersede stale Percepta mechanics wording, replace stale `--archive-bin`
   examples, and mark advisory scores as rounded-report-derived.
2. Patch partner transactional issues before any commit serialization:
   tracked import edits must land with their untracked dependency modules, or
   re-export edits should be deferred.
3. Clean or sentinel the ignored PR110 provisional tree so its many
   `archive.zip` and `archive_charge_proxy.zip` files cannot be mistaken for
   submission custody.
4. Continue LL as a $0 backprop/surrogate lane:
   build candidate-response rows from existing smokes, train or fit a small
   held-out surrogate, then generate saliency-masked residuals with the new
   budgeted selector.
