# Paper Proxy Claim Language Audit

Date: 2026-05-08
Owner: Worker D
Scope: claim-language audit for paper and roadmap surfaces that risk
overstating proxy/MPS/CPU evidence, Omega/Omega-OPT predictions, or
single-config negatives.

## Patched In Scope

- `docs/paper/04_results.md`
  - Recast Lane Omega and intN rows as legacy non-ranking projections.
  - Cited the exact int4 T4 JSON that invalidated the old int4 distortion
    model:
    `experiments/results/lightning_batch/apogee_int4_postfix_sanity_20260505T172500Z/contest_auth_eval.adjudicated.json`.
  - Added PR103-on-PR106 exact auth/compliance paths for the active A++ rate
    anchor:
    `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
    and
    `experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.contest_final.json`.
  - Replaced broad "falsified" hidden-gem language with scoped "blocked"
    language and explicit non-family-kill wording.
  - Added evidence-grade guardrails and reactivation criteria for `prediction`,
    `proxy`, `MPS-research-signal`, `CPU-stub`, byte-only, and `A-negative`
    rows.
- `docs/paper/SUBMISSION_CHECKLIST.md`
  - Added pre-submission checklist items for prediction/proxy/MPS/CPU/stub
    wording, scoped negative language, and Omega/Omega-OPT predicted scores.

## Stale Wording Still To Audit Outside This Worker's Write Scope

These surfaces were inspected but not edited because the requested write scope
was limited to `docs/paper/SUBMISSION_CHECKLIST.md`, `docs/paper/04_results.md`,
or a new `.omx/research` ledger.

1. `reports/latest.md`
   - Risk pattern: Lane Omega/Omega-W-V3 wording says "ready" or
     "launch-ready" near predicted bands such as `[0.194, 0.204]`, while the
     same section also identifies CPU-stub/all-ones sensitivity evidence.
   - Safe replacement language:
     "Lane Omega/Omega-W-V3 is a roadmap hypothesis with CPU-stub planning
     evidence only. The predicted band is non-ranking and non-dispatchable
     until a real CUDA sensitivity map exists, static preflight passes, a
     matching active dispatch claim is recorded, and exact T4
     `contest_auth_eval*.json` lands for byte-closed archive bytes."
   - Reactivation criteria: changed score-affecting payload bytes; old/new
     archive SHA-256s; runtime tree SHA-256; no-op/control where applicable;
     exact T4/T4-equivalent CUDA auth eval with component recomputation.

2. `reports/latest.md`
   - Risk pattern: apogee intN predicted bands are discussed as expected
     outcomes or safe-regime calibration after exact int4 had already failed
     badly.
   - Safe replacement language:
     "The intN rows are prediction-only calibration targets. Exact int4 T4
     evidence falsified the old distortion model for that implementation, so
     int5/int6/int7 cannot be ranked or used as monotone evidence until their
     own exact CUDA JSON artifacts exist."
   - Reactivation criteria: per-bit archive SHA-256, runtime tree SHA-256,
     static compliance, exact CUDA JSON, and a reconciler entry comparing
     predicted versus actual distortion.

3. `docs/paper/07_discussion.md`
   - Risk pattern: "killed KL distillation after two failed authoritative
     evaluations" reads as a broad KL-family kill.
   - Safe replacement language:
     "Retired the measured KL-distillation configurations after two
     authoritative failures; related KL/JBL/distillation-family variants remain
     forensic-gated hypotheses until a changed loss/runtime/payload contract
     produces exact CUDA evidence."
   - Reactivation criteria: named config change, old/new payload SHA-256s,
     exact runtime manifest, no-op control, and fresh T4 auth eval.

4. `docs/paper/06_related_work.md`
   - Risk pattern: "the ceiling we hit demonstrates the fundamental limitation
     of the wrapping approach ... regardless of capacity" overgeneralizes a
     measured CPU-lane postfilter result.
   - Safe replacement language:
     "The measured CPU-lane postfilter configuration hit an auth-score ceiling
     under the tested archive/runtime contract. This is negative evidence for
     that implementation, not a proof that all postfilter or neural-wrapping
     variants fail regardless of capacity."
   - Reactivation criteria: changed postfilter architecture or training target,
     byte-closed archive, runtime-budget proof, and exact CUDA auth eval.

## General Rule For Future Paper Passes

Use these exact grades in paper-facing wording:

- `A++`: exact archive/runtime evidence that can rank, with archive bytes,
  archive SHA-256, runtime tree SHA-256, exact CUDA/T4 auth-eval JSON, sample
  count, component distances, and recomputed formula.
- `A-negative`: exact negative for only the measured archive/runtime/config.
  It may block reruns of that config but cannot kill the family.
- `prediction`, `proxy`, `CPU-stub`, `MPS-research-signal`, and byte-only:
  roadmap or diagnostic only; no ranking, promotion, broad retirement, or paper
  empirical anchor.
- `invalid` / `external_quarantine`: compliance or custody lessons only unless
  a corrected byte-closed packet later passes the A++ gates.
