# Codex Memory — Source Doc Structure, Progress Cross-Reference, Rigor Rules

Date: 2026-04-30

## Source-of-Truth Planning Docs

1. `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
   - Role: strategic/scientific design spine.
   - Interprets the paradigm shifts toward the Shannon floor:
     - alpha: mask payload overhaul,
     - beta: sensitivity-aware renderer/codec stack,
     - gamma: joint score-aware codec/entropy/ADMM stack.
   - Use this doc to determine whether implementation matches the intended
     scientific hypothesis and whether a lane is structurally relevant to the
     floor.

2. `.omx/research/contest_grade_all_lane_results_audit_20260430.md`
   - Role: evidence and compliance ledger.
   - Defines Grade A score-grade and Grade A++ 1:1 contest-grade.
   - Use this doc to decide whether a result may rank, promote, kill, or anchor
     floor math. Implementation readiness is not score evidence.

3. `.omx/research/shannon_floor_execution_readiness_20260430.md`
   - Role: dispatch/roadmap/execution ordering.
   - Converts strategy and evidence into the shortest-wall-clock plan.
   - Use this doc to determine what to run next and what can safely parallelize.

4. `.omx/research/external_research_intake_shannon_floor_20260430.md`
   - Role: external papers/OSS intake.
   - Labels ideas as Copy/Translate/Watch.
   - External research can guide implementation, but cannot promote/kill a lane
     without exact contest archive evidence.

## Codex Progress Ledgers

1. `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
   - Tracks implementation progress against the Grand Council paradigm plan.
   - Current status: beta/OWV3 scaffold landed, alpha `.nrv` packaging partially
     unblocked, no new Grade A result.

2. `.omx/research/contest_grade_all_lane_results_audit_20260430_codex_progress.md`
   - Tracks evidence-grade impact of Codex changes.
   - Current status: Grade A table unchanged; OWV3 and Lane 12 remain below
     score-grade until exact CUDA archive eval.

3. `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`
   - Tracks execution state, active blockers, fastest-wall-clock ordering, and
     next-turn checklist.

## Cross-Reference Method

For every implementation or lane result:

1. Compare against Grand Council doc:
   - Does the code implement the intended alpha/beta/gamma hypothesis?
   - Are thresholds, objective terms, and stack assumptions non-arbitrary?
   - Does the lane target real score leverage under
     `100*seg + sqrt(10*pose) + 25*bytes/original_bytes`?

2. Compare against Contest Audit doc:
   - What evidence grade applies?
   - Is exact archive custody present?
   - Is SHA-256 recorded?
   - Is the manifest clean?
   - Is every inflate artifact inside `archive.zip` or fixed contest code?
   - Was `archive.zip -> inflate.sh -> upstream/evaluate.py` used?
   - Was CUDA/T4-equivalent hardware used?
   - Did the score recompute over 600 samples?

3. Compare against Execution Readiness doc:
   - Does this unblock a critical path item?
   - Can it run in parallel with other scientific hypotheses?
   - Does it create or remove a stack dependency?
   - What is the fastest next dispatch or patch?

4. Update the relevant Codex progress ledger:
   - Landed work,
   - evidence label,
   - blockers,
   - next-turn actions,
   - tests/evals run.

## Rigor Principles For AI-Agent-Assisted Development

- Treat agent output as hypotheses until verified by code, tests, artifact
  custody, and exact contest eval.
- Do not let implementation velocity dilute evidence labels.
- Never promote byte savings without scorer distortion.
- Never promote scorer improvements without exact archive byte accounting.
- Never trust CPU/MPS to rank or kill CUDA contest behavior.
- Every side-info table, neural weight, codebook, pose payload, or decoder
  dependency must be counted inside the archive or be fixed contest code.
- Prefer deterministic builders, fixed zip metadata, SHA recording, and
  manifest checks.
- Any adversarial review finding resets the clean-pass counter.
- Scientific claims need quantitative derivatives, error bars or bands, and
  falsifiable kill/promote criteria.
- Mathematical claims must distinguish derivation, prediction, synthetic test,
  empirical byte claim, and contest score.

## Current Codex State

- OWV3 implementation scaffold is landed and tested.
- Sensitivity artifact contract is landed and tested.
- `.nrv` archive validation and resolver support are landed and tested.
- OWV3 deterministic builder and Fisher-to-channel converter are landed.
- Fisher profiler mask decoding has been corrected from grayscale pixels to
  class IDs.
- Canonical archive builder now supports `masks.nrv` manifests and deterministic
  ZIP writing.
- Paper rigor blueprint exists at
  `.omx/research/shannon_floor_paper_rigor_writeup_blueprint_20260430.md`.
- No new Grade A score-grade result has landed.
- Fastest next path:
  1. PFP16 exact CUDA eval.
  2. OWV3 deterministic builder plus per-channel sensitivity artifact.
  3. Lane 12 clean dependency closure plus full CUDA `.nrv` eval.
  4. IMP harvest and hidden-gem recovery lanes.
  5. Gamma coordinator only after measured component streams exist.
