# Lossy_coarsening T0312 retired config — DO NOT re-dispatch ledger

**Date**: 2026-05-08
**Author**: Subagent FIX-CODEX-FINDINGS
**Scope**: durable cross-agent ledger; complementary to user-private memory file
   `feedback_lossy_coarsening_T0312_retired_config_do_not_redispatch_20260508.md`.

## Operational rule (binding)

**Lossy_coarsening Lightning T4 dispatch must use a NEW config (not the
T0312-noproject retired config) to avoid wasting GPU spend on a known-negative
result.**

## Evidence row pinned

`reports/cathedral_autopilot_evidence.jsonl:35` records the prior CUDA result
for this lane:

| Field | Value |
|---|---|
| `job_name` | `lossy-coarsening-cuda-20260508T0312-noproject` |
| `score_contest_cuda` | `0.351718793322788` |
| `evidence_grade` | `[contest-CUDA A-negative]` |
| `measured_config_status` | `measured_config_retired_exact_cuda_negative` |
| `archive_sha256` | `ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28` |
| `archive_bytes` | `156404` |
| `segnet_distortion` | `0.00186125` |
| `posenet_distortion` | `0.00037762` |
| `rate` | `0.00416572` |

The 0.3517 score is dominated by `seg + sqrt(10*pose) = 0.00186 + 0.0613 = 0.0632`
distortion vs `rate = 0.00417`. Per-tensor `K_budget=0.05` produced too much
weight distortion for the renderer to compensate. Per CLAUDE.md
`forbidden_premature_class_level_falsification` + KILL-is-last-resort: this
is **per-config retirement, NOT a class-level kill** of `lossy_coarsening_analytical`.

## Strategic context (Codex finding #4)

Codex adversarial review (memory:`feedback_codex_adversarial_review_4_landings_20260508.md`)
confirmed:
- Landing 1 Lightning canonical bootstrap fix (commit `256d6fe1`) is
  engineering-sound; all 31 guard tests pass; bootstrap delegation to
  `scripts/remote_archive_only_eval.sh` is correct; `forbidden_remote_bootstrap_inline`
  rule satisfied.
- The strategic blocker is NOT the bootstrap fix; it is reusing the same config
  the bootstrap fix would dispatch. If `experiments/lossy_coarsening_lightning_cuda_test.py`
  is invoked with default args (matching T0312), it will reproduce the negative
  result and burn $0.30-0.60.

## Pre-dispatch checklist (mandatory)

Before any new lossy_coarsening Lightning T4 dispatch:

1. **Config differs from T0312** in at least one of: `K_budget`, `rms_target`,
   source state_dict, brotli params, per-tensor allocation strategy. Document
   the difference in the dispatch claim notes.
2. **Predicted score (cathedral_autopilot) < active HNeRV anchor** (PR103-on-PR106
   = 0.20898 as of 2026-05-08). If predicted ≥ 0.21, do not dispatch.
3. **At least one reactivation criterion** from the T0312 evidence row addressed:
   - retrain or jointly optimize the renderer under scorer-aware loss instead
     of applying post-hoc lossy coarsening
   - prove byte-closed runtime packet with component-risk mitigation and exact
     CUDA score below the active HNeRV anchor
   - classify whether the loss is SegNet, PoseNet, or runtime/harness driven
4. **Pre-dispatch sanity ladder** clearance (`experiments/predispatch_sanity.py`)
   if a basin-parity anchor exists for this config family.

If all four items are satisfied AND the operator has authorized GPU spend
explicitly, proceed; the dispatch claim must reference this ledger and the
T0312 evidence row.

## Why this ledger exists

CLAUDE.md "race-mode rigor inversion" rule: agent default is action when the
parallel actuator is ready. That default is correct for NEW configs but wrong
for known-negative configs. This ledger acts as the pre-dispatch guardrail
extinguishing the bug class "agent re-fires recently-RETIRED config because a
wrapper-fix landed and the actuator is ready".

## Cross-references

- `feedback_codex_adversarial_review_4_landings_20260508.md` (codex source memo)
- `reports/cathedral_autopilot_evidence.jsonl:35` (T0312 evidence row)
- `experiments/lossy_coarsening_lightning_cuda_test.py` (entry point; Landing 1
  commit `256d6fe1`)
- `feedback_lossy_coarsening_lightning_6th_failure_use_canonical_bootstrap_20260508.md`
  (postmortem driving Landing 1)
- `.omx/research/lossy_coarsening_exact_cuda_result_review_20260508_codex.json`
  (formal review packet)
- `.omx/research/lossy_coarsening_exact_cuda_adversarial_review_20260508_worker_b.md`
  (adversarial review of T0312 result)
- `.omx/state/next_experiments.md` (operational gate, on-disk-only — gitignored)
