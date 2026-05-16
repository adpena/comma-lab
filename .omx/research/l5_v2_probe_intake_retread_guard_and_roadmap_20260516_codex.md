# L5 V2 Probe Intake Retread Guard And Roadmap

score_claim=false
promotion_eligible=false
rank_or_kill_eligible=false
ready_for_exact_eval_dispatch=false

## Bug Class Fixed

The TT5L-first L5-v2 readiness surface could keep recommending
`populate_and_evaluate_c1_z5_tt5l_probe_observations` after the probe-intake
artifact already existed and had valid planning structure. In the current
state, that intake is fail-closed for the right reason: C1, Z5, and TT5L still
need paired CPU/CUDA exact observations. Recommending the intake step again is a
retread loop and hides the material frontier action.

The readiness logic now distinguishes:

- no probe template: emit the C1/Z5/TT5L probe template;
- no valid planning intake: run the intake audit;
- valid fail-closed planning intake: build the measurement schedule and paired
  measurement dispatch plan, then fill byte-closed archive/runtime/SHA fields.

## Current L5 V2 Roadmap

Immediate score-lowering unblocker:

1. Materialize paired CPU/CUDA exact work units for C1 world-model foveation, Z5
   predictive coding, and TT5L autonomy.
2. Replace `FILL_ARCHIVE_ZIP`, `FILL_ARCHIVE_SHA256`, and
   `FILL_SUBMISSION_DIR` in each paired work unit with byte-closed candidate
   archives and runtime custody.
3. Dispatch only through the paired Modal CPU/CUDA path after lane-claim and
   operator execute intent are present.

Near-term staircase gates:

1. Convert returned paired results into probe observations with component
   deltas, archive bytes/SHA, runtime tree SHA, raw-output aggregate SHA, logs,
   and terminal lane claims.
2. Rebuild the C1/Z5/TT5L probe gate and choose the architecture-lock candidate
   only from paired evidence.
3. Measure the TT5L side-info effect curve across `zero`, `random_lsb`,
   `shuffled`, `trained`, and `ablated` variants on both CPU and CUDA.
4. Only after those gates, materialize timing-smoke and anchor-pair custody.

Score-lowering frontier work that should not be displaced by local-basin polish:

- TT5L move-level implementation and side-info causal usefulness.
- Z5 predictive-coding world-model anchor.
- C1 world-model/foveation anchor.
- Non-HNeRV and non-PR106 L5-v2 stack candidates, especially Ballé/CompressAI,
  Cool-Chic/C3, SIREN/FINER/WIRE/BACON, wavelet residuals, RAFT/ego-motion, and
  arithmetic/range/ANS compiler passes when they produce byte-closed packets.
- Stack-of-stacks composition only after at least one component has paired
  evidence and payload consumption proof.

## Hardening And Adversarial Review Queue

- Prevent false authority: no score, rank, promotion, or dispatch readiness from
  planning artifacts.
- Preserve CPU/CUDA separation: never promote a CPU-only or CUDA-only result as
  paired evidence.
- Require payload closure: every candidate must prove its archive bytes are
  consumed by `inflate.sh`.
- Require no-op controls for packer/entropy changes.
- Review negative results as measured configurations, not lane deaths, unless
  custody and implementation bugs have been ruled out.

## Validation

Focused regression coverage:

- `test_l5_v2_tt5l_probe_action_advances_after_template_exists`
- `test_l5_v2_tt5l_probe_action_uses_existing_fail_closed_intake`
- `test_l5_v2_paired_measurement_dispatch_plan.py`

These tests pin the intended transition from intake audit to paired measurement
materialization and keep the paired dispatch plan planning-only.
