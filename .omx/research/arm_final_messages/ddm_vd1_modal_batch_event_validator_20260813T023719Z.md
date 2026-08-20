READY_TO_FIRE and landed in commit `5afda1c9a7272b5deace39b45a723a0b0326acc5`.

The validator decodes CP135 once, validates all 200 singleton events through exact upstream T4 scorers, retains every payload/tensor/delta, and resumes from immutable Modal-volume checkpoints. It was not fired: MAIN owns dispatch, and the stale ps135 scorer claims require reconciliation.

K arithmetic: `393.566 + 200×1.211750 + 300 = 935.916 s`, versus 1,800 seconds. Conservative `Kmax=913`, so all 200 fit.

Pinned command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/ddm_vd1_modal_batch_event_validator.py::main \
  --archive /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip \
  --runtime /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime \
  --event-store /Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200 \
  --jo1-analysis /Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json \
  --output-dir .omx/state/ddm_vd1_modal_batch_event_validator/ddm_vd1_20260812 \
  --k 200 --run-id ddm_vd1_20260812 --resume-from ddm_vd1_20260812 \
  --lane-id ddm_vd1_modal_batch_event_validator \
  --instance-job-id modal:ddm_vd1_20260812 --claim-agent codex:ddm_vd1 \
  --detach --provider-detach-ack
```

Validation: 9 tests passed; Ruff, AST, diff check, P0 retention audit, and two review passes passed. Developer preflight was 17/25 green; all eight red denominators were pre-existing shared-repository findings, so no release-green claim is made.

Full handoff: [ddm_vd1_modal_batch_event_validator_20260812.md](/Users/adpena/Projects/pact/.omx/research/ddm_vd1_modal_batch_event_validator_20260812.md).

Effective frontier remains **CP135 S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**. Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN scorer-lane router. **Consumer store:** `comma-ddm-vd1-event-validator-retained/ddm_vd1_20260812/`. **Fire trigger:** reconcile ps135 claims, clear all scorer/Modal work, and adjudicate release preflight; then run the pinned K=200 command.
- **Disposition:** QUEUED-BEHIND-VALIDATOR. **Owner:** MAIN JO1 composer/exact-row router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/final_t4/`. **Fire trigger:** harvested interaction-safe selection clears the 0.000216 projection bar and produces a retained archive no more than +3 B.
- **Disposition:** QUEUED-WITH-A-FIRE-ORDER behind falsifier. **Owner:** future MAIN-routed EC1 producer. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/gen2_lane_over_road/`. **Fire trigger:** generation-1 optimistic n600 gain is below 0.000216; target LC1’s 5,557 Lane-over-Road pixels.

## LIVE-HYPOTHESES

- Full K=200 validation will fit because the 935.916-second projection includes a 10× per-event charge and 300-second storage reserve.
- A strict n600 singleton filter may recover useful events without JS7’s pose debt because it admits positive flip gain only within the corrected per-event global pose budget.
- If generation 1 fails, Lane-over-Road grammar-v2 remains plausible because LC1 measured a concentrated 5,557-pixel failure stratum.

## DEAD-ENDS

- n32 singleton sums are not composition authority: JS7 reversed Seg sign and accumulated excessive pose debt at exact n600.
- The JS7 44-event instance is closed at `S=0.16342603740620176`; the JO1 six-event instance is closed at `S=0.1621711682636563`.
- Local or MPS scoring is forbidden for this CUDA-locked vehicle.
- K independent full archive decodes are unnecessary; the landed validator decodes once and rerenders only affected frames.

