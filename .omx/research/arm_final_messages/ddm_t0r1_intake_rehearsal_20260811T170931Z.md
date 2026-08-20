The T0 rehearsal is complete and landed in commit `1158f84085`.

Measured `[scorer-free byte/custody apparatus]`:

- Role census: 6 bound, 1 typed-unresolved, denominator 7.
- m37 freshness: 3/3 passed.
- Activation join: 9/9 passed with zero missing levers.
- Retained tree: 615,640 bytes across 19 hashed records.
- No scorer, Modal, GPU, evaluator, or terminal store was touched.

The unresolved role is `terminal_sensitivity_map`. Pass-03 chunks belong to input archive `b8c3b118…`, while the selected archive is `93f8d7b4…`; binding them would violate same-parent freshness. The existing PS135 terminal emitter should produce the correct map after convergence.

Artifacts:

- [Rehearsal memo](/Users/adpena/Projects/pact/.omx/research/ddm_t0r1_intake_rehearsal_20260811.md)
- [Terminal-day runbook](/Users/adpena/Projects/pact/.omx/research/ddm_t0r1_terminal_day_runbook_20260811.md)
- [Retained result](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/t0_rehearsal_pass03/90_RESULT.json)
- [Binding manifest](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/t0_rehearsal_pass03/30_CONTENT_BINDINGS_REHEARSAL.json)

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: PS135 terminal producer; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision/`; fire trigger: Leg A reaches real convergence and writes final `RESULT.json`; action: retain the existing terminal sensitivity-map emission.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 content-binding owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/content_bindings/`; fire trigger: terminal archive, result, safe-run, and sensitivity disposition are complete; action: execute the runbook and require all seven roles plus final-parent equality.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 reseal owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/activation_audit/`; fire trigger: the exact terminal DSL config compiles; action: run the activation join and resolve every missing lever before T1.

## LIVE-HYPOTHESES

- The terminal sensitivity map will bind cleanly because convergence requires dry passes, making the last input and selected archive identical; final hash equality remains untested.
- Renderer and HPAC extraction should remain byte-stable because PS135 mutates only the CPR1 carrier; final parse-back must confirm this.
- The terminal activation join may expose levers absent from the v7.5.2 control because JS1 compiles a different active set.

## DEAD-ENDS

- Binding pass-03 sensitivity chunks to the selected archive is closed: their parent hashes differ.
- Adding another sensitivity-map producer is closed: PS135 already has the correct terminal emitter.
- Treating convergence receipts as absent is closed: they derive losslessly from ordered pass receipts.
- Treating embedded renderer or HPAC bytes as missing is closed: both were extracted losslessly.
- Treating the 9/9 activation control as terminal authorization is closed: terminal JS1 needs its own join.
- Treating this rehearsal as frontier movement is closed: own-vehicle frontier remains **S = 0.16959899569230852 @ 187,226 B** `[contest-CUDA T4, adjudicated, n600]`.