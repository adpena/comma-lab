# Premise falsification — precise batch-16 C1 anchor was already canonical

Date: 2026-07-26  
Lane: `lane_original_taskspace_inverse_witness_codec_capstone_20260726`  
Research only: `true`  
Pointer delta: `0`

## Falsified premise

The working premise inherited from the G50 audit was that the historical C1
object had only rounded batch-16 contest-CPU components and that a fresh precise
batch-16 replay was needed before the low-distortion coordinate could guide
rate allocation.

That premise is false. The repository already contained:

`.omx/research/original_taskspace_inverse_witness_codec_20260725/c1_live_target_debt_n600_batch16.json`

- file SHA-256:
  `0db8e47a994cad5367e5eb3028055e667bc4caf3f174026d13171be662e7fbd3`;
- sealed receipt SHA-256:
  `8b083df5b61955cce942b4e69133f28c639f287b3c4e3c9ba0351a57499aa853`;
- written `2026-07-26T00:41:37.774154+00:00`;
- exact n600 batch-16 `d_seg=0.00015196058485243054`;
- exact n600 batch-16 `d_pose=0.00010184347386600314`;
- conditional strict `<0.172` ceiling `187,563 B`;
- full scorer/tool/runtime/distribution custody and same-decoded-raw rounded
  Linux contest-CPU cross-check.

It was already consumed by
`src/tac/witness_dsl/taskspace_inverse_stack_receipt.py` and the sealed
`taskspace_inverse_codec_stack_receipt_v1.json`. Thus this was not an orphaned
file; it was a canonical input that the G50/root targeted recall failed to
consult.

## Impact

The new G54 replay remains useful as an independent reconstruction: it binds
the exact archive/raw/source/upstream closure, emits 38 resumable stage
receipts, and agrees with the prior anchor within
`1.8054371678233316e-9` distortion score units. It is not novelty and should
not displace the stronger canonical anchor.

No further replay is warranted. The active decision stays on full-n600
representation rate and receiver-closed archive construction.

## Root cause

The audit searched historical MS1/C1 score reports and correctly noticed that
the Linux contest-CPU report rounded its components. It did not join that
search to the newer `c1_live_target_debt_n600_batch16.json` or reopen the
task-space inverse stack receipt that already consumed it. Root accepted the
negative without a direct `rg` over the claimed missing coordinate.

This is precisely the rediscovery failure class prohibited by the operating
contract: a scoped negative about one report was accidentally generalized to
the whole repository.

## Durable correction

1. The G51 profiler and G52 full-n600 codec were explicitly redirected to the
   pre-existing canonical anchor as their primary planning coordinate.
2. G54 is labeled independent reproduction and records the prior anchor
   identity and numerical delta.
3. Future “anchor absent” decisions must query both the canonical stack receipt
   and exact-value/path search before launching a scorer replay.
4. Rounded contest authority and precise local planning telemetry remain
   separate: neither launders the other.

## Triality

- DSL: `prior_canonical_anchor` and `independent_reproduction` are distinct
  evidence roles.
- DAG: canonical stack recall -> premise check -> optional independent replay;
  a replay launch is refused when its only purpose is an already-settled value.
- Equation: both anchors imply the same strict byte ceiling, but their
  authority/custody types remain distinct.

## STORES CONSULTED

- `.omx/research/original_taskspace_inverse_witness_codec_20260725/c1_live_target_debt_n600_batch16.json`;
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/taskspace_inverse_codec_stack_receipt_v1.json`;
- `src/tac/witness_dsl/taskspace_inverse_stack_receipt.py`;
- G50 lossy selected-preimage audit;
- G54 replay preflight, stage receipts, and final receipt;
- historical C1 contest-CPU auth-eval receipt.

HISTORICAL_PROVENANCE: append-only premise falsification; no prior empirical
artifact was deleted or weakened.
