**Stopped under the charter’s ambiguity rule; implementation remains incomplete.**

Pr12 requires “Every referenced JSON or manifest” to carry `{path, bytes, sha256}`. Its mandatory historical T4 leg instead contains a nested receipt reference with only `{path, sha256}`—also the shape emitted by the unchanged completion builder. Whether the new requirement applies recursively needs clarification. Both retained files’ hashes match.

The [report and clause table](/Users/adpena/Projects/pact/.omx/research/ddm_ffi2_implement_prefire_intent_contract_20260910.md) preserve the evidence and MAIN’s accepted `file_sha256` / `file_bytes` / `digest` split.

- **Implementation commit / frozen-receipt SHA:** absent.
- **Tests:** 0; no source changes or Python review passes.
- **Report fallback commit:** `2995c0a81e5aff07187c463393bc3857ae27c42e`, independently verified. Sandbox Git-object denial prevented landing; MAIN and staged index remain unchanged.
- No Modal, fires, timing windows, scorer/n600 runs, payload changes, or edits to protected files, upstream, PR/sealed trees, sibling directories, timing requirements, or dispatch guards.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: [verified bundle](</Volumes/VertigoDataTier/pact/ddm_ffi2_prefire_contract/serializer/20260910T154044.914122Z-74989/intended-commit.bundle>); trigger: harvest in a Git-writable environment. Land the two report artifacts.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN/pr12; consumer store: the linked report and normative contract; trigger: STOP-packet harvest. Clarify nested legacy reference scope.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: ddm_ffi2; consumer store: charter-named source/tests and `ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`; trigger: binding clarification. Complete implementation, reviews, tests, and freeze.

## LIVE-HYPOTHESES

- New-reference-only custody may be intended because pr12 explicitly preserves legacy timing objects.
- Supplemental recursive bindings may satisfy both requirements because the historical receipt exists and hashes correctly; no supplemental schema is authorized yet.

## DEAD-ENDS

- The original digest ambiguity is resolved by MAIN’s split.
- Editing the historical leg would break its mandatory byte/hash pin.
- Claiming implementation, freeze, or positive-control success is unsupported; none occurred.