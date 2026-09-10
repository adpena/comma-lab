**Stopped under the charter’s ambiguity rule; implementation is not complete.**

Pr12 requires referenced JSON to have an “exact byte count and SHA-256” ([line 84](/Users/adpena/Projects/pact/.omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md:84)), but sets `authorization.intent.sha256` to `"<intent digest>"` ([line 308](/Users/adpena/Projects/pact/.omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md:308)). The defined canonical intent digest omits its self field; the file-byte hash includes it.

**Should `authorization.intent.sha256` mean the exact file-byte SHA-256, or the canonical self-omitting intent digest?**

The charter explicitly says: “Where the text is ambiguous, STOP and report the exact sentence.”

The [retained report](/Users/adpena/Projects/pact/.omx/research/ddm_ffi1_implement_prefire_intent_contract_20260910.md) contains the clause-status table and recall evidence. No source changes, tests, timing/scorer/n600 runs, or Modal dispatches occurred; protected trees and the staged index remain untouched. Implementation commit and frozen-receipt SHA: **absent**.

Git object writes were denied. The serializer retained a verified SSD bundle containing report-only commit `88d357477bc168ff4270e955903bfd3cd4fb539a`; it is **not landed on main**.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN/pr12; consumer store pr12 contract; trigger this report’s harvest: resolve the digest meaning.
- **QUEUED-WITH-A-FIRE-ORDER** — owner ddm_ffi1; consumer store charter source/tests and frozen receipt; trigger binding clarification: implement, test, review, and land before RLC2 resumes.
- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer store canonical Git history; trigger restored Git write access: land the retained report bundle.

## LIVE-HYPOTHESES

- File-byte SHA may be intended because it matches the general custody rule.
- Canonical digest may be intended because the authorization explicitly says “intent digest.”

## DEAD-ENDS

- Treating both hashes as interchangeable: they hash different inputs.
- Silently choosing an interpretation: forbidden by the charter’s STOP instruction.
- Claiming implementation or freeze from this report: neither exists.

