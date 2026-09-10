Staged move 43’s archive, runtime, report, PR draft, and **NOT RUN** swap commands. [Handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_swp2_pr140_swap_packet_restage_move43_20260910.md). Nothing published; frontier unchanged.

- **Score recomputed exactly:** 0.1372848557085275.
- **Compliance:** 85/93 PASS versus pk1’s 70/76. Eight refusals remain: five CPU-refusal schema checks, raw-evaluation policy, static import hygiene, and hosted manifest. README/MANIFEST also need refreshing.
- **Verification:** 48 files copied byte-exact; 45/45 evaluated code hashes match. Neither fallback was reached in the bounded bare-venv import smoke.
- **Serializer:** rc **17**, Git-object write denied. SSD fallback bundle verified **51/51 files**; MAIN must land it. Shared index unchanged.

Disclosure before: “I used coding agents (Claude as orchestrator of Codex subagents) extensively as research and engineering tools for the work behind this submission.”

After: “I used automated research and engineering tools extensively for the work behind this submission.”

No receiver edits, scorer runs, GPU dispatches, or new score measurements occurred.

composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer `ddm_swp2_20260910/LANDING_STATUS.json`; trigger verified rc17 bundle: land the metadata.
- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN/operator; consumer `ddm_swp2_20260910/BLOCKERS.json`; trigger handoff harvest: resolve refusals and stale documents.
- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN/operator; consumer `ddm_swp2_20260910/SWAP_COMMANDS.md`; trigger non-hosting clearance and explicit confirm: host/fetchback, require strict PASS, then publish.

## LIVE-HYPOTHESES

- An unchanged-receiver release may clear through policy adjudication: runtime hashes match and normal imports avoid the fallbacks. Acceptance remains untested.

## DEAD-ENDS

- The existing CPU refusal JSON does not satisfy the current checker; five failures prove that.
- Byte identity alone does not establish publication readiness.
- The import smoke does not prove full Linux/T4 execution or universally unreachable fallbacks.