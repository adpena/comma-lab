Implemented and verified on move 43. **Strict compliance improved from 85/93 to 91/93**, still rc 1.

| Checks | Before | After |
|---|---:|---:|
| Five CPU-refusal checks | 0/5 | 5/5 |
| Raw-policy adjudication | 0/1 | 1/1 |
| Import hygiene + hosted manifest | 0/2 | 0/2 |

**160 tests passed**, Ruff clean, two review passes per Python file. [Full memo](/Users/adpena/Projects/pact/.omx/research/ddm_cpx2_checker_cuda_only_refusal_adjudication_path_20260910.md).

The empty outer runtime digest was bound through retained provenance and live-file recomputation. No CPU metrics were created. Packet, receiver, volumes, upstream, raw receipts, and shared index remain unchanged; no publish, Modal dispatch, or scorer run occurred.

Serializer **rc 17**: sandbox denied Git object writes. Verified fallback commit `67f4eaa0` contains all 30 substantive files; **MAIN landing remains pending**. [Bundle verification and status](/Users/adpena/Projects/pact/.omx/research/ddm_cpx2_20260910/SERIALIZER_STATUS.json).

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer `.omx/research/ddm_cpx2_20260910/SERIALIZER_STATUS.json`; trigger: harvest the verified bundle and land it.
- **FOLDED** — owner operator + MAIN; consumer `.omx/research/ddm_swp2_20260910/BLOCKERS.json`; trigger: operator selects receiver correction; resolve imports and obtain fresh exact evidence.
- **FOLDED** — owner operator + MAIN; same consumer store; trigger: non-hosting clearance and explicit publication confirmation; produce the hosted manifest and obtain strict PASS.

composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43) unchanged.

## LIVE-HYPOTHESES

- Removing the fallback imports may preserve CUDA behavior because the retained smoke did not reach them. Untested; fresh exact evidence remains required.

## DEAD-ENDS

- Move-42 refusal reuse: archive/runtime bindings differ.
- Invented CPU metrics or `contest_cpu` selection: rejected by the typed path.
- Raw-flag edits or fabricated policy approvals: cannot replace current packet checks.
- Treating 91/93 as release clearance: two real blockers remain.