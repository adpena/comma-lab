Implemented and landed as commit `31e716f64a`.

- Class A — **CURED-STRUCTURAL:** 2/2 certify-and-move implementations and 4/4 live consumers now probe metadata representability, tar-wrap incompatible destinations, and verify content SHA, modes, and symlink targets before deletion. The original ExFAT hypothesis was refuted: the uv environment moved to local APFS; the mover itself dropped metadata.
- Class B — **CURED-STRUCTURAL:** the existing shared lint now refuses SHA prefix-match/divergent-tail errors in both keeper charters and serialized research memos. The actual sfp1 typo fired; canonical and unrelated hashes passed.
- Class C — **CURED-STRUCTURAL:** keeper-scaffolded charters now require `nohup` + `disown`, pidfile, checkpoints, monitoring, and a durable done receipt for compute exceeding 30 minutes. Coverage is honestly scaffold-only.

Verification: 131/131 tests passed after commit, Ruff/compile checks passed, and all 10 edited Python files received two review passes. Controls and full provenance are in [the SF2 receipt](/Users/adpena/Projects/pact/.omx/research/ddm_sf2_structural_fix_wave_20260901.md). Retained filesystem evidence is in [RESULT.json](/Volumes/APDataStore/pact/ddm_sf2/controls/RESULT.json).

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN storage apparatus; consumer store: the next certify-and-move ledger; fire trigger: the next source-retiring cold move; confirm it records the selected representation, destination capability probe, and successful content/metadata verification.
- **CONDITIONAL** — owner: storage apparatus maintainer; consumer store: the mover’s metadata manifest and receipt; fire trigger: a candidate containing ACLs, extended attributes, or hardlink topology; extend fidelity coverage before permitting retirement.

Own-vehicle frontier: **S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]**, afr1 archive SHA `cbb8d928…d405bf25` — **UNMOVED**.

## LIVE-HYPOTHESES

- ACLs, extended attributes, or hardlink topology may form the next metadata-loss class because the current verified contract intentionally covers content, basic POSIX modes, and symlink targets.
- The first full post-cure production move should choose tar on mode-incompatible APDataStore and direct-tree on compatible APFS; both branches are strongly supported by controls, but no new production payload has crossed the complete CLI since landing.

## DEAD-ENDS

- APDataStore ExFAT as the cause of the uv-environment loss: closed because that move targeted local APFS and the mover itself omitted links and modes.
- Filesystem-name checks: closed because the mounted ExFAT implementation supports symlinks but collapses requested modes; measured capability probes are required.
- Content-only SHA equality as deletion authority: closed because it cannot detect mode or symlink loss.
- A parallel SHA scanner: closed; both consumers use the existing shared premise-lint implementation.
- Editing `tac.subagent_contract.standard_contract`: closed because source tracing showed it does not feed keeper-generated Codex charters.