Implemented the charter’s bounded hygiene pass and wrote the full receipt-backed [handoff report](/Users/adpena/Projects/pact/.omx/research/ddm_sw1_portable_paths_secrets_scrub_20260820.md).

Key outcomes:

- Added the canonical portable resolver, migrated selected high-value callers, and removed 64 legacy path literals across 11 mapped files.
- Added the warn-only staged-file guard and executed a successful plant/fire/remove control.
- Fresh census: 43,967 tracked files; 8,408 matching files; classes a=921, b=4,953, c=2,534.
- Preserved SHA-pinned `checkpoint_retention.py` byte-identically.
- Found 16 real credential-shaped historical findings in one commit across three deleted console logs. Rotation/revocation is P0; no secret values were reproduced and history was not rewritten.
- Exact configured fleet IP matches are zero after scrubbing one resolver IP. Four short hostname aliases still have 1,229 unresolved matches.
- Verification: 198 focused tests passed; final core run passed 25/25. Broad run: 598 passed, 6 xfailed, 4 failed—three from unavailable Metal and one unrelated existing SNeRV-contract blocker.
- Final 16-file path guard passed. Final gitleaks scan contains only four adjudicated generic false positives.

The work is uncommitted. The main serializer was blocked by managed Git-object permissions; a clean fallback reached the hook but was refused by three Metal-only tests. No override was used. The implementation patch is retained at [ddm_sw1_portable_paths_secrets_scrub.patch](/Volumes/APDataStore/pact/ddm_sw1_20260820/patches/ddm_sw1_portable_paths_secrets_scrub.patch), SHA `e8c9b955…3c221`. About 4.3 GB of rebuildable temporary clone/scan scratch was removed after durable artifacts were retained.

## NEXT_IF_RESUMED

- **BLOCKED-P0** — owner: credential owner/operator; consumer store: `.omx/state/operator_p0_ledger.jsonl`; fire trigger: immediately; rotate or revoke the historical GCP/JWT credentials and record revocation evidence without storing secret values.
- **BLOCKED-LANDING** — owner: Metal-capable MAIN committer; consumer store: `main` through the commit serializer; fire trigger: Git object writes are permitted and the three CI-blind MLX tests pass; apply the retained patch, include the report, refresh post-edit hashes, and serialize.
- **QUEUED** — owner: security-hygiene successor; consumer store: `$PACT_TIER2/ddm_sw1_20260820/receipts/fleet_alias_adjudication.json`; fire trigger: after this patch lands; semantically classify the 1,229 short-hostname matches.
- **QUEUED** — owner: pin/custody successor; consumer store: `$PACT_TIER2/ddm_sw1_20260820/receipts/checkpoint_retention_supersession.json`; fire trigger: a superseding seal can replace the named SHA consumer; migrate the pinned checkpoint writer without mutating its current bytes.
- **QUEUED** — owner: portable-path successor; consumer store: a refreshed census and mapping receipt under `$PACT_TIER2/ddm_sw1_*`; fire trigger: the current patch is on `main`; continue the sensitivity-ranked class-a/class-b sweep while excluding every class-c row.

Own-vehicle frontier: **rc2_composed — S 0.14827847122030852 @ 180,456 B [contest-CUDA T4, n600]**. This hygiene arm measured no score row and left the pointer unchanged.

## LIVE-HYPOTHESES

- Most hostname matches are short-word collisions; zero exact fleet-IP matches support that interpretation.
- A second ranked sweep can remove substantially more portability debt without disturbing SHA pins.
- The prepared landing should pass unchanged on a Metal-visible session because both focused runs were green and the fallback hook failures were device-specific.

## DEAD-ENDS

- A global mechanical rewrite is closed: 2,534 matching files are protected or SHA-pinned.
- Editing `checkpoint_retention.py` in place is closed until its consumer is superseded.
- An allowlisted documentation key is not a valid gitleaks control; the live-rule control replaces it.
- History rewriting is closed; rotation/revocation is the credential cure.
- Landing from this managed session is closed without both writable Git objects and a Metal-visible hook environment.