# Task #543 PASS-1 seal cure specification (2026-07-18)

`research_only=true` · lane `v10_power_diagram_byteclose_20260718` · MAIN review required

## Trigger

The first whole-scope clean-pass candidate failed on three implementation
classes. This specification owns only their code/test/evidence-container cure;
it grants no receiver, score, pointer, cleanup, or resume authority.

## Required cures

1. Historical source containment: replace the plaintext source snapshot with a
   deterministic gzip container. Its manifest must bind both container
   bytes/SHA-256 and the in-memory decompressed original `62,907 B` /
   `be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9`.
   Validation may decompress only in memory and must never import, execute, or
   materialize the source. Direct `python <container>` invocation must fail
   before argparse/import execution and leave a mutation sentinel unchanged.
2. Post-run immutable custody: the governed frame-195 diagnostic must re-hash
   every verified input after inference, not only checkpoint/cache. Preserve
   inode/size/mtime checks as well.
3. Output confinement: harvester and diagnostic outputs must be `.json` files
   beneath the existing resolved `REPO_ROOT/.omx/research` tree. Refuse source,
   tools, main-checkout, SSD, transient, missing-parent, symlink-escape, and
   overwrite targets. The atomic writer must not create parent directories.
4. Checkpoint-construction API extinction: production read-only evidence code
   must not define or export builders for immutable identities, extraction
   checkpoints, or resume payloads. Historical parser fixtures belong only in
   tests, with an explicit public-surface absence regression.
5. Current execution custody: the prefix receipt must bind the canonical
   executing harvester and read-only evidence helper before measurement,
   full-hash recheck both after use, and record exact Python/platform/NumPy
   runtime custody. Missing, forged, or drifted custody must fail validation.
6. Raw-output symlink rejection: both run entrypoints must reject an explicit
   output path whose final component is a symlink before any resolution or
   work, including broken links targeting inside or outside the research tree.

## Owned files

- `tools/measure_v10_power_diagram_generator_byteclose.py`
- `tools/v10_power_diagram_blocked_evidence.py`
- `tools/harvest_v10_power_diagram_blocked_prefix.py`
- `tools/diagnose_v10_power_diagram_frame195.py`
- their four Task #543 test files
- the historical-source container and its manifest under
  `.omx/research/evidence/`

Do not edit findings, DAG/equation/matrix documents, storage plan, generated
receipts, lane/task/daemon state, or SSD artifacts. Root will move superseded
receipts, regenerate both current receipts, update all dependent hashes/paths,
and force-land the ignored storage plan.

## Acceptance

- Focused suite passes with explicit direct-interpreter, all-input-rehash,
  output-boundary, checkpoint-construction-API absence, and current execution
  source/runtime custody tests, including end-to-end raw broken-symlink cases
  for both entrypoints.
- Ruff, formatting, `py_compile`, and `git diff --check` pass.
- Exact original bytes recover in memory from the deterministic container.
- The live tool remains an unconditional tombstone.
- No real harvester or scorer diagnostic is run by the implementation lane.

## Root closure evidence

- Deterministic gzip: `16,187 B`, SHA-256
  `ee13d263b51f210fe7fd7bbfc6a21099260189573fce80715c0d69df0f2ef329`;
  in-memory decompressed original: `62,907 B`, SHA-256
  `be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9`.
- Focused implementation suite: `75 passed`; Ruff, format, and `py_compile`
  clean.
- Governed sealed prefix harvest: exit `0`, `4.560 s`, peak RSS `1,057 MiB`;
  current receipt SHA-256
  `3c64eb2849ced6dd8eb4492437744fccd03f89977c244bce73cf5a5e30db6e2f`.
- Governed sealed frame-195 diagnostic: exit `0`, `6.249 s`, peak RSS
  `965 MiB`; current receipt SHA-256
  `65d97194c6298a5502d0fcc792ee2fe3bf05599c69f1130d64c270dec5ec36ee`;
  every verified input was re-hashed after inference.
- Pre-seal receipts
  `ae890fb8ef27effd12572c1ec27e0faee008b042ccb165b8580bff946e57d747`
  and `61fd53dc344941b0f4e9fc34a832099006c5bd3685aad66ce164d1bcd95a461d`
  remain preserved under `.omx/research/evidence/` and are explicitly
  superseded/non-authorizing.
- Failed checkpoint-API seal receipts
  `e66b3056a03912cd1e46009204fb1a094c365684cc81030431cde0b7f143e9e5`
  and `c9e3b37ce949e86fa78b3d494bc7b3670b1128a84db78ffadf64d5199633623e`
  also remain preserved under hash-bearing `preseal2` paths and are explicitly
  superseded/non-authorizing.
- Failed current-execution-custody seal receipts
  `0a62dd1102f9b8462cbd12003891b852f4374fcf9c728a7a995e57e85b7031db`
  and `ce037fd1dd6b37ec209d730e10735ac4278c59200388da1fe103d833f88d52a2`
  remain preserved under hash-bearing `preseal3` paths and are explicitly
  superseded/non-authorizing.
- Failed raw-output-symlink seal receipts
  `9b1b05bb65910433ddac0d1ff3e5e3e727f3d969999767544760ffe55d71a8d6`
  and `9ea3eccc38ab08e9d734c16b833d9be9065f3dcb5dcb369b055dcfb7c8bc187e`
  remain preserved under hash-bearing `preseal4` paths and are explicitly
  superseded/non-authorizing.
