# Source custody

Use the canonical serializer with post-edit SHA-256 for every requested file,
`--no-co-author`, and both `[no-triality] [p0-ledger-ok]` tags. Do not stage
the shared lane, claim, checkpoint, review, or task-ledger files wholesale.
Owned receipts export the relevant facts and task rows.

The unchanged archive is ignored by the existing `*.zip` rule. It is retained
with its SHA-256 in the mandatory source delivery tar and six-file workspace;
it is not force-added through the shared index. A serializer rc 17 or 19
means MAIN must land the source delivery, not that the work was committed.

The delivery tar uses explicit owned paths, includes the unchanged archive,
and excludes compiled libraries, bytecode, AppleDouble metadata and proof bulk.
Its members are re-read and hashed against the original source bytes. All
bulk proof artifacts remain in the SSD consumer store with a retention census.
