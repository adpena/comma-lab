The chartered repair is complete and already landed as `f1f0abd27`; follow-up `05307d426` replaced one quarantine with an append-only bounded-target receipt. I did not create a duplicate commit.

The [final memo](/Users/adpena/Projects/pact/.omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md) records the complete per-module adjudication:

- EP725 adapter, G17 actuator, and monolithic PGA: pin refreshed after n1/n2 bit-exact decode verification.
- Bounded-target G: now passes against a new receipt while retaining the July receipt unchanged.
- Selected-preimage v1/v2, G17/G49, G72, and G82: 19 provenance-backed `xfail(strict=True)` tests for the retired V15 custody chain.
- Inverse stack: one strict xfail for the V9/PBR2 packet-manifest chain.
- Real regressions among the original 51: zero.
- The seven recovered canonical-equation IDs each still occur exactly once.

The serializer record confirms all 12 files were committed with post-edit SHA-256 guards and commit rc=0. Protected trees and gc1/gf2/xr1 files remain untouched; the index is empty.

A fresh current-HEAD run measured **1,340 passed / 20 xfailed / 6 failed / 17 errors in 1,241.68 s**. The [retained log](/Volumes/VertigoDataTier/pact/ddm_ql1_reverification_20260909/witness_dsl_full_suite_20260909.log) is 56,946 bytes, SHA-256 `209297fa803ef54df244d64062615ec94aea9f53fce48849868eb29376b6812d`.

The new red outcomes are not retired-lineage regressions:

- Four failures stop at MLX device creation because this sandbox exposes no Metal device; the charter forbids Metal work.
- Two failures and 17 fixture errors correctly refuse the September 6 official-leaderboard snapshot as older than 24 hours. The canonical refresh was attempted, but recorded `upstream_fetch_status=network_failure`; cached freshness was not fabricated.

No scorer, Modal job, successful Metal computation, archive mutation, or exact evaluation was performed.

Own-vehicle frontier unchanged: **S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN frontier custodian; consumer store: `.omx/state/canonical_frontier_pointer.json`; fire trigger: the canonical refresher can successfully fetch the official leaderboard. Refresh from the official source, then rerun the inverse-stack module.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: witness-DSL MLX test maintainer; consumer store: `/Volumes/VertigoDataTier/pact/ddm_ql1_reverification_20260909/`; fire trigger: a permitted Metal-capable host is available. Rerun the four MLX tests, then the full suite.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: retired V9/PBR2 maintainer; consumer store: `.omx/research/original_taskspace_inverse_witness_codec_20260725/`; fire trigger: the PBR2 packet and complete custody chain can be resealed against the current 13-source manifest. Delete the one strict xfail only after deterministic equality holds.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: retired V15 maintainer; consumer store: `.omx/research/original_taskspace_inverse_witness_codec_20260725/fresh_v15_semantic_base_n600_20260726/`; fire trigger: scorer authority is available. Produce a fresh compile receipt and remove the 19 strict xfails only if byte identity is proven.

## LIVE-HYPOTHESES

- A fresh official snapshot plus a permitted Metal host will restore 0 failed / 0 errors: every current red traceback terminates at one of those two boundaries.
- The V15 fresh compile will remain output-equivalent because the sealed archive still decodes identically and retains identical mutation coverage.
- The V9/PBR2 reseal will preserve semantic output because current sources already reproduce the sealed full-field semantic digest bit-for-bit.

## DEAD-ENDS

- Broadening the retired-lineage quarantine to cover today’s failures is closed: they are explicit environment and freshness refusals.
- Manually advancing the leaderboard timestamp is closed: the official fetch failed, so claiming freshness would be false.
- Rewriting sealed July receipts in place is closed: supersession must remain append-only.
- Repeating the full suite unchanged in this sandbox is closed until the snapshot or Metal boundary changes.