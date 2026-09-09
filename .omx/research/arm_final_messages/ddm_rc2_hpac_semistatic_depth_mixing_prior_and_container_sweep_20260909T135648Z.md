Implemented, measured, sealed, and committed as `fb39f2139`.

- Candidate: 181,414 B, SHA-256 `c810c2c7…`, saving 231 B.
- Projected unchanged-distortion score: `0.13885056455024844`; this is arithmetic, not yet a contest score.
- Winner: counted 8-byte shared logistic mixer. The 16-byte mixer won raw J by 5 B, but lost the archive objective by 1 B.
- Exact identity: restored 17,770 B IHS1 body plus unchanged semantic, carrier, and token-tail sections.
- Validation: twin archives identical, all 16 designs decoded exactly, all 36 container cells retained, public-path smoke passed, seal is `SEAL_VALID`.
- Verification: 59 focused tests passed; payload-retention and Catalog #344 checks found zero violations; two canonical equation anchors were recorded.

Artifacts: [memo](/Users/adpena/Projects/pact/.omx/research/ddm_rc2_hpac_semistatic_depth_mixing_prior_and_container_sweep_20260908.md), [RESULT.json](/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/RESULT.json), [validated seal](/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/SEAL_ddm_rc2_hpac_semistatic_mixing_contest_cuda.json).

No scorer, Modal, MPS/Metal, capacity change, permutation experiment, or live-tree mutation was performed.

Current frontier remains:

`sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]`

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN; consumer store: the validated contest-CUDA seal above; fire trigger: claim a unique T4 lane and confirm the pointer still names sj1 SHA `06c44dc4…` at 181,645 B.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: successor byte-coding arm; consumer store: the memo and `model_section_adaptive_recode_ceiling_v1`; fire trigger: only after the sealed RC2 row passes T4, and only with a closed-form counted-parameter design predicting at least another 150 B.

## LIVE-HYPOTHESES

- The sealed T4 replay should preserve sj1’s distortion components and land near `0.13885056`, because every decoded field-bearing section is exact and only 231 archive bytes changed.
- A richer but still globally shared low-parameter mixer may harvest more of the remaining approximately 953 B between RC2’s raw J and the H1 Miller–Madow bound. This is plausible because 8–16 counted weights already converted 216–221 B where per-cell tables failed.

## DEAD-ENDS

- Semi-static previous-value tables: closed at FORMULATION scope; the best counted J was 10,853 B, losing 1,210 B to RC1.
- Semi-static plus logistic combination: not opened because the charter’s “both help” precondition was false.
- `ck2=false` staging: closed on this format because RX1 has no free unambiguous reserved bit.
- Container retuning alone: closed here; q11 with lgwin 22/23/24 tied exactly, yielding 0 B.
- Two-thread public probe: closed because the receiver explicitly requires four threads; the valid probe used one process with four threads.
- In-process implementation dispatch in this sandbox: closed after two `Operation not permitted` failures; the charter’s local escape hatch was used.