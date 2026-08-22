NO WIN — site-time + Brotli q11 was best at 336,864 B, which is 223,087 B larger than the shipped 113,777 B HPAC/RC64 token stream. The ≥10% prediction is refuted for this DX2 instance.

Key findings:

- Exact token field: `(600, 384, 512)`, 117,964,800 uint8 symbols, alphabet `{0,1,2,3,4}`.
- The incumbent is learned HPAC/RC64 in event-group order, not pair-major raster Brotli.
- All 9 candidates inverted byte-exactly; 27/27 coder round-trips and deterministic repeats passed.
- The class-sorted gross stream’s apparent 218 B result is non-invertible. Counting its video-derived position maps yields 588,674 B.
- Best optimistic projection: 403,455 B archive, `S=0.2967643519`; no archive was built and this is not an exact score.
- No receiver, scorer, Modal, Metal, or upstream mutation occurred. The 42,382 B deficit remains unchanged.

Artifacts:

- [Governed memo](/Users/adpena/Projects/pact/.omx/research/ddm_to2_token_ordering_race_20260822.md)
- [Reproducible producer](/Users/adpena/Projects/pact/experiments/ddm_to2_token_ordering_race.py)
- [RESULT.json](/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1/RESULT.json), sha `a06281b8…`
- Retained 2.3 GiB on Vertigo with 101 verified payload receipts.
- Serializer commit: `f470c1cd71a6d4033285ae02d17d09702f4836b8`

Own-vehicle frontier unchanged: `S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN. Consumer store: `/Volumes/VertigoDataTier/pact/<claimed-rate-representation-lane>/RESULT.json`, then `.omx/state/main_hot_state.md`. Fire trigger: claim a non-duplicate lane and build an exact-invertible boundary/transition grammar counting every video-derived byte; integrate only below 113,777 B, prioritizing ≤71,395 B.

## LIVE-HYPOTHESES

- A boundary/transition grammar may beat flat serialization because only 343,431 of 117,734,400 horizontal neighbour pairs change class. This requires a genuinely new, fully counted representation—not another token permutation.

## DEAD-ENDS

- Flat raster, event-group, site-time, 8×8, Morton, and serpentine orderings with Brotli q11, LZMA1, or zlib9: all lose by at least 223,087 B on this exact field.
- Incumbent event traversal with generic coders: 895,353 B; the traversal alone is not the shipped compression mechanism.
- Class-sorting without counted positions: non-invertible and forbidden. With positions counted, it is 588,674 B.
- TO2 receiver integration and T4 evaluation: folded because no retained candidate beat the incumbent.