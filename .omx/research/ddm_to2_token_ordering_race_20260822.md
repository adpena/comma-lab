# DDM TO2 — DX2 token ordering race (2026-08-22)

**NO WIN — the best tested exact-invertible generic form is site-time + Brotli q11 at 336,864 B, which is 223,087 B (+196.07%) larger than the shipped 113,777 B HPAC/RC64 token stream. The registered ≥10% cut is refuted on this instance; no receiver build or evaluation fire-order is admissible.**

## Verdict

- `verdict_scope: INSTANCE` — exact DX2 archive sha `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; exact decoded token field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`; the nine forms and three generic coders listed below.
- `[macOS-CPU advisory]`, scorer-free: all 9 candidate inverses equal the 117,964,800-byte source array byte-for-byte. All 27 coded payloads decode to their retained raw form, and every primary payload equals its deterministic repeat.
- The prior prediction, site-time + Brotli q11 ≤102,399 B (at least 10% below 113,777 B), is false here. It measured 336,864 B.
- The charter's stronger falsifier, “every generic order within about 2% of incumbent,” did **not** occur. The generic winners are 196.07% to 686.94% larger than incumbent. Therefore this result closes only these tested order/coder combinations. It does **not** prove that every lossless ordering or every new token representation is impossible.
- The premise that the incumbent is plain pair-major raster is also false. The shipped token member is an RC64 arithmetic stream under a learned 19-member HPAC/context law. Decode traverses frame outermost, then groups `g=0..189` defined by `g=(x mod 64)+2*(y mod 64)`, then raster positions within each group. RX1 header `codec=2` selects Brotli for the HPAC model blob; it does not make the trailing token member a Brotli stream.
- No candidate beats the shipped member, so the 42,382 B archive gap is unchanged. No archive was built, no receiver was edited, no scorer ran, and no score claim was produced.

## Inherited custody and reproduced archive anatomy

All required pins matched before measurement:

| Object | Bytes | SHA-256 |
|---|---:|---|
| DX2 `archive.zip` | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| AD2 `measurement_v6/RESULT.json` | 134,747 | `80124acd71ff63d4d9379b87674d1a976e1aa73857b4062a1c9ea2afb1b73511` |
| RB1 memo | 18,498 | `fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09` |
| decoded-token checkpoint receipt | 3,511 | `c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9` |

The checkpoint receipt binds the decoded array to the exact archive, RC64 stream sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5`, CPU decoder contract, pair count, geometry, alphabet, and output sha. The anatomy reproduced directly from ZIP member `p` without disagreement:

| Physical region | Bytes | Denominator | Archive fraction |
|---|---:|---:|---:|
| ZIP framing | 100 | 180,368 B archive | 0.05544% |
| RX1 header | 14 | 180,368 B archive | 0.00776% |
| HPAC model blob | 13,515 | 180,368 B archive | 7.4930% |
| semantic renderer | 30,856 | 180,368 B archive | 17.1072% |
| carrier | 22,010 | 180,368 B archive | 12.2028% |
| compact residual | 96 | 180,368 B archive | 0.05322% |
| RC64 semantic-token stream | 113,777 | 180,368 B archive | 63.0805% |
| **Total** | **180,368** | **180,368 B archive** | **100%** |

The member itself is 180,268 B and is ZIP-STORED; ZIP framing is exactly 100 B.

## Exact token field

| Property | Measured value |
|---|---|
| Shape | `(600 pairs, 384 rows, 512 columns)` |
| Symbol count | 117,964,800 symbols / 117,964,800 expected positions |
| Raw dtype and bytes | `uint8`, 117,964,800 B |
| Alphabet | `{0,1,2,3,4}` / 5 observed classes |
| Exact array SHA-256 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |

| Class | Count | Denominator | Fraction |
|---:|---:|---:|---:|
| 0 | 27,406,888 | 117,964,800 | 23.233107% |
| 1 | 691,095 | 117,964,800 | 0.585848% |
| 2 | 58,413,222 | 117,964,800 | 49.517502% |
| 3 | 1,460,458 | 117,964,800 | 1.238046% |
| 4 | 29,993,137 | 117,964,800 | 25.425497% |

The prediction was sealed before any candidate compression. The structural profile already cut against its mechanism:

| Equality surface | Equal neighbours | Comparisons | Fraction |
|---|---:|---:|---:|
| horizontal | 117,390,969 | 117,734,400 | 99.708300% |
| vertical | 116,375,407 | 117,657,600 | 98.910234% |
| temporal | 116,297,673 | 117,768,192 | 98.751345% |

Temporal equality is high, but it is lower than horizontal equality. Site-time still wins inside the generic family, so temporal adjacency is useful to Brotli; it is nowhere near enough to replace the learned HPAC/RC64 representation.

## Real coder race

All byte counts below are retained payload lengths, not entropy estimates. `Δ vs shipped RC64` uses the 113,777 B token member as denominator. The “actual shipped” row is the authority baseline, not a generic recode. Every candidate row has a retained 117,964,800-byte inverse whose sha equals the source sha.

| Form | Raw B | Brotli q11 B | LZMA1 1 MiB B | zlib9 B | Best | Δ vs shipped RC64 | Exact inverse |
|---|---:|---:|---:|---:|---:|---:|---|
| **actual shipped HPAC/RC64** | — | — | — | — | **113,777** | baseline | checkpoint-bound |
| frame-raster | 117,964,800 | **429,894** | 490,704 | 585,896 | 429,894 | +316,117 (+277.84%) | PASS |
| incumbent event traversal + generic coders | 117,964,800 | **895,353** | 990,277 | 1,462,703 | 895,353 | +781,576 (+686.94%) | PASS |
| **site-time** | 117,964,800 | **336,864** | 351,788 | 751,562 | **336,864** | **+223,087 (+196.07%)** | PASS |
| 8×8 block, frame-inner | 117,964,800 | **442,311** | 453,462 | 718,635 | 442,311 | +328,534 (+288.75%) | PASS |
| 8×8 block, time-inner | 117,964,800 | **366,167** | 373,805 | 730,993 | 366,167 | +252,390 (+221.83%) | PASS |
| Morton, frame-inner | 117,964,800 | **524,346** | 597,323 | 973,901 | 524,346 | +410,569 (+360.85%) | PASS |
| Morton, time-inner | 117,964,800 | **416,291** | 420,668 | 758,322 | 416,291 | +302,514 (+265.88%) | PASS |
| serpentine, time-inner | 117,964,800 | **367,322** | 388,284 | 748,521 | 367,322 | +253,545 (+222.84%) | PASS |
| class-sorted + counted position maps | 176,947,252 | **588,674** | 605,132 | 1,016,941 | 588,674 | +474,897 (+417.39%) | PASS |

The generic incumbent event traversal is especially poor because its consecutive-symbol equality is only 94.5591%, versus 99.6935% for raster and 99.7066% for serpentine. HPAC/RC64 wins there because its coding probabilities and corrector state match the traversal; the traversal by itself is not the compression mechanism.

The class-sorted gross symbols compress to 218 B with Brotli q11, but that payload cannot reconstruct positions and is **not admitted**. Its fully counted packet includes the 117,964,800 sorted symbols, a 52 B header/default-class choice, and four content-derived position bitmaps totalling 58,982,400 B. The exact-invertible net result is 588,674 B. Reporting 218 B would hide 588,456 B of required coded addressing content and would be the local rule-118 fake.

## Rule-118 adjudication

| Candidate | Receiver rule | Receiver reads | Counted side information |
|---|---|---|---:|
| frame-raster | reshape in `(pair,y,x)` order | public shape only | 0 B |
| incumbent event traversal | for each pair, fixed 64×64 patch group formula, then raster positions | public shape, fixed patch 64 and delta 2 | 0 B |
| site-time | transpose fixed sites outside, pair inside | public shape only | 0 B |
| 8×8 frame/time | fixed 8×8 tile traversal with declared pair nesting | public shape and generic constant 8 | 0 B |
| Morton frame/time | integer Morton key from `(x,y)` and declared pair nesting | public shape only | 0 B |
| serpentine time | reverse odd rows, then put pair inside each fixed site | public shape and row parity | 0 B |
| class-sorted | read the packet header, sorted symbols, and explicit class-position bitmaps; restore each class to marked sites | public shape **plus the counted packet** | all 176,947,252 raw B / 588,674 coded B |

The generic algorithms may live in free receiver code, but no video-derived table was treated as free. A receiver specialized to any one generic form needs no archive selector; the projections below optimistically assume that specialization and add zero selector bytes. That is harmless here because every candidate already loses by at least 223,087 B.

## Archive and score projections

These rows are **PROJECTION, not measurement**. No archive was rebuilt. They replace 113,777 B in the 180,368 B DX2 archive with the candidate's best retained stream and hold measured distortion fixed at `0.0281202279752971`:

`S_projected = 0.0281202279752971 + 25·projected_archive_bytes / 37,545,489`.

| Candidate | Projected archive B | Δ archive B | Projected S | Below 137,986 B? |
|---|---:|---:|---:|---|
| frame-raster | 496,485 | +316,117 | 0.3587092103 | NO |
| incumbent event + generic coder | 961,944 | +781,576 | 0.6686392528 | NO |
| **site-time** | **403,455** | **+223,087** | **0.2967643519** | **NO** |
| 8×8 frame | 508,902 | +328,534 | 0.3669771809 | NO |
| 8×8 time | 432,758 | +252,390 | 0.3162760168 | NO |
| Morton frame | 590,937 | +410,569 | 0.4216009202 | NO |
| Morton time | 482,882 | +302,514 | 0.3496515310 | NO |
| serpentine time | 433,913 | +253,545 | 0.3170450839 | NO |
| class-sorted, positions counted | 655,265 | +474,897 | 0.4644342949 | NO |

DX2 needs to shed 42,382 B to reach the strict 137,986 B archive ceiling at fixed distortion. Holding every other byte fixed means the token member would need to fall from 113,777 B to at most 71,395 B, a 37.2501% cut. The best tested form is 265,469 B above that token ceiling and makes the score worse, not better.

## Fire-order and follow-on dispositions

- **FOLDED — TO2 receiver integration / byte-close / seal / T4.** Owner: MAIN. Consumer store: `.omx/state/main_hot_state.md`. Fire trigger was a retained, exact-invertible candidate materially below 113,777 B; it was not met. No receiver or scorer action should be created from this race.
- **QUEUED-WITH-A-FIRE-ORDER — a new exact boundary/transition grammar, not another flat ordering.** Owner: MAIN must first claim a non-duplicate lane. Consumer store: `/Volumes/VertigoDataTier/pact/<claimed-rate-representation-lane>/RESULT.json`, then MAIN's hot-state row. Fire trigger: a pre-registered, shape-generic decoder whose complete video-derived payload is counted and whose retained inverse equals the DX2 token sha; integrate only if the byte-closed token payload is below 113,777 B, and prioritize it for the sub-0.12 path only if it reaches ≤71,395 B. Plausibility comes from only 343,431 horizontal transitions among 117,734,400 horizontal comparisons, but this is a new representation hypothesis, not a TO2 win.

## RECALL EVIDENCE

I searched the full local corpus, not only the charter seeds:

- `.omx/research/` content queries: `token ordering`, `serialization`, `tile-major`, `time-major`, `Morton`, `Hilbert`, `serpentine`, `RC64`, `HPAC`, `context order`, and `explicit position`.
- canonical equation registry: `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for rate/entropy/placement/order laws.
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, specs/design docs, task ledgers, active claims, and `main_hot_state.md`, with the same rate/order/context queries plus `ddm_to2`.
- exact shipped receiver sources and decoder checkpoint receipt under the pinned DX2 custody tree.

Findings beyond the charter seeds and how they changed the measurement:

- `ddm_gd1_undecided_defaults_audit_20260731.md` measured a different SMEVR object where Hilbert lost 452 B, serpentine beat raster by 37 B, and a matched split attributed 425 B to context versus 27 B to order. This added the serpentine control and forced separate “traversal” from “context model” conclusions.
- `ddm_ba31_negative_surfaces_20260731.md` reinforced denominator-complete and bounded-negative reporting. This is why the verdict is INSTANCE-scoped and why the untriggered ±2% falsifier is stated explicitly.
- older PR91 tile/phase attempts failed when probability/context order and decode order did not match. That led to inspection of the actual DX2 receiver and discovery that the incumbent is event-group RC64, not the charter's assumed pair-major raster Brotli stream.
- no live `ddm_to2` claim or prior same-object DX2 token-order receipt was found in the searched task/claim surfaces. This charter remained the lane authority; no shared hot-state file was edited.

## Custody and verification

- SSD tier used: `/Volumes/VertigoDataTier/pact` (never APDataStore for new payloads).
- Receipt root: `/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1/`.
- `RESULT.json`: 50,010 B, sha `a06281b8331c92d7dd892034fd7ea92fa4a997d298252d8f958d8083d11cdac5`.
- `MANIFEST.json`: 26,630 B, sha `e29a37e27a10d526d4c4a305dc1bdee179ed5562f7b05eb10a07fc740441fd6c`.
- Retained tree: 2.3 GiB. It contains the extracted source array, archive/member/token stream, decoder receipt, every raw candidate, all 27 coder payloads and repeats, every exact inverse, class-sort gross diagnostic, source snapshots, profile, prior, stage state, result, and manifest.
- Independent post-run verification rehashed all 101 manifest payload receipts and decompressed all 27 candidate/coder payloads; every decompressed sha matched its raw candidate and every inverse sha matched the source.
- No scorer, Modal, Metal, receiver mutation, upstream mutation, or archive build occurred.

Own-vehicle frontier **UNCHANGED**: `S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600]`, DX2 archive sha `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.
