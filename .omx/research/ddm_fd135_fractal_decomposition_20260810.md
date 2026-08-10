# ddm_fd135 — PR135 recursive-fractal decomposition and unidentified-edge crosswalk

**Date:** 2026-08-10
**Scope:** one custodied PR135/F26 archive, its complete 231-file ExperimentBook, the custodied PR130 comparator, and bounded Pact-corpus/online recall
**Archive pin:** `186,724 B`, SHA-256 `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`
**Comparator pin:** PR130 `191,052 B`, SHA-256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`
**Measurement axis:** `[custodied exact bytes; scorer-free decomposition]` unless a row says otherwise

## Result first

PR135 is a derivative PR130 + PR133 vehicle, not a new codec family. Its exact `−4,328 B` archive reduction is a sequence of separately byte-closed changes:

| Sequential state | Archive bytes | Delta from preceding state | Mechanism |
|---|---:|---:|---|
| PR130 | 191,052 | — | range tokens + XZ models + CPR1/IHS1 |
| A2 | 191,000 | −52 | compact fixed residual (`RCF1`) |
| F3 | 190,664 | −336 | `IHS2`/Gate-A/fixed schema |
| F4 | 190,660 | −4 | exact carrier repack |
| F14 | 189,661 | −999 | WANS1 + CAP1 + container stack |
| F16 | 187,539 | −2,122 | RC64 token coder |
| F17 / PR135 final size | 186,724 | −815 | CBQ basis changes on atoms 2/5/9 |

These deltas telescope exactly to `191,052 − 186,724 = 4,328 B`. They are not independent Shapley credits: the jointly compressed model prefix makes the individual outer-LZ effects non-separable. On one common parent, WANS1 measured `−610 B`, CAP1 `−81 B`, and their composition `−691 B`.

The competitive mechanism is mostly joint learned-state movement, not packing. Relative to PR130, PR135 changes 5,989/27,648 basis symbols, confined exactly to atoms 2, 5, and 9; changes 7,044/7,200 int12 coefficients across all 600 rows; changes every coefficient scale and the three selected basis scales; changes two semantic W4 codes; and applies five sparse frame-0 selector events. The ExperimentBook also disproves the charter prior that segmentation was untouched: F26 improved `d_seg` over F24 by `1.017e-7` `[author ExperimentBook exact-row comparison]` through two FiLM codes plus jointly compensated carrier coordinates. What remains open is an implicit, rate-aware joint seg/pose search—not a generic post-hoc seg overlay.

The highest-value cheap falsifiers are:

1. same-state lc2 ANS on F26's exact symbols/probabilities (`178 B`, projected `−0.000118523 S` if the whole-archive advantage survives);
2. a current-base joint int12/basis/FiLM solve using our additional starts and pose engines (unpriced, but it attacks the 95.1%-pose gap to our lc2 row);
3. exact CAP1 metadata packing (at most 40 raw bytes before outer LZ, projected ceiling `−0.000026634 S`);
4. scorer-aware per-cell mixed precision and implicit edge conditioning, both unmeasured on this exact vehicle.

No scorer, evaluator, remote dispatch, or new candidate archive ran in this arm. The exact PR135 replay landed independently while this analysis was active: `S = 0.16226942370411543 @ 186,724 B`, `d_seg = 0.00029643`, `d_pose = 0.00000688` `[contest-CUDA, Modal Tesla T4, n600, custodied replay]`. Its evidence receipts are SHA-256 `78354987ab18311bcfadad55b227753891ba59b40ec656a0133ca303c27ab665` and `6f4bc66ec1b29442556e2000bed45eddcc84bf0a82fde0b71e5183cfdf999ebb`. The independent CPU attempt is a refusal, not a score: PR135's own `runtime/f26_inflate.py:105` requires a CUDA-capable GPU.

## Custody and retained payloads

Every materialized byte stream was retained under:

`/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/`

The final evidence manifest covers 77 retained files totaling 2,246,556 bytes and excludes itself to avoid recursive hashing:

| Receipt | Bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `EVIDENCE_MANIFEST.json` | 13,405 | `02d14341ded889b1331bddb029b070ca8a191e1c7a1e624253bfc226d3fcf002` | path/size/hash census |
| `MEASUREMENT.json` | retained | `14f485530f21d1c90bb75407a4fe05292fb3a1afdeec5e32498b742568d31d17` | exact archive map and PR130 diff |
| `FOLLOWUP_MEASUREMENT.json` | 9,732 | `2ac50c1775cc77cd266c47c4ce9821dab6c47f6405eb2027ea5404d93ad7918e` | atom/dimension entropy and displacement |
| `HPAC_MEASUREMENT.json` | 2,834 | `46b0f0fa79449a93384dcb30f322897dcc15b7f1034b404abf5051ef6602f877` | strict IHS2 field map |
| `EXPERIMENT_BOOK_INVENTORY.json` | 38,444 | `bc77b5d05a65c50993b59532f9f25d656bc0c9f6e1cb7b325a24bc4d12826450` | complete 231-file read census |
| `EXPERIMENT_BOOK_TESTS.junit.xml` | retained | `827056137bfa96841e781c878452a4379d0f617854d95f6f831deb8971a0b01c` | pinned local test receipt |
| `ONLINE_RESEARCH_RECEIPT.json` | 1,630 | `f5c0edfc225df8a9f4522d00d0e3def509b6db4f26cbbbea88f4f196f2dae31f` | bounded Aug-06→Aug-10 search receipt |

The retained tree includes both complete archives, both exact member `p` payloads, every recursively extracted section, canonical restored streams, parsed WANS streams, CAP1 fields, decoded arrays, and PR130↔PR135 displacement arrays. No payload was discarded after measurement.

## Fractal section map

### L0 — ZIP and routing

| Span | Offset | Bytes | Meaning |
|---|---:|---:|---|
| local ZIP header | 0 | 31 | one stored member named `p` |
| member `p` | 31 | 186,624 | exact payload SHA-256 `66da2921780038cad6b18d25ea36e066c63b07b755fec33c15ac013ff4bcfc3c` |
| central directory | 186,655 | 47 | ZIP metadata |
| EOCD | 186,702 | 22 | archive terminator |

ZIP overhead is exactly 100 bytes. Contrary to the charter's provisional L0 wording, F26 has no PR130-style 4-byte `model_word`. Its receiver discovers the end of the raw-LZMA2 prefix, restores `F24S`, and routes the fixed schema to `IHS2`, WANS1, CAP1, selector, compact residual, and RC64.

### L1 — physical member `p`

| Section | Offset | Bytes | Fraction of `p` |
|---|---:|---:|---:|
| raw-LZMA2 compressed models | 0 | 71,822 | 38.485% |
| compact residual body | 71,822 | 96 | 0.051% |
| RC64 token stream | 71,918 | 114,706 | 61.464% |
| **total** | 0 | **186,624** | **100%** |

The token stream remains the dominant stored section. ExperimentBook's RC64 probability-model ideal is `114,705.460054 B`, only `0.539946 B` below its realized `114,706 B`; this closes generic token-coder overhead on the exact RC64 model state. A byte-alphabet H0 computed on the already-compressed RC64 stream is `114,679.245 B`; that is a description of output-byte frequencies, not a stronger source-symbol coding bound.

### L2 — decompressed fixed model prefix

The raw-LZMA2 prefix expands from 71,822 to 74,860 bytes:

| Section | Raw offset | Physical bytes | Canonical bytes | Fixed bytes elided by schema |
|---|---:|---:|---:|---:|
| `F24S` magic | 0 | 4 | 4 | 0 |
| IHS2 body | 4 | 16,593 | 16,599 | 6-byte IHS2 header/prefix |
| WANS F12 body | 16,597 | 36,040 | 36,051 | 11-byte WANS prefix |
| CAP1 body | 52,637 | 22,214 | 22,222 | 8 fixed bytes |
| frame-0 selector body | 74,851 | 9 | 14 | 5-byte selector prefix |
| **raw model total** | 0 | **74,860** | — | — |

The WANS body contains a 30-byte offset table, 8,284 bytes of learned/fixed metadata, and 27,726 bytes of stream bodies. Its 16 stream lengths are:

`239, 1915, 3908, 432, 4050, 737, 432, 4030, 740, 432, 3933, 754, 431, 3958, 751, 984 B`.

Thirteen are ANS-coded and three are raw. Restoring the PR130-layout semantic state yields 40,252 bytes with memoryless byte H0 `36,805.347 B`; the physical WANS body is 36,040 bytes, 765 bytes below that memoryless bound because it uses context/adaptive structure. This is a real source-state comparison; no entropy claim is inferred from compressed-byte H0.

### L2.1 — IHS2 exact decomposition

Strict runtime parsing and exact IHS2→IHS1 reconstruction close the 16,599-byte canonical HPAC state:

| IHS2 v3 field | Bytes |
|---|---:|
| magic/version/flags (`IHS2`, v3, `0x31`) | 6 |
| original row depths, 517 nibbles | 259 |
| tightened row depths, 517 nibbles | 259 |
| signed weight rows, 102,834 bits | 12,855 |
| frame embedding, E0L0 int4 | 2,400 |
| nine bias widths | 5 |
| bias values, 2,383 bits | 298 |
| raw int8 exponents, 517 values | 517 |
| **total** | **16,599** |

The corrected flag interpretation matters: `0x31` enables E0L0, tightened rows, and packed biases, but **not** 3-bit exponents. The reconstructed canonical IHS1 is exactly 20,179 bytes. On the same learned state, IHS1 source-byte H0 is `16,566.648 B`; IHS2 is `26.352 B` above that elementary H0 while remaining 3,586 bytes smaller than raw IHS1. Outer LZ effects are not assigned to individual IHS2 fields.

### L3 — CAP1 carrier and selector

| CAP1 field | Bytes/bits | Detail |
|---|---:|---|
| canonical header | 14 B | magic + structural constants |
| predictor metadata | 36 B | 12 q8 factors + 12 signed biases |
| scales | 96 B | basis and coefficient scales |
| Huffman lengths | 32 B | basis code lengths |
| Rice `k` values | 12 B | dimensions use only 8 or 9 |
| basis codes | 98,213 bits = 12,277 stored B | 27,648 symbols |
| AR(1) residuals | 78,036 bits = 9,755 stored B | 7,200 coefficients |
| **canonical CAP1** | **22,222 B** | exact parse |

The canonical carrier with sparse selector is 22,242 bytes (`F0C1`); the PR130-shape restored carrier is 22,307 bytes. The selector has eight catalog modes but only five non-identity events: frames `60, 85, 116, 241, 373` with labels `4, 3, 4, 7, 4`.

The basis-code H0 is `12,156.649 B`; canonical Huffman consumption is 98,213 bits (`12,276.625` fractional bytes, 12,277 stored), a `119.976 B` coding gap before final-byte padding. The AR-residual H0 is `9,448.032 B`; Rice consumption is 78,036 bits (`9,754.5` fractional bytes, 9,755 stored), a `306.468 B` coding gap before final-byte padding. The signed coefficient values themselves have H0 `9,802.275 B`, showing why the AR transform is the correct comparison surface.

Atoms 2, 5, and 9 are both the only changed basis atoms and the loosest per-atom Huffman cells:

| Atom | Changed symbols | L1 displacement | max `|Δ|` | H0 bytes | Huffman bytes |
|---:|---:|---:|---:|---:|---:|
| 2 | 2,020/2,304 | 3,944 | 8 | 804.648 | 854.000 |
| 5 | 2,005/2,304 | 3,866 | 8 | 795.257 | 849.875 |
| 9 | 1,964/2,304 | 3,364 | 8 | 717.446 | 801.750 |

This is not proof those bytes can be removed without distortion: those atoms are exactly where PR135 bought its pose response.

### L3.1 — realized PR130→PR135 learned displacement

The semantic restored layouts differ in only two W4 codes, at FiLM code indices 703 and 1514. Carrier movement is much larger:

- Basis codes: 5,989/27,648 changed, all on atoms 2/5/9; total per-atom L1 displacement `3,944 / 3,866 / 3,364`; maximum absolute displacement 8.
- Basis scales: exactly three changed, atom 2 `0.0240247→0.0514815`, atom 5 `0.0259160→0.0555344`, atom 9 `0.0273433→0.0585929`.
- Coefficients: 7,044/7,200 changed; every one of the 600 coefficient rows moved; changed counts by dimension are `584, 593, 593, 588, 585, 594, 593, 589, 579, 578, 579, 589`.
- Coefficient L1 displacement by dimension is `10,323, 17,359, 14,517, 10,372, 11,386, 17,087, 11,230, 14,555, 11,887, 9,152, 9,352, 14,189`; maximum absolute displacement by dimension is `156, 165, 121, 108, 202, 199, 132, 143, 201, 160, 169, 147`.
- All 12 coefficient scales changed (`L1 = 6.5778353e-5`, maximum absolute change `1.6798178e-5`).

ExperimentBook's “473 rows / 2,234 symbols” figure is the later F24→F26 local Jacobian solve, not the full PR130→PR135 displacement. Conflating those denominators would erase the inherited PR133 solve.

## Per-section entropy ledger

All rows below are `[custodied exact bytes; memoryless H0 on the named source alphabet]`:

| Source surface | Stored/source size | H0 floor | Reading |
|---|---:|---:|---|
| `p` bytes | 186,624 B | 186,601.864 B | compressed output nearly byte-uniform; not an original-symbol bound |
| raw model prefix | 74,860 B | 74,364.188 B | outer raw-LZ stores 71,822 B |
| semantic PR130-layout raw | 40,252 B | 36,805.347 B | WANS body is 36,040 B via context |
| HPAC IHS1 same-state raw | 20,179 B | 16,566.648 B | IHS2 canonical is 16,599 B |
| basis symbols | 27,648 symbols | 12,156.649 B | Huffman is 98,213 bits |
| coefficient AR residuals | 7,200 symbols | 9,448.032 B | Rice is 78,036 bits |
| compact residual canonical | 100 B | 72.268 B | physical body is 96 B after fixed prefix elision |
| selector canonical | 14 B | 6.663 B | physical body is 9 B after fixed prefix elision |
| RC64 output bytes | 114,706 B | 114,679.245 B | exact model ideal is 114,705.460 B |

The remaining elementary gaps are concentrated in CAP1 basis/Rice fields, not RC64 termination. They are candidates for better context models only; they are not free-byte promises.

## ExperimentBook complete read

The inventory read all 231 files: README, CHECKPOINT_01 through CHECKPOINT_02E, 48 Markdown records, 146 Python files, source snapshots, tests, and manifests. The author's source-env report says 230 tests passed. In the retained copy, the pinned local run produced `181 passed / 32 skipped / 2 failed / 26 errors` in 4.84 seconds. Every observed failure/error is an omitted-environment dependency (`work/` archives/manifests or `third_party/challenge/frame_utils`), not a source assertion failure. Therefore the book is complete as a research record but not a self-contained reproducibility bundle.

### SHIPPED

| Surface | What shipped | Why it survived |
|---|---|---|
| fixed residual / schema | RCF1 + F24S fixed routing | exact receiver equality and smaller complete archive |
| HPAC | IHS2 v3, Gate A, tight rows, int4 frame, packed biases | exact IHS1 restoration; whole-archive win |
| semantic model | WANS1 adaptive streams | exact raw-state restoration; `−610 B` on common parent |
| carrier | CAP1 AR predictor + Huffman basis + Rice residuals | exact coefficient/basis restoration; `−81 B` on common parent |
| tokens | RC64 native decoder | exact token equality; `−2,122 B`; near-zero model overhead |
| learned carrier | CBQ atoms 2/5/9 + full PR133 carrier state | best joint pose/rate point, `−815 B` at F17 |
| optimization | Jacobian proposals + exact int12 ±1 search | accepted complete-score improvements only |
| joint semantic/carrier | two FiLM codes plus carrier compensation | small but real seg improvement without sacrificing final score |
| frame 0 | five-event sparse K=8 selector | retained only accepted joint pose events |
| runtime | compiled C RC64 decode on CUDA path | exact decoded output inside 30-minute CUDA budget |

### TRIED-DROPPED

| Family | Tested form and stopping reason | Verdict scope |
|---|---|---|
| uniform semantic W3 | `−852 B`, but n32 score `0.2087→0.9496`; QAT archives 188,896 B / score 0.8605 and 190,888 B / score 0.3843 | **FORMULATION:** static uniform W3 |
| simple token priors/range variants | alternative priors, range coders, and chunking did not beat RC64 complete bytes | **FORMULATION:** tested coders on F26 state |
| HPAC parameter edits | repacks, clipping, output bias, temperature, and LoRA; best LoRA cost `+280 B` | **FORMULATION:** tested post-hoc HPAC edits |
| motion replacement for HPAC | `0.080741 bits/token` versus HPAC `0.007788887 bits/token` | **FORMULATION:** tested motion representation |
| SSM gate | only 45 B potential against a 5,120 B implementation threshold; decoder not built | **INSTANCE preflight**, not family death |
| coefficient vector coding | CVQ/CVH K=8/K=16 cost `+40…+225 B` | **FORMULATION:** tested codebooks |
| alternate coefficient predictors | no complete-archive win over selected AR form | **FORMULATION:** tested predictors |
| carrier containers/order/compression | alternate formats, field orders, LZ, and deflate lost on exact archive bytes | **FORMULATION:** tested containers |
| dense frame-0 selectors | superseded by baking and five-event sparse selector | **FORMULATION:** dense storage |
| frame-1 basis / pose heads | failed complete score or byte threshold | **FORMULATION:** tested heads |
| explicit margin overlay | byte cost exceeded realized seg benefit | **FORMULATION:** explicit stored overlay only |
| renderer-only seg polish | lowered d_seg, but pose damage outweighed it in complete S | **FORMULATION:** unjoint renderer polish |
| F25 Jacobian state | superseded by F26 joint result | **INSTANCE:** F25 state |
| alternate deterministic CUDA kernels | failed exactness/speed selection against shipped runtime | **FORMULATION:** tested kernels |

### NEVER-TRIED in the retained book

- Same-state lc2 ANS on F26's exact token symbols/probabilities.
- Direct R-null or scorer-blind pixel filling. The archive does not store raw camera pixels, so direct application is structurally inapplicable; learned gauge-constrained retraining remains distinct and untested.
- Hood-static clamp on the F26 receiver.
- Scorer-aware per-cell mixed precision. Static W3 is not the adaptive/aware formulation.
- Implicit or mask-free edge conditioning. The explicit stored margin overlay was tested and dropped.
- Rate-aware carrier gauge/QAT on the F26 learned state.
- Exact bit-packing of CAP1 predictor/length/Rice metadata.
- New global starts or placement search beyond the book's local ±1 Jacobian neighborhood.

## Ranked unidentified-edge crosswalk

Rate projections use `25 / 37,545,489 = 6.658589531e-7 S/B` and hold distortion fixed. They are `[arithmetic projection; no scorer]`, never score claims.

| Rank | Edge | Evidence SHA / borrowed-substrate accounting | Projected ΔS on PR135 | Cheapest honest falsifier | Consumer and disposition |
|---:|---|---|---:|---|---|
| 1 | Current-base joint int12/basis/FiLM solve with our additional starts and pose engines | PR135 measurement `14f485530f21d1c90bb75407a4fe05292fb3a1afdeec5e32498b742568d31d17`; pi136 memo SHA `acd6a7972db2699bfb461075c437dd8b4315594ab700d27d046b5fb142188111`. Method transfers; learned values remain counted; vehicle is borrowed PR130+PR133. | Unpriced; must be measured. It attacks the live lc2→PR135 gap, 95.1% pose. | Reuse retained F26 arrays/targets, run deterministic resumable local proposal stages, retain every candidate, then score only Pareto survivors under the fleet slot. | `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN/#995 successor`; consumer current-base joint-solve store. |
| 2 | lc2 same-state ANS token recode | ai1 memo SHA `f5ed312171aa971d9155e05ef5696152cf1070bc46d39a8e13ee87e8924ace6a`; cross-state lc2 token is 114,528 B plus 39-B temporal sidecar versus F26 RC64 114,706 B. Coder method is ours; exact probabilities/state do not transfer. | `−0.000118523` if the 178-B whole-state advantage survives. | Encode F26's exact token/probability state with both coders, retain both payloads, strict equality decode, build complete archives. No scorer needed. | `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN/lossless-pack successor`; consumer F26 same-state coder race. |
| 3 | CAP1 fixed metadata packing | Exact CAP1 fields in `14f485…d17`: q8 factors can use base+7-bit deltas (−12 raw B), biases signed-6-bit (−3), lengths 4-bit (−16), Rice k base+bits (−9). This is a new exact-storage proposal; carrier values stay borrowed/counted. | Ceiling `−0.000026634` for 40 raw B before outer LZ; actual archive delta unknown. | Strict pack/unpack equality plus complete archive build and repeat hash. | `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN/lossless-pack successor`; consumer CAP1 pack-absorption store. |
| 4 | Adaptive scorer-aware per-cell mixed precision | #869/tw1 memo SHA `315dd1f2076d854f28ea6cc08fd9e025d0c4114175bb92b3877fbab5c80f7f76` found state-dependent marginal effects (+13.1% on 52/53 cells) and joint superadditivity up to 7%, but on a different IX2/D1 object. Static W3 death does not transfer. | No credible byte or S projection on F26. `−852 B` is only the catastrophic static-W3 ceiling, not expected gain. | Identify high-sensitivity cells from the exact F26 Jacobian, then one deterministic mixed W3/W4 complete archive with strict receiver equality. | `QUEUED-WITH-A-FIRE-ORDER`; owner `#869 successor folded into #995`; consumer current-base quantization screen. |
| 5 | Implicit edge-conditioned seg adjustment / m91 allocation | sg2 memo SHA `89c97…9e5`, result SHA `392d2…f4b`; Road↔Lane is a large participating interface but not the whole residual. Book killed explicit overlay, not implicit conditioning. | `100·Δd_seg`; no honest effect-size projection. | Add no stored mask: condition existing joint proposal priorities on measured edge cells, then retain complete candidates. | `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN/#995 successor`; consumer joint-solve proposal bank. |
| 6 | Hood-static clamp | #139 component evidence SHA `224b5d…3ed`; older vehicle measured a 34-B controller and large d_seg response, but no F26 transfer proof. | Rate floor `+0.000022639` before any distortion benefit; distortion unpriced. | Current-receiver, deterministic hood-only edit with strict component self-detection and full-population scorer only after a byte-closed candidate exists. | `FOLDED` into #995's conditional proposal screen; not a standalone arm. |
| 7 | #580 R-nullity / #401 blind fill | aa1 direct audit SHA `bdb293249c2e8d398d85db2a3ba76750cc3bc82820abd033d40a73e1bcc5583d`: PR130/CPR1 stores zero camera-residual bytes; direct reclaim is 0 B. PR135 has the same archive shape. | Direct form `0 B / 0 S`; learned gauge retraining unpriced. | No direct probe warranted. A future learned formulation must alter training/state, not fill absent raw pixels. | Direct path `FOLDED`; learned gauge form `QUEUED` inside #995 only. |
| 8 | pk2 pose-carrier representation attack | PK2 result SHA `ef164b…d800`, final receipt SHA `512315…059`: n120 best unchanged; exact low-rank+residual cost `+4,316 B`, projected `+0.002873847 S`; post-hoc gauge/capacity variants lost. | `+0.002873847` before distortion on tested parent. | None for frozen post-hoc form. Reopen only with learned rate-aware QAT and pre-gate `≥2,000 B` plus MSE `<2.5e-6`. | Frozen form `FOLDED`; conditional learned form owned by `pz4`. |
| 9 | Post-08-06 online same-family publication | Bounded receipt `f5c0ed…e31f`; no matching publication found in Aug-06→Aug-10 scope. GVC-RT was submitted Aug-05 and uses pretrained LFQ/generative detokenization; NeuroQuant is older mixed-precision calibration support. | None. | Repeat bounded primary-source search only after a new paper/date trigger. | `FOLDED` for this window; consumer research index. |

The edge table deliberately does not transfer #869's old `−113,555 B` task-lossy projection, #139's prior-vehicle d_seg result, or lc2's cross-state exact byte saving into PR135 as realized facts.

## Recall evidence

The recall pass exceeded the charter seeds before any candidate was ranked.

### Surfaces and queries searched

- `.omx/research/`, arm receipts, `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG, task/probe ledgers, and `.omx/state/main_hot_state.md` by content for: `PR135|F26|PR130|RC64|WANS|CAP1|ANS|blind|null|ker(A)|gauge|hood|per-cell|mixed precision|edge|Road Lane|carrier|pose representation|margin`.
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered with the same representation/rate/pose/seg terms.
- PR135 ExperimentBook's complete file inventory and contents, including `carrier_cbq`, `hpac_lora`, `frame0_selector`, `joint_pose_solve`, `residual_calibration`, `entropy_audit`, `margin_allocation_audit`, `renderer_weight_codec`, `rc64`, and `coder_overhead`.
- Exact corpus receipts for aa1/#401/#580, #139, pk2/pk3, #869/tw1, sg2/m91, ai1/lc2, pi135/pi136, and the current live board.
- Primary-source web search bounded to publications announced 2026-08-06 through 2026-08-10 that attack learned carrier/token/mixed-precision video coding.

### Beyond-seed findings and plan changes

1. **Seg was not untouched.** The book contains margin-allocation and renderer-polish work, and the final accepted state moves seg slightly through two FiLM codes plus carrier compensation. This changed the plan from “find the first seg attempt” to “separate dead explicit/post-hoc edits from still-open implicit joint conditioning.”
2. **Direct blind/null fill is a representation mismatch.** aa1 measured zero camera-resolution residual bytes in CPR1; that closes direct byte reclaim on this archive shape. The only live descendant is learned gauge-constrained retraining.
3. **Token waterfill is state-dependent.** #869's measured 52/53-cell interaction prevents importing old per-cell savings; it supports an adaptive screen, not a byte credit.
4. **The live PR135 row landed.** The CUDA replay supersedes the charter's in-flight state and supplies current authority; the CPU axis refused by the shipped CUDA lock.
5. **CAP1 leaves a 40-raw-byte exact metadata hypothesis** that was named in audits but not implemented in the retained book. It is small, honest, and composable with the same-state coder race.
6. **Online bounded absence:** no same-family publication was found in the Aug-06→Aug-10 window. The closest current primary source, GVC-RT, predates the window and changes the family to pretrained LFQ/generative detokenization; it is not a direct F26 edge.

## Borrowed-substrate accounting

PR135 borrows PR130's semantic-token renderer, IntegerHPAC renderer, CPR1 carrier semantics, counted learned weights/coefficients, and overall receiver shape. It borrows PR133's constrained basis and joint coordinate re-solve. F26 contributes exact lossless representations, two semantic codes, further exact-coordinate solving, sparse frame-0 selection, and native RC64 decode. None of this is ours-original.

The transferable pieces from Pact are methods only: lc2 coder construction, joint-solve starts/pose engines, adaptive state-aware allocation, implicit edge prioritization, and learned gauge constraints. Every F26 learned value remains video-derived and counted. Any resulting archive is a borrowed-vehicle successor until an original witness vehicle establishes a lower exact row.

## Checkpoint receipts

- **Checkpoint A — custody/map: COMPLETE.** Archive pins verified; all extracted payloads persisted; L0→L3 section sums close; strict receiver reconstruction passes.
- **Checkpoint B — book/recall: COMPLETE.** 231/231 ExperimentBook files read and inventoried; tests classified; corpus and online recall recorded with changed decisions.
- **Checkpoint C — handoff: COMPLETE.** Ranked edge table carries evidence, projection boundary, falsifier, owner, consumer, and disposition. No unnamed follow-on remains.

## Follow-on dispositions

- `current_base_joint_solve`: **QUEUED-WITH-A-FIRE-ORDER**; owner=`MAIN/#995 successor`; consumer_store=`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`; fire_trigger=`this memo lands, the lane is claimed, and the producer proves deterministic resume plus retained per-stage/per-candidate payloads`.
- `f26_same_state_ans_race`: **QUEUED-WITH-A-FIRE-ORDER**; owner=`MAIN/lossless-pack successor`; consumer_store=`/Volumes/VertigoDataTier/pact/pr135_f26_same_state_ans_20260810/`; fire_trigger=`the exact F26 symbol/probability export and receiver-equality harness are present, no duplicate coder lane is active, and every candidate payload is retained`.
- `cap1_metadata_pack`: **QUEUED-WITH-A-FIRE-ORDER**; owner=`MAIN/lossless-pack successor`; consumer_store=`/Volumes/VertigoDataTier/pact/pr135_cap1_metadata_pack_20260810/`; fire_trigger=`the same-state lossless builder can emit strict pack/unpack equality and repeat-identical complete archives`.
- `adaptive_mixed_precision`: **QUEUED-WITH-A-FIRE-ORDER**; owner=`#869 successor folded into MAIN/#995`; consumer_store=`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision/`; fire_trigger=`the exact F26 sensitivity map exists and a static-W3 byte credit is not assumed`.
- `implicit_edge_conditioning`: **QUEUED-WITH-A-FIRE-ORDER**; owner=`MAIN/#995 successor`; consumer_store=`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/`; fire_trigger=`the proposal uses no stored explicit mask and is evaluated jointly with pose/rate on retained complete candidates`.
- `hood_static`: **FOLDED** into the conditional #995 proposal screen; consumer_store=`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`; fire_trigger=`only if component self-detection and a byte-closed current-receiver candidate exist`.
- `direct_blind_or_null_fill`: **FOLDED** by aa1's zero-stored-camera-residual result; consumer=`research index`; no direct fire trigger. Learned gauge retraining remains inside #995.
- `pk2_frozen_posthoc`: **FOLDED**; owner=`pz4 for any learned successor`; consumer_store=`pz4 governed store`; fire_trigger=`only a learned rate-aware QAT proposal pre-proves ≥2,000 B and MSE <2.5e-6`.
- `online_aug06_aug10_same_family`: **FOLDED** as bounded absence; owner=`future intake`; consumer=`research index`; fire_trigger=`a new primary-source publication date or named paper`.

## Boundaries

Measured here: exact custody, every byte span/hash, strict parsing/restoration, byte-alphabet and source-symbol H0 on named surfaces, PR130↔PR135 learned-state displacement, and complete ExperimentBook inventory/test behavior.

Not measured here: a new candidate, any new d_seg/d_pose, same-state F26 ANS bytes, CAP1 whole-archive pack savings, learned gauge effects, adaptive mixed-precision effects, or implicit edge-conditioned score response. The CUDA authority row was consumed from MAIN's independent replay, not produced by this arm. The CPU refusal is not a CPU score.

This unit did not move the exact pointer and did not reach sub-0.15. The current own-vehicle frontier remains **lc2 `S = 0.16959899569230852 @ 187,226 B` `[contest-CUDA T4, adjudicated, n600]`**; PR135's `0.16226942370411543 @ 186,724 B` is the custodied external/borrowed bar.
