# DCF1 — duplicate-carry census, receiver tree-shake, and lossless factor/flatten verdict (2026-08-31)

## PRE-CHECK — prior work before each verb

| Charter verb | Prior surface checked | Status before this arm | DCF1 action |
|---|---|---|---|
| duplicate carry | MZ2 #1060, BZ2/BZ2D, RB1/W72 #1222, AR1B body census | The renderer/carrier relationship had evidence but no complete LB1 pair census. | Enumerate all 15 unordered pairs of the six stored quantities; answer the renderer-versus-carrier question from exact current section identity and receiver reads. |
| fullest shaking | #417 receiver-consumption bijection, MZ2 38/38 strict load, AR1B exact physical census | #417 supplies the counted-versus-consumed rule, but its AST extractor is specifically for `P[...]` parameter dictionaries, not LB1's `parts.<section>` object. | Use the #417 rule without pretending its schema-specific extractor can parse LB1: map every exact physical span to the current parser and real forward, and classify the complete 180,083-byte denominator. |
| factorization | D3B, MZ2, RR9, RA1, PC2, #1124 | Lane-chain factorization, semantic fixed-schema sharing, within-group reorder, and carrier post-hoc rank/refit were already measured. | Transfer the applicable verdicts to LB1 with explicit lineage limits; do not rerun closed surfaces. |
| flattening | #1200 addressing, JT23 section/ZIP audit, OC2 conditioning | Per-token/per-section addressing is already free; ZIP is at its 100-byte floor; section coders and conditioning are spent. JT23 called the RX1 header required grammar. | Do not remeasure addressing or coders. Test only whether the 14-byte RX1 header is recoverable from the already-stored self-delimiting streams. |

The pre-check therefore left one new exact question: whether the three Brotli end markers make the
RX1 section-length header redundant. Everything else below is a transfer, a bounded source proof, or
an explicitly absent candidate—not a disguised rerun.

## Outcome first

- **No kilobyte-scale saving exists in the enumerated current-body, post-hoc exact surfaces.** The
  charter's `>=1,000 B` prediction is falsified in this scope.
- The one new lossless format finding is **14 B**: the exact LB1 archive can be repackaged from
  180,083 B to **180,069 B** by omitting the RX1 header and discovering three consecutive Brotli
  boundaries. Candidate SHA-256: `c4e8874733dc2e2719840007787b0313d8312e62016c5d182792ca747e6b30c4`.
- This is a **research format proof, not a receiver-closed candidate**. The shipped receiver still
  requires `RX1M`; no Seg/Pose number, score, native identity, seal, or promotion is claimed.
- At the LB1 distortion, 14 B would be `-9.3220253437104e-06 S`, projecting
  `S=0.14802078380545025` only if a future native receiver proves byte-identical output. It closes
  **0.0332565%** of the 42,097-byte rate demand and leaves **42,083 B**.
- The frontier is **unmoved**. This unit found a small exact format fact; it did not advance the
  sub-0.12 goal.

## Shipped-body control

The scorer-free control ran before the candidate measurement.

| Artifact | Bytes | SHA-256 | Verdict |
|---|---:|---|---|
| pinned LB1 source | 180,083 | `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9` | pinned input |
| deterministic repack | 180,083 | `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9` | byte-identical |
| deterministic repack repeat | 180,083 | `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9` | byte-identical |
| original member `p` | 179,983 | `d13461716e8bedcea9e6d1eaecaf587cf39fe997036653437fcff572e55fdff7` | retained |

Receipt: `/Volumes/APDataStore/pact/ddm_dcf1_duplicate_carry_factorization/retained/final/stage_10_control.json`.
The experiment is resumable and refuses overwrite of any non-identical retained payload.

## Exact LB1 physical anatomy

| Physical region | Bytes | Exact current receiver use | #417 classification |
|---|---:|---|---|
| ZIP local header + central directory + EOCD | 100 | Locates the single stored member `p`; JT23 proved 31+47+22 is the one-character-name floor. | CONSUMED, 0 inert |
| RX1 header | 14 | `_decode_rx1_models` reads magic, version, codec, table mode, flags, and three section lengths. | CONSUMED today; DERIVABLE under the measured headerless grammar |
| HPAC stream | 13,515 | Decompresses to the 17,952-byte IHS1 model; `decode_production_tokens` materializes and loads it. | CONSUMED, 0 inert |
| semantic renderer stream | 30,856 | Decompresses to the 36,130-byte SM3R state; the real receiver loads it with `strict=True` and renders frame 1. | CONSUMED, 0 inert |
| frame-0 carrier stream | 22,010 | Restores the RR5/DX2 carrier, then supplies basis, 600x12 coefficients, and selector to rendering. | CONSUMED, 0 inert |
| residual correction table | 96 | `parts.table.values[feature]` is added to HPAC logits for every coding group. | CONSUMED, 0 inert |
| RC64 token stream | 113,492 | `NativeDecoder(..., parts.token_stream)` reconstructs the full semantic token field. | CONSUMED, 0 inert |
| **total** | **180,083** | Complete archive denominator. | **180,083 consumed; 0 inert; 0 indeterminate at top level** |

This is the fullest body-level tree-shake the current evidence permits. The #417 apparatus itself is
healthy (`tools/tests/test_levelset_receiver_bijection_gate.py`: **11 passed**); the broader parity
test cannot collect in this managed session because importing `mlx.nn` raises `No Metal device
available`. The apparatus hashes used here are:

- `tools/levelset_receiver_bijection_gate.py` — `5e94e67f9ca8aaa76dccd3759dd0cfac4f1879fea4e9228fb5592690ba0fb643`
- LB1 `runtime/residual_archive.py` — `aca361f3e94941f4f2800bacec79f5032335588e317e76ee1a306bbb5ba64530`
- LB1 `runtime/f26_inflate.py` — `5d705f93c051b2b540845dad4140f73d7dd61c721e4de2ed33b2ad32170c35c4`

The important distinction is that #417's exact AST vocabulary extractor recognizes literal or
formatted `P[...]` keys. LB1 consumes a typed `ResidualArchiveParts` object and a strict state dict.
Calling the extractor on that source would report a vacuous vocabulary, not a valid tree-shake.
DCF1 therefore applies #417's governing bijection—every counted span must have an exact receiver
consumer—using the actual `parts` reads and strict state load. No new generic gate was built.

One subfield is not credited as inert: RA1 showed that the carrier's 12 positive `basis_scales`
mathematically cancel under per-atom RMS normalization and priced their removal at 7 coded bytes,
but it also measured a `2.24e-7` float32 basis perturbation. Without retained full public-output
identity, those 48 raw bytes are **indeterminate at the final uint8 receiver surface**, not dead.

## Duplicate-carry census — 15/15 unordered pairs

The six stored quantities are: HPAC probability model `H`; semantic renderer `S`; frame-0 carrier
`C`; correction table `R`; token stream `T`; and physical framing `F` (100-byte ZIP + 14-byte RX1).
“Exact” below means derivable without learned/video-derived bytes. “Lossy” means a receiver-relevant
relationship exists but replacement would require a changed model or descent. “None” means neither
object determines the other.

| Pair | What each object stores / decoder need | Derivability verdict |
|---|---|---|
| H x S | H maps context to token probabilities; S maps decoded token indices to frame-1 pixels. Both are read on different sides of the token boundary. | NONE either direction |
| H x C | H is a token probability model; C is frame-0 basis, coefficient trajectory, and selector. | NONE |
| H x R | R is a 6x4 float correction table added after H's logits. It could only be absorbed by changing/refitting H, not reconstructed from current H. | LOSSY/retrained only; no exact carry |
| H x T | T is decoded under H, but a probability model does not determine a particular symbol sequence and a sequence does not determine its fitted model. | DEPENDENT but NONE exactly |
| H x F | F is container syntax and lengths; H is learned probability content. | NONE |
| S x C | S produces the semantic frame from T; C produces frame 0. PoseNet scores their composed frame pair. | COMPLEMENTARY, not duplicate |
| S x R | R changes token decoding probabilities; S consumes the resulting tokens. | DEPENDENT pipeline, NONE exactly |
| S x T | S is shared rendering weights; T is the per-site semantic field. Either can vary while holding the other fixed. | NONE |
| S x F | Syntax versus learned renderer content. | NONE |
| C x R | Frame-0 synthesis versus token-coder correction. | NONE |
| C x T | C supplies frame 0; T supplies the semantic field used for frame 1. | COMPLEMENTARY, not duplicate |
| C x F | Syntax versus carrier content. | NONE |
| R x T | R participates in the arithmetic decoding distribution, but neither it nor T determines the other. | DEPENDENT but NONE exactly |
| R x F | Syntax versus correction values. | NONE |
| T x F | The token bytes are content; framing locates them. The three model lengths are jointly derivable from Brotli end markers, but not from T alone. | NONE pairwise; one higher-order framing redundancy below |

### The carrier answer

The carrier does **not** encode pose information that makes the semantic renderer disposable, and the
semantic renderer does not make the carrier disposable. The current receiver uses the carrier to
construct frame 0 and the semantic state plus tokens to construct frame 1; PoseNet sees the pair.

The strongest same-object control is BZ2D: its body retained the semantic stream
`39d1be52ba629334...` and carrier stream `932b979f5181b331...` byte-identically to LB1, yet changing
the token/generator field made pose about **1,603.5x** worse. Thus the 22,010-byte carrier is neither
a stored PoseNet-output guarantee nor a duplicate of the 30,856-byte semantic renderer. It is one
input to a joint rendered-pair mechanism. Exact derivability is **NO** in both directions; any
renderer/carrier collapse is a changed-representation, joint-descent hypothesis, not a lossless
tree-shake.

### The one higher-order duplicate carry

`F_RX1` stores three lengths already recoverable from the end markers of `H || S || C`. DCF1's
one-byte-at-a-time parser found boundaries at offsets 13,515, 44,371, and 66,381 in the headerless
member, exactly matching the old header. The remaining 113,588 bytes split by fixed receiver grammar
into the 96-byte table and 113,492-byte token stream. Reinserting the old header reconstructs the
original member byte-for-byte.

## Factorization adjudication

| Candidate | Current-body transfer / real-coder price | Disposition |
|---|---|---|
| shared semantic 38-tensor representation (MZ2 #1060) | The strict receiver topology transfers: all 38 named tensors remain required by the same `SemanticTokenRenderer(96)` load contract. MZ2's exact e480b dense/sparse/dictionary/hybrid race was **+340 B**, with 0/38 structural constants. Its numeric archive price is not relabeled as a new LB1 measurement; the formulation-level negative transfers, so it was not rerun. | **FOLDED** — exact fixed-schema sharing closed |
| Lane-versus-rest chain (D3B) | All nine real RC64 candidates reconstructed the exact field. Best factor archive was 180,575 B, **+360 B vs GB1** and **+492 B vs LB1**. | **FOLDED** — exact lossless lane chain loses |
| carrier basis x coefficient rank/refit (RA1/#1124) | Rank 4 coded to 7,569 B in its real-coder pricing domain, saving 14,709 B, but changed the carrier by RMS 12.527 grey and missed pose by orders of magnitude. LB1 inherits the same carrier object; later RR5/DX2 changes its outer coding, not the linear-synthesis verdict. | **FOLDED** — post-hoc linear rank/refit family closed |
| carrier basis-scale gauge | All 12 signs were positive; magnitudes cancel analytically. RA1 measured 22,278 -> 22,271 B (**7 B**) through real Brotli, with `2.24e-7` float32 drift. | **QUEUED-WITH-A-FIRE-ORDER**, no exact-output credit yet |
| token permutation/factorization (RR9) | 114,000 groups / 117,964,800 symbols were genuinely permuted; real RC64 stayed 113,777 -> 113,777 B (**0 B**). Cross-group order is the trained HPAC mask, not a free permutation. | **FOLDED** — exact permutation family closed |

There is no unmeasured current-body cross-pair factor packet left in the recalled corpus. A genuinely
joint renderer/carrier or cross-pair representation remains conceivable, but it changes the trained
object and is therefore **ABSENT** from this scorer-free post-hoc arm, not an unpriced claim.

The G110 public-product specification reinforced the coding rule rather than supplying bytes: it
eliminates factor scale, fixes sign and integer gcd gauges, and stores only the irreducible product.
That is a plausible design pattern for a future born-factorized body, but G110 has no materialized
LB1 operand and cannot be transferred as a current-body saving.

## Flattening measurement

DCF1 retained the exact streams, decompressed bodies, tail, control, candidate, and deterministic
repeats under:

`/Volumes/APDataStore/pact/ddm_dcf1_duplicate_carry_factorization/retained/final/`

| Object | Bytes | SHA-256 |
|---|---:|---|
| HPAC stream / decoded | 13,515 / 17,952 | `602115b323b0e403...` / `e8c0cfd73d3275ad...` |
| semantic stream / decoded | 30,856 / 36,130 | `39d1be52ba629334...` / `cd99fdcdaae4d5c0...` |
| carrier stream / decoded | 22,010 / 22,008 | `932b979f5181b331...` / `b73eab2cee3ce71d...` |
| residual + token tail | 113,588 | `6f81a1d7dcea628f...` |
| headerless member | 179,969 | `f8dd84130290a2d3...` |
| headerless archive + repeat | 180,069 each | `c4e8874733dc2e27...` each |
| reconstructed original member | 179,983 | `d13461716e8bedce...` |

This **partially supersedes JT23 only on its statement that the RX1 14-byte header is required**.
JT23's real-coder closure and the 100-byte ZIP floor remain intact. #1200's free per-token addressing
also remains intact; DCF1 did not remeasure it. The new decoder rule is generic—consume three Brotli
streams until their end markers—so no video-derived length table is hidden in free code.

The candidate is not directly consumable by the old runtime and is deliberately not presented as
an archive that can be scored today. Native receiver patching, full retained output identity, repeat
identity, and seal would be mandatory before composition.

## Denominators and typed outcomes

- Duplicate-carry denominator: **15/15 unordered pairs** across six stored quantities enumerated;
  **15 adjudicated**, 0 exact pairwise duplicates, 0 pairs with >=1,000 B credit.
- Receiver tree-shake denominator: **7/7 physical regions**, **180,083/180,083 bytes classified**;
  current receiver consumes 180,083 B, top-level inert 0 B, top-level indeterminate 0 B.
- Factorization denominator: **5/5 named current-body families adjudicated**; 4 folded by transferred
  real evidence, 1 basis-scale gauge queued at a recalled 7 B pending exact receiver identity.
- Flattening denominator: **3/3 surfaces adjudicated**; per-token addressing closed by #1200, ZIP
  closed at 100 B, RX1 newly measured at 14 B.
- New DCF1 payload rows: **1/1 measured and retained**, plus a byte-identical repeat; no candidate
  was discarded. Changed-representation joint factorization is **ABSENT**, not counted as measured.
- Prediction result: the charter's `>=1,000 B` post-hoc current-body prediction is **FALSIFIED in
  this enumerated scope**. The largest new exact format saving is 14 B.

## RECALL EVIDENCE

Queries were bounded to the chartered body and exact mechanisms:

- `ddm_dcf1|duplicate carry|carry factor|factorization` in `MEMORY.md`: no prior DCF1 memory hit.
- `lb1|5b856e...|180083|RX1|semantic|carrier|token_stream` across `.omx/research`, receipts, and the
  LB1 retained runtime.
- `receiver-consumption|bijection|#417|P[...]|strict=True` across `tools`, tests, and receiver source.
- `MZ2|38/38|derive|D3B|Lane|rank/refit|basis_scales|RR9|reorder|JT23|section coder|#1200|addressing`
  across research memos, the task ledger, canonical index, DAG, designs, and specs.
- `factor|carry|receiver|section|entropy|gauge|basis|coefficient|flatten|self-delimit|zip|archive`
  through `tools/list_canonical_equations.py --json`.
- `G110|generic two-layer public product|product scale` in the original task-space codec specs.

Beyond the seed documents, two facts changed the decision. First, the exact G110 design confirms
that scale/sign gauges belong in a canonical born-factorized product, but it has no LB1 payload and
therefore cannot donate byte credit. Second, the exact current Brotli streams falsify JT23's narrow
RX1-required claim: their end markers recover all three section boundaries without stored lengths.
The canonical equation registry returned general entropy-coded locality and archive laws but no
unconsumed LB1 region or factor identity that survives the current receiver.

## Typed fire orders

- **QUEUED-WITH-A-FIRE-ORDER — headerless RX1 bank.** Owner: MAIN lossless-bank composer. Consumer:
  `/Volumes/APDataStore/pact/ddm_dcf1_duplicate_carry_factorization/retained/final/candidate_headerless_rx1.zip`.
  Fire only when another independently admitted same-body lossless candidate supplies at least 16 B,
  taking the composition to the approximately 30 B solo-fire bar. Then patch a copied runtime to
  discover three Brotli end markers, retain full n600 output, prove old/new byte identity and repeat
  identity, seal, and only then request authority evaluation.
- **QUEUED-WITH-A-FIRE-ORDER — basis-scale gauge.** Owner: the next MAIN carrier-format owner.
  Consumer: the same DCF1 lossless bank plus RA1's retained carrier payloads. Fire only as part of a
  >=30 B composition and only with a receiver that omits the 48 raw magnitude bytes; require full
  retained output identity because `2.24e-7` float32 drift prevents an exact claim today.
- **FOLDED — renderer/carrier duplicate deletion.** The objects serve different frames and neither
  derives the other; BZ2D's identical-section control proves the carrier is not a pose guarantee.
- **FOLDED — MZ2 semantic sharing, D3B lane chain, RA1 post-hoc rank/refit, RR9 permutation, JT23
  section coder, #1200 addressing.** Their scoped real measurements transfer; do not rerun them on
  unchanged objects.
- **ABSENT — born-joint cross-pair factorization.** It remains a new trained-representation problem,
  not a DCF1 current-body candidate. It may enter only with a real retained payload and a paired
  current-body control; no byte projection is carried forward.

## Verification and custody

- Experiment: `experiments/ddm_dcf1_duplicate_carry_factorization.py`.
- Result: `/Volumes/APDataStore/pact/ddm_dcf1_duplicate_carry_factorization/retained/final/DCF1_RESULT.json`.
- The final run's experiment source SHA is `9a2b8c8bf1eb8446c318fd9fc92dd8fbd94c4e42bb28173a2164fcf311c90e6e`;
  its result JSON SHA is `b57a2c2d85c70f5ae3bfd5196bc7d96669f9e5872958553492e2672d92a4c873`.
- The pre-format measured source was preserved at
  `/Volumes/APDataStore/pact/ddm_dcf1_duplicate_carry_factorization/retained/experiment_source_7906d01d91b9ae02ade308763a59c9e1ca3f1cff6fd5d4f4f77c1b6e702e242b.py`;
  the formatter-clean committed source was rerun from scratch in `retained/final`.
- `py_compile`: PASS.
- #417 focused tests: **11 passed**.
- #417 MLX parity collection: BLOCKED in this managed session by `No Metal device available`; no
  authority consequence because DCF1 neither changed nor scored the levelset receiver.
- No scorer, Modal, upstream mutation, queue-state mutation, or protected-file edit occurred.

[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9
