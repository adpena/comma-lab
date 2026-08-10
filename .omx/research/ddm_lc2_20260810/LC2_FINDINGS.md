# LC2 lossless coder stack findings

LC2 is **READY for MAIN's exact evaluator row**, not an exact score receipt. The composed archive is
187,226 bytes, 1,410 bytes below AI1, and its fresh-cache literal receiver reconstructs PR130's full
3,662,409,600-byte raw byte-for-byte in 856.04 seconds. Every row remains `score_claim=false`.
The pure-rate score prediction is `0.16959372113725102`, explicitly
`DERIVED_NOT_EXACT_RECEIPT`; the canonical pointer and own-vehicle frontier are unmoved.

## MEASURED RESULT

| step | complete `archive.zip` bytes | delta from prior | SHA-256 | axis |
|---|---:|---:|---|---|
| AI1 ANS + temporal source | 188,636 | - | `0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84` | `[archive-byte exact; scorer-free]` |
| split Brotli on the unchanged model sections | 187,771 | -865 | `41ebfa4a7b640c6f9b8c033bd750e46c96d4d79adee5d650455bcb4d06df53fe` | `[archive-byte exact; scorer-free]` |
| CX2 reversible xcodec + split Brotli | **187,226** | **-545** | `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45` | `[archive-byte exact; scorer-free]` |

The measured total is **-1,410 bytes versus AI1** and **-3,826 bytes versus PR130**. The corrected
charter prediction was 187,733 bytes, so the composed result is 507 bytes smaller. The split step
saved 865 bytes rather than VP1's isolated 903-byte estimate, a 38-byte unfavorable difference;
the reversible CX2 step then saved 545 bytes, leaving a favorable 507-byte net interaction against
the corrected prediction.

The derived score is:

`0.172141297491896447 - 25*(191,052 - 187,226)/37,545,489 = 0.16959372113725102`.

This is not an exact-evaluator claim. It is licensed only as a prediction because the raw bytes are
identical; one exact n600 row is still owed. The derived gap to 0.15 is about 0.019594 score units,
or 29,427 rate-only bytes at the same distortion.

## Optimal-form search and ties

The bounded denominator was **54 complete archives**: the one shipped CX2 reversible transform,
Brotli q9/q10/q11 independently on semantic/carrier/HPAC, and stored versus DEFLATE ZIP. Every
candidate archive and every section stream is retained under
`/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/retained/`.

The minimum has three parameter rows but one unique archive byte string:

- semantic: signed-zigzag + 4,096-byte even/odd lane transform, Brotli q10;
- carrier: identity, with q9/q10/q11 byte-identical;
- HPAC plus temporal wire: XOR-0x80, Brotli q10;
- outer ZIP: stored.

The best DEFLATE archive is 187,286 bytes, exactly 60 bytes worse. The complete search receipt is
`/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/search/complete_zip_search.json`, 140,514 bytes,
SHA-256 `bc582ee2b42171aa628db0e03d98c6937d860ac0d30d2a7bf68db7ad95af4676`.

## Receiver closure

The final archive parses through the shipped selector and actual loaders with every section
consumed. Exact identities are:

- reconstructed model wire: 83,540 bytes,
  `618ac80da2bfb82a52a94317877cfd79af71290f751e3d4f130a46258b29092a`;
- PR130 base model bytes: 83,493 bytes,
  `62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517`;
- ANS token payload: 114,528 bytes,
  `85d6c199ffb93ddab0fe1631448882a255e9fea1f6858bab5a04cea2310a7331`;
- temporal sidecar: 39 bytes,
  `f920f7be8108b83831971a8d07c9ef522eadb18abed095cf395bf3a6f871e796`.

The fresh-cache receiver ran from an empty token-checkpoint surface on macOS CPU, inherited the
pinned runtime's normal thread policy, decoded all 117,964,800 tokens, proved the ANS terminal state
empty, rendered all 600 pairs, and completed in **856.0356150830048 seconds**. Its retained raw is
`/Volumes/APDataStore/pact/ddm_lc2_20260810/cold_decode/0.raw`, 3,662,409,600 bytes, SHA-256
`a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`, byte-identical to PR130.
The cold receipt is
`/Volumes/APDataStore/pact/ddm_lc2_20260810/cold_decode/cold_decode_receipt.json`, SHA-256
`afcec2a11ad306f5d0217b0aedfeeb10293a270012f3af168b05eaa53c567783`.

## Payload custody and interruption evidence

All materialized archives, transformed sections, compressed streams, ties, repeats, token caches,
raws, and interrupted-run bytes were retained with byte counts and hashes.

The first local receiver launch imposed a non-reference two-thread cap and hit the 1,800-second
gate after completing the token stage and rendering 150 pairs. It retained both 600-frame token
checkpoints and a 1,043,786,736-byte partial raw. The launcher correctly refused to overwrite that
raw on the next attempt. LC2 then verified the timeout receipt and atomically cold-stored the partial
as
`/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/retained/decode/attempts/timeout_partial_b236448ae99308ed.raw`,
SHA-256 `b236448ae99308ed0de1787090680ebf854f1acefec05c2736dc398454d1dc0c`, before resuming. The
resumed receiver completed in 214.5683583340142 seconds and independently matched the raw SHA, but
it is not used as the cold runtime verdict. The later fresh-cache run supplies that verdict.

Terminal machine receipts:

- `/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/LC2_RESULT.json`, SHA-256
  `473fcc3b3d5303c626f7cfbd93b0714f0192a84e440835fac3a4fe6df83918ab`;
- `/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/READY_EXACT_EVAL.json`, SHA-256
  `f37045417f1ea6109a4d35870d2df86f751ba3ae39324448bda04b1246b1f7e8`;
- `.omx/research/ddm_lc2_20260810/LC2_FINAL_RECEIPT.json` is the compact repository receipt.

## RECALL EVIDENCE

The recall pass searched the full `research,equations,memory,dag,council,tasks,docs` corpus with
queries covering `PR130 split Brotli xcodec ANS temporal lossless coder stack #996`, `complete ZIP
reversible xcodec split model q4 AI1 CX2 VP1`, and `PPMd LDPC syndrome semantic section memoryless
slack`. It also queried the canonical equations registry for entropy/coder/context/residual terms and
searched the research indices, sub-0.15 DAG, task ledgers, harness bridge, receiver source, and
current queue/hot-state ownership.

Beyond the charter seeds, the pass found:

- VP1 explicitly recorded that the 73,065-byte unchanged-q4 split model pack had not been banked as
  a final public archive, and that complete-ZIP reversible xcodec required remeasurement on q4. This
  kept LC2 live rather than classifying it as duplicate.
- `RATE_AXIS_LOSSLESS_RACE.md` measured the unchanged-q4 isolated split win as 903 bytes. Together
  with AI1's exact 2,416-byte ANS-plus-temporal total, this corrected the withdrawn cross-regime
  187,250-byte arithmetic to 187,733 bytes.
- Task #996's `SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md` closed its named generic coder set but did
  not build the split public archive or the CX2 coordinate transform. LC2 therefore measured those
  missing composition cells instead of re-running #996.
- RC2 subsequently closed PPMd-style adaptive byte contexts on all four unchanged PR130 wire
  objects and LDPC/min-sum coding of the causal HPAC hit formulation. That prevented an unlicensed
  fourth coder from being added when the chartered stack already won.
- The actual receiver supports one CX2 transform, not an arbitrary transform selector. This narrowed
  the implementation to the shipped signed-zigzag/block-lane, identity, and XOR transforms while
  retaining the full legal Brotli/ZIP parameter surface.

No already-completed unchanged-q4 AI1 + split + CX2 public archive was found in those bounded stores.

## Authority boundaries

- No scorer or `upstream/evaluate.py` job ran. The arm did not own the scorer lane.
- `0.16959372113725102` is derived from immutable bytes and PR130's contest-CUDA anchor; it remains
  `score_claim=false` until MAIN lands the exact row.
- The local receiver axes are scorer-free macOS CPU proofs, not contest-CPU or contest-CUDA score
  rows.
- The PR130 bar, canonical pointer, and own-vehicle frontier did not move in this arm.

Own-vehicle frontier: `S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600` — UNMOVED.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN exact contest-row owner; consumer store: `/Volumes/APDataStore/pact/ddm_lc2_20260810/exact_eval`; fire trigger: claim the sole exact-eval lane, pass locked upstream-environment parity, and verify the archive is still 187,226 bytes with SHA-256 `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`; action: run one exact n600 row and either promote its exact receipt or report the discrepancy.
- `FOLDED` — owner: MAIN lossy-composition owner; consumer store: `/Volumes/APDataStore/pact/ddm_lc2_20260810/sd1m_separate_arm`; fire trigger: the LC2 exact receipt lands and a separate scorer lane is claimed; action: price SD1M as an explicitly lossy Seg/Pose/rate composition, never as part of LC2's lossless headline.

## LIVE-HYPOTHESES

- The exact row will land near the derived 0.16959372113725102 because both fresh and resumed literal
  receivers produced the exact PR130 raw from the immutable 187,226-byte archive; the remaining
  uncertainty is evaluator/environment receipt authority, not decoded content.
- SD1M may compose favorably after LC2 when priced as a separate lossy action because CX2 measured a
  byte win for it, but its changed semantic bytes require matched Seg and Pose evidence before any
  score claim.
- A representation change may reopen semantic/HPAC rate headroom because tensor or geometry
  structure can create a different probability object; #996 and RC2 close coder swaps on the current
  wire objects, not future representations.

## DEAD-ENDS

- The withdrawn 187,250-byte arithmetic is a cross-regime transfer from an SD1M-conditioned ladder;
  unchanged-q4 evidence fixes the prediction at 187,733 bytes.
- Outer ZIP DEFLATE is closed on this instance: its best complete archive is 187,286 bytes, 60 bytes
  worse than stored.
- Arbitrary transform search is not a legal LC2 receiver surface; only the shipped CX2 bijections are
  invertible under selector 3, so substitute transforms were removed before measurement.
- PPMd recoding of unchanged PR130 wires and LDPC/min-sum coding of the existing HPAC hit residual are
  family-scoped dead ends from RC2; none beats the incumbents.
- The artificial two-thread cold launch policy is closed: it timed out at 1,800 seconds, while the
  fresh-cache inherited-thread run completed in 856.04 seconds on the same archive and runtime.
- A checkpoint-resumed 214.57-second render is not a cold inflate proof; the separate empty-cache
  856.04-second run replaced it as the runtime verdict.
