# ddm_ntb1 — non-tail census and lossless prices; lossy re-solves remain owed

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false.

**Charter status: PARTIAL.** The census and bounded lossless roster are COMPLETE. No lossless winner exists among 13 tested formats (26 real encodes, independently reproduced after final formatting). HPAC pruning with a model-dependent tail re-encode was NOT implemented or measured. Renderer precision cuts with seg re-solve and n600 Seg/Pose were NOT run: the common contract assigns no scorer slot to this charter. Those are live obligations, not negative verdicts. Do not mark the full charter COMPLETE.

The competitive pointer did not move. No seal, no first-measurement intent, no Modal, no public cold n600 inflate, no new distortion measurement. There is no composed candidate to quote: the unchanged incumbent remains 180,406 B, sha `04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`, S 0.1372449041713402 [contest-CUDA T4 n600]. Replacing it with a zero-saving metadata variant would create no candidate benefit.

## Census — measured from the SHIPPED archive

Axis: [macOS-CPU advisory; exact bytes, scorer-free]. Original SSD sources read-only; execution uses a separate copy. Header `(RX1M,1,2,0,250,11911,29862,18610)` from the shipped `runtime/residual_archive.py`.

| Counted component | Bytes |
|---|---:|
| ZIP local fixed header + one-byte name p | 31 |
| ZIP central fixed directory + one-byte name p | 47 |
| ZIP end record | 22 |
| ZIP extras, comments, padding, data descriptor | 0 |
| RX1M header | 14 |
| HPAC section (Brotli-wrapped RC3H) | 11,911 |
| Renderer section (Brotli/CK2-wrapped SM1S) | 29,862 |
| Carrier section (Brotli; RR5/DX2) | 18,610 |
| **Non-tail total** | **60,497** |
| Tail: fixed table96 + counted RLC1 rider64 + arithmetic stream119749 | 119,909 |
| **Archive** | **180,406** |

The charter's term “token tail119909” includes the96-byte residual table and64-byte counted rider. Arithmetic token bytes alone are119749. No bytes go unaccounted for.

### HPAC detail

Restored RC3H rider:22 header +263 IHS1-prefix +25 mixer parameters +9198 arithmetic-coded weights +6351 tail =15859 B. After the actual `materialize_ihs1` and `cpr1/integer_model_io.py` loader: IHS1 is17770 B =4 magic +259 depth nibbles +11156 packed weights +6351 tail. The packed stream has7 padding bits. The integer loader executed successfully.

| Layer | Rows | Packed weight bits |
|---|---:|---:|
| frame_shift | 64 | 1,560 |
| frame_scale | 64 | 1,536 |
| conv_a | 64 | 49,910 |
| conv_b1 | 64 | 3,906 |
| conv_b2 | 64 | 1,425 |
| conv_past | 64 | 13,545 |
| spm_dw | 64 | 3,087 |
| spm_pw | 64 | 12,608 |
| head | 5 | 1,664 |

Every per-row precision is recorded in `CENSUS.json.hpac.fields`. Tail fields:

| Field | Type | Bytes |
|---|---|---:|
| frame_embed.weight | i1 | 4,800 |
| frame_shift.bias | <i2 | 128 |
| frame_shift.exponent | i1 | 64 |
| frame_scale.bias | <i2 | 128 |
| frame_scale.exponent | i1 | 64 |
| conv_a.bias | <i2 | 128 |
| conv_a.exponent | i1 | 64 |
| conv_b1.bias | <i2 | 128 |
| conv_b1.exponent | i1 | 64 |
| conv_b2.bias | <i2 | 128 |
| conv_b2.exponent | i1 | 64 |
| conv_past.bias | <i2 | 128 |
| conv_past.exponent | i1 | 64 |
| spm_dw.bias | <i2 | 128 |
| spm_dw.exponent | i1 | 64 |
| spm_pw.bias | <i2 | 128 |
| spm_pw.exponent | i1 | 64 |
| head.bias | <i2 | 10 |
| head.exponent | i1 | 5 |

### Renderer detail

The shipping SM1S reader restores a36130-byte SM3R mode6 body. Its code layout is driven by the shipping `rc1_adaptive_model_sections.walk_sm3r`; the real semantic decoder and strict renderer state-dict load pass. `runtime/entropy/renderer_weight_codec.py` was inspected: its WANS branch is not used by this shipped SM1S/SM3R representation. Using a WANS census here would inventory the wrong format.

| Layer | Bits/code | Retained codes | Packed body bytes |
|---|---:|---:|---:|
| token_embed.weight | 4 | 480 | 240 |
| frame_embed.weight | 3 | 4,800 | 1,800 |
| coord_mix.weight | 4 | 9,600 | 4,800 |
| blocks.0.dw.weight | 4 | 864 | 432 |
| blocks.0.pw.weight | 4 | 9,216 | 4,608 |
| blocks.0.film.weight | 3 | 1,536 | 576 |
| blocks.1.dw.weight | 4 | 864 | 432 |
| blocks.1.pw.weight | 4 | 9,216 | 4,608 |
| blocks.1.film.weight | 4 | 16 | 8 |
| blocks.2.dw.weight | 4 | 864 | 432 |
| blocks.2.pw.weight | 4 | 9,216 | 4,608 |
| blocks.2.film.weight | 4 | 16 | 8 |
| blocks.3.dw.weight | 4 | 864 | 432 |
| blocks.3.pw.weight | 4 | 9,216 | 4,608 |
| blocks.3.film.weight | 4 | 16 | 8 |
| head.weight | 4 | 2,592 | 1,296 |

Header/selection10 B; depth table8 B; remaining fp16 metadata/scales/prune masks are itemized by exact offsets in CENSUS.json. All body fields reconcile with zero remainder. Frame embedding and block0 FiLM are ALREADY3-bit; those cuts cannot be counted again. The three later FiLM weights each retain only16 codes.

Compressed section bytes cannot be allocated additively to individual layers: SM1/RC3 use shared arithmetic streams followed by whole-section Brotli. The table above reports exact restored body storage, not fictitious per-layer compressed prices. A layer treatment must be fully encoded and priced jointly.

### Carrier detail

The18645-byte decompressed stored body =6 bit counts +96 scales +40 packed metadata +12046 RR5 basis stream +6428 DX2 coefficient stream +29 selector tail. Counts are96368 basis bits and51424 coefficient bits, both byte-aligned. The shipping `rr5_arith_basis.split_carrier_body` supplies this map; “rice” is that parser's legacy key for the currently DX2-coded field. It does not mean the shipped coefficient stream is still Rice. Capacity was untouched; pc3 directories were untouched.

## Lever prices — real twin archives

Δbytes relative to the pinned180406-byte archive. ΔS below is **conditional rate arithmetic** `25*Δbytes/37545489`, not a new score. For every row the shipping parser returned identical HPAC blob, renderer blob, carrier blob, fixed residual payload, token stream, mixer weights and compensation. Thus Δd_seg=Δd_pose=0 is derived from unchanged decoder inputs, **not measured by a fresh n600 scorer**. No row meets the bar, which strictly requires31 saved integer bytes (30 B buys only1.9975769e-5 S).

| Lever | Δbytes | Conditional ΔS |
|---|---:|---:|
| zip_stored | +0 | +0.000000000 |
| zip_deflate | +58 | +0.000038620 |
| zip_bzip2 | +1,214 | +0.000808353 |
| zip_lzma | +2,492 | +0.001659321 |
| hpac_brotli_q11 | +22 | +0.000014649 |
| hpac_brotli_q10 | +0 | +0.000000000 |
| semantic_brotli_q11 | +91 | +0.000060593 |
| semantic_brotli_q10 | +1 | +0.000000666 |
| carrier_brotli_q11 | +7 | +0.000004661 |
| carrier_brotli_q10 | +2 | +0.000001332 |
| hpac_xz | +357 | +0.000237712 |
| semantic_plane_toggle | +591 | +0.000393523 |
| carrier_plane_toggle | +16 | +0.000010654 |

Member name and ordering are structurally fixed: the real parser requires exactly `['p']`; only one member exists. Wrong-name mutation on the real member was retained and rejected. Comments/extras/padding are already zero. The normalized ZIP control preserves size but changes external permission metadata (source0o100644 vs normalized0o600), so it is not falsely claimed to reproduce the source archive hash. All section and decoded-input identity comparisons remain exact.

The roster is bounded to the listed standard formats: **INSTANCE** negative only. No claim that all conceivable lossless repacks, renderer changes, or HPAC models are closed. HPAC quantization/pruning has no Δbytes/Δd/ΔS here; renderer precision/re-solve has no Δbytes/Δd/ΔS here. These are unmeasured, not zero.

## RECALL EVIDENCE

Queries and scoped excerpts: `ddm_ntb1_20260911/RECALL.json`; full canonical-equations command output: `equations_recall.json`; selected rows: `EQUATIONS_RELEVANT.json`. Searches by content covered `.omx/research/`, design docs in `docs/`, the canonical research index/DAG FEED surface, and canonical task-status ledger. Queries included `non.tail|model.section.*recode|HPAC.*prun|renderer.*int3|renderer.*precision|carrier.*repack`, `model.section|renderer.*pose|precision|repack`, and follow-up `hpac|semantic|non.tail|prun` in the ledger. The initial narrow task-ID query found0 lines in that ledger; broader content recall found cl3c's existing structural-coordinate follow-ons. This is a bounded recall, not a global absence assertion.

Beyond charter seeds, plan-changing findings:

- rc1's model-section recode and subsequent rc3/SM1 readers are already in move44. bz2d's13515/30856/22010 section counts are historical, not the starting price.
- `ddm_cl3c_closer_20260908.md`: on its historical object, λ2 lost224 joint bytes; extra seeds lost29/137 B; λ4 stopped under its preregistered rule. This rules out casually rerunning the same multiplier/seed sweep. Structural table/row pruning remains unmeasured on move44.
- `ddm_rf1_renderer_film_rung_20260824.md`: saving1078 B still caused large pose damage. A renderer-only byte count cannot supply admission; both scorers and the actual re-solve are mandatory.
- `ddm_rw1_boundary_local_renderer_weight_foldback_20260909.md` and ft1: exact code-grid and export parity matter; uniform-int4 training does not describe the current mixed-depth, row-pruned renderer.
- Canonical registry includes `section_coding_axis_closure_v1`, `model_section_edit_container_break_fee_v1`, `renderer_edge_layer_foldback_reach_v1` and HPAC joint-byte laws. The whole-container fee and model-plus-tail price determine admissibility.
- `experiments/ddm_jg2_tail_reencode.py::encode_tail` lacks move44's RLC1 mixer/observe path. Its reuse would silently price a different mechanism. The current shipping decoder explicitly creates `LaneMixer` for60 counted config bytes and observes each group. A correct model-pruning producer must preserve that full path. No proxy encoding was substituted.

Provenance seed pins verified: ls1 `b729faa146b62ea394ec502be37fb0d66d11f577202431cfcd9d5618f79f40bc`; move44 memo `f7638e1e171e0d19309f0d0b2f2f9f0acf9b5c70c5fb87f14bc029dfbaeaba8b`; bz2d `2c1bbe7b5ae43751bb387dcc7ed6bcc8585bfdd537698f1f6a8b7c3504ef9b69`. Archive matches charter SHA in full.

## Verification, custody and boundaries

Final producer: `experiments/ddm_ntb1_non_tail_bytes.py`, two review passes after final formatting, Ruff passes. Actual jobs run through `tools/launch_detached_process.py`; both census and price exit0. Final source reproduces all13 archive hashes from the first run. All26 twins and all intermediate bodies/sections remain on SSD. Stage receipts pin source archive, full copied source files, producer, seed, and upstream evaluator. Payload inventory is `ARTIFACT_MANIFEST.json`, generated outside the runtime trees. Original executed producer retained as `retained/producer_v1.py`; final run lives in `reproduction/` so no original input or stage evidence was overwritten. No large scratch created; automatic retention/write hook refuses below40 GiB free. No cleanup of another arm's artifacts.

Main receipts: `/Volumes/VertigoDataTier/pact/ddm_ntb1_non_tail_bytes/reproduction/CENSUS.json`, `/Volumes/VertigoDataTier/pact/ddm_ntb1_non_tail_bytes/reproduction/PRICES.json`; carrier and mutation proof: `/Volumes/VertigoDataTier/pact/ddm_ntb1_non_tail_bytes/VERIFY.json`; final source reproof: `.omx/research/ddm_ntb1_20260911/REPRODUCTION.json`. No upstream/PR/sealed-tree edits, no pc3/obx2/mx1 directory edits, no free-code relocation of learned content, no shared index manipulation.

Serializer result and bundle custody are in `ddm_ntb1_20260911/LANDING.json`. Git failure must not erase measurements; rc17 hands an exact bundle to MAIN. Canonical queue receipts are `TASK_EVENTS.json`; common-contract scorer restriction remains explicit. The full charter is PARTIAL, irrespective of whether the bounded lossless subtask is COMPLETE.

<!-- # FORMALIZATION_PENDING: bounded byte-only price receipt; no new exact row or family law claimed. -->

## NEXT_IF_RESUMED

- HPAC_JOINT — QUEUED-WITH-A-FIRE-ORDER; owner ddm_ntb1; consumer `/Volumes/VertigoDataTier/pact/ddm_ntb1_non_tail_bytes/hpac_joint`; fire next continuation after the current source stream is reproduced with an RLC1-aware all600 causal encoder; then measure one structural pruning treatment jointly with the tail.
- RENDERER_RESOLVE — QUEUED-WITH-A-FIRE-ORDER; owner ddm_ntb1, MAIN assigns scorer slot; consumer `/Volumes/VertigoDataTier/pact/ddm_ntb1_non_tail_bytes/renderer_resolve`; fire when MAIN assigns the sole slot and current source pins hold; run precision cuts with real seg re-solve and both frozen scorers n600, chunks≤120.
- COMPOSE_SEAL — QUEUED-WITH-A-FIRE-ORDER; owner ddm_ntb1, MAIN exact fire; consumer `/Volumes/VertigoDataTier/pact/ddm_ntb1_non_tail_bytes/composed`; fire at harvest of a treatment clearing the matched -2e-5 S bar; cold public600/manifest/census/smokes before seal, receiver changes route to first-measurement STOP.

## LIVE-HYPOTHESES

- Structural HPAC row/table pruning may reduce joint model-plus-tail bytes because multiplier/seed negatives do not test those coordinates. No current-object joint price exists here.
- Remaining4-bit renderer layers may pay after real seg re-solve, because the mixed-depth wire already admits3-bit values. Sensitivity, re-solve recovery and pose cost are unmeasured.

## DEAD-ENDS

- These13 fixed lossless formats on move44: best0 B saved, below the31-byte threshold. Do not rerun unchanged inputs/settings.
- ZIP rename/order/extra-field hygiene on this receiver: name must be `p`, one member only, extras/comments/padding already absent.
- Reusing JG2's old tail encoder unchanged: it omits the counted RLC1 mixer path, so it would measure the wrong coded tail.
- Counting already3-bit frame embedding/block0 FiLM as fresh int4-to-int3 savings: the shipping body disproves that starting premise.

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44) unchanged.
