# ddm_ntb2 — non-tail lossy levers on move44

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false.

**IN PROGRESS.** The landed RLC1 encoder reproduction is COMPLETE. Structural HPAC
prices, renderer precision/re-solve measurements, and composition are not complete.
No frontier move, seal, public cold n600 render, scorer measurement, or Modal dispatch
has been claimed. This file is a crash-resumable interim record, not a completion receipt.

## Encoder reproduction — MEASURED

Axis: **[macOS-CPU advisory; exact bytes, scorer-free]**, n600, 117,964,800 symbols.
`experiments/ddm_ntb2_control.py` binds the landed RLC1 triple encoder used by
`ddm_rlc4_rebase.py` and `ddm_rlc5_run.py`; no arithmetic or mixing algorithm was
rewritten. It copies and verifies all 600 retained pre-mixer trace frames, then
re-encodes from frame zero. Each completed frame has an immutable restart state.

| Object | Bytes | SHA-256 | Result |
|---|---:|---|---|
| RLC1 arithmetic stream, twin0 | 119,749 | c499972a33dac497412c18839b632a8e1bbf75d518528039db0e2aca80c8eb13 | shipped stream identical |
| RLC1 arithmetic stream, twin1 | 119,749 | c499972a33dac497412c18839b632a8e1bbf75d518528039db0e2aca80c8eb13 | shipped stream identical |
| full archive, both twins | 180,406 | 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e | shipped archive identical |
| inherited third encoder, pre-rider move43 control | 119,833 | 4b8fc5997982577eb2749567eec48aa4c9ca820af7961135898950af4fb89e30 | own historical control identical |

The third stream is explicitly NOT move44's stream. The source rider comparison and
both full-archive comparisons are the move44 falsifiers. Encoder-loop elapsed time
was 245.04 seconds; input copying is excluded from that time. This is not a decode
wall-clock authority receipt.

Custody: `/Volumes/VertigoDataTier/pact/ddm_ntb2_non_tail_lossy_levers/control/REPRODUCTION.json`.
Small projection: `.omx/research/ddm_ntb2_20260911/ENCODER_REPRODUCTION.json`.
Launcher receipt and full stdout remain under `launch_control/` in the same SSD store.

## HPAC structural table — not yet measured

The fresh producer is `experiments/ddm_ntb2_hpac.py`. Its control recomputes the
causal probabilities through the actual shipping receiver loop. Only the arithmetic
decode call is substituted with the landed native arithmetic ENCODER plus known
source symbols. Shipping RLC1 coding/observe calls and corrector updates are unchanged.
This producer operation is not mislabeled a public decode; its dummy decoder bit
position is not reported as evidence. Separate public decoding is required for a
candidate identity proof.

| Treatment | Counted coordinates | HPAC B | Tail B | Joint delta B | Status |
|---|---|---:|---:|---:|---|
| control | unchanged | — | — | — | live causal-loop control in progress |
| drop_row1 | lowest mean-absolute nonzero integer-weight row; stable row-index ties | — | — | — | not launched |
| drop_row4 | four lowest such rows | — | — | — | not launched |
| frame_even | 600×8 stored int8 frame embedding rounded to even integer values | — | — | — | not launched |

No negative verdict or gain attaches to any unmeasured cell. Actual layer and row
indices are written into each treatment's INPUTS.json before its encode. Dropping a
row means zeroing its fixed-schema values and recording depth zero; it does not
change model dimensions, masks, or receiver code. The frame treatment changes counted
tail values inside the HPAC model, not free interpreter constants.

Two early producer errors were caught before any causal treatment encode:

- IHS1's parser returns int64 depths; in-place OR into a uint8 packing buffer raised
  a NumPy casting error. The fix checks the nibble domain and explicitly converts it
  to uint8. No output price existed. Both review passes were renewed.
- RC3's q10/window24 control had the correct 11,911-byte length and exactly the same
  decompressed rider but different compressed bytes. The original RC3 landing's
  retained container receipt identifies q10/window22. The corrected generation uses
  that recipe and retains both failed generations. This is an encoder-recipe correction,
  not a new lossless sweep or evidence of a byte gain.

Active generation: `/Volumes/VertigoDataTier/pact/ddm_ntb2_non_tail_lossy_levers/hpac_v3/`.
Failed attempts remain under `hpac/`, `launch_hpac_control/`, and
`launch_hpac_control_v2/`. Diagnostics are in `ddm_ntb2_20260911/DEPTH_FIX.json`,
`HPAC_CONTAINER_DIAGNOSIS.json`, and `RC3_RECIPE.json`.

## Renderer precision table — the blocker resolved, and the lever MEASURED

### The token-identity blocker, resolved (Opus continuation, 2026-09-11)

The dead arm stopped on a real conflict: the charter's deliverable 2 names "the sj1 seg
re-solve chain", and sj1's actual operation is token PRE-DISTORTION, which changes token
labels; deliverable 3 says the token planes must be identical unless a model change
recoded them losslessly. The resolution taken here needs no clarification because it is
forced by the order of the work, not by a preference:

**The renderer lever is measured FIRST with the token plane FIXED at move 44's own field.**
A re-solve is a REPAIR of damage; its size is only knowable once the damage is known, and
the damage at fixed tokens is the quantity both deliverables actually want. If the fixed-token
damage lands inside the byte gain, no re-solve is needed at all; if it lands far outside,
no re-solve can close it and the token question never arises. That is exactly what happened,
so this arm changed no token and the token planes are byte-identical to move 44 throughout.

### The instrument

`experiments/ddm_ntb2_renderer_score.py` (committed) renders frame `2p+1` for all 600 pairs
through `jg1.render_frame1` — the receiver's own forward model at batch 1 — from move 44's
own token field (`control/trace_input/field.u8`, the same field whose re-encode reproduced
the shipped tail byte-identically), segments it with the frozen CPU SegNet through the
evaluator's preprocess, and poses it against the shipped frame-0 carrier (`up2.render_frame0`,
loaded from move 44's archive in EVERY treatment because this lever does not touch the carrier
section) with the frozen CPU PoseNet. GT lineage DALI for both legs. Axis
**[macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]** — absolute levels here are NOT the
contest-CUDA row's and are never quoted as one; the control run of this same producer on move
44's own archive is the reference.

Exchange rates at move 44's operating point (DERIVED from the T4 row
`avg_segnet_dist 0.00010345`, `avg_posenet_dist 4.59e-06`, `180,406 B`): one archive byte is
6.658589531221714e-07 S; 1e-6 of d_seg costs 150.18 B; 1e-6 of d_pose costs 1,108.36 B. So the
largest single-layer byte gain on this lever, 1,409 B, buys a budget of **9.38e-6 d_seg or
1.27e-6 d_pose** — against a live d_pose of only 4.59e-6 in total.

### n600 rows — MEASURED

| Treatment | Archive B | ΔB | pairs measured | d_seg | d_pose | verdict |
|---|---:|---:|---:|---|---|---|
| control (move 44's own archive) | 180,406 | 0 | in flight | — | — | reference |
| coord_mix.weight → 3-bit | 178,997 | −1,409 | 20 of 600 | ≥ 2.4375e-4 over n600 | ≥ 0.30837 over n600 | **REFUTED** |
| blocks.1.pw.weight → 3-bit | 179,016 | −1,390 | in flight | — | — | pose bound already binding |
| blocks.2.pw.weight → 3-bit | 179,020 | −1,386 | in flight | — | — | family best case |

The coord_mix bound is a LOWER BOUND, not a prefix estimate, and it is valid for that reason:
d is the mean of 600 non-negative per-pair values, so any partial sum divided by 600 is a floor
under the whole. Its pose leg floor is `sqrt(10 x 0.30837) = 1.756` against move 44's 0.006775 —
the cut costs at least **+1.749 S** to buy 0.000938 S of rate, about 1,860x its own gain. The run
was stopped at that bound rather than spending 40 more minutes to refine a number already three
orders of magnitude past the decision.

### Why no scale rule rescues it — MEASURED, weight space

`experiments/ddm_ntb2_quantizer_headroom.py` (committed) separates the two candidate causes.
Relative weight error against the shipped values, per tensor: re-quantizing the shipped weights
at 4-bit max-absolute is an EXACT identity (`rel_err 0.0`) — which independently confirms both
that the shipped section IS the per-axis max-absolute q4 representation and that
`frame_embed.weight` and `blocks.0.film.weight` are already 3-bit (their q3 error is 0.0 and
their q4 error is not). At 3 bits the max-absolute error is 0.358 (coord_mix), 0.254–0.265 (the
four pw), 0.136–0.146 (the four dw), 0.412 (head). Choosing each row's clip ratio to MINIMISE its
squared error — the best any scale-rule change can do at three bits without touching the receiver
format or refitting a weight — improves those by only **1.01x to 2.21x** (head 2.21x, coord_mix
1.54x, pw 1.22–1.31x). A 1.54x weight-error improvement cannot cross a 1,860x score gap. **The
3-bit damage is the bit budget, not the scale rule.** (Declared discrepancy: the probe's best-clip
column uses the per-row axis while `quantized_components` uses the per-column axis for
`*embed.weight`, so that column is not apples-to-apples for the two embedding tensors, which are
worth 16 B and 0 B on this lever.)

### Why no layer is benign — MEASURED, render space, SCREEN ONLY

`experiments/ddm_ntb2_renderer_screen.py` (committed) renders 4 seeded-RANDOM pairs (never a
prefix) through each retained packet's own archive and reports the pixel distance from the control
render of the same pairs. This is an ORDERING, not a verdict.

| Layer | ΔB | mean abs pixel delta | max | fraction of pixels changed |
|---|---:|---:|---:|---:|
| blocks.2.film.weight | **+17** | 0.0014 | 1 | 0.14% |
| blocks.3.film.weight | **+15** | 0.0078 | 1 | 0.78% |
| blocks.1.film.weight | **+1** | 0.0430 | 2 | 4.30% |
| blocks.3.dw.weight | −110 | 3.2277 | 39 | 87.23% |
| blocks.2.dw.weight | −72 | 3.9015 | 47 | 88.89% |
| blocks.2.pw.weight | −1,386 | 4.4937 | 74 | 90.96% |
| blocks.3.pw.weight | −1,352 | 6.4992 | 60 | 93.77% |
| blocks.1.pw.weight | −1,390 | 7.7569 | 109 | 94.88% |
| blocks.0.dw.weight | −85 | 8.5665 | 79 | 95.18% |
| blocks.0.pw.weight | −1,337 | 11.4508 | 110 | 95.88% |
| blocks.1.dw.weight | −77 | 12.6823 | 90 | 97.63% |

The shape is stark and it is the whole story of this lever: **every layer whose 3-bit cut SAVES
bytes moves 87–98% of the frame by 3–13 grey levels, and the only layers whose render barely moves
are the three late FiLM tensors, every one of which GROWS the archive.** There is no benign
byte-saving cut to find. The best byte-per-damage member is `blocks.2.pw.weight` (−1,386 B at 4.49
mean levels), which is why it is being measured at n600: a family is refuted at its best member.

### Verdict scope

This closes **3-bit at the shipped per-axis max-absolute quantizer with the token plane fixed and
no weight refit**, on move 44's body, at n600, on this arm's instrument. It does NOT close a
quantization-AWARE refit (training the remaining weights to compensate), a per-row mixed depth (a
receiver format change, therefore out of this charter), or any non-uniform codebook the receiver
does not already parse. Those are named, not dismissed.

## Composition and boundaries

No measured winning treatment exists yet; no composed candidate bytes, SHA, projected S,
or seal exist. No receiver files, upstream files, PR trees, sealed source trees, or
obx2/pc3/mxo1/gpp1/rbf1 directories were edited. No Modal. No scorer ran. All payloads
created so far are retained on the SSD. Source hashes and per-stage checkpoints bind
the outputs. Bulk writes refuse below a 40 GiB reserve; no payload deletion is authorized.
Process-list inspection (`ps`) was denied by the sandbox; no fleet-idle claim is made.

## RECALL EVIDENCE

Searched the full memo/receipt corpus by content, the canonical equations command,
design docs, task ledger, and research index/DAG surfaces. Exact commands/queries and
captured outputs: `.omx/research/ddm_ntb2_20260911/RECALL_SEARCHES.json`, `recall_0.txt`,
`recall_1.txt`, `recall_2.txt`, `equations_recall.json`, `EQUATIONS_RELEVANT.json`,
and `index_recall.txt`. Queries included `HPAC.*prun|structural.*HPAC|renderer.*precision|
renderer.*3.bit|model.*joint.*tail`, `joint_model_tail|section_coding_axis_closure|
renderer_edge_layer|model_section_edit_container`, and `HPAC|prun|renderer.*precision|non.tail`.
The raw content search included historical copies; claims below were checked against
the named actual source memos and implementations, not inferred from duplicate hits.

Findings beyond charter seeds that changed execution:

- `ddm_na12_post_sy2_negative_regrade_20260823.md` distinguishes MP3's free corrector
  families from the indivisible counted neural HPAC section. Its 34-byte historical
  pruning result does not price the structural neural coordinates tested here.
- `ddm_cl3c_closer_20260908.md` and `hpac_prior_capacity_slope_v1` scope the negative
  multiplier/seed results to their historical object and coordinates. They do not
  establish a structural-coordinate closure; no multiplier or seed rerun was added.
- `ddm_rc3_shared_mixer_successor_closed_form_gated_20260909.md` and its retained
  `CONTAINER_SWEEP.json` identify the exact q10/window22 container recipe. Equal
  byte count at window24 is insufficient for byte-identical reproduction.
- `model_section_edit_container_break_fee_v1` explicitly EXCLUDES measured HPAC
  fee evidence; transferring its semantic-section fee into these HPAC prices would
  be a hypothesis. Real whole-container twin encodes remain necessary.
- `renderer_edge_layer_foldback_reach_v1` excludes finer grids, fp16 scale edits,
  and row-pruned late FiLM changes from its prior negative. It does not close 3-bit
  precision cuts with real re-solve. The current SM1 rider differs from rw1's RC1
  rider, so rw1's old loader cannot be reused unchanged for current model pricing.
- sj1's implementation establishes the token-identity/re-solve conflict above.

Pins verified: ntb1 memo `58b1163d5a46e54683e63aa2c343651ddbd1695c4f8febe4ebf996bc15f2297a`;
ntb1 charter `161d759e192e4fd96af011e6011ec42787d16c41758a3020e94d7c007f74bf99`;
move44 encode receipt `17dded18372053ac869c88be204294404e16953e41b11795ab846f0d95f2e9dd`.
The live pointer/archive pin is checked again on each new producer launch.

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44) unchanged.
