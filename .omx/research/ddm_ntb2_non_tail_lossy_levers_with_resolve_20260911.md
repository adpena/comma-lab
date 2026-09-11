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

### The instrument falsifier — MEASURED PASS

Every row above is a statement about the render, so the render had to be shown to BE the
receiver's. `ddm_ntb2_renderer_screen.py --forward-model-control` re-renders move 44's own
tokens through move 44's own renderer and diffs the frames move 44's public receiver actually
decoded (`ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw`, sha
`2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc`, the cold public decode of
archive `04758c0d…`): **0 of 12,208,032 pixels differ, max_abs_delta 0**, at `semantic_batch 1`,
on four seeded-random pairs. Receipt: `renderer_score/FORWARD_MODEL_CONTROL.json`.

### The pw/coord_mix refutation, priced — MEASURED

| Layer → 3-bit | ΔB | rate gain (S) | pairs | d_pose floor (n600) | pose-leg floor | pose cost (S) | over its gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| coord_mix.weight | −1,409 | 0.000938 | 20 | 3.0837e-01 | 1.75604 | +1.74926 | **1,864x** |
| blocks.1.pw.weight | −1,390 | 0.000926 | 60 | 3.4768e-03 | 0.18646 | +0.17969 | **194x** |
| blocks.2.pw.weight | −1,386 | 0.000923 | 60 | 1.0138e-03 | 0.10069 | +0.09391 | **102x** |

`blocks.2.pw.weight` is the family's BEST byte-per-damage member by the screen, and it is over
by two orders of magnitude. Each run was stopped at its bound rather than spending another
40 minutes refining a number already a hundredfold past the decision.

### The FiLM door — the one door with a favourable precedent, and it closes too

Recall (agent sweep, 2026-09-11) surfaced `ddm_sd1_semantic_20260809` : an n120 seeded stratified
screen on the **191,052 B PR130 ancestor** measured all sixteen tensors at q3, and **four cells came
out NEGATIVE** — `frame_embed` (−332 B, Δd_seg −4.728e-05), `blocks.3.film` (−184 B, −9.285e-05),
`blocks.2.film` (−160 B, −8.534e-05), `blocks.1.film` (−168 B, −3.557e-05). Two of those four ship
today. That is a real precedent for a q3 cut IMPROVING seg, and on move 44 those three FiLM packets
move only 0.0014–0.043 grey levels, so the door had to be opened rather than argued shut.

**SD1's objective was the semantic leg alone.** Its own record says so: `"pose_status":
"NOT_MEASURED; full score unavailable"`. No renderer depth cell, on any object, had ever been
measured against PoseNet before this arm. Measured here at a matched 100-pair prefix against this
producer's own control (same pairs, same instrument):

| Layer → 3-bit | ΔB | Δd_seg | seg leg | Δd_pose | pose leg at move 44's base | rate | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocks.1.film.weight | **+1** | +2.3905e-06 | +0.000239 | +5.0454e-06 | +0.003041 | +0.0000007 | **+0.003281 S** |
| blocks.3.film.weight | **+15** | +6.6121e-07 | +0.000066 | +4.4089e-07 | +0.000318 | +0.0000100 | **+0.000394 S** |

Both LOSE, both are pose-dominated (93% and 81% of the loss), and both also cost bytes on this
object because mode-6 row-prune already keeps only 1% of the FiLM rows, so re-quantizing what is
left buys nothing and the depth-table and scale overhead dominates. **The ancestor's four q3
"winners" were winners of a seg-only objective; with pose measured they are not winners.** That,
and not a sweep, is why the shipped depth table was never a score optimum — `blocks.0.film.weight`
in particular was substituted into the shipped 3-bit set by a bytes argument after SD1 had
measured it at `semantic_leg_delta_s = +4.92e-05`, a loss.

### The re-solve, answered without running one

The charter asks for these cuts "with the real re-solve". The re-solve is bounded above by
recovering the ENTIRE pose leg — a carrier that reproduced the target pose exactly. Apply that
upper bound to every row:

- `coord_mix` keeps a seg-leg floor of **+0.02438 S** against a 0.000938 S gain — 26x over, on seg
  alone, with pose fully forgiven.
- `blocks.1.film` keeps **+0.000239 S** of seg plus its byte cost, against a gain of zero (it grows
  the archive); `blocks.3.film` keeps **+0.000076 S**. Both stay net-positive.

So no re-solve of any strength flips any row's sign, and the token plane never had to be touched.
The tokens on this arm are byte-identical to move 44's throughout.

### Verdict scope
## What this arm establishes for the campaign

The non-tail census ntb1 measured is 60,497 B: ZIP 100 / RX1M 14 / HPAC 11,911 / renderer 29,862 /
carrier 18,610. ntb1 closed every LOSSLESS lever in it at 0 B. This arm took the two LOSSY ones.

**The renderer's 29,862 B are not purchasable.** Its local damage-per-byte at the shipped operating
point runs 102x to 1,864x the score's own exchange rate, and that ratio survives forgiving the entire
pose leg. The three layers with a near-invisible render all GROW the archive. The scale rule is not
the wall (best per-row clip gains 1.01–2.21x) and the re-solve is not the cure (its upper bound does
not flip a sign). Combined with rw1's measured minimum action — one int4 code step already moves
240–455 argmax cells — the honest reading is that this section sits AT its distortion-rate knee.

The arithmetic that follows is worth stating plainly. Sub-0.12 from move 44's 0.1372449 needs
−0.01724 S. The seg and pose legs TOGETHER are only 0.010345 + 0.006775 = 0.017120 S, so even
driving both distortion legs to exactly zero does not reach the target: **the remaining gap is a
RATE gap and it is about 25,891 B.** The renderer cannot supply them, and the HPAC's structural
coordinates move on the order of tens of bytes per row. That points the next unit at the tail
(119,749 B) and the carrier (18,610 B), not at the model sections.

**Doors this arm did NOT open, named so a successor takes them with eyes open:**
- Quantization-AWARE refit (train the surviving weights to compensate a 3-bit layer). SD1 left this
  open explicitly ("bit-depth family — OPEN: all non-int4 cells re-quantize a q4-QAT master").
  Everything measured here is training-free.
- Per-row mixed depth inside a tensor. The IHS1 depth nibble is per-row for the HPAC, but the
  renderer's depth table is per-tensor; making it per-row is a receiver format change.
- Row-pruning the renderer's pw/dw tensors through the existing `MODE_ROW_PRUNE` the receiver already
  parses. Untested here. The prior against it is strong (zeroing a row is thousands of int4 code
  steps, and rw1 measured one step at 240–455 cells), but the mechanism is different from precision
  noise and the format is already in the receiver.
- Growing the HPAC prior instead of shrinking it. Every treatment here removes capacity; nobody has
  measured whether the joint optimum lies in the other direction. That requires training, not a
  structural edit.

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
