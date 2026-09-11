# ddm_ntb2 — non-tail lossy levers on move44

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false.

**STATE (Opus continuation, 2026-09-11 09:35).** The landed RLC1 encoder reproduction is COMPLETE
and byte-identical. **Lever 2, the renderer precision cuts, is COMPLETE and closes NEGATIVE on every
door** — with the first PoseNet measurement any renderer depth cell has ever received, on any object.
**Lever 1, the structural HPAC prunings, HAS A WINNER**: `frame_even` at a joint −245 B, output-lossless,
projected S 0.1370817687278253. Its cold public parse-back is running; until that returns
byte-identical raw there is no seal. No frontier move, no Modal dispatch, and no score claim. This
file is a crash-resumable record.

Also fixed here: the "re-solve blocked pending a clarification" state the codex arm left. It is
resolved in §Renderer, by ordering rather than by asking — and the resolution is that no token was
touched and none had to be.

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

## HPAC structural table — in flight

**The HPAC lever is OUTPUT-lossless, and that is the whole reason it is the cleaner of the two.**
The HPAC section is the arithmetic coder's PRIOR. Encoder and decoder both build it from the same
shipped bytes, so pruning a coordinate changes the code LENGTHS and never the decoded symbols: the
token stream parses back identically, the render is bit-identical, and d_seg and d_pose are unchanged
by construction rather than by measurement. A treatment is therefore priced entirely by one number —
(HPAC bytes + re-encoded tail bytes) — and a negative joint delta is a pure rate win needing no
scorer, no re-solve and no distortion budget at all. This is why the tail MUST be re-encoded with
the RLC1-aware encoder for every treatment, and why reproducing the shipped tail byte-identically
first (above) was the precondition and not a formality.


The fresh producer is `experiments/ddm_ntb2_hpac.py`. Its control recomputes the
causal probabilities through the actual shipping receiver loop. Only the arithmetic
decode call is substituted with the landed native arithmetic ENCODER plus known
source symbols. Shipping RLC1 coding/observe calls and corrector updates are unchanged.
This producer operation is not mislabeled a public decode; its dummy decoder bit
position is not reported as evidence. Separate public decoding is required for a
candidate identity proof.

| Treatment | Counted coordinates | HPAC B | ΔHPAC | Tail B | ΔTail | **Joint ΔB** | ΔS |
|---|---|---:|---:|---:|---:|---:|---:|
| control | unchanged | 11,911 | 0 | 119,749 | 0 | **0** | byte-identical to move 44 |
| drop_row1 | `frame_scale` row 13 (8 values, mean abs 0.125) | 11,934 | **+23** | 119,744 | −5 | **+18** | +1.199e-05 |
| drop_row4 | + `conv_b2` rows 45 and 61 (5 each), `spm_pw` row 13 (64) | 11,941 | **+30** | 119,745 | −4 | **+26** | +1.731e-05 |
| frame_even | `frame_embed.weight`: 2,320 of 4,800 stored int8 values to the nearest even | 11,308 | **−603** | 120,107 | +358 | **−245** | **−1.63135e-04** |

Both twins agree for every treatment. Archive shas: drop_row1 `b1df64e9…` 180,424 B; drop_row4
`906938ba…` 180,432 B; **frame_even `432e8f09853a907665d87d44f3ed5eda6c4eebc5783ac23e5452d486371d4db6`
180,161 B**.

**frame_even WINS by 245 B, and because the lever is output-lossless that is the entire price:
projected S = 0.1372449041713402 − 0.000163135443514932 = 0.1370817687278253**, with d_seg and
d_pose unchanged by construction rather than by measurement. The parse-back below is what turns
"by construction" into "by receipt".

The two row-drops are the more interesting negative. Zeroing a row and recording depth zero made the
tail very slightly BETTER (−5 B, −4 B: the prior barely uses those coordinates) and the HPAC section
BIGGER (+23 B, +30 B). A section that holds strictly less information got larger, because the depth
table and the packed-row layout moved and brotli liked the new arrangement less. That is the
container-break fee arriving on the side nobody prices, and it is why every cell here is a real twin
encode and not a ledger sum.

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

### The control reproduces the contest-CUDA row to 0.06 % — MEASURED, and it is the arm's most reusable result

The control is move 44's own archive through this producer, all 600 pairs. It was run to establish a
reference for the deltas. It did something better than that:

| leg | this instrument, n600 | move 44, contest-CUDA T4 n600 | ratio |
|---|---|---|---:|
| d_seg | 0.00010338677300347222 | 0.00010345 | 0.9994 |
| d_pose | 4.586732867899564e-06 | 4.59e-06 | 0.9993 |
| S recomputed | 0.137236169 | 0.1372449041713402 | Δ 8.7e-06 |

A macOS-CPU instrument, on the DALI GT lineage, reproduces BOTH contest-CUDA distortion legs to
within 0.07 % and the whole score to 8.7e-06 S, in 2,397 s. Two consequences. First, every delta in
the tables below is measured on an instrument that tracks the authority to about a part in 1,500, so
the refutations are not resting on an axis gap. Second — and this outlives this arm — **the
distortion half of a candidate can be screened locally at full n600 before any Modal byte is spent.**
The honest caveat is that agreement at ONE point does not prove agreement on every delta; it is
strong evidence, not a licence to quote a local number as a score.

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
measured against PoseNet before this arm. Measured here at **full n600** against this producer's own
control (same pairs, same instrument, same GT), receipt
`renderer_score/N600_SUMMARY.json`:

| Layer → 3-bit | ΔB | Δd_seg | seg leg | Δd_pose | pose leg at move 44's base | rate | **total** | with pose fully forgiven |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blocks.1.film.weight | **+1** | +2.7805e-06 | +0.000278 | +8.9354e-06 | +0.004855 | +0.0000007 | **+0.005134 S** | +0.000279 S |
| blocks.3.film.weight | **+15** | +8.9010e-07 | +0.000089 | +4.9322e-07 | +0.000355 | +0.0000100 | **+0.000454 S** | +0.000099 S |

Archive shas `28966cd0…` (180,407 B) and `14d6ac77…` (180,421 B). Both LOSE, both are
pose-dominated (95 % and 78 % of the loss), and both also COST bytes on this object, because
mode-6 row-prune already keeps only 1 % of the FiLM rows: re-quantizing what is left buys
nothing and the depth-table and scale overhead dominates. **The ancestor's four q3 "winners"
were winners of a seg-only objective; with pose measured they are not winners.** That, and not
a sweep, is why the shipped depth table was never a score optimum — `blocks.0.film.weight` in
particular was substituted into the shipped 3-bit set by a bytes argument after SD1 had already
measured it at `semantic_leg_delta_s = +4.92e-05`, a loss.

### The re-solve, answered without running one

The charter asks for these cuts "with the real re-solve". The re-solve is bounded above by
recovering the ENTIRE pose leg — a carrier that reproduced the target pose exactly. Apply that
upper bound to every row:

- `coord_mix` keeps a seg-leg floor of **+0.02438 S** against a 0.000938 S gain — 26x over, on seg
  alone, with pose fully forgiven.
- `blocks.1.film` keeps **+0.000279 S** and `blocks.3.film` **+0.000099 S** at full n600, against a
  rate gain of zero — both GROW the archive. Both stay net-positive.

So no re-solve of any strength flips any row's sign, and the token plane never had to be touched.
The tokens on this arm are byte-identical to move 44's throughout.

### Verdict scope
## The candidate, REBASED onto move 45

The pointer moved while this arm was proving its move-44 candidate: **move 45**, commit `01f2b66ad`,
S 0.1371383667388406 @ 180,246 B, archive `145e02e2…`, lane `ddm_pc3_cap1_predictor_refit…`. A seal
pinned to a stale pointer refuses, so the move-44 candidate (180,161 B, sha `432e8f09…`) is RETIRED
with its reason and all its receipts kept
(`.omx/research/ddm_ntb2_20260911/RETIRED_MOVE44_CANDIDATE.json`). No seal had been run against it.

**The rebase needed no re-encode, and that is a measured claim rather than a convenience.**
`experiments/ddm_ntb2_rebase45.py` checks the premise before it builds anything:

| member | move 44 | move 45 | identical |
|---|---:|---:|---:|
| `hpac` | 11,911 | 11,911 | **yes** |
| `tail` | 119,909 | 119,909 | **yes** |
| `semantic` | 29,862 | 29,862 | **yes** |
| `carrier` | 18,610 | 18,450 | no — pc3's −160 B |
| `header` | 14 | 14 | no — section lengths |

The 600-frame RLC1 re-encode this arm ran consumed the token field and the HPAC prior, neither of
which moved, so its retained `hpac.twin*.br` and `tail.twin*.rider` payloads are still the right
bytes and are reused. The falsifier is the control: reassembling move 45's OWN members reproduces
`145e02e2…` at 180,246 B in both twins — **PASSED** — and only then is a treatment archive admitted.

**Rebased candidate `ntb2_frame_even`: 180,001 B, sha
`a0de607df2ff566d4eb6fc43031179f5450f0cb622803d4ce71b987f8970d720`, both twins agreeing, −245 B
against move 45, ΔS −1.63135443514932e-04, projected S 0.13697523129532568** — 8.2× the −2e-5 admit
bar, at unchanged distortion. `semantic`, `carrier`, `tc1_weights` and `residual_payload` are
asserted unchanged against move 45 by the archive parser.

Staged faithfully from move 45's promoted tree: 51 shipped files, behavior digest EQUAL at
`9f6e7168…` (unchanged across moves 44 and 45, since pc3 moved the archive and not receiver code),
legacy digest differing by the archive pin alone. Move 45's retained raw is `2b762eba…` — byte-
identical to move 44's, read from pc3's own `RESULT.json`, which is exactly what "the carrier member
moved and the frames did not" means. The cold n600 parse-back against it is running on APDataStore.

## The seal is BLOCKED, and not by anything this candidate did

pr18 clears the receiver row (below), but a second row does not clear, and it is worth reporting
carefully because **it blocks every move-46 candidate, pc3's rebase included, not just this one.**

`make_candidate_seal.py` passes the LIVE pointer's `pointer_archive_sha256` into
`inherit_decode_wall_clock`. The inheritance chain is:

| move | archive | leg mode |
|---|---|---|
| 44 | `04758c0d…` | **`t4_direct`** — a real cold T4 decode, 1232.418725255 s |
| 45 | `145e02e2…` | `inherited`, from move 44's t4_direct |
| 46 | `a0de607d…` (this candidate) | none available |

Two doors, both measured rather than argued:

- **Move 45's leg as the source: forbidden by contract text** — "inheritance must point directly to a
  measured or t4_direct leg (never to an inherited one)".
- **Move 44's t4_direct leg as the source: REFUSED** — `source measurement is not the pointer
  archive`. The control that makes that refusal legible: the SAME leg with
  `pointer_archive_sha256` set to move 44 **PASSES** at 1232.418725255 s. So the refusal is about the
  POINTER, not about this candidate's tree.

No `t4_direct` leg exists anywhere for archive `145e02e2…`. Move 45 sealed validly because the
pointer was then move 44, so its source leg WAS the pointer archive. The contract tolerates
inheriting timing measured on a different archive once; it does not tolerate doing it twice, and
that is a defensible place to stop rather than a bug to route around.

Three resolutions, all MAIN's to choose and none taken here: fire a real `t4_direct` decode for move
45 and restore a measured anchor; seal with `--first-fire-intent` and complete it from the fired
measurement; or amend the contract to follow the inheritance CHAIN to its terminating measurement
when every link shares one receiver behavior digest — a contract change, therefore a second family's
call. Receipt: `.omx/research/ddm_ntb2_20260911/TIMING_INHERITANCE_CHAIN_BREAK.json`. Nothing was
patched.

## pr18 is LIVE on main, and it clears this candidate — a second family confirms the rule

MAIN said a pr18 arm was landing the versioned receiver digest and to expect a refusal until it did.
It is already on `main`, and this arm is the second family to exercise it:

`tac.decode_wall_clock.RECEIVER_DERIVED_LISTING == "MANIFEST.sha256"`, and `receiver_identity()`
returns both a `behavior_sha256` (the listing excluded) and a `legacy_sha256` (it included).
Measured, promoted tree vs this candidate:

| digest | promoted (move 44) | candidate | equal |
|---|---|---|---:|
| `behavior_sha256` | `9f6e7168…` | `9f6e7168…` | **YES** |
| `legacy_sha256` | `7e6da183…` | `106a9e66…` | no |

And that resolves my correction above in a satisfying way: the `9f6e7168…` I first measured on the
manifest-LESS trees is exactly the BEHAVIOR digest, because excluding `MANIFEST.sha256` from a tree
that has it gives the same answer as a tree that lacks it. My first reading was right about the
number and wrong about why.

Reading the inheritance block settles the rest without running it. In `inherited` mode the source
leg is re-validated against **its own** `runtime_dir`/`archive_path` (move 44's tree), not against
the candidate's, so the `t4_direct runtime projection` row is never asked of my tree; the candidate
is checked only for self-consistency (`candidate_t4_runtime_sha256 == measure_t4_runtime_digest`);
and the one cross-tree test is `behavior_digests_equal`, which my candidate PASSES.

**So this is a NORMAL seal with `--inherit-decode-wall-clock` and no first-measurement intent is
owed** — provided the cold parse-back returns byte-identical raw. Per MAIN, nothing in the contract
or the manifest was patched to reach this; the amendment was already there and the candidate simply
meets it.

### The exact seal command, and where the 27.6 s slack comes from

Move 44's `t4_direct` leg is
`.omx/research/ddm_rlc5_20260910/SEAL_ddm_rlc2_counted_cure_move43_rlc5_contest_cuda_v3.json.decode_wall_clock.json`
— archive `04758c0d…`, `runtime_dir` `ddm_rlc5_cure_on_move43/candidate_runtime` (present on this
host, which the inheritance needs), `projected_t4_decode_seconds` **1232.418725255** against the
policy limit of 1260 s. **That is where the charter's 27.6 s slack comes from: 1260 − 1232.419.**
It is a derived number, not a quoted one.

This candidate spends none of it. The HPAC section is 603 B SMALLER, so materializing the prior is
marginally cheaper; the arithmetic decoder performs the same 117,964,800 symbol decodes against a
slightly different prior, and the tail is 358 B longer. No new work is added at decode time, and the
inheritance scope — "identical normalized receiver code; candidate payload-dependent time not
remeasured" — is exactly the claim being made.

```
PYTHONPATH=<repo> .venv/bin/python tools/make_candidate_seal.py \
  --candidate-id ntb2_frame_even --axis contest_cuda \
  --runtime-dir /Volumes/APDataStore/pact/ddm_ntb2_public_proof/frame_even/candidate_runtime \
  --inherit-decode-wall-clock .omx/research/ddm_rlc5_20260910/SEAL_ddm_rlc2_counted_cure_move43_rlc5_contest_cuda_v3.json.decode_wall_clock.json \
  --twin-encode-receipt <hpac_v3/frame_even/PRICE.json> \
  --archive-parseback-receipt <public proof RESULT.json> \
  --raw-identity-receipt <public proof RESULT.json> \
  --candidate-manifest ... --manifest-validation ... --literal-census ... \
  --retention-manifest ... --falsifier ... --out <SEAL path>
```

## Tier pressure, and what it cost

`/Volumes/VertigoDataTier` reached 40 GiB — exactly the reserve the HPAC producers refuse below —
while the 3.66 GB cold parse-back was writing. Two things followed, and both are the guards working
rather than failing:

- **`frame_quad` died at frame 350 with `STORAGE_BLOCK at durable prior checkpoint`.** It stopped AT
  a checkpoint with nothing corrupted and resumes from `stage_0350`. Per MAIN's sequencing it is
  therefore a follow-on row, not a candidate for this seal: it can no longer beat `frame_even`
  before the parse-back finishes. Its question stands open — step 2 bought −603 HPAC B for +358 tail
  B, and 3,548 of 4,800 values move at step 4 against 2,320 at step 2.
- **The parse-back survived**, because the receipt-reserve fix landed earlier this unit: a kilobyte
  receipt written after a multi-hour decode is no longer gated by the 40 GiB reserve that gates the
  decode itself. Without that fix this unit would have lost the proof to save 3 KB.

708.2 MB of renderer-score resume checkpoints were CERTIFIED AND MOVED to
`/Volumes/APDataStore/pact/ddm_ntb2_coldstore/` — original path, bytes, sha256, destination, rebuild
command and rebuildable-reason per file, in `.omx/research/ddm_ntb2_20260911/COLDSTORE_MANIFEST.json`
and beside the bytes. **Nothing was deleted.**

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

**Candidate:** `ntb2_frame_even`, 180,161 B, sha `432e8f09…`, joint −245 B, ΔS −1.63135e-04,
projected S 0.1370817687278253, distortion unchanged (prior-only change). NOT sealed: the cold
public parse-back is still running, and nothing is sealed or claimed before its
`raw_byte_identical_to_move44` returns true. No Modal dispatch. No scorer ran on the candidate and
none needs to.

**Receiver:** UNCHANGED, measured — `measure_receiver_digest` of the staged tree equals move 44's,
so the seal is a NORMAL seal inheriting `t4_direct`. Only `inflate.py`'s two archive-pin constants
differ, and that function normalizes exactly those.

**Boundaries this unit declares:**
- Every renderer number is `[macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]`. The control
  agreeing with the T4 row to 0.07 % on both legs is strong evidence for the deltas, not a licence
  to quote a local number as a score.
- The renderer closure is scoped to **3-bit at the shipped per-axis max-absolute quantizer, fixed
  tokens, no weight refit**. A quantization-aware refit is NOT closed.
- `coord_mix`, `blocks.1.pw`, `blocks.2.pw` were stopped at 20/60/60 pairs. Those rows are LOWER
  BOUNDS (a mean of 600 non-negative terms is bounded below by any partial sum over 600), never
  prefix estimates, and they are quoted only as bounds.
- The three renderer treatments' argmax planes, pose vectors and per-chunk render shas are retained;
  their full 1.83 GB frame-1 rasters are NOT, because the tier has under 11 GiB above its reserve.
  They are exactly rebuildable from the retained archive plus the retained token field by the same
  producer, and the per-chunk shas certify that rebuild.
- **Tier pressure is live**: `/Volumes/VertigoDataTier` fell from 51 GiB to 41 GiB during this unit
  while five arms ran. The 40 GiB reserve in the HPAC producers is a fail-CLOSED guard, so a
  `frame_quad` or a future encode may refuse rather than corrupt; it resumes from its stage
  checkpoints. `ddm_rbf1` holds 18 G and `ddm_obx2` 4.1 G; this arm holds 3.1 G and deleted nothing.
- Rows 0–20 of the renderer control were produced by producer sha `da5714d2…` and the rest by later
  shas. The edits between them touch only the `--retain-frames` branch (never taken by any of these
  runs) and receipt fields; no measured quantity is on a changed path.
- Three parse-back attempts are retained, including the two that failed. Nothing was deleted; the
  superseded INPUTS and the half-done receiver checkpoints are under `public/frame_even/superseded/`.

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

Second recall pass (Opus continuation, 2026-09-11), which materially changed execution:

- `ddm_sd1_semantic_20260809/SD1_FINDINGS.md` + its retained `cpu_screen/results.json` hold a
  COMPLETE single-tensor q3/q5 sweep of all sixteen renderer tensors on the 191,052 B PR130
  ancestor — an n120 seeded stratified screen against the frozen CPU SegNet with real rebuilt
  archive bytes. Every layer that is 4-bit today LOST there (`coord_mix` +1.03602 ΔS_sem, `head`
  +0.10136, `blocks.1.pw` +0.04807, down to `blocks.3.dw` +0.00097), and **four cells WON**, three
  of them FiLM. That is what opened the FiLM door here rather than letting the pw refutations close
  the whole lever by analogy. Its `verdict_scope` is explicit: "INSTANCE and FORMULATION … no
  public-receiver, bit-matched-QAT, capacity, **pose**, or contest-score verdict", and
  `"pose_status": "NOT_MEASURED; full score unavailable"`.
- `experiments/ddm_sm3_semantic_representation.py:87-94` shows `blocks.0.film.weight` was put into
  the shipped 3-bit set by a BYTES argument ("the surviving marginal"), not by SD1's objective — SD1
  had measured it at `+4.92e-05`, a loss. So the shipped depth table was never a score optimum, and
  saying so is not a criticism of it: nobody had the pose leg to optimize against.
- `ddm_rw1_boundary_local_renderer_weight_foldback_20260909.md` independently closes the scale-rule
  axis this arm probed in weight space: 400 realized ULP moves on the shipped per-row fp16 scales,
  with a NEGATIVE rate toll so any repair would have won, accepted **zero**. Two different
  instruments, same answer.

Pins verified: ntb1 memo `58b1163d5a46e54683e63aa2c343651ddbd1695c4f8febe4ebf996bc15f2297a`;
ntb1 charter `161d759e192e4fd96af011e6011ec42787d16c41758a3020e94d7c007f74bf99`;
move44 encode receipt `17dded18372053ac869c88be204294404e16953e41b11795ab846f0d95f2e9dd`.
The live pointer/archive pin is checked again on each new producer launch.

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44) unchanged.
