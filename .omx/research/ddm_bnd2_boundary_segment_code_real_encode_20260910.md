# ddm_bnd2 generation 2 — six charged segment recodes lose bytes

`[no-triality] [p0-ledger-ok]` · owner `ddm_bnd2` · `research_only=true` · `score_claim=false`.
Charter object: move-37 shipped field, all 600 frames. Generation 2 executes MAIN Addendum 2's
mandatory `--nice 10 --nice-best-effort` launch path; scheduling refusal is recorded, not a stop.

**MEASURED: the smallest of six lossless segment-plus-TC1 encodes is 121,200 B,
1,416 B above the shipped 119,784 B envelope and 6,416 B above the 114,784 B draw gate.**
All six primary/repeat packets, arithmetic streams and full ZIPs are byte-identical twins.
All six full-field cached-row parsebacks reproduce the shipped field. The 12-pair lossy draw
is stopped by the charter's byte gate; it is not evidence that all lossy boundary methods fail.

**Measurement status: COMPLETE.** The selected archive passed a full n600 independent causal
decode after resuming from its frame-32 disk state. The masked control also completed.
Main landing is separate; its verified outcome is recorded in the final custody handoff.

## Real n600 price table

Axis **[exact serialized bytes; scorer-free macOS-CPU]**, not a contest scorer axis.
Each row encodes all **117,964,800 token cells**. The support column counts supplied coder misses,
not a subset-sized measurement. Every other token remains in the TC1 arithmetic stream.

| Orientation | Minimum assigned run | Supplied misses | Segment packet + length B | TC1 remainder envelope B | Total envelope B | Delta B | Actual ZIP B |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 232,843 | 366,611 | 36,973 | **403,584** | +283,800 | 464,188 |
| 0 | 2 | 28,975 | 35,027 | 110,028 | **145,055** | +25,271 | 205,659 |
| 0 | 4 | 2,306 | 2,144 | 119,152 | **121,296** | +1,512 | 181,900 |
| 1 | 1 | 232,608 | 343,425 | 37,017 | **380,442** | +260,658 | 441,046 |
| 1 | 2 | 22,917 | 28,802 | 112,472 | **141,274** | +21,490 | 201,878 |
| 1 | 4 | 2,125 | 2,013 | 119,187 | **121,200** | +1,416 | 181,804 |

Every row obeys the actual serialized identity
`B_total = B_segment_packet + 4-byte length + B_TC1_remainder_envelope`.
The replacement ZIP delta equals the envelope delta exactly. The baseline raw TC1 stream is
**119,779 B**; `R6D1` plus its one padding byte makes the **119,784 B** comparator.
The 35 counted mixer weights, residual prefix, all other archive sections and ZIP overhead stay charged.
For the best row, **2,013 B** of segment/framing removes only **597 B** from the TC1 remainder.
The actual ZIP is **181,804 B**, SHA
`cab4b3914b2d761daa096cb4203898b73db20ca1763c7db329943d7273c506d4`.

At `25/37,545,489 = 6.658589531221714e-07` score units per byte, its rate term increases by
**0.0009428562776209946**. This is rate arithmetic on real ZIP bytes, not a newly measured total score.
No new Seg/Pose scorer output or full rendered video was produced. All six cached-row parsebacks
preserve token cells; the selected archive also passed the full causal receiver proof. No contest-CPU/CUDA
score is inferred from a token-identity check.
The complete table and all six archive hashes are in `ddm_bnd2_20260910/generation2_price_table.json`.

## Recovered census and source identity

Axis **[macOS-CPU scorer-free census; retained DALI target lineage]**. Selection is all 600 frames,
every post-TC1 integer-frequency argmax mismatch, without the old saving threshold. Ties use lowest
class index, matching the source coder. Every percentage below has **token-cell** denominator.

| Population | Cells / denominator | On retained GT edge | On shipped-token edge |
|---|---:|---:|---:|
| All misses | 235,044 | 229,752 (97.7485% cells) | 232,217 (98.7972% cells) |
| Previously retained | 213,733 | 209,179 (97.8693% cells) | 211,510 (98.9599% cells) |
| Recovered | 21,311 | 20,573 (96.5370% cells) | 20,707 (97.1658% cells) |

The old 213,733 locations are verified unique and an exact `(frame,pos,sym,best)` subset;
the missing **21,311** are now explicitly retained. Every frame's integer-frequency argmax was
independently recomputed from the retained coding rows. A boundary cell has an unequal in-image
four-neighbour; both incident cells count, while an image border alone does not count.
GT edge cells total **2,551,338/117,964,800 (2.162796% cells)**; shipped edge cells total
**2,564,996/117,964,800 (2.174374% cells)**.

The complete field SHA is `361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94`.
The retained DALI label-cache SHA is
`a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994`.
No RGB GT decode or scorer load was required. Census cache lineage is checked before reading labels.
The full source causal trace re-encodes the exact original envelope with two encoders:
SHA `bc046883e9c0e5f49029ee3d0918e4d3315c851b16ad3b8c481056d7a1f38ce9`, **119,784 B each**.
It resumed from its retained frame-32 complete state and finished all 600 frames.

Rows 128–191 contain **73,822/235,044 (31.4077% cells)** of coder misses and rows 192–255
contain **127,558/235,044 (54.2698% cells)**. These are coder-miss populations, not the renderer's
12,614-cell residual population. Stored-token/coder-best and GT/coder-best matrices are retained;
they are not an adjacent GT class-pair census.

**Not measured:** a new joint census against a hash-bound rendered 12,614-cell map, GT distance
histograms, or adjacent GT class-pair histograms from bnd1's broader original ITEM 1.
The shipped pass-4 **12,614** and unshipped pass-5 **12,377** remain distinct. The MAIN correction
**86.63% singleton cells versus 93.92% singleton components** is recalled pass-4 evidence, not a new
bnd2 measurement. Those figures must not be confused with this arm's assigned-run statistics.

## Grammar, addressing, and run lengths

The new grammar replaces the old global axis-aligned curve chart with **actual unit crack edges**
between unequal shipped classes. Directions E/S/W/N supply local Euclidean normals. Orientation 0
puts the lower class on the right; orientation 1 puts the higher class on the right. Each segment
has a constant ordered class pair and offset `n` in `{-1,0,+1}`: left incident cell, right incident
cell, or one more cell inward on the right. This is a declared discrete offset convention.

For each miss, the deterministic greedy assignment prefers an incident other-side class equal to
coder-best, then smaller absolute offset, then vertex/direction order. Within each class-pair/offset
bucket, maximal directed paths of the assigned edges form segments. Gaps, branches, class changes
and offset changes create new charged starts. This measures **assigned-segment lengths**, not an
intrinsic or globally optimized coherence distribution. Changing assignment can change these lengths.

Frame counts, ordered classes, normals, signed start deltas, lengths, initial directions and
previous-direction turns are all serialized. The five byte sections share one actual Brotli q11
stream with charged length and CRC. No GT graph, current-plane topology, start address, class or
exception location is supplied free. Minimum-run cutoffs 1/2/4 send rejected segments back to TC1;
all uncovered misses also remain there. The decoder obtains support solely from counted packet bytes.

Statistics below include **all assigned segments before minimum-run filtering**.

| Orientation | Eligible support / all misses | Segments | Median length | Mean length | Max | Singleton share of segments | Singleton share of eligible cells |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 232,843/235,044 (99.0636% cells) | 216,963 | 1 | 1.073192 | 14 | 93.9644% segments | 87.5560% eligible cells |
| 1 | 232,608/235,044 (98.9636% cells) | 219,920 | 1 | 1.057694 | 14 | 95.3488% segments | 90.1478% eligible cells |

| Orientation | Length cells inclusive | Segment count | Share of segments | Token cells | Share of eligible token cells |
|---|---|---:|---:|---:|---:|
| 0 | 1–1 | 203,868 | 93.9644% | 203,868 | 87.5560% |
| 0 | 2–2 | 11,110 | 5.1207% | 22,220 | 9.5429% |
| 0 | 3–3 | 1,483 | 0.6835% | 4,449 | 1.9107% |
| 0 | 4–7 | 490 | 0.2258% | 2,191 | 0.9410% |
| 0 | 8–14 | 12 | 0.0055% | 115 | 0.0494% |
| 1 | 1–1 | 209,691 | 95.3488% | 209,691 | 90.1478% |
| 1 | 2–2 | 8,497 | 3.8637% | 16,994 | 7.3059% |
| 1 | 3–3 | 1,266 | 0.5757% | 3,798 | 1.6328% |
| 1 | 4–7 | 454 | 0.2064% | 2,013 | 0.8654% |
| 1 | 8–14 | 12 | 0.0055% | 112 | 0.0481% |

`generation2/RUN_LENGTHS.json`, per-variant `GRAMMAR.json` and per-frame geometry JSON preserve
the complete distribution. True class-pair crack checks passed the random 32-frame implementation
test and all 600 independently reconstructed frames of the selected archive.

## Receiver proof and masked intervention

**PASSED n600 independent causal proof:** selected archive `generation2/price_o1_min4/archive.zip`.
Receipt `generation2/price_o1_min4/causal_decode/RESULT.json`, SHA
`3a015a6b54f12561b881826648118acebe7fdc47a33292fdf16d2fed89dae595`,
records 600 frames, `resumed_from_frame=32`, every causal row identical, and the exact shipped-field SHA.
The independent decoder recomputes HPAC logits, correction and mixer probabilities from the
candidate's own model bytes and previously reconstructed symbols. It inserts segment and residual
symbols at their original group position, observes every group in the corrector, and updates every
frame in the mixer. Cached source rows are comparison oracles only, compared after computation; they
never supply reconstruction symbols, features or state. The completed separate receipt establishes all
600 frames and the source field hash. Only this selected variant received a fresh causal decode;
all six received complete cached-row arithmetic parseback and full ZIP twin checks. Individual price
receipts preserve their earlier encode-time pending status; the final price-table proof join is explicit.

The archive is a **research BND2 packet**. A generic in-memory rider adapter lets the copied cmp2
reader parse its counted models and new rider; no live cmp2 runtime was edited. No public `inflate.sh`
release, rendered witness or exact evaluator row is claimed. Losing bytes does not authorize deployment.

**MEASURED masked control: 129,316 B per twin, +9,532 B versus the original envelope.**
Both retained envelopes have SHA `db12361512eafa1564c8b4030a1b452b889ab659b89f6d525237b52e8e21e7e3`.
The control replaces all original 235,044 miss locations with their original integer-frequency argmax
and recomputes the entire changed field's causal trajectory. An independent retained-plane audit
confirms exactly 235,044 changed cells and field SHA
`3dc7e60020b388ccfe97716b1ede0336a94ed5b1fad8be20e8bfec1a659f123d`.
The new trajectory has **250,892 coder misses**, rather than zero. All changed token planes remain
retained. This intervention's byte difference is not a detachable 83 KB substream: changing the field
also changes later probabilities. The native R6D1 envelope includes its alignment padding, matching
the native source-control convention. See `ddm_bnd2_20260910/generation2_masked_control.json`.
The minimal-cutoff segment row's 36,973 B remainder is an actual unchanged-field remainder, with
2,201 uncovered original misses still inside it; its 82,811 B saving is offset by 366,611 B of
charged geometry. Neither attribution nor a log-probability sum is being quoted as a payload price.

## Prior versus measured result and verdict scope

The prior on-edge share >=88.99% is supported by the new **97.7485%** full-population GT-edge share.
The prior median run <=3 is supported **for this assignment** by median 1 in both orientations.
The prior price >=119,784 B is supported by all six rows. Its central **+5 to +25 KB** range
**misses the optimized tested result**: +1,416 B is below that range. The unfiltered variants lose
far more than it predicted. The decisive <=114,784 B falsifier is not met.

Verdict scope: the **tested FORMULATION**—this greedy crack assignment, two orientations,
constant signed offset per segment, declared serialization and three cutoffs—loses to shipped TC1.
These are achieved code sizes (upper bounds on attainable descriptions), never entropy floors.
Richer structure can reduce combined geometry and residual cost; adding grammar does not prove
monotonic total price. No universal boundary-family impossibility or optimal global assignment is claimed.
The draw is stopped by the charter's pre-registered rule, not by a measured lossy-repair impossibility.

## RECALL EVIDENCE

Read the charter/common contract, PROGRAM, governing AGENTS/CLAUDE sections, operating handoff,
main hot state, canonical pointer, bnd1 ITEM 1–3 and RECALL EVIDENCE in full, of1:58–109,
or1:111–143, the curve-relative code/build memo, tc1's coder/receipt, the actual cmp2 reader and
`upstream/evaluate.py`. Independent recall covered research memo/receipt content, the full 480-row
canonical-equation CLI export, research index/DAG, design/SPEC files and relevant task rows.
Queries included `boundary.segment|chain.cod|crack.edge|contour.string|curve.relative`,
`normal.offset|mispredict`, `FEED-residualkit`, and `bnd1.*ITEM|gb1.*VALID_BOUND_INPUTS|ddm_gb2|ddm_bnd2`.

| Beyond-seed source | Finding | Concrete effect |
|---|---|---|
| `SPEC_ddm_qbw1_packet_schema_v1_20260827.md:17–26,82–109` | Oriented real-crack grammar with charged starts/sides/lengths already exists | Use actual class-pair crack edges, not arbitrary sparse chains |
| `ddm_ltg1_lane_topology_generator_floor_20260831.md:80–103` | n600 residual count fell 691,095 to 140,409 while full price rose 221,717 to 239,818 B | Judge complete segment plus residual price, never repaired counts alone |
| `contour_string_flip_coding_n600_20260707.md:13–15,26–45` | Ancestor paths backtracked 299,059/740,388 symbols; anchors consumed 45% | Charge every fresh start and direction; ancestor percentages do not transfer |
| `v8_increment1_design_draft_20260709.md:59–71` | Added temporal structure can lower joint description cost | Scope negatives to tested addressing; reject a universal richer-grammar loss inference |
| `ddm_na2_negative_audit_20260803.md:389–398` | Historical 0.65 B/flip gate came from an n12 prefix | Do not reuse it as a current full-population kill threshold |
| `ddm_hc2_wrong_half_flip_location_decomposition_20260905.md:99–113` | Older fs2 boundary proximity was high while incumbent already captured most indicator cost | Proximity alone is not new compressible information |
| tc1 `rebase_pc2/INPUTS.json` and row traces | Pre-mixer trace binds ancestor field `a73289e...` | Reject reuse; trace the actual `361cc6c9...` post-TC1 field |
| `ddm_gb2_generated_basis_unconditional_lattice_bound_20260909.md:332–341` and harvest contract | Bound-only formulation already vacuous; gb1 proof debt is to be folded | No duplicate Jacobian/bound request or automatic generator BUILD |

Relevant recalled laws: `token_tail_context_mixing_bound_v1`, `context_model_reorder_savings_v1`,
`v8_geometric_rate_decomposition_v1`, and `leverd_flicker_residual_reactivation_economics_v1`.
The of1 2.88-pixel arclength is a prior, not a transfer measurement. Prior 0.99x/0.90x axis-chart
prices motivated the true local-crack replacement and did not become current-field prices.

## Retention, review, reproducibility, and integration

All bulk is retained under `/Volumes/VertigoDataTier/pact/ddm_bnd2_segment_code/`.
`generation2/INPUTS.json` binds copied runtime/model/archive/source hashes; `REVIEWS.json` binds
the five final Python source hashes and both clean review passes. All launch manifests record argv,
Git hash, authority, resource/storage checks and nice-best-effort disposition. Every encode uses the
mandated launcher. No Metal, scorer dispatch, live-tree edit, upstream write, bulk deletion or move.
The last storage check showed 77 GiB available, preserving the 40 GiB serializer reserve.

Seed 20260910, deterministic algorithms, one numeric thread per computation; at most two encode/decode
processes. Every stage keeps atomic, distinct checkpoints and complete causal/arithmetical state,
with periodic checkpoints every 20 frames. Both the source trace and selected decoder were measured after
resuming complete frame-32 state from disk through frame 600. Successful scratch has no abandoned bulk; retained
payloads and proof states remain active inputs under certify-or-block, not deletion candidates.

The focused real-input test uses a seeded random 32-frame selection from 128 retained source frames
as an **implementation check only**. It verifies all six packet grammars, exact supplied tokens,
actual crack sides, corrupted CRC/duplicate-segment refusals and native encoder/decoder restarts.
It is not a subset price or population verdict. Ruff and two code-review passes are clean for all
five new Python files; the final focused test passed. Source and candidate full-n600 identities are
separate measurement gates.

**Registered:** `boundary_segment_recode_price_v1` through the canonical locked API, with one
completed empirical anchor and non-promotable macOS provenance. Registration and the exact event
are retained in `generation2_equation_registration.json` and `generation2_equation_events.jsonl`.
Its callable is the actual serializer `experiments.ddm_bnd2_segment_encode:composite`; the measured
price is the length of its returned packet. The identity is accounting, not an entropy bound.

Six solver hooks: sensitivity/Pareto/per-tensor allocation/autopilot deployment are N/A for this
research-only losing code; no scorer sensitivity, production constraint, tensor importance or
archive-deployable public runtime changed. The numerical posterior hook is N/A for this research-only
accounting identity: no numerical posterior update is claimed. Evidence retention through the equation
registry and canonical task disposition is separate. Probe-disambiguation uses the actual full-price
comparison, completed causal masked intervention and completed full independent decoder. No unmeasured
alternative is promoted by metadata.

## Landing and task custody

The scientific unit and canonical API writes are complete. The actual serializer outcome, exact
main or isolated fallback commit, bundle/format-patch hashes and verification are recorded in
`ddm_bnd2_20260910/GENERATION2_FINAL_HANDOFF.json`. A fallback commit is not a main landing.
Only owned source, memo and receipt files enter the intended commit; unrelated shared state and
the staged index are preserved. API event snapshots permit narrow replay without importing whole
shared-state files. The final handoff is a post-serialization index, not a self-referential bundle member.

Generation 1 source/memo landed externally as `1a0206622`; its obsolete nice-refusal memo is
retained as `generation2/generation1_memo.md`. This generation implements and executes the real code.
The focused test, two source-review passes and Ruff passed. Serializer/main hook outcomes are stated
separately in the final handoff and must not be inferred from these checks.

Canonical dispositions are actually written and retained in `generation2_task_events.jsonl`:
`ddm_bnd2...::N600_LOSSLESS_GATE` is completed; bnd1 `::ITEM_2` is cancelled/FOLDED by the failed
byte gate; bnd1's broader `::ITEM_1` is cancelled/FOLDED with the unmeasured map/distance exclusions
explicit, not marked wholly fulfilled. The obsolete gb1 `::VALID_BOUND_INPUTS` demand is
cancelled/FOLDED into gb2's already-landed vacuous FORMULATION result and the-cross/gb2 successor
context. No duplicate bound, Jacobian request, scorer dispatch or automatic generator BUILD fires.
This does not claim that every separate gb2 equation/lane harvest event was replayed by bnd2.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer stores: the owned repo source/memo/receipts
  and `.omx/state/canonical_task_status.jsonl`, `.omx/state/canonical_equations_registry.jsonl`,
  `.omx/state/lane_registry.json`; fire trigger: harvest of the final reviewed serializer artifacts
  in a Git-writable context. Land and verify the exact unit selected by `GENERATION2_FINAL_HANDOFF.json`,
  replaying only its retained bnd2 API events if absent, and record the main commit and required checks.
  This is the existing `ddm_bnd2...::LANDING` obligation, not another measurement or generator dispatch.

## LIVE-HYPOTHESES

- A specified generator may create a cheaper evaluator-equivalent field with more useful coherence;
  this test only re-addresses the incumbent field, and gb2's vacuous bound excludes no specified construction.
- Joint edge assignment, gap bridging or another charged grammar may reduce total price; these six
  greedy assignments establish no optimum over those mechanisms.

## DEAD-ENDS

- This charged greedy segment formulation at the six measured settings loses bytes; it does not
  open the lossy draw or a public runtime build.
- The 83 KB attribution is not a free detachable payload, and achieved lengths are not lower bounds.
- Pass-5 residual maps cannot establish a move-37 shipped-field joint census; their field hashes differ.
- Repeating gb2's bound-only inference cannot admit a generator; a new specified construction is required.

Own-vehicle live frontier, rechecked from the pointer and recomputed from its recorded components: **S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600]**.
This arm has not moved the exact score pointer.
