# DDM LTG1 — Lane topology-generator floor

## Verdict

**FAMILY-CLOSED** on the charter's required verdict vocabulary, with
**FORMULATION** scope only: the finite n600 race comprising five adaptive
topology/visibility predictors and the exact shape grammar
`direct contour | integer q{1,2,4,8,16} x {2,3}-knot trunk + exact contour XOR`.
This does not prove that every possible analytic Lane generator is impossible.

The minimum retained exact construction is **233,262 B**
`[macOS-CPU scorer-free exact rate and receiver measurement, n600]`:

| component | measured bytes |
|---|---:|
| best all-component topology/event packet (`joint`) | 11,148 |
| best exact shape packet (`direct_contour`) | 221,717 |
| real LTGF1 container overhead | 397 |
| **total** | **233,262** |

That is 197,218 B above the 36,044 B GF1 stop gate and 211,563 B above the
21,699 B Lane-carriage bar. Even granting the topology packet and container
for free leaves the exact shape packet at 221,717 B, so possible duplication
between topology events and contour component counts cannot change the verdict.
Stage 1 and Stage 2 were not authorized and did not fire.

This is the exact minimum of the enumerated, retained coder grid, not an
information-theoretic lower bound on untested representations.

## Exact target and denominator

- Axis: `[macOS-CPU scorer-free exact rate and receiver measurement]`.
- Scope: all 600 frames, 384 x 512 = 117,964,800 positions; no sample or crop
  reduction.
- Exact D3 difference mask: 691,095 positive pixels in 14,669 8-connected
  components. The retained packbits are little-endian, 14,745,600 B, SHA-256
  `6ca82a7883411d0eb27addac7dcf662e84d2f9cc66404c299da2e15761c0e0cf`.
- The D3 mask is not the whole `GT argmax == 1` mask: their symmetric difference
  is 9,516 pixels. Its recorded meaning is narrower: a 1 restores source class 1
  over the quotient's class 0.
- Applying that mask to the pinned quotient field
  `deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07`
  materialized a retained 117,964,800 B receiver field with 0 mismatches and
  source SHA-256
  `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.
- No SegNet, PoseNet, R, d_seg, d_pose, S, archive.zip, contest-CPU, or
  contest-CUDA measurement was run.

## Visibility-law race

All streams use the in-tree real adaptive range coder and decode back to their
exact symbol sequence. Only receiver-known prior track state conditions a
survival decision; birth kind, references, and topology symbols are not
conditioned on geometry unavailable at their decision point.

| predictor | default event packet | coded event payload | survival stream |
|---|---:|---:|---:|
| unconditioned | 9,517 B | 9,275 B | 1,366 B |
| lifetime | 9,499 B | 9,257 B | 1,348 B |
| horizon + lifetime | 9,244 B | 9,002 B | 1,093 B |
| ego phase | 9,224 B | 8,982 B | 1,073 B |
| joint horizon/lifetime/ego/area | **9,156 B** | **8,914 B** | **1,005 B** |

The best law saves **361 B**, only 4.37% of the required 8,259 B. Applied to
the charter's recalled 29,958 B baseline with all non-event bytes fixed, it
projects to 29,597 B and does not clear the deficit.

The current exact D3-difference-mask replay of the older #425 default tracker
was 30,168 B excluding its 7,200 B xi section, 210 B above the recalled 29,958 B;
therefore the 29,958 B number is retained as a recalled arithmetic baseline,
not relabelled as reproduced by this run. The current payload and deterministic
repeat were both retained at 37,368 B and identical SHA-256
`1e17dea6b46e1f48573500c56e7e8020b4b20802f67c4215625a5277ae419588`.

The all-component event census was 1,893 births, 6,375 rebirths, 8,244 deaths,
240 merge symbols, 1,913 split symbols, and 2,400 complex symbols. These are
events under this tracker/6-pixel adjacency grammar, not universal topological
invariants of the source.

## Exact shape race

Each row is a real retained packet decoded over all 600 frames with zero mask
mismatches. `residual stream` is the sum of the four actual adaptive contour
streams; it is not an entropy estimate.

| representation | params coded | XOR pixels | residual stream | exact packet |
|---|---:|---:|---:|---:|
| direct contour | 252 B | 691,095 | 221,237 B | **221,717 B** |
| q1, 2 knots | 103,464 B | 240,537 | 160,023 B | 263,720 B |
| q2, 2 knots | 99,208 B | 301,514 | 192,330 B | 291,771 B |
| q4, 2 knots | 94,148 B | 359,193 | 216,287 B | 310,668 B |
| q8, 2 knots | 88,286 B | 471,588 | 248,397 B | 336,916 B |
| q16, 2 knots | 81,806 B | 629,977 | 285,340 B | 367,380 B |
| q1, 3 knots | 121,249 B | 140,409 | 118,336 B | 239,818 B |
| q2, 3 knots | 116,942 B | 209,801 | 162,553 B | 279,728 B |
| q4, 3 knots | 110,005 B | 270,147 | 194,028 B | 304,266 B |
| q8, 3 knots | 102,034 B | 401,640 | 238,451 B | 340,719 B |
| q16, 3 knots | 93,521 B | 616,210 | 288,856 B | 382,611 B |

The best parametric trunk, q1/3-knot, cuts the residual to 140,409 pixels but
costs 121,249 B of parameters; its 239,818 B packet still loses to direct
contour by 18,101 B. Coarser precision lowers parameter bytes but increases the
exact residual faster.

## Per-component table

The following standalone packets use the winning direct-contour shape codec.
They are separately retained and exact; because each packet has its own
container/model initialization, their byte counts are not additive to the
joint shape packet.

| component area | components | source pixels | standalone exact packet |
|---|---:|---:|---:|
| 1–2 | 2,259 | 3,455 | 5,139 B |
| 3–7 | 4,629 | 21,258 | 14,111 B |
| 8–31 | 3,764 | 58,089 | 27,550 B |
| 32+ | 4,017 | 608,293 | 172,612 B |
| **all** | **14,669** | **691,095** | **221,717 B joint** |

The 32+ bucket carries 88.02% of Lane pixels and dominates the exact shape
packet. A 14,669-row component ledger is retained for follow-up inspection.

## Custody and reproducibility

- Result: `/Volumes/APDataStore/pact/ddm_ltg1/RESULT.json`, 35,101 B, SHA-256
  `96de31a1234451ad66606854501bf120d9644fd006cb427a503181487f3f2d84`.
- Manifest: `/Volumes/APDataStore/pact/ddm_ltg1/MANIFEST.json`, 10,409 B,
  SHA-256
  `56d0fe85ddbc78c403281320e3f5b30286d5191fea1aa0a4f9e6e375204221b3`.
- Floor packet: 233,262 B, SHA-256
  `2b5f237e90b1599690dd849400249ea968b9be11dfae1c2e9385f97584ef5630`.
- Winning shape packet and repeat: 221,717 B each, identical SHA-256
  `3419dde9815895cdbab01fd3e5b1536f65c2ad0fb53e12934fd23f3b579effbc`.
- Independent post-run verification checked all 48 manifest records, exact
  shape decode (0 mismatches), source receiver identity, and the 233,262 B
  component recomposition. Verified resume returned rc=0 without recomputation.
- Runtime: 309.094930 s. No RNG is used.

## RECALL EVIDENCE

The bounded full-corpus recall used these query families:

- `lane topology`, `birth death merge rebirth`, `visibility law`, `dash phase`,
  `component lifetime`, and `ego motion phase`;
- exact pins `6ca82a...`, `cc10a7...`, `36,044`, `21,699`, `29,958`, and
  `8,259`;
- `Lane carriage`, `D3B`, `QBW2`, `Morse`, `vineyard`, `parametric lane`,
  and `curve-relative offset`.

Sources consumed included LC3's exact-carriage receipt, both GF1 receipts,
QBW2's temporal-bound verdict, MA2's merged-alphabet receipt, #425's phase
carrier receipts and implementation, the binary-representation mine, D3B's
lossless factorization, and the in-tree adaptive contour/range coder.

Beyond the charter seeds, recall changed the execution in four ways:

1. D3 custody showed the mask is little-endian and means “source Lane over
   quotient Road,” not the full GT class-1 plane. That found and corrected an
   initial fail-closed target mismatch before any coder result was admitted.
2. #425 supplied reusable real tracker machinery, but its 29,958 B seed did not
   reproduce on the exact current D3 difference mask; this memo keeps the
   current 30,168 B replay and the recalled arithmetic baseline separate.
3. QBW2's 93.2331% deterministic topology accompanied a 455,936 B exact
   Road-mask-plus-exceptions object. It supported testing visibility laws but
   prohibited treating determinism as a byte result.
4. No exact topology/shape Lane packet below GF1 was found in the searched
   corpus. The available curve/polynomial precedents explicitly leave dash
   phase, widths, gaps, and topology as residual obligations, so LTG1 used a
   real exact residual rather than transferring a neighbouring ratio.

The bounded conclusion is “did not find an existing under-36,044 B exact
topology/shape packet in the searched Pact research, retained receipts, source,
state, and charter corpus,” not a global nonexistence claim.

## Own-vehicle frontier

LTG1's own best exact Lane object is the 233,262 B finite-grid construction;
its best standalone exact shape object is 221,717 B. Neither beats the 36,044 B
GF1 Lane incumbent, so LTG1 produced no candidate archive and no exact score.
The effective contest pointer remains AFR1 at
`S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600]`, archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`;
LTG1 did not move it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store:
  `.omx/state/main_hot_state.md` task #1365 / `NEXT_BOUNDARIES`; fire trigger:
  this LTG1 receipt lands and the already-landed BLP1 floor closure is jointly
  harvested; action: retire this finite Lane-carriage formulation and route the
  next exact-row unit to the named object-change door, rather than another
  topology/visibility recode of the same D3 mask.

## LIVE-HYPOTHESES

- A representation that changes the object being carried, rather than coding
  the exact D3 difference mask component-by-component, can still move the rate
  corner because LTG1 only closed a finite exact-mask grammar.
- The 32+ component bucket is the only plausible residual target inside this
  evidence: it carries 608,293 of 691,095 pixels and 172,612 B standalone, so a
  genuinely different large-component sufficient statistic would attack the
  measured dominant term.

## DEAD-ENDS

- Receiver-valid lifetime/horizon/ego conditioning is closed as the missing
  8,259 B cure in this formulation: its best real saving is 361 B.
- Two- or three-knot integer boundary trunks at q1/q2/q4/q8/q16 are closed:
  every exact retained packet is 239,818–382,611 B, worse than direct contour.
- Direct adaptive contour coding of the exact D3 mask is closed against GF1:
  221,717 B for shape alone exceeds 36,044 B by 185,673 B.
