# ddm_eb1 — conditional partition covering bounds; incumbent entropy remains unidentified

`[no-triality] [p0-ledger-ok]` · owner `ddm_eb1` · `research_only=true` · `score_claim=false`.

**Reference classes, before the numbers:** **C** is the conservative product of independent,
paired, topology-preserving boundary-cell swaps. **N** is the larger natural local-profile
class allowing every observed-label permutation within interchangeable boundary sites.
Their exact construction is below. They preserve the video's measured per-frame geometry;
neither is a verified probability law or a proved family of physically realizable SegNet outputs.
The full same-profile class **F** contains N and C; its size is not asserted to be known.

**Exact frontier delta: zero.** At the charter's D = 12,540 cells, the certified conditional
minimax lower bounds are **1,404.5 B for C** and **2,100.25 B for N**. These are bit-equivalent
lower bounds, not encoded payload lengths. They establish neither near-optimality nor a large
achievable coder saving for this individual archive. The requested universal incumbent/tail
entropy verdict is **UNDERDETERMINED**, for the mathematical reasons below.

All new counts: **[real, n600 cached-label counts; scorer-free macOS-CPU]**.
All inequalities and bit bounds: **[closed-form, constants sourced]**.
`verdict_scope: FORMULATION` for the explicitly constructed reference classes.

Let `W = /Volumes/APDataStore/pact/ddm_eb1_entropy_bound`.
The landing's compact evidence is `.omx/research/ddm_eb1_20260910/SUMMARY.json`.
The complete exact integers, per-frame groups and retained alternate partitions are in
`W/retained/` and `W/profile/`. Nothing here is an encode, scorer measurement or new contest row.

## The required comparison

| Hard error budget D, cells | Conservative C lower bits | C lower B | Natural N minimax lower bits | N lower B |
|---:|---:|---:|---:|---:|
| 6,270 | 19,741 | 2,467.625 | 40,048 | 5,006.000 |
| 12,540 | 11,236 | 1,404.500 | 16,802 | 2,100.250 |
| 25,080 | 2,042 | 255.250 | 2,042 | 255.250 |

Source: `SUMMARY.json:3-64`; producer
`experiments/ddm_eb1_entropy_bound.py:270` and
`experiments/ddm_eb1_profile_class.py:155` (minimax nesting step).
The raw N counting converse at D = 25,080 is zero; the table inherits C's lower bound
because C is a subset of N. This transfer is valid for **worst-case covering**, not for
the expected prefix length of a uniform random element of N.

| D = 12,540 benchmark / lower bound | C | N |
|---|---:|---:|
| Historical move-37 whole tail envelope, 119,784 B | 85.286 | 57.033 |
| Move-37 support-and-symbol oracle attribution, 83,259 B | 59.280 | 39.642 |

These are **benchmark-to-bound ratios only**. Neither factor-2 nor factor-10 optimality is
certified. The charter's “unbounded” category applies in the sense **no applicable approximation
ratio for the incumbent is established**, not a claim that its true ratio is infinite.
A weak lower bound never proves that an achieved code is far from optimal: the unknown optimum
may be much larger than the bound. Conversely, a numerical population floor cannot certify this
single object's optimum, even if the ratio happens to be small. The attribution is not a physical
partition payload, and the historical tail is conditional on paid models. No savings are booked.

## Source custody and the three different objects

Population: all pair indices 0 through 599; each last-frame partition is 384 rows by 512 columns;
N_cells = 117,964,800; alphabet size five. Selection mode is `all_scored_pairs_0_through_599`.

- **GT:** `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt`,
  SHA-256 `a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994`.
  Its producer's `result_summary.json` verifies full coverage and Tesla T4/DALI lineage.
  It is the canonical authority-lineage cache used by JG1/BND2, not the older MLX NPZ.
  Only its cached `seg` labels were loaded; no video was decoded and no scorer instantiated.
  The retained NPY copy is bound by `retained/ARRAY_INPUTS.json` and `SUMMARY.json:214`.
- **Move-37 stored token field:** `ddm_cmp1_compose/retained/source/field.u8` under Vertigo,
  SHA-256 `361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94`.
- **Move-40 stored token field:**
  `/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/parseback/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8`,
  SHA-256 `b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5`.
  Source bindings are executable at `ddm_eb1_entropy_bound.py:37` and verified at `:303`.

The latter two are **renderer input tokens**, not the scorer's realized output partitions.
Their GT disagreements are **17,195** and **17,631**, respectively (`SUMMARY.json:199-201`).
Do not substitute the charter's 12,540 rendered-error budget for either token-field distance.
This is an additional surface mismatch, separate from the population/individual mismatch.

**Charter arithmetic correction:** 12,540 / 117,964,800 = 0.00010630289713541667,
not 0.00010636. The retained T4 receipt prints `avg_segnet_dist=0.00010636`, corresponding to
12,546.736128 when multiplied back by the denominator. It does not carry an integer disagreement
count. I did not manufacture one from rounded output. All requested D rows use the charter's
literal budgets; **D = 12,540 is not certified here as the exact T4 incumbent cell count**.
See `INCUMBENT_SOURCE_CHECK.json:16-18` and the receipt it hashes.

The historical benchmarks remain move-37 quantities. The current move-40 staged tail is
119,754 B (`/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/STAGE_TAIL.json:27-33`),
so the charter's 119,784 B must not silently acquire a move-40 label. Likewise the 83,259 B
attribution belongs only to BND3's move-37 free support-and-true-symbol intervention.

## Real geometry census

Boundary cells are the union of in-image cells with at least one differently labeled
4-neighbor. Boundary edges count each horizontal or vertical adjacent unequal pair **once**.
There is no exterior-frame boundary and no time edge in either spatial count. Connected
components use 4-connectivity, separately for each label.

| Object | Boundary cells | Undirected boundary edges | 4-connected components, all classes |
|---|---:|---:|---:|
| GT | 2,551,338 | 1,619,850 | 21,340 |
| Move-37 tokens | 2,564,996 | 1,637,797 | 24,599 |
| Move-40 tokens | 2,564,783 | 1,637,623 | 24,574 |

Sources: `SUMMARY.json:66-109`, `:110-153`, `:154-198`.
Count implementation: `ddm_eb1_entropy_bound.py:105`; independent invariant recounts at `:380`
and `ddm_eb1_profile_class.py:130`. Every frame has its own JSON checkpoint and full
component-size histogram in `W/retained/frames/000.json` through `599.json`.

| GT class | Total area, cells | Total 4-CCs | Per-frame area min–max | Per-frame CC min–max |
|---|---:|---:|---:|---:|
| 0 Road | 27,407,371 | 1,259 | 38,467–50,001 | 1–17 |
| 1 Lane | 690,753 | 16,626 | 542–1,981 | 17–45 |
| 2 Undrivable | 58,413,069 | 646 | 94,466–100,285 | 1–4 |
| 3 Movable | 1,460,386 | 2,209 | 210–10,584 | 1–10 |
| 4 MyCar | 29,993,221 | 600 | 48,549–52,052 | 1–1 |

Areas and components: `SUMMARY.json:67-82`. Distribution source:
`W/retained/TEMPORAL_AND_CLASS_DISTRIBUTIONS.json:1-73`; each min/max is taken over the full
per-frame rows, not a prefix. All original per-frame component-size histograms are retained;
component **sizes** are not claimed invariant across the reference classes.

| Unordered class pair | GT edges | Move-37 token edges | Move-40 token edges |
|---|---:|---:|---:|
| 0–1 | 814,105 | 820,316 | 820,195 |
| 0–2 | 290,128 | 293,937 | 293,952 |
| 0–3 | 90,320 | 92,602 | 92,600 |
| 0–4 | 317,551 | 318,595 | 318,574 |
| 1–2 | 1,567 | 2,153 | 2,143 |
| 1–3 | 1,296 | 1,843 | 1,833 |
| 1–4 | 5,119 | 5,433 | 5,425 |
| 2–3 | 99,601 | 102,661 | 102,644 |
| 2–4 | 0 | 50 | 50 |
| 3–4 | 163 | 207 | 207 |

Sources: `SUMMARY.json:83-94`, `:127-138`, `:171-182`. Each column sums exactly to its
boundary-edge count. The script also verifies that each class's component sizes sum to its area.

GT boundary-cell min/median/max are **3,291 / 4,271.5 / 5,387** per frame;
GT edge min/median/max are **2,048 / 2,716.5 / 3,492** (`SUMMARY.json:95-109`).
The charter's rough 1.5–3 million prediction is met for edges and for cell union counts,
but those are different quantities and cannot be interchanged.

## Constructing nontrivial reference classes without an i.i.d. source assumption

Fix the observed GT skeleton outside eligible sites. Sites are at least two cells inside the
frame and satisfy `(y + 2*x) mod 5 == 0`. Distinct sites have Manhattan distance at least three:
there is no nonzero displacement of Manhattan norm one or two satisfying that congruence.
Their affected center-plus-four-neighbor sets are therefore disjoint, and each radius-two collar
is fixed under all permitted mutations.

At every site, compute:

1. The histogram `h_c` of its four fixed neighbor labels.
2. The histogram `u_c` of **unsaturated neighbors**: neighbors whose other three fixed neighbors
   all have that neighbor's label. A saturated neighbor is already a boundary cell regardless
   of the new center label.
3. The set of **simple labels**: labels whose fixed surrounding 3x3 ring has exactly one
   4-connected component touching the center's cardinal neighbors. Removing or adding a center
   of such a label cannot create/delete a component, merge components, or split one: the
   touched neighbors already connect in that fixed ring, and at least one is present.
4. The frame identity and its 64-row spatial band.

Group sites with identical `(band,h,u,simple-label-set)`. Keep sites whose observed center is
simple and groups with more than one observed center label. Each observed label is therefore
valid at every site in its group. No source location, label table or group specification is
being declared free in a submission; this is the **conditioning that defines the experiment's
reference population**. We grant this fixed skeleton to the hypothetical decoder to make the
conditional lower bound conservative.

For a center assigned label c, the area's change depends only on c. Its undirected class-pair
edge contributions depend only on c and h. Its contribution to the boundary-cell union is

`beta(c) = 1[h_c < 4] + (4 - sum_a u_a) + sum_{a != c} u_a`.

Because affected sets do not overlap, the global B count is the unchanged exterior contribution
plus the sum of these local beta terms. Permuting each group's observed multiset preserves
area, every class-pair edge count, B and component counts **exactly**, for every permutation.
This proof is stronger than relying on two sampled alternatives. Those two alternatives were
nevertheless retained and independently recounted on **every one of the 600 frames**.

**C, conservative:** deterministically pair distinct observed labels inside each group, greedily
pairing the largest remaining color buckets. The executable assertion checks the resulting
`K_g = min(floor(m_g/2), m_g - max_c n_gc)` disjoint pairs. Each pair is one independently
switchable bit, differing in exactly two cells. All unpaired sites remain fixed. Thus
`|C| = 2^K`, **K = 34,963**.

**N, natural local-profile orbit:** allow all multiset permutations at every such group.
Their coordinates are disjoint, so different assignments give different full partitions:

`|N| = product_g m_g! / product_c n_gc!`.

MEASURED: **8,805 groups**, **129,963 mutable sites**, maximum observed group alphabet **two**;
DERIVED from exact integers: `log2 |N| = 76,287.0539110608465…`.
Group positions, counts, switch pairs and exact multinomial factors are retained in
`W/profile/frames/000.json` through `599.json`; the exact full cardinality is hexadecimal in
`W/profile/RESULT.json`. No entropy estimator or independent-pixel law enters this calculation.

**F, natural full same-profile class:** all ordered partition sequences sharing the GT's
per-frame areas, B, class-pair edge counts and component counts. C ⊆ N ⊆ F. I do not claim an
exact enumeration of F; the explicit N construction supplies a real, provable subset, and its
minimax lower bound transfers to F by monotonicity of covering numbers. A ball restricted to N
is never used as an upper bound on a ball in F.

**Plausibility level and its limit:** every C/N member preserves the measured geometry summaries
frame by frame, retains the original bulk regions and moves only local simple boundary sites.
This supports a conditional local-boundary population, much less permissive than arbitrary
five-color images with matching total counts. F matches all requested per-frame statistics but
can still contain unnatural arrangements. Counts alone cannot prove that every member of any
of these classes is produced by the frozen SegNet on a physically plausible driving video.
Nor do they supply a probability distribution or temporal transition law. This is the unresolved
premise in the literal “every element is a plausible GT” requirement; it is **not silently assumed**.

For scale, the GT has **1,467,270** changed cells across all **599** adjacent scored-frame
transitions (`W/retained/TEMPORAL_AND_CLASS_DISTRIBUTIONS.json`, complete directed class-pair
transition counts). A member of N differs from GT on at most 129,963 cells, so its total temporal
change count can change by at most 259,926, by the triangle inequality applied to each temporal
edge (each cell participates in at most two). We do not claim temporal-count invariance or
physical motion consistency. A natural video-conditioned population remains unidentified.

## The valid rate–distortion counting argument

Fix a finite class A **before choosing its member**, a decoder, and a hard distortion guarantee
`d_H(P, P_hat) <= D` for every P in A. Let

`V_A(D) = max_y |A intersect B_H(y,D)|`,

where y ranges over **all** reproduction partitions, including ones outside A. If M reproduction
messages cover A, the union bound gives `|A| <= M * V_A(D)`, hence

`M >= |A| / V_A(D)`. Minimizing over covering codes therefore gives
`R_wc(A,D) >= log2 |A| - log2 V_A(D)`.

More precisely the minimum fixed-length message width is at least
`ceil(log2 ceil(|A| / V_A(D)))`; the displayed tables deliberately round the proved real bound
**down** to whole bits. Whenever we replace V_A by a quantity U, we prove **V_A <= U**.
Subtracting `log2 U` therefore preserves a **lower**, not upper, bound on required description.
A ball just around P* is not enough unless separately proved maximal.

For a uniform member of A and hard per-source distortion, an analogous expected **prefix-code**
bound follows from `H(P|message) <= log2 V_A` and `H(message) <= E[length]`.
Without a source law or prefix restriction one must not import that expected-length statement.
For arbitrary variable-length strings capped at b bits, there are `2^(b+1)-1` possibilities,
so the corresponding worst-case lower bound loses at most one bit. None of these statements
bounds **each** object's description length.

This is the finite-block operational distinction studied in
[Kostina and Verdú, Fixed-length lossy compression in the finite blocklength regime](https://arxiv.org/abs/1102.3944)
and [Kostina, Polyanskiy and Verdú, Variable-length compression allowing errors](https://people.lids.mit.edu/yp/homepage/data/isit14_vlfcompr.pdf).
The counting proof above is self-contained; no memoryless approximation from those papers is applied.

### Exact ambient balls

For q labels and n cells, the full Hamming ball is exactly

`B_q(n,D) = sum_{k=0}^D binom(n,k) * (q-1)^k`.

Choose the k changed coordinates and one of q−1 other labels at each. There is no double count.
Its finite integer recurrence is implemented at `ddm_eb1_entropy_bound.py:248`, with divisibility
assertions. For the full GT lattice q = 5, n = 117,964,800:

| D | log2 full ambient ball, exact integer count rendered numerically |
|---:|---:|
| 6,270 | 110,608.8869478693… |
| 12,540 | 208,684.4259799300… |
| 25,080 | 392,294.5614749351… |

`W/retained/RESULT.json` retains each exact integer in hexadecimal and outward decimal log bounds.
The full ambient ball is a valid upper bound on any restricted ball, but makes both local
reference-class converses vacuous. Treating it as the size of the **reference class** and
subtracting itself would simply describe one distortion ball, not the required population.

For N, all fixed coordinates can only reduce the admissible radius. Each variable coordinate
has two possible observed labels, so `V_N(D) <= B_2(129963,D)` for **any** reproduction center.
If a reproduction coordinate is outside its permitted alphabet, moving it to a permitted label
cannot increase the distance to any class member. This handles off-class centers correctly.
The resulting exact upper-bound log volumes are **36,238.1698886708…**,
**59,484.7642456362…**, and **91,961.5073031071…**. Subtracting them from log2|N| gives
40,048, 16,802, and zero certified bits after nonnegativity and downward rounding.

### Conservative swap balls, including midpoint centers

For each swap pair the two alternatives have Hamming distance two. A matching center gives
distance-generating polynomial `1+z^2`; a midpoint gives `2z`. Other colors or mismatches only
increase distances. Since `2z <= 1+z^2` for 0 < z <= 1, **every** reconstruction center obeys

`V_C(D) <= z^(-D) * (1+z^2)^K`.

For D < K, optimizing at `z^2 = D/(2K-D)` yields

`log2 V_C(D) <= K * h2(D/(2K))`.

For D >= K use `V_C <= 2^K`; for D = 0 the exact volume is one.
The three log-volume upper bounds are **15,221.3069291757…**,
**23,726.3983152081…**, and **32,920.0892131664…**. These are proved upper bounds, not sampled
ball volumes. A tempting `sum binom(K,j), j<=floor(D/2)` would be wrong for arbitrary centers:
midpoints can cover both alternatives of a pair at cost one. The implemented Chernoff bound
explicitly avoids that trap.

All logs are enclosed with exact rational arithmetic: normalize an integer by a power of two,
use the positive atanh series for log with an explicit geometric remainder, and bracket any
discarded low bits. Display endpoints are rounded outward with Decimal. No floating-point
rounding error can turn the certified final floor upward (`ddm_eb1_entropy_bound.py:222-270`).

## What this says about the incumbent and the generator door

1. **Population is not an individual object.** The theorem says a decoder covering the entire
   class needs enough possible messages, or gives a uniform-population average under the stated
   coding contract. It does not say every member needs that many bits. The only universally
   justified pointwise lower bound supplied here is nonnegativity, zero bits. That is **not**
   a claim that a legal zero-byte witness exists. Hard-coding this video's learned labels into
   free code would remain forbidden.
2. **The tail is conditional.** Move 40's staged body already contains HPAC, semantic and carrier
   sections as well as the tail (`STAGE_TAIL.json:27-33`). Those paid bytes may carry partition
   information. A bound on total partition description is not automatically a lower bound on
   the tail given those sections; still less on the 83,259-B oracle attribution. The token
   field and realized SegNet partition also differ, as directly counted above.
3. **Side information can help partially.** Let Z be the information actually available to the
   receiver and A_z the remaining admissible partitions. The applicable converse uses
   `log2 |A_z| - log2 max_y |A_z intersect B(y,D)|`. It can fall below an unconditional bound
   even when P is **not** a deterministic function of Z. Being a function of Z is sufficient
   for zero residual partition description, not necessary for any reduction. Thus the charter's
   “unless it is a FUNCTION” exception is too narrow for partial side information.
4. **Name and charge Z.** It may include generic renderer code, dimensions/frame index and
   genuinely source-independent calibration; already decoded previous-frame partitions;
   and **counted** pose/statistic/model bytes already in the archive. Previous partitions are
   not free at the start of the sequence: their transmission belongs in the chain. The GT video,
   cached argmax tables, SegNet/PoseNet weights, a source-selected skeleton or video-specific
   decoder constants are not an uncharged receiver oracle. A generator cannot evade a valid
   conditional population converse by deterministic computation, but this experiment has not
   established its relevant A_z or a pointwise generator floor.

**Prior-law verdict:** the geometric edge-scale prediction is supported. Neither proposed
20–60 KB / 5–20 KB floor band is established at D = 12,540 by these rigorous constructions.
The “every valid class gives >=60 KB” falsifier does not fire; our nontrivial matched-profile
classes yield much weaker bounds, and shrinking a class always weakens such a universal
quantifier. This does **not** refute a stronger bound for a defensible larger/video-conditioned
population, and does **not** reopen a code family by claiming a measured rate saving. GS3's
“near its entropy / no coder can reach the corner” remains unsupported; its logical negation
is not proved either. The missing assumption is now explicit rather than hidden in a number.

## RECALL EVIDENCE

Read the entire charter and common contract; governing sources included PROGRAM, the identical
AGENTS/CLAUDE rules, the operating handoff, current hot state, pointer, the actual evaluator and
scorer geometry, and the recent MAIN directives for BND2 and PM2. The common contract's old
frontier line was superseded by the live pointer. The initial checkpoint lookup found no ddm_eb1
predecessor. No conflicting ddm_eb1 lane was found in the lane/task searches.

Original recall was not limited to charter seeds. Exact queries, scopes, counts and file hashes
are in `RECALL_SEARCHES.json`, with full outputs in `W/recall/`:

- Research memos/arm receipts: `entropy lower bound|rate.distortion lower|individual.object|partition.*entropy|boundary.*lower bound|near its entropy` — 215 matching lines.
- Research index and all `sub015_DAG_*` files: `entropy|rate.distortion|boundary.*bound|individual.object` — 476 lines.
- Design/docs: `entropy.*bound|rate.distortion|partition.*description|MDL` — 88 lines.
- Task/P0/lane ledgers: `entropy.*bound|boundary.*description|ddm_eb1|ddm_(tc[123]|bd1|mc1|gb2)` — 42 lines.
- The required full canonical-equations CLI export contained **482** entries and was searched
  for entropy, partition, boundary, description, context, and rate-distortion equations.

Beyond the seeds:

- `ddm_de1_description_efficiency_derivation_20260803.md:151-170,385-411` explicitly separates
  fixed-object operational descriptions from ensemble RDF and charges source-selected decoders.
  This changed the comparison into a conditional population result, not an incumbent floor.
- `codex_premise_falsification_mdl_ms_complex_k_lower_bound_20260718T063906Z_codex.md` rejects
  measured description length as a universal individual-complexity floor. No older MDL number
  was promoted into a bound.
- TC1's actual bound memo explicitly excludes a universal mixer ceiling and notes that BD1's
  counting-model ratios do not bound HPAC. This prevented transferring old token-level
  “entropy” or motion-context closures to the partition population.
- `ddm_bnd2_census.py:85-109` identified the authority-lineage DALI cache and its custody checks.
  `ddm_tc3_move40_trace.py:40-48` identified the actual move-40 stored-token field. These changed
  the inputs from the optional old MLX cache to the source-verified DALI cache and made the
  stored-token/realized-output distinction measurable.
- The source T4 receipt exposed the 12,540-versus-printed-d_seg arithmetic inconsistency.

Scoped negative: these searches did not find a proved video-conditioned population law or
applicable pointwise/conditional-tail lower bound. That is not a claim of global nonexistence.

## Verification, retention and completion boundary

Two visible review-tracker passes plus Ruff passed for each new Python file. The final
base-file landing removes two redundant import-noqa comments under the repository Ruff config;
AST identity is verified against the preserved measured source, and its two reviews were reset
and repeated. `COMMENT_ONLY_LINT_RECONCILIATION.json` binds both hashes. The base review
fixed outward numerical enclosures and resume input identity before launch; the profile review
checked the sufficient-statistic grouping, simple-point topology and all inequality directions.
Mathematical unit checks compare exact binomial counts to a separate direct sum, include every
binary off-class/midpoint center for small swap products, and verify exact power-of-two logs.
They are **mathematical unit checks, not synthetic empirical anchors**.

Actual empirical validation covers all 600 frames of GT and both shipped token fields. Both
control-class and broader-profile alternate partitions were persisted and their invariant
geometry independently recounted on all 600 frames. The original collar-conditioned control
is retained, including its weaker D = 12,540 bounds of 620.125 B for swaps and zero from the raw
permutation-ball converse. The profile construction removes unnecessary full-collar conditioning;
it does not tune an entropy estimate until it hits a preferred number.

Each launch used one numerical process, native thread caps, an APDataStore free-space reserve,
atomic per-frame checkpoints and a documented `--resume-from`. `nice(10)` was refused and the
recorded best-effort path continued; no scorer lane was requested. No train, encode, scorer,
remote/GPU job, live-tree edit, live index mutation, upstream write, publication or pointer move
was performed. Generic artifacts from the shared repo were read only. The isolated checkout
and all new evidence are under the charter's APDataStore directory.

All large arrays are required retained inputs or explicit class witnesses. The hygiene path
blocks writes below reserve, atomically promotes per-frame temporary files, resumes completed
frames without rematerializing them and deletes no required evidence. The final custody manifest
records each retained file's bytes/hash and the launch/source bindings. There is no unowned raw
video, scorer tensor tree or candidate archive sweep to clean up.

The canonical API registered and query-verified
**`partition_description_rate_distortion_lower_bound_v1`** in the isolated ledger; the landing
contains the narrow append event. Registration does not certify a physical source law.
Sensitivity/bit-allocation/autopilot hooks are N/A: there is no new actuator or candidate.
No hard production Pareto floor is installed from these conditional bounds. The law and
source/count receipts are the research consumer output; the explicit scope mismatch is the
probe disambiguator. There is no numerical scorer-posterior update.

**Completed:** real geometry counts, both explicit class constructions, exact cardinalities,
proved ball upper bounds at all requested D values, benchmark comparisons with levels,
retained witnesses, law registration in isolation, and a reviewable landing.
**Not established:** a probability law of natural driving-video partitions; physical SegNet
realizability of every class element; a nontrivial individual-video or marginal-tail floor;
an exact T4 12,540-cell count; achievable entropy gap or any new score. The literal universal
incumbent-entropy inference is not mathematically licensed by the requested statistics.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer stores** `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md`, `.omx/state/canonical_equations_registry.jsonl`, and `.omx/state/canonical_task_status.jsonl`; **fire trigger:** harvest this arm's verified isolated landing. Import the narrow law/memo/code changes and append the scoped result: conditional population bounds, no near-entropy verdict, token/output separation and charter-D arithmetic correction. No encoder/scorer dispatch is implied.

## LIVE-HYPOTHESES

- A stronger natural-video conditional population may yield a useful floor, because this proof
  deliberately freezes almost all of the observed geometry and only counts isolated local
  boundary degrees of freedom. Its source law and receiver-visible information remain unproved.
- A richer causal model or a generator may still beat the tail, because neither these small
  population bounds nor the prior achieved encodes establish this archive's optimum. Paid
  shared models, pose and previously decoded partitions could explain additional structure.

## DEAD-ENDS

- Treating achieved recode sizes, a sample entropy estimate, or one object's MDL as a universal
  floor: none establishes the required inequality or population contract.
- Subtracting a ball around P* without controlling the largest ball around any decoder output:
  off-class midpoint centers invalidate the naive paired-swap ball.
- Calling a weak lower bound proof of a large available saving, or a population lower bound a
  per-object/tail lower bound: both change the theorem's quantifiers or its conditional object.
- Treating 83,259 B as a detachable boundary payload or 12,540 as a stored-token disagreement
  count: the former is an oracle attribution; the latter conflicts with the actual token census.
- Claiming that matching areas and boundary statistics proves every field is a plausible
  SegNet output: those summaries do not establish video realizability or a source distribution.
- Repeating 12,540 / 117,964,800 = 0.00010636: that arithmetic is false. The requested D remains
  a charter budget, not a newly verified exact T4 integer count.

Own-vehicle frontier, **unchanged by ddm_eb1**: **S 0.13763861019288715 @ 180,233 B
[contest-CUDA T4 n600]**, archive SHA-256
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.
