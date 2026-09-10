# ddm_pr7 — second-family review of EB1's partition covering bounds

`[no-triality] [p0-ledger-ok]` · owner `ddm_pr7` · `research_only=true` ·
`score_claim=false` · review axis only · `verdict_scope: FORMULATION`.

## Verdict

EB1's **arithmetic and inequality directions pass**, but its two classes do not
satisfy the charter's load-bearing population premise. The reported numbers are
valid minimax lower bounds for two explicitly constructed, post-hoc finite
orbits, **C** and **N**. Neither orbit is proved to be a subset of partitions
that a receiver could genuinely be asked to reproduce from physically
plausible driving videos. Both are defined from this video's GT skeleton and
fitted group profile, which the construction grants as uncharged conditioning.
Therefore neither 1,404.5 B nor 2,100.25 B is an applicable lower bound on this
video, the move-40 realized partition, or the token tail.

This is not GB2's singleton/vacuity error: C and N are nontrivial finite sets,
so their internal covering converses have positive values. The failed step is
transferring those post-hoc sets to the real source or this individual member,
the same quantifier boundary GB2 names at
`ddm_gb2_generated_basis_unconditional_lattice_bound_20260909.md:189-206`.

The prior-law prediction was only partly right. **N** must be downgraded from a
“natural profile” population to a fitted-profile orbit, but **C has the same
applicability defect**; “conservative” describes its smaller in-orbit
cardinality, not a proved relation to a natural source population. The clean
part of EB1 is its own explicit no-transfer boundary
(`ddm_eb1_entropy_lower_bound_boundary_description_20260910.md:203-217,303-338`).
The charter's clean-review falsifier does not fire because the population-
validity premise fails even though both inequality directions survive.

GS3's follow-on summary does not preserve that boundary. Its statement that the
tail is “between ~2 KB ... and 119,784 B” compares different operational
objects, and its description of 12,540 as the move-40 argmax disagreement is
stronger than EB1's source receipt supports
(`ddm_gs3_gestalt_after_submission_20260903.md:1122-1123` versus
`ddm_eb1_entropy_lower_bound_boundary_description_20260910.md:73-88`).

All inherited counts are **[real, n600 cached-label counts; scorer-free
macOS-CPU]**. This review re-derived the bound arithmetic from retained exact
counts but performed no new empirical measurement, scorer run, encode, or
archive build.

## Corrected-statement table

Citation key: `eb1` is
`.omx/research/ddm_eb1_entropy_lower_bound_boundary_description_20260910.md`;
`gs3` is `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md`; and
`SUMMARY.json`, `INCUMBENT_SOURCE_CHECK.json`, and
`EQUATION_REGISTRATION.json` are under `.omx/research/ddm_eb1_20260910/`.

| class | EB1 or GS3 statement | second-family verdict | corrected statement |
|---|---|---|---|
| REFERENCE POPULATION — C | C is a conservative product of 34,963 independent two-cell swaps, each preserving the selected geometry (`eb1:180-185`). | **IN-CLASS VALID; NATURAL-POPULATION INCLUSION UNPROVED.** The swaps are genuine distinct partitions and the local invariant proof is sound, but no member except the observed GT is linked to a plausible video or shown reachable as a frozen-SegNet output (`eb1:203-210`). | C is an artificial paired-swap orbit conditioned on a source-selected skeleton/profile. Its lower bound applies to `R_wc(C,D)` only. “Conservative” must not be read as “proved subset of the real source population.” |
| REFERENCE POPULATION — N | N is the “natural local-profile orbit,” with 129,963 mutable sites in 8,805 groups and `log2|N| = 76,287.0539...` (`eb1:186-195`). | **CARDINALITY VALID; “NATURAL POPULATION” NOT ESTABLISHED.** The exact orbit count and invariant preservation survive, but matching areas, edge counts, boundary-cell counts, and 4-CC counts is not a source law or a physical-realizability proof (`eb1:197-217`). | Call N the **fitted local-profile permutation orbit**. It is larger than C but remains selected around this one GT sequence and grants its fixed skeleton/grouping as conditioning. |
| C subset N subset F | EB1 transfers a C minimax bound to N, and N to the full same-profile class F (`eb1:197-201`; profile implementation `experiments/ddm_eb1_profile_class.py:149-159`). | **PASS FOR WORST-CASE COVERING NUMBERS.** A cover of a superset covers every subset. It does not transfer a uniform-C expected-prefix bound to uniform N/F. | `M(F,D) >= M(N,D) >= M(C,D)` is valid for the declared finite sets. No relation from C/N/F to the unknown physically realizable population has been proved. |
| BALL VOLUME — N | `V_N(D) <= B_2(129963,D)` for arbitrary reproduction centers (`eb1:271-277`). | **PASS; UPPER direction is correct.** Projecting an off-alphabet center coordinate to an allowed binary label cannot increase its distance to any N member; ignoring group composition only enlarges the ball. | Subtracting this ambient binary-ball **upper** bound gives a valid lower bound on `R_wc(N,D)`. It is loose, not an exact restricted-orbit volume. |
| BALL VOLUME — C | `V_C(D) <= z^-D(1+z^2)^K`, optimized to `log2 V_C <= K h2(D/(2K))`, including midpoint centers (`eb1:279-296`). | **PASS; UPPER direction is correct.** A matching center contributes `1+z^2`; a midpoint contributes `2z <= 1+z^2` for `0<z<=1`; other centers cost no less. | The Chernoff expression safely upper-bounds the largest ball around **any** reproduction center. The naive even-radius sum around one C member would not. |
| CONVERSE DIRECTION | `R_wc(A,D) >= log2|A| - log2 V_A(D)` and any replacement U must obey `V_A<=U` (`eb1:219-236`; implementation `experiments/ddm_eb1_entropy_bound.py:270-300`). | **PASS.** EB1 subtracts ball-volume upper bounds, so the result remains a lower bound. Downward whole-bit rounding is conservative. | Preserve the quantifiers: finite A fixed before its member, hard distortion for every member, arbitrary reproduction centers, and fixed-length worst-case messages. |
| UNIFORM EXPECTED PREFIX LENGTH | EB1 extends the count to a uniform member of A under a prefix code, but refuses subset-to-superset inheritance (`eb1:238-243`; `SUMMARY.json:3-63`). | **PASS WITH ITS STATED CONTRACT.** The expected bound is for the declared uniform law on the same A, not for a natural-video law and not for each object. | State the probability law and prefix/error contract every time; no uniform post-hoc orbit may be silently renamed the video population. |
| D = 12,540 | EB1 uses 12,540 as the charter's literal hard budget, while its source receipt has only printed `d_seg=0.00010636` (`eb1:78-83`). GS3 later calls it the argmax disagreement (`gs3:1122-1123`). | **EB1 PASS; GS3 OVERSTATES.** `12,540/117,964,800 = 0.000106302897...`; the printed value implies 12,546.736128 cells and cannot certify an integer count (`ddm_eb1_20260910/INCUMBENT_SOURCE_CHECK.json:2-18`). | D=12,540 is a charter scenario, not a verified exact move-40 error count. A same-object comparison needs raw realized scorer partitions or an exact integer error receipt. |
| 17,631 | EB1 measures 17,631 GT disagreements in the stored move-40 token field (`eb1:66-76`; `SUMMARY.json:199-202`). | **PASS.** It is exact for the stored renderer-input token field and is not the realized SegNet-output distortion. | Keep 17,631 labeled `stored token field vs GT argmax`. Do not substitute it for D in a realized-partition bound without a theorem connecting the two objects. |
| 119,784 / lower bound | At D=12,540 EB1 reports arithmetic ratios 85.286 for C and 57.033 for N and calls them benchmark-to-bound ratios only (`eb1:42-53`). | **EB1'S NARROW WORDING PASSES.** The phrase “the bound is 50–85x below the tail” is numerically true only as division of unlike quantities. It is not an incumbent approximation ratio. | Say: “the in-orbit lower-bound numbers are 57.0–85.3 times smaller than the single observed-video 119,784-B tail.” No optimality bracket follows because the tail is not a code covering C/N at D and its paid side information is unconditioned. |
| 83,259 attribution / lower bound | EB1 reports 59.280 and 39.642 arithmetic ratios and says the attribution is not a physical partition payload (`eb1:42-53`). | **PASS WITH THAT REFUSAL.** The attribution cannot be an achievable-code upper endpoint or an approximation numerator. | It is an oracle ideal-codelength attribution on another object. Do not place it in a rate-distortion interval. |
| “tail between ~2 KB and 119,784 B” | GS3 turns EB1's N figure and the historical tail into a floor/ceiling bracket (`gs3:1122-1123`). | **FAIL — DIFFERENT OBJECT / DIFFERENT QUANTIFIER.** The lower endpoint is worst-case message width for a fitted orbit with free profile conditioning; the upper endpoint is one achieved tail for one token field with paid model context. | No common quantity is bracketed. The applicable incumbent/tail approximation factor remains **unidentified**, exactly as EB1 says at `eb1:47-53,303-329`. |
| GENERATOR DOOR | EB1 says the unconditional construction neither closes nor opens a saving and that partial side information can reduce the conditional converse (`eb1:303-338`). | **PASS.** A deterministic function of Z is sufficient for zero residual description, but not necessary for partial reduction; the correct class is conditional on all receiver-visible Z. | The generator door is open **as a matter of missing bounds**, not as an existence result. No achievable rate saving or lower floor was measured here. |

## Independent arithmetic re-derivation

The retained summary gives exact profile counts at
`.omx/research/ddm_eb1_20260910/SUMMARY.json:3-63,199-230`. Re-running the
registered callable on those exact integers reproduced every whole-bit floor:

| D | C lower bits / B | N raw lower bits / B | N minimax lower bits / B |
|---:|---:|---:|---:|
| 6,270 | 19,741 / 2,467.625 | 40,048 / 5,006.000 | 40,048 / 5,006.000 |
| 12,540 | 11,236 / 1,404.500 | 16,802 / 2,100.250 | 16,802 / 2,100.250 |
| 25,080 | 2,042 / 255.250 | 0 / 0 | 2,042 / 255.250, inherited from C subset N |

At D=12,540, `119,784 / 1,404.5 = 85.2858668565` and
`119,784 / 2,100.25 = 57.0332103321`; for 83,259 B the corresponding divisions
are 59.2801708793 and 39.6424235210. These divisions reproduce EB1, but they do
not acquire an approximation-ratio interpretation. Machine-readable output is
`.omx/research/ddm_pr7_20260910/ARITHMETIC_REDERIVATION.json`.

No retained geometry count was changed. The construction groups sites within a
frame and then products the per-frame cardinalities
(`experiments/ddm_eb1_profile_class.py:22-96,124-172`). Its separation,
simple-point, area, boundary-edge, boundary-cell, and 4-CC invariants are
mathematically sufficient for the summary statistics and were checked on both
retained alternate fields (`ddm_eb1_entropy_lower_bound_boundary_description_20260910.md:142-201,378-408`).
Those checks do not prove video or SegNet realizability.

## Registered-equation check

The live registry contains exactly one
`partition_description_rate_distortion_lower_bound_v1` row among 483 equations.
Its formula, ball maximum, hard-distortion/fixed-length scope, and explicit
exclusions match EB1's memo
(`ddm_eb1_20260910/EQUATION_REGISTRATION.json:2-39`). It correctly excludes a
positive pointwise-video bound, a marginal-tail floor, and inference of
non-near-optimality.

One statement is incomplete for this instantiation. The registry includes a
“fixed source-independent decoder and declared finite reference class” while
the empirical class is built from the selected video's GT skeleton and fitted
profile (`EQUATION_REGISTRATION.json:9-27,40-56`; `eb1:142-167`). Operationally,
that class specification is side information **Z**. A source-independent
receiver may consume it only if Z is declared and available; video-derived Z
must either be charged or the theorem must be labeled conditional on Z. The
current row's exclusions prevent misuse, but the formula should state the
conditioning explicitly:

`R_wc(A_z,D | Z=z) >= max(0, log2|A_z| - log2 sup_y |A_z intersect B_H(y,D)|)`.

This is an append-only wording correction, not a rejection of the registered
counting theorem or its callable.

## What a valid tighter bound would need

This is the boundary of EB2's brief; no EB2 measurement is performed here.

1. Define the receiver-visible side information Z exactly: previous decoded
   partition, transmitted pose, counted archive/model state, and genuinely
   source-independent code. Charge every video-derived profile, skeleton,
   calibration, fitted warp, or exception not already in Z.
2. Define `A_z` **before the tested member** as a conditional source population,
   or prove an explicit counted subset lies inside the population of frozen-
   SegNet outputs of admissible driving videos. Matching marginal geometry is
   not that inclusion proof.
3. Lower-bound `|A_z|` and upper-bound
   `sup_y |A_z intersect B_H(y,D)|`. A ball around P* alone is insufficient;
   sampled volume or an achieved coder length cannot replace the required upper
   bound on the maximum ball.
4. If the claim is expected entropy rather than worst-case covering, declare a
   probability law `Q(P|Z)`, its error/prefix convention, and prove the
   conditional rate-distortion converse for that law. A uniform fitted orbit is
   not automatically the natural-video law.
5. Bind D to the exact realized scorer partition with an integer receipt. Keep
   stored-token disagreement separate. Compare the lower bound only to a code
   with the same source population, conditioning, distortion, and message
   accounting.
6. A real serialized residual coder is useful as an **upper** bound. It cannot
   certify the lower bound or the optimum unless paired with the population and
   maximum-ball proof above.

## RECALL EVIDENCE

The source floor was read directly: the full charter/common contract, PROGRAM,
byte-identical AGENTS/CLAUDE rules, operating handoff, live hot state, EB1 memo
and retained summary/code/equation receipt, GB2, PR5/PR6, GS3 Addenda 19–20 and
corrections, both EB1/EB2 charters, and the live equation row.

Original content searches were not limited to charter seeds. Queries over
`.omx/research` included `partition.*(lower bound|entropy|description)` (120
matching paths), `rate.distortion.*(bound|converse)` (133),
`(individual.object|pointwise).*(bound|entropy|description)` (28),
`(profile|orbit).*(population|partition)` (33), and
`covering.*(Hamming|partition|message)` (8). The canonical equation export had
483 entries; the target ID had one live row. The research index/DAG, design/SPEC
corpus, canonical task ledger, P0 ledger, and lane registry were searched for
the same surfaces. Exact query accounting is retained in
`.omx/research/ddm_pr7_20260910/RECALL_SEARCHES.json`.

Beyond the seeds:

- `ddm_de1_description_efficiency_derivation_20260803.md:151-171,383-436,718-733`
  separates fixed-object operational lengths from ensemble RDF, and charges a
  video-selected receiver. This made the GS3 tail bracket inadmissible.
- `codex_premise_falsification_mdl_ms_complex_k_lower_bound_20260718T063906Z_codex.md:13-25,38-43`
  confirms an achieved description is an upper bound in its language, not an
  individual Kolmogorov lower bound.
- `ddm_tc1_token_tail_bound_and_shared_mixer_pricing_20260909.md:95-135,156-166,319-335`
  keeps fixed-family convex bounds and oracle entropy estimates scoped; neither
  supplies a universal tail floor.

Scoped negative: this search did not find in the local corpus a proved physical
conditional source population containing C or N, or a pointwise lower bound on
the move-40 realized partition/tail. It is not a global nonexistence claim.

## Boundaries

This was a read-only mathematical/source review plus writes to the charter-owned
memo/evidence paths. I re-ran only EB1's retained-count arithmetic. I did **not**
repeat the n600 geometry materialization, run a scorer, encode a payload, build
an archive, dispatch remotely, mutate the live registry/GS3/task ledgers, touch
the staged index directly, edit `upstream/`, or move the pointer. Existing n600
counts remain inherited evidence; every new conclusion here is DERIVED review
and `score_claim=false`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumers
  `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md` and
  `.omx/state/canonical_equations_registry.jsonl`; fire when MAIN harvests this
  verified memo. Append that no common quantity is bracketed by ~2 KB and
  119,784 B, D=12,540 is a charter scenario rather than a certified exact count,
  and the equation is explicitly conditional on charged/declared Z.
- **FOLDED** — owner `ddm_eb2`; consumer memo
  `.omx/research/ddm_eb2_conditional_partition_bound_20260910.md` and retained
  store `/Volumes/APDataStore/pact/ddm_eb2_conditional_bound/`; fire before any
  conditional floor is accepted. Require an ex-ante or inclusion-proved
  physical `A_z`, an upper bound on its maximum D-ball, exact realized D, and
  same-object conditioning/accounting; keep every achieved residual coder a
  ceiling only.

## LIVE-HYPOTHESES

- A physically justified conditional video population could yield a materially
  tighter bound because C/N freeze almost all of the observed space and omit
  temporal/pose source laws; plausibility requires an inclusion proof, not more
  fitted summary statistics.
- A generator or richer coder may still shorten the partition description
  because no pointwise lower bound or same-object approximation certificate was
  established. This is possibility, not achieved saving.

## DEAD-ENDS

- Treating either fitted C or fitted N as a natural source population: every
  member's physical/frozen-SegNet realizability is unproved.
- Retrying the sign objection against EB1's two ball bounds: both volumes are
  bounded from above and the converse direction is correct.
- Bracketing the single tail between an in-orbit minimax floor and its achieved
  byte count, or calling the 57–85 divisions approximation factors: the
  operational objects and quantifiers differ.
- Treating 12,540 as a certified move-40 integer or substituting the exact
  17,631 token disagreement: the first is a charter scenario derived from a
  rounded print; the second is a different field.
- Using an achieved residual/tail code, oracle attribution, or measured entropy
  estimate as a lower bound: each has the wrong inequality or source contract.

Own-vehicle frontier, **unchanged by ddm_pr7**: **S 0.13763861019288715 @
180,233 B [contest-CUDA T4 n600]**, archive SHA-256
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.
