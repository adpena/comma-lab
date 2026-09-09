# ddm_gb2: the scoped lattice/counting bound is vacuous; no generated family is admitted

2026-09-09 · owner `ddm_gb2` · lane `lane_ddm_gb2_generated_basis_bound_20260909`.
`[no-triality] [p0-ledger-ok]` · `research_only=true` · `score_claim=false` · `promotable=false`.
Axis: **[closed-form, constants sourced; exact local bytes, scorer-free]**.

**The bound-only design gate is closed at FORMULATION scope.** The constructions below
yield no non-vacuous individual-object byte floor for any of the three named families.
All nine family/K rows have only the valid **0 B counting lower bound**, which neither
rules out a saving nor establishes one. No family is proved able to save 300 B at the
pose budget; no family is proved unable to do so. No BUILD charter fires from zero.
The prior prediction of a floor above the stored basis minus 300 B remains untested,
not confirmed by pc1/pc2's instance failures and not falsified by a loose zero bound.

This completes the charter's explicit vacuous-construction alternative. It does **not**
prove that a positive bound cannot exist for a fully specified generator or for this
particular carrier under additional analysis. It closes the attempt to derive a decisive
generated-family bound solely from the supplied lattice data and unspecified families.

The memo, equation, lane, and closure routing are committed in the charter-owned isolated
SSD checkout. MAIN must harvest their events into the live stores; no live tree is written
by this arm. The exact pointer did not move because of this work.

## Measured live inputs and the correct byte comparison

Durable root `W = /Volumes/VertigoDataTier/pact/ddm_gb2_generated_basis_bound`.
`W/input_audit/AUDIT.json` retains the source archive, every parsed byte field,
all arrays, source snapshots, hashes, argv, and pointer identity. No source payload is
discarded. The second receiver path in `W/verification/CONTROLS.json` independently
reconstructs CAP1 -> CPR1 -> all codes, basis symbols, and scales and checks exact equality.

| Live cmp2 archive component | Measured bytes |
|---|---:|
| ZIP framing | 100 |
| RX1 header | 14 |
| HPAC stream | 11,911 |
| Semantic stream | 29,862 |
| **Carrier stream** | **18,586** |
| Token tail | 119,915 |
| **Total archive** | **180,388** |

These sum exactly. Archive SHA-256
`670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`;
source `/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/candidate_runtime/archive.zip`.
The cmp2 seal binds this archive at its `archive` member; its accompanying section table
is `.omx/research/ddm_cmp2_compose_sm1_fe1_20260909.md:153-160`. The seal JSON itself
does not contain a section table. Fresh tree-reader parsing agrees with the memo.
gb1's **18,580 B is an earlier pc2 carrier**, not this live carrier.

The restored, pre-Brotli carrier body is **18,877 B**:
6 bit-count bytes + 96 scale bytes + 40 packed metadata bytes +
**12,277 Huffman basis bytes (98,213 declared bits)** +
**6,429 Rice bytes (51,430 declared bits)** + 29 selector-tail bytes.
There are **27,648 basis symbols = 12 x 3 x 24 x 32**.
The 12,277 B basis field is an exact internal length, **not its separately attributable
final ZIP cost**: the compressed carrier jointly includes fields and framing.
Its raw-reference subtraction is 11,977 B; the operational 300 B target is
**a final carrier <=18,286 B**, or full archive <=180,088 B if all other sections and
container overhead stay fixed. No archive was built to test that condition.

All **600 x 12 = 7,200 integer codes** were retained. Range [-142,154]; 7,150 nonzero
codes; **600 distinct pair vectors**. The selector was decoded across all 600 pairs.
The current selector has **no compensation overlay**, so effective and base codes agree
in 7,200/7,200 coordinates. This is a code/reader result, not a rendered-pixel or pose result.
Basis scales are twelve 1.0 values. The coefficient scales are:

```
0.005224729422479868  0.005467622075229883  0.005400871858000755
0.008356369100511074  0.007224291563034058  0.005270141642540693
0.007709944155067205  0.005037846043705940  0.008611684665083885
0.005425599403679371  0.005216557066887617  0.007675567641854286
```

`effective_codes.npy`, `coefficients_fp32.npy`, `basis_codes.npy`, and `scales_fp32.npy`
retain the complete arrays. Shipped coefficient multiplication rounds to fp32; the
**codes** are the integer lattice. An exact-real scaled lattice is a reference geometry,
not a claim that the full rounded receiver is linear.

## Source lines for the constants and mechanism

Runtime-relative paths here mean the sealed cmp2 `candidate_runtime` above; snapshots
and their full SHA-256 values are in `AUDIT.json` and the final artifact manifest.

| Constant/mechanism | Source |
|---|---|
| n600, K=12, 3-channel 24x32 atoms, int12 | `cpr1/inflate.py:20-32`; channels/shape at `:245` |
| Codes and coefficient scales consumed | `cpr1/inflate.py:188-250` |
| Integer restoration and AR/Rice reader | `experiments/ddm_up3_carrier_splice.py:415-524`, `:542-558`; runtime `entropy/coefficient_ar1_codec.py:34-78`, `:87-131` |
| Live header/section extraction, CK2/riders | runtime `residual_archive.py:174-232`; `ddm_up3_carrier_splice.py:436-472` |
| Metadata, exact bit counts and payload boundaries | runtime `residual_archive.py:127-153`, `:381-432`; `ddm_up3_carrier_splice.py:469-503` |
| Basis Huffman unsigned symbols are zigzag, not int5 two's complement | `cpr1/carrier_codec.py:65-100`, `:201-215` |
| Overlay applicability | runtime `f26_inflate.py:481-497`; `compensation_overlay.py:146-186` |
| Normalize, resize, clamp, round, spatial rendering | `cpr1/inflate.py:290-297`, `:335-352`; selector additionally acts in F26 |
| Pose score is mean squared output error, then square root | `upstream/modules.py:84`, `upstream/evaluate.py:72-92` |
| Rate denominator 37,545,489 B | fresh `stat` of `upstream/videos/0.mkv`; `SSD_ARTIFACT_MANIFEST.json:source_video_bytes_measured` |
| Final archive, not a raw basis, is priced | `upstream/evaluate.py:63-65`, `:92` |
| Free tools/charged learned artifacts; runtime limit | `upstream/README.md:114-119`; charter's stricter counted-fitted-information rule |
| eps=0.000125 S; seed <=8 B; saving >=300 B | this arm's charter, MANDATE / PRIOR-LAW PREDICTION |
| Family K=6,8,12; steps 1,1/2,1/8 | gb1 memo's Requested matrix |

## The pose budget is not a coefficient-space radius

Let `r` be the 3,600-dimensional vector of six pose residuals on all 600 pairs.
The pose contribution is `sqrt(10/3600) ||r||_2`. For the conditional arithmetic below,
the charter's literal **S-unit** tolerance is interpreted as an allowed pose score
increase `eps=1.25e-4` relative to the shipped point. This is an explicit interpretation,
not a verified transfer of memory m110. The referenced m110 has a units/scope mismatch:
`ddm_gs3_gestalt_after_submission_20260903.md:146-148` identifies its 1.25e-4 as an
**absolute d_pose allowance for a new small object**, not a same-object score tolerance.
Reading the charter instead as total pose contribution <=1.25e-4 S would imply
`d_pose<=1.5625e-9`; reading m110 as d_pose<=1.25e-4 gives yet another acceptance set.
None supplies a coefficient radius or changes the vacuous counting verdict. A successor's
concrete generator contract must type its budget before any admission; this arm admits none.

```
sqrt(10*d_new) - sqrt(10*d_base) <= eps
<=> d_new <= (sqrt(10*d_base) + eps)^2 / 10.
```

Using the existing mirror's **report-8dp** d_base=5.05e-6 gives
`d_new <= 5.229220880044399e-6`, an increase of `1.792208800443992e-7`.
This is derived from rounded prior inputs, not a new exact pose measurement.
A sufficient output-change norm by the triangle inequality is
`||delta_pose_outputs||_F <= eps*sqrt(360) = 0.0023717082451262844`.
It is **not necessary**: output changes can reduce the existing residual.

Neither condition implies a maximum coefficient error without a valid inverse
stability inequality for the complete realized map. Normalization, bicubic resize,
clamps, integer rounding, selector operations, and PoseNet prevent substituting
`||delta_coefficients||` for `||delta_pose_outputs||`. A forward Lipschitz upper bound
would give a sufficient small-change condition; failing it would not exclude success.
Changing the basis also changes what any same-valued coefficient means.

The pure-rate credit of 300 B is `25*300/37,545,489 = 0.0001997576859366514 S`.
At the full allowed pose increase the net would be **-0.0000747576859366514 S**,
assuming unchanged seg and all other bytes. This is conditional arithmetic only.

## Construction 1: exact lattice distance and covering radius

In code coordinates let `Q={-2048,...,2047}^12`, and let the retained rows be `q_i`.
For any code-coordinate target `y`, the exact weighted rectangular-lattice distance is

```
d_Q(y)^2 = sum_j s_j^2 * min_{z_j in {-2048,...,2047}} (y_j-z_j)^2.
```

This follows by separability of the sum; each coordinate is rounded to a nearest
integer and clipped to the finite domain. It is an **integer** minimum, with no span
relaxation or empirical rounding penalty. At the actual retained targets `y=q_i`,
every summand is zero. All **600/600** targets belong to Q exactly. This identifies
the shipped code lattice, not the information needed to generate a different basis.

The full infinite reference lattice `diag(s) Z^12` has a rectangular Voronoi cell
with half-widths `s_j/2`. Its covering radius is
`rho = sqrt(sum_j s_j^2)/2 = 0.011299380194700023`.
This is a **worst-case upper bound on nearest-point distance**, attained at cell
corners; it is not a positive lower bound for any observed target. For targets
outside the finite Q box the infinite-lattice covering statement does not apply.
On the actual finite code targets, distance remains zero.

For a specified generated spatial matrix `G_theta`, a real finite-lattice feasibility
problem would instead ask whether an integer `z_i` produces acceptable realized
outputs. Even the stronger field-matching surrogate
`min_{z_i in Q_K} ||B diag(s) q_i - G_theta diag(t) z_i||`
requires the specified atoms, scales and normalization. That field minimum would
still not be a necessary pose error: a different field can have the same six pose outputs.
No G_theta or inverse-stability certificate is supplied here. The family names and K
do not supply one. **No span projection is used or reported as the requested bound.**

## Construction 2: counting Voronoi cells / acceptable targets

Fix a **source-independent receiver and a complete finite description language** first.
Let T be a finite set of source objects that the code must support. A message represents
one decoded object and can serve some set of acceptable targets; let V upper-bound the
size of that set for every message. For a fixed b-byte message:

```
|T| <= 256^b V,
b >= ceil(log_256(ceil(|T|/V))).
```

Proof: at most `256^b` messages, at most V target incidences per message; a cover of T
requires at least |T| incidences. Overlap only weakens the cover. This counting argument
applies to discrete lattice cells and does not require a continuous-volume surrogate.
The implementation computes the integer ceiling with `bit_length`, avoiding log rounding.
For variable lengths **up to** b, replace `256^b` by `sum_{j=0}^b 256^j`; a fixed-length
bound is not silently assigned to that different code convention.

At this task's unconditional individual-object scope, the sole required object is the
**entire ordered 600-pair carrier**. Thus `T={Q_star}`, where `Q_star=(q_0,...,q_599)`, has
cardinality **1**, not 600 or `4096^(600K)`. The safe incidence upper bound is V=1.
The numerical floor is therefore **0 B** for both length conventions. This says nothing
about attainability: if no acceptable message exists the true cost is infinite, which
also satisfies the zero lower bound.

The 600 distinct pair vectors do not change that conclusion. They are components of
one source, with pair indices already available to the decoder, not 600 alternative
full sources that a generic code is required to distinguish. Arbitrary int12 strings
could supply a different universal source alphabet, but the resulting worst-case bound
would be about **some** strings in that alphabet, not necessarily this array. Nor can
the measured Rice/Huffman lengths identify an individual incompressible member.

This is the precise quantifier obstruction. We do not assert a source-specific table is
free, invoke an illicit hardcoded-target decoder, or claim that this carrier has a short
generic program. We have no evidence excluding every short **legal** program for it.
Counting alone cannot identify which individual targets receive short descriptions.

## Construction 3: finite generated descriptions and fitted information

For a fixed generic algorithm a fixed-width 64-bit seed can select at most `2^64`
descriptions. **Eight bytes is a maximum allowed seed length, not a minimum byte cost.**
A predetermined seed needs no transmission. A seed chosen using the video is counted
in the archive; it may carry positional information by selecting an atom from a fixed
library. Therefore the prior statement "a seed cannot carry position" is too strong.
It cannot carry an unbounded fitted table for free, but 64 selection bits are real information.

All fitted atom coordinates, widths, frequencies, angles, transforms, normalization
parameters, scales, coefficient innovations, and residual sidecars must be counted.
Fixed generic algorithmic structure is free. A seed for running SVD does not specify
the fitted SVD input or output; regenerating learned atoms from that seed alone is invalid.
The 48 coefficient-scale bytes and 48 basis-scale bytes in this archive are recorded
facts, not proof that every alternative must use the same encoding or precision.

To certify a positive individual-object bound one must exclude all admissible messages
below the candidate byte cap for a specified language, or supply a sound analytic
certificate accomplishing that exclusion. K alone, an eight-byte seed cap, and a family
name do not specify its atom parameter domains, discretization, code syntax, or realized
pose acceptance sets. The seed count bounds the number of choices, **not whether this
target lies in one of their acceptable sets**. No n600 quantities in the archive fill
those missing definitions. No training, parameter search, CVP search, or archive build
was performed to replace the missing proof.

## Family table and decision

The three generic families are gb1's Zernike, steerable pyramid, and generated Gabor.
The fitted-SVD row is excluded from the generic class for the explicit counted-information
reason above. Each row below covers gb1's step multipliers {1,1/2,1/8}; these do not
alter the counting quantifiers. `FAMILY_TABLE.json` records all nine rows.

| Generic family | K | Valid unconditional counting floor | Achieved bytes at pose budget | >=300 B saving established? |
|---|---:|---:|---|---|
| Zernike | 6 | 0 B, vacuous | unknown | No |
| Zernike | 8 | 0 B, vacuous | unknown | No |
| Zernike | 12 | 0 B, vacuous | unknown | No |
| Steerable pyramid | 6 | 0 B, vacuous | unknown | No |
| Steerable pyramid | 8 | 0 B, vacuous | unknown | No |
| Steerable pyramid | 12 | 0 B, vacuous | unknown | No |
| Generated Gabor | 6 | 0 B, vacuous | unknown | No |
| Generated Gabor | 8 | 0 B, vacuous | unknown | No |
| Generated Gabor | 12 | 0 B, vacuous | unknown | No |

**0/9 families/K configurations have a non-vacuous generated-basis price certificate;
0/27 family/K/step combinations have an achieved-byte/pose result.** The table evaluates
the scoped bound, not a fabricated family codec. A lower bound below a cap is only a
necessary possibility screen; it is not proof that an affordable point exists. The
charter explicitly allows the vacuous outcome, so that outcome cannot itself fire its
BUILD clause. The prior above-11,977-B prediction remains untested.

The three gb1 defects are avoided explicitly: (1) no span-plus-rounding substitute;
(2) every fitted description is counted, with no seed-SVD loophole; (3) the complete
600-pair code population is read, and **zero** six-pair Jacobian extrapolations are used.
pc1's large basis-edit penalty and pc2's local fixed-point/step failures remain useful
instance priors. A local solver fixed point is not a global integer-lattice certificate.

## RECALL EVIDENCE

Own content searches and all outputs are retained in `W/recall/SEARCHES.json`:

- `.omx/research/` md/json/jsonl and arm receipts: `generated.{0,30}(basis|carrier)|carrier.{0,45}(lattice|Voronoi|covering)|Kolmogorov|positioned.subspace|basis.*39.?748` — 613 matching lines across 233 paths.
- `docs/` md/SPEC: `carrier.{0,45}(basis|lattice)|generated.{0,30}basis|Zernike|Gabor|steerable` — one matched path.
- `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*`, same expression — 16 matching lines.
- Task/P0/lane stores: `generated.{0,30}basis|lattice.{0,30}carrier|ddm_gb[12]|pose.{0,20}budget` — 19 matching lines.
- Full canonical-equations CLI export — **480 equations**. Relevant rows retained separately.
- No 2026-09-08/09 top-level directive filenames were found in the bounded directive-name check.

**Beyond the seeds:**

1. `codex_premise_falsification_mdl_ms_complex_k_lower_bound_20260718T063906Z_codex.md:15-25`
   already rejects measured MDL as a universal individual-complexity lower bound. This
   prevented promoting the 12,277 B Huffman length into a generated-program floor.
2. `ddm_de1_description_efficiency_derivation_20260803.md:151-170`, `:385-411`
   separates fixed-object operational length from ensemble RDF and charges source-selected
   decoder information. This changed the proof to one ordered n600 object and explicit
   source-independent description languages, with no entropy averages.
3. `carrier_rate_credit_pose_affordance_v1` supplies the nonlinear pose affordability
   equation; `pose_carrier_basis_rate_fidelity_exchange_v1` expressly excludes d_pose
   predictions. Their domains prevent converting field or spectral fidelity into score.
4. The live cmp2 section census changed the baseline from 18,580 B to **18,586 B**.
   The hot board folds gb1's open proof gate into gb2 and assigns current score-moving
   work elsewhere; this arm creates no duplicate scorer job.
5. A targeted `1.25...e-4|m110|pose_budget` follow-up in pc1, pc2, gs3, and qbw2 found
   gs3's explicit m110 correction at lines 146-148. This changed the pose arithmetic
   from an asserted inherited threshold to a labeled interpretation of this charter's
   S-unit wording. The proof outcome is independent of that unresolved acceptance-set
   choice, so no task is blocked on clarification and no candidate is admitted.

Research-index/DAG mentions of Gabor/directional features do not supply the missing
finite-language/pose acceptance certificate for this live archive. No such certificate
was found in these searched sources. Older context is used as scoped source evidence,
not as a current frontier number.

## Verification, integration, and remaining boundaries

The audit ran with one CPU process and one numeric thread. No Metal, scorer, GT decode,
training, remote dispatch, candidate search, or candidate archive build. `df -h` before
writes showed Vertigo 66 GiB and APDataStore 38 GiB free; only small retained evidence
and a sparse isolated Git checkout were created. No artifact cleanup or live-tree edit.

Controls: independent shipped CAP1/CPR1 reader identity for all 7,200 codes, all 27,648
basis symbols, and all 24 scales; six exact finite-count cases; four invalid-count
refusals; full manifest resume success; deliberate corrupted-hash manifest refusal.
The initial verification invocation used an incorrect reader entrypoint name and failed
before comparison; it was corrected to the source's `decode_compact_carrier`. No failed
attempt supplied a result. The audit source was reviewed twice after its final edit.
The review caught and corrected a basis-symbol zigzag interpretation before any audit run.

Adversarial checks: a covering radius has the wrong inequality direction for an
individual-distance floor; 600 rows are one sequence; fp32 realization is not a real
linear lattice; a short lower bound is not a witness; raw field bytes are not compressed
marginal bytes; fixed-point solver evidence is not exhaustive exclusion. Violating the
shared assumption that a new basis must reproduce the old field could unlock a different
pose-equivalent witness, so no generated family is retired.

`generated_carrier_basis_lattice_floor_v1` is registered through
`tac.canonical_equations.register_canonical_equation` in the isolated store, with the
actual finite-cover callable, source-inspection provenance, and explicit non-admission
domain. The source-array receipt anchors constants, not empirical proof of a pose floor.
Lane is L0 research-only; no production/scorer maturity gate is claimed. Integration
hooks for sensitivity, Pareto, byte allocation, and deployment are N/A: no score-changing
vehicle was produced. The equation and closure event are the consumers of this proof.

Composition: this result is orthogonal to live coder/token improvements. Keeping the
existing basis with a counted generated residual, per-pair fallback, or another fully
specified representation may remain viable; none is priced here. Those possibilities
cannot resurrect the closed counting inference. gb1's open `VALID_BOUND_INPUTS` gate is
FOLDED into this vacuous-formulation result rather than kept as an indefinite demand for
another identical proof attempt. A new construction needs a new, specified contract.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer `.omx/state/canonical_task_status.jsonl`, `.omx/state/canonical_equations_registry.jsonl`, `.omx/state/lane_registry.json`.** On harvest of the retained gb2 commit/bundle, import the memo and code, replay this arm's equation/lane/task events without overwriting concurrent state, and consume the gb1 `VALID_BOUND_INPUTS` debt as FOLDED into gb2's vacuous-formulation verdict. Do not dispatch a generated-basis build from the zero bound. The exact payload and event manifest are under W.

## LIVE-HYPOTHESES

- A fixed, generic generated family with counted selected parameters could yield a cheaper
  pose-equivalent witness: the evaluator constrains six outputs per pair, not the entire
  old carrier field. No admissible point or affordable payload has been demonstrated.
- A counted residual/fallback combined with generic atoms could avoid a few badly placed
  lattice cells. pc2's heterogeneous step effects make localized benefit plausible;
  the benefit and cost on cmp2 are untested.

## DEAD-ENDS

- **FORMULATION:** deriving an individual generated-program byte floor from the stored
  Huffman/Rice length or an ensemble count. They have different quantifiers.
- **FORMULATION:** treating a covering radius, span projection plus rounding loss, or
  local fixed-point failure as a lower bound on the realized pose optimum. None supplies
  the required integer/global/realized certificate.
- **FORMULATION:** granting fitted SVD atoms free status via a seed. The fitted information
  must reach the receiver in counted bytes.
- **FORMULATION:** firing BUILD because a lower bound lies below the byte cap. A vacuous
  zero bound proves neither existence nor affordability. All three families stay unadjudicated.

Own-vehicle frontier: **S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600],
unchanged by ddm_gb2.** Read from the live pointer and reproduced from the existing
report-8dp component mirror; this arm performed no new scoring.
