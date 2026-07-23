# DDM DR2 SCC crosswalk, outside view, and R-D ladder re-scope

Date: 2026-07-23  
Lane: `ddm_dr2_scc_outside_view`  
Lineage: v10/DDM describe line only  
Mode: `research_only=true`, `$0`, local CPU receipt analysis; no scorer run, launch, archive build, or pointer write  
Score claim: false  
Promotion eligible: false  
MAIN landing review required: true

## Verdict

**The SCC ideas are useful as original DDM implementation patterns, but the
measured exact per-record coder is dominated and the critical path is not
proven optimal.** Six records select causal track and five select absolute
re-key, yet a favorable schema-free envelope costs **70,700 B**, or **+2,236 B**
against the SHA-pinned 68,464 B SDWL1 row. More decisively, all eleven streams
fail the operator rate doctrine: scorer visibility is unproven, tolerances are
exact rather than sensitivity-priced, compact DOFs are unmeasured, the
arithmetic models are not cross-stream conditional, and all 110 ordered-pair
outer-zlib marginal probes show positive overlap. The existing 68,464 B fact
description is rate-feasible relative to the exact-C1 154,524 B cap only as a
counterfactual description-side fact; the aggregate inventory is not an
invertible RGB receiver and therefore does not establish sub-0.15 feasibility.

Pointer delta: **0**. Canonical pointer remains
`0.1910828242 [contest-CPU]`; this lane did not evaluate or modify it.

## Authority and stores consulted

The delegated prompt SHA-256 is
`7de8863d5e28224d1afb08fa7a9c6ce037ce75c5242e9759e1947f36eac6746c`
(5,777 B). Every live input below was consumed by content identity, not by
filename alone.

| Store | SHA-256 | Use |
|---|---|---|
| `ddm_dv1_description_vocabulary_n600_20260723T141407Z/receipt.json` | `e0c875346978f8768eedf96793bcff2fb472a37eba4ebbcec7254d107fc00333` | dv1 vocabulary and receiver-closure limits |
| `ddm_dv2_sdwl1_n600_20260723/receipt.json` | `efc43fcda1f12f28df2b6059cd5e51e7ee2509a356d99b59e317b253927a709c` | exact 68,464 B control and semantic inventory |
| `direct_description_g1_grammar_induction_20260722.json` | `f4cdfbc3765c4dbaaa9383741f4b9d2f8d4eb5ab411595fad100ec0373d263e5` | grammar knees and union negative |
| `ddm_g2_solve_diff_op_mining_n600_20260722T194000Z/receipt.json` | `ada87717b39bc34ad67a3104d652574e544d82938fa3a1ea898acdf624c2bd67` | inner-Jacobian blocker |
| `ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_receipt.json` | `6c4157092a7bdf7ba44b458cd470725cc470d84a8fc77ed7d3dedb59160734f5` | pair/stratum sensitivity evidence |
| `ddm_g4_spatial_stationarity_n600_20260722T212138Z/ddm_g4_spatial_stationarity_receipt.json` | `bea555b95aeaa11f4209df5333010c41c5495dd789def2a4f7a2a91973f3408c` | stationarity and wedge opportunity |
| `ddm_c1_composed_candidate_ledger_603_613_20260723.json` | `14fdf1570b43df65ac949fe157e68ea328ff584f7df1331acf25cca8f900d936` | current composed control and exact-set verdict |
| `ddm_m4_rate_floor_einstein_avenue_20260723_receipt.json` | `e74ae0079733a4a3babf46cef23f0583b143d6015e7b32a656690ea1ce724972` | strict sub-0.15 caps and scope |
| `xi_temporal_delta_coder_574_20260721T222234Z.json` | `7241ff90057287d58482e9abd55e5da17173cb6ebe09a39c2ca847cc02a9a8b9` | #574 xi-coder scope and still-open solved-object/stream families |
| `reports/latest.md` | `2c8e987723e68c7b7efcf058776c4052fc4b8990cab733194a7f9923350c291d` | canonical pointer context |
| `canonical_task_status.jsonl` | `5a0b6bd1218540686dee6f5c0f05536ad38196801fbbff567a7b22d1abbb470b` | queue comparison |

The isolated worktree does not materialize `upstream/modules.py`. I therefore
checked comma.ai's public `modules.py` at repository commit
`5387a097398ec6581c7e4e428231e1821fc62670` (raw bytes SHA-256
`065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`,
8,322 B). It establishes the only evaluator facts used here: SegNet takes only
the last frame, produces a five-class argmax, and measures disagreement;
PoseNet consumes the resized two-frame YUV6 stack and measures the first six
pose coordinates. No scorer weights, predictions, or private target values
were used.

## Premise correction before design

The prompt's “Road static-in-image 99.1%, 0% xi-proxy” premise is mislabeled.
The G4 receipt reports:

| G4 region | static in image | static in xi proxy | Scope |
|---|---:|---:|---|
| all flips | 98.8063% | 0.0000499% | target-cache cell recurrence |
| boundaries | 98.6645% | 0.0001755% | target-cache boundary recurrence |
| lane corridor | 97.7796% | 0% | target-cache region |
| movable band | **99.1493%** | 0% | source of the cited 99.1% |

G4 contains no Road-only 99.1% row. Moreover, pixel recurrence is not record
constancy: each of the eleven SDWL1 aggregate records has 600 unique states and
599 adjacent changes. Static exact coding is therefore inadmissible for every
current record. This does not refute a future *persistent primitive plus sparse
innovation* decomposition; it refutes only the present whole-record static
formulation.

## A. Screen-content-coding crosswalk

### Source facts, not borrowed formats

The HEVC-SCC overview identifies palette coding and intra-block copy as core
screen-content tools [S1]. The AV1 specification makes palette mode conditional
on screen-content tools, codes a palette index map with causal contexts, and
allows IntraBC to reference already decoded current-frame regions [S2]. VVC GPM
splits an inter-predicted coding unit by a quantized angle-plus-offset line,
selects predictor pairs and the partition by rate-distortion search, and
supports hard or soft blending [S3]. Wedgelets supply the related dyadic,
multiscale piecewise-constant edge dictionary [S4].

Those facts motivate our designs. **No codec bitstream, syntax table, source
code, normative constant, or claimed codec performance enters DDM.** Every
adopted element below is re-derived from the frozen evaluator, openpilot/road
geometry already named by dv1, or our level-set/Morse-Smale and score-byte
math.

| Rank | Imported idea, re-derived implementation | Disposition | Named consumer | Our-original provenance line | `$0` measurement on existing evidence | First rung |
|---:|---|---|---|---|---|---|
| 1 | Five-label palette as a typed semantic alphabet; causal neighbor/previous-section contexts for indices and numeric productions | **ADOPT** | `dv2` coder successor | Public `modules.py`: five SegNet argmax classes and last-frame-only loss; SDWL1 already owns typed class rows. Implement a new DDM context model, not AV1 syntax. | Recode the same SHA-pinned tensor while conditioning row \(B\) on decoded row \(A\); strict semantic parse-back and complete-byte race against 68,464 B. | Start with the ordered pair having the largest measured positive overlap; admit only if complete outer bytes fall and the doctrine audit closes. |
| 2 | Causal copy/reference plus residual for repeated semantic sub-vectors, boundary fragments, or persistent primitives | **ADOPT** | `dv2` coder successor | Decoder-causal copy is derived from SDWL1's deterministic row order; displacement and residual are charged. Road-chart/xi provenance remains the dv1 openpilot geometry registry. | Search exact repeated sub-vectors and translated bbox/cut tuples across earlier decoded records. Race `reference+residual` against track/re-key with exact parse-back. | Build an exact-match-only probe first; whole-record copy is already ruled out by 600/600 unique states. |
| 3 | Hard angle-plus-offset wedge atom with exact score-byte RDO | **ADOPT** | `dv1` primitive set, then `e1` receiver | dv1's level-set separatrices imply codimension-one curves; G4 independently measured a two-line Lane wedge. The atom and RDO objective are ours. | Use the existing G4 masks/receipt to race hard wedge parameters against sparse rules; do not infer RGB survival. Existing SDWL1 cut counts alone do not identify orientation. | Make wedge location receiver-derivable, then run one exact `R`-closed byte race against the same correction target. |
| 4 | Dyadic multiscale wedge dictionary with escape/residual for non-straight boundaries | **ADOPT, AFTER RANK 3** | `dv1` primitive set | Our level-set energy demands local boundary curvature; dyadic support is an original discretization of that derived geometry. | Enumerate a finite dyadic support/angle bank on the existing G4 boundary target, charge atom ids and residuals, and compare complete bytes. | Prove one finite-bank atom has negative exact reduced cost after the hard-line rung; otherwise stop this bank, not the family. |
| 5 | One persistent semantic worldsheet with topology events rather than independent local RGB blocks | **ALREADY BETTER AT DESCRIPTION SEMANTICS; RECEIVER OPEN** | `dv1`/`dv2` | dv1 Morse-Smale cells, separatrices, pair screw, and G1 topology productions are global and typed across n600. | No remeasurement: SHA-pinned dv1/dv2/G1 already settle description coverage. The owed test is receiver sufficiency, not another grammar inventory. | Land a description-completeness/identifiability certificate against `e1`; a semantically elegant non-invertible inventory is not a witness. |
| 6 | Per-unit RDO, but with the exact contest action rather than generic pixel error | **ALREADY BETTER IN OBJECTIVE; EXECUTION PARTIAL** | `dv2` successor and `e1` receiver | \(100d_{\rm seg}+\sqrt{10d_{\rm pose}}+25B/37{,}545{,}489\) comes from the contest evaluator; c1's score-byte dual supplies the local price. | Use the current mode-race harness for rate, then require realized-through-\(R\) distortion for admission. | Add sensitivity-priced tolerances and a common receiver master before expanding the mode bank. |
| 7 | Literal HEVC/AV1/VVC bitstreams, RGB palette entries, normative mode tables, and soft GPM blending | **N/A in this formulation** | none | DDM describes scorer-relevant semantics; importing a standard format would violate “techniques in, formats out.” Soft blending is also not justified for sharp argmax boundaries without an RGB receiver. | None. A future receiver may independently rediscover a useful RGB prototype, but it must be priced and derived from our scorer surface. | N/A |

### Ranked conclusion

The highest-EV SCC transfer is not “encode masks with a standard codec.” It is
**joint conditional mode selection over one decoder-owned semantic state**:
palette-like typed alphabets, IBC-like causal reuse, and wedge-like boundary
atoms compete inside one exact score/byte decision. The current SDWL1
serialization implements only independent arithmetic sections plus an outer
compressor, so rank 1 precedes adding more atoms.

## B. Per-record RDO mode decision

### Measured mode table

`tools/measure_ddm_dr2_scc_outside_view.py` reads the frozen
`stage_10_fact_inventory.json`, verifies payload bytes/SHA and semantic SHA,
races exact `track` against exact absolute `re_key`, rejects static unless all
600 states match, and refuses `xi_advect` because no decoder-free transport map
from xi to the aggregate facts exists. `track` means first absolute state plus
causal deltas; `re_key` is the whole-record absolute arithmetic control.

| Stratum / record | Unique / changes | static | xi-advect | track B | re-key B | Selected |
|---|---:|---|---|---:|---:|---|
| Road cell | 600 / 599 | inadmissible | unpriced/unmapped | 8,948 | 11,255 | track |
| Lane cell | 600 / 599 | inadmissible | unpriced/unmapped | 10,666 | 10,863 | track |
| Undrivable cell | 600 / 599 | inadmissible | unpriced/unmapped | 8,495 | 10,448 | track |
| Movable cell | 600 / 599 | inadmissible | unpriced/unmapped | 9,208 | 11,144 | track |
| MyCar cell | 600 / 599 | inadmissible | unpriced/unmapped | 8,160 | 10,494 | track |
| Road separatrix | 600 / 599 | inadmissible | unpriced/unmapped | 8,213 | 7,055 | re-key |
| Lane separatrix | 600 / 599 | inadmissible | unpriced/unmapped | 7,866 | 5,621 | re-key |
| Undrivable separatrix | 600 / 599 | inadmissible | unpriced/unmapped | 7,511 | 7,033 | re-key |
| Movable separatrix | 600 / 599 | inadmissible | unpriced/unmapped | 7,253 | 5,568 | re-key |
| MyCar separatrix | 600 / 599 | inadmissible | unpriced/unmapped | 7,040 | 6,279 | re-key |
| Pose pair screw | 600 / 599 | inadmissible | unpriced/unmapped | 15,322 | 16,350 | track |

Selected inner streams total 92,355 B. A strict measurement-only envelope with
row/mode/length/SHA framing is 92,866 B and zlib9-compresses to **70,700 B**
(`6a69aed0f991b91be757cd2e38950ff3fbdcac0481aed2481b594b56555ffde8`).
It restores semantic SHA
`e7dee11d0fd162470bb206acca3c4667c79100cc41259d5d1ecb293e31e225f3`
exactly. It omits the SDWL1 lexicon and schema, making the comparison favorable
to the challenger, yet loses to 68,464 B by **2,236 B**.

Verdict:
`DOMINATED_EXACT_CODER_LAYER_CONTROL`. This is scoped to the exact frozen fact
tensor and the declared measurement framing. It is not a negative on
sensitivity-quantized, copy-conditioned, wedge-augmented, or receiver-joint
mode selection.

### Binding four-clause rate audit

| Clause | Result | Consequence |
|---|---|---|
| Scorer visibility | **FAIL_UNPROVEN** | Aggregate facts have no RGB receiver and no per-DOF quotient against Seg/Pose null space. |
| Sensitivity-priced tolerance | **FAIL_EXACT_ONLY** | Every scalar is exact; no margin, G3 pair-lambda, or c1 dual quantization is applied. |
| Descriptive / compact / coder layers | **FAIL_INCOMPLETE** | Descriptive rows and coder bytes are known; irreducible DOF counts after gauge/null quotient are not. |
| Single-owner fact rule | **PASS within the declared tensor only** | Each of 45,600 scalar coordinates has one row owner; ownership against a future receiver/correction stream is open. |
| Cross-stream conditional coding | **FAIL_UNCONDITIONED** | Each row carries a self-contained arithmetic model. All 110 ordered-pair zlib marginal proxies show positive overlap; maximum 137 B, non-additive positive sum 8,716 B. |
| Dimension homes | **FAIL_UNMEASURED** | All 600 per-pair states are coded; persistent clip mass versus innovations/events is not separated. |
| Corrections are deltas | N/A | This control emits no correction stream. A future `e1` manifest must prove it. |

The pairwise probe is explicitly a detector, not mutual information:
\[
\rho_{A\rightarrow B}
=B_{\rm zlib}(B)
-\left[B_{\rm zlib}(A\Vert B)-B_{\rm zlib}(A)\right].
\]
Positive \(\rho\) detects overlap left by the current composition; zero would
not prove independence, and negative values would not earn rate credit. The
complete 110-row matrix is in
`ddm_dr2_scc_outside_view_mode_race_20260723.json`.

### Recovered obligations

**#574 consumed-by:** this table is the owed per-record
`static | xi-advect | track | re-key` admission surface. #574 measured a
separate S4 research-corpus xi temporal formulation, explicitly left the solved
object and several stream families open, and did not establish a decoder-free
xi transport for SDWL1 aggregate facts. DR2 therefore consumes the obligation
with a scoped result: `track` and `re_key` have real exact bytes; `static` is
empirically inadmissible; `xi_advect` is
`BLOCKED_NO_XI_TO_AGGREGATE_FACT_MAP_OR_PRICED_SIDE_INFORMATION`. No published
shape-bit saving prior is credited as a measurement.

**Einstein-Kolmogorov U5 partial coverage:** DR2 supplies a real per-record
layout/coder race over the exact SDWL1 inventory, with strict parse-back and a
complete outer-byte comparison. It does **not** fire the G4/G5
sensitivity-quantized, receiver-aware coder races. U5 is therefore
`PARTIALLY_COVERED_EXACT_LAYOUT_RACE`; the scorer-tolerance and receiver rows
remain `UNFIRED`.

## C. Fresh-team outside view

### What two fresh teams would build

**Conventional codec team.** Starting only from public `modules.py` and these
receipts, it would split the problem by evaluator responsibility: a sharp
five-label, last-frame semantic plane for SegNet and a paired motion-preserving
carrier for PoseNet. The semantic plane would use a small alphabet, causal
spatial index contexts, decoded-region copy, geometric wedges for hard
boundaries, and a residual escape. Every block/unit would race static, copy,
motion, wedge, and residual under a task-aware RDO cost. It would preserve only
the reference state needed by the decoder and would force pair/frame ownership
to avoid coding Seg facts in frame 0.

**Divergent first-principles team.** It would not begin with pixels or blocks.
It would search for the smallest complete coordinate system for an evaluator
equivalence class: one persistent kinetic cell complex, one stored xi serving
both Pose and Seg transport, sparse topology events, a deterministic
evaluator-inverse receiver, and sensitivity-priced correction deltas. The
optimization variable would be the complete description itself, with exact
archive bytes and realized-through-\(R\) Seg/Pose in the same master. A
description would be admitted only with a sufficiency certificate: two source
states mapping to the same description must be equivalent for the frozen
evaluator after decode.

### Diff against the live lines and queue

| Rank | Outside-view requirement | Live describe / correct / descend coverage | Queue coverage | Gap verdict and first rung |
|---:|---|---|---|---|
| 1 | **Description completeness / evaluator-sufficiency certificate** | dv1/dv2 describe aggregate cells, cuts, moments, boxes, and pose bits, but these do not identify masks or RGB. The D2 receiver blocker independently agrees. | Receiver work is queued, but no typed certificate proves the chosen fact inventory is sufficient. | **NEITHER HOLDS THE CERTIFICATE.** First rung: construct two distinct masks with equal SDWL1 facts or prove impossible; on a collision, extend the description only with the minimal receiver-visible distinguishing coordinate. |
| 2 | **One conditional mode compiler with single-owner export manifest** | Current mode choices and correction receipts are separate; this lane measured overlap across all exact rows. | The new operator clause requires the manifest, but no canonical queue row owns its implementation. | **MISSING IMPLEMENTATION.** First rung: add the ordered-pair redundancy matrix and owner/dimension columns to the `e1` candidate manifest, failing closed on unconditioned overlap. |
| 3 | **Semantic-domain causal copy dictionary** | G4 measures recurrence; dv2 has causal deltas; neither performs decoder-causal sub-vector/region reference plus residual. | No active task names this finite exact-match probe. | **NEW HIGH-LEVERAGE PROBE.** First rung: exact-match-only sub-vector copy on the frozen inventory; no approximate copy until scorer tolerances exist. |
| 4 | **Hard-wedge atom inside the same receiver RDO** | G4's 27 B two-line Lane wedge is cell-space positive but RGB survival is unknown. dv1 has separatrix semantics. | The idea is present, but no receiver-joint finite-bank task is registered. | **HELD AS IDEA, NOT EXECUTION.** First rung: one hard wedge in `e1`, exact complete-byte and joint-\(R\) race; do not expand the bank first. |
| 5 | **Common complete-description master over rate and both distortions** | c1 composes measured parts; v19b corrects realized states; #366 descends a receiver surface. No one row spans a complete SDWL1-derived receiver with exact code length. | #604/#366 aim at this end state. | **HELD, PARTIAL.** First rung: bridge description variables to one receiver and reuse one n600 master; do not spawn a duplicate global solver. |

### Adversarial optimality challenge

The current critical path is **locally rational but not proven globally
optimal**:

1. c1 certifies infeasibility only for its exact computed set. Its 133,941 B
   control still has `d_seg=0.027470296224` and `d_pose=163.061327281443`;
   composition has not closed the target box.
2. dv2's 68,464 B is a compact description of aggregate facts, not a complete
   witness. Optimizing that coder before proving receiver sufficiency can
   polish the wrong object.
3. The favorable per-record exact mode race loses, and all ordered stream pairs
   expose overlap. Independent record coding is therefore not the next
   score-moving rung.
4. The requested tolerance ladder has no lawful map from fact error to
   realized scorer error. Exact serialization is the wrong default under the
   operator doctrine.

Refutation result: **“more exact SDWL1 coder search, then receiver” is not
optimal.** The highest-EV next rung is a *receiver-joint sufficiency and
sensitivity quotient*: prove which description DOFs the scorer sees, assign
each one a single home, quantize it to local tolerance, and only then race SCC
modes. This supports—rather than displaces—the live `e1`/global-master
direction; it changes the gate order.

## D. U1 R-D tolerance ladder re-scope

Let the settled exact-C1 distortion anchor be
\(d_{\rm seg}^{(1)}=0.00015196\) and hold the settled C1 pose anchor
\(d_{\rm pose}=0.00010184\) only for a counterfactual budget calculation. The
largest integer byte budget under 0.15 at multiplier \(m\) is

\[
B_{\max}(m)=\left\lfloor
\frac{37{,}545{,}489}{25}
\left(0.15-100m d_{\rm seg}^{(1)}
-\sqrt{10d_{\rm pose}}\right)
\right\rfloor.
\]

| Tolerance target | Target \(d_{\rm seg}\) | Real SDWL1 description bytes at that tolerance | Derived sub-0.15 cap if pose is held | Honest read |
|---|---:|---:|---:|---|
| exact / 1x | 0.00015196 | **68,464 B exact-fact control** | 154,524 B | Description-side headroom 86,060 B, but no receiver maps those facts to the anchor distortion. **Not feasibility.** |
| 2x | 0.00030392 | **UNKNOWN** | 131,702 B | Reusing exact 68,464 B would be under the counterfactual cap by 63,238 B, but no lawful \(Q_\tau\) or receiver distortion exists. |
| 5x | 0.00075980 | **UNKNOWN** | 63,238 B | The exact-fact control is 5,226 B over this cap; a tolerance coder must save at least that much *and* realize the target. Neither is measured. |
| 10x | 0.00151960 | **N/A for sub-0.15** | negative | \(100d_{\rm seg}=0.15196>0.15\) before Pose or rate. This rung is provably outside the requested box. |

This is the first **non-vacuous rate-side** read because a real 68,464 B coder
exists and can be compared with positive, exact caps. It is not the requested
end-to-end feasibility ladder. The remaining negative is formulation-scoped:

`BLOCKED_NONINVERTIBLE_FACT_INVENTORY_NO_SENSITIVITY_QUANTIZER_OR_RGB_RECEIVER`.

The aggregate cell records contain area, coordinate sums, boxes, and component
counts; separatrix records contain cut and margin-band counts. Many masks can
share those summaries. Without a deterministic inverse and local
fact-to-scorer tolerance, coding altered facts would manufacture an
unverifiable distortion claim.

First rung: define a typed \(Q_\tau\) whose per-DOF bins come from head
flip-distance/margin fields, G3 pair lambdas, and the c1 score-byte dual; attach
it to a complete deterministic receiver; then run the unchanged real coder and
exact `R` replay at 1x/2x/5x. The 10x box rung remains excluded by arithmetic.

## Triality

### DSL leg

```text
DR2RecordModeDecision {
  record_id,
  descriptive_form,
  scorer_visibility_status,
  sensitivity_tolerance_status,
  compact_dof_count,
  single_owner,
  dimension_home,
  mode in {static, xi_advect, track, re_key},
  complete_stream_bytes,
  ordered_pair_redundancy[],
  parseback_semantic_sha256,
  candidate_admissible
}
```

Admission requires all doctrine fields closed, exact parse-back, a negative
complete score/byte delta through the receiver, and a MAIN-reviewed receipt.

### DAG leg

```text
SHA-pinned dv1/dv2/G1-G4/c1/M4 + public modules.py
                 |
                 +--> SCC primary-source crosswalk
                 |          |
                 |          v
                 |   original DDM mode candidates
                 |
                 +--> frozen SDWL1 inventory --> exact record race
                 |                                  |
                 |                                  v
                 |                       four-clause doctrine audit
                 |                                  |
                 |                     dominated control, no candidate
                 |
                 +--> sufficiency collision/proof --> e1 receiver
                                                    |
                                                    v
                                  sensitivity Q_tau + common exact-R master
                                                    |
                                                    v
                                         U1 1x/2x/5x ladder
```

### Equation leg

\[
m_r^*=\arg\min_{m\in\{\mathrm{static},\xi\text{-advect},
\mathrm{track},\mathrm{rekey}\}} B_r(m)
\]

subject to exact parse-back for this control; candidate admission additionally
requires

\[
\Delta\!\left(
100d_{\rm seg}+\sqrt{10d_{\rm pose}}
+25B/37{,}545{,}489
\right)<0
\]

after parse-back, RGB realization, uint8/evaluator \(R\), and both frozen
scorers. A hard wedge primitive is

\[
w_{\phi,\rho}(x,y)=
\mathbf 1[x\cos\phi+y\sin\phi\ge\rho],
\]

but its parameters have authority only when derived or stored once, charged,
and selected by the same complete action.

## Papers and standards checked

- **[S1]** Xu et al., “Overview of the Emerging HEVC Screen Content Coding
  Extension,” IEEE TCSVT, DOI
  [10.1109/TCSVT.2015.2478706](https://doi.org/10.1109/TCSVT.2015.2478706);
  author publication page:
  <https://www.merl.com/publications/TR2015-126>.
- **[S2]** Alliance for Open Media, *AV1 Bitstream & Decoding Process
  Specification*: <https://aomediacodec.github.io/av1-spec/av1-spec.pdf>.
- **[S3]** Gao et al., “Geometric Partitioning Mode in Versatile Video Coding:
  Algorithm Review and Analysis,” IEEE TCSVT, DOI
  [10.1109/TCSVT.2020.3040291](https://doi.org/10.1109/TCSVT.2020.3040291).
- **[S4]** Donoho, “Wedgelets: Nearly Minimax Estimation of Edges,” *Annals of
  Statistics*, DOI
  [10.1214/aos/1018031261](https://doi.org/10.1214/aos/1018031261).
- **Evaluator source:** comma.ai public
  [`modules.py`](https://github.com/commaai/comma_video_compression_challenge/blob/5387a097398ec6581c7e4e428231e1821fc62670/modules.py).

## MAIN review boundary

MAIN must review before landing:

1. the prompt-premise correction (Movable, not Road);
2. the mode definitions and +2,236 B comparison;
3. the ordered-pair redundancy proxy's limited interpretation;
4. the distinction between description-side rate headroom and receiver-closed
   sub-0.15 feasibility;
5. the claim that the next gate is sufficiency/sensitivity before further exact
   coder expansion; and
6. the #574 consumed-by and U5 partial-versus-unfired classification; and
7. that no old-lineage format, number, code, or pointer mutation entered this
   v10/DDM research arm.
