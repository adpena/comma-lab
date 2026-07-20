# Einstein--Kolmogorov crux: shortest low-action preimages of the frozen scorer

UTC: 2026-07-19T21:44:08Z  
lane_id: `lane_einstein_kolmogorov_crux_20260719`  
research_only: true  
axis: `[macOS-CPU advisory]`  
pointer: `0.1910828242 [contest-CPU Linux x86_64]`, UNMOVED  
review_status: fresh-eyes-reviewed(0); first recursive round found class defects and reset the seal  
verdict_scope: CONSTRUCTIVE-DESIGN plus complete n24 palette/lossy-label component probe; end-to-end archive still blocked

> **CURRENT MEASUREMENT, NOT YET A VERDICT SEAL.** Every numeric n24 row below was
> replayed after the reproducibility-closure and projected-bound class fixes from
> committed source `889d32db399709aca475c3c6120c81f189dd4b81`. Recursive clean-pass
> review is still pending, so the clean count remains zero.

## Answer first

**THE CRUX (DERIVED, with formulation-scoped MEASURED support):** find the shortest
counted program whose free decoder emits **any** uint8 witness in a low-action preimage of
the frozen scorer.  This is a rate--distortion covering/inverse problem, not exact
recovery of a quotient representative.  Realization through `R` is the binding wall for
the measured PDW1/PDW2 explicit-target charts; it is not a universal theorem that every
winning decoder must materialize their quotient coordinates or preserve every target
label.  A shorter program may decode directly to RGB and deliberately trade a few label
cells or Pose coordinates when the exact action improves.

**THE STRONGEST COMPOSED CANDIDATE SHAPE (DERIVED; not a uniqueness claim):** one
restricted chart codes `W=(G, xi, T)`--spatiotemporal class-boundary grammar `G`, one
shared SE(3) twist trajectory `xi`, and the smallest receiver-active texture/metamer
field `T`--then lets the free generic decoder expand those symbols to scorer planes and
choose a canonical exact uint8 preimage with the already-landed lattice solver.  `G`
owns topology and boundaries, `xi` owns both temporal transport and PoseNet, and `T`
supplies only the winner-rival margin needed where a flat cell cannot survive the
nonlinear SegNet trunk.  A direct counted-payload-to-RGB decoder is a co-equal chart and
is not required to expose `G`, `xi`, `T`, or exact target labels.

**THE COMPLETED SCOPED UNIT (MEASURED through hard R):** the PDW1 packet already spends
15 raw fill bytes per pair.  Changing
those bytes does not change packet length.  The existing direct-plane positive control
replaced per-pair source means with one global `5 x RGB` table and reduced n24 hard
mismatches from `38,077` to `37,276` (`d_B 0.0080695682 -> 0.0078998142`) at the same
19,859-byte packet size.  The receiver-closed, resumable within-family tournament built
below selected projected middle-point DSPSA-32 followed by coordinate-12: `26,795` mismatches,
`d_seg=0.0056786007351345485`, the same 19,859 bytes, strict parse-back, and exact
factor-2 realization on all 24 pairs.  A final fully fingerprinted replay reproduced
the candidate and metrics.  A separate lossy-topology family proved that short-run
simplification can exchange distortion for rate (`36,876` mismatches, 19,371 B at rung
2), but grafting it onto the fill winner raised the local action by `0.0150610065` and
was rejected.  These numbers are not a score and do not include a compact pose-closed
frame-0 stream.

## Stores consulted

STORES CONSULTED: `tools/corpus_query.py` and `tools/graph_memory_recall.py` over
PDW1 realization, PDW2 spatial nonidentifiability, Fisher/margin, inner-Jacobian,
curvelet/shearlet, reverse-waterfill, and Kolmogorov anchors; `CLAUDE.md`;
`AGENTS.md`; `docs/operating_manual_craft_handoff.md`; the v7.5 operating contract;
the v8 and v10 vehicle specifications; the latest Codex findings/session summaries;
the latest T3/design memos; the Claude top-10 memory; the anti-re-research ledger;
`generator_description_online_survey_20260719.md`;
`generator_description_crux_synthesis_20260719.md`;
`mpeg4_shape_coding_intake_and_crosswalk_20260719.md`;
`yhat_rd_ladder_20260719_codex.md`;
`rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`; the VCM/task-aware
survey cluster and its `p_suff`, d_seg, and pose receipts;
`sol_ultra_v10_true_final_form_review_20260717.md`;
`solve_the_right_problem_kolmogorov_sweep_20260718.md`; SPEC v10; the PDW1/PDW2
code and receipts; the exact scorer factorization; and the canonical-equation sources
named below.  Deliberately not consumed as authority: the retracted terminal finding,
invalidated round-1 curvelet ranks, MPS scores, or any memo number that could be cheaply
re-derived from code/bytes.  External fresh intake was restricted to the two primary
SPSA/DSPSA sources below and Frank Nielsen's full 41-page *Short stories on Bregman
divergences* deck, after loading prior #504/#550/#552, because the anti-re-research
ledger already covers the broad codec and geometry families.  The external Claude memory ledger is outside this
worktree's write authority, so the new-paper intake is durably recorded here for MAIN
to mirror rather than silently attempting an out-of-scope write.

## 1. One invariant and the Einstein-frame change of coordinates

Let `U` be the finite set of legal uint8 two-frame RGB witnesses and let

```
Q(u) = ( argmax SegNet(R u_1), PoseNet(YUV6(R u_0, R u_1)) ).
```

For a legal counted bitstring `c` and scorer-free deterministic decoder `D`, the contest
problem is

```
min_(c,D) 100*d_seg(D(c)) + sqrt(10*d_pose(D(c))) + 25*|archive(c,D)|/37,545,489.
```

Equivalently, each byte budget asks for the shortest program that hits a set-valued
low-distortion preimage under `Q`; exact `U / ~_Q` sections are only the zero-distortion
subcase.  The decoder must remain scorer-free.  In the explicit-target chart, the exact
factor-2 lattice realizes a requested uint8 scorer plane but does not choose a plane
inside the nonlinear SegNet cell; PDW2 names a compact quotient geometry but supplies no
decoder-computable spatial field.  These establish the realization wall for that chart,
not for every possible direct decoder.

Evidence labels:

- **MEASURED:** yhat rung B is 83,838 bytes / 139.7 bytes per pair, inside the
  216--264 KB pointer box, yet has `d_seg=0.0034556919` and repeat-frame pose near 63.
- **MEASURED:** PDW1 has `d_A=0`; its arithmetic labels equal `L*`, while real
  receiver re-scoring has `d_B=0.0080695682`, 38,077 n24 pixels.
- **MEASURED:** PDW2's 138-byte raw / 133-byte Brotli coefficient head is consumed
  but spatially non-identifying without an RGB/spatial consumer.
- **MEASURED:** the shared-resize decomposition puts roughly 52.4--52.9% of the
  current rendered energy in `ker(A)`; this is evidence of wasted coordinate energy,
  not a claim that the archive can automatically lose the same fraction of bytes.
- **DERIVED:** the invariant is the conditional description length of a low-action
  preimage, `min_{u in A_tau} K_D(u)`, for the declared legal decoder class `D`.  Exact
  section complexity is one restricted zero-distortion chart, not the global invariant.
- **UNKNOWN:** the exact instance Kolmogorov complexity of the 600-pair quotient.
  It is not computable from these receipts and is not claimed.

The Einstein-frame move is to optimize in the scorer's rate--distortion cover rather
than RGB fidelity.  The strongest currently composed explicit-target chart is:

```
W = (G, xi, T)
  G  : class cells, edges, saddles, births, and their inter-pair transport grammar
  xi : one SE(3) twist trajectory shared by temporal prediction and pose realization
  T  : receiver-range metamer values sufficient to clear winner-rival margins
```

This composes, rather than re-opens, the following registered laws:
`frozen_scorer_exact_factorization`; `resize_exploit_flip_fix_frontier_v1`;
`segnet_head_rank4_linear_flipdist_v1`;
`bounded_uint8_resize_preimage_cell_feasibility_v1`;
`realization_necessity_preimage_per_stratum_v1`;
`necessity_generator_seed_dseg_calibration_v1`;
`scorer_obligation_matrix_factorization_v1`;
`lane_band_ego_factorization_source_reparam_v1`;
`instant_projected_input_adjoint_v1`; `flip_margin_step_law_v1`;
`witness_measured_reverse_waterfill_v1`;
`meta_lagrangian_dual_solver_per_axis_kkt_residual_v1`; and
`cgauge_master_action_v1`.

### Cross-family evidence, not a fake common tournament

The operator-mandated family search is represented by the strongest receiver evidence
already in custody plus this lane's new within-family probe.  These rows have different
scopes and cannot be ranked as if they were a sealed same-input A/B:

| family/chart | receiver-closed evidence | honest disposition |
|---|---|---|
| direct counted payload -> RGB (`yhat` rung B) | **MEASURED** 83,838-byte full archive; fresh n24 diagnostic `d_seg=0.0034556919`, `d_pose=63.0309157` | Direct-generator and lossy-cell control: it bypasses explicit `Q/G` labels, but Pose makes it noncompetitive and the row is n24 diagnostic only |
| explicit fixed labels + palette (PDW1) | **MEASURED here** 19,859-byte n24 component; winner `d_seg=0.0056786007`; Pose unmeasured | Within-family palette probe complete; not a full archive |
| lossy row-run topology (PDW1) | **MEASURED here** source-fill rung 2: 19,371 B, `d_seg=0.0078150431`; graft on optimized fills worsens local action | Genuine rate--distortion control, but dominated by the fixed-label scoped winner |
| quotient/power packet (PDW2) | **MEASURED** 138 raw / 133 Brotli bytes and consumed; spatially non-identifying | Formulation-scoped blocker until a decoder-computable field/pullback exists |
| `xi` temporal/Pose chart (settled R1) | **MEASURED prior** full-n600 macOS advisory: 89,772 B, `d_seg=0.0045491197`, `d_pose=0.0016095472` | In-box control, not a new result; packet bytes were deleted by the settled run |
| curvelet/shearlet residual chart | **MEASURED prior** saved-OFF n600 through-R basis rows, but equal scalar support rather than equal bytes | Not archive-rankable; receiver-bound target inverse and exact bytes remain absent |
| MPEG-4 INTER-CAE / arithmetic self-compression | prior literature and local rate evidence only | Open design family; no new receiver-closed row in this lane |
| Bregman/Fisher dual chart | full Nielsen deck plus #504/#550/#552 and exact code derivation | Exact quadratic `R`-fiber projection and already-measured global-centroid control; no categorical pullback or byte theorem (§3.1) |

There is consequently **no defensible cross-family winner**.  The only new selected winner
is inside the fixed-label PDW1 palette family.  The full multi-family and new full-n600
mandates remain unmet because the first executable `xi` composition fails the mandatory
SSD custody preflight before loading the backend; the failure is a local execution-surface
blocker, not evidence against the unmeasured families.

## 2. Constructive minimum-description program

### Counted stream

1. A versioned manifest with dimensions, deterministic seed, grammar version, and
   hashes of the generic decoder contracts.
2. `G_0`, a first-pair power-diagram/contour description plus its necessary
   receiver-evaluated spatial field.  Temporal pairs use `xi`-warped prediction and an
   arithmetic-coded INTER-CAE-style boundary correction.  Birth/death side information
   is charged; no source-derived fact is hidden in code.
3. `xi`, quantized with a self-derived tolerance: choose the coarsest quantizer whose
   exact PoseNet replay remains within the score-optimal pose tube.  The same `xi`
   transports `G`; it is not duplicated in a second temporal predictor.
4. `T`, initially the existing raw `5 x RGB` fill table per pair.  A later residual is
   admitted only on the measured necessary edge/saddle set and only if a compact
   shearlet/curvelet packet beats the score-per-byte waterline through the real receiver.
5. Entropy-code all video-derived symbols.  Seeded generic algorithms and receiver code
   remain free under rule 118; learned/video-derived values are counted.

### Free deterministic decoder

1. Strictly parse and hash-check the stream; reject trailers and unconsumed sections.
2. Decode `G`, advect it by `xi`, and apply charged boundary/birth corrections.
3. Expand `G` with `T` to a uint8 scorer-plane proposal.
4. Apply the registered first-order + secant + bounded-QP correction in the
   winner-rival Fisher/margin metric.  A naive first-order step is forbidden because the
   registered realization gap measures the missing inner Jacobian.
5. Project the requested scorer planes through the exact bounded integer preimage
   solver.  Use canonical support-fill tie-breaking so decode is bit-identical.
6. Place pose actuation primarily in frame 0 (SegNet-free), with luma/chroma duties
   separated by the frozen factorization; preserve every stage output atomically.
7. Emit the exact required video and perform archive parse-back in the build path.
   The decoder itself never calls SegNet or PoseNet.

### Control laws (no TBD knobs)

- **Loss:** `S=100*d_seg + sqrt(10*d_pose) + 25*B/37_545_489`.
- **Byte cap:** for target `S_t`,
  `B_max=floor((S_t-100*d_seg-sqrt(10*d_pose))*37_545_489/25)`; refuse if the
  parenthesis is non-positive.
- **Admission:** accept a change iff exact replay gives
  `Delta S = 100 Delta d_seg + Delta sqrt(10 d_pose) + 25 Delta B/N < 0`.
- **Stop:** stop a rate-bearing family when its marginal non-rate improvement per byte
  is at or below `25/N = 6.658...e-7` score/byte.  A zero-byte fill replacement may be
  accepted only when its deterministic hard mismatch count strictly decreases.
- **Target order:** act on necessary edge/saddle strata first, then by smallest
  resize-realizable Fisher/winner-rival distance; never spend on necessity zero.
- **Basis:** a receiver-bound residual may use compact shearlet or genuine curvelet
  atoms, never Fourier.  The final 2026-07-14 saved-OFF advisory ranking does not select
  a basis because it compared equal scalar support, not equal bytes, and lacked the
  target-boundary inverse.
- **Projected middle-point DSPSA variant:** per pair, bounded integer domain `[0,255]^15`, Bernoulli `+/-1`
  perturbations from the declared seed, two exact function evaluations per iteration,
  middle-point rounding, projected update, and best-so-far preservation.  `a_k` follows
  `a/(A+k+1)^alpha`; `alpha=0.602`, `A=max(1, floor(0.1 M))`; choose `a` once so the
  first median nonzero component update is one byte.  If no component is nonzero, make
  no update.  Coordinate descent is the deterministic control.
- **Checkpoint:** atomically preserve config/fingerprint, PRNG state, current and best
  fills, completed pair/iteration, objective rows, input hashes, stable implementation
  dependency hashes (including dynamically imported `upstream/modules.py`), base git
  head, and runtime/package/thread closure after every pair.  Resume recomputes and
  refuses any mismatch.

Fresh method citation at point of use: James C. Spall, 1992, *Multivariate stochastic
approximation using a simultaneous perturbation gradient approximation*, IEEE
Transactions on Automatic Control, DOI `10.1109/9.119632` (two measurements estimate a
full perturbation direction).  The integer middle-point and bounded projection are from
Qi Wang, 2013, *Optimization with Discrete Simultaneous Perturbation Stochastic
Approximation Using Noisy Loss Function Measurements*, arXiv `1311.0042`, Chapter 2.
This implementation additionally projects the persistent real iterate to the declared
half-integer interior, so it is an honest bounded projected variant rather than a claim
of source-trajectory identity.  Wang's convergence conditions have not been established
for this nonconvex scorer loss; the method is a tournament arm, not a guaranteed optimum.

### Byte arithmetic

At the exact v10 spine distortion (`d_seg=1.5196e-4`):

| target / pose assumption | derived total byte cap | status |
|---|---:|---|
| pointer 0.1910828242, pose 0 | 264,150 B | DERIVED from exact score law |
| pointer 0.1910828242, `d_pose=1.0184e-4` | 216,223 B | DERIVED |
| target 0.15, pose 0 | 202,451 B | DERIVED |
| target 0.15, `d_pose=1.0184e-4` | 154,524 B | DERIVED |

The 83,838-byte generator fits every positive cap above.  It does **not** satisfy any of
them at its measured `d_seg=0.0034556919`; at that distortion the allowed byte term is
already negative even for the pointer.  This does not license a distortion-first
sequence: distortion is spendable whenever the exact rate saving wins.  The admissible
endpoint is the joint Seg/Pose/rate KKT waterline measured on full exact `S`, and same-pool
opportunities must not be added.  The compact counted `xi` and receiver-bound `T` byte
totals remain UNKNOWN until a full n600 archive is built and parsed back.

## 3. Honest K-floor argument

Kolmogorov complexity supplies the correct objective but not a computable certificate
for this particular instance.  The only honest two-sided statement is:

- **MEASURED:** the existing direct-generator archive has `archive_bytes=83,838` and the
  cited SHA-256/diagnostic receipt.
- **DERIVED program-length upper bound:** any complete byte-closed archive is a prefix
  program, so its exact archive length plus the fixed interpreter constant upper-bounds
  the description length of the witness it generates.  The 83,838-byte archive therefore
  bounds its own decoded low-fidelity scorer view, not the desired low-distortion cover.
- **CONDITIONAL lower bound:** after quotienting decoder-free symmetries, a decoder that
  must distinguish `M` admissible scorer cells requires `ceil(log2 M)` bits in the worst
  case.  Under a declared quotient source model, average prefix length is bounded below
  by its conditional entropy.  The edge-transition, birth/death, texture-choice, and
  pose-tube symbols must therefore be counted once unless deterministically implied by
  preceding symbols.
- **DERIVED coordinate-invariance guard:** on a declared finite serialized domain, a
  gauge-fixed Legendre map `eta=grad F(theta)` with computable inverse is a computable
  bijection, so `K(eta)=K(theta)+O(1)`.  Dual coordinates do not lower the instance floor
  by themselves.  Only quotienting a real invariance or a measured quantization/entropy
  gain can lower counted bytes; Task #550's 20/19-scalar construction is a gauge quotient.
- **UNKNOWN instance floor:** neither `M` for the realized target family nor the
  algorithmic randomness deficiency of this one video has been measured.  No exact
  `K(target)` or proof of global minimality is asserted.

The algorithmic-information definition follows A. N. Kolmogorov, 1965, *Three
approaches to the quantitative definition of information*, Problems of Information
Transmission 1(1) (English reprint DOI `10.1080/00207166808803030`).  The conditional
expected-code lower bound uses Claude E. Shannon, 1948, *A Mathematical Theory of
Communication*, DOI `10.1002/j.1538-7305.1948.tb01338.x`.  The new statement here is
the composition with this frozen scorer quotient; no external paper is claimed to prove
the contest-specific rate--distortion-covering statement.

The measured `ker(A)` fraction explains why RGB-space complexity is the wrong upper
bound: those coordinates cannot change either frozen scorer before later nonlinearities.
It is not itself a lower bound.  The residual gap above the conditional floor lives in
duplicated temporal description, full spatial label maps, non-necessary cell interiors,
and texture values that do not buy a receiver-closed margin.

### 3.1 Bregman scope: exact R-fiber projection, not scorer-cell projection

For each disjoint resize block let `a_j>0`, `sum_j a_j=1`, and define
`F_R(x)=1/2 sum_j a_j ||x_j||^2`, with a positive quadratic term on unowned
coordinates.  For an integer scorer-plane target `y`, the landed canonical support
fill uniquely solves

`argmin_{x in [0,255]^n, A x=y} B_{F_R}(x || 0)`,

and the minimizer is already uint8, so it is also the lattice minimizer.  Realization
is therefore an exact quadratic Bregman projection onto the fixed linear `R`-fiber.
It is not a categorical/Fisher projection onto a nonlinear SegNet argmax cell: the
target plane has already been selected and the realizer neither calls the scorer nor
optimizes a scorer-cell objective.

The global `5 x RGB` table is likewise the deterministic round/clip of each class's
arithmetic RGB mean, hence a lattice quadratic-Bregman centroid.  The theorem does not
promise a hard-label gain; the fresh real-decode row supplies that fact:
`38,077 -> 37,276` mismatches at 19,859 component bytes.  Frank Nielsen's 2026
*Short stories on Bregman divergences: Flat and curved geometries*, DOI
`10.13140/RG.2.2.35617.77929`, was read from the author-hosted 41-page deck pinned at
commit `b01cdbb9e4a807980f29e0fe615f37322167cd01` (PDF SHA-256
`d2031206b0f680cea6bfd85c2403eddc351310263253228535b25851eb39e7f4`).

## 4. $0 verification charter and implemented specification

The implementation adds, without modifying the archival experiment:

1. `src/tac/witness_dsl/einstein_kolmogorov_crux_20260719.py`: frozen typed config
   for input fingerprints, pair set, family, seed, iterations, gain schedule, bounds,
   singleton scorer geometry, and checkpoint cadence.  Validation rejects implicit
   paths, nondeterministic seeds, non-singleton authority claims, and unknown families;
   it includes a deterministic lossy-label run-length control.
2. `src/tac/optimization/einstein_kolmogorov_crux.py`: pure score arithmetic,
   deterministic coordinate candidates, projected DSPSA state transition, strict
   best-so-far admission, JSON round-trip, and resume-fingerprint helpers.  No Torch.
3. `tools/probe_einstein_kolmogorov_crux.py`: explicit-path CLI; source packet is
   read-only; all outputs use a new directory.  It must execute
   `PDW1P encode -> fresh decode -> re-encode identity -> plane expansion -> canonical
   factor-2 uint8 realization -> factor-2 certificate -> frozen singleton CPU-Torch
   SegNet -> L* mismatch`, streaming one camera frame at a time.
4. `tests/test_einstein_kolmogorov_crux.py`: canaries and mutation-sensitive tests for
   config refusal, deterministic perturbations, bounded projection, no-regression
   admission, score/byte waterline, checkpoint round-trip, source/runtime mutation
   refusal, packet parse-back, and deterministic simultaneous label simplification.

Pinned n24 inputs for the run:

- packet `/Volumes/VertigoDataTier/pact/evidence/pdw1_realization_20260719/pdw1p_n24_payload.bin`,
  19,859 B, SHA-256
  `8d66286c722e9202d77b54bcdf06942496641bad20ff9632c785800b53a0a02d`;
- L* `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n24.npz`;
- SegNet `/Users/adpena/Projects/pact/upstream/models/segnet.safetensors`, SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b`.

Tournament order and falsifiers:

| arm | purpose | pre-registered falsifier |
|---|---|---|
| source per-pair means | in-run baseline | refuse unless 38,077 mismatches and 19,859 B reproduce |
| zero palette | negative canary | meter invalid if it does not worsen baseline |
| global 5xRGB palette | positive control | reject direct-plane claim if hard-R replay does not improve |
| exact `+/-1` coordinate descent | deterministic local control | stop a sweep after zero accepted moves |
| projected DSPSA | fresh discrete-search arm | reject if best-so-far is worse, bounds fail, or same seed diverges |
| lossy label run 2/3/4 | separate topology/rate family | measure actual bytes; reject any graft that raises exact ordering action |
| Fisher/secant-QP palette | optimal-form follow-on | blocked until logits/inner-Jacobian custody is emitted in-run |
| shearlet/curvelet residual | structured residual follow-on | blocked until receiver-bound bytes and target-boundary inverse exist |

Every fixed-label candidate preserves exact packet size because PDW1 stores the 360 fill
bytes raw.  Lossy-label controls deliberately change encoded length.  Every arm measures
actual bytes and parse-back identity; neither property is inferred from source inspection.

## 5. Completed unit, next unit, and gates

The n24 **palette plus lossy-label component** tournament is complete at $0.  It is not
a common-scale full-archive tournament and its output is a component candidate: it has
no compact pose-compatible frame 0 and covers only 24 pairs.

Promotion chain after a positive n24 result:

1. rerun the same receiver-closed palette law on all 600 pairs with preserved per-pair
   checkpoints and exact batch/runtime custody;
2. compile the chosen palette/structured residual into the 83,838-byte generator
   receiver rather than shipping PDW1's full label maps;
3. add the compact `xi` frame-0/frame-1 pose stream and measure exact PoseNet;
4. build `archive.zip`, parse it back, run byte-identical inflate twice, and measure
   official `upstream/evaluate.py` on contest-CPU Linux x86_64;
5. only then compare to 0.1910828242 or claim an in-box full witness.

Step 2 remains blocked by the missing optimized-fill/residual section in LVLS1.  A typed
cross-checkpoint xi bridge closes the pose-composition call surface, but governed n600
execution failed before loading its 5.08 GB GT cache because this sandbox cannot write
the mandatory SSD evidence tier.  Even with filesystem access, the backend now refuses
until a hash-bound contract proves resume-from-disk and preserved atomic per-stage
checkpoints.  These are execution-surface blockers, not xi-family negatives.  Zero-cost
local full work is authorized; only paid dispatch remains operator-GO gated.  The pointer
is unchanged.

## 6. Build and measurement result

The complete typed within-family palette tournament was run through
`PDW1P encode -> fresh decode/re-encode -> plane expansion -> factor-2 uint8 realization
-> exact realization certificate -> frozen singleton CPU-Torch SegNet -> L* mismatch`.
Every arm covered all 24 packet pairs and preserved an immutable checkpoint after every
pair; DSPSA also preserved every integer iteration.

| arm | hard mismatches | d_seg | bytes | disposition |
|---|---:|---:|---:|---|
| source means | 38,077 | 0.008069568210 | 19,859 | final fingerprinted control |
| zero palette | 2,388,341 | 0.506155437893 | 19,859 | negative canary passed |
| global palette | 37,276 | 0.007899814182 | 19,859 | final fingerprinted positive control |
| coordinate-3 | 35,807 | 0.007588492499 | 19,859 | dominated/cap-limited |
| coordinate-12 | 32,509 | 0.006889555189 | 19,859 | dominated |
| projected middle-point DSPSA-8 | 30,667 | 0.006499184502 | 19,859 | repeated byte-for-byte |
| projected middle-point DSPSA-32 | 28,039 | 0.005942238702 | 19,859 | hybrid parent |
| DSPSA-32 then coordinate-12 | **26,795** | **0.005678600735** | **19,859** | winner; final fingerprinted replay |
| label-run 2, source fills | 36,876 | 0.007815043132 | 19,371 | improves source rate--distortion; dominated by winner |
| label-run 3, source fills | 37,728 | 0.007995605469 | 18,419 | dominated |
| label-run 4, source fills | 50,535 | 0.010709762573 | 17,207 | negative rung |
| winner then label-run 2 | 27,521 | 0.005832460192 | 19,371 | graft rejected: `Delta S_local=+0.0150610065` |

The winner removes 11,282 errors (29.6294%) from the source and 10,481 (28.1173%)
from the global control with zero additional packet bytes.  Its fixed-byte action delta
relative to the source is `-0.2390967475`; Pose cancels algebraically but remains
unmeasured.  The pose-zero value `0.5810833665` is therefore local ordering arithmetic,
not a contest score.  The lossy-label rung demonstrates the operator-directed joint-RD
move, but its best graft pays more Seg score than its 488-byte rate saving buys, so the
exact component waterline rejects it rather than banking a rate budget.

No new full-n600 in-box point was produced.  The existing settled R1 89,772-byte
macOS-CPU advisory row remains a control, not this lane's result.  The attempted xi
composition created neither packet nor receipt: the strict SSD preflight raised
`PacketOutputFilesystemCustodyError(classification=permission_denied)` and no bulky
local/tmp fallback was used.  A separate read-only receipt reverified all three input
hashes, including the 5,078,017,610-byte GT cache.  Exact custody and the heterogeneous cross-family evidence inventory are
in `.omx/research/einstein_kolmogorov_crux_measurement_20260719.json`; the executable
state transition is in `einstein_kolmogorov_crux_DAG_FEED_20260719.md`.

## Triality / system-intelligence wiring

- **DSL:** the typed config above owns the only legal local probe configuration.
- **Equations:**
  `src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py` now provides
  `einstein_kolmogorov_crux_action_rate_contract_v1`: exact action arithmetic, byte-cap
  refusal, fixed-byte palette delta, and typed receipt-to-decision edges.  It binds the
  exact aggregate measurement through canonical provenance and an empirical anchor;
  registration remains research-only and cannot promote the scoped winner.
- **DAG:** `FEED-EINSTEIN-KOLMOGOROV-20260719` must record the low-action-preimage obstruction,
  the n24 tournament receipt, the exact blocker to full archive composition, and the
  pointer non-delta.
- **Unified action:** Seg, pose, and exact bytes are admitted only through the existing
  `cgauge_master_action_v1`; no proxy objective may promote a candidate.
- **Sensitivity / bit allocation:** fill changes are zero-byte substitutions; any later
  residual is ordered by the Fisher/margin necessity field and reverse-waterfill law.
- **Autopilot:** research-only until a full parse-back archive and contest-axis scorer
  row exist.  No dispatch hook is armed by this memo.
- **Receiver-consumption bijection:** any downstream redesign must parse and consume
  every counted section through `R`; a mutation canary must prove each counted field is
  active.  Counted-but-inert bytes are forbidden, as are scorer weights or source-derived
  payload hidden in decoder code.
