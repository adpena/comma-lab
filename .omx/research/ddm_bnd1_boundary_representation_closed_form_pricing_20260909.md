# ddm_bnd1 — boundary pricing blocked by source and proof mismatches

`[no-triality] [p0-ledger-ok]` · owner `ddm_bnd1` · 2026-09-09.
`research_only=true`; `score_claim=false`; `promotable=false`.
Lane: `lane_ddm_bnd1_boundary_representation_20260909`.

**The source audit is complete; the requested segment pricing and draw verdict are not.**
The charter's 83 KB is an ideal coding-cost attribution, not a retained substream;
its 12,377 cells belongs to unshipped pass 5, not move 37. The supplied constants
do not determine a new segment draw's error. Verdict: **BLOCKED_SOURCE_PREMISES**,
scope **CHARTER_PROOF_PREMISES_ONLY**. No representation family is closed.

## Results and denominators

Axis: **[macOS-CPU scorer-free source audit]**. All newly counted candidate symbols
were checked against the retained pass-4 field associated with the shipped body.
This is a retained-field check, not a fresh public receiver decode.

| Quantity | Result | Authority and limitation |
|---|---:|---|
| Native token population | 117,964,800 | 600 × 384 × 512; RP1 ledger |
| Coder mispredictions | 235,044 | RP1 total minus coder-argmax matches |
| Retained unique misprediction locations | 213,733 | Fresh NPZ count; all 600 pairs; every stored symbol matches pass 4 |
| Missing misprediction locations | 21,311 (9.0668%) | Rank dump excludes savings below 0.25 bits |
| Retained tokens on a GT edge | 209,179 | Source census, not recomputed GT geometry |
| Full-population on-edge fraction | **at least 88.9957%** | Conservative bound 209,179 / 235,044; every omitted location may be off-edge |
| Whole retained mixer envelope | **119,784 B** | Actual bytes, identity-control SHA verified |
| Misprediction ideal-cost attribution | **83,258.63239658909 B** | Sum of `-log2(integer_frequency/TOTAL) / 8`; not an isolated serialized object |
| Shipped pass-4 residual cells | **12,614** | SJ1 pass-4 source measurement; not newly scored here |
| Unshipped pass-5 residual cells | **12,377** | Pass-5 `flips_after`; cannot be joined to the move-37 population as one field |
| New scored / drawn pairs | **0 / 0** | Draw grammar unpriced; scorer step queued under the common contract |
| New archives / exact evaluations | **0 / 0** | No build or dispatch |

Source receipts and payload hashes: [source_audit.json](ddm_bnd1_20260909/source_audit.json).
Producer: `experiments/ddm_bnd1_source_audit.py`, SHA-256
`9a2f190de466c779763d4f864317aebd97f3bbeba8a7475be0a5274d5a29e307`.
Retained final receipt:
`/Volumes/APDataStore/pact/ddm_bnd1_boundary_representation/retained/SOURCE_AUDIT.9a2f190de466c779.json`,
SHA-256 `d039d4e657c9cdf274873fe393824d7dfa47dd9f980fdf062e7d95907e237c1d`.

## Verified-at-source corrections

1. **83 KB is not the requested actual price.**
   `experiments/ddm_rp1_rate_rank.py:700–713` normalizes the integer frequency
   table, takes `-np.log2(freq)`, and sums the selected entries. The integer table
   is the coder's table, but its logarithmic attribution is still not a separate
   emitted payload. `experiments/ddm_rp1_mispredicted_census.py:153–156` converts
   that sum to bytes. RP1's full ideal length is 119,778.02311134017 B; the retained
   envelope is 119,784 B. That close agreement validates total accounting, not
   separability of the 83,258.632 B or a boundary entropy floor.
2. **Pass 5 is not the shipped field.**
   `ddm_cmp2_t4_sm1_semantic_coder_20260909_pointer_move_37_20260909.md:31`
   says move 37 preserves the renderer, field and tail. SJ1's pass-4 admission
   field hash is `813bf1e6770161b604348491433386661110001a5eefd8a4fdcd7d70bc25365c`;
   pass 5 is `48b5852e3c3adc3482de66375e98bc1771c34bb47db8c6f234593e1be9d3b9bf`.
   SJ1 memo §§20–21 and `passes/pass5_gt/PASS_RESULT.json:40–59` identify
   12,614 before and 12,377 after 237 repairs. The pass-5 count and approximately
   0.0105 S are not a move-37 incremental error allowance.
3. **The live demand is correctly sourced.**
   Move-37 memo:33–39 gives **26,908.6 B** at held distortion; exact seg term
   **0.010698 S**, pose term **0.007106335201775948 S**. The common contract's
   older frontier paragraph is superseded by the canonical pointer.
4. **The proposed closure is not a valid inference.**
   An explicit code of length L establishes `optimal_length <= L` for its stated
   decoder/side information. `L >= 78 KB` cannot establish that every admissible
   representation costs at least 78 KB. Neither can an empirical distribution's
   entropy prove a lower bound on a single structured object's shortest program.
   Qbw2:5–15 explicitly distinguishes its achievable code from a Shannon floor;
   gb1:74–83 rejects the same upper-bound-as-lower-bound error for a solver.

## Census geometry and actual-price obligations

The following counts were independently recomputed from the retained NPZ.
They describe **213,733 locations**, not all 235,044. Payloads
`retained/census_counts.npz` and `retained/census_counts.jsonl` preserve the counts,
including 600 per-pair counts and the full 5 × 5 stored-class/coder-best matrix.

| Row interval | Retained mispredictions |
|---|---:|
| [0,128) | 0 |
| [128,256) | 182,296 |
| [256,320) | 31,436 |
| [320,384) | 1 |

| Stored class | Coder best 0 | Coder best 1 | Coder best 2 | Coder best 3 | Coder best 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 48987 | 17553 | 9328 | 6426 |
| 1 | 69584 | 0 | 156 | 282 | 255 |
| 2 | 17669 | 108 | 0 | 9831 | 5 |
| 3 | 9489 | 154 | 12907 | 0 | 38 |
| 4 | 10677 | 263 | 8 | 13 | 0 |

This matrix is a **stored-symbol → coder-argmax transition table**, not a histogram
of the two GT classes sharing an edge. GT class-pair edges, distance histograms,
tangent runs and the joint residual-cell intersection remain unmeasured. The
source's on-edge count is explicitly distinguished from those missing products.

| Requested price | Actual retained bytes | Status |
|---|---:|---|
| Shipped coder on the full token envelope | 119,784 | Verified existing object; not solely mispredictions |
| Shipped coder's isolated misprediction description | unknown | 83,258.632 B is an ideal attribution, not this price |
| Class-pair polyline + per-segment normal offsets | unknown | Grammar, decoder side information and complete population not established |
| Same grammar with offsets in {-1,0,+1} px | unknown | Same blockers; quantization alone does not define a code |

DOF ratio: **unknown**. No numerical segment floor is reported.
GT edges cannot be treated as free decoder information. A causal representation
must charge starts, junctions, class pairs, topology, gaps, offset symbols and any
learned model, then reproduce its own parse-back. The remaining token stream and
its contexts must also be priced; subtracting 83 KB from the archive without that
joint accounting would double-count an attributed saving. Hc1:29–55 distinguishes
its acausal boundary statistic from decoder-visible causal contexts.

## Draw-error derivation: what is and is not determined

Let T be the shipped token field, C a completely specified segment code and
`Tq = draw_q(C)`. Let `A(T)` be frozen SegNet argmax after the **actual** receiver
renderer and R, and G the DALI-lineage target. Define
`E(T) = count(A(T) != G)`. For n600, N = 117,964,800:

`delta_S_seg = 100 * (E(Tq) - E(T)) / N`.

One net error cell is **8.477105034722222e-7 S**. If H counts cells whose argmax
changed, `abs(E(Tq)-E(T)) <= H`; H alone does not say whether the changes repaired
or introduced errors. These identities do not supply H from offset quantization.

| Supplied constant | What the source measured | Why it cannot price this draw |
|---|---|---|
| bz2d 1.1572382972400272 | Total d_seg / total token error on an ancestor GT-fit field at 1.123% token error (`ddm_bz2d_distortion_verdict_20260830.md:52–63`) | Not a derivative or uniform Lipschitz bound for a different local draw on the shipped field |
| rw1 240–455 cells/code step | Global renderer int4 weight edits; 240 extrapolates a 48-cell 120-pair screen result (`ddm_rw1_boundary_local_renderer_weight_foldback_20260909.md:520–535`) | Weight edits and local spatial offsets are different maps; renderer-weight draws are expressly forbidden |
| msr1 8.94% | Historical 2,123 / 23,757 global-actuator accounting; the manufactured population is 21,493, with separate global/per-pair/per-pixel addressing results (`ddm_msr1_manufactured_seg_reduction_20260823.md:13–22,242–265`) | Not a theorem limiting a segment-selective actuator |
| lb1 8.94× | Perfect Lane oracle on n32 QBF1-born labels (`ddm_lb1_lane_band_carrier_ceiling_on_born_field_20260904.md:7–10,40–65`) | Not a shipped-field local-offset measurement or universal accuracy multiplier |

Rasterize-into-tokens must measure `A(draw_q(C))`; offset-the-render must separately
define and measure the spatial image operation through R. Neither error curve is
identified by the two historical ratios. They may inform a labelled hypothesis,
but multiplying them into a claimed closed-form verdict would be false precision.
The approximately 0.0105 S is an entire historical seg contribution, not permission
to add that much error. Admission depends on the **net** Seg/Pose/rate change of
the same object. Seg-only n12 evidence cannot certify pose or an exact score.

## Preregistered 12-pair draw comparison

Seed **20260909**, NumPy PCG64, uniform without replacement over all 600 pairs:
**6, 52, 61, 82, 158, 275, 343, 455, 469, 499, 519, 589**.
The seed, RNG version and binding archive hash are retained in
`retained/DRAW_SELECTION.json`. Each pair has explicit null predicted/measured
entries in [draw_comparison.jsonl](ddm_bnd1_20260909/draw_comparison.jsonl).

| Pairs | Predicted flips | Measured flips | Status |
|---|---|---|---|
| All 12 listed above | unavailable | not run | Unpriced draw; scorer allocation not granted to this arm |

The common contract directs arms without a scorer allocation to queue this step.
No scorer was launched, including no n600 job. Once dependencies are resolved,
use only the receiver's renderer, frozen cpu_torch SegNet, DALI lineage, at most
two processes, nice 10, no Metal, and retain **all drawn token fields**. Any detached
launch must use the named launcher with `--nice 10 --nice-best-effort`. Keep the
12-pair outcome a prediction/measurement check; it cannot close a family or bank
an n600 score. No scorer-holder identity is written into a new charter.

## PRIOR-LAW PREDICTION versus evidence

| Prediction | Evidence now | Disposition |
|---|---|---|
| >=80% within 2 cells of a GT edge | Source on-edge subset alone gives >=88.9957% of all 235,044 | Supported conditional on retained source geometry; no fresh distance histogram |
| Tangent run >=4 for >=60%; 4–10× fewer DOF | Unmeasured; old OF1 had mean arclength 2.88 | Live hypothesis, not an inferred consequence of proximity |
| Segment price 25–50 KB / rate gain 20–40 KB | No charged grammar or actual segment payload | Untested |
| Seg side <=1,100 cells by the old 8.94% ceiling | Old global-actuator result does not transfer | Unsupported cap rejected |
| Joint floor S 0.125–0.130 | No same-object price/draw/pose closure | Not established |
| >=78 KB proves boundary entropy and closes door | Achievable code size has the opposite bound direction | Invalid closure rule rejected |

## RECALL EVIDENCE

Read the charter, common contract, PROGRAM, AGENTS/CLAUDE, operating handoff,
live hot state and canonical pointer. Queried the checkpoint before work and
used `tools/graph_memory_recall.py "boundary representation offset field class pair"`.
The retained graph result is [graph_recall.json](ddm_bnd1_20260909/graph_recall.json).

Beyond the charter seeds, content searches covered `.omx/research/` memos and
receipts, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, SPEC/design files and the
canonical task ledger. Queries included `polyline|normal.offset|boundary.segment|class.pair (cod|edge)|boundary (represent|grammar|cod)`,
`contour|boundary|spatial|codec`, and
`boundary representation|normal.offset|polyline|boundary.segment|PE3|OF1|ddm_bnd1|gb1`.
The live equation CLI exported **480 equations**; the identifier query
`boundary|contour|offset|polyline|edge|tail.*floor|real.*count|hc1|tc1|qbw2|gb1`
matched **15**. The selected records and full-export hash are retained in
[equations_recall_selected.json](ddm_bnd1_20260909/equations_recall_selected.json).

| Beyond-seed source | Finding | What changed |
|---|---|---|
| `ddm_of1_offset_field_and_flicker_coherence_20260729.md:58–109` | Ancestor residual within 1 px but mean arclength 2.88, 1.14–3.28 flips/DOF, 18% contour coverage | Proximity does not establish long coherent segments |
| `contour_string_flip_coding_n600_20260707.md:19–45` | Actual 361,953 B, anchors 162,934 B (45%), mean component 3.1 | Charge starts and connectivity, not just cheap direction symbols |
| `residual_kit_deshare_curverel_build_20260709.md:14–19,52–78,124–130`; `src/tac/boundary_math/curve_relative_offset_coder.py:1–38` | Existing `(segment,s,n)` coder only 0.99× horizon / 0.90× Lane; normal is axis-aligned | Do not rediscover it or label it true-normal geometry without repair |
| `ddm_pk1_20260805/PK1_RECEIPT.md:11–15,44–61,88–91` | Older retained 74,408 B grammar; 8,644 components; rendered seg 0.00660216 and pose 0.08182466 advisory | Byte size below 78 KB is possible on an older object, but does not establish an admissible draw |
| `ddm_or1_orthogonal_representation_regime_20260826.md:111–143` | Instance row-start packet 140,377 B for 352,297 starts vs old 113,624 B token stream | Address cost is material; not a universal segment floor |
| `ddm_gt2_gt_tongue_induction_20260803.md:235–261` | Lane contour 130,960 B lossless; 0.5-px option 129,864 B with 24,048 errors | Quantization must carry its actual error; old entropy-labelled result was corrected |
| `ddm_de1_description_efficiency_derivation_20260803.md:76–164` | 1.27310821533203125 B/repaired cell, pose separately | Compare score per byte on the joint object |

Material registry recalls: `leverd_flicker_residual_reactivation_economics_v1`,
`v8_geometric_rate_decomposition_v1`, `token_tail_context_mixing_bound_v1`.
No valid move-37 universal segment lower bound was found in this bounded search.

## Equations, retention, verification and completion boundary

`boundary_segment_description_floor_v1`: **FORMALIZATION_PENDING**, not registered.
Registering a numerical floor from an incomplete population or achieved code
length would violate NO FAKE. ITEM 1 owns resolution; this is an explicitly
unfulfilled charter deliverable, not a substituted equation under the same name.

All source-arm reads were read-only. No upstream, live receiver, pointer, sj1 or
rp1 content was changed. Bulky retained inputs and census NPZ live only in the
assigned APDataStore tree. SSD free-space preflight and a 64 MiB bounded-audit
guard passed. Every retained input/payload has size and SHA in the audit receipt;
the full equation export was moved losslessly to SSD under a hash/command
manifest. No evidence was discarded. Interrupted temporary writes remain
preserved; unique attempts permit resume.

Two independent clean code review passes and both tracker marks cover the
final producer hash. Validation verified candidate identity on all retained
locations, count consistency, the existing exact-score arithmetic, matching
resume, preservation of a stranded partial, rejection of changed retained bytes,
and replay after a simulated Git HEAD change. [validation.json](ddm_bnd1_20260909/validation.json)
and [reviews.json](ddm_bnd1_20260909/reviews.json) preserve the receipts. The first
audit invocation's `nice -n 10` reported `setpriority: Operation not permitted`;
this short scorer-free source audit still completed. No scorer launch used an
unverified priority setting.

Tasks A and the source-validation portion of B are complete. Full B, numerical C,
D, the E build/closure decision, and F's floor registration remain blocked. The
memo and partial census are retained. Neither a BUILD charter nor a family
closure is warranted by these inputs. Landing status is recorded separately by
the serializer receipt; no unrelated shared-state edits are part of that landing.

Serializer outcome: **NOT LANDED; no fallback bundle created**. Attempt 1 refused
three gitignored diagnostic logs (rc=13); those logs were retained on SSD and
excluded from attempt 2. Attempt 2 supplied post-edit hashes for 17 files. Git
object insertion failed with `Operation not permitted`; the automatic fallback
then refused the charter-assigned APDataStore tier (rc=19): 40,410,808,320 free B
against the 42,949,672,960 B reserve plus 602,482 projected artifact B. The refusal
is preserved at `.omx/state/commit_serializer_fallback_refusals/20260909T224625.734998Z-992/receipts.jsonl`.
The requested bundle is an outstanding deliverable. No reserve was lowered and
no fallback writes were rerouted outside the charter's arm tree. Commit hooks
did not run because object insertion failed; the code-review and audit-validation
receipts above are the completed checks. The final source patch and hashes are
retained for MAIN's landing; a patch is not called a commit or a Git bundle.

## ITEM 1 — Repair the population, causal pricing and proof premises

Disposition **QUEUED-WITH-A-FIRE-ORDER**; owner `ddm_bnd1`, coordinator MAIN;
consumer store `.omx/state/canonical_task_status.jsonl`, task suffix `::ITEM_1`.
Fire on MAIN's harvest of this receipt. Rebind to the then-live pointer and obtain
the complete coder-misprediction locations plus matching shipped residual maps;
do not join pass 5 to pass 4. Start with a scorer-free retained-ledger export if
available; any missing scorer-produced input must join the scorer queue. Define
and charge the causal grammar and decoder-visible side information, retain real
serialized alternatives and their parse-back checks, and price the whole tail
including containers. A valid lower bound needs an explicit admissible class
and proof; otherwise report achievable prices only. Resolve the equation's
FORMALIZATION_PENDING status only with a valid claim and census anchor. This is
one proof/pricing prerequisite, not authorization for training or an archive build.

## ITEM 2 — Run the retained 12-pair draw after the pricing prerequisite

Disposition **QUEUED-WITH-A-FIRE-ORDER**; owner `ddm_bnd1`, scorer coordination MAIN;
consumer store `.omx/state/canonical_task_status.jsonl`, task suffix `::ITEM_2`.
Fire only after ITEM 1 supplies a fully specified, genuinely affordable causal
draw, predictions and retained actual prices, and MAIN grants scorer allocation.
Re-read the pointer; use the preregistered pair IDs and the resource/retention
constraints above. Report predicted versus measured flips and signed error
changes. Complete the scoped E decision from that evidence; any BUILD charter
must charge all payload/container/remaining-coder effects and queue resumable
n600 joint validation before an exact-score claim. Do not infer a formulation
floor from failure of one codec or a universal draw failure from this screen.

The generator successor is **FOLDED**, not a newly spawned row, into existing
`ddm_gb1_generated_carrier_basis_lattice_priced_closed_form_20260909::VALID_BOUND_INPUTS`
(owner MAIN, consumer `.omx/state/canonical_task_status.jsonl`). Its existing
proof-input/scorer-allocation fire order remains in force; this audit supplies
no reason to retry the invalid rounded-residual lower bound.

## ITEM 3 — Land the reviewed source-audit handoff

Disposition **QUEUED-WITH-A-FIRE-ORDER**; owner `ddm_bnd1`, landing coordination MAIN;
consumer store `.omx/state/canonical_task_status.jsonl`, task suffix `::ITEM_3`.
Fire when MAIN harvests the retained patch in a Git-writable context, or the
assigned APDataStore tier again clears the canonical reserve for the serializer
fallback. Use the final manifest's hashes, preserve unrelated work and the staged
index, rerun required commit checks, and report either a verified landing or a
verified fallback bundle. This task does not mark the pricing charter complete.

## LIVE-HYPOTHESES

- A causal segment grammar may reduce rate because at least 88.9957% of the
  misprediction population is on an edge according to retained source geometry.
  Its coherence, topology/address cost and whole-tail saving remain untested.
- A local segment draw may differ from the failed global weight actuators;
  the old 240–455-cell response does not identify its response. The older 74,408 B
  grammar makes draw quality, rather than bytes alone, a concrete open question.

## DEAD-ENDS

- Treating 83,258.632 ideal bytes as a detachable retained substream: source code
  sums logarithms and emits only the full stream. It cannot be subtracted as cash.
- Joining the 12,377 pass-5 cells to move 37 as the shipped joint population:
  the hashes and pass results identify different fields.
- Proving a universal >=78 KB floor from a serializer or fitted model: an achieved
  length supplies the opposite inequality. No boundary family closure follows.
- Multiplying bz2d/rw1/msr1/lb1 ancestor constants into a local draw verdict:
  their objects and actuators differ; no transfer bound was established.

Own-vehicle frontier, **unchanged by this arm**: **S 0.13791730003757818 @ 180,388 B
[contest-CUDA T4 n600]**, archive SHA-256
`670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`.
