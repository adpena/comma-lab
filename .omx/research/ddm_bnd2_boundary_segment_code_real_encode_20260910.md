# ddm_bnd2 — source preflight passed; required nice 10 denied

`[no-triality] [p0-ledger-ok]` · owner `ddm_bnd2` · charter dated 2026-09-10;
execution 2026-09-09 UTC. `research_only=true`; `score_claim=false`;
`charter_complete=false`. Lane `lane_ddm_bnd2_segment_code_20260910`, L0.

**The requested real segment encode is unfinished.** The sandbox denied both
`os.nice(10)` and `os.setpriority(os.PRIO_PROCESS, 0, 10)` with errno 1,
`Operation not permitted`. Actual priority remained 0. The charter's explicit
nice-10 constraint therefore prevented the encoding launch. No waiver was
received during this execution. A source preflight was completed and retained;
it is not the segment encoder, an n600 pricing row, or a representation verdict.

Verdict **BLOCKED_EXECUTION_PRECONDITION**, scope **THIS_SANDBOX_LAUNCH**.
No family, formulation, or candidate has been closed by a new encode.

## Source results and denominators

Axis **[macOS-CPU scorer-free source-custody preflight]**. These are fresh file
sizes, hashes and retained-input joins; the total miss count is re-derived from
the existing RP1 census, not a newly traced coder population.

| Quantity | Result | Boundary |
|---|---:|---|
| Shipped cmp2 archive | 180,388 B | SHA `670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc` |
| Raw token stream returned by cmp2's own reader | **119,779 B** | SHA `ffcd64bd3013f313538d3a426fdccdd8ac82c5fbcb19b382d7f935c5163bb427` |
| RP1 control envelope | **119,784 B** | Reader stream + `R6D1` + one zero padding byte; byte-identical reconstruction |
| Counted tc1 weights | 35 B | Preserved unchanged; not included in the raw-stream size |
| Complete retained pass-4 field | 117,964,800 symbols | 600 × 384 × 512, uint8; hash `361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94` |
| Coder misses in source ledger | 235,044 | 117,964,800 minus 117,729,756 coder-argmax matches |
| Unique retained locations revalidated | 213,733 | Every stored symbol equals pass 4 at its retained location |
| Locations still unlocated | **21,311** | No full current-field trace generated |
| New coding rows / scorer pairs / draws | **0 / 0 / 0** | No partial-n price, no draw, no candidate archive |

The explicit envelope join is a useful correction to future pricing:
119,784 B is the charter comparator, but it is not the on-wire raw-stream length.
Compare like-for-like framing first, then the complete replacement tail/container.
Do not silently subtract either the 5-byte wrapper or the 35 counted weights as
if a new codec had removed them. The 96-byte compact residual prefix and remaining
archive sections also retain their actual charged homes.

The preflight used the actual
`ddm_cmp2_compose/candidate_runtime/runtime/residual_archive.py::read_residual_archive`
on a byte-verified retained copy of the pointer archive. It did **not** run the
token decoder or renderer. The source RP1 n600 control's schema, strict success
boolean, emitted/CMP1/stream lengths and hashes, census arithmetic, unique
coordinates and shipped-symbol join are validated explicitly.

## Required prices and run lengths — not measured

| Required row | Bytes / result | Disposition |
|---|---|---|
| Existing full envelope | 119,784 B | Source identity verified; existing encode |
| Full population masked to model prediction, re-encoded | unknown | Not launched |
| Charged class-pair chain/polyline + ternary offsets | unknown | Encoder not implemented |
| Remaining tokens under tc1 | unknown | No routed residual stream emitted |
| Twin segment + residual price | unknown | No primary or repeat candidate |
| Complete-population tangent run distribution | unknown | Missing locations not recovered |
| Falsifier ≤114,784 B | untested | Neither success nor failure established |

The source's 83,258.632 ideal attributed bytes are **not** an isolated stream or
a displaceable cash amount. Its 36,519-byte approximate correct-prediction flag
mass is also an attribution, not this arm's measured masked-field control.
bnd1's on-edge bound remains source evidence; no new GT-edge census is claimed.
Shipped residual **12,614 cells** remains the pass-4 source measurement; the
12,377-cell pass-5 count is unshipped and must not join to this field.

## Concrete implementation requirements retained for the next execution

This is a design handoff, not a claimed implemented grammar:

* Replace the old axis-aligned `(segment,s,n)` chart with explicitly charged
  class-pair unit-edge polylines, segment starts/topology, chain directions and
  ternary normal offsets; give every exception to the remaining tc1 stream.
  Charge the selected supports and any class/symbol/routing information. The
  actual serialized total, including framing, must decide the result.
* Recover **all** post-tc1 **integer-frequency** argmax misses. RP1's
  `saving >= 0.25` retention filter loses the required locations. Setting that
  threshold to zero alone is insufficient: correctly predicted and tied rows
  must be separated explicitly, and crash-resumability must be added before
  using the existing rank loop as a launch path.
* Route every supplied segment token into `current`, `corrector.observe` and
  mixer state at its original causal group position. An end-of-frame patch
  changes subsequent probabilities and cannot use the original trace as proof.
* Distinguish an actual whole-field masked-prediction re-encode, with the
  changed field's causal state, from a frozen-original-trajectory ablation.
  If both are useful, retain and label both; neither is an isolated entropy
  substream. No logarithmic estimate is an admitted price.
* Preserve independent twin encoder states and per-stage/periodic checkpoints,
  source bindings, every payload and a parse-back equality proof. Reuse jg2's
  structural corrector-state capture and tc1's complete mixer snapshot rather
  than copying an incomplete base-class state dictionary.

A measured loss would apply to the specified grammar, addressing, partition and
coder. The charter's statement that a richer grammar can only add side information
does not establish monotonic **total** price: extra structure can reduce address
and residual costs. NO-FAKE therefore prevents broadening an achieved-code loss
to every richer boundary family. The lossless-first and 5 KB draw gates remain
binding. No speculative draw was run to work around them.

## RECALL EVIDENCE

Read the charter and common contract in full, PROGRAM, governing AGENTS/CLAUDE
sections, the operating handoff, main hot state, pointer, bnd1 ITEM 1–3 and its
RECALL EVIDENCE, of1:58–109, or1:111–143, tc1's full pricing receipt, the curve-relative
coder and its build memo, and the actual cmp2 reader and RP1 mixer loop.
An independent read-only recall/review worker covered the full 480-equation CLI
export, research memos/receipts, index/DAG, design/SPEC and task surfaces.

Content queries included `boundary.segment|normal.offset|contour.string|mispredict`,
`curve.relative`, `FEED-residualkit`, `boundary representation offset field class pair`,
and `bnd1.*ITEM|gb1.*VALID_BOUND_INPUTS|ddm_gb2|ddm_bnd2`.
One shell glob query failed because `docs/*SPEC*` had no matches; it was replaced
by `rg` file globs over `docs`. No absence claim uses the failed query.

| Beyond-seed source | Finding | Effect |
|---|---|---|
| `ddm_hc2_wrong_half_flip_location_decomposition_20260905.md:45–109` | Older fs2 misses had median component size 1 and 81.10% singleton components | Edge proximity cannot supply the missing run-length measurement; older counts remain priors |
| `ddm_hc1_hpac_calibration_reliability_20260824.md:29–55` | Causal versus acausal geometry changes attribution substantially | Current-plane or GT boundaries must be charged rather than supplied free |
| `ddm_mi1_indicator_model_axis_20260824.md:13–40` | Temporal/run/spatial contexts already exist, but feature presence does not prove complete consumption | New segment contexts need actual residual byte evidence |
| `v8_increment1_design_draft_20260709.md:59–69` | Historical temporal generator amortization beat independent frame chains | Extra grammar structure can reduce total price; no richer-family monotonicity claim |
| `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md:9060` | Real contour coding spent 45% on anchors on an ancestor object | Starts/connectivity enter the first price, not a later top-up |
| tc1 `rebase_pc2/INPUTS.json` and `rows/frame_*.npz` | Full pre-mixer row traces exist, but bind ancestor field `a73289e…` | Rejected as current-field evidence; shipped field is `361cc6c9…` |

Relevant recalled laws: `token_tail_context_mixing_bound_v1`,
`context_model_reorder_savings_v1`, `v8_geometric_rate_decomposition_v1`,
`leverd_flicker_residual_reactivation_economics_v1`.
Did not find a complete current-field missing-location dump in the bounded
RP1/CMP1 stores inspected. HC2's model-ledger alternative and older curve-relative
prices are not new real-code prices for this shipped field.

## Retention, review and reproducibility

All retained bytes are in
`/Volumes/VertigoDataTier/pact/ddm_bnd2_segment_code/retained/`.
The complete source receipt is [source_preflight.json](ddm_bnd2_20260910/source_preflight.json).
It names the actual source files, current Git hash, producer/reader/upstream hashes,
argv, component sizes and payload SHA-256 values. No live tree or upstream file
was changed. Storage preflight measured 69,714,214,912 free bytes and preserved
the 40 GiB reserve. No large artifacts were deleted or moved.

Producer: `experiments/ddm_bnd2_source_preflight.py`, SHA-256
`85fef367933e931dc09005707d943098ad828ee0e2b37879b4a651ebc350dbfe`.
Two clean review passes cover this hash; earlier review findings about weak
census joins, archive snapshot races and overwrite-capable publication were
fixed before those clean passes. Ruff and review policy passed. Five tracked
entities are compliant. The first tool transport lost stdout; the complete
durable receipt survived. An explicit resume then returned **20**, with source
checks passed and `NICE_10_DENIED`; its stdout/stderr and exit receipt are retained
in `RESUME_VALIDATION.json`. This is a real fail-closed preflight, not a failed
codec experiment or evidence that the segment family loses.

Reproduce the source preflight only:

```sh
.venv/bin/python experiments/ddm_bnd2_source_preflight.py \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_bnd2_segment_code
```

`boundary_segment_recode_price_v1` is not registered.
# FORMALIZATION_PENDING: No real segment-plus-residual encode exists; source-file identity and a priority refusal cannot anchor a measured recode-price law.

All six solver integration hooks are N/A for this research-only blocked preflight:
no new sensitivity measurement, Pareto constraint, per-tensor allocation,
archive-deployable candidate, empirical codec posterior, or competing scored
interpretation was produced. Existing task rows consume the blocker and design
requirements; no production behavior or score pointer is changed.

## Landing custody

The required serializer refused the main checkout's Git object write with
`Operation not permitted` during `git apply --cached`. It successfully authored
an isolated fallback commit and retained a Git bundle and format-patch under
the assigned Vertigo tree; the bundle passed `git bundle verify`. **This is not
a landing on main.** The exact latest bundle/commit, hashes and verification
are in `ddm_bnd2_20260910/FINAL_HANDOFF.json` after final packaging. The staged
index remained empty. Only bnd2 source/receipt files enter the intended patch;
the lane registration and existing-task note snapshots document the narrowly
scoped canonical API writes, and unrelated shared dirty work is excluded.
Commit hooks did not run because object insertion failed; Ruff, two code-review
passes, review-policy compliance and the source preflight are the checks that
actually completed. The charter remains incomplete independently of landing.

## NEXT_IF_RESUMED

* **QUEUED-WITH-A-FIRE-ORDER** — owner `ddm_bnd2`, coordinated by MAIN; existing
  consumer `.omx/state/canonical_task_status.jsonl`, bnd1 `::ITEM_1`; bulk consumer
  `/Volumes/VertigoDataTier/pact/ddm_bnd2_segment_code/retained/`. Fire when nice 10
  can be verified or the operator explicitly changes that constraint. Re-read
  the pointer, recover all misses, implement and review the charged grammar,
  then run resumable n600 twins, masked control and parse-back. This is the
  unfinished bnd2 work, not a newly completed bnd1 obligation.
* **FOLDED** — owner MAIN for scorer coordination, execution successor `ddm_bnd2`;
  consumer `.omx/state/canonical_task_status.jsonl`, bnd1 `::ITEM_2`. Fire only
  after a real twin-verified lossless total ≤114,784 B and explicit scorer
  allocation. Use the retained 12-pair selection; retain every draw. A lossless
  loss stops this draw and routes the existing generator successor instead.
* **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer
  `.omx/state/canonical_task_status.jsonl`,
  `ddm_bnd2_boundary_segment_code_real_encode_20260910::LANDING`. Fire on harvest
  in a Git-writable context: land the final reviewed bundle using the serializer,
  run required commit checks, preserve unrelated changes and the staged index,
  and record the actual commit. Use `FINAL_HANDOFF.json` to select the final bundle.

## LIVE-HYPOTHESES

* A charged segment code may still win: retained misses are strongly concentrated
  on edges. Full-population coherence and its actual price are untested.
* The existing the-cross/gb2 generator successor remains plausible because changing
  the representation can create coherence instead of only addressing existing
  isolated misses. It stays folded into MAIN's existing gb1 `::VALID_BOUND_INPUTS`
  consumer and gb2 arm; this blocked preflight does not launch or duplicate it.

## DEAD-ENDS

* Using ancestor tc1 `a73289e…` traces as move-37 evidence: the field hashes differ.
* Pricing detached 83 KB attribution or treating achieved lengths as lower bounds:
  neither supplies the requested actual composite bytes.
* Repairing decoder state only after decoding a masked residual: later causal
  probability rows would differ.
* Inferring all richer grammars lose from one minimal grammar: total-price
  monotonicity is not established. No new segment formulation is closed here.
* Launching at priority 0 as if nice 10 succeeded: the measured syscall refused.

Own-vehicle frontier **unchanged**: **S 0.13791730003757818 @ 180,388 B
[contest-CUDA T4 n600]**, recomputed from existing anchor components; archive SHA
`670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`.
