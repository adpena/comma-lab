# LS1 — Lane-conditioned surprise on the shipped move-44 field

Owner `ddm_ls1`. Measurement axis: `[macOS-CPU advisory / scorer-free n600 receiver probabilities]`.
`research_only=true`; `score_claim=false`; `promotable=false`; `frontier_moved=false`.
The exact frontier is unchanged by this measurement.

## Measurement contract and interpretation

The baseline is the actual `runtime.residual_archive.decode_production_tokens` receiver in an
unchanged copy of move 44. It reads the real RC64 stream, sparse HPAC model, native adaptive
corrector and counted RLC1 mixer configuration. The observer records frequencies only after
the decoder has chosen each symbol. No encoder teacher-forcing or replacement predictor stands
in for decoding. Full-field equality and stream-length reconciliation are mandatory gates.

All pre-existing SSD trees are read-only. The retention instruction authorizes the fresh output
tree `/Volumes/VertigoDataTier/pact/ddm_ls1/`; no predecessor, PR, sealed runtime or upstream file
was edited. Every frequency row, bit row, context assignment, count checkpoint, gain array,
map payload and restart state is retained there. No training, scorer, Modal, rendering or
candidate archive was run. The copied `archive.zip` is the unchanged input, not a new candidate.

The shipped field is `subset6.u8`, not `pass6.u8`:

- Path: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8`.
- Bytes: 117,964,800, representing all 600 × 384 × 512 symbols.
- SHA-256: `a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8`.
- Binding encoder receipt: `/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/encode/RESULT.json`.
  Its bound field copy was independently rehashed and matched.
- Input archive: 180,406 B, SHA-256
  `04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`.
- `INPUTS.json`, `SOURCE_MEMO_PINS.json` and `FRONTIER_DERIVATION.json` in the owned store
  contain full source custody and the existing frontier's component rederivation.

The demanded saving is 25,899 B. The counted tail is 119,909 B = 119,749 B RC64 stream +
96 B residual prefix + 64 B RLC1 rider. The remaining archive is 60,497 B, so an unchanged
non-tail requires a tail at most 94,010 B. These are measured sizes or arithmetic on this body.

## Oracle definitions: what is and is not bounded

Every likelihood comparison is a **signed marginal replacement cost against the shipped
probability at that exact symbol**, on the fixed decoded history. It is not an average
bits-per-Lane-token price for editing a field. The full probability is used for the baseline.

The three context partitions are nested:

1. **TC1 joint:** exactly TC1's coding-winner/confidence bucket crossed with its five existing
   contexts: causal spatial order two and three, previous-plane label, clipped spatial/temporal
   run, and row band. This reruns the reference joint table on the shipped probabilities and field.
2. **Receiver Lane:** TC1 joint plus nearest horizontal Lane distance in the preceding decoded
   frame and nearest Lane distance in the preceding row restricted to strictly earlier HPAC
   groups. No same-group/future token, scored argmax, PoseNet target or GT partition is visible.
3. **Granted previous-row Lane:** receiver Lane plus distance using the complete true preceding
   row. This grants the within-frame information unavailable under the real group order.

Distances have the declared bins 0, 1, 2, 3–4, 5–8, 9–16, 17–32, and farther/no observed Lane.
The first row/frame uses the missing/far bin. These are **specified finite geometry summaries**,
not all possible geometry descriptions. Thus the ladder cannot adjudicate all receiver models
or the entire carrier/generator family. This is an explicit scope boundary, not a hidden
claim that the selected distance map is optimal.

For a context c with counts n(c,k), the plug-in oracle cost is
`−sum n(c,k) log2(n(c,k)/n(c))`. It is the minimum empirical cost among categorical distributions
constant in those cells: consequently its saving is an **in-sample upper bound within that
specified replacement family with free cell parameters**. Miller–Madow adds
`sum_c (occupied_symbols(c) − 1)/(2 ln 2)` bits. That companion is a bias-corrected estimate,
**not a certified upper bound**, and serial dependence/sparse cells can leave bias.
The charter's wording cannot turn it into a universal ceiling.

Class and geometry are attribution labels, never supplied to the oracle as true-label
predictors. For readable class attribution, each occupied symbol receives an equal share of its
cell's Miller–Madow degrees, spread over its occurrences. This allocation convention affects
individual class entries, not the total. Joint removals are measured directly; contexts are not summed.

The six GDC3 cells apply to the error mask `shipped_token != final_coding_row_argmax`:
horizontal error runs of length 1 / 2–8 / at least 9, crossed with Euclidean distance at most
one pixel from the two-sided shipped-field class boundary versus interior. GDC3's original
code is reused with this explicitly changed predictor. Correct predictions form an additional
accounting column; forcing them into six error classes would lose the stream reconciliation.

The added map cost is measured by transmitting a zero when receiver/granted distance bins
agree, otherwise granted-bin-plus-one, in complete frame-raster order. A generic zlib stream
with an explicit dimensions header is retained twice and decoded exactly. This is an **achieved
upper bound on this map's storage cost**, not a minimum cost and not the probability-table cost.

## RECALL EVIDENCE

`RECALL_SEARCHES.json` retains query strings, scopes, hit counts and hashes. Content search
covered research Markdown/JSON/JSONL and arm receipts, the canonical-equations CLI export,
research index and DAG FEEDs, design/SPEC documents, and task/P0 ledgers. Queries included
`Miller.Madow|surprise.atlas|receiver.visible|lane.condition|UNION.*SUM|positioned.subspace`,
`Lane|context.mixing|surprise|the.cross`, and
`Lane.*(entropy|carrier|context)|Miller.Madow|receiver.visible`. Current directives were read.

Findings beyond the charter seeds changed the interpretation:

- **HC1/HC2/MI1:** expensive flip sites and the full binary hit/miss question are different
  denominators. Context already present in a network need not be fully used; conversely an
  expensive boundary does not prove unexploited information. Preserve the full stream and both
  outcomes, and localize signed gain as well as surprise. Historical byte values do not transfer.
- **TC4 and its later actual result:** a richer context slate had a real local byte gain but its
  later contest execution produced no scored result. The build memo's ready-to-fire label is
  stale. A successor needs a receiver-compute gate, not merely an oracle gain. No old-field
  magnitude or timing ratio is imported into this atlas.
- **GDC4's underlying shipped-field receipts and code:** `shipped_field/adaptive/RESULT.json`
  explicitly says its adaptive cost is not a compressed file; `AdaptiveModel.code` accumulates
  logarithms and updates counts. `shipped_field/dense_floor/RESULT.json` and its producer instead
  sum empirical entropy and a chosen asymptotic table penalty. These are model-specific computed
  costs, not actual stream sizes and not lower bounds on all generators. The stronger physical
  pricing/family-ceiling wording in the memo is not inherited. There is no claim that these
  scalar-only probes discarded an encoded payload: they did not construct one.
- **Canonical `token_tail_context_mixing_bound_v1`:** its own reference implementation says
  Miller–Madow is not a universal mixer ceiling. The ladder preserves that exclusion.
- **Canonical `lane_boundary_context_map_bound_v1`:** excludes joint weight refits, arbitrary
  geometry, integer/archive bounds and successor-field transfer. TC2's roughly 5.5 KB denotes
  old-field oracle *gain*, not transmitted map cost. This atlas measures the map's actual cost
  instead of subtracting that borrowed number.
- **Canonical direction-dependence and decoder-side-information laws:** average/edited-field
  prices and unavailable receiver state cannot authorize a marginal byte claim. No scorer
  state is treated as decoder input.
- **The-cross, followed to its canonical r10 adjudication and corrections:** its born-object
  rate/distortion combination is a different object and advisory evidence. It cannot price
  a generator of this exact shipped field. PC1/PC2 similarly concern a positioned **pose**
  subspace; they supply no measured price for a Lane probability carrier.

No transferable universal bound or shipped-field Lane-carrier price was found in these
searched sources. This is a scoped absence, not a proof that such a mechanism cannot exist.

<!-- # FORMALIZATION_PENDING: this arm measures a new-field atlas through the existing TC1 entropy callable; finite-cell diagnostics are not promoted to a universal rate law. The measured tables and explicit consumer fire order carry the handoff. -->

## Measured result

**No constructed mechanism is shown to clear the 25,899 B demand.** The strongest specified oracle crosses it only in the optimistic uncorrected fit; the Miller–Madow estimate does not. Neither fit supplies a counted runtime model.

All **117,964,800 / 117,964,800** decoded symbols match the shipped field. The sum is **957,986.402 bits = 119,748.300 B** versus **119,749 physical RC64 bytes**. Framing/finite-coder difference is **0.699757115 B (0.000584353%)**, passing the ±0.5% falsifier. Headers are accounted separately above.

### Atlas headline: actual bits by true class and error geometry

| Class | Isolated boundary | Isolated interior | Short boundary | Short interior | Long boundary | Long interior | Correct | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Road | 177,809.199 | 573.197 | 47,837.638 | 469.343 | 13.739 | 0.000 | 144,463.649 | 371,166.765 |
| Lane | 202,559.939 | 38.248 | 48,661.251 | 10.615 | 0.000 | 0.000 | 69,449.583 | 320,719.636 |
| Undrivable | 57,251.119 | 109.096 | 15,761.097 | 6.550 | 57.879 | 0.000 | 41,981.259 | 115,167.000 |
| Movable | 57,597.365 | 41.892 | 16,378.509 | 15.130 | 0.000 | 0.000 | 25,375.080 | 99,407.975 |
| MyCar | 37,580.018 | 25.008 | 4,633.776 | 0.000 | 0.000 | 0.000 | 9,286.224 | 51,525.027 |

Exact symbol denominators, in the same column order:

| Class | Isolated boundary | Isolated interior | Short boundary | Short interior | Long boundary | Long interior | Correct | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Road | 74,196 | 418 | 18,965 | 347 | 11 | 0 | 27,312,686 | 27,406,623 |
| Lane | 61,250 | 12 | 14,726 | 5 | 0 | 0 | 615,684 | 691,677 |
| Undrivable | 23,336 | 78 | 6,242 | 3 | 29 | 0 | 58,383,370 | 58,413,058 |
| Movable | 19,491 | 20 | 4,749 | 7 | 0 | 0 | 1,435,991 | 1,460,258 |
| MyCar | 10,239 | 6 | 943 | 0 | 0 | 0 | 29,981,996 | 29,993,184 |

Lane occupies **691,677 symbols (0.586342% of area)** and costs **33.478517% of stream surprise**. Most Lane surprise is at isolated boundary errors; long error runs carry zero Lane surprise in this atlas. That is a statement about errors of the final coding-row winner, not the earlier generator residuals. Correct predictions carry 290,555.796 bits and must remain in the accounting.

The binary decomposition is 290,555.796 bits for correct predictions, 644,633.749 bits for the wrong-winner event, and 22,796.857 bits choosing the actual class after that event. The first two together are 97.620336% of stream surprise.

### Oracle ladder: whole-population signed savings, before model costs

| Oracle | Plug-in upper B, fixed cell family | MM estimated B | MM shortfall vs 25,899 B | Lane MM B | Occupied cells | Singleton cells |
| --- | --- | --- | --- | --- | --- | --- |
| tc1_joint | 11,568.821 | 8,218.071 | 17,680.929 | 2,587.622 | 190,452 | 58,073 |
| receiver_lane | 22,222.524 | 17,534.216 | 8,364.784 | 6,686.446 | 491,314 | 206,209 |
| granted_previous_row_lane | 29,809.287 | 24,863.638 | 1,035.362 | 11,303.496 | 637,700 | 288,821 |

The uncorrected granted-geometry upper bound is 3,910.287 B above demand; its Miller–Madow correction is 4,945.649 B. Sparse-cell bias is therefore large enough to change the threshold answer. No finite-sample confidence guarantee or universal family ceiling is claimed.

Receiver geometry adds **9,316.145 B** beyond the TC1 table; the further unavailable-row grant adds **7,329.422 B** beyond receiver geometry. These are nested-table differences, not sums of standalone context gains. Lane contributes only **11,303.496 B** of the strongest whole-population estimate; the answer is not a Lane-only 25,899 B pocket.

#### tc1_joint: signed MM gain bytes by class × geometry

| Class | Isolated boundary | Isolated interior | Short boundary | Short interior | Long boundary | Long interior | Correct | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Road | 1,809.271 | 8.933 | 935.959 | 4.405 | 1.135 | 0.000 | 170.557 | 2,930.261 |
| Lane | 1,610.662 | 1.774 | 797.090 | 0.307 | 0.000 | 0.000 | 177.790 | 2,587.622 |
| Undrivable | 721.090 | 0.870 | 451.230 | 0.195 | 3.580 | 0.000 | 113.005 | 1,289.970 |
| Movable | 397.505 | 1.536 | 354.740 | 1.068 | 0.000 | 0.000 | 103.296 | 858.146 |
| MyCar | 381.814 | 2.735 | 143.842 | 0.000 | 0.000 | 0.000 | 23.681 | 552.072 |

#### receiver_lane: signed MM gain bytes by class × geometry

| Class | Isolated boundary | Isolated interior | Short boundary | Short interior | Long boundary | Long interior | Correct | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Road | 3,681.388 | 12.267 | 1,568.308 | 10.900 | 1.135 | 0.000 | 994.607 | 6,268.606 |
| Lane | 4,278.621 | 2.179 | 1,690.131 | 0.738 | 0.000 | 0.000 | 714.776 | 6,686.446 |
| Undrivable | 1,200.756 | 1.629 | 592.199 | 0.332 | 3.598 | 0.000 | 215.748 | 2,014.261 |
| Movable | 871.318 | 1.756 | 489.150 | 1.170 | 0.000 | 0.000 | 302.802 | 1,666.196 |
| MyCar | 629.871 | 2.831 | 206.062 | 0.000 | 0.000 | 0.000 | 59.943 | 898.707 |

#### granted_previous_row_lane: signed MM gain bytes by class × geometry

| Class | Isolated boundary | Isolated interior | Short boundary | Short interior | Long boundary | Long interior | Correct | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Road | 4,491.146 | 38.402 | 1,908.624 | 36.744 | 1.135 | 0.000 | 1,813.753 | 8,289.804 |
| Lane | 8,065.210 | 4.097 | 2,127.636 | 0.821 | 0.000 | 0.000 | 1,105.733 | 11,303.496 |
| Undrivable | 1,321.515 | 2.446 | 626.917 | 0.334 | 3.584 | 0.000 | 269.729 | 2,224.525 |
| Movable | 1,024.706 | 1.898 | 523.559 | 1.544 | 0.000 | 0.000 | 377.778 | 1,929.485 |
| MyCar | 793.937 | 2.935 | 236.399 | 0.000 | 0.000 | 0.000 | 83.057 | 1,116.327 |

### Localization: narrow in rows, diffuse in time

| Oracle | Lane B | Pairs for 50% positive Lane gain | Pairs for 90% | Rows for 50% | Rows for 90% | Lane cells for 90% positive gain |
| --- | --- | --- | --- | --- | --- | --- |
| tc1_joint | 2,587.622 | 170 | 431 | 26 | 82 | 89,941 |
| receiver_lane | 6,686.446 | 217 | 489 | 26 | 78 | 89,170 |
| granted_previous_row_lane | 11,303.496 | 232 | 503 | 28 | 82 | 81,201 |

All Lane gain lies in rows 128–319. In the strongest grant it takes **503 of 600 pairs** to cover 90% of positive Lane gain, but only **82 of 384 rows**. Rows 192 and 256 rank highest; these coincide with patch-row boundaries. This coincidence is measured localization, not proof of a dash-phase mechanism. No physical dash phase was inferred from the labels.

Cumulative **signed** MM gain vs oracle-ranked Lane cells touched (the ranked addresses are an additional, unpaid hindsight grant):

| Lane cells touched | B: tc1_joint | B: receiver_lane | B: granted_previous_row_lane |
| --- | --- | --- | --- |
| 1 | 3.875 | 3.875 | 3.875 |
| 10 | 24.971 | 27.699 | 28.579 |
| 100 | 168.113 | 191.569 | 197.532 |
| 691 | 702.317 | 886.865 | 976.939 |
| 1,000 | 888.784 | 1,160.416 | 1,318.768 |
| 6,916 | 2,470.729 | 3,833.849 | 5,419.077 |
| 10,000 | 2,881.436 | 4,588.682 | 6,708.651 |
| 34,583 | 4,350.513 | 7,404.384 | 11,108.956 |
| 69,167 | 5,107.123 | 8,811.981 | 13,083.770 |
| 100,000 | 5,429.745 | 9,389.362 | 13,855.873 |
| 172,919 | 5,752.567 | 9,943.019 | 14,559.433 |
| 345,838 | 5,923.181 | 10,219.796 | 14,901.156 |
| 691,677 | 2,587.622 | 6,686.446 | 11,303.496 |

Pair/row curves, every per-pair class sum and the full spatial maps are retained in `atlas/RESULT.json` and `verification/spatial_atlas.npz`. The Lane symbol curve can first rise and then fall because later locations have negative gain; those losses are not discarded. For the strongest grant, positive Lane cells total 14,942.937 B, negative Lane cells total -3,639.441 B. Even granting free selection of only its favorable Lane cells does not meet the demand.

### Mechanism verdict and actual map cost

| Mechanism | Shipped-field evidence | Cost and demand disposition |
| --- | --- | --- |
| Bigger mixer | TC1 joint: 11,568.821 B plug-in upper; 8,218.071 B MM | Specified constant-cell replacement is short by 14,330.179 B even with free table. Bigger full-resolution mixer cost/gain UNKNOWN. |
| Receiver-visible context model | 22,222.524 B plug-in upper; 17,534.216 B MM | This specified replacement is short by 3,676.476 B before table cost. Other full-resolution models remain unmeasured. |
| Counted Lane side channel/carrier | 29,809.287 B plug-in upper; 24,863.638 B MM | Actual override map alone is 594,003 B; it cannot pay. Probability parameters are additional. A compact positioned Lane carrier is UNKNOWN; pose-carrier prices do not transfer. |
| Born-vehicle generator | No new generator measured on this shipped field | Cost, removable bytes and shortfall UNKNOWN. The-cross uses another body; GDC4 supplied selected-model computed lengths, not a universal lower bound. |

The map changes **3,206,632 context symbols**. Both **594,003 B** encoded twins have SHA-256 `8b3433ad1515c214f2575ef322d410df1dc7976b315a8993c86828812c6f0d2a`; decode reproduces the complete raw override map and every granted context. The map alone exceeds the complete incumbent tail. Even with free probability parameters, the optimistic plug-in accounting gives -564,193.713 B net, and the MM accounting gives -569,139.362 B net. These are diagnostic arithmetic, not a newly encoded candidate. The refusal is **INSTANCE** scope for this map representation.

For scale only, dense fp16 probabilities plus uint64 cell keys would require the following **DERIVED layout sizes**, not measured encoded models: tc1_joint 3,047,232 B, receiver_lane 7,861,024 B, granted_previous_row_lane 10,203,200 B. The table values were granted in the oracle; the existing tiny counted mixer cannot inherit their performance at its own byte price.

## Verification and retained custody

Independent n600 verification **PASS**: all symbol frequencies, independent `31−log2(frequency)` arithmetic, class/geometry recount, binary sum, nested contexts, marginal gain sums and hashes, and full override-map reconstruction. The separate restart control compared 4,915,200 symbols, 24,576,000 integer-frequency entries, and all 103 native/mixer checkpoint arrays; its resumed frame-50 checkpoint was byte-identical to uninterrupted decoding. The restart prefix is an implementation control, not a statistical verdict.

All four owned Python sources received two separate review-tracker passes. Fatal/correctness Ruff checks and compilation passed. The trace and analysis sources remain byte-frozen to preserve launch bindings; nonfunctional style-lint findings are disclosed in `ENVIRONMENT.json`. No review override is used on Python. Completed checkpoints and every stage payload are kept; the disk-hygiene disposition is KEEP, not uncertified deletion.

Reproduce from the repository root using the pinned source release and runtime:

```sh
.venv/bin/python experiments/ddm_ls1_shipped_surprise.py --resume-from /Volumes/VertigoDataTier/pact/ddm_ls1/receiver_checkpoints
.venv/bin/python experiments/ddm_ls1_oracle_atlas.py --resume-from /Volumes/VertigoDataTier/pact/ddm_ls1/atlas
.venv/bin/python experiments/ddm_ls1_verify_atlas.py --resume-from /Volumes/VertigoDataTier/pact/ddm_ls1/verification
```

The retained complete checkpoints skip finished count/decode work; attribution revalidates its per-frame gain checkpoints. `SOURCE_RELEASE.json` binds the exact producers and helpers. `MANIFEST.json` indexes retained payloads by path/bytes/SHA-256. `LANDING_RECEIPT.json` records the serializer result separately to avoid a self-referential commit hash.

## Next charter and disposition

The next charter is
`.omx/research/ddm_ls2_full_resolution_lane_probability_bound_charter_20260911.md`:
price compact causal probability corrections that retain the full shipped prior, with exact
parameter charges, held-out validation and receiver-work admission. Reuse this atlas; no new
decode is owed if its binding still holds. This is a proposed measurement, not a claimed winner.

`FIRE_ORDERS.json` is retained both in the SSD store and in
`.omx/research/ddm_ls1_20260911/`. Its two tasks were registered through the canonical locked
task API and reread as pending. Legacy unrelated ledger warnings were surfaced; no old lifecycle
was repaired or promoted by this arm. Only the two new narrow events belong to this handoff.

Composition remains
`S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.
This atlas did not lower the exact score or achieve sub-0.12.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store** `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md`; **fire trigger:** harvest the verified LS1 landing or serializer bundle. Import the exact owned files if needed and fold the oracle/cost/provenance corrections into GS3; canonical task `ddm_ls1::MAIN_HARVEST`.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN, execution owner ddm_ls2; consumer store** `/Volumes/VertigoDataTier/pact/ddm_ls2_full_resolution_lane_probability_bound/RESULT.json`; **fire trigger:** MAIN harvests LS1 and verifies the current archive/field binding. Execute the attached finite-family full-resolution probability-bound charter; rebind explicitly after pointer drift. Canonical task `ddm_ls1::LS2_BOUND`.

## LIVE-HYPOTHESES

- A compact full-resolution correction may retain gains that coarse tables discard: the
  receiver-distance table adds joint information, and the largest row effects coincide with
  patch boundaries. Its real priced gain and runtime remain untested; LS2 owns the next test.
- A compact positioned Lane carrier may convey useful geometry more cheaply than the explicit
  override map: the unavailable-row grant adds signal, but the tested map is expensive. No pose
  carrier's price or performance transfers, and no new carrier test is ordered here.
- A different generator may change the rate/distortion tradeoff: the recalled generator costs
  do not establish a universal lower bound. This atlas supplies no new generator construction.

## DEAD-ENDS

- Using `pass6.u8` as the shipped field: the encoder binds `subset6.u8` instead.
- Treating the six error-geometry cells as the entire stream: correct predictions carry material
  surprise and are required for exact accounting.
- These exact TC1-joint and receiver-distance constant-cell replacements as complete-demand
  solutions: even their free-parameter plug-in upper bounds are below the demand. This closure
  does not cover full-resolution models outside those cells.
- This explicit zlib override-map instance as a paid side channel: its map alone is larger than
  the whole incumbent tail. No carrier-family closure follows.
- Treating oracle table gains as tiny-mixer performance, a Miller–Madow estimate as a certified
  universal ceiling, TC2's oracle gain as map cost, or GDC4's computed costs as physical payload
  measurements: the underlying code and receipts do not support those claims.
