# CB2: beat-PR130 composition re-derived on corrected premises

## Verdict first

No qualifying composed vehicle is proved below the DALI bar in this arm. The corrected measured CPR1/bot tuple is

`S = 100(0.00029660) + sqrt(10(0.00002331)) + 25(191052)/37545489 = 0.172141297491896447...`

on `[contest-CUDA, DALI GT, n600]`. Substituting all three CPR1 components into TQ1C therefore reaches the bar exactly; it does not satisfy the strict `< 0.17214129749189644` requirement.

One already-built lossless XZ candidate is 8 bytes smaller: 191,044 B, SHA-256 `dc302953ff1e7f6d09210fff80cf6981cd7a36fafbf4299770bff61abdc462bd`. Holding the published rounded CPR1 distortions fixed gives `S = 0.172135970620271470...`, 0.000005326871624977 below the bar. That is a **derived candidate**, not a score: OP1R proved exact raw model/token parse-back and a receiver control, but no exact DALI evaluator replay exists. The honest campaign state is therefore “candidate arithmetic crosses; authority evidence does not yet cross.”

This arm was scorer-free and $0. It did not run Metal, MPS, CUDA, a scorer, an evaluator, a trainer, an archive builder, a dispatch, a launch, or a promotion.

## The corrected reference objects

| Object | `d_seg` / Seg term | `d_pose` / Pose term | Bytes / Rate term | Total | Axis and custody |
|---|---:|---:|---:|---:|---|
| TQ1C own vehicle | 0.004305419922 / 0.430541992200 | 0.000716508925 / 0.0846468502071991 | 357,837 / 0.2382689702083784 | 0.7534578126155775 | `[macOS-CPU frozen-scorer advisory, n600]`; archive SHA `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06` |
| PR130 CPR1/bot | 0.00029660 / 0.029660000000 | 0.00002331 / 0.0152676127799994 | 191,052 / 0.1272136847118971 | 0.172141297491896447 | `[contest-CUDA, DALI GT, n600]`; archive SHA `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd` |

The TQ1C row and the PR130 row are different GT axes. Their component subtraction is a planning decomposition, not a same-axis measured delta. A beat claim must be made on DALI, or an AV score must clear the conservative `0.11711984070155727` rail. The 0.055021456790339165 quantity used to derive that AV rail is a triangle-inequality upper bound on fixed-archive decoder-swap movement. It is not a score, loss, or recoverable credit.

Reference receipts:

- TQ1C: `.omx/research/ddm_tq1_20260805/tq1c/RECEIPT.md`, SHA-256 `907a30206f324f611e34ee6ba007590529fd92e608e0ac66de3bbc54dac90e30`.
- PR130 tuple/anatomy: `.omx/research/pr86_pr130_intake_20260728/anatomy_pr130.json`, SHA-256 `cae35b80e52fd2bbd415e2d9be621b665d3b8f91aed07422dc50485ba3401abd`.
- Axis adjudication: `.omx/research/ddm_ax2_20260809T121456Z/AX2_FINDINGS.md`, SHA-256 `d7b2d791054cdd4a3310359ea851ee6dcbe99fea041c64ef9890f5a0d324dec6`.
- Same-host GT bound: `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/result_summary.json`, SHA-256 `15f43860a2d0a32bd7191ee12f3f1f1308cf3345090a4ad1cf2f3bb67bc5aa2c`.

The externally reported `d_seg=0.00028609, d_pose=0.00001967` belongs to the 194,380-byte source/Ada object. Combining those distortions with 191,052 B produces `0.16984766243023947`, but it is a mixed-object projection and is forbidden as a measured CPR1 row.

## Re-derived three-axis gap

All terms below use `S = 100 d_seg + sqrt(10 d_pose) + 25 B / 37,545,489`.

| Axis | TQ1C term | PR130 term | Re-derived gap | Share of total gap | Old handed shorthand | What moved |
|---|---:|---:|---:|---:|---:|---|
| Seg | 0.430541992200 | 0.029660000000 | **0.400881992200** | 68.9611% | 0.4015 | moved down about 0.000618; the old 3.87x renderer premise is withdrawn |
| Pose | 0.0846468502071991 | 0.0152676127799994 | **0.0693792374271997** | 11.9348% | 0.2776 | moved down about 0.208221 because the old pose baseline was stale |
| Rate | 0.2382689702083784 | 0.1272136847118971 | **0.1110552854964813** | 19.1041% | 0.1126 | moved down about 0.001545 after replacing the stale byte baseline |
| Total | 0.7534578126155775 | 0.1721412974918964 | **0.5813165151236811** | 100% | not reusable | every column was re-anchored |

The current ratios are 14.5159134x in `d_seg`, 30.7382636x in `d_pose`, and 1.8729822x in bytes. The common contract's qo1 pointer and the charter's older shorthand were both superseded by the live TQ1C receipt; neither was patched forward.

## Seg: adopt or beat a known mechanism

SG2 closed the “missing post-render closer” search at complete 49-stage static-source scope. The mechanism is stage-07/08 int4-QAT semantic rendering, not a later correction stream. Stage 08 packs byte-identically to the archive semantic section. Its recorded AV-axis quantized `exact_seg` improved from 0.0002972496880425347 to 0.0002763705783420139 at the same 40,252 packed bytes: `100 * (0.0002972496880425347 - 0.0002763705783420139) = 0.00208791097005208` Seg-score units, a 7.0241% reduction.

The final-checkpoint Metal reproduction measured, without making a score claim:

| GT | Quantized `exact_seg` | Seg term | Relation |
|---|---:|---:|---|
| AV | 0.0002764044867621528 | 0.0276404486762153 | 1.000123x the stage-08 recorded AV row |
| DALI | 0.0002857038709852431 | 0.0285703870985243 | 0.998650x the published Ada DALI row |

Receipt: `/Volumes/VertigoDataTier/pact/ddm_pr130_quant_repro_20260809T124039Z/quantized_repro_receipt.json`, SHA-256 `78f39bf24f008cad7aed9fa8b31dc29e7e9ff3d90848c4ea795e89f5f029530b`; authority `[macOS-Metal advisory]`, `score_claim=false`. The exact-object AV-to-DALI renderer difference is `9.299384223090295e-06 d_seg = 0.0009299384223090295 S`, 5.31% of the separate 0.017523023416288197 Seg bound. This assigns SG2's small residual to decoder axis on this object; it does not convert the bound into recoverable credit or close the pose-side decoder effect.

Adopting PR130's **measured full-archive Seg component** on the TQ1C pose/rate surface closes 0.4008819922 S but yields `S = 0.3525758204155775`, still 0.1804345229236811 above the bar. Seg adoption alone is insufficient.

The semantic port itself is no longer the mechanism blocker: the bounded Metal train exited 0 at 1,704 MiB peak and the device-coverage receipt reported 38/38 gradients with zero CPU-fallback warnings. Its no-descent training verdict is not a portability failure. The durable run inputs are `/Volumes/VertigoDataTier/pact/ddm_mps_port_20260809T121108Z/launch_manifest.json` (SHA `a68680abe37db9bc3434f6843828432d55b8d0a433e28fbfb83fa6cabce20de1`), `history.json` (SHA `074dca69b8de28518ba8ae616ddcc63091477f019decff85cfc9c9d6617b2d5e`), and `run.log` (SHA `70d1edde5b8f898c7fbc98f3bd1c45de11500716de9540ec626c572d195cbbc8`). The 38/38 device-coverage row is in the bounded `.omx/state/probe_outcomes.jsonl` snapshot SHA `89a4b627c9fb43be3863b7865dafc299840b38422c40655d5412eab077a0e335`, row `pr130_semantic_renderer_mps_device_coverage`.

### Seg TOY-BRACKET

Combining the Metal-reproduced DALI renderer row with PR130's published pose and 191,052 bytes gives

`100(0.0002857038709852431) + sqrt(10(0.00002331)) + 25(191052)/37545489 = 0.171051684590420757...`

which is 0.00108961290147569 below the bar and corresponds to 1,636.401968 bytes of rate headroom. With those terms fixed, 192,688 B is below the bar and 192,689 B is above it. This is a declared **TOY-BRACKET**, not a score: the renderer-only row and the full-archive pose/rate row are not one evaluated archive. It is useful only because it sizes the end-to-end Seg residual that the exact replay can settle.

## Pose: portable with named work, not a borrowed ancestor

The honest adoption price is the measured CPR1/bot DALI component:

`sqrt(10(0.000716508925)) - sqrt(10(0.00002331)) = 0.0693792374271997 S`.

Adopting pose alone on TQ1C Seg/rate yields `S = 0.6840785751883778`, still 0.5119372776964813 above the bar. With PR130 Seg already adopted, pose adoption yields `S = 0.2831965829883778`; the remaining shortfall is exactly the 0.1110552854964813 rate gap.

PP2's verdict is static `portable-with-named-work`: 60 distinct execution families, 3 UNKNOWN (direct safetensors-to-MPS, sparse Embedding backward, and sparse COO/coalesce into `RowLocalSparseAdam`), one certain `torch.cuda.empty_cache()` edit, 15/24 pose stages using the audited trainer, and 9/24 non-trainer stages still unaudited. A stock dense Adam substitution is a mechanism change because it loses the row-local optimizer clocks. Receipt: `.omx/research/ddm_pp2_20260809T121528Z/PP2_FINDINGS.md`, SHA-256 `57f17964e6ee5743cb29b2f48d610e4758c8941ae5c700f8c0fa1ebc87346dfe`.

The raw 23,054-byte pose-carrier section would be 0.0153507123052785 rate-score units if it were separable. It is not: it shares the compressed model bundle with semantic and HPAC state. The pose mechanism cost is therefore “23,054 raw descriptive bytes plus an unmeasured whole-archive joint marginal,” not 23,054 additive archive bytes.

## Rate: only the whole archive settles the price

The exact CPR1 anatomy is:

- archive 191,052 B = 100 B ZIP overhead + 190,952 B member;
- member = 4 B prefix + 73,968 B XZ model bundle + 116,980 B HPAC token stream;
- raw model bundle 83,493 B = two 4-byte lengths + 40,252 B semantic + 23,054 B carrier + 20,179 B HPAC model.

The semantic renderer has 66,339 parameters. Its raw int4 code floor is `ceil(66339/2) = 33,170 B`; packed semantic state is 40,252 B; the 7,082 B difference is sub-rank-2 parameters and scale overhead. Their descriptive rate terms are 0.0220865414750624, 0.0268021545810736, and 0.00471561310601122 respectively. None is a marginal archive price. Semantic, carrier, and HPAC state are jointly XZ/LZMA-compressed.

The TQ1C-to-PR130 byte reduction is 166,785 B, closing 0.1110552854964813 S. At the published PR130 distortions, strict victory by rate alone requires at most 191,051 B, so TQ1C needs a 166,786-byte cut. One archive byte is `25/37545489 = 0.000000665858953122... S`.

OP1R already fired a scorer-free configured two-grid single-LZMA2 race: 3,375 valid round trips found a single 8-byte improvement, 73,968 to 73,960 B for the model bundle and 191,052 to 191,044 B for the archive. Further search of that exact configuration is **FOLDED at FORMULATION scope**. The exact candidate is retained at `/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/rate_race/cpr1_xz_bt2_d12lc0lp1pb0_n192.zip`; its ZIP integrity and 191,044-byte size were rechecked read-only in CB2. Source: `.omx/research/ddm_op1r_20260809/OP1R_PATH.md`, SHA-256 `9aabf71819380bcd1dc2d872f66f1bc059446fe9a3d5fdbafa86e0c586c9b765`.

## Component substitution sheet

These are arithmetic substitutions, not built candidates:

| PR130 components substituted into TQ1C | Derived S | Shortfall above PR130 bar |
|---|---:|---:|
| Seg only | 0.3525758204155775 | 0.1804345229236811 |
| Pose only | 0.6840785751883778 | 0.5119372776964813 |
| Rate only | 0.6424025271190962 | 0.4702612296271997 |
| Seg + pose | 0.2831965829883778 | 0.1110552854964813 |
| Seg + rate | 0.2415205349190962 | 0.0693792374271997 |
| Pose + rate | 0.5730232896918964 | 0.4008819922000000 |
| Seg + pose + 191,052 B | 0.1721412974918964 | 0; equality is not a strict win |
| Seg + pose + 191,044 B | 0.1721359706202715 | -0.000005326871625 derived; exact replay still owed |

The sheet's answer is not “add one closer.” Crossing requires all three PR130-class components on one DALI-measured, byte-closed object, or a compensating improvement on one component. The 8-byte candidate is the cheapest direct falsifier because it may already provide that compensation.

## Ranked swing variables

The ratio is ranked by decision value divided by a bounded cost class; no fake numeric dollar ratio is asserted because the receipts do not price operator time or an exact replay.

1. **Exact DALI score of the retained 191,044-byte candidate.** Decision swing: direct pass/fail against 0.17214129749189644; fixed-distortion arithmetic margin 0.000005326871625 S. Cost class: lowest — candidate, hashes, and parse-back already exist; one pinned authority replay remains. Owner: `MAIN #984 exact-eval/custody owner`. Fire trigger: recover and pin PR130's challenge evaluator/runtime, claim the exact-eval lane, verify the owned receiver and candidate SHA, then replay this exact ZIP. Consumer: `.omx/state/probe_outcomes.jsonl`.
2. **Whole-archive joint marginal after any semantic, carrier, or HPAC mutation.** Decision swing: the measured rate gap is 0.1110552854964813 S, while one byte decides a fixed-distortion tie. Cost class: low scorer-free pack-back per mutation, but it requires archive-build authority not held by CB2. Owner: `PR130 archive-compose/rate owner`. Fire trigger: any accepted semantic, carrier, or HPAC byte change; rebuild the whole archive, jointly XZ the complete bundle, parse twice, and report exact bytes instead of raw-section sums. Consumer: `.omx/state/probe_outcomes.jsonl`.
3. **DALI-targeted pose portability and same-object `d_pose`.** Decision swing: up to the 0.0693792374271997 S TQ1C-to-PR130 pose gap; it is the remaining shortfall after Seg+rate substitution. Cost class: medium — first a two-step sparse parity probe, then the full one-batch graph and the nine non-trainer stage audits before any governed pose run. Owner: `MAIN PR130 pose-port owner`. Fire trigger: bind the exact DALI cache SHA and pinned torch/timm runtime on a real-Metal host; require zero fallback warnings and row-local optimizer parity. Consumer: `.omx/state/probe_outcomes.jsonl`.

## RECALL EVIDENCE

I searched the full `.omx/research` corpus by content, the canonical research indexes, `sub015_DAG_*` FEED blocks, `.omx/state/main_hot_state.md`, task/probe ledgers, and the canonical-equations JSON listing. Queries included `PR130`, `CPR1`, `0.172141`, `TQ1C`, `66339`, `40252`, `joint LZMA`, `quantized_exact_seg`, `DALI`, `AVVideoDataset`, `pose carrier`, and `191044`.

Beyond the charter seeds, this found:

- OP1R's exact-lossless 191,044-byte candidate. That changed the plan from a static “composition reaches equality” answer to a direct, retained candidate whose exact replay can settle a strict crossing.
- The corrected CPR1/bot tuple and the mixed-object 194,380/191,052 trap. That prevented using the projected 0.16984766 row as measured evidence.
- MAP1-era semantic/pose split language that treated raw pieces as separable. SG2/anatomy evidence changed the sheet to whole-archive joint marginals only.
- The canonical score-marginal and gap-decomposition equations, but no new CB2-specific equation. CB2 uses the existing evaluator formula and adds no equation registry row.
- A prior `.omx/research/ddm_cb2_20260806` directory for unrelated task #983. This run uses a UTC-qualified directory and does not overwrite it.

The shared `.omx/state/probe_outcomes.jsonl` already contained unrelated dirty work. CB2 therefore did not append or stage it. `CB2_ROWS.jsonl` is the complete feeder, and canonical ingestion is **QUEUED-WITH-A-FIRE-ORDER** when the shared writer releases the ledger.

## Scope and boundaries

- `MEASURED` here means a pre-existing receipt or byte artifact whose path and SHA are named. CB2 made only read-only byte/ZIP-integrity checks and arithmetic derivations.
- The unpriced remainder is exact replay time/operator coordination, the full pose port, and every whole-archive marginal after component changes.
- The renderer/full-archive combination is explicitly a TOY-BRACKET; no borrowed number is promoted as a same-object score.
- No upstream or PR130-intake file was modified. No protected file was touched. No `.py` file changed, so review-tracker passes are not applicable.
- The frontier did not move.

## LIVE-HYPOTHESES

- The retained 191,044-byte candidate will beat the numeric PR130 bar on exact DALI because its model/token parse-back is exact and only the lossless XZ representation changed; full receiver/evaluator custody is the remaining proof.
- The Metal-reproduced DALI renderer result may survive full-archive composition closely enough to leave about 0.00109 S of headroom, because it matches the published Ada Seg row within 0.135%; this remains a TOY-BRACKET until one archive carries all terms.
- The PR130 pose leg will port without a mechanism rewrite because 57/60 audited execution families are statically covered and the three unknowns have narrow probes; sparse row-local optimizer parity is the decisive risk.

## DEAD-ENDS

- Searching stages 09-49 for a hidden post-render Seg closer is closed at the complete 49-stage SG2 source graph: none exists, and the old 3.87x result came from the float evaluator path.
- Treating stage 08 as rate-only is closed: its recorded quantized `exact_seg` improved 7.0241% at the same packed semantic size.
- Treating 40,252 semantic bytes, 23,054 carrier bytes, or their archive percentages as additive marginals is closed: they share one XZ model bundle.
- Mixing the 194,380-byte Ada distortions with the 191,052-byte CPR1 archive is closed as a measured row.
- Repeating OP1R's configured two-grid single-LZMA2 search on the unchanged raw bundle is closed at FORMULATION scope after 3,375 valid round trips yielded only 8 bytes.
- Using stock dense Adam as the pose-port fallback is closed because it changes row-local clock and update semantics.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory] n600`.
