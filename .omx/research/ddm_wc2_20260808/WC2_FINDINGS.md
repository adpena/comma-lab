# ddm_wc2 findings

## Verdict

WC2 implemented the memory/heterogeneous-compute admission hooks on the live `mx1` vehicle and wrote a plan-only extended bench ticket. No Metal run, n600 scorer job, archive build, ANE parity run, or exact eval was performed by this arm.

| deliverable | status | evidence |
|---|---|---|
| WC1 bench consumed | MEASURED INPUT | `.omx/research/ddm_wc1_20260807/wc1_bench_receipts.jsonl` sha256 `4efecaf1fbdc20dcfd510da1530654df6cd49d5d145007080841528810335680` |
| Memory-for-compute law | IMPLEMENTED + QUEUED | `--cache-residency {selected,ram-full}`; `ram-cache` row in `wc2_bench_ticket.json` |
| Derived microbatch law | IMPLEMENTED | `_derive_train_microbatch_plan`; default `auto` selects 4-pair chunks from measured WC1 anchor, not unlabeled `gpu_default_4_pairs` |
| Concurrent CPU verdict | IMPLEMENTED + QUEUED | `concurrent-cpu-verdict` bench row starts a CPU-torch subprocess group and reclaims it with timeout/killpg |
| ANE/CoreML parity gate | IMPLEMENTED + QUEUED | `coreml-segnet-parity` mode converts the real frozen upstream SegNet and compares argmax on rendered real checkpoint frames |
| Custom Metal conv fit | FOLDED | #478 banked measured rows already show this simdgroup formulation loses to MLX native on this M5 host |

## WC1 Bench Input

| variant | status | seconds/step | d_seg sanity | read |
|---|---|---:|---:|---|
| fp16-train | passed | 8.519557237625 | 0.001041730245 | fastest measured WC1 row; use as default recommendation only after CPU verdict gating |
| compile | passed | 10.304331827164 | 0.001042366028 | second fastest, but compile remains region-scoped |
| baseline | passed | 10.441456365585 | 0.001042683919 | measured four-pair chunk anchor |
| threads | passed | 10.775010013580 | 0.001042683919 | slower than baseline in current receipt |
| batched | passed | 11.596535015106 | 0.001042683958 | full n32 microbatch lost here; do not promote blind full-batch |

## Memory Law

The real cache sizes are `943,720,090 B` for tq1c input labels and `943,720,076 B` for GT target labels, or `1,887,440,166 B` combined. Against the 116 GiB operator ceiling, the full RAM-resident pair is small enough to measure, but not free enough to assume. WC2 therefore added `--cache-residency ram-full` as a measured mem-probe variant instead of changing the default.

The default microbatch policy is now `auto`, but on this vehicle it still selects `4` pairs because WC1 measured four-pair baseline faster than full n32. The point of the change is provenance and future replacement: the receipt says `wc2_auto_empirical_wallclock_anchor`, and the `derived-microbatch-4` row exists so MAIN can remeasure the law with the rest of the matrix.

## Heterogeneous Pipeline

`concurrent-cpu-verdict` runs the real `torch-verdict` mode in a subprocess process group with CPU thread env capped at 4 while the guarded Metal train row runs. Its output is custody-recorded separately and folded into the row after the train subprocess exits.

`ane-verdict` is parity-gated before use. The new `coreml-segnet-parity` mode loads the real upstream SegNet, renders real checkpoint frames, converts SegNet through CoreML FP32 with `CPU_AND_NE`, then compares CoreML argmax/logits against CPU-torch authority. If `coremltools`, `CPU_AND_NE`, conversion, predict, or shape parity fails, the row is blocked/failed rather than advisory-promoted.

## Bench Ticket

Plan-only command run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_wc1_wallclock_bench.py --ticket-out .omx/research/ddm_wc2_20260808/wc2_bench_ticket.json --receipts-jsonl .omx/research/ddm_wc2_20260808/wc2_bench_receipts.jsonl --output-root /Volumes/VertigoDataTier/pact/ddm_wc2_20260808/wallclock_bench
```

Outputs:

| artifact | status | sha256 |
|---|---|---|
| `.omx/research/ddm_wc2_20260808/wc2_bench_ticket.json` | PLANNED ticket, not fired | `f0e6c97cb2773a7b5ff4c08b66f34c9ddacd5292a35394d08f25a2bfb35281b4` |
| `.omx/research/ddm_wc2_20260808/wc2_bench_receipts.jsonl` | 9 planned rows | `cbdf35e2c3958d7f21940066e628448cb602f24cc835f01c3471fa683cef142f` |
| `.omx/research/ddm_wc2_20260808/WC2_TYPED_ROWS.jsonl` | typed WC2 rows | `b095561d8d0ea02652d7819443eeebefcf6614f1914bb4082deb6cd1eddeb918` |

Fire order for MAIN, after a Metal slot is clear:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_wc1_wallclock_bench.py --ticket-out .omx/research/ddm_wc2_20260808/wc2_bench_ticket.json --receipts-jsonl .omx/research/ddm_wc2_20260808/wc2_bench_receipts.jsonl --output-root /Volumes/VertigoDataTier/pact/ddm_wc2_20260808/wallclock_bench --execute
```

## n120/n600 Recommendation

Use `--train-compute-dtype fp16` as the first n120 candidate only with the CPU-torch verdict gate active, because it was the fastest WC1 row and its d_seg sanity did not degrade in the five-step bench. Keep `--microbatch-policy auto` at 4-pair chunks until a fresh same-vehicle matrix proves a larger chunk is faster. Use `--cache-residency ram-full` only after its mem-probe row passes; otherwise keep selected-row cache loading.

For CPU verdict/facets, start with `--verdict-batch-size 16` when concurrent with Metal training and keep n600 scorer chunks `<=120` per the common contract. Raise verdict batch size only after the concurrent CPU row records RSS/elapsed evidence; do not infer from MLX memory telemetry.

## RECALL EVIDENCE

| scope | query / source | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing contract | `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `_common_contract.md` | CPU-only arm, no Metal fire, no scorer slot, shared dirty worktree, own frontier `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]` | Wrote plan-only tickets and default-off hooks; no n600 scorer or exact eval |
| Memory index | `rg "ddm_wc2|wc2|common_contract|waterfill|ddm_" MEMORY.md` | no direct wc2 precedent; prior Pact memory emphasized scoped denominators, source-verified queue ownership, and serializer/git sandbox caveats | Kept measured/derived/queued labels separate and planned for serializer failure |
| WC1 receipts | `.omx/research/ddm_wc1_20260807/wc1_bench_receipts.jsonl` | measured rows existed post-fire; `fp16-train` won and full n32 lost | microbatch auto keeps 4; n120 recommendation starts with fp16 plus CPU verdict gate |
| CoreML/ANE | `docs/mlx_contest_scorer_port_guide.md` | CoreML FP32 SegNet had prior 0/1.97M argmax-diff row, but ANE must still parity-gate on this vehicle | Implemented a real `coreml-segnet-parity` mode; no advisory use before parity |
| Custom Metal conv | `rg "#478|custom Metal conv"` + DAG feed | MAIN-local measured rows show pointwise/depthwise custom simdgroup kernels lose to MLX native on this host | Folded new custom-kernel port work for this arm; no duplicate Metal fire |
| Batch/memory saturation | `.omx/research/batch_saturation_throughput_20260623.json` and canonical task-status memory-ceiling row | larger batches had old killed rows and the 116 GiB ceiling survives | queued measured mem-probe variants instead of arithmetic promotion |

## Follow-on Disposition

| follow-on | disposition | fire order / reason |
|---|---|---|
| Extended WC2 bench matrix | QUEUED-WITH-FIRE-ORDER | run the `--execute` command above in a MAIN-owned Metal gap |
| ANE verdict use | QUEUED-WITH-FIRE-ORDER | contained in `ane-verdict`; use only if `coreml-segnet-parity` status is `passed` |
| Concurrent CPU verdict | QUEUED-WITH-FIRE-ORDER | contained in `concurrent-cpu-verdict`; subprocess group receipt required |
| Custom Metal conv port | FOLDED | #478 measured this formulation slower than MLX native; reopen only with a new sparse/different formulation |
| n120/n600 receiver build config | QUEUED-WITH-FIRE-ORDER | after extended bench, fire n120 with the measured winner, `--verdict-batch-size 16`, and scorer chunk `<=120` when the scorer slot is owned |

## Boundaries

- No Metal command was executed by WC2.
- No scorer slot, n600 job, archive build, exact eval, or upstream edit was performed.
- ANE/CoreML code was implemented but not run here; any ANE result remains pending the parity command.
- MPS/ANE are advisory only; CPU-torch remains the local authority for these verdict/facet rows.
- Score claim is false.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved at `0.1910828242`.
