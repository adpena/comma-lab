# ddm_js1b CUDA argmax-field materializer

**Status: `READY_TO_FIRE`; no Modal job or scorer ran in this arm.** The exact contest pointer and
the own-vehicle frontier are unchanged.

## Deliverable

The JS1 Stage-0 axis blocker now has a governed T4 materializer and a scorer-free local consumer:

- `experiments/ddm_js1b_modal_cuda_argmax_field_materializer.py` packages the exact CP135 and T1R1
  archives and adapted runtime trees, binds the retained C1 control field, claims the Modal lane,
  enforces single flight, records the call ID, and commits the retained volume every 20 seconds.
- `experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py` runs each runtime's unmodified
  `inflate.sh` on a T4, retains both complete raw videos, decodes GT with the exact upstream DALI
  batch-16 surface, and retains every GT RGB batch, SegNet input, logit tensor, and argmax field.
- `experiments/ddm_js1_stage0_per_edge.py summarize --from-argmax-fields ...` consumes only a
  content-hash-bound downloaded field bundle. It runs no local forward and refuses to decompose
  unless the remote controls are exactly CP135 `34,964` and C1 `17,926` flips.

The remote result is a component receipt, not a score. Its axis is
`[contest-CUDA T4 frozen-SegNet argmax fields, n600, batch=16] COMPONENT-ONLY` and it forces
`score_claim=false`, `promotion_eligible=false`, and `pointer_moved=false`.

## Payload retention and resume contract

Every materialized evidence payload remains on `comma-ddm-js1b-argmax-retained` under the immutable
run ID. The retained payload denominator before metadata is:

| retained payload | bytes |
|---|---:|
| two exact receiver raw videos | 7,324,819,200 |
| all 600 decoded GT RGB pairs | 3,662,409,600 |
| three full SegNet-input populations | 4,246,732,800 |
| three full float32 logit populations | 7,077,888,000 |
| GT, CP135, T1R1, and C1-control NPY argmax fields | 471,859,712 |
| **total before metadata/runtime bundles** | **22,783,709,312** |

The worker fails closed unless the retained tier has the remaining payload capacity plus a 4 GiB
reserve. Inputs, runtime extraction, each exact receiver, each of the three scorer passes, and the
final receipt have distinct immutable checkpoints. Each scorer pass also writes an atomic progress
receipt after every batch. A killed receiver's unreceipted raw is moved losslessly into a distinct
`failed_attempts/` record before the exact deterministic rerun; it is never deleted or overwritten.

## K=2 time arithmetic

This is a projection from already measured T4 component times, not a JS1B measurement:

| term | arithmetic | seconds |
|---|---:|---:|
| exact receiver decodes | `2 x 466.0` | 932.000 |
| GT + CP135 + T1R1 SegNet passes | `3 x 39.405` | 118.215 |
| subtotal | | 1,050.215 |
| reserve | | 300.000 |
| **projected total** | | **1,350.215** |
| 30-minute limit headroom | `1800 - 1350.215` | **449.785** |

Epistemic label: `DERIVED_FROM_MEASURED_PRIOR_T4_COMPONENT_TIMES_NOT_YET_JS1B_MEASURED`.

## Exact MAIN fire and consume commands

Before firing, MAIN reconciles the active-claims and call-ID ledgers. The dispatcher performs the
claim and single-flight checks itself.

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/ddm_js1b_modal_cuda_argmax_field_materializer.py::main \
  --run-id ddm_js1b_20260813 \
  --resume-from ddm_js1b_20260813 \
  --lane-id ddm_js1b_cuda_argmax_field_materializer \
  --instance-job-id modal:ddm_js1b_20260813 \
  --claim-agent main:ddm_js1b \
  --output-dir /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813_dispatch \
  --detach \
  --provider-detach-ack
```

Harvest the detached call:

```bash
.venv/bin/python experiments/ddm_js1b_modal_cuda_argmax_field_materializer.py recover \
  --output-dir /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813_dispatch
```

Download only the receipt and the four fields; the roughly 22.8 GB source/scorer custody remains on
the retained volume:

```bash
.venv/bin/modal volume get --force comma-ddm-js1b-argmax-retained \
  ddm_js1b_20260813/FINAL_RESULT.json \
  /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813/FINAL_RESULT.json
```

```bash
.venv/bin/modal volume get --force comma-ddm-js1b-argmax-retained \
  ddm_js1b_20260813/retained/fields/ \
  /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813/retained/fields/
```

Run the scorer-free per-edge consumer:

```bash
.venv/bin/python experiments/ddm_js1_stage0_per_edge.py summarize \
  --output /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/stage0_from_js1b \
  --from-argmax-fields /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813
```

If either control differs, the worker emits `BLOCKED_AXIS_MISMATCH` and the local consumer refuses
to run. That outcome is a field/custody finding; MAIN must not fire V0-V5 from it.

## RECALL EVIDENCE

The full durable corpus was queried before implementation. Bounded content searches covered the
research index, sub-0.15 DAG, research memos/receipts, hot state, P0 ledger, canonical-equation
registry, exact evaluator, and both retained archive/runtime trees. Exact query families were:

- `BLOCKED_AXIS_MISMATCH 50395 47950 17927 34964 17926`
- `DaliVideoDataset batch16 argmax field CUDA CPU`
- `ddm_vd1 lazy mount top-level reserved kwarg Brotli Modal Volume`
- `per-edge Road Lane rho_required m91 m94`
- `et4 batch shape instrument`
- `CP135 T1R1 adapted runtime cpr1 exact receiver`

Findings beyond the charter seeds changed or confirmed the implementation:

1. JS2 proves the local and promoted CP135 `0.raw` hashes already differ before scoring. Therefore
   changing only the local scorer batch cannot repair the 15,431-flip gap; both exact receivers must
   run on the T4 surface.
2. ET4 proves batch shape is part of the forward instrument because oneDNN batch-1 and batch-16
   forwards differ at a tie-adjacent argmax pixel. The worker therefore pins batch 16, threads 2,
   prefetch 4, and seed 1234 exactly as the promoted evaluator.
3. VD1b proves bypassing the adapted entrypoint loses its Brotli bootstrap and that the decoder's
   canonical code directory is `cpr1/`. JS1B therefore executes each shipped `inflate.sh` unchanged
   instead of hand-rolling the parser or token decoder.
4. M91/PC2 and the LV2 crosswalk require one all-edge graph: Road participates in 87.8% of the older
   field, but lane-only accounting is incomplete. The local consumer preserves all 20 directed cells
   and all 10 undirected interfaces.
5. M94 requires instrument and object capacity in the same units. The admitted object is the full
   117,964,800-pixel CUDA field, not the previous zero-CUDA-pixel local instrument.
6. The retained C1 target field is a separate control object. T1R1's archive output is the composed
   candidate and is not substituted for the `17,926`-flip C1 reference.

The equation search found no registered equation that displaced exact archive identity, exact
receiver use, batch-16 control reproduction, or the full-population denominator.

## Unified-Lagrangian wire-in

- **Sensitivity-map contribution: N/A** - this unit materializes complete argmax fields and does
  not create or modify a sensitivity map.
- **Pareto constraint: N/A** - no candidate, score row, or Pareto selection is produced.
- **Bit-allocator hook: N/A** - archive bytes and allocation are unchanged.
- **Cathedral autopilot dispatch hook: ACTIVE** - the fire path uses the canonical claim,
  single-flight, call-ID, recovery, and terminal-claim surfaces.
- **Continual-learning posterior update: N/A** - no exact score or promotable observation exists.
- **Probe-disambiguator: ACTIVE** - CP135 `34,964` and C1 `17,926` are independent hard controls;
  mismatch routes to a custody question and cannot silently become a Stage-0 rho.

## Validation and honest boundary

- Focused tests: `11 passed` across the new JS1B suite and existing JS1 Stage-0 tests.
- Ruff and Python compilation: passed for all touched Python files.
- P0 measure-and-discard audit: zero findings for the dispatcher, worker, consumer, and focused test.
- Modal dispatched: false. Scorer ran: false. New argmax fields: not measured in this arm.
- Upstream, both archives, and both adapted runtime custody trees: read-only and unmodified.
- Borrowed substrate: Modal auth image, upstream evaluator/datasets/SegNet, CP135/T1R1 archives and
  receivers, C1 target field, VD1 dispatch governance, and JS1 decomposition math. Original work in
  this unit is their exact dual-object materializer, payload/resume contract, admission binding, and
  scorer-free field consumer.

The effective frontier remains CP135 `S=0.16195513827824176 @ 186,252 B`
`[contest-CUDA T4, n600]`. The own-vehicle frontier remains LC2
`S=0.16959899569230852 @ 187,226 B` `[contest-CUDA T4, n600]`. This unit did not achieve sub-0.15.
