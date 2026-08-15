# DDM WC1 HPAC throughput port — environment-blocked profile gate

**Disposition:** `QUEUED-WITH-A-FIRE-ORDER / BLOCKED-ENVIRONMENT`.
The required two-epoch reference profile could not be launched with the
charter-mandated `nice -n 10` priority inside this managed sandbox. The one
process that started after `nice` reported `Operation not permitted` was
stopped after 42.060 seconds and retained a governed killed receipt. No hot
stage was measured, so the profile-first gate forbids choosing or building a
throughput rung in this arm.

This is a score-neutral instrument arm. It did not create an RX2 candidate,
run a scorer, move the exact pointer, or change the live RX2 trainer or its
run directories.

## MEASURED OUTCOME

Axis: `[macOS-CPU operational telemetry; no score claim]`.

The live identity-pinned RX2 child remained PID `63183`. The current trainer
reports six Torch intra-op threads on an 18-core M5 Max host. Five one-second
samples immediately before the attempted profile measured live CPU use
`86.9, 111.5, 91.2, 119.4, 125.0%` and RSS between `8,722,366,464` and
`10,930,765,824` bytes. Five samples after the abort measured CPU use
`133.2, 194.6, 323.2, 188.8, 186.3%` and RSS between `7,494,107,136` and
`9,431,990,272` bytes. These short samples straddle different live-trainer
phases and are not a cadence comparison.

Durable telemetry:

- `/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/live_before.json`,
  1,104 B, SHA-256
  `7622331031d220f64e07406c2f1a9f416e0ad5e71c7904d76067ff1bfbad64de`;
- `/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/live_after_abort.json`,
  1,114 B, SHA-256
  `99afdc346cb0a87e17a97128fa95f15c7011f9fba5152032458ca6788c7a96c6`.

Storage routing was compliant: Vertigo had only 1.0 GiB free and APDataStore
had 491 GiB free, so all new bulk/receipts went to APDataStore. No artifact
was deleted or moved.

## PROFILE GATE RECEIPT

The exact current RX2 config was invoked with `--epochs 60` and
`--stop-after-epoch 2`; this preserves the embedded preregistration while
bounding the run to two epochs. It used separate save/out roots and cProfile:

```text
nice -n 10 env PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python tools/safe_run.py \
  --rss-mb 16384 --timeout 7200 --poll 0.2 \
  --label ddm_wc1_hpac_profile_cpu_reference \
  --status-receipt /Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/safe_run_status.json \
  --child-pidfile /Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/safe_run_status.json.child.pid \
  --quiet -- .venv/bin/python -m cProfile \
  -o /Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/profile.pstats \
  tools/train_ddm_cl1_hpac_capacity.py \
  --profile rx2_mc36 \
  --cache /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/inputs/mc36_spatial_tokens_uint8.pt \
  --init /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/tq1c/hpac_p64_exact_from_archive.pt \
  --epochs 60 --batch-size 8 --eval-batch-size 4 --eval-every 2 \
  --lr 0.003 --lr-exponent 0.0002 --lr-bits 0.01 --bit-eps 1e-6 \
  --rate-lambda 1.0 --qat-fraction 0.5 --init-bits 8.0 \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --target-mode raw \
  --seed 20260716 --ema-target-seed-fraction 0.01 --device cpu \
  --save /Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/checkpoints/profile_reference.pt \
  --out /Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/reports/profile_reference.json \
  --stop-after-epoch 2 --min-free-bytes 10737418240
```

`nice` printed `setpriority: Operation not permitted` but continued into the
child. The arm immediately treated that as noncompliant and sent interrupt to
the dedicated safe-run session. The retained status is `status=killed`,
`exit=130`, elapsed `42.060 s`, with a recorded
`SIGTERM_then_SIGKILL_process_group` external-signal action:

- `/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/safe_run_status.json`,
  2,645 B, SHA-256
  `2df49c05e71122127eb435aa9f33a5aef7e693542fc1f2c9ac8db45fb32ffcc4`.

The child is gone. `profile.pstats`, an epoch checkpoint, and a trainer report
do not exist. The only trainer-created durable file is the initial empty
resume-lineage JSON, 75 B, SHA-256
`82b4b5ffc7c08ffc4c7dce22bacc757435dd8fcea45c95f7ee7c8cdc103d9559`.
It was retained. No materialized payload was discarded.

Two bounded alternatives also failed before launching the trainer:

- `taskpolicy -b ...` returned rc `70` with
  `taskpolicy: setpriority(): Operation not permitted`;
- `launchctl submit` of a trivial `nice -n 10 sleep 8` returned rc `1`, so
  launchd could not provide a verified out-of-sandbox priority path.

## SPEEDUP AND PARITY TABLE

| column | result | boundary |
|---|---|---|
| measured hot stage | UNMEASURED | compliant two-epoch profile did not run |
| reference epoch seconds | UNMEASURED in this arm | recalled ~24 min is scheduling context, not this receipt |
| port epoch seconds | UNMEASURED | no port was selected or built |
| speedup | UNMEASURED | cannot claim the charter's >=3x gate |
| six-epoch bpp divergence | UNMEASURED | parity run not authorized past the profile gate |
| six-epoch top1 divergence | UNMEASURED | same |
| six-epoch joint-byte divergence | UNMEASURED | same |
| endpoint real IHS1 bytes | UNMEASURED | no parity endpoints exist |

No implementation file was created. Source inspection suggests several
possible hot regions, but selecting one without the required profile would be
the naive first pass that the charter explicitly forbids.

## RECALL EVIDENCE

Sources searched before adjudication:

- full-content queries `train_ddm_cl1_hpac_capacity`, `rx2_mc36`, `HPAC`,
  `IHS1`, `24 min/epoch`, `throughput`, `torch MPS`, `MLX`, `batch shape`,
  and `thread count` across `.omx/research/`, arm-final receipts,
  `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG, design/SPEC files, live
  state, lane/task/probe ledgers, tools, source, and tests;
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  HPAC, rate, entropy, training, EMA, batch, thread, throughput, and
  wall-clock surfaces;
- the real trainer, intake `hpac_integer.py` / `hpac_self_compress.py`, RX2
  identity-race packer, RX1 protocol, RX2 charter/memo, HB/HM/HP receipts,
  and the prior WC1/WC2 wall-clock receipts.

Beyond the charter seeds, the prior WC1/WC2 campaign supplied a useful
receipt pattern and warned that thread/batch/precision wins are
instrument-specific. Its measured speedups belong to a different MLX semantic
renderer and were not transferred to HPAC. The RX2 terminal race already
contains `_pack_terminal_ihs1`, which serializes the real IHS1 bytes and
checks exact deployed state plus logits after parse-back; that would be the
endpoint parity column, not `estimated_model_bytes`. The live RX2 memo also
confirms XI1/XI2 own context-extension work and that RX2 must remain the
matched HB1/HB2 spatial treatment. No completed HPAC throughput port or prior
profile of this exact trainer was found in the searched scope.

These findings changed the plan by (1) refusing borrowed speedup numbers,
(2) pinning parity to the existing shipping packer, and (3) refusing any
mechanism or context change before a measured hot-stage receipt.

## REPOSITORY AND AUTHORITY BOUNDARIES

- The staged index was empty at intake and was not touched except for the
  serializer-owned staging of this memo.
- The worktree contained extensive unrelated user/agent changes, including a
  pre-existing modification to the protected
  `src/tac/optimization/direct_description_carrier_compose.py`; none were
  modified, staged, or committed by this arm.
- `tools/train_ddm_cl1_hpac_capacity.py`, the live RX2 directories, sealed
  race gates, and `upstream/` were not edited.
- No scorer slot was owned or used. No scorer, pack, archive, or exact eval
  ran. MPS was not used as authority.
- The exact own frontier is unchanged: MC36 Variant C
  `S = 0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`, archive
  SHA-256
  `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / BLOCKED-ENVIRONMENT** — owner: MAIN or an
  unsandboxed WC1 executor; consumer store:
  `/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/`; fire
  trigger: an execution context proves `nice -n 10` actually sets priority
  10, the live RX2 PID/cadence guard is healthy, and the exact two-epoch
  cProfile command above is rerun in a fresh output root.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: successor WC1 builder; consumer store:
  `.omx/research/ddm_wc1_hpac_throughput_port_<date>.md`; fire trigger: the
  compliant profile names its measured hot stage; implement only the first
  CPU-vectorization, MPS, or MLX rung that measures at least 3x faster.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: successor WC1 verifier; consumer
  store: the same SSD WC1 root plus the receipt memo; fire trigger: a >=3x
  rung exists; run identical-seed six-epoch reference/port parity with at
  least two QAT epochs, retain every endpoint, and serialize both through the
  real IHS1 packer.

## LIVE-HYPOTHESES

- Convolution forward/backward is likely the dominant stage because every
  batch executes full-resolution masked 7x7 convolution plus depthwise 5x5
  and 3x3, past-context convolution, SPM convolutions, and backward. This is
  plausible from the real model source, but remains untested until cProfile
  assigns time.
- Torch MPS may clear 3x if convolution/backward dominates. The model is
  convolution-heavy and the M5 Max has unified memory, but integer-style STE,
  one-hot expansion, deterministic-algorithm support, and transfer/sync costs
  can erase that advantage; only the six-epoch trajectory and pack parity can
  promote it.
- A CPU rung may exist in avoiding repeated layout/one-hot materialization or
  increasing batch work per dispatch. This is plausible from the repeated
  `_to_patches`, `contiguous`, and `one_hot` calls, but batch shape is a new
  instrument and no edit is justified before the hot-stage breakdown.

## DEAD-ENDS

- Literal `nice -n 10` from this managed sandbox is closed as an execution
  route: self `setpriority`, `/usr/bin/nice`, and `taskpolicy` all return
  `Operation not permitted`.
- Launchd handoff from this sandbox is closed as a verified niceness route:
  the bounded `launchctl submit` control returned rc 1.
- Transferring WC1/WC2 renderer speedups to HPAC is closed: those receipts
  measure a different MLX model, loss, batch geometry, and hot stage.
- Building from source inspection alone is closed by the charter's
  profile-first rule. A plausible hot-stage guess is not a measurement and
  cannot authorize a throughput port.
