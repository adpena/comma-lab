# DDM WC2 HPAC MPS throughput port — built, parity and race gated

**Disposition:** `BUILT / QUEUED-WITH-A-FIRE-ORDER / BLOCKED-ENVIRONMENT`.

The Torch-MPS training instrument, matched six-epoch CPU/MPS comparator, real
IHS1 endpoint retention, timing projection, and governed watcher configs are
built. They have not run on Metal: this execution context reports Torch MPS as
built but unavailable. The matched CPU parity run also did not launch because
this managed context cannot apply or verify niceness 10. Nothing in this arm is
a score row, and the exact frontier is unchanged.

## Measured routing facts

Axis: `[macOS-CPU operational telemetry; no score claim]`.

| fact | measured value | provenance |
|---|---:|---|
| reference profile wall | 2,486.478 s | real RX2 config, two epochs plus three evals |
| `torch.conv2d` forward | 1,282.433 s, 51.6%, 4,200 calls | `profile.pstats` |
| autograd backward | 1,008.073 s, 40.5% | `profile.pstats` |
| conv forward plus backward | about 92% | sum of measured profile components |
| peak RSS | 10,524.375 MiB = 10.278 GiB | successful `safe_run` receipt |
| measured thread need | 6 | reference trainer configuration |

The profile is
`/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference_main/profile.pstats`,
SHA-256 `aa958d0857707348ad3ffd27801c65ea438e07804f92316d920f6ec762068a15`.
The reference trainer remains unedited at SHA-256
`8392a9b9f2d303698de59e627fa489a792ab0b0b38170cebd425f9310162059e`.
The intake model remains SHA-256
`6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f`.

The measured cost is inside convolution forward/backward, so the CPU
vectorization rung is closed for this profile. The admitted first build is the
Torch-MPS device port.

## Built instrument

`tools/train_ddm_cl1_hpac_capacity_mps.py` imports the exact hash-pinned
reference trainer and changes only the device and epoch admission envelope.
`parity-cpu` and `parity-mps` both run six epochs with QAT beginning at epoch
four; epochs four, five, and six are QAT, and epochs four and six are evaluated.
`full-mps` runs the sealed 60-epoch treatment. The reference model, STE, loss,
optimizer, scheduler, EMA, evaluation, checkpoint, and result JSON code is not
copied or altered.

MPS runs are labeled `mps_trained=true`; the run identity says they are seeded
but not bit-reproducible. Their retained checkpoint and manifest hashes are the
reproducibility anchors. `PYTORCH_ENABLE_MPS_FALLBACK=0` is mandatory. A
successful six-epoch run therefore proves kernel availability for every model,
STE, backward, optimizer, and EMA operation actually exercised.

`tools/compare_ddm_cl1_hpac_capacity_mps.py` rejects source drift, unmatched
configs, malformed trajectories, nonterminal checkpoints, invalid causal
hashes, and failed timing receipts. It reports the maximum relative divergence
over bpp, top-1 error, and estimated joint bytes at epochs 1, 2, 4, and 6. It
then packs each terminal checkpoint twice through the real RX2 IHS1 packer,
retains all four raw and four XZ payloads, inventories their bytes and SHA-256,
and requires exact repeat identity. Its output is explicitly not score
authority.

The race projection is generated from the two successful detached-launch
receipts:

```text
cpu_s_per_epoch = cpu_parity_elapsed_s / 6
mps_s_per_epoch = mps_parity_elapsed_s / 6
port_speedup = cpu_s_per_epoch / mps_s_per_epoch
cpu_remaining_hours = (60 - live_cpu_epoch) * cpu_s_per_epoch / 3600
full_mps_hours = 60 * mps_s_per_epoch / 3600
finish_margin_hours = cpu_remaining_hours - full_mps_hours
```

Only a positive final subtraction says the MPS race is projected to finish
first; this is a projection, not a completed-race measurement.

## Exact MAIN fire order

All three launch roots below are currently absent. The launcher fails closed if
a root is no longer fresh, derives its resource envelope from the measured
10.278 GiB and six-thread need, records the 116 GiB governor ceiling, and arms
both canonical watchers before releasing the child. The commands intentionally
do not use the obsolete 16,384 MiB literal.

### 1. Matched CPU parity

Fire only where niceness 10 can be applied and verified while the live RX2
cadence remains healthy.

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 \
.venv/bin/python tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/cpu/launcher \
  --cwd /Users/adpena/Projects/pact \
  --purpose "rx2 WC2 matched six-epoch CPU parity" \
  --authority "macOS CPU-vs-MPS training parity; no score claim" \
  --env PYTHONHASHSEED=0 --env TAC_ADMISSION_ENFORCE=1 \
  --env PYTORCH_ENABLE_MPS_FALLBACK=0 \
  --fresh-root /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/cpu \
  --nice 10 --derive-resource-budgets \
  --measured-peak-rss-gib 10.278 --measured-thread-need 6 \
  --walltime-cap-s 21600 \
  --done-receipt rx2_wc2_parity_cpu \
  --arm-watchers \
  --liveness-config .omx/research/ddm_wc2_hpac_mps_port_20260814/parity_cpu_liveness.json \
  --quality-config .omx/research/ddm_wc2_hpac_mps_port_20260814/parity_cpu_quality.json \
  -- .venv/bin/python tools/train_ddm_cl1_hpac_capacity_mps.py \
  --port-mode parity-cpu --profile rx2_mc36 \
  --cache /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/inputs/mc36_spatial_tokens_uint8.pt \
  --init /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/tq1c/hpac_p64_exact_from_archive.pt \
  --epochs 6 --batch-size 8 --eval-batch-size 4 --eval-every 2 \
  --lr 0.003 --lr-exponent 0.0002 --lr-bits 0.01 --bit-eps 1e-6 \
  --rate-lambda 1.0 --qat-fraction 0.5 --init-bits 8.0 \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --target-mode raw \
  --seed 20260716 --ema-target-seed-fraction 0.01 --device cpu \
  --save /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/cpu/checkpoints/parity_cpu.pt \
  --out /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/cpu/reports/trainer.json \
  --min-free-bytes 10737418240
```

### 2. Matched MPS parity

Fire on the Metal host only after `torch.backends.mps.is_available()` is true.
Fallback remains zero.

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
.venv/bin/python tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/mps/launcher \
  --cwd /Users/adpena/Projects/pact \
  --purpose "rx2 WC2 matched six-epoch MPS parity" \
  --authority "macOS CPU-vs-MPS training parity; MPS is not score authority" \
  --env PYTHONHASHSEED=0 --env TAC_ADMISSION_ENFORCE=1 \
  --env PYTORCH_ENABLE_MPS_FALLBACK=0 \
  --fresh-root /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/mps \
  --nice 10 --derive-resource-budgets \
  --measured-peak-rss-gib 10.278 --measured-thread-need 6 \
  --walltime-cap-s 21600 \
  --done-receipt rx2_wc2_parity_mps \
  --arm-watchers \
  --liveness-config .omx/research/ddm_wc2_hpac_mps_port_20260814/parity_mps_liveness.json \
  --quality-config .omx/research/ddm_wc2_hpac_mps_port_20260814/parity_mps_quality.json \
  -- .venv/bin/python tools/train_ddm_cl1_hpac_capacity_mps.py \
  --port-mode parity-mps --profile rx2_mc36 \
  --cache /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/inputs/mc36_spatial_tokens_uint8.pt \
  --init /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/tq1c/hpac_p64_exact_from_archive.pt \
  --epochs 6 --batch-size 8 --eval-batch-size 4 --eval-every 2 \
  --lr 0.003 --lr-exponent 0.0002 --lr-bits 0.01 --bit-eps 1e-6 \
  --rate-lambda 1.0 --qat-fraction 0.5 --init-bits 8.0 \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --target-mode raw \
  --seed 20260716 --ema-target-seed-fraction 0.01 --device mps \
  --save /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/mps/checkpoints/parity_mps.pt \
  --out /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/mps/reports/trainer.json \
  --min-free-bytes 10737418240
```

### 3. Parity comparison, retained endpoint packs, and race arithmetic

Fire after both parity done receipts have `rc=0`. If the live CPU run rotates
from `detached_resume_r5`, update only `--live-cpu-log` to the active lineage
log before firing; otherwise this exact command applies.

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 \
.venv/bin/python tools/compare_ddm_cl1_hpac_capacity_mps.py \
  --cpu-result /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/cpu/reports/trainer.json \
  --mps-result /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/mps/reports/trainer.json \
  --cpu-checkpoint /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/cpu/checkpoints/parity_cpu.checkpoints/qat_stage_end_epoch_0006.pt \
  --mps-checkpoint /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/mps/checkpoints/parity_mps.checkpoints/qat_stage_end_epoch_0006.pt \
  --cpu-done-receipt /Users/adpena/Projects/pact/.omx/tmp/codex_runs/rx2_wc2_parity_cpu.done \
  --mps-done-receipt /Users/adpena/Projects/pact/.omx/tmp/codex_runs/rx2_wc2_parity_mps.done \
  --live-cpu-log /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/training/detached_resume_r5/run.log \
  --output-root /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/comparison
```

### 4. Full MPS race

Fire only after the comparator completes, every pack repeat is exact, MAIN has
classified the measured trajectory gap, and the projection's final subtraction
is positive. A material trajectory divergence is a different instrument and
must not be promoted by calling it parity.

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
.venv/bin/python tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full/launcher \
  --cwd /Users/adpena/Projects/pact \
  --purpose "rx2 WC2 full 60-epoch MPS throughput race" \
  --authority "MPS training research signal; CPU pack remains serialization authority" \
  --env PYTHONHASHSEED=0 --env TAC_ADMISSION_ENFORCE=1 \
  --env PYTORCH_ENABLE_MPS_FALLBACK=0 \
  --fresh-root /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full \
  --nice 10 --derive-resource-budgets \
  --measured-peak-rss-gib 10.278 --measured-thread-need 6 \
  --walltime-cap-s 86400 \
  --done-receipt rx2_wc2_full_mps \
  --arm-watchers \
  --liveness-config .omx/research/ddm_wc2_hpac_mps_port_20260814/full_mps_liveness.json \
  --quality-config .omx/research/ddm_wc2_hpac_mps_port_20260814/full_mps_quality.json \
  -- .venv/bin/python tools/train_ddm_cl1_hpac_capacity_mps.py \
  --port-mode full-mps --profile rx2_mc36 \
  --cache /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/inputs/mc36_spatial_tokens_uint8.pt \
  --init /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/tq1c/hpac_p64_exact_from_archive.pt \
  --epochs 60 --batch-size 8 --eval-batch-size 4 --eval-every 2 \
  --lr 0.003 --lr-exponent 0.0002 --lr-bits 0.01 --bit-eps 1e-6 \
  --rate-lambda 1.0 --qat-fraction 0.5 --init-bits 8.0 \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --target-mode raw \
  --seed 20260716 --ema-target-seed-fraction 0.01 --device mps \
  --save /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full/checkpoints/full_mps.pt \
  --out /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full/reports/trainer.json \
  --min-free-bytes 10737418240
```

## MLX and deterministic-Metal adaptation cost

The recalled fixed-point/Metal assets are real, but they are not a drop-in HPAC
training backend. `metal_fixedpoint_verdict.py` and its exact-int64 twins are
forward/verdict machinery for frozen weights. `calibrated_fixedpoint_scorer.py`
can derive dynamic exact-absmax scales, but does not define useful gradients
through evolving quantization scales. `metal_grouped_conv_backward.py` supplies
an MLX NHWC grouped/depthwise VJP and measured large speedups on the scorer
vehicle, not this HPAC graph.

A real MLX HPAC port therefore still owes the dense masked 7x7 path, the
depthwise 5x5/3x3 paths, layout conversion, one-hot and STE semantics, optimizer,
scheduler, EMA, RNG/resume state, checkpoint conversion, and CPU IHS1 endpoint
parity. An int64 training version additionally owes a defined gradient for
dynamic range handling as weights evolve. That is a high adaptation cost and
no prior scorer speedup transfers numerically. MLX remains a co-equal live rung
if raw MPS is slow or materially divergent; it was not speculatively built in
this arm.

## RECALL EVIDENCE

The pre-build search covered full-content queries for `HPAC`, `rx2_mc36`,
`train_ddm_cl1_hpac_capacity`, `torch MPS`, `MLX`, `fixedpoint`, `conv2d`,
`throughput`, `IHS1`, `EMA`, `MPS noise`, and `deterministic` across
`.omx/research/`, arm-final receipts, canonical research indexes, the sub-0.15
DAG, design/SPEC files, the task and probe ledgers, source, tools, and tests.
The canonical-equation registry was queried for MPS drift, portability,
fixed-point accumulation, EMA, reproducibility, and floating-point reorder
surfaces. The real trainer, intake model, RX2 race packer, WC1 profile receipt,
and the local-acceleration assets named in the charter addendum were read.

Beyond the charter seeds, recall found that the RX2 race module already owns a
real `_pack_terminal_ihs1` path with parse-back checks, so endpoint-byte parity
uses that code instead of a new surrogate. The equation registry's
architecture-dependent MPS-drift and floating-point reorder rules ruled out
borrowing old GPU speed/parity claims. The custom grouped-backward speedup is
real but scorer-vehicle-specific, and the fixed-point paths are frozen-forward
instruments rather than an evolving-weight training implementation.

These findings changed the build by pinning both the reference trainer and real
packer source hashes, retaining every raw/XZ endpoint pack twice, reporting raw
MPS as a potentially different instrument, and leaving MLX live but unbuilt
until measured MPS parity or speed justifies its substantial adaptation cost.

## Repository custody

Repository HEAD moved beyond the charter's snapshot to
`6504608bd5a973ecc761d7edfec6f49baafecc76`, while the reference trainer's
required content hash remained exact. The worktree and protected
`src/tac/optimization/direct_description_carrier_compose.py` already contained
unrelated user/agent changes; this arm did not modify, stage, or absorb them.

The required serializer was invoked with all eleven files explicitly named,
`new` base declarations, post-edit SHA-256 declarations, `[no-triality]`,
`[p0-ledger-ok]`, and no co-author trailer. Git refused before staging:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_wc2_hpac_mps_port_20260814.md: failed to insert into database
fatal: adding files failed
```

The staged index remained empty. The implementation and receipt are verified
but uncommitted in this checkout; custody must be retried only through the same
serializer in a Git-writable execution context.

## Verification and authority boundary

Focused lint, compilation, and tests cover the wrapper envelope, immutable
reference hash, unchanged config delta, MPS provenance, exact trajectory shape,
config mismatch refusal, source pins, relative divergence, and race arithmetic.
Watcher configs validate through their canonical executables. The three
launcher commands validated in dry-run mode. Dry-run persisted three small
launch manifests inside the reserved roots; they were losslessly moved to
`/Volumes/APDataStore/pact/ddm_wc2_hpac_mps_port_20260814/dry_run_validation/gpu_race/`
under `DRY_RUN_RETENTION_MANIFEST.json` (2,140 B, SHA-256
`7b87b37705bb25c2bb149918a7d2a11d1cac6069096a464a6692accbabaaea2a`).
All source bytes and AppleDouble metadata were retained, and the three real
launch roots are absent again.

This arm did not touch the live trainer, live run directories, sealed race
gates, armed receipts, scorer, or upstream evaluator. The existing exact own
frontier remains MC36 Variant C at
`S=0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`, archive SHA-256
`f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`.
This WC2 arm did not move it.
