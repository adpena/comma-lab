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

## MAIN trajectory-gap classification (2026-08-14, comparator run comparison_r4)

**Verdict: DIFFERENT-INSTRUMENT, ADMITTED FOR THE RACE.** Receipt:
`/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/comparison_r4/PARITY_RESULT.json`.

The measured gates, in order of the §4 contract:

1. **Comparator completed** — schema `ddm_cl1_hpac_capacity_mps_parity.v1`,
   all materialized payloads retained (4 endpoint packs, raw + xz, SHA-pinned).
2. **Every pack repeat exact** — CPU endpoint `deterministic_repeat_exact=true`
   (raw sha `97330ccd…` identical across repeats, 24,937 B) and MPS endpoint
   `deterministic_repeat_exact=true` (raw sha `527346ec…` identical, 24,917 B).
   Decode logit diff 0.0 on both. `verified_exact_semantics =
   idempotent_pack_and_deterministic_decode` — the custody invariant that
   survived the two latent-blocker fixes (schema modulo the 9 training-only
   `.bit_depth` buffers; quantization delta demoted to QAT telemetry).
3. **Projection subtraction positive** — `finish_margin_hours = +13.11`
   (cpu remaining 14.05 h − full-mps 0.93 h), measured speedup 18.81×,
   `projection_not_measurement=true` honored.
4. **The gap itself** — relative divergence compounds through QAT: bpp 0.026%
   at ep0 → 0.375% at ep6 (joint-bytes 0.294%, top1 0.220%). This is float
   drift through a quantization-aware nonlinearity under full Metal kernel
   coverage (`PYTORCH_ENABLE_MPS_FALLBACK=0`, rc=0), not a defect. It is
   MATERIAL as parity and therefore NOT called parity: MPS is admitted as a
   second SEARCH instrument, not a bit-parity clone. No promotion claim rides
   parity. The authority boundaries that make this safe: (a) the sealed race
   gates in `experiments/ddm_rx2_mc36_identity_race.py` are device-blind —
   they gate the terminal checkpoint's IHS1 pack, archive bytes, and T4 exact
   eval; (b) serialization authority stays the CPU IHS1 pack path, proven
   idempotent + decode-deterministic on BOTH endpoints above; (c) score
   authority is T4 exact eval only; the MPS trainer.json is labeled
   `[macOS-MPS research-signal]`.

Honest note: the CPU ep6 endpoint was marginally better (est joint 140,969 B
vs 141,384 B, 0.29%). The live CPU run continues untouched, so the race yields
TWO terminal endpoints and the sealed device-blind chain adjudicates them; a
losing MPS endpoint costs nothing but the ~0.93 h of Metal time.

## §5 Extended-epoch run (operator 2026-08-14: "Because it's so much faster, we can give it more time. More epochs I mean. Once we confirm working and understand the trajectory better.")

The 60-epoch budget was sized by the CPU instrument's economics (60 × 17.6 min
≈ 17.6 h). At the measured 56 s/epoch on MPS, epochs are ~19× cheaper, so the
budget constraint moved. Protocol, in order — no step fires early:

1. **Confirm working** — `rx2_wc2_full_mps` receipt rc=0 AND the terminal
   epoch-60 pack passes the same custody invariants comparison_r4 proved
   (idempotent pack, deterministic decode).
2. **Read the trajectory** — the full eval-every-2 curve from the 60-epoch
   run (continuous ep0–30, QAT ep31–60). Classify: knee-at-K vs
   still-descending-at-60, separately for the continuous and QAT phases.
   Early live evidence: MPS ep12 bpp 0.0078589, slightly ahead of the CPU
   instrument's same-epoch 0.0078908; descent healthy.
3. **Size from the curve, then fire** — extended run with ALL constants
   identical except `--epochs N` (pure SCOPE extension: the phase split and
   `--ema-target-seed-fraction` derive from total epochs by design, so no
   latched per-epoch constant is silently re-scoped; candidate default
   N=480 ≈ 7.5 h, adjusted at the boundary from the measured slope).
   Fresh root `gpu_race/full_eN`, receipt `rx2_wc2_full_mps_eN`, watcher
   configs derived from the full_mps pair with phase-knee epoch = N/2+1.
   Single Metal fire: the extended run launches only after the 60-epoch
   receipt (governor law) and only from this protocol.
4. **Owed edit at extended harvest** — the sealed identity race hard-pins
   epoch 60 three ways (`TERMINAL_CHECKPOINT` :53 filename +
   `checkpoint.epoch != 60` :188 + `history[-1].epoch != 60` :206) and
   points at the CPU lineage checkpoint dir. Parameterize these to the
   declared run config (checkpoint path + terminal epoch as inputs, gates
   otherwise unchanged) in ONE commit that also updates the comparator's
   `RACE_PACKER_SHA256` pin. This edit is owed for the MPS 60-epoch
   endpoint too — it is the same generalization, done once.

### §5a Equations-leg instrument (operator recall 2026-08-14: EventGated / canonical-equations / DE discipline)

The sizing step is now LAW-DERIVED, not eyeballed: `tools/fit_hpac_descent_law.py`
fits exponential-to-floor vs power-law per phase over the retained eval rows,
selects by SSE, and derives N* against byte bars in canonical-band units
(1 band = 3.5e-6 S = 5.256 B on the rate term). Partial fit on the live race's
first 10 continuous points (receipt `gpu_race/full/descent_law_fit_partial.json`):
exp_floor wins (asymptote ≈ 136,413 B joint-est, τ ≈ 7.6 ep); remaining
continuous gain at phase exit: N=60 → ~190 B · N=120 → ~3.6 B (<1 band) ·
N=240 → ~0. Preliminary implication: continuous saturates near ep~60, so the
law points at N≈120–240, NOT the naive 480 default — the constants-are-poison
cure paying out in real time. Caveats owed at the boundary refit: 10 early
points can under-estimate a slow second timescale; QAT phase entirely unfitted
until the 60-ep run's ep32–60 rows land (and stays ENTRY-STATE-CONDITIONAL).
At the fit boundary, register the fitted law as canonical equation
`hpac_mc36_joint_descent_law_v1` (producer: the fitter; anchors: fit receipt +
log shas; consumers: extension sizing + the race epoch-gate parameterization).

### §5b Resume-vs-fresh adjudication (operator 2026-08-14 "Shouldn't it be able to just be resumed?")

YES in architecture, NOT YET in the gate — adjudicated at source:
- The natural resume point EXISTS by design: `continuous_stage_end_epoch_0030.pt`
  is written at qat_start−1 (trainer:734/:1292), plus per-eval periodic
  checkpoints, all under sha-pinned resume lineage (the r5 CPU run is itself a
  5th-generation resume of this machinery).
- The schedule is horizon-DERIVED throughout: qat_start = f(epochs, fraction)
  (:1041), cosine LR T_max = epochs (:1037), EMA decay from
  --ema-target-seed-fraction. Extension-resume is therefore the law-following
  shape: change ONLY the horizon, everything re-derives.
- BUT the resume identity gate REFUSES it: `epochs` is in the compared
  training_config (:616–645 keys, refusal at :1146), the restored scheduler
  state bakes in the OLD T_max, and the ema_policy equality check (:~1152)
  refuses the re-derived decay. Three named touch points.

DECISION for THIS extension: FRESH-FIRE, not resume. Economics: the patch
touches a sha-pinned custody surface (run identity + EMA policy + scheduler
restore) whose pins cascade through the wc2 wrapper (reference_trainer sha
8392a9b9…) and the parity evidence chain — a careful ~1h two-landing change —
versus ~28 min of Metal (30 continuous epochs × 56 s) saved by resuming.
Fresh-fire also keeps the extension a pure single-variable run against the
fitted law. The 60-epoch run is NOT waste: it is the calibration instrument,
the QAT-curve producer, and a sealed-race entrant in its own right.

CAPABILITY GAP FILED (dt1 determinization genus — make the law-following path
legal): typed EXTENSION-RESUME on this trainer — identity gate treats `epochs`
as extension-legal (grow-only, recorded in lineage as an extension event),
scheduler re-derived at the new horizon then advanced to start_epoch, EMA
policy re-derived with the shadow carried. Lands as a deliberate two-landing
patch with pin updates at a quiet boundary, never mid-race.

### §5c Truly-optimal composition boundary (operator: "other things to compose and include?")

IN the extended run (law-derivations only, no new mechanisms):
law-derived N (§5a fitter, refit with QAT data) · phase boundary AT the
measured knee (choose N × fraction so qat_start ≈ knee; N=120 @ 0.5 → 61 ≈ the
partial-fit knee 57–60) · EMA + cosine horizons re-derive automatically ·
canonical watchers + QAT-knee serialization cadence (sp2 analog).

AT the terminal chain, NOT in the run (banked, owners live): qs2 −4.375e-6
(+34 B) + re1 −1.207e-6 (0 B) micro-edit bank (edits the token FIELD — compose
edits then re-encode with the winning model; union fire still held at the
≥1e-5 pool bar) · pz4/pz4a variable-precision pose recode (2,000 B pre-proof
gate) · HP4 repack. They meet the race at its parameterized epoch gate (§5
owed edit).

REJECTED in-run: any mechanism change (architecture/lr/activation/coder) —
breaks the single-variable evidence chain and the comparator's parity basis;
new mechanisms are separate raced arms per charter-time optimal form.

### §5d Loss/schedule adjudication (operator 2026-08-14 "better loss than just cosine or whatever... I don't know if that works for this")

LOSS: already optimal for THIS vehicle — trainer :1234-1236 is
CE(tokens) + rate_lambda·ln2·model_bits/pixels, i.e. the training loss IS the
deployed joint-rate objective (source coding theorem: CE nats = arithmetic-
coder bits). Zero proxy gap; witness-line loss machinery cures a gap this
vehicle does not have. NO CHANGE.

SCHEDULE: cosine (T_max=epochs, :1037) is the weak inherited piece — it bakes
the horizon into every step, which is mechanically why extension-resume was
refused (§5b). ADOPTED ROUTE: WSD/trapezoid (per ng1 + px1 crosswalks,
horizon-FREE plateau + short derived tail) lands as ONE design with the dt1
extension-resume patch — a horizon-extensible trainer wants a horizon-free
schedule. NOT in the live race or the single-variable extension (mechanism
change = separate evidence chain); raced arm optional after the patch.

TERMINAL-CHAIN ADDITION to §5c bank: terminal BIT-DEPTH RE-SOLVE — post-hoc
sensitivity waterfill over the learned per-tensor bit_depth assignments on the
winning endpoint checkpoint (the #157/#69 solve-don't-train family applied to
this vehicle), priced through the real IHS1 pack + archive bytes; no trainer
edit, composes with the micro-edit bank at the parameterized race gate.

## §5e Boundary-protocol EXECUTION record (2026-08-15, MAIN)

The §5 protocol ran end-to-end at the 60-epoch receipt boundary. Receipts:

1. **60-epoch race completed clean.** rc=0, 2,978.03 s (49.6 s/epoch), receipt
   `rx2_wc2_full_mps.done` (counter 8). Endpoint ep60 joint estimate
   **134,323 B** (tokens 114,251 + model 20,072; QAT cut model bytes
   27,026→20,072), bpp 0.0077481.
2. **Endpoint pack custody GREEN** via the comparator's proven `_pack_twice`:
   `deterministic_repeat_exact: true`, raw sha `6547fdb0728e1b18…`,
   20,076 B raw / 15,132 B xz, decode logit diff 0.0
   (`gpu_race/full/endpoint_pack_check/`).
3. **Full-curve refit** (`tools/fit_hpac_descent_law.py`, receipt
   `gpu_race/full/descent_law_fit_full60.json`): continuous phase exp_floor
   asymptote 135,248 B (N*(1 band)=74.4 rel); QAT phase exp_floor asymptote
   **132,798 B** — the ep60 endpoint sits **1,525 B above** its own QAT
   asymptote with τ≈40 ep, N*(1 band)=249 rel. Label: ENTRY_STATE_CONDITIONAL.
4. **N=480 selected from the law** (not a guess): at qat_fraction 0.5,
   N=480 gives 240 continuous epochs (saturated, remaining gain ~0 B) +
   240 QAT epochs (~6τ, remaining gain ~7 B < 2 bands) = full squeeze,
   ~6.6 h at the measured 49.6 s/epoch. N=240 would leave ~137 B (~9.1e-5 S
   in rate) on the table.
5. **First fire REFUSED by the wrapper's sealed envelope** (`full-mps` pins
   epochs=60) — the 4th correct fail-closed refusal of this chain. Cure:
   NEW sealed mode `full-mps-e480` (identical in every other field), landed
   with review-mark via the serializer; the `full-mps` 60-seal untouched.
   Failed-attempt residue retained at `gpu_race/full_e480/` per
   certify-or-block; relaunch used fresh suffixed root `full_e480b`.
6. **N=480 extension FIRED**: pid 13787, launch counter 10, watchers armed at
   launch (liveness+quality), receipt name `rx2_wc2_full_mps_e480b`, root
   `gpu_race/full_e480b/`. ep0 entry point matches the 60-ep run exactly
   (bpp 0.0079963) — same init, same seed, expected. Quality watcher knee
   pinned at epoch 241 (the derived qat_start), bar 186,269 B (the MC36
   frontier archive bytes).

Owed at the e480b endpoint (~6.6 h): pack custody check → refit (does the
QAT law hold at the deeper entry state? the ENTRY_STATE_CONDITIONAL label's
first test) → sealed identity-race parameterization (checkpoint path + terminal
epoch as declared inputs, comparator pin update same commit) → race the best
endpoint → if archive < 186,269 B, T4 fire (projected S ≈ 0.134).

## §5f e480b ENDPOINT + the race's first execution (2026-08-15, MAIN)

**Endpoint (advisory, ADVISORY_ESTIMATE_NOT_SERIALIZED):** ep480 terminal
joint estimate **131,220 B** (tokens 113,229 + model 17,991), bpp 0.0076788,
top1_error 0.0019019 — **−3,103 B vs the 60-ep endpoint** (134,323), and
**below the fitted 60-ep QAT asymptote** (132,798). Wall 23,372.5 s (6.49 h,
on the 6.6 h projection). rc=0, receipt counter 10. Pack custody GREEN on
`qat_stage_end_epoch_0480.pt`: model 17,996 B raw / 13,688 B xz, repeats
byte-identical, decode logit diff 0.0
(`gpu_race/full_e480b/endpoint_pack_check/`).

**Trajectory finding — the schedule-dimension constant-transfer bite.** Under
the re-derived cosine (T_max=480) the continuous phase is NON-MONOTONIC:
144,937 → 135,953 @ep30 (matching the 60-ep run early, where the schedules
still agree) → back UP to 144,527 @ep240 as the mid-schedule LR stays high.
The 240-epoch QAT anneal then does all the work: 144,527 → 131,220. The
fitted "continuous saturates by ep75-120" projection encoded T_max=60's LR
trajectory implicitly; re-deriving the scheduler (the very reason fresh-fire
beat resume) invalidated it. The law's own `constants_run_scoped` /
`ENTRY_STATE_CONDITIONAL` labels covered this; the sizing arithmetic
transferred anyway — logged as a cross-regime transfer in the SCHEDULE
dimension (m21/m22 genus). Direct evidence for the dt1-filed WSD/trapezoid
schedule: hold-then-decay would likely have kept the ep30 gains. Refit
receipt: `gpu_race/full_e480b/descent_law_fit_e480b.json`.

**The sealed race's first execution caught a LATENT BUG (fixed a364573c13).**
`prepare` refused with "MC36 residual-table accounting changed": the check
compared `_hp4_fields`' full member TAIL (96-byte table + token stream,
115,334 B — rx1's own consumer saves it whole as
`residual_and_tokens.compact`) against a bare-96-byte expectation. The chain
had NEVER run (no PREPARE_RESULT.json existed), so the bug was invisible
until now. $0 measurement verified the true contract byte-exact
(`residual_payload == b"RCF1" + tail[:96]` AND `tail[96:] == token_stream`)
and the check was STRENGTHENED to it, not loosened. Comparator pin updated
same-commit both times (declared-inputs parameterization ec72ca343b + this
fix). Attempt-2 race LIVE (counter 12, receipt
`rx2_e480b_identity_race_r2`): preflight ✓ prepare ✓ export-base in
progress.

## §5g e480b identity race COMPLETE + T4 fire (2026-08-15)

The race finished after two more driver defects, both the same genus as §5f's
(the sealed chain's per-variant middle section was driven once): r3 ran `build`
without the variant's `RC64_RESULT.json` (encode is per-variant); r4 ran
`encode` without `EXPORT_RESULT.json` (materialize is per-variant). r5 drove
the full per-variant chain materialize→encode→build for the 4 remaining
fit-selected variants, then cpu-decode + finalize: rc=0 end to end. Driver
committed (`e480b_race_finish.sh`); lesson: read the stage dependency graph
from source after the FIRST missing-artifact refusal, not the second.

**FINAL_RESULT (schema ddm_rx2_final.v1, [macOS-CPU advisory, scorer-free
lossless composition], score_claim=false):**
- WINNER `s1p25_c1p0` @ brotli_q10: archive **183,502 B**
  (sha e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3),
  **−2,767 B vs the MC36 frontier archive** (186,269 B). Repeat build
  byte-identical. Sections: token 112,749 B + model 70,557 B + table 100 B +
  residual 96 B.
- `measured_delta_distortion = 0.0` BY CONSTRUCTION and VERIFIED: decoded
  tokens byte-identical to MC36's; full 3,662,409,600-byte raw RGB decode on
  the lifted F26 CPU runtime (4 threads, 915.5 s) sha-identical to the MC36
  CPU decode (e5539653f598a1c3…). Receiver parseback + component identity
  green. All payloads retained (17,394 files / 13.4 GB inventory).
- Projected S if MC36 distortion holds: **0.16009202615715576**
  (0.1619344578804448 − 25·2,767/37,545,489) — NOT a score claim.
- The e480b HPAC endpoint's advisory joint estimate (131,220 B) vs the
  realized delta (−2,767 B): the estimate replaces a DIFFERENT accounting
  basis than the archive's realized token+model sections; the REAL number is
  the archive stat, per the rate-term law.

**T4 fire (the sealed MAIN-owned fire order, trigger met):** submission dir
`/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/e480b_submission/`
= MC36's `adapted_runtime` tree byte-identical (diff -r clean) + the candidate
archive. Dispatched via the canonical paired launcher; CUDA leg LIVE
`fc-01M02Q4XFXSE2NHFPGWV5NMB8A` (2026-08-15T12:43:02Z, ~$0.25); the CPU
sibling was refused by modal-single-flight AFTER the CUDA spawn — acceptable:
the CPU axis is structurally infeasible on this CUDA-locked runtime (the MC36
promotion measured it, 3.2 s InflationError). Closer pid 57240, receipt
`rx2_e480b_t4_harvest`.

## 5h. T4 fire record r3→v2 — the receiver chain of custody (2026-08-15)

**r3 refused correctly in seconds**: `ResidualArchiveError: invalid F24S model
section`. The candidate's HPAC model section uses the race's new IHS1-pack
serialization; MC36's SHIPPED receiver has no reader for it. Fail-closed
behavior — the archive format and the receiver must ship together.

**Adjudication — ship the race's ADAPTED receiver.** The correct submission
tree already existed at
`retained/cpu_decode/best_rx2/lifted_submission_cpu/` (built by the race's
cpu-decode stage, the SAME tree that produced the raw-identity proof):
- archive.zip = the winner (sha `e3e6f440…`, 183,502 B) — verified in place;
- inflate.py already pins `ARCHIVE_SHA256`/`ARCHIVE_BYTES` for THIS candidate;
- exactly 3 code files differ from MC36's shipped tree: `inflate.py`,
  `runtime/f26_inflate.py` (device-flexible `_configure_device` accepting
  cpu AND cuda), `runtime/residual_archive.py` (the new model-section reader).

**One patch for the T4 axis**: the lifted entrypoint hardcoded
`device_name="cpu"` (its CPU-lift purpose). Changed to
`"cuda" if torch.cuda.is_available() else "cpu"` — no MPS anywhere in the
chain. Rationale: the HPAC token decode is integer/deterministic
(device-independent); after tokens, the neural render on CUDA is the SAME
computation as MC36's frontier row, so the MC36 distortion components carry
and the projected S 0.16009203 rests only on the realized −2,767 B rate delta.
A CPU decode inside the T4 container would instead create a new unmeasured
cell (decode-CPU × score-CUDA; cf. #1054's 21× CPU pose degradation on the
score side).

**v2 dispatch**: clean tree (AppleDouble/`__pycache__` excluded) at
`/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/e480b_submission_v2/`;
r3 lane claim closed terminal
(`failed_shipped_receiver_lacks_new_model_section_terminal`); CUDA call
`fc-01M02QMN3SQ9SNHXZMRWXYEJEW`; closer pid 70027, receipt
`rx2_e480b_t4_harvest_v2`. Modal spend ≈$5.5/$20 after the two short refusals.

Lesson (r1→r3 chain, recorded once): a byte-closed candidate is
archive+receiver JOINTLY — the sha pin caught the archive mismatch (r1), the
format reader caught the receiver mismatch (r3). Both refusals were the
custody system working; the race's own decode tree was always the submission.

## 5i. POINTER MOVED + the composition campaign (2026-08-15)

**THE ROW LANDED.** v2 T4 auth eval (call `fc-01M02QMN3SQ9SNHXZMRWXYEJEW`, 421.6 s, score_claim=True):
**S = 0.1600920261571558 @ 183,502 B [contest-CUDA T4 n600]** — seg 0.029611 + pose
0.0082946 (both IDENTICAL to MC36, as the identity race guaranteed) + rate 0.1221864.
Realized score matches the projection to the last printed digit. Δ vs the prior frontier
0.1619344578804448: **−1.8424e-3** (~526 canonical bands) — the largest single pointer
move of the micro-campaign era. Pointer promoted via posterior backfill (accepted: 1;
custody dir `experiments/results/modal_auth_eval/ddm_rx2_e480b_hpac_winner_v2_…_cuda/`;
one extraction defect cured en route: the remote stored artifacts as Python bytes-repr
strings — recovered via ast.literal_eval, 8 files). effective_frontier now 0.1600920261571558.

**Projection reconciliation (operator "you projected a sub 0.15 score earlier" — answered
with receipts):** the trainer's ep480 estimate was tokens 113,229 + model 17,991 = 131,220 B
→ S ≈ 0.1253 IF that were the archive. Tokens realized almost exactly (112,749). The model
section realized at **70,557 B vs the 17,991 B estimate — a 3.92× serialization gap =
52,566 B = 0.0350 S.** That gap is now THE named lever: sub-0.15 needs only ≥15,153 B of
it recovered losslessly (→ 0.14999); full closure → ~0.1251. Arm ddm_mz1_model_section_rate_race
SPAWNED (charter .omx/research/charters/ddm_mz1_model_section_rate_race_20260815.md).

**The composition campaign (operator "Composing QAT long burn and micro edits and all"):**
three independent axes on the SAME vehicle, composed in dependency order:
1. **QAT long burn (tokens).** The e480b endpoint refit (descent_law_fit_e480b.json)
   REVERSED the law form: power (alpha 0.14, rms 680 B) beats exp-floor at this depth,
   fitted QAT asymptote 118,147 B — the descent has NOT floored. Projected ~1.2 KB/doubling
   of QAT epochs (~−8e-4 S per doubling; the 60-ep exp-floor's "~7 B left" was
   ENTRY_STATE_CONDITIONAL exactly as labeled). Next fire: e960 continuation from the ep480
   checkpoint, overnight Metal (MAIN-fire, resume-flag verification owed first — never-invent-flags).
2. **mz1 model-section recode (rate).** Up to −52,566 B; the identity race's cpu-decode
   stage is the built verifier. LIVE.
3. **Micro-edit bank (distortion).** qs2 −4.375e-6 + re1 −1.207e-6 measured on the
   MC36/cp135 lineage; decoded tokens are byte-identical in the e480b winner so the edits
   transfer semantically but must be RE-COMPILED against the final composed archive's coder.
Order: burn endpoint → identity race → model recode → micro-edit recompile → ONE T4 row.
Modal ≈$5.9/$20 (r1+r3 refusals + v2 421 s).

## 5j. mz1 consumption + the e960 continuation fire (2026-08-15, MAIN)

**mz1 verdict consumed (memo `ddm_mz1_model_section_rate_race_20260815.md`, commit
31711735fa).** The projected "52,566 B model-serialization gap" that fed the earlier
sub-0.15 projection was an ATTRIBUTION ERROR: the 70,557 B model section decomposes
as 13,619 HPAC + 34,763 semantic + 22,161 carrier + 14 wrapper. HPAC ships 4,372 B
BELOW its 17,991 B estimate; 8/8 lossless coder races LOST to the shipped
split-Brotli q10/q11/q11 (best alternative +41 B); exact pack savings = 0 B.
Sub-0.15 does NOT come from a pack fix. Revised route ranking: (a) THIS e960 QAT
token burn (~1.2 KB/doubling ≈ −8e-4 S per the power-law tail), (b) representation
attack on the 56,938 B frozen semantic/carrier sections (mz1 LIVE-HYPOTHESIS,
unowned), (c) js1 pose line (−0.0083 ceiling).

**e960 fire receipt (attempt 5, LIVE).** Trainer pid 47772, launch counter 23,
watchers armed at launch (liveness 47773 / quality 47776, quality bar 131,220 B
from ep481). Resume = `full_e480b/.../qat_stage_end_epoch_0480.pt`, mode
`full-mps-e960` (sealed: eta_min LR hold past ep480 at 6e-5 + EMA parent-geometry
hold 72000→36000 updates → decay 0.9998720867875375 dict-equal to parent).
Attempts 1–4 died to fail-closed gates, each cured at source: env gates ×3
(PYTHONHASHSEED/TAC_ADMISSION_ENFORCE/PYTORCH_ENABLE_MPS_FALLBACK) → resume-identity
drift (cured by the wrapper continuation adapter: torch.load stash + closed
allowlist reconciliation + epochs-only config rule, commit 6aee906d52) → typed
resume_lineage custody (cured by moving wrapper provenance OUT of the trainer's
typed lineage chain into `reports/wrapper_continuation_receipt.json`, schema
`wc2_wrapper_epoch_extension_continuation.v1`, commit c156d7fabc). First forward
row ep482: joint est **131,743 B** — already 1,055 B below the old QAT exp-floor
132,798 (the §5f power-law refit's "not floored" call confirmed at first contact),
top1_error 0.00191, EMA-shadow evaluated. ETA ~6.6 h (478 ep × ~49.6 s/ep).
Endpoint obligations: descent-law refit at ep960 (the ENTRY_STATE_CONDITIONAL
label's second test) → identity race → micro-edit recompile (qs2 −4.375e-6 +
re1 −1.207e-6 vs the FINAL coder) → ONE composed T4 row (#1058).

---

## REBASE NOTE (appended 2026-08-16 by `ddm_fb1`) — APPEND-ONLY, nothing above is changed

**The body above was CORRECT WHEN WRITTEN. This note exists so the bar is not consumed stale.**
Per Catalog #110/#113 HISTORICAL_PROVENANCE no line above is rewritten; this is a superseding row.

At the time of writing, the frontier was `S = 0.1619344578804448 @ 186,269 B` (MC36 Variant C).
**It has since moved twice:** `MC36 -> e480b v2 (183,502 B) -> hv1 ep0634`.

**LIVE BASE as of 2026-08-16:**
`S = 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]`,
sha `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`
(`.omx/state/canonical_frontier_pointer.json`, `effective_frontier`).

**WHY THIS MATTERS — the staleness runs in the dangerous direction.** The `186,269 B` bar sits
**3,510 B ABOVE what we already ship**. A candidate can PASS the bar written above while scoring
**+0.002337165 WORSE** than the incumbent — 233.7x the 1e-5 naming bar.

**USE THIS INSTEAD — a bar that does not go stale.** `seg + pose` is decode-identical across the
whole `cp135 -> MC36 -> e480b v2 -> hv1` lineage (measured to 1e-15), so only rate moves:

```
sub-0.15  <=>  archive <= 168,345.5977 B      (from the live 182,759 B: cut 14,413.4 B)
beat the incumbent  <=>  archive <  182,759 B  (at equal-or-better distortion)
```

Caveat that travels with the invariant: it is a PURE-RATE target, valid only while distortion is
held. Any candidate that CHANGES `d_seg` or `d_pose` must re-measure against the live pointer.

Full derivation, the repo-wide sweep with its denominator, and the bank-union verdict:
`.omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md`.

**SECOND, WC2-SPECIFIC REBASE — two further items in this file are superseded.**

1. **`:510` "if archive < 186,269 B, T4 fire (projected S ~ 0.134)".** Both halves are stale. The
   bar is rebased above. The `~0.134` projection was never reachable: §5f of this same file
   records the realized winner at **183,502 B -> S 0.16009202615715576**, and states that the joint
   estimate "replaces a DIFFERENT accounting basis than the archive's realized token+model
   sections; the REAL number is the archive stat." The body self-corrected; the fire trigger did
   not inherit the correction.
2. **The final endpoint obligation "micro-edit recompile (qs2 -4.375e-6 + re1 -1.207e-6 vs the
   FINAL coder) -> ONE composed T4 row" is REFUSED as budgeted.** Measured by `ddm_fb1`:
   - The two credits do **not** compose onto the hv1 base. `ddm_mc36_dual_axis_t4_verdict_20260814.md`
     follow-on #3 already recorded non-composition onto **MC36**; hv1 is MC36's descendant, and
     hv1's admission rests on exact decoded-token **and** full-raw byte identity, which any qs2
     (189 px) or re1 (4 px) edit breaks by construction. `re1`'s object no longer exists at all —
     hv1 REPLACES the probability object `re1` edited.
   - `qs2`'s pose compensation was Schur-solved at base `d_pose 6.885642960696714e-06` (CP135 at
     186,252 B). hv1's is `6.88e-06`. The gap is worth **3.4009e-06 S = 60.9% of the entire pool**,
     and carrying a stale compensation cost `qs4` **+2.396e-4** (43x the pool).
   - Even granting perfect composition, `qs2 + re1 = -5.5817878492e-06` = **55.8%** of the 1e-5
     naming bar. It cannot clear by waiting.

   The recompile is a **NEW candidate** requiring a fresh in-compile Schur solve and its own
   dual-axis row, not a sum of banked deltas. Do not fire it without a pre-registered projection
   <= -1e-5 on the hv1 base.
