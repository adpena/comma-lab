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
