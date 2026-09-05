# Runbook — selective-precision engineering on the CoreML/ANE scorer path (ddm_ane2)

Companion to `docs/runbook_ane_screening_20260905.md` (ane1, which converts the scorers and
proves placement). This runbook covers the ddm_ane2 instrument
`experiments/ddm_ane2_engineer_precision_drift.py`, which asks whether **selective precision** —
some ops fp16 on the ANE, the rest fp32 — moves either scorer's drift onto its bar.

Everything here is `[macOS-CPU/ANE advisory]`. `score_claim=false`, `promotable=false`. The
authority for both scorers is 1-thread CPU-torch fp32; for the contest score it remains
`upstream/evaluate.py`. `upstream/` is READ-ONLY — the scorers are converted from copies held in
memory, never patched.

## Environment

```bash
# coremltools 9.0 + torch 2.12.0 live in the private research venv, NOT in .venv
export PYTHONPATH=src:upstream
PY=.venv_executorch_spike/bin/python
EXP=experiments/ddm_ane2_engineer_precision_drift.py
OUT=/Volumes/VertigoDataTier/pact/ddm_ane2_precision
```

`src/tac/ane_precision.py` imports cleanly from the main `.venv` (it has no `coremltools`
dependency), so `pytest src/tac/tests/test_ane_precision.py` runs anywhere.

## The order the subcommands must run in

Each step consumes the previous step's artifact. The op inventory in particular is the IDENTITY
every split point indexes: every later conversion re-derives the op sequence and refuses if it
drifted, so a rung can never be silently relabelled.

### 1. `enumerate` — the MIL compute-op inventory

```bash
$PY $EXP enumerate --model both --out-dir $OUT/packages --out $OUT/enumerate.json
```

Writes an ordinal → `(name, op_type)` list per scorer (SegNet 297 compute ops, PoseNet 286) plus
op-type counts. Const ops are excluded: a split point that indexed one would move a weight's
dtype without moving any arithmetic.

### 2. `reference` — cache the CPU-torch fp32 authority once

```bash
$PY $EXP reference --model both --raw experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
    --pairs 600 --threads 1 --out-dir $OUT/reference --out $OUT/reference_n600.json
```

Caches SegNet argmax + top-2 margin and PoseNet poses + preprocessed inputs, so every later rung
scores against ONE reference instead of recomputing it. `--raw` accepts either a
`gt_n600.npz` GT cache or a 1200-frame `0.raw` decode of a generated body; the report records
which. Downstream commands take `--eval-pairs N` to subsample the cached reference by a
stratified stride (never a prefix).

### 3. `ladder` — fp16 prefix on the ANE, fp32 tail of k ops

```bash
$PY $EXP ladder --model posenet --enumerate-json $OUT/enumerate.json \
    --reference $OUT/reference/reference_posenet.npz \
    --raw experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
    --splits 0 1 2 4 8 16 32 64 128 192 286 \
    --modes CPU_AND_NE CPU_ONLY --reps 20 --eval-pairs 120 \
    --out-dir $OUT/screen --out $OUT/screen/ladder_posenet_n120.json
```

`k=0` is the all-fp16 model, `k=<compute ops>` the all-fp32 one, so the ladder interpolates
between two endpoints ane1 already measured. Per rung: `MLComputePlan` placement census, latency
per compute-unit mode, and the per-axis fidelity verdict.

### 4. `sensitivity` — one contiguous op group fp16, the rest fp32

```bash
$PY $EXP sensitivity --model posenet --enumerate-json $OUT/enumerate.json \
    --reference $OUT/reference/reference_posenet.npz --raw <frames> \
    --groups 16 --eval-pairs 120 --placement \
    --out-dir $OUT/screen --out $OUT/screen/sensitivity_posenet_n120.json
```

This is the measurement that says WHERE the drift is born. Add `--keep-packages` to retain each
group's `.mlpackage`; without it the package is deleted after measurement and only its sha256 and
the per-pair fidelity payload are retained (the package is deterministically rebuildable from the
op set, the weights sha and the coremltools version).

### 5. `selective` — an arbitrary named fp32 set, everything else fp16

```bash
$PY $EXP selective --model posenet --enumerate-json $OUT/enumerate.json \
    --reference $OUT/reference/reference_posenet.npz --raw <frames> \
    --fp32-set hd1=0:1 hd2=0:2 hd4=0:4 hd8=0:8 hd16=0:16 hd32=0:32 \
    --modes CPU_AND_NE --reps 20 --eval-pairs 120 \
    --out-dir $OUT/stage2 --out $OUT/stage2/mirror_posenet_n120.json
```

`label=ordinals` where ordinals are comma-separated indices or `lo:hi` ranges. This is how the
fp32-HEAD mirror of the tail ladder is run: no new code path, just a different set.

### 6. `units` — the same package under different `ComputeUnit` requests

```bash
$PY $EXP units --model posenet --package <pkg.mlpackage> \
    --reference $OUT/reference/reference_posenet.npz --raw <frames> \
    --modes CPU_ONLY CPU_AND_NE ALL CPU_AND_GPU --eval-pairs 120 \
    --out $OUT/stage2/units_posenet_fp32.json
```

The control that separates "the input changed" from "the compute-unit REQUEST changed the
numerics". Run it before trusting any drift number measured under `ALL`.

### 7. `castcount` — the boundary cost of a split

```bash
$PY $EXP castcount --packages $OUT/screen/*.mlpackage --out $OUT/stage2/castcount.json
```

Counts `cast` ops in each saved MIL program. A residual network's skip connections make a
topological prefix cut many edges, so a split's cast count is not 1.

### 8. `hybrid` — the realized exact-argmax hybrid

```bash
$PY $EXP hybrid --fp16-package <k0.mlpackage> --fp32-dense-package <k297.mlpackage> \
    --reference $OUT/reference_generated/reference_segnet.npz \
    --reference-report $OUT/reference_generated_n120.json \
    --raw <0.raw> --band 0.4456 --tile 64 --halo 32 --eval-pairs 40 --dense-reps 10 \
    --out $OUT/stage5/hybrid_band0.4456_t64_h32.json
```

fp16 ANE dense pass, band selected from the **fp16** margin (the quantity an inference-time
decoder actually has), fp32 recompute on FIXED-size crops through a real CoreML model converted
for the crop shape. Both denominators — 1-thread CPU-torch and dense `coreml_cpu_fp32` — are
timed inside the same run, because the reference report's per-pair wall clock includes frame
decode and dividing by it would inflate every speedup.

The report always carries `crop_vs_fullframe_argmax_disagreements`: a U-Net crop is not a window
on the full-frame result (EfficientNet-B2's 23 squeeze-excitation blocks pool globally), so a
nonzero count there is the crop realization failing exactness, not fp32 drift.

## Retained runbooks

The staged shell drivers live beside the reports and are the reproducible form of the above:

| script | what it runs |
|---|---|
| `$OUT/run_screen.sh` | both ladders + both sensitivity profiles, n120 |
| `$OUT/run_stage2.sh` | fp32-HEAD mirror ladders, compute-unit controls, cast census |
| `$OUT/run_stage3.sh` | SegNet drift on a GENERATED decode (the input control) |
| `$OUT/run_stage4.sh` | pr1's 39-point selector replay on `coreml_cpu_fp32` |
| `$OUT/run_stage5.sh` | the realized crop hybrid at two band widths |
| `$OUT/render_tables.py` | regenerates the memo's tables from the retained JSON |

Each was launched through `tools/launch_detached_process.py` with a distinct `--done-receipt`;
the launch manifests and `run.log` files are retained under `$OUT/run_*/`.

## Reading the results

The memo is `.omx/research/ddm_ane2_engineer_the_precision_drift_20260905.md`. The one-line
summary of what the instrument is for: **a split point is not a knob you tune, it is a hypothesis
about WHERE the error is born** — and the `sensitivity` subcommand is the only one of these that
tests that hypothesis directly.
