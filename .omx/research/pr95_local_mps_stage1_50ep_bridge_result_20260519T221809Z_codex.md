# PR95 Local MPS Stage 1 50-Epoch Bridge Result - 2026-05-19T22:18:09Z

score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
rank_or_kill_eligible: false
evidence_grade: local_training_portability_probe_advisory
axis: [macOS-CPU advisory]

## Purpose

Run a longer local PR95/HNeRV source-faithful Stage 1 bridge on MPS, then
export the resulting archive and replay it through the local CPU auth-eval
bridge. This is an MPS/Metal portability and training-velocity probe, not a
contest score claim.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python tools/run_pr95_local_training_probe.py \
  --device mps \
  --stage-epochs 1=50 \
  --eval-every 5 \
  --run-codec-stage \
  --run-auth-eval \
  --auth-eval-device cpu \
  --require-auth-eval-comparable \
  --output-dir experiments/results/pr95_local_mps_stage1_50ep_bridge_20260519T215303Z
```

## Result

- Status: `ok=true`
- Total wall time: `1415.833s`
- Training device: `mps`
- MPS fallback: `PYTORCH_ENABLE_MPS_FALLBACK=0`
- Stage: `stage1_v328_ce`
- Epochs: `50`
- Best training bridge epoch: `50`
- Best training bridge score: `1.0372789495308168`
- Best training bridge SegNet distortion: `0.007472254658738772`
- Best training bridge PoseNet distortion: `0.0020068354795997343`
- Best exported member bytes: `222856`

Archive:

- Path: `experiments/results/pr95_local_mps_stage1_50ep_bridge_20260519T215303Z/archive.zip`
- Bytes: `222964`
- SHA-256: `876a942007f03dcf1c7bfaa95a47bcc8c73981aaa949c989636dd5c8d84b4360`
- Member: `0.bin`
- Member bytes: `222856`
- Member SHA-256: `e2837925759eae9fefc32e3f990ebedd3343c043ff51a0aab4eb80607753e9e3`

Local auth-eval bridge:

- Axis: `[macOS-CPU advisory]`
- Canonical score: `1.0373463793792668`
- Reported final: `1.04`
- SegNet distortion: `0.00747224`
- PoseNet distortion: `0.00200675`
- Archive bytes: `222964`
- Score delta vs training bridge: `0.00006742984844998468`
- Comparable under bridge tolerance: `true`
- Auth-eval elapsed: `509.837s`
- Durable auth-eval JSON SHA-256: `e92289f2011669712fc061035d7b59ebd576be6344163c8e550c8e35a88e27ac`
- Manifest SHA-256: `a78f9dea26662e968b5e828f26ae905aade1d26048598e526f9c2a53898a0c12`

## Interpretation

The run establishes that the local MPS training path can complete a longer
source-faithful PR95 Stage 1 bridge and produce a byte-closed archive whose
local CPU auth-eval replay agrees with the training bridge within `6.8e-5`.
That is a real portability and pipeline-integrity signal.

It is not a contest-authoritative score. The auth-eval artifact is explicitly
`[macOS-CPU advisory]`, with `hardware_compliance_blocker:
contest_cpu_requires_linux_x86_64`, `score_claim=false`, and
`promotion_eligible=false`.

## Next Engineering Step

Use this as the local smoke gate before any longer PR95 curriculum run. The
next contest-relevant step is not to promote this score; it is to either:

1. run the full PR95 curriculum locally if the goal is MPS/Metal throughput
   validation, or
2. dispatch the byte-closed archive/runtime family to claimed exact
   `[contest-CPU]` and `[contest-CUDA/T4]` only after a frontier-relevant
   candidate is produced.
