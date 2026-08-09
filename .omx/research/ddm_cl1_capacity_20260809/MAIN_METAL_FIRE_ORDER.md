# DDM CL1 MAIN Metal fire order

Status: **QUEUED-WITH-A-FIRE-ORDER; DO NOT FIRE FROM A SANDBOXED TOOL
SESSION**. `score_claim=false`. Training and codec measurements are
`[macOS-MPS research-signal]`; packing is `[macOS-CPU advisory]`.

The n600 ANS competitor is terminal: result SHA-256
`8816f91afcc21060753a6612cda4e1b7f3b483a7aa073cbfa1b9b5d7e520d451`,
done-receipt SHA-256
`f099a42cb2990e06b0f4614b17f1ce737ce6e8a094ff02f41d7e5ffb4d97e5af`,
and former PID 89557 now returns ESRCH. That clears the prior Metal occupancy
refusal; it does not prove this trainer's resume path.

This fire order uses one fixed estimand per lambda: the immutable epoch-60 QAT
stage checkpoint. It does not search epochs or claim the terminal checkpoint is
the within-run byte optimum. An earlier checkpoint could have a smaller real
joint; that is outside this smallest controlled lambda ladder.

## Gate 0 — hard admission, live-process check, and claim

Run from an unsandboxed MAIN/operator shell:

```bash
set -euo pipefail
cd /Users/adpena/Projects/pact
export TAC_ADMISSION_ENFORCE=1
export PYTHONHASHSEED=0
export PYTORCH_ENABLE_MPS_FALLBACK=0
.venv/bin/python tools/spawn_durable_daemon.py --reconcile
.venv/bin/python tools/spawn_durable_daemon.py --status
.venv/bin/python tools/system_memory_governor.py --ceiling
.venv/bin/python tools/system_memory_governor.py --admit --projected-gib 12
.venv/bin/python tools/claim_lane_dispatch.py summary --format json
DDM_CL1_LIVE_MATCHES="$(ps -axo pid=,pgid=,etime=,command= | rg -i '[m]lx|[m]ps|[m]etal|[t]rain_|[c]odec_|[a]ns_real' || true)"
if [[ -n "$DDM_CL1_LIVE_MATCHES" ]]; then
  print -u2 -- "$DDM_CL1_LIVE_MATCHES"
  return 1 2>/dev/null || exit 1
fi
df -h /Volumes/VertigoDataTier
```

Before this arm claims, refuse if any live Metal/MLX job or any live
`local_metal` claim remains. After the claim below, every repeated check must
find exactly this arm's own live claim—lane
`lane_ddm_cl1_hpac_capacity_20260809`, job
`ddm_cl1_capacity_ladder_20260809`, agent `MAIN`—and no competing
`local_metal` claim. The own claim is required, not a refusal. Also refuse if
the governor denies a 12-GiB projected peak, SSD free space would fall below
30 GB, or the committed hashes/tests named in `BLOCKED_RECEIPT.md` no longer
match. `TAC_ADMISSION_ENFORCE=1` is mandatory: advisory admission is not a fire
gate. `PYTORCH_ENABLE_MPS_FALLBACK=0` is mandatory: CPU fallback is not this
axis.

Ledger emptiness alone is not process absence. `claim_lane_dispatch.py`
prevents same-lane duplicates but does not globally serialize `local_metal`,
and `--ttl-hours` is conflict-evaluation input rather than a persisted lease.
Therefore repeat the governor admission, claim summary, and live `ps` check in
the same observation immediately before every MPS training/encode/decode
launch. Any ambiguity is a refusal; there is no TOCTOU waiver.

Then claim the registered lane exactly once:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_ddm_cl1_hpac_capacity_20260809 \
  --platform local_metal \
  --instance-job-id ddm_cl1_capacity_ladder_20260809 \
  --agent MAIN \
  --status training \
  --ttl-hours 12 \
  --notes 'DDM CL1 fixed-topology terminal-QAT lambda ladder; manual global one-Metal recheck before every fire; scorer-free; score_claim=false'
```

## Gate 1 — literal SIGKILL and fresh-root resume

The initial 12-GiB RSS ceiling and 7,200-second wall cap are **DERIVED**, not
measured. Twelve GiB is a conservative envelope over the two resident n600
int64 fields, activations, optimizer state, and MPS allocator overhead. The
wall cap is nearly 3x the prior same-shape 2,431.933-second measurement. The
interrupted run must replace the RSS projection with an observed receipt.

Require a fresh interrupted root, then launch the exact receiver-closed command:

```bash
test ! -e /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted
mkdir -p /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/run
env TAC_ADMISSION_ENFORCE=1 PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  SAFE_RUN_STATUS_RECEIPT=/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/run/interrupted.safe_run.json \
  .venv/bin/python tools/spawn_durable_daemon.py \
  --log /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/run/interrupted.log \
  --label ddm_cl1_lambda1_interrupt \
  --projected-gb 12 --min-free-gb 30 \
  --rss-cap-mb 12288 --walltime-cap-s 7200 \
  --projected-peak-gib 12 --priority 50 --verify-s 3 --job-class training \
  -- \
  .venv/bin/python tools/train_ddm_cl1_hpac_capacity.py \
  --cache /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt \
  --init /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/gt/hpac_p64_exact_from_archive.pt \
  --epochs 60 --batch-size 8 --eval-batch-size 4 --eval-every 2 \
  --lr 0.003 --lr-exponent 0.0002 --lr-bits 0.01 --bit-eps 1e-6 \
  --rate-lambda 1.0 --qat-fraction 0.5 --init-bits 8.0 \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --target-mode raw \
  --seed 20260716 --ema-target-seed-fraction 0.01 --device mps \
  --save /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/checkpoints/best_ema.pt \
  --out /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/reports/trainer.json
```

Wait until
`interrupted/checkpoints/best_ema.checkpoints/periodic/epoch_0001.pt` is
durable, its embedded `causal_state_sha256` verifies, and the log shows work
beyond publication. Before signaling, require a current receipt with
`status=running`, `exit=null`, and
`pidfile integer == child_pid == pgid`; require live `ps` argv to match the
trainer command above. Never select a PID by pattern alone.

```bash
DDM_CL1_INNER_PID="$(tr -d '[:space:]' < /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/run/interrupted.safe_run.json.child.pid)"
.venv/bin/python - "$DDM_CL1_INNER_PID" <<'PY'
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

receipt_path = Path('/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/run/interrupted.safe_run.json')
receipt = json.loads(receipt_path.read_text())
pid = int(sys.argv[1])
assert receipt['status'] == 'running' and receipt['exit'] is None
assert pid == receipt['child_pid'] == receipt['pgid']
tool_path = Path('/Users/adpena/Projects/pact/tools/fit_ddm_cl1_hpac_capacity.py')
spec = importlib.util.spec_from_file_location('ddm_cl1_fit_kill_gate', tool_path)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
expected = tool._expected_training_argv(
    rate_lambda=1.0,
    save=Path('/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/checkpoints/best_ema.pt'),
    out=Path('/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/reports/trainer.json'),
)
assert tool._normalize_argv(receipt['argv']) == expected
observed = subprocess.run(
    ['ps', '-ww', '-p', str(pid), '-o', 'command='],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert observed == ' '.join(receipt['argv'])
print(pid)
PY
kill -KILL -- "-$DDM_CL1_INNER_PID"
```

Do not SIGKILL the outer durable-daemon/safe-run group: SIGKILL bypasses
safe-run's cascade handler and can orphan the inner trainer. Reconcile only
after safe-run records `status="ok"`, `exit=-9`,
`peak_rss_observed=true`, and `peak_rss_mib>0`. Verify the exact trainer group
is gone before continuing.

Derive the continuation cap exactly from that receipt:

```bash
DDM_CL1_CAP_MIB="$(.venv/bin/python - <<'PY'
import json
import math
from pathlib import Path

p = Path('/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/run/interrupted.safe_run.json')
r = json.loads(p.read_text())
assert r['status'] == 'ok' and r['exit'] == -9
assert r['peak_rss_observed'] is True and r['peak_rss_mib'] > 0
print(math.ceil(1.5 * r['peak_rss_mib'] / 256) * 256)
PY
)"
DDM_CL1_CAP_GIB="$(.venv/bin/python - "$DDM_CL1_CAP_MIB" <<'PY'
import math
import sys
print(math.ceil(int(sys.argv[1]) / 1024))
PY
)"
```

Resume into a fresh output tree. Reusing the interrupted tree is forbidden
because work beyond epoch 1 may already have published immutable later
checkpoints.

```bash
test ! -e /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed
mkdir -p /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/run
env TAC_ADMISSION_ENFORCE=1 PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  SAFE_RUN_STATUS_RECEIPT=/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/run/resumed.safe_run.json \
  .venv/bin/python tools/spawn_durable_daemon.py \
  --log /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/run/resumed.log \
  --label ddm_cl1_lambda1_resume \
  --projected-gb "$DDM_CL1_CAP_GIB" --min-free-gb 30 \
  --rss-cap-mb "$DDM_CL1_CAP_MIB" --walltime-cap-s 7200 \
  --projected-peak-gib "$DDM_CL1_CAP_GIB" --priority 50 --verify-s 3 --job-class training \
  -- \
  .venv/bin/python tools/train_ddm_cl1_hpac_capacity.py \
  --cache /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt \
  --init /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/gt/hpac_p64_exact_from_archive.pt \
  --epochs 60 --batch-size 8 --eval-batch-size 4 --eval-every 2 \
  --lr 0.003 --lr-exponent 0.0002 --lr-bits 0.01 --bit-eps 1e-6 \
  --rate-lambda 1.0 --qat-fraction 0.5 --init-bits 8.0 \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --target-mode raw \
  --seed 20260716 --ema-target-seed-fraction 0.01 --device mps \
  --save /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/checkpoints/best_ema.pt \
  --out /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/reports/trainer.json \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/checkpoints/best_ema.checkpoints/periodic/epoch_0001.pt
```

## Gate 2 — uninterrupted twin and one-factor equality

After the resumed run completes, repeat Gate 0's one-Metal checks. Set the
following exact tuple for the uninterrupted twin, then execute the canonical
fresh-run block:

```bash
DDM_CL1_FRESH_ROOT=/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_uninterrupted_twin/training
DDM_CL1_FRESH_LAMBDA=1.0
DDM_CL1_FRESH_LABEL=ddm_cl1_lambda1_uninterrupted
```

```bash
test ! -e "$DDM_CL1_FRESH_ROOT"
mkdir -p "$DDM_CL1_FRESH_ROOT/run"
env TAC_ADMISSION_ENFORCE=1 PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  SAFE_RUN_STATUS_RECEIPT="$DDM_CL1_FRESH_ROOT/run/training.safe_run.json" \
  .venv/bin/python tools/spawn_durable_daemon.py \
  --log "$DDM_CL1_FRESH_ROOT/run/training.log" \
  --label "$DDM_CL1_FRESH_LABEL" \
  --projected-gb "$DDM_CL1_CAP_GIB" --min-free-gb 30 \
  --rss-cap-mb "$DDM_CL1_CAP_MIB" --walltime-cap-s 7200 \
  --projected-peak-gib "$DDM_CL1_CAP_GIB" --priority 50 --verify-s 3 --job-class training \
  -- \
  .venv/bin/python tools/train_ddm_cl1_hpac_capacity.py \
  --cache /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt \
  --init /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/gt/hpac_p64_exact_from_archive.pt \
  --epochs 60 --batch-size 8 --eval-batch-size 4 --eval-every 2 \
  --lr 0.003 --lr-exponent 0.0002 --lr-bits 0.01 --bit-eps 1e-6 \
  --rate-lambda "$DDM_CL1_FRESH_LAMBDA" --qat-fraction 0.5 --init-bits 8.0 \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --target-mode raw \
  --seed 20260716 --ema-target-seed-fraction 0.01 --device mps \
  --save "$DDM_CL1_FRESH_ROOT/checkpoints/best_ema.pt" \
  --out "$DDM_CL1_FRESH_ROOT/reports/trainer.json"
```

The fitter requires this exact expanded argv, a successful trainer
result/manifest, and the terminal checkpoint:

`training/checkpoints/best_ema.checkpoints/qat_stage_end_epoch_0060.pt`.

The two terminal checkpoints must have equal embedded
`causal_state_sha256`. Their packed EMA model, Range stream, decoded raw token
tensor, and encode/decode logit hashes must be equal. Any difference blocks the
ladder; it is not treated as an error bar.

## Gate 3 — attested real pack, encode, and decode

For each terminal checkpoint, define `DDM_CL1_RUNG_ROOT` as that rung's exact
absolute SSD root and `DDM_CL1_TERMINAL` as its exact epoch-60 QAT checkpoint.
For the resumed control, for example:

```bash
DDM_CL1_RUNG_ROOT=/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed
DDM_CL1_TERMINAL=/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/checkpoints/best_ema.checkpoints/qat_stage_end_epoch_0060.pt
mkdir -p "$DDM_CL1_RUNG_ROOT/serialized" "$DDM_CL1_RUNG_ROOT/run"
```

Pack through the owned attesting runner. Its safe-run receipt proves the runner
argv; its immutable attestation proves the exact external child argv, runner
bytes, all imported HPAC source bytes, runtime identity, and every
input/output/report SHA-256 and byte count.

```bash
env TAC_ADMISSION_ENFORCE=1 PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  SAFE_RUN_STATUS_RECEIPT="$DDM_CL1_RUNG_ROOT/run/terminal.pack.safe_run.json" \
  .venv/bin/python tools/safe_run.py \
  --rss-mb 4096 --timeout 600 --projected-gib 4 --label ddm_cl1_terminal_pack -- \
  .venv/bin/python tools/fit_ddm_cl1_hpac_capacity.py pack \
  --checkpoint "$DDM_CL1_TERMINAL" \
  --blob "$DDM_CL1_RUNG_ROOT/serialized/terminal.model.bin.xz" \
  --report "$DDM_CL1_RUNG_ROOT/serialized/terminal.pack.json" \
  --attestation "$DDM_CL1_RUNG_ROOT/serialized/terminal.pack.attestation.json"
```

Repeat the one-Metal checks, then encode through a distinct governed launch:

```bash
env TAC_ADMISSION_ENFORCE=1 PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  SAFE_RUN_STATUS_RECEIPT="$DDM_CL1_RUNG_ROOT/run/terminal.encode.safe_run.json" \
  .venv/bin/python tools/spawn_durable_daemon.py \
  --log "$DDM_CL1_RUNG_ROOT/run/terminal.encode.log" \
  --label ddm_cl1_terminal_encode \
  --projected-gb 12 --min-free-gb 30 \
  --rss-cap-mb 12288 --walltime-cap-s 1800 \
  --projected-peak-gib 12 --priority 50 --verify-s 3 --job-class training \
  -- \
  .venv/bin/python tools/fit_ddm_cl1_hpac_capacity.py encode \
  --checkpoint "$DDM_CL1_TERMINAL" \
  --tokens "$DDM_CL1_RUNG_ROOT/serialized/terminal.range.bin" \
  --report "$DDM_CL1_RUNG_ROOT/serialized/terminal.encode.json" \
  --attestation "$DDM_CL1_RUNG_ROOT/serialized/terminal.encode.attestation.json"
```

Wait for `status="ok", exit=0`, reconcile, and verify the encoder process group
is gone. Repeat the one-Metal checks, then decode the same bytes:

```bash
env TAC_ADMISSION_ENFORCE=1 PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  SAFE_RUN_STATUS_RECEIPT="$DDM_CL1_RUNG_ROOT/run/terminal.decode.safe_run.json" \
  .venv/bin/python tools/spawn_durable_daemon.py \
  --log "$DDM_CL1_RUNG_ROOT/run/terminal.decode.log" \
  --label ddm_cl1_terminal_decode \
  --projected-gb 12 --min-free-gb 30 \
  --rss-cap-mb 12288 --walltime-cap-s 1800 \
  --projected-peak-gib 12 --priority 50 --verify-s 3 --job-class training \
  -- \
  .venv/bin/python tools/fit_ddm_cl1_hpac_capacity.py decode \
  --checkpoint "$DDM_CL1_TERMINAL" \
  --tokens "$DDM_CL1_RUNG_ROOT/serialized/terminal.range.bin" \
  --raw "$DDM_CL1_RUNG_ROOT/serialized/terminal.raw.u8" \
  --report "$DDM_CL1_RUNG_ROOT/serialized/terminal.decode.json" \
  --attestation "$DDM_CL1_RUNG_ROOT/serialized/terminal.decode.attestation.json"
```

Admit a row only when the attesting runner and fitter accept it. In particular,
the pack must be logit-bit-exact, the n600 decode must be exact, the raw token
SHA-256 must be
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`,
and decode wall time must not exceed 1,800 seconds.

## Gate 4 — conditional lambdas and exact fitter input

- Launch `lambda_0p5` by rerunning Gate 2's canonical fresh-run block after
  setting exactly:

  ```bash
  DDM_CL1_FRESH_ROOT=/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_0p5/training
  DDM_CL1_FRESH_LAMBDA=0.5
  DDM_CL1_FRESH_LABEL=ddm_cl1_lambda0p5
  ```

  Use its one successful training receipt and the same terminal
  pack/encode/decode sequence.
- If the exact terminal-QAT Range secant is at least `-1`, or model bytes do not
  grow monotonically, stop. Lambda 0.25 is forbidden.
- Only if the first secant is strictly below `-1`, rerun Gate 2's block after
  setting exactly:

  ```bash
  DDM_CL1_FRESH_ROOT=/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_0p25/training
  DDM_CL1_FRESH_LAMBDA=0.25
  DDM_CL1_FRESH_LABEL=ddm_cl1_lambda0p25
  ```
- An endpoint is not a continuous optimum. The fitter reports only adjacent
  secants and a conditional three-point descriptive OLS interval.

The measurement JSON must have this top-level schema. Each row uses exact
absolute paths; `training_receipt_paths` has two ordered entries only for the
resume control (interrupted, resumed) and one for every fresh run:

```json
{
  "schema": "ddm_cl1_capacity_measurement_set.v1",
  "rows": [
    {
      "rung_id": "lambda_1p0_resume_control",
      "training_receipt_paths": [
        "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/interrupted/run/interrupted.safe_run.json",
        "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/run/resumed.safe_run.json"
      ],
      "selected_checkpoint_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/checkpoints/best_ema.checkpoints/qat_stage_end_epoch_0060.pt",
      "packed_model_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.model.bin.xz",
      "range_token_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.range.bin",
      "decoded_raw_token_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.raw.u8",
      "pack_report_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.pack.json",
      "encode_report_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.encode.json",
      "decode_report_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.decode.json",
      "pack_attestation_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.pack.attestation.json",
      "encode_attestation_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.encode.attestation.json",
      "decode_attestation_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/serialized/terminal.decode.attestation.json",
      "pack_receipt_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/run/terminal.pack.safe_run.json",
      "encode_receipt_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/run/terminal.encode.safe_run.json",
      "decode_receipt_path": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/resumed/run/terminal.decode.safe_run.json"
    }
  ]
}
```

Add the analogous absolute twin and lambda-0.5 rows; add lambda-0.25 only after
the fitter's first secant passes. Run the fitter exactly as follows:

```bash
.venv/bin/python tools/fit_ddm_cl1_hpac_capacity.py fit \
  --measurements /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/measurements.json \
  --out-json /Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/ddm_cl1_capacity_fit.json \
  --out-md /Users/adpena/Projects/pact/.omx/research/ddm_cl1_capacity_20260809/LADDER_RESULT.md
```

The fitter opens every byte, recomputes checkpoint causal state, proves the
literal interruption/resume lineage and fresh twin, requires one full run
identity modulo only `rate_lambda`, verifies terminal epoch-60 QAT selection,
and binds pack/codec outputs to owned-runner attestations and safe-run receipts.

## Gate 5 — close the claim on every terminal outcome

After all processes are terminal and receipts are preserved, append exactly one
new same-job terminal row. On successful harvest:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_ddm_cl1_hpac_capacity_20260809 \
  --platform local_metal \
  --instance-job-id ddm_cl1_capacity_ladder_20260809 \
  --agent MAIN \
  --status completed_harvested \
  --notes 'DDM CL1 terminal-QAT lambda ladder harvested; see SSD receipts and LADDER_RESULT.md'
```

If a gate fails after all launched processes are stopped, use the same command
with `--status stopped_blocked` and put the exact blocker and receipt path in
`--notes`. Never leave a dead job as a nonterminal claim.
