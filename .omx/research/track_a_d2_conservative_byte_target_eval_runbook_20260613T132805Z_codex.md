# Track-A D2 conservative byte-target exact-eval runbook

This runbook is for Track-A Item B / D2 only: conservative byte-target
waterfill on the base_ch20 torch-vehicle adapter. It intentionally uses
`--variable-level-waterfill-byte-target 2731.0` and does not expose or route the
falsified `net_stop` path.

## D2 driver flags

Use these flags on the Track-A export/training command that should emit the D2
archive:

```bash
RD_TABLE=.omx/research/track_a_itemB_waterfill_rd_table_rd24_20260612.json

--variable-level-waterfill-enabled \
--variable-level-waterfill-rd-table "$RD_TABLE" \
--variable-level-waterfill-byte-target 2731.0
```

The flag is default-off. If omitted, the driver must remain byte-identical to
the vendored base_ch20 archive path.

## Exact CPU/CUDA eval

Set `D2_EXPORT_OUT_DIR` to the D2-enabled run directory that contains
`best/best_archive.bin`. Do not point these commands at a non-D2 archive.

```bash
set -euo pipefail

RUN_ID="${RUN_ID:-track_a_d2_2731_$(date -u +%Y%m%dT%H%M%SZ)}"
D2_EXPORT_OUT_DIR="${D2_EXPORT_OUT_DIR:?set to the D2-enabled Track-A run dir}"
ARCHIVE_BIN="$D2_EXPORT_OUT_DIR/best/best_archive.bin"
ARCHIVE_DIR="$D2_EXPORT_OUT_DIR/d2_submission_${RUN_ID}"
ARCHIVE_ZIP="$ARCHIVE_DIR/archive.zip"
ARCHIVE_MANIFEST="$ARCHIVE_DIR/archive_manifest.json"
RUNTIME_DIR=experiments/public_runtime_adapters/torch_vehicle_d2_hnerv_adapter
LANE_ID=lane_track_a_d2_conservative_byte_target_driver_20260613
CPU_JOB="${RUN_ID}_cpu"
CUDA_JOB="${RUN_ID}_cuda"
PAIR_GROUP="${RUN_ID}_dual_axis"
CPU_OUT="experiments/results/modal_auth_eval_cpu/${CPU_JOB}"
CUDA_OUT="experiments/results/modal_auth_eval/${CUDA_JOB}"

test -f "$ARCHIVE_BIN"
uv run python tools/build_torch_vehicle_d2_archive_zip.py \
  --input-bin "$ARCHIVE_BIN" \
  --output-zip "$ARCHIVE_ZIP" \
  --manifest-json "$ARCHIVE_MANIFEST"
ARCHIVE_SHA="$(shasum -a 256 "$ARCHIVE_ZIP" | awk '{print $1}')"

uv run python tools/claim_lane_dispatch.py claim \
  --lane-id "$LANE_ID" \
  --platform modal \
  --instance-job-id "$CPU_JOB" \
  --agent codex:gpt-5 \
  --status eval_cpu \
  --notes "D2 conservative byte_target=2731 exact CPU archive_sha=${ARCHIVE_SHA}"

PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval_cpu.py \
  --archive "$ARCHIVE_ZIP" \
  --expected-archive-sha256 "$ARCHIVE_SHA" \
  --submission-dir "$RUNTIME_DIR" \
  --inflate-sh inflate.sh \
  --expected-runtime-tree-sha256 auto \
  --output-dir "$CPU_OUT" \
  --detach \
  --provider-detach-ack \
  --lane-id "$LANE_ID" \
  --instance-job-id "$CPU_JOB" \
  --claim-policy require_active \
  --pair-group-id "$PAIR_GROUP" \
  --claim-notes "D2 conservative byte_target=2731 exact CPU archive_sha=${ARCHIVE_SHA}"

uv run python tools/claim_lane_dispatch.py claim \
  --lane-id "$LANE_ID" \
  --platform modal \
  --instance-job-id "$CUDA_JOB" \
  --agent codex:gpt-5 \
  --status eval_cuda \
  --allow-parallel \
  --child-of "$CPU_JOB" \
  --parallel-reason "paired dual-axis exact eval for same D2 archive" \
  --notes "D2 conservative byte_target=2731 exact CUDA archive_sha=${ARCHIVE_SHA}"

PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval.py \
  --archive "$ARCHIVE_ZIP" \
  --expected-archive-sha256 "$ARCHIVE_SHA" \
  --submission-dir "$RUNTIME_DIR" \
  --inflate-sh inflate.sh \
  --expected-runtime-tree-sha256 auto \
  --output-dir "$CUDA_OUT" \
  --gpu T4 \
  --scorer-device cuda \
  --inflate-device auto \
  --detach \
  --provider-detach-ack \
  --lane-id "$LANE_ID" \
  --instance-job-id "$CUDA_JOB" \
  --claim-policy require_active \
  --pair-group-id "$PAIR_GROUP" \
  --claim-notes "D2 conservative byte_target=2731 exact CUDA archive_sha=${ARCHIVE_SHA}"

uv run python tools/recover_modal_auth_eval.py --output-dir "$CPU_OUT"
uv run python tools/recover_modal_auth_eval.py --output-dir "$CUDA_OUT"
```

Promotion remains blocked until both recovered artifacts carry the exact
`[contest-CPU]` and `[contest-CUDA]` results for this archive SHA.
