#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MANIFEST="${ROOT}/FX4_GT_PROVENANCE_MANIFEST.json"

if [ "$#" -ne 2 ]; then
  printf '%s\n' 'usage: fx4_gt_provenance_guard.sh LEFT_LEG RIGHT_LEG' >&2
  exit 2
fi

left_leg=$1
right_leg=$2
left_axis=$(jq -er --arg leg "$left_leg" '.retained_boundary_replay[$leg].target_axis_id' "$MANIFEST") || {
  printf '%s\n' "unknown retained-replay leg: ${left_leg}" >&2
  exit 2
}
right_axis=$(jq -er --arg leg "$right_leg" '.retained_boundary_replay[$leg].target_axis_id' "$MANIFEST") || {
  printf '%s\n' "unknown retained-replay leg: ${right_leg}" >&2
  exit 2
}

if [ "$left_axis" != "$right_axis" ]; then
  jq -nc \
    --arg status REFUSE \
    --arg reason target_cache_lineage_mismatch \
    --arg left_leg "$left_leg" \
    --arg left_axis "$left_axis" \
    --arg right_leg "$right_leg" \
    --arg right_axis "$right_axis" \
    '{status:$status, reason:$reason, left_leg:$left_leg, left_target_axis_id:$left_axis, right_leg:$right_leg, right_target_axis_id:$right_axis, score_claim:false}'
  exit 42
fi

jq -nc \
  --arg status PASS \
  --arg left_leg "$left_leg" \
  --arg right_leg "$right_leg" \
  --arg axis "$left_axis" \
  '{status:$status, left_leg:$left_leg, right_leg:$right_leg, target_axis_id:$axis, score_claim:false}'
