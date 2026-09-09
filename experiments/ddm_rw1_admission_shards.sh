#!/usr/bin/env bash
# ddm_rw1 admission chain: render -> seg -> pose, sharded STRIDED (never a prefix).
#
# Configured by environment, not by flags, so the launcher records the whole
# configuration in one manifest:
#
#   CKPT     the trainer checkpoint whose EMA shadow is exported (required)
#   STAGE    render | seg | pose            (required)
#   ROOT     receipts + argmax destination  (default WORK/admission)
#   RAWDIR   candidate decode destination   (default BULK/renders/candidate)
#   SHARDS   shard count                    (default 4 -- the charter's CPU cap)
#   THREADS  torch threads per shard        (default 3)
#
# Every stage refuses partial coverage downstream: the seg merge requires all 600
# pairs and the admission requires all 600 pose rows.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
ARM="${REPO}/experiments/ddm_rw1_renderer_edge_foldback.py"

WORK="/Volumes/VertigoDataTier/pact/ddm_rw1_renderer_edge_foldback"
BULK="/Volumes/APDataStore/pact/ddm_rw1_renderer_edge_foldback"
ROOT="${ROOT:-${WORK}/admission}"
RAWDIR="${RAWDIR:-${BULK}/renders/candidate}"
SHARDS="${SHARDS:-4}"
THREADS="${THREADS:-3}"
STAGE="${STAGE:?set STAGE=render|seg|pose}"

mkdir -p "${ROOT}" "${RAWDIR}"

case "${STAGE}" in
  render)
    : "${CKPT:?set CKPT=<trainer checkpoint>}"
    # Seed the decode ONCE before any shard opens it read-write.
    "${PY}" "${ARM}" render --seed-only \
        --out-dir "${RAWDIR}" --out "${ROOT}/RENDER_SEED.json" --threads 1
    pids=()
    for ((i = 0; i < SHARDS; i++)); do
      "${PY}" "${ARM}" render \
          --checkpoint "${CKPT}" --weights shadow \
          --out-dir "${RAWDIR}" \
          --out "${ROOT}/render_shard_${i}.json" \
          --shard-index "${i}" --shard-count "${SHARDS}" \
          --threads "${THREADS}" --progress \
          > "${ROOT}/render_shard_${i}.log" 2>&1 &
      pids+=("$!")
    done
    ;;
  seg)
    pids=()
    for ((i = 0; i < SHARDS; i++)); do
      "${PY}" "${ARM}" seg \
          --raw "${RAWDIR}/0.raw" \
          --out "${ROOT}/seg_shard_${i}.json" \
          --out-argmax "${ROOT}/seg_argmax_shard_${i}.npy" \
          --shard-index "${i}" --shard-count "${SHARDS}" \
          --threads "${THREADS}" --progress \
          > "${ROOT}/seg_shard_${i}.log" 2>&1 &
      pids+=("$!")
    done
    ;;
  pose)
    pids=()
    for ((i = 0; i < SHARDS; i++)); do
      "${PY}" "${ARM}" pose \
          --raw "${RAWDIR}/0.raw" \
          --out "${ROOT}/pose_shard_${i}.json" \
          --out-rows "${ROOT}/pose_rows_${i}.jsonl" \
          --shard-index "${i}" --shard-count "${SHARDS}" \
          --threads "${THREADS}" --progress \
          > "${ROOT}/pose_shard_${i}.log" 2>&1 &
      pids+=("$!")
    done
    ;;
  *)
    echo "unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "ddm_rw1 ${STAGE}: at least one shard failed" >&2
  exit "${status}"
fi

if [[ "${STAGE}" == "seg" ]]; then
  "${PY}" "${ARM}" seg-merge \
      --shards "${ROOT}"/seg_shard_*.json \
      --out "${ROOT}/SEG.json"
fi
echo "ddm_rw1 ${STAGE}: all ${SHARDS} shards ok"
