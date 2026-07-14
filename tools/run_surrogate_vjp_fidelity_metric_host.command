#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT=${0:A:h:h}
cd "$ROOT"

MODE=${MODE:-remeasure-retained}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}

if [[ "$MODE" == "remeasure-retained" ]]; then
  OUTPUT=${OUTPUT:-"$ROOT/.omx/research/surrogate_vjp_fidelity_metric_remeasurement_20260714.json"}
  exec "$PYTHON" tools/probe_surrogate_vjp_fidelity_metric.py \
    remeasure-retained \
    --round2-root "${ROUND2_ROOT:-$ROOT/experiments/results/frozen_replay_convex_head_95kill_n600_20260713}" \
    --round3-root "${ROUND3_ROOT:-$ROOT/experiments/results/replace_round3_fidelity_wall_20260713}" \
    --output "$OUTPUT"
fi

if [[ "$MODE" == "preflight-n600" ]]; then
  required=(CACHE_MANIFEST BUNDLE_ROOT)
  for name in $required; do
    if [[ -z "${(P)name:-}" ]]; then
      print -u2 "BLOCKED: MODE=preflight-n600 requires $name"
      exit 2
    fi
  done
  PREFLIGHT_OUTPUT=${PREFLIGHT_OUTPUT:-"$ROOT/.omx/research/surrogate_vjp_fidelity_metric_n600_preflight_20260714.json"}
  exec "$PYTHON" tools/probe_surrogate_vjp_fidelity_metric.py \
    preflight-n600 \
    --cache-manifest "$CACHE_MANIFEST" \
    --bundle-root "$BUNDLE_ROOT" \
    --output "$PREFLIGHT_OUTPUT"
fi

if [[ "$MODE" != "refit-measure" ]]; then
  print -u2 "BLOCKED: MODE must be remeasure-retained, preflight-n600, or refit-measure"
  exit 2
fi

required=(CACHE_MANIFEST BUNDLE_ROOT SEALED_GATE_RECEIPT SEALED_TERMINALITY_RECEIPT)
for name in $required; do
  if [[ -z "${(P)name:-}" ]]; then
    print -u2 "BLOCKED: MODE=refit-measure requires $name"
    exit 2
  fi
done

if [[ ! -s "$CACHE_MANIFEST" || ! -d "$BUNDLE_ROOT" || ! -s "$SEALED_GATE_RECEIPT" || ! -s "$SEALED_TERMINALITY_RECEIPT" ]]; then
  print -u2 "BLOCKED: cache, bundle, gate receipt, and terminality receipt must already exist"
  exit 2
fi

# Authenticate the base cache first, then refuse.  The legacy whole-teacher
# fitter optimizes boundary-masked ambient RGB Sobolev cosine/L2.  It has no
# renderer state/J_R, active decision Jacobian, applied-step outcome, or sealed
# density ratio, so invoking it would NOT train the argmax-native objective and
# would make any Metal timing packet semantically false.
PREFLIGHT_OUTPUT=${PREFLIGHT_OUTPUT:-"$ROOT/.omx/research/surrogate_vjp_fidelity_metric_n600_preflight_20260714.json"}
"$PYTHON" tools/probe_surrogate_vjp_fidelity_metric.py \
  preflight-n600 \
  --cache-manifest "$CACHE_MANIFEST" \
  --bundle-root "$BUNDLE_ROOT" \
  --output "$PREFLIGHT_OUTPUT"

print -u2 "BLOCKED_IMPLEMENTATION: corrected argmax-native fit driver is not landed"
print -u2 "The legacy whole-teacher driver is a raw boundary-Sobolev baseline, not the requested training target."
print -u2 "Required: renderer/decision sufficient statistics + joint value/Jacobian objective + exact functional gate + terminality guard + resumable SSD run contract."
exit 2
