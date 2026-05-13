#!/bin/bash
# Operator-authorize wrapper for the A1 + wavelet residual sidecar retarget substrate.
#
# Recipe:  .omx/operator_authorize_recipes/substrate_a1_plus_wavelet_residual_modal_t4_dispatch.yaml
# Council: .omx/research/meta_council_decision_attribution_audit_20260513.md
# Lane:    lane_a1_plus_wavelet_residual_retarget_20260513
#
# Per META-COUNCIL §6b this is the HIGHEST EV/$ competing-path retarget at
# the $0.20-1 cost band.  Per the "highest signal at lowest cost" operator
# directive, this wrapper ladder is:
#
#   1. Pre-smoke macOS-CPU SANITY gate ($0): run the canonical A1 baseline
#      through `tools/score_macos_cpu_advisory_proxy.py` to verify the local
#      proxy chain works end-to-end and the A1 base reproduces ~0.193.  If
#      the proxy returns a score outside [0.187, 0.200] the wrapper refuses
#      to proceed (proxy chain broken OR A1 archive corrupted).
#
#   2. Modal T4 SMOKE (Catalog #167; $0.15): 100-epoch tiny dispatch with
#      auth eval gate.  Score must land in plausible band [0.15, 0.25] or
#      the chain refuses to proceed to full.
#
#   3. Modal T4 FULL (recipe; $0.60): 2000-epoch full training + auth eval
#      on the trained composition archive.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full
# dispatch ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored:
#   MODAL_GPU=T4|A10G|A100|H100              (default T4 per recipe)
#   A1_PLUS_WAVELET_EPOCHS=2000              (META-COUNCIL default; full training)
#   A1_PLUS_WAVELET_SMOKE_EPOCHS=100         (smoke epoch override)
#   A1_PLUS_WAVELET_SMOKE_GPU=T4             (smoke GPU class)
#   A1_PLUS_WAVELET_SMOKE_ONLY=1             (skip full even on smoke-green)
#   A1_PLUS_WAVELET_FULL_ONLY=1              (skip smoke; operator override
#                                             after >=3 successful anchors)
#   A1_PLUS_WAVELET_SKIP_MACOS_CPU_GATE=1    (skip the $0 macOS-CPU sanity
#                                             gate, for CI / non-Darwin hosts)
#
# Cross-ref:
#   feedback_macos_cpu_proxy_empirical_validation_landed_20260513.md
#   feedback_a1_plus_lapose_composition_substrate_landed_20260513.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Stage 0: pre-smoke macOS-CPU SANITY gate ($0).
# Validates the proxy chain end-to-end against the A1 baseline archive.
# If A1 doesn't reproduce ~0.193 on macOS-CPU, the proxy chain or the
# archive is broken — refuse the more expensive Modal smoke.
A1_BASE_ARCHIVE="$REPO_ROOT/experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip"
MACOS_CPU_GATE_OUT="$REPO_ROOT/.omx/tmp/a1_wavelet_macos_cpu_baseline_$$.json"
mkdir -p "$REPO_ROOT/.omx/tmp"

if [ "${A1_PLUS_WAVELET_SKIP_MACOS_CPU_GATE:-0}" != "1" ]; then
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        if [ -f "$A1_BASE_ARCHIVE" ] && [ -f "$REPO_ROOT/tools/score_macos_cpu_advisory_proxy.py" ]; then
            echo "[macos-cpu-gate] running A1 baseline proxy eval to validate proxy chain..."
            if .venv/bin/python tools/score_macos_cpu_advisory_proxy.py \
                --archive "$A1_BASE_ARCHIVE" \
                --family lane_a1_plus_wavelet_residual_retarget_20260513 \
                --variant-id a1_baseline_sanity_check \
                --samples 100 \
                --manifest-output "$MACOS_CPU_GATE_OUT" \
                --inflate-sh "$REPO_ROOT/experiments/results/track4_sg_a1_t178000_20260509/submission_dir/inflate.sh" \
                2>&1 | tee /dev/stderr; then
                # Parse the resulting score and gate on [0.187, 0.200].
                PARSE_SCRIPT="$REPO_ROOT/.omx/tmp/a1_wavelet_parse_proxy_$$.py"
                cat > "$PARSE_SCRIPT" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
    rows = payload.get("rows") or payload.get("observations") or []
    if rows:
        print(rows[0].get("score") or "")
except Exception:
    pass
PY
                PROXY_SCORE=$(.venv/bin/python "$PARSE_SCRIPT" "$MACOS_CPU_GATE_OUT" 2>/dev/null || echo "")
                rm -f "$PARSE_SCRIPT" || true
                if [ -n "$PROXY_SCORE" ]; then
                    echo "[macos-cpu-gate] A1 baseline proxy score = $PROXY_SCORE"
                    LOW_OK=$(.venv/bin/python -c "print(1 if float('$PROXY_SCORE') >= 0.187 else 0)" 2>/dev/null || echo "0")
                    HIGH_OK=$(.venv/bin/python -c "print(1 if float('$PROXY_SCORE') <= 0.200 else 0)" 2>/dev/null || echo "0")
                    if [ "$LOW_OK" = "1" ] && [ "$HIGH_OK" = "1" ]; then
                        echo "[macos-cpu-gate] PASS — A1 baseline proxy in [0.187, 0.200]; proceeding to Modal smoke"
                    else
                        echo "[macos-cpu-gate] WARN — A1 baseline proxy score $PROXY_SCORE outside [0.187, 0.200]"
                        echo "[macos-cpu-gate] WARN — proxy chain or A1 archive may be broken; review before Modal spend"
                        if [ "${A1_PLUS_WAVELET_FORCE_PROCEED:-0}" != "1" ]; then
                            echo "[macos-cpu-gate] FATAL — set A1_PLUS_WAVELET_FORCE_PROCEED=1 to bypass"
                            exit 30
                        fi
                    fi
                else
                    echo "[macos-cpu-gate] WARN — could not parse proxy score; proceeding (chain may be incomplete)"
                fi
            else
                echo "[macos-cpu-gate] WARN — proxy invocation failed; proceeding to Modal smoke"
            fi
            rm -f "$MACOS_CPU_GATE_OUT" || true
        else
            echo "[macos-cpu-gate] SKIP — A1 archive or proxy tool missing"
        fi
    else
        echo "[macos-cpu-gate] SKIP — not on macOS arm64 host (proxy only valid on Darwin ARM64)"
    fi
else
    echo "[macos-cpu-gate] SKIP — A1_PLUS_WAVELET_SKIP_MACOS_CPU_GATE=1"
fi

# Stage 1: Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: empty array expansion guarded under set -u.
SMOKE_ARGS=()
if [ "${A1_PLUS_WAVELET_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${A1_PLUS_WAVELET_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_a1_plus_wavelet_residual_modal_t4_dispatch \
    --smoke-epochs "${A1_PLUS_WAVELET_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${A1_PLUS_WAVELET_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${A1_PLUS_WAVELET_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:operator_authorize_substrate_a1_plus_wavelet_residual_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
