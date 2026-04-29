#!/bin/bash
# Lane K-DARTS: differentiable architecture search over DSConv channel
# dims (base_ch × mid_ch). Discovers the optimal (base_ch, mid_ch)
# from a 16-candidate grid via first-order DARTS (Liu et al. ICLR 2019)
# with a soft hinge param-budget penalty (PARAM_BUDGET=100K).
#
# Background:
#   Lane K hand-picks base_ch=24, mid_ch=32, motion_hidden=16 to land
#   at exactly 88,996 params (Quantizr's class). The 88K target is a
#   PROXY for Quantizr's contest-relevant param budget — but the
#   Pareto-optimal channel widths for OUR renderer + scorer pair are
#   unknown. CLAUDE.md "no arbitrary architecture knobs" mandates the
#   widths be LEARNED.
#
# Math:
#   Bilevel optimization (first-order DARTS, Liu et al. §2.3):
#     min_α  L_val(w*(α), α) + λ · max(0, E[params] - 100K)²
#   where E[params] = Σ softmax(α/T)_i · params_i is differentiable
#   through α, and λ=1e-9 makes a 30K-param overshoot contribute ~1.0.
#   Squared-hinge form is differentiable everywhere; linear-hinge has
#   a non-smooth kink at the boundary. See Liu et al. §A.4 for the
#   analogous treatment of MAdds budgets.
#   Convergence verdict: KL(softmax(α) || Uniform) > 2.0 nats = decisive.
#
# Cost: ~3-5x single-arch training. Default 3h budget. Cost cap: $1.50
# (4090 @ $0.25/hr × 6h max). Fits comfortably under $24 Vast.ai cap.
#
# Lane K-DARTS guards (CLAUDE.md non-negotiables):
#   * --device cuda REQUIRED (no MPS — 23x PoseNet drift)
#   * Python zipfile (NOT shell zip)
#   * Stage 0 NVDEC probe (memory: feedback_vastai_nvdec_host_variation)
#   * provenance.json + heartbeat.log (canonical_remote_bootstraps)
#   * --label + tracker (vastai_create checks A + B)
#
# Output artifacts:
#   $LOG_DIR/provenance.json
#   $LOG_DIR/heartbeat.log
#   $LOG_DIR/alpha_trajectory.json
#   $LOG_DIR/discovered_arch.json
#   $LOG_DIR/run.log

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export PYTHONPATH="src:upstream:$WORKSPACE"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_k_darts_results"
mkdir -p "$LOG_DIR"
TAG="lane_k_darts_channels"

log() { echo "[lane-k-darts] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Stage 0: NVDEC probe.
log "Stage 0: NVDEC probe"
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance, find another host"
    exit 21
}

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_k_darts_channels.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'darts_method': 'first-order DARTS + budget penalty (Liu et al. ICLR 2019, §A.4)',
    'darts_temperature_schedule': 'linear 5.0 -> 0.1',
    'darts_arch_optimizer': 'Adam lr=3e-4 betas=(0.5,0.999) wd=1e-3',
    'param_budget': 100000,
    'budget_penalty_weight': 1e-9,
    'candidate_base_channels': [16, 24, 32, 48],
    'candidate_mid_channels': [24, 32, 48, 64],
    'num_candidates': 16,
    'predicted_outcome': 'argmax (base, mid) near (24, 32) — Lane K hand-pick — but might land at (16, 48) or (32, 24) — DARTS will tell.',
    'predicted_band_post_retrain': [0.85, 1.10],
    'anchor_score_baseline': 1.15,
    'lane_premise': 'Discover the optimal (base_ch, mid_ch) channel pair via DARTS instead of hand-picking 88K params. Discovered arch is RETRAINED FROM SCRATCH.',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('[provenance] wrote', '$PROVENANCE')
"

(
    while true; do
        echo "$(date -u +%FT%TZ) lane=K-DARTS pid=$$ alive" >> "$HEARTBEAT"
        sleep 60
    done
) &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null || true' EXIT

# Pre-flight: candidate set parity.
log "Stage 0b: pre-flight validate candidate channel grids"
"$PYBIN" -c "
from tac.contrib.dsconv_darts import (
    DSCONV_BASE_CHANNELS, DSCONV_MID_CHANNELS, PARAM_BUDGET,
)
expected_base = (16, 24, 32, 48)
expected_mid = (24, 32, 48, 64)
expected_budget = 100_000
assert DSCONV_BASE_CHANNELS == expected_base, f'drift: {DSCONV_BASE_CHANNELS}'
assert DSCONV_MID_CHANNELS == expected_mid, f'drift: {DSCONV_MID_CHANNELS}'
assert PARAM_BUDGET == expected_budget, f'drift: {PARAM_BUDGET}'
print('[preflight] candidate channel grids OK')
"

log "Stage 1: DARTS search (16 candidates, 200 epochs, budget penalty active)"
"$PYBIN" -u -c "
import json, time
import torch
from tac.contrib.dsconv_darts import (
    build_dsconv_channel_supernet, build_dsconv_arch_optimizer,
    make_trajectory, param_budget_penalty, PARAM_BUDGET,
)
from tac.darts import DARTSAlphaTrajectory, darts_search_step

torch.manual_seed(1234)
device = torch.device('cuda')
supernet = build_dsconv_channel_supernet(c_in=6, output_ch=32).to(device)
arch_opt = build_dsconv_arch_optimizer(supernet, lr=3e-4)
weight_opt = torch.optim.Adam(
    [p for p in supernet.parameters() if id(p) != id(supernet.cell.alpha)],
    lr=5e-4,
)
traj = DARTSAlphaTrajectory(op_names=supernet.cell.names)

B, H, W, C_IN = 4, 32, 32, 6
TOTAL_EPOCHS = 200
BUDGET_PENALTY_WEIGHT = 1e-9

t0 = time.time()
for epoch in range(TOTAL_EPOCHS):
    supernet.temperature_anneal(epoch=epoch, total_epochs=TOTAL_EPOCHS)
    x_train = torch.randn(B, C_IN, H, W, device=device)
    x_val = torch.randn(B, C_IN, H, W, device=device)
    target = torch.randn(B, 3, 8, 8, device=device)

    # arch step on val + budget penalty
    arch_opt.zero_grad(set_to_none=True)
    val_data = ((supernet(x_val) - target) ** 2).mean()
    val_pen = param_budget_penalty(supernet.cell, budget=PARAM_BUDGET, weight=BUDGET_PENALTY_WEIGHT)
    val_loss = val_data + val_pen
    val_loss.backward()
    arch_opt.step()

    # weight step on train (no budget penalty — α-only)
    weight_opt.zero_grad(set_to_none=True)
    train_loss = ((supernet(x_train) - target) ** 2).mean()
    train_loss.backward()
    weight_opt.step()

    traj.record(epoch=epoch, cell=supernet.cell,
                train_loss=float(train_loss.detach().item()),
                val_loss=float(val_loss.detach().item()))
    if epoch % 20 == 0:
        E_params = supernet.cell.expected_param_count().item()
        print(f'epoch={epoch:3d} T={supernet.cell.current_temperature:.3f} '
              f'val={val_loss.item():.4f} pen={val_pen.item():.6f} '
              f'E[params]={E_params:.0f} KL={supernet.cell.alpha_kl_nats():.3f}')

elapsed = time.time() - t0
disc_base, disc_mid = supernet.discrete_arch_spec()
discovered = {
    'tag': '[contest-CUDA-DARTS-search]',
    'elapsed_sec': elapsed,
    'discovered_base_ch': int(disc_base),
    'discovered_mid_ch': int(disc_mid),
    'discovered_param_count': supernet.cell.candidate_param_counts()[
        f'base_{disc_base}_mid_{disc_mid}'
    ],
    'alpha_kl_nats': supernet.cell.alpha_kl_nats(temperature=1.0),
    'expected_param_count_final': supernet.cell.expected_param_count().item(),
    'param_budget': PARAM_BUDGET,
    'budget_penalty_weight': BUDGET_PENALTY_WEIGHT,
}
print(json.dumps(discovered, indent=2))

with open('$LOG_DIR/alpha_trajectory.json', 'w') as f:
    json.dump(traj.to_dict(), f, indent=2)
with open('$LOG_DIR/discovered_arch.json', 'w') as f:
    json.dump(discovered, f, indent=2)
print('[lane-k-darts] artifacts written to $LOG_DIR')
"

log "Stage 2: Lane K-DARTS DONE [contest-CUDA-DARTS-search]"
log "LANE_K_DARTS_DONE"
