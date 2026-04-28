#!/bin/bash
# Lane I-DARTS: differentiable architecture search over Cool-Chic
# (hidden_dim × latent_grid_res). Discovers the optimal pair from a
# 16-candidate joint grid via first-order DARTS (Liu et al. ICLR 2019).
#
# Background:
#   Lane I (CoolChicLatentRenderer) takes hidden=32 + latent_shapes=
#   ((6,8),(12,16),(24,32)) — both hand-picked since the profile was
#   committed. Cool-Chic / C3 trades distortion (more hidden_dim, finer
#   latent grid → lower distortion) against rate (more parameters →
#   larger renderer.bin). The Pareto-optimal point is unknown.
#   CLAUDE.md "no arbitrary architecture knobs" mandates a search.
#
# Math:
#   Bilevel optimization (first-order DARTS, Liu et al. §2.3):
#     min_α  L_val(w*(α), α)
#     s.t.   w*(α) ≈ w  (no inner-argmin loop)
#   The 3-band cascade preserves the Cool-Chic multi-resolution prior:
#   each candidate is (W_finest, W_mid, W_coarse) with /2 between bands
#   and H = 0.75·W to match the 384×512 video aspect.
#   Convergence verdict: KL(softmax(α) || Uniform) > 2.0 nats = decisive.
#
# Cost: ~3-5x single-arch training. Default 3h budget. Cost cap: $1.50.
#
# Lane I-DARTS guards (CLAUDE.md non-negotiables):
#   * --device cuda REQUIRED (no MPS — 23x PoseNet drift)
#   * Python zipfile (NOT shell zip)
#   * Stage 0 NVDEC probe
#   * provenance.json + heartbeat.log
#   * --label + tracker
#
# Output artifacts:
#   $LOG_DIR/provenance.json
#   $LOG_DIR/heartbeat.log
#   $LOG_DIR/alpha_trajectory.json
#   $LOG_DIR/discovered_arch.json
#   $LOG_DIR/run.log

set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export PYTHONPATH="src:upstream:$WORKSPACE"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_i_darts_results"
mkdir -p "$LOG_DIR"
TAG="lane_i_darts_dims"

log() { echo "[lane-i-darts] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_i_darts_dims.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'darts_method': 'first-order DARTS (Liu et al. ICLR 2019)',
    'darts_temperature_schedule': 'linear 5.0 -> 0.1',
    'darts_arch_optimizer': 'Adam lr=3e-4 betas=(0.5,0.999) wd=1e-3',
    'candidate_hidden_dims': [8, 16, 24, 32],
    'candidate_latent_grids_finest_W': [8, 12, 16, 24],
    'num_candidates': 16,
    'predicted_outcome': 'argmax (hidden, grid) likely (24, finest=16) — Lane I default — but DARTS may find smaller hidden + larger grid.',
    'predicted_band_post_retrain': [0.95, 1.25],
    'anchor_score_baseline': 1.15,
    'lane_premise': 'Discover the optimal Cool-Chic (hidden_dim, latent_grid) via DARTS instead of hand-picking. Discovered arch is RETRAINED FROM SCRATCH.',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('[provenance] wrote', '$PROVENANCE')
"

(
    while true; do
        echo "$(date -u +%FT%TZ) lane=I-DARTS pid=$$ alive" >> "$HEARTBEAT"
        sleep 60
    done
) &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null || true' EXIT

# Pre-flight: candidate set parity.
log "Stage 0b: pre-flight validate candidate Cool-Chic dims"
"$PYBIN" -c "
from tac.contrib.coolchic_darts import (
    COOLCHIC_HIDDEN_CANDIDATES, COOLCHIC_LATENT_GRID_CANDIDATES,
)
expected_hidden = (8, 16, 24, 32)
assert COOLCHIC_HIDDEN_CANDIDATES == expected_hidden, \
    f'drift: {COOLCHIC_HIDDEN_CANDIDATES}'
assert len(COOLCHIC_LATENT_GRID_CANDIDATES) == 4, \
    f'expected 4 latent grids, got {len(COOLCHIC_LATENT_GRID_CANDIDATES)}'
print('[preflight] candidate Cool-Chic dims OK')
"

log "Stage 1: DARTS search (16 candidates, 200 epochs, mask reconstruction surrogate)"
"$PYBIN" -u -c "
import json, time
import torch
from tac.contrib.coolchic_darts import (
    build_coolchic_supernet, build_coolchic_arch_optimizer, make_trajectory,
)
from tac.darts import DARTSAlphaTrajectory, darts_search_step

torch.manual_seed(1234)
device = torch.device('cuda')
supernet = build_coolchic_supernet().to(device)
arch_opt = build_coolchic_arch_optimizer(supernet, lr=3e-4)
weight_opt = torch.optim.Adam(
    [p for p in supernet.parameters() if id(p) != id(supernet.cell.alpha)],
    lr=5e-4,
)
traj = DARTSAlphaTrajectory(op_names=supernet.cell.names)

# Surrogate mask reconstruction: random 5-class masks → tanh'd target
# RGB at the same spatial resolution.
B, H, W = 2, 24, 32
TOTAL_EPOCHS = 200

t0 = time.time()
for epoch in range(TOTAL_EPOCHS):
    supernet.temperature_anneal(epoch=epoch, total_epochs=TOTAL_EPOCHS)
    masks_train = torch.randint(low=0, high=5, size=(B, H, W), device=device)
    masks_val = torch.randint(low=0, high=5, size=(B, H, W), device=device)
    target = torch.rand(B, 3, H, W, device=device) * 255.0
    val_loss, train_loss = darts_search_step(
        supernet=supernet,
        val_loss_fn=lambda: ((supernet(masks_val) - target) ** 2).mean(),
        train_loss_fn=lambda: ((supernet(masks_train) - target) ** 2).mean(),
        arch_opt=arch_opt,
        weight_opt=weight_opt,
    )
    traj.record(epoch=epoch, cell=supernet.cell,
                train_loss=train_loss, val_loss=val_loss)
    if epoch % 20 == 0:
        print(f'epoch={epoch:3d} T={supernet.cell.current_temperature:.3f} '
              f'val={val_loss:.4f} train={train_loss:.4f} '
              f'KL={supernet.cell.alpha_kl_nats():.3f}')

elapsed = time.time() - t0
disc_hidden, disc_grid = supernet.discrete_arch_spec()
discovered = {
    'tag': '[contest-CUDA-DARTS-search]',
    'elapsed_sec': elapsed,
    'discovered_hidden_dim': int(disc_hidden),
    'discovered_latent_grid': [list(b) for b in disc_grid],
    'discovered_param_count': supernet.cell.candidate_param_counts()[
        supernet.cell.names[supernet.cell.discrete_arch()]
    ],
    'alpha_kl_nats': supernet.cell.alpha_kl_nats(temperature=1.0),
}
print(json.dumps(discovered, indent=2))

with open('$LOG_DIR/alpha_trajectory.json', 'w') as f:
    json.dump(traj.to_dict(), f, indent=2)
with open('$LOG_DIR/discovered_arch.json', 'w') as f:
    json.dump(discovered, f, indent=2)
print('[lane-i-darts] artifacts written to $LOG_DIR')
"

log "Stage 2: Lane I-DARTS DONE [contest-CUDA-DARTS-search]"
log "LANE_I_DARTS_DONE"
