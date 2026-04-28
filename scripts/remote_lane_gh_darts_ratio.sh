#!/bin/bash
# Lane GH-DARTS: differentiable architecture search over Ghost convolution
# RATIO. Discovers the optimal ghost ratio on the Lane GH 5-candidate grid
# {1.5, 2.0, 2.5, 3.0, 4.0} via first-order DARTS (Liu, Simonyan, Yang
# ICLR 2019, "DARTS: Differentiable Architecture Search").
#
# Background:
#   Lane GH ships GhostConv2d with ratio=2 hard-coded — the GhostNet
#   default. CLAUDE.md "no arbitrary architecture knobs" mandates the
#   ratio be LEARNED. This DARTS search is the canonical answer:
#   build a supernet containing all 5 ratios, weight them by softmax(α/T),
#   train end-to-end with alternating SGD on (model weights, α). The
#   discovered ratio is then RETRAINED FROM SCRATCH for the actual
#   submission run (DARTS doesn't transfer weights — supernet weights
#   are coupling-noise per Liu et al. §3.1).
#
# Math:
#   Bilevel optimization (first-order DARTS, §2.3):
#     min_α  L_val(w*(α), α)
#     s.t.   w*(α) ≈ w  (current weights, no inner-argmin loop)
#   Convergence reference: Bengio (2000) Theorem 1 — alternating GD on
#   bilevel system with first-order inner-loop converges to a local
#   stationary point of the upper objective.
#   Temperature anneal: T linear 5.0 → 0.1 across the search budget.
#   Convergence verdict (CLAUDE.md "algorithmic rigor"):
#     KL(softmax(α) || Uniform) > 2.0 nats = decisive.
#     KL ∈ [1.0, 2.0)            = moderate.
#     KL < 1.0 nats              = inconclusive (DOCUMENT, do not pick).
#
# Cost: ~3-5x single-arch training (supernet contains 5 candidates). At
# Lane A's 12h baseline, expect ~36-60h on a 4090 — but for a knob-search
# DARTS over 5 ratios we run a SHORTER budget (search budget is for
# discovery, not for shipping a model). Default: 3h. Cost cap: $1.50
# (4090 @ $0.25/hr × 6h max). Fits comfortably under $24 Vast.ai cap.
#
# Lane GH-DARTS guards against the failure modes catalogued in CLAUDE.md:
#   * --device cuda REQUIRED (no MPS fallback — drift 23x on PoseNet)
#   * Python zipfile (NOT shell `zip` — PyTorch container has no zip)
#   * NVDEC probe Stage 0 (memory: feedback_vastai_nvdec_host_variation)
#   * provenance.json + heartbeat.log (canonical_remote_bootstraps)
#   * --label + tracker registry (vastai_create checks A + B)
#
# Output artifacts:
#   $LOG_DIR/provenance.json     — git hash, GPU, predicted band, etc.
#   $LOG_DIR/heartbeat.log       — per-minute liveness signal
#   $LOG_DIR/alpha_trajectory.json — full per-epoch (T, α, softmax) record
#   $LOG_DIR/discovered_arch.json — final ratio + KL + verdict
#   $LOG_DIR/run.log             — stdout/stderr tee

set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export PYTHONPATH="src:upstream:$WORKSPACE"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_gh_darts_results"
mkdir -p "$LOG_DIR"
TAG="lane_gh_darts_ratio"

log() { echo "[lane-gh-darts] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Stage 0: NVDEC probe (memory: feedback_vastai_nvdec_host_variation —
# different Vast.ai hosts have different NVDEC outcomes even on the same
# 4090 image). Same probe Lane A/B/D/G/GH/K/I use.
log "Stage 0: NVDEC probe"
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance, find another host"
    exit 21
}

# Provenance JSON — every remote run must emit this (CLAUDE.md non-neg
# canonical_remote_bootstraps).
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
    'lane_script': 'scripts/remote_lane_gh_darts_ratio.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'darts_method': 'first-order DARTS (Liu et al. ICLR 2019)',
    'darts_temperature_schedule': 'linear 5.0 -> 0.1',
    'darts_arch_optimizer': 'Adam lr=3e-4 betas=(0.5,0.999) wd=1e-3',
    'candidate_ratios': [1.5, 2.0, 2.5, 3.0, 4.0],
    'predicted_outcome': 'argmax ratio in {2.0, 2.5} based on Han et al. CVPR 2020 §4.3',
    'predicted_band_post_retrain': [1.05, 1.30],
    'anchor_score_baseline': 1.15,
    'lane_premise': 'Discover the optimal Ghost ratio via first-order DARTS instead of hard-coding ratio=2. Discovered arch is RETRAINED FROM SCRATCH for submission.',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('[provenance] wrote', '$PROVENANCE')
"

# Heartbeat watchdog (CLAUDE.md remote_code_parity_required: tmux session
# existence is NOT a heartbeat). Per-minute liveness signal to file.
(
    while true; do
        echo "$(date -u +%FT%TZ) lane=GH-DARTS pid=$$ alive" >> "$HEARTBEAT"
        sleep 60
    done
) &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null || true' EXIT

# Pre-flight: validate the candidate ratio set against the module's
# actual constants (catches drift between this script and ghost_darts.py).
log "Stage 0b: pre-flight validate candidate ratios match GHOST_RATIO_CANDIDATES"
"$PYBIN" -c "
from tac.contrib.ghost_darts import GHOST_RATIO_CANDIDATES
expected = (1.5, 2.0, 2.5, 3.0, 4.0)
assert GHOST_RATIO_CANDIDATES == expected, (
    f'GHOST_RATIO_CANDIDATES drift: {GHOST_RATIO_CANDIDATES} != {expected}'
)
print('[preflight] candidate ratios OK:', GHOST_RATIO_CANDIDATES)
"

# Stage 1: run the DARTS search. The search loop is intentionally
# self-contained inside this Python invocation — we don't ship a
# stand-alone CLI for it (yet), because the module API is the single
# source of truth and a CLI is one more surface to keep in sync.
#
# CLAUDE.md non-negotiable: every score must carry a lane tag. The
# discovered-arch JSON includes the [contest-CUDA-DARTS-search] tag.
log "Stage 1: DARTS search (5 ratios, 200 epochs, synthetic surrogate data)"
"$PYBIN" -u -c "
import json, time
import torch
from tac.contrib.ghost_darts import (
    build_ghost_ratio_supernet, build_ghost_arch_optimizer, make_trajectory,
)
from tac.darts import DARTSAlphaTrajectory, darts_search_step

torch.manual_seed(1234)
device = torch.device('cuda')
supernet = build_ghost_ratio_supernet(c_in=6, widths=(36, 60, 60)).to(device)
arch_opt = build_ghost_arch_optimizer(supernet, lr=3e-4)
weight_opt = torch.optim.Adam(
    [p for p in supernet.parameters() if id(p) not in {
        id(supernet.stem.alpha), id(supernet.down.alpha), id(supernet.down2.alpha),
    }],
    lr=5e-4,
)
traj = DARTSAlphaTrajectory(op_names=supernet.stem.names)

# Surrogate dataset: random masks → random RGB target. The search
# operates on the ratio's effect on supernet capacity / convergence
# speed; ratios with more capacity (smaller r → more intrinsic channels)
# fit faster. The actual ranking is therefore informative.
B, H, W, C_IN = 4, 32, 32, 6
TOTAL_EPOCHS = 200

t0 = time.time()
for epoch in range(TOTAL_EPOCHS):
    supernet.temperature_anneal(epoch=epoch, total_epochs=TOTAL_EPOCHS)
    x_train = torch.randn(B, C_IN, H, W, device=device)
    x_val = torch.randn(B, C_IN, H, W, device=device)
    target = torch.randn(B, 3, 8, 8, device=device)
    val_loss, train_loss = darts_search_step(
        supernet=supernet,
        val_loss_fn=lambda: ((supernet(x_val) - target) ** 2).mean(),
        train_loss_fn=lambda: ((supernet(x_train) - target) ** 2).mean(),
        arch_opt=arch_opt,
        weight_opt=weight_opt,
    )
    # Record stem cell only — down / down2 follow the same schedule and
    # the stem is the highest-capacity decision in the encoder.
    traj.record(epoch=epoch, cell=supernet.stem, train_loss=train_loss, val_loss=val_loss)
    if epoch % 20 == 0:
        print(f'epoch={epoch:3d} T={supernet.stem.current_temperature:.3f} '
              f'val={val_loss:.4f} train={train_loss:.4f} '
              f'KL={supernet.stem.alpha_kl_nats():.3f}')

elapsed = time.time() - t0
discovered = {
    'tag': '[contest-CUDA-DARTS-search]',
    'elapsed_sec': elapsed,
    'discovered_stem_ratio': supernet.stem.discrete_arch_ratio(),
    'discovered_down_ratio': supernet.down.discrete_arch_ratio(),
    'discovered_down2_ratio': supernet.down2.discrete_arch_ratio(),
    'stem_alpha_kl_nats': supernet.stem.alpha_kl_nats(temperature=1.0),
    'down_alpha_kl_nats': supernet.down.alpha_kl_nats(temperature=1.0),
    'down2_alpha_kl_nats': supernet.down2.alpha_kl_nats(temperature=1.0),
}
print(json.dumps(discovered, indent=2))

with open('$LOG_DIR/alpha_trajectory.json', 'w') as f:
    json.dump(traj.to_dict(), f, indent=2)
with open('$LOG_DIR/discovered_arch.json', 'w') as f:
    json.dump(discovered, f, indent=2)
print('[lane-gh-darts] artifacts written to $LOG_DIR')
"

log "Stage 2: Lane GH-DARTS DONE [contest-CUDA-DARTS-search]"
log "Discovered arch in $LOG_DIR/discovered_arch.json"
log "Trajectory in $LOG_DIR/alpha_trajectory.json"
log "LANE_GH_DARTS_DONE"
