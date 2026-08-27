# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""ddm_fd1 S1 smoke: one GN/CG proposal on the W_joint warm-start block.

Measures (not asserts): loss_and_grad cost, one HVP cost (mx.jvp+vjp through
paint -> uint8-STE -> fused R -> SegNet), a small CG solve, and the
model-reduction sanity of the returned step against the block proxy objective.
[macOS-MLX research-signal] only; score_claim=false; pointer UNMOVED.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = Path("/Users/adpena/Projects/pact")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MAIN))


def main() -> int:
    import os

    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.optimization.ddm_family_d_gn_description import FamilyDGaussNewtonEngineV1
    from tac.optimization.direct_description_joint_descent import (
        DirectDescriptionJointDescentMLXModule,
        lift_v15_archive,
        parameter_group_indices,
        realized_training_state,
    )

    source = Path(
        "/Volumes/VertigoDataTier/pact/experiments/results/"
        "ddm_ws2_warm_start_custody_producer_20260724T053455Z/01_archives/W_joint.zip.receipt-bytes"
    )
    archive = source.read_bytes()
    ckpt = np.load(
        "/Volumes/VertigoDataTier/pact/experiments/results/"
        "ddm_ws3_w_joint_exact_history_20260724T132200Z/checkpoints/"
        "01_residual_bucket_realized_acceptance_intra_global000004.npz"
    )
    theta = np.ascontiguousarray(ckpt["theta"], dtype=np.float32)

    t0 = time.time()
    lift = lift_v15_archive(archive)
    groups = parameter_group_indices(lift)
    cache = MAIN / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
    labels = open_stored_npy_memmap(cache, "lstars")
    poses = open_stored_npy_memmap(cache, "gt_poses")
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(str(MAIN / "upstream"), device="cpu")
    model = DirectDescriptionJointDescentMLXModule(
        lift=lift, scorer_adapter=adapter, seg_targets=labels, pose_targets=poses
    )
    print(f"[fd1-smoke] setup {time.time()-t0:.1f}s parameter_count={model.parameter_count}")

    pair_ids = (447, 448, 449, 450)
    active_groups = ("island_worldsheet", "shared_template_dof")
    t1 = time.time()
    base_camera, masks, basis, basis_idx, local_theta, _ = realized_training_state(
        lift, theta, pair_ids=pair_ids, active_groups=active_groups, include_lane_programs=False
    )
    print(f"[fd1-smoke] realized_training_state {time.time()-t1:.1f}s basis K={basis.shape[0]}")

    t2 = time.time()
    loss, grad = model.loss_and_grad(
        local_theta,
        pair_ids=pair_ids,
        base_camera=base_camera,
        template_masks=masks,
        realized_secant_basis=basis,
        realized_secant_indices=basis_idx,
        pose_objective_weight=0.0,
    )
    t_grad = time.time() - t2
    active = sorted(set().union(*(groups[g] for g in active_groups)) | set(basis_idx))
    grad = grad.copy()
    inactive = [i for i in range(len(grad)) if i not in set(active)]
    grad[inactive] = 0.0
    print(f"[fd1-smoke] loss_and_grad {t_grad:.1f}s loss={loss:.6f} |g_active|={np.linalg.norm(grad):.4e}")

    engine = FamilyDGaussNewtonEngineV1(model, repository_root=str(REPO), hutchinson_probes=2, seed=0)
    print(f"[fd1-smoke] metric custody: {engine.metric_custody.get('bundle_status')}")

    t3 = time.time()
    delta, diag = engine.propose(
        local_theta,
        grad,
        pair_ids=pair_ids,
        base_camera=base_camera,
        template_masks=masks,
        realized_secant_basis=basis,
        realized_secant_indices=basis_idx,
        active_indices=active,
        damping=1.0e-3,
        cg_iterations=3,
    )
    t_gn = time.time() - t3
    payload = diag.to_payload()
    payload["propose_seconds"] = t_gn
    payload["per_hvp_seconds"] = t_gn / max(payload["hvp_calls"], 1)

    # Block-proxy check of the GN model: does theta + delta reduce the proxy?
    metrics = model.measure_components(
        local_theta + delta,
        pair_ids=pair_ids,
        base_camera=base_camera,
        template_masks=masks,
        realized_secant_basis=basis,
        realized_secant_indices=basis_idx,
    )
    payload["block_proxy_before"] = float(loss)
    payload["block_proxy_after_full_step"] = 100.0 * float(metrics["seg_ce_margin"])
    payload["block_proxy_delta_full_step"] = payload["block_proxy_after_full_step"] - float(loss)
    payload["score_claim"] = False
    out = REPO / ".omx/research/ddm_fd1_gn_engine_smoke_20260728.json"
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    print(f"[fd1-smoke] receipt -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
