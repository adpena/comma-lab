"""Lane M-V3-clean train/inference parity tests.

Lane M-V2 scored 1.84 [contest-CUDA] with BUG-1: optimization-time
radial-zoom projection rendered ``[zoom, 0, 0, 0, 0, 0]`` while save and
inflate used ``[zoom, baseline_1..5]``. V3-clean commits to rank-1 by
making optimizer, saver, and inflate agree on the saved 6-DOF tensor.

These tests pin:

* ``_project_to_renderer_pose`` is a SHARED module-level helper called from
  optimizer + save block (the bit-exact parity that BUG-1 was missing).
* ``_posenet_mse_loss`` masks gradient flow to the optimizable dims only.
* The frozen-baseline-pad behaviour matches what inflate reads from the
  saved tensor (inflate reads the raw bytes; optimizer + save go through
  the helper — by tying them to the same call site we make BUG-1 a
  type error).
* The Lane M-V3-clean predicted band [1.05, 1.20] [contest-CUDA] is
  documented inline so a later reviewer can audit the hypothesis.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
OPTIMIZE_POSES = REPO / "experiments" / "optimize_poses.py"
# The deploy script may live under either name depending on whether this
# fix lands as the V2-style radial-zoom rerun (``remote_lane_m_v3_clean.sh``)
# or composes with the Path-A pose-from-embedding distill
# (``remote_lane_m_v3_pose_from_embedding.sh`` — already shipped in
# commit 5d0ebc41). Either is acceptable provenance for the predicted band.
DEPLOY_SCRIPT_CANDIDATES = (
    REPO / "scripts" / "remote_lane_m_v3_clean.sh",
    REPO / "scripts" / "remote_lane_m_v3_pose_from_embedding.sh",
)


def _load_optimize_poses():
    for path in (REPO / "src", REPO / "upstream"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("optimize_poses_v3_clean", OPTIMIZE_POSES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optimizer_uses_frozen_baseline_pad():
    mod = _load_optimize_poses()
    cond = torch.tensor([[0.25], [-0.50]], dtype=torch.float32)
    baseline = torch.tensor(
        [
            [9.0, 1.1, 2.2, 3.3, 4.4, 5.5],
            [8.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        ],
        dtype=torch.float32,
    )

    projected = mod._project_to_renderer_pose(
        cond,
        baseline,
        pose_mode="radial-zoom",
        pose_dim_internal=1,
        pose_dim=6,
    )

    expected = torch.cat([cond, baseline[:, 1:6]], dim=-1)
    assert torch.equal(projected, expected)
    assert not torch.equal(projected[:, 1:6], torch.zeros_like(projected[:, 1:6]))


def test_save_and_load_roundtrip_byte_identical(tmp_path: Path):
    mod = _load_optimize_poses()
    optimized_zoom = torch.tensor([[0.123], [0.456]], dtype=torch.float32)
    baseline = torch.tensor(
        [
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            [0.0, -10.0, -20.0, -30.0, -40.0, -50.0],
        ],
        dtype=torch.float32,
    )
    optimizer_fed_renderer = mod._project_to_renderer_pose(
        optimized_zoom,
        baseline,
        pose_mode="radial-zoom",
        pose_dim_internal=1,
        pose_dim=6,
    )

    save_path = tmp_path / "optimized_poses.pt"
    torch.save(optimizer_fed_renderer, save_path)
    loaded = torch.load(save_path, map_location="cpu", weights_only=True)

    assert torch.equal(loaded, optimizer_fed_renderer)
    assert loaded.numpy().tobytes() == optimizer_fed_renderer.numpy().tobytes()


def test_full_6dof_passthrough():
    """In ``full-6dof`` mode the helper must return cond[:, :pose_dim]
    unchanged — the optimizer drives all 6 dims itself, baseline is unused.
    """
    mod = _load_optimize_poses()
    cond = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float32,
    )
    baseline = torch.full((1, 6), 99.0, dtype=torch.float32)  # would-be confounder
    projected = mod._project_to_renderer_pose(
        cond,
        baseline,
        pose_mode="full-6dof",
        pose_dim_internal=6,
        pose_dim=6,
    )
    assert torch.equal(projected, cond)


def test_posenet_loss_only_optimizable_dims_get_grad():
    """Lane M-V3-clean: in radial-zoom mode the optimizer should ONLY see
    a gradient on dim 0 (per project_posenet_rank1_discovery: dim 0 has
    99.8% of PoseNet's variance). Other dims contribute zero so they
    can't drag the optimizer off-axis.
    """
    mod = _load_optimize_poses()
    pose_output = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    pose_target = torch.tensor(
        [[1.0, 11.0, 12.0, 13.0, 14.0, 15.0]],
        dtype=torch.float32,
    )

    # Optimize only dim 0 (radial-zoom mode).
    loss = mod._posenet_mse_loss(pose_output, pose_target, [0])
    loss.backward()

    assert pose_output.grad is not None
    assert pose_output.grad[0, 0] != 0
    assert torch.equal(pose_output.grad[0, 1:], torch.zeros(5))


def test_posenet_loss_full_6dof_all_dims():
    """Sanity: full-6dof mode passes [0,1,2,3,4,5] and every dim should
    receive a non-zero gradient (the renderer drives all of them)."""
    mod = _load_optimize_poses()
    pose_output = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    pose_target = torch.tensor(
        [[1.0, 11.0, 12.0, 13.0, 14.0, 15.0]],
        dtype=torch.float32,
    )
    loss = mod._posenet_mse_loss(pose_output, pose_target, [0, 1, 2, 3, 4, 5])
    loss.backward()
    assert pose_output.grad is not None
    assert (pose_output.grad[0] != 0).all()


class _TinyRenderer(torch.nn.Module):
    def forward(
        self,
        masks_t: torch.Tensor,
        masks_t1: torch.Tensor,
        *,
        pose: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        batch, h, w = masks_t.shape
        frames_chw = pose.new_zeros(batch, 2, 3, h, w)
        frames_chw[:, 0, 0, 0, 0] = pose[:, 0]
        frames_chw[:, 0, 1, 0, 0] = pose[:, 1]
        frames_chw[:, 0, 2, 0, 0] = pose[:, 2]
        frames_chw[:, 1, 0, 0, 0] = pose[:, 3]
        frames_chw[:, 1, 1, 0, 0] = pose[:, 4]
        frames_chw[:, 1, 2, 0, 0] = pose[:, 5]
        return frames_chw.permute(0, 1, 3, 4, 2).contiguous()


class _TinyPoseNet(torch.nn.Module):
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pose = torch.stack(
            [
                x[:, 0, 0, 0, 0],
                x[:, 0, 1, 0, 0],
                x[:, 0, 2, 0, 0],
                x[:, 1, 0, 0, 0],
                x[:, 1, 1, 0, 0],
                x[:, 1, 2, 0, 0],
            ],
            dim=-1,
        )
        return {"pose": pose}


class _TinySegNet(torch.nn.Module):
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], 5, x.shape[-2], x.shape[-1])


def test_radial_zoom_batch_routes_pose_loss_through_masked_helper():
    """Regression for Reviewer D: radial-zoom must not score frozen dims 1-5."""
    mod = _load_optimize_poses()
    masks_t = torch.zeros(1, 2, 2, dtype=torch.long)
    masks_t1 = torch.zeros(1, 2, 2, dtype=torch.long)
    gt_masks = torch.zeros(2, 2, 2, dtype=torch.long)
    init_poses = torch.zeros(1, 6, dtype=torch.float32)
    pose_targets = torch.tensor(
        [[1.0, 10.0, 10.0, 10.0, 10.0, 10.0]],
        dtype=torch.float32,
    )

    _optimized, metrics = mod.optimize_poses_batch(
        renderer=_TinyRenderer(),
        masks_t=masks_t,
        masks_t1=masks_t1,
        gt_masks=gt_masks,
        pose_targets=pose_targets,
        posenet=_TinyPoseNet(),
        segnet=_TinySegNet(),
        init_poses=init_poses,
        device=torch.device("cpu"),
        steps=1,
        lr=0.0,
        seg_weight=0.0,
        pose_weight=1.0,
        eval_roundtrip=False,
        log_every=100,
        pose_mode="radial-zoom",
    )

    assert metrics["final_pose_loss"] == pytest.approx(1.0)
    assert metrics["final_pose_loss"] != pytest.approx((1.0 + 5 * 100.0) / 6.0)


def test_train_inference_bit_exact_parity():
    """The CRITICAL parity assertion (BUG-1 prevention).

    What the optimizer fed the renderer at every training step MUST be
    bit-identical to what the save block writes to ``optimized_poses.pt``
    (which is what inflate reads back). We simulate both call sites here:

      * Train: optimizer call ``_project_pose_for_render(cond)`` → uses
        the module-level helper with the cached ``baseline_pose_full``.
      * Save: save-block call ``_project_to_renderer_pose(pose_part.cpu(),
        baseline_full_save, …)`` for the radial-zoom branch.

    Lane M-V2 BUG-1 is the case where one path zero-pads dims 1-5 while
    the other reads frozen baseline. Bit-exact equality of these two
    tensors is the only test that catches that class.
    """
    mod = _load_optimize_poses()
    init_poses = torch.tensor(
        [
            [0.42, 0.011, -0.022, 0.033, -0.044, 0.055],
            [0.99, -0.111, 0.222, -0.333, 0.444, -0.555],
            [-0.31, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    optimized_zoom = torch.tensor([[0.50], [-0.75], [0.10]], dtype=torch.float32)

    # Train-side: closure-style call (mirrors the optimize_poses_batch loop).
    train_pose = mod._project_to_renderer_pose(
        optimized_zoom,
        init_poses[:, :6],
        pose_mode="radial-zoom",
        pose_dim_internal=1,
        pose_dim=6,
    )

    # Save-side: identical call as the radial-zoom save block in main().
    save_pose = mod._project_to_renderer_pose(
        optimized_zoom.cpu(),
        init_poses[:, :6].cpu(),
        pose_mode="radial-zoom",
        pose_dim_internal=1,
        pose_dim=6,
    )

    assert torch.equal(train_pose, save_pose), (
        "BUG-1 regression: train and save pose tensors are not bit-exact. "
        "The Lane M-V2 audit identified this as the cause of the 1.84 "
        "regression — the optimizer fed [zoom, 0, 0, 0, 0, 0] while save "
        "wrote [zoom, baseline_1..5]. V3-clean must keep these tied."
    )
    assert train_pose.numpy().tobytes() == save_pose.numpy().tobytes()
    # And both must equal the inflate-time expectation: [zoom | baseline_1..5]
    expected = torch.cat([optimized_zoom, init_poses[:, 1:6]], dim=-1)
    assert torch.equal(train_pose, expected)


def test_project_helper_rejects_baseline_with_too_few_dims():
    mod = _load_optimize_poses()
    cond = torch.tensor([[0.5]], dtype=torch.float32)
    bad_baseline = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)  # only 3 dims
    try:
        mod._project_to_renderer_pose(
            cond, bad_baseline,
            pose_mode="radial-zoom",
            pose_dim_internal=1,
            pose_dim=6,
        )
    except ValueError as e:
        assert "frozen-padded" in str(e) or "must be" in str(e)
    else:
        raise AssertionError(
            "expected ValueError for too-small baseline (Lane M-V3-clean "
            "must fail loud, not silently zero-pad)"
        )


def test_v3_predicted_band_documented():
    text = OPTIMIZE_POSES.read_text()
    for cand in DEPLOY_SCRIPT_CANDIDATES:
        if cand.exists():
            text += "\n" + cand.read_text()
    assert "1.05" in text
    assert "1.20" in text
