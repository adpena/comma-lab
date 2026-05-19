# SPDX-License-Identifier: MIT
"""Phase B Option B regression tests: PoseNet shape-adapter fix in _eval_on_device.

Verifies that the 1-line shape-adapter fix in
``tac.mps_gap_experiment.harvest_and_verdict._eval_on_device`` (commit
landing 2026-05-19 per `.omx/research/mps_phase_b_re_fire_split_device_verdict_20260519T060500Z.md`
op-routable #2) produces a REAL numerical value for `posenet_mean_output`
(not NaN) when the canonical scorer preprocess pipeline is honored
(Catalog #164 + upstream PoseNet.preprocess_input).

PRE-FIX behavior: `posenet(reconstruction)` with reconstruction shape
``(B, 2, 3, 384, 512)`` raised inside PoseNet's vision backbone (FastViT-T12
expects 12-channel YUV6 ``(B, 12, 192, 256)``) and the diagnostic try/except
fell back to NaN. POST-FIX behavior: `posenet.preprocess_input(reconstruction)`
applies the canonical
``rearrange + interpolate + rgb_to_yuv6 + rearrange`` chain, yielding
``(B, 12, 192, 256)``, and the forward returns the `{'pose': ...}` head
dict from which we extract the canonical mean.

Catalog cross-refs: #164 (canonical scorer-preprocess routing), #229
(premise verification before edit), #192 (macOS-CPU advisory
non-promotion), #317 (local research-signal evidence stamping), #1
(MPS-fallback trap), #205 (canonical inflate device selector).

NOT a contest substrate suite — purely diagnostic infrastructure tests.
"""

from __future__ import annotations

from pathlib import Path

import math

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("timm")
pytest.importorskip("segmentation_models_pytorch")

from tac.mps_gap_experiment.harvest_and_verdict import (
    _eval_on_device,
    compute_local_mps_reference_components,
)
from tac.mps_gap_experiment.tiny_renderer import build_tiny_renderer


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
UPSTREAM_DIR = REPO_ROOT / "upstream"
POSENET_PATH = UPSTREAM_DIR / "models" / "posenet.safetensors"
SEGNET_PATH = UPSTREAM_DIR / "models" / "segnet.safetensors"


pytestmark = pytest.mark.skipif(
    not POSENET_PATH.exists() or not SEGNET_PATH.exists(),
    reason="upstream scorer weights not available locally",
)


@pytest.fixture(scope="module")
def cached_checkpoint_and_frames(tmp_path_factory) -> tuple[Path, Path]:
    """Produce a tiny renderer checkpoint + frame cache for the test surface.

    Generates the artifacts once per module (slow when not cached upstream of
    the test session) and reuses across the 5 tests below.
    """
    tmp = tmp_path_factory.mktemp("mps_gap_posenet_fix")
    torch.manual_seed(42)
    model = build_tiny_renderer(seed=42)
    ckpt_path = tmp / "checkpoint_ema.pt"
    torch.save(model.state_dict(), ckpt_path)

    # 2-pair synthetic batch keeps the test cheap; the canonical helper does
    # not require real video frames for the shape-adapter regression check.
    frame_cache = torch.rand(2, 2, 3, 384, 512)
    cache_path = tmp / "frame_cache.pt"
    torch.save(frame_cache, cache_path)
    return ckpt_path, cache_path


def test_posenet_mean_output_is_finite_after_shape_fix_cpu(cached_checkpoint_and_frames):
    """POST-FIX: posenet_mean_output is a finite real number (not NaN).

    Regression for the predecessor NaN bug: the 5-D reconstruction shape
    `(B, 2, 3, 384, 512)` previously raised inside PoseNet.vision and the
    diagnostic try/except returned NaN. The fix routes through
    `posenet.preprocess_input` and `pose_out["pose"]` per Catalog #164.
    """
    ckpt_path, cache_path = cached_checkpoint_and_frames
    _, components = _eval_on_device(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        device="cpu",
        include_scorer_components=True,
        upstream_dir=UPSTREAM_DIR,
    )
    assert "posenet_mean_output" in components
    pose_val = components["posenet_mean_output"]
    assert math.isfinite(pose_val), (
        f"PoseNet mean output should be finite post-fix; got {pose_val!r}"
    )


def test_segnet_mean_output_still_finite_after_preprocess_routing(cached_checkpoint_and_frames):
    """SegNet path still produces a finite value after preprocess routing.

    The original code called `segnet(reconstruction[:, -1, ...])` which
    accidentally worked because SegNet's preprocess does
    `x = x[:, -1, ...]` then `interpolate`. The new code routes through
    `segnet.preprocess_input` for canonical consistency with the PoseNet
    fix. Both paths produce a finite mean; this test pins the post-fix
    behavior so a future refactor cannot regress SegNet by accident.
    """
    ckpt_path, cache_path = cached_checkpoint_and_frames
    _, components = _eval_on_device(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        device="cpu",
        include_scorer_components=True,
        upstream_dir=UPSTREAM_DIR,
    )
    seg_val = components["segnet_mean_output"]
    assert math.isfinite(seg_val), (
        f"SegNet mean output should be finite post-fix; got {seg_val!r}"
    )


def test_compute_local_mps_reference_components_emits_finite_posenet(
    cached_checkpoint_and_frames, tmp_path
):
    """End-to-end: canonical helper writes JSON with finite posenet field.

    The CLI surface (`tac.mps_gap_experiment.harvest_and_verdict_cli reference
    --include-scorer`) wires through this helper; the JSON it persists is the
    artifact that the diff stage consumes to compute the aggregate verdict.
    Pre-fix the persisted JSON carried `posenet_mean_output: NaN` which
    poisoned the aggregate via `classify_verdict` NaN-fallback semantics.
    """
    import json

    ckpt_path, cache_path = cached_checkpoint_and_frames
    out_dir = tmp_path / "ref_out"
    components_path = compute_local_mps_reference_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=out_dir,
        device="cpu",
        include_scorer_components=True,
        upstream_dir=UPSTREAM_DIR,
    )
    data = json.loads(Path(components_path).read_text())
    pose_val = data["components"]["posenet_mean_output"]
    assert isinstance(pose_val, float)
    assert math.isfinite(pose_val), (
        f"persisted posenet_mean_output should be finite; got {pose_val!r}"
    )


def test_posenet_preprocess_returns_canonical_12_channel_shape(cached_checkpoint_and_frames):
    """Sanity: PoseNet.preprocess_input produces the documented (B, 12, H/2, W/2) shape.

    Pins the upstream contract the fix relies on (Catalog #229 premise
    verification). If a future upstream snapshot changes the preprocess
    contract this test fails-loud BEFORE the diagnostic surface regresses
    to NaN silently.
    """
    from tac.scorer import load_default_scorers

    posenet, _segnet = load_default_scorers(upstream_dir=UPSTREAM_DIR, device="cpu")
    ckpt_path, cache_path = cached_checkpoint_and_frames
    frame_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    pose_in = posenet.preprocess_input(frame_cache)
    # canonical IN_CHANS=12 (6 YUV per frame * 2 frames) per upstream/modules.py
    assert pose_in.shape[0] == frame_cache.shape[0], "batch dim preserved"
    assert pose_in.shape[1] == 12, (
        f"preprocess_input should produce 12-channel YUV6; got {tuple(pose_in.shape)}"
    )
    # SegNet model input size is (W, H) per upstream/frame_utils.py; the
    # bilinear interp target is `(segnet_model_input_size[1], segnet_model_input_size[0])`
    # which is (H, W). The contest canonical pose-input is 192x256.
    assert pose_in.shape[2:] == (192, 256), (
        f"preprocess_input should produce (192, 256) spatial; got {tuple(pose_in.shape)}"
    )


def test_posenet_forward_returns_pose_head_dict(cached_checkpoint_and_frames):
    """Sanity: PoseNet forward returns a dict with 'pose' key after preprocess.

    Pins the Hydra-head contract the fix relies on (upstream/modules.py:79
    `ret = self.hydra(summary)` returns `{name: tensor}` per Hydra.forward).
    The fix extracts `pose_out["pose"]` rather than `pose_out` directly.
    """
    from tac.scorer import load_default_scorers

    posenet, _segnet = load_default_scorers(upstream_dir=UPSTREAM_DIR, device="cpu")
    ckpt_path, cache_path = cached_checkpoint_and_frames
    frame_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        pose_in = posenet.preprocess_input(frame_cache)
        pose_out = posenet(pose_in)
    assert isinstance(pose_out, dict), (
        f"PoseNet forward should return a dict head; got {type(pose_out).__name__}"
    )
    assert "pose" in pose_out, f"PoseNet output dict should contain 'pose'; got keys {list(pose_out.keys())}"
    assert torch.isfinite(pose_out["pose"]).all(), (
        "PoseNet pose head should be all finite values"
    )
