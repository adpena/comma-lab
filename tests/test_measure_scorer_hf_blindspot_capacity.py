import importlib.util
import inspect
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools/measure_scorer_hf_blindspot_capacity.py"

spec = importlib.util.spec_from_file_location("s2sbs_tool", TOOL_PATH)
s2sbs = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = s2sbs
spec.loader.exec_module(s2sbs)


def test_decode_frames_uses_upstream_cpu_frame_contract_static():
    source = inspect.getsource(s2sbs._decode_frames)

    assert "AVVideoDataset" in source
    assert "yuv420_to_rgb" in source
    assert "to_ndarray" not in source
    assert "format=\"rgb24\"" not in source


def test_hf_mask_zeros_lf_window_and_keeps_hf_corners():
    mask = s2sbs._hf_mask_hw(16, 20, (8, 10))

    assert mask[8 - 4 : 8 + 4, 10 - 5 : 10 + 5].sum().item() == 0.0
    assert mask[0, 0].item() == 1.0
    assert mask[-1, -1].item() == 1.0


def test_make_hf_perturbation_is_deterministic_and_lf_zeroed():
    pert_a = s2sbs._make_hf_perturbation((1, 16, 16), (8, 8), 1.0, seed=123)
    pert_b = s2sbs._make_hf_perturbation((1, 16, 16), (8, 8), 1.0, seed=123)

    assert torch.allclose(pert_a, pert_b)
    assert torch.isclose(pert_a.abs().max(), torch.tensor(1.0), atol=1e-6)

    shifted_fft = torch.fft.fftshift(torch.fft.fft2(pert_a[0]), dim=(-2, -1))
    lf = shifted_fft[8 - 4 : 8 + 4, 8 - 4 : 8 + 4]
    assert lf.abs().max().item() < 1e-4


def test_hf_hermitian_pairs_are_unique_symmetric_and_in_mask():
    mask = s2sbs._hf_mask_hw(16, 16, (8, 8))
    pairs = s2sbs._hf_hermitian_coordinate_pairs(mask)

    assert pairs
    seen = set()
    for coord, conj in pairs:
        assert coord != conj
        assert coord not in seen
        assert conj not in seen
        assert mask[coord[0], coord[1]].item() == 1.0
        assert mask[conj[0], conj[1]].item() == 1.0
        assert s2sbs._shifted_fft_conjugate_coord(*coord, 16, 16) == conj
        seen.add(coord)
        seen.add(conj)


def test_scorer_helpers_use_preprocess_and_first6_pose():
    class FakeSeg:
        def preprocess_input(self, x):
            assert tuple(x.shape) == (2, 2, 3, 4, 5)
            return x[:, -1]

        def __call__(self, x):
            assert tuple(x.shape) == (2, 3, 4, 5)
            logits = torch.zeros(2, 5, 4, 5)
            logits[:, 3] = 1.0
            return logits

    class FakePose:
        def preprocess_input(self, x):
            assert tuple(x.shape) == (2, 2, 3, 4, 5)
            return torch.zeros(2, 12, 2, 2)

        def __call__(self, x):
            assert tuple(x.shape) == (2, 12, 2, 2)
            return {"pose": torch.arange(24, dtype=torch.float32).reshape(2, 12)}

    pair = torch.zeros(2, 2, 4, 5, 3)

    argmax, logits = s2sbs._segnet_argmax(FakeSeg(), pair)
    pose = s2sbs._posenet_pose(FakePose(), pair)

    assert tuple(argmax.shape) == (2, 4, 5)
    assert tuple(logits.shape) == (2, 5, 4, 5)
    assert torch.equal(argmax, torch.full((2, 4, 5), 3))
    assert tuple(pose.shape) == (2, 6)
    assert torch.equal(pose[0], torch.arange(6, dtype=torch.float32))
    assert torch.equal(pose[1], torch.arange(12, 18, dtype=torch.float32))


def test_rendered_memo_keeps_advisory_and_research_only_caveats():
    res = s2sbs.AuditResult(
        timestamp_utc="2026-05-13T00:00:00Z",
        host="test",
        repo_head_sha="abc123",
        video_path="upstream/videos/0.mkv",
        video_sha256="v",
        segnet_sd_sha256="s",
        posenet_sd_sha256="p",
        n_frames_sampled=2,
        config_description={"upstream_scorer_contract": {"ok": True}},
        rows=[],
        aggregate={
            "n_pairs": 1,
            "max_bytes_per_frame_joint_advisory": 0.0,
            "max_bytes_per_frame_segnet_only_advisory": 0.0,
        },
        prbs_demo={"bit_error_rate_after_uint8_roundtrip": 0.5},
    )

    memo = s2sbs._render_memo(res, REPO_ROOT / "experiments/results/x/blindspot_capacity.json")

    assert "**[macOS-CPU advisory]**" in memo
    assert "`research_only=true`" in memo
    assert "`score_claim=false`" in memo
    assert "`ready_for_exact_eval_dispatch=false`" in memo
    assert "THEORETICAL" in memo
    assert "not byte-closed" in memo
    assert "PoseNet visibility is real" in memo
    assert "**NO-GO" in memo
