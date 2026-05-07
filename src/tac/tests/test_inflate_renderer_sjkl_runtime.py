"""Runtime tests for optional sjkl.bin robust_current integration."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
INFLATE_RENDERER_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
UNPACK_RENDERER_PAYLOAD_PATH = (
    REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _uleb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def test_qp1_pose_loader_preserves_public_float32_precision(tmp_path: Path) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_qp1_pose_test")
    q0 = 10241
    q1 = 10242
    payload = b"QP1" + q0.to_bytes(2, "little") + _uleb128(_zigzag(q1 - q0))
    pose_path = tmp_path / "optimized_poses.qp1"
    pose_path.write_bytes(payload)

    poses = inflate._decode_qp1_poses_float32(pose_path, pose_dim=6)

    assert poses.dtype == torch.float32
    assert poses.shape == (2, 6)
    assert float(poses[0, 0]) == pytest.approx(20.0 + q0 / 512.0)
    assert float(poses[1, 0]) == pytest.approx(20.0 + q1 / 512.0)
    # The old fp16 materialization boundary rounds this value by 1/64.
    assert float(poses[0, 0]) != float(poses[0, 0].half().float())
    assert torch.count_nonzero(poses[:, 1:]).item() == 0


def _build_sjkl_payload(
    *,
    h: int = 4,
    w: int = 5,
    target_h: int | None = None,
    target_w: int | None = None,
    alphas: tuple[float, ...] = (2.0, -1.0),
) -> tuple[bytes, object]:
    from experiments.build_sjkl_residual import pack_alpha_block, pack_full_sjkl_payload
    from tac.sjkl_basis import SJKLBasis, pack_sjkl_basis

    basis = SJKLBasis(
        basis_coarse=torch.ones(1, 3, h, w),
        scale=torch.tensor([1.5]),
        target_h=target_h or h,
        target_w=target_w or w,
    ).renormalize()
    basis_bytes = pack_sjkl_basis(basis)
    alpha_qs = [np.array([0], dtype=np.uint8) for _ in alphas]
    mins = [float(a) for a in alphas]
    steps = [0.0 for _ in alphas]
    block_bytes = pack_alpha_block(alpha_qs, mins, steps, alpha_bits=6)
    return pack_full_sjkl_payload(basis_bytes, block_bytes), basis


def test_sjkl_loader_and_apply_do_not_import_scorers(tmp_path: Path, monkeypatch) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_sjkl_runtime_test")
    payload, basis = _build_sjkl_payload()
    (tmp_path / "sjkl.bin").write_bytes(payload)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modules" or name.startswith("tac.scorer"):
            raise AssertionError(f"unexpected scorer import during SJ-KL runtime load: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    state = inflate._load_sjkl_residual_from_archive_dir(tmp_path)

    pairs = torch.zeros(2, 2, basis.target_h, basis.target_w, 3, dtype=torch.float32)
    out = inflate._apply_sjkl_residual_to_pairs(pairs.clone(), state, pair_start=0)

    full = basis.upsample()[0].permute(1, 2, 0).contiguous()
    assert torch.allclose(out[0, 0], 2.0 * basis.scale[0] * full, atol=1e-3)
    assert torch.allclose(out[1, 0], -1.0 * basis.scale[0] * full, atol=1e-3)
    assert torch.equal(out[:, 1], torch.zeros_like(out[:, 1]))
    assert state["applied_pair_count"] == 2


def test_sjkl_sparse_bitpacked_payload_applies_only_selected_pairs(tmp_path: Path) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_sjkl_sparse_test")
    from experiments.build_sjkl_residual import pack_alpha_block, pack_full_sjkl_payload
    from tac.sjkl_basis import SJKLBasis, pack_sjkl_basis

    basis = SJKLBasis(
        basis_coarse=torch.ones(1, 3, 1, 1),
        scale=torch.tensor([1.0]),
        target_h=4,
        target_w=5,
    ).renormalize()
    payload = pack_full_sjkl_payload(
        pack_sjkl_basis(basis),
        pack_alpha_block(
            [
                np.array([0], dtype=np.uint8),
                np.array([0], dtype=np.uint8),
            ],
            [3.0, 5.0],
            [0.0, 0.0],
            alpha_bits=3,
            pair_indices=[1, 4],
            sparse_bitpacked=True,
        ),
    )
    (tmp_path / "sjkl.bin").write_bytes(payload)
    state = inflate._load_sjkl_residual_from_archive_dir(tmp_path)
    pairs = torch.zeros(5, 2, basis.target_h, basis.target_w, 3, dtype=torch.float32)

    out = inflate._apply_sjkl_residual_to_pairs(pairs.clone(), state, pair_start=0)

    assert state["alpha_block_format"] == "sparse_bitpacked_v2"
    assert state["applied_pair_count"] == 2
    assert "pair_index_not_selected" in state["skip_reasons"]
    assert int(out[0, 0].sum()) == 0
    assert int(out[2, 0].sum()) == 0
    assert int(out[3, 0].sum()) == 0
    assert float(out[1, 0].sum()) > 0.0
    assert float(out[4, 0].sum()) > float(out[1, 0].sum())
    assert torch.equal(out[:, 1], torch.zeros_like(out[:, 1]))


def test_archive_mask_source_missing_mask_fails_closed(tmp_path: Path, monkeypatch) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_mask_source_guard_test")
    archive_dir = tmp_path / "archive"
    inflated_dir = tmp_path / "inflated"
    video_names = tmp_path / "video_names.txt"
    archive_dir.mkdir()
    video_names.write_text("0.mkv\n")
    monkeypatch.delenv("INFLATE_MASK_SOURCE", raising=False)

    with pytest.raises(FileNotFoundError, match="Refusing to fall back to SegNet"):
        inflate.inflate_renderer(
            str(archive_dir),
            str(inflated_dir),
            str(video_names),
        )


def test_inflate_require_cuda_fails_before_cpu_renderer_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_cuda_guard_test")
    video_names = tmp_path / "video_names.txt"
    video_names.write_text("0.mkv\n")
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    monkeypatch.setenv("INFLATE_REQUIRE_CUDA", "1")
    monkeypatch.setattr(inflate.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="INFLATE_REQUIRE_CUDA=1"):
        inflate.inflate_renderer(
            str(archive_dir),
            str(tmp_path / "inflated"),
            str(video_names),
        )


def test_sjkl_shape_mismatch_and_absent_payload_are_noops(tmp_path: Path) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_sjkl_noop_test")
    payload, _basis = _build_sjkl_payload(h=4, w=5)
    (tmp_path / "sjkl.bin").write_bytes(payload)
    state = inflate._load_sjkl_residual_from_archive_dir(tmp_path)

    pairs = torch.full((1, 2, 3, 5, 3), 7.0, dtype=torch.float32)
    assert torch.equal(
        inflate._apply_sjkl_residual_to_pairs(pairs.clone(), state, pair_start=0),
        pairs,
    )
    assert state["applied_pair_count"] == 0
    assert "target_shape_mismatch" in state["skip_reasons"]
    assert torch.equal(
        inflate._apply_sjkl_residual_to_pairs(pairs.clone(), None, pair_start=0),
        pairs,
    )


def test_sjkl_strict_contract_fails_when_charged_payload_is_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_sjkl_strict_test")
    payload, _basis = _build_sjkl_payload(h=4, w=5)
    (tmp_path / "sjkl.bin").write_bytes(payload)
    state = inflate._load_sjkl_residual_from_archive_dir(tmp_path)
    pairs = torch.full((1, 2, 3, 5, 3), 7.0, dtype=torch.float32)
    inflate._apply_sjkl_residual_to_pairs(pairs, state, pair_start=0)

    monkeypatch.setenv("SJKL_REQUIRE_APPLIED", "1")
    with pytest.raises(RuntimeError, match=r"charged sjkl\.bin did not affect any"):
        inflate._finalize_sjkl_application_contract(state)


def test_sjkl_strict_contract_passes_after_application(tmp_path: Path, monkeypatch) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_sjkl_strict_pass_test")
    payload, basis = _build_sjkl_payload()
    (tmp_path / "sjkl.bin").write_bytes(payload)
    state = inflate._load_sjkl_residual_from_archive_dir(tmp_path)
    pairs = torch.zeros(1, 2, basis.target_h, basis.target_w, 3, dtype=torch.float32)
    inflate._apply_sjkl_residual_to_pairs(pairs, state, pair_start=0)

    monkeypatch.setenv("SJKL_REQUIRE_APPLIED", "1")
    inflate._finalize_sjkl_application_contract(state)


def test_generate_and_write_applies_sjkl_only_to_joint_frame_fake1(tmp_path: Path) -> None:
    inflate = _load_module(INFLATE_RENDERER_PATH, "_inflate_renderer_sjkl_generate_test")
    h, w = 4, 5
    payload, _basis = _build_sjkl_payload(
        h=1,
        w=1,
        target_h=inflate.SEG_H,
        target_w=inflate.SEG_W,
        alphas=(1000.0,),
    )
    (tmp_path / "sjkl.bin").write_bytes(payload)
    state = inflate._load_sjkl_residual_from_archive_dir(tmp_path)

    class TinyJointFrameGenerator(nn.Module):
        q_faithful = True
        pose_dim = 0

        def forward(self, mask_t, mask_t1, **_kwargs):
            b, out_h, out_w = mask_t.shape
            return torch.zeros(b, 2, out_h, out_w, 3, device=mask_t.device)

    out_path = tmp_path / "out.raw"
    masks = torch.zeros(2, h, w, dtype=torch.long)
    n_written = inflate._generate_and_write(
        masks,
        TinyJointFrameGenerator(),
        str(out_path),
        "cpu",
        batch_size=2,
        out_h=h,
        out_w=w,
        sjkl_residual=state,
    )
    assert n_written == 2
    frames = torch.from_numpy(np.frombuffer(out_path.read_bytes(), dtype=np.uint8).copy())
    frames = frames.reshape(2, h, w, 3)
    assert int(frames[0].sum()) > 0
    assert int(frames[1].sum()) == 0


def test_renderer_payload_unpacker_allows_charged_sjkl_member() -> None:
    unpacker = _load_module(UNPACK_RENDERER_PAYLOAD_PATH, "_unpack_renderer_payload_sjkl_test")
    assert unpacker._safe_member_name("sjkl.bin") == "sjkl.bin"
