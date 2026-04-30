"""Lane 12 NeRV clean-inflate dependency closure checks.

These tests cover the narrow boundary between the `.nrv` mask payload and the
contest inflate/package path.  They intentionally avoid editing the shared
renderer while proving what is already wired and what still blocks promotion.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import torch

from tac.nerv_mask_codec import NeRVMaskCodec, encode_nerv_codec
from tac.submission_archive import (
    RENDERER_NRV_MANIFEST,
    build_submission_archive,
    detect_pose_manifest,
    validate_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_nerv_mask_codec_top_level_import_closure_is_clean() -> None:
    """Inflate-time codec import needs only stdlib, numpy, and torch."""
    path = REPO_ROOT / "src" / "tac" / "nerv_mask_codec.py"
    tree = ast.parse(path.read_text())

    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                imports.add("<relative>")
            else:
                imports.add((node.module or "").split(".")[0])

    assert imports <= {"__future__", "dataclasses", "io", "numpy", "struct", "torch"}
    assert "tac" not in imports, (
        "tac.nerv_mask_codec must not import the wider tac stack at module import; "
        "inflate only needs decode_nerv_codec/render_mask_argmax."
    )


def _load_inflate_renderer_module():
    path = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
    spec = importlib.util.spec_from_file_location("_lane12_inflate_renderer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_contest_auth_eval_module():
    path = REPO_ROOT / "experiments" / "contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location("_lane12_contest_auth_eval", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_train_nerv_mask_module():
    path = REPO_ROOT / "experiments" / "train_nerv_mask.py"
    spec = importlib.util.spec_from_file_location("_lane12_train_nerv_mask", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_inflate_renderer_decodes_tiny_nrv_payload(tmp_path: Path) -> None:
    """`masks.nrv` can inflate through the renderer helper with only tac+torch."""
    codec = NeRVMaskCodec(num_freqs=1, hidden_dim=4, num_classes=5, depth=2)
    nrv_path = tmp_path / "masks.nrv"
    nrv_path.write_bytes(encode_nerv_codec(codec, weight_dtype="fp16"))

    inflate_mod = _load_inflate_renderer_module()
    masks = inflate_mod._load_masks_from_nrv(
        nrv_path,
        expected_frames=2,
        height=3,
        width=4,
    )

    assert isinstance(masks, torch.Tensor)
    assert masks.shape == (2, 3, 4)
    assert masks.dtype == torch.long
    assert int(masks.min()) >= 0
    assert int(masks.max()) < 5


def test_train_nerv_mask_segnet_source_uses_preprocess_input(
    monkeypatch, tmp_path: Path
) -> None:
    """Compress-time mask extraction must use SegNet's scorer preprocessing."""
    train_mod = _load_train_nerv_mask_module()

    class _FakeSegNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.preprocess_calls = 0

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            assert x.ndim == 5
            assert x.shape[1] == 1
            assert x.shape[2] == 3
            self.preprocess_calls += 1
            return x[:, -1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.ndim == 4
            b, _c, h, w = x.shape
            logits = torch.zeros(b, 5, h, w, device=x.device)
            logits[:, 2] = 1.0
            return logits

    fake_segnet = _FakeSegNet()
    fake_scorer = types.ModuleType("tac.scorer")
    fake_scorer.load_differentiable_scorers = lambda upstream_dir, device: (
        object(),
        fake_segnet,
    )
    monkeypatch.setitem(sys.modules, "tac.scorer", fake_scorer)

    video = tmp_path / "videos" / "0.mkv"
    video.parent.mkdir()
    video.write_bytes(b"stub")

    def fake_run(cmd, **_kwargs):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="8,4\n", stderr="")
        if cmd[0] == "ffmpeg":
            frames = np.zeros((2, 4, 8, 3), dtype=np.uint8)
            return subprocess.CompletedProcess(cmd, 0, stdout=frames.tobytes(), stderr=b"")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    masks = train_mod._load_segnet_argmax_masks(tmp_path, device="cpu", num_frames=2)

    assert tuple(masks.shape) == (2, 384, 512)
    assert fake_segnet.preprocess_calls == 1
    assert set(masks.unique().tolist()) == {2}


def test_canonical_archive_validation_accepts_masks_nrv(tmp_path: Path) -> None:
    """Canonical manifest detection now recognizes Lane 12 masks.nrv."""
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"R" * 10_001)
        zf.writestr("masks.nrv", b"NRV1unit-test-payload")
        zf.writestr("optimized_poses.pt", b"P" * 2_000)

    manifest = detect_pose_manifest(archive)
    result = validate_archive(archive, manifest=manifest, strict=True)

    assert result.valid, result.summary()
    assert result.files_missing == []
    assert result.files_unexpected == []


def test_contest_auth_eval_allows_nrv_archive_member() -> None:
    """The exact archive eval harness does not reject Lane 12 .nrv payloads."""
    contest_auth_eval = _load_contest_auth_eval_module()

    contest_auth_eval._validate_archive_members(
        ["renderer.bin", "masks.nrv", "optimized_poses.pt"]
    )


def test_build_submission_archive_supports_masks_nrv_manifest(tmp_path: Path) -> None:
    renderer = tmp_path / "renderer.bin"
    masks = tmp_path / "masks.nrv"
    poses = tmp_path / "optimized_poses.pt"
    archive = tmp_path / "archive.zip"
    renderer.write_bytes(b"R" * 10_001)
    masks.write_bytes(b"NRV1unit-test-payload")
    poses.write_bytes(b"P" * 2_000)

    result = build_submission_archive(
        archive,
        renderer_bin=renderer,
        masks_nrv=masks,
        optimized_poses_pt=poses,
        manifest=RENDERER_NRV_MANIFEST,
        validate=True,
    )

    assert result.valid, result.summary()
    with zipfile.ZipFile(archive, "r") as zf:
        assert zf.namelist() == ["renderer.bin", "masks.nrv", "optimized_poses.pt"]
        assert zf.getinfo("masks.nrv").date_time == (1980, 1, 1, 0, 0, 0)


def test_remote_lane_nerv_runs_exact_cuda_archive_eval() -> None:
    """Remote Lane 12 uses archive.zip -> inflate.sh -> upstream/evaluate.py."""
    script = (REPO_ROOT / "scripts" / "remote_lane_nerv.sh").read_text()

    assert '"$PYBIN" -m pip install -e .' in script
    assert "experiments/contest_auth_eval.py" in script
    assert "scripts/adjudicate_contest_auth_eval.py" in script
    assert '--archive "$ARCHIVE"' in script
    assert "--inflate-sh submissions/robust_current/inflate.sh" in script
    assert '--device "${AUTH_EVAL_DEVICE:-cuda}"' in script
    assert "--keep-work-dir" in script
    assert '--work-dir "$EVAL_WORK_DIR"' in script
    assert 'CONTEST_JSON="$EVAL_WORK_DIR/contest_auth_eval.json"' in script
    assert '--contest-json "$CONTEST_JSON"' in script
    assert '--result-copy "$RESULT_JSON"' in script
    assert 'score_delta_vs_lane_g_v3' in script
    assert "refusing log JSON scrape" in script
    assert "auth_eval_renderer.py" not in script
