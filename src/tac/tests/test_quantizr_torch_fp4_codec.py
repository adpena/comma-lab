from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import brotli
import pytest
import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import decode_qzs3_state_dict
from tac.quantizr_torch_fp4_codec import (
    decode_torch_fp4_payload,
    encode_torch_fp4_state_dict,
    is_torch_fp4_payload,
    load_torch_fp4_bytes,
)


REPO = Path(__file__).resolve().parents[3]
PR63_ARCHIVE = (
    REPO
    / "experiments/results/top_submission_current_floor_20260501/external_archives/pr63_qpose14_archive.zip"
)
PR63_UNPACKED_RUNTIME = (
    REPO
    / "experiments/results/top_submission_reverse_roundtrip_20260501/"
    / "public_pr63_unpacked_runtime_20260501T2158Z"
)


def _load_builder():
    path = REPO / "experiments/repack_quantizr_faithful_qzs3_archive.py"
    spec = importlib.util.spec_from_file_location("_qfaithful_repack_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_torch_fp4_codec_round_trips_into_jointframegenerator() -> None:
    torch.manual_seed(0)
    source = build_quantizr_faithful_renderer().eval()

    payload = encode_torch_fp4_state_dict(source)
    decoded = load_torch_fp4_bytes(payload, device="cpu")

    assert sum(p.numel() for p in decoded.parameters()) == 87836
    mask = torch.zeros(1, 384, 512, dtype=torch.long)
    pose = torch.zeros(1, 6)
    with torch.no_grad():
        f1, f2 = decoded(mask, pose)
    assert f1.shape == (1, 3, 384, 512)
    assert f2.shape == (1, 3, 384, 512)
    assert torch.isfinite(f1).all()
    assert torch.isfinite(f2).all()


def test_torch_fp4_codec_is_deterministic() -> None:
    torch.manual_seed(123)
    source = build_quantizr_faithful_renderer().eval()

    first = encode_torch_fp4_state_dict(source)
    second = encode_torch_fp4_state_dict(source)

    assert first == second


def test_torch_fp4_payload_shape_detector_and_state_decode() -> None:
    source = build_quantizr_faithful_renderer().eval()
    payload = torch.load(
        io.BytesIO(encode_torch_fp4_state_dict(source)),
        map_location="cpu",
        weights_only=False,
    )

    assert is_torch_fp4_payload(payload)
    state = decode_torch_fp4_payload(payload, device="cpu")
    fresh = build_quantizr_faithful_renderer()
    fresh.load_state_dict(state, strict=True)


def test_loader_decodes_public_pr63_model_payload_when_available() -> None:
    if not PR63_ARCHIVE.exists():
        pytest.skip(f"public PR63 archive fixture missing: {PR63_ARCHIVE}")
    with zipfile.ZipFile(PR63_ARCHIVE) as zf:
        packed = zf.read("p")
    model_br = packed[219472 : 219472 + 66841]
    model_raw = brotli.decompress(model_br)

    model = load_torch_fp4_bytes(model_raw, device="cpu")

    assert sum(p.numel() for p in model.parameters()) == 87836
    mask = torch.zeros(1, 384, 512, dtype=torch.long)
    pose = torch.zeros(1, 6)
    with torch.no_grad():
        f1, f2 = model(mask, pose)
    assert torch.isfinite(f1).all()
    assert torch.isfinite(f2).all()


@pytest.mark.xfail(
    reason="current repack builder exposes QZS3/QZS4 only; Torch-FP4 builder promotion is not wired",
    strict=True,
)
def test_repack_builder_emits_torch_fp4_archive_from_qfai(tmp_path: Path) -> None:
    from tac.quantizr_faithful_export import save_qfai
    from tac.submission_archive import RENDERER_COMPACT_MANIFEST, build_submission_archive

    builder = _load_builder()
    model = build_quantizr_faithful_renderer().eval()
    renderer = tmp_path / "renderer.bin"
    save_qfai(model, renderer)
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"mask" * 1024)
    poses = tmp_path / "optimized_poses.bin"
    poses.write_bytes(torch.zeros(600, 6, dtype=torch.float16).numpy().tobytes())
    source = tmp_path / "source.zip"
    build_submission_archive(
        source,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        manifest=RENDERER_COMPACT_MANIFEST,
        validate=False,
    )

    out = tmp_path / "torchfp4.zip"
    provenance = builder.build_archive(
        source,
        out,
        renderer_codec=builder.RENDERER_CODEC_TORCH_FP4,
    )

    assert provenance["renderer"]["renderer_codec"] == builder.RENDERER_CODEC_TORCH_FP4
    assert provenance["renderer"]["source_renderer_format"] == "QFAI"
    with zipfile.ZipFile(out) as zf:
        raw = zf.read("renderer.bin")
    load_torch_fp4_bytes(raw, device="cpu")


def test_repack_builder_emits_qzs3_archive_from_public_pr63_torch_fp4(tmp_path: Path) -> None:
    if not PR63_UNPACKED_RUNTIME.exists():
        pytest.skip(f"public PR63 unpacked runtime fixture missing: {PR63_UNPACKED_RUNTIME}")
    builder = _load_builder()
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in ("renderer.bin", "masks.mkv", "optimized_poses.bin"):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, (PR63_UNPACKED_RUNTIME / name).read_bytes())

    out = tmp_path / "qzs3.zip"
    provenance = builder.build_archive(
        source,
        out,
        renderer_codec=builder.RENDERER_CODEC_QZS3,
    )

    assert provenance["renderer"]["renderer_codec"] == builder.RENDERER_CODEC_QZS3
    assert provenance["renderer"]["source_renderer_format"] == "Torch-FP4"
    with zipfile.ZipFile(out) as zf:
        raw = zf.read("renderer.bin")
    assert raw.startswith(b"QZS3")
    assert len(raw) == 59288
    decode_qzs3_state_dict(raw, device="cpu")
