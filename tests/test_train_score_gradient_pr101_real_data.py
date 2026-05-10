from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    path = REPO_ROOT / "experiments/train_score_gradient_pr101_finetune.py"
    name = "train_score_gradient_pr101_finetune_real_data_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeFrame:
    def __init__(self, value: int):
        self.value = value

    def to_ndarray(self, *, format: str):
        raise AssertionError(
            f"load_real_frame_pairs must use upstream yuv420_to_rgb, not {format}"
        )


class _FakeContainer:
    def __init__(self, frames: list[_FakeFrame]):
        self._frames = frames
        self.streams = types.SimpleNamespace(video=[object()])
        self.closed = False

    def decode(self, _stream):
        yield from self._frames

    def close(self):
        self.closed = True


def test_load_real_frame_pairs_uses_upstream_yuv420_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_train_module()
    frames = [_FakeFrame(10), _FakeFrame(20), _FakeFrame(30)]
    container = _FakeContainer(frames)
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _path: container))
    calls: list[int] = []

    def fake_yuv420_to_rgb(frame: _FakeFrame) -> torch.Tensor:
        calls.append(frame.value)
        return torch.full((4, 6, 3), frame.value, dtype=torch.uint8)

    monkeypatch.setattr(mod, "_load_upstream_yuv420_to_rgb", lambda: fake_yuv420_to_rgb)
    video = tmp_path / "0.mkv"
    video.write_bytes(b"fake-video")

    pairs = mod.load_real_frame_pairs(video, frame_h=2, frame_w=3)

    assert calls == [10, 20, 30]
    assert container.closed is True
    assert pairs.shape == (1, 2, 2, 3, 3)
    assert torch.all(pairs[0, 0] == 10)
    assert torch.all(pairs[0, 1] == 20)


def test_load_real_frame_pairs_honors_max_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_train_module()
    frames = [_FakeFrame(1), _FakeFrame(2), _FakeFrame(3)]
    monkeypatch.setitem(
        sys.modules,
        "av",
        types.SimpleNamespace(open=lambda _path: _FakeContainer(frames)),
    )
    seen: list[int] = []

    def fake_yuv420_to_rgb(frame: _FakeFrame) -> torch.Tensor:
        seen.append(frame.value)
        return torch.full((2, 2, 3), frame.value, dtype=torch.uint8)

    monkeypatch.setattr(mod, "_load_upstream_yuv420_to_rgb", lambda: fake_yuv420_to_rgb)
    video = tmp_path / "0.mkv"
    video.write_bytes(b"fake-video")

    pairs = mod.load_real_frame_pairs(video, frame_h=2, frame_w=2, max_frames=2)

    assert seen == [1, 2]
    assert pairs.shape == (1, 2, 2, 2, 3)


def test_train_requires_explicit_batch_source() -> None:
    mod = _load_train_module()

    with pytest.raises(ValueError, match="explicit batch_source"):
        mod.train(
            decoder=object(),
            posenet=object(),
            segnet=object(),
            epochs=1,
            steps_per_epoch=1,
            batch_size=1,
            latent_dim=1,
            frame_h=1,
            frame_w=1,
            lr=1e-4,
            device=torch.device("cpu"),
            output_dir=Path("."),
            aux_kl_weight=1.0,
            aux_pixel_l1_weight=0.0,
            batch_source=None,
        )


def test_train_saves_best_proxy_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_train_module()

    class TinyDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2 * 3 * 8 * 8)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.proj(z)).reshape(z.shape[0], 2, 3, 8, 8) * 255.0

    monkeypatch.setattr(mod, "simulate_eval_roundtrip", lambda x, *, noise_std: x)
    decoder = TinyDecoder()
    posenet, segnet = mod.load_smoke_scorers(torch.device("cpu"))

    def batch_source(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros((batch_size, 2), dtype=torch.float32)
        gt = torch.full((batch_size, 2, 8, 8, 3), 127.0, dtype=torch.float32)
        return z, gt

    result = mod.train(
        decoder=decoder,
        posenet=posenet,
        segnet=segnet,
        epochs=1,
        steps_per_epoch=1,
        batch_size=1,
        latent_dim=2,
        frame_h=8,
        frame_w=8,
        lr=1e-4,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        aux_kl_weight=0.0,
        aux_pixel_l1_weight=0.0,
        batch_source=batch_source,
    )

    assert (tmp_path / mod.FINAL_EMA_CHECKPOINT).is_file()
    assert (tmp_path / mod.BEST_PROXY_CHECKPOINT).is_file()
    log_row = json.loads((tmp_path / "train_log.jsonl").read_text().splitlines()[0])
    assert log_row["step_metrics"][0]["decoder_grad_l2"] > 0.0
    manifest = json.loads((tmp_path / mod.BEST_PROXY_MANIFEST).read_text())
    assert manifest["selection_objective"] == "min_epoch_last_step_weighted_proxy"
    assert manifest["selected_epoch"] == 0
    assert result.best_proxy_epoch == 0
    assert result.best_proxy_value is not None


def test_score_gradient_reachability_requires_primary_scorer_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_train_module()

    class TinyDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2 * 3 * 4 * 4)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.proj(z)).reshape(z.shape[0], 2, 3, 4, 4) * 255.0

    def dead_primary_loss(pred, gt, *_args, **_kwargs):
        zero = pred.sum() * 0.0
        return zero, torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)

    monkeypatch.setattr(mod, "simulate_eval_roundtrip", lambda x, *, noise_std: x)
    monkeypatch.setattr(mod, "scorer_loss_terms_btchw", dead_primary_loss)

    decoder = TinyDecoder()
    posenet, segnet = mod.load_smoke_scorers(torch.device("cpu"))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
    ema = mod.EMA(decoder, decay=0.9)
    z_batch = torch.zeros((1, 2), dtype=torch.float32)
    gt_pair = torch.full((1, 2, 4, 4, 3), 127.0, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="primary_scorer reachability failed"):
        mod.train_one_step(
            decoder=decoder,
            posenet=posenet,
            segnet=segnet,
            z_batch=z_batch,
            gt_pair_hwc=gt_pair,
            optimizer=optimizer,
            ema=ema,
            lagrangian=mod.LagrangianState(),
            aux_kl_weight=0.0,
            aux_pixel_l1_weight=0.01,
            enable_eval_roundtrip_in_training=True,
            segmentation_surrogate=mod.DEFAULT_SEGMENTATION_SURROGATE,
            segmentation_temperature=mod.DEFAULT_SEGMENTATION_TEMPERATURE,
            fisher_rao_eps=mod.DEFAULT_FISHER_RAO_EPS,
            grad_clip_norm=1.0,
            require_score_gradient_reachability=True,
        )


def test_build_selected_archives_invokes_pr101_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_train_module()
    pr101_archive = tmp_path / "source_archive.zip"
    pr101_archive.write_bytes(b"archive")
    pr101_source_dir = tmp_path / "pr101_src"
    pr101_source_dir.mkdir()
    (tmp_path / mod.BEST_PROXY_CHECKPOINT).write_bytes(b"best")
    (tmp_path / mod.FINAL_EMA_CHECKPOINT).write_bytes(b"ema")

    from tools import build_pr101_finetuned_archive

    calls: list[list[str]] = []

    def fake_builder(argv: list[str]) -> int:
        calls.append(argv)
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True)
        (out_dir / "archive.zip").write_bytes(b"candidate")
        (out_dir / "build_manifest.json").write_text(
            json.dumps(
                {
                    "archive_bytes": 9,
                    "archive_sha256": "0" * 64,
                    "source_archive_sha256": "1" * 64,
                    "score_affecting_payload_changed": True,
                    "no_op_detector": {
                        "score_affecting_payload_changed": True,
                        "decoder_payload_changed": True,
                    },
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
        )
        return 0

    monkeypatch.setattr(build_pr101_finetuned_archive, "main", fake_builder)

    payload = mod.build_selected_archives(
        output_dir=tmp_path,
        pr101_archive=pr101_archive,
        pr101_source_dir=pr101_source_dir,
        checkpoints={
            "best_proxy": mod.BEST_PROXY_CHECKPOINT,
            "final_ema": mod.FINAL_EMA_CHECKPOINT,
        },
    )

    assert len(calls) == 2
    assert [row["label"] for row in payload["builds"]] == ["best_proxy", "final_ema"]
    assert all(row["status"] == "built" for row in payload["builds"])
    assert all(row["score_claim"] is False for row in payload["builds"])
    assert all(row["score_affecting_payload_changed"] is True for row in payload["builds"])
    assert (tmp_path / mod.ARCHIVE_BUILD_MANIFEST).is_file()


def test_non_smoke_requires_pr101_source_dir_for_archive_closure(tmp_path: Path) -> None:
    mod = _load_train_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    args = types.SimpleNamespace(
        enable_eval_roundtrip_in_training=True,
        enable_differentiable_yuv6=True,
        pr101_archive=archive,
        pr101_source_dir=None,
    )

    with pytest.raises(RuntimeError, match="requires --pr101-source-dir"):
        mod.validate_non_smoke_archive_closure_args(args)


def test_build_selected_archives_fails_closed_on_missing_checkpoint(
    tmp_path: Path,
) -> None:
    mod = _load_train_module()
    pr101_archive = tmp_path / "source_archive.zip"
    pr101_archive.write_bytes(b"archive")
    pr101_source_dir = tmp_path / "pr101_src"
    pr101_source_dir.mkdir()

    with pytest.raises(RuntimeError, match="selected checkpoint missing"):
        mod.build_selected_archives(
            output_dir=tmp_path,
            pr101_archive=pr101_archive,
            pr101_source_dir=pr101_source_dir,
            checkpoints={"best_proxy": mod.BEST_PROXY_CHECKPOINT},
        )


def test_real_pair_batch_source_uses_archive_latent_rows() -> None:
    mod = _load_train_module()
    frame_pairs = torch.stack(
        [
            torch.full((2, 2, 2, 3), float(idx), dtype=torch.float32)
            for idx in range(6)
        ],
        dim=0,
    )
    latents = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    source = mod.RealPairBatchSource(
        frame_pairs=frame_pairs,
        latents=latents,
        device=torch.device("cpu"),
        seed=123,
    )

    z, gt = source.next_batch(5)

    assert z.shape == (5, 2)
    for row in range(z.shape[0]):
        idx = int(z[row, 0].item() // 2)
        assert torch.equal(z[row], latents[idx])
        assert torch.all(gt[row] == float(idx))


def test_real_pair_batch_source_rejects_missing_latent_rows() -> None:
    mod = _load_train_module()
    frame_pairs = torch.zeros((4, 2, 2, 2, 3), dtype=torch.float32)
    latents = torch.zeros((3, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="latents row count"):
        mod.RealPairBatchSource(
            frame_pairs=frame_pairs,
            latents=latents,
            device=torch.device("cpu"),
            seed=1,
        )
