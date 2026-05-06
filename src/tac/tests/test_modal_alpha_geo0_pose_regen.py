"""Static and behavior checks for the Modal Alpha-Geo-0 pose-regeneration path."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "experiments" / "modal_alpha_geo0_pose_regen.py"
LOCAL_TOOL_PATH = REPO_ROOT / "experiments" / "alpha_geo0_pose_regen.py"


class _FakeImage:
    @classmethod
    def debian_slim(cls, **_kwargs):
        return cls()

    def apt_install(self, *_args):
        return self

    def pip_install(self, *_args, **_kwargs):
        return self

    def run_commands(self, *_args):
        return self

    def add_local_dir(self, *_args, **_kwargs):
        return self

    def add_local_file(self, *_args, **_kwargs):
        return self


class _FakeWrappedFunction:
    def __init__(self, fn):
        self._fn = fn

    def get_raw_f(self):
        return self._fn

    def spawn(self, *_args, **_kwargs):
        return types.SimpleNamespace(object_id="fc-test")

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeApp:
    def __init__(self, name):
        self.name = name

    def function(self, **_kwargs):
        def decorator(fn):
            return _FakeWrappedFunction(fn)

        return decorator

    def local_entrypoint(self):
        def decorator(fn):
            return fn

        return decorator


def _load_module(monkeypatch: pytest.MonkeyPatch):
    fake_modal = types.SimpleNamespace(
        App=_FakeApp,
        Image=_FakeImage,
        FunctionCall=types.SimpleNamespace(from_id=lambda _call_id: None),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    spec = importlib.util.spec_from_file_location("modal_alpha_geo0_pose_regen_mod", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modal_alpha_geo0_pose_regen_mod"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_source_is_alpha_geo0_not_retraining() -> None:
    text = TOOL_PATH.read_text()

    assert "lane_12_alpha_geo0_pose_regen" in text
    assert "no NeRV retraining" in text
    assert "experiments/train_nerv_mask.py" not in text
    assert "gt_masks_source=segnet" not in text
    assert '"masks.nrv"' in text
    assert '"optimized_poses.bin"' in text


def test_source_runs_pose_regen_against_materialized_candidate_masks() -> None:
    text = TOOL_PATH.read_text()

    assert "candidate_masks.pt" in text
    assert "_materialize_candidate_masks_from_cache" in text
    assert "torch.save(masks.detach().cpu().contiguous(), output_path)" in text
    assert "experiments/optimize_poses.py" in text
    assert '"--threshold-preset",\n            "promotion"' in text
    assert '"--masks"' in text
    assert '"--gt-pose-targets"' in text
    assert '"--device"' in text
    assert '"cuda"' in text


def test_local_alpha_geo0_runner_uses_promotion_geometry_thresholds() -> None:
    text = LOCAL_TOOL_PATH.read_text()

    assert '"--threshold-preset",\n                "promotion"' in text


def test_source_uses_modal_t4_cuda_auth_eval_and_adjudication() -> None:
    text = TOOL_PATH.read_text()

    assert 'modal.App(APP_NAME)' in text
    assert 'APP_NAME = "comma-alpha-geo0-pose-regen"' in text
    assert '@app.function(image=run_image, gpu="T4"' in text
    assert "run_alpha_geo0_pose_regen_t4.spawn" in text
    assert "modal.FunctionCall.from_id" in text
    assert "experiments/contest_auth_eval.py" in text
    assert '"--device",\n            "cuda"' in text
    assert '"--device", "cpu"' not in text
    assert "scripts/adjudicate_contest_auth_eval.py" in text
    assert '"--required-device",\n            "cuda"' in text
    assert '"--required-samples",\n            str(REQUIRED_SAMPLES)' in text
    assert '"--allow-component-gate-forensic-success"' in text


def test_source_stages_narrow_submission_files_not_whole_eval_runs() -> None:
    text = TOOL_PATH.read_text()

    assert '.add_local_dir("submissions/robust_current"' not in text
    assert "submissions/robust_current/inflate.sh" in text
    assert "submissions/robust_current/config.env" in text
    assert "submissions/robust_current/inflate_renderer.py" in text
    assert "submissions/robust_current/eval_runs" not in text


def test_validate_contest_result_rejects_cpu_and_non_t4(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module(monkeypatch)
    archive_sha = "a" * 64
    payload = {
        "archive_size_bytes": 123,
        "n_samples": 600,
        "score_recomputed_from_components": 1.0,
        "provenance": {
            "device": "cpu",
            "cuda_available": False,
            "gpu_t4_match": False,
            "archive_sha256": archive_sha,
        },
    }

    errors = mod._validate_contest_result(
        payload,
        expected_archive_sha256=archive_sha,
        expected_archive_size_bytes=123,
    )

    assert any("expected 'cuda'" in error for error in errors)
    assert any("cuda_available" in error for error in errors)
    assert any("gpu_t4_match" in error for error in errors)


def test_validate_contest_result_accepts_cuda_t4_custody(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module(monkeypatch)
    archive_sha = "b" * 64
    payload = {
        "archive_size_bytes": 456,
        "n_samples": 600,
        "score_recomputed_from_components": 0.99,
        "provenance": {
            "device": "cuda",
            "cuda_available": True,
            "gpu_t4_match": True,
            "archive_sha256": archive_sha,
        },
    }

    assert mod._validate_contest_result(
        payload,
        expected_archive_sha256=archive_sha,
        expected_archive_size_bytes=456,
    ) == []


def test_archive_requirement_validation_rejects_unsafe_member(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_module(monkeypatch)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../renderer.bin", b"bad")
        zf.writestr("masks.nrv", b"NRV1")

    with pytest.raises(ValueError, match="unsafe archive member"):
        mod._validate_archive_requirements(archive, required_members={"renderer.bin", "masks.nrv"})


def test_archive_requirement_validation_rejects_duplicate_member(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_module(monkeypatch)
    archive = tmp_path / "dup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("renderer.bin", b"one")
        with pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr("renderer.bin", b"two")
        zf.writestr("masks.nrv", b"NRV1")

    with pytest.raises(ValueError, match="duplicate archive member"):
        mod._validate_archive_requirements(archive, required_members={"renderer.bin", "masks.nrv"})


def test_materialize_candidate_masks_unwraps_cache_dict_for_optimize_poses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_module(monkeypatch)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    masks = torch.tensor([[[0, 1], [2, 3]], [[4, 0], [1, 2]]], dtype=torch.uint8)
    decoded_sha = mod._mask_tensor_sha256(masks)
    torch.save({"masks": masks}, cache_dir / "candidate.pt")
    (cache_dir / "candidate.json").write_text(
        json.dumps(
            {
                "fingerprint": {
                    "source_sha256": "c" * 64,
                    "archive_member_resolved": "masks.nrv",
                },
                "tensor_file": "candidate.pt",
                "decoded_mask_sha256": decoded_sha,
            }
        )
    )

    out = tmp_path / "candidate_masks.pt"
    meta = mod._materialize_candidate_masks_from_cache(
        cache_dir,
        candidate_archive_sha256="c" * 64,
        output_path=out,
    )

    loaded = torch.load(out, map_location="cpu", weights_only=True)
    assert isinstance(loaded, torch.Tensor)
    assert torch.equal(loaded, masks)
    assert meta["materialized_format"] == "torch.Tensor"
    assert meta["decoded_mask_sha256"] == decoded_sha
