from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as retained
from experiments import ddm_pz4r_full_n600_eval as pz4r


def test_dataset_uses_av_for_cpu_gt_and_tensor_for_candidate(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, dict]] = []

    class FakeDataset:
        label = "fake"

        def __init__(self, *args, **kwargs):
            calls.append((self.label, kwargs))

        def prepare_data(self):
            calls.append((self.label + "_prepared", {}))

    class AV(FakeDataset):
        label = "av"

    class Dali(FakeDataset):
        label = "dali"

    class Tensor(FakeDataset):
        label = "tensor"

    fake = types.ModuleType("frame_utils")
    fake.AVVideoDataset = AV
    fake.DaliVideoDataset = Dali
    fake.TensorVideoDataset = Tensor
    monkeypatch.setitem(sys.modules, "frame_utils", fake)
    monkeypatch.setattr(retained, "UPSTREAM", tmp_path)
    (tmp_path / "public_test_video_names.txt").write_text("0.mkv\n")
    cpu = types.SimpleNamespace(type="cpu")
    retained._dataset("gt", None, cpu)
    retained._dataset("candidate", tmp_path, cpu)
    assert [label for label, _ in calls] == ["av", "av_prepared", "tensor", "tensor_prepared"]


def test_score_argmax_field_runs_on_cpu_without_cuda_sync(monkeypatch, tmp_path: Path):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(retained, "N_PAIRS", 1)
    monkeypatch.setattr(retained, "BATCH_SIZE", 1)
    monkeypatch.setattr(retained, "SEG_HEIGHT", 1)
    monkeypatch.setattr(retained, "SEG_WIDTH", 1)
    monkeypatch.setattr(
        retained,
        "_dataset",
        lambda *_args, **_kwargs: [("candidate", 0, torch.zeros((1, 2, 1, 1, 3)))],
    )

    class Scorer:
        def preprocess_input(self, value):
            return value[:, -1]

        def __call__(self, value):
            logits = torch.zeros((1, 5, 1, 1))
            logits[:, 3] = 1
            return logits

    raw = tmp_path / "0.raw"
    raw.write_bytes(b"x")
    result = retained.score_argmax_field(
        source="candidate",
        raw_root=tmp_path,
        raw_record=retained.file_record(raw),
        scorer=Scorer(),
        device=torch.device("cpu"),
        run_root=tmp_path / "run",
    )
    field = np.load(result["argmax"]["path"], allow_pickle=False)
    assert field.tolist() == [[[3]]]
    assert result["axis"].startswith("[macOS-CPU advisory")


def test_score_delta_is_sum_of_all_three_terms():
    base = pz4r.score(0.001, 1e-5, 186_252)
    candidate = pz4r.score(0.0009, 2e-5, 183_137)
    delta = pz4r.score_delta(base, candidate)
    assert delta["total"] == pytest.approx(delta["seg"] + delta["pose"] + delta["rate"])
    assert delta["rate"] == pytest.approx(25 * (183_137 - 186_252) / 37_545_489)
