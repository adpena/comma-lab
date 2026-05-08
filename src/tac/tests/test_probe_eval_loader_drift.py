from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_tool(name: str):
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_tensors_reports_lsb_drift_and_nonzero_fraction() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    a = torch.tensor([[[[[0, 10, 20], [30, 40, 50]]]]], dtype=torch.uint8)
    b = torch.tensor([[[[[0, 11, 18], [30, 44, 49]]]]], dtype=torch.uint8)

    comparison = mod.compare_tensors(a, b)

    assert comparison["shape_match"] is True
    assert comparison["max_abs_lsb"] == 4.0
    assert comparison["nonzero_fraction"] == 4 / 6
    assert 1.91 < comparison["rms_abs_lsb"] < 1.92


def test_per_channel_compare_uses_last_rgb_axis() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    a = torch.zeros(1, 1, 1, 2, 3, dtype=torch.uint8)
    b = a.clone()
    b[..., 1] = 7

    rows = mod.per_channel_compare(a, b)

    assert [row["channel"] for row in rows] == [0, 1, 2]
    assert rows[0]["max_abs_lsb"] == 0.0
    assert rows[1]["max_abs_lsb"] == 7.0
    assert rows[2]["max_abs_lsb"] == 0.0


def test_next_batch_advances_existing_iterator() -> None:
    mod = _load_tool("probe_eval_loader_drift")

    class TinyDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield "a.mp4", 0, torch.zeros(1)
            yield "b.mp4", 1, torch.ones(1)

    iterator = iter(mod._batch_iterator(TinyDataset()))

    first = mod._next_batch(iterator)
    second = mod._next_batch(iterator)

    assert first[0] == "a.mp4"
    assert second[0] == "b.mp4"
    assert second[1] == 1


def test_next_batch_or_none_returns_none_on_exhaustion() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    iterator = iter([])

    assert mod._next_batch_or_none(iterator) is None


def test_default_data_dir_matches_upstream_evaluator() -> None:
    mod = _load_tool("probe_eval_loader_drift")
    args = mod.parse_args([])

    assert args.data_dir == mod.UPSTREAM / "videos"


def test_unavailable_probe_is_non_promotable(monkeypatch) -> None:
    mod = _load_tool("probe_eval_loader_drift")
    monkeypatch.setattr(mod, "_cuda_dali_available", lambda: (False, "no cuda fixture"))
    args = mod.parse_args(["--max-batches", "1"])

    report = mod.build_probe_report(args)

    assert report["schema"] == "eval_loader_device_drift_probe.v1"
    assert report["comparison_available"] is False
    assert report["comparison_unavailable_reason"] == "no cuda fixture"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["evidence_grade"] == "diagnostic"
    assert report["diagnostic_kind"] == "loader_drift_probe"
