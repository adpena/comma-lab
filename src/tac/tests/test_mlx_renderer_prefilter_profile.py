# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tac.local_acceleration import mlx_renderer_prefilter_profile as profile_mod
from tac.local_acceleration import mlx_scorer_adapters


def test_gpu_prefilter_loads_torch_scorer_on_cpu_but_records_gpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []

    @contextmanager
    def fake_temporary_mlx_device(device_type: str) -> Iterator[None]:
        calls.append(("temporary_mlx_device", device_type))
        yield

    def fake_load_adapter(upstream_dir: str | Path, *, device: str) -> str:
        calls.append(("torch_load_device", device))
        return "adapter"

    def fake_build_loaded(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["adapter"] == "adapter"
        return {
            "schema": profile_mod.HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
            "scorer_device": kwargs["scorer_device"],
            "scorer_batch_pairs": kwargs["scorer_batch_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        mlx_scorer_adapters,
        "temporary_mlx_device",
        fake_temporary_mlx_device,
    )
    monkeypatch.setattr(
        mlx_scorer_adapters,
        "load_mlx_distortion_scorer_adapter_from_upstream",
        fake_load_adapter,
    )
    monkeypatch.setattr(
        profile_mod,
        "build_mlx_renderer_prefilter_profile_loaded",
        fake_build_loaded,
    )

    out = profile_mod.write_mlx_renderer_prefilter_profile(
        bundle=object(),
        output_path=tmp_path / "prefilter.json",
        archive_bytes=123,
        archive_sha256="a" * 64,
        upstream_dir=tmp_path / "upstream",
        scorer_device="gpu",
        scorer_batch_pairs=8,
    )

    assert calls == [
        ("temporary_mlx_device", "gpu"),
        ("torch_load_device", "cpu"),
    ]
    assert out["scorer_device"] == "gpu"
    assert out["scorer_batch_pairs"] == 8

