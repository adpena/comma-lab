# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
TRACE_PATH = REPO / "experiments" / "contest_component_trace.py"


def _load_trace_module():
    spec = importlib.util.spec_from_file_location("_contest_component_trace_test", TRACE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


def _fake_ffmpeg_script(*, include_color_contract: bool) -> str:
    options = "in_range out_range in_color_matrix"
    if include_color_contract:
        options += " in_primaries in_transfer"
    return f"#!/bin/sh\nprintf '%s\\n' '{options}'\n"


def test_summarize_samples_matches_contest_formula_and_ranks() -> None:
    trace = _load_trace_module()
    samples = [
        trace.ComponentSample(0, "0.hevc", 0, 0, 0.001, 0.002),
        trace.ComponentSample(1, "0.hevc", 1, 2, 0.004, 0.001),
        trace.ComponentSample(2, "0.hevc", 2, 4, 0.002, 0.006),
    ]

    summary = trace.summarize_samples(
        samples,
        archive_size_bytes=300,
        uncompressed_size_bytes=1_000,
        top_k=2,
    )

    assert summary["score_claim"] is False
    assert summary["n_samples"] == 3
    assert summary["avg_posenet_dist"] == (0.001 + 0.004 + 0.002) / 3
    assert summary["avg_segnet_dist"] == (0.002 + 0.001 + 0.006) / 3
    expected_score = (
        100.0 * summary["avg_segnet_dist"]
        + (10.0 * summary["avg_posenet_dist"]) ** 0.5
        + 25.0 * (300 / 1_000)
    )
    assert summary["score_recomputed_from_components"] == expected_score
    assert [r["pair_index"] for r in summary["top_pose_samples"]] == [1, 2]
    assert [r["pair_index"] for r in summary["top_seg_samples"]] == [2, 0]


def test_delta_from_baseline_ranks_excess_repair_opportunity() -> None:
    trace = _load_trace_module()
    baseline = [
        trace.ComponentSample(0, "0.hevc", 0, 0, 0.001, 0.001),
        trace.ComponentSample(1, "0.hevc", 1, 2, 0.001, 0.001),
        trace.ComponentSample(2, "0.hevc", 2, 4, 0.001, 0.001),
    ]
    candidate = [
        trace.ComponentSample(0, "0.hevc", 0, 0, 0.010, 0.001),
        trace.ComponentSample(1, "0.hevc", 1, 2, 0.001, 0.020),
        trace.ComponentSample(2, "0.hevc", 2, 4, 0.002, 0.003),
    ]

    summary = trace.summarize_samples(
        candidate,
        archive_size_bytes=300,
        uncompressed_size_bytes=1_000,
        top_k=2,
        baseline_samples=baseline,
    )
    delta = summary["delta_from_baseline"]

    assert delta["baseline_n_samples"] == 3
    assert delta["top_excess_pose_samples"][0]["pair_index"] == 0
    assert delta["top_excess_seg_samples"][0]["pair_index"] == 1
    assert delta["top_excess_combined_samples"][0]["pair_index"] in {0, 1}


def test_component_trace_auto_selects_parity_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trace = _load_trace_module()
    good = _write_executable(tmp_path / "ffmpeg-good", _fake_ffmpeg_script(include_color_contract=True))
    bad = _write_executable(tmp_path / "ffmpeg-bad", _fake_ffmpeg_script(include_color_contract=False))

    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.setattr(trace, "PARITY_FFMPEG_CANDIDATE_PATHS", (bad, good))

    assert trace._ensure_parity_ffmpeg_env() == good.resolve()
    assert Path(trace.os.environ["FFMPEG_BIN"]) == good.resolve()


def test_component_trace_rejects_bad_explicit_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trace = _load_trace_module()
    bad = _write_executable(tmp_path / "ffmpeg-bad", _fake_ffmpeg_script(include_color_contract=False))

    monkeypatch.setenv("FFMPEG_BIN", str(bad))

    with pytest.raises(RuntimeError, match="not parity-compatible"):
        trace._ensure_parity_ffmpeg_env()


def test_component_trace_isolates_inflate_uv_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace = _load_trace_module()
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.delenv("UV_LINK_MODE", raising=False)

    uv_env = trace._ensure_isolated_inflate_uv_env(tmp_path / "trace_work")

    assert uv_env == (tmp_path / "trace_work" / "uv_project_env").resolve()
    assert Path(trace.os.environ["UV_PROJECT_ENVIRONMENT"]) == uv_env
    assert trace.os.environ["UV_LINK_MODE"] == "copy"


def test_component_trace_preserves_explicit_inflate_uv_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace = _load_trace_module()
    explicit = tmp_path / "explicit_uv_env"
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(explicit))

    uv_env = trace._ensure_isolated_inflate_uv_env(tmp_path / "trace_work")

    assert uv_env == explicit.resolve()
    assert trace.os.environ["UV_PROJECT_ENVIRONMENT"] == str(explicit)


def test_preflight_guards_component_trace_runtime_parity() -> None:
    from tac.preflight import check_contest_component_trace_runtime_parity

    assert check_contest_component_trace_runtime_parity(REPO, strict=True, verbose=False) == []


def test_preflight_rejects_component_trace_without_parity_guards(tmp_path: Path) -> None:
    from tac.preflight import check_contest_component_trace_runtime_parity

    repo = tmp_path / "repo"
    experiments = repo / "experiments"
    experiments.mkdir(parents=True)
    shutil.copy2(TRACE_PATH, experiments / "contest_component_trace.py")
    text = (experiments / "contest_component_trace.py").read_text()
    text = text.replace("_ensure_parity_ffmpeg_env()", "# parity resolver removed")
    text = text.replace("component_trace_runtime_env.json", "component_trace.json")
    (experiments / "contest_component_trace.py").write_text(text)

    violations = check_contest_component_trace_runtime_parity(repo, strict=False, verbose=False)

    assert any("parity ffmpeg resolver" in violation for violation in violations)
    assert any("runtime environment sidecar" in violation for violation in violations)
