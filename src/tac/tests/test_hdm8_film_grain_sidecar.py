# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
RUNTIME_DIR = REPO / "submissions" / "hdm8_film_grain_sidecar"
RUNTIME_INFLATE = RUNTIME_DIR / "inflate.py"
BUILDER_PATH = REPO / "tools" / "build_hdm8_film_grain_sidecar_packet.py"
HDM8_ARCHIVE = (
    REPO
    / "experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/"
    "exact_eval_static_release_surface/archive.zip"
)
CURRENT_PROXY = REPO / "experiments/results/hdm8_postfilter_sweep_20260514_codex/proxy_4pairs_cpu.json"
N_SELECTOR_PAIRS = 600


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime():
    for name in ("codec", "model", "pr101_grammar"):
        sys.modules.pop(name, None)
    return _load_module(RUNTIME_INFLATE, "hdm8_film_grain_sidecar_inflate_test")


def _builder():
    return _load_module(BUILDER_PATH, "build_hdm8_film_grain_sidecar_packet_test")


def _selector_proxy_payload(
    *,
    axis: str = "local-mps-proxy-prefix",
    n_pairs: int = N_SELECTOR_PAIRS,
    baseline_pose: float = 0.001,
    candidate_pose: float = 0.0001,
) -> dict[str, object]:
    return {
        "axis": axis,
        "archive_bytes": 186395,
        "n_pairs": n_pairs,
        "modes": [
            {
                "mode": "none",
                "score_proxy": 0.30,
                "avg_posenet_dist": baseline_pose,
                "avg_segnet_dist": 0.001,
                "pair_posenet_dist": [baseline_pose] * n_pairs,
                "pair_segnet_dist": [0.001] * n_pairs,
            },
            {
                "mode": "even_bias:1",
                "score_proxy": 0.20,
                "avg_posenet_dist": candidate_pose,
                "avg_segnet_dist": 0.001,
                "pair_posenet_dist": [candidate_pose] * n_pairs,
                "pair_segnet_dist": [0.001] * n_pairs,
            },
        ],
    }


def test_postfilter_config_is_fixed_runtime_contract(tmp_path: Path) -> None:
    inflate = _runtime()
    cfg = tmp_path / "postfilter_config.json"
    cfg.write_text(
        json.dumps(
            {
                "schema": inflate.POSTFILTER_CONFIG_SCHEMA,
                "mode": "unsharp:0.03",
                "score_claim": False,
            }
        ),
        encoding="utf-8",
    )

    assert inflate.load_postfilter_mode(cfg) == "unsharp:0.03"

    cfg.write_text(json.dumps({"schema": "wrong", "mode": "none"}), encoding="utf-8")
    with pytest.raises(ValueError, match="bad postfilter config schema"):
        inflate.load_postfilter_mode(cfg)


def test_postfilter_none_and_bias_are_deterministic() -> None:
    inflate = _runtime()
    frames = torch.arange(1 * 3 * 10 * 10, dtype=torch.float32).view(1, 3, 10, 10)

    assert torch.equal(inflate.apply_postfilter(frames, "none", frame_start=0), frames)
    shifted = inflate.apply_postfilter(frames, "bias:-1", frame_start=0)

    assert torch.equal(shifted, frames - 1.0)


def test_postfilter_grain_modes_are_frame_index_deterministic() -> None:
    inflate = _runtime()
    frames = torch.full((2, 3, 10, 10), 128.0)

    a = inflate.apply_postfilter(frames, "grain_luma:0.25", frame_start=10)
    b = inflate.apply_postfilter(frames, "grain_luma:0.25", frame_start=10)
    c = inflate.apply_postfilter(frames, "grain_luma:0.25", frame_start=11)

    assert torch.equal(a, b)
    assert not torch.equal(a, frames)
    assert not torch.equal(a, c)


def test_postfilter_tile_chroma_moves_red_and_blue_opposite_directions() -> None:
    inflate = _runtime()
    frames = torch.full((1, 3, 8, 8), 128.0)

    filtered = inflate.apply_postfilter(frames, "tile_chroma:3", frame_start=0)

    assert torch.equal(filtered[:, 1], frames[:, 1])
    assert torch.equal(filtered[:, 0] - frames[:, 0], -(filtered[:, 2] - frames[:, 2]))
    assert torch.count_nonzero(filtered[:, 0] - frames[:, 0]).item() == 64


def test_postfilter_unsharp_soften_and_parity_modes_preserve_shape() -> None:
    inflate = _runtime()
    frames = torch.linspace(0, 255, steps=2 * 3 * 12 * 12).view(2, 3, 12, 12)

    unsharp = inflate.apply_postfilter(frames, "unsharp:0.03", frame_start=0)
    soften = inflate.apply_postfilter(frames, "soften:0.02", frame_start=0)
    even_bias = inflate.apply_postfilter(frames, "even_bias:1", frame_start=0)

    assert unsharp.shape == frames.shape
    assert soften.shape == frames.shape
    assert even_bias.shape == frames.shape
    assert torch.equal(even_bias[0], frames[0] + 1.0)
    assert torch.equal(even_bias[1], frames[1])


def test_postfilter_rgb_bias_and_composite_modes() -> None:
    inflate = _runtime()
    frames = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    rgb = inflate.apply_postfilter(frames, "rgb_bias:2,-1,0.5", frame_start=0)
    composite = inflate.apply_postfilter(frames, "rgb_bias:2,-1,0.5+bias:1", frame_start=0)

    assert torch.equal(rgb[:, 0], torch.full((1, 4, 4), 2.0))
    assert torch.equal(rgb[:, 1], torch.full((1, 4, 4), -1.0))
    assert torch.equal(rgb[:, 2], torch.full((1, 4, 4), 0.5))
    assert torch.equal(composite, rgb + 1.0)


def test_postfilter_rgb_scale_contrast_and_gamma_modes() -> None:
    inflate = _runtime()
    frames = torch.full((1, 3, 4, 4), 128.0, dtype=torch.float32)

    scaled = inflate.apply_postfilter(frames, "rgb_scale:1.01,0.99,1", frame_start=0)
    contrast = inflate.apply_postfilter(frames, "contrast:0.1", frame_start=0)
    gamma = inflate.apply_postfilter(frames, "gamma:1.03", frame_start=0)

    assert torch.allclose(scaled[:, 0], torch.full((1, 4, 4), 129.28))
    assert torch.allclose(scaled[:, 1], torch.full((1, 4, 4), 126.72))
    assert torch.allclose(scaled[:, 2], torch.full((1, 4, 4), 128.0))
    assert torch.allclose(contrast, torch.full_like(frames, 128.05))
    assert gamma.shape == frames.shape
    assert not torch.equal(gamma, frames)


def test_postfilter_selector_applies_pair_specific_modes() -> None:
    inflate = _runtime()
    frames = torch.zeros((4, 3, 4, 4), dtype=torch.float32)
    config = {
        "schema": inflate.POSTFILTER_CONFIG_SCHEMA,
        "mode": "selector",
        "palette": ["none", "even_bias:3", "even_rgb_bias:1,2,3"],
        "selector_indices": [1, 2],
    }

    filtered = inflate.apply_postfilter_config(frames, config, pair_start=0)

    assert torch.equal(filtered[0], torch.full((3, 4, 4), 3.0))
    assert torch.equal(filtered[1], frames[1])
    assert torch.equal(filtered[2, 0], torch.full((4, 4), 1.0))
    assert torch.equal(filtered[2, 1], torch.full((4, 4), 2.0))
    assert torch.equal(filtered[2, 2], torch.full((4, 4), 3.0))
    assert torch.equal(filtered[3], frames[3])


def test_postfilter_selector_config_validation(tmp_path: Path) -> None:
    inflate = _runtime()
    cfg = tmp_path / "postfilter_config.json"
    cfg.write_text(
        json.dumps(
            {
                "schema": inflate.POSTFILTER_CONFIG_SCHEMA,
                "mode": "selector",
                "palette": ["none", "even_bias:1"],
                "selector_indices": [0, 1, 0],
                "score_claim": False,
            }
        ),
        encoding="utf-8",
    )

    config = inflate.load_postfilter_config(cfg)

    assert config["mode"] == "selector"
    assert config["palette"] == ["none", "even_bias:1"]
    assert config["selector_indices"] == [0, 1, 0]


def test_runtime_parser_consumes_current_hdm8_archive_shape() -> None:
    inflate = _runtime()
    with zipfile.ZipFile(HDM8_ARCHIVE) as zf:
        assert zf.namelist() == ["x"]
        payload = zf.read("x")

    format_id, pr106_bytes, sidecar_blob, framing_meta = inflate.parse_sidecar_archive(payload)
    decoder_sd, latents, meta = inflate.parse_packed_archive(pr106_bytes)

    assert format_id == inflate.SIDECAR_FORMAT_PR101_GRAMMAR
    assert sidecar_blob
    assert framing_meta is not None
    assert meta["n_pairs"] == 600
    assert tuple(meta["eval_size"]) == (384, 512)
    assert latents.shape == (600, 28)
    assert decoder_sd


def test_builder_materializes_non_dispatchable_packet_for_nonpositive_proxy(tmp_path: Path) -> None:
    builder = _builder()

    manifest = builder.build_packet(
        archive=HDM8_ARCHIVE,
        runtime_template=RUNTIME_DIR,
        output_dir=tmp_path / "packet",
        mode="none",
        proxy_json=CURRENT_PROXY,
    )

    assert manifest["static_packet_ready"] is True
    assert manifest["positive_proxy_candidate_for_cuda_probe"] is False
    assert manifest["ready_for_exact_cuda_after_positive_proxy"] is False
    assert manifest["cuda_transfer_policy"]["rankable_on_cuda"] is False
    assert "postfilter_mode_none" in manifest["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in manifest["dispatch_blockers"]
    assert manifest["source_transparency"]["schema"] == "tac_source_transparency_v1"
    assert manifest["source_transparency"]["release_contract"][
        "include_in_submission_packets"
    ] is True
    assert "architecture" in manifest["experiment_transparency"]
    assert "training_curriculum" in manifest["experiment_transparency"]
    assert manifest["packet_build_command_template"][0:2] == [
        ".venv/bin/python",
        "tools/build_hdm8_film_grain_sidecar_packet.py",
    ]
    assert (tmp_path / "packet" / "archive.zip").exists()
    assert (tmp_path / "packet" / "submission_dir" / "inflate.sh").exists()
    assert (tmp_path / "packet" / "packet_manifest.json").exists()
    assert "## Source Transparency" in (tmp_path / "packet" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "## Experiment Transparency" in (tmp_path / "packet" / "README.md").read_text(
        encoding="utf-8"
    )


def test_builder_requires_positive_proxy_when_requested(tmp_path: Path) -> None:
    builder = _builder()

    with pytest.raises(SystemExit, match="lacks positive proxy evidence"):
        builder.build_packet(
            archive=HDM8_ARCHIVE,
            runtime_template=RUNTIME_DIR,
            output_dir=tmp_path / "packet",
            mode="unsharp:0.20",
            proxy_json=CURRENT_PROXY,
            require_positive_proxy=True,
        )


def test_builder_accepts_synthetic_positive_proxy(tmp_path: Path) -> None:
    builder = _builder()
    proxy = tmp_path / "positive_proxy.json"
    proxy.write_text(
        json.dumps(
            {
                "axis": "local-cpu-proxy-prefix",
                "n_pairs": 2,
                "modes": [
                    {"mode": "none", "score_proxy": 0.21, "delta_vs_none": 0.0},
                    {"mode": "grain_luma:0.25", "score_proxy": 0.20, "delta_vs_none": -0.01},
                ],
                "best": {
                    "mode": "grain_luma:0.25",
                    "score_proxy": 0.20,
                    "delta_vs_none": -0.01,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = builder.build_packet(
        archive=HDM8_ARCHIVE,
        runtime_template=RUNTIME_DIR,
        output_dir=tmp_path / "packet",
        mode="grain_luma:0.25",
        proxy_json=proxy,
        require_positive_proxy=True,
    )

    assert manifest["positive_proxy_candidate_for_cuda_probe"] is True
    assert manifest["ready_for_exact_cuda_after_positive_proxy"] is False
    assert manifest["cuda_transfer_policy"]["rankable_on_cuda"] is False
    assert "positive_proxy_requires_cuda_transfer_confirmation" in manifest["dispatch_blockers"]
    assert manifest["proxy"]["positive"] is True
    assert manifest["postfilter_mode"] == "grain_luma:0.25"
    assert "--expected-runtime-tree-sha256" in manifest["exact_cuda_auth_eval_command_template"]


def test_builder_materializes_selector_from_pair_proxy(tmp_path: Path) -> None:
    builder = _builder()
    proxy = tmp_path / "pair_proxy.json"
    proxy.write_text(
        json.dumps(
            {
                "axis": "local-mps-proxy-prefix",
                "archive_bytes": 186395,
                "n_pairs": 2,
                "modes": [
                    {
                        "mode": "none",
                        "score_proxy": 0.30,
                        "avg_posenet_dist": 0.001,
                        "avg_segnet_dist": 0.001,
                        "pair_posenet_dist": [0.001, 0.001],
                        "pair_segnet_dist": [0.001, 0.001],
                    },
                    {
                        "mode": "even_bias:1",
                        "score_proxy": 0.20,
                        "avg_posenet_dist": 0.0001,
                        "avg_segnet_dist": 0.001,
                        "pair_posenet_dist": [0.0001, 0.002],
                        "pair_segnet_dist": [0.001, 0.001],
                    },
                    {
                        "mode": "even_rgb_bias:1,0,-1",
                        "score_proxy": 0.19,
                        "avg_posenet_dist": 0.0001,
                        "avg_segnet_dist": 0.001,
                        "pair_posenet_dist": [0.002, 0.0001],
                        "pair_segnet_dist": [0.001, 0.001],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = builder.build_packet(
        archive=HDM8_ARCHIVE,
        runtime_template=RUNTIME_DIR,
        output_dir=tmp_path / "packet",
        mode="ignored",
        proxy_json=proxy,
        selector_from_proxy_json=True,
        require_positive_proxy=True,
    )
    config = json.loads(
        (tmp_path / "packet" / "submission_dir" / "postfilter_config.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["postfilter_mode"] == "selector"
    assert manifest["proxy"]["positive"] is True
    assert manifest["positive_proxy_candidate_for_cuda_probe"] is True
    assert manifest["ready_for_exact_cuda_after_positive_proxy"] is False
    assert manifest["cuda_transfer_policy"]["required_before_ranking"] == [
        "cuda_component_risk_gate_passed",
        "lane_dispatch_claim",
        "byte_closed_archive_runtime_exact_contest_cuda_auth_eval",
    ]
    assert manifest["cuda_component_risk_gate_required"] is True
    assert manifest["cuda_component_risk_gate"]["passed"] is False
    assert (
        "mps_or_local_proxy_axis_requires_cuda_component_probe"
        in manifest["cuda_component_risk_gate"]["blockers"]
    )
    assert manifest["proxy"]["selector_config_bytes_if_charged"] > 0
    assert "selector_side_information_must_be_packed_into_archive_for_submission" in manifest[
        "dispatch_blockers"
    ]
    assert "hdm8_selector_cuda_component_risk_gate_not_passed" in manifest[
        "dispatch_blockers"
    ]
    assert config["mode"] == "selector"
    assert config["selector_indices"] == [1, 2]


def test_builder_can_pack_selector_into_archive(tmp_path: Path) -> None:
    builder = _builder()
    inflate = _runtime()
    proxy = tmp_path / "pair_proxy.json"
    proxy.write_text(json.dumps(_selector_proxy_payload()), encoding="utf-8")

    manifest = builder.build_packet(
        archive=HDM8_ARCHIVE,
        runtime_template=RUNTIME_DIR,
        output_dir=tmp_path / "packet",
        mode="ignored",
        proxy_json=proxy,
        selector_from_proxy_json=True,
        pack_selector_into_archive=True,
        require_positive_proxy=True,
    )
    with zipfile.ZipFile(tmp_path / "packet" / "archive.zip") as zf:
        payload = zf.read("x")

    format_id, _pr106, _sidecar, _framing, embedded = inflate.parse_sidecar_archive_with_selector(
        payload
    )

    assert format_id == inflate.SIDECAR_FORMAT_PR101_SELECTOR_BROTLI
    assert embedded is not None
    assert embedded["mode"] == "selector"
    assert embedded["selector_indices"] == [1] * N_SELECTOR_PAIRS
    assert manifest["selector_packed_in_archive"] is True
    assert manifest["positive_proxy_candidate_for_cuda_probe"] is True
    assert manifest["ready_for_exact_cuda_after_positive_proxy"] is False
    assert manifest["cuda_transfer_policy"]["ready_for_exact_eval_dispatch"] is False
    assert manifest["cuda_component_risk_gate_required"] is True
    assert manifest["cuda_component_risk_gate"]["passed"] is False
    assert manifest["archive"]["selector_pack_manifest"]["selector_codec"] == "brotli"
    assert manifest["archive"]["selector_pack_manifest"]["format_id"] == "0x04"
    assert manifest["archive"]["selector_pack_manifest"]["selector_encoded_bytes"] <= manifest[
        "archive"
    ]["selector_pack_manifest"]["selector_json_bytes"]
    assert manifest["proxy"]["archive_bytes_if_charged"] == manifest["archive"]["bytes"]
    assert manifest["proxy"]["positive_charged"] is True
    assert manifest["proxy"]["delta_vs_none_charged"] < 0
    assert not any("selector_side_information" in b for b in manifest["dispatch_blockers"])
    assert "hdm8_selector_cuda_component_risk_gate_not_passed" in manifest[
        "dispatch_blockers"
    ]


def test_builder_refuses_to_pack_incomplete_prefix_selector(tmp_path: Path) -> None:
    builder = _builder()
    proxy = tmp_path / "prefix_proxy.json"
    proxy.write_text(
        json.dumps(_selector_proxy_payload(axis="modal-t4-cuda-proxy-prefix", n_pairs=32)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 600 pair indices"):
        builder.build_packet(
            archive=HDM8_ARCHIVE,
            runtime_template=RUNTIME_DIR,
            output_dir=tmp_path / "packet",
            mode="ignored",
            proxy_json=proxy,
            selector_from_proxy_json=True,
            pack_selector_into_archive=True,
            require_positive_proxy=True,
        )


def test_builder_cuda_prefix_selector_component_gate_can_pass(tmp_path: Path) -> None:
    builder = _builder()
    proxy = tmp_path / "cuda_pair_proxy.json"
    proxy.write_text(
        json.dumps(_selector_proxy_payload(axis="modal-t4-cuda-proxy-prefix")),
        encoding="utf-8",
    )

    manifest = builder.build_packet(
        archive=HDM8_ARCHIVE,
        runtime_template=RUNTIME_DIR,
        output_dir=tmp_path / "packet",
        mode="ignored",
        proxy_json=proxy,
        selector_from_proxy_json=True,
        pack_selector_into_archive=True,
        require_positive_proxy=True,
    )

    assert manifest["cuda_component_risk_gate_required"] is True
    assert manifest["cuda_component_risk_gate"]["passed"] is True
    assert manifest["cuda_component_risk_gate"]["status"] == "passed_cuda_prefix_component_check"
    assert manifest["ready_for_exact_cuda_after_positive_proxy"] is True
    assert manifest["cuda_transfer_policy"]["ready_for_exact_eval_dispatch"] is True
    assert "hdm8_selector_cuda_component_risk_gate_not_passed" not in manifest[
        "dispatch_blockers"
    ]
    archive_manifest = json.loads(
        (tmp_path / "packet" / "archive_manifest.json").read_text(encoding="utf-8")
    )
    assert archive_manifest["cuda_component_risk_gate"]["passed"] is True


def test_builder_can_pack_uncompressed_selector_for_legacy_parser(tmp_path: Path) -> None:
    builder = _builder()
    inflate = _runtime()
    proxy = tmp_path / "pair_proxy.json"
    proxy.write_text(json.dumps(_selector_proxy_payload()), encoding="utf-8")

    builder.build_packet(
        archive=HDM8_ARCHIVE,
        runtime_template=RUNTIME_DIR,
        output_dir=tmp_path / "packet",
        mode="ignored",
        proxy_json=proxy,
        selector_from_proxy_json=True,
        pack_selector_into_archive=True,
        selector_codec="json",
        require_positive_proxy=True,
    )
    with zipfile.ZipFile(tmp_path / "packet" / "archive.zip") as zf:
        payload = zf.read("x")

    format_id, _pr106, _sidecar, _framing, embedded = inflate.parse_sidecar_archive_with_selector(
        payload
    )

    assert format_id == inflate.SIDECAR_FORMAT_PR101_SELECTOR
    assert embedded is not None
    assert embedded["selector_indices"] == [1] * N_SELECTOR_PAIRS
