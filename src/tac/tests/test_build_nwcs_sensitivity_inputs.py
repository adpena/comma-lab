from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.component_sensitivity_artifact import (
    materialize_component_sensitivity_manifest,
    write_component_sensitivity_manifest,
)
from tac.neural_weight_corpus import build_corpus_manifest, write_corpus_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "build_nwcs_sensitivity_inputs.py"


class TinyRenderer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.linspace(0.1, 1.0, 32))
        self.conv = nn.Conv2d(3, 4, kernel_size=3, bias=False)
        self.head = nn.Linear(16, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.head(x)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_nwcs_sensitivity_inputs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair_record(pair_index: int) -> dict[str, int]:
    return {
        "video": 0,
        "pair_index": pair_index,
        "t": 2 * pair_index,
        "t1": 2 * pair_index + 1,
    }


def _sample_plan() -> dict[str, object]:
    calibration = list(range(480))
    holdout = list(range(480, 600))
    split_hash = hashlib.sha256(
        json.dumps(
            {"calibration": calibration, "holdout": holdout, "split_seed": 123},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "calibration_pairs": [_pair_record(i) for i in calibration],
        "holdout_pairs": [_pair_record(i) for i in holdout],
        "split_seed": 123,
        "split_hash": split_hash,
    }


def _write_component_manifest(
    root: Path,
    *,
    anchor_renderer: Path,
    anchor_archive: Path,
    map_payload: dict[str, object],
) -> Path:
    upstream = root / "upstream"
    upstream.mkdir(exist_ok=True)
    (upstream / "README").write_text("synthetic upstream\n")
    video = root / "video.mkv"
    video.write_bytes(b"video")
    contest = root / "contest_auth_eval.json"
    contest.write_text(
        json.dumps(
            {
                "n_samples": 600,
                "archive_size_bytes": anchor_archive.stat().st_size,
                "provenance": {
                    "device": "cuda",
                    "archive_sha256": hashlib.sha256(anchor_archive.read_bytes()).hexdigest(),
                },
            },
            sort_keys=True,
        )
        + "\n"
    )

    map_paths: dict[str, Path] = {}
    curve_paths: dict[str, Path] = {}
    readouts = {
        "posenet": "official_pose_mse",
        "segnet": "official_argmax_disagreement",
        "combined": "official_component_formula",
    }
    for component in ("posenet", "segnet", "combined"):
        path = root / f"{component}_map.pt"
        torch.save(map_payload, path)
        map_paths[component] = path
        curve_path = root / f"{component}_curve.json"
        curve_path.write_text(
            json.dumps(
                {
                    "count": 3,
                    "holdout_error": 0.01,
                    "official_component_response": True,
                    "passed": True,
                    "gate_spec": {
                        "holdout_error_max": 0.05,
                        "spearman_min": 0.3,
                        "zero_repro_tolerance": 1e-7,
                    },
                    "gate_results": {
                        "coverage_passed": True,
                        "finite_values": True,
                        "observed_delta_max": 0.02,
                        "prediction_error_passed": True,
                        "promotion_gate_passed": True,
                        "signal_present": True,
                        "zero_repro": True,
                        "zero_repro_error": 0.0,
                    },
                    "promotion_blockers": [],
                    "component_readout": readouts[component],
                    "response_kind": "symmetric",
                    "epsilon_ladder": [-0.1, 0.0, 0.1],
                },
                sort_keys=True,
            )
            + "\n"
        )
        curve_paths[component] = curve_path

    draft = {
        "schema_version": 1,
        "format": "component_sensitivity_v1",
        "device": "cuda",
        "promotion_eligible": True,
        "evidence_grade": "A",
        "inputs": {
            "checkpoint": {"path": str(anchor_renderer)},
            "video": {"path": str(video)},
            "upstream": {"path": str(upstream)},
        },
        "sample_plan": _sample_plan(),
        "component_maps": {
            component: {
                "path": str(map_paths[component]),
                "scorer_target": component,
                "tensor_count": 3,
                "dtype": "float32",
                "map_format": "tac_score_sensitivity_map_v1",
                "certification": {
                    "format": "component_sensitivity_map_certification_v1",
                    "component": component,
                    "device": "cuda",
                    "official_component_response": True,
                    "canonical_scorer_path": True,
                    "promotion_eligible": True,
                    "source_map_sha256": "a" * 64,
                    "official_response_curve_sha256": "b" * 64,
                    "stability_sha256": "c" * 64,
                    "sample_plan_sha256": "d" * 64,
                    "baseline_archive_sha256": "e" * 64,
                    "baseline_archive_bytes": anchor_archive.stat().st_size,
                    "contest_auth_eval_json_sha256": "f" * 64,
                    "review_clean_passes": 3,
                    "review_unresolved_blockers": [],
                    "response_gate_results": {
                        "finite_values": True,
                        "coverage_passed": True,
                        "zero_repro": True,
                        "zero_repro_error": 0.0,
                        "signal_present": True,
                        "observed_delta_max": 0.01,
                        "prediction_error_passed": True,
                        "max_relative_prediction_error": 0.02,
                        "promotion_gate_passed": True,
                    },
                    "stability_gate_results": {
                        "passed": True,
                        "cv_max": 0.04,
                        "spearman_min": 0.96,
                        "top_decile_overlap_min": 0.91,
                    },
                },
            }
            for component in ("posenet", "segnet", "combined")
        },
        "stability": {
            "cv": {"posenet": 0.01, "segnet": 0.02, "combined": 0.03},
            "rank": {"posenet": 0.98, "segnet": 0.97, "combined": 0.96},
            "top_k": {
                "posenet": {"k": 8, "overlap": 0.9},
                "segnet": {"k": 8, "overlap": 0.9},
                "combined": {"k": 8, "overlap": 0.9},
            },
            "thresholds": {
                "cv_max": 0.35,
                "spearman_min": 0.3,
                "top_decile_overlap_min": 0.5,
            },
            "passed": True,
        },
        "response_curves": {
            component: {"path": str(curve_paths[component])}
            | json.loads(curve_paths[component].read_text())
            for component in ("posenet", "segnet", "combined")
        },
        "contest_eval": {
            "archive": {"path": str(anchor_archive)},
            "contest_auth_eval_json": {"path": str(contest)},
            "device": "cuda",
            "n_samples": 600,
        },
        "promotion_blockers": [],
    }
    manifest = materialize_component_sensitivity_manifest(draft, promotion=True)
    out = root / "component_sensitivity_v1.json"
    write_component_sensitivity_manifest(out, manifest)
    return out


def _write_corpus_manifest(root: Path, model: TinyRenderer) -> Path:
    ckpt = root / "corpus_renderer.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    manifest = build_corpus_manifest(
        [ckpt],
        block_size=16,
        max_blocks_per_ckpt=1000,
        min_checkpoint_bytes=0,
        corpus_dir=root,
    )
    out = root / "corpus_manifest.json"
    write_corpus_manifest(manifest, out)
    return out


def _valid_map_payload(model: TinyRenderer) -> dict[str, object]:
    return {
        "format": "tac_mixed_parameter_sensitivity_map_v1",
        "sensitivities": {
            "offset": torch.linspace(0.2, 1.1, model.offset.numel()),
            "conv.weight": torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32),
            "head.weight": torch.linspace(
                0.1,
                2.0,
                model.head.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(model.head.weight),
        },
        "metadata": {
            "promotion_eligible": True,
            "official_component_response": True,
        },
    }


def _base_argv(paths: dict[str, Path]) -> list[str]:
    return [
        "--component-sensitivity-manifest",
        str(paths["component_manifest"]),
        "--anchor-renderer",
        str(paths["anchor_renderer"]),
        "--anchor-archive",
        str(paths["anchor_archive"]),
        "--corpus-manifest",
        str(paths["corpus_manifest"]),
        "--anchor-output",
        str(paths["anchor_output"]),
        "--corpus-output",
        str(paths["corpus_output"]),
    ]


def _write_inputs(tmp_path: Path, map_payload: dict[str, object]) -> tuple[TinyRenderer, dict[str, Path]]:
    torch.manual_seed(1)
    model = TinyRenderer()
    anchor_renderer = tmp_path / "renderer.bin"
    anchor_renderer.write_bytes(b"renderer")
    anchor_archive = tmp_path / "archive.zip"
    anchor_archive.write_bytes(b"archive")
    component_manifest = _write_component_manifest(
        tmp_path,
        anchor_renderer=anchor_renderer,
        anchor_archive=anchor_archive,
        map_payload=map_payload,
    )
    corpus_manifest = _write_corpus_manifest(tmp_path, model)
    paths = {
        "anchor_renderer": anchor_renderer,
        "anchor_archive": anchor_archive,
        "component_manifest": component_manifest,
        "corpus_manifest": corpus_manifest,
        "anchor_output": tmp_path / "anchor_sensitivity.pt",
        "corpus_output": tmp_path / "corpus_sensitivity.pt",
    }
    return model, paths


def test_builds_anchor_and_corpus_sensitivity_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    model, paths = _write_inputs(tmp_path, _valid_map_payload(TinyRenderer()))
    monkeypatch.setattr(module, "load_any_renderer_checkpoint", lambda _path, device="cpu": model)

    summary = module.build_inputs(module.parse_args(_base_argv(paths)))

    assert summary["promotion_eligible"] is True
    anchor = torch.load(paths["anchor_output"], map_location="cpu", weights_only=False)
    corpus = torch.load(paths["corpus_output"], map_location="cpu", weights_only=False)
    assert anchor["format"] == module.ANCHOR_OUTPUT_FORMAT
    assert corpus["format"] == module.CORPUS_OUTPUT_FORMAT
    assert anchor["metadata"]["source"] == "component_sensitivity_v1.combined"
    assert corpus["metadata"]["source"] == "anchor_parameter_sensitivity_projected_to_corpus_manifest"
    assert set(anchor["sensitivities"]) == {"offset", "conv.weight", "head.weight"}
    assert anchor["metadata"]["component_sensitivity_manifest_sha256"] == hashlib.sha256(
        paths["component_manifest"].read_bytes()
    ).hexdigest()
    assert anchor["metadata"]["anchor_renderer_sha256"] == hashlib.sha256(
        paths["anchor_renderer"].read_bytes()
    ).hexdigest()
    assert anchor["metadata"]["parameters"]["conv.weight"]["source_kind"] == "per_output_channel"
    assert anchor["metadata"]["parameters"]["head.weight"]["source_kind"] == "per_element"
    assert anchor["sensitivities"]["conv.weight"].numel() == model.conv.weight.numel() // 16
    assert anchor["sensitivities"]["head.weight"].numel() == model.head.weight.numel() // 16
    assert corpus["sensitivities"].numel() == corpus["metadata"]["num_blocks"]
    assert corpus["metadata"]["corpus_manifest_sha256"] == hashlib.sha256(
        paths["corpus_manifest"].read_bytes()
    ).hexdigest()
    assert torch.isfinite(corpus["sensitivities"]).all()
    assert corpus["sensitivities"].max() > corpus["sensitivities"].min()


def test_rejects_stale_corpus_manifest_checkpoint_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    model, paths = _write_inputs(tmp_path, _valid_map_payload(TinyRenderer()))
    monkeypatch.setattr(module, "load_any_renderer_checkpoint", lambda _path, device="cpu": model)

    stale_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    stale_state["offset"] = stale_state["offset"] + 1.0
    torch.save({"state_dict": stale_state}, tmp_path / "corpus_renderer.pt")

    with pytest.raises(
        module.NWCSSensitivityInputBuildError,
        match=r"corpus manifest (size|sha256) mismatch",
    ):
        module.build_inputs(module.parse_args(_base_argv(paths)))


def test_rejects_uniform_sensitivity_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    model = TinyRenderer()
    payload = {
        "format": "tac_mixed_parameter_sensitivity_map_v1",
        "sensitivities": {
            "offset": torch.ones_like(model.offset),
            "conv.weight": torch.ones_like(model.conv.weight),
            "head.weight": torch.ones_like(model.head.weight),
        },
        "metadata": {"promotion_eligible": True, "official_component_response": True},
    }
    model, paths = _write_inputs(tmp_path, payload)
    monkeypatch.setattr(module, "load_any_renderer_checkpoint", lambda _path, device="cpu": model)

    with pytest.raises(
        module.NWCSSensitivityInputBuildError,
        match="uniform/fake sensitivity is non-promotable",
    ):
        module.build_inputs(module.parse_args(_base_argv(paths)))


def test_rejects_non_promotable_component_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    model, paths = _write_inputs(tmp_path, _valid_map_payload(TinyRenderer()))
    monkeypatch.setattr(module, "load_any_renderer_checkpoint", lambda _path, device="cpu": model)
    manifest = json.loads(paths["component_manifest"].read_text())
    manifest["promotion_eligible"] = False
    manifest["evidence_grade"] = "empirical"
    paths["component_manifest"].write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(
        module.NWCSSensitivityInputBuildError,
        match="not a promotable component_sensitivity_v1 manifest",
    ):
        module.build_inputs(module.parse_args(_base_argv(paths)))


def test_rejects_conv_only_maps_that_do_not_cover_anchor_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    payload = {
        "format": "tac_score_sensitivity_map_v1",
        "sensitivities": {
            "conv.weight": torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32),
        },
        "metadata": {
            "promotion_eligible": True,
            "sensitivity_granularity": "per_channel",
            "official_component_response": True,
        },
    }
    model, paths = _write_inputs(tmp_path, payload)
    monkeypatch.setattr(module, "load_any_renderer_checkpoint", lambda _path, device="cpu": model)

    with pytest.raises(
        module.NWCSSensitivityInputBuildError,
        match="does not cover .*float anchor parameter",
    ):
        module.build_inputs(module.parse_args(_base_argv(paths)))


def test_rejects_fake_marked_map_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    model = TinyRenderer()
    payload = _valid_map_payload(model)
    payload["metadata"] = {
        "promotion_eligible": True,
        "fake_sensitivity": True,
        "official_component_response": True,
    }
    model, paths = _write_inputs(tmp_path, payload)
    monkeypatch.setattr(module, "load_any_renderer_checkpoint", lambda _path, device="cpu": model)

    with pytest.raises(
        module.NWCSSensitivityInputBuildError,
        match="diagnostic/non-promotable marker",
    ):
        module.build_inputs(module.parse_args(_base_argv(paths)))


def test_rejects_profile_component_sensitivity_proxy_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    model = TinyRenderer()
    payload = _valid_map_payload(model)
    payload["metadata"] = {
        "promotion_eligible": True,
        "official_component_response": True,
        "tool": "experiments/profile_component_sensitivity.py",
        "evidence_grade": "diagnostic_cuda_fisher_proxy",
    }
    model, paths = _write_inputs(tmp_path, payload)
    monkeypatch.setattr(module, "load_any_renderer_checkpoint", lambda _path, device="cpu": model)

    with pytest.raises(
        module.NWCSSensitivityInputBuildError,
        match="diagnostic/non-promotable marker",
    ):
        module.build_inputs(module.parse_args(_base_argv(paths)))
