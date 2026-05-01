from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_component_sensitivity.py"


def _load_module():
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("profile_component_sensitivity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_has_required_cli_flags() -> None:
    src = SCRIPT.read_text()
    flags = set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))
    expected = {
        "checkpoint",
        "video",
        "masks-mkv",
        "poses",
        "upstream",
        "output-dir",
        "pair-weights",
        "all-pairs",
        "top-k-pairs",
        "pair-batch",
        "response-top-k",
        "response-epsilons",
        "split-seed",
        "holdout-fraction",
        "aggregate",
        "device",
        "allow-diagnostic-cpu",
        "archive",
        "contest-auth-eval-json",
        "manifest-output",
        "evidence-grade",
        "promotion-finite-difference",
        "finite-difference-epsilon",
        "finite-difference-shard-index",
        "finite-difference-shard-count",
        "merge-shard-dir",
    }
    assert not (expected - flags)
    assert "add_mutually_exclusive_group(required=True)" in src
    assert "archive.zip -> inflate.sh -> upstream/evaluate.py" in src


def test_direct_cli_help_imports_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_device_policy_rejects_unmarked_cpu_and_mps() -> None:
    module = _load_module()

    with pytest.raises(SystemExit, match="diagnostic-only"):
        module._require_device("cpu", allow_diagnostic_cpu=False)
    with pytest.raises(SystemExit, match="mps forbidden"):
        module._require_device("mps", allow_diagnostic_cpu=True)
    assert module._require_device("cpu", allow_diagnostic_cpu=True).type == "cpu"


def test_component_score_terms_use_contest_contributions() -> None:
    module = _load_module()
    pose = torch.tensor([0.0, 0.4])
    seg = torch.tensor([0.01, 0.02])

    terms = module.component_score_terms(pose, seg)

    assert torch.allclose(terms["posenet"], torch.sqrt(10.0 * pose + module.SCORE_EPS))
    assert torch.allclose(terms["segnet"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(terms["combined"], terms["posenet"] + terms["segnet"])


def test_component_formula_from_mean_distortions_uses_global_pose_sqrt() -> None:
    module = _load_module()

    values = module.component_formula_from_mean_distortions(
        pose_dist=0.4,
        seg_dist=0.02,
    )

    assert values["posenet"] == 0.4
    assert values["segnet"] == 0.02
    assert values["combined"] == pytest.approx(2.0 + (10.0 * 0.4) ** 0.5)


def test_make_sample_plan_is_deterministic_and_splits() -> None:
    module = _load_module()

    first = module.make_sample_plan(n_pairs=10, split_seed=7, holdout_fraction=0.2)
    second = module.make_sample_plan(n_pairs=10, split_seed=7, holdout_fraction=0.2)

    assert first == second
    assert len(first["holdout_pairs"]) == 2
    assert len(first["calibration_pairs"]) == 8
    assert len(first["split_hash"]) == 64


def test_make_sample_plan_for_indices_records_absolute_pair_ids() -> None:
    module = _load_module()

    plan = module.make_sample_plan_for_indices(
        pair_indices=[7, 11, 42, 99],
        split_seed=13,
        holdout_fraction=0.25,
    )

    recorded = {
        item["pair_index"]
        for item in plan["calibration_pairs"] + plan["holdout_pairs"]
    }
    assert recorded == {7, 11, 42, 99}
    for item in plan["calibration_pairs"] + plan["holdout_pairs"]:
        assert item["t"] == 2 * item["pair_index"]
        assert item["t1"] == 2 * item["pair_index"] + 1


def test_make_sample_plan_split_hash_matches_manifest_builder_contract() -> None:
    module = _load_module()
    plan = module.make_sample_plan_for_indices(
        pair_indices=[7, 11, 42, 99],
        split_seed=13,
        holdout_fraction=0.25,
    )

    expected_hash = hashlib.sha256(
        json.dumps(
            {
                "calibration_pairs": plan["calibration_pairs"],
                "holdout_pairs": plan["holdout_pairs"],
                "split_seed": plan["split_seed"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert plan["split_hash"] == expected_hash


def test_response_epsilons_allow_symmetric_directional_values() -> None:
    module = _load_module()

    assert module._parse_epsilons("-0.002,-0.001,0,0.001,0.002") == [
        -0.002,
        -0.001,
        0.0,
        0.001,
        0.002,
    ]
    assert module.DEFAULT_RESPONSE_EPSILONS == [
        -0.002,
        -0.001,
        -0.0005,
        0.0,
        0.0005,
        0.001,
        0.002,
    ]


def test_response_epsilon_metadata_classifies_symmetric_and_directional() -> None:
    module = _load_module()

    symmetric = module._response_epsilon_metadata([-0.1, 0.0, 0.1, 0.2])
    directional = module._response_epsilon_metadata([0.0, 0.1, 0.2])

    assert symmetric["response_kind"] == "symmetric"
    assert symmetric["symmetric_epsilon_pairs"] == 1
    assert symmetric["epsilon_ladder"] == [-0.1, 0.0, 0.1, 0.2]
    assert directional["response_kind"] == "directional"
    assert directional["directional_action"]["epsilon_values"] == [0.0, 0.1, 0.2]


def test_proxy_profile_outputs_cannot_assemble_promotable_manifest() -> None:
    module = _load_module()

    with pytest.raises(module.ComponentSensitivityProfileError, match="Fisher-proxy"):
        module.build_manifest_from_profile_outputs(
            summary={"device": "cuda", "promotion_eligible": False},
            checkpoint="renderer.bin",
            video_mkv="video.mkv",
            upstream_dir="upstream",
            archive="archive.zip",
            contest_auth_eval_json="contest_auth_eval.json",
            output="component_sensitivity_v1.json",
        )


def test_manifest_output_cli_rejected_before_profile(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def _unexpected_profile_call(**_: object) -> dict[str, object]:
        raise AssertionError("manifest-output validation must run before profiling")

    monkeypatch.setattr(module, "profile_component_sensitivity", _unexpected_profile_call)
    with pytest.raises(SystemExit):
        module.main(
            [
                "--checkpoint",
                "renderer.bin",
                "--video",
                "0.mkv",
                "--masks-mkv",
                "masks.mkv",
                "--output-dir",
                "profile",
                "--all-pairs",
                "--archive",
                "archive.zip",
                "--contest-auth-eval-json",
                "contest_auth_eval.json",
                "--manifest-output",
                "component_sensitivity_v1.json",
            ]
        )

    stderr = capsys.readouterr().err
    assert "diagnostic direct-renderer artifacts" in stderr
    assert "archive.zip -> inflate.sh -> upstream/evaluate.py" in stderr


def test_manifest_output_cli_rejected_even_for_finite_difference(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def _unexpected_profile_call(**_: object) -> dict[str, object]:
        raise AssertionError("manifest-output validation must run before profiling")

    monkeypatch.setattr(module, "profile_component_sensitivity", _unexpected_profile_call)
    with pytest.raises(SystemExit):
        module.main(
            [
                "--checkpoint",
                "renderer.bin",
                "--video",
                "0.mkv",
                "--masks-mkv",
                "masks.mkv",
                "--output-dir",
                "profile",
                "--all-pairs",
                "--promotion-finite-difference",
                "--archive",
                "archive.zip",
                "--contest-auth-eval-json",
                "contest_auth_eval.json",
                "--manifest-output",
                "component_sensitivity_v1.json",
            ]
        )

    stderr = capsys.readouterr().err
    assert "diagnostic direct-renderer artifacts" in stderr
    assert "archive.zip -> inflate.sh -> upstream/evaluate.py" in stderr


def test_select_top_channels_is_stable() -> None:
    module = _load_module()
    sensitivities = {
        "b.weight": torch.tensor([1.0, 3.0]),
        "a.weight": torch.tensor([3.0, 2.0]),
    }

    selected = module.select_top_channels(sensitivities, top_k=3)

    assert selected == [
        ("a.weight", 0, 3.0),
        ("b.weight", 1, 3.0),
        ("a.weight", 1, 2.0),
    ]


def test_channel_perturbation_restores_original_weights() -> None:
    module = _load_module()

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 1, bias=False)

    model = Toy()
    with torch.no_grad():
        model.conv.weight.copy_(torch.tensor([[[[2.0]]], [[[0.0]]]]))
    original = model.conv.weight.detach().clone()

    saved = module.apply_channel_perturbation(
        model,
        [("conv.weight", 0, 1.0), ("conv.weight", 1, 0.5)],
        epsilon=0.1,
    )

    assert not torch.equal(model.conv.weight, original)
    assert model.conv.weight[0, 0, 0, 0] > original[0, 0, 0, 0]
    assert model.conv.weight[1, 0, 0, 0] > original[1, 0, 0, 0]
    module.restore_perturbation(saved)
    assert torch.equal(model.conv.weight, original)


def test_channel_perturbation_accepts_negative_epsilon_for_symmetric_curves() -> None:
    module = _load_module()

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1, bias=False)

    model = Toy()
    with torch.no_grad():
        model.conv.weight.fill_(2.0)
    original = model.conv.weight.detach().clone()

    saved = module.apply_channel_perturbation(
        model,
        [("conv.weight", 0, 1.0)],
        epsilon=-0.1,
    )

    assert model.conv.weight[0, 0, 0, 0] < original[0, 0, 0, 0]
    module.restore_perturbation(saved)
    assert torch.equal(model.conv.weight, original)


def test_perturbation_basis_records_atoms_and_absolute_pairs() -> None:
    module = _load_module()
    plan = module.make_sample_plan_for_indices(
        pair_indices=[4, 8, 12, 16],
        split_seed=20260430,
        holdout_fraction=0.25,
    )

    payload = module.build_perturbation_basis_payload(
        selected_by_component={
            "posenet": [("conv.weight", 0, 3.0)],
            "segnet": [("conv.weight", 1, 2.0)],
            "combined": [("conv.weight", 2, 5.0)],
        },
        sample_plan=plan,
        response_epsilons=[-0.1, 0.0, 0.1],
        response_top_k=1,
        input_metadata={
            "checkpoint": "renderer.bin",
            "video_mkv": "0.mkv",
            "masks_mkv": "masks.mkv",
            "poses": "optimized_poses.pt",
            "upstream_dir": "upstream",
            "pair_weights": None,
            "n_pairs_total": 600,
            "n_pairs_selected": 4,
        },
    )

    assert payload["format"] == "perturbation_basis_v1"
    assert payload["split_hash"] == plan["split_hash"]
    assert set(payload["holdout_pair_indices"] + payload["calibration_pair_indices"]) == {
        4,
        8,
        12,
        16,
    }
    atom = payload["components"]["combined"]["atoms"][0]
    assert atom["key"] == "conv.weight"
    assert atom["channel"] == 2
    assert atom["atom_id"].startswith("combined:000000:")


def test_prediction_calibration_fits_symmetric_quadratic_response() -> None:
    module = _load_module()
    points = [
        {"epsilon": -0.1, "delta": 0.02},
        {"epsilon": 0.0, "delta": 0.0},
        {"epsilon": 0.1, "delta": 0.02},
    ]

    enriched, metrics = module._prediction_enriched_points(
        points,
        selected_channels=[("conv.weight", 0, 2.0)],
    )

    assert metrics["implemented"] is True
    assert metrics["fit_status"] == "least_squares_scale_fit"
    assert metrics["passed"] is True
    assert metrics["max_relative_error"] == pytest.approx(0.0)
    assert enriched[0]["prediction"]["raw_delta"] == pytest.approx(0.02)
    assert enriched[1]["prediction"]["raw_delta"] == 0.0


def test_write_response_curve_records_direct_renderer_blocker(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "segnet_response_curve.json"
    baseline = {"posenet": 0.01, "segnet": 0.02, "combined": 2.316227766}
    points = [
        {
            "epsilon": -0.1,
            "value": 0.03,
            "baseline": 0.02,
            "delta": 0.01,
            "all_components": {"posenet": 0.011, "segnet": 0.03, "combined": 3.331662479},
        },
        {
            "epsilon": 0.0,
            "value": 0.02,
            "baseline": 0.02,
            "delta": 0.0,
            "all_components": baseline,
        },
        {
            "epsilon": 0.1,
            "value": 0.04,
            "baseline": 0.02,
            "delta": 0.02,
            "all_components": {"posenet": 0.012, "segnet": 0.04, "combined": 4.346410162},
        },
    ]

    module._write_response_curve(
        path,
        component="segnet",
        baseline=baseline,
        points=points,
        selected_channels=[("conv.weight", 0, 1.0)],
        device="cuda",
    )

    payload = json.loads(path.read_text())
    assert payload["official_component_response"] is False
    assert payload["canonical_scorer_path"] is False
    assert payload["component_response_path"] == "direct_renderer_tensor_inprocess_scorer"
    assert payload["passed"] is False
    assert payload["component_readout"] == "official_argmax_disagreement"
    assert payload["response_kind"] == "symmetric"
    assert payload["symmetric_epsilon_pairs"] == 1
    assert payload["gate_spec"]["zero_repro_tolerance"] == 1e-7
    assert payload["gate_results"]["zero_repro"] is True
    assert payload["gate_results"]["prediction_error_gate_implemented"] is True
    assert payload["prediction_calibration"]["fit_status"] == "least_squares_scale_fit"
    assert payload["points"][0]["prediction"]["raw_model"] == (
        "epsilon^2 * sum_selected_channel_sensitivity"
    )
    assert payload["promotion_blockers"][0]["code"] == (
        "fisher_proxy_not_official_component_response"
    )
    assert any(
        blocker["code"] == "non_canonical_component_response"
        for blocker in payload["promotion_blockers"]
    )


def test_write_response_curve_blocks_direct_renderer_finite_difference_promotion(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "posenet_response_curve.json"
    baseline = {"posenet": 0.01, "segnet": 0.02, "combined": 2.316227766}
    points = [
        {
            "epsilon": -0.1,
            "value": 0.03,
            "baseline": 0.01,
            "delta": 0.02,
            "all_components": baseline,
        },
        {
            "epsilon": 0.0,
            "value": 0.01,
            "baseline": 0.01,
            "delta": 0.0,
            "all_components": baseline,
        },
        {
            "epsilon": 0.1,
            "value": 0.03,
            "baseline": 0.01,
            "delta": 0.02,
            "all_components": baseline,
        },
    ]

    module._write_response_curve(
        path,
        component="posenet",
        baseline=baseline,
        points=points,
        selected_channels=[("conv.weight", 0, 2.0)],
        device="cuda",
        promotion_eligible=True,
        sensitivity_source="official_cuda_finite_difference_component_response",
    )

    payload = json.loads(path.read_text())
    assert payload["promotion_eligible"] is False
    assert payload["passed"] is False
    assert payload["evidence_grade"] == "diagnostic_cuda_direct_renderer_component_response"
    assert payload["official_component_response"] is False
    assert payload["canonical_scorer_path"] is False
    assert payload["promotion_blockers"][0]["code"] == "not_canonical_inflate_eval_path"
    assert payload["sensitivity_source"] == (
        "official_cuda_finite_difference_component_response"
    )


def test_write_response_curve_blocks_zero_signal_promotion(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "posenet_response_curve.json"
    baseline = {"posenet": 0.01, "segnet": 0.02, "combined": 2.316227766}
    points = [
        {
            "epsilon": -0.1,
            "value": 0.01,
            "baseline": 0.01,
            "delta": 0.0,
            "all_components": baseline,
        },
        {
            "epsilon": 0.0,
            "value": 0.01,
            "baseline": 0.01,
            "delta": 0.0,
            "all_components": baseline,
        },
        {
            "epsilon": 0.1,
            "value": 0.01,
            "baseline": 0.01,
            "delta": 0.0,
            "all_components": baseline,
        },
    ]

    module._write_response_curve(
        path,
        component="posenet",
        baseline=baseline,
        points=points,
        selected_channels=[("conv.weight", 0, 2.0)],
        device="cuda",
        promotion_eligible=True,
        sensitivity_source="official_cuda_finite_difference_component_response",
    )

    payload = json.loads(path.read_text())
    assert payload["promotion_eligible"] is False
    assert payload["passed"] is False
    assert payload["gate_results"]["signal_present"] is False
    assert payload["gate_results"]["promotion_gate_passed"] is False
    blocker_codes = [blocker["code"] for blocker in payload["promotion_blockers"]]
    assert blocker_codes[0] == "not_canonical_inflate_eval_path"
    assert "finite_difference_response_gate_failed" in blocker_codes


def test_official_component_distortions_use_argmax_disagreement() -> None:
    module = _load_module()

    class ConstantRenderer(nn.Module):
        def forward(self, masks_t: torch.Tensor, masks_t1: torch.Tensor, **_: object) -> torch.Tensor:
            bsz, height, width = masks_t.shape
            return torch.ones((bsz, 2, height, width, 3), dtype=torch.float32)

    class MeanPoseNet(nn.Module):
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            mean = x.float().mean(dim=(1, 2, 3, 4), keepdim=False).reshape(-1, 1)
            return {"pose": torch.cat([mean.repeat(1, 6), torch.zeros_like(mean).repeat(1, 6)], dim=1)}

    class ThresholdSegNet(nn.Module):
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            signal = x[:, 0]
            return torch.stack([-signal, signal], dim=1)

    masks = torch.zeros((2, 2, 2), dtype=torch.long)
    gt = torch.zeros((2, 3, 2, 2), dtype=torch.float32)

    terms = module._official_component_distortions_from_pairs(
        model=ConstantRenderer(),
        batch=[0],
        masks_cpu=masks,
        gt_frames_cpu=gt,
        poses=None,
        posenet=MeanPoseNet(),
        segnet=ThresholdSegNet(),
        device=torch.device("cpu"),
        zoom_warp=None,
    )

    assert terms["posenet"].shape == (1,)
    assert terms["posenet"].item() > 0.0
    assert terms["segnet"].item() == 1.0


def test_finite_difference_component_maps_measure_conv_channels() -> None:
    module = _load_module()

    class ToyRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1, bias=False)
            with torch.no_grad():
                self.conv.weight.fill_(1.0)

        def forward(self, masks_t: torch.Tensor, masks_t1: torch.Tensor, **_: object) -> torch.Tensor:
            del masks_t1
            frame = self.conv(masks_t.float().unsqueeze(1)).repeat(1, 3, 1, 1)
            pair = torch.stack([frame, frame], dim=1)
            return pair.permute(0, 1, 3, 4, 2)

    class MeanPoseNet(nn.Module):
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            mean = x.float().mean(dim=(1, 2, 3, 4), keepdim=False).reshape(-1, 1)
            return {"pose": torch.cat([mean.repeat(1, 6), torch.zeros_like(mean).repeat(1, 6)], dim=1)}

    class ThresholdSegNet(nn.Module):
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            signal = x[:, 0]
            return torch.stack([-signal, signal], dim=1)

    model = ToyRenderer()
    masks = torch.ones((2, 2, 2), dtype=torch.long)
    gt = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    baseline = module._evaluate_component_means(
        model=model,
        pair_indices=[0],
        masks_cpu=masks,
        gt_frames_cpu=gt,
        poses=None,
        posenet=MeanPoseNet(),
        segnet=ThresholdSegNet(),
        device=torch.device("cpu"),
        zoom_warp=None,
        pair_batch=1,
    )

    maps = module._finite_difference_component_channel_maps(
        model=model,
        pair_indices=[0],
        baseline=baseline,
        masks_cpu=masks,
        gt_frames_cpu=gt,
        poses=None,
        posenet=MeanPoseNet(),
        segnet=ThresholdSegNet(),
        device=torch.device("cpu"),
        zoom_warp=None,
        pair_batch=1,
        epsilon=0.1,
    )

    assert set(maps) == {"posenet", "segnet", "combined"}
    assert maps["posenet"]["conv.weight"].shape == (1,)
    assert torch.isfinite(maps["combined"]["conv.weight"]).all()
    assert maps["combined"]["conv.weight"].item() >= 0.0


def test_finite_difference_shard_plan_partitions_channels() -> None:
    module = _load_module()

    model = nn.Sequential(
        nn.Conv2d(1, 3, 1, bias=False),
        nn.Conv2d(3, 2, 1, bias=False),
    )

    plans = [
        module._finite_difference_shard_plan(
            model,
            shard_index=index,
            shard_count=3,
        )
        for index in range(3)
    ]
    all_refs = module._refs_from_payload(plans[0]["all_channel_refs"], label="all")
    assigned = [
        ref
        for plan in plans
        for ref in module._refs_from_payload(plan["assigned_channel_refs"], label="assigned")
    ]

    assert len(all_refs) == 5
    assert sorted(assigned) == sorted(all_refs)
    assert len(set(assigned)) == len(assigned)
    assert {plan["all_channel_sha256"] for plan in plans} == {module._channel_ref_sha256(all_refs)}
    assert all(plan["merge_required_for_certification_handoff"] is True for plan in plans)


def test_finite_difference_component_maps_can_measure_owned_channel_subset() -> None:
    module = _load_module()

    class ToyRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 1, bias=False)
            with torch.no_grad():
                self.conv.weight.fill_(1.0)

        def forward(self, masks_t: torch.Tensor, masks_t1: torch.Tensor, **_: object) -> torch.Tensor:
            del masks_t1
            channels = self.conv(masks_t.float().unsqueeze(1))
            frame = channels[:, :1].repeat(1, 3, 1, 1)
            pair = torch.stack([frame, frame], dim=1)
            return pair.permute(0, 1, 3, 4, 2)

    class MeanPoseNet(nn.Module):
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            mean = x.float().mean(dim=(1, 2, 3, 4), keepdim=False).reshape(-1, 1)
            return {"pose": torch.cat([mean.repeat(1, 6), torch.zeros_like(mean).repeat(1, 6)], dim=1)}

    class ThresholdSegNet(nn.Module):
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            signal = x[:, 0]
            return torch.stack([-signal, signal], dim=1)

    model = ToyRenderer()
    masks = torch.ones((2, 2, 2), dtype=torch.long)
    gt = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    baseline = module._evaluate_component_means(
        model=model,
        pair_indices=[0],
        masks_cpu=masks,
        gt_frames_cpu=gt,
        poses=None,
        posenet=MeanPoseNet(),
        segnet=ThresholdSegNet(),
        device=torch.device("cpu"),
        zoom_warp=None,
        pair_batch=1,
    )

    maps = module._finite_difference_component_channel_maps(
        model=model,
        pair_indices=[0],
        baseline=baseline,
        masks_cpu=masks,
        gt_frames_cpu=gt,
        poses=None,
        posenet=MeanPoseNet(),
        segnet=ThresholdSegNet(),
        device=torch.device("cpu"),
        zoom_warp=None,
        pair_batch=1,
        epsilon=0.1,
        channel_refs=[("conv.weight", 1)],
    )

    assert maps["combined"]["conv.weight"][0].item() == 0.0
    assert torch.isfinite(maps["combined"]["conv.weight"][1])


def _write_fd_shard_dir(
    module,
    root: Path,
    *,
    shard_index: int,
    shard_count: int,
    refs: list[tuple[str, int]],
    all_refs: list[tuple[str, int]],
    value: float,
) -> dict[str, object]:
    shard = {
        "schema": module.FINITE_DIFFERENCE_SHARD_SCHEMA,
        "is_shard": True,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_channel_count": len(refs),
        "all_channel_count": len(all_refs),
        "assigned_channel_refs": module._channel_ref_payload(refs),
        "all_channel_refs": module._channel_ref_payload(all_refs),
        "all_channel_sha256": module._channel_ref_sha256(all_refs),
        "assigned_channel_sha256": module._channel_ref_sha256(refs),
        "merge_required_for_certification_handoff": True,
    }
    summary = {
        "tool": "experiments/profile_component_sensitivity.py",
        "device": "cuda",
        "sensitivity_source": "direct_renderer_cuda_finite_difference_component_response",
        "n_pairs_total": 600,
        "n_pairs_selected": 600,
        "n_pairs_calibration": 480,
        "n_pairs_holdout": 120,
        "split_seed": 20260430,
        "finite_difference_epsilon": 0.001,
        "component_response_path": "direct_renderer_tensor_inprocess_scorer",
        "finite_difference_shard": shard,
    }
    root.mkdir(parents=True)
    (root / "component_sensitivity_profile_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for component in module.COMPONENT_OUTPUTS:
        tensor = torch.zeros(2)
        for key, channel in refs:
            assert key == "a.weight"
            tensor[channel] = value + channel
        metadata = {
            "device": "cuda",
            "component": component,
            "scorer_target": component,
            "score_claim": False,
            "promotion_eligible": False,
            "official_component_response": False,
            "canonical_scorer_path": False,
            "sensitivity_source": "direct_renderer_cuda_finite_difference_component_response",
            "finite_difference_shard": shard,
        }
        module.save_sensitivity_map(root / f"{component}_sensitivity_map.pt", {"a.weight": tensor}, metadata=metadata)
        module.save_sensitivity_map(
            root / f"{component}_holdout_sensitivity_map.pt",
            {"a.weight": tensor + 10.0},
            metadata={**metadata, "split": "holdout"},
        )
    return summary


def test_merge_finite_difference_shards_reconstructs_full_maps(tmp_path: Path) -> None:
    module = _load_module()
    all_refs = [("a.weight", 0), ("a.weight", 1)]
    shard0 = tmp_path / "s0"
    shard1 = tmp_path / "s1"
    _write_fd_shard_dir(module, shard0, shard_index=0, shard_count=2, refs=[all_refs[0]], all_refs=all_refs, value=3.0)
    _write_fd_shard_dir(module, shard1, shard_index=1, shard_count=2, refs=[all_refs[1]], all_refs=all_refs, value=3.0)

    calibration, holdout, merge = module._merge_finite_difference_shard_maps([shard1, shard0])

    assert merge["coverage"] == "exactly_once"
    assert merge["source_shard_count"] == 2
    assert calibration["combined"]["a.weight"].tolist() == [3.0, 4.0]
    assert holdout["combined"]["a.weight"].tolist() == [13.0, 14.0]


def test_merge_finite_difference_shards_rejects_duplicate_atoms(tmp_path: Path) -> None:
    module = _load_module()
    all_refs = [("a.weight", 0), ("a.weight", 1)]
    shard0 = tmp_path / "s0"
    shard1 = tmp_path / "s1"
    _write_fd_shard_dir(module, shard0, shard_index=0, shard_count=2, refs=[all_refs[0]], all_refs=all_refs, value=3.0)
    _write_fd_shard_dir(module, shard1, shard_index=1, shard_count=2, refs=[all_refs[0]], all_refs=all_refs, value=5.0)

    with pytest.raises(ValueError, match="duplicate finite-difference shard channel"):
        module._merge_finite_difference_shard_maps([shard0, shard1])


def test_promotion_finite_difference_requires_exact_1200_frames() -> None:
    module = _load_module()

    module._require_full_contest_frame_count(
        n_frames=1200,
        n_mask_frames=1200,
        promotion_finite_difference=True,
    )
    module._require_full_contest_frame_count(
        n_frames=1201,
        n_mask_frames=1201,
        promotion_finite_difference=False,
    )
    with pytest.raises(SystemExit, match="exact 1200 contest frames"):
        module._require_full_contest_frame_count(
            n_frames=1201,
            n_mask_frames=1201,
            promotion_finite_difference=True,
        )
    with pytest.raises(SystemExit, match="masks/video frame counts"):
        module._require_full_contest_frame_count(
            n_frames=1200,
            n_mask_frames=1199,
            promotion_finite_difference=True,
        )


def test_stability_json_records_thresholds_and_rank_metadata(tmp_path: Path) -> None:
    module = _load_module()
    cal = {
        "a.weight": torch.tensor([4.0, 3.0, 2.0, 1.0]),
        "b.weight": torch.tensor([2.0, 1.0]),
    }
    holdout = {
        "a.weight": torch.tensor([4.1, 3.1, 1.9, 0.9]),
        "b.weight": torch.tensor([2.1, 0.9]),
    }
    calibration_maps = {component: cal for component in module.COMPONENT_OUTPUTS}
    holdout_maps = {component: holdout for component in module.COMPONENT_OUTPUTS}
    path = tmp_path / "stability.json"

    payload = module._write_stability_json(
        path,
        calibration_maps=calibration_maps,
        holdout_maps=holdout_maps,
        top_k=2,
    )

    saved = json.loads(path.read_text())
    assert saved == payload
    assert saved["thresholds"]["cv_max"] == 0.35
    assert saved["passed"] is True
    assert saved["component_passed"]["segnet"] is True
    assert saved["correlation"]["posenet"]["spearman"] > 0.0
    assert saved["top_fraction"]["combined"]["top_10pct"]["k"] == 1
    assert saved["counts"]["posenet"]["calibration"]["channels"] == 6


def test_accumulate_component_fisher_shapes() -> None:
    module = _load_module()
    layer = nn.Linear(2, 1, bias=False)
    x = torch.tensor([[1.0, 2.0]])
    y = layer(x).reshape(())
    losses = {
        "posenet": y,
        "segnet": y * 2.0,
        "combined": y * 3.0,
    }
    accum = {
        component: {"layer.weight": torch.zeros_like(layer.weight, dtype=torch.float64)}
        for component in module.COMPONENT_OUTPUTS
    }

    module.accumulate_component_fisher(losses, {"layer.weight": layer.weight}, accum)

    assert accum["posenet"]["layer.weight"].shape == layer.weight.shape
    assert torch.all(accum["combined"]["layer.weight"] >= accum["segnet"]["layer.weight"])


def test_manifest_handoff_requires_cuda_summary(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(module.ComponentSensitivityProfileError, match="CUDA-authored"):
        module.build_manifest_from_profile_outputs(
            summary={"device": "cpu"},
            checkpoint="renderer.bin",
            video_mkv="0.mkv",
            upstream_dir="upstream",
            archive="archive.zip",
            contest_auth_eval_json="contest_auth_eval.json",
            output=str(tmp_path / "manifest.json"),
        )
