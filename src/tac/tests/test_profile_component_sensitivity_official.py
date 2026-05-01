from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_component_sensitivity_official.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("profile_component_sensitivity_official", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_archive(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _write_eval_json(
    path: Path,
    *,
    archive: Path,
    pose: float,
    seg: float,
    device: str = "cuda",
    n_samples: int = 600,
) -> Path:
    rate = archive.stat().st_size / 37_545_489
    score = 100.0 * seg + (10.0 * pose) ** 0.5 + 25.0 * rate
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "final_score": score,
                "avg_posenet_dist": pose,
                "avg_segnet_dist": seg,
                "rate_unscaled": rate,
                "score_recomputed_from_components": score,
                "archive_size_bytes": archive.stat().st_size,
                "n_samples": n_samples,
                "provenance": {
                    "device": device,
                    "archive_sha256": _sha(archive),
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _write_basic_inputs(root: Path) -> dict[str, Path]:
    upstream = root / "upstream"
    upstream.mkdir()
    (upstream / "evaluate.py").write_text("print('unused')\n")
    names = root / "public_test_video_names.txt"
    names.write_text("0.mkv\n")
    inflate = root / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nexit 99\n")
    contest_script = root / "contest_auth_eval.py"
    contest_script.write_text("raise SystemExit('unused')\n")

    baseline = _write_archive(root / "baseline.zip", b"baseline")
    neg = _write_archive(root / "neg.zip", b"negative")
    pos = _write_archive(root / "pos.zip", b"positive")

    eval_dir = root / "evals_in"
    eval_dir.mkdir()
    _write_eval_json(eval_dir / "baseline.json", archive=baseline, pose=0.04, seg=0.01)
    _write_eval_json(eval_dir / "neg.json", archive=neg, pose=0.041, seg=0.011)
    _write_eval_json(eval_dir / "pos.json", archive=pos, pose=0.043, seg=0.013)
    return {
        "upstream": upstream,
        "video_names": names,
        "inflate": inflate,
        "contest_script": contest_script,
        "baseline": baseline,
        "neg": neg,
        "pos": pos,
        "baseline_json": eval_dir / "baseline.json",
        "neg_json": eval_dir / "neg.json",
        "pos_json": eval_dir / "pos.json",
    }


def _combined_delta(*, baseline_pose: float, baseline_seg: float, pose: float, seg: float) -> float:
    return 100.0 * (seg - baseline_seg) + (10.0 * pose) ** 0.5 - (10.0 * baseline_pose) ** 0.5


def test_help_imports_directly() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--perturbation-plan" in result.stdout
    assert "--same-run-zero-baseline" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_contest_auth_eval_command_has_single_timeout_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    seen_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen_cmds.append(list(cmd))
        (tmp_path / "point" / "contest_auth_eval.json").write_text("{}\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    out_json = module._run_contest_auth_eval(
        archive=tmp_path / "archive.zip",
        point_dir=tmp_path / "point",
        contest_auth_eval_script=tmp_path / "contest_auth_eval.py",
        inflate_sh=tmp_path / "inflate.sh",
        upstream_dir=tmp_path / "upstream",
        video_names_file=tmp_path / "names.txt",
        device="cuda",
        inflate_timeout=17,
        evaluate_timeout=23,
    )

    assert out_json == tmp_path / "point" / "contest_auth_eval.json"
    assert len(seen_cmds) == 1
    cmd = seen_cmds[0]
    assert cmd.count("--inflate-timeout") == 1
    assert cmd[cmd.index("--inflate-timeout") + 1] == "17"
    assert cmd.count("--evaluate-timeout") == 1
    assert cmd[cmd.index("--evaluate-timeout") + 1] == "23"


def test_produces_passed_official_response_curves_from_existing_eval_json(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "perturbation": {
                    "basis": "synthetic_signed_archive_delta",
                    "epsilon_units": "test",
                },
                "points": [
                    {
                        "epsilon": -0.001,
                        "archive": str(paths["neg"]),
                        "contest_auth_eval_json": str(paths["neg_json"]),
                        "predicted_delta": {
                            "posenet": 0.001,
                            "segnet": 0.001,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.041,
                                seg=0.011,
                            ),
                        },
                    },
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                        "predicted_delta": {
                            "posenet": 0.003,
                            "segnet": 0.003,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.043,
                                seg=0.013,
                            ),
                        },
                    },
                ],
            }
        )
        + "\n"
    )

    summary = module.produce_official_component_response_curves(
        baseline_archive=paths["baseline"],
        baseline_contest_auth_eval_json=paths["baseline_json"],
        perturbation_plan=plan,
        output_dir=tmp_path / "out",
        contest_auth_eval_script=paths["contest_script"],
        inflate_sh=paths["inflate"],
        upstream_dir=paths["upstream"],
        video_names_file=paths["video_names"],
        device="cuda",
        inflate_timeout=1,
        evaluate_timeout=1,
        max_relative_error=1e-12,
        zero_repro_tolerance=1e-12,
        min_observed_delta=1e-12,
        allow_directional=False,
    )

    assert summary["promotion_eligible"] is True
    assert summary["response_coverage"]["response_kind"] == "symmetric"
    for component in ("posenet", "segnet", "combined"):
        curve = json.loads(Path(summary["response_curve_paths"][component]).read_text())
        assert curve["official_component_response"] is True
        assert curve["canonical_scorer_path"] is True
        assert curve["component_response_path"] == "archive_zip_inflate_sh_upstream_evaluate_py"
        assert curve["promotion_eligible"] is True
        assert curve["passed"] is True
        assert curve["promotion_blockers"] == []
        assert curve["count"] == 3
        assert curve["holdout_error"] == pytest.approx(0.0)
        assert curve["gate_results"]["prediction_error_passed"] is True
        assert curve["symmetric_epsilon_pairs"] == 1
        assert curve["epsilon_ladder"] == [-0.001, 0.0, 0.001]


def test_nonnegative_magnitude_predictions_compare_against_abs_response_delta(
    tmp_path: Path,
) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    _write_eval_json(paths["neg_json"], archive=paths["neg"], pose=0.039, seg=0.009)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "perturbation": {
                    "prediction_model": {
                        "prediction_delta_semantics": "nonnegative_component_delta_magnitude",
                        "prediction_error_mode": "absolute_magnitude",
                    },
                },
                "points": [
                    {
                        "epsilon": -0.001,
                        "archive": str(paths["neg"]),
                        "contest_auth_eval_json": str(paths["neg_json"]),
                        "predicted_delta": {
                            "posenet": 0.001,
                            "segnet": 0.001,
                            "combined": abs(
                                _combined_delta(
                                    baseline_pose=0.04,
                                    baseline_seg=0.01,
                                    pose=0.039,
                                    seg=0.009,
                                )
                            ),
                        },
                    },
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                        "predicted_delta": {
                            "posenet": 0.003,
                            "segnet": 0.003,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.043,
                                seg=0.013,
                            ),
                        },
                    },
                ],
            }
        )
        + "\n"
    )

    summary = module.produce_official_component_response_curves(
        baseline_archive=paths["baseline"],
        baseline_contest_auth_eval_json=paths["baseline_json"],
        perturbation_plan=plan,
        output_dir=tmp_path / "out",
        contest_auth_eval_script=paths["contest_script"],
        inflate_sh=paths["inflate"],
        upstream_dir=paths["upstream"],
        video_names_file=paths["video_names"],
        device="cuda",
        inflate_timeout=1,
        evaluate_timeout=1,
        max_relative_error=1e-12,
        zero_repro_tolerance=1e-12,
        min_observed_delta=1e-12,
        allow_directional=False,
    )

    assert summary["promotion_eligible"] is True
    curve = json.loads(Path(summary["response_curve_paths"]["posenet"]).read_text())
    neg_point = next(point for point in curve["points"] if point["epsilon"] == -0.001)
    assert neg_point["delta"] == pytest.approx(-0.001)
    assert neg_point["prediction"]["prediction_error_mode"] == "absolute_magnitude"
    assert neg_point["prediction"]["observed_delta_for_error"] == pytest.approx(0.001)
    assert neg_point["prediction"]["relative_error"] == pytest.approx(0.0)
    assert curve["holdout_error_kind"].endswith("_abs_magnitude")


def test_explicit_baseline_eval_json_overrides_stale_plan_metadata(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "baseline_contest_auth_eval_json": str(tmp_path / "missing_host_absolute_eval.json"),
                "points": [
                    {
                        "epsilon": -0.001,
                        "archive": str(paths["neg"]),
                        "contest_auth_eval_json": str(paths["neg_json"]),
                        "predicted_delta": {"posenet": 0.001, "segnet": 0.001, "combined": 0.1},
                    },
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                        "predicted_delta": {"posenet": 0.003, "segnet": 0.003, "combined": 0.3},
                    },
                ],
            }
        )
        + "\n"
    )

    summary = module.produce_official_component_response_curves(
        baseline_archive=paths["baseline"],
        baseline_contest_auth_eval_json=paths["baseline_json"],
        perturbation_plan=plan,
        output_dir=tmp_path / "out",
        contest_auth_eval_script=paths["contest_script"],
        inflate_sh=paths["inflate"],
        upstream_dir=paths["upstream"],
        video_names_file=paths["video_names"],
        device="cuda",
        inflate_timeout=1,
        evaluate_timeout=1,
        max_relative_error=10.0,
        zero_repro_tolerance=1e-12,
        min_observed_delta=1e-12,
        allow_directional=False,
    )

    assert summary["baseline_contest_auth_eval_json"]["path"].endswith("evals/point_001_eps_p0/contest_auth_eval.json")


def test_require_passed_reruns_zero_baseline_in_same_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    _write_eval_json(paths["baseline_json"], archive=paths["baseline"], pose=0.04, seg=0.01)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "points": [
                    {
                        "epsilon": -0.001,
                        "archive": str(paths["neg"]),
                        "contest_auth_eval_json": str(paths["neg_json"]),
                        "predicted_delta": {
                            "posenet": 0.001,
                            "segnet": 0.001,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.041,
                                seg=0.011,
                            ),
                        },
                    },
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                        "predicted_delta": {
                            "posenet": 0.003,
                            "segnet": 0.003,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.043,
                                seg=0.013,
                            ),
                        },
                    },
                ],
            }
        )
        + "\n"
    )
    eval_calls: list[Path] = []

    def fake_run_contest_auth_eval(*, archive: Path, point_dir: Path, **kwargs) -> Path:
        eval_calls.append(archive)
        point_dir.mkdir(parents=True, exist_ok=True)
        return _write_eval_json(
            point_dir / "contest_auth_eval.json",
            archive=archive,
            pose=0.04,
            seg=0.01,
        )

    monkeypatch.setattr(module, "_run_contest_auth_eval", fake_run_contest_auth_eval)

    summary = module.produce_official_component_response_curves(
        baseline_archive=paths["baseline"],
        baseline_contest_auth_eval_json=paths["baseline_json"],
        perturbation_plan=plan,
        output_dir=tmp_path / "out",
        contest_auth_eval_script=paths["contest_script"],
        inflate_sh=paths["inflate"],
        upstream_dir=paths["upstream"],
        video_names_file=paths["video_names"],
        device="cuda",
        inflate_timeout=1,
        evaluate_timeout=1,
        max_relative_error=1e-12,
        zero_repro_tolerance=1e-12,
        min_observed_delta=1e-12,
        allow_directional=False,
        require_passed=True,
    )

    assert eval_calls == [paths["baseline"].resolve()]
    assert summary["promotion_eligible"] is True
    assert summary["same_run_zero_baseline"] is True
    assert summary["baseline_contest_auth_eval_json"]["path"].endswith(
        "evals/point_001_eps_p0/contest_auth_eval.json"
    )
    assert summary["external_baseline_contest_auth_eval_json"]["path"] == str(
        paths["baseline_json"].resolve()
    )
    curve = json.loads(Path(summary["response_curve_paths"]["posenet"]).read_text())
    assert curve["baseline"]["values"]["posenet"] == pytest.approx(0.04)
    assert curve["baseline"]["values"]["segnet"] == pytest.approx(0.01)
    assert curve["gate_results"]["external_baseline_repro"] is True
    assert curve["gate_results"]["external_baseline_repro_error"] == pytest.approx(0.0)
    zero_point = [point for point in curve["points"] if point["epsilon"] == 0.0][0]
    assert zero_point["point_metadata"]["same_run_zero_baseline"] is True
    assert "external_zero_contest_auth_eval_json_ignored_for_curve" in zero_point["point_metadata"]


def test_require_passed_fails_on_external_baseline_component_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    _write_eval_json(paths["baseline_json"], archive=paths["baseline"], pose=0.20, seg=0.20)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "points": [
                    {
                        "epsilon": -0.001,
                        "archive": str(paths["neg"]),
                        "contest_auth_eval_json": str(paths["neg_json"]),
                        "predicted_delta": {
                            "posenet": 0.001,
                            "segnet": 0.001,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.041,
                                seg=0.011,
                            ),
                        },
                    },
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                        "predicted_delta": {
                            "posenet": 0.003,
                            "segnet": 0.003,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.043,
                                seg=0.013,
                            ),
                        },
                    },
                ],
            }
        )
        + "\n"
    )

    def fake_run_contest_auth_eval(*, archive: Path, point_dir: Path, **kwargs) -> Path:
        point_dir.mkdir(parents=True, exist_ok=True)
        return _write_eval_json(
            point_dir / "contest_auth_eval.json",
            archive=archive,
            pose=0.04,
            seg=0.01,
        )

    monkeypatch.setattr(module, "_run_contest_auth_eval", fake_run_contest_auth_eval)

    summary = module.produce_official_component_response_curves(
        baseline_archive=paths["baseline"],
        baseline_contest_auth_eval_json=paths["baseline_json"],
        perturbation_plan=plan,
        output_dir=tmp_path / "out",
        contest_auth_eval_script=paths["contest_script"],
        inflate_sh=paths["inflate"],
        upstream_dir=paths["upstream"],
        video_names_file=paths["video_names"],
        device="cuda",
        inflate_timeout=1,
        evaluate_timeout=1,
        max_relative_error=1e-12,
        zero_repro_tolerance=1e-12,
        min_observed_delta=1e-12,
        allow_directional=False,
        require_passed=True,
    )

    assert summary["promotion_eligible"] is False
    curve = json.loads(Path(summary["response_curve_paths"]["posenet"]).read_text())
    assert curve["gate_results"]["external_baseline_repro"] is False
    blocker_codes = {item["code"] for item in curve["promotion_blockers"]}
    assert "external_baseline_reproduction_failed" in blocker_codes


def test_missing_prediction_deltas_fail_closed(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "epsilon": -0.001,
                        "archive": str(paths["neg"]),
                        "contest_auth_eval_json": str(paths["neg_json"]),
                    },
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                    },
                ]
            }
        )
        + "\n"
    )

    summary = module.produce_official_component_response_curves(
        baseline_archive=paths["baseline"],
        baseline_contest_auth_eval_json=paths["baseline_json"],
        perturbation_plan=plan,
        output_dir=tmp_path / "out",
        contest_auth_eval_script=paths["contest_script"],
        inflate_sh=paths["inflate"],
        upstream_dir=paths["upstream"],
        video_names_file=paths["video_names"],
        device="cuda",
        inflate_timeout=1,
        evaluate_timeout=1,
        max_relative_error=0.35,
        zero_repro_tolerance=1e-7,
        min_observed_delta=1e-12,
        allow_directional=False,
    )

    assert summary["promotion_eligible"] is False
    curve = json.loads(Path(summary["response_curve_paths"]["segnet"]).read_text())
    assert curve["official_component_response"] is True
    assert curve["promotion_eligible"] is False
    assert curve["passed"] is False
    blocker_codes = {item["code"] for item in curve["promotion_blockers"]}
    assert "missing_prediction_deltas" in blocker_codes
    assert "prediction_error_gate_failed" in blocker_codes


def test_require_passed_preflight_rejects_missing_prediction_deltas_before_eval(
    tmp_path: Path,
) -> None:
    paths = _write_basic_inputs(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "epsilon": -0.001,
                        "archive": str(paths["neg"]),
                        "contest_auth_eval_json": str(paths["neg_json"]),
                    },
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                    },
                ]
            }
        )
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-archive",
            str(paths["baseline"]),
            "--baseline-contest-auth-eval-json",
            str(paths["baseline_json"]),
            "--perturbation-plan",
            str(plan),
            "--output-dir",
            str(tmp_path / "out"),
            "--contest-auth-eval-script",
            str(paths["contest_script"]),
            "--inflate-sh",
            str(paths["inflate"]),
            "--upstream",
            str(paths["upstream"]),
            "--video-names-file",
            str(paths["video_names"]),
            "--require-passed",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "promotion preflight failed before evaluate_points" in result.stderr
    assert "posenet and segnet predicted_delta" in result.stderr
    assert "official component response gates did not pass" not in result.stderr
    assert not (tmp_path / "out" / "evals").exists()


def test_contest_eval_archive_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    paths["baseline_json"].write_text(
        json.dumps(
            {
                "avg_posenet_dist": 0.04,
                "avg_segnet_dist": 0.01,
                "archive_size_bytes": paths["baseline"].stat().st_size,
                "n_samples": 600,
                "provenance": {"device": "cuda", "archive_sha256": "a" * 64},
            }
        )
        + "\n"
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                        "predicted_delta": {"posenet": 0.003, "segnet": 0.003},
                    }
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(module.OfficialComponentResponseError, match="archive_sha256"):
        module.produce_official_component_response_curves(
            baseline_archive=paths["baseline"],
            baseline_contest_auth_eval_json=paths["baseline_json"],
            perturbation_plan=plan,
            output_dir=tmp_path / "out",
            contest_auth_eval_script=paths["contest_script"],
            inflate_sh=paths["inflate"],
            upstream_dir=paths["upstream"],
            video_names_file=paths["video_names"],
            device="cuda",
            inflate_timeout=1,
            evaluate_timeout=1,
            max_relative_error=0.35,
            zero_repro_tolerance=1e-7,
            min_observed_delta=1e-12,
            allow_directional=False,
        )


def test_baseline_archive_at_nonzero_epsilon_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["baseline"]),
                        "contest_auth_eval_json": str(paths["baseline_json"]),
                        "predicted_delta": {
                            "posenet": 0.0,
                            "segnet": 0.0,
                            "combined": 0.0,
                        },
                    },
                    {
                        "epsilon": 0.002,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                        "predicted_delta": {
                            "posenet": 0.003,
                            "segnet": 0.003,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.043,
                                seg=0.013,
                            ),
                        },
                    },
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(module.OfficialComponentResponseError, match="nonzero epsilon"):
        module.produce_official_component_response_curves(
            baseline_archive=paths["baseline"],
            baseline_contest_auth_eval_json=paths["baseline_json"],
            perturbation_plan=plan,
            output_dir=tmp_path / "out",
            contest_auth_eval_script=paths["contest_script"],
            inflate_sh=paths["inflate"],
            upstream_dir=paths["upstream"],
            video_names_file=paths["video_names"],
            device="cuda",
            inflate_timeout=1,
            evaluate_timeout=1,
            max_relative_error=0.35,
            zero_repro_tolerance=1e-7,
            min_observed_delta=1e-12,
            allow_directional=True,
        )


def test_one_sided_response_requires_explicit_directional_mode(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_basic_inputs(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "epsilon": 0.001,
                        "archive": str(paths["pos"]),
                        "contest_auth_eval_json": str(paths["pos_json"]),
                        "predicted_delta": {
                            "posenet": 0.003,
                            "segnet": 0.003,
                            "combined": _combined_delta(
                                baseline_pose=0.04,
                                baseline_seg=0.01,
                                pose=0.043,
                                seg=0.013,
                            ),
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    summary = module.produce_official_component_response_curves(
        baseline_archive=paths["baseline"],
        baseline_contest_auth_eval_json=paths["baseline_json"],
        perturbation_plan=plan,
        output_dir=tmp_path / "out",
        contest_auth_eval_script=paths["contest_script"],
        inflate_sh=paths["inflate"],
        upstream_dir=paths["upstream"],
        video_names_file=paths["video_names"],
        device="cuda",
        inflate_timeout=1,
        evaluate_timeout=1,
        max_relative_error=1e-12,
        zero_repro_tolerance=1e-12,
        min_observed_delta=1e-12,
        allow_directional=False,
    )
    assert summary["promotion_eligible"] is False
    assert summary["response_coverage_passed"] is False

    directional = module.produce_official_component_response_curves(
        baseline_archive=paths["baseline"],
        baseline_contest_auth_eval_json=paths["baseline_json"],
        perturbation_plan=plan,
        output_dir=tmp_path / "out_directional",
        contest_auth_eval_script=paths["contest_script"],
        inflate_sh=paths["inflate"],
        upstream_dir=paths["upstream"],
        video_names_file=paths["video_names"],
        device="cuda",
        inflate_timeout=1,
        evaluate_timeout=1,
        max_relative_error=1e-12,
        zero_repro_tolerance=1e-12,
        min_observed_delta=1e-12,
        allow_directional=True,
    )
    assert directional["promotion_eligible"] is True
    assert directional["response_coverage"]["response_kind"] == "directional"
