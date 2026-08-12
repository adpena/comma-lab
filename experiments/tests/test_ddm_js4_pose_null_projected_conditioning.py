from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import ddm_js4_pose_null_projected_conditioning as js4


def _small_geometry(monkeypatch: pytest.MonkeyPatch, *, height: int = 1, width: int = 2) -> int:
    monkeypatch.setattr(js4, "H", height)
    monkeypatch.setattr(js4, "W", width)
    return 3 * height * width


def _recipe_args(tmp_path: Path) -> argparse.Namespace:
    cache = tmp_path / "projector"
    cache.mkdir(parents=True)
    (cache / "MANIFEST.json").write_text("{}\n")
    return argparse.Namespace(
        output=tmp_path,
        projector_cache=cache,
        hidden=4,
        max_delta=6.0,
        lr=1e-3,
        stage_steps=(25,),
        checkpoint_every=5,
        pose_every=25,
        pose_weight=0.0,
        ema_decay=0.99,
        grad_clip=1.0,
        max_wall_seconds=1800.0,
        resume=True,
    )


def _smoke_metrics(*, beneficial: int = 1) -> dict[str, int]:
    return {"robust_beneficial_flips": beneficial}


def _leakage(*, gate: bool = True, attribution: str | None = None) -> dict[str, bool | str]:
    return {
        "uint8_gate_pass": gate,
        "attribution": attribution
        or ("neither-continuous-nor-uint8-exceeds-gate" if gate else "continuous-nonlinearity-already-exceeds-gate"),
    }


def test_token_identity_is_content_addressed_and_geometry_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_geometry(monkeypatch, height=2, width=3)
    first = np.arange(6, dtype=np.uint8).reshape(2, 3)
    second = first.copy()
    second[0, 0] += 1
    assert js4.token_sha256(first) != js4.token_sha256(second)
    with pytest.raises(js4.JS4Error, match="token geometry differs"):
        js4.token_sha256(np.zeros((3, 2), dtype=np.uint8))


def test_projection_removes_the_active_row_space(monkeypatch: pytest.MonkeyPatch) -> None:
    dimension = _small_geometry(monkeypatch)
    correction = torch.arange(dimension, dtype=torch.float32).reshape(1, 3, 1, 2)
    basis = torch.zeros((1, js4.PROJECTOR_ROWS, dimension))
    basis[0, 0, 0] = 1.0
    basis[0, 1, 2] = 1.0
    projected = js4.project_with_row_basis(correction, basis).reshape(-1)
    assert float(projected[0]) == 0.0
    assert float(projected[2]) == 0.0
    assert torch.equal(projected[[1, 3, 4, 5]], correction.reshape(-1)[[1, 3, 4, 5]])


def test_projection_is_idempotent_and_backpropagates_in_the_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dimension = _small_geometry(monkeypatch)
    correction = torch.randn((1, 3, 1, 2), generator=torch.Generator().manual_seed(4), requires_grad=True)
    basis = torch.zeros((1, js4.PROJECTOR_ROWS, dimension))
    basis[0, 0, 1] = 1.0
    first = js4.project_with_row_basis(correction, basis)
    second = js4.project_with_row_basis(first, basis)
    assert torch.equal(first, second)
    first.square().sum().backward()
    assert correction.grad is not None
    assert float(correction.grad.reshape(-1)[1]) == 0.0


def test_projection_rejects_correction_or_basis_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    dimension = _small_geometry(monkeypatch)
    basis = torch.zeros((1, js4.PROJECTOR_ROWS, dimension))
    with pytest.raises(js4.JS4Error, match="correction geometry differs"):
        js4.project_with_row_basis(torch.zeros((1, 2, 1, 2)), basis)
    with pytest.raises(js4.JS4Error, match="projector geometry differs"):
        js4.project_with_row_basis(torch.zeros((1, 3, 1, 2)), basis[:, :5])


def test_jacobian_is_recentered_on_custody_despite_base_rerender_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_geometry(monkeypatch)

    def camera_roundtrip(_torch, _functional, pre_r, correction):
        return pre_r + correction, pre_r + correction

    def preprocess(_torch, _functional, pair):
        return pair[:, 1].reshape(pair.shape[0], 6)

    class PoseNet:
        def __call__(self, value):
            return {"pose": value}

    monkeypatch.setattr(js4.js3, "camera_roundtrip", camera_roundtrip)
    monkeypatch.setattr(js4, "differentiable_pose_preprocess", preprocess)
    context = SimpleNamespace(
        modules=SimpleNamespace(torch=torch, functional=None),
        sample=np.array([0]),
        base_pairs=np.zeros((1, 2, 1, 2, 3), dtype=np.uint8),
        base_pose_input=torch.full((1, 6), 10.0),
        custody_pose=np.zeros((1, 6), dtype=np.float32),
        posenet=PoseNet(),
    )
    jacobian, mismatch = js4.compute_pair_jacobian(
        context,
        np.zeros((3, 1, 2), dtype=np.float32),
        0,
    )
    assert mismatch == 10.0
    assert np.array_equal(jacobian, np.eye(6, dtype=np.float32))


def test_cached_basis_validation_accepts_orthonormal_padded_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dimension = _small_geometry(monkeypatch)
    basis = np.zeros((js4.PROJECTOR_ROWS, dimension), dtype=np.float32)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    js4.validate_padded_row_basis(basis, rank=2)


@pytest.mark.parametrize("defect", ["rank", "orthogonality", "padding"])
def test_cached_basis_validation_fails_closed(monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    dimension = _small_geometry(monkeypatch)
    basis = np.zeros((js4.PROJECTOR_ROWS, dimension), dtype=np.float32)
    basis[0, 0] = 1.0
    rank = 1
    if defect == "rank":
        rank = 0
    elif defect == "orthogonality":
        basis[0, 0] = 2.0
    else:
        basis[1, 1] = 1.0
    with pytest.raises(js4.JS4Error):
        js4.validate_padded_row_basis(basis, rank)


def test_router_requires_activation_before_consuming_a_correction() -> None:
    router = object.__new__(js4.ProjectionRouter)
    router.active_basis = None
    with pytest.raises(js4.JS4Error, match="not activated"):
        router.project(torch.zeros((1, 3, js4.H, js4.W)))


def test_distribution_reports_denominator_and_guard_relevant_tail() -> None:
    row = js4._distribution(np.array([-2.0, 0.0, 1.0, 5.0]))
    assert row["count"] == 4
    assert row["p50"] == pytest.approx(0.5)
    assert row["max"] == 5.0


def test_js4_run_config_uses_the_unpatched_js3_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _recipe_args(tmp_path)
    monkeypatch.setattr(js4, "_JS3_RUN_CONFIG", lambda _args: {"pose_weight": 0.0})
    monkeypatch.setattr(js4.js3, "run_config", lambda _args: pytest.fail("recursive patched call"))
    config = js4.js4_run_config(args, {"pairs": [{"receipt": {"path": "pair.json"}}]})
    assert config["pose_weight"] == 0.0
    assert config["projector_fixed"] is True
    assert len(config["projector_manifest_sha256"]) == 64


def test_runtime_patch_projects_forward_and_restores_js3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Model(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return 2.0 * value

    class Router:
        def __init__(self) -> None:
            self.activated = False

        def activate(self, _tokens: torch.Tensor) -> None:
            self.activated = True

        def project(self, correction: torch.Tensor) -> torch.Tensor:
            return correction + 1.0

    args = _recipe_args(tmp_path)
    cache = SimpleNamespace(manifest={"pairs": [{"receipt": {"path": "pair.json"}}]})
    router = Router()
    original_build = lambda *_args, **_kwargs: Model()  # noqa: E731
    original_context = lambda *_args: torch.tensor(7.0)  # noqa: E731
    original_config = js4.js3.run_config
    original_atomic_json = js4.js3.atomic_json
    monkeypatch.setattr(js4.js3, "build_model", original_build)
    monkeypatch.setattr(js4.js3, "fixed_context", original_context)
    with js4.projected_js3_runtime(router, args, cache):
        model = js4.js3.build_model(None, None, 4, 6.0)
        assert float(model(torch.tensor(2.0))) == 5.0
        assert float(model._js4_unprojected_forward(torch.tensor(2.0))) == 4.0
        assert float(js4.js3.fixed_context(None, None, torch.zeros(1), None)) == 7.0
        assert router.activated is True
        assert js4.js3.run_config(args)["schema"] == "ddm_js4_run_config.v1"
    assert js4.js3.build_model is original_build
    assert js4.js3.fixed_context is original_context
    assert js4.js3.run_config is original_config
    assert js4.js3.atomic_json is original_atomic_json


def test_stage_checkpoint_is_promoted_only_after_stage_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _recipe_args(tmp_path)
    cache = SimpleNamespace(manifest={"pairs": [{"receipt": {"path": "pair.json"}}]})
    router = SimpleNamespace(project=lambda value: value, activate=lambda _tokens: None)
    checkpoint = {
        "schema": "ddm_js4_checkpoint_record.v1",
        "checkpoint": {"path": str(tmp_path / "checkpoint.pt"), "bytes": 1, "sha256": "0" * 64},
        "step": 25,
        "stage": "stage_01_step_000025",
    }
    result_path = tmp_path / "stages/stage_01_step_000025/RESULT.json"
    with js4.projected_js3_runtime(router, args, cache):
        js4.js3.atomic_json(result_path, {"checkpoint": checkpoint})
    latest = json.loads((tmp_path / "checkpoints/LATEST.json").read_text())
    pending = json.loads((tmp_path / "checkpoints/PENDING_STAGE.json").read_text())
    assert latest["step"] == 25
    assert latest["checkpoint"] == checkpoint["checkpoint"]
    assert pending["status"] == "PROMOTED_AFTER_STAGE_RESULT"
    assert pending["stage_result"] == js4.js3.file_record(result_path)


def test_checkpoint_retains_state_and_keeps_stage_pending(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    config = {"seed": js4.SEED}
    record = js4.save_checkpoint(
        model,
        optimizer,
        ema,
        tmp_path,
        "stage_01_step_000025",
        25,
        [{"step": 25}],
        torch,
        config,
    )
    checkpoint_path = Path(record["checkpoint"]["path"])
    assert record["checkpoint"] == js4.js3.file_record(checkpoint_path)
    assert not (tmp_path / "checkpoints/LATEST.json").exists()
    pending = json.loads((tmp_path / "checkpoints/PENDING_STAGE.json").read_text())
    assert pending["status"] == "AWAITING_STAGE_RESULT"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "ddm_js4_checkpoint.v1"
    assert payload["config"] == config


def test_collect_completed_stage_results_restores_resumed_rows(tmp_path: Path) -> None:
    args = _recipe_args(tmp_path)
    args.stage_steps = (25, 100)
    for index, step in enumerate(args.stage_steps, start=1):
        label = f"stage_{index:02d}_step_{step:06d}"
        path = tmp_path / "stages" / label / "RESULT.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"stage": label, "step": step}))
    training = js4.collect_completed_stage_results(
        args,
        {"steps_completed": 100, "stages": []},
    )
    assert [row["step"] for row in training["stages"]] == [25, 100]


def test_sealed_recipe_queues_only_a_receiver_closed_payload(tmp_path: Path) -> None:
    args = _recipe_args(tmp_path)
    queued = js4.sealed_recipe(
        args,
        2.0,
        700,
        700,
        _smoke_metrics(),
        _leakage(),
        receiver_integration_complete=True,
    )
    assert queued["disposition"] == "QUEUED-WITH-A-FIRE-ORDER"
    assert queued["current_receiver_payload_lower_bound_bytes"] == 1400
    assert queued["receiver_integration_complete"] is True
    assert queued["long_burn_launched_by_ddm_js4"] is False
    assert "25,100,300" in queued["command"]
    assert queued["command"][queued["command"].index("--pose-weight") + 1] == "0"


@pytest.mark.parametrize(
    ("beneficial", "pose_gate", "projector_bytes", "reason"),
    [
        (0, True, 700, "F1"),
        (1, False, 700, "continuous"),
        (1, True, 900, "basis"),
    ],
)
def test_sealed_recipe_folds_each_charter_falsifier(
    tmp_path: Path,
    beneficial: int,
    pose_gate: bool,
    projector_bytes: int,
    reason: str,
) -> None:
    args = _recipe_args(tmp_path)
    recipe = js4.sealed_recipe(
        args,
        2.0,
        700,
        projector_bytes,
        _smoke_metrics(beneficial=beneficial),
        _leakage(gate=pose_gate),
    )
    assert recipe["disposition"] == "FOLDED"
    assert reason in recipe["fire_trigger"]


def test_sealed_recipe_routes_quantum_only_failure_to_cvp(tmp_path: Path) -> None:
    recipe = js4.sealed_recipe(
        _recipe_args(tmp_path),
        2.0,
        700,
        700,
        _smoke_metrics(),
        _leakage(gate=False, attribution="quantum-alone"),
    )
    assert recipe["disposition"] == "FOLDED"
    assert "rounding/CVP" in recipe["fire_trigger"]


def test_parser_defaults_to_bounded_resumable_smoke() -> None:
    args = js4.parser().parse_args([])
    assert args.stage_steps == (25,)
    assert args.checkpoint_every == 5
    assert args.pose_every == 25
    assert args.pose_weight == 0.0
    assert args.resume is True


def test_single_writer_refuses_an_overlapping_run(tmp_path: Path) -> None:
    with (
        js4.single_writer(tmp_path),
        pytest.raises(js4.JS4Error, match="another JS4 writer"),
        js4.single_writer(tmp_path),
    ):
        pytest.fail("overlapping writer acquired the lock")
    receipt = json.loads((tmp_path / "RUN_LOCK.json").read_text())
    assert receipt["status"] == "RELEASED"


def test_pair_receipt_rejects_wrong_base_custody(tmp_path: Path) -> None:
    path = tmp_path / "PAIR.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ddm_js4_pose_projector_pair.v1",
                "seed": js4.SEED,
                "base_archive_sha256": "wrong",
            }
        )
    )
    with pytest.raises(js4.JS4Error, match="custody differs"):
        js4._record_from_receipt(path)
