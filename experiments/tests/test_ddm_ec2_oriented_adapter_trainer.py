from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional

from experiments import ddm_ec1_implicit_edge_conditioning as ec1
from experiments import ddm_ec2_modal_oriented_adapter_trainer as dispatch
from experiments import ddm_ec2_oriented_adapter_trainer_worker as worker
from experiments.ddm_ec1_runtime import ec1_latent_conditioner as runtime
from tac.differentiable_eval_roundtrip import (
    CameraLiftKernel,
    EvalRoundTripOrdering,
    apply_camera_uint8_lift_during_training,
    apply_eval_roundtrip_during_training,
)


class _TinyBlock(nn.Module):
    def forward(self, value: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        return value + frame[:, : value.shape[1], None, None] * 0.0


class _TinySemantic(nn.Module):
    def __init__(self, height: int = 8, width: int = 12) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.token_embed = nn.Embedding(5, 96)
        self.coord_mix = nn.Conv2d(100, 96, 1)
        self.frame_embed = nn.Embedding(4, 96)
        self.blocks = nn.ModuleList([_TinyBlock() for _ in range(4)])
        self.head = nn.Conv2d(96, 3, 3, padding=1)

    def coordinates(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, 4, self.height, self.width, device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
        value = self.token_embed(tokens).permute(0, 3, 1, 2)
        value = self.coord_mix(
            torch.cat(
                [value, self.coordinates(value.shape[0], value.device, value.dtype)],
                dim=1,
            )
        )
        frame = self.frame_embed(pair_indices)
        for block in self.blocks:
            value = block(value, frame)
        return torch.sigmoid(self.head(functional.gelu(value))) * 255.0


def test_zero_adapter_is_exact_identity_and_nonzero_moves_same_object() -> None:
    torch.manual_seed(worker.SEED)
    semantic = _TinySemantic().eval()
    adapter = worker.initialize_conditioner(torch.device("cpu"), "oriented")
    tokens = torch.randint(0, 5, (1, 8, 12))
    indices = torch.tensor([2])
    base = semantic(tokens, indices)
    zero = worker.conditioned_semantic_forward(semantic, adapter, tokens, indices)
    assert torch.equal(base, zero)
    with torch.no_grad():
        adapter.head.weight[0, 0, 0, 0] = 0.5
    moved = worker.conditioned_semantic_forward(semantic, adapter, tokens, indices)
    assert not torch.equal(base, moved)


def test_composite_camera_roundtrip_is_canonical_public_ordering() -> None:
    torch.manual_seed(4)
    source = torch.rand(2, 3, 17, 23) * 255.0
    camera = apply_camera_uint8_lift_during_training(
        source,
        lift_kernel=CameraLiftKernel.BILINEAR,
        simulate_uint8=True,
        ste_round=True,
        target_h=31,
        target_w=37,
    )
    explicit = functional.interpolate(camera, size=(17, 23), mode="bilinear", align_corners=False)
    canonical = apply_eval_roundtrip_during_training(
        source,
        ordering=EvalRoundTripOrdering.CAMERA_UINT8,
        lift_kernel=CameraLiftKernel.BILINEAR,
        simulate_uint8=True,
        simulate_resize=True,
        ste_round=True,
        target_h=31,
        target_w=37,
    )
    assert torch.equal(explicit, canonical)
    assert torch.equal(camera, camera.round().clamp(0, 255))


def test_stratified_order_is_deterministic_full_population_not_prefix() -> None:
    target = np.zeros((600, 2, 3), dtype=np.uint8)
    base = target.copy()
    for pair in range(600):
        base[pair].flat[: pair % 7] = 1
    first = worker.stratified_pair_order(base, target, 1)
    second = worker.stratified_pair_order(base, target, 1)
    assert first == second
    assert sorted(first) == list(range(600))
    assert first != list(range(600))


def test_realized_flip_loss_backpropagates_and_reports_typed_counts() -> None:
    logits = torch.randn(1, 5, 3, 4, requires_grad=True)
    target = torch.randint(0, 5, (1, 3, 4))
    base_error = torch.zeros((1, 3, 4), dtype=torch.bool)
    base_error[:, 0, 0] = True
    loss, metrics = worker.realized_flip_loss(
        logits,
        target,
        base_error,
        error_weight=4.0,
        correct_weight=0.25,
        margin_weight=0.05,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.count_nonzero(logits.grad)
    assert set(metrics) == {
        "loss",
        "hard_flips",
        "fixed_base_errors",
        "introduced_errors",
    }


def test_packaged_stage_is_deterministic_and_receiver_parseable(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = worker.initialize_conditioner(torch.device("cpu"), "oriented")
    with torch.no_grad():
        model.head.weight[0, 0, 0, 0] = 0.25
    ema = worker.EMA(model, decay=worker.EMA_DECAY, warmup=True)
    result = worker.package_stage(
        tmp_path,
        stage_name="test_stage",
        base_archive=dispatch.DEFAULT_ARCHIVE,
        model=model,
        ema=ema,
    )
    archive = Path(result["ema"]["archive"]["path"])
    repeat = Path(result["ema"]["archive_repeat"]["path"])
    assert archive.read_bytes() == repeat.read_bytes()
    header, state = runtime.parse_module(Path(result["ema"]["module"]["path"]).read_bytes())
    assert header["family"] == "oriented"
    assert set(state) == set(model.state_dict())


def test_qat_forward_matches_serialized_receiver_parseback(tmp_path: Path) -> None:
    torch.manual_seed(9)
    semantic = _TinySemantic().eval()
    model = worker.initialize_conditioner(torch.device("cpu"), "oriented")
    with torch.no_grad():
        model.head.weight.normal_(0.0, 0.15)
        model.head.bias.normal_(0.0, 0.03)
    tokens = torch.randint(0, 5, (1, 8, 12))
    indices = torch.tensor([1])
    qat = worker.conditioned_semantic_forward(semantic, model, tokens, indices)
    serialized = ec1.serialize_module(model, tmp_path, "candidate")
    parsed = runtime.load_conditioner(Path(serialized["coded"]["path"]).read_bytes(), torch.device("cpu"))
    receiver = worker.conditioned_semantic_forward(semantic, parsed, tokens, indices)
    assert torch.equal(qat, receiver)


def test_toy_gate_retains_every_materialized_tensor(tmp_path: Path) -> None:
    result = worker.run_toy_gate(tmp_path)
    assert result["passed"]
    assert result["scope"].startswith("STRUCTURAL_ONLY")
    for record in result["retained_payloads"].values():
        path = Path(record["path"])
        assert path.is_file()
        assert dispatch.sha256_file(path) == record["sha256"]


def test_endpoint_is_batch_atomic_and_resume_verifies_retained_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "N_PAIRS", 2)
    monkeypatch.setattr(worker, "WORK_HW", (2, 3))
    monkeypatch.setattr(worker, "ENDPOINT_BATCH_SIZE", 1)
    monkeypatch.setattr(worker, "BASE_FLIPS", 2)

    def fake_receiver(
        semantic: nn.Module,
        model: nn.Module,
        tokens: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del semantic, model, tokens
        value = indices[:, None, None, None].float().expand(-1, 3, 2, 3)
        return value, value.round(), value

    class FakeScorer(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(value.shape[0], 5, 2, 3)
            logits[:, 0] = 1.0
            return logits

    monkeypatch.setattr(worker, "composite_receiver", fake_receiver)
    tokens = np.zeros((2, 2, 3), dtype=np.uint8)
    target = np.zeros_like(tokens)
    base = np.ones_like(tokens)
    first = worker.full_endpoint(
        tmp_path,
        semantic=nn.Identity(),
        model=nn.Identity(),
        scorer=FakeScorer(),
        tokens=tokens,
        target=target,
        base_field=base,
        device=torch.device("cpu"),
    )
    first_sha = first["retained_payloads"]["argmax_n600"]["sha256"]
    second = worker.full_endpoint(
        tmp_path,
        semantic=nn.Identity(),
        model=nn.Identity(),
        scorer=FakeScorer(),
        tokens=tokens,
        target=target,
        base_field=base,
        device=torch.device("cpu"),
    )
    assert first["flips"] == second["flips"] == 0
    assert second["retained_payloads"]["argmax_n600"]["sha256"] == first_sha
    assert len(list((tmp_path / "endpoint/retained/batches").glob("*/BATCH_RECEIPT.json"))) == 2


def test_request_seals_only_one_family_and_schedule_is_derived() -> None:
    request = dispatch.make_request(
        family="oriented",
        run_id="unit",
        runtime_manifest={"bundle": {"bytes": 1, "sha256": "0" * 64}},
        payloads={},
    )
    assert request["family"] == "oriented"
    assert "families" not in request
    assert request["schedule_derivation"]["fits_hard_cap"]
    assert request["schedule_derivation"]["projected_seconds"] == 9_900
    with pytest.raises(dispatch.EC2DispatchError):
        dispatch.make_request(
            family="oriented,class_only",
            run_id="bad",
            runtime_manifest={},
            payloads={},
        )


def test_control_entrypoint_has_hard_break_even_guard() -> None:
    source = Path(dispatch.__file__).read_text()
    assert 'if not endpoint.get("clears_oriented_break_even")' in source
    assert "controls remain blocked" in source
    assert "expected_class_request_sha256" in source
    assert "expected_undirected_request_sha256" in source


def test_worker_has_no_cpu_authority_fallback_and_persists_stage_checkpoints() -> None:
    source = Path(worker.__file__).read_text()
    assert "requires a CUDA T4; no fallback" in source
    assert 'gpu="T4"' in Path(dispatch.__file__).read_text()
    assert 'root / "live.pt"' in source
    assert 'root / "ema.pt"' in source
    assert 'label=f"periodic_g' in source
    assert 'label=f"stage_{stage[' in source


def test_fire_order_schema_requires_precreated_harvest_destination() -> None:
    command_fragment = "mkdir -p"
    source = Path(dispatch.__file__).read_text()
    assert command_fragment in source
    assert "ONLY after the harvested oriented SELECTED_RESULT" in source
