from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools/train_ddm_cl1_hpac_capacity.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("train_ddm_cl1_hpac_capacity", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimum_args(tool, *, save: Path, out: Path):
    return tool._build_parser().parse_args(
        [
            "--cache",
            "/inputs/cache.pt",
            "--init",
            "/inputs/init.pt",
            "--save",
            str(save),
            "--out",
            str(out),
        ]
    )


def test_parser_preserves_canonical_structural_defaults(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _minimum_args(
        tool,
        save=tmp_path / "model.pt",
        out=tmp_path / "result.json",
    )
    assert (args.channels, args.patch, args.delta, args.frame_dim) == (64, 64, 2, 8)
    assert args.rate_lambda == 1.0
    assert args.device == "mps"
    assert args.resume_from is None


def test_main_checks_governed_admission_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()

    def refuse(_label: str) -> None:
        raise RuntimeError("admission checked first")

    monkeypatch.setattr(tool, "assert_governed_admission", refuse)
    with pytest.raises(RuntimeError, match="admission checked first"):
        tool.main()


def test_operational_pause_and_output_paths_do_not_change_run_identity(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    first = _minimum_args(
        tool,
        save=tmp_path / "a.pt",
        out=tmp_path / "a.json",
    )
    second = _minimum_args(
        tool,
        save=tmp_path / "b.pt",
        out=tmp_path / "b.json",
    )
    second.stop_after_epoch = 10
    second.resume_from = tmp_path / "resume.pt"
    first_identity = tool._run_identity(
        first,
        cache_sha256="a" * 64,
        init_sha256="b" * 64,
        source_sha256={"source.py": "c" * 64},
        ema_policy={"equation_id": "test", "decay": 0.9},
    )
    second_identity = tool._run_identity(
        second,
        cache_sha256="a" * 64,
        init_sha256="b" * 64,
        source_sha256={"source.py": "c" * 64},
        ema_policy={"equation_id": "test", "decay": 0.9},
    )
    assert first_identity == second_identity
    assert len(first_identity["seed_schedule_identity_sha256"]) == 64
    assert len(first_identity["trainer_source_identity_sha256"]) == 64


def test_atomic_torch_save_refuses_immutable_overwrite(tmp_path: Path) -> None:
    tool = _load_tool()
    destination = tmp_path / "checkpoint.pt"
    tool._atomic_torch_save(destination, {"value": torch.tensor([1])}, immutable=True)
    tool._atomic_torch_save(destination, {"value": torch.tensor([1])}, immutable=True)
    with pytest.raises(tool.CL1TrainingError, match="immutable checkpoint"):
        tool._atomic_torch_save(
            destination,
            {"value": torch.tensor([2])},
            immutable=True,
        )
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    assert loaded["value"].item() == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_cpu_tree_detaches_optimizer_tensors() -> None:
    tool = _load_tool()
    source = torch.tensor([1.0], requires_grad=True)
    copied = tool._cpu_tree({"nested": [source]})
    assert copied["nested"][0].device.type == "cpu"
    assert not copied["nested"][0].requires_grad
    source.data.fill_(2.0)
    assert copied["nested"][0].item() == 1.0


def test_residual_transform_round_trips() -> None:
    tool = _load_tool()
    tokens = torch.tensor([[0, 1, 4], [4, 0, 3], [2, 4, 1]])
    encoded = tool._residuals(tokens)
    decoded = encoded.clone()
    for index in range(1, len(decoded)):
        decoded[index] = (decoded[index] + decoded[index - 1]) % 5
    assert torch.equal(decoded, tokens)


def test_ssd_boundary_rejects_local_output() -> None:
    tool = _load_tool()
    accepted = Path("/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/a.pt")
    assert tool._require_ssd_path(accepted, "test") == accepted
    fallback = Path("/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/a.pt")
    assert tool._require_ssd_path(fallback, "test") == fallback
    with pytest.raises(tool.CL1TrainingError, match="admitted SSD tier"):
        tool._require_ssd_path(REPO_ROOT / "local.pt", "test")


def test_pinned_intake_hashes_are_verified_without_host_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    expected = {}
    for name in tool.EXPECTED_INTAKE_SHA256:
        path = tmp_path / name
        path.write_bytes(f"fixture:{name}".encode())
        expected[name] = tool._sha256_file(path)
    monkeypatch.setattr(tool, "EXPECTED_INTAKE_SHA256", expected)
    assert tool._verify_intake(tmp_path) == expected
    (tmp_path / next(iter(expected))).write_bytes(b"changed")
    with pytest.raises(tool.CL1TrainingError, match="pinned intake source changed"):
        tool._verify_intake(tmp_path)


def test_preregistered_config_is_receiver_closed(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _minimum_args(
        tool,
        save=tmp_path / "model.pt",
        out=tmp_path / "result.json",
    )
    for key, value in tool.PREREGISTERED_CONFIG.items():
        setattr(args, key, value)
    tool._assert_preregistered_config(args)
    args.target_mode = "residual"
    with pytest.raises(tool.CL1TrainingError, match="receiver-closed"):
        tool._assert_preregistered_config(args)


def test_rx2_profile_changes_only_the_admitted_execution_substrate(tmp_path: Path) -> None:
    tool = _load_tool()
    args = _minimum_args(
        tool,
        save=tmp_path / "model.pt",
        out=tmp_path / "result.json",
    )
    args.profile = "rx2_mc36"
    for key, value in tool.RX2_PREREGISTERED_CONFIG.items():
        setattr(args, key, value)
    tool._assert_preregistered_config(args)
    assert args.device == "cpu"
    assert {key: value for key, value in tool.RX2_PREREGISTERED_CONFIG.items() if key != "device"} == {
        key: value for key, value in tool.PREREGISTERED_CONFIG.items() if key != "device"
    }
    args.rate_lambda = 0.5
    with pytest.raises(tool.CL1TrainingError, match="receiver-closed"):
        tool._assert_preregistered_config(args)


def test_rx2_cache_gate_refuses_noncanonical_shape() -> None:
    tool = _load_tool()
    with pytest.raises(tool.CL1TrainingError, match="shape"):
        tool._verify_rx2_cache_payload(
            {
                "seg": torch.zeros((1, 2, 3), dtype=torch.uint8),
                "spatial_token_sha256": tool.EXPECTED_RX2_SPATIAL_TOKEN_SHA256,
            }
        )


def test_jf1_profile_preserves_reference_architecture_and_requires_exact_local_root(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    args = _minimum_args(
        tool,
        save=tool.JF1_LOCAL_ROOT / "training/null/model.pt",
        out=tool.JF1_LOCAL_ROOT / "training/null/result.json",
    )
    args.profile = "jf1_joint_refit"
    args.explicit_local_output_opt_in = True
    for key, value in tool.JF1_PREREGISTERED_CONFIG.items():
        setattr(args, key, value)
    tool._assert_preregistered_config(args)
    assert tool._storage_root_for_args(args.save, "--save", args) == tool.JF1_LOCAL_ROOT.resolve()
    assert tool.JF1_PREREGISTERED_CONFIG == tool.RX2_PREREGISTERED_CONFIG
    with pytest.raises(tool.CL1TrainingError, match="JF1 local receipt root"):
        tool._storage_root_for_args(tmp_path / "outside.pt", "--save", args)


def test_jf1_cache_gate_accepts_only_the_pinned_raw_field() -> None:
    tool = _load_tool()
    seg = torch.zeros((600, 384, 512), dtype=torch.uint8)
    digest = tool.hashlib.sha256(seg.numpy().tobytes(order="C")).hexdigest()
    payload = {"seg": seg, "spatial_token_sha256": digest}
    assert tool._verify_jf1_cache_payload(payload, digest) == digest
    with pytest.raises(tool.CL1TrainingError, match="token SHA differs"):
        tool._verify_jf1_cache_payload(payload, "f" * 64)


def test_stage_controls_do_not_add_selection_observations() -> None:
    tool = _load_tool()
    off_cadence = tool._epoch_controls(
        epoch=3,
        epochs=8,
        qat_start=4,
        eval_every=2,
    )
    assert off_cadence == {
        "discrete": False,
        "should_evaluate": False,
        "continuous_stage_end": True,
        "qat_stage_end": False,
        "should_checkpoint": True,
    }
    no_qat_terminal = tool._epoch_controls(
        epoch=8,
        epochs=8,
        qat_start=9,
        eval_every=2,
    )
    assert no_qat_terminal["continuous_stage_end"] is True
    assert no_qat_terminal["qat_stage_end"] is False


def test_output_layout_refuses_collisions_and_fresh_reuse(tmp_path: Path) -> None:
    tool = _load_tool()
    collision = _minimum_args(
        tool,
        save=tmp_path / "same.pt",
        out=tmp_path / "same.pt",
    )
    with pytest.raises(tool.CL1TrainingError, match="collision"):
        tool._validate_output_layout(collision)
    clean = _minimum_args(
        tool,
        save=tmp_path / "model.pt",
        out=tmp_path / "result.json",
    )
    tool._validate_output_layout(clean)
    clean.save.write_bytes(b"occupied")
    with pytest.raises(tool.CL1TrainingError, match="already exists"):
        tool._validate_output_layout(clean)


def _optimizer_step(model, optimizer, scheduler, ema, generator) -> None:
    order = torch.randperm(4, generator=generator)
    x = torch.tensor([[1.0, -1.0], [0.5, 2.0], [-0.25, 0.75], [1.5, 0.25]])[order]
    y = torch.tensor([[0.25], [-0.5], [0.1], [0.9]])[order]
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()
    ema.update(model)
    scheduler.step()
    random.random()
    np.random.random()
    torch.rand(1)


def _assert_tree_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for l_value, r_value in zip(left, right, strict=True):
            _assert_tree_equal(l_value, r_value)
    else:
        assert left == right


def test_full_state_resume_matches_live_ema_optimizer_scheduler_and_rng(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    ema = tool.EMA(model, decay=0.9, warmup=True)
    generator = torch.Generator().manual_seed(17)
    for _ in range(3):
        _optimizer_step(model, optimizer, scheduler, ema, generator)
    best = {
        "epoch": 3,
        "estimated_joint_bytes": 10,
        "state_dict": tool._cpu_tree(ema.state_dict()),
    }
    payload = tool._checkpoint_payload(
        epoch=3,
        phase="continuous",
        model=model,
        ema=ema,
        ema_policy={"equation_id": "test", "decay": 0.9, "warmup": True},
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        device=torch.device("cpu"),
        best=best,
        history=[{"epoch": 2}, {"epoch": 3}],
        run_identity={"identity": "test"},
        resume_lineage=[],
        qat_start=4,
    )
    checkpoint = tmp_path / "resume.pt"
    tool._atomic_torch_save(checkpoint, payload, immutable=True)
    for _ in range(3):
        _optimizer_step(model, optimizer, scheduler, ema, generator)
    uninterrupted_live = tool._clone_cpu_state_dict(model)
    uninterrupted_ema = tool._cpu_tree(ema.state_dict())
    uninterrupted_optimizer = tool._cpu_tree(optimizer.state_dict())
    uninterrupted_scheduler = scheduler.state_dict()
    uninterrupted_generator = generator.get_state().clone()
    uninterrupted_random = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1)),
    )

    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert restored["causal_state_sha256"] == tool._causal_state_sha256(restored)
    tampered = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_live_key = next(iter(tampered["live_state_dict"]))
    tampered["live_state_dict"][first_live_key].view(-1)[0] += 1.0
    assert tampered["causal_state_sha256"] != tool._causal_state_sha256(tampered)
    resumed_model = torch.nn.Linear(2, 1)
    resumed_model.load_state_dict(restored["live_state_dict"])
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(restored["optimizer_state_dict"])
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=6)
    resumed_scheduler.load_state_dict(restored["scheduler_state_dict"])
    resumed_ema = tool.EMA(resumed_model, decay=0.9, warmup=True)
    tool._restore_ema_training_state(resumed_ema, restored["ema"], device=torch.device("cpu"))
    resumed_generator = torch.Generator().manual_seed(0)
    tool._restore_rng_payload(torch.device("cpu"), resumed_generator, restored["rng"])
    for _ in range(3):
        _optimizer_step(
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            resumed_ema,
            resumed_generator,
        )
    resumed_random = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1)),
    )
    for key, value in uninterrupted_live.items():
        assert torch.equal(resumed_model.state_dict()[key], value)
    for key, value in uninterrupted_ema.items():
        assert torch.equal(resumed_ema.state_dict()[key], value)
    _assert_tree_equal(resumed_optimizer.state_dict(), uninterrupted_optimizer)
    _assert_tree_equal(resumed_scheduler.state_dict(), uninterrupted_scheduler)
    assert torch.equal(resumed_generator.get_state(), uninterrupted_generator)
    assert resumed_random == uninterrupted_random
    assert restored["deployment_weights"] == "ema_shadow"
    for key, value in restored["state_dict"].items():
        assert torch.equal(value, restored["ema"]["shadow"][key])


def test_checkpoint_embeds_authoritative_lineage_and_recustodies_parents(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    ancestor = tmp_path / "old_root" / "ancestor.pt"
    ancestor.parent.mkdir(parents=True)
    ancestor.write_bytes(b"immutable parent checkpoint")
    digest = tool._sha256_file(ancestor)
    entry = {
        "source_path": str(ancestor),
        "preserved_path": str(ancestor),
        "bytes": ancestor.stat().st_size,
        "sha256": digest,
        "epoch": 2,
    }
    checkpoint_payload = {"resume_lineage": [entry]}
    # A mutable sidecar may describe a later branch.  The checkpoint snapshot,
    # not that adjacent view, is the fork authority.
    tool._atomic_json(
        ancestor.parent / "resume_lineage.json",
        {
            "schema": "ddm_cl1_hpac_capacity_resume_lineage.v1",
            "entries": [entry, {**entry, "sha256": "f" * 64, "epoch": 4}],
        },
    )
    embedded = tool._checkpoint_lineage(checkpoint_payload)
    new_root = tmp_path / "new_root" / "model.checkpoints"
    preserved = tool._preserve_lineage_parents(embedded, new_root)
    copied = new_root / "resume_parents" / f"{digest}.pt"
    assert copied.read_bytes() == ancestor.read_bytes()
    assert preserved == [{**entry, "preserved_path": str(copied)}]
    with pytest.raises(tool.CL1TrainingError, match="embedded authoritative"):
        tool._checkpoint_lineage({})


def test_cl2_profile_is_the_jf1_law_on_metal_with_the_cl1_lambda_bracket(tmp_path: Path) -> None:
    """ddm_cl2: same reference law as JF1, device mps, cl1's {1, 1/2, 1/4} bracket, SSD output."""
    tool = _load_tool()
    args = _minimum_args(
        tool,
        save=tool.PRIMARY_SSD_ROOT / "ddm_cl2_x/training/model.pt",
        out=tool.PRIMARY_SSD_ROOT / "ddm_cl2_x/training/result.json",
    )
    args.profile = "cl2_shipped_ladder"
    for key, value in tool.CL2_PREREGISTERED_CONFIG.items():
        setattr(args, key, value)
    assert args.device == "mps"
    assert {k: v for k, v in tool.CL2_PREREGISTERED_CONFIG.items() if k != "device"} == {
        k: v for k, v in tool.JF1_PREREGISTERED_CONFIG.items() if k != "device"
    }
    for admitted in (1.0, 0.5, 0.25):
        args.rate_lambda = admitted
        tool._assert_preregistered_config(args)
    args.rate_lambda = 0.125
    with pytest.raises(tool.CL1TrainingError, match="receiver-closed"):
        tool._assert_preregistered_config(args)
    args.rate_lambda = 1.0
    assert "cl2_shipped_ladder" in tool.CALLER_PINNED_INPUT_PROFILES
    # ddm_cl3 (2026-09-05) widened the bracket by the SMALLER-model direction {2.0, 4.0};
    # cl2's own three lambdas stay admitted, which is what the loop above proves.
    assert tool.PREREGISTERED_RATE_LAMBDAS_BY_PROFILE["cl2_shipped_ladder"] == frozenset({4.0, 2.0, 1.0, 0.5, 0.25})
    # No local opt-in for CL2: output must sit on an admitted SSD tier.
    assert tool._storage_root_for_args(args.save, "--save", args) == tool.PRIMARY_SSD_ROOT.resolve()
    with pytest.raises(tool.CL1TrainingError, match="admitted SSD tier"):
        tool._storage_root_for_args(tmp_path / "local.pt", "--save", args)


def test_cl3_widens_cl2_only_by_the_smaller_prior_lambdas_and_two_seeds(tmp_path: Path) -> None:
    """ddm_cl3: the smaller-model direction (lambda 2.0 / 4.0) and seed selection at lambda 1.0.

    cl2 MEASURED the bigger direction closed (lambda 1.0 -> 0.5 slope +0.446 against the -1
    break-even).  cl3 opens the opposite side plus two extra seeds, and NOTHING else: every
    other profile keeps exactly the single seed pinned in its own preregistered config, and
    every non-seed config key stays a strict equality.
    """

    tool = _load_tool()
    args = _minimum_args(
        tool,
        save=tool.PRIMARY_SSD_ROOT / "ddm_cl3_x/training/model.pt",
        out=tool.PRIMARY_SSD_ROOT / "ddm_cl3_x/training/result.json",
    )
    args.profile = "cl2_shipped_ladder"
    for key, value in tool.CL2_PREREGISTERED_CONFIG.items():
        setattr(args, key, value)

    # The smaller-prior direction is admitted; the ladder floor is still refused.
    for admitted in (2.0, 4.0):
        args.rate_lambda = admitted
        tool._assert_preregistered_config(args)
    args.rate_lambda = 8.0
    with pytest.raises(tool.CL1TrainingError, match="receiver-closed"):
        tool._assert_preregistered_config(args)
    args.rate_lambda = 1.0

    # Seed selection: cl2's seed plus the two cl3 rungs, and nothing else.
    assert tool.PREREGISTERED_SEEDS_BY_PROFILE["cl2_shipped_ladder"] == frozenset({20260716, 20260717, 20260718})
    for admitted_seed in (20260716, 20260717, 20260718):
        args.seed = admitted_seed
        tool._assert_preregistered_config(args)
    args.seed = 20260719
    with pytest.raises(tool.CL1TrainingError, match="receiver-closed"):
        tool._assert_preregistered_config(args)
    args.seed = tool.CL2_PREREGISTERED_CONFIG["seed"]

    # Every other profile keeps its single pinned seed -- the widening is cl2-local.
    for profile, config in tool.PREREGISTERED_CONFIG_BY_PROFILE.items():
        if profile == "cl2_shipped_ladder":
            continue
        assert tool.PREREGISTERED_SEEDS_BY_PROFILE[profile] == frozenset({config["seed"]})

    # A non-seed config key is still a strict equality (the loop skip is seed-only).
    args.target_mode = "residual"
    with pytest.raises(tool.CL1TrainingError, match="receiver-closed"):
        tool._assert_preregistered_config(args)
