from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    ExactV9SemanticRootError,
    _checkpoint_config,
)

REPO = Path(__file__).resolve().parents[3]
TRAINER = REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
HAS_MLX = importlib.util.find_spec("mlx") is not None


@pytest.fixture(scope="module")
def trainer():
    if not HAS_MLX:
        pytest.skip("mlx unavailable")
    name = "_v9_target_checkpoint_integration"
    spec = importlib.util.spec_from_file_location(name, TRAINER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        n_hidden=2,
        hidden_dim=8,
        mod_dim=4,
        activation="hosc",
        chroma=True,
        wire_w0=30.0,
        wire_s0=10.0,
        hosc_beta=1.0,
        hosc_omega=1.0,
        bank_n_scales=4,
        bank_n_orient0=6,
        bank_f0=2.0,
        bank_base=2.0,
        bank_n_iso=4,
        max_bank_freq=None,
        lane_edge_weight=0.0,
        lane_edge_class=1,
        basis="legacy_fourier_ab_control",
        self_orient=False,
        n_dir_freqs=4,
        freq_across=32.0,
        freq_along=8.0,
        reorient_every=0,
        w_pose=1.0,
        curriculum=True,
        tau_softplus_start_epoch=300,
        l7_start_epoch=800,
        fresh_producer=True,
        seed=105,
        _fresh_lineage_root_sha256="b" * 64,
        _fresh_lineage_initial_state_sha256="c" * 64,
        _fresh_lineage_dsl_compile_hash="d" * 64,
        _fresh_lineage_current_launch_dsl_compile_hash="d" * 64,
        _fresh_lineage_target_projection_sha256="a" * 64,
        _v9_target_checkpoint_arrays={
            "__cfg_g109_target_projection_json": np.asarray(
                '{"schema":"tac.taskspace_v9_training_target_binding.v1"}'
            ),
            "__cfg_g109_target_projection_sha256": np.asarray("a" * 64),
            "__cfg_verdict_batch": np.asarray(16),
        },
    )


def test_target_binding_reaches_deploy_and_resume_checkpoints(
    trainer,
    tmp_path: Path,
) -> None:
    args = _args()
    deploy = trainer._build_ema_checkpoint_arrays(
        {"weight": np.ones((2, 2), dtype=np.float32)},
        args=args,
        softmax_temp=0.5,
        render_h=384,
        render_w=512,
        epoch=7,
        in_feat=4,
    )
    for key, expected in args._v9_target_checkpoint_arrays.items():
        assert np.asarray(deploy[key]).item() == np.asarray(expected).item()
    assert int(deploy["__cfg_fresh_producer"]) == 1
    assert int(deploy["__cfg_fresh_seed"]) == args.seed

    resume = trainer._build_resume_state_arrays(
        {"weight": np.ones((2, 2), dtype=np.float32)},
        {"weight": np.full((2, 2), 2.0, dtype=np.float32)},
        {"weight.m": np.zeros((2, 2), dtype=np.float32)},
        args=args,
        epoch=7,
        in_feat=4,
    )
    path = tmp_path / "resume.npz"
    trainer._atomic_savez(path, resume)
    restored = trainer._load_resume_state(path)
    for key, expected in args._v9_target_checkpoint_arrays.items():
        assert restored["cfg"][key] == np.asarray(expected).item()
    assert int(restored["cfg"]["__cfg_fresh_producer"]) == 1
    assert int(restored["cfg"]["__cfg_fresh_seed"]) == args.seed


def test_pose_carrier_decode_contract_reaches_deploy_resume_and_drift_guard(
    trainer,
) -> None:
    args = _args()
    args.pose_carrier = True
    args.pose_carrier_source = "generated_y1"
    args.pose_carrier_residual_mode = "table"
    args.pose_carrier_residual_scale = 1.0
    args.pose_carrier_s_t = 0.044
    args.pose_carrier_s_r = 0.0
    args.pose_carrier_pitch = 0.0
    args._pose_carrier_effective_s_t = 0.044
    args._pose_carrier_native_hw = (874, 1164)
    deploy = trainer._build_ema_checkpoint_arrays(
        {
            "pose_carrier.xi_stored": np.zeros((600, 6), dtype=np.float32),
            "pose_carrier.dxi": np.zeros((600, 6), dtype=np.float32),
        },
        args=args,
        softmax_temp=0.5,
        render_h=384,
        render_w=512,
        epoch=7,
        in_feat=4,
    )
    resume = trainer._build_resume_state_arrays(
        {"pose_carrier.dxi": np.zeros((600, 6), dtype=np.float32)},
        {"pose_carrier.dxi": np.zeros((600, 6), dtype=np.float32)},
        None,
        args=args,
        epoch=7,
        in_feat=4,
    )
    for arrays in (deploy, resume):
        assert int(arrays["__cfg_pose_carrier"]) == 1
        assert str(arrays["__cfg_pose_carrier_contract_schema"]) == (
            "tac.v9_pose_carrier_checkpoint_contract.v2"
        )
        assert str(arrays["__cfg_pose_carrier_source"]) == "generated_y1"
        assert str(arrays["__cfg_pose_carrier_residual_mode"]) == "table"
        assert float(arrays["__cfg_pose_carrier_residual_scale"]) == 1.0
        assert float(arrays["__cfg_pose_carrier_s_t"]) == 0.044
        assert tuple(arrays["__cfg_pose_carrier_native_hw"]) == (874, 1164)
        assert str(arrays["__cfg_pose_carrier_xi_formula"]) == (
            "xi_stored+residual_scale*dxi"
        )
        assert str(
            arrays["__cfg_pose_carrier_y1_selected_preimage_schema"]
        ) == "tac.v10_factor2_selected_preimage.v1"
    args.pose_carrier_source = "generated"
    divergences = trainer._resume_lever_divergences(
        {
            key: np.asarray(value).item()
            if np.asarray(value).size == 1
            else np.asarray(value)
            for key, value in resume.items()
            if key.startswith("__cfg_")
        },
        args,
    )
    assert any("pose_carrier_source" in item for item in divergences)


def test_target_checkpoint_binding_must_be_mapping(trainer) -> None:
    args = _args()
    args._v9_target_checkpoint_arrays = "not-a-mapping"
    with pytest.raises(RuntimeError, match="must be a mapping"):
        trainer._build_ema_checkpoint_arrays(
            {"weight": np.ones((1,), dtype=np.float32)},
            args=args,
            softmax_temp=0.5,
            render_h=384,
            render_w=512,
            epoch=0,
            in_feat=4,
        )


def test_fresh_producer_refuses_foreign_warm_start_before_device_initialization(
    trainer,
) -> None:
    args = SimpleNamespace(
        micro_batch_pairs=1,
        fresh_producer=True,
        resume_from=None,
        warm_start_weights_only=True,
    )
    with pytest.raises(ValueError, match="cannot warm-start foreign weights"):
        trainer.run_train(args)


def test_fresh_producer_resume_requires_external_physical_parent_custody(
    trainer,
) -> None:
    missing = SimpleNamespace(
        micro_batch_pairs=1,
        fresh_producer=True,
        resume_from="/physical/resume.npz",
        warm_start_weights_only=False,
        fresh_lineage_parent_receipt=None,
        fresh_lineage_parent_receipt_sha256=None,
    )
    with pytest.raises(ValueError, match="cannot self-attest its ancestry"):
        trainer.run_train(missing)

    half = SimpleNamespace(
        micro_batch_pairs=1,
        fresh_producer=True,
        resume_from="/physical/resume.npz",
        warm_start_weights_only=False,
        fresh_lineage_parent_receipt="/physical/receipt.json",
        fresh_lineage_parent_receipt_sha256=None,
    )
    with pytest.raises(ValueError, match="must be supplied together"):
        trainer.run_train(half)

    foreign = SimpleNamespace(
        micro_batch_pairs=1,
        fresh_producer=False,
        resume_from="/physical/resume.npz",
        warm_start_weights_only=False,
        fresh_lineage_parent_receipt="/physical/receipt.json",
        fresh_lineage_parent_receipt_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="valid only"):
        trainer.run_train(foreign)


def test_fresh_producer_refuses_deploy_only_film_projection(trainer) -> None:
    args = SimpleNamespace(
        micro_batch_pairs=1,
        fresh_producer=True,
        film_stiefel=True,
        resume_from=None,
        warm_start_weights_only=False,
    )
    with pytest.raises(ValueError, match="deploy EMA identity"):
        trainer.run_train(args)


def test_fresh_producer_resume_requires_complete_state_bound_lineage(trainer) -> None:
    args = _args()
    live = {"weight": np.ones((2, 2), dtype=np.float32)}
    ema = {"weight": np.full((2, 2), 2.0, dtype=np.float32)}
    optimizer = {"weight.m": np.zeros((2, 2), dtype=np.float32)}
    resume_arrays = {
        f"{trainer._RESUME_LIVE_PREFIX}weight": live["weight"],
        f"{trainer._RESUME_EMA_PREFIX}weight": ema["weight"],
        f"{trainer._RESUME_OPT_PREFIX}weight.m": optimizer["weight.m"],
        "__resume_epoch": np.asarray(7),
        "__resume_stage": np.asarray("stageCE"),
        "__rng_np_pos": np.asarray(17),
        "__recent_losses": np.asarray([3.0, 2.0], dtype=np.float64),
        "__cfg_fresh_producer": np.asarray(1),
        **trainer._fresh_producer_root_arrays(args),
        "__cfg_fresh_current_launch_dsl_compile_hash": np.asarray("d" * 64),
    }
    lineage = trainer._fresh_producer_checkpoint_lineage_arrays(
        args,
        resume_state_arrays=resume_arrays,
        epoch=7,
        stage="stageCE",
        parent_checkpoint_id=trainer._FRESH_LINEAGE_ROOT_PARENT,
    )
    cfg = {
        **{
            key: (
                np.asarray(value).item()
                if np.asarray(value).size == 1
                else np.asarray(value).tolist()
            )
            for key, value in resume_arrays.items()
            if key.startswith("__")
        },
        **{
            key: (
                np.asarray(value).item()
                if np.asarray(value).size == 1
                else np.asarray(value).tolist()
            )
            for key, value in lineage.items()
        },
    }
    state = {
        "live": live,
        "ema": ema,
        "opt": optimizer,
        "polyak": {},
        "epoch": 7,
        "cfg": cfg,
    }
    trainer._validate_fresh_producer_resume_lineage(args, state)

    with pytest.raises(ValueError, match="lacks its cold own-lineage state"):
        trainer._validate_fresh_producer_resume_lineage(args, {})
    copied_marker_foreign_state = {
        **state,
        "live": {"weight": np.zeros((2, 2), dtype=np.float32)},
    }
    with pytest.raises(ValueError, match="state hash differs"):
        trainer._validate_fresh_producer_resume_lineage(
            args,
            copied_marker_foreign_state,
        )
    copied_marker_foreign_rng = {
        **state,
        "cfg": {**cfg, "__rng_np_pos": 18},
    }
    with pytest.raises(ValueError, match="state hash differs"):
        trainer._validate_fresh_producer_resume_lineage(
            args,
            copied_marker_foreign_rng,
        )
    copied_marker_foreign_losses = {
        **state,
        "cfg": {**cfg, "__recent_losses": [3.0, 1.0]},
    }
    with pytest.raises(ValueError, match="state hash differs"):
        trainer._validate_fresh_producer_resume_lineage(
            args,
            copied_marker_foreign_losses,
        )
    copied_marker_foreign_polyak = {
        **state,
        "polyak": {"weight": np.ones((2, 2), dtype=np.float64)},
    }
    with pytest.raises(ValueError, match="state hash differs"):
        trainer._validate_fresh_producer_resume_lineage(
            args,
            copied_marker_foreign_polyak,
        )


def test_fresh_lineage_root_binds_dsl_target_seed_and_initial_state(
    trainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAC_DSL_COMPILE_HASH", "e" * 64)
    args = _args()
    args.seed = 105
    initial = {"weight": np.arange(8, dtype=np.float32).reshape(2, 4)}
    trainer._initialize_fresh_producer_lineage(args, initial_state=initial)
    first = args._fresh_lineage_root_sha256
    assert args._fresh_lineage_dsl_compile_hash == "e" * 64
    assert args._fresh_lineage_target_projection_sha256 == "a" * 64

    changed = _args()
    changed.seed = 106
    trainer._initialize_fresh_producer_lineage(changed, initial_state=initial)
    assert changed._fresh_lineage_root_sha256 != first

    monkeypatch.setenv("TAC_DSL_COMPILE_HASH", "f" * 64)
    resumed = _args()
    resumed.seed = 105
    resumed._fresh_lineage_resume_root_dsl_compile_hash = "e" * 64
    trainer._initialize_fresh_producer_lineage(resumed, initial_state=initial)
    assert resumed._fresh_lineage_root_sha256 == first
    assert resumed._fresh_lineage_current_launch_dsl_compile_hash == "f" * 64

    monkeypatch.delenv("TAC_DSL_COMPILE_HASH")
    with pytest.raises(ValueError, match="TAC_DSL_COMPILE_HASH"):
        trainer._initialize_fresh_producer_lineage(
            _args(),
            initial_state=initial,
        )


def test_g105_requires_fresh_producer_marker() -> None:
    with pytest.raises(ExactV9SemanticRootError, match="fresh-producer marker"):
        _checkpoint_config({}, {})
