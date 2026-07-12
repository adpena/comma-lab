import copy
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.train_levelset_witness_realized_through_R_torch import (
    _accumulated_pair_step,
    _attach_generated_pose_carrier,
    _canonical_checkpoint_due,
    _checkpoint_blob,
    _generated_pose_pair_dispatch,
    _hosc_beta_at_epoch,
    _load_validated_gt_cache,
    _load_validated_resume,
    _resolve_resume_intent,
    _restore,
    _run_structured_prefit,
    _runtime_epoch_window,
    _softmax_temp_at_epoch,
    _validate_scorer_custody,
    build_parser,
    derive_config,
    main,
)
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom
from tac.cuda_levelset_training import (
    CudaLevelSetConfig,
    DeterministicPairCursor,
    TorchLevelSetWitness,
    TorchPoseCarrier,
)
from tac.cuda_v9_controller_runtime import TorchProtectedIslandSeed
from tac.witness_control.tail_cycles import TailController, TailCycleConfig
from tac.witness_run_artifacts import TORCH_RESUME_PT

_SEG_SHA = "1" * 64
_POSE_SHA = "2" * 64


def _patch_preflight_scorers(monkeypatch, *, coverage_status="COMPLETE_1_TO_1"):
    load_calls = []

    def fake_custody(paths=None, expected_sha256=None):
        assert paths is None
        assert expected_sha256 == {"segnet": _SEG_SHA, "posenet": _POSE_SHA}
        return {
            name: {
                "path": f"upstream/models/{name}.safetensors",
                "bytes": 1,
                "sha256": digest,
                "tensor_count": 1,
                "expected_sha256": digest,
                "sha_authority": "PLAN_EXPECTED_MATCH",
            }
            for name, digest in (("segnet", _SEG_SHA), ("posenet", _POSE_SHA))
        }

    def fake_load_scorers(device):
        load_calls.append(str(device))
        networks = (torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        for network in networks:
            network.eval()
            for parameter in network.parameters():
                parameter.requires_grad_(False)
        return networks

    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch._validate_scorer_custody",
        fake_custody,
    )
    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch.cuda_v9_port_receipt",
        lambda: {
            "status": coverage_status,
            "blockers": [] if coverage_status == "COMPLETE_1_TO_1" else ["missing_surface"],
        },
    )
    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch._load_scorers",
        fake_load_scorers,
    )
    return load_calls


def _scorer_sha_args():
    return [
        "--expected-segnet-sha256",
        _SEG_SHA,
        "--expected-posenet-sha256",
        _POSE_SHA,
    ]


def _pose_flags():
    return {
        "--pose-carrier": True,
        "--pose-carrier-source": "generated",
        "--pose-carrier-residual-mode": "table",
        "--pose-carrier-residual-scale": 1.0,
        "--pose-carrier-s-t": 0.044,
        "--pose-carrier-s-r": 0.0,
        "--pose-carrier-pitch": 0.0,
    }


def test_generated_pose_dispatch_changes_only_even_frame_with_dxi():
    cfg = CudaLevelSetConfig(
        n_pairs=2, in_feat=5, hidden_dim=8, n_hidden=1, mod_dim=4,
        render_h=6, render_w=8, camera_h=10, camera_w=12,
    )
    model = TorchLevelSetWitness.build(cfg, seed=13)
    geom = GroundHomographyGeom.eon(native_hw=(cfg.camera_h, cfg.camera_w), pitch=0.0)
    carrier = TorchPoseCarrier.build(np.zeros((cfg.n_pairs, 6), np.float32), geom)
    model.pose_carrier = carrier  # child attach before EMA/optimizer in production
    feats = torch.randn(cfg.render_h * cfg.render_w, cfg.in_feat)
    calls = {"out_sdf": 0}
    hook = model.out_sdf.register_forward_hook(
        lambda *_args: calls.__setitem__("out_sdf", calls["out_sdf"] + 1)
    )
    before, _ = _generated_pose_pair_dispatch(model, feats, [0, 1], carrier, cfg)
    assert calls["out_sdf"] == 2  # one B=2 f0 + one B=2 f1 witness forward
    hook.remove()
    with torch.no_grad():
        carrier.dxi[0, 0] = 0.02
        carrier.dxi[0, 5] = 0.01
    after, _ = _generated_pose_pair_dispatch(model, feats, [0, 1], carrier, cfg)
    assert not torch.allclose(before[0, 0], after[0, 0])
    assert torch.equal(before[1, 0], after[1, 0])
    assert torch.equal(before[:, 1], after[:, 1])
    after[:, 0].square().mean().backward()
    assert carrier.dxi.grad is not None
    assert float(carrier.dxi.grad[0].abs().sum()) > 0.0


def test_typed_pose_attach_registers_child_before_optimizer_and_ema():
    cfg = CudaLevelSetConfig(
        n_pairs=3, in_feat=5, hidden_dim=8, n_hidden=1, mod_dim=4,
        render_h=6, render_w=8, camera_h=10, camera_w=12,
    )
    model = TorchLevelSetWitness.build(cfg, seed=2)
    carrier, row = _attach_generated_pose_carrier(
        model, _pose_flags(), np.zeros((3, 6), np.float32), (10, 12), torch.device("cpu")
    )
    assert carrier is model.pose_carrier and row["s_t"] == pytest.approx(0.044)
    names = dict(model.named_parameters())
    assert "pose_carrier.dxi" in names
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert any(names["pose_carrier.dxi"] is p for g in opt.param_groups for p in g["params"])


def test_structured_prefit_uses_typed_values_and_skips_resume():
    cfg = CudaLevelSetConfig(
        n_pairs=2, in_feat=5, hidden_dim=12, n_hidden=1, mod_dim=4,
        render_h=12, render_w=16,
    )
    model = TorchLevelSetWitness.build(cfg, seed=4)
    feats = torch.randn(cfg.render_h * cfg.render_w, cfg.in_feat)
    lstars = np.zeros((4, cfg.render_h, cfg.render_w), np.int64)
    lstars[:, :3] = 2
    lstars[:, 9:] = 4
    lstars[:, 4:9, 7:8] = 1
    lstars[:, 5:7, 2:5] = 3
    flags = {
        "--structured-init": True,
        "--structured-init-include-lane": True,
        "--structured-init-thresh": 0.5,
        "--structured-init-steps": 10,
        "--structured-init-lr": 5e-3,
        "--structured-init-subsample": 192,
        "--structured-init-sdf-clip": 20.0,
    }
    code_before = model.code.detach().clone()
    row = _run_structured_prefit(model, flags, lstars, feats, seed=4, is_resume=False)
    assert row["active"] and row["applied"] and row["steps"] == 10
    assert torch.equal(model.code, code_before)
    frozen = copy.deepcopy(model.state_dict())
    skipped = _run_structured_prefit(model, flags, lstars, feats, seed=4, is_resume=True)
    assert skipped["reason"] == "resume_preserves_checkpoint"
    assert all(torch.equal(v, frozen[k]) for k, v in model.state_dict().items())


def test_checkpoint_roundtrip_restores_pair_cursor_and_controller_state():
    model = torch.nn.Linear(2, 1)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cursor = DeterministicPairCursor(7, seed=11)
    cursor.begin_epoch(3)
    first = cursor.next_epoch_indices(3)
    controllers = {"event": {"engaged": True}, "ladder": {"rung": 2}}
    seed = TorchProtectedIslandSeed(
        torch.ones(7, 2, 3, 3),
        torch.ones(7, 2, 3, dtype=torch.bool),
        mode="shield",
        damp=0.1,
    )
    seed_opt = torch.optim.AdamW(seed.parameters(), lr=0.02, weight_decay=0.0)
    tail = TailController(
        TailCycleConfig(k_max=2, cycle_floor_epochs=20, dwell_min=5),
        tau_ref=0.8,
        lr_ref=1e-3,
        tau0=0.8,
    )
    tail.step(3, [(2, 0.1)])
    blob = copy.deepcopy(_checkpoint_blob(
        model, ema, opt, 3, "cfg", ("python", "trainer"),
        pair_cursor=cursor, controller_state=controllers,
        protected_seed=seed, seed_optimizer=seed_opt,
        tail_controller=tail,
        scorer_sha256={"segnet": _SEG_SHA, "posenet": _POSE_SHA},
    ))
    assert blob["scorer_sha256"] == {"segnet": _SEG_SHA, "posenet": _POSE_SHA}
    with torch.no_grad():
        seed.residual.zero_()
    restored_cursor = DeterministicPairCursor(7, seed=0)
    restored_controllers = {"stale": True}
    restored_tail = TailController(
        TailCycleConfig(k_max=2, cycle_floor_epochs=20, dwell_min=5),
        tau_ref=0.8,
        lr_ref=1e-3,
        tau0=0.8,
    )
    epoch = _restore(
        blob, model, ema, opt, "cfg",
        pair_cursor=restored_cursor, controller_state=restored_controllers,
        protected_seed=seed, seed_optimizer=seed_opt,
        tail_controller=restored_tail,
    )
    assert epoch == 3 and restored_controllers == controllers
    assert torch.all(seed.residual == 1.0)
    assert restored_tail.state_dict() == tail.state_dict()
    rest = []
    while not restored_cursor.epoch_complete():
        rest.extend(restored_cursor.next_epoch_indices(3))
    assert sorted(first + rest) == list(range(7))


def test_torch_schedule_helpers_match_mlx_authority_endpoints_and_shapes():
    flags = {
        "--softmax-temp-start": 1.0,
        "--softmax-temp-end": 0.25,
        "--tau-anneal-shape": "geometric",
        "--anneal-epochs": 9,
        "--hosc-beta": 1.0,
        "--hosc-beta-end": 5.0,
        "--hosc-beta-anneal": "cosine",
    }
    assert _softmax_temp_at_epoch(1, 9, flags) == 1.0
    assert _softmax_temp_at_epoch(9, 9, flags) == 0.25
    assert _softmax_temp_at_epoch(5, 9, flags) == pytest.approx(0.5)
    assert _hosc_beta_at_epoch(1, 9, flags) == 1.0
    assert _hosc_beta_at_epoch(9, 9, flags) == 5.0
    assert _hosc_beta_at_epoch(5, 9, flags) == pytest.approx(3.0)


def test_accumulated_pair_step_matches_one_step_on_mean_accepted_loss():
    model = torch.nn.Linear(2, 1, bias=False)
    expected = copy.deepcopy(model)
    xs = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, -1.0]])]
    ys = [torch.tensor([[0.5]]), torch.tensor([[-0.2]])]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt_expected = torch.optim.SGD(expected.parameters(), lr=0.1)

    row = _accumulated_pair_step(
        model, opt, [0, 1],
        lambda i: (model(xs[i]) - ys[i]).square().mean(),
        grad_clip=1e9,
    )
    opt_expected.zero_grad(set_to_none=True)
    mean_loss = torch.stack([
        (expected(xs[i]) - ys[i]).square().mean() for i in (0, 1)
    ]).mean()
    mean_loss.backward()
    opt_expected.step()
    assert row["accepted"] == 1 and row["attempted"] == 1
    assert row["pair_count"] == 2
    assert row["accepted_frac"] == 1.0 and row["weights_stepped"]
    assert torch.allclose(model.weight, expected.weight, atol=1e-7)


def test_accumulated_pair_step_rejects_entire_chunk_without_partial_update():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    before = model.weight.detach().clone()
    row = _accumulated_pair_step(
        model, opt, [0, 1, 2],
        lambda i: None if i == 1 else model(torch.tensor([[2.0]])).square().mean(),
        grad_clip=1e9,
    )
    assert row["accepted"] == 0 and row["attempted"] == 1
    assert row["accepted_frac"] == 0.0 and not row["weights_stepped"]
    assert row["pair_count"] == 3
    assert torch.equal(model.weight, before)


def test_accumulated_pair_step_visits_each_pair_once():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    visited = []

    def loss_builder(i):
        visited.append(i)
        return model(torch.tensor([[float(i + 1)]])).square().mean()

    row = _accumulated_pair_step(
        model, opt, [3, 1, 2, 0], loss_builder, grad_clip=1e9,
    )
    assert visited == [3, 1, 2, 0]
    assert row["pair_count"] == 4 and row["weights_stepped"]


def test_runtime_stop_is_additional_bounded_and_outside_typed_hash():
    assert _runtime_epoch_window(0, 3000, 3) == (1, 3)
    assert _runtime_epoch_window(219, 3000, 3) == (220, 222)
    assert _runtime_epoch_window(2999, 3000, 3) == (3000, 3000)
    with pytest.raises(ValueError, match="zero-work"):
        _runtime_epoch_window(3000, 3000, 3)
    for invalid in (0, 4):
        with pytest.raises(ValueError, match=r"1\.\.3"):
            _runtime_epoch_window(10, 3000, invalid)

    parser = build_parser()
    base = [
        "--epochs", "3000", "--num-pairs", "1",
        "--gt-cache", "unused.npz", "--out-dir", "experiments/results/money_test",
    ]
    one = parser.parse_args(
        [*base, "--stop-after-epochs", "1", *_scorer_sha_args()]
    )
    three = parser.parse_args([*base, "--stop-after-epochs", "3"])
    assert derive_config(one)[1] == derive_config(three)[1]


def test_runtime_stop_refreshes_full_resume_after_every_completed_epoch():
    assert [
        _canonical_checkpoint_due(
            epoch,
            ckpt_every=50,
            run_end_epoch=3,
            runtime_stop_after_epochs=3,
            tail_stop_after_epoch=False,
        )
        for epoch in (1, 2, 3)
    ] == [True, True, True]
    assert [
        _canonical_checkpoint_due(
            epoch,
            ckpt_every=2,
            run_end_epoch=5,
            runtime_stop_after_epochs=None,
            tail_stop_after_epoch=False,
        )
        for epoch in (1, 2, 3, 4, 5)
    ] == [False, True, False, True, True]
    assert _canonical_checkpoint_due(
        3,
        ckpt_every=50,
        run_end_epoch=5,
        runtime_stop_after_epochs=None,
        tail_stop_after_epoch=True,
    )


def test_strict_resume_binds_scorer_custody_and_legacy_non_strict_survives(tmp_path):
    model = torch.nn.Linear(2, 1)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    blob = _checkpoint_blob(
        model,
        ema,
        optimizer,
        11,
        "cfg",
        ("python", "trainer"),
        scorer_sha256={"segnet": _SEG_SHA, "posenet": _POSE_SHA},
    )
    path = tmp_path / TORCH_RESUME_PT
    torch.save(blob, path)
    loaded = _load_validated_resume(
        path,
        "cfg",
        3000,
        expected_scorer_sha256={"segnet": _SEG_SHA, "posenet": _POSE_SHA},
        require_scorer_custody=True,
    )
    assert loaded["epoch"] == 11
    with pytest.raises(ValueError, match="differs from the strict execution plan"):
        _load_validated_resume(
            path,
            "cfg",
            3000,
            expected_scorer_sha256={"segnet": "3" * 64, "posenet": _POSE_SHA},
            require_scorer_custody=True,
        )

    legacy = dict(blob)
    legacy.pop("scorer_sha256")
    torch.save(legacy, path)
    with pytest.raises(ValueError, match="missing scorer SHA-256 custody"):
        _load_validated_resume(
            path,
            "cfg",
            3000,
            expected_scorer_sha256={"segnet": _SEG_SHA, "posenet": _POSE_SHA},
            require_scorer_custody=True,
        )
    assert _load_validated_resume(path, "cfg", 3000)["epoch"] == 11


def test_resume_is_explicit_and_gt_cache_geometry_fails_closed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert _resolve_resume_intent(out, None, no_implicit_resume=True) is None
    (out / "remote_asset_custody.json").write_text("{}")
    assert _resolve_resume_intent(out, None, no_implicit_resume=True) is None
    (out / "receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="implicit resume/overwrite"):
        _resolve_resume_intent(out, None, no_implicit_resume=True)
    with pytest.raises(ValueError, match="does not exist"):
        _resolve_resume_intent(
            tmp_path / "fresh",
            str(tmp_path / "missing.pt"),
            no_implicit_resume=True,
        )

    legacy_out = tmp_path / "legacy"
    legacy_out.mkdir()
    legacy_checkpoint = legacy_out / TORCH_RESUME_PT
    legacy_checkpoint.write_bytes(b"checkpoint fixture")
    assert (
        _resolve_resume_intent(legacy_out, None, no_implicit_resume=False)
        == legacy_checkpoint
    )
    assert (
        _resolve_resume_intent(tmp_path / "legacy_fresh", None, no_implicit_resume=False)
        is None
    )

    cache = tmp_path / "gt.npz"
    np.savez(
        cache,
        n_pairs=np.asarray(2),
        lstars=np.zeros((2, 4, 5), np.int64),
        margins=np.zeros((2, 4, 5), np.float32),
        gt_f1=np.zeros((2, 8, 9, 3), np.uint8),
        gt_poses=np.zeros((2, 6), np.float32),
    )
    loaded = _load_validated_gt_cache(str(cache), 1)
    assert loaded["lstars"].shape == (1, 4, 5)
    bad = tmp_path / "bad.npz"
    np.savez(
        bad,
        n_pairs=np.asarray(2),
        lstars=np.zeros((2, 4, 5), np.int64),
        margins=np.zeros((2, 4, 6), np.float32),
        gt_f1=np.zeros((2, 8, 9, 3), np.uint8),
        gt_poses=np.zeros((2, 6), np.float32),
    )
    with pytest.raises(ValueError, match="identical"):
        _load_validated_gt_cache(str(bad), 1)


def test_preflight_only_validates_exact_inputs_without_output_or_cuda(
    tmp_path, monkeypatch, capsys
):
    cache = tmp_path / "gt_exact.npz"
    np.savez(
        cache,
        n_pairs=np.asarray(1),
        lstars=np.zeros((1, 384, 512), np.uint8),
        margins=np.zeros((1, 384, 512), np.float32),
        gt_f1=np.zeros((1, 874, 1164, 3), np.uint8),
        gt_poses=np.zeros((1, 6), np.float32),
    )
    out = tmp_path / "preflight_must_not_exist"
    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch._FORBIDDEN_TMP",
        (),
    )
    load_calls = _patch_preflight_scorers(monkeypatch)
    rc = main(
        [
            "--preflight-only",
            "--no-implicit-resume",
            "--gt-cache",
            str(cache),
            "--num-pairs",
            "1",
            "--epochs",
            "3000",
            "--stop-after-epochs",
            "3",
            "--out-dir",
            str(out),
            "--device",
            "cuda",
            *_scorer_sha_args(),
        ]
    )
    receipt = json.loads(capsys.readouterr().out)
    assert rc == 0 and receipt["schema"] == "v9_cgauge_torch_preflight.v1"
    assert receipt["status"] == "passed"
    assert receipt["gt_geometry"] == {
        "render_hw": [384, 512],
        "camera_hw": [874, 1164],
    }
    assert receipt["runtime_epoch_window"] == [1, 3]
    assert receipt["no_implicit_resume"] is True
    assert load_calls == ["cpu"]
    assert receipt["scorer_constructor_load"]["status"] == "passed"
    assert receipt["scorer_constructor_load"]["device"] == "cpu"
    assert all(
        row == {"class": "Linear", "eval": True, "frozen": True}
        for row in receipt["scorer_constructor_load"]["networks"].values()
    )
    assert receipt["output_created"] is False and not out.exists()


def test_preflight_only_refuses_exact_geometry_mismatch_before_output(
    tmp_path, monkeypatch
):
    cache = tmp_path / "gt_bad_geometry.npz"
    np.savez(
        cache,
        n_pairs=np.asarray(1),
        lstars=np.zeros((1, 383, 512), np.uint8),
        margins=np.zeros((1, 383, 512), np.float32),
        gt_f1=np.zeros((1, 874, 1164, 3), np.uint8),
        gt_poses=np.zeros((1, 6), np.float32),
    )
    out = tmp_path / "preflight_must_not_exist"
    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch._FORBIDDEN_TMP",
        (),
    )
    _patch_preflight_scorers(monkeypatch)
    with pytest.raises(ValueError, match="typed render geometry"):
        main(
            [
                "--preflight-only",
                "--no-implicit-resume",
                "--gt-cache",
                str(cache),
                "--num-pairs",
                "1",
                "--epochs",
                "3000",
                "--stop-after-epochs",
                "3",
                "--out-dir",
                str(out),
                *_scorer_sha_args(),
            ]
        )
    assert not out.exists()


def test_preflight_only_refuses_resume_hash_mismatch_before_output(
    tmp_path, monkeypatch
):
    cache = tmp_path / "gt_exact.npz"
    np.savez(
        cache,
        n_pairs=np.asarray(1),
        lstars=np.zeros((1, 384, 512), np.uint8),
        margins=np.zeros((1, 384, 512), np.float32),
        gt_f1=np.zeros((1, 874, 1164, 3), np.uint8),
        gt_poses=np.zeros((1, 6), np.float32),
    )
    resume = tmp_path / TORCH_RESUME_PT
    torch.save(
        {
            "schema": "v9_cgauge_torch_resume_v2",
            "epoch": 17,
            "model": {},
            "ema": {},
            "optimizer": {},
            "torch_rng": None,
            "numpy_rng": None,
            "python_rng": None,
            "config_hash": "0" * 64,
            "dsl_argv": [],
        },
        resume,
    )
    out = tmp_path / "preflight_must_not_exist"
    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch._FORBIDDEN_TMP",
        (),
    )
    _patch_preflight_scorers(monkeypatch)
    with pytest.raises(ValueError, match="config hash differs"):
        main(
            [
                "--preflight-only",
                "--no-implicit-resume",
                "--resume-from",
                str(resume),
                "--gt-cache",
                str(cache),
                "--num-pairs",
                "1",
                "--epochs",
                "3000",
                "--stop-after-epochs",
                "3",
                "--out-dir",
                str(out),
                *_scorer_sha_args(),
            ]
        )
    assert not out.exists()


def test_scorer_custody_validates_expected_hash_headers_missing_and_symlink(tmp_path):
    from safetensors.numpy import save_file

    seg = tmp_path / "segnet.safetensors"
    pose = tmp_path / "posenet.safetensors"
    save_file({"weight": np.ones((2, 3), np.float32)}, seg)
    save_file({"weight": np.zeros((3, 2), np.float32)}, pose)
    expected = {
        "segnet": hashlib.sha256(seg.read_bytes()).hexdigest(),
        "posenet": hashlib.sha256(pose.read_bytes()).hexdigest(),
    }
    receipt = _validate_scorer_custody(
        {"segnet": seg, "posenet": pose}, expected_sha256=expected
    )
    assert receipt["segnet"]["sha_authority"] == "PLAN_EXPECTED_MATCH"
    assert receipt["posenet"]["tensor_count"] == 1

    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="not parseable"):
        _validate_scorer_custody(
            {"segnet": corrupt, "posenet": pose},
            expected_sha256={
                "segnet": hashlib.sha256(corrupt.read_bytes()).hexdigest(),
                "posenet": expected["posenet"],
            },
        )
    with pytest.raises(ValueError, match="regular non-symlink"):
        _validate_scorer_custody(
            {"segnet": tmp_path / "missing.safetensors", "posenet": pose}
        )
    link = tmp_path / "segnet-link.safetensors"
    link.symlink_to(seg)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _validate_scorer_custody({"segnet": link, "posenet": pose})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_scorer_custody(
            {"segnet": seg, "posenet": pose},
            expected_sha256={"segnet": "0" * 64, "posenet": expected["posenet"]},
        )


def test_strict_preflight_requires_both_valid_expected_scorer_hashes():
    with pytest.raises(ValueError, match="requires expected SegNet and PoseNet"):
        main(["--preflight-only", "--no-implicit-resume"])
    with pytest.raises(ValueError, match="requires both"):
        main(
            [
                "--preflight-only",
                "--no-implicit-resume",
                "--expected-segnet-sha256",
                _SEG_SHA,
            ]
        )
    with pytest.raises(ValueError, match="64 hex"):
        main(
            [
                "--preflight-only",
                "--no-implicit-resume",
                "--expected-segnet-sha256",
                "not-a-hash",
                "--expected-posenet-sha256",
                _POSE_SHA,
            ]
        )


def test_preflight_only_refuses_incomplete_cuda_port_before_output(
    tmp_path, monkeypatch
):
    cache = tmp_path / "gt_exact.npz"
    np.savez(
        cache,
        n_pairs=np.asarray(1),
        lstars=np.zeros((1, 384, 512), np.uint8),
        margins=np.zeros((1, 384, 512), np.float32),
        gt_f1=np.zeros((1, 874, 1164, 3), np.uint8),
        gt_poses=np.zeros((1, 6), np.float32),
    )
    out = tmp_path / "preflight_must_not_exist"
    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch._FORBIDDEN_TMP",
        (),
    )
    load_calls = _patch_preflight_scorers(monkeypatch, coverage_status="BLOCKED")
    with pytest.raises(RuntimeError, match="incomplete V9 CUDA control semantics"):
        main(
            [
                "--preflight-only",
                "--no-implicit-resume",
                "--gt-cache",
                str(cache),
                "--num-pairs",
                "1",
                "--epochs",
                "3000",
                "--stop-after-epochs",
                "3",
                "--out-dir",
                str(out),
                *_scorer_sha_args(),
            ]
        )
    assert not out.exists()
    assert load_calls == []


def test_preflight_only_scorer_architecture_load_failure_precedes_output_and_cuda(
    tmp_path, monkeypatch
):
    cache = tmp_path / "gt_exact.npz"
    np.savez(
        cache,
        n_pairs=np.asarray(1),
        lstars=np.zeros((1, 384, 512), np.uint8),
        margins=np.zeros((1, 384, 512), np.float32),
        gt_f1=np.zeros((1, 874, 1164, 3), np.uint8),
        gt_poses=np.zeros((1, 6), np.float32),
    )
    out = tmp_path / "preflight_must_not_exist"
    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch._FORBIDDEN_TMP",
        (),
    )
    _patch_preflight_scorers(monkeypatch)
    load_devices = []

    def fail_load(device):
        load_devices.append(str(device))
        raise RuntimeError("state-dict architecture mismatch")

    monkeypatch.setattr(
        "experiments.train_levelset_witness_realized_through_R_torch._load_scorers",
        fail_load,
    )
    with pytest.raises(RuntimeError, match="state-dict architecture mismatch"):
        main(
            [
                "--preflight-only",
                "--no-implicit-resume",
                "--gt-cache",
                str(cache),
                "--num-pairs",
                "1",
                "--epochs",
                "3000",
                "--stop-after-epochs",
                "3",
                "--out-dir",
                str(out),
                "--device",
                "cuda",
                *_scorer_sha_args(),
            ]
        )
    assert load_devices == ["cpu"]
    assert not out.exists()
