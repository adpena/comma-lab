"""Scorer-free execution tests for pose-base gates at artifact boundaries."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments"))
from comma_lab.instrument_gates import ConfoundAlarm


@pytest.fixture(params=["ddm_sj1_joint_admission", "ddm_rp1_pose"])
def pose_cli(request, monkeypatch):
    mod = importlib.import_module(request.param)
    inst = SimpleNamespace(
        state=SimpleNamespace(codes=np.zeros((600, 12)), coefficient_scales=np.ones(12)),
        posenet=None, raw=None, targets=None,
    )
    if "sj1" in request.param:
        monkeypatch.setattr(mod, "_set_threads", lambda *_: None)
        monkeypatch.setattr(mod, "load_pose_instrument", lambda *_: inst)
    else:
        monkeypatch.setattr(mod.rp1, "set_threads", lambda *_: None)
        monkeypatch.setattr(mod, "assert_pointer_and_carrier", lambda: {})
        monkeypatch.setattr(mod, "load_instrument", lambda *_: inst)
    monkeypatch.setattr(mod.up2, "codes_to_coefficients", lambda *a: a[0])
    monkeypatch.setattr(mod.up2, "measure_pose", lambda *a, **k: (np.full(600, .0025), None))
    return mod


def pose_args(tmp_path, tag="base", rationale=None):
    return SimpleNamespace(threads=1, overlay=None, codes=None, tag=tag,
                           batch_size=8, out=tmp_path / "new" / "base.npy",
                           pose_base_differs_because=rationale)


def test_original_missing_overlay_refuses_before_any_base_artifact(pose_cli, monkeypatch, tmp_path):
    monkeypatch.setattr(pose_cli, "snapshot_pose_pointer", lambda: {})
    def refuse(value, **kwargs):
        assert value == pytest.approx(.0025)
        raise ConfoundAlarm("pose_base_magnitude", "500x missing overlay")
    monkeypatch.setattr(pose_cli, "check_pose_base", refuse)
    with pytest.raises(ConfoundAlarm):
        pose_cli.cmd_pose(pose_args(tmp_path))
    assert not (tmp_path / "new").exists()


def test_success_retains_gate_receipt_and_exact_vector(pose_cli, monkeypatch, tmp_path):
    snapshot = {"capture": "before measurement"}
    monkeypatch.setattr(pose_cli, "snapshot_pose_pointer", lambda: snapshot)
    def check(value, **kwargs):
        assert kwargs["snapshot"] is snapshot
        return {"event": "instrument_gate_pass"}
    monkeypatch.setattr(pose_cli, "check_pose_base", check)
    args = pose_args(tmp_path)
    assert pose_cli.cmd_pose(args) == 0
    np.testing.assert_array_equal(np.load(args.out), np.full(600, .0025))
    assert json.loads((args.out.parent / "POSE_base.json").read_text())["pose_base_gate"] == {"event": "instrument_gate_pass"}


def test_candidate_measurement_is_not_misclassified_as_base(pose_cli, monkeypatch, tmp_path):
    def forbidden(*a, **kw):
        pytest.fail("candidate must not be compared to base magnitude")
    monkeypatch.setattr(pose_cli, "snapshot_pose_pointer", forbidden)
    monkeypatch.setattr(pose_cli, "check_pose_base", forbidden)
    assert pose_cli.cmd_pose(pose_args(tmp_path, "stale")) == 0


def test_explicit_waiver_reaches_gate(pose_cli, monkeypatch, tmp_path):
    monkeypatch.setattr(pose_cli, "snapshot_pose_pointer", lambda: {})
    rationale = "The intentionally changed carrier is a separate control"
    def check(value, **kwargs):
        assert kwargs["rationale"] == rationale
        return {"event": "instrument_gate_waiver", "valid": False}
    monkeypatch.setattr(pose_cli, "check_pose_base", check)
    assert pose_cli.cmd_pose(pose_args(tmp_path, rationale=rationale)) == 0


def test_gate_switch_is_on_parser(pose_cli, tmp_path):
    args = pose_cli.build_parser().parse_args([
        "pose", "--tag", "base", "--out", str(tmp_path / "base.npy"),
        "--pose-base-differs-because", "Purposefully different carrier under measurement",
    ])
    assert args.pose_base_differs_because.startswith("Purposefully")


@pytest.fixture
def pair_price(monkeypatch):
    mod = importlib.import_module("ddm_fe1_pose_price")
    monkeypatch.delenv("TAC_INSTRUMENT_GATES", raising=False)
    monkeypatch.setattr(mod, "snapshot_pose_pointer", lambda: {"d_pose": 5e-6})
    monkeypatch.setattr(mod, "check_pose_base", lambda *a, **kw: {"valid": True})
    return mod


def test_pair_screen_requires_population_reference(pair_price):
    with pytest.raises(ConfoundAlarm, match="reference"):
        pair_price.prepare_pose_reference(SimpleNamespace(base_mean_d_pose=5e-6))


@pytest.mark.parametrize("vector", [np.ones(12), np.full(600, np.nan), -np.ones(600)])
def test_bad_population_vector_refuses(pair_price, tmp_path, vector):
    path = tmp_path / "reference.npy"
    np.save(path, vector)
    with pytest.raises(ConfoundAlarm):
        pair_price.prepare_pose_reference(SimpleNamespace(base_mean_d_pose=5e-6, base_pose_reference=path))


def test_pair_compares_to_same_pair_not_population(pair_price, monkeypatch, tmp_path):
    reference = np.full(600, 5e-6)
    reference[14] = 0.001
    path = tmp_path / "reference.npy"
    np.save(path, reference)
    gate = pair_price.prepare_pose_reference(SimpleNamespace(base_mean_d_pose=5e-6, base_pose_reference=path))
    monkeypatch.setattr(pair_price.br1, "evaluate_codes", lambda *a: [.001])
    assert pair_price.evaluate_base_codes(None, 14, None, gate) == .001


def test_wrong_pair_base_refuses(pair_price, monkeypatch, tmp_path):
    path = tmp_path / "reference.npy"
    np.save(path, np.full(600, 5e-6))
    gate = pair_price.prepare_pose_reference(SimpleNamespace(base_mean_d_pose=5e-6, base_pose_reference=path))
    monkeypatch.setattr(pair_price.br1, "evaluate_codes", lambda *a: [.0025])
    with pytest.raises(ConfoundAlarm, match="pose_pair_magnitude"):
        pair_price.evaluate_base_codes(None, 14, None, gate)


def test_legacy_env_escape_avoids_reference_io(pair_price, monkeypatch):
    monkeypatch.setenv("TAC_INSTRUMENT_GATES", "0")
    def forbidden():
        pytest.fail("legacy escape must not resolve pointer")
    monkeypatch.setattr(pair_price, "snapshot_pose_pointer", forbidden)
    assert pair_price.prepare_pose_reference(SimpleNamespace())["disabled"] is True


def test_fe1_population_base_refuses_before_save(monkeypatch, tmp_path):
    mod = importlib.import_module("ddm_fe1_admit_and_build")
    monkeypatch.setattr(mod.fe1, "_set_threads", lambda *_: None)
    monkeypatch.setattr(mod.fe1, "_open_raw", lambda *_: None)
    state = SimpleNamespace(codes=np.zeros((600, 12)), coefficient_scales=np.ones(12))
    monkeypatch.setattr(mod.price, "build_pose_instrument", lambda *_: SimpleNamespace(state=state, posenet=None, raw=None, targets=None))
    monkeypatch.setattr(mod.price, "snapshot_pose_pointer", lambda: {})
    monkeypatch.setattr(mod.up2, "codes_to_coefficients", lambda *a: a[0])
    monkeypatch.setattr(mod.up2, "measure_pose", lambda *a, **k: (np.ones(600), None))
    def refuse(*a, **kw):
        raise ConfoundAlarm("pose_base_magnitude", "wrong tree")
    monkeypatch.setattr(mod.price, "check_pose_base", refuse)
    out = tmp_path / "new"
    with pytest.raises(ConfoundAlarm):
        mod.cmd_base_pose(SimpleNamespace(threads=1, batch_size=8, out_dir=out))
    assert not out.exists()


def test_rw1_missing_reference_refuses_before_instrument(monkeypatch, tmp_path):
    mod = importlib.import_module("ddm_rw1_renderer_edge_foldback")
    def refuse(*a, **kw):
        raise ConfoundAlarm("pose_base_reference", "reference required")
    monkeypatch.setattr(mod.pose_price, "prepare_pose_reference", refuse)
    def forbidden(*a):
        pytest.fail("no instrument or pointer work before reference admission")
    monkeypatch.setattr(mod, "verify_live_pointer", forbidden)
    with pytest.raises(ConfoundAlarm):
        mod.cmd_pose(SimpleNamespace())
    assert list(tmp_path.iterdir()) == []


def test_rw1_parser_accepts_reference(tmp_path):
    mod = importlib.import_module("ddm_rw1_renderer_edge_foldback")
    parser = mod.build_parser()
    args = parser.parse_args([
        "pose", "--out", str(tmp_path / "out.json"),
        "--out-rows", str(tmp_path / "rows.jsonl"), "--raw", str(tmp_path / "raw"),
        "--base-pose-reference", str(tmp_path / "base.npy"),
    ])
    assert args.base_pose_reference == tmp_path / "base.npy"
