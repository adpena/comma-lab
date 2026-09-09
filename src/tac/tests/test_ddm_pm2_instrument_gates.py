"""Incident-shaped fixtures only: no scorer or research encode is launched."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from comma_lab import instrument_gates as gates


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def custody(tmp_path, monkeypatch):
    monkeypatch.delenv("TAC_INSTRUMENT_GATES", raising=False)
    live = tmp_path / "live"
    live.mkdir()
    (live / "archive.zip").write_bytes(b"retained archive fixture")
    (live / "inflate.py").write_text("print('fixture receiver')\n")
    sha = hashlib.sha256((live / "archive.zip").read_bytes()).hexdigest()
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    write_json(
        receipt, {"expected_archive_sha256": sha, "avg_posenet_dist": 0.000005, "n_samples": 600, "passed": True, "expected_runtime_tree_sha256": "distinct-modal-digest"}
    )
    mirror = tmp_path / "mirror.json"
    write_json(
        mirror, {"archive_sha256": sha, "source_receipt": str(receipt), "source_receipt_sha256": gates._sha(receipt)}
    )
    pointer = tmp_path / "pointer.json"
    write_json(
        pointer,
        {
            "our_local_frontier_contest_cuda": {
                "source_path": str(mirror),
                "archive_sha256": sha,
                "extra": {"runtime_tree_sha256": "distinct-modal-digest", "source_receipt_sha256": gates._sha(receipt)},
            }
        },
    )
    manifest = tmp_path / "FIRE_MANIFEST.json"
    write_json(
        manifest,
        {
            "runtime_dir": str(live),
            "stage3_runtime_digests": {
                "seal_runtime": {
                    "digest_definition": gates.DIGEST_DEFINITION,
                    "sha256": gates.measure_runtime_digest(live).sha256,
                },
                "modal_uploaded_runtime": {"runtime_tree_sha256": "distinct-modal-digest"},
            },
        },
    )
    monkeypatch.setattr(gates, "POINTER", pointer)
    return SimpleNamespace(live=live, pointer=pointer, receipt=receipt, mirror=mirror, manifest=manifest, tmp=tmp_path)


@pytest.mark.parametrize("ratio", [1 / 3, 0.8, 1, 1.2, 3])
def test_pose_band_accepts_promoted_scale(custody, ratio):
    assert gates.check_pose_base(0.000005 * ratio)["valid"]


@pytest.mark.parametrize("ratio", [0, 0.32, 3.01, 500])
def test_missing_overlay_shape_refused(custody, ratio, capsys):
    with pytest.raises(gates.ConfoundAlarm, match="missing --overlay"):
        gates.check_pose_base(0.000005 * ratio)
    assert '"event": "confound_alarm"' in capsys.readouterr().err


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_pose_cannot_be_waived(custody, value):
    with pytest.raises(gates.ConfoundAlarm):
        gates.check_pose_base(value, rationale="Historic axis calibration comparison")


@pytest.mark.parametrize("reason", ["", "TODO", "<rationale>", "placeholder measurement reason", "okay okay okay"])
def test_placeholder_waivers_refused(custody, reason):
    with pytest.raises(gates.ConfoundAlarm, match="non-placeholder"):
        gates.check_pose_base(0.001, rationale=reason)


def test_explicit_pose_waiver_remains_invalid(custody):
    result = gates.check_pose_base(0.001, rationale="Historic no-overlay calibration comparison")
    assert result["event"] == "instrument_gate_waiver" and not result["valid"]


def test_pointer_move_during_measurement_refused(custody):
    snapshot = gates.snapshot_pose_pointer()
    custody.pointer.write_text(custody.pointer.read_text() + "\n")
    with pytest.raises(gates.ConfoundAlarm, match="moved"):
        gates.check_pose_base(0.000005, snapshot=snapshot)


def test_source_receipt_hash_drift_refused(custody):
    custody.receipt.write_text("{}")
    with pytest.raises(gates.ConfoundAlarm, match="hash changed"):
        gates.check_pose_base(0.000005)


def test_same_archive_receipt_substitution_refused(custody):
    receipt = json.loads(custody.receipt.read_text())
    receipt["avg_posenet_dist"] = 0.003
    write_json(custody.receipt, receipt)
    mirror = json.loads(custody.mirror.read_text())
    mirror["source_receipt_sha256"] = gates._sha(custody.receipt)
    write_json(custody.mirror, mirror)
    with pytest.raises(gates.ConfoundAlarm, match="pointer's pinned receipt"):
        gates.check_pose_base(0.003)


def test_same_archive_wrong_runtime_receipt_refused(custody):
    receipt = json.loads(custody.receipt.read_text())
    receipt["expected_runtime_tree_sha256"] = "different-runtime"
    write_json(custody.receipt, receipt)
    for path in (custody.mirror, custody.pointer):
        obj = json.loads(path.read_text())
        target = obj if path == custody.mirror else obj["our_local_frontier_contest_cuda"]["extra"]
        target["source_receipt_sha256"] = gates._sha(custody.receipt)
        write_json(path, obj)
    with pytest.raises(gates.ConfoundAlarm, match="different runtime"):
        gates.check_pose_base(0.000005)


def test_wrong_archive_component_refused(custody):
    mirror = json.loads(custody.mirror.read_text())
    mirror["archive_sha256"] = "wrong"
    write_json(custody.mirror, mirror)
    with pytest.raises(gates.ConfoundAlarm, match="different archive"):
        gates.check_pose_base(0.000005)


def test_pair_reference_not_population_mean():
    assert gates.check_pose_pair(0.001, 0.0011)["valid"]


def test_zero_pair_reference():
    assert gates.check_pose_pair(0, 0)["valid"]
    with pytest.raises(gates.ConfoundAlarm):
        gates.check_pose_pair(0.000001, 0)


def test_encoder_tree_identical_copy_passes(custody):
    clone = custody.tmp / "clone"
    shutil.copytree(custody.live, clone)
    result = gates.check_pointer_coder(clone)
    assert result["valid"] and result["encoder_sha256"] == result["pointer_sha256"]


def test_stale_cl2_encoder_refused_before_encode(custody):
    old = custody.tmp / "cl2"
    shutil.copytree(custody.live, old)
    (old / "inflate.py").write_text("print('old HPAC coder')\n")
    with pytest.raises(gates.ConfoundAlarm, match="encoder_not_pointer_coder"):
        gates.check_pointer_coder(old)


def test_changed_coder_waiver_explicit(custody):
    old = custody.tmp / "cl2"
    shutil.copytree(custody.live, old)
    (old / "inflate.py").write_text("print('new experimental coder')\n")
    result = gates.check_pointer_coder(old, rationale="Comparing experimental mixer against shipping HPAC")
    assert not result["valid"] and result["event"] == "instrument_gate_waiver"


def test_digest_definitions_are_not_compared(custody):
    result = gates.check_pointer_coder(custody.live)
    assert result["pointer_sha256"] != "distinct-modal-digest"
    assert result["digest_definition"] == gates.DIGEST_DEFINITION


def test_live_tree_drift_cannot_be_waived(custody):
    (custody.live / "inflate.py").write_text("drift")
    with pytest.raises(gates.ConfoundAlarm, match="drifted"):
        gates.check_pointer_coder(custody.live, rationale="Comparing experimental mixer against shipping HPAC")


def test_wrong_fire_manifest_refused(custody):
    manifest = json.loads(custody.manifest.read_text())
    manifest["stage3_runtime_digests"]["modal_uploaded_runtime"]["runtime_tree_sha256"] = "wrong"
    write_json(custody.manifest, manifest)
    with pytest.raises(gates.ConfoundAlarm, match="not bound"):
        gates.check_pointer_coder(custody.live)


def test_empty_encoder_refused(custody):
    root = custody.tmp / "empty"
    root.mkdir()
    with pytest.raises(gates.ConfoundAlarm, match="empty"):
        gates.check_pointer_coder(root)


def test_missing_pointer_fails_closed(tmp_path):
    with pytest.raises(gates.ConfoundAlarm):
        gates.check_pointer_coder(tmp_path, pointer_path=tmp_path / "missing")


def test_missing_encoder_is_typed_alarm(custody):
    with pytest.raises(gates.ConfoundAlarm, match="coder_custody"):
        gates.check_pointer_coder(custody.tmp / "absent")


def test_new_experiment_codec_requires_rationale_even_on_live_source(custody):
    with pytest.raises(gates.ConfoundAlarm, match="experimental_coder"):
        gates.check_experimental_coder(custody.live)


def test_experiment_codec_has_explicit_door(custody):
    result = gates.check_experimental_coder(custody.live, rationale="Price novel semantic coder against current source")
    assert result["kind"] == "experimental_coder" and not result["valid"]


@pytest.mark.parametrize("entry", ["pose", "coder", "experiment"])
def test_midrun_env_escape_is_loud(custody, monkeypatch, capsys, entry):
    monkeypatch.setenv("TAC_INSTRUMENT_GATES", "0")
    result = {
        "pose": lambda: gates.check_pose_base(0.03),
        "coder": lambda: gates.check_pointer_coder(Path("/absent")),
        "experiment": lambda: gates.check_experimental_coder(Path("/absent")),
    }[entry]()
    assert not result["valid"] and "instrument_gate_bypass" in capsys.readouterr().err


def test_cli_flags():
    parser = argparse.ArgumentParser()
    gates.add_pose_gate_argument(parser)
    gates.add_coder_gate_argument(parser)
    args = parser.parse_args(["--coder-differs-because", "New shared entropy mixer experiment"])
    assert args.coder_differs_because and args.pose_base_differs_because is None


def test_jg2_refuses_before_build_or_store(custody, monkeypatch):
    from experiments import ddm_jg2_tail_reencode as jg2

    def no_build():
        pytest.fail("build started before gate")

    monkeypatch.setattr(jg2, "load_route_b", no_build)
    bad = custody.tmp / "bad"
    bad.mkdir()
    (bad / "inflate.py").write_text("old coder")
    store = custody.tmp / "must-not-exist"
    with pytest.raises(gates.ConfoundAlarm):
        jg2._prepare(SimpleNamespace(runtime_root=bad, store=store), "control")
    assert not store.exists()


@pytest.fixture
def instrument_sources(tmp_path):
    directory = tmp_path / "experiments"
    directory.mkdir()
    names = set(gates.POSE_BINDINGS) | set(gates.CODER_CLIS) | {"ddm_jg2_tail_reencode.py"}
    for name in names:
        shutil.copyfile(gates.REPO / "experiments" / name, directory / name)
    return tmp_path


def test_static_wiring_current_sources(instrument_sources):
    assert gates.audit_instrument_gate_wiring(instrument_sources) == []


@pytest.mark.parametrize("name,function,guard", [
    ("ddm_sj1_joint_admission.py", "cmd_pose", "check_pose_base"),
    ("ddm_rp1_pose.py", "cmd_pose", "check_pose_base"),
    ("ddm_fe1_admit_and_build.py", "cmd_base_pose", "check_pose_base"),
    ("ddm_fe1_pose_price.py", "cmd_price", "evaluate_base_codes"),
    ("ddm_jg2_tail_reencode.py", "_prepare", "check_pointer_coder"),
])
def test_static_guard_deletion_fails(instrument_sources, name, function, guard):
    path = instrument_sources / "experiments" / name
    source = path.read_text().replace(guard + "(", "unguarded_measurement(")
    path.write_text(source)
    assert any(function in finding for finding in gates.audit_instrument_gate_wiring(instrument_sources))


def test_static_coder_after_build_fails(instrument_sources):
    path = instrument_sources / "experiments/ddm_jg2_tail_reencode.py"
    path.write_text('def _prepare(args, tag):\n    compile_rc64()\n    check_pointer_coder(args.runtime_root)\n')
    assert any("after persistence" in item for item in gates.audit_instrument_gate_wiring(instrument_sources))


@pytest.mark.parametrize("name", gates.CODER_CLIS)
def test_experimental_cli_guard_deletion_fails(instrument_sources, name):
    path = instrument_sources / "experiments" / name
    path.write_text(path.read_text().replace("check_experimental_coder(", "wrong_guard("))
    assert any(name in item for item in gates.audit_instrument_gate_wiring(instrument_sources))


def test_live_umbrella_is_strict(instrument_sources):
    from tac.preflight import PreflightError, check_instrument_binds_to_live_pointer

    (instrument_sources / "experiments/ddm_jg2_tail_reencode.py").write_text('def _prepare(args, tag):\n    compile_rc64()\n')
    with pytest.raises(PreflightError):
        check_instrument_binds_to_live_pointer(repo_root=instrument_sources, strict=True)


@pytest.mark.parametrize("body", [
    '    def unused():\n        check_pointer_coder(args.runtime_root)\n    compile_rc64()\n',
    '    if False:\n        check_pointer_coder(args.runtime_root)\n    compile_rc64()\n',
])
def test_dead_gate_is_not_protection(instrument_sources, body):
    path = instrument_sources / "experiments/ddm_jg2_tail_reencode.py"
    path.write_text('def _prepare(args, tag):\n' + body)
    assert any("_prepare" in item for item in gates.audit_instrument_gate_wiring(instrument_sources))


def test_string_is_not_binding_waiver(instrument_sources):
    path = instrument_sources / "experiments/ddm_jg2_tail_reencode.py"
    path.write_text('def _prepare(args, tag="INSTRUMENT_BINDING_OK: deliberate experiment comparison"):\n    compile_rc64()\n')
    assert any("_prepare" in item for item in gates.audit_instrument_gate_wiring(instrument_sources))
