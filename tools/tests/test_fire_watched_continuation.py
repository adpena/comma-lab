"""Incident-shaped controls for the local continuation-launch determinizer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO / "tools/fire_watched_continuation.py"
    spec = importlib.util.spec_from_file_location("fire_watched_continuation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FWC = _load_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _parent_fixture(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "full_e480b"
    save = parent / "checkpoints/full_mps_e480.pt"
    checkpoint = save.with_name(save.stem + ".checkpoints") / "qat_stage_end_epoch_0480.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"retained-parent-checkpoint")
    log = parent / "launcher/run.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        json.dumps(
            {
                "epoch": 480,
                "phase": "discrete_qat",
                "estimated_joint_bytes": 131220,
                "bpp": 0.1,
                "top1_error": 0.01,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    liveness = tmp_path / "parent_liveness.json"
    quality = tmp_path / "parent_quality.json"
    _write_json(
        liveness,
        {
            "schema": "pact.run_liveness_watcher.config.v1",
            "pid_file": str(parent / "launcher/run.pid"),
            "alert_path": str(parent / "watchers/liveness.alert.json"),
            "poll_s": 60,
            "initial_delay_s": 0,
            "warmup_s": 1800,
            "receipt_checks": [],
            "heartbeat_checks": [],
            "artifact_checks": [
                {
                    "label": "full_e480b_log",
                    "path": str(log),
                    "max_age_s": 10800,
                    "grace_s": 1800,
                }
            ],
        },
    )
    _write_json(
        quality,
        {
            "schema": "pact.run_quality_poller.config.v1",
            "log_path": str(log),
            "pid_file": str(parent / "launcher/run.pid"),
            "telemetry_path": str(parent / "watchers/quality.jsonl"),
            "alert_path": str(parent / "watchers/quality.alert.json"),
            "poll_s": 60,
            "eval_period_s": 3600,
            "stale_periods": 3,
            "startup_grace_s": 1800,
            "json_marker": "estimated_joint_bytes",
            "fields": {
                "epoch": "epoch",
                "value": "estimated_joint_bytes",
                "phase": "phase",
                "finite": ["bpp", "top1_error"],
            },
            "bar_value": 140000,
            "bar_start_epoch": 1,
            "phase_knee": {
                "epoch": 1,
                "window_epochs": 3,
                "shock_multiplier": 1.25,
                "continuous_phase": "continuous",
            },
            "best_not_latest": {"phase": "discrete_qat", "min_rows": 4, "lag_epochs": 6},
            "alert_conditions": {
                "joint_regression": False,
                "qat_knee_shock": False,
                "nan_or_garbage": True,
                "stale_telemetry": True,
            },
        },
    )
    argv = [
        ".venv/bin/python",
        "tools/train_ddm_cl1_hpac_capacity_mps.py",
        "--port-mode",
        "full-mps-e480",
        "--profile",
        "rx2_mc36",
        "--cache",
        str(tmp_path / "cache.pt"),
        "--init",
        str(tmp_path / "init.pt"),
        "--epochs",
        "480",
        "--batch-size",
        "8",
        "--eval-batch-size",
        "4",
        "--eval-every",
        "2",
        "--lr",
        "0.003",
        "--lr-exponent",
        "0.0002",
        "--lr-bits",
        "0.01",
        "--bit-eps",
        "1e-6",
        "--rate-lambda",
        "1.0",
        "--qat-fraction",
        "0.5",
        "--init-bits",
        "8.0",
        "--channels",
        "64",
        "--patch",
        "64",
        "--delta",
        "2",
        "--frame-dim",
        "8",
        "--norm-mode",
        "none",
        "--activation",
        "relu",
        "--frame-scale",
        "--weight-bound",
        "127",
        "--activation-bound",
        "127",
        "--weight-scales",
        "--weight-exponent-min",
        "-6",
        "--spm",
        "--target-mode",
        "raw",
        "--seed",
        "20260716",
        "--ema-target-seed-fraction",
        "0.01",
        "--device",
        "mps",
        "--save",
        str(save),
        "--out",
        str(parent / "reports/trainer.json"),
        "--min-free-bytes",
        "10737418240",
    ]
    _write_json(
        parent / "launcher/launch_manifest.json",
        {
            "schema": "detached_local_process_launch.v2",
            "argv": argv,
            "requested_nice": 10,
            "resource_budget": {
                "measured_peak_rss_gib": 10.278,
                "measured_thread_need": 6,
                "walltime_cap_s": 86400.0,
            },
            "watchers": [
                {"kind": "liveness", "config_path": str(liveness)},
                {"kind": "quality", "config_path": str(quality)},
            ],
        },
    )
    return parent, checkpoint


def test_source_parser_finds_all_live_reference_environment_gates() -> None:
    gates = FWC.parse_required_environment_gates(FWC.REFERENCE_TRAINER)
    assert gates == {
        "PYTHONHASHSEED": "0",
        "TAC_ADMISSION_ENFORCE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }


def test_source_parser_tracks_a_new_gate_without_a_hardcoded_list(tmp_path: Path) -> None:
    source = tmp_path / "trainer.py"
    source.write_text(
        "import os\n"
        "class CL1TrainingError(Exception): pass\n"
        "def main():\n"
        "    if os.environ.get('NEW_REFERENCE_GATE') != 'sealed':\n"
        "        raise CL1TrainingError('set NEW_REFERENCE_GATE=sealed before launch')\n",
        encoding="utf-8",
    )
    assert FWC.parse_required_environment_gates(source) == {"NEW_REFERENCE_GATE": "sealed"}


def test_compose_positive_control_seals_full_command_and_derived_watchers(tmp_path: Path) -> None:
    parent, checkpoint = _parent_fixture(tmp_path)
    run_root = tmp_path / "full_e480b_e960_control"
    receipt_name = f"lh2_control_{os.getpid()}_{abs(hash(str(tmp_path))) % 10_000_000}"
    result = FWC.compose_continuation(
        parent_run_dir=parent,
        port_mode="full-mps-e960",
        run_root=run_root,
        overrides=[],
        done_receipt=receipt_name,
        closer_deadline_s=3600,
        closer_poll_s=5,
    )
    assert result["schema"] == FWC.COMPOSITION_SCHEMA
    assert result["resume_checkpoint"]["sha256"] == FWC._sha256(checkpoint)
    assert result["watcher_configs"]["quality_bar_value"] == 131220
    assert result["watcher_configs"]["bar_start_epoch"] == 481
    assert result["environment_gate_verification"]["status"] == "PASS"
    assert (
        result["environment_gate_verification"]["source_required"]
        == result["environment_gate_verification"]["launcher_assignments"]
    )
    quality = json.loads((run_root / "launcher/generated_quality.json").read_text())
    assert quality["bar_value"] == 131220 and quality["bar_start_epoch"] == 481
    quality_log = Path(quality["log_path"])
    assert run_root in quality_log.parents and parent not in quality_log.parents
    launch = (run_root / "launcher/launch.sh").read_text()
    assert launch.index("local_endpoint_close.py") < launch.index("train_ddm_cl1_hpac_capacity_mps.py")
    for assignment in (
        "PYTHONHASHSEED=0",
        "TAC_ADMISSION_ENFORCE=1",
        "PYTORCH_ENABLE_MPS_FALLBACK=0",
    ):
        assert assignment in launch
    assert "--resume-from" in launch and str(checkpoint) in launch
    assert "--arm-watchers" in launch and "--done-receipt" in launch
    assert not (run_root / "launcher/run.log").exists()


def test_newest_checkpoint_ambiguity_refuses(tmp_path: Path) -> None:
    parent, _ = _parent_fixture(tmp_path)
    manifest = json.loads((parent / "launcher/launch_manifest.json").read_text())
    save = Path(FWC._option_value(manifest["argv"], "--save"))
    duplicate = save.with_name(save.stem + ".checkpoints") / "other/qat_stage_end_epoch_0480.pt"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"ambiguous")
    with pytest.raises(FWC.ContinuationCompositionError, match="ambiguous newest"):
        FWC.locate_resume_checkpoint(manifest["argv"], 480)


def test_sealed_or_unknown_overrides_refuse(tmp_path: Path) -> None:
    parent, _ = _parent_fixture(tmp_path)
    for override, message in (
        ("--epochs=1000", "sealed continuation field"),
        ("--invented-flag=yes", "no reference-trainer flag"),
    ):
        with pytest.raises(FWC.ContinuationCompositionError, match=message):
            FWC.compose_continuation(
                parent_run_dir=parent,
                port_mode="full-mps-e960",
                run_root=tmp_path / ("child_" + override.split("=", 1)[0].removeprefix("--")),
                overrides=[override],
                done_receipt=f"lh2_override_{override.split('=', 1)[0].removeprefix('--')}",
                closer_deadline_s=3600,
                closer_poll_s=5,
            )


def test_gate_verifier_ignores_historical_attempts_and_reports_healthy_chain(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("CL1TrainingError: set PYTHONHASHSEED=0 before launch\n", encoding="utf-8")
    offset = log.stat().st_size
    with log.open("a", encoding="utf-8") as handle:
        handle.write("[continuation] epoch-extension identity reconciled: accepted drift []\n")
        handle.write(json.dumps({"epoch": 480, "resume": True}) + "\n")
        handle.write(
            json.dumps(
                {
                    "epoch": 482,
                    "phase": "discrete_qat",
                    "estimated_joint_bytes": 130000,
                }
            )
            + "\n"
        )
    result = FWC.verify_gate_chain(
        log_path=log,
        done_receipt_path=tmp_path / "absent.done",
        start_offset=offset,
        parent_epoch=480,
        timeout_s=0,
        poll_s=0.001,
        env_gates={"PYTHONHASHSEED": "0"},
    )
    assert [row["type"] for row in result["events"]] == [
        "RECONCILED",
        "RESUMED",
        "FIRST_EPOCH_ROW",
    ]
    assert result["outcome"] == {"type": "FIRST_EPOCH_ROW", "epoch": 482, "joint_bytes": 130000}


@pytest.mark.parametrize(
    ("message", "gate", "cure"),
    [
        (
            "set TAC_ADMISSION_ENFORCE=1 for a hard governor gate",
            "TAC_ADMISSION_ENFORCE",
            "SOURCE_PARSED_REQUIRED_ENVIRONMENT_ASSIGNMENT",
        ),
        (
            "--resume-from run identity differs; config/input/source drift refused",
            "resume_identity",
            "SEALED_WRAPPER_CONTINUATION_IDENTITY_ADAPTER",
        ),
        (
            "resume lineage entry 0 lacks required custody fields: embedded checkpoint lineage",
            "resume_lineage_custody",
            "TYPED_LINEAGE_UNTOUCHED_WRAPPER_PROVENANCE_RECEIPT",
        ),
    ],
)
def test_gate_verifier_names_refusal_and_known_cure(
    tmp_path: Path, message: str, gate: str, cure: str
) -> None:
    log = tmp_path / "run.log"
    log.write_text(f"_trainer.CL1TrainingError: {message}\n", encoding="utf-8")
    result = FWC.verify_gate_chain(
        log_path=log,
        done_receipt_path=tmp_path / "absent.done",
        start_offset=0,
        parent_epoch=480,
        timeout_s=0,
        poll_s=0.001,
        env_gates={"TAC_ADMISSION_ENFORCE": "1"},
    )
    assert result["outcome"]["type"] == "GATE_REFUSED"
    assert result["outcome"]["gate"] == gate
    assert result["outcome"]["known_cure_class"] == cure


def test_gate_verifier_reports_done_receipt_as_dead_not_timeout(tmp_path: Path) -> None:
    done = tmp_path / "run.done"
    _write_json(done, {"schema": FWC.DONE_SCHEMA, "rc": 7})
    result = FWC.verify_gate_chain(
        log_path=tmp_path / "missing.log",
        done_receipt_path=done,
        start_offset=0,
        parent_epoch=480,
        timeout_s=0,
        poll_s=0.001,
        env_gates={},
    )
    assert result["outcome"] == {"type": "DEAD", "rc": 7}
    assert result["bounded_timeout_is_process_failure"] is False


def test_gate_verifier_bounded_deadline_is_pending_not_dead(tmp_path: Path) -> None:
    result = FWC.verify_gate_chain(
        log_path=tmp_path / "missing.log",
        done_receipt_path=tmp_path / "missing.done",
        start_offset=0,
        parent_epoch=480,
        timeout_s=0,
        poll_s=0.001,
        env_gates={},
    )
    assert result["outcome"] == {"type": "PENDING_BOUNDED"}
    assert result["bounded_timeout_is_process_failure"] is False


def test_parent_endpoint_uses_highest_epoch_not_last_append(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        json.dumps({"epoch": 480, "estimated_joint_bytes": 100})
        + "\n"
        + json.dumps({"epoch": 12, "estimated_joint_bytes": 999})
        + "\n",
        encoding="utf-8",
    )
    assert FWC._parent_endpoint(log)["epoch"] == 480


def test_parent_endpoint_skips_malformed_json_rows(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        json.dumps({"epoch": "bad", "estimated_joint_bytes": 999})
        + "\n"
        + json.dumps({"epoch": 480, "estimated_joint_bytes": 100})
        + "\n",
        encoding="utf-8",
    )
    assert FWC._parent_endpoint(log)["joint_bytes"] == 100


def test_hot_state_staleness_is_warn_only(tmp_path: Path) -> None:
    hot = tmp_path / "hot.md"
    pointer = tmp_path / "pointer.json"
    hot.write_text("## POINTER_LINE\nfrontier **S 0.2 @ 1 B**\n\n## NEXT\n", encoding="utf-8")
    _write_json(pointer, {"effective_frontier": {"score": 0.19}})
    result = FWC.check_hot_state_staleness(hot, pointer)
    assert result["status"] == "WARN_STALE"
    assert result["mode"] == "WARN_ONLY"
    _write_json(pointer, {"effective_frontier": {"score": 0.2}})
    assert FWC.check_hot_state_staleness(hot, pointer)["status"] == "CURRENT"


def test_closer_done_receipt_stays_distinct_at_max_training_name(tmp_path: Path) -> None:
    parent, _ = _parent_fixture(tmp_path)
    name = "a" * 128
    result = FWC.compose_continuation(
        parent_run_dir=parent,
        port_mode="full-mps-e960",
        run_root=tmp_path / "max_receipt_child",
        overrides=[],
        done_receipt=name,
        closer_deadline_s=3600,
        closer_poll_s=5,
    )
    assert Path(result["training_done_receipt_path"]).name == name + ".done"
    assert Path(result["closer_done_receipt_path"]).name != name + ".done"
    assert Path(result["closer_done_receipt_path"]).stem.endswith(".endpoint-close")
