from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from experiments import ddm_qbt1_qbflow_trainer as qbt
from tac import confound_gates as cg
from tac.preflight import PreflightError
from tac.training import EMA
from tac.witness_dsl.curriculum_dsl import EmaDecayCalibrated


def _module(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "experiments" / "probe.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return path


def _original_shape(warmup: str = "True", suffix: str = "") -> str:
    return (
        "from tac.training import EMA\n"
        "from tac.witness_dsl.curriculum_dsl import EmaDecayCalibrated\n"
        "def compile_config():\n"
        "    return EmaDecayCalibrated(5000, target_seed_fraction=0.01)\n"
        "def run(model, config):\n"
        f"    return EMA(model, decay=config['ema']['value'], warmup={warmup}){suffix}\n"
    )


def test_typed_factory_defaults_to_constant_decay() -> None:
    lever = EmaDecayCalibrated(5_000, target_seed_fraction=0.01)
    assert lever.policy_contracts["ema_execution"] == {
        "mode": "constant_decay",
        "warmup": False,
        "ablation_declared": False,
        "sealed_law": "constant_decay",
        "source": "tac.witness_dsl.curriculum_dsl.EmaDecayCalibrated",
    }


def test_typed_factory_requires_explicit_warmup_ablation() -> None:
    lever = EmaDecayCalibrated(
        5_000,
        target_seed_fraction=0.01,
        execution_mode="warmup_ablation",
    )
    policy = lever.policy_contracts["ema_execution"]
    assert policy["mode"] == "warmup_ablation"
    assert policy["warmup"] is True
    assert policy["ablation_declared"] is True


def test_typed_factory_rejects_unknown_execution_mode() -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        EmaDecayCalibrated(
            5_000,
            target_seed_fraction=0.01,
            execution_mode="silent_auto",
        )


def test_sealed_constant_decay_coefficient_matches_prior_law() -> None:
    sealed = qbt.resolve_ema_law(5_000)
    assert sealed["execution"]["warmup"] is False
    assert sealed["ema_law_sealed"]["name"] == "constant_decay"
    assert math.isclose(
        sealed["ema_law_sealed"]["terminal_seed_coefficient"],
        0.010000000000000278,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    )


def test_warmup_coefficient_reproduces_wc2_alarm() -> None:
    decay = qbt.resolve_ema_law(5_000)["value"]
    coefficient = qbt.ema_terminal_seed_coefficient(decay, 5_000, warmup=True)
    assert math.isclose(coefficient, 1.8380018548760466e-27, rel_tol=1.0e-13)


def test_construct_ema_consumes_compiled_constant_policy() -> None:
    model = torch.nn.Linear(2, 1)
    ema, provenance = qbt.construct_ema_from_config(
        model,
        qbt.resolve_ema_law(5_000),
        total_updates=5_000,
    )
    assert ema.warmup is False
    assert provenance["matched"] is True
    assert provenance["ema_law_executed"]["name"] == "constant_decay"


def test_construct_ema_consumes_explicit_warmup_ablation_policy() -> None:
    model = torch.nn.Linear(2, 1)
    sealed = qbt.resolve_ema_law(20, execution_mode="warmup_ablation")
    ema, provenance = qbt.construct_ema_from_config(model, sealed, total_updates=20)
    assert ema.warmup is True
    assert provenance["matched"] is True
    assert provenance["ema_law_sealed"]["name"] == "warmup"


def test_runtime_alarm_is_typed_and_halts(capsys: pytest.CaptureFixture[str]) -> None:
    model = torch.nn.Linear(2, 1)
    sealed = qbt.resolve_ema_law(5_000)
    wrong = EMA(model, decay=sealed["value"], warmup=True)
    with pytest.raises(qbt.QBT1Error, match=r"confound_alarm\(ema_law_mismatch\)"):
        qbt.verify_ema_executable_law(wrong, sealed, total_updates=5_000)
    emitted = capsys.readouterr().out
    assert '"stage": "confound_alarm"' in emitted
    assert '"alarm": "ema_law_mismatch"' in emitted
    assert "terminal_seed_coefficient" in emitted


def test_runtime_decay_mismatch_emits_same_typed_alarm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = torch.nn.Linear(2, 1)
    sealed = qbt.resolve_ema_law(5_000)
    wrong = EMA(model, decay=float(sealed["value"]) - 1.0e-5, warmup=False)
    with pytest.raises(qbt.QBT1Error, match=r"confound_alarm\(ema_law_mismatch\)"):
        qbt.verify_ema_executable_law(wrong, sealed, total_updates=5_000)
    emitted = capsys.readouterr().out
    assert '"decay_matched": false' in emitted
    assert '"alarm": "ema_law_mismatch"' in emitted


def test_runtime_recomputes_and_refuses_tampered_sealed_coefficient(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = torch.nn.Linear(2, 1)
    sealed = qbt.resolve_ema_law(5_000)
    sealed["ema_law_sealed"]["terminal_seed_coefficient"] = 0.5
    ema = EMA(model, decay=sealed["value"], warmup=False)
    with pytest.raises(qbt.QBT1Error, match=r"confound_alarm\(ema_law_mismatch\)"):
        qbt.verify_ema_executable_law(ema, sealed, total_updates=5_000)
    emitted = capsys.readouterr().out
    assert '"sealed_record_matched": false' in emitted
    assert '"alarm": "ema_law_mismatch"' in emitted


def test_runtime_refuses_untyped_execution_policy() -> None:
    model = torch.nn.Linear(2, 1)
    with pytest.raises(qbt.QBT1Error, match="typed Lever"):
        qbt.construct_ema_from_config(
            model,
            {"value": 0.9},
            total_updates=10,
        )


def test_gate_catches_original_qbr1_literal_true(tmp_path: Path) -> None:
    _module(tmp_path, _original_shape())
    violations = cg.check_ema_executable_law_matches_sealed_law(
        repo_root=tmp_path, verbose=False
    )
    assert len(violations) == 1
    assert "warmup=True" in violations[0]
    assert "probe.py" in violations[0]


def test_gate_catches_literal_false_too(tmp_path: Path) -> None:
    _module(tmp_path, _original_shape(warmup="False"))
    violations = cg.check_ema_executable_law_matches_sealed_law(
        repo_root=tmp_path, verbose=False
    )
    assert len(violations) == 1
    assert "warmup=False" in violations[0]


def test_gate_accepts_config_threaded_warmup(tmp_path: Path) -> None:
    _module(tmp_path, _original_shape(warmup="config['ema']['execution']['warmup']"))
    assert cg.check_ema_executable_law_matches_sealed_law(
        repo_root=tmp_path, verbose=False
    ) == []


def test_gate_accepts_substantive_same_line_ablation_waiver(tmp_path: Path) -> None:
    _module(
        tmp_path,
        _original_shape(
            suffix="  # EMA_WARMUP_ABLATION_OK:sealed warmup control cell A7"
        ),
    )
    assert cg.check_ema_executable_law_matches_sealed_law(
        repo_root=tmp_path, verbose=False
    ) == []


@pytest.mark.parametrize("rationale", ["<rationale>", "TODO", "TBD", "FIXME", "placeholder"])
def test_gate_rejects_placeholder_ablation_waiver(
    tmp_path: Path,
    rationale: str,
) -> None:
    _module(
        tmp_path,
        _original_shape(suffix=f"  # EMA_WARMUP_ABLATION_OK:{rationale}"),
    )
    assert len(
        cg.check_ema_executable_law_matches_sealed_law(
            repo_root=tmp_path, verbose=False
        )
    ) == 1


def test_gate_ignores_literal_ema_without_lawref_resolution(tmp_path: Path) -> None:
    _module(
        tmp_path,
        "from tac.training import EMA\n"
        "def run(model):\n"
        "    return EMA(model, decay=0.997, warmup=True)\n",
    )
    assert cg.check_ema_executable_law_matches_sealed_law(
        repo_root=tmp_path, verbose=False
    ) == []


def test_gate_detects_explicit_equation_id_resolution(tmp_path: Path) -> None:
    _module(
        tmp_path,
        "from tac.training import EMA\n"
        "EMA_LAW = 'ema_decay_run_geometry_v1'\n"
        "def run(model):\n"
        "    return EMA(model, decay=0.9, warmup=True)\n",
    )
    assert len(
        cg.check_ema_executable_law_matches_sealed_law(
            repo_root=tmp_path, verbose=False
        )
    ) == 1


def test_gate_strict_mode_raises(tmp_path: Path) -> None:
    _module(tmp_path, _original_shape())
    with pytest.raises(PreflightError, match="check_ema_executable_law_matches_sealed_law"):
        cg.check_ema_executable_law_matches_sealed_law(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_gate_is_registered_with_positive_control() -> None:
    assert cg.check_ema_executable_law_matches_sealed_law in cg.CONFOUND_GATES
    controls = [
        control
        for control in cg.POSITIVE_CONTROLS
        if control.gate == "check_ema_executable_law_matches_sealed_law"
    ]
    assert len(controls) == 1
    assert controls[0].must_mention == "planted_qbr1.py"


def test_live_repo_has_no_literal_ema_law_mismatch_sites() -> None:
    assert cg.check_ema_executable_law_matches_sealed_law(verbose=False) == []


def test_qbr1_cured_store_does_not_overlap_invalidated_live_run() -> None:
    from experiments import ddm_qbr1_born_fairform_burn_prep as qbr

    invalidated = Path("/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/runs")
    assert invalidated != qbr.RUN_ROOT
    assert str(qbr.RUN_ROOT).startswith(
        "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/"
    )
