# SPDX-License-Identifier: MIT
"""Tests for src/tac/sweep_runner.py — the canonical Bayesian sweep harness.

Coverage:
  * search_space schema validation (all 5 distribution kinds + bad inputs)
  * non-negotiable enforcement (eval_roundtrip, device=cuda)
  * search_space hash determinism + sensitivity
  * placeholder substitution (template MUST contain every __PARAM_X__ token)
  * Optuna integration via local_smoke (no GPU)
  * sidecar JSON parsing (default parser + custom parser)
  * RESULT_JSON log scraping (the auth_eval contract)
  * trial_history.jsonl provenance (no signal loss rule)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.sweep_runner import (
    BayesianSweep,
    OptunaTrialDispatcher,
    _enforce_non_negotiables,
    _hash_search_space,
    _placeholder_token,
    _validate_search_space,
)


# ---------- search-space validation ---------------------------------------


def test_validate_search_space_accepts_all_kinds():
    space = {
        "lr":        ("loguniform", 1e-6, 1e-3),
        "dropout":   ("uniform", 0.0, 0.5),
        "batch":     ("int", 4, 32),
        "schedule":  ("categorical", ["cosine", "linear"]),
        "device":    ("fixed", "cuda"),
    }
    # Must not raise.
    _validate_search_space(space)


def test_validate_search_space_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _validate_search_space({})


def test_validate_search_space_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        _validate_search_space({"x": ("normal", 0.0, 1.0)})


def test_validate_search_space_rejects_loguniform_zero_low():
    with pytest.raises(ValueError, match="loguniform low must be > 0"):
        _validate_search_space({"x": ("loguniform", 0.0, 1.0)})


def test_validate_search_space_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="must be < high"):
        _validate_search_space({"x": ("uniform", 1.0, 0.0)})


def test_validate_search_space_rejects_int_bounds_swapped():
    with pytest.raises(ValueError, match="must be <= high"):
        _validate_search_space({"x": ("int", 10, 5)})


def test_validate_search_space_rejects_empty_categorical():
    with pytest.raises(ValueError, match="categorical choices empty"):
        _validate_search_space({"x": ("categorical", [])})


# ---------- non-negotiable enforcement ------------------------------------


def test_enforce_eval_roundtrip_fixed_true_ok():
    _enforce_non_negotiables({"eval_roundtrip": ("fixed", True)})


def test_enforce_eval_roundtrip_fixed_false_rejected():
    with pytest.raises(ValueError, match="eval_roundtrip must be True"):
        _enforce_non_negotiables({"eval_roundtrip": ("fixed", False)})


def test_enforce_eval_roundtrip_categorical_with_false_rejected():
    with pytest.raises(ValueError, match="False-equivalent"):
        _enforce_non_negotiables({"eval_roundtrip": ("categorical", [True, False])})


def test_enforce_eval_roundtrip_loguniform_rejected():
    # eval_roundtrip is bool — only fixed/categorical make sense
    with pytest.raises(ValueError, match="eval_roundtrip"):
        _enforce_non_negotiables({"eval_roundtrip": ("loguniform", 1e-3, 1.0)})


def test_enforce_device_cuda_only():
    _enforce_non_negotiables({"device": ("fixed", "cuda")})


def test_enforce_device_mps_rejected():
    with pytest.raises(ValueError, match="device must be 'cuda'"):
        _enforce_non_negotiables({"device": ("fixed", "mps")})


def test_enforce_device_cpu_rejected():
    with pytest.raises(ValueError, match="device must be 'cuda'"):
        _enforce_non_negotiables({"device": ("fixed", "cpu")})


def test_enforce_device_categorical_with_mps_rejected():
    with pytest.raises(ValueError, match="non-cuda value"):
        _enforce_non_negotiables({"device": ("categorical", ["cuda", "mps"])})


# ---------- search-space hash --------------------------------------------


def test_hash_is_deterministic():
    space = {"a": ("uniform", 0.0, 1.0), "b": ("int", 1, 10)}
    h1 = _hash_search_space(space)
    h2 = _hash_search_space(space)
    assert h1 == h2
    assert len(h1) == 16  # SHA256[:16]


def test_hash_is_order_invariant():
    a_first = {"a": ("uniform", 0.0, 1.0), "b": ("int", 1, 10)}
    b_first = {"b": ("int", 1, 10), "a": ("uniform", 0.0, 1.0)}
    assert _hash_search_space(a_first) == _hash_search_space(b_first)


def test_hash_changes_when_bounds_change():
    s1 = {"a": ("uniform", 0.0, 1.0)}
    s2 = {"a": ("uniform", 0.0, 2.0)}
    assert _hash_search_space(s1) != _hash_search_space(s2)


# ---------- placeholder token --------------------------------------------


def test_placeholder_token_uppercase():
    assert _placeholder_token("tto_steps") == "__PARAM_TTO_STEPS__"
    assert _placeholder_token("lr") == "__PARAM_LR__"


# ---------- BayesianSweep construction + template ------------------------


def _write_template(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "tmpl.sh"
    p.write_text(content)
    return p


def test_bayesian_sweep_construction_validates(tmp_path: Path):
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__\nbatch=__PARAM_BATCH__\n")
    sweep = BayesianSweep(
        name="t",
        script_template=tmpl,
        search_space={
            "lr": ("loguniform", 1e-5, 1e-2),
            "batch": ("int", 4, 16),
        },
        n_trials=3,
        output_dir=tmp_path / "out",
    )
    assert sweep.search_space_hash
    assert len(sweep.search_space_hash) == 16


def test_bayesian_sweep_rejects_bad_n_trials(tmp_path: Path):
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__")
    with pytest.raises(ValueError, match="n_trials"):
        BayesianSweep(
            name="t",
            script_template=tmpl,
            search_space={"lr": ("loguniform", 1e-5, 1e-2)},
            n_trials=0,
            output_dir=tmp_path / "out",
        )


def test_bayesian_sweep_rejects_bad_direction(tmp_path: Path):
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__")
    with pytest.raises(ValueError, match="direction"):
        BayesianSweep(
            name="t",
            script_template=tmpl,
            search_space={"lr": ("loguniform", 1e-5, 1e-2)},
            n_trials=3,
            direction="bigger_better",
            output_dir=tmp_path / "out",
        )


def test_template_missing_placeholder_caught(tmp_path: Path):
    # Template lacks __PARAM_BATCH__ — should be caught when we read template.
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__\n")  # no batch token
    sweep = BayesianSweep(
        name="t",
        script_template=tmpl,
        search_space={
            "lr": ("loguniform", 1e-5, 1e-2),
            "batch": ("int", 4, 16),
        },
        n_trials=1,
        output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="missing placeholder"):
        sweep._read_template()


def test_template_substitution_renders_concrete_values(tmp_path: Path):
    tmpl = _write_template(tmp_path, "LR=__PARAM_LR__ BATCH=__PARAM_BATCH__ ROUND=__PARAM_DO_ROUND__\n")
    sweep = BayesianSweep(
        name="t",
        script_template=tmpl,
        search_space={
            "lr": ("loguniform", 1e-5, 1e-2),
            "batch": ("int", 4, 16),
            "do_round": ("fixed", True),
        },
        n_trials=1,
        output_dir=tmp_path / "out",
    )
    txt = sweep._substitute(tmpl.read_text(), {"lr": 0.001, "batch": 8, "do_round": True}, trial_number=42)
    assert "LR=0.001" in txt
    assert "BATCH=8" in txt
    assert "ROUND=1" in txt  # bool True → "1"


def test_substitute_includes_trial_provenance_tags(tmp_path: Path):
    tmpl = _write_template(
        tmp_path,
        "lr=__PARAM_LR__ name=__SWEEP_NAME__ trial=__SWEEP_TRIAL_NUMBER__ hash=__SWEEP_SEARCH_SPACE_HASH__\n",
    )
    sweep = BayesianSweep(
        name="my_sweep",
        script_template=tmpl,
        search_space={"lr": ("loguniform", 1e-5, 1e-2)},
        n_trials=1,
        output_dir=tmp_path / "out",
    )
    txt = sweep._substitute(tmpl.read_text(), {"lr": 0.001}, trial_number=7)
    assert "name=my_sweep" in txt
    assert "trial=7" in txt
    assert f"hash={sweep.search_space_hash}" in txt


# ---------- Optuna local-smoke --------------------------------------------


def test_local_smoke_runs_and_writes_history(tmp_path: Path):
    pytest.importorskip("optuna")
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__ batch=__PARAM_BATCH__\n")
    sweep = BayesianSweep(
        name="smoke_t",
        script_template=tmpl,
        search_space={
            "lr": ("loguniform", 1e-5, 1e-2),
            "batch": ("int", 4, 16),
        },
        n_trials=4,
        output_dir=tmp_path / "out",
    )
    dispatcher = OptunaTrialDispatcher(sweep)
    summary = dispatcher.local_smoke()
    assert summary["sweep_name"] == "smoke_t"
    assert summary["n_trials"] == 4
    assert summary["best_value"] is not None
    assert isinstance(summary["best_params"], dict)
    # Trial history file written.
    history = (tmp_path / "out" / "trial_history.jsonl").read_text().splitlines()
    assert len(history) == 4
    parsed = [json.loads(line) for line in history]
    for record in parsed:
        assert record["sweep_name"] == "smoke_t"
        assert "lr" in record["params"]
        assert "batch" in record["params"]


def test_local_smoke_emits_best_so_far_curve(tmp_path: Path):
    pytest.importorskip("optuna")
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__\n")
    sweep = BayesianSweep(
        name="conv_t",
        script_template=tmpl,
        search_space={"lr": ("loguniform", 1e-5, 1e-2)},
        n_trials=5,
        output_dir=tmp_path / "out",
    )
    dispatcher = OptunaTrialDispatcher(sweep)
    summary = dispatcher.local_smoke()
    bsf = summary["best_so_far"]
    assert len(bsf) == 5
    # Monotonically non-increasing for minimize direction.
    for i in range(1, len(bsf)):
        assert bsf[i] <= bsf[i - 1] + 1e-12


# ---------- result parsing ------------------------------------------------


def test_parse_remote_result_from_sidecar_json(tmp_path: Path):
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__\n")
    sweep = BayesianSweep(
        name="t",
        script_template=tmpl,
        search_space={"lr": ("loguniform", 1e-5, 1e-2)},
        n_trials=1,
        output_dir=tmp_path / "out",
    )
    script_path = tmp_path / "out" / "trial.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# ...")
    sidecar = script_path.with_suffix(".result.json")
    sidecar.write_text(json.dumps({"final_score": 0.987, "rate": 0.4}))
    assert sweep.parse_remote_result(script_path) == pytest.approx(0.987)


def test_parse_remote_result_from_log_resultjson(tmp_path: Path):
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__\n")
    sweep = BayesianSweep(
        name="t",
        script_template=tmpl,
        search_space={"lr": ("loguniform", 1e-5, 1e-2)},
        n_trials=1,
        output_dir=tmp_path / "out",
    )
    script_path = tmp_path / "out" / "trial.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# ...")
    log = script_path.parent / "auth_eval.log"
    log.write_text(
        "some preamble\n"
        'RESULT_JSON: {"final_score": 1.234, "avg_posenet_dist": 0.05}\n'
        "trailer\n"
    )
    assert sweep.parse_remote_result(script_path) == pytest.approx(1.234)


def test_parse_remote_result_missing_raises(tmp_path: Path):
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__\n")
    sweep = BayesianSweep(
        name="t",
        script_template=tmpl,
        search_space={"lr": ("loguniform", 1e-5, 1e-2)},
        n_trials=1,
        output_dir=tmp_path / "out",
    )
    script_path = tmp_path / "out" / "trial.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# ...")
    with pytest.raises(FileNotFoundError, match="no remote result"):
        sweep.parse_remote_result(script_path)


def test_parse_remote_result_custom_parser(tmp_path: Path):
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__\n")
    sweep = BayesianSweep(
        name="t",
        script_template=tmpl,
        search_space={"lr": ("loguniform", 1e-5, 1e-2)},
        n_trials=1,
        output_dir=tmp_path / "out",
        result_parser=lambda p: 42.0,
    )
    script_path = tmp_path / "out" / "trial.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# ...")
    assert sweep.parse_remote_result(script_path) == 42.0


def test_parse_remote_result_json_missing_score_key_raises(tmp_path: Path):
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__\n")
    sweep = BayesianSweep(
        name="t",
        script_template=tmpl,
        search_space={"lr": ("loguniform", 1e-5, 1e-2)},
        n_trials=1,
        output_dir=tmp_path / "out",
        objective="auth_score",
    )
    script_path = tmp_path / "out" / "trial.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# ...")
    sidecar = script_path.with_suffix(".result.json")
    sidecar.write_text(json.dumps({"unrelated_key": 9}))
    with pytest.raises(KeyError, match="lacks a known score key"):
        sweep.parse_remote_result(script_path)


# ---------- dispatch_remote writes a script -------------------------------


def test_dispatch_remote_writes_unique_script(tmp_path: Path):
    pytest.importorskip("optuna")
    tmpl = _write_template(tmp_path, "lr=__PARAM_LR__ batch=__PARAM_BATCH__\n")
    sweep = BayesianSweep(
        name="dispatch_t",
        script_template=tmpl,
        search_space={
            "lr": ("loguniform", 1e-5, 1e-2),
            "batch": ("int", 4, 16),
        },
        n_trials=2,
        output_dir=tmp_path / "out",
    )

    import optuna
    study = optuna.create_study(direction="minimize")

    def _objective(trial):
        path = sweep.dispatch_remote(trial)
        assert path.exists()
        text = path.read_text()
        # Both placeholders gone, concrete values present.
        assert "__PARAM_LR__" not in text
        assert "__PARAM_BATCH__" not in text
        return 0.5

    study.optimize(_objective, n_trials=2)
    # Expect 2 unique scripts.
    scripts = sorted((tmp_path / "out").glob("dispatch_t_trial_*.sh"))
    assert len(scripts) == 2
    assert scripts[0].name != scripts[1].name
