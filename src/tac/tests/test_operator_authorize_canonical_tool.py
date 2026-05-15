# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

import tools.operator_authorize as op


@pytest.fixture(autouse=True)
def _isolate_local_pre_deploy(monkeypatch) -> None:
    """Operator routing tests mock dispatch gates; local harness has own coverage."""
    monkeypatch.setattr(op, "_run_local_pre_deploy_check", lambda *_: None)


def _band() -> op.CostBandPrediction:
    return op.CostBandPrediction(
        p10_cost_usd=0.0,
        p50_cost_usd=0.0,
        p90_cost_usd=0.0,
        n_anchors=0,
        confidence_tag="test",
        source="test",
    )


def test_cost_band_request_full_run_metadata_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_GPU", raising=False)
    recipe = op._load_recipe("substrate_sane_hnerv_modal_a100_dispatch")

    request = op._resolve_cost_band_request(recipe)

    assert request.context_label == "recipe_full"
    assert request.platform_key == "modal"
    assert request.gpu_key == "A100"
    assert request.epochs == 2000
    assert request.fallback_p50_usd == 8.00
    assert request.full_run_gpu_key == "A100"
    assert request.full_run_epochs == 2000
    assert request.full_run_fallback_p50_usd == 8.00


def test_cost_band_request_smoke_override_scales_cold_start_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODAL_GPU", "T4")
    recipe = op._load_recipe("substrate_sane_hnerv_modal_a100_dispatch")

    request = op._resolve_cost_band_request(
        recipe,
        cost_band_epochs_override=100,
    )

    assert request.context_label == "smoke_override"
    assert request.is_smoke_scaled is True
    assert request.platform_key == "modal"
    assert request.gpu_key == "T4"
    assert request.epochs == 100
    assert request.fallback_p50_usd == pytest.approx(0.40)
    assert request.full_run_gpu_key == "A100"
    assert request.full_run_epochs == 2000
    assert request.full_run_fallback_p50_usd == 8.00


def test_operator_authorize_smoke_override_banner_and_claim_use_smoke_cost(
    monkeypatch,
    capsys,
) -> None:
    claim_calls: list[dict[str, object]] = []
    predict_calls: list[dict[str, object]] = []

    def fake_predict(**kwargs: object) -> op.CostBandPrediction:
        predict_calls.append(kwargs)
        p50 = float(kwargs["hand_calibrated_fallback_p50_usd"])
        return op.CostBandPrediction(
            p10_cost_usd=p50 * 0.5,
            p50_cost_usd=p50,
            p90_cost_usd=p50 * 2.0,
            n_anchors=0,
            confidence_tag="hand_calibrated_fallback",
            source="hand_calibrated_fallback",
        )

    monkeypatch.setenv("MODAL_GPU", "T4")
    monkeypatch.setattr(op, "_predict_cost_band", fake_predict)
    monkeypatch.setattr(op, "_validate_declared_local_paths", lambda *_: None)
    monkeypatch.setattr(op, "_validate_required_input_files", lambda *_: None)
    monkeypatch.setattr(op, "_native_dispatch_preflight", lambda *_: None)
    monkeypatch.setattr(op, "_claim_lane", lambda **kwargs: claim_calls.append(kwargs))
    monkeypatch.setattr(op, "_run_dispatch", lambda *_args, **_kwargs: 0)

    rc = op.main(
        [
            "--recipe",
            "substrate_sane_hnerv_modal_a100_dispatch",
            "--yes",
            "--label-suffix",
            "__smoke__100ep",
            "--timeout-hours-override",
            "1",
            "--cost-band-epochs-override",
            "100",
        ]
    )

    assert rc == 0
    assert len(predict_calls) == 1
    assert predict_calls[0]["platform_key"] == "modal"
    assert predict_calls[0]["gpu_key"] == "T4"
    assert predict_calls[0]["epochs"] == 100
    assert predict_calls[0]["all_flags_on"] is True
    assert predict_calls[0]["hand_calibrated_fallback_p50_usd"] == pytest.approx(0.40)
    out = capsys.readouterr().out
    assert "cost band p10/p50/p90:   $0.20/$0.40/$0.80" in out
    assert "cost context:            smoke override modal/T4 x 100 epochs" in out
    assert "full-run reference modal/A100 x 2000 epochs, fallback p50=$8.00" in out
    assert "cold-start fallback uses the smoke bucket" in out
    assert len(claim_calls) == 1
    notes = str(claim_calls[0]["notes"])
    assert "expected p50 cost $0.40" in notes
    assert "smoke cost context modal/T4 x 100ep" in notes
    assert "full-run reference modal/A100 x 2000ep fallback p50 $8.00" in notes


def test_operator_authorize_lists_recipes(capsys) -> None:
    rc = op.main(["--list"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "phase1_t1_balle_cheap_config_dispatch" in out
    assert "kaggle_t1_balle_sweep" in out


def test_operator_authorize_dry_run_does_not_claim_or_dispatch(monkeypatch, capsys) -> None:
    monkeypatch.setattr(op, "_predict_cost_band", lambda **_: _band())
    monkeypatch.setattr(
        op,
        "_claim_lane",
        lambda **_: (_ for _ in ()).throw(AssertionError("claim should not fire")),
    )
    monkeypatch.setattr(
        op,
        "_run_dispatch",
        lambda *_: (_ for _ in ()).throw(AssertionError("dispatch should not fire")),
    )

    rc = op.main(["--recipe", "phase1_t1_balle_cheap_config_dispatch", "--dry-run"])

    assert rc == 0
    assert "--dry-run; no confirmation prompt, no dispatch" in capsys.readouterr().out


def test_operator_authorize_refuses_direct_full_for_smoke_only_recipe(
    monkeypatch,
) -> None:
    monkeypatch.setattr(op, "_predict_cost_band", lambda **_: _band())
    monkeypatch.setattr(
        op,
        "_claim_lane",
        lambda **_: (_ for _ in ()).throw(AssertionError("claim should not fire")),
    )
    monkeypatch.setattr(
        op,
        "_run_dispatch",
        lambda *_: (_ for _ in ()).throw(AssertionError("dispatch should not fire")),
    )

    with pytest.raises(SystemExit, match="smoke_only=true"):
        op.main(
            [
                "--recipe",
                "substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch",
                "--yes",
            ]
        )


def test_operator_authorize_allows_smoke_wrapper_epoch_override_for_smoke_only_recipe(
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(op, "_predict_cost_band", lambda **_: _band())
    monkeypatch.setattr(op, "_validate_declared_local_paths", lambda *_: None)
    monkeypatch.setattr(op, "_validate_required_input_files", lambda *_: None)
    monkeypatch.setattr(op, "_native_dispatch_preflight", lambda *_: events.append("preflight"))
    monkeypatch.setattr(op, "_claim_lane", lambda **_: events.append("claim"))
    monkeypatch.setattr(op, "_run_dispatch", lambda *_args, **_kwargs: events.append("dispatch") or 0)

    rc = op.main(
        [
            "--recipe",
            "substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch",
            "--yes",
            "--cost-band-epochs-override",
            "100",
        ]
    )

    assert rc == 0
    assert events == ["preflight", "claim", "dispatch"]


def test_z3_operator_recipe_explicitly_selects_v2_latent_replacement() -> None:
    recipe = op._load_recipe("substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch")

    env_overrides = op._build_env_overrides(recipe, "unit_job_z3_v2")

    assert "Z3_BALLE_ENABLE_V2_LATENT_REPLACEMENT=1" in env_overrides.split(",")
    assert recipe.raw["active_dispatch_contract"] == "z3_v2_latent_replacement_a1_base"
    assert recipe.raw["dispatch_contracts"][0]["archive_role"] == (
        "z3hv2_section_replaces_a1_latent_blob"
    )


def test_z3_remote_driver_defaults_to_v2_latent_replacement() -> None:
    text = (
        op.REPO_ROOT
        / "scripts"
        / "remote_lane_substrate_z3_balle_hyperprior_bolton.sh"
    ).read_text()

    default_ladder = (
        'Z3_BALLE_ENABLE_V2_LATENT_REPLACEMENT="'
        "${Z3_BALLE_ENABLE_V2_LATENT_REPLACEMENT:-1}\""
    )
    assert default_ladder in text
    assert "V2_LATENT_REPLACEMENT_ARGS+=(--enable-v2-latent-replacement)" in text
    assert (
        '${V2_LATENT_REPLACEMENT_ARGS[@]+"${V2_LATENT_REPLACEMENT_ARGS[@]}"}'
        in text
    )
    assert "Z3_BALLE_ENABLE_V2_LATENT_REPLACEMENT:-0" not in text


def test_z4_z5_recipes_depend_on_current_z3_recover_lane() -> None:
    stale = "lane_z3_balle_hyperprior_bolton_campaign_20260514"
    current = "lane_z3_balle_hyperprior_bolton_recover_20260514"

    for recipe_name in (
        "substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.yaml",
        "substrate_z5_predictive_coding_world_model_modal_t4_dispatch.yaml",
    ):
        text = (op.RECIPES_DIR / recipe_name).read_text()
        assert stale not in text
        assert current in text


def test_operator_authorize_noop_recipe_skips_phantom_lane_claim(monkeypatch, capsys) -> None:
    claim_calls: list[dict[str, object]] = []
    dispatch_calls: list[tuple[op.Recipe, str]] = []
    monkeypatch.setattr(op, "_predict_cost_band", lambda **_: _band())
    monkeypatch.setattr(op, "_confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(op, "_claim_lane", lambda **kwargs: claim_calls.append(kwargs))
    monkeypatch.setattr(
        op,
        "_run_dispatch",
        lambda recipe, instance_job_id: dispatch_calls.append((recipe, instance_job_id)) or 0,
    )

    rc = op.main(["--recipe", "crates_io_publish"])

    assert rc == 0
    assert claim_calls == []
    assert len(dispatch_calls) == 1
    out = capsys.readouterr().out
    assert "skipping lane claim so no phantom active dispatch row is created" in out


def test_operator_authorize_modal_recipe_claims_before_dispatch(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(op, "_predict_cost_band", lambda **_: _band())
    monkeypatch.setattr(op, "_validate_required_input_files", lambda *_: None)
    monkeypatch.setattr(op, "_confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(op, "_native_dispatch_preflight", lambda *_: events.append("preflight"))
    monkeypatch.setattr(op, "_claim_lane", lambda **_: events.append("claim"))
    monkeypatch.setattr(op, "_run_dispatch", lambda *_: events.append("dispatch") or 0)

    rc = op.main(["--recipe", "phase1_t1_balle_cheap_config_dispatch"])

    assert rc == 0
    assert events == ["preflight", "claim", "dispatch"]


def test_operator_authorize_modal_dispatch_threads_recipe_lane_id(monkeypatch) -> None:
    calls: list[list[str]] = []
    recipe = op.Recipe(
        name="unit_modal",
        path=op.RECIPES_DIR / "unit_modal.yaml",
        raw={
            "lane_id": "lane_substrate_siren_20260512",
            "platform": "modal",
            "gpu": "A100",
            "timeout_hours": 1.0,
            "remote_driver": "scripts/remote_lane_substrate_siren.sh",
            "modal": {
                "lane_script": "scripts/remote_lane_substrate_siren.sh",
            },
        },
    )
    monkeypatch.setattr(
        op,
        "_run_modal_dispatch_process",
        lambda cmd: calls.append([str(part) for part in cmd])
        or op.subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    rc = op._dispatch_modal(recipe, "unit_job_001", "DISPATCH_INSTANCE_JOB_ID=unit_job_001")

    assert rc == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert "--lane-id" in cmd
    assert cmd[cmd.index("--lane-id") + 1] == "lane_substrate_siren_20260512"
    assert "--label" in cmd
    assert cmd[cmd.index("--label") + 1] == "unit_job_001"


def test_operator_authorize_modal_dispatch_retries_transient_mount_upload_race(
    monkeypatch,
) -> None:
    recipe = op.Recipe(
        name="unit_modal",
        path=op.RECIPES_DIR / "unit_modal.yaml",
        raw={
            "lane_id": "lane_d4_wyner_ziv_frame_0_substrate_20260514",
            "platform": "modal",
            "gpu": "T4",
            "timeout_hours": 1.0,
            "remote_driver": "scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh",
            "modal": {
                "lane_script": "scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh",
            },
        },
    )
    attempts: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(cmd: list[str]):
        attempts.append([str(part) for part in cmd])
        if len(attempts) == 1:
            return op.subprocess.CompletedProcess(
                cmd,
                1,
                "",
                "MountUploadRaceError: Modal mount set fingerprint is unstable\n",
            )
        return op.subprocess.CompletedProcess(
            cmd,
            0,
            "DISPATCHED via .spawn() - call_id=fc-unit\n",
            "",
        )

    monkeypatch.setattr(op, "_run_modal_dispatch_process", fake_run)
    monkeypatch.setattr(op, "_modal_mount_upload_retry_settings", lambda: (3, 0.0))
    monkeypatch.setattr(op.time, "sleep", lambda s: sleeps.append(s))

    rc = op._dispatch_modal(recipe, "unit_job_002", "")

    assert rc == 0
    assert len(attempts) == 2
    assert sleeps == []


def test_operator_authorize_modal_dispatch_fails_closed_after_mount_retry_budget(
    monkeypatch,
) -> None:
    recipe = op.Recipe(
        name="unit_modal",
        path=op.RECIPES_DIR / "unit_modal.yaml",
        raw={
            "lane_id": "lane_d4_wyner_ziv_frame_0_substrate_20260514",
            "platform": "modal",
            "gpu": "T4",
            "timeout_hours": 1.0,
            "remote_driver": "scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh",
            "modal": {
                "lane_script": "scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh",
            },
        },
    )
    attempts: list[list[str]] = []

    def fake_run(cmd: list[str]):
        attempts.append([str(part) for part in cmd])
        return op.subprocess.CompletedProcess(
            cmd,
            1,
            "",
            "experiments/train_substrate_d4_wyner_ziv_frame_0.py "
            "was modified during build process\n",
        )

    monkeypatch.setattr(op, "_run_modal_dispatch_process", fake_run)
    monkeypatch.setattr(op, "_modal_mount_upload_retry_settings", lambda: (2, 0.0))
    monkeypatch.setattr(op.time, "sleep", lambda _s: None)

    rc = op._dispatch_modal(recipe, "unit_job_003", "")

    assert rc == 1
    assert len(attempts) == 2


def test_operator_authorize_prevalidates_before_claim(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(op, "_predict_cost_band", lambda **_: _band())
    monkeypatch.setattr(op, "_validate_required_input_files", lambda *_: None)
    monkeypatch.setattr(op, "_confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        op,
        "_native_dispatch_preflight",
        lambda *_: (_ for _ in ()).throw(SystemExit("missing provider driver")),
    )
    monkeypatch.setattr(op, "_claim_lane", lambda **_: events.append("claim"))
    monkeypatch.setattr(op, "_run_dispatch", lambda *_: events.append("dispatch") or 0)

    with pytest.raises(SystemExit, match="missing provider driver"):
        op.main(["--recipe", "phase1_t1_balle_cheap_config_dispatch"])

    assert events == []


def test_operator_authorize_closes_claim_on_dispatch_rc(monkeypatch) -> None:
    events: list[tuple[str, str | None]] = []
    monkeypatch.setattr(op, "_predict_cost_band", lambda **_: _band())
    monkeypatch.setattr(op, "_validate_required_input_files", lambda *_: None)
    monkeypatch.setattr(op, "_confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(op, "_native_dispatch_preflight", lambda *_: events.append(("preflight", None)))
    monkeypatch.setattr(op, "_claim_lane", lambda **_: events.append(("claim", None)))
    monkeypatch.setattr(op, "_run_dispatch", lambda *_: events.append(("dispatch", None)) or 7)
    monkeypatch.setattr(
        op,
        "_terminal_claim",
        lambda **kwargs: events.append(("terminal", str(kwargs["status"]))),
    )

    rc = op.main(["--recipe", "phase1_t1_balle_cheap_config_dispatch"])

    assert rc == 7
    assert events == [
        ("preflight", None),
        ("claim", None),
        ("dispatch", None),
        ("terminal", "failed_dispatch_rc_7"),
    ]


def test_operator_authorize_phase1_platform_env_reaches_vastai(
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setenv("PHASE1_PLATFORM", "vastai")
    monkeypatch.setattr(op, "_predict_cost_band", lambda **kwargs: events.append(kwargs["platform_key"]) or _band())
    monkeypatch.setattr(op, "_validate_required_input_files", lambda *_: None)
    monkeypatch.setattr(op, "_confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(op, "_native_dispatch_preflight", lambda recipe: events.append(recipe.platform))
    monkeypatch.setattr(op, "_claim_lane", lambda **kwargs: events.append(str(kwargs["platform"])))
    monkeypatch.setattr(op, "_run_dispatch", lambda recipe, *_: events.append(f"dispatch:{recipe.platform}") or 0)

    rc = op.main(["--recipe", "phase1_t1_balle_cheap_config_dispatch"])

    assert rc == 0
    assert events == ["vastai", "vastai", "vastai", "dispatch:vastai"]
