from __future__ import annotations

import pytest

import tools.operator_authorize as op


def _band() -> op.CostBandPrediction:
    return op.CostBandPrediction(
        p10_cost_usd=0.0,
        p50_cost_usd=0.0,
        p90_cost_usd=0.0,
        n_anchors=0,
        confidence_tag="test",
        source="test",
    )


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
