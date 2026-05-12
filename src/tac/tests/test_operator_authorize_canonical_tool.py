from __future__ import annotations

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
    monkeypatch.setattr(op, "_claim_lane", lambda **_: events.append("claim"))
    monkeypatch.setattr(op, "_run_dispatch", lambda *_: events.append("dispatch") or 0)

    rc = op.main(["--recipe", "phase1_t1_balle_cheap_config_dispatch"])

    assert rc == 0
    assert events == ["claim", "dispatch"]

