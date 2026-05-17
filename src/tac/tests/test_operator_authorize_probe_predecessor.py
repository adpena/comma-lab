# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

import tools.operator_authorize as operator_authorize
from tac import probe_outcomes_ledger
from tac.probe_outcomes_ledger import (
    VERDICT_INDEPENDENT,
    register_probe_outcome,
)


def _register_blocking_atw_probe(path: Path) -> None:
    register_probe_outcome(
        probe_id="atw_v2_d4_h_latent_given_scorer_class_20260516",
        substrate="atw_codec_v2",
        recipe_path=".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml",
        probe_kind="h_latent_given_scorer_class",
        verdict=VERDICT_INDEPENDENT,
        metric_name="mutual_information_bits_per_symbol",
        metric_value=0.006385502752,
        threshold=0.5,
        threshold_token="MEANINGFUL_CONDITIONING",
        evidence_path=".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md",
        next_action="do_not_dispatch_atw_v2_phase2_from_this_signal",
        path=path,
        lock_path=path.with_suffix(path.suffix + ".lock"),
    )


def test_operator_authorize_probe_predecessor_blocks_matching_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    _register_blocking_atw_probe(ledger)
    monkeypatch.setattr(probe_outcomes_ledger, "PROBE_OUTCOMES_LEDGER_PATH", ledger)
    monkeypatch.delenv("OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT", raising=False)
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE",
        raising=False,
    )

    with pytest.raises(SystemExit, match="probe-disambiguator predecessor verdict"):
        operator_authorize._check_predecessor_probe_outcome(
            Path(".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml")
        )


def test_operator_authorize_probe_predecessor_requires_paired_bypass_rationale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    _register_blocking_atw_probe(ledger)
    monkeypatch.setattr(probe_outcomes_ledger, "PROBE_OUTCOMES_LEDGER_PATH", ledger)
    monkeypatch.setenv("OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT", "1")
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE",
        raising=False,
    )

    with pytest.raises(SystemExit, match="requires paired"):
        operator_authorize._check_predecessor_probe_outcome(
            Path(".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml")
        )


def test_operator_authorize_probe_predecessor_allows_paired_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    _register_blocking_atw_probe(ledger)
    monkeypatch.setattr(probe_outcomes_ledger, "PROBE_OUTCOMES_LEDGER_PATH", ledger)
    monkeypatch.setenv("OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT", "1")
    monkeypatch.setenv(
        "OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE",
        "operator approved fresh sister probe 2026-05-17",
    )

    operator_authorize._check_predecessor_probe_outcome(
        Path(".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml")
    )

    captured = capsys.readouterr()
    assert "PROBE-PREDECESSOR BYPASS ACTIVE" in captured.err
