# SPDX-License-Identifier: MIT
from __future__ import annotations

import inspect
from pathlib import Path

from tac import preflight
from tac.probe_outcomes_ledger import (
    VERDICT_INDEPENDENT,
    register_probe_outcome,
)


def _register_blocking_atw_outcome(ledger: Path, lock: Path) -> None:
    register_probe_outcome(
        probe_id="atw_v2_d4_h_latent_given_scorer_class_20260516",
        substrate="atw_codec_v2",
        recipe_path=(
            ".omx/operator_authorize_recipes/"
            "substrate_atw_codec_v2_modal_a100_dispatch.yaml"
        ),
        probe_kind="h_latent_given_scorer_class",
        verdict=VERDICT_INDEPENDENT,
        metric_name="mutual_information_bits_per_symbol",
        metric_value=0.006385,
        threshold=0.5,
        threshold_token="MEANINGFUL_CONDITIONING",
        evidence_path=".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md",
        next_action="do_not_dispatch_atw_v2_phase2_from_this_signal",
        path=ledger,
        lock_path=lock,
    )


def test_check_313_blocks_common_operator_authorize_recipe_name(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "state" / "probe_outcomes.jsonl"
    lock = tmp_path / "state" / "probe_outcomes.jsonl.lock"
    _register_blocking_atw_outcome(ledger, lock)
    wrapper = tmp_path / "tools" / "fire_atw.py"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "cmd = 'tools/operator_authorize.py --recipe "
        "substrate_atw_codec_v2_modal_a100_dispatch'\n",
        encoding="utf-8",
    )

    violations = preflight.check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=tmp_path,
        ledger_path=ledger,
    )

    assert len(violations) == 1
    assert "fire_atw.py:1" in violations[0]
    assert "INDEPENDENT" in violations[0]
    assert "atw_v2_d4_h_latent_given_scorer_class_20260516" in violations[0]


def test_check_313_blocks_direct_recipe_yaml_literal(tmp_path: Path) -> None:
    ledger = tmp_path / "state" / "probe_outcomes.jsonl"
    lock = tmp_path / "state" / "probe_outcomes.jsonl.lock"
    _register_blocking_atw_outcome(ledger, lock)
    wrapper = tmp_path / "scripts" / "fire_atw.sh"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "modal run tools/operator_authorize.py --recipe "
        ".omx/operator_authorize_recipes/"
        "substrate_atw_codec_v2_modal_a100_dispatch.yaml\n",
        encoding="utf-8",
    )

    violations = preflight.check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=tmp_path,
        ledger_path=ledger,
    )

    assert len(violations) == 1
    assert "fire_atw.sh:1" in violations[0]


def test_check_313_allows_same_line_operator_override_waiver(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "state" / "probe_outcomes.jsonl"
    lock = tmp_path / "state" / "probe_outcomes.jsonl.lock"
    _register_blocking_atw_outcome(ledger, lock)
    wrapper = tmp_path / "tools" / "fire_atw.py"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "cmd = 'tools/operator_authorize.py --recipe "
        "substrate_atw_codec_v2_modal_a100_dispatch' "
        "# PROBE_PREDECESSOR_OVERRIDE_OK:fresh alternative reducer adjudicated\n",
        encoding="utf-8",
    )

    violations = preflight.check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=tmp_path,
        ledger_path=ledger,
    )

    assert violations == []


def test_check_313_is_wired_into_preflight_all() -> None:
    source = inspect.getsource(preflight.preflight_all)

    assert "check_dispatch_target_has_no_predecessor_adjudicated_outcome" in source
    assert (
        "check_dispatch_target_has_no_predecessor_adjudicated_outcome(\n"
        "            strict=True"
    ) in source
