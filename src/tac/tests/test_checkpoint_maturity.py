# SPDX-License-Identifier: MIT
"""Tests for the ``_dev``/``_prod`` checkpoint-maturity convention.

Covers: parse cases (dev / prod / untagged-vehicle / legacy / malformed /
case-insensitive / ambiguous), the strict ``is_pointer_promotable`` truth
table, the grandfathering ``pointer_promotion_verdict``, the prod-bank
immutability guard (dev never clobbers prod), the composed bank naming, and
the canonical-frontier-pointer refusal gate (a dev row truly cannot promote —
the prior anchor is kept and the refusal is recorded).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.canonical_frontier_pointer import (
    AnchorRecord,
    _checkpoint_maturity_refusal,
    _gate_axis_anchor,
)
from tac.checkpoint_maturity import (
    MATURITY_DEV,
    MATURITY_PROD,
    MATURITY_UNKNOWN,
    CheckpointMaturity,
    ProdBankImmutableError,
    assert_bank_dir_writable,
    bank_dir_name,
    is_pointer_promotable,
    parse_checkpoint_maturity,
    pointer_promotion_verdict,
)

# ---------------------------------------------------------------- parse cases


def test_parse_v9c3_dev() -> None:
    cm = parse_checkpoint_maturity("v9c3_dev")
    assert cm == CheckpointMaturity(name="v9c3_dev", vehicle="v9c3", maturity=MATURITY_DEV, reason=cm.reason)
    assert cm.maturity == MATURITY_DEV
    assert cm.vehicle == "v9c3"


def test_parse_v10_prod_with_suffix() -> None:
    cm = parse_checkpoint_maturity("v10_prod_bank_20260801")
    assert cm.vehicle == "v10"
    assert cm.maturity == MATURITY_PROD


def test_parse_untagged_vehicle_bank_is_unknown_safe_side() -> None:
    # The real hand-created bank dir: vehicle-shaped, no maturity tag.
    cm = parse_checkpoint_maturity("v9c2_defensive_bank_20260718")
    assert cm.vehicle == "v9c2"
    assert cm.maturity == MATURITY_UNKNOWN
    # 'defensive' must NOT fuzzy-match 'dev' (exact-token discipline).
    assert "dev" not in cm.reason.split("'")[0].lower() or "safe-side" in cm.reason


def test_parse_legacy_non_vehicle_name() -> None:
    cm = parse_checkpoint_maturity("levelset_n600_witness_20260717T113932Z")
    assert cm.vehicle is None
    assert cm.maturity == MATURITY_UNKNOWN


def test_parse_legacy_pr_lane_name() -> None:
    cm = parse_checkpoint_maturity("pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515")
    assert cm.vehicle is None
    assert cm.maturity == MATURITY_UNKNOWN


def test_parse_malformed_and_empty() -> None:
    assert parse_checkpoint_maturity("").maturity == MATURITY_UNKNOWN
    assert parse_checkpoint_maturity("___").maturity == MATURITY_UNKNOWN
    assert parse_checkpoint_maturity("   ").vehicle is None


def test_parse_case_insensitive() -> None:
    cm = parse_checkpoint_maturity("V10_PROD")
    assert cm.vehicle == "v10"
    assert cm.maturity == MATURITY_PROD


def test_parse_ambiguous_both_tokens_dev_wins() -> None:
    cm = parse_checkpoint_maturity("v9c3_dev_prod")
    assert cm.maturity == MATURITY_DEV  # safe side: never silently promotable


def test_parse_path_input_uses_final_component() -> None:
    cm = parse_checkpoint_maturity(Path("/x/banks/v9c3_dev_bank_20260718"))
    assert cm.vehicle == "v9c3"
    assert cm.maturity == MATURITY_DEV


def test_parse_strips_known_extension() -> None:
    cm = parse_checkpoint_maturity("v10_prod_ema.npz")
    assert cm.maturity == MATURITY_PROD


# ---------------------------------------------------- is_pointer_promotable


def test_is_pointer_promotable_truth_table() -> None:
    assert is_pointer_promotable("v10_prod") is True
    assert is_pointer_promotable("v10_prod_bank_20260801") is True
    assert is_pointer_promotable("v9c3_dev") is False
    assert is_pointer_promotable("v9c2_defensive_bank_20260718") is False  # untagged vehicle
    assert is_pointer_promotable("levelset_n600_witness_20260717T113932Z") is False  # legacy: strict surface
    assert is_pointer_promotable("") is False


# ------------------------------------------------ pointer_promotion_verdict


def test_verdict_prod_allowed() -> None:
    allowed, reason = pointer_promotion_verdict("v10_prod_byteclose")
    assert allowed is True
    assert "_prod" in reason


def test_verdict_dev_refused() -> None:
    allowed, reason = pointer_promotion_verdict("v9c3_dev_byteclose")
    assert allowed is False
    assert "NON-promotable" in reason


def test_verdict_untagged_vehicle_refused() -> None:
    allowed, reason = pointer_promotion_verdict("v9c2_defensive_bank_20260718")
    assert allowed is False
    assert "safe-side" in reason


def test_verdict_legacy_grandfathered() -> None:
    allowed, reason = pointer_promotion_verdict("pr101_frame_exploit_selector_fec6")
    assert allowed is True
    assert "grandfathered" in reason


# -------------------------------------------------------------- bank naming


def test_bank_dir_name_composes() -> None:
    assert bank_dir_name("v9c3", "dev", "bank", "20260718") == "v9c3_dev_bank_20260718"
    assert bank_dir_name("v10", "prod", "bank", "20260801") == "v10_prod_bank_20260801"


def test_bank_dir_name_refuses_bad_inputs() -> None:
    with pytest.raises(ValueError):
        bank_dir_name("v9c3", "dve", "bank", "20260718")  # typo fail-closed
    with pytest.raises(ValueError):
        bank_dir_name("witness", "dev", "bank", "20260718")  # not a vehicle token


# -------------------------------------------- prod-bank immutability guard


def test_bank_guard_refuses_existing_prod_dir(tmp_path: Path) -> None:
    prod = tmp_path / "v10_prod_bank_20260801"
    prod.mkdir()
    with pytest.raises(ProdBankImmutableError):
        assert_bank_dir_writable(prod)


def test_bank_guard_allows_dev_and_new_dirs(tmp_path: Path) -> None:
    dev = tmp_path / "v9c3_dev_bank_20260718"
    dev.mkdir()
    assert assert_bank_dir_writable(dev) == dev  # dev lane may iterate in place
    fresh = tmp_path / "v10_prod_bank_20260901"  # does not exist yet -> allowed
    assert assert_bank_dir_writable(fresh) == fresh


def test_bank_guard_dev_and_prod_coexist(tmp_path: Path) -> None:
    (tmp_path / "v9c3_prod_bank_20260718").mkdir()
    dev = tmp_path / "v9c3_dev_bank_20260718"
    # A dev bank write for the SAME vehicle lands beside the prod bank —
    # distinct name, prod dir untouched.
    assert assert_bank_dir_writable(dev) == dev
    assert (tmp_path / "v9c3_prod_bank_20260718").is_dir()


# ------------------------------------------------- pointer refusal gate (B)


def _anchor(lane_id: str | None = None, source_path: str | None = None, **extra: str) -> AnchorRecord:
    return AnchorRecord(
        score=0.150,
        axis="contest_cpu",
        archive_sha256="deadbeef" * 8,
        lane_id=lane_id,
        hardware_substrate="linux_x86_64",
        measured_at_utc="2026-07-18T00:00:00+00:00",
        evidence_grade="[contest-CPU]",
        source_path=source_path,
        extra=dict(extra),
    )


def test_refusal_none_for_legacy_anchor() -> None:
    assert _checkpoint_maturity_refusal(_anchor(lane_id="pr101_frame_exploit_selector_fec6")) is None
    assert _checkpoint_maturity_refusal(None) is None


def test_refusal_for_dev_lane_anchor() -> None:
    reason = _checkpoint_maturity_refusal(_anchor(lane_id="v9c3_dev_byteclose_20260718"))
    assert reason is not None and "NON-promotable" in reason


def test_refusal_scans_source_path_segments() -> None:
    reason = _checkpoint_maturity_refusal(
        _anchor(source_path="experiments/results/v9c3_dev_run/contest_auth_eval.json")
    )
    assert reason is not None


def test_refusal_scans_extra_run_dir() -> None:
    reason = _checkpoint_maturity_refusal(_anchor(run_dir="experiments/results/banks/v9c2_defensive_bank_20260718"))
    assert reason is not None and "safe-side" in reason


def test_no_refusal_for_prod_anchor() -> None:
    assert _checkpoint_maturity_refusal(_anchor(lane_id="v10_prod_byteclose_20260901")) is None


def test_gate_keeps_prior_anchor_on_dev_candidate(capsys: pytest.CaptureFixture[str]) -> None:
    prior = _anchor(lane_id="pr101_frame_exploit_selector_fec6")
    dev_candidate = _anchor(lane_id="v9c3_dev_byteclose_20260718")
    anchor, refusal = _gate_axis_anchor(dev_candidate, prior, axis_label="contest_cpu")
    assert anchor is prior  # fail-closed: pointer untouched
    assert refusal is not None
    assert refusal["axis"] == "contest_cpu"
    assert refusal["refused_lane_id"] == "v9c3_dev_byteclose_20260718"
    err = capsys.readouterr().err
    assert "REFUSED pointer promotion" in err  # loud, not silent


def test_gate_passes_prod_candidate_through() -> None:
    prior = _anchor(lane_id="pr101_frame_exploit_selector_fec6")
    prod_candidate = _anchor(lane_id="v10_prod_byteclose_20260901")
    anchor, refusal = _gate_axis_anchor(prod_candidate, prior, axis_label="contest_cpu")
    assert anchor is prod_candidate
    assert refusal is None


def test_gate_dev_candidate_with_no_prior_yields_none() -> None:
    dev_candidate = _anchor(lane_id="v9c3_dev_byteclose_20260718")
    anchor, refusal = _gate_axis_anchor(dev_candidate, None, axis_label="contest_cuda")
    assert anchor is None  # still fail-closed: never the dev row
    assert refusal is not None and refusal["axis"] == "contest_cuda"
