"""Tests for the live-vehicle lever activation ingester.

Anchor: `.omx/research/ddm_todo_p0_live_lever_queue_20260817.md`. The retired-registry defect this
module cures is a HARDCODED VEHICLE POINTER, so the load-bearing test is that the lever set is
DERIVED and moves with the source (`test_live_levers_are_derived_not_hardcoded`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.pr130_lift import live_lever_activation as lla
from tac.witness_dsl import activation_ledger as al

EMISSION = {
    "F1_weight_perturb_robustness": {
        "active": False,
        "reason_if_off": "sigma == 0 (default)",
        "sigma_in_quantiser_steps": 0.0,
    },
    "F2_weight_qat_q3q4": {"active": True, "high_bits": 4, "low_bits": 3, "reason_if_off": None},
    "F3_film_row_dropout": {"active": False, "probability": 0.0, "reason_if_off": "p == 0 (default)"},
    "F4_carrier_rank_penalty": {
        "active": False,
        "reason_if_off": "weight == 0 or no carrier tensors named (default)",
        "tensors": [],
    },
    "F5_gate_aware_conditioning": {"active": False, "state": "DECLARED_UNBUILT_FOLLOW_ON"},
}


def _write_log(tmp_path: Path, blob: dict, *, prefix: str = "", suffix: str = "") -> Path:
    log = tmp_path / "run.log"
    log.write_text(
        f"{prefix}[b2e] editability levers ACTIVE: {json.dumps(blob)}\n{suffix}",
        encoding="utf-8",
    )
    return log


# --- derivation (the anti-staleness contract) -------------------------------------------------


def test_live_levers_are_derived_not_hardcoded(tmp_path: Path) -> None:
    """A synthetic trainer yields ITS OWN flags — proof nothing is baked in."""
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--zzz-invented-lever', type=float, default=0.0)\n"
        "p.add_argument('--out', required=True)\n",
        encoding="utf-8",
    )
    levs = lla.live_levers(fake)
    assert [l.flag for l in levs] == ["--zzz-invented-lever"], "must derive, and must drop plumbing"
    assert levs[0].default == "0.0"


def test_live_trainer_flags_exclude_plumbing_and_include_known_levers() -> None:
    flags = {l.flag for l in lla.live_levers()}
    assert "--film-row-dropout" in flags
    assert "--carrier-rank-penalty" in flags
    assert "--distill-weight" in flags
    for plumbing in ("--cache", "--out", "--device", "--seed", "--eval-every"):
        assert plumbing not in flags, f"{plumbing} is plumbing, not a lever"


def test_ledger_name_normalizes_the_flag() -> None:
    assert lla.LiveLever("--film-row-dropout", "0.0").ledger_name == "film_row_dropout"


def test_store_true_default_is_reported_as_such(tmp_path: Path) -> None:
    fake = tmp_path / "t.py"
    fake.write_text(
        "import argparse\np = argparse.ArgumentParser()\n"
        "p.add_argument('--a-switch', action='store_true')\n",
        encoding="utf-8",
    )
    assert lla.live_levers(fake)[0].default == "store_true"


# --- emission parsing -------------------------------------------------------------------------


def test_parse_splits_active_from_inactive(tmp_path: Path) -> None:
    em = lla.parse_editability_emission(_write_log(tmp_path, EMISSION))
    assert em is not None
    assert em.active == ("weight_qat_q3q4",)
    assert em.inactive == (
        "carrier_rank_penalty",
        "film_row_dropout",
        "gate_aware_conditioning",
        "weight_perturb_robustness",
    )


def test_parse_keeps_reason_if_off_and_falls_back_to_state(tmp_path: Path) -> None:
    em = lla.parse_editability_emission(_write_log(tmp_path, EMISSION))
    assert em.reasons_if_off["film_row_dropout"] == "p == 0 (default)"
    # F5 has no reason_if_off; its `state` carries the honest reason instead.
    assert em.reasons_if_off["gate_aware_conditioning"] == "DECLARED_UNBUILT_FOLLOW_ON"


def test_parse_survives_surrounding_log_noise(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        EMISSION,
        prefix="[admission-guard] admission OK (governed)\n{'not': 'the emission'}\n",
        suffix='{"step": 0, "quantized_exact_seg": 0.00028}\n',
    )
    em = lla.parse_editability_emission(log)
    assert em is not None and em.active == ("weight_qat_q3q4",)


def test_parse_handles_braces_inside_strings(tmp_path: Path) -> None:
    """Brace matching must not be fooled by a `}` inside a reason string."""
    blob = {"F1_x": {"active": False, "reason_if_off": 'off because cfg{"a": 1} says so'}}
    em = lla.parse_editability_emission(_write_log(tmp_path, blob))
    assert em is not None and em.inactive == ("x",)


def test_parse_returns_none_when_absent(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("nothing to see here\n", encoding="utf-8")
    assert lla.parse_editability_emission(log) is None


def test_parse_returns_none_on_malformed_json(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("[b2e] editability levers ACTIVE: {not json,,}\n", encoding="utf-8")
    assert lla.parse_editability_emission(log) is None


def test_unprefixed_keys_are_accepted_verbatim(tmp_path: Path) -> None:
    em = lla.parse_editability_emission(_write_log(tmp_path, {"plain_name": {"active": True}}))
    assert em.active == ("plain_name",)


# --- ingest (the load-bearing behavior) -------------------------------------------------------


def test_ingest_records_only_active_levers(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    rows = lla.ingest_run_log(_write_log(tmp_path, EMISSION), path=ledger)
    assert [r["lever"] for r in rows] == ["weight_qat_q3q4"]
    assert all(r["event"] == al.EVENT_FIRED for r in rows)


def test_ingest_never_writes_a_row_for_an_inactive_lever(tmp_path: Path) -> None:
    """The queue derives from ABSENCE — recording an off lever would corrupt the signal."""
    ledger = tmp_path / "ledger.jsonl"
    lla.ingest_run_log(_write_log(tmp_path, EMISSION), path=ledger)
    written = {json.loads(line)["lever"] for line in ledger.read_text().splitlines() if line.strip()}
    assert "film_row_dropout" not in written
    assert "carrier_rank_penalty" not in written
    assert written == {"weight_qat_q3q4"}


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    rows = lla.ingest_run_log(_write_log(tmp_path, EMISSION), path=ledger, dry_run=True)
    assert rows and not ledger.exists()


def test_ingest_is_a_noop_without_an_emission(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("no emission here\n", encoding="utf-8")
    assert lla.ingest_run_log(log, path=tmp_path / "l.jsonl") == ()


def test_run_ref_defaults_to_the_run_directory(tmp_path: Path) -> None:
    em = lla.parse_editability_emission(_write_log(tmp_path, EMISSION))
    assert em.run_ref == str(tmp_path)


def test_explicit_run_ref_overrides(tmp_path: Path) -> None:
    em = lla.parse_editability_emission(_write_log(tmp_path, EMISSION), run_ref="EF6000")
    assert em.run_ref == "EF6000"


# --- manifest ingest (the other half: 14 levers have NO per-run telemetry) --------------------


def _write_manifest(tmp_path: Path, argv: list[str]) -> Path:
    m = tmp_path / "launch_manifest.json"
    m.write_text(json.dumps({"effective_argv": argv}), encoding="utf-8")
    return m


BASE_ARGV = ["python", "-m", "tac.pr130_lift.train_semantic_quantized_resumable"]


def test_manifest_records_off_default_values(tmp_path: Path) -> None:
    ledger = tmp_path / "l.jsonl"
    m = _write_manifest(tmp_path, BASE_ARGV + ["--ce-fraction", "0.0", "--lr", "9e-5"])
    rows = lla.ingest_launch_manifest(m, path=ledger)
    assert {r["lever"] for r in rows} == {"ce_fraction", "lr"}


def test_manifest_skips_values_equal_to_the_default(tmp_path: Path) -> None:
    """Passing a lever at its own default is NOT a firing — it changes nothing."""
    ledger = tmp_path / "l.jsonl"
    m = _write_manifest(tmp_path, BASE_ARGV + ["--ce-fraction", "0.5"])
    assert lla.ingest_launch_manifest(m, path=ledger) == ()


def test_manifest_default_comparison_is_numeric_not_textual(tmp_path: Path) -> None:
    """`2e-5` and the source's `2e-05` are the same number; a string compare would misfire."""
    ledger = tmp_path / "l.jsonl"
    m = _write_manifest(tmp_path, BASE_ARGV + ["--lr", "2e-5"])
    assert lla.ingest_launch_manifest(m, path=ledger) == ()


def test_manifest_store_true_presence_is_a_firing(tmp_path: Path) -> None:
    ledger = tmp_path / "l.jsonl"
    m = _write_manifest(tmp_path, BASE_ARGV + ["--weight-qat-q3q4"])
    assert [r["lever"] for r in lla.ingest_launch_manifest(m, path=ledger)] == ["weight_qat_q3q4"]


def test_manifest_ignores_other_trainers(tmp_path: Path) -> None:
    ledger = tmp_path / "l.jsonl"
    m = _write_manifest(tmp_path, ["python", "-m", "some.other.trainer", "--ce-fraction", "0.0"])
    assert lla.ingest_launch_manifest(m, path=ledger) == ()


def test_manifest_ignores_plumbing_flags(tmp_path: Path) -> None:
    ledger = tmp_path / "l.jsonl"
    m = _write_manifest(tmp_path, BASE_ARGV + ["--seed", "1234", "--device", "cpu"])
    assert lla.ingest_launch_manifest(m, path=ledger) == ()


def test_manifest_tolerates_missing_or_malformed_file(tmp_path: Path) -> None:
    assert lla.ingest_launch_manifest(tmp_path / "nope.json") == ()
    bad = tmp_path / "launch_manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    assert lla.ingest_launch_manifest(bad) == ()


def test_manifest_accepts_a_string_argv(tmp_path: Path) -> None:
    m = tmp_path / "launch_manifest.json"
    m.write_text(
        json.dumps({"argv": " ".join(BASE_ARGV + ["--ce-fraction", "0.0"])}), encoding="utf-8"
    )
    assert [r["lever"] for r in lla.ingest_launch_manifest(m, dry_run=True)] == ["ce_fraction"]


# --- the queue itself -------------------------------------------------------------------------


def test_live_never_fired_reflects_ingested_events(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    before = lla.live_never_fired(path=ledger)
    assert "weight_qat_q3q4" in before, "empty ledger: every live lever is never-fired"
    lla.ingest_run_log(_write_log(tmp_path, EMISSION), path=ledger)
    after = lla.live_never_fired(path=ledger)
    assert "weight_qat_q3q4" not in after, "ingest must retire it from the queue"
    assert "film_row_dropout" in after, "an off lever stays in the queue"
    assert set(after) < set(before)


def test_live_queue_is_disjoint_from_the_retired_registry_queue() -> None:
    """The measured defect, pinned: the retired queue answers for a different vehicle.

    If a future change makes the registry describe the live vehicle, this test SHOULD fail — that
    is the signal to retire this module's `known` override, not to weaken the assertion.
    """
    live = {l.ledger_name for l in lla.live_levers()}
    retired = set(al.never_fired())
    assert live, "derivation returned nothing — the trainer path is wrong"
    assert not (live & retired), (
        "live levers now appear in the retired-registry queue; re-check whether "
        "lever_registry.describes_live_vehicle has been repaired"
    )


@pytest.mark.parametrize("flag", ["--film-row-dropout", "--distill-weight", "--carrier-rank-penalty"])
def test_evidence_backed_levers_are_in_the_live_queue(flag: str) -> None:
    """These three carry measured evidence in memos and have never been passed to this trainer."""
    assert flag.removeprefix("--").replace("-", "_") in lla.live_never_fired()


def test_manifest_skips_a_value_taking_flag_with_no_value(tmp_path: Path) -> None:
    """Malformed argv: we cannot know the value, so recording a firing would be a guess."""
    m = _write_manifest(tmp_path, BASE_ARGV + ["--ce-fraction"])
    assert lla.ingest_launch_manifest(m, dry_run=True) == ()
    m2 = _write_manifest(tmp_path, BASE_ARGV + ["--ce-fraction", "--lr", "9e-5"])
    assert [r["lever"] for r in lla.ingest_launch_manifest(m2, dry_run=True)] == ["lr"]


def test_manifest_store_true_at_end_of_argv_still_fires(tmp_path: Path) -> None:
    """A store_true flag needs no value — its presence IS the firing, even last."""
    m = _write_manifest(tmp_path, BASE_ARGV + ["--weight-qat-q3q4"])
    assert [r["lever"] for r in lla.ingest_launch_manifest(m, dry_run=True)] == ["weight_qat_q3q4"]
