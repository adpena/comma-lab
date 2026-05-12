"""Tests for `tac.cost_band_calibration` (the calibration-of-calibration posterior).

Covers:
- Append + read roundtrip
- p10/p50/p90 percentile correctness
- Confidence-tag transitions (hand_calibrated_fallback → weak_posterior → empirical_posterior)
- fcntl-locked concurrent appends preserve every record
- Epochs-bucket matching
- Hand-calibrated fallback used when N=0
- JSON conformance (no NaN/Infinity emission)
"""
from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

from tac.cost_band_calibration import (
    SCHEMA_VERSION,
    CostBandAnchor,
    CostBandPrediction,
    _epochs_bucket,
    _percentile,
    append_anchor,
    load_anchors,
    parse_actual_cost_usd,
    predict,
    summary_by_bucket,
)

# -- Anchor roundtrip --------------------------------------------------------


def test_parse_actual_cost_usd_absent_is_not_zero_anchor() -> None:
    assert parse_actual_cost_usd(None) is None
    assert parse_actual_cost_usd("") is None
    assert parse_actual_cost_usd("  ") is None


def test_parse_actual_cost_usd_rejects_nonfinite_or_negative() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_actual_cost_usd("nan")
    with pytest.raises(ValueError):
        parse_actual_cost_usd("-0.01")


def test_parse_actual_cost_usd_accepts_measured_zero() -> None:
    assert parse_actual_cost_usd("0.0") == 0.0

def _make_anchor(
    *,
    label: str = "test",
    platform: str = "modal",
    gpu: str = "T4",
    epochs: int = 3000,
    batch: int = 32,
    flags: bool = True,
    wallclock_sec: float = 5400.0,
    cost: float = 8.0,
) -> CostBandAnchor:
    return CostBandAnchor(
        logged_at_utc="2026-05-12T18:00:00+00:00",
        dispatch_label=label,
        trainer="experiments/train_x.py",
        platform=platform,
        gpu=gpu,
        epochs=epochs,
        batch_size=batch,
        all_flags_on=flags,
        actual_wall_clock_sec=wallclock_sec,
        actual_cost_usd=cost,
    )


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    append_anchor(_make_anchor(label="A", cost=8.0), posterior_path=pp, lock_path=lp)
    append_anchor(_make_anchor(label="B", cost=10.0), posterior_path=pp, lock_path=lp)
    out = load_anchors(pp)
    assert [a.dispatch_label for a in out] == ["A", "B"]
    assert [a.actual_cost_usd for a in out] == [8.0, 10.0]


def test_load_handles_missing_file(tmp_path: Path) -> None:
    pp = tmp_path / "nonexistent.jsonl"
    assert load_anchors(pp) == []


def test_load_skips_malformed_lines(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(
        json.dumps({"schema": SCHEMA_VERSION, "logged_at_utc": "2026-05-12T00:00:00+00:00",
                    "dispatch_label": "A", "trainer": "x", "platform": "modal",
                    "gpu": "T4", "epochs": 3000, "batch_size": 32,
                    "all_flags_on": True, "actual_wall_clock_sec": 100.0,
                    "actual_cost_usd": 1.0}) + "\n"
        + "not valid json\n"
        + json.dumps({"schema": "wrong_schema_v999"}) + "\n"
    )
    out = load_anchors(pp)
    assert len(out) == 1 and out[0].dispatch_label == "A"


# -- Percentile + epochs-bucket ----------------------------------------------

def test_percentile_basic() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(vals, 0) == 1.0
    assert _percentile(vals, 50) == 3.0
    assert _percentile(vals, 100) == 5.0


def test_percentile_single_value() -> None:
    assert _percentile([7.5], 50) == 7.5
    assert _percentile([7.5], 90) == 7.5


def test_percentile_empty() -> None:
    assert _percentile([], 50) == 0.0


def test_epochs_bucket_boundaries() -> None:
    assert _epochs_bucket(1) == 50
    assert _epochs_bucket(50) == 50
    assert _epochs_bucket(150) == 100
    assert _epochs_bucket(600) == 500
    assert _epochs_bucket(3000) == 3000
    assert _epochs_bucket(3500) == 3000  # 3500 ≤ 4500 → bucket 3000
    assert _epochs_bucket(5000) == 6000  # 5000 > 4500 → next 3k bucket
    assert _epochs_bucket(10_000) == 12_000


# -- predict() confidence-tag transitions ------------------------------------

def test_predict_no_anchors_uses_hand_calibrated_fallback(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    p = predict("modal", "T4", 3000, all_flags_on=True, posterior_path=pp)
    assert p.confidence_tag == "hand_calibrated_fallback"
    assert p.n_anchors == 0
    assert p.p50_cost_usd > 0  # hand-calibrated stub for (modal, T4, 3000, True) exists


def test_predict_no_anchors_no_stub_returns_zero_band(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    p = predict("kaggle", "K80", 99999, all_flags_on=False, posterior_path=pp)
    assert p.confidence_tag == "hand_calibrated_fallback"
    assert p.p50_cost_usd == 0.0
    assert "no anchors AND no hand-calibrated stub" in p.fallback_rationale


def test_predict_weak_posterior_with_one_anchor(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    append_anchor(_make_anchor(label="A", cost=6.0), posterior_path=pp, lock_path=lp)
    p = predict("modal", "T4", 3000, all_flags_on=True, posterior_path=pp)
    assert p.confidence_tag == "weak_posterior"
    assert p.n_anchors == 1
    # Single-value percentile = that value; widened by 1.5 on either side.
    assert p.p50_cost_usd == 6.0
    assert p.p10_cost_usd < p.p50_cost_usd
    assert p.p90_cost_usd > p.p50_cost_usd


def test_predict_empirical_posterior_with_three_anchors(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    for i, c in enumerate([4.0, 6.0, 8.0]):
        append_anchor(_make_anchor(label=f"A{i}", cost=c), posterior_path=pp, lock_path=lp)
    p = predict("modal", "T4", 3000, all_flags_on=True, posterior_path=pp)
    assert p.confidence_tag == "empirical_posterior"
    assert p.n_anchors == 3
    assert p.p50_cost_usd == 6.0


def test_predict_matches_only_same_bucket(tmp_path: Path) -> None:
    """Anchors with different (platform, gpu, epochs_bucket, flags) are ignored."""
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    # 3 anchors at T4 + 2 anchors at A100 with different costs.
    for i in range(3):
        append_anchor(
            _make_anchor(label=f"T4{i}", gpu="T4", cost=5.0),
            posterior_path=pp, lock_path=lp,
        )
    for i in range(2):
        append_anchor(
            _make_anchor(label=f"A100{i}", gpu="A100", cost=20.0),
            posterior_path=pp, lock_path=lp,
        )
    p_t4 = predict("modal", "T4", 3000, all_flags_on=True, posterior_path=pp)
    p_a100 = predict("modal", "A100", 3000, all_flags_on=True, posterior_path=pp)
    assert p_t4.n_anchors == 3 and p_t4.confidence_tag == "empirical_posterior"
    assert p_t4.p50_cost_usd == 5.0
    assert p_a100.n_anchors == 2 and p_a100.confidence_tag == "weak_posterior"
    assert p_a100.p50_cost_usd == 20.0


def test_predict_flags_off_separate_bucket(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    for i in range(3):
        append_anchor(_make_anchor(label=f"on{i}", flags=True, cost=8.0),
                       posterior_path=pp, lock_path=lp)
    p_on = predict("modal", "T4", 3000, all_flags_on=True, posterior_path=pp)
    p_off = predict("modal", "T4", 3000, all_flags_on=False, posterior_path=pp)
    assert p_on.n_anchors == 3
    assert p_off.n_anchors == 0
    assert p_off.confidence_tag == "hand_calibrated_fallback"


# -- fcntl-locked concurrent appends -----------------------------------------

def _spawn_appender(args: tuple[str, str, str, int, float]) -> None:
    pp_str, lp_str, label, epochs, cost = args
    from tac.cost_band_calibration import (
        CostBandAnchor as _CBA,
    )
    from tac.cost_band_calibration import (
        append_anchor as _ap,
    )
    anchor = _CBA(
        logged_at_utc="2026-05-12T19:00:00+00:00",
        dispatch_label=label,
        trainer="experiments/train_x.py",
        platform="modal", gpu="T4",
        epochs=epochs, batch_size=32, all_flags_on=True,
        actual_wall_clock_sec=3600.0, actual_cost_usd=cost,
    )
    _ap(anchor, posterior_path=Path(pp_str), lock_path=Path(lp_str))


def test_concurrent_appenders_preserve_all_records(tmp_path: Path) -> None:
    """4-process spawn pool simultaneously appends 5 records each;
    fcntl LOCK_EX serializes; all 20 records survive."""
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    ctx = multiprocessing.get_context("spawn")
    args = [
        (str(pp), str(lp), f"P{p}-A{i}", 3000, 5.0 + i * 0.1)
        for p in range(4) for i in range(5)
    ]
    with ctx.Pool(4) as pool:
        pool.map(_spawn_appender, args)
    out = load_anchors(pp)
    assert len(out) == 20
    assert len({a.dispatch_label for a in out}) == 20


# -- summary_by_bucket --

def test_summary_by_bucket_aggregates(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    for c in [5.0, 7.0, 9.0]:
        append_anchor(_make_anchor(cost=c), posterior_path=pp, lock_path=lp)
    out = summary_by_bucket(pp)
    assert len(out) == 1
    row = out[0]
    assert row["platform"] == "modal" and row["gpu"] == "T4"
    assert row["n_anchors"] == 3
    assert row["p50_cost_usd"] == 7.0
    assert row["min_cost_usd"] == 5.0
    assert row["max_cost_usd"] == 9.0


# -- JSON conformance --

def test_appended_lines_are_rfc8259_conformant(tmp_path: Path) -> None:
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    append_anchor(_make_anchor(cost=8.0), posterior_path=pp, lock_path=lp)
    text = pp.read_text(encoding="utf-8")
    # No Infinity / NaN tokens per RFC 8259.
    assert "Infinity" not in text
    assert "NaN" not in text
    # And each line is valid JSON.
    for line in text.splitlines():
        json.loads(line)


# -- Sister Catalog #128 atomicity invariant --

def test_lock_path_distinct_from_posterior_path(tmp_path: Path) -> None:
    """Lock file is a sibling, NOT the posterior itself. Required so a
    reader of the posterior doesn't take a lock while an appender holds it."""
    pp = tmp_path / "p.jsonl"
    lp = tmp_path / "p.lock"
    append_anchor(_make_anchor(), posterior_path=pp, lock_path=lp)
    assert pp.exists() and lp.exists()
    assert pp != lp


# -- CostBandPrediction.as_dict shape --

def test_prediction_as_dict_round_trip() -> None:
    p = CostBandPrediction(
        platform="modal", gpu="T4", epochs=3000, all_flags_on=True,
        n_anchors=0, p10_cost_usd=1.0, p50_cost_usd=2.0, p90_cost_usd=3.0,
        p10_wall_clock_hr=0.5, p50_wall_clock_hr=1.0, p90_wall_clock_hr=2.0,
        confidence_tag="hand_calibrated_fallback", freshness_seconds=None,
        fallback_rationale="test",
    )
    d = p.as_dict()
    assert d["platform"] == "modal"
    assert d["confidence_tag"] == "hand_calibrated_fallback"
    # No NaN/Infinity creeping in.
    s = json.dumps(d, allow_nan=False)
    assert "Infinity" not in s and "NaN" not in s


def test_append_cost_band_anchor_cli_uses_canonical_tool_bootstrap(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    posterior = tmp_path / "posterior.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "append_cost_band_anchor.py"),
            "--dispatch-label",
            "cli_smoke",
            "--trainer",
            "experiments/train_x.py",
            "--platform",
            "modal",
            "--gpu",
            "T4",
            "--epochs",
            "3000",
            "--batch-size",
            "16",
            "--all-flags-on",
            "--actual-wall-clock-sec",
            "60",
            "--actual-cost-usd",
            "0.01",
            "--posterior-path",
            str(posterior),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "appended cli_smoke" in proc.stdout
    anchors = load_anchors(posterior)
    assert [anchor.dispatch_label for anchor in anchors] == ["cli_smoke"]


def test_append_cost_band_anchor_cli_accepts_github_cpu_and_failed_outcome(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    posterior = tmp_path / "posterior.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "append_cost_band_anchor.py"),
            "--dispatch-label",
            "gha_failed_smoke",
            "--trainer",
            "upstream/.github/workflows/eval.yml",
            "--platform",
            "github",
            "--gpu",
            "cpu",
            "--epochs",
            "0",
            "--batch-size",
            "16",
            "--actual-wall-clock-sec",
            "60",
            "--actual-cost-usd",
            "0.00",
            "--outcome",
            FAILED_DISPATCH,
            "--returncode",
            "1",
            "--posterior-path",
            str(posterior),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    anchors = load_anchors(posterior)
    assert len(anchors) == 1
    assert anchors[0].platform == "github"
    assert anchors[0].gpu == "cpu"
    assert anchors[0].outcome == FAILED_DISPATCH
    assert anchors[0].returncode == 1


def test_append_cost_band_anchor_cli_has_no_manual_sys_path_mutation() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    text = (repo_root / "tools" / "append_cost_band_anchor.py").read_text()
    assert "tools.tool_bootstrap" in text
    assert "sys.path.insert" not in text


# -- NV7: anchor-outcome discipline (review-omni 2026-05-12) -------------------
#
# Bug class: failed-dispatch anchors (returncode=1 in 14-72 seconds) were
# folded into the percentile band, underestimating real cost by 400-750x.
# The Modal A100 anchor `fc-01KREXK209TRX7ED5ZRVXHY1VT` (14.77 sec rc=1 from
# WWW4) made the band predict "weak_posterior $0.016" instead of the
# hand-calibrated stub $3-8. predict() now excludes outcome != successful_dispatch
# by default; include_failed=True opt-in restores the legacy behavior.

import pytest  # noqa: E402

from tac.cost_band_calibration import (  # noqa: E402
    FAILED_DISPATCH,
    HARVESTED_PARTIAL,
    SUCCESSFUL_DISPATCH,
    TIMED_OUT,
    VALID_OUTCOMES,
    append_platform_training_anchor,
)


def _failed_anchor(*, label: str = "F", cost: float = 0.016, rc: int = 1) -> CostBandAnchor:
    """A failed-dispatch anchor mirroring the WWW4 fc-01KREXK209... shape."""
    return CostBandAnchor(
        logged_at_utc="2026-05-12T17:15:30+00:00",
        dispatch_label=label,
        trainer="experiments/train_substrate_sane_hnerv.py",
        platform="modal",
        gpu="A100",
        epochs=2000,
        batch_size=32,
        all_flags_on=True,
        actual_wall_clock_sec=14.77,  # the WWW4 crash wall-clock
        actual_cost_usd=cost,
        outcome=FAILED_DISPATCH,
        returncode=rc,
        notes="WWW4 fc-01KREXK209TRX7ED5ZRVXHY1VT regression fixture",
    )


def test_nv7_failed_anchors_default_excluded_from_predict(tmp_path: Path) -> None:
    """Posterior with ONLY failed anchors → falls back to hand-calibrated stub."""
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    append_anchor(_failed_anchor(label="F1"), posterior_path=pp, lock_path=lp)
    append_anchor(_failed_anchor(label="F2"), posterior_path=pp, lock_path=lp)
    append_anchor(_failed_anchor(label="F3"), posterior_path=pp, lock_path=lp)
    # 3 anchors total in file BUT all failed -> predict ignores them.
    p = predict("modal", "A100", 2000, all_flags_on=True, posterior_path=pp)
    assert p.confidence_tag == "hand_calibrated_fallback"
    assert p.n_anchors == 0
    # And the hand-calibrated stub at (modal, A100, 3000, True) returns ~$5 p50.
    assert p.p50_cost_usd > 0.5  # NOT $0.016 from the failed anchors


def test_nv7_predict_include_failed_true_folds_them_in(tmp_path: Path) -> None:
    """Explicit include_failed=True opt-in reproduces the legacy behavior."""
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    for i in range(3):
        append_anchor(_failed_anchor(label=f"F{i}", cost=0.016), posterior_path=pp, lock_path=lp)
    p = predict(
        "modal", "A100", 2000,
        all_flags_on=True,
        posterior_path=pp,
        include_failed=True,
    )
    assert p.confidence_tag == "empirical_posterior"
    assert p.n_anchors == 3
    assert p.p50_cost_usd == 0.016  # legacy poisoning visible when opt-in


def test_nv7_predict_mixes_successful_and_failed(tmp_path: Path) -> None:
    """3 successful at $5 + 2 failed at $0.016 -> predict uses only the 3 successful."""
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    for i in range(3):
        append_anchor(
            CostBandAnchor(
                logged_at_utc=f"2026-05-12T18:0{i}:00+00:00",
                dispatch_label=f"S{i}",
                trainer="experiments/train_substrate_sane_hnerv.py",
                platform="modal", gpu="A100", epochs=2000, batch_size=32,
                all_flags_on=True,
                actual_wall_clock_sec=4500.0,
                actual_cost_usd=5.0,
                outcome=SUCCESSFUL_DISPATCH,
                returncode=0,
            ),
            posterior_path=pp, lock_path=lp,
        )
    for i in range(2):
        append_anchor(_failed_anchor(label=f"F{i}", cost=0.016), posterior_path=pp, lock_path=lp)
    p = predict("modal", "A100", 2000, all_flags_on=True, posterior_path=pp)
    assert p.confidence_tag == "empirical_posterior"
    assert p.n_anchors == 3
    assert p.p50_cost_usd == 5.0  # NOT poisoned by the 2 failed at $0.016


def test_nv7_anchor_with_no_outcome_field_defaults_to_successful(tmp_path: Path) -> None:
    """Pre-NV7 anchors (no outcome field) load as successful_dispatch."""
    pp = tmp_path / "posterior.jsonl"
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(
        json.dumps({
            "schema": SCHEMA_VERSION,
            "logged_at_utc": "2026-05-12T18:00:00+00:00",
            "dispatch_label": "pre_nv7_legacy",
            "trainer": "experiments/x.py", "platform": "modal", "gpu": "T4",
            "epochs": 3000, "batch_size": 32, "all_flags_on": True,
            "actual_wall_clock_sec": 100.0, "actual_cost_usd": 5.0,
            # NO "outcome" field
        }) + "\n",
        encoding="utf-8",
    )
    anchors = load_anchors(pp)
    assert len(anchors) == 1
    assert anchors[0].outcome == SUCCESSFUL_DISPATCH
    assert anchors[0].returncode is None


def test_nv7_migration_tool_is_dry_run_by_default_and_apply_tags_failed(tmp_path: Path) -> None:
    """Historical rows with returncode=1 are migrated only under explicit --apply."""
    repo_root = Path(__file__).resolve().parents[3]
    posterior = tmp_path / "posterior.jsonl"
    lock = tmp_path / "posterior.lock"
    legacy_row = {
        "schema": SCHEMA_VERSION,
        "logged_at_utc": "2026-05-12T18:00:00+00:00",
        "dispatch_label": "legacy_failed",
        "trainer": "experiments/x.py",
        "platform": "modal",
        "gpu": "A100",
        "epochs": 2000,
        "batch_size": 32,
        "all_flags_on": True,
        "actual_wall_clock_sec": 14.77,
        "actual_cost_usd": 0.016,
        "notes": "cost_estimate_source=modal; returncode=1; timed_out=False",
    }
    posterior.write_text(json.dumps(legacy_row, sort_keys=True) + "\n", encoding="utf-8")
    before = posterior.read_text(encoding="utf-8")
    tool = repo_root / "tools" / "migrate_cost_band_posterior_failed_anchors.py"

    dry = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--posterior-path",
            str(posterior),
            "--lock-path",
            str(lock),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    dry_summary = json.loads(dry.stdout)
    assert dry_summary["dry_run"] is True
    assert dry_summary["migrated"] == 1
    assert posterior.read_text(encoding="utf-8") == before

    applied = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--apply",
            "--posterior-path",
            str(posterior),
            "--lock-path",
            str(lock),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr
    applied_summary = json.loads(applied.stdout)
    assert applied_summary["migrated"] == 1
    assert Path(applied_summary["backup_path"]).is_file()
    migrated = json.loads(posterior.read_text(encoding="utf-8"))
    assert migrated["outcome"] == FAILED_DISPATCH
    assert migrated["returncode"] == 1


def test_nv7_anchor_with_unknown_outcome_skipped_as_malformed(tmp_path: Path) -> None:
    """A row with outcome='something_weird' is skipped, not silent-coerced."""
    pp = tmp_path / "posterior.jsonl"
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(
        json.dumps({
            "schema": SCHEMA_VERSION,
            "logged_at_utc": "2026-05-12T18:00:00+00:00",
            "dispatch_label": "bad", "trainer": "x", "platform": "modal", "gpu": "T4",
            "epochs": 3000, "batch_size": 32, "all_flags_on": True,
            "actual_wall_clock_sec": 100.0, "actual_cost_usd": 5.0,
            "outcome": "definitely_not_a_valid_outcome",
        }) + "\n",
        encoding="utf-8",
    )
    assert load_anchors(pp) == []


def test_nv7_append_rejects_invalid_outcome() -> None:
    """append_anchor() refuses CostBandAnchor with outcome not in VALID_OUTCOMES."""
    bad_anchor = CostBandAnchor(
        logged_at_utc="2026-05-12T18:00:00+00:00",
        dispatch_label="bad", trainer="x", platform="modal", gpu="T4",
        epochs=3000, batch_size=32, all_flags_on=True,
        actual_wall_clock_sec=100.0, actual_cost_usd=5.0,
        outcome="not_in_the_enum",
    )
    with pytest.raises(ValueError, match="NV7"):
        append_anchor(bad_anchor)


def test_nv7_outcome_field_serialized_to_jsonl(tmp_path: Path) -> None:
    """Round-trip preserves outcome + returncode on the JSONL line."""
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    append_anchor(_failed_anchor(label="F", cost=0.016, rc=137), posterior_path=pp, lock_path=lp)
    text = pp.read_text(encoding="utf-8")
    parsed = json.loads(text.splitlines()[0])
    assert parsed["outcome"] == FAILED_DISPATCH
    assert parsed["returncode"] == 137
    # And it loads back faithfully.
    loaded = load_anchors(pp)
    assert loaded[0].outcome == FAILED_DISPATCH
    assert loaded[0].returncode == 137


def test_nv7_valid_outcomes_constant_is_frozen_set() -> None:
    """VALID_OUTCOMES is the source of truth — must contain the four canonical values."""
    assert SUCCESSFUL_DISPATCH in VALID_OUTCOMES
    assert FAILED_DISPATCH in VALID_OUTCOMES
    assert TIMED_OUT in VALID_OUTCOMES
    assert HARVESTED_PARTIAL in VALID_OUTCOMES
    assert isinstance(VALID_OUTCOMES, frozenset)


def test_nv7_append_platform_training_anchor_derives_outcome_from_rc(tmp_path: Path) -> None:
    """append_platform_training_anchor() now derives outcome from result['returncode']."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    metadata = {
        "label": "rc0_smoke",
        "gpu": "A100",
        "cost_band_anchor": {
            "trainer": "experiments/train_substrate_sane_hnerv.py",
            "epochs": 2000,
            "batch_size": 32,
            "all_flags_on": True,
        },
    }
    # Success case: rc=0 -> successful_dispatch
    manifest = append_platform_training_anchor(
        "modal",
        out_dir=out_dir,
        metadata=metadata,
        result={"returncode": 0, "elapsed_seconds": 3600.0, "timed_out": False},
        posterior_path=pp,
        lock_path=lp,
    )
    assert manifest["outcome"] == SUCCESSFUL_DISPATCH
    assert manifest["returncode"] == 0
    loaded = load_anchors(pp)
    assert loaded[0].outcome == SUCCESSFUL_DISPATCH


def test_nv7_append_platform_training_anchor_rc_nonzero_is_failed(tmp_path: Path) -> None:
    """rc=1 -> failed_dispatch (mirrors the WWW4 fc-01KREXK209... rc=1 case)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    metadata = {
        "label": "www4_regression_fixture",
        "gpu": "A100",
        "cost_band_anchor": {
            "trainer": "experiments/train_substrate_sane_hnerv.py",
            "epochs": 2000,
            "batch_size": 32,
            "all_flags_on": True,
        },
    }
    manifest = append_platform_training_anchor(
        "modal",
        out_dir=out_dir,
        metadata=metadata,
        result={"returncode": 1, "elapsed_seconds": 14.77, "timed_out": False},
        posterior_path=pp,
        lock_path=lp,
    )
    assert manifest["outcome"] == FAILED_DISPATCH
    assert manifest["returncode"] == 1
    # And predict() ignores it.
    p = predict("modal", "A100", 2000, all_flags_on=True, posterior_path=pp)
    assert p.confidence_tag == "hand_calibrated_fallback"
    assert p.n_anchors == 0


def test_nv7_append_platform_training_anchor_timed_out_is_distinct(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    metadata = {
        "label": "timeout_fixture",
        "gpu": "A100",
        "cost_band_anchor": {
            "trainer": "experiments/train_x.py",
            "epochs": 2000,
            "batch_size": 32,
            "all_flags_on": True,
        },
    }
    manifest = append_platform_training_anchor(
        "modal",
        out_dir=out_dir,
        metadata=metadata,
        result={"returncode": -9, "elapsed_seconds": 14400.0, "timed_out": True},
        posterior_path=pp,
        lock_path=lp,
    )
    assert manifest["outcome"] == TIMED_OUT
    assert manifest["returncode"] == -9


def test_nv7_summary_by_bucket_excludes_failed_by_default(tmp_path: Path) -> None:
    """summary_by_bucket() mirrors predict() — failed anchors are surfaced as n_failed."""
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    # 2 successful + 1 failed in the same bucket.
    for c in [5.0, 7.0]:
        append_anchor(_make_anchor(cost=c), posterior_path=pp, lock_path=lp)
    append_anchor(
        CostBandAnchor(
            logged_at_utc="2026-05-12T18:00:00+00:00",
            dispatch_label="F", trainer="experiments/train_x.py",
            platform="modal", gpu="T4", epochs=3000, batch_size=32,
            all_flags_on=True, actual_wall_clock_sec=14.0, actual_cost_usd=0.01,
            outcome=FAILED_DISPATCH, returncode=1,
        ),
        posterior_path=pp, lock_path=lp,
    )
    out = summary_by_bucket(pp)
    assert len(out) == 1
    row = out[0]
    assert row["n_anchors"] == 2  # only successful
    assert row["n_failed"] == 1
    assert row["p50_cost_usd"] == 6.0  # (5+7)/2
    assert row["min_cost_usd"] == 5.0


def test_nv7_weak_posterior_warning_when_only_two_successful(tmp_path: Path) -> None:
    """The Phase B-1 canary case: 2 successful + 1 failed = weak_posterior on 2 successful."""
    pp = tmp_path / "posterior.jsonl"
    lp = tmp_path / "lock"
    for c in [4.0, 6.0]:
        append_anchor(
            CostBandAnchor(
                logged_at_utc="2026-05-12T18:00:00+00:00",
                dispatch_label=f"S{c}", trainer="experiments/x.py",
                platform="modal", gpu="A100", epochs=2000, batch_size=32,
                all_flags_on=True, actual_wall_clock_sec=3600.0, actual_cost_usd=c,
                outcome=SUCCESSFUL_DISPATCH, returncode=0,
            ),
            posterior_path=pp, lock_path=lp,
        )
    append_anchor(_failed_anchor(label="F", cost=0.016), posterior_path=pp, lock_path=lp)
    p = predict("modal", "A100", 2000, all_flags_on=True, posterior_path=pp)
    assert p.confidence_tag == "weak_posterior"
    assert p.n_anchors == 2  # 2 successful, 1 failed-excluded
    # p50 of [4.0, 6.0] = 5.0
    assert p.p50_cost_usd == 5.0
