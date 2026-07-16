"""DELTA-BENCH efficiency lever (operator-directed 2026-07-16) — behavior tests.

The lever: ``tools/launch_witness_run.py --dry-start-delta-from <prior_run_dir>`` inherits the
transferable proofs (boot / memory envelope / throughput) from a prior GREEN
``full_config_dry_start`` receipt and runs a REDUCED 2-epoch bench (fresh boot + the
NEVER-transferable crash-resume round-trip + a peak-RSS envelope cross-check) — but ONLY when
the structural flag diff vs the prior run is non-empty and every differing flag is in
``SCORE_NEUTRAL_BENCH_INHERIT_WHITELIST``. Everything else REFUSES (rc=7, typed reason).
The default (no-flag) path must remain behavior-identical (regression tests below).
"""
import json
import pathlib
import subprocess
import sys
import types
import datetime as dt

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "tools"))

import launch_witness_run as L  # noqa: E402

CONFIG = "c2_surgical_warm"

BASE_ARGV = [
    ".venv/bin/python", "experiments/train_levelset_witness_realized_through_R_mlx.py",
    "--seed", "0", "--verdict-live-gap-every", "50", "--profile-timing",
    "--out-dir", "/runs/prior",
]


def _fresh_argv(**changes):
    """BASE_ARGV with flag-value substitutions (flag -> new value; None -> drop flag+value)."""
    argv = list(BASE_ARGV)
    # every fresh argv gets its own out-dir (run identity — excluded from the structural diff)
    argv[argv.index("--out-dir") + 1] = "/runs/fresh"
    for flag, val in changes.items():
        if flag in argv:
            i = argv.index(flag)
            if val is None:
                nxt = i + 1
                take = 2 if nxt < len(argv) and not argv[nxt].startswith("--") else 1
                del argv[i:i + take]
            else:
                argv[i + 1] = str(val)
        elif val is not None:
            argv += [flag] + ([str(val)] if val != "" else [])
    return argv


def _write_prior(tmp_path, *, green=True, gate="full_config_dry_start", config=CONFIG,
                 ts=None, peak=2.0, argv=None, receipt=True):
    prior = tmp_path / "prior_run"
    prior.mkdir(parents=True, exist_ok=True)
    if receipt:
        (prior / "dry_start_report.json").write_text(json.dumps({
            "gate": gate, "green": green, "config": config,
            "ts": ts if ts is not None else L._utc(),
            "peak_rss_gib": peak, "typed_config_hash": "priorhash123",
        }))
    (prior / "launch_manifest.json").write_text(json.dumps({
        "resolved_launch_argv": argv if argv is not None else BASE_ARGV,
    }))
    return prior


def _write_fresh_manifest(tmp_path, argv):
    out = tmp_path / "fresh_run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "launch_manifest.json").write_text(json.dumps({"resolved_launch_argv": argv}))
    return out


# ───────────────────────── eligibility (pure, fail-closed) ─────────────────────────

def test_whitelist_only_diff_accepted(tmp_path):
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, payload = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert ok, reason
    assert "--verdict-live-gap-every" in payload["delta_flags"]
    assert payload["delta_flags"]["--verdict-live-gap-every"]["prior"] == (("50",),)
    assert payload["delta_flags"]["--verdict-live-gap-every"]["fresh"] == (("25",),)
    assert payload["inherited_from"]["path"] == str(prior)
    assert payload["inherited_from"]["typed_config_hash"] == "priorhash123"
    assert payload["inherited_from"]["fields"] == ["boot_ok", "peak_rss_gib", "throughput_gate"]
    assert payload["inherited_peak_rss_gib"] == 2.0


def test_score_affecting_diff_refused_names_flags(tmp_path):
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(
        tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25", "--seed": "1"}))
    ok, reason, payload = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and payload == {}
    assert "--seed" in reason and "NON-whitelisted" in reason
    assert "run the full bench" in reason


def test_base_flag_added_refused(tmp_path):
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--muon-lr": "0.002"}))
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "--muon-lr" in reason


def test_whitelisted_flag_removed_accepted(tmp_path):
    # removal of a whitelisted telemetry flag is itself a whitelisted structural delta
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--profile-timing": None}))
    ok, reason, payload = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert ok, reason
    assert payload["delta_flags"]["--profile-timing"]["fresh"] is None


def test_empty_diff_refused_identical_config(tmp_path):
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(tmp_path, _fresh_argv())  # only --out-dir differs (excluded)
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok
    assert "identical config — reuse the prior receipt, no bench needed" in reason


def test_missing_prior_dir_refused(tmp_path):
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, _ = L.delta_bench_eligibility(
        tmp_path / "nope", out / "launch_manifest.json", CONFIG)
    assert not ok and "does not exist on disk" in reason


def test_missing_prior_receipt_refused(tmp_path):
    prior = _write_prior(tmp_path, receipt=False)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "not found" in reason and "run the full bench" in reason


def test_non_green_prior_refused(tmp_path):
    prior = _write_prior(tmp_path, green=False)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "not green" in reason


def test_wrong_gate_refused(tmp_path):
    prior = _write_prior(tmp_path, gate="some_other_gate")
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "full_config_dry_start" in reason


def test_config_mismatch_refused(tmp_path):
    prior = _write_prior(tmp_path, config="proven_base")
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "cross-config inheritance is forbidden" in reason


def test_stale_prior_refused(tmp_path):
    old_ts = (dt.datetime.now(dt.UTC) - dt.timedelta(days=15)).strftime("%Y%m%dT%H%M%SZ")
    prior = _write_prior(tmp_path, ts=old_ts)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "staleness guard" in reason


def test_unparseable_ts_refused_fail_closed(tmp_path):
    prior = _write_prior(tmp_path, ts="not-a-timestamp")
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "unparseable" in reason


def test_missing_prior_peak_refused(tmp_path):
    prior = _write_prior(tmp_path, peak=None)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "peak_rss_gib" in reason


def test_changed_trainer_script_is_non_whitelisted(tmp_path):
    prior = _write_prior(tmp_path)
    argv = _fresh_argv(**{"--verdict-live-gap-every": "25"})
    argv[1] = "experiments/other_trainer.py"
    out = _write_fresh_manifest(tmp_path, argv)
    ok, reason, _ = L.delta_bench_eligibility(prior, out / "launch_manifest.json", CONFIG)
    assert not ok and "_argv_head" in reason


# ───────────────────────── envelope cross-check (pure) ─────────────────────────

def test_envelope_ok_within_tolerance():
    ok, detail = L.delta_bench_envelope_ok(2.0, 2.15)
    assert ok and detail.startswith("OK")


def test_envelope_violated_over_tolerance():
    ok, detail = L.delta_bench_envelope_ok(2.0, 2.3)
    assert not ok and "inherited envelope violated" in detail


def test_envelope_fail_closed_on_unmeasured_fresh_peak():
    ok, detail = L.delta_bench_envelope_ok(2.0, None)
    assert not ok and "fail-closed" in detail


# ───────────────────── _run_dry_start end-to-end (stubbed passes) ─────────────────────

class _FakeCompleted:
    def __init__(self, stdout):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


_PASS1_LOG = (
    '{"stage":"gt","secs":10}\n{"ep":1}\n'
    '{"stage":"checkpoint","resume_latest":"levelset_resume_state.npz","epoch":1}\n'
    '{"ep":2}\n{"stage":"checkpoint","resume_latest":"levelset_resume_state.npz","epoch":2}\n'
)
_PASS2_LOG = (
    '{"stage":"resume_model_source"}\n'
    '{"resume_start_epoch":3,"resume_ckpt_epoch":2}\n{"ep":3}\n{"ep":4}\n'
)


def _stub_launcher(monkeypatch, pass_logs, peaks_mib):
    """Stub compile+launch machinery so _run_dry_start's pass loop runs hermetically.
    Returns the list of per-pass override dicts handed to with_internal_dsl_lever."""
    calls = {"i": 0}
    seen_overrides: list[dict] = []

    class _Cfg:
        typed = None

    def fake_derive(config, gt_cache, *, num_pairs, epochs, overfit):
        return _Cfg()

    def fake_lever(cfg, *, name, overrides):
        seen_overrides.append(dict(overrides))
        return cfg

    def fake_write(cfg, sub, **kw):
        pathlib.Path(sub).mkdir(parents=True, exist_ok=True)
        launch = pathlib.Path(sub) / "launch.sh"
        launch.write_text("#!/usr/bin/env bash\n")
        return launch, None, None, None

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        i = calls["i"]
        calls["i"] += 1
        return _FakeCompleted(pass_logs[i] + f"peak_rss={peaks_mib[i]}MiB\n")

    monkeypatch.setattr(L, "derive_named_config", fake_derive)
    monkeypatch.setattr(L, "with_internal_dsl_lever", fake_lever)
    monkeypatch.setattr(L, "write_dsl_bound_launch", fake_write)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen_overrides


def _args(**over):
    base = dict(dry_start=2, dry_start_boot_budget_s=1.0, dry_start_per_ep_budget_s=1.0,
                rss_cap_mb=1024, num_pairs=600, gt_cache="gt.npz",
                admission_override_rationale=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_delta_receipt_carries_mode_inherited_from_delta_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("TAC_BENCH_INHERIT_FROM", "sentinel")  # register key for teardown restore
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    _stub_launcher(monkeypatch, [_PASS1_LOG, _PASS2_LOG], [2048, 2048])
    rc = L._run_dry_start(_args(dry_start_delta_from=str(prior)), CONFIG, True, out, "t", None, None)
    assert rc == 0
    rep = json.loads((out / "dry_start_report.json").read_text())
    assert rep["gate"] == "full_config_dry_start"
    assert rep["mode"] == "delta_bench"
    assert rep["green"] is True and rep["boot_ok"] and rep["resume_round_trip_ok"]
    assert rep["peak_envelope_ok"] is True
    assert rep["inherited_from"]["path"] == str(prior)
    assert rep["inherited_from"]["typed_config_hash"] == "priorhash123"
    assert rep["inherited_from"]["fields"] == ["boot_ok", "peak_rss_gib", "throughput_gate"]
    assert "--verdict-live-gap-every" in rep["delta_flags"]
    assert rep["dry_start_target_epochs"] == L.DELTA_BENCH_EPOCHS
    # Boot-side inheritance stamp lives NEXT TO the inherited_from provenance fields.
    assert rep["boot_baseline_verdict"] == "inherited"
    # NO-FAKE: the note must say which fields are inherited provenance vs freshly measured
    assert "PROVENANCE" in rep["note"] and "MEASURED FRESH" in rep["note"]


def test_delta_green_false_on_envelope_violation(tmp_path, monkeypatch):
    prior = _write_prior(tmp_path, peak=1.0)  # inherited 1.0 GiB; fresh 2 GiB > 1.1 GiB limit
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    _stub_launcher(monkeypatch, [_PASS1_LOG, _PASS2_LOG], [2048, 2048])
    rc = L._run_dry_start(_args(dry_start_delta_from=str(prior)), CONFIG, True, out, "t", None, None)
    assert rc == 6
    rep = json.loads((out / "dry_start_report.json").read_text())
    assert rep["green"] is False
    assert rep["peak_envelope_ok"] is False
    assert rep["green_false_reason"] == "inherited envelope violated"
    assert rep["boot_ok"] and rep["resume_round_trip_ok"]  # fresh proofs passed; envelope failed


def test_delta_green_false_on_resume_failure(tmp_path, monkeypatch):
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    _stub_launcher(monkeypatch, [_PASS1_LOG, '{"ep":3}\n'], [2048, 2048])  # no resume rows
    rc = L._run_dry_start(_args(dry_start_delta_from=str(prior)), CONFIG, True, out, "t", None, None)
    assert rc == 6
    rep = json.loads((out / "dry_start_report.json").read_text())
    assert rep["green"] is False and rep["resume_round_trip_ok"] is False
    assert rep["mode"] == "delta_bench"


def test_delta_refusal_exits_rc7_before_any_pass(tmp_path, monkeypatch, capsys):
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(
        tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25", "--seed": "1"}))

    def boom(*a, **k):  # no pass may run on a refused delta
        raise AssertionError("a bench pass ran despite the refusal")
    monkeypatch.setattr(subprocess, "run", boom)
    rc = L._run_dry_start(_args(dry_start_delta_from=str(prior)), CONFIG, True, out, "t", None, None)
    assert rc == 7
    err = capsys.readouterr().err
    assert "REFUSING delta bench" in err and "--seed" in err


# ───────────────────── default (no-flag) path regression ─────────────────────

def test_default_path_receipt_shape_unchanged(tmp_path, monkeypatch):
    """No --dry-start-delta-from (attr absent entirely): the receipt must carry EXACTLY the
    pre-lever key set — no mode/inherited_from/delta_flags — and green == boot_ok and resume_ok."""
    out = tmp_path / "fresh_run"
    out.mkdir()
    _stub_launcher(monkeypatch, [_PASS1_LOG, _PASS2_LOG], [2048, 2048])
    rc = L._run_dry_start(_args(), CONFIG, True, out, "t", None, None)
    assert rc == 0
    rep = json.loads((out / "dry_start_report.json").read_text())
    assert set(rep) == {
        "gate", "owed", "typed_config_hash", "config", "num_pairs",
        "dry_start_target_epochs", "pass_timeout_s", "boot_ok", "resume_round_trip_ok",
        "green", "peak_rss_gib", "sec_per_ep_gross", "sec_per_ep_marginal",
        "pass1", "pass2", "note", "ts",
    }
    assert rep["green"] is True


def test_default_path_ignores_delta_helpers_and_bound_check(tmp_path, monkeypatch, capsys):
    """Default path keeps the 1..3 bound refusal (rc=2) byte-for-byte."""
    out = tmp_path / "fresh_run"
    out.mkdir()
    rc = L._run_dry_start(_args(dry_start=5), CONFIG, True, out, "t", None, None)
    assert rc == 2
    assert "--dry-start must be in 1..3" in capsys.readouterr().err


def test_parser_declares_delta_flag_default_none_and_implies_dry_start():
    src = pathlib.Path(L.__file__).read_text()
    assert '"--dry-start-delta-from", default=None' in src
    assert "args.dry_start_delta_from and not args.dry_start" in src
    assert "args.dry_start = DELTA_BENCH_EPOCHS" in src


def test_whitelist_is_exactly_the_operator_directed_trio():
    assert L.SCORE_NEUTRAL_BENCH_INHERIT_WHITELIST == (
        "--verdict-live-gap-every",
        "--component-wallclock-probe-every",
        "--profile-timing",
    )


def test_out_dir_is_excluded_from_structural_diff():
    prior = L.parse_argv_flags(["p", "t.py", "--out-dir", "/a", "--seed", "0"])
    fresh = L.parse_argv_flags(["p", "t.py", "--out-dir", "/b", "--seed", "0"])
    assert L.structural_flag_diff(prior, fresh) == {}


def test_resume_from_is_structural_not_run_identity():
    # a changed warm-start SOURCE is config structure => non-whitelisted => refusal territory
    prior = L.parse_argv_flags(["p", "t.py", "--resume-from", "a.npz"])
    fresh = L.parse_argv_flags(["p", "t.py", "--resume-from", "b.npz"])
    assert "--resume-from" in L.structural_flag_diff(prior, fresh)


# ───────────── boot-side inheritance: --skip-boot-baseline-verdict (follow-up directive) ─────────────

_TRAINER_SRC = (_REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py").read_text()


def test_delta_passes_carry_skip_boot_baseline_verdict_and_env(tmp_path, monkeypatch):
    """BOTH delta passes get the bare skip flag at the bench-pass argv layer; the inherit-from
    env names the prior receipt's typed hash for the trainer's honest skipped row."""
    monkeypatch.setenv("TAC_BENCH_INHERIT_FROM", "sentinel")  # register key for teardown restore
    prior = _write_prior(tmp_path)
    out = _write_fresh_manifest(tmp_path, _fresh_argv(**{"--verdict-live-gap-every": "25"}))
    seen = _stub_launcher(monkeypatch, [_PASS1_LOG, _PASS2_LOG], [2048, 2048])
    rc = L._run_dry_start(_args(dry_start_delta_from=str(prior)), CONFIG, True, out, "t", None, None)
    assert rc == 0
    assert len(seen) == 2  # pass1 + pass2
    for ov in seen:
        assert ov.get("--skip-boot-baseline-verdict") is True
    import os as _os
    assert _os.environ["TAC_BENCH_INHERIT_FROM"] == "priorhash123"


def test_full_bench_passes_do_not_carry_skip_flag(tmp_path, monkeypatch):
    """Default (no --dry-start-delta-from): NEITHER pass carries the skip flag — the full bench
    always measures its own boot baseline."""
    out = tmp_path / "fresh_run"
    out.mkdir()
    seen = _stub_launcher(monkeypatch, [_PASS1_LOG, _PASS2_LOG], [2048, 2048])
    rc = L._run_dry_start(_args(), CONFIG, True, out, "t", None, None)
    assert rc == 0
    assert len(seen) == 2
    for ov in seen:
        assert "--skip-boot-baseline-verdict" not in ov
        assert ov.get("--ckpt-every") == 1  # the pre-existing override machinery is intact


def test_real_launch_argv_never_carries_skip_flag():
    """The REAL c2_surgical_warm launch argv (typed compile, no bench lever) must not contain the
    skip flag — it exists only on the launcher-owned bench-pass lever."""
    cfg = L.derive_named_config(
        "c2_surgical_warm", "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600, epochs=None, overfit=True)
    flags = list(cfg.to_trainer_flags("REAL_LAUNCH_ARGV_AUDIT"))
    tokens = {t for pair in flags for t in (pair if isinstance(pair, (list, tuple)) else [pair])}
    assert "--skip-boot-baseline-verdict" not in tokens
    assert "--skip-boot-baseline-verdict" not in cfg.to_command("REAL_LAUNCH_ARGV_AUDIT")


def test_typed_config_hash_invariance_c2_surgical_warm():
    """The boot-side inheritance feature must not perturb the typed config surface: the c2 hash
    is pinned to its pre-feature value (coordinator amendment 2d486e3bff...)."""
    # NOTE: the gt-cache path STRING is part of the typed surface — the canonical launcher
    # invocation uses the repo-relative path, which is what the pinned hash was compiled with.
    cfg = L.derive_named_config(
        "c2_surgical_warm", "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600, epochs=None, overfit=True)
    assert cfg.typed.typed_config_hash() == (
        "2d486e3bff935949c4018a9b998b621cb439e8000640772aa8abd23dd21d5a8d")


def test_trainer_flag_is_store_true_default_off():
    """Default-off regression at the trainer surface: store_true (no value token, default False)
    and the v0 gate reads it with a False default (absent-attr safe)."""
    assert '"--skip-boot-baseline-verdict", action="store_true"' in _TRAINER_SRC
    assert 'getattr(args, "skip_boot_baseline_verdict", False)' in _TRAINER_SRC


def test_trainer_skip_emits_honest_skipped_row():
    """The skip path must emit the baseline_verdict_skipped row naming the inherited receipt —
    the run record never looks like a verdict silently vanished."""
    assert '"stage": "baseline_verdict_skipped"' in _TRAINER_SRC
    assert '"delta_bench_inherited_from "' in _TRAINER_SRC
    assert 'os.environ.get("TAC_BENCH_INHERIT_FROM", "unspecified")' in _TRAINER_SRC


def test_skip_flag_is_a_legal_boolean_lever_override():
    """with_internal_dsl_lever compiles a True override to a bare token ONLY for flags the DSL's
    trainer-argparse scan classifies as boolean — the new flag must be in that set."""
    from tac.witness_dsl.curriculum_dsl import real_boolean_flags
    assert "--skip-boot-baseline-verdict" in real_boolean_flags()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
