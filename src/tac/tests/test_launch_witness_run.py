# SPDX-License-Identifier: MIT
"""Tests for tools/launch_witness_run.py — the canonical ONE-COMMAND witness
launcher (operator 2026-06-30 "automatically in the future" + the build-the-
automated-value-generator guiding principle).

Covers: never-invent-a-flag validation, the script-based launch.sh (no word-split
fragility), perf-env verification parsing, and the --dry-run safe path. NO real
spawn, NO GPU — --dry-run + unit helpers only (the live n600 run is untouched)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(_REPO / rel))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


lw = _load("launch_witness_run_under_test", "tools/launch_witness_run.py")
rld = pytest.importorskip("render_levelset_dashboard")
from tac import witness_autoconfig as wac  # noqa: E402

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _cfg():
    return wac.derive_config(_GT, num_pairs=600, overfit=True, epochs=1000)


# ───────────────────────── never-invent-a-flag ─────────────────────────
def test_real_trainer_flags_nonempty_and_has_known():
    flags = lw.real_trainer_flags()
    assert "--out-dir" in flags and "--curriculum" in flags and "--muon-start-epoch" in flags
    assert len(flags) > 30


def test_real_trainer_flags_includes_boolean_optional_negations_only():
    """(CLASS-fix 2026-07-07) the DSL merge renders a False override as --no-<flag>; the
    validator must accept the negation for BooleanOptionalAction flags and STILL refuse it
    for store_true flags (whose --no- form is a genuine invention)."""
    flags = lw.real_trainer_flags()
    # BooleanOptionalAction => negation is a REAL argparse token
    assert "--lane-prior-phi1" in flags and "--no-lane-prior-phi1" in flags
    assert "--no-seed-islands" in flags
    # store_true flags gain NO negation (e.g. --chroma / --siren-init are store_true)
    from tac.witness_dsl.curriculum_dsl import real_store_true_flags
    st = real_store_true_flags()
    assert st, "expected at least one store_true flag in the trainer"
    for f in sorted(st)[:5]:
        assert f.replace("--", "--no-", 1) not in flags, f"store_true {f} must not gain --no- form"


def test_mod32_control_base_plus_islands_treatment_validates(tmp_path):
    """The islands PROCEED-class treatment arm (FEED-07c/07d): proven_base +
    Mod32SegOnlyControlBase + the treatment levers validates clean through the launcher's
    never-invent-a-flag guard (including the --no-lane-prior-phi1 negation)."""
    import dataclasses as dc
    cfg = dc.replace(_cfg(), dsl_levers=(
        "Mod32SegOnlyControlBase", "SeedIslandBirth", "SeedIslandEased", "AmplifyIsland",
        "EventTriggeredCurriculum", "MuonWarmStart", "SegFocalGamma"))
    ok, results = lw.validate_emitted_flags(cfg, "out")
    assert ok is True, [f for f, passed in results if not passed]
    d = dict(cfg.to_trainer_flags("out"))
    assert d["--mod-dim"] == 32 and d["--verdict-pairs"] == 0 and d["--eikonal-weight"] == 0.0
    assert d["--seg-focal-gamma"] == 2.0 and d["--amplify-weight"] == 1.0
    assert "--no-lane-prior-phi1" in d and "--lane-prior-phi1" not in d


def test_validate_emitted_flags_all_pass():
    ok, results = lw.validate_emitted_flags(_cfg(), "out")
    assert ok is True
    assert all(passed for _, passed in results)
    assert len(results) > 40


def test_validate_detects_invented_flag(monkeypatch):
    # simulate the trainer argparse missing every flag -> all emitted are "invented"
    monkeypatch.setattr(lw, "real_trainer_flags", lambda: frozenset())
    ok, results = lw.validate_emitted_flags(_cfg(), "out")
    assert ok is False
    assert any(not passed for _, passed in results)


# ───────────────────────── launch.sh (no word-split fragility) ─────────────────────────
def test_build_launch_sh_structure():
    body = lw.build_launch_sh(_cfg(), "out", repo_root=Path("/repo"))
    assert body.startswith("#!/bin/bash\nset -euo pipefail\ncd /repo\n")
    assert "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in body  # perf-env prefix present
    assert "train_levelset_witness_realized_through_R_mlx.py" in body
    assert "--curriculum" in body and "--stage-checkpoints" in body  # resumable


def test_write_launch_sh_roundtrips_through_schedule_parser(tmp_path):
    launch = lw.write_launch_sh(_cfg(), tmp_path)
    assert launch.exists() and os.access(launch, os.X_OK)
    # the parser (dashboard side) must recover the curriculum schedule from it
    cfg = rld.parse_run_config(tmp_path)
    assert cfg["source"] == "launch.sh"
    sched = cfg["schedule"]
    assert sched["tau_start"] is not None and sched["l7_start"] is not None
    assert sched["muon_start"] is not None and sched["epochs"] == 1000


# ───────────────────────── perf-env verification ─────────────────────────
def test_verify_perf_env_active(tmp_path):
    log = tmp_path / "run.log"
    log.write_text('{"stage": "custom_grouped_backward", "active": true, "note": "x"}\n')
    status, line = lw.verify_perf_env(log, timeout_s=1.0)
    assert status == "active" and line is not None


def test_verify_perf_env_inactive(tmp_path):
    log = tmp_path / "run.log"
    log.write_text('{"stage": "custom_grouped_backward", "active": false}\n')
    status, _ = lw.verify_perf_env(log, timeout_s=1.0)
    assert status == "inactive"


def test_verify_perf_env_not_seen(tmp_path):
    log = tmp_path / "run.log"
    log.write_text('{"stage": "gt", "n_pairs": 600}\n')
    status, line = lw.verify_perf_env(log, timeout_s=0.5, poll_s=0.1)
    assert status == "not_seen" and line is None


# ───────────────────────── dashboard ensure (down path; hermetic) ─────────────────────────
def test_ensure_dashboard_down_returns_false(capsys):
    # nothing is listening on this port -> False + an actionable warning
    assert lw.ensure_dashboard(59997) is False
    err = capsys.readouterr().err
    assert "NOT serving" in err and "dashboard_reload.py" in err


# ───────────────────────── main --dry-run (safe; no spawn) ─────────────────────────
def test_main_dry_run_writes_launch_sh_no_spawn(tmp_path, capsys):
    out = tmp_path / "run1"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--epochs", "1000",
                  "--out-dir", str(out), "--dry-run", "--no-dashboard"])
    assert rc == 0
    assert (out / "launch.sh").exists()
    assert not (out / "run.log").exists()  # DRY-RUN must NOT spawn
    out_txt = capsys.readouterr().out
    assert "DRY-RUN" in out_txt and "55/55" in out_txt or "flags exist" in out_txt


def test_main_refuses_invented_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lw, "real_trainer_flags", lambda: frozenset())
    out = tmp_path / "run2"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600",
                  "--out-dir", str(out), "--dry-run", "--no-dashboard"])
    assert rc == 2
    assert not (out / "launch.sh").exists()  # refused BEFORE writing
    err = capsys.readouterr().err
    assert "invented flag" in err


# ───────────────────────── --dsl-lever composability (CLASS-fix, review 2026-07-06) ─────────────
def test_main_dsl_lever_muon_clean_typed_refusal(tmp_path, capsys):
    # --dsl-lever Muon previously crashed the config generator with a raw TypeError AFTER the
    # launcher surface accepted it; now it is a clean one-line refusal BEFORE any gate/spawn work.
    out = tmp_path / "muon"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--epochs", "1000",
                  "--out-dir", str(out), "--dry-run", "--no-dashboard",
                  "--dsl-lever", "Muon"])
    assert rc == 2
    assert not (out / "launch.sh").exists()  # refused BEFORE writing anything
    err = capsys.readouterr().err
    assert "requires explicit args" in err and "composable" in err
    assert "Traceback" not in err


def test_main_dsl_lever_dm1minimal_clean_typed_refusal(tmp_path, capsys):
    # the composite half of the crash family (tuple[Lever, Lever] → AttributeError on .overrides)
    out = tmp_path / "dm1"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--epochs", "1000",
                  "--out-dir", str(out), "--dry-run", "--no-dashboard",
                  "--dsl-lever", "DM1Minimal"])
    assert rc == 2
    assert not (out / "launch.sh").exists()
    err = capsys.readouterr().err
    assert "returns tuple" in err and "composable" in err


def test_main_dsl_lever_composable_still_works_dry_run(tmp_path, capsys):
    # a genuinely composable lever still flows through the full dry-run path
    out = tmp_path / "seeded"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--epochs", "1000",
                  "--out-dir", str(out), "--dry-run", "--no-dashboard",
                  "--dsl-lever", "SeedIslandEased"])
    assert rc == 0
    body = (out / "launch.sh").read_text()
    assert "--seed-island-eased" in body
    assert "DSL levers composed: SeedIslandEased" in capsys.readouterr().out


def test_dsl_lever_help_enumeration_derived_from_predicate():
    # the --dsl-lever help text enumerates the DSL predicate's composable set (never hand-typed);
    # the crash-family names must NOT be advertised.
    from tac.witness_dsl.lever_registry import name_composable_levers
    names = lw._composable_lever_names()
    assert names == name_composable_levers()
    assert "Muon" not in names and "DM1Minimal" not in names


# ───────────────────────── C5 (SEAL review 2026-07-04): fresh_seeded + passthrough ─────────────
def test_main_dry_run_fresh_seeded_emits_the_review_deltas(tmp_path, capsys):
    out = tmp_path / "fresh"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--epochs", "1000",
                  "--config", "fresh_seeded", "--out-dir", str(out), "--dry-run", "--no-dashboard"])
    assert rc == 0
    body = (out / "launch.sh").read_text()
    for token in ("--lane-prior-phi1-mode paint", "--seed-islands", "--eikonal-weight 0.05",
                  "--eikonal-weight-end 0.1", "--tau-anneal-shape geometric",
                  "--softmax-temp-end 1.0", "--mod-dim 19", "--film-stiefel",
                  "--muon-warm-start-momentum", "--muon-lr-final-frac 0.1",
                  "--lane-band-start-epoch 350", "--stage-transition-rewarmup-epochs 20",
                  "--stage-transition-rewarmup-shape cosine", "--closed-loop-control",
                  "--l7-start-epoch 1001", "--hosc-beta-end 5.134", "--verdict-batch 64"):
        assert token in body, f"fresh_seeded launch.sh missing {token}"
    # the review's excluded levers must NOT appear (C1/C2/C3)
    assert "--curriculum-event-triggered" not in body
    assert "--bank-n-scales" not in body
    assert not (out / "run.log").exists()  # DRY-RUN must NOT spawn


def test_parse_extra_trainer_flags_validates_against_argparse():
    toks, invented = lw.parse_extra_trainer_flags("--eikonal-weight 0.07 --seed-islands")
    assert toks == ["--eikonal-weight", "0.07", "--seed-islands"]
    assert invented == []
    toks, invented = lw.parse_extra_trainer_flags("--totally-made-up-flag 1")
    assert invented == ["--totally-made-up-flag"]
    assert lw.parse_extra_trainer_flags(None) == ([], [])
    assert lw.parse_extra_trainer_flags("   ") == ([], [])


def test_main_extra_trainer_flags_appended_to_launch_sh(tmp_path, capsys):
    out = tmp_path / "extras"
    # --seed-islands / --seed-anneal-epochs are NOT emitted by proven_base -> no C13 duplicate;
    # appended verbatim. (A multi-token value so argparse treats it as a value, not an option.)
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--out-dir", str(out),
                  "--extra-trainer-flags", "--seed-islands --seed-anneal-epochs 300",
                  "--dry-run", "--no-dashboard"])
    assert rc == 0
    body = (out / "launch.sh").read_text()
    assert "--seed-islands" in body and "--seed-anneal-epochs 300" in body


# ───────────────────────── emit-side confound fixes (confound_hunt_synthesis_20260705.md) ─────────
def test_duplicate_long_flags_pure():
    assert lw.duplicate_long_flags(["--a", "--b", "--a", "--c", "--b"]) == ["--a", "--b"]
    assert lw.duplicate_long_flags(["--a", "--b", "--c"]) == []


def test_c13_refuses_duplicate_between_config_and_extra(tmp_path, capsys):
    # proven_base already emits --eikonal-weight; passing it again = the C13 last-wins schedule shift.
    out = tmp_path / "dup"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--out-dir", str(out),
                  "--extra-trainer-flags", "--eikonal-weight 0.07",
                  "--dry-run", "--no-dashboard"])
    assert rc == 2
    assert not (out / "launch.sh").exists()  # refused BEFORE writing
    err = capsys.readouterr().err
    assert "DUPLICATE long-flag" in err and "--eikonal-weight" in err


def test_fix4_injects_per_group_grad_clip_by_default(tmp_path):
    out = tmp_path / "pgc"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--out-dir", str(out),
                  "--config", "proven_base", "--dry-run", "--no-dashboard"])
    assert rc == 0
    assert "--per-group-grad-clip" in (out / "launch.sh").read_text()


def test_fix4_opt_out_no_per_group_grad_clip(tmp_path):
    out = tmp_path / "no_pgc"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--out-dir", str(out),
                  "--config", "proven_base", "--no-per-group-grad-clip",
                  "--dry-run", "--no-dashboard"])
    assert rc == 0
    assert "per-group-grad-clip" not in (out / "launch.sh").read_text()


def test_fix4_respects_user_supplied_polarity():
    # user already set the flag -> no injection (no C13 dup).
    ef, note = lw.inject_per_group_grad_clip(["--per-group-grad-clip"], [], enable=True)
    assert ef == ["--per-group-grad-clip"] and note is None
    ef, note = lw.inject_per_group_grad_clip(["--no-per-group-grad-clip"], [], enable=True)
    assert ef == ["--no-per-group-grad-clip"] and note is None


def test_c8_palliative_implies_warm_start():
    ef, note = lw.couple_palliative_warm_start(
        ["--resume-from", "ck_ep100.npz", "--resume-clear-spike-guard"], [])
    assert "--warm-start-weights-only" in ef and note is not None
    # already present -> no double-inject
    ef2, note2 = lw.couple_palliative_warm_start(
        ["--resume-allow-lever-drift", "--warm-start-weights-only"], [])
    assert ef2.count("--warm-start-weights-only") == 1 and note2 is None
    # no palliative -> untouched
    ef3, note3 = lw.couple_palliative_warm_start(["--resume-from", "ck.npz"], [])
    assert ef3 == ["--resume-from", "ck.npz"] and note3 is None


def test_c16_seed_anneal_relative_to_resume():
    # seed-islands from the CONFIG side + resume ep100 (start 101) + N=101 <= 101 -> corrected.
    ef, note = lw.seed_anneal_relative_to_resume(
        ["--resume-from", "stage_muon_ep100.npz", "--seed-anneal-epochs", "101"],
        ["--seed-islands"], anneal_window=200)
    assert lw._flag_value(ef, "--seed-anneal-epochs") == "301" and note is not None
    # window already extends past the resume epoch -> untouched.
    ef2, note2 = lw.seed_anneal_relative_to_resume(
        ["--resume-from", "stage_muon_ep100.npz", "--seed-anneal-epochs", "350"],
        ["--seed-islands"], anneal_window=200)
    assert lw._flag_value(ef2, "--seed-anneal-epochs") == "350" and note2 is None
    # no --seed-islands anywhere -> untouched (fresh run, no seed crutch).
    ef3, note3 = lw.seed_anneal_relative_to_resume(
        ["--resume-from", "ck_ep100.npz", "--seed-anneal-epochs", "50"], [])
    assert note3 is None and ef3[-1] == "50"


def test_apply_emit_side_fixes_end_to_end_dup_detected():
    # a fresh config that emits --eikonal-weight + a user dup -> reports the dup for refusal.
    ef, notes, dups = lw.apply_emit_side_confound_fixes(
        ["--eikonal-weight", "0.07"], ["--eikonal-weight", "--curriculum"],
        per_group_grad_clip=True)
    assert dups == ["--eikonal-weight"]


def test_main_extra_trainer_flags_invented_refused(tmp_path, capsys):
    out = tmp_path / "extras_bad"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--out-dir", str(out),
                  "--extra-trainer-flags", "--not-a-real-flag 3",
                  "--dry-run", "--no-dashboard"])
    assert rc == 2
    assert not (out / "launch.sh").exists()  # refused BEFORE writing
    assert "invented flag" in capsys.readouterr().err


def test_main_extra_bank6_is_caught_by_the_fixed_mem_preflight(tmp_path, monkeypatch, capsys):
    """C4+C5 composing end-to-end: a memory-relevant EXTRA flag (--bank-n-scales 6) lands in the
    emitted launch.sh, the FIXED in_feat-aware preflight projects the true 110.81 GiB and REFUSES
    (rc=4) — the exact FALSE-SAFE path the review executed, now closed."""
    import witness_memory_preflight as wmp
    monkeypatch.setattr(wmp, "_total_ram_gib", lambda: 128.0)  # pin RAM so the gate is deterministic
    out = tmp_path / "bank6"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--config", "sealed_205",
                  "--out-dir", str(out),
                  "--extra-trainer-flags", "--bank-n-scales 6",
                  "--dry-run", "--no-dashboard"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "REFUSING to launch" in err


# ───────────────── safe-frac policy derivation (L5/XC-ii, operator memory policy 2026-07-04) ────
def test_derive_safe_frac_explicit_cli_always_wins(monkeypatch):
    def _must_not_read():
        raise AssertionError("explicit CLI value must not consult governor state")
    monkeypatch.setattr(lw, "_governed_active_jobs", _must_not_read)
    frac, branch, why = lw.derive_safe_frac(0.6)
    assert frac == 0.6 and branch == "explicit"


def test_derive_safe_frac_single_workload_branch(monkeypatch):
    """No other governed heavy job admitted/running -> 0.85 (sole-workload; the branch that
    activates on this box AFTER the #205 preserve+stop)."""
    monkeypatch.setattr(lw, "_governed_active_jobs", lambda: [])
    frac, branch, why = lw.derive_safe_frac(None)
    assert frac == lw.SAFE_FRAC_SINGLE_WORKLOAD == 0.85
    assert branch == "single_workload" and "sole-workload" in why


def test_derive_safe_frac_concurrent_branch(monkeypatch):
    """A governed heavy job admitted/running -> 0.70 (the CORRECT branch on this box today,
    while #205 is live)."""
    monkeypatch.setattr(lw, "_governed_active_jobs",
                        lambda: [{"label": "levelset_witness_n600", "pid": 29129}])
    frac, branch, why = lw.derive_safe_frac(None)
    assert frac == lw.SAFE_FRAC_CONCURRENT == 0.70
    assert branch == "concurrent" and "levelset_witness_n600" in why


def test_derive_safe_frac_unreadable_state_falls_back_conservative(monkeypatch):
    def _boom():
        raise OSError("registry unreadable")
    monkeypatch.setattr(lw, "_governed_active_jobs", _boom)
    frac, branch, why = lw.derive_safe_frac(None)
    assert frac == 0.70 and branch == "fallback_conservative" and "unreadable" in why


def test_governed_active_jobs_readonly_registry_view(tmp_path, monkeypatch):
    """The read path itself: running rows with a LIVE pid AND a heavy signature are counted;
    dead-pid (stale), non-running, and telemetry/control-plane rows (no projection, cmd outside
    the governor's heavy vocabulary — e.g. memory_blackbox) are dropped, so a telemetry daemon
    can never pin the box at 0.70 after the real run stops. Pure read — never modified."""
    import json as _json

    import system_memory_governor as gov
    me = os.getpid()
    reg = tmp_path / "durable_daemons.json"
    rows = [
        # heavy via recorded governed projection
        {"label": "live_run", "pid": me, "status": "running", "projected_peak_gib": 67.6},
        # heavy via the governor's OUR_JOBS_PATTERN cmd vocabulary (no projection recorded)
        {"label": "legacy_trainer", "pid": me, "status": "running",
         "cmd": [".venv/bin/python", "experiments/train_levelset_witness_realized_through_R_mlx.py"]},
        # telemetry daemon: no projection, cmd outside the heavy vocabulary -> NOT a workload
        {"label": "memory_blackbox", "pid": me, "status": "running",
         "cmd": [".venv/bin/python", "tools/memory_blackbox.py"]},
        # sub-threshold projection -> NOT heavy
        {"label": "tiny_probe", "pid": me, "status": "running", "projected_peak_gib": 0.5},
        {"label": "stale_dead", "pid": 987654, "status": "running",   # ESRCH -> dropped
         "projected_peak_gib": 67.6},
        {"label": "finished", "pid": me, "status": "exited", "projected_peak_gib": 67.6},
    ]
    reg.write_text(_json.dumps(rows))
    before = reg.read_text()
    monkeypatch.setattr(gov, "_DURABLE_DAEMON_REGISTRY", reg)
    jobs = lw._governed_active_jobs()
    assert [j["label"] for j in jobs] == ["live_run", "legacy_trainer"]
    assert reg.read_text() == before  # READ-ONLY


def test_main_dry_run_prints_safe_frac_policy_branch(tmp_path, monkeypatch, capsys):
    """Observability: the launcher prints WHICH policy branch fired and why (both branches)."""
    out1 = tmp_path / "single"
    monkeypatch.setattr(lw, "_governed_active_jobs", lambda: [])
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--out-dir", str(out1),
                  "--dry-run", "--no-dashboard"])
    assert rc == 0
    assert "safe-frac 0.85 [single_workload]" in capsys.readouterr().out
    out2 = tmp_path / "concurrent"
    monkeypatch.setattr(lw, "_governed_active_jobs",
                        lambda: [{"label": "levelset_205", "pid": 29129}])
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--out-dir", str(out2),
                  "--dry-run", "--no-dashboard"])
    assert rc == 0
    assert "safe-frac 0.70 [concurrent]" in capsys.readouterr().out
