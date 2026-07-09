"""Tests for tac.witness_autoconfig — the clip -> witness_config actuator.

These tests verify BEHAVIOR (the generators actually compute / route), not just
constants: the flag-validation test parses the real trainer argparse so a config
that emitted an invented flag would FAIL; the intrinsic-dim test recovers a known
manifold dimension from real data. Pure CPU / numpy; no GPU, no heavy I/O.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tac import witness_autoconfig as wac

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"
_GT_N600 = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _real_trainer_flags() -> frozenset[str]:
    text = _TRAINER.read_text()
    return frozenset(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))


# --------------------------------------------------------------------------
# Whitney embedding generator
# --------------------------------------------------------------------------
def test_whitney_clamp_m9_to_19():
    assert wac.whitney_mod_dim(9) == 19  # 2*9+1 = 19, in band


def test_whitney_clamp_m12_to_25():
    assert wac.whitney_mod_dim(12) == 25  # 2*12+1 = 25


def test_whitney_clamp_m20_ceiling_26():
    assert wac.whitney_mod_dim(20) == 26  # 2*20+1 = 41 -> clamp to 26


def test_whitney_clamp_low_floor_19():
    assert wac.whitney_mod_dim(5) == 19  # 2*5+1 = 11 -> clamp up to 19


# --------------------------------------------------------------------------
# intrinsic-dim generator: MEASURES from data, FALLS BACK when absent (NO-FAKE)
# --------------------------------------------------------------------------
def test_intrinsic_dim_fallback_flagged_when_absent():
    pv = wac.intrinsic_dim(None)
    assert pv.is_fallback
    assert pv.source == wac.SRC_FALLBACK
    assert pv.value == pytest.approx(9.0)


def test_intrinsic_dim_measures_known_low_dim_manifold():
    # a 2-D plane linearly embedded in 12-D -> intrinsic dim ~2, NOT fallback.
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((300, 2))
    basis = rng.standard_normal((2, 12))
    with np.errstate(all="ignore"):  # spurious Accelerate matmul FP-flag on macOS
        X = latent @ basis + 1e-3 * rng.standard_normal((300, 12))
    pv = wac.intrinsic_dim(X)
    assert pv.source == wac.SRC_MEASURED
    assert not pv.is_fallback
    assert 1.3 < float(pv.value) < 3.5  # ~2


# --------------------------------------------------------------------------
# mod / hidden / muon-lr / verdict-pairs exact values (the dogfood revisions)
# --------------------------------------------------------------------------
def test_mod_dim_overfit_ships_26():
    assert wac.mod_dim_generator(None, overfit=True).value == 26


def test_mod_dim_aggressive_uses_whitney_floor():
    # overfit=False with fallback m=9 -> Whitney floor 19 (aggressive theta*).
    assert wac.mod_dim_generator(None, overfit=False).value == 19


def test_hidden_dim_is_96_not_120():
    assert wac.hidden_dim_generator(None).value == 96


def test_hidden_dim_picks_rd_min_when_sweep_supplied():
    pv = wac.hidden_dim_generator({96: 90621, 120: 111902, 128: 161000})
    assert pv.value == 96
    assert pv.source == wac.SRC_MEASURED


def test_muon_lr_is_proven_0p002():
    assert wac.muon_lr_generator().value == pytest.approx(0.002)


def test_verdict_pairs_is_96():
    assert wac.verdict_pairs_generator(600).value == 96


# --------------------------------------------------------------------------
# curriculum schedule generator
# --------------------------------------------------------------------------
def test_curriculum_schedule_proven_1000():
    s = wac.curriculum_schedule(1000)
    assert s["tau_softplus_start_epoch"].value == 300
    assert s["l7_start_epoch"].value == 600
    assert s["muon_start_epoch"].value == 726


def test_curriculum_schedule_scales_proportionally():
    s = wac.curriculum_schedule(2000)
    assert s["tau_softplus_start_epoch"].value == 600
    assert s["l7_start_epoch"].value == 1200
    assert s["muon_start_epoch"].value == 1452


# --------------------------------------------------------------------------
# lever priors (attribution-clean) + portability split
# --------------------------------------------------------------------------
def test_lever_priors_attribution_clean_first():
    lp = wac.lever_priors()
    assert lp["surgical_levers_enabled"] is False
    assert lp["dm1_enabled"] is False
    assert "margin_saliency" in lp["deferred_levers"]
    assert "lane_prior_phi1" in lp["active_geometric_priors"]


def test_portability_split_has_all_three_classes():
    p = wac.portability_split()
    vals = set(p.values())
    assert wac.Portability.SCORER_FIXED in vals
    assert wac.Portability.DOMAIN in vals
    assert wac.Portability.INSTANCE in vals
    # mod/hidden are instance-conditioned; muon-lr is scorer-fixed.
    assert p["mod_dim"] == wac.Portability.INSTANCE
    assert p["muon_lr"] == wac.Portability.SCORER_FIXED


def test_warp_priors_per_class_design_level():
    w = wac.warp_priors()
    assert w["per_class"]["Road"]["warp"] == "ground_homography"
    assert w["per_class"]["hood"]["warp"] == "identity"
    assert w["per_class"]["sky"]["warp"] == "rotation_only"
    assert "design" in w["status"]


# --------------------------------------------------------------------------
# derive_config: valid ranges, exact values, determinism
# --------------------------------------------------------------------------
def test_derive_config_valid_ranges_and_stage_order():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    assert 19 <= cfg.mod_dim <= 26
    assert cfg.hidden_dim > 0
    assert cfg.muon_lr > 0
    assert cfg.epochs > 0
    # curriculum monotone: 0 < tau < l7 <= epochs and muon >= l7.
    assert 0 < cfg.tau_softplus_start_epoch < cfg.l7_start_epoch <= cfg.epochs
    assert cfg.muon_start_epoch >= cfg.l7_start_epoch


def test_derive_config_exact_dogfood_values():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    assert cfg.mod_dim == 26
    assert cfg.hidden_dim == 96
    assert cfg.muon_lr == pytest.approx(0.002)
    assert cfg.verdict_pairs == 96
    assert cfg.surgical_levers_enabled is False
    assert cfg.dm1_enabled is False


def test_derive_config_deterministic():
    a = wac.derive_config(_GT_N600, num_pairs=600)
    b = wac.derive_config(_GT_N600, num_pairs=600)
    assert a == b


def test_derive_config_provenance_present_for_derived_fields():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    for k in ("mod_dim", "hidden_dim", "muon_lr", "verdict_pairs",
              "tau_softplus_start_epoch", "l7_start_epoch", "muon_start_epoch"):
        assert k in cfg.provenance
        assert isinstance(cfg.provenance[k], wac.ProvenancedValue)
        assert cfg.provenance[k].provenance  # non-empty rationale string


# --------------------------------------------------------------------------
# the flag-validation contract: EVERY emitted flag is a REAL trainer flag
# (behavior test — would fail if any flag were invented)
# --------------------------------------------------------------------------
def test_emitted_flags_all_exist_in_trainer_argparse():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    real = _real_trainer_flags()
    emitted = [flag for flag, _ in cfg.to_trainer_flags("out/dir")]
    missing = [f for f in emitted if f not in real]
    assert missing == [], f"invented flags not in trainer argparse: {missing}"


def test_command_has_critical_revisions_and_is_attribution_clean():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    cmd = cfg.to_command("experiments/results/levelset_n600_v2_x")
    # the 4 binding revisions
    assert "--muon-lr 0.002" in cmd
    assert "--mod-dim 26" in cmd
    assert "--hidden-dim 96" in cmd
    assert "--verdict-pairs 96" in cmd
    # attribution-clean: NO surgical levers / DM1
    for off in ("--margin-saliency", "--lane-thin", "--hardness",
                "--film-stiefel", "--code-spectral-entropy", "--dm1-telemetry"):
        assert off not in cmd, f"{off} should be OFF in attribution-clean launch"
    # from-scratch: NO resume
    assert "--resume-from" not in cmd
    # perf-env prefix present
    assert "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in cmd
    # baseline does NOT emit the all-levers flags (proven_base stays available)
    for off in ("--all-levers", "--render-aa", "--lane-render-band", "--persistence-loss-weight",
                "--amplify-weight", "--hosc-beta-end", "--adam-beta2"):
        assert off not in cmd, f"{off} should be OFF in the baseline (non-all-levers) config"


# --------------------------------------------------------------------------
# F1: all-levers config MATCHES the deep-math #205 artifact + is flag-valid
# --------------------------------------------------------------------------
_ARTIFACT = _REPO / ".omx/research/capstone_witness_launch_config_deepmath_optimal_20260702.md"


def _parse_flag_dict(pairs) -> dict:
    """(flag -> value) from a to_trainer_flags list; bare flag -> True."""
    return {f: (True if v is None else str(v)) for f, v in pairs}


def _parse_artifact_argv() -> dict:
    """Parse the ```bash``` launch block of the #205 artifact into a (flag -> value) dict."""
    m = re.search(r"```bash\n(.*?)\n```", _ARTIFACT.read_text(), re.S)
    assert m, "no ```bash``` block in the #205 artifact"
    block = m.group(1).replace("\\\n", " ")
    toks = block.split()
    i = 0
    while i < len(toks) and not toks[i].startswith("--"):  # skip env + python + script
        i += 1
    out: dict = {}
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[t] = toks[i + 1]
                i += 2
            else:
                out[t] = True
                i += 1
        else:
            i += 1
    return out


def _num_or_str(v):
    if v is True:
        return True
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def test_all_levers_argv_matches_deepmath_artifact():
    cfg = wac.derive_config(_GT_N600, num_pairs=600, epochs=1000, all_levers=True)
    launcher = _parse_flag_dict(cfg.to_trainer_flags("OUT"))
    artifact = _parse_artifact_argv()
    drop = {"--out-dir", "--gt-cache"}  # paths differ by construction
    art_keys = set(artifact) - drop
    lau_keys = set(launcher) - drop
    # every artifact flag is emitted by the launcher
    assert art_keys - lau_keys == set(), f"artifact flags missing from launcher: {art_keys - lau_keys}"
    # Wave D: --adam-beta2 is now IN the artifact argv (the --adam-beta2 flag was added in Wave C, so
    # the artifact no longer flags it as an un-emittable code change) => no launcher extras.
    assert lau_keys - art_keys == set(), f"unexpected launcher extras: {lau_keys - art_keys}"
    # every shared flag's VALUE matches (numeric-aware so 1e-3 == 0.001)
    mism = [(k, artifact[k], launcher[k]) for k in (art_keys & lau_keys)
            if _num_or_str(artifact[k]) != _num_or_str(launcher[k])]
    assert mism == [], f"value mismatches vs artifact: {mism}"


def test_all_levers_flags_all_exist_in_trainer_argparse():
    cfg = wac.derive_config(_GT_N600, num_pairs=600, epochs=1000, all_levers=True)
    real = _real_trainer_flags()
    emitted = [flag for flag, _ in cfg.to_trainer_flags("out/dir")]
    missing = [f for f in emitted if f not in real]
    assert missing == [], f"invented flags not in trainer argparse: {missing}"


def test_all_levers_derived_fields():
    cfg = wac.derive_config(_GT_N600, num_pairs=600, epochs=1000, all_levers=True)
    assert cfg.all_levers is True
    assert cfg.mod_dim == 19          # Whitney floor (aggressive) regardless of overfit
    assert cfg.verdict_pairs == 0     # ALL pairs, async
    assert cfg.l7_start_epoch == 1000  # l7 DEMOTED to epochs
    assert cfg.adam_beta2 == pytest.approx(0.9999999)
    # muon<l7 after demote is intentional (benign trainer WARN); do NOT assert muon>=l7 here.
    assert cfg.muon_start_epoch == 726


def test_all_levers_beta2_clears_smalln_threshold():
    # #222: 1-beta2 must be <= (1-0.9^5)/n^3.5 for n=75.
    n = 75
    thresh = (1.0 - 0.9 ** 5) / (n ** 3.5)
    cfg = wac.derive_config(_GT_N600, num_pairs=600, epochs=1000, all_levers=True)
    assert (1.0 - cfg.adam_beta2) <= thresh, "all-levers beta2 must clear the small-n threshold"


def test_baseline_beta2_is_mlx_default_bit_identical():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)  # all_levers default False
    assert cfg.adam_beta2 == pytest.approx(0.999)  # == MLX AdamW default => bit-identical path


# --------------------------------------------------------------------------
# Wave D AA CORRECTION: all-levers ships --render-aa none + lane-render-band
# (supersample DISQUALIFIED per aa_feasibility_reconciliation_20260702.md)
# --------------------------------------------------------------------------
def test_all_levers_aa_is_none_not_supersample():
    cfg = wac.derive_config(_GT_N600, num_pairs=600, epochs=1000, all_levers=True)
    fd = _parse_flag_dict(cfg.to_trainer_flags("OUT"))
    # --render-aa NONE is the contest-feasible optimal AA (Wave D correction)
    assert fd.get("--render-aa") == "none", f"all-levers must ship --render-aa none, got {fd.get('--render-aa')}"
    # the analytic coverage-integrated lane-render-band IS the AA (kept)
    assert "--lane-render-band" in fd
    # the disqualified supersample flags are NOT emitted by the launch config
    assert "--aa-supersample" not in fd, "supersample DISQUALIFIED (hurts -49% + decode over budget)"
    assert "--aa-self-orient-fine-mode" not in fd, "fine dir-feat cache dropped with supersample"


def test_all_levers_base_dict_has_no_supersample_keys():
    base = wac._all_levers_base(300)
    assert base["render_aa"] == "none"
    assert "aa_supersample" not in base
    assert "aa_self_orient_fine_mode" not in base


# --------------------------------------------------------------------------
# C5 (SEAL review 2026-07-04): the FRESH SEEDED run-1 config — sealed_205 +
# EXACTLY the review's deltas; event-triggered / bank-6 / dilate EXCLUDED.
# --------------------------------------------------------------------------
def _fresh_cfg():
    return wac.derive_fresh_seeded_config(_GT_N600, num_pairs=600, epochs=1000)


def test_fresh_seeded_carries_all_review_deltas():
    fd = _parse_flag_dict(_fresh_cfg().to_trainer_flags("OUT"))
    # the 13-lever revised shape, value by value (numeric-aware where rendering varies)
    assert fd["--lane-prior-phi1-mode"] == "paint"
    assert "--seed-islands" in fd
    assert float(fd["--eikonal-weight"]) == 0.05
    assert float(fd["--eikonal-weight-end"]) == 0.10
    assert fd["--tau-anneal-shape"] == "geometric"
    assert float(fd["--softmax-temp-end"]) == 1.0
    assert fd["--mod-dim"] == "19"
    assert "--film-stiefel" in fd
    assert "--muon-warm-start-momentum" in fd
    assert float(fd["--muon-lr-final-frac"]) == 0.1
    assert fd["--lane-band-start-epoch"] == "350"
    assert fd["--stage-transition-rewarmup-epochs"] == "20"
    assert fd["--stage-transition-rewarmup-shape"] == "cosine"
    assert "--closed-loop-control" in fd
    assert fd["--l7-start-epoch"] == "1001"
    assert float(fd["--hosc-beta-end"]) == 5.134
    assert fd["--verdict-batch"] == "64"


def test_fresh_seeded_constant_tau_is_inert_exact():
    """Review L2: geometric shape with start==end==1.0 => tau constant 1.0 EXACTLY."""
    fd = _parse_flag_dict(_fresh_cfg().to_trainer_flags("OUT"))
    assert float(fd["--softmax-temp-start"]) == 1.0
    assert float(fd["--softmax-temp-end"]) == 1.0
    assert fd["--tau-anneal-shape"] == "geometric"


def test_fresh_seeded_excludes_the_critical_findings():
    """C1/C2: NO event-triggered curriculum. C3: NO bank change (bank-4 = trainer default, memory-
    UNSAFE at bank-6). Dilate KEEP 1 (sealed value)."""
    fd = _parse_flag_dict(_fresh_cfg().to_trainer_flags("OUT"))
    assert "--curriculum-event-triggered" not in fd, "C1/C2: run-2 lever, must NOT be in run 1"
    assert "--bank-n-scales" not in fd, "C3: bank stays at the trainer default 4 (bank-6 REFUSEd)"
    assert fd["--island-dilate-px"] == "1"


def test_fresh_seeded_flags_all_exist_in_trainer_argparse():
    real = _real_trainer_flags()
    emitted = [flag for flag, _ in _fresh_cfg().to_trainer_flags("out/dir")]
    missing = [f for f in emitted if f not in real]
    assert missing == [], f"invented flags not in trainer argparse: {missing}"


def test_fresh_seeded_inherits_sealed_non_delta_values():
    """Reuse-not-retype: every non-delta knob must equal the sealed config's value."""
    sealed = _parse_flag_dict(
        wac.derive_sealed_205_config(_GT_N600, num_pairs=600, epochs=1000).to_trainer_flags("OUT"))
    fresh = _parse_flag_dict(_fresh_cfg().to_trainer_flags("OUT"))
    deltas = {"--lane-prior-phi1-mode", "--eikonal-weight", "--tau-anneal-shape",
              "--softmax-temp-end", "--mod-dim", "--lane-band-start-epoch",
              "--stage-transition-rewarmup-epochs", "--stage-transition-rewarmup-shape",
              "--l7-start-epoch", "--hosc-beta-end", "--verdict-batch"}
    new_only = {"--seed-islands", "--eikonal-weight-end", "--film-stiefel",
                "--muon-warm-start-momentum", "--muon-lr-final-frac", "--closed-loop-control",
                # #314 / DRIFT-D2 fix: the store-nothing pose source is an EXPLICIT fresh delta now
                # (sealed keeps the field default real_keyframe => never emits the flag).
                "--pose-carrier-source"}
    assert set(fresh) - set(sealed) == new_only
    for k in set(sealed) & set(fresh) - deltas:
        assert fresh[k] == sealed[k], f"non-delta knob drifted: {k}: {sealed[k]} -> {fresh[k]}"


def test_sealed_205_unchanged_by_fresh_seeded_machinery():
    """Regression: the new verdict_batch/fresh_seeded fields default so sealed_205 is byte-stable
    (--verdict-batch 32, NONE of the fresh-only flags emitted)."""
    cfg = wac.derive_sealed_205_config(_GT_N600, num_pairs=600, epochs=1000)
    assert cfg.fresh_seeded is False and cfg.verdict_batch == 32
    fd = _parse_flag_dict(cfg.to_trainer_flags("OUT"))
    assert fd["--verdict-batch"] == "32"
    for f in ("--seed-islands", "--eikonal-weight-end", "--film-stiefel",
              "--muon-warm-start-momentum", "--muon-lr-final-frac", "--closed-loop-control"):
        assert f not in fd, f"sealed_205 must not emit the fresh-only flag {f}"


def test_fresh_seeded_carries_store_nothing_pose_carrier_source():
    """#314 / DAG DRIFT-D2 regression pin (2026-07-06): the fresh_seeded family's INTENDED pose
    frame0 source is the STORE-NOTHING generated render (the #205 argv ledger KEEP row). The v1->v5
    drift happened because derive_fresh_seeded_config inherited derive_sealed_205_config's field
    default real_keyframe and the flag is emitted only when != default — so the intent silently
    vanished from the argv (rate-accounting drift: counted uint8-keyframe vs ~1 KB store-nothing).
    Pin BOTH sides: fresh emits --pose-carrier-source generated; sealed stays byte-identical
    (never emits the flag)."""
    cfg = _fresh_cfg()
    assert cfg.pose_carrier_source == "generated"
    fd = _parse_flag_dict(cfg.to_trainer_flags("OUT"))
    assert fd["--pose-carrier-source"] == "generated", \
        "#314 regression: fresh_seeded must EXPLICITLY emit the store-nothing pose source"
    # provenance row records the drift + the fix (queryable, no rediscovery)
    assert "pose_carrier_source" in cfg.provenance
    assert "#314" in cfg.provenance["pose_carrier_source"].provenance
    # sealed_205 byte-identity unchanged: field default real_keyframe => flag not emitted
    sealed = _parse_flag_dict(
        wac.derive_sealed_205_config(_GT_N600, num_pairs=600, epochs=1000).to_trainer_flags("OUT"))
    assert "--pose-carrier-source" not in sealed
    # store_nothing_205 (the original Track B arm) still emits generated — the two surfaces agree
    sn = _parse_flag_dict(
        wac.derive_store_nothing_205_config(
            _GT_N600, num_pairs=600, epochs=1000).to_trainer_flags("OUT"))
    assert sn["--pose-carrier-source"] == "generated"


def test_fresh_seeded_every_delta_key_materializes_in_argv():
    """CLASS guard for the #314/D2 drift class (config-orphan confound): a key added to
    _FRESH_SEEDED_DELTAS but never consumed by derive_fresh_seeded_config silently does nothing —
    exactly how the v1->v5 pose-source intent vanished. For EVERY delta key, assert the fresh argv
    actually DIFFERS from the sealed argv at that key's flag; the coverage assertion makes an
    unmapped new key fail loudly here instead of orphaning."""
    key_to_flag = {
        "lane_prior_phi1_mode": "--lane-prior-phi1-mode",
        "eikonal_weight": "--eikonal-weight",
        "softmax_temp_end": "--softmax-temp-end",
        "st_rewarmup_epochs": "--stage-transition-rewarmup-epochs",
        "st_rewarmup_shape": "--stage-transition-rewarmup-shape",
        "tau_anneal_shape": "--tau-anneal-shape",
        "lane_band_start_epoch": "--lane-band-start-epoch",
        "hosc_beta_end": "--hosc-beta-end",
        "mod_dim": "--mod-dim",
        "l7_start_epoch": "--l7-start-epoch",
        "verdict_batch": "--verdict-batch",
        "pose_carrier_source": "--pose-carrier-source",
    }
    missing_map = set(wac._FRESH_SEEDED_DELTAS) - set(key_to_flag)
    assert missing_map == set(), \
        f"new _FRESH_SEEDED_DELTAS key(s) {missing_map} not mapped here — map them AND consume them " \
        "in derive_fresh_seeded_config, or the intent orphans (the #314/D2 drift class)"
    sealed = _parse_flag_dict(
        wac.derive_sealed_205_config(_GT_N600, num_pairs=600, epochs=1000).to_trainer_flags("OUT"))
    fresh = _parse_flag_dict(_fresh_cfg().to_trainer_flags("OUT"))
    for key, flag in key_to_flag.items():
        assert flag in fresh, f"delta {key} did not materialize: {flag} absent from fresh argv"
        assert fresh.get(flag) != sealed.get(flag), \
            f"delta {key} is a NO-OP: {flag} identical in sealed and fresh argv " \
            "(the value was not consumed — #314/D2 drift class)"


def test_tracked_deltas_runtime_guard_unit():
    """Unit-test the #314 CLASS-fix mechanism (_TrackedDeltas): a read marks CONSUMED, membership
    does NOT, .raw is untracked (so provenance echoes can't mask an orphan), assert_all_consumed
    raises iff a key was never read. This is the runtime fail-closed core the point-fix lacked."""
    raw = {"a": 1, "b": 2, "c": 3}
    t = wac._TrackedDeltas(raw)
    # membership + .raw snapshot must NOT count as consumption
    assert "a" in t
    assert dict(t.raw) == raw
    with pytest.raises(ValueError) as ei:
        t.assert_all_consumed("fn", "_D")
    msg = str(ei.value)
    assert "a" in msg and "b" in msg and "c" in msg and "#314" in msg
    # read each key -> now fully consumed -> no raise
    assert (t["a"], t["b"], t["c"]) == (1, 2, 3)
    t.assert_all_consumed("fn", "_D")  # must not raise
    # a partially-consumed dict names ONLY the orphan
    t2 = wac._TrackedDeltas({"x": 1, "y": 2})
    _ = t2["x"]
    with pytest.raises(ValueError) as ei2:
        t2.assert_all_consumed("fn", "_D")
    assert "'y'" in str(ei2.value) and "'x'" not in str(ei2.value)


def test_fresh_seeded_runtime_guard_fires_on_unconsumed_delta_key(monkeypatch):
    """#314 CLASS fix (runtime): adding a key to _FRESH_SEEDED_DELTAS that derive_fresh_seeded_config
    does NOT extract must FAIL LOUD at derive time — not silently orphan into provenance-only (the
    exact D2 mechanism). monkeypatch.setitem auto-restores the module dict after the test."""
    monkeypatch.setitem(wac._FRESH_SEEDED_DELTAS, "__bogus_orphan_key__", 123)
    with pytest.raises(ValueError) as ei:
        wac.derive_fresh_seeded_config(_GT_N600, num_pairs=600, epochs=1000)
    msg = str(ei.value)
    assert "__bogus_orphan_key__" in msg
    assert "_FRESH_SEEDED_DELTAS" in msg and "DRIFT-D2" in msg


def test_fresh_seeded_runtime_guard_admits_the_real_config():
    """The real config consumes EVERY _FRESH_SEEDED_DELTAS key, so derive must NOT raise, and the
    provenance snapshot (read via .raw, untracked) still records all delta keys. Guards against the
    guard itself becoming a false-positive that blocks the launch config."""
    cfg = wac.derive_fresh_seeded_config(_GT_N600, num_pairs=600, epochs=1000)
    snap = cfg.provenance["fresh_seeded_deltas"].value
    assert set(snap) == set(wac._FRESH_SEEDED_DELTAS)
    # and pose_carrier_source specifically survives (the original D2 casualty)
    assert cfg.pose_carrier_source == "generated"


def test_fresh_seeded_provenance_records_the_deltas():
    cfg = _fresh_cfg()
    assert cfg.fresh_seeded is True
    for k in ("mod_dim", "l7_start_epoch", "fresh_seeded_deltas"):
        assert k in cfg.provenance
        assert "2026-07-04" in cfg.provenance[k].provenance or \
               "fresh" in cfg.provenance[k].provenance.lower() or \
               "SEAL review" in cfg.provenance[k].provenance


def test_332_dsl_lever_composition_byte_identical_when_empty_and_composes_when_set():
    """#332: --dsl-lever composes named DSL Lever factories over the base config, delegating to
    the DSL SoT; byte-identical to the base when empty (every existing gate unchanged)."""
    import dataclasses as _dc
    cfg = wac.derive_config(_GT_N600, num_pairs=600, overfit=True, epochs=1000)
    base = cfg.to_trainer_flags("OUT")
    assert _dc.replace(cfg, dsl_levers=()).to_trainer_flags("OUT") == base, "empty must be byte-identical"
    composed = _dc.replace(
        cfg, dsl_levers=("SeedIslandEased", "EventTriggeredCurriculum", "AmplifyIsland")
    ).to_trainer_flags("OUT")
    fd = dict(composed)
    assert fd.get("--seed-island-eased") is None            # bare boolean rendered as (flag, None)
    assert fd.get("--curriculum-nucleus-guard") is None
    assert fd.get("--amplify-weight") == 1.0
    # every composed flag is a REAL trainer flag (no invented flags)
    from tac.witness_dsl.curriculum_dsl import real_trainer_flags
    realf = set(real_trainer_flags(None))
    assert all(f in realf for f, _ in composed), "composed argv must not invent flags"


def test_332_dsl_lever_bad_name_raises():
    import dataclasses as _dc
    import pytest as _pytest
    cfg = wac.derive_config(_GT_N600, num_pairs=600, overfit=True, epochs=1000)
    with _pytest.raises(ValueError, match="not a curriculum_dsl Lever factory"):
        _dc.replace(cfg, dsl_levers=("NotARealLever",)).to_trainer_flags("OUT")


def test_332_merge_dsl_levers_false_override_agrees_with_compile_trainer_argv():
    """#334 review MED: the two 'one emitter' surfaces — witness_autoconfig._merge_dsl_levers and
    curriculum_dsl.WitnessProgram.compile_trainer_argv — MUST agree that a ``False`` override emits
    the BooleanOptionalAction negation ``--no-<flag>`` (previously _merge silently skipped it, a
    latent divergence). This asserts both surfaces negate identically."""
    import dataclasses as _dc

    from tac.witness_dsl.curriculum_dsl import BASELINE, Lever

    # (a) compile_trainer_argv (the DSL emitter): a False value -> --no-<flag> bare token.
    prog = _dc.replace(BASELINE, base={**BASELINE.base, "--chroma": False})
    argv = prog.compile_trainer_argv()
    assert "--no-chroma" in argv, "DSL emitter must negate a False flag to --no-<flag>"
    assert "--chroma" not in argv, "the positive form must not co-occur with the negation"

    # (b) _merge_dsl_levers: a lever whose override sets an existing base flag to False must
    #     replace that flag IN-PLACE with (--no-<flag>, None) — matching the emitter's negation.
    cfg = wac.derive_config(_GT_N600, num_pairs=600, overfit=True, epochs=1000)
    base = cfg.to_trainer_flags("OUT")
    # pick a real bare-boolean flag present in the base to negate — one with a legal --no- form
    # (BooleanOptionalAction, NOT store_true: a False on store_true is now REFUSED, see
    # test_merge_dsl_levers_false_on_store_true_refused).
    from tac.witness_dsl.curriculum_dsl import real_store_true_flags
    _st = real_store_true_flags(None)
    bare_flags = [f for f, v in base if v is None and f not in _st]
    assert bare_flags, "base must carry at least one negatable (non-store_true) bare-boolean flag"
    target = bare_flags[0]

    # inject a synthetic lever factory carrying a False override on that flag.
    import tac.witness_dsl.curriculum_dsl as _cd

    factory_name = "_TestFalseOverrideLever"
    setattr(_cd, factory_name, lambda: Lever(name="_test_false", overrides={target: False}))
    try:
        merged = _dc.replace(cfg, dsl_levers=(factory_name,)).to_trainer_flags("OUT")
    finally:
        delattr(_cd, factory_name)

    md = dict(merged)
    negated = target.replace("--", "--no-", 1)
    assert negated in md and md[negated] is None, f"{target} must be negated to {negated} in-place"
    assert target not in md, f"the positive {target} must be replaced (not left alongside {negated})"
    # in-place: no net length change (replacement, not append).
    assert len(merged) == len(base), "False override must replace in-place, not append"


def test_dsl_lever_muon_and_dm1minimal_refused_with_typed_error():
    """CLASS-FIX (review 2026-07-06): --dsl-lever Muon previously crashed the config generator
    with a raw TypeError (required start_epoch arg) and --dsl-lever DM1Minimal with an
    AttributeError (tuple has no .overrides). Both must now raise the CLEAR typed
    LeverCompositionError naming the reason + the composable set — never a raw traceback."""
    import dataclasses as _dc

    import pytest as _pytest

    from tac.witness_dsl.lever_registry import LeverCompositionError

    cfg = wac.derive_config(_GT_N600, num_pairs=600, overfit=True, epochs=1000)
    with _pytest.raises(LeverCompositionError, match="requires explicit args"):
        _dc.replace(cfg, dsl_levers=("Muon",)).to_trainer_flags("OUT")
    with _pytest.raises(LeverCompositionError, match="returns tuple"):
        _dc.replace(cfg, dsl_levers=("DM1Minimal",)).to_trainer_flags("OUT")
    # the message is operator-actionable (lists what IS composable)
    try:
        _dc.replace(cfg, dsl_levers=("Muon",)).to_trainer_flags("OUT")
    except LeverCompositionError as exc:
        assert "composable" in str(exc) and "SeedIslandEased" in str(exc)


def test_merge_dsl_levers_false_on_store_true_refused():
    """C2 mirror at the merge surface: a lever override setting False on a plain store_true
    flag (no --no- form) must be REFUSED with the typed error — the negation --no-<flag>
    would crash the trainer argparse at launch (curriculum_dsl validate() C2)."""
    import dataclasses as _dc

    import pytest as _pytest

    import tac.witness_dsl.curriculum_dsl as _cd
    from tac.witness_dsl.curriculum_dsl import Lever, real_store_true_flags
    from tac.witness_dsl.lever_registry import LeverCompositionError

    target = "--stage-transition-reset-moments"          # store_true in the trainer argparse
    assert target in real_store_true_flags(None), "test premise: target must be store_true"
    cfg = wac.derive_config(_GT_N600, num_pairs=600, overfit=True, epochs=1000)
    factory_name = "_TestFalseOnStoreTrueLever"
    setattr(_cd, factory_name, lambda: Lever(name="_test_st_false", overrides={target: False}))
    try:
        with _pytest.raises(LeverCompositionError, match="store_true"):
            _dc.replace(cfg, dsl_levers=(factory_name,)).to_trainer_flags("OUT")
    finally:
        delattr(_cd, factory_name)


# --------------------------------------------------------------------------
# T5 CRUCIBLE v6.2 (seal-round-2 BLOCKER-1 fix) — config-materialization tests:
# design-doc schedule == emitted launch tokens, guarded as a CLASS.
# --------------------------------------------------------------------------
def _crucible_cfg(epochs: int = 3000):
    return wac.derive_crucible_v6_config(_GT_N600, num_pairs=600, epochs=epochs)


def _tau_law(ep: int, *, anneal_epochs: int, start: float, end: float,
             shape: str, hold_frac: float) -> float:
    """Replica of the trainer's _softmax_temp_for_epoch (pure law, L2341-2386):
    prog_t=(ep-1)/max(ae-1,1); cosine_hold returns `end` at prog>=hold_frac, else rescales
    prog/hold_frac into the cosine. Cross-checked below against the mod32cap MEASURED anchor."""
    prog = (ep - 1) / max(anneal_epochs - 1, 1)
    if shape == "cosine_hold" and hold_frac < 1.0:
        if prog >= hold_frac:
            return end
        prog = prog / hold_frac
    return float(end + 0.5 * (start - end) * (1 + np.cos(np.pi * prog)))


def test_tau_law_replica_reproduces_mod32cap_measured_anchor():
    """The replica must reproduce the REAL vehicle: mod32cap (den 1000, end 0.05, cosine)
    measured tau(650)=0.3098 (the ep650-best anchor tau_end 0.31 is derived from) and the
    muon-freeze value tau(726)=0.2157 (the draft's END leg)."""
    t650 = _tau_law(650, anneal_epochs=1000, start=1.0, end=0.05, shape="cosine", hold_frac=1.0)
    t726 = _tau_law(726, anneal_epochs=1000, start=1.0, end=0.05, shape="cosine", hold_frac=1.0)
    assert abs(t650 - 0.3098) < 5e-4, t650
    assert abs(t726 - 0.2157) < 5e-4, t726


def test_crucible_v6_schedule_matches_design_doc():
    """BLOCKER-1 config-materialization gate (seal_round2_v6_verdict_20260708.md §1.1): the
    EMITTED tokens, run through the trainer's own tau law, must realize the v6 schedule —
    descent completes at ABSOLUTE ep600, tau(675) [fire band] and tau(726) [Muon freeze] sit
    AT the 0.31 anchor, and the stage anchors are ABSOLUTE (never family-scaled 0.726*epochs).
    The round-2 measured failure modes are pinned as anti-targets: den-3000-cosine gave
    tau(675)=0.886; family scaling gave --muon-start-epoch 2178 and NO --anneal-epochs token."""
    fd = _parse_flag_dict(_crucible_cfg().to_trainer_flags("OUT"))
    # the schedule tokens EXIST and are explicit (the round-2 blocker: no --anneal-epochs at all)
    assert fd["--anneal-epochs"] == "3000"
    assert fd["--tau-anneal-shape"] == "cosine_hold"
    assert fd["--tau-hold-frac"] == "0.2"
    assert fd["--softmax-temp-end"] == "0.31"
    # ABSOLUTE anchors, not family-scaled (2178 / 900 were the measured wrong emissions)
    assert fd["--muon-start-epoch"] == "726"
    assert fd["--tau-softplus-start-epoch"] == "300"
    kw = dict(anneal_epochs=int(fd["--anneal-epochs"]),
              start=float(fd["--softmax-temp-start"]), end=float(fd["--softmax-temp-end"]),
              shape=fd["--tau-anneal-shape"], hold_frac=float(fd["--tau-hold-frac"]))
    # descent completes at ABSOLUTE ep ~600 (hold_frac * anneal_epochs) and HOLDS at 0.31
    assert abs(_tau_law(600, **kw) - 0.31) < 1e-4
    assert _tau_law(675, **kw) == 0.31          # fire band [670,700]: anneal-complete, HELD
    assert _tau_law(726, **kw) == 0.31          # Muon fail-safe cap freeze value
    assert _tau_law(3000, **kw) == 0.31         # no late-run drift below the anchor
    # anti-target: the round-2 wrong emission (den=epochs cosine) does NOT hold the anchor
    assert _tau_law(675, anneal_epochs=3000, start=1.0, end=0.31,
                    shape="cosine", hold_frac=1.0) > 0.85
    # anti-target: plain cosine at den 600 REBOUNDS past the hold (the derived-not-guessed pin)
    assert _tau_law(726, anneal_epochs=600, start=1.0, end=0.31,
                    shape="cosine", hold_frac=1.0) > 0.38


def _beta_law(ep: int, *, start: float, end: float, anneal_epochs: int, shape: str = "linear") -> float:
    """Replica of the trainer's _hosc_beta_for_epoch (pure law, L2318-2338): prog=(ep-1)/(ae-1);
    linear (default) => start + (end-start)*prog; shares --anneal-epochs with the τ + LR schedules."""
    prog = (ep - 1) / max(anneal_epochs - 1, 1)
    if shape == "cosine":
        return float(end + 0.5 * (start - end) * (1 + np.cos(np.pi * prog)))
    return float(start + (end - start) * prog)


def test_crucible_v6_beta_pin_reproduces_control_trajectory():
    """v6.3 MAJOR-2(i): the emitted β pin (--hosc-beta-end 10.0, linear, den 3000) must reproduce the
    control's β(ep) on [1,726] to ≤0.1% — the anchors were measured at the control's JOINT β state.
    Control = mod32cap: den 1000, start 1.0, end 4.0, linear. The un-pinned inherited end=4.0 at
    den 3000 gives β(726)=1.7252 (the value the round-1 verdict corrected the 1.41 cosine misprint to);
    the pin restores β(726)≈3.176. The 1.41 anti-target is the COSINE-shape value, NOT the emitted shape."""
    fd = _parse_flag_dict(_crucible_cfg().to_trainer_flags("OUT"))
    assert fd["--hosc-beta-anneal"] == "linear"
    b_start, b_end = float(fd["--hosc-beta"]), float(fd["--hosc-beta-end"])
    ae = int(fd["--anneal-epochs"])
    # control trajectory (the trace the anchors were measured on)
    ctrl_726 = _beta_law(726, start=1.0, end=4.0, anneal_epochs=1000)
    ctrl_650 = _beta_law(650, start=1.0, end=4.0, anneal_epochs=1000)
    assert abs(ctrl_726 - 3.177) < 1e-3 and abs(ctrl_650 - 2.9489) < 1e-3
    # emitted pin reproduces it within 0.1% at both absolute anchors
    assert abs(_beta_law(726, start=b_start, end=b_end, anneal_epochs=ae) - ctrl_726) / ctrl_726 < 1e-3
    assert abs(_beta_law(650, start=b_start, end=b_end, anneal_epochs=ae) - ctrl_650) / ctrl_650 < 1e-3
    # anti-target: the un-pinned inherited end=4.0 at the SHARED den 3000 deviates ~1.8× (β(726)=1.7252)
    assert abs(_beta_law(726, start=1.0, end=4.0, anneal_epochs=ae) - 1.7252) < 1e-3
    # anti-target: 1.41 was the COSINE-shape misprint, not the emitted (linear) shape
    assert abs(_beta_law(726, start=1.0, end=4.0, anneal_epochs=ae, shape="cosine") - 1.4122) < 1e-3


def _load_trainer_module():
    """Lazily import the trainer module (0.1s; main() is __main__-guarded, no MLX init at import)
    to test the REAL _lr_scheduled_for_epoch — the strongest byte-identity guard (not a replica)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tlw_lr_test", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _LrArgs:
    """Minimal args namespace for _lr_scheduled_for_epoch (the LR trio + warmup + hold-frac)."""
    def __init__(self, lr=1e-3, lr_end=1e-4, warmup_epochs=1, lr_hold_frac=1.0):
        self.lr = lr
        self.lr_end = lr_end
        self.warmup_epochs = warmup_epochs
        self.lr_hold_frac = lr_hold_frac


def _lr_inline(ep: int, ae: int, a: "_LrArgs") -> float:
    """The PRE-BUILD inline LR formula (warmup -> plain cosine at denominator ``ae``), reproduced
    here so the default-off byte-identity test compares the real helper against the exact prior code."""
    if ep <= a.warmup_epochs:
        return a.lr * ep / max(a.warmup_epochs, 1)
    prog = (ep - a.warmup_epochs) / max(ae - a.warmup_epochs, 1)
    return a.lr_end + 0.5 * (a.lr - a.lr_end) * (1 + np.cos(np.pi * prog))


def test_lr_scheduled_default_off_is_bit_identical():
    """DEFAULT-OFF gate: --lr-anneal-epochs unset (=> lr_anneal_epochs == the shared anneal_epochs)
    AND --lr-hold-frac 1.0 => the real _lr_scheduled_for_epoch equals the pre-build inline cosine
    EXACTLY over the full run [1,3000] (both warmup and cosine branches). Byte-identical == max |Δ|=0."""
    m = _load_trainer_module()
    a = _LrArgs(lr_hold_frac=1.0)
    for ae in (1000, 3000):
        maxd = max(abs(m._lr_scheduled_for_epoch(ep, a, ae) - _lr_inline(ep, ae, a))
                   for ep in range(1, 3001))
        assert maxd == 0.0, f"default-off not bit-identical at den {ae}: max|Δ|={maxd}"


def test_crucible_v6_lr_pin_reproduces_control_trajectory():
    """v6.4 MAJOR-2(ii) BUILD: the emitted LR pin (--lr-anneal-epochs 1000, --lr-hold-frac 1.0) run
    through the REAL trainer helper must reproduce the mod32cap CONTROL's LR(ep) on [1,726]
    BIT-IDENTICALLY (the ν/settle-237/s*/fire-band laws were measured at that annealed LR). Control =
    mod32cap: den 1000 (= its --epochs), lr/lr_end/warmup = the shared 1e-3/1e-4/1 defaults. The
    un-pinned shared den 3000 is the anti-target: it runs the AdamW phase at 2.83× (ep675) → 3.41×
    (ep726) the control LR (the 3× deviation that staled the window laws)."""
    m = _load_trainer_module()
    fd = _parse_flag_dict(_crucible_cfg().to_trainer_flags("OUT"))
    # the LR pin tokens EXIST and are explicit (the v6.3 residual was: no LR denominator flag at all)
    assert fd["--lr-anneal-epochs"] == "1000"
    assert fd["--lr-hold-frac"] == "1.0"
    # the LR trio is the shared default (so den-split ALONE reproduces control); guard it
    assert float(fd["--lr"]) == 1e-3 and float(fd["--lr-end"]) == 1e-4
    lr_ae = int(fd["--lr-anneal-epochs"])
    a = _LrArgs(lr=float(fd["--lr"]), lr_end=float(fd["--lr-end"]),
                lr_hold_frac=float(fd["--lr-hold-frac"]))
    ctrl = _LrArgs(lr=1e-3, lr_end=1e-4, warmup_epochs=1, lr_hold_frac=1.0)  # mod32cap defaults
    # BIT-IDENTICAL to the control on [1,726] (the anchor band the window laws were measured on)
    maxd = max(abs(m._lr_scheduled_for_epoch(ep, a, lr_ae) - _lr_inline(ep, 1000, ctrl))
               for ep in range(1, 727))
    assert maxd == 0.0, f"LR pin not bit-identical to control on [1,726]: max|Δ|={maxd}"
    # anti-target: the shared den 3000 deviates 2.6-3.4× across the fire->freeze band (the RISK ROW)
    for ep, lo, hi in ((675, 2.6, 3.0), (726, 3.3, 3.5)):
        ratio = (m._lr_scheduled_for_epoch(ep, a, 3000) / _lr_inline(ep, 1000, ctrl))
        assert lo < ratio < hi, f"ep{ep} shared-den/control ratio {ratio:.3f} outside [{lo},{hi}]"


def test_crucible_v6_pose_block_pinned():
    """MAJOR-A2 + #314 guard: the pose leg is pinned AT the config surface (inherited
    structurally from store_nothing_205, asserted here so inheritance drift is a test failure,
    not a silent revert): w-pose 1.0 + --pose-carrier + residual-mode table + SOURCE generated
    (store-nothing; real_keyframe is the excluded wrong mover)."""
    cfg = _crucible_cfg()
    assert cfg.pose_carrier_source == "generated"
    fd = _parse_flag_dict(cfg.to_trainer_flags("OUT"))
    assert fd["--w-pose"] == "1.0"
    assert fd["--pose-carrier"] is True
    assert fd["--pose-carrier-residual-mode"] == "table"
    assert fd["--pose-carrier-source"] == "generated"


def test_crucible_v6_knob_pins_and_dsl_levers_materialize():
    """Every v6 §1.1 knob with a 1:1 trainer flag materializes in the argv."""
    fd = _parse_flag_dict(_crucible_cfg().to_trainer_flags("OUT"))
    assert fd["--fused-r-kernel"] is True                       # F-DET (fold 1)
    # v6.3 MAJOR-1: --curriculum-plateau-windows is NOT emitted (wrong surface; V=5 binds the B1
    # spec only, no trainer flag). It must be ABSENT from the argv.
    assert "--curriculum-plateau-windows" not in fd
    assert fd["--curriculum-reanchor-levers"] is True           # v6.3 MAJOR-3 re-anchor leg
    assert fd["--curriculum-min-stage-epochs"] == "250"         # v6.3 MINOR-4 dwell-law pin
    assert fd["--hosc-beta-end"] == "10.0"                      # v6.3 MAJOR-2(i) β-pin
    assert fd["--hosc-beta"] == "1.0"                           # start (inherited); shape linear
    assert fd["--hosc-beta-anneal"] == "linear"
    assert fd["--lr-anneal-epochs"] == "1000"                  # v6.4 MAJOR-2(ii) LR-pin (control den)
    assert fd["--lr-hold-frac"] == "1.0"                       # v6.4 LR-pin: no hold (bit-identical)
    assert fd["--curriculum-event-triggered"] is True           # handoff="event"
    assert fd["--curriculum-nucleus-guard"] is True
    assert fd["--seg-chroma-boundary-weight"] == "0.1"          # ChromaBoundarySharpen
    assert fd["--seg-chroma-boundary-margin-band"] == "1.0"
    assert fd["--seg-chroma-boundary-start-epoch"] == "300"     # start="tau_fire" = tau@300
    assert fd["--render-aa"] == "ipe"                           # AACoverageRender(mode="ipe")
    assert fd["--lane-band-start-epoch"] == "350"               # AnalyticLaneRenderBand(start=350)
    assert fd["--persistence-warmup-epochs"] == "275"           # PersistenceTopology(warmup=275)
    assert fd["--seed-islands"] is True                         # SeedIslandBirth
    assert fd["--witness-alone-island-loss"] is True
    assert fd["--seed-island-eased"] is True                    # SeedIslandEased
    assert fd["--logit-adjust-loss-tau"] == "1.0"               # LogitAdjust(tau=1.0)
    assert fd["--length-sigma-matrix"] == "fitted-20260707"     # LengthSigma
    assert fd["--cache-gt-skeleton"] is True                    # CacheGtSkeleton
    assert fd["--muon-warm-start-momentum"] is True             # MuonWarmStart
    assert fd["--muon-lr-final-frac"] == "0.1"
    assert fd["--weight-entropy-penalty-lambda"] == "15.0"      # WeightEntropyPenaltyMLX(lam=15)
    assert fd["--mod-dim"] == "32"                              # sealed Q4 inherited
    assert fd["--verdict-batch"] == "32"                        # R2 OOM fix inherited
    assert fd["--verdict-pairs"] == "0"                         # n600-scale rule inherited


def test_crucible_v6_flags_all_exist_in_trainer_argparse():
    real = _real_trainer_flags()
    emitted = [flag for flag, _ in _crucible_cfg().to_trainer_flags("out/dir")]
    missing = [f for f in emitted if f not in real]
    assert missing == [], f"invented flags not in trainer argparse: {missing}"


def test_crucible_v6_no_duplicate_long_flags():
    """C13 at the SOURCE: the variant pins --softmax-temp-end 0.31 itself, so the argv carries
    each long flag exactly once (the round-2 extras route was REFUSED on this exact gate)."""
    emitted = [flag for flag, _ in _crucible_cfg().to_trainer_flags("OUT")]
    dups = sorted({f for f in emitted if emitted.count(f) > 1})
    assert dups == [], f"duplicate long-flags in the emitted argv: {dups}"


def test_store_nothing_205_unchanged_by_crucible_machinery():
    """Regression: the crucible_v6 field defaults False so the store-nothing (and sealed) argvs
    are byte-stable — none of the crucible-only pins leaks into them."""
    cfg = wac.derive_store_nothing_205_config(_GT_N600, num_pairs=600, epochs=1000)
    assert cfg.crucible_v6 is False
    fd = _parse_flag_dict(cfg.to_trainer_flags("OUT"))
    for f in ("--anneal-epochs", "--tau-hold-frac", "--fused-r-kernel",
              "--curriculum-reanchor-levers", "--curriculum-min-stage-epochs",
              "--seg-chroma-boundary-weight",
              "--seed-islands", "--weight-entropy-penalty-lambda"):
        assert f not in fd, f"store_nothing_205 must not emit the crucible-only flag {f}"
    # --curriculum-plateau-windows must be emitted by NEITHER (v6.3 MAJOR-1: crucible dropped it too)
    assert "--curriculum-plateau-windows" not in fd
    assert fd["--softmax-temp-end"] == "0.05"
    assert fd["--tau-anneal-shape"] == "cosine"
    assert fd["--hosc-beta-end"] == "4.0"      # the β-pin is crucible-only; base keeps the family 4.0


# --------------------------------------------------------------------------
# #351 LawRef constant migration — VALUE-IDENTITY IS THE LAW (resolver path)
# --------------------------------------------------------------------------
def test_crucible_v6_lawref_resolved_path_is_byte_identical_to_literal_path():
    """The launch.sh byte-identity acceptance bar: the CONSUMED constants (τ_end / β-pin / LR-pin)
    are now COMPILED from LawRefs, so the emitted command must equal the command the pure-literal
    ``_CRUCIBLE_V6_DELTAS`` path would emit. Force the literal path by clearing the resolved delta
    dict (the trailing block then reads the module global) — the two commands must be identical."""
    import dataclasses as _dc

    cfg = _crucible_cfg()
    resolved_cmd = cfg.to_command("OUT")
    literal_path_cmd = _dc.replace(cfg, crucible_v6_deltas={}).to_command("OUT")
    assert resolved_cmd == literal_path_cmd


def test_crucible_v6_resolved_deltas_bitmatch_module_literals():
    """The resolved deltas the config consumes equal the sealed module literals — value AND type
    (int lr_anneal_epochs stays int, else str(1000.0) would shift the token)."""
    cfg = _crucible_cfg()
    for k, lit in wac._CRUCIBLE_V6_DELTAS.items():
        got = cfg.crucible_v6_deltas[k]
        assert got == lit and type(got) is type(lit), f"{k}: {got!r} != {lit!r}"


def test_crucible_v6_constants_manifest_schema_and_completeness():
    """The manifest carries every MIGRATED (consumed) constant with {value, equation_id, inputs+shas,
    ladder_class} — the constants_manifest.json content the launcher writes beside launch.sh."""
    from tac.witness_dsl.lawref_builtins import CRUCIBLE_V6_CONSUMED_LAWREFS

    man = _crucible_cfg().constants_manifest
    assert set(man) == set(CRUCIBLE_V6_CONSUMED_LAWREFS)
    for key, row in man.items():
        assert set(row) >= {"value", "equation_id", "ladder_class", "fallback_used", "inputs"}
        assert row["value"] == wac._CRUCIBLE_V6_DELTAS[key]
        for inp in row["inputs"]:
            assert set(inp) >= {"name", "kind", "value", "source", "sha256"}
            if inp["kind"] == "anchor":
                assert inp["sha256"] is not None  # anchor inputs carry their artifact sha


def test_crucible_v6_value_identity_guard_fails_closed_on_drift(monkeypatch):
    """The migration self-protection: if a LawRef ever resolves to a value != the sealed literal,
    ``derive_crucible_v6_config`` fails CLOSED (never silently emits a different launch config)."""
    drifted = dict(wac._CRUCIBLE_V6_DELTAS)
    drifted["hosc_beta_end"] = 9.5  # pretend the sealed literal disagrees with the LawRef's 10.0
    monkeypatch.setattr(wac, "_CRUCIBLE_V6_DELTAS", drifted)
    with pytest.raises(ValueError, match="value-identity violation"):
        wac.derive_crucible_v6_config(_GT_N600, num_pairs=600, epochs=3000)


def test_non_crucible_configs_have_empty_migration_fields():
    """The migration fields are crucible-only; every other config path leaves them empty (so no
    provenance-only field can leak into a non-crucible argv or manifest)."""
    for cfg in (
        wac.derive_sealed_205_config(_GT_N600, num_pairs=600, epochs=1000),
        wac.derive_store_nothing_205_config(_GT_N600, num_pairs=600, epochs=1000),
        wac.derive_config(_GT_N600, num_pairs=600, epochs=1000),
    ):
        assert cfg.crucible_v6_deltas == {}
        assert cfg.constants_manifest == {}
