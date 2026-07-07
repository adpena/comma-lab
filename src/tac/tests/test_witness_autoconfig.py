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
                "--muon-warm-start-momentum", "--muon-lr-final-frac", "--closed-loop-control"}
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
