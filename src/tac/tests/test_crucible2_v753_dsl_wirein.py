"""T5 CRUCIBLE-2 v7.5.3 DSL wire-in (OWED-BUILD, task #398 thread B).

The committed, reproducible standing form of the v7.5.3 fractal-synthesis typed-delta
(`fullstack_fractal_optimal_synthesis_20260710.md` §2/§3) authored AS a typed
`TypedWitnessConfig` (via `witness_autoconfig.derive_crucible_v753_config`):

  * VALIDATES clean AND its compiled argv PARSES through the trainer's REAL argparse (0 unknown
    flags) — the never-invent-flags gate, on BOTH pre-registered decision-rule branches.
  * The byte-diff v753-vs-v752 argv is EXACTLY the typed delta PER BRANCH (off ⇒ empty; on ⇒ the
    coherent rebalanced lane_carried allocation).
  * The Δ2/Δ3 A/B arms compose their EXACT flag deltas; Δ5 MC-finisher is ARGV-INERT (a TOOL).
  * The Δ4 10-rung ladder factories all EXIST in the DSL (no genuinely-missing stub owed).
  * The lever registry has NO NEW unmapped flag (the OWED `--out-tex-hidden` wire-in is mapped).
  * ADVERSARIAL (round-1 self-review): the emitted argv ⊆ the trainer argparse surface (dead-flag
    regression — introspect the REAL argparse, assert every emitted flag exists).

means != ends: config plumbing, NOT a score. Pointer contest-CPU 0.19110 UNMOVED; only a byte-closed
upstream/evaluate.py n600 row < 0.19110 moves it. The launch ships banked-R1 pose (d_pose 0.001610 →
0.127) as its floor; d_seg is the open axis.
"""
from __future__ import annotations

import pytest

from tac import witness_autoconfig as wac
from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl import lever_registry as LR
from tac.witness_dsl.typed_config import verify_launch_manifest

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
_IGNORE = {"--out-dir", "--gt-cache"}  # run-dir artifacts, not config semantics


def _flag_dict(typed):
    """The canonical emitted flag→value dict (captures store_true bools as True), sans run-dir keys."""
    fd = dict(typed.to_program().flag_dict())
    for k in _IGNORE:
        fd.pop(k, None)
    return fd


def _delta(a, b):
    """{flag: (a_val, b_val)} for every flag whose emitted value differs between typed configs a,b."""
    fa, fb = _flag_dict(a), _flag_dict(b)
    keys = set(fa) | set(fb)
    return {k: (fa.get(k, "∅"), fb.get(k, "∅")) for k in sorted(keys) if fa.get(k) != fb.get(k)}


# ─────────────────────────── 1. compiles + validates (both branches) ────────────────────────────
def test_v753_off_validates_and_parses_end_to_end():
    typed = wac.derive_crucible_v753_config(_GT, num_pairs=600, epochs=3000, trunk_basis="off")
    assert typed.name == "crucible_v753"
    assert typed.validate_program() == [], "v7.5.3 OFF-branch must be DSL-valid"
    ap = cd.build_real_trainer_parser()
    try:
        ap.parse_args(list(typed.to_program().compile_trainer_argv())[2:])
    except SystemExit as exc:  # pragma: no cover
        raise AssertionError(f"v753 OFF argv rejected by real argparse (rc={exc.code})") from exc


def test_v753_on_validates_and_parses_end_to_end():
    typed = wac.derive_crucible_v753_config(_GT, num_pairs=600, epochs=3000, trunk_basis="on")
    assert typed.name == "crucible_v753"
    assert typed.validate_program() == [], "v7.5.3 ON-branch must be DSL-valid"
    ap = cd.build_real_trainer_parser()
    try:
        ap.parse_args(list(typed.to_program().compile_trainer_argv())[2:])
    except SystemExit as exc:  # pragma: no cover
        raise AssertionError(f"v753 ON argv rejected by real argparse (rc={exc.code})") from exc


def test_v753_all_arms_on_validate_and_parse():
    """The FULL arm set (on-branch + both A/B arms + lane-band-training + mc-finisher) still validates
    clean AND parses — no arm introduces an invented/type-incompatible flag."""
    typed = wac.derive_crucible_v753_config(
        _GT, num_pairs=600, epochs=3000, trunk_basis="on", texture_trunk=True,
        out_tex_hidden=16, analytic_lane_band_training=True, mc_finisher_terminal=True)
    assert typed.validate_program() == []
    ap = cd.build_real_trainer_parser()
    ap.parse_args(list(typed.to_program().compile_trainer_argv())[2:])  # raises SystemExit on any unknown flag


# ─────────────────────────── 2. byte-diff v753−v752 == the typed delta (per branch) ──────────────
def test_v753_off_argv_byte_identical_to_v752_self_orient_off():
    """Δ1 OFF-arm (the §2 pre-registered default): the default v753 config's argv is EXACTLY the GO'd
    v7.5.2 self-orient-OFF launch — the delta is registered levers (metadata), never argv."""
    v753 = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    v752 = wac.derive_crucible_v752_config(_GT, num_pairs=600, self_orient=False)
    assert _delta(v753, v752) == {}, "v753(off) argv must be byte-identical to v752(self_orient=False)"


def test_v753_on_argv_delta_is_exactly_the_rebalanced_lane_carried_allocation():
    """Δ1 ON-arm: the ONLY argv delta vs v7.5.2(self_orient=True) is the coherent rebalanced
    lane_carried allocation — freq_along 6→26 AND the regime-coupled persistence/logit flags."""
    v753 = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="on")
    v752 = wac.derive_crucible_v752_config(_GT, num_pairs=600, self_orient=True)
    d = _delta(v753, v752)
    assert set(d) == {"--freq-along", "--persistence-classes", "--logit-adjust-classes"}, d
    assert d["--freq-along"] == (26.0, 6.0)
    assert d["--persistence-classes"] == ("auto", "3")
    assert d["--logit-adjust-classes"] == ("all", "3")


def test_v753_texture_trunk_arm_delta_is_exactly_the_trunk_flags():
    base = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    arm = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off", texture_trunk=True)
    d = _delta(arm, base)
    assert set(d) == {"--texture-trunk", "--texture-trunk-band-hi",
                      "--texture-trunk-annulus-power", "--texture-trunk-coeff-scale"}, d
    assert d["--texture-trunk"] == (True, "∅")


def test_v753_out_tex_hidden_arm_delta_is_exactly_the_owed_flag():
    base = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    arm = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off", out_tex_hidden=16)
    d = _delta(arm, base)
    assert d == {"--out-tex-hidden": (16, "∅")}, d


def test_v753_lane_band_training_arm_flips_start_epoch_to_zero_trained_in():
    """Δ3: the trained-in lever's only delta over the default (which already carries the render-band
    at start-epoch 500) is start-epoch → 0 (the band participates in the trained render from ep0),
    reusing the LANDED lane-band flags (never-invent-flags)."""
    base = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    arm = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off",
                                          analytic_lane_band_training=True)
    d = _delta(arm, base)
    assert d.get("--lane-band-start-epoch") == (0, base.to_program().flag_dict().get("--lane-band-start-epoch")), d
    assert d["--lane-band-start-epoch"][0] == 0


def test_v753_mc_finisher_terminal_is_argv_inert_but_recorded_in_purpose():
    """Δ5: the MC finisher is a TOOL, not a Lever/flag — selecting it changes NO argv, only the purpose."""
    base = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    arm = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off", mc_finisher_terminal=True)
    assert _delta(arm, base) == {}, "mc_finisher_terminal must not perturb the launch argv (it is a TOOL)"
    assert "mc_finisher_terminal" in arm.purpose
    assert wac._CRUCIBLE_V753_MC_FINISHER_TOOL in arm.purpose


# ─────────────────────────── 3. decision-rule branches both compile + are encoded ────────────────
def test_v753_both_decision_rule_branches_compile():
    for tb in ("off", "on"):
        typed = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis=tb)
        assert typed.validate_program() == [], f"branch {tb!r} must validate"


def test_v753_default_branch_is_the_pre_registered_off_arm():
    """The default (no trunk_basis) MUST be the OFF-arm per the §2 pre-registration — the builder never
    hardcodes the ON winner."""
    default = wac.derive_crucible_v753_config(_GT, num_pairs=600)
    off = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    assert _delta(default, off) == {}
    assert "off" in default.purpose and "OFF-arm" in default.purpose


def test_v753_trunk_basis_decision_rule_is_documented_and_outcomes_bounded():
    rule = wac._CRUCIBLE_V753_TRUNK_BASIS_DECISION_RULE
    assert "owed16v2" in rule and "freq_along 26" in rule and "DEFAULT='off'" in rule
    assert wac._CRUCIBLE_V753_TRUNK_BASIS_OUTCOMES == ("off", "on")


def test_v753_invalid_trunk_basis_is_refused():
    with pytest.raises(ValueError, match="trunk_basis must be one of"):
        wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="maybe")


def test_v753_negative_out_tex_hidden_is_refused():
    with pytest.raises(ValueError, match="out_tex_hidden must be >= 0"):
        wac.derive_crucible_v753_config(_GT, num_pairs=600, out_tex_hidden=-1)


# ─────────────────────────── 4. DSL validate fail-closes on an invented flag ─────────────────────
def test_v753_dsl_validate_fail_closes_on_invented_flag():
    """A composed lever emitting a flag the trainer argparse does NOT expose is REFUSED by
    validate_program (the never-invent-flags law at the DSL layer)."""
    typed = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    bad_lever = cd.Lever("invented", overrides={"--this-flag-does-not-exist": True})
    poisoned = typed.model_copy(update={"levers": typed.levers + (
        __import__("tac.witness_dsl.typed_config", fromlist=["TypedLever"]).TypedLever(
            name=bad_lever.name, overrides=dict(bad_lever.overrides)),)})
    viol = poisoned.validate_program()
    assert any("INVENTED FLAG" in v for v in viol), viol


# ─────────────────────────── 5. lever registry completeness — no new unmapped ────────────────────
def test_v753_owed_wire_in_out_tex_hidden_is_mapped_not_a_new_gap():
    """The OWED `--out-tex-hidden` trainer flag is HELD by the DSL (OutTexHidden factory) ⇒ it is
    MAPPED, not a new unmapped gap. The lane-band-training mode invents NO new trainer flag."""
    c = LR.completeness()
    assert "--out-tex-hidden" in c.mapped
    assert "--out-tex-hidden" not in c.unmapped
    assert c.stale == [], f"no DSL-emitted flag may be absent from the trainer (drift): {c.stale}"


def test_v753_new_dsl_factories_exist_and_emit_real_flags():
    assert hasattr(cd, "OutTexHidden") and hasattr(cd, "AnalyticLaneBandTraining")
    real = cd.real_trainer_flags()
    for lev in (cd.OutTexHidden(16), cd.AnalyticLaneBandTraining()):
        for flag in lev.overrides:
            assert flag in real, f"{lev.name} emits {flag} which is NOT in the trainer argparse"


# ─────────────────────────── 6. Δ4 ladder — every rung factory exists (no stub owed) ─────────────
def test_v753_ladder_rung_factories_all_exist_in_dsl():
    """Δ4: every one of the 11 rungs + operator-GO maps to a REAL curriculum_dsl factory (verify each
    exists; emit_stub_lever is owed ONLY for a genuinely-missing one — none are today, per #397)."""
    ladder = wac.crucible_v753_ladder()
    assert len(ladder) == 12, f"expected 11 rungs + operator-GO, got {len(ladder)}"
    missing = [(label, fac) for (label, fac, _note) in ladder if not hasattr(cd, fac)]
    assert missing == [], f"genuinely-missing ladder factories (emit_stub_lever owed): {missing}"
    # each factory actually constructs a Lever (or composite) — it is a real DSL surface, not a name.
    facs = LR.lever_factories()
    for _label, fac, _note in ladder:
        assert fac in facs, f"{fac} is not an AST-discovered DSL lever factory"


# ─────────────────────────── 6b. CHROMA add-back rung (LEVER-4c / #276) ───────────────────────────
def test_v753_chroma_rung_is_registered_in_the_ladder():
    """The chroma add-back A/B is a registered v7.5.3 ladder rung mapping to the SegChromaBoundary
    DSL factory (never a new/duplicate factory — the existing LEVER-4c surface is REUSED)."""
    ladder = wac.crucible_v753_ladder()
    chroma = [(lbl, fac, note) for (lbl, fac, note) in ladder if fac == "SegChromaBoundary"]
    assert len(chroma) == 1, f"expected exactly ONE chroma rung (no duplicate factory), got {chroma}"
    lbl, _fac, note = chroma[0]
    assert lbl == "chroma_annulus_addback_ab"
    # the note names the DOF-vs-add-back distinction so a reviewer cannot read it as a settled GAIN.
    assert "UNMEASURED" in note and "S5-N10" in note


def test_v753_chroma_rung_factory_emits_only_real_trainer_flags():
    """SegChromaBoundary(weight,margin_band,start_epoch) emits ONLY real levelset-trainer flags
    (never-invent-flags; the trainer chroma-routing wiring ALREADY exists — no new flag built)."""
    real = cd.real_trainer_flags()
    lev = cd.SegChromaBoundary(weight=0.05, margin_band=1.0, start_epoch=0)
    assert lev.name == "seg_chroma_boundary"
    for flag in lev.overrides:
        assert flag in real, f"SegChromaBoundary emits {flag} NOT in the trainer argparse"
    assert set(lev.overrides) == {
        "--seg-chroma-boundary-weight",
        "--seg-chroma-boundary-margin-band",
        "--seg-chroma-boundary-start-epoch",
    }


def test_v753_chroma_rung_is_registered_off_zero_argv_delta_in_default_config():
    """BYTE-IDENTITY: the chroma rung is registered-OFF — the default v7.5.3(off) config emits ZERO
    chroma-MATCH argv (the annulus match term stays inactive), so adding the rung did NOT perturb the
    launch. (The inherited base's own --seg-chroma-boundary-* pins are a SEPARATE, sealed concern —
    this test asserts the ladder rung itself composes nothing into the default.)"""
    v753 = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    # no lever named seg_chroma_boundary is composed by the registered-off rung.
    assert "seg_chroma_boundary" not in {lv.name for lv in v753.levers}


def test_v753_chroma_addback_decision_rule_is_preregistered_and_byte_matched():
    """The chroma A/B carries a PRE-REGISTERED decision rule + noise-floor (P2) discipline, states its
    0-byte BYTE-MATCHED property, names the ADVISORY_evaluator obligation (d_seg primary / d_pose only
    via 2x2 block-mean chroma), and does NOT hardcode a winner (default stays OFF)."""
    rule = wac._CRUCIBLE_V753_CHROMA_ADDBACK_DECISION_RULE
    assert "BYTE-MATCHED" in rule
    assert "noise floor" in rule and "single-seed" in rule.lower()
    assert "PAYS" in rule and "wash" in rule and "WORSE" in rule  # the 3-way outcome
    assert "DEFAULT stays OFF" in rule
    # verdict-scope discipline: a wash is FORMULATION-scoped, never a paradigm kill of the DOF.
    assert "FORMULATION" in rule


def test_v753_chroma_rung_factory_validates_a_fireable_arm_end_to_end():
    """A fired chroma arm (the A/B ON arm) composes into a v7.5.3 program whose compiled argv PARSES
    through the REAL trainer argparse (0 unknown flags) — the rung is genuinely FIREABLE, not a name."""
    typed = wac.derive_crucible_v753_config(_GT, num_pairs=600, epochs=3000, trunk_basis="off")
    prog = typed.to_program().with_lever(cd.SegChromaBoundary(weight=0.05, margin_band=1.0, start_epoch=0))
    argv = list(prog.compile_trainer_argv())
    assert "--seg-chroma-boundary-weight" in argv
    ap = cd.build_real_trainer_parser()
    try:
        ap.parse_args(argv[2:])
    except SystemExit as exc:  # pragma: no cover
        raise AssertionError(f"fired chroma arm argv rejected by real argparse (rc={exc.code})") from exc


def test_v753_chroma_rung_factory_fails_closed_on_bad_args():
    """SegChromaBoundary fail-closes on a negative weight, a non-positive margin_band (empty annulus =
    silent no-op), and a negative start_epoch — the DSL never composes a silently-inert chroma arm."""
    import pytest as _pt
    with _pt.raises(ValueError):
        cd.SegChromaBoundary(weight=-0.1)
    with _pt.raises(ValueError):
        cd.SegChromaBoundary(weight=0.05, margin_band=0.0)
    with _pt.raises(ValueError):
        cd.SegChromaBoundary(weight=0.05, start_epoch=-5)


def test_v753_chroma_rung_adds_no_new_unmapped_flag_registry_unchanged():
    """Registry completeness: the chroma flags are HELD by the DSL (mapped, not a new gap) and the rung
    introduces NO stale/unmapped drift — the trainer chroma-routing wiring pre-exists."""
    c = LR.completeness()
    for flag in ("--seg-chroma-boundary-weight", "--seg-chroma-boundary-margin-band",
                 "--seg-chroma-boundary-start-epoch"):
        assert flag in c.mapped, f"{flag} must be DSL-mapped (SegChromaBoundary)"
        assert flag not in c.unmapped
    assert c.stale == [], f"no DSL-emitted flag may be absent from the trainer (drift): {c.stale}"


def test_v753_chroma_twin_is_luma_invariant_orthogonal_to_luma_levers():
    """NO-FAKE property: the reference twin chroma := rgb − BT.601-luma is INVARIANT to an achromatic
    (equal-channel) luma shift ⇒ the chroma add-back is provably ORTHOGONAL to every luma lever."""
    import numpy as np
    from tac.boundary_math import chroma_boundary_match as cbm
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 255, size=(4, 5, 3))
    shift = rng.uniform(-30, 30, size=(4, 5, 1))  # per-pixel achromatic luma shift, replicated to 3 ch
    base = cbm.bt601_chroma(rgb)
    shifted = cbm.bt601_chroma(rgb + shift)  # broadcast adds the SAME value to R,G,B
    assert np.allclose(base, shifted, atol=1e-4), "bt601_chroma must be invariant to achromatic shifts"


def test_v753_chroma_twin_annulus_mask_selects_only_below_band():
    """The θ-independent fragile-annulus mask selects ONLY margin<band pixels (STRICT <, mirroring the
    trainer) — the chroma-match term is confined to the knife-edge where chroma has its d_seg power."""
    import numpy as np
    from tac.boundary_math import chroma_boundary_match as cbm
    margin = np.array([[0.0, 0.5, 1.0], [1.5, 0.99, 2.0]])
    m = cbm.annulus_mask(margin, band=1.0)
    assert m.tolist() == [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]  # 1.0 is NOT < 1.0 (strict)


def test_v753_chroma_rung_equations_are_registered():
    """The DOF-existence equation and the add-back hinge equation are both registered (triality eqs leg
    already landed for #276; the rung's own ΔS row is deferred-to-first-measured-A/B)."""
    from tac.canonical_equations import query_equations
    ids = {e.equation_id for e in query_equations()}
    assert "chroma_decides_lane_and_movable_at_annulus_v1" in ids  # the MEASURED DOF (ablation)
    assert "chroma_boundary_annulus_match_hinge_v1" in ids          # the add-back mechanism (ΔS owed)


def test_v753_chroma_decision_rule_names_the_evaluator_pose_obligation():
    """The A/B plan cites the ADVISORY_evaluator obligation: chroma reaches d_pose ONLY through the 2x2
    block-mean chroma (weak, incidental) — so the add-back is optimized for d_seg FIRST, pose reported."""
    # the plan text (comment header on the rule constant) is the operator-facing surface; the rule
    # constant itself must at minimum carry the d_seg-primary framing.
    rule = wac._CRUCIBLE_V753_CHROMA_ADDBACK_DECISION_RULE
    assert "d_seg" in rule.lower()
    assert "realized" in rule.lower() and "n600" in rule.lower()  # measured through-R, at scale


def test_v753_ladder_is_registered_off_not_composed_into_the_default_config():
    """Δ4 discipline: the ladder is a duty-to-measure QUEUE (registered-OFF), NOT composed into the
    default config — the default config's levers carry NONE of the ladder-only factory names beyond
    what v7.5.2 already composed (argv-inert queue)."""
    v753 = wac.derive_crucible_v753_config(_GT, num_pairs=600, trunk_basis="off")
    v752 = wac.derive_crucible_v752_config(_GT, num_pairs=600, self_orient=False)
    # same lever NAME set as v752(off) — the ladder adds registered-but-unfired factories, not levers.
    assert {lv.name for lv in v753.levers} == {lv.name for lv in v752.levers}


# ─────────────────────────── 7. launcher path + provenance manifest ──────────────────────────────
def test_v753_launch_config_resolves_and_manifest_verifies():
    cfg = wac.compile_crucible_v753_launch_config(_GT, num_pairs=600, epochs=3000)
    assert cfg.name == "crucible_v753"
    assert cfg.dsl_program_manifest["program_name"] == "crucible_v753"
    argv = cfg.typed.to_program().compile_trainer_argv()
    emitted = sorted({f for f, _ in wac._crucible_v7_argv_pairs(argv)})
    ok, detail = verify_launch_manifest(cfg.dsl_program_manifest, emitted)
    assert ok, f"v753 dsl_program_manifest must verify: {detail}"


def test_v753_derive_named_config_resolves_not_proven_base():
    import tools.launch_witness_run as L
    cfg = L.derive_named_config("crucible_v753", _GT, num_pairs=8, epochs=3000, overfit=True)
    assert cfg.name == "crucible_v753", "the launcher must resolve crucible_v753 (never fall through)"


# ─────────────────────────── 8. ADVERSARIAL round-1: dead-flag regression ────────────────────────
def test_v753_emitted_argv_is_subset_of_real_argparse_surface_all_arms():
    """ROUND-1 SELF-REVIEW (dead-flag class, CLAUDE.md 'NEVER invent CLI flags'): introspect the REAL
    trainer argparse and assert EVERY emitted flag (across every arm combination) exists on it. Catches
    an invented/renamed flag that `validate_program` might (in principle) miss."""
    real = cd.real_trainer_flags()
    combos = [
        dict(trunk_basis="off"),
        dict(trunk_basis="on"),
        dict(trunk_basis="off", texture_trunk=True),
        dict(trunk_basis="off", out_tex_hidden=16),
        dict(trunk_basis="off", analytic_lane_band_training=True),
        dict(trunk_basis="on", texture_trunk=True, out_tex_hidden=8,
             analytic_lane_band_training=True, mc_finisher_terminal=True),
    ]
    for kw in combos:
        typed = wac.derive_crucible_v753_config(_GT, num_pairs=600, **kw)
        emitted = set(typed.to_program().flag_dict())
        invented = {f for f in emitted if f not in real}
        assert invented == set(), f"arm {kw} emits flags not in the trainer argparse: {invented}"
