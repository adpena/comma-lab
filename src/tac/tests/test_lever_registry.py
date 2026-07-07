"""Consumer + regression guard for the DSL completeness surface (so it is not an orphan).

Locks the adversarial-review fixes (2026-07-06): M2 tuple-factory discovery, M3 stale-noise
removal, L7 stub guarding, and determinism.
"""
from __future__ import annotations

from tac.witness_dsl import lever_registry as LR


def test_factories_discovered_include_tuple_composite():
    facs = LR.lever_factories()
    # A sample of the known direct factories...
    for name in ("Muon", "LanePrior", "AnalyticLaneRenderBand", "MicroBatch", "UniWARD"):
        assert name in facs, f"{name} factory not discovered"
    # ...and the M2 fix: the tuple-returning composite must NOT vanish, and must carry the
    # flags of the sub-factories it delegates to.
    assert "DM1Minimal" in facs, "M2 regression: tuple[Lever,Lever] composite dropped"
    assert facs["DM1Minimal"], "DM1Minimal should carry its delegated flags, not be empty"


def test_program_constructors_excluded_from_levers():
    facs = LR.lever_factories()
    assert "sealed_205_program" not in facs
    assert "openpilot_seeded_opening" not in facs


def test_stale_is_real_drift_not_reporter_noise():
    # M3 fix: stale is DSL-EMITTED flags absent from the trainer, so launcher/daemon/`--no-`
    # artifacts must NOT appear. On the current tree there is no real drift → empty.
    c = LR.completeness()
    for noise in ("--no-", "--label", "--log", "--min-free-gb", "--rss-cap-mb"):
        assert noise not in c.stale, f"M3 regression: reporter noise {noise} in stale"


def test_completeness_reports_the_gap():
    c = LR.completeness()
    assert c.trainer_total > 100          # the trainer really has ~235 flags
    assert c.unmapped, "the DSL does not yet cover every flag — the gap must be visible"
    assert 0.0 <= c.coverage_frac <= 1.0
    # every mapped/unmapped flag is a real flag, never the --no- artifact
    assert all(f.startswith("--") and not f.startswith("--no-") for f in c.mapped + c.unmapped)


def test_emit_stub_lever_valid_and_guarded():
    import ast as _ast
    stub = LR.emit_stub_lever("--seg-focal-gamma")
    _ast.parse(stub)                       # must be valid Python
    assert "def SegFocalGamma(" in stub
    for bad in ("--no-x", "not-a-flag", "--"):
        try:
            LR.emit_stub_lever(bad)
            raise AssertionError(f"emit_stub_lever accepted {bad!r}")
        except ValueError:
            pass


def test_deterministic():
    assert LR.lever_factories() == LR.lever_factories()
    c1, c2 = LR.completeness(), LR.completeness()
    assert (c1.mapped, c1.unmapped, c1.stale) == (c2.mapped, c2.unmapped, c2.stale)


def test_r5_async_def_included_in_func_defs():
    # r5 fix: an ``async def`` factory returning a Lever must be discoverable (previously
    # only ast.FunctionDef was scanned). Assert the tuple + behavior on a synthetic module.
    import ast as _ast
    assert _ast.AsyncFunctionDef in LR._FUNC_DEFS and _ast.FunctionDef in LR._FUNC_DEFS
    src = "async def Foo() -> Lever:\n    return Lever('foo', overrides={'--foo': True})\n"
    tree = _ast.parse(src)
    node = tree.body[0]
    assert isinstance(node, _ast.AsyncFunctionDef)
    assert LR._returns_lever(node) and LR._constructs(node, "Lever")  # both duck-type on async


def test_r5_anno_is_lever_handles_stringized_composite():
    # r5 fix: a FULLY stringized composite annotation must be recognized, not just bare "Lever".
    import ast as _ast
    def anno(src):
        return _ast.parse(f"def f() -> {src!r}: ...").body[0].returns  # stringized (a str Constant)
    assert LR._anno_is_lever(anno("Lever")) is True
    assert LR._anno_is_lever(anno("tuple[Lever, Lever]")) is True
    assert LR._anno_is_lever(anno("dict[str, Lever]")) is False       # still rejected
    assert LR._anno_is_lever(anno("not valid python {{{")) is False   # bad string → fail closed


def test_r5_completeness_bad_path_raises_clear_error():
    import pytest as _pytest
    with _pytest.raises(FileNotFoundError, match="not a file"):
        LR.completeness("/no/such/trainer_file_zzz.py")
    # the normal default path still works (no raise)
    assert LR.completeness().trainer_total > 100


def test_332_deorphaned_levers_discovered_compose_and_validate():
    # #332 de-orphaning wave: the 7 designed-but-orphaned levers are now DSL factories that
    # (a) are AST-discovered, (b) reference only REAL trainer flags, (c) compose into a
    # WitnessProgram that validate()s clean and compiles to argv — the DSL GENERATES the config.
    from tac.witness_dsl import curriculum_dsl as cd
    facs = LR.lever_factories()
    new = ("SeedIslandEased", "EventTriggeredCurriculum", "EikonalViscosity", "AmplifyIsland",
           "BoundaryDistance", "SegFocalGamma", "AdamBeta2")
    for n in new:
        assert n in facs, f"#332 lever {n} not discovered by the registry"
        assert facs[n], f"#332 lever {n} emits no flags"
    prog = cd.BASELINE
    for fac in (cd.SeedIslandEased(), cd.EventTriggeredCurriculum(), cd.EikonalViscosity(),
                cd.AmplifyIsland(), cd.BoundaryDistance(), cd.SegFocalGamma(), cd.AdamBeta2()):
        prog = prog.with_lever(fac)
    assert prog.validate() == [], "composed de-orphaned levers must reference only real flags"
    argv = prog.compile_trainer_argv()
    assert "--seed-island-eased" in argv and "--curriculum-nucleus-guard" in argv


def test_332_coverage_rose_from_deorphaning():
    # The de-orphaning wave raised DSL coverage; assert it is meaningfully above the pre-wave 34%.
    c = LR.completeness()
    assert c.coverage_frac >= 0.38, f"coverage {c.coverage_frac} below post-#332 floor"
    assert c.stale == [], "no DSL-emitted flag may be absent from the trainer"


# ---------------------------------------------------------------------------------------------
# --dsl-lever composability predicate + typed refusal (CLASS-fix, review 2026-07-06:
# Muon(start_epoch) crashed the config generator with TypeError; DM1Minimal with
# AttributeError on .overrides — both raw tracebacks after passing the launcher surface).
# ---------------------------------------------------------------------------------------------
def test_composability_predicate_partitions_factories():
    from tac.witness_dsl.curriculum_dsl import Lever
    comp = LR.name_composable_levers()
    assert comp == tuple(sorted(comp)), "deterministic sorted enumeration"
    # the two known non-composables are EXCLUDED (the crash family)
    assert "Muon" not in comp, "Muon requires explicit args — must not be name-composable"
    assert "DM1Minimal" not in comp, "DM1Minimal returns tuple[Lever, Lever] — must not be name-composable"
    # known composables are INCLUDED
    for name in ("SeedIslandEased", "EventTriggeredCurriculum", "EikonalViscosity",
                 "StiefelW", "CodeSpectralEntropy", "UniWARD", "MicroBatch"):
        assert name in comp, f"{name} should be name-composable"
    # every enumerated name actually resolves to a single Lever (the predicate is honest)
    for name in comp:
        assert isinstance(LR.resolve_composable_lever(name), Lever)


def test_resolve_composable_lever_typed_refusals():
    import pytest as _pytest
    with _pytest.raises(LR.LeverCompositionError, match="requires explicit args"):
        LR.resolve_composable_lever("Muon")
    with _pytest.raises(LR.LeverCompositionError, match="returns tuple"):
        LR.resolve_composable_lever("DM1Minimal")
    with _pytest.raises(LR.LeverCompositionError, match="not a curriculum_dsl Lever factory"):
        LR.resolve_composable_lever("NotARealLever")
    # the refusal message is operator-actionable: it lists the composable set
    try:
        LR.resolve_composable_lever("Muon")
    except LR.LeverCompositionError as exc:
        assert "SeedIslandEased" in str(exc) and "composable" in str(exc)
    # LeverCompositionError subclasses ValueError (pre-existing except-ValueError callers keep working)
    assert issubclass(LR.LeverCompositionError, ValueError)


def _render_argv(pairs) -> list[str]:
    """Render (flag, value) pairs the way the launcher does: bare flag when value is None."""
    argv: list[str] = []
    for flag, val in pairs:
        argv.append(flag)
        if val is not None:
            argv.append(str(val))
    return argv


def test_every_composable_lever_parses_through_real_trainer_argparse():
    """THE CLASS TEST (round-2 review demand): compose EVERY composable lever onto the sealed
    config and PARSE the resulting argv through the trainer's REAL argparse (the trainer's own
    add_argument statements, executed — build_real_trainer_parser). Any lever emitting a
    wrong-typed value (the EikonalViscosity bare-True-on-type=float family) or an invented flag
    fails HERE at CI time, not after daemon spawn."""
    import dataclasses as dc

    from tac import witness_autoconfig as wac
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    ap = build_real_trainer_parser()
    cfg = wac.derive_sealed_205_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz", num_pairs=600, epochs=1000)
    # control: the un-levered sealed base itself must parse
    base_argv = _render_argv(cfg.to_trainer_flags("OUT"))
    try:
        ap.parse_args(base_argv)
    except SystemExit as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(f"sealed base argv failed the real trainer argparse: rc={exc.code}") from exc
    # every composable lever, one at a time (per-lever attribution of any failure)
    for name in LR.name_composable_levers():
        argv = _render_argv(dc.replace(cfg, dsl_levers=(name,)).to_trainer_flags("OUT"))
        try:
            ap.parse_args(argv)
        except SystemExit as exc:
            raise AssertionError(
                f"--dsl-lever {name} emitted an argv the REAL trainer argparse rejects "
                f"(rc={exc.code}) — the EikonalViscosity/Muon crash class") from exc


def test_build_real_trainer_parser_matches_regex_flag_surface():
    # the built parser and the regex helper must agree on the flag universe (extraction-drift guard)
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser, real_trainer_flags
    ap = build_real_trainer_parser()
    built = {opt for a in ap._actions for opt in a.option_strings if opt.startswith("--")
             and not opt.startswith("--no-")}
    regex = set(real_trainer_flags())
    assert regex <= built, f"parser missing regex-visible flags: {sorted(regex - built)[:10]}"
