# SPDX-License-Identifier: MIT
"""ddm_pt2 — tests for the factory<->instance JOIN and the four ported TR1 levers.

Every assertion here is written so that it FAILS if the thing it names stops being true; the
round-trip and the negative controls exist specifically because a control that cannot fail is
not a control (and because ``ddm_lr2`` round 2 was reset by exactly that mistake).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tac.witness_dsl.lever_name_join import (
    AMBIGUOUS,
    RESOLVED_LITERAL,
    RESOLVED_TEMPLATE,
    UNRESOLVED_NOT_A_FACTORY,
    dynamic_name_factories,
    factory_name_forms,
    resolve_instance,
    resolve_instances,
)
from tac.witness_dsl.pt2_ported_levers_20260803 import (
    PT2_PORTED_LEVERS,
    TRAINER_RELPATH,
    lever_fisher_density,
    lever_head_natural_grad,
    lever_seg_focal_gamma,
    lever_tau_softplus_tau,
)
from tac.witness_dsl.spec_tr1_renderer_20260728 import (
    TR1RendererProgramV1,
    default_t1_smoke_program,
    lever_ema_decay,
    lever_seg_margin_weight,
    lever_seg_physics,
    lever_token_grid,
    lever_token_init,
    lever_variant,
)
from tac.witness_dsl.spec_tr1_renderer_20260728 import (
    trainer_declared_flags as tr1_declared_flags,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


# ── the JOIN ────────────────────────────────────────────────────────────────────────────────
def test_join_forms_are_nonempty_and_typed():
    forms = factory_name_forms()
    assert len(forms) > 100, "an empty/near-empty scan is VACUOUS, never a pass"
    kinds = {f.kind for f in forms}
    assert kinds <= {RESOLVED_LITERAL, RESOLVED_TEMPLATE}
    assert any(f.kind == RESOLVED_TEMPLATE for f in forms), "f-string names must be captured"


@pytest.mark.parametrize(
    ("factory", "args"),
    [(lever_variant, ("lotto",)), (lever_token_grid, (16, 4)), (lever_seg_physics, ("ce",)),
     (lever_token_init, ("solve_project",)), (lever_seg_margin_weight, (1.0,)),
     (lever_ema_decay, (0.99986667,))],
)
def test_join_roundtrip_positive_control(factory, args):
    """POSITIVE CONTROL: construct a real Lever, resolve its name, land back on ITS factory.

    This can fail three ways — a template-extraction bug, a precedence bug, or a missing
    ``f"...".rstrip()`` unwrap (``lever_ema_decay`` is the live instance of the third).
    """
    lv = factory(*args)
    r = resolve_instance(lv.name)
    assert r.resolved, f"{lv.name} did not resolve uniquely: {r.to_dict()}"
    assert r.factory == factory.__name__, f"{lv.name} -> {r.factory}, expected {factory.__name__}"


def test_join_literal_precedence_beats_an_overlapping_template():
    """``lever_seg_physics`` emits ``f"tr1_seg_{form}"`` whose regex also matches the LITERAL
    ``tr1_seg_margin_weight``. Literal-first precedence must make that exact, not ambiguous."""
    forms = factory_name_forms()
    tmpl = [f for f in forms if f.factory == "lever_seg_physics" and f.kind == RESOLVED_TEMPLATE]
    assert tmpl, "the overlapping template must exist for this test to mean anything"
    assert re.match(tmpl[0].pattern, "tr1_seg_margin_weight"), "premise: the template DOES match"
    r = resolve_instance("tr1_seg_margin_weight")
    assert r.kind == RESOLVED_LITERAL and r.factory == "lever_seg_margin_weight"


def test_join_negative_control_unknown_name_does_not_resolve():
    for bad in ("zz_definitely_not_a_lever", "", "lever_token_grid"):
        r = resolve_instance(bad)
        assert r.kind == UNRESOLVED_NOT_A_FACTORY, f"{bad!r} should not resolve: {r.to_dict()}"


def test_join_reports_dynamic_name_factories_as_the_honest_residue():
    """Factories whose name is neither literal nor f-string cannot be joined statically. They must
    be ENUMERABLE, so the join's denominator is visible rather than silently omitted."""
    dyn = dynamic_name_factories()
    assert isinstance(dyn, tuple)
    for mod, fac in dyn:
        assert mod.endswith(".py") and fac


def test_join_resolves_the_live_ticket_names_it_can_and_names_the_rest():
    """The two names ddm_pt2 MEASURED as non-DSL constructions must resolve to NOT_A_FACTORY.

    That is the config-orphan surface, and a fallback that swallowed them would hide it.
    """
    forms = factory_name_forms()
    for orphan in ("tr1_coupling_field_only", "qa86_live_config_pin"):
        r = resolve_instance(orphan, forms)
        assert r.kind == UNRESOLVED_NOT_A_FACTORY, (
            f"{orphan} is built inline outside the DSL; the join must surface it, got {r.to_dict()}")


def test_join_flags_a_genuinely_ambiguous_name_instead_of_first_matching():
    """Ambiguity must be REPORTED. Built from two synthetic forms so the assertion is about the
    resolver's behaviour, not about today's tree happening to be unambiguous."""
    from tac.witness_dsl.lever_name_join import FactoryNameForm

    forms = (
        FactoryNameForm("a.py", "fa", RESOLVED_TEMPLATE, None, r"^zz_(?:.+)$"),
        FactoryNameForm("b.py", "fb", RESOLVED_TEMPLATE, None, r"^zz_x(?:.+)$"),
    )
    r = resolve_instance("zz_xyz", forms)
    assert r.kind == AMBIGUOUS and len(r.factory_refs) == 2 and not r.resolved


def test_every_literal_lever_name_self_resolves_except_two_measured_duplicates():
    """PACKAGE-WIDE self-resolution census, with its denominator stated.

    MEASURED 2026-08-03: 151 literal names, 147 self-resolve uniquely. The 4 failing rows are 2
    NAMES emitted by two different config-compilers each — a genuine duplicate-name defect in the
    retired tree, not a resolver bug, and the join correctly reports AMBIGUOUS rather than picking
    one. Pinning the exact allowlist means a NEW collision (the class this arm's own first draft
    hit) fails this test instead of quietly joining the noise.
    """
    forms = factory_name_forms()
    lits = [f for f in forms if f.kind == RESOLVED_LITERAL]
    assert len(lits) > 100, "a near-empty literal scan would make this test VACUOUS"
    offenders = set()
    for f in lits:
        r = resolve_instance(f.literal, forms)
        if not (r.resolved and r.factory == f.factory):
            offenders.add(f.literal)
    assert offenders == {"c2_component_wallclock_telemetry", "c2_speed_stack"}, (
        f"self-resolution census changed: {sorted(offenders)} (expected exactly the 2 measured "
        f"duplicate names emitted by both compile_c2_surgical_warm_launch_config and "
        f"compile_v9c3_duty_ab_config)")


def test_resolve_instances_is_batched_and_equals_the_scalar_path():
    forms = factory_name_forms()
    names = ["tr1_seg_margin_weight", "tr1_token_grid_D16_c4", "zz_nope"]
    assert [r.to_dict() for r in resolve_instances(names, forms)] == \
           [resolve_instance(n, forms).to_dict() for n in names]


# ── the PORT ────────────────────────────────────────────────────────────────────────────────
def test_ported_module_declares_the_live_trainer():
    """ddm_lr2 §1: without this the registry files these levers under the RETIRED trainer by
    silent default and no TR1-scoped duty query can ever surface them."""
    assert TRAINER_RELPATH == "experiments/train_tr1_partition_renderer_mlx.py"
    assert (_REPO_ROOT / TRAINER_RELPATH).is_file()


@pytest.mark.parametrize(
    ("factory", "args"),
    [(lever_seg_focal_gamma, (2.0,)), (lever_fisher_density, (1.0, "model")),
     (lever_fisher_density, (0.5, "gt")), (lever_head_natural_grad, (1e-3,)),
     (lever_head_natural_grad, (1e-2,)), (lever_tau_softplus_tau, (0.15,))],
)
def test_ported_levers_compile_through_the_fail_closed_validate(factory, args):
    """The never-invent-flags gate is the real check: ``compile_trainer_argv`` AST-scans the TR1
    trainer's argparse and RAISES on any flag it does not declare. A port that only renamed a
    flag would fail here."""
    base = default_t1_smoke_program("plain", "/dev/null/out")
    prog = TR1RendererProgramV1(levers=base.levers + (factory(*args),), num_pairs=24,
                                out_dir="/dev/null/out")
    argv = prog.compile_trainer_argv()
    for flag in factory(*args).overrides:
        assert flag in argv, f"{flag} did not reach the compiled argv"


def test_every_ported_flag_exists_on_the_tr1_trainer():
    declared = tr1_declared_flags()
    emitted = set()
    for lv in (lever_seg_focal_gamma(2.0), lever_fisher_density(1.0, "model"),
               lever_head_natural_grad(1e-2), lever_tau_softplus_tau(0.15)):
        emitted |= set(lv.overrides)
    missing = sorted(emitted - declared)
    assert not missing, f"ported levers emit flags the TR1 trainer does not declare: {missing}"


def test_ported_levers_are_visible_to_the_duty_queue():
    """'Off is a tracked queue': a default-off score-affecting lever must be REGISTERED, and the
    registration path is the package AST scan (no parallel registry)."""
    from tac.witness_dsl.activation_ledger import package_known_levers

    known = set(package_known_levers())
    for f in PT2_PORTED_LEVERS:
        assert f.__name__ in known, f"{f.__name__} is not in the duty-to-measure universe"


def test_ported_lever_names_join_back_to_their_factories():
    forms = factory_name_forms()
    for factory, args in ((lever_seg_focal_gamma, (2.0,)), (lever_fisher_density, (1.0, "model")),
                          (lever_head_natural_grad, (1e-3,)), (lever_tau_softplus_tau, (0.15,))):
        r = resolve_instance(factory(*args).name, forms)
        assert r.resolved and r.factory == factory.__name__


def test_ported_factories_refuse_their_off_value():
    """A Lever for the OFF value would be a lever that does nothing — the marker-without-work
    shape. Each factory must refuse it rather than emit an inert flag."""
    with pytest.raises(ValueError):
        lever_seg_focal_gamma(0.0)
    with pytest.raises(ValueError):
        lever_fisher_density(0.0)
    with pytest.raises(ValueError):
        lever_fisher_density(1.0, "nope")
    with pytest.raises(ValueError):
        lever_head_natural_grad(0.0)
    with pytest.raises(ValueError):
        lever_tau_softplus_tau(0.0)


def test_head_natural_grad_omits_the_value_flag_at_its_default():
    """The trainer REFUSES a set-but-ungated value flag, and an emitted-at-default value flag is
    the inert-flag genus. At eps == the trainer default only the gate is emitted."""
    assert set(lever_head_natural_grad(1e-3).overrides) == {"--head-natural-grad"}
    assert set(lever_head_natural_grad(1e-2).overrides) == {"--head-natural-grad",
                                                            "--head-natural-grad-eps"}


def test_trainer_passes_every_ported_parameter_into_make_loss_fn():
    """SOURCE-LEVEL proof that the port is threading, not decoration: the ``make_loss_fn`` call in
    the TR1 trainer must carry each ported keyword. If someone adds the flag but not the
    pass-through, the flag is a marker without work and this fails."""
    src = (_REPO_ROOT / TRAINER_RELPATH).read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "make_loss_fn"]
    assert len(calls) == 1, f"expected exactly one make_loss_fn call site, found {len(calls)}"
    kwargs = {k.arg for k in calls[0].keywords}
    for need in ("tau_softplus_tau", "focal_gamma", "fisher_density_weight",
                 "fisher_density_source", "head_natural_grad", "head_natural_grad_eps"):
        assert need in kwargs, f"{need} is declared as a flag but never reaches make_loss_fn"


def test_ported_force_gate_guard_refuses_a_value_without_its_gate():
    """The silent-no-op guard, exercised in BOTH directions.

    Importing the whole trainer module fails in a bare test environment (it is a script with
    module-level side effects), and a SKIP is a control that did not run — the vacuity genus this
    arm exists to avoid. So compile ONLY the guard's own function def out of the trainer source:
    the predicate is pure, so this is the real function, always executed.
    """
    src = (_REPO_ROOT / TRAINER_RELPATH).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "assert_ported_force_scalars_have_their_gate"]
    assert len(fdefs) == 1, "the guard must exist exactly once in the trainer"
    ns: dict = {}
    exec(compile(ast.Module(body=fdefs, type_ignores=[]), "<tr1_guard>", "exec"), ns)  # noqa: S102
    fn = ns["assert_ported_force_scalars_have_their_gate"]
    fn(0.0, "model", "off", 1e-3)                 # all defaults -> silent, no raise
    fn(1.0, "gt", "on", 1e-2)                     # both gated -> no raise
    with pytest.raises(SystemExit):
        fn(0.0, "gt", "off", 1e-3)                # source set, weight off
    with pytest.raises(SystemExit):
        fn(0.0, "model", "off", 1e-2)             # eps set, gate off
