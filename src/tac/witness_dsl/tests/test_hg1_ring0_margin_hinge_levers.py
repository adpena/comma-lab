# SPDX-License-Identifier: MIT
"""Tests for the HG1 signed ring-0 margin hinge levers.

The load-bearing tests are the two that could not be faked by a metadata assertion:
``test_every_emitted_flag_parses_against_the_real_trainer_argparse`` (no invented flag,
checked against the trainer's own argparse rather than a hand-typed list) and
``test_margin_target_is_resolved_from_the_artifact_not_hardcoded`` (the target moves when
the MEASURED artifact moves, so it is derived and not a literal wearing a law's name).
"""

from __future__ import annotations

import json

import pytest

from tac.witness_dsl.curriculum_dsl import (
    _REPO_ROOT,
    Lever,
    real_trainer_flags,
)
from tac.witness_dsl.hg1_ring0_margin_hinge_levers_20260816 import (
    DEFAULT_HEADROOM,
    TRAINER_DEFAULT_MARGIN_TARGET,
    TRAINER_RELPATH,
    lever_hg1_q3_constrained_seg_grad,
    lever_hg1_ring0_margin_hinge,
)

TRAINER_PATH = _REPO_ROOT / TRAINER_RELPATH


def _all_levers() -> list[Lever]:
    return [lever_hg1_ring0_margin_hinge(), lever_hg1_q3_constrained_seg_grad()]


# ---------------------------------------------------------------------------------------
# no invented flags -- checked against the trainer, never against a hand-typed list
# ---------------------------------------------------------------------------------------
def test_every_emitted_flag_parses_against_the_real_trainer_argparse() -> None:
    real = real_trainer_flags(TRAINER_PATH)
    assert real, "trainer flag extraction returned nothing -- the check would be vacuous"
    for lever in _all_levers():
        for flag in lever.overrides:
            assert flag in real, (
                f"{lever.name} emits {flag}, which {TRAINER_RELPATH} does not declare"
            )


def _trainer_add_argument_specs() -> dict[str, dict]:
    """Extract ``choices``/``type`` per flag straight from the trainer's own AST.

    ``build_real_trainer_parser`` is written for the levelset trainer's ``main()`` shape
    and raises ``LookupError`` on this trainer, so the acceptance check is done against
    the trainer source directly rather than skipped -- a skipped check here would be
    vacuous exactly where an invented value would hide.
    """
    import ast

    tree = ast.parse(TRAINER_PATH.read_text())
    specs: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            continue
        spec: dict = {}
        for kw in node.keywords:
            if kw.arg == "choices":
                try:
                    spec["choices"] = list(ast.literal_eval(kw.value))
                except (ValueError, SyntaxError):
                    pass
            elif kw.arg == "type" and isinstance(kw.value, ast.Name):
                spec["type"] = kw.value.id
        specs[node.args[0].value] = spec
    return specs


def test_emitted_values_are_accepted_by_the_trainer_argument_spec() -> None:
    """A flag can exist and still reject our value (bad choice, wrong type)."""
    specs = _trainer_add_argument_specs()
    assert specs, "AST extraction found no add_argument calls -- the check would be vacuous"
    for lever in _all_levers():
        for flag, value in lever.overrides.items():
            spec = specs.get(flag)
            assert spec is not None, f"{flag} has no add_argument in {TRAINER_RELPATH}"
            if "choices" in spec:
                assert value in spec["choices"], (
                    f"{lever.name} emits {flag}={value!r}, not in choices {spec['choices']}"
                )
            if spec.get("type") == "float":
                assert isinstance(value, float), f"{flag} is type=float but got {type(value)}"

    hinge = lever_hg1_ring0_margin_hinge()
    assert specs["--seg-form-start"]["choices"].count("margin_hinge") == 1
    assert isinstance(hinge.overrides["--margin-target"], float)


# ---------------------------------------------------------------------------------------
# the target is DERIVED, not a literal
# ---------------------------------------------------------------------------------------
def test_margin_target_is_resolved_from_the_artifact_not_hardcoded(tmp_path) -> None:
    baseline = lever_hg1_ring0_margin_hinge().overrides["--margin-target"]

    artifact = tmp_path / "delta_R_noise_floor.json"
    source = json.loads((_REPO_ROOT / "reports" / "delta_R_noise_floor.json").read_text())
    source["delta_R"] = float(source["delta_R"]) * 2.0
    artifact.write_text(json.dumps(source))

    moved = lever_hg1_ring0_margin_hinge(delta_r_artifact=artifact).overrides["--margin-target"]
    assert moved == pytest.approx(baseline * 2.0), (
        "the target did not track the MEASURED artifact -- it is hardcoded, not derived"
    )


def test_margin_target_scales_with_headroom() -> None:
    one = lever_hg1_ring0_margin_hinge(headroom=1.0).overrides["--margin-target"]
    two = lever_hg1_ring0_margin_hinge(headroom=2.0).overrides["--margin-target"]
    assert two == pytest.approx(2.0 * one)


def test_default_target_is_far_below_the_trainer_default() -> None:
    """The whole point of the lever: stop spending gradient on already-safe pixels."""
    target = lever_hg1_ring0_margin_hinge().overrides["--margin-target"]
    assert 0.0 < target < TRAINER_DEFAULT_MARGIN_TARGET
    assert TRAINER_DEFAULT_MARGIN_TARGET / target > 10.0, (
        "the retarget is not material; the lever would not be worth composing"
    )


def test_target_carries_a_lawref_on_the_provenance_ladder() -> None:
    lever = lever_hg1_ring0_margin_hinge()
    assert set(lever.constant_refs) == {"--margin-target"}
    ref = lever.constant_refs["--margin-target"]
    assert ref.equation_id == "margin_band_satisficing_threshold_v1"
    assert "delta_r" in ref.inputs
    assert lever.constant_manifest["--margin-target"]["single_value_owner"] == ref.equation_id


def test_constant_manifest_keys_are_override_flags() -> None:
    """TypedLever refuses a manifest/lawref key that is not also an override."""
    for lever in _all_levers():
        # runtime_receipt_schemas are keyed by telemetry row name, not by flag, so they are
        # deliberately excluded from the TypedLever override-key constraint.
        for mapping in (lever.constant_manifest, lever.constant_refs):
            stray = set(mapping) - set(lever.overrides)
            assert not stray, f"{lever.name}: {sorted(stray)} not in overrides"


# ---------------------------------------------------------------------------------------
# guards fail closed
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_headroom_refuses(bad: float) -> None:
    with pytest.raises(ValueError, match="headroom must be > 0"):
        lever_hg1_ring0_margin_hinge(headroom=bad)


def test_headroom_that_would_not_lower_the_target_refuses(tmp_path) -> None:
    """A resolution at or above the trainer default means the lever is inert."""
    artifact = tmp_path / "delta_R_noise_floor.json"
    source = json.loads((_REPO_ROOT / "reports" / "delta_R_noise_floor.json").read_text())
    source["delta_R"] = 10.0
    artifact.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="is not below"):
        lever_hg1_ring0_margin_hinge(delta_r_artifact=artifact)


# ---------------------------------------------------------------------------------------
# composition + honesty contracts
# ---------------------------------------------------------------------------------------
def test_margin_weighted_is_optional_and_omitted_when_off() -> None:
    on = lever_hg1_ring0_margin_hinge(margin_weighted=True)
    off = lever_hg1_ring0_margin_hinge(margin_weighted=False)
    assert on.overrides["--margin-weighted-loss"] == "on"
    assert "--margin-weighted-loss" not in off.overrides


def test_margin_hinge_is_honored_by_the_trainers_margin_weighted_guard() -> None:
    """Composing --margin-weighted-loss on must not trip the trainer's own refusal."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tr1_probe", TRAINER_PATH)
    assert spec is not None and spec.loader is not None
    # The guard's membership set is a module constant; read it without executing main().
    src = TRAINER_PATH.read_text()
    marker = "MARGIN_WEIGHTED_HONORING_SEG_FORMS = frozenset("
    assert marker in src
    block = src[src.index(marker): src.index(marker) + 200]
    assert '"margin_hinge"' in block, (
        "margin_hinge left the trainer's honoring set; --margin-weighted-loss would refuse"
    )


def test_no_lever_claims_a_score_or_guarantees_zero_pose_damage() -> None:
    hinge = lever_hg1_ring0_margin_hinge()
    q3 = lever_hg1_q3_constrained_seg_grad()
    assert hinge.policy_contracts["score_claim"] is False
    assert q3.policy_contracts["score_claim"] is False
    assert hinge.policy_contracts["adds_new_loss_term"] is False, (
        "the hinge term already exists in the trainer; claiming a new term would be a fake"
    )
    assert q3.policy_contracts["guarantees_zero_pose_damage"] is False, (
        "the exact kernel holds pre-quantization only (#532); a zero-damage claim is unsupported"
    )


def test_levers_have_distinct_names_and_disjoint_flags() -> None:
    hinge = lever_hg1_ring0_margin_hinge()
    q3 = lever_hg1_q3_constrained_seg_grad()
    assert hinge.name != q3.name
    assert not (set(hinge.overrides) & set(q3.overrides)), (
        "overlapping flags would make composition order silently significant"
    )


def test_default_headroom_matches_the_sister_satisficing_lever() -> None:
    """Two forces on the same annulus must not disagree about what 'safe' means."""
    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        resolve_margin_band_threshold,
    )

    sister = resolve_margin_band_threshold(
        headroom=None, artifact_path="reports/delta_R_noise_floor.json", repo_root=_REPO_ROOT
    )
    assert pytest.approx(sister.headroom) == DEFAULT_HEADROOM
    assert lever_hg1_ring0_margin_hinge().overrides["--margin-target"] == pytest.approx(
        sister.m_safe
    )
