from __future__ import annotations

import json
import math

import pytest

from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
    DELTA_R_ARTIFACT,
    EQUATION_ID,
    FALLBACK_DELTA_R,
    FALLBACK_FULL_R_ANNULUS_P95,
    build_margin_band_satisficing_threshold_v1,
    margin_safe_lawref,
    margin_safe_threshold,
    minimum_integer_headroom,
    populate_margin_band_satisficing_threshold_equation,
    resolve_margin_band_threshold,
)
from tac.witness_dsl.curriculum_dsl import MarginBandSatisficing
from tac.witness_dsl.lawref import LADDER_DERIVED_LIVE, resolve


def _write_artifact(path, *, delta_r: float, full_r_p95: float) -> None:
    path.write_text(
        json.dumps(
            {
                "measurement": "delta_R_noise_floor",
                "n_frames": 96,
                "band": 1.0,
                "delta_R": delta_r,
                "cross_check_full_R_vs_gt_direct": {
                    "annulus": {"p95": full_r_p95},
                },
            }
        ),
        encoding="utf-8",
    )


def test_threshold_is_exact_product() -> None:
    assert margin_safe_threshold(FALLBACK_DELTA_R, 2.0) == (
        2.0 * FALLBACK_DELTA_R
    )


def test_threshold_supports_explicit_headroom_three_as_nondefault_treatment() -> None:
    assert margin_safe_threshold(FALLBACK_DELTA_R, 3.0) == (
        3.0 * FALLBACK_DELTA_R
    )


@pytest.mark.parametrize("delta_r", [0.0, -0.1, math.nan, math.inf])
def test_threshold_rejects_invalid_delta_r(delta_r: float) -> None:
    with pytest.raises(ValueError, match="delta_r"):
        margin_safe_threshold(delta_r, 2.0)


@pytest.mark.parametrize("headroom", [0.0, 0.999, -1.0, math.nan, math.inf])
def test_threshold_rejects_invalid_headroom(headroom: float) -> None:
    with pytest.raises(ValueError, match="headroom"):
        margin_safe_threshold(FALLBACK_DELTA_R, headroom)


def test_default_headroom_is_derived_as_two_from_measured_crosscheck() -> None:
    assert minimum_integer_headroom(
        FALLBACK_DELTA_R, FALLBACK_FULL_R_ANNULUS_P95
    ) == 2.0


def test_default_two_times_delta_covers_full_r_p95() -> None:
    threshold = margin_safe_threshold(FALLBACK_DELTA_R, 2.0)
    assert threshold >= FALLBACK_FULL_R_ANNULUS_P95
    assert FALLBACK_DELTA_R < FALLBACK_FULL_R_ANNULUS_P95


def test_resolver_reads_delta_r_artifact() -> None:
    resolved = resolve_margin_band_threshold()
    assert resolved.artifact_path == DELTA_R_ARTIFACT
    assert resolved.delta_r == pytest.approx(0.021881818771362305)  # n600 artifact (ddm_dr1)
    assert resolved.artifact_fallback_used is False
    assert resolved.lawref_fallback_used is False


def test_resolver_derives_default_msafe_from_artifact() -> None:
    resolved = resolve_margin_band_threshold()
    assert resolved.headroom == 2.0
    assert resolved.m_safe == pytest.approx(0.04376363754272461)
    assert resolved.m_safe == pytest.approx(resolved.headroom * resolved.delta_r)


def test_resolver_tracks_custom_artifact_value(tmp_path) -> None:
    artifact = tmp_path / "delta.json"
    _write_artifact(artifact, delta_r=0.025, full_r_p95=0.049)
    resolved = resolve_margin_band_threshold(artifact_path=artifact)
    assert resolved.delta_r == pytest.approx(0.025)
    assert resolved.headroom == 2.0
    assert resolved.m_safe == pytest.approx(0.05)
    assert resolved.artifact_fallback_used is False


def test_resolver_derives_larger_headroom_if_artifact_requires_it(tmp_path) -> None:
    artifact = tmp_path / "delta.json"
    _write_artifact(artifact, delta_r=0.02, full_r_p95=0.041)
    resolved = resolve_margin_band_threshold(artifact_path=artifact)
    assert resolved.headroom == 3.0
    assert resolved.m_safe == pytest.approx(0.06)


def test_resolver_missing_artifact_uses_documented_fallback(tmp_path) -> None:
    resolved = resolve_margin_band_threshold(artifact_path=tmp_path / "missing.json")
    assert resolved.delta_r == FALLBACK_DELTA_R
    assert resolved.headroom == 2.0
    assert resolved.m_safe == 2.0 * FALLBACK_DELTA_R
    assert resolved.artifact_fallback_used is True
    assert resolved.lawref_fallback_used is True
    assert resolved.lawref_manifest["warnings"]


def test_resolver_invalid_artifact_schema_fails_loud(tmp_path) -> None:
    artifact = tmp_path / "delta.json"
    artifact.write_text(json.dumps({"delta_R": FALLBACK_DELTA_R}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schema"):
        resolve_margin_band_threshold(artifact_path=artifact)


def test_lawref_resolves_equation_and_records_anchor_sha() -> None:
    resolved = resolve(margin_safe_lawref(headroom=2.0))
    manifest = resolved.to_dict()
    assert manifest["equation_id"] == EQUATION_ID
    assert manifest["ladder_class"] == LADDER_DERIVED_LIVE
    assert manifest["fallback_used"] is False
    anchor = next(row for row in manifest["inputs"] if row["name"] == "delta_r")
    assert anchor["source"].endswith(DELTA_R_ARTIFACT)
    assert len(anchor["sha256"]) == 64


def test_factory_default_emits_artifact_derived_values() -> None:
    lever = MarginBandSatisficing()
    overrides = lever.overrides
    assert overrides["--seg-margin-satisfice-delta-r"] == pytest.approx(
        0.021881818771362305
    )
    assert overrides["--seg-margin-satisfice-headroom"] == 2.0
    assert overrides["--seg-margin-satisfice-msafe"] == pytest.approx(
        0.04376363754272461
    )


def test_factory_keeps_lever_default_off_at_trainer_surface() -> None:
    lever = MarginBandSatisficing()
    assert lever.name == "margin_band_satisficing"
    # The factory remains a treatment lever; this repair does not compose or fire it.
    assert lever.overrides["--seg-margin-satisfice-weight"] == pytest.approx(0.2)


def test_factory_explicit_headroom_three_derives_msafe() -> None:
    lever = MarginBandSatisficing(headroom=3.0)
    delta = lever.overrides["--seg-margin-satisfice-delta-r"]
    assert lever.overrides["--seg-margin-satisfice-headroom"] == 3.0
    assert lever.overrides["--seg-margin-satisfice-msafe"] == pytest.approx(3.0 * delta)


def test_factory_accepts_matching_msafe_compatibility_override() -> None:
    expected = 2.0 * FALLBACK_DELTA_R
    lever = MarginBandSatisficing(msafe=expected)
    assert lever.overrides["--seg-margin-satisfice-msafe"] == expected


@pytest.mark.parametrize("bad_msafe", [0.06, 0.0392, FALLBACK_DELTA_R])
def test_factory_rejects_msafe_override_that_drifted_from_law(bad_msafe: float) -> None:
    with pytest.raises(ValueError, match="canonical invariant"):
        MarginBandSatisficing(msafe=bad_msafe)


def test_factory_accepts_matching_delta_r_compatibility_override() -> None:
    lever = MarginBandSatisficing(delta_r=FALLBACK_DELTA_R)
    assert lever.overrides["--seg-margin-satisfice-delta-r"] == FALLBACK_DELTA_R


@pytest.mark.parametrize("bad_delta_r", [0.0196, 0.02, 0.01])
def test_factory_rejects_delta_r_override_that_drifted_from_artifact(
    bad_delta_r: float,
) -> None:
    with pytest.raises(ValueError, match="does not match the MEASURED artifact"):
        MarginBandSatisficing(delta_r=bad_delta_r)


def test_factory_custom_artifact_drives_both_emitted_values(tmp_path) -> None:
    artifact = tmp_path / "delta.json"
    _write_artifact(artifact, delta_r=0.025, full_r_p95=0.049)
    lever = MarginBandSatisficing(delta_r_artifact=artifact)
    assert lever.overrides["--seg-margin-satisfice-delta-r"] == pytest.approx(0.025)
    assert lever.overrides["--seg-margin-satisfice-msafe"] == pytest.approx(0.05)


def test_factory_missing_artifact_fallback_still_obeys_invariant(tmp_path) -> None:
    lever = MarginBandSatisficing(delta_r_artifact=tmp_path / "missing.json")
    delta = lever.overrides["--seg-margin-satisfice-delta-r"]
    headroom = lever.overrides["--seg-margin-satisfice-headroom"]
    assert delta == FALLBACK_DELTA_R
    assert lever.overrides["--seg-margin-satisfice-msafe"] == headroom * delta


def test_factory_docstring_relation_is_mutually_consistent() -> None:
    doc = MarginBandSatisficing.__doc__ or ""
    assert "m_safe = headroom·δ_R" in doc
    assert "DERIVED headroom 2" in doc
    assert "0.04376363754272461" in doc  # n600 (ddm_dr1); the n96 value is named as history
    assert "3·δ_R" not in doc


def test_equation_anchor_labels_headroom_three_as_open() -> None:
    equation = build_margin_band_satisficing_threshold_v1()
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == EQUATION_ID
    assert anchor.empirical_output["derived_headroom"] == 2.0
    assert anchor.empirical_output["derived_m_safe"] == pytest.approx(
        0.04376363754272461
    )
    assert anchor.empirical_output["headroom_3_status"] == (
        "OPEN_UNMEASURED_TREATMENT_NOT_DEFAULT"
    )
    assert "claim that headroom 3 improves d_seg" in equation.domain_of_validity["excluded"]


def test_equation_callable_points_to_real_threshold_function() -> None:
    equation = build_margin_band_satisficing_threshold_v1()
    assert equation.python_callable_module_path.endswith(":margin_safe_threshold")
    assert margin_safe_threshold(0.02, 2.0) == pytest.approx(0.04)


def test_equation_populator_uses_append_only_registry(tmp_path) -> None:
    path = tmp_path / "equations.jsonl"
    lock_path = tmp_path / "equations.lock"
    populate_margin_band_satisficing_threshold_equation(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="margin-band-test",
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_id"] == EQUATION_ID


def test_package_exports_equation_and_resolver() -> None:
    from tac.canonical_equations import (
        build_margin_band_satisficing_threshold_v1 as exported_builder,
    )
    from tac.canonical_equations import (
        resolve_margin_band_threshold as exported_resolver,
    )

    assert exported_builder is build_margin_band_satisficing_threshold_v1
    assert exported_resolver is resolve_margin_band_threshold


# ── ddm_ql3 (2026-09-04): the retired n96-prefix m_safe must not survive anywhere live ──────
# dr1 MEASURED delta_R at n600 = 0.021881818771362305; the n96 CONTIGUOUS PREFIX read
# 0.019590163230895963, 11.70% LOW (law ``annulus_restricted_prefix_bias_detector_v1``). The
# census found the derived n96 m_safe still hardcoded in two live harnesses as an argparse
# default with NO provenance comment — so no grep for "n96" could find it. These two tests are
# the structural detector: the literal is refused repo-wide outside its documented historical
# homes, and both harnesses must resolve through the law rather than restate a number.
_RETIRED_N96_M_SAFE = "0.039180326461791926"
_RETIRED_N96_DELTA_R = "0.019590163230895963"


def test_retired_n96_m_safe_literal_has_no_live_home() -> None:
    """A retired literal in a live VALUE position is invisible to provenance greps.

    AST, not text: a comment or docstring that names the old value as history is exactly what
    good provenance looks like, so only real numeric constants count. The one legal way to keep
    the value live is to say so in the NAME (``*_N96*`` / ``*RETIRED*`` / ``*HISTORICAL*``) — the
    two harnesses this test was written for held it under the neutral name ``DEFAULT_M_SAFE``.
    """
    import ast
    import warnings
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    retired = {float(_RETIRED_N96_M_SAFE), float(_RETIRED_N96_DELTA_R)}
    labelled = ("N96", "RETIRED", "HISTORICAL", "PREFIX")
    offenders: list[str] = []
    for rel_dir in ("src/tac", "tools", "experiments"):
        for path in sorted((root / rel_dir).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if "/results/" in rel or "/archive/" in rel:
                continue
            if "/tests/" in rel or Path(rel).name.startswith("test_"):
                continue  # fixtures may pin a retired value on purpose; they launch nothing
            try:
                with warnings.catch_warnings():
                    # a sister file's bad escape sequence is not this gate's finding
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                names += [t.attr for t in node.targets if isinstance(t, ast.Attribute)]
                if any(tok in n.upper() for n in names for tok in labelled):
                    continue
                for leaf in ast.walk(node.value):
                    if (
                        isinstance(leaf, ast.Constant)
                        and isinstance(leaf.value, float)
                        and leaf.value in retired
                    ):
                        offenders.append(f"{rel}:{leaf.lineno}: {names or ['<expr>']} = {leaf.value!r}")
    assert offenders == [], (
        "the retired n96-prefix constant is live under a neutral name — it is ANTI-conservative "
        "(a satisficing target 11.70% too low declares pixels R-safe that uint8 noise still "
        "flips). Resolve through resolve_margin_band_threshold(), or rename it to say it is "
        "historical:\n" + "\n".join(offenders)
    )


def test_both_uint8_harnesses_derive_m_safe_from_the_law() -> None:
    """The two harnesses the ddm_ql3 census caught must stay law-derived, never literal."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    expected = resolve_margin_band_threshold().m_safe
    for rel in (
        "tools/measure_uint8_lattice_feasibility.py",
        "tools/constructive_inverse_solve_harness.py",
    ):
        name = f"_ql3_{Path(rel).stem}"
        spec = importlib.util.spec_from_file_location(name, root / rel)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            resolved = float(module.DEFAULT_M_SAFE)
            assert resolved == expected, f"{rel} drifted from the canonical law"
            assert resolved != float(_RETIRED_N96_M_SAFE)
        finally:
            sys.modules.pop(name, None)
