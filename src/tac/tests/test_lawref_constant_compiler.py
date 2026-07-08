# SPDX-License-Identifier: MIT
"""Tests for the LawRef constant-compiler (task #351).

Covers: sealed-value resolution (bit-match), config-conditionality fail-closed,
sha256 / staleness refusal, fallback-with-waiver + manifest, determinism,
manifest schema, evaluator surface, InputRef/LawRef validation, and the
WitnessProgram compile integration.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest

from tac.canonical_equations.evaluators import (
    EvaluatorError,
    EvaluatorNotRegisteredError,
    get_evaluator,
    has_evaluator,
    populate_lawref_evaluators,
    register_evaluator,
    resolve_equation_value,
)
from tac.witness_dsl.lawref import (
    LADDER_DERIVED_AT_CONFIG,
    LADDER_HARDCODED_WAIVER,
    LADDER_MEASURED_ANCHOR,
    ConfigConditionalityViolation,
    InputRef,
    LawRef,
    LawRefError,
    LawResolveError,
    resolve,
    resolve_flag_dict_constants,
)
from tac.witness_dsl.lawref_builtins import (
    BUILTIN_LAWREFS,
    R_STAR_CRITICAL_NUCLEUS,
    S_STAR_FORFEIT_MATCHED_EXIT,
    TAU_STAR_MASLOV_Q90,
)

_REPO = Path(__file__).resolve().parents[3]
_ARTIFACTS = _REPO / ".omx/research/t5_crucible/artifacts"
_HAVE_ARTIFACTS = (_ARTIFACTS / "tau_mq_confirm_end_20260708.json").exists()
_needs_artifacts = pytest.mark.skipif(
    not _HAVE_ARTIFACTS, reason="T5-crucible artifacts not present in this checkout"
)


# --------------------------------------------------------------------------
# 1. Sealed-value resolution (bit-match)
# --------------------------------------------------------------------------
@_needs_artifacts
def test_s_star_resolves_to_sealed_value():
    rc = resolve(S_STAR_FORFEIT_MATCHED_EXIT, {"schedule": "mod32cap", "stage": "tau_softplus"})
    # s* = ν·forfeit; deterministic product of the two cited inputs.
    nu = 0.012653403634932212
    forfeit = 0.0005450778537325913
    assert rc.value == nu * forfeit
    assert round(rc.value, 10) == round(6.8971e-6, 10)  # the sealed value
    assert rc.fallback_used is False
    assert rc.equation_id == "forfeit_matched_exit_v1"


@_needs_artifacts
def test_tau_star_bit_matches_artifact_stored_tau_star():
    rc = resolve(TAU_STAR_MASLOV_Q90, {"leg": "END_ep1000", "convention": "q90"})
    # bit-identical to the artifact's own stored tau_star.q90.
    doc = json.loads((_ARTIFACTS / "tau_mq_confirm_end_20260708.json").read_text())
    stored = doc["results"][0]["table"]["tau_star"]["q90"]
    assert rc.value == stored == 0.4619441215759677


def test_r_star_resolves_all_literal():
    # r* = 0.95·σ_eff = 1.425 — hermetic (no artifact).
    rc = resolve(R_STAR_CRITICAL_NUCLEUS, {})
    assert rc.value == 0.95 * 1.5 == 1.4249999999999998
    assert rc.equation_id == "critical_nucleus_release_v1"
    # σ_eff carries an unverifiable config tag -> warning, not a raise.
    assert any("UNVERIFIABLE config_tag 'sigma_probe'" in w for w in rc.warnings)


def test_builtin_lawrefs_all_present():
    assert set(BUILTIN_LAWREFS) == {
        "s_star_forfeit_matched_exit",
        "tau_star_maslov_q90",
        "r_star_critical_nucleus",
    }


# --------------------------------------------------------------------------
# 2. Config-conditionality fail-closed (the P-CT1 protection)
# --------------------------------------------------------------------------
@_needs_artifacts
def test_config_conditionality_conflict_fails_closed_named():
    with pytest.raises(ConfigConditionalityViolation) as ei:
        resolve(S_STAR_FORFEIT_MATCHED_EXIT, {"schedule": "mod48cap", "stage": "tau_softplus"})
    msg = str(ei.value)
    assert "schedule" in msg and "mod32cap" in msg and "mod48cap" in msg


def test_config_conditionality_checked_on_literal_too():
    # A literal MEASURED value with a conflicting config tag also fails closed.
    lr = LawRef(
        equation_id="critical_nucleus_release_v1",
        inputs={
            "coeff": InputRef.literal(0.95, "coeff"),
            "sigma_eff": InputRef.literal(
                1.5, "knee σ", config_tags={"sigma_probe": "knee_0630"}
            ),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    with pytest.raises(ConfigConditionalityViolation):
        resolve(lr, {"sigma_probe": "knee_0701"})


def test_config_conditionality_matching_tag_resolves():
    lr = LawRef(
        equation_id="critical_nucleus_release_v1",
        inputs={
            "coeff": InputRef.literal(0.95, "coeff"),
            "sigma_eff": InputRef.literal(1.5, "knee σ", config_tags={"probe": "p0"}),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    rc = resolve(lr, {"probe": "p0"})
    assert rc.value == 1.4249999999999998
    assert rc.warnings == ()  # matching tag -> no unverifiable warning


# --------------------------------------------------------------------------
# 3. sha256 / staleness / missing-artifact refusal (fail-closed)
# --------------------------------------------------------------------------
def _write_artifact(tmp_path: Path, value: float, name: str = "art.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"probe": {"nu": value}}))
    return p


def _nu_lawref(artifact_rel: str, **anchor_kw) -> LawRef:
    return LawRef(
        equation_id="forfeit_matched_exit_v1",
        inputs={
            "nu": InputRef.anchor(artifact_rel, "probe/nu", "test ν", **anchor_kw),
            "forfeit": InputRef.literal(1.0, "unit forfeit"),
        },
        ladder_class=LADDER_MEASURED_ANCHOR,
    )


def test_missing_artifact_fails_closed(tmp_path):
    lr = _nu_lawref("does_not_exist.json")
    with pytest.raises(LawResolveError, match="not found"):
        resolve(lr, {}, repo_root=tmp_path)


def test_sha256_mismatch_fails_closed(tmp_path):
    _write_artifact(tmp_path, 0.02)
    lr = _nu_lawref("art.json", expected_sha256="0" * 64)
    with pytest.raises(LawResolveError, match="sha256 mismatch"):
        resolve(lr, {}, repo_root=tmp_path)


def test_sha256_match_resolves(tmp_path):
    import hashlib

    p = _write_artifact(tmp_path, 0.02)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    lr = _nu_lawref("art.json", expected_sha256=sha)
    rc = resolve(lr, {}, repo_root=tmp_path)
    assert rc.value == 0.02  # 0.02 * 1.0
    assert rc.resolved_inputs[0].sha256 == sha


def test_staleness_refusal(tmp_path):
    p = _write_artifact(tmp_path, 0.02)
    old = time.time() - 10 * 86400  # 10 days ago
    os.utime(p, (old, old))
    lr = _nu_lawref("art.json", max_staleness_days=5)
    with pytest.raises(LawResolveError, match="stale"):
        resolve(lr, {}, repo_root=tmp_path)


def test_staleness_ok_within_window(tmp_path):
    _write_artifact(tmp_path, 0.03)
    lr = _nu_lawref("art.json", max_staleness_days=5)
    rc = resolve(lr, {}, repo_root=tmp_path)
    assert rc.value == 0.03
    assert rc.resolved_inputs[0].staleness_days is not None


def test_missing_extract_key_fails_closed(tmp_path):
    _write_artifact(tmp_path, 0.02)
    lr = LawRef(
        equation_id="forfeit_matched_exit_v1",
        inputs={
            "nu": InputRef.anchor("art.json", "probe/MISSING", "bad path"),
            "forfeit": InputRef.literal(1.0, "f"),
        },
        ladder_class=LADDER_MEASURED_ANCHOR,
    )
    with pytest.raises(LawResolveError, match="not found"):
        resolve(lr, {}, repo_root=tmp_path)


# --------------------------------------------------------------------------
# 4. Fallback-with-waiver path + manifest records it
# --------------------------------------------------------------------------
def test_fallback_used_on_missing_artifact_records_in_manifest(tmp_path):
    lr = LawRef(
        equation_id="forfeit_matched_exit_v1",
        inputs={
            "nu": InputRef.anchor("gone.json", "probe/nu", "ν"),
            "forfeit": InputRef.literal(1.0, "f"),
        },
        ladder_class=LADDER_HARDCODED_WAIVER,
        fallback=1.23e-5,
        fallback_waiver_reason="artifact archived; last-known s* pinned pending re-fit",
    )
    rc = resolve(lr, {}, repo_root=tmp_path)
    assert rc.value == 1.23e-5
    assert rc.fallback_used is True
    d = rc.to_dict()
    assert d["fallback_used"] is True
    assert any("FALLBACK USED" in w for w in d["warnings"])


def test_config_conflict_not_swallowed_by_fallback(tmp_path):
    # A config-conditionality conflict must fail closed EVEN with a fallback.
    _write_artifact(tmp_path, 0.02)
    lr = LawRef(
        equation_id="forfeit_matched_exit_v1",
        inputs={
            "nu": InputRef.anchor(
                "art.json", "probe/nu", "ν", config_tags={"schedule": "mod32cap"}
            ),
            "forfeit": InputRef.literal(1.0, "f"),
        },
        ladder_class=LADDER_HARDCODED_WAIVER,
        fallback=9.9e-9,
        fallback_waiver_reason="pinned",
    )
    with pytest.raises(ConfigConditionalityViolation):
        resolve(lr, {"schedule": "mod48cap"}, repo_root=tmp_path)


# --------------------------------------------------------------------------
# 5. Determinism + manifest schema
# --------------------------------------------------------------------------
def test_determinism_two_resolves_identical_value(tmp_path):
    _write_artifact(tmp_path, 0.017)
    lr = _nu_lawref("art.json")
    a = resolve(lr, {}, repo_root=tmp_path)
    b = resolve(lr, {}, repo_root=tmp_path)
    assert a.value == b.value  # value never depends on wall-clock
    # resolved_at is metadata only; may differ, must not enter the value.
    assert a.to_dict()["value"] == b.to_dict()["value"]


def test_manifest_schema_keys(tmp_path):
    _write_artifact(tmp_path, 0.02)
    lr = _nu_lawref("art.json")
    rc = resolve(lr, {}, repo_root=tmp_path)
    d = rc.to_dict()
    assert set(d) >= {
        "value", "equation_id", "ladder_class", "fallback_used",
        "resolved_at", "inputs", "warnings",
    }
    inp = d["inputs"][0]
    assert set(inp) >= {
        "name", "kind", "value", "source", "sha256", "config_tags",
        "provenance", "staleness_days",
    }
    assert inp["sha256"] is not None  # anchor input carries its sha


# --------------------------------------------------------------------------
# 6. Evaluator surface
# --------------------------------------------------------------------------
def test_populate_and_resolve_equation_value():
    ids = populate_lawref_evaluators()
    assert "forfeit_matched_exit_v1" in ids
    assert has_evaluator("tau_star_maslov_quantile_v1")
    v = resolve_equation_value("tau_star_maslov_quantile_v1", {"m_q": math.log(5)})
    assert v == 1.0  # ln5/ln5


def test_evaluator_not_registered_raises():
    with pytest.raises(EvaluatorNotRegisteredError):
        get_evaluator("totally_unregistered_law_v1")


def test_register_evaluator_duplicate_guard():
    def f(inputs):
        return 1.0

    def g(inputs):
        return 2.0

    register_evaluator("dup_guard_law_v1", f)
    register_evaluator("dup_guard_law_v1", f)  # same callable -> ok
    with pytest.raises(EvaluatorError):
        register_evaluator("dup_guard_law_v1", g)  # different -> raise
    register_evaluator("dup_guard_law_v1", g, overwrite=True)  # explicit ok


def test_register_evaluator_bad_id():
    with pytest.raises(EvaluatorError):
        register_evaluator("NotSnakeCase", lambda i: 1.0)


# --------------------------------------------------------------------------
# 7. InputRef / LawRef validation
# --------------------------------------------------------------------------
def test_inputref_literal_rejects_anchor_fields():
    with pytest.raises(LawRefError):
        InputRef(kind="literal", provenance="p", value=1.0, artifact_path="x.json")


def test_inputref_literal_rejects_bool():
    with pytest.raises(LawRefError):
        InputRef.literal(True, "p")  # type: ignore[arg-type]


def test_inputref_anchor_requires_extract():
    with pytest.raises(LawRefError):
        InputRef(kind="anchor", provenance="p", artifact_path="x.json")


def test_inputref_requires_provenance():
    with pytest.raises(LawRefError):
        InputRef.literal(1.0, "   ")


def test_inputref_bad_sha_format():
    with pytest.raises(LawRefError):
        InputRef.anchor("x.json", "a/b", "p", expected_sha256="XYZ")


def test_lawref_bad_ladder_class():
    with pytest.raises(LawRefError):
        LawRef(
            equation_id="critical_nucleus_release_v1",
            inputs={"sigma_eff": InputRef.literal(1.5, "s")},
            ladder_class="not_a_class",
        )


def test_lawref_empty_inputs():
    with pytest.raises(LawRefError):
        LawRef(
            equation_id="critical_nucleus_release_v1",
            inputs={},
            ladder_class=LADDER_DERIVED_AT_CONFIG,
        )


def test_lawref_fallback_requires_reason():
    with pytest.raises(LawRefError):
        LawRef(
            equation_id="critical_nucleus_release_v1",
            inputs={"sigma_eff": InputRef.literal(1.5, "s")},
            ladder_class=LADDER_DERIVED_AT_CONFIG,
            fallback=1.0,  # no reason
        )


def test_lawref_hardcoded_waiver_requires_fallback():
    with pytest.raises(LawRefError):
        LawRef(
            equation_id="critical_nucleus_release_v1",
            inputs={"sigma_eff": InputRef.literal(1.5, "s")},
            ladder_class=LADDER_HARDCODED_WAIVER,  # no fallback
        )


def test_lawref_bad_equation_id():
    with pytest.raises(LawRefError):
        LawRef(
            equation_id="NoVersion",
            inputs={"sigma_eff": InputRef.literal(1.5, "s")},
            ladder_class=LADDER_DERIVED_AT_CONFIG,
        )


# --------------------------------------------------------------------------
# 8. Compile integration
# --------------------------------------------------------------------------
def test_resolve_flag_dict_passthrough_and_manifest():
    fd = {"--epochs": 1500, "--island-dilate-px": R_STAR_CRITICAL_NUCLEUS}
    resolved, manifest = resolve_flag_dict_constants(fd, {})
    assert resolved["--epochs"] == 1500  # non-LawRef passes through
    assert resolved["--island-dilate-px"] == 1.4249999999999998
    assert "--island-dilate-px" in manifest
    assert manifest["--island-dilate-px"]["equation_id"] == "critical_nucleus_release_v1"
    assert "--epochs" not in manifest  # only LawRef flags recorded


def test_witnessprogram_no_lawref_is_byte_identical():
    from tac.witness_dsl.curriculum_dsl import BASELINE

    plain = BASELINE.compile_trainer_argv()
    argv, manifest = BASELINE.compile_trainer_argv_with_constants()
    assert argv == plain
    assert manifest == {}


def test_witnessprogram_resolves_lawref_flag():
    from dataclasses import replace

    from tac.witness_dsl.curriculum_dsl import BASELINE

    prog = replace(BASELINE, base={**BASELINE.base, "--island-dilate-px": R_STAR_CRITICAL_NUCLEUS})
    argv, manifest = prog.compile_trainer_argv_with_constants({})
    # the resolved value appears in argv as a string token after the flag.
    i = argv.index("--island-dilate-px")
    assert argv[i + 1] == str(1.4249999999999998)
    assert manifest["--island-dilate-px"]["value"] == 1.4249999999999998
