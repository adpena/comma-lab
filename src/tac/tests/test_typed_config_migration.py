"""Migration tests: the crucible_v6 autoconfig seam is DSL-typed-validated + carries
a DSL-provenance manifest, WITHOUT changing the emitted argv (byte-identity law).
"""
from __future__ import annotations

import dataclasses as dc

import pytest

import tac.witness_autoconfig as wac
from tac.witness_dsl.typed_config import (
    PROGRAM_MANIFEST_SCHEMA,
    verify_launch_manifest,
)

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _crucible(epochs=3000):
    return wac.derive_crucible_v6_config(_GT, num_pairs=600, epochs=epochs)


def _names(cfg, out="OUT"):
    return [flag for flag, _ in cfg.to_trainer_flags(out)]


# ── byte-identity: the emitted argv is unchanged (the migration is argv-inert) ──
def test_crucible_argv_is_stable_across_derivations():
    a = _crucible().to_trainer_flags("OUT")
    b = _crucible().to_trainer_flags("OUT")
    assert a == b, "crucible argv must be deterministic"


def test_dsl_manifest_is_not_emitted_as_a_flag():
    """The manifest is provenance-only — it must NEVER appear in the emitted argv."""
    argv = _crucible().to_trainer_flags("OUT")
    joined = " ".join(str(v) for pair in argv for v in pair)
    assert "dsl_program_manifest" not in joined
    assert not any("dsl-program-manifest" in str(f) for f, _ in argv)


def test_crucible_argv_flag_names_unchanged_by_out_dir():
    """Flag NAMES are out-dir-independent (the migration touches no emission)."""
    assert _names(_crucible(), "A") == _names(_crucible(), "B")


# ── the typed-validation gate is REAL (fail-closed) ─────────────────────────
def test_crucible_carries_typed_validated_manifest():
    cfg = _crucible()
    man = cfg.dsl_program_manifest
    assert man and man["schema"] == PROGRAM_MANIFEST_SCHEMA
    assert man["typed_validated"] is True
    assert man["program_name"] == "crucible_v6"
    assert man["flag_names"] and man["flag_fingerprint"]


def test_crucible_manifest_fingerprint_matches_emitted_argv():
    cfg = _crucible()
    ok, detail = verify_launch_manifest(cfg.dsl_program_manifest, _names(cfg, "REAL/RUN/DIR"))
    assert ok, detail


def test_verify_fails_when_manifest_absent():
    cfg = _crucible()
    ok, detail = verify_launch_manifest({}, _names(cfg))
    assert not ok and "no DSL program manifest" in detail


def test_verify_fails_when_config_mutated_after_authoring():
    """If the emitted flag set drifts from the attested manifest, verify REFUSES."""
    cfg = _crucible()
    emitted = _names(cfg) + ["--totally-new-flag"]
    ok, detail = verify_launch_manifest(cfg.dsl_program_manifest, emitted)
    assert not ok and "disagrees" in detail


def test_verify_fails_when_not_typed_validated():
    cfg = _crucible()
    man = dict(cfg.dsl_program_manifest)
    man["typed_validated"] = False
    ok, detail = verify_launch_manifest(man, _names(cfg))
    assert not ok and "not typed_validated" in detail


def test_typed_gate_fails_closed_on_bad_schedule(monkeypatch):
    """If the authored typed schedule produced an INVENTED flag, the derive raises
    (the config is never launched un-validated)."""
    import tac.witness_dsl.typed_config as tc

    orig = tc.TypedWitnessConfig.validate_program

    def _boom(self, trainer_path=None):
        return ["INVENTED FLAG (not in trainer argparse): --nope"]

    monkeypatch.setattr(tc.TypedWitnessConfig, "validate_program", _boom)
    with pytest.raises(ValueError, match="DSL-authored-config gate"):
        _crucible()
    monkeypatch.setattr(tc.TypedWitnessConfig, "validate_program", orig)


# ── non-crucible configs are UNCHANGED (no manifest => byte-identical) ───────
def test_store_nothing_205_has_empty_manifest():
    cfg = wac.derive_store_nothing_205_config(_GT, num_pairs=600, epochs=3000)
    assert cfg.dsl_program_manifest == {}


def test_sealed_205_has_empty_manifest():
    cfg = wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=3000)
    assert cfg.dsl_program_manifest == {}


def test_non_crucible_argv_has_no_manifest_field_effect():
    """Adding the dataclass field must not perturb a non-crucible argv."""
    a = wac.derive_store_nothing_205_config(_GT, num_pairs=600, epochs=3000)
    b = dc.replace(a, dsl_program_manifest={"x": 1})
    assert a.to_trainer_flags("OUT") == b.to_trainer_flags("OUT")
