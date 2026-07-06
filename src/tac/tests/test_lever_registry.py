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
