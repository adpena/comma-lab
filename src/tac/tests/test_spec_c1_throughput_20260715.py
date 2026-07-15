# SPDX-License-Identifier: MIT
"""Reconciled C1 throughput leg — supersession + delegation tests (#507).

The original strict-identity test suite exercised ten Lever factories whose
curriculum_dsl/typed_config hot-file edits were never harvested to main (the
module was import-broken on main from its first commit). The reconciled module
maps every historical factory onto its live main-tree surface and delegates its
compile entry points; these tests pin that contract.
"""

from __future__ import annotations

from types import SimpleNamespace

from tac.witness_dsl.spec_c1_throughput_20260715 import (
    EXCLUDED_OR_HELD,
    HISTORICAL_FACTORY_RECONCILIATION,
    PROGRAM_NAME,
    compile_c1_sr_parent_launch_config,
    compile_c1_throughput_launch_config,
)

_ORIGINAL_FACTORY_NAMES = (
    "SafeCompileReference",
    "SerialPairRouting",
    "FusedRKernel",
    "CacheGtSkeleton",
    "GroupedBackwardReference",
    "PersistencePoolKernel",
    "FrozenScorerOneThread",
    "AsyncVerdictOffload",
    "VerdictChunking",
    "ComponentWallclockTelemetry",
)


def test_every_unharvested_factory_is_reconciled() -> None:
    assert set(HISTORICAL_FACTORY_RECONCILIATION) == set(_ORIGINAL_FACTORY_NAMES)
    for name, disposition in HISTORICAL_FACTORY_RECONCILIATION.items():
        assert disposition.strip(), name


def test_historical_strict_identity_rows_preserved_with_supersessions() -> None:
    # The historical strict-identity record is intact ...
    for key in ("whole_step_megakernel_356", "custom_grouped_backward",
                "micro_batch_pairs_gt1", "pose_verdict_gate"):
        assert key in EXCLUDED_OR_HELD
    # ... and the rows the 2026-07-15 directive re-adjudicated say so explicitly.
    assert "REFUTED" in EXCLUDED_OR_HELD["custom_grouped_backward"]["superseded_20260715"]
    assert "CODE-BLOCKED" in EXCLUDED_OR_HELD["micro_batch_pairs_gt1"]["superseded_20260715"]
    # Megakernel stays excluded on measured economics — no supersession key.
    assert "superseded_20260715" not in EXCLUDED_OR_HELD["whole_step_megakernel_356"]


def test_throughput_compile_delegates_and_stamps_supersession(monkeypatch) -> None:
    sentinel = SimpleNamespace(
        typed=object(),
        constants_manifest={"k": 1},
        dsl_program_manifest={"program_name": "c1_optimal_form"},
        schedule_governance={"g": True},
    )
    from tac.witness_dsl import spec_c1_optimal_form_20260715 as opt

    monkeypatch.setattr(
        opt, "compile_c1_optimal_form_launch_config",
        lambda *args, **kwargs: sentinel)
    cfg = compile_c1_throughput_launch_config()
    assert cfg.typed is sentinel.typed
    record = cfg.dsl_program_manifest["superseded_strict_identity_variant"]
    assert record["historical_program_name"] == PROGRAM_NAME
    assert record["factory_reconciliation"] == HISTORICAL_FACTORY_RECONCILIATION
    assert cfg.constants_manifest == {"k": 1}
    assert cfg.schedule_governance == {"g": True}


def test_sr_parent_delegates_to_official_factory(monkeypatch) -> None:
    seen: dict = {}

    def _stub(**kwargs):
        seen.update(kwargs)
        return "OFFICIAL_C1A"

    from tac.witness_dsl import spec_v9_cgauge

    monkeypatch.setattr(
        spec_v9_cgauge, "compile_v9_cgauge_ideal_mod19_sR_launch_config", _stub)
    out = compile_c1_sr_parent_launch_config(num_pairs=24, epochs=7)
    assert out == "OFFICIAL_C1A"
    assert seen["num_pairs"] == 24
    assert seen["epochs"] == 7
