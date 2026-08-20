# SPDX-License-Identifier: MIT
"""Vertical contract tests for the G121 exhaustive stage harvester."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.witness_control import fresh_producer_lineage_v1 as lineage
from tac.witness_control import (
    taskspace_g112_exact_checkpoint_partition_v1 as g112,
)
from tac.witness_control import taskspace_g121_resumable_stage_harvest_v1 as g121
from tac.witness_control.tests import (
    test_fresh_producer_lineage_v1 as lineage_fixtures,
)


def _binding(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_g112_stage_partition_creates_parent_and_reopens_canonical_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path.resolve() / "producer"
    producer.mkdir()
    checkpoint_id = "a" * 64
    deploy = _binding(producer / "source" / "deploy.npz", b"deploy")
    resume = _binding(producer / "source" / "resume.npz", b"resume")
    lineage_receipt = _binding(
        producer / "source" / "lineage.json",
        b"lineage",
    )
    pair = SimpleNamespace(
        checkpoint_id_sha256=checkpoint_id,
        deploy=SimpleNamespace(**deploy),
        resume=SimpleNamespace(**resume),
        current_launch_dsl_compile_hash="b" * 64,
    )
    chain = SimpleNamespace(
        current=SimpleNamespace(
            pair=pair,
            receipt_path=Path(str(lineage_receipt["path"])),
            receipt_sha256=str(lineage_receipt["sha256"]),
        ),
    )
    stage = {"chain": chain}
    calls = 0

    def materialize(**kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        output_root = kwargs["output_root"]
        assert isinstance(output_root, Path)
        assert output_root.parent.is_dir()
        output_root.mkdir()
        receipt_path = output_root / g112.RECEIPT_NAME
        receipt_path.write_bytes(b"g112")
        return SimpleNamespace(
            receipt_path=receipt_path.resolve(),
            receipt_sha256=hashlib.sha256(b"g112").hexdigest(),
        )

    def reopen(
        path: Path,
        *,
        expected_sha256: str,
    ) -> SimpleNamespace:
        assert path.name == g112.RECEIPT_NAME
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        return SimpleNamespace(
            receipt_path=path.resolve(),
            receipt_bytes=len(payload),
            receipt_sha256=expected_sha256,
        )

    monkeypatch.setattr(
        g112,
        "materialize_g112_checkpoint_partition",
        materialize,
    )
    monkeypatch.setattr(g112, "open_g112_partition_receipt", reopen)

    first = g121._materialize_or_open_g112_stage(
        producer=producer,
        stage=stage,
    )
    second = g121._materialize_or_open_g112_stage(
        producer=producer,
        stage=stage,
    )

    assert calls == 1
    assert first == second
    assert Path(str(first["path"])).name == g112.RECEIPT_NAME


def _target(
    score: str = "0.172",
    *,
    pointer: str = "a",
) -> dict[str, object]:
    fraction = Fraction(Decimal(score))
    return {
        "score_decimal": score,
        "score_rational": {
            "numerator": fraction.numerator,
            "denominator": fraction.denominator,
        },
        "pointer_snapshot_identity_sha256": pointer * 64,
        "postverified_pointer_identity_sha256": pointer * 64,
    }


def _physical(root: Path, tag: str) -> dict[str, object]:
    bindings = {
        name: _binding(root / tag / f"{name}.bin", f"{tag}:{name}".encode())
        for name in (
            "g112_partition_receipt",
            "g112_semantic_child",
            "g112_pose_initializer",
            "g111_deploy_checkpoint",
            "g111_full_state_resume_checkpoint",
            "g111_fresh_lineage_receipt",
        )
    }
    return {
        **bindings,
        "g111_checkpoint_id_sha256": hashlib.sha256(
            f"checkpoint:{tag}".encode()
        ).hexdigest(),
        "g111_stage": tag,
        "g111_epoch": 1,
        "fresh_lineage_complete": True,
    }


def _alternatives(
    root: Path,
    tag: str,
    selected: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    matrix = (
        ("RAW_I16_LE", "STORE"),
        ("RAW_I16_LE", "DEFLATE"),
        ("DELTA_RICE_BEST_K", "STORE"),
        ("DELTA_RICE_BEST_K", "DEFLATE"),
    )
    for index, (codec, method) in enumerate(matrix):
        archive = (
            selected
            if index == 0
            else _binding(
                root / tag / f"{codec}.{method}.zip",
                f"{tag}:{codec}:{method}".encode(),
            )
        )
        rows.append(
            {
                "alternative_identity_sha256": hashlib.sha256(
                    f"{tag}:{codec}:{method}:identity".encode()
                ).hexdigest(),
                "y1_wire_codec": codec,
                "outer_zip_method": method,
                "archive": archive,
            }
        )
    return rows


def _raw(
    root: Path,
    *,
    tag: str,
    physical: dict[str, object],
    live_target: dict[str, object],
    wire_k: int,
    semantic_bytes: int = 100,
    source_k: int | None = None,
    qat_status: str = "not_required",
    terminal_k: int | None = None,
) -> dict[str, object]:
    measurement_identity = hashlib.sha256(f"measurement:{tag}".encode()).hexdigest()
    observation_identity = hashlib.sha256(
        (
            f"observation:{tag}:"
            f"{live_target['pointer_snapshot_identity_sha256']}"
        ).encode()
    ).hexdigest()
    selected = _binding(
        root / tag / "selected.zip",
        (f"selected:{tag}".encode() + b"x" * semantic_bytes),
    )
    measurement = _binding(
        root / tag / "measurement.json",
        f"measurement:{tag}".encode(),
    )
    observation = _binding(
        root / tag / "observation.json",
        f"observation:{tag}".encode(),
    )
    production = _binding(
        root / tag / "production.json",
        f"production:{tag}".encode(),
    )
    if source_k is None:
        source = {
            "status": "unmeasured",
            "disagreement_pixels": None,
            "pixel_denominator": g121.EXACT_SEG_PIXEL_DENOMINATOR,
            "measurement_receipt": None,
        }
        regret = {
            "status": "unmeasured",
            "disagreement_delta_pixels": None,
            "rational": None,
            "receipt": None,
        }
    else:
        source_receipt = _binding(
            root / tag / "source_float.json",
            f"source:{tag}".encode(),
        )
        regret_receipt = _binding(
            root / tag / "regret.json",
            f"regret:{tag}".encode(),
        )
        delta = wire_k - source_k
        source = {
            "status": "measured",
            "disagreement_pixels": source_k,
            "pixel_denominator": g121.EXACT_SEG_PIXEL_DENOMINATOR,
            "measurement_receipt": source_receipt,
        }
        regret = {
            "status": "measured",
            "disagreement_delta_pixels": delta,
            "rational": {
                "numerator": delta,
                "denominator": g121.EXACT_SEG_PIXEL_DENOMINATOR,
            },
            "receipt": regret_receipt,
        }
    if qat_status == "terminal_stage_measured":
        qat = {
            "status": qat_status,
            "terminal_stage_physical_identity_sha256": hashlib.sha256(
                f"qat:{tag}".encode()
            ).hexdigest(),
            "disagreement_pixels": terminal_k,
            "pixel_denominator": g121.EXACT_SEG_PIXEL_DENOMINATOR,
            "receipt": _binding(
                root / tag / "g115_qat.json",
                f"qat:{tag}".encode(),
            ),
        }
    else:
        qat = {
            "status": qat_status,
            "terminal_stage_physical_identity_sha256": None,
            "disagreement_pixels": None,
            "pixel_denominator": g121.EXACT_SEG_PIXEL_DENOMINATOR,
            "receipt": None,
        }
    fraction = Fraction(
        live_target["score_rational"]["numerator"],
        live_target["score_rational"]["denominator"],
    )
    obstruction = g121._exact_obstruction(
        wire_disagreements=wire_k,
        target=fraction,
        source_disagreements=source_k,
        g115_qat=qat,
    )
    physical_sha = hashlib.sha256(g121._canonical_json(physical)).hexdigest()
    public_wire = {
        "disagreement_pixels": wire_k,
        "pixel_denominator": g121.EXACT_SEG_PIXEL_DENOMINATOR,
        "d_seg_rational": {
            "numerator": wire_k,
            "denominator": g121.EXACT_SEG_PIXEL_DENOMINATOR,
        },
        "d_seg_display_float": wire_k / g121.EXACT_SEG_PIXEL_DENOMINATOR,
        "measurement_identity_sha256": measurement_identity,
    }
    return {
        "stage_tag": tag,
        "measurement_receipt": measurement,
        "measurement_identity_sha256": measurement_identity,
        "public_wire_seg": public_wire,
        "physical_stage_identity": physical,
        "physical_stage_identity_sha256": physical_sha,
        "pose_initializer_identity_sha256": physical[
            "g112_pose_initializer"
        ]["sha256"],
        "selected_archive": selected,
        "alternatives": _alternatives(root, tag, selected),
        "source_float_seg": source,
        "wire_regret": regret,
        "g115_qat": qat,
        "live_target": live_target,
        "prepose_obstruction": obstruction,
        "observation_receipt": observation,
        "observation_identity_sha256": observation_identity,
        "production_receipt": production,
        "public_runtime_tree": {"tree_sha256": "f" * 64},
    }


def _run(
    tmp_path: Path,
    *,
    stages: list[dict[str, object]],
    target: dict[str, object],
    provider,
) -> g121.G121StageHarvestResultV1:
    launch = tmp_path / "launch_manifest.json"
    if not launch.exists():
        launch.write_bytes(b"fixture launch")
    return g121._harvest_g111_stages_v1_test_only(
        stages=stages,
        live_target=target,
        launch_manifest=launch,
        output_dir=tmp_path / "out",
        progress_dir=tmp_path / "progress",
        provider=provider,
        injected_inputs_are_test_only=True,
    )


def test_public_surface_is_frozen() -> None:
    assert g121.RETAINED_PREPOSE_SCHEMA == "tac.g121_retained_prepose.v2"
    assert g121.RETAINED_PREPOSE_BASENAME == "g121_retained_prepose.json"
    assert callable(g121.harvest_g111_stages_v1)
    assert callable(g121.harvest_g111_available_stages_v1)
    assert callable(g121.open_g121_retained_prepose_v1)


def test_production_rejects_unsafe_g120_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = type(
        "UnsafeG120V1",
        (),
        {
            "PRODUCTION_SCHEMA": "tac.g120_parsed_stage_production_authority.v1",
            "MEASUREMENT_SCHEMA": "tac.g120_public_plugin_measurement.v1",
            "OBSERVATION_SCHEMA": None,
        },
    )()
    monkeypatch.setattr(
        g121.importlib,
        "import_module",
        lambda _name: unsafe,
    )
    with pytest.raises(g121.G121StageHarvestError, match="unsafe G120"):
        g121.harvest_g111_stages_v1(
            producer_run_dir=tmp_path,
            expected_launch_manifest_sha256="0" * 64,
            output_dir=tmp_path / "out",
            progress_dir=tmp_path / "progress",
        )


def test_live_and_final_public_entrypoints_select_distinct_authority_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes: list[bool] = []
    progress = g121.G121StageHarvestProgressV1(
        stage_ledger_path=tmp_path / "ledger",
        stage_ledger_sha256="a" * 64,
        discovered_stage_count=1,
        accounted_stage_count=1,
        scorer_replay_count=1,
        reused_measurement_count=0,
    )
    final = g121.G121StageHarvestResultV1(
        retained_prepose_path=tmp_path / "retained",
        retained_prepose_sha256="b" * 64,
        completion_receipt_path=tmp_path / "complete",
        completion_receipt_sha256="c" * 64,
        stage_ledger_path=tmp_path / "ledger",
        stage_ledger_sha256="d" * 64,
        scheduling_hint_path=None,
        scheduling_hint_sha256=None,
        discovered_stage_count=1,
        accounted_stage_count=1,
        retained_stage_count=1,
        deferred_stage_count=0,
        pruned_stage_count=0,
        blocked_stage_count=0,
        scorer_replay_count=0,
        reused_measurement_count=1,
    )
    monkeypatch.setattr(g121, "_require_safe_g120_v2", lambda: object())

    def fake_impl(**kwargs):
        modes.append(kwargs["finalize"])
        return final if kwargs["finalize"] else progress

    monkeypatch.setattr(g121, "_harvest_g111_stages_impl", fake_impl)
    common = {
        "producer_run_dir": tmp_path,
        "expected_launch_manifest_sha256": "e" * 64,
        "output_dir": tmp_path / "out",
        "progress_dir": tmp_path / "progress",
    }
    assert (
        g121.harvest_g111_available_stages_v1(**common)
        .exhaustive_enumeration_proven
        is False
    )
    assert g121.harvest_g111_stages_v1(**common).exhaustive_enumeration_proven
    assert modes == [False, True]


def test_equality_is_not_retained() -> None:
    target = Fraction(1, 8)
    k = g121.EXACT_SEG_PIXEL_DENOMINATOR // 800
    obstruction = g121._exact_obstruction(
        wire_disagreements=k,
        target=target,
        source_disagreements=None,
        g115_qat={
            "status": "required_unmeasured",
            "disagreement_pixels": None,
        },
    )
    assert obstruction["lhs"] == obstruction["rhs"]
    assert obstruction["strict_distortion_open"] is False
    assert obstruction["disposition"] == g121.DEFER_G115_WIRE_QAT


def test_exact_rational_escapes_float_collision() -> None:
    k = g121.EXACT_SEG_PIXEL_DENOMINATOR // 800
    lower = Fraction(Decimal("0.12499999999999999999"))
    upper = Fraction(Decimal("0.12500000000000000001"))
    assert float(lower) == float(upper) == 0.125
    qat = {"status": "required_unmeasured", "disagreement_pixels": None}
    assert (
        g121._exact_obstruction(
            wire_disagreements=k,
            target=lower,
            source_disagreements=k,
            g115_qat=qat,
        )["disposition"]
        == g121.PRUNE_EXACT_DISTORTION_OBSTRUCTION
    )
    assert (
        g121._exact_obstruction(
            wire_disagreements=k,
            target=upper,
            source_disagreements=k - 1,
            g115_qat=qat,
        )["disposition"]
        == g121.RETAIN_POST_G105_POSE
    )


def test_dominated_semantic_bytes_do_not_prune(tmp_path: Path) -> None:
    target = _target()
    physical_a = _physical(tmp_path, "a")
    physical_b = _physical(tmp_path, "b")
    stages = [
        {"stage_tag": "a", "physical_stage_identity": physical_a},
        {"stage_tag": "b", "physical_stage_identity": physical_b},
    ]

    def provider(stage, _prior):
        return (
            _raw(
                tmp_path,
                tag=stage["stage_tag"],
                physical=stage["physical_stage_identity"],
                live_target=target,
                wire_k=100,
                semantic_bytes=100_000 if stage["stage_tag"] == "b" else 1,
            ),
            True,
        )

    result = _run(tmp_path, stages=stages, target=target, provider=provider)
    assert result.retained_stage_count == 2
    assert result.accounted_stage_count == 2


def test_qat_uncertainty_is_deferred_not_pruned(tmp_path: Path) -> None:
    target = _target("0.125")
    physical = _physical(tmp_path, "qat_defer")
    stage = {"stage_tag": "qat_defer", "physical_stage_identity": physical}
    equality_k = g121.EXACT_SEG_PIXEL_DENOMINATOR // 800

    def provider(_stage, _prior):
        return (
            _raw(
                tmp_path,
                tag="qat_defer",
                physical=physical,
                live_target=target,
                wire_k=equality_k,
                source_k=equality_k - 1,
                qat_status="required_unmeasured",
            ),
            True,
        )

    result = _run(tmp_path, stages=[stage], target=target, provider=provider)
    assert result.retained_stage_count == 0
    assert result.deferred_stage_count == 1
    assert result.pruned_stage_count == 0


def test_terminal_qat_exact_count_controls_disposition() -> None:
    target = Fraction(1, 8)
    k = g121.EXACT_SEG_PIXEL_DENOMINATOR // 800
    obstruction = g121._exact_obstruction(
        wire_disagreements=k,
        target=target,
        source_disagreements=k - 1,
        g115_qat={
            "status": "terminal_stage_measured",
            "disagreement_pixels": k - 1,
        },
    )
    assert obstruction["disposition"] == g121.RETAIN_POST_G105_POSE


def test_pointer_refresh_reuses_external_measurement(tmp_path: Path) -> None:
    first_target = _target("0.172", pointer="a")
    second_target = _target("0.171", pointer="b")
    physical = _physical(tmp_path, "refresh")
    stage = {"stage_tag": "refresh", "physical_stage_identity": physical}
    calls: list[bool] = []

    def first_provider(_stage, prior):
        calls.append(prior is not None)
        return (
            _raw(
                tmp_path,
                tag="refresh",
                physical=physical,
                live_target=first_target,
                wire_k=100,
            ),
            prior is None,
        )

    first = _run(
        tmp_path,
        stages=[stage],
        target=first_target,
        provider=first_provider,
    )
    assert first.scorer_replay_count == 1

    def second_provider(_stage, prior):
        calls.append(prior is not None)
        assert prior["g120"]["measurement_receipt"]["sha256"]
        return (
            _raw(
                tmp_path,
                tag="refresh",
                physical=physical,
                live_target=second_target,
                wire_k=100,
            ),
            False,
        )

    second = _run(
        tmp_path,
        stages=[stage],
        target=second_target,
        provider=second_provider,
    )
    assert second.scorer_replay_count == 0
    assert second.reused_measurement_count == 1
    assert calls == [False, True]


def test_idempotent_resume_does_not_call_provider(tmp_path: Path) -> None:
    target = _target()
    physical = _physical(tmp_path, "resume")
    stage = {"stage_tag": "resume", "physical_stage_identity": physical}

    def provider(_stage, _prior):
        return (
            _raw(
                tmp_path,
                tag="resume",
                physical=physical,
                live_target=target,
                wire_k=100,
            ),
            True,
        )

    first = _run(tmp_path, stages=[stage], target=target, provider=provider)
    ledger_before = first.stage_ledger_path.read_bytes()

    def forbidden(*_args):
        raise AssertionError("provider was replayed")

    second = _run(tmp_path, stages=[stage], target=target, provider=forbidden)
    assert second.stage_ledger_path.read_bytes() == ledger_before
    assert second.scorer_replay_count == 0
    assert second.reused_measurement_count == 1


def test_exhaustive_completion_accounts_scoped_blocker(tmp_path: Path) -> None:
    target = _target()
    good = _physical(tmp_path, "good")
    bad = _physical(tmp_path, "bad")
    stages = [
        {"stage_tag": "good", "physical_stage_identity": good},
        {"stage_tag": "bad", "physical_stage_identity": bad},
    ]

    def provider(stage, _prior):
        if stage["stage_tag"] == "bad":
            raise RuntimeError("physical stage corrupt")
        return (
            _raw(
                tmp_path,
                tag="good",
                physical=good,
                live_target=target,
                wire_k=100,
            ),
            True,
        )

    result = _run(tmp_path, stages=stages, target=target, provider=provider)
    assert result.exhaustive_enumeration_proven is True
    assert result.discovered_stage_count == result.accounted_stage_count == 2
    assert result.blocked_stage_count == 1
    assert result.retained_stage_count == 1


def test_custodied_scoped_obstruction_supplies_its_own_live_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    physical = _physical(tmp_path, "scoped")
    source = g121._source_from_completed_physical(physical)
    obstruction = _binding(
        tmp_path / "scoped" / "g120_obstruction.json",
        b"g120-scoped-obstruction",
    )
    opened = SimpleNamespace(
        receipt_path=Path(str(obstruction["path"])),
        receipt_bytes=obstruction["bytes"],
        receipt_sha256=obstruction["sha256"],
        receipt={
            "live_target": target,
            "physical_stage_identity": physical,
        },
    )
    fake_g120 = SimpleNamespace(
        open_g120_exact_distortion_obstruction_v1=(
            lambda path, *, expected_sha256: opened
            if path == opened.receipt_path
            and expected_sha256 == opened.receipt_sha256
            else pytest.fail("wrong scoped obstruction binding")
        ),
    )
    monkeypatch.setattr(
        g121,
        "_require_safe_g120_v2",
        lambda: fake_g120,
    )
    row = g121._compile_blocked_attempt(
        stage_tag="scoped",
        source_stage_identity=source,
        live_target=target,
        blocker_code="G120ExactDistortionObstruction",
        blocker_detail="engine prefix crossed exact target",
        g120_scoped_obstruction=obstruction,
    )
    assert g121._validate_blocked_attempt(row) == row
    assert row["live_target"] == target
    assert row["prepose_obstruction"] == {
        "disposition": g121.BLOCKED_SCOPED,
    }
    assert row["g120_scoped_obstruction"] == obstruction

    opened.receipt["live_target"] = _target("0.17", pointer="b")
    with pytest.raises(
        g121.G121StageHarvestError,
        match="differs from blocked stage custody",
    ):
        g121._validate_blocked_attempt(row)


def test_pointer_matching_scoped_obstruction_reuses_without_scorer_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    physical = _physical(tmp_path, "reusable-scoped")
    source = g121._source_from_completed_physical(physical)
    obstruction = _binding(
        tmp_path / "reusable-scoped" / "g120-obstruction.json",
        b"g120-obstruction",
    )
    snapshot = g121.DynamicFrontierTargetSnapshot(
        pointer_path=str(tmp_path / "frontier.json"),
        pointer_bytes=100,
        pointer_sha256="1" * 64,
        pointer_device=1,
        pointer_inode=2,
        pointer_mtime_ns=3,
        last_refreshed_utc="2026-07-27T12:00:00+00:00",
        source_snapshot_at_utc=None,
        target_score=0.172,
        selected_axis="contest-CPU",
        selected_source="our_local_frontier_contest_cpu",
        selected_source_kind="exact",
        selected_score_precision="exact",
        selected_custody="archive",
        selected_evidence_grade="authority",
        selected_archive_sha256="2" * 64,
        selected_lane_id="lane",
        selected_hardware_substrate="cpu",
        selection_rule="minimum",
    )
    opened = SimpleNamespace(
        receipt={
            "pointer_snapshot": dict(vars(snapshot)),
            "live_target": target,
            "physical_stage_identity": physical,
            "seg_scorer": {"fresh_direct_scorer_calls": 0},
        },
    )
    fake_g120 = SimpleNamespace(
        open_g120_exact_distortion_obstruction_v1=(
            lambda path, *, expected_sha256: opened
            if path == Path(str(obstruction["path"]))
            and expected_sha256 == obstruction["sha256"]
            else pytest.fail("wrong reusable obstruction binding")
        ),
    )
    prior = g121._compile_blocked_attempt(
        stage_tag="reusable-scoped",
        source_stage_identity=source,
        live_target=target,
        blocker_code="G120ExactDistortionObstruction",
        blocker_detail="exact prefix crossed",
        g120_scoped_obstruction=obstruction,
    )
    stage = {
        "stage_tag": "reusable-scoped",
        "source_stage_identity": source,
    }
    monkeypatch.setattr(
        g121,
        "verify_dynamic_frontier_target_snapshot",
        lambda value: value,
    )
    assert (
        g121._open_reusable_scoped_obstruction(
            g120_module=fake_g120,
            prior=prior,
            stage=stage,
        )
        is opened
    )
    assert g121._scoped_obstruction_scorer_replay_count(opened) == 0
    opened.receipt["seg_scorer"]["fresh_direct_scorer_calls"] = 2
    assert g121._scoped_obstruction_scorer_replay_count(opened) == 1

    monkeypatch.setattr(
        g121,
        "verify_dynamic_frontier_target_snapshot",
        lambda _value: (_ for _ in ()).throw(
            g121.DynamicFrontierTargetError("pointer moved")
        ),
    )
    assert (
        g121._open_reusable_scoped_obstruction(
            g120_module=fake_g120,
            prior=prior,
            stage=stage,
        )
        is None
    )


def test_scoped_obstruction_replay_telemetry_rejects_malformed_calls() -> None:
    opened = SimpleNamespace(
        receipt={
            "seg_scorer": {
                "fresh_direct_scorer_calls": True,
            },
        },
    )
    with pytest.raises(
        g121.G121StageHarvestError,
        match="telemetry differs",
    ):
        g121._scoped_obstruction_scorer_replay_count(opened)


def test_strict_fixture_opener_round_trips_population(tmp_path: Path) -> None:
    target = _target()
    physical = _physical(tmp_path, "open")
    stage = {"stage_tag": "open", "physical_stage_identity": physical}

    def provider(_stage, _prior):
        return (
            _raw(
                tmp_path,
                tag="open",
                physical=physical,
                live_target=target,
                wire_k=100,
            ),
            True,
        )

    result = _run(tmp_path, stages=[stage], target=target, provider=provider)
    opened = g121._open_g121_retained_prepose_v1_test_only(
        result.retained_prepose_path,
        expected_sha256=result.retained_prepose_sha256,
        injected_inputs_are_test_only=True,
    )
    assert opened.schema == "tac.g121_retained_prepose.v2"
    assert len(opened.rows) == 1
    assert opened.rows[0].to_dict()["prepose_obstruction"][
        "disposition"
    ] == g121.RETAIN_POST_G105_POSE


def test_best_and_rolling_files_are_not_discovered(tmp_path: Path) -> None:
    lineage = tmp_path / "fresh_lineage"
    lineage.mkdir()
    (lineage / "levelset_best.json").write_text("{}")
    (lineage / "levelset_witness_ema_BEST.npz").write_bytes(b"best")
    assert g121._discover_physical_stages(
        tmp_path,
        expected_current_launch_dsl_compile_hash="0" * 64,
    ) == []


def test_discovery_uses_latest_tip_ancestry_filters_unaliased_tip_and_accepts_resume_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path.resolve()
    physical = producer / "fresh_lineage"
    physical.mkdir()
    old_hash = hashlib.sha256(b"old launch").hexdigest()
    latest_hash = hashlib.sha256(b"latest launch").hexdigest()
    root_sha = hashlib.sha256(b"cold root").hexdigest()

    def node(label: str, epoch: int, launch_hash: str):
        checkpoint_id = hashlib.sha256(f"checkpoint:{label}".encode()).hexdigest()
        receipt = physical / f"{checkpoint_id}.receipt.json"
        receipt.write_bytes(f"receipt:{label}".encode())
        deploy_payload = f"deploy:{label}".encode()
        resume_payload = f"resume:{label}".encode()
        deploy_path = physical / f"{checkpoint_id}.deploy.npz"
        resume_path = physical / f"{checkpoint_id}.resume.npz"
        deploy_path.write_bytes(deploy_payload)
        resume_path.write_bytes(resume_payload)
        pair = SimpleNamespace(
            checkpoint_id_sha256=checkpoint_id,
            epoch=epoch,
            stage="unify_tau",
            current_launch_dsl_compile_hash=launch_hash,
            deploy=SimpleNamespace(
                path=deploy_path,
                bytes=len(deploy_payload),
                sha256=hashlib.sha256(deploy_payload).hexdigest(),
            ),
            resume=SimpleNamespace(
                path=resume_path,
                bytes=len(resume_payload),
                sha256=hashlib.sha256(resume_payload).hexdigest(),
            ),
            native=None,
            complete_state_manifest_proven=True,
        )
        return SimpleNamespace(
            receipt_path=receipt,
            receipt_bytes=receipt.stat().st_size,
            receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            sequence_index=epoch - 1,
            pair=pair,
            complete_trajectory_proven=True,
        )

    preserved_old = node("preserved-old", 1, old_hash)
    periodic_latest = node("periodic-latest", 2, latest_hash)
    chain = lineage.FreshProducerPhysicalCheckpointChainV1(
        nodes=(preserved_old, periodic_latest),
        current=periodic_latest,
        root_sha256=root_sha,
        current_launch_dsl_compile_hash=latest_hash,
        complete_trajectory_proven=True,
    )
    monkeypatch.setattr(
        lineage,
        "open_fresh_physical_checkpoint_chain_v1",
        lambda *_args, **_kwargs: chain,
    )
    monkeypatch.setattr(
        lineage,
        "open_fresh_producer_checkpoint_pair_v1",
        lambda **_kwargs: preserved_old.pair,
    )
    (producer / "levelset_ckpt_stageCE_ep1.npz").write_bytes(
        preserved_old.pair.deploy.path.read_bytes()
    )
    (producer / "levelset_resume_stageCE_ep1.npz").write_bytes(
        preserved_old.pair.resume.path.read_bytes()
    )
    (producer / "fresh_lineage_tip.json").write_text(
        json.dumps(
            {
                "schema": "tac.fresh_producer_lineage_tip.v1",
                "receipt_path": str(periodic_latest.receipt_path),
                "receipt_sha256": periodic_latest.receipt_sha256,
                "receipt_bytes": periodic_latest.receipt_bytes,
                "checkpoint_id_sha256": (
                    periodic_latest.pair.checkpoint_id_sha256
                ),
                "root_sha256": root_sha,
                "sequence_index": periodic_latest.sequence_index,
                "epoch": periodic_latest.pair.epoch,
                "stage": periodic_latest.pair.stage,
                "complete_trajectory_proven": True,
            },
            sort_keys=True,
        )
    )

    stages = g121._discover_physical_stages(
        producer,
        expected_current_launch_dsl_compile_hash=latest_hash,
    )
    assert len(stages) == 1
    assert stages[0]["stage_tag"].startswith("stageCE.epoch_1.")
    assert stages[0]["chain"].current is preserved_old
    assert (
        stages[0]["chain"].current_launch_dsl_compile_hash == old_hash
    )


def test_preserved_alias_is_admitted_by_semantic_checkpoint_id_not_zip_bytes(
    tmp_path: Path,
) -> None:
    deploy, resume, launch_hash = lineage_fixtures._source_pair_arrays(
        epoch=0,
        stage="stageColdRoot",
    )
    node = lineage_fixtures._publish_node(
        tmp_path,
        name="semantic",
        deploy=deploy,
        resume=resume,
        launch_dsl=launch_hash,
    )
    producer = tmp_path / "run"
    with np.load(node.pair.deploy.path, allow_pickle=False) as archive:
        np.savez(
            producer / "levelset_ckpt_stageColdRoot_ep0.npz",
            **{key: archive[key] for key in archive.files},
        )
    with np.load(node.pair.resume.path, allow_pickle=False) as archive:
        np.savez(
            producer / "levelset_resume_stageColdRoot_ep0.npz",
            **{key: archive[key] for key in archive.files},
        )
    mapped = g121._preserved_stage_checkpoint_ids(producer, (node,))
    assert mapped == {node.pair.checkpoint_id_sha256: "stageColdRoot"}


def test_preserved_native_alias_reopens_complete_physical_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "run"
    producer.mkdir()
    checkpoint_id = "a" * 64
    launch_hash = "b" * 64
    for name in (
        "levelset_ckpt_stageColdRoot_ep0.npz",
        "levelset_resume_stageColdRoot_ep0.npz",
        "levelset_g111_native_stageColdRoot_ep0.npz",
    ):
        (producer / name).write_bytes(name.encode("ascii"))
    pair = SimpleNamespace(
        checkpoint_id_sha256=checkpoint_id,
        current_launch_dsl_compile_hash=launch_hash,
        epoch=0,
        native=SimpleNamespace(),
    )
    node = SimpleNamespace(pair=pair)
    calls: list[dict[str, object]] = []

    def _open_pair(**kwargs: object) -> object:
        calls.append(kwargs)
        return pair

    monkeypatch.setattr(
        lineage,
        "open_fresh_producer_checkpoint_pair_v1",
        _open_pair,
    )
    mapped = g121._preserved_stage_checkpoint_ids(producer, (node,))

    assert mapped == {checkpoint_id: "stageColdRoot"}
    assert len(calls) == 1
    assert calls[0]["native_checkpoint"] == (
        producer / "levelset_g111_native_stageColdRoot_ep0.npz"
    ).resolve()
    assert calls[0]["expected_native_sha256"] == hashlib.sha256(
        b"levelset_g111_native_stageColdRoot_ep0.npz"
    ).hexdigest()


def test_complete_periodic_native_triplet_reopens_physical_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "run"
    producer.mkdir()
    checkpoint_id = "a" * 64
    launch_hash = "b" * 64
    names = (
        "levelset_periodic_ema_stage_unify_tau_ep25.npz",
        "levelset_periodic_resume_stage_unify_tau_ep25.npz",
        "levelset_g111_native_stage_unify_tau_periodic_ep25.npz",
    )
    for name in names:
        (producer / name).write_bytes(name.encode("ascii"))
    pair = SimpleNamespace(
        checkpoint_id_sha256=checkpoint_id,
        current_launch_dsl_compile_hash=launch_hash,
        epoch=25,
        native=SimpleNamespace(),
    )
    node = SimpleNamespace(pair=pair)
    calls: list[dict[str, object]] = []

    def _open_pair(**kwargs: object) -> object:
        calls.append(kwargs)
        return pair

    monkeypatch.setattr(
        lineage,
        "open_fresh_producer_checkpoint_pair_v1",
        _open_pair,
    )
    mapped = g121._preserved_stage_checkpoint_ids(producer, (node,))

    assert mapped == {checkpoint_id: "stage_unify_tau_periodic"}
    assert len(calls) == 1
    assert calls[0]["deploy_checkpoint"] == (producer / names[0]).resolve()
    assert calls[0]["resume_checkpoint"] == (producer / names[1]).resolve()
    assert calls[0]["native_checkpoint"] == (producer / names[2]).resolve()


def test_incomplete_or_symlinked_periodic_triplet_is_not_discovered(
    tmp_path: Path,
) -> None:
    producer = tmp_path / "run"
    producer.mkdir()
    deploy = producer / "levelset_periodic_ema_stage_unify_tau_ep25.npz"
    resume = producer / "levelset_periodic_resume_stage_unify_tau_ep25.npz"
    native = (
        producer
        / "levelset_g111_native_stage_unify_tau_periodic_ep25.npz"
    )
    deploy.write_bytes(b"deploy")
    resume.write_bytes(b"resume")

    assert g121._complete_periodic_alias_triplets(producer) == ()

    native.symlink_to(deploy)
    assert g121._complete_periodic_alias_triplets(producer) == ()


def test_private_injection_requires_explicit_fixture_marker(tmp_path: Path) -> None:
    with pytest.raises(g121.G121StageHarvestError, match="test_only"):
        g121._harvest_g111_stages_v1_test_only(
            stages=[],
            live_target=_target(),
            launch_manifest=tmp_path / "missing",
            output_dir=tmp_path / "out",
            progress_dir=tmp_path / "progress",
            provider=lambda *_args: ({}, False),
            injected_inputs_are_test_only=False,
        )
