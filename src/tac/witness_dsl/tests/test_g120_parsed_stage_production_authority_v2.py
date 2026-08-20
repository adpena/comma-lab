from __future__ import annotations

import hashlib
import inspect
import io
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.witness_control import taskspace_g121_resumable_stage_harvest_v1 as g121
from tac.witness_dsl import g120_parsed_stage_production_authority_v2 as subject
from tac.witness_dsl.dynamic_frontier_target import DynamicFrontierTargetSnapshot


def test_production_signature_is_physical_and_has_no_cross_stage_reducer() -> None:
    assert set(inspect.signature(subject.run_g120_parsed_stage_production_authority_v2).parameters) == {
        "repo_root",
        "g112_partition_receipt",
        "expected_g112_partition_receipt_sha256",
        "out_dir",
        "progress_dir",
        "measurement_cache_dir",
        "prior_measurement_receipt",
        "expected_prior_measurement_receipt_sha256",
    }
    assert subject.MEASUREMENT_SCHEMA == "tac.g120_stage_measurement.v2"
    assert subject.OBSERVATION_SCHEMA == "tac.g120_stage_observation.v2"
    assert subject.MINIMUM_SAFE_AUTHORITY_COMPLETE is True
    source = inspect.getsource(subject.run_g120_parsed_stage_production_authority_v2)
    assert "cross_stage" not in source
    assert "BEST" not in source
    assert "pareto" not in source.lower()
    assert 'durable_progress / "g117_engine_progress"' in source
    assert 'engine_root / "progress"' not in source
    assert "except G111ParsedG105ExactPrefixObstruction" in source
    assert "raise G120ExactDistortionObstruction" in source


def _scoped_obstruction_inputs() -> dict[str, object]:
    pointer_identity = _sha("scoped-pointer")
    physical_identity = {
        "g112_partition_receipt": {
            "path": "/physical/g112.json",
            "bytes": 10,
            "sha256": _sha("g112"),
        },
    }
    authority = SimpleNamespace(
        stage_tag="stage_unify_tau.epoch_25.chk_aaaaaaaaaaaa",
        physical_stage_identity=physical_identity,
        physical_stage_identity_sha256=_sha("physical-stage"),
        target_labels=np.zeros((1, 2, 2), dtype=np.uint8),
        seg_scorer_identity_sha256=_sha("seg-scorer"),
        g112=SimpleNamespace(
            initializer=SimpleNamespace(
                checkpoint_sha256=_sha("pose-initializer"),
            ),
        ),
        g109_custody={
            "segnet_weights": {
                "path": "/physical/segnet.pth",
                "bytes": 10,
                "sha256": _sha("weights"),
            },
            "upstream_closure": {"tree_sha256": _sha("upstream")},
            "scorer_runtime_identity_sha256": _sha("seg-scorer"),
        },
    )
    engine_receipt = {
        "stage_tag": authority.stage_tag,
        "completed_batch_count": 2,
        "effective_frontier_target_exact": {
            "decimal": "0.172",
            "numerator": 43,
            "denominator": 250,
        },
        "progress_identity": {
            "stage_tag": authority.stage_tag,
            "source_checkpoint_identity_sha256": (
                authority.physical_stage_identity_sha256
            ),
            "target_labels_sha256": hashlib.sha256(
                memoryview(authority.target_labels)
            ).hexdigest(),
            "seg_scorer_identity_sha256": (
                authority.seg_scorer_identity_sha256
            ),
            "pointer_snapshot_identity_sha256": pointer_identity,
            "pose_initializer_identity_sha256": (
                authority.g112.initializer.checkpoint_sha256
            ),
            "pair_count": subject.PRODUCTION_PAIR_COUNT,
            "batch_sizes": list(subject.VERDICT_BATCH_SIZES),
        },
    }
    engine = SimpleNamespace(
        receipt_path=Path("/physical/g111-obstruction.json"),
        receipt_sha256=_sha("g111-obstruction"),
        receipt_bytes=100,
        receipt=engine_receipt,
    )
    runtime = {
        "root": "/physical/runtime",
        "files": [],
        "tree_sha256": _sha("runtime"),
        "physical_tree_identity_sha256": _sha("runtime-physical"),
    }
    postverified = {
        "segnet_weights_postverified": authority.g109_custody[
            "segnet_weights"
        ],
        "upstream_closure_postverified": authority.g109_custody[
            "upstream_closure"
        ],
        "public_runtime_postverified": runtime,
    }
    return {
        "engine_obstruction": engine,
        "authority": authority,
        "snapshot": _snapshot(0.172, suffix="scoped"),
        "public_runtime_pre": runtime,
        "postverified": postverified,
        "scorer_calls": 2,
        "pointer_identity": pointer_identity,
    }


def test_engine_prefix_obstruction_is_scoped_not_public_prune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _scoped_obstruction_inputs()
    monkeypatch.setattr(
        subject,
        "_exact_target_from_snapshot",
        lambda _snapshot_value: ("0.172", 43, 250),
    )
    monkeypatch.setattr(
        subject._v1,
        "dynamic_snapshot_identity_sha256",
        lambda _snapshot_value: inputs["pointer_identity"],
    )
    receipt = subject._build_exact_distortion_obstruction_receipt(
        **{
            key: value
            for key, value in inputs.items()
            if key != "pointer_identity"
        },
    )
    assert receipt["schema"] == (
        subject.EXACT_DISTORTION_OBSTRUCTION_SCHEMA
    )
    assert receipt["disposition"] == subject.BLOCKED_SCOPED
    assert receipt["production_authority_closed"] is False
    assert receipt["public_runtime"][
        "public_prefix_execution_performed"
    ] is False
    assert receipt["public_runtime"]["public_prefix_equality"] is False
    assert receipt["public_runtime"]["cross_wire_prefix_equality"] is False
    assert receipt["false_authority"]["public_wire_prune_claim"] is False
    assert receipt["false_authority"]["family_wide_claim"] is False
    assert receipt["obstruction_identity_sha256"] == (
        subject._receipt_identity(
            receipt,
            identity_field="obstruction_identity_sha256",
        )
    )


def test_scoped_obstruction_opener_recurses_target_scorer_and_axis_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _scoped_obstruction_inputs()
    authority = inputs["authority"]
    engine = inputs["engine_obstruction"]
    monkeypatch.setattr(
        subject,
        "_exact_target_from_snapshot",
        lambda _snapshot_value: ("0.172", 43, 250),
    )
    monkeypatch.setattr(
        subject._v1,
        "dynamic_snapshot_identity_sha256",
        lambda _snapshot_value: inputs["pointer_identity"],
    )
    receipt = subject._build_exact_distortion_obstruction_receipt(
        **{
            key: value
            for key, value in inputs.items()
            if key != "pointer_identity"
        },
    )
    receipt_path = tmp_path / "g120-obstruction.json"
    payload = subject._canonical_json(receipt)
    receipt_path.write_bytes(payload)
    monkeypatch.setattr(
        subject,
        "_reopen_binding",
        lambda _value, *, name: b"receipt",
    )
    monkeypatch.setattr(
        subject,
        "open_g111_parsed_g105_exact_prefix_obstruction_v1",
        lambda *_args, **_kwargs: engine,
    )
    monkeypatch.setattr(
        subject,
        "open_g112_partition_receipt",
        lambda *_args, **_kwargs: authority.g112,
    )
    monkeypatch.setattr(
        subject._v1,
        "_physical_stage_identity",
        lambda _g112: (
            authority.physical_stage_identity,
            authority.physical_stage_identity_sha256,
            authority.stage_tag,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_reopen_g109_target_custody",
        lambda _value, *, g112, context: (
            authority.g109_custody,
            authority.target_labels,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_reopen_runtime_tree",
        lambda value: value,
    )
    monkeypatch.setattr(
        subject._v1,
        "_regular_file_identity",
        lambda _path, *, name: authority.g109_custody[
            "segnet_weights"
        ],
    )
    monkeypatch.setattr(
        subject._v1,
        "_reopen_upstream_closure",
        lambda value: value,
    )

    opened = subject.open_g120_exact_distortion_obstruction_v1(
        receipt_path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert opened.receipt["evidence_axis"] == subject.EVIDENCE_AXIS

    monkeypatch.setattr(
        subject,
        "_reopen_g109_target_custody",
        lambda _value, *, g112, context: (
            authority.g109_custody,
            np.ones_like(authority.target_labels),
        ),
    )
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="custody, target, or authority",
    ):
        subject.open_g120_exact_distortion_obstruction_v1(
            receipt_path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )

    monkeypatch.setattr(
        subject,
        "_reopen_g109_target_custody",
        lambda _value, *, g112, context: (
            authority.g109_custody,
            authority.target_labels,
        ),
    )
    changed = dict(receipt)
    changed["evidence_axis"] = "[wrong-axis]"
    changed["obstruction_identity_sha256"] = subject._receipt_identity(
        changed,
        identity_field="obstruction_identity_sha256",
    )
    changed_payload = subject._canonical_json(changed)
    changed_path = tmp_path / "g120-obstruction-wrong-axis.json"
    changed_path.write_bytes(changed_payload)
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="custody, target, or authority",
    ):
        subject.open_g120_exact_distortion_obstruction_v1(
            changed_path,
            expected_sha256=hashlib.sha256(changed_payload).hexdigest(),
        )


def test_scoped_obstruction_refuses_runtime_change_and_pointer_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _scoped_obstruction_inputs()
    monkeypatch.setattr(
        subject,
        "_exact_target_from_snapshot",
        lambda _snapshot_value: ("0.172", 43, 250),
    )
    monkeypatch.setattr(
        subject._v1,
        "dynamic_snapshot_identity_sha256",
        lambda _snapshot_value: inputs["pointer_identity"],
    )
    changed = dict(inputs["postverified"])
    changed["public_runtime_postverified"] = {
        **inputs["public_runtime_pre"],
        "tree_sha256": _sha("changed-runtime"),
    }
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="postverified public runtime",
    ):
        subject._build_exact_distortion_obstruction_receipt(
            engine_obstruction=inputs["engine_obstruction"],
            authority=inputs["authority"],
            snapshot=inputs["snapshot"],
            public_runtime_pre=inputs["public_runtime_pre"],
            postverified=changed,
            scorer_calls=2,
        )

    repo = tmp_path.resolve() / "repo"
    out = repo / "out"
    out.mkdir(parents=True)
    monkeypatch.setattr(
        subject,
        "verify_dynamic_frontier_target_snapshot",
        lambda _snapshot_value: (_ for _ in ()).throw(
            RuntimeError("pointer moved"),
        ),
    )
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="frontier pointer changed",
    ):
        subject._commit_exact_distortion_obstruction(
            repo_root=repo,
            out_dir=out,
            engine_obstruction=inputs["engine_obstruction"],
            authority=inputs["authority"],
            snapshot=inputs["snapshot"],
            public_runtime_pre=inputs["public_runtime_pre"],
            postverified=inputs["postverified"],
            scorer_calls=2,
        )
    assert not tuple(out.iterdir())


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _snapshot(score: float, *, suffix: str) -> DynamicFrontierTargetSnapshot:
    return DynamicFrontierTargetSnapshot(
        pointer_path=f"/physical/pointer-{suffix}.json",
        pointer_bytes=100,
        pointer_sha256=_sha(f"pointer-{suffix}"),
        pointer_device=1,
        pointer_inode=2,
        pointer_mtime_ns=3,
        last_refreshed_utc="2026-07-27T12:00:00+00:00",
        source_snapshot_at_utc=None,
        target_score=score,
        selected_axis="contest-CPU",
        selected_source="our_local_frontier_contest_cpu",
        selected_source_kind="exact",
        selected_score_precision="exact",
        selected_custody="archive",
        selected_evidence_grade="authority",
        selected_archive_sha256=_sha("archive"),
        selected_lane_id="lane",
        selected_hardware_substrate="cpu",
        selection_rule="minimum",
    )


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _unmeasured_coordinates() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return subject._measurement_status_defaults()


def _measured_coordinates(
    tmp_path: Path,
    *,
    source_k: int,
    wire_k: int,
) -> tuple[dict[str, object], dict[str, object]]:
    source_receipt = tmp_path / f"source-{source_k}.json"
    source_receipt.write_text("source\n", encoding="ascii")
    regret_receipt = tmp_path / f"regret-{source_k}-{wire_k}.json"
    regret_receipt.write_text("regret\n", encoding="ascii")
    return (
        {
            "status": "measured",
            "disagreement_pixels": source_k,
            "pixel_denominator": subject.PIXEL_DENOMINATOR,
            "measurement_receipt": _binding(source_receipt),
        },
        {
            "status": "measured",
            "disagreement_delta_pixels": wire_k - source_k,
            "rational": {
                "numerator": wire_k - source_k,
                "denominator": subject.PIXEL_DENOMINATOR,
            },
            "receipt": _binding(regret_receipt),
        },
    )


def test_prior_measurement_arguments_are_both_or_neither(tmp_path: Path) -> None:
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="supplied together",
    ):
        subject.run_g120_parsed_stage_production_authority_v2(
            repo_root=tmp_path.resolve(),
            g112_partition_receipt=(tmp_path / "g112.json").resolve(),
            expected_g112_partition_receipt_sha256=_sha("g112"),
            out_dir=(tmp_path / "out").resolve(),
            progress_dir=(tmp_path / "progress").resolve(),
            measurement_cache_dir=(tmp_path / "cache").resolve(),
            prior_measurement_receipt=(tmp_path / "prior.json").resolve(),
        )


def test_exact_equality_prunes_only_with_measured_blocked_float(
    tmp_path: Path,
) -> None:
    k = 177
    target = Fraction(100 * k, subject.PIXEL_DENOMINATOR)
    source, regret = _measured_coordinates(
        tmp_path,
        source_k=k,
        wire_k=k,
    )
    _unused_source, _unused_regret, qat = _unmeasured_coordinates()
    observed = subject.exact_prepose_obstruction(
        disagreement_pixels=k,
        pixel_denominator=subject.PIXEL_DENOMINATOR,
        target_numerator=target.numerator,
        target_denominator=target.denominator,
        source_float_seg=source,
        wire_regret=regret,
        g115_qat=qat,
    )
    assert observed["strict_distortion_open"] is False
    assert observed["disposition"] == subject.PRUNE_EXACT_DISTORTION_OBSTRUCTION
    assert int(observed["lhs"]) == int(observed["rhs"])


def test_wire_blocked_float_open_defers_and_signed_regret_is_real(
    tmp_path: Path,
) -> None:
    wire_k = 200
    source_k = 100
    target = Fraction(15_000, subject.PIXEL_DENOMINATOR)
    source, regret = _measured_coordinates(
        tmp_path,
        source_k=source_k,
        wire_k=wire_k,
    )
    _unused_source, _unused_regret, qat = _unmeasured_coordinates()
    observed = subject.exact_prepose_obstruction(
        disagreement_pixels=wire_k,
        pixel_denominator=subject.PIXEL_DENOMINATOR,
        target_numerator=target.numerator,
        target_denominator=target.denominator,
        source_float_seg=source,
        wire_regret=regret,
        g115_qat=qat,
    )
    assert observed["strict_distortion_open"] is False
    assert observed["disposition"] == subject.DEFER_G115_WIRE_QAT

    negative_source, negative_regret = _measured_coordinates(
        tmp_path,
        source_k=200,
        wire_k=100,
    )
    retained = subject.exact_prepose_obstruction(
        disagreement_pixels=100,
        pixel_denominator=subject.PIXEL_DENOMINATOR,
        target_numerator=target.numerator,
        target_denominator=target.denominator,
        source_float_seg=negative_source,
        wire_regret=negative_regret,
        g115_qat=qat,
    )
    assert negative_regret["disagreement_delta_pixels"] == -100
    assert retained["disposition"] == subject.RETAIN_POST_G105_POSE


def test_forged_signed_regret_is_rejected(tmp_path: Path) -> None:
    source, regret = _measured_coordinates(
        tmp_path,
        source_k=10,
        wire_k=20,
    )
    regret["disagreement_delta_pixels"] = -10
    regret["rational"] = {
        "numerator": -10,
        "denominator": subject.PIXEL_DENOMINATOR,
    }
    _unused_source, _unused_regret, qat = _unmeasured_coordinates()
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="source/wire counts",
    ):
        subject.exact_prepose_obstruction(
            disagreement_pixels=20,
            pixel_denominator=subject.PIXEL_DENOMINATOR,
            target_numerator=1,
            target_denominator=1,
            source_float_seg=source,
            wire_regret=regret,
            g115_qat=qat,
        )


def test_terminal_qat_requires_exact_k_and_never_false_prunes_open_child(
    tmp_path: Path,
) -> None:
    source, regret = _measured_coordinates(
        tmp_path,
        source_k=1,
        wire_k=200,
    )
    terminal_receipt = tmp_path / "terminal.json"
    terminal_receipt.write_text("terminal\n", encoding="ascii")
    target = Fraction(15_000, subject.PIXEL_DENOMINATOR)
    qat = {
        "status": "terminal_stage_measured",
        "terminal_stage_physical_identity_sha256": _sha("terminal"),
        "disagreement_pixels": 100,
        "pixel_denominator": subject.PIXEL_DENOMINATOR,
        "receipt": _binding(terminal_receipt),
    }
    observed = subject.exact_prepose_obstruction(
        disagreement_pixels=200,
        pixel_denominator=subject.PIXEL_DENOMINATOR,
        target_numerator=target.numerator,
        target_denominator=target.denominator,
        source_float_seg=source,
        wire_regret=regret,
        g115_qat=qat,
    )
    assert observed["disposition"] == subject.RETAIN_POST_G105_POSE
    qat["disagreement_pixels"] = 200
    observed = subject.exact_prepose_obstruction(
        disagreement_pixels=200,
        pixel_denominator=subject.PIXEL_DENOMINATOR,
        target_numerator=target.numerator,
        target_denominator=target.denominator,
        source_float_seg=source,
        wire_regret=regret,
        g115_qat=qat,
    )
    assert observed["disposition"] == subject.PRUNE_EXACT_DISTORTION_OBSTRUCTION


def test_g120_g121_exact_terminal_qat_dispositions_are_identical(
    tmp_path: Path,
) -> None:
    source, regret = _measured_coordinates(
        tmp_path,
        source_k=1,
        wire_k=200,
    )
    terminal_receipt = tmp_path / "terminal-cross-module.json"
    terminal_receipt.write_text("terminal\n", encoding="ascii")
    target = Fraction(15_000, subject.PIXEL_DENOMINATOR)
    for terminal_k in (100, 200):
        qat = {
            "status": "terminal_stage_measured",
            "terminal_stage_physical_identity_sha256": _sha("terminal"),
            "disagreement_pixels": terminal_k,
            "pixel_denominator": subject.PIXEL_DENOMINATOR,
            "receipt": _binding(terminal_receipt),
        }
        g120_observed = subject.exact_prepose_obstruction(
            disagreement_pixels=200,
            pixel_denominator=subject.PIXEL_DENOMINATOR,
            target_numerator=target.numerator,
            target_denominator=target.denominator,
            source_float_seg=source,
            wire_regret=regret,
            g115_qat=qat,
        )
        g121_observed = g121._exact_obstruction(
            wire_disagreements=200,
            target=target,
            source_disagreements=1,
            g115_qat=qat,
        )
        assert g120_observed == g121_observed


def test_exact_pointer_decimal_preserves_lexical_precision(
    tmp_path: Path,
) -> None:
    pointer = tmp_path / "pointer.json"
    raw = b'{"our_local_frontier_contest_cpu":{"score":0.15000000000000001}}\n'
    pointer.write_bytes(raw)
    snapshot = DynamicFrontierTargetSnapshot(
        pointer_path=str(pointer.resolve()),
        pointer_bytes=len(raw),
        pointer_sha256=hashlib.sha256(raw).hexdigest(),
        pointer_device=pointer.stat().st_dev,
        pointer_inode=pointer.stat().st_ino,
        pointer_mtime_ns=pointer.stat().st_mtime_ns,
        last_refreshed_utc="2026-07-27T12:00:00+00:00",
        source_snapshot_at_utc=None,
        target_score=float("0.15000000000000001"),
        selected_axis="contest-CPU",
        selected_source="our_local_frontier_contest_cpu",
        selected_source_kind="exact",
        selected_score_precision="exact",
        selected_custody="archive",
        selected_evidence_grade="authority",
        selected_archive_sha256=_sha("archive"),
        selected_lane_id="lane",
        selected_hardware_substrate="cpu",
        selection_rule="minimum",
    )
    lexical, numerator, denominator = subject._exact_target_from_snapshot(snapshot)
    assert lexical == "0.15000000000000001"
    assert Fraction(numerator, denominator) == Fraction(
        15_000_000_000_000_001,
        100_000_000_000_000_000,
    )


def test_fresh_prediction_broker_bypasses_disk_cache_and_memoizes() -> None:
    class Scorer:
        def __init__(self) -> None:
            self.direct_calls = 0
            self.cache_calls = 0

        def __call__(self, _camera: np.ndarray) -> np.ndarray:
            self.cache_calls += 1
            raise AssertionError("legacy disk cache was trusted")

        def _predict(self, camera: np.ndarray) -> np.ndarray:
            self.direct_calls += 1
            return np.zeros(
                (camera.shape[0], *subject.PRODUCTION_SEG_HW),
                dtype=np.uint8,
            )

    scorer = Scorer()
    broker = subject._FreshPredictionBroker(scorer)
    camera = np.zeros((1, 874, 1164, 3), dtype=np.uint8)
    assert np.array_equal(broker(camera), broker(camera))
    assert scorer.direct_calls == 1
    assert scorer.cache_calls == 0
    assert broker.actual_scorer_calls == 1


def test_physical_prediction_array_tamper_is_recomputed_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "VERDICT_BATCH_SIZES", (1,))
    monkeypatch.setattr(subject, "PRODUCTION_PAIR_COUNT", 1)
    monkeypatch.setattr(subject, "PRODUCTION_SEG_HW", (2, 2))
    target = np.zeros((1, 2, 2), dtype=np.uint8)
    predicted = np.zeros((1, 2, 2), dtype=np.uint8)
    buffer = io.BytesIO()
    np.save(buffer, predicted, allow_pickle=False)
    prediction_path = tmp_path / "prediction.npy"
    prediction_path.write_bytes(buffer.getvalue())
    body = {
        "schema": subject.BATCH_SCHEMA,
        "measurement_execution_key_sha256": _sha("execution"),
        "batch_index": 0,
        "pair_start": 0,
        "pair_stop": 1,
        "target_labels_batch_sha256": hashlib.sha256(memoryview(target)).hexdigest(),
        "scorer_y1_batch_sha256": _sha("scorer"),
        "camera_y1_batch_sha256": _sha("camera"),
        "predicted_labels_batch_sha256": hashlib.sha256(memoryview(predicted)).hexdigest(),
        "prediction_file": _binding(prediction_path),
        "disagreement_pixels": 0,
    }
    row = {
        **body,
        "row_identity_sha256": hashlib.sha256(subject._canonical_json(body)).hexdigest(),
    }
    receipt_path = tmp_path / "batch.json"
    receipt_path.write_bytes(subject._canonical_json(row))
    embedded = {**row, "physical_receipt": _binding(receipt_path)}
    assert (
        subject._reopen_prediction_batches(
            [embedded],
            target_labels=target,
        )[1]
        == 0
    )
    with prediction_path.open("wb") as stream:
        np.save(stream, np.ones((1, 2, 2), dtype=np.uint8), allow_pickle=False)
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="physical identity differs",
    ):
        subject._reopen_prediction_batches(
            [embedded],
            target_labels=target,
        )


def test_measurement_identity_ignores_pointer_but_binds_nested_public_wire() -> None:
    receipt = {
        "schema": subject.MEASUREMENT_SCHEMA,
        "measurement_identity_sha256": None,
        "public_wire_seg": {
            "measurement_identity_sha256": None,
            "disagreement_pixels": 1,
        },
        "physical": _sha("physical"),
    }
    identity = subject._measurement_identity(receipt)
    receipt["measurement_identity_sha256"] = identity
    receipt["public_wire_seg"]["measurement_identity_sha256"] = identity
    assert subject._measurement_identity(receipt) == identity
    observation_a = {"target": "0.15", "measurement": identity}
    observation_b = {"target": "0.14", "measurement": identity}
    assert observation_a != observation_b
    assert receipt["measurement_identity_sha256"] == identity


def test_four_way_matrix_is_exhaustive_and_never_dominance_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "_reopen_binding",
        lambda value, *, name: b"semantic" if "semantic" in name else b"product" if "product" in name else b"archive",
    )
    monkeypatch.setattr(
        subject,
        "parse_g110_public_archive",
        lambda _archive: b"product",
    )
    monkeypatch.setattr(
        subject,
        "parse_g110_counted_archive_variant",
        lambda _archive, _method: b"product",
    )
    monkeypatch.setattr(
        subject,
        "parse_g110_two_layer_v1",
        lambda _product: SimpleNamespace(
            packet=b"product",
            semantic_packet=b"semantic",
        ),
    )
    monkeypatch.setattr(
        subject._v1.g105_adapter,
        "parse_packet",
        lambda _semantic: b"parsed-semantic",
    )
    monkeypatch.setattr(
        subject._v1.g105_adapter,
        "encode_packet",
        lambda _program: b"semantic",
    )
    rows = []
    for codec in ("RAW_I16_LE", "DELTA_RICE_BEST_K"):
        for method in ("STORE", "DEFLATE"):
            body = {
                "y1_wire_codec": codec,
                "outer_zip_method": method,
                "disagreement_pixels": 1,
                "pixel_denominator": subject.PIXEL_DENOMINATOR,
                "d_seg_rational": {
                    "numerator": 1,
                    "denominator": subject.PIXEL_DENOMINATOR,
                },
                "d_seg_display_float": 1 / subject.PIXEL_DENOMINATOR,
                "semantic_action_display_float": subject._v1.semantic_stage_action(
                    d_seg=1 / subject.PIXEL_DENOMINATOR,
                    archive_bytes=100,
                ),
                "semantic_packet": {},
                "g110_product_packet": {},
                "archive": {"bytes": 100, "sha256": _sha("archive")},
                "g105_quantization_receipt_sha256": _sha("quant"),
                "scorer_y1_population_sha256": _sha("scorer"),
                "camera_y1_population_sha256": _sha("camera"),
                "predicted_labels_population_sha256": _sha("prediction"),
                "engine_progress_is_authority": False,
            }
            rows.append(
                {
                    **body,
                    "alternative_identity_sha256": hashlib.sha256(subject._canonical_json(body)).hexdigest(),
                }
            )
    selected = subject._validate_alternatives(
        rows,
        selected_identity=rows[0]["alternative_identity_sha256"],
    )
    assert selected is rows[0]
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="canonical exact G117 minimum",
    ):
        subject._validate_alternatives(
            rows,
            selected_identity=rows[-1]["alternative_identity_sha256"],
        )
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="four-way",
    ):
        subject._validate_alternatives(
            rows[:-1],
            selected_identity=rows[0]["alternative_identity_sha256"],
        )
    monkeypatch.setattr(
        subject,
        "parse_g110_two_layer_v1",
        lambda _product: SimpleNamespace(
            packet=b"product",
            semantic_packet=b"detached-semantic",
        ),
    )
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="parse-back",
    ):
        subject._validate_alternatives(
            rows,
            selected_identity=rows[0]["alternative_identity_sha256"],
        )


def test_public_dispatch_binds_nested_g105_packet_not_outer_g110_packet() -> None:
    semantic_packet = b"nested-g105"
    product_packet = b"outer-g110"
    row = {
        "semantic_packet": {"sha256": hashlib.sha256(semantic_packet).hexdigest()},
        "g110_product_packet": {"sha256": hashlib.sha256(product_packet).hexdigest()},
    }
    subject._verify_selected_semantic_packet(semantic_packet, row)
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="G105 semantic packet",
    ):
        subject._verify_selected_semantic_packet(product_packet, row)


def test_runtime_mutation_between_capture_and_postverify_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    weight_binding = _binding(weights)
    closure = {"sealed": True}
    authority = SimpleNamespace(
        g109_custody={
            "upstream_closure": closure,
            "segnet_weights": weight_binding,
        }
    )
    monkeypatch.setattr(
        subject._v1,
        "_reopen_upstream_closure",
        lambda value: value,
    )
    monkeypatch.setattr(
        subject._v1,
        "_regular_file_identity",
        lambda _path, *, name: weight_binding,
    )
    monkeypatch.setattr(
        subject,
        "_capture_public_runtime",
        lambda _repo: ({"tree_sha256": _sha("changed")}, {}),
    )
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="public runtime changed",
    ):
        subject._postverify_authority_sources(
            repo_root=tmp_path,
            authority=authority,
            public_runtime_pre={"tree_sha256": _sha("before")},
        )


def test_pointer_only_refresh_reuses_completed_measurement_without_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject._v1, "_EPHEMERAL_ROOTS", ())
    monkeypatch.setattr(
        subject,
        "_ssd_cache_dir",
        lambda path: subject._durable_dir(path, name="cache"),
    )
    repo = tmp_path.resolve()
    runtime_root = (repo / subject._v1.PUBLIC_RUNTIME_RELATIVE_ROOT).resolve()
    measurement_path = repo / "prior-measurement.json"
    measurement_path.write_bytes(b"prior\n")
    measurement_sha = hashlib.sha256(b"prior\n").hexdigest()
    g112_path = repo / "g112.json"
    measurement_receipt = {
        "stage_tag": "tau",
        "measurement_identity_sha256": _sha("measurement"),
        "g112_partition_receipt": {
            "path": str(g112_path),
            "bytes": 1,
            "sha256": _sha("g112"),
        },
        "public_runtime": {"source_pre": {"root": str(runtime_root)}},
        "public_wire_seg": {
            "disagreement_pixels": 1,
            "pixel_denominator": subject.PIXEL_DENOMINATOR,
            "d_seg_rational": {
                "numerator": 1,
                "denominator": subject.PIXEL_DENOMINATOR,
            },
            "d_seg_display_float": 1 / subject.PIXEL_DENOMINATOR,
            "measurement_identity_sha256": _sha("measurement"),
        },
        "source_float_seg": _unmeasured_coordinates()[0],
        "wire_regret": _unmeasured_coordinates()[1],
        "g115_qat": _unmeasured_coordinates()[2],
    }
    measurement = subject.G120StageMeasurementV2(
        receipt_path=measurement_path,
        receipt_sha256=measurement_sha,
        receipt_bytes=len(b"prior\n"),
        receipt=measurement_receipt,
    )
    snapshots = [_snapshot(0.15, suffix="a"), _snapshot(0.14, suffix="b")]
    pointer_ids = [_sha("pointer-a"), _sha("pointer-b")]
    monkeypatch.setattr(
        subject,
        "load_dynamic_frontier_target",
        lambda **_kwargs: snapshots.pop(0),
    )
    monkeypatch.setattr(
        subject._v1,
        "dynamic_snapshot_identity_sha256",
        lambda snapshot: pointer_ids[0] if snapshot.target_score == 0.15 else pointer_ids[1],
    )
    monkeypatch.setattr(
        subject,
        "_exact_target_from_snapshot",
        lambda snapshot: (
            str(snapshot.target_score),
            Fraction(str(snapshot.target_score)).numerator,
            Fraction(str(snapshot.target_score)).denominator,
        ),
    )
    monkeypatch.setattr(
        subject,
        "open_g120_stage_measurement_v2",
        lambda *_args, **_kwargs: measurement,
    )
    monkeypatch.setattr(
        subject._v1,
        "_open_production_authority",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("fresh scorer/model path was invoked")),
    )
    monkeypatch.setattr(
        subject,
        "verify_dynamic_frontier_target_snapshot",
        lambda snapshot: snapshot,
    )

    def open_observation(path: Path, *, expected_sha256: str):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        receipt = json.loads(raw)
        return subject.G120StageObservationV2(
            receipt_path=path,
            receipt_sha256=expected_sha256,
            receipt_bytes=len(raw),
            receipt=receipt,
        )

    monkeypatch.setattr(
        subject,
        "open_g120_stage_observation_v2",
        open_observation,
    )
    kwargs = {
        "repo_root": repo,
        "g112_partition_receipt": g112_path,
        "expected_g112_partition_receipt_sha256": _sha("g112"),
        "out_dir": repo / "out",
        "progress_dir": repo / "progress",
        "measurement_cache_dir": repo / "cache",
        "prior_measurement_receipt": measurement_path,
        "expected_prior_measurement_receipt_sha256": measurement_sha,
    }
    first = subject.run_g120_parsed_stage_production_authority_v2(**kwargs)
    second = subject.run_g120_parsed_stage_production_authority_v2(**kwargs)
    assert first.measurement.measurement_identity_sha256 == second.measurement.measurement_identity_sha256
    assert first.observation.observation_identity_sha256 != second.observation.observation_identity_sha256
    before = set((repo / "out").glob("*g120_stage_observation.v2.json"))
    snapshots.append(_snapshot(0.13, suffix="c"))
    monkeypatch.setattr(
        subject,
        "verify_dynamic_frontier_target_snapshot",
        lambda _snapshot_value: (_ for _ in ()).throw(RuntimeError("pointer changed")),
    )
    with pytest.raises(
        subject.G120ProductionAuthorityV2Error,
        match="pointer changed",
    ):
        subject.run_g120_parsed_stage_production_authority_v2(**kwargs)
    assert set((repo / "out").glob("*g120_stage_observation.v2.json")) == before


def test_fresh_measurement_resumes_completed_physical_batch_without_rescoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "VERDICT_BATCH_SIZES", (1, 1))
    monkeypatch.setattr(subject, "PRODUCTION_PAIR_COUNT", 2)
    monkeypatch.setattr(subject, "PRODUCTION_SEG_HW", (2, 2))
    monkeypatch.setattr(subject, "PIXEL_DENOMINATOR", 8)

    class Semantic:
        @staticmethod
        def parse_packet(_packet: bytes) -> object:
            return object()

        @staticmethod
        def render_scorer_y1(
            _parsed: object,
            pair_id: int,
        ) -> np.ndarray:
            return np.full(
                (2, 2, 3),
                pair_id,
                dtype=np.uint8,
            )

    class Inflate:
        @staticmethod
        def _realize_factor2(scorer_y1: np.ndarray) -> np.ndarray:
            return np.full(
                (874, 1164, 3),
                int(scorer_y1[0, 0, 0]),
                dtype=np.uint8,
            )

    class Frame0:
        @staticmethod
        def render_camera_y0(
            _state: object,
            _pair_id: int,
            _scorer_y1: np.ndarray,
            camera_y1: np.ndarray,
        ) -> np.ndarray:
            return camera_y1

        @staticmethod
        def verify_final_y1_population(
            _state: object,
            _digest: bytes,
        ) -> None:
            return None

    monkeypatch.setattr(
        subject,
        "_load_public_plugins",
        lambda **_kwargs: (
            Inflate(),
            Semantic(),
            Frame0(),
            object(),
            b"semantic-packet",
        ),
    )
    predicted_sha = hashlib.sha256(
        memoryview(np.zeros((1, 2, 2), dtype=np.uint8))
    ).hexdigest()
    scorer_shas = [
        hashlib.sha256(
            memoryview(np.full((2, 2, 3), pair_id, dtype=np.uint8))
        ).hexdigest()
        for pair_id in range(2)
    ]
    camera_shas = [
        hashlib.sha256(
            memoryview(
                np.full(
                    (874, 1164, 3),
                    pair_id,
                    dtype=np.uint8,
                )
            )
        ).hexdigest()
        for pair_id in range(2)
    ]
    expected_rows = [
        {
            "batch_index": batch_index,
            "scorer_y1_batch_sha256": scorer_shas[batch_index],
            "camera_y1_batch_sha256": camera_shas[batch_index],
            "predicted_labels_batch_sha256": predicted_sha,
        }
        for batch_index in range(2)
    ]
    selected = SimpleNamespace(
        archive=b"archive",
        scorer_y1_population_sha256=subject._batch_chain(
            expected_rows,
            "scorer_y1_batch_sha256",
        ),
        camera_y1_population_sha256=subject._batch_chain(
            expected_rows,
            "camera_y1_batch_sha256",
        ),
        predicted_labels_sha256=subject._batch_chain(
            expected_rows,
            "predicted_labels_batch_sha256",
        ),
        disagreement_pixels=0,
        d_seg=0.0,
    )
    engine = SimpleNamespace(
        selected=selected,
        alternatives=(
            SimpleNamespace(
                y1_wire_codec=SimpleNamespace(name="RAW_I16_LE"),
                outer_zip_method=SimpleNamespace(name="STORE"),
                archive=b"archive",
            ),
        ),
    )
    authority = SimpleNamespace(
        physical_stage_identity_sha256=_sha("physical-stage"),
        target_labels=np.zeros((2, 2, 2), dtype=np.uint8),
        seg_scorer_identity_sha256=_sha("seg-scorer"),
    )
    selected_row = {
        "archive": {
            "sha256": hashlib.sha256(b"archive").hexdigest(),
        },
        "semantic_packet": {
            "sha256": hashlib.sha256(
                b"semantic-packet"
            ).hexdigest(),
        },
    }

    class CrashAfterFirstBatch:
        def __init__(self) -> None:
            self.calls = 0

        def _predict(self, camera: np.ndarray) -> np.ndarray:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated process crash")
            return np.zeros(
                (camera.shape[0], 2, 2),
                dtype=np.uint8,
            )

    first_scorer = CrashAfterFirstBatch()
    with pytest.raises(RuntimeError, match="simulated process crash"):
        subject._measure_public_surface_fresh(
            authority=authority,
            engine=engine,
            broker=subject._FreshPredictionBroker(first_scorer),
            sealed_root=tmp_path,
            progress_dir=tmp_path,
            measurement_cache_dir=(tmp_path / "cache"),
            public_runtime_tree_sha256=_sha("public-runtime"),
            selected_row=selected_row,
        )
    first_receipt = next(
        (tmp_path / "cache").rglob(
            "batch_000_000_001.receipt.json"
        )
    )
    first_receipt_sha = hashlib.sha256(
        first_receipt.read_bytes()
    ).hexdigest()

    class ResumeScorer:
        def __init__(self) -> None:
            self.calls = 0

        def _predict(self, camera: np.ndarray) -> np.ndarray:
            self.calls += 1
            assert int(camera[0, 0, 0, 0]) == 1
            return np.zeros(
                (camera.shape[0], 2, 2),
                dtype=np.uint8,
            )

    resume_scorer = ResumeScorer()
    result = subject._measure_public_surface_fresh(
        authority=authority,
        engine=engine,
        broker=subject._FreshPredictionBroker(resume_scorer),
        sealed_root=tmp_path,
        progress_dir=tmp_path,
        measurement_cache_dir=(tmp_path / "cache"),
        public_runtime_tree_sha256=_sha("public-runtime"),
        selected_row=selected_row,
    )
    assert resume_scorer.calls == 1
    assert result["fresh_measured_batch_count"] == 1
    assert result["resumed_physical_batch_count"] == 1
    assert (
        hashlib.sha256(first_receipt.read_bytes()).hexdigest()
        == first_receipt_sha
    )
