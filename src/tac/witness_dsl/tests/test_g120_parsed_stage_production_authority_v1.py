from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.witness_dsl import g120_parsed_stage_production_authority_v1 as subject
from tac.witness_dsl.dynamic_frontier_target import DynamicFrontierTargetSnapshot


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _snapshot(*, score: float = 0.15, pointer_sha: str | None = None) -> DynamicFrontierTargetSnapshot:
    return DynamicFrontierTargetSnapshot(
        pointer_path="/physical/canonical_frontier_pointer.json",
        pointer_bytes=123,
        pointer_sha256=pointer_sha or _sha("pointer"),
        pointer_device=1,
        pointer_inode=2,
        pointer_mtime_ns=3,
        last_refreshed_utc="2026-07-27T12:00:00+00:00",
        source_snapshot_at_utc=None,
        target_score=score,
        selected_axis="[contest-CPU]",
        selected_source="local",
        selected_source_kind="qualifying_exact",
        selected_score_precision="exact",
        selected_custody="archive",
        selected_evidence_grade="authority",
        selected_archive_sha256=_sha("frontier-archive"),
        selected_lane_id="lane",
        selected_hardware_substrate="cpu",
        selection_rule="minimum",
    )


def _physical_stage_identity(label: str) -> dict[str, object]:
    return {
        "g112_partition_receipt": {
            "path": f"/physical/{label}/g112.json",
            "bytes": 10,
            "sha256": _sha(f"{label}-g112"),
        },
        "g112_semantic_child": {
            "path": f"/physical/{label}/semantic.npz",
            "bytes": 11,
            "sha256": _sha(f"{label}-semantic"),
            "semantic_packet_sha256": _sha(f"{label}-packet"),
        },
        "g112_pose_initializer": {
            "path": f"/physical/{label}/pose.npz",
            "bytes": 12,
            "sha256": _sha(f"{label}-pose"),
            "target_projection_sha256": _sha("projection"),
        },
        "g111_deploy_checkpoint": {
            "path": f"/physical/{label}/deploy.npz",
            "bytes": 13,
            "sha256": _sha(f"{label}-deploy"),
        },
        "g111_full_state_resume_checkpoint": {
            "path": f"/physical/{label}/resume.npz",
            "bytes": 14,
            "sha256": _sha(f"{label}-resume"),
        },
        "g111_fresh_lineage_receipt": {
            "path": f"/physical/{label}/lineage.json",
            "bytes": 15,
            "sha256": _sha(f"{label}-lineage"),
        },
        "g111_checkpoint_id_sha256": _sha(f"{label}-checkpoint"),
        "g111_stage": label,
        "g111_epoch": 1,
        "fresh_lineage_complete": True,
    }


def _row(
    label: str,
    *,
    d_seg: float,
    archive_bytes: int,
    target: float = 0.15,
    pointer: str | None = None,
) -> subject.G120CrossStageParetoRowV1:
    physical = _physical_stage_identity(label)
    values = {
        "stage_tag": label,
        "row_identity_sha256": "0" * 64,
        "d_seg_wire": d_seg,
        "exact_archive_bytes": archive_bytes,
        "semantic_action": subject.semantic_stage_action(
            d_seg=d_seg,
            archive_bytes=archive_bytes,
        ),
        "distortion_only_value": 100.0 * d_seg,
        "retained_for_post_g105_pose": 100.0 * d_seg < target,
        "source_float_to_wire_regret": {
            "status": "unmeasured",
            "value": None,
            "reason": "fixture",
        },
        "pose_initializer_identity_sha256": _sha(f"{label}-pose"),
        "physical_stage_identity": physical,
        "physical_stage_identity_sha256": subject._sha256(subject._canonical_json(physical)),
        "measurement_identity_sha256": _sha(f"{label}-measurement"),
        "public_runtime_tree_sha256": _sha("public-tree"),
        "selected_archive": {
            "path": f"/physical/{label}/semantic.archive.zip",
            "bytes": archive_bytes,
            "sha256": _sha(f"{label}-archive"),
        },
        "selected_archive_sha256": _sha(f"{label}-archive"),
        "pointer_snapshot_identity_sha256": pointer or _sha("pointer-snapshot"),
        "live_target_score": target,
    }
    provisional = subject.G120CrossStageParetoRowV1(**values)
    body = provisional.to_dict()
    body.pop("row_identity_sha256")
    values["row_identity_sha256"] = subject._sha256(subject._canonical_json(body))
    return subject.G120CrossStageParetoRowV1(**values)


@pytest.fixture
def allow_test_output_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "_EPHEMERAL_ROOTS", ())


def test_production_signature_has_no_self_attested_inputs() -> None:
    parameters = inspect.signature(subject.run_g120_parsed_stage_production_authority_v1).parameters
    assert set(parameters) == {
        "repo_root",
        "g112_partition_receipt",
        "expected_g112_partition_receipt_sha256",
        "out_dir",
        "progress_dir",
        "measurement_cache_dir",
        "cross_stage_dir",
    }
    forbidden = {
        "target_labels",
        "seg_argmax_batch_scorer",
        "effective_frontier_target",
        "pointer_snapshot_identity_sha256",
        "seg_scorer_identity_sha256",
        "source_checkpoint_identity_sha256",
        "pose_initializer_identity_sha256",
        "stage_tag",
    }
    assert not forbidden.intersection(parameters)
    with pytest.raises(TypeError):
        subject.run_g120_parsed_stage_production_authority_v1(  # type: ignore[call-arg]
            arbitrary_callback=lambda _value: _value
        )


def test_dynamic_pointer_changes_observation_not_measurement_identity() -> None:
    first = _snapshot(score=0.15)
    second = dataclasses.replace(first, target_score=0.14)
    assert subject.dynamic_snapshot_identity_sha256(first) != (subject.dynamic_snapshot_identity_sha256(second))
    measurement_identity = _sha("pointer-independent-measurement")
    assert measurement_identity == _sha("pointer-independent-measurement")


def test_exact_shipped_public_plugin_tree_is_physically_sealed() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    identity = subject.public_plugin_tree_identity(repo_root)
    assert identity["root"].endswith("submissions/robust_current/g110_two_layer_receiver")
    assert [row["relative_path"] for row in identity["files"]] == list(subject.PUBLIC_RUNTIME_EXPECTED_FILES)
    assert len(identity["tree_sha256"]) == 64


def test_upstream_closure_rehash_rejects_changed_source(tmp_path: Path) -> None:
    root = tmp_path / "upstream"
    root.mkdir()
    rows = []
    digest = hashlib.sha256()
    for relative in (
        "evaluate.py",
        "frame_utils.py",
        "modules.py",
        "public_test_video_names.txt",
    ):
        path = root / relative
        path.write_text(relative, encoding="ascii")
        row = {
            "relative_path": relative,
            **subject._regular_file_identity(path.resolve(), name=relative),
        }
        rows.append(row)
        digest.update(subject.canonical_json_bytes(row))
        digest.update(b"\n")
    closure = {
        "root": str(root.resolve()),
        "members": rows,
        "closure_sha256": digest.hexdigest(),
    }
    assert subject._reopen_upstream_closure(closure) == closure
    (root / "modules.py").write_text("changed", encoding="ascii")
    with pytest.raises(
        subject.G120ProductionAuthorityError,
        match="closure changed",
    ):
        subject._reopen_upstream_closure(closure)


def test_semantic_initializer_and_lineage_projection_mismatch_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_sha = _sha("projection")
    fake_g112 = SimpleNamespace(
        semantic_child=SimpleNamespace(
            g105_scalars={
                subject.CHECKPOINT_PROJECTION_KEY: "{}",
                subject.CHECKPOINT_PROJECTION_SHA_KEY: projection_sha,
            }
        ),
        initializer=SimpleNamespace(target_projection_sha256=_sha("different-projection")),
        source_chain=SimpleNamespace(
            current=SimpleNamespace(pair=SimpleNamespace(target_projection_sha256=projection_sha))
        ),
    )
    monkeypatch.setattr(
        subject,
        "open_g112_partition_receipt",
        lambda *_args, **_kwargs: fake_g112,
    )
    monkeypatch.setattr(
        subject,
        "reopen_v9_training_target_projection",
        lambda **_kwargs: {"aggregate_receipt": {}},
    )
    with pytest.raises(
        subject.G120ProductionAuthorityError,
        match="projections differ",
    ):
        subject._open_production_authority(
            repo_root=tmp_path.resolve(),
            g112_partition_receipt=(tmp_path / "g112.json").resolve(),
            expected_g112_partition_receipt_sha256=_sha("g112"),
            measurement_cache_dir=tmp_path,
        )


def test_pointer_independent_public_batch_receipt_resumes_exactly(
    tmp_path: Path,
) -> None:
    identity = _sha("measurement")
    body = {
        "schema": subject.PUBLIC_BATCH_SCHEMA,
        "measurement_identity_sha256": identity,
        "batch_index": 0,
        "pair_start": 0,
        "pair_stop": 16,
        "target_batch_sha256": _sha("target"),
        "scorer_y1_batch_sha256": _sha("scorer"),
        "camera_y1_batch_sha256": _sha("camera"),
        "predicted_labels_batch_sha256": _sha("predicted"),
        "disagreement_pixels": 17,
    }
    row = {
        **body,
        "row_body_sha256": subject._sha256(subject._canonical_json(body)),
    }
    path = tmp_path / "batch.json"
    path.write_bytes(subject._canonical_json(row))
    reopened = subject._load_public_batch(
        path,
        identity_sha256=identity,
        batch_index=0,
        pair_start=0,
        pair_stop=16,
        target_batch_sha256=_sha("target"),
        scorer_y1_batch_sha256=_sha("scorer"),
        camera_y1_batch_sha256=_sha("camera"),
    )
    assert reopened == row
    with pytest.raises(
        subject.G120ProductionAuthorityError,
        match="identity differs",
    ):
        subject._load_public_batch(
            path,
            identity_sha256=identity,
            batch_index=0,
            pair_start=0,
            pair_stop=16,
            target_batch_sha256=_sha("target"),
            scorer_y1_batch_sha256=_sha("changed-scorer"),
            camera_y1_batch_sha256=_sha("camera"),
        )


def test_pointer_independent_prediction_cache_rehashes_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = subject._PointerIndependentPredictionCache(
        model=object(),
        scorer_identity_sha256=_sha("scorer"),
        cache_dir=tmp_path,
    )
    monkeypatch.setattr(
        subject._PointerIndependentPredictionCache,
        "_predict",
        lambda _self, batch: np.zeros(
            (batch.shape[0], *subject.PRODUCTION_SEG_HW),
            dtype=np.uint8,
        ),
    )
    camera = np.zeros((1, 874, 1164, 3), dtype=np.uint8)
    assert np.count_nonzero(cache(camera)) == 0
    assert np.count_nonzero(cache(camera)) == 0
    data_path = next(tmp_path.glob("*.predicted_labels.npy"))
    with data_path.open("wb") as stream:
        np.save(
            stream,
            np.ones((1, *subject.PRODUCTION_SEG_HW), dtype=np.uint8),
            allow_pickle=False,
        )
    with pytest.raises(
        subject.G120ProductionAuthorityError,
        match="receipt identity differs",
    ):
        cache(camera)


def test_retains_every_distortion_open_stage_and_tie_breaks_deterministically() -> None:
    dominated_but_pose_unmeasured = _row(
        "second",
        d_seg=0.0010,
        archive_bytes=2000,
    )
    semantic_best = _row(
        "first",
        d_seg=0.0009,
        archive_bytes=1000,
    )
    obstruction = _row(
        "obstruction",
        d_seg=0.0015,
        archive_bytes=1,
    )
    retained = subject.retain_cross_stage_rows([dominated_but_pose_unmeasured, obstruction, semantic_best])
    assert [row.stage_tag for row in retained] == ["first", "second"]
    assert all(row.retained_for_post_g105_pose for row in retained)

    same_action_a = _row("a", d_seg=0.001, archive_bytes=100)
    same_action_b = _row("b", d_seg=0.001, archive_bytes=100)
    assert [row.stage_tag for row in subject.retain_cross_stage_rows([same_action_b, same_action_a])] == ["a", "b"]


def test_cross_stage_manifest_keeps_all_open_rows_and_rejects_legacy_best(
    tmp_path: Path,
    allow_test_output_roots: None,
) -> None:
    first = _row("first", d_seg=0.0009, archive_bytes=1000)
    second = _row("second", d_seg=0.0010, archive_bytes=2000)
    pareto_path, best_path = subject._publish_cross_stage_files(
        cross_stage_dir=tmp_path,
        row=second,
    )
    subject._publish_cross_stage_files(
        cross_stage_dir=tmp_path,
        row=first,
    )
    ledger = json.loads(pareto_path.read_bytes())
    assert [item["stage_tag"] for item in ledger["rows"]] == [
        "first",
        "second",
    ]
    best = json.loads(best_path.read_bytes())
    assert best["row_identity_sha256"] == first.row_identity_sha256
    assert best["pareto_filename"] == subject.PARETO_FILENAME

    best_path.write_bytes(
        subject._canonical_json(
            {
                "schema": subject.BEST_POINTER_SCHEMA,
                "legacy": "levelset_best.json",
            }
        )
    )
    with pytest.raises(
        subject.G120ProductionAuthorityError,
        match="legacy",
    ):
        subject._publish_cross_stage_files(
            cross_stage_dir=tmp_path,
            row=first,
        )


def _strict_run_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    alternative_count: int = 4,
    pointer_verification_error: Exception | None = None,
) -> tuple[SimpleNamespace, subject.G120ProductionStageResultV1 | None]:
    snapshot = _snapshot()
    pointer_identity = subject.dynamic_snapshot_identity_sha256(snapshot)
    target_labels = np.zeros((2, 2, 2), dtype=np.uint8)
    stage = _physical_stage_identity("tau")
    stage_identity_sha = subject._sha256(subject._canonical_json(stage))
    pose_sha = str(stage["g112_pose_initializer"]["sha256"])  # type: ignore[index]
    g112 = SimpleNamespace(
        semantic_child=SimpleNamespace(
            shared_params={},
            code_y1=np.zeros((1, 1), dtype=np.float32),
        ),
        initializer=SimpleNamespace(checkpoint_sha256=pose_sha),
    )
    authority = subject._OpenedProductionAuthorityV1(
        g112=g112,
        target_projection={},
        target_labels=target_labels,
        config=object(),
        scorer=lambda _batch: np.zeros((1, 1, 1), dtype=np.uint8),
        seg_scorer_identity_sha256=_sha("scorer"),
        public_runtime_tree={
            "root": "/physical/public",
            "files": [],
            "tree_sha256": _sha("public-tree"),
        },
        g109_custody={"aggregate_receipt": {"sha256": _sha("g109")}},
        stage_tag="tau",
        physical_stage_identity=stage,
        physical_stage_identity_sha256=stage_identity_sha,
    )
    selected = SimpleNamespace(
        archive=b"exact-archive",
        d_seg=0.001,
        scorer_y1_population_sha256=_sha("repo-scorer-population"),
        camera_y1_population_sha256=_sha("repo-camera-population"),
        predicted_labels_sha256=_sha("repo-predicted-population"),
        y1_wire_codec=subject.Y1WireCodecV1.RAW_I16_LE,
        outer_zip_method=subject.G110OuterZipMethodV1.STORE,
    )
    alternatives = [selected]
    alternatives.extend(
        SimpleNamespace(
            **{
                **vars(selected),
                "y1_wire_codec": codec,
                "outer_zip_method": method,
            }
        )
        for codec, method in (
            (
                subject.Y1WireCodecV1.RAW_I16_LE,
                subject.G110OuterZipMethodV1.DEFLATE,
            ),
            (
                subject.Y1WireCodecV1.DELTA_RICE_BEST_K,
                subject.G110OuterZipMethodV1.STORE,
            ),
            (
                subject.Y1WireCodecV1.DELTA_RICE_BEST_K,
                subject.G110OuterZipMethodV1.DEFLATE,
            ),
        )
    )
    archive_path = tmp_path / "selected.archive.zip"
    archive_path.write_bytes(selected.archive)
    receipt_path = tmp_path / "engine.receipt.json"
    engine_receipt = {
        "engine_only": True,
        "production_authority_closed": False,
        "production_wrapper_required": True,
        "stage_tag": "tau",
        "source_checkpoint_identity_sha256": stage_identity_sha,
        "pose_initializer_identity_sha256": pose_sha,
        "target_labels_sha256": subject._sha256(memoryview(target_labels)),
        "seg_scorer_identity_sha256": _sha("scorer"),
        "pointer_snapshot_identity_sha256": pointer_identity,
    }
    receipt_path.write_bytes(subject._canonical_json(engine_receipt))
    engine = SimpleNamespace(
        selected=selected,
        alternatives=tuple(alternatives[:alternative_count]),
        archive_path=archive_path,
        receipt_path=receipt_path,
        receipt=engine_receipt,
    )
    measurement = subject.G120PublicPluginMeasurementV1(
        d_seg=selected.d_seg,
        disagreement_pixels=1,
        measurement_identity_sha256=_sha("measurement"),
        public_runtime_tree_sha256=_sha("public-tree"),
        repository_scorer_y1_population_sha256=selected.scorer_y1_population_sha256,
        public_scorer_y1_population_sha256=selected.scorer_y1_population_sha256,
        repository_camera_y1_population_sha256=selected.camera_y1_population_sha256,
        public_camera_y1_population_sha256=selected.camera_y1_population_sha256,
        repository_predicted_labels_sha256=selected.predicted_labels_sha256,
        public_predicted_labels_sha256=selected.predicted_labels_sha256,
        batch_receipt_chain_sha256=_sha("batch-chain"),
        resumed_batch_count=38,
    )
    monkeypatch.setattr(subject, "_EPHEMERAL_ROOTS", ())
    monkeypatch.setattr(
        subject,
        "_ssd_cache_dir",
        lambda path: path.resolve(),
    )
    monkeypatch.setattr(
        subject,
        "_open_production_authority",
        lambda **_kwargs: authority,
    )
    monkeypatch.setattr(
        subject,
        "load_dynamic_frontier_target",
        lambda **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        subject,
        "compile_select_parsed_g105_stage_v1",
        lambda **_kwargs: engine,
    )
    monkeypatch.setattr(
        subject,
        "_measure_public_plugin_surface",
        lambda **_kwargs: measurement,
    )

    def verify(_snapshot_value: object) -> object:
        if pointer_verification_error is not None:
            raise pointer_verification_error
        return _snapshot_value

    monkeypatch.setattr(
        subject,
        "verify_dynamic_frontier_target_snapshot",
        verify,
    )
    try:
        result = subject.run_g120_parsed_stage_production_authority_v1(
            repo_root=tmp_path.resolve(),
            g112_partition_receipt=(tmp_path / "g112.json").resolve(),
            expected_g112_partition_receipt_sha256=_sha("g112"),
            out_dir=(tmp_path / "out").resolve(),
            progress_dir=(tmp_path / "progress").resolve(),
            measurement_cache_dir=(tmp_path / "cache").resolve(),
            cross_stage_dir=(tmp_path / "cross-stage").resolve(),
        )
    except subject.G120ProductionAuthorityError:
        result = None
        raise
    return engine, result


def test_strict_production_fixture_emits_separate_nonpromotable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, result = _strict_run_fixture(tmp_path, monkeypatch)
    assert result is not None
    assert result.engine is engine
    assert result.receipt["production_authority_closed"] is True
    assert result.receipt["authoritative_through_R_for_semantic_screen"] is True
    assert result.receipt["exact_shipped_public_plugin_tree_scored"] is True
    assert result.receipt["measurement_cache"]["pointer_independent"] is True
    assert result.receipt["measurement_cache"]["resumed_batch_count"] == 38
    assert result.receipt["physical_stage_identity"]["g111_full_state_resume_checkpoint"]["sha256"] == _sha(
        "tau-resume"
    )
    assert result.receipt["physical_stage_identity"]["g111_fresh_lineage_receipt"]["sha256"] == _sha("tau-lineage")
    assert result.receipt["candidate_claim"] is False
    assert result.receipt["contest_score_claim"] is False
    assert result.receipt["promotion_eligible"] is False
    assert result.receipt["pointer_moved"] is False
    assert result.cross_stage_row.retained_for_post_g105_pose is True
    assert result.receipt_path != engine.receipt_path


def test_engine_handoff_requires_exact_four_way_arbitration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        subject.G120ProductionAuthorityError,
        match="engine handoff",
    ):
        _strict_run_fixture(
            tmp_path,
            monkeypatch,
            alternative_count=3,
        )


def test_changed_pointer_after_screen_publishes_no_production_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        subject.G120ProductionAuthorityError,
        match="pointer changed",
    ):
        _strict_run_fixture(
            tmp_path,
            monkeypatch,
            pointer_verification_error=RuntimeError("changed"),
        )
    out = tmp_path / "out"
    assert not list(out.glob("*g120_production_authority.receipt.json"))
    assert not (tmp_path / "cross-stage" / subject.PARETO_FILENAME).exists()
