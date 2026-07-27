# SPDX-License-Identifier: MIT
"""Atomic G119-row to G110 public-release materializer tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.witness_dsl import (
    taskspace_g110_release_materializer_v1 as subject,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _seal(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: _sha(_canonical(body))}


def _binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": _sha(payload),
    }


def _runtime(repo_root: Path) -> Path:
    root = repo_root / subject.PUBLIC_RUNTIME_RELATIVE_ROOT
    for relative in subject.PUBLIC_RUNTIME_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"runtime:{relative}\n".encode("ascii"))
        path.chmod(
            0o755
            if relative in {"inflate.sh", "inflate.py"}
            else 0o644
        )
    return root


def _ledger(
    tmp_path: Path,
) -> tuple[Path, str, str, dict[str, object]]:
    config = tmp_path / "population_config.json"
    config.write_bytes(b"{}\n")
    g121_manifest = tmp_path / "g121_retained_prepose.json"
    g121_manifest.write_bytes(b"g121\n")
    g121_completion = tmp_path / "g121_completion.json"
    g121_completion.write_bytes(b"complete\n")
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"c")
    target = tmp_path / "target.json"
    target.write_bytes(b"target\n")
    g112 = tmp_path / "g112.json"
    g112.write_bytes(b"g112\n")
    candidate_state = tmp_path / "candidate_state.npz"
    candidate_state.write_bytes(b"state\n")
    archive_bytes = len(b"archive-under-test")
    archive_sha = _sha(b"archive-under-test")
    candidate = _seal(
        {
            "schema": subject.CANDIDATE_SCHEMA,
            "config_sha256": _sha(b"config-body"),
            "q_levels": 8,
            "pose_mse": 0.01,
            "complete_archive_bytes": archive_bytes,
            "complete_archive_sha256": archive_sha,
            "selected_xip2_coder": "delta_ar_zlib",
            "global_wire_winner_xip2_coder": "delta_ar_zlib",
            "global_wire_winner_archive_bytes": archive_bytes,
            "global_wire_winner_archive_sha256": archive_sha,
            "source_g112_partition_receipt_sha256": _sha(b"g112\n"),
            "target_capsule_receipt_sha256": _sha(b"target\n"),
            "candidate_state": _binding(candidate_state),
            "g110_selected_xip2_coder_abi_closed": True,
            "exact_public_receiver_in_loop": True,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        "candidate_receipt_sha256",
    )
    candidate_path = tmp_path / "candidate_receipt.json"
    candidate_path.write_bytes(_canonical(candidate))
    run = _seal(
        {
            "schema": subject.POST_G105_REFIT_RUN_SCHEMA,
            "final_checkpoint": _binding(checkpoint),
            "selected_xip2_coder": "delta_ar_zlib",
            "source_g112_partition_receipt_sha256": _sha(b"g112\n"),
            "target_capsule_receipt_sha256": _sha(b"target\n"),
            "g110_selected_xip2_coder_abi_closed": True,
            "exact_public_receiver_in_loop": True,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        "receipt_sha256",
    )
    run_path = tmp_path / "run.json"
    run_path.write_bytes(_canonical(run))
    audit = _seal(
        {
            "schema": subject.AUDIT_SCHEMA,
            "config": _binding(config),
            "candidate_rows": [_binding(candidate_path)],
            "selected_q_levels": 8,
            "selected_pose_mse": 0.01,
            "selected_complete_archive_bytes": archive_bytes,
            "selected_complete_archive_sha256": archive_sha,
            "selected_xip2_coder": "delta_ar_zlib",
            "g110_selected_xip2_coder_abi_closed": True,
            "final_checkpoint": _binding(checkpoint),
            "run_receipt": _binding(run_path),
            "exact_public_receiver_in_loop": True,
            "upstream_evaluate_py_not_run": True,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        "audit_receipt_sha256",
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(_canonical(audit))
    row = _seal(
        {
            "schema": "tac.post_g105_pose_refit_joint_axis_row.v1",
            "g121_row_identity_sha256": _sha(b"g121-row"),
            "stage_tag": "ep0",
            "physical_stage_identity_sha256": _sha(b"physical-stage"),
            "d_seg_numerator": 1,
            "d_seg_denominator": 10,
            "d_seg_wire": 0.1,
            "live_target_score_decimal": "0.172",
            "live_target_numerator": 43,
            "live_target_denominator": 250,
            "pointer_snapshot_identity_sha256": _sha(b"pointer"),
            "postverified_pointer_identity_sha256": _sha(
                b"post-pointer"
            ),
            "d_pose_exact": 0.01,
            "final_archive_bytes": archive_bytes,
            "final_archive_sha256": archive_sha,
            "selected_q_levels": 8,
            "selected_xip2_coder": "delta_ar_zlib",
            "g110_selected_xip2_coder_abi_closed": True,
            "post_g105_refit_checkpoint": _binding(checkpoint),
            "post_g105_refit_run_receipt": _binding(run_path),
            "post_g105_refit_audit_receipt": _binding(audit_path),
            "exact_public_receiver_in_loop": True,
            "upstream_evaluate_py_run": False,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        "joint_row_sha256",
    )
    row_sha = str(row["joint_row_sha256"])
    ledger = _seal(
        {
            "schema": subject.JOINT_LEDGER_SCHEMA,
            "config": _binding(config),
            "g121_retained_prepose": _binding(g121_manifest),
            "g121_completion_receipt": _binding(g121_completion),
            "g121_manifest_sha256": _sha(b"g121\n"),
            "g121_pointer_snapshot_identity_sha256": _sha(b"pointer"),
            "g121_postverified_pointer_identity_sha256": _sha(
                b"post-pointer"
            ),
            "g121_live_target_score_decimal": "0.172",
            "g121_live_target_numerator": 43,
            "g121_live_target_denominator": 250,
            "g121_exhaustive_enumeration_proven": True,
            "retained_stage_count": 1,
            "processed_stage_count": 1,
            "every_retained_stage_processed": True,
            "axes": [
                "d_seg_numerator/d_seg_denominator",
                "d_pose_exact",
                "final_archive_bytes",
            ],
            "rows": [row],
            "nondominated_joint_row_sha256": [row_sha],
            "cross_stage_winner_selected": False,
            "selection_deferred_to_whole_archive_evaluate": True,
            "upstream_evaluate_py_run": False,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        "joint_ledger_sha256",
    )
    path = tmp_path / "g119_post_g105_joint_axes.json"
    payload = _canonical(ledger)
    path.write_bytes(payload)
    return path, _sha(payload), row_sha, row


def test_open_selected_row_requires_physical_and_self_hashed_nondominated_row(
    tmp_path: Path,
) -> None:
    path, file_sha, row_sha, row = _ledger(tmp_path)
    opened = subject.open_selected_g119_release_row_v1(
        joint_ledger_path=path,
        expected_joint_ledger_file_sha256=file_sha,
        joint_row_sha256=row_sha,
    )
    assert opened.row == row
    assert opened.ledger_file_sha256 == file_sha
    assert opened.ledger_body_sha256

    value = json.loads(path.read_text("ascii"))
    value["rows"][0]["final_archive_bytes"] += 1
    tampered = _canonical(value)
    path.write_bytes(tampered)
    with pytest.raises(
        subject.G110ReleaseMaterializerError,
        match="self-hash differs",
    ):
        subject.open_selected_g119_release_row_v1(
            joint_ledger_path=path,
            expected_joint_ledger_file_sha256=_sha(tampered),
            joint_row_sha256=row_sha,
        )


def test_open_selected_row_recomputes_nondominance(
    tmp_path: Path,
) -> None:
    path, _file_sha, row_sha, row = _ledger(tmp_path)
    ledger = json.loads(path.read_text("ascii"))
    better_body = {
        key: value
        for key, value in row.items()
        if key != "joint_row_sha256"
    }
    better_body.update(
        {
            "g121_row_identity_sha256": _sha(b"g121-better"),
            "stage_tag": "ep1",
            "physical_stage_identity_sha256": _sha(b"physical-better"),
        }
    )
    better = _seal(better_body, "joint_row_sha256")
    ledger_body = {
        key: value
        for key, value in ledger.items()
        if key != "joint_ledger_sha256"
    }
    ledger_body.update(
        {
            "retained_stage_count": 2,
            "processed_stage_count": 2,
            "rows": [row, better],
            "nondominated_joint_row_sha256": [row_sha],
        }
    )
    resealed = _seal(ledger_body, "joint_ledger_sha256")
    payload = _canonical(resealed)
    path.write_bytes(payload)
    with pytest.raises(
        subject.G110ReleaseMaterializerError,
        match="independently Pareto recomputed",
    ):
        subject.open_selected_g119_release_row_v1(
            joint_ledger_path=path,
            expected_joint_ledger_file_sha256=_sha(payload),
            joint_row_sha256=row_sha,
        )


def test_resealed_row_cannot_forge_a_better_pose_axis(
    tmp_path: Path,
) -> None:
    path, _file_sha, _row_sha, row = _ledger(tmp_path)
    ledger = json.loads(path.read_text("ascii"))
    forged_body = {
        key: value
        for key, value in row.items()
        if key != "joint_row_sha256"
    }
    forged_body["d_pose_exact"] = 0.0
    forged = _seal(forged_body, "joint_row_sha256")
    ledger_body = {
        key: value
        for key, value in ledger.items()
        if key != "joint_ledger_sha256"
    }
    ledger_body["rows"] = [forged]
    ledger_body["nondominated_joint_row_sha256"] = [
        forged["joint_row_sha256"]
    ]
    resealed = _seal(ledger_body, "joint_ledger_sha256")
    payload = _canonical(resealed)
    path.write_bytes(payload)
    with pytest.raises(
        subject.G110ReleaseMaterializerError,
        match="axes differ from its physical post-G105",
    ):
        subject.open_selected_g119_release_row_v1(
            joint_ledger_path=path,
            expected_joint_ledger_file_sha256=_sha(payload),
            joint_row_sha256=str(forged["joint_row_sha256"]),
        )


def test_resolve_compile_custody_rejects_truncated_resealed_g119_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, file_sha, row_sha, _row = _ledger(tmp_path)
    selected = subject.open_selected_g119_release_row_v1(
        joint_ledger_path=path,
        expected_joint_ledger_file_sha256=file_sha,
        joint_row_sha256=row_sha,
    )
    config_path = Path(str(selected.config_binding["path"]))
    target_path = tmp_path / "target.json"
    target_path.write_bytes(b"target\n")
    g112_path = tmp_path / "g112.json"
    g112_path.write_bytes(b"g112\n")
    config = SimpleNamespace(
        config_path=config_path,
        config_sha256=_sha(b"config-body"),
        g121_retained_prepose=selected.ledger[
            "g121_retained_prepose"
        ],
        target_capsule_receipt=_binding(target_path),
    )
    row = selected.row
    common = {
        "disagreement_pixels": row["d_seg_numerator"],
        "pixel_denominator": row["d_seg_denominator"],
        "d_seg_wire": row["d_seg_wire"],
        "live_target_score_decimal": row[
            "live_target_score_decimal"
        ],
        "live_target_numerator": row["live_target_numerator"],
        "live_target_denominator": row["live_target_denominator"],
        "pointer_snapshot_identity_sha256": row[
            "pointer_snapshot_identity_sha256"
        ],
        "postverified_pointer_identity_sha256": row[
            "postverified_pointer_identity_sha256"
        ],
        "g112_partition_receipt": _binding(g112_path),
    }
    selected_stage = SimpleNamespace(
        **common,
        row_identity_sha256=row["g121_row_identity_sha256"],
        stage_tag=row["stage_tag"],
        physical_stage_identity_sha256=row[
            "physical_stage_identity_sha256"
        ],
    )
    omitted_stage = SimpleNamespace(
        **common,
        row_identity_sha256=_sha(b"omitted-g121-row"),
        stage_tag="ep1",
        physical_stage_identity_sha256=_sha(b"omitted-physical"),
    )
    opened = SimpleNamespace(
        stages=(selected_stage, omitted_stage),
        completion_receipt=selected.ledger[
            "g121_completion_receipt"
        ],
        manifest_sha256=selected.ledger["g121_manifest_sha256"],
        pointer_snapshot_identity_sha256=selected.ledger[
            "g121_pointer_snapshot_identity_sha256"
        ],
        postverified_pointer_identity_sha256=selected.ledger[
            "g121_postverified_pointer_identity_sha256"
        ],
        live_target_score_decimal=selected.ledger[
            "g121_live_target_score_decimal"
        ],
        live_target_numerator=selected.ledger[
            "g121_live_target_numerator"
        ],
        live_target_denominator=selected.ledger[
            "g121_live_target_denominator"
        ],
    )
    monkeypatch.setattr(
        subject,
        "load_population_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        subject,
        "_open_g121_retained_population",
        lambda _config: opened,
    )
    with pytest.raises(
        subject.G110ReleaseMaterializerError,
        match="cover every physical G121 retained stage exactly once",
    ):
        subject._resolve_compile_custody(selected)


def test_runtime_capture_is_exact_allowlisted_and_tree_pinned(
    tmp_path: Path,
) -> None:
    root = _runtime(tmp_path)
    snapshot = subject.capture_public_runtime_v1(
        repo_root=tmp_path,
    )
    assert tuple(
        row["relative_path"] for row in snapshot.files
    ) == subject.PUBLIC_RUNTIME_FILES
    assert snapshot.tree_sha256 == _sha(
        subject._canonical_json(list(snapshot.files))
    )
    subject.capture_public_runtime_v1(
        repo_root=tmp_path,
        expected_tree_sha256=snapshot.tree_sha256,
    )

    (root / "undeclared.py").write_bytes(b"pass\n")
    with pytest.raises(
        subject.G110ReleaseMaterializerError,
        match="file census differs",
    ):
        subject.capture_public_runtime_v1(repo_root=tmp_path)


def test_release_source_discovery_includes_parent_packages_and_g120_lazy_load(
) -> None:
    repo_root = Path(subject.__file__).resolve().parents[3]
    paths = set(subject._discover_release_source_paths(repo_root))
    assert Path("src/tac/witness_control/__init__.py") in paths
    assert Path("src/tac/witness_dsl/__init__.py") in paths
    assert (
        Path(
            "src/tac/witness_dsl/"
            "g120_parsed_stage_production_authority_v2.py"
        )
        in paths
    )


def test_output_root_requires_the_storage_root_to_exist(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "absent-volume" / "pact"
    output = missing_root / "release" / "submission"
    with pytest.raises(
        subject.G110ReleaseMaterializerError,
        match="existing non-symlink storage root",
    ):
        subject._validate_output_root(
            output,
            allowed_output_roots=(missing_root,),
            minimum_free_bytes=0,
        )
    assert not missing_root.exists()


def test_materializer_atomically_publishes_and_idempotently_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime(tmp_path)
    runtime = subject.capture_public_runtime_v1(repo_root=tmp_path)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_bytes(b"ledger")
    archive = b"archive-under-test"
    row = {
        "joint_row_sha256": _sha(b"joint-row"),
        "final_archive_bytes": len(archive),
        "final_archive_sha256": _sha(archive),
        "selected_xip2_coder": "delta_ar_zlib",
    }
    selected = subject.SelectedG119ReleaseRowV1(
        ledger_path=ledger_path,
        ledger_bytes=len(b"ledger"),
        ledger_file_sha256=_sha(b"ledger"),
        ledger_body_sha256=_sha(b"ledger-body"),
        config_binding={
            "path": str((tmp_path / "config.json").resolve()),
            "bytes": 0,
            "sha256": _sha(b""),
        },
        ledger={},
        rows=(row,),
        axis_proofs=(),
        row=row,
    )
    inputs: dict[str, dict[str, object]] = {}
    for name, payload in (
        ("target.json", b"t"),
        ("g112.json", b"g"),
        ("refit.npz", b"c"),
        ("run.json", b"r"),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        inputs[name] = _binding(path)
    custody = subject.G110ReleaseCompileCustodyV1(
        target_capsule_receipt=inputs["target.json"],
        g112_partition_receipt=inputs["g112.json"],
        post_g105_refit_checkpoint=inputs["refit.npz"],
        post_g105_refit_run_receipt=inputs["run.json"],
        g121_manifest=inputs["g112.json"],
        g121_row_identity_sha256=_sha(b"g121"),
        physical_stage_identity_sha256=_sha(b"physical"),
    )
    compiled = SimpleNamespace(
        archive=archive,
        archive_bytes=len(archive),
        archive_sha256=_sha(archive),
        packet_sha256=_sha(b"packet"),
        final_y1_binding_sha256=_sha(b"final-y1"),
        g111_source_checkpoint_id_sha256=_sha(b"g111-checkpoint"),
        g111_source_root_sha256=_sha(b"g111-root"),
        g112_semantic_child_sha256=_sha(b"g112-semantic"),
        g112_pose_initializer_sha256=_sha(b"g112-pose"),
        refit_xi_sha256=_sha(b"refit-xi"),
        selected_y1_wire_codec=SimpleNamespace(name="RAW_I16_LE"),
        selected_xip2_coder="delta_ar_zlib",
        selected_outer_zip_method=SimpleNamespace(name="DEFLATE"),
        g112_partition_receipt_sha256=inputs["g112.json"]["sha256"],
        refit_checkpoint_sha256=inputs["refit.npz"]["sha256"],
        refit_run_receipt_sha256=inputs["run.json"]["sha256"],
    )
    monkeypatch.setattr(
        subject,
        "open_selected_g119_release_row_v1",
        lambda **_kwargs: selected,
    )
    monkeypatch.setattr(
        subject,
        "_resolve_compile_custody",
        lambda _selected: custody,
    )
    monkeypatch.setattr(
        subject,
        "compile_g110_generated_y1_pose_v1",
        lambda **_kwargs: compiled,
    )
    monkeypatch.setattr(
        subject,
        "_git_head",
        lambda _root: "1" * 40,
    )
    monkeypatch.setattr(
        subject,
        "capture_public_runtime_v1",
        lambda **_kwargs: runtime,
    )
    source = subject.ReleaseSourceSnapshotV1(
        repo_root=Path(subject.__file__).resolve().parents[3],
        git_sha="1" * 40,
        files=(),
        tree_sha256=_sha(b"source"),
        all_files_equal_git_head=True,
    )
    monkeypatch.setattr(
        subject,
        "capture_release_source_closure_v1",
        lambda **_kwargs: source,
    )
    output = tmp_path / "release" / "submission"
    kwargs = {
        "joint_ledger_path": ledger_path,
        "expected_joint_ledger_file_sha256": _sha(b"ledger"),
        "joint_row_sha256": row["joint_row_sha256"],
        "expected_runtime_tree_sha256": runtime.tree_sha256,
        "output_root": output,
        "command": ["materialize", "--resume-from", str(output)],
        "allowed_output_roots": (tmp_path,),
        "minimum_free_bytes": 0,
    }
    result = subject.materialize_g110_release_v1(**kwargs)
    assert result.archive_path.read_bytes() == archive
    assert result.archive_sha256 == _sha(archive)
    expected_members = {
        subject.ARCHIVE_BASENAME,
        subject.RECEIPT_BASENAME,
        *subject.PUBLIC_RUNTIME_FILES,
    }
    assert {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    } == expected_members
    receipt = json.loads(result.release_receipt_path.read_text("ascii"))
    assert receipt["archive"]["sha256"] == _sha(archive)
    assert receipt["public_runtime"]["tree_sha256"] == runtime.tree_sha256
    assert receipt["receiver_files_packaged"] is True
    assert receipt["receiver_packaging_closed"] is False
    assert receipt["clean_public_entrypoint_double_decode_run"] is False
    assert receipt["upstream_evaluate_py_run"] is False
    assert receipt["score_claim"] is False

    resumed = subject.materialize_g110_release_v1(**kwargs)
    assert resumed.release_receipt_file_sha256 == (
        result.release_receipt_file_sha256
    )

    bad_compiled = SimpleNamespace(
        **{
            **vars(compiled),
            "archive": b"inconsistent-archive-payload",
        }
    )
    monkeypatch.setattr(
        subject,
        "compile_g110_generated_y1_pose_v1",
        lambda **_kwargs: bad_compiled,
    )
    bad_output = tmp_path / "release" / "bad-submission"
    with pytest.raises(
        subject.G110ReleaseMaterializerError,
        match="recompiled G110 archive/custody differs",
    ):
        subject.materialize_g110_release_v1(
            **{
                **kwargs,
                "output_root": bad_output,
            }
        )
    assert not bad_output.exists()

    monkeypatch.setattr(
        subject,
        "compile_g110_generated_y1_pose_v1",
        lambda **_kwargs: compiled,
    )
    result.archive_path.write_bytes(b"corrupt")
    with pytest.raises(
        subject.G110ReleaseMaterializerError,
        match=r"existing release archive\.zip differs",
    ):
        subject.materialize_g110_release_v1(**kwargs)
