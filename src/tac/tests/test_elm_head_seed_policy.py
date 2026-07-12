# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import tools.elm_inr_head_seed as elm_tool
from tac.boundary_math.elm_inr_head_solve import (
    StreamingPartitionedRidge,
    StreamingRidgeNormalEquations,
    atomic_save_npz,
)
from tac.witness_control.resume_registry import (
    RESUME_REGISTRY_MANIFEST_KEY,
    ResumeIntegrityError,
)
from tac.witness_dsl.elm_head_seed_policy import (
    ElmHeadSeedPolicy,
    ElmHeadSeedScope,
    compile_elm_head_seed_policy,
)
from tac.witness_dsl.typed_config import ProvenanceClass
from tools.elm_inr_head_seed import (
    ElmHeadSeedResumeController,
    RealHeadSeedContext,
    _array_payload_sha256,
    _atomic_write_json,
    _gauss_newton_handoff,
    _load_pair0_seed_receipt,
    _restore_state,
    _resume_registry,
    _save_state,
    _through_r_source_custody,
    _verify_declared_digest,
    build_parser,
)
from tools.elm_inr_head_seed import (
    main as elm_head_seed_main,
)


def _provenanced(value, *, source: str = "test law") -> dict:
    return {
        "value": value,
        "provenance": ProvenanceClass.HARDCODED_WITH_WAIVER.value,
        "unit": "dimensionless",
        "source": source,
        "waiver": "test fixture only; rederive for every production policy",
    }


def _policy_dict(*, scope: str = "diagnostic", pair_limit: int | None = 1) -> dict:
    policy = {
        "schema": "elm_head_seed_policy.v1",
        "ridge": _provenanced(0.0),
        "pinv_rcond": _provenanced(1e-12),
        "label_smoothing": _provenanced(0.1),
        "target_temperature": _provenanced(0.3),
        "grid_rows": _provenanced(2),
        "grid_cols": _provenanced(2),
        "pixel_chunk": _provenanced(32768),
        "scope": _provenanced(scope),
        "source_checkpoint_sha256": "0" * 64,
        "feature_state_sha256": "1" * 64,
        "labels_sha256": "2" * 64,
    }
    if pair_limit is not None:
        policy["pair_limit"] = _provenanced(pair_limit)
    return policy


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tiny_real_cli_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    """Write a real, tiny level-set checkpoint/cache accepted by RealHeadSeedContext."""

    rng = np.random.default_rng(101)
    params_path = root / "source.npz"
    feature_path = root / "features.npz"
    labels_path = root / "labels.npz"
    policy_path = root / "policy.json"
    height, width = 4, 4
    hidden, mod_dim, classes = 5, 2, 3
    # n_scales=1,n_orient0=1,n_iso=1 -> 2 frequency vectors -> 4 sin/cos
    # features; n_dir_freqs=1 -> another 4 directional features.
    in_features = 8
    arrays: dict[str, np.ndarray] = {
        "code": rng.normal(size=(4, mod_dim)).astype(np.float32),
        "in_proj.weight": rng.normal(scale=0.2, size=(hidden, in_features)).astype(np.float32),
        "in_proj.bias": rng.normal(scale=0.1, size=hidden).astype(np.float32),
        "film.weight": rng.normal(scale=0.03, size=(2 * hidden, mod_dim)).astype(np.float32),
        "film.bias": rng.normal(scale=0.02, size=2 * hidden).astype(np.float32),
        "hidden.0.weight": rng.normal(scale=0.2, size=(hidden, hidden)).astype(np.float32),
        "hidden.0.bias": rng.normal(scale=0.1, size=hidden).astype(np.float32),
        "out_sdf.weight": rng.normal(scale=0.2, size=(classes, hidden)).astype(np.float32),
        "out_sdf.bias": rng.normal(scale=0.1, size=classes).astype(np.float32),
        "__render_hw": np.asarray([height, width], np.int64),
        "__bank_n_scales": np.asarray(1, np.int64),
        "__bank_n_orient0": np.asarray(1, np.int64),
        "__bank_f0": np.asarray(1.0, np.float64),
        "__bank_base": np.asarray(2.0, np.float64),
        "__bank_n_iso": np.asarray(1, np.int64),
        "__cfg_max_bank_freq": np.asarray(-1.0, np.float64),
        "__cfg_n_hidden": np.asarray(1, np.int64),
        "__cfg_hidden_dim": np.asarray(hidden, np.int64),
        "__cfg_activation": np.asarray("relu"),
        "__cfg_wire_w0": np.asarray(20.0, np.float64),
        "__cfg_wire_s0": np.asarray(10.0, np.float64),
        "__cfg_hosc_beta": np.asarray(2.5, np.float64),
        "__cfg_hosc_omega": np.asarray(1.0, np.float64),
        "__cfg_n_dir_freqs": np.asarray(1, np.int64),
        "__cfg_freq_across": np.asarray(2.0, np.float64),
        "__cfg_freq_along": np.asarray(1.0, np.float64),
        "__cfg_in_feat": np.asarray(in_features, np.int64),
        "__cfg_softmax_temp": np.asarray(0.3, np.float64),
    }
    np.savez(params_path, **arrays)
    checker = (np.indices((height, width)).sum(axis=0) % classes).astype(np.int8)
    np.savez(
        feature_path,
        pairs=np.asarray([0, 1], np.int64),
        argmax_prev=np.stack([checker, np.roll(checker, 1, axis=1)]),
    )
    labels = rng.integers(0, classes, size=(2, height, width), dtype=np.uint8)
    np.savez(labels_path, n_pairs=np.asarray(2, np.int64), lstars=labels)
    policy = _policy_dict(pair_limit=2)
    policy["source_checkpoint_sha256"] = _file_sha256(params_path)
    policy["feature_state_sha256"] = _file_sha256(feature_path)
    policy["labels_sha256"] = _file_sha256(labels_path)
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return params_path, feature_path, labels_path, policy_path


def _run_tiny_cli_pipeline(
    *,
    paths: tuple[Path, Path, Path, Path],
    output_dir: Path,
    interrupted: bool,
) -> None:
    params_path, feature_path, labels_path, policy_path = paths
    common = [
        "--policy",
        str(policy_path),
        "--params",
        str(params_path),
        "--feature-state",
        str(feature_path),
        "--labels",
        str(labels_path),
        "--output-dir",
        str(output_dir),
        "--tag",
        "tiny",
        "--max-seconds",
        "30",
    ]
    if interrupted:
        assert elm_head_seed_main(["accumulate", *common, "--max-pairs-per-invocation", "1"]) == 0
        assert elm_head_seed_main(["accumulate", *common, "--max-pairs-per-invocation", "1"]) == 0
        assert elm_head_seed_main(["project", *common, "--max-pairs-per-invocation", "1"]) == 0
        assert elm_head_seed_main(["project", *common, "--max-pairs-per-invocation", "1"]) == 0
    else:
        assert elm_head_seed_main(["accumulate", *common]) == 0
        assert elm_head_seed_main(["project", *common]) == 0
    assert elm_head_seed_main(["finalize", *common]) == 0
    # Re-entering a done state exercises final receipt/checkpoint custody validation.
    assert elm_head_seed_main(["finalize", *common]) == 0


def test_typed_policy_compiles_deterministically_and_carries_every_provenance(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_policy_dict(), indent=2), encoding="utf-8")
    first = compile_elm_head_seed_policy(path)
    second = compile_elm_head_seed_policy(path)

    assert first == second
    assert first.scope is ElmHeadSeedScope.DIAGNOSTIC
    assert first.pair_limit == 1
    assert first.grid_shape == (2, 2)
    assert first.policy_file_sha256 == second.policy_file_sha256
    assert first.policy_manifest_sha256 == second.policy_manifest_sha256
    for name in (
        "ridge",
        "pinv_rcond",
        "label_smoothing",
        "target_temperature",
        "grid_rows",
        "grid_cols",
        "pixel_chunk",
        "scope",
        "pair_limit",
    ):
        assert first.manifest[name]["source"]
        assert first.manifest[name]["provenance"]


def test_policy_refuses_raw_literals_unknown_flags_and_ambiguous_scope() -> None:
    raw = _policy_dict()
    raw["ridge"] = 0.0
    with pytest.raises(ValidationError):
        ElmHeadSeedPolicy.model_validate(raw)

    unknown = _policy_dict()
    unknown["invented_knob"] = _provenanced(4)
    with pytest.raises(ValidationError, match="invented_knob"):
        ElmHeadSeedPolicy.model_validate(unknown)

    with pytest.raises(ValidationError, match="requires a provenanced pair_limit"):
        ElmHeadSeedPolicy.model_validate(_policy_dict(pair_limit=None))
    with pytest.raises(ValidationError, match="must omit pair_limit"):
        ElmHeadSeedPolicy.model_validate(_policy_dict(scope="full_p600", pair_limit=1))


def test_full_policy_has_exact_p600_scope_without_a_pair_limit() -> None:
    policy = ElmHeadSeedPolicy.model_validate(_policy_dict(scope="full_p600", pair_limit=None))
    assert policy.scope.value == "full_p600"
    assert policy.pair_limit is None


def test_cli_has_no_raw_semantic_bypass() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "policy" in destinations
    assert not destinations.intersection(
        {
            "ridge",
            "pinv_rcond",
            "label_smoothing",
            "target_temperature",
            "grid_rows",
            "grid_cols",
            "pixel_chunk",
            "pair_limit",
        }
    )


def test_content_digest_verification_refuses_same_path_with_changed_bytes(tmp_path) -> None:
    source = tmp_path / "labels.npz"
    source.write_bytes(b"canonical labels")
    declared = "71f1820a41bac912e47ed92e35855601184c2e55e5f306265671b929fa6f8d03"
    assert _verify_declared_digest(source, declared, "labels") == declared
    source.write_bytes(b"tampered labels")
    with pytest.raises(RuntimeError, match="custody mismatch"):
        _verify_declared_digest(source, declared, "labels")


def test_event_mode_resume_manifest_and_payload_digest_fail_closed(tmp_path) -> None:
    rng = np.random.default_rng(17)
    hidden = rng.normal(size=(19, 3))
    targets = rng.normal(size=(19, 2))
    coords = rng.uniform(-1.0, 1.0, size=(19, 2))
    local = StreamingPartitionedRidge(3, 2, grid_shape=(1, 1), ridge=0.0)
    direct = StreamingRidgeNormalEquations(3, 2, ridge=0.0)
    local.update(hidden, targets, coords)
    direct.update(hidden, targets)
    config = {"schema": "test", "labels_sha256": "2" * 64}
    state_path = tmp_path / "state.npz"
    _save_state(
        state_path,
        config=config,
        stage="accumulate",
        cursor=1,
        local=local,
        direct_target=direct,
    )
    with np.load(state_path, allow_pickle=False) as archive:
        state = {key: np.array(archive[key], copy=True) for key in archive.files}
    assert RESUME_REGISTRY_MANIFEST_KEY in state
    restored = _restore_state(state_path, config)
    assert restored[0:2] == ("accumulate", 1)

    state["direct_target__gram"][0, 0] += 1.0
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **state)
    with pytest.raises(ResumeIntegrityError, match="payload SHA-256 changed"):
        _restore_state(tampered, config)


def test_resume_registry_refuses_vanished_event_controller_keys() -> None:
    controller = ElmHeadSeedResumeController(
        expected_config_sha256="a" * 64,
        stage="project",
        cursor=3,
        payload_sha256="b" * 64,
    )
    arrays = _resume_registry(controller).state_arrays()
    cfg = {
        key: (np.asarray(value).item() if np.asarray(value).shape == () else value)
        for key, value in arrays.items()
        if not key.startswith("__ehs_")
    }
    fresh = ElmHeadSeedResumeController(expected_config_sha256="a" * 64)
    with pytest.raises(ResumeIntegrityError, match="persisted"):
        _resume_registry(fresh).restore(cfg)


def test_payload_digest_is_key_dtype_shape_and_content_sensitive() -> None:
    arrays = {"a": np.arange(4, dtype=np.float32), "b": np.asarray("x")}
    baseline = _array_payload_sha256(arrays)
    changed = {**arrays, "a": np.arange(4, dtype=np.int32)}
    assert _array_payload_sha256(changed) != baseline


def test_atomic_npz_and_json_writers_fsync_containing_directory(tmp_path, monkeypatch) -> None:
    real_open = os.open
    real_fsync = os.fsync
    real_close = os.close
    directory_fds: set[int] = set()
    fsynced_directory_fds: list[int] = []

    def tracked_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        if Path(path) == tmp_path:
            directory_fds.add(fd)
        return fd

    def tracked_fsync(fd):
        if fd in directory_fds:
            fsynced_directory_fds.append(fd)
        return real_fsync(fd)

    def tracked_close(fd):
        directory_fds.discard(fd)
        return real_close(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "fsync", tracked_fsync)
    monkeypatch.setattr(os, "close", tracked_close)
    atomic_save_npz(tmp_path / "state.npz", {"x": np.arange(4)}, compressed=True)
    _atomic_write_json(tmp_path / "receipt.json", {"ok": True})
    assert len(fsynced_directory_fds) == 2
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8")) == {"ok": True}


def test_through_r_source_custody_hashes_runtime_sources_and_environment_manifests() -> None:
    custody = _through_r_source_custody()
    assert set(custody) == {"upstream_and_runtime_sources", "environment_manifests"}
    assert set(custody["environment_manifests"]) == {"pyproject.toml", "uv.lock"}
    assert {
        "upstream/evaluate.py",
        "upstream/modules.py",
        "upstream/frame_utils.py",
        "experiments/train_witness_realized_through_R_mlx.py",
        "src/tac/boundary_math/seg_core.py",
        "src/tac/boundary_math/lever_b_levelset_generator.py",
        "src/tac/boundary_math/lever_b_generator.py",
    } == set(custody["upstream_and_runtime_sources"])
    for group in custody.values():
        for row in group.values():
            path = Path(row["path"])
            assert path.is_file()
            assert row["sha256"] == _file_sha256(path)
            assert row["bytes"] == path.stat().st_size


def test_full_p600_gn_handoff_rehashes_exact_tag_feature_state_at_emission(
    tmp_path,
    monkeypatch,
) -> None:
    direct = Path("direct.npz")
    fold = Path("fold.npz")
    tag = "contract_test"
    monkeypatch.setattr(elm_tool, "PROBE_DIR", tmp_path)
    canonical = tmp_path / f"feats_state_{tag}.npz"
    canonical.write_bytes(b"declared canonical feature state")
    declared_sha256 = _file_sha256(canonical)
    commands, blocker, custody = _gauss_newton_handoff(
        diagnostic_slice=False,
        feature_state_path=canonical,
        declared_feature_state_sha256=declared_sha256,
        tag=tag,
        direct_checkpoint=direct,
        fold_checkpoint=fold,
    )
    assert blocker is None
    assert custody["exact_path_match"] is True
    assert custody["sha256_match"] is True
    assert custody["actual_feature_state_sha256"] == declared_sha256
    assert len(commands) == 2
    assert all("--k-pairs 600" in command for command in commands)
    assert all("--tag contract_test" in command for command in commands)

    canonical.write_bytes(b"same path, adversarially mutated bytes")
    commands, blocker, custody = _gauss_newton_handoff(
        diagnostic_slice=False,
        feature_state_path=canonical,
        declared_feature_state_sha256=declared_sha256,
        tag=tag,
        direct_checkpoint=direct,
        fold_checkpoint=fold,
    )
    assert commands == []
    assert blocker is not None
    assert blocker["code"] == "FEATURE_STATE_SHA256_MISMATCH"
    assert custody["exact_path_match"] is True
    assert custody["sha256_match"] is False


def test_real_context_preserves_or_safely_compacts_validated_integer_labels(tmp_path) -> None:
    params, features, labels, _policy = _write_tiny_real_cli_fixture(tmp_path)
    context = RealHeadSeedContext(params, features, labels)
    assert context.labels.dtype == np.uint8
    assert context.labels.nbytes == context.labels.size

    with np.load(labels, allow_pickle=False) as archive:
        widened = np.asarray(archive["lstars"], np.int64)
        n_pairs = np.array(archive["n_pairs"], copy=True)
    widened_path = tmp_path / "labels_widened.npz"
    np.savez(widened_path, n_pairs=n_pairs, lstars=widened)
    widened_context = RealHeadSeedContext(params, features, widened_path)
    assert widened_context.labels_source_dtype == "int64"
    assert widened_context.labels.dtype == np.uint8
    assert widened_context.labels.nbytes == widened_context.labels.size

    noninteger_path = tmp_path / "labels_float.npz"
    np.savez(noninteger_path, n_pairs=n_pairs, lstars=widened.astype(np.float32))
    with pytest.raises(ValueError, match="must use an integer dtype"):
        RealHeadSeedContext(params, features, noninteger_path)


def test_cli_interrupted_resume_matches_uninterrupted_checkpoint_bytes_and_custody(
    tmp_path,
    capsys,
) -> None:
    paths = _write_tiny_real_cli_fixture(tmp_path)
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    _run_tiny_cli_pipeline(paths=paths, output_dir=uninterrupted_dir, interrupted=False)
    _run_tiny_cli_pipeline(paths=paths, output_dir=resumed_dir, interrupted=True)
    capsys.readouterr()

    scope = "diagnostic_p2_g2x2"
    for stem in ("elm_head_seed_direct_global", "elm_head_seed_pou_fold"):
        uninterrupted = uninterrupted_dir / f"{stem}_{scope}.npz"
        resumed = resumed_dir / f"{stem}_{scope}.npz"
        assert uninterrupted.read_bytes() == resumed.read_bytes()
        assert _file_sha256(uninterrupted) == _file_sha256(resumed)

    uninterrupted_receipt_path = uninterrupted_dir / f"elm_head_seed_receipt_{scope}.json"
    resumed_receipt_path = resumed_dir / f"elm_head_seed_receipt_{scope}.json"
    uninterrupted_receipt = json.loads(uninterrupted_receipt_path.read_text(encoding="utf-8"))
    resumed_receipt = json.loads(resumed_receipt_path.read_text(encoding="utf-8"))
    for receipt, receipt_path in (
        (uninterrupted_receipt, uninterrupted_receipt_path),
        (resumed_receipt, resumed_receipt_path),
    ):
        assert receipt["labels_dtype"] == "uint8"
        assert receipt["labels_resident_nbytes"] == 32
        assert receipt["gauss_newton_commands"] == []
        assert receipt["execution_provenance"]["git"]["head"]
        assert receipt["execution_provenance"]["numpy_version"] == np.__version__
        for checkpoint in ("direct_global_checkpoint", "pou_fold_checkpoint"):
            custody = receipt[checkpoint]
            assert _file_sha256(Path(custody["path"])) == custody["sha256"]
            assert custody["preservation"]["all_non_head_arrays_exact"] is True
        assert receipt_path.is_file()

    uninterrupted_history = uninterrupted_receipt["execution_provenance"]["invocations"]
    resumed_history = resumed_receipt["execution_provenance"]["invocations"]
    assert len(uninterrupted_history) == 3
    assert len(resumed_history) == 5
    assert [row["stage_before"] for row in resumed_history] == [
        "accumulate",
        "accumulate",
        "project",
        "project",
        "finalize",
    ]
    assert all(row["elapsed_seconds"] > 0.0 for row in resumed_history)
    assert resumed_history[0]["resumed_from_existing_state"] is False
    assert all(row["resumed_from_existing_state"] for row in resumed_history[1:])


def test_pair0_comparison_seed_receipt_custody_refuses_same_path_checkpoint_mutation(
    tmp_path,
    capsys,
) -> None:
    paths = _write_tiny_real_cli_fixture(tmp_path)
    policy_payload = json.loads(paths[3].read_text(encoding="utf-8"))
    policy_payload["pair_limit"]["value"] = 1
    paths[3].write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")
    output_dir = tmp_path / "seed"
    _run_tiny_cli_pipeline(paths=paths, output_dir=output_dir, interrupted=False)
    capsys.readouterr()
    params, features, labels, _policy = paths
    context = RealHeadSeedContext(params, features, labels)
    config = {
        "source_checkpoint_sha256": _file_sha256(params),
        "feature_state_sha256": _file_sha256(features),
        "labels_sha256": _file_sha256(labels),
    }
    receipt = output_dir / "elm_head_seed_receipt_diagnostic_p1_g2x2.json"
    payload, custody, checkpoints = _load_pair0_seed_receipt(
        receipt,
        context=context,
        config=config,
    )
    assert custody["sha256"] == _file_sha256(receipt)
    assert payload["direct_global_checkpoint"]["sha256"] == _file_sha256(
        checkpoints["direct_global"]
    )

    with checkpoints["direct_global"].open("ab") as handle:
        handle.write(b"same-path mutation")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _load_pair0_seed_receipt(receipt, context=context, config=config)
