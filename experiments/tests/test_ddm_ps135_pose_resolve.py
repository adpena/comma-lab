from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments" / "ddm_ps135_pose_resolve.py"
SPEC = importlib.util.spec_from_file_location("ddm_ps135_pose_resolve", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ps135 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ps135
SPEC.loader.exec_module(ps135)


@pytest.fixture(scope="module")
def governed_pose_payload_store() -> Path:
    root = (
        ps135.DEFAULT_OUTPUT
        / "test_retained"
        / "runner_runtime_custody_p0"
        / f"run_{os.getpid()}_{time.time_ns()}"
    )
    storage = ps135.require_vertigo_free_space(
        root,
        required_free_bytes=50_000_000,
        stage="ddm_ps135_runner_focused_tests",
    )
    root.mkdir(parents=True, exist_ok=True)
    ps135.atomic_json(
        root / "TEST_RUN_STARTED.json",
        {
            "schema": "ddm_ps135_runner_test_run.v1",
            "complete": False,
            "score_claim": False,
            "storage_preflight": storage,
            "payloads_retained": True,
        },
    )
    yield root
    files = [
        ps135.file_record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "TEST_RUN_COMPLETE.json"
    ]
    ps135.atomic_json(
        root / "TEST_RUN_COMPLETE.json",
        {
            "schema": "ddm_ps135_runner_test_run.v1",
            "complete": True,
            "score_claim": False,
            "file_count": len(files),
            "files": files,
            "payloads_retained": True,
        },
    )


def test_delta_zigzag_round_trip_full_signed_int12() -> None:
    rng = np.random.default_rng(20260810)
    values = rng.integers(-2048, 2048, size=(ps135.N, ps135.D), dtype=np.int16)
    encoded = ps135.delta_zigzag_from_signed_codes(values)
    restored = ps135.signed_codes_from_delta_zigzag(encoded)
    assert np.array_equal(restored, values)
    assert encoded.dtype == np.int32


def test_delta_zigzag_refuses_wrong_shape_and_range() -> None:
    with pytest.raises(ps135.PoseResolveError):
        ps135.delta_zigzag_from_signed_codes(np.zeros((1, ps135.D), dtype=np.int16))
    values = np.zeros((ps135.N, ps135.D), dtype=np.int32)
    values[0, 0] = 2048
    with pytest.raises(ps135.PoseResolveError):
        ps135.delta_zigzag_from_signed_codes(values)


def test_process_scan_uses_lsof_when_ps_exec_is_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command[0] == "/bin/ps":
            raise PermissionError(1, "sandbox denied process table")
        return SimpleNamespace(
            returncode=0,
            stdout="COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n",
            stderr="",
        )

    monkeypatch.setattr(ps135.subprocess, "run", fake_run)
    receipt = ps135.process_scan_receipt()

    assert calls == [
        ["/bin/ps", "-axo", "command"],
        ["/usr/sbin/lsof", "-nP", "-c", "Python"],
    ]
    assert receipt["canonical_returncode"] == 126
    assert "PermissionError" in receipt["canonical_stderr"]
    assert receipt["fallback_returncode"] == 0
    assert receipt["passes"] is True


def test_score_matches_upstream_formula() -> None:
    expected = 100 * 0.00029662 + np.sqrt(10 * 0.00002332)
    expected += 25 * ps135.LC2_ARCHIVE_BYTES / ps135.ORIGINAL_BYTES
    assert ps135.score(0.00029662, 0.00002332, ps135.LC2_ARCHIVE_BYTES) == expected


def test_current_artifact_bindings_fail_closed_on_mutation(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    coefficients = tmp_path / "coefficients.int16.npy"
    outputs = tmp_path / "pose_outputs.float32.npy"
    errors = tmp_path / "pair_errors.float64.npy"
    ps135.persist_exact(archive, b"candidate archive")
    ps135.atomic_numpy(
        coefficients, np.zeros((ps135.N, ps135.D), dtype=np.int16)
    )
    ps135.atomic_numpy(
        outputs, np.zeros((ps135.N, ps135.POSE_DIMS), dtype=np.float32)
    )
    ps135.atomic_numpy(errors, np.zeros(ps135.N, dtype=np.float64))
    bindings = ps135.current_artifact_bindings(
        archive=archive,
        coefficients=coefficients,
        pose_outputs_path=outputs,
        pair_errors=errors,
    )

    loaded_archive, loaded_codes, loaded_outputs, loaded_errors, paths = (
        ps135.load_current_artifacts(bindings)
    )
    assert loaded_archive == b"candidate archive"
    assert loaded_codes.dtype == np.int16
    assert loaded_outputs.dtype == np.float32
    assert loaded_errors.dtype == np.float64
    assert paths["archive"] == archive.resolve()

    for artifact in (archive, coefficients, outputs, errors):
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"mutated")
        with pytest.raises(ps135.PoseResolveError, match="bound artifact changed"):
            ps135.load_current_artifacts(bindings)
        artifact.write_bytes(original)


def test_fleet_lock_is_exclusive_and_releasable(tmp_path: Path) -> None:
    path = tmp_path / "fleet.lock"
    first = ps135.acquire_lock(path, purpose="test fleet scorer")
    try:
        with pytest.raises(ps135.PoseResolveError, match="test fleet scorer"):
            ps135.acquire_lock(path, purpose="test fleet scorer")
    finally:
        first.close()
    second = ps135.acquire_lock(path, purpose="test fleet scorer")
    second.close()


def test_dynamic_vertigo_storage_check_passes_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ps135, "VERTIGO_ROOT", tmp_path)

    class Usage:
        free = 50

    monkeypatch.setattr(ps135.shutil, "disk_usage", lambda _path: Usage())
    receipt = ps135.require_vertigo_free_space(
        tmp_path / "arm" / "pass_01",
        required_free_bytes=40,
        stage="unit-pass",
    )
    assert receipt["passes"] is True
    assert receipt["required_free_bytes"] == 40
    assert receipt["observed_free_bytes"] == 50

    with pytest.raises(ps135.PoseResolveError, match="needs 51 free Vertigo bytes"):
        ps135.require_vertigo_free_space(
            tmp_path / "arm" / "pass_01",
            required_free_bytes=51,
            stage="unit-pass",
        )
    with pytest.raises(ps135.PoseResolveError, match="outside the governed Vertigo"):
        ps135.require_vertigo_free_space(
            tmp_path.parent / "outside",
            required_free_bytes=1,
            stage="unit-pass",
        )


def test_protocol_constants_exceed_public_eight_pass_shape() -> None:
    assert ps135.MIN_PASSES == 8
    assert ps135.DRY_PASSES == 3
    assert ps135.DEFAULT_MAX_PASSES > ps135.MIN_PASSES
    assert ps135.MAX_CODE_STEP == 32.0
    assert ps135.JRD_LINEAGE_STEPS == (1, 2, 4, 8, 16, 32)
    assert ps135.JRD_GRID_SIZE == 145
    assert ps135.GRID_MAX == 53


def test_jrd_reusable_prior_compiles_dormant_and_retains_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = ps135.compile_jrd_reusable_prior_receipt()
    compiled = receipt["compiled_no_confirmation"]
    assert compiled["state"] == "DORMANT_N1_SCREEN"
    assert compiled["active"] is False
    assert compiled["precision_actuation"] == "REFUSED_PENDING_N600_CONFIRMATION"
    assert compiled["activation_receipt"] is None
    assert compiled["live_trainer_argv"] == []
    assert receipt["consumption"]["disposition"] == "ORDERING_DATA_ONLY_NOT_ACTUATED"
    assert receipt["consumption"]["precision_assignment_from_pr110"] is False
    assert receipt["score_claim"] is False
    assert receipt["promotion_eligible"] is False

    first = ps135.retain_jrd_reusable_prior_receipt(tmp_path)
    second = ps135.retain_jrd_reusable_prior_receipt(tmp_path)
    assert first == second
    assert json.loads(Path(first["path"]).read_text(encoding="utf-8")) == receipt

    monkeypatch.setattr(ps135, "JRD_PRIOR_POLICY_SHA256", "0" * 64)
    with pytest.raises(ps135.PoseResolveError, match="SHA-256 differs from its pin"):
        ps135.compile_jrd_reusable_prior_receipt()


def test_stage_c_route_preserves_int12_map_and_binds_sd1m_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not ps135.SR1_RECEIVER_PROOF.is_file() or not ps135.LC2_RUNTIME.is_dir():
        pytest.skip("pinned LC2/SR1 route sources are not mounted")
    output = tmp_path / "output"
    chunk_root = output / "leg_a" / "passes" / "pass_01" / "search_chunks"
    ps135.atomic_npz(
        chunk_root / "chunk_0000_0600.npz",
        jacobian=np.zeros((ps135.N, ps135.POSE_DIMS, ps135.D), dtype=np.float64),
        gn_update=np.zeros((ps135.N, ps135.D), dtype=np.float64),
        active_dimensions=np.zeros((ps135.N, ps135.NEIGHBOUR_DIMS), dtype=np.int8),
    )
    compatibility_store = tmp_path / "compatibility_store"
    monkeypatch.setattr(ps135, "LEGACY_STAGE_C_MAP_STORE", compatibility_store)

    disposition = ps135.emit_sensitivity_and_stage_c_disposition(
        output,
        {"passes_completed": 1},
    )
    assert disposition["status"] == "ROUTED_TO_SEPARATE_LC2_SD1M_DRIVER"
    assert disposition["verdict_scope"] == "ROUTING_ONLY_NO_STAGE_C_SCORE"
    assert disposition["receiver_capability"]["format"] == (
        "counted SD1M v1 per-tensor allocation"
    )
    assert disposition["route_sources"]["implementation_spec"]["sha256"] == (
        ps135.STAGE_C_IMPLEMENTATION_SPEC_SHA256
    )
    assert disposition["route_sources"]["sr1_receiver_proof"]["sha256"] == (
        ps135.SR1_RECEIVER_PROOF_SHA256
    )
    assert Path(disposition["measured_map"]["payload"]["path"]).is_file()
    assert (output / "stage_c" / "STAGE_C_DISPOSITION.json").is_file()
    assert not (output / "stage_c" / "blocker.json").exists()


def test_compact_lc2_carrier_parse_reencode_is_exact(
    governed_pose_payload_store: Path,
) -> None:
    path = ps135.LC2_ROOT / "retained" / "inputs" / "carrier.raw"
    if not path.is_file():
        pytest.skip("pinned LC2 carrier is not mounted")
    payload = path.read_bytes()
    state = ps135.decode_carrier(payload)
    reencoded = ps135.encode_carrier(state, state.codes)
    root = governed_pose_payload_store / "carrier_reencode"
    record = ps135.persist_exact(root / "carrier.reencoded.cpr1", reencoded)
    ps135.atomic_json(
        root / "receipt.json",
        {
            "schema": "ddm_ps135_runner_test_payload.v1",
            "complete": reencoded == payload,
            "score_claim": False,
            "source": ps135.file_record(path),
            "generated": record,
            "payloads_retained": True,
        },
    )
    assert reencoded == payload
    assert state.codes.shape == (ps135.N, ps135.D)


def test_exact_renderer_matches_retained_lc2_slave(
    governed_pose_payload_store: Path,
) -> None:
    carrier_path = ps135.LC2_INPUTS / "carrier.raw"
    if not carrier_path.is_file() or not ps135.LC2_RAW.is_file():
        pytest.skip("pinned LC2 carrier/raw pair is not mounted")
    state = ps135.decode_carrier(carrier_path.read_bytes())
    _, _, inflate = ps135.import_runtime_modules()
    torch = __import__("torch")
    renderer = ps135.ExactCarrierRenderer(state, inflate, torch)
    rendered = renderer.render(state.codes[:1])
    root = governed_pose_payload_store / "carrier_renderer_parity"
    record = ps135.persist_exact(
        root / "rendered_slave_0000.uint8.raw",
        rendered.tobytes(order="C"),
    )
    raw = ps135.raw_memmap()
    assert rendered.shape == (1, ps135.CAMERA_H, ps135.CAMERA_W, 3)
    parity = np.array_equal(rendered[0], np.asarray(raw[0]))
    ps135.atomic_json(
        root / "receipt.json",
        {
            "schema": "ddm_ps135_runner_test_payload.v1",
            "complete": parity,
            "score_claim": False,
            "raw_frame_index": 0,
            "source": ps135.file_record(ps135.LC2_RAW),
            "generated": record,
            "payloads_retained": True,
        },
    )
    assert parity


def test_retained_q4_candidate_parses_as_exact_lc2() -> None:
    if not ps135.LC2_ARCHIVE.is_file():
        pytest.skip("pinned LC2 archive is not mounted")
    source = ps135.load_lc2_source()
    archive = ps135.LC2_ARCHIVE.read_bytes()
    receipt = ps135.parse_candidate_archive(
        archive,
        source.carrier,
        source,
    )
    assert archive == ps135.LC2_ARCHIVE.read_bytes()
    assert len(archive) == ps135.LC2_ARCHIVE_BYTES
    assert receipt["archive_sha256"] == ps135.LC2_ARCHIVE_SHA256
    with zipfile.ZipFile(ps135.LC2_ARCHIVE) as retained:
        member = retained.read("p")
    assert receipt["member_sha256"] == ps135.sha256_bytes(member)


def test_public_pr133_warm_start_is_complete_carrier() -> None:
    if not ps135.PR133_ARCHIVE.is_file():
        pytest.skip("pinned PR133 archive is not mounted")
    carrier = ps135.extract_pr133_carrier()
    state = ps135.decode_carrier(carrier)
    assert ps135.sha256_bytes(carrier) == ps135.PR133_CARRIER_SHA256
    assert state.basis_scales.shape == (ps135.D,)
    assert state.coefficient_scales.shape == (ps135.D,)
    assert state.codes.shape == (ps135.N, ps135.D)


def test_official_global_batch_geometry_is_37_full_plus_final_eight() -> None:
    assert ps135.official_batch_sizes(16) == [16] * 37 + [8]


def test_upstream_report_parser_requires_n600_and_exact_bytes() -> None:
    report = "\n".join(
        (
            "=== Evaluation results over 600 samples ===",
            "  Average PoseNet Distortion: 0.00000688",
            "  Average SegNet Distortion: 0.00029662",
            "  Submission file size: 187,226 bytes",
            "  Original uncompressed size: 37,545,489 bytes",
        )
    )
    parsed = ps135.parse_upstream_report(
        report, expected_archive_bytes=ps135.LC2_ARCHIVE_BYTES
    )
    assert parsed["pair_count"] == ps135.N
    assert parsed["d_pose"] == 0.00000688
    with pytest.raises(ps135.PoseResolveError, match="n600 denominator"):
        ps135.parse_upstream_report(
            report.replace("600 samples", "599 samples"),
            expected_archive_bytes=ps135.LC2_ARCHIVE_BYTES,
        )
    with pytest.raises(ps135.PoseResolveError, match="archive byte count"):
        ps135.parse_upstream_report(report, expected_archive_bytes=1)


def _decode_proof_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], Path, Path, str]:
    monkeypatch.setattr(ps135, "LC2_RAW_BYTES", 3)
    packet = tmp_path / "submission"
    packet.mkdir()
    archive_path = packet / "archive.zip"
    member_path = packet / "p"
    member_path.write_bytes(b"member")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("p", b"member")
    raw_path = packet / "inflated" / "0.raw"
    raw_path.parent.mkdir()
    raw_path.write_bytes(b"raw")
    decode_log = tmp_path / "attempts" / "attempt_0001.decode.log"
    decode_log.parent.mkdir()
    decode_log.write_text("decoded\n", encoding="utf-8")
    token_checkpoint = tmp_path / "token_checkpoint" / "tokens.npz"
    token_checkpoint.parent.mkdir()
    token_checkpoint.write_bytes(b"tokens")
    authority_sources = {
        "decode": {"runtime": ps135.file_record(decode_log)},
        "scorer": {},
    }
    archive_sha = ps135.sha256_file(archive_path)
    proof = {
        "schema": ps135.EXACT_DECODE_SUCCESS_SCHEMA,
        "complete": True,
        "archive_sha256": archive_sha,
        "archive": ps135.file_record(archive_path),
        "member": ps135.file_record(member_path),
        "decoded_raw": ps135.file_record(raw_path),
        "decode_log": ps135.file_record(decode_log),
        "token_checkpoints": [ps135.file_record(token_checkpoint)],
        "authority_sources": authority_sources,
    }
    proof_path = decode_log.parent / "attempt_0001.decode_success.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    return proof, authority_sources, proof_path, raw_path, archive_sha


def test_full_size_decode_resume_requires_and_reuses_bound_success_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof, authority, proof_path, raw_path, archive_sha = _decode_proof_fixture(
        tmp_path, monkeypatch
    )
    reusable = ps135.reusable_decode_success(
        proof_path.parent,
        archive_sha=archive_sha,
        raw_path=raw_path,
        authority_sources=authority,
    )
    assert reusable is not None
    resumed, resumed_path = reusable
    assert resumed_path == proof_path
    assert resumed["decode_log"] == proof["decode_log"]
    Path(proof["decode_log"]["path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ps135.PoseResolveError, match="bound artifact changed"):
        ps135.reusable_decode_success(
            proof_path.parent,
            archive_sha=archive_sha,
            raw_path=raw_path,
            authority_sources=authority,
        )


def test_exact_receipt_revalidates_report_sources_and_decode_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof, authority, proof_path, raw_path, archive_sha = _decode_proof_fixture(
        tmp_path, monkeypatch
    )
    evaluate_source = tmp_path / "evaluate.py"
    evaluate_source.write_text("# pinned\n", encoding="utf-8")
    authority["scorer"] = {
        "upstream/evaluate.py": ps135.file_record(evaluate_source)
    }
    proof["authority_sources"] = authority
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    report_path = tmp_path / "report.txt"
    archive_bytes = int(proof["archive"]["bytes"])
    report_path.write_text(
        "\n".join(
            (
                "=== Evaluation results over 600 samples ===",
                "  Average PoseNet Distortion: 0.00000688",
                "  Average SegNet Distortion: 0.00029662",
                f"  Submission file size: {archive_bytes:,} bytes",
                "  Original uncompressed size: 37,545,489 bytes",
            )
        ),
        encoding="utf-8",
    )
    eval_log = tmp_path / "evaluate.log"
    eval_log.write_text("evaluated\n", encoding="utf-8")
    receipt = {
        "schema": ps135.EXACT_EVAL_SCHEMA,
        "complete": True,
        "pair_count": ps135.N,
        "authority_sources": authority,
        "archive": proof["archive"],
        "member": proof["member"],
        "decoded_raw": proof["decoded_raw"],
        "decode_log": proof["decode_log"],
        "decode_proof": ps135.file_record(proof_path),
        "evaluate_source": ps135.file_record(evaluate_source),
        "evaluate_report": ps135.file_record(report_path),
        "evaluate_log": ps135.file_record(eval_log),
        "token_checkpoints": proof["token_checkpoints"],
        "d_pose_report_precision": 0.00000688,
        "d_seg_report_precision": 0.00029662,
        "archive_bytes": archive_bytes,
    }
    assert (
        ps135.validate_exact_evaluation_receipt(
            receipt,
            archive_sha=archive_sha,
            raw_path=raw_path,
            authority_sources=authority,
        )
        is receipt
    )
    report_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ps135.PoseResolveError, match="bound artifact changed"):
        ps135.validate_exact_evaluation_receipt(
            receipt,
            archive_sha=archive_sha,
            raw_path=raw_path,
            authority_sources=authority,
        )


def test_target_cache_rejects_pre_global_geometry_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ps135.TARGET_CACHE_DIRNAME
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "ddm_ps135_av_target_manifest.v1",
                "complete": True,
                "pair_count": ps135.N,
                "gt_decoder_required_function": "frame_utils.yuv420_to_rgb",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ps135.PoseResolveError, match="pinned complete AV cache"):
        ps135.load_target_cache(tmp_path)
    monkeypatch.setattr(ps135, "scorer_source_pins", lambda: {"scorer": "current"})
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": ps135.TARGET_CACHE_SCHEMA,
                "complete": True,
                "pair_count": ps135.N,
                "batch_size": 16,
                "observed_batch_sizes": [16] * 37 + [8],
                "final_partial_batch_padded": False,
                "scorer_sources": {"scorer": "stale"},
                "gt_decoder_required_function": "frame_utils.yuv420_to_rgb",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ps135.PoseResolveError, match="pinned complete AV cache"):
        ps135.load_target_cache(tmp_path)


def test_jrd_binding_covers_full_template_protocol_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ps135, "exact_authority_source_pins", lambda: {"pin": "x"})
    prior_receipt = ps135.persist_exact(tmp_path / "prior.json", b"{}\n")
    template = ps135.CarrierState(
        carrier=b"carrier",
        basis_scales=np.ones(ps135.D, dtype=np.float32),
        basis_codes=np.zeros(
            ps135.D * 3 * ps135.CARRIER_H * ps135.CARRIER_W, dtype=np.int8
        ),
        coefficient_scales=np.ones(ps135.D, dtype=np.float32),
        codes=np.zeros((ps135.N, ps135.D), dtype=np.int16),
    )
    protocol = ps135.jrd_protocol_binding(
        template,
        batch_size=16,
        jrd_prior_receipt=prior_receipt,
        master_provider=SimpleNamespace(binding={"provider": "test"}),
    )
    assert protocol["jrd_lineage_steps"] == [1, 2, 4, 8, 16, 32]
    assert protocol["jrd_grid_size"] == 145
    assert protocol["authority_sources"] == {"pin": "x"}
    binding = ps135.jrd_chunk_binding_json(
        protocol,
        chunk_start=0,
        chunk_end=1,
        input_codes=template.codes[:1],
        input_outputs=np.zeros((1, ps135.POSE_DIMS), dtype=np.float32),
        pose_targets=np.zeros((1, ps135.POSE_DIMS), dtype=np.float32),
    )
    changed_template = ps135.dataclasses.replace(
        template, basis_scales=np.full(ps135.D, 2.0, dtype=np.float32)
    )
    changed_protocol = ps135.jrd_protocol_binding(
        changed_template,
        batch_size=16,
        jrd_prior_receipt=prior_receipt,
        master_provider=SimpleNamespace(binding={"provider": "test"}),
    )
    assert ps135.canonical_json(protocol) != ps135.canonical_json(changed_protocol)
    changed_codes = template.codes[:1].copy()
    changed_codes[0, 0] = 1
    assert binding != ps135.jrd_chunk_binding_json(
        protocol,
        chunk_start=0,
        chunk_end=1,
        input_codes=changed_codes,
        input_outputs=np.zeros((1, ps135.POSE_DIMS), dtype=np.float32),
        pose_targets=np.zeros((1, ps135.POSE_DIMS), dtype=np.float32),
    )
    Path(prior_receipt["path"]).write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ps135.PoseResolveError, match="bound artifact changed"):
        ps135.jrd_protocol_binding(
            template,
            batch_size=16,
            jrd_prior_receipt=prior_receipt,
            master_provider=SimpleNamespace(binding={"provider": "test"}),
        )


def test_exact_population_refresh_checks_every_eligible_nested_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ps135, "import_runtime_modules", lambda: (None, None, None))
    monkeypatch.setattr(ps135, "ExactCarrierRenderer", lambda *args: object())
    calls: list[tuple[int, bool]] = []

    def fake_pose_outputs(
        renderer,
        posenet,
        raw,
        codes,
        rows,
        batch_size,
        *,
        pad_partial,
    ):
        calls.append((len(rows), pad_partial))
        outputs = np.ones((ps135.N, ps135.POSE_DIMS), dtype=np.float32)
        outputs[0] = 0.0
        outputs[1] = 2.0
        return outputs

    monkeypatch.setattr(ps135, "pose_outputs", fake_pose_outputs)
    current_codes = np.zeros((ps135.N, ps135.D), dtype=np.int16)
    aggregate_codes = current_codes.copy()
    aggregate_codes[:2, 0] = 1
    trim_codes = current_codes.copy()
    trim_codes[0, 0] = 1
    current_outputs = np.ones((ps135.N, ps135.POSE_DIMS), dtype=np.float32)
    current_errors = ps135.pose_pair_errors(
        current_outputs, np.zeros_like(current_outputs)
    )

    def candidate(index: int, kind: str, codes: np.ndarray, byte: bytes):
        root = tmp_path / f"variant_{index}"
        root.mkdir()
        archive_path = root / "archive.zip"
        archive = byte * 100
        archive_path.write_bytes(archive)
        return {
            "variant": index,
            "kind": kind,
            "bundle": {"records": {"archive": ps135.file_record(archive_path)}},
            "d_pose": 0.0 if index == 0 else 0.5,
            "archive_bytes": len(archive),
            "archive_sha256": ps135.sha256_bytes(archive),
            "score": 0.0 if index == 0 else 0.5,
            "eligible": True,
            "codes": codes,
            "outputs": current_outputs.copy(),
            "errors": current_errors.copy(),
            "archive": archive,
        }

    aggregate = candidate(0, "aggregate", aggregate_codes, b"a")
    trim = candidate(1, "rate_trim", trim_codes, b"b")
    selection = {
        "selected": aggregate,
        "candidate_rows": [
            {"variant": 0, "d_pose": aggregate["d_pose"]},
            {"variant": 1, "d_pose": trim["d_pose"]},
        ],
        "_materialized_candidates": [aggregate, trim],
        "moved_rows_proposed": 2,
        "rate_trim_denominator": 2,
    }
    template = ps135.CarrierState(
        carrier=b"",
        basis_scales=np.ones(ps135.D, dtype=np.float32),
        basis_codes=np.zeros(1, dtype=np.int8),
        coefficient_scales=np.ones(ps135.D, dtype=np.float32),
        codes=current_codes,
    )
    refreshed = ps135.exact_population_refresh(
        selection,
        template=template,
        current_codes=current_codes,
        current_outputs=current_outputs,
        current_errors=current_errors,
        current_archive=b"c" * 100,
        d_seg=0.0,
        pose_targets=np.zeros_like(current_outputs),
        posenet=object(),
        master_provider=SimpleNamespace(binding={"provider": "test"}),
        batch_size=16,
    )
    assert calls == [(ps135.N, False)]
    assert refreshed["selected"]["variant"] == 1
    assert refreshed["exact_refresh"]["all_eligible_variants_refreshed"] is True
    assert refreshed["exact_refresh"]["materialized_variant_count"] == 2
    assert (tmp_path / "variant_0" / "exact_population_refresh.json").is_file()
    assert (tmp_path / "variant_1" / "exact_population_refresh.json").is_file()
