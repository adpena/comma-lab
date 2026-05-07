from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle
from tac.pr91_hpm1_codec import (
    DEFAULT_PR91_ARCHIVE,
    Pr91Hpm1Error,
    build_hpm1_mask_segment,
    raw_tokens_to_mod5_residual_symbols,
    reconstruct_raw_tokens_from_mod5_residual_symbols,
    run_pr91_hpm1_context_window_probe,
    run_pr91_hpm1_entropy_failure_grammar_probe,
    run_pr91_hpm1_first_symbol_state_probe,
    run_pr91_hpm1_preflight,
    run_pr91_hpm1_probability_variant_matrix,
    run_pr91_hpm1_semantic_decode_trench,
    run_pr91_hpm1_spatial_group_order_probe,
    split_hpm1_mask_segment,
    validate_hpm1_static_contract,
)

REPO = Path(__file__).resolve().parents[3]
PREFLIGHT_SCRIPT = REPO / "experiments" / "preflight_pr91_pr92_replay_contracts.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_preflight_script():
    spec = importlib.util.spec_from_file_location("preflight_pr91_pr92_replay_contracts_test", PREFLIGHT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_x_archive(path: Path, x_body: bytes) -> None:
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, x_body)


def _synthetic_hpm1_segment() -> bytes:
    return build_hpm1_mask_segment(
        (np.arange(32, dtype=np.uint32) % 5).tobytes(),
        b"synthetic-hpac-ppmd",
        N=2,
        H=4,
        W=4,
        P=2,
        delta=1,
        ch=4,
        use_spm=False,
        hpac_d_film=2,
    )


def _synthetic_hpm1_archive(path: Path) -> Path:
    segments = {name: f"{name}-payload".encode("ascii") for name in SEGMENT_ORDER}
    segments["mask"] = _synthetic_hpm1_segment()
    segments["bias"] = b"b" * 223
    segments["region"] = b"r" * 273
    _stored_x_archive(path, pack_pr85_bundle(segments, header_mode="v5"))
    return path


def _synthetic_hpm1_archive_with_hpac_model(path: Path, *, tokens_blob: bytes | None = None) -> Path:
    torch = pytest.importorskip("torch")
    pyppmd = pytest.importorskip("pyppmd")
    from tac.pr86_hpac_codec import PPMD_MAX_ORDER, PPMD_MEM_SIZE, HPACMini

    model = HPACMini(num_pairs=2, P=2, delta=1, ch=4, d_film=2, use_spm=False)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        for buffer in model.buffers():
            if buffer.dtype.is_floating_point:
                buffer.zero_()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    hpac = pyppmd.compress(
        buf.getvalue(),
        max_order=PPMD_MAX_ORDER,
        mem_size=PPMD_MEM_SIZE,
    )
    segment = build_hpm1_mask_segment(
        tokens_blob if tokens_blob is not None else (np.arange(32, dtype=np.uint32) % 5).tobytes(),
        hpac,
        N=2,
        H=4,
        W=4,
        P=2,
        delta=1,
        ch=4,
        use_spm=False,
        hpac_d_film=2,
    )
    segments = {name: f"{name}-payload".encode("ascii") for name in SEGMENT_ORDER}
    segments["mask"] = segment
    segments["bias"] = b"b" * 223
    segments["region"] = b"r" * 273
    _stored_x_archive(path, pack_pr85_bundle(segments, header_mode="v5"))
    return path


def test_hpm1_segment_builder_and_parser_are_byte_closed() -> None:
    segment = _synthetic_hpm1_segment()
    payload = split_hpm1_mask_segment(segment)

    assert payload.config() == {
        "n_frames": 2,
        "height": 4,
        "width": 4,
        "predictor_count": 2,
        "delta": 1,
        "channels": 4,
        "use_spm": 0,
        "hpac_d_film": 2,
        "tokens_len": 128,
        "hpac_len": 19,
        "ppmd_order": 4,
    }
    assert payload.extra["tokens_sha256"] == _sha(payload.tokens)
    assert payload.extra["hpac_sha256"] == _sha(payload.hpac)
    assert validate_hpm1_static_contract(payload)["status"] == "passed"


def test_hpm1_segment_builder_fails_closed_on_misaligned_tokens() -> None:
    with pytest.raises(Pr91Hpm1Error, match="tokens_blob_must_be_nonempty_uint32_aligned"):
        build_hpm1_mask_segment(
            b"abc",
            b"model",
            N=1,
            H=1,
            W=1,
            P=1,
            delta=0,
            ch=1,
            use_spm=False,
            hpac_d_film=1,
        )


def test_residual_symbol_helpers_roundtrip_raw_tokens() -> None:
    raw = np.array(
        [
            [[0, 1, 2], [3, 4, 0]],
            [[1, 1, 3], [3, 0, 4]],
            [[4, 0, 3], [2, 2, 1]],
        ],
        dtype=np.uint8,
    )

    residual, prev = raw_tokens_to_mod5_residual_symbols(raw)
    reconstructed = reconstruct_raw_tokens_from_mod5_residual_symbols(residual, prev)

    assert np.array_equal(reconstructed, raw)
    assert np.array_equal(prev[1], raw[0])


def test_pr91_probability_matrix_is_fail_closed_not_notimplemented(tmp_path: Path) -> None:
    archive = _synthetic_hpm1_archive(tmp_path / "archive.zip")

    report = run_pr91_hpm1_probability_variant_matrix(
        archive,
        variants=("source_float64_perfect_false", "source_float32_perfect_true"),
        write_json=False,
    )

    assert report["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["failed_variants"] == ["source_float64_perfect_false", "source_float32_perfect_true"]
    assert {row["failure_stage"] for row in report["variant_results"]} == {"hpac_probability_range_decode"}


def test_pr91_preflight_is_local_only_and_blocks_dispatch(tmp_path: Path) -> None:
    archive = _synthetic_hpm1_archive(tmp_path / "archive.zip")

    report = run_pr91_hpm1_preflight(archive, max_frames=1, attempt_reencode=True, write_json=False)

    assert report["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["dispatch_unlocked"] is False
    assert report["hpm1_static_contract"]["status"] == "passed"


def test_pr91_first_symbol_probe_is_local_fail_closed(tmp_path: Path) -> None:
    archive = _synthetic_hpm1_archive(tmp_path / "archive.zip")

    report = run_pr91_hpm1_first_symbol_state_probe(
        archive,
        reference_tokens_path=tmp_path / "missing_tokens.bin",
        variants=("source_float64_perfect_false",),
        symbol_count=8,
        symbol_offset=2,
    )

    assert report["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["symbol_window"] == {
        "start_global_symbol": 2,
        "requested_count": 8,
        "end_global_symbol_exclusive": 10,
    }
    assert report["reference_tokens"]["exists"] is False
    assert report["variant_results"][0]["failure_reason"] == (
        "first_symbol_trace_requires_byte_closed_hpm1_replay"
    )


def test_pr91_context_window_probe_is_local_fail_closed(tmp_path: Path) -> None:
    archive = _synthetic_hpm1_archive(tmp_path / "archive.zip")

    report = run_pr91_hpm1_context_window_probe(
        archive,
        reference_tokens_path=tmp_path / "missing_tokens.bin",
        windows=((3, 2), (3, 2), (10, 1)),
        variants=("source_float64_perfect_false",),
        context_modes=("decoded_context", "reference_context"),
        prob_eps_values=(1e-7,),
        require_expected_pr91_identity=False,
    )

    assert report["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["windows"] == [
        {"start_global_symbol": 3, "count": 2, "end_global_symbol_exclusive": 5},
        {"start_global_symbol": 10, "count": 1, "end_global_symbol_exclusive": 11},
    ]
    assert len(report["window_results"]) == 4
    assert {row["context_mode"] for row in report["window_results"]} == {
        "decoded_context",
        "reference_context",
    }


def test_pr91_semantic_decode_trench_loads_model_rows_and_refuses_parity(
    tmp_path: Path,
) -> None:
    archive = _synthetic_hpm1_archive_with_hpac_model(tmp_path / "archive.zip")

    report = run_pr91_hpm1_semantic_decode_trench(
        archive,
        probability_row_count=3,
        attempt_prefix_decode=False,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_semantic_decode_trench_v1"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["hpac_model_load"]["loaded"] is True
    assert report["hpac_model_load"]["decompressed_torch_state"]["bytes"] > 0
    assert len(report["hpac_model_load"]["decompressed_torch_state"]["sha256"]) == 64
    assert report["packed_state_inventory"]["loaded"] is True
    assert report["packed_state_inventory"]["tensor_count"] > 0
    assert report["reconstructed_state_inventory"]["loaded"] is True
    assert report["reconstructed_state_inventory"]["tensor_count"] > 0
    assert report["probability_row_probe"]["passed"] is True
    assert report["probability_row_probe"]["raw_softmax_rows"]["shape"] == [3, 5]
    assert report["full_decode"]["passed"] is False
    assert report["byte_exact_semantic_reencode"]["passed"] is False
    assert "prefix_decode_not_attempted" in report["semantic_decode_blockers"]


def test_pr91_entropy_failure_grammar_probe_records_exact_failure_row() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")

    report = run_pr91_hpm1_entropy_failure_grammar_probe(
        DEFAULT_PR91_ARCHIVE,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_entropy_failure_grammar_probe_v1"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["exact_missing_grammar"]["missing_wire_contract"].startswith(
        "semantic HPAC probability/range grammar"
    )
    trace = report["entropy_failure_trace"]
    assert trace["passed"] is False
    assert trace["failure"]["stage"] == "submitted_tokens_decode"
    assert trace["failure"]["reason"] == "hpac_entropy_decode_contract_mismatch"
    assert trace["failing_probability_row"]["normalized_for_categorical"]["sha256"]
    assert trace["context_before_failing_group"]["current_frame"]["shape"] == [384, 512]
    assert trace["failure"]["frame"] == 0
    assert trace["failure"]["group"] == 10
    assert trace["failure"]["symbol_in_group"] == 191
    assert trace["token_stream"]["uint32_word_count"] == 29199
    word_order = report["token_word_order_probe"]
    assert word_order["schema"] == "pr91_hpm1_token_word_order_probe_v1"
    assert word_order["dispatch_allowed"] is False
    assert word_order["source_little_reproduces_exact_failure_row"] is True
    assert word_order["status"] == "not_explained_by_uint32_endian_or_word_reversal"
    assert word_order["non_source_candidates_matching_source_failure_row"] == []
    assert {row["candidate"] for row in word_order["candidate_results"]} == {
        "source_little_uint32",
        "source_native_uint32",
        "big_endian_uint32_words",
        "reversed_little_uint32_words",
        "reversed_big_endian_uint32_words",
    }


def test_pr91_semantic_decode_trench_cli_records_tool_manifest(tmp_path: Path) -> None:
    archive = _synthetic_hpm1_archive_with_hpac_model(tmp_path / "archive.zip")
    out = tmp_path / "semantic_decode_trench.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_pr91_hpm1_semantic_decode_trench.py"),
            "--archive",
            str(archive),
            "--skip-prefix-decode",
            "--probability-row-count",
            "2",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_semantic_decode_trench_v1"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["hpac_model_load"]["loaded"] is True
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_semantic_decode_trench.py"
    )


def test_pr91_entropy_failure_grammar_probe_cli_records_tool_manifest(tmp_path: Path) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    out = tmp_path / "entropy_failure_grammar_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_pr91_hpm1_entropy_failure_grammar_probe.py"),
            "--archive",
            str(DEFAULT_PR91_ARCHIVE),
            "--skip-word-order-probe",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_entropy_failure_grammar_probe_v1"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_entropy_failure_grammar_probe.py"
    )
    assert payload["token_word_order_probe"]["status"] == "not_attempted_by_request"


def test_pr91_in_group_context_update_probe_cli_narrows_false_lead(tmp_path: Path) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    out = tmp_path / "in_group_context_update_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_pr91_hpm1_in_group_context_update_probe.py"),
            "--archive",
            str(DEFAULT_PR91_ARCHIVE),
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_in_group_context_update_probe_v1"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_in_group_context_update_probe.py"
    )
    assert payload["status"] == "narrowed_serial_in_group_context_false_lead"
    probe = payload["in_group_context_update_probe"]
    assert probe["status"] == "not_explained_by_serial_in_group_prefix_context"
    assert probe["target_failure"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
        "decoded_symbol_count_before_failure": 5951,
    }
    assert probe["replayed_to_target"]["matches_source_decoded_before"] is True
    assert probe["serial_prefix_decode"]["passed"] is False
    assert probe["serial_prefix_decode"]["exception_type"] == "AssertionError"
    assert probe["serial_prefix_context"]["assigned_prior_symbols_in_group"] == 191
    assert probe["row_comparison"]["source_argmax_symbol"] == 2
    assert probe["row_comparison"]["serial_argmax_symbol"] == 2
    assert probe["row_comparison"]["max_abs_probability_delta"] < 1e-6


def test_pr91_spatial_group_order_probe_records_candidate_failure_rows() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")

    report = run_pr91_hpm1_spatial_group_order_probe(
        DEFAULT_PR91_ARCHIVE,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_spatial_group_order_probe_v1"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["status"] == "spatial_group_order_hypothesis_still_open"
    probe = report["spatial_group_order_probe"]
    assert probe["source_order_reproduces_exact_failure_row"] is True
    assert probe["non_source_candidates_passing_source_failure_row"] == [
        "tile_major_row_major",
        "phase_major_row_major",
    ]
    rows = {row["candidate"]: row for row in probe["candidate_results"]}
    assert rows["source_mask_row_major"]["failure_signature"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
        "decoded_symbol_count_before_failure": 5951,
    }
    assert rows["full_col_major"]["failure_signature"] == {
        "frame": 0,
        "group": 5,
        "symbol_in_group": 473,
        "decoded_symbol_count_before_failure": 2201,
    }
    assert rows["tile_major_row_major"]["failure_signature"] == {
        "frame": 0,
        "group": 12,
        "symbol_in_group": 210,
        "decoded_symbol_count_before_failure": 8274,
    }
    assert rows["phase_major_row_major"]["failure_signature"] == {
        "frame": 0,
        "group": 11,
        "symbol_in_group": 14,
        "decoded_symbol_count_before_failure": 6926,
    }
    assert rows["tile_major_row_major"]["passes_source_failure_row"] is True
    assert rows["phase_major_row_major"]["passes_source_failure_row"] is True
    assert rows["full_col_major"]["passes_source_failure_row"] is False


def test_pr91_spatial_group_order_probe_cli_records_tool_manifest(tmp_path: Path) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    out = tmp_path / "spatial_group_order_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_pr91_hpm1_spatial_group_order_probe.py"),
            "--archive",
            str(DEFAULT_PR91_ARCHIVE),
            "--spatial-order-candidates",
            "source_mask_row_major,full_col_major",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_spatial_group_order_probe_v1"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_spatial_group_order_probe.py"
    )
    rows = payload["spatial_group_order_probe"]["candidate_results"]
    assert [row["candidate"] for row in rows] == [
        "source_mask_row_major",
        "full_col_major",
    ]


def test_pr91_probe_contracts_fail_closed_on_bad_inputs(tmp_path: Path) -> None:
    archive = _synthetic_hpm1_archive(tmp_path / "archive.zip")

    with pytest.raises(Pr91Hpm1Error, match="symbol_count_must_be_positive"):
        run_pr91_hpm1_first_symbol_state_probe(archive, symbol_count=0)
    with pytest.raises(Pr91Hpm1Error, match="unsupported_context_mode"):
        run_pr91_hpm1_context_window_probe(
            archive,
            context_modes=("bad_context",),
            require_expected_pr91_identity=False,
        )
    with pytest.raises(Pr91Hpm1Error, match="expected_canonical_pr91_archive"):
        run_pr91_hpm1_context_window_probe(archive)
    with pytest.raises(Pr91Hpm1Error, match="unsupported_spatial_order_candidate"):
        run_pr91_hpm1_spatial_group_order_probe(
            archive,
            candidates=("bad_spatial_order",),
        )


def test_preflight_pr91_pr92_replay_contracts_accepts_synthetic_pr92_and_blocks_pr91(
    tmp_path: Path,
) -> None:
    module = _load_preflight_script()
    pr91_archive = _synthetic_hpm1_archive(tmp_path / "pr91.zip")
    probability_report = tmp_path / "probability.json"
    probability_report.write_text(
        json.dumps(
            run_pr91_hpm1_probability_variant_matrix(
                pr91_archive,
                variants=("source_float64_perfect_false",),
                write_json=False,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exact = {
        "avg_segnet_dist": 0.001,
        "avg_posenet_dist": 0.004,
        "archive_size_bytes": 12345,
        "provenance": {
            "archive_sha256": "a" * 64,
            "inflate_runtime_manifest": {
                "runtime_root": "/tmp/replay_runtime/inflate.sh",
                "runtime_tree_sha256": "b" * 64,
            },
        },
    }
    exact["score_recomputed_from_components"] = (
        100 * exact["avg_segnet_dist"]
        + math.sqrt(10 * exact["avg_posenet_dist"])
        + 25 * exact["archive_size_bytes"] / module.SCORE_DENOMINATOR
    )
    exact_json = tmp_path / "contest_auth_eval.adjudicated.json"
    exact_json.write_text(json.dumps(exact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "candidate_archive": {"sha256": "a" * 64, "bytes": 12345},
        "exact_eval_runtime_contract": {"required_inflate_sh": "inflate.sh"},
        "dispatch_readiness": {"ready_for_exact_eval_dispatch": True},
        "randmulti_decoded_row_parity": {"decoded_rows_match": True},
        "non_noop_byte_change": {"changed": True},
        "next_safe_build_command": "build synthetic",
        "next_safe_exact_eval_command_if_rebuilt": "eval synthetic",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args = module.parse_args(
        [
            "--pr91-archive",
            str(pr91_archive),
            "--pr91-probability-report",
            str(probability_report),
            "--pr92-manifest",
            str(manifest_path),
            "--pr92-exact-json",
            str(exact_json),
            "--output-json",
            str(tmp_path / "out.json"),
            "--ledger-md",
            str(tmp_path / "ledger.md"),
        ]
    )
    report = module.build_report(args)

    assert report["status"] == "passed_pr92_a_plus_plus_pr91_fail_closed"
    assert report["pr91_hpm1"]["dispatch_allowed"] is False
    assert report["pr92_rmb1_stack"]["evidence_grade"] == "A++"


def test_preflight_pr91_pr92_replay_contracts_recovers_pr92_from_logs(tmp_path: Path) -> None:
    module = _load_preflight_script()
    pr91_archive = _synthetic_hpm1_archive(tmp_path / "pr91.zip")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    exact = {
        "schema_version": 1,
        "final_score": 0.25,
        "avg_posenet_dist": 0.0001894,
        "avg_segnet_dist": 0.00057185,
        "archive_size_bytes": 229480,
        "score_recomputed_from_components": 0.2535063602939779,
        "provenance": {
            "archive_sha256": "f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774",
            "archive_size_bytes": 229480,
            "inflate_script": "/teamspace/pact/replay_submission_stbm_rmb1/inflate.sh",
            "inflate_runtime_manifest": {
                "runtime_root": "/teamspace/pact/replay_submission_stbm_rmb1",
                "runtime_tree_sha256": "9a9a71afefe7c154ecc188068bea26f01212369883c4b32c9706b32951e267ba",
            },
        },
    }
    adjudication = {
        "archive_bytes": 229480,
        "archive_sha256": "f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774",
        "avg_posenet_dist": 0.0001894,
        "avg_segnet_dist": 0.00057185,
        "component_gate_triggered": False,
        "contest_equivalent_hardware": True,
        "evidence_grade": "A++ contest T4",
        "gpu_t4_match": True,
        "promotion_eligible": True,
        "score_recomputed": 0.2535063602939779,
    }
    (log_dir / "auth_eval.log").write_text(
        "noise\nRESULT_JSON: " + json.dumps(exact, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (log_dir / "adjudication.log").write_text(
        "noise\nADJUDICATION_JSON: " + json.dumps(adjudication, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    args = module.parse_args(
        [
            "--pr91-archive",
            str(pr91_archive),
            "--rerun-pr91-prefix",
            "--pr92-manifest",
            str(tmp_path / "missing_manifest.json"),
            "--pr92-exact-json",
            str(tmp_path / "missing_exact.json"),
            "--pr92-log-dir",
            str(log_dir),
            "--output-json",
            str(tmp_path / "out.json"),
            "--ledger-md",
            str(tmp_path / "ledger.md"),
        ]
    )
    report = module.build_report(args)

    assert report["status"] == "passed_pr92_a_plus_plus_pr91_fail_closed"
    pr92 = report["pr92_rmb1_stack"]
    assert pr92["source"]["mode"] == "recovered_from_logs"
    assert pr92["evidence_grade"] == "A++"
    assert pr92["exact_eval"]["score"] == 0.2535063602939779
    assert pr92["exact_eval"]["runtime_tree_sha256"].startswith("9a9a71af")


def test_preflight_pr91_pr92_replay_contracts_fails_closed_on_missing_pr92(tmp_path: Path) -> None:
    module = _load_preflight_script()
    pr91_archive = _synthetic_hpm1_archive(tmp_path / "pr91.zip")
    args = module.parse_args(
        [
            "--pr91-archive",
            str(pr91_archive),
            "--rerun-pr91-prefix",
            "--pr92-manifest",
            str(tmp_path / "missing_manifest.json"),
            "--pr92-exact-json",
            str(tmp_path / "missing_exact.json"),
            "--pr92-log-dir",
            str(tmp_path / "missing_logs"),
            "--output-json",
            str(tmp_path / "out.json"),
            "--ledger-md",
            str(tmp_path / "ledger.md"),
        ]
    )

    report = module.build_report(args)

    assert report["status"] == "failed_closed"
    assert report["pr91_hpm1"]["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["pr92_rmb1_stack"]["status"] == "failed_closed_missing_pr92_artifact"
