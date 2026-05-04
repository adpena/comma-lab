from __future__ import annotations

import io
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle
from tac.pr86_hpac_codec import HPACMini, encode_tokens_hpac, sha256_bytes
from tac.pr91_hpm1_codec import (
    DEFAULT_PR85_QMA9_TOKEN_SOURCE,
    DEFAULT_PR85_STBM_ARCHIVE,
    DEFAULT_PR91_ARCHIVE,
    Pr91Hpm1Error,
    analyze_pr91_hpm1_runtime_sources,
    build_hpm1_mask_segment,
    compare_hpm1_to_pr86_hpac_contract,
    extract_pr91_hpm1_payload,
    plan_pr91_hpm1_pr85_stbm_fusion,
    prototype_reencode_hpm1_from_raw_tokens,
    prototype_reencode_hpm1_residual_from_raw_tokens,
    raw_tokens_to_mod5_residual_symbols,
    reconstruct_raw_tokens_from_mod5_residual_symbols,
    run_pr91_hpm1_context_window_probe,
    run_pr91_hpm1_first_symbol_state_probe,
    run_pr91_hpm1_probability_variant_matrix,
    run_pr91_hpm1_reference_prefix_probe,
    run_pr91_hpm1_preflight,
    run_pr91_hpm1_stream_transform_probe,
    split_hpm1_mask_segment,
    validate_hpm1_static_contract,
)


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "replay_pr91_hpm1_mask.py"


def _load_cli_script():
    spec = importlib.util.spec_from_file_location("replay_pr91_hpm1_mask_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ppmd_torch(payload) -> bytes:
    pyppmd = pytest.importorskip("pyppmd")
    buf = io.BytesIO()
    torch.save(payload, buf)
    return pyppmd.compress(buf.getvalue(), max_order=4, mem_size=16 << 20)


def _synthetic_hpm1_segment() -> tuple[bytes, np.ndarray]:
    pytest.importorskip("constriction")
    model = HPACMini(num_pairs=2, P=2, delta=1, ch=4, d_film=2, use_spm=False).eval()
    tokens = np.array(
        [
            [[0, 1, 2, 3], [4, 0, 1, 2], [3, 4, 0, 1], [2, 3, 4, 0]],
            [[1, 1, 2, 2], [3, 3, 4, 4], [0, 0, 1, 1], [2, 2, 3, 3]],
        ],
        dtype=np.uint8,
    )
    token_blob, _report = encode_tokens_hpac(model, tokens, P=2, delta=1)
    hpac_ppmd = _ppmd_torch(model.state_dict())
    segment = build_hpm1_mask_segment(
        token_blob,
        hpac_ppmd,
        N=2,
        H=4,
        W=4,
        P=2,
        delta=1,
        ch=4,
        use_spm=False,
        hpac_d_film=2,
    )
    return segment, tokens


def _synthetic_archive(tmp_path: Path) -> tuple[Path, np.ndarray]:
    segment, tokens = _synthetic_hpm1_segment()
    archive = _synthetic_bundle_archive(tmp_path / "archive.zip", segment)
    return archive, tokens


def _synthetic_bundle_archive(path: Path, mask_segment: bytes, *, header_mode: str = "explicit_30") -> Path:
    segments = {name: (name.encode("ascii") + b"x") for name in SEGMENT_ORDER}
    segments["mask"] = mask_segment
    segments["model"] = b"model"
    segments["pose"] = b"pose"
    segments["post"] = b"post"
    segments["shift"] = b"shift"
    segments["frac"] = b"frac"
    segments["frac2"] = b"frac2"
    segments["frac3"] = b"frac3"
    segments["bias"] = (b"b" * 223) if header_mode == "v5" else b"bias"
    segments["region"] = (b"r" * 273) if header_mode == "v5" else b"region"
    segments["randmulti"] = b"randmulti"
    raw = pack_pr85_bundle(segments, header_mode=header_mode)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("x", raw)
    return path


def test_hpm1_segment_parser_validates_token_and_hpac_blobs() -> None:
    segment, _tokens = _synthetic_hpm1_segment()

    contract, token_blob, hpac_blob = split_hpm1_mask_segment(segment)

    assert contract.codec == "HPM1"
    assert contract.metadata["N"] == 2
    assert contract.metadata["H"] == 4
    assert contract.metadata["W"] == 4
    assert contract.metadata["tokens_sha256"] == sha256_bytes(token_blob)
    assert contract.metadata["hpac_ppmd_sha256"] == sha256_bytes(hpac_blob)
    assert contract.metadata["tokens_uint32_aligned"] is True


def test_residual_symbol_helpers_roundtrip_raw_tokens() -> None:
    raw = np.array(
        [
            [[0, 1, 2], [3, 4, 0]],
            [[1, 1, 3], [3, 0, 4]],
        ],
        dtype=np.uint8,
    )

    residual, prev = raw_tokens_to_mod5_residual_symbols(raw)
    reconstructed = reconstruct_raw_tokens_from_mod5_residual_symbols(residual, prev)

    assert np.array_equal(reconstructed, raw)
    assert np.array_equal(prev[0], np.zeros_like(raw[0]))
    assert np.array_equal(prev[1], raw[0])


def test_cli_raw_token_loader_normalizes_qma9_storage_layout(tmp_path: Path) -> None:
    module = _load_cli_script()
    storage_nwh = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3) % 5
    token_file = tmp_path / "tokens.bin"
    token_file.write_bytes(storage_nwh.tobytes(order="C"))

    render_nhw = module._load_raw_tokens(
        token_file,
        "2,4,3",
        "qma9_storage_wh_to_render_hw",
    )

    assert render_nhw.shape == (2, 3, 4)
    assert np.array_equal(render_nhw, np.transpose(storage_nwh, (0, 2, 1)))


def test_residual_hpm1_prototype_is_local_only_and_roundtrip_grounded(tmp_path: Path) -> None:
    archive, raw_tokens = _synthetic_archive(tmp_path)
    payload = extract_pr91_hpm1_payload(archive)

    report = prototype_reencode_hpm1_residual_from_raw_tokens(raw_tokens, payload)

    assert report["status"] == "passed"
    assert report["score_claim"] is False
    assert report["local_only"] is True
    assert report["frames_encoded"] == 2
    assert report["residual_roundtrip_raw_tokens_sha256"] == sha256_bytes(raw_tokens.tobytes(order="C"))
    assert report["hpm1_encode"]["symbol_context_contract"].startswith("symbols_nhw")


def test_hpm1_segment_builder_fails_closed_on_misaligned_tokens() -> None:
    with pytest.raises(Pr91Hpm1Error) as excinfo:
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

    assert excinfo.value.stage == "hpm1_encoder_contract"
    assert excinfo.value.reason == "tokens_blob_must_be_nonempty_uint32_words"


def test_pr91_hpm1_pr85_stbm_fusion_planner_proves_byte_swap_on_synthetic(tmp_path: Path) -> None:
    hpm1_segment, _tokens = _synthetic_hpm1_segment()
    stbm_segment = b"STBM1BR\0" + (b"s" * (len(hpm1_segment) + 16))
    stbm_archive = _synthetic_bundle_archive(tmp_path / "stbm.zip", stbm_segment, header_mode="v5")
    hpm1_archive = _synthetic_bundle_archive(tmp_path / "hpm1.zip", hpm1_segment, header_mode="v5")

    report = plan_pr91_hpm1_pr85_stbm_fusion(
        pr85_stbm_archive=stbm_archive,
        pr91_archive=hpm1_archive,
        pr85_stbm_adjudicated_json=None,
        include_hpm1_prefix_probe=False,
    )

    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["status"] == "blocked_pending_hpm1_full_decode_byte_parity"
    assert report["segment_comparison"]["all_non_mask_segments_identical"] is True
    assert report["segment_comparison"]["changed_segments"] == ["mask"]
    assert report["byte_faithful_fusion"]["exists"] is True
    assert report["byte_faithful_fusion"]["fusion_member_equals_existing_pr91_member"] is True
    assert report["byte_faithful_fusion"]["mask_bytes_delta_hpm1_vs_stbm"] < 0
    assert report["fallback_semantics"]["fallback_to_stbm_or_qma9_after_hpm1_failure_allowed"] is False
    assert report["hpm1_replay_gate"]["status"] == "not_run"


def test_real_pr91_static_contract_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")

    payload = extract_pr91_hpm1_payload(DEFAULT_PR91_ARCHIVE)
    report = validate_hpm1_static_contract(payload)

    assert report["status"] == "passed"
    assert report["mask"]["bytes"] == 145087
    assert report["tokens"]["bytes"] == 116796
    assert report["tokens"]["sha256"] == "541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b"
    assert report["hpac_ppmd"]["sha256"] == "de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd"


def test_real_pr91_hpm1_pr85_stbm_fusion_is_byte_faithful_but_blocked_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")
    if not DEFAULT_PR85_STBM_ARCHIVE.is_file():
        pytest.skip("PR85+STBM archive is not present")

    report = plan_pr91_hpm1_pr85_stbm_fusion(include_hpm1_prefix_probe=False)

    assert report["score_claim"] is False
    assert report["dispatch_unlocked"] is False
    assert report["status"] == "blocked_pending_hpm1_full_decode_byte_parity"
    assert report["segment_comparison"]["changed_segments"] == ["mask"]
    assert report["segment_comparison"]["all_non_mask_segments_identical"] is True
    assert report["byte_faithful_fusion"]["exists"] is True
    assert report["byte_faithful_fusion"]["fusion_member_equals_existing_pr91_member"] is True
    assert report["byte_faithful_fusion"]["archive_bytes_delta_vs_pr85_stbm"] == -7352
    assert report["byte_faithful_fusion"]["mask_bytes_delta_hpm1_vs_stbm"] == -7352
    projection = report["score_projection_if_hpm1_mask_is_semantically_identical"]
    assert projection["evidence_grade"] == "prediction"
    assert projection["score_claim"] is False
    assert projection["matches_expected_public_pr91_self_report"] is True
    assert projection["projected_score"] == pytest.approx(0.24879480490416128)


def test_real_pr91_prefix_decode_reproduces_entropy_failure_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")

    report = run_pr91_hpm1_preflight(DEFAULT_PR91_ARCHIVE, max_frames=1)

    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["status"] == "failed_closed"
    assert report["failure_stage"] == "submitted_tokens_decode"
    assert report["failure_reason"] == "hpac_entropy_decode_contract_mismatch"
    assert report["blocker_class"] == "real_invalid_entropy_or_probability_model_contract_mismatch"
    assert report["failure_context"]["failed_at"] == {"frame": 0, "group": 10, "symbol_in_group": 191}
    assert (
        report["pr86_hpac_relationship"]["relationship"]
        == "pr91_reuses_pr86_hpac_model_with_distinct_hpm1_token_stream"
    )


def test_real_pr91_reuses_pr86_hpac_model_but_not_tokens_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")

    payload = extract_pr91_hpm1_payload(DEFAULT_PR91_ARCHIVE)
    report = compare_hpm1_to_pr86_hpac_contract(payload)

    if report["status"] == "failed_closed_pr86_archive_unavailable":
        pytest.skip("PR86 archive is not present")
    assert report["status"] == "passed"
    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["hpac_model"]["same_as_pr86_hpac_ppmd"] is True
    assert report["tokens"]["same_as_pr86_tokens"] is False
    assert report["tokens"]["pr91_minus_pr86_bytes"] == 2896
    assert report["tokens"]["common_prefix_bytes"] == 164
    assert report["tokens"]["first_mismatch_uint32_word"] == 41


def test_pr91_runtime_source_contract_is_hpm1_cuda_sensitive_if_available() -> None:
    report = analyze_pr91_hpm1_runtime_sources()

    if report["status"] == "failed_closed_missing_sources":
        pytest.skip("downloaded PR91 runtime sources are not present")
    assert report["status"] == "passed"
    assert report["hpm1_runtime_contract"]["hpm1_branch_present"] is True
    assert report["hpm1_runtime_contract"]["decode_function"] == "pr86_hpac.decompress_tokens_hpac"
    assert report["hpm1_runtime_contract"]["decode_passes_main_runtime_device"] is True
    assert report["hpm1_runtime_contract"]["decode_device_argument"] == "str(device)"
    assert report["hpm1_runtime_contract"]["explicit_hpac_cpu_force_detected"] is False
    assert report["hpm1_runtime_contract"]["hpac_cpu_force_comment_detected"] is True
    assert report["hpm1_runtime_contract"]["hpac_cpu_force_comment_matches_hpm1_call"] is False
    assert report["hpm1_runtime_contract"]["fallback_on_hpm1_entropy_failure_detected"] is False
    assert report["probability_model_contract"]["probability_numpy_dtype"] == "float64"
    assert report["probability_model_contract"]["categorical_perfect_false"] is True


def test_hpm1_probability_variant_matrix_passes_on_synthetic_archive(tmp_path: Path) -> None:
    archive, _tokens = _synthetic_archive(tmp_path)

    report = run_pr91_hpm1_probability_variant_matrix(
        archive,
        variants=("source_float64_perfect_false",),
        max_frames=None,
        attempt_reencode=True,
        require_expected_pr91_identity=False,
    )

    assert report["status"] == "passed"
    assert report["local_decode_byte_parity_proven"] is True
    assert report["pr91_ready_for_exact_eval"] is False
    assert report["dispatch_unlocked"] is False
    assert report["local_decode_variants"] == ["source_float64_perfect_false"]
    assert report["source_contract_byte_parity_variants"] == ["source_float64_perfect_false"]
    result = report["variant_results"][0]
    assert result["byte_parity_achieved"] is True
    assert result["decode"]["frames_decoded"] == 2
    assert result["reencode"]["byte_exact_reencode"] is True


def test_hpm1_context_window_probe_covers_context_and_eps_variants_on_synthetic(
    tmp_path: Path,
) -> None:
    archive, tokens = _synthetic_archive(tmp_path)
    reference = tmp_path / "reference_tokens.bin"
    reference.write_bytes(tokens.tobytes(order="C"))

    report = run_pr91_hpm1_context_window_probe(
        archive,
        reference_tokens_path=reference,
        reference_layout="legacy_assume_nhw",
        windows=((0, 4), (10, 4)),
        variants=("source_float64_perfect_false",),
        context_modes=("decoded_context", "reference_context"),
        prob_eps_values=(1e-7, 1e-9),
        require_expected_pr91_identity=False,
    )

    assert report["status"] == "passed"
    assert report["score_claim"] is False
    assert report["dispatch"] is False
    assert report["dispatch_unlocked"] is False
    assert len(report["context_results"]) == 4
    assert {
        (row["context_mode"], row["prob_eps"], row["status"])
        for row in report["context_results"]
    } == {
        ("decoded_context", 1e-7, "passed"),
        ("reference_context", 1e-7, "passed"),
        ("decoded_context", 1e-9, "passed"),
        ("reference_context", 1e-9, "passed"),
    }
    first = report["context_results"][0]
    assert first["all_windows_complete"] is True
    assert [row["recorded_count"] for row in first["window_results"]] == [4, 4]
    assert first["first_reference_mismatch"] is None
    teacher = report["teacher_forced_reference_probability_windows"]
    assert [row["status"] for row in teacher] == ["passed", "passed"]
    assert teacher[0]["range_decoder_consumed"] is False


def test_real_pr91_probability_matrix_fails_closed_without_local_decode_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")

    report = run_pr91_hpm1_probability_variant_matrix(
        DEFAULT_PR91_ARCHIVE,
        variants=None,
        max_frames=1,
    )

    assert report["status"] == "failed_closed"
    assert report["pr91_ready_for_exact_eval"] is False
    assert report["dispatch_unlocked"] is False
    assert report["failure_reason"] == "no_probability_variant_decodes_pr91_hpm1_prefix"
    assert report["blocker_class"] == "real_invalid_entropy_or_probability_model_contract_mismatch"
    results = {row["variant"]: row for row in report["variant_results"]}
    assert set(results) == {
        "source_float64_perfect_false",
        "source_float32_perfect_false",
        "source_float64_perfect_true",
        "source_float32_perfect_true",
    }
    assert results["source_float64_perfect_false"]["failure_stage"] == "submitted_tokens_decode"
    assert results["source_float64_perfect_false"]["failure_context"]["failed_at"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
    }
    assert results["source_float64_perfect_false"]["failure_context"][
        "decoded_symbol_count_before_failure"
    ] == 5951
    assert results["source_float32_perfect_false"]["failure_context"][
        "decoded_symbol_count_before_failure"
    ] == 30513


def test_real_pr91_reference_prefix_probe_shrinks_pr85_identity_claim_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")
    if not DEFAULT_PR85_QMA9_TOKEN_SOURCE.is_file():
        pytest.skip("PR85 QMA9 reference token source is not present")

    report = run_pr91_hpm1_reference_prefix_probe(
        DEFAULT_PR91_ARCHIVE,
        variants=("source_float64_perfect_false",),
        max_frames=1,
    )

    assert report["status"] == "failed_closed"
    assert report["failure_reason"] == "no_local_probability_variant_proves_pr91_hpm1_pr85_reference_prefix"
    assert report["reference_tokens"]["matches_expected_pr85_qma9_token_source"] is True
    assert (
        report["reference_tokens"]["layout"]
        == "qma9_storage_wh_to_render_hw"
    )
    assert (
        report["reference_tokens"]["render_order_sha256"]
        == "0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45"
    )
    source = report["source_contract_summary"]
    assert source["status"] == "failed_closed"
    assert source["decoded_symbol_count_before_failure"] == 5951
    assert source["prefix_matches_reference_until_failure"] is False
    assert source["first_reference_mismatch"] == {
        "global_symbol": 33,
        "frame": 0,
        "group": 0,
        "symbol_in_group": 33,
        "pixel_yx": {"y": 64, "x": 32},
        "decoded": 4,
        "reference": 2,
    }
    assert source["failure_context"]["failed_at"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
    }
    assert source["failure_context"]["reference_symbol_at_failure"] == 2


def test_real_pr91_stream_transform_probe_rules_out_byte_word_order_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")

    report = run_pr91_hpm1_stream_transform_probe(DEFAULT_PR91_ARCHIVE, max_frames=1)

    assert report["status"] == "failed_closed"
    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["pr91_ready_for_exact_eval"] is False
    assert report["failure_reason"] == "no_token_stream_transform_decodes_pr91_hpm1_prefix"
    assert report["blocker_class"] == "not_byte_or_word_order_contract_mismatch"
    assert report["decode_variants"] == []
    results = {row["variant"]: row for row in report["transform_results"]}
    assert set(results) == {"raw_le_u32", "word_byteswap", "word_reverse", "byte_reverse"}
    assert results["raw_le_u32"]["failure_context"]["failed_at"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
    }
    assert all(row["status"] == "failed_closed" for row in results.values())


def test_real_pr91_first_symbol_state_probe_exposes_source_contract_prefix_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")
    if not DEFAULT_PR85_QMA9_TOKEN_SOURCE.is_file():
        pytest.skip("PR85 QMA9 reference token source is not present")

    report = run_pr91_hpm1_first_symbol_state_probe(
        DEFAULT_PR91_ARCHIVE,
        variants=("source_float64_perfect_false",),
        symbol_count=16,
    )

    assert report["status"] == "passed"
    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["dispatch_unlocked"] is False
    source = report["source_contract_summary"]
    assert source["status"] == "passed"
    assert source["emitted_symbol_count"] == 16
    assert source["decoded_symbols"] == [2] * 16
    assert source["reference_symbols"] == [2] * 16
    assert source["first_reference_mismatch"] is None

    trace = report["variant_results"][0]["trace"]
    row7 = trace[7]
    assert row7["pixel_yx"] == {"y": 0, "x": 224}
    assert row7["context_before_decode"]["current_left"] == 0
    assert row7["matches_reference"] is True
    probs7 = row7["probability_state"]["variant_rows"]["source_float64_perfect_false"]
    assert probs7["argmax"] == 2
    assert probs7["decoded_symbol_rank"] == 0
    assert probs7["reference_symbol_rank"] == 0
    assert probs7["values"][2] == pytest.approx(0.939719, rel=1e-6)


def test_real_pr91_symbol_window_probe_shrinks_entropy_failure_if_available() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")
    if not DEFAULT_PR85_QMA9_TOKEN_SOURCE.is_file():
        pytest.skip("PR85 QMA9 reference token source is not present")

    report = run_pr91_hpm1_first_symbol_state_probe(
        DEFAULT_PR91_ARCHIVE,
        variants=("source_float64_perfect_false",),
        symbol_offset=5948,
        symbol_count=8,
    )

    assert report["status"] == "failed_closed"
    assert report["score_claim"] is False
    assert report["dispatch_unlocked"] is False
    result = report["variant_results"][0]
    assert result["status"] == "failed_closed"
    assert result["emitted_symbol_count"] == 3
    assert result["decoded_symbol_count"] == 5951
    assert result["symbol_window"] == {
        "start_global_symbol": 5948,
        "requested_count": 8,
        "end_global_symbol_exclusive": 5956,
        "recorded_count": 3,
    }
    assert [row["global_symbol"] for row in result["trace"]] == [5948, 5949, 5950]
    failure = result["failure_context"]
    assert failure["decoded_symbol_count_before_failure"] == 5951
    assert failure["window_recorded_symbol_count_before_failure"] == 3
    assert failure["failed_at"] == {"frame": 0, "group": 10, "symbol_in_group": 191}
    assert failure["reference_symbol_at_failure"] == 2
    assert (
        failure["probability_state_at_failure"]["variant_rows"]["source_float64_perfect_false"][
            "reference_symbol_rank"
        ]
        == 0
    )


def test_prototype_reencode_hpm1_from_synthetic_tokens(tmp_path: Path) -> None:
    archive, tokens = _synthetic_archive(tmp_path)
    payload = extract_pr91_hpm1_payload(archive)

    report = prototype_reencode_hpm1_from_raw_tokens(tokens, payload)

    assert report["status"] == "passed"
    assert report["prototype_only"] is True
    assert report["score_claim"] is False
    assert report["frames_encoded"] == 2
    assert report["input_tokens_sha256"] == sha256_bytes(tokens.tobytes(order="C"))
    assert report["candidate_hpm1_segment"]["tokens_len"] > 0


def test_cli_writes_json_report(tmp_path: Path) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")
    script = _load_cli_script()
    out = tmp_path / "report.json"

    assert script.main(["--archive", str(DEFAULT_PR91_ARCHIVE), "--max-frames", "1", "--json-out", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tool"] == "tac.pr91_hpm1_codec.run_pr91_hpm1_preflight"
    assert payload["score_claim"] is False
    assert payload["local_only"] is True


def test_cli_probability_variant_matrix_writes_fail_closed_blocker(tmp_path: Path) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")
    script = _load_cli_script()
    out = tmp_path / "probability_matrix.json"

    assert (
        script.main(
            [
                "--archive",
                str(DEFAULT_PR91_ARCHIVE),
                "--probability-variant-matrix",
                "--probability-variants",
                "source_float64_perfect_false",
                "--max-frames",
                "1",
                "--json-out",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tool"] == "tac.pr91_hpm1_codec.run_pr91_hpm1_probability_variant_matrix"
    assert payload["status"] == "failed_closed"
    assert payload["score_claim"] is False
    assert payload["dispatch_unlocked"] is False
    assert payload["pr91_ready_for_exact_eval"] is False
    assert payload["failure_reason"] == "no_probability_variant_decodes_pr91_hpm1_prefix"
    assert payload["blocker_class"] == "real_invalid_entropy_or_probability_model_contract_mismatch"
    result = payload["variant_results"][0]
    assert result["variant"] == "source_float64_perfect_false"
    assert result["failure_context"]["failed_at"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
    }


def test_cli_context_window_probe_writes_structured_failure(tmp_path: Path) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("PR91 archive is not present")
    if not DEFAULT_PR85_QMA9_TOKEN_SOURCE.is_file():
        pytest.skip("PR85 QMA9 reference token source is not present")
    script = _load_cli_script()
    out = tmp_path / "context_windows.json"

    assert (
        script.main(
            [
                "--archive",
                str(DEFAULT_PR91_ARCHIVE),
                "--context-window-probe",
                "--probability-variants",
                "source_float64_perfect_false",
                "--symbol-windows",
                "33:4,5948:8",
                "--json-out",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tool"] == "tac.pr91_hpm1_codec.run_pr91_hpm1_context_window_probe"
    assert payload["score_claim"] is False
    assert payload["dispatch"] is False
    assert payload["dispatch_unlocked"] is False
    assert payload["pr91_ready_for_exact_eval"] is False
    assert payload["blocker_class"] == "range_probability_numeric_contract_at_first_divergence"
    assert (
        payload["source_contract_summary"]["classification"]
        == "reference_context_fails_earlier_after_first_range_divergence"
    )
    assert payload["source_contract_summary"]["decoded_context"]["first_reference_mismatch"][
        "global_symbol"
    ] == 33
