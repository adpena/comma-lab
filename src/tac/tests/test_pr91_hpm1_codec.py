# SPDX-License-Identifier: MIT
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
    DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
    DEFAULT_PR91_ARCHIVE,
    Pr91Hpm1Error,
    _build_range_prefix_checkpoint_report,
    _classify_range_prefix_reconstruction,
    _prefix_checkpoint_symbol_counts,
    _summarize_reference_to_submitted_symbol_bridge,
    build_hpm1_mask_segment,
    raw_tokens_to_mod5_residual_symbols,
    reconstruct_raw_tokens_from_mod5_residual_symbols,
    run_pr91_hpm1_context_window_probe,
    run_pr91_hpm1_entropy_failure_grammar_probe,
    run_pr91_hpm1_failure_row_probability_scan_probe,
    run_pr91_hpm1_first_symbol_state_probe,
    run_pr91_hpm1_next_row_suffix_scan_probe,
    run_pr91_hpm1_phase_major_prefix_reencode_blocker_probe,
    run_pr91_hpm1_preflight,
    run_pr91_hpm1_probability_variant_matrix,
    run_pr91_hpm1_reference_teacher_forcing_probe,
    run_pr91_hpm1_semantic_decode_trench,
    run_pr91_hpm1_semantic_symbol_bridge_probe,
    run_pr91_hpm1_spatial_group_order_probe,
    run_pr91_hpm1_submitted_prefix_token_recovery_probe,
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


def test_range_prefix_report_uses_full_submitted_stream_for_replay_fidelity() -> None:
    from tac import pr86_hpac_codec as hpac
    from tac.pr86_hpac_codec import _categorical_from_probs, resolve_hpac_probability_variant

    if hpac.constriction is None:
        pytest.skip("constriction range coder not available")

    variant = resolve_hpac_probability_variant("source_float64_perfect_false")
    prob_eps = 1e-7
    raw_rows = [
        [0.13967194, 0.35001042, 0.036982268, 0.18408185, 0.28925356],
        [0.32995138, 0.12997232, 0.14624487, 0.069891952, 0.32393944],
        [0.10581783, 0.19149975, 0.35863638, 0.0025626249, 0.34148338],
        [0.23717724, 0.14965372, 0.22940776, 0.34873247, 0.035028793],
        [0.030576656, 0.153622, 0.046496011, 0.41406411, 0.3552413],
        [0.14635567, 0.16271545, 0.3738628, 0.12896068, 0.1881054],
        [0.16463628, 0.24333154, 0.27017367, 0.20877452, 0.11308403],
        [0.11138923, 0.16113052, 0.21459962, 0.26848581, 0.24439484],
        [0.20496184, 0.23433173, 0.1810969, 0.24599564, 0.13361396],
        [0.27676371, 0.35983056, 0.01177596, 0.2360076, 0.11562212],
        [0.14937177, 0.33929735, 0.12357872, 0.23127869, 0.15647344],
    ]
    raw_rows_np = [
        np.asarray(row, dtype=np.float32)
        for row in raw_rows
    ]
    reference_symbols = [2, 3, 3, 3, 1, 0, 1, 1, 3, 0, 3]
    encoder = hpac.constriction.stream.queue.RangeEncoder()
    for index, symbol in enumerate(reference_symbols):
        encoder.encode(
            int(symbol),
            _categorical_from_probs(raw_rows_np[index], prob_eps=prob_eps, variant=variant),
        )
    submitted_words = np.ascontiguousarray(encoder.get_compressed(), dtype=np.uint32)

    report = _build_range_prefix_checkpoint_report(
        raw_rows_np,
        reference_symbols,
        submitted_words,
        {
            "label": "deterministic_prefix_fixture",
            "symbol_count": 9,
            "symbols_before_failure_remaining": 0,
            "includes_failure_symbol": False,
        },
        probability_variant=variant,
        prob_eps=prob_eps,
        replay_symbol_limit=32,
    )

    assert report["submitted_word_comparison"]["same_word_count_prefix_matches"] is False
    assert report["submitted_same_word_count_replay"]["passed"] is False
    assert report["submitted_full_stream_prefix_replay"]["passed"] is True
    assert report["submitted_full_stream_prefix_replay"]["decoded_symbol_count"] == 9
    assert report["submitted_word_comparison"][
        "submitted_full_stream_prefix_replay_passed"
    ] is True


def test_range_prefix_checkpoints_include_seed_counts_before_deep_failure() -> None:
    checkpoints = _prefix_checkpoint_symbol_counts(
        target_decoded_before=15989,
        target_group_start=15552,
        total_collected_symbols=15990,
        range_prefix_window_symbols=(1024, 1),
        range_prefix_seed_symbol_counts=(1, 8, 64, 20000),
    )

    by_count = {row["symbol_count"]: row for row in checkpoints}
    assert by_count[0]["label"] == "empty_stream"
    assert by_count[1]["label"] == "first_1_symbols"
    assert by_count[8]["label"] == "first_8_symbols"
    assert by_count[64]["label"] == "first_64_symbols"
    assert 20000 not in by_count
    assert by_count[14965]["label"] == "failure_minus_1024_symbols"
    assert by_count[15988]["label"] == "failure_minus_1_symbols"
    assert by_count[15989]["label"] == "before_failure_row"
    assert by_count[15990]["label"] == "including_failure_row"


def test_range_prefix_reconstruction_classifies_seed_reference_mismatch() -> None:
    summary = _classify_range_prefix_reconstruction(
        [
            {
                "label": "empty_stream",
                "symbol_count": 0,
                "submitted_full_stream_prefix_replay": {
                    "status": "decoded_empty_prefix",
                    "passed": True,
                },
                "submitted_word_comparison": {
                    "same_word_count_prefix_matches": True,
                    "common_prefix_word_count": 0,
                },
            },
            {
                "label": "first_8_symbols",
                "symbol_count": 8,
                "submitted_full_stream_prefix_replay": {
                    "status": "decoded_symbol_mismatch",
                    "passed": False,
                    "decoded_symbol_count": 8,
                    "first_mismatch": {
                        "symbol_index": 7,
                        "decoded_symbol": 2,
                        "reference_symbol": 0,
                    },
                },
                "submitted_word_comparison": {
                    "same_word_count_prefix_matches": False,
                    "common_prefix_word_count": 0,
                    "local_word_count": 1,
                    "submitted_total_word_count": 18268,
                },
            },
        ]
    )

    assert summary["schema"] == (
        "pr91_hpm1_range_prefix_reconstruction_classification_v1"
    )
    assert summary["status"] == (
        "submitted_stream_reference_symbol_mismatch_in_seed_prefix"
    )
    assert summary["score_claim"] is False
    assert summary["dispatch_allowed"] is False
    assert summary["byte_exact_reencode_proven"] is False
    assert summary["first_submitted_reference_prefix_failure"] == {
        "checkpoint_label": "first_8_symbols",
        "symbol_count": 8,
        "status": "decoded_symbol_mismatch",
        "decoded_symbol_count": 8,
        "first_mismatch": {
            "symbol_index": 7,
            "decoded_symbol": 2,
            "reference_symbol": 0,
        },
    }


def test_symbol_bridge_summary_rejects_conflicting_seed_mapping() -> None:
    summary = _summarize_reference_to_submitted_symbol_bridge(
        reference_symbols=[0, 1, 2, 3, 4, 0, 1, 0],
        submitted_symbols=[0, 1, 2, 3, 4, 0, 1, 2],
        previous_reference_symbols=[0, 0, 0, 0, 0, 0, 0, 0],
    )

    assert summary["schema"] == "pr91_hpm1_reference_to_submitted_symbol_bridge_v1"
    assert summary["status"] == "no_simple_reference_to_submitted_symbol_bridge_for_prefix"
    assert summary["score_claim"] is False
    assert summary["dispatch_allowed"] is False
    assert summary["bridge_found"] is False
    assert summary["first_identity_mismatch"] == {
        "symbol_index": 7,
        "candidate_symbol": 0,
        "submitted_symbol": 2,
    }
    permutation = summary["global_label_permutation_bridge"]
    assert permutation["status"] == "conflicting_reference_symbol_mappings"
    assert permutation["perfect_candidate_found"] is False
    assert permutation["conflicting_reference_symbols"] == [
        {"reference_symbol": 0, "observed_submitted_symbols": [0, 2]}
    ]


def test_symbol_bridge_summary_records_permutation_candidate() -> None:
    summary = _summarize_reference_to_submitted_symbol_bridge(
        reference_symbols=[0, 1, 2, 3, 4, 0, 2],
        submitted_symbols=[2, 0, 4, 1, 3, 2, 4],
        previous_reference_symbols=[0, 0, 0, 0, 0, 0, 0],
    )

    assert summary["status"] == "global_label_permutation_bridge_matches_submitted_prefix"
    assert summary["bridge_found"] is True
    permutation = summary["global_label_permutation_bridge"]
    assert permutation["perfect_candidate_found"] is True
    assert permutation["consistent_permutation_count"] == 1
    assert permutation["first_consistent_permutations"] == [[2, 0, 4, 1, 3]]
    assert summary["identity_bridge"]["all_symbols_match"] is False


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
    assert report["dispatch_allowed"] is False
    assert report["dispatch_performed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
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


def test_pr91_semantic_decode_trench_attaches_missing_symbol_bridge_proof() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    if not DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE.is_file():
        pytest.skip("PR85/QMA9 decoded reference token source not available")

    report = run_pr91_hpm1_semantic_decode_trench(
        DEFAULT_PR91_ARCHIVE,
        probability_row_count=1,
        prefix_max_frames=1,
        semantic_bridge_symbol_count=8,
        write_json=False,
    )

    bridge = report["semantic_symbol_bridge_probe"]
    assert report["prefix_decode"]["status"] == "failed_closed"
    assert bridge["attempted"] is True
    assert bridge["passed"] is True
    assert bridge["bridge_found"] is False
    assert bridge["bridge_missing"] is True
    assert bridge["decoded_symbol_count"] == 8
    assert bridge["first_identity_mismatch"] == {
        "symbol_index": 7,
        "candidate_symbol": 0,
        "submitted_symbol": 2,
    }
    assert (
        "semantic_symbol_bridge_missing:no_simple_pr85_qma9_to_pr91_prefix_bridge"
        in report["semantic_decode_blockers"]
    )
    assert report["prefix_decode"]["semantic_symbol_bridge_probe"] == bridge


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


def test_pr91_submitted_prefix_token_recovery_probe_recovers_bounded_prefix() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")

    report = run_pr91_hpm1_submitted_prefix_token_recovery_probe(
        DEFAULT_PR91_ARCHIVE,
        reference_tokens_path=None,
        max_symbols=8,
        row_preview_limit=4,
        mismatch_limit=0,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_submitted_prefix_token_recovery_probe_v1"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["status"] == "recovered_requested_submitted_prefix"
    trace = report["submitted_prefix_token_recovery"]
    assert trace["schema"] == "pr91_hpm1_submitted_prefix_token_recovery_trace_v1"
    assert trace["decoded_symbol_count"] == 8
    assert trace["submitted_symbols"]["first"] == [2, 2, 2, 2, 2, 2, 2, 2]
    assert trace["probability_row_trace"]["row_count"] == 8
    assert trace["probability_row_trace"]["normalized_rows_sha256"] == (
        "e0d10a91f0b9b42283aebbb3bde4d618950d8145c056c6e9d5801b10e0359cc9"
    )
    assert trace["failure"] is None
    assert trace["reference_comparison"]["attempted"] is False


def test_pr91_submitted_prefix_token_recovery_probe_can_probe_spatial_order_past_source_failure() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")

    report = run_pr91_hpm1_submitted_prefix_token_recovery_probe(
        DEFAULT_PR91_ARCHIVE,
        reference_tokens_path=None,
        spatial_order_candidate="tile_major_row_major",
        max_symbols=5960,
        row_preview_limit=0,
        mismatch_limit=0,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_submitted_prefix_token_recovery_probe_v1"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["status"] == "recovered_requested_submitted_prefix"
    assert report["spatial_order_probe"]["candidate"] == "tile_major_row_major"
    assert report["spatial_order_probe"]["source_contract"] is False
    trace = report["submitted_prefix_token_recovery"]
    assert trace["spatial_order_candidate"] == "tile_major_row_major"
    assert trace["decoded_symbol_count"] == 5960
    assert trace["full_decode_proven"] is False
    assert trace["byte_exact_reencode_proven"] is False
    assert trace["failure"] is None


def test_pr91_submitted_prefix_token_recovery_cli_records_tool_manifest(
    tmp_path: Path,
) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    out = tmp_path / "submitted_prefix_token_recovery_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(
                REPO
                / "tools"
                / "audit_pr91_hpm1_submitted_prefix_token_recovery_probe.py"
            ),
            "--archive",
            str(DEFAULT_PR91_ARCHIVE),
            "--max-symbols",
            "8",
            "--skip-reference",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_submitted_prefix_token_recovery_probe_v1"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["submitted_prefix_token_recovery"]["decoded_symbol_count"] == 8
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_submitted_prefix_token_recovery_probe.py"
    )


def test_pr91_submitted_prefix_token_recovery_cli_lists_spatial_order_flag() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO
                / "tools"
                / "audit_pr91_hpm1_submitted_prefix_token_recovery_probe.py"
            ),
            "--help",
        ],
        check=True,
        cwd=REPO,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "--spatial-order-candidate" in result.stdout
    assert "tile_major_row_major" in result.stdout


def test_pr91_next_row_suffix_scan_narrows_tile_major_blocker() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")

    report = run_pr91_hpm1_next_row_suffix_scan_probe(
        DEFAULT_PR91_ARCHIVE,
        spatial_order_candidate="tile_major_row_major",
        valid_preview_limit=4,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_next_row_suffix_scan_probe_v1"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["status"] == "narrowed_not_single_next_coordinate_error"
    trace = report["next_row_suffix_scan"]
    assert trace["schema"] == "pr91_hpm1_next_row_suffix_scan_trace_v1"
    assert trace["decoded_symbol_count_before_failure"] == 8274
    assert trace["failure"]["frame"] == 0
    assert trace["failure"]["group"] == 12
    assert trace["failure"]["symbol_in_group"] == 210
    assert trace["failure"]["failing_probability_row"]["normalized_sha256"] == (
        "8216c3d82263ef0fc10c88ddf28439b0916ae83865c8d14d9e37bd785bd2b7cd"
    )
    scan = trace["suffix_scan"]
    assert scan["classification"] == "no_remaining_group_row_decodes_from_failure_state"
    assert scan["candidate_rows_tested"] == 1134
    assert scan["valid_next_row_count"] == 0
    assert scan["valid_next_row_preview"] == []
    assert report["exact_missing_grammar"]["narrowed"] == (
        "same-group next-coordinate/order explanation at the first failure row"
    )


def test_pr91_next_row_suffix_scan_cli_records_tool_manifest(tmp_path: Path) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    out = tmp_path / "next_row_suffix_scan_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_pr91_hpm1_next_row_suffix_scan_probe.py"),
            "--archive",
            str(DEFAULT_PR91_ARCHIVE),
            "--valid-preview-limit",
            "2",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_next_row_suffix_scan_probe_v1"
    assert payload["status"] == "narrowed_not_single_next_coordinate_error"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["next_row_suffix_scan"]["suffix_scan"]["valid_next_row_count"] == 0
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_next_row_suffix_scan_probe.py"
    )


def test_pr91_failure_row_probability_scan_narrows_tile_major_blocker() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")

    report = run_pr91_hpm1_failure_row_probability_scan_probe(
        DEFAULT_PR91_ARCHIVE,
        spatial_order_candidate="tile_major_row_major",
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_failure_row_probability_scan_probe_v1"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["status"] == "narrowed_failure_row_probability_numeric_false_lead"
    assert report["grand_council_stop_rule"]["material_unlock_found"] is False
    assert report["grand_council_stop_rule"]["redirect_recommendation"] == (
        "stop HPM1 wall-clock unless a real encoder trace/source appears; "
        "prioritize other frontier replacements or categorical candidates"
    )
    trace = report["failure_row_probability_probe"]["prefix_trace"]
    assert trace["schema"] == "pr91_hpm1_failure_row_probability_scan_trace_v1"
    assert trace["decoded_symbol_count_before_failure"] == 8274
    assert trace["failure"]["frame"] == 0
    assert trace["failure"]["group"] == 12
    assert trace["failure"]["symbol_in_group"] == 210
    assert trace["failure"]["failing_probability_row"]["normalized_sha256"] == (
        "8216c3d82263ef0fc10c88ddf28439b0916ae83865c8d14d9e37bd785bd2b7cd"
    )
    scan = trace["failure_row_probability_scan"]
    assert scan["classification"] == (
        "failure_row_not_decodable_under_probability_numeric_scan"
    )
    assert scan["candidate_rows_tested"] == 192
    assert scan["decodable_candidate_count"] == 0
    assert scan["baseline_source_variant_decodes"] is False
    assert scan["failures_by_exception_type"] == {"AssertionError": 192}


def test_pr91_failure_row_probability_scan_cli_records_tool_manifest(
    tmp_path: Path,
) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    out = tmp_path / "failure_row_probability_scan_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(
                REPO
                / "tools"
                / "audit_pr91_hpm1_failure_row_probability_scan_probe.py"
            ),
            "--archive",
            str(DEFAULT_PR91_ARCHIVE),
            "--scan-variants",
            "source_float64_perfect_false",
            "--scan-prob-eps-values",
            "1e-7",
            "--uniform-mix-masses",
            "0",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_failure_row_probability_scan_probe_v1"
    assert payload["status"] == "narrowed_failure_row_probability_numeric_false_lead"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    scan = payload["failure_row_probability_probe"]["prefix_trace"][
        "failure_row_probability_scan"
    ]
    assert scan["candidate_rows_tested"] == 1
    assert scan["decodable_candidate_count"] == 0
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_failure_row_probability_scan_probe.py"
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


def test_pr91_reference_teacher_forcing_probe_cli_narrows_false_lead(
    tmp_path: Path,
) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    if not DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE.is_file():
        pytest.skip("PR85/QMA9 decoded reference token source not available")
    out = tmp_path / "reference_teacher_forcing_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(
                REPO
                / "tools"
                / "audit_pr91_hpm1_reference_teacher_forcing_probe.py"
            ),
            "--archive",
            str(DEFAULT_PR91_ARCHIVE),
            "--reference-tokens",
            str(DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE),
            "--reference-layout",
            "legacy_assume_nhw",
            "--spatial-order-candidates",
            "tile_major_row_major",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_reference_teacher_forcing_probe_v1"
    assert payload["status"] == "narrowed_pr85_qma9_reference_teacher_forcing_false_lead"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["evidence_grade"] == "empirical"
    assert payload["evidence_scope"] == (
        "local_cpu_hpm1_reference_teacher_forcing_hypothesis_probe"
    )
    assert payload["reference_tokens"]["matches_expected_pr85_qma9_token_source"] is True
    assert payload["reference_token_sha256_contract"]["matches_expected"] is True
    assert payload["source_order_baseline"]["failure_signature"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
        "decoded_symbol_count_before_failure": 5951,
    }
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_reference_teacher_forcing_probe.py"
    )
    probe = payload["reference_teacher_forcing_probe"]
    assert probe["status"] == "not_explained_by_pr85_qma9_reference_teacher_forcing"
    progress = probe["candidate_progress_summary"]
    assert progress["schema"] == (
        "pr91_hpm1_reference_teacher_forcing_candidate_progress_v1"
    )
    assert progress["advancing_candidates"] == []
    assert progress["regressing_candidates"] == ["tile_major_row_major"]
    assert progress["candidate_rows"][0]["status"] == (
        "reference_forcing_regresses_candidate"
    )
    assert probe["candidate_scope"] == {
        "requested_candidates": ["tile_major_row_major"],
        "label": "tile-major row-major",
    }
    assert "phase-major" not in probe["narrowed_hypothesis"]
    assert probe["advanced_candidates"] == []
    row = probe["candidate_results"][0]
    assert row["candidate"] == "tile_major_row_major"
    assert row["decoded_context"]["failure_signature"] == {
        "frame": 0,
        "group": 12,
        "symbol_in_group": 210,
        "decoded_symbol_count_before_failure": 8274,
    }
    assert row["reference_teacher_forced_context"]["failure_signature"] == {
        "frame": 0,
        "group": 5,
        "symbol_in_group": 305,
        "decoded_symbol_count_before_failure": 2033,
    }
    assert row["reference_teacher_forced_context"][
        "advances_beyond_decoded_context"
    ] is False
    assert row["reference_teacher_forced_context"]["full_decode_proven"] is False
    assert row["reference_teacher_forced_context"]["byte_exact_reencode_proven"] is False
    assert row["reference_teacher_forced_context"][
        "reference_mismatch_count_before_failure"
    ] == 683
    assert row["reference_teacher_forced_context"][
        "first_decoded_reference_mismatch"
    ] == {
        "global_symbol": 7,
        "frame": 0,
        "group": 0,
        "symbol_in_group": 7,
        "pixel_yx": {"y": 0, "x": 224},
        "decoded_symbol": 2,
        "reference_symbol": 0,
    }
    reference_window = row["reference_teacher_forced_context"][
        "canonical_reference_symbol_window"
    ]
    assert reference_window["schema"] == "pr91_hpm1_reference_group_symbol_window_v1"
    assert reference_window["frame"] == 0
    assert reference_window["group"] == 5
    assert reference_window["failure_symbol_in_group"] == 305
    assert reference_window["next_reference_symbol_count"] == 5
    assert reference_window["rows"][2]["relative_to_failure"] == 0
    assert row["reference_teacher_forced_context"]["range_decoder_diagnostic"][
        "not_stream_exhaustion"
    ] is True


def test_pr91_reference_teacher_forcing_probe_cli_help_lists_range_prefix_flags() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_pr91_hpm1_reference_teacher_forcing_probe.py"),
            "--help",
        ],
        check=True,
        cwd=REPO,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "--range-prefix-probe" in result.stdout
    assert "--range-prefix-window-symbols" in result.stdout
    assert "--range-prefix-seed-symbol-counts" in result.stdout
    assert "--range-prefix-replay-symbol-limit" in result.stdout
    assert "--range-prefix-max-target-decoded-before" in result.stdout


def test_pr91_reference_teacher_forcing_probe_records_phase_major_next_row_artifact() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    if not DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE.is_file():
        pytest.skip("PR85/QMA9 decoded reference token source not available")

    report = run_pr91_hpm1_reference_teacher_forcing_probe(
        DEFAULT_PR91_ARCHIVE,
        reference_tokens_path=DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
        reference_layout="legacy_assume_nhw",
        candidates=("phase_major_row_major",),
        reference_window_before=1,
        reference_window_after=3,
        run_range_prefix_probe=True,
        range_prefix_window_symbols=(1,),
        range_prefix_replay_symbol_limit=0,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_reference_teacher_forcing_probe_v1"
    assert report["status"] == "reference_teacher_forcing_hypothesis_still_open"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    probe = report["reference_teacher_forcing_probe"]
    assert probe["advanced_candidates"] == ["phase_major_row_major"]
    progress = probe["candidate_progress_summary"]
    assert progress["advancing_candidates"] == ["phase_major_row_major"]
    assert progress["regressing_candidates"] == []
    assert progress["candidate_rows"][0]["delta_reference_vs_decoded"] == 9063
    row = probe["candidate_results"][0]["reference_teacher_forced_context"]
    assert row["failure_signature"] == {
        "frame": 0,
        "group": 17,
        "symbol_in_group": 437,
        "decoded_symbol_count_before_failure": 15989,
    }
    assert row["advances_beyond_decoded_context"] is True
    assert row["range_decoder_diagnostic"]["not_stream_exhaustion"] is True
    assert row["range_decoder_diagnostic"]["state_before_decode"][
        "maybe_exhausted"
    ] is False
    interpretation = row["failure_row_interpretation"]
    assert interpretation["schema"] == "pr91_hpm1_failure_row_interpretation_v1"
    assert interpretation["status"] == (
        "not_explained_by_current_row_reference_symbol_probability"
    )
    assert interpretation["reference_symbol"] == 2
    assert interpretation["reference_symbol_is_argmax"] is True
    assert interpretation["reference_symbol_rank"] == 1
    assert interpretation["reference_symbol_probability"] == pytest.approx(
        0.6799724541
    )
    assert interpretation["decoder_not_stream_exhaustion"] is True
    roundtrip = interpretation["single_row_range_model_roundtrip"]
    assert roundtrip["schema"] == "pr91_hpm1_single_row_range_model_roundtrip_v1"
    assert roundtrip["status"] == "passed_all_symbols_roundtrip"
    assert roundtrip["all_symbols_roundtrip"] is True
    assert roundtrip["failed_symbols"] == []
    assert [entry["symbol"] for entry in roundtrip["symbol_results"]] == [0, 1, 2, 3, 4]
    classification = report["hypothesis_classification"]
    assert classification["schema"] == (
        "pr91_hpm1_decode_failure_hypothesis_classification_v1"
    )
    assert classification["status"] == (
        "narrowed_to_range_or_probability_context_grammar_after_phase_major_reference_row"
    )
    assert classification["row_ordering"]["status"] == (
        "phase_major_advances_but_does_not_decode_full_prefix"
    )
    assert classification["semantic_token_interpretation"]["status"] == (
        "not_explained_by_current_row_available_reference_symbol"
    )
    assert classification["range_coder_contract"]["status"] == (
        "still_open_prior_range_state_or_encoder_finalization_contract"
    )
    assert classification["probability_context_grammar"]["status"] == (
        "still_open_prior_context_or_probability_numeric_contract"
    )
    reference_window = row["canonical_reference_symbol_window"]
    assert reference_window["failure_symbol_in_group"] == 437
    assert reference_window["window_before"] == 1
    assert reference_window["window_after"] == 3
    assert reference_window["next_reference_symbol_count"] == 3
    assert [
        entry["relative_to_failure"] for entry in reference_window["rows"]
    ] == [-1, 0, 1, 2, 3]
    assert len(reference_window["reference_symbols_sha256"]) == 64
    assert reference_window["failed_reference_symbol"] in range(5)
    range_prefix = row["range_state_prefix_probe"]
    assert range_prefix["schema"] == "pr91_hpm1_range_state_prefix_probe_v1"
    assert range_prefix["status"] == "not_attempted_target_exceeds_symbol_budget"
    assert range_prefix["score_claim"] is False
    assert range_prefix["dispatch_allowed"] is False
    assert range_prefix["full_decode_proven"] is False
    assert range_prefix["byte_exact_reencode_proven"] is False
    assert range_prefix["target_failure"] == {
        "frame": 0,
        "group": 17,
        "symbol_in_group": 437,
        "decoded_symbol_count_before_failure": 15989,
    }
    assert range_prefix["target_decoded_symbols_before_failure"] == 15989
    assert range_prefix["range_prefix_max_target_decoded_before"] == 4096
    assert "slow forensic runs" in range_prefix["reason"]


def test_pr91_phase_major_prefix_reencode_blocker_records_exact_seed_failure() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    if not DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE.is_file():
        pytest.skip("PR85/QMA9 decoded reference token source not available")

    report = run_pr91_hpm1_phase_major_prefix_reencode_blocker_probe(
        DEFAULT_PR91_ARCHIVE,
        reference_tokens_path=DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
        range_prefix_seed_symbol_counts=(1, 8, 64),
        range_prefix_replay_symbol_limit=64,
        range_prefix_max_target_decoded_before=20000,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_phase_major_prefix_reencode_blocker_v1"
    assert report["status"] == (
        "blocked_phase_major_reference_prefix_not_byte_exact_reencode"
    )
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["blocker_classified"] is True
    assert report["archive"]["sha256"] == (
        "4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f"
    )
    assert report["reference_tokens"]["sha256"] == (
        "c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a"
    )
    range_probe = report["range_state_prefix_probe"]
    assert range_probe["status"] == "target_failure_reproduced_with_reference_context"
    assert range_probe["target_failure_reproduced"] is True
    assert range_probe["observed_failure"] == {
        "decoded_symbol_count_before_failure": 15989,
        "frame": 0,
        "group": 17,
        "symbol_in_group": 437,
    }
    rows_before = range_probe["probability_row_sequence_hashes"][
        "rows_before_failure"
    ]
    assert rows_before["count"] == 15989
    assert rows_before["normalized_for_categorical_sha256"] == (
        "eafd81c8b25c675b79ad0b3096274d9ba4e4c8c0084d9d1195cae61a57edf638"
    )
    exact_findings = report["exact_fail_closed_findings"]
    assert exact_findings["first_local_reference_word_mismatch"] == {
        "checkpoint_label": "first_1_symbols",
        "common_prefix_word_count": 0,
        "local_word_count": 1,
        "submitted_total_word_count": 29199,
        "symbol_count": 1,
    }
    assert exact_findings["first_submitted_reference_prefix_failure"] == {
        "checkpoint_label": "first_8_symbols",
        "decoded_symbol_count": 8,
        "first_mismatch": {
            "decoded_symbol": 2,
            "reference_symbol": 0,
            "symbol_index": 7,
        },
        "status": "decoded_symbol_mismatch",
        "symbol_count": 8,
    }
    assert exact_findings["byte_exact_reencode_proven"] is False


def test_pr91_phase_major_prefix_reencode_blocker_cli_help_lists_controls() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO
                / "tools"
                / "audit_pr91_hpm1_phase_major_prefix_reencode_blocker.py"
            ),
            "--help",
        ],
        check=True,
        cwd=REPO,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "--range-prefix-seed-symbol-counts" in result.stdout
    assert "--range-prefix-replay-symbol-limit" in result.stdout
    assert "--range-prefix-max-target-decoded-before" in result.stdout


def test_pr91_semantic_symbol_bridge_probe_narrows_first_8_phase_major_seed() -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    if not DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE.is_file():
        pytest.skip("PR85/QMA9 decoded reference token source not available")

    report = run_pr91_hpm1_semantic_symbol_bridge_probe(
        DEFAULT_PR91_ARCHIVE,
        reference_tokens_path=DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
        reference_layout="legacy_assume_nhw",
        spatial_order_candidate="phase_major_row_major",
        symbol_count=8,
        row_preview_limit=8,
        mismatch_limit=8,
        write_json=False,
    )

    assert report["schema"] == "pr91_hpm1_semantic_symbol_bridge_probe_v1"
    assert report["status"] == "narrowed_no_simple_pr85_qma9_to_pr91_symbol_bridge"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    trace = report["symbol_bridge_probe"]["prefix_trace"]
    assert trace["status"] == "submitted_prefix_has_no_simple_reference_symbol_bridge"
    assert trace["prefix_completed"] is True
    assert trace["decoded_symbol_count"] == 8
    assert trace["symbol_sequences"]["first_reference_symbols"] == [2, 2, 2, 2, 2, 2, 2, 0]
    assert trace["symbol_sequences"]["first_submitted_symbols"] == [2, 2, 2, 2, 2, 2, 2, 2]
    assert trace["symbol_sequences"]["first_reference_submitted_mismatch"] == {
        "symbol_index": 7,
        "reference_symbol": 0,
        "submitted_symbol": 2,
    }
    bridge = trace["bridge_summary"]
    assert bridge["status"] == "no_simple_reference_to_submitted_symbol_bridge_for_prefix"
    assert bridge["bridge_found"] is False
    assert bridge["identity_bridge"]["match_count"] == 7
    assert bridge["global_label_permutation_bridge"]["status"] == "no_consistent_permutation"
    assert bridge["global_label_permutation_bridge"]["consistent_permutation_count"] == 0
    assert bridge["best_mod5_offset_bridge"]["candidate"] == "reference_plus_0_mod5"
    assert bridge["best_mod5_offset_bridge"]["match_count"] == 7
    assert trace["mismatch_rows"][0]["global_symbol"] == 7
    assert trace["mismatch_rows"][0]["pixel_yx"] == {"y": 0, "x": 224}
    assert trace["mismatch_rows"][0]["reference_symbol_probability"] == pytest.approx(0.0343694585)
    assert trace["mismatch_rows"][0]["submitted_symbol_probability"] == pytest.approx(0.939719006)


def test_pr91_semantic_symbol_bridge_probe_cli_records_tool_manifest(tmp_path: Path) -> None:
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip("canonical PR91 archive not available")
    if not DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE.is_file():
        pytest.skip("PR85/QMA9 decoded reference token source not available")
    out = tmp_path / "semantic_symbol_bridge_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(
                REPO
                / "tools"
                / "audit_pr91_hpm1_semantic_symbol_bridge_probe.py"
            ),
            "--archive",
            str(DEFAULT_PR91_ARCHIVE),
            "--reference-tokens",
            str(DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE),
            "--symbol-count",
            "8",
            "--row-preview-limit",
            "4",
            "--mismatch-limit",
            "4",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr91_hpm1_semantic_symbol_bridge_probe_v1"
    assert payload["score_claim"] is False
    assert payload["dispatch_allowed"] is False
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/audit_pr91_hpm1_semantic_symbol_bridge_probe.py"
    )
    assert payload["symbol_bridge_probe"]["prefix_trace"]["decoded_symbol_count"] == 8


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
    with pytest.raises(Pr91Hpm1Error, match="reference_tokens_missing"):
        run_pr91_hpm1_reference_teacher_forcing_probe(
            archive,
            reference_tokens_path=tmp_path / "missing_tokens.raw",
            candidates=("tile_major_row_major",),
        )
    wrong_tokens = tmp_path / "wrong_tokens.raw"
    wrong_tokens.write_bytes(b"\0" * 32)
    with pytest.raises(
        Pr91Hpm1Error,
        match="unexpected_pr85_qma9_reference_token_sha256",
    ):
        run_pr91_hpm1_reference_teacher_forcing_probe(
            archive,
            reference_tokens_path=wrong_tokens,
            candidates=("tile_major_row_major",),
        )
    with pytest.raises(Pr91Hpm1Error, match="reference_window_counts_must_be_nonnegative"):
        run_pr91_hpm1_reference_teacher_forcing_probe(
            archive,
            reference_window_after=-1,
        )
    with pytest.raises(Pr91Hpm1Error, match="range_prefix_window_symbols_must_be_positive"):
        run_pr91_hpm1_reference_teacher_forcing_probe(
            archive,
            run_range_prefix_probe=True,
            range_prefix_window_symbols=(0,),
        )
    with pytest.raises(Pr91Hpm1Error, match="range_prefix_seed_symbol_counts_must_be_positive"):
        run_pr91_hpm1_reference_teacher_forcing_probe(
            archive,
            run_range_prefix_probe=True,
            range_prefix_seed_symbol_counts=(0,),
        )
    with pytest.raises(Pr91Hpm1Error, match="replay_symbol_limit_must_be_nonnegative"):
        run_pr91_hpm1_reference_teacher_forcing_probe(
            archive,
            run_range_prefix_probe=True,
            range_prefix_replay_symbol_limit=-1,
        )
    with pytest.raises(
        Pr91Hpm1Error,
        match="range_prefix_max_target_decoded_before_must_be_nonnegative",
    ):
        run_pr91_hpm1_reference_teacher_forcing_probe(
            archive,
            run_range_prefix_probe=True,
            range_prefix_max_target_decoded_before=-1,
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
