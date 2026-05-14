# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli
import pytest

from tac.hnerv_decoder_recode import (
    PACKED_STATE_SCHEMA,
    decode_hdm3_q_brotli_split_fixture,
    decode_hdm4_q_brotli_split_fixture,
    decode_hdm6_q_brotli_tuned_fixture,
    decode_hdm7_q_brotli_len_elided_fixture,
    decode_hdm8_q_brotli_recipe_elided_fixture,
)
from tac.hnerv_hdm3_archive_candidate import (
    build_hdm3_archive_candidate,
    build_hdm3_exact_eval_packet_readiness,
)
from tac.hnerv_lowlevel_packer import (
    parse_ff_packed_brotli_hnerv,
    read_packed_archive_view,
    read_strict_single_member_zip,
    write_stored_single_member_zip,
)
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106SidecarPacket,
    emit_pr106_sidecar_packet,
    parse_pr106_sidecar_packet,
)

REPO = Path(__file__).resolve().parents[3]


def test_build_hdm3_archive_candidate_is_byte_closed_but_not_dispatch_ready(
    tmp_path: Path,
) -> None:
    source_archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(source_archive)
    source_packed = parse_ff_packed_brotli_hnerv(source.payload)
    source_raw = brotli.decompress(source_packed.decoder_packed_brotli)

    manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="PR106x frontier",
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_build_gate"] is True
    assert manifest["candidate_variant"] == "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales"
    assert manifest["candidate_rate_positive"] is True
    assert manifest["candidate_decoder_section_byte_delta"] < 0
    assert manifest["candidate_archive_sha256"] != source.archive_sha256
    assert manifest["decoder_raw_equivalence"]["raw_equal"] is True
    assert manifest["decoder_raw_equivalence"]["q_roundtrip_equal"] is True
    assert manifest["decoder_raw_equivalence"]["scale_roundtrip_equal"] is True
    assert manifest["runtime_adapter_proof"]["runtime_adapter_parity_proven"] is False
    assert manifest["runtime_adapter_proof"]["runtime_adapter_module"] == "tac.hnerv_hdm3_runtime_adapter"
    assert manifest["fixed_runtime_preflight"]["ready_for_fixed_runtime_exact_eval_readiness"] is False
    assert "hdm3_runtime_adapter_equivalence_not_proven" in manifest["fixed_runtime_preflight"][
        "blockers"
    ]
    assert "hdm3_runtime_adapter_archive_parity_proof_missing" in manifest["dispatch_blockers"]
    assert "strict_pre_submission_compliance_json_missing" in manifest["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in manifest["dispatch_blockers"]
    assert manifest["exact_eval_packet_readiness"]["static_packet_ready"] is False
    readiness_path = REPO / manifest["exact_eval_packet_readiness"]["path"]
    if not readiness_path.exists():
        readiness_path = Path(manifest["exact_eval_packet_readiness"]["path"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["ready_for_exact_eval_packet"] is False
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["lane_dispatch_claim"]["required_before_gpu"] is True

    release_surface = manifest["exact_eval_release_surface"]
    release_archive = REPO / release_surface["archive_path"]
    release_inflate = REPO / release_surface["inflate_sh"]
    release_report = REPO / release_surface["report_txt"]
    release_manifest = REPO / release_surface["archive_manifest_json"]
    for path in (release_archive, release_inflate, release_report, release_manifest):
        if not path.exists():
            path = Path(str(path).removeprefix(f"{REPO.as_posix()}/"))
        assert path.exists()
    assert os.access(release_inflate, os.X_OK)

    candidate_archive = REPO / manifest["candidate_archive_path"]
    if not candidate_archive.exists():
        candidate_archive = Path(manifest["candidate_archive_path"])
    assert candidate_archive.exists()
    candidate = read_strict_single_member_zip(candidate_archive)
    assert candidate.member_name == source.member_name
    candidate_packed = parse_ff_packed_brotli_hnerv(candidate.payload)
    assert candidate_packed.decoder_packed_brotli.startswith(b"HDM3")
    assert candidate_packed.latents_and_sidecar_brotli == source_packed.latents_and_sidecar_brotli
    restored = decode_hdm3_q_brotli_split_fixture(candidate_packed.decoder_packed_brotli)
    assert restored.to_raw() == source_raw

    with zipfile.ZipFile(candidate_archive) as zf:
        infos = zf.infolist()
    assert [info.filename for info in infos] == ["x"]
    assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)


def test_build_hdm3_archive_candidate_preserves_pr106_sidecar_wrapper(
    tmp_path: Path,
) -> None:
    source_archive = _source_sidecar_archive(tmp_path)
    source_view = read_packed_archive_view(source_archive)
    source_packet = parse_pr106_sidecar_packet(source_view.archive.payload)
    source_raw = brotli.decompress(source_view.packed.decoder_packed_brotli)

    manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="PR106 R2 sidecar",
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["source_payload_kind"] == "pr106_sidecar_wrapper"
    assert manifest["candidate_rate_positive"] is True
    candidate_archive = Path(manifest["candidate_archive_path"])
    assert candidate_archive.exists()
    candidate_view = read_packed_archive_view(candidate_archive)
    candidate_packet = parse_pr106_sidecar_packet(candidate_view.archive.payload)
    assert candidate_view.payload_kind == "pr106_sidecar_wrapper"
    assert candidate_packet.format_id == source_packet.format_id
    assert candidate_packet.sidecar_payload == source_packet.sidecar_payload
    assert candidate_view.packed.decoder_packed_brotli.startswith(b"HDM3")
    assert candidate_view.packed.latents_and_sidecar_brotli == source_view.packed.latents_and_sidecar_brotli
    restored = decode_hdm3_q_brotli_split_fixture(candidate_view.packed.decoder_packed_brotli)
    assert restored.to_raw() == source_raw


def test_build_hdm4_archive_candidate_uses_distinct_lane_and_preserves_raw_decoder(
    tmp_path: Path,
) -> None:
    source_archive = _source_sidecar_archive(tmp_path)
    source_view = read_packed_archive_view(source_archive)
    source_raw = brotli.decompress(source_view.packed.decoder_packed_brotli)

    manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="PR106 R2 sidecar",
        decoder_recode_variant="hdm4",
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["candidate_decoder_recode_key"] == "hdm4"
    assert manifest["candidate_variant"] == "hdm4_q_brotli_split_fixed_recipe_dp4_plus_raw_scales"
    assert manifest["lane_id"] == "hnerv_hdm4_q_brotli_split_exact_eval"
    assert manifest["candidate_rate_positive"] is True
    assert manifest["candidate_decoder_section_byte_delta"] < 0
    assert manifest["hdm3_stats"] == {}
    assert manifest["hdm4_stats"]["recipe_id"] == 1
    assert manifest["hdm4_stats"]["split_points"] == [6, 9, 26, 28]

    candidate_archive = Path(manifest["candidate_archive_path"])
    candidate_view = read_packed_archive_view(candidate_archive)
    assert candidate_view.payload_kind == "pr106_sidecar_wrapper"
    assert candidate_view.packed.decoder_packed_brotli.startswith(b"HDM4")
    assert candidate_view.packed.latents_and_sidecar_brotli == source_view.packed.latents_and_sidecar_brotli
    restored = decode_hdm4_q_brotli_split_fixture(candidate_view.packed.decoder_packed_brotli)
    assert restored.to_raw() == source_raw
    blockers = manifest["exact_eval_packet_readiness"]["remaining_dispatch_blockers"]
    assert "hdm3_runtime_adapter_archive_parity_proof_missing" in blockers
    assert "strict_pre_submission_compliance_json_missing" in blockers
    assert "lane_dispatch_claim_missing" in blockers
    assert "exact_cuda_auth_eval_missing" in blockers


def test_build_hdm6_archive_candidate_accepts_hdm4_source_and_saves_bytes(
    tmp_path: Path,
) -> None:
    source_archive = _source_sidecar_archive(tmp_path)
    hdm4_manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "hdm4",
        source_label="PR106 R2 sidecar",
        decoder_recode_variant="hdm4",
        repo_root=REPO,
    )
    hdm4_archive = Path(hdm4_manifest["candidate_archive_path"])
    hdm4_view = read_packed_archive_view(hdm4_archive)
    hdm4_raw = decode_hdm4_q_brotli_split_fixture(
        hdm4_view.packed.decoder_packed_brotli
    ).to_raw()

    manifest = build_hdm3_archive_candidate(
        source_archive=hdm4_archive,
        output_dir=tmp_path / "hdm6",
        source_label="PR106 R2 HDM4 sidecar",
        decoder_recode_variant="hdm6",
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["candidate_decoder_recode_key"] == "hdm6"
    assert manifest["candidate_variant"] == "hdm6_q_brotli_split_fixed_recipe_tuned_lgwin_plus_raw_scales"
    assert manifest["lane_id"] == "hnerv_hdm6_q_brotli_tuned_exact_eval"
    assert manifest["source_decoder_section_codec"] == "hdm4_q_brotli_split"
    assert manifest["candidate_rate_positive"] is True
    assert manifest["candidate_decoder_section_byte_delta"] < 0
    assert manifest["hdm3_stats"] == {}
    assert manifest["hdm4_stats"] == {}
    assert manifest["hdm6_stats"]["recipe_id"] == 1
    assert manifest["hdm6_stats"]["brotli_params_by_chunk"] == [
        {"quality": 11, "lgwin": 18, "mode": brotli.MODE_GENERIC},
        {"quality": 11, "lgwin": 16, "mode": brotli.MODE_GENERIC},
        {"quality": 11, "lgwin": 16, "mode": brotli.MODE_GENERIC},
        {"quality": 10, "lgwin": 16, "mode": brotli.MODE_GENERIC},
    ]

    candidate_archive = Path(manifest["candidate_archive_path"])
    candidate_view = read_packed_archive_view(candidate_archive)
    assert candidate_view.payload_kind == "pr106_sidecar_wrapper"
    assert candidate_view.packed.decoder_packed_brotli.startswith(b"HDM6")
    assert candidate_view.packed.latents_and_sidecar_brotli == hdm4_view.packed.latents_and_sidecar_brotli
    restored = decode_hdm6_q_brotli_tuned_fixture(candidate_view.packed.decoder_packed_brotli)
    assert restored.to_raw() == hdm4_raw


def test_build_hdm7_archive_candidate_accepts_hdm6_source_and_elides_final_length(
    tmp_path: Path,
) -> None:
    source_archive = _source_sidecar_archive(tmp_path)
    hdm4_manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "hdm4",
        source_label="PR106 R2 sidecar",
        decoder_recode_variant="hdm4",
        repo_root=REPO,
    )
    hdm6_manifest = build_hdm3_archive_candidate(
        source_archive=hdm4_manifest["candidate_archive_path"],
        output_dir=tmp_path / "hdm6",
        source_label="PR106 R2 HDM4 sidecar",
        decoder_recode_variant="hdm6",
        repo_root=REPO,
    )
    hdm6_archive = Path(hdm6_manifest["candidate_archive_path"])
    hdm6_view = read_packed_archive_view(hdm6_archive)
    hdm6_raw = decode_hdm6_q_brotli_tuned_fixture(
        hdm6_view.packed.decoder_packed_brotli
    ).to_raw()

    manifest = build_hdm3_archive_candidate(
        source_archive=hdm6_archive,
        output_dir=tmp_path / "hdm7",
        source_label="PR106 R2 HDM6 sidecar",
        decoder_recode_variant="hdm7",
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["candidate_decoder_recode_key"] == "hdm7"
    assert (
        manifest["candidate_variant"]
        == "hdm7_q_brotli_split_fixed_recipe_tuned_lgwin_final_len_elided_plus_raw_scales"
    )
    assert manifest["lane_id"] == "hnerv_hdm7_final_len_elided_exact_eval"
    assert manifest["source_decoder_section_codec"] == "hdm6_q_brotli_tuned_split"
    assert manifest["candidate_rate_positive"] is True
    assert manifest["candidate_decoder_section_byte_delta"] == -3
    assert manifest["candidate_archive_byte_delta"] == -3
    assert manifest["hdm3_stats"] == {}
    assert manifest["hdm4_stats"] == {}
    assert manifest["hdm6_stats"] == {}
    assert manifest["hdm7_stats"]["recipe_id"] == 1
    assert manifest["hdm7_stats"]["header_bytes"] == 14
    assert manifest["hdm7_stats"]["elided_len24_bytes"] == 3
    assert manifest["hdm7_stats"]["final_chunk_len_elided"] is True

    candidate_archive = Path(manifest["candidate_archive_path"])
    candidate_view = read_packed_archive_view(candidate_archive)
    assert candidate_view.payload_kind == "pr106_sidecar_wrapper"
    assert candidate_view.packed.decoder_packed_brotli.startswith(b"HDM7")
    assert candidate_view.packed.latents_and_sidecar_brotli == hdm6_view.packed.latents_and_sidecar_brotli
    restored = decode_hdm7_q_brotli_len_elided_fixture(candidate_view.packed.decoder_packed_brotli)
    assert restored.to_raw() == hdm6_raw


def test_build_hdm8_archive_candidate_accepts_hdm7_source_and_elides_recipe_id(
    tmp_path: Path,
) -> None:
    hdm7_archive = (
        REPO
        / "experiments/results/pr106_r2_hdm6_hlm2_hdm7_candidate_20260514_codex/"
        "pr106_r2_hdm6_hlm2_xmember_hdm7_archive_candidate.zip"
    )
    if not hdm7_archive.exists():
        pytest.skip("HDM7 exact-CUDA candidate artifact is not present in this checkout")
    hdm7_view = read_packed_archive_view(hdm7_archive)
    hdm7_raw = decode_hdm7_q_brotli_len_elided_fixture(
        hdm7_view.packed.decoder_packed_brotli
    ).to_raw()

    manifest = build_hdm3_archive_candidate(
        source_archive=hdm7_archive,
        output_dir=tmp_path / "hdm8",
        source_label="PR106 R2 HDM7 sidecar",
        decoder_recode_variant="hdm8",
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["candidate_decoder_recode_key"] == "hdm8"
    assert (
        manifest["candidate_variant"]
        == "hdm8_q_brotli_split_fixed_recipe_tuned_lgwin_recipe_chunk_lengths_elided_plus_raw_scales"
    )
    assert manifest["lane_id"] == "hnerv_hdm8_fixed_lengths_exact_eval"
    assert manifest["source_decoder_section_codec"] == "hdm7_q_brotli_len_elided_split"
    assert manifest["candidate_rate_positive"] is True
    assert manifest["candidate_decoder_section_byte_delta"] == -10
    assert manifest["candidate_archive_byte_delta"] == -10
    assert manifest["hdm3_stats"] == {}
    assert manifest["hdm4_stats"] == {}
    assert manifest["hdm6_stats"] == {}
    assert manifest["hdm7_stats"] == {}
    assert manifest["hdm8_stats"]["header_bytes"] == 4
    assert manifest["hdm8_stats"]["elided_len24_bytes"] == 9
    assert manifest["hdm8_stats"]["elided_recipe_id_bytes"] == 1
    assert manifest["hdm8_stats"]["recipe_id_elided"] is True
    assert manifest["hdm8_stats"]["fixed_chunk_lengths_enforced"] is True
    assert manifest["hdm8_stats"]["runtime_fixed_chunk_lengths"] == [
        130887,
        2769,
        4397,
        31805,
    ]

    candidate_archive = Path(manifest["candidate_archive_path"])
    candidate_view = read_packed_archive_view(candidate_archive)
    assert candidate_view.payload_kind == "pr106_sidecar_wrapper"
    assert candidate_view.packed.decoder_packed_brotli.startswith(b"HDM8")
    assert candidate_view.packed.latents_and_sidecar_brotli == hdm7_view.packed.latents_and_sidecar_brotli
    restored = decode_hdm8_q_brotli_recipe_elided_fixture(candidate_view.packed.decoder_packed_brotli)
    assert restored.to_raw() == hdm7_raw


def test_hdm3_exact_eval_packet_readiness_clears_static_only_with_strict_inputs(
    tmp_path: Path,
) -> None:
    source_archive = _source_archive(tmp_path)
    out = tmp_path / "out"
    manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=out,
        source_label="PR106x frontier",
        repo_root=REPO,
    )
    proof_path = out / "runtime_adapter_proof.with_tool_run.json"
    proof_path.write_text(
        json.dumps(
            {
                "contract": "hnerv_hdm3_runtime_adapter_archive_parity_v1",
                "candidate_archive_sha256": manifest["candidate_archive_sha256"],
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_public_runtime_inflate": True,
                "inflate_output_parity_proven_by_payload_identity": True,
                "restored_payload_matches_source": True,
                "restored_decoder_section_matches_source": True,
                "latents_and_sidecar_match_source": True,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    compliance_path = out / "pre_submission_compliance.static.json"
    compliance_path.write_text(
        json.dumps(
            {
                "schema": "pre_submission_compliance_check_v1",
                "passed": True,
                "archive": {
                    "sha256": manifest["candidate_archive_sha256"],
                    "bytes": manifest["candidate_archive_bytes"],
                },
                "checks": [
                    {
                        "name": "archive_exists",
                        "passed": True,
                        "severity": "error",
                        "details": "synthetic static compliance closure",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = build_hdm3_exact_eval_packet_readiness(
        manifest,
        output_dir=out,
        repo_root=REPO,
        write=True,
    )

    assert readiness["score_claim"] is False
    assert readiness["dispatch_attempted"] is False
    assert readiness["ready_for_exact_eval_packet"] is True
    assert readiness["static_packet_ready"] is True
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["runtime_adapter_payload_identity"]["runtime_adapter_parity_proven"] is True
    runtime_contract = readiness["runtime_tree_inflate_output_parity"]
    assert runtime_contract["runtime_tree_closure_proven"] is True
    assert runtime_contract["inflate_output_parity_proven_by_payload_identity"] is True
    assert runtime_contract["exact_frame_output_parity_run"] is False
    assert runtime_contract["ready_for_exact_eval_dispatch"] is False
    assert len(runtime_contract["runtime_tree_sha256"]) == 64
    assert len(runtime_contract["modal_uploaded_submission_dir_runtime_tree_sha256"]) == 64
    assert (
        runtime_contract["modal_expected_runtime_tree_sha256"]
        == runtime_contract["modal_uploaded_submission_dir_runtime_tree_sha256"]
    )
    assert (
        runtime_contract["modal_uploaded_submission_dir_runtime_content_tree_sha256"]
        == runtime_contract["inflate_runtime_manifest"]["runtime_content_tree_sha256"]
    )
    assert runtime_contract["runtime_tree_manifest_source"].endswith(
        "contest_auth_eval.py::_runtime_dependency_manifest"
    )
    assert readiness["fixed_runtime_preflight"]["ready_for_fixed_runtime_exact_eval_readiness"] is True
    assert readiness["fixed_runtime_preflight"]["ready_for_exact_eval_dispatch"] is False
    assert readiness["fixed_runtime_preflight"]["runtime_tree_sha256"] == runtime_contract[
        "runtime_tree_sha256"
    ]
    assert readiness["fixed_runtime_preflight"]["modal_expected_runtime_tree_sha256"] == (
        runtime_contract["modal_expected_runtime_tree_sha256"]
    )
    assert readiness["exact_cuda_auth_eval"][
        "modal_uploaded_submission_dir_expected_runtime_tree_sha256"
    ] == runtime_contract["modal_expected_runtime_tree_sha256"]
    assert (
        "--expected-runtime-tree-sha256"
        in readiness["exact_cuda_auth_eval"]["modal_auth_eval_command_template"]
    )
    assert readiness["strict_static_compliance"]["passed"] is True
    assert readiness["static_blockers"] == []
    assert readiness["dispatch_blockers"] == [
        "lane_dispatch_claim_missing",
        "exact_cuda_auth_eval_missing",
    ]
    assert (out / "hdm3_exact_eval_packet_readiness.json").exists()
    assert (out / "hdm3_runtime_tree_closure.json").exists()


def test_hdm3_readiness_keeps_lossless_decoder_equivalence_separate_from_payload_identity(
    tmp_path: Path,
) -> None:
    source_archive = _source_archive(tmp_path)
    out = tmp_path / "out"
    manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=out,
        source_label="PR106x frontier",
        repo_root=REPO,
    )
    (out / "runtime_adapter_proof.json").write_text(
        json.dumps(
            {
                "contract": "hnerv_hdm3_runtime_adapter_archive_parity_v1",
                "candidate_archive_sha256": manifest["candidate_archive_sha256"],
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_public_runtime_inflate": True,
                "inflate_output_parity_proven_by_payload_identity": False,
                "inflate_output_parity_proven_by_lossless_decoder_equivalence": True,
                "submission_runtime_candidate_parse_claim": True,
                "submission_runtime_equivalence_claim": True,
                "restored_payload_matches_source": False,
                "restored_decoder_section_matches_source": False,
                "latents_and_sidecar_match_source": True,
                "full_frame_inflate_output_parity_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "pre_submission_compliance.static.json").write_text(
        json.dumps(
            {
                "schema": "pre_submission_compliance_check_v1",
                "passed": True,
                "archive": {
                    "sha256": manifest["candidate_archive_sha256"],
                    "bytes": manifest["candidate_archive_bytes"],
                },
                "checks": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = build_hdm3_exact_eval_packet_readiness(
        manifest,
        output_dir=out,
        repo_root=REPO,
        write=True,
    )

    runtime_evidence = readiness["runtime_adapter_payload_identity"]
    assert runtime_evidence["payload_identity_proven"] is False
    assert runtime_evidence["lossless_decoder_equivalence_proven"] is True
    assert runtime_evidence["runtime_equivalence_proven"] is True
    assert runtime_evidence["runtime_adapter_parity_proven"] is True
    runtime_contract = readiness["runtime_tree_inflate_output_parity"]
    assert runtime_contract["inflate_output_parity_proven_by_payload_identity"] is False
    assert runtime_contract["lossless_decoder_equivalence_proven"] is True
    assert runtime_contract["runtime_equivalence_proven"] is True
    assert runtime_contract["full_frame_inflate_output_parity_claim"] is False
    assert runtime_contract["runtime_tree_closure_proven"] is True
    assert readiness["ready_for_exact_eval_packet"] is True


def test_build_hdm3_archive_candidate_fails_closed_without_rate_win(tmp_path: Path) -> None:
    raw = _synthetic_context_decoder_raw()
    decoder_brotli = brotli.compress(raw, quality=11)
    latents = brotli.compress(b"latents", quality=11)
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(
        source_archive,
        member_name="x",
        payload=_packed_payload(decoder_brotli, latents),
    )

    manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="no win",
        repo_root=REPO,
    )

    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "hdm3_decoder_section_not_rate_positive" in manifest["archive_build_blockers"]
    assert manifest["candidate_archive_path"] == ""
    assert manifest["decoder_raw_equivalence"]["raw_equal"] is True


def test_build_hnerv_hdm3_archive_candidate_cli_writes_manifest(tmp_path: Path) -> None:
    source_archive = _source_archive(tmp_path)
    json_out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_hdm3_archive_candidate.py"),
            "--source-archive",
            str(source_archive),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-label",
            "PR106x",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["ready_for_archive_preflight"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["tool_run_manifest"]["tool"] == "tools/build_hnerv_hdm3_archive_candidate.py"
    assert Path(payload["candidate_archive_path"]).exists()


def _source_archive(tmp_path: Path) -> Path:
    raw = _synthetic_context_decoder_raw()
    decoder_brotli = brotli.compress(raw, quality=0)
    latents = brotli.compress(b"latents" * 100, quality=5)
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(
        source_archive,
        member_name="x",
        payload=_packed_payload(decoder_brotli, latents),
    )
    return source_archive


def _source_sidecar_archive(tmp_path: Path) -> Path:
    raw = _synthetic_context_decoder_raw()
    decoder_brotli = brotli.compress(raw, quality=0)
    latents = brotli.compress(b"latents" * 100, quality=5)
    inner = _packed_payload(decoder_brotli, latents)
    source_archive = tmp_path / "source_sidecar.zip"
    sidecar = emit_pr106_sidecar_packet(
        PR106SidecarPacket(
            format_id=PR106_SIDECAR_FORMAT_BROTLI,
            pr106_bytes=inner,
            sidecar_payload=brotli.compress(b"\x00\x00", quality=5),
        )
    )
    write_stored_single_member_zip(
        source_archive,
        member_name="0.bin",
        payload=sidecar,
    )
    return source_archive


def _packed_payload(decoder_brotli: bytes, latents_brotli: bytes) -> bytes:
    return bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents_brotli


def _synthetic_context_decoder_raw() -> bytes:
    q_parts = []
    scale_parts = []
    for index, (_name, shape) in enumerate(PACKED_STATE_SCHEMA):
        count = 1
        for dim in shape:
            count *= dim
        pattern = bytes(((index + i // 5) % 17) for i in range(64))
        repeats, remainder = divmod(count, len(pattern))
        q_parts.append(pattern * repeats + pattern[:remainder])
        scale_parts.append((index + 1).to_bytes(4, "little"))
    return b"".join(q_parts) + b"".join(scale_parts)
