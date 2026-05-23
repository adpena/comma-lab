# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli
import numpy as np

from tac.hnerv_lowlevel_packer import write_stored_single_member_zip
from tac.hnerv_pr103_lc_ac_schema import (
    Pr103LcAcLayout,
    encode_pr103_merged_ac_stream,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    BYTE_RANGE_CANDIDATE_MANIFEST_NAME,
    CHAIN_SCHEMA,
    PR103_CANDIDATE_MANIFEST_NAME,
    build_byte_range_entropy_recode_chain,
)
from tac.optimization.byte_range_entropy_recode_materializer import (
    CANDIDATE_SCHEMA,
    MATERIALIZER_ID,
    RECEIVER_CONTRACT_ID,
    RECEIVER_PROOF_SCHEMA,
    TARGET_KIND,
    build_byte_range_entropy_recode_plan,
    build_byte_range_entropy_recode_receiver_proof,
    materialize_byte_range_entropy_recode_candidate,
    verify_byte_range_entropy_recode_receiver_contract,
)
from tac.pr103_arithmetic_transform_plan import (
    CANDIDATE_SCHEMA as PR103_CANDIDATE_SCHEMA,
)
from tac.pr103_arithmetic_transform_plan import (
    build_pr103_arithmetic_histogram_beam_probe,
)
from tac.repo_io import sha256_bytes, sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]
PROOF_SCRIPT = REPO / "tools" / "build_byte_range_entropy_recode_receiver_proof.py"


def test_byte_range_entropy_recode_plan_requires_runtime_proof() -> None:
    plan = build_byte_range_entropy_recode_plan(
        archive_member_name="x",
        archive_byte_range={
            "archive_member_name": "x",
            "section_name": "ac_histograms_brotli",
            "candidate_start": 2,
            "candidate_end": 10,
        },
    )

    assert plan["target_kind"] == TARGET_KIND
    assert plan["materializer_id"] == MATERIALIZER_ID
    assert plan["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_missing" in plan["readiness_blockers"]
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False


def test_byte_range_entropy_recode_receiver_proof_accepts_strict_contract() -> None:
    proof = {
        "schema": RECEIVER_PROOF_SCHEMA,
        "ready_for_exact_eval_runtime": True,
        "archive_member_name": "x",
        "candidate_archive_sha256": "a" * 64,
        "candidate_member_sha256": "b" * 64,
        "archive_byte_ranges": [
            {
                "archive_member_name": "x",
                "section_name": "ac_histograms_brotli",
                "candidate_start": 2,
                "candidate_end": 10,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }

    verification = verify_byte_range_entropy_recode_receiver_contract(
        runtime_consumption_proof=proof,
        required_archive_member_name="x",
        required_candidate_archive_sha256="a" * 64,
        required_candidate_member_sha256="b" * 64,
    )

    assert verification["receiver_contract_id"] == RECEIVER_CONTRACT_ID
    assert verification["receiver_contract_satisfied"] is True
    assert verification["blockers"] == []
    assert verification["ready_for_exact_eval_dispatch"] is False


def test_pr103_backed_byte_range_entropy_materializer_emits_byte_closed_candidate(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)
    beam = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight0",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=2,
    )
    output_archive = tmp_path / "candidate.zip"

    report = materialize_byte_range_entropy_recode_candidate(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam,),
        output_archive=output_archive,
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert report["schema"] == CANDIDATE_SCHEMA
    assert report["materializer_id"] == MATERIALIZER_ID
    assert report["target_kind"] == TARGET_KIND
    assert output_archive.is_file()
    assert report["byte_closed_candidate_emitted"] is True
    assert report["candidate_archive"]["sha256"]
    assert report["archive_diff_manifest"]["candidate_non_noop"] is True
    assert report["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_missing" in report["readiness_blockers"]
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    assert "byte_range_entropy_recode_receiver_contract_not_satisfied" in report[
        "readiness_blockers"
    ]
    changed_sections = {row["section_name"] for row in report["archive_byte_ranges"]}
    assert "ac_histograms_brotli" in changed_sections
    assert all(
        row["candidate_end"] > row["candidate_start"]
        for row in report["archive_byte_ranges"]
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False

    receiver_proof = {
        "schema": RECEIVER_PROOF_SCHEMA,
        "ready_for_exact_eval_runtime": True,
        "archive_member_name": report["archive_member_name"],
        "candidate_archive_sha256": report["candidate_archive"]["sha256"],
        "candidate_member_sha256": report["candidate_archive"]["member_sha256"],
        "archive_byte_ranges": report["archive_byte_ranges"],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    proven_report = materialize_byte_range_entropy_recode_candidate(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam,),
        output_archive=tmp_path / "candidate_with_receiver_proof.zip",
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        runtime_consumption_proof=receiver_proof,
    )

    assert proven_report["receiver_contract_satisfied"] is True
    assert "candidate_runtime_adapter_missing" not in proven_report[
        "readiness_blockers"
    ]
    assert "byte_range_entropy_recode_receiver_contract_not_satisfied" not in proven_report[
        "readiness_blockers"
    ]
    assert "candidate_inflate_output_parity_missing" in proven_report[
        "readiness_blockers"
    ]


def test_byte_range_entropy_receiver_proof_transcodes_pr103_runtime_adapter(
    tmp_path: Path,
) -> None:
    fixture = _receiver_proof_fixture(tmp_path)

    proof = build_byte_range_entropy_recode_receiver_proof(
        runtime_adapter_manifest=fixture["runtime_adapter_manifest"],
        repo_root=tmp_path,
    )

    assert proof["schema"] == RECEIVER_PROOF_SCHEMA
    assert proof["ready_for_exact_eval_runtime"] is True
    assert proof["candidate_archive_sha256"] == sha256_file(fixture["candidate_archive"])
    assert proof["candidate_member_sha256"] == "b" * 64
    assert proof["archive_byte_ranges"] == [_expected_receiver_range()]
    verification = verify_byte_range_entropy_recode_receiver_contract(
        runtime_consumption_proof=proof,
        required_archive_member_name="x",
        required_candidate_archive_sha256=sha256_file(fixture["candidate_archive"]),
        required_candidate_member_sha256="b" * 64,
    )
    assert verification["receiver_contract_satisfied"] is True
    assert verification["blockers"] == []


def test_byte_range_entropy_receiver_proof_cli_writes_json(tmp_path: Path) -> None:
    fixture = _receiver_proof_fixture(tmp_path)
    json_out = tmp_path / "receiver_proof.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(PROOF_SCRIPT),
            "--runtime-adapter-manifest",
            str(fixture["runtime_adapter_manifest"]),
            "--json-out",
            str(json_out),
            "--fail-if-not-ready",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    proof = json.loads(json_out.read_text(encoding="utf-8"))
    assert proof["schema"] == RECEIVER_PROOF_SCHEMA
    assert proof["ready_for_exact_eval_runtime"] is True
    assert proof["tool_run_manifest"]["ready_for_exact_eval_dispatch"] is False
    assert proof["archive_byte_ranges"] == [_expected_receiver_range()]


def test_byte_range_entropy_chain_runs_materialize_adapter_proof_verify(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)
    beam = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight0",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=2,
    )
    runtime = _write_chain_runtime(tmp_path / "runtime", fixture["layout"])

    chain = build_byte_range_entropy_recode_chain(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam,),
        source_runtime_dir=runtime,
        output_dir=tmp_path / "chain",
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert chain["schema"] == CHAIN_SCHEMA
    assert chain["byte_closed_candidate_emitted"] is True
    assert chain["receiver_contract_satisfied"] is True
    assert chain["candidate_runtime_adapter_blocker_cleared"] is True
    assert "candidate_runtime_adapter_missing" not in chain["readiness_blockers"]
    assert "candidate_inflate_output_parity_missing" in chain["readiness_blockers"]
    assert "inflate_or_full_frame_parity" in chain["next_required_gates"]
    chain_dir = tmp_path / "chain"
    assert chain_dir.joinpath(BYTE_RANGE_CANDIDATE_MANIFEST_NAME).is_file()
    assert chain_dir.joinpath(PR103_CANDIDATE_MANIFEST_NAME).is_file()
    assert chain_dir.joinpath("byte_range_entropy_recode_chain_manifest.json").is_file()
    assert chain["artifacts"]["byte_range_candidate_manifest"]["sha256"]
    assert chain["artifacts"]["pr103_candidate_manifest"]["sha256"]
    assert chain["score_claim"] is False


def _receiver_proof_fixture(tmp_path: Path) -> dict[str, Path]:
    candidate_archive = tmp_path / "candidate.zip"
    write_stored_single_member_zip(candidate_archive, member_name="x", payload=b"abcdefgh")
    candidate_manifest = {
        "schema": PR103_CANDIDATE_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_archive": {
            "path": "candidate.zip",
            "bytes": candidate_archive.stat().st_size,
            "sha256": sha256_file(candidate_archive),
            "member_name": "x",
            "member_sha256": "b" * 64,
        },
        "section_diffs": [
            {
                "name": "ac_histograms_brotli",
                "source_start": 2,
                "source_end": 6,
                "source_bytes": 4,
                "source_sha256": "c" * 64,
                "candidate_start": 2,
                "candidate_end": 5,
                "candidate_bytes": 3,
                "candidate_sha256": "d" * 64,
                "byte_delta": -1,
                "changed": True,
            }
        ],
    }
    candidate_manifest_path = tmp_path / "candidate_manifest.json"
    write_json(candidate_manifest_path, candidate_manifest)
    runtime_adapter_manifest = {
        "schema": "pr103_lc_ac_runtime_adapter_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_manifest": {
            "path": "candidate_manifest.json",
            "bytes": candidate_manifest_path.stat().st_size,
            "sha256": sha256_file(candidate_manifest_path),
        },
        "candidate_archive": {
            "path": "candidate.zip",
            "bytes": candidate_archive.stat().st_size,
            "sha256": sha256_file(candidate_archive),
        },
        "runtime_consumption_probe": {"passed": True},
        "decoder_state_parity_proof": {"passed": True},
        "runtime_tree_sha256": "e" * 64,
        "readiness_blockers": ["full_frame_render_output_parity_missing"],
    }
    runtime_adapter_manifest_path = tmp_path / "runtime_adapter_manifest.json"
    write_json(runtime_adapter_manifest_path, runtime_adapter_manifest)
    return {
        "candidate_archive": candidate_archive,
        "candidate_manifest": candidate_manifest_path,
        "runtime_adapter_manifest": runtime_adapter_manifest_path,
    }


def _expected_receiver_range() -> dict[str, object]:
    return {
        "schema": "byte_range_entropy_recode_archive_range_v1",
        "archive_member_name": "x",
        "section_name": "ac_histograms_brotli",
        "source_start": 2,
        "source_end": 6,
        "source_bytes": 4,
        "source_sha256": "c" * 64,
        "candidate_start": 2,
        "candidate_end": 5,
        "candidate_bytes": 3,
        "candidate_sha256": "d" * 64,
        "byte_delta": -1,
    }


def _probe_fixture(tmp_path: Path) -> dict:
    histograms = np.ones((2, 256), dtype=np.uint8)
    histograms[0, :4] = np.asarray([2, 3, 5, 7], dtype=np.uint8)
    histograms[1, :4] = np.asarray([7, 5, 3, 2], dtype=np.uint8)
    hi_histogram = np.asarray([3, 1, 4], dtype="<u2")
    stream_specs = (
        ("fixture.weight0", 4, 0),
        ("fixture.weight1", 3, 1),
    )
    hi_symbol_count = 5
    symbol_streams = [
        np.asarray([0, 1, 2, 2], dtype=np.int32),
        np.asarray([0, 0, 1], dtype=np.int32),
        np.asarray([2, 0, 2, 1, 0], dtype=np.int32),
    ]
    merged_ac = encode_pr103_merged_ac_stream(
        symbol_streams,
        [histograms[0], histograms[1], hi_histogram],
    )
    scales = b"sc"
    non_ac = brotli.compress(b"non-ac-weights")
    hists = brotli.compress(histograms.tobytes())
    latent_meta = b"meta"
    low = brotli.compress(b"\x00" * 512)
    hi_hist = brotli.compress(hi_histogram.tobytes())
    layout = Pr103LcAcLayout(
        scales_fp16=len(scales),
        non_ac_weights_brotli=len(non_ac),
        ac_histograms_brotli=len(hists),
        merged_range_coded_weights_and_hi_latents=len(merged_ac),
        latent_min_scale_fp16=len(latent_meta),
        latent_low_bytes_brotli=len(low),
        latent_hi_histogram_brotli=len(hi_hist),
    )
    payload = scales + non_ac + hists + merged_ac + latent_meta + low + hi_hist
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    manifest = {
        "planning_only": True,
        "score_claim": False,
        "ready_for_schema_review": True,
        "source_archive": {
            "path": "source.zip",
            "bytes": archive.stat().st_size,
            "sha256": "a" * 64,
            "member_name": "x",
            "member_bytes": len(payload),
            "member_sha256": sha256_bytes(payload),
        },
        "merged_arithmetic_stream": {
            "source_bytes": len(merged_ac),
            "source_sha256": sha256_bytes(merged_ac),
            "decoded_symbol_count": 12,
            "decoder_maybe_exhausted": True,
            "reencoded_byte_identical": True,
        },
        "next_arithmetic_schema_targets": [
            {
                "label": "fixture.weight0",
                "role": "ac_weight_tensor",
                "schema_index": 0,
                "symbol_count": 4,
                "alphabet_size": 256,
                "decoded_symbols_sha256": sha256_bytes(
                    symbol_streams[0].astype(np.uint16).tobytes()
                ),
                "observed_entropy_bytes_floor": 1,
                "model_cross_entropy_bytes_floor": 2,
                "model_gap_bytes_estimate": 1,
            },
            {
                "label": "fixture.weight1",
                "role": "ac_weight_tensor",
                "schema_index": 1,
                "symbol_count": 3,
                "alphabet_size": 256,
                "decoded_symbols_sha256": sha256_bytes(
                    symbol_streams[1].astype(np.uint16).tobytes()
                ),
                "observed_entropy_bytes_floor": 1,
                "model_cross_entropy_bytes_floor": 2,
                "model_gap_bytes_estimate": 1,
            },
        ],
    }
    return {
        "archive": archive,
        "manifest": manifest,
        "layout": layout,
        "stream_specs": stream_specs,
        "hi_symbol_count": hi_symbol_count,
    }


def _write_chain_runtime(runtime: Path, layout: Pr103LcAcLayout) -> Path:
    runtime.mkdir()
    runtime.joinpath("inflate.py").write_text(
        f"""
from __future__ import annotations

import brotli
import numpy as np

SCA_LEN = {layout.scales_fp16}
BR_LEN = {layout.non_ac_weights_brotli}
HIST_LEN = {layout.ac_histograms_brotli}
MERGED_AC_LEN = {layout.merged_range_coded_weights_and_hi_latents}
LATENT_META_LEN = {layout.latent_min_scale_fp16}
LO_LEN = {layout.latent_low_bytes_brotli}
HI_HIST_LEN = {layout.latent_hi_histogram_brotli}


def parse_archive(blob):
    o = 0
    sca = blob[o:o + SCA_LEN]; o += SCA_LEN
    br = blob[o:o + BR_LEN]; o += BR_LEN
    hists_b = blob[o:o + HIST_LEN]; o += HIST_LEN
    merged_ac = blob[o:o + MERGED_AC_LEN]; o += MERGED_AC_LEN
    mins_scales = blob[o:o + LATENT_META_LEN]; o += LATENT_META_LEN
    lo_b = blob[o:o + LO_LEN]; o += LO_LEN
    hi_hist_b = blob[o:o + HI_HIST_LEN]; o += HI_HIST_LEN
    wrp_b = blob[o:]
    return sca, br, hists_b, merged_ac, mins_scales, lo_b, hi_hist_b, wrp_b


def build_state_dict(br_b, hists_b, merged_ac, sca, hi_hist):
    return {{"w": np.zeros(4, dtype=np.float32)}}, np.asarray([0], dtype=np.uint16)


def decode_latents(mins_scales, lo_b, hi_decoded):
    return np.zeros((1, 1), dtype=np.float32)


def apply_corrections(latents, wrp_b):
    return latents
""".lstrip(),
        encoding="utf-8",
    )
    runtime.joinpath("inflate.sh").write_text(
        '#!/usr/bin/env bash\nHERE="$(cd "$(dirname "$0")" && pwd)"\nSRC="$1"\nDST="$2"\npython "$HERE/inflate.py" "$SRC" "$DST"\n',
        encoding="utf-8",
    )
    return runtime
