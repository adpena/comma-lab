# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.nerv_decoder_weight_waterfill import (
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
)
from tac.analysis.snerv_trained_ladder_waterfill import (
    SNERV_TRAINED_LADDER_WATERFILL_SCHEMA,
    SnervTrainedLadderWaterfillError,
    build_snerv_trained_ladder_waterfill,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    encode_decoder_payload,
    pack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    HfGenerationDecoder,
    SnervModelSizeConfig,
)
from tools.build_snerv_trained_ladder_waterfill import main as tool_main


def test_snerv_trained_ladder_waterfill_decodes_zip_member_and_plans(
    tmp_path: Path,
) -> None:
    archive_zip = _snerv_archive_zip(tmp_path, "candidate.zip")
    payload = _trained_row_payload(archive_zip, n_pairs=600, receiver_passed=True)

    report = build_snerv_trained_ladder_waterfill(
        payload,
        saliency_by_name={"decoder.level0": 0.0},
        action_bits=(0, 2, 32),
        candidate_id="candidate_a",
    )

    assert report["schema"] == SNERV_TRAINED_LADDER_WATERFILL_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["row_count"] == 1
    row = report["rows"][0]
    assert row["waterfill_plan_schema"] == NERV_DECODER_WEIGHT_WATERFILL_SCHEMA
    assert row["waterfill_summary"]["group_count"] == 3
    assert row["waterfill_summary"]["total_selected_byte_delta"] < 0
    assert row["decoder_state_group_names"] == [
        "decoder.level0.HH.kernel",
        "decoder.level0.HL.kernel",
        "decoder.level0.LH.kernel",
    ]
    assert report["section_value_rows"][0]["trained_ladder_row_id"] == row["row_id"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]


def test_snerv_trained_ladder_waterfill_preserves_local_only_blockers(
    tmp_path: Path,
) -> None:
    archive_zip = _snerv_archive_zip(tmp_path, "local.zip")
    payload = _trained_row_payload(archive_zip, n_pairs=1, receiver_passed=False)
    payload["blockers"] = ["sample_pair_count_below_full600"]
    payload["rows"][0]["emission_blockers"] = ["sample_pair_count_below_full600"]

    report = build_snerv_trained_ladder_waterfill(payload)

    row = report["rows"][0]
    assert row["waterfill_summary"]["total_selected_byte_delta"] == 0
    assert "sample_pair_count_below_full600" in row["blockers"]
    assert "decoder_weight_saliency_missing_for_some_groups" in row["blockers"]
    assert "full_video_coverage_missing" in row["blockers"]
    assert "receiver_proof_not_satisfied" in row["blockers"]
    assert row["waterfill_plan"]["rows"][0]["selected_action"] == "fp32_protect"


def test_snerv_trained_ladder_waterfill_blocks_zip_without_0_bin(
    tmp_path: Path,
) -> None:
    archive_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("not_0.bin", b"SNAR1-nope")
    payload = _trained_row_payload(archive_zip, n_pairs=600, receiver_passed=True)

    report = build_snerv_trained_ladder_waterfill(payload)

    assert report["rows"][0]["waterfill_plan"] is None
    assert "contest_archive_zip_missing_0_bin" in report["rows"][0]["blockers"]
    assert "contest_archive_zip_missing_0_bin" in report["blockers"]


def test_snerv_trained_ladder_waterfill_rejects_non_snerv_family(
    tmp_path: Path,
) -> None:
    archive_zip = _snerv_archive_zip(tmp_path, "candidate.zip")
    payload = _trained_row_payload(archive_zip, n_pairs=600, receiver_passed=True)
    payload["family"] = "hinerv"

    with pytest.raises(SnervTrainedLadderWaterfillError, match="family must be snerv"):
        build_snerv_trained_ladder_waterfill(payload)


def test_build_snerv_trained_ladder_waterfill_cli_smoke(tmp_path: Path) -> None:
    archive_zip = _snerv_archive_zip(tmp_path, "candidate.zip")
    payload_path = tmp_path / "trained_row.json"
    saliency_path = tmp_path / "saliency.json"
    output_json = tmp_path / "waterfill.json"
    output_md = tmp_path / "waterfill.md"
    payload_path.write_text(
        json.dumps(_trained_row_payload(archive_zip, n_pairs=600, receiver_passed=True)),
        encoding="utf-8",
    )
    saliency_path.write_text(
        json.dumps({"decoder.level0": 0.0}),
        encoding="utf-8",
    )

    rc = tool_main(
        [
            "--trained-ladder-row-json",
            str(payload_path),
            "--saliency-json",
            str(saliency_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--action-bits",
            "0,2,32",
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    assert payload["section_value_rows"][0]["trained_ladder_row_id"] == "snerv_row"
    assert "SNeRV trained ladder decoder waterfill" in output_md.read_text(
        encoding="utf-8"
    )


def _snerv_archive_zip(tmp_path: Path, name: str) -> Path:
    model_size = SnervModelSizeConfig(fc_dim=3, emb_size=0, patch_radius=0)
    decoder = HfGenerationDecoder(
        levels=1,
        model_size=model_size,
        kernels={
            0: {
                "LH": np.asarray([0.25, -0.5, 1.0], dtype=np.float64),
                "HL": np.asarray([0.125, 0.0, -0.25], dtype=np.float64),
                "HH": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            }
        },
    )
    packet = pack_snerv_archive(
        metadata_payload=np.asarray([0.0], dtype="<f4").tobytes(),
        lf_payload=b"lf",
        decoder_payload=encode_decoder_payload(decoder, codec="float32_lzma"),
        step_map_packet=b"steps",
        metadata={"n_pairs": 600, "levels": 1, "carrier_hw": [1, 1], "wavelet": "haar"},
    ).packet
    archive_zip = tmp_path / name
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", packet)
        zf.writestr("inflate.sh", "#!/bin/sh\n")
        zf.writestr("inflate.py", "print('ok')\n")
    return archive_zip


def _trained_row_payload(
    archive_zip: Path,
    *,
    n_pairs: int,
    receiver_passed: bool,
) -> dict[str, object]:
    digest = sha256(archive_zip.read_bytes()).hexdigest()
    return {
        "schema": "nerv_trained_ladder_row_payload.v1",
        "family": "snerv",
        "status": "trained_ladder_row_ready" if receiver_passed else "trained_ladder_row_blocked",
        "verdict": "NO_GO_SCORE_OR_EXACT_AUTH",
        "axis_tag": "[planning/control:false-authority]",
        "blockers": [],
        "rows": [
            {
                "row_id": "snerv_row",
                "family": "snerv",
                "n_pairs": n_pairs,
                "archive_path": archive_zip.as_posix(),
                "archive_bytes": archive_zip.stat().st_size,
                "archive_sha256": digest,
                "receiver_archive_replay_verified": True,
                "receiver_proof_passed": receiver_passed,
                "receiver_codec_mode": "contest_archive_zip",
                "decoder_precision_mode": "float32_lzma",
                "emission_blockers": [],
            }
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
