# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.nerv_decoder_weight_waterfill import (
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
    NervDecoderWeightWaterfillError,
    build_nerv_decoder_weight_waterfill_plan,
    load_saliency_json,
    load_state_npz_from_manifest,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    NERV_BYTE_PRICE_CONTROLLER_SCHEMA,
)
from tools.build_nerv_decoder_weight_waterfill_plan import main as tool_main


def test_decoder_weight_waterfill_protects_when_saliency_missing() -> None:
    report = build_nerv_decoder_weight_waterfill_plan(
        {"blocks.0.weight": np.asarray([0.125, -0.75, 1.0], dtype=np.float32)},
        action_bits=(0, 2, 32),
        zero_run_overhead_bytes=0,
        full_video_coverage=True,
        receiver_proof_status="runtime_consumption_proof_ready",
        archive_sha256="a" * 64,
    )

    assert report["schema"] == NERV_DECODER_WEIGHT_WATERFILL_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["group_count"] == 1
    assert report["rows"][0]["selected_action"] == "fp32_protect"
    assert report["rows"][0]["selected_byte_delta"] == 0
    assert report["total_selected_byte_delta"] == 0
    assert "decoder_weight_saliency_missing_for_some_groups" in report["blockers"]
    assert "decoder_weight_group_saliency_missing" in report["rows"][0]["blockers"]
    assert report["byte_price_plan"]["schema"] == NERV_BYTE_PRICE_CONTROLLER_SCHEMA


def test_decoder_weight_waterfill_uses_measured_saliency_for_actions() -> None:
    state = {"blocks.0.weight": np.asarray([0.125, -0.75, 1.0], dtype=np.float32)}
    common = {
        "state_dict": state,
        "action_bits": (0, 2, 32),
        "zero_run_overhead_bytes": 0,
        "full_video_coverage": True,
        "receiver_proof_status": "runtime_consumption_proof_ready",
        "archive_sha256": "b" * 64,
    }

    cheap = build_nerv_decoder_weight_waterfill_plan(
        saliency_by_name={"blocks.0.weight": 0.0},
        **common,
    )
    protected = build_nerv_decoder_weight_waterfill_plan(
        saliency_by_name={"blocks.0.weight": 1_000_000.0},
        **common,
    )

    assert cheap["rows"][0]["selected_action"] == "zero_rle"
    assert cheap["rows"][0]["selected_delta_total_score_proxy"] < 0.0
    assert protected["rows"][0]["selected_action"] == "fp32_protect"
    assert protected["rows"][0]["selected_delta_total_score_proxy"] == 0.0


def test_decoder_weight_waterfill_loads_zero_saliency_rows(tmp_path: Path) -> None:
    saliency_path = tmp_path / "saliency.json"
    saliency_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "group_name": "blocks.0.weight",
                        "saliency": 0.0,
                        "decoder_weight_saliency": 123.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_saliency_json(saliency_path) == {"blocks.0.weight": 0.0}


def test_decoder_weight_waterfill_loads_state_from_verified_npz_manifest(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.npz"
    np.savez(state_path, **{"blocks.0.weight": np.asarray([1.0], dtype=np.float32)})
    digest = sha256(state_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "state_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "framework_agnostic_npz_bridge_manifest.v1",
                "artifact_path": state_path.name,
                "artifact_sha256": digest,
                "consumption_recommended": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_state_npz_from_manifest(manifest_path)

    assert set(loaded) == {"blocks.0.weight"}
    assert loaded["blocks.0.weight"].tolist() == [1.0]


def test_decoder_weight_waterfill_rejects_stale_npz_manifest_sha(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.npz"
    np.savez(state_path, **{"blocks.0.weight": np.asarray([1.0], dtype=np.float32)})
    manifest_path = tmp_path / "state_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "framework_agnostic_npz_bridge_manifest.v1",
                "artifact_path": state_path.name,
                "artifact_sha256": "0" * 64,
                "consumption_recommended": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NervDecoderWeightWaterfillError, match="sha256 mismatch"):
        load_state_npz_from_manifest(manifest_path)


def test_build_nerv_decoder_weight_waterfill_plan_cli_smoke(tmp_path: Path) -> None:
    state_path = tmp_path / "state.npz"
    saliency_path = tmp_path / "saliency.json"
    output_json = tmp_path / "waterfill.json"
    output_md = tmp_path / "waterfill.md"
    np.savez(
        state_path,
        **{
            "blocks.0.weight": np.asarray([0.125, -0.75, 1.0], dtype=np.float32),
            "latents.0": np.asarray([99.0], dtype=np.float32),
        },
    )
    saliency_path.write_text(
        json.dumps({"blocks.0.weight": 0.0}),
        encoding="utf-8",
    )

    rc = tool_main(
        [
            "--state-npz",
            str(state_path),
            "--saliency-json",
            str(saliency_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--action-bits",
            "0,2,32",
            "--full-video-coverage",
            "--receiver-proof-status",
            "runtime_consumption_proof_ready",
            "--archive-sha256",
            "c" * 64,
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["group_count"] == 1
    assert payload["rows"][0]["group_name"] == "blocks.0.weight"
    assert payload["rows"][0]["selected_action"] in {"int2", "zero_rle"}
    assert payload["rows"][0]["selected_byte_delta"] < 0
    assert payload["rows"][0]["selected_delta_total_score_proxy"] < 0.0
    assert "latents.0" not in output_md.read_text(encoding="utf-8")


def test_build_nerv_decoder_weight_waterfill_plan_cli_accepts_npz_manifest(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.npz"
    saliency_path = tmp_path / "saliency.json"
    manifest_path = tmp_path / "state_manifest.json"
    output_json = tmp_path / "waterfill.json"
    np.savez(
        state_path,
        **{"blocks.0.weight": np.asarray([0.125, -0.75, 1.0], dtype=np.float32)},
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "framework_agnostic_npz_bridge_manifest.v1",
                "artifact_path": state_path.as_posix(),
                "artifact_sha256": sha256(state_path.read_bytes()).hexdigest(),
                "consumption_recommended": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    saliency_path.write_text(json.dumps({"blocks.0.weight": 0.0}), encoding="utf-8")

    rc = tool_main(
        [
            "--state-npz-manifest",
            str(manifest_path),
            "--saliency-json",
            str(saliency_path),
            "--output-json",
            str(output_json),
            "--action-bits",
            "0,2,32",
            "--full-video-coverage",
            "--receiver-proof-status",
            "runtime_consumption_proof_ready",
            "--archive-sha256",
            "d" * 64,
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["group_count"] == 1
    assert payload["rows"][0]["selected_byte_delta"] < 0
