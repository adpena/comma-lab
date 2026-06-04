# SPDX-License-Identifier: MIT
"""Tests for the NeRV model-size budget builder CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.build_nerv_modelsize_budget as tool
from tac.repo_io import ArtifactWriteError, sha256_file


def test_build_nerv_modelsize_budget_tool_writes_both_family_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"
    md = tmp_path / "budget.md"

    rc = tool.main(
        [
            "--output-hinerv-json",
            str(hinerv),
            "--output-snerv-json",
            str(snerv),
            "--output-md",
            str(md),
            "--hard-byte-ceiling",
            "36000",
            "--num-pairs",
            "17",
            "--per-ceiling-limit",
            "2",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == "nerv_modelsize_budget_build.v1"
    assert summary["inputs"] == {
        "hard_byte_ceilings": [36000],
        "num_pairs": 17,
        "per_ceiling_limit": 2,
        "target_modelsize_mparams": [],
        "hinerv_target_modelsize_mparams": [],
        "hinerv_official_controls_only": True,
        "snerv_emb_sizes": [0],
        "snerv_fc_dims": [9],
        "snerv_model_size_adapter": "snerv_fc_dim_emb_size_adapter_v1",
        "snerv_modelsize_control_profile": {
            "blockers": [],
            "dec_strds": [5, 4, 2, 2, 2],
            "enc_strds": [5, 4, 2, 2, 2],
            "modelsize_solve_supported": True,
            "profile_id": "contest_receiver_profile",
            "source_family": "PACT SNeRV receiver adapter",
            "source_notes": (
                "Not an upstream README default; selected for PACT contest receiver "
                "custody with explicit measured-archive authority required."
            ),
            "source": "pact_receiver_closed_snar1_profile",
        },
        "snerv_modelsize_control_profile_id": "contest_receiver_profile",
        "snerv_official_dec_strds": [5, 4, 2, 2, 2],
        "snerv_official_enc_strds": [5, 4, 2, 2, 2],
        "snerv_official_modelsize_mparams": [],
        "snerv_temporal_context": 0,
        "snerv_temporal_modes": ["delta"],
        "snerv_official_skip_high_modes": ["full"],
    }
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["snerv_invalid_candidate_count"] == 0
    assert hinerv.is_file()
    assert snerv.is_file()
    assert md.is_file()

    hinerv_payload = json.loads(hinerv.read_text(encoding="utf-8"))
    snerv_payload = json.loads(snerv.read_text(encoding="utf-8"))
    assert hinerv_payload["schema"] == "nerv_modelsize_budget.v1"
    assert snerv_payload["schema"] == "snerv_modelsize_budget.v1"
    assert hinerv_payload["family"] == "hi_nerv"
    assert hinerv_payload["official_controls_only"] is True
    assert all(
        row["use_hierarchical_feature_grid"] is True
        and row["use_convnext_blocks"] is True
        for row in hinerv_payload["selected_candidates"]
    )
    assert snerv_payload["family"] == "snerv"
    assert snerv_payload["modelsize_control_profile_id"] == "contest_receiver_profile"
    assert snerv_payload["selected_candidates"][0]["candidate_id"].startswith(
        "snerv_np17_"
    )
    assert "_mfu" in snerv_payload["selected_candidates"][0]["candidate_id"]


def test_build_nerv_modelsize_budget_tool_exposes_hinerv_target_modelsize(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"

    rc = tool.main(
        [
            "--output-hinerv-json",
            str(hinerv),
            "--output-snerv-json",
            str(snerv),
            "--hard-byte-ceiling",
            "2000000000",
            "--num-pairs",
            "17",
            "--per-ceiling-limit",
            "4",
            "--hinerv-target-modelsize-mparams",
            "0.03",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["inputs"]["hinerv_target_modelsize_mparams"] == [0.03]
    payload = json.loads(hinerv.read_text(encoding="utf-8"))
    assert payload["target_modelsize_mparams"] == [0.03]
    selected = payload["selected_candidates"]
    assert selected
    assert any(
        row["capacity_source"] == "local_hinerv_target_modelsize"
        and row["target_modelsize_mparams"] == 0.03
        and row["candidate_id"].endswith("_tgtmp0p03")
        for row in selected
    )
    assert all(row["score_claim"] is False for row in selected)


def test_build_nerv_modelsize_budget_tool_exposes_shared_target_modelsize(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"

    rc = tool.main(
        [
            "--output-hinerv-json",
            str(hinerv),
            "--output-snerv-json",
            str(snerv),
            "--hard-byte-ceiling",
            "2000000000",
            "--num-pairs",
            "17",
            "--per-ceiling-limit",
            "4",
            "--target-modelsize-mparams",
            "0.03",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["inputs"]["target_modelsize_mparams"] == [0.03]
    assert summary["inputs"]["hinerv_target_modelsize_mparams"] == [0.03]
    assert summary["inputs"]["snerv_official_modelsize_mparams"] == [0.03]

    hinerv_payload = json.loads(hinerv.read_text(encoding="utf-8"))
    snerv_payload = json.loads(snerv.read_text(encoding="utf-8"))
    assert any(
        row["capacity_source"] == "local_hinerv_target_modelsize"
        and row["target_modelsize_mparams"] == 0.03
        and row["candidate_id"].endswith("_tgtmp0p03")
        for row in hinerv_payload["selected_candidates"]
    )
    assert any(
        row["capacity_source"] == "official_snerv_modelsize"
        and row["modelsize_mparams"] == 0.03
        and row["official_modelsize_solution"] is not None
        for row in snerv_payload["selected_candidates"]
    )


def test_build_nerv_modelsize_budget_tool_exposes_official_snerv_modelsize(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"

    rc = tool.main(
        [
            "--output-hinerv-json",
            str(hinerv),
            "--output-snerv-json",
            str(snerv),
            "--hard-byte-ceiling",
            "178000",
            "--num-pairs",
            "600",
            "--per-ceiling-limit",
            "4",
            "--snerv-fc-dim",
            "11",
            "--snerv-emb-size",
            "0",
            "--snerv-emb-size",
            "2",
            "--snerv-official-modelsize-mparams",
            "0.05",
            "--output-md",
            str(tmp_path / "budget.md"),
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["inputs"]["snerv_fc_dims"] == [11]
    assert summary["inputs"]["snerv_emb_sizes"] == [0, 2]
    assert summary["inputs"]["snerv_official_modelsize_mparams"] == [0.05]
    assert summary["snerv_invalid_candidate_count"] > 0
    payload = json.loads(snerv.read_text(encoding="utf-8"))
    assert payload["invalid_candidate_count"] == summary["snerv_invalid_candidate_count"]
    assert payload["invalid_candidates"][0]["score_claim"] is False
    assert payload["invalid_candidates"][0]["ready_for_exact_eval_dispatch"] is False
    md = (tmp_path / "budget.md").read_text(encoding="utf-8")
    assert "Skipped SNeRV Official Controls" in md
    selected = payload["selected_candidates"]
    assert selected
    assert any(row["modelsize_mparams"] == 0.05 for row in selected)
    assert any(
        row["official_modelsize_solution"] is not None
        and row["capacity_source"] == "official_snerv_modelsize"
        for row in selected
        if row["candidate_id"].find("_fc11e0_") >= 0
    )


def test_build_nerv_modelsize_budget_tool_enumerates_official_skip_ladder_by_default(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"

    rc = tool.main(
        [
            "--output-hinerv-json",
            str(hinerv),
            "--output-snerv-json",
            str(snerv),
            "--hard-byte-ceiling",
            "178000",
            "--num-pairs",
            "600",
            "--per-ceiling-limit",
            "8",
            "--snerv-model-size-adapter",
            "snerv_official_mfu_hfr_tub_primitives_adapter",
            "--snerv-official-modelsize-mparams",
            "0.05",
            "--snerv-temporal-mode",
            "official_haar_dwt1d_lowpass",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["inputs"]["snerv_official_skip_high_modes"] == [
        "full",
        "shared_mean",
        "channel_mean",
        "scalar_mean",
    ]
    payload = json.loads(snerv.read_text(encoding="utf-8"))
    selected_modes = {
        row["official_skip_high_mode"] for row in payload["selected_candidates"]
    }
    assert {"channel_mean", "scalar_mean"} <= selected_modes
    full_rows = [
        row
        for row in payload["selected_candidates"]
        if row["official_skip_high_mode"] == "full"
    ]
    assert full_rows
    assert all(row["nominal_under_ceiling"] is False for row in full_rows)
    assert all(
        row["nominal_under_ceiling"] is True
        for row in payload["selected_candidates"]
        if row["official_skip_high_mode"] in {"channel_mean", "scalar_mean"}
    )


def test_build_nerv_modelsize_budget_tool_exposes_snerv_temporal_modes(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"

    rc = tool.main(
        [
            "--output-hinerv-json",
            str(hinerv),
            "--output-snerv-json",
            str(snerv),
            "--hard-byte-ceiling",
            "178000",
            "--num-pairs",
            "17",
            "--per-ceiling-limit",
            "8",
            "--snerv-temporal-context",
            "1",
            "--snerv-temporal-mode",
            "delta",
            "--snerv-temporal-mode",
            "official_haar_dwt1d_lowpass",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["inputs"]["snerv_temporal_context"] == 1
    assert summary["inputs"]["snerv_temporal_modes"] == [
        "delta",
        "official_haar_dwt1d_lowpass",
    ]
    payload = json.loads(snerv.read_text(encoding="utf-8"))
    assert payload["temporal_context"] == 1
    assert payload["temporal_modes"] == ["delta", "official_haar_dwt1d_lowpass"]
    assert any(
        row["temporal_mode"] == "official_haar_dwt1d_lowpass"
        and "_tmhaar1_" in row["candidate_id"]
        for row in payload["selected_candidates"]
    )


def test_build_nerv_modelsize_budget_tool_exposes_official_adapter_skip_high_modes(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"

    rc = tool.main(
        [
            "--output-hinerv-json",
            str(hinerv),
            "--output-snerv-json",
            str(snerv),
            "--hard-byte-ceiling",
            "2000000000",
            "--num-pairs",
            "600",
            "--per-ceiling-limit",
            "32",
            "--snerv-model-size-adapter",
            "snerv_official_mfu_hfr_tub_primitives_adapter",
            "--snerv-official-modelsize-mparams",
            "0.05",
            "--snerv-temporal-mode",
            "official_haar_dwt1d_lowpass",
            "--snerv-official-skip-high-mode",
            "full",
            "--snerv-official-skip-high-mode",
            "shared_mean",
            "--snerv-official-skip-high-mode",
            "channel_mean",
            "--snerv-official-skip-high-mode",
            "scalar_mean",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["inputs"]["snerv_model_size_adapter"] == (
        "snerv_official_mfu_hfr_tub_numeric_primitives_v1"
    )
    assert summary["inputs"]["snerv_official_skip_high_modes"] == [
        "full",
        "shared_mean",
        "channel_mean",
        "scalar_mean",
    ]
    payload = json.loads(snerv.read_text(encoding="utf-8"))
    assert payload["official_skip_high_modes"] == [
        "full",
        "shared_mean",
        "channel_mean",
        "scalar_mean",
    ]
    rows = payload["selected_candidates"]
    official_rows = [
        row
        for row in rows
        if row["snerv_model_size_adapter"]
        == "snerv_official_mfu_hfr_tub_numeric_primitives_v1"
    ]
    assert official_rows
    assert any("_adofficial_" in row["candidate_id"] for row in official_rows)
    assert all(row["lf_coeff_count_total"] == 1 for row in official_rows)
    assert all("nominal_skip_high_payload_bytes" in row for row in official_rows)
    assert all(row["nominal_lf_payload_bytes"] < 512 for row in official_rows)
    assert all(row["nominal_step_map_payload_bytes"] < 512 for row in official_rows)
    assert all(row["nominal_metadata_payload_bytes"] < 64 for row in official_rows)


def test_build_nerv_modelsize_budget_tool_requires_expected_hashes_for_overwrite(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"
    md = tmp_path / "budget.md"
    argv = [
        "--output-hinerv-json",
        str(hinerv),
        "--output-snerv-json",
        str(snerv),
        "--output-md",
        str(md),
        "--hard-byte-ceiling",
        "36000",
        "--num-pairs",
        "17",
        "--per-ceiling-limit",
        "2",
    ]

    assert tool.main(argv) == 0

    with pytest.raises(ArtifactWriteError, match="expected_existing_sha256"):
        tool.main([*argv, "--allow-overwrite"])


def test_build_nerv_modelsize_budget_tool_overwrites_with_explicit_hash_custody(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"
    md = tmp_path / "budget.md"
    argv = [
        "--output-hinerv-json",
        str(hinerv),
        "--output-snerv-json",
        str(snerv),
        "--output-md",
        str(md),
        "--hard-byte-ceiling",
        "36000",
        "--num-pairs",
        "17",
        "--per-ceiling-limit",
        "2",
    ]

    assert tool.main(argv) == 0
    capsys.readouterr()
    assert tool.main(
        [
            *argv,
            "--allow-overwrite",
            "--expected-output-hinerv-json-sha256",
            sha256_file(hinerv),
            "--expected-output-snerv-json-sha256",
            sha256_file(snerv),
            "--expected-output-md-sha256",
            sha256_file(md),
        ]
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == "nerv_modelsize_budget_build.v1"
    assert summary["promotion_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    md_text = md.read_text(encoding="utf-8")
    assert "payload=`None`" not in md_text
    assert "nominal_under_ceiling=`" in md_text


def test_build_nerv_modelsize_budget_tool_can_rerun_with_guarded_overwrite(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"
    md = tmp_path / "budget.md"
    args = [
        "--output-hinerv-json",
        str(hinerv),
        "--output-snerv-json",
        str(snerv),
        "--output-md",
        str(md),
        "--hard-byte-ceiling",
        "36000",
        "--num-pairs",
        "17",
        "--per-ceiling-limit",
        "1",
    ]

    assert tool.main(args) == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert tool.main(
        [
            *args,
            "--allow-overwrite",
            "--expected-output-hinerv-json-sha256",
            sha256_file(hinerv),
            "--expected-output-snerv-json-sha256",
            sha256_file(snerv),
            "--expected-output-md-sha256",
            sha256_file(md),
        ]
    ) == 0
    second_summary = json.loads(capsys.readouterr().out)

    assert second_summary["hinerv_output_sha256"] == first_summary["hinerv_output_sha256"]
    assert second_summary["snerv_output_sha256"] == first_summary["snerv_output_sha256"]
    assert second_summary["output_md_sha256"] == first_summary["output_md_sha256"]
