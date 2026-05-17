# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.tt5l_sideinfo_variant_packets import (
    TT5L_CONTEST_NUM_PAIRS,
    build_tt5l_sideinfo_variant_packets,
    read_tt5l_archive_zip,
    tt5l_sideinfo_variant_arrays,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    pack_archive,
    parse_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _toy_state_dict() -> dict[str, torch.Tensor]:
    return {
        "renderer.hidden.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "renderer.hidden.0.bias": torch.zeros(3),
        "renderer.output_layer.weight": torch.ones(2, 3),
        "renderer.output_layer.bias": torch.zeros(2),
    }


def _write_tt5l_archive_zip(path: Path, sideinfo: np.ndarray) -> None:
    blob = pack_archive(
        world_model_state_dict=_toy_state_dict(),
        per_pair_side_info=sideinfo,
        meta={"int8_scale": 64.0, "fixture": True},
        num_pairs=int(sideinfo.shape[0]),
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=384,
        output_width=512,
        foveation_grid_h=4,
        foveation_grid_w=4,
        pose_dim=6,
        per_pair_bytes=int(sideinfo.shape[1]),
        ac_state=b"fixture-ac-state",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr(info, blob)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tt5l_sideinfo_variant_arrays_are_deterministic_and_meaningful() -> None:
    source = np.zeros((4, 45), dtype=np.int8)
    source[:, 0] = np.arange(1, 5, dtype=np.int8)
    source[:, 36] = 7

    first = tt5l_sideinfo_variant_arrays(source, seed=123)
    second = tt5l_sideinfo_variant_arrays(source, seed=123)

    assert tuple(first) == L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        assert np.array_equal(first[variant], second[variant])
        assert first[variant].dtype == np.int8
        assert first[variant].shape == source.shape
    assert np.count_nonzero(first["zero"]) == 0
    assert np.count_nonzero(first["random_lsb"]) == source.size
    assert np.array_equal(first["trained"], source)
    assert not np.array_equal(first["shuffled"], source)
    assert np.count_nonzero(first["ablated"][:, 36:45]) == 0
    assert np.array_equal(first["ablated"][:, :36], source[:, :36])


def test_build_tt5l_sideinfo_variant_packets_writes_byte_closed_archives(
    tmp_path: Path,
) -> None:
    source_sideinfo = np.zeros((3, 45), dtype=np.int8)
    source_sideinfo[0, 0] = 1
    source_sideinfo[1, 12] = -2
    source_sideinfo[2, 36] = 3
    source_archive = tmp_path / "source" / "archive.zip"
    _write_tt5l_archive_zip(source_archive, source_sideinfo)

    manifest = build_tt5l_sideinfo_variant_packets(
        source_archive=source_archive,
        output_root=tmp_path / "variants",
        repo_root=tmp_path,
        seed=99,
    )

    assert manifest["schema"] == "tt5l_sideinfo_variant_packets_v1"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["variant_count"] == 5
    assert "tt5l_source_trained_sideinfo_all_zero" not in manifest["blockers"]
    assert (
        f"tt5l_source_num_pairs_not_full_contest:3_expected_{TT5L_CONTEST_NUM_PAIRS}"
        in manifest["blockers"]
    )
    rows = {row["variant"]: row for row in manifest["variants"]}
    assert tuple(rows) == L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS

    for variant, row in rows.items():
        archive_path = tmp_path / row["archive_path"]
        assert archive_path.exists()
        assert row["archive_sha256"] == _sha_file(archive_path)
        assert row["score_claim"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["dispatch_attempted"] is False
        assert row["parsed_sections"]["per_pair_side_info_blob"]["length"] > 0
        assert row["non_target_sections_identical_to_source"] == {
            "world_model_blob": True,
            "ac_state_blob": True,
            "meta_blob": True,
        }
        member_info, member_bytes = read_tt5l_archive_zip(archive_path)
        assert member_info.filename == "0.bin"
        parsed = parse_archive(member_bytes)
        expected = tt5l_sideinfo_variant_arrays(source_sideinfo, seed=99)[variant]
        assert np.array_equal(parsed.per_pair_side_info, expected)

    assert rows["random_lsb"]["sideinfo_liveness"]["nonzero_values"] == 3 * 45
    assert rows["trained"]["sideinfo_equal_source"] is True
    assert rows["zero"]["sideinfo_equal_zero"] is True
    assert rows["ablated"]["sideinfo_liveness"]["nonzero_values"] == 2


def test_build_tt5l_sideinfo_variant_packets_fails_closed_on_all_zero_source(
    tmp_path: Path,
) -> None:
    source_sideinfo = np.zeros((3, 45), dtype=np.int8)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_tt5l_archive_zip(source_archive, source_sideinfo)

    manifest = build_tt5l_sideinfo_variant_packets(
        source_archive=source_archive,
        output_root=tmp_path / "variants",
        repo_root=tmp_path,
    )

    assert "tt5l_source_trained_sideinfo_all_zero" in manifest["blockers"]
    assert (
        f"tt5l_source_num_pairs_not_full_contest:3_expected_{TT5L_CONTEST_NUM_PAIRS}"
        in manifest["blockers"]
    )
    rows = {row["variant"]: row for row in manifest["variants"]}
    assert "trained_variant_degenerate_from_zero_source" in rows["trained"]["blockers"]
    assert "shuffled_variant_degenerate_from_zero_source" in rows["shuffled"]["blockers"]
    assert "ablated_variant_degenerate_from_zero_source" in rows["ablated"]["blockers"]
    assert rows["random_lsb"]["sideinfo_liveness"]["nonzero_values"] == 3 * 45


def test_build_tt5l_sideinfo_variant_packets_is_deterministic(tmp_path: Path) -> None:
    source_sideinfo = np.zeros((3, 45), dtype=np.int8)
    source_sideinfo[:, 0] = [1, 2, 3]
    source_archive = tmp_path / "source" / "archive.zip"
    _write_tt5l_archive_zip(source_archive, source_sideinfo)

    first = build_tt5l_sideinfo_variant_packets(
        source_archive=source_archive,
        output_root=tmp_path / "variants-a",
        repo_root=tmp_path,
        seed=7,
    )
    second = build_tt5l_sideinfo_variant_packets(
        source_archive=source_archive,
        output_root=tmp_path / "variants-b",
        repo_root=tmp_path,
        seed=7,
    )

    first_rows = {row["variant"]: row for row in first["variants"]}
    second_rows = {row["variant"]: row for row in second["variants"]}
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        assert first_rows[variant]["archive_sha256"] == second_rows[variant]["archive_sha256"]
        assert first_rows[variant]["sideinfo_sha256"] == second_rows[variant]["sideinfo_sha256"]


def test_build_tt5l_sideinfo_variant_packets_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    source_archive = tmp_path / "source" / "archive.zip"
    _write_tt5l_archive_zip(source_archive, np.zeros((2, 45), dtype=np.int8))
    output_root = tmp_path / "repo" / "experiments" / "results" / "tt5l_variants"
    output_json = tmp_path / "repo" / ".omx" / "research" / "variants.json"
    output_md = tmp_path / "repo" / ".omx" / "research" / "variants.md"
    output_root.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_tt5l_sideinfo_variant_packets.py"),
            "--source-archive",
            str(source_archive),
            "--output-root",
            str(output_root),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--repo-root",
            str(tmp_path / "repo"),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["variant_count"] == 5
    assert "tt5l_source_trained_sideinfo_all_zero" in payload["blockers"]
    assert (
        f"tt5l_source_num_pairs_not_full_contest:2_expected_{TT5L_CONTEST_NUM_PAIRS}"
        in payload["blockers"]
    )
    assert "submission_runtime_dir_missing" in payload["blockers"]
    assert payload["runtime"]["available"] is False
    assert output_md.read_text(encoding="utf-8").startswith(
        "# L5 v2 TT5L side-info variant packets"
    )
