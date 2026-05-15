# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from tac.repo_io import read_json
from tac.substrates.d1_segnet_margin_polytope import (
    D1PolytopeConfig,
    compute_logit_margin_map_dummy,
    encode_polytope_payload,
    pack_archive,
    parse_archive,
)
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/build_d1_overlay_policy_candidates.py",
        "build_d1_overlay_policy_candidates_test",
    )


def _write_d1_inputs(tmp_path: Path) -> tuple[Path, Path]:
    a1_bytes = b"synthetic-a1-base"
    a1_path = tmp_path / "a1.bin"
    a1_path.write_bytes(a1_bytes)
    base_sha = hashlib.sha256(a1_bytes).hexdigest()
    cfg = D1PolytopeConfig(
        margin_map_resolution=(8, 8),
        polytope_payload_bits=128,
        jacobian_lipschitz=10.0,
    )
    margin_map = compute_logit_margin_map_dummy(
        resolution=(8, 8), constant_value=20.0
    )
    payload = encode_polytope_payload(
        torch.full((8, 8), 20.0),
        jacobian_lipschitz=10.0,
        budget_bits=128,
    )
    d1_bytes = pack_archive(
        margin_map=margin_map,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=len(a1_bytes),
        config=cfg,
        extra_meta={"overlay_channel_policy": "rgb"},
    )
    d1_path = tmp_path / "d1_polytope.bin"
    d1_path.write_bytes(d1_bytes)
    return d1_path, a1_path


def test_d1_policy_builder_materializes_channel_amplitude_sign_product(tmp_path: Path) -> None:
    module = _load_tool()
    d1_path, a1_path = _write_d1_inputs(tmp_path)
    out_dir = tmp_path / "out"

    rc = module.main(
        [
            "--d1-bin",
            str(d1_path),
            "--a1-bin",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--policies",
            "green,neg_green",
            "--amplitude-scales",
            "0.5,1.0",
            "--sign-policies",
            "payload,negate_payload",
        ]
    )

    assert rc == 0
    summary = read_json(out_dir / "d1_overlay_policy_candidates_manifest.json")
    assert summary["policy_count"] == 8
    assert summary["amplitude_scales"] == [0.5, 1.0]
    assert summary["sign_policies"] == ["payload", "negate_payload"]
    row = next(
        item
        for item in summary["candidates"]
        if item["overlay_channel_policy"] == "neg_green"
        and item["overlay_amplitude_scale"] == 0.5
        and item["overlay_sign_policy"] == "negate_payload"
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["base_member_name"] == "a1.bin"
    assert row["base_member_bytes"] == len(b"synthetic-a1-base")
    assert row["source_base_archive_bytes"] == len(b"synthetic-a1-base")
    assert row["archive_delta_vs_source_base_archive_bytes"] == (
        row["archive_bytes"] - len(b"synthetic-a1-base")
    )
    assert row["d1_sidecar_bytes"] == row["d1_bin_bytes"]
    candidate_d1 = Path(row["submission_dir"]) / "d1_polytope.bin"
    parsed = parse_archive(candidate_d1.read_bytes())
    assert parsed.meta["overlay_channel_policy"] == "neg_green"
    assert parsed.meta["overlay_amplitude_scale"] == 0.5
    assert parsed.meta["overlay_sign_policy"] == "negate_payload"
    assert Path(row["archive_zip"]).is_file()
    diag = row["d1_overlay_diagnostics"]
    assert diag["decoded_noise_nonzero_pixels"] > 0
    assert diag["unsafe_nonzero_pixels"] == 0
    assert diag["decoded_noise_abs_sum"] >= diag["decoded_noise_nonzero_pixels"]
    assert diag["attenuated_overlay_abs_sum"] >= diag["attenuated_overlay_nonzero_pixels"]
    assert diag["estimated_changed_lsb_l1_upper_bound_per_pair"] >= (
        diag["estimated_changed_bytes_upper_bound_per_pair"]
    )
    duplicate = next(
        item
        for item in summary["candidates"]
        if item["duplicate_of_candidate_id"] is not None
    )
    assert duplicate["duplicate_of_candidate_id"] is not None
    assert any(
        blocker.startswith("d1_overlay_effect_duplicate_of_")
        for blocker in duplicate["dispatch_blockers"]
    )


def test_d1_policy_builder_marks_decoded_zero_payload_blocker(tmp_path: Path) -> None:
    module = _load_tool()
    a1_bytes = b"synthetic-a1-base"
    a1_path = tmp_path / "a1.bin"
    a1_path.write_bytes(a1_bytes)
    base_sha = hashlib.sha256(a1_bytes).hexdigest()
    cfg = D1PolytopeConfig(margin_map_resolution=(8, 8), polytope_payload_bits=128)
    zero_margin = torch.zeros(8, 8)
    payload = encode_polytope_payload(
        zero_margin,
        jacobian_lipschitz=10.0,
        budget_bits=128,
    )
    d1_path = tmp_path / "d1_zero.bin"
    d1_path.write_bytes(
        pack_archive(
            margin_map=zero_margin,
            polytope_payload=payload,
            jacobian_lipschitz=10.0,
            base_substrate_id="a1",
            base_archive_sha256=base_sha,
            base_archive_bytes=len(a1_bytes),
            config=cfg,
            extra_meta={},
        )
    )
    out_dir = tmp_path / "out_zero"

    rc = module.main(
        [
            "--d1-bin",
            str(d1_path),
            "--a1-bin",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--policies",
            "rgb",
        ]
    )

    assert rc == 0
    summary = read_json(out_dir / "d1_overlay_policy_candidates_manifest.json")
    row = summary["candidates"][0]
    assert row["d1_overlay_diagnostics"]["decoded_noise_nonzero_pixels"] == 0
    assert row["d1_overlay_diagnostics"]["attenuated_overlay_abs_sum"] == 0
    assert (
        row["d1_overlay_diagnostics"][
            "estimated_changed_lsb_l1_upper_bound_per_pair"
        ]
        == 0
    )
    assert "d1_decoded_polytope_payload_all_zero" in row["dispatch_blockers"]


def test_d1_policy_builder_can_rebuild_payload_budget_sweep(tmp_path: Path) -> None:
    module = _load_tool()
    d1_path, a1_path = _write_d1_inputs(tmp_path)
    out_dir = tmp_path / "out_sweep"

    rc = module.main(
        [
            "--d1-bin",
            str(d1_path),
            "--a1-bin",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--policies",
            "rgb",
            "--payload-budget-bits",
            "512",
            "--jacobian-lipschitz",
            "1.0",
        ]
    )

    assert rc == 0
    summary = read_json(out_dir / "d1_overlay_policy_candidates_manifest.json")
    assert summary["payload_budget_bits"] == [512]
    assert summary["jacobian_lipschitz_values"] == [1.0]
    row = summary["candidates"][0]
    assert row["payload_budget_bits"] == 512
    assert row["jacobian_lipschitz_override"] == 1.0
    assert "budget_512_L_1" in row["candidate_id"]
    parsed = parse_archive((Path(row["submission_dir"]) / "d1_polytope.bin").read_bytes())
    assert parsed.meta["payload_sweep_candidate"] is True


def test_d1_policy_builder_can_rebuild_shrunk_margin_resolution(tmp_path: Path) -> None:
    module = _load_tool()
    d1_path, a1_path = _write_d1_inputs(tmp_path)
    out_dir = tmp_path / "out_shrunk"

    rc = module.main(
        [
            "--d1-bin",
            str(d1_path),
            "--a1-bin",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--policies",
            "rgb",
            "--payload-budget-bits",
            "512",
            "--jacobian-lipschitz",
            "1.0",
            "--margin-map-resolution",
            "4x4",
        ]
    )

    assert rc == 0
    summary = read_json(out_dir / "d1_overlay_policy_candidates_manifest.json")
    assert summary["margin_map_resolution"] == [4, 4]
    row = summary["candidates"][0]
    assert row["margin_map_resolution_override"] == [4, 4]
    assert "res_4x4" in row["candidate_id"]
    parsed = parse_archive((Path(row["submission_dir"]) / "d1_polytope.bin").read_bytes())
    assert (parsed.height, parsed.width) == (4, 4)
    assert parsed.meta["source_margin_map_resolution"] == [8, 8]


def test_d1_policy_builder_materializes_pair_mask_policy(tmp_path: Path) -> None:
    module = _load_tool()
    d1_path, a1_path = _write_d1_inputs(tmp_path)
    out_dir = tmp_path / "out_pair_mask"
    mask_path = tmp_path / "pair_mask.json"
    signs = [1, 0, -1, 0, 1]
    mask_path.write_text(json.dumps({"pair_signs": signs}) + "\n", encoding="utf-8")

    rc = module.main(
        [
            "--d1-bin",
            str(d1_path),
            "--a1-bin",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--policies",
            "green",
            "--sign-policies",
            "pair_mask",
            "--pair-sign-mask-json",
            str(mask_path),
            "--pair-sign-mask-label",
            "unit",
            "--expected-pairs",
            str(len(signs)),
        ]
    )

    assert rc == 0
    summary = read_json(out_dir / "d1_overlay_policy_candidates_manifest.json")
    assert summary["sign_policies"] == ["pair_mask"]
    assert summary["pair_sign_mask"]["active_pairs"] == 3
    row = summary["candidates"][0]
    assert row["candidate_id"].endswith("_sign_pair_mask_pairmask_unit")
    assert row["pair_sign_mask"]["positive_pairs"] == 2
    assert row["pair_sign_mask"]["expected_pairs"] == len(signs)
    assert row["pair_sign_mask"]["partial_smoke_allowed"] is False
    assert row["pair_sign_mask"]["provenance"]["mask_scope"] == "full_contest_selector"
    assert row["pair_sign_mask"]["provenance"]["source_sha256"] == hashlib.sha256(
        mask_path.read_bytes()
    ).hexdigest()
    assert row["pair_sign_mask"]["provenance"]["packed_raw_bytes"] == 2
    assert row["pair_sign_mask"]["provenance"]["packed_base85_chars"] == 3
    assert row["compressed_rate_accounting"]["archive_bytes_are_zip_compressed"] is True
    assert row["compressed_rate_accounting"]["pair_mask_packed_raw_bytes"] == 2
    assert row["compressed_rate_accounting"]["pair_mask_packed_base85_chars"] == 3
    assert len(row["deterministic_provenance_sha256"]) == 64
    assert summary["pair_sign_mask"]["provenance"]["mask_scope"] == (
        "full_contest_selector"
    )
    assert len(summary["deterministic_provenance_sha256"]) == 64
    assert row["d1_overlay_diagnostics"]["pair_mask_active_pairs"] == 3
    parsed = parse_archive((Path(row["submission_dir"]) / "d1_polytope.bin").read_bytes())
    assert parsed.meta["overlay_sign_policy"] == "pair_mask"
    assert "pair_mask_b85" in parsed.meta
    assert "pair_mask_n" in parsed.meta
    assert "overlay_pair_sign_mask_b64" not in parsed.meta
    assert "overlay_pair_sign_mask_bits_hex" not in parsed.meta
    assert "overlay_pair_sign_mask_sha256" not in parsed.meta
    assert parsed.meta["pair_mask_n"] == len(signs)


def test_d1_policy_builder_blocks_all_zero_pair_mask(tmp_path: Path) -> None:
    module = _load_tool()
    d1_path, a1_path = _write_d1_inputs(tmp_path)
    out_dir = tmp_path / "out_zero_pair_mask"
    mask_path = tmp_path / "pair_mask_zero.json"
    mask_path.write_text(json.dumps({"pair_signs": [0, 0, 0]}) + "\n", encoding="utf-8")

    rc = module.main(
        [
            "--d1-bin",
            str(d1_path),
            "--a1-bin",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--policies",
            "green",
            "--sign-policies",
            "pair_mask",
            "--pair-sign-mask-json",
            str(mask_path),
            "--expected-pairs",
            "3",
        ]
    )

    assert rc == 0
    row = read_json(out_dir / "d1_overlay_policy_candidates_manifest.json")[
        "candidates"
    ][0]
    assert row["d1_overlay_diagnostics"]["pair_mask_active_pairs"] == 0
    assert "d1_pair_mask_has_no_active_pairs" in row["dispatch_blockers"]


def test_d1_policy_builder_rejects_partial_pair_mask_by_default(tmp_path: Path) -> None:
    module = _load_tool()
    d1_path, a1_path = _write_d1_inputs(tmp_path)
    mask_path = tmp_path / "partial_pair_mask.json"
    mask_path.write_text(json.dumps({"pair_signs": [1, 0, -1]}) + "\n", encoding="utf-8")

    try:
        module.main(
            [
                "--d1-bin",
                str(d1_path),
                "--a1-bin",
                str(a1_path),
                "--output-dir",
                str(tmp_path / "out_partial_reject"),
                "--policies",
                "green",
                "--sign-policies",
                "pair_mask",
                "--pair-sign-mask-json",
                str(mask_path),
            ]
        )
    except SystemExit as exc:
        assert "pair count mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("partial pair mask should require explicit smoke waiver")


def test_d1_policy_builder_marks_partial_pair_mask_smoke(tmp_path: Path) -> None:
    module = _load_tool()
    d1_path, a1_path = _write_d1_inputs(tmp_path)
    out_dir = tmp_path / "out_partial_smoke"
    mask_path = tmp_path / "partial_pair_mask.json"
    mask_path.write_text(json.dumps({"pair_signs": [1, 0, -1]}) + "\n", encoding="utf-8")

    rc = module.main(
        [
            "--d1-bin",
            str(d1_path),
            "--a1-bin",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--policies",
            "green",
            "--sign-policies",
            "pair_mask",
            "--pair-sign-mask-json",
            str(mask_path),
            "--allow-partial-smoke",
        ]
    )

    assert rc == 0
    row = read_json(out_dir / "d1_overlay_policy_candidates_manifest.json")[
        "candidates"
    ][0]
    assert row["pair_sign_mask"]["n_pairs"] == 3
    assert row["pair_sign_mask"]["expected_pairs"] == 600
    assert row["pair_sign_mask"]["partial_smoke_allowed"] is True
    assert row["pair_sign_mask"]["provenance"]["mask_scope"] == "partial_smoke_selector"
    assert row["pair_sign_mask"]["provenance"]["full_expected_length"] is False
    assert "d1_pair_mask_partial_smoke_not_contest_packet" in row["dispatch_blockers"]


def test_d1_policy_builder_rejects_noninteger_pair_mask_values(tmp_path: Path) -> None:
    module = _load_tool()
    d1_path, a1_path = _write_d1_inputs(tmp_path)
    mask_path = tmp_path / "bad_pair_mask.json"
    mask_path.write_text(
        json.dumps({"pair_signs": [1, True, -1]}) + "\n",
        encoding="utf-8",
    )

    try:
        module.main(
            [
                "--d1-bin",
                str(d1_path),
                "--a1-bin",
                str(a1_path),
                "--output-dir",
                str(tmp_path / "out_bad_mask"),
                "--policies",
                "green",
                "--sign-policies",
                "pair_mask",
                "--pair-sign-mask-json",
                str(mask_path),
                "--expected-pairs",
                "3",
            ]
        )
    except SystemExit as exc:
        assert "integer -1, 0, or 1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("bool pair-mask values should fail closed")


def test_d1_policy_builder_rejects_bad_amplitude_scale() -> None:
    module = _load_tool()
    try:
        module._parse_amplitude_scales("0.5,1.5")
    except SystemExit as exc:
        assert "unsupported amplitude scale" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("bad amplitude scale should exit")
