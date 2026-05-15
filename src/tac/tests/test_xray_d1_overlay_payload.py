# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from tac.repo_io import read_json
from tac.substrates.d1_segnet_margin_polytope import (
    D1PolytopeConfig,
    encode_polytope_payload,
    pack_archive,
)
from tac.substrates.d1_segnet_margin_polytope.overlay import pack_pair_sign_mask
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_xray_d1_overlay_payload_reports_zero_payload_blocker(tmp_path: Path) -> None:
    module = load_repo_tool(
        REPO_ROOT,
        "tools/xray_d1_overlay_payload.py",
        "xray_d1_overlay_payload_test",
    )
    base_bytes = b"synthetic-a1-base"
    base_sha = hashlib.sha256(base_bytes).hexdigest()
    zero_margin = torch.zeros(8, 8)
    cfg = D1PolytopeConfig(margin_map_resolution=(8, 8), polytope_payload_bits=128)
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
            base_archive_bytes=len(base_bytes),
            config=cfg,
            extra_meta={},
        )
    )
    json_out = tmp_path / "xray.json"

    rc = module.main(["--d1-bin", str(d1_path), "--json-out", str(json_out)])

    assert rc == 0
    payload_json = read_json(json_out)
    diag = payload_json["d1_overlay_diagnostics"]
    assert diag["decoded_noise_nonzero_pixels"] == 0
    assert "d1_decoded_polytope_payload_all_zero" in diag["dispatch_blockers"]


def test_xray_d1_overlay_payload_decodes_pair_mask_metadata(tmp_path: Path) -> None:
    module = load_repo_tool(
        REPO_ROOT,
        "tools/xray_d1_overlay_payload.py",
        "xray_d1_overlay_payload_pair_mask_test",
    )
    base_bytes = b"synthetic-a1-base"
    base_sha = hashlib.sha256(base_bytes).hexdigest()
    margin = torch.full((8, 8), 8.0)
    cfg = D1PolytopeConfig(margin_map_resolution=(8, 8), polytope_payload_bits=256)
    payload = encode_polytope_payload(
        margin,
        jacobian_lipschitz=2.0,
        budget_bits=256,
    )
    mask = (1, 0, -1, 0)
    d1_path = tmp_path / "d1_pair_mask.bin"
    d1_path.write_bytes(
        pack_archive(
            margin_map=margin,
            polytope_payload=payload,
            jacobian_lipschitz=2.0,
            base_substrate_id="a1",
            base_archive_sha256=base_sha,
            base_archive_bytes=len(base_bytes),
            config=cfg,
            extra_meta={
                "overlay_sign_policy": "pair_mask",
                "pair_mask_b85": pack_pair_sign_mask(mask),
                "pair_mask_n": len(mask),
            },
        )
    )
    json_out = tmp_path / "xray_pair_mask.json"

    rc = module.main(["--d1-bin", str(d1_path), "--json-out", str(json_out)])

    assert rc == 0
    payload_json = read_json(json_out)
    diag = payload_json["d1_overlay_diagnostics"]
    assert diag["overlay_sign_policy"] == "pair_mask"
    assert diag["pair_mask_active_pairs"] == 2
    assert payload_json["pair_sign_mask_source"] == "archive_meta"


def test_xray_d1_overlay_payload_pair_mask_override(tmp_path: Path) -> None:
    module = load_repo_tool(
        REPO_ROOT,
        "tools/xray_d1_overlay_payload.py",
        "xray_d1_overlay_payload_pair_mask_override_test",
    )
    base_bytes = b"synthetic-a1-base"
    base_sha = hashlib.sha256(base_bytes).hexdigest()
    margin = torch.full((8, 8), 8.0)
    cfg = D1PolytopeConfig(margin_map_resolution=(8, 8), polytope_payload_bits=256)
    payload = encode_polytope_payload(
        margin,
        jacobian_lipschitz=2.0,
        budget_bits=256,
    )
    d1_path = tmp_path / "d1_pair_mask_no_meta.bin"
    d1_path.write_bytes(
        pack_archive(
            margin_map=margin,
            polytope_payload=payload,
            jacobian_lipschitz=2.0,
            base_substrate_id="a1",
            base_archive_sha256=base_sha,
            base_archive_bytes=len(base_bytes),
            config=cfg,
            extra_meta={"overlay_sign_policy": "pair_mask"},
        )
    )
    mask_json = tmp_path / "pair_mask.json"
    mask_json.write_text('{"pair_signs":[1,0,-1,1]}', encoding="utf-8")
    json_out = tmp_path / "xray_pair_mask_override.json"

    rc = module.main(
        [
            "--d1-bin",
            str(d1_path),
            "--pair-sign-mask-json",
            str(mask_json),
            "--json-out",
            str(json_out),
        ]
    )

    assert rc == 0
    payload_json = read_json(json_out)
    diag = payload_json["d1_overlay_diagnostics"]
    assert diag["pair_mask_active_pairs"] == 3
    assert payload_json["pair_sign_mask_source"] == mask_json.as_posix()
