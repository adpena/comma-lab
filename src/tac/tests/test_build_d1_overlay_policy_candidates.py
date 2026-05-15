# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
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
        resolution=(8, 8), constant_value=2.0
    )
    payload = encode_polytope_payload(
        torch.ones(8, 8),
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
    candidate_d1 = Path(row["submission_dir"]) / "d1_polytope.bin"
    parsed = parse_archive(candidate_d1.read_bytes())
    assert parsed.meta["overlay_channel_policy"] == "neg_green"
    assert parsed.meta["overlay_amplitude_scale"] == 0.5
    assert parsed.meta["overlay_sign_policy"] == "negate_payload"
    assert Path(row["archive_zip"]).is_file()


def test_d1_policy_builder_rejects_bad_amplitude_scale() -> None:
    module = _load_tool()
    try:
        module._parse_amplitude_scales("0.5,1.5")
    except SystemExit as exc:
        assert "unsupported amplitude scale" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("bad amplitude scale should exit")
