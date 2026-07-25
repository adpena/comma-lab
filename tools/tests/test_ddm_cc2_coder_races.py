from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from tac.optimization.ddm_cc2_coder_races import (
    build_per_stream_price_table,
    extract_recursive_zip_leaves,
    terminal_quantization_thetas,
)
from tools.run_ddm_cc2_coder_races import _config_hash, _load_config


def _stored_zip(members: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, payload)
    return buffer.getvalue()


def test_terminal_quantizer_proxies_are_seeded_and_source_labeled() -> None:
    theta = np.asarray([-1.5001, -0.5001, 0.4999, 1.5001], dtype=np.float32)
    first = terminal_quantization_thetas(theta, seed=0)
    second = terminal_quantization_thetas(theta, seed=0)
    assert list(first) == [
        "CAMERA_Q8_EXACT",
        "C3_ORIGINAL_TERMINAL_PROXY",
        "COOL_CHIC_V5_TERMINAL_PROXY",
    ]
    for arm_id in first:
        np.testing.assert_array_equal(first[arm_id][0], second[arm_id][0])
    assert first["C3_ORIGINAL_TERMINAL_PROXY"][1]["source_commit"].startswith("e63e7519")
    assert first["COOL_CHIC_V5_TERMINAL_PROXY"][1]["source_commit"].startswith("a6fe38a4")
    assert all(
        "NOT_RETRAINING" in first[arm_id][1]["operator_scope"]
        for arm_id in ("C3_ORIGINAL_TERMINAL_PROXY", "COOL_CHIC_V5_TERMINAL_PROXY")
    )


def test_recursive_inventory_and_price_table_reconcile_exact_bytes() -> None:
    nested = _stored_zip(
        [
            ("predict/movable_polygon_worldsheet.g1s", b"A" * 256),
            ("state/example.ddq8", bytes(range(64))),
        ]
    )
    composition = _stored_zip(
        [
            ("manifest/pc1.json", b'{"schema":"test"}'),
            ("pose/pc1.ddp", b"P" * 40),
            ("parent/w_joint.zip", nested),
        ]
    )
    leaves, overhead = extract_recursive_zip_leaves(composition)
    assert {row.category for row in leaves} >= {
        "V15_G1_PAYLOAD",
        "POSE_40B_HOME",
        "W_JOINT_STATE_STREAM",
    }
    assert overhead + sum(len(row.payload) for row in leaves) == len(composition)
    table = build_per_stream_price_table(composition)
    assert table["recursive_fixed_zip_overhead_bytes"] == overhead
    assert table["current_leaf_bytes"] + overhead == len(composition)
    assert table["selected_total_delta_dseg"] == 0.0
    assert table["selected_total_delta_dpose"] == 0.0
    assert all(row["parseback_exact_all_arms"] for row in table["rows"])
    assert table["receiver_status"].startswith("PRICE_TABLE_ONLY")


def test_checked_in_config_is_strict_ssd_advisory_and_hashable() -> None:
    path = Path(".omx/research/configs/ddm_cc2_coder_races_20260725.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _load_config(path)
    assert config == raw
    assert config["seed"] == 0
    assert config["output_root"].startswith("/Volumes/VertigoDataTier/pact/")
    assert len(_config_hash(config)) == 64
