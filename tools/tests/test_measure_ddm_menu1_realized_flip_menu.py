# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tac.optimization.ddm_realized_flip_menu import encode_local_statistics
from tools.measure_ddm_menu1_realized_flip_menu import (
    CAMERA_HW,
    SEG_HW,
    Menu1Config,
    _geometry_statistics_camera,
    _storage_preflight,
    _targeted_camera,
)


def test_checked_in_config_is_strict() -> None:
    config = Menu1Config.model_validate_json(
        Path(
            ".omx/research/configs/ddm_menu1_realized_flip_menu_20260723.json"
        ).read_bytes()
    )
    assert config.execution_allowed is True
    assert config.score_claim is False
    assert config.checkpoint_root.startswith("/Volumes/VertigoDataTier/pact/")


def test_storage_preflight_receipt_excludes_volatile_free_bytes(monkeypatch) -> None:
    config = Menu1Config.model_validate_json(
        Path(
            ".omx/research/configs/ddm_menu1_realized_flip_menu_20260723.json"
        ).read_bytes()
    )
    observed_free = iter((1 << 30, 2 << 30))
    monkeypatch.setattr(
        "tools.measure_ddm_menu1_realized_flip_menu.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=next(observed_free)),
    )
    first = _storage_preflight(config)
    second = _storage_preflight(config)
    assert first == second
    assert first["observed_free_bytes_at_least"] == 512 << 20
    assert "observed_free_bytes" not in first


def test_targeted_camera_changes_only_frame1_mask() -> None:
    camera = np.full((1, 2, *CAMERA_HW, 3), 10, dtype=np.uint8)
    masks = np.zeros((1, *SEG_HW), dtype=bool)
    masks[:, 10:20, 30:40] = True
    palette = np.zeros((5, 3), dtype=np.uint8)
    palette[0] = (100, 110, 120)
    result = _targeted_camera(
        current=camera,
        masks=masks,
        palette=palette,
    )
    assert np.array_equal(result[:, 0], camera[:, 0])
    assert np.count_nonzero(result[:, 1] != camera[:, 1]) > 0


def test_geometry_composition_preserves_unowned_pixels() -> None:
    camera = np.full((1, 2, *CAMERA_HW, 3), 17, dtype=np.uint8)
    semantic = np.zeros((1, *SEG_HW), dtype=np.uint8)
    semantic[:, :, SEG_HW[1] // 2 :] = 1
    owned = np.zeros_like(semantic, dtype=bool)
    owned[:, 100:200, 100:200] = True
    palette = np.asarray(
        [(10, 20, 30), (40, 50, 60), (70, 80, 90), (1, 2, 3), (4, 5, 6)],
        dtype=np.uint8,
    )
    payload = encode_local_statistics(
        np.ones((5, 16, 3), dtype=np.float32),
        np.zeros((5, 16, 3), dtype=np.float32),
    )
    result = _geometry_statistics_camera(
        base_camera=camera,
        semantic=semantic,
        owned=owned,
        palette=palette,
        statistics_payload=payload,
    )
    assert np.array_equal(result[:, 0], camera[:, 0])
    assert np.array_equal(result[:, :, 0, 0], camera[:, :, 0, 0])
    assert np.count_nonzero(result != camera) > 0
