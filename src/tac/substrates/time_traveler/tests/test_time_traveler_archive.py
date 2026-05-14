# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import torch

from tac.substrates import time_traveler as compat
from tac.substrates import time_traveler_l5_autonomy as canonical
from tac.substrates.time_traveler import archive as compat_archive
from tac.substrates.time_traveler_l5_autonomy import archive as canonical_archive


def test_time_traveler_package_is_compatibility_alias() -> None:
    assert compat.TIME_TRAVELER_METADATA["compatibility_alias"] is True
    assert compat.TIME_TRAVELER_METADATA["canonical_package"] == (
        "tac.substrates.time_traveler_l5_autonomy"
    )
    assert compat.TimeTravelerConfig is canonical.TimeTravelerConfig
    assert compat.TimeTravelerArchive is canonical.TimeTravelerArchive
    assert compat.pack_archive is canonical.pack_archive
    assert compat.parse_archive is canonical.parse_archive
    assert compat_archive.TT5L_MAGIC == canonical_archive.TT5L_MAGIC


def test_time_traveler_alias_roundtrips_through_canonical_grammar() -> None:
    cfg = compat.TimeTravelerConfig(
        hidden_dim=8,
        num_hidden_layers=1,
        num_pairs=2,
        output_height=4,
        output_width=4,
        foveation_grid_h=2,
        foveation_grid_w=2,
        per_pair_side_info_bytes=4,
    )
    model = compat.TimeTravelerSubstrate(cfg)
    side_info = np.zeros((cfg.num_pairs, cfg.per_pair_side_info_bytes), dtype=np.int8)
    blob = compat.pack_archive(
        world_model_state_dict=model.state_dict(),
        per_pair_side_info=side_info,
        meta={"coord_dim": cfg.coord_dim},
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        foveation_grid_h=cfg.foveation_grid_h,
        foveation_grid_w=cfg.foveation_grid_w,
        pose_dim=cfg.pose_dim,
        per_pair_bytes=cfg.per_pair_side_info_bytes,
    )

    parsed = canonical.parse_archive(blob)
    assert parsed.schema_version == canonical.TT5L_SCHEMA_VERSION
    assert parsed.num_pairs == 2
    assert parsed.per_pair_side_info.shape == (2, 4)
    assert all(isinstance(value, torch.Tensor) for value in parsed.world_model_state_dict.values())
