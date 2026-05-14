# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch

from tac.hnerv_arch_schema import (
    HNeRVArchConfig,
    compare_schema_shapes,
    generate_hnerv_state_schema,
    initialize_state_dict_by_overlap,
    schema_fingerprint,
    schema_numel,
    select_base_channels_for_element_retention,
    state_dict_schema_rows,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA


def test_default_generated_schema_matches_pr101_contract() -> None:
    schema = generate_hnerv_state_schema(HNeRVArchConfig())

    assert schema == FIXED_STATE_SCHEMA
    assert schema_numel(schema) == sum(torch.empty(*shape).numel() for _, shape in schema)
    assert len(schema_fingerprint(schema)) == 64


def test_select_base_channels_for_retention_is_floor_bounded() -> None:
    baseline = generate_hnerv_state_schema(HNeRVArchConfig(base_channels=36))
    selected = select_base_channels_for_element_retention(
        element_retention=0.45,
        baseline_base_channels=36,
    )
    selected_schema = generate_hnerv_state_schema(selected)

    assert selected.base_channels < 36
    assert schema_numel(selected_schema) <= schema_numel(baseline) * 0.45


def test_overlap_initialization_preserves_target_schema_and_prefix_values() -> None:
    source_schema = generate_hnerv_state_schema(
        HNeRVArchConfig(latent_dim=4, base_channels=8, eval_size=(64, 64))
    )
    target_schema = generate_hnerv_state_schema(
        HNeRVArchConfig(latent_dim=4, base_channels=5, eval_size=(64, 64))
    )
    source = {
        name: torch.arange(int(torch.empty(*shape).numel()), dtype=torch.float32).reshape(shape)
        for name, shape in source_schema
    }

    target = initialize_state_dict_by_overlap(source, target_schema=target_schema)
    findings = compare_schema_shapes(target_schema, state_dict_schema_rows(target))

    assert findings == []
    assert target["stem.weight"].shape == target_schema[0][1]
    assert torch.equal(
        target["stem.weight"][:2, :2],
        source["stem.weight"][:2, :2],
    )


def test_overlap_initialization_zero_fills_missing_tensors() -> None:
    target_schema = (("new.weight", (2, 3)),)
    target = initialize_state_dict_by_overlap({}, target_schema=target_schema)

    assert target["new.weight"].shape == (2, 3)
    assert torch.count_nonzero(target["new.weight"]).item() == 0
