# SPDX-License-Identifier: MIT
"""G87 exact G78-to-G72 adapter behavior and fail-closed tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import tac.witness_control.taskspace_g87_g78_to_g72_exact_compile_adapter_v1 as g87
from tac.witness_control.taskspace_batch16_margin_base_scorer_cache_v1 import (
    REMAINING_G72_BLOCKERS,
    canonical_json_bytes,
)
from tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 import (
    FRESH_BATCH16_MARGIN_CUSTODY_OWED,
    FRESH_V15_BASE_SCORER_CACHE_OWED,
    derive_v9_boundary_shearlet_stage_proposals,
)

HEX_A = "a" * 64
HEX_B = "b" * 64


def _readonly(
    dtype: np.dtype[Any] | type[np.generic],
) -> np.ndarray:
    singleton = np.zeros((1, 1, 1), dtype=dtype)
    return np.broadcast_to(singleton, (120, 384, 512))


def _fake_receipt(root: Path) -> tuple[dict[str, Any], list[Any]]:
    stages: list[Any] = []
    stage_bindings: list[dict[str, Any]] = []
    for index in range(5):
        start = index * 120
        stop = start + 120
        stage_self = hashlib.sha256(f"stage-{index}".encode()).hexdigest()
        stage_receipt = {
            "stage_receipt_sha256": stage_self,
            "target_cells": {
                "dtype": "uint8",
                "global_bank_sha256": HEX_A,
                "shape": [120, 384, 512],
                "slice_sha256": hashlib.sha256(f"target-{index}".encode()).hexdigest(),
                "source_path": str(root / "target.u8"),
            },
            "files": {
                "target_margins_f32": {
                    "bytes": 120 * 384 * 512 * 4,
                    "dtype": "float32_le",
                    "path": str(root / f"target-margins-{index}.f32"),
                    "sha256": hashlib.sha256(f"tm-{index}".encode()).hexdigest(),
                    "shape": [120, 384, 512],
                },
                "described_cells_u8": {
                    "bytes": 120 * 384 * 512,
                    "dtype": "uint8",
                    "path": str(root / f"described-{index}.u8"),
                    "sha256": hashlib.sha256(f"dc-{index}".encode()).hexdigest(),
                    "shape": [120, 384, 512],
                },
                "described_margins_f32": {
                    "bytes": 120 * 384 * 512 * 4,
                    "dtype": "float32_le",
                    "path": str(root / f"described-margins-{index}.f32"),
                    "sha256": hashlib.sha256(f"dm-{index}".encode()).hexdigest(),
                    "shape": [120, 384, 512],
                },
            },
        }
        stage_path = root / f"stage-{index}.json"
        stage_path.write_bytes(canonical_json_bytes(stage_receipt))
        stage_bindings.append(
            {
                "bytes": stage_path.stat().st_size,
                "path": str(stage_path),
                "sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
                "stage_index": index,
                "pair_range": [start, stop],
                "stage_receipt_sha256": stage_self,
                "digest_chain_sha256": HEX_B,
            }
        )
        stages.append(
            SimpleNamespace(
                stage_index=index,
                pair_range=(start, stop),
                target_cells_u8=_readonly(np.uint8),
                target_margins_f32=_readonly(np.dtype("<f4")),
                described_cells_u8=_readonly(np.uint8),
                described_margins_f32=_readonly(np.dtype("<f4")),
            )
        )
    receipt = {
        "aggregate_receipt_sha256": HEX_B,
        "pair_count": 600,
        "stage_pairs": 120,
        "stage_count": 5,
        "scorer_batch_pairs": 16,
        "scorer_hw": [384, 512],
        "class_count": 5,
        "closed_blockers": [
            FRESH_BATCH16_MARGIN_CUSTODY_OWED,
            FRESH_V15_BASE_SCORER_CACHE_OWED,
        ],
        "remaining_g72_blockers_unmodified": list(REMAINING_G72_BLOCKERS),
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
        "dense_fields_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "semantic_custody": {
            "archive": {"sha256": HEX_A, "bytes": 1, "path": "semantic"},
            "compile_receipt": {
                "sha256": HEX_A,
                "bytes": 1,
                "path": "compile",
            },
            "executed_receiver_contract_id": (
                "tac.optimization.direct_description_carrier_compose.CarrierComposeReceiverV1.render_camera_pairs.v15"
            ),
        },
        "target_custody": {"identity": HEX_A},
        "g51_y0_y1_custody": {"identity": HEX_A},
        "stages": stage_bindings,
    }
    return receipt, stages


def _install_fake_loader(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    mutate_receipt: Any | None = None,
    mutate_stages: Any | None = None,
) -> Path:
    aggregate = root / "aggregate.json"
    aggregate.write_bytes(b"aggregate")
    receipt, stages = _fake_receipt(root)
    if mutate_receipt is not None:
        mutate_receipt(receipt)
    if mutate_stages is not None:
        mutate_stages(stages)
    loader = SimpleNamespace(
        receipt=receipt,
        iter_stages=lambda: tuple(stages),
    )
    monkeypatch.setattr(
        g87.MarginBaseScorerCacheLoaderV1,
        "open",
        lambda *_args, **_kwargs: loader,
    )
    monkeypatch.setattr(g87, "sha256_file", lambda _path: HEX_A)
    return aggregate


def test_exact_five_stage_mapping_retains_described_margin_and_executes_g72(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aggregate = _install_fake_loader(monkeypatch, tmp_path)
    opened = g87.open_g87_g78_to_g72_compile_input(
        aggregate,
        expected_file_sha256=HEX_A,
        expected_self_sha256=HEX_B,
    )

    assert len(opened.stages) == 5
    assert [stage.plan.stage_index for stage in opened.stages] == list(range(5))
    assert opened.receipt["closed_blockers"] == [
        FRESH_BATCH16_MARGIN_CUSTODY_OWED,
        FRESH_V15_BASE_SCORER_CACHE_OWED,
    ]
    assert opened.receipt["remaining_g72_blockers"] == list(REMAINING_G72_BLOCKERS)
    assert opened.receipt["additional_open_blockers"] == []
    assert opened.receipt["g72_compiler_source"]["path"] == str(g87.G72_COMPILER_SOURCE_PATH)
    assert opened.receipt["g72_compiler_source"]["sha256"]
    assert opened.receipt["proposal_enumeration_policy"] == {
        "minimum_component_sites": 1,
        "maximum_components_per_pair_role": 4096,
        "selection_thresholds_used": False,
        "cap_must_be_proven_nonbinding_before_materialization": True,
    }
    assert opened.receipt["selected_preimage_operand_emitted"] is False
    assert opened.stages[0].described_margins_f32.shape == (120, 384, 512)

    proposals = derive_v9_boundary_shearlet_stage_proposals(
        **opened.stages[0].g72_derivation_kwargs(),
        minimum_component_sites=g87.COMPLETE_MINIMUM_COMPONENT_SITES,
        maximum_components_per_pair_role=(g87.COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE),
    )
    assert proposals == ()
    census = g87.census_complete_g72_proposal_universe(opened)
    assert [row.proposal_count for row in census] == [0, 0, 0, 0, 0]


def test_wrong_aggregate_self_hash_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aggregate = _install_fake_loader(monkeypatch, tmp_path)
    with pytest.raises(
        g87.G87ExactCompileAdapterError,
        match="self hash differs",
    ):
        g87.open_g87_g78_to_g72_compile_input(
            aggregate,
            expected_file_sha256=HEX_A,
            expected_self_sha256="c" * 64,
        )


def test_unowned_blocker_closure_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aggregate = _install_fake_loader(
        monkeypatch,
        tmp_path,
        mutate_receipt=lambda value: value["closed_blockers"].append(
            "G72_FIVE_STAGE_EXACT_WHOLE_OBJECT_JOINT_ADMISSION_OWED"
        ),
    )
    with pytest.raises(
        g87.G87ExactCompileAdapterError,
        match="blocker closure differs",
    ):
        g87.open_g87_g78_to_g72_compile_input(
            aggregate,
            expected_file_sha256=HEX_A,
            expected_self_sha256=HEX_B,
        )


def test_nonbijective_stage_range_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aggregate = _install_fake_loader(
        monkeypatch,
        tmp_path,
        mutate_stages=lambda stages: setattr(
            stages[2],
            "pair_range",
            (239, 359),
        ),
    )
    with pytest.raises(
        g87.G87ExactCompileAdapterError,
        match="map bijectively",
    ):
        g87.open_g87_g78_to_g72_compile_input(
            aggregate,
            expected_file_sha256=HEX_A,
            expected_self_sha256=HEX_B,
        )


def test_writeable_dense_view_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def mutate(stages: list[Any]) -> None:
        stages[0].target_cells_u8 = np.zeros(
            (120, 384, 512),
            dtype=np.uint8,
        )

    aggregate = _install_fake_loader(
        monkeypatch,
        tmp_path,
        mutate_stages=mutate,
    )
    with pytest.raises(
        g87.G87ExactCompileAdapterError,
        match="read-only encoder-side view",
    ):
        g87.open_g87_g78_to_g72_compile_input(
            aggregate,
            expected_file_sha256=HEX_A,
            expected_self_sha256=HEX_B,
        )
