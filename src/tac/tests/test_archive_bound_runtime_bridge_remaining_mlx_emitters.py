# SPDX-License-Identifier: MIT
"""Archive-bound runtime bridge coverage for remaining MLX archive exporters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import numpy as np


class _FakeExportModel:
    def __init__(
        self,
        *,
        cfg: Any,
        state_dict: dict[str, torch.Tensor],
        selectors: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.selectors = selectors
        self._state_dict = state_dict

    def export_state_dict(self) -> dict[str, np.ndarray]:
        return {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in self._state_dict.items()
        }


def _load_package(output_dir: Path) -> dict[str, Any]:
    package_path = output_dir / "archive_bound_candidate_adapter_package.json"
    assert package_path.is_file()
    return json.loads(package_path.read_text(encoding="utf-8"))


def _assert_fail_closed_package(
    package: dict[str, Any],
    *,
    wrapper_schema: str,
    family: str,
    required_tags: set[str],
    forbidden_tags: set[str] | None = None,
) -> None:
    assert package["schema"] == wrapper_schema
    assert package["score_claim"] is False
    assert package["promotion_eligible"] is False
    assert package["ready_for_exact_eval_dispatch"] is False
    shared = package["archive_bound_candidate_adapter_package"]
    assert shared["candidate_family"] == family
    assert shared["ready_contract_count"] == 0
    assert shared["receiver_proof_gate_passed_count"] == 0
    contract = shared["candidate_rows"][0]["archive_bound_candidate_contract"]
    assert contract["archive_bound_candidate_ready_for_exact_handoff"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert required_tags <= set(contract["archive_substrate_tags"])
    if forbidden_tags:
        assert set(contract["archive_substrate_tags"]).isdisjoint(forbidden_tags)


def test_z5_export_emits_shared_archive_bound_package_fail_closed(tmp_path: Path) -> None:
    from tac.substrates.time_traveler_l5_z5.architecture import (
        Z5RaoBallardConfig,
        Z5RaoBallardSubstrate,
    )
    from tac.substrates.time_traveler_l5_z5.archive_candidate import (
        Z5_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        Z5_MLX_ARCHIVE_CANDIDATE_FAMILY,
        export_z5_mlx_archive,
    )

    cfg = Z5RaoBallardConfig(num_pairs=1, output_height=8, output_width=8)
    torch.manual_seed(5)
    model = Z5RaoBallardSubstrate(cfg)
    fake = _FakeExportModel(cfg=cfg, state_dict=model.state_dict())

    archive_zip, sha, size = export_z5_mlx_archive(fake, tmp_path / "z5_export")

    assert archive_zip.is_file()
    assert len(sha) == 64
    assert size == archive_zip.stat().st_size
    package = _load_package(archive_zip.parent)
    _assert_fail_closed_package(
        package,
        wrapper_schema=Z5_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        family=Z5_MLX_ARCHIVE_CANDIDATE_FAMILY,
        required_tags={"neural_archive", "mlx_substrate", "predictive_coding", "z5"},
        forbidden_tags={"z7_mamba2"},
    )


def test_selector_v2_export_emits_shared_archive_bound_package_fail_closed(
    tmp_path: Path,
) -> None:
    from tac.substrates.pact_nerv_selector_v2.architecture import (
        PactNervSelectorV2Config,
        PactNervSelectorV2Substrate,
    )
    from tac.substrates.pact_nerv_selector_v2.archive_candidate import (
        PACT_NERV_SELECTOR_V2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        PACT_NERV_SELECTOR_V2_MLX_ARCHIVE_CANDIDATE_FAMILY,
        export_pact_nerv_selector_v2_mlx_archive,
    )

    cfg = PactNervSelectorV2Config(num_pairs=1, output_height=8, output_width=8)
    torch.manual_seed(6)
    model = PactNervSelectorV2Substrate(cfg)
    fake = _FakeExportModel(cfg=cfg, state_dict=model.state_dict())

    archive_zip, _, _ = export_pact_nerv_selector_v2_mlx_archive(
        fake,
        tmp_path / "psv2_export",
    )

    package = _load_package(archive_zip.parent)
    _assert_fail_closed_package(
        package,
        wrapper_schema=PACT_NERV_SELECTOR_V2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        family=PACT_NERV_SELECTOR_V2_MLX_ARCHIVE_CANDIDATE_FAMILY,
        required_tags={"selector", "neural_archive", "mlx_substrate", "pact_nerv"},
        forbidden_tags={"predictive_coding", "z7_mamba2"},
    )


def test_selector_v3_export_emits_shared_archive_bound_package_fail_closed(
    tmp_path: Path,
) -> None:
    from tac.substrates.pact_nerv_selector_v3.architecture import (
        PactNervSelectorV3Config,
        PactNervSelectorV3Substrate,
    )
    from tac.substrates.pact_nerv_selector_v3.archive_candidate import (
        PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_CANDIDATE_FAMILY,
        export_pact_nerv_selector_v3_mlx_archive,
    )

    cfg = PactNervSelectorV3Config(num_pairs=1, output_height=8, output_width=8)
    torch.manual_seed(7)
    model = PactNervSelectorV3Substrate(cfg)
    fake = _FakeExportModel(cfg=cfg, state_dict=model.state_dict())

    archive_zip, _, _ = export_pact_nerv_selector_v3_mlx_archive(
        fake,
        tmp_path / "psv3_export",
    )

    package = _load_package(archive_zip.parent)
    _assert_fail_closed_package(
        package,
        wrapper_schema=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        family=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_CANDIDATE_FAMILY,
        required_tags={"selector", "neural_archive", "mlx_substrate", "pact_nerv"},
        forbidden_tags={"predictive_coding", "z7_mamba2"},
    )


def test_selector_v4_export_emits_shared_archive_bound_package_fail_closed(
    tmp_path: Path,
) -> None:
    from tac.substrates.pact_nerv_selector_v4.architecture import (
        PactNervSelectorV4Config,
        PactNervSelectorV4Substrate,
    )
    from tac.substrates.pact_nerv_selector_v4.archive_candidate import (
        PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_CANDIDATE_FAMILY,
        export_pact_nerv_selector_v4_mlx_archive,
    )

    cfg = PactNervSelectorV4Config(num_pairs=1, output_height=8, output_width=8)
    torch.manual_seed(8)
    model = PactNervSelectorV4Substrate(cfg)
    fake = _FakeExportModel(cfg=cfg, state_dict=model.state_dict())

    archive_zip, _, _ = export_pact_nerv_selector_v4_mlx_archive(
        fake,
        tmp_path / "psv4_export",
    )

    package = _load_package(archive_zip.parent)
    _assert_fail_closed_package(
        package,
        wrapper_schema=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        family=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_CANDIDATE_FAMILY,
        required_tags={"selector", "neural_archive", "mlx_substrate", "pact_nerv"},
        forbidden_tags={"predictive_coding", "z7_mamba2"},
    )


def test_pact_nerv_vq_export_emits_shared_archive_bound_package_fail_closed(
    tmp_path: Path,
) -> None:
    from tac.substrates.pact_nerv_vq.architecture import (
        PactNervVqConfig,
        PactNervVqSubstrate,
    )
    from tac.substrates.pact_nerv_vq.archive_candidate import (
        PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        PACT_NERV_VQ_MLX_ARCHIVE_CANDIDATE_FAMILY,
        export_pact_nerv_vq_mlx_archive,
    )

    cfg = PactNervVqConfig(
        num_pairs=1,
        output_height=8,
        output_width=8,
        codebook_size=8,
    )
    torch.manual_seed(9)
    model = PactNervVqSubstrate(cfg)
    fake = _FakeExportModel(cfg=cfg, state_dict=model.state_dict())

    archive_zip, _, _ = export_pact_nerv_vq_mlx_archive(
        fake,
        tmp_path / "pvq_export",
    )

    package = _load_package(archive_zip.parent)
    _assert_fail_closed_package(
        package,
        wrapper_schema=PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        family=PACT_NERV_VQ_MLX_ARCHIVE_CANDIDATE_FAMILY,
        required_tags={"neural_archive", "mlx_substrate", "pact_nerv", "vq"},
        forbidden_tags={"predictive_coding", "z7_mamba2"},
    )
