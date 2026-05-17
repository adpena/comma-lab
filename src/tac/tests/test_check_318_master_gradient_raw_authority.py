# SPDX-License-Identifier: MIT
from __future__ import annotations

import inspect

import pytest

import tac.preflight as preflight_module
from tac.preflight import (
    PreflightError,
    check_master_gradient_raw_byte_authority_not_landed,
)


def test_check_318_blocks_raw_archive_byte_master_gradient_authority(tmp_path) -> None:
    target = tmp_path / "src" / "tac" / "master_gradient.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "\n".join(
            [
                "# SPDX-License-Identifier: MIT",
                "from collections.abc import Mapping",
                "",
                "class MasterGradient:",
                "    gradient_array_path = 'gradient.npy'",
                "    measurement_method = 'finite_difference_bit_flip'",
                "",
                "    def predict_delta_s(",
                "        self, byte_modifications: Mapping[int, float]",
                "    ) -> float:",
                "        '''Predict from an (N_archive_bytes, 3) sidecar.'''",
                "        return 0.0",
            ]
        )
        + "\n"
    )

    violations = check_master_gradient_raw_byte_authority_not_landed(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "raw_bit_or_byte_finite_difference_method" in violations[0]
    assert "raw_archive_byte_response_tensor" in violations[0]
    assert "CandidateModificationSpec" in violations[0]
    with pytest.raises(PreflightError, match="Catalog #318"):
        check_master_gradient_raw_byte_authority_not_landed(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_check_318_allows_operator_response_route_tokens(tmp_path) -> None:
    target = tmp_path / "tools" / "extract_master_gradient.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "\n".join(
            [
                "# SPDX-License-Identifier: MIT",
                "from tac.master_gradient_operator_plan import CandidateModificationSpec",
                "",
                "MUTATION_GRAIN = 'grammar_aware_operator'",
                "",
                "def build_rows():",
                "    return []",
            ]
        )
        + "\n"
    )

    assert (
        check_master_gradient_raw_byte_authority_not_landed(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_check_318_waiver_requires_real_rationale(tmp_path) -> None:
    target = tmp_path / "src" / "tac" / "master_gradient.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "# MASTER_GRADIENT_RAW_AUTHORITY_OK:<rationale>\n"
        "measurement_method = 'finite_difference_bit_flip'\n"
    )
    assert check_master_gradient_raw_byte_authority_not_landed(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    target.write_text(
        "# MASTER_GRADIENT_RAW_AUTHORITY_OK: diagnostic-only fixture, no score authority\n"
        "measurement_method = 'finite_difference_bit_flip'\n"
    )
    assert (
        check_master_gradient_raw_byte_authority_not_landed(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_check_318_is_wired_into_preflight_all_strict() -> None:
    source = inspect.getsource(preflight_module.preflight_all)
    idx = source.find("check_master_gradient_raw_byte_authority_not_landed")
    assert idx >= 0, "Catalog #318 must be wired into preflight_all"
    call_window = source[idx : idx + 180]
    assert "strict=True" in call_window
