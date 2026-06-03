# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tac.substrates.hi_nerv.official_grid import (
    HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF,
    OfficialGridTrilinear3D,
    OfficialGridTrilinear3DError,
    official_grid_trilinear3d_forward,
)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("t_in,t_out", [(2, 5), (5, 2), (3, 3), (1, 4)])
def test_official_grid_trilinear3d_matches_torch_interpolate(
    *,
    align_corners: bool,
    t_in: int,
    t_out: int,
) -> None:
    rng = np.random.default_rng(1403 + t_in * 17 + t_out)
    x = rng.normal(size=(t_in, 2, 3, 4)).astype(np.float32)

    actual = official_grid_trilinear3d_forward(
        x,
        output_size=(t_out, 2, 3),
        align_corners=align_corners,
    )

    torch_in = torch.from_numpy(x).reshape(1, 1, t_in, 2 * 3 * 4)
    expected = F.interpolate(
        torch_in,
        size=(t_out, 2 * 3 * 4),
        mode="bilinear",
        align_corners=align_corners,
    ).reshape(t_out, 2, 3, 4)
    np.testing.assert_allclose(
        actual,
        expected.detach().cpu().numpy(),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_official_grid_trilinear3d_rejects_spatial_resize() -> None:
    x = np.zeros((2, 3, 4, 1), dtype=np.float32)

    with pytest.raises(OfficialGridTrilinear3DError, match="only supports temporal"):
        official_grid_trilinear3d_forward(x, output_size=(4, 6, 4))


def test_official_grid_trilinear3d_contract_is_false_authority() -> None:
    layer = OfficialGridTrilinear3D(output_size=(4, 2, 3), align_corners=False)
    contract = layer.as_jsonable_contract()

    assert HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF in contract["proof_marker"]
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
