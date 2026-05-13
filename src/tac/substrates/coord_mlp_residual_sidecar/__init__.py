"""H15 Coord-MLP residual sidecar substrate readiness surface.

The package provides a charged sidecar byte contract and a deterministic,
scorer-free apply helper. It is proxy-safe by construction: manifests keep
``score_claim=false`` and ``ready_for_exact_eval_dispatch=false`` until a
separate byte-closed archive/runtime custody path exists and exact eval
adjudicates the packet.
"""

from tac.substrates.coord_mlp_residual_sidecar.archive import (
    H15_LANE_ID,
    CoordMlpPatch,
    CoordMlpResidualSidecarError,
    CoordMlpResidualWeights,
    ParsedCoordMlpResidualSidecar,
    build_readiness_manifest,
    pack_sidecar,
    parse_sidecar,
)
from tac.substrates.coord_mlp_residual_sidecar.inflate import (
    CoordMlpInflateResult,
    apply_sidecar_to_rgb,
)

__all__ = [
    "H15_LANE_ID",
    "CoordMlpInflateResult",
    "CoordMlpPatch",
    "CoordMlpResidualSidecarError",
    "CoordMlpResidualWeights",
    "ParsedCoordMlpResidualSidecar",
    "apply_sidecar_to_rgb",
    "build_readiness_manifest",
    "pack_sidecar",
    "parse_sidecar",
]
