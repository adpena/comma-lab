# SPDX-License-Identifier: MIT
"""MLX-first to NumPy-portable contract helpers.

This module is deliberately small: it does not train, score, or export. It
records whether an MLX-local substrate has crossed each portability boundary:
MLX training state, NumPy-array export/bridge, archive payload, and inflate
runtime. That lets compact-substrate planners distinguish contest-compliant
Torch receivers from truly NumPy-only receivers without losing the signal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MLX_NUMPY_PORTABILITY_CONTRACT_SCHEMA = "mlx_numpy_portability_contract.v1"

FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
}


def build_mlx_numpy_portability_contract(
    *,
    substrate_id: str,
    training_backend: str = "mlx",
    exported_state_kind: str,
    archive_payload_kind: str,
    receiver_runtime_kind: str,
    receiver_dependencies: Sequence[str],
    numpy_array_export: bool,
    canonical_npz_bridge_required: bool = True,
    canonical_npz_bridge_used: bool = False,
    pure_numpy_inflate: bool,
    notes: str = "",
) -> dict[str, Any]:
    """Return a machine-readable MLX-to-NumPy portability contract.

    ``pure_numpy_inflate`` means the receiver can decode with NumPy plus the
    standard library only. A Torch receiver can still be contest-compliant, but
    it is not NumPy-portable and receives a portability blocker here.
    """

    sid = str(substrate_id).strip()
    if not sid:
        raise ValueError("substrate_id must be non-empty")
    deps = tuple(str(item).strip() for item in receiver_dependencies if str(item).strip())
    if not deps:
        raise ValueError("receiver_dependencies must not be empty")
    blockers: list[str] = []
    if str(training_backend) != "mlx":
        blockers.append("training_backend_not_mlx")
    if not bool(numpy_array_export):
        blockers.append("numpy_array_export_not_declared")
    if bool(canonical_npz_bridge_required) and not bool(canonical_npz_bridge_used):
        blockers.append("canonical_npz_bridge_not_used_or_not_applicable")
    non_numpy_deps = sorted(
        dep for dep in deps if dep not in {"numpy", "python_stdlib"}
    )
    if not bool(pure_numpy_inflate):
        blockers.append("inflate_runtime_not_pure_numpy")
    if bool(pure_numpy_inflate) and non_numpy_deps:
        blockers.append("pure_numpy_inflate_declared_with_non_numpy_dependencies")

    if not blockers:
        status = "pure_numpy_inflate_ready"
    elif bool(numpy_array_export):
        status = "numpy_export_bridge_ready_receiver_not_numpy"
    else:
        status = "not_numpy_portable"

    return {
        "schema": MLX_NUMPY_PORTABILITY_CONTRACT_SCHEMA,
        "substrate_id": sid,
        "training_backend": str(training_backend),
        "exported_state_kind": str(exported_state_kind),
        "archive_payload_kind": str(archive_payload_kind),
        "receiver_runtime_kind": str(receiver_runtime_kind),
        "receiver_dependencies": list(deps),
        "numpy_array_export": bool(numpy_array_export),
        "canonical_npz_bridge_required": bool(canonical_npz_bridge_required),
        "canonical_npz_bridge_used": bool(canonical_npz_bridge_used),
        "pure_numpy_inflate": bool(pure_numpy_inflate),
        "non_numpy_receiver_dependencies": non_numpy_deps,
        "portability_status": status,
        "portability_blockers": blockers,
        "notes": str(notes),
        **FALSE_AUTHORITY_FIELDS,
    }


__all__ = [
    "MLX_NUMPY_PORTABILITY_CONTRACT_SCHEMA",
    "build_mlx_numpy_portability_contract",
]
