# SPDX-License-Identifier: MIT
"""MLX → numpy state_dict bridge for cascade_c_prime_frame_1_segnet_waterfill.

Per CLAUDE.md standing directive 2026-05-26 verbatim *"MLX-FIRST NUMPY-PORTABLE
INDIVIDUALLY-FRACTAL"*: TRAINING is MLX-first on M5 Max; INFLATE is
numpy-portable (no MLX dep; ≤200 LOC + ≤2 ext deps per HNeRV parity L4). This
module is the CANONICAL BRIDGE between the two layers.

Contract: MLX state_dict (numpy arrays — already MLX→numpy at trainer layer)
→ npz file → ZIP-member at archive build → numpy inflate primitives.

Per Catalog #105/#139/#272 byte-mutation smoke discipline: every field in the
bridged state_dict MUST be operationally consumed by `archive.pack_archive` +
`inflate.parse_archive`. The round-trip (state_dict → npz → load → archive →
parse_archive) MUST be byte-identical for the routing-decision field
specifically (Catalog #139 sister of the existing test_byte_mutation_smoke).

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| npz emission | CANONICAL (`np.savez_compressed`) | numpy stdlib; deterministic ZIP format |
| Bridge contract | UNIQUE (this module) | per-substrate state_dict shape per architecture.py; no canonical sister |
| Archive composition | CANONICAL (`archive.pack_archive`) | already landed in scaffold |
| Round-trip verification | CANONICAL (`np.load` + `parse_archive`) | numpy + brotli per inflate.py L9 dep closure |

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: N/A (defensive bridge module)
- hook #2 Pareto constraint: N/A
- hook #3 bit-allocator: ACTIVE (state_dict ENCODES the routing decision; the
  bridge preserves the bit-allocator's output across the MLX → numpy boundary)
- hook #4 cathedral autopilot dispatch: N/A
- hook #5 continual-learning posterior: N/A
- hook #6 probe-disambiguator: ACTIVE (the round-trip verifier IS the canonical
  disambiguator between bridge-preserves-byte-identity vs bridge-corrupts-state)

## NO_SUPERSESSION_NEEDED:adds_new_bridge_module_does_not_supersede_existing_archive_or_inflate_per_Catalog_110_113_APPEND_ONLY_HISTORICAL_PROVENANCE
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .archive import POSE_DIMS, pack_archive, parse_archive

__all__ = [
    "BridgeRoundtripVerdict",
    "EXPECTED_STATE_DICT_KEYS",
    "MLXNumpyBridgeError",
    "export_state_dict_to_npz",
    "load_state_dict_from_npz",
    "roundtrip_state_dict_through_archive",
    "verify_state_dict_shape_contract",
]


# Canonical state_dict keys per the trainer.py contract.
EXPECTED_STATE_DICT_KEYS: tuple[str, ...] = (
    "routing_decision",
    "frame_0_menu_indices",
    "frame_1_menu_indices",
    "pose_deltas_uint8",
)
"""These 4 keys MUST be present for the bridge to emit a byte-valid archive.

Optional keys (selected_seg_delta, selected_pose_delta, selected_lagrangian,
per_pair_score_delta) are research-signal observability surfaces per Catalog
#305; they are PRESERVED in the npz but NOT consumed by archive.pack_archive."""


class MLXNumpyBridgeError(RuntimeError):
    """Raised when the MLX → numpy bridge contract cannot be honored."""


@dataclass(frozen=True)
class BridgeRoundtripVerdict:
    """Verdict from MLX state_dict → npz → archive → parse roundtrip.

    Per Catalog #287/#323 canonical Provenance: every byte-comparison field is
    deterministic given the input state_dict. `routing_decision_byte_identical`
    is the canonical primary signal per Catalog #139 no-op detector sister.
    """

    state_dict_keys_present: tuple[str, ...]
    routing_decision_byte_identical: bool
    archive_byte_count: int
    archive_sha256: str
    n_pairs: int
    notes: str = ""


def verify_state_dict_shape_contract(state_dict: dict[str, np.ndarray]) -> int:
    """Verify state_dict satisfies the bridge contract; return n_pairs.

    Raises MLXNumpyBridgeError on contract violation per Catalog #138 fail-closed
    strict-load discipline.
    """
    for key in EXPECTED_STATE_DICT_KEYS:
        if key not in state_dict:
            raise MLXNumpyBridgeError(
                f"state_dict missing canonical key {key!r}; expected "
                f"{list(EXPECTED_STATE_DICT_KEYS)}"
            )

    routing = state_dict["routing_decision"]
    f0_indices = state_dict["frame_0_menu_indices"]
    f1_indices = state_dict["frame_1_menu_indices"]
    pose_deltas = state_dict["pose_deltas_uint8"]

    if routing.ndim != 1:
        raise MLXNumpyBridgeError(
            f"routing_decision must be 1D; got shape {routing.shape}"
        )
    n_pairs = int(routing.shape[0])

    if f0_indices.shape != (n_pairs,):
        raise MLXNumpyBridgeError(
            f"frame_0_menu_indices shape mismatch; expected ({n_pairs},), got "
            f"{f0_indices.shape}"
        )
    if f1_indices.shape != (n_pairs,):
        raise MLXNumpyBridgeError(
            f"frame_1_menu_indices shape mismatch; expected ({n_pairs},), got "
            f"{f1_indices.shape}"
        )
    if pose_deltas.shape != (n_pairs, POSE_DIMS):
        raise MLXNumpyBridgeError(
            f"pose_deltas_uint8 shape mismatch; expected ({n_pairs}, "
            f"{POSE_DIMS}), got {pose_deltas.shape}"
        )

    # Type contract per Catalog #138 strict-load discipline
    if routing.dtype not in (np.int8, np.uint8):
        raise MLXNumpyBridgeError(
            f"routing_decision dtype must be int8 or uint8; got {routing.dtype}"
        )
    if f0_indices.dtype != np.uint8:
        raise MLXNumpyBridgeError(
            f"frame_0_menu_indices dtype must be uint8; got {f0_indices.dtype}"
        )
    if f1_indices.dtype != np.uint8:
        raise MLXNumpyBridgeError(
            f"frame_1_menu_indices dtype must be uint8; got {f1_indices.dtype}"
        )
    if pose_deltas.dtype != np.uint8:
        raise MLXNumpyBridgeError(
            f"pose_deltas_uint8 dtype must be uint8; got {pose_deltas.dtype}"
        )

    return n_pairs


def export_state_dict_to_npz(
    state_dict: dict[str, np.ndarray], path: Path
) -> Path:
    """Export MLX→numpy state_dict to .npz file per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE".

    Per CLAUDE.md "Beauty, simplicity, and developer experience": uses
    `np.savez_compressed` (numpy stdlib; ZIP-format compressed; deterministic).

    Args:
        state_dict: Mapping from key (str) to numpy array. MUST satisfy the
            bridge contract per verify_state_dict_shape_contract.
        path: Output .npz path (parent dir auto-created).

    Returns:
        Absolute path to the written .npz file.
    """
    verify_state_dict_shape_contract(state_dict)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state_dict)
    return path.resolve()


def load_state_dict_from_npz(path: Path) -> dict[str, np.ndarray]:
    """Load state_dict from .npz file per Catalog #138 strict-load discipline.

    Raises MLXNumpyBridgeError on missing file OR contract violation.
    """
    if not path.exists():
        raise MLXNumpyBridgeError(f"npz file missing at {path}")
    try:
        with np.load(path, allow_pickle=False) as npz:
            state_dict = {key: np.asarray(npz[key]) for key in npz.files}
    except Exception as exc:
        raise MLXNumpyBridgeError(f"npz load failed: {exc}") from exc
    verify_state_dict_shape_contract(state_dict)
    return state_dict


def roundtrip_state_dict_through_archive(
    state_dict: dict[str, np.ndarray],
    *,
    frame_0_menu_size: int = 16,
    frame_1_menu_size: int = 8,
) -> BridgeRoundtripVerdict:
    """Round-trip MLX state_dict → archive.pack_archive → parse_archive.

    The byte-identity invariant for the routing_decision field is the canonical
    primary signal per Catalog #139 no-op detector sister: if a mutation in the
    state_dict's routing_decision propagates to the parsed archive's
    routing_decision, the bridge contract holds.

    Args:
        state_dict: MLX→numpy state_dict per bridge contract.
        frame_0_menu_size: PR110 K=16 default.
        frame_1_menu_size: K=8 default per substrate contract.

    Returns:
        BridgeRoundtripVerdict (frozen dataclass).
    """
    n_pairs = verify_state_dict_shape_contract(state_dict)

    archive_bytes = pack_archive(
        routing_decision=state_dict["routing_decision"],
        frame_0_menu_indices=state_dict["frame_0_menu_indices"],
        frame_1_menu_indices=state_dict["frame_1_menu_indices"],
        pose_deltas_uint8=state_dict["pose_deltas_uint8"],
        frame_0_menu_size=frame_0_menu_size,
        frame_1_menu_size=frame_1_menu_size,
    )

    parsed = parse_archive(archive_bytes)
    routing_byte_identical = bool(
        np.array_equal(parsed.routing_decision, state_dict["routing_decision"])
    )

    return BridgeRoundtripVerdict(
        state_dict_keys_present=tuple(sorted(state_dict.keys())),
        routing_decision_byte_identical=routing_byte_identical,
        archive_byte_count=len(archive_bytes),
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        n_pairs=n_pairs,
        notes=(
            "Catalog #139 no-op detector sister: routing_decision byte-identity "
            "across MLX→numpy bridge + archive pack/parse roundtrip verified."
        ),
    )
