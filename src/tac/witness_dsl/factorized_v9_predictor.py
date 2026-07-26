# SPDX-License-Identifier: MIT
"""Identity-bound semantic decoder for the original factorized V9 program.

The counted program is an existing receiver-closed DDM V9 carrier archive.  Its
Road/Undrivable bulk, Lane chart, Movable topology/worldsheet, MyCar static
component, and shared temporal/Pose6 state are parsed by the production V9
receiver.  This adapter exposes the *semantic partition* those factors describe
without calling SegNet and without storing or inferring a ground-truth table.

The semantic rule is deliberately small and explicit: start from the
Undrivable complement, then paint the five decoded factor masks in canonical
V9 realization order.  A PBR1 residual can therefore bind the exact archive
bytes, the declared interpreter-source set, and the exact decoded class
stream instead of relying on a caller-attested reference frame.  The decoded
stream digest is the final reference authority; the declared source-set digest
separately names the local interpreter/wire implementation.

This is a source-model interface, not a score claim. PBR1 may measure exact
debt and conditional entropy against these cells, but its lossless target-event
bytes are an encoder-only teacher and MUST NOT enter a candidate. Candidate
economics and evaluator equivalence belong to a non-exhaustive generative
correction `G` realized and evaluated through the frozen contest path.

This contract does not consume or reopen the deferred JRD coefficient-prefix
probe or the killed kinetic-Laguerre formulation.  It only interprets the
unaffected, already receiver-closed V9 factor primitives present in its counted
program bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    RECEIVER_SCHEMA,
    ROLE_CLASS_IDS,
    CarrierComposeReceiverV1,
    compile_carrier_compose_archive,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_entropy_priced_member import (
    StructuredS4SourcesV1,
    compile_composed_structured_member_archive,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.witness_dsl.predictor_bound_residual import (
    build_predictor_bound_partition_residual,
)

PREDICTOR_CONTRACT_ID: Final = "tac.factorized_v9_semantic_predictor.v1"
SOURCE_MANIFEST_SCHEMA: Final = "tac.factorized_v9_renderer_source_manifest.v1"
SEMANTIC_BINDING_SCHEMA: Final = "tac.factorized_v9_semantic_binding.v1"
SEMANTIC_HEIGHT: Final = 384
SEMANTIC_WIDTH: Final = 512
UNDRIVABLE_CLASS_ID: Final = ROLE_CLASS_IDS["UndrivableBoundary"]

# Declared repository modules that own program compilation, parsing, PBR
# binding, or factor-mask semantics.  Exact decoded bytes are bound separately,
# so a deeper dependency drift that changes output still fails PBR1 identity.
RENDERER_SOURCE_PATHS: Final = (
    "src/tac/witness_dsl/factorized_v9_predictor.py",
    "src/tac/optimization/direct_description_carrier_compose.py",
    "src/tac/optimization/direct_description_entropy_priced_member.py",
    "src/tac/optimization/direct_description_entropy_streams.py",
    "src/tac/optimization/direct_description_measurement_ladder.py",
    "src/tac/optimization/direct_description_g1_worldsheet.py",
    "src/tac/optimization/predictor_upgrade_xi_chart.py",
    "src/tac/optimization/direct_description_minimizer.py",
    "src/tac/optimization/direct_description_polytope_membership.py",
    "src/tac/optimization/direct_description_receiver_priced_member.py",
    "src/tac/optimization/predictor_r3_causal.py",
    "src/tac/optimization/s4_archive_composer.py",
    "src/tac/witness_dsl/predictor_bound_residual.py",
)


class FactorizedV9PredictorError(ValueError):
    """Fail-closed V9 program, source-custody, or semantic-decode error."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FactorizedV9PredictorError("renderer source manifest is not canonical JSON") from exc


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise FactorizedV9PredictorError(f"{label} must be a lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise FactorizedV9PredictorError(f"{label} must be a lowercase SHA-256 digest") from exc
    if len(decoded) != 32:
        raise FactorizedV9PredictorError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _repository_root(repository_root: Path | None) -> Path:
    root = (
        Path(__file__).resolve(strict=True).parents[3]
        if repository_root is None
        else Path(repository_root).resolve(strict=True)
    )
    if not root.is_dir():
        raise FactorizedV9PredictorError("repository root is not a directory")
    return root


def renderer_source_manifest(repository_root: Path | None = None) -> dict[str, Any]:
    """Return the deterministic declared-source identity of the interpreter."""

    root = _repository_root(repository_root)
    rows: list[dict[str, Any]] = []
    for relative in RENDERER_SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FactorizedV9PredictorError(f"renderer source is missing, non-regular, or symlinked: {relative}")
        payload = path.read_bytes()
        rows.append({"path": relative, "bytes": len(payload), "sha256": _sha256(payload)})
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "predictor_contract_id": PREDICTOR_CONTRACT_ID,
        "semantic_geometry": [SEMANTIC_HEIGHT, SEMANTIC_WIDTH],
        "semantic_paint_order": list(REALIZATION_PAINT_ORDER),
        "default_complement_class_id": UNDRIVABLE_CLASS_ID,
        "source_identity_scope": "declared_semantic_interpreter_and_wire_modules.v1",
        "sources": rows,
    }


def renderer_source_sha256(repository_root: Path | None = None) -> str:
    """Hash the full source closure used as PBR1 renderer identity."""

    return _sha256(_canonical_json(renderer_source_manifest(repository_root)))


def _read_regular_file_once(path: Path) -> bytes:
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise FactorizedV9PredictorError(f"cannot open predictor program: {target}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise FactorizedV9PredictorError("predictor program must be a regular file")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise FactorizedV9PredictorError("predictor program was truncated during its single read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise FactorizedV9PredictorError("predictor program grew during its single read")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if not payload:
        raise FactorizedV9PredictorError("predictor program must be non-empty")
    return payload


def _receive_exact_v9(program: bytes) -> CarrierComposeReceiverV1:
    try:
        receiver = receive_carrier_compose_archive(program, verify_member_effects=True)
    except DirectDescriptionError as exc:
        raise FactorizedV9PredictorError("program is not a strict receiver-closed V9 carrier archive") from exc
    if receiver.custody.get("schema") != RECEIVER_SCHEMA:
        raise FactorizedV9PredictorError("predictor program must use the exact V9 factor grammar")
    required_false = (
        "scorer_weights_present",
        "ground_truth_argmax_present",
        "decode_scorer_dependency",
        "pixel_coordinate_or_rgb_patch_present",
    )
    if any(receiver.custody.get(field) is not False for field in required_false):
        raise FactorizedV9PredictorError("V9 predictor admitted a forbidden scorer, target-table, or pixel payload")
    if (
        receiver.custody.get("all_five_roles_consumed") is not True
        or receiver.custody.get("nested_pose6_owner_reused") is not True
    ):
        raise FactorizedV9PredictorError("V9 predictor did not bind all five roles and its counted temporal state")
    return receiver


@dataclass(frozen=True, slots=True)
class FactorizedV9SemanticReceiver:
    """One parsed V9 program plus its exact semantic/source identities."""

    program: bytes
    receiver: CarrierComposeReceiverV1
    source_manifest: Mapping[str, Any]
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.program, bytes) or not self.program:
            raise FactorizedV9PredictorError("receiver program must be non-empty counted bytes")
        if self.receiver.archive != self.program or self.receiver.custody.get("archive_sha256") != _sha256(
            self.program
        ):
            raise FactorizedV9PredictorError("admission receiver does not belong to the exact predictor program")
        if _sha256(_canonical_json(self.source_manifest)) != self.source_manifest_sha256:
            raise FactorizedV9PredictorError("renderer source manifest identity mismatch")
        if (
            self.source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
            or self.source_manifest.get("predictor_contract_id") != PREDICTOR_CONTRACT_ID
        ):
            raise FactorizedV9PredictorError("renderer source manifest contract mismatch")

    def _fresh_receiver(self) -> CarrierComposeReceiverV1:
        if _sha256(_canonical_json(self.source_manifest)) != self.source_manifest_sha256:
            raise FactorizedV9PredictorError("renderer source manifest changed after admission")
        fresh = _receive_exact_v9(self.program)
        if int(fresh.z.n_pairs) != self.pair_count or int(fresh.predictor.source_pair_start) != self.source_pair_start:
            raise FactorizedV9PredictorError("fresh program decode changed the admitted pair window")
        return fresh

    @property
    def pair_count(self) -> int:
        return int(self.receiver.z.n_pairs)

    @property
    def source_pair_start(self) -> int:
        return int(self.receiver.predictor.source_pair_start)

    @property
    def source_pair_ids(self) -> tuple[int, ...]:
        return tuple(range(self.source_pair_start, self.source_pair_start + self.pair_count))

    @property
    def program_sha256(self) -> str:
        return _sha256(self.program)

    def _decode_semantic_pairs_from(
        self,
        receiver: CarrierComposeReceiverV1,
        pair_ids: Sequence[int],
    ) -> np.ndarray:
        indexes = tuple(pair_ids)
        if (
            not indexes
            or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in indexes)
            or len({int(value) for value in indexes}) != len(indexes)
        ):
            raise FactorizedV9PredictorError("pair_ids must be a non-empty unique integer sequence")
        local_ids = tuple(int(value) for value in indexes)
        if min(local_ids) < 0 or max(local_ids) >= self.pair_count:
            raise FactorizedV9PredictorError("pair_ids escape the predictor program window")

        layers = {layer.role: layer for layer in receiver.layers}
        if set(layers) != set(REALIZATION_PAINT_ORDER):
            raise FactorizedV9PredictorError("predictor does not expose exactly one layer for every semantic role")
        if {role: int(layer.class_id) for role, layer in layers.items()} != ROLE_CLASS_IDS:
            raise FactorizedV9PredictorError("predictor role/class identity differs from the canonical five classes")

        labels = np.full(
            (len(local_ids), SEMANTIC_HEIGHT, SEMANTIC_WIDTH),
            UNDRIVABLE_CLASS_ID,
            dtype=np.uint8,
        )
        for role in REALIZATION_PAINT_ORDER:
            layer = layers[role]
            for output_index, pair_id in enumerate(local_ids):
                # This adapter intentionally shares the production receiver's
                # factor-mask contract instead of maintaining a second decoder.
                mask = receiver._mask_for_layer(
                    layer,
                    pair_id,
                    replace_g1_movable=True,
                )
                if mask.shape != (SEMANTIC_HEIGHT, SEMANTIC_WIDTH) or mask.dtype != np.bool_:
                    raise FactorizedV9PredictorError(f"{role} factor mask changed semantic geometry or dtype")
                labels[output_index, mask] = np.uint8(layer.class_id)
        if int(labels.min()) < 0 or int(labels.max()) > 4:
            raise FactorizedV9PredictorError("decoded semantic classes escaped [0,4]")
        return np.ascontiguousarray(labels)

    def decode_semantic_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        """Decode local pair IDs after strictly reopening the exact program.

        ``pair_ids`` are local program coordinates.  No scorer, target cache,
        caller-provided labels, RGB nearest-neighbour classifier, or mutable
        cached receiver state is used.
        """

        return self._decode_semantic_pairs_from(self._fresh_receiver(), pair_ids)

    def decode_all_semantics(self, *, batch_size: int = 16) -> np.ndarray:
        """Decode the complete program in bounded deterministic batches."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise FactorizedV9PredictorError("batch_size must be a positive integer")
        fresh = self._fresh_receiver()
        pair_count = int(fresh.z.n_pairs)
        output = np.empty(
            (pair_count, SEMANTIC_HEIGHT, SEMANTIC_WIDTH),
            dtype=np.uint8,
        )
        for start in range(0, pair_count, batch_size):
            stop = min(pair_count, start + batch_size)
            output[start:stop] = self._decode_semantic_pairs_from(fresh, tuple(range(start, stop)))
        return np.ascontiguousarray(output)

    def semantic_binding(self, labels: np.ndarray | None = None) -> dict[str, Any]:
        """Bind exact program, declared source set, and decoded semantic bytes."""

        value = self.decode_all_semantics() if labels is None else np.asarray(labels)
        if value.dtype != np.uint8 or value.shape != (
            self.pair_count,
            SEMANTIC_HEIGHT,
            SEMANTIC_WIDTH,
        ):
            raise FactorizedV9PredictorError("semantic binding requires this receiver's complete uint8 stream")
        value = np.ascontiguousarray(value)
        replay = self.decode_all_semantics()
        if not np.array_equal(value, replay):
            raise FactorizedV9PredictorError("caller-provided semantic stream differs from fresh program decode")
        fresh = self._fresh_receiver()
        pose6_codes = np.ascontiguousarray(fresh.pose6_codes)
        return {
            "schema": SEMANTIC_BINDING_SCHEMA,
            "predictor_contract_id": PREDICTOR_CONTRACT_ID,
            "predictor_program_bytes": len(self.program),
            "predictor_program_sha256": self.program_sha256,
            "predictor_renderer_sha256": self.source_manifest_sha256,
            "predictor_semantic_bytes": int(value.size),
            "predictor_semantic_sha256": _sha256(memoryview(value).cast("B")),
            "semantic_geometry": list(value.shape),
            "semantic_class_ids": list(range(5)),
            "semantic_paint_order": list(REALIZATION_PAINT_ORDER),
            "default_complement_class_id": UNDRIVABLE_CLASS_ID,
            "source_pair_start": self.source_pair_start,
            "source_pair_stop_exclusive": self.source_pair_start + self.pair_count,
            "temporal_pose6_shape": list(pose6_codes.shape),
            "temporal_pose6_sha256": _sha256(memoryview(pose6_codes).cast("B")),
            "all_five_factor_roles_consumed": True,
            "counted_temporal_pose6_bound": bool(fresh.custody.get("nested_pose6_owner_reused")),
            "target_table_bytes": 0,
            "decode_scorer_dependency": False,
            "score_claim": False,
            "promotion_eligible": False,
        }

    def build_pbr1(self, target_labels: np.ndarray) -> bytes:
        """Recompute a PBR1 syndrome against this exact decoded predictor."""

        predictor = self.decode_all_semantics()
        return build_predictor_bound_partition_residual(
            predictor_program=self.program,
            predictor_contract_id=PREDICTOR_CONTRACT_ID,
            predictor_renderer_sha256=self.source_manifest_sha256,
            predictor_labels=predictor,
            target_labels=target_labels,
        )


def receive_factorized_v9_predictor(
    program: bytes,
    *,
    repository_root: Path | None = None,
) -> FactorizedV9SemanticReceiver:
    """Strictly parse counted V9 bytes and bind their semantic interpreter."""

    if not isinstance(program, bytes) or not program:
        raise FactorizedV9PredictorError("program must be non-empty counted bytes")
    receiver = _receive_exact_v9(program)
    manifest = renderer_source_manifest(repository_root)
    manifest_sha = _sha256(_canonical_json(manifest))
    value = FactorizedV9SemanticReceiver(
        program=program,
        receiver=receiver,
        source_manifest=manifest,
        source_manifest_sha256=manifest_sha,
    )
    # Admission performs a real factor decode; a valid archive that cannot
    # produce the semantic contract is not admitted as a predictor program.
    value.decode_semantic_pairs((0,))
    return value


def compile_factorized_v9_predictor(
    baseline_program: bytes,
    sources: StructuredS4SourcesV1,
    *,
    pair_start: int = 0,
    repository_root: Path | None = None,
) -> FactorizedV9SemanticReceiver:
    """Compile typed factor sources into one exact counted V9 program.

    The existing DDM compilers own the wire grammar.  This seam deliberately
    adds no second factor encoding, learned/public payload, target-label table,
    or implicit caller-attested semantic stream. Input-lineage custody remains
    the caller's responsibility and is not inferred from the source arrays.
    """

    if not isinstance(baseline_program, bytes) or not baseline_program:
        raise FactorizedV9PredictorError("baseline_program must be non-empty counted bytes")
    try:
        structured_program = compile_composed_structured_member_archive(
            baseline_program,
            sources,
            pair_start=pair_start,
        )[0]
        program = compile_carrier_compose_archive(structured_program)[0]
    except DirectDescriptionError as exc:
        raise FactorizedV9PredictorError("typed V9 factor compilation failed") from exc
    return receive_factorized_v9_predictor(program, repository_root=repository_root)


def load_factorized_v9_predictor(
    path: Path,
    *,
    expected_sha256: str,
    repository_root: Path | None = None,
) -> FactorizedV9SemanticReceiver:
    """Read exact program bytes once, verify custody, then strictly receive."""

    program = _read_regular_file_once(path)
    expected = _require_sha256(expected_sha256, "expected_sha256")
    if _sha256(program) != expected:
        raise FactorizedV9PredictorError("predictor program SHA-256 custody mismatch")
    return receive_factorized_v9_predictor(
        program,
        repository_root=repository_root,
    )


__all__ = [
    "PREDICTOR_CONTRACT_ID",
    "RENDERER_SOURCE_PATHS",
    "SEMANTIC_BINDING_SCHEMA",
    "FactorizedV9PredictorError",
    "FactorizedV9SemanticReceiver",
    "compile_factorized_v9_predictor",
    "load_factorized_v9_predictor",
    "receive_factorized_v9_predictor",
    "renderer_source_manifest",
    "renderer_source_sha256",
]
