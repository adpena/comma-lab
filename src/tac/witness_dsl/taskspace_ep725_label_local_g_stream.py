# SPDX-License-Identifier: MIT
"""Path-backed ep725/V2 label-local G pairwise receiver.

The frozen ep725 receiver owns the base camera frames and its frame-1
``phi.argmax``.  This module observes that exact calculation, constructs a
one-pair :class:`TaskspacePredictorStateV2` with :class:`NoTransportV2`, applies
one counted label-local G page through the existing V2 seam, and delegates
camera realization to the existing signal-preserving overlay.

Only one pair is resident.  The chronological raw and immutable per-pair
checkpoints are path-backed, so the same code shape scales from one pair to the
full 600-pair population without an n600 Python allocation.  This module does
not score, package a candidate, or confer authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import types
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    PredictorPreservingOverlayResultV1,
    overlay_g_on_predictor_camera_y1,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    GTransportRequirementV2,
    NoTransportV2,
    TaskspacePredictorStateV2,
    require_g_transport_admission,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    apply_generative_taskspace_correction_v2,
    parse_generative_taskspace_correction_v2,
)

CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CHANNELS: Final = 3
MAX_PAIRS: Final = 600
FRAME_BYTES: Final = CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS
PAIR_BYTES: Final = 2 * FRAME_BYTES
DEFAULT_STORAGE_SAFETY_RESERVE_BYTES: Final = 8 * 1024**3
PAIR_CHECKPOINT_SCHEMA: Final = "tac.ep725_label_local_g_pair_checkpoint.v1"
EXECUTION_RECEIPT_SCHEMA: Final = "tac.ep725_label_local_g_stream_execution.v1"
_CHECKPOINT_ROOT_DOMAIN: Final = b"tac.ep725_label_local_g_checkpoint_root.v1\0"
_PAGE_ROOT_DOMAIN: Final = b"tac.ep725_label_local_g_page_root.v1\0"
_SHA256_HEX_LENGTH: Final = 64


class Ep725LabelLocalGStreamError(ValueError):
    """A source, receiver, G-page, chronology, or durable-state contract failed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Ep725LabelLocalGStreamError(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Ep725LabelLocalGStreamError("artifact is not canonical JSON") from exc


def _stable_read(path: Path, *, label: str) -> bytes:
    absolute = path.absolute()
    try:
        before = absolute.lstat()
    except FileNotFoundError as exc:
        raise Ep725LabelLocalGStreamError(f"{label} is absent: {absolute}") from exc
    if absolute.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise Ep725LabelLocalGStreamError(f"{label} must be a regular non-symlink file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise Ep725LabelLocalGStreamError(f"{label} path identity changed before open")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = absolute.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after_descriptor = (
        after_descriptor.st_dev,
        after_descriptor.st_ino,
        after_descriptor.st_size,
        after_descriptor.st_mtime_ns,
    )
    identity_after_path = (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    )
    if identity_before != identity_after_descriptor or identity_before != identity_after_path:
        raise Ep725LabelLocalGStreamError(f"{label} changed while being read")
    payload = b"".join(blocks)
    if len(payload) != before.st_size:
        raise Ep725LabelLocalGStreamError(f"{label} was read short")
    return payload


def _read_exact_range(path: Path, *, offset: int, length: int, label: str) -> bytes:
    if type(offset) is not int or type(length) is not int or offset < 0 or length < 0:
        raise Ep725LabelLocalGStreamError(f"{label} range must use nonnegative exact integers")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or offset + length > metadata.st_size:
            raise Ep725LabelLocalGStreamError(f"{label} range exceeds its regular file")
        payload = bytearray()
        cursor = offset
        while len(payload) < length:
            block = os.pread(descriptor, min(1 << 20, length - len(payload)), cursor)
            if not block:
                raise Ep725LabelLocalGStreamError(f"{label} range was read short")
            payload.extend(block)
            cursor += len(block)
    finally:
        os.close(descriptor)
    return bytes(payload)


def _file_size_sha256(path: Path, *, label: str) -> tuple[int, str]:
    """Hash a path-backed realization without retaining it in Python memory."""

    absolute = path.absolute()
    before = absolute.lstat()
    if absolute.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise Ep725LabelLocalGStreamError(f"{label} must be a regular non-symlink file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise Ep725LabelLocalGStreamError(f"{label} path identity changed before hash")
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = absolute.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after_descriptor = (
        after_descriptor.st_dev,
        after_descriptor.st_ino,
        after_descriptor.st_size,
        after_descriptor.st_mtime_ns,
    )
    identity_after_path = (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    )
    if identity_before != identity_after_descriptor or identity_before != identity_after_path:
        raise Ep725LabelLocalGStreamError(f"{label} changed while being hashed")
    return before.st_size, digest.hexdigest()


def _pwrite_exact(path: Path, *, offset: int, payload: bytes) -> None:
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        view = memoryview(payload)
        cursor = offset
        while view:
            written = os.pwrite(descriptor, view, cursor)
            if written <= 0:
                raise Ep725LabelLocalGStreamError("output raw pwrite was short")
            cursor += written
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _stable_read(path, label="immutable artifact") != payload:
            raise Ep725LabelLocalGStreamError(f"immutable artifact differs: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    descriptor = os.open(
        partial,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Ep725LabelLocalGStreamError(f"immutable artifact write was short: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.exists():
        partial.unlink()
        if _stable_read(path, label="immutable artifact") != payload:
            raise Ep725LabelLocalGStreamError(f"immutable write race differs: {path}")
        return
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _is_ssd_path(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return any(
        resolved == root or root in resolved.parents
        for root in (
            Path("/Volumes/VertigoDataTier/pact"),
            Path("/Volumes/APDataStore/pact"),
        )
    )


def _validate_run_root(path: Path, *, allow_non_ssd_for_tests: bool) -> Path:
    root = path.absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise Ep725LabelLocalGStreamError("run root must be one real directory")
    if not allow_non_ssd_for_tests and not _is_ssd_path(root):
        raise Ep725LabelLocalGStreamError("run root must live on the configured SSD tier")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise Ep725LabelLocalGStreamError("run root became a symlink")
    return root


def _strict_single_member(archive: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
            infos = handle.infolist()
            if len(infos) != 1 or infos[0].filename != "0.bin" or infos[0].is_dir():
                raise Ep725LabelLocalGStreamError("source archive must contain only regular 0.bin")
            if infos[0].flag_bits & 0x1:
                raise Ep725LabelLocalGStreamError("source archive member must not be encrypted")
            return handle.read(infos[0])
    except Ep725LabelLocalGStreamError:
        raise
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise Ep725LabelLocalGStreamError("source archive is not an exact readable ZIP") from exc


@dataclass(frozen=True, slots=True)
class Ep725StreamSourceContractV1:
    """Exact source/runtime/program custody for one stream execution."""

    source_archive_path: Path
    source_member_path: Path
    runtime_path: Path
    source_archive_sha256: str
    source_archive_bytes: int
    source_member_sha256: str
    source_member_bytes: int
    runtime_sha256: str
    runtime_bytes: int
    predictor_renderer_sha256: str
    n_pairs: int = MAX_PAIRS

    def __post_init__(self) -> None:
        for field_name in (
            "source_archive_sha256",
            "source_member_sha256",
            "runtime_sha256",
            "predictor_renderer_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        for field_name in (
            "source_archive_bytes",
            "source_member_bytes",
            "runtime_bytes",
            "n_pairs",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise Ep725LabelLocalGStreamError(f"{field_name} must be a positive exact integer")
        if self.n_pairs > MAX_PAIRS:
            raise Ep725LabelLocalGStreamError("source population exceeds the contest 600-pair universe")


@dataclass(frozen=True, slots=True)
class GPageRefV1:
    """Path-backed exact counted G page for one source pair."""

    pair_index: int
    path: Path
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        if type(self.pair_index) is not int or not 0 <= self.pair_index < MAX_PAIRS:
            raise Ep725LabelLocalGStreamError("G page pair index escaped [0,600)")
        _require_sha256(self.sha256, "G page sha256")
        if type(self.byte_length) is not int or self.byte_length < 1:
            raise Ep725LabelLocalGStreamError("G page byte length must be positive")


@dataclass(frozen=True, slots=True)
class Ep725PairCheckpointV1:
    schema: Literal["tac.ep725_label_local_g_pair_checkpoint.v1"]
    pair_index: int
    chronological_output_pair_index: int
    output_raw_path: str
    output_pair_offset: int
    output_pair_bytes: int
    source_archive_sha256: str
    source_member_sha256: str
    runtime_sha256: str
    predictor_renderer_sha256: str
    g_page_sha256: str
    g_page_bytes: int
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_labels_sha256: str
    decoded_g_labels_sha256: str
    base_y0_sha256: str
    base_y1_sha256: str
    base_pair_sha256: str
    output_y0_sha256: str
    output_y1_sha256: str
    output_pair_sha256: str
    overlay_receipt_path: str
    overlay_receipt_sha256: str
    changed_scorer_cells: int
    actually_changed_camera_values: int
    y0_preserved: bool
    unowned_y1_preserved: bool
    transport_kind: Literal["NONE"] = "NONE"
    transport_requirement: Literal["LABEL_LOCAL"] = "LABEL_LOCAL"
    scorer_invoked: bool = False
    score_claim: bool = False
    candidate_claim: bool = False
    promotion_eligible: bool = False
    research_only: bool = True

    def __post_init__(self) -> None:
        if self.schema != PAIR_CHECKPOINT_SCHEMA:
            raise Ep725LabelLocalGStreamError("pair checkpoint schema changed")
        exact_integers = (
            "pair_index",
            "chronological_output_pair_index",
            "output_pair_offset",
            "output_pair_bytes",
            "g_page_bytes",
            "changed_scorer_cells",
            "actually_changed_camera_values",
        )
        if any(type(getattr(self, name)) is not int for name in exact_integers):
            raise Ep725LabelLocalGStreamError("pair checkpoint counters must be exact integers")
        if self.chronological_output_pair_index != self.pair_index:
            raise Ep725LabelLocalGStreamError("pair checkpoint chronology differs from source pair")
        if self.output_pair_offset != self.pair_index * PAIR_BYTES or self.output_pair_bytes != PAIR_BYTES:
            raise Ep725LabelLocalGStreamError("pair checkpoint output range changed")
        if self.g_page_bytes < 1 or self.changed_scorer_cells < 1 or self.actually_changed_camera_values < 1:
            raise Ep725LabelLocalGStreamError("pair checkpoint does not describe a realized G change")
        for field_name in (
            "source_archive_sha256",
            "source_member_sha256",
            "runtime_sha256",
            "predictor_renderer_sha256",
            "g_page_sha256",
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_labels_sha256",
            "decoded_g_labels_sha256",
            "base_y0_sha256",
            "base_y1_sha256",
            "base_pair_sha256",
            "output_y0_sha256",
            "output_y1_sha256",
            "output_pair_sha256",
            "overlay_receipt_sha256",
        ):
            _require_sha256(getattr(self, field_name), f"pair checkpoint {field_name}")
        truth = {
            "y0_preserved": True,
            "unowned_y1_preserved": True,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if any(getattr(self, name) is not expected for name, expected in truth.items()):
            raise Ep725LabelLocalGStreamError("pair checkpoint truth labels became permissive")

    def to_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_bytes())


@dataclass(frozen=True, slots=True)
class Ep725StreamExecutionReceiptV1:
    schema: Literal["tac.ep725_label_local_g_stream_execution.v1"]
    pair_count: int
    chronological_pair_order: tuple[int, ...]
    source_archive_sha256: str
    source_archive_bytes: int
    source_member_sha256: str
    source_member_bytes: int
    runtime_sha256: str
    runtime_bytes: int
    predictor_renderer_sha256: str
    g_page_root_sha256: str
    g_page_total_bytes: int
    output_raw_path: str
    output_raw_size_bytes: int
    output_raw_sha256: str
    pair_checkpoint_root_sha256: str
    pair_checkpoint_sha256s: tuple[str, ...]
    storage_tier: str
    path_backed_output: bool = True
    full_population_resident_in_python: bool = False
    transport_kind: Literal["NONE"] = "NONE"
    scorer_invoked: bool = False
    score_claim: bool = False
    candidate_claim: bool = False
    promotion_eligible: bool = False
    research_only: bool = True

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_RECEIPT_SCHEMA:
            raise Ep725LabelLocalGStreamError("execution receipt schema changed")
        if type(self.pair_count) is not int or not 1 <= self.pair_count <= MAX_PAIRS:
            raise Ep725LabelLocalGStreamError("execution pair count escaped [1,600]")
        if self.chronological_pair_order != tuple(range(self.pair_count)):
            raise Ep725LabelLocalGStreamError("execution chronology must be the exact contest prefix")
        if self.output_raw_size_bytes != self.pair_count * PAIR_BYTES:
            raise Ep725LabelLocalGStreamError("execution output byte cardinality changed")
        if len(self.pair_checkpoint_sha256s) != self.pair_count:
            raise Ep725LabelLocalGStreamError("execution checkpoint population is incomplete")
        for field_name in (
            "source_archive_sha256",
            "source_member_sha256",
            "runtime_sha256",
            "predictor_renderer_sha256",
            "g_page_root_sha256",
            "output_raw_sha256",
            "pair_checkpoint_root_sha256",
        ):
            _require_sha256(getattr(self, field_name), f"execution {field_name}")
        for digest in self.pair_checkpoint_sha256s:
            _require_sha256(digest, "execution checkpoint sha256")
        truth = {
            "path_backed_output": True,
            "full_population_resident_in_python": False,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if any(getattr(self, name) is not expected for name, expected in truth.items()):
            raise Ep725LabelLocalGStreamError("execution receipt truth labels became permissive")

    def to_bytes(self) -> bytes:
        body = asdict(self)
        body["chronological_pair_order"] = list(self.chronological_pair_order)
        body["pair_checkpoint_sha256s"] = list(self.pair_checkpoint_sha256s)
        return _canonical_json(body)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_bytes())


@dataclass(frozen=True, slots=True)
class _VerifiedSourceV1:
    contract: Ep725StreamSourceContractV1
    archive: bytes
    member: bytes
    runtime: bytes


@dataclass(frozen=True, slots=True)
class _VerifiedPageV1:
    reference: GPageRefV1
    packet: bytes


def _verify_source(contract: Ep725StreamSourceContractV1) -> _VerifiedSourceV1:
    if type(contract) is not Ep725StreamSourceContractV1:
        raise Ep725LabelLocalGStreamError("source contract must use the exact V1 type")
    archive = _stable_read(contract.source_archive_path, label="source archive")
    member = _stable_read(contract.source_member_path, label="source member")
    runtime = _stable_read(contract.runtime_path, label="source runtime")
    observed = {
        "source_archive_sha256": _sha256(archive),
        "source_archive_bytes": len(archive),
        "source_member_sha256": _sha256(member),
        "source_member_bytes": len(member),
        "runtime_sha256": _sha256(runtime),
        "runtime_bytes": len(runtime),
    }
    mismatches = [name for name, value in observed.items() if getattr(contract, name) != value]
    if mismatches:
        raise Ep725LabelLocalGStreamError(f"source contract drifted: {sorted(mismatches)}")
    if _strict_single_member(archive) != member:
        raise Ep725LabelLocalGStreamError("source archive 0.bin differs from the counted member path")
    return _VerifiedSourceV1(contract=contract, archive=archive, member=member, runtime=runtime)


def _validate_page_refs(page_refs: Sequence[GPageRefV1], *, pair_count: int) -> tuple[GPageRefV1, ...]:
    if type(page_refs) not in {tuple, list} or len(page_refs) != pair_count:
        raise Ep725LabelLocalGStreamError("G pages must exactly cover the requested pair population")
    for reference in page_refs:
        if type(reference) is not GPageRefV1:
            raise Ep725LabelLocalGStreamError("G page references must use the exact V1 type")
    if tuple(row.pair_index for row in page_refs) != tuple(range(pair_count)):
        raise Ep725LabelLocalGStreamError("G page order must be exactly 0..pair_count-1")
    return tuple(page_refs)


def _verify_page(reference: GPageRefV1) -> _VerifiedPageV1:
    """Reopen one page so page payload memory remains O(one pair)."""

    packet = _stable_read(reference.path, label=f"G page {reference.pair_index}")
    if len(packet) != reference.byte_length or _sha256(packet) != reference.sha256:
        raise Ep725LabelLocalGStreamError(f"G page {reference.pair_index} bytes/hash drifted")
    return _VerifiedPageV1(reference=reference, packet=packet)


def _load_runtime(source: _VerifiedSourceV1) -> types.ModuleType:
    module = types.ModuleType(f"_tac_g45_ep725_runtime_{os.getpid()}_{id(source)}")
    module.__file__ = str(source.contract.runtime_path.absolute())
    module.__package__ = ""
    try:
        exec(compile(source.runtime, module.__file__, "exec", dont_inherit=True), module.__dict__)
    except Exception as exc:
        raise Ep725LabelLocalGStreamError(f"exact runtime source could not be loaded: {exc}") from exc
    required = ("_setup", "_render_pair", "_outputs_from_h0", "_G")
    if any(not hasattr(module, name) for name in required):
        raise Ep725LabelLocalGStreamError(f"runtime lacks the required private state API: {required}")
    if not all(callable(getattr(module, name)) for name in required[:3]):
        raise Ep725LabelLocalGStreamError("runtime state API contains a non-callable")
    if getattr(module, "_FP32", True) is not False:
        raise Ep725LabelLocalGStreamError("runtime must execute the portable fp64 path")
    module._setup(str(source.contract.source_member_path.absolute()))
    if _stable_read(source.contract.source_member_path, label="source member after setup") != source.member:
        raise Ep725LabelLocalGStreamError("source member changed during runtime setup")
    state = module._G
    observed = {
        "n_pairs": int(state["m"]["n_pairs"]),
        "render_h": int(state["rh"]),
        "render_w": int(state["rw"]),
        "camera_h": int(state["ch"]),
        "camera_w": int(state["cw"]),
        "frame_bytes": int(state["framebytes"]),
    }
    expected = {
        "n_pairs": source.contract.n_pairs,
        "render_h": SCORER_HEIGHT,
        "render_w": SCORER_WIDTH,
        "camera_h": CAMERA_HEIGHT,
        "camera_w": CAMERA_WIDTH,
        "frame_bytes": FRAME_BYTES,
    }
    if observed != expected:
        raise Ep725LabelLocalGStreamError(f"runtime geometry/population drifted: {observed} != {expected}")
    return module


def _capture_base_pair(
    runtime: types.ModuleType,
    *,
    pair_index: int,
    output_partial_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    code = np.asarray(runtime._G.get("code"))
    if code.ndim < 2 or code.shape[0] != 2 * int(runtime._G["m"]["n_pairs"]):
        raise Ep725LabelLocalGStreamError("runtime code table does not carry two rows per pair")
    target_code_row = code[2 * pair_index + 1]
    original = runtime._outputs_from_h0
    captures: list[np.ndarray] = []

    def capture(
        p: Any,
        h0: Any,
        code_row: Any,
        manifest: Any,
        want_rgb: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original(p, h0, code_row, manifest, want_rgb, *args, **kwargs)
        row = np.asarray(code_row)
        if want_rgb is True and np.shares_memory(row, target_code_row):
            if not isinstance(result, tuple) or not result:
                raise Ep725LabelLocalGStreamError("runtime frame-1 renderer did not return phi")
            phi = np.asarray(result[0])
            if phi.ndim != 2 or phi.shape[0] != SCORER_HEIGHT * SCORER_WIDTH or phi.shape[1] != 5:
                raise Ep725LabelLocalGStreamError("runtime frame-1 phi ABI changed")
            if not np.all(np.isfinite(phi)):
                raise Ep725LabelLocalGStreamError("runtime frame-1 phi contains non-finite values")
            captures.append(
                np.ascontiguousarray(phi.argmax(axis=-1).reshape(SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint8)
            )
        return result

    runtime._outputs_from_h0 = capture
    runtime._G["dst"] = str(output_partial_path)
    try:
        returned = runtime._render_pair(pair_index)
    finally:
        runtime._outputs_from_h0 = original
    if returned != pair_index:
        raise Ep725LabelLocalGStreamError("runtime _render_pair return identity changed")
    if len(captures) != 1:
        raise Ep725LabelLocalGStreamError(
            f"runtime must expose exactly one actual frame-1 phi, observed {len(captures)}"
        )
    pair = _read_exact_range(
        output_partial_path,
        offset=pair_index * PAIR_BYTES,
        length=PAIR_BYTES,
        label=f"base pair {pair_index}",
    )
    y0 = np.frombuffer(pair[:FRAME_BYTES], dtype=np.uint8).reshape(CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS).copy()
    y1 = np.frombuffer(pair[FRAME_BYTES:], dtype=np.uint8).reshape(CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS).copy()
    return y0, y1, captures[0]


def _state_for_pair(
    source: _VerifiedSourceV1,
    *,
    pair_index: int,
    labels: np.ndarray,
) -> TaskspacePredictorStateV2:
    return TaskspacePredictorStateV2(
        predictor_program=source.member,
        predictor_renderer_sha256=source.contract.predictor_renderer_sha256,
        source_archive_sha256=source.contract.source_archive_sha256,
        source_runtime_sha256=source.contract.runtime_sha256,
        source_member_name="0.bin",
        source_pair_ids=(pair_index,),
        labels=np.ascontiguousarray(labels[None], dtype=np.uint8),
        transport=NoTransportV2(),
    )


def _execute_pair(
    source: _VerifiedSourceV1,
    runtime: types.ModuleType,
    page: _VerifiedPageV1,
    *,
    output_partial_path: Path,
    output_final_path: Path,
    checkpoint_directory: Path,
) -> Ep725PairCheckpointV1:
    pair_index = page.reference.pair_index
    y0, y1, labels = _capture_base_pair(
        runtime,
        pair_index=pair_index,
        output_partial_path=output_partial_path,
    )
    state = _state_for_pair(source, pair_index=pair_index, labels=labels)
    parsed = parse_generative_taskspace_correction_v2(page.packet, predictor_state=state)
    requirement = require_g_transport_admission(state, parsed.program)
    if requirement is not GTransportRequirementV2.LABEL_LOCAL:
        raise Ep725LabelLocalGStreamError("G45 admits only label-local G under NoTransportV2")
    decoded = apply_generative_taskspace_correction_v2(page.packet, predictor_state=state)
    overlay: PredictorPreservingOverlayResultV1 = overlay_g_on_predictor_camera_y1(
        y1[None],
        labels[None],
        decoded,
    )
    corrected_y1 = np.ascontiguousarray(overlay.camera_y1[0], dtype=np.uint8)
    _pwrite_exact(
        output_partial_path,
        offset=pair_index * PAIR_BYTES + FRAME_BYTES,
        payload=memoryview(corrected_y1).cast("B").tobytes(),
    )
    output_pair = _read_exact_range(
        output_partial_path,
        offset=pair_index * PAIR_BYTES,
        length=PAIR_BYTES,
        label=f"realized pair {pair_index}",
    )
    output_y0 = output_pair[:FRAME_BYTES]
    output_y1 = output_pair[FRAME_BYTES:]
    base_y0 = memoryview(np.ascontiguousarray(y0)).cast("B").tobytes()
    base_y1 = memoryview(np.ascontiguousarray(y1)).cast("B").tobytes()
    expected_y1 = memoryview(corrected_y1).cast("B").tobytes()
    if output_y0 != base_y0 or output_y1 != expected_y1:
        raise Ep725LabelLocalGStreamError("path-backed pair differs from the exact delegated state transition")
    if not np.array_equal(corrected_y1[~overlay.owned_camera_mask[0]], y1[~overlay.owned_camera_mask[0]]):
        raise Ep725LabelLocalGStreamError("G45 changed an unowned Y1 camera value")

    overlay_payload = overlay.receipt.to_receipt_bytes()
    overlay_digest = _sha256(overlay_payload)
    overlay_path = checkpoint_directory / "overlays" / f"pair_{pair_index:04d}.{overlay_digest}.json"
    _write_once(overlay_path, overlay_payload)
    checkpoint = Ep725PairCheckpointV1(
        schema=PAIR_CHECKPOINT_SCHEMA,
        pair_index=pair_index,
        chronological_output_pair_index=pair_index,
        output_raw_path=str(output_final_path.absolute()),
        output_pair_offset=pair_index * PAIR_BYTES,
        output_pair_bytes=PAIR_BYTES,
        source_archive_sha256=source.contract.source_archive_sha256,
        source_member_sha256=source.contract.source_member_sha256,
        runtime_sha256=source.contract.runtime_sha256,
        predictor_renderer_sha256=source.contract.predictor_renderer_sha256,
        g_page_sha256=page.reference.sha256,
        g_page_bytes=page.reference.byte_length,
        predictor_state_binding_sha256=state.binding_sha256,
        predictor_semantic_binding_sha256=state.semantic_binding_sha256,
        predictor_labels_sha256=state.labels_sha256,
        decoded_g_labels_sha256=_sha256(memoryview(decoded.labels).cast("B")),
        base_y0_sha256=_sha256(base_y0),
        base_y1_sha256=_sha256(base_y1),
        base_pair_sha256=_sha256(base_y0 + base_y1),
        output_y0_sha256=_sha256(output_y0),
        output_y1_sha256=_sha256(output_y1),
        output_pair_sha256=_sha256(output_pair),
        overlay_receipt_path=str(overlay_path.absolute()),
        overlay_receipt_sha256=overlay_digest,
        changed_scorer_cells=overlay.receipt.changed_scorer_cells,
        actually_changed_camera_values=overlay.receipt.actually_changed_camera_values,
        y0_preserved=True,
        unowned_y1_preserved=True,
    )
    checkpoint_path = checkpoint_directory / f"pair_{pair_index:04d}.json"
    _write_once(checkpoint_path, checkpoint.to_bytes())
    return checkpoint


def _parse_pair_checkpoint(payload: bytes) -> Ep725PairCheckpointV1:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Ep725LabelLocalGStreamError("pair checkpoint is not JSON") from exc
    if type(value) is not dict:
        raise Ep725LabelLocalGStreamError("pair checkpoint root must be an object")
    try:
        parsed = Ep725PairCheckpointV1(**value)
    except TypeError as exc:
        raise Ep725LabelLocalGStreamError("pair checkpoint fields changed") from exc
    if parsed.to_bytes() != payload:
        raise Ep725LabelLocalGStreamError("pair checkpoint is not byte-canonical")
    return parsed


def _validate_checkpoint(
    checkpoint: Ep725PairCheckpointV1,
    *,
    source: _VerifiedSourceV1,
    page: _VerifiedPageV1,
    output_partial_path: Path,
    output_final_path: Path,
) -> None:
    expected = {
        "pair_index": page.reference.pair_index,
        "source_archive_sha256": source.contract.source_archive_sha256,
        "source_member_sha256": source.contract.source_member_sha256,
        "runtime_sha256": source.contract.runtime_sha256,
        "predictor_renderer_sha256": source.contract.predictor_renderer_sha256,
        "g_page_sha256": page.reference.sha256,
        "g_page_bytes": page.reference.byte_length,
        "output_raw_path": str(output_final_path.absolute()),
    }
    mismatches = [name for name, value in expected.items() if getattr(checkpoint, name) != value]
    if mismatches:
        raise Ep725LabelLocalGStreamError(f"committed pair contract drifted: {sorted(mismatches)}")
    output_pair = _read_exact_range(
        output_partial_path,
        offset=checkpoint.output_pair_offset,
        length=checkpoint.output_pair_bytes,
        label=f"committed pair {checkpoint.pair_index}",
    )
    if _sha256(output_pair) != checkpoint.output_pair_sha256:
        raise Ep725LabelLocalGStreamError(f"committed pair {checkpoint.pair_index} output range drifted")
    if _sha256(output_pair[:FRAME_BYTES]) != checkpoint.output_y0_sha256:
        raise Ep725LabelLocalGStreamError(f"committed pair {checkpoint.pair_index} Y0 drifted")
    if _sha256(output_pair[FRAME_BYTES:]) != checkpoint.output_y1_sha256:
        raise Ep725LabelLocalGStreamError(f"committed pair {checkpoint.pair_index} Y1 drifted")
    overlay_path = Path(checkpoint.overlay_receipt_path)
    if _sha256(_stable_read(overlay_path, label="overlay receipt")) != checkpoint.overlay_receipt_sha256:
        raise Ep725LabelLocalGStreamError(f"committed pair {checkpoint.pair_index} overlay receipt drifted")


def _root_hash(domain: bytes, digests: Sequence[str]) -> str:
    digest = hashlib.sha256(domain)
    for value in digests:
        digest.update(bytes.fromhex(_require_sha256(value, "root child digest")))
    return digest.hexdigest()


def _parse_execution_receipt(payload: bytes) -> Ep725StreamExecutionReceiptV1:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Ep725LabelLocalGStreamError("execution receipt is not JSON") from exc
    if type(value) is not dict:
        raise Ep725LabelLocalGStreamError("execution receipt root must be an object")
    value["chronological_pair_order"] = tuple(value.get("chronological_pair_order", ()))
    value["pair_checkpoint_sha256s"] = tuple(value.get("pair_checkpoint_sha256s", ()))
    try:
        parsed = Ep725StreamExecutionReceiptV1(**value)
    except TypeError as exc:
        raise Ep725LabelLocalGStreamError("execution receipt fields changed") from exc
    if parsed.to_bytes() != payload:
        raise Ep725LabelLocalGStreamError("execution receipt is not byte-canonical")
    return parsed


def _preallocate(path: Path, *, expected_bytes: int) -> None:
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.ftruncate(descriptor, expected_bytes)
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)


def execute_ep725_label_local_g_stream(
    source_contract: Ep725StreamSourceContractV1,
    g_pages: Sequence[GPageRefV1],
    *,
    pair_count: int,
    run_root: str | Path,
    output_name: str = "ep725_label_local_g.raw",
    storage_safety_reserve_bytes: int = DEFAULT_STORAGE_SAFETY_RESERVE_BYTES,
    allow_non_ssd_for_tests: bool = False,
) -> Ep725StreamExecutionReceiptV1:
    """Execute a chronological prefix of exact ep725/V2 label-local G pages.

    ``allow_non_ssd_for_tests`` is an explicit mechanics-fixture escape hatch;
    receipts remain research-only and can never acquire score authority.
    """

    if type(pair_count) is not int or not 1 <= pair_count <= MAX_PAIRS:
        raise Ep725LabelLocalGStreamError("pair_count must be an exact integer in [1,600]")
    if pair_count > source_contract.n_pairs:
        raise Ep725LabelLocalGStreamError("requested prefix exceeds source population")
    if type(storage_safety_reserve_bytes) is not int or storage_safety_reserve_bytes < 0:
        raise Ep725LabelLocalGStreamError("storage reserve must be a nonnegative exact integer")
    if (
        type(output_name) is not str
        or not output_name
        or Path(output_name).name != output_name
        or output_name.endswith(".partial")
    ):
        raise Ep725LabelLocalGStreamError("output_name must be one safe leaf name")

    root = _validate_run_root(Path(run_root), allow_non_ssd_for_tests=allow_non_ssd_for_tests)
    source = _verify_source(source_contract)
    page_refs = _validate_page_refs(g_pages, pair_count=pair_count)
    final_path = root / output_name
    partial_path = root / f"{output_name}.partial"
    checkpoint_directory = root / "pair_checkpoints"
    receipt_path = root / "execution_receipt.json"
    promotion_receipt_directory = root / "promotion_receipts"
    expected_bytes = pair_count * PAIR_BYTES

    if final_path.exists():
        if partial_path.exists():
            raise Ep725LabelLocalGStreamError("final output lacks one unambiguous completion state")
        if receipt_path.exists():
            receipt_payload = _stable_read(receipt_path, label="execution receipt")
            recovered_receipt = False
        else:
            promotion_receipts = tuple(sorted(promotion_receipt_directory.glob("*.json")))
            if len(promotion_receipts) != 1:
                raise Ep725LabelLocalGStreamError("final output lacks one unambiguous pre-promotion receipt")
            receipt_payload = _stable_read(promotion_receipts[0], label="pre-promotion execution receipt")
            recovered_receipt = True
        receipt = _parse_execution_receipt(receipt_payload)
        final_size, final_sha256 = _file_size_sha256(final_path, label="final output raw")
        if final_size != receipt.output_raw_size_bytes or final_sha256 != receipt.output_raw_sha256:
            raise Ep725LabelLocalGStreamError("final output raw differs from execution receipt")
        if receipt.pair_count != pair_count:
            raise Ep725LabelLocalGStreamError("completed output population differs from request")
        # A completed raw remains valid only while every exact counted page is
        # still present. Reopen serially to retain the O(one page) shape.
        for reference in page_refs:
            _verify_page(reference)
        page_digests = tuple(reference.sha256 for reference in page_refs)
        expected_receipt = {
            "source_archive_sha256": source.contract.source_archive_sha256,
            "source_member_sha256": source.contract.source_member_sha256,
            "runtime_sha256": source.contract.runtime_sha256,
            "predictor_renderer_sha256": source.contract.predictor_renderer_sha256,
            "g_page_root_sha256": _root_hash(_PAGE_ROOT_DOMAIN, page_digests),
            "g_page_total_bytes": sum(reference.byte_length for reference in page_refs),
            "output_raw_path": str(final_path.absolute()),
        }
        mismatches = [name for name, value in expected_receipt.items() if getattr(receipt, name) != value]
        if mismatches:
            raise Ep725LabelLocalGStreamError(f"completed execution input contract drifted: {sorted(mismatches)}")
        if recovered_receipt:
            _write_once(receipt_path, receipt_payload)
        return receipt

    if not partial_path.exists():
        free_bytes = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
        if free_bytes < expected_bytes + storage_safety_reserve_bytes:
            raise Ep725LabelLocalGStreamError(
                "storage preflight refused output: free bytes below output plus safety reserve"
            )
        _preallocate(partial_path, expected_bytes=expected_bytes)
    else:
        metadata = partial_path.lstat()
        if partial_path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
            raise Ep725LabelLocalGStreamError("resumed partial output path/size drifted")

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = tuple(checkpoint_directory.glob("pair_*.json"))
    expected_names = {f"pair_{index:04d}.json" for index in range(pair_count)}
    if any(path.name not in expected_names for path in checkpoint_paths):
        raise Ep725LabelLocalGStreamError("checkpoint directory contains an out-of-population pair")
    existing_ids = sorted(int(path.stem.split("_")[1]) for path in checkpoint_paths)
    if existing_ids != list(range(len(existing_ids))):
        raise Ep725LabelLocalGStreamError("pair checkpoints must form one gap-free chronological prefix")

    checkpoints: list[Ep725PairCheckpointV1] = []
    for index in existing_ids:
        page = _verify_page(page_refs[index])
        checkpoint = _parse_pair_checkpoint(
            _stable_read(checkpoint_directory / f"pair_{index:04d}.json", label="pair checkpoint")
        )
        _validate_checkpoint(
            checkpoint,
            source=source,
            page=page,
            output_partial_path=partial_path,
            output_final_path=final_path,
        )
        checkpoints.append(checkpoint)

    runtime = _load_runtime(source)
    for index in range(len(checkpoints), pair_count):
        page = _verify_page(page_refs[index])
        checkpoint = _execute_pair(
            source,
            runtime,
            page,
            output_partial_path=partial_path,
            output_final_path=final_path,
            checkpoint_directory=checkpoint_directory,
        )
        checkpoints.append(checkpoint)

    # Reopen every committed range after the last mutation before promotion.
    for index, checkpoint in enumerate(checkpoints):
        page = _verify_page(page_refs[index])
        _validate_checkpoint(
            checkpoint,
            source=source,
            page=page,
            output_partial_path=partial_path,
            output_final_path=final_path,
        )
    metadata = partial_path.lstat()
    if metadata.st_size != expected_bytes:
        raise Ep725LabelLocalGStreamError("completed partial output byte cardinality changed")
    output_size, output_sha256 = _file_size_sha256(partial_path, label="completed partial output")
    if output_size != expected_bytes:
        raise Ep725LabelLocalGStreamError("completed output size changed during final hash")
    checkpoint_digests = tuple(checkpoint.receipt_sha256 for checkpoint in checkpoints)
    page_digests = tuple(reference.sha256 for reference in page_refs)
    receipt = Ep725StreamExecutionReceiptV1(
        schema=EXECUTION_RECEIPT_SCHEMA,
        pair_count=pair_count,
        chronological_pair_order=tuple(range(pair_count)),
        source_archive_sha256=source.contract.source_archive_sha256,
        source_archive_bytes=source.contract.source_archive_bytes,
        source_member_sha256=source.contract.source_member_sha256,
        source_member_bytes=source.contract.source_member_bytes,
        runtime_sha256=source.contract.runtime_sha256,
        runtime_bytes=source.contract.runtime_bytes,
        predictor_renderer_sha256=source.contract.predictor_renderer_sha256,
        g_page_root_sha256=_root_hash(_PAGE_ROOT_DOMAIN, page_digests),
        g_page_total_bytes=sum(reference.byte_length for reference in page_refs),
        output_raw_path=str(final_path.absolute()),
        output_raw_size_bytes=expected_bytes,
        output_raw_sha256=output_sha256,
        pair_checkpoint_root_sha256=_root_hash(_CHECKPOINT_ROOT_DOMAIN, checkpoint_digests),
        pair_checkpoint_sha256s=checkpoint_digests,
        storage_tier=("configured_ssd" if _is_ssd_path(root) else "explicit_test_local"),
    )
    receipt_payload = receipt.to_bytes()
    promotion_receipt_path = promotion_receipt_directory / f"execution_receipt.{receipt.receipt_sha256}.json"
    # Durable intent precedes promotion.  If the process dies after rename but
    # before the canonical receipt lands, the next invocation verifies the
    # final bytes against this content-addressed receipt and finishes the
    # promotion without rerendering or guessing.
    _write_once(promotion_receipt_path, receipt_payload)
    os.replace(partial_path, final_path)
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    _write_once(receipt_path, receipt_payload)
    return receipt


__all__ = [
    "CAMERA_HEIGHT",
    "CAMERA_WIDTH",
    "EXECUTION_RECEIPT_SCHEMA",
    "FRAME_BYTES",
    "MAX_PAIRS",
    "PAIR_BYTES",
    "PAIR_CHECKPOINT_SCHEMA",
    "SCORER_HEIGHT",
    "SCORER_WIDTH",
    "Ep725LabelLocalGStreamError",
    "Ep725PairCheckpointV1",
    "Ep725StreamExecutionReceiptV1",
    "Ep725StreamSourceContractV1",
    "GPageRefV1",
    "execute_ep725_label_local_g_stream",
]
