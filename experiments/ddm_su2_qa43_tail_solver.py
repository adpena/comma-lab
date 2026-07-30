#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""QA43 top-k tail solver and whole-archive pricing harness.

The generic harness deliberately does not assume a PoseNet, receiver, or
correction basis.  A production adapter must build exact outer-archive bytes,
decode those bytes through the shipped public receiver, and expose the frozen
scorer only as an injected callback.  The built-in v4d adapter supplies one
concrete same-pool warp-tail receiver; it explicitly cannot instantiate
QA43's distinct free-frame0 counterfactual.

For every proposed integer coefficient vector the harness performs:

    adapter.build_archive(population coefficients, codec, 600-bit pair map)
      -> adapter.decode_pair(exact archive, pair)
      -> assert decoded pair == adapter.realize_pair(pair, coefficients)
      -> assert frame1 is byte-identical to the exact parent
      -> adapter.score_pose6(decoded uint8 pair)

The hardest 200 pairs are ranked from a fresh n600 parent replay.  Solves are
resumable and nested stage checkpoints are preserved at k={56,112,200}.  Each
stage races complete SMEVR and Brotli outer archives, re-decodes every selected
pair from both candidates, and prices the winner by exact archive bytes.  The
adapter, not this harness, owns the stream grammar; therefore an unsupported
coder or missing receiver consumption fails closed.

No result from this tool is a contest score.  Axis:
[macOS-CPU advisory], score_claim=false, local custody pointer
0.1910828242 [contest-CPU] UNMOVED; official competitive bar ~0.172141.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np

PAIR_COUNT = 600
STAGE_KS = (56, 112, 200)
CODECS = ("smevr", "brotli11")
PROGRAM_KINDS = ("warp-tail", "terminal-frame0")
RATE_DENOMINATOR = 37_545_489.0
PAIR_MAP_BYTES = PAIR_COUNT // 8
POINTER = "0.1910828242 [contest-CPU] UNMOVED"
AXIS = "[macOS-CPU advisory]"
_SHA256_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_STATE_PREFIXES = (
    "/tmp/",
    "/var/tmp/",
    "/private/tmp/",
    "/private/var/tmp/",
)


class QA43Error(RuntimeError):
    """Fail-closed adapter, custody, receiver, scorer, or resume error."""


@runtime_checkable
class QA43ArchiveAdapter(Protocol):
    """Exact archive/receiver/scorer boundary required by the harness."""

    def custody(self) -> Mapping[str, Any]:
        """Return immutable hashes and receiver/scorer closure flags."""

    def parent_archive(self) -> bytes:
        """Return exact parent outer-archive bytes."""

    def coefficient_rank(self) -> int:
        """Return the count of integer terminal coordinates per active pair."""

    def parent_pair(self, pair_index: int) -> np.ndarray:
        """Decode one exact parent pair as uint8 [2,H,W,3]."""

    def initial_coefficients(self, pair_index: int) -> np.ndarray:
        """Return the integer correction home for one pair."""

    def target_pose6(self, pair_index: int) -> np.ndarray:
        """Return the six-value frozen target used only by the encoder."""

    def realize_pair(
        self,
        pair_index: int,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Apply the candidate receiver correction and return uint8 pair."""

    def build_archive(
        self,
        updates: Mapping[int, np.ndarray],
        *,
        codec: str,
        pair_map: bytes,
    ) -> bytes:
        """Build complete deterministic candidate archive bytes."""

    def decode_pair(self, archive: bytes, pair_index: int) -> np.ndarray:
        """Parse the candidate archive and decode one pair through its receiver."""

    def public_decode_pair(self, archive: bytes, pair_index: int) -> np.ndarray:
        """Decode via the shipped receiver's exact filesystem constructor."""

    def score_pose6(self, pair_index: int, pair_u8: np.ndarray) -> np.ndarray:
        """Run the injected frozen scorer on only the decoded uint8 pair."""

    def accounting(self, archive: bytes) -> Mapping[str, Any]:
        """Return exact member accounting and receiver-consumption status."""

    def extract_pair_map(self, archive: bytes) -> bytes:
        """Parse and return the exact 75-byte tail map consumed by the receiver."""


@dataclass(frozen=True)
class PairEvaluation:
    pair: int
    coefficients: tuple[int, ...]
    pose6: tuple[float, ...]
    d_pose: float
    archive_bytes: int
    archive_sha256: str


@dataclass(frozen=True)
class SolveConfig:
    relinearizations: int = 3
    damping: float = 1.0e-3
    line_search: tuple[float, ...] = (1.0, 0.5, 0.25)
    coefficient_limit: int = 7

    def __post_init__(self) -> None:
        if self.relinearizations not in (2, 3):
            raise QA43Error("relinearizations must be 2 or 3")
        if not math.isfinite(self.damping) or self.damping <= 0:
            raise QA43Error("damping must be finite and positive")
        if (
            not self.line_search
            or tuple(sorted(set(self.line_search), reverse=True))
            != self.line_search
            or any(not math.isfinite(value) or not 0 < value <= 1 for value in self.line_search)
        ):
            raise QA43Error("line_search must be unique, descending, and inside (0,1]")
        if not 1 <= self.coefficient_limit <= 7:
            raise QA43Error("coefficient_limit must be in [1,7] for the 4-bit coder race")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_HEX)
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_json(dict(value)) + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise QA43Error(f"unreadable JSON checkpoint: {path}") from exc
    if not isinstance(value, dict):
        raise QA43Error(f"checkpoint must contain an object: {path}")
    return value


def _uint8_pair(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.uint8
        or array.ndim != 4
        or array.shape[0] != 2
        or array.shape[-1] != 3
        or array.shape[1] < 1
        or array.shape[2] < 1
    ):
        raise QA43Error(f"{name} must be uint8 [2,H,W,3]")
    return np.ascontiguousarray(array)


def _pose6(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise QA43Error(f"{name} must be finite shape (6,)")
    return np.ascontiguousarray(array)


def _coefficients(
    value: object,
    *,
    rank: int,
    limit: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (rank,) or raw.dtype.kind not in ("i", "u"):
        raise QA43Error(f"{name} must be an integer vector of rank {rank}")
    if np.any(raw < -limit) or np.any(raw > limit):
        raise QA43Error(f"{name} exceeds signed 4-bit coefficient limit {limit}")
    result = np.ascontiguousarray(raw, dtype=np.int16)
    return result


def _d_pose(pose: np.ndarray, target: np.ndarray) -> float:
    residual = np.asarray(pose, np.float64) - np.asarray(target, np.float64)
    return float(np.mean(residual * residual, dtype=np.float64))


def _pair_map(active_pairs: Sequence[int]) -> bytes:
    mask = np.zeros(PAIR_COUNT, dtype=np.uint8)
    for pair in active_pairs:
        if not 0 <= int(pair) < PAIR_COUNT:
            raise QA43Error(f"pair outside n600: {pair}")
        if mask[int(pair)]:
            raise QA43Error(f"duplicate active pair: {pair}")
        mask[int(pair)] = 1
    packed = np.packbits(mask, bitorder="big").tobytes()
    if len(packed) != PAIR_MAP_BYTES:
        raise AssertionError("600-bit pair map is not exactly 75 bytes")
    return packed


def _source_bundle_sha256(paths: Sequence[Path]) -> str:
    rows = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise QA43Error(f"custody source is missing: {resolved}")
        rows.append(
            {
                "path": str(resolved),
                "sha256": _sha256(resolved.read_bytes()),
            }
        )
    return _sha256(_canonical_json(rows))


def _is_forbidden_state_path(path: Path) -> bool:
    text = str(path)
    return any(
        text == prefix.rstrip("/") or text.startswith(prefix)
        for prefix in _FORBIDDEN_STATE_PREFIXES
    )


class V4DWarpTailAdapter:
    """Concrete real-v4d adapter for the same-pool ``warp-tail`` program.

    The parent is parsed by the committed v4d receiver.  Candidate archives add
    the section consumed by ``inflate_runner_v4d_qa43_tail.Decoder``.  The
    adapter invokes that exact receiver class for every candidate parseback.
    It deliberately cannot instantiate ``terminal-frame0``.
    """

    _BASE_MEMBER_NAMES = (
        "manifest.json",
        "state/tokens.dr7t",
        "state/renderer.sec",
        "state/selector.sec",
        "state/pose_stub.sec",
        "state/pose_warp.stp",
    )

    def __init__(self, args: Mapping[str, str]) -> None:
        allowed = {"parent_archive", "receiver_deps_dir"}
        if set(args) - allowed:
            raise QA43Error(
                f"v4d-warp adapter received unknown args: {sorted(set(args) - allowed)}"
            )
        parent_value = args.get("parent_archive")
        if not parent_value:
            raise QA43Error("v4d-warp adapter requires parent_archive=/absolute/path")
        self._parent_path = Path(parent_value)
        if not self._parent_path.is_absolute() or not self._parent_path.is_file():
            raise QA43Error("v4d-warp parent_archive must be an existing absolute file")
        default_deps = (
            "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/"
            "d1/eval_root/submissions/pfs1"
        )
        self._deps_dir = Path(args.get("receiver_deps_dir", default_deps))
        if not self._deps_dir.is_absolute() or not self._deps_dir.is_dir():
            raise QA43Error("v4d-warp receiver_deps_dir must be an existing absolute dir")

        repo = Path(__file__).resolve().parents[1]
        experiments_dir = repo / "experiments"
        source_dir = repo / "src"
        for location in (str(experiments_dir), str(source_dir), str(self._deps_dir)):
            if location not in sys.path:
                sys.path.insert(0, location)

        self._solver_source = Path(__file__).resolve()
        self._tail_receiver_source = experiments_dir / "inflate_runner_v4d_qa43_tail.py"
        self._base_receiver_source = experiments_dir / "inflate_runner_v4d.py"
        self._r7_source = experiments_dir / "ddm_r7_token_coder.py"
        self._pfs1_receiver_source = self._deps_dir / "pfs1_warp_receiver.py"
        self._tr1_runtime_source = self._deps_dir / "ddm_tr1_runtime.py"
        self._repair_source = self._deps_dir / "repair_entropy_coder_runtime_adapters.py"
        self._scorer_source = experiments_dir / "ddm_p3v2_optimal_form_pose_resolve.py"
        self._upstream_root = Path(
            "/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709"
        )
        self._scorer_model = self._upstream_root / "models/posenet.safetensors"
        self._upstream_modules = self._upstream_root / "modules.py"
        self._targets_path = Path(
            "/Volumes/VertigoDataTier/pact/"
            "ddm_ms4_metric_producers_and_measurement_20260724T042005Z/"
            "pose_metric_n600_batch32.json"
        )
        for source in (
            self._solver_source,
            self._tail_receiver_source,
            self._base_receiver_source,
            self._r7_source,
            self._pfs1_receiver_source,
            self._tr1_runtime_source,
            self._repair_source,
            self._scorer_source,
            self._scorer_model,
            self._upstream_modules,
            self._targets_path,
        ):
            if not source.is_file():
                raise QA43Error(f"v4d-warp custody file is missing: {source}")

        self._tail_module = importlib.import_module("inflate_runner_v4d_qa43_tail")
        self._base_module = importlib.import_module("inflate_runner_v4d")
        if Path(self._tail_module.__file__).resolve() != self._tail_receiver_source:
            raise QA43Error("v4d-warp imported the wrong tail receiver source")
        if Path(self._base_module.__file__).resolve() != self._base_receiver_source:
            raise QA43Error("v4d-warp imported the wrong base receiver source")

        self._parent_archive = self._parent_path.read_bytes()
        (
            self._parent_names,
            self._parent_members,
            self._parent_compression,
        ) = self._read_zip(self._parent_archive)
        if self._parent_names != self._BASE_MEMBER_NAMES:
            raise QA43Error("v4d-warp parent member order/shape differs")
        if self._tail_module.TAIL_MEMBER in self._parent_members:
            raise QA43Error("v4d-warp parent already carries a QA43 tail")

        with tempfile.TemporaryDirectory(prefix="ddm-su2-parent-") as temporary:
            archive_dir = Path(temporary)
            for name, payload in self._parent_members.items():
                target = archive_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            try:
                self._parent_decoder = self._base_module.Decoder(archive_dir)
            except (Exception, SystemExit) as exc:
                raise QA43Error("v4d-warp parent receiver parse failed") from exc
        if int(self._parent_decoder.n_pairs) != PAIR_COUNT:
            raise QA43Error("v4d-warp parent is not n600")

        target_bundle = json.loads(self._targets_path.read_text())
        rows = target_bundle.get("rows")
        if (
            target_bundle.get("output_dimension") != 6
            or not isinstance(rows, list)
            or len(rows) != PAIR_COUNT
        ):
            raise QA43Error("v4d-warp target bundle differs")
        self._targets = np.stack(
            [np.asarray(row["center"], dtype=np.float64) for row in rows],
        )
        if self._targets.shape != (PAIR_COUNT, 6) or not np.all(
            np.isfinite(self._targets)
        ):
            raise QA43Error("v4d-warp targets are not finite n600x6")

        self._pose_module: ModuleType | None = None
        self._posenet: Any = None
        self._pair_cache_index: int | None = None
        self._pair_cache_value: np.ndarray | None = None
        self._decoder_cache: OrderedDict[str, tuple[Any, bytes]] = OrderedDict()
        self._public_decoder_cache: OrderedDict[str, tuple[Any, bytes]] = OrderedDict()

    def custody(self) -> Mapping[str, Any]:
        receiver_sources = (
            self._tail_receiver_source,
            self._base_receiver_source,
            self._pfs1_receiver_source,
            self._tr1_runtime_source,
            self._repair_source,
            self._r7_source,
        )
        scorer_sources = (
            self._scorer_source,
            self._upstream_modules,
            self._scorer_model,
        )
        solver_sha = _sha256(self._solver_source.read_bytes())
        return {
            "schema": "ddm_qa43_archive_adapter.v1",
            "program_kind": "warp-tail",
            "real_receiver": True,
            "uint8_closed": True,
            "outer_archive_parseback": True,
            "public_entrypoint_closed": True,
            "frame1_frozen": True,
            "frozen_pose_scorer": True,
            "all_tail_bytes_consumed": True,
            "pair_local_correction": True,
            "score_claim": False,
            "pair_count": PAIR_COUNT,
            "parent_archive_sha256": _sha256(self._parent_archive),
            "adapter_sha256": solver_sha,
            "builder_sha256": solver_sha,
            "receiver_sha256": _source_bundle_sha256(receiver_sources),
            "r7_source_sha256": _sha256(self._r7_source.read_bytes()),
            "scorer_sha256": _source_bundle_sha256(scorer_sources),
            "targets_sha256": _sha256(self._targets_path.read_bytes()),
            "receiver_sources": {
                str(path.resolve()): _sha256(path.read_bytes())
                for path in receiver_sources
            },
            "scorer_sources": {
                str(path.resolve()): _sha256(path.read_bytes())
                for path in scorer_sources
            },
            "warp_quanta": list(self._tail_module.V4D_WARP_QUANTA),
            "same_pool_as": ["PFS1", "TT1", "QA43-two-plane-warp"],
            "explicit_exclusion": "terminal-frame0/free-frame0",
        }

    def parent_archive(self) -> bytes:
        return self._parent_archive

    def coefficient_rank(self) -> int:
        return int(self._tail_module.COEFFICIENT_RANK)

    def parent_pair(self, pair_index: int) -> np.ndarray:
        self._check_pair(pair_index)
        if self._pair_cache_index != pair_index:
            frame1 = np.asarray(self._parent_decoder.f1(pair_index), dtype=np.uint8)
            frame0 = np.asarray(
                self._parent_decoder.f0(pair_index, frame1),
                dtype=np.uint8,
            )
            self._pair_cache_value = np.stack((frame0, frame1), axis=0)
            self._pair_cache_index = pair_index
        assert self._pair_cache_value is not None
        return self._pair_cache_value.copy()

    def initial_coefficients(self, pair_index: int) -> np.ndarray:
        self._check_pair(pair_index)
        return np.zeros(self.coefficient_rank(), dtype=np.int16)

    def target_pose6(self, pair_index: int) -> np.ndarray:
        self._check_pair(pair_index)
        return self._targets[pair_index].copy()

    def realize_pair(
        self,
        pair_index: int,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        self._check_pair(pair_index)
        values = _coefficients(
            coefficients,
            rank=self.coefficient_rank(),
            limit=7,
            name="v4d-warp coefficients",
        )
        pair_map = _pair_map([pair_index])
        tail = self._tail_module.encode_tail(
            values.reshape(1, -1),
            pair_map,
            codec="smevr",
        )
        manifest = self._candidate_manifest(tail, pair_map, "smevr")
        decoder = self._tail_module.Decoder.from_parent_decoder(
            self._parent_decoder,
            manifest,
            tail,
        )
        frame1 = self.parent_pair(pair_index)[1]
        frame0 = np.asarray(decoder.f0(pair_index, frame1), dtype=np.uint8)
        return np.stack((frame0, frame1), axis=0)

    def build_archive(
        self,
        updates: Mapping[int, np.ndarray],
        *,
        codec: str,
        pair_map: bytes,
    ) -> bytes:
        active = self._active_pairs(pair_map)
        if active != sorted(updates):
            raise QA43Error("v4d-warp update keys differ from pair map")
        values = np.stack(
            [
                _coefficients(
                    updates[pair],
                    rank=self.coefficient_rank(),
                    limit=7,
                    name=f"v4d-warp pair {pair} coefficients",
                )
                for pair in active
            ],
        ) if active else np.empty((0, self.coefficient_rank()), dtype=np.int16)
        tail = self._tail_module.encode_tail(values, pair_map, codec=codec)
        manifest = self._candidate_manifest(tail, pair_map, codec)
        members = {
            **self._parent_members,
            "manifest.json": _canonical_json(manifest),
            self._tail_module.TAIL_MEMBER: tail,
        }
        return self._write_zip(members)

    def decode_pair(self, archive: bytes, pair_index: int) -> np.ndarray:
        self._check_pair(pair_index)
        if archive == self._parent_archive:
            return self.parent_pair(pair_index)
        decoder, _ = self._decoder_for_archive(archive)
        frame1 = np.asarray(decoder.f1(pair_index), dtype=np.uint8)
        frame0 = np.asarray(decoder.f0(pair_index, frame1), dtype=np.uint8)
        return np.stack((frame0, frame1), axis=0)

    def public_decode_pair(self, archive: bytes, pair_index: int) -> np.ndarray:
        self._check_pair(pair_index)
        if archive == self._parent_archive:
            return self.parent_pair(pair_index)
        decoder, _ = self._public_decoder_for_archive(archive)
        frame1 = np.asarray(decoder.f1(pair_index), dtype=np.uint8)
        frame0 = np.asarray(decoder.f0(pair_index, frame1), dtype=np.uint8)
        return np.stack((frame0, frame1), axis=0)

    def score_pose6(self, pair_index: int, pair_u8: np.ndarray) -> np.ndarray:
        self._check_pair(pair_index)
        pair = _uint8_pair(pair_u8, "v4d-warp scorer pair")
        if self._pose_module is None:
            self._pose_module = importlib.import_module(
                "ddm_p3v2_optimal_form_pose_resolve"
            )
            if Path(self._pose_module.__file__).resolve() != self._scorer_source:
                raise QA43Error("v4d-warp imported the wrong scorer wrapper")
            self._posenet, _ = self._pose_module.load_posenet()
        return np.asarray(
            self._pose_module.pose6_u8(self._posenet, pair[0], pair[1]),
            dtype=np.float64,
        )

    def accounting(self, archive: bytes) -> Mapping[str, Any]:
        decoder, pair_map = self._decoder_for_archive(archive)
        names, members, _ = self._read_zip(archive)
        tail = members[self._tail_module.TAIL_MEMBER]
        return {
            "codec": decoder.qa43_tail_codec,
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "zip_members": list(names),
            "zip_member_uncompressed_bytes": sum(len(payload) for payload in members.values()),
            "pair_map_raw_bytes": PAIR_MAP_BYTES,
            "pair_map_sha256": _sha256(pair_map),
            "tail_member_bytes": len(tail),
            "tail_member_sha256": _sha256(tail),
            "r7_canonical_roundtrip": True,
            "zip_parseback": True,
            "all_bytes_consumed": True,
        }

    def extract_pair_map(self, archive: bytes) -> bytes:
        _, pair_map = self._decoder_for_archive(archive)
        return pair_map

    @staticmethod
    def _check_pair(pair_index: int) -> None:
        if (
            isinstance(pair_index, (bool, np.bool_))
            or not isinstance(pair_index, (int, np.integer))
            or not 0 <= int(pair_index) < PAIR_COUNT
        ):
            raise QA43Error("v4d-warp pair index outside n600")

    @staticmethod
    def _active_pairs(pair_map: bytes) -> list[int]:
        if len(pair_map) != PAIR_MAP_BYTES:
            raise QA43Error("v4d-warp pair map is not exactly 75 bytes")
        bits = np.unpackbits(
            np.frombuffer(pair_map, dtype=np.uint8),
            bitorder="big",
        )
        return [int(pair) for pair in np.flatnonzero(bits)]

    @staticmethod
    def _read_zip(
        archive: bytes,
    ) -> tuple[tuple[str, ...], dict[str, bytes], dict[str, int]]:
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as handle:
                infos = handle.infolist()
                names = tuple(info.filename for info in infos)
                if len(names) != len(set(names)):
                    raise QA43Error("candidate ZIP has duplicate members")
                for name in names:
                    path = Path(name)
                    if path.is_absolute() or ".." in path.parts:
                        raise QA43Error("candidate ZIP has an unsafe member")
                members = {info.filename: handle.read(info.filename) for info in infos}
                compression = {
                    info.filename: int(info.compress_type)
                    for info in infos
                }
        except (OSError, zipfile.BadZipFile) as exc:
            raise QA43Error("candidate outer archive is not a valid ZIP") from exc
        return names, members, compression

    def _write_zip(self, members: Mapping[str, bytes]) -> bytes:
        expected = (*self._parent_names, self._tail_module.TAIL_MEMBER)
        if tuple(members) != expected:
            raise QA43Error("v4d-warp candidate member order differs")
        stream = io.BytesIO()
        with zipfile.ZipFile(
            stream,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=False,
        ) as handle:
            for name, payload in members.items():
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.flag_bits = 0
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if name == self._tail_module.TAIL_MEMBER
                    else self._parent_compression[name]
                )
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    handle.writestr(info, payload, compresslevel=9)
                else:
                    handle.writestr(info, payload)
        result = stream.getvalue()
        names, parsed, _ = self._read_zip(result)
        if names != expected or parsed != dict(members):
            raise QA43Error("v4d-warp deterministic ZIP parseback differs")
        return result

    def _candidate_manifest(
        self,
        tail: bytes,
        pair_map: bytes,
        codec: str,
    ) -> dict[str, Any]:
        manifest = json.loads(self._parent_members["manifest.json"])
        manifest.update(
            {
                "qa43_tail_schema": self._tail_module.TAIL_SCHEMA,
                "qa43_tail_codec": codec,
                "qa43_tail_sha256": _sha256(tail),
                "qa43_pair_map_sha256": _sha256(pair_map),
                "qa43_warp_quanta": list(self._tail_module.V4D_WARP_QUANTA),
                "qa43_program_kind": "warp-tail",
                "qa43_score_claim": False,
            }
        )
        return manifest

    def _decoder_for_archive(self, archive: bytes) -> tuple[Any, bytes]:
        digest = _sha256(archive)
        cached = self._decoder_cache.get(digest)
        if cached is not None:
            self._decoder_cache.move_to_end(digest)
            return cached
        names, members, _ = self._read_zip(archive)
        expected_names = (*self._parent_names, self._tail_module.TAIL_MEMBER)
        if names != expected_names:
            raise QA43Error("v4d-warp candidate member order differs")
        for name in self._parent_names:
            if name != "manifest.json" and members[name] != self._parent_members[name]:
                raise QA43Error(f"v4d-warp candidate changed frozen base member: {name}")
        tail = members[self._tail_module.TAIL_MEMBER]
        try:
            codec, pair_map, _ = self._tail_module.parse_tail(tail)
        except Exception as exc:
            raise QA43Error("v4d-warp tail parseback failed") from exc
        manifest = json.loads(members["manifest.json"])
        if manifest != self._candidate_manifest(tail, pair_map, codec):
            raise QA43Error("v4d-warp candidate manifest differs")
        try:
            decoder = self._tail_module.Decoder.from_parent_decoder(
                self._parent_decoder,
                manifest,
                tail,
            )
        except (Exception, SystemExit) as exc:
            raise QA43Error("v4d-warp shipped receiver rejected candidate") from exc
        value = (decoder, pair_map)
        self._decoder_cache[digest] = value
        self._decoder_cache.move_to_end(digest)
        while len(self._decoder_cache) > 6:
            self._decoder_cache.popitem(last=False)
        return value

    def _public_decoder_for_archive(self, archive: bytes) -> tuple[Any, bytes]:
        digest = _sha256(archive)
        cached = self._public_decoder_cache.get(digest)
        if cached is not None:
            self._public_decoder_cache.move_to_end(digest)
            return cached
        _, expected_pair_map = self._decoder_for_archive(archive)
        _, members, _ = self._read_zip(archive)
        with tempfile.TemporaryDirectory(prefix="ddm-su2-public-receiver-") as temporary:
            archive_dir = Path(temporary)
            for name, payload in members.items():
                target = archive_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            try:
                decoder = self._tail_module.Decoder(archive_dir)
            except (Exception, SystemExit) as exc:
                raise QA43Error(
                    "v4d-warp shipped public receiver rejected candidate"
                ) from exc
        if decoder.qa43_pair_map != expected_pair_map:
            raise QA43Error("v4d-warp public receiver pair map differs")
        value = (decoder, expected_pair_map)
        self._public_decoder_cache[digest] = value
        self._public_decoder_cache.move_to_end(digest)
        while len(self._public_decoder_cache) > 4:
            self._public_decoder_cache.popitem(last=False)
        return value


def create_v4d_warp_adapter(args: Mapping[str, str]) -> V4DWarpTailAdapter:
    """CLI factory for the only currently concrete same-pool real receiver."""

    return V4DWarpTailAdapter(args)


def _validate_custody(
    adapter: QA43ArchiveAdapter,
    *,
    program_kind: str,
) -> tuple[dict[str, Any], bytes, int]:
    if not isinstance(adapter, QA43ArchiveAdapter):
        raise QA43Error("adapter does not implement the QA43ArchiveAdapter protocol")
    custody = dict(adapter.custody())
    if custody.get("schema") != "ddm_qa43_archive_adapter.v1":
        raise QA43Error("adapter custody schema differs")
    if custody.get("program_kind") != program_kind:
        raise QA43Error("adapter program_kind differs from requested program")
    for flag in (
        "real_receiver",
        "uint8_closed",
        "outer_archive_parseback",
        "public_entrypoint_closed",
        "frame1_frozen",
        "frozen_pose_scorer",
        "all_tail_bytes_consumed",
        "pair_local_correction",
    ):
        if custody.get(flag) is not True:
            raise QA43Error(f"adapter custody flag is not true: {flag}")
    if custody.get("score_claim") is not False:
        raise QA43Error("adapter must set score_claim=false")
    if custody.get("pair_count") != PAIR_COUNT:
        raise QA43Error("adapter must bind exactly n600")
    for name in (
        "parent_archive_sha256",
        "adapter_sha256",
        "builder_sha256",
        "receiver_sha256",
        "r7_source_sha256",
        "scorer_sha256",
        "targets_sha256",
    ):
        if not _is_sha256(custody.get(name)):
            raise QA43Error(f"adapter custody lacks lowercase SHA-256: {name}")
    parent = adapter.parent_archive()
    if not isinstance(parent, bytes) or not parent:
        raise QA43Error("adapter parent_archive must return nonempty bytes")
    if adapter.parent_archive() != parent:
        raise QA43Error("adapter parent_archive is not deterministic")
    if _sha256(parent) != custody["parent_archive_sha256"]:
        raise QA43Error("adapter parent archive differs from custody")
    rank = adapter.coefficient_rank()
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 64:
        raise QA43Error("adapter coefficient rank must be an integer in [1,64]")
    return custody, parent, rank


def _adapter_binding(
    custody: Mapping[str, Any],
    *,
    program_kind: str,
    rank: int,
    config: SolveConfig,
    stage_ks: tuple[int, ...],
    loader_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = {
        "schema": "ddm_su2_qa43_tail_solver_binding.v1",
        "program_kind": program_kind,
        "custody": dict(custody),
        "coefficient_rank": rank,
        "coefficient_limit": config.coefficient_limit,
        "relinearizations": config.relinearizations,
        "damping": config.damping,
        "line_search": list(config.line_search),
        "stage_ks": list(stage_ks),
        "rng": "none",
        "pair_count": PAIR_COUNT,
        "codecs": list(CODECS),
        "solver_source_sha256": _sha256(Path(__file__).read_bytes()),
        "adapter_loader": dict(loader_binding),
    }
    return payload, _sha256(_canonical_json(payload))


def _programmatic_loader_binding(
    adapter: QA43ArchiveAdapter,
) -> dict[str, Any]:
    adapter_type = type(adapter)
    module = sys.modules.get(adapter_type.__module__)
    source = getattr(module, "__file__", None)
    if not source:
        raise QA43Error("programmatic adapter module has no source path")
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise QA43Error("programmatic adapter source path is not a file")
    return {
        "module_spec": adapter_type.__module__,
        "module_path": str(source_path),
        "module_sha256": _sha256(source_path.read_bytes()),
        "factory": "programmatic",
        "adapter_args": {},
        "adapter_class": f"{adapter_type.__module__}.{adapter_type.__qualname__}",
    }


def _validate_accounting(
    adapter: QA43ArchiveAdapter,
    archive: bytes,
    *,
    codec: str,
    expected_pair_map: bytes,
) -> dict[str, Any]:
    accounting = dict(adapter.accounting(archive))
    if accounting.get("codec") != codec:
        raise QA43Error("adapter accounting codec differs")
    if accounting.get("archive_bytes") != len(archive):
        raise QA43Error("adapter accounting archive length differs")
    if accounting.get("archive_sha256") != _sha256(archive):
        raise QA43Error("adapter accounting archive SHA-256 differs")
    if accounting.get("pair_map_raw_bytes") != PAIR_MAP_BYTES:
        raise QA43Error("adapter did not account for the exact 75-byte pair map")
    if accounting.get("all_bytes_consumed") is not True:
        raise QA43Error("adapter reports unconsumed candidate bytes")
    if accounting.get("r7_canonical_roundtrip") is not True:
        raise QA43Error("adapter did not prove canonical R7 roundtrip")
    if accounting.get("zip_parseback") is not True:
        raise QA43Error("adapter did not prove outer ZIP parseback")
    parsed_pair_map = adapter.extract_pair_map(archive)
    if (
        not isinstance(parsed_pair_map, bytes)
        or len(parsed_pair_map) != PAIR_MAP_BYTES
        or parsed_pair_map != expected_pair_map
    ):
        raise QA43Error("receiver-consumed pair map differs from requested map")
    if accounting.get("pair_map_sha256") != _sha256(parsed_pair_map):
        raise QA43Error("adapter accounting pair-map SHA-256 differs")
    tail_member = accounting.get("tail_member_bytes")
    if isinstance(tail_member, bool) or not isinstance(tail_member, int) or tail_member <= 0:
        raise QA43Error("adapter accounting tail_member_bytes is invalid")
    return accounting


def _build_deterministic_archive(
    adapter: QA43ArchiveAdapter,
    updates: Mapping[int, np.ndarray],
    *,
    codec: str,
) -> tuple[bytes, dict[str, Any]]:
    if codec not in CODECS:
        raise QA43Error(f"unsupported candidate codec: {codec}")
    active = sorted(updates)
    pair_map = _pair_map(active)
    archive = adapter.build_archive(updates, codec=codec, pair_map=pair_map)
    if not isinstance(archive, bytes) or not archive:
        raise QA43Error("adapter build_archive must return nonempty bytes")
    repeat = adapter.build_archive(updates, codec=codec, pair_map=pair_map)
    if repeat != archive:
        raise QA43Error("candidate outer archive is not deterministic")
    accounting = _validate_accounting(
        adapter,
        archive,
        codec=codec,
        expected_pair_map=pair_map,
    )
    return archive, accounting


def _evaluate_candidate(
    adapter: QA43ArchiveAdapter,
    *,
    pair: int,
    coefficients: np.ndarray,
    updates: Mapping[int, np.ndarray],
    codec: str,
) -> PairEvaluation:
    population = {int(key): np.asarray(value, dtype=np.int16).copy() for key, value in updates.items()}
    population[pair] = np.asarray(coefficients, dtype=np.int16).copy()
    archive, _ = _build_deterministic_archive(adapter, population, codec=codec)
    decoded = _uint8_pair(adapter.decode_pair(archive, pair), "decoded candidate pair")
    realized = _uint8_pair(
        adapter.realize_pair(pair, np.asarray(coefficients, dtype=np.int16)),
        "adapter realized pair",
    )
    parent = _uint8_pair(adapter.parent_pair(pair), "parent pair")
    if not np.array_equal(decoded, realized):
        raise QA43Error(
            f"pair {pair}: exact archive decode differs from proposed receiver realization"
        )
    if not np.array_equal(decoded[1], parent[1]):
        raise QA43Error(f"pair {pair}: terminal correction changed frozen frame1")
    pose = _pose6(adapter.score_pose6(pair, decoded), "scorer pose6")
    target = _pose6(adapter.target_pose6(pair), "target pose6")
    return PairEvaluation(
        pair=pair,
        coefficients=tuple(int(value) for value in coefficients),
        pose6=tuple(float(value) for value in pose),
        d_pose=_d_pose(pose, target),
        archive_bytes=len(archive),
        archive_sha256=_sha256(archive),
    )


def _fresh_baseline(
    adapter: QA43ArchiveAdapter,
    *,
    state_dir: Path,
    binding_sha256: str,
    parent_archive: bytes,
    deadline: float | None,
) -> list[dict[str, Any]]:
    path = state_dir / "baseline_n600.json"
    if path.exists():
        cached = _load_json(path)
        if cached.get("binding_sha256") != binding_sha256:
            raise QA43Error("baseline checkpoint binding differs")
        rows = cached.get("rows")
        if not isinstance(rows, list) or len(rows) != PAIR_COUNT:
            raise QA43Error("baseline checkpoint is not n600")
        return [dict(row) for row in rows]

    partial_dir = state_dir / "baseline_pairs"
    partial_dir.mkdir(parents=True, exist_ok=True)
    row_paths = sorted(partial_dir.glob("pair_*.json"))
    rows = [_load_json(row_path) for row_path in row_paths]
    for expected_pair, (row_path, row) in enumerate(zip(row_paths, rows, strict=True)):
        expected_name = f"pair_{expected_pair:03d}.json"
        if row_path.name != expected_name:
            raise QA43Error("baseline pair checkpoints are not a contiguous prefix")
        if row.get("binding_sha256") != binding_sha256:
            raise QA43Error("baseline partial checkpoint binding differs")
        if row.get("pair") != expected_pair:
            raise QA43Error("baseline partial checkpoint is not an exact n600 prefix")
    if len(rows) > PAIR_COUNT:
        raise QA43Error("baseline partial checkpoint exceeds n600")

    for pair in range(len(rows), PAIR_COUNT):
        if deadline is not None and time.monotonic() >= deadline:
            return rows
        parent_pair = _uint8_pair(adapter.parent_pair(pair), "parent pair")
        decoded = _uint8_pair(
            adapter.decode_pair(parent_archive, pair),
            "decoded parent pair",
        )
        if not np.array_equal(decoded, parent_pair):
            raise QA43Error(f"pair {pair}: parent archive parseback differs")
        pose = _pose6(adapter.score_pose6(pair, decoded), "baseline scorer pose6")
        target = _pose6(adapter.target_pose6(pair), "target pose6")
        row = {
            "schema": "ddm_su2_qa43_baseline_pair.v1",
            "binding_sha256": binding_sha256,
            "pair": pair,
            "d_pose": _d_pose(pose, target),
            "pose6": [float(value) for value in pose],
            "target_pose6": [float(value) for value in target],
            "parent_pair_sha256": _sha256(decoded.tobytes()),
        }
        _atomic_json(partial_dir / f"pair_{pair:03d}.json", row)
        rows.append(row)
    payload = {
        "schema": "ddm_su2_qa43_baseline.v1",
        "binding_sha256": binding_sha256,
        "pair_count": PAIR_COUNT,
        "rows": rows,
        "axis": AXIS,
        "score_claim": False,
        "pointer": POINTER,
    }
    _atomic_json(path, payload)
    return rows


def _solve_pair(
    adapter: QA43ArchiveAdapter,
    *,
    pair: int,
    initial: np.ndarray,
    accepted_updates: Mapping[int, np.ndarray],
    config: SolveConfig,
) -> tuple[np.ndarray, PairEvaluation, PairEvaluation, list[dict[str, Any]]]:
    current = np.asarray(initial, dtype=np.int16).copy()
    initial_eval = _evaluate_candidate(
        adapter,
        pair=pair,
        coefficients=current,
        updates=accepted_updates,
        codec="smevr",
    )
    current_eval = initial_eval
    target = _pose6(adapter.target_pose6(pair), "target pose6")
    traces: list[dict[str, Any]] = []
    rank = current.size

    for iteration in range(config.relinearizations):
        before = current.copy()
        before_eval = current_eval
        jacobian = np.zeros((6, rank), dtype=np.float64)
        fd_evaluations = 0
        for coordinate in range(rank):
            plus = current.astype(np.int64)
            minus = current.astype(np.int64)
            plus[coordinate] = min(config.coefficient_limit, int(plus[coordinate]) + 1)
            minus[coordinate] = max(-config.coefficient_limit, int(minus[coordinate]) - 1)
            denominator = int(plus[coordinate] - minus[coordinate])
            if denominator == 0:
                raise QA43Error("finite-difference coordinate is pinned")
            plus_eval = _evaluate_candidate(
                adapter,
                pair=pair,
                coefficients=plus.astype(np.int16),
                updates=accepted_updates,
                codec="smevr",
            )
            minus_eval = _evaluate_candidate(
                adapter,
                pair=pair,
                coefficients=minus.astype(np.int16),
                updates=accepted_updates,
                codec="smevr",
            )
            fd_evaluations += 2
            jacobian[:, coordinate] = (
                np.asarray(plus_eval.pose6) - np.asarray(minus_eval.pose6)
            ) / denominator

        residual = target - np.asarray(current_eval.pose6, dtype=np.float64)
        normal = jacobian.T @ jacobian + config.damping * np.eye(rank)
        rhs = jacobian.T @ residual
        try:
            delta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        if not np.all(np.isfinite(delta)):
            raise QA43Error("GN update is nonfinite")

        best: tuple[float, np.ndarray, PairEvaluation] | None = None
        seen = {current.tobytes()}
        for alpha in config.line_search:
            candidate = np.rint(current.astype(np.float64) + alpha * delta)
            candidate = np.clip(
                candidate,
                -config.coefficient_limit,
                config.coefficient_limit,
            ).astype(np.int16)
            if candidate.tobytes() in seen:
                continue
            seen.add(candidate.tobytes())
            evaluation = _evaluate_candidate(
                adapter,
                pair=pair,
                coefficients=candidate,
                updates=accepted_updates,
                codec="smevr",
            )
            if evaluation.d_pose < before_eval.d_pose and (
                best is None or evaluation.d_pose < best[2].d_pose
            ):
                best = (alpha, candidate.copy(), evaluation)

        admitted = best is not None
        alpha_used: float | None = None
        if best is not None:
            alpha_used, current, current_eval = best
        traces.append(
            {
                "iteration": iteration,
                "coefficients_before": [int(value) for value in before],
                "coefficients_after": [int(value) for value in current],
                "d_pose_before": before_eval.d_pose,
                "d_pose_after": current_eval.d_pose,
                "fd_evaluations": fd_evaluations,
                "line_search_alpha": alpha_used,
                "admitted": admitted,
            }
        )
        if not admitted:
            break
    return current, initial_eval, current_eval, traces


def _read_solved_rows(
    directory: Path,
    *,
    binding_sha256: str,
    rank: int,
    limit: int,
) -> dict[int, dict[str, Any]]:
    solved: dict[int, dict[str, Any]] = {}
    directory.mkdir(parents=True, exist_ok=True)
    row_paths = sorted(directory.glob("rank_*.json"))
    for rank_index, row_path in enumerate(row_paths):
        row = _load_json(row_path)
        pair = row.get("pair")
        if isinstance(pair, bool) or not isinstance(pair, int) or not 0 <= pair < PAIR_COUNT:
            raise QA43Error("pair checkpoint id differs")
        expected_name = f"rank_{rank_index:03d}_pair_{pair:03d}.json"
        if row_path.name != expected_name:
            raise QA43Error("pair checkpoints are not a contiguous rank prefix")
        if row.get("binding_sha256") != binding_sha256:
            raise QA43Error("pair checkpoint binding differs")
        if row.get("fresh_tail_rank") != rank_index:
            raise QA43Error(f"pair {pair} checkpoint has the wrong fresh-tail rank")
        _coefficients(
            row.get("coefficients_final"),
            rank=rank,
            limit=limit,
            name="checkpoint coefficients",
        )
        previous = solved.get(pair)
        if previous is not None and previous != row:
            raise QA43Error(f"pair {pair} has non-identical duplicate checkpoints")
        solved[pair] = row
    return solved


def _stage_checkpoint(
    adapter: QA43ArchiveAdapter,
    *,
    state_dir: Path,
    binding_sha256: str,
    baseline: Sequence[Mapping[str, Any]],
    order: Sequence[int],
    solved: Mapping[int, Mapping[str, Any]],
    k: int,
    parent_archive: bytes,
    rank: int,
    limit: int,
) -> dict[str, Any]:
    receipt_path = state_dir / f"stage_k{k:03d}.json"
    archive_path = state_dir / f"candidate_k{k:03d}.archive.zip"
    attempted = list(order[:k])
    if any(pair not in solved for pair in attempted):
        missing = [pair for pair in attempted if pair not in solved]
        raise QA43Error(f"stage k={k} is incomplete; missing {missing[:8]}")
    active = [
        pair
        for pair in attempted
        if solved[pair].get("strict_realized_improvement_vs_parent") is True
    ]
    updates = {
        pair: _coefficients(
            solved[pair]["coefficients_final"],
            rank=rank,
            limit=limit,
            name=f"pair {pair} final coefficients",
        )
        for pair in active
    }
    nulls: dict[str, tuple[bytes, dict[str, Any]]] = {}
    candidates: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for codec in CODECS:
        nulls[codec] = _build_deterministic_archive(adapter, {}, codec=codec)
        candidates[codec] = _build_deterministic_archive(
            adapter,
            updates,
            codec=codec,
        )

    chosen_codec = min(
        CODECS,
        key=lambda codec: (len(candidates[codec][0]), codec),
    )
    chosen_archive, chosen_accounting = candidates[chosen_codec]
    chosen_null, chosen_null_accounting = nulls[chosen_codec]

    # Full-population closure is mandatory.  Every same-codec null must decode
    # the exact parent, every inactive candidate pair must remain exact, every
    # active candidate must equal the standalone receiver realization, and
    # both codec arms must be semantically identical on all 600 pairs.
    d_final = np.empty(PAIR_COUNT, dtype=np.float64)
    decoded_sha: dict[int, str] = {}
    active_set = set(active)
    for pair in range(PAIR_COUNT):
        parent = _uint8_pair(adapter.parent_pair(pair), "parent pair")
        expected = (
            _uint8_pair(adapter.realize_pair(pair, updates[pair]), "realized pair")
            if pair in active_set
            else parent
        )
        candidate_decoded: dict[str, np.ndarray] = {}
        for codec in CODECS:
            null_decoded = _uint8_pair(
                adapter.decode_pair(nulls[codec][0], pair),
                f"{codec} null decoded pair",
            )
            if not np.array_equal(null_decoded, parent):
                raise QA43Error(f"pair {pair}: {codec} same-codec null changed parent")
            decoded = _uint8_pair(
                adapter.decode_pair(candidates[codec][0], pair),
                f"{codec} decoded pair",
            )
            if not np.array_equal(decoded, expected):
                raise QA43Error(
                    f"pair {pair}: {codec} candidate differs from expected receiver pair"
                )
            if not np.array_equal(decoded[1], parent[1]):
                raise QA43Error(f"pair {pair}: {codec} candidate changed frame1")
            candidate_decoded[codec] = decoded
        if not np.array_equal(
            candidate_decoded["smevr"],
            candidate_decoded["brotli11"],
        ):
            raise QA43Error(f"pair {pair}: SMEVR and Brotli decode semantics differ")
        chosen_pair = candidate_decoded[chosen_codec]
        pose = _pose6(adapter.score_pose6(pair, chosen_pair), "stage scorer pose6")
        target = _pose6(adapter.target_pose6(pair), "target pose6")
        d_final[pair] = _d_pose(pose, target)
        decoded_sha[pair] = _sha256(chosen_pair.tobytes())

    parent_bytes = len(parent_archive)
    whole_action_delta_bytes = len(chosen_archive) - parent_bytes
    fixed_empty_tail_and_repack_delta_bytes = len(chosen_null) - parent_bytes
    tail_marginal_bytes = len(chosen_archive) - len(chosen_null)
    d_base = np.asarray([float(row["d_pose"]) for row in baseline], dtype=np.float64)
    d_base_mean = float(d_base.mean())
    d_final_mean = float(d_final.mean())
    pose_base = math.sqrt(10.0 * d_base_mean)
    pose_final = math.sqrt(10.0 * d_final_mean)
    rate_delta = 25.0 * whole_action_delta_bytes / RATE_DENOMINATOR
    tail_per_attempted = tail_marginal_bytes / k
    tail_per_admitted = tail_marginal_bytes / len(active) if active else None
    whole_action_per_admitted = (
        whole_action_delta_bytes / len(active) if active else None
    )
    tail_member_delta = (
        int(chosen_accounting["tail_member_bytes"])
        - int(chosen_null_accounting["tail_member_bytes"])
    )
    receipt: dict[str, Any] = {
        "schema": "ddm_su2_qa43_tail_stage.v2",
        "binding_sha256": binding_sha256,
        "stage_k": k,
        "attempted_pairs": attempted,
        "active_pairs": active,
        "admitted_pairs": len(active),
        "active_pair_map_bytes": PAIR_MAP_BYTES,
        "coefficient_rank": rank,
        "codecs": {
            codec: {
                "null": {
                    "archive_bytes": len(nulls[codec][0]),
                    "archive_sha256": _sha256(nulls[codec][0]),
                    "accounting": nulls[codec][1],
                },
                "candidate": {
                    "archive_bytes": len(candidates[codec][0]),
                    "archive_sha256": _sha256(candidates[codec][0]),
                    "accounting": candidates[codec][1],
                },
            }
            for codec in CODECS
        },
        "chosen_codec": chosen_codec,
        "parent_archive_bytes": parent_bytes,
        "parent_archive_sha256": _sha256(parent_archive),
        "same_codec_null_archive_bytes": len(chosen_null),
        "same_codec_null_archive_sha256": _sha256(chosen_null),
        "candidate_archive_bytes": len(chosen_archive),
        "candidate_archive_sha256": _sha256(chosen_archive),
        "whole_action_delta_bytes_vs_parent": whole_action_delta_bytes,
        "fixed_empty_tail_and_repack_delta_bytes_vs_parent": (
            fixed_empty_tail_and_repack_delta_bytes
        ),
        "whole_action_delta_bytes_per_admitted_pair": whole_action_per_admitted,
        "tail_marginal_bytes_vs_same_codec_null": tail_marginal_bytes,
        "tail_marginal_bytes_per_attempted_pair": tail_per_attempted,
        "tail_marginal_bytes_per_admitted_pair": tail_per_admitted,
        "tail_member_delta_bytes_vs_null": tail_member_delta,
        "d_pose_mean_parent": d_base_mean,
        "d_pose_mean_candidate": d_final_mean,
        "pose_contribution_parent": pose_base,
        "pose_contribution_candidate": pose_final,
        "delta_pose_contribution": pose_final - pose_base,
        "delta_rate_contribution": rate_delta,
        "delta_joint_pose_rate": pose_final - pose_base + rate_delta,
        "attempted_pairs_at_or_below_1e3": int(
            sum(d_final[pair] <= 1.0e-3 for pair in attempted)
        ),
        "all_n600_frame1_frozen": True,
        "all_n600_codec_semantics_equal": True,
        "decoded_pair_sha256": {str(pair): digest for pair, digest in decoded_sha.items()},
        "falsifiers": {
            "no_pairs_admitted": not active,
            "tail_price_gt_600B_per_admitted_pair": (
                not active
                or (
                    whole_action_per_admitted is not None
                    and whole_action_per_admitted > 600.0
                )
            ),
            "archive_level_joint_action_not_improved": (
                pose_final - pose_base + rate_delta >= 0.0
            ),
        },
        "axis": AXIS,
        "score_claim": False,
        "pointer": POINTER,
        "verdict_scope": (
            "INSTANCE: exact adapter parent, selected basis, integer lattice, "
            f"and nested top-{k} population"
        ),
    }
    if receipt_path.exists() or archive_path.exists():
        if not receipt_path.is_file() or not archive_path.is_file():
            raise QA43Error(f"stage k={k} cache is only partially present")
        if _load_json(receipt_path) != receipt:
            raise QA43Error(f"stage k={k} recomputed receipt differs")
        if archive_path.read_bytes() != chosen_archive:
            raise QA43Error(f"stage k={k} recomputed archive differs")
    else:
        _atomic_write(archive_path, chosen_archive)
        _atomic_json(receipt_path, receipt)
    return receipt


def run_solver(
    adapter: QA43ArchiveAdapter,
    *,
    program_kind: str,
    state_dir: Path,
    config: SolveConfig,
    min_free_bytes: int,
    max_seconds: float = 0.0,
    stage_ks: tuple[int, ...] = STAGE_KS,
    loader_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or resume one exact-adapter top-k solve.

    ``stage_ks`` is injectable only for focused tests.  The CLI refuses every
    production value except the registered (56,112,200) ladder.
    """

    if program_kind not in PROGRAM_KINDS:
        raise QA43Error(f"unknown program kind: {program_kind}")
    if (
        isinstance(min_free_bytes, bool)
        or not isinstance(min_free_bytes, int)
        or min_free_bytes < 0
    ):
        raise QA43Error("min_free_bytes must be a nonnegative integer")
    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not math.isfinite(max_seconds)
        or max_seconds < 0
    ):
        raise QA43Error("max_seconds must be finite and nonnegative")
    if (
        not isinstance(stage_ks, tuple)
        or not stage_ks
        or any(
            isinstance(k, bool) or not isinstance(k, int)
            for k in stage_ks
        )
        or tuple(sorted(set(stage_ks))) != stage_ks
        or any(not 1 <= k <= PAIR_COUNT for k in stage_ks)
    ):
        raise QA43Error("stage_ks must be unique, increasing integers inside n600")
    if not state_dir.is_absolute():
        raise QA43Error("--resume-from must be an absolute path")
    if _is_forbidden_state_path(state_dir):
        raise QA43Error("--resume-from may not use a /tmp-class tier")
    if _is_forbidden_state_path(state_dir.resolve(strict=False)):
        raise QA43Error("--resume-from may not resolve into a /tmp-class tier")
    state_dir.mkdir(parents=True, exist_ok=True)
    if state_dir.is_symlink():
        raise QA43Error("--resume-from may not be a symlink")
    free_bytes = shutil.disk_usage(state_dir).free
    if free_bytes < min_free_bytes:
        raise QA43Error(
            f"storage preflight failed: {free_bytes} free < {min_free_bytes} required"
        )
    custody, parent_archive, rank = _validate_custody(
        adapter,
        program_kind=program_kind,
    )
    if loader_binding is None:
        loader_binding = _programmatic_loader_binding(adapter)
    binding, binding_sha256 = _adapter_binding(
        custody,
        program_kind=program_kind,
        rank=rank,
        config=config,
        stage_ks=stage_ks,
        loader_binding=loader_binding,
    )
    manifest_path = state_dir / "manifest.json"
    manifest = {
        **binding,
        "binding_sha256": binding_sha256,
        "storage_preflight": {
            "path": str(state_dir),
            "free_bytes": free_bytes,
            "required_free_bytes": min_free_bytes,
            "status": "PASS",
        },
        "axis": AXIS,
        "score_claim": False,
        "pointer": POINTER,
    }
    if manifest_path.exists():
        existing_manifest = _load_json(manifest_path)
        if any(existing_manifest.get(key) != value for key, value in binding.items()):
            raise QA43Error("resume manifest differs from requested run")
        if existing_manifest.get("binding_sha256") != binding_sha256:
            raise QA43Error("resume manifest binding digest differs")
    else:
        _atomic_json(manifest_path, manifest)

    target_count = max(stage_ks)
    started = time.monotonic()
    deadline = started + max_seconds if max_seconds > 0 else None
    baseline = _fresh_baseline(
        adapter,
        state_dir=state_dir,
        binding_sha256=binding_sha256,
        parent_archive=parent_archive,
        deadline=deadline,
    )
    if len(baseline) != PAIR_COUNT:
        summary = {
            "schema": "ddm_su2_qa43_tail_solver.v1",
            "binding_sha256": binding_sha256,
            "program_kind": program_kind,
            "pair_count": PAIR_COUNT,
            "baseline_pairs_replayed": len(baseline),
            "pairs_solved": 0,
            "required_pairs": target_count,
            "stage_ks_complete": [],
            "halted_by_falsifier": None,
            "complete": False,
            "resume_from": str(state_dir),
            "axis": AXIS,
            "score_claim": False,
            "pointer": POINTER,
            "main_landing_review_required": True,
            "verdict": "IN_PROGRESS_BASELINE_RESUME_REQUIRED",
        }
        _atomic_json(state_dir / "summary.json", summary)
        return summary
    order = sorted(
        range(PAIR_COUNT),
        key=lambda pair: (-float(baseline[pair]["d_pose"]), pair),
    )
    order_payload = {
        "schema": "ddm_su2_qa43_fresh_tail_order.v1",
        "binding_sha256": binding_sha256,
        "pair_count": PAIR_COUNT,
        "order": order,
        "d_pose_ordered": [float(baseline[pair]["d_pose"]) for pair in order],
    }
    order_path = state_dir / "fresh_tail_order.json"
    if order_path.exists():
        if _load_json(order_path) != order_payload:
            raise QA43Error("fresh tail ordering checkpoint differs")
    else:
        _atomic_json(order_path, order_payload)

    pair_dir = state_dir / "pair_solves"
    solved = _read_solved_rows(
        pair_dir,
        binding_sha256=binding_sha256,
        rank=rank,
        limit=config.coefficient_limit,
    )
    expected_prefix = order[: len(solved)]
    if set(solved) != set(expected_prefix):
        raise QA43Error("pair checkpoints are not an exact fresh-tail prefix")
    for rank_index, pair in enumerate(expected_prefix):
        if solved[pair].get("fresh_tail_rank") != rank_index:
            raise QA43Error(f"pair {pair} checkpoint has the wrong fresh-tail rank")
    accepted_updates: dict[int, np.ndarray] = {
        pair: _coefficients(
            row["coefficients_final"],
            rank=rank,
            limit=config.coefficient_limit,
            name=f"resumed pair {pair} coefficients",
        )
        for pair, row in solved.items()
        if row.get("strict_realized_improvement_vs_parent") is True
    }
    stages: list[dict[str, Any]] = []
    halted_by_falsifier: dict[str, Any] | None = None
    for k in stage_ks:
        for rank_index, pair in enumerate(order[:k]):
            if pair in solved:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                break
            initial = _coefficients(
                adapter.initial_coefficients(pair),
                rank=rank,
                limit=config.coefficient_limit,
                name=f"pair {pair} initial coefficients",
            )
            final, initial_eval, final_eval, traces = _solve_pair(
                adapter,
                pair=pair,
                initial=initial,
                accepted_updates=accepted_updates,
                config=config,
            )
            row = {
                "schema": "ddm_su2_qa43_pair_solve.v1",
                "binding_sha256": binding_sha256,
                "pair": pair,
                "fresh_tail_rank": rank_index,
                "coefficients_initial": [int(value) for value in initial],
                "coefficients_final": [int(value) for value in final],
                "d_pose_initial": initial_eval.d_pose,
                "d_pose_final": final_eval.d_pose,
                "strict_realized_improvement": final_eval.d_pose < initial_eval.d_pose,
                "d_pose_parent": float(baseline[pair]["d_pose"]),
                "strict_realized_improvement_vs_parent": (
                    final_eval.d_pose < float(baseline[pair]["d_pose"])
                ),
                "initial_archive_sha256": initial_eval.archive_sha256,
                "final_archive_sha256": final_eval.archive_sha256,
                "traces": traces,
                "axis": AXIS,
                "score_claim": False,
            }
            _atomic_json(
                pair_dir / f"rank_{rank_index:03d}_pair_{pair:03d}.json",
                row,
            )
            solved[pair] = row
            if row["strict_realized_improvement_vs_parent"]:
                accepted_updates[pair] = final.copy()
        if not all(pair in solved for pair in order[:k]):
            break
        stage = _stage_checkpoint(
            adapter,
            state_dir=state_dir,
            binding_sha256=binding_sha256,
            baseline=baseline,
            order=order,
            solved=solved,
            k=k,
            parent_archive=parent_archive,
            rank=rank,
            limit=config.coefficient_limit,
        )
        stages.append(stage)
        fired = [
            name
            for name, value in dict(stage["falsifiers"]).items()
            if value is True
        ]
        if fired:
            halted_by_falsifier = {
                "stage_k": k,
                "falsifiers": fired,
                "required_action": "REFIT_BEFORE_NEXT_NESTED_STAGE",
            }
            break

    complete = len(stages) == len(stage_ks) and halted_by_falsifier is None
    summary = {
        "schema": "ddm_su2_qa43_tail_solver.v1",
        "binding_sha256": binding_sha256,
        "program_kind": program_kind,
        "pair_count": PAIR_COUNT,
        "pairs_solved": len(solved),
        "required_pairs": target_count,
        "stage_ks_complete": [int(stage["stage_k"]) for stage in stages],
        "halted_by_falsifier": halted_by_falsifier,
        "complete": complete,
        "resume_from": str(state_dir),
        "axis": AXIS,
        "score_claim": False,
        "pointer": POINTER,
        "main_landing_review_required": True,
        "verdict": (
            "MEASURED_ADVISORY_EXACT_ADAPTER_LADDER"
            if complete
            else (
                "HARD_FALSIFIER_REFIT_REQUIRED"
                if halted_by_falsifier is not None
                else "IN_PROGRESS_RESUME_REQUIRED"
            )
        ),
    }
    _atomic_json(state_dir / "summary.json", summary)
    return summary


def validate_adapter(
    adapter: QA43ArchiveAdapter,
    *,
    program_kind: str,
    coefficient_limit: int = 7,
) -> dict[str, Any]:
    """Scorer-free closure check for both candidate coders on two pairs."""

    custody, parent, rank = _validate_custody(adapter, program_kind=program_kind)
    for pair in (0, PAIR_COUNT - 1):
        _coefficients(
            adapter.initial_coefficients(pair),
            rank=rank,
            limit=coefficient_limit,
            name=f"pair {pair} initial coefficients",
        )
    updates = {
        pair: np.asarray(
            [
                coefficient_limit
                if (coordinate + pair) % 2 == 0
                else -coefficient_limit
                for coordinate in range(rank)
            ],
            dtype=np.int16,
        )
        for pair in (0, PAIR_COUNT - 1)
    }
    candidates = {
        codec: _build_deterministic_archive(adapter, updates, codec=codec)
        for codec in CODECS
    }
    for pair, coefficients in updates.items():
        parent_pair = _uint8_pair(adapter.parent_pair(pair), "parent pair")
        realized = _uint8_pair(adapter.realize_pair(pair, coefficients), "realized pair")
        if np.array_equal(realized[0], parent_pair[0]):
            raise QA43Error(
                f"pair {pair}: nonzero coefficients did not change receiver frame0"
            )
        for codec, (archive, _) in candidates.items():
            decoded = _uint8_pair(adapter.decode_pair(archive, pair), f"{codec} pair")
            public_decoded = _uint8_pair(
                adapter.public_decode_pair(archive, pair),
                f"{codec} public receiver pair",
            )
            if not np.array_equal(decoded, realized):
                raise QA43Error(f"{codec}: decoded pair differs from receiver realization")
            if not np.array_equal(public_decoded, realized):
                raise QA43Error(
                    f"{codec}: public receiver differs from receiver realization"
                )
            if not np.array_equal(decoded[1], parent_pair[1]):
                raise QA43Error(f"{codec}: frame1 changed")
    return {
        "schema": "ddm_su2_qa43_adapter_validation.v1",
        "program_kind": program_kind,
        "custody": custody,
        "parent_archive_bytes": len(parent),
        "parent_archive_sha256": _sha256(parent),
        "coefficient_rank": rank,
        "codecs": {
            codec: {
                "archive_bytes": len(candidate[0]),
                "archive_sha256": _sha256(candidate[0]),
                "accounting": candidate[1],
            }
            for codec, candidate in candidates.items()
        },
        "pair_map_bytes": PAIR_MAP_BYTES,
        "nonzero_frame0_change_pairs": sorted(updates),
        "public_entrypoint_closed": True,
        "status": "PASS",
        "axis": AXIS,
        "score_claim": False,
        "pointer": POINTER,
    }


def _load_module(module_spec: str) -> tuple[ModuleType, str]:
    module_name, separator, factory_name = module_spec.rpartition(":")
    if not separator or not module_name or not factory_name:
        raise QA43Error("--receiver-adapter must be MODULE:FACTORY or /path/file.py:FACTORY")
    candidate = Path(module_name)
    if candidate.suffix == ".py" or candidate.is_absolute():
        if not candidate.is_file():
            raise QA43Error(f"adapter module path does not exist: {candidate}")
        synthetic = f"_ddm_su2_adapter_{_sha256(str(candidate.resolve()).encode())[:12]}"
        spec = importlib.util.spec_from_file_location(synthetic, candidate)
        if spec is None or spec.loader is None:
            raise QA43Error(f"cannot import adapter module: {candidate}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[synthetic] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return module, factory_name


def _adapter_args(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or key in result:
            raise QA43Error("--adapter-arg values must be unique KEY=VALUE pairs")
        result[key] = item
    return result


def load_adapter(
    module_spec: str,
    values: Sequence[str],
) -> tuple[QA43ArchiveAdapter, dict[str, Any]]:
    module, factory_name = _load_module(module_spec)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise QA43Error(f"adapter factory is not callable: {module_spec}")
    parsed_args = _adapter_args(values)
    adapter = factory(parsed_args)
    if not isinstance(adapter, QA43ArchiveAdapter):
        raise QA43Error("adapter factory returned an incompatible object")
    module_path = Path(module.__file__).resolve() if module.__file__ else None
    if module_path is None or not module_path.is_file():
        raise QA43Error("adapter module has no hashable source file")
    return adapter, {
        "module_spec": module_spec,
        "module_path": str(module_path),
        "module_sha256": _sha256(module_path.read_bytes()),
        "factory": factory_name,
        "adapter_args": parsed_args,
        "adapter_class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
    }


def _parse_top_k(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--top-k must be comma-separated integers") from exc
    if result != STAGE_KS:
        raise argparse.ArgumentTypeError("--top-k is fixed at 56,112,200")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "solve"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--program-kind", choices=PROGRAM_KINDS, required=True)
        sub.add_argument("--receiver-adapter", required=True)
        sub.add_argument("--adapter-arg", action="append", default=[])
        if command == "solve":
            sub.add_argument("--top-k", type=_parse_top_k, default=STAGE_KS)
            sub.add_argument("--relinearizations", type=int, choices=(2, 3), default=3)
            sub.add_argument("--damping", type=float, default=1.0e-3)
            sub.add_argument("--coefficient-limit", type=int, default=7)
            sub.add_argument("--resume-from", type=Path, required=True)
            sub.add_argument("--max-seconds", type=float, default=0.0)
            sub.add_argument("--min-free-bytes", type=int, default=1 << 30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        adapter, loader_binding = load_adapter(args.receiver_adapter, args.adapter_arg)
        if args.command == "validate":
            result = validate_adapter(adapter, program_kind=args.program_kind)
        else:
            result = run_solver(
                adapter,
                program_kind=args.program_kind,
                state_dir=args.resume_from,
                config=SolveConfig(
                    relinearizations=args.relinearizations,
                    damping=args.damping,
                    coefficient_limit=args.coefficient_limit,
                ),
                min_free_bytes=args.min_free_bytes,
                max_seconds=args.max_seconds,
                loader_binding=loader_binding,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("complete", True) else 75
    except QA43Error as exc:
        print(f"QA43 REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
