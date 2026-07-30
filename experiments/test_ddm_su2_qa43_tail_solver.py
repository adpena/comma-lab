# SPDX-License-Identifier: MIT
"""Scorer-free closure tests for the QA43 tail-solver harness."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from experiments.ddm_r7_token_coder import (
    decode_token_codes,
    encode_token_codes,
    frame_accounting,
)
from experiments.ddm_su2_qa43_tail_solver import (
    PAIR_COUNT,
    PAIR_MAP_BYTES,
    QA43Error,
    SolveConfig,
    V4DWarpTailAdapter,
    _coefficients,
    _is_forbidden_state_path,
    run_solver,
    validate_adapter,
)

_MEMBERS = ("state.bin", "pair.map", "coeff.r7", "manifest.json")
_V4D_PARENT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/"
    "v4d_composed_refine_celldrop50_archive.zip"
)
_V4D_DEPS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/"
    "d1/eval_root/submissions/pfs1"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip(members: Mapping[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", allowZip64=False) as handle:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            handle.writestr(info, payload)
    return stream.getvalue()


class SyntheticArchiveAdapter:
    """Tiny valid-ZIP adapter exercising real R7 frames and exact parseback."""

    def __init__(
        self,
        *,
        corrupt_decode: bool = False,
        zero_targets: bool = False,
    ) -> None:
        self._state = b"synthetic-parent-state-v2" + bytes(range(256))
        self._parent_archive = _zip({"state.bin": self._state})
        self._corrupt_decode = corrupt_decode
        self._targets = np.stack(
            [
                np.asarray(
                    [7 - pair // 100, 6 - pair // 100, 0, 0, 0, 0],
                    dtype=np.float64,
                )
                for pair in range(PAIR_COUNT)
            ]
        )
        if zero_targets:
            self._targets.fill(0.0)
        self._cache: dict[str, tuple[str, bytes, np.ndarray]] = {}

    def custody(self) -> Mapping[str, object]:
        source = Path(__file__).read_bytes()
        r7_source = (
            Path(__file__).parent / "ddm_r7_token_coder.py"
        ).read_bytes()
        return {
            "schema": "ddm_qa43_archive_adapter.v1",
            "program_kind": "terminal-frame0",
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
            "adapter_sha256": _sha256(source),
            "builder_sha256": _sha256(source),
            "receiver_sha256": _sha256(b"synthetic-receiver-v2"),
            "r7_source_sha256": _sha256(r7_source),
            "scorer_sha256": _sha256(b"synthetic-frozen-pose6-v2"),
            "targets_sha256": _sha256(self._targets.tobytes()),
        }

    def parent_archive(self) -> bytes:
        return self._parent_archive

    def coefficient_rank(self) -> int:
        return 2

    def parent_pair(self, pair_index: int) -> np.ndarray:
        self._check_pair(pair_index)
        pair = np.full((2, 1, 2, 3), 128, dtype=np.uint8)
        pair[1, :, :, :] = np.uint8(32 + pair_index % 191)
        return pair

    def initial_coefficients(self, pair_index: int) -> np.ndarray:
        self._check_pair(pair_index)
        return np.zeros(2, dtype=np.int16)

    def target_pose6(self, pair_index: int) -> np.ndarray:
        self._check_pair(pair_index)
        return self._targets[pair_index].copy()

    def realize_pair(
        self,
        pair_index: int,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        pair = self.parent_pair(pair_index)
        values = _coefficients(
            coefficients,
            rank=2,
            limit=7,
            name="synthetic coefficients",
        )
        pair[0].reshape(-1)[:2] = (128 + values).astype(np.uint8)
        return pair

    def build_archive(
        self,
        updates: Mapping[int, np.ndarray],
        *,
        codec: str,
        pair_map: bytes,
    ) -> bytes:
        active = self._active_pairs(pair_map)
        if active != sorted(updates):
            raise QA43Error("synthetic pair map and update keys differ")
        if active:
            codes = np.stack(
                [
                    _coefficients(
                        updates[pair],
                        rank=2,
                        limit=7,
                        name="synthetic update",
                    )
                    + 8
                    for pair in active
                ]
            ).astype(np.uint8)
            frame = encode_token_codes(
                codes.reshape(len(active), 1, 2, 1),
                levels=16,
                codec=codec,
            )
        else:
            frame = b""
        manifest = json.dumps(
            {
                "codec": codec,
                "frame_sha256": _sha256(frame),
                "map_sha256": _sha256(pair_map),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return _zip(
            {
                "state.bin": self._state,
                "pair.map": pair_map,
                "coeff.r7": frame,
                "manifest.json": manifest,
            }
        )

    def decode_pair(self, archive: bytes, pair_index: int) -> np.ndarray:
        if archive == self._parent_archive:
            return self.parent_pair(pair_index)
        _, pair_map, codes = self._parse_archive(archive)
        active = self._active_pairs(pair_map)
        if pair_index not in active:
            return self.parent_pair(pair_index)
        row = active.index(pair_index)
        coefficients = codes[row, 0, :, 0].astype(np.int16) - 8
        pair = self.realize_pair(pair_index, coefficients)
        if self._corrupt_decode:
            pair[0, 0, 0, 2] ^= np.uint8(1)
        return pair

    def public_decode_pair(self, archive: bytes, pair_index: int) -> np.ndarray:
        return self.decode_pair(archive, pair_index)

    def score_pose6(self, pair_index: int, pair_u8: np.ndarray) -> np.ndarray:
        self._check_pair(pair_index)
        pair = np.asarray(pair_u8)
        if pair.dtype != np.uint8 or pair.shape != (2, 1, 2, 3):
            raise QA43Error("synthetic scorer input differs")
        return pair[0].reshape(-1).astype(np.float64) - 128.0

    def accounting(self, archive: bytes) -> Mapping[str, object]:
        codec, pair_map, codes = self._parse_archive(archive)
        del codes
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            members = {info.filename: handle.read(info.filename) for info in handle.infolist()}
        return {
            "codec": codec,
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "pair_map_raw_bytes": PAIR_MAP_BYTES,
            "pair_map_sha256": _sha256(pair_map),
            "tail_member_bytes": len(pair_map) + len(members["coeff.r7"]),
            "r7_canonical_roundtrip": True,
            "zip_parseback": True,
            "all_bytes_consumed": True,
        }

    def extract_pair_map(self, archive: bytes) -> bytes:
        return self._parse_archive(archive)[1]

    @staticmethod
    def _check_pair(pair_index: int) -> None:
        if not 0 <= pair_index < PAIR_COUNT:
            raise QA43Error("synthetic pair index outside n600")

    @staticmethod
    def _active_pairs(pair_map: bytes) -> list[int]:
        if len(pair_map) != PAIR_MAP_BYTES:
            raise QA43Error("synthetic pair map is not exactly 75 bytes")
        bits = np.unpackbits(
            np.frombuffer(pair_map, dtype=np.uint8),
            bitorder="big",
        )
        return [int(pair) for pair in np.flatnonzero(bits)]

    def _parse_archive(
        self,
        archive: bytes,
    ) -> tuple[str, bytes, np.ndarray]:
        digest = _sha256(archive)
        if digest in self._cache:
            return self._cache[digest]
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as handle:
                infos = handle.infolist()
                if tuple(info.filename for info in infos) != _MEMBERS:
                    raise QA43Error("synthetic ZIP member order differs")
                members = {
                    info.filename: handle.read(info.filename)
                    for info in infos
                }
        except zipfile.BadZipFile as exc:
            raise QA43Error("synthetic outer archive is not a ZIP") from exc
        if members["state.bin"] != self._state:
            raise QA43Error("synthetic frozen state differs")
        manifest = json.loads(members["manifest.json"])
        codec = manifest["codec"]
        pair_map = members["pair.map"]
        frame = members["coeff.r7"]
        active = self._active_pairs(pair_map)
        if (
            manifest["frame_sha256"] != _sha256(frame)
            or manifest["map_sha256"] != _sha256(pair_map)
        ):
            raise QA43Error("synthetic manifest digest differs")
        if active:
            accounting = frame_accounting(frame)
            if accounting.codec != codec:
                raise QA43Error("synthetic outer and R7 codecs differ")
            codes = decode_token_codes(frame)
            if codes.shape != (len(active), 1, 2, 1):
                raise QA43Error("synthetic R7 coefficient lattice differs")
            expected_updates = codes[:, 0, :, 0].astype(np.int16) - 8
            rebuilt = self.build_archive(
                {
                    pair: expected_updates[row]
                    for row, pair in enumerate(active)
                },
                codec=codec,
                pair_map=pair_map,
            )
            if rebuilt != archive:
                raise QA43Error("synthetic archive is not canonical")
        else:
            if frame:
                raise QA43Error("synthetic empty map carries R7 bytes")
            codes = np.empty((0, 1, 2, 1), dtype=np.uint8)
        value = (codec, pair_map, codes)
        self._cache[digest] = value
        return value


def test_validate_adapter_closes_both_r7_codec_arms() -> None:
    receipt = validate_adapter(
        SyntheticArchiveAdapter(),
        program_kind="terminal-frame0",
    )

    assert receipt["status"] == "PASS"
    assert receipt["pair_map_bytes"] == 75
    assert receipt["nonzero_frame0_change_pairs"] == [0, 599]
    assert receipt["public_entrypoint_closed"] is True
    assert set(receipt["codecs"]) == {"smevr", "brotli11"}
    assert all(
        candidate["accounting"]["all_bytes_consumed"]
        for candidate in receipt["codecs"].values()
    )


def test_solver_is_resumable_and_preserves_nested_stage_archives(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path.resolve() / "qa43-state"
    adapter = SyntheticArchiveAdapter()
    kwargs = {
        "program_kind": "terminal-frame0",
        "state_dir": state_dir,
        "config": SolveConfig(relinearizations=2),
        "min_free_bytes": 0,
        "stage_ks": (2, 4),
    }

    first = run_solver(adapter, **kwargs)
    pair_rows = {
        path.name: path.read_bytes()
        for path in (state_dir / "pair_solves").glob("*.json")
    }
    second = run_solver(adapter, **kwargs)

    assert first == second
    assert first["complete"] is True
    assert first["pairs_solved"] == 4
    assert first["stage_ks_complete"] == [2, 4]
    assert {
        path.name: path.read_bytes()
        for path in (state_dir / "pair_solves").glob("*.json")
    } == pair_rows
    for k in (2, 4):
        assert (state_dir / f"candidate_k{k:03d}.archive.zip").is_file()
        receipt = json.loads((state_dir / f"stage_k{k:03d}.json").read_text())
        assert receipt["active_pair_map_bytes"] == 75
        assert receipt["all_n600_frame1_frozen"] is True
        assert len(receipt["decoded_pair_sha256"]) == PAIR_COUNT
        assert (
            receipt["falsifiers"]["tail_price_gt_600B_per_admitted_pair"]
            is False
        )
        assert receipt["whole_action_delta_bytes_per_admitted_pair"] <= 600
        assert receipt["d_pose_mean_candidate"] < receipt["d_pose_mean_parent"]


def test_stage_hard_falsifier_stops_before_next_nested_stage(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path.resolve() / "qa43-no-win"
    summary = run_solver(
        SyntheticArchiveAdapter(zero_targets=True),
        program_kind="terminal-frame0",
        state_dir=state_dir,
        config=SolveConfig(relinearizations=2),
        min_free_bytes=0,
        stage_ks=(2, 4),
    )

    assert summary["complete"] is False
    assert summary["verdict"] == "HARD_FALSIFIER_REFIT_REQUIRED"
    assert summary["halted_by_falsifier"]["stage_k"] == 2
    assert summary["stage_ks_complete"] == [2]
    assert not (state_dir / "stage_k004.json").exists()


def test_receiver_identity_mismatch_fails_closed() -> None:
    with pytest.raises(QA43Error, match="decoded pair differs"):
        validate_adapter(
            SyntheticArchiveAdapter(corrupt_decode=True),
            program_kind="terminal-frame0",
        )


@pytest.mark.parametrize("value", [65535, 65536, 2**64 - 1])
def test_unsigned_integer_wrap_is_refused(value: int) -> None:
    with pytest.raises(QA43Error, match="exceeds"):
        _coefficients(
            np.asarray([value, 0], dtype=np.uint64),
            rank=2,
            limit=7,
            name="overflow",
        )


def test_relative_resume_path_is_refused() -> None:
    with pytest.raises(QA43Error, match="absolute path"):
        run_solver(
            SyntheticArchiveAdapter(),
            program_kind="terminal-frame0",
            state_dir=Path("relative-state"),
            config=SolveConfig(relinearizations=2),
            min_free_bytes=0,
            stage_ks=(2,),
        )


@pytest.mark.parametrize(
    "path",
    [
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/private/tmp"),
        Path("/private/var/tmp"),
    ],
)
def test_exact_tmp_class_roots_are_refused_without_writing(path: Path) -> None:
    assert _is_forbidden_state_path(path)


@pytest.mark.skipif(
    not (_V4D_PARENT.is_file() and _V4D_DEPS.is_dir()),
    reason="custodied v4d archive/receiver dependencies are not mounted",
)
def test_real_v4d_warp_adapter_closes_both_codecs_without_scorer() -> None:
    adapter = V4DWarpTailAdapter(
        {
            "parent_archive": str(_V4D_PARENT),
            "receiver_deps_dir": str(_V4D_DEPS),
        }
    )

    receipt = validate_adapter(adapter, program_kind="warp-tail")

    assert receipt["status"] == "PASS"
    assert receipt["parent_archive_sha256"] == (
        "f1f3288062468e97c090ffe88ac81a6d6f76925743bd83aecb15307c0314a220"
    )
    assert set(receipt["codecs"]) == {"smevr", "brotli11"}
    assert receipt["nonzero_frame0_change_pairs"] == [0, 599]
    assert receipt["public_entrypoint_closed"] is True
    with pytest.raises(QA43Error, match="pair index outside"):
        adapter.parent_pair("0")  # type: ignore[arg-type]
    tail_module = importlib.import_module("inflate_runner_v4d_qa43_tail")
    with pytest.raises(
        tail_module.QA43TailReceiverError,
        match="signed int4 lattice",
    ):
        tail_module.encode_tail(
            np.zeros((1, 6), dtype=np.uint8),
            bytes([128]) + bytes(74),
            codec="smevr",
        )
