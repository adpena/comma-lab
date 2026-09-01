#!/usr/bin/env python3
"""RXC1 restartable exact-coder gate on the retained AFR1 stream.

This module does not implement a coder.  It imports the byte-identity-proven JG2
mirror and supplies the missing orchestration surface: immutable pair-boundary
state, exact suffix replay, retained one-cell edits, and a crash-resumable n>=32
full-vs-incremental screen.  Every reported delta is a physical RC64 stream and
archive stat; no entropy or differentiable surrogate is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jg2_tail_reencode as jg2

STORE = Path("/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder")
RUNTIME_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/"
    "runtime_candidate_native"
)
ARCHIVE = RUNTIME_ROOT / "archive.zip"
TOKENS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/identity_v1/"
    "out/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
JC1_MEMO = REPO / ".omx/research/ddm_jc1_afr_rc64_joint_redesign_20260901.md"
CM1_MEMO = REPO / ".omx/research/ddm_cm1_coder_matched_surrogate_20260826.md"

AFR1_ARCHIVE_BYTES = 180_002
AFR1_ARCHIVE_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
AFR1_STREAM_BYTES = 113_411
AFR1_STREAM_SHA256 = "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
AFR1_HPAC_BYTES = 13_515
AFR1_HPAC_SHA256 = "602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98"
TOKENS_BYTES = 117_964_800
TOKENS_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
JC1_MEMO_SHA256 = "fb035f4db92c78fba3357285b707995f5d0265b2ed4a38187c951a0b5fcbe05a"
CM1_MEMO_SHA256 = "2cf78a6b477074cbca1f55e5fa8b13a2f02c7c881a6a360db0a4d5de2061619b"
ROUTE_B_SHA256 = "c2d9759a77e793d643ca1d4a557934cdb66f39473b244f382dd9f0b8faaf89e5"

RUNTIME_PINS = {
    "runtime/residual_archive.py": "aca361f3e94941f4f2800bacec79f5032335588e317e76ee1a306bbb5ba64530",
    "runtime/fx2_model_axis_corrector.py": "6462ba51ddf29dbb60b091e22043d591a1d081d9583a4864348f2cb1525aa064",
    "runtime/free_corrector.py": "dd337159bd84e96e767cbde9a6dffecc909e824c2f092399e09095bebaf094a5",
    "cpr1/inflate.py": "ff446edd9237148bdc898be2f8f8c4782bf231a50cf3830c4b0b21a4474a736b",
}

PAIR_COUNT = 600
PLANE = 384 * 512
SCREEN_N = 32
SCREEN_SEED = 20_260_901
STRIDES = (200, 300)
CHECKPOINT_FRAMES = frozenset(
    {0, PAIR_COUNT}
    | {frame for stride in STRIDES for frame in range(stride, PAIR_COUNT, stride)}
)
MINIMUM_FREE_BYTES = 3_200_000_000
RESERVE_BYTES = 1 << 30
FULL_REFERENCE_SECONDS = 897.675
AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"


class Rxc1Error(RuntimeError):
    """A custody, state-identity, or correlation gate refused."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise Rxc1Error(f"missing pinned input: {path}")
    fact = file_fact(path)
    if fact["bytes"] != expected_bytes or fact["sha256"] != expected_sha256:
        raise Rxc1Error(
            f"pin mismatch for {path}: {fact['bytes']}/{fact['sha256']} != "
            f"{expected_bytes}/{expected_sha256}"
        )
    return fact


def require_sha(path: Path, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise Rxc1Error(f"missing pinned source: {path}")
    fact = file_fact(path)
    if fact["sha256"] != expected_sha256:
        raise Rxc1Error(
            f"source pin mismatch for {path}: {fact['sha256']} != {expected_sha256}"
        )
    return fact


def read_sections() -> tuple[dict[str, bytes], bytes]:
    member = jg2.read_archive_member(ARCHIVE)
    sections = jg2.split_member(member)
    tail = sections["tail"]
    stream = tail[jg2.RESIDUAL_COMPACT_BYTES :]
    if len(stream) != AFR1_STREAM_BYTES or sha256_bytes(stream) != AFR1_STREAM_SHA256:
        raise Rxc1Error("AFR1 archive no longer contains the pinned RC64 stream")
    if len(sections["hpac"]) != AFR1_HPAC_BYTES:
        raise Rxc1Error("AFR1 HPAC section length drifted")
    if sha256_bytes(sections["hpac"]) != AFR1_HPAC_SHA256:
        raise Rxc1Error("AFR1 HPAC section hash drifted")
    return sections, stream


def validate_fact(fact: dict[str, object]) -> None:
    path = Path(str(fact["path"]))
    if not path.is_file():
        raise Rxc1Error(f"retained artifact disappeared: {path}")
    observed = file_fact(path)
    if observed["bytes"] != fact["bytes"] or observed["sha256"] != fact["sha256"]:
        raise Rxc1Error(f"retained artifact changed: {path}")


def validate_run_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    for key in ("stream", "archive", "bits_per_frame_ledger"):
        validate_fact(payload[key])
    terminal = payload.get("terminal_checkpoint")
    if terminal:
        validate_fact(terminal["checkpoint"])
        validate_fact(terminal["encoder_state"])
    if payload.get("edit", {}).get("edits_file"):
        validate_fact(payload["edit"]["edits_file"])
    if payload.get("base_stream", {}).get("sha256") != AFR1_STREAM_SHA256:
        raise Rxc1Error(f"run receipt is not bound to the AFR1 stream: {path}")
    if payload.get("base_archive", {}).get("sha256") != AFR1_ARCHIVE_SHA256:
        raise Rxc1Error(f"run receipt is not bound to the AFR1 archive: {path}")
    if payload.get("jg2_source_sha256") != sha256_file(Path(jg2.__file__)):
        raise Rxc1Error(f"run receipt was produced by a different JG2 source: {path}")
    return payload


def validate_preflight_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    inputs = payload.get("inputs", {})
    for key in ("archive", "tokens", "jc1_memo", "cm1_memo", "rxc1_source", "jg2_source", "route_b_source"):
        validate_fact(inputs[key])
    if inputs["archive"]["sha256"] != AFR1_ARCHIVE_SHA256:
        raise Rxc1Error("preflight archive is not AFR1")
    if inputs["tokens"]["sha256"] != TOKENS_SHA256:
        raise Rxc1Error("preflight token field changed")
    if inputs["rxc1_source"]["sha256"] != sha256_file(Path(__file__)):
        raise Rxc1Error("RXC1 source changed after preflight; rerun preflight")
    if inputs["jg2_source"]["sha256"] != sha256_file(Path(jg2.__file__)):
        raise Rxc1Error("JG2 source changed after preflight; rerun preflight")
    for name, fact in inputs["runtime"].items():
        validate_fact(fact)
        if fact["sha256"] != RUNTIME_PINS[name]:
            raise Rxc1Error(f"preflight runtime source changed: {name}")
    return payload


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, matching Spearman's standard definition."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2.0 + 1.0
        begin = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) != len(right) or len(left) < 2:
        raise Rxc1Error("correlation needs equal arrays with at least two rows")
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        raise Rxc1Error("correlation is undefined because a delta column is constant")
    return float(np.corrcoef(left, right)[0, 1])


def compare_bytes(left: Path, right: Path) -> dict[str, object]:
    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    overlap = min(len(left_bytes), len(right_bytes))
    differing = sum(a != b for a, b in zip(left_bytes, right_bytes, strict=False))
    differing += abs(len(left_bytes) - len(right_bytes))
    return {
        "left": file_fact(left),
        "right": file_fact(right),
        "compared_bytes": max(len(left_bytes), len(right_bytes)),
        "overlap_bytes": overlap,
        "differing_bytes": differing,
        "byte_identical": differing == 0,
    }


def sample_pairs(seed: int = SCREEN_SEED) -> list[dict[str, int]]:
    """Four seeded random pairs from each of eight equal 75-pair strata."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int]] = []
    for stratum in range(8):
        start = stratum * 75
        for pair in sorted(int(value) for value in rng.choice(np.arange(start, start + 75), 4, replace=False)):
            rows.append({"stratum": stratum, "pair": pair})
    if len(rows) != SCREEN_N or len({row["pair"] for row in rows}) != SCREEN_N:
        raise Rxc1Error("the preregistered sampler did not produce 32 unique pairs")
    return rows


def create_edit_payloads(tokens: np.ndarray, root: Path) -> list[dict[str, object]]:
    """Persist one deterministic one-cell replacement plane for every sampled pair."""
    rng = np.random.default_rng(SCREEN_SEED ^ 0x52584331)
    rows: list[dict[str, object]] = []
    for sample in sample_pairs():
        pair = sample["pair"]
        flat = int(rng.integers(0, PLANE))
        plane = np.asarray(tokens[pair], dtype=np.uint8).copy()
        old = int(plane.reshape(-1)[flat])
        new = int((old + 1 + int(rng.integers(0, 4))) % 5)
        if new == old:
            raise Rxc1Error("synthetic edit failed to change its selected token")
        plane.reshape(-1)[flat] = new
        path = root / f"pair_{pair:04d}.one_cell.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as prior:
                if prior.files != [str(pair)] or not np.array_equal(prior[str(pair)], plane):
                    raise Rxc1Error(f"existing edit payload does not match preregistration: {path}")
        else:
            atomic_npz(path, {str(pair): plane})
        rows.append(
            {
                **sample,
                "flat_position": flat,
                "row": flat // 512,
                "column": flat % 512,
                "old_token": old,
                "new_token": new,
                "changed_tokens": 1,
                "payload": file_fact(path),
            }
        )
    return rows


def preflight() -> dict[str, object]:
    STORE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STORE).free
    if free < MINIMUM_FREE_BYTES:
        raise Rxc1Error(
            f"storage preflight failed: {free} B free < {MINIMUM_FREE_BYTES} B; "
            "do not launch the retained n32 exact-coder screen"
        )
    archive = require_file(ARCHIVE, AFR1_ARCHIVE_BYTES, AFR1_ARCHIVE_SHA256)
    tokens = require_file(TOKENS, TOKENS_BYTES, TOKENS_SHA256)
    jc1 = require_sha(JC1_MEMO, JC1_MEMO_SHA256)
    cm1 = require_sha(CM1_MEMO, CM1_MEMO_SHA256)
    route = require_sha(jg2.ROUTE_B, ROUTE_B_SHA256)
    runtime = {name: require_sha(RUNTIME_ROOT / name, sha) for name, sha in RUNTIME_PINS.items()}
    sections, stream = read_sections()
    source_root = STORE / "retained/source"
    source_root.mkdir(parents=True, exist_ok=True)
    retained_sources = []
    for source in (Path(__file__), Path(jg2.__file__), jg2.ROUTE_B):
        source_sha = sha256_file(source)
        destination = source_root / f"{source.stem}.{source_sha[:16]}{source.suffix}"
        if not destination.is_file():
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        if sha256_file(destination) != source_sha:
            raise Rxc1Error(f"retained source copy changed: {destination}")
        retained_sources.append(file_fact(destination))
    payload = {
        "schema": "ddm_rxc1_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "storage": {
            "root": str(STORE),
            "free_bytes": free,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "post_run_reserve_bytes": RESERVE_BYTES,
            "status": "PASS",
        },
        "inputs": {
            "archive": archive,
            "tokens": tokens,
            "jc1_memo": jc1,
            "cm1_memo": cm1,
            "rxc1_source": file_fact(Path(__file__)),
            "jg2_source": file_fact(Path(jg2.__file__)),
            "route_b_source": route,
            "runtime": runtime,
            "rc64_base_sha256": jg2.RC64_BASE_SHA,
        },
        "archive_sections": {
            "hpac": {
                "bytes": len(sections["hpac"]),
                "sha256": sha256_bytes(sections["hpac"]),
            },
            "token_stream": {"bytes": len(stream), "sha256": sha256_bytes(stream)},
        },
        "screen": {
            "seed": SCREEN_SEED,
            "n": SCREEN_N,
            "strata": "8 contiguous index strata of 75 pairs; 4 random without replacement per stratum",
            "edit": "one deterministic changed token in one retained replacement plane per pair",
            "strides": list(STRIDES),
            "checkpoint_frames": sorted(CHECKPOINT_FRAMES),
            "gate": {"pearson_min": 0.9, "spearman_min": 0.9},
        },
        "retained_executed_sources": retained_sources,
    }
    atomic_json(STORE / "PREFLIGHT.json", payload)
    return payload


class RestartableExactCoder:
    """Outer-loop API: retained edit -> physical exact delta from a boundary state."""

    def __init__(self, store: Path = STORE) -> None:
        self.store = store
        preflight_path = store / "PREFLIGHT.json"
        if not preflight_path.is_file():
            raise Rxc1Error("run preflight before constructing the exact-coder API")
        validate_preflight_receipt(preflight_path)
        self.route_b = jg2.load_route_b()
        build_root = store / "retained/build"
        build_receipt = build_root / "BUILD.json"
        if build_receipt.is_file():
            self.build = json.loads(build_receipt.read_text())
            for key in ("base_source", "generated", "library"):
                validate_fact(self.build[key])
            self.library = Path(str(self.build["library"]["path"]))
        else:
            self.library, self.build = jg2.compile_rc64(
                build_root, self.route_b, "rxc1"
            )
            atomic_json(build_receipt, self.build)
        self.residual, self.renderer, self.renderer_dir = jg2.load_runtime(RUNTIME_ROOT)
        self.parts = self.residual.read_residual_archive(ARCHIVE)
        self.sections, self.shipped_stream = read_sections()
        self.tokens = jg2.load_tokens(TOKENS)

    def checkpoint(self, frame: int) -> tuple[Path, Path]:
        paths = jg2.checkpoint_bundle_paths(
            self.store / "retained/baseline/checkpoints", frame
        )
        if not all(path.is_file() for path in paths):
            raise Rxc1Error(f"baseline checkpoint frame {frame} is missing")
        return paths

    @staticmethod
    def nearest_checkpoint(pair: int, stride: int) -> int:
        if pair < 0 or pair >= PAIR_COUNT:
            raise Rxc1Error(f"pair must be in 0..{PAIR_COUNT - 1}")
        if stride <= 0:
            raise Rxc1Error("stride must be positive")
        return (pair // stride) * stride

    def _pack_archive(self, emitted: bytes, destination: Path) -> dict[str, object]:
        sections = dict(self.sections)
        sections["tail"] = (
            self.sections["tail"][: jg2.RESIDUAL_COMPACT_BYTES] + emitted
        )
        member = jg2.join_member(sections)
        if destination.is_file():
            if jg2.read_archive_member(destination) != member:
                raise Rxc1Error(f"immutable exact archive changed on retry: {destination}")
        else:
            jg2.pack_archive(member, destination)
        return file_fact(destination)

    def run(
        self,
        *,
        edit_path: Path | None,
        run_dir: Path,
        resume_frame: int | None,
        immutable_frames: set[int] | None = None,
    ) -> dict[str, Any]:
        """Run or resume one physical encode and retain stream, state, archive, receipt."""
        receipt_path = run_dir / "RESULT.json"
        if receipt_path.is_file():
            return validate_run_receipt(receipt_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        field, edit_report = jg2.load_edit_overlay(self.tokens, edit_path)
        edited_pairs = [int(value) for value in edit_report["edited_pairs"]]
        if resume_frame is not None and any(pair < resume_frame for pair in edited_pairs):
            raise Rxc1Error(
                f"restart frame {resume_frame} is after an edited pair {edited_pairs}; "
                "that would silently omit the edit from the coder state"
            )
        run_spec = {
            "schema": "ddm_rxc1_run_spec.v1",
            "archive_sha256": AFR1_ARCHIVE_SHA256,
            "token_stream_sha256": AFR1_STREAM_SHA256,
            "tokens_sha256": TOKENS_SHA256,
            "jg2_source_sha256": sha256_file(Path(jg2.__file__)),
            "route_b_source_sha256": ROUTE_B_SHA256,
            "edit": edit_report,
            "resume_frame": resume_frame,
        }
        run_spec_path = run_dir / "RUN_SPEC.json"
        if run_spec_path.is_file():
            if json.loads(run_spec_path.read_text()) != run_spec:
                raise Rxc1Error(
                    f"unfinished run state is bound to a different run spec: {run_spec_path}"
                )
        else:
            atomic_json(run_spec_path, run_spec)
        resume_checkpoint = None
        resume_encoder = None
        actual_resume_frame = 0 if resume_frame is None else resume_frame
        history_root = run_dir / "checkpoints"
        retained_frames = sorted(
            int(path.stem.removeprefix("frame_"))
            for path in history_root.glob("frame_*.npz")
            if path.stem.removeprefix("frame_").isdigit()
            and path.with_suffix(".encoder.bin").is_file()
            and int(path.stem.removeprefix("frame_")) < PAIR_COUNT
        )
        if retained_frames:
            actual_resume_frame = retained_frames[-1]
            resume_checkpoint, resume_encoder = jg2.checkpoint_bundle_paths(
                history_root, actual_resume_frame
            )
        elif resume_frame is not None:
            resume_checkpoint, resume_encoder = self.checkpoint(resume_frame)
        periodic_frames = {
            frame
            for frame in range(300, PAIR_COUNT, 300)
            if frame > actual_resume_frame
        }
        if immutable_frames is not None:
            periodic_frames.update(immutable_frames)
        wall_started = time.perf_counter()
        result = jg2.encode_tail(
            residual=self.residual,
            renderer=self.renderer,
            renderer_dir=self.renderer_dir,
            parts=self.parts,
            target=field,
            library=self.library,
            route_b=self.route_b,
            work=run_dir,
            tag="exact",
            frames=PAIR_COUNT,
            checkpoint_every=0,
            resume=False,
            resume_checkpoint=resume_checkpoint,
            resume_encoder_state=resume_encoder,
            checkpoint_history=(history_root if periodic_frames else None),
            checkpoint_frames=periodic_frames,
            retain_terminal_checkpoint=True,
        )
        wall_seconds = time.perf_counter() - wall_started
        stream_path = Path(str(result["stream"]["path"]))
        emitted = stream_path.read_bytes()
        archive = self._pack_archive(emitted, run_dir / "archive.zip")
        payload = {
            "schema": "ddm_rxc1_exact_run.v1",
            "axis": AXIS,
            "score_claim": False,
            "mode": "full_from_zero" if resume_frame is None else "incremental_from_exact_state",
            "resume_frame": resume_frame,
            "actual_resume_frame": actual_resume_frame,
            "edit": edit_report,
            "stream": result["stream"],
            "archive": archive,
            "bits_per_frame_ledger": result["bits_per_frame_ledger"],
            "terminal_checkpoint": result["terminal_checkpoint"],
            "immutable_checkpoints": result["immutable_checkpoints"],
            "resumed_from": result["resumed_from"],
            "stream_delta_bytes": len(emitted) - len(self.shipped_stream),
            "archive_delta_bytes": int(archive["bytes"]) - AFR1_ARCHIVE_BYTES,
            "encode_seconds": result["elapsed_seconds"],
            "wall_seconds": wall_seconds,
            "start_frame": result["start_frame"],
            "frames_encoded": PAIR_COUNT - int(result["start_frame"]),
            "build": self.build,
            "jg2_source_sha256": sha256_file(Path(jg2.__file__)),
            "base_stream": {"bytes": len(self.shipped_stream), "sha256": AFR1_STREAM_SHA256},
            "base_archive": {"bytes": AFR1_ARCHIVE_BYTES, "sha256": AFR1_ARCHIVE_SHA256},
        }
        if payload["stream_delta_bytes"] != payload["archive_delta_bytes"]:
            raise Rxc1Error("stored ZIP archive delta differs from physical stream delta")
        atomic_json(receipt_path, payload)
        return payload

    def exact_delta(self, edit_path: Path, pair: int, stride: int, run_dir: Path) -> dict[str, Any]:
        """Return a real exact archive delta by replaying from the nearest full state."""
        with np.load(edit_path, allow_pickle=False) as blob:
            edit_pairs = sorted(int(key) for key in blob.files)
        if edit_pairs != [pair]:
            raise Rxc1Error(
                f"exact_delta pair {pair} does not match retained edit pairs {edit_pairs}"
            )
        frame = self.nearest_checkpoint(pair, stride)
        return self.run(
            edit_path=edit_path,
            run_dir=run_dir,
            resume_frame=frame,
        )


def stage_baseline(api: RestartableExactCoder) -> dict[str, object]:
    receipt_path = STORE / "BASELINE.json"
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text())
        validate_run_receipt(STORE / "retained/baseline/RESULT.json")
        validate_fact(payload["stream"])
        validate_fact(payload["archive"])
        for checkpoint in payload["checkpoints"]:
            validate_fact(checkpoint["checkpoint"])
            validate_fact(checkpoint["encoder_state"])
        if not payload.get("byte_identical"):
            raise Rxc1Error("retained baseline receipt records a failed identity gate")
        return payload
    result = api.run(
        edit_path=None,
        run_dir=STORE / "retained/baseline",
        resume_frame=None,
        immutable_frames=set(CHECKPOINT_FRAMES),
    )
    stream_comparison = compare_bytes(Path(str(result["stream"]["path"])), _retain_base_stream(api))
    checkpoints = []
    for frame in sorted(CHECKPOINT_FRAMES):
        checkpoint, encoder = api.checkpoint(frame)
        checkpoints.append(
            {
                "frame": frame,
                "checkpoint": file_fact(checkpoint),
                "encoder_state": file_fact(encoder),
            }
        )
    observed_frames = {int(row["frame"]) for row in checkpoints}
    if observed_frames != CHECKPOINT_FRAMES:
        raise Rxc1Error(
            f"baseline checkpoint denominator changed: {sorted(observed_frames)} != "
            f"{sorted(CHECKPOINT_FRAMES)}"
        )
    payload = {
        "schema": "ddm_rxc1_baseline.v1",
        "axis": AXIS,
        "score_claim": False,
        "stream": result["stream"],
        "archive": result["archive"],
        "stream_comparison": stream_comparison,
        "checkpoint_count": len(checkpoints),
        "checkpoint_frames": sorted(observed_frames),
        "checkpoints": checkpoints,
        "wall_seconds": result["wall_seconds"],
        "frames": PAIR_COUNT,
        "tokens": TOKENS_BYTES,
        "byte_identical": bool(stream_comparison["byte_identical"]),
        "status": "PASS" if stream_comparison["byte_identical"] else "FAILED_BYTE_IDENTITY",
    }
    atomic_json(receipt_path, payload)
    if not stream_comparison["byte_identical"]:
        raise Rxc1Error("Stage 0 failed: full exact encode is not AFR1-byte-identical")
    return payload


def _retain_base_stream(api: RestartableExactCoder) -> Path:
    path = STORE / "retained/input/afr1_token_stream.rc64.bin"
    if not path.is_file():
        atomic_bytes(path, api.shipped_stream)
    if path.stat().st_size != AFR1_STREAM_BYTES or sha256_file(path) != AFR1_STREAM_SHA256:
        raise Rxc1Error("retained AFR1 stream pin failed")
    return path


def stage_null(api: RestartableExactCoder) -> dict[str, object]:
    receipt_path = STORE / "NULL_REPLAY.json"
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text())
        if not payload.get("byte_identical"):
            raise Rxc1Error("retained null replay receipt records a failed identity gate")
        return payload
    baseline = stage_baseline(api)
    base_stream = Path(str(baseline["stream"]["path"]))
    rows = []
    for stride in STRIDES:
        for frame in range(0, PAIR_COUNT, stride):
            if frame == 0:
                result = json.loads((STORE / "retained/baseline/RESULT.json").read_text())
            else:
                result = api.run(
                    edit_path=None,
                    run_dir=STORE / f"retained/null/stride_{stride:03d}/frame_{frame:04d}",
                    resume_frame=frame,
                )
            comparison = compare_bytes(Path(str(result["stream"]["path"])), base_stream)
            rows.append(
                {
                    "stride": stride,
                    "checkpoint_frame": frame,
                    "frames_replayed": PAIR_COUNT - frame,
                    "wall_seconds": result["wall_seconds"],
                    "comparison": comparison,
                    "terminal_checkpoint": result["terminal_checkpoint"],
                }
            )
    failures = [row for row in rows if not row["comparison"]["byte_identical"]]
    compared = sum(int(row["comparison"]["compared_bytes"]) for row in rows)
    payload = {
        "schema": "ddm_rxc1_null_replay.v1",
        "axis": AXIS,
        "score_claim": False,
        "rows": rows,
        "strides": list(STRIDES),
        "checkpoint_replays": len(rows),
        "stream_bytes_per_replay": AFR1_STREAM_BYTES,
        "compared_bytes": compared,
        "differing_bytes": sum(
            int(row["comparison"]["differing_bytes"]) for row in rows
        ),
        "byte_identical": not failures,
        "status": "PASS" if not failures else "FAILED_BYTE_IDENTITY",
        "denominator": {
            "checkpoint_starts": len(rows),
            "unique_checkpoint_frames": sorted({int(row["checkpoint_frame"]) for row in rows}),
            "full_streams": len(rows),
        },
    }
    atomic_json(receipt_path, payload)
    if failures:
        raise Rxc1Error(f"null replay failed at {len(failures)} checkpoint starts")
    return payload


def _screen_rows(api: RestartableExactCoder) -> list[dict[str, object]]:
    edits = create_edit_payloads(api.tokens, STORE / "retained/edits")
    rows: list[dict[str, object]] = []
    for index, edit in enumerate(edits):
        pair = int(edit["pair"])
        edit_path = Path(str(edit["payload"]["path"]))
        row_root = STORE / f"retained/screen/row_{index:02d}_pair_{pair:04d}"
        full = api.run(
            edit_path=edit_path,
            run_dir=row_root / "full",
            resume_frame=None,
        )
        incremental = {}
        comparisons = {}
        for stride in STRIDES:
            nearest = api.nearest_checkpoint(pair, stride)
            reused_full_exact = nearest == 0
            result = (
                full
                if reused_full_exact
                else api.exact_delta(
                    edit_path,
                    pair,
                    stride,
                    row_root / f"incremental_stride_{stride:03d}",
                )
            )
            comparison = compare_bytes(
                Path(str(result["stream"]["path"])), Path(str(full["stream"]["path"]))
            )
            incremental[str(stride)] = {
                **result,
                "nearest_checkpoint": nearest,
                "reused_full_exact": reused_full_exact,
            }
            comparisons[str(stride)] = comparison
        row = {
            "row_index": index,
            "sample": edit,
            "full_delta_bytes": full["archive_delta_bytes"],
            "full": full,
            "incremental": incremental,
            "stream_comparisons": comparisons,
        }
        atomic_json(row_root / "ROW.json", row)
        rows.append(row)
    return rows


def _stride_stats(rows: list[dict[str, object]], stride: int) -> dict[str, object]:
    full = np.array([float(row["full_delta_bytes"]) for row in rows])
    incremental = np.array(
        [float(row["incremental"][str(stride)]["archive_delta_bytes"]) for row in rows]
    )
    errors = incremental - full
    correlation_failure = None
    try:
        pearson = correlation(incremental, full)
        spearman = correlation(rankdata(incremental), rankdata(full))
    except Rxc1Error as error:
        pearson = None
        spearman = None
        correlation_failure = str(error)
    signs = np.sign(incremental) == np.sign(full)
    walls = np.array(
        [float(row["incremental"][str(stride)]["wall_seconds"]) for row in rows]
    )
    frames = np.array(
        [int(row["incremental"][str(stride)]["frames_encoded"]) for row in rows]
    )
    return {
        "stride": stride,
        "n": len(rows),
        "pearson": pearson,
        "spearman": spearman,
        "correlation_failure": correlation_failure,
        "max_abs_error_bytes": float(np.max(np.abs(errors))),
        "mean_abs_error_bytes": float(np.mean(np.abs(errors))),
        "sign_agreement_count": int(signs.sum()),
        "sign_agreement_fraction": float(signs.mean()),
        "stream_identity_count": sum(
            bool(row["stream_comparisons"][str(stride)]["byte_identical"]) for row in rows
        ),
        "wall_seconds_median": float(np.median(walls)),
        "wall_seconds_mean": float(np.mean(walls)),
        "frames_encoded_median": float(np.median(frames)),
        "speedup_vs_897p675_median": FULL_REFERENCE_SECONDS / float(np.median(walls)),
        "gate_pass": (
            pearson is not None
            and spearman is not None
            and pearson >= 0.9
            and spearman >= 0.9
        ),
    }


def stage_screen(api: RestartableExactCoder) -> dict[str, object]:
    receipt_path = STORE / "SCREEN.json"
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text())
        if not payload.get("gate_pass"):
            raise Rxc1Error("retained screen receipt records a failed correlation gate")
        return payload
    stage_null(api)
    rows = _screen_rows(api)
    full_values = np.array([float(row["full_delta_bytes"]) for row in rows])
    if len(rows) != SCREEN_N:
        raise Rxc1Error(f"screen denominator is {len(rows)}, expected {SCREEN_N}")
    full_walls = np.array([float(row["full"]["wall_seconds"]) for row in rows])
    stride_stats = [_stride_stats(rows, stride) for stride in STRIDES]
    all_streams_identical = all(
        row["stream_comparisons"][str(stride)]["byte_identical"]
        for row in rows
        for stride in STRIDES
    )
    correlation_gate_pass = all(bool(row["gate_pass"]) for row in stride_stats)
    gate_pass = correlation_gate_pass and all_streams_identical
    payload = {
        "schema": "ddm_rxc1_screen.v1",
        "axis": AXIS,
        "score_claim": False,
        "seed": SCREEN_SEED,
        "sampling": "8x75 pair-index strata; 4 seeded random without replacement per stratum",
        "n": len(rows),
        "edit": "one retained changed token per sampled pair",
        "full_delta_unique_values": sorted({int(value) for value in full_values}),
        "full_delta_min_bytes": int(np.min(full_values)),
        "full_delta_max_bytes": int(np.max(full_values)),
        "full_wall_seconds_median": float(np.median(full_walls)),
        "full_wall_seconds_mean": float(np.mean(full_walls)),
        "reference_full_seconds_cm1": FULL_REFERENCE_SECONDS,
        "strides": stride_stats,
        "all_incremental_streams_byte_identical_to_full": all_streams_identical,
        "correlation_gate_pass": correlation_gate_pass,
        "gate_pass": gate_pass,
        "status": (
            "PASS"
            if gate_pass and all_streams_identical
            else "FAILED_STREAM_IDENTITY"
            if not all_streams_identical
            else "FAILED_CORRELATION"
        ),
        "rows": [
            {
                "row_index": row["row_index"],
                "sample": row["sample"],
                "full_delta_bytes": row["full_delta_bytes"],
                "full_receipt": file_fact(
                    Path(str(row["full"]["stream"]["path"])).parent / "RESULT.json"
                ),
                "incremental": {
                    stride: {
                        "resume_frame": row["incremental"][stride]["resume_frame"],
                        "nearest_checkpoint": row["incremental"][stride][
                            "nearest_checkpoint"
                        ],
                        "reused_full_exact": row["incremental"][stride][
                            "reused_full_exact"
                        ],
                        "delta_bytes": row["incremental"][stride]["archive_delta_bytes"],
                        "wall_seconds": row["incremental"][stride]["wall_seconds"],
                        "receipt": file_fact(
                            Path(str(row["incremental"][stride]["stream"]["path"])).parent
                            / "RESULT.json"
                        ),
                    }
                    for stride in (str(value) for value in STRIDES)
                },
            }
            for row in rows
        ],
    }
    atomic_json(receipt_path, payload)
    if not all_streams_identical:
        raise Rxc1Error("an incremental stream differs from its full exact re-encode")
    if not gate_pass:
        raise Rxc1Error("preregistered Pearson/Spearman >=0.9 gate failed")
    return payload


def stage_manifest() -> dict[str, object]:
    manifest_path = STORE / "MANIFEST.json"
    entries = []
    for path in sorted(STORE.rglob("*")):
        if (
            not path.is_file()
            or path == manifest_path
            or path.name.startswith(".")
            or path.name.endswith(".partial")
        ):
            continue
        entries.append(
            {
                "path": str(path.relative_to(STORE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "ddm_rxc1_manifest.v1",
        "root": str(STORE),
        "entry_count": len(entries),
        "total_bytes": sum(int(row["bytes"]) for row in entries),
        "free_bytes_after_capture": shutil.disk_usage(STORE).free,
        "entries": entries,
    }
    atomic_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "baseline", "null", "screen", "manifest", "all"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage in {"preflight", "all"}:
        preflight()
        if args.stage == "preflight":
            return 0
    if not (STORE / "PREFLIGHT.json").is_file():
        raise Rxc1Error("preflight receipt is required")
    api = RestartableExactCoder()
    stages = {
        "baseline": lambda: stage_baseline(api),
        "null": lambda: stage_null(api),
        "screen": lambda: stage_screen(api),
        "manifest": stage_manifest,
    }
    if args.stage == "all":
        for name in ("baseline", "null", "screen", "manifest"):
            stages[name]()
    else:
        stages[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
