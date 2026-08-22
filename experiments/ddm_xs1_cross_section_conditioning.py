#!/usr/bin/env python3
"""DDM XS1: retained cross-section conditioning on the exact DX2 token field.

This scorer-free measurement reuses TO2's exact decoded token array and the
unchanged DX2 receiver.  It first materializes only section content that is
available before token decoding, then extends the incumbent 19-member adaptive
context model with one additional member.  The real RC64 stream is encoded and
decoded again; a candidate is valid only when its decoded field is byte-identical
to TO2 over all 117,964,800 positions.

The legal section features are:

* compensated carrier RGB at the token grid;
* the already-decoded selector mode, broadcast over its pair;
* a fixed all-zero-token probe through the already-decoded semantic renderer;
* a fixed joint quantization of all three.

Actual semantic RGB and selector-modified output are produced after token decode
and are deliberately absent.  HPAC output and the compact residual table are
containment controls: together they are inputs to the incumbent probability law,
so presenting them again is not a new conditioning source.

Every materialized payload is retained under the one charter-owned Vertigo root.
Every long stage is resumable at frame boundaries and preserves its final state.
No scorer, upstream file, shipped receiver, Modal job, or Metal job is touched.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
jg2 = importlib.import_module("experiments.ddm_jg2_tail_reencode")


STORE_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_xs1_cross_section_conditioning")
DEFAULT_STORE = STORE_ROOT / "measurement_v1"
TO2_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1")
TOKENS_SOURCE = TO2_ROOT / "retained/input/dx2_tokens_decoded.u8"
STREAM_SOURCE = TO2_ROOT / "retained/input/dx2_token_stream_rc64.bin"
CHECKPOINT_SOURCE = TO2_ROOT / "retained/input/tokens_cpu_stage_complete.json"
ARCHIVE_SOURCE = TO2_ROOT / "retained/input/archive.zip"
AD2_SOURCE = TO2_ROOT / "retained/input/ad2_result.json"
TO2_RESULT_SOURCE = TO2_ROOT / "RESULT.json"
RUNTIME_ROOT = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
RB1_SOURCE = REPO / ".omx/research/ddm_rb1_rate_bound_decomposition_20260822.md"
CX3_SOURCE = REPO / ".omx/research/ddm_cx3_context_axis_ceiling_20260822.md"
JX1_SOURCE = REPO / ".omx/research/ddm_jx1_joint_exchange_envelope_20260822.md"
TO2_MEMO_SOURCE = REPO / ".omx/research/ddm_to2_token_ordering_race_20260822.md"

T, H, W = 600, 384, 512
PLANE = H * W
POSITIONS = T * PLANE
CLASSES = 5
INCUMBENT_STREAM_BYTES = 113_777
INCUMBENT_ARCHIVE_BYTES = 180_368
INCUMBENT_SCORE = 0.14821987563243377
INCUMBENT_DISTORTION = 0.028120227975693968
FIXED_DISTORTION_CEILING = 137_986
REQUIRED_CUT_BYTES = 42_382
RATE_S_PER_BYTE = 25.0 / 37_545_489.0
DECODE_WALL_SECONDS = 498.0
CONTEST_BUDGET_SECONDS = 1800.0
CHECKPOINT_EVERY = 25
MIN_FREE_BYTES = 6 * (1 << 30)
RX1_COMPACT_RESIDUAL_BYTES = 96

EXPECTED: dict[Path, tuple[int | None, str]] = {
    ARCHIVE_SOURCE: (
        180_368,
        "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    ),
    TOKENS_SOURCE: (
        117_964_800,
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    ),
    STREAM_SOURCE: (
        113_777,
        "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5",
    ),
    CHECKPOINT_SOURCE: (
        3_511,
        "c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9",
    ),
    AD2_SOURCE: (
        134_747,
        "80124acd71ff63d4d9379b87674d1a976e1aa73857b4062a1c9ea2afb1b73511",
    ),
    RB1_SOURCE: (
        None,
        "fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09",
    ),
    JX1_SOURCE: (
        None,
        "9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd",
    ),
}

FEATURES: dict[str, tuple[str, np.dtype, int]] = {
    "carrier": ("carrier_rgb222.u8", np.dtype(np.uint8), 64),
    "semantic_probe": ("semantic_zero_rgb222.u8", np.dtype(np.uint8), 64),
    "selector": ("selector_mode.u8", np.dtype(np.uint8), 16),
    "joint": ("joint_c4_s4_selector4.u16", np.dtype("<u2"), 1024),
}
VARIANTS = ("baseline", *FEATURES)


class Xs1Error(RuntimeError):
    """Fail-closed XS1 refusal."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def file_fact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def progress(**record: object) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def require_store(store: Path) -> Path:
    resolved = store.resolve()
    root = STORE_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise Xs1Error(f"receipts must remain under {root}, not {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    probe = resolved / f".write_probe_{os.getpid()}"
    probe.write_bytes(b"xs1")
    probe.unlink()
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise Xs1Error(
            f"storage preflight requires {MIN_FREE_BYTES} free bytes; Vertigo has {free}"
        )
    return resolved


def verify_pin(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise Xs1Error(f"required input is absent: {path}")
    expected_bytes, expected_sha = EXPECTED[path]
    observed_bytes = path.stat().st_size
    observed_sha = sha256_file(path)
    if expected_bytes is not None and observed_bytes != expected_bytes:
        raise Xs1Error(f"byte pin mismatch for {path}: {observed_bytes} != {expected_bytes}")
    if observed_sha != expected_sha:
        raise Xs1Error(f"SHA-256 pin mismatch for {path}: {observed_sha} != {expected_sha}")
    return {"path": str(path), "bytes": observed_bytes, "sha256": observed_sha}


def retain_input(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    if source.stat().st_size != destination.stat().st_size:
        raise Xs1Error(f"retained input size drift: {destination}")
    source_sha = sha256_file(source)
    if sha256_file(destination) != source_sha:
        raise Xs1Error(f"retained input hash drift: {destination}")
    fact = file_fact(destination)
    fact["source_path"] = str(source.resolve())
    return fact


def stage_preflight(store: Path) -> dict[str, object]:
    receipt = store / "PRECHECK.json"
    if receipt.is_file():
        value = json.loads(receipt.read_text())
        for fact in value["retained_inputs"].values():
            path = Path(fact["path"])
            if file_fact(path)["sha256"] != fact["sha256"]:
                raise Xs1Error(f"resumed retained input drift: {path}")
        return value

    pins = {str(path): verify_pin(path) for path in EXPECTED}
    input_root = store / "retained/input"
    sources = {
        "archive": ARCHIVE_SOURCE,
        "tokens": TOKENS_SOURCE,
        "token_stream": STREAM_SOURCE,
        "token_checkpoint": CHECKPOINT_SOURCE,
        "ad2_result": AD2_SOURCE,
        "to2_result": TO2_RESULT_SOURCE,
        "rb1_memo": RB1_SOURCE,
        "cx3_memo": CX3_SOURCE,
        "jx1_memo": JX1_SOURCE,
        "to2_memo": TO2_MEMO_SOURCE,
    }
    retained = {
        name: retain_input(source, input_root / source.name)
        for name, source in sources.items()
    }
    runtime_files = sorted(
        path
        for root in (RUNTIME_ROOT / "runtime", RUNTIME_ROOT / "cpr1")
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    runtime_manifest = [file_fact(path) for path in runtime_files]
    payload = {
        "schema": "ddm_xs1.preflight.v1",
        "complete": True,
        "axis": "[macOS-CPU advisory / scorer-free exact lossless measurement]",
        "git_head": git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "pins": pins,
        "retained_inputs": retained,
        "runtime_root": str(RUNTIME_ROOT.resolve()),
        "runtime_source_manifest": runtime_manifest,
        "storage": {
            "root": str(store),
            "free_bytes_after_inputs": shutil.disk_usage(store).free,
            "minimum_required_bytes": MIN_FREE_BYTES,
        },
        "forbidden_actions": {
            "scorer": False,
            "modal": False,
            "metal": False,
            "upstream_write": False,
            "shipped_receiver_write": False,
        },
    }
    atomic_json(receipt, payload)
    return payload


def _load_decode_sections():
    """Load exact receiver objects and all pre-token decoded section state."""
    import torch

    residual, renderer, renderer_dir = jg2.load_runtime(RUNTIME_ROOT)
    parts = residual.read_residual_archive(ARCHIVE_SOURCE)
    carrier_repack = importlib.import_module("runtime.carrier_repack")
    compensation = importlib.import_module("runtime.compensation_overlay")
    weight_codec = importlib.import_module("runtime.entropy.renderer_weight_codec")

    carrier_blob, selector_blob = carrier_repack.split_frame0_selector_carrier(
        parts.carrier_blob
    )
    if selector_blob is None:
        raise Xs1Error("DX2 carrier lacks its already-decoded selector section")
    canonical_carrier = carrier_repack.materialize_cpr1(carrier_blob, renderer)
    semantic_marker = bytes(40_252)
    semantic_pose = (
        struct.pack("<II", len(semantic_marker), len(canonical_carrier))
        + semantic_marker
        + canonical_carrier
    )
    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
    if parts.compensation_blob is not None:
        basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
        _, _, coefficient_scales, encoded = renderer.decode_compact_carrier(
            canonical_carrier,
            basis_count=basis_count,
            frames=renderer.N,
            dimensions=renderer.CARRIER_DIM,
        )
        delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
        base_codes = np.cumsum(delta, axis=0) & 0xFFF
        base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes).astype(
            np.int32
        )
        candidate_codes = compensation.apply_compensation_overlay(
            base_codes, parts.compensation_blob
        )
        coefficients = torch.from_numpy(candidate_codes).float() * torch.from_numpy(
            coefficient_scales
        )[None]

    semantic = renderer.SemanticTokenRenderer(96)
    tagged = renderer.unpack_variant_semantic_or_none(
        parts.semantic_blob, semantic.state_dict()
    )
    if tagged is None:
        records = weight_codec.decode_wans1(parts.semantic_blob)
        tagged = {
            record.schema.name: torch.from_numpy(
                np.ascontiguousarray(record.values, dtype=np.float32)
            )
            for record in records
        }
    semantic.load_state_dict(tagged, strict=True)
    selector_module = importlib.import_module("runtime.frame0_selector")
    modes, selector_indices = selector_module.decode_selector(selector_blob)
    if selector_indices.shape != (T,) or len(modes) > 16:
        raise Xs1Error("selector feature exceeds the fixed 16-mode legal feature domain")
    return residual, renderer, renderer_dir, parts, semantic, basis, coefficients, selector_indices


def rgb222(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.uint8)
    return (
        ((value[..., 0] >> 6) << 4)
        | ((value[..., 1] >> 6) << 2)
        | (value[..., 2] >> 6)
    ).astype(np.uint8)


def _feature_paths(store: Path, partial: bool = False) -> dict[str, Path]:
    suffix = ".partial" if partial else ""
    root = store / "retained/features"
    return {
        "carrier_rgb": root / f"carrier_eval_rgb.u8{suffix}",
        "semantic_rgb": root / f"semantic_zero_eval_rgb.u8{suffix}",
        "carrier": root / f"carrier_rgb222.u8{suffix}",
        "semantic_probe": root / f"semantic_zero_rgb222.u8{suffix}",
        "selector": root / f"selector_mode.u8{suffix}",
        "joint": root / f"joint_c4_s4_selector4.u16{suffix}",
    }


def stage_features(store: Path) -> dict[str, object]:
    import torch

    receipt_path = store / "FEATURES.json"
    final = _feature_paths(store)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        for fact in receipt["payloads"].values():
            if file_fact(Path(fact["path"]))["sha256"] != fact["sha256"]:
                raise Xs1Error(f"resumed feature payload drift: {fact['path']}")
        return receipt

    partial = _feature_paths(store, partial=True)
    partial["carrier_rgb"].parent.mkdir(parents=True, exist_ok=True)
    progress_path = store / "work/features.progress.json"
    start = 0
    if progress_path.is_file():
        state = json.loads(progress_path.read_text())
        start = int(state["next_frame"])
        missing = [str(path) for path in partial.values() if not path.is_file()]
        if missing:
            raise Xs1Error(f"feature resume state exists but partial payloads are absent: {missing}")
        mode = "r+"
    else:
        mode = "w+"

    carrier_rgb = np.memmap(partial["carrier_rgb"], mode=mode, dtype=np.uint8, shape=(T, H, W, 3))
    semantic_rgb = np.memmap(partial["semantic_rgb"], mode=mode, dtype=np.uint8, shape=(T, H, W, 3))
    carrier_bins = np.memmap(partial["carrier"], mode=mode, dtype=np.uint8, shape=(T, H, W))
    semantic_bins = np.memmap(
        partial["semantic_probe"], mode=mode, dtype=np.uint8, shape=(T, H, W)
    )
    selector_bins = np.memmap(partial["selector"], mode=mode, dtype=np.uint8, shape=(T, H, W))
    joint_bins = np.memmap(partial["joint"], mode=mode, dtype="<u2", shape=(T, H, W))

    (
        _residual,
        renderer,
        _renderer_dir,
        _parts,
        semantic,
        basis,
        coefficients,
        selector_indices,
    ) = _load_decode_sections()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    device = torch.device("cpu")
    semantic = semantic.eval().to(device)
    basis = renderer.normalized_basis(basis.to(device))
    coefficients = coefficients.to(device)
    zeros = torch.zeros((1, H, W), dtype=torch.long, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        for frame in range(start, T):
            carrier = torch.einsum("k,kchw->chw", coefficients[frame], basis)
            carrier = carrier / math.sqrt(renderer.CARRIER_DIM)
            carrier_u8 = (
                (127.5 + renderer.CARRIER_AMPLITUDE * carrier)
                .clamp(0.0, 255.0)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            index = torch.tensor([frame], dtype=torch.long, device=device)
            semantic_u8 = (
                semantic(zeros, index)[0]
                .clamp(0.0, 255.0)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            cbin = rgb222(carrier_u8)
            sbin = rgb222(semantic_u8)
            selector = int(selector_indices[frame])
            carrier_rgb[frame] = carrier_u8
            semantic_rgb[frame] = semantic_u8
            carrier_bins[frame] = cbin
            semantic_bins[frame] = sbin
            selector_bins[frame] = selector
            joint_bins[frame] = (
                ((cbin.astype(np.uint16) >> 2) * 64)
                + ((sbin.astype(np.uint16) >> 2) * 4)
                + min(selector, 3)
            )
            if (frame + 1) % CHECKPOINT_EVERY == 0 or frame + 1 == T:
                for array in (
                    carrier_rgb,
                    semantic_rgb,
                    carrier_bins,
                    semantic_bins,
                    selector_bins,
                    joint_bins,
                ):
                    array.flush()
                state = {
                    "schema": "ddm_xs1.features.progress.v1",
                    "next_frame": frame + 1,
                    "elapsed_seconds": time.perf_counter() - started,
                    "partial_payloads": {name: str(path) for name, path in partial.items()},
                }
                atomic_json(progress_path, state)
                atomic_json(
                    store / f"checkpoints/features_frame_{frame + 1:04d}.json", state
                )
                progress(stage="features", frame=frame + 1, elapsed=state["elapsed_seconds"])

    del carrier_rgb, semantic_rgb, carrier_bins, semantic_bins, selector_bins, joint_bins
    for name, path in partial.items():
        os.replace(path, final[name])
    receipt = {
        "schema": "ddm_xs1.features.v1",
        "complete": True,
        "frames": T,
        "positions": POSITIONS,
        "decode_order_legality": {
            "carrier": "decoded and compensation-applied before token decode",
            "selector": "selector indices decoded before token decode; application occurs later",
            "semantic_probe": "fixed all-zero token input; weights and frame index are known before token decode",
            "joint": "fixed quantization of the three legal features",
            "excluded_actual_semantic_rgb": "depends on the token field and is circular",
            "excluded_selected_frame0_output": "rendered and selector-modified after token decode",
        },
        "quantization": {
            "carrier": "RGB222, 64 fixed bins",
            "semantic_probe": "RGB222, 64 fixed bins",
            "selector": "exact decoded mode index in a fixed 16-bin domain",
            "joint": "carrier RGB222>>2 x semantic RGB222>>2 x selector capped at 3; 1024 fixed bins",
        },
        "payloads": {name: file_fact(path) for name, path in final.items()},
    }
    atomic_json(receipt_path, receipt)
    return receipt


def load_feature_maps(store: Path) -> dict[str, np.ndarray]:
    paths = _feature_paths(store)
    maps: dict[str, np.ndarray] = {}
    for name, (filename, dtype, _bins) in FEATURES.items():
        path = paths[name]
        if path.name != filename:
            raise Xs1Error(f"feature filename contract drift for {name}: {path.name}")
        expected = POSITIONS * dtype.itemsize
        if path.stat().st_size != expected:
            raise Xs1Error(f"feature payload size drift for {name}: {path}")
        maps[name] = np.memmap(path, mode="r", dtype=dtype, shape=(T, H, W))
    return maps


def make_corrector(feature_name: str | None, feature_bins: int = 0):
    free_module = importlib.import_module("runtime.free_corrector")
    if feature_name is None:
        return free_module.FreeCorrector(PLANE)

    fx1 = importlib.import_module("runtime.fx1_logistic_mixer_corrector")
    rr4 = importlib.import_module("runtime.rr4_free_corrector")
    base_class = free_module.FreeCorrector
    mixer_family = fx1.MixerFamily
    weight_one = fx1.WEIGHT_STORE_ONE
    u_bins = rr4.U_BINS

    class CrossSectionCorrector(base_class):
        def __init__(self) -> None:
            super().__init__(PLANE)
            self._xs_current = np.zeros(PLANE, dtype=np.int64)
            self._xs_positions = np.zeros(0, dtype=np.int64)

            def rule(features):
                section = self._xs_current[self._xs_positions]
                return (
                    (features["cls"] * u_bins + features["ubin"]) * feature_bins
                    + section
                )

            self.families.append(
                mixer_family(
                    f"xs1_{feature_name}", CLASSES * u_bins * feature_bins, rule
                )
            )
            old = self.weights
            self.weights = np.zeros((old.shape[0], old.shape[1] + 1), dtype=np.int64)
            self.weights[:, : old.shape[1]] = old
            self.weights[:, 0] = weight_one

        def set_feature(self, frame_feature: np.ndarray) -> None:
            flat = np.asarray(frame_feature, dtype=np.int64).reshape(-1)
            if flat.shape != (PLANE,) or np.any(flat < 0) or np.any(flat >= feature_bins):
                raise Xs1Error(f"{feature_name} frame feature is outside its domain")
            self._xs_current[:] = flat

        def group_state(self, probability, predicted, positions):
            self._xs_positions = np.asarray(positions, dtype=np.int64).reshape(-1)
            return super().group_state(probability, predicted, positions)

    CrossSectionCorrector.__name__ = f"Xs1{feature_name.title()}Corrector"
    return CrossSectionCorrector()


def _save_corrector_checkpoint(
    path: Path,
    corrector: object,
    *,
    frame: int,
    bits: float,
    per_frame: np.ndarray,
    schema: str,
) -> None:
    state = jg2.corrector_state(corrector)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial.npz")
    np.savez(
        temporary,
        schema=np.array([schema]),
        frame=np.array([frame], dtype=np.int64),
        code_bits=np.array([bits], dtype=np.float64),
        per_frame=per_frame,
        **state,
    )
    os.replace(temporary, path)


def _load_corrector_checkpoint(
    path: Path, corrector: object, schema: str
) -> tuple[int, float, np.ndarray]:
    blob = np.load(path, allow_pickle=False)
    observed = str(np.asarray(blob["schema"]).reshape(-1)[0])
    if observed != schema:
        raise Xs1Error(f"refusing checkpoint schema {observed!r}; expected {schema!r}")
    state = {
        key: blob[key]
        for key in blob.files
        if key not in {"schema", "frame", "code_bits", "per_frame"}
    }
    # The cross-section member retains the last group's position vector.  Its
    # length varies by group, so a cold construction cannot know the checkpoint
    # shape.  Resize only such already-declared ndarray attributes before using
    # JG2's exact dtype/round-trip restore; no key may be invented here.
    for key, value in state.items():
        owner, attribute = jg2._resolve_target(corrector, key)
        current = getattr(owner, attribute, None)
        array = np.asarray(value)
        if isinstance(current, np.ndarray) and current.shape != array.shape:
            setattr(owner, attribute, np.empty(array.shape, dtype=current.dtype))
    jg2.load_corrector_state(
        corrector,
        state,
    )
    return (
        int(blob["frame"][0]),
        float(blob["code_bits"][0]),
        np.asarray(blob["per_frame"], dtype=np.float64).copy(),
    )


def _row_bits(rows: np.ndarray, symbols: np.ndarray) -> float:
    selected = rows.astype(np.float64)[np.arange(symbols.size), symbols]
    return float(-np.log2(np.maximum(selected, 1e-300)).sum())


def _raw_encoder_body(encoder: object) -> bytes:
    """Finish RC64 and retain the exact raw body stored by the DX2 receiver.

    ``NativeRc64Encoder.finish`` returns a transport envelope
    ``R6D1 || body || alignment`` for its own decoder wrapper.  DX2 stores only
    ``body`` in the RX1 token section.  Reading the C encoder's exact byte count
    avoids both the four-byte envelope and an unsafe ``rstrip(0)`` that could
    delete a legitimate final zero byte.
    """
    encoder.finish()
    size = int(encoder.library.rc64_encoder_size(encoder.context))
    pointer = encoder.library.rc64_encoder_data(encoder.context)
    if not size or not pointer:
        raise Xs1Error("RC64 encoder produced no raw body")
    return ctypes.string_at(pointer, size)


def _incumbent_context(coding: np.ndarray, state: object) -> np.ndarray:
    rr4 = importlib.import_module("runtime.rr4_free_corrector")
    arg = np.asarray(state.arg, dtype=np.int64)
    q = coding.astype(np.float64)[np.arange(arg.size), arg]
    one_minus = np.maximum(1.0 - q, rr4.PROB_EPS)
    below = np.searchsorted(rr4._SURPRISE_ASC, one_minus, side="left")
    qbin = np.clip((rr4.U_BINS - 1) - below, 0, rr4.U_BINS - 1).astype(np.int64)
    return (arg * rr4.U_BINS + qbin).astype("<u2")


def _runtime_replay_setup():
    import torch

    residual, renderer, renderer_dir = jg2.load_runtime(RUNTIME_ROOT)
    parts = residual.read_residual_archive(ARCHIVE_SOURCE)
    device = torch.device("cpu")
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    hpac_inference = importlib.import_module("runtime.hpac_inference")
    hpac_inference.optimize_sparse_evaluator(sparse)
    plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        plans.append((torch.from_numpy(flat).to(device), flat))
    return torch, residual, renderer, renderer_dir, parts, device, model, sparse, plans


def _new_correctors() -> dict[str, object]:
    correctors = {"baseline": make_corrector(None)}
    for name, (_filename, _dtype, bins) in FEATURES.items():
        correctors[name] = make_corrector(name, bins)
    return correctors


def _checkpoint_all_correctors(
    store: Path,
    correctors: dict[str, object],
    encoders: dict[str, object],
    per_frame: dict[str, np.ndarray],
    code_bits: dict[str, float],
    model_seconds: dict[str, float],
    frame: int,
    schema: str,
) -> None:
    root = store / "work/encode_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    cold = _new_correctors()
    for name in VARIANTS:
        state = jg2.corrector_state(correctors[name])
        lost = jg2.uncaptured_divergent_state(correctors[name], cold[name], set(state))
        if lost:
            raise Xs1Error(f"checkpoint would lose {name} corrector state: {lost}")
        atomic_bytes(root / f"{name}.encoder.bin", encoders[name].snapshot())
        _save_corrector_checkpoint(
            root / f"{name}.corrector.npz",
            correctors[name],
            frame=frame,
            bits=code_bits[name],
            per_frame=per_frame[name],
            schema=schema,
        )
    checkpoint = {
        "schema": schema,
        "next_frame": frame,
        "variants": list(VARIANTS),
        "model_seconds": model_seconds,
    }
    atomic_json(root / "STATE.json", checkpoint)
    atomic_json(store / f"checkpoints/encode_frame_{frame:04d}.json", checkpoint)


def _splice_archive(stream: bytes, destination: Path) -> None:
    member = jg2.read_archive_member(ARCHIVE_SOURCE)
    sections = jg2.split_member(member)
    if sections["tail"][RX1_COMPACT_RESIDUAL_BYTES:] != STREAM_SOURCE.read_bytes():
        raise Xs1Error("DX2 tail split disagrees with the pinned TO2 token stream")
    sections["tail"] = sections["tail"][:RX1_COMPACT_RESIDUAL_BYTES] + stream
    jg2.pack_archive(jg2.join_member(sections), destination)


def stage_encode(store: Path, resume: bool) -> dict[str, object]:
    receipt_path = store / "ENCODE.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        for row in receipt["variants"].values():
            for key in ("stream", "archive", "per_frame_bits", "final_corrector_state"):
                fact = row[key]
                if file_fact(Path(fact["path"]))["sha256"] != fact["sha256"]:
                    raise Xs1Error(f"resumed encode payload drift: {fact['path']}")
        return receipt

    feature_maps = load_feature_maps(store)
    tokens = np.memmap(TOKENS_SOURCE, mode="r", dtype=np.uint8, shape=(T, H, W))
    (
        torch,
        residual,
        renderer,
        _renderer_dir,
        parts,
        device,
        model,
        sparse,
        plans,
    ) = _runtime_replay_setup()
    route_b = jg2.load_route_b()
    library, build = jg2.compile_rc64(store / "work", route_b, "xs1")
    correctors = _new_correctors()
    encoders: dict[str, object] = {}
    per_frame = {name: np.zeros(T, dtype=np.float64) for name in VARIANTS}
    code_bits = dict.fromkeys(VARIANTS, 0.0)
    model_seconds = dict.fromkeys(VARIANTS, 0.0)
    start_frame = 0
    schema = "ddm_xs1.encode_checkpoint.v1"
    checkpoint_root = store / "work/encode_checkpoints"
    state_path = checkpoint_root / "STATE.json"
    if resume and state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("schema") != schema or tuple(state.get("variants", ())) != VARIANTS:
            raise Xs1Error("encode checkpoint contract drift")
        start_frame = int(state["next_frame"])
        model_seconds = {
            name: float(state.get("model_seconds", {}).get(name, 0.0))
            for name in VARIANTS
        }
        for name in VARIANTS:
            observed_frame, code_bits[name], per_frame[name] = _load_corrector_checkpoint(
                checkpoint_root / f"{name}.corrector.npz", correctors[name], schema
            )
            if observed_frame != start_frame:
                raise Xs1Error("encode corrector checkpoints disagree on frame")
            encoders[name] = route_b.NativeRc64Encoder(
                library, (checkpoint_root / f"{name}.encoder.bin").read_bytes()
            )
    else:
        encoders = {name: route_b.NativeRc64Encoder(library) for name in VARIANTS}

    context_partial = store / "retained/mi/incumbent_context_arg_qbin.u16.partial"
    context_final = store / "retained/mi/incumbent_context_arg_qbin.u16"
    context_partial.parent.mkdir(parents=True, exist_ok=True)
    if start_frame:
        if not context_partial.is_file():
            raise Xs1Error("encode checkpoint exists without its incumbent context payload")
        context_map = np.memmap(context_partial, mode="r+", dtype="<u2", shape=(T, H, W))
    else:
        context_map = np.memmap(context_partial, mode="w+", dtype="<u2", shape=(T, H, W))

    previous = (
        torch.from_numpy(np.asarray(tokens[start_frame - 1], dtype=np.int64)).reshape(1, H, W)
        if start_frame
        else torch.zeros((1, H, W), dtype=torch.long, device=device)
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for frame in range(start_frame, T):
            current = torch.zeros_like(previous)
            index = torch.tensor([frame], dtype=torch.long, device=device)
            hpac_context = model.prepare_frame_context(index, previous)
            if frame:
                boundary = residual._boundary_buckets(
                    previous[0].to(dtype=torch.uint8).cpu().numpy()
                ).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            for name, corrector in correctors.items():
                corrector.begin_frame(boundary)
                if name != "baseline":
                    corrector.set_feature(feature_maps[name][frame])
            target_flat = np.asarray(tokens[frame], dtype=np.uint8).reshape(-1)
            for group, (device_positions, flat) in enumerate(plans):
                selected = sparse.selected_logits(current, hpac_context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                table_feature = boundary[flat].astype(np.int64) * CLASSES + predicted
                corrected = base_logits + parts.table.values[table_feature]
                probability = residual._probability_table(
                    corrected, renderer.HPAC_LOGIT_PRECISION
                )
                symbols = target_flat[flat].astype(np.int64)
                for name in VARIANTS:
                    started_variant = time.perf_counter()
                    state = correctors[name].group_state(probability, predicted, flat)
                    coding = correctors[name].coding_row(state)
                    if name == "baseline":
                        context_map[frame].reshape(-1)[flat] = _incumbent_context(
                            coding, state
                        )
                    bits = _row_bits(coding, symbols)
                    encoders[name].encode(symbols.astype(np.int32), coding)
                    correctors[name].observe(state, symbols)
                    model_seconds[name] += time.perf_counter() - started_variant
                    per_frame[name][frame] += bits
                    code_bits[name] += bits
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
            frame_tokens = current[0].to(dtype=torch.uint8).cpu().numpy()
            if not np.array_equal(frame_tokens.reshape(-1), target_flat):
                raise Xs1Error(f"encode trajectory diverged at frame {frame}")
            for corrector in correctors.values():
                corrector.end_frame(frame_tokens.reshape(-1))
            previous = current
            if (frame + 1) % CHECKPOINT_EVERY == 0 and frame + 1 < T:
                context_map.flush()
                _checkpoint_all_correctors(
                    store,
                    correctors,
                    encoders,
                    per_frame,
                    code_bits,
                    model_seconds,
                    frame + 1,
                    schema,
                )
                progress(
                    stage="encode",
                    frame=frame + 1,
                    elapsed=time.perf_counter() - started,
                )

    context_map.flush()
    del context_map
    os.replace(context_partial, context_final)
    retained = store / "retained/encode"
    retained.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, object]] = {}
    baseline_stream: bytes | None = None
    for name in VARIANTS:
        stream = _raw_encoder_body(encoders[name])
        stream_path = retained / f"token_stream_{name}.bin"
        atomic_bytes(stream_path, stream)
        bits_path = retained / f"bits_per_frame_{name}.npy"
        temporary_bits = bits_path.with_name(f".{bits_path.name}.partial.npy")
        np.save(temporary_bits, per_frame[name])
        os.replace(temporary_bits, bits_path)
        state_path = retained / f"corrector_final_{name}.npz"
        _save_corrector_checkpoint(
            state_path,
            correctors[name],
            frame=T,
            bits=code_bits[name],
            per_frame=per_frame[name],
            schema="ddm_xs1.encode_final.v1",
        )
        archive_path = retained / f"candidate_{name}.zip"
        _splice_archive(stream, archive_path)
        if name == "baseline":
            baseline_stream = stream
        rows[name] = {
            "stream": file_fact(stream_path),
            "archive": file_fact(archive_path),
            "per_frame_bits": file_fact(bits_path),
            "final_corrector_state": file_fact(state_path),
            "ideal_code_bits": code_bits[name],
            "ideal_code_bytes": code_bits[name] / 8.0,
            "measured_model_and_coder_seconds": model_seconds[name],
        }
    shipped = STREAM_SOURCE.read_bytes()
    if baseline_stream != shipped:
        common = min(len(baseline_stream or b""), len(shipped))
        prefix = next(
            (
                index
                for index in range(common)
                if (baseline_stream or b"")[index] != shipped[index]
            ),
            common,
        )
        raise Xs1Error(
            f"baseline control failed: output differs at byte {prefix}; no candidate delta is trusted"
        )
    if sha256_file(retained / "candidate_baseline.zip") != sha256_file(ARCHIVE_SOURCE):
        raise Xs1Error("baseline archive repack is not byte-identical to DX2")

    receipt = {
        "schema": "ddm_xs1.encode.v1",
        "complete": True,
        "baseline_byte_identical": True,
        "positions_encoded": POSITIONS,
        "incumbent_context": file_fact(context_final),
        "rc64_build": build,
        "variants": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _save_decode_checkpoint(
    store: Path,
    correctors: dict[str, object],
    decoders: dict[str, object],
    model_seconds: dict[str, float],
    frame: int,
    schema: str,
) -> None:
    root = store / "work/decode_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    cold = _new_correctors()
    for name in VARIANTS:
        state = jg2.corrector_state(correctors[name])
        lost = jg2.uncaptured_divergent_state(correctors[name], cold[name], set(state))
        if lost:
            raise Xs1Error(f"decode checkpoint would lose {name} state: {lost}")
        atomic_bytes(root / f"{name}.decoder.bin", decoders[name].get_compressed().tobytes())
        _save_corrector_checkpoint(
            root / f"{name}.corrector.npz",
            correctors[name],
            frame=frame,
            bits=0.0,
            per_frame=np.zeros(T, dtype=np.float64),
            schema=schema,
        )
    state = {
        "schema": schema,
        "next_frame": frame,
        "variants": list(VARIANTS),
        "model_seconds": model_seconds,
    }
    atomic_json(root / "STATE.json", state)
    atomic_json(store / f"checkpoints/decode_frame_{frame:04d}.json", state)


def stage_verify(store: Path, resume: bool) -> dict[str, object]:
    receipt_path = store / "VERIFY.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        for fact in receipt["decoded_outputs"].values():
            if file_fact(Path(fact["path"]))["sha256"] != fact["sha256"]:
                raise Xs1Error(f"resumed decoded output drift: {fact['path']}")
        return receipt

    encode_receipt = json.loads((store / "ENCODE.json").read_text())
    feature_maps = load_feature_maps(store)
    tokens = np.memmap(TOKENS_SOURCE, mode="r", dtype=np.uint8, shape=(T, H, W))
    (
        torch,
        residual,
        renderer,
        _renderer_dir,
        parts,
        device,
        model,
        sparse,
        plans,
    ) = _runtime_replay_setup()
    route_b = jg2.load_route_b()
    library = Path(encode_receipt["rc64_build"]["library"]["path"])
    correctors = _new_correctors()
    decoders: dict[str, object] = {}
    schema = "ddm_xs1.decode_checkpoint.v1"
    checkpoint_root = store / "work/decode_checkpoints"
    state_path = checkpoint_root / "STATE.json"
    start_frame = 0
    model_seconds = dict.fromkeys(VARIANTS, 0.0)
    if resume and state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("schema") != schema or tuple(state.get("variants", ())) != VARIANTS:
            raise Xs1Error("decode checkpoint contract drift")
        start_frame = int(state["next_frame"])
        model_seconds = {
            name: float(state.get("model_seconds", {}).get(name, 0.0))
            for name in VARIANTS
        }
        for name in VARIANTS:
            observed_frame, _bits, _ledger = _load_corrector_checkpoint(
                checkpoint_root / f"{name}.corrector.npz", correctors[name], schema
            )
            if observed_frame != start_frame:
                raise Xs1Error("decode corrector checkpoints disagree on frame")
            decoders[name] = route_b.NativeRc64Decoder(
                library, (checkpoint_root / f"{name}.decoder.bin").read_bytes()
            )
    else:
        for name in VARIANTS:
            stream_path = Path(encode_receipt["variants"][name]["stream"]["path"])
            raw = stream_path.read_bytes()
            decoders[name] = route_b.NativeRc64Decoder(
                library, route_b.TOKEN_MAGIC + raw
            )

    output_root = store / "retained/decode"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, np.memmap] = {}
    for name in VARIANTS:
        path = output_root / f"decoded_{name}.u8.partial"
        mode = "r+" if start_frame else "w+"
        if start_frame and not path.is_file():
            raise Xs1Error(f"decode resume output is absent: {path}")
        outputs[name] = np.memmap(path, mode=mode, dtype=np.uint8, shape=(T, H, W))

    previous = (
        torch.from_numpy(np.asarray(tokens[start_frame - 1], dtype=np.int64)).reshape(1, H, W)
        if start_frame
        else torch.zeros((1, H, W), dtype=torch.long, device=device)
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for frame in range(start_frame, T):
            current = torch.zeros_like(previous)
            index = torch.tensor([frame], dtype=torch.long, device=device)
            hpac_context = model.prepare_frame_context(index, previous)
            if frame:
                boundary = residual._boundary_buckets(
                    previous[0].to(dtype=torch.uint8).cpu().numpy()
                ).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            for name, corrector in correctors.items():
                corrector.begin_frame(boundary)
                if name != "baseline":
                    corrector.set_feature(feature_maps[name][frame])
            target_flat = np.asarray(tokens[frame], dtype=np.uint8).reshape(-1)
            for group, (device_positions, flat) in enumerate(plans):
                selected = sparse.selected_logits(current, hpac_context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                table_feature = boundary[flat].astype(np.int64) * CLASSES + predicted
                corrected = base_logits + parts.table.values[table_feature]
                probability = residual._probability_table(
                    corrected, renderer.HPAC_LOGIT_PRECISION
                )
                group_symbols: dict[str, np.ndarray] = {}
                for name in VARIANTS:
                    started_variant = time.perf_counter()
                    corrector_state = correctors[name].group_state(probability, predicted, flat)
                    coding = correctors[name].coding_row(corrector_state)
                    symbols = decoders[name].decode(None, coding).astype(np.int64)
                    group_symbols[name] = symbols
                    correctors[name].observe(corrector_state, symbols)
                    model_seconds[name] += time.perf_counter() - started_variant
                    outputs[name][frame].reshape(-1)[flat] = symbols.astype(np.uint8)
                expected = target_flat[flat].astype(np.int64)
                for name, symbols in group_symbols.items():
                    if not np.array_equal(symbols, expected):
                        mismatch = int(np.flatnonzero(symbols != expected)[0])
                        for output in outputs.values():
                            output.flush()
                        raise Xs1Error(
                            f"{name} decoder diverged at frame {frame}, group {group}, local {mismatch}"
                        )
                current.reshape(-1)[device_positions] = torch.from_numpy(expected).to(device)
            for name, corrector in correctors.items():
                corrector.end_frame(np.asarray(outputs[name][frame]).reshape(-1))
            previous = current
            if (frame + 1) % CHECKPOINT_EVERY == 0 and frame + 1 < T:
                for output in outputs.values():
                    output.flush()
                _save_decode_checkpoint(
                    store,
                    correctors,
                    decoders,
                    model_seconds,
                    frame + 1,
                    schema,
                )
                progress(
                    stage="verify",
                    frame=frame + 1,
                    elapsed=time.perf_counter() - started,
                )

    decoded_facts: dict[str, dict[str, object]] = {}
    for name, output in outputs.items():
        output.flush()
        del output
        partial = output_root / f"decoded_{name}.u8.partial"
        final = output_root / f"decoded_{name}.u8"
        os.replace(partial, final)
        fact = file_fact(final)
        if fact["sha256"] != EXPECTED[TOKENS_SOURCE][1]:
            raise Xs1Error(f"{name} final decoded field is not TO2 byte-identical")
        if not decoders[name].is_empty():
            raise Xs1Error(f"{name} decoder did not consume exactly {POSITIONS} symbols")
        decoded_facts[name] = fact
    receipt = {
        "schema": "ddm_xs1.verify.v1",
        "complete": True,
        "positions_verified_per_variant": POSITIONS,
        "target_sha256": EXPECTED[TOKENS_SOURCE][1],
        "all_variants_byte_identical": True,
        "decoded_outputs": decoded_facts,
        "measured_model_and_decoder_seconds": model_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def conditional_mutual_information(counts: np.ndarray) -> dict[str, float]:
    values = np.asarray(counts, dtype=np.float64)
    n = float(values.sum())
    if n != POSITIONS:
        raise Xs1Error(f"MI denominator is {n}, expected {POSITIONS}")
    n_cz = values.sum(axis=2, keepdims=True)
    n_cy = values.sum(axis=1, keepdims=True)
    n_c = values.sum(axis=(1, 2), keepdims=True)
    expected_denominator = n_cz * n_cy
    mask = (values > 0.0) & (expected_denominator > 0.0)
    term = np.zeros_like(values)
    numerator = values * n_c
    term[mask] = values[mask] * np.log2(numerator[mask] / expected_denominator[mask])
    bits_per_symbol = float(term.sum() / n)
    total_bits = bits_per_symbol * n
    return {
        "plugin_cmi_bits_per_symbol": bits_per_symbol,
        "plugin_cmi_total_bits": total_bits,
        "plugin_cmi_total_bytes": total_bits / 8.0,
    }


def stage_mi(store: Path) -> dict[str, object]:
    receipt_path = store / "MI.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        for row in receipt["sections"].values():
            fact = row["counts"]
            if file_fact(Path(fact["path"]))["sha256"] != fact["sha256"]:
                raise Xs1Error(f"resumed MI counts drift: {fact['path']}")
        return receipt

    contexts = np.memmap(
        store / "retained/mi/incumbent_context_arg_qbin.u16",
        mode="r",
        dtype="<u2",
        shape=(T, H, W),
    )
    tokens = np.memmap(TOKENS_SOURCE, mode="r", dtype=np.uint8, shape=(T, H, W))
    features = load_feature_maps(store)
    jg2.load_runtime(RUNTIME_ROOT)
    rr4 = importlib.import_module("runtime.rr4_free_corrector")
    context_bins = CLASSES * rr4.U_BINS
    block_root = store / "retained/mi/blocks"
    count_root = store / "retained/mi/counts"
    block_root.mkdir(parents=True, exist_ok=True)
    count_root.mkdir(parents=True, exist_ok=True)
    sections: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for name, (_filename, _dtype, feature_bins) in FEATURES.items():
        block_paths: list[Path] = []
        for start in range(0, T, CHECKPOINT_EVERY):
            end = min(start + CHECKPOINT_EVERY, T)
            block_path = block_root / f"{name}_{start:04d}_{end:04d}.npy"
            block_paths.append(block_path)
            if block_path.is_file():
                expected_bytes = 128 + context_bins * feature_bins * CLASSES * 8
                if block_path.stat().st_size != expected_bytes:
                    raise Xs1Error(f"MI block size drift: {block_path}")
                continue
            c = np.asarray(contexts[start:end], dtype=np.int64).reshape(-1)
            z = np.asarray(features[name][start:end], dtype=np.int64).reshape(-1)
            y = np.asarray(tokens[start:end], dtype=np.int64).reshape(-1)
            if np.any(c >= context_bins) or np.any(z >= feature_bins) or np.any(y >= CLASSES):
                raise Xs1Error(f"{name} MI input is outside its declared domain")
            flat = (c * feature_bins + z) * CLASSES + y
            counts = np.bincount(
                flat, minlength=context_bins * feature_bins * CLASSES
            ).reshape(context_bins, feature_bins, CLASSES).astype(np.int64)
            temporary = block_path.with_name(f".{block_path.name}.{os.getpid()}.partial.npy")
            np.save(temporary, counts)
            os.replace(temporary, block_path)
            atomic_json(
                store / f"checkpoints/mi_{name}_{end:04d}.json",
                {
                    "schema": "ddm_xs1.mi_block.v1",
                    "section": name,
                    "start_frame": start,
                    "end_frame": end,
                    "positions": int(counts.sum()),
                    "counts": file_fact(block_path),
                },
            )
            progress(stage="mi", section=name, frame=end)
        total = np.zeros((context_bins, feature_bins, CLASSES), dtype=np.int64)
        block_facts = []
        for path in block_paths:
            block = np.load(path, mmap_mode="r", allow_pickle=False)
            total += block
            block_facts.append(file_fact(path))
        counts_path = count_root / f"counts_{name}.npy"
        temporary = counts_path.with_name(f".{counts_path.name}.{os.getpid()}.partial.npy")
        np.save(temporary, total)
        os.replace(temporary, counts_path)
        cmi = conditional_mutual_information(total)
        sections[name] = {
            "conditioning": "exact incumbent emitted argmax class x 64-bin emitted hit-probability statistic",
            "feature_bins": feature_bins,
            "positions": int(total.sum()),
            **cmi,
            "counts": file_fact(counts_path),
            "retained_blocks": block_facts,
        }
    receipt = {
        "schema": "ddm_xs1.mi.v1",
        "complete": True,
        "denominator_positions": POSITIONS,
        "context_bins": context_bins,
        "context_payload": file_fact(
            store / "retained/mi/incumbent_context_arg_qbin.u16"
        ),
        "sections": sections,
        "containment_controls": {
            "hpac": {
                "cmi_bits": 0.0,
                "reason": "HPAC probabilities are already an input to the incumbent emitted law",
            },
            "compact_residual": {
                "cmi_bits": 0.0,
                "reason": "the fixed residual table is already applied before the incumbent emitted law",
            },
            "actual_semantic_rgb": {
                "status": "ILLEGAL_CIRCULAR",
                "reason": "it consumes the token field being decoded",
            },
            "selected_frame0_output": {
                "status": "ILLEGAL_CIRCULAR",
                "reason": "render and selector application occur after token decode",
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def stage_result(store: Path) -> dict[str, object]:
    encode = json.loads((store / "ENCODE.json").read_text())
    verify = json.loads((store / "VERIFY.json").read_text())
    mi = json.loads((store / "MI.json").read_text())
    precheck = json.loads((store / "PRECHECK.json").read_text())
    if not encode["baseline_byte_identical"] or not verify["all_variants_byte_identical"]:
        raise Xs1Error("cannot report a byte delta without both identity controls")
    source_snapshot = retain_input(
        Path(__file__).resolve(),
        store / "retained/source/ddm_xs1_cross_section_conditioning.py",
    )
    runtime_snapshots = []
    for expected in precheck["runtime_source_manifest"]:
        source = Path(expected["path"])
        if file_fact(source)["sha256"] != expected["sha256"]:
            raise Xs1Error(f"runtime source drift since preflight: {source}")
        runtime_snapshots.append(
            retain_input(
                source,
                store / "retained/source/runtime_tree" / source.relative_to(RUNTIME_ROOT),
            )
        )
    baseline = encode["variants"]["baseline"]
    baseline_stream_bytes = int(baseline["stream"]["bytes"])
    baseline_archive_bytes = int(baseline["archive"]["bytes"])
    if baseline_stream_bytes != INCUMBENT_STREAM_BYTES or baseline_archive_bytes != INCUMBENT_ARCHIVE_BYTES:
        raise Xs1Error("baseline sizes drifted from DX2")
    rows: dict[str, dict[str, object]] = {}
    for name in FEATURES:
        row = encode["variants"][name]
        stream_delta = int(row["stream"]["bytes"]) - baseline_stream_bytes
        archive_delta = int(row["archive"]["bytes"]) - baseline_archive_bytes
        saving = -archive_delta
        percent = 100.0 * saving / baseline_stream_bytes
        base_seconds = float(verify["measured_model_and_decoder_seconds"]["baseline"])
        candidate_seconds = float(verify["measured_model_and_decoder_seconds"][name])
        projected_wall = DECODE_WALL_SECONDS + max(0.0, candidate_seconds - base_seconds)
        rows[name] = {
            "stream_bytes": row["stream"]["bytes"],
            "stream_delta_bytes": stream_delta,
            "archive_bytes": row["archive"]["bytes"],
            "archive_delta_bytes": archive_delta,
            "gross_saving_bytes": saving,
            "model_description_bytes": 0,
            "model_description_reason": "fixed generic receiver code; empty online tables; no stored learned table or scalar",
            "net_saving_bytes": saving,
            "percent_of_incumbent_token_stream": percent,
            "percent_of_required_42382_byte_cut": 100.0 * saving / REQUIRED_CUT_BYTES,
            "score_equivalent_at_fixed_distortion": saving * RATE_S_PER_BYTE,
            "distortion_equivalent_seg_cells": saving * RATE_S_PER_BYTE * POSITIONS / 100.0,
            "meets_5_percent_prediction": saving >= math.ceil(0.05 * INCUMBENT_STREAM_BYTES),
            "decode_cost": {
                "baseline_measured_component_seconds": base_seconds,
                "candidate_measured_component_seconds": candidate_seconds,
                "projected_full_decode_wall_seconds": projected_wall,
                "projection_basis": "498 s measured wall plus nonnegative local candidate-minus-baseline model/coder component time",
                "contest_budget_seconds": CONTEST_BUDGET_SECONDS,
            },
            "plugin_cmi": mi["sections"][name],
            "stream": row["stream"],
            "archive": row["archive"],
            "decoded_output": verify["decoded_outputs"][name],
        }
    winner_name = min(rows, key=lambda key: int(rows[key]["archive_bytes"]))
    winner = rows[winner_name]
    minimum_prediction = math.ceil(0.05 * INCUMBENT_STREAM_BYTES)
    verdict = (
        "WIN_REQUIRES_SEALED_MAIN_FIRE_ORDER"
        if int(winner["net_saving_bytes"]) >= minimum_prediction
        else "FALSIFIED_BELOW_5_PERCENT_TRIGGER"
    )
    payload = {
        "schema": "ddm_xs1.result.v1",
        "complete": True,
        "axis": "[macOS-CPU advisory / scorer-free exact lossless measurement]",
        "score_claim": False,
        "promotion_eligible": False,
        "incumbent": {
            "archive_bytes": INCUMBENT_ARCHIVE_BYTES,
            "archive_sha256": EXPECTED[ARCHIVE_SOURCE][1],
            "token_stream_bytes": INCUMBENT_STREAM_BYTES,
            "token_stream_sha256": EXPECTED[STREAM_SOURCE][1],
            "score": INCUMBENT_SCORE,
            "distortion": INCUMBENT_DISTORTION,
            "fixed_distortion_ceiling_bytes": FIXED_DISTORTION_CEILING,
            "required_cut_bytes": REQUIRED_CUT_BYTES,
        },
        "denominator_positions": POSITIONS,
        "prediction_trigger_bytes": minimum_prediction,
        "rows": rows,
        "winner": winner_name,
        "winner_net_saving_bytes": winner["net_saving_bytes"],
        "verdict": verdict,
        "main_fire_status": "DO_NOT_FIRE",
        "frontier_moved": False,
        "frontier": {
            "own_vehicle": "DX2",
            "score": INCUMBENT_SCORE,
            "archive_bytes": INCUMBENT_ARCHIVE_BYTES,
            "reason": "XS1 is lossless and no candidate was fired or exact-scored",
        },
        "measurement_source": source_snapshot,
        "runtime_source_snapshots": runtime_snapshots,
        "stage_receipts": {
            name: file_fact(store / filename)
            for name, filename in (
                ("preflight", "PRECHECK.json"),
                ("features", "FEATURES.json"),
                ("encode", "ENCODE.json"),
                ("verify", "VERIFY.json"),
                ("mi", "MI.json"),
            )
        },
    }
    atomic_json(store / "RESULT.json", payload)
    manifest = []
    for path in sorted(store.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest.append(file_fact(path))
    atomic_json(
        store / "MANIFEST.json",
        {
            "schema": "ddm_xs1.manifest.v1",
            "root": str(store),
            "files": manifest,
        },
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument(
        "--stage",
        choices=("preflight", "features", "encode", "verify", "mi", "result", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    store = require_store(Path(args.store))
    stages = {
        "preflight": lambda: stage_preflight(store),
        "features": lambda: stage_features(store),
        "encode": lambda: stage_encode(store, args.resume),
        "verify": lambda: stage_verify(store, args.resume),
        "mi": lambda: stage_mi(store),
        "result": lambda: stage_result(store),
    }
    order = ("preflight", "features", "encode", "verify", "mi", "result")
    selected = order if args.stage == "all" else (args.stage,)
    for name in selected:
        started = time.perf_counter()
        value = stages[name]()
        progress(
            stage=name,
            event="done",
            elapsed=time.perf_counter() - started,
            complete=bool(value.get("complete", False)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
