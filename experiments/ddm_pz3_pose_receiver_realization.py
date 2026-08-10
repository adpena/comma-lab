#!/usr/bin/env python3
"""Realize the retained PZ2 pose targets through a counted PR130 receiver.

This arm is scorer-slot-free.  It constructs real evaluator-runnable archives,
parses them through the public receiver, and proves selected-frame identity to
the already measured PR130 control.  It never substitutes target quantization
MSE for realized PoseNet error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import lzma
import math
import os
import platform
import struct
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "src/tac/pr130_runtime/fx1_runtime_tree"
BASE_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip"
)
TARGET_PACKET = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz2_pose_representation_20260810_v3/retained/"
    "candidates/direct/direct_p092_b12-6-6-5-5-5.pz2"
)
SELECTION_RECEIPT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pk2_20260809/checkpoints/"
    "masters_n120_seed20260809.json"
)
PK2_SCORE_RECEIPT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pk2_20260809/"
    "score_receipt_n120_seed20260809.json"
)
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_pz3_20260810")

EXPECTED_BASE_ARCHIVE_SHA256 = (
    "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
)
EXPECTED_TARGET_PACKET_SHA256 = (
    "f332b3f6a52cdb4661baf35d6767d1cb6325d422d9c91ebcd5d061794174e668"
)
EXPECTED_SELECTION_RECEIPT_SHA256 = (
    "2e2778fd65e69c2af3ddcd1bff1bed3db3737a54743df89393e1e8f673a90f99"
)
EXPECTED_TARGET_PACKET_BYTES = 2_860
EXPECTED_BASE_ARCHIVE_BYTES = 191_052
EXPECTED_INTAKE_HEAD = "e34f31bc4969042c0051ac81aa3c56884419a231"
INTAKE = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo")

N = 600
DIM = 12
BASIS_SHAPE = (DIM, 3, 24, 32)
CAMERA_H = 874
CAMERA_W = 1164
REFERENCE_BYTES = 37_545_489
BASE_D_SEG = 0.00029660
BASE_D_POSE = 0.00002331
BASE_SCORE = 0.17214129749189644
FALSIFIER_SCORE = 0.16110432236983460
AXIS = "[macOS-CPU advisory]"
DERIVED_AXIS = (
    "[DERIVED same-output rate action over contest-CUDA,DALI,n600 base components; "
    "no new exact eval]"
)
XZ_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 16,
        "lc": 0,
        "lp": 1,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


def utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_output(path: Path) -> Path:
    resolved = path.resolve()
    root = Path("/Volumes/VertigoDataTier/pact")
    if resolved == root or root not in resolved.parents:
        raise ValueError("PZ3 persisted output must use an arm-specific VertigoDataTier path")
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise ValueError("PZ3 persisted evidence may not use /tmp")
    return resolved


def storage_preflight(path: Path, required_bytes: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(path)
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    receipt = {
        "path": str(path),
        "required_free_bytes": required_bytes,
        "free_bytes": free_bytes,
        "passed": free_bytes >= required_bytes,
        "measured_at_utc": utcnow(),
    }
    if not receipt["passed"]:
        raise RuntimeError(
            f"storage preflight failed at {path}: {free_bytes} < {required_bytes}"
        )
    return receipt


def setup_runtime() -> tuple[Any, Any, Any]:
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))
    codec = importlib.import_module("carrier_codec")
    receiver = importlib.import_module("pose_target_receiver")
    inflate = importlib.import_module("inflate")
    return codec, receiver, inflate


def deterministic_archive(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def extract_bundle(archive_blob: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(archive_blob), "r") as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("PR130 archive must contain exactly the p member")
        member = archive.read("p")
    model_bytes = struct.unpack_from("<I", member)[0]
    models_compressed = member[4 : 4 + model_bytes]
    tokens = member[4 + model_bytes :]
    models_raw = lzma.decompress(models_compressed)
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_start = 8
    carrier_start = semantic_start + semantic_bytes
    hpac_start = carrier_start + carrier_bytes
    if hpac_start > len(models_raw):
        raise ValueError("PR130 model sections exceed the raw model bundle")
    return {
        "member": member,
        "models_raw": models_raw,
        "semantic": models_raw[semantic_start:carrier_start],
        "carrier": models_raw[carrier_start:hpac_start],
        "hpac": models_raw[hpac_start:],
        "tokens": tokens,
    }


def replace_carrier(bundle: dict[str, bytes], carrier: bytes) -> bytes:
    raw = (
        struct.pack("<II", len(bundle["semantic"]), len(carrier))
        + bundle["semantic"]
        + carrier
        + bundle["hpac"]
    )
    models = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=XZ_FILTERS)
    member = struct.pack("<I", len(models)) + models + bundle["tokens"]
    return deterministic_archive(member)


def split_cpr1(
    blob: bytes, codec: Any
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basis_scales, basis_codes, coefficient_scales, encoded = (
        codec.decode_compact_carrier(
            blob,
            basis_count=math.prod(BASIS_SHAPE),
            frames=N,
            dimensions=DIM,
        )
    )
    magic, basis_bits, coefficient_bits = codec.HEADER.unpack_from(blob)
    if magic != codec.MAGIC:
        raise ValueError("base carrier is not CPR1")
    cursor = codec.HEADER.size
    scale_bytes = DIM * 4
    saved_basis_scales = blob[cursor : cursor + scale_bytes]
    cursor += scale_bytes
    cursor += scale_bytes
    lengths = blob[cursor : cursor + codec.ALPHABET_SIZE]
    cursor += codec.ALPHABET_SIZE
    cursor += DIM
    basis_payload_bytes = (basis_bits + 7) // 8
    basis_payload = blob[cursor : cursor + basis_payload_bytes]
    cursor += basis_payload_bytes
    if len(blob) - cursor != (coefficient_bits + 7) // 8:
        raise RuntimeError("CPR1 component boundary mismatch")
    basis_component = (
        struct.pack("<I", basis_bits)
        + saved_basis_scales
        + lengths
        + basis_payload
    )
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    unsigned = np.cumsum(delta, axis=0) & 0xFFF
    absolute = np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(
        np.int32
    )
    basis = basis_codes.reshape(BASIS_SHAPE).astype(np.float32)
    basis *= basis_scales[:, None, None, None]
    coefficients = absolute.astype(np.float32) * coefficient_scales[None]
    return basis_component, basis, coefficients, coefficient_scales, absolute


def fit_predictor(
    receiver: Any,
    target_codes: np.ndarray,
    absolute: np.ndarray,
    feature_mode: int,
    shift: int,
) -> Any:
    features = receiver.feature_matrix(target_codes, feature_mode, absolute)
    feature_offsets = np.rint(features.mean(axis=0)).astype(np.int64)
    output_offsets = np.rint(absolute.mean(axis=0)).astype(np.int64)
    centered_features = features.astype(np.float64) - feature_offsets[None]
    centered_outputs = absolute.astype(np.float64) - output_offsets[None]
    solution, _, _, _ = np.linalg.lstsq(
        centered_features,
        centered_outputs,
        rcond=None,
    )
    scaled = np.rint(solution * float(1 << shift))
    if np.any(scaled < np.iinfo(np.int32).min) or np.any(
        scaled > np.iinfo(np.int32).max
    ):
        raise ValueError("fitted PZ3R predictor weights exceed int32")
    predictor = receiver.Predictor(
        feature_mode=feature_mode,
        shift=shift,
        feature_offsets=feature_offsets.astype(np.int32),
        output_offsets=output_offsets.astype(np.int32),
        weights=scaled.astype(np.int32),
    )
    # Exercise the exact receiver arithmetic before any payload is launched.
    receiver.predict_coefficients(target_codes, predictor, absolute)
    return predictor


def render_selected_slaves(
    path: Path,
    inflate: Any,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    indices: list[int],
) -> dict[str, Any]:
    temporary = path.with_suffix(path.suffix + ".pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(len(indices), 3, CAMERA_H, CAMERA_W),
    )
    normalized = inflate.normalized_basis(basis)
    for start in range(0, len(indices), 8):
        current = indices[start : start + 8]
        selected = coefficients[current]
        carrier = torch.einsum("bk,kchw->bchw", selected, normalized)
        carrier = carrier / math.sqrt(DIM)
        slave_eval = (
            127.5 + inflate.CARRIER_AMPLITUDE * carrier
        ).clamp(0.0, 255.0).round()
        slaves = (
            F.interpolate(
                slave_eval,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        output[start : start + len(current)] = slaves
    output.flush()
    del output
    os.replace(temporary, path)
    array = np.load(path, mmap_mode="r")
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "tensor_sha256": array_sha256(array),
    }
    del array
    return result


def score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, float]:
    seg = 100.0 * d_seg
    pose = math.sqrt(10.0 * d_pose)
    rate = 25.0 * archive_bytes / REFERENCE_BYTES
    return {"seg": seg, "pose": pose, "rate": rate, "total": seg + pose + rate}


def validate_pins() -> dict[str, Any]:
    pins = {
        "base_archive": artifact(BASE_ARCHIVE, "pinned PR130 base archive"),
        "target_packet": artifact(TARGET_PACKET, "pinned PZ2 direct target packet"),
        "selection_receipt": artifact(
            SELECTION_RECEIPT, "pinned stratified n120 selection receipt"
        ),
        "pk2_score_receipt": artifact(
            PK2_SCORE_RECEIPT, "existing measured PR130 n120 scorer receipt"
        ),
    }
    if pins["base_archive"]["sha256"] != EXPECTED_BASE_ARCHIVE_SHA256:
        raise RuntimeError("base archive SHA-256 drift")
    if pins["base_archive"]["bytes"] != EXPECTED_BASE_ARCHIVE_BYTES:
        raise RuntimeError("base archive byte count drift")
    if pins["target_packet"]["sha256"] != EXPECTED_TARGET_PACKET_SHA256:
        raise RuntimeError("target packet SHA-256 drift")
    if pins["target_packet"]["bytes"] != EXPECTED_TARGET_PACKET_BYTES:
        raise RuntimeError("target packet byte count drift")
    if pins["selection_receipt"]["sha256"] != EXPECTED_SELECTION_RECEIPT_SHA256:
        raise RuntimeError("selection receipt SHA-256 drift")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=INTAKE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != EXPECTED_INTAKE_HEAD:
        raise RuntimeError("PR130 intake head drift")
    pins["intake_head"] = head
    return pins


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--required-free-bytes", type=int, default=2 << 30)
    args = parser.parse_args()

    output = validate_output(args.output)
    retained = output / "retained"
    checkpoints = output / "checkpoints"
    state_path = args.resume_from or checkpoints / "state.json"
    if state_path.resolve() != (checkpoints / "state.json").resolve():
        raise ValueError("--resume-from must be the canonical checkpoint in --output")
    output.mkdir(parents=True, exist_ok=True)
    retained.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"schema": "ddm_pz3_resume.v1", "created_at_utc": utcnow()}
    )
    codec, receiver, inflate = setup_runtime()

    if not state.get("preflight_complete"):
        preflight = storage_preflight(output, args.required_free_bytes)
        pins = validate_pins()
        state.update(
            {
                "preflight_complete": True,
                "storage_preflight": preflight,
                "pins": pins,
                "source": {
                    "git_head": subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=REPO,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip(),
                    "runtime_files": {
                        "carrier_codec.py": sha256_file(RUNTIME / "carrier_codec.py"),
                        "inflate.py": sha256_file(RUNTIME / "inflate.py"),
                        "pose_target_receiver.py": sha256_file(
                            RUNTIME / "pose_target_receiver.py"
                        ),
                    },
                },
                "platform": {
                    "python": platform.python_version(),
                    "system": platform.platform(),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                },
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_preflight_complete.json", state)

    if not state.get("materialize_complete"):
        base_blob = BASE_ARCHIVE.read_bytes()
        target_packet = TARGET_PACKET.read_bytes()
        bundle = extract_bundle(base_blob)
        basis_component, base_basis, base_coefficients, coefficient_scales, absolute = (
            split_cpr1(bundle["carrier"], codec)
        )
        target_codes, decoded_targets = receiver.decode_pz2_packet(target_packet)
        shared_dir = retained / "shared"
        atomic_bytes(shared_dir / "base_carrier.cpr1", bundle["carrier"])
        atomic_bytes(shared_dir / "basis_component.bin", basis_component)
        atomic_bytes(shared_dir / "target_packet.pz2", target_packet)
        atomic_npz(
            shared_dir / "base_arrays.npz",
            basis=base_basis,
            coefficients=base_coefficients,
            absolute_coefficients=absolute,
            coefficient_scales=coefficient_scales,
            target_codes=target_codes,
            decoded_targets=decoded_targets,
        )

        modes = {
            "target": receiver.FEATURE_TARGET,
            "target_quadratic": receiver.FEATURE_TARGET_QUADRATIC,
            "target_previous": receiver.FEATURE_TARGET_PREVIOUS,
            "target_quadratic_previous": receiver.FEATURE_TARGET_QUADRATIC_PREVIOUS,
        }
        candidates: list[dict[str, Any]] = []
        for name, feature_mode in modes.items():
            for shift in (12, 16, 20):
                candidate_name = f"{name}_q{shift}"
                candidate_dir = retained / "candidates" / candidate_name
                predictor = fit_predictor(
                    receiver,
                    target_codes,
                    absolute,
                    feature_mode,
                    shift,
                )
                carrier = receiver.encode_pose_target_carrier(
                    basis_component=basis_component,
                    target_packet=target_packet,
                    predictor=predictor,
                    coefficient_scales=coefficient_scales,
                    absolute_coefficients=absolute,
                )
                repeat_carrier = receiver.encode_pose_target_carrier(
                    basis_component=basis_component,
                    target_packet=target_packet,
                    predictor=predictor,
                    coefficient_scales=coefficient_scales,
                    absolute_coefficients=absolute,
                )
                if carrier != repeat_carrier:
                    raise RuntimeError(f"{candidate_name}: nondeterministic carrier")
                archive = replace_carrier(bundle, carrier)
                repeat_archive = replace_carrier(bundle, repeat_carrier)
                if archive != repeat_archive:
                    raise RuntimeError(f"{candidate_name}: nondeterministic archive")
                parsed_basis, parsed_coefficients = receiver.decode_pose_target_carrier(
                    carrier
                )
                if not np.array_equal(parsed_basis, base_basis):
                    raise RuntimeError(f"{candidate_name}: basis parse-back mismatch")
                if not np.array_equal(parsed_coefficients, base_coefficients):
                    raise RuntimeError(f"{candidate_name}: coefficient parse-back mismatch")

                atomic_bytes(candidate_dir / "carrier.pz3r", carrier)
                atomic_bytes(candidate_dir / "carrier.repeat.pz3r", repeat_carrier)
                atomic_bytes(candidate_dir / "archive.zip", archive)
                atomic_bytes(candidate_dir / "archive.repeat.zip", repeat_archive)
                atomic_npz(
                    candidate_dir / "parsed_arrays.npz",
                    basis=parsed_basis,
                    coefficients=parsed_coefficients,
                )
                candidates.append(
                    {
                        "name": candidate_name,
                        "feature_mode": feature_mode,
                        "shift": shift,
                        "carrier": artifact(
                            candidate_dir / "carrier.pz3r", "counted PZ3R carrier"
                        ),
                        "carrier_repeat": artifact(
                            candidate_dir / "carrier.repeat.pz3r",
                            "determinism repeat PZ3R carrier",
                        ),
                        "archive": artifact(
                            candidate_dir / "archive.zip", "byte-closed candidate archive"
                        ),
                        "archive_repeat": artifact(
                            candidate_dir / "archive.repeat.zip",
                            "determinism repeat candidate archive",
                        ),
                        "parsed_arrays": artifact(
                            candidate_dir / "parsed_arrays.npz",
                            "receiver parse-back arrays",
                        ),
                        "basis_exact": True,
                        "coefficients_exact": True,
                    }
                )
                print(
                    f"materialized {candidate_name}: carrier={len(carrier)} "
                    f"archive={len(archive)}",
                    flush=True,
                )
        selected = min(candidates, key=lambda row: row["archive"]["bytes"])
        materialize_receipt = {
            "schema": "ddm_pz3_materialize.v1",
            "created_at_utc": utcnow(),
            "axis": "[macOS-CPU byte-closure; scorer-free]",
            "score_claim": False,
            "base_archive": artifact(BASE_ARCHIVE, "pinned PR130 base archive"),
            "shared_payloads": [
                artifact(shared_dir / "base_carrier.cpr1", "retained base carrier"),
                artifact(shared_dir / "basis_component.bin", "retained CPR1 basis component"),
                artifact(shared_dir / "target_packet.pz2", "retained exact PZ2 packet"),
                artifact(shared_dir / "base_arrays.npz", "retained decoded source arrays"),
            ],
            "candidates": candidates,
            "selected": selected["name"],
        }
        materialize_path = output / "materialize_receipt.json"
        atomic_json(materialize_path, materialize_receipt)
        state.update(
            {
                "materialize_complete": True,
                "materialize_receipt": str(materialize_path),
                "materialize_receipt_sha256": sha256_file(materialize_path),
                "selected": selected["name"],
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_materialize_complete.json", state)

    if not state.get("verify_complete"):
        materialize_path = Path(state["materialize_receipt"])
        if sha256_file(materialize_path) != state["materialize_receipt_sha256"]:
            raise RuntimeError("materialize receipt hash drift")
        materialized = json.loads(materialize_path.read_text(encoding="utf-8"))
        selected = next(
            row for row in materialized["candidates"] if row["name"] == state["selected"]
        )
        base_blob = BASE_ARCHIVE.read_bytes()
        base_bundle = extract_bundle(base_blob)
        selected_blob = Path(selected["archive"]["path"]).read_bytes()
        selected_bundle = extract_bundle(selected_blob)
        if selected_bundle["semantic"] != base_bundle["semantic"]:
            raise RuntimeError("selected archive changed semantic bytes")
        if selected_bundle["hpac"] != base_bundle["hpac"]:
            raise RuntimeError("selected archive changed HPAC bytes")
        if selected_bundle["tokens"] != base_bundle["tokens"]:
            raise RuntimeError("selected archive changed token bytes")

        base_semantic_pose = (
            struct.pack(
                "<II", len(base_bundle["semantic"]), len(base_bundle["carrier"])
            )
            + base_bundle["semantic"]
            + base_bundle["carrier"]
        )
        selected_semantic_pose = (
            struct.pack(
                "<II",
                len(selected_bundle["semantic"]),
                len(selected_bundle["carrier"]),
            )
            + selected_bundle["semantic"]
            + selected_bundle["carrier"]
        )
        _, base_basis, base_coefficients = inflate.unpack_semantic_pose(
            base_semantic_pose
        )
        _, selected_basis, selected_coefficients = inflate.unpack_semantic_pose(
            selected_semantic_pose
        )
        if not torch.equal(base_basis, selected_basis):
            raise RuntimeError("public receiver basis differs from CPR1 control")
        if not torch.equal(base_coefficients, selected_coefficients):
            raise RuntimeError("public receiver coefficients differ from CPR1 control")

        selection = json.loads(SELECTION_RECEIPT.read_text(encoding="utf-8"))
        indices = [int(value) for value in selection["selection_indices"]]
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(20260809)
        np.random.seed(20260809)
        rendered_dir = retained / "rendered"
        base_frames_path = rendered_dir / "baseline_cpr1_slaves_n120.npy"
        selected_frames_path = rendered_dir / f"{state['selected']}_slaves_n120.npy"
        base_render = render_selected_slaves(
            base_frames_path,
            inflate,
            base_basis,
            base_coefficients,
            indices,
        )
        selected_render = render_selected_slaves(
            selected_frames_path,
            inflate,
            selected_basis,
            selected_coefficients,
            indices,
        )
        base_frames = np.load(base_frames_path, mmap_mode="r")
        selected_frames = np.load(selected_frames_path, mmap_mode="r")
        frame_identity = bool(np.array_equal(base_frames, selected_frames))
        del base_frames, selected_frames
        if not frame_identity or base_render["tensor_sha256"] != selected_render["tensor_sha256"]:
            raise RuntimeError("PZ3R rendered frames differ from CPR1 control")

        # The packet is causal, not decorative: changing a target code changes
        # the selected predictor and the unchanged residual then fails integrity.
        target_packet = TARGET_PACKET.read_bytes()
        target_codes, _ = receiver.decode_pz2_packet(target_packet)
        candidate_carrier = Path(selected["carrier"]["path"]).read_bytes()
        fields = receiver.HEADER.unpack_from(candidate_carrier)
        basis_bytes, target_bytes, model_bytes = fields[5:8]
        model_start = receiver.HEADER.size + basis_bytes + target_bytes
        predictor, _ = receiver.deserialize_predictor(
            candidate_carrier[model_start : model_start + model_bytes]
        )
        _, _, _, _, absolute = split_cpr1(base_bundle["carrier"], codec)
        prediction = receiver.predict_coefficients(target_codes, predictor, absolute)
        mutated_codes = target_codes.copy()
        mutated_codes[0, 0] ^= 1
        mutated_prediction = receiver.predict_coefficients(
            mutated_codes, predictor, absolute
        )
        prediction_changed = bool(np.any(prediction != mutated_prediction))
        if not prediction_changed:
            raise RuntimeError("selected PZ3R predictor does not consume target code 0")

        pk2 = json.loads(PK2_SCORE_RECEIPT.read_text(encoding="utf-8"))
        pk2_base = pk2["baseline"]
        selected_bytes = selected["archive"]["bytes"]
        advisory_terms = score_from_components(
            pk2_base["d_seg"], pk2_base["d_pose"], selected_bytes
        )
        derived_terms = score_from_components(BASE_D_SEG, BASE_D_POSE, selected_bytes)
        base_terms = score_from_components(BASE_D_SEG, BASE_D_POSE, EXPECTED_BASE_ARCHIVE_BYTES)
        if not math.isclose(base_terms["total"], BASE_SCORE, rel_tol=0.0, abs_tol=5e-16):
            raise RuntimeError("canonical base score recompute drift")
        verdict = (
            "REALIZATION-LIMITED"
            if derived_terms["total"] >= FALSIFIER_SCORE
            else "REALIZED-CANDIDATE"
        )
        verify_receipt = {
            "schema": "ddm_pz3_verify.v1",
            "created_at_utc": utcnow(),
            "score_claim": False,
            "selected": selected,
            "public_receiver": {
                "base_basis_tensor_sha256": sha256_bytes(
                    base_basis.contiguous().numpy().tobytes()
                ),
                "selected_basis_tensor_sha256": sha256_bytes(
                    selected_basis.contiguous().numpy().tobytes()
                ),
                "base_coefficient_tensor_sha256": sha256_bytes(
                    base_coefficients.contiguous().numpy().tobytes()
                ),
                "selected_coefficient_tensor_sha256": sha256_bytes(
                    selected_coefficients.contiguous().numpy().tobytes()
                ),
                "basis_byte_identical": True,
                "coefficients_byte_identical": True,
                "target_prediction_changed_on_code_mutation": prediction_changed,
                "scorer_weights_in_receiver": False,
            },
            "rendered_positive_control": {
                "selection_seed": 20260809,
                "selection_n": len(indices),
                "selection_receipt": artifact(
                    SELECTION_RECEIPT, "stratified n120 selection receipt"
                ),
                "baseline_frames": {
                    **artifact(
                        base_frames_path, "retained CPR1 control slave frames n120"
                    ),
                    **base_render,
                },
                "selected_frames": {
                    **artifact(
                        selected_frames_path, "retained PZ3R slave frames n120"
                    ),
                    **selected_render,
                },
                "frame_byte_identical": frame_identity,
            },
            "inherited_n120_row": {
                "axis": (
                    "[macOS-CPU advisory; exact-output identity to measured PK2 n120 control]"
                ),
                "score_claim": False,
                "d_seg": pk2_base["d_seg"],
                "d_pose": pk2_base["d_pose"],
                "archive_bytes": selected_bytes,
                "score_terms": advisory_terms,
                "provenance_receipt": artifact(
                    PK2_SCORE_RECEIPT, "existing measured PR130 n120 scorer receipt"
                ),
                "new_scorer_run": False,
            },
            "derived_n600_rate_action": {
                "axis": DERIVED_AXIS,
                "score_claim": False,
                "d_seg": BASE_D_SEG,
                "d_pose": BASE_D_POSE,
                "archive_bytes": selected_bytes,
                "score_terms": derived_terms,
                "base_score_terms": base_terms,
                "delta_archive_bytes": selected_bytes - EXPECTED_BASE_ARCHIVE_BYTES,
                "delta_score": derived_terms["total"] - base_terms["total"],
                "not_measured": "no new n600 scorer and no contest CPU/CUDA evaluation",
            },
            "falsifier": {
                "threshold": FALSIFIER_SCORE,
                "realized_derived_score": derived_terms["total"],
                "verdict": verdict,
                "verdict_scope": "FORMULATION: exact residual realization of the PZ2 target packet through the frozen PR130 carrier basis",
            },
        }
        verify_path = output / "verify_receipt.json"
        atomic_json(verify_path, verify_receipt)
        state.update(
            {
                "verify_complete": True,
                "verify_receipt": str(verify_path),
                "verify_receipt_sha256": sha256_file(verify_path),
                "verdict": verdict,
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_verify_complete.json", state)

    if not state.get("finalize_complete"):
        materialize = json.loads(
            Path(state["materialize_receipt"]).read_text(encoding="utf-8")
        )
        verify = json.loads(Path(state["verify_receipt"]).read_text(encoding="utf-8"))
        payloads = list(materialize["shared_payloads"])
        for candidate in materialize["candidates"]:
            payloads.extend(
                [
                    candidate["carrier"],
                    candidate["carrier_repeat"],
                    candidate["archive"],
                    candidate["archive_repeat"],
                    candidate["parsed_arrays"],
                ]
            )
        payloads.extend(
            [
                verify["rendered_positive_control"]["baseline_frames"],
                verify["rendered_positive_control"]["selected_frames"],
            ]
        )
        for entry in payloads:
            path = Path(entry["path"])
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"retained payload drift: {path}")
        final = {
            "schema": "ddm_pz3_pose_receiver_realization.v1",
            "created_at_utc": utcnow(),
            "axis": AXIS,
            "score_claim": False,
            "verdict": state["verdict"],
            "selected": verify["selected"],
            "public_receiver": verify["public_receiver"],
            "rendered_positive_control": verify["rendered_positive_control"],
            "inherited_n120_row": verify["inherited_n120_row"],
            "derived_n600_rate_action": verify["derived_n600_rate_action"],
            "falsifier": verify["falsifier"],
            "retained_payloads": payloads,
            "retained_payload_count": len(payloads),
            "materialize_receipt": artifact(
                Path(state["materialize_receipt"]), "materialization receipt"
            ),
            "verify_receipt": artifact(Path(state["verify_receipt"]), "verification receipt"),
            "boundaries": {
                "new_pose_scorer_run": False,
                "full_n600_scorer_run": False,
                "contest_cpu_cuda_eval": False,
                "modal_or_paid_dispatch": False,
                "upstream_modified": False,
                "intake_modified": False,
            },
            "borrowed_substrate_accounting": {
                "borrowed": "PR130 semantic renderer, frozen carrier basis, coefficient scales, HPAC, token stream, CPR1 entropy primitives, and public frame renderer",
                "pz3_original": "PZ2 packet decoder, counted fixed-point target-conditioned predictor, exact residual packet, PZ3R public receiver dispatch, and byte-closed receiver proof",
            },
            "resumability": {
                "resume_argument": "--resume-from /Volumes/VertigoDataTier/pact/ddm_pz3_20260810/checkpoints/state.json",
                "stage_checkpoints": [
                    str(checkpoints / "stage_preflight_complete.json"),
                    str(checkpoints / "stage_materialize_complete.json"),
                    str(checkpoints / "stage_verify_complete.json"),
                    str(checkpoints / "stage_finalize_complete.json"),
                ],
            },
        }
        final_path = output / "PZ3_RESULT.json"
        atomic_json(final_path, final)
        state.update(
            {
                "finalize_complete": True,
                "final_result": str(final_path),
                "final_result_sha256": sha256_file(final_path),
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_finalize_complete.json", state)

    closure_sources = (
        json.loads(Path(state["runtime_closure_receipt"]).read_text(encoding="utf-8"))[
            "source_files"
        ]
        if state.get("runtime_closure_complete")
        and Path(state.get("runtime_closure_receipt", "")).is_file()
        else {}
    )
    closure_stale = any(
        closure_sources.get(name, {}).get("sha256") != sha256_file(RUNTIME / name)
        for name in (
            "inflate.py",
            "inflate.sh",
            "pose_target_receiver.py",
            "runtime-dependencies.json",
        )
    )
    if not state.get("runtime_closure_complete") or closure_stale:
        final_path = Path(state["final_result"])
        final = json.loads(final_path.read_text(encoding="utf-8"))
        selected_archive = Path(final["selected"]["archive"]["path"])
        with zipfile.ZipFile(selected_archive, "r") as archive:
            if archive.namelist() != ["p"]:
                raise RuntimeError("selected archive runtime closure found unexpected members")
            member = archive.read("p")
        closure_dir = retained / "runtime_closure"
        member_path = closure_dir / "p"
        atomic_bytes(member_path, member)
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHON": sys.executable,
                "PR130_DEPENDENCY_SELECTION_ONLY": "1",
            }
        )
        dependency_selection = subprocess.run(
            [
                str(RUNTIME / "inflate.sh"),
                str(closure_dir),
                "0",
                str(closure_dir / "unused.raw"),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        expected_selection = (
            "PR130_DEPENDENCY_SELECTION model_codec=legacy_lzma needs_brotli=1"
        )
        if dependency_selection != expected_selection:
            raise RuntimeError(
                f"PZ3R runtime dependency selection mismatch: {dependency_selection}"
            )
        closure_receipt = {
            "schema": "ddm_pz3_runtime_closure.v1",
            "created_at_utc": utcnow(),
            "axis": "[macOS-CPU receiver dependency-selection proof]",
            "score_claim": False,
            "archive": artifact(selected_archive, "selected byte-closed archive"),
            "extracted_member": artifact(
                member_path, "retained p member used by public runtime entrypoint"
            ),
            "dependency_selection": dependency_selection,
            "source_files": {
                name: artifact(RUNTIME / name, "public receiver runtime source")
                for name in (
                    "inflate.py",
                    "inflate.sh",
                    "pose_target_receiver.py",
                    "runtime-dependencies.json",
                )
            },
            "brotli_for_pz3r": True,
            "full_linux_or_contest_replay": False,
        }
        closure_path = output / "runtime_closure_receipt.json"
        atomic_json(closure_path, closure_receipt)
        final["runtime_closure"] = {
            **closure_receipt,
            "receipt": artifact(closure_path, "runtime dependency closure receipt"),
        }
        if not any(
            entry["path"] == closure_receipt["extracted_member"]["path"]
            for entry in final["retained_payloads"]
        ):
            final["retained_payloads"].append(closure_receipt["extracted_member"])
        final["retained_payload_count"] = len(final["retained_payloads"])
        closure_checkpoint = str(checkpoints / "stage_runtime_closure_complete.json")
        if closure_checkpoint not in final["resumability"]["stage_checkpoints"]:
            final["resumability"]["stage_checkpoints"].append(closure_checkpoint)
        atomic_json(final_path, final)
        state.update(
            {
                "runtime_closure_complete": True,
                "runtime_closure_receipt": str(closure_path),
                "runtime_closure_receipt_sha256": sha256_file(closure_path),
                "final_result_sha256": sha256_file(final_path),
            }
        )
        atomic_json(state_path, state)
        atomic_json(checkpoints / "stage_runtime_closure_complete.json", state)

    print(
        json.dumps(
            {
                "result": state["final_result"],
                "sha256": state["final_result_sha256"],
                "selected": state["selected"],
                "verdict": state["verdict"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
