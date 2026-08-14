#!/usr/bin/env python3
"""PK3 scorer-free build of a joint frame-0 pose representation on CP135.

This bounded arm fits a temporal control lattice through previously retained
PoseNet Jacobians, compiles it into the real CP135 receiver, retains every
materialized payload, and seals one exact dual-axis job.  The Jacobian sample is
an explicitly non-authoritative toy bracket: this runner never invokes a scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_cp135_rate_compose as cp135
from experiments import ddm_jo1_joint_probability_object as jo1
from experiments import ddm_pk3_frame0_pose_overlay_runtime as overlay_codec
from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs2_compensation_rate_rung as qs2

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pk3_20260813")
CONSUMER_STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_eu4_pose1000_joint_20260813"
)
QS1_STORE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qs1_20260813")
CP135_ARCHIVE: Final = qs1.CP135_ARCHIVE
CP135_BYTES: Final = 186_252
CP135_SHA256: Final = qs1.CP135_ARCHIVE_SHA256
BROTLI: Final = Path("/opt/homebrew/bin/brotli")
SPLIT_HEADER: Final = struct.Struct("<HHH")
RATE_S_PER_BYTE: Final = 25.0 / 37_545_489
TARGET_DPOSE: Final = 3.44e-6
MAX_DELTA_BYTES: Final = 1_000
STORAGE_EXPECTED_BYTES: Final = 4 * 1024**3
STORAGE_RESERVE_BYTES: Final = 8 * 1024**3
AXIS: Final = "[macOS-CPU scorer-free exact bytes/receiver; n9 TOY-BRACKET model]"
KNOTS: Final = (2, 4, 6, 8, 12, 16)
RIDGES: Final = (1e-8, 1e-6, 1e-4)
GAINS: Final = (0.25, 0.5, 1.0)


class PK3Error(RuntimeError):
    """A source pin, retained payload, fit, or receiver gate failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return qs1.file_record(path)


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())
    return qs1.file_record(path)


def require_record(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file() or qs1.file_record(path) != record:
        raise PK3Error(f"retained {label} failed custody: {path}")
    return path


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    retained = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    required = max(0, STORAGE_EXPECTED_BYTES - retained) + STORAGE_RESERVE_BYTES
    free = shutil.disk_usage(output).free
    result = {
        "schema": "ddm_pk3_storage_preflight.v1",
        "tier": str(output.resolve()),
        "already_retained_bytes": retained,
        "expected_total_bytes": STORAGE_EXPECTED_BYTES,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "free_bytes": free,
        "passed": free >= required,
        "cleanup_policy": "certify-or-block; no payload deletion",
    }
    atomic_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise PK3Error(f"SSD storage preflight failed: free={free}, required={required}")
    return result


def source_preflight(output: Path) -> dict[str, Any]:
    sources = {
        "cp135_archive": qs1.require_file(
            CP135_ARCHIVE,
            expected_bytes=CP135_BYTES,
            expected_sha256=CP135_SHA256,
        ),
        "cp135_base_pose": qs1.require_file(qs1.CP135_BASE_POSE),
        "gt_pose": qs1.require_file(qs1.GT_POSE),
        "cp135_basis": qs1.require_file(qs1.CP135_BASIS),
        "cp135_coefficients": qs1.require_file(qs1.CP135_COEFFICIENTS),
        "brotli": qs1.require_file(BROTLI),
        "runner": qs1.require_file(Path(__file__).resolve()),
        "overlay_runtime": qs1.require_file(
            REPO / "experiments/ddm_pk3_frame0_pose_overlay_runtime.py"
        ),
        "dispatcher": qs1.require_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker": qs1.require_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
    }
    result = {
        "schema": "ddm_pk3_source_preflight.v1",
        "sources": sources,
        "seed": 135,
        "axis": AXIS,
        "owns_scorer": False,
        "scorer_calls": 0,
        "resume_from": str(output.resolve()),
        "passed": True,
    }
    atomic_json(output / "checkpoints/stage_00_source_preflight.json", result)
    return result


def _zip_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["p"] or archive.testzip() is not None:
            raise PK3Error("archive member census or CRC differs")
        return archive.read("p")


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def _brotli(payload: bytes, *arguments: str) -> bytes:
    completed = subprocess.run(
        [str(BROTLI), *arguments],
        input=payload,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise PK3Error(completed.stderr.decode(errors="replace"))
    return completed.stdout


def rate_sources() -> tuple[bytes, bytes, bytes, bytes]:
    member = _zip_member(CP135_ARCHIVE)
    streams, suffix = qs2._split_member(member)
    base_carrier = _brotli(streams[2], "-d", "-c")
    if len(base_carrier) != 22_183:
        raise PK3Error(f"CP135 packed carrier length differs: {len(base_carrier)}")
    return streams[0], streams[1], base_carrier, suffix


def temporal_weights(pair: int, knots: int) -> np.ndarray:
    if not 0 <= pair < overlay_codec.PAIR_COUNT or not 2 <= knots <= 64:
        raise PK3Error("temporal weight domain differs")
    position = pair * (knots - 1) / (overlay_codec.PAIR_COUNT - 1)
    left = min(int(position), knots - 2)
    fraction = position - left
    weights = np.zeros(knots, dtype=np.float64)
    weights[left] = 1.0 - fraction
    weights[left + 1] = fraction
    return weights


def retained_jacobians() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for result_path in sorted((QS1_STORE / "retained/proposals").glob("*/RESULT.json")):
        row = json.loads(result_path.read_text())
        if row.get("schema") != "ddm_qs1_schur_pair_result.v1":
            continue
        jacobian = result_path.parent / "stage_30_jacobian/J_POSE0.float64.npy"
        if not jacobian.is_file():
            raise PK3Error(f"retained QS1 Jacobian missing: {jacobian}")
        grouped[int(row["pair"])].append(jacobian)
    pairs = np.asarray(sorted(grouped), dtype=np.int16)
    if pairs.tolist() != [17, 105, 133, 176, 178, 510, 517, 523, 532]:
        raise PK3Error(f"retained unique-pair census differs: {pairs.tolist()}")
    jacobians = []
    records = []
    for pair in pairs:
        paths = grouped[int(pair)]
        values = [np.load(path, allow_pickle=False) for path in paths]
        if any(value.shape != (6, 12) for value in values):
            raise PK3Error(f"Jacobian geometry differs for pair {pair}")
        jacobians.append(np.mean(np.stack(values), axis=0))
        records.append(
            {
                "pair": int(pair),
                "source_denominator": len(paths),
                "sources": [qs1.file_record(path) for path in paths],
                "aggregation": "arithmetic mean across retained JS6 frame-1 partners",
            }
        )
    return pairs, np.stack(jacobians), records


def design_matrix(pairs: np.ndarray, jacobians: np.ndarray, knots: int) -> np.ndarray:
    blocks = []
    for pair, jacobian in zip(pairs, jacobians, strict=True):
        weights = temporal_weights(int(pair), knots)
        blocks.append(np.concatenate([weight * jacobian for weight in weights], axis=1))
    return np.concatenate(blocks, axis=0)


def fit_controls(
    pairs: np.ndarray,
    jacobians: np.ndarray,
    errors: np.ndarray,
    knots: int,
    ridge: float,
    gain: float,
) -> np.ndarray:
    matrix = design_matrix(pairs, jacobians, knots)
    target = -np.asarray(errors, dtype=np.float64).reshape(-1)
    augmented = np.concatenate(
        (matrix, np.sqrt(ridge) * np.eye(knots * overlay_codec.DIMENSIONS)), axis=0
    )
    rhs = np.concatenate((target, np.zeros(knots * overlay_codec.DIMENSIONS)))
    solution = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    controls = np.rint(gain * solution).reshape(knots, overlay_codec.DIMENSIONS)
    return np.clip(
        controls, overlay_codec.MIN_CONTROL, overlay_codec.MAX_CONTROL
    ).astype(np.int32)


def modeled_dpose(
    pairs: np.ndarray,
    jacobians: np.ndarray,
    errors: np.ndarray,
    expanded: np.ndarray,
) -> float:
    residuals = []
    for pair, jacobian, error in zip(pairs, jacobians, errors, strict=True):
        residuals.append(error + jacobian @ expanded[int(pair)])
    return float(np.mean(np.square(np.stack(residuals, dtype=np.float64))))


def lopo_dpose(
    pairs: np.ndarray,
    jacobians: np.ndarray,
    errors: np.ndarray,
    knots: int,
    ridge: float,
    gain: float,
) -> float:
    residuals = []
    for held_out in range(len(pairs)):
        keep = np.arange(len(pairs)) != held_out
        controls = fit_controls(
            pairs[keep], jacobians[keep], errors[keep], knots, ridge, gain
        )
        expanded = overlay_codec.expand_pose_controls(controls)
        residuals.append(
            errors[held_out] + jacobians[held_out] @ expanded[int(pairs[held_out])]
        )
    return float(np.mean(np.square(np.stack(residuals, dtype=np.float64))))


def build_candidate(
    *,
    output: Path,
    knots: int,
    ridge: float,
    gain: float,
    pairs: np.ndarray,
    jacobians: np.ndarray,
    errors: np.ndarray,
    base_codes: np.ndarray,
    sources: tuple[bytes, bytes, bytes, bytes],
) -> dict[str, Any]:
    label = f"k{knots:02d}_r{ridge:.0e}_g{gain:.2f}".replace("+", "")
    root = output / "retained/candidates" / label
    result_path = root / "RESULT.json"
    if result_path.is_file():
        resumed = json.loads(result_path.read_text())
        for name, record in resumed["records"].items():
            require_record(record, f"resumed candidate {label} {name}")
        if "int12_lattice_valid" not in resumed:
            expanded_path = Path(resumed["records"]["expanded_deltas"]["path"])
            expanded = np.load(expanded_path, allow_pickle=False)
            candidate_codes = base_codes + expanded
            violations = int(
                np.count_nonzero((candidate_codes < -2048) | (candidate_codes > 2047))
            )
            resumed["int12_violation_count"] = violations
            resumed["int12_lattice_valid"] = violations == 0
            atomic_json(result_path, resumed)
        return resumed
    controls = fit_controls(pairs, jacobians, errors, knots, ridge, gain)
    overlay = overlay_codec.encode_pose_overlay(controls)
    if not np.array_equal(overlay_codec.decode_pose_overlay(overlay), controls):
        raise PK3Error("overlay encode/decode differs")
    expanded = overlay_codec.expand_pose_controls(controls)
    candidate_codes = base_codes + expanded
    int12_violations = int(
        np.count_nonzero((candidate_codes < -2048) | (candidate_codes > 2047))
    )
    stream_a, stream_b, base_carrier, suffix = sources
    carrier_source = base_carrier + overlay
    stream_c = _brotli(carrier_source, "-q", "11", "-c")
    if max(len(stream_a), len(stream_b), len(stream_c)) >= 1 << 16:
        raise PK3Error("split-model stream exceeds u16")
    models = SPLIT_HEADER.pack(len(stream_a), len(stream_b), len(stream_c))
    models += stream_a + stream_b + stream_c
    member = models + suffix
    archive = deterministic_zip(member)
    records = {
        "controls": atomic_npy(root / "controls.int32.npy", controls),
        "expanded_deltas": atomic_npy(root / "expanded_deltas.int32.npy", expanded),
        "overlay": qs1.retain_bytes(root / "pose_overlay.p0j1", overlay),
        "carrier_source": qs1.retain_bytes(
            root / "carrier_selector_plus_overlay.raw", carrier_source
        ),
        "carrier_stream": qs1.retain_bytes(
            root / "carrier_selector_plus_overlay.q11.br", stream_c
        ),
        "split_models": qs1.retain_bytes(root / "split_models.bin", models),
        "member": qs1.retain_bytes(root / "p", member),
        "archive": qs1.retain_bytes(root / "archive.zip", archive),
    }
    delta_bytes = int(records["archive"]["bytes"]) - CP135_BYTES
    row = {
        "schema": "ddm_pk3_pose_candidate.v1",
        "label": label,
        "knots": knots,
        "ridge": ridge,
        "gain": gain,
        "nonzero_controls": int(np.count_nonzero(controls)),
        "nonzero_expanded_coordinates": int(np.count_nonzero(expanded)),
        "int12_violation_count": int12_violations,
        "int12_lattice_valid": int12_violations == 0,
        "model_sample_dpose": modeled_dpose(
            pairs, jacobians, errors, expanded
        ),
        "model_lopo_dpose": lopo_dpose(
            pairs, jacobians, errors, knots, ridge, gain
        ),
        "records": records,
        "archive_delta_bytes_vs_cp135": delta_bytes,
        "rate_delta_s": delta_bytes * RATE_S_PER_BYTE,
        "axis": AXIS,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(result_path, row)
    return row


def fit_ladder(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = output / "checkpoints/stage_10_fit_ladder.json"
    pairs, jacobians, jacobian_sources = retained_jacobians()
    base_pose = np.load(qs1.CP135_BASE_POSE, allow_pickle=False).astype(np.float64)
    gt_pose = np.load(qs1.GT_POSE, allow_pickle=False).astype(np.float64)
    if base_pose.shape != (600, 6) or gt_pose.shape != base_pose.shape:
        raise PK3Error("pose vector geometry differs")
    errors = base_pose[pairs.astype(np.int64)] - gt_pose[pairs.astype(np.int64)]
    source_records = {
        "pairs": atomic_npy(output / "retained/fit/pairs.int16.npy", pairs),
        "jacobians": atomic_npy(
            output / "retained/fit/jacobians.float64.npy", jacobians
        ),
        "errors": atomic_npy(output / "retained/fit/base_errors.float64.npy", errors),
    }
    sources = rate_sources()
    base_codes = qs1._load_cp135_carrier_codes()
    rows = [
        build_candidate(
            output=output,
            knots=knots,
            ridge=ridge,
            gain=gain,
            pairs=pairs,
            jacobians=jacobians,
            errors=errors,
            base_codes=base_codes,
            sources=sources,
        )
        for knots in KNOTS
        for ridge in RIDGES
        for gain in GAINS
    ]
    base_sample_dpose = float(np.mean(np.square(errors)))
    admissible = [
        row
        for row in rows
        if row["nonzero_controls"] > 0
        and row["int12_lattice_valid"]
        and row["archive_delta_bytes_vs_cp135"] <= MAX_DELTA_BYTES
        and row["model_sample_dpose"] < base_sample_dpose
    ]
    if not admissible:
        raise PK3Error("no nonzero sub-1 KB candidate improves the toy sample model")
    winner = min(
        admissible,
        key=lambda row: (
            float(row["model_lopo_dpose"]),
            float(row["model_sample_dpose"]),
            int(row["archive_delta_bytes_vs_cp135"]),
            str(row["label"]),
        ),
    )
    summary = {
        "schema": "ddm_pk3_fit_ladder.v1",
        "candidate_denominator": len(rows),
        "admissible_denominator": len(admissible),
        "sample_pairs": pairs.astype(int).tolist(),
        "sample_pair_denominator": len(pairs),
        "selection_mode": (
            "minimum leave-one-pair-out modeled dpose among nonzero exact archives "
            "within 1000 bytes and improving the in-sample linear model"
        ),
        "base_sample_dpose": base_sample_dpose,
        "winner": winner,
        "source_records": source_records,
        "jacobian_sources": jacobian_sources,
        "honesty_boundary": (
            "TOY-BRACKET only: n9 non-random pairs; QS1 Jacobians were measured with "
            "JS6 candidate frame-1 partners, not the exact CP135 base frame-1. No "
            "population dpose, score, or negative verdict is inferred."
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(output / "FIT_LADDER_RESULT.json", summary)
    atomic_json(checkpoint, summary)
    return rows, summary


def _replace_runtime_overlay(runtime_root: Path) -> dict[str, Any]:
    source = REPO / "experiments/ddm_pk3_frame0_pose_overlay_runtime.py"
    target = runtime_root / "runtime/compensation_overlay.py"
    atomic_bytes(target, source.read_bytes())
    return qs1.file_record(target)


def compile_winner(output: Path, winner: dict[str, Any]) -> dict[str, Any]:
    partial = output / "compiled_winner"
    if partial.is_dir() and not (partial / "COMPILE_RESULT.json").is_file():
        partial_receipt = {
            "schema": "ddm_pk3_retained_partial_compile.v1",
            "status": "RETAINED_AFTER_FAIL_CLOSED_INT12_OVERFLOW",
            "reason": (
                "the first toy-model winner reached the compile gate before full-n600 "
                "signed-int12 validity was added to candidate selection"
            ),
            "tree": cp135.tree_record(partial),
            "rebuildable": True,
            "deleted": False,
            "score_claim": False,
        }
        atomic_json(output / "RETAINED_PARTIAL_COMPILE_RECEIPT.json", partial_receipt)
    root = output / "compiled_candidates" / str(winner["label"])
    result_path = root / "COMPILE_RESULT.json"
    if result_path.is_file():
        resumed = json.loads(result_path.read_text())
        if (
            resumed["generic_runtime_patches"]["overlay"]
            != resumed["final_overlay_source"]
        ):
            resumed["generic_runtime_patches"]["overlay"] = resumed[
                "final_overlay_source"
            ]
            atomic_json(result_path, resumed)
        for name in (
            "archive",
            "archive_repeat",
            "candidate_codes",
            "sample_frame0_camera",
            "final_overlay_source",
        ):
            require_record(resumed[name], f"resumed compiled {name}")
        runtime_root = Path(resumed["runtime_root"])
        if cp135.tree_record(runtime_root) != resumed["runtime_tree"]:
            raise PK3Error("resumed compiled runtime tree failed custody")
        return resumed
    source_archive = Path(winner["records"]["archive"]["path"])
    archive_payload = source_archive.read_bytes()
    archive_record = qs1.retain_bytes(root / "archive.zip", archive_payload)
    repeat_payload = deterministic_zip(_zip_member(source_archive))
    repeat_record = qs1.retain_bytes(root / "archive.repeat.zip", repeat_payload)
    if archive_record["sha256"] != repeat_record["sha256"]:
        raise PK3Error("independent deterministic archive repeat differs")
    runtime_root = root / "adapted_runtime"
    runtime_copy = jo1.copy_runtime(runtime_root, archive_payload)
    generic_patches = qs2.patch_runtime(runtime_root)
    overlay_source = _replace_runtime_overlay(runtime_root)
    generic_patches["overlay"] = overlay_source
    parseback = qs2.runtime_parseback(
        runtime_root=runtime_root,
        archive=runtime_root / "archive.zip",
        expected_overlay=winner["records"]["overlay"],
    )
    base_codes = qs1._load_cp135_carrier_codes()
    payload = Path(winner["records"]["overlay"]["path"]).read_bytes()
    candidate_codes = overlay_codec.apply_compensation_overlay(base_codes, payload)
    expected = base_codes + np.load(
        winner["records"]["expanded_deltas"]["path"], allow_pickle=False
    )
    if not np.array_equal(candidate_codes, expected):
        raise PK3Error("receiver overlay does not reproduce the selected code lattice")
    code_record = atomic_npy(root / "candidate_codes.int32.npy", candidate_codes)
    surface, surface_pins = qs1.CP135Surface.load()
    sample_pairs = np.asarray(
        json.loads((output / "FIT_LADDER_RESULT.json").read_text())["sample_pairs"],
        dtype=np.int16,
    )
    rendered = np.concatenate(
        [surface.render(candidate_codes[pair : pair + 1], int(pair)) for pair in sample_pairs],
        axis=0,
    )
    rendered_record = atomic_npy(
        root / "retained/sample_frame0_camera.uint8.npy", rendered
    )
    tree = cp135.tree_record(runtime_root)
    result = {
        "schema": "ddm_pk3_compiled_candidate.v1",
        "candidate_label": winner["label"],
        "archive": archive_record,
        "archive_repeat": repeat_record,
        "archive_repeat_byte_identical": True,
        "archive_delta_bytes_vs_cp135": archive_record["bytes"] - CP135_BYTES,
        "overlay": winner["records"]["overlay"],
        "candidate_codes": code_record,
        "sample_frame0_camera": rendered_record,
        "sample_pairs": sample_pairs.astype(int).tolist(),
        "runtime_root": str(runtime_root.resolve()),
        "runtime_copy": runtime_copy,
        "generic_runtime_patches": generic_patches,
        "final_overlay_source": overlay_source,
        "runtime_tree": tree,
        "runtime_parseback": parseback,
        "surface_pins": surface_pins,
        "receiver_parseback": {
            "overlay_exact": True,
            "int12_lattice_exact": True,
            "deterministic_archive_repeat": True,
            "sample_frame0_rendered_through_real_cp135_surface": True,
        },
        "unchanged_object_boundary": (
            "candidate starts from the exact CP135 member and appends only the P0J1 "
            "frame-0 carrier overlay; full public decode and Seg/Pose measurement remain owed"
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(result_path, result)
    atomic_json(output / "checkpoints/stage_20_receiver_compile.json", result)
    return result


def seal_fire_order(
    output: Path, compiled: dict[str, Any], fit: dict[str, Any]
) -> dict[str, Any]:
    from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b

    run_id = "ddm_pk3_dual_axis_20260813_r1"
    fire_root = output / "fire_order"
    input_root = fire_root / "fire_inputs"
    archive_path = Path(compiled["archive"]["path"])
    runtime_bundle, runtime_manifest = js1b.build_runtime_bundle(
        Path(compiled["runtime_root"]), label="ddm_pk3_frame0_pose_p0j1"
    )
    screen = {
        "schema": "ddm_pk3_pose_screen.v1",
        "disposition": "TOY-BRACKET_CANDIDATE_REQUIRES_EXACT_VERDICT",
        "sample_pair_denominator": fit["sample_pair_denominator"],
        "base_sample_dpose": fit["base_sample_dpose"],
        "model_sample_dpose": fit["winner"]["model_sample_dpose"],
        "model_lopo_dpose": fit["winner"]["model_lopo_dpose"],
        "archive_delta_bytes_vs_cp135": compiled["archive_delta_bytes_vs_cp135"],
        "target_dpose": TARGET_DPOSE,
        "max_delta_bytes": MAX_DELTA_BYTES,
        "honesty_boundary": fit["honesty_boundary"],
        "score_claim": False,
        "promotion_eligible": False,
    }
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
        "POSE_SCREEN_RESULT.json": (
            json.dumps(screen, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    for name, payload in payloads.items():
        qs1.retain_bytes(input_root / name, payload)
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": run_id,
        "resume_from": run_id,
        "lane_id": "ddm_pk3_dual_axis_n600_20260813",
        "instance_job_id": f"modal:{run_id}",
        "claim_agent": "MAIN",
        "seed": 135,
        "batch_size": 16,
        "retain_pose_vectors": True,
        "candidate_archive": qs1.file_record(archive_path),
        "candidate_runtime": compiled["runtime_tree"],
        "runtime_manifest": runtime_manifest,
        "inputs": {name: js1b.payload_record(payload) for name, payload in payloads.items()},
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": hashlib.sha256(git_status).hexdigest(),
        "dispatcher_source_sha256": cp135.sha256_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker_source_sha256": cp135.sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": cp135.sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    request_record = atomic_json(fire_root / "SEALED_REQUEST.json", request)
    dispatch_output = output / "dispatch" / run_id
    command = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_qs1_modal_t4_dual_axis.py::main",
        "--sealed-request",
        request_record["path"],
        "--fire-input-dir",
        str(input_root.resolve()),
        "--expected-request-sha256",
        request_record["sha256"],
        "--output-dir",
        str(dispatch_output.resolve()),
        "--detach",
        "--provider-detach-ack",
    ]
    order = {
        "schema": "ddm_pk3_sealed_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": (
            "MAIN confirms no active n600 exact-eval/Modal lane, claims lane "
            "ddm_pk3_dual_axis_n600_20260813, verifies the sealed request and every "
            "fire-input SHA, then executes exact_command_argv"
        ),
        "fresh_run_id": run_id,
        "request": request_record,
        "fire_inputs": str(input_root.resolve()),
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "budget_ledger": "#381",
        "remote_scope": (
            "one candidate public decode plus n600 frozen T4 SegNet argmax field and "
            "official PoseNet first-six vectors with a repeat"
        ),
        "admission": {
            "pose_target": "d_pose <= 3.44e-6",
            "rate_target": "archive delta <= 1000 bytes",
            "seg_target": "no d_seg increase versus CP135",
            "score_target": "complete recomputed S < 0.16195513827824176",
        },
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(output / "SEALED_FIRE_ORDER.json", order)
    atomic_json(output / "checkpoints/stage_30_sealed_fire_order.json", order)
    return order


def write_consumer_handoff(
    output: Path,
    compiled: dict[str, Any],
    fit: dict[str, Any],
    order: dict[str, Any],
) -> dict[str, Any]:
    CONSUMER_STORE.mkdir(parents=True, exist_ok=True)
    handoff = {
        "schema": "ddm_pk3_to_eu4_handoff.v1",
        "producer": "ddm_pk3",
        "disposition": order["disposition"],
        "candidate_archive": compiled["archive"],
        "archive_delta_bytes_vs_cp135": compiled["archive_delta_bytes_vs_cp135"],
        "fit_ladder": qs1.file_record(output / "FIT_LADDER_RESULT.json"),
        "compiled_result": qs1.file_record(
            Path(compiled["runtime_root"]).parent / "COMPILE_RESULT.json"
        ),
        "sealed_fire_order": qs1.file_record(output / "SEALED_FIRE_ORDER.json"),
        "owner": order["owner"],
        "fire_trigger": order["fire_trigger"],
        "honesty_boundary": fit["honesty_boundary"],
        "score_claim": False,
        "promotion_eligible": False,
    }
    return atomic_json(CONSUMER_STORE / "retained/PK3_HANDOFF.json", handoff)


def run(output: Path) -> dict[str, Any]:
    final_path = output / "FINAL_RESULT.json"
    storage = storage_preflight(output)
    sources = source_preflight(output)
    rows, fit = fit_ladder(output)
    compiled = compile_winner(output, fit["winner"])
    order = seal_fire_order(output, compiled, fit)
    handoff = write_consumer_handoff(output, compiled, fit, order)
    result = {
        "schema": "ddm_pk3_final.v1",
        "status": "BUILT_BYTE_CLOSED_CANDIDATE_AND_QUEUED_EXACT_VERDICT",
        "axis": AXIS,
        "storage_preflight": storage,
        "source_preflight": sources,
        "candidate_denominator": len(rows),
        "fit": qs1.file_record(output / "FIT_LADDER_RESULT.json"),
        "winner": fit["winner"],
        "compiled": compiled,
        "fire_order": order,
        "consumer_handoff": handoff,
        "charter_targets": {
            "d_pose": TARGET_DPOSE,
            "delta_bytes": MAX_DELTA_BYTES,
            "seg_harm": 0.0,
        },
        "measured_now": [
            "exact retained archive bytes",
            "deterministic archive repeat",
            "overlay parse-back and exact signed-int12 code lattice",
            "real CP135 frame-0 renderer smoke on 9 retained pairs",
        ],
        "not_measured_now": [
            "PoseNet d_pose",
            "SegNet d_seg",
            "complete contest score",
            "contest-CPU or contest-CUDA authority",
        ],
        "owns_scorer": False,
        "scorer_calls": 0,
        "all_materialized_payloads_retained": True,
        "resumable": True,
        "per_stage_checkpoints": True,
        "own_vehicle_frontier_moved": False,
        "effective_frontier_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(final_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() != args.resume_from.resolve():
        raise SystemExit("FATAL: --output and --resume-from must identify one durable store")
    result = run(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
