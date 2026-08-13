#!/usr/bin/env python3
"""Reprice EC1 events inside CP135's actual HP3/RC64 probability object.

This runner is deliberately scorer-free.  It materializes edited n600 semantic
token planes, exports the exact teacher-forced F26/HP3 probability lattice,
fresh-encodes RC64, and closes complete CP135 archives.  Every payload and
per-frame/per-24-frame checkpoint is retained on the SSD tier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_cp135_rate_compose as cp135
from experiments import ddm_ec1_event_coordinate_producer as ec1
from experiments import ddm_js6_event_proposal_acceptance as js6

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_jo1_20260812")
BASE_CANDIDATE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/candidates/"
    "hp3_step2/split_brotli_per_section_opt_cap1_metadata__rc64"
)
BASE_ARCHIVE: Final = BASE_CANDIDATE / "archive.zip"
BASE_SPATIAL: Final = ec1.BASE_TOKENS
BASE_PROBABILITIES: Final = (
    Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810")
    / "retained/probabilities/hp3_step2"
)
CP135_RUNTIME: Final = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime")
PR135_ARCHIVE: Final = cp135.DEFAULT_ARCHIVE
EXPERIMENT_BOOK: Final = cp135.DEFAULT_EXPERIMENT_BOOK
JS7_TABLE: Final = Path("/Volumes/APDataStore/pact/ddm_js7_20260812/ACCEPTANCE_TABLE.json")
JS7_COMPOSE: Final = Path("/Volumes/APDataStore/pact/ddm_js7_20260812/compose/COMPOSE_RESULT.json")
JS7_TRIALS: Final = Path("/Volumes/APDataStore/pact/ddm_js7_20260812/compose/trials")

FRAMES: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
EVENTS_PER_FRAME: Final = HEIGHT * WIDTH
TOTAL_EVENTS: Final = FRAMES * EVENTS_PER_FRAME
BASE_BYTES: Final = 186_252
BASE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
BASE_SPATIAL_SHA256: Final = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
BASE_D_SEG: Final = 0.00029643
BASE_D_POSE: Final = 0.00000688
BASE_SCORE: Final = 0.16195513827824176
RATE_DENOMINATOR: Final = 37_545_489
POSE_STACK_BUDGET: Final = 1.3e-7
JS7_EXACT_TOKEN_SHA256: Final = "a78bb2992b3e711b602909ca90ca72dc98c1ab8f6cfcea30594d2f18c53810e0"
JS7_EXACT_ARCHIVE_BYTES: Final = 186_575
JS7_EXACT_SCORE: Final = 0.16342603740620176
JS7_EXACT_D_SEG: Final = 0.00029675
JS7_EXACT_D_POSE: Final = 0.00000906
MIN_FREE_BYTES: Final = 8 * 1024**3
AXIS: Final = "[macOS-CPU scorer-free direct-token HP3/RC64 n600 reclose]"
COMPOSED_NAME: Final = "calibrated_pose_nonspending"


class JO1Error(RuntimeError):
    """A JO1 custody, receiver, resume, or accounting invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes, *, executable: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if executable:
        temporary.chmod(0o755)
    os.replace(temporary, path)
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def raw_array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    flat = value.reshape(-1)
    chunk = 8 * 1024 * 1024
    for start in range(0, flat.size, chunk):
        digest.update(np.asarray(flat[start : start + chunk]).tobytes())
    return digest.hexdigest()


def require(path: Path, *, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file():
        raise JO1Error(f"required artifact is missing: {path}")
    if size is not None and path.stat().st_size != size:
        raise JO1Error(f"byte count differs for {path}")
    if digest is not None and sha256_file(path) != digest:
        raise JO1Error(f"SHA-256 differs for {path}")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def save_state(output: Path, stage: str, complete: list[str]) -> None:
    atomic_json(
        output / "state.json",
        {
            "schema": "ddm_jo1_state.v1",
            "status": "COMPLETE" if stage == "complete" else "RUNNING",
            "active_stage": stage,
            "completed_stages": complete,
            "resumable": True,
            "scorer_run": False,
            "score_claim": False,
        },
    )


def preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if not output.resolve().is_relative_to(Path("/Volumes/VertigoDataTier/pact")):
        raise JO1Error("JO1 output must remain on the primary SSD tier")
    usage = shutil.disk_usage(output)
    if usage.free < MIN_FREE_BYTES:
        raise JO1Error("SSD storage preflight failed closed")
    require(BASE_ARCHIVE, size=BASE_BYTES, digest=BASE_SHA256)
    require(BASE_SPATIAL, size=TOTAL_EVENTS, digest=BASE_SPATIAL_SHA256)
    for name in ("models.bin", "residual.compact.bin", "tokens.rc64", "p"):
        require(BASE_CANDIDATE / name)
    if (BASE_CANDIDATE / "p").read_bytes() != (
        (BASE_CANDIDATE / "models.bin").read_bytes()
        + (BASE_CANDIDATE / "residual.compact.bin").read_bytes()
        + (BASE_CANDIDATE / "tokens.rc64").read_bytes()
    ):
        raise JO1Error("CP135 retained member does not equal its three physical sections")
    result = {
        "schema": "ddm_jo1_preflight.v1",
        "axis": AXIS,
        "git_head": git_head(),
        "free_bytes": usage.free,
        "required_free_bytes": MIN_FREE_BYTES,
        "base_archive": file_record(BASE_ARCHIVE),
        "base_spatial_tokens": file_record(BASE_SPATIAL),
        "base_probability_export": file_record(BASE_PROBABILITIES / "EXPORT_RESULT.json"),
        "proposal_index": file_record(js6.DEFAULT_PROPOSAL_ROOT / "proposal_index.jsonl"),
        "js7_table": file_record(JS7_TABLE),
        "js7_compose": file_record(JS7_COMPOSE),
        "scorer_run": False,
        "all_payloads_retained": True,
    }
    atomic_json(output / "00_PREFLIGHT.json", result)
    return result


def load_inputs() -> tuple[Any, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    store = js6.load_store(js6.DEFAULT_PROPOSAL_ROOT)
    table = json.loads(JS7_TABLE.read_text())
    compose = json.loads(JS7_COMPOSE.read_text())
    trials = [json.loads(path.read_text()) for path in sorted(JS7_TRIALS.glob("*/RESULT.json"))]
    if len(store.rows) != 200 or len(table.get("rows", ())) != 200:
        raise JO1Error("EC1/JS7 proposal population is incomplete")
    if len(compose.get("selected_ids", ())) != 44 or Counter(row["reason"] for row in trials) != {
        "ACCEPTED": 44,
        "POSE_GATE": 20,
        "NO_MARGINAL_ROBUST_GAIN": 1,
    }:
        raise JO1Error("JS7 terminal trial population differs from the charter crosswalk")
    return store, list(table["rows"]), compose, trials


def proposal_payload(row: dict[str, Any]) -> bytes:
    record = row["consumer_payloads"]["event.ec1p"]
    path = Path(record["path"])
    if file_record(path) != record:
        raise JO1Error(f"proposal payload failed custody: {path}")
    return path.read_bytes()


def contour_bending_energy(mask: np.ndarray) -> float:
    """Discrete arclength/turning energy on lattice contour components."""

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    total = 0.0
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        if len(points) < 4:
            continue
        steps = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(steps, axis=1)
        keep = lengths > 0
        if int(keep.sum()) < 3:
            continue
        tangent = steps[keep] / lengths[keep, None]
        prior = np.roll(tangent, 1, axis=0)
        cross = prior[:, 0] * tangent[:, 1] - prior[:, 1] * tangent[:, 0]
        dot = np.clip((prior * tangent).sum(axis=1), -1.0, 1.0)
        angles = np.arctan2(cross, dot)
        arc = 0.5 * (lengths[keep] + np.roll(lengths[keep], 1))
        total += float(np.sum(np.square(angles) / np.maximum(arc, 1e-12)))
    return total


def bending_delta(base_frame: np.ndarray, source: int, target: int, indices: np.ndarray) -> dict[str, Any]:
    ys, xs = np.divmod(indices.astype(np.int64), WIDTH)
    y0, y1 = max(0, int(ys.min()) - 12), min(HEIGHT, int(ys.max()) + 13)
    x0, x1 = max(0, int(xs.min()) - 12), min(WIDTH, int(xs.max()) + 13)
    before = np.asarray(base_frame[y0:y1, x0:x1]).copy()
    after = before.copy()
    local_y, local_x = ys - y0, xs - x0
    if np.any(after[local_y, local_x] != source):
        raise JO1Error("bending feature source precondition failed")
    after[local_y, local_x] = target
    by_class: dict[str, Any] = {}
    total_before = 0.0
    total_after = 0.0
    for class_id in sorted({source, target}):
        before_energy = contour_bending_energy(before == class_id)
        after_energy = contour_bending_energy(after == class_id)
        by_class[str(class_id)] = {
            "before": before_energy,
            "after": after_energy,
            "delta": after_energy - before_energy,
        }
        total_before += before_energy
        total_after += after_energy
    return {
        "bbox_yx": [y0, x0, y1, x1],
        "before": total_before,
        "after": total_after,
        "delta": total_after - total_before,
        "by_class": by_class,
    }


def _ranking_score(rows: list[dict[str, Any]], key: str, pairs: set[int], top_k: int = 5) -> dict[str, Any]:
    chosen: list[dict[str, Any]] = []
    for pair in sorted(pairs):
        group = [row for row in rows if int(row["pair"]) == pair and bool(row["pose_gate_pass"])]
        group.sort(key=lambda row: (float(row[key]), int(row["ordinal"])))
        chosen.extend(group[:top_k])
    gains = [max(0, -int(row["projected_robust_delta_flips"])) for row in chosen]
    return {
        "pairs": sorted(pairs),
        "selected": len(chosen),
        "useful": sum(gain > 0 for gain in gains),
        "useful_yield": 0.0 if not chosen else sum(gain > 0 for gain in gains) / len(chosen),
        "robust_flip_gain": sum(gains),
    }


def analyze(output: Path) -> dict[str, Any]:
    store, rows, compose, trials = load_inputs()
    base = np.memmap(BASE_SPATIAL, mode="r", dtype=np.uint8, shape=(FRAMES, HEIGHT, WIDTH))
    store_by_id = {str(row["proposal_id"]): row for row in store.rows}
    features: list[dict[str, Any]] = []
    for row in rows:
        owned = store_by_id[str(row["proposal_id"])]
        frame, source, target, event_type, indices = ec1.decode_proposal(proposal_payload(owned))
        feature = bending_delta(base[frame], source, target, indices)
        features.append({**row, "frame": frame, "source_class": source, "target_class": target, **feature})
    pairs = sorted({int(row["pair"]) for row in features})
    folds: list[dict[str, Any]] = []
    sign_votes: list[int] = []
    for heldout in pairs:
        train_pairs = set(pairs) - {heldout}
        signed_candidates = []
        for sign in (-1, 1):
            for row in features:
                row[f"bending_rank_{sign}"] = sign * float(row["delta"])
            score = _ranking_score(features, f"bending_rank_{sign}", train_pairs)
            signed_candidates.append((score["robust_flip_gain"], score["useful"], -sign, sign))
        sign = max(signed_candidates)[-1]
        sign_votes.append(sign)
        bending = _ranking_score(features, f"bending_rank_{sign}", {heldout})
        for row in features:
            ratio = row["bytes_per_projected_robust_flip"]
            row["bpf_rank"] = 1e30 if ratio is None else float(ratio)
        baseline = _ranking_score(features, "bpf_rank", {heldout})
        folds.append({"heldout_pair": heldout, "trained_sign": sign, "bending": bending, "bpf": baseline})
    bending_gain = sum(row["bending"]["robust_flip_gain"] for row in folds)
    baseline_gain = sum(row["bpf"]["robust_flip_gain"] for row in folds)
    bending_useful = sum(row["bending"]["useful"] for row in folds)
    baseline_useful = sum(row["bpf"]["useful"] for row in folds)
    family_signs = {}
    for family in ("boundary", "lane"):
        family_rows = [row for row in features if row["family"] == family]
        scored = []
        for sign in (-1, 1):
            key = f"bending_rank_{sign}"
            scored.append((_ranking_score(family_rows, key, set(pairs))["robust_flip_gain"], sign))
        family_signs[family] = max(scored)[1]
    adopted = (
        bending_gain > baseline_gain
        and bending_useful >= baseline_useful
        and len(set(sign_votes)) == 1
        and len(set(family_signs.values())) == 1
    )
    selected_rows: list[dict[str, Any]] = []
    for pair in pairs:
        eligible = [
            row
            for row in features
            if int(row["pair"]) == pair
            and bool(row["bare_admission"])
            and float(row["pose_delta"]) <= 0.0
            and int(row["projected_robust_delta_flips"]) < 0
        ]
        if adopted:
            stable_sign = sign_votes[0]
            eligible.sort(key=lambda row: (stable_sign * float(row["delta"]), int(row["ordinal"])))
        else:
            eligible.sort(
                key=lambda row: (
                    float(row["bytes_per_projected_robust_flip"]),
                    int(row["projected_robust_delta_flips"]),
                    int(row["ordinal"]),
                )
            )
        if eligible:
            selected_rows.append(eligible[0])
    if any(float(row["pose_delta"]) > 0 for row in selected_rows):
        raise JO1Error("calibrated stack contains a pose-spending event without the Schur antidote")
    candidate_sets = {
        "ec1_all_200": [str(row["proposal_id"]) for row in rows],
        "js7_selected_44": [str(value) for value in compose["selected_ids"]],
        "js7_pose_rejected_20": [
            str(row["proposal_id"]) for row in trials if not row["accepted"] and row["reason"] == "POSE_GATE"
        ],
        COMPOSED_NAME: [str(row["proposal_id"]) for row in selected_rows],
    }
    result = {
        "schema": "ddm_jo1_analysis.v1",
        "axis": AXIS,
        "scorer_run": False,
        "proposal_rows": len(features),
        "features": features,
        "bending_gate": {
            "disposition": "ADOPT" if adopted else "REJECT",
            "metric": "leave-one-pair-out top-5 robust useful yield inside the existing pose gate",
            "folds": folds,
            "heldout_bending_robust_flip_gain": bending_gain,
            "heldout_bpf_robust_flip_gain": baseline_gain,
            "heldout_bending_useful": bending_useful,
            "heldout_bpf_useful": baseline_useful,
            "sign_votes": sign_votes,
            "family_signs": family_signs,
            "falsifier": "reject unless held-out gain beats B/flip and preferred sign is stable by fold and stratum",
        },
        "pose_stack_budget_d_pose": POSE_STACK_BUDGET,
        "frame0_schur_antidote": {
            "invoked": False,
            "reason": "every composed event has non-positive singleton n32 pose delta",
            "fire_if": "a future selected event spends pose or n600 validation reports positive pose debt",
        },
        "composed_selection": [
            {
                "proposal_id": row["proposal_id"],
                "pair": row["pair"],
                "pose_delta": row["pose_delta"],
                "projected_robust_delta_flips": row["projected_robust_delta_flips"],
                "bytes_per_projected_robust_flip": row["bytes_per_projected_robust_flip"],
            }
            for row in selected_rows
        ],
        "candidate_sets": candidate_sets,
    }
    atomic_json(output / "10_ANALYSIS.json", result)
    return result


def candidate_root(output: Path, name: str, *, repeat: bool = False) -> Path:
    suffix = "determinism_repeat" if repeat else "primary"
    return output / "retained/candidates" / name / suffix


def materialize_candidate(output: Path, name: str, proposal_ids: list[str], *, repeat: bool = False) -> dict[str, Any]:
    root = candidate_root(output, name, repeat=repeat)
    root.mkdir(parents=True, exist_ok=True)
    store, _, _, _ = load_inputs()
    store_by_id = {str(row["proposal_id"]): row for row in store.rows}
    spatial_path = root / "spatial_tokens.u8"
    applications_path = root / "EVENT_APPLICATIONS.json"
    if not spatial_path.is_file():
        temporary = spatial_path.with_name(f".{spatial_path.name}.{os.getpid()}.partial")
        shutil.copyfile(BASE_SPATIAL, temporary)
        spatial = np.memmap(temporary, mode="r+", dtype=np.uint8, shape=(FRAMES, HEIGHT, WIDTH))
        touched: dict[tuple[int, int], int] = {}
        applications = []
        for proposal_id in proposal_ids:
            row = store_by_id[proposal_id]
            payload_record = row["consumer_payloads"]["event.ec1p"]
            frame, source, target, event_type, indices = ec1.decode_proposal(proposal_payload(row))
            flat = spatial[frame].reshape(-1)
            if np.any(flat[indices] != source):
                raise JO1Error(f"proposal source precondition failed: {proposal_id}")
            for index in indices.tolist():
                key = (frame, int(index))
                if key in touched and touched[key] != target:
                    raise JO1Error(f"conflicting target assignments at {key}")
                touched[key] = target
            flat[indices] = target
            applications.append(
                {
                    "proposal_id": proposal_id,
                    "payload": payload_record,
                    "frame": frame,
                    "source_class": source,
                    "target_class": target,
                    "event_type": event_type,
                    "indices": indices.astype(int).tolist(),
                }
            )
        spatial.flush()
        del spatial
        os.replace(temporary, spatial_path)
        atomic_json(applications_path, {"schema": "ddm_jo1_event_applications.v1", "rows": applications})
    applications = json.loads(applications_path.read_text())["rows"]
    if [row["proposal_id"] for row in applications] != proposal_ids:
        raise JO1Error(f"retained proposal order differs for {name}")
    spatial = np.memmap(spatial_path, mode="r", dtype=np.uint8, shape=(FRAMES, HEIGHT, WIDTH))
    spatial_raw_sha = raw_array_sha256(spatial)
    if spatial_raw_sha != sha256_file(spatial_path):
        raise JO1Error("raw spatial-token digest differs from file digest")
    changed_frames = sorted({int(row["frame"]) for row in applications})
    affected_frames = sorted(set(changed_frames) | {frame + 1 for frame in changed_frames if frame + 1 < FRAMES})

    group_positions = cp135._group_positions(CP135_RUNTIME)
    positions = np.concatenate(group_positions)
    if positions.shape != (EVENTS_PER_FRAME,) or np.unique(positions).size != EVENTS_PER_FRAME:
        raise JO1Error("F26 group positions are not a permutation")
    event_path = root / "event_order.npy"
    progress_path = root / "EVENT_ORDER_PROGRESS.json"
    if event_path.is_file():
        event_order = np.lib.format.open_memmap(event_path, mode="r+")
        progress = json.loads(progress_path.read_text())
        start = int(progress["next_frame"])
    else:
        event_order = np.lib.format.open_memmap(event_path, mode="w+", dtype=np.uint8, shape=(TOTAL_EVENTS,))
        start = 0
        progress = {"next_frame": 0, "prefix_sha256": hashlib.sha256().hexdigest(), "complete": False}
        atomic_json(progress_path, progress)
    prefix = hashlib.sha256()
    for frame in range(start):
        first = frame * EVENTS_PER_FRAME
        prefix.update(np.asarray(event_order[first : first + EVENTS_PER_FRAME]).tobytes())
    if prefix.hexdigest() != progress["prefix_sha256"]:
        raise JO1Error("event-order checkpoint prefix failed custody")
    for frame in range(start, FRAMES):
        values = np.asarray(spatial[frame]).reshape(-1)[positions]
        first = frame * EVENTS_PER_FRAME
        event_order[first : first + EVENTS_PER_FRAME] = values
        prefix.update(values.tobytes())
        if (frame + 1) % 24 == 0 or frame == FRAMES - 1:
            event_order.flush()
            progress = {
                "next_frame": frame + 1,
                "prefix_sha256": prefix.hexdigest(),
                "complete": frame + 1 == FRAMES,
            }
            atomic_json(progress_path, progress)
    event_raw_sha = raw_array_sha256(event_order)
    if event_raw_sha != prefix.hexdigest():
        raise JO1Error("event-order terminal digest differs from checkpoint chain")
    manifest = {
        "schema": "ddm_jo1_event_order_manifest.v1",
        "complete": True,
        "chunks": [
            {
                "start_frame": 0,
                "end_frame": FRAMES,
                "symbols_path": str(event_path.resolve()),
                "symbols_sha256": sha256_file(event_path),
                "symbols_bytes": event_path.stat().st_size,
                "tokens": TOTAL_EVENTS,
            }
        ],
    }
    manifest_path = root / "chunk_manifest.json"
    atomic_json(manifest_path, manifest)
    result = {
        "schema": "ddm_jo1_materialized_candidate.v1",
        "name": name,
        "repeat": repeat,
        "proposal_count": len(proposal_ids),
        "proposal_ids": proposal_ids,
        "changed_sites": sum(len(row["indices"]) for row in applications),
        "changed_frames": changed_frames,
        "probability_affected_frames": affected_frames,
        "spatial_tokens": file_record(spatial_path),
        "spatial_raw_sha256": spatial_raw_sha,
        "event_order": file_record(event_path),
        "event_order_raw_sha256": event_raw_sha,
        "source_manifest": file_record(manifest_path),
        "event_applications": file_record(applications_path),
        "resumable": True,
    }
    atomic_json(root / "20_MATERIALIZE_RESULT.json", result)
    return result


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        started = time.time()
        log.write(json.dumps({"argv": command, "started_unix_s": started}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        log.write(json.dumps({"returncode": returncode, "wall_s": time.time() - started}) + "\n")
        log.flush()
        os.fsync(log.fileno())
    if returncode:
        raise JO1Error(f"subprocess failed rc={returncode}: {command}")


def link_unaffected_probabilities(adapter: Path, affected_frames: set[int]) -> None:
    destination = adapter / "retained/probabilities/hp3_step2"
    destination.mkdir(parents=True, exist_ok=True)
    for frame in range(FRAMES):
        if frame in affected_frames:
            continue
        source = BASE_PROBABILITIES / f"codes_{frame:04d}.npy"
        target = destination / source.name
        require(source)
        if target.exists():
            if file_record(target)["sha256"] != file_record(source)["sha256"]:
                raise JO1Error(f"reused probability checkpoint differs: {target}")
        else:
            os.link(source, target)
        atomic_json(
            destination / f"codes_{frame:04d}.json",
            cp135._frame_record(target, frame, "hp3_step2"),
        )


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def copy_runtime(destination: Path, archive_payload: bytes) -> dict[str, Any]:
    for source in sorted(CP135_RUNTIME.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
            continue
        relative = source.relative_to(CP135_RUNTIME)
        if relative.as_posix() in {"archive.zip", "inflate.py"}:
            continue
        target = destination / relative
        if not target.is_file():
            atomic_bytes(target, source.read_bytes(), executable=os.access(source, os.X_OK))
        elif target.read_bytes() != source.read_bytes():
            raise JO1Error(f"runtime copy differs at {relative}")
    archive_record = atomic_bytes(destination / "archive.zip", archive_payload)
    source_text = (CP135_RUNTIME / "inflate.py").read_text()
    new_sha = sha256_bytes(archive_payload)
    if source_text.count(BASE_SHA256) != 1 or source_text.count(f"ARCHIVE_BYTES = {BASE_BYTES:_}") != 1:
        raise JO1Error("CP135 inflate archive pin surface drifted")
    source_text = source_text.replace(BASE_SHA256, new_sha).replace(
        f"ARCHIVE_BYTES = {BASE_BYTES:_}", f"ARCHIVE_BYTES = {len(archive_payload):_}"
    )
    inflate_record = atomic_bytes(destination / "inflate.py", source_text.encode(), executable=True)
    return {"archive": archive_record, "inflate": inflate_record, "source_runtime": str(CP135_RUNTIME)}


def shipped_receiver_parseback(
    root: Path,
    adapter: Path,
    token: bytes,
    expected_event_sha256: str,
    expected_spatial_sha256: str,
) -> dict[str, Any]:
    receiver = root / "receiver_state"
    result_path = receiver / "SHIPPED_RC64_PARSEBACK.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        for key in ("decoded_symbols", "decoded_spatial_tokens", "source", "library"):
            if file_record(Path(result[key]["path"])) != result[key]:
                raise JO1Error(f"retained shipped receiver payload failed custody: {key}")
        if (
            result.get("event_order_sha256") != expected_event_sha256
            or result.get("spatial_token_sha256") != expected_spatial_sha256
        ):
            raise JO1Error("retained shipped receiver result is bound to a different source")
        return result
    receiver.mkdir(parents=True, exist_ok=True)
    source = CP135_RUNTIME / "runtime/entropy/rc64_backend.c"
    library = receiver / "librc64_shipped.so"
    temporary_library = library.with_name(f".{library.name}.{os.getpid()}.partial")
    command = ["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(temporary_library)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    compile_receipt = {
        "schema": "ddm_jo1_shipped_rc64_compile.v1",
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "source": file_record(source),
    }
    if completed.returncode:
        atomic_json(receiver / "COMPILE_RESULT.json", compile_receipt)
        raise JO1Error("shipped RC64 backend compilation failed")
    os.replace(temporary_library, library)
    compile_receipt["library"] = file_record(library)
    atomic_json(receiver / "COMPILE_RESULT.json", compile_receipt)
    spec = importlib.util.spec_from_file_location("ddm_jo1_shipped_rc64", CP135_RUNTIME / "runtime/entropy/rc64.py")
    if spec is None or spec.loader is None:
        raise JO1Error("could not load the shipped RC64 Python binding")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoder = module.NativeDecoder(library, token)
    group_positions = cp135._group_positions(CP135_RUNTIME)
    expected = np.load(root / "event_order.npy", mmap_mode="r", allow_pickle=False)
    event_output = receiver / "decoded_symbols.shipped.bin"
    spatial_output = receiver / "decoded_spatial_tokens.shipped.bin"
    event_temporary = event_output.with_name(f".{event_output.name}.{os.getpid()}.partial")
    spatial_temporary = spatial_output.with_name(f".{spatial_output.name}.{os.getpid()}.partial")
    event_digest = hashlib.sha256()
    spatial_digest = hashlib.sha256()
    started = time.time()
    with event_temporary.open("wb") as event_stream, spatial_temporary.open("wb") as spatial_stream:
        for frame in range(FRAMES):
            codes = np.load(
                adapter / f"retained/probabilities/hp3_step2/codes_{frame:04d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            decoded = decoder.decode(cp135.probability_from_codes(codes, 8)).astype(np.uint8)
            first = frame * EVENTS_PER_FRAME
            if not np.array_equal(decoded, np.asarray(expected[first : first + EVENTS_PER_FRAME])):
                raise JO1Error(f"shipped RC64 receiver differs at frame {frame}")
            event_raw = decoded.tobytes()
            spatial_raw = cp135.spatial_frame(decoded, group_positions).tobytes()
            event_stream.write(event_raw)
            spatial_stream.write(spatial_raw)
            event_digest.update(event_raw)
            spatial_digest.update(spatial_raw)
        event_stream.flush()
        os.fsync(event_stream.fileno())
        spatial_stream.flush()
        os.fsync(spatial_stream.fileno())
    decoder_bit_position = decoder.bit_position
    decoder.close()
    os.replace(event_temporary, event_output)
    os.replace(spatial_temporary, spatial_output)
    if event_digest.hexdigest() != expected_event_sha256 or spatial_digest.hexdigest() != expected_spatial_sha256:
        raise JO1Error("shipped RC64 receiver terminal digest differs")
    result = {
        "schema": "ddm_jo1_shipped_rc64_parseback.v1",
        "complete": True,
        "events": TOTAL_EVENTS,
        "event_order_sha256": event_digest.hexdigest(),
        "spatial_token_sha256": spatial_digest.hexdigest(),
        "decoded_symbols": file_record(event_output),
        "decoded_spatial_tokens": file_record(spatial_output),
        "source": file_record(source),
        "library": file_record(library),
        "decoder_bit_position": decoder_bit_position,
        "wall_s": time.time() - started,
    }
    atomic_json(result_path, result)
    return result


def reclose_candidate(output: Path, name: str, *, repeat: bool = False) -> dict[str, Any]:
    root = candidate_root(output, name, repeat=repeat)
    prepared = json.loads((root / "20_MATERIALIZE_RESULT.json").read_text())
    manifest = Path(prepared["source_manifest"]["path"])
    event_sha = prepared["event_order_raw_sha256"]
    spatial_sha = prepared["spatial_raw_sha256"]
    adapter = root / "hp3_adapter"
    link_unaffected_probabilities(adapter, set(prepared["probability_affected_frames"]))
    common = [
        sys.executable,
        str(REPO / "experiments/ddm_cp135_rate_compose.py"),
        "--variant",
        "hp3_step2",
        "--archive",
        str(PR135_ARCHIVE),
        "--runtime",
        str(CP135_RUNTIME),
        "--output",
        str(adapter),
        "--dt1-manifest",
        str(manifest),
        "--experiment-book",
        str(EXPERIMENT_BOOK),
        "--expected-event-order-sha256",
        event_sha,
        "--expected-spatial-token-sha256",
        spatial_sha,
    ]
    export_command = common.copy()
    export_command.insert(2, "export")
    run_logged(export_command, root / "logs/30_HP3_EXPORT.log")
    encode_command = common.copy()
    encode_command.insert(2, "encode-rc64")
    run_logged(encode_command, root / "logs/40_RC64.log")
    export_path = adapter / "retained/probabilities/hp3_step2/EXPORT_RESULT.json"
    rc64_path = adapter / "retained/coders/hp3_step2/FRESH_RC64_RESULT.json"
    export = json.loads(export_path.read_text())
    rc64 = json.loads(rc64_path.read_text())
    if (
        export.get("complete_n600") is not True
        or export.get("source_symbol_sha256") != event_sha
        or rc64.get("symbol_identity") is not True
        or rc64.get("decoded_event_order_sha256") != event_sha
        or rc64.get("decoded_spatial_token_sha256") != spatial_sha
    ):
        raise JO1Error(f"HP3/RC64 identity failed for {name}")
    token_path = Path(rc64["token_payload"]["path"])
    models = (BASE_CANDIDATE / "models.bin").read_bytes()
    residual = (BASE_CANDIDATE / "residual.compact.bin").read_bytes()
    token = token_path.read_bytes()
    member = models + residual + token
    archive_payload = deterministic_zip(member)
    archive_repeat = deterministic_zip(member)
    if archive_payload != archive_repeat:
        raise JO1Error("deterministic ZIP repeat differs")
    objects = root / "objects"
    atomic_bytes(objects / "models.bin", models)
    atomic_bytes(objects / "residual.compact.bin", residual)
    atomic_bytes(objects / "tokens.rc64", token)
    atomic_bytes(objects / "p", member)
    archive_record = atomic_bytes(objects / "archive.zip", archive_payload)
    repeat_record = atomic_bytes(objects / "archive.repeat.zip", archive_repeat)
    runtime_module = cp135.load_runtime(CP135_RUNTIME)
    base_parts = runtime_module.read_residual_archive(BASE_ARCHIVE)
    parts = runtime_module.read_residual_archive(objects / "archive.zip")
    if (
        parts.hpac_blob != base_parts.hpac_blob
        or parts.semantic_blob != base_parts.semantic_blob
        or parts.carrier_blob != base_parts.carrier_blob
        or parts.residual_payload != base_parts.residual_payload
        or parts.token_stream != token
    ):
        raise JO1Error("archive physical parse-back differs from the intended CP135 object")
    source_backend = Path(rc64["source_backend"]["path"])
    receiver_backend = CP135_RUNTIME / "runtime/entropy/rc64_backend.c"
    shipped_parseback = shipped_receiver_parseback(root, adapter, token, event_sha, spatial_sha)
    runtime = None
    if name == COMPOSED_NAME and not repeat:
        runtime = copy_runtime(root / "adapted_runtime", archive_payload)
    result = {
        "schema": "ddm_jo1_reclose_result.v1",
        "axis": AXIS,
        "name": name,
        "repeat": repeat,
        "scorer_run": False,
        "score_claim": False,
        "archive": archive_record,
        "zip_repeat": repeat_record,
        "zip_repeat_byte_identical": archive_record["sha256"] == repeat_record["sha256"],
        "delta_bytes_vs_cp135": archive_record["bytes"] - BASE_BYTES,
        "delta_rate_score": 25.0 * (archive_record["bytes"] - BASE_BYTES) / RATE_DENOMINATOR,
        "models": file_record(objects / "models.bin"),
        "residual": file_record(objects / "residual.compact.bin"),
        "token": file_record(objects / "tokens.rc64"),
        "member": file_record(objects / "p"),
        "probability_export": file_record(export_path),
        "rc64_result": file_record(rc64_path),
        "decoded_spatial_tokens": rc64["decoded_spatial_tokens"],
        "encoder_verifier_backend": file_record(source_backend),
        "shipped_receiver_backend": file_record(receiver_backend),
        "receiver_backend_source_identity": sha256_file(source_backend) == sha256_file(receiver_backend),
        "shipped_receiver_behavior_identity": True,
        "shipped_receiver_parseback": file_record(
            root / "receiver_state/SHIPPED_RC64_PARSEBACK.json"
        ),
        "shipped_decoded_spatial_tokens": shipped_parseback["decoded_spatial_tokens"],
        "physical_parseback": {
            "hp3_identity_to_cp135": True,
            "semantic_identity_to_cp135": True,
            "carrier_identity_to_cp135": True,
            "residual_identity_to_cp135": True,
            "token_identity_to_fresh_rc64": True,
        },
        "runtime": runtime,
        "receiver_closed": True,
        "all_payloads_retained": True,
    }
    atomic_json(root / "50_RECLOSE_RESULT.json", result)
    return result


def finalize(output: Path) -> dict[str, Any]:
    analysis = json.loads((output / "10_ANALYSIS.json").read_text())
    rows_by_id = {row["proposal_id"]: row for row in analysis["features"]}
    table = []
    for name, ids in analysis["candidate_sets"].items():
        receipt = json.loads((candidate_root(output, name) / "50_RECLOSE_RESULT.json").read_text())
        prepared = json.loads((candidate_root(output, name) / "20_MATERIALIZE_RESULT.json").read_text())
        projected_flips = sum(int(rows_by_id[value]["projected_robust_delta_flips"]) for value in ids)
        projected_pose = sum(float(rows_by_id[value]["pose_delta"]) for value in ids)
        row = {
            "name": name,
            "events": len(ids),
            "sites": sum(int(rows_by_id[value]["site_count"]) for value in ids),
            "archive": receipt["archive"],
            "delta_bytes_vs_cp135": receipt["delta_bytes_vs_cp135"],
            "delta_rate_score": receipt["delta_rate_score"],
            "singleton_n32_projected_robust_delta_flips_sum": projected_flips,
            "singleton_n32_pose_delta_sum": projected_pose,
            "projection_admissible": False,
            "projection_boundary": "JS7 exact n600 reversed the singleton robust-flip sign; sums are diagnostics only",
        }
        if name == "js7_selected_44":
            if prepared["spatial_raw_sha256"] != JS7_EXACT_TOKEN_SHA256:
                raise JO1Error("direct 44-event token plane differs from the exact JS7 rendered object")
            row["exact_output_identity_transfer"] = {
                "source_archive_bytes": JS7_EXACT_ARCHIVE_BYTES,
                "source_score": JS7_EXACT_SCORE,
                "same_semantic_token_sha256": JS7_EXACT_TOKEN_SHA256,
                "same_d_seg": JS7_EXACT_D_SEG,
                "same_d_pose": JS7_EXACT_D_POSE,
                "repriced_score": JS7_EXACT_SCORE
                - 25.0 * (JS7_EXACT_ARCHIVE_BYTES - receipt["archive"]["bytes"]) / RATE_DENOMINATOR,
                "boundary": "derived from byte-identical rendered token object; this archive was not rerun through upstream/evaluate.py",
            }
        table.append(row)
    primary = json.loads((candidate_root(output, COMPOSED_NAME) / "50_RECLOSE_RESULT.json").read_text())
    repeated = json.loads(
        (candidate_root(output, COMPOSED_NAME, repeat=True) / "50_RECLOSE_RESULT.json").read_text()
    )
    if primary["token"]["sha256"] != repeated["token"]["sha256"]:
        raise JO1Error("independent RC64 repeat token differs")
    if primary["archive"]["sha256"] != repeated["archive"]["sha256"]:
        raise JO1Error("independent composed archive repeat differs")
    composed_row = next(row for row in table if row["name"] == COMPOSED_NAME)
    optimistic_diagnostic_score = float(
        BASE_SCORE
        - np.sqrt(10.0 * BASE_D_POSE)
        + 100.0 * composed_row["singleton_n32_projected_robust_delta_flips_sum"] / TOTAL_EVENTS
        + composed_row["delta_rate_score"]
    )
    final = {
        "schema": "ddm_jo1_final_result.v1",
        "axis": AXIS,
        "scorer_run": False,
        "score_claim": False,
        "complete_container_table": table,
        "composed_archive": primary["archive"],
        "composed_runtime": primary["runtime"],
        "receiver_closed": primary["receiver_closed"],
        "independent_repeat": repeated["archive"],
        "independent_rc64_repeat_byte_identical": True,
        "independent_archive_repeat_byte_identical": True,
        "composed_optimistic_diagnostic": {
            "score_if_pose_were_zero_and_discredited_n32_flip_sum_transferred": optimistic_diagnostic_score,
            "target": 0.15,
            "crosses_target": optimistic_diagnostic_score < 0.15,
            "authority": "NON-AUTHORITY prefilter; not a bound and not a score projection",
        },
        "n600_seg_flip_validation": "PENDING_MAIN_CUDA_SCORER_OWNERSHIP",
        "exact_contest_eval": "QUEUED_TO_MAIN_T4_VALIDATION_THEN_EXACT",
        "base_score": BASE_SCORE,
        "base_d_seg": BASE_D_SEG,
        "base_d_pose": BASE_D_POSE,
        "all_payloads_retained": True,
    }
    atomic_json(output / "FINAL_RESULT.json", final)
    save_state(output, "complete", ["preflight", "analysis", "materialize", "reclose", "repeat", "finalize"])
    return final


def all_stages(output: Path) -> dict[str, Any]:
    preflight(output)
    save_state(output, "analysis", ["preflight"])
    analysis_result = analyze(output)
    save_state(output, "materialize", ["preflight", "analysis"])
    for name, proposal_ids in analysis_result["candidate_sets"].items():
        materialize_candidate(output, name, proposal_ids)
        reclose_candidate(output, name)
    save_state(output, "repeat", ["preflight", "analysis", "materialize", "reclose"])
    proposal_ids = analysis_result["candidate_sets"][COMPOSED_NAME]
    materialize_candidate(output, COMPOSED_NAME, proposal_ids, repeat=True)
    reclose_candidate(output, COMPOSED_NAME, repeat=True)
    return finalize(output)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=("preflight", "analyze", "finalize", "all"))
    value.add_argument("--output", type=Path, default=OUTPUT)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.stage == "preflight":
        result = preflight(args.output)
    elif args.stage == "analyze":
        preflight(args.output)
        result = analyze(args.output)
    elif args.stage == "finalize":
        result = finalize(args.output)
    else:
        result = all_stages(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
