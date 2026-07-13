#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Free local JRD prefix smoke for PR110-lineage decoder coefficients.

The exhaustive response surface is a configurable first-pair prefix screen.
The selected combined candidate is then re-measured over ``--final-eval-pairs``
(600 for the Phase-1 handoff).  Every measurement uses the archive's own
decoder/parser and the frozen local CPU scorers through the exact resize/uint8
receiver chain.  This tool never calls ``upstream/evaluate.py`` and never edits
the canonical frontier pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.click_polish import (  # noqa: E402
    FrozenPacket,
    Renderer,
    Scorer,
    load_gt_targets,
)
from tac.contest_score import compute_contest_score  # noqa: E402
from tac.packet_compiler import jrd_pr110_runtime_custody as runtime_custody_module  # noqa: E402
from tac.packet_compiler.jrd_coefficient_prefix import (  # noqa: E402
    MAX_INT8_PREFIX_PLANES,
    PREFIX_FAMILIES,
    PrefixMeasurement,
    component_safe,
    generate_prefix_chain,
    select_best_byte_safe,
    select_last_safe_plane,
)
from tac.packet_compiler.jrd_pr110_coefficient_prefix import (  # noqa: E402
    Pr110CoefficientPacket,
)

RUNTIME_RELATIVE_FILES = runtime_custody_module.RUNTIME_RELATIVE_FILES

DEFAULT_SUBMISSION_DIR = REPO / (
    "experiments/results/pr110_payload_entropy_recode_20260610/submission_dir"
)
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache"
MIN_FREE_BYTES = 10 * 1024**3
EXPECTED_RAW_BYTES = 1200 * 874 * 1164 * 3
INFLATE_SCRATCH_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(canonical_json_bytes(payload))
    os.replace(temp, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def refuse_unsafe_scope(archive: Path, out_dir: Path) -> None:
    results_root = (REPO / "experiments/results").resolve()
    resolved_out = out_dir.resolve()
    if results_root not in resolved_out.parents:
        raise ValueError(f"output must be a child of {results_root}")
    if archive.resolve() == (out_dir / "candidate_archive.zip").resolve():
        raise ValueError("source archive and candidate path must differ")
    if str(resolved_out).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise ValueError("durable evidence must not be written under a transient directory")
    if shutil.disk_usage(resolved_out.parent).free < MIN_FREE_BYTES:
        raise RuntimeError("storage preflight refused: less than 10 GiB free")
    if os.environ.get("REVIEW_GATE_OVERRIDE") == "1":
        raise RuntimeError("unsafe review-gate override is set")


def _runtime_custody(submission_dir: Path) -> dict[str, Any]:
    return runtime_custody_module.runtime_custody(submission_dir, REPO)


def resolve_gt_cache_path(gt_cache: Path, n_pairs: int) -> Path:
    for cand_n in (1, 6, 24, 96, 200, 600):
        if cand_n < n_pairs:
            continue
        for stem in (f"gt_n{cand_n}.npz", f"gt_strided_n{cand_n}.npz"):
            path = gt_cache / stem
            if path.is_file():
                with np.load(path) as data:
                    if "lstars" in data.files and "gt_poses" in data.files:
                        return path
    raise FileNotFoundError(f"no GT cache in {gt_cache} covering {n_pairs} pairs")


def _file_custody(path: Path) -> dict[str, Any]:
    try:
        display_path = path.relative_to(REPO).as_posix()
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _proof_path_matches(recorded: object, expected: Path) -> bool:
    if not isinstance(recorded, str):
        return False
    path = Path(recorded)
    if not path.is_absolute():
        path = REPO / path
    return path.resolve() == expected.resolve()


def reusable_runtime_inflate_proof(
    proof: object,
    *,
    candidate_path: Path,
    submission_dir: Path,
    expected_raw_sha256: str,
    expected_raw_bytes: int,
) -> bool:
    """Fail closed unless a native/FIFO proof binds every custody surface."""

    if not isinstance(proof, dict):
        return False
    schema = proof.get("schema")
    if schema not in {
        "jrd_pr110_runtime_inflate_proof.v1",
        "jrd_pr110_runtime_inflate_fifo_proof.v1",
    }:
        return False
    runtime = proof.get("runtime_inflate_py")
    inflate_py = (submission_dir / "inflate.py").resolve()
    if not isinstance(runtime, dict) or not inflate_py.is_file():
        return False
    if proof.get("submission_runtime") != _runtime_custody(submission_dir):
        return False
    if not _proof_path_matches(runtime.get("path"), inflate_py):
        return False
    if runtime.get("bytes") != inflate_py.stat().st_size:
        return False
    if runtime.get("sha256") != sha256_file(inflate_py):
        return False
    if (
        proof.get("bit_exact") is not True
        or proof.get("candidate_sha256") != sha256_file(candidate_path)
        or proof.get("candidate_bytes") != candidate_path.stat().st_size
        or proof.get("expected_in_process_raw_sha256") != expected_raw_sha256
        or proof.get("expected_raw_bytes") != expected_raw_bytes
        or proof.get("scratch_cleaned_on_success") is not True
    ):
        return False
    if (
        schema == "jrd_pr110_runtime_inflate_fifo_proof.v1"
        and proof.get("streaming_fifo_no_bulk_raw_materialized") is not True
    ):
        return False
    passes = proof.get("passes")
    if not isinstance(passes, list) or len(passes) != 2:
        return False
    for expected_pass, row in enumerate(passes, start=1):
        if not isinstance(row, dict) or row.get("pass") != expected_pass:
            return False
        if row.get("returncode") != 0:
            return False
        if row.get("raw_sha256") != expected_raw_sha256:
            return False
        if row.get("raw_bytes") != expected_raw_bytes:
            return False
        if schema == "jrd_pr110_runtime_inflate_fifo_proof.v1" and (
            row.get("execution_error") is not None
            or row.get("reader_daemon") is not True
            or row.get("reader_alive_after_join") is not False
            or "error" in row
        ):
            return False
        log_path = row.get("log_path")
        if not isinstance(log_path, str):
            return False
        resolved_log = Path(log_path)
        if not resolved_log.is_absolute():
            resolved_log = REPO / resolved_log
        if not resolved_log.is_file():
            return False
        if resolved_log.resolve().parent != candidate_path.resolve().parent:
            return False
        if row.get("log_sha256") != sha256_file(resolved_log):
            return False
    return True


def _dependency_custody(gt_cache_path: Path) -> dict[str, Any]:
    paths = [
        gt_cache_path,
        REPO / "src/tac/click_polish.py",
        REPO / "src/tac/scorer.py",
        REPO / "upstream/evaluate.py",
        REPO / "upstream/modules.py",
        REPO / "upstream/models/posenet.safetensors",
        REPO / "upstream/models/segnet.safetensors",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"scorer/target custody is incomplete: {missing}")
    rows = [_file_custody(path) for path in paths]
    return {"files": rows, "tree_sha256": sha256_bytes(canonical_json_bytes(rows))}


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relevant_environment() -> dict[str, str | None]:
    names = (
        "PYTHONPATH",
        "PACT_PYTHON_BIN",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TAC_GOVERNED_ADMISSION",
        "REVIEW_GATE_OVERRIDE",
    )
    return {name: os.environ.get(name) for name in names}


def _runtime_versions() -> dict[str, Any]:
    packages = ("numpy", "torch", "safetensors", "Brotli", "constriction")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT-INSTALLED"
    executable = Path(sys.executable).resolve()
    return {
        "python_version": sys.version,
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "packages": versions,
    }


def run_fingerprint(
    *,
    archive: Path,
    submission_dir: Path,
    section_names: list[str],
    screen_eval_pairs: int,
    final_eval_pairs: int,
    gt_cache_path: Path,
) -> tuple[str, dict[str, Any]]:
    runtime = _runtime_custody(submission_dir)
    dependencies = _dependency_custody(gt_cache_path)
    payload = {
        "schema": "jrd_pr110_phase1_fingerprint.v2",
        "archive_path": archive.relative_to(REPO).as_posix(),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "submission_runtime": runtime,
        "scorer_and_target_dependencies": dependencies,
        "argv": sys.argv,
        "relevant_environment": _relevant_environment(),
        "runtime_versions": _runtime_versions(),
        "tool_sha256": sha256_file(Path(__file__)),
        "adapter_sha256": sha256_file(
            REPO / "src/tac/packet_compiler/jrd_pr110_coefficient_prefix.py"
        ),
        "oracle_sha256": sha256_file(
            REPO / "src/tac/packet_compiler/jrd_coefficient_prefix.py"
        ),
        "section_names": section_names,
        "families": list(PREFIX_FAMILIES),
        "max_planes": MAX_INT8_PREFIX_PLANES,
        "screen_eval_pairs": screen_eval_pairs,
        "final_eval_pairs": final_eval_pairs,
        "axis": "macOS-CPU advisory",
        "score_claim": False,
        "upstream_evaluate_py_run": False,
    }
    fingerprint = sha256_bytes(canonical_json_bytes(payload))
    payload["fingerprint_sha256"] = fingerprint
    payload["nonbinding_observations"] = {
        "git_head": _git_head(),
        "note": (
            "Git HEAD is receipt custody only. Relevant source/runtime bytes are SHA-bound; "
            "unrelated sibling commits do not invalidate a shared-tree resume."
        ),
    }
    return fingerprint, payload


def _load_resume(out_dir: Path, fingerprint: str) -> dict[str, Any]:
    path = out_dir / "resume/state.json"
    if not path.exists():
        return {
            "fingerprint": fingerprint,
            "rows": [],
            "completed_sections": [],
            "combined_rows": [],
            "final_measurements": {},
        }
    state = json.loads(path.read_text())
    if state.get("fingerprint") != fingerprint:
        raise RuntimeError("resume fingerprint changed")
    if not isinstance(state.get("rows"), list) or not isinstance(
        state.get("completed_sections"), list
    ):
        raise RuntimeError("resume state is incomplete")
    state.setdefault("combined_rows", [])
    state.setdefault("final_measurements", {})
    return state


def _write_resume(out_dir: Path, state: dict[str, Any]) -> None:
    atomic_json(out_dir / "resume/state.json", state)


def validate_controls(
    baseline: dict[str, Any],
    positive_repeat: dict[str, Any],
    negative_control: dict[str, Any],
) -> None:
    if any(
        positive_repeat[key] != baseline[key]
        for key in (
            "archive_zip_bytes",
            "archive_zip_sha256",
            "d_seg",
            "d_pose",
            "raw_sha256",
            "raw_bytes",
        )
    ):
        raise RuntimeError("positive-repeat noise floor is nonzero")
    if negative_control["archive_zip_sha256"] == baseline["archive_zip_sha256"]:
        raise RuntimeError("negative control did not change archive bytes")
    if (
        negative_control["d_seg"] == baseline["d_seg"]
        or negative_control["d_pose"] == baseline["d_pose"]
    ):
        raise RuntimeError("negative control did not separate both scorer components")


def validate_positive_repeat(
    baseline: dict[str, Any], positive_repeat: dict[str, Any]
) -> None:
    if any(
        positive_repeat[key] != baseline[key]
        for key in (
            "archive_zip_bytes",
            "archive_zip_sha256",
            "d_seg",
            "d_pose",
            "raw_sha256",
            "raw_bytes",
        )
    ):
        raise RuntimeError("matched final positive-repeat noise floor is nonzero")


def measure_final_controls(
    measure: Any, source_bytes: bytes, negative_bytes: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Measure and validate the n600 final meter's positive and negative canaries."""

    baseline = measure("baseline_final", source_bytes)
    positive = measure("baseline_final_positive_repeat", source_bytes)
    negative = measure("negative_all_decoder_coefficients_zero_final", negative_bytes)
    if any(int(row.get("eval_pairs", -1)) != 600 for row in (baseline, positive, negative)):
        raise RuntimeError("final controls must each be measured over exactly 600 pairs")
    validate_controls(baseline, positive, negative)
    return baseline, positive, negative


class ExactLocalMeter:
    """Archive bytes -> exact-R local CPU component measurement."""

    def __init__(
        self,
        *,
        submission_dir: Path,
        gt_cache: Path,
        out_dir: Path,
        max_pairs: int,
    ):
        self.submission_dir = submission_dir
        self.out_dir = out_dir
        self.scorer = Scorer(upstream_dir=REPO / "upstream", device="cpu")
        self.lstars, self.poses, self.gt_path = load_gt_targets(gt_cache, max_pairs)

    def measure(self, archive_bytes: bytes, *, eval_pairs: int, label: str) -> dict[str, Any]:
        scratch = self.out_dir / "scratch/current_candidate.zip"
        scratch.parent.mkdir(parents=True, exist_ok=True)
        atomic_bytes(scratch, archive_bytes)
        started = time.monotonic()
        try:
            packet = FrozenPacket.parse(scratch, self.submission_dir)
            renderer = Renderer(packet, device="cpu")
            raw_digest = hashlib.sha256()
            raw_bytes = 0
            dseg = np.empty(eval_pairs, np.float64)
            dpose = np.empty(eval_pairs, np.float64)
            batch_pairs = min(16, eval_pairs)
            for start in range(0, eval_pairs, batch_pairs):
                stop = min(start + batch_pairs, eval_pairs)
                indices = list(range(start, stop))
                frames = renderer.render(packet.Q0, indices, batch_pairs=batch_pairs)
                frame_bytes = frames.tobytes(order="C")
                raw_digest.update(frame_bytes)
                raw_bytes += len(frame_bytes)
                ds, dp = self.scorer.per_pair(
                    frames,
                    self.lstars[start:stop],
                    self.poses[start:stop],
                )
                dseg[start:stop] = ds
                dpose[start:stop] = dp
                del frames
        finally:
            scratch.unlink(missing_ok=True)
        d_seg = float(np.mean(dseg, dtype=np.float64))
        d_pose = float(np.mean(dpose, dtype=np.float64))
        archive_size = len(archive_bytes)
        return {
            "label": label,
            "eval_pairs": eval_pairs,
            "archive_zip_bytes": archive_size,
            "archive_zip_sha256": sha256_bytes(archive_bytes),
            "raw_sha256": raw_digest.hexdigest(),
            "raw_bytes": raw_bytes,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "implied_score_advisory": compute_contest_score(d_seg, d_pose, archive_size),
            "elapsed_seconds": time.monotonic() - started,
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "upstream_evaluate_py_run": False,
            "verdict_scope": (
                f"INSTANCE: first {eval_pairs} pair(s), named archive bytes, local exact-R"
            ),
            "review_status": "recovery-written-UNREVIEWED",
        }


def _prefix_row(
    row: dict[str, Any], *, section: str, family: str, bits_removed: int
) -> PrefixMeasurement:
    return PrefixMeasurement(
        section=section,
        family=family,  # type: ignore[arg-type]
        bits_removed=bits_removed,
        archive_bytes=int(row["archive_zip_bytes"]),
        d_seg=float(row["d_seg"]),
        d_pose=float(row["d_pose"]),
    )


def _section_summary(
    *, section: str, family: str, baseline: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    base = _prefix_row(baseline, section=section, family=family, bits_removed=0)
    measurements = [
        _prefix_row(
            row,
            section=section,
            family=family,
            bits_removed=int(row["bits_removed"]),
        )
        for row in rows
    ]
    last = select_last_safe_plane(
        measurements, base, seg_tolerance=0.0, pose_tolerance=0.0
    )
    best = select_best_byte_safe(
        measurements, base, seg_tolerance=0.0, pose_tolerance=0.0
    )

    def encode(choice: PrefixMeasurement | None) -> dict[str, Any] | None:
        if choice is None:
            return None
        return {
            "bits_removed": choice.bits_removed,
            "archive_zip_bytes": choice.archive_bytes,
            "archive_bytes_saved": base.archive_bytes - choice.archive_bytes,
            "delta_d_seg": choice.d_seg - base.d_seg,
            "delta_d_pose": choice.d_pose - base.d_pose,
        }

    n_pairs = int(baseline["eval_pairs"])
    screen_scope = (
        "INSTANCE: all 600 contest pairs; per-section n600 advisory safety verdict"
        if n_pairs == 600
        else (
            f"INSTANCE-SCREEN: first {n_pairs} pair(s) only; "
            "not an n600 per-section safety verdict"
        )
    )
    return {
        "section": section,
        "family": family,
        "last_safe": encode(last),
        "best_byte_safe": encode(best),
        "eval_pairs": n_pairs,
        "axis": baseline["axis"],
        "verdict_scope": screen_scope,
        "review_status": "recovery-written-UNREVIEWED",
    }


def _choose_inflate_scratch_root(required_bytes: int) -> Path:
    for root in INFLATE_SCRATCH_ROOTS:
        if root.is_dir() and shutil.disk_usage(root).free >= required_bytes:
            return root
    raise RuntimeError(
        "inflate storage preflight refused: no configured SSD tier has enough free space"
    )


def verify_candidate_runtime_inflate(
    *,
    candidate_path: Path,
    submission_dir: Path,
    receipt_dir: Path,
    expected_raw_sha256: str,
    expected_raw_bytes: int = EXPECTED_RAW_BYTES,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Execute the candidate's own inflate runtime twice and compare raw bytes."""

    required_bytes = int(expected_raw_bytes * 1.25) + candidate_path.stat().st_size
    root = scratch_root or _choose_inflate_scratch_root(required_bytes)
    if scratch_root is not None:
        root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < required_bytes:
        raise RuntimeError("inflate scratch root lacks required free space")
    candidate_sha = sha256_file(candidate_path)
    attempt_id = f"{time.time_ns()}_{os.getpid()}"
    candidate_scratch = (
        root / "jrd_pr110_task453_inflate_scratch" / candidate_sha[:16]
    )
    scratch = candidate_scratch / f"attempt_{attempt_id}"
    scratch.mkdir(parents=True, exist_ok=True)
    member_path = scratch / "x"
    passes: list[dict[str, Any]] = []
    proof = {
        "schema": "jrd_pr110_runtime_inflate_proof.v1",
        "attempt_id": attempt_id,
        "candidate_sha256": candidate_sha,
        "candidate_bytes": candidate_path.stat().st_size,
        "expected_in_process_raw_sha256": expected_raw_sha256,
        "expected_raw_bytes": expected_raw_bytes,
        "runtime_inflate_py": _file_custody(submission_dir / "inflate.py"),
        "submission_runtime": _runtime_custody(submission_dir),
        "scratch_root": str(root),
        "scratch_attempt_path": str(scratch),
        "passes": passes,
        "bit_exact": False,
        "scratch_cleaned_on_success": False,
        "failure_cleanup_disposition": None,
        "retained_failure_artifacts": [],
    }
    try:
        with zipfile.ZipFile(candidate_path, "r") as archive_zip:
            names = archive_zip.namelist()
            if names != ["x"]:
                raise RuntimeError(f"candidate must contain exactly member x, got {names}")
            atomic_bytes(member_path, archive_zip.read("x"))

        command_base = [
            sys.executable,
            str(submission_dir / "inflate.py"),
            str(member_path),
        ]
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"})
        for pass_index in (1, 2):
            raw_tmp = scratch / f"pass_{pass_index}.tmp.raw"
            raw_final = scratch / f"pass_{pass_index}.raw"
            raw_tmp.unlink(missing_ok=True)
            raw_final.unlink(missing_ok=True)
            command = [*command_base, str(raw_tmp)]
            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=submission_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            log_path = receipt_dir / (
                f"inflate_attempt_{attempt_id}_pass_{pass_index}.log"
            )
            atomic_bytes(
                log_path,
                (
                    f"command={json.dumps(command)}\n"
                    f"returncode={completed.returncode}\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}\n"
                ).encode(),
            )
            row = {
                "pass": pass_index,
                "command": command,
                "returncode": completed.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "log_path": (
                    log_path.relative_to(REPO).as_posix()
                    if REPO in log_path.parents
                    else str(log_path)
                ),
                "log_sha256": sha256_file(log_path),
            }
            passes.append(row)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"submission inflate pass {pass_index} failed rc={completed.returncode}"
                )
            os.replace(raw_tmp, raw_final)
            row.update(
                {
                "raw_bytes": raw_final.stat().st_size,
                "raw_sha256": sha256_file(raw_final),
                }
            )
            if row["raw_bytes"] != expected_raw_bytes:
                raise RuntimeError(
                    f"submission inflate pass {pass_index} emitted {row['raw_bytes']} bytes"
                )
            if row["raw_sha256"] != expected_raw_sha256:
                raise RuntimeError(
                    f"submission inflate pass {pass_index} differs from in-process exact-R bytes"
                )
            raw_final.unlink()
        if passes[0]["raw_sha256"] != passes[1]["raw_sha256"]:
            raise RuntimeError("submission inflate repeated outputs are not bit-identical")
        proof["bit_exact"] = True
        member_path.unlink(missing_ok=True)
        scratch.rmdir()
        if not any(candidate_scratch.iterdir()):
            candidate_scratch.rmdir()
        proof["scratch_cleaned_on_success"] = True
        atomic_json(
            receipt_dir / "checkpoints" / f"runtime_inflate_attempt_{attempt_id}.json",
            proof,
        )
        atomic_json(receipt_dir / "runtime_inflate_proof.json", proof)
        return proof
    except Exception as exc:
        proof["error"] = f"{type(exc).__name__}: {exc}"
        retained: list[dict[str, Any]] = []
        for path in sorted(item for item in scratch.rglob("*") if item.is_file()):
            retained.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        proof["retained_failure_artifacts"] = retained
        proof["failure_cleanup_disposition"] = (
            "retained fail-closed; later attempts use a new immutable attempt directory"
        )
        atomic_json(
            receipt_dir / "checkpoints" / f"runtime_inflate_attempt_{attempt_id}.json",
            proof,
        )
        atomic_json(receipt_dir / "runtime_inflate_proof.json", proof)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive.resolve()
    out_dir = args.out_dir.resolve()
    submission_dir = args.submission_dir.resolve()
    refuse_unsafe_scope(archive, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    packet = Pr110CoefficientPacket(archive, submission_dir)
    if packet.no_op_archive() != archive.read_bytes():
        raise RuntimeError("archive no-op decoder re-encode is not byte-identical")
    all_names = [section.name for section in packet.sections]
    section_names = args.section or all_names
    if len(section_names) != len(set(section_names)):
        raise ValueError("duplicate --section")
    for name in section_names:
        packet.section_by_name(name)

    gt_cache_path = resolve_gt_cache_path(
        args.gt_cache.resolve(), max(args.screen_eval_pairs, args.final_eval_pairs)
    )
    fingerprint, fingerprint_payload = run_fingerprint(
        archive=archive,
        submission_dir=submission_dir,
        section_names=section_names,
        screen_eval_pairs=args.screen_eval_pairs,
        final_eval_pairs=args.final_eval_pairs,
        gt_cache_path=gt_cache_path,
    )
    state = _load_resume(out_dir, fingerprint)
    atomic_json(out_dir / "run_fingerprint.json", fingerprint_payload)
    meter = ExactLocalMeter(
        submission_dir=submission_dir,
        gt_cache=args.gt_cache.resolve(),
        out_dir=out_dir,
        max_pairs=max(args.screen_eval_pairs, args.final_eval_pairs),
    )
    if Path(meter.gt_path).resolve() != gt_cache_path.resolve():
        raise RuntimeError("loaded GT cache differs from fingerprinted GT cache")
    source_bytes = archive.read_bytes()
    baseline = meter.measure(
        source_bytes, eval_pairs=args.screen_eval_pairs, label="baseline_screen"
    )
    positive_repeat = meter.measure(
        source_bytes, eval_pairs=args.screen_eval_pairs, label="positive_repeat_screen"
    )
    zero_replacements = {
        section.name: np.zeros(section.storage_shape, dtype=np.int8)
        for section in packet.sections
    }
    negative_bytes = packet.repack_archive_replacements(zero_replacements)
    negative_control = meter.measure(
        negative_bytes,
        eval_pairs=args.screen_eval_pairs,
        label="negative_all_decoder_coefficients_zero",
    )
    validate_controls(baseline, positive_repeat, negative_control)

    indexed = {
        (row["section"], row["family"], int(row["bits_removed"])): row
        for row in state["rows"]
    }
    summaries: list[dict[str, Any]] = []
    for section_name in section_names:
        section = packet.section_by_name(section_name)
        original = packet.read_section(section)
        for family in PREFIX_FAMILIES:
            chain = generate_prefix_chain(original, family=family)
            family_rows: list[dict[str, Any]] = []
            for bits_removed in range(1, MAX_INT8_PREFIX_PLANES + 1):
                key = (section_name, family, bits_removed)
                row = indexed.get(key)
                if row is None:
                    candidate = packet.repack_archive(section, chain[bits_removed])
                    row = meter.measure(
                        candidate,
                        eval_pairs=args.screen_eval_pairs,
                        label=f"{section_name}:{family}:plane{bits_removed}",
                    )
                    row.update(
                        {
                            "section": section_name,
                            "family": family,
                            "bits_removed": bits_removed,
                        }
                    )
                    state["rows"].append(row)
                    indexed[key] = row
                    _write_resume(out_dir, state)
                family_rows.append(row)
            summaries.append(
                _section_summary(
                    section=section_name,
                    family=family,
                    baseline=baseline,
                    rows=family_rows,
                )
            )
        if section_name not in state["completed_sections"]:
            state["completed_sections"].append(section_name)
            checkpoint = {
                "schema": "jrd_pr110_section_checkpoint.v1",
                "fingerprint": fingerprint,
                "section": section_name,
                "rows": [row for row in state["rows"] if row["section"] == section_name],
            }
            atomic_json(
                out_dir / "checkpoints" / f"section_{section.storage_position:02d}.json",
                checkpoint,
            )
            _write_resume(out_dir, state)

    best_choices = [
        summary
        for summary in summaries
        if summary["best_byte_safe"] is not None
        and summary["best_byte_safe"]["archive_bytes_saved"] > 0
    ]
    best_choices.sort(
        key=lambda item: (-item["best_byte_safe"]["archive_bytes_saved"], item["section"], item["family"])
    )
    accepted: dict[str, np.ndarray] = {}
    accepted_rows: list[dict[str, Any]] = []
    current = baseline
    combined_index = {
        (
            row["section"],
            row["family"],
            int(row["bits_removed"]),
        ): row
        for row in state["combined_rows"]
    }
    for step_index, choice in enumerate(best_choices):
        name = choice["section"]
        if name in accepted:
            continue
        section = packet.section_by_name(name)
        q = packet.read_section(section)
        chain = generate_prefix_chain(q, family=choice["family"])
        bits_removed = int(choice["best_byte_safe"]["bits_removed"])
        proposal = {**accepted, name: chain[bits_removed]}
        candidate = packet.repack_archive_replacements(proposal)
        combined_key = (name, choice["family"], bits_removed)
        cached = combined_index.get(combined_key)
        if cached is None:
            measured = meter.measure(
                candidate,
                eval_pairs=args.screen_eval_pairs,
                label=f"combined_screen_add:{name}:{choice['family']}",
            )
        else:
            measured = cached["measurement"]
            if measured["archive_zip_sha256"] != sha256_bytes(candidate):
                raise RuntimeError("combined-stage resume candidate changed")
        baseline_measurement = _prefix_row(
            baseline, section="combined", family="uniform", bits_removed=0
        )
        proposal_measurement = _prefix_row(
            measured, section="combined", family="uniform", bits_removed=1
        )
        admitted = component_safe(
            proposal_measurement,
            baseline_measurement,
            seg_tolerance=0.0,
            pose_tolerance=0.0,
        ) and measured["archive_zip_bytes"] < current["archive_zip_bytes"]
        combined_row = {
            "section": name,
            "family": choice["family"],
            "bits_removed": bits_removed,
            "admitted": admitted,
            "measurement": measured,
            "eval_pairs": args.screen_eval_pairs,
            "axis": measured["axis"],
            "verdict_scope": (
                f"INSTANCE-SCREEN: first {args.screen_eval_pairs} pair(s), combined "
                "greedy admission only; final n600 gate is separate"
            ),
            "review_status": "recovery-written-UNREVIEWED",
        }
        if cached is None:
            state["combined_rows"].append(combined_row)
            combined_index[combined_key] = combined_row
            atomic_json(
                out_dir / "checkpoints" / f"combined_step_{step_index:02d}.json",
                {
                    "schema": "jrd_pr110_combined_checkpoint.v1",
                    "fingerprint": fingerprint,
                    "step_index": step_index,
                    "row": combined_row,
                },
            )
            _write_resume(out_dir, state)
        elif bool(cached["admitted"]) != admitted:
            raise RuntimeError("combined-stage resume admission verdict changed")
        accepted_rows.append(combined_row)
        if admitted:
            accepted = proposal
            current = measured

    selected_bytes = (
        packet.repack_archive_replacements(accepted) if accepted else source_bytes
    )
    candidate_path = out_dir / "candidate_archive.zip"
    atomic_bytes(candidate_path, selected_bytes)
    selected_noop = Pr110CoefficientPacket(candidate_path, submission_dir).no_op_archive()
    if selected_noop != selected_bytes:
        raise RuntimeError("selected candidate does not reparse/re-encode byte-identically")

    def cached_final_measurement(label: str, payload: bytes) -> dict[str, Any]:
        cached = state["final_measurements"].get(label)
        payload_sha = sha256_bytes(payload)
        if cached is not None:
            if (
                cached["archive_zip_sha256"] != payload_sha
                or int(cached["eval_pairs"]) != args.final_eval_pairs
            ):
                raise RuntimeError(f"final-stage resume input changed for {label}")
            return cached
        measured = meter.measure(
            payload, eval_pairs=args.final_eval_pairs, label=label
        )
        state["final_measurements"][label] = measured
        atomic_json(
            out_dir / "checkpoints" / f"{label}.json",
            {
                "schema": "jrd_pr110_final_measurement_checkpoint.v1",
                "fingerprint": fingerprint,
                "measurement": measured,
            },
        )
        _write_resume(out_dir, state)
        return measured

    baseline_final, baseline_final_repeat, negative_control_final = (
        measure_final_controls(
            cached_final_measurement, source_bytes, negative_bytes
        )
    )
    if selected_bytes == source_bytes:
        proposed_selected_final = baseline_final.copy()
        proposed_selected_final["label"] = "selected_final_byte_identical_to_baseline"
    else:
        proposed_selected_final = cached_final_measurement(
            "selected_final", selected_bytes
        )
    final_safe = (
        proposed_selected_final["d_seg"] <= baseline_final["d_seg"]
        and proposed_selected_final["d_pose"] <= baseline_final["d_pose"]
    )
    exact_bytes_saved = (
        baseline_final["archive_zip_bytes"]
        - proposed_selected_final["archive_zip_bytes"]
    )
    paid_eval_ready = bool(exact_bytes_saved > 0 and final_safe)
    selected_final = proposed_selected_final
    if not paid_eval_ready and selected_bytes != source_bytes:
        # Fail closed: retain the measured unsafe proposal separately and hand
        # Phase 2 only a byte-identical no-op candidate.
        unsafe_path = out_dir / "screen_selected_rejected_archive.zip"
        atomic_bytes(unsafe_path, selected_bytes)
        atomic_bytes(candidate_path, source_bytes)
        selected_final = baseline_final.copy()
        selected_final["label"] = "fail_closed_baseline_candidate"
        exact_bytes_saved = 0

    runtime_inflate_proof_path = out_dir / "runtime_inflate_proof.json"
    if runtime_inflate_proof_path.exists():
        runtime_inflate_proof = json.loads(runtime_inflate_proof_path.read_text())
        reusable_runtime_proof = reusable_runtime_inflate_proof(
            runtime_inflate_proof,
            candidate_path=candidate_path,
            submission_dir=submission_dir,
            expected_raw_sha256=selected_final["raw_sha256"],
            expected_raw_bytes=EXPECTED_RAW_BYTES,
        )
        if not reusable_runtime_proof:
            failed_proof_path = out_dir / "checkpoints" / (
                f"runtime_inflate_nonreusable_{time.time_ns()}.json"
            )
            failed_proof_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(runtime_inflate_proof_path, failed_proof_path)
            runtime_inflate_proof = verify_candidate_runtime_inflate(
                candidate_path=candidate_path,
                submission_dir=submission_dir,
                receipt_dir=out_dir,
                expected_raw_sha256=selected_final["raw_sha256"],
            )
    else:
        runtime_inflate_proof = verify_candidate_runtime_inflate(
            candidate_path=candidate_path,
            submission_dir=submission_dir,
            receipt_dir=out_dir,
            expected_raw_sha256=selected_final["raw_sha256"],
        )
    paid_eval_ready = bool(
        paid_eval_ready
        and args.final_eval_pairs == 600
        and runtime_inflate_proof.get("bit_exact") is True
    )
    state["runtime_inflate_proof"] = runtime_inflate_proof
    _write_resume(out_dir, state)

    receipt = {
        "schema": "jrd_pr110_phase1_measurement.v1",
        "task_id": "task453_pointer_jrd_pr110_phase1_20260712",
        "fingerprint": fingerprint,
        "run_fingerprint": fingerprint_payload,
        "source": {
            "path": archive.relative_to(REPO).as_posix(),
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
        },
        "candidate": {
            "path": candidate_path.relative_to(REPO).as_posix(),
            "sha256": sha256_file(candidate_path),
            "bytes": candidate_path.stat().st_size,
        },
        "baseline_screen": baseline,
        "positive_repeat_screen": positive_repeat,
        "negative_control_screen": negative_control,
        "baseline_final": baseline_final,
        "baseline_final_positive_repeat": baseline_final_repeat,
        "negative_control_final": negative_control_final,
        "proposed_selected_final": proposed_selected_final,
        "selected_final": selected_final,
        "screen_eval_pairs": args.screen_eval_pairs,
        "final_eval_pairs": args.final_eval_pairs,
        "per_section_summary": summaries,
        "combined_screen_steps": accepted_rows,
        "runtime_inflate_proof": runtime_inflate_proof,
        "accepted_sections": [
            {
                "section": row["section"],
                "family": row["family"],
                "bits_removed": row["bits_removed"],
            }
            for row in accepted_rows
            if row["admitted"]
        ],
        "delta": {
            "archive_bytes_saved": exact_bytes_saved,
            "delta_d_seg": selected_final["d_seg"] - baseline_final["d_seg"],
            "delta_d_pose": selected_final["d_pose"] - baseline_final["d_pose"],
            "projected_delta_s_rate_only": -25.0 * exact_bytes_saved / 37_545_489,
        },
        "recommendation": "PAID-EVAL-READY" if paid_eval_ready else "NO-GO",
        "verdict_scope": "INSTANCE: named archive bytes; 28 signed-int8 decoder tensors; local macOS-CPU exact-R advisory",
        "review_status": "recovery-written-UNREVIEWED",
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "upstream_evaluate_py_run": False,
            "paid_dispatch": False,
            "canonical_frontier_pointer_edited": False,
        },
        "custody": packet.custody(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "gt_cache": str(meter.gt_path),
            "argv": sys.argv,
            "git_head": fingerprint_payload["nonbinding_observations"]["git_head"],
            "relevant_environment": fingerprint_payload["relevant_environment"],
            "free_bytes_at_start": shutil.disk_usage(out_dir).free,
        },
    }
    atomic_json(out_dir / "measurement_receipt.json", receipt)
    atomic_json(
        out_dir / "section_precision_response_curves.json",
        {
            "schema": "jrd_pr110_section_response_curves.v1",
            "fingerprint": fingerprint,
            "baseline": baseline,
            "rows": state["rows"],
            "summaries": summaries,
        },
    )
    shutil.rmtree(out_dir / "scratch", ignore_errors=True)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--screen-eval-pairs", type=int, default=1)
    parser.add_argument("--final-eval-pairs", type=int, default=600)
    parser.add_argument("--section", action="append", default=[])
    args = parser.parse_args(argv)
    value = args.screen_eval_pairs
    if isinstance(value, bool) or not 1 <= value <= 600:
        parser.error("--screen-eval-pairs must be in [1,600]")
    if args.final_eval_pairs != 600:
        parser.error("--final-eval-pairs must equal 600 for Phase-1 readiness")
    return args


def main(argv: list[str] | None = None) -> int:
    receipt = run(parse_args(argv))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
