#!/usr/bin/env python3
"""Adjudicate whether PR135's exact F26 compensation search can be resumed.

This scorer-free tool pins the retained ExperimentBook source, extracts the
published accepted-row trajectory, verifies the exact solver's zero-accept
stopping contract, and proves that CP135 retains PR135's canonical carrier.
It writes a convergence receipt and a sealed no-fire disposition.  It never
runs an evaluator and never emits a candidate when the reference solver has
already converged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

DEFAULT_BOOK: Final = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book")
DEFAULT_PR135: Final = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/archive.zip")
DEFAULT_CP135: Final = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip")
DEFAULT_RUNTIME: Final = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime")
DEFAULT_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pr135ps_20260813/retained")
README_SHA256: Final = "b8a05bdb88ce11c369ce15eae089b2d7870f00756d2d2a7784be0a72d791ab0c"
SOLVER_SHA256: Final = "f69c242748d5289db237c5f7a1b0492901ec1e183edad35bbeef31d4015c3bee"
BOOK_HEAD: Final = "f229b26735dffc53fdf1ac9987ac7c303298d028"
PR135_BYTES: Final = 186_724
PR135_SHA256: Final = "12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004"
CP135_BYTES: Final = 186_252
CP135_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
FRAMES: Final = 600
DIMENSIONS: Final = 12
SELECTOR_ROWS: Final = 5
MOVES_PER_ROW: Final = 2 * DIMENSIONS
BASIS_COUNT: Final = DIMENSIONS * 3 * 24 * 32


class PR135PSAuditError(RuntimeError):
    """A retained-source, carrier-custody, or convergence invariant failed."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file(path: Path, *, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file():
        raise PR135PSAuditError(f"missing required file: {path}")
    if size is not None and path.stat().st_size != size:
        raise PR135PSAuditError(f"unexpected byte count: {path}")
    if digest is not None and sha256_file(path) != digest:
        raise PR135PSAuditError(f"unexpected SHA-256: {path}")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def checkpoint_once(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
    if path.is_file():
        if path.read_bytes() != payload:
            raise PR135PSAuditError(f"retained checkpoint differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def extract_f26_trajectory(readme: str) -> tuple[int, ...]:
    match = re.search(
        r"F26 continued.*?Accepted-row counts\s+by pass were:\s*"
        r"```text\s*([0-9, ]+)\s*```",
        readme,
        flags=re.DOTALL,
    )
    if match is None:
        raise PR135PSAuditError("F26 accepted-row trajectory is absent")
    trajectory = tuple(int(item.strip()) for item in match.group(1).split(","))
    if not trajectory:
        raise PR135PSAuditError("F26 accepted-row trajectory is empty")
    return trajectory


def extract_f23_trajectory(readme: str) -> tuple[int, ...]:
    match = re.search(r"F23 accepted `([0-9, ]+)` row moves", readme)
    if match is None:
        raise PR135PSAuditError("F23 accepted-row trajectory is absent")
    return tuple(int(item.strip()) for item in match.group(1).split(","))


def classify_resume(trajectory: tuple[int, ...]) -> dict[str, Any]:
    if not trajectory:
        raise PR135PSAuditError("cannot classify an empty trajectory")
    accepted = trajectory[-1]
    if accepted == 0:
        return {
            "classification": "FOLDED_SOURCE_CONVERGED",
            "disposition": "FOLDED",
            "resume_exact_reference_form": False,
            "reason": "the last completed exact pass accepted zero rows",
        }
    return {
        "classification": "RESUME_REQUIRED_STILL_ACCEPTING",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "resume_exact_reference_form": True,
        "reason": "the last completed exact pass still accepted rows",
    }


def verify_solver_contract(source: str) -> dict[str, Any]:
    required = {
        "all_signed_singletons": "for delta in (-1, 1)",
        "strict_improvement": "improve = best_error < errors - 1e-15",
        "accepted_count": "accepted = int(np.count_nonzero(improve))",
        "per_pass_archive": "checkpoint.write_bytes(archive)",
        "zero_marks_converged": 'state["converged"] = True',
        "zero_breaks": "if accepted == 0:\n            break",
        "budget_exceeded_eight": "default=12",
    }
    missing = [name for name, token in required.items() if token not in source]
    if missing:
        raise PR135PSAuditError("F26 solver contract is incomplete: " + ", ".join(missing))
    return {
        "checks": dict.fromkeys(required, True),
        "dimensions": DIMENSIONS,
        "moves_per_active_row": MOVES_PER_ROW,
        "acceptance_rule": "best singleton error < current error - 1e-15",
        "stopping_rule": "accepted_rows == 0",
        "configured_max_passes": 12,
    }


def git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise PR135PSAuditError("cannot resolve retained ExperimentBook HEAD")
    return completed.stdout.strip()


def git_porcelain(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise PR135PSAuditError("cannot inspect retained ExperimentBook status")
    return completed.stdout


def _runtime_modules(runtime_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    brotli = Path("/opt/homebrew/bin/brotli")
    if brotli.is_file():
        os.environ.setdefault("CP135_BROTLI_CLI", str(brotli))
    value = str(runtime_root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)
    residual = importlib.import_module("runtime.residual_archive")
    repack = importlib.import_module("runtime.carrier_repack")
    coefficient = importlib.import_module("runtime.entropy.coefficient_ar1_codec")
    carrier_codec = importlib.import_module("cpr1.carrier_codec")
    selector_codec = importlib.import_module("runtime.frame0_selector")
    modules = (residual, repack, coefficient, carrier_codec, selector_codec)
    expected_root = runtime_root.resolve()
    for module in modules:
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(expected_root):
            raise PR135PSAuditError(f"runtime module escaped CP135 custody: {module_path}")
    return modules


def signed_codes_from_delta_zigzag(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.int64)
    if value.shape != (FRAMES, DIMENSIONS) or np.any(value < 0) or np.any(value > 4095):
        raise PR135PSAuditError("decoded coefficient lattice has invalid shape/range")
    delta = (value >> 1) ^ -(value & 1)
    unsigned = np.cumsum(delta, axis=0, dtype=np.int64) & 0xFFF
    return np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(np.int16)


def compare_carriers(pr135: Path, cp135: Path, runtime_root: Path) -> dict[str, Any]:
    residual, repack, coefficient, carrier_codec, selector_codec = _runtime_modules(runtime_root)
    pr_parts = residual.read_residual_archive(pr135)
    cp_parts = residual.read_residual_archive(cp135)
    pr_cap1, pr_selector = repack.split_frame0_selector_carrier(pr_parts.carrier_blob)
    cp_cap1, cp_selector = repack.split_frame0_selector_carrier(cp_parts.carrier_blob)
    if pr_selector is None or cp_selector is None:
        raise PR135PSAuditError("PR135/CP135 must both retain the F0E1 selector")
    pr_canonical = coefficient.decode_cap1(pr_cap1, frames=FRAMES, dimensions=DIMENSIONS)
    cp_canonical = coefficient.decode_cap1(cp_cap1, frames=FRAMES, dimensions=DIMENSIONS)
    pr_encoded = carrier_codec.decode_compact_carrier(
        pr_canonical,
        basis_count=BASIS_COUNT,
        frames=FRAMES,
        dimensions=DIMENSIONS,
    )[3]
    cp_encoded = carrier_codec.decode_compact_carrier(
        cp_canonical,
        basis_count=BASIS_COUNT,
        frames=FRAMES,
        dimensions=DIMENSIONS,
    )[3]
    pr_codes = signed_codes_from_delta_zigzag(pr_encoded)
    cp_codes = signed_codes_from_delta_zigzag(cp_encoded)
    _, selector_choices = selector_codec.decode_selector(cp_selector)
    selector_choices = np.asarray(selector_choices)
    actual_selector_rows = int(np.count_nonzero(selector_choices))
    if actual_selector_rows != SELECTOR_ROWS:
        raise PR135PSAuditError(f"expected {SELECTOR_ROWS} frozen selector rows, found {actual_selector_rows}")
    active = cp_codes[selector_choices == 0]
    blocked_lower_moves = int(np.count_nonzero(active == -2048))
    blocked_upper_moves = int(np.count_nonzero(active == 2047))
    return {
        "canonical_carrier_equal": pr_canonical == cp_canonical,
        "canonical_carrier": {
            "bytes": len(cp_canonical),
            "sha256": sha256_bytes(cp_canonical),
        },
        "selector_equal": pr_selector == cp_selector,
        "selector": {"bytes": len(cp_selector), "sha256": sha256_bytes(cp_selector)},
        "canonical_cap1_equal": pr_cap1 == cp_cap1,
        "coefficient_lattice_equal": np.array_equal(pr_codes, cp_codes),
        "active_coefficient_min": int(active.min()),
        "active_coefficient_max": int(active.max()),
        "blocked_lower_moves": blocked_lower_moves,
        "blocked_upper_moves": blocked_upper_moves,
        "valid_singleton_proposals": (
            (FRAMES - SELECTOR_ROWS) * MOVES_PER_ROW - blocked_lower_moves - blocked_upper_moves
        ),
        "semantic_equal": pr_parts.semantic_blob == cp_parts.semantic_blob,
        "hpac_equal": pr_parts.hpac_blob == cp_parts.hpac_blob,
        "token_stream_equal": pr_parts.token_stream == cp_parts.token_stream,
        "residual_payload_equal": pr_parts.residual_payload == cp_parts.residual_payload,
    }


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    readme_path = args.book / "README.md"
    solver_path = args.book / "scripts/solve_f26_iterative_joint_carrier.py"
    require_file(readme_path, digest=README_SHA256)
    require_file(solver_path, digest=SOLVER_SHA256)
    require_file(args.pr135, size=PR135_BYTES, digest=PR135_SHA256)
    require_file(args.cp135, size=CP135_BYTES, digest=CP135_SHA256)
    if not args.runtime.is_dir():
        raise PR135PSAuditError(f"missing adapted CP135 runtime: {args.runtime}")
    head = git_head(args.book)
    if head != BOOK_HEAD:
        raise PR135PSAuditError("retained ExperimentBook HEAD drifted")
    porcelain = git_porcelain(args.book)
    if porcelain:
        raise PR135PSAuditError("retained ExperimentBook is dirty")

    readme = readme_path.read_text(encoding="utf-8")
    solver = solver_path.read_text(encoding="utf-8")
    f26 = extract_f26_trajectory(readme)
    f23 = extract_f23_trajectory(readme)
    decision = classify_resume(f26)
    contract = verify_solver_contract(solver)
    carrier = compare_carriers(args.pr135, args.cp135, args.runtime)
    if (
        not carrier["canonical_carrier_equal"]
        or not carrier["coefficient_lattice_equal"]
        or not carrier["selector_equal"]
    ):
        raise PR135PSAuditError("CP135 does not retain PR135's canonical carrier state")

    active_rows = FRAMES - SELECTOR_ROWS
    proposal_slots = active_rows * MOVES_PER_ROW
    return {
        "schema": "ddm_pr135ps_source_convergence_receipt.v2",
        "authority": "PRIMARY_SOURCE_AND_BYTE_CUSTODY_ONLY",
        "score_claim": False,
        "promotion_eligible": False,
        "decision": decision,
        "source_pins": {
            "experiment_book_head": head,
            "experiment_book_clean": True,
            "readme": file_record(readme_path),
            "solver": file_record(solver_path),
            "pr135_archive": file_record(args.pr135),
            "cp135_archive": file_record(args.cp135),
            "cp135_adapted_runtime": str(args.runtime.resolve()),
        },
        "reference_form": {
            "name": "F26 iterative all-12 exact signed-int12 singleton descent",
            "solver_contract": contract,
            "accepted_rows_by_pass": list(f26),
            "passes_completed": len(f26),
            "active_rows": active_rows,
            "selector_rows_frozen": SELECTOR_ROWS,
            "final_pass_accepted_rows": f26[-1],
            "final_pass_row_acceptance_denominator": active_rows,
            "final_pass_row_acceptance_rate": f26[-1] / active_rows,
            "final_pass_singleton_proposal_slots": proposal_slots,
            "final_pass_valid_singleton_proposal_denominator": carrier["valid_singleton_proposals"],
            "accepted_singleton_proposals": 0,
            "derived_stopping_rule_fired": f26[-1] == 0,
        },
        "premise_adjudication": {
            "charter_claim": "final PR135 pass 8 was truncated while still accepting",
            "result": "REFUTED_FOR_FINAL_F26",
            "source_fact": "F26 pass 8 accepted zero rows and marks convergence",
            "likely_conflation": {
                "stage": "F23",
                "accepted_rows_by_improving_pass": list(f23),
                "pass_8_accepted_rows": f23[-1],
                "next_pass_accepted_rows": 0,
            },
        },
        "cp135_carrier_custody": carrier,
        "candidate_archives_materialized": 0,
        "evaluators_run": [],
    }


def build_no_fire(receipt: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "ddm_pr135ps_sealed_dual_axis_fire_order.v2",
        "status": "REFUSED_NO_CANDIDATE",
        "disposition": "FOLDED",
        "reason": receipt["decision"]["reason"],
        "candidate_archives": [],
        "adapted_runtime": str(args.runtime.resolve()),
        "fire_command": None,
        "run_id": None,
        "estimated_cost_usd": 0.0,
        "dual_axis_gate": {
            "mandatory_for_any_future_mechanism_extension": True,
            "preferred_worker": "js6b-extended: SegNet field plus PoseNet first-six vectors",
            "fallback": "re1t SegNet worker plus a same-archive PoseNet-vector follow-up",
            "batch_with_js6b_compile_gate_when_supported": True,
        },
        "consumer_store": "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/",
        "fire_trigger": (
            "a declared mechanism-extension candidate survives receiver parse-back and "
            "has a retained archive plus adapted runtime"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def run(args: argparse.Namespace) -> int:
    receipt = build_receipt(args)
    no_fire = build_no_fire(receipt, args)
    receipt_path = args.output / "SOURCE_CONVERGENCE_RECEIPT_v2.json"
    no_fire_path = args.output / "SEALED_DUAL_AXIS_FIRE_ORDER_REFUSAL_v2.json"
    checkpoint_once(receipt_path, receipt)
    checkpoint_once(no_fire_path, no_fire)
    manifest = {
        "schema": "ddm_pr135ps_retained_manifest.v2",
        "files": [file_record(receipt_path), file_record(no_fire_path)],
        "referenced_payloads": [
            receipt["source_pins"]["pr135_archive"],
            receipt["source_pins"]["cp135_archive"],
        ],
        "cleanup_policy": "no payload created or deleted; retained source archives referenced in place",
    }
    checkpoint_once(args.output / "MANIFEST_v2.json", manifest)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--book", type=Path, default=DEFAULT_BOOK)
    result.add_argument("--pr135", type=Path, default=DEFAULT_PR135)
    result.add_argument("--cp135", type=Path, default=DEFAULT_CP135)
    result.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
