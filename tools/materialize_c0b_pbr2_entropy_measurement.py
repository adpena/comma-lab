#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the research-only V9-conditioned PBR1/PBR2 entropy comparison.

This tool deliberately cannot build a candidate.  Both residuals losslessly
encode a caller-selected frozen GT-argmax slice and are forbidden candidate
payloads.  Its useful output is a reproducible conditional-entropy measurement
plus exact byte artifacts for subsequent research falsification.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import os
import platform
import shlex
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.witness_dsl.factorized_v9_predictor import (  # noqa: E402
    PREDICTOR_CONTRACT_ID,
    load_factorized_v9_predictor,
)
from tac.witness_dsl.predictor_bound_residual import packet_accounting as pbr1_accounting  # noqa: E402
from tac.witness_dsl.progressive_geometry_residual import (  # noqa: E402
    build_progressive_geometry_residual,
)
from tac.witness_dsl.progressive_geometry_residual import (  # noqa: E402
    packet_accounting as pbr2_accounting,
)
from tac.witness_dsl.progressive_v9_entropy_measurement import (  # noqa: E402
    apply_progressive_v9_entropy_measurement,
)

SCHEMA = "tac.c0b_pbr2_progressive_geometry_measurement.v2"


def _sha256_bytes(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_bound_file(path: Path, expected_sha256: str, label: str) -> tuple[Path, int]:
    value = path.resolve(strict=True)
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"{label} must be one regular file")
    size = value.stat().st_size
    if _sha256_file(value) != expected_sha256:
        raise ValueError(f"{label} SHA-256 custody mismatch")
    return value, size


def _durable_output(path: Path) -> Path:
    value = path.resolve()
    if value == Path("/tmp") or Path("/tmp") in value.parents:
        raise ValueError("durable measurement outputs cannot live under /tmp")
    value.parent.mkdir(parents=True, exist_ok=True)
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    target = _durable_output(path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _implementation_custody() -> dict[str, dict[str, Any]]:
    paths = (
        Path("src/tac/witness_dsl/factorized_v9_predictor.py"),
        Path("src/tac/witness_dsl/predictor_bound_residual.py"),
        Path("src/tac/witness_dsl/progressive_geometry_residual.py"),
        Path("src/tac/witness_dsl/progressive_v9_entropy_measurement.py"),
        Path("src/tac/witness_dsl/tests/test_predictor_bound_residual.py"),
        Path("src/tac/witness_dsl/tests/test_factorized_v9_predictor.py"),
        Path("src/tac/witness_dsl/tests/test_progressive_geometry_residual.py"),
        Path("tools/materialize_c0b_pbr2_entropy_measurement.py"),
    )
    return {
        str(path): {
            "bytes": (REPO / path).stat().st_size,
            "sha256": _sha256_file(REPO / path),
        }
        for path in paths
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-program", type=Path, required=True)
    parser.add_argument("--predictor-sha256", required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--gt-cache-sha256", required=True)
    parser.add_argument("--expected-pair-start", type=int, required=True)
    parser.add_argument("--expected-pair-count", type=int, required=True)
    parser.add_argument("--output-pbr1", type=Path, required=True)
    parser.add_argument("--output-pbr2", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()

    predictor_path, predictor_file_bytes = _require_bound_file(
        args.predictor_program,
        args.predictor_sha256,
        "predictor program",
    )
    cache_path, cache_file_bytes = _require_bound_file(
        args.gt_cache,
        args.gt_cache_sha256,
        "GT cache",
    )
    predictor = load_factorized_v9_predictor(
        predictor_path,
        expected_sha256=args.predictor_sha256,
    )
    if predictor.source_pair_start != args.expected_pair_start or predictor.pair_count != args.expected_pair_count:
        raise ValueError("predictor pair population differs from explicit expected coordinates")
    pair_stop = args.expected_pair_start + args.expected_pair_count
    with np.load(cache_path, allow_pickle=False) as bundle:
        if "lstars" not in bundle.files:
            raise ValueError("GT cache has no lstars member")
        target = np.asarray(
            bundle["lstars"][args.expected_pair_start : pair_stop],
            dtype=np.uint8,
        )
    expected_shape = (args.expected_pair_count, 384, 512)
    if target.shape != expected_shape or np.any(target > 4):
        raise ValueError("GT semantic target geometry or class alphabet differs")
    target = np.ascontiguousarray(target)
    predictor_labels = predictor.decode_all_semantics()
    mismatch_cells = int(np.count_nonzero(predictor_labels != target))

    started = time.perf_counter()
    pbr1 = predictor.build_pbr1(target)
    pbr1_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pbr2 = build_progressive_geometry_residual(
        predictor_program=predictor.program,
        predictor_contract_id=PREDICTOR_CONTRACT_ID,
        predictor_renderer_sha256=predictor.source_manifest_sha256,
        predictor_labels=predictor_labels,
        target_labels=target,
        source_pair_ids=predictor.source_pair_ids,
        target_semantic_lineage="frozen_gt_argmax",
    )
    pbr2_seconds = time.perf_counter() - started
    recovered = apply_progressive_v9_entropy_measurement(predictor.program, pbr2)
    if not np.array_equal(recovered, target):
        raise ValueError("fresh V9 receiver plus PBR2 did not recover the exact target")

    pbr1_output = _durable_output(args.output_pbr1)
    pbr2_output = _durable_output(args.output_pbr2)
    receipt_output = _durable_output(args.output_receipt)
    _atomic_bytes(pbr1_output, pbr1)
    _atomic_bytes(pbr2_output, pbr2)
    account1 = pbr1_accounting(pbr1)
    account2 = pbr2_accounting(pbr2)
    if account2["candidate_archive_admissible"] is not False:
        raise ValueError("PBR2 candidate prohibition drifted")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "authority_axis": "[research-only exact conditional semantic bytes; n64 formulation scope]",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "candidate_payload_allowed": False,
        "candidate_archive_blocker": account2["candidate_archive_blocker"],
        "purpose": "encoder-side conditional-entropy bound and grammar teacher only",
        "verdict_scope": (
            f"original factorized V9 predictor conditional residual, source pairs "
            f"{args.expected_pair_start}:{pair_stop}"
        ),
        "git_head_before_uncommitted_landing": _git_head(),
        "inputs": {
            "predictor_program": {
                "path": _relative(predictor_path),
                "bytes": predictor_file_bytes,
                "sha256": args.predictor_sha256,
                "contract_id": PREDICTOR_CONTRACT_ID,
                "renderer_sha256": predictor.source_manifest_sha256,
            },
            "gt_cache": {
                "path": _relative(cache_path),
                "bytes": cache_file_bytes,
                "sha256": args.gt_cache_sha256,
                "source_member": f"lstars[{args.expected_pair_start}:{pair_stop}]",
            },
        },
        "target": {
            "shape": list(target.shape),
            "dtype": str(target.dtype),
            "semantic_bytes": int(target.size),
            "semantic_sha256": _sha256_bytes(memoryview(target).cast("B")),
            "pbr2_reconstructs_exact_gt_argmax": True,
            "reconstructed_gt_argmax_bytes": int(target.size),
        },
        "exact_difference": {
            "mismatch_cells": mismatch_cells,
            "total_cells": int(target.size),
            "mismatch_fraction": mismatch_cells / int(target.size),
        },
        "pbr1": {
            **account1,
            "artifact_path": _relative(pbr1_output),
            "build_seconds": pbr1_seconds,
            "candidate_payload_allowed": False,
        },
        "pbr2": {
            **account2,
            "artifact_path": _relative(pbr2_output),
            "build_seconds": pbr2_seconds,
        },
        "comparison": {
            "pbr2_minus_pbr1_bytes": len(pbr2) - len(pbr1),
            "pbr2_bytes_saved_vs_pbr1": len(pbr1) - len(pbr2),
            "pbr2_fraction_of_pbr1": len(pbr2) / len(pbr1),
            "conditional_predictor_plus_pbr1_bytes": len(predictor.program) + len(pbr1),
            "conditional_predictor_plus_pbr2_bytes": len(predictor.program) + len(pbr2),
        },
        "receiver_closure": {
            "predictor_semantics_rederived_from_counted_program": True,
            "exact_target_recovered_without_gt_cache_at_decode": True,
            "fresh_process_regression_test": (
                "src/tac/witness_dsl/tests/test_factorized_v9_predictor.py::"
                "test_progressive_measurement_rederives_v9_semantics_in_fresh_process"
            ),
            "standalone_candidate_receiver": False,
            "candidate_payload_allowed": False,
        },
        "implementation": _implementation_custody(),
        "reproduction": {
            "argv": [sys.executable, *sys.argv],
            "shell_command": shlex.join([sys.executable, *sys.argv]),
            "rng_used": False,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "zlib_compile": zlib.ZLIB_VERSION,
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            "bz2_module": bz2.__name__,
            "lzma_module": lzma.__name__,
        },
        "limitations": [
            "n64 conditional-entropy measurement only; not n600 authority",
            "lossless frozen GT-argmax residual is forbidden from candidate archives",
            "no RGB preimage, inflate archive, upstream evaluation, or score authority",
            "repository receiver code is not yet a standalone inflate source bundle",
        ],
    }
    _atomic_json(receipt_output, receipt)
    print(json.dumps(receipt["comparison"], sort_keys=True))
    print(f"pbr2_sha256={_sha256_bytes(pbr2)}")
    print(f"receipt={receipt_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
