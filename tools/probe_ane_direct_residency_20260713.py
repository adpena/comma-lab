#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and run the safe direct-ANE private-runtime introspection probe."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    base_receipt,
    sha256_file,
)

SOURCE = REPO / "tools/native/ane_direct_residency_probe_20260713.m"
DEFAULT_OUT = REPO / "experiments/results/ane_unlock_followup_20260713/direct_ane_probe.json"
FRAMEWORKS = (
    Path("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"),
    Path("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    compile_command: list[str]
    with tempfile.TemporaryDirectory(prefix="pact_ane_direct_probe_") as scratch:
        binary = Path(scratch) / "ane_direct_probe"
        compile_command = [
            "/usr/bin/clang",
            "-fblocks",
            "-framework",
            "Foundation",
            str(SOURCE),
            "-o",
            str(binary),
        ]
        built = subprocess.run(compile_command, capture_output=True, text=True, check=False)
        runtime: dict[str, object] = {}
        run = None
        if built.returncode == 0:
            run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
            try:
                runtime = json.loads(run.stdout) if run.returncode == 0 else {}
            except json.JSONDecodeError as exc:
                runtime = {"parse_error": str(exc), "stdout_prefix": run.stdout[:2000]}
        backward_candidates = runtime.get("backward_selector_candidates", [])
        direct_forward = bool(runtime.get("direct_forward_executed", False))
        backward_executed = bool(runtime.get("backward_vjp_executed", False))
        receipt = base_receipt(
            schema="ane_direct_residency_probe.v1",
            written_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            platform=platform.platform(),
            machine=platform.machine(),
            source_path=str(SOURCE.relative_to(REPO)),
            source_sha256=sha256_file(SOURCE),
            compile_command=["<SCRATCH>" if str(token).startswith(scratch) else token for token in compile_command],
            compile_returncode=built.returncode,
            compile_stdout=built.stdout[-4000:],
            compile_stderr=built.stderr[-4000:],
            binary_sha256=sha256_file(binary) if binary.exists() else None,
            run_returncode=run.returncode if run is not None else None,
            run_stderr=run.stderr[-4000:] if run is not None else "",
            framework_files=[
                {
                    "path": str(path),
                    "framework_directory_present": path.parent.exists(),
                    "standalone_binary_present": path.exists(),
                    "sha256": sha256_file(path) if path.exists() else None,
                    "dyld_cache_caveat": (
                        "a missing standalone binary may still be loadable from the signed dyld shared cache"
                    ),
                }
                for path in FRAMEWORKS
            ],
            runtime_inventory=runtime,
            direct_ane_residency_status=("MEASURED_EXECUTED" if direct_forward else "BUILT_PROBED_NOT_EXECUTED"),
            backward_vjp_reachable_on_ane=bool(backward_executed),
            backward_vjp_evidence=(
                "MEASURED_EXECUTED" if backward_executed else "MEASURED_NO_EXECUTION_SURFACE__INFERRED_API_CAP"
            ),
            backward_selector_candidate_count=len(backward_candidates) if isinstance(backward_candidates, list) else None,
            verdict_scope=(
                "private-runtime classes/selectors visible on this macOS build and safe probe only; absence of an "
                "executed VJP rejects this direct-inference API formulation, not all future Apple firmware/APIs"
            ),
            req_R=(
                "provide a signed/entitled direct-ANE model artifact plus a documented gradient/VJP execution selector "
                "and cotangent parity receipt on this exact OS build"
            ),
        )
        atomic_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["compile_returncode"] == 0 and receipt["run_returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
