#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: argv only forwarded on venv re-exec; no options consumed
"""Fail-closed PRIMARY consumer plus local Task #603 custody runner.

The current PRIMARY spec seals ``execution_allowed=false``.  ``preflight``
prints the authoritative DRAFT readiness record; ``optimize`` refuses before
any subprocess, provider, GPU, scorer, or archive mutation.  The separately
typed ``custody-smoke`` mode runs only the deterministic n64 integer receiver
and bounded description-space search; it has no scorer or launch authority.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _bootstrap_repo_python() -> None:
    """Re-exec the CLI under the nearest repository-owned virtualenv.

    ``/usr/bin/env python3`` is the portable program entry point, but PATH may
    resolve to a Python without the locked repository dependencies.  Direct
    execution therefore pins itself to the nearest ancestor ``.venv`` before
    importing any project module.  Missing environment custody is a refusal,
    never an accidental system-Python traceback.
    """

    candidates = tuple(root / ".venv/bin/python" for root in (REPO_ROOT, *REPO_ROOT.parents))
    interpreter = next((path for path in candidates if path.is_file()), None)
    if interpreter is None:
        print("REFUSE: repository Python environment unavailable", file=sys.stderr)
        raise SystemExit(2)
    try:
        already_pinned = Path(sys.executable).resolve() == interpreter.resolve()
    except OSError:
        already_pinned = False
    if already_pinned:
        return
    if os.environ.get("DDM_REPO_PYTHON_BOOTSTRAPPED") == "1":
        print("REFUSE: repository Python bootstrap did not converge", file=sys.stderr)
        raise SystemExit(2)
    environment = dict(os.environ)
    environment["DDM_REPO_PYTHON_BOOTSTRAPPED"] = "1"
    try:
        os.execve(
            str(interpreter),
            [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )
    except OSError as exc:
        print(f"REFUSE: repository Python bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    _bootstrap_repo_python()

from tac.optimization.direct_description_minimizer import (  # noqa: E402
    DirectDescriptionCustodyProgramV1,
    DirectDescriptionError,
    DirectDescriptionOptimizerConfigV1,
    build_direct_description_arg_parser,
    build_launch_readiness,
    load_stage_checkpoint,
    run_n64_deterministic_custody_smoke,
    storage_preflight,
)


def _read_json_no_duplicates(path: Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DirectDescriptionError(f"duplicate owner-manifest key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"owner manifest is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError("owner manifest must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_direct_description_arg_parser()
    args = parser.parse_args(argv)
    try:
        owner = _read_json_no_duplicates(args.owner_manifest)
        if args.execution_allowed != "false":
            raise DirectDescriptionError("current PRIMARY manifest only compiles --execution-allowed false")
        if args.operator_go is not None:
            raise DirectDescriptionError(
                "an operator-GO cannot supersede execution_allowed:false; a reviewed spec update is required"
            )
        if args.mode == "custody-smoke":
            if args.custody_config is None or args.output_dir is None:
                raise DirectDescriptionError(
                    "custody-smoke requires typed --custody-config and --output-dir"
                )
            if args.resume_from is not None:
                raise DirectDescriptionError(
                    "custody-smoke performs its own disk-resume control; external --resume-from is ambiguous"
                )
            # Reuse the owner verifier without pretending its launch gates are green.
            build_launch_readiness(
                owner,
                storage_receipt={"outcome": "REFUSE", "reason": "local custody mode"},
                memory_preflight_outcome="REFUSE",
                governor_outcome="REFUSE",
                operator_go=None,
            )
            config = DirectDescriptionOptimizerConfigV1.model_validate_json(
                json.dumps(
                    _read_json_no_duplicates(args.custody_config),
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            program = DirectDescriptionCustodyProgramV1(
                owner_manifest_path=str(args.owner_manifest),
                custody_config_path=str(args.custody_config),
                output_directory=str(args.output_dir),
            )
            semantic_argv = program.compile_consumer_argv()
            receipt, receipt_path = run_n64_deterministic_custody_smoke(
                config,
                output_directory=args.output_dir,
                semantic_argv=semantic_argv,
            )
            print(
                json.dumps(
                    {"receipt_path": str(receipt_path), "receipt": receipt},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.custody_config is not None or args.output_dir is not None:
            raise DirectDescriptionError("custody arguments are valid only in custody-smoke mode")
        storage = storage_preflight(2 * 1024**3)
        readiness = build_launch_readiness(
            owner,
            storage_receipt=storage,
            memory_preflight_outcome="REFUSE",
            governor_outcome="REFUSE",
            operator_go=None,
        )
        if args.resume_from is not None:
            checkpoint = load_stage_checkpoint(
                args.resume_from,
                expected_config_sha256=owner["typed_config_hash"],
                expected_dsl_compile_hash=owner["dsl_compile_hash"],
                expected_argv=owner["consumer_argv"],
            )
            readiness["resume_checkpoint_audit"] = {
                "status": "BOUND_SCHEMA_RESTORE_ONLY",
                "run_id": checkpoint.run_id,
                "stage_name": checkpoint.stage_name,
                "global_step": checkpoint.global_step,
                "continuation_runner_ready": False,
            }
        print(json.dumps(readiness, sort_keys=True, separators=(",", ":")))
        if args.mode == "optimize":
            raise DirectDescriptionError("DRAFT_DO_NOT_FIRE: direct-description launch readiness predicates are red")
        if readiness.get("launch_ready") is not True:
            raise DirectDescriptionError("PREFLIGHT_REFUSE: launch readiness predicates are red")
    except (DirectDescriptionError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
