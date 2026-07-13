#!/usr/bin/env python3
"""Materialize the held 2026-07-13 all-compatible-levers ticket (never launch).

The output is small provenance/config metadata only.  ``launch.sh`` is emitted
from the same typed config object the governed launcher consumes; this tool does
not spawn a process, touch a run directory outside the requested ticket path, or
perform training.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tac.witness_dsl.spec_next_launch_all_levers_20260713 import (  # noqa: E402
    DEFAULT_OUT_DIR,
    FULL_VARIANT,
    MEMORY_VARIANTS,
    TRIMMED_COMPLIANT_VARIANT,
    TRIMMED_OUT_DIR,
    compile_next_launch_all_levers_ticket,
)
from tac.witness_dsl.typed_config import REQUIRED_PERF_ENV  # noqa: E402

import launch_witness_run as launcher  # noqa: E402


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _include_rows(cfg) -> list[dict[str, Any]]:
    rows = [
        {
            "lever": lv.name,
            "status": "IN",
            "surface": "active_typed_dsl_lever",
            "provenance": lv.notes,
        }
        for lv in cfg.typed.levers
    ]
    rows.extend(
        [
            {
                "lever": "trainer_one_thread_standard",
                "status": "IN",
                "surface": "trainer canonical equation (no launch flag)",
                "provenance": (
                    "MEASURED 2.995x CPU frozen-SegNet-forward subcomponent vs six threads; "
                    "not promoted to a whole-step multiplier"
                ),
            },
            {
                "lever": "custom_grouped_backward",
                "status": "IN",
                "surface": "required perf env TAC_MLX_CUSTOM_GROUPED_BACKWARD=1",
                "provenance": "bit-exact custom VJP fast path; launcher structurally requires the env",
            },
            {
                "lever": "custom_persistence_pool",
                "status": "IN",
                "surface": "required perf env TAC_MLX_CUSTOM_PERSISTENCE_POOL=1",
                "provenance": "bit-identical full-loss parity on real n600 GT; shared perf-env SoT",
            },
            {
                "lever": "async_verdict",
                "status": "IN",
                "surface": "sealed base flag --async-verdict",
                "provenance": "observer-only worker; measured critical-path wait was zero in prior receipt",
            },
            {
                "lever": "DsegAwareTaper",
                "status": "IN",
                "surface": "sealed v7.5.2 base",
                "provenance": (
                    "duty-to-measure rank 1, 78.9%, ESTIMATED DeltaS=0.03; this run records its "
                    "trajectory but is not an isolated causal A/B"
                ),
            },
            {
                "lever": "latent_table_truncate_d18_k90",
                "status": "IN",
                "surface": "terminal byte-close A/B slot (not a training flag)",
                "provenance": "duty rank 4, 2.6%, ESTIMATED DeltaS=0.001; run at stop-time",
            },
            {
                "lever": "mod32_neutrality_19_ab",
                "status": "IN",
                "surface": "terminal matched byte-close A/B slot (not a training flag)",
                "provenance": "duty rank 5, 1.3%, ESTIMATED DeltaS=0.0005; compare exact terminal bytes",
            },
        ]
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gt-cache",
        default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    )
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--variant", choices=MEMORY_VARIANTS, default=FULL_VARIANT)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    default_out = TRIMMED_OUT_DIR if args.variant == TRIMMED_COMPLIANT_VARIANT else DEFAULT_OUT_DIR
    out_dir = Path(args.out_dir or default_out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    cfg = compile_next_launch_all_levers_ticket(
        args.gt_cache,
        num_pairs=args.num_pairs,
        epochs=args.epochs,
        out_dir=str(out_dir.relative_to(REPO)),
        variant=args.variant,
    )
    violations = cfg.typed.validate_program()
    if violations:
        raise SystemExit(f"typed DSL validation refused: {violations[:8]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    launch_sh = launcher.write_launch_sh(cfg, out_dir)
    constants_path = launcher.write_constants_manifest(cfg, out_dir)
    program = cfg.typed.to_program()
    compiled_argv = program.compile_trainer_argv()
    include_rows = _include_rows(cfg)
    exclude_rows = [
        {"lever": name, "status": "EXCLUDED", "reason": reason}
        for name, reason in cfg.dsl_program_manifest.get("excluded_levers", {}).items()
    ]

    _atomic_json(out_dir / "dsl_program_manifest.json", cfg.dsl_program_manifest)
    _atomic_json(out_dir / "typed_witness_config.json", cfg.typed.model_dump(mode="json", by_alias=True))
    _atomic_json(out_dir / "witness_program.json", dataclasses.asdict(program))
    _atomic_json(out_dir / "compiled_trainer_argv.json", compiled_argv)
    _atomic_json(
        out_dir / "include_exclude_table.json",
        {
            "schema": "next_launch_include_exclude.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "include_count": len(include_rows),
            "exclude_count": len(exclude_rows),
            "includes": include_rows,
            "excludes": exclude_rows,
        },
    )
    receipt = {
        "schema": "next_launch_ticket_compile_receipt.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "program_name": cfg.name,
        "memory_variant": args.variant,
        "typed_validation": {"ok": True, "violations": []},
        "active_dsl_lever_count": len(cfg.dsl_levers),
        "include_count": len(include_rows),
        "exclude_count": len(exclude_rows),
        "launch_blocker_count": len(cfg.dsl_program_manifest.get("launch_blockers", [])),
        "required_perf_env": REQUIRED_PERF_ENV,
        "artifacts": {},
        "containment": "NO_LAUNCH; config/provenance materialization only",
    }
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "ticket_compile_receipt.json":
            receipt["artifacts"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _atomic_json(out_dir / "ticket_compile_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
