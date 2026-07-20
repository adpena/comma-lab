#!/usr/bin/env python3
"""Materialize the honest R1b3 producer preflight and typed xi[0] bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.r1b2_mdl_xi0_compile import (  # noqa: E402
    R1B2CompileError,
    audit_vjp_campaign,
    audit_xi0,
)
from tac.optimization.r1b3_producer_preflight import (  # noqa: E402
    R1B3ProducerError,
    build_producer_preflight_receipt,
    build_xi0_bundle,
    sha256_file,
)

DEFAULT_STAGE: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r2b_sparse_target_selection_20260720T1621Z/"
    "baseline_stages_a7192f938785_31d77be9ab9f_107a7d3a179d"
)
DEFAULT_FULL_KERNEL: Final = REPO / ".omx/research/null_compiler_full_kernel_20260720T163500Z.json"
DEFAULT_R2B: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r2b_sparse_target_selection_20260720T1621Z/receipt.json"
)
DEFAULT_GT_CACHE: Final = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_SEGNET: Final = Path("/Users/adpena/Projects/pact/upstream/models/segnet.safetensors")
DEFAULT_BASE_DECODER: Final = Path(
    "/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py"
)
DEFAULT_PARSER_SOURCE: Final = REPO / "src/tac/boundary_math/integer_plane_emitter_byte_close.py"
DEFAULT_VJP: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json"
)
MINIMUM_FREE_BYTES: Final = 1 << 20


class R1B3BuildError(RuntimeError):
    """Fail-closed output, custody, or materialization error."""


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise R1B3BuildError(f"output overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if partial.exists():
            partial.unlink()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_bytes(path, payload)


def _file_custody(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _git_custody() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"head": head, "dirty_paths": dirty}


def _source_custody() -> dict[str, dict[str, Any]]:
    return {
        name: _file_custody(path)
        for name, path in {
            "builder": Path(__file__),
            "producer_preflight": SRC / "tac/optimization/r1b3_producer_preflight.py",
            "r1b2_compiler": SRC / "tac/optimization/r1b2_mdl_xi0_compile.py",
        }.items()
    }


def storage_preflight(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(resolved)
    row = {
        "schema": "r1b3_small_artifact_storage_preflight.v1",
        "artifact_root": str(resolved),
        "artifact_class": "small_manifest_and_1500_byte_payload",
        "required_free_bytes": MINIMUM_FREE_BYTES,
        "free_bytes": usage.free,
        "ok": usage.free >= MINIMUM_FREE_BYTES,
    }
    if not row["ok"]:
        raise R1B3BuildError(f"storage preflight refused: {usage.free} B free < {MINIMUM_FREE_BYTES} B required")
    return row


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    result.add_argument("--segnet-weights", type=Path, default=DEFAULT_SEGNET)
    result.add_argument("--full-kernel-receipt", type=Path, default=DEFAULT_FULL_KERNEL)
    result.add_argument("--r2b-receipt", type=Path, default=DEFAULT_R2B)
    result.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    result.add_argument("--base-decoder", type=Path, default=DEFAULT_BASE_DECODER)
    result.add_argument("--production-parser-source", type=Path, default=DEFAULT_PARSER_SOURCE)
    result.add_argument("--vjp-campaign", type=Path, default=DEFAULT_VJP)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    return result


def execute(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.expanduser().resolve()
    preflight = storage_preflight(artifact_dir)
    xi_payload = artifact_dir / "xi0.xi0"
    xi_manifest = artifact_dir / "xi0_manifest.json"

    receipt = build_producer_preflight_receipt(
        stage_dir=args.stage_dir,
        segnet_weights=args.segnet_weights,
        full_kernel_receipt=args.full_kernel_receipt,
        r2b_receipt=args.r2b_receipt,
        gt_cache=args.gt_cache,
        base_decoder=args.base_decoder,
        production_parser_source=args.production_parser_source,
    )
    xi = build_xi0_bundle(args.gt_cache, payload_path=xi_payload)
    _atomic_bytes(xi_payload, xi["payload_bytes"])
    _atomic_json(xi_manifest, xi["manifest"])
    xi_compiler_audit, xi_blockers = audit_xi0(xi_manifest)
    if xi_compiler_audit is None or xi_blockers:
        raise R1B3BuildError(f"materialized xi0 bundle failed R1b2 audit: {xi_blockers}")

    vjp = audit_vjp_campaign(args.vjp_campaign)
    receipt["captured_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    receipt["storage_preflight"] = preflight
    receipt["git"] = _git_custody()
    receipt["source_custody"] = _source_custody()
    receipt["vjp"] = vjp
    receipt["p3"] = {key: value for key, value in xi.items() if key != "payload_bytes"}
    receipt["p3"]["compiler_manifest_emitted"] = True
    receipt["p3"]["compiler_manifest"] = _file_custody(xi_manifest)
    receipt["p3"]["r1b2_compiler_audit_passed"] = True
    receipt["blockers"] = list(dict.fromkeys([*vjp["blockers"], *receipt["blockers"]]))
    receipt["verdict"] = "XI0_BUNDLE_MATERIALIZED_OTHER_PRODUCERS_AND_RECEIVER_BLOCKED"
    receipt["argv"] = sys.argv
    _atomic_json(args.receipt.expanduser().resolve(), receipt)
    print(
        json.dumps(
            {
                "receipt": str(args.receipt.expanduser().resolve()),
                "xi0_payload": _file_custody(xi_payload),
                "xi0_manifest": _file_custody(xi_manifest),
                "vjp_status": vjp["status"],
                "vjp_completed_pair_count": vjp["completed_pair_count"],
                "vjp_refused_pair_ids": vjp["refused_pair_ids"],
                "verdict": receipt["verdict"],
                "blocker_count": len(receipt["blockers"]),
            },
            sort_keys=True,
        )
    )
    return 3


def main() -> None:
    try:
        raise SystemExit(execute(parser().parse_args()))
    except (R1B2CompileError, R1B3ProducerError, R1B3BuildError) as exc:
        raise SystemExit(f"R1B3_PRODUCER_REFUSED: {exc}") from exc


if __name__ == "__main__":
    main()
