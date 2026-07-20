#!/usr/bin/env python3
"""Build, seal, double-decode, and hard-score an R1b4 xi0 receiver smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.integer_plane_emitter import (  # noqa: E402
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PLANE_COUNT,
    RGB_CHANNELS,
)
from tac.boundary_math.integer_plane_emitter_byte_close import (  # noqa: E402
    decode_counted_archive,
)
from tac.boundary_math.r1b4_section_receiver import (  # noqa: E402
    R1B4ReceiverError,
    build_r1b4_archive,
    decode_r1b4_archive,
    encode_replay_payload,
    seal_output_assertion,
    sha256_file,
)
from tac.boundary_math.windowed_curvelet_frame import WindowedCurveletConfig  # noqa: E402
from tac.optimization.boundary_coordinate_joint_solve import (  # noqa: E402
    BoundaryCoordinatePacket,
    FrameFamily,
    encode_boundary_packet,
)
from tac.optimization.r1b2_mdl_xi0_compile import audit_vjp_campaign  # noqa: E402

DEFAULT_BASE: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m1_c2_glue_rebuild_20260719/pre_archive.zip"
)
DEFAULT_DECODER: Final = Path(
    "/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py"
)
DEFAULT_XI0: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b3_producers_20260720T185300Z/xi0.xi0"
)
DEFAULT_CACHE: Final = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_UPSTREAM: Final = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
DEFAULT_VJP: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/"
    "campaign_receipt.json"
)
EXPECTED_XI0_SHA256: Final = "1b3c72fbe1df7209533a0e92e368fc65253bcb55cbe7196e921502ceb757e58a"
CONTROL_ARCHIVE_BYTES: Final = 94_344
CONTROL_D_SEG: Final = 0.003515794640406966
CONTROL_D_POSE: Final = 127.36588287353516
CONTROL_SCORE: Final = 36.10275630841103


class R1B4SmokeError(RuntimeError):
    """Fail-closed receiver-smoke custody or measurement error."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise R1B4SmokeError(f"receipt overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if partial.exists():
        raise R1B4SmokeError(f"stale receipt temporary requires review: {partial}")
    partial_owned = False
    try:
        with partial.open("xb") as handle:
            partial_owned = True
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
        if partial_owned and partial.exists():
            partial.unlink()


def _custody(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _source_custody() -> dict[str, Any]:
    paths = {
        "tool": Path(__file__),
        "receiver": SRC / "tac/boundary_math/r1b4_section_receiver.py",
        "r1b2_compiler": SRC / "tac/optimization/r1b2_mdl_xi0_compile.py",
        "r1b3_xi0_codec": SRC / "tac/optimization/r1b3_producer_preflight.py",
    }
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_head": head,
        "git_dirty_paths": status,
        "files": {name: _custody(path) for name, path in paths.items()},
    }


def _storage_preflight(root: Path, pair_cap: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    raw_bytes = pair_cap * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS
    required = raw_bytes * 6 + (64 << 20)
    usage = shutil.disk_usage(root)
    result = {
        "root": str(root),
        "tier": (
            "VertigoDataTier"
            if str(root).startswith("/Volumes/VertigoDataTier/")
            else "APDataStore"
            if str(root).startswith("/Volumes/APDataStore/")
            else "local_explicit"
        ),
        "raw_bytes_each": raw_bytes,
        "required_free_bytes": required,
        "free_bytes": usage.free,
        "ok": usage.free >= required,
    }
    if not result["ok"]:
        raise R1B4SmokeError("receiver-smoke storage preflight failed")
    return result


def _zero_boundary_payload() -> bytes:
    packet = BoundaryCoordinatePacket(
        family=FrameFamily.WINDOWED_CURVELET,
        frame_config=asdict(WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)),
        scorer_height=384,
        scorer_width=512,
        atom_indices=np.asarray([0], dtype=np.uint32),
        coefficients=np.zeros((600, 1, 3), dtype=np.int8),
        scales=np.ones(600, dtype=np.float16),
    )
    return encode_boundary_packet(packet)


def _hard_measure(
    *,
    control_raw: Path,
    candidate_raw: Path,
    cache: Path,
    upstream: Path,
    pair_cap: int,
    cpu_threads: int,
) -> dict[str, Any]:
    from tools.measure_c2_integer_plane_emitter import (
        _distortion,
        _distortion_outputs,
        _load_distortion_net,
        _load_real_cache,
    )

    cache_fields, cache_sha = _load_real_cache(cache.expanduser().resolve())
    model, torch, scorer_hashes = _load_distortion_net(upstream.expanduser().resolve(), cpu_threads)
    shape = (pair_cap, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS)
    control = np.memmap(control_raw, mode="r", dtype=np.uint8, shape=shape)
    candidate = np.memmap(candidate_raw, mode="r", dtype=np.uint8, shape=shape)
    rows = []
    for pair_index in range(pair_cap):
        source_outputs = _distortion_outputs(
            model,
            torch,
            np.asarray(cache_fields["gt_f0"][pair_index]),
            np.asarray(cache_fields["gt_f1"][pair_index]),
        )
        control_outputs = _distortion_outputs(
            model,
            torch,
            np.asarray(control[pair_index, 0]),
            np.asarray(control[pair_index, 1]),
        )
        candidate_outputs = _distortion_outputs(
            model,
            torch,
            np.asarray(candidate[pair_index, 0]),
            np.asarray(candidate[pair_index, 1]),
        )
        rows.append(
            {
                "pair_index": pair_index,
                "control": _distortion(model, control_outputs, source_outputs),
                "candidate": _distortion(model, candidate_outputs, source_outputs),
            }
        )
    del control, candidate

    def aggregate(label: str) -> dict[str, float]:
        d_pose = float(np.mean([row[label]["d_pose"] for row in rows]))
        d_seg = float(np.mean([row[label]["d_seg"] for row in rows]))
        return {
            "d_pose": d_pose,
            "d_seg": d_seg,
            "pose_component": math.sqrt(10.0 * d_pose),
            "seg_component": 100.0 * d_seg,
        }

    control_row = aggregate("control")
    candidate_row = aggregate("candidate")
    return {
        "axis": "[macOS-CPU capped hard-oracle advisory]",
        "scope": f"receiver_smoke_prefix_n{pair_cap}_non_n600_non_promotion",
        "seed": 1234,
        "batch_geometry": "per-pair smoke; not n600 batch16 authority",
        "cpu_threads": cpu_threads,
        "cache_sha256": cache_sha,
        "scorer_hashes": scorer_hashes,
        "control": control_row,
        "candidate": candidate_row,
        "delta": {key: candidate_row[key] - control_row[key] for key in candidate_row},
        "per_pair": rows,
        "score_claim": False,
    }


def execute(args: argparse.Namespace) -> int:
    started = time.monotonic()
    root = args.artifact_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise R1B4SmokeError(f"artifact root is not empty: {root}")
    storage = _storage_preflight(root, args.pair_cap)
    base = args.base_archive.expanduser().resolve(strict=True)
    decoder = args.base_decoder.expanduser().resolve(strict=True)
    xi0 = args.xi0.expanduser().resolve(strict=True)
    if xi0.stat().st_size != 1_500 or sha256_file(xi0) != EXPECTED_XI0_SHA256:
        raise R1B4SmokeError("banked xi0 payload byte/hash custody drifted")
    unsealed = root / "r1b4_receiver_smoke_unsealed.zip"
    sealed = root / "r1b4_receiver_smoke_sealed.zip"
    discovery_raw = root / "discovery.raw"
    first_raw = root / "determinism_first.raw"
    second_raw = root / "determinism_second.raw"
    control_raw = root / "control.raw"
    boundary = _zero_boundary_payload()
    replay = encode_replay_payload(())
    build = build_r1b4_archive(
        base_archive=base,
        boundary_payload=boundary,
        replay_payload=replay,
        xi0_payload=xi0.read_bytes(),
        source_manifest_hashes={
            "r1b4_build_spec": sha256_file(
                REPO / ".omx/research/r1b4_receiver_actuator_build_spec_20260720.md"
            ),
            "r1b3_producers": sha256_file(REPO / ".omx/research/r1b3_producers_20260720T185300Z.json"),
        },
        output=unsealed,
        artifact_role="receiver_smoke_only",
        pair_cap=args.pair_cap,
    )
    discovery = decode_r1b4_archive(
        archive=unsealed,
        base_decoder=decoder,
        scratch_root=root / "scratch_discovery",
        output_raw=discovery_raw,
        receipt_path=root / "discovery_decode.json",
        workers=args.decode_workers,
        allow_unsealed_discovery=True,
    )
    seal = seal_output_assertion(unsealed, decoded_path=discovery_raw, output=sealed)
    first = decode_r1b4_archive(
        archive=sealed,
        base_decoder=decoder,
        scratch_root=root / "scratch_first",
        output_raw=first_raw,
        receipt_path=root / "determinism_first_decode.json",
        workers=args.decode_workers,
    )
    second = decode_r1b4_archive(
        archive=sealed,
        base_decoder=decoder,
        scratch_root=root / "scratch_second",
        output_raw=second_raw,
        receipt_path=root / "determinism_second_decode.json",
        workers=args.decode_workers,
    )
    deterministic = first["decoded"]["sha256"] == second["decoded"]["sha256"]
    if not deterministic or first["decoded"]["bytes"] != second["decoded"]["bytes"]:
        raise R1B4SmokeError("two sealed receiver decodes are not byte-identical")
    control_decode = decode_counted_archive(
        archive=base,
        base_decoder=decoder,
        scratch_root=root / "scratch_control",
        pair_cap=args.pair_cap,
        output_raw=control_raw,
        workers=args.decode_workers,
    )
    hard_started = time.monotonic()
    hard = _hard_measure(
        control_raw=control_raw,
        candidate_raw=first_raw,
        cache=args.cache,
        upstream=args.upstream,
        pair_cap=args.pair_cap,
        cpu_threads=args.cpu_threads,
    )
    hard_seconds = time.monotonic() - hard_started
    vjp = audit_vjp_campaign(args.vjp_campaign)
    if not vjp["blockers"]:
        raise R1B4SmokeError("VJP unexpectedly terminal; L3 requires a separately authorized assembly pass")
    raw_paths = (discovery_raw, first_raw, second_raw, control_raw)
    cleanup_rows = [
        {
            **_custody(path),
            "reason": "success-only scorer/decode input reproducible from hash-bound archive, receiver, and decoder",
            "cold_store_destination": None,
            "delete_after_measurement_receipt_fsync": True,
            "false_authority_flags": {
                "score_claim": False,
                "promotion_eligible": False,
                "scope": f"prefix_n{args.pair_cap}_receiver_smoke",
            },
        }
        for path in raw_paths
    ]
    sealed_archive_bytes = sealed.stat().st_size
    result = {
        "schema": "r1b4_receiver_smoke_measurement.v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "verdict": "MEASURED_RECEIVER_SMOKE_XI0_ACTUATOR_UNDERDETERMINED_L3_BLOCKED",
        "verdict_scope": (
            f"exact prefix n{args.pair_cap} receiver-smoke archive on macOS CPU only; not n600, "
            "not an R1b2 boundary candidate, no pose-actuator family negative, no score/promotion claim"
        ),
        "authority": {
            "axis": "[macOS-CPU capped hard-oracle advisory]",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        },
        "storage_preflight": storage,
        "inputs": {
            "base_archive": _custody(base),
            "base_decoder": _custody(decoder),
            "xi0": _custody(xi0),
            "cache": _custody(args.cache.expanduser().resolve(strict=True)),
            "vjp_campaign": vjp["campaign"],
        },
        "receiver_smoke": {
            "build": build,
            "seal": seal,
            "sealed_archive": _custody(sealed),
            "archive_delta_bytes_vs_control": sealed_archive_bytes - CONTROL_ARCHIVE_BYTES,
            "boundary_role": "typed_zero_receiver_smoke_section_noncandidate",
            "replay_role": "strict_zero_selection_replay_consumer_smoke",
            "xi0_role": "real_banked_600_value_coordinate_zero_payload",
            "discovery_decode": discovery,
            "determinism_first": first,
            "determinism_second": second,
            "deterministic_equal_output_bytes": deterministic,
            "receiver_search_invocations": 0,
            "control_decode": control_decode,
        },
        "hard_oracle": hard,
        "hard_oracle_seconds": hard_seconds,
        "settled_n600_carrier_absent_control_not_remeasured": {
            "archive_bytes": CONTROL_ARCHIVE_BYTES,
            "d_seg": CONTROL_D_SEG,
            "d_pose": CONTROL_D_POSE,
            "score": CONTROL_SCORE,
            "scope": "settled n600 row; distinct from capped smoke comparison",
        },
        "actuator_interpretation": {
            "status": "R1B4_XI0_TARGET_TO_FRAME0_POSE_ACTUATOR_UNDERDETERMINED",
            "mapping": "generic integer horizontal translation with edge replication",
            "grounding": "dim0 target dominant and frame0 seg-free; exact PoseNet coordinate-to-warp calibration absent",
            "adoption_authority": "hard-oracle row only; no proxy adoption",
        },
        "l3_l4": {
            "status": "BLOCKED_FAIL_CLOSED",
            "vjp_status": vjp["status"],
            "completed_pair_count": vjp["completed_pair_count"],
            "missing_pair_count": vjp["missing_pair_count"],
            "missing_pair_ids": vjp["missing_pair_ids"],
            "refused_pair_ids": vjp["refused_pair_ids"],
            "blockers": vjp["blockers"],
            "no_l3_assembly_attempted": True,
        },
        "cleanup": {
            "schema": "r1b4_receiver_smoke_raw_cleanup.v1",
            "certify_or_block": True,
            "rows": cleanup_rows,
            "deletion_action": "success_only_after_this_receipt_fsync",
        },
        "source_custody": _source_custody(),
        "argv": sys.argv,
        "runtime": {
            "total_seconds": time.monotonic() - started,
            "platform": platform.platform(),
            "python": sys.version,
        },
        "stores_consulted": [
            "committed r1b4 build spec",
            "exact C2 control archive and pinned decoder",
            "banked r1b3 xi0 payload",
            "live read-only VJP campaign receipt",
            "frozen hard-oracle cache and scorer sources",
        ],
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, result)
    for path in raw_paths:
        path.unlink()
    print(
        json.dumps(
            {
                "receipt": str(output),
                "verdict": result["verdict"],
                "pair_cap": args.pair_cap,
                "archive_bytes": sealed_archive_bytes,
                "d_pose": hard["candidate"]["d_pose"],
                "delta_d_pose": hard["delta"]["d_pose"],
                "vjp_completed": vjp["completed_pair_count"],
                "vjp_refused": vjp["refused_pair_ids"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-archive", type=Path, default=DEFAULT_BASE)
    result.add_argument("--base-decoder", type=Path, default=DEFAULT_DECODER)
    result.add_argument("--xi0", type=Path, default=DEFAULT_XI0)
    result.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    result.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    result.add_argument("--vjp-campaign", type=Path, default=DEFAULT_VJP)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--pair-cap", type=int, default=2)
    result.add_argument("--decode-workers", type=int, default=1)
    result.add_argument("--cpu-threads", type=int, default=1)
    return result


def main() -> None:
    try:
        raise SystemExit(execute(parser().parse_args()))
    except (R1B4ReceiverError, R1B4SmokeError, OSError, ValueError) as exc:
        raise SystemExit(f"R1B4_RECEIVER_SMOKE_REFUSED: {exc}") from exc


if __name__ == "__main__":
    main()
