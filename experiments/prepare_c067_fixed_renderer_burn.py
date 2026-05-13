#!/usr/bin/env python3
"""Prepare fixed C067 inputs and command shapes for renderer-only training.

This is a local/no-dispatch helper. It extracts the exact C067 logical runtime
members, records byte/SHA custody, and writes replayable command shapes for a
fresh fixed-mask/fixed-pose renderer training burn. It does not start training,
does not run scorers, and does not claim a score.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

BLOCKFP_BUILDER_PATH = REPO_ROOT / "experiments" / "build_blockfp_c067_archive.py"
ARGPARSE_DRYRUN_PATH = REPO_ROOT / "tools" / "argparse_dryrun.py"
SCHEMA = "c067_fixed_renderer_training_burn_prep_v1"
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip"
)
DEFAULT_CLEARANCE_PACKET = REPO_ROOT / ".omx/state/lane12_nerv_l2_clearance.json"
DEFAULT_PROFILE = "q_faithful_dilated_88k"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
)
EXPECTED_MEMBER_SHA256 = {
    "renderer.bin": "5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb",
    "masks.mkv": "a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb",
    "optimized_poses.bin": "5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f",
}
REQUIRED_MEMBERS = tuple(EXPECTED_MEMBER_SHA256)


def _load_blockfp_builder() -> Any:
    spec = importlib.util.spec_from_file_location("_c067_burn_blockfp_builder", BLOCKFP_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BLOCKFP_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BLOCKFP_BUILDER = _load_blockfp_builder()


def _load_argparse_dryrun() -> Any:
    spec = importlib.util.spec_from_file_location("_c067_burn_argparse_dryrun", ARGPARSE_DRYRUN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {ARGPARSE_DRYRUN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARGPARSE_DRYRUN = _load_argparse_dryrun()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _shell_arg(value: str) -> str:
    if value == "$PWD":
        return '"$PWD"'
    return _shell_quote(value)


def _clearance_gate(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {
            "cleared_for_retraining_dispatch": False,
            "clearance_packet_path": str(path),
            "blockers": ["lane12_l2_clearance_packet_missing_or_unreadable"],
        }
    blockers: list[str] = []
    if payload.get("cleared_for_retraining_unblock") is not True:
        blockers.append("cleared_for_retraining_unblock_not_true")
    if payload.get("lane12_l2") is not True:
        blockers.append("lane12_l2_not_true")
    if payload.get("geometry_gate_passed") is not True:
        blockers.append("geometry_gate_passed_not_true")
    clean_passes = payload.get("grand_council_clean_passes")
    if not isinstance(clean_passes, int) or clean_passes < 3:
        blockers.append("grand_council_clean_passes_lt_3")
    return {
        "cleared_for_retraining_dispatch": not blockers,
        "clearance_packet_path": str(path),
        "blockers": sorted(blockers),
    }


def _train_command(*, run_dir: Path, profile: str, seed: int, wall_clock_timeout: int) -> list[str]:
    inputs = run_dir / "inputs"
    return [
        ".venv/bin/python",
        "-u",
        "src/tac/experiments/train_renderer.py",
        "--profile",
        profile,
        "--video",
        "upstream/videos/0.mkv",
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--deterministic",
        "--tag",
        run_dir.name,
        "--output-dir",
        _repo_rel(run_dir / "train"),
        "--qfaithful-training-poses",
        _repo_rel(inputs / "optimized_poses.bin"),
        "--mask-noise-mkv",
        _repo_rel(inputs / "masks.mkv"),
        "--mask-noise-prob",
        "1.0",
        "--auth-eval-masks",
        _repo_rel(inputs / "masks.mkv"),
        "--auth-eval-poses",
        _repo_rel(inputs / "optimized_poses.bin"),
        "--no-auth-eval-on-best",
        "--wall-clock-timeout",
        str(wall_clock_timeout),
    ]


def _snapshot_command(*, run_dir: Path, profile: str) -> list[str]:
    inputs = run_dir / "inputs"
    return [
        ".venv/bin/python",
        "-u",
        "scripts/q_faithful_snapshot_loop.py",
        "--workspace",
        "$PWD",
        "--python-bin",
        ".venv/bin/python",
        "--checkpoint-dir",
        _repo_rel(run_dir / "train"),
        "--checkpoint-glob",
        "training_state_*.pt",
        "--masks-mkv",
        _repo_rel(inputs / "masks.mkv"),
        "--mask-frame-contract",
        "auto",
        "--poses-pt",
        _repo_rel(inputs / "optimized_poses.bin"),
        "--output-root",
        _repo_rel(run_dir / "snapshots"),
        "--min-checkpoint-age-seconds",
        "60",
        "--poll-seconds",
        "120",
        "--max-idle-polls",
        "720",
        "--profile",
        profile,
        "--state-source",
        "ema_shadow",
        "--renderer-codec",
        "qzs3",
        "--qzs3-block-size",
        "32",
        "--submission-layout",
        "pr64_mask_first_single_blob",
        "--pose-codec",
        "raw",
        "--brotli-quality",
        "11",
        "--eval-mode",
        "none",
        "--dispatch-claim-mode",
        "none",
    ]


def _write_script(path: Path, *, run_dir: Path, train_cmd: list[str], snapshot_cmd: list[str]) -> None:
    train_log = _shell_quote(_repo_rel(run_dir / "logs" / "train_renderer.log"))
    snapshot_log = _shell_quote(_repo_rel(run_dir / "logs" / "q_faithful_snapshot_loop.log"))
    train_line = " ".join(_shell_arg(item) for item in train_cmd) + " 2>&1 | tee " + train_log
    snapshot_line = " ".join(_shell_arg(item) for item in snapshot_cmd) + " 2>&1 | tee " + snapshot_log
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export PYTHONHASHSEED=1234",
        "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
        "",
        "# Run the exporter beside training so long burns produce checkpoint",
        "# archives before provider timeout or operator stop. The snapshot loop",
        "# waits for checkpoints and stays non-scoring (`--eval-mode none`).",
        f"({snapshot_line}) &",
        "SNAPSHOT_PID=$!",
        "cleanup_snapshot_loop() { kill ${SNAPSHOT_PID} 2>/dev/null || true; }",
        "trap cleanup_snapshot_loop EXIT",
        "set +e",
        train_line,
        "TRAIN_STATUS=${PIPESTATUS[0]}",
        "if kill -0 ${SNAPSHOT_PID} 2>/dev/null; then",
        "  kill ${SNAPSHOT_PID} 2>/dev/null || true",
        "  wait ${SNAPSHOT_PID} 2>/dev/null || true",
        "  SNAPSHOT_STATUS=0",
        "else",
        "  wait ${SNAPSHOT_PID}",
        "  SNAPSHOT_STATUS=$?",
        "fi",
        "trap - EXIT",
        "set -e",
        "if [[ ${TRAIN_STATUS} -ne 0 ]]; then exit ${TRAIN_STATUS}; fi",
        "if [[ ${SNAPSHOT_STATUS} -ne 0 ]]; then exit ${SNAPSHOT_STATUS}; fi",
        "",
    ]
    path.write_text("\n".join(lines))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _argparse_dryrun_record(*, target: Path, command: list[str]) -> dict[str, Any]:
    argv = command[3:]
    ok, message = ARGPARSE_DRYRUN.dryrun_parse(target, argv)
    return {
        "target": _repo_rel(target),
        "argv_from_command_index": 3,
        "ok": bool(ok),
        "message": str(message),
    }


def prepare_burn(
    *,
    source_archive: Path,
    output_dir: Path,
    run_id: str,
    profile: str = DEFAULT_PROFILE,
    seed: int = 20260503,
    wall_clock_timeout: int = 82_800,
    clearance_packet: Path = DEFAULT_CLEARANCE_PACKET,
    force: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_id
    inputs_dir = run_dir / "inputs"
    logs_dir = run_dir / "logs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    source_archive = source_archive.resolve()
    source_sha = _sha256_file(source_archive)
    if source_sha != EXPECTED_SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            f"source archive SHA mismatch: expected={EXPECTED_SOURCE_ARCHIVE_SHA256} actual={source_sha}"
        )
    members, source_contract = BLOCKFP_BUILDER.extract_runtime_members(source_archive)
    missing = [name for name in REQUIRED_MEMBERS if name not in members]
    if missing:
        raise ValueError(f"source archive missing logical members: {missing}")

    member_manifest: dict[str, Any] = {}
    for name in REQUIRED_MEMBERS:
        payload = members[name]
        out = inputs_dir / name
        out.write_bytes(payload)
        sha = _sha256_bytes(payload)
        if sha != EXPECTED_MEMBER_SHA256[name]:
            raise ValueError(
                f"{name} SHA mismatch: expected={EXPECTED_MEMBER_SHA256[name]} actual={sha}"
            )
        member_manifest[name] = {
            "path": _repo_rel(out),
            "bytes": len(payload),
            "sha256": sha,
        }

    train_cmd = _train_command(
        run_dir=run_dir,
        profile=profile,
        seed=seed,
        wall_clock_timeout=wall_clock_timeout,
    )
    snapshot_cmd = _snapshot_command(run_dir=run_dir, profile=profile)
    script_path = run_dir / "run_fixed_renderer_burn.sh"
    _write_script(script_path, run_dir=run_dir, train_cmd=train_cmd, snapshot_cmd=snapshot_cmd)

    summary = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "run_id": run_id,
        "source_archive": {
            "path": _repo_rel(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": source_sha,
            **source_contract,
        },
        "fixed_runtime_members": member_manifest,
        "training_dispatch_gate": _clearance_gate(clearance_packet),
        "preflight": {
            "argparse_static_dryrun": {
                "train_renderer": _argparse_dryrun_record(
                    target=REPO_ROOT / "src/tac/experiments/train_renderer.py",
                    command=train_cmd,
                ),
                "q_faithful_snapshot_loop": _argparse_dryrun_record(
                    target=REPO_ROOT / "scripts/q_faithful_snapshot_loop.py",
                    command=snapshot_cmd,
                ),
            }
        },
        "commands": {
            "train_renderer": train_cmd,
            "q_faithful_snapshot_loop": snapshot_cmd,
            "shell_script": _repo_rel(script_path),
        },
        "next_required_gates": [
            "run only after retraining dispatch clearance or explicit audited operator override",
            "export snapshots with no exact eval",
            "run experiments/preflight_trained_renderer_transplant.py to build byte-closed candidates",
            "run experiments/preflight_renderer_transplant_pose_safety.py for selected archive SHA pairs",
            "rerun transplant preflight with --pose-safety-json before exact CUDA dispatch",
            "claim c067_trained_renderer_self_compression_blockfp before any paid remote exact eval",
        ],
    }
    manifest_path = run_dir / "fixed_c067_renderer_burn_manifest.json"
    manifest_path.write_bytes(_json_bytes(summary))
    (inputs_dir / "fixed_c067_member_manifest.json").write_bytes(
        _json_bytes(
            {
                "source_archive": summary["source_archive"],
                "fixed_runtime_members": member_manifest,
            }
        )
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--wall-clock-timeout", type=int, default=82_800)
    parser.add_argument("--clearance-packet", type=Path, default=DEFAULT_CLEARANCE_PACKET)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = prepare_burn(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        run_id=args.run_id,
        profile=args.profile,
        seed=args.seed,
        wall_clock_timeout=args.wall_clock_timeout,
        clearance_packet=args.clearance_packet,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
