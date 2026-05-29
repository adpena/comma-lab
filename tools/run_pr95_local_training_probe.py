#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the public PR95 HNeRV trainer locally without mutating the intake tree.

This is an execution harness around the lifted PR95 source, not a rewrite of
that source. It exists so local Apple Silicon / MPS timing and gradient
portability can be measured with explicit authority labels before any full
campaign uses the results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    write_representation_training_probe_manifest,
)

DEFAULT_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon"
)
DEFAULT_PUBLIC_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "archive.zip"
)
LANE_ID = "lane_pr95_local_mps_source_faithful_training_probe_20260519"

STAGE_MODULES: tuple[str, ...] = (
    "stage1_v328_ce",
    "stage2_v331_softplus",
    "stage3_v332_smooth",
    "stage4_v332_qat",
    "stage5_c1a_l7",
    "stage6_lambda_sweep",
    "stage7_sigma_sweep",
    "stage8_muon_finetune",
)

IGNORED_TREE_PARTS = {
    "__pycache__",
    "ckpts",
    ".git",
}


@dataclass(frozen=True)
class Pr95SourceLayout:
    source_dir: Path
    source_stack_dir: Path
    challenge_root: Path
    train_py: Path
    compress_sh: Path
    inflate_sh: Path
    public_archive: Path | None


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_stored_single_member_zip(payload_path: Path, archive_path: Path) -> dict[str, Any]:
    """Write a contest archive without platform extra fields or recompression."""
    payload_path = payload_path.resolve()
    archive_path = archive_path.resolve()
    if payload_path.name != "0.bin":
        raise ValueError(f"PR95 archive payload must be named 0.bin, got {payload_path.name!r}")
    data = payload_path.read_bytes()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    info.extra = b""
    with zipfile.ZipFile(archive_path, mode="w") as zf:
        zf.comment = b""
        zf.writestr(info, data)
    with zipfile.ZipFile(archive_path, mode="r") as zf:
        infos = zf.infolist()
    if len(infos) != 1 or infos[0].filename != "0.bin":
        raise RuntimeError("PR95 archive writer emitted a non-single-member archive")
    if infos[0].compress_type != zipfile.ZIP_STORED or infos[0].extra:
        raise RuntimeError("PR95 archive writer emitted compression or ZIP extra fields")
    return {
        "path": _rel(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256_file(archive_path),
        "member": "0.bin",
        "member_bytes": payload_path.stat().st_size,
        "member_sha256": _sha256_file(payload_path),
        "compression_method": "stored",
        "extra_field_bytes": len(infos[0].extra),
    }


def _auth_eval_axis_label(device: str) -> str:
    if device == "cuda":
        return "contest-CUDA-candidate"
    if device == "cpu" and platform.system() == "Linux" and platform.machine().lower() in {
        "x86_64",
        "amd64",
    }:
        return "contest-CPU-candidate"
    if device == "cpu":
        return "macOS-CPU advisory" if platform.system() == "Darwin" else "CPU advisory"
    if device == "mps":
        return "MPS diagnostic"
    return f"{device} diagnostic"


def _latest_training_score(results: list[dict[str, Any]]) -> float | None:
    for result in reversed(results):
        score = result.get("best_score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def run_auth_eval_bridge(
    *,
    archive_zip: Path,
    layout: Pr95SourceLayout,
    output_dir: Path,
    device: str,
    training_score: float | None,
    score_delta_tolerance: float,
    require_comparable: bool,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> dict[str, Any]:
    """Run canonical auth eval on the just-emitted archive and summarize parity.

    This intentionally preserves the evidence-grade boundary: local macOS CPU
    and MPS runs are useful bridge signals, not promotion authority.
    """

    if not archive_zip.is_file():
        raise FileNotFoundError(
            f"--run-auth-eval requires an emitted archive.zip; missing {_rel(archive_zip)}. "
            "Pass --run-codec-stage first."
        )
    if score_delta_tolerance < 0:
        raise ValueError("--auth-eval-score-delta-tolerance must be non-negative")

    auth_eval_py = REPO_ROOT / "experiments" / "contest_auth_eval.py"
    json_out = output_dir / f"contest_auth_eval_{device}.json"
    work_dir = output_dir / f"contest_auth_eval_{device}_workdir"
    cmd = [
        sys.executable,
        str(auth_eval_py),
        "--archive",
        str(archive_zip),
        "--inflate-sh",
        str(layout.inflate_sh),
        "--upstream-dir",
        str(layout.challenge_root),
        "--device",
        device,
        "--work-dir",
        str(work_dir),
        "--json-out",
        str(json_out),
        "--inflate-timeout",
        str(inflate_timeout),
        "--evaluate-timeout",
        str(evaluate_timeout),
    ]
    env = os.environ.copy()
    python_path_entries = [
        Path(sys.prefix) / "bin",
        Path(sys.executable).resolve().parent,
    ]
    env["PATH"] = os.pathsep.join(
        [str(path) for path in python_path_entries] + [env.get("PATH", "")]
    )
    started = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=inflate_timeout + evaluate_timeout + 300,
        env=env,
    )
    elapsed = time.perf_counter() - started
    bridge: dict[str, Any] = {
        "schema": "pr95_local_auth_eval_bridge_v1",
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "command": cmd,
        "elapsed_seconds": elapsed,
        "archive_path": _rel(archive_zip),
        "archive_bytes": archive_zip.stat().st_size,
        "archive_sha256": _sha256_file(archive_zip),
        "inflate_sh": _rel(layout.inflate_sh),
        "upstream_dir": _rel(layout.challenge_root),
        "json_out": _rel(json_out),
        "work_dir": _rel(work_dir),
        "auth_eval_device": device,
        "score_axis": _auth_eval_axis_label(device),
        "outer_auth_eval_python_path_prepend": [str(path) for path in python_path_entries],
        "training_best_score": training_score,
        "score_delta_tolerance": score_delta_tolerance,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if result.returncode != 0:
        (output_dir / f"auth_eval_bridge_{device}.failed.json").write_text(
            json.dumps(bridge, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise RuntimeError(
            "PR95 local auth-eval bridge failed "
            f"(rc={result.returncode}); see stdout_tail/stderr_tail in bridge payload"
        )
    if not json_out.is_file():
        raise RuntimeError(f"auth eval completed but did not write {_rel(json_out)}")

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    auth_score = payload.get("canonical_score")
    bridge["auth_eval_json_sha256"] = _sha256_file(json_out)
    bridge["auth_eval_canonical_score"] = auth_score
    bridge["auth_eval_final_score"] = payload.get("final_score")
    bridge["auth_eval_archive_bytes"] = payload.get("archive_size_bytes")
    bridge["auth_eval_archive_sha256"] = (
        payload.get("provenance", {}).get("archive_sha256")
        if isinstance(payload.get("provenance"), dict)
        else None
    )
    if isinstance(training_score, (int, float)) and isinstance(auth_score, (int, float)):
        delta = abs(float(auth_score) - float(training_score))
        bridge["absolute_score_delta"] = delta
        bridge["score_comparable"] = delta <= score_delta_tolerance
        if require_comparable and delta > score_delta_tolerance:
            (output_dir / f"auth_eval_bridge_{device}.failed.json").write_text(
                json.dumps(bridge, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            raise RuntimeError(
                "PR95 local auth-eval bridge score mismatch: "
                f"training={training_score} auth_eval={auth_score} "
                f"delta={delta} tolerance={score_delta_tolerance}"
            )
    else:
        bridge["absolute_score_delta"] = None
        bridge["score_comparable"] = None
    return bridge


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_layout(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    public_archive: Path = DEFAULT_PUBLIC_ARCHIVE,
) -> Pr95SourceLayout:
    source_dir = source_dir.resolve()
    source_stack_dir = source_dir / "src"
    challenge_root = source_dir.parent.parent
    train_py = source_stack_dir / "train.py"
    compress_sh = source_dir / "compress.sh"
    inflate_sh = source_dir / "inflate.sh"
    required = [
        source_stack_dir / "stages/common.py",
        train_py,
        compress_sh,
        inflate_sh,
        challenge_root / "frame_utils.py",
        challenge_root / "modules.py",
        challenge_root / "models/segnet.safetensors",
        challenge_root / "models/posenet.safetensors",
        challenge_root / "videos/0.mkv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(_rel(path) for path in missing)
        raise FileNotFoundError(f"PR95 source layout incomplete: {joined}")
    archive = public_archive.resolve() if public_archive.exists() else None
    return Pr95SourceLayout(
        source_dir=source_dir,
        source_stack_dir=source_stack_dir,
        challenge_root=challenge_root.resolve(),
        train_py=train_py,
        compress_sh=compress_sh,
        inflate_sh=inflate_sh,
        public_archive=archive,
    )


def source_tree_sha256(layout: Pr95SourceLayout) -> str:
    """Hash small source/config files, excluding checkpoints, videos, and weights."""

    roots = [
        layout.source_dir,
        layout.challenge_root / "pyproject.toml",
        layout.challenge_root / "uv.lock",
        layout.challenge_root / "README.md",
        layout.challenge_root / "public_test_video_names.txt",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORED_TREE_PARTS for part in path.parts):
                continue
            if path.suffix in {".pt", ".pth", ".bin", ".zip", ".mkv", ".safetensors"}:
                continue
            files.append(path)
    h = hashlib.sha256()
    for path in sorted(set(files), key=lambda p: str(p.relative_to(layout.challenge_root))):
        rel = str(path.relative_to(layout.challenge_root)).encode("utf-8")
        h.update(len(rel).to_bytes(4, "little"))
        h.update(rel)
        data = path.read_bytes()
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def parse_stage_epoch_overrides(raw: list[str]) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"stage override must be N=EPOCHS, got {item!r}")
        left, right = item.split("=", 1)
        stage = int(left)
        epochs = int(right)
        if stage < 1 or stage > len(STAGE_MODULES):
            raise ValueError(f"stage index {stage} outside 1..{len(STAGE_MODULES)}")
        if epochs < 1:
            raise ValueError("stage epochs must be positive")
        overrides[stage] = epochs
    return overrides


def select_device(requested: str, *, allow_mps_fallback: bool) -> Any:
    import torch

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested mps but torch.backends.mps.is_available() is false")
    if (
        requested == "mps"
        and _truthy(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        and not allow_mps_fallback
    ):
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing MPS probe because "
            "silent CPU fallback would destroy portability evidence"
        )
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device {requested!r}")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _install_pr95_imports(layout: Pr95SourceLayout) -> None:
    os.environ["COMMA_CHALLENGE_ROOT"] = str(layout.challenge_root)
    for path in (layout.source_stack_dir, layout.challenge_root, layout.challenge_root.parent):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _stage_builder_modules() -> list[Any]:
    from stages import (  # type: ignore[import-not-found]
        stage1_v328_ce,
        stage2_v331_softplus,
        stage3_v332_smooth,
        stage4_v332_qat,
        stage5_c1a_l7,
        stage6_lambda_sweep,
        stage7_sigma_sweep,
        stage8_muon_finetune,
    )

    return [
        stage1_v328_ce,
        stage2_v331_softplus,
        stage3_v332_smooth,
        stage4_v332_qat,
        stage5_c1a_l7,
        stage6_lambda_sweep,
        stage7_sigma_sweep,
        stage8_muon_finetune,
    ]


def _recommended_execution_command(
    *,
    output_dir: Path,
    device: str,
    full_curriculum: bool,
    stage_limit: int,
    stage_epoch_overrides: dict[int, int],
    eval_every: int | None,
    allow_mps_fallback: bool,
    seed: int,
    source_dir: Path,
    public_archive: Path | None,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/run_pr95_local_training_probe.py",
        "--source-dir",
        _rel(source_dir),
        "--output-dir",
        _rel(output_dir),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if public_archive is not None:
        command.extend(["--public-archive", _rel(public_archive)])
    if full_curriculum:
        command.append("--full-curriculum")
    else:
        command.extend(["--stage-limit", str(stage_limit)])
    for stage_index, epochs in sorted(stage_epoch_overrides.items()):
        command.extend(["--stage-epochs", f"{stage_index}={epochs}"])
    if eval_every is not None:
        command.extend(["--eval-every", str(eval_every)])
    if allow_mps_fallback:
        command.append("--allow-mps-fallback")
    return command


def build_plan(
    *,
    layout: Pr95SourceLayout,
    output_dir: Path,
    device: str,
    full_curriculum: bool,
    stage_limit: int,
    stage_epoch_overrides: dict[int, int],
    eval_every: int | None,
    allow_mps_fallback: bool,
    seed: int,
) -> dict[str, Any]:
    selected_count = len(STAGE_MODULES) if full_curriculum else stage_limit
    selected_count = max(1, min(selected_count, len(STAGE_MODULES)))
    public_archive = (
        {
            "path": _rel(layout.public_archive),
            "sha256": _sha256_file(layout.public_archive),
            "bytes": layout.public_archive.stat().st_size,
        }
        if layout.public_archive is not None
        else None
    )
    stages = []
    for index, name in enumerate(STAGE_MODULES[:selected_count], start=1):
        stages.append({
            "index": index,
            "module": name,
            "epoch_override": stage_epoch_overrides.get(index),
            "eval_every_override": eval_every,
        })
    command = _recommended_execution_command(
        output_dir=output_dir,
        device=device,
        full_curriculum=full_curriculum,
        stage_limit=stage_limit,
        stage_epoch_overrides=stage_epoch_overrides,
        eval_every=eval_every,
        allow_mps_fallback=allow_mps_fallback,
        seed=seed,
        source_dir=layout.source_dir,
        public_archive=layout.public_archive,
    )
    return {
        "schema": "pr95_local_training_probe_plan_v1",
        "lane_id": LANE_ID,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_dir": _rel(layout.source_dir),
        "source_stack_dir": _rel(layout.source_stack_dir),
        "challenge_root": _rel(layout.challenge_root),
        "source_tree_sha256": source_tree_sha256(layout),
        "public_archive": public_archive,
        "output_dir": _rel(output_dir),
        "device_requested": device,
        "allow_mps_fallback": allow_mps_fallback,
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "seed": seed,
        "full_curriculum": full_curriculum,
        "stage_count": selected_count,
        "stages": stages,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "local_training_portability_probe_advisory",
        "authority_contract": {
            "local_mps_or_cpu": "training_velocity_and_transfer_probe_only",
            "score_authority": "requires byte_closed_archive_replay_on_contest_CPU_and_contest_CUDA",
            "fallback_policy": "fail_closed_unless_allow_mps_fallback_for_debug",
        },
        "recommended_execution": {
            "schema": "local_training_recommended_execution.v1",
            "tool": "tools/run_pr95_local_training_probe.py",
            "training_backend": "torch",
            "device": device,
            "resource_kind": (
                "local_cuda"
                if device == "cuda"
                else "local_mps"
                if device == "mps"
                else "local_cpu"
                if device == "cpu"
                else "local"
            ),
            "output_manifest": _rel(output_dir / "manifest.json"),
            "representation_manifest": _rel(
                output_dir / "representation_training_manifest.json"
            ),
            "plan_manifest": _rel(output_dir / "plan.json"),
            "python_command_args": command,
            "candidate_generation_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "source_faithful_command": (
            f"cd {layout.challenge_root} && COMMA_CHALLENGE_ROOT=$PWD "
            "python -m submissions.hnerv_muon.src.train"
        ),
    }


def _stage_modules_from_payload(payload: dict[str, Any]) -> list[str]:
    modules: list[str] = []
    for stage in payload.get("stages") or []:
        if isinstance(stage, dict) and stage.get("module"):
            modules.append(str(stage["module"]))
    for result in payload.get("results") or []:
        if isinstance(result, dict) and result.get("stage_module"):
            module = str(result["stage_module"])
            if module not in modules:
                modules.append(module)
    return modules


def _pr95_generic_candidate_id(payload: dict[str, Any]) -> str:
    if payload.get("candidate_id"):
        return str(payload["candidate_id"])
    device = payload.get("device_selected") or payload.get("device_requested") or "unknown"
    stage_count = payload.get("stage_count") or len(_stage_modules_from_payload(payload)) or "unknown"
    seed = payload.get("seed") or "unknown"
    return f"pr95_muon_hnerv_local_{device}_stages{stage_count}_seed{seed}"


def _write_pr95_representation_training_sidecar(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    schema: str,
    filename: str,
) -> Path:
    stage_modules = _stage_modules_from_payload(payload)
    archive_zip = payload.get("archive_zip")
    auth_eval_bridge = payload.get("auth_eval_bridge")
    stage_count = int(payload.get("stage_count") or len(stage_modules) or 0)
    sidecar_path = output_dir / filename
    write_representation_training_probe_manifest(
        sidecar_path,
        schema=schema,
        candidate_id=_pr95_generic_candidate_id(payload),
        lane_id=str(payload.get("lane_id") or LANE_ID),
        lane_class="pr95_hnerv_muon_local_training_proxy",
        candidate_family="pr95_hnerv_muon_training_probe",
        representation_family="hnerv",
        substrate_family="nerv_family",
        profile="pr95_hnerv_muon_training_smoke",
        param_schema="pr95_hnerv_muon_local_training_manifest_params_v1",
        training_signal_kind="local_representation_training_optimizer_schedule_probe",
        seed=payload.get("seed"),
        device_requested=payload.get("device_requested"),
        device_selected=payload.get("device_selected"),
        source_tree_sha256=payload.get("source_tree_sha256"),
        output_dir=payload.get("output_dir") or _rel(output_dir),
        stages=payload.get("stages") or [],
        results=payload.get("results") or [],
        stage_count=stage_count,
        training_recipe={
            "id": "pr95_source_faithful_curriculum",
            "full_curriculum": bool(payload.get("full_curriculum", False)),
            "stage_count": stage_count,
        },
        optimizer_recipe={
            "id": "pr95_stage8_muon_partition",
            "uses_stage8_muon": "stage8_muon_finetune" in stage_modules,
            "hidden_2d_plus_weights": "Muon",
            "bias_norm_scalar_stem_rgb_head": "AdamW",
        },
        scheduler_recipe={
            "id": "pr95_source_faithful_stage_schedules",
            "stage_epoch_overrides": {
                str(stage.get("index")): stage.get("epoch_override")
                for stage in payload.get("stages") or []
                if isinstance(stage, dict) and stage.get("epoch_override") is not None
            },
        },
        candidate_params={
            "stage_count": stage_count,
            "seed": payload.get("seed"),
            "device": payload.get("device_selected") or payload.get("device_requested"),
            "full_curriculum": bool(payload.get("full_curriculum", False)),
            "uses_stage8_muon": "stage8_muon_finetune" in stage_modules,
            "archive_exported": isinstance(archive_zip, dict),
            "auth_eval_bridge_present": isinstance(auth_eval_bridge, dict),
        },
        archive_zip=archive_zip if isinstance(archive_zip, dict) else None,
        auth_eval_bridge=auth_eval_bridge if isinstance(auth_eval_bridge, dict) else None,
        dispatch_blockers=[
            "pr95_local_training_probe_is_proxy_signal",
            "requires_byte_closed_archive_export",
            "requires_runtime_consumption_proof",
            "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            "requires_lane_claim_before_dispatch",
        ],
        evidence_grade=str(
            payload.get("evidence_grade") or "local_training_portability_probe_advisory"
        ),
        source_anchor="PR95 HNeRV Muon local training probe manifest",
        score_lowering_hypothesis=(
            "Use source-faithful PR95/HNeRV training telemetry to rank Muon, "
            "scheduler, and archive-export variants before exact auth replay."
        ),
        variant_axes=[
            "source_faithful_control",
            "optimizer_recipe",
            "scheduler_recipe",
            "stage_curriculum",
            "archive_export",
        ],
        paired_modes=[
            "source_faithful_control",
            "optimizer_variant",
            "scheduler_variant",
            "substrate_variant",
        ],
        extra_fields={
            "source_pr95_schema": payload.get("schema"),
            "public_archive": payload.get("public_archive"),
            "source_faithful_command": payload.get("source_faithful_command"),
            "authority_contract": payload.get("authority_contract"),
            "recommended_execution": payload.get("recommended_execution"),
            "ok": payload.get("ok"),
        },
    )
    return sidecar_path


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    layout = resolve_layout(args.source_dir, args.public_archive)
    output_dir = args.output_dir or (
        REPO_ROOT / "experiments/results" / f"pr95_local_training_probe_{_utc_stamp()}"
    )
    output_dir = output_dir.resolve()
    # Canonical HISTORICAL_PROVENANCE safety per Catalog #113 + anti-pattern
    # `research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1`
    # (registered 2026-05-28 from preflight audit; 24 of 77 violations were on
    # pr95_mlx_runtime_consumption_queue from re-running on existing output_dir).
    from tac.research_pipeline_output_dir_safety import (
        enforce_research_pipeline_output_dir,
    )

    enforce_research_pipeline_output_dir(
        output_dir,
        repo_root=REPO_ROOT,
        allow_overwrite_existing_historical_provenance=getattr(
            args, "allow_overwrite_existing_historical_provenance", False
        ),
        waiver_rationale=getattr(args, "overwrite_rationale", None),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_overrides = parse_stage_epoch_overrides(args.stage_epochs)
    plan = build_plan(
        layout=layout,
        output_dir=output_dir,
        device=args.device,
        full_curriculum=args.full_curriculum,
        stage_limit=args.stage_limit,
        stage_epoch_overrides=stage_overrides,
        eval_every=args.eval_every,
        allow_mps_fallback=args.allow_mps_fallback,
        seed=args.seed,
    )
    (output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    representation_plan_path = _write_pr95_representation_training_sidecar(
        plan,
        output_dir=output_dir,
        schema="representation_training_probe_plan_v1",
        filename="representation_training_plan.json",
    )
    if args.plan_only:
        return {
            "plan": plan,
            "manifest_path": str(output_dir / "plan.json"),
            "representation_training_plan_path": str(representation_plan_path),
        }

    import torch

    _install_pr95_imports(layout)
    device = select_device(args.device, allow_mps_fallback=args.allow_mps_fallback)
    _seed_everything(args.seed)

    from stages import codec_stage  # type: ignore[import-not-found]
    from stages.common import train_stage  # type: ignore[import-not-found]

    from data import get_default_video_path  # type: ignore[import-not-found]

    builders = _stage_builder_modules()
    selected_count = len(builders) if args.full_curriculum else args.stage_limit
    selected_count = max(1, min(selected_count, len(builders)))
    video_path = get_default_video_path()
    shared_state: dict[str, Any] = {}
    prev: Path | None = None
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    manifest = {
        **plan,
        "schema": "pr95_local_training_probe_manifest_v1",
        "device_selected": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "video_path": _rel(Path(video_path)),
        "started_utc": datetime.now(UTC).isoformat(),
        "ok": False,
        "results": results,
    }
    (output_dir / "manifest.started.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    try:
        for stage_index, module in enumerate(builders[:selected_count], start=1):
            stage_out = output_dir / f"stage{stage_index}_{STAGE_MODULES[stage_index - 1]}"
            if stage_index == 1:
                cfg = module.make_config(stage_out)
            else:
                assert prev is not None
                cfg = module.make_config(prev, stage_out)
            if stage_index in stage_overrides:
                cfg.epochs = stage_overrides[stage_index]
            elif not args.full_curriculum and not args.stage_epochs:
                cfg.epochs = 1
            if args.eval_every is not None:
                cfg.eval_every = args.eval_every
            stage_started = time.perf_counter()
            result = train_stage(cfg, device, video_path=video_path, shared_state=shared_state)
            stage_result = {
                **{
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in result.items()
                },
                "stage_index": stage_index,
                "stage_module": STAGE_MODULES[stage_index - 1],
                "epochs_run": cfg.epochs,
                "eval_every": cfg.eval_every,
                "wall_seconds": time.perf_counter() - stage_started,
            }
            results.append(stage_result)
            prev = stage_out
            (output_dir / "manifest.partial.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        if args.run_codec_stage:
            if prev is None:
                raise RuntimeError("codec stage requested before any training stage ran")
            codec_out = output_dir / "submission_archive"
            codec_started = time.perf_counter()
            codec_result = codec_stage.run_codec_stage(prev, codec_out, video_path)
            archive_zip = output_dir / "archive.zip"
            archive_zip_result = write_stored_single_member_zip(
                codec_out / "0.bin",
                archive_zip,
            )
            manifest["codec_stage"] = {
                **codec_result,
                "wall_seconds": time.perf_counter() - codec_started,
            }
            manifest["archive_zip"] = archive_zip_result
        if args.run_auth_eval:
            if "archive_zip" not in manifest:
                raise RuntimeError("--run-auth-eval requires --run-codec-stage")
            manifest["auth_eval_bridge"] = run_auth_eval_bridge(
                archive_zip=output_dir / "archive.zip",
                layout=layout,
                output_dir=output_dir,
                device=args.auth_eval_device,
                training_score=_latest_training_score(results),
                score_delta_tolerance=args.auth_eval_score_delta_tolerance,
                require_comparable=args.require_auth_eval_comparable,
                inflate_timeout=args.auth_eval_inflate_timeout,
                evaluate_timeout=args.auth_eval_evaluate_timeout,
            )
    except Exception as exc:
        manifest.update({
            "ok": False,
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "wall_seconds": time.perf_counter() - started,
            "finished_utc": datetime.now(UTC).isoformat(),
        })
        (output_dir / "manifest.failed.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        raise
    manifest.update({
        "ok": True,
        "wall_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(UTC).isoformat(),
    })
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    representation_manifest_path = _write_pr95_representation_training_sidecar(
        manifest,
        output_dir=output_dir,
        schema="representation_training_probe_manifest_v1",
        filename="representation_training_manifest.json",
    )
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "representation_training_manifest_path": str(representation_manifest_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--public-archive", type=Path, default=DEFAULT_PUBLIC_ARCHIVE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--allow-mps-fallback", action="store_true")
    parser.add_argument("--full-curriculum", action="store_true")
    parser.add_argument("--stage-limit", type=int, default=1)
    parser.add_argument(
        "--stage-epochs",
        action="append",
        default=[],
        metavar="N=EPOCHS",
        help="Override a stage epoch count, repeatable. Default smoke runs stage 1 for 1 epoch.",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-codec-stage", action="store_true")
    parser.add_argument(
        "--run-auth-eval",
        action="store_true",
        help="After --run-codec-stage, run canonical contest_auth_eval on the emitted archive.",
    )
    parser.add_argument(
        "--auth-eval-device",
        choices=["cuda", "cpu", "mps"],
        default="cpu",
        help=(
            "Device for --run-auth-eval. Local macOS CPU/MPS remain advisory; "
            "promotion still requires contest-axis custody."
        ),
    )
    parser.add_argument(
        "--auth-eval-score-delta-tolerance",
        type=float,
        default=1e-3,
        help="Diagnostic comparability threshold between training best_score and auth_eval canonical_score.",
    )
    parser.add_argument(
        "--require-auth-eval-comparable",
        action="store_true",
        help="Fail if auth_eval canonical_score differs from training best_score by more than the tolerance.",
    )
    parser.add_argument("--auth-eval-inflate-timeout", type=int, default=1800)
    parser.add_argument("--auth-eval-evaluate-timeout", type=int, default=1800)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--allow-overwrite-existing-historical-provenance",
        action="store_true",
        help=(
            "Opt-in to overwriting an existing .omx/research/<dir>/ that already "
            "contains canonical HISTORICAL_PROVENANCE JSON files. Per Catalog #113 + "
            "anti-pattern "
            "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1, "
            "the default behavior is fail-closed; requires --overwrite-rationale."
        ),
    )
    parser.add_argument(
        "--overwrite-rationale",
        type=str,
        default=None,
        help=(
            "Substantive operator rationale (>=4 chars; non-placeholder per "
            "Catalog #287) required when --allow-overwrite-existing-historical-provenance "
            "is set."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_probe(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
