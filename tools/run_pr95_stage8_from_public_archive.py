#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run PR95 Stage 8 from the public archive under byte-custody controls.

This is the scorer-faithful PR95 reference lane: it seeds the public PR95
``stage8_muon_finetune`` code from the released ``archive.zip`` itself, not from
MLX proxy weights, and any executed survivor is packaged back through the shared
archive-bound runtime bridge. Local PR95 scorer output remains advisory until a
contest CPU/CUDA exact eval signs the byte-closed packet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
DEFAULT_SOURCE_ARCHIVE_ZIP = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view"
    / "public_pr95_intake_20260505_auto/archive.zip"
)
DEFAULT_PUBLIC_SUBMISSION_ROOT = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view"
    / "public_pr95_intake_20260505_auto/source/submissions/hnerv_muon"
)
DEFAULT_CHALLENGE_ROOT = REPO_ROOT / "upstream"
DEFAULT_SOURCE_VIDEO_PATH = DEFAULT_CHALLENGE_ROOT / "videos/0.mkv"
DEFAULT_SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")

PR95_STAGE8_LANE_SCHEMA = "pr95_stage8_from_public_archive_lane.v1"
PR95_STAGE8_SEED_SCHEMA = "pr95_stage8_public_archive_seed.v1"
PR95_STAGE8_COMPARISON_SCHEMA = "compact_base_renderer_byte_grammar.v1"
LANE_ID = "pr95_stage8_from_public_archive"
BYTE_CEILINGS = (100_000, 178_417, 216_000, 285_000)

FALSE_AUTHORITY: dict[str, Any] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "dispatch_packet_ready": False,
}


class Pr95Stage8LaneError(RuntimeError):
    """Raised when the PR95 Stage-8 lane cannot preserve custody."""


@dataclass(frozen=True)
class Stage8Seed:
    seed_dir: Path
    decoder_pt: Path
    latents_pt: Path
    bundle_pt: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _torch_tensor_from_numpy(value: Any) -> Any:
    import numpy as np
    import torch

    return torch.from_numpy(np.asarray(value).copy())


def _resolve_default_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if not DEFAULT_SSD_ROOT.exists():
        raise Pr95Stage8LaneError(
            "ssd_default_unavailable_explicit_output_dir_required: expected "
            f"{DEFAULT_SSD_ROOT}; pass --output-dir to an external artifact tier"
        )
    return DEFAULT_SSD_ROOT / "pr95_stage8_from_public_archive" / _utc_now()


def _byte_ceiling_report(archive_bytes: int | None) -> dict[str, Any]:
    if archive_bytes is None:
        return {
            "schema": "byte_ceiling_report.v1",
            "archive_zip_bytes": None,
            "ceilings": [
                {"ceiling_bytes": ceiling, "status": "not_measured"}
                for ceiling in BYTE_CEILINGS
            ],
        }
    return {
        "schema": "byte_ceiling_report.v1",
        "archive_zip_bytes": int(archive_bytes),
        "ceilings": [
            {
                "ceiling_bytes": ceiling,
                "fits": int(archive_bytes) <= ceiling,
                "delta_bytes": int(archive_bytes) - ceiling,
            }
            for ceiling in BYTE_CEILINGS
        ],
    }


def prepare_stage8_seed_from_archive(
    *,
    source_archive_zip: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Stage8Seed:
    """Decode PR95 archive.zip into the exact files expected by Stage 8."""

    import torch

    from tac.local_acceleration.pr95_hnerv_mlx import parse_pr95_public_archive_zip

    source_archive_zip = Path(source_archive_zip)
    seed_dir = Path(output_dir) / "seed_from_public_archive"
    if seed_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"seed dir exists and overwrite is false: {seed_dir}"
            )
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    packet = parse_pr95_public_archive_zip(source_archive_zip)
    decoder_state = {
        key: _torch_tensor_from_numpy(value) for key, value in packet.state_dict.items()
    }
    latents = _torch_tensor_from_numpy(packet.latents)

    decoder_pt = seed_dir / "final_decoder.pt"
    latents_pt = seed_dir / "final_latents.pt"
    bundle_pt = seed_dir / "seed_stage8_public_archive_bundle.pt"
    torch.save(decoder_state, decoder_pt)
    torch.save(latents, latents_pt)
    torch.save({**decoder_state, "latents": latents}, bundle_pt)

    manifest = {
        "schema": PR95_STAGE8_SEED_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "lane_id": LANE_ID,
        "source_archive": packet.custody_manifest(),
        "seed_dir": seed_dir.as_posix(),
        "decoder_pt_path": decoder_pt.as_posix(),
        "decoder_pt_sha256": _sha256_file(decoder_pt),
        "decoder_pt_bytes": decoder_pt.stat().st_size,
        "latents_pt_path": latents_pt.as_posix(),
        "latents_pt_sha256": _sha256_file(latents_pt),
        "latents_pt_bytes": latents_pt.stat().st_size,
        "bundle_pt_path": bundle_pt.as_posix(),
        "bundle_pt_sha256": _sha256_file(bundle_pt),
        "bundle_pt_bytes": bundle_pt.stat().st_size,
        "stage8_expected_files": ["final_decoder.pt", "final_latents.pt"],
        "score_axis": "[macOS-CPU advisory]",
        "score_authority": "none",
        **FALSE_AUTHORITY,
    }
    manifest_path = seed_dir / "seed_manifest.json"
    _write_json(manifest_path, manifest)
    return Stage8Seed(
        seed_dir=seed_dir,
        decoder_pt=decoder_pt,
        latents_pt=latents_pt,
        bundle_pt=bundle_pt,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _clean_public_modules() -> None:
    prefixes = ("stages",)
    names = {
        "codec",
        "data",
        "losses",
        "model",
        "optim",
        "score",
        "stages",
    }
    for name in list(sys.modules):
        if name in names or any(name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]


def _select_torch_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_best_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _write_combined_checkpoint(
    *,
    decoder_pt: Path,
    latents_pt: Path,
    output_pt: Path,
) -> Path:
    import torch

    decoder_state = torch.load(decoder_pt, map_location="cpu", weights_only=True)
    latents = torch.load(latents_pt, map_location="cpu", weights_only=True)
    if not isinstance(decoder_state, dict):
        raise Pr95Stage8LaneError(
            f"decoder checkpoint must be a state_dict, got {type(decoder_state)}"
        )
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**decoder_state, "latents": latents}, output_pt)
    return output_pt


def _package_stage8_survivor(
    *,
    bundle_pt: Path,
    source_archive_zip: Path,
    public_submission_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from tools.package_pr95_mlx_pytorch_state_dict_to_contest_archive import (
        package_pytorch_state_dict_to_contest_archive,
    )

    submission_dir = output_dir / "byte_closed_submission"
    package_report_path = output_dir / "byte_closed_submission_report.json"
    return package_pytorch_state_dict_to_contest_archive(
        input_pt=bundle_pt,
        source_archive_zip=source_archive_zip,
        output_submission_dir=submission_dir,
        source_submission_root=public_submission_root,
        archive_bound_package_dir=output_dir / "archive_bound_candidate",
        latents_from_pt=True,
        overwrite=True,
        report_out=package_report_path,
    )


def build_compact_byte_grammar_reference(
    *,
    pr95_report: dict[str, Any],
) -> dict[str, Any]:
    """Return the shared comparison spine for compact renderer families."""

    reference_paths = [
        ".omx/research/codex_pr95_hnerv_faithful_stack_design_20260601T122712Z_subagentA.md",
        ".omx/research/codex_rnerv_srneerv_byte_grammar_stack_design_20260601T122839Z_subagentB.md",
        ".omx/research/codex_pvq_rt_vq_nerv_stack_design_20260601T122758Z_subagentC.md",
    ]
    rows: list[dict[str, Any]] = []
    for family in (
        "pr95_hnerv_stage8_from_public_archive",
        "rnerv",
        "srnerv",
        "boostnerv",
        "pvq_nerv",
        "rt_vq_nerv",
    ):
        executable = family == "pr95_hnerv_stage8_from_public_archive"
        rows.append(
            {
                "family": family,
                "byte_grammar": "pr95_style_single_archive_zip_member_or_equivalent_sectioned_packet",
                "required_sections": [
                    "trained_decoder_weights_or_program",
                    "trained_latents_or_indices",
                    "runtime_config",
                    "archive_manifest",
                    "receiver_proof",
                    "exact_gate_or_blocker",
                    "full_video_scorer_value_per_byte",
                ],
                "hard_byte_ceilings": list(BYTE_CEILINGS),
                "executable_now": executable,
                "entrypoint": (
                    "tools/run_pr95_stage8_from_public_archive.py --execute"
                    if executable
                    else None
                ),
                "status": "reference_executable" if executable else "adapter_required",
                "proxy_promotion_allowed": False,
                **FALSE_AUTHORITY,
            }
        )
    return {
        "schema": PR95_STAGE8_COMPARISON_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "authority_surface": "archive_zip_bytes_plus_receiver_proof_plus_exact_cpu_cuda_gate",
        "reference_lane": LANE_ID,
        "reference_report_path": pr95_report.get("report_path"),
        "reference_archive_zip_bytes": pr95_report.get("candidate_archive_zip_bytes"),
        "reference_exact_blockers": pr95_report.get("exact_gate", {}).get("blockers"),
        "subagent_design_references": reference_paths,
        "families": rows,
        **FALSE_AUTHORITY,
    }


def run_pr95_stage8_from_public_archive(
    *,
    source_archive_zip: Path,
    public_submission_root: Path,
    challenge_root: Path,
    source_video_path: Path,
    output_dir: Path,
    epochs: int,
    eval_every: int,
    batch_size: int,
    muon_weight_decay: float,
    device: str,
    execute: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = prepare_stage8_seed_from_archive(
        source_archive_zip=source_archive_zip,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    stage_dir = output_dir / "stage8_public_archive_finetune"
    package_report: dict[str, Any] | None = None
    local_training_result: dict[str, Any] | None = None
    stage_best_meta: dict[str, Any] | None = None
    blockers: list[str] = []
    archive_bytes: int | None = Path(source_archive_zip).stat().st_size

    if not execute:
        blockers.append("stage8_training_not_executed_plan_only")
    else:
        if stage_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"stage dir exists and overwrite is false: {stage_dir}"
                )
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        if int(epochs) <= 0:
            selected_device = device
            result = {
                "status": "source_seed_packaged_without_training",
                "public_stage8_train_stage_called": False,
                "reason": "epochs<=0 avoids public scheduler division by zero and proves custody only",
            }
            local_training_result = {
                "schema": "pr95_public_stage8_training_result.v1",
                "stage_dir": stage_dir.as_posix(),
                "device": selected_device,
                "epochs": int(epochs),
                "eval_every": int(eval_every),
                "batch_size": int(batch_size),
                "raw_result": result,
                "best_meta": None,
                "score_axis": "[macOS-CPU advisory]",
                **FALSE_AUTHORITY,
            }
            best_decoder = seed.decoder_pt
            best_latents = seed.latents_pt
            blockers.append("stage8_zero_epoch_source_seed_packaged_no_training")
        else:
            src_dir = Path(public_submission_root) / "src"
            stage8_file = src_dir / "stages/stage8_muon_finetune.py"
            if not stage8_file.is_file():
                raise FileNotFoundError(
                    f"public PR95 Stage-8 source missing: {stage8_file}"
                )
            if not Path(source_video_path).is_file():
                raise FileNotFoundError(f"source video missing: {source_video_path}")

            os.environ["COMMA_CHALLENGE_ROOT"] = str(Path(challenge_root))
            _clean_public_modules()
            sys.path.insert(0, str(src_dir))
            try:
                stage8 = importlib.import_module("stages.stage8_muon_finetune")
                common = importlib.import_module("stages.common")
                selected_device = _select_torch_device(device)
                import torch

                cfg = stage8.make_config(
                    seed.seed_dir,
                    stage_dir,
                    epochs=int(epochs),
                    muon_weight_decay=float(muon_weight_decay),
                )
                cfg.eval_every = int(eval_every)
                cfg.batch_size = int(batch_size)
                result = common.train_stage(
                    cfg,
                    torch.device(selected_device),
                    video_path=str(source_video_path),
                    shared_state={},
                )
            finally:
                try:
                    sys.path.remove(str(src_dir))
                except ValueError:
                    pass

            stage_best_meta = _load_best_meta(stage_dir / "best_meta.json")
            local_training_result = {
                "schema": "pr95_public_stage8_training_result.v1",
                "stage_dir": stage_dir.as_posix(),
                "device": selected_device,
                "epochs": int(epochs),
                "eval_every": int(eval_every),
                "batch_size": int(batch_size),
                "raw_result": result,
                "best_meta": stage_best_meta,
                "score_axis": "[macOS-CPU advisory]",
                **FALSE_AUTHORITY,
            }

            best_decoder = stage_dir / "decoder_f32.pt"
            best_latents = stage_dir / "latents_f32.pt"
            if not best_decoder.is_file() or not best_latents.is_file():
                best_decoder = stage_dir / "final_decoder.pt"
                best_latents = stage_dir / "final_latents.pt"
                blockers.append(
                    "stage8_best_checkpoint_missing_packaged_final_checkpoint"
                )
        survivor_bundle = _write_combined_checkpoint(
            decoder_pt=best_decoder,
            latents_pt=best_latents,
            output_pt=stage_dir / "stage8_survivor_bundle.pt",
        )
        package_report = _package_stage8_survivor(
            bundle_pt=survivor_bundle,
            source_archive_zip=Path(source_archive_zip),
            public_submission_root=Path(public_submission_root),
            output_dir=output_dir,
        )
        archive_bytes = int(package_report["archive_zip_bytes"])

    if package_report is None:
        blockers.extend(
            [
                "byte_closed_stage8_candidate_not_packaged",
                "receiver_proof_missing_until_packaged_archive_runs_inflate",
            ]
        )
    blockers.append("contest_cpu_cuda_exact_eval_missing")

    report_path = output_dir / "pr95_stage8_from_public_archive_report.json"
    report: dict[str, Any] = {
        "schema": PR95_STAGE8_LANE_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "lane_id": LANE_ID,
        "mode": "execute" if execute else "plan_only",
        "tool": "tools/run_pr95_stage8_from_public_archive.py",
        "report_path": report_path.as_posix(),
        "source_archive_zip": Path(source_archive_zip).as_posix(),
        "source_archive_zip_sha256": _sha256_file(Path(source_archive_zip)),
        "public_submission_root": Path(public_submission_root).as_posix(),
        "challenge_root": Path(challenge_root).as_posix(),
        "source_video_path": Path(source_video_path).as_posix(),
        "output_dir": output_dir.as_posix(),
        "reproducibility": {
            "schema": "pr95_stage8_reproducibility.v1",
            "argv_template": [
                "tools/run_pr95_stage8_from_public_archive.py",
                "--source-archive-zip",
                Path(source_archive_zip).as_posix(),
                "--public-submission-root",
                Path(public_submission_root).as_posix(),
                "--challenge-root",
                Path(challenge_root).as_posix(),
                "--source-video-path",
                Path(source_video_path).as_posix(),
                "--output-dir",
                output_dir.as_posix(),
                "--epochs",
                str(int(epochs)),
                "--eval-every",
                str(int(eval_every)),
                "--batch-size",
                str(int(batch_size)),
                "--muon-weight-decay",
                str(float(muon_weight_decay)),
                "--device",
                device,
            ]
            + (["--execute"] if execute else [])
            + (["--overwrite"] if overwrite else []),
            "env": {
                "COMMA_CHALLENGE_ROOT": Path(challenge_root).as_posix()
                if execute
                else None,
            },
            "storage_policy": {
                "default_output_root": DEFAULT_SSD_ROOT.as_posix(),
                "local_disk_default_allowed": False,
                "large_artifacts_are_rebuildable_from_seed_manifest": True,
            },
        },
        "seed": seed.manifest,
        "stage8_parameters": {
            "epochs": int(epochs),
            "eval_every": int(eval_every),
            "batch_size": int(batch_size),
            "muon_weight_decay": float(muon_weight_decay),
            "device": device,
        },
        "local_training_result": local_training_result,
        "package_report": package_report,
        "candidate_archive_zip_path": None
        if package_report is None
        else package_report["archive_zip_path"],
        "candidate_archive_zip_bytes": archive_bytes,
        "candidate_archive_zip_sha256": None
        if package_report is None
        else package_report["archive_zip_sha256"],
        "byte_ceilings": _byte_ceiling_report(archive_bytes),
        "score_axis": "[macOS-CPU advisory]",
        "score_authority": "none_until_contest_cpu_cuda_exact_eval",
        "exact_gate": {
            "schema": "exact_gate_blocker.v1",
            "ready_for_exact_eval_dispatch": False,
            "blockers": blockers,
        },
        **FALSE_AUTHORITY,
    }
    comparison = build_compact_byte_grammar_reference(pr95_report=report)
    comparison_path = output_dir / "compact_base_renderer_byte_grammar.json"
    _write_json(comparison_path, comparison)
    report["compact_base_renderer_byte_grammar_path"] = comparison_path.as_posix()
    _write_json(report_path, report)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive-zip",
        type=Path,
        default=DEFAULT_SOURCE_ARCHIVE_ZIP,
        help="Public PR95 archive.zip to resume Stage 8 from.",
    )
    parser.add_argument(
        "--public-submission-root",
        type=Path,
        default=DEFAULT_PUBLIC_SUBMISSION_ROOT,
        help="Public PR95 hnerv_muon submission root containing src/.",
    )
    parser.add_argument(
        "--challenge-root",
        type=Path,
        default=DEFAULT_CHALLENGE_ROOT,
        help="Comma challenge root used by the public PR95 scorer code.",
    )
    parser.add_argument(
        "--source-video-path",
        type=Path,
        default=DEFAULT_SOURCE_VIDEO_PATH,
        help="Contest source video used by the public PR95 training loop.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--muon-weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run public PR95 Stage 8. Default only prepares custody.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = _resolve_default_output_dir(args.output_dir)
    report = run_pr95_stage8_from_public_archive(
        source_archive_zip=args.source_archive_zip,
        public_submission_root=args.public_submission_root,
        challenge_root=args.challenge_root,
        source_video_path=args.source_video_path,
        output_dir=output_dir,
        epochs=args.epochs,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        muon_weight_decay=args.muon_weight_decay,
        device=args.device,
        execute=bool(args.execute),
        overwrite=bool(args.overwrite),
    )
    print(
        "[pr95-stage8] "
        f"mode={report['mode']} output={report['output_dir']} "
        f"archive_bytes={report['candidate_archive_zip_bytes']} "
        f"report={report['report_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
