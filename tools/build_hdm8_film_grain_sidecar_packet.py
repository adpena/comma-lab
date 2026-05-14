#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize an exact-evaluable HDM8 deterministic postfilter packet."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli  # type: ignore[import-not-found]

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import read_json, repo_relative, sha256_file, write_json  # noqa: E402
from tac.deploy.modal.auth_eval import modal_uploaded_submission_dir_runtime_manifest  # noqa: E402
from tac.reproducibility import (  # noqa: E402
    collect_source_transparency,
    transparency_report_markdown,
)

CONFIG_SCHEMA = "hdm8_film_grain_sidecar_postfilter_config_v1"
MANIFEST_SCHEMA = "hdm8_film_grain_sidecar_packet_manifest_v1"
RATE_DENOMINATOR_BYTES = 37_545_489
SIDECAR_MAGIC = 0xFE
SIDECAR_FORMAT_PR101_GRAMMAR = 0x02
SIDECAR_FORMAT_PR101_SELECTOR = 0x03
SIDECAR_FORMAT_PR101_SELECTOR_BROTLI = 0x04
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/"
    "exact_eval_static_release_surface/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/hdm8_film_grain_20260514_codex"
DEFAULT_RUNTIME_TEMPLATE = REPO_ROOT / "submissions/hdm8_film_grain_sidecar"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--runtime-template", type=Path, default=DEFAULT_RUNTIME_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", default="none")
    parser.add_argument("--proxy-json", type=Path)
    parser.add_argument(
        "--selector-from-proxy-json",
        action="store_true",
        help="Build a per-pair selector config from proxy JSON pair component arrays.",
    )
    parser.add_argument(
        "--pack-selector-into-archive",
        action="store_true",
        help="Store selector config inside the archive payload so selector bytes are charged.",
    )
    parser.add_argument(
        "--selector-codec",
        choices=["json", "brotli"],
        default="brotli",
        help="Archive selector payload codec when --pack-selector-into-archive is used.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--require-positive-proxy",
        action="store_true",
        help="Fail unless --proxy-json shows this exact mode beating none.",
    )
    return parser.parse_args(argv)


def _load_contest_auth_eval_module() -> Any:
    path = REPO_ROOT / "experiments/contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location("contest_auth_eval_runtime_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import runtime manifest helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proxy_verdict(proxy_json: Path | None, mode: str) -> dict[str, Any]:
    if proxy_json is None:
        return {
            "present": False,
            "mode": mode,
            "positive": False,
            "blocker": "positive_proxy_json_missing",
        }
    payload = read_json(proxy_json)
    by_mode = {str(item.get("mode")): item for item in payload.get("modes", [])}
    selected = by_mode.get(mode)
    baseline = by_mode.get("none")
    if selected is None:
        return {
            "present": True,
            "path": repo_relative(proxy_json, REPO_ROOT),
            "mode": mode,
            "positive": False,
            "blocker": "mode_missing_from_proxy_json",
        }
    if "delta_vs_none" in selected:
        delta = float(selected["delta_vs_none"])
    elif baseline is not None:
        delta = float(selected["score_proxy"]) - float(baseline["score_proxy"])
    else:
        delta = float("inf")
    best = payload.get("best") if isinstance(payload.get("best"), dict) else {}
    return {
        "present": True,
        "path": repo_relative(proxy_json, REPO_ROOT),
        "axis": payload.get("axis"),
        "n_pairs": payload.get("n_pairs"),
        "mode": mode,
        "mode_score_proxy": selected.get("score_proxy"),
        "delta_vs_none": delta,
        "best_mode": best.get("mode"),
        "best_delta_vs_none": best.get("delta_vs_none"),
        "positive": delta < 0.0 and mode != "none",
    }


def _score_from_components(*, avg_pose: float, avg_seg: float, archive_bytes: int) -> float:
    return (
        100.0 * avg_seg
        + (10.0 * max(0.0, avg_pose)) ** 0.5
        + 25.0 * archive_bytes / RATE_DENOMINATOR_BYTES
    )


def _selector_from_proxy_json(proxy_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(proxy_json)
    modes = payload.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"{proxy_json} does not contain non-empty modes list")

    baseline = next((item for item in modes if item.get("mode") == "none"), None)
    if baseline is None:
        raise ValueError(f"{proxy_json} does not contain a 'none' baseline mode")
    n_pairs = int(payload.get("n_pairs") or baseline.get("n_pairs") or 0)
    if n_pairs <= 0:
        raise ValueError(f"{proxy_json} does not contain a positive n_pairs")

    palette: list[str] = []
    pair_pose_by_mode: list[list[float]] = []
    pair_seg_by_mode: list[list[float]] = []
    for item in modes:
        mode = str(item.get("mode"))
        pose_values = item.get("pair_posenet_dist")
        seg_values = item.get("pair_segnet_dist")
        if not isinstance(pose_values, list) or not isinstance(seg_values, list):
            raise ValueError(
                f"mode {mode!r} in {proxy_json} lacks pair_posenet_dist/pair_segnet_dist; "
                "rerun proxy with --include-per-pair"
            )
        if len(pose_values) != n_pairs or len(seg_values) != n_pairs:
            raise ValueError(
                f"mode {mode!r} pair arrays have lengths "
                f"{len(pose_values)}/{len(seg_values)}, expected {n_pairs}"
            )
        palette.append(mode)
        pair_pose_by_mode.append([float(x) for x in pose_values])
        pair_seg_by_mode.append([float(x) for x in seg_values])

    baseline_pose = float(baseline["avg_posenet_dist"])
    pose_weight = 5.0 / max((10.0 * baseline_pose) ** 0.5, 1e-9)
    selector_indices: list[int] = []
    selected_pose: list[float] = []
    selected_seg: list[float] = []
    for pair_idx in range(n_pairs):
        best_idx = min(
            range(len(palette)),
            key=lambda mode_idx: (
                100.0 * pair_seg_by_mode[mode_idx][pair_idx]
                + pose_weight * pair_pose_by_mode[mode_idx][pair_idx]
            ),
        )
        selector_indices.append(best_idx)
        selected_pose.append(pair_pose_by_mode[best_idx][pair_idx])
        selected_seg.append(pair_seg_by_mode[best_idx][pair_idx])

    archive_bytes = int(payload.get("archive_bytes") or 0)
    if archive_bytes <= 0:
        raise ValueError(f"{proxy_json} lacks positive archive_bytes")
    avg_pose = sum(selected_pose) / n_pairs
    avg_seg = sum(selected_seg) / n_pairs
    selector_score = _score_from_components(
        avg_pose=avg_pose,
        avg_seg=avg_seg,
        archive_bytes=archive_bytes,
    )
    baseline_score = float(baseline["score_proxy"])
    histogram: dict[str, int] = {}
    for idx in selector_indices:
        histogram[palette[idx]] = histogram.get(palette[idx], 0) + 1

    config = {
        "schema": CONFIG_SCHEMA,
        "mode": "selector",
        "palette": palette,
        "selector_indices": selector_indices,
        "score_claim": False,
    }
    config_bytes = len(json.dumps(config, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    proxy = {
        "present": True,
        "path": repo_relative(proxy_json, REPO_ROOT),
        "axis": payload.get("axis"),
        "n_pairs": n_pairs,
        "mode": "selector",
        "mode_score_proxy": selector_score,
        "delta_vs_none": selector_score - baseline_score,
        "baseline_score_proxy": baseline_score,
        "avg_posenet_dist": avg_pose,
        "avg_segnet_dist": avg_seg,
        "baseline_avg_posenet_dist": baseline.get("avg_posenet_dist"),
        "baseline_avg_segnet_dist": baseline.get("avg_segnet_dist"),
        "selector_histogram": histogram,
        "selector_config_bytes_if_charged": config_bytes,
        "selector_rate_score_if_charged": 25.0 * config_bytes / RATE_DENOMINATOR_BYTES,
        "positive": selector_score < baseline_score,
        "compliance_risk": "selector_indices_are_video_side_information_until_packed_into_archive",
    }
    config["proxy"] = proxy
    return config, proxy


def _copy_runtime_template(runtime_template: Path, runtime_out: Path) -> None:
    if runtime_out.exists():
        raise FileExistsError(
            f"{runtime_out} already exists; choose a fresh --output-dir for a fixed packet"
        )
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "archive.zip")
    shutil.copytree(runtime_template, runtime_out, ignore=ignore)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    return info


def _read_single_member_payload(archive: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one archive member in {archive}, found {len(infos)}")
        return infos[0].filename, zf.read(infos[0].filename)


def _pack_selector_archive(
    source_archive: Path,
    output_archive: Path,
    config: dict[str, Any],
    *,
    selector_codec: str = "brotli",
) -> dict[str, Any]:
    member_name, payload = _read_single_member_payload(source_archive)
    if len(payload) < 2 or payload[0] != SIDECAR_MAGIC:
        raise ValueError("HDM8 selector packing expects sidecar payload with 0xFE magic")
    if payload[1] != SIDECAR_FORMAT_PR101_GRAMMAR:
        raise ValueError(
            f"HDM8 selector packing expects format_id=0x02, got 0x{payload[1]:02X}"
        )
    selector_config = {
        "schema": CONFIG_SCHEMA,
        "mode": "selector",
        "palette": list(config["palette"]),
        "selector_indices": list(config["selector_indices"]),
        "score_claim": False,
    }
    selector_bytes = json.dumps(
        selector_config,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    raw_selector_bytes = selector_bytes
    if selector_codec == "brotli":
        selector_bytes = brotli.compress(selector_bytes, quality=11)
        format_id = SIDECAR_FORMAT_PR101_SELECTOR_BROTLI
    elif selector_codec == "json":
        format_id = SIDECAR_FORMAT_PR101_SELECTOR
    else:
        raise ValueError(f"unsupported selector codec {selector_codec!r}")
    if len(selector_bytes) > 65_535:
        raise ValueError(f"selector payload too large for u16 trailer: {len(selector_bytes)} bytes")
    packed_payload = (
        bytes([SIDECAR_MAGIC, format_id])
        + payload[2:]
        + struct.pack("<H", len(selector_bytes))
        + selector_bytes
    )
    with zipfile.ZipFile(output_archive, "w") as zf:
        zf.writestr(_zip_info(member_name), packed_payload)
    return {
        "member_name": member_name,
        "source_payload_bytes": len(payload),
        "packed_payload_bytes": len(packed_payload),
        "selector_payload_bytes": len(selector_bytes),
        "selector_encoded_bytes": len(selector_bytes),
        "selector_json_bytes": len(raw_selector_bytes),
        "selector_raw_json_bytes": len(raw_selector_bytes),
        "selector_codec": selector_codec,
        "archive_byte_delta_vs_source": output_archive.stat().st_size
        - source_archive.stat().st_size,
        "format_id": f"0x{format_id:02X}",
    }


def _with_archive_charged_proxy(
    proxy: dict[str, Any],
    *,
    archive_bytes: int,
    selector_pack_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    charged = dict(proxy)
    if not {"avg_posenet_dist", "avg_segnet_dist", "baseline_score_proxy"} <= set(charged):
        return charged
    charged_score = _score_from_components(
        avg_pose=float(charged["avg_posenet_dist"]),
        avg_seg=float(charged["avg_segnet_dist"]),
        archive_bytes=int(archive_bytes),
    )
    charged_delta = charged_score - float(charged["baseline_score_proxy"])
    charged.update(
        {
            "archive_bytes_if_charged": int(archive_bytes),
            "mode_score_proxy_charged": charged_score,
            "delta_vs_none_charged": charged_delta,
            "positive_charged": charged_delta < 0.0,
        }
    )
    if selector_pack_manifest is not None:
        selector_payload_bytes = int(
            selector_pack_manifest.get(
                "selector_payload_bytes",
                selector_pack_manifest.get("selector_json_bytes", 0),
            )
        )
        charged.update(
            {
                "selector_payload_bytes_charged": selector_payload_bytes,
                "selector_rate_score_charged": 25.0
                * selector_payload_bytes
                / RATE_DENOMINATOR_BYTES,
                "selector_archive_byte_delta_vs_source": selector_pack_manifest.get(
                    "archive_byte_delta_vs_source"
                ),
                "selector_codec": selector_pack_manifest.get("selector_codec"),
            }
        )
    return charged


def _write_markdown_manifest(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# HDM8 Film-Grain/Postfilter Runtime Packet",
        "",
        f"- mode: `{manifest['postfilter_mode']}`",
        f"- archive: `{manifest['archive']['path']}`",
        f"- archive_sha256: `{manifest['archive']['sha256']}`",
        f"- runtime_tree_sha256: `{manifest['runtime']['runtime_tree_sha256']}`",
        f"- static_packet_ready: `{manifest['static_packet_ready']}`",
        f"- research_only: `{manifest['research_only']}`",
        f"- positive_proxy_candidate_for_cuda_probe: "
        f"`{manifest['positive_proxy_candidate_for_cuda_probe']}`",
        f"- ready_for_exact_cuda_after_positive_proxy: "
        f"`{manifest['ready_for_exact_cuda_after_positive_proxy']}`",
        f"- dispatch_attempted: `{manifest['dispatch_attempted']}`",
        f"- score_claim: `{manifest['score_claim']}`",
        "",
        "## Readiness",
        "",
        "This packet is scorer-free at inflate and consumes the existing single-member "
        "HDM8 archive shape. Selector packets must pack selector bytes into the "
        "archive payload before exact eval so video side information is charged.",
        "",
        "## Risks",
        "",
        "- CPU/MPS proxy selectors can invert on CUDA; contest-CUDA replay is the "
        "only promotion axis.",
        "- Runtime changes can move PoseNet and SegNet in opposite directions; every "
        "packet needs component-level exact-CUDA classification before promotion.",
        "- Selector mode is submission-grade only when `selector_packed_in_archive` "
        "is true in the packet manifest.",
        "",
        "## Experiment Transparency",
        "",
        f"- architecture: {manifest['experiment_transparency']['architecture']}",
        f"- training_curriculum: {manifest['experiment_transparency']['training_curriculum']}",
        f"- experiment_axis: `{manifest['experiment_transparency']['experiment_axis']}`",
        f"- deployment_path: `{manifest['experiment_transparency']['deployment_path']}`",
        f"- eval_contract: {manifest['experiment_transparency']['eval_contract']}",
        "",
        transparency_report_markdown(manifest["source_transparency"]),
        "",
        "## CUDA Command",
        "",
        "```bash",
        " ".join(manifest["exact_cuda_auth_eval_command_template"]),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _packet_build_command_template(
    *,
    archive: Path,
    runtime_template: Path,
    output_dir: Path,
    mode: str,
    proxy_json: Path | None,
    selector_from_proxy_json: bool,
    pack_selector_into_archive: bool,
    selector_codec: str,
    require_positive_proxy: bool,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/build_hdm8_film_grain_sidecar_packet.py",
        "--archive",
        repo_relative(archive, REPO_ROOT),
        "--runtime-template",
        repo_relative(runtime_template, REPO_ROOT),
        "--output-dir",
        repo_relative(output_dir, REPO_ROOT),
        "--mode",
        mode,
        "--selector-codec",
        selector_codec,
    ]
    if proxy_json is not None:
        command.extend(["--proxy-json", repo_relative(proxy_json, REPO_ROOT)])
    if selector_from_proxy_json:
        command.append("--selector-from-proxy-json")
    if pack_selector_into_archive:
        command.append("--pack-selector-into-archive")
    if require_positive_proxy:
        command.append("--require-positive-proxy")
    return command


def build_packet(
    *,
    archive: Path,
    runtime_template: Path,
    output_dir: Path,
    mode: str,
    proxy_json: Path | None,
    selector_from_proxy_json: bool = False,
    pack_selector_into_archive: bool = False,
    selector_codec: str = "brotli",
    require_positive_proxy: bool = False,
) -> dict[str, Any]:
    archive = archive.resolve()
    runtime_template = runtime_template.resolve()
    output_dir = output_dir.resolve()
    proxy_json = proxy_json.resolve() if proxy_json is not None else None
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    if not runtime_template.exists():
        raise FileNotFoundError(f"runtime template not found: {runtime_template}")

    selector_config: dict[str, Any] | None = None
    if selector_from_proxy_json:
        if proxy_json is None:
            raise SystemExit("--selector-from-proxy-json requires --proxy-json")
        selector_config, proxy = _selector_from_proxy_json(proxy_json)
        mode = "selector"
    else:
        proxy = _proxy_verdict(proxy_json, mode)
    if pack_selector_into_archive and mode != "selector":
        raise SystemExit("--pack-selector-into-archive requires selector mode")
    if pack_selector_into_archive:
        proxy = dict(proxy)
        proxy["selector_packed_in_archive"] = True
        proxy["compliance_risk"] = None
    if require_positive_proxy and not proxy.get("positive"):
        raise SystemExit(
            f"mode {mode!r} lacks positive proxy evidence: {json.dumps(proxy, sort_keys=True)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_out = output_dir / "submission_dir"
    archive_out = output_dir / "archive.zip"
    _copy_runtime_template(runtime_template, runtime_out)
    if archive_out.exists():
        raise FileExistsError(
            f"{archive_out} already exists; choose a fresh --output-dir for a fixed packet"
        )
    archive_pack_manifest: dict[str, Any] | None = None
    if pack_selector_into_archive:
        if selector_config is None:
            raise SystemExit("--pack-selector-into-archive requires selector config")
        archive_pack_manifest = _pack_selector_archive(
            archive,
            archive_out,
            selector_config,
            selector_codec=selector_codec,
        )
    else:
        shutil.copy2(archive, archive_out)
    if pack_selector_into_archive:
        proxy = _with_archive_charged_proxy(
            proxy,
            archive_bytes=archive_out.stat().st_size,
            selector_pack_manifest=archive_pack_manifest,
        )
        if require_positive_proxy and not proxy.get("positive_charged"):
            raise SystemExit(
                f"mode {mode!r} lacks positive charged proxy evidence: "
                f"{json.dumps(proxy, sort_keys=True)}"
            )

    config = (
        {
            "schema": CONFIG_SCHEMA,
            "mode": "none",
            "score_claim": False,
            "selector_source": "archive_payload_format_0x03",
            "proxy": proxy,
        }
        if pack_selector_into_archive
        else selector_config
    ) or {
        "schema": CONFIG_SCHEMA,
        "mode": mode,
        "score_claim": False,
        "proxy": proxy,
    }
    write_json(runtime_out / "postfilter_config.json", config)

    contest_auth_eval = _load_contest_auth_eval_module()
    runtime_manifest = contest_auth_eval._runtime_dependency_manifest(
        runtime_out / "inflate.sh",
        REPO_ROOT / "upstream",
        repo_root=REPO_ROOT,
    )
    modal_runtime_manifest = modal_uploaded_submission_dir_runtime_manifest(runtime_manifest)
    archive_manifest = {
        "schema": "hdm8_film_grain_sidecar_archive_manifest_v1",
        "archive_bytes": archive_out.stat().st_size,
        "archive_sha256": sha256_file(archive_out),
        "archive_source": repo_relative(archive, REPO_ROOT),
        "archive_path": repo_relative(archive_out, REPO_ROOT),
        "member_shape": (
            "single_member_x_hdm8_pr106_sidecar_wrapper_plus_archive_selector"
            if pack_selector_into_archive
            else "single_member_x_hdm8_pr106_sidecar_wrapper"
        ),
        "selector_packed_in_archive": pack_selector_into_archive,
        "selector_pack_manifest": archive_pack_manifest,
        "score_claim": False,
    }
    write_json(output_dir / "archive_manifest.json", archive_manifest)

    positive_proxy_candidate = bool(proxy.get("positive"))
    blockers: list[str] = []
    if mode == "none":
        blockers.append("postfilter_mode_none")
    if not positive_proxy_candidate:
        blockers.append(str(proxy.get("blocker") or "positive_proxy_missing_or_nonpositive"))
    else:
        blockers.append("positive_proxy_requires_cuda_transfer_confirmation")
    if mode == "selector" and not pack_selector_into_archive:
        blockers.append("selector_side_information_must_be_packed_into_archive_for_submission")
    blockers.extend(["lane_dispatch_claim_missing", "exact_cuda_auth_eval_missing"])

    exact_cmd = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/modal_auth_eval.py",
        "--archive",
        repo_relative(archive_out, REPO_ROOT),
        "--submission-dir",
        repo_relative(runtime_out, REPO_ROOT),
        "--inflate-sh",
        "inflate.sh",
        "--gpu",
        "T4",
        "--expected-runtime-tree-sha256",
        modal_runtime_manifest["runtime_tree_sha256"],
        "--detach",
        "--provider-detach-ack",
    ]
    claim_cmd = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        "hnerv_hdm8_film_grain_sidecar_exact_eval",
        "--platform",
        "modal",
        "--instance-job-id",
        "<job-id>",
        "--agent",
        "codex:gpt-5.5",
        "--status",
        "eval",
        "--notes",
        f"HDM8 postfilter mode={mode}; archive_sha256={archive_manifest['archive_sha256']}",
    ]
    build_cmd = _packet_build_command_template(
        archive=archive,
        runtime_template=runtime_template,
        output_dir=output_dir,
        mode=mode,
        proxy_json=proxy_json,
        selector_from_proxy_json=selector_from_proxy_json,
        pack_selector_into_archive=pack_selector_into_archive,
        selector_codec=selector_codec,
        require_positive_proxy=require_positive_proxy,
    )
    source_transparency = collect_source_transparency(
        repo_root=REPO_ROOT,
        source_paths=[
            Path(__file__),
            runtime_template / "inflate.py",
            runtime_template / "inflate.sh",
            archive,
            *([proxy_json] if proxy_json is not None else []),
        ],
        artifact_paths=[
            archive_out,
            output_dir / "archive_manifest.json",
            runtime_out / "inflate.py",
            runtime_out / "inflate.sh",
            runtime_out / "postfilter_config.json",
        ],
        commands=[
            build_cmd,
            claim_cmd,
            exact_cmd,
        ],
    )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "research_only": not bool(proxy.get("positive")),
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_attempted": False,
        "lane_id": "hnerv_hdm8_film_grain_sidecar_exact_eval",
        "postfilter_mode": mode,
        "selector_packed_in_archive": pack_selector_into_archive,
        "experiment_transparency": {
            "architecture": (
                "HDM8/PR106 HNeRV base archive plus PR101 ranked-Huffman sidecar "
                "grammar and deterministic scorer-free postfilter runtime"
            ),
            "training_curriculum": (
                "No new training is performed by this packet builder; it preserves "
                "the source archive weights/latents and applies an auditable "
                "post-decode transform selected from proxy component sweeps."
            ),
            "experiment_axis": "first-frame-safe deterministic postfilter selector",
            "deployment_path": "inflate.sh -> inflate.py -> scorer-free frame decode",
            "eval_contract": (
                "Proxy artifacts are advisory only; promotion requires byte-closed "
                "archive/runtime custody through contest-CUDA auth eval."
            ),
        },
        "archive": {
            "path": archive_manifest["archive_path"],
            "sha256": archive_manifest["archive_sha256"],
            "bytes": archive_manifest["archive_bytes"],
            "source": archive_manifest["archive_source"],
            "selector_pack_manifest": archive_pack_manifest,
        },
        "runtime": {
            "path": repo_relative(runtime_out, REPO_ROOT),
            "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
            "runtime_content_tree_sha256": runtime_manifest["runtime_content_tree_sha256"],
            "runtime_file_count": runtime_manifest["runtime_file_count"],
            "manifest": runtime_manifest,
            "modal_uploaded_runtime_tree_sha256": modal_runtime_manifest[
                "runtime_tree_sha256"
            ],
            "modal_uploaded_runtime_content_tree_sha256": modal_runtime_manifest[
                "runtime_content_tree_sha256"
            ],
            "modal_uploaded_manifest": modal_runtime_manifest,
        },
        "proxy": proxy,
        "static_packet_ready": True,
        "positive_proxy_candidate_for_cuda_probe": positive_proxy_candidate,
        "ready_for_exact_cuda_after_positive_proxy": False,
        "cuda_transfer_policy": {
            "schema": "hdm8_proxy_to_cuda_transfer_policy_v1",
            "proxy_axis": proxy.get("axis"),
            "positive_proxy_candidate_for_cuda_probe": positive_proxy_candidate,
            "rankable_on_cuda": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "required_before_ranking": [
                "byte_closed_archive_runtime_exact_contest_cuda_auth_eval",
                "non_positive_posenet_delta_on_contest_cuda",
                "non_positive_charged_score_delta_on_contest_cuda",
            ],
            "blockers": [
                "local_cpu_mps_proxy_not_score_truth",
                "fes1_hdm8_proxy_selectors_observed_cuda_pose_regression",
                "requires_exact_cuda_transfer_confirmation",
            ],
        },
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": blockers,
        "source_transparency": source_transparency,
        "packet_build_command_template": build_cmd,
        "claim_command_template": claim_cmd,
        "exact_cuda_auth_eval_command_template": exact_cmd,
        "risks": [
            "current local CPU proxy artifacts do not show a positive postfilter mode",
            "film-grain modes must be judged on contest-CUDA, not CPU proxy alone",
            "postfilter may trade PoseNet improvement for SegNet regression",
            "selector mode is an upper-bound probe until selector bytes are charged in the archive",
        ],
    }
    if pack_selector_into_archive:
        manifest["risks"] = [
            risk
            for risk in manifest["risks"]
            if risk != "selector mode is an upper-bound probe until selector bytes are charged in the archive"
        ]
    write_json(output_dir / "packet_manifest.json", manifest)
    _write_markdown_manifest(manifest, output_dir / "README.md")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_packet(
        archive=args.archive,
        runtime_template=args.runtime_template,
        output_dir=args.output_dir,
        mode=args.mode,
        proxy_json=args.proxy_json,
        selector_from_proxy_json=args.selector_from_proxy_json,
        pack_selector_into_archive=args.pack_selector_into_archive,
        selector_codec=args.selector_codec,
        require_positive_proxy=args.require_positive_proxy,
    )
    json_out = args.json_out or (args.output_dir / "packet_manifest.json")
    if json_out.resolve() != (args.output_dir.resolve() / "packet_manifest.json"):
        write_json(json_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
