#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a local-first HDM8 postfilter sweep before spending CUDA.

The workflow is deliberately staged:

1. Generate a broad first-frame-safe deterministic postfilter palette.
2. Run a cheap CPU guard slice and an MPS prefix sweep locally.
3. Promote only top/near-top local candidates to a full 600-pair MPS sweep.
4. Emit a compact Modal CUDA shortlist JSON and launch command.

All outputs are proxy artifacts with ``score_claim=false``. The only promoted
score path remains a charged archive/runtime packet through contest CUDA eval.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import read_json, repo_relative, sha256_file, write_json  # noqa: E402

DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/"
    "exact_eval_static_release_surface/archive.zip"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex"
)
SCREEN_TOOL = REPO_ROOT / "tools/screen_hdm8_postfilter_sweep.py"
RATE_DENOMINATOR_BYTES = 37_545_489
MODAL_SWEEP_SCRIPT = "experiments/modal_hdm8_postfilter_sweep.py"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def build_mode_palette(
    *,
    profile: str = "broad",
    include_last_frame_risky: bool = False,
) -> list[str]:
    """Build a deterministic HDM8 first-frame postfilter palette.

    By default only ``even_*`` modes are generated. In this archive runtime,
    frame pairs are emitted as ``(2p, 2p+1)`` and the contest SegNet path scores
    the second/last frame only, so first-frame-only modes are the safe search
    surface.
    """

    modes = ["none"]
    scalar = [
        -16,
        -12,
        -8,
        -6,
        -4,
        -3,
        -2,
        -1,
        -0.5,
        0.5,
        1,
        2,
        3,
        4,
        6,
        8,
        12,
        16,
    ]
    chroma_amps = [0.5, 1, 2, 3, 4, 6, 8]
    grain_amps = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
    if profile == "fast":
        scalar = [-8, -4, -2, -1, 1, 2, 4, 8]
        chroma_amps = [1, 2, 4]
        grain_amps = [0.5, 1, 2, 4]
    elif profile == "aggressive":
        scalar.extend([-24, -20, 20, 24])
        chroma_amps.extend([10, 12])
        grain_amps.extend([8, 10, 12])
    elif profile != "broad":
        raise ValueError(f"unsupported profile {profile!r}")

    first_prefixes = ["even"]
    if include_last_frame_risky:
        first_prefixes.append("odd")

    for prefix in first_prefixes:
        for value in scalar:
            modes.append(f"{prefix}_bias:{value:g}")
        for amp in chroma_amps:
            triples = [
                (amp, -0.5 * amp, -0.5 * amp),
                (-amp, 0.5 * amp, 0.5 * amp),
                (amp, 0, -amp),
                (-amp, 0, amp),
                (0, amp, -amp),
                (0, -amp, amp),
                (amp, -amp, 0),
                (-amp, amp, 0),
            ]
            for dr, dg, db in triples:
                modes.append(f"{prefix}_rgb_bias:{dr:g},{dg:g},{db:g}")
        contrast_values = [-0.10, -0.06, -0.03, -0.015, 0.015, 0.03, 0.06, 0.10]
        gamma_values = [0.94, 0.97, 0.985, 1.015, 1.03, 1.06]
        scale_amps = [0.005, 0.01, 0.02, 0.04]
        if profile == "fast":
            contrast_values = [-0.06, -0.03, 0.03, 0.06]
            gamma_values = [0.97, 1.03]
            scale_amps = [0.01, 0.02]
        elif profile == "aggressive":
            contrast_values.extend([-0.16, 0.16])
            gamma_values.extend([0.90, 1.10])
            scale_amps.extend([0.06, 0.08])
        for value in contrast_values:
            modes.append(f"{prefix}_contrast:{value:g}")
        for value in gamma_values:
            modes.append(f"{prefix}_gamma:{value:g}")
        for amp in scale_amps:
            scale_triples = [
                (1 + amp, 1 - 0.5 * amp, 1 - 0.5 * amp),
                (1 - amp, 1 + 0.5 * amp, 1 + 0.5 * amp),
                (1 + amp, 1, 1 - amp),
                (1 - amp, 1, 1 + amp),
                (1, 1 + amp, 1 - amp),
                (1, 1 - amp, 1 + amp),
            ]
            for sr, sg, sb in scale_triples:
                modes.append(f"{prefix}_rgb_scale:{sr:g},{sg:g},{sb:g}")
        for amp in grain_amps:
            for name in ("grain_luma", "grain_chroma", "grain_var", "blue", "checker", "tile_chroma"):
                modes.append(f"{prefix}_{name}:{amp:g}")
        for value in (0.03, 0.06, 0.10, 0.16, 0.24, 0.35):
            modes.append(f"{prefix}_unsharp:{value:g}")
            modes.append(f"{prefix}_soften:{value:g}")
        for value in (0.25, 0.50, 0.85, 1.20):
            modes.append(f"{prefix}_adaptive:{value:g}")
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                if dy or dx:
                    modes.append(f"{prefix}_translate:{dy},{dx}")

    if profile in {"broad", "aggressive"}:
        for amp in (1, 2, 4):
            modes.extend(
                [
                    f"even_rgb_bias:{amp:g},{-0.5 * amp:g},{-0.5 * amp:g}+even_grain_chroma:1",
                    f"even_rgb_bias:{-amp:g},{0.5 * amp:g},{0.5 * amp:g}+even_grain_chroma:1",
                    f"even_rgb_bias:{amp:g},0,{-amp:g}+even_blue:1",
                    f"even_rgb_bias:{-amp:g},0,{amp:g}+even_blue:1",
                ]
            )

    return _dedupe(modes)


def _score_value(row: dict[str, Any]) -> float:
    return float(row.get("score_proxy", float("inf")))


def _delta_value(row: dict[str, Any]) -> float:
    return float(row.get("delta_vs_none", float("inf")))


def select_modes_for_next_stage(
    payload: dict[str, Any],
    *,
    top_k: int,
    margin: float,
    required_modes: list[str] | None = None,
) -> list[str]:
    modes = payload.get("modes")
    if not isinstance(modes, list):
        raise ValueError("sweep payload lacks modes list")
    sorted_rows = sorted(
        [row for row in modes if isinstance(row, dict) and isinstance(row.get("mode"), str)],
        key=lambda row: (_score_value(row), str(row.get("mode"))),
    )
    selected = ["none"]
    if required_modes:
        selected.extend(required_modes)
    baseline = next((row for row in sorted_rows if row.get("mode") == "none"), None)
    cutoff = float(baseline.get("score_proxy", float("inf"))) + float(margin) if baseline else float("inf")
    for row in sorted_rows[: max(0, top_k)]:
        selected.append(str(row["mode"]))
    for row in sorted_rows:
        if _score_value(row) <= cutoff:
            selected.append(str(row["mode"]))
    return _dedupe(selected)


def select_cpu_guard_modes(modes: list[str], *, max_modes: int) -> list[str]:
    """Select a representative CPU guard subset from a broad palette.

    CPU is a deterministic sanity guard here; MPS owns the comprehensive local
    search surface. The subset keeps mechanism diversity so a broken transform
    family is still caught before full MPS/CUDA work.
    """

    if max_modes <= 0 or max_modes >= len(modes):
        return list(modes)
    anchors = [
        "none",
        "even_bias:-8",
        "even_bias:-4",
        "even_bias:-2",
        "even_bias:-1",
        "even_bias:1",
        "even_bias:2",
        "even_bias:4",
        "even_bias:8",
        "even_rgb_bias:1,-0.5,-0.5",
        "even_rgb_bias:-1,0.5,0.5",
        "even_rgb_bias:2,-1,-1",
        "even_rgb_bias:-2,1,1",
        "even_rgb_bias:4,-2,-2",
        "even_rgb_bias:-4,2,2",
        "even_contrast:-0.06",
        "even_contrast:0.06",
        "even_gamma:0.97",
        "even_gamma:1.03",
        "even_rgb_scale:1.02,0.99,0.99",
        "even_rgb_scale:0.98,1.01,1.01",
        "even_grain_chroma:0.5",
        "even_grain_chroma:1",
        "even_grain_chroma:2",
        "even_grain_luma:1",
        "even_blue:1",
        "even_checker:1",
        "even_tile_chroma:1",
        "even_tile_chroma:2",
        "even_tile_chroma:3",
        "even_translate:-1,0",
        "even_translate:1,0",
        "even_translate:0,-1",
        "even_translate:0,1",
        "even_rgb_bias:1,-0.5,-0.5+even_grain_chroma:1",
        "even_rgb_bias:-1,0.5,0.5+even_grain_chroma:1",
    ]
    selected = [mode for mode in anchors if mode in modes]
    if len(selected) < max_modes:
        stride = max(1, len(modes) // max(1, max_modes - len(selected)))
        selected.extend(modes[::stride])
    return _dedupe(selected)[:max_modes]


def _write_modes_json(path: Path, modes: list[str], *, source: str) -> None:
    write_json(
        path,
        {
            "schema": "hdm8_postfilter_mode_palette_v1",
            "source": source,
            "modes": modes,
            "score_claim": False,
            "promotion_eligible": False,
        },
    )


def _run_screen(
    *,
    archive: Path,
    output_json: Path,
    device: str,
    n_pairs: int,
    modes: list[str],
    include_per_pair: bool,
    decode_batch_pairs: int,
    score_batch_pairs: int,
    mode_batch_size: int,
) -> dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCREEN_TOOL),
        "--archive",
        str(archive),
        "--output-json",
        str(output_json),
        "--device",
        device,
        "--n-pairs",
        str(int(n_pairs)),
        "--decode-batch-pairs",
        str(int(decode_batch_pairs)),
        "--score-batch-pairs",
        str(int(score_batch_pairs)),
        "--mode-batch-size",
        str(int(mode_batch_size)),
    ]
    if include_per_pair:
        cmd.append("--include-per-pair")
    for mode in modes:
        cmd.extend(["--mode", mode])
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'upstream'}:{REPO_ROOT}",
    }
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_json.with_suffix(".stdout.log")).write_text(proc.stdout, encoding="utf-8")
    (output_json.with_suffix(".stderr.log")).write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{device} screen failed rc={proc.returncode}; see "
            f"{repo_relative(output_json.with_suffix('.stderr.log'), REPO_ROOT)}"
        )
    payload = read_json(output_json)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{output_json} did not contain a JSON object")
    payload["local_first_driver_elapsed_seconds"] = time.time() - started
    write_json(output_json, payload)
    return payload


def _screen_command(
    *,
    archive: Path,
    output_json: Path,
    device: str,
    n_pairs: int,
    modes: list[str],
    include_per_pair: bool,
    decode_batch_pairs: int,
    score_batch_pairs: int,
    mode_batch_size: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCREEN_TOOL),
        "--archive",
        str(archive),
        "--output-json",
        str(output_json),
        "--device",
        device,
        "--n-pairs",
        str(int(n_pairs)),
        "--decode-batch-pairs",
        str(int(decode_batch_pairs)),
        "--score-batch-pairs",
        str(int(score_batch_pairs)),
        "--mode-batch-size",
        str(int(mode_batch_size)),
    ]
    if include_per_pair:
        cmd.append("--include-per-pair")
    for mode in modes:
        cmd.extend(["--mode", mode])
    return cmd


def split_modes_for_shards(modes: list[str], *, shard_size: int) -> list[list[str]]:
    """Split modes into shards with a local ``none`` baseline in every shard."""

    deduped = _dedupe(modes)
    if "none" not in deduped:
        raise ValueError("mode shards require a 'none' baseline")
    if shard_size <= 0 or len(deduped) <= shard_size:
        return [deduped]
    payload_modes = [mode for mode in deduped if mode != "none"]
    chunk_size = max(1, shard_size - 1)
    return [["none", *payload_modes[start : start + chunk_size]] for start in range(0, len(payload_modes), chunk_size)]


def _rss_kb_for_pids(pids: list[int]) -> int:
    if not pids:
        return 0
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", ",".join(str(pid) for pid in pids)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return 0
    total = 0
    for line in out.splitlines():
        try:
            total += int(line.strip())
        except ValueError:
            continue
    return total


def merge_shard_payloads(
    payloads: list[dict[str, Any]],
    *,
    output_json: Path,
    started_at: float,
    shard_manifest: list[dict[str, Any]],
    max_observed_rss_kb: int,
) -> dict[str, Any]:
    """Merge screen shard JSONs into one canonical proxy payload."""

    if not payloads:
        raise ValueError("cannot merge zero shard payloads")
    first = payloads[0]
    merged_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    baseline: dict[str, Any] | None = None
    for payload in payloads:
        for key in ("schema", "score_claim", "promotion_eligible", "axis", "archive", "archive_bytes", "n_pairs"):
            if payload.get(key) != first.get(key):
                raise ValueError(f"shard payload mismatch for {key}: {payload.get(key)!r} != {first.get(key)!r}")
        for row in payload.get("modes", []):
            if not isinstance(row, dict) or not isinstance(row.get("mode"), str):
                continue
            mode = str(row["mode"])
            if mode == "none" and baseline is None:
                baseline = dict(row)
            if mode in seen:
                continue
            seen.add(mode)
            merged_rows.append(dict(row))
    if baseline is None:
        raise ValueError("merged shards missing 'none' baseline")
    baseline_score = float(baseline["score_proxy"])
    for row in merged_rows:
        row["delta_vs_none"] = float(row["score_proxy"]) - baseline_score
    merged = {
        "schema": first.get("schema"),
        "score_claim": False,
        "promotion_eligible": False,
        "axis": first.get("axis"),
        "archive": first.get("archive"),
        "archive_bytes": first.get("archive_bytes"),
        "n_pairs": first.get("n_pairs"),
        "elapsed_seconds": time.time() - started_at,
        "modes": merged_rows,
        "best": min(merged_rows, key=lambda item: float(item["score_proxy"])),
        "sharded": True,
        "shard_count": len(payloads),
        "shard_manifest": shard_manifest,
        "rss_guard": {
            "max_observed_rss_kb": max_observed_rss_kb,
            "max_observed_rss_gb": max_observed_rss_kb / (1024.0 * 1024.0),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, merged)
    return merged


def _run_screen_parallel(
    *,
    archive: Path,
    output_json: Path,
    device: str,
    n_pairs: int,
    modes: list[str],
    include_per_pair: bool,
    decode_batch_pairs: int,
    score_batch_pairs: int,
    mode_batch_size: int,
    workers: int,
    shard_size: int,
    max_rss_gb: float,
) -> dict[str, Any]:
    shards = split_modes_for_shards(modes, shard_size=shard_size)
    if workers <= 1 or len(shards) <= 1:
        return _run_screen(
            archive=archive,
            output_json=output_json,
            device=device,
            n_pairs=n_pairs,
            modes=modes,
            include_per_pair=include_per_pair,
            decode_batch_pairs=decode_batch_pairs,
            score_batch_pairs=score_batch_pairs,
            mode_batch_size=mode_batch_size,
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'upstream'}:{REPO_ROOT}",
    }
    max_rss_kb = int(max(0.1, max_rss_gb) * 1024 * 1024)
    active: list[dict[str, Any]] = []
    completed_payloads: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []
    next_idx = 0
    max_observed_rss_kb = 0
    started_at = time.time()

    while next_idx < len(shards) or active:
        active_pids = [int(item["proc"].pid) for item in active]
        current_rss = _rss_kb_for_pids(active_pids)
        max_observed_rss_kb = max(max_observed_rss_kb, current_rss)
        while (
            next_idx < len(shards)
            and len(active) < max(1, workers)
            and (not active or current_rss < max_rss_kb)
        ):
            shard_modes = shards[next_idx]
            shard_json = output_json.with_name(f"{output_json.stem}.shard{next_idx:03d}.json")
            cmd = _screen_command(
                archive=archive,
                output_json=shard_json,
                device=device,
                n_pairs=n_pairs,
                modes=shard_modes,
                include_per_pair=include_per_pair,
                decode_batch_pairs=decode_batch_pairs,
                score_batch_pairs=score_batch_pairs,
                mode_batch_size=mode_batch_size,
            )
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            active.append(
                {
                    "proc": proc,
                    "idx": next_idx,
                    "json": shard_json,
                    "modes": shard_modes,
                    "started_at": time.time(),
                }
            )
            next_idx += 1
            active_pids = [int(item["proc"].pid) for item in active]
            current_rss = _rss_kb_for_pids(active_pids)
            max_observed_rss_kb = max(max_observed_rss_kb, current_rss)

        for item in list(active):
            proc: subprocess.Popen[str] = item["proc"]
            if proc.poll() is None:
                continue
            stdout, stderr = proc.communicate()
            shard_json = Path(item["json"])
            shard_json.with_suffix(".stdout.log").write_text(stdout or "", encoding="utf-8")
            shard_json.with_suffix(".stderr.log").write_text(stderr or "", encoding="utf-8")
            elapsed = time.time() - float(item["started_at"])
            shard_manifest.append(
                {
                    "shard_index": item["idx"],
                    "path": repo_relative(shard_json, REPO_ROOT),
                    "mode_count": len(item["modes"]),
                    "returncode": proc.returncode,
                    "elapsed_seconds": elapsed,
                }
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{device} shard {item['idx']} failed rc={proc.returncode}; see "
                    f"{repo_relative(shard_json.with_suffix('.stderr.log'), REPO_ROOT)}"
                )
            payload = read_json(shard_json)
            if not isinstance(payload, dict):
                raise RuntimeError(f"{shard_json} did not contain a JSON object")
            completed_payloads.append(payload)
            active.remove(item)
        if active:
            time.sleep(0.5)

    return merge_shard_payloads(
        completed_payloads,
        output_json=output_json,
        started_at=started_at,
        shard_manifest=shard_manifest,
        max_observed_rss_kb=max_observed_rss_kb,
    )


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _charged_selector_rate_estimate(n_modes: int, n_pairs: int) -> dict[str, float | int]:
    bits_per_index = max(1, (max(1, n_modes) - 1).bit_length())
    raw_bits = bits_per_index * n_pairs
    raw_bytes = (raw_bits + 7) // 8
    return {
        "palette_modes": n_modes,
        "n_pairs": n_pairs,
        "bits_per_index": bits_per_index,
        "raw_selector_bytes_lower_bound": raw_bytes,
        "raw_selector_rate_score_lower_bound": 25.0 * raw_bytes / RATE_DENOMINATOR_BYTES,
    }


def _summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best") if isinstance(payload.get("best"), dict) else {}
    none = next(
        (
            row
            for row in payload.get("modes", [])
            if isinstance(row, dict) and row.get("mode") == "none"
        ),
        {},
    )
    return {
        "axis": payload.get("axis"),
        "n_pairs": payload.get("n_pairs"),
        "mode_count": len(payload.get("modes", []) or []),
        "none_score_proxy": none.get("score_proxy"),
        "none_avg_posenet_dist": none.get("avg_posenet_dist"),
        "none_avg_segnet_dist": none.get("avg_segnet_dist"),
        "best_mode": best.get("mode"),
        "best_score_proxy": best.get("score_proxy"),
        "best_delta_vs_none": best.get("delta_vs_none"),
        "best_avg_posenet_dist": best.get("avg_posenet_dist"),
        "best_avg_segnet_dist": best.get("avg_segnet_dist"),
    }


def run_local_first(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = build_mode_palette(
        profile=args.profile,
        include_last_frame_risky=args.include_last_frame_risky,
    )
    if args.max_modes:
        modes = modes[: int(args.max_modes)]
        if "none" not in modes:
            modes.insert(0, "none")
    _write_modes_json(out_dir / "local_first_full_palette.json", modes, source=args.profile)
    cpu_guard_modes = select_cpu_guard_modes(modes, max_modes=args.cpu_guard_max_modes)
    _write_modes_json(
        out_dir / "local_first_cpu_guard_modes.json",
        cpu_guard_modes,
        source=f"{args.profile}:representative_cpu_guard",
    )

    stages: list[dict[str, Any]] = []
    cpu_guard_payload = _run_screen_parallel(
        archive=archive,
        output_json=out_dir / "cpu_guard_sweep.json",
        device="cpu",
        n_pairs=args.cpu_guard_pairs,
        modes=cpu_guard_modes,
        include_per_pair=False,
        decode_batch_pairs=args.cpu_decode_batch_pairs,
        score_batch_pairs=args.cpu_score_batch_pairs,
        mode_batch_size=args.cpu_mode_batch_size,
        workers=args.cpu_parallel_workers,
        shard_size=args.cpu_shard_size,
        max_rss_gb=args.max_local_rss_gb,
    )
    stages.append({"stage": "cpu_guard", "path": "cpu_guard_sweep.json", **_summary_from_payload(cpu_guard_payload)})

    if args.skip_mps:
        mps_prefix_payload = cpu_guard_payload
    else:
        if not _mps_available():
            raise RuntimeError("MPS is not available; pass --skip-mps to run CPU-only")
        mps_prefix_payload = _run_screen_parallel(
            archive=archive,
            output_json=out_dir / "mps_prefix_sweep.json",
            device="mps",
            n_pairs=args.mps_prefix_pairs,
            modes=modes,
            include_per_pair=False,
            decode_batch_pairs=args.mps_decode_batch_pairs,
            score_batch_pairs=args.mps_score_batch_pairs,
            mode_batch_size=args.mps_mode_batch_size,
            workers=args.mps_prefix_parallel_workers,
            shard_size=args.mps_prefix_shard_size,
            max_rss_gb=args.max_local_rss_gb,
        )
        stages.append(
            {"stage": "mps_prefix", "path": "mps_prefix_sweep.json", **_summary_from_payload(mps_prefix_payload)}
        )

    full_modes = select_modes_for_next_stage(
        mps_prefix_payload,
        top_k=args.full_top_k,
        margin=args.full_margin,
        required_modes=select_modes_for_next_stage(
            cpu_guard_payload,
            top_k=args.cpu_carry_top_k,
            margin=args.cpu_carry_margin,
        ),
    )
    _write_modes_json(out_dir / "local_first_full_mps_modes.json", full_modes, source="cpu_guard+mps_prefix")

    full_payload = _run_screen_parallel(
        archive=archive,
        output_json=out_dir / "mps_full_sweep.json" if not args.skip_mps else out_dir / "cpu_full_sweep.json",
        device="cpu" if args.skip_mps else "mps",
        n_pairs=args.full_pairs,
        modes=full_modes,
        include_per_pair=True,
        decode_batch_pairs=args.mps_decode_batch_pairs if not args.skip_mps else args.cpu_decode_batch_pairs,
        score_batch_pairs=args.mps_score_batch_pairs if not args.skip_mps else args.cpu_score_batch_pairs,
        mode_batch_size=args.mps_mode_batch_size if not args.skip_mps else args.cpu_mode_batch_size,
        workers=args.cpu_parallel_workers if args.skip_mps else args.mps_full_parallel_workers,
        shard_size=args.cpu_shard_size if args.skip_mps else args.mps_full_shard_size,
        max_rss_gb=args.max_local_rss_gb,
    )
    stages.append(
        {
            "stage": "mps_full" if not args.skip_mps else "cpu_full",
            "path": "mps_full_sweep.json" if not args.skip_mps else "cpu_full_sweep.json",
            **_summary_from_payload(full_payload),
        }
    )

    cuda_modes = select_modes_for_next_stage(
        full_payload,
        top_k=args.cuda_top_k,
        margin=args.cuda_margin,
    )
    _write_modes_json(out_dir / "cuda_shortlist_modes.json", cuda_modes, source="local_first_full")

    run_id = f"hdm8_local_first_cuda_confirm_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    modal_output_dir = (
        "experiments/results/modal_hdm8_postfilter_sweep/"
        f"{run_id}"
    )
    modal_command = [
        "PYTHONPATH=src:upstream:$PWD",
        ".venv/bin/modal",
        "run",
        "--detach",
        MODAL_SWEEP_SCRIPT,
        "--archive",
        repo_relative(archive, REPO_ROOT),
        "--output-dir",
        modal_output_dir,
        "--n-pairs",
        str(args.full_pairs),
        "--modes-from-json",
        repo_relative(out_dir / "cuda_shortlist_modes.json", REPO_ROOT),
        "--include-per-pair",
        "--decode-batch-pairs",
        str(args.cuda_decode_batch_pairs),
        "--score-batch-pairs",
        str(args.cuda_score_batch_pairs),
        "--mode-batch-size",
        str(args.cuda_mode_batch_size),
        "--timeout-seconds",
        str(args.cuda_timeout_seconds),
        "--detach",
        "--provider-detach-ack",
        "--lane-id",
        args.cuda_lane_id,
        "--instance-job-id",
        run_id,
        "--claim-agent",
        args.claim_agent,
        "--claim-notes",
        (
            "Local-first HDM8 first-frame postfilter CUDA confirmation; "
            f"archive_sha256={sha256_file(archive)}; "
            f"local_modes={len(modes)}; cuda_modes={len(cuda_modes)}; "
            "score_claim=false"
        ),
    ]

    summary = {
        "schema": "hdm8_local_first_postfilter_sweep_summary_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "archive": repo_relative(archive, REPO_ROOT),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "profile": args.profile,
        "generated_mode_count": len(modes),
        "cpu_guard_mode_count": len(cpu_guard_modes),
        "parallelism": {
            "max_local_rss_gb": args.max_local_rss_gb,
            "cpu_parallel_workers": args.cpu_parallel_workers,
            "mps_prefix_parallel_workers": 0 if args.skip_mps else args.mps_prefix_parallel_workers,
            "mps_full_parallel_workers": 0 if args.skip_mps else args.mps_full_parallel_workers,
            "cpu_shard_size": args.cpu_shard_size,
            "mps_prefix_shard_size": args.mps_prefix_shard_size,
            "mps_full_shard_size": args.mps_full_shard_size,
        },
        "stages": stages,
        "full_stage_modes": len(full_modes),
        "cuda_shortlist_modes": len(cuda_modes),
        "cuda_selector_rate_lower_bound": _charged_selector_rate_estimate(
            len(cuda_modes), int(args.full_pairs)
        ),
        "modal_cuda_confirmation_command": modal_command,
        "modal_cuda_confirmation_output_dir": modal_output_dir,
        "next_steps": [
            "Run the Modal CUDA confirmation command if local signal clears the byte penalty.",
            "Build a format_id=0x03 charged selector packet only from the CUDA sweep JSON.",
            "Promote only after contest-CUDA auth eval of the charged archive/runtime packet.",
        ],
    }
    write_json(out_dir / "local_first_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profile", choices=["fast", "broad", "aggressive"], default="broad")
    parser.add_argument("--max-modes", type=int, default=0)
    parser.add_argument("--include-last-frame-risky", action="store_true")
    parser.add_argument("--cpu-guard-pairs", type=int, default=8)
    parser.add_argument("--cpu-guard-max-modes", type=int, default=48)
    parser.add_argument("--mps-prefix-pairs", type=int, default=64)
    parser.add_argument("--full-pairs", type=int, default=600)
    parser.add_argument("--full-top-k", type=int, default=32)
    parser.add_argument("--full-margin", type=float, default=0.00075)
    parser.add_argument("--cpu-carry-top-k", type=int, default=8)
    parser.add_argument("--cpu-carry-margin", type=float, default=0.0005)
    parser.add_argument("--cuda-top-k", type=int, default=12)
    parser.add_argument("--cuda-margin", type=float, default=0.00035)
    parser.add_argument("--cpu-decode-batch-pairs", type=int, default=2)
    parser.add_argument("--cpu-score-batch-pairs", type=int, default=2)
    parser.add_argument("--cpu-mode-batch-size", type=int, default=4)
    parser.add_argument("--cpu-parallel-workers", type=int, default=4)
    parser.add_argument("--cpu-shard-size", type=int, default=16)
    parser.add_argument("--mps-decode-batch-pairs", type=int, default=8)
    parser.add_argument("--mps-score-batch-pairs", type=int, default=4)
    parser.add_argument("--mps-mode-batch-size", type=int, default=4)
    parser.add_argument("--mps-prefix-parallel-workers", type=int, default=2)
    parser.add_argument("--mps-prefix-shard-size", type=int, default=32)
    parser.add_argument("--mps-full-parallel-workers", type=int, default=2)
    parser.add_argument("--mps-full-shard-size", type=int, default=16)
    parser.add_argument("--max-local-rss-gb", type=float, default=48.0)
    parser.add_argument("--cuda-decode-batch-pairs", type=int, default=8)
    parser.add_argument("--cuda-score-batch-pairs", type=int, default=4)
    parser.add_argument("--cuda-mode-batch-size", type=int, default=4)
    parser.add_argument("--cuda-timeout-seconds", type=int, default=6900)
    parser.add_argument("--cuda-lane-id", default="hdm8_local_first_postfilter_cuda_confirm_20260514")
    parser.add_argument("--claim-agent", default="codex:hdm8_local_first_postfilter")
    parser.add_argument("--skip-mps", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run_local_first(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
