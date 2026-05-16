#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the ATW v2 D4 side-information probe on A1 latents.

This is the pre-dispatch empirical gate required by
``atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md`` and the
T3 batched Phase 2 council. It is deliberately diagnostic only:
``score_claim=false`` and ``promotion_eligible=false``. It answers whether the
SegNet-class side-information channel has enough mutual information with A1
latents to justify an ATW v2 Phase 2 lift.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import struct
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
A1_ARCHIVE = REPO_ROOT / "submissions" / "a1" / "archive.zip"
A1_SRC_DIR = REPO_ROOT / "submissions" / "a1" / "src"
DEFAULT_CLASS_JSON = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "tishby_ib_pure_d4_probe_20260516T212557Z"
    / "per_pair_segnet_class.json"
)
DEFAULT_RESEARCH_JSON = (
    REPO_ROOT / ".omx" / "research" / "atw_codec_v2_d4_probe_verdict_20260516_codex.json"
)
DEFAULT_RESEARCH_MD = (
    REPO_ROOT / ".omx" / "research" / "atw_codec_v2_d4_probe_verdict_20260516_codex.md"
)
DEFAULT_STATE_JSON = (
    REPO_ROOT / ".omx" / "state" / "h_latent_given_scorer_class_atw_codec_v2.json"
)
LATENT_DIM = 28
N_PAIRS = 600

for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "upstream", A1_SRC_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from tools.probe_latent_conditional_entropy_h_latent_given_scorer_class import (  # noqa: E402
    compute_h_latent_given_scorer_class,
)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _replay_command() -> str:
    return " ".join([".venv/bin/python", _repo_rel(Path(__file__)), *sys.argv[1:]])


def _default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "experiments" / "results" / f"atw_codec_v2_d4_probe_{stamp}"


def _load_a1_latents() -> tuple[torch.Tensor, dict[str, Any]]:
    if not A1_ARCHIVE.is_file():
        raise FileNotFoundError(f"A1 archive missing: {A1_ARCHIVE}")
    archive_zip_bytes = A1_ARCHIVE.read_bytes()
    with zipfile.ZipFile(A1_ARCHIVE, "r") as zf:
        archive_bytes = zf.read("x")
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    decoder_blob = archive_bytes[4:section_total]
    latent_blob_len = 15_387
    latent_blob = archive_bytes[section_total : section_total + latent_blob_len]
    sidecar_blob = archive_bytes[section_total + latent_blob_len :]

    from codec import apply_latent_sidecar, decode_latents_compact

    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    if tuple(latents.shape) != (N_PAIRS, LATENT_DIM):
        raise ValueError(
            f"expected A1 latents shape {(N_PAIRS, LATENT_DIM)}, got {tuple(latents.shape)}"
        )
    info = {
        "a1_archive_path": _repo_rel(A1_ARCHIVE),
        "a1_archive_zip_sha256": _sha256_bytes(archive_zip_bytes),
        "a1_inner_member": "x",
        "a1_inner_member_sha256": _sha256_bytes(archive_bytes),
        "decoder_blob_sha256": _sha256_bytes(decoder_blob),
        "latent_blob_sha256": _sha256_bytes(latent_blob),
        "sidecar_blob_sha256": _sha256_bytes(sidecar_blob),
        "latent_shape": [int(v) for v in latents.shape],
    }
    return latents, info


def _quantize_u8(tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    values = tensor.detach().to("cpu", dtype=torch.float32)
    lo = float(values.min().item())
    hi = float(values.max().item())
    value_range = max(hi - lo, 1e-9)
    quantized = ((values - lo) / value_range * 255.0).round().clamp(0, 255)
    return quantized.to(torch.uint8), {
        "min": lo,
        "max": hi,
        "range": value_range,
    }


def _load_classes(path: Path) -> tuple[list[int], dict[str, Any]]:
    data = json.loads(path.read_text())
    classes = data.get("per_pair_class")
    if not isinstance(classes, list):
        raise ValueError(f"{path} missing per_pair_class list")
    classes_i = [int(item) for item in classes]
    if len(classes_i) != N_PAIRS:
        raise ValueError(f"expected {N_PAIRS} per-pair classes, got {len(classes_i)}")
    if any(item < 0 or item > 255 for item in classes_i):
        raise ValueError("per-pair classes must fit one byte")
    latent_dim = int(data.get("latent_dim", LATENT_DIM))
    if latent_dim != LATENT_DIM:
        raise ValueError(f"expected latent_dim={LATENT_DIM}, got {latent_dim}")
    distinct = sorted(set(classes_i))
    counts = {str(label): classes_i.count(label) for label in distinct}
    info = {
        "class_json_path": _repo_rel(path),
        "class_json_sha256": _sha256_file(path),
        "num_pairs": len(classes_i),
        "latent_dim": latent_dim,
        "distinct_classes": distinct,
        "per_class_counts": counts,
        "scorer_source": str(data.get("scorer_source") or ""),
    }
    return classes_i, info


def _tile_classes(classes_per_pair: list[int]) -> bytes:
    return bytes(label for label in classes_per_pair for _ in range(LATENT_DIM))


def _class_centroid_residual_stream(
    latents_u8: torch.Tensor, classes_per_pair: list[int]
) -> tuple[bytes, bytes, dict[str, Any]]:
    values = latents_u8.to(torch.float32)
    class_tensor = torch.tensor(classes_per_pair, dtype=torch.int64)
    global_center = values.mean(dim=0, keepdim=True)
    global_residual = values - global_center

    class_pred = torch.empty_like(values)
    class_means: dict[str, list[float]] = {}
    for label in sorted(set(classes_per_pair)):
        mask = class_tensor == int(label)
        mean = values[mask].mean(dim=0, keepdim=True)
        class_pred[mask] = mean
        class_means[str(label)] = [float(x) for x in mean.squeeze(0).tolist()]
    class_residual = values - class_pred

    global_u8, global_range = _quantize_u8(global_residual)
    class_u8, class_range = _quantize_u8(class_residual)
    info = {
        "residual_model": "per_latent_dim_class_centroid_minus_a1_latents_u8",
        "global_residual_quantizer": global_range,
        "class_centroid_residual_quantizer": class_range,
        "class_mean_count": len(class_means),
        "class_means_by_label": class_means,
    }
    return bytes(global_u8.flatten().tolist()), bytes(class_u8.flatten().tolist()), info


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _render_markdown(payload: dict[str, Any]) -> str:
    verdict = payload["canonical_wz_sideinfo_verdict"]
    residual = payload["wz_residual_surface_proxy"]
    provenance = payload["provenance"]
    status = payload["atw_v2_phase2_status"]
    return "\n".join(
        [
            "# ATW Codec V2 D4 Probe Verdict",
            "",
            f"- observed_at_utc: `{payload['observed_at_utc']}`",
            "- score_claim: `false`",
            "- promotion_eligible: `false`",
            "- ready_for_exact_eval_dispatch: `false`",
            "- axis_label: `[diagnostic-CPU; H(latent|scorer_class) probe]`",
            "- dispatch_attempted: `false`",
            "",
            "## Verdict",
            "",
            f"- substrate_id: `{verdict['substrate_id']}`",
            f"- verdict: `{verdict['verdict']}`",
            f"- I(latent; scorer_class): `{verdict['mutual_information_bits']:.12f}` bits/symbol",
            f"- H(latent): `{verdict['h_latent_unconditional_bits_per_symbol']:.12f}` bits/symbol",
            (
                "- H(latent | scorer_class): "
                f"`{verdict['h_latent_given_scorer_class_bits_per_symbol']:.12f}` bits/symbol"
            ),
            (
                "- Wyner-Ziv gain ceiling fraction: "
                f"`{verdict['wyner_ziv_gain_ceiling_fraction']:.12f}`"
            ),
            f"- unique class signatures: `{verdict['num_unique_classes']}`",
            "",
            "## ATW V2 Phase 2 Consequence",
            "",
            f"- phase2_status: `{status['phase2_status']}`",
            f"- recommended_variant: `{status['recommended_variant']}`",
            f"- next_action: `{status['next_action']}`",
            "",
            "The measured mutual information is below the canonical independence",
            "tolerance, so ATW v2's class-conditional WZ surface should not receive",
            "Phase 2 lift authority from this class signal. This is a deferral of",
            "the measured A1-latent/class-conditioning configuration, not a kill of",
            "the broader cooperative-receiver paradigm.",
            "",
            "## WZ Residual-Surface Proxy",
            "",
            (
                "- global-centroid residual entropy: "
                f"`{residual['global_residual_entropy_bits_per_symbol']:.12f}` bits/symbol"
            ),
            (
                "- class-centroid residual entropy: "
                f"`{residual['class_centroid_residual_entropy_bits_per_symbol']:.12f}` bits/symbol"
            ),
            (
                "- residual entropy delta: "
                f"`{residual['class_centroid_residual_delta_bits_per_symbol']:.12f}` bits/symbol"
            ),
            (
                "- residual gain fraction: "
                f"`{residual['class_centroid_residual_gain_fraction']:.12f}`"
            ),
            "",
            "This proxy subtracts per-class, per-latent-dimension centroids before",
            "re-estimating byte entropy. It is diagnostic only; the canonical WZ",
            "decision remains the H(latent|class) verdict above.",
            "",
            "## Provenance",
            "",
            f"- command: `{payload['command']}`",
            f"- research_json: `{payload['research_json_path']}`",
            f"- output_dir: `{payload['output_dir']}`",
            f"- A1 archive: `{provenance['a1_archive_path']}`",
            f"- A1 archive sha256: `{provenance['a1_archive_zip_sha256']}`",
            f"- A1 inner member sha256: `{provenance['a1_inner_member_sha256']}`",
            f"- class artifact: `{provenance['class_json_path']}`",
            f"- class artifact sha256: `{provenance['class_json_sha256']}`",
            f"- latent stream sha256: `{provenance['latent_u8_sha256']}`",
            f"- tiled class stream sha256: `{provenance['class_stream_sha256']}`",
            f"- global residual stream sha256: `{provenance['global_residual_u8_sha256']}`",
            f"- class residual stream sha256: `{provenance['class_residual_u8_sha256']}`",
            "",
            "## Reactivation Criteria",
            "",
            "1. Replace the saturated per-pair SegNet composite class with a richer",
            "   side-information signal, such as per-region class histograms, logits,",
            "   pose bins, or hard-pair/object-state features.",
            "2. Rerun the same probe on trained ATW v2 residuals rather than A1",
            "   HNeRV latents if a non-promotional timing smoke produces them.",
            "3. Require paired CPU/CUDA exact-eval custody before any score, rank, or",
            "   promotion claim.",
        ]
    ) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ATW v2 D4 H(latent|scorer_class) probe from A1 latents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--class-json", type=Path, default=DEFAULT_CLASS_JSON)
    parser.add_argument("--research-json", type=Path, default=DEFAULT_RESEARCH_JSON)
    parser.add_argument("--research-md", type=Path, default=DEFAULT_RESEARCH_MD)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--skip-state", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    observed_at = _utc_now()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latents, a1_info = _load_a1_latents()
    latents_u8, latent_quantizer = _quantize_u8(latents)
    latent_bytes = bytes(latents_u8.flatten().tolist())
    classes_per_pair, class_info = _load_classes(args.class_json.resolve())
    class_bytes = _tile_classes(classes_per_pair)
    if len(latent_bytes) != len(class_bytes):
        raise ValueError(
            f"latent stream len {len(latent_bytes)} != class stream len {len(class_bytes)}"
        )

    global_residual_bytes, class_residual_bytes, residual_info = (
        _class_centroid_residual_stream(latents_u8, classes_per_pair)
    )
    zeros = bytes(len(latent_bytes))
    canonical_verdict = compute_h_latent_given_scorer_class(
        substrate_id="atw_codec_v2",
        latent_stream=latent_bytes,
        class_stream=class_bytes,
        notes=(
            "ATW v2 D4 pre-dispatch gate on A1 latents. A1 latents are "
            "requantized to uint8 exactly as the sister Tishby IB-pure D4 "
            "probe; class stream reuses the recorded SegNet per-pair composite "
            "class artifact to avoid rerendering 600 pairs. Diagnostic CPU; "
            "score_claim=false; not promotion authority."
        ),
    )
    global_residual_verdict = compute_h_latent_given_scorer_class(
        substrate_id="atw_codec_v2_global_centroid_residual_proxy",
        latent_stream=global_residual_bytes,
        class_stream=zeros,
        notes="Unconditional entropy of global-centroid A1 latent residuals.",
    )
    class_residual_verdict = compute_h_latent_given_scorer_class(
        substrate_id="atw_codec_v2_class_centroid_residual_proxy",
        latent_stream=class_residual_bytes,
        class_stream=zeros,
        notes="Unconditional entropy of per-class-centroid A1 latent residuals.",
    )
    residual_delta = (
        global_residual_verdict.h_latent_unconditional_bits_per_symbol
        - class_residual_verdict.h_latent_unconditional_bits_per_symbol
    )
    residual_gain_fraction = (
        residual_delta / global_residual_verdict.h_latent_unconditional_bits_per_symbol
        if global_residual_verdict.h_latent_unconditional_bits_per_symbol > 0.0
        else 0.0
    )

    latent_path = output_dir / "a1_latents_u8.bin"
    class_path = output_dir / "segnet_class_tiled.bin"
    global_residual_path = output_dir / "a1_global_centroid_residual_u8.bin"
    class_residual_path = output_dir / "a1_class_centroid_residual_u8.bin"
    latent_path.write_bytes(latent_bytes)
    class_path.write_bytes(class_bytes)
    global_residual_path.write_bytes(global_residual_bytes)
    class_residual_path.write_bytes(class_residual_bytes)

    if canonical_verdict.verdict == "MEANINGFUL_CONDITIONING":
        phase2_status = "probe_passed_ready_for_variant_b_smoke_claim_lifecycle"
        recommended_variant = "B_WZ_ONLY"
        next_action = "claim_and_dispatch_atw_v2_variant_b_paired_smoke"
    elif canonical_verdict.verdict == "WEAK_CONDITIONING":
        phase2_status = "defer_or_run_richer_sideinfo_probe_before_paid_smoke"
        recommended_variant = "none_until_richer_sideinfo"
        next_action = "build_richer_sideinfo_probe"
    else:
        phase2_status = "defer_measured_a1_latent_class_conditioning_surface"
        recommended_variant = "none"
        next_action = "do_not_dispatch_atw_v2_phase2_from_this_signal"

    provenance: dict[str, Any] = {
        **a1_info,
        **class_info,
        "latent_quantizer": latent_quantizer,
        "latent_u8_path": _repo_rel(latent_path),
        "latent_u8_sha256": _sha256_bytes(latent_bytes),
        "class_stream_path": _repo_rel(class_path),
        "class_stream_sha256": _sha256_bytes(class_bytes),
        "global_residual_u8_path": _repo_rel(global_residual_path),
        "global_residual_u8_sha256": _sha256_bytes(global_residual_bytes),
        "class_residual_u8_path": _repo_rel(class_residual_path),
        "class_residual_u8_sha256": _sha256_bytes(class_residual_bytes),
        "residual_surface": residual_info,
    }
    payload: dict[str, Any] = {
        "schema": "atw_codec_v2_d4_probe_verdict_v1",
        "observed_at_utc": observed_at,
        "command": _replay_command(),
        "output_dir": _repo_rel(output_dir),
        "research_json_path": _repo_rel(args.research_json.resolve()),
        "research_md_path": _repo_rel(args.research_md.resolve()),
        "state_json_path": "" if args.skip_state else _repo_rel(args.state_json.resolve()),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "evidence_grade": "diagnostic_cpu",
        "axis_label": "[diagnostic-CPU; H(latent|scorer_class) probe]",
        "canonical_wz_sideinfo_verdict": asdict(canonical_verdict),
        "wz_residual_surface_proxy": {
            "global_residual_entropy_bits_per_symbol": (
                global_residual_verdict.h_latent_unconditional_bits_per_symbol
            ),
            "class_centroid_residual_entropy_bits_per_symbol": (
                class_residual_verdict.h_latent_unconditional_bits_per_symbol
            ),
            "class_centroid_residual_delta_bits_per_symbol": residual_delta,
            "class_centroid_residual_gain_fraction": residual_gain_fraction,
            "score_claim": False,
            "promotion_eligible": False,
            "axis_label": "[diagnostic-CPU; WZ residual entropy proxy]",
        },
        "atw_v2_phase2_status": {
            "phase2_status": phase2_status,
            "recommended_variant": recommended_variant,
            "next_action": next_action,
            "reason": "D4 probe verdict gates ATW v2 Phase 2 lift per 2026-05-16 council.",
        },
        "provenance": provenance,
    }
    _write_json(output_dir / "atw_codec_v2_d4_probe_verdict.json", payload)
    _write_json(args.research_json.resolve(), payload)
    args.research_md.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.research_md.resolve().write_text(_render_markdown(payload))
    if not args.skip_state:
        _write_json(args.state_json.resolve(), asdict(canonical_verdict))

    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
