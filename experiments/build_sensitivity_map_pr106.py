#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build per-channel β-Fisher sensitivity map for PR106 HNeRV decoder.

Lane Ω-W-V3 step 2 of 3 (revival_plan_08_sensitivity_map_pr106_producer +
revival_plan_01_water_filling_codec_v2_pr106_decoder).

Inputs:
    state_dict.pt: dict[str, torch.Tensor] from extract_pr106_decoder.py
    latents.pt: torch.Tensor (600, 28) of frame-pair latents from same source
    upstream-dir: contest upstream/ checkout (for PoseNet + SegNet scorers + GT video pair iterator)

Method:
    For each contest frame-pair (i, i+1) in 0..599:
        1. Forward: f_pair = HNeRVDecoder(latents[i])
        2. Score: posenet_loss + 100 * segnet_loss against GT pair
        3. Backward: collect per-parameter grad
        4. Accumulate per-tensor grad² (β-Fisher diagonal of FIM)
    Aggregate per-tensor sensitivities into 1-D-per-output-channel via reduce-sum
    over (input-channel, kernel_h, kernel_w) dims. This matches the API of
    src/tac/sensitivity_map.py:save_sensitivity_map().

Output:
    sensitivity_map.pt — saved via tac.sensitivity_map.save_sensitivity_map()
        Format: {"format": "...", "sensitivities": dict[str, 1-D tensor (n_out_channels,)],
                 "metadata": {"score_formula": "100*seg + sqrt(10*pose)",
                              "n_pairs": ..., "score_proxy_value": ..., "device": ...}}

Output channel reduction (per-tensor):
    Conv2d weight (out_ch, in_ch, kH, kW): sum over (in_ch, kH, kW) → (out_ch,)
    Conv2d bias (out_ch,): keep as-is
    Linear weight (out_features, in_features): sum over (in_features,) → (out_features,)
    Linear bias (out_features,): keep as-is

CUDA REQUIRED for honest β-Fisher (per CLAUDE.md "MPS auth eval is NOISE" — the
score-gradient on MPS will not match contest-CUDA T4 within FIM eigenvalue
tolerance). For DESIGN-time parser plumbing, --device cpu/mps requires the
explicit --allow-stub-design-mode flag and emits a non-promotable stub.

Usage (CUDA dispatch, $0.50 RTX 4090):
    .venv/bin/python experiments/build_sensitivity_map_pr106.py \\
        --state-dict experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt \\
        --latents experiments/results/sensitivity_map_pr106_20260504_claude/latents.pt \\
        --upstream-dir upstream \\
        --out experiments/results/sensitivity_map_pr106_20260504_claude/sensitivity_map.pt \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PR106_SRC_PATH = Path(__file__).parent / "results" / (
    "public_pr106_belt_and_suspenders_intake_20260504_codex/source/"
    "submissions/belt_and_suspenders/src"
)
sys.path.insert(0, str(PR106_SRC_PATH.resolve()))

from tac.repo_io import sha256_file
from tac.sensitivity_map import (
    build_contiguous_pair_manifest,
    canonical_sensitivity_json_bytes,
    save_sensitivity_map,
    sensitivity_manifest_sha256,
)

REAL_PR106_CONTEST_PAIR_ITERATOR_IMPLEMENTED = False
REAL_PR106_CONTEST_PAIR_ITERATOR_BLOCKER = (
    "PR106 β-Fisher producer still lacks a reviewed canonical "
    "archive.zip -> inflate.sh -> upstream/evaluate.py pair iterator."
)
DEFAULT_PR106_ARCHIVE = Path(__file__).parent / "results" / (
    "public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)


def _load_pr106_decoder(state_dict_path: Path, device: torch.device) -> torch.nn.Module:
    from model import HNeRVDecoder  # type: ignore[import-not-found]

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise ValueError(f"{state_dict_path}: expected dict, got {type(sd)}")
    model = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(384, 512))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARN: missing keys when loading state_dict: {missing}", file=sys.stderr)
    if unexpected:
        print(f"WARN: unexpected keys when loading state_dict: {unexpected}", file=sys.stderr)
    model.to(device).eval()
    return model


def _allow_stub_design_mode(device: str, allow_stub_design_mode: bool) -> bool:
    return device != "cuda" and bool(allow_stub_design_mode)


def _load_extract_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected extract metadata object")
    return payload


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _source_archive_binding(
    source_archive: Path | None,
    extract_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return SHA/byte binding for the PR106 source archive and fail on drift."""
    recorded_sha = extract_metadata.get("archive_sha256")
    recorded_bytes = _optional_int(extract_metadata.get("archive_size_bytes"))
    binding: dict[str, Any] = {}

    if source_archive is not None and source_archive.is_file():
        actual_sha = sha256_file(source_archive)
        actual_bytes = source_archive.stat().st_size
        if isinstance(recorded_sha, str) and recorded_sha != actual_sha:
            raise ValueError(
                "extract metadata archive_sha256 does not match --source-archive: "
                f"metadata={recorded_sha} actual={actual_sha}"
            )
        if recorded_bytes is not None and recorded_bytes != actual_bytes:
            raise ValueError(
                "extract metadata archive_size_bytes does not match --source-archive: "
                f"metadata={recorded_bytes} actual={actual_bytes}"
            )
        binding.update(
            {
                "source_archive": str(source_archive),
                "source_archive_sha256": actual_sha,
                "source_archive_bytes": actual_bytes,
            }
        )
    elif isinstance(recorded_sha, str) and recorded_bytes is not None:
        binding.update(
            {
                "source_archive": extract_metadata.get("archive_path"),
                "source_archive_sha256": recorded_sha,
                "source_archive_bytes": recorded_bytes,
                "source_archive_binding_mode": "extract_metadata_only",
            }
        )

    return binding


def _reduce_to_per_output_channel(name: str, grad_sq: torch.Tensor) -> torch.Tensor:
    """Sum grad² over input-channel + kernel dims, leaving (n_out_channels,)."""
    if grad_sq.dim() == 4:  # Conv2d weight: (O, I, kH, kW)
        return grad_sq.sum(dim=(1, 2, 3))
    if grad_sq.dim() == 2:  # Linear weight: (O, I)
        return grad_sq.sum(dim=1)
    if grad_sq.dim() == 1:  # bias / 1-D scale
        return grad_sq
    raise ValueError(
        f"sensitivity reduction: unhandled grad shape {grad_sq.shape} for {name!r}"
    )


def _score_proxy(
    pred_pair: torch.Tensor, gt_pair: torch.Tensor, *, posenet, segnet
) -> torch.Tensor:
    """Score proxy: 100 * segnet_dist + sqrt(10 * posenet_dist).
    pred_pair / gt_pair: (1, 2, 3, H, W) uint8-equivalent in [0, 255].
    """
    pose_dist = posenet(gt_pair, pred_pair).mean()
    seg_dist = segnet(gt_pair, pred_pair).mean()
    return 100.0 * seg_dist + torch.sqrt(torch.clamp(10.0 * pose_dist, min=1e-12))


def _iter_contest_pairs(upstream_dir: Path, n_pairs: int = 600):
    """Yield (idx, gt_pair_tensor (1, 2, 3, H, W) float in [0, 255])."""
    sys.path.insert(0, str(upstream_dir.resolve()))
    # contest video iter is at upstream/evaluate.py — placeholder import
    # actual loader will be filled in by the GPU dispatch wrapper which
    # already wires upstream's video pair iterator correctly.
    raise NotImplementedError(
        "_iter_contest_pairs: GPU dispatch wrapper provides the contest pair "
        "iterator from upstream/evaluate.py. This script delegates to the "
        "wrapper; stub here is intentional. See scripts/remote_lane_omega_w_v3_pr106.sh"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source-archive",
        type=Path,
        help=(
            "PR106 source archive.zip. If omitted, the producer uses "
            "<state-dict-dir>/metadata.json when available, then the canonical "
            "local PR106 intake archive if present."
        ),
    )
    parser.add_argument(
        "--extract-metadata",
        type=Path,
        help=(
            "Stage-1 extract metadata.json for source archive custody. Defaults "
            "to <state-dict-dir>/metadata.json when present."
        ),
    )
    parser.add_argument(
        "--pair-manifest-out",
        type=Path,
        help=(
            "Write the deterministic pair/sample plan sidecar. Defaults to "
            "<out-stem>.pair_manifest.json."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' (REQUIRED for honest β-Fisher per CLAUDE.md). "
                             "'cpu' / 'mps' allowed for design-time sanity check ONLY; "
                             "output is tagged [advisory only].")
    parser.add_argument("--n-pairs", type=int, default=600,
                        help="Number of frame-pairs to accumulate FIM over.")
    parser.add_argument(
        "--allow-stub-design-mode",
        action="store_true",
        help=(
            "Allow non-CUDA design-mode all-ones sensitivity for local parser "
            "plumbing only. CUDA mode never emits a stub; missing scorer-pair "
            "wiring fails closed."
        ),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda but torch.cuda.is_available() == False. "
              "FAIL LOUD per CLAUDE.md silent-CPU-fallback rule.", file=sys.stderr)
        return 2
    if args.device != "cuda":
        if not args.allow_stub_design_mode:
            print(
                f"ERROR: --device {args.device} requires --allow-stub-design-mode. "
                "Non-CUDA sensitivity is local plumbing only and must be "
                "explicitly non-promotable.",
                file=sys.stderr,
            )
            return 2
        print(f"WARN: --device {args.device} — output sensitivity tagged [advisory only]. "
              f"For honest β-Fisher use --device cuda on T4-equivalent.", file=sys.stderr)

    device = torch.device(args.device)
    model = _load_pr106_decoder(args.state_dict, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sensitivity-pr106] loaded HNeRVDecoder: {n_params} params on {device}")

    latents = torch.load(args.latents, map_location=device, weights_only=False)
    if latents.dim() != 2 or latents.shape[1] != 28:
        raise ValueError(f"latents: expected (N, 28), got {latents.shape}")
    n_pairs_eff = min(args.n_pairs, latents.shape[0])
    print(f"[sensitivity-pr106] latents: {tuple(latents.shape)}; using {n_pairs_eff} pairs")

    extract_metadata_path = args.extract_metadata
    if extract_metadata_path is None:
        candidate = args.state_dict.parent / "metadata.json"
        extract_metadata_path = candidate if candidate.is_file() else None
    extract_metadata = _load_extract_metadata(extract_metadata_path)

    source_archive = args.source_archive
    if source_archive is None:
        recorded_archive = extract_metadata.get("archive_path")
        if isinstance(recorded_archive, str) and Path(recorded_archive).is_file():
            source_archive = Path(recorded_archive)
        elif DEFAULT_PR106_ARCHIVE.is_file():
            source_archive = DEFAULT_PR106_ARCHIVE

    state_dict_sha = sha256_file(args.state_dict)
    latents_sha = sha256_file(args.latents)
    source_bindings = {
        **_source_archive_binding(source_archive, extract_metadata),
        "state_dict": str(args.state_dict),
        "state_dict_sha256": state_dict_sha,
        "state_dict_source_sha256": state_dict_sha,
        "model_sha256": state_dict_sha,
        "latents": str(args.latents),
        "latents_sha256": latents_sha,
        "extract_metadata": str(extract_metadata_path) if extract_metadata_path else None,
    }
    if args.device == "cuda" and "source_archive_sha256" not in source_bindings:
        print(
            "ERROR: CUDA sensitivity production requires source archive SHA/bytes "
            "binding via --source-archive or Stage-1 metadata.json.",
            file=sys.stderr,
        )
        return 2

    pair_manifest = build_contiguous_pair_manifest(
        n_pairs_eff,
        latent_rows=int(latents.shape[0]),
        source_bindings=source_bindings,
    )
    pair_manifest_sha = sensitivity_manifest_sha256(pair_manifest)
    pair_manifest_out = args.pair_manifest_out or args.out.with_name(
        f"{args.out.stem}.pair_manifest.json"
    )
    pair_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    pair_manifest_out.write_bytes(canonical_sensitivity_json_bytes(pair_manifest))
    print(
        f"[sensitivity-pr106] pair manifest: {pair_manifest_out} "
        f"(sha256={pair_manifest_sha[:16]}..., n_pairs={n_pairs_eff})"
    )

    # Load contest scorers (PoseNet + SegNet) from upstream — GPU-dispatch wrapper
    # script wires this up and calls into this module's compute path.
    # For DESIGN-TIME SCAFFOLDING we stop here and emit a stub sensitivity map
    # of all-ones so downstream consumers can be wired/tested.
    print(f"[sensitivity-pr106] computing per-tensor grad² accumulation over {n_pairs_eff} pairs...")
    grad_sq_acc: dict[str, torch.Tensor] = {
        n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()
    }

    stub_design_mode = False
    try:
        # GPU dispatch path: real β-Fisher accumulation
        posenet = None  # type: ignore
        segnet = None  # type: ignore
        # Actual scorer-load + pair iteration happens inside the dispatch wrapper.
        for i, gt_pair in _iter_contest_pairs(args.upstream_dir, n_pairs=n_pairs_eff):
            model.zero_grad()
            pred_pair = model(latents[i:i+1])
            score = _score_proxy(pred_pair, gt_pair, posenet=posenet, segnet=segnet)
            score.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    grad_sq_acc[n] += p.grad.detach() ** 2
    except NotImplementedError as e:
        if not _allow_stub_design_mode(args.device, args.allow_stub_design_mode):
            print(
                f"[sensitivity-pr106] FATAL: real contest-pair iterator is not wired: {e}",
                file=sys.stderr,
            )
            print(
                "[sensitivity-pr106] Refusing to emit a dummy/stub sensitivity map "
                "for a CUDA/promotable run.",
                file=sys.stderr,
            )
            return 3
        stub_design_mode = True
        print(f"[sensitivity-pr106] DESIGN-MODE: {e}", file=sys.stderr)
        print("[sensitivity-pr106] DESIGN-MODE: emitting stub all-ones sensitivity (no GPU available)", file=sys.stderr)
        # Stub sensitivity = all-ones-per-channel so downstream wiring is testable
        # without GPU. Tagged [stub] in metadata so water-fill knows to refuse use.
        for n, p in model.named_parameters():
            grad_sq_acc[n] = torch.ones_like(p, device=device)

    # Reduce to per-output-channel 1-D vectors
    sensitivities: dict[str, torch.Tensor] = {}
    for name, gs in grad_sq_acc.items():
        sensitivities[name] = _reduce_to_per_output_channel(name, gs).cpu()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "score_formula": "100*seg + sqrt(10*pose) per CLAUDE.md",
        "n_pairs": n_pairs_eff,
        "device": str(device),
        "is_stub": stub_design_mode,
        "tag": "[stub-design-mode]" if stub_design_mode else "[contest-CUDA]",
        "n_params_total": n_params,
        "state_dict_source": str(args.state_dict),
        "latents_source": str(args.latents),
        "source_binding_status": (
            "sha_bound" if "source_archive_sha256" in source_bindings else "missing_source_archive"
        ),
        "pair_manifest_format": pair_manifest["format"],
        "pair_manifest": str(pair_manifest_out),
        "pair_manifest_sha256": pair_manifest_sha,
        "sample_plan_sha256": pair_manifest_sha,
        **source_bindings,
    }
    if stub_design_mode:
        metadata["stub_reason"] = "contest_pair_iterator_not_wired"
    save_sensitivity_map(args.out, sensitivities, metadata=metadata)
    print(f"[sensitivity-pr106] saved {args.out}")
    print(f"[sensitivity-pr106] tag: {metadata['tag']}")

    summary = {n: float(v.sum().item()) for n, v in sensitivities.items()}
    print("[sensitivity-pr106] per-tensor sensitivity-sum (top 5 by magnitude):")
    for n, s in sorted(summary.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {n}: total={s:.4g}, n_channels={sensitivities[n].numel()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
