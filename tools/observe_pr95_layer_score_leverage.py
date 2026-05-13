#!/usr/bin/env python
"""Ablation observability: zero each PR95 layer; measure forward-output impact.

For each weight tensor in PR95's parsed state_dict, temporarily zero it (in a
copy), forward the same latent batch, and measure the L2 difference from the
unablated output. The result is a per-layer "score-leverage proxy" ranking that
identifies which layers, if dropped, would most damage the rendered output.

Note: this is a FORWARD-OUTPUT proxy, NOT a scorer-distortion proxy. The
canonical scorer-distortion ablation requires loading SegNet+PoseNet and is
contest-CUDA-bound; this tool runs on macOS-CPU (or any CPU/CUDA) as an
ADVISORY rank-only signal. Per CLAUDE.md "MPS auth eval is NOISE" + "Apples-to-
apples evidence discipline" the output is `[macOS-CPU advisory only]` /
`[advisory only]` and must NOT be used to promote/demote/kill any lane.

Outputs:
    experiments/results/lane_pr95_artifact_lora_dora_surgery_20260513_<UTC>/
        layer_ablation_rank.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.pr95_lora_dora.pr95_base import HNeRVDecoder  # noqa: E402


def ablation_score_leverage(parsed_pt_path: Path, n_sample: int = 16,
                            seed: int = 0) -> dict:
    """Per-layer ablation rank by ||decoder(z) - decoder_ablated(z)||_2."""
    state = torch.load(parsed_pt_path, weights_only=False, map_location="cpu")
    decoder_sd = state["decoder_sd"]
    latents = state["latents"]
    meta = state["meta"]

    torch.manual_seed(seed)
    idx = torch.randperm(latents.shape[0])[:n_sample]
    z = latents[idx]

    def make_decoder(sd):
        d = HNeRVDecoder(latent_dim=meta["latent_dim"],
                         base_channels=meta["base_channels"],
                         eval_size=tuple(meta["eval_size"]))
        d.load_state_dict(sd)
        d.eval()
        return d

    base = make_decoder(decoder_sd)
    with torch.no_grad():
        out_base = base(z)

    results = []
    for name, tensor in decoder_sd.items():
        if not name.endswith(".weight"):
            continue
        sd_ablated = {k: v.clone() for k, v in decoder_sd.items()}
        sd_ablated[name] = torch.zeros_like(sd_ablated[name])
        try:
            d_a = make_decoder(sd_ablated)
            with torch.no_grad():
                out_a = d_a(z)
            l2 = (out_a - out_base).pow(2).mean().sqrt().item()
            results.append({
                "name": name,
                "shape": tuple(tensor.shape),
                "numel": tensor.numel(),
                "ablation_l2_rmse": l2,
                "leverage_density": l2 / max(tensor.numel(), 1),
            })
        except Exception as e:  # pragma: no cover — defensive
            results.append({"name": name, "error": str(e)})

    results.sort(key=lambda r: -r.get("ablation_l2_rmse", 0.0))
    return {
        "method": "weight_ablation_output_l2_rmse_proxy",
        "evidence_grade": "macOS-CPU-advisory-only",
        "score_claim": False,
        "promotion_eligible": False,
        "n_sample": n_sample,
        "seed": seed,
        "results": results,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parsed-pt", type=Path,
        default=REPO_ROOT / ".omx" / "tmp" / "pr95_artifact" / "pr95_parsed.pt",
        help="Path to PR95 parsed state pickle"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-sample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.output_dir is None:
        utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO_ROOT / "experiments" / "results" / f"lane_pr95_artifact_lora_dora_surgery_20260513_{utc}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.parsed_pt.exists():
        print(f"FATAL: parsed PR95 state missing: {args.parsed_pt}", file=sys.stderr)
        print("Hint: run the parse step (see lane README) to materialize it.", file=sys.stderr)
        return 1

    print(f"Running layer ablation on {args.parsed_pt}...")
    result = ablation_score_leverage(args.parsed_pt, n_sample=args.n_sample,
                                     seed=args.seed)
    out_path = args.output_dir / "layer_ablation_rank.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Pretty-print top-5 and bottom-5
    print(f"\n{'NAME':<32s} {'SHAPE':<20s} {'L2_RMSE':>12s} {'LEV/PARAM':>14s}")
    rows = [r for r in result["results"] if "ablation_l2_rmse" in r]
    for r in rows[:5]:
        print(f"  TOP   {r['name']:<26s} {r['shape']!s:<20s} {r['ablation_l2_rmse']:>12.4f} {r['leverage_density']:>14.6e}")
    print()
    for r in rows[-5:]:
        print(f"  BOT   {r['name']:<26s} {r['shape']!s:<20s} {r['ablation_l2_rmse']:>12.4f} {r['leverage_density']:>14.6e}")

    print(f"\nFull rank saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
