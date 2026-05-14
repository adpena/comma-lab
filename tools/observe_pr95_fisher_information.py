#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Fisher-proxy observability: per-parameter leverage density for PR95.

Computes diag(F)_i = E_z[ ||d/d theta_i  L(decoder(z))||^2 ] using a uniform
proxy loss L(out) = (out**2).mean(). This is the OUTPUT-MAGNITUDE-Fisher diagonal,
NOT the contest-scorer Fisher. The result is a rank-only signal identifying
which parameters of the PR95 base have the highest leverage on the decoder
output magnitude.

Use cases:
  - LoRA/DoRA target selection (high Fisher/param = good adapter target)
  - Pruning candidates (low Fisher/param = candidates for zero/freezing)
  - Sensitivity-map seeding (hook 1 of CLAUDE.md "Subagent coherence-by-default")

Per CLAUDE.md "MPS auth eval is NOISE" + "Apples-to-apples evidence discipline"
the output is `[macOS-CPU advisory only]` / `[advisory only]` and must NOT be
used to promote/demote/kill any lane. The canonical Fisher computation requires
the contest scorer Hessian on `[contest-CUDA]` and is out of scope here.
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


def fisher_proxy(parsed_pt_path: Path, n_sample: int = 8,
                 seed: int = 0) -> dict:
    """Per-parameter Fisher diagonal under output-magnitude proxy distribution."""
    state = torch.load(parsed_pt_path, weights_only=False, map_location="cpu")
    decoder_sd = state["decoder_sd"]
    latents = state["latents"]
    meta = state["meta"]

    torch.manual_seed(seed)
    idx = torch.randperm(latents.shape[0])[:n_sample]
    z = latents[idx]

    dec = HNeRVDecoder(latent_dim=meta["latent_dim"],
                       base_channels=meta["base_channels"],
                       eval_size=tuple(meta["eval_size"]))
    dec.load_state_dict(decoder_sd)
    dec.train()
    for p in dec.parameters():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()

    out = dec(z)
    loss = (out ** 2).mean()
    loss.backward()

    results = []
    total = 0.0
    for name, p in dec.named_parameters():
        g2 = (p.grad ** 2).sum().item() if p.grad is not None else 0.0
        fpp = g2 / max(p.numel(), 1)
        results.append({
            "name": name,
            "shape": tuple(p.shape),
            "numel": p.numel(),
            "fisher_proxy": g2,
            "fisher_per_param": fpp,
        })
        total += g2

    results.sort(key=lambda r: -r["fisher_per_param"])
    return {
        "method": "output_squared_grad_diagonal_fisher_proxy",
        "evidence_grade": "macOS-CPU-advisory-only",
        "score_claim": False,
        "promotion_eligible": False,
        "n_sample": n_sample,
        "seed": seed,
        "total_fisher_proxy": total,
        "results": results,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parsed-pt", type=Path,
        default=REPO_ROOT / ".omx" / "tmp" / "pr95_artifact" / "pr95_parsed.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.output_dir is None:
        utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO_ROOT / "experiments" / "results" / f"lane_pr95_artifact_lora_dora_surgery_20260513_{utc}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.parsed_pt.exists():
        print(f"FATAL: parsed PR95 state missing: {args.parsed_pt}", file=sys.stderr)
        return 1

    print(f"Running Fisher-proxy on {args.parsed_pt}...")
    result = fisher_proxy(args.parsed_pt, n_sample=args.n_sample, seed=args.seed)
    out_path = args.output_dir / "fisher_information_proxy.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'NAME':<32s} {'SHAPE':<20s} {'NUMEL':>8s} {'FISHER':>14s} {'PER_PARAM':>14s}")
    for r in result["results"][:5]:
        print(f"  TOP   {r['name']:<26s} {r['shape']!s:<20s} {r['numel']:>8d} {r['fisher_proxy']:>14.4e} {r['fisher_per_param']:>14.4e}")
    print()
    for r in result["results"][-5:]:
        print(f"  BOT   {r['name']:<26s} {r['shape']!s:<20s} {r['numel']:>8d} {r['fisher_proxy']:>14.4e} {r['fisher_per_param']:>14.4e}")
    print(f"\nTotal Fisher proxy: {result['total_fisher_proxy']:.4e}")
    print(f"Full rank saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
