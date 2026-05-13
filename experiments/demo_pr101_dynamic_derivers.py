"""Demo: run the 5 dynamic-learning derivers on PR106's HNeRV decoder
weights (and latents), compare against PR101's hardcoded constants, and
report per-deriver byte savings.

Usage:

    .venv/bin/python experiments/demo_pr101_dynamic_derivers.py \
        --state-dict experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt \
        --latents    experiments/results/sensitivity_map_pr106_20260504_claude/latents.pt

The CLI prints a per-deriver breakdown (storage-order alone, stream-ends
alone, conv4-perms alone, latent-dim-order alone, sidecar-codebook alone,
plus the 3-deriver structural composition) along with the PR101-default
baseline. All numbers are tagged ``[empirical]`` since they come from
in-process brotli measurement on the actual substrate.

Strict-scorer-rule: this demo loads NO scorer weights and is CPU-only.
Per CLAUDE.md "no /tmp paths in any persisted artifact" — output JSON is
written to ``experiments/results/<lane>_<UTC-stamp>/`` if ``--output-dir``
is omitted.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from tac.pr101_split_brotli_codec import (
    DECODER_STORAGE_ORDER,
    encode_decoder_compact,
)
from tac.pr101_split_brotli_codec_derivers import (
    SIDECAR_DELTAS_X100_PR101_DEFAULT,
    derive_conv4_perms,
    derive_latent_dim_order,
    derive_sidecar_codebook,
    derive_storage_order,
    derive_stream_ends,
)

logger = logging.getLogger("demo_pr101_dynamic_derivers")
PR101_LATENT_DIM_ORDER = (
    26, 0, 17, 15, 10, 24, 20, 12, 14, 21, 22, 18, 4, 11,
    3, 7, 16, 2, 6, 8, 19, 23, 5, 9, 1, 13, 27, 25,
)


def _measure(state_dict: dict[str, torch.Tensor], **kwargs) -> int:
    """Return ``len(encode_decoder_compact(state_dict, **kwargs))``."""
    return len(encode_decoder_compact(state_dict, **kwargs))


def _run_demo(
    sd: dict[str, torch.Tensor],
    latents: torch.Tensor | None,
    sidecar_deltas: np.ndarray | None,
    *,
    brotli_quality: int = 11,
) -> dict[str, object]:
    """Run all 5 derivers and a 6-cell measurement matrix.

    Cells (each is a separately-encoded blob; we report ``len(blob)``):

    1. PR101 baseline (all hardcoded constants).
    2. Derived storage_order alone.
    3. Derived stream_ends alone (paired with PR101 storage_order).
    4. Derived conv4_perms alone.
    5. All 3 structural derivers together.
    """
    rows: list[dict[str, object]] = []

    # 1) Baseline.
    baseline = _measure(sd, brotli_quality=brotli_quality)
    rows.append({"label": "PR101 hardcoded baseline", "bytes": baseline, "delta_vs_baseline": 0})

    # 2) Storage order only.
    so = derive_storage_order(sd)
    so_bytes = _measure(sd, brotli_quality=brotli_quality, derived_storage_order=so)
    rows.append({
        "label": "derived storage_order only",
        "bytes": so_bytes,
        "delta_vs_baseline": so_bytes - baseline,
        "derived_value": so,
    })

    # 3) Stream ends only (paired with PR101 storage_order).
    se = derive_stream_ends(sd, DECODER_STORAGE_ORDER, brotli_quality=brotli_quality)
    se_bytes = _measure(sd, brotli_quality=brotli_quality, derived_stream_ends=se)
    rows.append({
        "label": "derived stream_ends only",
        "bytes": se_bytes,
        "delta_vs_baseline": se_bytes - baseline,
        "derived_value": se,
    })

    # 4) Conv4 perms only.
    cp = derive_conv4_perms(sd, brotli_quality=brotli_quality)
    cp_bytes = _measure(sd, brotli_quality=brotli_quality, derived_conv4_perms=cp)
    rows.append({
        "label": "derived conv4_perms only",
        "bytes": cp_bytes,
        "delta_vs_baseline": cp_bytes - baseline,
        "derived_value": cp,
    })

    # 5) All 3 structural derivers together.
    se_joint = derive_stream_ends(sd, so, brotli_quality=brotli_quality)
    full_bytes = _measure(
        sd,
        brotli_quality=brotli_quality,
        derived_storage_order=so,
        derived_stream_ends=se_joint,
        derived_conv4_perms=cp,
    )
    rows.append({
        "label": "ALL 3 structural derivers",
        "bytes": full_bytes,
        "delta_vs_baseline": full_bytes - baseline,
    })

    # 6) Latent dim order is reported as a derived-value diagnostic only —
    # it doesn't affect the decoder-blob bytes (it lives in the latent blob).
    if latents is not None:
        ldo = derive_latent_dim_order(latents)
        rows.append({
            "label": "derived latent_dim_order (diagnostic — no decoder-blob impact)",
            "bytes": None,
            "delta_vs_baseline": None,
            "derived_value": ldo,
            "pr101_value": PR101_LATENT_DIM_ORDER,
            "matches_pr101": list(ldo) == list(PR101_LATENT_DIM_ORDER),
        })

    # 7) Sidecar codebook (diagnostic — affects sidecar bytes, not decoder).
    if sidecar_deltas is not None:
        cb = derive_sidecar_codebook(sidecar_deltas)
        rows.append({
            "label": "derived sidecar_codebook (diagnostic — no decoder-blob impact)",
            "bytes": None,
            "delta_vs_baseline": None,
            "derived_value": cb.tolist(),
            "pr101_value": SIDECAR_DELTAS_X100_PR101_DEFAULT.tolist(),
            "matches_pr101": (cb == SIDECAR_DELTAS_X100_PR101_DEFAULT).all().item(),
        })

    return {"rows": rows, "baseline_bytes": baseline, "brotli_quality": brotli_quality}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Demo the 5 PR101 dynamic-learning derivers on a substrate."
    )
    parser.add_argument(
        "--state-dict",
        type=Path,
        default=Path("experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt"),
        help="Path to the HNeRV decoder state_dict.pt.",
    )
    parser.add_argument(
        "--latents",
        type=Path,
        default=Path("experiments/results/sensitivity_map_pr106_20260504_claude/latents.pt"),
        help="Path to the latents.pt (shape (600, 28)). Optional.",
    )
    parser.add_argument(
        "--sidecar-deltas",
        type=Path,
        default=None,
        help="Optional .npy file with sidecar-delta samples for Lloyd-Max codebook.",
    )
    parser.add_argument(
        "--brotli-quality",
        type=int,
        default=11,
        help="Brotli compression level (PR101 ships at 11).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write demo_results.json. Default: experiments/results/<lane>_<UTC-stamp>/",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.state_dict.is_file():
        logger.error(f"--state-dict not found: {args.state_dict}")
        return 2
    sd = torch.load(args.state_dict, map_location="cpu", weights_only=True)

    latents = None
    if args.latents is not None and args.latents.is_file():
        latents = torch.load(args.latents, map_location="cpu", weights_only=True)
        if latents.dim() != 2 or latents.shape[1] != 28:
            logger.warning(
                f"--latents shape {tuple(latents.shape)} unexpected; skipping latent-dim deriver."
            )
            latents = None

    sidecar_deltas = None
    if args.sidecar_deltas is not None and args.sidecar_deltas.is_file():
        sidecar_deltas = np.load(args.sidecar_deltas)

    logger.info(f"Running derivers on substrate: {args.state_dict}")
    logger.info(f"  brotli_quality = {args.brotli_quality}")
    logger.info("")

    result = _run_demo(
        sd, latents, sidecar_deltas, brotli_quality=args.brotli_quality
    )

    # Pretty-print the rows.
    fmt = "{:<55}  {:>10}  {:>10}"
    logger.info(fmt.format("CELL", "BYTES", "Δ vs baseline"))
    logger.info("-" * 80)
    for row in result["rows"]:
        bytes_disp = "n/a" if row["bytes"] is None else f"{row['bytes']:,}"
        delta_disp = (
            "n/a"
            if row["delta_vs_baseline"] is None
            else f"{row['delta_vs_baseline']:+,}"
        )
        logger.info(fmt.format(str(row["label"])[:55], bytes_disp, delta_disp))
    logger.info("")
    logger.info(
        f"All numbers above tagged [empirical] — measured via brotli on "
        f"{args.state_dict.name} weights."
    )
    logger.info(
        "Note: latent_dim_order + sidecar_codebook do NOT affect the "
        "decoder-blob bytes; they affect the latent / sidecar streams (out "
        "of scope for this demo)."
    )

    # Write JSON output. CLAUDE.md "no /tmp paths" — default to dated lane dir.
    if args.output_dir is None:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("experiments/results") / f"lane_pr101_dynamic_derivers_{stamp}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_results.json"

    # Convert numpy/tuple objects for JSON serialization.
    def _json_default(obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"not JSON-serializable: {type(obj)}")

    out_path.write_text(json.dumps(result, indent=2, default=_json_default))
    logger.info(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
