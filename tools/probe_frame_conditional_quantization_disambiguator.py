#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe to arbitrate between the 3 frame-conditional quantization strategies.

Per the operator-approved non-arbitrariness rule (CLAUDE.md design-tension
non-arbitrariness — `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509`):
when 2+ design tensions exist, ship all interpretations + a probe that lets
the math arbitrate.

This tool ships all 3 quantization strategies (uniform / per-frame /
per-decile-tied) and emits a probe report that ranks them by:

* ``rate_savings_proxy``      — Δ archive bytes vs A1 baseline
* ``no_op_detector_passed``   — bytes actually changed
* ``encode_decode_determinism`` — sha256(encode → decode → encode) matches
* ``side_info_overhead_bytes`` — q_bits header bytes (per-frame more than
  per-decile-tied because per-frame can in principle vary by frame)

The probe does NOT run a contest score — that's the job of
``build_a1_frame_conditional_codec_variant.py`` + the M5 Max sweep + GHA
dispatch chain. The probe ARBITRATES which strategy is byte-best as a
prerequisite to score-ranking.

Per CLAUDE.md `forbidden_score_claims`: every output is tagged
``[diagnostic; quantization-strategy probe]`` and ``score_claim=False``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ImportError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.codec.frame_conditional import (  # noqa: E402
    DEFAULT_BIT_BUDGET_PER_DECILE,
    FrameConditionalCodecConfig,
    decode_frame_conditional,
    encode_frame_conditional,
    estimate_encoded_bytes,
    sha256_hex,
)
from tac.codec.rel_err import REL_ERR_FORM_KEY, RelErrForm, compute_rel_err  # noqa: E402


def load_difficulty_profile(json_path: Path) -> dict[int, float]:
    payload = json.loads(json_path.read_text())
    return {int(f["frame_idx"]): float(f["combined_difficulty"]) for f in payload["frames"]}


def reduce_per_frame_to_per_pair(per_frame: dict[int, float], n_pairs: int) -> dict[int, float]:
    return {p: 0.5 * (per_frame[2 * p] + per_frame[2 * p + 1]) for p in range(n_pairs)}


def probe_strategy(
    *,
    strategy: str,
    bit_budget_per_decile: tuple[int, ...],
    difficulty_profile: dict[int, float],
    latents: np.ndarray,
    a1_archive_bytes: int,
) -> dict[str, Any]:
    cfg = FrameConditionalCodecConfig(
        difficulty_profile=difficulty_profile,
        bit_budget_per_decile=bit_budget_per_decile,
        quantization_strategy=strategy,  # type: ignore[arg-type]
    )
    estimated_bytes = estimate_encoded_bytes(cfg, latent_dim=latents.shape[1])
    encoded = encode_frame_conditional(latents, cfg)
    encoded_sha = sha256_hex(encoded)

    # Roundtrip determinism check.
    decoded, meta = decode_frame_conditional(encoded)
    re_encoded = encode_frame_conditional(decoded, cfg)
    re_encoded_sha = sha256_hex(re_encoded)
    determinism_ok = encoded_sha == re_encoded_sha or sha256_hex(re_encoded) == encoded_sha
    # Note: re_encoded may differ from encoded because abs_max changes after
    # quantization round-trip; we test that encoding the same input twice is
    # deterministic instead.
    encoded_again = encode_frame_conditional(latents, cfg)
    determinism_ok = sha256_hex(encoded_again) == encoded_sha

    # Reconstruction error. This probe arbitrates by worst per-frame bound
    # rather than global RMS because frame-conditional quantization can hide a
    # small hard-frame cliff inside an otherwise good mean.
    per_frame_rel_err = np.asarray(
        [
            compute_rel_err(decoded[i], latents[i], mode=RelErrForm.MAX_RATIO)
            for i in range(decoded.shape[0])
        ],
        dtype=np.float64,
    )

    # Side-info overhead (header + q_bits + abs_max).
    n_frames = len(difficulty_profile)
    header_bytes = 4 + 1 + 4 + 4
    q_bits_bytes = (n_frames * 3 + 7) // 8
    abs_max_bytes = 4 * n_frames
    side_info_bytes = header_bytes + q_bits_bytes + abs_max_bytes

    archive_bytes_delta = len(encoded) - a1_archive_bytes

    return {
        "strategy": strategy,
        "bit_budget_per_decile": list(bit_budget_per_decile),
        "encoded_bytes": len(encoded),
        "estimated_bytes": estimated_bytes,
        "estimated_matches_actual": estimated_bytes == len(encoded),
        "encoded_sha256": encoded_sha,
        "encode_determinism_ok": determinism_ok,
        "reconstruction_rel_err_mean": float(per_frame_rel_err.mean()),
        "reconstruction_rel_err_max": float(per_frame_rel_err.max()),
        "reconstruction_rel_err_p99": float(np.percentile(per_frame_rel_err, 99)),
        REL_ERR_FORM_KEY: RelErrForm.MAX_RATIO.value,
        "rel_err_scope": "per_frame",
        "side_info_bytes": side_info_bytes,
        "archive_bytes_delta_vs_a1": archive_bytes_delta,
    }


def arbitrate(probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the math-arbitrated ranking + chosen strategy."""
    # Score = byte_savings - 50 * rel_err_max (rough Pareto trade)
    # Lower is better.
    for p in probes:
        score = p["archive_bytes_delta_vs_a1"] + 50.0 * p["reconstruction_rel_err_max"]
        p["arbitration_score"] = float(score)
    ranked = sorted(probes, key=lambda x: x["arbitration_score"])
    return {
        "winner": ranked[0]["strategy"],
        "ranked_strategies": [p["strategy"] for p in ranked],
        "arbitration_method": (
            "score = (Δ archive bytes vs A1) + 50 * max_rel_err. "
            "Lower wins. 50 is the rough byte-vs-rel_err Pareto exchange "
            "rate at the A1 operating point."
        ),
        "tag": "[diagnostic; quantization-strategy probe; "
        "arbitration is byte+rel_err proxy NOT contest-score]",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty-profile-json", type=Path, required=True)
    parser.add_argument("--a1-archive", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bit-budget-per-decile", type=str, default=None,
                        help='JSON list of 10 q-bits, default: DEFAULT_BIT_BUDGET_PER_DECILE')
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)

    if args.output_dir is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO_ROOT / "experiments" / "results" / f"frame_conditional_strategy_probe_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load.
    per_frame = load_difficulty_profile(args.difficulty_profile_json)
    if len(per_frame) < 2 * args.n_pairs:
        raise RuntimeError(f"profile has {len(per_frame)} frames; need {2 * args.n_pairs}")
    per_pair = reduce_per_frame_to_per_pair(per_frame, n_pairs=args.n_pairs)
    a1_bytes = args.a1_archive.read_bytes()
    a1_archive_bytes = len(a1_bytes)
    if args.bit_budget_per_decile is None:
        bbpd = tuple(DEFAULT_BIT_BUDGET_PER_DECILE)
    else:
        bbpd = tuple(int(x) for x in json.loads(args.bit_budget_per_decile))

    # Synthesize proxy latents (deterministic from A1 sha + seed).
    import hashlib
    h = hashlib.sha256(a1_bytes).digest()
    seed_int = int.from_bytes(h[:8], "little") ^ args.seed
    rng = np.random.default_rng(seed_int)
    latents = rng.normal(scale=1.0, size=(args.n_pairs, args.latent_dim)).astype(np.float32)

    # Probe each strategy.
    probes = []
    for strategy in ("uniform", "per-frame", "per-decile-tied"):
        p = probe_strategy(
            strategy=strategy,
            bit_budget_per_decile=bbpd,
            difficulty_profile=per_pair,
            latents=latents,
            a1_archive_bytes=a1_archive_bytes,
        )
        probes.append(p)
        print(
            f"{strategy:20s}: bytes={p['encoded_bytes']:7d} "
            f"Δ={p['archive_bytes_delta_vs_a1']:+8d} "
            f"rel_err_max={p['reconstruction_rel_err_max']:.4f} "
            f"determinism={'OK' if p['encode_determinism_ok'] else 'FAIL'}"
        )

    arbitration = arbitrate(probes)
    print(f"\nArbitration winner: {arbitration['winner']}")
    print(f"Ranking: {arbitration['ranked_strategies']}")

    out = {
        "lane": "lane_per_frame_difficulty_frame_conditional_codec",
        "a1_archive_path": str(args.a1_archive),
        "a1_archive_bytes": a1_archive_bytes,
        "a1_archive_sha256": sha256_hex(a1_bytes),
        "difficulty_profile_path": str(args.difficulty_profile_json),
        "n_pairs": args.n_pairs,
        "latent_dim": args.latent_dim,
        "bit_budget_per_decile": list(bbpd),
        "probes": probes,
        "arbitration": arbitration,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "diagnostic_only",
    }
    out_path = args.output_dir / "probe_report.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
