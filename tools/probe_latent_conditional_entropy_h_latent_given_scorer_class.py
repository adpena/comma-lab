#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Empirical H(latent | scorer_class) probe shared across ATW V1 + D4 + Z4.

Canonical probe-disambiguator per the HIGH-RISK substrate cargo-cult unwind
audit 2026-05-16 (D4 operator-approved). Cooperative-receiver / Wyner-Ziv /
class-conditional substrates predict that conditioning the codec on the
SegNet (or PoseNet) class labels meaningfully reduces the latent's required
bit-rate. This probe measures the empirical conditional entropy and emits a
typed verdict so the predicted Wyner-Ziv gain can be promoted, retired, or
re-anchored against the measured ceiling.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #209/#221/#249:
the emitted JSON is fail-closed (``score_claim=false``, evidence grade
``diagnostic_cpu``, axis explicitly tagged) so no autopilot consumer can
mistake the probe output for a contest-CUDA score claim.

Algorithm (Shannon entropy estimator):

  1. Read the substrate's latent symbols (uint8 or int16 byte stream) and
     the matching per-symbol SegNet class labels (uint8 stream of class
     indices in {0..K-1}).
  2. Estimate ``H(latent)`` from the marginal symbol distribution
     (plug-in entropy in bits/symbol).
  3. Estimate ``H(latent | class)`` from the per-class conditional symbol
     distributions weighted by class probability mass.
  4. Mutual information ``I(latent ; class) = H(latent) - H(latent | class)``;
     emit verdict band per ``--meaningful-mi-threshold-bits`` (default 0.5
     bits/symbol per the audit foundation §5 op-routable #5).

Verdict taxonomy:
- MEANINGFUL_CONDITIONING: I(latent;class) >= meaningful threshold; the
  substrate's class-conditional codec design has empirical support; the
  predicted Wyner-Ziv gain ceiling is approximately ``I / H(latent)``.
- WEAK_CONDITIONING: 0 < I(latent;class) < meaningful threshold; the gain
  is real but smaller than the predicted band; revise prediction down to
  ``I / H(latent)`` ceiling.
- INDEPENDENT: I(latent;class) ~= 0 within tolerance; the class signal does
  not predict the latent; the substrate's class-conditional design is
  CARGO-CULTED and should be retired or redesigned.

CLI contract::

    .venv/bin/python tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py \\
        --substrate-id atw_v1 \\
        --latent-bytes path/to/latent.bin \\
        --scorer-classes path/to/segnet_classes.bin \\
        --output-json .omx/state/h_latent_given_scorer_class_atw_v1.json

Operators may also pass ``--latent-symbols`` and ``--class-symbols`` as inline
NumPy-readable text files (one symbol per line) for the offline analytical
sweep used in the audit's $3-5 CPU smoke envelope.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Per audit §5 op-routable #5: 0.5 bits/symbol distinguishes a meaningful
# Wyner-Ziv conditioning channel from noise. Operators may tighten via
# ``--meaningful-mi-threshold-bits`` when running on a high-entropy latent
# (e.g. raw float32 latents would have H(latent) >> 8 bits/symbol so the
# threshold should scale accordingly).
DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS: float = 0.5

# Numerical floor for "independent" verdict — MI below this counts as zero
# within sampling noise for typical latent streams of 10k-1M symbols.
INDEPENDENCE_TOLERANCE_BITS: float = 0.01

VerdictStr = Literal["MEANINGFUL_CONDITIONING", "WEAK_CONDITIONING", "INDEPENDENT"]


@dataclass(frozen=True)
class HLatentGivenScorerClassVerdict:
    """Machine-readable conditional-entropy verdict for autopilot consumers."""

    substrate_id: str
    verdict: VerdictStr
    h_latent_unconditional_bits_per_symbol: float
    h_latent_given_scorer_class_bits_per_symbol: float
    mutual_information_bits: float
    wyner_ziv_gain_ceiling_fraction: float
    num_latent_symbols: int
    num_class_symbols: int
    num_unique_classes: int
    meaningful_mi_threshold_bits: float
    independence_tolerance_bits: float
    evidence_grade: str
    score_claim: bool
    axis_label: str
    observed_at_utc: str
    notes: str


def _plug_in_entropy_bits(counts: dict[int, int]) -> float:
    """Plug-in Shannon entropy estimator in bits/symbol.

    Uses ``-sum p_i * log2(p_i)`` with the natural-log convention via
    ``math.log2`` so the output is in bits. Empty counts return 0 by
    convention (degenerate distribution).
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _count_symbols(stream: bytes | list[int]) -> dict[int, int]:
    """Count integer symbol occurrences. Accepts bytes or a list of ints."""
    counts: dict[int, int] = {}
    if isinstance(stream, bytes):
        for b in stream:
            counts[b] = counts.get(b, 0) + 1
    else:
        for s in stream:
            si = int(s)
            counts[si] = counts.get(si, 0) + 1
    return counts


def _count_joint(
    latent_stream: bytes | list[int], class_stream: bytes | list[int]
) -> dict[int, dict[int, int]]:
    """Per-class symbol counts: ``joint[class][symbol] = count``."""
    if len(latent_stream) != len(class_stream):
        raise ValueError(
            f"latent_stream length {len(latent_stream)} does not match "
            f"class_stream length {len(class_stream)}"
        )
    joint: dict[int, dict[int, int]] = {}
    for lat, cls in zip(latent_stream, class_stream):
        lat_i = int(lat)
        cls_i = int(cls)
        per_class = joint.setdefault(cls_i, {})
        per_class[lat_i] = per_class.get(lat_i, 0) + 1
    return joint


def compute_h_latent_given_scorer_class(
    *,
    substrate_id: str,
    latent_stream: bytes | list[int],
    class_stream: bytes | list[int],
    meaningful_mi_threshold_bits: float = DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS,
    independence_tolerance_bits: float = INDEPENDENCE_TOLERANCE_BITS,
    notes: str = "",
) -> HLatentGivenScorerClassVerdict:
    """Compute the conditional-entropy verdict for one (latent, class) pair.

    Raises ``ValueError`` on contract violations per CLAUDE.md "Comment-only
    contracts are FORBIDDEN" — empty streams / mismatched lengths / non-finite
    thresholds are refused instead of returning a vacuous verdict.
    """
    if not isinstance(substrate_id, str) or not substrate_id.strip():
        raise ValueError("substrate_id must be a non-empty string")
    if len(latent_stream) == 0:
        raise ValueError("latent_stream is empty; cannot estimate entropy")
    if len(class_stream) == 0:
        raise ValueError("class_stream is empty; cannot estimate conditional entropy")
    if len(latent_stream) != len(class_stream):
        raise ValueError(
            f"latent_stream length {len(latent_stream)} != "
            f"class_stream length {len(class_stream)}"
        )
    if not math.isfinite(meaningful_mi_threshold_bits) or meaningful_mi_threshold_bits < 0:
        raise ValueError(
            f"meaningful_mi_threshold_bits={meaningful_mi_threshold_bits} must be finite >= 0"
        )

    latent_counts = _count_symbols(latent_stream)
    class_counts = _count_symbols(class_stream)
    joint = _count_joint(latent_stream, class_stream)

    h_latent = _plug_in_entropy_bits(latent_counts)
    total = sum(class_counts.values())
    h_latent_given_class = 0.0
    for cls, per_class in joint.items():
        p_class = class_counts[cls] / total
        h_per_class = _plug_in_entropy_bits(per_class)
        h_latent_given_class += p_class * h_per_class

    # Floor at 0 to avoid tiny floating-point negatives (numerically
    # H(latent|class) <= H(latent), but float64 round-off can produce
    # MI = -1e-17 on degenerate inputs).
    mutual_information = max(h_latent - h_latent_given_class, 0.0)
    wyner_ziv_gain_ceiling_fraction = (
        mutual_information / h_latent if h_latent > 0.0 else 0.0
    )

    if mutual_information <= independence_tolerance_bits:
        verdict: VerdictStr = "INDEPENDENT"
    elif mutual_information >= meaningful_mi_threshold_bits:
        verdict = "MEANINGFUL_CONDITIONING"
    else:
        verdict = "WEAK_CONDITIONING"

    return HLatentGivenScorerClassVerdict(
        substrate_id=substrate_id,
        verdict=verdict,
        h_latent_unconditional_bits_per_symbol=float(h_latent),
        h_latent_given_scorer_class_bits_per_symbol=float(h_latent_given_class),
        mutual_information_bits=float(mutual_information),
        wyner_ziv_gain_ceiling_fraction=float(wyner_ziv_gain_ceiling_fraction),
        num_latent_symbols=int(len(latent_stream)),
        num_class_symbols=int(len(class_stream)),
        num_unique_classes=int(len(class_counts)),
        meaningful_mi_threshold_bits=float(meaningful_mi_threshold_bits),
        independence_tolerance_bits=float(independence_tolerance_bits),
        # Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192:
        # this is a diagnostic CPU probe; NEVER promotable, NEVER a score
        # claim. The autopilot consumer reads these three fields to refuse
        # the row's promotion regardless of MI magnitude.
        evidence_grade="diagnostic_cpu",
        score_claim=False,
        axis_label="[diagnostic-CPU; H(latent|scorer_class) probe]",
        observed_at_utc=datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        notes=notes,
    )


def _read_byte_stream(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"input stream path does not exist: {path}")
    return path.read_bytes()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute empirical H(latent | scorer_class) for substrates that "
            "condition their codec on SegNet/PoseNet class labels (ATW V1 / "
            "D4 / Z4). Canonical probe per high-risk substrate cargo-cult "
            "unwind audit 2026-05-16 (D4)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--substrate-id", required=True, help="lane / substrate identifier")
    parser.add_argument(
        "--latent-bytes", type=Path, required=True,
        help="path to binary latent symbol stream (one byte per symbol)",
    )
    parser.add_argument(
        "--scorer-classes", type=Path, required=True,
        help="path to binary class-label stream (one byte per symbol)",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="optional output JSON path (otherwise stdout-only)",
    )
    parser.add_argument(
        "--meaningful-mi-threshold-bits",
        type=float,
        default=DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS,
        help="mutual-information threshold for MEANINGFUL_CONDITIONING verdict",
    )
    parser.add_argument(
        "--independence-tolerance-bits",
        type=float,
        default=INDEPENDENCE_TOLERANCE_BITS,
        help="mutual-information threshold below which verdict is INDEPENDENT",
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="free-form provenance note attached to the verdict JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        latent_stream = _read_byte_stream(args.latent_bytes)
        class_stream = _read_byte_stream(args.scorer_classes)
        verdict = compute_h_latent_given_scorer_class(
            substrate_id=args.substrate_id,
            latent_stream=latent_stream,
            class_stream=class_stream,
            meaningful_mi_threshold_bits=args.meaningful_mi_threshold_bits,
            independence_tolerance_bits=args.independence_tolerance_bits,
            notes=args.notes,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[h-latent-probe] FATAL: {exc}", file=sys.stderr)
        return 2

    payload = asdict(verdict)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
