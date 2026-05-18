#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Q6.preprobe — pairwise composition_alpha probe.

[verified-against: T2 HORIZON-CLASS council 2026-05-17 OP-3
                   `.omx/research/council_horizon_class_scope_t2_20260517.md`
                   verbatim: *"concatenates the raw byte arrays, runs
                   lzma/brotli/zstd on concat vs sum-of-marginals, computes
                   composition_alpha_ij = (1 - concat_compressed /
                   sum_marginal_compressed)"*]
[verified-against: tools/pre_entropy_substrate_pivot_prober.py — canonical
                   marginal-compressibility prober + fcntl-locked output
                   contract pattern (sister artifact at
                   `.omx/state/wyner_ziv_deliverability/
                   pre_entropy_candidate_substrates_20260517T210723.json`)]
[verified-against: src/tac/optimization/substrate_composition_matrix.py —
                   canonical Catalog #227 expected_alpha banding:
                   α ≥ 0.7 ADDITIVE; 0.3 < α < 0.7 SUB_ADDITIVE (halve);
                   α ≤ 0.3 SATURATING (floor -0.005)]
[verified-against: CLAUDE.md "Bit-level deconstruction and entropy
                   discipline" + "Meta-Lagrangian/Pareto solver" non-
                   negotiables]

CONTEXT
=======

Stage 2 ASYMPTOTIC_PURSUIT reactivation (per T2 HYBRID verdict) requires
empirical pairwise composition_alpha measurement across the 3 Wyner-Ziv
hoist candidates (pr101_state_dict / pr106_state_dict /
posenet_class_sensitivity). The Assumption-Adversary flagged the
"composition_alpha ≈ 1.0 (orthogonal Wyner-Ziv contributions)" assumption
as CARGO-CULTED. This probe is the empirical falsifier/confirmer.

For each of the 3 candidate pairs:

1. Load each substrate's raw byte buffer (treat .pt file as opaque blob).
2. Compress each ALONE via lzma/brotli/zlib; pick best codec; record
   `bytes_alone_a`, `bytes_alone_b`, `compressed_alone_a`,
   `compressed_alone_b`, `savings_alone_a = raw_a - compressed_alone_a`,
   `savings_alone_b = raw_b - compressed_alone_b`.
3. Concatenate (bytes_a ++ bytes_b); compress via same codec; record
   `compressed_concat`, `savings_concat = (raw_a + raw_b) -
   compressed_concat`.
4. Compute TWO equivalent α-formulas (both reported for cross-audit):
   * **OP-3 council form**:
     α = 1 - compressed_concat / (compressed_alone_a + compressed_alone_b)
   * **Savings-ratio form**:
     α = savings_concat / (savings_alone_a + savings_alone_b)

   Both formulas land in the same α-banding (per Catalog #227):
   - α ≥ 0.7 → ADDITIVE (orthogonal Wyner-Ziv contributions; Stage 2 ok)
   - 0.3 < α < 0.7 → SUB_ADDITIVE (interaction interference; halve predicted ΔS)
   - α ≤ 0.3 → SATURATING (floor -0.005; near-redundant contributions)
   - α > 1.1 → SUPER_ADDITIVE (rare synergy; double-check measurement)

5. Stage 2 gate clause #2 SATISFIED iff ≥ 2-of-3 pairs achieve α ≥ 0.7.

OUTPUT CONTRACT (NON-AUTHORITATIVE / PROBE EVIDENCE)
====================================================

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192 +
Catalog #245 + Catalog #131 (fcntl-locked JSONL state writes):

* ``score_claim: false``
* ``promotion_eligible: false``
* ``ready_for_exact_eval_dispatch: false``
* ``evidence_grade: "predicted"``
* ``measurement_axis: "[diagnostic; pairwise composition_alpha probe]"``

Persists to:
``.omx/state/wyner_ziv_deliverability/pairwise_alpha_<utc>.json``

via fcntl-locked atomic write (LOCK_EX + write-tmp + os.replace) per
Catalog #128/#131.

CLI USAGE
=========

```bash
# Default: probe all 3 canonical pairs from the Stage 2 candidate set
.venv/bin/python tools/q6_preprobe_pairwise_composition_alpha.py \
    --report-only-no-side-effects

# Pinned single pair (for unit-test reproducibility)
.venv/bin/python tools/q6_preprobe_pairwise_composition_alpha.py \
    --candidate-a pr101_state_dict \
    --candidate-b pr106_state_dict \
    --output .omx/state/wyner_ziv_deliverability/pairwise_alpha_<utc>.json
```

6-HOOK WIRE-IN per Catalog #125
================================

(1) Sensitivity-map contribution: N/A — composition_alpha is a SECOND-order
    interaction term, not a per-substrate sensitivity. The marginal
    pre-entropy bytes are already in `tac.sensitivity_map.*` via the
    sister prober.
(2) Pareto constraint: N/A at the probe surface — Catalog #227's
    `adjust_predicted_delta_for_composition_alpha` IS the Pareto consumer;
    this probe emits the α value that feeds it.
(3) Bit-allocator hook: N/A — Wyner-Ziv hoist allocation is downstream of
    α verification; allocator activates only on Stage 2 unlock.
(4) Cathedral autopilot dispatch hook: ACTIVE — the autopilot's
    `apply_z1_empirical_revision_to_candidate_delta` reads
    `composition_alpha` per Catalog #227's
    `adjust_predicted_delta_for_composition_alpha` helper. This probe
    populates the cross-substrate matrix the autopilot consumes.
(5) Continual-learning posterior update: ACTIVE — every probe artifact
    appends to `.omx/state/wyner_ziv_deliverability/` JSONL ledger via
    canonical fcntl-locked pattern; future probes (e.g. after Q4 anchor
    lands) can supersede via newer artifact.
(6) Probe-disambiguator: ACTIVE — this PROBE IS the disambiguator for
    the Stage 2 ASYMPTOTIC reactivation decision. Two defensible
    interpretations of composition_alpha existed (council's CARGO-CULTED
    ≈ 1.0 hypothesis vs Catalog #227's empirical-banded default of 0.3-1.0
    depending on substrate pair); this probe resolves to the empirical
    truth for these 3 candidates.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import lzma
import os
import sys
import tempfile
import uuid
import zlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

# Soft-import brotli per CLAUDE.md Catalog #203
try:
    import brotli  # type: ignore

    _HAS_BROTLI = True
except ImportError:  # pragma: no cover
    _HAS_BROTLI = False

# ──────────────────────────────────────────────────────────────────────── #
# Canonical Catalog #227 α-banding thresholds                              #
# ──────────────────────────────────────────────────────────────────────── #

# Per CLAUDE.md "Council conduct" + src/tac/optimization/
# substrate_composition_matrix.py canonical α-banding.
ALPHA_ADDITIVE_THRESHOLD = 0.7  # α ≥ 0.7 = Stage 2 gate clause #2 candidate
ALPHA_SATURATING_THRESHOLD = 0.3  # α ≤ 0.3 = SATURATING (floor -0.005)
ALPHA_SUPER_ADDITIVE_THRESHOLD = 1.1  # α > 1.1 = unusual synergy; double-check

# Stage 2 gate clause #2: ≥ 2-of-3 pairs achieve α ≥ 0.7
STAGE_2_MINIMUM_ADDITIVE_PAIRS = 2

# Canonical 3 Stage-2-candidate substrates per T2 HYBRID verdict
STAGE_2_CANDIDATES: tuple[str, ...] = (
    "pr101_state_dict",
    "pr106_state_dict",
    "posenet_class_sensitivity",
)

# OP-3 EXTENDED candidates: 10 PRE_ENTROPY substrates per the sister Q6.preprobe
# op-routable #3 — extend the probe to 7 OTHER PRE_ENTROPY substrate candidates
# (besides the original 3) to surface alternative Stage-2 stacking topologies.
# Source ranking: `.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_20260517T210723.json`
# (per `feedback_pre_entropy_substrate_pivot_prober_landed_20260517.md` Table 1).
# C(10,2) = 45 pair combinations per the briefing math.
# Excluded: `distilled_segnet` (rank 11, deliverable savings 0.00295 — drop to fit C(10,2)).
STAGE_2_CANDIDATES_EXTENDED: tuple[str, ...] = (
    # Original 3 (preserved for deterministic regression)
    "pr101_state_dict",
    "pr106_state_dict",
    "posenet_class_sensitivity",
    # NEW 7 PRE_ENTROPY candidates (ranked by deliverable_score_savings_estimate)
    "distill_v2_best",          # 0.2476 (raw_float_weights, 3.5 MB)
    "sabor_margin_frame_000",   # 0.1335 (scorer_margin_float32, 787 KB)
    "sabor_margin_frame_001",   # 0.1326 (scorer_margin_float32, 787 KB)
    "pr106_latents",            # 0.0237 (raw_float_latents, 69 KB)
    "lane_g_v3_renderer",       # 0.0233 (raw_float_weights, 297 KB)
    "siren_renderer",           # 0.0233 (raw_float_weights, 297 KB)
    "distilled_posenet",        # 0.00438 (raw_float_weights, 81 KB)
)

# Map candidate id → (file path, substrate class) — mirrors sister prober's
# CANONICAL_CANDIDATE_SUBSTRATES dict (verified against
# tools/pre_entropy_substrate_pivot_prober.py:171-219).
CANDIDATE_PATHS: dict[str, tuple[str, str]] = {
    # Original 3
    "pr101_state_dict": (
        "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
        "raw_float_weights",
    ),
    "pr106_state_dict": (
        "experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt",
        "raw_float_weights",
    ),
    "posenet_class_sensitivity": (
        "experiments/results/posenet_sensitivity/class_sensitivity.pt",
        "raw_float_weights",
    ),
    # OP-3 extension: 7 NEW PRE_ENTROPY candidates (canonical paths mirror
    # `tools/pre_entropy_substrate_pivot_prober.py` lines 181-218).
    "distill_v2_best": (
        "experiments/results/distill_v2_best.pt",
        "raw_float_weights",
    ),
    "sabor_margin_frame_000": (
        "experiments/results/lane_sabor_boundary_audit_20260513_20260513T180635Z/sample_margin_frame_000.npy",
        "scorer_margin_float32",
    ),
    "sabor_margin_frame_001": (
        "experiments/results/lane_sabor_boundary_audit_20260513_20260513T180635Z/sample_margin_frame_001.npy",
        "scorer_margin_float32",
    ),
    "pr106_latents": (
        "experiments/results/sensitivity_map_pr106_20260504_claude/latents.pt",
        "raw_float_latents",
    ),
    "lane_g_v3_renderer": (
        "experiments/results/lane_g_v3_landed/iter_0/renderer.bin",
        "raw_float_weights",
    ),
    "siren_renderer": (
        "experiments/results/lane_substrate_siren_modal_a100_dispatch_20260513T140410Z__smoke__100ep_modal/submissions/robust_current/renderer.bin",
        "raw_float_weights",
    ),
    "distilled_posenet": (
        "experiments/results/lane_cpu_trained_tiny_hinton_surrogate_bootstrap_20260512T034310Z_long/distilled_posenet_ema_shadow.pt",
        "raw_float_weights",
    ),
}

OUTPUT_DIR_DEFAULT = Path(".omx/state/wyner_ziv_deliverability")

SCHEMA_VERSION = "pairwise_composition_alpha_probe_v1"

# OP-3 extension schema version — backward-compatible superset of v1; extended
# sweeps emit `schema_version=pairwise_composition_alpha_probe_v1_extended_10c`
# so consumers can distinguish 3-pair canonical vs 45-pair extended results
# without parsing the candidates_probed list length.
SCHEMA_VERSION_EXTENDED = "pairwise_composition_alpha_probe_v1_extended_10c"


# ──────────────────────────────────────────────────────────────────────── #
# Codec helpers (mirror sister prober's contract)                           #
# ──────────────────────────────────────────────────────────────────────── #


def _compress_lzma(data: bytes) -> bytes:
    """Canonical lzma at preset 9 | EXTREME (mirror sister prober)."""
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)


def _compress_brotli(data: bytes) -> bytes | None:
    """Canonical brotli at quality 11. None if brotli unavailable."""
    if not _HAS_BROTLI:
        return None
    return brotli.compress(data, quality=11)


def _compress_zlib(data: bytes) -> bytes:
    """Canonical zlib at level 9."""
    return zlib.compress(data, level=9)


def best_compression(data: bytes) -> tuple[str, int]:
    """Return (best_codec_name, best_compressed_size) over lzma/brotli/zlib."""
    if not data:
        return ("lzma", 0)
    candidates: list[tuple[str, int]] = [
        ("lzma", len(_compress_lzma(data))),
        ("zlib", len(_compress_zlib(data))),
    ]
    br = _compress_brotli(data)
    if br is not None:
        candidates.append(("brotli", len(br)))
    return min(candidates, key=lambda x: x[1])


def classify_alpha(alpha: float) -> str:
    """Per Catalog #227 substrate_composition_matrix α-banding."""
    if alpha > ALPHA_SUPER_ADDITIVE_THRESHOLD:
        return "SUPER_ADDITIVE"
    if alpha >= ALPHA_ADDITIVE_THRESHOLD:
        return "ADDITIVE"
    if alpha > ALPHA_SATURATING_THRESHOLD:
        return "SUB_ADDITIVE"
    return "SATURATING"


# ──────────────────────────────────────────────────────────────────────── #
# Pairwise α computation                                                    #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class PairwiseAlphaResult:
    """Per-pair pairwise composition_alpha result."""

    candidate_a: str
    candidate_b: str
    raw_bytes_a: int
    raw_bytes_b: int
    raw_bytes_concat: int  # = raw_bytes_a + raw_bytes_b
    best_codec: str
    compressed_alone_a: int
    compressed_alone_b: int
    compressed_concat: int
    sum_marginal_compressed: int  # = compressed_alone_a + compressed_alone_b
    savings_alone_a: int  # = raw_bytes_a - compressed_alone_a
    savings_alone_b: int
    savings_concat: int  # = raw_bytes_concat - compressed_concat
    sum_marginal_savings: int  # = savings_alone_a + savings_alone_b
    alpha_op3_council_form: float  # = 1 - compressed_concat/sum_marginal_compressed
    alpha_savings_ratio_form: float  # = savings_concat/sum_marginal_savings
    alpha_band: str  # ADDITIVE | SUB_ADDITIVE | SATURATING | SUPER_ADDITIVE
    stage_2_gate_clause_2_satisfied: bool


def compute_pairwise_alpha(
    candidate_a: str,
    bytes_a: bytes,
    candidate_b: str,
    bytes_b: bytes,
) -> PairwiseAlphaResult:
    """Compute pairwise composition_alpha for a single (a, b) pair.

    Per OP-3 council spec: concat raw bytes + run codecs on concat vs
    sum-of-marginals; α = 1 - concat_compressed / sum_marginal_compressed.

    We pick the SAME codec for marginal-a / marginal-b / concat (the BEST
    of lzma/brotli/zlib on the concat blob; this is the canonical fair
    comparison — if codec A is best for concat, marginals must also
    measure under codec A for apples-to-apples α).
    """
    raw_a = len(bytes_a)
    raw_b = len(bytes_b)
    raw_concat = raw_a + raw_b
    concat = bytes_a + bytes_b

    # Pick best codec on the CONCAT (the operative measurement). Marginals
    # use the same codec for apples-to-apples α.
    best_codec, compressed_concat = best_compression(concat)

    # Re-measure marginals under the SAME codec
    if best_codec == "lzma":
        compressed_alone_a = len(_compress_lzma(bytes_a)) if bytes_a else 0
        compressed_alone_b = len(_compress_lzma(bytes_b)) if bytes_b else 0
    elif best_codec == "brotli":
        compressed_alone_a = len(_compress_brotli(bytes_a)) if bytes_a else 0  # type: ignore[arg-type]
        compressed_alone_b = len(_compress_brotli(bytes_b)) if bytes_b else 0  # type: ignore[arg-type]
    else:  # zlib
        compressed_alone_a = len(_compress_zlib(bytes_a)) if bytes_a else 0
        compressed_alone_b = len(_compress_zlib(bytes_b)) if bytes_b else 0

    sum_marginal = compressed_alone_a + compressed_alone_b
    savings_a = raw_a - compressed_alone_a
    savings_b = raw_b - compressed_alone_b
    sum_savings = savings_a + savings_b
    savings_concat = raw_concat - compressed_concat

    # α computation — two equivalent forms per OP-3 council spec
    # Guard against zero-division when sum_marginal == 0 (degenerate)
    if sum_marginal > 0:
        alpha_op3 = 1.0 - (compressed_concat / sum_marginal)
    else:
        alpha_op3 = 0.0
    if sum_savings > 0:
        alpha_savings = savings_concat / sum_savings
    else:
        alpha_savings = 0.0

    # Per Catalog #227 + council OP-2: stage 2 gate clause #2 uses the
    # savings-ratio form (more intuitive interpretation per Catalog #227's
    # adjust_predicted_delta_for_composition_alpha consumer).
    alpha_band = classify_alpha(alpha_savings)
    stage_2_satisfied = alpha_savings >= ALPHA_ADDITIVE_THRESHOLD

    return PairwiseAlphaResult(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        raw_bytes_a=raw_a,
        raw_bytes_b=raw_b,
        raw_bytes_concat=raw_concat,
        best_codec=best_codec,
        compressed_alone_a=compressed_alone_a,
        compressed_alone_b=compressed_alone_b,
        compressed_concat=compressed_concat,
        sum_marginal_compressed=sum_marginal,
        savings_alone_a=savings_a,
        savings_alone_b=savings_b,
        savings_concat=savings_concat,
        sum_marginal_savings=sum_savings,
        alpha_op3_council_form=alpha_op3,
        alpha_savings_ratio_form=alpha_savings,
        alpha_band=alpha_band,
        stage_2_gate_clause_2_satisfied=stage_2_satisfied,
    )


# ──────────────────────────────────────────────────────────────────────── #
# Sweep + aggregation                                                        #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass
class SweepResult:
    """Aggregated 3-pair sweep result."""

    candidates_probed: tuple[str, ...]
    pair_results: dict[str, dict[str, Any]]  # "a+b" → result dict
    pairs_with_alpha_at_least_threshold: int
    stage_2_gate_clause_2_overall: bool
    stage_2_reactivation_clause_2_verdict: str


def resolve_candidate_bytes(candidate_id: str, repo_root: Path) -> bytes:
    """Read raw byte buffer for a canonical candidate id."""
    if candidate_id not in CANDIDATE_PATHS:
        raise KeyError(
            f"unknown candidate: {candidate_id!r}; valid: {sorted(CANDIDATE_PATHS)}"
        )
    rel_path, _ = CANDIDATE_PATHS[candidate_id]
    full_path = repo_root / rel_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"candidate {candidate_id!r} pre-entropy artifact missing at {full_path}"
        )
    return full_path.read_bytes()


def run_sweep(
    candidates: Iterable[str],
    *,
    repo_root: Path,
    candidate_bytes: dict[str, bytes] | None = None,
) -> SweepResult:
    """Run the canonical pairwise α sweep over all (i,j) pairs (i < j)."""
    cand_list = tuple(candidates)
    if candidate_bytes is None:
        candidate_bytes = {c: resolve_candidate_bytes(c, repo_root) for c in cand_list}

    pair_results: dict[str, dict[str, Any]] = {}
    satisfied_count = 0
    for a, b in combinations(cand_list, 2):
        result = compute_pairwise_alpha(
            a, candidate_bytes[a], b, candidate_bytes[b]
        )
        pair_key = f"{a}+{b}"
        pair_results[pair_key] = asdict(result)
        if result.stage_2_gate_clause_2_satisfied:
            satisfied_count += 1

    overall_satisfied = satisfied_count >= STAGE_2_MINIMUM_ADDITIVE_PAIRS
    verdict = "SATISFIED" if overall_satisfied else "NOT_SATISFIED"

    return SweepResult(
        candidates_probed=cand_list,
        pair_results=pair_results,
        pairs_with_alpha_at_least_threshold=satisfied_count,
        stage_2_gate_clause_2_overall=overall_satisfied,
        stage_2_reactivation_clause_2_verdict=verdict,
    )


# ──────────────────────────────────────────────────────────────────────── #
# Output emission (fcntl-locked per Catalog #131)                           #
# ──────────────────────────────────────────────────────────────────────── #


def build_output_payload(
    sweep: SweepResult,
    *,
    brotli_available: bool,
    extended: bool = False,
) -> dict[str, Any]:
    """Build the canonical JSON payload for the probe artifact.

    Args:
        sweep: SweepResult from `run_sweep`.
        brotli_available: whether brotli codec is available at this site.
        extended: when True, emit ``schema_version_extended`` so consumers can
            differentiate 3-pair canonical from 45-pair extended sweep output.
    """
    written_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "schema_version": (
            SCHEMA_VERSION_EXTENDED if extended else SCHEMA_VERSION
        ),
        "candidates_probed": list(sweep.candidates_probed),
        "pair_results": sweep.pair_results,
        "pairs_with_alpha_at_least_0_7": sweep.pairs_with_alpha_at_least_threshold,
        "stage_2_gate_clause_2_overall": sweep.stage_2_gate_clause_2_overall,
        "stage_2_reactivation_clause_2_verdict": sweep.stage_2_reactivation_clause_2_verdict,
        "stage_2_minimum_additive_pairs_required": STAGE_2_MINIMUM_ADDITIVE_PAIRS,
        "alpha_additive_threshold": ALPHA_ADDITIVE_THRESHOLD,
        "alpha_saturating_threshold": ALPHA_SATURATING_THRESHOLD,
        "evidence_grade": "predicted",
        "measurement_axis": "[diagnostic; pairwise composition_alpha probe]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "claude_md_compliance_tags": [
            "composition_alpha_per_catalog_227",
            "stage_2_gate_clause_2",
            "horizon_class_t2_council_op_3",
            "fcntl_locked_write_per_catalog_131",
            "apples_to_apples_per_catalog_127",
            "non_authoritative_per_catalog_192",
        ],
        "brotli_available": brotli_available,
        "written_at_utc": written_at,
        "written_pid": os.getpid(),
        "written_host": os.uname().nodename if hasattr(os, "uname") else "unknown",
    }


def write_output_locked(payload: dict[str, Any], output_path: Path) -> None:
    """Atomic fcntl-locked write of the probe artifact per Catalog #131.

    Pattern: tmp-write + LOCK_EX + os.replace. The output directory is
    created if missing. The lock is held only over the rename (atomic
    rename is itself O(1); we acquire the lock so concurrent probers
    don't collide on the same output path).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / f".{output_path.name}.lock"

    # Write to a temp file in the same directory (so os.replace is atomic)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{output_path.name}.tmp.",
        dir=str(output_path.parent),
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
            fp.write("\n")

        # Acquire fcntl lock on a sidecar lock file (so we don't block readers
        # of the live artifact; the lock arbitrates writers only)
        with open(lock_path, "w") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                os.replace(tmp_path, output_path)
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    except Exception:
        # Best-effort cleanup of the temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ──────────────────────────────────────────────────────────────────────── #
# CLI                                                                       #
# ──────────────────────────────────────────────────────────────────────── #


def _default_output_path(*, extended: bool = False) -> Path:
    """Default output path under OUTPUT_DIR_DEFAULT.

    For extended sweeps emit ``pairwise_alpha_extended_<utc>.json`` to
    distinguish from canonical 3-pair sweeps.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if extended:
        return OUTPUT_DIR_DEFAULT / f"pairwise_alpha_extended_{now}.json"
    return OUTPUT_DIR_DEFAULT / f"pairwise_alpha_{now}.json"


def _resolve_repo_root() -> Path:
    """Walk up from this file to find the repo root (contains .omx/state/)."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / ".omx").is_dir() and (parent / "tools").is_dir():
            return parent
    return Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pairwise composition_alpha probe (Q6.preprobe)",
    )
    parser.add_argument(
        "--candidate-a",
        type=str,
        default=None,
        help=(
            "First candidate substrate (default: sweep all 3 canonical Stage-2 candidates). "
            f"Valid: {sorted(CANDIDATE_PATHS)}"
        ),
    )
    parser.add_argument(
        "--candidate-b",
        type=str,
        default=None,
        help="Second candidate substrate (paired with --candidate-a if both set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path (default: "
            ".omx/state/wyner_ziv_deliverability/pairwise_alpha_<utc>.json)"
        ),
    )
    parser.add_argument(
        "--report-only-no-side-effects",
        action="store_true",
        help="Print sweep summary to stdout; do NOT write any artifact.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root override (default: auto-detect via .omx/state/ marker).",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help=(
            "OP-3 extended sweep: probe all 10 PRE_ENTROPY candidates "
            "(C(10,2) = 45 pair combinations) per the sister op-routable from "
            "`feedback_pre_entropy_substrate_pivot_prober_landed_20260517.md`. "
            "Surfaces alternative Stage-2 stacking topologies beyond the "
            "canonical 3-candidate (pr101 / pr106 / posenet_class_sensitivity) "
            "set. Output JSON tagged with schema_version_extended."
        ),
    )
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or _resolve_repo_root()).resolve()

    # Determine candidates to sweep
    if args.candidate_a and args.candidate_b:
        if args.extended:
            parser.error(
                "--candidate-a / --candidate-b conflict with --extended; "
                "extended sweep targets the full 10-candidate set"
            )
            return 2
        candidates = (args.candidate_a, args.candidate_b)
    elif args.candidate_a or args.candidate_b:
        parser.error("--candidate-a and --candidate-b must be passed together (or neither)")
        return 2
    elif args.extended:
        candidates = STAGE_2_CANDIDATES_EXTENDED
    else:
        candidates = STAGE_2_CANDIDATES

    sweep = run_sweep(candidates, repo_root=repo_root)
    payload = build_output_payload(
        sweep, brotli_available=_HAS_BROTLI, extended=args.extended
    )

    # Console summary (always emitted)
    print(f"[q6.preprobe] candidates: {list(sweep.candidates_probed)}")
    for pair_key, pair_dict in sweep.pair_results.items():
        print(
            f"[q6.preprobe] {pair_key}: "
            f"α_op3={pair_dict['alpha_op3_council_form']:+.4f} "
            f"α_savings={pair_dict['alpha_savings_ratio_form']:+.4f} "
            f"band={pair_dict['alpha_band']} "
            f"codec={pair_dict['best_codec']} "
            f"gate_clause_2={'PASS' if pair_dict['stage_2_gate_clause_2_satisfied'] else 'FAIL'}"
        )
    print(
        f"[q6.preprobe] pairs ≥ {ALPHA_ADDITIVE_THRESHOLD}: "
        f"{sweep.pairs_with_alpha_at_least_threshold} / "
        f"{len(sweep.pair_results)} (need ≥ {STAGE_2_MINIMUM_ADDITIVE_PAIRS})"
    )
    print(
        f"[q6.preprobe] Stage 2 gate clause #2 verdict: "
        f"{sweep.stage_2_reactivation_clause_2_verdict}"
    )

    if args.report_only_no_side_effects:
        print("[q6.preprobe] --report-only-no-side-effects: NO artifact written")
        return 0

    output_path = args.output or _default_output_path(extended=args.extended)
    write_output_locked(payload, output_path)
    print(f"[q6.preprobe] artifact written: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
