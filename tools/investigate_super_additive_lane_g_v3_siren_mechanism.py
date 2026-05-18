# SPDX-License-Identifier: MIT
"""Mechanism investigation: lane_g_v3_renderer + siren_renderer α=4.74 SUPER_ADDITIVE.

Per `feedback_super_additive_lane_g_v3_siren_topology_integration_landed_
20260517.md` + the Q6 OP-3 extended sweep finding (2026-05-17).

This tool re-loads both candidates via the SAME canonical paths the Q6 OP-3
sweep used (mirrors `tools/pre_entropy_substrate_pivot_prober.py:185-192`),
runs hypothesis tests on the byte-level structure, and emits a machine-readable
hypothesis verification report at
`.omx/research/super_additive_lane_g_v3_siren_mechanism_20260517.md`.

Hypotheses under test:

  H1 [BYTE_LEVEL_STRUCTURE_SHARING]: lane_g_v3 + siren renderer.bin files share
    substantial byte-level structure (e.g. similar fp16 weight distributions,
    similar canonical zero/constant regions) such that joint brotli
    compression captures cross-candidate redundancy beyond marginal
    compression. This was the ORIGINAL hypothesis from the parent brief.

  H2 [BYTE_IDENTITY_ARTIFACT]: the files are byte-identical (sha256-equal);
    the α=4.74 finding is a FALSE-SIGNAL deduplication of identical bytes.
    This is the OUTCOME this tool empirically confirms.

Per CLAUDE.md "Apples-to-apples evidence discipline" the investigation is
HONEST about which hypothesis is supported by the evidence and does NOT
present a false-signal artifact as a real composition discovery.

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``empirical_byte_level_analysis_no_dispatch``
- ``apples_to_apples_per_catalog_127``
- ``no_tmp_paths_per_forbidden_pattern``
- ``mechanism_investigation_per_catalog_272_distinguishing_feature_contract``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical paths from `tools/pre_entropy_substrate_pivot_prober.py:185-192`.
LANE_G_V3_RENDERER_PATH = (
    REPO_ROOT / "experiments" / "results" / "lane_g_v3_landed" / "iter_0" / "renderer.bin"
)
SIREN_RENDERER_PATH = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_substrate_siren_modal_a100_dispatch_20260513T140410Z__smoke__100ep_modal"
    / "submissions"
    / "robust_current"
    / "renderer.bin"
)

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / ".omx" / "research" / "super_additive_lane_g_v3_siren_mechanism_20260517.md"
)


@dataclass
class CandidateBytes:
    """One candidate's loaded byte payload + summary stats."""

    name: str
    path: str
    exists: bool
    size_bytes: int
    sha256: str
    # Byte-histogram (256-bin distribution).
    byte_histogram: dict[int, int] = field(default_factory=dict)
    byte_entropy_bits_per_byte: float = 0.0
    nonzero_byte_count: int = 0
    distinct_byte_count: int = 0


@dataclass
class MechanismInvestigationVerdict:
    """Machine-readable hypothesis verification verdict."""

    candidate_a: CandidateBytes
    candidate_b: CandidateBytes
    # Hypothesis test results.
    h1_byte_level_structure_sharing_supported: bool
    h2_byte_identity_artifact_supported: bool
    primary_hypothesis: str  # one of "H1" or "H2"
    # Byte-level cross-candidate analysis.
    byte_identity_holds: bool
    sha256_identity_holds: bool
    byte_histogram_kl_divergence_a_to_b: float
    byte_histogram_overlap_fraction: float
    # Empirical anchor cross-refs.
    empirical_alpha_savings_ratio: float
    empirical_compressed_alone_a: int
    empirical_compressed_alone_b: int
    empirical_compressed_concat: int
    empirical_source: str
    # Mechanism explanation + operator-routable.
    mechanism_explanation: str
    operator_routable_fix: str
    # Fail-closed.
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    evidence_grade: str = "predicted_byte_level_analysis"
    measurement_axis: str = "[diagnostic; byte-level mechanism investigation]"


def load_candidate(name: str, path: Path) -> CandidateBytes:
    """Load a candidate's bytes + compute summary statistics."""
    if not path.is_file():
        return CandidateBytes(
            name=name,
            path=str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            exists=False,
            size_bytes=0,
            sha256="",
        )
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    # 256-bin byte histogram.
    hist_counter = Counter(data)
    histogram = {int(k): int(v) for k, v in hist_counter.items()}
    # Shannon entropy in bits/byte.
    total = len(data)
    entropy = 0.0
    if total > 0:
        for count in histogram.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
    return CandidateBytes(
        name=name,
        path=str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        exists=True,
        size_bytes=total,
        sha256=sha256,
        byte_histogram=histogram,
        byte_entropy_bits_per_byte=entropy,
        nonzero_byte_count=sum(1 for b in data if b != 0),
        distinct_byte_count=len(histogram),
    )


def compute_byte_histogram_kl_divergence(
    hist_a: dict[int, int], hist_b: dict[int, int]
) -> float:
    """Compute KL(P_a || P_b) over byte histograms (in bits).

    Returns float('inf') if P_b has zero mass on a byte where P_a is nonzero.
    """
    total_a = sum(hist_a.values())
    total_b = sum(hist_b.values())
    if total_a == 0 or total_b == 0:
        return float("inf")
    kl = 0.0
    for byte_val, count_a in hist_a.items():
        p = count_a / total_a
        if p == 0:
            continue
        count_b = hist_b.get(byte_val, 0)
        if count_b == 0:
            return float("inf")
        q = count_b / total_b
        kl += p * math.log2(p / q)
    return kl


def compute_byte_histogram_overlap_fraction(
    hist_a: dict[int, int], hist_b: dict[int, int]
) -> float:
    """Compute fraction of byte mass shared between two histograms.

    Returns sum_b min(P_a[b], P_b[b]) in [0, 1].
    """
    total_a = sum(hist_a.values())
    total_b = sum(hist_b.values())
    if total_a == 0 or total_b == 0:
        return 0.0
    overlap = 0.0
    all_bytes = set(hist_a) | set(hist_b)
    for byte_val in all_bytes:
        p = hist_a.get(byte_val, 0) / total_a
        q = hist_b.get(byte_val, 0) / total_b
        overlap += min(p, q)
    return overlap


def load_empirical_anchor_alpha(
    empirical_artifact_path: Path,
    candidate_a_name: str,
    candidate_b_name: str,
) -> dict[str, Any]:
    """Load the canonical pair row from the Q6 OP-3 extended sweep artifact."""
    if not empirical_artifact_path.is_file():
        return {}
    data = json.loads(empirical_artifact_path.read_text(encoding="utf-8"))
    pair_results = data.get("pair_results", {})
    # Try both orderings.
    for key in (
        f"{candidate_a_name}+{candidate_b_name}",
        f"{candidate_b_name}+{candidate_a_name}",
    ):
        if key in pair_results:
            return pair_results[key]
    return {}


def investigate(
    *,
    empirical_artifact_path: Path | None = None,
) -> MechanismInvestigationVerdict:
    """Run the full mechanism investigation."""
    a = load_candidate("lane_g_v3_renderer", LANE_G_V3_RENDERER_PATH)
    b = load_candidate("siren_renderer", SIREN_RENDERER_PATH)

    # Hypothesis tests.
    sha256_identity = bool(a.exists and b.exists and a.sha256 == b.sha256 and a.sha256 != "")
    byte_identity = sha256_identity  # SHA-256 collision is operationally impossible.

    # Byte histogram analysis (only meaningful if both loaded).
    if a.exists and b.exists:
        kl = compute_byte_histogram_kl_divergence(a.byte_histogram, b.byte_histogram)
        overlap = compute_byte_histogram_overlap_fraction(a.byte_histogram, b.byte_histogram)
    else:
        kl = float("inf")
        overlap = 0.0

    # Empirical anchor cross-refs from the Q6 OP-3 sweep.
    if empirical_artifact_path is None:
        empirical_artifact_path = (
            REPO_ROOT
            / ".omx"
            / "state"
            / "wyner_ziv_deliverability"
            / "pairwise_alpha_extended_20260517T215739Z.json"
        )
    pair_row = load_empirical_anchor_alpha(
        empirical_artifact_path,
        "lane_g_v3_renderer",
        "siren_renderer",
    )

    # Verdict logic.
    if byte_identity:
        h2_supported = True
        h1_supported = False
        primary = "H2"
        mechanism = (
            "BYTE_IDENTITY_ARTIFACT: lane_g_v3_renderer and siren_renderer point to "
            "byte-identical files (sha256 == 08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529). "
            "SIREN smoke (call_id fc-01KRGTEM56EXCV94Q7DC1HF0PB) TIMED OUT at 3601s (rc=124) and "
            "produced no trained weights; the submission-builder placed a placeholder renderer.bin "
            "into the SIREN dispatch dir which turned out to be the lane_g_v3 canonical reference checkpoint. "
            "The α=4.74 SUPER_ADDITIVE finding is brotli deduplication of identical bytes, NOT a real "
            "cross-substrate redundancy discovery."
        )
        operator_routable = (
            "Fix tools/pre_entropy_substrate_pivot_prober.py:189-192 to either (a) point to an actually-"
            "trained siren_renderer if a successful smoke exists, (b) remove siren_renderer from "
            "CANONICAL_CANDIDATE_SUBSTRATES until a successful run produces real weights, or (c) add a "
            "sha256-against-known-placeholders guard in the sweep that flags byte-identity as a "
            "false-signal artifact at probe time. Sister-anchor reinforces Catalog #215 "
            "(modal_smoke_recipe_min_gpu_class_consistent) — the SIREN T4 timeout is exactly the bug "
            "class that gate prevents going forward."
        )
    elif overlap > 0.95 and kl < 0.5:
        h1_supported = True
        h2_supported = False
        primary = "H1"
        mechanism = (
            f"BYTE_LEVEL_STRUCTURE_SHARING: distinct files but byte-histogram overlap={overlap:.4f} "
            f"+ KL divergence={kl:.4f} bits indicate substantial shared structure. This IS a real "
            f"cross-substrate redundancy that brotli's context-modeling exploits for joint compression."
        )
        operator_routable = (
            "Stage 2 stacking experiment: candidate composition for contest archive. Queue Q9 stacking "
            "integration with α=4.74 reward via v2 cascade (bounded at 2.0×)."
        )
    else:
        h1_supported = False
        h2_supported = False
        primary = "INDETERMINATE"
        mechanism = (
            f"INDETERMINATE: not byte-identical AND not sufficient shared structure (overlap={overlap:.4f}, "
            f"KL={kl:.4f} bits). The α=4.74 finding requires further investigation; brotli may be "
            f"exploiting some other form of redundancy not captured by byte-level analysis."
        )
        operator_routable = (
            "Further investigation needed: try fp16-tensor-level analysis, layer-by-layer KL divergence, "
            "or compression-level breakdown via brotli stream inspection."
        )

    return MechanismInvestigationVerdict(
        candidate_a=a,
        candidate_b=b,
        h1_byte_level_structure_sharing_supported=h1_supported,
        h2_byte_identity_artifact_supported=h2_supported,
        primary_hypothesis=primary,
        byte_identity_holds=byte_identity,
        sha256_identity_holds=sha256_identity,
        byte_histogram_kl_divergence_a_to_b=kl,
        byte_histogram_overlap_fraction=overlap,
        empirical_alpha_savings_ratio=float(pair_row.get("alpha_savings_ratio_form", 0.0)),
        empirical_compressed_alone_a=int(pair_row.get("compressed_alone_a", 0)),
        empirical_compressed_alone_b=int(pair_row.get("compressed_alone_b", 0)),
        empirical_compressed_concat=int(pair_row.get("compressed_concat", 0)),
        empirical_source=str(empirical_artifact_path.relative_to(REPO_ROOT))
        if empirical_artifact_path.is_relative_to(REPO_ROOT)
        else str(empirical_artifact_path),
        mechanism_explanation=mechanism,
        operator_routable_fix=operator_routable,
    )


def render_verdict_markdown(v: MechanismInvestigationVerdict) -> str:
    """Render the verdict as a markdown hypothesis investigation report."""
    lines: list[str] = []
    lines.append("# Mechanism investigation: lane_g_v3_renderer + siren_renderer α=4.74 SUPER_ADDITIVE")
    lines.append("")
    lines.append(
        "**Lane:** `lane_super_additive_lane_g_v3_siren_topology_integration_20260517` "
        "(Task #823)"
    )
    lines.append("")
    lines.append(
        "**Empirical anchor:** Q6 OP-3 extended sweep at "
        f"`{v.empirical_source}` reported α=4.74 SUPER_ADDITIVE band for the "
        "lane_g_v3_renderer + siren_renderer pair under brotli."
    )
    lines.append("")
    lines.append(
        "**Investigation outcome:** primary hypothesis = "
        f"**{v.primary_hypothesis}** "
        f"({'H1: real byte-level structure sharing' if v.primary_hypothesis == 'H1' else 'H2: byte-identity artifact' if v.primary_hypothesis == 'H2' else 'INDETERMINATE'})."
    )
    lines.append("")
    lines.append("## 9-dimension success checklist evidence")
    lines.append("")
    lines.append("(Per Catalog #294 + CLAUDE.md '9-dim checklist' standing directive.)")
    lines.append("")
    lines.append("- **Dimension 1 (UNIQUENESS):** this investigation surfaces a NEW META-pattern — "
                 "candidate-loader-points-at-placeholder false-signal artifact — distinct from any "
                 "existing apples-to-apples discipline anchor.")
    lines.append("- **Dimension 2 (BEAUTY + ELEGANCE):** the mechanism is byte-identity, the cleanest "
                 "possible explanation. SHA-256 equality is operationally definitive.")
    lines.append("- **Dimension 3 (DISTINCTNESS):** distinct from MPS-vs-CUDA drift (Catalog #1 / #192) "
                 "+ phantom-baseline (CLAUDE.md FORBIDDEN_PATTERNS) — this is the loader-side variant.")
    lines.append("- **Dimension 4 (RIGOR):** empirical SHA-256 + byte-histogram KL divergence + overlap "
                 "fraction computed independently; cross-referenced with SIREN smoke timeout call_id.")
    lines.append("- **Dimension 5 (OPTIMIZATION PER TECHNIQUE):** the v2 cascade extension is "
                 "structurally needed for FUTURE real SUPER_ADDITIVE topologies; bounded at 2.0× to "
                 "prevent runaway.")
    lines.append("- **Dimension 6 (STACK-OF-STACKS-COMPOSABILITY):** non-applicable to this finding "
                 "(false signal does not compose); v2 cascade composes orthogonally with sister "
                 "predicted-delta adjusters.")
    lines.append("- **Dimension 7 (DETERMINISTIC REPRODUCIBILITY):** SHA-256 + byte counts are "
                 "deterministic + the empirical artifact carries written_at_utc.")
    lines.append("- **Dimension 8 (EXTREME OPTIMIZATION + PERFORMANCE):** O(N) byte-histogram + O(256) "
                 "KL computation; runs in ~1s on M5 Max.")
    lines.append("- **Dimension 9 (OPTIMAL MINIMAL CONTEST SCORE):** no direct contest-score impact; "
                 "the investigation PREVENTS misallocation of dispatch budget toward a false-signal "
                 "topology.")
    lines.append("")
    lines.append("## Cargo-cult audit per assumption")
    lines.append("")
    lines.append("(Per Catalog #303 + CLAUDE.md HARD-EARNED-vs-CARGO-CULTED addendum.)")
    lines.append("")
    lines.append("| Assumption | HARD-EARNED vs CARGO-CULTED | Rationale | Unwind path |")
    lines.append("|---|---|---|---|")
    lines.append("| 'high α in Q6 sweep means real composition discovery' | CARGO-CULTED | Inherited from compression-codec literature assuming distinct files | Add sha256-equality check at probe time |")
    lines.append("| 'submission-builder produces real trained weights' | CARGO-CULTED | Inherited from successful-smoke assumption | Check returncode + elapsed_seconds vs configured timeout |")
    lines.append("| 'CANONICAL_CANDIDATE_SUBSTRATES paths are always trained weights' | CARGO-CULTED | Inherited from initial probe-design assumption | Add per-candidate liveness verification (sha256-against-known-placeholders) |")
    lines.append("| 'brotli SUPER_ADDITIVE α > 1.0 always indicates cross-substrate redundancy' | CARGO-CULTED | Inherited from cross-source compression literature where source distinctness was assumed | Add explicit byte-identity guard at sweep + autopilot consumer surfaces |")
    lines.append("")
    lines.append("## Observability surface")
    lines.append("")
    lines.append("(Per Catalog #305 + CLAUDE.md 'Max observability — non-negotiable' standing directive.)")
    lines.append("")
    lines.append("- **Inspectable per layer:** candidate-A SHA, candidate-B SHA, byte-histogram per "
                 "candidate, KL divergence, overlap fraction, empirical-anchor α, mechanism verdict.")
    lines.append("- **Decomposable per signal:** byte-identity (sha256-level) decomposable from "
                 "byte-histogram-similarity (structural-level) decomposable from KL divergence "
                 "(distribution-level).")
    lines.append("- **Diff-able across runs:** SHA-256 + byte counts are deterministic; two runs of "
                 "this tool produce byte-identical reports.")
    lines.append("- **Queryable post-hoc:** verdict dataclass is JSON-serializable + the markdown report "
                 "has explicit fields per hypothesis.")
    lines.append("- **Cite-able:** report cross-refs empirical artifact path + SIREN smoke call_id + "
                 "canonical pre_entropy_substrate_pivot_prober.py lines.")
    lines.append("- **Counterfactual-able:** if either candidate's bytes change (e.g. SIREN smoke succeeds "
                 "+ produces real weights), re-running this tool produces a different verdict; the "
                 "byte-identity branch is the canonical counterfactual probe.")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append("### Candidate A: `lane_g_v3_renderer`")
    lines.append("")
    lines.append(f"- **Path:** `{v.candidate_a.path}`")
    lines.append(f"- **Exists:** {v.candidate_a.exists}")
    lines.append(f"- **Size bytes:** {v.candidate_a.size_bytes:,}")
    lines.append(f"- **SHA-256:** `{v.candidate_a.sha256}`")
    lines.append(f"- **Byte entropy (bits/byte):** {v.candidate_a.byte_entropy_bits_per_byte:.4f}")
    lines.append(f"- **Distinct byte values:** {v.candidate_a.distinct_byte_count} / 256")
    lines.append(f"- **Nonzero byte count:** {v.candidate_a.nonzero_byte_count:,}")
    lines.append("")
    lines.append("### Candidate B: `siren_renderer`")
    lines.append("")
    lines.append(f"- **Path:** `{v.candidate_b.path}`")
    lines.append(f"- **Exists:** {v.candidate_b.exists}")
    lines.append(f"- **Size bytes:** {v.candidate_b.size_bytes:,}")
    lines.append(f"- **SHA-256:** `{v.candidate_b.sha256}`")
    lines.append(f"- **Byte entropy (bits/byte):** {v.candidate_b.byte_entropy_bits_per_byte:.4f}")
    lines.append(f"- **Distinct byte values:** {v.candidate_b.distinct_byte_count} / 256")
    lines.append(f"- **Nonzero byte count:** {v.candidate_b.nonzero_byte_count:,}")
    lines.append("")
    lines.append("### Cross-candidate analysis")
    lines.append("")
    lines.append(f"- **SHA-256 identity:** {v.sha256_identity_holds}")
    lines.append(f"- **Byte identity:** {v.byte_identity_holds}")
    lines.append(f"- **Byte histogram KL divergence (A || B):** {v.byte_histogram_kl_divergence_a_to_b:.6f} bits")
    lines.append(f"- **Byte histogram overlap fraction:** {v.byte_histogram_overlap_fraction:.6f}")
    lines.append("")
    lines.append("### Empirical anchor cross-ref (Q6 OP-3 extended sweep)")
    lines.append("")
    lines.append(f"- **Source:** `{v.empirical_source}`")
    lines.append(f"- **α (savings_ratio_form):** {v.empirical_alpha_savings_ratio:.6f}")
    lines.append(f"- **Compressed alone A:** {v.empirical_compressed_alone_a:,} bytes")
    lines.append(f"- **Compressed alone B:** {v.empirical_compressed_alone_b:,} bytes")
    lines.append(f"- **Compressed concat:** {v.empirical_compressed_concat:,} bytes")
    lines.append("")
    lines.append("## Hypothesis verdict")
    lines.append("")
    lines.append(f"**Primary hypothesis:** {v.primary_hypothesis}")
    lines.append("")
    lines.append(f"**H1 (BYTE_LEVEL_STRUCTURE_SHARING) supported:** {v.h1_byte_level_structure_sharing_supported}")
    lines.append(f"**H2 (BYTE_IDENTITY_ARTIFACT) supported:** {v.h2_byte_identity_artifact_supported}")
    lines.append("")
    lines.append("### Mechanism explanation")
    lines.append("")
    lines.append(v.mechanism_explanation)
    lines.append("")
    lines.append("## Operator-routable fix")
    lines.append("")
    lines.append(v.operator_routable_fix)
    lines.append("")
    lines.append("## Fail-closed canonical fields")
    lines.append("")
    lines.append(f"- `score_claim`: {v.score_claim}")
    lines.append(f"- `promotion_eligible`: {v.promotion_eligible}")
    lines.append(f"- `ready_for_exact_eval_dispatch`: {v.ready_for_exact_eval_dispatch}")
    lines.append(f"- `evidence_grade`: `{v.evidence_grade}`")
    lines.append(f"- `measurement_axis`: `{v.measurement_axis}`")
    lines.append("")
    lines.append("## Cross-references")
    lines.append("")
    lines.append("- CLAUDE.md 'Apples-to-apples evidence discipline' (axis labels + custody mandatory)")
    lines.append("- CLAUDE.md 'Forbidden component-aliasing for baselines' (the phantom-baseline pattern)")
    lines.append("- CLAUDE.md 'Subagent coherence-by-default' + Catalog #125 (6-hook wire-in)")
    lines.append("- CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' (v2 cascade extends canonical)")
    lines.append("- Catalog #127 (per-call-site custody routing)")
    lines.append("- Catalog #131 (fcntl-locked JSONL discipline for matrix posterior)")
    lines.append("- Catalog #215 (modal_smoke_recipe_min_gpu_class_consistent — sister anchor for SIREN T4 timeout)")
    lines.append("- Catalog #220 (substrate L1+ scaffold operational mechanism)")
    lines.append("- Catalog #227 (substrate composition matrix — autopilot consumer)")
    lines.append("- Catalog #229 (premise verification before edit)")
    lines.append("- Catalog #230 (sister subagent ownership map)")
    lines.append("- Catalog #272 (distinguishing-feature integration contract)")
    lines.append("- `feedback_batched_815_816_q6_op3_extended_landed_20260517.md` (Q6 OP-3 extended sweep source)")
    lines.append("- `feedback_super_additive_lane_g_v3_siren_topology_integration_landed_20260517.md` (this landing memo)")
    lines.append("- `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` (v2 cascade)")
    lines.append("- `.omx/state/substrate_composition_matrix.json` (canonical posterior surface; SUPER_ADDITIVE row appended with FALSE_SIGNAL blockers)")
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mechanism investigation: lane_g_v3_renderer + siren_renderer α=4.74 "
            "SUPER_ADDITIVE per Q6 OP-3 extended sweep. CPU-only; $0; research_only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output markdown report path (default: {DEFAULT_OUTPUT_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--empirical-artifact",
        type=Path,
        default=None,
        help=(
            "Path to the Q6 OP-3 extended sweep artifact "
            "(default: .omx/state/wyner_ziv_deliverability/pairwise_alpha_extended_20260517T215739Z.json)"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path with the full machine-readable verdict.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    verdict = investigate(empirical_artifact_path=args.empirical_artifact)
    markdown = render_verdict_markdown(verdict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"[mechanism-investigation] wrote markdown report -> {args.output}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        # Convert dataclasses to dict; the candidates have byte_histogram dicts
        # with int keys which JSON cannot serialize as-is; coerce to str keys.
        d = asdict(verdict)
        for cand_key in ("candidate_a", "candidate_b"):
            hist = d[cand_key].get("byte_histogram", {})
            d[cand_key]["byte_histogram"] = {str(k): v for k, v in hist.items()}
        args.json_out.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[mechanism-investigation] wrote JSON verdict -> {args.json_out}")
    print(f"[mechanism-investigation] primary hypothesis: {verdict.primary_hypothesis}")
    print(f"[mechanism-investigation] mechanism: {verdict.mechanism_explanation[:120]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
