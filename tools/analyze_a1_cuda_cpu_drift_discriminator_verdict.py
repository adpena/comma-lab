#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
r"""Compute the CUDA-CPU drift mechanism verdict from the 4 A1 discriminator
variants' paired CPU+CUDA eval results.

Companion to ``tools/build_a1_cuda_cpu_drift_discriminator_variants.py``. Once
both CPU and CUDA exact evals have returned for all 4 variants, this tool:

  1. Loads each variant's ``discriminator_manifest.json`` (build artifact).
  2. Loads each variant's ``contest_auth_eval.adjudicated.json`` for CPU
     (from ``tools/dispatch_cpu_eval_via_github_actions.py``) and the CUDA
     eval JSON (from the canonical CUDA harvest path).
  3. Computes per-variant drift:
       Δ_score   = cuda_score - cpu_score
       R_pose    = avg_posenet_dist_cuda / avg_posenet_dist_cpu
       R_seg     = avg_segnet_dist_cuda / avg_segnet_dist_cpu
  4. Compares the 3 isolated variants against ``v_baseline``:
       primary_mechanism: an isolation that drops R_pose from baseline ~5.04
                          to < 2.0
       multi_mechanism:   2+ isolations both narrow drift (mechanisms partially
                          independent / multiplicative)
       fourth_mechanism:  no isolation narrows drift; surface as operator decision
                          (NEGATIVE-RESULT-IS-EVIDENCE per
                          ``forbidden_premature_kill_without_research_exhaustion``)
  5. Emits a verdict table (markdown + JSON) and a registry-update spec:
       - if LOADER dominant: add ``loader_drift_correction`` per class
       - if CONV dominant:  add ``conv_kernel_determinism_required`` per class
       - if HYDRA dominant: add ``head_quantize_post_inference_dtype`` per class

The verdict is conservative by design — every threshold is the same across
variants, every comparison uses the SAME baseline, and ambiguous results are
reported as "INCONCLUSIVE" rather than forced into a verdict.

Per CLAUDE.md:
  - Score tags are validated: only ``[contest-CPU]`` (Linux x86_64 GHA) and
    ``[contest-CUDA]`` (T4/4090/A100) are accepted; macOS-CPU and MPS are refused.
  - Per ``forbidden_premature_kill_without_research_exhaustion``: the
    ``fourth_mechanism`` verdict surfaces an operator-decision item, NOT a
    "discriminator KILLED" verdict.
  - Per "Beauty, simplicity, and developer experience": output schema is typed,
    documented, and stable enough for the registry-update writer to consume
    without prose interpretation.

Output:

  ``<analysis-output-dir>/discriminator_verdict.json``  — typed verdict
  ``<analysis-output-dir>/discriminator_verdict.md``    — human-readable table

Exit codes:
  0 — verdict computed (may be INCONCLUSIVE; that's fine)
  2 — input validation error (missing manifest, missing eval, tag mismatch)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical baseline R_pose / R_seg per the per-architecture-class registry
# (HNeRV cluster). Used as a fallback "expected" value in case the discriminator
# v_baseline measurement is missing (it shouldn't be).
CANONICAL_R_POSE_HNERV = 5.04
CANONICAL_R_SEG_HNERV = 1.17
CANONICAL_SCORE_GAP_HNERV = 0.0330

# A drop in R_pose from baseline ~5.04 to BELOW this threshold is treated as
# "this isolation reduced drift dramatically" → primary mechanism.
PRIMARY_MECHANISM_R_POSE_THRESHOLD = 2.0

# A drop in R_pose by AT LEAST this fraction (e.g. 0.30 → 30% reduction) is
# treated as "this isolation reduced drift significantly" → contributing
# mechanism (used for multi-mechanism detection).
CONTRIBUTING_MECHANISM_R_POSE_FRACTION = 0.30

# Authoritative tag substrings (Linux x86_64 / NVIDIA only).
ACCEPTED_CPU_TAGS = ("[contest-CPU]", "contest-CPU-1to1", "contest-cpu-1to1")
ACCEPTED_CUDA_TAGS = ("[contest-CUDA]", "contest-CUDA-1to1", "contest-cuda-1to1")
# Refused-tag check runs FIRST (before accepted-tag check) so that
# substrings like "[contest-CPU advisory]" or "[macOS-CPU advisory only]"
# get rejected even though "contest-CPU" might still be a substring.
REFUSED_TAGS = (
    "macOS-CPU",
    "macos-cpu",
    "MPS-PROXY",
    "[macOS-CPU advisory only]",
    "[contest-CPU advisory]",
    "[contest-CUDA advisory]",
    "[advisory only]",
    "[macOS-CPU calibrated]",
    "MPS-research-signal",
)


class VariantPair:
    """Holds the (manifest, cpu_eval, cuda_eval) triple for one variant.

    Plain class instead of ``@dataclass`` because the importlib-loaded
    invocation pattern in tests confuses dataclass module resolution.
    """

    __slots__ = (
        "variant_id",
        "mechanism_hypothesis",
        "archive_sha256",
        "cpu_score",
        "cuda_score",
        "cpu_pose",
        "cuda_pose",
        "cpu_seg",
        "cuda_seg",
        "cpu_evidence_grade",
        "cuda_evidence_grade",
        "cpu_evidence_path",
        "cuda_evidence_path",
    )

    def __init__(
        self,
        *,
        variant_id: str,
        mechanism_hypothesis: str,
        archive_sha256: str,
        cpu_score: float,
        cuda_score: float,
        cpu_pose: float,
        cuda_pose: float,
        cpu_seg: float,
        cuda_seg: float,
        cpu_evidence_grade: str,
        cuda_evidence_grade: str,
        cpu_evidence_path: str,
        cuda_evidence_path: str,
    ) -> None:
        self.variant_id = variant_id
        self.mechanism_hypothesis = mechanism_hypothesis
        self.archive_sha256 = archive_sha256
        self.cpu_score = cpu_score
        self.cuda_score = cuda_score
        self.cpu_pose = cpu_pose
        self.cuda_pose = cuda_pose
        self.cpu_seg = cpu_seg
        self.cuda_seg = cuda_seg
        self.cpu_evidence_grade = cpu_evidence_grade
        self.cuda_evidence_grade = cuda_evidence_grade
        self.cpu_evidence_path = cpu_evidence_path
        self.cuda_evidence_path = cuda_evidence_path

    def score_gap(self) -> float:
        return self.cuda_score - self.cpu_score

    def r_pose(self) -> float:
        # Guard against /0 — pose can be ~0 at frontier but never exactly 0.
        return self.cuda_pose / max(self.cpu_pose, 1e-12)

    def r_seg(self) -> float:
        return self.cuda_seg / max(self.cpu_seg, 1e-12)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _validate_eval_tag(
    record: dict[str, Any], path: Path, *, axis: str
) -> None:
    """Refuse records carrying refused (non-1:1) hardware tags."""
    grade = str(record.get("evidence_grade", ""))
    tag = str(record.get("lane_tag", ""))
    blob = grade + " " + tag
    for refused in REFUSED_TAGS:
        if refused in blob:
            raise ValueError(
                f"refused eval record at {path}: carries non-1:1 tag "
                f"{refused!r} (axis={axis}); per CLAUDE.md dual-eval mandate "
                f"only Linux x86_64 [contest-CPU] and CUDA-substrate "
                f"[contest-CUDA] are authoritative"
            )
    accepted = ACCEPTED_CPU_TAGS if axis == "cpu" else ACCEPTED_CUDA_TAGS
    if not any(a in blob for a in accepted):
        raise ValueError(
            f"refused eval record at {path}: missing authoritative {axis} tag "
            f"(grade={grade!r}, lane_tag={tag!r}); expected one of {accepted}"
        )


def _read_eval_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"eval record not found at {path}; run the GHA/CUDA dispatcher first"
        )
    return json.loads(path.read_text())


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"discriminator manifest not found at {path}; build the variants first"
        )
    return json.loads(path.read_text())


def load_variant_pair(
    variant_dir: Path,
    cpu_eval_path: Path,
    cuda_eval_path: Path,
) -> VariantPair:
    """Load a discriminator variant's manifest + paired CPU/CUDA eval records.

    Returns a ``VariantPair`` with score components extracted; raises
    ``ValueError`` if any tag is refused.
    """
    manifest_path_p = variant_dir / "discriminator_manifest.json"
    manifest = _read_manifest(manifest_path_p)
    cpu = _read_eval_record(cpu_eval_path)
    cuda = _read_eval_record(cuda_eval_path)

    _validate_eval_tag(cpu, cpu_eval_path, axis="cpu")
    _validate_eval_tag(cuda, cuda_eval_path, axis="cuda")

    archive_sha = str(manifest["archive_sha256"])
    if str(cpu.get("archive_sha256", "")).lower() not in ("", archive_sha.lower()):
        raise ValueError(
            f"CPU eval archive_sha256 mismatch for {manifest_path_p}: "
            f"manifest={archive_sha} cpu_record={cpu['archive_sha256']}"
        )
    if str(cuda.get("archive_sha256", "")).lower() not in ("", archive_sha.lower()):
        raise ValueError(
            f"CUDA eval archive_sha256 mismatch for {manifest_path_p}: "
            f"manifest={archive_sha} cuda_record={cuda['archive_sha256']}"
        )

    return VariantPair(
        variant_id=str(manifest["variant_id"]),
        mechanism_hypothesis=str(manifest["mechanism_hypothesis"]),
        archive_sha256=archive_sha,
        cpu_score=float(cpu["canonical_score"]),
        cuda_score=float(cuda["canonical_score"]),
        cpu_pose=float(cpu["avg_posenet_dist"]),
        cuda_pose=float(cuda["avg_posenet_dist"]),
        cpu_seg=float(cpu["avg_segnet_dist"]),
        cuda_seg=float(cuda["avg_segnet_dist"]),
        cpu_evidence_grade=str(cpu.get("evidence_grade", "")),
        cuda_evidence_grade=str(cuda.get("evidence_grade", "")),
        cpu_evidence_path=str(cpu_eval_path),
        cuda_evidence_path=str(cuda_eval_path),
    )


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def per_variant_row(pair: VariantPair) -> dict[str, Any]:
    return {
        "variant_id": pair.variant_id,
        "mechanism_hypothesis": pair.mechanism_hypothesis,
        "cuda_score": pair.cuda_score,
        "cpu_score": pair.cpu_score,
        "score_gap": pair.score_gap(),
        "r_pose": pair.r_pose(),
        "r_seg": pair.r_seg(),
        "cpu_evidence_grade": pair.cpu_evidence_grade,
        "cuda_evidence_grade": pair.cuda_evidence_grade,
        "cpu_evidence_path": pair.cpu_evidence_path,
        "cuda_evidence_path": pair.cuda_evidence_path,
    }


def compute_verdict(pairs: list[VariantPair]) -> dict[str, Any]:
    by_id = {p.variant_id: p for p in pairs}
    if "v_baseline" not in by_id:
        return {
            "verdict": "INCONCLUSIVE_NO_BASELINE",
            "verdict_rationale": (
                "v_baseline missing from inputs; cannot compute per-variant deltas"
            ),
            "rows": [per_variant_row(p) for p in pairs],
        }
    baseline = by_id["v_baseline"]
    baseline_r_pose = baseline.r_pose()
    baseline_r_seg = baseline.r_seg()
    baseline_gap = baseline.score_gap()

    isolation_findings: list[dict[str, Any]] = []
    for vid, hypothesis in (
        ("v_loader_isolated", "loader_byte_drift"),
        ("v_conv_isolated", "conv_kernel_accumulation_drift"),
        ("v_hydra_isolated", "hydra_head_numerical_sensitivity"),
    ):
        if vid not in by_id:
            isolation_findings.append(
                {
                    "variant_id": vid,
                    "mechanism_hypothesis": hypothesis,
                    "verdict": "INCONCLUSIVE_VARIANT_MISSING",
                }
            )
            continue
        var = by_id[vid]
        r_pose = var.r_pose()
        r_seg = var.r_seg()
        gap = var.score_gap()
        d_r_pose = baseline_r_pose - r_pose
        d_r_pose_frac = (
            d_r_pose / baseline_r_pose if baseline_r_pose > 0 else 0.0
        )
        gap_reduction_frac = (
            (baseline_gap - gap) / baseline_gap if baseline_gap != 0 else 0.0
        )

        if r_pose < PRIMARY_MECHANISM_R_POSE_THRESHOLD:
            verdict = "PRIMARY_MECHANISM"
        elif d_r_pose_frac >= CONTRIBUTING_MECHANISM_R_POSE_FRACTION:
            verdict = "CONTRIBUTING_MECHANISM"
        else:
            verdict = "MECHANISM_NOT_DOMINANT"

        isolation_findings.append(
            {
                "variant_id": vid,
                "mechanism_hypothesis": hypothesis,
                "r_pose": r_pose,
                "r_seg": r_seg,
                "score_gap": gap,
                "delta_r_pose_vs_baseline": d_r_pose,
                "delta_r_pose_fraction_vs_baseline": d_r_pose_frac,
                "score_gap_reduction_fraction_vs_baseline": gap_reduction_frac,
                "verdict": verdict,
            }
        )

    primaries = [
        f for f in isolation_findings if f.get("verdict") == "PRIMARY_MECHANISM"
    ]
    contributors = [
        f for f in isolation_findings
        if f.get("verdict") == "CONTRIBUTING_MECHANISM"
    ]
    inconclusive_missing = [
        f for f in isolation_findings
        if f.get("verdict") == "INCONCLUSIVE_VARIANT_MISSING"
    ]

    if inconclusive_missing:
        overall = "INCONCLUSIVE_VARIANTS_MISSING"
        rationale = (
            f"{len(inconclusive_missing)} of 3 isolation variant(s) missing — "
            f"cannot conclude. Missing: "
            f"{[f['variant_id'] for f in inconclusive_missing]}"
        )
    elif len(primaries) == 1 and not contributors:
        overall = "PRIMARY_MECHANISM_IDENTIFIED"
        rationale = (
            f"{primaries[0]['mechanism_hypothesis']} reduced R_pose from "
            f"{baseline_r_pose:.2f} to {primaries[0]['r_pose']:.2f} — "
            f"single dominant mechanism."
        )
    elif len(primaries) >= 2:
        overall = "MULTI_MECHANISM_PRIMARY"
        rationale = (
            f"{len(primaries)} isolation variants each cut R_pose below "
            f"{PRIMARY_MECHANISM_R_POSE_THRESHOLD} — mechanisms appear "
            f"multiplicative or each independently sufficient."
        )
    elif primaries and contributors:
        overall = "MULTI_MECHANISM_PRIMARY_PLUS_CONTRIBUTING"
        rationale = (
            f"1 primary + {len(contributors)} contributing isolation(s) — "
            f"mechanisms are partially independent."
        )
    elif contributors:
        overall = "MULTI_MECHANISM_CONTRIBUTING_ONLY"
        rationale = (
            f"{len(contributors)} contributing isolation(s) but none dropped "
            f"R_pose below {PRIMARY_MECHANISM_R_POSE_THRESHOLD} — combination "
            f"of partial mechanisms."
        )
    else:
        overall = "FOURTH_MECHANISM_HYPOTHESIS"
        rationale = (
            "None of the 3 isolated variants narrowed R_pose by >= "
            f"{CONTRIBUTING_MECHANISM_R_POSE_FRACTION:.0%}. A 4th unmodeled "
            "mechanism is implicated. Per "
            "'forbidden_premature_kill_without_research_exhaustion', this "
            "is NEGATIVE-RESULT-IS-EVIDENCE — surface as operator decision; "
            "do NOT kill the discriminator family."
        )

    registry_update_spec: dict[str, Any] = {
        "applies_to_architecture_class": "hnerv_ft_microcodec",
        "fields_to_add_or_update": [],
        "rationale": rationale,
    }
    for f in primaries + contributors:
        hypothesis = f["mechanism_hypothesis"]
        if hypothesis == "loader_byte_drift":
            registry_update_spec["fields_to_add_or_update"].append(
                {
                    "field": "loader_drift_correction",
                    "kind": "scalar",
                    "value_source": (
                        f"observed_r_pose_drop = "
                        f"{f.get('delta_r_pose_vs_baseline', 0.0):.4f}"
                    ),
                    "field_semantics": (
                        "scalar correction subtracted from CUDA pose to estimate "
                        "the loader-byte-drift component of the gap on this "
                        "architecture class"
                    ),
                }
            )
        elif hypothesis == "conv_kernel_accumulation_drift":
            registry_update_spec["fields_to_add_or_update"].append(
                {
                    "field": "conv_kernel_determinism_required",
                    "kind": "bool",
                    "value": True,
                    "field_semantics": (
                        "if True, archive builders should set "
                        "torch.use_deterministic_algorithms(True) and "
                        "torch.backends.cudnn.deterministic=True in inflate.py"
                    ),
                }
            )
        elif hypothesis == "hydra_head_numerical_sensitivity":
            registry_update_spec["fields_to_add_or_update"].append(
                {
                    "field": "head_quantize_post_inference_dtype",
                    "kind": "string",
                    "value": "uint8_round_multiple_of_2",
                    "field_semantics": (
                        "if set, archive builders should pre-quantize inflate "
                        "output to a coarser grid that washes out tiny upstream "
                        "perturbations before they reach the high-condition "
                        "PoseNet Hydra head"
                    ),
                }
            )

    return {
        "verdict": overall,
        "verdict_rationale": rationale,
        "baseline_r_pose": baseline_r_pose,
        "baseline_r_seg": baseline_r_seg,
        "baseline_score_gap": baseline_gap,
        "canonical_r_pose_hnerv": CANONICAL_R_POSE_HNERV,
        "canonical_r_seg_hnerv": CANONICAL_R_SEG_HNERV,
        "canonical_score_gap_hnerv": CANONICAL_SCORE_GAP_HNERV,
        "primary_mechanism_r_pose_threshold": PRIMARY_MECHANISM_R_POSE_THRESHOLD,
        "contributing_mechanism_r_pose_fraction": CONTRIBUTING_MECHANISM_R_POSE_FRACTION,
        "isolation_findings": isolation_findings,
        "registry_update_spec": registry_update_spec,
        "rows": [per_variant_row(p) for p in pairs],
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(verdict: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# A1 CUDA-CPU drift discriminator — verdict\n")
    lines.append(
        f"**Verdict**: `{verdict['verdict']}`  \n"
        f"**Rationale**: {verdict['verdict_rationale']}\n"
    )
    lines.append("## Per-variant table\n")
    lines.append("| variant | hypothesis | CUDA | CPU | Δ | R_seg | R_pose |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in verdict.get("rows", []):
        lines.append(
            f"| `{row['variant_id']}` | `{row['mechanism_hypothesis']}` | "
            f"{row['cuda_score']:.6f} | {row['cpu_score']:.6f} | "
            f"{row['score_gap']:+.6f} | {row['r_seg']:.3f} | "
            f"{row['r_pose']:.3f} |"
        )
    if "baseline_r_pose" in verdict:
        lines.append(
            f"\nBaseline R_pose = {verdict['baseline_r_pose']:.3f} "
            f"(canonical HNeRV-cluster anchor: {verdict['canonical_r_pose_hnerv']:.3f})"
        )
        lines.append(
            f"Baseline R_seg = {verdict['baseline_r_seg']:.3f} "
            f"(canonical: {verdict['canonical_r_seg_hnerv']:.3f})"
        )
        lines.append(
            f"Baseline Δ score = {verdict['baseline_score_gap']:+.6f} "
            f"(canonical: {verdict['canonical_score_gap_hnerv']:+.6f})"
        )
        lines.append("")
    if verdict.get("isolation_findings"):
        lines.append("## Isolation findings\n")
        for f in verdict["isolation_findings"]:
            lines.append(
                f"- **{f['variant_id']}** ({f['mechanism_hypothesis']}): "
                f"`{f.get('verdict')}`"
            )
            for k in (
                "r_pose",
                "delta_r_pose_vs_baseline",
                "delta_r_pose_fraction_vs_baseline",
                "score_gap_reduction_fraction_vs_baseline",
            ):
                if k in f:
                    lines.append(f"    - {k}: `{f[k]:.4f}`")
        lines.append("")
    spec = verdict.get("registry_update_spec", {})
    if spec.get("fields_to_add_or_update"):
        lines.append("## Registry update spec\n")
        lines.append(
            f"Apply to architecture class: "
            f"`{spec['applies_to_architecture_class']}`\n"
        )
        for entry in spec["fields_to_add_or_update"]:
            lines.append(
                f"- `{entry['field']}` ({entry['kind']}): "
                f"{entry.get('value', entry.get('value_source', ''))}"
            )
            lines.append(f"    - {entry['field_semantics']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--variant-dir",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "discriminator-variant directory containing discriminator_manifest.json "
            "(repeatable; one per variant)"
        ),
    )
    p.add_argument(
        "--cpu-eval",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "path to contest_auth_eval.adjudicated.json from GHA CPU dispatch "
            "(repeatable; SAME ORDER as --variant-dir)"
        ),
    )
    p.add_argument(
        "--cuda-eval",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "path to CUDA exact-eval JSON (repeatable; SAME ORDER as --variant-dir)"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write discriminator_verdict.json + .md",
    )
    args = p.parse_args()

    if not (len(args.variant_dir) == len(args.cpu_eval) == len(args.cuda_eval)):
        print(
            "FATAL: --variant-dir / --cpu-eval / --cuda-eval lengths must match "
            f"(got {len(args.variant_dir)} / {len(args.cpu_eval)} / "
            f"{len(args.cuda_eval)})",
            file=sys.stderr,
        )
        return 2

    pairs: list[VariantPair] = []
    for vdir, cpu_path, cuda_path in zip(
        args.variant_dir, args.cpu_eval, args.cuda_eval
    ):
        try:
            pair = load_variant_pair(vdir, cpu_path, cuda_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
        pairs.append(pair)

    verdict = compute_verdict(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "discriminator_verdict.json"
    md_path = args.output_dir / "discriminator_verdict.md"
    verdict_with_meta = {
        "lane_id": "lane_avvideodataset_cuda_path_mechanism_discriminator",
        "schema_version": "discriminator_verdict_v1",
        "computed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        **verdict,
    }
    json_path.write_text(json.dumps(verdict_with_meta, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(verdict_with_meta))
    print(f"[ok] verdict={verdict_with_meta['verdict']}")
    print(f"[ok] {json_path}")
    print(f"[ok] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
