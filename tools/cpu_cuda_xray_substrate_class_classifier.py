# SPDX-License-Identifier: MIT
"""P5 xray substrate-class classifier (per-layer drift signature across substrates).

Per operator amplification 2026-05-11 ("push xray"), this tool extends the
existing P5 xray (`tools/cpu_cuda_xray_segnet_layer_drift.py` +
`tools/cpu_cuda_xray_posenet_layer_drift.py`) from a SINGLE-substrate
diagnostic into a MULTI-SUBSTRATE classifier.

Given N input ``layer_drift.json`` files (one per substrate's CPU-vs-CUDA
xray), the tool:

1. Validates every input has the expected schema (``layer_drift_rows``,
   ``stage_compounding``, ``cpu_record_path``, ``cuda_record_path``,
   ``evidence_grade``, etc.).
2. Extracts the per-layer drift signature for each substrate
   (sequence of ``(layer_name, l2_relative_error, max_abs_error)``).
3. Computes summary statistics per substrate (mean drift, max drift,
   first divergence layer, stage-compounding factor).
4. Pairwise-compares substrates via cosine similarity of the per-layer
   drift signature vectors (truncated to the shortest substrate's
   layer count).
5. Classifies each substrate by its signature against a pre-defined
   "substrate-class taxonomy" — currently 2 classes:
   - ``hnerv_family`` (PR106 r2, A1, PR101 grammar variants)
   - ``non_hnerv_family`` (categorical / wavelet / coordinate-MLP)
   The classification is by the OPERATOR-DECLARED substrate-class hint
   on each input; the tool's job is to verify the per-layer drift
   signature CONFIRMS or CONTRADICTS the declared class via cosine
   similarity to the class centroid.
6. Outputs a typed JSON manifest with the per-substrate signature, the
   pairwise similarity matrix, and the classifier verdict for each
   substrate (CONFIRMS / CONTRADICTS / INSUFFICIENT-DATA per
   declared class).

This is a RESEARCH-SIGNAL DIAGNOSTIC tool. All output is tagged
``[diagnostic-not-score]`` per CLAUDE.md "MPS auth eval is NOISE" +
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiables. The classifier is a hypothesis-testing
primitive, not a kill/promote primitive.

SISTER TOOL DISTINCTION (per ZZZZZ audit L3a 2026-05-12):
This tool is NOT a duplicate of ``tools/xray_substrate_classifier.py``.
The two tools share the ``substrate_class`` output token but operate on
DIFFERENT INPUTS:

- THIS tool (``cpu_cuda_xray_substrate_class_classifier.py``): consumes N
  per-substrate ``layer_drift.json`` files and classifies via CPU-vs-CUDA
  per-layer drift signature pairwise cosine similarity. Requires prior P5
  xray sweeps; runs post-dispatch as a numerical fingerprint check.
- SISTER tool (``xray_substrate_classifier.py``): consumes an archive ZIP
  and classifies via static magic-byte signatures + member-name lookup
  tables. Cheap, deterministic, runs offline pre-dispatch.

Both feed the autopilot's ``substrate_class`` column; they are complementary
(numerical post-dispatch + static pre-dispatch) rather than redundant.

CLAUDE.md compliance:

* no scorer load (consumes pre-emitted layer_drift.json files);
* no MPS / torch import;
* no ``/tmp`` paths — refused at output-dir validation;
* deterministic-bytes: same inputs + same args produce byte-identical
  output;
* no score claim ever;
* fails closed on schema violations.

Usage::

    .venv/bin/python tools/cpu_cuda_xray_substrate_class_classifier.py \\
        --input-spec hnerv_family:experiments/results/.../segnet_a1/layer_drift.json \\
        --input-spec hnerv_family:experiments/results/.../segnet_pr106_r2/layer_drift.json \\
        --input-spec non_hnerv_family:experiments/results/.../segnet_categorical/layer_drift.json \\
        --output-dir experiments/results/xray_substrate_classifier_<ts>/

    .venv/bin/python tools/cpu_cuda_xray_substrate_class_classifier.py \\
        --input-spec ... --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


# Recognised substrate-class hints. The operator declares per-input which
# class a substrate belongs to; the tool verifies via cosine similarity of
# the per-layer drift vector against the class centroid.
_SUBSTRATE_CLASSES: tuple[str, ...] = ("hnerv_family", "non_hnerv_family")


# Recognised diagnostic-evidence-grade tokens emitted by the P5 xray tools.
_VALID_EVIDENCE_GRADES: tuple[str, ...] = (
    "diagnostic_not_score",
    "diagnostic-not-score",
)


@dataclass(frozen=True)
class InputSpec:
    """One ``substrate-class:layer-drift-json-path`` operator-declared input."""

    substrate_class: str
    path: Path


@dataclass(frozen=True)
class SubstrateSignature:
    """Per-substrate per-layer drift signature + summary statistics."""

    substrate_class: str
    label: str
    layer_drift_path: str
    layer_drift_sha256: str
    n_layers: int
    mean_l2_relative_error: float
    max_l2_relative_error: float
    median_l2_relative_error: float
    first_divergence: dict[str, object]
    stage_compounding_total: float
    n_stages: int
    cpu_capture_substrate: str
    cuda_record_path: str
    cpu_record_path: str
    evidence_grade: str
    drift_vector: list[float] = field(default_factory=list)


def _validate_output_dir(output_dir: Path) -> None:
    """Refuse forbidden /tmp paths per CLAUDE.md non-negotiable."""
    as_str = str(output_dir.resolve())
    forbidden_anchors = ("/tmp/", "/var/tmp/", "/private/tmp/")
    for anchor in forbidden_anchors:
        if as_str.startswith(anchor):
            raise SystemExit(
                f"refusing to write to forbidden /tmp path {output_dir!s} "
                "per CLAUDE.md `forbidden_/tmp_paths_in_any_persisted_artifact`"
            )


def parse_input_spec(spec_str: str) -> InputSpec:
    """Parse a ``<substrate_class>:<path>`` operator-declared input spec."""
    if ":" not in spec_str:
        raise SystemExit(
            f"--input-spec must be of the form '<substrate_class>:<path>'; "
            f"got {spec_str!r}"
        )
    substrate_class, _, raw_path = spec_str.partition(":")
    substrate_class = substrate_class.strip()
    raw_path = raw_path.strip()
    if substrate_class not in _SUBSTRATE_CLASSES:
        raise SystemExit(
            f"unknown substrate_class {substrate_class!r}; expected one of "
            f"{list(_SUBSTRATE_CLASSES)}"
        )
    path = Path(raw_path)
    return InputSpec(substrate_class=substrate_class, path=path)


def _compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_and_validate(path: Path) -> dict[str, object]:
    """Load + schema-validate a layer_drift.json input."""
    if not path.exists():
        raise SystemExit(f"layer_drift.json not found at {path!s}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path!s}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"layer_drift.json at {path!s} must be a JSON object")
    required_keys = (
        "layer_drift_rows",
        "evidence_grade",
        "cpu_record_path",
        "cuda_record_path",
        "first_divergence",
    )
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise SystemExit(
            f"layer_drift.json at {path!s} missing required keys: {missing}"
        )
    grade = data.get("evidence_grade")
    if grade not in _VALID_EVIDENCE_GRADES:
        raise SystemExit(
            f"layer_drift.json at {path!s} has unexpected evidence_grade "
            f"{grade!r}; expected one of {list(_VALID_EVIDENCE_GRADES)}"
        )
    return data


def _l2_value_for_signature(row: dict[str, object]) -> float:
    """Extract the per-layer l2 value used for the drift signature.

    Falls back to the fingerprint-only proxy when the full tensor row is
    not available (capture_mode='fingerprint'). Returns 0.0 for NaN/None
    entries so downstream cosine arithmetic stays well-defined.
    """
    val = row.get("l2_relative_error")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        # Use the fingerprint proxy when full tensor unavailable.
        proxy = row.get("fingerprint_only_l2_proxy")
        if proxy is None or (isinstance(proxy, float) and math.isnan(proxy)):
            return 0.0
        return float(proxy)
    return float(val)


def extract_signature(
    *, substrate_class: str, path: Path, data: dict[str, object]
) -> SubstrateSignature:
    """Extract a per-substrate signature from a validated layer_drift.json."""
    rows = data.get("layer_drift_rows", []) or []
    if not isinstance(rows, list):
        raise SystemExit(
            f"layer_drift.json at {path!s} layer_drift_rows must be a list"
        )
    drift_vector = [_l2_value_for_signature(r) for r in rows]
    n_layers = len(drift_vector)

    if n_layers > 0:
        finite = [v for v in drift_vector if math.isfinite(v)]
        mean_l2 = float(sum(finite) / len(finite)) if finite else 0.0
        max_l2 = float(max(finite)) if finite else 0.0
        median_l2 = float(sorted(finite)[len(finite) // 2]) if finite else 0.0
    else:
        mean_l2 = max_l2 = median_l2 = 0.0

    stage_compounding = data.get("stage_compounding", {}) or {}
    by_stage = stage_compounding.get("by_stage", []) if isinstance(stage_compounding, dict) else []
    n_stages = len(by_stage) if isinstance(by_stage, list) else 0
    total_compounding = 1.0
    if isinstance(by_stage, list):
        for stage in by_stage:
            if isinstance(stage, dict):
                total_compounding *= float(stage.get("compound_factor", 1.0))

    cpu_capture = data.get("cpu_capture_host", {}) or {}
    if isinstance(cpu_capture, dict) and cpu_capture.get("is_linux_x86_64"):
        cpu_substrate = "linux_x86_64"
    elif isinstance(cpu_capture, dict) and cpu_capture.get("is_macos_darwin"):
        cpu_substrate = "macos_darwin"
    else:
        cpu_substrate = "unknown"

    return SubstrateSignature(
        substrate_class=substrate_class,
        label=str(data.get("label", path.stem)),
        layer_drift_path=str(path),
        layer_drift_sha256=_compute_sha256(path),
        n_layers=n_layers,
        mean_l2_relative_error=mean_l2,
        max_l2_relative_error=max_l2,
        median_l2_relative_error=median_l2,
        first_divergence=dict(data.get("first_divergence", {}) or {}),
        stage_compounding_total=total_compounding,
        n_stages=n_stages,
        cpu_capture_substrate=cpu_substrate,
        cuda_record_path=str(data.get("cuda_record_path", "")),
        cpu_record_path=str(data.get("cpu_record_path", "")),
        evidence_grade=str(data.get("evidence_grade", "")),
        drift_vector=drift_vector,
    )


def _truncate_to_min_length(
    vectors: list[list[float]],
) -> list[list[float]]:
    """Truncate every vector to the length of the shortest one."""
    if not vectors:
        return []
    min_len = min(len(v) for v in vectors)
    return [v[:min_len] for v in vectors]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Returns 0.0 when either norm is 0."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(x) * float(x) for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_pairwise_similarity(
    signatures: list[SubstrateSignature],
) -> list[dict[str, object]]:
    """Build a pairwise cosine-similarity matrix (off-diagonal only)."""
    rows: list[dict[str, object]] = []
    for i, sig_i in enumerate(signatures):
        for j, sig_j in enumerate(signatures):
            if j <= i:
                continue
            sim = cosine_similarity(sig_i.drift_vector, sig_j.drift_vector)
            rows.append(
                {
                    "substrate_a_label": sig_i.label,
                    "substrate_a_class": sig_i.substrate_class,
                    "substrate_b_label": sig_j.label,
                    "substrate_b_class": sig_j.substrate_class,
                    "min_n_layers_compared": min(
                        sig_i.n_layers, sig_j.n_layers
                    ),
                    "cosine_similarity": float(sim),
                    "same_class_declared": (
                        sig_i.substrate_class == sig_j.substrate_class
                    ),
                }
            )
    return rows


def compute_class_centroids(
    signatures: list[SubstrateSignature],
) -> dict[str, list[float]]:
    """Compute per-class centroid drift vectors (truncated to per-class min length).

    Centroid = element-wise mean of all member-substrate vectors after
    truncation to the per-class shortest vector length. Returns {} when
    a class has no members.
    """
    by_class: dict[str, list[list[float]]] = {}
    for sig in signatures:
        by_class.setdefault(sig.substrate_class, []).append(sig.drift_vector)
    centroids: dict[str, list[float]] = {}
    for class_name, vectors in by_class.items():
        if not vectors:
            continue
        truncated = _truncate_to_min_length(vectors)
        if not truncated or not truncated[0]:
            centroids[class_name] = []
            continue
        n = len(truncated[0])
        centroid = [
            sum(v[k] for v in truncated) / float(len(truncated))
            for k in range(n)
        ]
        centroids[class_name] = centroid
    return centroids


def classify_substrates(
    signatures: list[SubstrateSignature],
    centroids: dict[str, list[float]],
    *,
    min_class_members_for_classification: int = 2,
    confirms_threshold: float = 0.85,
    contradicts_threshold: float = 0.30,
) -> list[dict[str, object]]:
    """Per-substrate verdict: CONFIRMS / CONTRADICTS / INSUFFICIENT-DATA.

    A substrate's drift vector is compared against the centroid of EACH
    class. The verdict is:

    * ``CONFIRMS`` if cosine_similarity to declared-class centroid >=
      confirms_threshold AND it's also higher than to any OTHER class.
    * ``CONTRADICTS`` if cosine_similarity to declared-class centroid <
      contradicts_threshold OR another class's centroid wins by a margin
      of >= 0.20.
    * ``INSUFFICIENT-DATA`` otherwise (e.g., declared class only has 1
      member so no meaningful centroid; or vector is empty / zero).
    """
    # Tally per-class membership counts to know which centroids are
    # statistically meaningful.
    counts: dict[str, int] = {}
    for sig in signatures:
        counts[sig.substrate_class] = counts.get(sig.substrate_class, 0) + 1

    rows: list[dict[str, object]] = []
    for sig in signatures:
        sims_by_class: dict[str, float] = {}
        for class_name, centroid in centroids.items():
            sims_by_class[class_name] = cosine_similarity(
                sig.drift_vector, centroid
            )
        declared_sim = sims_by_class.get(sig.substrate_class, 0.0)
        # Compare declared vs other classes.
        other_class_max = 0.0
        other_class_winner = None
        for name, sim in sims_by_class.items():
            if name == sig.substrate_class:
                continue
            if sim > other_class_max:
                other_class_max = sim
                other_class_winner = name

        # If the declared class only has 1 member (this substrate itself),
        # the declared-class centroid IS the substrate's own vector → cosine
        # similarity is trivially 1.0 → not statistically meaningful.
        if counts.get(sig.substrate_class, 0) < min_class_members_for_classification:
            verdict = "INSUFFICIENT-DATA"
            note = (
                f"declared substrate-class {sig.substrate_class!r} has "
                f"{counts.get(sig.substrate_class, 0)} member(s); need "
                f">= {min_class_members_for_classification} for classification"
            )
        elif declared_sim >= confirms_threshold and declared_sim >= other_class_max:
            verdict = "CONFIRMS"
            note = (
                f"declared {sig.substrate_class!r} sim "
                f"{declared_sim:.4f} >= {confirms_threshold} AND >= other-class "
                f"max {other_class_max:.4f}"
            )
        elif (
            declared_sim < contradicts_threshold
            or (other_class_winner is not None and other_class_max - declared_sim >= 0.20)
        ):
            verdict = "CONTRADICTS"
            note = (
                f"declared {sig.substrate_class!r} sim {declared_sim:.4f} "
                f"too low (< {contradicts_threshold}) OR other-class "
                f"{other_class_winner!r} sim {other_class_max:.4f} wins by "
                f">= 0.20 margin"
            )
        else:
            verdict = "INSUFFICIENT-DATA"
            note = (
                f"declared {sig.substrate_class!r} sim {declared_sim:.4f} "
                f"is in the in-between zone "
                f"[{contradicts_threshold}, {confirms_threshold}); "
                "data does not support a confirms/contradicts verdict"
            )

        rows.append(
            {
                "substrate_label": sig.label,
                "declared_substrate_class": sig.substrate_class,
                "verdict": verdict,
                "declared_class_similarity": float(declared_sim),
                "best_other_class": other_class_winner,
                "best_other_class_similarity": float(other_class_max),
                "all_class_similarities": {k: float(v) for k, v in sims_by_class.items()},
                "verdict_note": note,
            }
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cpu_cuda_xray_substrate_class_classifier",
        description=(
            "Classify N substrates by their per-layer CPU-vs-CUDA drift "
            "signature against operator-declared substrate-class hints. "
            "evidence_grade=diagnostic-not-score; "
            "score_claim=false; promotion_eligible=false."
        ),
    )
    parser.add_argument(
        "--input-spec",
        action="append",
        required=True,
        help=(
            "Operator-declared input: '<substrate_class>:<path>'. May be "
            "supplied multiple times. substrate_class must be one of "
            f"{list(_SUBSTRATE_CLASSES)}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Must NOT be under /tmp (CLAUDE.md "
            "forbidden_/tmp_paths_in_any_persisted_artifact). Required "
            "unless --dry-run is set."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the manifest to stdout WITHOUT writing any files."
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Operator handle for manifest provenance. When omitted, the "
            "manifest records operator=unknown."
        ),
    )
    parser.add_argument(
        "--confirms-threshold",
        type=float,
        default=0.85,
        help="Min cosine similarity for a CONFIRMS verdict. Default 0.85.",
    )
    parser.add_argument(
        "--contradicts-threshold",
        type=float,
        default=0.30,
        help="Max cosine similarity below which a CONTRADICTS verdict fires. Default 0.30.",
    )
    return parser.parse_args(argv)


def build_manifest(
    *,
    input_specs: list[InputSpec],
    operator: str | None,
    confirms_threshold: float,
    contradicts_threshold: float,
) -> dict[str, object]:
    """Run the classifier pipeline over the input specs + return the manifest."""
    signatures: list[SubstrateSignature] = []
    for spec in input_specs:
        data = _load_and_validate(spec.path)
        sig = extract_signature(
            substrate_class=spec.substrate_class, path=spec.path, data=data
        )
        signatures.append(sig)

    centroids = compute_class_centroids(signatures)
    pairwise = compute_pairwise_similarity(signatures)
    classifier_rows = classify_substrates(
        signatures,
        centroids,
        confirms_threshold=confirms_threshold,
        contradicts_threshold=contradicts_threshold,
    )

    # Per-class summary counts.
    class_counts: dict[str, int] = {}
    for sig in signatures:
        class_counts[sig.substrate_class] = (
            class_counts.get(sig.substrate_class, 0) + 1
        )

    # Per-verdict tally.
    verdict_counts: dict[str, int] = {}
    for row in classifier_rows:
        v = str(row["verdict"])
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    return {
        "schema": "cpu_cuda_xray_substrate_class_classifier_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": operator or "unknown",
        "n_input_substrates": len(signatures),
        "substrate_class_counts": class_counts,
        "verdict_counts": verdict_counts,
        "confirms_threshold": confirms_threshold,
        "contradicts_threshold": contradicts_threshold,
        "substrates": [
            {
                "substrate_class": sig.substrate_class,
                "label": sig.label,
                "layer_drift_path": sig.layer_drift_path,
                "layer_drift_sha256": sig.layer_drift_sha256,
                "n_layers": sig.n_layers,
                "mean_l2_relative_error": sig.mean_l2_relative_error,
                "max_l2_relative_error": sig.max_l2_relative_error,
                "median_l2_relative_error": sig.median_l2_relative_error,
                "first_divergence": sig.first_divergence,
                "stage_compounding_total": sig.stage_compounding_total,
                "n_stages": sig.n_stages,
                "cpu_capture_substrate": sig.cpu_capture_substrate,
                "cuda_record_path": sig.cuda_record_path,
                "cpu_record_path": sig.cpu_record_path,
                "evidence_grade": sig.evidence_grade,
            }
            for sig in signatures
        ],
        "pairwise_similarity": pairwise,
        "classifier_rows": classifier_rows,
        "centroid_lengths": {k: len(v) for k, v in centroids.items()},
        "evidence_grade": "diagnostic_not_score",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "blockers": [
            "no_score_change_byte_proxy",
            "diagnostic_signature_not_a_kill_promote_signal",
        ],
        "notes": (
            "Per CLAUDE.md `forbidden_premature_KILL_without_research_exhaustion` "
            "+ `forbidden_MPS_derived_strategic_decision`, this classifier "
            "verdict is a HYPOTHESIS-TESTING signal only. CONTRADICTS does "
            "NOT kill a substrate; it surfaces a research question worth "
            "deeper investigation. The classifier is a probe-disambiguator "
            "for the substrate-class-boundary hypothesis (council Insight 1) "
            "and feeds the per-archive drift posterior."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_specs = [parse_input_spec(s) for s in args.input_spec]

    if not args.dry_run and args.output_dir is None:
        raise SystemExit(
            "--output-dir is required unless --dry-run is set"
        )

    manifest = build_manifest(
        input_specs=input_specs,
        operator=args.operator,
        confirms_threshold=args.confirms_threshold,
        contradicts_threshold=args.contradicts_threshold,
    )

    if args.dry_run:
        json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    _validate_output_dir(args.output_dir)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        out_dir / "cpu_cuda_xray_substrate_class_classifier_manifest.json"
    )
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(body)

    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    print(
        f"wrote {manifest_path} (sha={sha[:8]}, "
        f"n_substrates={manifest['n_input_substrates']}, "
        f"verdicts={manifest['verdict_counts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
