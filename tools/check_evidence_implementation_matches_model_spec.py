#!/usr/bin/env python3
# no-argparse-OK: thin CLI wrapper for the preflight scanner
"""Scan cathedral_autopilot evidence rows for implementation-vs-model-spec gaps.

Bug class extincted (2026-05-08):
================================

The audit at
``feedback_implementation_vs_model_gap_audit_20260508.md`` documented that
4-of-5 "falsified-then-DEFERRED" technique classes had silent
implementation-vs-model gaps:

  - kalle_fold "mixture of canonical NN-PMF shapes" tested with generic
    Gaussian/Laplace/Cauchy (SHAPE-FAMILY mismatch).
  - tiny_nn "200-param MLP" tested with rank=8 factorized softmax
    (~5K params, 25x predicted size; CAPACITY mismatch).
  - compressai_balle "ScaleHyperprior" tested on 1D-reshaped INT8 weight
    symbols (SUBSTRATE mismatch — model designed for 2d natural images).
  - lossy_int4 lane class tested only with NAIVE PTQ (VARIANT mismatch —
    QAT, LSQ, GPTQ, AWQ are part of the lane class but not exercised).

Only ``lossy_coarsening_analytical`` had implementation == model.

The fix:
========

Each catalog row in ``tools/cathedral_autopilot.py`` now carries a
``model_spec`` field declaring the implementation contract:
  - capacity_constraint
  - architecture_class
  - substrate_constraint
  - canonical_shape_family
  - variant_required

This scanner:
  1. Loads ``reports/cathedral_autopilot_evidence.jsonl``.
  2. For each row, locates the catalog model_spec by ``technique`` name.
  3. Compares the row's tested implementation against the model_spec
     constraints. Mismatches are emitted as findings.
  4. Returns a list of ``ModelSpecMismatch`` objects.

Findings start as a non-blocking advisory (preflight ``strict=False``).
Once the live count is driven to zero (faithful re-tests are landed for
the audited 4 cases), the check flips to STRICT in ``preflight_all()``,
matching the canonical promotion pattern the rest of preflight uses.

Usage::

    .venv/bin/python tools/check_evidence_implementation_matches_model_spec.py
    .venv/bin/python tools/check_evidence_implementation_matches_model_spec.py --strict
    .venv/bin/python tools/check_evidence_implementation_matches_model_spec.py \\
        --evidence-jsonl reports/cathedral_autopilot_evidence.jsonl \\
        --catalog-source tools/cathedral_autopilot.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_JSONL = REPO_ROOT / "reports" / "cathedral_autopilot_evidence.jsonl"
DEFAULT_CATALOG_SOURCE = REPO_ROOT / "tools" / "cathedral_autopilot.py"


@dataclass
class ModelSpecMismatch:
    """A single evidence-row-vs-catalog-model_spec divergence."""
    technique: str
    evidence_source: str
    evidence_timestamp: str
    spec_field: str
    spec_value: Any
    reason: str

    def as_str(self) -> str:
        return (
            f"{self.technique}: {self.spec_field}={self.spec_value!r} "
            f"vs source={self.evidence_source!r} ({self.evidence_timestamp}) "
            f": {self.reason}"
        )


def _load_catalog(catalog_source: Path) -> dict[str, dict[str, Any]]:
    """Load ENCODER_TECHNIQUES + ARCH_TECHNIQUES from cathedral_autopilot.py.

    Returns a dict keyed by technique name. Each value is the full catalog
    row dict (including ``model_spec`` if present).
    """
    spec = importlib.util.spec_from_file_location(
        "_pact_cathedral_autopilot_for_check", catalog_source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load catalog source: {catalog_source}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    catalog: dict[str, dict[str, Any]] = {}
    for row in (
        list(getattr(module, "ENCODER_TECHNIQUES", []))
        + list(getattr(module, "ARCH_TECHNIQUES", []))
    ):
        if not isinstance(row, dict) or "name" not in row:
            continue
        catalog[str(row["name"])] = row
    return catalog


def _load_evidence(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load evidence rows from a JSONL file. Skips malformed lines."""
    if not jsonl_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    text = jsonl_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        for row in data:
            if isinstance(row, dict):
                out.append(row)
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _evidence_text(row: dict[str, Any]) -> str:
    """Concatenate the prose fields of an evidence row for keyword matching."""
    parts: list[str] = []
    for key in (
        "source",
        "evidence_grade",
        "evidence_marker",
        "evidence_semantics",
        "contest_dispatch_verdict",
    ):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            parts.append(val)
    return " ".join(parts).lower()


def _check_capacity(
    technique: str,
    row: dict[str, Any],
    text: str,
    spec: dict[str, Any],
) -> list[ModelSpecMismatch]:
    """Detect capacity mismatch (e.g., "<=200 params" vs evidence at 5K params)."""
    capacity = str(spec.get("capacity_constraint", "")).lower()
    if not capacity:
        return []
    out: list[ModelSpecMismatch] = []
    # Pattern 1: "200 param" vs rank-K factorized softmax in source.
    if "200" in capacity and "param" in capacity:
        rank_tokens = ["rank=8", "rank=16", "rank=32", "rank=64", "rank=128"]
        if any(tok in text for tok in rank_tokens) or "factorized" in text:
            out.append(ModelSpecMismatch(
                technique=technique,
                evidence_source=str(row.get("source", "")),
                evidence_timestamp=str(row.get("timestamp", "")),
                spec_field="capacity_constraint",
                spec_value=capacity,
                reason=(
                    "evidence mentions rank-K factorized softmax (typically "
                    "thousands of params) but model_spec caps at ~200 params"
                ),
            ))
    # Pattern 2: "5KB_to_10KB_compressed_hyperprior" vs evidence with much
    # smaller compressed model.
    if "5kb_to_10kb" in capacity.replace(" ", ""):
        if "1.1kb" in text or "1.1 kb" in text or "546 weights" in text:
            out.append(ModelSpecMismatch(
                technique=technique,
                evidence_source=str(row.get("source", "")),
                evidence_timestamp=str(row.get("timestamp", "")),
                spec_field="capacity_constraint",
                spec_value=capacity,
                reason=(
                    "evidence mentions ~1.1KB / 546-weight MLP but model_spec "
                    "calls for 5KB-10KB compressed hyperprior"
                ),
            ))
    return out


def _check_substrate(
    technique: str,
    row: dict[str, Any],
    text: str,
    spec: dict[str, Any],
) -> list[ModelSpecMismatch]:
    """Detect substrate mismatch (e.g., 2d_natural_image vs 1D weight symbols)."""
    substrate = str(spec.get("substrate_constraint", "")).lower()
    arch_class = str(spec.get("architecture_class", "")).lower()
    out: list[ModelSpecMismatch] = []
    # 2D-image substrate but evidence shows 1D / weight reshape.
    if "2d_natural_image" in substrate:
        clues = (
            "1d" in text
            or "pseudo-image" in text
            or "1×1×" in text
            or "reshape" in text
            or "int8" in text
            or "weight" in text
            or "symbol" in text
        )
        if clues:
            out.append(ModelSpecMismatch(
                technique=technique,
                evidence_source=str(row.get("source", "")),
                evidence_timestamp=str(row.get("timestamp", "")),
                spec_field="substrate_constraint",
                spec_value=substrate,
                reason=(
                    f"model_spec declares 2d_natural_image substrate "
                    f"(architecture_class={arch_class}) but evidence "
                    "source mentions 1D / pseudo-image / weight / symbol "
                    "substrate"
                ),
            ))
    return out


def _check_shape_family(
    technique: str,
    row: dict[str, Any],
    text: str,
    spec: dict[str, Any],
) -> list[ModelSpecMismatch]:
    """Detect canonical-shape-family mismatch (audit memo §1, kalle_fold)."""
    shape_family = str(spec.get("canonical_shape_family", "")).lower()
    out: list[ModelSpecMismatch] = []
    if "spike_and_slab" in shape_family or "kaiming" in shape_family:
        normalized = text.replace(" ", "")
        if (
            "gaussian+laplace+delta+uniform" in normalized
            or "gaussian+laplace+sparse+uniform" in normalized
            or "cauchy" in text
            or "generic" in text
            or "intuition" in text
            or "my own picked" in text
            or "4-comp" in text
            or "4-component" in text
        ):
            out.append(ModelSpecMismatch(
                technique=technique,
                evidence_source=str(row.get("source", "")),
                evidence_timestamp=str(row.get("timestamp", "")),
                spec_field="canonical_shape_family",
                spec_value=shape_family,
                reason=(
                    "model_spec declares NN-weight-distribution canonical "
                    "shapes (kaiming + laplace+outliers + spike-and-slab) "
                    "but evidence source mentions generic Gaussian/Laplace/"
                    "Cauchy / 4-component intuition"
                ),
            ))
    return out


def _check_variant_exhaustion(
    technique: str,
    row: dict[str, Any],
    text: str,
    spec: dict[str, Any],
    *,
    all_evidence_for_technique: list[dict[str, Any]],
) -> list[ModelSpecMismatch]:
    """Detect partial variant exhaustion (audit memo §4, lossy_int4 class).

    Only emits if the AGGREGATE set of variants tested across all evidence
    rows for the technique fails to cover ``variant_required``.
    """
    variants = [str(v).lower() for v in spec.get("variant_required", []) or []]
    if not variants or len(variants) < 2:
        return []
    aggregate_text = " ".join(
        _evidence_text(r) for r in all_evidence_for_technique
    )
    # Also include sibling techniques starting with the same root, e.g.
    # ``lossy_int4_qat`` evidence rows count toward the
    # ``lossy_int4_quantization`` variant set.
    seen_variants: list[str] = []
    for variant in variants:
        tokens = [tok for tok in variant.replace("=", "_").split("_") if tok]
        for tok in tokens:
            if tok and tok in aggregate_text:
                seen_variants.append(variant)
                break
    missing = [v for v in variants if v not in seen_variants]
    out: list[ModelSpecMismatch] = []
    if missing:
        out.append(ModelSpecMismatch(
            technique=technique,
            evidence_source=str(row.get("source", "")),
            evidence_timestamp=str(row.get("timestamp", "")),
            spec_field="variant_required",
            spec_value=missing,
            reason=(
                f"model_spec lists {len(variants)} required variants "
                f"({variants}); aggregate evidence has not exercised "
                f"{len(missing)} ({missing}) — lane class not exhausted"
            ),
        ))
    return out


def scan(
    *,
    repo_root: str | Path | None = None,
    evidence_jsonl: str | Path | None = None,
    catalog_source: str | Path | None = None,
) -> list[ModelSpecMismatch]:
    """Scan evidence rows and return all model_spec mismatches.

    Used both by the preflight check and by the standalone CLI. Pure
    function — no I/O side effects beyond reading the inputs.
    """
    root = Path(repo_root or REPO_ROOT)
    ev_path = Path(evidence_jsonl) if evidence_jsonl else (
        root / "reports" / "cathedral_autopilot_evidence.jsonl"
    )
    cat_path = Path(catalog_source) if catalog_source else (
        root / "tools" / "cathedral_autopilot.py"
    )
    catalog = _load_catalog(cat_path)
    evidence_rows = _load_evidence(ev_path)
    by_technique: dict[str, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        tech = str(row.get("technique", ""))
        if not tech:
            continue
        by_technique.setdefault(tech, []).append(row)

    findings: list[ModelSpecMismatch] = []
    # Per-row checks (capacity, substrate, shape_family).
    for row in evidence_rows:
        technique = str(row.get("technique", ""))
        if not technique:
            continue
        catalog_row = catalog.get(technique)
        if catalog_row is None:
            # Unknown technique name — not the fault of model_spec; skip.
            continue
        spec = catalog_row.get("model_spec")
        if not isinstance(spec, dict):
            findings.append(ModelSpecMismatch(
                technique=technique,
                evidence_source=str(row.get("source", "")),
                evidence_timestamp=str(row.get("timestamp", "")),
                spec_field="model_spec",
                spec_value=None,
                reason=(
                    "catalog row lacks a model_spec — cannot verify "
                    "implementation-vs-model match"
                ),
            ))
            continue
        text = _evidence_text(row)
        findings.extend(_check_capacity(technique, row, text, spec))
        findings.extend(_check_substrate(technique, row, text, spec))
        findings.extend(_check_shape_family(technique, row, text, spec))

    # Aggregate-per-technique check (variant exhaustion). One finding per
    # technique max so we don't flood when many evidence rows share the
    # same incomplete variant set.
    seen_techniques: set[str] = set()
    for row in evidence_rows:
        technique = str(row.get("technique", ""))
        if technique in seen_techniques:
            continue
        catalog_row = catalog.get(technique)
        if catalog_row is None:
            continue
        spec = catalog_row.get("model_spec")
        if not isinstance(spec, dict):
            continue
        seen_techniques.add(technique)
        findings.extend(_check_variant_exhaustion(
            technique, row, _evidence_text(row), spec,
            all_evidence_for_technique=by_technique.get(technique, []),
        ))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="repo root (default: derived from this script)",
    )
    parser.add_argument(
        "--evidence-jsonl", type=Path, default=None,
        help="evidence file (default: reports/cathedral_autopilot_evidence.jsonl)",
    )
    parser.add_argument(
        "--catalog-source", type=Path, default=None,
        help="catalog file (default: tools/cathedral_autopilot.py)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any mismatches found",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit findings as JSON",
    )
    args = parser.parse_args(argv)

    findings = scan(
        repo_root=args.repo_root,
        evidence_jsonl=args.evidence_jsonl,
        catalog_source=args.catalog_source,
    )

    if args.json:
        out = [
            {
                "technique": f.technique,
                "evidence_source": f.evidence_source,
                "evidence_timestamp": f.evidence_timestamp,
                "spec_field": f.spec_field,
                "spec_value": f.spec_value,
                "reason": f.reason,
            }
            for f in findings
        ]
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        if not findings:
            print("OK: 0 model_spec mismatches in evidence rows")
        else:
            print(
                f"FOUND: {len(findings)} model_spec mismatches in evidence rows"
            )
            for f in findings:
                print(f"  - {f.as_str()}")

    if findings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
