#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: thin CLI wrapper for the preflight scanner
"""Scan cathedral_autopilot evidence rows for implementation-vs-model-spec gaps.

Bug class guarded (2026-05-08):
===============================

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
the audited cases), the check can flip to STRICT in ``preflight_all()``,
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
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_JSONL = REPO_ROOT / "reports" / "cathedral_autopilot_evidence.jsonl"
DEFAULT_CATALOG_SOURCE = REPO_ROOT / "tools" / "cathedral_autopilot.py"

VARIANT_TECHNIQUE_ALIASES: dict[str, tuple[str, ...]] = {
    "lossy_int4_quantization": ("naive_ptq",),
    # The QAT reactivation tool trains LSQ-style learnable scales, so one
    # positive row covers both the QAT and LSQ contract dimensions.
    "lossy_int4_quantization_qat": ("qat", "lsq"),
    "lossy_int4_per_channel_scales": ("per_channel_scales",),
    "lossy_int4_mixed_precision": ("mixed_precision_int4_int6_int8",),
    "lossy_int4_quantization_gptq": ("gptq",),
    "lossy_int4_quantization_awq": ("awq",),
}

PARENT_TECHNIQUE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "lossy_int4_quantization": (
        "lossy_int4_quantization",
        "lossy_int4_quantization_qat",
        "lossy_int4_per_channel_scales",
        "lossy_int4_mixed_precision",
        "lossy_int4_quantization_gptq",
        "lossy_int4_quantization_awq",
    ),
}

VARIANT_TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "naive_ptq": ("naive ptq", "ptq", "block size 1024", "block_size 1024"),
    "qat": ("qat",),
    "lsq": ("lsq", "lsq style", "lsq-style"),
    "per_channel_scales": (
        "per channel",
        "per-channel",
        "per output channel",
        "per-output-channel",
        "per channel scales",
    ),
    "mixed_precision_int4_int6_int8": (
        "mixed precision",
        "mixed-precision",
        "int4 int6 int8",
        "int4/int6/int8",
    ),
    "gptq": ("gptq",),
    "awq": ("awq",),
    # Keep CompressAI Balle aliases intentionally narrow. Free prose such as
    # "FULL ScaleHyperprior" is not enough to prove both variants; the rigorous
    # closure path is structured manifest fields or explicit variant lists.
    "scale_hyperprior": ("scale_hyperprior", "model class scale"),
    "mean_scale_hyperprior": (
        "mean_scale_hyperprior",
        "mean_scale",
        "model class mean scale",
    ),
}

NEGATED_VARIANT_PHRASES = (
    "not tested",
    "not exercised",
    "unexercised",
    "missing",
    "remaining",
    "not yet",
)

SUPPLEMENTAL_MANIFEST_GLOBS: dict[str, tuple[str, ...]] = {
    "compressai_balle_hyperprior": (
        "reports/raw/pr101_compressai_balle_full_*/manifest.json",
    ),
}

BALLE_FULL_SCHEMA = "pr101_compressai_balle_hyperprior_full.v1"
BALLE_FULL_TOOL = "tools/pr101_compressai_balle_hyperprior_full.py"
BALLE_CAPACITY_MIN_BYTES = 5 * 1024
BALLE_CAPACITY_MAX_BYTES = 10 * 1024


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


def _source_path(root: Path, path: Path) -> str:
    """Return a stable source path for diagnostic output."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _iso_timestamp_from_path(path: Path) -> str:
    """Extract ``YYYY-mm-ddTHH:MM:SSZ`` from paths containing YYYYmmddTHHMMSSZ."""
    match = re.search(r"(20\d{6})T(\d{6})Z", str(path))
    if match is None:
        return ""
    date, time = match.groups()
    return (
        f"{date[0:4]}-{date[4:6]}-{date[6:8]}T"
        f"{time[0:2]}:{time[2:4]}:{time[4:6]}Z"
    )


def _int_values(items: list[Any], key: str) -> list[int]:
    """Collect positive integer values from a list of dict-ish manifest rows."""
    values: list[int] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            values.append(value)
    return values


def _balle_full_manifest_to_evidence_row(
    *,
    root: Path,
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Convert a rigorously identified full Balle manifest into evidence.

    This is deliberately narrower than text scraping: it only trusts the
    known full-reactivation schema/tool pair and derives variant coverage from
    structured ``configs_swept`` / ``results[].model_class`` values. The row is
    supplemental evidence for guard accounting, not a score claim and not a
    faithful model-spec supersession.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    if (
        manifest.get("schema") != BALLE_FULL_SCHEMA
        or manifest.get("tool") != BALLE_FULL_TOOL
    ):
        return None

    configs_raw = manifest.get("configs_swept")
    results_raw = manifest.get("results")
    configs = [str(v) for v in configs_raw] if isinstance(configs_raw, list) else []
    results = results_raw if isinstance(results_raw, list) else []

    variants: set[str] = set()
    for config in configs:
        config_norm = config.strip().lower()
        if config_norm.startswith("scale:"):
            variants.add("scale_hyperprior")
        if config_norm.startswith("mean_scale:"):
            variants.add("mean_scale_hyperprior")
    for result in results:
        if not isinstance(result, dict):
            continue
        model_class = str(result.get("model_class", "")).strip().lower()
        if model_class == "scale":
            variants.add("scale_hyperprior")
        if model_class == "mean_scale":
            variants.add("mean_scale_hyperprior")

    model_blob_bytes = _int_values(results, "model_blob_brotli_bytes")
    model_range = (
        [min(model_blob_bytes), max(model_blob_bytes)]
        if model_blob_bytes else []
    )
    model_range_text = (
        f"{model_range[0]}-{model_range[1]}" if model_range else "unknown"
    )
    substrate = str(manifest.get("substrate_adaptation_choice") or "")
    source = (
        f"[MPS-research-signal supplemental-manifest] "
        f"{_source_path(root, manifest_path)} "
        f"(schema={BALLE_FULL_SCHEMA}; tool={BALLE_FULL_TOOL}; "
        f"configs_swept={','.join(configs)}; "
        f"variants_tested={','.join(sorted(variants))}; "
        f"model_blob_brotli_bytes={model_range_text}; "
        f"substrate={substrate})"
    )

    capacity_matches = (
        bool(model_range)
        and model_range[0] >= BALLE_CAPACITY_MIN_BYTES
        and model_range[1] <= BALLE_CAPACITY_MAX_BYTES
    )
    substrate_matches = not any(
        token in substrate.lower()
        for token in ("int8", "symbol", "pseudo-image", "reshape", "1x1x")
    )
    return {
        "technique": "compressai_balle_hyperprior",
        "empirical_archive_bytes": manifest.get("best_archive_bytes"),
        "empirical_rel_err": manifest.get("best_rel_err"),
        "evidence_grade": manifest.get("evidence_grade"),
        "score_claim": manifest.get("score_claim", False),
        "promotion_eligible": manifest.get("promotion_eligible", False),
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "supplemental_manifest_evidence": True,
        "source": source,
        "timestamp": _iso_timestamp_from_path(manifest_path),
        "contest_dispatch_verdict": (
            "DEFERRED-pending-research; supplemental full Balle manifest "
            "used only for implementation-vs-model guard accounting"
        ),
        "supersedes_prior_DEFERRED_audit": True,
        "variants_tested": sorted(variants),
        "model_blob_brotli_bytes_range": model_range,
        "substrate_adaptation_choice": substrate,
        "model_spec_fidelity_audit": {
            "implementation_matches_model_spec": False,
            "variant_mapping_rigorous": variants == {
                "scale_hyperprior",
                "mean_scale_hyperprior",
            },
            "variant_mapping_basis": (
                "manifest schema/tool plus configs_swept/results.model_class"
            ),
            "capacity_matches_5kb_to_10kb": capacity_matches,
            "substrate_matches_2d_natural_image": substrate_matches,
            "score_claim": False,
        },
    }


def _source_mentions_path(source: str, root: Path, path: Path) -> bool:
    """Return true if an existing evidence source already cites a path."""
    return str(path) in source or _source_path(root, path) in source


def _load_supplemental_manifest_evidence(
    root: Path,
    *,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load narrow supplemental manifest evidence for known guard gaps."""
    supplemental: list[dict[str, Any]] = []
    existing_sources = [
        str(row.get("source") or "") for row in existing_rows
        if isinstance(row, dict)
    ]
    for technique, patterns in SUPPLEMENTAL_MANIFEST_GLOBS.items():
        if technique != "compressai_balle_hyperprior":
            continue
        for pattern in patterns:
            for manifest_path in sorted(root.glob(pattern)):
                if any(
                    _source_mentions_path(source, root, manifest_path)
                    for source in existing_sources
                ):
                    continue
                row = _balle_full_manifest_to_evidence_row(
                    root=root,
                    manifest_path=manifest_path,
                )
                if row is not None:
                    supplemental.append(row)
    return supplemental


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


def _normalized_variant_text(text: str) -> str:
    """Normalize evidence prose for conservative variant matching."""
    out = text.lower()
    for ch in "[](){}:,;/.\\|+-_=":
        out = out.replace(ch, " ")
    return " ".join(out.split())


def _has_negation_near(normalized: str, needle: str) -> bool:
    """Return True when a variant mention sits in a local negative clause."""
    start = normalized.find(needle)
    while start >= 0:
        end = start + len(needle)
        window = normalized[max(0, start - 48) : min(len(normalized), end + 96)]
        if any(phrase in window for phrase in NEGATED_VARIANT_PHRASES):
            return True
        start = normalized.find(needle, end)
    return False


def _variant_mentioned_positively(variant: str, text: str) -> bool:
    """Detect a positive variant mention without crediting "NOT tested" prose."""
    normalized = _normalized_variant_text(text)
    aliases = VARIANT_TEXT_ALIASES.get(variant, (variant.replace("_", " "),))
    for alias in aliases:
        alias_norm = _normalized_variant_text(alias)
        if not alias_norm or alias_norm not in normalized:
            continue
        if _has_negation_near(normalized, alias_norm):
            continue
        return True
    return False


def _variant_hits_for_row(row: dict[str, Any]) -> set[str]:
    """Return model_spec variant names positively evidenced by one row."""
    hits: set[str] = set(VARIANT_TECHNIQUE_ALIASES.get(str(row.get("technique", "")), ()))
    for key in ("reactivation_criteria_tested", "variants_tested"):
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                item_norm = str(item).lower()
                for variant in VARIANT_TEXT_ALIASES:
                    if (
                        item_norm == variant
                        or item_norm.replace("-", "_") == variant
                        or _variant_mentioned_positively(variant, item_norm)
                    ):
                        hits.add(variant)
    text = _evidence_text(row)
    for variant in VARIANT_TEXT_ALIASES:
        if _variant_mentioned_positively(variant, text):
            hits.add(variant)
    return hits


def _row_timestamp(row: dict[str, Any]) -> str:
    """Return an ISO-ish timestamp string for conservative supersession checks."""
    return str(row.get("timestamp") or "")


def _is_model_faithful_supersession(row: dict[str, Any]) -> bool:
    """Return true for rows explicitly marked as faithful model-spec re-tests."""
    if row.get("supersedes_prior_model_spec_mismatch") is not True:
        return False
    audit = row.get("model_spec_fidelity_audit")
    if not isinstance(audit, dict):
        return False
    return (
        audit.get("1:1_fidelity") is True
        or audit.get("implementation_matches_model_spec") is True
    )


def _is_balle_full_implementation_supersession(row: dict[str, Any]) -> bool:
    """Return true for later full Balle implementation rows.

    These rows supersede stale "full ScaleHyperprior NOT tested" diagnosis for
    current guard accounting, but they do not mark the implementation faithful.
    Capacity/substrate checks still run against the later row and fail closed.
    """
    if str(row.get("technique", "")) != "compressai_balle_hyperprior":
        return False
    if row.get("supersedes_prior_DEFERRED_audit") is not True:
        return False
    source = str(row.get("source") or "").lower()
    return (
        row.get("supplemental_manifest_evidence") is True
        or "full scalehyperprior reactivation" in source
        or "pr101_compressai_balle_hyperprior_full.py" in source
        or "pr101_compressai_balle_full_" in source
    )


def _is_superseded_by_later_model_faithful_row(
    row: dict[str, Any],
    *,
    technique_rows: list[dict[str, Any]],
) -> bool:
    """Return true when a later explicit faithful re-test supersedes this row.

    Historical negative/mismatched rows are kept in the evidence ledger, but once
    a later row for the same technique is explicitly marked as a 1:1 model-spec
    re-test, the scanner should route current guard pressure to the newest
    faithful row instead of repeatedly flagging stale diagnosis prose.
    """
    row_ts = _row_timestamp(row)
    if not row_ts:
        return False
    for candidate in technique_rows:
        if candidate is row:
            continue
        if _row_timestamp(candidate) <= row_ts:
            continue
        if _is_model_faithful_supersession(candidate):
            return True
    return False


def _is_superseded_by_later_current_implementation_row(
    row: dict[str, Any],
    *,
    technique_rows: list[dict[str, Any]],
) -> bool:
    """Return true when a later current implementation row supersedes diagnosis."""
    row_ts = _row_timestamp(row)
    if not row_ts:
        return False
    for candidate in technique_rows:
        if candidate is row:
            continue
        if _row_timestamp(candidate) <= row_ts:
            continue
        if _is_balle_full_implementation_supersession(candidate):
            return True
    return False


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
    if "5kb_to_10kb" in capacity.replace(" ", "") and (
        "1.1kb" in text or "1.1 kb" in text or "546 weights" in text
    ):
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
    byte_range = row.get("model_blob_brotli_bytes_range")
    if (
        "5kb_to_10kb" in capacity.replace(" ", "")
        and isinstance(byte_range, list)
        and len(byte_range) == 2
        and all(isinstance(value, int) for value in byte_range)
    ):
        low, high = byte_range
        if low < BALLE_CAPACITY_MIN_BYTES or high > BALLE_CAPACITY_MAX_BYTES:
            out.append(ModelSpecMismatch(
                technique=technique,
                evidence_source=str(row.get("source", "")),
                evidence_timestamp=str(row.get("timestamp", "")),
                spec_field="capacity_constraint",
                spec_value=capacity,
                reason=(
                    "full Balle manifest model_blob_brotli_bytes range "
                    f"{low}-{high} falls outside the model_spec 5KB-10KB "
                    "compressed-hyperprior capacity band"
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
    seen_variants: set[str] = set()
    for evidence_row in all_evidence_for_technique:
        seen_variants.update(_variant_hits_for_row(evidence_row))
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
                f"{len(missing)} ({missing}); positively seen variants were "
                f"{sorted(seen_variants)} — lane class not exhausted"
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
    default_ev_path = root / "reports" / "cathedral_autopilot_evidence.jsonl"
    if ev_path.resolve() == default_ev_path.resolve():
        evidence_rows.extend(_load_supplemental_manifest_evidence(
            root,
            existing_rows=evidence_rows,
        ))
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
        if _is_superseded_by_later_model_faithful_row(
            row,
            technique_rows=by_technique.get(technique, []),
        ):
            continue
        if _is_superseded_by_later_current_implementation_row(
            row,
            technique_rows=by_technique.get(technique, []),
        ):
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
        related_names = PARENT_TECHNIQUE_EVIDENCE.get(technique, (technique,))
        related_rows: list[dict[str, Any]] = []
        for related_name in related_names:
            related_rows.extend(by_technique.get(related_name, []))
        findings.extend(_check_variant_exhaustion(
            technique, row, _evidence_text(row), spec,
            all_evidence_for_technique=related_rows,
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
