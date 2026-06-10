#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Materialize the ``evaluator_invisibility_basis.v1`` artifact (task #47).

Builds the two-tier invisibility basis and persists it on the durable SSD tier:

  - TIER 1 (CERTIFIED EXACT): derived in closed form (no torch, no scorer, no
    GPU).  Persisted both as the JSONL header summary AND as an ``.npz`` with the
    zero-weight row/col index arrays + the boolean per-pixel invisibility mask
    (the queryable spatial surface consumers read).
  - TIER 2 (MEASURED LOW-SENSITIVITY): optionally projected from a #36
    ``EvaluatorResponseAtlas`` JSONL (``--atlas-jsonl``), one scoped row per pair
    (+ optional per-region rows).  Kept SEPARATE from tier 1 (Catalog #385).

Evidence grade: TIER 1 ``mathematical-derivation`` (hardware-independent); TIER 2
``[macOS-CPU advisory]``.  No score claim; no dispatch; ``promotable=false``.

Per CLAUDE.md disk hygiene: artifacts land on the SSD waterfall
(``/Volumes/VertigoDataTier/pact`` -> ``/Volumes/APDataStore/pact`` -> local opt-in)
in a timestamped run dir with a sha-cited manifest.  No ``/tmp``.

Usage
-----
    # Tier-1 only (closed-form; instant):
    PYTHONPATH=src:upstream .venv/bin/python \
        tools/build_evaluator_invisibility_basis.py

    # With tier-2 from the #36 atlas:
    PYTHONPATH=src:upstream .venv/bin/python \
        tools/build_evaluator_invisibility_basis.py \
        --atlas-jsonl /Volumes/VertigoDataTier/pact/evaluator_response_atlas_*/evaluator_response_atlas.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tac.optimization.evaluator_invisibility_basis import (
    EVALUATOR_INVISIBILITY_BASIS_SCHEMA,
    build_evaluator_invisibility_basis,
    tier2_rows_from_atlas,
)

SSD_WATERFALL = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


def _pick_output_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for root in SSD_WATERFALL:
        if root.exists():
            return root
    # local opt-in fallback (durable, NOT /tmp) per CLAUDE.md
    return Path(".omx") / "research" / "artifacts"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas-jsonl", default=None,
                    help="optional #36 EvaluatorResponseAtlas JSONL for tier-2 rows")
    ap.add_argument("--output-root", default=None,
                    help="override the SSD waterfall output root (durable, not /tmp)")
    ap.add_argument("--region-classes", default="",
                    help="comma-separated SegNet class ids for per-region tier-2 rows")
    args = ap.parse_args(argv)

    basis = build_evaluator_invisibility_basis(
        provenance={
            "subagent": "evaluator_null_space_compiler_20260609",
            "task": "#47 evaluator null-space compiler / invisibility basis",
            "tier1_source": "closed-form bilinear-resize null space (upstream/modules.py:73/109)",
        }
    )

    # ---- tier 2 (optional, from the #36 atlas) -----------------------------
    tier2_rows = ()
    atlas_ref = None
    if args.atlas_jsonl:
        from tac.optimization.evaluator_response_atlas import EvaluatorResponseAtlas

        atlas_path = Path(args.atlas_jsonl)
        lines = atlas_path.read_text().splitlines()
        atlas = EvaluatorResponseAtlas.from_jsonl_lines(lines)
        region_classes = [int(x) for x in args.region_classes.split(",") if x.strip()]
        tier2_rows = tier2_rows_from_atlas(
            atlas,
            pairs=len(atlas.rows),
            authority_tier="macos_cpu_advisory",
            artifact_path=str(atlas_path),
            confidence_interval="atlas 600-pair cpu_torch fields",
            region_classes=region_classes,
        )
        atlas_ref = {"atlas_jsonl": str(atlas_path), "n_pairs": len(atlas.rows)}
        basis = build_evaluator_invisibility_basis(
            tier2_rows=tier2_rows,
            provenance={**basis.provenance, "tier2_atlas": atlas_ref},
        )

    # ---- persist on the SSD tier -------------------------------------------
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _pick_output_root(args.output_root) / f"evaluator_invisibility_basis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSONL (header + tier-2 rows).
    jsonl_lines = basis.to_jsonl_lines()
    jsonl_text = "\n".join(jsonl_lines) + "\n"
    jsonl_path = out_dir / "evaluator_invisibility_basis.jsonl"
    jsonl_path.write_text(jsonl_text)
    jsonl_sha = _sha256_text(jsonl_text)
    header_sha = _sha256_text(jsonl_lines[0])

    # NPZ (tier-1 spatial surface: zero-weight indices + boolean mask).
    t1 = basis.tier1_resize
    npz_path = out_dir / "tier1_resize_null_space.npz"
    np.savez_compressed(
        npz_path,
        zero_weight_rows=np.asarray(t1.zero_weight_rows, dtype=np.int32),
        zero_weight_cols=np.asarray(t1.zero_weight_cols, dtype=np.int32),
        zero_weight_pixel_mask=t1.zero_weight_pixel_mask(),
        camera_h=np.int32(t1.camera_h),
        camera_w=np.int32(t1.camera_w),
        scorer_h=np.int32(t1.scorer_h),
        scorer_w=np.int32(t1.scorer_w),
    )
    npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    manifest = {
        "schema": EVALUATOR_INVISIBILITY_BASIS_SCHEMA,
        "generated_at_utc": stamp,
        "evidence": {
            "tier1": "mathematical-derivation",
            "tier2": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotable": False,
        },
        "tier1_summary": t1.to_summary(),
        "frame0_corollary": basis.frame0_corollary.to_summary(),
        "n_tier2_rows": len(basis.tier2_rows),
        "tier2_atlas_ref": atlas_ref,
        "files": {
            "jsonl": {"path": str(jsonl_path), "sha256": jsonl_sha,
                      "header_sha256": header_sha},
            "npz": {"path": str(npz_path), "sha256": npz_sha},
        },
        "provenance": basis.provenance,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"[evaluator_invisibility_basis] wrote artifact -> {out_dir}")
    print(f"  tier-1 zero-weight pixels/channel: {t1.n_zero_weight_pixels_per_channel} "
          f"({100*t1.zero_weight_pixel_fraction:.4f}%)")
    print(f"  tier-1 full resize null dim:       {t1.full_null_dim} "
          f"({100*t1.full_null_fraction:.4f}%)")
    print(f"  frame0 SegNet-invisible:           100% "
          f"({basis.frame0_corollary.segnet_invisible_directions} directions)")
    print(f"  tier-2 measured rows:              {len(basis.tier2_rows)}")
    print(f"  header sha256:                     {header_sha[:16]}")
    print(f"  jsonl: {jsonl_path}")
    print(f"  npz:   {npz_path}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
