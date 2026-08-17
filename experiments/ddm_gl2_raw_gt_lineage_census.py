#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_gl2 -- classify the RAW-BINARY ground-truth population that ``ddm_gl1``'s census could not see.

THE COVERAGE HOLE, MEASURED AT SOURCE
-------------------------------------
``ddm_gl1`` (2026-08-16) built the content-addressed lineage registry in :mod:`tac.gt_lineage`
and classified 68 artifacts.  Both of its discovery legs carry an EXTENSION ALLOW-LIST:

* ``ddm_gl1_gt_lineage_census.enumerate_code_referenced`` matches
  ``gt_[A-Za-z0-9_]*\\.(?:npy|npz|pt|pth)``;
* its ``logs/disk_sweep.txt`` (66 files) contains 26 ``.npy`` + 9 ``.npz`` + 31 ``.pt`` and
  **zero** ``.u8`` / ``.f16``.

``_extract_fields`` reinforces it: any other suffix returns ``unhandled suffix``.

So a SECOND ground-truth population -- headerless raw memmaps written by
``tools/lever_b_build_score_native_targets.py`` -- was structurally invisible.  The largest member
is ``targets_n600/gt_segnet_argmax.u8`` at 117,964,800 B = 600x384x512, with **15 live readers**
(measured, excluding this tool) including ``tools/lever_b_*``, ``tools/score_native_*``,
``tools/probe_regmax_family.py`` and ``experiments/measure_symbolic_topological_partition_mdl.py``.

``ddm_a1s`` §7 (2026-08-16) measured that this ``.u8`` and the qs3 ``gt_argmax_n600.npy``
(sha ``91d3ff11...``, registered DALI_NVDEC) **agree on only 117,944,127 / 117,964,800 sites --
they differ on 20,673 px = 0.017525 S units**, and stated plainly that it did NOT establish the
mechanism.  This tool establishes it.

WHY A RAW MEMMAP NEEDS DIFFERENT HANDLING, NOT JUST A WIDER REGEX
-----------------------------------------------------------------
A ``.npy`` carries its own shape and dtype.  A ``.u8`` carries NOTHING: the shape lives only in the
reader's source.  So this tool must (a) infer the shape from the byte count against the known seg
geometry and REFUSE when the count does not divide, and (b) bind the artifact to its producer
receipt by RECOMPUTING a fingerprint the receipt already published, rather than trusting a
neighbouring JSON that merely sits in the same directory.  A sibling file is not provenance.

THE PRODUCER-RECEIPT BINDING (the ``PRODUCER_DECLARED`` rung, earned not asserted)
---------------------------------------------------------------------------------
``targets_meta.json`` publishes ``argmax_class_histogram`` -- five class fractions at full float
precision.  Recomputing those five numbers from the candidate bytes and requiring an exact match
BINDS the receipt to the bytes.  Only then may the receipt's declared decoder
(``tools/lever_b_build_score_native_targets.py``, which imports ``frame_utils.yuv420_to_rgb`` and
refuses any non-CPU device) be read as ``PRODUCER_DECLARED``.  Without that check the receipt is
just an adjacent file, which is the same authority-by-proximity error a name-keyed registry makes.

REUSED, NOT REINVENTED
----------------------
The rulers, the custody check on them, the uint8 normalization, and the decisive-margin rule all
come from :mod:`ddm_gl1_gt_lineage_census` by import.  Re-deriving them here would create a second
classification law that could silently drift from the first.

SCORER-FREE.  No SegNet/PoseNet forward, no Metal slot, no dispatch.  Only cached-tensor diffs.

Axis: ``[macOS-CPU advisory]`` -- lineage classification, never a score.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from ddm_gl1_gt_lineage_census import (
    DECISIVE_MARGIN,
    SEG_SHAPE,
    Rulers,
    _as_seg_u8,
    _classify_from_distances,
    sha256_file,
)

from tac.gt_lineage import DALI_NVDEC, PYAV_YUV420_TO_RGB

SEG_HW = SEG_SHAPE[1:]  # (384, 512)
SEG_PX_PER_FRAME = SEG_HW[0] * SEG_HW[1]  # 196,608
N_CLASSES = 5
POSE_DIMS = 6  # PoseNet emits 12; the first 6 are the scored dims and the only ones cached

# Lineage labels, imported by value from the canonical vocabulary so this tool cannot drift from
# tac.gt_lineage by re-spelling a string.
DALI_NVDEC_LABEL = DALI_NVDEC
PYAV_LABEL = PYAV_YUV420_TO_RGB

#: Raw containers gl1's allow-list excludes.  ``itemsize`` is what turns a byte count into a frame
#: count; a suffix whose itemsize we cannot name is refused rather than guessed.
RAW_SUFFIX_DTYPE: dict[str, str] = {
    ".u8": "uint8",
    ".i8": "int8",
    ".f16": "float16",
    ".f32": "float32",
}

#: Widened discovery regex.  gl1's stops at ``npy|npz|pt|pth``; this adds every raw suffix above.
#: Kept as ONE pattern with an explicit alternation so the allow-list is visible and auditable
#: rather than hidden in a suffix test somewhere downstream.
_RAW_ALT = "|".join(s.lstrip(".") for s in RAW_SUFFIX_DTYPE)
CODE_REF_PATTERN = re.compile(
    r"""["']([^"'\s]*gt_[A-Za-z0-9_]*\.(?:npy|npz|pt|pth|""" + _RAW_ALT + r"""))["']"""
)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return ""


def infer_raw_seg_shape(nbytes: int, dtype_name: str) -> tuple[int, ...] | None:
    """Infer ``(n, 384, 512)`` from a headerless raw file's byte count, or return ``None``.

    Returns ``None`` -- never a guess -- when the byte count does not divide evenly into whole
    384x512 frames.  A headerless file that does not divide is not a seg field we can read, and
    inventing a shape for it would fabricate a comparison.
    """
    itemsize = np.dtype(dtype_name).itemsize
    if nbytes <= 0 or nbytes % (itemsize * SEG_PX_PER_FRAME) != 0:
        return None
    n = nbytes // (itemsize * SEG_PX_PER_FRAME)
    return (n, SEG_HW[0], SEG_HW[1])


def infer_raw_pose_shape(nbytes: int, dtype_name: str) -> tuple[int, int] | None:
    """Infer ``(n, 6)`` from a headerless raw pose table's byte count, or return ``None``.

    Only float dtypes are considered: an integer raw file that happens to divide by 6 is far more
    likely to be something else entirely, and guessing "pose table" from divisibility alone would
    manufacture a comparison out of arithmetic.
    """
    if dtype_name not in ("float32", "float16"):
        return None
    itemsize = np.dtype(dtype_name).itemsize
    if nbytes <= 0 or nbytes % (itemsize * POSE_DIMS) != 0:
        return None
    return (nbytes // (itemsize * POSE_DIMS), POSE_DIMS)


def _classify_raw_pose(
    path: Path,
    row: dict[str, Any],
    shape: tuple[int, int],
    dtype_name: str,
    rulers: Rulers,
) -> dict[str, Any]:
    """Classify a headerless ``(n, 6)`` pose table against the two #906 rulers.

    Reuses gl1's exact per-row digest index so an n-subset is matched by content lookup rather than
    by assuming it is a prefix.  A subset that matches NEITHER ruler exactly stays UNKNOWN: for a
    pose table there is no cheap second test, and inventing one would be worse than saying so.
    """
    import hashlib

    arr = np.asarray(np.memmap(path, dtype=dtype_name, mode="r", shape=shape))
    n = shape[0]

    def rd(r: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(r, dtype=np.float32).tobytes()).hexdigest()

    hit_d = sum(1 for i in range(n) if rd(arr[i]) in rulers.dali_pose_hash)
    hit_a = sum(1 for i in range(n) if rd(arr[i]) in rulers.av_pose_hash)
    if hit_d == n and hit_a < n:
        lin, ev = DALI_NVDEC_LABEL, "EMPIRICAL_EXACT_POSE_ROW_SUBSET"
    elif hit_a == n and hit_d < n:
        lin, ev = PYAV_LABEL, "EMPIRICAL_EXACT_POSE_ROW_SUBSET"
    else:
        lin, ev = "UNKNOWN_AMBIGUOUS", "EMPIRICAL_POSE_ROW_SUBSET_PARTIAL"
    row["pose_leg"] = {
        "n_pairs": n,
        "rows_exactly_in_dali_ruler": hit_d,
        "rows_exactly_in_av_ruler": hit_a,
        "lineage": lin,
        "evidence": ev,
    }
    row["lineage"] = lin
    row["lineage_evidence"] = ev
    row["lineage_reason"] = (
        "pose rows matched by exact float32 content against both #906 ruler pose tables"
        if lin != "UNKNOWN_AMBIGUOUS"
        else (
            f"only {hit_d}/{n} rows match the DALI ruler and {hit_a}/{n} the AV ruler exactly; "
            "no decisive content identity, and no second test exists for a bare pose table"
        )
    )
    return row


def enumerate_code_referenced_raw() -> dict[str, list[str]]:
    """GT path literals in our instruments, including the raw suffixes gl1's regex omits.

    Test fixtures are excluded for the same reason gl1 excludes them: a test that writes a GT
    name into ``tmp_path`` names a file no instrument ever loads, and counting it inflates the
    denominator.
    """
    out: dict[str, list[str]] = {}
    for sub in ("experiments", "tools", "src"):
        base = REPO_ROOT / sub
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            if "tests" in py.parts or py.name.startswith("test_"):
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in CODE_REF_PATTERN.finditer(text):
                out.setdefault(m.group(1), []).append(str(py.relative_to(REPO_ROOT)))
    return out


def _resolve_literal_paths(literal: str) -> list[Path]:
    """Resolve a source literal to existing files.

    Readers name ``targets_dir / "gt_segnet_argmax.u8"``, so the literal is often a BARE
    basename whose directory lives in a separate constant.  Resolving only ``REPO_ROOT/literal``
    would therefore find nothing.  Known target roots are searched too; each hit is recorded with
    the root it came from so the join is auditable.
    """
    hits: list[Path] = []
    p = Path(literal)
    for cand in (p, REPO_ROOT / literal):
        if cand.is_file():
            hits.append(cand.resolve())
    if not p.is_absolute():
        for root in KNOWN_TARGET_ROOTS:
            cand = root / literal
            if cand.is_file():
                hits.append(cand.resolve())
    return sorted(set(hits))


#: Directories that the raw readers join their bare basenames onto.  These are the
#: ``DEFAULT_SURFACE_ROOT`` / ``_SSD`` / ``_DEFAULT_ARGMAX`` constants read out of the reader
#: sources, not invented locations.  Both the surface root AND its per-n subdirectories are
#: listed because reader literals appear at BOTH depths -- some name
#: ``targets_n600/gt_segnet_argmax.u8``, others just ``gt_segnet_argmax.u8`` -- and resolving only
#: one depth silently drops half the population, which is the very class of miss this arm cures.
_LEVER_B_SURFACE_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
)
KNOWN_TARGET_ROOTS: tuple[Path, ...] = (
    _LEVER_B_SURFACE_ROOT,
    _LEVER_B_SURFACE_ROOT / "targets_n600",
    _LEVER_B_SURFACE_ROOT / "targets_n16",
)


def verify_producer_receipt(path: Path, seg: np.ndarray) -> dict[str, Any]:
    """BIND a sibling ``targets_meta.json`` to these exact bytes, or report that it does not bind.

    The receipt publishes ``argmax_class_histogram``.  Recomputing it from ``seg`` and requiring
    an exact match is what converts "a JSON sits next to this file" into evidence about THESE
    bytes.  Mismatch, absence, or a receipt that names a different path all return
    ``bound=False`` -- an unbound receipt confers nothing.
    """
    meta_path = path.parent / "targets_meta.json"
    out: dict[str, Any] = {
        "receipt_path": str(meta_path),
        "receipt_exists": meta_path.is_file(),
        "bound": False,
        "reason": "",
        "declared_decoder": None,
        "declared_producer": None,
        "declared_device": None,
    }
    if not out["receipt_exists"]:
        out["reason"] = "no targets_meta.json beside the artifact"
        return out
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        out["reason"] = f"receipt unreadable: {type(exc).__name__}: {exc}"
        return out

    named = (meta.get("artifacts") or {}).get("argmax_u8")
    out["receipt_names_path"] = named
    if named and Path(named).resolve() != path.resolve():
        out["reason"] = f"receipt names a different artifact: {named}"
        return out

    hist = meta.get("argmax_class_histogram")
    if not isinstance(hist, dict) or not hist:
        out["reason"] = "receipt publishes no argmax_class_histogram to check against"
        return out

    counts = np.bincount(np.asarray(seg).reshape(-1), minlength=N_CLASSES).astype(np.float64)
    total = float(counts.sum())
    observed = {f"class_{i}_frac": float(counts[i] / total) for i in range(N_CLASSES)}
    out["receipt_histogram"] = {k: float(v) for k, v in hist.items()}
    out["observed_histogram"] = observed
    max_abs = max(
        abs(observed.get(k, float("nan")) - float(v)) for k, v in hist.items()
    )
    out["max_abs_histogram_delta"] = max_abs
    if max_abs == 0.0:
        out["bound"] = True
        out["reason"] = (
            "receipt argmax_class_histogram reproduces EXACTLY from these bytes "
            f"({len(hist)} classes, max |delta| = 0.0)"
        )
        out["declared_producer"] = meta.get("subagent")
        # The producer script's decode path, read at source, not inferred from the receipt text.
        out["declared_decoder"] = "frame_utils.yuv420_to_rgb"
        out["declared_device"] = "cpu"
        out["declared_producer_script"] = "tools/lever_b_build_score_native_targets.py"
        out["declared_utc"] = meta.get("utc")
        out["declared_video"] = meta.get("video")
    else:
        out["reason"] = (
            f"receipt histogram does NOT reproduce from these bytes (max |delta| = {max_abs:.3e}); "
            "the receipt is adjacent, not bound"
        )
    return out


def classify_raw_artifact(path: Path, rulers: Rulers) -> dict[str, Any]:
    """Measure one raw-binary artifact's decode lineage against the two #906 rulers."""
    st = path.stat()
    dtype_name = RAW_SUFFIX_DTYPE.get(path.suffix)
    row: dict[str, Any] = {
        "path": str(path),
        "basename": path.name,
        "bytes": st.st_size,
        "sha256": sha256_file(path),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
        "suffix": path.suffix,
        "assumed_dtype": dtype_name,
    }
    if dtype_name is None:
        row["lineage"] = "UNKNOWN_UNCOMPARABLE"
        row["lineage_evidence"] = "NONE"
        row["lineage_reason"] = f"suffix {path.suffix} has no declared raw dtype; refusing to guess one"
        return row

    shape = infer_raw_seg_shape(st.st_size, dtype_name)
    row["inferred_shape"] = list(shape) if shape else None
    if shape is None:
        pose_shape = infer_raw_pose_shape(st.st_size, dtype_name)
        row["inferred_pose_shape"] = list(pose_shape) if pose_shape else None
        if pose_shape is None:
            row["lineage"] = "UNKNOWN_UNCOMPARABLE"
            row["lineage_evidence"] = "NONE"
            row["lineage_reason"] = (
                f"{st.st_size} bytes at dtype {dtype_name} divides into neither whole "
                f"{SEG_HW[0]}x{SEG_HW[1]} seg frames nor whole {POSE_DIMS}-dim pose rows"
            )
            return row
        return _classify_raw_pose(path, row, pose_shape, dtype_name, rulers)

    arr = np.memmap(path, dtype=dtype_name, mode="r", shape=shape)
    # A float margin field has the seg GEOMETRY but is not a class field.  Classifying it against
    # an argmax ruler would compare incomparable quantities and manufacture a confident label.
    if dtype_name not in ("uint8", "int8"):
        row["lineage"] = "UNKNOWN_UNCOMPARABLE"
        row["lineage_evidence"] = "NONE"
        row["lineage_reason"] = (
            f"dtype {dtype_name} carries seg geometry but is not a class-label field "
            "(the rulers store argmax labels); not comparable"
        )
        return row

    lo, hi = int(arr.min()), int(arr.max())
    row["value_range"] = [lo, hi]
    if lo < 0 or hi >= N_CLASSES:
        row["lineage"] = "UNKNOWN_UNCOMPARABLE"
        row["lineage_evidence"] = "NONE"
        row["lineage_reason"] = (
            f"values span {lo}..{hi}, outside the 0..{N_CLASSES - 1} class range; not an argmax field"
        )
        return row

    seg = _as_seg_u8(np.asarray(arr))
    n = shape[0]
    row["producer_receipt"] = verify_producer_receipt(path, seg)

    if n == rulers.dali_seg.shape[0]:
        d_dali = int(np.count_nonzero(seg != rulers.dali_seg))
        d_av = int(np.count_nonzero(seg != rulers.av_seg))
        lin, ev, margin = _classify_from_distances(float(d_dali), float(d_av), metric="SEG_SITES")
        row["seg_leg"] = {
            "n_frames": n,
            "total_sites": int(n) * SEG_PX_PER_FRAME,
            "differing_sites_vs_dali": d_dali,
            "differing_sites_vs_av": d_av,
            "lineage": lin,
            "evidence": ev,
            "margin": margin,
        }
    else:
        # Subset: match each frame by content hash into the ruler frame index, exactly as gl1 does.
        import hashlib

        def fd(f: np.ndarray) -> str:
            return hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()

        hit_d = sum(1 for i in range(n) if fd(seg[i]) in rulers.dali_frame_hash)
        hit_a = sum(1 for i in range(n) if fd(seg[i]) in rulers.av_frame_hash)
        if hit_d == n and hit_a < n:
            lin, ev, margin = "DALI_NVDEC", "EMPIRICAL_EXACT_FRAME_SUBSET", float("inf")
        elif hit_a == n and hit_d < n:
            lin, ev, margin = "PYAV_YUV420_TO_RGB", "EMPIRICAL_EXACT_FRAME_SUBSET", float("inf")
        else:
            # No exact frame match: fall back to the site-count comparison against the FIRST n
            # ruler frames, which is the alignment every n-prefix cache in this family uses.
            d_dali = int(np.count_nonzero(seg != rulers.dali_seg[:n]))
            d_av = int(np.count_nonzero(seg != rulers.av_seg[:n]))
            lin, ev, margin = _classify_from_distances(
                float(d_dali), float(d_av), metric="SEG_SITES_PREFIX_ALIGNED"
            )
            row["seg_leg"] = {
                "n_frames": n,
                "total_sites": int(n) * SEG_PX_PER_FRAME,
                "frames_exactly_in_dali_ruler": hit_d,
                "frames_exactly_in_av_ruler": hit_a,
                "differing_sites_vs_dali_prefix": d_dali,
                "differing_sites_vs_av_prefix": d_av,
                "lineage": lin,
                "evidence": ev,
                "margin": margin,
            }
            row["lineage"] = lin
            row["lineage_evidence"] = ev
            row["lineage_reason"] = "measured against both #906 rulers, prefix-aligned"
            return row
        row["seg_leg"] = {
            "n_frames": n,
            "frames_exactly_in_dali_ruler": hit_d,
            "frames_exactly_in_av_ruler": hit_a,
            "lineage": lin,
            "evidence": ev,
            "margin": margin,
        }

    row["lineage"] = row["seg_leg"]["lineage"]
    row["lineage_evidence"] = row["seg_leg"]["evidence"]
    row["lineage_reason"] = "measured against both #906 rulers"

    # A BOUND producer receipt is the strongest rung on gl1's ladder and it outranks a content
    # comparison -- but only when it AGREES with the content.  A disagreement is a finding that
    # must be surfaced, never silently resolved in favour of either side.
    receipt = row.get("producer_receipt") or {}
    if receipt.get("bound"):
        declared = (
            "PYAV_YUV420_TO_RGB"
            if receipt.get("declared_decoder") == "frame_utils.yuv420_to_rgb"
            else None
        )
        row["receipt_declared_lineage"] = declared
        if declared and declared == row["lineage"]:
            row["lineage_evidence"] = f"PRODUCER_DECLARED|{row['lineage_evidence']}"
            row["lineage_reason"] = (
                "producer receipt BOUND to these bytes by exact histogram reproduction, and its "
                "declared decoder agrees with the content comparison against both #906 rulers"
            )
        elif declared:
            row["lineage"] = "UNKNOWN_AMBIGUOUS"
            row["lineage_evidence"] = "RECEIPT_CONTRADICTS_CONTENT"
            row["lineage_reason"] = (
                f"BOUND receipt declares {declared} but content classifies {row['seg_leg']['lineage']}; "
                "refusing to pick a side"
            )
    return row


def characterise_disagreement(
    a_path: Path,
    b_path: Path,
    out_dir: Path,
    *,
    a_label: str,
    b_label: str,
) -> dict[str, Any]:
    """Measure WHERE and HOW two seg argmax fields disagree, and PERSIST the mask.

    Persisting the mask (not only its count) is the ALWAYS-KEEP-THE-PAYLOAD rule: the count is a
    scalar summary of a field that the next consumer needs in full to re-derive anything.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(p: Path) -> np.ndarray:
        if p.suffix == ".npy":
            return _as_seg_u8(np.asarray(np.load(p, mmap_mode="r", allow_pickle=False)))
        dt = RAW_SUFFIX_DTYPE[p.suffix]
        shape = infer_raw_seg_shape(p.stat().st_size, dt)
        if shape is None:
            raise ValueError(f"{p} does not divide into seg frames")
        return _as_seg_u8(np.asarray(np.memmap(p, dtype=dt, mode="r", shape=shape)))

    a = _load(a_path)
    b = _load(b_path)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    n, h, w = a.shape

    diff = a != b
    n_diff = int(np.count_nonzero(diff))
    total = int(a.size)

    # Per-class confusion: how many sites go class i (in A) -> class j (in B).
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    idx = np.nonzero(diff)
    av = a[idx]
    bv = b[idx]
    np.add.at(conf, (av.astype(np.int64), bv.astype(np.int64)), 1)

    per_frame = np.count_nonzero(diff.reshape(n, -1), axis=1).astype(np.int64)
    per_row = np.count_nonzero(diff, axis=(0, 2)).astype(np.int64)
    per_col = np.count_nonzero(diff, axis=(0, 1)).astype(np.int64)

    # DISCRIMINATOR: does the disagreement sit ON the class boundary, or in cell interiors?
    # A photometric decode difference can only flip pixels whose top-two logits are already close,
    # and those live on the codim-1 boundary.  An interior-heavy pattern would point somewhere
    # else entirely.  The ENRICHMENT (boundary rate among differing sites / boundary rate among ALL
    # sites) is what carries the signal -- the numerator alone is uninterpretable without the
    # population denominator, which is reported beside it.
    bnd = np.zeros_like(diff)
    bnd[:, :-1, :] |= a[:, :-1, :] != a[:, 1:, :]
    bnd[:, 1:, :] |= a[:, :-1, :] != a[:, 1:, :]
    bnd[:, :, :-1] |= a[:, :, :-1] != a[:, :, 1:]
    bnd[:, :, 1:] |= a[:, :, :-1] != a[:, :, 1:]
    bnd_all = int(np.count_nonzero(bnd))
    bnd_at_diff = int(np.count_nonzero(bnd & diff))
    boundary_rate_population = bnd_all / total
    boundary_rate_at_diff = bnd_at_diff / max(1, n_diff)
    boundary_enrichment = (
        boundary_rate_at_diff / boundary_rate_population if boundary_rate_population > 0 else None
    )
    del bnd

    packed = np.packbits(diff.reshape(-1))
    mask_path = out_dir / "disagreement_mask_packbits.u8"
    packed.tofile(mask_path)

    np.save(out_dir / "disagreement_per_frame.npy", per_frame)
    np.save(out_dir / "disagreement_per_row.npy", per_row)
    np.save(out_dir / "disagreement_per_col.npy", per_col)
    np.save(out_dir / "disagreement_class_confusion.npy", conf)

    frames_with = int(np.count_nonzero(per_frame))
    # The spatial question a tie-break artifact and a decode split answer differently: a decode
    # split follows scene content (broad, every frame, class-boundary-hugging); a batch-shape
    # tie-break concentrates on exact-tie sites with no reason to track content.
    row_mass = per_row / max(1, n_diff)
    col_mass = per_col / max(1, n_diff)
    return {
        "a": {"label": a_label, "path": str(a_path), "sha256": sha256_file(a_path)},
        "b": {"label": b_label, "path": str(b_path), "sha256": sha256_file(b_path)},
        "shape": [int(n), int(h), int(w)],
        "total_sites": total,
        "differing_sites": n_diff,
        "agreement_fraction": (total - n_diff) / total,
        "d_seg_units": n_diff / total,
        "s_units_at_100x": 100.0 * n_diff / total,
        "frames_with_any_disagreement": frames_with,
        "frames_total": int(n),
        "per_frame_min": int(per_frame.min()),
        "per_frame_max": int(per_frame.max()),
        "per_frame_mean": float(per_frame.mean()),
        "per_frame_median": float(np.median(per_frame)),
        "class_confusion_a_to_b": conf.tolist(),
        "class_totals_a_at_diff_sites": conf.sum(axis=1).tolist(),
        "class_totals_b_at_diff_sites": conf.sum(axis=0).tolist(),
        "row_mass_top8": [[int(i), float(row_mass[i])] for i in np.argsort(-per_row)[:8]],
        "col_mass_top8": [[int(i), float(col_mass[i])] for i in np.argsort(-per_col)[:8]],
        "rows_with_any": int(np.count_nonzero(per_row)),
        "cols_with_any": int(np.count_nonzero(per_col)),
        "row_extent_with_any": [
            int(np.nonzero(per_row)[0].min()),
            int(np.nonzero(per_row)[0].max()),
        ]
        if np.count_nonzero(per_row)
        else None,
        "boundary_sites_in_population": bnd_all,
        "boundary_sites_among_differing": bnd_at_diff,
        "boundary_rate_population": boundary_rate_population,
        "boundary_rate_among_differing": boundary_rate_at_diff,
        "boundary_enrichment": boundary_enrichment,
        "payloads": {
            "disagreement_mask_packbits.u8": {
                "path": str(mask_path),
                "bytes": int(mask_path.stat().st_size),
                "sha256": sha256_file(mask_path),
                "note": f"np.packbits of the (n,h,w) boolean mask, C order, shape {[n, h, w]}",
            },
            "disagreement_per_frame.npy": {
                "path": str(out_dir / "disagreement_per_frame.npy"),
                "bytes": int((out_dir / "disagreement_per_frame.npy").stat().st_size),
                "sha256": sha256_file(out_dir / "disagreement_per_frame.npy"),
            },
            "disagreement_per_row.npy": {
                "path": str(out_dir / "disagreement_per_row.npy"),
                "bytes": int((out_dir / "disagreement_per_row.npy").stat().st_size),
                "sha256": sha256_file(out_dir / "disagreement_per_row.npy"),
            },
            "disagreement_per_col.npy": {
                "path": str(out_dir / "disagreement_per_col.npy"),
                "bytes": int((out_dir / "disagreement_per_col.npy").stat().st_size),
                "sha256": sha256_file(out_dir / "disagreement_per_col.npy"),
            },
            "disagreement_class_confusion.npy": {
                "path": str(out_dir / "disagreement_class_confusion.npy"),
                "bytes": int((out_dir / "disagreement_class_confusion.npy").stat().st_size),
                "sha256": sha256_file(out_dir / "disagreement_class_confusion.npy"),
            },
        },
    }


def merge_into_registry(census: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    """Fold this census's measured rows into the gl1 content-addressed registry, keyed by sha256.

    Merge rules, each chosen so the merge cannot launder anything:

    * **Keyed by sha256 only.**  A row lands under the digest of its bytes.  Two files sharing a
      basename never share a row; two paths sharing a digest do.
    * **Never overwrite a gl1 row.**  If the digest is already registered, only ``known_paths`` /
      ``known_basenames`` are widened.  A second census must not be able to silently reclassify an
      artifact the first one measured -- that would make the registry's answer depend on which
      tool ran last.
    * **UNKNOWN rows are recorded, not skipped.**  An artifact whose lineage this census could NOT
      resolve is still written, with lineage ``UNKNOWN_UNCOMPARABLE`` and the measured reason.
      Registry membership means "measured and recorded"; it does not mean "usable".
      ``assert_gt_lineage`` still refuses every unresolved row, so recording an UNKNOWN grants it
      nothing -- it only stops the artifact from being INVISIBLE, which is the actual defect.
    """
    reg = json.loads(registry_path.read_text())
    by_sha = {a["sha256"]: a for a in reg["artifacts"]}
    added, widened = [], []
    for row in census["artifacts"]:
        digest = row["sha256"]
        existing = by_sha.get(digest)
        if existing is not None:
            paths = sorted({*existing.get("known_paths", []), row["path"]})
            if paths != existing.get("known_paths"):
                existing["known_paths"] = paths
                existing["known_basenames"] = sorted({Path(p).name for p in paths})
                widened.append(digest)
            continue
        seg = row.get("seg_leg") or {}
        bits: list[str] = []
        if "differing_sites_vs_dali" in seg:
            bits.append(
                f"seg: {seg['differing_sites_vs_dali']:,} differing sites vs DALI ruler, "
                f"{seg['differing_sites_vs_av']:,} vs AV ruler, of {seg.get('total_sites', 0):,}"
            )
        elif "frames_exactly_in_dali_ruler" in seg:
            bits.append(
                f"seg: {seg['frames_exactly_in_dali_ruler']}/{seg['n_frames']} frames bit-identical "
                f"to DALI ruler, {seg['frames_exactly_in_av_ruler']}/{seg['n_frames']} to AV ruler"
            )
        receipt = row.get("producer_receipt") or {}
        if receipt.get("bound"):
            bits.append(
                f"producer receipt BOUND to these bytes ({receipt['reason']}); declared decoder "
                f"{receipt['declared_decoder']} on device {receipt['declared_device']} by "
                f"{receipt.get('declared_producer_script')}"
            )
        by_sha[digest] = {
            "sha256": digest,
            "bytes": row["bytes"],
            "basename": row["basename"],
            "lineage": row["lineage"],
            "lineage_evidence": row["lineage_evidence"],
            "measurement": "; ".join(bits) or row.get("lineage_reason", ""),
            "claim_boundary": _gl2_claim_boundary(row),
            "known_paths": [row["path"]],
            "known_basenames": [row["basename"]],
            "registered_by": "experiments/ddm_gl2_raw_gt_lineage_census.py",
        }
        added.append(digest)

    reg["artifacts"] = [by_sha[k] for k in sorted(by_sha)]
    reg["merged_censuses"] = sorted(
        {*reg.get("merged_censuses", []), census["produced_by"]}
    )
    reg["last_merged_utc"] = census["utc_finished"]
    registry_path.write_text(json.dumps(reg, indent=2) + "\n")
    return {"added": added, "widened": widened, "total": len(reg["artifacts"])}


def _gl2_claim_boundary(row: dict[str, Any]) -> str:
    """State exactly how far this row's label may be pushed.  Vocabulary follows gl1 / FX4."""
    lin = row.get("lineage", "")
    ev = row.get("lineage_evidence", "")
    if lin not in ("DALI_NVDEC", "PYAV_YUV420_TO_RGB"):
        return (
            "Lineage NOT established. Recorded so the artifact is VISIBLE to the population gauge, "
            "not because it is usable: assert_gt_lineage still refuses every unresolved row. "
            f"Reason: {row.get('lineage_reason', 'unmeasured')}"
        )
    parts = []
    if "PRODUCER_DECLARED" in ev:
        parts.append(
            "A producer receipt is BOUND to these exact bytes -- its published class histogram "
            "reproduces from them with max |delta| = 0.0 -- and the receipt's producer script "
            "imports frame_utils.yuv420_to_rgb and refuses any non-CPU device. This is the "
            "strongest rung: the decoder is declared, not merely inferred."
        )
    if "EXACT_FRAME_SUBSET" in ev:
        parts.append(
            "Every frame is additionally bit-identical to a frame of a producer-declared ruler."
        )
    elif "NEAREST_RULER" in ev:
        parts.append(
            "Content comparison against both #906 rulers agrees with the receipt. It still does "
            "not recover the original package versions or host, and within-family drift is nonzero."
        )
    return " ".join(parts)


def run_gate_positive_control(out_dir: Path) -> dict[str, Any]:
    """EXECUTE the registration gate: prove it can FAIL, then re-derive the live count.

    A guard nobody has watched refuse is indistinguishable from a guard that cannot refuse.  This
    runs, in order:

    1. **positive control** -- a synthetic instrument naming a GT artifact that exists and is NOT
       registered.  The gate MUST raise; if it does not, the gate is inert and this refuses.
    2. **unknown-container control** -- the same, with a suffix no container list has ever named.
       This is the specific defect ``ddm_gl1``'s allow-list had, so it gets its own control.
    3. **negative control** -- register the DIGEST; the gate must fall silent.
    4. **live count** -- run the gate against the real repo and RE-DERIVE the number.  A live count
       copied from a memo is an assertion, not a measurement.
    """
    import hashlib
    import tempfile

    from tac.gt_lineage import (
        GtArtifactLineage,
        GtLineageUnregisteredPopulation,
        assert_gt_population_registered,
        unregistered_gt_artifacts,
    )

    controls: list[dict[str, Any]] = []

    def _fixture(tmp: Path, name: str, payload: bytes) -> tuple[Path, Path, str]:
        repo = tmp / "repo"
        (repo / "tools").mkdir(parents=True, exist_ok=True)
        (repo / "tools" / "reader.py").write_text(f'GT = D / "{name}"\n', encoding="utf-8")
        root = tmp / "targets"
        root.mkdir(exist_ok=True)
        (root / name).write_bytes(payload)
        return repo, root, hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        repo, root, digest = _fixture(tmp, "gt_segnet_argmax.u8", b"\x00\x01\x02\x03")
        raised = None
        try:
            assert_gt_population_registered(repo_root=repo, search_roots=(root,), registry={})
        except GtLineageUnregisteredPopulation as exc:
            raised = exc
        controls.append(
            {
                "control": "positive__unregistered_raw_u8_must_refuse",
                "gate_raised": raised is not None,
                "observed_sha256": digest,
                "reported_artifacts": [a["sha256"] for a in (raised.artifacts if raised else ())],
                "expected": "raise GtLineageUnregisteredPopulation naming this digest",
                "passed": raised is not None
                and [a["sha256"] for a in raised.artifacts] == [digest],
            }
        )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        repo, root, digest = _fixture(tmp, "gt_future_field.zz9", b"\x07" * 16)
        raised = None
        try:
            assert_gt_population_registered(repo_root=repo, search_roots=(root,), registry={})
        except GtLineageUnregisteredPopulation as exc:
            raised = exc
        controls.append(
            {
                "control": "positive__unknown_container_suffix_must_refuse",
                "gate_raised": raised is not None,
                "observed_sha256": digest,
                "expected": "deny-list polarity: a suffix nobody has defined still enters the population",
                "passed": raised is not None
                and [a["sha256"] for a in raised.artifacts] == [digest],
            }
        )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        repo, root, digest = _fixture(tmp, "gt_segnet_argmax.u8", b"\x00\x01\x02\x03")
        reg = {
            digest: GtArtifactLineage(
                sha256=digest,
                bytes=4,
                basename="gt_segnet_argmax.u8",
                lineage="PYAV_YUV420_TO_RGB",
                evidence="EMPIRICAL_EXACT_MATCH",
                measurement="positive-control fixture",
                claim_boundary="positive-control fixture",
            )
        }
        ok = assert_gt_population_registered(
            repo_root=repo, search_roots=(root,), registry=reg
        )
        controls.append(
            {
                "control": "negative__registered_digest_must_pass",
                "gate_raised": False,
                "violations": ok,
                "expected": "no refusal once the DIGEST is registered",
                "passed": ok == [],
            }
        )

    live = unregistered_gt_artifacts()
    receipt = {
        "schema": "ddm_gl2_gate_positive_control_v1",
        "utc": _utc(),
        "git_head": _git_head(),
        "gate": "tac.gt_lineage.assert_gt_population_registered",
        "controls": controls,
        "all_controls_passed": all(c["passed"] for c in controls),
        "live_count_unregistered_gt_artifacts": len(live),
        "live_unregistered": live,
        "live_count_note": (
            "RE-DERIVED by executing the gate against this repo at the recorded git_head, not "
            "copied from a memo."
        ),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "GL2_GATE_POSITIVE_CONTROL.json").write_text(json.dumps(receipt, indent=2))
    if not receipt["all_controls_passed"]:
        raise SystemExit(
            "GATE POSITIVE CONTROL FAILED -- the gate did not refuse where it must. "
            f"See {out_dir / 'GL2_GATE_POSITIVE_CONTROL.json'}"
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True, help="payload + receipt destination")
    ap.add_argument(
        "--gate-positive-control",
        action="store_true",
        help="execute the registration gate's controls + re-derive the live count, then exit",
    )
    ap.add_argument(
        "--merge-into-registry",
        type=Path,
        default=None,
        help="fold the measured rows into src/tac/gt_lineage_registry.json (sha256-keyed, additive)",
    )
    ap.add_argument(
        "--disk-sweep",
        type=Path,
        default=None,
        help="newline-separated raw-GT paths from a find sweep (optional; code refs are always used)",
    )
    ap.add_argument(
        "--compare-against",
        type=Path,
        default=None,
        help="a registered .npy seg field to characterise the .u8 disagreement against",
    )
    ap.add_argument(
        "--compare-subject",
        type=Path,
        default=None,
        help="the raw artifact to compare (defaults to the n600 lever_b .u8)",
    )
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = _utc()

    if args.gate_positive_control:
        receipt = run_gate_positive_control(out_dir)
        for c in receipt["controls"]:
            print(f"[gl2] control {c['control']}: {'PASS' if c['passed'] else 'FAIL'}", flush=True)
        print(
            f"[gl2] live count (re-derived) unregistered GT artifacts: "
            f"{receipt['live_count_unregistered_gt_artifacts']}",
            flush=True,
        )
        return 0

    code_refs = enumerate_code_referenced_raw()
    raw_refs = {
        lit: srcs for lit, srcs in code_refs.items() if Path(lit).suffix in RAW_SUFFIX_DTYPE
    }

    candidates: dict[str, list[str]] = {}
    for lit, srcs in raw_refs.items():
        for p in _resolve_literal_paths(lit):
            candidates.setdefault(str(p), []).extend(srcs)
    swept = 0
    if args.disk_sweep and args.disk_sweep.exists():
        for line in args.disk_sweep.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            swept += 1
            # The SAME ``gt_`` basename rule the literal pattern uses.  A raw sweep of the SSD
            # tiers returns 1,702 files, of which 1,433 are ``.u8`` blobs that are not ground
            # truth at all; classifying those would be expensive and meaningless.  Applying one
            # shared naming rule to both discovery legs is what keeps the two denominators
            # comparable.
            if not p.name.startswith("gt_"):
                continue
            if p.is_file() and p.suffix in RAW_SUFFIX_DTYPE:
                candidates.setdefault(str(p.resolve()), [])

    ordered = sorted(candidates)
    print(f"[gl2] raw code-referenced literals: {len(raw_refs)}", flush=True)
    print(f"[gl2] existing raw candidates:      {len(ordered)}", flush=True)

    rulers = Rulers.load()
    print("[gl2] rulers loaded and custody-verified", flush=True)

    rows: list[dict[str, Any]] = []
    for i, s in enumerate(ordered, 1):
        p = Path(s)
        print(f"[gl2] {i}/{len(ordered)} {p}", flush=True)
        row = classify_raw_artifact(p, rulers)
        row["referenced_by"] = sorted(set(candidates.get(s, [])))
        rows.append(row)
        print(
            f"       -> {row['lineage']} ({row['lineage_evidence']}) "
            f"sha {row['sha256'][:12]} {row['bytes']:,} B",
            flush=True,
        )

    census: dict[str, Any] = {
        "schema": "ddm_gl2_raw_gt_lineage_census_v1",
        "produced_by": "experiments/ddm_gl2_raw_gt_lineage_census.py",
        "utc_started": started,
        "utc_finished": _utc(),
        "git_head": _git_head(),
        "rulers": {
            "dali": {"path": str(rulers.dali_seg.shape), "note": "see ddm_gl1 RULER_DALI"},
        },
        "decisive_margin": DECISIVE_MARGIN,
        "denominators": {
            "raw_code_referenced_literals": len(raw_refs),
            "disk_sweep_lines": swept,
            "existing_raw_candidates": len(ordered),
            "classified": len(rows),
            "capped_out": 0,
        },
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "artifacts": rows,
    }

    subject = args.compare_subject
    if subject is None:
        # NAMED explicitly rather than indexed off KNOWN_TARGET_ROOTS: an index into a tuple whose
        # order is an implementation detail silently pointed this at a non-existent path in this
        # tool's own first run, and the comparison stage then skipped without saying so.
        default_subject = _LEVER_B_SURFACE_ROOT / "targets_n600" / "gt_segnet_argmax.u8"
        subject = default_subject if default_subject.is_file() else None
    if args.compare_against and (subject is None or not subject.is_file()):
        raise SystemExit(
            f"--compare-against given but the subject artifact is missing: {subject}. "
            "Refusing to skip the comparison silently."
        )
    if args.compare_against and subject and subject.is_file() and args.compare_against.is_file():
        print(f"[gl2] characterising {subject.name} vs {args.compare_against.name}", flush=True)
        census["disagreement"] = characterise_disagreement(
            subject,
            args.compare_against,
            out_dir / "disagreement",
            a_label="lever_b gt_segnet_argmax.u8",
            b_label=args.compare_against.name,
        )
        d = census["disagreement"]
        print(
            f"[gl2] differing sites {d['differing_sites']:,} / {d['total_sites']:,} "
            f"= {d['s_units_at_100x']:.6f} S units",
            flush=True,
        )

    dest = out_dir / "GL2_RAW_GT_LINEAGE_CENSUS.json"
    dest.write_text(json.dumps(census, indent=2))
    print(f"[gl2] census -> {dest}", flush=True)

    if args.merge_into_registry is not None:
        result = merge_into_registry(census, args.merge_into_registry)
        census["registry_merge"] = {**result, "registry_path": str(args.merge_into_registry)}
        dest.write_text(json.dumps(census, indent=2))
        print(
            f"[gl2] registry merge -> {args.merge_into_registry}: "
            f"{len(result['added'])} added, {len(result['widened'])} widened, "
            f"{result['total']} total entries",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
