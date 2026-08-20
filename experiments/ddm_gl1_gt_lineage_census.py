#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_gl1 — MEASURE the ground-truth cache population and classify each artifact's decode lineage.

WHY THIS EXISTS
---------------
``ddm_pi2`` (2026-08-16, commit ``ed153d0203``) measured that our advisory scorer instrument
reads TWO DIFFERENT GROUND TRUTHS at once: the seg half loads a cached ``gt_argmax_n600.npy``
of DALI/nvdec lineage, while the pose half decodes GT fresh with PyAV every run.  That split
was the entire 21.4x advisory-vs-CUDA pose offset.  pi2's own §6 item 8:

    "I did not verify ``gt_argmax_n600.npy``'s documented provenance.  I established its lineage
    *empirically*.  No producer receipt was located that says so, which means the seg line has
    been relying on an authority-grade cache **by luck, undocumented**."

This tool supplies the missing measurement at population scale.  It is the PRODUCER of the
content-addressed registry in :mod:`tac.gt_lineage`; the registry is seeded from this census,
never hand-typed.

THE RULERS (known-lineage references)
-------------------------------------
Modal job #906 (``experiments/modal_dali_av_gt_cache_diff.py``, 2026-08-09) ran PR130's own
``build_gt_cache_official.py`` TWICE IN ONE CONTAINER ON ONE TESLA T4 -- ``--dataset av`` then
``--dataset dali`` -- so the two caches differ ONLY by the GT decode path, with scorer, host,
driver and clock fixed.  Those two caches are this tool's rulers.  Their lineage is
PRODUCER_DECLARED (the ``--dataset`` argv selected the decoder), not merely inferred.

CLASSIFICATION IS RELATIVE, NEVER ABSOLUTE
------------------------------------------
"Exactly equal to the DALI ruler" is the WRONG criterion: FX4 measured within-DALI-family drift
of 1,644 seg sites between retained-Ada DALI and fresh-T4 DALI.  So an artifact is classified by
the MARGIN between its distance to each ruler, and the margin is always reported.  An artifact
that is not decisively closer to one ruler is ``UNKNOWN_AMBIGUOUS`` -- an honest UNKNOWN is worth
more than a guessed label.

SCORER-FREE BY CONSTRUCTION.  This tool runs no SegNet/PoseNet forward, owns no Metal slot, and
dispatches nothing.  It only differences cached tensors.

Axis: ``[macOS-CPU advisory]`` -- lineage classification, never a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- the two rulers -----------------------------------------------------------------------
# Produced by Modal job #906 in ONE container on ONE Tesla T4, differing only in --dataset.
RULER_DALI = Path("/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")
RULER_AV = Path("/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_av.pt")
# Both are FULL sha256 digests, measured by this tool.  pi2 §2 quotes only 8-char prefixes; a
# constant holding a prefix under a name that reads like a full digest is how prefix-vs-full
# comparisons go wrong, so the full values are recorded here.
RULER_DALI_SHA = "a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994"
RULER_AV_SHA = "837b5852dc71ded7ffd20f59f0e8192a4ce753fe1a7b36882ed8e09f211e1f99"

SEG_SHAPE = (600, 384, 512)
SEG_SITES = 600 * 384 * 512  # 117,964,800

# Decisive-margin factor.  A candidate must be at least this many times closer to one ruler
# than the other to earn a family label.  10x is deliberately conservative: the observed
# within-DALI drift is 1,644 sites while the DALI-vs-AV separation is 20,671 sites, a ratio of
# only 12.6x, so a threshold above ~12 would refuse to classify a legitimately drifted DALI
# cache.  Every classification reports its actual margin so the reader can re-judge.
DECISIVE_MARGIN = 10.0


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _as_seg_u8(arr: np.ndarray) -> np.ndarray:
    """Normalize a seg argmax field to uint8 before ANY byte-level comparison.

    MEASURED DEFECT IN THIS TOOL'S OWN FIRST PASS (ddm_gl1, 2026-08-16): the rulers store seg as
    uint8 while several capstone caches store the identical class labels as int64.  Hashing raw
    bytes therefore matched nothing and mislabelled 24 artifacts UNKNOWN_AMBIGUOUS.  One of them,
    ``gt_targets_n100.pt``, is in fact 100/100 frames bit-identical to the AV ruler.  Class labels
    are 0..4, so the narrowing is lossless; the assertion below keeps it honest if that ever changes.
    """
    if arr.dtype == np.uint8:
        return arr
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi > 255:
        raise ValueError(f"seg field values {lo}..{hi} do not fit uint8; refusing a lossy narrowing")
    return arr.astype(np.uint8)


def _frame_digest(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(_as_seg_u8(frame)).tobytes()).hexdigest()


def _pose_row_digest(row: np.ndarray) -> str:
    """Hash a pose row at float32 -- the dtype both rulers and every consumer actually store."""
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest()


@dataclass
class Rulers:
    """The two known-lineage reference caches, plus per-frame hash indices."""

    dali_seg: np.ndarray
    av_seg: np.ndarray
    dali_pose: np.ndarray
    av_pose: np.ndarray
    dali_frame_hash: dict[str, int] = field(default_factory=dict)
    av_frame_hash: dict[str, int] = field(default_factory=dict)
    dali_pose_hash: dict[str, int] = field(default_factory=dict)
    av_pose_hash: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Rulers:
        import torch

        # CUSTODY: verify the rulers ARE the #906 caches before classifying anything against
        # them.  Every lineage label in the registry inherits its authority from these two
        # files; if the wrong bytes are mounted here, the entire census is silently wrong.
        for path, expect in ((RULER_DALI, RULER_DALI_SHA), (RULER_AV, RULER_AV_SHA)):
            if not path.is_file():
                raise SystemExit(f"ruler missing: {path}")
            got = sha256_file(path)
            if got != expect:
                raise SystemExit(
                    f"RULER CUSTODY FAILURE: {path}\n  expected sha256 {expect}\n  observed sha256 {got}\n"
                    "Refusing to classify against an unverified reference."
                )

        d = torch.load(RULER_DALI, map_location="cpu", weights_only=False)
        a = torch.load(RULER_AV, map_location="cpu", weights_only=False)
        obj = cls(
            dali_seg=d["seg"].numpy(),
            av_seg=a["seg"].numpy(),
            dali_pose=d["pose"].numpy().astype(np.float64),
            av_pose=a["pose"].numpy().astype(np.float64),
        )
        # Per-frame hashes let ANY n-subset be matched by dict lookup instead of an
        # O(n*600) full-field comparison.  This is what makes the n96/n24 subset caches
        # classifiable at all.
        for i in range(obj.dali_seg.shape[0]):
            obj.dali_frame_hash[_frame_digest(obj.dali_seg[i])] = i
        for i in range(obj.av_seg.shape[0]):
            obj.av_frame_hash[_frame_digest(obj.av_seg[i])] = i
        # Pose rows get the same treatment so an n-subset pose table is classifiable too,
        # rather than being written off as "subset matching not implemented".
        for i in range(obj.dali_pose.shape[0]):
            obj.dali_pose_hash[_pose_row_digest(obj.dali_pose[i])] = i
        for i in range(obj.av_pose.shape[0]):
            obj.av_pose_hash[_pose_row_digest(obj.av_pose[i])] = i
        return obj


def _classify_from_distances(
    d_dali: float | None, d_av: float | None, *, metric: str
) -> tuple[str, str, float | None]:
    """Return (lineage, evidence, margin) from two ruler distances.

    ``lineage`` is one of DALI_NVDEC / PYAV_YUV420_TO_RGB / UNKNOWN_AMBIGUOUS.
    """
    if d_dali is None or d_av is None:
        return "UNKNOWN_UNCOMPARABLE", "NONE", None
    if d_dali == 0.0 and d_av == 0.0:
        # Degenerate: the rulers themselves differ, so this cannot happen for real data.
        return "UNKNOWN_AMBIGUOUS", "EMPIRICAL_CONTENT_MATCH", 1.0
    if d_dali == 0.0:
        return "DALI_NVDEC", "EMPIRICAL_EXACT_MATCH", float("inf")
    if d_av == 0.0:
        return "PYAV_YUV420_TO_RGB", "EMPIRICAL_EXACT_MATCH", float("inf")
    margin = d_av / d_dali if d_dali > 0 else float("inf")
    if margin >= DECISIVE_MARGIN:
        return "DALI_NVDEC", f"EMPIRICAL_NEAREST_RULER_{metric}", margin
    if margin <= 1.0 / DECISIVE_MARGIN:
        return "PYAV_YUV420_TO_RGB", f"EMPIRICAL_NEAREST_RULER_{metric}", margin
    return "UNKNOWN_AMBIGUOUS", f"EMPIRICAL_NEAREST_RULER_{metric}", margin


def _npz_member_headers(path: Path) -> dict[str, tuple[tuple[int, ...], Any]]:
    """Read every ``.npz`` member's shape/dtype from its NPY header, without reading the data.

    A GT ``.npz`` can hold multi-GiB RGB frame stacks alongside the small field we want.
    Materializing every member merely to learn its shape would spike RSS by tens of GiB.
    """
    import zipfile

    from numpy.lib import format as npformat

    out: dict[str, tuple[tuple[int, ...], Any]] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            key = name[:-4] if name.endswith(".npy") else name
            try:
                with zf.open(name) as fh:
                    version = npformat.read_magic(fh)
                    shape, _fortran, dtype = npformat._read_array_header(fh, version)
                out[key] = (tuple(shape), dtype)
            except Exception:
                out[key] = ((), "UNREADABLE_HEADER")
    return out


def _extract_fields(path: Path) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Best-effort extraction of (seg_argmax_field, pose_table, description).

    Returns ``(None, None, desc)`` when the artifact holds neither, with ``desc`` recording why.
    """
    desc: dict[str, Any] = {"container": path.suffix, "members": None, "note": None}
    seg = None
    pose = None
    try:
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            desc["members"] = {"<array>": [list(arr.shape), str(arr.dtype)]}
            if arr.ndim == 3 and arr.shape[1:] == SEG_SHAPE[1:]:
                seg = np.asarray(arr)
            elif arr.ndim == 2 and arr.shape[1] == 6:
                pose = np.asarray(arr, dtype=np.float64)
            else:
                desc["note"] = "array shape is neither a (n,384,512) seg field nor an (n,6) pose table"
        elif path.suffix == ".npz":
            # PEEK member headers without materializing them.  Some GT npz files hold
            # multi-GiB RGB frame stacks; reading every member to learn its shape would
            # spike RSS by tens of GiB for no information gain.
            heads = _npz_member_headers(path)
            desc["members"] = {k: [list(shp), str(dt)] for k, (shp, dt) in heads.items()}
            seg_key = next(
                (k for k, (shp, _) in heads.items() if len(shp) == 3 and tuple(shp[1:]) == SEG_SHAPE[1:]),
                None,
            )
            pose_key = next(
                (k for k, (shp, _) in heads.items() if len(shp) == 2 and shp[1] == 6), None
            )
            if seg_key is not None or pose_key is not None:
                with np.load(path, allow_pickle=False) as z:
                    if seg_key is not None:
                        seg = np.asarray(z[seg_key])
                    if pose_key is not None:
                        pose = np.asarray(z[pose_key], dtype=np.float64)
            else:
                desc["note"] = "no member matched a seg field or a pose table"
        elif path.suffix in (".pt", ".pth"):
            import torch

            # Prefer mmap so a multi-GiB frame stack is paged, not resident.  Older
            # non-zipfile checkpoints refuse mmap; fall back to a normal load for those.
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                desc["load_mode"] = "mmap"
            except Exception:
                obj = torch.load(path, map_location="cpu", weights_only=False)
                desc["load_mode"] = "full"
            if isinstance(obj, dict):
                desc["members"] = {}
                for k, v in obj.items():
                    shp = list(getattr(v, "shape", ()) or ())
                    desc["members"][str(k)] = [shp, str(getattr(v, "dtype", type(v).__name__))]
                    if hasattr(v, "numpy"):
                        arr = v.numpy()
                        if seg is None and arr.ndim == 3 and arr.shape[1:] == SEG_SHAPE[1:]:
                            seg = arr
                        elif pose is None and arr.ndim == 2 and arr.shape[1] == 6:
                            pose = arr.astype(np.float64)
            elif hasattr(obj, "numpy"):
                # A bare tensor is a perfectly ordinary way to store a GT pose table or seg
                # field.  Rejecting it as "not a dict" manufactured 5 false UNKNOWNs in this
                # tool's own first pass, including a (600,6) pose table.
                arr = obj.numpy()
                desc["members"] = {"<bare tensor>": [list(arr.shape), str(arr.dtype)]}
                if arr.ndim == 3 and arr.shape[1:] == SEG_SHAPE[1:]:
                    seg = arr
                elif arr.ndim == 2 and arr.shape[1] == 6:
                    pose = arr.astype(np.float64)
                else:
                    desc["note"] = f"bare tensor of shape {list(arr.shape)} is neither a seg field nor a pose table"
            else:
                desc["note"] = f"container is {type(obj).__name__}, not a dict"
        else:
            desc["note"] = f"unhandled suffix {path.suffix}"
    except Exception as exc:
        desc["note"] = f"load failed: {type(exc).__name__}: {exc}"
    return seg, pose, desc


def classify_artifact(path: Path, rulers: Rulers) -> dict[str, Any]:
    """Measure one artifact's lineage against both rulers.  Never raises."""
    st = path.stat()
    row: dict[str, Any] = {
        "path": str(path),
        "basename": path.name,
        "bytes": st.st_size,
        "sha256": sha256_file(path),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
    }
    seg, pose, desc = _extract_fields(path)
    row["container"] = desc

    if seg is None and pose is None:
        row["lineage"] = "UNKNOWN_UNCOMPARABLE"
        row["lineage_evidence"] = "NONE"
        row["lineage_reason"] = desc.get("note") or "artifact holds neither a seg field nor a pose table"
        return row

    seg_res: dict[str, Any] | None = None
    subset_index_map: list[int] | None = None
    if seg is not None:
        seg = _as_seg_u8(seg)
        n = seg.shape[0]
        if n == rulers.dali_seg.shape[0]:
            d_dali = int(np.count_nonzero(seg != rulers.dali_seg))
            d_av = int(np.count_nonzero(seg != rulers.av_seg))
            lin, ev, margin = _classify_from_distances(float(d_dali), float(d_av), metric="SEG_SITES")
            seg_res = {
                "n_frames": n,
                "differing_sites_vs_dali": d_dali,
                "differing_sites_vs_av": d_av,
                "total_sites": int(seg.size),
                "lineage": lin,
                "evidence": ev,
                "margin": margin,
            }
        else:
            # Subset: match every frame by hash against both ruler indices.
            hit_d = hit_a = 0
            idx_d: list[int] = []
            idx_a: list[int] = []
            for i in range(n):
                fh = _frame_digest(seg[i])
                if fh in rulers.dali_frame_hash:
                    hit_d += 1
                    idx_d.append(rulers.dali_frame_hash[fh])
                if fh in rulers.av_frame_hash:
                    hit_a += 1
                    idx_a.append(rulers.av_frame_hash[fh])
            # A DECISIVE MAJORITY is evidence, not ambiguity.  Requiring every frame to match
            # wrote off four MLX-fleet caches that scored 399/400, 199/200, 95/96 and 23/24
            # against the AV ruler and 0 against DALI.  Calling that "unknown" discards a
            # decisive measurement and files phantom debt for the next arm to chase.
            share_d, share_a = hit_d / n, hit_a / n
            if hit_d == n and hit_a < n:
                lin, ev, margin = "DALI_NVDEC", "EMPIRICAL_EXACT_FRAME_SUBSET", float("inf")
            elif hit_a == n and hit_d < n:
                lin, ev, margin = "PYAV_YUV420_TO_RGB", "EMPIRICAL_EXACT_FRAME_SUBSET", float("inf")
            elif share_d >= 0.9 and hit_d >= max(1, hit_a) * DECISIVE_MARGIN:
                lin, ev, margin = "DALI_NVDEC", "EMPIRICAL_MAJORITY_FRAME_SUBSET", share_d
            elif share_a >= 0.9 and hit_a >= max(1, hit_d) * DECISIVE_MARGIN:
                lin, ev, margin = "PYAV_YUV420_TO_RGB", "EMPIRICAL_MAJORITY_FRAME_SUBSET", share_a
            else:
                lin, ev, margin = (
                    "UNKNOWN_AMBIGUOUS",
                    "EMPIRICAL_FRAME_SUBSET_PARTIAL",
                    None,
                )
            seg_res = {
                "n_frames": n,
                "frames_exactly_in_dali_ruler": hit_d,
                "frames_exactly_in_av_ruler": hit_a,
                "unmatched_frames": n - max(hit_d, hit_a),
                "matched_dali_indices_head": idx_d[:8],
                "lineage": lin,
                "evidence": ev,
                "margin": margin,
            }
            # Carry the recovered pair-index mapping so the pose leg below can be compared by
            # MSE even though the artifact does not record which pairs it holds.
            subset_index_map = idx_d if hit_d >= hit_a else idx_a
    row["seg_leg"] = seg_res

    pose_res: dict[str, Any] | None = None
    if pose is not None and pose.shape[0] == rulers.dali_pose.shape[0]:
        m_dali = float(np.mean((pose - rulers.dali_pose) ** 2))
        m_av = float(np.mean((pose - rulers.av_pose) ** 2))
        lin, ev, margin = _classify_from_distances(m_dali, m_av, metric="POSE_MSE")
        pose_res = {
            "n_pairs": int(pose.shape[0]),
            "mse_vs_dali": m_dali,
            "mse_vs_av": m_av,
            "lineage": lin,
            "evidence": ev,
            "margin": margin,
        }
    elif pose is not None:
        # Subset pose table: match every row by hash against both ruler indices, exactly as the
        # seg subset path does.  Leaving this as "not implemented" would have manufactured
        # UNKNOWNs that are really just unmeasured.
        n = int(pose.shape[0])
        hit_d = sum(1 for i in range(n) if _pose_row_digest(pose[i]) in rulers.dali_pose_hash)
        hit_a = sum(1 for i in range(n) if _pose_row_digest(pose[i]) in rulers.av_pose_hash)
        mse_d = mse_a = None
        if hit_d == n and hit_a < n:
            lin, ev = "DALI_NVDEC", "EMPIRICAL_EXACT_POSE_ROW_SUBSET"
        elif hit_a == n and hit_d < n:
            lin, ev = "PYAV_YUV420_TO_RGB", "EMPIRICAL_EXACT_POSE_ROW_SUBSET"
        elif subset_index_map is not None and len(subset_index_map) == n:
            # Exact float32 row equality is brittle across scorer builds.  The seg leg already
            # recovered WHICH pairs this subset holds, so fall back to the same MSE comparison
            # the n600 path uses, aligned through that mapping.
            sel = np.asarray(subset_index_map, dtype=np.int64)
            mse_d = float(np.mean((pose - rulers.dali_pose[sel]) ** 2))
            mse_a = float(np.mean((pose - rulers.av_pose[sel]) ** 2))
            lin, ev, _m = _classify_from_distances(mse_d, mse_a, metric="POSE_MSE_VIA_SEG_INDEX_MAP")
        else:
            lin, ev = "UNKNOWN_AMBIGUOUS", "EMPIRICAL_POSE_ROW_SUBSET_PARTIAL"
        pose_res = {
            "n_pairs": n,
            "rows_exactly_in_dali_ruler": hit_d,
            "rows_exactly_in_av_ruler": hit_a,
            "mse_vs_dali_via_index_map": mse_d,
            "mse_vs_av_via_index_map": mse_a,
            "lineage": lin,
            "evidence": ev,
            "margin": None,
        }
    row["pose_leg"] = pose_res

    # Roll up.  If both legs classify and DISAGREE, that is exactly the pi2 bug inside one
    # artifact and it must be surfaced, never averaged away.
    legs = [r for r in (seg_res, pose_res) if r and r.get("lineage", "").startswith(("DALI", "PYAV"))]
    lineages = {r["lineage"] for r in legs}
    if not legs:
        row["lineage"] = "UNKNOWN_AMBIGUOUS"
        row["lineage_evidence"] = "EMPIRICAL_INDECISIVE"
        row["lineage_reason"] = "measured against both rulers but no leg reached a decisive margin"
    elif len(lineages) == 1:
        row["lineage"] = legs[0]["lineage"]
        row["lineage_evidence"] = "|".join(sorted({r["evidence"] for r in legs}))
        row["lineage_reason"] = "measured against both #906 rulers"
    else:
        row["lineage"] = "SPLIT_LINEAGE_WITHIN_ARTIFACT"
        row["lineage_evidence"] = "EMPIRICAL_CONTENT_MATCH"
        row["lineage_reason"] = f"seg and pose legs disagree: {sorted(lineages)}"
    return row


def build_registry(census: dict[str, Any]) -> dict[str, Any]:
    """Collapse a census into the CONTENT-ADDRESSED registry consumed by :mod:`tac.gt_lineage`.

    Keyed by sha256, never by path or basename.  ``ddm_gl1`` measured six distinct files named
    ``gt_argmax_n600.npy``, and two of them -- IDENTICAL in name and byte count -- carry
    DIFFERENT lineages.  A name-keyed registry would let one verified file launder the others.
    """
    by_sha: dict[str, dict[str, Any]] = {}
    for row in census["artifacts"]:
        digest = row.get("sha256")
        if not digest:
            continue
        seg = row.get("seg_leg") or {}
        pose = row.get("pose_leg") or {}
        bits: list[str] = []
        if seg:
            if "differing_sites_vs_dali" in seg:
                bits.append(
                    f"seg: {seg['differing_sites_vs_dali']:,} differing sites vs DALI ruler, "
                    f"{seg['differing_sites_vs_av']:,} vs AV ruler, of {seg.get('total_sites', 0):,}"
                )
            else:
                bits.append(
                    f"seg: {seg.get('frames_exactly_in_dali_ruler')}/{seg.get('n_frames')} frames "
                    f"bit-identical to DALI ruler, {seg.get('frames_exactly_in_av_ruler')}/"
                    f"{seg.get('n_frames')} to AV ruler"
                )
        if pose and "mse_vs_dali" in pose:
            bits.append(f"pose: MSE {pose['mse_vs_dali']:.6e} vs DALI ruler, {pose['mse_vs_av']:.6e} vs AV ruler")

        existing = by_sha.get(digest)
        if existing is None:
            by_sha[digest] = {
                "sha256": digest,
                "bytes": row.get("bytes", 0),
                "basename": row.get("basename", ""),
                "lineage": row.get("lineage", "UNKNOWN_UNCOMPARABLE"),
                "lineage_evidence": row.get("lineage_evidence", "NONE"),
                "measurement": "; ".join(bits) or (row.get("lineage_reason") or ""),
                "claim_boundary": _claim_boundary(row),
                "known_paths": [row["path"]],
            }
        else:
            existing["known_paths"].append(row["path"])
    for entry in by_sha.values():
        entry["known_paths"] = sorted(set(entry["known_paths"]))
        # One content can be known by SEVERAL filenames.  ddm_gl1 measured the DALI argmax field
        # stored both as `gt_argmax.npy` and as `gt_argmax_n600.npy`.  Recording only the first
        # basename hid a real cross-lineage name collision from this arm's own first-pass check,
        # so every name the bytes are known by is carried.
        entry["known_basenames"] = sorted({Path(p).name for p in entry["known_paths"]})
    return {
        "schema": "ddm_gl1_gt_lineage_registry_v1",
        "produced_by": "experiments/ddm_gl1_gt_lineage_census.py --emit-registry",
        "produced_utc": census["utc_finished"],
        "git_head": census.get("git_head", ""),
        "rulers": census["rulers"],
        "decisive_margin": census["decisive_margin"],
        "denominators": census["denominators"],
        "keying": "sha256 of file bytes -- NEVER path or basename",
        "axis": census["axis"],
        "score_claim": False,
        "artifacts": [by_sha[k] for k in sorted(by_sha)],
    }


class RegistryMergeConflict(RuntimeError):
    """A sha256 already in the registry classified differently on re-measurement."""


def merge_registry(existing: dict[str, Any], addition: dict[str, Any]) -> dict[str, Any]:
    """Fold newly classified artifacts into an existing registry, additively.

    WHY A MERGE PATH EXISTS (ddm_dg1, 2026-08-20).  The registry is content-addressed
    and rebuilt wholesale from a full census, but a full census needs a disk sweep of
    every GT artifact on both SSD tiers.  When ONE new artifact appears -- the
    2026-08-19 GT-lineage cure materialised ``gt_first6_dali_n600.npy`` and repointed
    ``qs1.GT_POSE`` at it -- the choice was between an expensive full re-census and
    leaving the cure's own target UNREGISTERED.  It was left unregistered, so
    ``assert_gt_lineage`` refused the very table the cure installed: the guard could
    not certify the fix.  This path closes that without inviting a hand-edited
    registry (the parallel-authority failure this module exists to prevent).

    Merge rules, in order of strictness:

    * A sha256 present in BOTH must classify IDENTICALLY.  Content-addressed lineage
      is a function of the bytes; disagreement means one measurement is wrong and is
      raised, never silently resolved by preferring one side.
    * ``known_paths`` / ``known_basenames`` are UNIONED.  One content is legitimately
      known by several names -- that is the collision ``ddm_gl1`` measured and the
      reason the registry is keyed by sha in the first place.
    * The header's ``denominators`` describe the ORIGINAL full census and are kept
      verbatim; an ``amendments`` list records each additive pass so a reader can
      always tell a full census from a full census plus patches.

    Raises:
        RegistryMergeConflict: a shared sha256 carries two different lineages.
    """
    merged = {k: v for k, v in existing.items() if k != "artifacts"}
    by_sha: dict[str, dict[str, Any]] = {
        row["sha256"]: dict(row) for row in existing.get("artifacts", [])
    }
    added: list[str] = []
    updated: list[str] = []
    for row in addition.get("artifacts", []):
        digest = row["sha256"]
        prior = by_sha.get(digest)
        if prior is None:
            by_sha[digest] = dict(row)
            added.append(digest)
            continue
        if prior.get("lineage") != row.get("lineage"):
            raise RegistryMergeConflict(
                f"sha256 {digest} is {prior.get('lineage')} in the existing registry but "
                f"{row.get('lineage')} on re-measurement. Content-addressed lineage must be "
                "a function of the bytes; resolve the disagreement before merging."
            )
        names = sorted(set(prior.get("known_paths", [])) | set(row.get("known_paths", [])))
        if names != prior.get("known_paths"):
            updated.append(digest)
        prior["known_paths"] = names
        prior["known_basenames"] = sorted({Path(p).name for p in names})
    merged["artifacts"] = [by_sha[k] for k in sorted(by_sha)]
    amendments = list(existing.get("amendments", []))
    amendments.append(
        {
            "utc": addition.get("produced_utc", ""),
            "git_head": addition.get("git_head", ""),
            "added_sha256": sorted(added),
            "path_unioned_sha256": sorted(set(updated)),
            "note": (
                "additive merge via --merge-into; denominators above describe the "
                "original full census, not this pass"
            ),
        }
    )
    merged["amendments"] = amendments
    return merged


def _claim_boundary(row: dict[str, Any]) -> str:
    """State exactly how far each lineage label may be pushed.  Vocabulary follows FX4."""
    ev = row.get("lineage_evidence", "NONE")
    lin = row.get("lineage", "")
    if lin == "SPLIT_LINEAGE_WITHIN_ARTIFACT":
        return (
            "This artifact's seg and pose halves classify to DIFFERENT lineages. It reproduces the "
            "ddm_pi2 defect inside a single file and must not be used as a reference until resolved."
        )
    if lin not in ("DALI_NVDEC", "PYAV_YUV420_TO_RGB"):
        return (
            "Lineage NOT established. No producer receipt was located and content comparison against "
            "the #906 rulers was indecisive or impossible. Recorded as UNKNOWN on purpose: a guessed "
            "label here is the exact failure ddm_pi2 surfaced."
        )
    if "EXACT_MATCH" in ev:
        return (
            "Bit-identical to a producer-declared ruler over the full population; content identity is "
            "as strong as empirical evidence gets. It still does not recover the original argv, "
            "package versions, or host."
        )
    if "FRAME_SUBSET" in ev:
        return (
            "Every frame is bit-identical to a frame of a producer-declared ruler. Strong content "
            "identity for the frames present; says nothing about frames absent from this subset."
        )
    return (
        "Classified by nearest ruler with the margin recorded. Content strongly indicates the family "
        "but does not prove the exact historical invocation, and within-family drift is nonzero "
        "(FX4 measured 1,644 seg sites between two DALI builds)."
    )


def enumerate_code_referenced() -> dict[str, list[str]]:
    """Every GT-artifact path literal appearing in our own instruments.

    This is the denominator that matters: artifacts our code can actually load as ground truth.
    """
    pat = re.compile(r"""["']([^"']*gt_[A-Za-z0-9_]*\.(?:npy|npz|pt|pth))["']""")
    out: dict[str, list[str]] = {}
    for sub in ("experiments", "tools", "src"):
        base = REPO_ROOT / sub
        for py in base.rglob("*.py"):
            # Test fixtures name GT files they invent in tmp_path; counting them would inflate
            # the denominator with paths no instrument ever loads.  This tool's own test suite
            # did exactly that (99 -> 100) before the exclusion.
            if "tests" in py.parts or py.name.startswith("test_"):
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in pat.finditer(text):
                out.setdefault(m.group(1), []).append(str(py.relative_to(REPO_ROOT)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--disk-sweep",
        type=Path,
        default=None,
        help="output of a `find ... | xargs stat -f '%z %N'` sweep, one '<bytes> <path>' per line",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="census JSON destination (required unless --from-census rebuilds a registry)",
    )
    ap.add_argument(
        "--max-classify",
        type=int,
        default=0,
        help="cap on artifacts to classify (0 = no cap). A cap is ALWAYS logged with its denominator.",
    )
    ap.add_argument(
        "--emit-registry",
        type=Path,
        default=None,
        help="also write the content-addressed registry consumed by tac.gt_lineage",
    )
    ap.add_argument(
        "--from-census",
        type=Path,
        default=None,
        help="skip measurement and rebuild the registry from an existing census JSON",
    )
    ap.add_argument(
        "--classify-path",
        type=Path,
        action="append",
        default=None,
        help=(
            "classify these specific artifacts instead of running a full sweep; "
            "intended for --merge-into (registering one newly materialised table)"
        ),
    )
    ap.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help=(
            "fold the classified artifacts ADDITIVELY into this existing registry "
            "instead of overwriting it; refuses if a shared sha256 reclassifies"
        ),
    )
    args = ap.parse_args(argv)

    if args.classify_path:
        # Validate BEFORE the expensive classification: loading both 117 MB rulers and
        # measuring against them costs real time, and erroring afterwards wastes it.
        if args.merge_into is None:
            ap.error("--classify-path requires --merge-into (refusing to overwrite a full census)")
        if not args.merge_into.is_file():
            ap.error(f"--merge-into {args.merge_into} does not exist; nothing to merge into")
        rulers = Rulers.load()
        rows = [classify_artifact(Path(p), rulers) for p in args.classify_path]
        for row in rows:
            print(
                f"[gl1] {row.get('basename')} sha={row.get('sha256', '')[:12]} "
                f"-> {row.get('lineage')} ({row.get('lineage_evidence')})",
                flush=True,
            )
        addition = build_registry(
            {
                "artifacts": rows,
                "utc_finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_head": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=REPO_ROOT,
                ).stdout.strip(),
                "rulers": {"dali": {"path": str(RULER_DALI), "sha256": RULER_DALI_SHA},
                           "av": {"path": str(RULER_AV), "sha256": RULER_AV_SHA}},
                "decisive_margin": DECISIVE_MARGIN,
                "denominators": {"classified": len(rows)},
                "axis": "[macOS-CPU advisory] lineage classification -- NEVER a score",
            }
        )
        existing = json.loads(args.merge_into.read_text())
        merged = merge_registry(existing, addition)
        before = len(existing.get("artifacts", []))
        args.merge_into.write_text(json.dumps(merged, indent=2))
        print(
            f"[gl1] merged {args.merge_into}: {before} -> {len(merged['artifacts'])} entries",
            flush=True,
        )
        return 0

    if args.from_census is not None:
        census = json.loads(args.from_census.read_text())
        if args.emit_registry is None:
            ap.error("--from-census requires --emit-registry")
        reg = build_registry(census)
        args.emit_registry.parent.mkdir(parents=True, exist_ok=True)
        args.emit_registry.write_text(json.dumps(reg, indent=2))
        print(f"[gl1] rebuilt registry from {args.from_census} -> {args.emit_registry}", flush=True)
        print(f"[gl1] {len(reg['artifacts'])} distinct sha256 entries", flush=True)
        return 0

    if args.out is None:
        ap.error("--out is required unless --from-census is given")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    code_refs = enumerate_code_referenced()

    disk_paths: list[Path] = []
    if args.disk_sweep and args.disk_sweep.exists():
        for line in args.disk_sweep.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line == "SWEEP_DONE":
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                disk_paths.append(Path(parts[1]))

    # Candidate set = every existing file among (code-referenced literals resolved) U (disk sweep).
    candidates: dict[str, None] = {}
    for lit in code_refs:
        p = Path(lit)
        for cand in (p, REPO_ROOT / lit):
            if cand.is_file():
                candidates[str(cand.resolve())] = None
    for p in disk_paths:
        if p.is_file():
            candidates[str(p.resolve())] = None

    ordered = sorted(candidates)
    denominator = len(ordered)
    capped = ordered if args.max_classify <= 0 else ordered[: args.max_classify]

    print(f"[gl1] code-referenced literals: {len(code_refs)}", flush=True)
    print(f"[gl1] disk-sweep files:        {len(disk_paths)}", flush=True)
    print(f"[gl1] existing candidates:     {denominator}", flush=True)
    print(f"[gl1] classifying:             {len(capped)} of {denominator}", flush=True)

    rulers = Rulers.load()
    print("[gl1] rulers loaded", flush=True)

    rows: list[dict[str, Any]] = []
    for i, s in enumerate(capped, 1):
        p = Path(s)
        try:
            row = classify_artifact(p, rulers)
        except Exception as exc:
            row = {
                "path": s,
                "basename": p.name,
                "lineage": "UNKNOWN_UNCOMPARABLE",
                "lineage_evidence": "NONE",
                "lineage_reason": f"classify raised {type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(
            f"[gl1] {i}/{len(capped)} {row['lineage']:<32} {p.name}  ({row.get('bytes', 0):,} B)",
            flush=True,
        )

    tally: dict[str, int] = {}
    for r in rows:
        tally[r["lineage"]] = tally.get(r["lineage"], 0) + 1

    payload = {
        "schema": "ddm_gl1_gt_lineage_census_v1",
        "arm": "ddm_gl1",
        "utc_started": started,
        "utc_finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU advisory] lineage classification -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip(),
        "rulers": {
            "dali": {"path": str(RULER_DALI), "sha256": RULER_DALI_SHA},
            "av": {"path": str(RULER_AV), "sha256": RULER_AV_SHA},
            "provenance": (
                "Modal job #906 experiments/modal_dali_av_gt_cache_diff.py, 2026-08-09: PR130's "
                "build_gt_cache_official.py run TWICE in ONE container on ONE Tesla T4, "
                "--dataset av then --dataset dali. Lineage is PRODUCER_DECLARED by that argv."
            ),
        },
        "decisive_margin": DECISIVE_MARGIN,
        "denominators": {
            "code_referenced_path_literals": len(code_refs),
            "disk_sweep_files": len(disk_paths),
            "existing_candidates": denominator,
            "classified": len(capped),
            "capped_out": denominator - len(capped),
        },
        "tally": tally,
        "code_references": {k: sorted(set(v)) for k, v in sorted(code_refs.items())},
        "artifacts": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"[gl1] wrote {args.out}", flush=True)
    print(f"[gl1] tally: {json.dumps(tally, indent=2)}", flush=True)

    if args.emit_registry is not None:
        reg = build_registry(payload)
        args.emit_registry.parent.mkdir(parents=True, exist_ok=True)
        args.emit_registry.write_text(json.dumps(reg, indent=2))
        print(
            f"[gl1] wrote registry {args.emit_registry} ({len(reg['artifacts'])} distinct sha256)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
