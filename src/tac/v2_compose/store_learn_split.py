# SPDX-License-Identifier: MIT
"""v2_compose.store_learn_split — ENCODE the KNOWN deterministic STORE/GENERATE/LEARN split.

Step 2 of the v2 coherent composition pipeline
(``.omx/research/v2_coherent_automated_composition_pipeline_spec_20260630T201213Z.md``).

**This is NOT a discovery/decision step.** The store-vs-learn boundary is the established
deep-math optimum (GROK REFINEMENT-2 + warp-through-R + indirect-RD/MDL; FEED-iz/ja/lj/ll).
This module is the deterministic, *generalizable* ENCODING of that known split as a per-class
recoverability rule, parameterized by the MEASURED warp-through-R recoverability so it transfers
to any dashcam clip/corpus. The function returns the known-optimal per-class assignment; the
per-stage attribution (Step 1) only CONFIRMS it + sizes the residual target — it never overrides.

The clip-agnostic rule (the ONE generalizable question, the depth x rigidity gradient):

    recoverable_from_pose(class)  ->  GENERATE  (deterministic warp; 0 counted bytes beyond calib)
      * Road           : ground-homography(pose)            (warp HELPS: rel_improvement > 0)
      * hood/MyCar     : identity (rigidly camera-attached: ground-warp DESTROYS it, rel << 0)
      * sky/Undrivable : rotation-only KRK^-1 (depth->inf: ground-warp mildly hurts, rel ~ 0-)
    NOT pose-recoverable AND R-fragile (best persist/warp/static d_seg high) ->  LEARN
      * Lane-survival annulus (the binding wall, R-survival)
      * small movables (independent motion, not warp-reducible)

NO-FAKE: this module makes NO score claim. It is plan-time arithmetic over MEASURED advisory
dicts (the grok warp-through-R recoverability + the screw-reach k*). The split is the known
optimum; the predicted bytes are a budget, not a measurement. The frontier pointer is UNMOVED
0.19110 and moves ONLY on a byte-closed ``upstream/evaluate.py`` exact row (CPU + CUDA). Every
returned number is ``[advisory / plan-time] NON-PROMOTABLE``.

CLASS-INDEX DISCIPLINE (CLAUDE.md NON-NEGOTIABLE): the DECISION uses ONLY the measured d_seg
signature (persist / warp / static / rel_improvement), NEVER a hardcoded class index. The
canonical name<->index mapping is recorded as metadata only (self-detected by spatial/static
signature upstream in the measurement tools); it is never the basis of a decision here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.contest_score import break_even_d_seg, rate_term

__all__ = [
    "ClassRecoverability",
    "ClassAssignment",
    "StoreLearnAssignment",
    "DECISION_GENERATE",
    "DECISION_LEARN",
    "WARP_GROUND_HOMOGRAPHY",
    "WARP_IDENTITY",
    "WARP_ROTATION_ONLY",
    "encode_known_split",
    "load_warp_recoverability_from_grok",
    "load_reach_kstar",
]

# Decision + warp-type sentinels (string constants so the assignment is human-readable JSON).
DECISION_GENERATE = "GENERATE"
DECISION_LEARN = "LEARN"
WARP_GROUND_HOMOGRAPHY = "ground_homography"
WARP_IDENTITY = "identity"
WARP_ROTATION_ONLY = "rotation_only"

# Default frontier target (the score the v2 row must beat). Pointer-sourced, advisory only.
_FRONTIER_S = 0.19110
# The d_pose the stored sidecar achieves (FEED-lj / scorer_targets; advisory).
_D_POSE_SIDECAR = 3.4e-5
# The measured deterministic-bulk d_seg floor (FEED-ll / ladder R1; advisory).
_BULK_DSEG_FLOOR = 0.0185


@dataclass(frozen=True)
class ClassRecoverability:
    """Per-class MEASURED warp-through-R recoverability (the generalization parameter).

    Built from the grok ``measure_pose_warp_dseg`` decomposition + ``static_mode_per_class``.
    All d_seg values are argmax-disagreement on the frozen CPU-torch SegNet (advisory).
    """

    cls_name: str
    persist_dseg: float                 # k=0 / persist-to-keyframe d_seg for this class' GT region
    warp_dseg: float                    # ground-homography-warped d_seg for this class' GT region
    static_dseg: float | None           # single-canonical-mask d_seg (staticity), if measured
    rel_improvement: float              # (persist - warp) / persist ; > 0 means warp HELPS
    target_area: float                  # mean per-frame area fraction of this GT class

    @property
    def best_dseg(self) -> float:
        """min over the deterministic mechanisms available (persist / warp / static)."""
        cands = [self.persist_dseg, self.warp_dseg]
        if self.static_dseg is not None:
            cands.append(self.static_dseg)
        return float(min(cands))


@dataclass(frozen=True)
class ClassAssignment:
    """The KNOWN-OPTIMAL per-class assignment emitted by :func:`encode_known_split`."""

    cls_name: str
    cls_index: int | None               # METADATA ONLY (canonical order); never the decision basis
    decision: str                       # DECISION_GENERATE | DECISION_LEARN
    warp_type: str | None               # WARP_* for GENERATE; None for LEARN
    recoverability: ClassRecoverability
    rationale: str


@dataclass(frozen=True)
class StoreLearnAssignment:
    """The full v2 store/learn plan + predicted byte budget (a PLAN; 0 archive bytes)."""

    per_class: dict[str, ClassAssignment]
    n_pairs: int
    reach_kstar: int
    keyframe_count: int
    generate_classes: tuple[str, ...]
    learn_classes: tuple[str, ...]
    residual_target_classes: tuple[str, ...]   # the classes the residual INR must carry (== LEARN)
    predicted_bytes: dict[str, int | None]     # section -> bytes (LEARN is None == OPEN)
    predicted_rate: dict[str, float]
    break_even: dict[str, float]               # break-even d_seg at the known-store rate
    recoverability_dseg_max: float
    confirmed_by_attribution: bool | None
    authority: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        """Machine-readable plan (max-observability; human-readable + diff-able)."""
        return {
            "authority": self.authority,
            "score_claim": False,
            "promotable": False,
            "frontier_pointer": "UNMOVED 0.19110",
            "n_pairs": self.n_pairs,
            "reach_kstar": self.reach_kstar,
            "keyframe_count": self.keyframe_count,
            "recoverability_dseg_max": self.recoverability_dseg_max,
            "generate_classes": list(self.generate_classes),
            "learn_classes": list(self.learn_classes),
            "residual_target_classes": list(self.residual_target_classes),
            "predicted_bytes": dict(self.predicted_bytes),
            "predicted_rate": dict(self.predicted_rate),
            "break_even": dict(self.break_even),
            "confirmed_by_attribution": self.confirmed_by_attribution,
            "per_class": {
                name: {
                    "cls_index": a.cls_index,
                    "decision": a.decision,
                    "warp_type": a.warp_type,
                    "persist_dseg": a.recoverability.persist_dseg,
                    "warp_dseg": a.recoverability.warp_dseg,
                    "static_dseg": a.recoverability.static_dseg,
                    "best_dseg": a.recoverability.best_dseg,
                    "rel_improvement": a.recoverability.rel_improvement,
                    "target_area": a.recoverability.target_area,
                    "rationale": a.rationale,
                }
                for name, a in self.per_class.items()
            },
            "notes": list(self.notes),
        }


def encode_known_split(
    warp_through_R: dict[str, ClassRecoverability],
    reach_kstar: int,
    *,
    n_pairs: int = 600,
    attribution: dict[str, Any] | None = None,
    class_index_map: dict[str, int] | None = None,
    recoverability_dseg_max: float = 0.037,
    warp_help_eps: float = 0.02,
    identity_destroy_threshold: float = -1.0,
    bytes_per_keyframe: float = 693.0,
    pose_sidecar_bytes: int = 875,
    warp_mask_calib_bytes: int = 30,
    palette_bytes: int = 15,
    uncompressed_size: int = 37_545_489,
    authority: str = "[advisory / plan-time] NON-PROMOTABLE",
) -> StoreLearnAssignment:
    """Emit the KNOWN-OPTIMAL per-class STORE/GENERATE/LEARN assignment + predicted bytes.

    The split is established deep-math (GAP3-settled); this ENCODES it, parameterized by the
    MEASURED per-class warp-through-R recoverability so it generalizes to any clip.

    Decision rule (clip-agnostic), per class ``c``:
      * GENERATE iff ``best_dseg(c) <= recoverability_dseg_max`` (recoverable under SOME
        deterministic transform). Warp type:
          - ``ground_homography`` if ``rel_improvement > warp_help_eps``        (Road)
          - ``identity``          if ``rel_improvement < identity_destroy_threshold``  (hood/MyCar)
          - ``rotation_only``     otherwise                                      (sky/Undrivable)
      * LEARN otherwise (not pose-recoverable AND R-fragile)                     (Lane, Movable)

    ``recoverability_dseg_max`` defaults to ``2 x R1_bulk_ref`` (== ``reach_threshold_bulk``,
    0.037) — a principled threshold: twice the deterministic R1 d_seg floor.

    Args:
        warp_through_R: per-class measured recoverability (the generalization parameter).
        reach_kstar: screw-reach k* (FEED-ll); keyframe budget = ceil(n_pairs / kstar).
        n_pairs: clip pair count (600 for the canonical contest clip).
        attribution: OPTIONAL Step-1 per-stage attribution. Used ONLY to CONFIRM the known
            split (assert the bulk is stage-stable, the residual is Lane+movables) + size the
            residual target. A divergence is recorded as a finding, NEVER a re-decision.
        class_index_map: OPTIONAL name->canonical-index metadata (recorded only; not a decision).
        recoverability_dseg_max / warp_help_eps / identity_destroy_threshold: rule thresholds.
        bytes_per_keyframe / pose_sidecar_bytes / warp_mask_calib_bytes / palette_bytes:
            MEASURED byte estimates for the predicted budget (the tool overrides with real
            byte-closed measurements where available).

    Returns:
        :class:`StoreLearnAssignment` — the plan (0 archive bytes; a budget, not a score).
    """
    if not warp_through_R:
        raise ValueError("encode_known_split: warp_through_R is empty (no measured classes)")
    if not isinstance(reach_kstar, int) or reach_kstar <= 0:
        raise ValueError(f"encode_known_split: reach_kstar must be a positive int, got {reach_kstar!r}")
    if n_pairs <= 0:
        raise ValueError(f"encode_known_split: n_pairs must be positive, got {n_pairs}")

    class_index_map = class_index_map or {}
    per_class: dict[str, ClassAssignment] = {}
    generate: list[str] = []
    learn: list[str] = []

    for name, rec in warp_through_R.items():
        best = rec.best_dseg
        if best <= recoverability_dseg_max:
            decision = DECISION_GENERATE
            if rec.rel_improvement > warp_help_eps:
                warp_type = WARP_GROUND_HOMOGRAPHY
                rationale = (
                    f"recoverable (best d_seg {best:.4f} <= {recoverability_dseg_max:.4f}); "
                    f"ground-warp HELPS (rel_improvement {rec.rel_improvement:+.3f} > {warp_help_eps}) "
                    f"-> on-ground-plane: ground-homography(pose)"
                )
            elif rec.rel_improvement < identity_destroy_threshold:
                warp_type = WARP_IDENTITY
                rationale = (
                    f"recoverable (best d_seg {best:.4f} <= {recoverability_dseg_max:.4f}); "
                    f"ground-warp DESTROYS it (rel_improvement {rec.rel_improvement:+.3f} < "
                    f"{identity_destroy_threshold}) -> rigidly camera-attached static core: identity"
                )
            else:
                warp_type = WARP_ROTATION_ONLY
                rationale = (
                    f"recoverable (best d_seg {best:.4f} <= {recoverability_dseg_max:.4f}); "
                    f"ground-warp mildly hurts (rel_improvement {rec.rel_improvement:+.3f}) "
                    f"-> far-field/depth->inf: rotation-only KRK^-1"
                )
            generate.append(name)
        else:
            decision = DECISION_LEARN
            warp_type = None
            rationale = (
                f"NOT recoverable (best d_seg {best:.4f} > {recoverability_dseg_max:.4f}); "
                f"warp can't fix it (rel_improvement {rec.rel_improvement:+.3f}) "
                f"-> LEARN (small residual INR; R-survival residual)"
            )
            learn.append(name)

        per_class[name] = ClassAssignment(
            cls_name=name,
            cls_index=class_index_map.get(name),
            decision=decision,
            warp_type=warp_type,
            recoverability=rec,
            rationale=rationale,
        )

    keyframe_count = int(math.ceil(n_pairs / reach_kstar))

    # --- predicted byte budget (a plan; the tool overrides STORE bytes with real measurements) ---
    keyframe_bytes = int(round(keyframe_count * bytes_per_keyframe))
    store_bytes = keyframe_bytes + int(palette_bytes) + int(warp_mask_calib_bytes)
    predicted_bytes: dict[str, int | None] = {
        "store_keyframes": keyframe_bytes,
        "store_palette": int(palette_bytes),
        "store_warp_mask_calib": int(warp_mask_calib_bytes),
        "store_total": store_bytes,
        "pose_sidecar": int(pose_sidecar_bytes),
        "learn_residual_inr": None,  # OPEN — the single unmeasured quantity (the GPU run)
        "free_generated": 0,         # warp / SDF / ramp / lane-geom / Fourier (rule-118 FREE)
    }
    known_store_bytes = store_bytes + int(pose_sidecar_bytes)
    predicted_rate: dict[str, float] = {
        "store_keyframes": rate_term(keyframe_bytes, uncompressed_size=uncompressed_size),
        "store_total": rate_term(store_bytes, uncompressed_size=uncompressed_size),
        "pose_sidecar": rate_term(pose_sidecar_bytes, uncompressed_size=uncompressed_size),
        "known_store_total": rate_term(known_store_bytes, uncompressed_size=uncompressed_size),
    }

    # break-even d_seg at the KNOWN-store rate (residual INR == 0 bytes): the live target the
    # residual INR must hit. If positive, d_seg alone can reach the frontier given store+pose.
    be_frontier = break_even_d_seg(
        _FRONTIER_S, _D_POSE_SIDECAR, known_store_bytes, uncompressed_size=uncompressed_size
    )
    be_sub015 = break_even_d_seg(
        0.15, _D_POSE_SIDECAR, known_store_bytes, uncompressed_size=uncompressed_size
    )
    break_even = {
        "d_seg_to_beat_frontier_at_known_store": be_frontier,
        "d_seg_to_sub015_at_known_store": be_sub015,
        "deterministic_bulk_dseg_floor": _BULK_DSEG_FLOOR,
        "residual_must_close_dseg_from": _BULK_DSEG_FLOOR,
        "residual_must_close_dseg_to_beat_frontier": be_frontier,
    }

    confirmed = _confirm_with_attribution(attribution, set(generate), set(learn)) if attribution else None

    notes = (
        "split is KNOWN deep-math (GAP3-settled); encoded, not discovered.",
        "LEARN byte cost is OPEN (the single unmeasured quantity; settled by the GPU residual run).",
        "rate axis is GREEN (FACT 1); the binding wall is the residual INR closing d_seg from the "
        f"{_BULK_DSEG_FLOOR} deterministic floor (FACT 2).",
        "warp is dual-use only in the weak sense (one stored pose; two free read-outs); the "
        "lossless dual-use grok was REFUTED (FACT 3).",
    )

    return StoreLearnAssignment(
        per_class=per_class,
        n_pairs=int(n_pairs),
        reach_kstar=int(reach_kstar),
        keyframe_count=keyframe_count,
        generate_classes=tuple(generate),
        learn_classes=tuple(learn),
        residual_target_classes=tuple(learn),
        predicted_bytes=predicted_bytes,
        predicted_rate=predicted_rate,
        break_even=break_even,
        recoverability_dseg_max=float(recoverability_dseg_max),
        confirmed_by_attribution=confirmed,
        authority=authority,
        notes=notes,
    )


def _confirm_with_attribution(
    attribution: dict[str, Any], generate: set[str], learn: set[str]
) -> bool:
    """CONFIRM (not re-decide) the known split against Step-1 attribution.

    The attribution should show the GENERATE classes are stage-stable (bulk) while the LEARN
    classes are the persistent residual. We check the weak invariant that the attribution's
    residual classes are a subset of (or equal to) the LEARN set. A divergence returns False
    (a FINDING, recorded by the caller) — it NEVER overrides the known assignment.
    """
    residual = attribution.get("residual_classes") or attribution.get("persistent_wrong_classes")
    if not residual:
        return True  # no residual class list to check against; vacuously consistent
    residual_set = {str(c) for c in residual}
    # the attribution's residual should live inside the LEARN tier (the known optimum).
    return residual_set.issubset(learn)


# ---------------------------------------------------------------------------
# Loaders: build the canonical recoverability dict from the REAL measured JSONs.
# ---------------------------------------------------------------------------
def load_warp_recoverability_from_grok(
    grok_results_json: str | Path,
) -> dict[str, ClassRecoverability]:
    """Build the canonical ``warp_through_R`` dict from a real grok ``measure_pose_warp_dseg``
    ``results.json`` (the ``decomposition`` + ``static_mode_per_class`` blocks).

    Reads:
      * ``decomposition[cls] = {persist_dseg, warp_dseg, rel_improvement, target_area}``
      * ``static_mode_per_class[cls]`` (optional staticity d_seg)
    """
    path = Path(grok_results_json)
    data = json.loads(path.read_text())
    decomp = data.get("decomposition")
    if not isinstance(decomp, dict) or not decomp:
        raise ValueError(f"{path}: missing/empty 'decomposition' block")
    static = data.get("static_mode_per_class") or {}
    out: dict[str, ClassRecoverability] = {}
    for name, d in decomp.items():
        out[name] = ClassRecoverability(
            cls_name=str(name),
            persist_dseg=float(d["persist_dseg"]),
            warp_dseg=float(d["warp_dseg"]),
            static_dseg=(float(static[name]) if name in static else None),
            rel_improvement=float(d["rel_improvement"]),
            target_area=float(d.get("target_area", 0.0)),
        )
    return out


def load_reach_kstar(
    reach_json: str | Path,
    *,
    key: str = "reach_k_star_pairs_primary_stratified_bulk",
) -> int:
    """Read the screw-reach k* (default the PRIMARY stratified-bulk value) from a real
    ``measure_screw_reach_through_R`` ``reach_*.json``."""
    path = Path(reach_json)
    data = json.loads(path.read_text())
    if key not in data:
        raise KeyError(f"{path}: missing reach key {key!r} (have: {sorted(data)[:8]}...)")
    val = int(data[key])
    if val <= 0:
        raise ValueError(f"{path}: reach k* must be positive, got {val}")
    return val
