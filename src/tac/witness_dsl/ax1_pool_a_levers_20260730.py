"""ddm_pa1b — Pool-A lever LOGIC: the two ax1 stubs realized as tested pure functions.

The gc10 Pool-A race (op-routable 3, build #793) needs REAL arms.  This module holds the
scorer-free, numpy-only, fully-tested LOGIC for the two ax1 token-byte levers that the tr1
trainer wires (the ph3 fold-and-delete pattern dw1 used for QA75: the DESIGN-stubs in
``ax1_derived_levers_20260730`` are superseded by the real factories in
``spec_tr1_renderer_20260728`` + this logic module):

  (a) **ax1 §2a margin-coupled token quant** — per-cell EFFECTIVE quant precision allocated
      by the MEASURED QA80 exact flip-distance field (custody
      ``/Volumes/VertigoDataTier/pact/ddm_zb1_qa80_field_20260730``).  The flip-distance law
      d=|m|/‖Δw‖ (segnet-fractal) says quant noise is seg-safe where the flip distance is
      LARGE and dangerous where it is SMALL (100% of realized flips are in the bottom
      GT-margin decile).  So spend precision where the field is tight, coarsen where it is
      slack.  Realized as a per-cell effective-level map ≤ ``token_quant_levels`` stored in the
      SAME uint8 lattice — coarse cells snap to a sublattice → SMEVR codes them with lower
      entropy → the bytes materialize through the SHIPPED coder (identical mechanism to the
      QA84 rowband tie).  The allocation law is a **rank transform of the measured field's own
      flip-mass order statistic** — NO bare α/β constant; the field IS the law.

  (b) **ax1 §4a/§5 delta group-sparsity** — group-L2 (group-lasso) shrinkage on the per-pair
      token deltas.  op1 P2 measured **98.806% image-stationary** flip mass, but the trainer
      has NO delta-shrinkage force — SMEVR exploits stationarity only at coding time.  Shrink
      whole-cell deltas at the SOURCE so bytes fall through the coder's zero-delta runs.  The
      weight field is ξ-informed (§5): relax the shrinkage where the ego-motion prior says
      deltas legitimately move (the movable band + lane corridor — DERIVED from the QA80
      ``winner``-class field), tighten it on the static mass.

BYTE-IDENTITY: both levers are default-OFF; when off, ``margin_coupled_level_map`` returns the
uniform full-level map and ``delta_group_sparsity_penalty`` is never called — the trainer forward
+ byte-close are bit-identical to the control (structurally proven by the off-identity tests).
NEITHER lever adds a TRAINABLE parameter: the level map + weight field are FIXED buffers
(``_FixedBank``, not in ``trainable_parameters`` → not checkpointed → not in the EMA shadow →
byte-identical resume by construction; the resume-registry/EMA obligation is satisfied vacuously
and documented here rather than by adding state).

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED.  numpy reference authority; score_claim=False;
every number a burn produces from these levers is [macOS-CPU advisory] until byte-closed by MAIN.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Canonical QA80 exact flip-distance field custody (zb1 item-1, MEASURED n600, 600/600 sha).
QA80_FIELD_CUSTODY = "/Volumes/VertigoDataTier/pact/ddm_zb1_qa80_field_20260730"

#: comma10k canonical class order (MEASURED, DAG; NEVER luma-sort re-derived).  The ξ-informed
#: delta weight relaxes on the DYNAMIC classes (lane markings move as dashes; movable objects
#: translate) and tightens on the STATIC mass (road/undrivable/mycar-hood).
CLASS_ROAD, CLASS_LANE, CLASS_UNDRIVABLE, CLASS_MOVABLE, CLASS_MYCAR = 0, 1, 2, 3, 4
DYNAMIC_CLASSES: tuple[int, ...] = (CLASS_LANE, CLASS_MOVABLE)  # ego-motion prior (ax1 §5)


class QA80FieldError(ValueError):
    """The QA80 field custody is missing, malformed, or sha-mismatched (fail-closed)."""


# ---------------------------------------------------------------------------
# Typed QA80 field loader (streams the per-pair custody; sha-verified).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellFieldAggregate:
    """Per-CELL aggregate of the QA80 exact flip-distance field at a token-grid geometry.

    ``flip_mass``  (gh,gw) float64: mean over pairs of the in-cell flip-proximity mass
        (fraction of pixels whose flip distance is below the field's own median scale q50) —
        HIGH where flips are likely (needs precision), LOW where slack (can coarsen).
    ``min_distance`` (gh,gw) float64: the tightest (min over pixels & pairs) flip distance in
        the cell — the seg-safety headroom (a single flipped pixel there costs d_seg).
    ``dynamic_frac`` (gh,gw) float64: fraction of (pixel,pair) samples whose winner class is a
        DYNAMIC class (lane/movable) — the ξ-informed delta-relax field (ax1 §5).
    """

    grid_h: int
    grid_w: int
    downsample: int
    n_pairs: int
    q50_scale: float
    flip_mass: np.ndarray
    min_distance: np.ndarray
    dynamic_frac: np.ndarray
    field_custody: str

    def summary(self) -> dict[str, object]:
        return {
            "grid": [self.grid_h, self.grid_w], "downsample": self.downsample,
            "n_pairs": self.n_pairs, "q50_scale": round(float(self.q50_scale), 6),
            "flip_mass_mean": round(float(self.flip_mass.mean()), 6),
            "flip_mass_max": round(float(self.flip_mass.max()), 6),
            "min_distance_min": round(float(self.min_distance.min()), 6),
            "dynamic_frac_mean": round(float(self.dynamic_frac.mean()), 6),
            "field_custody": self.field_custody,
        }


def _cell_reduce(pix: np.ndarray, downsample: int, op: str) -> np.ndarray:
    """Reduce a (H,W) pixel array to (H//d, W//d) by ``op`` over each d×d block."""
    h, w = pix.shape
    gh, gw = h // downsample, w // downsample
    if gh * downsample != h or gw * downsample != w:
        raise QA80FieldError(
            f"field ({h},{w}) not divisible by downsample {downsample} (never-invent geometry)")
    blocks = pix[: gh * downsample, : gw * downsample].reshape(gh, downsample, gw, downsample)
    if op == "mean":
        return blocks.mean(axis=(1, 3))
    if op == "min":
        return blocks.min(axis=(1, 3))
    raise ValueError(f"unknown reduce op {op!r}")


def load_qa80_cell_field(
    grid_h: int, grid_w: int, *, downsample: int, field_custody: str = QA80_FIELD_CUSTODY,
    max_pairs: int | None = None, verify_sha: bool = True,
) -> CellFieldAggregate:
    """Load + aggregate the MEASURED QA80 field custody to per-cell statistics at (grid_h,grid_w).

    Streams the per-pair ``pair-XXXXXX.npz`` (keys ``exact_flip_distance``/``winner``/``runner``),
    sha-verifies each against the manifest (staleness/custody law), and accumulates the per-cell
    aggregates ONLINE (never holds all 600 pairs).  ``max_pairs`` bounds it for a smoke; the full
    n600 field is the authority.  Fail-closed on custody/sha/geometry problems (never-invent)."""
    root = Path(field_custody)
    manifest_path = root / "field_pass_manifest.json"
    if not manifest_path.is_file():
        raise QA80FieldError(f"QA80 field manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    geom = manifest.get("geometry", {})
    seg_h, seg_w = int(geom.get("seg_h", 384)), int(geom.get("seg_w", 512))
    if seg_h // downsample != grid_h or seg_w // downsample != grid_w:
        raise QA80FieldError(
            f"grid ({grid_h},{grid_w}) != field ({seg_h},{seg_w})//{downsample} "
            f"=({seg_h // downsample},{seg_w // downsample}) — fail-closed (never-invent geometry)")
    pairs = manifest["pairs"]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    # Pass 1: the field's own median flip-distance scale q50 (a MEASURED anchor, not a bare
    # constant — the allocation/mass thresholds ride the field's own distribution).  Sampled
    # over a bounded pair subset for O(1) memory; the median is stable (zb1 measured q50 1.8181).
    scale_samples: list[np.ndarray] = []
    for rec in pairs[: min(len(pairs), 16)]:
        d = _load_pair(root, rec, "exact_flip_distance", verify_sha)
        scale_samples.append(d[::8, ::8].ravel())
    q50 = float(np.median(np.concatenate(scale_samples))) if scale_samples else 1.0
    q50 = max(q50, 1e-6)

    mass = np.zeros((grid_h, grid_w), dtype=np.float64)
    min_d = np.full((grid_h, grid_w), np.inf, dtype=np.float64)
    dyn = np.zeros((grid_h, grid_w), dtype=np.float64)
    n = 0
    for rec in pairs:
        d = _load_pair(root, rec, "exact_flip_distance", verify_sha)
        win = _load_pair(root, rec, "winner", verify_sha)
        # proximity-weighted flip mass: relu(1 - d/q50) emphasises the TIGHTEST (at-risk) pixels
        # (d->0 => 1, d>=q50 => 0), anchored to the field's OWN median scale — a continuous,
        # tail-weighted measure (not a median-split indicator which spreads mass too broadly).
        prox = np.maximum(0.0, 1.0 - d.astype(np.float64) / q50)
        mass += _cell_reduce(prox, downsample, "mean")
        min_d = np.minimum(min_d, _cell_reduce(d.astype(np.float64), downsample, "min"))
        is_dyn = np.isin(win, DYNAMIC_CLASSES).astype(np.float64)
        dyn += _cell_reduce(is_dyn, downsample, "mean")
        n += 1
    if n == 0:
        raise QA80FieldError("QA80 field custody has zero pairs")
    return CellFieldAggregate(
        grid_h=grid_h, grid_w=grid_w, downsample=downsample, n_pairs=n, q50_scale=q50,
        flip_mass=mass / n, min_distance=min_d, dynamic_frac=dyn / n,
        field_custody=str(field_custody))


def _load_pair(root: Path, rec: dict, key: str, verify_sha: bool) -> np.ndarray:
    path = root / rec["path"]
    if not path.is_file():
        raise QA80FieldError(f"QA80 pair missing: {path}")
    if verify_sha:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != rec["sha256"]:
            raise QA80FieldError(
                f"QA80 pair sha mismatch {path.name}: {got[:12]} != {rec['sha256'][:12]} "
                "(custody/staleness fail-closed)")
    with np.load(path) as z:
        if key not in z.files:
            raise QA80FieldError(f"QA80 pair {path.name} missing key {key!r}")
        return np.asarray(z[key])


# ---------------------------------------------------------------------------
# (a) margin-coupled token quant — the per-cell effective-level allocation LAW.
# ---------------------------------------------------------------------------
def margin_coupled_level_map(
    flip_mass: np.ndarray, *, base_levels: int, min_levels: int, n_tiers: int = 0,
) -> np.ndarray:
    """Per-cell EFFECTIVE quant level count from the measured flip-mass field (allocation LAW).

    DERIVATION (no bare constant): the per-cell precision is a RANK TRANSFORM of the field's
    own flip-mass order statistic.  Sort cells by flip-mass; the LOW-mass tail (slack, seg-safe
    to coarsen) maps to ``min_levels``, the HIGH-mass tail (100% of flips live here) maps to
    ``base_levels`` (=``token_quant_levels``), linearly in RANK.  The mapping has NO free
    parameter — it is the empirical CDF of the measured field.  ``base_levels``/``min_levels``
    are the config's raced level ladder endpoints, not asserted constants.

    Returns a (gh,gw) int array in [min_levels, base_levels].  When ``min_levels==base_levels``
    (the OFF state) it is uniform ``base_levels`` — byte-identical to the scalar-L control.

    ``n_tiers`` (0 => continuous rank) snaps the allocation to power-of-two-friendly sublattice
    tiers (fewer distinct level counts = simpler byte-close); the tiers are derived from the
    [min_levels, base_levels] span, not hand-picked."""
    if base_levels < 2 or min_levels < 2 or min_levels > base_levels:
        raise ValueError(f"levels: 2<=min({min_levels})<=base({base_levels})")
    if min_levels == base_levels:
        return np.full(flip_mass.shape, base_levels, dtype=np.int64)  # OFF => uniform
    flat = flip_mass.astype(np.float64).ravel()
    # empirical-CDF rank in [0,1] via the field's own order statistic: ties collapse to the SAME
    # rank fraction (so a uniform field => uniform allocation), deterministic + field-derived,
    # NO free parameter (the field IS the law).
    uniq, inv = np.unique(flat, return_inverse=True)
    inv = np.asarray(inv).ravel()
    tier_frac = (inv.astype(np.float64) / float(uniq.size - 1) if uniq.size > 1
                 else np.zeros_like(flat))
    lvl = min_levels + tier_frac * (base_levels - min_levels)
    if n_tiers and n_tiers >= 2:
        edges = np.linspace(min_levels, base_levels, n_tiers)
        lvl = edges[np.clip(np.round(tier_frac * (n_tiers - 1)).astype(int), 0, n_tiers - 1)]
    return np.clip(np.round(lvl), min_levels, base_levels).astype(np.int64).reshape(flip_mass.shape)


def apply_per_cell_quant_np(tokens: np.ndarray, level_map: np.ndarray) -> np.ndarray:
    """numpy REFERENCE for the per-cell effective quantization (authority for the MLX path).

    ``tokens`` (gh,gw,c) in [-1,1]; ``level_map`` (gh,gw) int effective levels.  Each cell snaps
    to its own ``L=level-1`` sublattice: ``q = round((t+1)/2*L)/L*2-1``.  A uniform ``level_map``
    reproduces the scalar-L control EXACTLY (off-identity)."""
    t = np.clip(tokens, -1.0, 1.0)
    L = (level_map.astype(np.float64) - 1.0)[..., None]          # (gh,gw,1)
    x01 = (t + 1.0) * 0.5
    return np.round(x01 * L) / L * 2.0 - 1.0


# ---------------------------------------------------------------------------
# (b) delta group-sparsity — the ξ-informed weight field + the group-L2 penalty.
# ---------------------------------------------------------------------------
def xi_informed_delta_weight(dynamic_frac: np.ndarray, *, floor: float = 0.1) -> np.ndarray:
    """Per-cell delta-shrinkage weight (ax1 §5): RELAX where the ego-motion prior says deltas
    legitimately move (high ``dynamic_frac`` = lane/movable), TIGHTEN on the static mass.

    ``w = floor + (1-floor)*(1 - dynamic_frac)``: a static-mass cell gets weight 1, a fully-dynamic
    cell gets ``floor``.  ``floor`` keeps a minimum shrinkage everywhere (deltas are near-zero even
    in dynamic cells 98.8% of the time).  Returns a (gh,gw) float in [floor,1].  The "uniform" mode
    is handled by the caller (weight_field="uniform" => no weight built => the penalty uses None)."""
    df = np.clip(dynamic_frac.astype(np.float64), 0.0, 1.0)
    return floor + (1.0 - floor) * (1.0 - df)


def delta_group_sparsity_penalty(
    deltas: np.ndarray, weight_field: np.ndarray | None = None, *, eps: float = 1e-8,
) -> float:
    """numpy REFERENCE group-L2 (group-lasso) penalty on per-pair token deltas.

    ``deltas`` (P,gh,gw,c): per-pair residual off the shared base.  The GROUP is a (pair,cell)
    delta vector over its ``c`` channels; the penalty is the sum of per-group L2 norms
    (structured sparsity → whole cells go to zero-delta → SMEVR zero-delta runs).  ``weight_field``
    (gh,gw) scales each cell's contribution (the ξ-informed relax map); None => uniform.
    Returns a scalar (the MLX trainer term mirrors this exactly; this is the authority)."""
    g = np.sqrt(np.sum(deltas.astype(np.float64) ** 2, axis=-1) + eps)   # (P,gh,gw)
    if weight_field is not None:
        g = g * weight_field.astype(np.float64)[None]
    return float(g.mean())


__all__ = [
    "DYNAMIC_CLASSES",
    "QA80_FIELD_CUSTODY",
    "CellFieldAggregate",
    "QA80FieldError",
    "apply_per_cell_quant_np",
    "delta_group_sparsity_penalty",
    "load_qa80_cell_field",
    "margin_coupled_level_map",
    "xi_informed_delta_weight",
]
