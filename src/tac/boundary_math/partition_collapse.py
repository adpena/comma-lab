# SPDX-License-Identifier: MIT
"""Per-class partition-collapse feasibility primitives (v8 rate-half probe).

The v8 SPEC (`.omx/research/SPEC_v8_perclass_decomposition_20260708.md`) rests on
the #284 thesis "argmax = Laguerre power diagram -> store the GENERATORS, not the
boundaries" (Nielsen non-Euclidean CG = Bregman power diagrams).  This module
measures that thesis on the REAL cached SegNet argmax labels (no scorer forward),
per class, with each class measured against its geometry-matched representation:

* **Global Laguerre / power diagram fit** (sites + weights + class labels) with a
  greedy error-driven K sweep -> the rate--fidelity curve for "one power diagram
  for everything" (bulk + boundary-band agreement, per class).
* **Road/Undrivable bulk**: small-K power diagram on the inpainted 2-class base
  partition (the v8 "ONE bulk-boundary field" carrier).
* **Lane**: low-order polynomial curve + width band (openpilot lane geometry;
  a power diagram is the WRONG model for a 1-D curve), plus dash occupancy
  models (per-run endpoints vs periodic-in-image vs periodic-in-ego-distance).
* **Movable**: sparse islands -> moment ellipses.
* **MyCar**: static mask (temporal median), amortized across frames.
* **Contour-codec byte floor** (~1.25 bits/crack-edge px, #307) and zlib label
  map bytes as the "store the boundaries" comparators.
* **Union hybrid**: compose the per-class-matched carriers and measure the
  honest residual disagreement that no carrier collapses.

All metrics are geometric fidelity vs the cached argmax labels — an ADVISORY
feasibility surface (`[macOS-MLX advisory]`-class), NOT a byte-closed score:
the real d_seg re-reads the rendered frame through R + the frozen SegNet.

Canonical class order (MEASURED, CLAUDE.md non-negotiable — never re-derive):
``0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

NUM_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
ROAD, LANE, UNDRIV, MOVABLE, MYCAR = range(NUM_CLASSES)

# Byte-model constants (stated assumptions; raw independent-frame coding).
_BITS_SITE_ROW = 9  # 384 rows
_BITS_SITE_COL = 9  # 512 cols
_BITS_WEIGHT = 8
_BITS_CLASS = 3
BITS_PER_GENERATOR = _BITS_SITE_ROW + _BITS_SITE_COL + _BITS_WEIGHT + _BITS_CLASS  # 29
CONTOUR_BITS_PER_EDGE_PX = 1.25  # #307 chain-code/context-arith floor band 1.0-1.5


class PartitionCollapseError(ValueError):
    """Raised on invalid inputs to the partition-collapse primitives."""


def _require_labels(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise PartitionCollapseError(f"labels must be (H,W); got {arr.shape}")
    if arr.size == 0:
        raise PartitionCollapseError("labels must be non-empty")
    if int(arr.min()) < 0 or int(arr.max()) >= NUM_CLASSES:
        raise PartitionCollapseError("labels must be in [0, NUM_CLASSES)")
    return arr.astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# Power / Laguerre diagram
# ---------------------------------------------------------------------------
@dataclass
class PowerDiagram:
    """Labeled Laguerre diagram: label(p) = class of argmin_i ||p-x_i||^2 - w_i."""

    sites: np.ndarray  # (K,2) float32 (row, col)
    weights: np.ndarray  # (K,) float32
    classes: np.ndarray  # (K,) int32

    def __post_init__(self) -> None:
        self.sites = np.asarray(self.sites, dtype=np.float32).reshape(-1, 2)
        self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        self.classes = np.asarray(self.classes, dtype=np.int32).reshape(-1)
        k = self.sites.shape[0]
        if self.weights.shape[0] != k or self.classes.shape[0] != k:
            raise PartitionCollapseError("sites/weights/classes length mismatch")

    @property
    def k(self) -> int:
        return int(self.sites.shape[0])

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.classes, minlength=NUM_CLASSES)


def power_assign(
    diagram: PowerDiagram, shape: tuple[int, int], *, row_chunk: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Assign every pixel to its power-diagram winner.

    Returns ``(best_idx, pred_labels)`` each of shape ``shape``. Chunked over
    pixel rows so peak memory stays ~(row_chunk*W*K) float32.
    """
    h, w = shape
    if diagram.k == 0:
        raise PartitionCollapseError("empty diagram")
    sites = diagram.sites
    site_norm = (sites**2).sum(axis=1) - diagram.weights  # (K,)
    cols = np.arange(w, dtype=np.float32)
    best_idx = np.empty((h, w), dtype=np.int32)
    for r0 in range(0, h, row_chunk):
        r1 = min(h, r0 + row_chunk)
        rows = np.arange(r0, r1, dtype=np.float32)
        pix = np.empty(((r1 - r0) * w, 2), dtype=np.float32)
        pix[:, 0] = np.repeat(rows, w)
        pix[:, 1] = np.tile(cols, r1 - r0)
        # ||p-x||^2 - w = ||p||^2 - 2 p.x + (||x||^2 - w); ||p||^2 constant per pixel.
        with np.errstate(all="ignore"):  # Accelerate BLAS sets spurious FP flags on macOS
            scores = pix @ sites.T  # (P,K)
        scores *= -2.0
        scores += site_norm[None, :]
        best_idx[r0:r1] = np.argmin(scores, axis=1).astype(np.int32).reshape(r1 - r0, w)
    pred = diagram.classes[best_idx]
    return best_idx, pred


def _interior_point(mask: np.ndarray) -> tuple[int, int]:
    """Most-interior pixel of a boolean component (max distance transform)."""
    from scipy import ndimage

    dt = ndimage.distance_transform_edt(mask)
    flat = int(np.argmax(dt))
    return int(flat // mask.shape[1]), int(flat % mask.shape[1])


def _lloyd_anchor_step(
    diagram: PowerDiagram, labels: np.ndarray, best_idx: np.ndarray
) -> PowerDiagram:
    """Recenter each site to the centroid of its correctly-labeled cell pixels."""
    h, w = labels.shape
    flat_idx = best_idx.ravel()
    gt = labels.ravel()
    correct = gt == diagram.classes[flat_idx]
    sel = flat_idx[correct]
    k = diagram.k
    cnt = np.bincount(sel, minlength=k).astype(np.float64)
    rr = (np.arange(h * w) // w).astype(np.float64)
    cc = (np.arange(h * w) % w).astype(np.float64)
    sum_r = np.bincount(sel, weights=rr[correct], minlength=k)
    sum_c = np.bincount(sel, weights=cc[correct], minlength=k)
    sites = diagram.sites.copy()
    nz = cnt > 0
    sites[nz, 0] = (sum_r[nz] / cnt[nz]).astype(np.float32)
    sites[nz, 1] = (sum_c[nz] / cnt[nz]).astype(np.float32)
    return replace(diagram, sites=sites)


def boundary_band_mask(labels: np.ndarray, radius: int = 3) -> np.ndarray:
    """Pixels within ``radius`` of any inter-class edge (what d_seg sees)."""
    from scipy import ndimage

    labels = _require_labels(labels)
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edges[1:, :] |= labels[:-1, :] != labels[1:, :]
    if radius <= 1:
        return edges
    return ndimage.binary_dilation(edges, iterations=radius - 1)


def evaluate_partition(
    pred: np.ndarray, gt: np.ndarray, band: np.ndarray | None = None
) -> dict:
    """Bulk + boundary-band fidelity of a reconstructed partition vs the argmax."""
    gt = _require_labels(gt)
    pred = np.asarray(pred, dtype=np.int32)
    if pred.shape != gt.shape:
        raise PartitionCollapseError("pred/gt shape mismatch")
    if band is None:
        band = boundary_band_mask(gt)
    agree = pred == gt
    out: dict = {
        "bulk_agreement": float(agree.mean()),
        "disagreement": float(1.0 - agree.mean()),
        "band_agreement": float(agree[band].mean()) if band.any() else 1.0,
        "band_frac": float(band.mean()),
    }
    per_class: dict[str, dict] = {}
    for c in range(NUM_CLASSES):
        m = gt == c
        n = int(m.sum())
        row: dict = {"px": n}
        if n:
            row["recall"] = float(agree[m].mean())
            mb = m & band
            row["band_recall"] = float(agree[mb].mean()) if mb.any() else 1.0
            inter = int((m & (pred == c)).sum())
            union = int((m | (pred == c)).sum())
            row["iou"] = float(inter / union) if union else 1.0
        per_class[CLASS_NAMES[c]] = row
    out["per_class"] = per_class
    return out


def fit_power_diagram_greedy(
    labels: np.ndarray,
    k_max: int,
    *,
    checkpoints: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024),
    lloyd_iters: int = 2,
    growth: float = 1.4,
    band_radius: int = 3,
    keep_snapshots: bool = False,
) -> tuple[PowerDiagram, list[dict]]:
    """Greedy error-driven Laguerre fit with a K sweep.

    Seeds one site per largest component per class, then repeatedly inserts
    sites at the interior points of the largest misclassified components
    (class = true label there), interleaved with anchored-Lloyd recentering.
    Records fidelity metrics at each checkpoint K (plus the diagram snapshot
    when ``keep_snapshots``, so callers can weight-refine each checkpoint).
    """
    from scipy import ndimage

    labels = _require_labels(labels)
    band = boundary_band_mask(labels, band_radius)
    sites: list[tuple[float, float]] = []
    classes: list[int] = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        if not mask.any():
            continue
        lab, n = ndimage.label(mask)
        if n == 0:
            continue
        largest = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
        r, cc_ = _interior_point(lab == largest)
        sites.append((float(r), float(cc_)))
        classes.append(c)
    if not sites:
        raise PartitionCollapseError("no classes present")
    diagram = PowerDiagram(
        np.asarray(sites, np.float32),
        np.zeros(len(sites), np.float32),
        np.asarray(classes, np.int32),
    )
    ckpts = sorted({int(c) for c in checkpoints if int(c) <= k_max} | {int(k_max)})
    next_ck = 0
    curve: list[dict] = []
    while True:
        for _ in range(max(1, lloyd_iters)):
            best_idx, _pred = power_assign(diagram, labels.shape)
            diagram = _lloyd_anchor_step(diagram, labels, best_idx)
        best_idx, pred = power_assign(diagram, labels.shape)
        err = pred != labels
        while next_ck < len(ckpts) and diagram.k >= ckpts[next_ck]:
            row = evaluate_partition(pred, labels, band)
            row["k"] = diagram.k
            row["k_per_class"] = diagram.class_counts().tolist()
            if keep_snapshots:
                row["snapshot"] = replace(
                    diagram,
                    sites=diagram.sites.copy(),
                    weights=diagram.weights.copy(),
                    classes=diagram.classes.copy(),
                )
            curve.append(row)
            next_ck += 1
        if diagram.k >= k_max or not err.any():
            if not err.any() and (not curve or curve[-1]["k"] != diagram.k):
                row = evaluate_partition(pred, labels, band)
                row["k"] = diagram.k
                row["k_per_class"] = diagram.class_counts().tolist()
                if keep_snapshots:
                    row["snapshot"] = diagram
                curve.append(row)
            break
        n_new = min(max(1, int(diagram.k * (growth - 1.0))), k_max - diagram.k)
        lab, n = ndimage.label(err)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        order = np.argsort(sizes)[::-1][:n_new]
        add_sites = []
        add_classes = []
        for comp in order:
            comp_mask = lab == (comp + 1)
            r, cc_ = _interior_point(comp_mask)
            add_sites.append((float(r), float(cc_)))
            add_classes.append(int(labels[r, cc_]))
        diagram = PowerDiagram(
            np.concatenate([diagram.sites, np.asarray(add_sites, np.float32)]),
            np.concatenate([diagram.weights, np.zeros(len(add_sites), np.float32)]),
            np.concatenate([diagram.classes, np.asarray(add_classes, np.int32)]),
        )
    return diagram, curve


def refine_weights_ce(
    diagram: PowerDiagram,
    labels: np.ndarray,
    *,
    steps: int = 150,
    n_sample: int = 20000,
    lr: float = 0.5,
    tau: float = 50.0,
    optimize_positions: bool = False,
    seed: int = 0,
) -> PowerDiagram:
    """Softmax-CE (tau-relaxed tropical) refinement of Laguerre weights.

    The per-class field is ``phi_c(p) = max_{i in c} (w_i - ||p-x_i||^2)`` —
    exactly the tau->0 tropical witness structure; we anneal with a fixed tau
    and Adam-update the weights (and optionally positions) on a stratified
    pixel sample (half boundary-band, half uniform).
    """
    labels = _require_labels(labels)
    h, w = labels.shape
    rng = np.random.default_rng(seed)
    band = boundary_band_mask(labels)
    band_pts = np.argwhere(band)
    n_band = min(n_sample // 2, len(band_pts))
    sel_band = band_pts[rng.choice(len(band_pts), size=n_band, replace=False)]
    n_unif = n_sample - n_band
    unif = np.column_stack(
        [rng.integers(0, h, n_unif), rng.integers(0, w, n_unif)]
    )
    pts = np.concatenate([sel_band, unif]).astype(np.float32)
    y = labels[pts[:, 0].astype(int), pts[:, 1].astype(int)]
    n = len(pts)

    sites = diagram.sites.copy()
    weights = diagram.weights.copy()
    cls = diagram.classes
    class_lists = [np.flatnonzero(cls == c) for c in range(NUM_CLASSES)]
    present = [c for c in range(NUM_CLASSES) if len(class_lists[c])]

    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_x = np.zeros_like(sites)
    v_x = np.zeros_like(sites)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        with np.errstate(all="ignore"):  # spurious Accelerate BLAS flags
            scores = pts @ sites.T
        scores *= -2.0
        scores += ((sites**2).sum(axis=1) - weights)[None, :]
        scores *= -1.0  # now  w - ||p-x||^2 + ||p||^2 (constant offset cancels in softmax over classes)
        phi = np.full((n, NUM_CLASSES), -np.inf, dtype=np.float32)
        arg = np.zeros((n, NUM_CLASSES), dtype=np.int32)
        for c in present:
            sub = scores[:, class_lists[c]]
            a = np.argmax(sub, axis=1)
            phi[:, c] = sub[np.arange(n), a]
            arg[:, c] = class_lists[c][a]
        z = phi / tau
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g_out = p.copy()
        g_out[np.arange(n), y] -= 1.0
        g_out /= tau * n  # d loss / d phi_c
        grad_w = np.zeros_like(weights)
        grad_x = np.zeros_like(sites)
        for c in present:
            gc = g_out[:, c]
            idx = arg[:, c]
            np.add.at(grad_w, idx, gc)  # d phi/d w = 1 at the argmax site
            if optimize_positions:
                diff = 2.0 * (pts - sites[idx])  # d phi/d x = 2(p - x)
                np.add.at(grad_x, idx, diff * gc[:, None])
        m_w = b1 * m_w + (1 - b1) * grad_w
        v_w = b2 * v_w + (1 - b2) * grad_w**2
        weights -= lr * (m_w / (1 - b1**t)) / (np.sqrt(v_w / (1 - b2**t)) + eps)
        if optimize_positions:
            m_x = b1 * m_x + (1 - b1) * grad_x
            v_x = b2 * v_x + (1 - b2) * grad_x**2
            sites -= lr * (m_x / (1 - b1**t)) / (np.sqrt(v_x / (1 - b2**t)) + eps)
    return PowerDiagram(sites, weights, cls)


def generator_bytes(k: int, *, with_weights: bool = True) -> float:
    """Raw per-frame byte cost of K generators (quantized, uncoded)."""
    bits = BITS_PER_GENERATOR if with_weights else BITS_PER_GENERATOR - _BITS_WEIGHT
    return k * bits / 8.0


def synthetic_power_diagram(
    shape: tuple[int, int], k: int, *, seed: int = 0, weight_scale: float = 200.0
) -> tuple[PowerDiagram, np.ndarray]:
    """Random labeled power diagram + its rendered label map (test control)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    sites = np.column_stack(
        [rng.uniform(0, h, k), rng.uniform(0, w, k)]
    ).astype(np.float32)
    weights = rng.uniform(0, weight_scale, k).astype(np.float32)
    classes = rng.integers(0, NUM_CLASSES, k).astype(np.int32)
    diagram = PowerDiagram(sites, weights, classes)
    _, labels = power_assign(diagram, shape)
    return diagram, labels


# ---------------------------------------------------------------------------
# Contour / label-map byte baselines ("store the boundaries" comparator)
# ---------------------------------------------------------------------------
def contour_stats(labels: np.ndarray) -> dict:
    """Crack-edge counts (total + per class-pair) and the #307 byte floor."""
    labels = _require_labels(labels)
    right = labels[:, :-1] != labels[:, 1:]
    down = labels[:-1, :] != labels[1:, :]
    total = int(right.sum() + down.sum())
    pair_counts: dict[str, int] = {}
    for diff, a, b in (
        (right, labels[:, :-1], labels[:, 1:]),
        (down, labels[:-1, :], labels[1:, :]),
    ):
        aa = a[diff]
        bb = b[diff]
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        key = lo * NUM_CLASSES + hi
        for kk, cnt in zip(*np.unique(key, return_counts=True)):
            name = f"{CLASS_NAMES[int(kk) // NUM_CLASSES]}-{CLASS_NAMES[int(kk) % NUM_CLASSES]}"
            pair_counts[name] = pair_counts.get(name, 0) + int(cnt)
    return {
        "edge_px": total,
        "bytes_floor": total * CONTOUR_BITS_PER_EDGE_PX / 8.0,
        "per_pair": dict(sorted(pair_counts.items(), key=lambda kv: -kv[1])),
    }


def zlib_label_bytes(labels: np.ndarray, level: int = 9) -> int:
    """zlib bytes of the raw uint8 label map (practical whole-frame comparator)."""
    import zlib

    labels = _require_labels(labels)
    return len(zlib.compress(labels.astype(np.uint8).tobytes(), level))


# ---------------------------------------------------------------------------
# Lane: polynomial curve + width band (+ dash occupancy models)
# ---------------------------------------------------------------------------
@dataclass
class LaneCurve:
    coeffs: np.ndarray  # polynomial col = poly(row)
    width: float
    row_min: int
    row_max: int
    n_px: int
    occupancy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def fit_lane_curves(
    labels: np.ndarray,
    *,
    deg: int = 3,
    min_px: int = 25,
    dilate_iters: int = 6,
) -> tuple[list[LaneCurve], int]:
    """Fit low-order polynomial curves (col = poly(row)) + width to the Lane class.

    Dashes belonging to one lane line are merged by isotropic dilation before
    component grouping. Returns (curves, uncovered_lane_px) where uncovered
    counts lane pixels in components too small to fit (the honest residual).
    """
    from scipy import ndimage

    labels = _require_labels(labels)
    lane = labels == LANE
    if not lane.any():
        return [], 0
    merged = ndimage.binary_dilation(lane, iterations=dilate_iters)
    lab, n = ndimage.label(merged)
    curves: list[LaneCurve] = []
    uncovered = 0
    for comp in range(1, n + 1):
        mask = lane & (lab == comp)
        n_px = int(mask.sum())
        if n_px < min_px:
            uncovered += n_px
            continue
        rr, cc = np.nonzero(mask)
        r_min, r_max = int(rr.min()), int(rr.max())
        span = max(1, r_max - r_min)
        d = min(deg, max(1, len(np.unique(rr)) - 1))
        coeffs = np.polyfit(rr.astype(np.float64), cc.astype(np.float64), d)
        # mean per-occupied-row thickness
        per_row = np.bincount(rr - r_min, minlength=span + 1)
        occ = per_row > 0
        width = float(per_row[occ].mean()) if occ.any() else 1.0
        curves.append(
            LaneCurve(
                coeffs=coeffs,
                width=width,
                row_min=r_min,
                row_max=r_max,
                n_px=n_px,
                occupancy=occ,
            )
        )
    return curves, uncovered


def render_lane_band(
    curves: list[LaneCurve],
    shape: tuple[int, int],
    *,
    use_occupancy: bool = True,
) -> np.ndarray:
    """Rasterize the fitted lane curves as bands of the fitted width."""
    h, w = shape
    out = np.zeros(shape, dtype=bool)
    cols = np.arange(w, dtype=np.float32)
    for cv in curves:
        rows = np.arange(cv.row_min, cv.row_max + 1)
        centers = np.polyval(cv.coeffs, rows.astype(np.float64))
        half = max(0.5, cv.width / 2.0)
        for j, r in enumerate(rows):
            if use_occupancy and j < len(cv.occupancy) and not cv.occupancy[j]:
                continue
            c = centers[j]
            lo = int(np.floor(c - half))
            hi = int(np.ceil(c + half))
            if hi < 0 or lo >= w or r < 0 or r >= h:
                continue
            out[r, max(0, lo) : min(w, hi + 1)] = True
    _ = cols
    return out


def lane_curve_bytes(
    curves: list[LaneCurve], *, per_run_bits: int = 18, model: str = "runs"
) -> float:
    """Byte model for the lane carrier.

    ``model='solid'``: coeffs (4x12b) + width (6b) + row range (2x9b) per curve.
    ``model='runs'``: solid + per-dash-run endpoints (2x9b each).
    ``model='periodic'``: solid + 3 dash params (period/phase/duty ~ 30b).
    """
    total_bits = 0.0
    for cv in curves:
        total_bits += 4 * 12 + 6 + 2 * 9
        if model == "runs":
            occ = cv.occupancy
            n_runs = int(np.count_nonzero(np.diff(occ.astype(np.int8)) == 1)) + int(
                bool(len(occ)) and bool(occ[0])
            )
            total_bits += n_runs * per_run_bits
        elif model == "periodic":
            total_bits += 30
    return total_bits / 8.0


def dash_occupancy_models(curve: LaneCurve) -> dict:
    """Fit dash occupancy along a lane curve with three models.

    Models: (a) per-run endpoints (exact, costed), (b) periodic in image rows,
    (c) periodic in ego-distance u = 1/(row - horizon) (dash phase = ego
    distance, #215). Returns per-model agreement + parameter counts.
    """
    occ = curve.occupancy.astype(bool)
    n = len(occ)
    if n < 8:
        return {"n_rows": n, "skipped": True}
    frac_on = float(occ.mean())
    out: dict = {"n_rows": n, "frac_on": frac_on, "skipped": False}
    transitions = np.diff(occ.astype(np.int8))
    n_runs = int(np.count_nonzero(transitions == 1)) + int(bool(occ[0]))
    out["n_runs"] = n_runs
    out["runs_bytes"] = n_runs * 2 * 9 / 8.0
    if frac_on >= 0.999:
        out["solid"] = True
        return out
    out["solid"] = False

    rows = np.arange(curve.row_min, curve.row_max + 1, dtype=np.float64)[:n]

    # seed the period search from the measured run lengths
    x = occ.astype(np.int8)
    change = np.flatnonzero(np.diff(x)) + 1
    bounds = np.concatenate([[0], change, [n]])
    run_lens = np.diff(bounds)
    run_vals = x[bounds[:-1]]
    on_runs = run_lens[run_vals == 1]
    off_runs = run_lens[run_vals == 0]
    med_on = float(np.median(on_runs)) if len(on_runs) else 1.0
    med_off = float(np.median(off_runs)) if len(off_runs) else 1.0
    period_rows = med_on + med_off
    duty0 = med_on / max(period_rows, 1e-6)
    out["median_on_rows"] = med_on
    out["median_off_rows"] = med_off

    def _best_periodic(coord: np.ndarray, period0: float) -> float:
        span = float(coord.max() - coord.min())
        if span <= 0 or period0 <= 0:
            return frac_on
        best = 0.0
        periods = np.concatenate(
            [period0 * np.linspace(0.7, 1.3, 31), span * np.linspace(0.05, 1.0, 20)]
        )
        duties = np.clip(duty0 * np.array([0.7, 0.85, 1.0, 1.15, 1.3]), 0.05, 0.95)
        for period in periods:
            ph_base = (coord - coord.min()) / period
            for phase in np.linspace(0.0, 1.0, 24, endpoint=False):
                ph = (ph_base + phase) % 1.0
                for duty in duties:
                    pred = ph < duty
                    best = max(best, float((pred == occ).mean()))
        return best

    out["periodic_image_agreement"] = _best_periodic(rows, period_rows)
    horizon = curve.row_min - 5.0
    u = 1.0 / np.maximum(rows - horizon, 1.0)
    # in u-space the ground-plane dash period maps to a locally uniform step;
    # seed with the median u-step per image-space period
    du = float(np.median(np.abs(np.diff(u)))) * period_rows
    out["periodic_egodist_agreement"] = _best_periodic(u, du)
    out["periodic_bytes"] = 30 / 8.0
    return out


# ---------------------------------------------------------------------------
# Movable: sparse islands -> moment ellipses
# ---------------------------------------------------------------------------
def fit_movable_ellipses(labels: np.ndarray, *, min_px: int = 12) -> tuple[list[dict], int]:
    """Fit a moment ellipse per Movable island; returns (ellipses, uncovered_px)."""
    from scipy import ndimage

    labels = _require_labels(labels)
    mask = labels == MOVABLE
    if not mask.any():
        return [], 0
    lab, n = ndimage.label(mask)
    ellipses: list[dict] = []
    uncovered = 0
    for comp in range(1, n + 1):
        m = lab == comp
        n_px = int(m.sum())
        if n_px < min_px:
            uncovered += n_px
            continue
        rr, cc = np.nonzero(m)
        mu = np.array([rr.mean(), cc.mean()])
        cov = np.cov(np.stack([rr, cc]).astype(np.float64)) + 1e-6 * np.eye(2)
        ellipses.append({"center": mu, "cov": cov, "n_px": n_px})
    return ellipses, uncovered


def render_ellipses(ellipses: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize moment ellipses (2-sigma Mahalanobis fill)."""
    out = np.zeros(shape, dtype=bool)
    h, w = shape
    for e in ellipses:
        mu = e["center"]
        inv = np.linalg.inv(e["cov"])
        # bounding box at 2.2 sigma of the major axis
        rad = 2.2 * float(np.sqrt(np.linalg.eigvalsh(e["cov"]).max()))
        r0 = max(0, int(mu[0] - rad))
        r1 = min(h, int(mu[0] + rad) + 1)
        c0 = max(0, int(mu[1] - rad))
        c1 = min(w, int(mu[1] + rad) + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        d = np.stack([rr - mu[0], cc - mu[1]], axis=-1).astype(np.float64)
        maha = np.einsum("...i,ij,...j->...", d, inv, d)
        out[r0:r1, c0:c1] |= maha <= 2.0 * 2.0
    return out


def movable_bytes(n_islands: int) -> float:
    """Byte model: center (18b) + cov 3 params (3x8b) per island."""
    return n_islands * (18 + 24) / 8.0


# ---------------------------------------------------------------------------
# MyCar: static mask
# ---------------------------------------------------------------------------
def static_mycar_mask(label_stack: np.ndarray) -> np.ndarray:
    """Temporal-majority MyCar mask over a (N,H,W) label stack."""
    arr = np.asarray(label_stack)
    if arr.ndim != 3:
        raise PartitionCollapseError("label_stack must be (N,H,W)")
    votes = (arr == MYCAR).mean(axis=0)
    return votes >= 0.5


def static_mask_bytes(mask: np.ndarray) -> int:
    """zlib bytes of the packed static mask (amortize across frames)."""
    import zlib

    return len(zlib.compress(np.packbits(np.asarray(mask, dtype=bool)).tobytes(), 9))


# ---------------------------------------------------------------------------
# Road/Undrivable base partition (2-class inpainted) + union hybrid
# ---------------------------------------------------------------------------
def road_undriv_base(labels: np.ndarray) -> np.ndarray:
    """Inpaint non-{Road,Undriv} pixels with the nearest {Road,Undriv} label."""
    from scipy import ndimage

    labels = _require_labels(labels)
    keep = (labels == ROAD) | (labels == UNDRIV)
    if not keep.any():
        raise PartitionCollapseError("no Road/Undrivable pixels")
    ind = ndimage.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    return labels[tuple(ind)]


def union_hybrid_reconstruction(
    labels: np.ndarray,
    *,
    ru_k: int = 24,
    mycar_mask: np.ndarray | None = None,
    band_radius: int = 3,
    refine_steps: int = 200,
) -> dict:
    """Compose the per-class-matched carriers and measure the honest residual.

    Paint order: Road/Undriv small-K power diagram base -> MyCar mask (static
    if provided, else this frame's own) -> Movable ellipses -> Lane band.
    """
    labels = _require_labels(labels)
    base_gt = road_undriv_base(labels)
    ru_diagram, ru_curve = fit_power_diagram_greedy(
        base_gt, ru_k, checkpoints=(ru_k,), lloyd_iters=2
    )
    if refine_steps > 0:
        ru_diagram = refine_weights_ce(
            ru_diagram,
            base_gt,
            steps=refine_steps,
            n_sample=8000,
            tau=10.0,
            optimize_positions=True,
        )
    _, pred = power_assign(ru_diagram, labels.shape)
    if mycar_mask is None:
        mycar_mask = labels == MYCAR
    pred = pred.copy()
    pred[mycar_mask] = MYCAR
    ellipses, mov_uncov = fit_movable_ellipses(labels)
    mov_mask = render_ellipses(ellipses, labels.shape)
    pred[mov_mask] = MOVABLE
    curves, lane_uncov = fit_lane_curves(labels)
    lane_mask = render_lane_band(curves, labels.shape)
    pred[lane_mask] = LANE
    band = boundary_band_mask(labels, band_radius)
    metrics = evaluate_partition(pred, labels, band)
    bytes_breakdown = {
        "road_undriv_generators": generator_bytes(ru_diagram.k),
        "lane_curves_runs": lane_curve_bytes(curves, model="runs"),
        "movable_ellipses": movable_bytes(len(ellipses)),
        "mycar_static_amortized_600": None,  # filled by caller (needs the stack)
    }
    return {
        "metrics": metrics,
        "ru_k": ru_diagram.k,
        "ru_curve": ru_curve,
        "n_lane_curves": len(curves),
        "lane_uncovered_px": lane_uncov,
        "n_movable_islands": len(ellipses),
        "movable_uncovered_px": mov_uncov,
        "bytes_breakdown": bytes_breakdown,
        "pred": pred,
    }
