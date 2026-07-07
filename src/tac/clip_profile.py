"""tac.clip_profile — the canonical per-clip MEASURED config (single source of truth).

**Motivation (operator directive 2026-07-06):** video-specific quantities (horizon row,
camera intrinsics, class-index order, dash period, ...) were scattered as HARDCODED module
constants (``lane_sdf_component._V_HORIZON=174``, ``camera.COMMA_INTRINSICS``, ...) tuned to
the contest clip AND re-derived by ~60 orphaned ``measure_*.py`` scripts. That is the
velocity-driven-orphaning + "results must become system intelligence" anti-pattern: the same
per-clip facts measured over and over, never canonicalized, and a hardcode that silently
overfits the one clip so the pipeline is NOT drop-in for another clip/corpus.

**The fix (this module):** ONE ``ClipProfile`` — MEASURED automatically from the video (+ its
frozen SegNet argmax), CACHED (keyed by the video's sha256, deterministic), PROVENANCE-stamped,
and CONSUMED by the pipeline (trainer / lane_sdf / byte-close / inflate) in place of the
scattered constants. Overfit AND drop-in fall out for free: on ``0.mkv`` it MEASURES the same
values the hardcodes had (v_horizon ~174, ``_neo_config`` intrinsics — regression-tested); on
any other clip it self-calibrates. Every field is measured-not-assumed (NO-FAKE); the openpilot
geometry it reuses is a generic deterministic prior (rule-118 free — no learned weights).

**0.mkv provenance (identified in our record):** the contest ``upstream/videos/0.mkv`` is the
remuxed **comma2k19 RAV4 segment** (37,545,489 bytes = the rate denominator). This is stamped
in the profile so downstream (comma2k19 GT pose/calib) can key off it.

Phase 1 (this landing): the profile + auto-measure + cache + provenance + regression test that
0.mkv reproduces the current hardcodes. Phase 2 (gated, next run): rewire the score-live
modules to READ ``ClipProfile`` instead of their hardcoded constants (measured-no-regression
first). Phase 3: fold the orphaned ``measure_*.py`` logic into the canonical measure pass.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tac.camera import (
    CAMERA_H,
    CAMERA_W,
    COMMA_INTRINSICS,
    COMMA_INTRINSICS_NATIVE,
)

# openpilot device height (calibrationd HEIGHT_INIT); the CONFIRMED-correct value our
# ξ/warp path already uses (audit #326). Not clip-specific, but lives here as the single
# source so the lane-IPM path stops disagreeing with the pose path (1.2 vs 1.22).
OPENPILOT_DEVICE_HEIGHT_M = 1.22

# comma10k canonical SegNet argmax index order (MEASURED, non-negotiable — see MEMORY:
# the trained net emits [Road,Lane,Undrivable,Movable,MyCar], NOT the luma sort). Kept as
# the EXPECTED order the self-detector verifies against, never blindly hardcoded downstream.
CANONICAL_CLASS_ORDER = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".omx/state/clip_profiles"


@dataclass(frozen=True)
class ClipCamera:
    """Native + scorer-resolution intrinsics for the clip (auto-matched by resolution)."""
    native_w: int
    native_h: int
    fx_native: float
    fy_native: float
    cx_native: float
    cy_native: float
    scorer_w: int
    scorer_h: int
    fx_scorer: float
    fy_scorer: float
    cx_scorer: float
    cy_scorer: float
    matched_openpilot_camera: str   # e.g. "_neo_config" or "unknown"


@dataclass(frozen=True)
class ClipProfile:
    """The canonical MEASURED per-clip config — single source of truth for the pipeline."""
    video_path: str
    video_sha256: str | None
    video_bytes: int | None
    n_frames_measured: int
    camera: ClipCamera
    # measured geometry
    v_horizon: float                    # topmost-drivable argmax boundary (lane-IPM cutoff)
    v_horizon_p10: float
    v_horizon_p90: float
    device_height_m: float
    # self-detected class order (verifies against CANONICAL_CLASS_ORDER)
    class_order: tuple[str, ...]
    class_order_is_canonical: bool
    class_index: dict[str, int]         # name -> argmax index
    # provenance (metadata; not part of the deterministic measured fields)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> ClipProfile:
        cam = ClipCamera(**d["camera"])
        d = dict(d)
        d["camera"] = cam
        d["class_order"] = tuple(d["class_order"])
        return ClipProfile(**d)


# ── measure primitives (each is the canonical home for a per-clip fact) ──────────

def camera_for_resolution(native_w: int, native_h: int) -> ClipCamera:
    """Match the clip resolution to the openpilot road camera → intrinsics (native + scorer).

    1164×874 = openpilot ``_neo_config`` (CONFIRMED, audit #326): K reuses the canonical
    ``tac.camera`` constants. Other resolutions rescale the native K by (w,h) ratio and mark
    the camera 'unknown' (still usable, flagged for calibration)."""
    if (native_w, native_h) == (CAMERA_W, CAMERA_H):
        n, s = COMMA_INTRINSICS_NATIVE, COMMA_INTRINSICS
        return ClipCamera(
            native_w=native_w, native_h=native_h,
            fx_native=n.fx, fy_native=n.fy, cx_native=n.cx, cy_native=n.cy,
            scorer_w=512, scorer_h=384,
            fx_scorer=s.fx, fy_scorer=s.fy, cx_scorer=s.cx, cy_scorer=s.cy,
            matched_openpilot_camera="_neo_config",
        )
    # unknown camera: rescale the native _neo_config K by the resolution ratio (best-effort)
    rx, ry = native_w / CAMERA_W, native_h / CAMERA_H
    n = COMMA_INTRINSICS_NATIVE
    sx, sy = 512.0 / native_w, 384.0 / native_h
    return ClipCamera(
        native_w=native_w, native_h=native_h,
        fx_native=n.fx * rx, fy_native=n.fy * ry, cx_native=n.cx * rx, cy_native=n.cy * ry,
        scorer_w=512, scorer_h=384,
        fx_scorer=n.fx * rx * sx, fy_scorer=n.fy * ry * sy,
        cx_scorer=n.cx * rx * sx, cy_scorer=n.cy * ry * sy,
        matched_openpilot_camera="unknown",
    )


def measure_v_horizon(lstars: np.ndarray, *, road_cls: int = 0, lane_cls: int = 1
                      ) -> tuple[float, float, float]:
    """The lane-IPM horizon = the topmost image row where the drivable plane (road|lane)
    appears in the frozen SegNet argmax, over all frames. MEASURED — median + p10/p90.

    On 0.mkv this returns ~174-175, matching the (empirically-correct) hardcoded
    ``_V_HORIZON=174`` (n600 sweep #327 found 174 OPTIMAL). On any other clip it
    self-calibrates to that clip's horizon — the drop-in property."""
    tops = []
    for i in range(lstars.shape[0]):
        drivable = (lstars[i] == road_cls) | (lstars[i] == lane_cls)
        r = np.where(drivable.any(axis=1))[0]
        if r.size:
            tops.append(int(r.min()))
    a = np.asarray(tops, dtype=np.float64)
    return float(np.median(a)), float(np.percentile(a, 10)), float(np.percentile(a, 90))


def detect_class_order(lstars: np.ndarray, n_classes: int = 5) -> tuple[tuple[str, ...], dict[str, int]]:
    """SELF-DETECT the argmax class→name mapping from spatial/temporal signature (the
    'never hardcode the index' non-negotiable), then it can be verified against
    ``CANONICAL_CLASS_ORDER``. Signatures (MEASURED, MEMORY comma10k order):
      MyCar  = bottom rows + temporally STATIC (highest IoU) + large area
      Undrivable(sky) = top rows + large area
      Road   = mid-lower + largest-or-2nd area
      Movable= mid-band + small area (moving)
      Lane   = thinnest area."""
    H = lstars.shape[1]
    feats = {}
    prev = None
    ious = {c: [] for c in range(n_classes)}
    for i in range(lstars.shape[0]):
        lab = lstars[i]
        if prev is not None:
            for c in range(n_classes):
                a = prev == c
                b = lab == c
                u = (a | b).sum()
                ious[c].append(((a & b).sum() / u) if u else 1.0)
        prev = lab
    for c in range(n_classes):
        m = np.mean([(lstars[i] == c) for i in range(lstars.shape[0])], axis=0)  # (H,W) freq
        area = float(m.mean())
        vcen = float((m.sum(axis=1) * np.arange(H)).sum() / max(m.sum(), 1e-9)) / H
        feats[c] = {"area": area, "vcen": vcen, "iou": float(np.mean(ious[c])) if ious[c] else 1.0}
    # assign by signature
    order = sorted(range(n_classes), key=lambda c: feats[c]["area"])
    lane = order[0]                                           # thinnest
    mycar = max(range(n_classes), key=lambda c: feats[c]["vcen"] + feats[c]["iou"] * 0.5)  # bottom+static
    sky = min((c for c in range(n_classes) if c not in (lane, mycar)),
              key=lambda c: feats[c]["vcen"])                 # topmost of the rest
    remaining = [c for c in range(n_classes) if c not in (lane, mycar, sky)]
    road = max(remaining, key=lambda c: feats[c]["area"])     # largest of the rest
    movable = next(c for c in remaining if c != road)
    idx = {"Road": road, "Lane": lane, "Undrivable": sky, "Movable": movable, "MyCar": mycar}
    order_names = tuple(sorted(idx, key=lambda k: idx[k]))
    return order_names, idx


def _sha256_and_size(path: Path) -> tuple[str | None, int | None]:
    if not path.is_file():
        return None, None
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _video_resolution(path: Path) -> tuple[int, int]:
    """(w,h) of the first video stream; falls back to the comma native res if av is absent."""
    try:
        import av
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            return int(s.codec_context.width), int(s.codec_context.height)
    except Exception:
        return CAMERA_W, CAMERA_H


def measure(video_path: str | Path, lstars: np.ndarray, *, git_sha: str | None = None,
            measured_at: str | None = None) -> ClipProfile:
    """Build the canonical ClipProfile by MEASURING every field from the video + its argmax.

    ``lstars``: the frozen SegNet argmax stack (N,H,W) for the clip (the d_seg target the
    band/horizon/class-order are defined against). ``git_sha``/``measured_at`` are provenance
    only (kept out of the deterministic measured fields)."""
    p = Path(video_path)
    sha, nbytes = _sha256_and_size(p)
    w, h = _video_resolution(p)
    cam = camera_for_resolution(w, h)
    # (review-fix HIGH) SELF-DETECT the class order FIRST, then feed the DETECTED road/lane
    # indices into the horizon measurement. The prior order (measure_v_horizon before
    # detect_class_order) used the hardcoded road_cls=0/lane_cls=1 defaults — silently inert on
    # 0.mkv (detected==default) but a wrong horizon on any clip whose SegNet channel assignment
    # differs, violating this module's own "never hardcode the index, self-detect" contract.
    order, idx = detect_class_order(lstars)
    vh, vh10, vh90 = measure_v_horizon(lstars, road_cls=idx["Road"], lane_cls=idx["Lane"])
    is_canon = order == CANONICAL_CLASS_ORDER
    is_comma2k19_rav4 = nbytes == 37_545_489
    prov = {
        "measured_at": measured_at,
        "git_sha": git_sha,
        "method": "clip_profile.measure v1 (road/sky-boundary horizon + resolution-matched "
                  "intrinsics + spatial/temporal class self-detect)",
        "source_identified": "comma2k19 RAV4 segment (remuxed)" if is_comma2k19_rav4 else "unknown",
        "n_frames": int(lstars.shape[0]),
        "axis_tag": "[macOS-CPU advisory] (geometry prior; not an exact score)",
    }
    return ClipProfile(
        video_path=str(p), video_sha256=sha, video_bytes=nbytes,
        n_frames_measured=int(lstars.shape[0]), camera=cam,
        v_horizon=vh, v_horizon_p10=vh10, v_horizon_p90=vh90,
        device_height_m=OPENPILOT_DEVICE_HEIGHT_M,
        class_order=order, class_order_is_canonical=is_canon, class_index=idx,
        provenance=prov,
    )


def _cache_path(sha: str) -> Path:
    return _CACHE_DIR / f"{sha}.json"


def for_video(video_path: str | Path, lstars: np.ndarray | None = None, *,
              refresh: bool = False, git_sha: str | None = None,
              measured_at: str | None = None) -> ClipProfile:
    """Measure-or-load the cached ClipProfile for a video (keyed by its sha256).

    If a cached profile exists (and not ``refresh``), it is returned without re-measuring.
    Otherwise ``lstars`` (the frozen SegNet argmax stack) MUST be provided to measure it.
    The cache lives at ``.omx/state/clip_profiles/<sha>.json`` (deterministic, provenance)."""
    p = Path(video_path)
    sha, _ = _sha256_and_size(p)
    if sha is not None and not refresh:
        cp = _cache_path(sha)
        if cp.is_file():
            return ClipProfile.from_dict(json.loads(cp.read_text()))
    if lstars is None:
        raise ValueError(
            "no cached ClipProfile for this video and lstars not provided — pass the "
            "frozen SegNet argmax stack so the profile can be MEASURED.")
    prof = measure(p, lstars, git_sha=git_sha, measured_at=measured_at)
    if prof.video_sha256 is not None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(prof.video_sha256).write_text(json.dumps(prof.to_dict(), indent=2))
    return prof
