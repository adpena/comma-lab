# SPDX-License-Identifier: MIT
"""Golden-vector generator for the lane AA-SDF render-band rasterizer (task #283).

Produces the committed golden vector for the level-set inflate's `_lane_coverage`
(the openpilot-IPM lane polynomial -> analytic signed-distance sub-pixel COVERAGE
per pixel). The rasterizer is MEASURED matmul- and transcendental-FREE (only
`np.polyval` Horner + elementwise `/ * - abs max min clip mod` + comparisons), so
it is BIT-EXACT AND PORTABLE across hosts (pure IEEE-754 fp64 in a fixed order, no
BLAS reduction ambiguity, no libm transcendental ULP gap, no FMA contraction) --
this is THE ship-able Rust inflate piece (contrast #281 fp32 / #282 neural forward,
which are NOT portable).

The inputs are REAL (no synthetic RNG): per-pair lane lines are fitted from the
FROZEN contest CPU-torch SegNet argmax label maps (`gt_n96.npz['lstars']`) via the
SAME canonical helpers the shipped byte-close/inflate uses
(`tac.boundary_math.analytic_lane_render_band.build_lane_band_pairs_from_lstars`
-> `serialize_lane_band` -> the LBND1 blob), then rasterized by the ACTUAL shipped
inflate `_lane_coverage` extracted from `tools/levelset_byte_close_and_eval.py`.

The vector pins:
  * `levelset_lane_coverage_v1_band.bin`   -- the REAL LBND1 serialized lane blob
    (LANE_BAND_MAGIC | u32 header_len | header_json(utf8) | float64 coeff payload).
    This is the COUNTED video-derived payload the decoder reads; the exact fp64
    coeff bits are preserved (binary, not JSON).
  * `levelset_lane_coverage_v1.json`       -- manifest: sha256 of the stacked
    per-pair coverage (float32 `<f4` C-order, pair-major) the Rust port must
    reproduce bit-for-bit, plus rh/rw (render resolution) + shape metadata.

Run:  .venv/bin/python runtime-rs/crates/tac-levelset-inflate/generate_lane_golden_vectors.py

Determinism: the gt cache is frozen; the fit + serialize + raster are all
deterministic, so the committed fixtures reproduce byte-for-byte. Regenerate only
when the lane-band grammar or the rasterizer changes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GV = HERE / "golden_vectors"
REPO = HERE.parents[2]  # runtime-rs/crates/tac-levelset-inflate -> repo root
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n96.npz"
TEMPLATE = REPO / "tools/levelset_byte_close_and_eval.py"

sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    deserialize_lane_band,
    rasterize_lane_coverage_range_dependent,
    serialize_lane_band,
)

# Render resolution the level-set witness (proven_base) uses -- the coverage raster
# is computed at render_h/render_w (SCORER_FIXED 384x512), independent of the
# LBND header geometry (which carries the same-rig IPM constants).
RENDER_H, RENDER_W = 384, 512
N_PAIRS = 32  # >= 24 (task requirement: representative pair set / end-to-end proof)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_shipped_inflate_lane_fns():
    """Extract the ACTUAL shipped inflate `_lane_parse` + `_lane_coverage` (the oracle).

    They live inside the `_INFLATE_PY` raw string of the byte-close template (the exact
    text written to the archive's inflate.py). We import the template module to obtain
    that string, then exec it in an isolated namespace so the parity is against the
    literal shipped code (NO-FAKE), not a re-derivation.
    """

    spec = importlib.util.spec_from_file_location("_lvls_byte_close_template", str(TEMPLATE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs the template's top-level imports (torch/tac.*)
    inflate_src = mod._INFLATE_PY
    ns: dict = {"__name__": "_shipped_inflate_oracle"}  # __name__ != "__main__" -> no main()
    exec(compile(inflate_src, "<shipped_inflate_py>", "exec"), ns)
    return ns["_lane_parse"], ns["_lane_coverage"], inflate_src


def main() -> int:
    GV.mkdir(parents=True, exist_ok=True)
    if not GT_CACHE.exists():
        print(f"gt cache not found: {GT_CACHE}", file=sys.stderr)
        return 2
    if not TEMPLATE.exists():
        print(f"byte-close template not found: {TEMPLATE}", file=sys.stderr)
        return 2

    _lane_parse, _lane_coverage, _inflate_src = _load_shipped_inflate_lane_fns()

    d = np.load(GT_CACHE)
    lstars = np.asarray(d["lstars"])[:N_PAIRS]  # (N, 384, 512) int64 -- frozen SegNet argmax
    assert lstars.ndim == 3, f"unexpected lstars shape {lstars.shape}"

    # Fit REAL per-pair lane lines from the REAL argmax + serialize the LBND1 blob
    # exactly as the byte-close does. dash_gate=True exercises the range-dependent dash path.
    cfg = LaneBandRenderConfig(softness=1.0, dash_gate=True, weight=1.0, lane_cls=1)
    pairs_lines, fit_stats = build_lane_band_pairs_from_lstars(list(lstars), cfg)
    blob = serialize_lane_band(pairs_lines, cfg)

    # ORACLE 1 (primary): the shipped inflate `_lane_parse` + `_lane_coverage`.
    parsed_pairs, hdr = _lane_parse(blob)
    cov_stack = np.stack(
        [_lane_coverage(parsed_pairs[i], RENDER_H, RENDER_W, hdr) for i in range(N_PAIRS)], axis=0
    ).astype("<f4")
    assert cov_stack.shape == (N_PAIRS, RENDER_H, RENDER_W)
    assert cov_stack.dtype == np.dtype("<f4")

    # ORACLE 2 (cross-check): the canonical source module -- must be BIT-IDENTICAL to
    # the shipped inflate mirror (proves template == module; two independent oracles agree).
    src_pairs, src_hdr = deserialize_lane_band(blob)
    cov_src = np.stack(
        [
            rasterize_lane_coverage_range_dependent(
                src_pairs[i], h=RENDER_H, w=RENDER_W, softness=float(src_hdr["softness"]),
                dash_gate=bool(src_hdr["dash_gate"]),
                dash_forward_max_m=float(src_hdr["dash_forward_max_m"]),
                v_h=float(src_hdr["v_h"]),
                cx=(None if src_hdr.get("cx") is None else float(src_hdr["cx"])),
            )
            for i in range(N_PAIRS)
        ],
        axis=0,
    ).astype("<f4")
    if _sha(cov_stack.tobytes(order="C")) != _sha(cov_src.tobytes(order="C")):
        print("FATAL: shipped-inflate oracle != canonical source module oracle", file=sys.stderr)
        return 3

    # Non-triviality (NO-FAKE): the raster must NOT be all zeros.
    n_lines_total = int(sum(len(p) for p in parsed_pairs))
    n_dash = int(sum(1 for p in parsed_pairs for ln in p if ln[2] > 0.0))
    nonzero = int(np.count_nonzero(cov_stack))
    if n_lines_total == 0 or nonzero == 0:
        print(f"FATAL: trivial coverage (lines={n_lines_total}, nonzero_px={nonzero})", file=sys.stderr)
        return 4

    stacked_bytes = cov_stack.tobytes(order="C")
    digest = _sha(stacked_bytes)

    (GV / "levelset_lane_coverage_v1_band.bin").write_bytes(blob)
    per_pair_n_lines = [len(p) for p in parsed_pairs]
    manifest = {
        "schema": "levelset_lane_coverage.v1",
        "sha256": digest,
        "render_h": RENDER_H,
        "render_w": RENDER_W,
        "n_pairs": N_PAIRS,
        "band_blob_bytes": int(len(blob)),
        "per_pair_n_lines": per_pair_n_lines,
        "n_lines_total": n_lines_total,
        "n_dash_lines": n_dash,
        "nonzero_coverage_px": nonzero,
        "coverage_max": float(cov_stack.max()),
        "band_recall_mean": (None if np.isnan(fit_stats["band_recall_mean"]) else float(fit_stats["band_recall_mean"])),
        "source": (
            "real frozen SegNet argmax gt_n96.npz['lstars'][:%d] -> "
            "build_lane_band_pairs_from_lstars -> serialize_lane_band (LBND1)" % N_PAIRS
        ),
        "oracle": "tools/levelset_byte_close_and_eval.py::_INFLATE_PY::_lane_coverage (shipped inflate); "
        "cross-checked bit-identical vs tac.boundary_math.analytic_lane_render_band."
        "rasterize_lane_coverage_range_dependent",
        "output_encoding": "stacked per-pair coverage (N,render_h,render_w) float32 little-endian (<f4), C-order",
        "bit_exact_basis": "np.polyval Horner (fixed sequential mul+add, no FMA) + elementwise IEEE-754 fp64 "
        "(/ * - abs max min clip mod) + f64->f32 round-to-nearest-even cast; matmul/transcendental-FREE "
        "-> portable across x86-64 / ARM64 (no BLAS reduction order, no libm ULP gap)",
    }
    (GV / "levelset_lane_coverage_v1.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print("wrote lane-coverage golden vector to", GV)
    print(
        "  levelset_lane_coverage_v1 : n_pairs=%d render=%dx%d lines=%d dash_lines=%d "
        "nonzero_px=%d blob=%dB sha=%s"
        % (N_PAIRS, RENDER_H, RENDER_W, n_lines_total, n_dash, nonzero, len(blob), digest[:16])
    )
    print("  cross-check: shipped-inflate oracle == canonical source module (bit-identical). OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
