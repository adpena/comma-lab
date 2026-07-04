# SPDX-License-Identifier: MIT
"""End-to-end swap-in parity + speedup: numpy inflate `_lane_coverage` vs the Rust port.

Proves that swapping the bit-exact Rust AA-SDF rasterizer INTO the numpy inflate leaves
the decoded output BYTE-IDENTICAL — the property that makes it a SHIP candidate (unlike
the fp32 raster #281 / neural forward #282, which are not portable).

What it does (all on REAL data, $0 / CPU, #205-untouched):
  1. Extract the ACTUAL shipped inflate `_lane_parse` + `_lane_coverage` from
     `tools/levelset_byte_close_and_eval.py::_INFLATE_PY` (the numpy oracle).
  2. Load the committed golden-vector LBND1 blob (REAL lane coeffs fitted from the
     frozen SegNet argmax `gt_n96`).
  3. numpy path:  parse -> per-pair coverage -> stack (N, rh, rw) float32.
  4. Rust path:   load the `cdylib` via `ctypes` (NO PyO3), call
     `tac_lane_coverage_all_lbnd1` on the SAME blob -> stacked float32.
  5. Assert sha256(numpy stack `<f4`) == sha256(rust stack `<f4`)  -> BYTE-IDENTICAL.
  6. Time both -> report the Rust speedup on the per-pixel raster hot path + estimate
     its share of the contest 4-core ~16-min inflate (MEANS; pointer 0.19110 UNMOVED).

Run:  .venv/bin/python runtime-rs/crates/tac-levelset-inflate/lane_end_to_end_parity.py
Exit 0 = byte-identical (Rust ships where numpy runs). Nonzero = divergence.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GV = HERE / "golden_vectors"
REPO = HERE.parents[2]
RUNTIME_RS = REPO / "runtime-rs"
TEMPLATE = REPO / "tools/levelset_byte_close_and_eval.py"

sys.path.insert(0, str(REPO / "src"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_shipped_inflate_lane_fns():
    spec = importlib.util.spec_from_file_location("_lvls_byte_close_template_e2e", str(TEMPLATE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns: dict = {"__name__": "_shipped_inflate_oracle_e2e"}
    exec(compile(mod._INFLATE_PY, "<shipped_inflate_py>", "exec"), ns)
    return ns["_lane_parse"], ns["_lane_coverage"]


def _find_or_build_cdylib() -> Path:
    names = ["libtac_levelset_inflate.dylib", "libtac_levelset_inflate.so"]
    cands = [RUNTIME_RS / "target" / "release" / n for n in names]
    cands += [REPO / "target" / "release" / n for n in names]
    for c in cands:
        if c.exists():
            return c
    # Build it (offline; $0 / CPU).
    subprocess.run(
        ["cargo", "build", "--release", "-p", "tac-levelset-inflate", "--offline"],
        cwd=str(RUNTIME_RS), check=True,
    )
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"cdylib not found after build; looked in {[str(c) for c in cands]}")


def _rust_coverage_all(lib, blob: bytes, rh: int, rw: int) -> np.ndarray:
    u8 = ctypes.POINTER(ctypes.c_uint8)
    lib.tac_lane_band_n_pairs.restype = ctypes.c_int64
    lib.tac_lane_band_n_pairs.argtypes = [u8, ctypes.c_size_t]
    lib.tac_lane_coverage_all_lbnd1.restype = ctypes.c_int64
    lib.tac_lane_coverage_all_lbnd1.argtypes = [
        u8, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]
    blob_arr = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
    n = int(lib.tac_lane_band_n_pairs(blob_arr, len(blob)))
    if n < 0:
        raise RuntimeError(f"tac_lane_band_n_pairs error {n}")
    out_len = n * rh * rw
    out = (ctypes.c_float * out_len)()
    got = int(lib.tac_lane_coverage_all_lbnd1(blob_arr, len(blob), rh, rw, out, out_len))
    if got != n:
        raise RuntimeError(f"tac_lane_coverage_all_lbnd1 error/short {got} != {n}")
    return np.frombuffer(out, dtype=np.float32).reshape(n, rh, rw)


def main() -> int:
    manifest_path = GV / "levelset_lane_coverage_v1.json"
    band_path = GV / "levelset_lane_coverage_v1_band.bin"
    if not manifest_path.exists() or not band_path.exists():
        print("golden vector absent — run generate_lane_golden_vectors.py first", file=sys.stderr)
        return 2
    import json

    m = json.loads(manifest_path.read_text())
    rh, rw, n_pairs = int(m["render_h"]), int(m["render_w"]), int(m["n_pairs"])
    blob = band_path.read_bytes()

    _lane_parse, _lane_coverage = _load_shipped_inflate_lane_fns()
    parsed_pairs, hdr = _lane_parse(blob)
    assert len(parsed_pairs) == n_pairs

    # numpy oracle stack + timing (the per-pixel raster hot path).
    t0 = time.perf_counter()
    np_stack = np.stack(
        [_lane_coverage(parsed_pairs[i], rh, rw, hdr) for i in range(n_pairs)], axis=0
    ).astype("<f4")
    t_np = time.perf_counter() - t0

    lib = ctypes.CDLL(str(_find_or_build_cdylib()))
    # warm-up (page-in / branch predictor) then time.
    _ = _rust_coverage_all(lib, blob, rh, rw)
    t1 = time.perf_counter()
    rs_stack = _rust_coverage_all(lib, blob, rh, rw).astype("<f4")
    t_rs = time.perf_counter() - t1

    np_sha = _sha(np_stack.tobytes(order="C"))
    rs_sha = _sha(rs_stack.tobytes(order="C"))
    manifest_sha = m["sha256"]

    ok = np_sha == rs_sha == manifest_sha
    print("=== lane AA-SDF coverage — end-to-end swap-in parity (#283) ===")
    print(f"  n_pairs={n_pairs}  render={rh}x{rw}  (>=24 pairs: {'YES' if n_pairs >= 24 else 'NO'})")
    print(f"  numpy  stack sha256 = {np_sha}")
    print(f"  rust   stack sha256 = {rs_sha}")
    print(f"  manifest oracle sha = {manifest_sha}")
    print(f"  BYTE-IDENTICAL (numpy == rust == oracle): {'YES' if ok else 'NO'}")
    if not ok:
        # Localize the first divergence for diagnostics.
        diff = np.argwhere(np_stack != rs_stack)
        print(f"  FIRST DIVERGENCE(S): {diff[:5].tolist()}  (total {len(diff)})", file=sys.stderr)
        return 1

    speedup = (t_np / t_rs) if t_rs > 0 else float("inf")
    per_pair_np_ms = t_np / n_pairs * 1e3
    per_pair_rs_ms = t_rs / n_pairs * 1e3
    # The shipped inflate rasterizes coverage ONCE per pair over 600 pairs (shared across
    # the pair's 2 frames). Extrapolate the per-pair raster cost to n600.
    np_n600_s = per_pair_np_ms * 600 / 1e3
    rs_n600_s = per_pair_rs_ms * 600 / 1e3
    print("  --- wall-clock (MEANS; decode-time only; pointer 0.19110 UNMOVED) ---")
    print(f"  numpy _lane_coverage : {per_pair_np_ms:.4f} ms/pair  ({t_np*1e3:.2f} ms / {n_pairs} pairs)")
    print(f"  rust  lane_coverage  : {per_pair_rs_ms:.4f} ms/pair  ({t_rs*1e3:.2f} ms / {n_pairs} pairs)")
    print(f"  speedup (numpy/rust) : {speedup:.2f}x  on the per-pixel raster hot path")
    print(f"  extrapolated n600 raster: numpy ~{np_n600_s:.3f} s  ->  rust ~{rs_n600_s:.3f} s "
          f"(saves ~{np_n600_s - rs_n600_s:.3f} s of the ~16-min inflate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
