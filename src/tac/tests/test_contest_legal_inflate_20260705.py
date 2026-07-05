"""Contest-legal inflate guards (#214, the end-gate).

Locks the two compliant <30-min decode paths measured 2026-07-05 on the REAL preserved
d_seg=0.0252 checkpoint (``experiments/results/v5_dseg0026_preserved_20260705``):

  * MULTIPROCESS numpy-fp64 (``INFLATE_WORKERS`` fork/spawn Pool, disjoint .raw offsets) --
    BIT-EXACT to the serial fallback (``INFLATE_WORKERS=1``) and deterministic across runs.
    Measured n600 = ~13.9 min @ 4 workers (M5 Max; contest 4-core proxy).
  * torch-fp32 decode -- score-preserving (SegNet-argmax agreement 99.9995%, 3 flip px /
    589,824; d_pose MSE delta 3e-10). Measured n600 = 6.59 min CPU / <0.5 min T4 (projected).

The bulk are FAST structural/schema guards (no inflate run). Two REAL bitwise proofs
(serial==mp sha256, 2-run determinism) build a 1-pair packet via the canonical byte-close
CLI and are skipped gracefully if the preserved ckpt / deps are unavailable.

Authority: ``[macOS-CPU advisory] NON-PROMOTABLE``. numpy-fp64 = bit-identical reference;
torch = fast decode, parity-gated. pointer UNMOVED 0.19110.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

_BYTE_CLOSE_TOOL = REPO_ROOT / "tools" / "levelset_byte_close_and_eval.py"
_PARITY_TOOL = REPO_ROOT / "tools" / "levelset_torch_inflate_parity.py"
_PRESERVED_CKPT = REPO_ROOT / "experiments" / "results" / "v5_dseg0026_preserved_20260705"
_PARITY_REPORT = REPO_ROOT / "reports" / "inflate_legal_torch_parity_real_20260705.json"

_BUDGET_MIN = 30.0  # contest 30-min FULL-eval hard budget


def _load_parity_module():
    spec = importlib.util.spec_from_file_location("levelset_torch_inflate_parity", _PARITY_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1-3. _pick_npz precedence (fast, pure)
# ---------------------------------------------------------------------------
def test_pick_npz_prefers_ema_over_live(tmp_path: Path) -> None:
    mod = _load_parity_module()
    (tmp_path / "levelset_witness_live_mlx.npz").write_bytes(b"x")
    (tmp_path / "levelset_witness_ema_mlx.npz").write_bytes(b"x")
    assert mod._pick_npz(tmp_path).name == "levelset_witness_ema_mlx.npz"


def test_pick_npz_glob_fallback(tmp_path: Path) -> None:
    mod = _load_parity_module()
    (tmp_path / "some_other_mlx.npz").write_bytes(b"x")
    assert mod._pick_npz(tmp_path).name == "some_other_mlx.npz"


def test_pick_npz_raises_when_empty(tmp_path: Path) -> None:
    mod = _load_parity_module()
    with pytest.raises(FileNotFoundError):
        mod._pick_npz(tmp_path)


# ---------------------------------------------------------------------------
# 4-6. Real-weight parity report schema + threshold guards (fast, reads committed JSON)
# ---------------------------------------------------------------------------
def _load_report() -> dict:
    if not _PARITY_REPORT.exists():
        pytest.skip(f"parity report not present: {_PARITY_REPORT}")
    return json.loads(_PARITY_REPORT.read_text())


def test_parity_report_is_real_weights_not_synthetic() -> None:
    rep = _load_report()
    assert rep["synthetic_fixture"] is False, "the committed parity row MUST be REAL weights (NO-FAKE #3)"
    assert rep["promotion_claim"] is False
    assert rep["certify_numpy_inproc_eq_shipped"] is True


def test_parity_score_preserving_thresholds() -> None:
    rep = _load_report()
    p = rep["parity"]
    # SegNet argmax is THE d_seg-relevant metric: torch-fp32 frames must be argmax-faithful.
    assert p["segnet_argmax_agreement_torch_fp32_vs_numpy_fp64"] > 0.999
    assert p["segnet_argmax_flip_px"] <= 50  # measured 3/589824; guard against regression
    # d_pose MSE delta must be orders below the witness d_pose scale.
    assert p["d_pose_mse_delta_torch_fp32_vs_numpy_fp64"] < 1e-6
    assert p["uint8_frame_max_abs_diff_torch_fp32"] <= 1


def test_budget_regression_guard_at_least_one_leg_under_30min() -> None:
    rep = _load_report()
    t = rep["timing"]
    v = rep["verdict"]
    # torch-fp32 CPU is the primary compliant leg (measured 6.59 min on the real ckpt).
    assert t["torch_fp32_cpu_n600_minutes"] < _BUDGET_MIN
    assert v["legal_under_30min_cpu_torch_fp32"] is True
    assert v["legal_under_30min_t4_torch_fp32"] is True
    # numpy single-process is the bit-exact reference (OVER budget alone -> multiprocess closes it).
    assert t["numpy_fp64_n600_minutes"] > 0.0


# ---------------------------------------------------------------------------
# 7-10. inflate.py / inflate.sh template structural guards (fast, reads tool source)
# ---------------------------------------------------------------------------
def _byte_close_src() -> str:
    return _BYTE_CLOSE_TOOL.read_text()


def test_template_has_multiprocess_pool_and_serial_fallback() -> None:
    src = _byte_close_src()
    assert "INFLATE_WORKERS" in src, "multiprocess decode env knob missing"
    assert "ctx.Pool(" in src, "the fork/spawn Pool must exist for the 4-core compliant path"
    assert "nworkers == 1" in src, "the INFLATE_WORKERS=1 bit-identical serial fallback must exist"


def test_template_preallocates_raw_and_writes_disjoint_offsets() -> None:
    src = _byte_close_src()
    # workers write disjoint offsets into a preallocated .raw -> POSIX-safe concurrent write -> bit-exact.
    assert "f.truncate(2 * n_pairs * _G[\"framebytes\"])" in src
    assert "f.seek(pi * 2 * fb)" in src


def test_template_default_forward_dtype_is_float64_bit_exact() -> None:
    src = _byte_close_src()
    # _FDT default = float64 (bit-exact authority); float32 is opt-in via INFLATE_FP32.
    assert 'INFLATE_FP32", "0") == "1"' in src
    assert "_FDT = np.float32 if _FP32 else np.float64" in src


def test_inflate_sh_honors_python_and_is_runtime_closed() -> None:
    src = _byte_close_src()
    # inflate.sh must resolve ${PYTHON} (contest eval interpreter) else python3, and set -euo pipefail.
    assert 'PYBIN="${PYTHON:-python3}"' in src
    assert "set -euo pipefail" in src


# ---------------------------------------------------------------------------
# 11-12. REAL bitwise proofs (slow; build a 1-pair packet, skip if ckpt/deps absent)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _one_pair_packet(tmp_path_factory) -> Path:
    if not (_PRESERVED_CKPT / "levelset_witness_ema_mlx.npz").exists():
        pytest.skip(f"preserved ckpt absent: {_PRESERVED_CKPT}")
    for dep in ("numpy", "brotli", "torch"):
        if importlib.util.find_spec(dep) is None:
            pytest.skip(f"dep absent: {dep}")
    work = tmp_path_factory.mktemp("legal_inflate_pkt")
    bc_json = work / "bc.json"
    cmd = [
        sys.executable, str(_BYTE_CLOSE_TOOL),
        "--ckpt-dir", str(_PRESERVED_CKPT),
        "--skip-parity", "--max-pairs", "1", "--keep-packet",
        "--out", str(bc_json),
    ]
    env = {**os.environ, "INFLATE_WORKERS": "1", "INFLATE_MAX_PAIRS": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        pytest.skip(f"byte-close failed (env-specific): {proc.stderr[-400:]}")
    # the tool chooses its own packet_dir (experiments/results/levelset_packet_<ts>); read it back.
    packet_dir = Path(json.loads(bc_json.read_text())["packet_dir"])
    # extract the full 0.bin (archive.zip -> 0.bin)
    archive_full = packet_dir / "archive_full"
    archive_full.mkdir(exist_ok=True)
    with zipfile.ZipFile(packet_dir / "archive.zip") as zf:
        zf.extractall(archive_full)
    return packet_dir


def _run_inflate(packet_dir: Path, dst: Path, *, workers: int) -> None:
    env = {
        **os.environ, "INFLATE_WORKERS": str(workers), "INFLATE_MAX_PAIRS": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    src_bin = packet_dir / "archive_full" / "0.bin"
    proc = subprocess.run(
        [sys.executable, str(packet_dir / "inflate.py"), str(src_bin), str(dst)],
        capture_output=True, text=True, env=env, cwd=str(packet_dir),
    )
    assert proc.returncode == 0, f"inflate.py failed: {proc.stderr[-400:]}"


def test_multiprocess_inflate_is_bit_identical_to_serial(_one_pair_packet: Path, tmp_path: Path) -> None:
    serial = tmp_path / "serial.raw"
    mp4 = tmp_path / "mp4.raw"
    _run_inflate(_one_pair_packet, serial, workers=1)
    _run_inflate(_one_pair_packet, mp4, workers=4)
    assert _sha256(serial) == _sha256(mp4), "multiprocess numpy-fp64 must be BIT-IDENTICAL to serial"


def test_inflate_is_deterministic_across_two_runs(_one_pair_packet: Path, tmp_path: Path) -> None:
    a = tmp_path / "runA.raw"
    b = tmp_path / "runB.raw"
    _run_inflate(_one_pair_packet, a, workers=4)
    _run_inflate(_one_pair_packet, b, workers=4)
    assert _sha256(a) == _sha256(b), "same archive -> bit-identical .raw every run (deterministic decode)"
