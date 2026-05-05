#!/usr/bin/env python3
"""Dispatch dry-run for Lane Ω-W-V3: validate the lane WOULD succeed locally
without burning $0.30 on Vast.ai.

The wrapper (scripts/remote_lane_omega_w_v3_pr106.sh) has 4 stages:
  Stage 1 (CPU): extract PR106 HNeRV decoder
  Stage 2 (CUDA): build per-channel β-Fisher sensitivity_map.pt — REQUIRES GPU
  Stage 3 (CPU): repack via water_filling_codec_v2 → apogee_v2_archive.zip
  Stage 4 (CUDA): contest_auth_eval — REQUIRES GPU

This dry-run runs every check that doesn't require CUDA / network / GPU:

  1. wrapper-syntax: bash -n on scripts/remote_lane_omega_w_v3_pr106.sh
  2. pr106-artifacts: PR106 archive on disk
  3. stub-sensitivity-on-disk: all-ones sensitivity stub at canonical path
  4. extract-script-exists: experiments/extract_pr106_decoder.py
  5. repack-script-exists: experiments/repack_pr106_with_water_filling.py
  6. inflate-adapter: submissions/apogee_v2/inflate.{py,sh} + vendored modules
  7. stage1-extract-e2e: runs Stage 1 against PR106 archive locally
  8. stage3-repack-e2e: runs Stage 3 with stub sensitivity → byte-exact 164,087
  9. parser-roundtrip: parse_apogee_v2_archive recovers 28 tensors / 228,958 params

Exit 0 = GO. Non-zero = FAIL with per-check explanation.

Sister tool of tools/dispatch_dryrun_apogee_intN.py.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "remote_lane_omega_w_v3_pr106.sh"
EXTRACT_SCRIPT = REPO / "experiments" / "extract_pr106_decoder.py"
REPACK_SCRIPT = REPO / "experiments" / "repack_pr106_with_water_filling.py"
PR106_ARCHIVE = REPO / "experiments" / "results" / "public_pr106_belt_and_suspenders_intake_20260504_codex" / "archive.zip"
SENSITIVITY_STUB = REPO / "experiments" / "results" / "sensitivity_map_pr106_20260504_claude" / "sensitivity_map_stub.pt"
INFLATE_PY = REPO / "submissions" / "apogee_v2" / "inflate.py"
INFLATE_SH = REPO / "submissions" / "apogee_v2" / "inflate.sh"

EXPECTED_APOGEE_V2_BYTES = 164087
EXPECTED_TOTAL_PARAMS = 228958
EXPECTED_N_TENSORS = 28
EXPECTED_LATENT_SHAPE = (600, 28)


class CheckFailure(Exception):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def check_wrapper_exists_and_parses() -> str:
    _check(WRAPPER.is_file(), f"wrapper {WRAPPER.relative_to(REPO)} missing")
    proc = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True, text=True)
    _check(proc.returncode == 0, f"bash -n on wrapper failed: {proc.stderr.strip()}")
    return f"wrapper {WRAPPER.relative_to(REPO)} parses cleanly"


def check_pr106_artifact() -> str:
    _check(PR106_ARCHIVE.is_file(), f"PR106 archive missing: {PR106_ARCHIVE.relative_to(REPO)}")
    return f"PR106 archive ({PR106_ARCHIVE.stat().st_size:,}b) on disk"


def check_stub_sensitivity_on_disk() -> str:
    _check(SENSITIVITY_STUB.is_file(),
           f"sensitivity stub missing: {SENSITIVITY_STUB.relative_to(REPO)}")
    return f"stub sensitivity ({SENSITIVITY_STUB.stat().st_size:,}b) on disk"


def check_producer_scripts_exist() -> str:
    _check(EXTRACT_SCRIPT.is_file(), f"extract script missing: {EXTRACT_SCRIPT.relative_to(REPO)}")
    _check(REPACK_SCRIPT.is_file(), f"repack script missing: {REPACK_SCRIPT.relative_to(REPO)}")
    return "extract + repack scripts on disk"


def check_inflate_adapter_modules() -> str:
    _check(INFLATE_PY.is_file(), f"inflate.py missing: {INFLATE_PY.relative_to(REPO)}")
    _check(INFLATE_SH.is_file(), f"inflate.sh missing: {INFLATE_SH.relative_to(REPO)}")
    src_dir = INFLATE_PY.parent / "src"
    _check(src_dir.is_dir(), f"inflate.py vendored src/ missing: {src_dir.relative_to(REPO)}")
    for mod in ("model.py", "codec.py"):
        _check((src_dir / mod).is_file(), f"vendored module missing: {(src_dir / mod).relative_to(REPO)}")
    proc = subprocess.run(["bash", "-n", str(INFLATE_SH)], capture_output=True, text=True)
    _check(proc.returncode == 0, f"bash -n on inflate.sh failed: {proc.stderr.strip()}")
    return "apogee_v2 inflate.{py,sh} + vendored model.py + codec.py present + parse"


def check_stage1_extract_e2e(workdir: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(EXTRACT_SCRIPT),
         "--archive", str(PR106_ARCHIVE),
         "--out-dir", str(workdir)],
        capture_output=True, text=True,
    )
    _check(proc.returncode == 0, f"Stage 1 extract crashed: {proc.stderr.strip()[:300]}")
    for f in ("state_dict.pt", "latents.pt", "metadata.json"):
        _check((workdir / f).is_file(), f"Stage 1 did not emit {f}")
    sd_size = (workdir / "state_dict.pt").stat().st_size
    return f"Stage 1 extract OK (state_dict.pt {sd_size:,}b + latents.pt + metadata.json)"


def check_stage3_repack_byte_exact(workdir: Path) -> str:
    """Stage 3 must produce EXACTLY 164,087 bytes per the documented stub-mode invariant."""
    proc = subprocess.run(
        [sys.executable, str(REPACK_SCRIPT),
         "--state-dict", str(workdir / "state_dict.pt"),
         "--sensitivity", str(SENSITIVITY_STUB),
         "--pr106-archive", str(PR106_ARCHIVE),
         "--target-bytes", "145000",
         "--out-dir", str(workdir)],
        capture_output=True, text=True,
    )
    _check(proc.returncode == 0, f"Stage 3 repack crashed: {proc.stderr.strip()[:300]}")
    archive = workdir / "apogee_v2_archive.zip"
    _check(archive.is_file(), "Stage 3 did not emit apogee_v2_archive.zip")
    actual = archive.stat().st_size
    _check(actual == EXPECTED_APOGEE_V2_BYTES,
           f"Stage 3 byte drift: produced {actual:,}b, expected {EXPECTED_APOGEE_V2_BYTES:,}b "
           f"per documented stub-mode invariant. The codec changed without the wrapper-doc + "
           f"test_lane_omega_w_v3_local_smoke being updated.")
    return f"Stage 3 repack OK (apogee_v2_archive.zip {actual:,}b — byte-exact invariant held)"


def check_parser_roundtrip(workdir: Path) -> str:
    sys.path.insert(0, str(REPO))
    try:
        from submissions.apogee_v2.inflate import parse_apogee_v2_archive
    finally:
        sys.path.pop(0)
    archive = workdir / "apogee_v2_archive.zip"
    with zipfile.ZipFile(archive) as z:
        bin_bytes = z.read("0.bin")
    sd, lat, meta = parse_apogee_v2_archive(bin_bytes)
    _check(len(sd) == EXPECTED_N_TENSORS,
           f"parser returned {len(sd)} tensors, expected {EXPECTED_N_TENSORS}")
    _check(tuple(lat.shape) == EXPECTED_LATENT_SHAPE,
           f"parser returned latents shape {tuple(lat.shape)}, expected {EXPECTED_LATENT_SHAPE}")
    total_params = sum(t.numel() for t in sd.values())
    _check(total_params == EXPECTED_TOTAL_PARAMS,
           f"parser returned {total_params:,} params, expected {EXPECTED_TOTAL_PARAMS:,}")
    return (f"parser-roundtrip OK ({len(sd)} tensors, latents {tuple(lat.shape)}, "
            f"{total_params:,} params)")


def run_dryrun(verbose: bool = False) -> int:
    failures: list[str] = []
    passes: list[str] = []

    def _attempt(name: str, fn, *args):
        try:
            msg = fn(*args)
            passes.append(f"  ✓ {name}: {msg}")
        except CheckFailure as e:
            failures.append(f"  ✗ {name}: {e}")

    _attempt("wrapper-syntax", check_wrapper_exists_and_parses)
    _attempt("pr106-artifact", check_pr106_artifact)
    _attempt("stub-sensitivity", check_stub_sensitivity_on_disk)
    _attempt("producer-scripts", check_producer_scripts_exist)
    _attempt("inflate-adapter", check_inflate_adapter_modules)

    # Stage 1+3 + parser-roundtrip share the same workdir (chained)
    if not failures:  # only run e2e if structural checks all pass
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _attempt("stage1-extract-e2e", check_stage1_extract_e2e, workdir)
            if any(p for p in passes if "stage1-extract-e2e" in p):
                _attempt("stage3-repack-byte-exact", check_stage3_repack_byte_exact, workdir)
                if any(p for p in passes if "stage3-repack-byte-exact" in p):
                    _attempt("parser-roundtrip", check_parser_roundtrip, workdir)

    if verbose or failures:
        for line in passes:
            print(line)
    for line in failures:
        print(line)

    if failures:
        print(f"\nLANE Ω-W-V3 DISPATCH DRY-RUN FAILED: {len(failures)} of {len(failures) + len(passes)} checks failed.")
        print("Do NOT dispatch — fix the failures above first.")
        return 1
    print(f"\nLANE Ω-W-V3 DISPATCH DRY-RUN PASSED: all {len(passes)} checks OK.")
    print("Stages 1+3 + parser-roundtrip validated locally; Stages 2+4 require CUDA.")
    print("Dispatch is GO — operator can run `bash scripts/remote_lane_omega_w_v3_pr106.sh` "
          "on Vast.ai 4090 with confidence.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print PASS lines too.")
    args = parser.parse_args(argv)
    return run_dryrun(verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
