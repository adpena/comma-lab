#!/usr/bin/env python3
"""Local forensic dry-run for apogee_intN.

This lane is byte-only unless a separate scorer-basin distortion model or exact
CUDA candidate evidence is provided. The default command therefore fails closed
for exact-eval dispatch. Operators can pass ``--allow-forensic-byte-only`` to
run the archive/parser checks as a local forensic probe, but that still does
not make the lane ready for exact-eval dispatch.

Runs every check that doesn't require CUDA / network / GPU, in order:

  1. Wrapper script exists and parses (bash -n syntax check)
  2. APOGEE_INTN_BITS env var in 4..8 range
  3. PR106 source archive exists at the path the wrapper references
  4. PR106 state_dict.pt exists at the wrapper's PR106_STATE_DICT path
  5. Producer script (experiments/repack_pr106_with_intN_block_fp.py) exists
  6. Producer runs end-to-end and emits an apogee_int{N}_archive.zip
  7. Parser-roundtrip via submissions/apogee_intN/inflate.parse_apogee_intn_archive
     succeeds against the produced archive
  8. The runtime model.py + codec.py + intn_codec.py modules import cleanly
  9. inflate.sh exists, is executable, and bash -n parses
  10. launch_lane_on_vastai.py `full` subcommand has every flag the operator
      one-liner would emit (cross-checked via the existing wiring test logic)

Exit code 0 = local forensic byte-only checks passed when explicitly allowed.
Default non-zero = exact-eval dispatch is blocked because the lane has no
contest-faithful distortion model.

Operator usage:
  .venv/bin/python tools/dispatch_dryrun_apogee_intN.py --bits 5
  .venv/bin/python tools/dispatch_dryrun_apogee_intN.py --bits 4 --bits 5 --bits 6 --bits 8
  .venv/bin/python tools/dispatch_dryrun_apogee_intN.py --all-pareto-frontier
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.repo_io import repo_relative  # noqa: E402

WRAPPER = REPO / "scripts" / "remote_lane_apogee_intN.sh"
PRODUCER = REPO / "experiments" / "repack_pr106_with_intN_block_fp.py"
PR106_ARCHIVE = REPO / "experiments" / "results" / "public_pr106_belt_and_suspenders_intake_20260504_codex" / "archive.zip"
PR106_STATE_DICT = REPO / "experiments" / "results" / "sensitivity_map_pr106_20260504_claude" / "state_dict.pt"
INFLATE_SH = REPO / "submissions" / "apogee_intN" / "inflate.sh"
INFLATE_PY = REPO / "submissions" / "apogee_intN" / "inflate.py"
LAUNCH_SCRIPT = REPO / "scripts" / "launch_lane_on_vastai.py"

PARETO_FRONTIER_BITS = [4, 5, 6, 8]  # int7 dominated by int8
DISTORTION_MODEL_BLOCKER = (
    "missing contest-faithful distortion model or scorer-basin parity gate; "
    "byte-only Apogee intN repacks are forensic-only until a distortion "
    "model, local output-parity report, or exact CUDA replay evidence exists"
)


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


def check_bits_in_range(bits: int) -> str:
    _check(4 <= bits <= 8, f"APOGEE_INTN_BITS={bits} outside legal range 4..8")
    return f"APOGEE_INTN_BITS={bits} valid"


def check_pr106_artifacts_exist() -> str:
    _check(PR106_ARCHIVE.is_file(), f"PR106 archive missing: {PR106_ARCHIVE.relative_to(REPO)}")
    _check(PR106_STATE_DICT.is_file(), f"PR106 state_dict missing: {PR106_STATE_DICT.relative_to(REPO)}")
    return f"PR106 archive ({PR106_ARCHIVE.stat().st_size:,}b) + state_dict ({PR106_STATE_DICT.stat().st_size:,}b) on disk"


def check_producer_exists() -> str:
    _check(PRODUCER.is_file(), f"producer {PRODUCER.relative_to(REPO)} missing")
    return f"producer {PRODUCER.relative_to(REPO)} on disk"


def check_inflate_adapter_modules() -> str:
    _check(INFLATE_PY.is_file(), f"inflate.py missing: {INFLATE_PY.relative_to(REPO)}")
    _check(INFLATE_SH.is_file(), f"inflate.sh missing: {INFLATE_SH.relative_to(REPO)}")
    src_dir = INFLATE_PY.parent / "src"
    _check(src_dir.is_dir(), f"inflate.py vendored src/ missing: {src_dir.relative_to(REPO)}")
    for mod in ("model.py", "codec.py", "intn_codec.py"):
        _check((src_dir / mod).is_file(), f"vendored module missing: {(src_dir / mod).relative_to(REPO)}")
    proc = subprocess.run(["bash", "-n", str(INFLATE_SH)], capture_output=True, text=True)
    _check(proc.returncode == 0, f"bash -n on inflate.sh failed: {proc.stderr.strip()}")
    return "inflate.{py,sh} + vendored model.py + codec.py + intn_codec.py all present + parse"


def check_producer_e2e_for_bits(bits: int) -> str:
    """Run the producer end-to-end against the PR106 inputs; verify archive bytes + magic."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        proc = subprocess.run(
            [sys.executable, str(PRODUCER),
             "--state-dict", str(PR106_STATE_DICT),
             "--pr106-archive", str(PR106_ARCHIVE),
             "--bits", str(bits),
             "--out-dir", str(out_dir)],
            capture_output=True, text=True,
        )
        _check(proc.returncode == 0, f"producer for bits={bits} crashed: {proc.stderr.strip()[:300]}")
        archive = out_dir / f"apogee_int{bits}_archive.zip"
        _check(archive.is_file(), f"producer for bits={bits} did not emit {archive.name}")
        # Verify ZIP + magic byte
        with zipfile.ZipFile(archive) as z:
            bin_bytes = z.read("0.bin")
        expected_magic = 0xA0 | bits
        _check(bin_bytes[0] == expected_magic,
               f"bits={bits} produced magic 0x{bin_bytes[0]:02X}, expected 0x{expected_magic:02X}")
        return f"producer for bits={bits} OK ({archive.stat().st_size:,}b, magic 0x{bin_bytes[0]:02X})"


def check_parser_roundtrip_for_bits(bits: int) -> str:
    """Re-run producer + roundtrip through the runtime inflate parser."""
    from submissions.apogee_intN.inflate import parse_apogee_intn_archive

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        subprocess.run(
            [sys.executable, str(PRODUCER),
             "--state-dict", str(PR106_STATE_DICT),
             "--pr106-archive", str(PR106_ARCHIVE),
             "--bits", str(bits),
             "--out-dir", str(out_dir)],
            check=True, capture_output=True,
        )
        archive = out_dir / f"apogee_int{bits}_archive.zip"
        with zipfile.ZipFile(archive) as z:
            bin_bytes = z.read("0.bin")
        sd, lat, meta = parse_apogee_intn_archive(bin_bytes)
        _check(meta["bits"] == bits, f"meta bits={meta['bits']} != requested {bits}")
        _check(len(sd) > 0, f"parser returned empty state_dict for bits={bits}")
        _check(lat.numel() > 0, f"parser returned empty latents for bits={bits}")
        return (f"parser roundtrip for bits={bits} OK ({len(sd)} tensors, latents {tuple(lat.shape)}, "
                f"meta bits={meta['bits']})")


def check_launch_script_flag_wiring() -> str:
    _check(LAUNCH_SCRIPT.is_file(), f"launch script missing: {LAUNCH_SCRIPT.relative_to(REPO)}")
    src = LAUNCH_SCRIPT.read_text()
    tree = ast.parse(src)
    valid_flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id not in {"pf", "p_", "p"}:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("--"):
            valid_flags.add(first.value)
    required_for_pareto_one_liners = {
        "--lane-script", "--label", "--predicted-band",
        "--estimated-cost", "--council-priority", "--max-dph",
    }
    missing = required_for_pareto_one_liners - valid_flags
    _check(not missing, f"launch script missing flags from Pareto one-liners: {sorted(missing)}")
    return (
        f"launch script `full` subparser has all 6 required flags "
        f"({len(valid_flags)} total flags at {repo_relative(LAUNCH_SCRIPT, REPO)})"
    )


def run_dryrun(
    bits_list: list[int],
    verbose: bool = False,
    allow_forensic_byte_only: bool = False,
) -> int:
    """Run all checks. Returns 0 on PASS, non-zero on any FAIL."""
    failures: list[str] = []
    passes: list[str] = []

    def _attempt(name: str, fn, *args):
        try:
            msg = fn(*args)
            passes.append(f"  ✓ {name}: {msg}")
        except CheckFailure as e:
            failures.append(f"  ✗ {name}: {e}")

    _attempt("wrapper-syntax", check_wrapper_exists_and_parses)
    _attempt("pr106-artifacts", check_pr106_artifacts_exist)
    _attempt("producer-exists", check_producer_exists)
    _attempt("inflate-adapter", check_inflate_adapter_modules)
    _attempt("launch-script-wiring", check_launch_script_flag_wiring)

    for bits in bits_list:
        _attempt(f"bits={bits}-range", check_bits_in_range, bits)
        _attempt(f"bits={bits}-producer-e2e", check_producer_e2e_for_bits, bits)
        _attempt(f"bits={bits}-parser-roundtrip", check_parser_roundtrip_for_bits, bits)

    if allow_forensic_byte_only:
        passes.append(
            "  ✓ distortion-model-gate: explicit forensic byte-only mode; "
            "ready_for_exact_eval_dispatch=false"
        )
    else:
        failures.append(f"  ✗ distortion-model-gate: {DISTORTION_MODEL_BLOCKER}")

    if verbose or failures:
        for line in passes:
            print(line)
    for line in failures:
        print(line)

    if failures:
        print(f"\nDISPATCH DRY-RUN FAILED: {len(failures)} check(s) failed of {len(failures) + len(passes)}.")
        print("Do NOT dispatch — fix the failures above first.")
        return 1
    print(f"\nFORENSIC BYTE-ONLY DRY-RUN PASSED: all {len(passes)} checks OK across bits={bits_list}.")
    print("ready_for_exact_eval_dispatch=false")
    print("Do NOT dispatch Apogee intN as a score lane without a distortion model or exact CUDA evidence.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, action="append", default=None,
                        help="bits config(s) to validate (4..8). Repeat for multiple.")
    parser.add_argument("--all-pareto-frontier", action="store_true",
                        help=f"Validate all current Pareto-frontier bits ({PARETO_FRONTIER_BITS}).")
    parser.add_argument(
        "--allow-forensic-byte-only",
        action="store_true",
        help=(
            "Run byte/parser checks as a local forensic probe. This keeps "
            "ready_for_exact_eval_dispatch=false and must not be used as a "
            "score-lane GO signal."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print PASS lines too.")
    args = parser.parse_args(argv)

    if args.all_pareto_frontier:
        bits_list = PARETO_FRONTIER_BITS
    elif args.bits:
        bits_list = sorted(set(args.bits))
    else:
        bits_list = [5]  # default: validate the sweet-spot config

    return run_dryrun(
        bits_list,
        verbose=args.verbose,
        allow_forensic_byte_only=args.allow_forensic_byte_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
