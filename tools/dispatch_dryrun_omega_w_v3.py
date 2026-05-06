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
  3. sensitivity-on-disk: default all-ones sensitivity stub, or --sensitivity
     if supplied
  3b. real-sensitivity-metadata: only with --require-real-sensitivity; rejects
      stub/planning/stale sensitivity maps and source-archive SHA mismatches
  4. extract-script-exists: experiments/extract_pr106_decoder.py
  5. repack-script-exists: experiments/repack_pr106_with_water_filling.py
  6. inflate-adapter: submissions/apogee_v2/inflate.{py,sh} + vendored modules
  7. stage1-extract-e2e: runs Stage 1 against PR106 archive locally
  8. stage3-repack-e2e: runs Stage 3 with stub sensitivity → byte-exact 164,087
  9. parser-roundtrip: parse_apogee_v2_archive recovers 28 tensors / 228,958 params

Exit 0 in default mode = local smoke/parity passed, not remote dispatch
readiness. Exit 0 with --require-real-sensitivity = local smoke/parity plus
real CUDA sensitivity provenance passed, so the remote CUDA wrapper is ready
to dispatch.

Sister tool of tools/dispatch_dryrun_apogee_intN.py.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.repo_io import repo_relative, sha256_file  # noqa: E402
from tools.preflight_cache import build_cache_key, load_valid_cache, write_pass_cache  # noqa: E402

WRAPPER = REPO / "scripts" / "remote_lane_omega_w_v3_pr106.sh"
EXTRACT_SCRIPT = REPO / "experiments" / "extract_pr106_decoder.py"
REPACK_SCRIPT = REPO / "experiments" / "repack_pr106_with_water_filling.py"
PR106_ARCHIVE = REPO / "experiments" / "results" / "public_pr106_belt_and_suspenders_intake_20260504_codex" / "archive.zip"
SENSITIVITY_STUB = REPO / "experiments" / "results" / "sensitivity_map_pr106_20260504_claude" / "sensitivity_map_stub.pt"
INFLATE_PY = REPO / "submissions" / "apogee_v2" / "inflate.py"
INFLATE_SH = REPO / "submissions" / "apogee_v2" / "inflate.sh"
INFLATE_SRC = INFLATE_PY.parent / "src"
WATER_FILLING_CODEC = REPO / "src" / "tac" / "water_filling_codec_v2.py"
SENSITIVITY_MAP = REPO / "src" / "tac" / "sensitivity_map.py"
CACHE_NAME = "dispatch_dryrun_omega_w_v3"

EXPECTED_APOGEE_V2_BYTES = 164087
EXPECTED_TOTAL_PARAMS = 228958
EXPECTED_N_TENSORS = 28
EXPECTED_LATENT_SHAPE = (600, 28)

REAL_SENSITIVITY_BAD_BOOL_KEYS = frozenset(
    {
        "advisory_only",
        "debug",
        "design_mode",
        "fake_sensitivity",
        "is_planning",
        "is_planning_only",
        "is_stub",
        "planning",
        "planning_only",
        "proxy_only",
        "random_sensitivity",
        "source_archive_stale",
        "stale",
        "stub",
        "uniform_sensitivity",
    }
)
REAL_SENSITIVITY_TEXT_KEYS = frozenset(
    {
        "evidence_grade",
        "kind",
        "mode",
        "notes",
        "provenance",
        "source",
        "source_kind",
        "status",
        "tag",
    }
)
REAL_SENSITIVITY_BAD_TEXT_MARKERS = (
    "stub",
    "planning-only",
    "planning_only",
    "design-mode",
    "design_mode",
    "advisory-only",
    "advisory_only",
    "advisory only",
    "stale",
    "superseded",
    "placeholder",
)
REAL_SENSITIVITY_SOURCE_SHA_KEYS = (
    "source_archive_sha256",
    "source_archive_sha",
    "baseline_archive_sha256",
)
REAL_SENSITIVITY_SOURCE_BYTES_KEYS = (
    "source_archive_bytes",
    "baseline_archive_bytes",
)
REAL_SENSITIVITY_CERTIFIED_SOURCE_SHA_KEYS = (
    "baseline_archive_sha256",
)
REAL_SENSITIVITY_CERTIFIED_SOURCE_BYTES_KEYS = (
    "baseline_archive_bytes",
)


class CheckFailure(Exception):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def _display_path(path: Path) -> str:
    return repo_relative(path, REPO)


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left == right


def _flatten_metadata(metadata: Mapping[str, Any], prefix: str = ""):
    for key, value in metadata.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten_metadata(value, full_key)
        else:
            yield full_key, value


def _metadata_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in metadata:
            return metadata[key]
    for container in ("source_archive", "baseline_archive", "source", "certification"):
        nested = metadata.get(container)
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested:
                    return nested[key]
            for key in ("sha256", "archive_sha256"):
                if key in nested and any(candidate.endswith("sha256") for candidate in keys):
                    return nested[key]
            if "bytes" in nested and any(candidate.endswith("bytes") for candidate in keys):
                return nested["bytes"]
    return None


def _real_sensitivity_metadata_blockers(metadata: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for full_key, value in _flatten_metadata(metadata):
        key = full_key.rsplit(".", 1)[-1]
        key_norm = key.lower().replace("-", "_")
        if key_norm in REAL_SENSITIVITY_BAD_BOOL_KEYS and value is True:
            blockers.append(f"{full_key}=true")
        if key_norm in REAL_SENSITIVITY_TEXT_KEYS and isinstance(value, str):
            value_norm = value.lower()
            for marker in REAL_SENSITIVITY_BAD_TEXT_MARKERS:
                if marker in value_norm:
                    blockers.append(f"{full_key} contains {marker!r}")
                    break
        if key_norm in {"status", "mode", "kind"} and isinstance(value, str):
            value_exact = value.strip().lower().replace("_", "-")
            if value_exact in {"planning", "design", "debug", "proxy", "prototype"}:
                blockers.append(f"{full_key}={value!r}")
    return blockers


def check_wrapper_exists_and_parses() -> str:
    _check(WRAPPER.is_file(), f"wrapper {WRAPPER.relative_to(REPO)} missing")
    proc = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True, text=True)
    _check(proc.returncode == 0, f"bash -n on wrapper failed: {proc.stderr.strip()}")
    return f"wrapper {WRAPPER.relative_to(REPO)} parses cleanly"


def check_pr106_artifact(source_archive: Path = PR106_ARCHIVE) -> str:
    _check(source_archive.is_file(), f"PR106 archive missing: {_display_path(source_archive)}")
    return f"PR106 archive ({source_archive.stat().st_size:,}b) on disk"


def check_sensitivity_on_disk(sensitivity_path: Path = SENSITIVITY_STUB) -> str:
    _check(sensitivity_path.is_file(),
           f"sensitivity artifact missing: {_display_path(sensitivity_path)}")
    kind = "stub sensitivity" if _same_path(sensitivity_path, SENSITIVITY_STUB) else "sensitivity artifact"
    return f"{kind} ({sensitivity_path.stat().st_size:,}b) on disk"


def check_real_sensitivity_metadata(
    sensitivity_path: Path,
    source_archive: Path = PR106_ARCHIVE,
) -> str:
    _check(source_archive.is_file(), f"source archive missing: {_display_path(source_archive)}")
    try:
        from tac.sensitivity_map import (
            SensitivityMapError,
            load_sensitivity_map,
            require_authoritative_device,
            validate_certified_sensitivity_map_metadata,
        )
    except Exception as e:  # pragma: no cover - import failure is environment-specific.
        raise CheckFailure(f"cannot import tac.sensitivity_map: {e}") from e

    try:
        sensitivities, metadata = load_sensitivity_map(sensitivity_path)
    except Exception as e:
        raise CheckFailure(f"{_display_path(sensitivity_path)} is not a valid sensitivity map: {e}") from e

    failures: list[str] = []
    if not sensitivities:
        failures.append("sensitivity map has no tensor entries")

    try:
        require_authoritative_device(metadata.get("device"))
    except SensitivityMapError as e:
        failures.append(str(e))
    try:
        certification = validate_certified_sensitivity_map_metadata(metadata)
    except SensitivityMapError as e:
        failures.append(f"certification rejected: {e}")
        certification = {}

    failures.extend(_real_sensitivity_metadata_blockers(metadata))

    actual_sha = sha256_file(source_archive)
    recorded_sha = _metadata_value(
        metadata,
        REAL_SENSITIVITY_SOURCE_SHA_KEYS + REAL_SENSITIVITY_CERTIFIED_SOURCE_SHA_KEYS,
    )
    if not isinstance(recorded_sha, str) or len(recorded_sha.strip()) != 64:
        failures.append("metadata missing 64-hex source archive SHA")
    elif recorded_sha.strip().lower() != actual_sha:
        failures.append(
            "metadata source archive SHA is stale or mismatched: "
            f"metadata={recorded_sha.strip().lower()} actual={actual_sha}"
        )
    certified_sha = certification.get("baseline_archive_sha256")
    if isinstance(certified_sha, str) and certified_sha != actual_sha:
        failures.append(
            "certification baseline_archive_sha256 is stale or mismatched: "
            f"metadata={certified_sha} actual={actual_sha}"
        )

    recorded_bytes = _metadata_value(
        metadata,
        REAL_SENSITIVITY_SOURCE_BYTES_KEYS + REAL_SENSITIVITY_CERTIFIED_SOURCE_BYTES_KEYS,
    )
    if recorded_bytes is not None:
        try:
            recorded_bytes_int = int(recorded_bytes)
        except (TypeError, ValueError):
            failures.append(f"metadata source archive bytes is not an integer: {recorded_bytes!r}")
        else:
            actual_bytes = source_archive.stat().st_size
            if recorded_bytes_int != actual_bytes:
                failures.append(
                    "metadata source archive bytes is stale or mismatched: "
                    f"metadata={recorded_bytes_int} actual={actual_bytes}"
                )
    certified_bytes = certification.get("baseline_archive_bytes")
    if isinstance(certified_bytes, int) and not isinstance(certified_bytes, bool):
        actual_bytes = source_archive.stat().st_size
        if certified_bytes != actual_bytes:
            failures.append(
                "certification baseline_archive_bytes is stale or mismatched: "
                f"metadata={certified_bytes} actual={actual_bytes}"
            )

    if failures:
        joined = "; ".join(failures)
        raise CheckFailure(f"real sensitivity metadata rejected: {joined}")

    return (
        f"real sensitivity metadata OK ({len(sensitivities)} tensors, "
        f"certified component={certification.get('component')!r}, "
        f"source sha {actual_sha[:16]}..., device={metadata.get('device')!r})"
    )


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


def check_stage1_extract_e2e(workdir: Path, source_archive: Path = PR106_ARCHIVE) -> str:
    try:
        from experiments.extract_pr106_decoder import extract_pr106_decoder

        extract_pr106_decoder(source_archive, workdir, verbose=False)
    except Exception as exc:
        raise CheckFailure(f"Stage 1 extract crashed: {type(exc).__name__}: {exc}") from exc
    for f in ("state_dict.pt", "latents.pt", "metadata.json"):
        _check((workdir / f).is_file(), f"Stage 1 did not emit {f}")
    sd_size = (workdir / "state_dict.pt").stat().st_size
    return f"Stage 1 extract OK (state_dict.pt {sd_size:,}b + latents.pt + metadata.json)"


def check_stage3_repack(
    workdir: Path,
    sensitivity_path: Path = SENSITIVITY_STUB,
    source_archive: Path = PR106_ARCHIVE,
    *,
    enforce_stub_byte_exact: bool = True,
) -> str:
    """Stage 3 must produce EXACTLY 164,087 bytes in default stub mode."""
    try:
        from experiments.repack_pr106_with_water_filling import repack_pr106_with_water_filling

        repack_pr106_with_water_filling(
            workdir / "state_dict.pt",
            sensitivity_path,
            source_archive,
            workdir,
            target_bytes=145000,
            verbose=False,
        )
    except Exception as exc:
        raise CheckFailure(f"Stage 3 repack crashed: {type(exc).__name__}: {exc}") from exc
    archive = workdir / "apogee_v2_archive.zip"
    _check(archive.is_file(), "Stage 3 did not emit apogee_v2_archive.zip")
    actual = archive.stat().st_size
    if enforce_stub_byte_exact:
        _check(actual == EXPECTED_APOGEE_V2_BYTES,
               f"Stage 3 byte drift: produced {actual:,}b, expected {EXPECTED_APOGEE_V2_BYTES:,}b "
               f"per documented stub-mode invariant. The codec changed without the wrapper-doc + "
               f"test_lane_omega_w_v3_local_smoke being updated.")
        return f"Stage 3 repack OK (apogee_v2_archive.zip {actual:,}b — byte-exact invariant held)"
    return f"Stage 3 repack OK (apogee_v2_archive.zip {actual:,}b with selected sensitivity)"


def check_parser_roundtrip(workdir: Path) -> str:
    from submissions.apogee_v2.inflate import parse_apogee_v2_archive

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


def _cache_files(sensitivity_path: Path, source_archive: Path) -> list[Path]:
    return [
        Path(__file__),
        WRAPPER,
        EXTRACT_SCRIPT,
        REPACK_SCRIPT,
        source_archive,
        sensitivity_path,
        INFLATE_PY,
        INFLATE_SH,
        INFLATE_SRC / "model.py",
        INFLATE_SRC / "codec.py",
        WATER_FILLING_CODEC,
        SENSITIVITY_MAP,
    ]


def _e2e_cache_key(
    sensitivity_path: Path,
    source_archive: Path,
    *,
    require_real_sensitivity: bool,
) -> dict[str, object]:
    return build_cache_key(
        name=CACHE_NAME,
        files=_cache_files(sensitivity_path, source_archive),
        config={
            "source_archive": _display_path(source_archive),
            "sensitivity": _display_path(sensitivity_path),
            "require_real_sensitivity": bool(require_real_sensitivity),
            "expected_bytes": EXPECTED_APOGEE_V2_BYTES,
            "expected_n_tensors": EXPECTED_N_TENSORS,
            "expected_latent_shape": list(EXPECTED_LATENT_SHAPE),
            "expected_total_params": EXPECTED_TOTAL_PARAMS,
        },
    )


def run_dryrun(
    verbose: bool = False,
    *,
    sensitivity_path: Path = SENSITIVITY_STUB,
    source_archive: Path = PR106_ARCHIVE,
    require_real_sensitivity: bool = False,
) -> int:
    failures: list[str] = []
    passes: list[str] = []

    def _attempt(name: str, fn, *args, **kwargs):
        try:
            msg = fn(*args, **kwargs)
            passes.append(f"  ✓ {name}: {msg}")
        except CheckFailure as e:
            failures.append(f"  ✗ {name}: {e}")

    _attempt("wrapper-syntax", check_wrapper_exists_and_parses)
    _attempt("pr106-artifact", check_pr106_artifact, source_archive)
    _attempt("sensitivity-on-disk", check_sensitivity_on_disk, sensitivity_path)
    if require_real_sensitivity:
        _attempt("real-sensitivity-metadata", check_real_sensitivity_metadata, sensitivity_path, source_archive)
    _attempt("producer-scripts", check_producer_scripts_exist)
    _attempt("inflate-adapter", check_inflate_adapter_modules)

    # Stage 1+3 + parser-roundtrip share the same workdir (chained)
    if not failures:  # only run e2e if structural checks all pass
        cache_key = _e2e_cache_key(
            sensitivity_path,
            source_archive,
            require_real_sensitivity=require_real_sensitivity,
        )
        cache_payload = load_valid_cache(CACHE_NAME, cache_key)
        if cache_payload is not None:
            result = cache_payload.get("result", {})
            passes.append(
                "  ✓ e2e-cache: SHA-tied Stage 1+3/parser cache valid "
                f"(archive {result.get('archive_size_bytes', '?')}b, "
                f"{result.get('n_tensors', '?')} tensors)"
            )
        else:
            with tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)
                _attempt("stage1-extract-e2e", check_stage1_extract_e2e, workdir, source_archive)
                if any(p for p in passes if "stage1-extract-e2e" in p):
                    enforce_stub_byte_exact = _same_path(sensitivity_path, SENSITIVITY_STUB)
                    stage3_name = "stage3-repack-byte-exact" if enforce_stub_byte_exact else "stage3-repack-e2e"
                    _attempt(
                        stage3_name,
                        check_stage3_repack,
                        workdir,
                        sensitivity_path,
                        source_archive,
                        enforce_stub_byte_exact=enforce_stub_byte_exact,
                    )
                    if any(p for p in passes if stage3_name in p):
                        _attempt("parser-roundtrip", check_parser_roundtrip, workdir)
                        if not failures:
                            archive = workdir / "apogee_v2_archive.zip"
                            write_pass_cache(
                                CACHE_NAME,
                                cache_key,
                                {
                                    "archive_size_bytes": archive.stat().st_size,
                                    "n_tensors": EXPECTED_N_TENSORS,
                                    "latent_shape": list(EXPECTED_LATENT_SHAPE),
                                    "total_params": EXPECTED_TOTAL_PARAMS,
                                    "score_claim": False,
                                    "ready_for_remote_cuda_dispatch": bool(require_real_sensitivity),
                                },
                            )

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
    if require_real_sensitivity:
        print("ready_for_remote_cuda_dispatch=true")
        print("Remote CUDA dispatch is allowed for `bash scripts/remote_lane_omega_w_v3_pr106.sh`.")
    else:
        print("ready_for_remote_cuda_dispatch=false")
        print(
            "Default stub-mode is local smoke only. Re-run with "
            "--require-real-sensitivity and a CUDA sensitivity map before dispatch."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print PASS lines too.")
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=SENSITIVITY_STUB,
        help="Sensitivity map artifact for Stage 3. Defaults to the existing stub-mode artifact.",
    )
    parser.add_argument(
        "--pr106-archive",
        type=Path,
        default=PR106_ARCHIVE,
        help="Source PR106 archive whose SHA must match strict sensitivity metadata.",
    )
    parser.add_argument(
        "--require-real-sensitivity",
        action="store_true",
        help=(
            "Fail unless --sensitivity is a non-stub CUDA sensitivity map whose metadata "
            "records the selected source archive SHA."
        ),
    )
    args = parser.parse_args(argv)
    return run_dryrun(
        verbose=args.verbose,
        sensitivity_path=args.sensitivity,
        source_archive=args.pr106_archive,
        require_real_sensitivity=args.require_real_sensitivity,
    )


if __name__ == "__main__":
    raise SystemExit(main())
