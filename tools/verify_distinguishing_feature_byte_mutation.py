# SPDX-License-Identifier: MIT
"""Verify a substrate's distinguishing-feature bytes flow end-to-end.

Canonical helper for the Distinguishing-Feature Integration Contract
(Catalog #272). Mutates one byte at every declared distinguishing-bytes
offset of an archive, re-runs the inflate path, and confirms the
inflated output frames CHANGE (i.e. the bytes are operationally
consumed, not dead-section padding).

This is the per-substrate-distinguishing-feature variant of
``_verify_runtime_consumes_payload_bytes_executable`` (Catalog #139's
generic per-archive byte-mutation smoke). It is more pointed: instead
of checking that *some* byte mutation changes output, it checks that
mutating the bytes claimed as distinguishing changes output. A
substrate that ships a 1 KB hyperprior-CDF section but never reads it
during inflate (the Z3-G1 anchor) FAILS this verifier even though
random-byte mutation of unrelated sections (e.g. renderer.bin) WOULD
pass Catalog #139.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable + "HNeRV / leaderboard-implementation parity discipline"
lesson 11 (no-op detector + provenance).

Usage
-----

    .venv/bin/python tools/verify_distinguishing_feature_byte_mutation.py \\
        --archive <path/to/archive.zip> \\
        --inflate-sh <path/to/inflate.sh> \\
        --distinguishing-bytes-path <archive_section_name> \\
        [--distinguishing-bytes-path <another_section>] ... \\
        --output-json <experiments/results/<lane>/distinguishing_feature_byte_mutation_proof.json> \\
        [--mutation-offsets-per-section 4] \\
        [--epsilon-hash-bytes 0]

Exit codes
----------

  0 = PASSED — at least one mutation per distinguishing section produced
      a different inflated output (bytes are operationally consumed).
  1 = FAILED — at least one declared distinguishing section did NOT
      change inflated output under any mutation (the section is dead
      and the substrate is in the "research-substrate trap" per
      CLAUDE.md 8th forbidden pattern).
  2 = INFRASTRUCTURE_ERROR — missing archive / inflate.sh / declared
      section not found in archive / etc. (use blockers, not refusal).

Output JSON schema
------------------

    {
      "schema_version": 1,
      "archive_sha256": "<sha>",
      "archive_size_bytes": <int>,
      "inflate_sh_sha256": "<sha>",
      "distinguishing_bytes_paths": [...],
      "section_results": [
        {
          "section": "<name>",
          "section_size_bytes": <int>,
          "mutations_attempted": <int>,
          "mutations_changed_output": <int>,
          "passed": <bool>,
          "first_changed_inflated_output_sha256": "<sha or null>",
          "baseline_inflated_output_sha256": "<sha>"
        },
        ...
      ],
      "overall_passed": <bool>,
      "verdict": "PASSED" | "FAILED" | "INFRASTRUCTURE_ERROR",
      "elapsed_seconds": <float>
    }
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_MUTATIONS_PER_SECTION = 4


@dataclasses.dataclass
class SectionResult:
    section: str
    section_size_bytes: int
    mutations_attempted: int
    mutations_changed_output: int
    passed: bool
    first_changed_inflated_output_sha256: str | None
    baseline_inflated_output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _list_archive_sections(archive: Path) -> list[tuple[str, int]]:
    """Return (member_name, size) for every entry in the archive ZIP."""
    sections: list[tuple[str, int]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            sections.append((info.filename, info.file_size))
    return sections


def _read_archive_section(archive: Path, section: str) -> bytes:
    with zipfile.ZipFile(archive, "r") as zf:
        return zf.read(section)


def _write_archive_with_mutated_section(
    src_archive: Path,
    dst_archive: Path,
    section: str,
    mutated_bytes: bytes,
) -> None:
    """Copy src_archive to dst_archive, replacing one section's bytes."""
    with zipfile.ZipFile(src_archive, "r") as src_zf:
        with zipfile.ZipFile(
            dst_archive, "w", compression=zipfile.ZIP_STORED
        ) as dst_zf:
            for info in src_zf.infolist():
                if info.filename == section:
                    dst_zf.writestr(info.filename, mutated_bytes)
                else:
                    dst_zf.writestr(info, src_zf.read(info.filename))


def _hash_directory_contents(directory: Path) -> str:
    """SHA-256 of every regular file in the directory (sorted by relpath)."""
    h = hashlib.sha256()
    files: list[tuple[str, Path]] = []
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(directory))
            files.append((rel, p))
    if not files:
        return ""
    for rel, p in files:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _run_inflate(
    inflate_sh: Path,
    archive_dir: Path,
    output_dir: Path,
    file_list: Path,
    timeout_seconds: int = 600,
) -> tuple[int, str]:
    """Run inflate.sh; return (returncode, output_dir_hash)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(inflate_sh),
        str(archive_dir),
        str(output_dir),
        str(file_list),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return 124, ""
    if result.returncode != 0:
        return result.returncode, ""
    return 0, _hash_directory_contents(output_dir)


def _generate_mutations(
    section_bytes: bytes,
    n: int,
    seed_offset: int = 0,
) -> list[bytes]:
    """Generate n distinct single-byte mutations of section_bytes.

    Picks evenly-spaced offsets across the section (with seed_offset
    rotation so two callers don't collide on identical mutations).
    Skips empty sections (n=0 mutations).
    """
    size = len(section_bytes)
    if size == 0:
        return []
    n = min(n, size)
    mutations: list[bytes] = []
    # Pick n offsets evenly spaced across the section.
    for i in range(n):
        offset = (seed_offset + (i * max(1, size // n))) % size
        original = section_bytes[offset]
        # Flip the high bit to guarantee a distinct value.
        mutated_value = (original ^ 0x80) & 0xFF
        if mutated_value == original:
            mutated_value = (original + 1) & 0xFF
        mutated = bytearray(section_bytes)
        mutated[offset] = mutated_value
        mutations.append(bytes(mutated))
    return mutations


def verify_distinguishing_feature_byte_mutation(
    *,
    archive: Path,
    inflate_sh: Path,
    distinguishing_bytes_paths: list[str],
    output_json: Path,
    mutations_per_section: int = DEFAULT_MUTATIONS_PER_SECTION,
    file_list: Path | None = None,
    inflate_timeout_seconds: int = 600,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the distinguishing-feature byte-mutation smoke.

    Returns the JSON-serializable result dict. Writes the same dict to
    ``output_json``.
    """
    started = time.time()

    if not archive.is_file():
        result = _infra_error(
            f"archive {archive} does not exist",
            distinguishing_bytes_paths,
            output_json,
            started,
        )
        return result
    if not inflate_sh.is_file():
        result = _infra_error(
            f"inflate_sh {inflate_sh} does not exist",
            distinguishing_bytes_paths,
            output_json,
            started,
        )
        return result

    archive_sha = _sha256_file(archive)
    archive_size = archive.stat().st_size
    inflate_sha = _sha256_file(inflate_sh)

    # List sections and verify all declared distinguishing paths exist.
    section_index = {name: size for name, size in _list_archive_sections(archive)}
    missing = [p for p in distinguishing_bytes_paths if p not in section_index]
    if missing:
        result = _infra_error(
            f"distinguishing_bytes_paths not in archive: {missing}",
            distinguishing_bytes_paths,
            output_json,
            started,
            archive_sha=archive_sha,
            archive_size_bytes=archive_size,
            inflate_sh_sha256=inflate_sha,
        )
        return result

    with tempfile.TemporaryDirectory(prefix="dfic_") as tmp_str:
        tmp = Path(tmp_str)
        # Stage archive_dir (inflate.sh expects a directory containing the ZIP).
        archive_dir = tmp / "archive_dir"
        archive_dir.mkdir(parents=True, exist_ok=True)
        staged_archive = archive_dir / archive.name
        shutil.copy2(archive, staged_archive)

        # File list — minimal default if not provided.
        if file_list is None:
            file_list_path = tmp / "file_list.txt"
            file_list_path.write_text("0.mkv\n", encoding="utf-8")
        else:
            file_list_path = file_list

        # Baseline inflate.
        baseline_out = tmp / "baseline_out"
        rc, baseline_hash = _run_inflate(
            inflate_sh, archive_dir, baseline_out, file_list_path,
            timeout_seconds=inflate_timeout_seconds,
        )
        if rc != 0 or not baseline_hash:
            result = _infra_error(
                f"baseline inflate failed rc={rc}",
                distinguishing_bytes_paths,
                output_json,
                started,
                archive_sha=archive_sha,
                archive_size_bytes=archive_size,
                inflate_sh_sha256=inflate_sha,
            )
            return result

        if verbose:
            print(f"[dfic] baseline inflate OK hash={baseline_hash[:16]}")

        # Per-section mutation runs.
        section_results: list[SectionResult] = []
        for sec_idx, section in enumerate(distinguishing_bytes_paths):
            section_bytes = _read_archive_section(archive, section)
            section_size = len(section_bytes)
            if section_size == 0:
                # Empty distinguishing section: VIOLATION (Z3-G1 anchor).
                section_results.append(
                    SectionResult(
                        section=section,
                        section_size_bytes=0,
                        mutations_attempted=0,
                        mutations_changed_output=0,
                        passed=False,
                        first_changed_inflated_output_sha256=None,
                        baseline_inflated_output_sha256=baseline_hash,
                    )
                )
                if verbose:
                    print(
                        f"[dfic] section={section!r} EMPTY (0 bytes) "
                        "FAIL — distinguishing feature has zero bytes"
                    )
                continue

            mutations = _generate_mutations(
                section_bytes, mutations_per_section,
                seed_offset=sec_idx * 7,
            )
            changed = 0
            first_changed_hash: str | None = None
            for m_idx, mutated in enumerate(mutations):
                mut_archive = tmp / f"mut_{sec_idx}_{m_idx}_{archive.name}"
                _write_archive_with_mutated_section(
                    staged_archive, mut_archive, section, mutated,
                )
                # Stage mutated archive and re-inflate.
                mut_archive_dir = tmp / f"mut_archive_dir_{sec_idx}_{m_idx}"
                mut_archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mut_archive, mut_archive_dir / archive.name)
                mut_out = tmp / f"mut_out_{sec_idx}_{m_idx}"
                rc, mut_hash = _run_inflate(
                    inflate_sh, mut_archive_dir, mut_out, file_list_path,
                    timeout_seconds=inflate_timeout_seconds,
                )
                if rc == 0 and mut_hash and mut_hash != baseline_hash:
                    changed += 1
                    if first_changed_hash is None:
                        first_changed_hash = mut_hash

            section_results.append(
                SectionResult(
                    section=section,
                    section_size_bytes=section_size,
                    mutations_attempted=len(mutations),
                    mutations_changed_output=changed,
                    passed=changed > 0,
                    first_changed_inflated_output_sha256=first_changed_hash,
                    baseline_inflated_output_sha256=baseline_hash,
                )
            )
            if verbose:
                verdict = "PASS" if changed > 0 else "FAIL"
                print(
                    f"[dfic] section={section!r} size={section_size} "
                    f"mutations={len(mutations)} changed={changed} "
                    f"verdict={verdict}"
                )

    overall_passed = all(s.passed for s in section_results) and bool(section_results)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "inflate_sh_sha256": inflate_sha,
        "distinguishing_bytes_paths": list(distinguishing_bytes_paths),
        "section_results": [s.to_dict() for s in section_results],
        "overall_passed": overall_passed,
        "verdict": "PASSED" if overall_passed else "FAILED",
        "elapsed_seconds": round(time.time() - started, 3),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _infra_error(
    reason: str,
    distinguishing_bytes_paths: list[str],
    output_json: Path,
    started: float,
    *,
    archive_sha: str = "",
    archive_size_bytes: int = 0,
    inflate_sh_sha256: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size_bytes,
        "inflate_sh_sha256": inflate_sh_sha256,
        "distinguishing_bytes_paths": list(distinguishing_bytes_paths),
        "section_results": [],
        "overall_passed": False,
        "verdict": "INFRASTRUCTURE_ERROR",
        "infrastructure_error_reason": reason,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--inflate-sh", required=True, type=Path)
    parser.add_argument(
        "--distinguishing-bytes-path",
        action="append",
        required=True,
        dest="distinguishing_bytes_paths",
        help="Archive section name (ZIP member). Repeat for multiple sections.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--mutations-per-section",
        type=int,
        default=DEFAULT_MUTATIONS_PER_SECTION,
    )
    parser.add_argument("--file-list", type=Path, default=None)
    parser.add_argument("--inflate-timeout-seconds", type=int, default=600)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    result = verify_distinguishing_feature_byte_mutation(
        archive=args.archive,
        inflate_sh=args.inflate_sh,
        distinguishing_bytes_paths=args.distinguishing_bytes_paths,
        output_json=args.output_json,
        mutations_per_section=args.mutations_per_section,
        file_list=args.file_list,
        inflate_timeout_seconds=args.inflate_timeout_seconds,
        verbose=args.verbose,
    )

    verdict = result.get("verdict", "FAILED")
    if verdict == "PASSED":
        return 0
    if verdict == "INFRASTRUCTURE_ERROR":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
