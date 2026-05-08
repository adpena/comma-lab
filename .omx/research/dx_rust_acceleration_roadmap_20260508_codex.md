# DX / Rust Acceleration Roadmap - 2026-05-08

Owner: Worker C / OSS-DX-Rust acceleration

Scope: preflight and local-test developer experience only. No score-lowering
code, archive candidates, dispatch claims, or contest evidence are changed by
this ledger.

## Current Hot-Spot Profile

Existing timing evidence in `src/tac/preflight_fs_cache.py` identifies the
dominant preflight cost: repeated file walks and source reads across roughly 50
checks. The recorded before/after for the source-tree filesystem cache is:

- before cache: `preflight_all` about 425 seconds on a roughly 12,500-file tree
- after cache: `preflight_all` about 30-60 seconds
- cause: repeated `Path.rglob("*.py")` and repeated `Path.read_text()` /
  `Path.read_bytes()` over the same source files

Code scan on 2026-05-08 supports that diagnosis:

- `src/tac/preflight.py` contains 112 `.rglob(` call sites.
- `src/tac/preflight.py` contains 330 `.read_text(` call sites.
- authored Python files excluding `.venv`, `workspace`, and large result trees:
  2,517 files. Including public-intake/recovery trees explains the larger
  existing 12k-file timing note.
- `preflight.py` comments already measure some proactive gates as cheap:
  Python compile around 0.75s, shell syntax around 0.45s, pytest collect around
  1.3s at the time those gates were added. Those are not first Rust targets.

Immediate implementation in this tranche:

- `preflight_all()` now applies the existing measured filesystem cache for
  programmatic codebase scans, not only for `python -m tac.preflight`.
- `tools/all_lanes_preflight.py --timings-json PATH` now writes deterministic
  timing JSON with wall time, serial-equivalent time, per-step rows, and
  slowest-step ordering. This makes future slow-gate selection trendable.

## Rust Build Surface Already Present

The repo already has a Rust workspace in `runtime-rs/` with these useful
patterns:

- `runtime-rs/crates/zipwire`: ZIP central/local header inspection,
  single-member identity rewrite proof, strict blockers, golden-vector tests.
- `runtime-rs/crates/python-ast-indexer`: RustPython-parser based top-level
  Python source indexer.
- `runtime-rs/crates/qma-codec`, `stbm1br-codec`, `residual-codec`,
  `raw-writer`, and `inflate-cli`: small crate layout with CLI/library split.

Do not introduce a second Rust build system. Future acceleration should extend
this workspace and keep Python fallback paths.

## Rust Tranche Order

1. ZIP header scan and archive wire inspection

   Candidate gates/kernels:

   - `scripts/pre_submission_compliance_check.py --contest-final --strict`
   - `tac.submission_archive.validate_archive`
   - `tools/check_inflate_wire_format_no_dead_bytes.py`
   - PR91/HPM1 readiness ZIP member custody checks inside all-lanes preflight

   Why first: `zipwire` already exists and has golden vectors. ZIP inspection is
   deterministic byte parsing with a narrow contract: local header, central
   directory, EOCD, member names, methods, flags, CRCs, duplicate names, hidden
   files, zip-slip names, and local/central parity.

   Expected speedup: 5-20x for ZIP metadata scans on multi-archive batches;
   smaller on single tiny archives. Bigger DX gain comes from shared JSON output
   replacing several independent Python `zipfile` walks.

   Required parity vectors:

   - stored single-member, deflated single-member, multi-member
   - duplicate names, local/central name mismatch, encrypted flag
   - data descriptor flag, extra fields, comments, truncated EOCD
   - zip-slip paths, absolute paths, `__MACOSX`, `.DS_Store`, resource forks
   - CRC/size mismatch and unsupported compression methods

   Fallback: Python validators remain authoritative until Rust JSON matches all
   vectors and a live archive corpus. If the Rust binary is absent, Python runs.
   If Rust and Python disagree in strict mode, fail closed and print both
   blocker sets.

2. Source inventory and preflight scan index

   Candidate gates/kernels:

   - repeated source file discovery used by `src/tac/preflight.py`
   - `check_dispatch_cli_shell_hazards`
   - reverse-engineering tree curation and public-release hygiene scans
   - shell/runtime-reference inventory checks

   Why second: current measured hot path is repeated traversal and reads. The
   Python filesystem cache removes most repeated I/O inside one process; Rust
   should be an indexed source-inventory CLI only after timing JSON shows which
   gates remain slow.

   Expected speedup: 2-8x for scanner-heavy preflight after the Python cache;
   up to 10x+ versus uncached repeated Python walks. Main win is one traversal,
   stable exclude policy, and reusable per-file hashes/text snippets.

   Required parity vectors:

   - fixture tree with `src/`, `tools/`, `scripts/`, `experiments/`,
     `runtime-rs/`, `reverse_engineering/`
   - ignored result/provider/cache directories
   - symlinks, broken symlinks, deleted paths, permission errors
   - mutated temp files proving source-cache and temp-fixture boundaries
   - Python scanner outputs before/after index substitution

   Fallback: Python `Path.rglob` + cache remains default. Rust source index is
   opt-in until at least three preflight scanners consume one shared manifest.

3. Hashing and manifest custody

   Candidate gates/kernels:

   - `tac.repo_io.sha256_file`
   - runtime tree custody manifests
   - archive manifest freshness checks
   - all-lanes artifact hash self-consistency checks

   Why third: Python `hashlib` already uses native code, so raw hashing speedup
   is likely modest. Rust helps when many files need parallel hashes plus one
   canonical manifest emission.

   Expected speedup: 1.2-3x for large file batches; near zero for single files.
   More important benefit is deterministic, shared custody JSON.

   Required parity vectors:

   - empty file, one-byte file, binary payload, large payload
   - stable path normalization and repo-relative output
   - mode bits where manifests record executable status
   - canonical JSON hash excluding embedded manifest fields

   Fallback: Python hashing remains the oracle. Rust output must include input
   bytes, file count, path normalization mode, and per-file SHA-256 so mismatch
   diagnosis is local.

4. Manifest validation

   Candidate gates/kernels:

   - pre-submission compliance JSON validation
   - runtime custody manifest validation
   - release/public hygiene manifest checks
   - all-lanes timing/profile JSON validation

   Why fourth: schemas churn quickly in research code. Move only stable
   manifests with golden fixtures and compatibility tests.

   Expected speedup: 2-4x only for large batches. Primary gain is stronger
   typed contracts for OSS consumers and native tooling.

   Required parity vectors:

   - valid minimal manifest, valid full manifest
   - missing required field, wrong type, stale SHA, stale byte count
   - extra tolerated fields versus forbidden fields
   - canonical JSON roundtrip preserving hash semantics

   Fallback: Python dataclass/dict validators remain source of truth until Rust
   schemas are generated or mirrored from one checked-in schema record.

5. Byte and entropy scans

   Candidate gates/kernels:

   - archive byte anatomy profilers
   - entropy/rate decomposition tools
   - byte histogram and zero-order entropy scans for HNeRV/PR101/PR106 payloads
   - arithmetic-container profile loops

   Why fifth: high potential, but more score-adjacent. Move after ZIP/source
   custody is stable so acceleration does not change scientific conclusions.

   Expected speedup: 5-30x for repeated byte histograms and entropy tables,
   especially when multiple payload sections are scanned in one process.

   Required parity vectors:

   - all-zero bytes, uniform 0..255 bytes, repeated-symbol bytes
   - known Shannon entropy and histogram counts
   - section-offset manifest with exact byte ranges and SHA-256s
   - malformed/truncated section records

   Fallback: Python profiler remains oracle for paper/contest claims. Rust
   entropy output is empirical/DX evidence until byte-for-byte parity and
   independent formula recomputation pass.

## Operating Rules

- Missing Rust accelerator: fall back to Python and preserve pass/fail
  semantics.
- Accelerator mismatch: fail closed in strict preflight and print both outputs.
- Every promoted Rust kernel needs Python oracle parity tests and Rust golden
  vectors.
- No Rust path may become score evidence without exact archive custody,
  runtime closure, and the normal CUDA auth-eval process.
- New Rust crates must live under `runtime-rs/` and reuse the existing
  workspace, CLI/test pattern, and no-secret public-release hygiene.
