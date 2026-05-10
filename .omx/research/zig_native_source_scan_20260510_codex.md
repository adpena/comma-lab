# Zig native source-scan leaf (2026-05-10)

## Purpose

The operator called out Zig's fast startup and small binaries while pushing for
preflight wall-clock reduction without signal loss. This landing adds a narrow
native leaf for exact-substring source scans:

- `native/zig/source_needle_scan.zig`
- `tools/zig_source_needle_scan.py`
- opt-in profiler surface: `tools/profile_preflight_latency.py --surface zig-source-scan`

This is not a replacement for `tac.preflight` semantics. It is a measured native
accelerator candidate whose output is checked against a Python oracle before any
semantic preflight guard may rely on it.

## Evidence

Zig toolchain:

```text
zig version
0.16.0
```

Focused tests:

```text
.venv/bin/python -m pytest tests/test_zig_source_needle_scan.py src/tac/tests/test_profile_preflight_latency.py -q
14 passed in 8.13s
```

Profiler, cold build:

```text
.venv/bin/python tools/profile_preflight_latency.py --surface zig-source-scan \
  --json-out .omx/research/artifacts/zig_source_scan_profile_20260510_codex.json --top 5

4.452s total; build native scanner 3.839s; scan source substrings 0.613s
```

Profiler, warm binary:

```text
.venv/bin/python tools/profile_preflight_latency.py --surface zig-source-scan \
  --json-out .omx/research/artifacts/zig_source_scan_profile_warm_20260510_codex.json --top 5

0.404s total; scan source substrings 0.404s over 2,555 files; 1,940 matches
```

Native binary:

```text
.omx/cache/zig/source_needle_scan
422K
```

## Guardrails

- Output is deterministic JSON sorted by repo-relative path.
- The wrapper builds into ignored `.omx/cache/zig/`, never into tracked source.
- The profiler surface is opt-in and observational.
- The tests compare native results to a Python oracle for `any` and
  `require_all` matching.
- The scanner intentionally performs exact substring tests only; AST, parser,
  manifest, and custody semantics remain in Python until parity vectors justify
  a native port.

## Next migration

1. Add a stable conformance-vector directory for source-scan fixtures shared by
   Python, Zig, and any future Rust/Rayon implementation.
2. Replace repeated preflight discovery reads only where the Python oracle and
   Zig output agree byte-for-byte on the same tracked file set.
3. Add native multi-needle batch APIs for the slowest SourceIndex-backed guards
   instead of shelling out to one scanner per check.
4. Consider Zig thread-pool or Rust/Rayon file partitioning only after the
   single-process native leaf is a proven correctness-preserving speedup.
