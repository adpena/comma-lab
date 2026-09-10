# ddm_gdc3 geometry-law probe implementation spec (2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]`

## Scope

Implement `experiments/ddm_gdc3_geometry_law_probe.py` plus focused unit tests. The tool is a scorer-free, CPU-only, full-n600 measurement over exactly two read-only predecessor renders. It may write only beneath `/Volumes/VertigoDataTier/pact/ddm_gdc3_next_construction_against_the_geometry_law/geometry_probe_v1/` (apart from normal test scratch). It must not mutate either input tree.

## Pinned inputs

- target/move-44 inherited token field: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8`, 117,964,800 B, SHA-256 `78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6`
- GDC1 K=8 teacher: `/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/k08/receiver_render.u8`, 117,964,800 B, SHA-256 `abd130921beec23255112e8b56148d55c49cf4fede03e99e0e99b55378b16ea2`
- GDC2 best retained render: `/Volumes/VertigoDataTier/pact/ddm_gdc2_categorical_coolchic_k8_distill/governed_v1/exports/stageC_lam1_0.0003_final/render.u8`, 117,964,800 B, SHA-256 `62a25bc44658b3872df7e4913900e70441d4bc4a06251b00e50fa7db160890d7`
- shape `(600, 384, 512)`, dtype uint8, classes 0..4 (`Road`, `Lane`, `Undrivable`, `Movable`, `MyCar`).

Fail closed before measurement if any path, size, digest, dtype-domain, or free-space precondition fails.

## Geometry definition

For each predecessor render independently, let `E = target != render`.

1. Find maximal horizontal runs of `E` within each `(pair, y)` row. Every mismatch receives one exclusive run bucket: `isolated` for run length 1, `short` for lengths 2 through 8, and `long` for lengths at least 9.
2. Build the two-sided target boundary: each endpoint of any horizontal or vertical 4-neighbour target-label transition is a boundary pixel. Euclidean distance is measured to that set. Every mismatch receives one exclusive proximity bucket: `boundary_adjacent` when distance is at most 1.0 pixel, otherwise `interior`.
3. The six exclusive cells are the Cartesian product of run bucket and proximity bucket. Class is the target class. Report class 1 explicitly as Lane.

Unit-test all threshold edges, row/pair isolation, two-sided boundary marking, and exact partition coverage.

## Physical residual measurement

Reuse the exact HG1/GDC2 residual wire format and coder implementations from `experiments/ddm_hg1_heterogeneous_analytic_generator_gate.py`; do not invent an entropy proxy.

For each predecessor render, measure:

- the full mismatch set;
- each of the six geometry marginals;
- each of the five target-class marginals;
- every non-empty geometry-by-target-class cell.

For every measured subset, serialize both required orders, `tile64_time` and `frame_raster`. Race the physical `brotli_q11`, `zlib_9`, and `lzma2_extreme` coders. Persist the raw residual, every coded payload, and every deterministic repeat. Verify coded repeat identity and exact decompression. For the full mismatch set, apply each raw order to a retained copy of the predecessor render and prove the result byte-identical to the target; retain the exact decoded field and its digest. For subset rows, parse back the raw record stream and prove that its addresses exactly equal the selected set and its labels equal the target.

The main table must include source, scope (`full`, geometry marginal, class marginal, or cell), geometry, target class, mismatches, mismatch share, selected-order raw bytes, each coder's bytes, winning coder, winning coded bytes, and winning B/mismatch. Standalone subset sizes are non-additive because each stream pays its own framing and compression context; label this explicitly. Also report counts of runs and run-length quantiles for every geometry/class cell.

## Custody and reproducibility

- Seed metadata: `20260910`; the computation itself must be deterministic and contain no random selection.
- Preflight enough free space for retained raw/coded/repeat payloads and two 117,964,800-byte exact-closure fields; fail closed rather than delete anything.
- Use atomic same-directory writes. Resume only after rehashing every existing retained payload and revalidating parse-back; otherwise fail closed. Never overwrite a predecessor artifact.
- Write `RESULT.json`, `MANIFEST.json`, and `LATEST.json` atomically. The manifest must list byte count and SHA-256 for every retained payload other than the manifest itself. Record source path/bytes/hash, source-code path/hash, axis `[macOS-CPU advisory / scorer-free exact byte measurement]`, `selection_mode=full n600, no subset`, and `score_claim=false`.
- Print a concise JSON completion record. No scorer, Modal, archive build, or score claim.

## Verification

Run `ruff format --check`, `ruff check`, `py_compile`, focused pytest, and `tac.preflight.check_no_measure_and_discard_payload` on the Python source. The main agent will perform two separate source-review passes before serialization.

<!-- # FORMALIZATION_PENDING: research memo; the geometry law lands in the equations leg with the first passing construction -->
