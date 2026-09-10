# ddm_gdc3 no-training key-row ribbon falsifier implementation spec (2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]`

## Candidate and scope

Implement `experiments/ddm_gdc3_anisotropic_keyrow_falsifier.py` plus focused tests. This is the $0, scorer-free, full-n600 falsifier for the **fixed anisotropic key-row ribbon program**. The program stores exact target rows at a deterministic sparse vertical schedule and reconstructs every omitted row by zero-order hold from the preceding stored row. Its expected errors are horizontal boundary runs, the cheapest measured GDC3 geometry. It is not GDC1's per-row cap on horizontal segments and not GC1's class-blind dyadic block paint.

The target is `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8`, exactly 117,964,800 B and SHA-256 `78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6`. The geometry receipt is `/Volumes/VertigoDataTier/pact/ddm_gdc3_next_construction_against_the_geometry_law/geometry_probe_v1/RESULT.json`, SHA-256 `9513573ec8cb2eadb072d966e4217e80e3b5aa4970bf6dfc335339c56cab9476`. Fail closed on identity, domain, space, or parse-back disagreement.

## Fixed optimal-form roster

Measure exactly these four deterministic schedules; the schedule definition is generic/free decoder code and the selected variant ID is counted in the packet:

1. `uniform_2`: anchors `range(0,384,2)`.
2. `uniform_4`: anchors `range(0,384,4)`.
3. `bands_8_4_2`: stride 8 on rows 0..127, stride 4 on 128..255, stride 2 on 256..383.
4. `bands_16_8_4_2`: stride 16 on rows 0..95, stride 8 on 96..191, stride 4 on 192..287, stride 2 on 288..383.

Every band start must be an anchor. The decoder repeats the most recent anchor downwards; row 0 is mandatory. Unit-test exact schedules, full row coverage, and no cross-pair state.

For every variant, retain:

- raw program bytes containing only a versioned header followed by the target-derived anchor rows;
- all three real coder outputs (`brotli_q11`, `zlib_9`, `lzma2_extreme`) and deterministic repeats;
- a counted, receiver-parseable winning packet and identical packet repeat, including variant ID, coder ID, raw/coded sizes, and raw/coded SHA-256;
- the full 600x384x512 receiver render, with SHA-256;
- exact mismatch count, per-target-class counts, and the six exclusive GDC3 geometry counts (reuse the reviewed GDC3 geometry classifier).

The program packet is source-derived and COUNTED. The fixed schedule algorithm, parser, zero-order-hold decoder, and coder implementation are generic and FREE. Nothing target-derived may be embedded in free code.

## Selection and exact residual

Before any real candidate residual is coded, project each variant's residual by summing its own six geometry counts times the GDC2 geometry-marginal winning B/mismatch from the pinned GDC3 geometry table. Label this `DERIVED`, non-additive, and selection-only. Select the minimum `counted_packet_bytes + projected_residual_bytes`; ties use roster order.

Only the selected variant proceeds to the exact falsifier. Serialize its full residual in both `tile64_time` and `frame_raster` using the existing HG1 wire format, race all three physical coders with retained repeats, and select the smallest real payload. Apply the raw residual to the receiver render, retain the exact-closure field, and prove it is byte-identical to the target. Report `packet + R_exact` against the 94,010 B door with integer excess/margin and ratio. This exact result, not the projection, is the verdict.

## Custody and execution

- Output only beneath `/Volumes/VertigoDataTier/pact/ddm_gdc3_next_construction_against_the_geometry_law/keyrow_falsifier_v1/`; predecessor trees stay read-only.
- Seed metadata `20260910`; computation deterministic and RNG-free.
- Preflight at least 1.5 GB free. Atomic writes; distinct retained stage outputs; no deletion. Resume by verifying existing bytes/digests/parse-back and fail closed on divergence.
- `RESULT.json`, `MANIFEST.json`, `LATEST.json`; manifest every retained file except itself. Record implementation path/hash, git commit, axis `[macOS-CPU advisory / scorer-free exact byte measurement]`, `selection_mode=full n600, no subset`, and `score_claim=false`.
- No scorer, Modal, candidate archive, or training.

Run `ruff format --check`, `ruff check`, `py_compile`, focused pytest, and the strict payload-retention census. The main agent performs two explicit `review_tracker.py` passes before serialization.

<!-- # FORMALIZATION_PENDING: research memo; the geometry law lands in the equations leg with the first passing construction -->
