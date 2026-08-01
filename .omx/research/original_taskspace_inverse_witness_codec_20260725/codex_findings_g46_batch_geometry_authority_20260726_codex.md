# G46 — frozen teacher batch-geometry authority closure

Date: 2026-07-26  
Lane: `lane_original_taskspace_inverse_witness_codec_capstone_20260726`  
Authority: encoder-only target custody; `[macOS-CPU advisory]`; not a score claim  
Repository HEAD at materialization: `0058123af31779d83d1fc10a728389b0ce7823ec`

## Executive verdict

The fresh n600 target coordinate is now complete and compile-gated at the frozen public evaluator's declared default SegNet pair-batch geometry: **16 pairs per forward**. The sealed bank contains all 600 chronological target-label planes, has SHA-256 `6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85`, and is legal only as encoder-side inverse-solve custody. It is forbidden candidate payload.

Batch geometry is not a harmless throughput setting. Against the same frozen source and scorer, batch 4 changes 3 of 117,964,800 label cells relative to batch 16; the historical batch-32 cache changes a different set of 3 cells. A compiler that mixes those banks solves a nearby but different evaluator fiber. The compile-ready loader now fails closed unless the receipt declares batch 16, proves equality to the upstream default, links the sealed stage-0 preflight, and preserves the exact target-bank hash.

This closes target-coordinate ambiguity; it does **not** move the frontier. The pointer delta is **zero**. The score path remains the full-system bridge established by G47: fresh V9/V15-scale task-space grammar -> counted selected-preimage quotient -> generic V10 factor-2 realization -> exact public evaluator. This bank supplies the encode-side boundary condition for that bridge and must never become its representation.

## Exact receipts

| Object | Receipt |
|---|---|
| Full materialization | `/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/12_encoder_only_receipt.json`; file SHA-256 `556c6b20f12ae7f5c6b8a1a2d08c6d2c2e32e7831ab552938c7866f3256dfad1`; sealed receipt SHA-256 `58db7f01674c60f060a46b955fee8c4f777f31f528ebba404e871b26b17972a7` |
| Stage-0 custody/storage preflight | same run root, `00_custody_storage_preflight.json`; file SHA-256 `8f2f7a056e79269000cf7c5bf6013de338cd66fdb8c14dec271606c1877664ce`; sealed preflight SHA-256 `98bc94c2416c606b0efe2ae2285d84efa548b58b5b809488fd17eddcea68ab29` |
| Target bank | same run root, `11_target_labels/target_labels_n600_or_bounded.u8`; 117,964,800 bytes; shape `[600,384,512]`; dtype `uint8`; SHA-256 `6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85` |
| Batch-geometry audit v2 | `.omx/research/original_taskspace_inverse_witness_codec_20260725/g46_teacher_batch_geometry_audit_v2_20260726.json`; 9,504 bytes; file SHA-256 `d1e74b6fdafbc59bf8274851e2c26e402ecc965ec018d08301195de7122597d9`; sealed audit SHA-256 `7a29b8f35fb2177071048a84c75347e473f106a5cd9bc4bc61818c10e21b51aa` |

The run covered 600 pairs / 1,200 source frames in chronological order, used the declared upstream default batch size 16, completed under the governed storage/RSS wrapper, and peaked at 6,378 MiB RSS. Its axis is explicitly `[macOS-CPU advisory]`, `encoder_only=true`, `contest_axis_authority=false`, `score_claim=false`, and `pointer_mutation_allowed=false`.

The earlier `g46_teacher_batch_geometry_audit_20260726.json` remains immutable historical provenance. It found the right cells but could only label the legacy receipt `batchNone_receipt`. Audit v2 is the authority because it reconstructs batch 4 from the sealed stage-0 argv and marks that recovery `LEGACY_RECEIPT_GEOMETRY_UNDECLARED`.

## Measured geometry drift

Primary coordinate: batch 16, the default in frozen `upstream/evaluate.py`.

| Comparison | Pair | Cell `(row,col)` | batch 16 | comparison |
|---|---:|---:|---:|---:|
| legacy batch 4 | 18 | `(286,448)` | 4 | 0 |
| legacy batch 4 | 137 | `(204,441)` | 0 | 2 |
| legacy batch 4 | 381 | `(206,433)` | 2 | 0 |
| historical batch 32 | 11 | `(286,399)` | 4 | 0 |
| historical batch 32 | 18 | `(286,448)` | 4 | 0 |
| historical batch 32 | 381 | `(206,433)` | 2 | 0 |

Each comparison has exact mismatch count 3 and fraction `3 / 117,964,800 = 2.5431315104166668e-08`. The tiny fraction is precisely why this was dangerous: headline metrics can hide a coordinate mismatch that matters to exact inverse solving and hard-oracle repair.

Verdict scope: this establishes host-observed target-label sensitivity to scorer batch geometry and identifies batch 16 as the frozen public evaluator's declared coordinate. It does not claim that every host produces bit-identical scorer outputs, does not confer contest-CPU/CUDA score authority, and does not prove a candidate score.

## Triality closure

### DSL

`taskspace_fresh_teacher_materializer_v1.py` now declares `scorer_pair_batch_size`, `upstream_evaluate_default_pair_batch_size`, `batch_geometry_matches_upstream_default`, the SegNet frame selector, source sequence length, evidence axis, and candidate-payload prohibitions. `load_compile_ready_materialization_receipt(...)` is the strict consumer: old receipts and non-16 geometry are research evidence, never compile authority.

`taskspace_teacher_batch_geometry_audit_v1.py` is the typed disambiguator. It compares banks chunkwise, emits complete changed-cell records and transition histograms, recovers a legacy batch size only from a sealed stage-0 receipt, and seals the resulting audit.

### DAG

`frozen source video -> sealed stage-0 custody -> exact 16-pair SegNet forwards -> chronological n600 uint8 target bank -> sealed materialization receipt -> strict compile-ready loader -> selected-preimage compiler`

The missing/changed-row resume rule is intentionally batch-atomic: if one row in a scorer batch is missing, the materializer re-forwards the entire original batch and selects the missing rows afterward. Forwarding only the missing subset would change the computation and can recreate the rare-cell drift.

### Equations

For source pair batch `B_b` and frozen SegNet `F`, the authoritative target plane is

`L_i^(b) = argmax_c F(B_b)[i,c,:,:]`, with `b = 16`.

The measured non-invariance is

`sum_i ||L_i^(16) != L_i^(4)||_0 = 3` and `sum_i ||L_i^(16) != L_i^(32)||_0 = 3`.

This is an encoder boundary condition, not a separable score target. Candidate arbitration remains

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489`,

measured only after the counted selected-preimage program decodes the full expected video through the public receiver.

## Unified-stack wire-in

1. **Sensitivity map:** the audit records the exact batch-sensitive pairs/cells; these are hard coordinate annotations, not generic high-weight pixels.
2. **Pareto constraint:** no independent component cap is added. The strict gate prevents optimization against the wrong fiber; final admission remains coupled score and bytes.
3. **Bit allocator:** the 117,964,800-byte bank has zero candidate-byte eligibility. It may guide encode-side factor/residual allocation, but every video-derived decoded operand remains counted.
4. **Cathedral/autopilot:** selected-preimage compilation must call the strict receipt loader before consuming target custody; legacy batch-4/batch-32 banks fail closed.
5. **Continual learning:** the sealed v2 audit plus this finding are the durable empirical anchor; rediscovering or averaging the three geometries is forbidden.
6. **Probe-disambiguator:** batch 4, 16, and 32 are shipped as explicit audit inputs, with batch 16 selected by frozen upstream code rather than an arbitrary local threshold.

## Verification

Focused static and behavioral verification passed:

`ruff check` on the materializer, audit tool, and strict-loader tests: clean.

`pytest` on the materializer, geometry audit, and CLI suites: **17 passed**.

The suite covers full-batch recomputation during partial resume, compile-ready batch-geometry enforcement, sealed stage-0 linkage/tamper rejection, exact mismatch accounting, and legacy batch-size recovery.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, current lane registry, subagent progress, recent directives, and project memory.
- Frozen `upstream/evaluate.py` and the materializer's recursive scorer import surface.
- The clean batch-16 run, legacy batch-4 run, historical batch-32 `gt_n600.npz::lstars` cache, and their custody receipts.
- Canonical frontier pointer and G47 low-distortion selected-state path audit.

## Next score-directed action

Consume this bank only through the strict loader while the selected-preimage program measures the *description length of the solution*, not the target table: shared V9/V15 semantic/worldsheet factors first, deterministic V10 realization second, and only the irreducible coupled residual trained or stored. The next meaningful receipt is a full-n600 factor/residual byte profile or a receiver-closed archive row; another target-bank materialization is dominated unless upstream scorer code or host authority changes.
