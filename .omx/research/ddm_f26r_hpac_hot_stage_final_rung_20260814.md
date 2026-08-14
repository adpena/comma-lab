# DDM F26R — HPAC hot-stage final rung (2026-08-14)

## Verdict

**SEALED_FIRE_ORDER** for the canonical contest-CPU exact row. F26R reduced the selected full-n600 M5 token decode from 203.843359 s to 147.005377 s, removing 56.837983 s against the charter's required 16.015113 s. All full-field identity gates, the optimized repeat, and the forced-scalar twin pass. Applying the charter-prescribed measured f26q M5-to-Modal token-stage ratio gives a **DERIVED**, not measured, contest-CPU total of 1321.647333 s, 278.352667 s below the 1600 s fire gate.

The exact pointer did not move. No Modal run and no scorer run were made by this arm, so this is not an exact score row and not goal completion.

Verdict scope: **INSTANCE(F26R direct int16 frame context plus precomputed conv-A class deltas on the inherited F26/MC36 archive and HPAC model, measured on M5 and projected with the measured f26q cross-host ratio).**

## Measured result

| Quantity | Value | Authority |
|---|---:|---|
| Parent f26q token decode | 203.8433591669891 s | `[M5-CPU 4-thread scorer-free]`, measured |
| F26R selected token decode | 147.005376584013 s | `[M5-CPU 4-thread scorer-free native token decode]`, measured |
| Removed | 56.83798258297611 s | measured difference |
| Speedup | 1.386638801271962× | measured ratio |
| Required ceiling | 187.8282463550312 s | charter derivation |
| Required removal | 16.01511281195789 s | charter derivation |
| Optimized repeat token decode | 140.94695816596504 s | `[M5-CPU 4-thread scorer-free native token decode]`, measured; determinism receipt, not selected estimate |
| Forced-scalar twin token decode | 146.75189054198563 s | `[M5-CPU 4-thread scorer-free native token decode]`, measured; parity control, not x86 timing |

The primary selected stage receipt is:

| Stage | F26R seconds | f26q seconds where comparable |
|---|---:|---:|
| Native int16 frame context | 2.4285287857055664 | 30.169883078080602 Python/Torch context |
| Sparse hidden and logits | 114.39019799232483 | 140.57528018951416 |
| Incremental conv update | 16.981491565704346 | 22.297621250152588 |
| Probability plus RC64 | 4.356314420700073 | 4.99371337890625 |
| Checkpoint persistence | 5.465759792365134 | not part of the native fused-stage comparison |
| Digest updates | 1.435388662852347 | not part of the native fused-stage comparison |
| Full measured token decode | 147.005376584013 | 203.8433591669891 |

These are composition measurements. The direct-context change, pixel-major context layout, sparse-hidden path, and precomputed deltas interact, so the per-stage differences are not independent causal attributions.

## Full-n600 identity and portability gates

All three full runs decoded 117,964,800 bytes with SHA-256 `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` and RC64 bit position `921964`.

| Gate | Primary optimized | Optimized repeat | Forced-scalar twin |
|---|---|---|---|
| Exact decoded byte count | PASS | PASS | PASS |
| Exact token SHA-256 | PASS | PASS | PASS |
| Corrected quantized-logit SHA-256 `617e9fcf…` | PASS | PASS | PASS |
| Corrected CDF-input SHA-256 `ba0d529b…` | PASS | PASS | PASS |
| Exact decoder bit position | PASS | PASS | PASS |
| Full digest scope | PASS | PASS | PASS |
| Repeat/twin parity to primary | reference | PASS | PASS |

The optimized binaries rebuilt deterministically at 69,424 bytes with SHA-256 `1cf0e61b53d5b25a2b0cbb6adb47232921ebd442aa461cfcbb8db97d664a6aae`. The forced-scalar binaries rebuilt deterministically at 69,424 bytes with SHA-256 `64efe1e803aa0d22dbb0e3d02df5e7799a2e76b7ae4298311e78ab96cc86f4a8`.

The shipped source has compile-guarded NEON, AVX2, and scalar accumulation paths. NEON and forced-scalar execution were measured on M5. AVX2 was source-audited and syntax/build-covered but was **not executed on an x86 host in this arm**; the charter-allowed portable-twin gate is the full-n600 forced-scalar equality receipt. The exact contest-CPU fire remains the x86 execution proof.

## Projection arithmetic

The projection uses the same f26q derivation required by the charter:

| Term | Value |
|---|---:|
| Measured f26q M5-to-Modal token-stage ratio | 6.818547260549954 |
| F26R selected M5 token stage | 147.005376584013 s |
| Derived Modal token stage | 1002.3631077930361 s |
| Measured fixed non-token time from the failed Modal run | 319.28422536200014 s |
| **Derived Modal total** | **1321.6473331550362 s** |
| Margin below 1600 s fire gate | 278.3526668449638 s |
| Margin below 1800 s contest budget | 478.3526668449638 s |

This is a cross-host projection from one measured ratio. It is not a Modal runtime measurement, and it provides no score authority.

## What changed

The inherited f26q v13 HPAC/RC64 vehicle and archive are unchanged. F26R's narrow original runtime work is:

- direct generation of the int16 frame context in native C from counted archive model arrays and prior decoded tokens;
- persistent context workspaces and a pixel-major prior-token layout for contiguous hidden access;
- archive-derived precomputed int16 conv-A class deltas, replacing class-zero subtraction in every update;
- compile-guarded NEON/AVX2/scalar sparse accumulation with a forced-scalar full-field twin;
- a resumable rung driver with distinct per-run checkpoints and retained primary, repeat, and scalar payloads;
- canonical CPU-wrapper retention of `inflated_outputs_manifest.json` in `comma-auth-eval-cache-artifacts`, with a returned `inflated_outputs_volume_manifest.json` that records its volume path, bytes, and SHA-256.

Borrowed substrate: the MC36 archive, F26 vehicle, HPAC model, RC64 coder, f26q native decoder, fixed output renderer, and score components. No originality claim is made over that vehicle or archive. F26R claims only the runtime lowering and custody work named above.

## Custody and fire order

Work root: `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/`

Selected rung: `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/rungs/direct_context_delta_v1/`

Sealed submission: `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/rungs/direct_context_delta_v1/submission_native_sealed/`

Sealed archive: `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de` at 186,269 bytes. Runtime manifest SHA-256: `39112d31c9fc59e3a7ae4671654650445ece1ff8ad26509fe843c0384070288d`.

Every materialized binary, full token payload, prefix payload, build manifest, checkpoint set, and receipt remains retained. The three full token payloads are each 117,964,800 bytes and byte-identical; none was deleted. Post-run free space is recorded as 1,069,273,088 bytes. This is a post-retention fact; the heavy-run storage preflight passed before materialization.

The machine-readable order is `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/SEALED_FIRE_ORDER.json`. Its disposition is **QUEUED_WITH_A_FIRE_ORDER**: owner `MAIN`; consumer `.omx/state/main_hot_state.md` plus the Modal returned-artifact store; fire trigger is the passing full-n600 F26R identity receipt plus derived total at or below 1600 s, followed by serializer landing and a unique active lane claim. The order uses `experiments/modal_auth_eval_cpu.py`, the exact archive SHA, four CPU threads, required failure sentinels, and volume-backed per-frame output-manifest retention. This arm did not claim a lane or dispatch it.

## Validation

- `experiments/ddm_f26r_python_reference_equivalence_test.py`: PASS, including n4/n32 Python-oracle prefixes, three full-field receipts, deterministic rebuilds, scalar-twin parity, projection, and sealed fire order.
- Focused canonical CPU-wrapper tests: 3 PASS, 52 deselected.
- Full canonical CPU-wrapper test file: 54 PASS; one pre-existing mount-ignore test could not bind its temporary AF_UNIX socket because this managed sandbox returned `PermissionError: [Errno 1] Operation not permitted`. The failing test does not exercise F26R or the new volume-manifest helper.
- Python byte compilation: PASS.
- Native C syntax check: PASS.
- Archive payload manifest JSON validation: PASS.
- Source/binary constant audit: PASS; no oracle-hash embedding found.
- `git diff --check` on the F26R change set: PASS.

## Git landing

Disposition: **GIT_BLOCKED_SHA_HANDOFF**. The required serializer invocation was made with post-edit SHA-256 declarations for every file, base hashes, `[no-triality] [p0-ledger-ok]`, and no co-author trailer. Git refused the first object write with `error: unable to create temporary file: Operation not permitted` and `fatal: updating files failed`. The staged index remained empty. No direct-commit or review override was attempted.

The hash-addressed landing manifest is `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/GIT_BLOCKED_SHA_HANDOFF.json`. This is the charter's Git-blocked memo-SHA branch; the implementation and evidence are complete but uncommitted.

## RECALL EVIDENCE

Sources searched:

- full corpus query across `research,equations,memory,dag,council,tasks,docs` for `F26 HPAC sparse hidden logits integer requantization NEON AVX2 int16 frame context conv-A class deltas token sha 9ba2e52b 20260814`;
- canonical equation registry via `tools/list_canonical_equations.py --json`;
- content search across `.omx/research/`, canonical research indices, `sub015_DAG_*`, specs/docs, and task status for F26/HPAC/RC64 hot-stage and native-context terms;
- direct read of the f26q parent memo, result receipt, retained rung receipts, and native-source audit.

Beyond the charter's seeds, `ddm_rc64p_native_cpu_decode_20260810.md` independently showed that direct entropy decode was only 1.11–3.13 s and more than 99.5% of the wall was outside entropy; it also documented no pre-existing exact sparse native HPAC implementation in its searched scope. The bounded ARC negative audit likewise did not supply a closer receiver implementation. That evidence changed the plan by ruling out another RC64-centered rung and selecting direct native context production plus archive-derived conv-A deltas. No reusable exact int16 HPAC hot-stage implementation was found in the searched corpus.

## Measured boundaries

- Measured: scorer-free M5 token runtimes, stage timings, byte counts, hashes, bit positions, deterministic rebuilds, optimized repeat equality, forced-scalar equality, archive/runtime hashes, and retained-artifact custody.
- Derived: the 1321.647333 s Modal total and its margins.
- Not measured: Modal x86 runtime, AVX2 runtime, inflation total on the contest host, Seg/Pose components, exact score, contest-CUDA parity, and any frontier movement.
- The selected archive's score remains borrowed prior evidence. F26R changed its receiver runtime, not its learned/video-derived bytes.

## NEXT_IF_RESUMED

- **GIT_BLOCKED_SHA_HANDOFF** — owner: `MAIN`; consumer store: Git `HEAD` plus `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/GIT_BLOCKED_SHA_HANDOFF.json`; fire trigger: a workspace with a writable Git object store while every declared post-edit SHA still matches; action: rerun the exact serializer landing with message `F26R: seal HPAC hot-stage final rung [no-triality] [p0-ledger-ok]`, no co-author trailer, and no review override.
- **QUEUED_WITH_A_FIRE_ORDER** — owner: `MAIN`; consumer store: `.omx/state/main_hot_state.md` and the Modal returned-artifact store; fire trigger: this serializer commit is landed, the exact `SEALED_FIRE_ORDER.json` archive/runtime hashes still match, and `lane_ddm_f26r_mc36_contest_cpu_20260814` is uniquely claimed active; action: execute the sealed canonical contest-CPU command once, then harvest runtime, score components, failure sentinel, and the volume-backed inflated-output manifest before adjudicating the pointer.

## LIVE-HYPOTHESES

- The contest-CPU exact run will finish below 1600 s because the measured M5 reduction leaves 278.35 s of projection margin, while the shipped x86 source has an AVX2 path and a proven exact scalar fallback. This remains plausible but untested on Modal.
- AVX2 will be at least competitive with the exact scalar fallback on the shipped x86 host because the hot sparse accumulation is contiguous in the new pixel-major layout. The sign and size are not established until the exact host receipt exists.
- The receiver-only change will preserve the archive's prior Seg/Pose components because all three full-n600 runs reproduce the exact decoded token field and all upstream HPAC digests. Only evaluator execution can promote that inference to an exact row.

## DEAD-ENDS

- **FORMULATION(f26q RC64-only lowering):** closed; entropy coding was only 4.99 s and cannot remove the required hot-stage time.
- **INSTANCE(f26q single-thread native, compiler-flags-only, incremental conv-A alone, persistent OpenMP team):** closed by the parent receipts; none met the gate, so F26R did not retry them.
- **INSTANCE(Python/Torch frame-context production in this receiver):** superseded; direct native int16 production is exact and reduces the measured context stage from 30.17 s to 2.43 s in the adopted composition.
- **FORMULATION(architecture-specific fast path without an exact portable twin):** closed; the forced-scalar full-n600 receipt reproduces all bytes and digests, so no NEON-only candidate is being handed off.

Own-vehicle frontier remains **S = 0.7539807296911207 @ 357,836 B `[macOS-CPU advisory]` n600**; F26R did not move it.
