# JRD PR110 pointer completion — exact real-archive run

**Date:** 2026-07-13  
**Verdict:** **NULL / NO-GO FOR JRD ON THIS NAMED ARCHIVE**  
**Authority:** `[macOS-CPU advisory]`, `score_claim=false`, `pointer_moved=false`, `promotion_eligible=false`, `upstream_evaluate.py` not run, paid dispatch not used.  
**Verdict scope:** `INSTANCE` — source archive SHA-256 `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e`; the negative does not kill coefficient-prefix methods on a different archive or search formulation.

## Outcome

JRD saved **MEASURED 0 bytes**. The final byte-closed archive is **MEASURED 177,169 bytes** with SHA-256 **MEASURED `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e`**, byte-identical to the sealed source. Receiver FIFO parse-back is **MEASURED PASS** on **MEASURED 2 independent native-runtime inflations**. No coefficient prefix satisfied the built two-stage component-safe controller, so no truncation was forced.

Durable eval-ready/no-op archive:

`experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/archive.zip`

Primary receipts:

- `experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/measurement_receipt.json`
- `experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/section_precision_response_curves.json`
- `experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/runtime_inflate_proof.json`
- `experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/resume/state.json`

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; SPEC v7.5 operating contract; latest sister findings/session/design/council memos; `reports/latest.md`; `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; `.omx/state/canonical_task_status.jsonl`; `.omx/state/canonical_frontier_pointer.json`; the named source archive and PR110 submission runtime; JRD source, tests, response curves, measurement receipt, and FIFO proof; the source archive's historical contest-CPU receipt at `experiments/results/pr110_payload_entropy_recode_20260610/recoded_cpu_eval/contest_auth_eval.json`.

## Custody correction

The requested archive path and SHA are internally consistent: **MEASURED 177,169 bytes**, SHA **MEASURED `b46897267ded...`**. The supplied score association is not current-byte-consistent:

- This exact `b46897267ded...` archive has a pre-existing historical exact `[contest-CPU]` row of **MEASURED-HISTORICAL 0.19109982419209975**.
- The current submittable `[contest-CPU]` pointer **MEASURED 0.1910828242** in `reports/latest.md` belongs to the later **MEASURED `ad02b0124cbb...`** click-polished archive, also **MEASURED 177,169 bytes**.

The operator-requested comparison to **MEASURED-BENCHMARK 0.19108282** is still reported below, but it is a cross-SHA and cross-axis diagnostic, not a same-archive JRD delta.

## Sealed n600 baseline

| Quantity | Value | Status |
|---|---:|---|
| Evaluation cohort | 600 pairs | **MEASURED** |
| Archive bytes | 177,169 | **MEASURED** |
| Archive SHA-256 | `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e` | **MEASURED** |
| `d_seg` | 0.0005598873564546617 | **MEASURED exact-R local CPU** |
| `d_pose` | 0.00002941618268067714 | **MEASURED exact-R local CPU** |
| Advisory `S` | 0.19110944702697785 | **DERIVED** from the measured components and bytes |
| Raw receiver bytes | 3,662,409,600 | **MEASURED** |
| Raw receiver SHA-256 | `dacf6b33faac74e3a8f9f730c06f04ddee3dfa7df1971944128fbcba99d865bb` | **MEASURED** |

The full positive repeat was **MEASURED bit-identical**: equal archive SHA, raw SHA, raw byte count, `d_seg`, and `d_pose`; the measured component noise floor is **MEASURED 0.0**. The all-decoder-zero negative control separated both components: `d_seg` **MEASURED 0.5048244815568129** and `d_pose` **MEASURED 160.75346749623617**, proving that the scorer path was responsive rather than cached or inert.

## Search and admission law

The adapter verified **MEASURED 28 signed-int8 decoder tensors** containing **MEASURED 228,958 coefficients**. It exhaustively measured **MEASURED 2 nested clean-room prefix families** (`uniform`, `laplace_dead_zone`) at **MEASURED 8 nonzero depths** per tensor, for **DERIVED 448 receiver/scorer rows = 28 × 2 × 8**. Exhaustion is stronger than assuming monotonicity for the requested eight-depth binary search.

Every row reparsed the candidate with the submission's own packet/runtime path before rendering. The per-tensor controller used its pre-registered pair-zero receiver screen; the only packet eligible for handoff then had to pass the full **MEASURED n600** comparison against the sealed baseline. All **MEASURED 56 tensor-family summaries** returned `last_safe=null` and `best_byte_safe=null`. Therefore **MEASURED 0 proposals** reached combined composition, and the final **MEASURED n600** packet was the byte-identical baseline. This is the built tool's fail-closed NULL path; no n600 safety claim is made for an individually rejected pair-zero-screen proposal.

### Per-tensor admissions

| Tensor | Uniform depths tested | Dead-zone depths tested | Admitted depth | Final bytes saved | Combined-check verdict |
|---|---:|---:|---:|---:|---|
| `skips.2.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `refine.1.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.2.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.2.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `skips.4.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.4.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `rgb_0.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.1.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `refine.0.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.3.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.5.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `skips.2.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.1.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.4.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `skips.4.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `stem.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `refine.0.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.0.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `rgb_1.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.5.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.0.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `rgb_1.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `rgb_0.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `skips.3.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `skips.3.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `refine.1.bias` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `blocks.3.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |
| `stem.weight` | **MEASURED 1–8** | **MEASURED 1–8** | none | **DERIVED 0** | NOT-ADMITTED; no combined step |

## Byte-close and receiver proof

- Source SHA → final SHA: **MEASURED `b46897267ded...` → `b46897267ded...`**.
- Source bytes → final bytes: **MEASURED 177,169 → 177,169**.
- Archive bytes saved: **MEASURED 0**.
- Final no-op decoder re-encode: **MEASURED byte-identical PASS**.
- FIFO native-runtime pass count: **MEASURED 2**.
- Return codes: **MEASURED 0 and 0**.
- Raw bytes per pass: **MEASURED 3,662,409,600 and 3,662,409,600**.
- Raw SHA per pass: **MEASURED `dacf6b33faac...` and `dacf6b33faac...`**.
- FIFO scratch success cleanup: **MEASURED PASS**.
- Runtime tree SHA-256: **MEASURED `b0ed91718503008b9ae2b53ae2f32d908c93da54a5374135adb489827ff0d695`**.

The runtime tree did not change, so no copied inflate tree is required beside the final archive. Its bound runtime remains `experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/`.

## Advisory score row

| Archive | Axis | Bytes | `d_seg` | `d_pose` | Advisory `S` | `score_claim` | `pointer_moved` |
|---|---|---:|---:|---:|---:|---|---|
| JRD final / NULL | `[macOS-CPU advisory]` | **MEASURED 177,169** | **MEASURED 0.0005598873564546617** | **MEASURED 0.00002941618268067714** | **DERIVED 0.19110944702697785** | `false` | `false` |

- Same-axis JRD advisory delta versus the sealed local baseline: **DERIVED 0.0**.
- Operator-requested advisory delta versus benchmark **MEASURED-BENCHMARK 0.19108282**: **DERIVED +0.000026627026977865675**. This is local/contest and cross-SHA drift, not JRD harm; the final bytes equal the source.
- Pure rate-term delta `25*delta_bytes/37,545,489`: **DERIVED 0.0** from **MEASURED delta_bytes 0**.

## Provenance and triality

- Run fingerprint: **MEASURED `6429b51f0429a5c43007a75bed9a71b00339873a7f4c20eacd027252f54bc954`**.
- Recorded source-state head: **MEASURED `2380c753f4d4be0ccf428067613e799c27ad1c4a`**.
- Adapter SHA-256: **MEASURED `20dcef88208c7bce20dba39022613b0f4946a85426c3f810b8f2e03832297d9d`**.
- Probe-tool SHA-256: **MEASURED `22ed351f8908a81271114c3d292f8c53bb7c124de826d5634cd8ae4510e99d73`**.
- Scorer/target dependency-tree SHA-256: **MEASURED `a3352e0833fc902fe7c7ae29c445de8f1bf33afa387423dea67dab64e5fedf17`**.
- FIFO proof attempt: **MEASURED `1783912188244722000_30572`**.
- Tests: **MEASURED 32 focused pytest cases passed**; ruff checks **MEASURED PASS**.
- DSL leg: **N/A with reason** — this is a post-training packet transformation, not a trainer/config lever.
- Equation leg: existing `jrd_exact_coefficient_prefix_selection_v1` control law consumed unchanged.
- DAG leg: task `task453_pointer_jrd_pr110_phase1_20260712` → exact named archive → exhaustive prefix screen → full n600 seal → FIFO receiver proof → NULL artifact.

## Final disposition

**NULL closes the built JRD last-safe coefficient-prefix formulation on the named `b46897267ded...` archive without signal loss.** No candidate archive is queued for paid exact evaluation because it would be byte-identical to an already evaluated source. `score_claim=false`; `pointer_moved=false`.
