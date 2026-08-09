# DDM-FX2 shared inflate contract repair

Status: **PASS**. The shared FX1 tree now accepts the evaluator's real three-argument call, emits the required per-video raw path, and is protected by an executed Catalog #146 mutation control. This was scorer-free shipping work; the PR130 base pointer did not move.

## Source-derived contract and violation

Pinned upstream is clean at `11ad728f563d8970929e8947a1cf6124ee6303e4`. `upstream/evaluate.sh:41-47` extracts `archive.zip` and invokes:

`bash inflate.sh <archive_dir> <inflated_output_dir> <video_names_file>`

Lines 49-63 then require `<inflated_output_dir>/<base>.raw` for every non-empty listed name. The pre-repair shared shell, SHA `972f49c...`, instead ran `inflate.py "$@"`. Its Python primitive expects `<archive-dir> <base> <destination.raw>` and rejects `base != "0"`. The evaluator's output directory therefore became `base` and failed before render; the names-file path occupied the destination slot. The defect verdict is INSTANCE-scoped to the pre-FX2 shared runtime.

## Repair and provenance

The shell adapter is ported from our pinned DV1 repair at commit `9a1b483b220eb67c560521cc65ad1efe93cffbab`, `src/tac/pr130_runtime/dv1_cpu_runtime/inflate.sh:237-252`. It validates three arguments, creates the output directory, skips blank names, derives each base, and calls the unchanged one-video Python primitive with `<output>/<base>.raw`.

Only the pinned hunk was used; the live DV1 tree contains unrelated concurrent CX2 changes. The shared `inflate.py` also ports DV1's explicit-or-auto CPU/CUDA selector so the charter's real, no-Modal proof could execute on the available CPU host. CUDA still disables TF32 and retains its CUDA-only cache operation.

Post-edit runtime hashes:

- `inflate.sh`: `6926683f68b7b0017c3c79ad33f5da82e8d8d5f4394c11549c6b8af914538b21`
- `inflate.py`: `75e76ac070c973a5403562eb1db1d9c479d11d9a8c8489760f5da5dfe55806ea`
- `runtime-dependencies.json`: `1b5515b8841bde0f544c806a70391c1f039696fe10cbd5213375e639dae22b2a`

## Executed three-argument replay

The replay used a never-existing SSD output directory, the exact 191,052-byte archive `0491d5df...d7cd`, extracted member `p` `fcc6a3c2...d84`, and the exact upstream six-byte names file `0.mkv\n` (`7ff99d08...29a8`). The real shared wrapper ran with absolute arguments and `PR130_INFLATE_DEVICE=cpu`.

Measured `[macOS-CPU scorer-free behavioral replay]`:

- shell return code: `0`
- decoded tokens: `600/600`
- rendered masters/carriers: `600/600` each
- terminal marker: `wrote .../inflated/0.raw (3662409600 bytes)`
- wall time: `979.61 s`
- raw bytes: `3,662,409,600`
- raw SHA-256: `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`
- independent DDM-DT1 raw SHA match: **yes**

The command, environment, log, and cleanup certificate remain under `/Volumes/VertigoDataTier/pact/ddm_fx2_20260809/replay_20260809_WFKO4z/`. After certification, the rebuildable 3.66 GB raw was deleted; the retained certificate is `output_certification.json`, SHA `87a726ff...920a`. It is not directly recoverable, but the retained inputs and command deterministically rebuild it.

## Guard and positive control

Catalog #146 was extended rather than creating a new gate. It now refuses a missing shared FX1 shell, comment-only contract fragments, missing evaluator-to-per-video fragments, and the legacy `inflate.py "$@"` pass-through. The executed positive control copied the live repaired shell, replaced only its repaired tail with the exact historical pass-through, and required strict Catalog #146 to raise on the shared FX1 path. The static check is deliberately paired with the real replay; static shell text alone is not behavioral evidence.

Final validation: ruff, Python compilation, shell syntax, manifest JSON, live strict Catalog #146, cleanup absence, 49 Catalog/closure tests, and 7 PR103 adapter tests all passed. Each changed Python file also received the required `fx2_pass_1` and `fx2_pass_2` review-tracker passes. Durable validation log: `/Volumes/VertigoDataTier/pact/ddm_fx2_20260809/final_validation.log`, SHA `63ecb549...12c5`.

## Sister bare-Python hazard

FX1 does not have the named hazard: it binds `PYBIN=${PYTHON:-python3}` and uses `"$PYBIN"` for every Python execution. The charter's PR103 statement is stale: commit `0b47f132c8` already installed the fail-closed `PYTHON` override, then `python3`, then `python` fallback and now requires one portable invocation with zero legacy invocations. The other BL1 emitters remain separate work.

## RECALL EVIDENCE

The search covered the current 28,841-file `.omx/research` corpus with content queries for `fx1_runtime_tree|shared FX1|shared-runtime` (18 matching files), `three[- ]argument|3[- ]arg|video.names.file` (418), and `bare[- ]python|python3[- ]only|legacy bare-python` (10). It also covered the one `CANONICAL_RESEARCH_INDEX*` file, all 11 `sub015_DAG_*` files, the design/SPEC query surface (400 files), `.omx/state/canonical_task_status.jsonl`, the harness bridge, current arm receipts/queue, and the canonical-equation registry. No FX1/three-argument/bare-Python-specific equation was found in those named scopes; the relevant generic equation is `modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1`.

Beyond the charter seeds, recall found the already-landed BL1 PR103 fix, the manifest hash consumer, upstream's stale-output masking risk, DV1's syntax-only proof boundary, the historical DV1 verifier's moving-reference hazard, and the requirement to recompute runtime-tree hashes after this edit. Those findings changed the run to use a fresh path plus rc/marker/full-hash proof, update both runtime custody hashes, consolidate into Catalog #146, and avoid the dirty live DV1 tree.

## Authority boundaries

No scorer, `upstream/evaluate.py`, Modal job, Linux run, or CUDA run occurred. No `d_seg`, `d_pose`, or score was measured. Raw byte equality proves receiver output on this macOS CPU axis; it does not promote Linux/CUDA scoring authority. The 979.61 seconds excludes checkout, evaluator loading, and scoring.

## Follow-on dispositions

- **QUEUED-WITH-A-FIRE-ORDER** — owner: next shared-FX1 contest dispatcher; consumer store: that run's `contest_auth_eval.json` runtime manifest plus this receipt; fire trigger: before the first Linux CPU or CUDA evaluation/dispatch of the repaired shared tree; action: recompute local/worker runtime hashes and replay the three-argument contract on that axis.
- **HELD-SEPARATE** — owner: BL1 successor; consumer store: `.omx/research/ddm_bl1_20260805/RECEIPT.md` and `NEXT_IF_RESUMED.md`; fire trigger: the next scorer-free apparatus cleanup boundary explicitly owning the three remaining emitters; action: finish that separate cleanup without attributing it to FX2.

## LIVE-HYPOTHESES

- The wrapper should preserve output on Linux CPU and CUDA because its adapter is POSIX and the one-video primitive is unchanged; target-axis execution remains untested.
- A whole contest job may fit 30 minutes because macOS CPU inflate took 979.61 seconds, leaving 820.39 seconds for omitted stages; Linux and scorer timing remain untested.

## DEAD-ENDS

- Do not dispatch the pre-FX2 pass-through; it maps output directory to Python base and fails before render.
- Do not substitute DV1 syntax checks, file existence, or allocated size for completion; only rc, terminal marker, and full digest closed this replay.
- Do not copy the live DV1 tree wholesale; it contains unrelated CX2 work.
- Do not repair bare Python in FX1 or PR103; FX1 already uses `PYBIN`, and PR103 was fixed by `0b47f132c8`.
- Do not reuse a pre-FX2 runtime-tree digest; the shell, Python entrypoint, and manifest changed.
- Do not feed the moving live FX1 tree to the historical DV1 verifier; materialize the pre-DV1 reference at `5de03569ad804d0a087bb6ae5d1b17bb48baa0c5`.

PR130 base remains **S = 0.172141297491896447 at 191,052 B `[contest-CUDA, DALI GT, n600]`**.
