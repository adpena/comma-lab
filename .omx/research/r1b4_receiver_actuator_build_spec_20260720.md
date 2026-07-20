# R1b4 production receiver and xi0 actuator — build specification

This arm will build a new, versioned production receiver that consumes the existing C2 grammar plus every R1b2 archive section and proves byte-causality, deterministic decode, exact final output custody, and zero receiver-side search. It will also bind the real 600-value xi0 payload to a deterministic frame-0 actuator and measure its pose effect through the hard CPU-Torch oracle, while refusing to invent rank-4/secant or full-kernel producer custody if the live VJP campaign remains nonterminal.

`lane_id=lane_r1b4_receiver_actuator_20260720` · `$0 local` · `[macOS-CPU advisory]` · `research_only=true` · pointer `0.1910828242 [contest-CPU] UNMOVED` · MAIN review required

## Authority and immutable inputs

- Delegated authority: `r1b4_receiver_actuator_20260720T190615Z.wrapped.prompt.txt`, SHA-256 `b4d1f8156eb60441d0d28db2c22289e3489e20c07c16a9fcc0c250f1b7b59132`.
- Required lineage merge `7ec6c7e7d5` is already present as merge commit `f514688f1f`.
- Do not edit the pinned C2 decoder `/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py`, SHA-256 `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224`.
- Do not edit the pinned C2 parser `src/tac/boundary_math/integer_plane_emitter_byte_close.py`, SHA-256 `7b404063e35d1f0352fc651ed91d62912ad513f8949ea0f1fa16e4e9f2c611e1`.
- Real xi0 payload: `/Volumes/VertigoDataTier/pact/evidence/r1b3_producers_20260720T185300Z/xi0.xi0`, exactly `1,500 B`, SHA-256 `1b3c72fbe1df7209533a0e92e368fc65253bcb55cbe7196e921502ceb757e58a`. Its landed strict codec is `decode_xi0_payload` in `r1b3_producer_preflight.py`.
- Settled control: `94,344 B`, `d_seg=0.003515794640406966`, `d_pose=127.36588287353516`, `S=36.10275630841103`, batch 16, seed 1234, hard CPU-Torch through R, advisory only. Do not re-label it as a candidate.
- Live VJP receipt is read-only and must be re-read at the P1 milestone. A nonterminal or refused campaign cannot be silently promoted to complete custody.

## Owned implementation surfaces

Prefer a narrow new module such as `src/tac/boundary_math/r1b4_section_receiver.py`, its focused tests, and a new measurement/build CLI. Reuse existing C2, boundary-packet, factor-2 realization, xi0 codec, R1b2 archive, and hard-oracle helpers. Narrowly modify `r1b2_mdl_xi0_compile.py` or its CLI only when necessary to bind the new receiver contract or to make the prompt's P2-non-gate policy explicit; preserve all prior fail-closed contracts and tests.

Do not touch pinned originals, upstream scorer files, frontier pointers, live run directories, provider dispatch, unrelated trainer/DSL hot files, or other worktrees. Large decoded raw files and scratch must use the SSD storage waterfall, be atomically promoted, and be deleted only after a durable reproducibility receipt certifies archive/receiver/source hashes, command, environment, bytes, decoded hash, and rebuildability.

## L1 — versioned consuming receiver

1. Parse one R1b2 candidate archive exactly: inherited C2 base members followed by canonical `r1b2_manifest.json`, `boundary_coordinate.bgj`, `full_kernel_replay.r1k`, and `xi0.xi0`. Reject duplicates, directories, encryption, reordered/missing/unknown members, noncanonical manifests, length/hash drift, trailing bytes, unsafe names, and declared/actual custody mismatch.
2. Decode the inherited C2 base through the pinned decoder semantics without modifying the pinned file. A wrapper/extractor is acceptable only if it proves the extracted base bytes exactly match the manifest and the complete versioned receiver remains the sole production entrypoint for the candidate.
3. Give every counted R1b2 section an explicit decode-time consumer:
   - the manifest governs and cryptographically binds receiver policy, base custody, section custody, pair count, ordering, zero-search policy, and final output assertions;
   - `boundary_coordinate.bgj` is strictly decoded with the existing localized curvelet/shearlet packet and applied to frame 1 through exact factor-2 uint8 realization;
   - `full_kernel_replay.r1k` is a strict compact replay grammar applied without search. The prompt explicitly says P2 is not an efficacy gate: a zero-selected production replay may be admitted only if its grammar still has an explicit validated consumer and the receipt honestly reports zero selected effects; the mutation fixture must use a nonzero valid replay so byte-causality is testable;
   - `xi0.xi0` is strictly decoded and conditions frame 0 through the L2 actuator.
4. Preserve frame ordering and exact raw geometry. Write to a same-filesystem partial file, fsync, verify exact final byte count and SHA-256, then `os.replace`; refuse overwrite and clean success-only scratch only after receipt fsync.
5. No scorer, scorer weights, ground truth, optimizer, search loop, or video-specific uncounted table is permitted in the receiver. Generic deterministic raster/warp/realization code is allowed. Record `receiver_search_invocations=0` and enforce it structurally.
6. Prove two independent decodes of identical exact archive bytes have equal final SHA-256. Measure decode time and enforce `<=1800 s` on the measured n600 run.

## L1 mutation and regression acceptance

Focused tests must prove:

- strict valid parse and exact full consumption;
- corruption/truncation/trailing/duplicate/reordered/unknown/hash-drift refusal;
- deterministic equal output bytes across two runs;
- each of the four R1b2 members is operationally consumed. For semantic payload sections, construct a separately valid mutation, reseal required hashes, and prove output bytes differ. For the manifest, prove a semantically meaningful valid receiver-policy mutation changes output bytes; bare checksum/hash corruption must fail closed. Never satisfy #417 with arbitrary hash noise or a hidden-data-in-code mapping;
- a nonzero compact replay mutation changes decoded bytes, while an admitted zero-selection replay is reported as zero-effect rather than falsely claimed causal;
- section effects are localized to their declared frame/surface and final raw bytes remain exact.

## L2 — xi0 actuator and hard measurement

1. Decode the banked xi0 bundle exactly and require 600 finite coordinate-zero values with no other pose coordinates.
2. Implement a deterministic scorer-free frame-0 actuator whose output is a genuine function of each pair's xi0 value. It may use a fixed generic geometric warp/raster family and counted receiver metadata, but must not use PoseNet, GT frames, archive-time hidden tables, output hashing, or receiver-time search.
3. Frame 1 must remain byte-identical when only xi0 changes; this is the structural SegNet-free factorization test. At least two distinct xi0 values must induce distinct frame-0 bytes.
4. Do not assert efficacy from target presence. Run the new receiver and the inherited hard CPU-Torch oracle on real decoded bytes. Report `d_pose` and all action components versus the settled carrier-absent control. If a full n600 archive cannot yet be compiled because P1 custody is absent, build a strictly counted receiver-smoke archive from the exact C2 base plus typed test/zero boundary and replay sections, mark it `receiver_smoke_only`, and measure xi0 on that exact archive; do not call it an R1b boundary candidate.
5. A diagnostic choice among a small preregistered actuator sign/scale family may occur offline, but final decode contains no search. Record all arms and choose by a separately decoded hard-oracle row; no proxy-only adoption. If no grounded pose-bearing mapping can be built, land `R1B4_XI0_TARGET_TO_FRAME0_POSE_ACTUATOR_UNDERDETERMINED` with target-conditioned receiver smoke and measured pose row rather than fabricating a collapse.

## L3 — terminal VJP/P1 custody

At the P1 milestone re-read `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json` and hash it. Only if terminal, assemble receiver-coordinate winner/rival Jacobians and distinct typed Frechet/realized-uint8 secant tensors using the exact rank-4 head chart and batch-16 native winners. Re-derive each refused pair from fresh batch-16 native winners; never waive a refusal. If it still refuses, preserve pair-scoped verdict and proceed only if the existing compiler contract explicitly admits the resulting coverage. Do not broaden this arm into a new full-backbone VJP producer when custody remains nonterminal; land the exact terminal blocker and retain all completed L1/L2 evidence.

## L4 — compile and measured row

When the compiler's required real custody is green, compile exact bytes through `tools/compile_r1b2_mdl_xi0.py`, parse them through the new receiver, run full n600 hard CPU-Torch at seed 1234/batch 16, and report archive SHA/bytes, decoded SHA/bytes, parse-back hash, runtime, `d_seg`, `d_pose`, Seg/Pose/rate/action components, per-stratum realization, and deltas versus control. Enforce `d_seg<=3.39e-4`, archive `<=477.8 B/pair`, decode `<=1800 s`, and carrier `<=1852 B` unless a newly measured realization fraction recomputes break-even through the existing canonical equation. P2 zero-selection is not a blocker by itself.

If P1 remains nonterminal, the required terminal result is: completed L1/L2 implementation and receipts, a real receiver-smoke or n600 xi0 row through exact archive bytes, plus the current hash/count/refusal blocker for L3/L4. No partial custody may be converted into a candidate or family verdict.

## Verification, review, and durable outputs

- Run focused pytest for every touched receiver/compiler/measurement surface, Ruff, compileall or py_compile, JSON parse, and `git diff --check`.
- Record two clean `review_tracker` passes for every changed Python file after the final content change; fixes reset the clean-pass counter.
- Commit through `tools/subagent_commit_serializer.py` using `--expected-content-sha256` for each committed path. No co-author trailer.
- Land `.omx/research/r1b4_receiver_actuator_<UTC>.json`, `.md`, and `_DAG_FEED_<UTC>.md` with `verdict_scope`, stores consulted, pointer honesty, DSL/control disposition, sensitivity/Pareto/bit-allocation/autopilot/continual-learning hooks, and no new canonical equation unless a genuinely new measured law is established.
- End with clean git proof, exact commit IDs, and an explicit MAIN landing review request. MAIN must independently inspect section causality, scorer-free receiver closure, xi0 frame isolation, receipt authority, and rerun focused tests before merging.
