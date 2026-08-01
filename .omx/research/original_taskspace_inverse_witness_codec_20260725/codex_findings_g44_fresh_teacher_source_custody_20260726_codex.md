# G44 findings — fresh current-G17 teacher source custody

Date: 2026-07-26
Lane: `lane_g44_fresh_teacher_source_custody_20260726`
Mode: read-only forest audit; no heavy scorer run, candidate build, exact evaluation, dispatch, commit, or pointer mutation by G44
Repository HEAD observed: `0058123af31779d83d1fc10a728389b0ce7823ec`

## Verdict

A lawful, fresh, full-population target bank now exists. G46 materialized the exact original
`upstream/videos/0.mkv` through the frozen local SegNet at the upstream evaluator's default
pair-batch geometry, 16. The compile-ready encoder input is:

`/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/11_target_labels/target_labels_n600_or_bounded.u8`

It is `(600,384,512)` `uint8`, `117,964,800` bytes, SHA-256
`6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85`.
Its sealed materialization receipt says `encoder_only=true`,
`candidate_payload_allowed=false`, `target_labels_serialized_in_candidate=false`, and
`scorer_weights_serialized_in_candidate=false`. It may seed a newly compiled G; it may not
enter `archive.zip`, `inflate.py`, or any target-dependent public-runtime table.

This closes source-to-target custody, not the current-P-to-first-G program. The remaining
blocking edge is an encoder integration edge:

1. `FrozenTargetSliceCustodyV1` still accepts only a historical cache-shaped custody object
   whose member is literally `lstars`; it cannot honestly bind the fresh raw bank/receipt.
2. G45 accepts already-built per-pair G pages, but obtains current ep725 `phi.argmax` labels
   only inside the execution call. The encoder therefore lacks a public page-factory/current-P
   label materialization seam with which to compile those pages first.
3. G45 has no single counted composite archive or standalone public `inflate.sh`/`inflate.py`
   path containing P plus G (and any V15/J2 selected-preimage operands). No changed n600 row
   or public evaluator closure exists.

No completed current batch-16 SegNet head-feature cache was found. It is not needed for the
first label-local G because exact target labels are sufficient.

## Exact source and scorer custody

| Object | Exact path / identity | Role and authority |
|---|---|---|
| Original source video | `/Users/adpena/Projects/pact/upstream/videos/0.mkv`; `37,545,489` B; SHA-256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` | Sole lawful fresh target source located; Git-LFS object size/OID agree. |
| Public sequence list | `/Users/adpena/Projects/pact/upstream/public_test_video_names.txt`; `6` B; SHA-256 `7ff99d08c8351dd8167ec09213b758da5bbb705dedabe361ba881217374029a8` | Binds the one-video public population. |
| Frozen SegNet | `/Users/adpena/Projects/pact/upstream/models/segnet.safetensors`; `38,502,892` B; SHA-256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` | Encoder-only frozen scorer; forbidden candidate/runtime content. |
| Frozen upstream sources | `evaluate.py` `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`; `frame_utils.py` `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90`; `modules.py` `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` | Content-frozen closure SHA-256 `9c588c725d66c6e840c157568fc5414c37f175348921c6353a64c6431a26cd99`; upstream commit `11ad728f563d8970929e8947a1cf6124ee6303e4`. Local upstream diffs are mode-only, not content changes. |
| Evaluator geometry | `upstream/evaluate.py`: batch size `16`, dataset threads `2`, seed `1234`; README repeats batch size `16` | Target labels are batch-geometry sensitive. Local batch-16 bank matches the evaluator's pair batching but remains a macOS-CPU advisory encoder artifact, not contest-axis score authority. |

The frozen target operation is the evaluator's last-frame semantic term:

`AVVideoDataset(B=16) -> uint8 pair -> SegNet.preprocess_input -> SegNet(frame index 1) -> argmax(dim=1)`.

G46 uses the same upstream `AVVideoDataset`, `modules.SegNet`, preprocessing, frame selection,
and 16-pair forward geometry. It sets Torch CPU threads to 4 for this local deterministic
materialization; this is not a claim that a local label receipt is a contest CPU/CUDA score.

## Fresh batch-16 materialization receipt

| Artifact | Custody |
|---|---|
| Stage-00 preflight | `/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/00_custody_storage_preflight.json`; file SHA-256 `8f2f7a056e79269000cf7c5bf6013de338cd66fdb8c14dec271606c1877664ce`; sealed payload SHA-256 `98bc94c2416c606b0efe2ae2285d84efa548b58b5b809488fd17eddcea68ab29` |
| Aggregate labels | `/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/11_target_labels/target_labels_n600_or_bounded.u8`; `117,964,800` B; SHA-256 `6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85` |
| Materialization receipt | `/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/12_encoder_only_receipt.json`; `488,267` B; file SHA-256 `556c6b20f12ae7f5c6b8a1a2d08c6d2c2e32e7831ab552938c7866f3256dfad1`; sealed receipt SHA-256 `58db7f01674c60f060a46b955fee8c4f777f31f528ebba404e871b26b17972a7` |
| Per-pair root | 600 atomic checkpoints/shards; root SHA-256 `cc9e4722cd50bee7aa708302cab3ff209af4b458529935727aaf20bbf4c4c3d7` |
| Run log | `/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/materialize_batch16.log`; SHA-256 `31483190b839e2a8c5b106dd35fb522b0a9befeda8ba8cadcb3aa2a9406e877f`; governed run succeeded in `204.24 s`, peak RSS `6,378 MiB` under an `8,192 MiB` cap |
| Geometry audit | `.omx/research/original_taskspace_inverse_witness_codec_20260725/g46_teacher_batch_geometry_audit_20260726.json`; file SHA-256 `c9501ea32ece894464b22529e8af15e3888da04abdc30e883c9b27bd15e23080`; sealed audit SHA-256 `58277c584ebc82d92e9c5cbb149005c51ea0f6b6de4800e3003d64667e1ddc78` |

Receipt axis: `[macOS-CPU encoder-only frozen-scorer evidence]`;
`batch_geometry_authority=UPSTREAM_DEFAULT_MATCH_MACOS_CPU_ADVISORY`;
`contest_axis_authority=false`; `full_public_population_proven=true`.

The current implementation identities observed after the batch-geometry hardening are:

- `src/tac/witness_control/taskspace_fresh_teacher_materializer_v1.py` SHA-256
  `3ad49bcb52719d219ec565734fbee019d916472cbc20db955228ca23db7bbc9b`;
- `tools/materialize_taskspace_fresh_teacher.py` SHA-256
  `782ff512240f9deab159d78e314e8babc79d7cf60be497aed3d38e1b9b3f63f1`.

These are dirty-worktree artifact identities; HEAD alone does not identify them.

### Reopen/status command

```bash
.venv/bin/python tools/materialize_taskspace_fresh_teacher.py \
  --mode status \
  --output-root /Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726
```

For code consumers, the fail-closed gate is
`load_compile_ready_materialization_receipt(...)`; it rejects non-16 geometry and still labels
the accepted bank macOS-CPU advisory.

### Exact rebuild recipe

Stage 00:

```bash
.venv/bin/python tools/materialize_taskspace_fresh_teacher.py \
  --mode preflight \
  --source-video upstream/videos/0.mkv \
  --upstream-root upstream \
  --output-root /Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726 \
  --pair-count 600 --batch-size 16 --num-threads 4 --seed 1234 \
  --safety-reserve-gib 8
```

The heavy child was launched by G46, not G44, under:

```bash
.venv/bin/python tools/safe_run.py \
  --rss-mb 8192 --skip-admission-gate --projected-gib 8.0 \
  --timeout 1800.0 --label g46_fresh_teacher_batch16_20260726 -- \
  .venv/bin/python tools/materialize_taskspace_fresh_teacher.py \
    --mode run \
    --preflight-receipt /Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/00_custody_storage_preflight.json
```

The materializer forwards a full original batch whenever any row in that batch is missing, then
retains only missing rows. This preserves batch-sensitive argmax behavior across resume. All
pair shards/checkpoints are reopened and rehashed before the aggregate receipt is accepted.

## Batch-geometry quarantine and historical cache

The earlier G46 output at
`/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_20260726` is not a teacher input. Its
preflight used batch size 4, while the old receipt omitted the geometry field. Its aggregate
SHA-256 is `60644426ac84f07e3d408893d77cdc9b9a239bcdee19c1adf8ab7882d1861abf`;
receipt file SHA-256 is `2165f45ca84f0eec64d61323bc22c715320334fdf58953ff1b8cf98d4e199ec4`;
sealed receipt SHA-256 is `6c64a7a1af12734d45b52c82fb14a8fd10a3572a491de51ea50fad4c0fa118fb`.
The batch-16 audit found exactly three differing cells:

| Pair, row, col | batch-16 target | legacy batch-4 target |
|---|---:|---:|
| `18,286,448` | 4 | 0 |
| `137,204,441` | 0 | 2 |
| `381,206,433` | 2 | 0 |

The historical local cache
`/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz` is
`5,078,017,610` B, SHA-256
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
It contains `gt_f0`, `gt_f1`, `lstars`, `margins`, and `gt_poses`; canonical `uint8(lstars)`
SHA-256 is `f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557`.
The batch-16 audit found exactly three historical-label mismatches, at pairs 11, 18, and 381.
It is a lawful encoder-only historical cross-check, but not a fresh-current teacher seed.

Historical reproduction only, **do not rerun as-is**:

```bash
.venv/bin/python -u tools/build_shared_gt_cache_for_mlx_fleet.py --num-pairs 600
```

That old producer (`tools/build_shared_gt_cache_for_mlx_fleet.py`, SHA-256
`7275e9ac0fc445a243798d407c419cc0a99e6342397c23ac1be8e56f7b1490b0`)
stacks a dense roughly 5 GB local NPZ and lacks the current SSD waterfall, per-pair resume,
atomic stage checkpoints, and explicit batch-geometry receipt. G46 supersedes it.

## No current SegNet head-feature cache

Targeted local, VertigoDataTier, and APDataStore searches found no completed current n600
SegNet head-feature cache. The available extractor is
`tools/extract_segnet_head_features_n600.py`, SHA-256
`4dff07c0cbffb5732937ac3a9e5e44a30b6930391e7944f150876acbf4ae6614`, with cache support in
`src/tac/witness_control/segnet_head_feature_cache.py`, SHA-256
`bf8f3ddd5f4c11dcc6c8e35697d90a67d5a8a97eea0bafa4e87e6e179d3b3040`.
It reads the historical GT cache and hard-requires batch/chunk size 1, so it is not an
evaluator-default batch-16 target-authority surface. Do not launch it for the first
label-local G; exact labels already close that need.

The existing `lstars_f0_n600_batch32.npy` on VertigoDataTier is the first-frame label field,
not the evaluator's last-frame target, and is also wrong for this purpose.

## Current ep725 P and current-P label producer

G17's frozen source P remains the original ep725 object; G20's xcodec rewrite is a rate-only
control and must not silently replace it.

| Object | Exact custody |
|---|---|
| P archive wrapper | `/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/archive.zip`; `83,838` B; SHA-256 `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3` |
| P member | sole `0.bin`; `84,536` B; SHA-256 `f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c`; extracted verified copy at `/Volumes/VertigoDataTier/pact/g22_ep725_xcodec_n600_equality_replay_20260726/full_n600/inputs/source.0.bin` |
| Frozen generic runtime | `/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py`; `56,814` B; SHA-256 `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224` |
| Renderer source identity expected by current contracts | `tools/levelset_byte_close_and_eval.py`; SHA-256 `1cecaa3ee9873d1378eaeb66b04cee56b621bccabe3fcecc01f60d3a0e692ddc` |
| Full-n600 base replay | G22 receipt `.omx/research/original_taskspace_inverse_witness_codec_20260725/g22_ep725_xcodec_n600_equality_replay_20260726/full_n600_decode_receipt.json`; file SHA-256 `3a01e81abfd19a78db86e5851f1b0c453ff553c1fe7d5fad830f95bcd5ec3efd`; 600 pairs/1,200 frames; exact raw SHA-256 `8565df10cbff8f86f02233fd20ececd74857a0d3806caf278a385a4d5421dcae` before certified scratch cleanup; axis `[macOS-CPU frozen receiver structural proof]` |

G45 implements the real current-P label capture and changed-state receiver in
`src/tac/witness_dsl/taskspace_ep725_label_local_g_stream.py`, SHA-256
`353a16a75b42a5a79fbcf5b12ed978931f4e5162865cb8f21908bb28a9f7b5d6`:

```python
execute_ep725_label_local_g_stream(
    source_contract: Ep725StreamSourceContractV1,
    g_pages: Sequence[GPageRefV1],
    *,
    pair_count: int,
    run_root: str | Path,
    output_name: str = "ep725_label_local_g.raw",
    storage_safety_reserve_bytes: int = 8 << 30,
) -> Ep725StreamExecutionReceiptV1
```

It wraps the actual runtime `_outputs_from_h0` while calling the unmodified `_render_pair(p)`,
captures the exact frame-1 `phi.argmax`, constructs `TaskspacePredictorStateV2` with
`NoTransportV2`, parses/applies the supplied label-local G page, and calls
`overlay_g_on_predictor_camera_y1`. It is path-backed, resumable, one-page-per-pair resident,
and preserves Y0 plus unowned Y1.

The API ordering is the blocker: `g_pages` must exist before the call that privately captures
P labels. Add either (a) an encoder-only page-factory callback invoked after each exact P capture,
or (b) a separate exact P-label materializer that shares G45's runtime/capture contract. Do not
duplicate renderer physics in an unrelated encoder.

Also harden `Ep725StreamSourceContractV1.predictor_renderer_sha256`: G45 currently validates
only that the caller supplied a syntactically valid digest. It reopens and hashes the archive,
member, and runtime, but does not reopen a renderer-source path to derive that digest.

## Fresh teacher to first semantic-G API chain

The intended encoder-only chain is now exact up to two missing adapters:

1. Reopen G46 with `load_compile_ready_materialization_receipt` and memory-map one target shard.
2. Capture the matching exact current-P `phi.argmax` through G45's runtime hook.
3. Construct fresh teacher evidence and compile a label-local program with
   `compile_bounded_target_g_v2(...)` or the selective acquisition APIs.
4. Parse/apply via `parse_generative_taskspace_correction_v2`,
   `apply_generative_taskspace_correction_v2`, and the predictor-preserving overlay.
5. Persist the selected G page plus its receipt atomically; release all arrays/proposals; repeat.
6. Pack P plus the newly derived counted G pages into one current public candidate and replay all
   1,200 frames before any score claim.

`src/tac/witness_dsl/bounded_target_g_encoder.py` (SHA-256
`2381d3d9beba1e3a8b73430574a1e23521b710733b6d02f4857b15ed55da03d2`)
currently types `compile_bounded_target_g_v2(..., target_custody=FrozenTargetSliceCustodyV1)`.
That custody type hard-requires `member_name == "lstars"` and a cache SHA. Passing the fresh raw
bank under a fabricated `lstars` name would be a NO-FAKE violation. Add a fresh-target custody
variant binding the G46 receipt file/sealed hashes, aggregate hash, pair checkpoint, source and
upstream closure, batch geometry, and exact slice hash; then admit it explicitly at the compiler.

## n600 memory and staging verdict

The existing selective acquisition fitter is **not** an n600 monolith. It hard-caps one evidence
window at four contiguous pairs (`MAX_BOUNDED_PAIRS=4`) and emits every resolved prefix across
three orderings and three grammar families. It retains the proposal collection for selection;
up to 64 prefix requests multiplies into up to 576 proposal rows per bounded window. In addition,
the counted format caps topology events at `0xFFFF`.

Required production shape:

- prefer 600 independent one-pair page jobs, or at most 150 four-pair shards;
- memory-map one G46 target shard and retain one G45 current-P pair at a time;
- compile/select/persist one page plus receipt atomically, then release dense masks and proposals;
- resume from the next missing pair, verify chronological completeness and a page-root hash;
- stream exactly one verified page per pair through G45.

This is both structurally required and OOM-safe. G46 batch-16 measured `6,378 MiB` peak RSS; the
old dense cache builder is approximately 5 GB before compiler intermediates. A 600-pair in-memory
acquisition lift would violate the bounded API and is unnecessary. The exact changed-state G45
n600 peak/runtime remains unmeasured; it must use the governed SSD launcher and a reviewed cap.

## Public/candidate and forest-level constraints

- Target labels, target RGB, logits/features, SegNet weights, scorer products, teacher state,
  historical MS1/MS2R factors, V15/C1 payload, and public-PR payload are encoder evidence only.
- Only newly derived video-specific G/V15/J2 operands may be counted. Generic legal receiver
  machinery may live outside `archive.zip`; target-derived values/branches may not be hidden there.
- G45 currently reads P from a one-member source archive and G pages from external paths. This is
  not a counted same-object candidate proof. A composite packer/parser/public runtime is still owed.
- G45's default output leaf is not public `inflated/0.raw`; standalone packaging must bind the
  exact output name/order/size and double-decode all 1,200 frames within the contest wall.
- The ep725/G path is a real actuator seam, not the already-proven low-distortion selected-preimage
  solution. G47's forest audit identifies the fresh V15/J2 selected-preimage composition as the
  macro score route. Do not import its historical payload; compile new operands from current source
  and encoder evidence.

## Ordered blockers and next exact action

1. **Fresh custody adapter:** replace the false `lstars`-only compiler edge with a typed G46
   batch-16 receipt/slice custody edge.
2. **Current-P page factory:** expose G45's exact captured P labels to an encoder callback or exact
   materializer so a G page can be compiled before execution.
3. **Shard driver:** run one-pair or bounded-four compile/select checkpoints across all 600 pairs;
   no dense n600 acquisition object.
4. **Renderer foreign key:** derive/reopen the renderer identity rather than caller-attesting it.
5. **Counted composite/public closure:** package exact P + fresh pages/selected-preimage operands,
   decode to `inflated/0.raw`, verify 1,200 frames/double decode/runtime, then call
   `upstream/evaluate.py` on contest CPU/CUDA.

The shortest executable next unit is (1)+(2), followed immediately by a resumable n600 one-page
compile. Source-to-target custody is no longer the blocker.

## Verification performed by G44

- Re-read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, G17/G41/G42/G45/G47 artifacts, and lane registry.
- Rehashed the source, scorer closure, G46 artifacts, ep725 P/member/runtime, G22 receipt, and the
  current G45/compiler/acquisition implementations.
- Reopened the G46 receipt/status path and inspected the batch-geometry audit.
- The combined current post-hardening G46 materializer/geometry and G45 stream/overlay suite passed:
  `28 tests`; Ruff passed on the same implementation closure.
- No empirical G45 n600 run was made by G44.
- Targeted cache search found no completed current n600 SegNet head-feature cache.

Pointer honesty: G44 produced no candidate, no archive bytes, no scorer/evaluator row, no dispatch,
and no pointer delta. The exact frontier remains unchanged by this audit.
