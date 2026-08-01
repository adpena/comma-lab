# G32 findings — R10 n600 maximum-inverse fitter/compiler

Date: 2026-07-26  
Lane: `lane_g32_r10_n600_inverse_fitter_20260726`  
Scope: local build plus bounded real-input verification; `research_only=true`  
Pointer delta: **false**

## Outcome

G32 now has a real source-only encoder fitter/compiler for every counted
physical operand consumed by frozen G27 R10:

1. `PAIR_INDEX`;
2. `GEOMETRY`;
3. `BASE_FEATURE`;
4. `TEXTURE`;
5. `SHOOTING_KNOT`;
6. `XIP2`;
7. `DASH1`;
8. `PULLBACK_POLYGON`; and
9. `STRATIFIED_FLOW`.

This is not an R10 fixture generator. The encoder fits exact G27 operands from
the G20/G22 selected-base bytes and encoder-only source RGB by finite geometry
and twist inversion, least-squares/rank-one fixed-point feature inversion,
finite texture phase/frequency inversion, projected shooting correction,
source-residual DASH support coding, and finite counted polygon/affine-flow
inversion. It then executes one real post-interaction block-coordinate pass in
the order:

`GEOMETRY+XIP2 -> BASE_FEATURE -> TEXTURE+DASH1 -> SHOOTING_KNOT -> PULLBACK_POLYGON+STRATIFIED_FLOW`.

Both the initial sequential corner and the post-interaction corner are replayed
through the exact full-resolution integer source objective. The emitted packet
is selected by minimum full-resolution objective and canonical operand bytes;
the newer corner is not forced merely because it is newer. No training or
learned residual is admitted.

## Landed files

| file | bytes | SHA-256 |
|---|---:|---|
| `src/tac/witness_dsl/taskspace_r10_n600_maximum_inverse_fitter.py` | 84,554 | `7c30f49d27c63e54f7e13c5d8cc9872208285b846d532abd1bdc6bbfd2aae81f` |
| `tools/fit_taskspace_r10_n600_maximum_inverse.py` | 40,217 | `6b7a046221c85be160ef0da2a36efc489ab1441b1bc2703a351032743a972f07` |
| `src/tac/witness_dsl/tests/test_taskspace_r10_n600_maximum_inverse_fitter.py` | 10,023 | `1a50c123f4fa5e721da114d620e19688cfb52c62593432cbf28403aabef40438` |
| `tools/tests/test_fit_taskspace_r10_n600_maximum_inverse.py` | 5,775 | `ac3fd8752db3d286b778f5438297136f07364bd2b27737bd3a74c74db07df10c` |
| `SPEC_g32_r10_n600_maximum_inverse_fitter_20260726.md` | 19,513 | `5449d64302bce4856436d9c13e069bfc0a243da436169ce563a129b300c8f217` |

No G27, G20, G22, G23, G29, pointer, candidate, or public decoder file was
edited. The shared working tree was already heavily dirty; no unrelated change
was modified or claimed. No commit or push was made.

## Production properties

### Exact custody

The launcher reopens and fails closed on the exact objects:

| object | bytes | SHA-256 |
|---|---:|---|
| source video | 37,545,489 | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` |
| selected G20 archive | 81,027 | `8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8` |
| reopened selected `0.bin` | 81,738 | `4789bf6b5f15272cc5f8a573f25137a9daf7e21755e81aa48a8fba84947b5634` |
| frozen FP64 runtime | 56,814 | `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224` |
| frozen G27 receiver | 65,411 | `13cd771d10c333a458c9977f8b21b916a4baf80b063bb4f849f001a6f660e11d` |
| G22 full-n600 receipt | 24,210 | `3a01e81abfd19a78db86e5851f1b0c453ff553c1fe7d5fad830f95bcd5ec3efd` |

The launch binding also hashes the current G32 tool and fitter implementation,
so a resume under changed encoder code refuses rather than inheriting stale
stage state.

### Streaming and storage

- Pair populations are uint8 memmaps on the SSD tier; no n600 RGB tensor is
  loaded into resident memory.
- Selected-base and source materialization are ordered, range-checkpointed, and
  resumable by pair chunks.
- The base realization SHA-256 uses the exact G27 domain and shape prefix in a
  streaming pass.
- Storage preflight includes selected/source raw, three RGB work populations,
  the support-label population, and caller reserve. Local spill requires an
  explicit hidden test-only flag.
- Native runtime metadata distinguishes its 384x512 internal render grid from
  the actual 874x1164 camera/output bytes.

### Immutable resume

The launcher preserves a contiguous content-addressed checkpoint chain:

```text
000_custody
010_selected_base
020_pair_index
030_geometry
040_xip2
050_base_feature
060_texture
070_shooting_knot
080_dash1
090_pullback_polygon
100_stratified_flow
110_joint_refit
120_packet_adapter
130_bounded_decode_receipt
140_cleanup_certificate
```

Fitted pitch, exact XIP2 payload/coordinates, base records, texture/DASH state,
knots, polygons, and flows are retained as sufficient resume state. A resumed
fit skips those completed inverse solves, deterministically rebuilds only
rebuildable intermediates, and must reproduce the exact packet. A completed
run returns directly from the immutable packet/receipt custody; the real n1
completion resume returned the same packet, wrapper, and receipt hashes.

### Counted-state gate

The manifest exactly covers the packet header/directory and every section byte
without gaps or overlaps. Active video-specific values require nonempty packet
spans whose hashes are reopened. The audit refuses:

- an uncounted active claim;
- span/hash drift;
- hidden source-selected thresholds;
- hidden per-pair exception state; and
- packaged video-specific code or weights falsely marked generic/free.

The generic bounded repair ABI is deliberately inactive identity with zero
iterations and zero workspace. For every correction-bearing section, generic
exact regeneration and generic repair remain unavailable and the verdict is
`STORED_COUNTED_STATE_REQUIRED`; no zero-byte repair win is invented.

### G27/G23 and continuation identity

The compiler requires strict G27 parse/re-emit identity, builds the frozen G27
selected-solution adapter, and wraps the exact packet as a deterministic STORE
ZIP member. It constructs an actual `G17PhysicalCodingGroupV1` over reopened
archive/member bytes and exposes packet-, member-, and outer-archive-relative
section/constraint spans.

The continuation-equivalence identity binds:

- exact source and pair-population identities;
- exact selected-base realization;
- exact counted packet bytes/hash;
- frozen G27 receiver operation and source-closure hash;
- canonical pair order; and
- exact uint8 output shape/dtype.

A later G29/G33 public endpoint must use that same base/packet/receiver closure
and reproduce the exact endpoint bytes. It cannot substitute a proxy or a
different base object.

## Real n1 mechanism receipt

The only real-input execution was native n1, using selected pair 0 and source
frames 0/1. It did not import or run the scorer, mux a public video, construct a
candidate, invoke `upstream/evaluate.py`, or move the pointer.

Durable root:

`/Volumes/VertigoDataTier/pact/g32_r10_n600_maximum_inverse_fitter_20260726/n1_real_smoke_v4`

Artifacts:

| artifact | bytes | SHA-256 |
|---|---:|---|
| counted R10 packet | 675 | `3006db7af8122da54a4e03e546fbbe651aa648f9a12dc7e1e0a6a8413959d6f9` |
| deterministic STORE wrapper | 793 | `006842e0fc4fd012ebb9bb112d3f454bf2fc9e983f4024ec2df92eef276869e9` |
| full G32 receipt | — | `269f7f2368533013795c689260ba4fe492e8fda65e981bdde684fe706fc4cc90` |
| cleanup certificate | — | `dd515db4d8360f3c862711eef548fafee422aa9c27fa219c8d4d9e576543bda6` |

Physical section spans are all present in canonical order:

| section | offset | bytes | SHA-256 |
|---|---:|---:|---|
| `PAIR_INDEX` | 260 | 2 | `96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7` |
| `GEOMETRY` | 262 | 4 | `df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119` |
| `BASE_FEATURE` | 266 | 14 | `fe6529a3ee92a69b49192ea14b2454eedd4cc73b6f9d1cabadf4583e77fbb2ef` |
| `TEXTURE` | 280 | 8 | `32163807fba52b20237daf2f6b84b790525722ba4aff7e86cdee318a8b6ccace` |
| `SHOOTING_KNOT` | 288 | 14 | `9b5580edd0e0f1b413717e64067a800831b2b4c65c8a5fee459972cd5b92a135` |
| `XIP2` | 302 | 49 | `8edc65c22b4a41dc4c1c7d11dc3d2d458d1d0a91f854d09d4fef1b81e18c5dfb` |
| `DASH1` | 351 | 286 | `9005c048fb5c318e7a62c768f204db51fee5fe4e3963a36642808d321acca06d` |
| `PULLBACK_POLYGON` | 637 | 22 | `1f60570f5de670dcd49b7249c251815c4cddb0f3bdc5bbeeab5d8e57f1193f47` |
| `STRATIFIED_FLOW` | 659 | 16 | `1a8905855d7edde8d4eded0ca201d8cdfecff199ad334046f6bfaf143a02eb2e` |

Receiver evidence:

- deterministic G27 double replay: `true`;
- selected-base realization:
  `06722660ed18e3f60ae81beb90c35ecebdfe864295ffb673c9d1bd42a00a2467`;
- realized output:
  `e00c2720fb839abcf56adcfff6e86c1386967d46d7a3550bfe5f9f328191af4e`;
- continuation-equivalence identity:
  `d009d994e35598a158696e2067b1cace943f9c6c0c04576ce3d139749c71172c`;
- G27 receiver source closure:
  `b2a0bd4857c995b99b817b57face911c034c3dc2f919b8c9e5464341050fb281`;
- actual changed values: 5,739,429;
- actual changed pixels: 1,986,254;
- actual L1 RGB delta: 48,264,413;
- max absolute RGB delta: 100;
- encoder-only source RGB integer objective: 4,435,693,241 ->
  4,031,037,074.

The post-interaction corner was actually run and preserved but had objective
4,032,401,828, so the exact distortion-first selector correctly retained the
initial sequential corner. This RGB coordinate is encoder fitting evidence,
not d_seg/d_pose or score evidence.

G23 group:

- group ID: `g32.r10.3006db7af8122da5`;
- archive/range SHA-256:
  `006842e0fc4fd012ebb9bb112d3f454bf2fc9e983f4024ec2df92eef276869e9`;
- member SHA-256:
  `3006db7af8122da54a4e03e546fbbe651aa648f9a12dc7e1e0a6a8413959d6f9`;
- receiver operation:
  `tac.witness_dsl.taskspace_r10_feature_texture_relay.decode_r10_packet`.

All six large/rebuildable scratch objects were hashed, certified, and removed.
The cleanup certificate names original paths, bytes/hashes, rebuild command and
environment, source/runtime/archive identities, false-authority flags, and the
rebuildability reason. A first failed smoke preallocation was separately
certified and removed under receipt
`3e9c469528dedbea62b478726ccba9b681183708a707c44dc37dde28290090ec`;
no failed-run bytes were silently deleted.

## Verification

```text
PYTHONPATH=src .venv/bin/pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_r10_n600_maximum_inverse_fitter.py \
  tools/tests/test_fit_taskspace_r10_n600_maximum_inverse.py
21 passed in 0.40s

.venv/bin/ruff check <four G32 Python files>
All checks passed!

.venv/bin/ruff format --check <four G32 Python files>
4 files already formatted

python3 -m py_compile <G32 library> <G32 launcher>
PASS

git diff --check -- <G32 files/spec/findings>
PASS
```

The tests cover real section construction/decode, strict parse/re-emission,
real G23 ZIP custody and offsets, exact counted-state coverage, hidden
threshold/selector/exception/code-as-data attacks, inactive repair truthfulness,
streaming realization identity, n600 double confirmation, immutable contiguous
resume, raw range checkpoints, SSD gating, storage accounting, and
certify-before-delete cleanup.

## Dormant n600 fire command — not executed

This command remains dormant while G28/G14 own live heavy/evaluation work. It
must be re-authorized against current lane ownership before use.

```bash
PYTHONPATH=src .venv/bin/python \
  tools/fit_taskspace_r10_n600_maximum_inverse.py \
  --resume-from /Volumes/VertigoDataTier/pact/g32_r10_n600_maximum_inverse_fitter_20260726/n600_fire \
  --video upstream/videos/0.mkv \
  --selected-archive .omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_lossless_xcodec_recode_20260726/ep725_lossless_xcodec_recode.not_a_candidate.zip \
  --runtime /Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py \
  --pair-count 600 \
  --chunk-pairs 2 \
  --seed 3200260726 \
  --sample-stride 8 \
  --reserve-bytes 2147483648 \
  --execute-reviewed \
  --confirm-full-n600 \
  --confirm-no-live-heavy-owner
```

## Exact remaining blockers

1. No full-n600 G32 fit has been fired. Therefore full-n600 analytic exhaustion,
   packet bytes, endpoint hash, runtime, and memory telemetry do not exist yet.
2. G29 still owns public receiver/mux closure. The 793-byte research wrapper is
   physical-custody evidence, not a public contest archive or complete candidate.
3. No full-n600 scorer or complete `archive.zip` evaluation has run. `d_seg`,
   `d_pose`, complete candidate ZIP bytes, score units per byte, contest score,
   and promotion eligibility remain null.
4. Root/G23 must bind the G32 physical-group adapter into the product placement
   manifest; G32 does not edit or clear G23's current receiver-consumption
   blocker.
5. The generic repair ABI is inactive. No correction section has a proven exact
   generic regenerator from other counted state.
6. Terminal joint descent remains unadmitted. It may be typed only after an
   authorized full-n600 maximum-inverse run inventories receiver-realized debt;
   it must remain terminal-only and counted.
7. The canonical frontier pointer is unmoved. This unit built a means toward an
   exact row; it did not lower the exact contest score and therefore did not
   achieve the project goal.

## Stores consulted

- byte-identical `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, and craft handoff;
- top project memory: inverse-frozen-space thesis, V10 realization crux,
  canonical preimage, realization completeness, Kolmogorov projection,
  no-duplicate-data, vehicle naming, checkpoint maturity, and flat-amplitude
  exhaustion;
- latest DDM RS1 feature-relay directive, canonical pointer, lane registry, and
  live subagent ownership;
- G20 selected archive/runtime receipts, G22 full-n600 exact equality receipt,
  frozen G27 packet/receiver/adapter/spec/findings, and G23 physical-group plus
  terminal inverse-solve schedule.
