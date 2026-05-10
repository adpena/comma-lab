# Custom codec composition roadmap + PR106 exhaustive recode result (2026-05-10)

Generated: `2026-05-10T17:45:00Z`

`score_claim=false`; `dispatch_attempted=false`; `research_only=false`;
`ready_for_exact_eval_dispatch=false`.

## Operator framing

The operator explicitly wants the public PR deconstruction to feed a broader
cross-paradigm stack, not become a local minimum. The correct response is a
byte-closed packet compiler and score-aware training stack that can compose:

`representation -> prediction -> quantization -> hyperprior -> arithmetic -> pack`

with runtime arithmetic, sidecars, wavelet/foveation priors, VQ/codebook/FiLM,
Ballé/T1/T6, Lane 12-v2 renderer work, PR95 training discipline, and HNeRV
public-frontier artifacts. HStack/VStack/cross-archive composition is allowed
only after a verified contest-CUDA substrate anchor and a single runtime that
actually consumes the composed bytes.

## Work completed in this tranche

### 1. Frontier xray comparison

Generated:

- `experiments/results/xray_custom_codec_frontier_20260510_codex/section_entropy/heatmap.md`
- `experiments/results/xray_custom_codec_frontier_20260510_codex/layout_compare/layout_compare.md`
- `experiments/results/xray_custom_codec_frontier_20260510_codex/op_cost/op_catalog.md`
- `experiments/results/xray_custom_codec_frontier_20260510_codex/packet_compiler_pr106_q10_release_surface.json`
- `experiments/results/xray_custom_codec_frontier_20260510_codex/packet_compiler_pr103_drop_u32_release_surface.json`
- `experiments/results/xray_custom_codec_frontier_20260510_codex/packet_compiler_pr106_exhaustive_release_surface.json`

Key results:

- Across PR101, PR103-PR106-AC, destructive PR103 drop-u32, PR106 source, and
  PR106 q10, the section entropy upper-bound shows only `167` total recoverable
  bytes if each single archive member hit its payload entropy floor.
- PR101 `x` has `30` recoverable bytes by this diagnostic; PR103-PR106-AC has
  `26`; PR106 source has `44`; PR106 q10 has `41`. Raw entropy shaving is
  saturated at medal-band scale.
- Inflate op-cost xray over PR100/101/103/106 reports `298` static ops and only
  PR101 has the 3 per-channel mutations that match the known medal-delta bias
  correction pattern.

Interpretation: blind payload shaving and arbitrary runtime constants are now
low-EV. The next score movement must come from either scorer-aware training or
grammar-preserving model-aware recoding with exact runtime-consumption proof.

### 2. PR106 exhaustive grammar-preserving Brotli recode

Command:

```bash
.venv/bin/python tools/build_hnerv_lowlevel_repack_candidate.py \
  --source-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
  --scorecard experiments/results/hnerv_frontier_scorecard_refresh_20260510_codex/scorecard.json \
  --source-label PR106 \
  --output-dir experiments/results/hnerv_lowlevel_repack_pr106_exhaustive_20260510_codex \
  --quality 0 --quality 1 --quality 2 --quality 3 --quality 4 --quality 5 \
  --quality 6 --quality 7 --quality 8 --quality 9 --quality 10 --quality 11 \
  --lgwin default --lgwin 10 --lgwin 11 --lgwin 12 --lgwin 13 --lgwin 14 \
  --lgwin 15 --lgwin 16 --lgwin 17 --lgwin 18 --lgwin 19 --lgwin 20 \
  --lgwin 21 --lgwin 22 --lgwin 23 --lgwin 24 \
  --lgblock default --lgblock 16 --lgblock 17 --lgblock 18 --lgblock 19 \
  --lgblock 20 --lgblock 21 --lgblock 22 --lgblock 23 --lgblock 24 \
  --jobs 8 \
  --json-out experiments/results/hnerv_lowlevel_repack_pr106_exhaustive_20260510_codex/result.json
```

Result:

- Candidate archive:
  `experiments/results/hnerv_lowlevel_repack_pr106_exhaustive_20260510_codex/pr106_hnerv_brotli_repack_candidate.zip`
- Archive SHA-256:
  `a0bcd3f2288edd53dc9a7ae7a8e37e7b0384ae8a8dbdae7b2ae978f33bf5b139`
- Archive bytes: `186087`
- Delta vs PR106 source: `-152` bytes
- Delta vs prior q10 packet: `-1` byte
- Best transform: `decoder_packed_brotli`, `quality=10`, `lgblock=16`,
  default `lgwin`
- `candidate_diff_audit.blockers=[]`
- `rate_score_delta_if_components_equal=-0.000101210561`

This is a valid grammar-preserving, raw-equivalent byte candidate, but it is
not worth exact CUDA spend alone because PR106 q10 exact CUDA was still worse
than the active floor and this improves q10 by only one additional byte.

### 3. Static packet build and DX hardening

The first static packet build exposed a tool bug:
`refresh_dispatch_readiness()` passed `args.now_utc` directly to
`subprocess.run`; when omitted, `None` entered the command list and crashed.

Fix:

- `tools/build_hnerv_lowlevel_exact_eval_packet.py` now normalizes omitted
  `--now-utc` through `_format_utc(_now_utc(args))`.
- Regression test added:
  `test_refresh_dispatch_readiness_default_now_is_subprocess_safe`.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_lowlevel_exact_eval_packet.py -q
# 7 passed
```

After the fix, the PR106 exhaustive static packet built successfully:

- Packet:
  `experiments/results/hnerv_lowlevel_repack_pr106_exhaustive_packet_20260510_codex/packet.json`
- Release surface:
  `experiments/results/hnerv_lowlevel_repack_pr106_exhaustive_packet_20260510_codex/release_surface/`
- `static_packet_ready=true`
- `ready_for_exact_eval_dispatch_claim=true`
- `static_blockers=[]`
- `ready_for_submit=false`
- Remaining blockers:
  `missing_lightning_environment`, `missing_active_lane_dispatch_claim`,
  `missing_operator_exact_cuda_approval`, and no exact CUDA score.
- `tools/submission_packet_compiler.py ... --mode inspect` on the release
  surface reports `blockers=0`.

## Current interpretation

The public PR work has not been wasted, but the path is narrower than earlier
roadmaps implied:

1. PR106 q10/exhaustive proves safe, grammar-preserving recoding can create
   tiny rate-only wins.
2. PR103 drop-u32 proves destructive raw range-stream edits can catastrophically
   corrupt high-saliency tensors.
3. PR101 bias tuning proves CPU/proxy medal-band numbers can fail on CUDA.
4. The HNeRV/Brotli payload is saturated enough that one-off entropy changes are
   not the route to sub-0.17.
5. The high-EV paths are now T1/T6 score-aware training, model-aware section
   transforms, sidecar search with scorer feedback, and a typed packet transform
   compiler that can prove runtime consumption.

## Concrete next implementation tranche

P0. Harvest active T1 Modal.

- Active lane: `t1_balle_128k_endtoend`
- Active job: `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- Recover command:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_phase1_ab2d0f6_20260510T1437Z
```

Do not launch another T1 job until this claim closes terminally.

P1. Build the missing typed packet-transform bridge.

The repo already has:

- `src/tac/submission_packet_compiler.py`: generic inspect/identity custody.
- `src/tac/phase1_packet_compiler.py`: fuller phase-1 custody compiler.
- `src/tac/analysis/hnerv_packet_sections.py`: HNeRV section manifests.
- `src/tac/hnerv_lowlevel_packer.py`: strict single-member HNeRV repacking.
- `src/tac/codec_pipeline.py`: state-dict codec operations.
- `src/tac/pr103_arithmetic_codec.py`: reusable PR103 arithmetic codec.

Missing durable abstraction:

- `PacketIR`: archive identity, member identity, parser-section manifest,
  runtime tree manifest.
- `PacketSectionTransform`: `applies_to(section)`, `transform(bytes)`,
  `runtime_adapter_required`, `score_affecting_payload_changed`.
- `TransformResult`: old/new section SHA, byte delta, raw-equivalence proof,
  no-op proof, runtime-consumption proof.
- `CandidatePacketCompiler`: applies transforms, writes deterministic ZIP,
  emits manifest, refuses score claims, and hands exact eval only to existing
  dispatch gates.

Smallest useful slice: wrap the existing PR106 Brotli recode and PR103
arithmetic paths behind this interface, with golden vectors from the packet
compiler/xray artifacts above.

P2. Rebase model-aware recoding onto the active PR101/A1 export path.

Do not dispatch HDM3/PR106 recodes as-is; current byte wins are dominated. Use
the transform bridge to test whether PR103 arithmetic, PR101 split-Brotli, or
derived byte-map choices can lower the active PR101/A1 export packet while
preserving runtime outputs.

P3. Keep wavelet/foveation as score-domain priors and runtime-consumed atoms.

Wavelet/telescopic foveation should be used in:

- T1/T10/T6 score-aware training masks.
- Lagrangian bit allocation features.
- Runtime-consumed charged atoms only after no-op proof.

Do not promote weight-domain wavelet/foveation proxies as score claims.

P4. Use HStack/VStack only through a single verified substrate.

Cross-archive composition is not forbidden as a research idea. It is forbidden
as a contest dispatch until there is one verified contest-CUDA substrate anchor
and a single packet compiler/runtime that consumes the composed bytes. The
implementation target is section-transform composition, not concatenating
archives.

P5. Provider execution.

Lightning is likely unavailable. Use Modal first for exact CUDA when justified;
use Kaggle/MPS only for proxy sweeps, CMA-ES/Optuna config discovery, and curve
measurement. They must never become auth-eval claims.

P6. Do not lose exact-eval custody.

Every future score-lowering candidate needs:

- archive path, bytes, SHA-256;
- runtime tree SHA;
- old/new section hashes and byte deltas;
- raw-equivalence or scorer-visible-change proof;
- no-op proof;
- 600-sample exact CUDA result before promotion;
- terminal dispatch claim;
- result-review classification.
