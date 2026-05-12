# SIREN L2 FFT encoder bridge - 2026-05-12

## Status

SIREN remains the highest-priority non-HNeRV residual-basis lane, but it is
still **not a score claim**. This landing closes the concrete implementation
gap between the earlier `empty` / `zero` / `probe` scaffold and an actual
byte-emitting residual fit:

- `tools/materialize_siren_residual_pr106_sidecar.py --residual-mode l2_encoded`
  now reads decoded RGB raw frames and ground-truth RGB raw frames, computes
  `gt - decoded`, selects the largest low-frequency FFT coefficients inside a
  configurable signed-frequency window, quantizes them into the SIREN
  9-byte coefficient grammar, and emits a byte-budgeted residual payload.
- The encoder can optionally repack the dense coefficient stream through the
  canonical sparse residual wrapper via `--encoding sparse`.
- The materialization manifest records `l2_encoder_diagnostics` so every
  emitted packet carries the frame count, candidate frequency bins, selected
  coefficient count, emitted residual bytes, byte budget, max-k window, and
  sparse-wire flag.
- The manifest diagnostics also record raw-input custody for L2 builds:
  decoded/ground-truth raw paths, SHA-256 hashes, byte counts, frame counts,
  decoded axis, decoded inflate device, and decoded runtime SHA fields. For
  non-synthetic builds, the materializer refuses to run unless a decoded runtime
  SHA or runtime-tree SHA is supplied.

## Evidence boundary

This is a deterministic proxy encoder and a dispatch enabler. It does **not**
promote SIREN, retire adjacent lanes, or imply any CPU/CUDA score movement.
Per the latest Claude memories, this is the **residual-basis sidecar** SIREN
path, not the full `src/tac/substrates/siren/` renderer substrate. The full
substrate remains governed by the substrate taxonomy and the Catalog #164
real-scorer `preprocess_input` sister-fix wave.

Required before any score claim:

1. Build a byte-closed SIREN candidate from real PR106 decoded raw frames and
   the matching contest ground-truth raw frames.
2. Prove the residual payload bytes differ from scaffold/probe bytes and are
   consumed by `submissions/pr106_siren_residual_sidecar/inflate.py`.
3. Run exact auth eval on the intended axis with archive SHA, runtime-tree SHA,
   hardware, command, logs, component deltas, and formula recomputation.
4. Keep `[contest-CPU]` and `[contest-CUDA]` separate; no conversion, ranking,
   or submission decision from one axis to the other.

## Verification

Local verification on 2026-05-12:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_materialize_residual_pr106_sidecars.py \
  src/tac/tests/test_residual_basis_nonhnerv_scaffolds.py \
  src/tac/tests/test_residual_basis_pr106_sidecar_packing.py \
  src/tac/tests/test_residual_basis_pr106_materializer_helpers.py -q
# 151 passed in 24.14s

.venv/bin/python -m pytest src/tac/tests/test_preflight_proactive_checks.py -q
# 24 passed in 6.62s

.venv/bin/python -m ruff check \
  tools/materialize_siren_residual_pr106_sidecar.py \
  src/tac/tests/test_materialize_residual_pr106_sidecars.py \
  src/tac/tests/test_preflight_proactive_checks.py
# All checks passed.

env GITHUB_ACTIONS=true HOME=/tmp/pact-ci-home-scheduled \
  PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 \
  PYTHONPATH=src:upstream:$PWD \
  .venv/bin/python -m tac.preflight \
  --timings-json /tmp/pact_preflight_ci_scheduled_cold.json
# PREFLIGHT PASSED; wall_elapsed_s = 8.026546
```

## Adversarial fixes before landing

- Sparse L2 output was protected from accidental double-repack. The
  materializer now tracks whether `l2_encoded` already emitted sparse PacketIR
  bytes, and only applies the generic dense-to-sparse repacker to dense payloads.
- Heap ordering avoids raw complex values as comparison keys, so deterministic
  tie-breaking cannot fall into Python's non-orderable complex comparison.
- Explicit `--n-frames` now refuses decoded/ground-truth raw inputs that are too
  short instead of silently truncating to the shorter file. This prevents hidden
  signal loss in candidate builds.
- Coefficient extraction is vectorized per frame: one batched FFT over the two
  spatial axes produces all RGB channel spectra, then `argpartition` selects the
  frame-local top components across channel/band candidates before the global
  heap. This preserves top-K correctness while avoiding three separate FFT calls
  per frame.
- Non-DC frequency selection is atomized as real-signal conjugate pairs. Because
  inflate applies `real(ifft2(...))`, a lone non-self-conjugate complex bin
  reconstructs at half amplitude; the encoder now selects either self-conjugate
  singletons or complete conjugate-pair atoms.
- Sparse byte-budget search considers only complete atom prefixes, so rate
  pressure cannot split a conjugate pair into a half-amplitude artifact.
- Sparse L2 refuses to emit an empty no-op payload when nonzero coefficients
  were selected but cannot fit inside the sparse byte budget.
- Regression tests cover dense L2 DC recovery, sparse L2 decode parity, and the
  silent-truncation refusal.

## Latest memory refresh

Reviewed on 2026-05-12:

- `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md`
- `feedback_session_2026_05_12_lessons_learned_canonicalization_discipline.md`
- `feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md`
- `feedback_l2_sparse_aware_encoders_first_dispatch_landed_20260511.md`
- `feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512.md`
- `feedback_fix_h_trainer_shape_fix_landed_20260512.md`

Implications applied here:

1. SIREN has two valid traditions: `tac.residual_basis.siren_residual` is a
   sidecar primitive; `tac.substrates.siren` is a renderer substrate. This
   patch touches only the sidecar materializer.
2. Sparse PacketIR is runtime-valid, but prior L2 sparse-aware work found that
   raw YUV6/L2 proxy residuals do not reliably align with the contest score at
   the PR106 r2 operating point. Therefore this materializer is a candidate
   builder, not a promotion/ranking authority.
3. The next scientifically meaningful SIREN upgrade is scorer-aligned residual
   selection: Hinton-distilled scorer surrogate, offline PoseNet/SegNet saliency
   masks, or pose-axis-only residual targeting.
4. Any full SIREN substrate dispatch must first clear the Catalog #164
   real-scorer preprocess-input bug class.

## Next score-lowering action

The immediate next SIREN tranche is not another memo. It is a real candidate
build:

```bash
.venv/bin/python tools/materialize_siren_residual_pr106_sidecar.py \
  --pr106-archive submissions/pr106_latent_sidecar_r2/archive.zip \
  --output-dir experiments/results/lane_siren_l2_fft_pr106_$(date -u +%Y%m%dT%H%M%SZ) \
  --residual-mode l2_encoded \
  --decoded-raw <pr106_decoded_874x1164_rgb.raw> \
  --gt-raw <contest_gt_874x1164_rgb.raw> \
  --n-frames 1200 \
  --byte-budget 2048 \
  --max-k 32 \
  --encoding sparse \
  --decoded-axis contest_cuda \
  --decoded-inflate-device cuda \
  --decoded-runtime-tree-sha256 <runtime_tree_sha256>
```

If that candidate is byte-closed and runtime-consumed, dispatch it as a paired
CPU/CUDA exact-eval lane with explicit dispatch claims. If decoded/GT raw files
are unavailable, generate them first and treat that as the next blocker to burn
down.
