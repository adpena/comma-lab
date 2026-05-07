# HNeRV Low-Level Brotli Expanded Saturation - 2026-05-07

## Scope

Deterministic local-only Brotli parameter sweep on the current PR106x
low-level Brotli candidate. No lane was claimed, no remote/GPU dispatch was
attempted, and no score is claimed.

## Command

```bash
.venv/bin/python tools/build_hnerv_lowlevel_repack_candidate.py \
  --source-archive experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip \
  --source-label PR106x \
  --output-dir experiments/results/hnerv_lowlevel_repack_pr106x_expanded_20260507_codex \
  --target-section decoder_packed_brotli \
  --target-section latents_and_sidecar_brotli \
  --quality 0 --quality 1 --quality 2 --quality 3 --quality 4 --quality 5 \
  --quality 6 --quality 7 --quality 8 --quality 9 --quality 10 --quality 11 \
  --lgwin default --lgwin 10 --lgwin 11 --lgwin 12 --lgwin 13 --lgwin 14 \
  --lgwin 15 --lgwin 16 --lgwin 17 --lgwin 18 --lgwin 19 --lgwin 20 \
  --lgwin 21 --lgwin 22 --lgwin 23 --lgwin 24 \
  --lgblock default --lgblock 16 --lgblock 17 --lgblock 18 --lgblock 19 \
  --lgblock 20 --lgblock 21 --lgblock 22 --lgblock 23 --lgblock 24 \
  --jobs 8 \
  --json-out experiments/results/hnerv_lowlevel_repack_pr106x_expanded_20260507_codex/result.json
```

## Result

- source archive bytes: `186080`
- best candidate archive bytes: `186079`
- total byte delta: `-1`
- best decoder recode: `quality=10`, `lgblock=16`, `lgwin=default`
- decoder section delta: `170127 -> 170126` (`-1`)
- latents/sidecar best delta: `0`
- ready_for_exact_eval_dispatch: `false`
- dispatch blockers: archive-manifest preflight, lane claim, exact CUDA auth eval

## Interpretation

The generic deterministic Brotli parameter basin is saturated around the
existing `lgblock=16` one-byte decoder win. Further local score-lowering work
should prioritize grammar-changing candidates such as HPM1/categorical
recovery, semantic sidechannels, or stacked component-affecting lanes rather
than more Brotli parameter search.

The scratch output directory was rebuildable and left untracked; this ledger is
the canonical retained signal for the sweep.
