# PR110 Sparse LFV1 + Contest Atom Lattice Execution Memo

Date: 2026-05-20T16:51:24Z
Author: Codex
Scope: PR110-focused local engineering, diagnostic only

## Verdict

PR110 is still open, not merged. Current upstream PR state:

- head: `adpena/comma_video_compression_challenge:hnerv_fec6_fixed_huffman_k16`
- head SHA: `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`
- state: `open`
- merged: `false`
- updated_at: `2026-05-20T14:46:27Z`

Engineering should continue against PR110-compatible artifacts for now to avoid
post-merge debt, but the work below is not a PR110 mutation and makes no score
claim.

## PR110 Runtime Custody

Downloaded PR110 head tarball:

- path: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/source/pr110_head_ec6cc7f.tar.gz`
- SHA-256: `d9286e8bb5831da2c3e381c710866d812ec3f54f0bb76046b5c5658ba3c984c6`

The downloaded PR110 runtime source was byte-identical to the local
authoritative submission runtime for `inflate.py` and `inflate.sh`. The PR110
tarball does not contain `archive.zip`; the local authoritative archive was
used for provisional engineering.

Authoritative archive:

- path: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/archive.zip`
- bytes: `178517`
- SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`

Patched provisional runtime:

- path: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/inflate.py`
- SHA-256: `d5d9d19934b51f56fde42fc0c213bc8945af347fde64c4857256131712acfb19`
- supports no-sidecar original behavior, full HFV1 sidecar, LFV1 sparse v1,
  and LFV1 sparse v2 fine-alpha sidecars.

## Sparse LFV1 Diagnostic Results

Baseline raw advisory eval:

- axis: `[macOS-CPU advisory]`
- score_claim: `false`
- canonical score: `0.19206142414659494`
- PoseNet: `2.943e-05`
- SegNet: `0.00056039`
- rate: `0.00475469`
- archive bytes: `178517`
- raw SHA-256: `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`

LFV1 v1 top-8 pose-selected candidate:

- candidate: `lfv1_pose_top8_alpha00045_radius070_budget160`
- archive bytes: `178765`
- archive delta: `+248`
- sidecar bytes: `116`
- archive SHA-256: `7b89c1d0cbb107bffb6510718cf84489e75e18b1de9521c2c335683fdac4e2ba`
- official inflate locality control: passed
- changed frames matched selected pairs exactly
- advisory score: `0.33289345900974876`
- delta vs baseline: `+0.14083203486315382`
- failure mode: destructive PoseNet regression (`+0.00227894`)

LFV1 v2 top-8 micro-alpha candidate:

- candidate: `lfv1v2_pose_top8_alpha00002_radius070_budget160`
- archive bytes: `178765`
- archive delta: `+248`
- sidecar bytes: `116`
- archive SHA-256: `be7691a7ea0cd7eb249829df46886222ffeeecba2bc9e4be5043d31dff2ef057`
- official inflate locality control: passed
- changed frames matched selected pairs exactly
- advisory score: `0.19222942414659494`
- delta vs baseline: `+0.00016800000000000148`
- interpretation: non-destructive but not beneficial; the loss is essentially
  the rate penalty plus tiny SegNet movement.

## Canonical Contest Atom Lattice

New reusable module:

- `src/tac/atom/contest_granularity.py`
- SHA-256: `4156ab6e35367b18f09a78598b8c38341db085dd062f7945fe803282ea218ad2`

New lattice builder:

- `tools/build_contest_atom_lattice.py`
- SHA-256: `e13f7c9be1f759323f32f6e81ebf206941286502bfadad87250e5fcbfc211122`

Artifact:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_atom_lattice/pr110_fec6_lfv1v2_lattice.json`
- SHA-256: `1dd84f2ad06bdecd13aa4bc73ed7f71a01f1aad3ff9ce3d862790b3d80d60820`

Lattice contents:

- total atoms: `768`
- byte atoms: `64`
- frame atoms: `128`
- pair atoms: `448`
- pixel-region atoms: `128`
- Venn counts:
  - `master_gradient`: `64`
  - `pair_component`: `392`
  - `pair_component&sidecar_selected`: `56`
  - `xray_pair`: `128`
  - `xray_pixel`: `128`
- pair-level overlap:
  - total scorer pairs represented: `118`
  - `pair_component`: `46`
  - `pair_component&sidecar_selected`: `8`
  - `pair_component&xray_pair&xray_pixel`: `10`
  - `xray_pair&xray_pixel`: `54`

The lattice exports:

- `atoms`
- `meta_lagrangian_rows`
- `cathedral_candidate_rows`
- `top_waterfill_atoms`
- pair-level Venn overlap

This is the canonical bridge for bytes, pixels, frames, pairs, master-gradient
anchors, xray surfaces, and cathedral/autopilot ingestion.

## Deterministic Oracle Search Seed Queue

New reusable planner:

- `src/tac/optimization/contest_oracle_search.py`
- SHA-256: `3f606040181aa82d5e95d119d7d37b95f71b0911f008a24f0fdf6c674d0b1af6`

New CLI:

- `tools/plan_contest_oracle_search.py`
- SHA-256: `b15c3bf4b11d8929fda158ffbce7ff09faef147583bd35751339a396f570e7a0`

Artifact:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/lfv1v2_seed_queue.json`
- SHA-256: `9192b5d0fb0ee706ea72915f8e8d7285d912812b6dde3d6f876fbc1864ecc942`

Queue contents:

- candidates: `64`
- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`
- k=1..8 byte staircase covered evenly:
  - k=1: 8 candidates, estimated delta `157` bytes
  - k=2: 8 candidates, estimated delta `170` bytes
  - k=3: 8 candidates, estimated delta `183` bytes
  - k=4: 8 candidates, estimated delta `196` bytes
  - k=5: 8 candidates, estimated delta `209` bytes
  - k=6: 8 candidates, estimated delta `222` bytes
  - k=7: 8 candidates, estimated delta `235` bytes
  - k=8: 8 candidates, estimated delta `248` bytes

The queue is dry-run/planning-only. It is meant to seed a later materializer
that builds byte-closed archives, runs official inflate controls, then runs the
cached CPU scorer-oracle wrapper.

## Scorer Oracle Cache

`tools/run_raw_advisory_eval.py` now supports `--reuse-cache` for exact
raw/archive/device/settings matches. This is required before any genetic,
beam, coordinate, or water-bucket loop is scaled beyond a handful of local CPU
full-scorer calls.

Tool SHA-256:

- `ee900e27c1e8093bbced0825d1eba8a72f8cb32411cf95add78f8e6afd45ad6a`

## Tests

Focused tests passed:

```text
.venv/bin/python -m pytest -q \
  src/tac/atom/tests/test_contest_granularity.py \
  src/tac/tests/test_contest_oracle_search.py

3 passed in 0.18s
```

Also compiled successfully:

- `tools/build_hfv1_sidecar_candidate.py`
- `tools/run_raw_advisory_eval.py`
- `tools/build_contest_atom_lattice.py`
- `tools/plan_contest_oracle_search.py`
- `src/tac/atom/contest_granularity.py`
- `src/tac/optimization/contest_oracle_search.py`
- provisional `runtime_hfv1/inflate.py`

## Rust/Zig Boundary

Keep Python as the canonical integration and custody layer. Lower hot loops
into Rust or Zig only behind stable JSON/bytes contracts:

- candidate queue rows in;
- byte-closed archive/raw candidate out;
- raw/archive SHA-256 and scorer deltas back into the lattice.

Likely accelerator targets, in order:

1. archive mutation and LFV1/HFV1 materialization;
2. pixel-region transforms and frame diffing;
3. byte/gradient feature extraction;
4. scorer-oracle scheduling and cache lookup.

The scorer itself remains the dominant wall-clock cost on local CPU, so search
quality and caching matter more immediately than rewriting orchestration.

## M5 Max Local Saturation Runner

New batch runner:

- `tools/run_contest_oracle_batch.py`
- SHA-256: `f724603efe66d6638d49fcd356324eeaa6f558d4266faf24c612d32c9e9d95c5`

Dry-run artifact:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_batch/lfv1v2_m5max_top4/batch_manifest.json`
- SHA-256: `825725d9dcb11b1f99b286eee3c32e05b862e37424c52e8c71b06bff3d91fd80`

Observed local hardware:

- CPU: Apple M5 Max
- logical CPUs: `18`
- RAM: `137438953472` bytes
- current free disk on volume: about `133 GiB`

Runner contract:

- queue JSON in;
- PR110-compatible byte-closed LFV1 candidates out;
- optional cached `[macOS-CPU advisory]` raw eval out;
- `score_claim=false`;
- `ready_for_exact_eval_dispatch=false`.

Disk caution: each inflated raw is about `3.66 GB`. Until the separate build
artifact cleanup frees space, run this runner in dry-run mode or with small
caps such as `--max-candidates 4 --max-parallel 4`; do not launch a full 64-row
scorer batch on the current disk headroom.

## Next Action

Do not broaden again yet. The frontier-moving artifact path is:

1. Materialize the first small batch from
   `contest_oracle_search/lfv1v2_seed_queue.json`.
2. Use `tools/build_hfv1_sidecar_candidate.py` to build byte-closed candidates.
3. Use official inflate locality controls.
4. Use `tools/run_raw_advisory_eval.py --reuse-cache`.
5. Append measured deltas back into a new lattice/probe report.
6. Only dispatch exact CUDA if a byte-closed candidate beats the local
   `[macOS-CPU advisory]` baseline by more than the expected CPU/CUDA
   uncertainty and rate penalty.

Current measured LFV1 conclusion: v1 coarse-alpha is destructive; v2 micro-alpha
is safe but neutral. The next search should prioritize Venn-overlap pairs from
the new lattice, not the older pose-only top-8 selection.
