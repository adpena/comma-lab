# molt pact-collab addendum — contribution note (2026-06-29)

## Situation found on molt `main`

The task assumed the prior pact↔molt state was the 2026-05-11 byte-transducer
plan (`molt as a weight-blob native backend`) and that the `pact-collab` branch
was gone. On clone, the reality on `origin/main` was **far more advanced**: the
`collab/pact/` channel had already evolved through reports **001–007** plus a
runnable witness-kernel bundle (`pact_witness_kernel/` with fixtures, reference
outputs, and `check_parity.py` oracle). Specifically:

- **`006_precise_contract_full_witness_pipeline.md`** already captured the v2
  task-space witness pivot, the exact two kernels (A = Morse-Smale field-solve,
  B = level-set INR forward), and the full determinism-gate table.
- **`007_molt_response_*.md`** is the **molt team's reply**: they greened the
  NumPy/SciPy C-API scan / missing-symbol layer (447 NumPy + 592 SciPy source
  files, zero missing symbols) and set the current milestone to Kernel A WASM
  parity (`check_parity.py candidate_outputs.npz` → PASS).
- The `README.md` and `STATUS.md` on `main` are already current (dated
  2026-06-29) and track a specific live blocker (Kernel A WASM package-native
  closure).

The prior `pact-collab` branch had been merged to `main` and deleted (no remote
`pact-collab` head at clone time).

## What I contributed (ADDITIVE — no clobber, no signal loss)

Per the no-signal-loss + don't-disrupt-the-molt-team disciplines, I did **not**
overwrite the current `README.md` / `STATUS.md` (doing so would have regressed
the channel to a less-current state). Instead, additively:

1. Added **`collab/pact/008_addendum_v2_witness_decoder_20260629.md`** — a
   consolidation + extension that (a) maps the full deterministic geometric
   decode CHAIN (SE(3) screw-warp `src/tac/se3.py` + EON ground homography
   `src/tac/camera.py`, per class) as molt compile targets above the 006
   kernels; (b) states the rule-118 rate-half value prop; (c) adds the explicit
   contest-runtime hard contracts (30-min T4-or-CPU budget; CPU/CUDA separate
   axes); (d) sharpens the open compat question to "is WebGPU/WASM available in
   the headless contest `inflate.sh` runner"; (e) documents the runtime-rs Rust
   native sister-backend relationship (one numpy-fp32 oracle, two backends).
2. Added one **row to the `README.md` correspondence table** pointing at 008
   (purely additive edit; all existing content preserved).

The 008 addendum explicitly builds on 001–007 and does **not** re-litigate the
v2 pivot (already in 006) or claim the C-API layer as open (already greened in
007). It is honest about being additive.

## Files in this local fallback dir

- `008_addendum_v2_witness_decoder_20260629.md` — exact copy of the file pushed
  to molt `collab/pact/`.
- `CONTRIBUTION_NOTE.md` — this note.

This is a MEANS (a collab deliverable). The sub-0.15 exact-score pointer is
UNMOVED at 0.19110.
