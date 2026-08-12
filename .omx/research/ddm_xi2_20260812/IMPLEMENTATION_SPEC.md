# DDM XI2 full-scale xi-context promotion implementation spec

## Objective

Build a scorer-free, full-n600, 60-epoch, lambda-1.0 promotion runner for the
decode-derivable xi-warped-previous-partition HPAC context.  The treatment must
be comparable to the already-banked CL1 spatial control without retraining that
control.  This turn stops at `READY_TO_FIRE`; it must not execute Metal/MPS.

## Architectural decision

Add a new owned runner, `tools/run_ddm_xi2_xi_context_full_scale.py`, derived
from and SHA-pinned to XI1, while reusing the proven CL1 topology, seed,
optimizer, EMA law, 60-epoch schedule, terminal-EMA packer, sparse HPAC
inference, and Range coder.  Do not edit the CL1 trainer or fitter: their
current bytes are part of the banked-control attestation.  Do not edit the XI1
runner: its old receipts pin its current bytes.  Import/reuse their helpers only
after verifying their source SHA-256 values.

The banked comparator is the CL1 lambda-1.0 uninterrupted terminal:

- Range payload: 116,716 B, SHA-256
  `ac2c549c1f48756ad33c6c99af8563f2170db1de61cd50d0615d4c1a0cdd7b87`
- decoded canonical tokens: SHA-256
  `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`
- real training safe-run elapsed: 2,894.155 s for 60 epochs, or
  48.2359166667 s/epoch
- terminal pack/encode/decode receipts and artifacts live under
  `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_uninterrupted_twin/training/`

The XI1 zero-plane arm is not this comparator.  XI2 replaces CL1's unwarped
previous decoded partition with its xi-warped version; capacity and every other
training coordinate remain matched.

## Required files

- New: `tools/run_ddm_xi2_xi_context_full_scale.py`
- New: `tools/tests/test_run_ddm_xi2_xi_context_full_scale.py`
- Do not touch any other source/test file while implementing this spec.

## Runner contract

1. Parser exposes a real `--leg` argument with explicit stages such as
   `prepare`, `train`, `pack`, `encode`, `decode`, `finalize`, and `all`.
   Expose `--resume-from`; `--resume-from auto` must start fresh only when no
   checkpoint exists and must continue from the durable latest checkpoint
   after a crash.  No caller may use an undeclared flag.
2. All state and payloads live under
   `/Volumes/APDataStore/pact/ddm_xi2_20260812/`.  Writes are atomic.  Refuse
   conflicting overwrite; allow byte-identical replay.  Every materialized
   checkpoint, context, packed model, Range stream, repeat Range stream, and
   decoded token tensor is retained with bytes and SHA-256 in machine-readable
   receipts.
3. Pin and verify XI1, CL1 trainer, canonical cache, initializer, pose plane,
   calibration plane, banked control Range stream, banked control decoded raw,
   and relevant PR130 intake sources before work.
4. Full treatment is n600, lambda=1.0, epochs=60, batch=8, eval batch=4,
   eval every 2, lr=0.003, lr-exponent=0.0002, lr-bits=0.01, bit-eps=1e-6,
   QAT fraction=0.5, initial bits=8, channels=64, patch=64, delta=2,
   frame-dim=8, norm=none, activation=relu, frame scale on, bounds=127,
   weight scales on, exponent min=-6, SPM on, target raw, seed=20260716,
   EMA target seed fraction=0.01, train device MPS.  Use an MPS generator and
   MPS `randperm`, matching CL1 rather than XI1's n120 CPU-permutation screen.
5. Preserve an initial checkpoint, every epoch checkpoint, continuous-stage
   end, QAT-stage end, and latest pointer.  Checkpoints contain live weights,
   EMA shadow, optimizer, scheduler, all RNG states, complete config/input/source
   identity, and resume lineage.  Resume must fail closed on any drift.
6. The context algorithm is exactly XI1's class composite:
   previous decoded partition plus carried pose row -> `tac.lie` SE(3) ->
   ground homography for classes 0/1/3, rotation-only for class 2, identity
   preservation for class 4.  Frame 0 uses zeros.  Prepare and retain the full
   n600 context and a deterministic repeat.  Training consumes that context.
7. Encoding and decoding must derive the context causally, not read a stored
   n600 context sidecar.  At each frame, derive it from the previous exact raw
   partition already encoded/decoded and the pinned already-counted pose row.
   Encoder and decoder logit hashes and derived-context hashes must match;
   decoded raw tokens must equal the canonical cache exactly.  This is the
   named decode path used by the legality receipt.  The large prepared context
   is training/debug evidence only and must be explicitly excluded from the
   counted package.
8. Pack the terminal epoch-60 EMA shadow with the real PR130 self-compressor.
   Encode with the real `constriction` Range coder and sparse integer HPAC.
   Retain the Range payload, a byte-identical deterministic repeat, and the
   decoded full raw tensor.  Never emit scalar-only byte measurements.
9. `prepare` is CPU-only and must run in this sandbox.  It must emit a
   `READY_TO_FIRE.json`, `BUILD_RECEIPT.json`, and machine-readable queue order
   under the XI2 SSD root.  Record the real-config memory projection using the
   measured CL1 peak (1,673.391 MiB), full context/tensor sizes, observed free
   storage, and the exact safe-run command.  Do not claim that this is a live
   MPS memory measurement.
10. `finalize` emits the expected-vs-actual byte table and applies the
    preregistered falsifier: the xi Range payload must be strictly below 98% of
    116,716 B, i.e. at most 114,381 B.  Otherwise mark the xi-context
    formulation closed at full scale.  Never transfer XI1's 14.6x ratio.
11. Axis is scorer-free: `[macOS-MPS research-signal training; real Range
    bytes; scorer-free]`; `score_claim=false`, `frontier_moved=false`.

## CPU tests

The targeted pytest must:

- prove the context builder is deterministic and identity-preserving for a
  zero screw on a synthetic 384x512 class partition, including frame-0 zero;
- prove the decoder-side causal builder consumes the supplied previous decoded
  partition rather than a future/ground-truth plane;
- round-trip a synthetic trained-bit-depth EMA checkpoint through real pack and
  unpack on CPU and show exact logits;
- prove parser stages and `--resume-from auto` exist;
- prove the banked-control threshold arithmetic is 114,381 B and the strict
  >2% decision boundary behaves correctly.

Any packed/checkpoint bytes made by tests must be retained beneath the XI2 SSD
root with bytes and SHA-256; do not rely on pytest temporary deletion for those
payloads.

## Acceptance commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q tools/tests/test_run_ddm_xi2_xi_context_full_scale.py
.venv/bin/python -m ruff check tools/run_ddm_xi2_xi_context_full_scale.py tools/tests/test_run_ddm_xi2_xi_context_full_scale.py
.venv/bin/python -m ruff format --check tools/run_ddm_xi2_xi_context_full_scale.py tools/tests/test_run_ddm_xi2_xi_context_full_scale.py
.venv/bin/python -m py_compile tools/run_ddm_xi2_xi_context_full_scale.py tools/tests/test_run_ddm_xi2_xi_context_full_scale.py
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python tools/safe_run.py --rss-mb 2048 --projected-gib 2 \
  --timeout 1800 --label ddm_xi2_prepare --status-receipt \
  /Volumes/APDataStore/pact/ddm_xi2_20260812/run/prepare.safe_run.json -- \
  .venv/bin/python tools/run_ddm_xi2_xi_context_full_scale.py \
  --leg prepare --resume-from auto
```

Also run the strict targeted payload-retention preflight appropriate for these
files.  Do not run `--leg train`, `--leg all`, MPS, Metal, or any scorer.

## Do not touch

- `tools/train_ddm_cl1_hpac_capacity.py`
- `tools/fit_ddm_cl1_hpac_capacity.py`
- `tools/run_ddm_xi1_screw_conditioned_learned_prior.py`
- all pre-existing dirty or untracked files
- the staged index
- the three common-contract protected files

Do not commit.  The parent session owns two review passes, serializer commit,
and final evidence adjudication.
