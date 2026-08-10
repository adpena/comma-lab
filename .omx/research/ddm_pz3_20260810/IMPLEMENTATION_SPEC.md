# PZ3 target-packet receiver implementation spec

## Objective

Build and byte-close a real PR130 receiver variant that consumes the retained 2,860-byte PZ2
target packet, predicts PR130's twelve deployed carrier coefficients deterministically, and
stores a counted residual sufficient to reconstruct the incumbent carrier exactly. Measure the
actual archive bytes and prove the receiver path through parse-back and rendered-frame identity.

## Constraints

- Preserve the legacy CPR1 receiver byte-for-byte and add a fail-closed PZ3R dispatch.
- The PZ2 packet must be causally consumed. Mutating a decoded target code must either change the
  coefficient prediction or fail the residual integrity check; decorative packet carriage is
  forbidden.
- The public receiver may use NumPy, PyTorch, Brotli, and the existing PR130 codec. It must not
  load PoseNet, SegNet, scorer weights, ground-truth caches, or any uncounted video-derived table.
- Store the exact incumbent basis component, the exact PZ2 packet, fixed-point predictor state,
  and exact coefficient residual inside the counted carrier section.
- Use only integer fixed-point prediction at decode time. All fitted video-derived values are
  counted in the archive.
- Materialize and retain every candidate packet, archive, repeat archive, parse-back array, and
  receipt under `/Volumes/VertigoDataTier/pact/ddm_pz3_20260810/retained/`, with bytes and SHA-256.
- Runs are deterministic, stage-checkpointed, atomic, and resumable from disk. No scorer launch:
  this arm does not own the scorer slot.
- Do not edit `upstream/`, the PR130 intake clone, protected files named by the common contract,
  the shared staged index, or unrelated dirty files.

## Files and areas

- Add `src/tac/pr130_runtime/fx1_runtime_tree/pose_target_receiver.py` for the PZ2 decoder and
  PZ3R carrier codec.
- Update `src/tac/pr130_runtime/fx1_runtime_tree/inflate.py` only to dispatch PZ3R to the new
  decoder; CPR1 behavior must remain unchanged.
- Add `src/tac/pr130_runtime/tests/test_pose_target_receiver.py` for format, causality, corruption,
  and deterministic decode tests.
- Add `experiments/ddm_pz3_pose_receiver_realization.py` for retained construction, archive
  byte-closure, parse-back, selected-frame identity, score-component recomputation, and receipts.
- Add final research receipts only after the run finishes.

## Predictor and packet

For each coefficient dimension, fit one or more bounded fixed-point linear predictors over the
six integer PZ2 target-code streams and optionally the already decoded previous coefficient row.
Serialize signed integer weights, shift, and intercept. Predict with integer dot products and a
specified ties-away-from-zero division. Store modular signed coefficient residuals and entropy-code
them with the existing deterministic Rice primitives. The decoder reconstructs exact incumbent
absolute int12 coefficient codes sequentially and verifies a stored SHA-256 of the reconstructed
arrays. The selected candidate is the smallest real `archive.zip`, not the smallest inner stream.

## Acceptance criteria

- `python -m pytest -q src/tac/pr130_runtime/tests/test_pose_target_receiver.py`
- Legacy CPR1 parse-back arrays equal the base arrays exactly.
- Every PZ3R candidate encodes twice to identical bytes and decodes to exact incumbent basis and
  coefficient arrays.
- A target-code mutation changes prediction or triggers integrity failure.
- The selected PZ3R archive and repeat archive are byte-identical, have recorded SHA-256/bytes,
  and reparse through the updated real inflate module.
- On the pinned stratified n=120 indices, PZ3R and CPR1 slave-frame tensors are byte-identical;
  the incumbent measured advisory d_pose may then be inherited with an explicit same-object proof.
- The result recomputes full precision score terms from pinned contest-CUDA base components plus
  the measured archive byte action, labels this as derived/non-authority, and applies the charter's
  REALIZATION-LIMITED falsifier.
- Two review-tracker passes approve every Python file before serializer commit.
