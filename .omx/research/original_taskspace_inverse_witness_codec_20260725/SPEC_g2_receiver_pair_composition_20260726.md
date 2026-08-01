# G2 receiver and Pair composition specification

## Objective

Close the first causal execution seam in the original task-space inverse codec:
counted bytes for `G` and `A` must be parsed by the receiver and must change the
decoded evaluator-facing raw bytes.  Compose those programs with the existing
`PairPopulationEnvelope` without treating packet presence, sidecars, or caller
attestations as receiver-consumption proof.

This landing is structural and `research_only=true`.  It is not a candidate,
score, frontier, promotion, or originality claim.

## Ownership split

- `lane_g2_receiver_consumption_archive_20260726` owns the strict counted `G`
  receiver path and its causal mutation/counterfactual tests.
- `lane_a2_pair_counted_preimage_adapter_20260726` owns the two-mode `A` to
  counted-Pair adapter and its source-custody/parse-back tests.
- Root owns integration review, composed tests, receipts, review tracking,
  serializer commit, and push.

The units meet at typed bytes and decoded `uint8` pair outputs; neither arm
edits the other arm's owned source file.

## Required invariants

1. The archive has one counted payload grammar.  `G` and `A` are typed sections
   inside it, not extra uncounted ZIP members or sidecars.
2. Strict parse-back rejects truncation, trailing bytes, unknown modes, hash
   mismatch, source mismatch, and non-canonical re-encoding.
3. Receiver order is reverse-causal: establish/verify semantic `Y1`, then
   recover `Y0 | Y1`, then emit the pair through the realized `uint8` path.
4. A valid mutation to counted `G` or `A` bytes must change decoded raw output;
   deletion or corruption must fail closed.  A caller-authored receipt cannot
   satisfy this proof.
5. The matched counterfactual keeps the same runtime and all non-target section
   bytes fixed, so the measured delta isolates the counted section.
6. Any source-bound instance records the exact source artifact path, byte count,
   SHA-256, producer command/config where available, and payload-lineage class.
7. Do not serialize Pose6 or evaluator outputs as causal `A` inputs.  Frozen
   evaluator/scorer truth remains encoder-only and never ships.
8. Preserve deterministic decode, atomic/checkpoint-safe output, and repository
   dependency disclosure.  No heavy, paid, GPU, or authoritative eval launch.
9. Do not touch the operator-owned untracked fire-authority config.

## Acceptance

- Focused tests for both new units pass.
- Existing `generative_taskspace_correction`, `coupled_preimage_program`,
  `pair_population_envelope`, `c0b_counted_receiver_codec`, and stack-receipt
  suites remain green.
- Ruff lint and format checks pass for new/changed Python files.
- At least one deterministic double-decode test and one counted-section
  mutation-causality test pass for each implemented section.
- A durable receipt states exactly what is causal, what is merely structural,
  what source custody exists, and the next blocker to an `n600` archive.

## Explicitly out of scope

- Public archive, checkpoint, latent, selector, or payload reuse.
- HNeRV/PR130 as a vehicle.
- Hardcoded score threshold or stale leaderboard value.
- Claiming an `n1`/synthetic structural fixture as a contest candidate.
- Training before inverse solving and finite program structure are exhausted.
