# Codex findings — DSL compile hash enforcement

## Premise verification

CONFIRMED: the old launcher gate was not airtight. It allowed missing manifests
in migration/dry-run/skip/rationale modes and treated verifier infrastructure
exceptions as warnings. The durable daemon and trainer admitted the marker
`TAC_GOVERNED_ADMISSION=1` without recomputing DSL origin.

## Landing verdict

The launcher and durable governor now independently reconstruct
`TypedWitnessConfig -> WitnessProgram`, recompile exact argv and LawRefs, rebuild
the #332 bijection, and compare them with the exact on-disk launch triple. Both
hand-rule proofs refuse rc 8 before spawn. Native dispatch and trainer `sys.argv`
are bound to the same token. The static Catalog #406 gate is WARN-only at landing;
runtime enforcement is strict.

Bounded calibration and dry-start paths use the identical artifact writer. Their
checkpoint/resume deltas are typed internal Levers, so these valid smoke paths
remain live without recreating a trailing-argv exception. The governor also reads
shell entrypoints before admission and refuses a witness hidden under any script
name other than canonical `launch.sh`.

## Own three-clean-pass review

No code changed between these three concluding passes:

1. Runtime/bypass control flow: 15 targeted launcher/governor/trainer/native/
   internal-path tests passed; manual launcher and governor hand-rule probes both
   returned rc 8; all `write_launch_sh` execution callsites were re-derived.
2. Hash/provenance: 14 determinism/tamper/LawRef tests passed; two run identities
   produced the same hash, a seed change changed it, and exact artifact
   recomputation revalidated 7 LawRef rows plus the #332 snapshot.
3. Structural/triality: Catalog #406 static live count was 0; 42 focused tests,
   `py_compile`, `git diff --check`, and Ruff F/E9/I passed; WARN-only wire-in,
   DAG FEED, and `NO_EQUATION_NEEDED` were present.

## Honest limits

- This is cryptographic integrity/custody, not a signature against a malicious
  repository writer. The structural authority is the reviewed compiler and both
  recomputation callsites.
- The existing broad #332 completeness audit remains WARN-only and reports a
  historical ownership/provenance backlog. Catalog #406 does not falsely claim
  that backlog is drained; it prevents any *post-compile* flag from entering the
  hashed argv and preserves the exact #332 snapshot for review.
- No GPU/training/score run occurred. Pointer UNMOVED.

## Review target for MAIN

Review the volatility boundary, the independent typed-program recompile, and the
lexical-before-Popen governor call. Then independently rerun the actual V9 dry-run
and both rc-8 counterexamples before strict-flipping the static gate.
