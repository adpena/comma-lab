# BUILD SPEC — V10 typed compiler and receiver keystone

Date: 2026-07-18  
Lane: `v10_compiler_receiver_20260718`  
Authority: delegated tasks #529, #530, and #531  
Mode: BUILD + local verification only; no launch, no score, no pointer movement  
Verdict scope: `V10 COMPILER / RECEIVER / BYTE-CUSTODY STRUCTURAL CERTIFICATE`

## Outcome

Replace the historical presence-checking V10 skeleton with a real, deterministic success path that compiles one typed cold-start program, resolves its LawRefs, round-trips its argv through the real level-set trainer parser, emits one self-describing counted payload, and parses and executes that payload through a strict receiver. The landing is a receiver/compiler certificate, not a V10 launch certificate or an evaluator score.

The implementation must close the following structural defects without claiming that incomplete scientific factors are measured or adopted:

1. #529: `compile` returns a real result on valid in-memory evidence. It may not use `hasattr`, `Path.exists`, symbol presence, or a contextual blocker count as readiness authority.
2. #530: cold birth has a distinct `InitHeadSolve` instruction. `ForkHeadSolve`, `ForkEmaClearance`, `ResumeLRWarmup`, `--resume-from`, and all fork/resume-only state are rejected by the cold compiler.
3. #531: `T` is an explicit class-/cell-conditioned factor-5 quotient residual, terminal after the exact six paid predecessors and conditioned on base factors `{1,3a,3b,4,6,7,8,9}`. Its trainable parameter group is disjoint from every earlier owner; overlap, duplicate ownership, an early `T`, base drift, or an undeclared cell/class route is a compile refusal.

## Reused canonical surfaces

- `tac.witness_dsl.curriculum_dsl.WitnessProgram`, `build_real_trainer_parser`, and `compile_trainer_argv_with_constants` are the only trainer-argv authority.
- `tac.witness_dsl.typed_config.TypedWitnessConfig`, `typed_config_hash`, and `program_manifest` are the typed-config/hash authority.
- `tac.witness_dsl.lawref` is the only constant compiler.
- Existing exact-length parser conventions and runtime-consumption proof patterns may be reused, but no historical packet grammar is silently declared to be V10.
- The exact frozen-forward factor set is `("1", "2", "3a", "3b", "4", "5", "6", "7", "8", "9", "10")`; a manifest with any other key set refuses.

Avoid edits to the high-blast-radius core files above unless no wrapper can establish the required semantics. Prefer one new V10 module and focused tests.

## Owned files

Primary implementation target:

- `src/tac/witness_dsl/v10_compiler_receiver.py`
- `src/tac/tests/test_v10_compiler_receiver.py`

Durable evidence/triality targets:

- `.omx/research/v10_compiler_receiver_20260718.md`
- `.omx/research/v10_compiler_receiver_DAG_FEED_20260718.md`
- `.omx/research/inverse_solve_completeness_matrix_20260718.md` (factor 1 and factor 5 disposition only, plus the explicit effect on 3a/3b/4/6/7/8/9 certification)
- `.omx/research/v10_compiler_receiver_equation_candidates_20260718.jsonl` as a tracked, candidate-only, non-registry equation surface until real byte-close parity exists.

Do not touch the sacred run directory `experiments/results/levelset_n600_witness_20260717T113932Z`, the pinned upstream snapshot, main, or sibling worktrees.

## One-program wire format and frozen v2 routes

Define `v10_counted_program.v2` with a fixed magic/version, canonical JSON
header, and the following exact seven-section sequence. These routes are the
frozen semantic authority; caller-supplied metadata cannot remap them.

| order | typed instruction | exact factor custody | required meaning |
|---:|---|---|---|
| 0 | `CountedGenerator` | `1` | one atomic generator payload containing both generated-frame inputs and counted seed bytes |
| 1 | `Frame0PoseSixCarrier` | `7,8` | frame-0 freedom plus the exact first-six Pose carrier |
| 2 | `InitHeadSolve` | `6` | cold-only Seg head/winner-cell initialization; never a fork alias |
| 3 | `SharedResizePreimage` | `3a,3b` | one shared `A`/resize-preimage payload and one shared byte range for both leaves |
| 4 | `RgbYuv6Projection` | `4` | RGB adjustment followed by the frozen integer BT.601/YUV6 semantic projection |
| 5 | `BlindFillRateGrammar` | `9` | blind-coordinate fill plus counted rate grammar |
| 6 | `QuotientResidualT` | `5` only | terminal class/cell residual over the exact quotient base `{1,3a,3b,4,6,7,8,9}` |

Factors `2` and `10` have no paid section. They must remain explicit
`MISSING` rows with `consumer_id=BLOCKED`, no interaction receipt, and no
measurement receipt.

Each section must carry at least:

```text
section_id, factor_ids, producer_id, consumer_id, encoding,
video_derived, byte_length, sha256, apply_order,
owned_parameter_groups, frozen_parameter_groups,
class_ids, cell_ids, depends_on, quotient_base_factor_ids
```

The parser must verify canonical header bytes, exact section count and order,
exact lengths and SHA-256 values, contiguous byte ranges, no gaps, no overlap,
no trailing bytes, and no duplicate section/factor/parameter ownership. Every
payload byte belongs to exactly one section. Factors `3a` and `3b` deliberately
resolve to the same `SharedResizePreimage` range; that shared range is counted
once, not duplicated. All seven payloads are authoritatively
`video_derived=true` and are counted exactly once. Factor 1's generator and
seed are indivisible within its single range. Prefix and header bytes remain
part of total program bytes rather than being treated as free metadata.

The receiver must execute every declared section through a named handler and emit a runtime consumption receipt with the exact byte range/hash and `consumption_count == 1`. Unknown encodings, missing handlers, handler non-consumption, double consumption, or an unconsumed section refuse. A deterministic small reference renderer is acceptable for local tests only if it consumes every encoded field semantically and is explicitly labeled non-score/non-promotable; opaque byte hashing without semantic decode is not acceptable.

The compiler result must bind together:

- typed config hash;
- parser-verified trainer argv and argv hash;
- resolved LawRef manifest;
- canonical DSL program manifest;
- payload program bytes/hash/counts;
- parser-level contiguous-byte proof;
- receiver-level consumption proof;
- resume/replay schema and replay result;
- completeness rows and `launch_ready` separately.

`compile` success and `launch_ready` are different states. The local structural fixture compiles successfully while factors 2 and 10 remain honestly non-launch-ready. No incomplete row may be upgraded by inference, and this local compiler always returns `launch_ready=false`.

The canonical #332 self-recompile/hash provenance check is a compile gate. The
separate strict repository-wide flag-to-Lever/runtime audit is evidence, not a
false-pass token: the result must expose both `dsl_bijection_complete` and
`dsl_bijection_violations`, with the Boolean equal to the absence of reported
violations. A false Boolean with concrete debt is a valid non-authorizing local
result; it must never be described as a complete #332 bijection.

## Cold-init and quotient order

The canonical paid receiver order is:

```text
validate typed evidence
  -> CountedGenerator[factor 1: generator + seed atomically]
  -> Frame0PoseSixCarrier[factors 7,8]
  -> InitHeadSolve[factor 6; fresh state only]
  -> SharedResizePreimage[factors 3a,3b; one shared range]
  -> RgbYuv6Projection[factor 4; RGB/BT.601/YUV6]
  -> BlindFillRateGrammar[factor 9]
  -> QuotientResidualT[factor 5 only; terminal]
  -> receiver parse-back
```

The compiler must model `InitHeadSolve` and `ForkHeadSolve` as different instruction kinds even if they share a pure numerical helper. Cold V10 requires `resume_from is None`, fresh optimizer/EMA state, and no fork instruction or fork-only state key. A fork instruction requires explicit resumed-state custody and is outside this cold compiler.

`T` must declare exact class and cell IDs, its unique parameter group, all six
predecessor section dependencies, and all six predecessor groups frozen. Its
sealed quotient base is exactly `{1,3a,3b,4,6,7,8,9}` while its own factor
custody is exactly `{5}`. The compiler refuses early `T`, missing predecessors,
base drift, group overlap, or any attempt to relearn a predecessor. Tests must
exercise the refusal, not search source strings.

## Receipt validation

Evidence is supplied as bytes or reopened and hashed by the compiler. Empty files, schema-only JSON, stale hashes, adverse verdicts, wrong authority axes, wrong producer/consumer IDs, incomplete coverage, or mismatched config/program hashes refuse. `Path.exists` and import/symbol presence may be diagnostic fields only and can never clear a blocker.

Receipt validation must cover the eleven-leaf completeness schema fields already sealed in `inverse_solve_completeness_matrix_20260718.md`. `FOLDED` is admissible only with a real broader consumer and factor-specific interaction receipt; `MISSING` is never launch-ready.

## Resume certificate

The receiver must support an atomic serializable checkpoint between sections containing the program hash, typed-config hash, next section index, already-consumed section IDs, receiver state bytes/hash, and schema version. Resume must refuse program/config drift, duplicate replay, missing state, or a non-prefix consumed set. A stop/resume decode must be byte-identical to uninterrupted decode.

## Required behavior tests

At minimum, land tests that prove:

1. A valid fixture returns a compile result, resolved constants manifest, exact config/argv hashes, and argv accepted by `build_real_trainer_parser().parse_args`.
2. The success path emits exactly one payload program and receiver parse-back is deterministic.
3. Status cannot be clear while a post-gate fold or semantic receipt is absent; empty/stale/adverse receipts refuse.
4. Cold compilation accepts `InitHeadSolve` and rejects `ForkHeadSolve`, `ForkEmaClearance`, `ResumeLRWarmup`, `resume_from`, or fork state.
5. `T` is terminal, factor-5-only, class-/cell-conditioned, sealed to the exact eight-factor quotient base, and has a disjoint unique trainable group; order/base/overlap/duplicate-owner counterexamples refuse.
6. Every section byte range is contiguous, disjoint, hashed, counted, and consumed exactly once; aggregate video-derived bytes equal the sum of the declared video-derived sections.
7. Recompile after mutating one valid payload byte changes the decoded frame/state while non-target section identities remain stable. Raw in-place corruption without a matching manifest hash refuses.
8. Truncating at every structural boundary refuses; appending a byte refuses.
9. Interrupted decode plus resume is byte-identical to uninterrupted decode; state/program/config drift refuses.
10. False route maps, factorless/extra sections, duplicate factor ownership, a missing or custom handler, an unknown encoding, and trailing/unconsumed bytes refuse.
11. Golden-vector compilation is byte-identical across repeated runs and changes when a typed seed/config value changes.
12. Factors 1 and 5 receive honest local `FOLDED/PARTIAL` dispositions; factors 3a/3b/4/6/7/8/9 gain only a structural consumer route and are not called measured/complete; factors 2 and 10 remain literal `MISSING/BLOCKED` rows with no section.

The named positive control for the confound hunt is the payload-byte mutation: a semantically consumed byte must alter decoded output. The named negative controls are raw corruption, truncation, trailing data, duplicate ownership, cold/fork mixing, `T` parameter-group overlap, a counted section with zero semantic or decoded-frame delta, seed bytes that reach neither generated frame, a paid zero `T` residual, two `T` updates claiming the same frame/index residual, exported registry rebinding, suffix-only resume receipts, and noncanonical checkpoint base64.

## Verification and review

Run focused tests, Ruff F/E9/I on owned Python files, `py_compile`, `git diff --check`, and deterministic repeat/golden checks. Perform Round-1 author self-review against false-clear readiness, counted-but-inert bytes, mutation-hash-only behavior, cold/fork aliasing, `T` metadata-without-enforcement, resume double-consumption, wrong factor custody, #332 false-pass language, and factor-status overclaim.

Current local evidence is **MEASURED**: the focused suite reports `60 passed`;
Ruff F/E9/I, `py_compile`, and `git diff --check` are clean. This measurement
does not constitute an independent fresh-eyes clean pass.

Then obtain three consecutive independent fresh-eyes clean passes under `.omx/research/build_wave_fresh_eyes_confound_hunt_contract_20260718.md`; any finding resets the counter. Commit only through `tools/subagent_commit_serializer.py` with a post-edit `--expected-content-sha256` for every Python file. Final verdict must state pointer delta `0`, no launch/score, and require MAIN landing review.
