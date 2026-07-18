# V10 compiler / receiver structural certificate

Date: 2026-07-18  
Lane: `v10_compiler_receiver_20260718`  
Tasks: #529, #530, #531  
Verdict scope: `V10 COMPILER / RECEIVER / BYTE-CUSTODY STRUCTURAL CERTIFICATE`  
Authority axis: `[local-CPU structural/non-score]`  
Pointer delta: `0`

## Outcome

`src/tac/witness_dsl/v10_compiler_receiver.py` now supplies a deterministic
success path for one typed n600 cold program. The path:

1. accepts only `TypedWitnessConfig`, requires `num_pairs == 600`, requires the
   typed flag `--verdict-pairs 0`, and validates the resulting
   `WitnessProgram`;
2. resolves live LawRefs with `tac.witness_dsl.lawref`, compares their values
   to the typed DSL overrides, and retains a timestamp-free resolved manifest;
3. compiles argv only through
   `WitnessProgram.compile_trainer_argv_with_constants` and round-trips the
   emitted flags through `build_real_trainer_parser().parse_args`;
4. reuses `build_dsl_compile_provenance_document` to verify resolved argv,
   LawRef custody, and the canonical #332 typed self-recompile hash; it then
   records the separate strict repository-wide #332 audit as a Boolean plus
   its concrete violations instead of assuming that audit is clean;
5. emits one `v10_counted_program.v2` binary with fixed magic/version, one
   canonical JSON header, and exactly seven ordered section bodies;
6. reopens that binary, reconstructs typed sections from the bytes, and proves
   exact count/order/length/hash, frozen factor ownership, dependency order,
   parameter custody, and contiguous coverage without trailing bytes; the full
   wire byte count is partitioned exactly into prefix + header + payload, so
   metadata is not treated as free;
7. executes every section through the sealed v2 handler registry and emits one
   `v10_receiver_consumption.v2` receipt per exact byte range; the aggregate
   registry seal binds the source digests of every handler and their shared
   semantic helpers, and the public receiver accepts canonical program bytes
   only and reparses them before use;
8. interrupts between sections, serializes a
   `v10_receiver_checkpoint.v2`, resumes from the exact prefix, and requires
   bit-identical final state; resume first deterministically replays the claimed
   prefix with the active handlers and requires the checkpoint state bytes/hash
   to match, so a caller cannot forge alternate canonical state and recompute
   its hash; resumed results reconstruct and return the full ordered prefix and
   suffix receipt sequence, not suffix receipts alone;
9. validates the eleven-leaf `inverse_solve_completeness_manifest.v2` rows
   separately from compile success and returns `launch_ready=false`.

This is real compiler/receiver behavior, not symbol or path presence. No
`Path.exists`, `hasattr`, import success, contextual blocker count, or hash-only
payload handler can clear a factor.

## Wire and byte custody

Every section declares:

`section_id`, `factor_ids`, `producer_id`, `consumer_id`, `encoding`,
`video_derived`, `byte_length`, `sha256`, `apply_order`,
`owned_parameter_groups`, `frozen_parameter_groups`, `class_ids`, `cell_ids`,
`depends_on`, and `quotient_base_factor_ids`.

The header seals the exact eleven-leaf inventory
`("1","2","3a","3b","4","5","6","7","8","9","10")`, the exact
implemented subset `("1","3a","3b","4","5","6","7","8","9")`, and
the exact missing subset `("2","10")`. Factors `2` and `10` have no section,
no wire range, and no receipt. The parser refuses malformed/noncanonical
headers, route drift, duplicate section/factor/parameter ownership, length/hash
drift, gaps, overlap, truncation, and trailing bytes.

The seven paid routes are frozen as follows:

| order | instruction | factors | byte-custody meaning |
|---:|---|---|---|
| 0 | `CountedGenerator` | `1` | generator inputs and seed bytes are one atomic paid payload |
| 1 | `Frame0PoseSixCarrier` | `7,8` | frame-0 delta and exactly six Pose values |
| 2 | `InitHeadSolve` | `6` | fresh-state head bias; cold only |
| 3 | `SharedResizePreimage` | `3a,3b` | one shared resize/preimage payload and one shared byte range |
| 4 | `RgbYuv6Projection` | `4` | semantic RGB bias plus integer BT.601/YUV6 projection |
| 5 | `BlindFillRateGrammar` | `9` | blind fill and counted rate tokens |
| 6 | `QuotientResidualT` | `5` only | terminal class/cell quotient residual |

Every payload is authoritatively `video_derived=true`; each of the seven
unique ranges is counted and semantically consumed exactly once. The 3a/3b
factor-range records intentionally point to the same range. Aggregate
video-derived bytes equal the sum of the seven payload lengths, while total
program bytes additionally include the fixed prefix and canonical header.

The local semantic receiver applies, in order:

`CountedGenerator -> Frame0PoseSixCarrier -> InitHeadSolve ->
SharedResizePreimage -> RgbYuv6Projection -> BlindFillRateGrammar ->
QuotientResidualT`.

The small state transform is deliberately a structural reference renderer. It
is deterministic and every encoded field has a semantic decode, but it is not
the V10 production renderer, not evaluator evidence, and not promotable.

## Cold birth and quotient residual

`InstructionKind` is the exact typed distinction. Cold compilation requires
`InitHeadSolve` and explicitly refuses the encodings for `ForkHeadSolve`,
`ForkEmaClearance`, and `ResumeLRWarmup`. A defense-in-depth scan also rejects
resume/fork optimizer or EMA state keys. `resume_from` is not admissible.
Every one of the exact seven cold instructions owns exactly one unique
parameter group. Typed seed equality, n600, and full `--verdict-pairs 0`
survive both the DSL emitter and real-parser readback.

`QuotientResidualT` owns factor `5` only. It requires nonempty exact class/cell
IDs, exactly one owned parameter group, every predecessor section as a
dependency, and every predecessor parameter group frozen. Its exact quotient
base is `{1,3a,3b,4,6,7,8,9}`; base drift, predecessor omission, overlap, early
placement, or relearning refuses. The parser re-runs these constraints on the
reopened wire metadata; they are not comments or source-string tests.

This proves unique structural custody. It does not prove that the numerical
`InitHeadSolve` or `T` improves a scorer, converges, or is sufficient for V10.

## Evidence and false-clear closure

Each factor interaction artifact is supplied as bytes and reopened. It binds:

- factor, producer, consumer, config hash, and whole-program hash;
- the exact covered section ID and section SHA-256;
- the SHA-256 of the actual receiver consumption receipt;
- a favorable local structural verdict and explicit non-score axis.

The compiler checks these against `ReceiverResult`, including
`consumption_count == 1`. Caller prose with only a program hash is insufficient.
Empty, stale, adverse, misrouted, wrong-axis, forged-section, or forged-receipt
evidence refuses.

The local compiler refuses every `strict_certificate=COMPLETE` row. Factors
`2` and `10` must remain `MISSING`, use `consumer_id=BLOCKED`, and have no
interaction or measurement receipt. Factors `1` and `5` may be
`FOLDED/PARTIAL` only when their evidence reopens against the actual frozen
receiver receipt. The other implemented factors are `HAVE/PARTIAL`; their
route existence is not measurement evidence.

The #332 surfaces are deliberately separated:

- canonical self-recompile, resolved-argv identity, and `dsl_compile_hash` are
  required for compile success;
- the strict repository-wide bijection audit is reported as
  `dsl_bijection_complete` plus `dsl_bijection_violations` and may honestly be
  false with concrete violations;
- neither outcome grants launch authority. `launch_ready` remains false.

Exact evaluator-axis measurements and production adoption need a separate
reviewed authority gate; they cannot be inferred from this certificate.

## Verification

Commands run from the isolated worktree:

```text
/Users/adpena/Projects/pact/.venv/bin/python -m pytest src/tac/tests/test_v10_compiler_receiver.py -q
# 60 passed

/Users/adpena/Projects/pact/.venv/bin/python -m ruff check --select F,E9,I \
  src/tac/witness_dsl/v10_compiler_receiver.py \
  src/tac/tests/test_v10_compiler_receiver.py
# All checks passed

/Users/adpena/Projects/pact/.venv/bin/python -m py_compile \
  src/tac/witness_dsl/v10_compiler_receiver.py \
  src/tac/tests/test_v10_compiler_receiver.py
# pass

git diff --check
# pass
```

**MEASURED:** `60 passed`; Ruff F/E9/I, `py_compile`, and `git diff --check`
were clean in this worktree. The tests cover the exact seven-route map; the
shared 3a/3b range; atomic generator/seed custody; semantic mutation of every
section; all-program byte custody; raw corruption, truncation, and trailing
bytes; bytes-only public parse-back; exact schema types and registry seals;
custom-handler and exported-registry-rebinding authority refusal; handler and
shared-semantic source-digest custody; counted-but-inert or decoded-frame-inert
section refusal; exact two-frame seed reachability; cold/fork/resume exclusion;
terminal factor-5-only quotient order/base/group freezing; zero and
double-owned `T` residual refusal; cumulative resume receipts, canonical
checkpoint base64, resume equality, and state forgery; clip-not-modulo alias
control;
eleven-row generator preservation; blocked factors 2/10; #332 audit exposure;
and completeness false-clear attempts.

## Round-1 confound findings and corrected resolution

The first review did not pass. Its findings were re-derived into code/tests and
this receipt as follows:

1. **Wrong six-route/factor map:** the earlier draft assigned paid sections to
   factors 2/10 and grouped unrelated factors under init/projection/`T`.
   **Corrected:** the exact seven-route v2 map above is sealed and negatively
   tested; factors 2/10 own no section.
2. **#332 false-pass wording:** self-recompile provenance was being conflated
   with a clean repository-wide bijection. **Corrected:** Boolean and violation
   evidence are exposed separately and cannot authorize launch.
3. **Trusted parsed objects and caller metadata:** a forged parsed object,
   `video_derived=false`, or producer/consumer/factor remap could undermine
   custody. **Corrected:** the public receiver accepts bytes only, reparses the
   wire, and binds exact v2 route and handler registry seals.
4. **Handler self-attestation:** a custom handler could claim consumption.
   **Corrected:** only identity-matched frozen handlers can issue authoritative
   receipts; the custom-handler seam is private, rebinding the exported mapping
   does not change the public authority path, and both cases are refusal-tested.
   The program/receipts bind source digests for each handler and their shared
   semantic helpers. A frozen handler is also refused if its paid section makes
   no semantic state change or leaves both decoded reference frames unchanged.
5. **Completeness/resume/schema confounds:** one-shot row iterables, permissive
   scalar types, object checkpoints, and recomputed forged state could falsely
   clear structure. **Corrected:** inputs are frozen once, schemas/keysets and
   integer types are exact, checkpoints cross the public boundary as canonical
   bytes with canonical base64 spelling, state is authenticated by deterministic
   prefix replay, and resumed results return the full ordered receipt sequence.
6. **Counted-but-unused payload confounds:** an extra generator seed byte could
   be paid without reaching a generated sample, and `T` could pay a zero update
   or two updates for the same frame/index residual. **Corrected:** every seed
   byte must reach at least one of the two generated frames, including the exact
   `N+1` reachability boundary; `T` requires nonzero deltas and exactly one owner
   for each updated frame/index. Pose-six, shared-preimage, and rate-carrier
   fields now have visible decoded-frame effects in the deterministic reference
   path, and signed updates clip rather than alias modulo 256.
7. **Untracked equation and stale DAG hooks:** candidate equations lived under
   temporary storage and stop-hook node numbers did not name the failing stage.
   **Corrected:** the tracked research JSONL and corrected DAG feed are linked
   below.

This is an author Round-1 resolution, **not** an independent fresh-eyes clean
pass and not a final seal. The required independent clean-pass sequence and
MAIN landing review remain outstanding.

## Honest disposition

- Factor 1: `FOLDED / PARTIAL` locally. Deterministic counted-generator bytes
  now have receiver parse-back and resume custody. Production renderer parity,
  byte-close archive custody, and exact scorer evidence remain owed.
- Factor 5: `FOLDED / PARTIAL` locally. A unique class/cell quotient consumer
  now freezes all predecessor groups and is receipt-bound. Scorer efficacy,
  residual-only training behavior on production state, and exact adoption
  remain owed.
- Factors 3a/3b/4/6/7/8/9: structurally routable to their exact named
  consumers (`SharedResizePreimage`, `RgbYuv6Projection`, `InitHeadSolve`,
  `Frame0PoseSixCarrier`, and `BlindFillRateGrammar`), so future
  factor-specific receipts have a receiver surface; none is measured,
  adopted, or complete here.
- Factors 2 and 10: remain literal blockers. No lattice-feasible exact solve or
  byte-closed three-axis residual R-D/KKT authority was produced.

No run was launched, no score was measured, no sacred run or frontier pointer
was touched, and MAIN landing review remains required.

## Stores consulted

- Delegated authority prompt:
  `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/v10_compiler_receiver_20260718_20260718T115615Z.wrapped.prompt.txt`.
- `.omx/research/BUILD_SPEC_v10_compiler_receiver_20260718.md`.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and
  `docs/operating_manual_craft_handoff.md`.
- `.omx/research/build_wave_fresh_eyes_confound_hunt_contract_20260718.md`.
- `.omx/research/frozen_scorer_exact_factorization_20260715.md`.
- `.omx/research/inverse_solve_completeness_matrix_20260718.md`.
- `.omx/research/v10_compiler_receiver_equation_candidates_20260718.jsonl`
  (candidate-only; no equation-registry write).
- V10 spec/review surfaces named by the authority packet, including the
  capstone §14 factorization and true-final review findings; no settled score or
  launch claim was reused as current authority.
- Canonical lane/subagent/inbox surfaces:
  `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`,
  `/Users/adpena/Projects/pact/.omx/tmp/codex_inbox/v10_compiler_receiver_20260718.jsonl`,
  and `/Users/adpena/Projects/pact/.omx/tmp/codex_inbox/_broadcast.jsonl`.
