# DAG FEED — V10 typed compiler / receiver

Date: 2026-07-18  
Lane: `v10_compiler_receiver_20260718`  
Research-only: `true`  
Authority: `[local-CPU structural/non-score]`

## Nodes

```text
N0 TypedWitnessConfig[n600, verdict_pairs=0, cold]
N1 WitnessProgram.validate
N2 LawRef resolve + typed-value identity
N3 compile_trainer_argv_with_constants
N4 real trainer argparse parse-back
N5 #332 DSL self-recompile / resolved-argv hash provenance gate
N6 canonical seven-section V10 v2 binary program
N7 bytes-only reopen + source-bound route/handler seals + contiguous ownership proof
N8 frozen semantic receiver + decoded-frame-changing exactly-once receipts
N9 canonical prefix-checkpoint bytes
N10 deterministic prefix replay authentication + resumed output
N11 eleven-leaf completeness rows, including BLOCKED factors 2 and 10
N12 strict repository-wide #332 audit evidence [bool + violations]
N13 compile_success=true
N14 launch_ready=false
```

## Edges

```text
N0 -> N1 -> N2 -> N3 -> N4 -> N5 -> N6 -> N7 -> N8
N8 -> N9 -> N10
N7 -> N11
N8 -> N11
N10 -> N11
N11 -> N12 -> N13
N11 -> N14
N12 -> N14
```

`N13` and `N14` are deliberately different outputs. `N12` may report
`dsl_bijection_complete=false` together with concrete violations; that debt is
preserved rather than mislabeled as a complete #332 audit. The local compiler
can establish a valid deterministic structural program without authorizing
launch, and `N14` remains false.

## Receiver section order

```text
S0 CountedGenerator          owns factor 1 atomically [generator + seed]
S1 Frame0PoseSixCarrier      owns factors 7,8
S2 InitHeadSolve             owns factor 6 [cold only]
S3 SharedResizePreimage      owns factors 3a,3b [one shared byte range]
S4 RgbYuv6Projection        owns factor 4 [RGB -> integer BT.601/YUV6]
S5 BlindFillRateGrammar      owns factor 9
S6 QuotientResidualT         owns factor 5 only [terminal]
```

All seven payloads are `video_derived=true`, form disjoint contiguous ranges,
and are counted and consumed once. Factor-range entries `3a` and `3b` both
point to `S3`; they do not duplicate its bytes. Factors `2` and `10` have no
section and remain `MISSING/BLOCKED` at `N11`.

`S6.depends_on == (S0,S1,S2,S3,S4,S5)`. Its unique parameter group is disjoint
from all predecessor owners, its frozen set equals the complete predecessor
group union, and its exact quotient base is
`{1,3a,3b,4,6,7,8,9}`. `ForkHeadSolve`, `ForkEmaClearance`, and
`ResumeLRWarmup` are exact typed instruction exclusions in this cold graph;
non-null `resume_from` and fork/resume state tokens also refuse.

## Stop hooks

- Any cold-state, typed-config, LawRef, canonical self-recompile, resolved-argv,
  or real-parser mismatch stops at its producing node `N0`–`N5`, before binary
  emission.
- Any false route map, missing/extra section, wrong factor/base/dependency/group
  custody, or non-video-derived payload stops at `N6`.
- Any noncanonical header, registry-seal drift, bad range/hash, malformed type,
  gap, overlap, truncation, or trailing byte stops at `N7`.
- Any unknown/missing/custom or exported-registry-rebound handler, semantic
  source-seal drift, semantic decode failure, payload-hash drift, consumption
  count other than one, counted-but-unchanged state or decoded frames, unused
  generator seed byte, paid zero `T` residual, or double-owned `T` frame/index
  residual stops at `N8`.
- Any checkpoint schema, keyset, type, state-hash, noncanonical base64, or
  canonical-byte failure stops at `N9`.
- Any non-prefix, completed-prefix replay, missing state, suffix-only receipt
  custody, or program/config drift stops at `N10`.
- Any missing folded interaction receipt, forged section/receiver binding,
  adverse axis/verdict, false-clear factor 2/10 row, or `COMPLETE` local claim
  stops at `N11`.
- Strict #332 debt is preserved at `N12` as `false + violations`; it cannot turn
  `N14` true.

## Triality

- DSL/code: `src/tac/witness_dsl/v10_compiler_receiver.py`.
- DAG: this file.
- Equation candidate:
  `.omx/research/v10_compiler_receiver_equation_candidates_20260718.jsonl`.

No equation registry was changed. No launcher/autopilot/bit-allocator hook is
activated because this is a receiver/compiler structural certificate with
`research_only=true`, `score_claim=false`, and `launch_ready=false`.

Round-1 corrected the earlier wrong route map, false-pass #332 interpretation,
and off-by-one stop-hook references. This DAG is not an independent fresh-eyes
clean-pass receipt; MAIN landing review remains required.
