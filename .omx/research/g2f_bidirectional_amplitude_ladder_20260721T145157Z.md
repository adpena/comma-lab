# Task #578 G2f bidirectional amplitude ladder

`lane_id=lane_g2f_bidirectional_amplitude_ladder_578_20260721` ·
`verdict=MEASURED_G2F_BIDIRECTIONAL_QP_NO_ADMISSION_N64_FAMILY_OPEN` ·
`research_only=true` · `[macOS-CPU advisory]` · `score_claim=false` ·
`promotion_eligible=false` · `pointer=0.1910828242 [contest-CPU] UNMOVED` ·
`MAIN_REVIEW_REQUIRED=true`

## Outcome first

The G2e single-amplitude diagnosis was directionally correct but incomplete.
The paired response curve found a real quantization knee at amplitude `1.0`:
`209/462` class/bucket trust regions and `76/254` effective pair-directions
were usable at that rung. Across all rungs, `582/2772` trust regions were
usable. The local response family is therefore not empty.

The receiver-closed endpoint still did not admit a correction. Only `5/64`
pairs had a usable rung for every effective direction, and all five resulting
active-set QPs were infeasible. The other `59/64` pairs were refused before
the QP. Correction bytes remained zero and D4 did not run.

This is an exact contiguous n64, openpilot-base, paired-amplitude result for
the measured local-linear chart formulation. It is not a negative verdict on
the exact fp32 rank-4 quotient geometry, direct integer-lattice/NFS-style
preimages, or the broader predict-project family. The n600 continuation was
deliberately stopped: n64 exercised the D3 endpoint and produced zero receiver
admissions, so the delegated gates did not warrant another roughly 536 pairs.

## D1 — amplitude-response curve

The ladder was derived rather than guessed. The exact factor-2 receiver has
induced L-infinity gain `1.0`, so the sub-LSB boundary is
`1/(2 ||R||_inf) = 0.5`. The upper bracket is one dyadic octave above the
largest G2e prior amplitude (`8.0`), yielding
`[0.5, 1, 2, 4, 8, 16]`.

MEASURED n64 custody:

- `1524` paired direction/rung observations and `3048` signed branches;
- `2772` class/bucket trust regions, `582` usable;
- `64/64` inherited G2e branches consumed exactly once and not remeasured;
- `8076` per-write responses, all in the measured `nonpositive` margin bucket
  and `boundary_codim1` stratum;
- target-class 0: `1788/3624` sign-consistent responses (`0.4933774834`);
- target-class 1: `2354/4452` sign-consistent responses (`0.5287511231`).

| amplitude | usable pair-directions | usable trust regions | negative / positive uint8 saturations |
|---:|---:|---:|---:|
| 0.5 | 0 / 254 | 0 / 462 | 0 / 0 |
| 1.0 | 76 / 254 | 209 / 462 | 142 / 236 |
| 2.0 | 61 / 254 | 186 / 462 | 205 / 303 |
| 4.0 | 25 / 254 | 101 / 462 | 279 / 381 |
| 8.0 | 11 / 254 | 58 / 462 | 401 / 485 |
| 16.0 | 4 / 254 | 28 / 462 | 518 / 635 |

Refusal counts are nonexclusive: `1872` sign/zero failures, `1549` relative
secant-residual failures, and `508` zero-applied-RGB-branch failures. The
factor-of-two decay after amplitude `1.0`, together with rising saturation,
places the measured knee at the first whole-LSB rung.

## D2 — rebuilt trust regions

MEASURED:

- all-directions-usable pairs: `5/64`;
- failed trust regions: `2190/2772`;
- residual-associated failures: `1549`;
- sign-associated failures: `1872`;
- saturation-associated failures: `1502`.

The G2e blocker `TRUST_REGION_EMPTY_N16` is superseded by the narrower n64
endpoint blocker
`R1B2_RANK4_BIDIRECTIONAL_RECEIVER_QP_NO_ADMISSION_N64_OPENPILOT`.
Bidirectional amplitude selection repaired enough local rows to reach the QP;
it did not produce a jointly feasible receiver correction.

## D3 — receiver-closed QP and admission

The deterministic complete active-set solver fired only after every effective
direction on a pair had a usable rung. Its measured status histogram was:

- `TRUST_REGION_REFUSED`: `59` pairs;
- `QP_INFEASIBLE`: `5` pairs;
- receiver-closed admissions: `0` pairs.

No tested correction was promoted to an admitted #557 packet. The counted
base stayed at `121,128` bytes, correction bytes stayed at `0`, and the
`+95,094`-byte target-box headroom was not spent. The registered marginal
break-even remained `25/37,545,489` score units per correction byte; no
zero-byte or empty-packet row was misreported as positive marginal value.

The unmodified `predict_project_realization_admissibility_v1` returned false.
It passed exact factor-2 uint8 realization and double-decode identity, and
failed exactly `n600`, `semantic_cells_to_rgb_exact`,
`pose_within_declared_tube`, `zero_added_seed_bytes`, and
`receiver_derived_rgb`. The openpilot base carries `42,159` counted bytes and
is not a zero-byte receiver-derived semantic decoder.

An append-only empirical anchor was registered through the canonical helper:
`realization_g2f_bidirectional_amplitude_n64_20260721`. The anchor cites the
full receipt, gate predicates, knee, trust counts, QP histogram, and explicit
reactivation criteria; it changes no equation or threshold.

## D4 — openpilot-base delta

`NOT_RUN_D3_NO_ADMISSION`. Admitted pairs, correction bytes, and semantic
score-unit delta are all zero. Nothing was routed to #598 r5, and no score or
pointer claim was made.

## Reformulation queue and stop rule

The cheapest next discriminating probe is n16 direct integer-lattice/NFS-style
exact preimage enumeration for the five chart states that reached a QP at
n64, compared against the preserved bidirectional local-linear rows. It must
retain the same receiver parse-back, hard Seg/Pose oracle, per-pair byte
custody, and rate threshold. Only a receiver-closed positive n16 admission can
authorize n64; only a surviving n64 admission can justify n600.

This is a reformulation queue, not a GO and not a family closure.

## REUSE MANIFEST

| Required surface | Disposition | Evidence |
|---|---|---|
| `src/tac/optimization/realized_secant_custody.py` | EXTENDED IN PLACE | typed bidirectional branches, odd/even pairing, rung trust/selection, strict receipt rederivation |
| `tools/measure_realization_g2_lattice.py` | EXTENDED IN PLACE | resumable n16/n64 runner, candidate-state measurements, receiver QP, hard oracle |
| G2e final receipt | CONSUMED, NOT REMEASURED | all 64 prior branch hashes preserved; file SHA `e89157bb8dfc6b11b20aecccd4dbe82113ea706c9a1eb054de989c26a740dbc4` |
| `predict_project_realization_admissibility_v1` | REUSED UNMODIFIED | false predicate table plus append-only empirical anchor |
| #583 rank-4 prototype bank | REUSED | prototype SHA `5ce0458949acb1cde21022aef7bf642b4491ddb03d1ab66838866b20cb7b162f` |
| #580 full-kernel projector | REUSED | exact R gain and factor-2 uint8 proof in every pair stage |
| #557 correction codec | REUSED | canonical parse/re-encode path; no admitted nonempty packet |
| new measurement CLI / solver fork | NOT CREATED | existing runner and custody module were the correct extension points |

## Custody and verification

Full external receipt:
`/Volumes/VertigoDataTier/pact/evidence/g2f_amplitude_20260721/receipt.json`

- file SHA-256:
  `0a09b7b5022ff64eebc54d086f00c89378d7eb7091c5963cf1056120469bc38e`;
- embedded canonical receipt SHA-256:
  `3ddd1a51b2e238fe9f20e85f0f6b293df5cbbffa5f2007eb71e85cedecfc9ce1`;
- config SHA-256:
  `a8a2393e266da0aa629898b3d935e70b678244e1c48940af109c428906ee2a3e`;
- receipt bytes: `84,257,468`;
- external tree: `215` files, `253,025,211` bytes, sorted
  `sha256 + relative-path` manifest hash
  `02483759c0b70f9a36f70592a7b45e848a5ac99370f0eaaf5e2105d95aaca364`;
- immutable stages: `64`; prefix checkpoint: `prefix_n64.json`;
- focused verification: `34 passed`; Ruff check and format check, Python
  compile, strict receipt validation, and `git diff --check` passed.

## STORES CONSULTED

Delegated checkpoint key and authority file; `CLAUDE.md`; `AGENTS.md`; n16
checkpoint; both delegated inboxes; operator Fisher/margin,
inner-Jacobian/secant/QP, reverse-waterfill, curvelet/shearlet, and xi
directives; G2e receipt/stages/findings; #583 prototype receipt; #580
projector; #557 codec; seed, n600 GT cache, frozen scorer weights, openpilot
base charts; lane/progress state; canonical equation registry; preserved n16
through n64 G2f stages and chunk/prefix checkpoints.

## Triality and landing boundary

- DSL/code: typed bidirectional response/trust/receipt contracts and the
  resumable measurement runner extend the existing realization path.
- DAG: `g2f_bidirectional_amplitude_ladder_DAG_FEED_20260721T145157Z.md`
  routes the measured knee and QP infeasibility to the six consumers.
- Equations: `predict_project_realization_admissibility_v1` received one
  append-only empirical anchor; its callable and thresholds are unchanged.

MAIN must review the branch diff, canonical registry append, external receipt
custody, and formulation scope before merging. Until that merge-boundary
review, this branch is not repository truth.
