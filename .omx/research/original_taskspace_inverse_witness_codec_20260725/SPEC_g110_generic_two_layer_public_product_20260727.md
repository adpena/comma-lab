# G110 — generic final-Y1 provider × conditional-Y0 public product

Status: receiver and counted wire implemented; producer compilation fails
closed pending a materialized G109 aggregate paired to a fresh G105 semantic
packet plus a fresh conditional-operand receipt.

Authority: receiver/runtime/parse-back implementation only. No trained
candidate, exact evaluator row, score, promotion, or pointer movement.

## Why this is the missing composition layer

G107 represented the intended conditional equation but hard-cast its semantic
base to G103, depended on an untracked outer archive helper, required every
pair to have a nonzero residual, stored two scales whose product was the only
decoder-effective quantity, and left coefficients as a raw int16 table.
Consequently it could not compose with the exact V9 program and it charged a
gauge rather than a quotient.

G110 makes the product boundary the rendered final Y1 population:

1. self-describing semantic bytes select exactly one provider;
2. both G103 and exact G105 V9 implement the same public
   `accepts_packet/parse_packet/render_scorer_y1` ABI;
3. the receiver hashes all 600 rendered Y1 planes before publishing output;
4. conditional Y0 consumes that rendered uint8 Y1, never a semantic class;
5. the conditional quotient stores one combined per-rank scale;
6. chronological int16 coefficient differences use canonical minimal-k Rice;
7. a zero pair row is legal and reproduces Y1 exactly; unused whole ranks are
   rejected;
8. both scorer planes pass the exact public V10 factor-2 realization.

This is a type-level composition fix, not another local representation arm.

## Equations

For semantic provider `q` and pair `p`:

`Y1_p = q.render(P_sem, p)`.

The final-Y1 custody coordinate is:

`H_Y1 = SHA256(domain || SHA256(P_sem) || SHA256(concat_p(u16be(p), Y1_p)))`.

The conditional quotient is:

`R_p = upsample(sum_r c[p,r] s[r] B[r])`

`Y0_p = clip_uint8(round(Y1_p + R_p))`.

Only `s[r]` is stored. Any factorization
`s[r] = s_basis[r] s_coefficient[r]` is the eliminated gauge.

Chronological coefficients use
`delta[p,r] = c[p,r] - c[p-1,r]`, zig-zag mapping, and Rice parameter

`k* = argmin_(0<=k<=15) sum_(p,r)(floor(z(delta[p,r])/2^k) + 1 + k)`.

The outer ZIP contains exactly one counted member,
`taskspace_two_layer_v1.bin`.

## Public receiver closure

`submissions/robust_current/g110_two_layer_receiver` is standalone NumPy plus
Python standard library:

- `inflate.sh` invokes only its sibling `inflate.py`;
- semantic plugins contain the committed G103 receiver and receiver-only G105
  V9 HOSC implementation, with no repository/canonical-loader import;
- the conditional plugin owns strict envelope/Rice parsing, Y0 reconstruction,
  and the final-Y1 population-binding check;
- repo parse-back refuses oversized compressed members before decompression,
  and producer-custody files are rejected as symlinks before resolution;
- `inflate.py` keeps output under a temporary name until all 600 Y1 frames
  close the packet binding, realizes Y0 and Y1 independently through exact
  factor-2, checks exact raw length, fsyncs, then atomically publishes.

The runtime contains no scorer, source, target table, or video-derived value.
All learned semantic and conditional values are counted packet sections.

## Producer custody and fail-closed boundary

The repo-side compiler cannot construct custody from hashes. It content-reads:

1. G109 `tac.taskspace_v9_training_target_capsule_aggregate.v1` through its
   strict loader, including all 600 labels/margins/PoseNet-6 targets, every
   batch checkpoint, recursive G46 source custody, scorer files, and the
   portable upstream closure;
2. an exact `CompiledFreshV9SemanticRootV1`, the physical fresh checkpoint,
   its packet/wire identity, same-forward batch-16 margin binding, live
   verdict batch 16, and portable source closure;
3. a self-hashed
   `tac.g110_fresh_conditional_y0_operand_receipt.v1` whose array hashes bind
   the exact basis, combined scales, coefficient matrix, semantic packet,
   G109 receipt, PoseNet target array, joint-pose conditioning, and fresh
   own-lineage claim.

G109 is committed at `1689d2025d` and its strict preflight is green. No
materialized G109 aggregate paired to a fresh G105 semantic packet has been
supplied to this compiler, and no conditional-operand producer writes the
third receipt yet. Therefore candidate compilation is deliberately blocked.
The lower-level receiver packet builder is exercised only by tests and carries
no source/candidate claim.

## Canonical-vs-unique decision per layer

- Commit serializer, review tracker, G109 strict loader, and G105 packet parser:
  adopt canonical because their content-reading custody and wire identities
  fit this product exactly.
- Outer archive and conditional wire: unique G110 implementation because G107
  depended on an untracked helper and represented a redundant scale gauge.
- Public runtime: unique complete G110 tree so recursive receiver closure does
  not rely on repository imports or mutate G108 historical evidence.
- Semantic backends: preserve each architecture-specific implementation behind
  one behavioral ABI; do not reinterpret V9 as G103.

## Triality and wire-in

- DSL: the typed G110 envelope, provider dispatcher, source/pose custody, and
  conditional-operand receipt gate.
- DAG: G103/G105 semantic packet → final-Y1 population binding → conditional
  Rice quotient → two scorer planes → V10 factor-2 → public raw.
- Equations: the quotient and temporal code equations above.

Sensitivity-map/bit-allocation contribution: the packet exposes exact semantic,
basis, scale, and Rice-stream bytes for whole-archive arbitration. Pareto
constraint: only joint `100*d_seg + sqrt(10*d_pose) + rate` exact rows may
promote it. Autopilot/continual-learning hooks remain blocked until the first
receiver-closed exact row exists; registering proxy effects would create
false authority. The G109 and conditional-receipt failures are the explicit
probe-disambiguators.

## Stores consulted

- `CLAUDE.md` / `AGENTS.md` and `PROGRAM.md`;
- current lane registry and subagent progress;
- G103 wire and committed G108 public runtime;
- frozen G105 exact V9 adapter and tests;
- frozen G107 conditional product and tests;
- committed G109 aggregate loader/spec at `1689d2025d`;
- V10 factor-2 realization implementation.

## Verification after G109 rebase

- Ruff: clean on all G110 Python files.
- `py_compile`: clean on all G110 Python files.
- focused suite: 11 passed in 32.03s on HEAD `1689d2025d`.
- public G103 frames match the committed repo receiver at pairs 0, 137, 599.
- public V9 frames match G105 at pairs 0, 17, 599.
- public conditional decode matches repo decode, including a legal zero row.
- both scorer planes pass exact V10 factor-2 verification.
- ZIP parse-back is deterministic and one-member.
- oversized compressed archive members and custody-file symlinks fail closed.

Pointer delta: zero. The effective frontier remains an external target, not a
G110 result.
