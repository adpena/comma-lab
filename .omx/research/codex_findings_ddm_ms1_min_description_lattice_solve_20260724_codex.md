# Codex findings — DDM MS1 minimum-description lattice solve

Date: 2026-07-24
Lane: `lane_ddm_ms1_min_description_lattice_solve_20260723`
Authority: `[macOS-CPU frozen-scorer advisory]`
`research_only=true` · `execution_allowed=false` · `score_claim=false` ·
pointer `0.1910828242 [contest-CPU]` unchanged · MAIN landing review required.

## Result

The full resumable n600 diagnostic completed. Previous-frame conditional coding
reduced the unchanged exact-solve member from `744,608,961` to `731,622,325`
bytes, a measured saving of `12,986,636` bytes (`1.744088%`). It remains
`4,734.684x` the `154,524`-byte target. The pose-xi conditional form worsened
the total to `757,559,811` bytes.

Every saturated local closest-vector proposal lost its real zlib race:

| Diagnostic conditioning expansion | Proposals | Wins | Changed pairs | Canonical conditional stage sum | Proposed stage sum |
|---|---:|---:|---:|---:|---:|
| Previous frame | 1,200 | 0 | 0 | 731,362,522 B | 1,063,499,710 B |
| Counted pose6 to xi | 1,200 | 0 | 0 | 757,236,164 B | 1,238,557,814 B |

The precise verdict is
`FULL_N600_LOCAL_CVP_MEMBER_SELECTION_NEGATIVE_CONDITIONAL_CODER_ONLY`.
It closes only this saturated-local-CVP proposal and diagnostic-zlib instance.
It does not close global lattice search, exact sieve/branch-and-bound, a joint
Seg-cell/Pose-tube tolerance solve, the own-lineage stored-problem family, a
receiver, or the paradigm.

## Campaign headline is withheld

The campaign quantity is

```text
(stored problem bytes + solve-exception bytes, realized d_seg, realized d_pose).
```

The v1 run has no counted own-lineage stored problem, no SHA-bound
receiver-decodable exception stream, no receiver-closed deterministic expansion,
and no Pose tube active inside member selection. The executable headline
contract therefore emits these exact blockers:

1. `OWN_LINEAGE_STORED_PROBLEM_NOT_PROVEN`
2. `STORED_PROBLEM_BYTE_CUSTODY_MISSING`
3. `SOLVE_EXCEPTION_BYTE_CUSTODY_MISSING`
4. `STORED_PROBLEM_EXPANSION_NOT_RECEIVER_CLOSED`
5. `POSE_TUBE_NOT_ACTIVE_IN_SOLVE`

The final batch-32 frozen CPU-Torch oracle is still useful diagnostic evidence:
all `600/600` Seg argmax pairs and all `600/600` Pose6 outputs are identical to
the unchanged canonical member, with `d_seg=0.0001519690619574653` and
`d_pose=0.00010184327939026322`. It proves realized survival of the unchanged
selection; it does not retroactively make Pose an active solve constraint.

The historical immutable v1 receipt uses `origin` in old field names. New
authority surfaces translate those rows to diagnostic conditioning expansions.
No donor archive or PR-lineage spine is admitted. M7 is retained only as an
existence observation that enough conditioning can close both distortions; it
supplies no MS1 bytes, expansion, or pattern.

## What changed in the apparatus

- `MdlPolytopeMemberSolver` now constructs a saturated primitive
  `ker_Z(A)` basis for every 2x2 resize row, performs deterministic exact size
  reduction, proposes integer closest vectors, and admits only strict real
  modular-zlib improvements with exact parse-back.
- A 24,576-by-16 geometry-only facet lookup is cached once per solver. The
  retired path cost `2.6080500420648605 s/pair` locally, or roughly 26 minutes
  over n600, while adding no information.
- Typed SENSE rows expose Fisher winner-rival active sets per class and stratum,
  exact rate, local facet degeneracy, and honest unavailable duals. The run
  produced 600 SHA-bound rows.
- Six numerical SVD factors clear the measured one-byte coder floor. Zero are
  distilled: the required per-stratum coder races were not measured. A strict
  winner routes `SKELETON` to pf1/g1 token coders or `FIBER` to
  transform/quantize/entropy plus amplitude-law coding; ties and missing races
  remain blocked.
- `build_minimum_description_headline` is a fail-closed campaign firewall. It
  refuses incomplete bytes/SHA, donor conditioning, missing own-lineage
  expansion closure, inactive Pose tubes, or unrealized scorer acceptance.

The lane's three-clean gate is registered and its evidence path exists. The
repository-wide `lane_maturity.py validate` remains non-clean with exactly 110
pre-existing missing-evidence-path errors in unrelated historical lanes. No
MS1 inconsistency appears in that output, and this lane does not claim a global
registry seal.

## Custody

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Immutable v1 n600 receipt | 8,624 | `546a7fddb0225edb15b2254ab73e362758b7b0f244e4ff39cb7bfef25f779098` |
| Pair SENSE JSONL | 1,170,365 | `276dde04cc0d6f4f4df1bfb1c7544f997800da189d49e789d00f87e699073803` |
| Historical factorization | 5,232 | `1c798be26b6e8aeb4b259d9e56beedd0cd99f5e5d6b5c2c6ba59f1a0ee03b450` |

The SSD evidence root is
`/Volumes/VertigoDataTier/pact/evidence/ddm_ms1_min_description_lattice_solve_20260723_final`.
All 600 atomic pair stages are preserved; zero changed-payload files exist
because zero proposals won. Source bytes were never copied, moved, or deleted.

## Triality

- **Equations:** `ddm_ms1_min_description_lattice_solve_canonical_equations_20260724.md`
  owns the campaign total, gauge condition, saturated integer kernel, joint
  constraints, factor arbitration, and headline admission law.
- **DAG:** `ddm_ms1_min_description_lattice_solve_DAG_FEED_20260724.md` owns the
  own-lineage stored-problem to deterministic-expansion to joint-solve to
  headline graph, with the v1 diagnostic branch kept separate.
- **DSL:** N/A with rationale. This lane adds no trainer, curriculum, submission,
  or launch lever; inventing a flag would create false executable state.

## Stores consulted

- Authority prompt, SHA-verified before work.
- Checkpoint ledger for
  `codex_delegate:ddm_ms1_min_description_lattice_solve:20260723T233549Z`.
- `CLAUDE.md`, `AGENTS.md`, top-ten Claude memory, current canonical frontier,
  lane registry, subagent-progress ownership, predecessor #547/#549/#602
  artifacts, and operator inbox through `2026-07-24T01:06:43Z`.
- SHA-pinned raw witness, n600 targets, 600 stage receipts, frozen upstream
  scorer code/weights, SENSE rows, and factorization receipt.

## Next exact rung

Bind a counted own-lineage stored problem to a deterministic receiver expansion,
activate both the Seg cell and Pose tube in the integer solve, sweep the
tolerance knees, and emit a SHA-bound receiver-decodable exception stream.
Before any lattice-family negative, test a bounded exact sieve or
branch-and-bound. Before any factor becomes vocabulary, run the measured
per-stratum SKELETON-versus-FIBER coder race.

This isolated branch is ready only for MAIN landing review; it does not
self-promote, dispatch, move the pointer, or claim a contest score.
