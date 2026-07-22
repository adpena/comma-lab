---
schema: ddm_g1_grammar_induction_landing.v1
date_utc: 2026-07-22
lane_id: ddm_g1_grammar_induction
axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
execution_allowed: false
score_claim: false
candidate_archive: false
main_landing_review_required: true
---

# DDM G1 per-stratum grammar induction — measured candidate-set result

## Outcome first

The measured winner does **not** clear the joint bar. Lane active-slot delta grammar plus Movable
absolute-shape track grammar costs **57,502 counted bytes** and has a **DERIVED** clean-rest union
upper bound `d_seg = 0.005228635999891493`. It fits the 60,000-byte box with 2,498 B slack, but is
26,971 mask errors / `0.000228635999891493` above 0.005. It is 4.5074x the separate 0.00116 box.
This is mask-grammar evidence, not receiver-visible RGB, through-R, Pose, final-ZIP, or score
evidence.

| rank in measured set | stratum | grammar | counted bytes | decoded mask error | clean-rest d_seg | disposition |
|---:|---|---|---:|---:|---:|---|
| 1 | Movable | track + absolute relative shape, eps1 | 29,810 | 33,378 | 0.000282948812 | lossy knee |
| 2 | Lane | active slots + delta dash, tol x2 | 27,692 | 583,417 | 0.004945687188 | lossy knee |
| 3 | Boundary | open arc, eps2 | 219,288 | 2,245,366 | 0.019034203423 | rejected for 60 KB line |
| L0 | Movable | exact row runs | 59,481 | 0 | 0 | lossless winner |
| L0 | Lane | exact row runs | 180,701 | 0 | 0 | lossless winner |
| L0 | Boundary | exact transition row runs | 453,123 | 0 | 0 | lossless winner |

“Winner” and “optimal” mean minimum within the emitted, semantic-parse-backed candidate set only.
No global grammar-optimality claim is supported.

## Measured bytes per production

Each total includes a five-byte envelope header. Every production independently ran through actual
Brotli-q11, raw LZMA1 preset-1/1-MiB-dictionary, and zlib-9; the smallest real stream won, with its
codec id and ten-byte stream frame counted. No entropy estimate ranked a row.

| stratum knee | production | counted bytes | winning coder |
|---|---|---:|---|
| Movable eps1 | EVENT | 419 | raw LZMA1 |
|  | CENTROID | 2,354 | Brotli |
|  | SHAPE | 27,032 | Brotli |
| Lane tol x2 | EVENT | 198 | raw LZMA1 |
|  | LANE_CENTER | 12,367 | Brotli |
|  | LANE_WIDTH | 5,632 | Brotli |
|  | LANE_DASH | 6,314 | Brotli |
|  | LANE_RANGE | 3,176 | Brotli |
| Boundary eps2 | ARC_EVENT | 448 | Brotli |
|  | ARC_VERTEX | 218,835 | Brotli |

The exact one-production row-run totals likewise include framing: Movable 59,476+5, Lane
180,696+5, and Boundary 453,118+5 bytes.

## Extracted corpus and induced productions

The frozen `lstars` member is `(600,384,512) int64`. Movable extraction found at the eps1 knee 180
births, 2,017 persists, 158 deaths, at most ten slots, and 19,150 polygon vertices. Its syntax is
`BIRTH(abs-centroid,shape)`, `PERSIST(delta-centroid,morph-or-shape)`, `DIE`; absence emits no
centroid or shape values. Split/merge is conservatively represented as death plus birth because
the 48-pixel Hungarian continuation gate does not establish lineage authority.

Lane extraction found 2,967 fitted lines over 600 pairs, six coherent slots, one birth, and no
death. Its syntax is `EVENT`, then birth-absolute / active-persist-delta `CENTER`, `WIDTH`, `DASH`,
and `RANGE`. Inactive slot-frames emit no parameter values. A competing free persistent-dash
default was measured at every tolerance, not assumed. Boundary syntax is `ARC_EVENT` plus
delta-coded `ARC_VERTEX`; its large false-positive band makes this formulation a scoped negative.

No xi trajectory exists in the frozen label cache, so “ride-xi” and xi-keyed dash phase were not
silently treated as free. Likewise no cross-stratum Lane-subset-of-Road derivation was counted as
free without a deterministic receiver rule. Those remain honest candidate extensions, not
measured productions here. No transform residual was introduced, so the operator's rule that any
future residual basis be curvelet/shearlet is preserved; Fourier is absent.

## Lossy ladders

| tolerance | Movable selected mode | bytes | clean-rest d_seg | Lane delta-dash bytes | Lane clean-rest d_seg |
|---:|---|---:|---:|---:|---:|
| 0 / x0.5 | absolute shape | 71,084 | 0.000000084771 | 34,243 | 0.004616038005 |
| 0.5 / x1 | absolute shape | 53,559 | 0.000047887166 | 30,910 | 0.004738794963 |
| 1 / x2 | absolute shape | 29,810 | 0.000282948812 | 27,692 | 0.004945687188 |
| 2 / x4 | morph delta | 20,763 | 0.000624245538 | 23,525 | 0.005344958835 |
| 4 / x8 | morph delta | 14,875 | 0.001318622165 | 19,065 | 0.006176749335 |
| 8 / x16 | morph delta | 11,067 | 0.002440321181 | 15,492 | 0.006426968045 |

Morph deltas are not universally better: at eps1 they cost 29,960 B versus 29,810 B for absolute
relative shape; they first win at eps2, 20,763 B versus 20,967 B. The semantic disambiguator, not
intuition, chooses the row.

## Coverage projection and blocker delta versus #603

`Lane(tol x2) + Movable(eps1) = 27,692 + 29,810 = 57,502 B`; decoded error counts sum to
`583,417 + 33,378 = 616,795`. Dividing by `600*384*512` gives the DERIVED union upper bound
0.005228635999891493. Road, Undrivable, and MyCar are assumed exact, and cross-stratum overwrite
interactions are omitted. Therefore this is deliberately more limited than a through-R score.

The #603 blocker changes from an unmeasured bounded-atom inventory to a concrete predictor-native
grammar deficit: at most 60 KB, this candidate set still owes 26,971 mask-error removals, then a
receiver-visible RGB realization, cross-stratum composition, exact parse/re-encode in the final
container, Pose custody, and contest-axis replay. The v12 200-KB correction plateau remains a
formulation-scoped predecessor, not evidence against the describe-line family.

## Round-1 adversarial self-review

1. The first Lane encoder paid values during inactive slots, contradicting the free-absence
   grammar. It was replaced with birth-absolute and active-persist deltas, saving 530 B at tol x2.
2. Envelope decompression alone was insufficient reconstruction proof. Movable absolute/morph and
   Boundary arc semantic decoders were added; all reported masks are now decoded from emitted
   production bytes. Exact row-run and Lane semantics were already parse-backed.
3. The intuitive morph-delta production lost at the selected eps1 knee; both outcomes remain in the
   receipt. Boundary arcs are retained as a scoped negative, not erased from the table.
4. The projection is not Fisher/margin ranked and has no corrected inner-Jacobian or Pose term.
   It cannot authorize an exception waterfill, candidate archive, launch, or score claim.

Ten focused tests pass. The three stage checkpoints are atomic and preserved on the preferred SSD;
the completed run hash-validated and resumed in 2.567 seconds.

## Bounded re-derivation

```bash
/Users/adpena/Projects/pact/.venv/bin/python experiments/direct_description/induce_per_stratum_grammar.py --cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz --output-directory /Volumes/VertigoDataTier/pact/experiments/results/ddm_g1_grammar_induction_20260722T182036Z_semantic_v4 --execution-allowed false --resume
```

Primary receipt SHA-256:
`aeeb916f973523d5ffa3389ee8d744901fe9477cc149af7e756726e2ead907f6`. The compact repository
receipt is `.omx/research/direct_description_g1_grammar_induction_20260722.json`.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and `docs/operating_manual_craft_handoff.md`
- v7.5 §8 and v8 operating specs; #596 representation-mining memo; #610 wrong-levels/per-stratum
  surface-4 memo and reuse manifest
- DDM v9-v12 code, receipts, findings, DAG FEEDs, and equation notes; v12 residual decomposition
- #229 Lane-band and #234 coherent-slot code; #503/Crux-3 object-domain grammar surfaces; #287
  dash-comb surface; truly-optimal-coder survey and stream-specific real-coder verdicts
- frozen n600 GT cache; `reports/latest.md`; lane registry; canonical task/subagent state; per-arm
  inbox (empty) and broadcast through `2026-07-21T13:15:53Z`
- operator 2026-07-19 EV/rate-break-even and Fisher/margin, corrected-inner-Jacobian,
  curvelet/shearlet, and xi-factorization directives

Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review is required before any consumer
uses this table.
