# Codex findings — DDM J5 #366 realized-acceptance warm start

Date: 2026-07-23  
Lane: `ddm_j5_366_realized_acceptance_warmstart`

## Disposition

`READY_TO_FIRE_UNDER_STANDING_GO`, conditional on independent MAIN landing
review. No campaign was launched, no contest score was claimed, and the
frontier pointer did not move.

## Findings

1. **J4's fixed cap was the wrong control.** Its preserved beta2 rewarmup,
   delayed Pose force, and exact rollback remain sound, but the quarter-quantum
   cap held every opening coordinate inside the same receiver cell.
2. **The preliminary J5 proposal mask was over-broad.** Applying x+1 to all 163
   tracks refused with `G1 Movable polygon escaped scorer geometry`. Re-reading
   v19 showed the candidate name `active` meant both a fixed eight-pair screen
   and whole-lifecycle geometry feasibility. This was an implementation defect,
   not a family result.
3. **The corrected proposal is exact, not approximate.** The Q8/Adam path
   produces archive SHA `d4eb1450...`, byte-identical to v19's measured x+1
   grammar candidate. A regression pins this identity.
4. **The v19 eight-pair signal generalizes to exact n600.** The candidate
   improves both components, saves five archive bytes, and has strict joint
   `delta S=-0.002843840398518996`.
5. **The move reaches C1's residual debt.** It removes 2,013 residual-trunk
   errors and 1,314 role/correction errors. The per-class decomposition is Road
   -1,780; Lane -30; Undrivable +597; Movable -1,284; MyCar -830. The aggregate
   residual improvement is therefore real despite localized Undrivable harm.
6. **Resume compatibility needed an additive fallback.** Old J4 baseline rows
   have `per_class` but not `c1_debt_buckets`; J5 now derives the same fixed
   partition from historical per-class custody instead of failing a resumed
   run.

## Exact custody

- ticket semantic SHA: `13e194a8a354d53489f0ff68a5042237e69b4b6841a6b7959a15873fffa7b6e8`;
- typed hash: `d43608af799b2f2d04e248413ceb944c093701441eafb222f2b3cdf3d32b8d80`;
- full-run receipt SHA: `975a3529481fac12a63b94a4820709e1f51e08e56ceefded0bd310fa1a25ab45`;
- proposal receipt SHA: `335d15db69dc25b0de80258e10e472f6e9ff282cd7d0325eca989bf92bade1f2`;
- fresh governed preflight SHA:
  `1b5a9c4a77c97709f3b12f458387c785c354fd5503479ccb82bf821568aaf0f0`;
- checkpoint SHA: `1399292164682955bd9937d204692521542687f69223ddc9cc1e669ae05a944e`.

## MAIN review request

Adversarially re-derive the lifecycle bounds and exact archive-identity test,
then inspect pure-price admission/shrink/rollback, C1 partition semantics,
historical resume fallback, and the sealed ticket. If clean, MAIN—not this
branch—may use standing authority for the resumable 13.3–13.8 hour campaign.
