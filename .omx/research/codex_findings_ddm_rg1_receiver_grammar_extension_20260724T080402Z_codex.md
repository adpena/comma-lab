# Codex findings — DDM RG1 receiver grammar extension

Date: 2026-07-24  
Lane: `lane_ddm_rg1_receiver_grammar_extension_20260724`  
Verdict scope: `INSTANCE_EXTENDED_GRAMMAR_RG1`  
Evidence axis: `[macOS-CPU frozen-scorer advisory]`  
`score_claim=false`; `research_only=true`; `main_landing_review_required=true`

## Verdict

RG1 makes all 24 Lane coordinates and all six preregistered G2CS1 coordinates
legal, counted, isolated receiver inputs. The inactive compiler is
byte-identical to the sealed nested carrier, and the ten explicitly named
bounded geometry alternatives do not silently replace the original failures.

The bounded rerun is complete: 80/80 new signed probes, comprising 78 measured
argmax perturbations and two measured empty-raster bounded alternatives. All
48 Lane signed probes are receiver-feasible and perturbative. The merged
MS5-schema table binds 768 checkpoint revisions while preserving the original
748-checkpoint root and all 70 original infeasibility rows.

The G3 result is a real partial closure, not a gate pass: missing exact
pair/bucket blocks fell from 106 to 64, but only pair 21 is complete. Therefore
`producer_rerun_eligible=false`; MS4 was not invoked. The pointer remains
`0.1910828242 [contest-CPU]` UNMOVED.

## Stores consulted

- Delegated authority SHA-256
  `f3edabf7dd2d7e64a091f678d23394f478211915f3ddd3e23d7104f3cf71a5d0`,
  8,013 bytes.
- V19C outer base SHA-256
  `dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9`.
- Sealed nested carrier SHA-256
  `7990fce786aac1f24bcb977882348867ca2d9cbc4d95d0337dd1167e593f46c6`.
- G2G quantum receipt SHA-256
  `fa49a2ca71cb2960b1e497d425f05c4a496cc7634c45b2e193e3977dfa0667da`.
- Original v2 checkpoint digest chain
  `27f6c56928068b383fa930095e3b4d3f6faa2433c40a0dc81c39e8d3cf9dabdc`.
- RG1 merged producer digest chain
  `3149967c3c78ef829740a5441df5dd63c72a7100f99b1954004f977d7bb36cb9`.
- Assignment table file/content SHA-256
  `2274c8e654262b90ef35a604280c6c8a4e07a7403480b568d1c3ac8ea8141170`
  and
  `576d2995e8c729b5ff450c7498fe7b07b38bdf51b819ef6a4d2190843d6c75a0`.
- Summary file/content SHA-256
  `1960687be407b89eb3ccf97f5a5abad1ef62eb939600d164285c94059fc893da`
  and
  `0293c976bf6209bcb314d641a19ac38062b72cca2a4be87f99d2d8aaf337505c`.

## Finding 1 — RG1 is additive and inactive-byte-identical

RG1 is a version-bumped outer grammar. It never mutates V13. Its members are:

- `base/v13_v19c_carrier.zip`;
- `production/lane_program_coordinates.rg1lp`, tagged `SKELETON/L1_program`;
- `correction/lane_chart_symbols.g2cs2`, tagged `RESIDUAL/L2_chart`.

The composition order is sealed base → Lane production → post-solve correction
→ inherited raster/R. Empty production and correction streams emit no wrapper,
so `P_0=C_0=I`; the input carrier bytes are returned exactly. The manifest
declares compatibility with all five `TypedStreamTag` types and forbids pixel,
RGB, scorer-weight, or GT-argmax payloads.

## Finding 2 — all 24 Lane DOFs are true isolated coordinates

The six Lane lines expose four coordinates apiece:
`dash_phase_origin_q8`, `dash_phase_xi_gain_q8`, `width_bias_q8`, and
`width_slope_q12`. The packet enforces one canonical address per row, sorted
uniqueness, int16 signed quanta, CRC, strict parse-back, and a maximum of 24
rows.

The measured proof has 24 unique coordinate IDs, 48 signed probes, 48 unique
candidate archive SHA-256 values, zero infeasible Lane rows, and 48
`MEASURED_ARGMAX_PERTURBATION` rows. Each probe compiler call carries exactly
one Lane coordinate.

## Finding 3 — typed G2CS1 and geometry alternatives are receiver-effective

The separate correction stream lifts the old V13 vocabulary firewall without
weakening it. Six G2CS1 coordinates use exact magnitudes bound to the prior G2G
receipt; all 12 signed probes are measured perturbations.

The ten old island `center_x` failures remain preserved under their original
actuator IDs. Ten new `.bounded_clamp` coordinate IDs project the full polygon
onto
`[-min(relative), extent-1-max(relative)]`. Both signs were measured for each
alternative: 18 are argmax perturbations and two are explicit empty-raster
results. The alternative label is never silently substituted.

## Finding 4 — exact G3 coverage improves but remains structurally incomplete

| G3 pair | Required / joined / missing buckets |
|---:|---:|
| 523 | 20 / 12 / 8 |
| 54 | 16 / 15 / 1 |
| 1 | 13 / 11 / 2 |
| 90 | 18 / 14 / 4 |
| 21 | 10 / 10 / 0 |
| 446 | 18 / 16 / 2 |
| 0 | 13 / 12 / 1 |
| 14 | 13 / 10 / 3 |
| 18 | 16 / 12 / 4 |
| 327 | 31 / 25 / 6 |
| 7 | 13 / 10 / 3 |
| 60 | 19 / 16 / 3 |
| 49 | 13 / 10 / 3 |
| 41 | 12 / 11 / 1 |
| 323 | 25 / 22 / 3 |
| 44 | 14 / 11 / 3 |
| 38 | 13 / 12 / 1 |
| 42 | 11 / 10 / 1 |
| 4 | 10 / 9 / 1 |
| 36 | 12 / 11 / 1 |
| 320 | 20 / 17 / 3 |
| 55 | 13 / 10 / 3 |
| 56 | 15 / 12 / 3 |
| 16 | 14 / 10 / 4 |

The 64 residual rows are machine-readable in
`ddm_rg1_receiver_support_summary.json`. Their typed keys derive the next
coordinate families rather than guessing from convenience:

- Lane keys → counted Lane-band productions;
- Movable keys → bounded G1 polygon coordinates;
- transient keys → pair-local post-solve corrections;
- remaining boundary/cell keys → event-local boundary productions or
  per-stratum SKELETON amplitude fields.

The first three families now exist in RG1 but do not join every exact hard
pair. The residual reason is
`NO_MEASURED_RG1_PROBE_JOIN_AT_EXACT_PAIR_BUCKET`, scoped only to this extended
grammar instance. Event-local/per-stratum SKELETON production remains the next
build family; this is not a family/paradigm negative.

## No-orphan routing

- Sensitivity map: the assignment table carries exact PF2 event incidence and
  exact joined pair IDs for every measured signed probe.
- Pareto/bit allocator: no RG1 coordinate is admitted by support alone; it
  still owes measured score-units-per-byte and the operator's reverse-waterfill
  threshold.
- Cathedral/autopilot: MS4 remains refused while any exact G3 block is absent.
- Continual learning: the summary, findings, and canonical equation registry
  preserve the partial closure and the 64-row residue.
- Probe disambiguator: original and bounded geometry coordinate kinds remain
  simultaneously callable and separately measured.

## Review disposition

Round 1 caught a bookkeeping error in the draft receipt: ten original
geometry-escape rows were initially counted as twenty by multiplying unique
actuator IDs by both directions. The final producer instead counts the exact
preserved infeasible rows. Three post-fix clean passes are recorded separately.
MAIN must review the branch diff and all custody claims before merge.
