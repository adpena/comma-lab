# DDM FR1 Fisher actuator base curves — custody finding

`lane_id=lane_ddm_fr1_fisher_actuator_base_curves_20260724` ·
`research_only=true` · `[macOS-CPU frozen-scorer advisory]` ·
`score_claim=false` · `pointer=0.1910828242 [contest-CPU] UNMOVED` ·
`MAIN_REVIEW_REQUIRED=true`

## Outcome

The requested cross-base measurement did **not** start. The exact #583 rank-1
Fisher candidate is recoverable, but the correction-status artifact bound by
the same M1 manifest says:

- `realized_backbone_secants=ABSENT`;
- `qp_receiver_closure=ABSENT`;
- `formalization=FORMALIZATION_PENDING`.

The rank-1 row is therefore an ordered first-order VJP candidate, not the
corrected-inner-Jacobian actuator required by the charter. Applying its naive
unit pullback would violate the operator's explicit first-order + secant + QP
law. The DDM runtime-sensitivity API also accepts chart/semantic coordinates,
while the #583 row is a `(pair,row,col,RGB pullback)` coordinate; no typed bridge
between those domains exists.

The strict preflight emitted
`verdict=BLOCKED_PREMEASUREMENT_CORRECTED_INNER_JACOBIAN_ACTUATOR_NOT_EXECUTABLE`
and no scorer call, camera tensor, candidate archive, or fabricated delta.
Receipt SHA-256:
`877b3d9f2c02f8e3b5924dc93f1d8ef31d34751f576ae9bf8aa85da8eeb0774f`.

## Rank-1 identity and provenance

| field | value |
|---|---:|
| candidate id | `pdw1_fisher_rank_00001_pair_0022_cell_225_0045_lane_from_road` |
| rank | 1 of 38,077 |
| pair / scorer cell | `22 / (225,45)` |
| desired transition | `Road -> Lane` |
| top1-top2 margin | `4.8160552978515625e-05` |
| Fisher trace | `0.49999999971007014` |
| head-normal flip distance | `1.2181893519719522e-05` |
| exact resize support taps | 4 |
| VJP arrangement | native match |
| unit pullback RGB | `[0.47628337144851685,-0.09180968999862671,-0.8744856715202332]` |
| ordering SHA-256 | `765457d424eaf1de7e05ed8703853175ef415bd3f19fb00137a74a29de52ae00` |

This is the candidate ID. There is no honest corrected executable actuator ID
yet.

## Independent base custody

The preflight preserved the curve split and revalidated every consumed hash.

| base | exact parent status | typed row |
|---|---|---|
| V19C endpoint | materialized archive, 137,827 B, SHA `dc767b59...52e4c9` | 2,923,991 errors; `d_seg=0.024786978827582466`; `d_pose=163.06121002915629` |
| WS1 `W_seg` | scored endpoint over recompiled base, **not a materialized W_seg state** | 2,845,843 errors; `d_seg=0.024124510023328993`; `d_pose=146.3649324958955`; 138,031 B |
| WS2 | no `ddm_ws2*receipt*.json` landed in the canonical research tree | fallback remains WS1 endpoint with endpoint != state caveat |

The V19C endpoint was not confused with the 8,318,787-error Menu1 joined
candidate. E-line bytes were not subtracted or composed. No base was mixed with
another base's endpoint, state, or bytes.

## Headline decomposition

- `delta_errors_per_class=NOT_MEASURED`
- `delta_d_seg=NOT_MEASURED`
- `delta_d_pose=NOT_MEASURED`
- `delta_bytes=NOT_MEASURED`
- `joint_delta_S=NOT_MEASURED`
- `base_dependence_hypothesis=NO_VERDICT_DATA_CUSTODY`
- heavy phase: not started
- governor/memory admission: not run because the heavy phase was refused before
  admission was needed

The charter's falsifier cannot be evaluated. Neither base-dependence nor
base-independence is supported.

## Exact blockers

1. `FR1_CORRECTED_INNER_JACOBIAN_REALIZED_SECANTS_ABSENT`
2. `FR1_CORRECTED_INNER_JACOBIAN_QP_RECEIVER_CLOSURE_ABSENT`
3. `FR1_CORRECTED_INNER_JACOBIAN_FORMALIZATION_PENDING`
4. `FR1_PER_CANDIDATE_EXACT_PREFIX_BYTE_MARGINAL_ABSENT`
5. `FR1_RANK1_TO_DDM_RUNTIME_PERTURBATION_BRIDGE_ABSENT`

G2e does not close these blockers for this experiment: it measured a different
openpilot base at n16, found zero usable trust regions out of 31, invoked zero
QPs, and explicitly forbids n600 or cross-base substitution.

## FIRST-RUNG

Build one typed bridge from the exact rank-1 `(pair,row,col,RGB pullback)` row to
an independently serializable DDM perturbation, then measure paired signed
receiver-closed secants on the current candidate state. Admit the actuator only
after a nonempty class/margin trust region, deterministic QP solve, positive
rounded hard-oracle realization, and exact prefix-byte marginal are all bound
in one receipt. Only then rerun this unchanged preflight and apply the identical
actuator independently to V19C and the WS1/WS2 state selected by custody.

## Triality and system intelligence

- DAG: `ddm_fr1_fisher_actuator_base_curves_DAG_FEED_20260724.md`.
- Equations: `ddm_fr1_fisher_actuator_base_curves_canonical_equations_20260724.md`.
- Directive consumption:
  `ddm_fr1_fisher_actuator_base_curves_directive_consumption_20260724.json`.
- Three-pass adversarial review:
  `ddm_fr1_fisher_actuator_base_curves_review_receipt_20260724.json` (three
  clean passes; 16 tests per pass).
- Reusable guard: `tac.optimization.ddm_fr1_fisher_preflight` plus its CLI and
  regression tests. This turns the missing-custody result into a strict
  pre-measurement refusal rather than a chat-only observation.

## Verdict scope

`INSTANCE:CUSTODY` of the #583 rank-1 candidate, its missing corrected
receiver realization, and the named V19C/WS1/WS2 parents. This is not an
actuator efficacy negative and does not close any formulation, family, or
paradigm.

## MAIN landing requirement

MAIN must review the isolated branch diff, rerun the focused tests, rehash the
SSD ordering/status artifacts, and confirm that the guard cannot be satisfied
by a first-order-only row or the G2e n16 receipt. The branch has no authority
until explicit MAIN merge review.
