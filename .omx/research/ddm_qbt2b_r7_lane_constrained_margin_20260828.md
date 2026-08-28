# ddm_qbt2b r7 — Lane/Movable constrained margin law built, byte-closed n1 smoke repeated, r7 treatment sealed; n32 fire BLOCKED by live storage preflight

Date: 2026-08-28  
Arm: `ddm_qbt2b_r7_lane_constrained_margin`  
Disposition: **IMPLEMENTED + SEALED; QUEUED-WITH-A-FIRE-ORDER; NOT LAUNCHED**  
Score claim: `false`

The per-class primal-dual law is live in the real qbt1 margin stage. The
unconstrained path remains the default and preserves the old objective branch
exactly. A bounded n1 CPU smoke and a second fresh-directory repeat produced
identical λ trajectories and identical retained archives. The real n32 r7
treatment config is sealed, validated before write and after JSON readback, but
it is not authorized to launch: the measured APDataStore free space was
951,503,377 B below the fail-closed requirement. No Metal, Modal, n600, or exact
contest evaluation was invoked.

## 1. Constraint law and constant provenance

For each margin step and each protected class `c in {Lane, Movable}`:

```text
g_c       = realized_werr_c - bound_c
lambda_c  = clip(lambda_c + eta_lambda * g_c, 0, lambda_max)
penalty_c = 100 * lambda_c * E[flip_probability | target_class = c]
```

`realized_werr_c` is measured on the actual
`render -> R -> uint8 -> frozen SegNet argmax` result. It is not a native-logit
or soft proxy. The differentiable primal penalty is restricted to real
target-class pixels and uses the same expected-flip probability as the existing
margin law. The factor 100 places λ in the existing score-scaled segmentation
units; the unconstrained objective already prices realized and native-interface
expected flips as `100 * (...)`.

| Field | Pinned value | Provenance and meaning |
|---|---:|---|
| mode | `lane_movable_werr_primal_dual` | New config-gated mode; missing/`unconstrained` remains legacy off. |
| Lane bound | `0.12` werr | Outward envelope of the retained r6 born verdicts, 0.116328 and 0.119581; r5 endpoint was 0.0980. This protects the measured born field rather than shifting its target. |
| Movable bound | `0.009` werr | Outward envelope of r6 born verdicts 0.007490 and 0.008856; r5 endpoint was 0.0065. |
| `eta_lambda` | `0.11387788414126129` | Derived from r6's retained Lane endpoint 0.9981336319522209 as `1 / (10 * (0.9981336319522209 - 0.12))`: persistent endpoint-sized violation raises λ by one natural unit in ten margin steps. |
| `lambda_max` | `5.0` | Reused reviewed #808 `ddm_lg1` ceiling. Ceiling contact is emitted as an infeasibility alarm, not treated as success. |

This is an absolute born-envelope constraint, not the `ddm_bs2` monotone
ratchet. The latter protects every later win but was designed around a different
vehicle and gate statistic. Here the falsifiable question is narrower: can the
r5-born qbt field remain feasible while the r6 margin law descends?

## 2. Implementation and configuration identity

Owned edits:

- `experiments/ddm_qbt1_qbflow_trainer.py`
  - adds the two-class constraint mode and pinned `(mode, bounds, eta_lambda)`
    group;
  - computes real through-R per-class werr before every margin update;
  - persists λ in `curriculum_state` at every checkpoint;
  - emits bounds, werr, residual, binding state, λ before/after, ceiling state,
    and realization path in every constrained margin history row;
  - adds the class-restricted differentiable penalty only when the mode is on;
  - adds a bounded `constraint-smoke` command and a fail-closed
    `compile-r7-config` command;
  - fixes the pre-existing storage projector's assumption that the final
    history row must be a birth row; it now selects the latest retained birth
    verdict, allowing a successful margin tail to be projected.
- `experiments/tests/test_ddm_qbt1_qbflow_trainer.py`
  - adds default-off exact identity, live penalty/gradient, dual rise/decay,
    paired pin refusal, legacy validation, config identity/cross-mode resume,
    and margin-tail storage-projection coverage.

The `config_identity` exclusion set remains exactly
`{action, resume_from, launch_authorized, scorer_lane, metal_lane}`. The three
constraint fields therefore participate automatically. A checkpoint written in
legacy/unconstrained mode refuses a constrained-mode resume with
`resume config identity differs`.

Legacy configs with all three fields absent validate as unconstrained with empty
bounds and `eta_lambda=0`. The unconstrained call and an explicit
`margin_constraint_lambdas=None` call return bit-identical tensors and the same
component-key set.

## 3. Verification matrix

| Surface | Command / check | Result |
|---|---|---|
| Focused/full trainer test file | `.venv/bin/python -m pytest -q experiments/tests/test_ddm_qbt1_qbflow_trainer.py` | **24 passed** |
| Ruff source | `.venv/bin/python -m ruff check experiments/ddm_qbt1_qbflow_trainer.py` | **passed** |
| Ruff tests | `.venv/bin/python -m ruff check experiments/tests/test_ddm_qbt1_qbflow_trainer.py` | **passed** |
| Patch whitespace | `git diff --check -- <owned files>` | **passed** |
| Default off | Exact `torch.equal` objective and components | **passed** |
| Group pins | mode-only, values-only, and wrong-eta combinations | **refused as required** |
| Legacy config | all new fields absent | **validated unchanged** |
| Resume | unconstrained checkpoint into constrained config | **refused as required** |
| Dual law | above-bound rises; below-bound projects back to zero | **passed** |
| Primal law | both class penalties nonzero and scorer-logit gradient nonzero | **passed** |
| Review pass 1 | `review_tracker`, adversarial math/backward-compatibility pass | **80 source + 25 test entities marked clean** |
| Review pass 2 | `review_tracker`, custody/resume/fail-closed pass after a six-test focused rerun and retained-byte audit | **80 source + 25 test entities marked clean** |

## 4. Bounded n1 mechanism smoke

Scope: pair 62 only, CPU, 10 balanced-CE birth steps, two constrained margin
steps. This is a legality/determinism proof, not a family verdict. Axis for all
numbers in this section: **`[macOS-CPU mechanism smoke; not a verdict]`**;
`score_claim=false`.

Primary custody:

- result:
  `/Volumes/APDataStore/pact/ddm_qbt2b_r7_lane_constrained_margin/smoke_n1/RESULT.json`
  - SHA-256 `76db3e8c84d6aae9d044c7a45072bb46593dd12bede32a4ed24ebba3f0d689fe`
  - elapsed 48.32016133307479 s
- deterministic repeat:
  `/Volumes/APDataStore/pact/ddm_qbt2b_r7_lane_constrained_margin/smoke_n1_repeat/RESULT.json`
  - SHA-256 `fa1fa95bbd4fce9d9b641528ba88f127390fbc2bd3de82cbededb201ca721a5b`
  - elapsed 47.08004770800471 s
- retained trees: 71 MiB each; `all_payloads_retained=true` in both results.

| Margin step | Lane werr / bound | Lane λ before -> after | Movable werr / bound | Movable λ before -> after | Constraint penalty |
|---:|---:|---:|---:|---:|---:|
| 11 | 0.3312159709618875 / 0.12 | 0 -> 0.024052827869981832 | 0.04081632653061224 / 0.009 | 0 -> 0.0036231759464535984 | 0.846222996711731 |
| 12 | 0.3411978221415608 / 0.12 | 0.024052827869981832 -> 0.049242367832117814 | 0.04224015187470337 / 0.009 | 0.0036231759464535984 -> 0.007408494110478997 | 1.7555546760559082 |

Both constraints were binding, both multipliers rose by the exact projected
dual law, and the primal term was nonzero. Neither multiplier was at its
ceiling. The two fresh-directory runs matched exactly on the full constraint
telemetry and objective rows.

Resume/byte-closure evidence:

- 03a boundary checkpoint reload: live state, EMA state, RNG state, cursor,
  curriculum state, and re-encoded archive identity passed in each run.
- pre/post reload 03a archive: 107,243 B,
  SHA-256 `68a7dfebd7af6d03d5adb8d0e0d78636f5eff7ad2ac49c7581ca77f9cde24ade`.
- margin/stage-03 archive, independently repeated through stages 04 and 05:
  107,276 B,
  SHA-256 `412c39e516be5f55122cc75966e89fd2d8d5da14db6a6609eef1a697606595e9`.
- retained payload tar: 1,536,000 B,
  SHA-256 `e2c78464e0c17803e6e1b10ed5ef7fa314a9ab523f67a0be511a3f8cd2acd350`.
- both fresh-directory runs emitted the same archive and tar hashes at every
  corresponding stage.

Checkpoint file hashes differ between the two output roots because the
checkpoint payload retains path-bearing provenance. That is not claimed as
byte identity. The independently decoded/re-encoded shipped payload bytes and
the λ/objective trajectories are the deterministic equality surfaces.

The smoke's one-pair score-like estimator was deliberately refused as a score:
it is an unweighted bounded smoke mean, lacks a same-budget QBW1 control, and is
neither n32 nor contest authority.

## 5. Sealed r7 config and single-variable audit

Config:

- path:
  `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/AUTHORIZED_N32_R7_5020_20260828.json`
- bytes: 6,537
- file SHA-256:
  `883213a00d77eac0c2b725f53b9c6ce23c83e2e976c945b87a6e077c0d47b05f`
- canonical config SHA-256:
  `042a63c4e00fb092bbbf08fbf5f5afb16ac961dfb016ffe520ea01b7d8a396b0`
- config-identity SHA-256:
  `d0096062d0bc302a15ecf1348f1e26b08cecf4be3be76e7ba7dfeac022304483`
- validation: passed before atomic write and after JSON readback; the parsed
  object and canonical hash were identical.

Pinned treatment:

- `birth_event_mode=existence_majority`
- `birth_max_steps=20`
- `margin_steps=5000` (`steps=5020`)
- `birth_class_weight_mode=balanced`
- `margin_constraint_mode=lane_movable_werr_primal_dual`
- bounds `{Lane: 0.12, Movable: 0.009}`
- `margin_constraint_eta_lambda=0.11387788414126129`
- initialized from
  `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r5/initialized_r6_from_r5_cap_ema_state.pt`,
  SHA-256 `4b40acc584546be39839bba7490c9e1ae53286f73a038d30c2070ec27b6d700b`.

After normalizing only the EMA LawRef resolution timestamp and excluding
dispatch/output authority, the r6 and r7 configs have no unexpected
differences. R6 has no constraint fields; r7 adds exactly the pinned constraint
tuple. R7 dispatch fields are intentionally
`launch_authorized=false`, `scorer_lane.claimed=false`, and
`metal_lane.claimed=false`; the arm did not counterfeit MAIN's claims.

## 6. Storage preflight — launch blocker

Receipt:
`/Volumes/APDataStore/pact/ddm_qbt2b_r7_lane_constrained_margin/R7_STORAGE_PROJECTION_20260828.json`,
SHA-256 `176f81d904ba3cb9cbd6dcb2636b8a047b31ee9a80deef1fa78a5578f6e29fdc`.
Axis: **`[macOS APDataStore on-disk projection; no score claim]`**.

| Quantity | Bytes |
|---|---:|
| projected retained run | 20,182,632,742 |
| 10% safety reserve | 2,018,263,275 |
| required post-run free floor | 8,589,934,592 |
| required live free | 30,790,830,609 |
| measured live free | 29,839,327,232 |
| shortfall | **951,503,377** |

`passes_live_df=false`; therefore the sealed config status is
`SEALED_BLOCKED_LIVE_STORAGE_PREFLIGHT`. No bytes were deleted or moved. The
certify-or-block rule requires an operator/storage owner to recover or
cold-route at least the shortfall with a machine-readable provenance manifest,
then rerun the compiler/preflight before claiming lanes.

## 7. READY_TO_FIRE handoff — QUEUED-WITH-A-FIRE-ORDER

Disposition: **QUEUED-WITH-A-FIRE-ORDER**  
Owner: **MAIN**  
Consumer store:
`/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r7`  
Fire trigger: a fresh storage receipt has `passes_live_df=true`; the exact
treatment/config identity is unchanged; MAIN has written active, non-duplicate
r7 scorer and Metal claims; `launch_authorized=true` and those real claim IDs
have been written through the compiler/validator surface.

Exact governed argv after those triggers, mirroring counter-693 with r6 paths
changed to r7:

```bash
.venv/bin/python tools/safe_run.py \
  --rss-mb 118784 \
  --projected-gib 52.0 \
  --timeout 43200.0 \
  --status-receipt /Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r7/resource_safe_run_status.json \
  --child-pidfile /Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r7/resource_safe_run_child.pid \
  --quiet -- \
  .venv/bin/python experiments/ddm_qbt1_qbflow_trainer.py run-config \
  /Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/AUTHORIZED_N32_R7_5020_20260828.json
```

Do not run that argv against the currently sealed file: it is deliberately
unauthorized and storage-blocked. MAIN must retain all payloads and preserve
each stage checkpoint under a distinct stage-encoded name, as the trainer does.

Harvest decision:

- If Lane werr stays at or below 0.12 throughout the full 5,000-step margin
  window and total realized flip finishes within about `2 * 0.00972`, the
  constraint prediction survives and the exact byte-closed row is the next
  consumer.
- If λ_Lane reaches 5.0 while total flip remains much greater than 0.01944, or
  Lane werr breaches 0.50 with active λ, classify this **FORMULATION-INFEASIBLE
  on the r5-born qbt basis** and route Lane to the m131/d3a analytic
  Lane-carrier leg. Do not reweight CE or expected-flip.

## RECALL EVIDENCE

The full-corpus recall used content queries, not filename-only recall:

- `.omx/research/` queries:
  `Lane primal-dual`, `realized werr`, `class constraint`, `expected-flip`,
  `born Lane`, `margin erasure`, `budget ratchet`, `aggregate class scalar`,
  `Lane carrier`, and `qbt2b`.
- canonical equations:
  `.venv/bin/python tools/list_canonical_equations.py --json`, then searched
  for constraint/KKT/dual/Lane/QBT entries.
- graph/index:
  `CANONICAL_RESEARCH_INDEX*` plus `sub015_DAG_*` FEED blocks, searched by the
  same mechanism terms and the r3-r7 lineage.
- design/SPEC/task surfaces:
  qbf1 packet schema, qbt1/qbt2b receipts, active lane registry, harness/task
  bridge references, and bounded task-ledger filename/content searches.

Beyond the charter seeds, recall found two important scope qualifications:

1. `ddm_bs2` showed a fixed start-budget can remain slack and license give-back,
   and its ratchet was added to expose/close that failure. Change to this plan:
   this receipt states explicitly that r7 protects the absolute measured born
   envelope, not every later improvement. A ratchet is not silently claimed.
2. `ddm_lt1` warned that an aggregate class scalar was invalid for a different
   PR130 directed-edge problem. Change to this plan: r7's two aggregate werr
   constraints are claimed only for the present qbt vehicle and the measured
   r6 class-erasure failure. They are not generalized as an edge-complete
   protection law.

No qbt-specific registered canonical equation superseded the chartered law.
No additional active r7 lane claim or independently owned r7 consumer was found
in the searched active-lane/task surfaces.

## Boundaries and family disposition

- **MEASURED:** unit behavior; default-off tensor identity; pin/refusal laws;
  n1 real frozen-scorer constraint engagement; λ persistence through
  checkpoint/resume; deterministic archive bytes; storage shortfall.
- **NOT MEASURED:** n32 endpoint, 5,000-step feasibility, population behavior,
  exact archive score, contest CPU/CUDA score, or frontier movement.
- **INSTANCE scope only:** pair-62 n1 smoke. It licenses mechanism and custody,
  not an efficacy verdict.
- **FORMULATION still open:** primal-dual Lane/Movable protection on the fixed
  r5-born qbt basis. Its decisive row is the governed n32 run.
- **DEAD by prior evidence:** more birth-side machinery here, unweighted CE for
  Lane birth, class weights as the margin-stage protection, and per-edge label
  injection as a replacement for this constraint.

Own-vehicle frontier: **S = 0.14811799921260607 @ 180,215 B
`[contest-CUDA T4 n600]` (gb1), UNMOVED.** This arm produced no authority score
and did not advance the sub-0.12 goal.
