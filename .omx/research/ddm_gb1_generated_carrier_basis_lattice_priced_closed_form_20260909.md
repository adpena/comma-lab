# ddm_gb1: pricing is blocked by an invalid bound premise and missing live derivatives

`[no-triality] [p0-ledger-ok]` · owner `ddm_gb1` · 2026-09-09.
`research_only=true`, `score_claim=false`, `promotable=false`.

**The requested lattice-bound table is NOT completed: 0/36 numerical rows,
0 new PoseNet pairs, 0 solver launches, 0 candidate builds, 0 seals.** The audit
and retained source evidence are complete. No generated family is closed.
This arm did not move the exact pointer. MAIN's concurrent pc2 promotion moved
it to **S 0.13882326433317044 @ 181,373 B [contest-CUDA T4 n600]**.

The charter cannot be executed as a valid proof by substituting a Jacobian
projection and rounding residual for a lower bound. Its requested replacement
Jacobian job also requires the scorer slot, which the common contract does not
assign to this arm. The scorer-free work is retained; the remaining work has a
fire order in the canonical task store. This is a blocked charter, not a negative
result about Zernike, steerable pyramids, or Gabor carriers.

## Measured bytes and source-verified corrections

Fresh parse of the live pc2 archive, through `ddm_up3.parse_shipped_body` and the
named receiver's readers; axis **[exact local byte arithmetic; scorer-free]**:

| Archive component | Bytes |
|---|---:|
| ZIP overhead | 100 |
| RX1 header | 14 |
| HPAC stream | 12,112 |
| Semantic stream | 30,246 |
| Carrier stream | **18,580** |
| Token tail | 120,321 |
| Total | **181,373** |

Archive SHA-256:
`e138ee097905902ad6e1d49841b2ff2f043736298079f66c0a1628031bcd8372`.
Runtime source:
`/Volumes/VertigoDataTier/pact/ddm_pc2_carrier_kwidth_rankcut/rebase_scales_resolve/candidate_runtime`.
The pointer start/end checks passed on this archive. The first attempt against
rc2 correctly refused after the concurrent pointer move. No staging or seal was
attempted. The runtime is a parse source here, not a newly validated scoring runtime.

The restored carrier body is 18,872 B: 6 count bytes + 96 scale bytes + 40 metadata
bytes + 12,277 Huffman basis bytes + 6,424 Rice bytes + 29 tail bytes. Those are
RESTORED field lengths, not their separately attributable final compressed costs.
The base CAP1 code table is 600x12 with range [-142,155]. The audit does not claim
compensation-overlay application, rendered-pixel identity, or a new d_pose.
The carrier acts on frame 0 only; d_seg is carried structurally and was not measured.

| Charter number or claim | Verification and scope |
|---|---|
| 18,621 B live carrier | Correct for the earlier rc2 body; the live pc2 body is now **18,580 B**. |
| V4 saves 12,043 B | pc1 `rate_v4/RATE.json`: 179,982 - 167,939 = 12,043 B on archive `08ec8533…`. Historical price; not a generated-family price on pc2. |
| DCT coverage 2.49% | pc1 `coverage/COVERAGE.json`: **0.02491304829617269**. Its own note explicitly says field coverage is not a d_pose claim. |
| d_pose cap 1.694e-5 | Verified in pc1 and pr1's source memos; originally a same-object/25%-seg-cut affordability ceiling. Retained as the charter's cap, not a universal physical threshold. |
| pc2 Jacobians | `jacobian_leverage.json` covers pairs **0,100,200,300,400,500**. Only column aggregates are retained there, not the 6x12 tensors or residual vectors. |
| pc2 rank-8 floor: 39 pairs, 48.9x | Original memo reports 39 rows. The current retained shards contain **40 distinct pairs**, sum **0.3788886483898605**, floor **0.0006314810806497675** for that recorded solver output. At its historical bar 1.292442e-5 this is still about 48.9x. It bounds that output's population mean, not every possible solver or basis. |
| Half-step fails paired | The eight retained paired ratios range from **0.1426885 to 2.7260151** (plain/half d_pose). This is heterogeneous, not a transferable penalty constant. |
| Eighth-step about 2,500x worse | Pair 0 is **2542.4937535893564x** worse than its plain cut; pair 25 is 0.9706x and pair 50 is 312.76x. Instance evidence, not a monotone law. |
| More solving: 17/7,200 coordinates, 0.076% | Reaggregated all **600** retained resolve rows: **17 coordinates**, **9 pairs improved**; mean 5.0939863022125565e-6 -> 5.090131601413916e-6. Historical cpu_torch/DALI-GT receipt, not a new scoring run. |
| pr1 demands beyond +/-2 and lattice | Reaggregated 600 distinct pairs: **600/600** demand >2, **58/600 (9.6667%)** demand >4095, median **444.14484** units; receipt `PR1_DEMAND.json`. These demands belong to the **damaged step600 renderer**, not live pc2. A GN demand is not a proof that the finite lattice has no other solution. |

Every cited reaggregated d_pose row has its original per-pair JSONL or NPZ copied
with SHA-256 under `source_verification/sources/`. No old candidate was rerun.

## Why the proposed bound does not follow

1. **The supplied derivative is relaxed.**
   `experiments/ddm_up2_shipping_pose_solve.py::_round_ste` preserves hard-round
   forward values and substitutes an identity backward. `jacobian_and_residual`
   calls that differentiable path. The exact rounded map is locally constant
   almost everywhere; this STE derivative is useful for proposals but provides
   no uniform error certificate after replacing the atoms.

2. **A rounded or polished point supplies an upper bound on a minimum.**
   For affine loss `q(z)=||r+A*Delta*z||^2/6` and any feasible rounded point z0,
   `min_z q(z) <= q(z0)`. Local polishing preserves that direction of inequality.
   A span projection is a lower bound for the affine problem alone; adding the
   observed rounding penalty does not convert it to a lower bound on the lattice
   optimum. Neither number bounds the nonlinear realized map without an error
   certificate. A first-order jet does not determine a function away from its
   expansion point: a smooth correction with zero value and derivative there can
   change another point's residual arbitrarily. The actual rounded chain makes
   this gap more severe, not less.

3. **Old coefficient Jacobians cannot identify a new spatial family.**
   Writing `J_old = H*B_old`, all operators `H + U*(I-B_old*B_old^+)` agree on
   the old columns. They can disagree on a new atom outside `span(B_old)`.
   Therefore even n600 old 6x12 tensors cannot determine `H*F` for arbitrary
   Zernike, steerable, or Gabor atoms. Need derivatives in a sufficiently rich
   spatial parameterization, or direct family directional derivatives, with the
   receiver normalizations and selector included. No such live bank was found
   in the scoped source inventory.

4. **The +/-2 polish is not the solver's global reach.**
   `jg5.refine_pair` alternates GN steps with repeated coordinate moves of
   {-2,-1,1,2}. Successful polish steps repeat from the new center, and GN may
   move much farther. A box of radius two around the projected start is not the
   solver's reachable set. Nor are pc2's paired loss ratios an absolute n600
   penalty applicable to another basis.

5. **Fitted SVD vectors are not free generic atoms.**
   SVD of stacked 6x12 Jacobians yields 12-dimensional coefficient directions;
   turning them into spatial atoms still needs `B_old`. SVD of spatial
   Jacobians produces video/scorer-derived spatial vectors. Both must transmit
   their fitted information. A seed for the SVD algorithm is not that information.
   A valid generic alternative selects parameters/seed for a fixed generator,
   counts the selected description, and regenerates it without video or scorers.

Maximizing a stacked singular value also does not maximize the worst per-pair
smallest singular value. Atom and coefficient scales must be normalized before
comparing either statistic. The int12 range is **[-2048,2047]**, not symmetric
[-2047,2047]. These are additional requirements on a successor's price model.

## Equations and the preregistered margin

The rule was written before the audit in `ddm_gb1_prereg_20260909.md`:

    T(b) = min(1.694e-5, (sqrt(10*d_base) + 25*b/37545489)^2/10)
    b >= 2000; L + 2*P <= T(b).

`P` must be a matched absolute d_pose penalty, not a ratio transferred from pc2.
Neither P nor a nontrivial certified L is available. Passing a lower-bound test
is a necessary possibility screen, not proof of achieved performance.

A mathematically valid conditional certificate would be:

    ell_i <= min_{z in int12^K} ||r_i + A_i*Delta*z||_2
    ||actual_residual_i(z) - (r_i + A_i*Delta*z)||_2 <= epsilon_i for every z
    d_pose >= sum_i max(0, ell_i-epsilon_i)^2 / 3600.

This permits a certified lattice relaxation followed by a uniform realized-map
remainder. Both conditions are owed. A full-rank continuous projection often has
zero residual, making its lower bound vacuous. The only unconditional bound
established here is nonnegativity, zero, retained explicitly as a DERIVED
tautology with 600 per-pair zeros and **not** used to admit or refuse anything.

The canonical equation `pose_carrier_basis_rate_fidelity_exchange_v1` already
excludes **any d_pose prediction** from its validity domain. Its new gb1 anchor
records the live 18,580 B parse and these source-inspection limits; it does not
launder a projected distortion into an empirical result. Registration evidence
is exported to `ddm_gb1_registry_events_20260909.json` for replay/custody.

## Requested matrix: blocked work, not a numerical bound table

Every cell below represents the three preregistered step multipliers {1,1/2,1/8}.
The 36 explicit rows are retained in `input_audit_v2/pricing_rows.json`.

| Family | K | Absolute Delta | Counted B | Pose lower bound | Net Delta S |
|---|---|---|---|---|---|
| Zernike | 6,8,12 | unknown | unknown | not certified | unknown |
| Steerable pyramid | 6,8,12 | unknown | unknown | not certified | unknown |
| Generated Gabor | 6,8,12 | unknown | unknown | not certified | unknown |
| Jacobian-fitted SVD | 6,8,12 | unknown | unknown; fitted atoms counted | not certified | unknown |

An exact Rice estimate requires the ordered projected coefficient codes and
their AR predictor/bias residuals. A magnitude histogram of unpredicted codes
does not determine that length. CABAC adds ordered context adaptation; histogram
entropy alone does not determine the shipped bitstream length. The final
container size remains a measured encode requirement after admission. No
generated coefficient sequence, rate estimate, or chosen optimal family exists
in this delivery. The charter's prior-law prediction is **UNTESTED**.

## RECALL EVIDENCE

Full-corpus content searches are retained with exact argv, result paths, return
codes, and errors in `source_verification/RECALL_SEARCHES.json`:

- Research memos/JSON/JSONL: `generated.{0,25}(carrier|basis)|carrier.{0,30}(jacobian|lattice)|Zernike|Gabor|steerable`.
- Design docs, canonical research index and `sub015_DAG_*`: `carrier.{0,40}(basis|lattice)|generated.{0,20}basis|Zernike|Gabor|steerable`.
- Task status, P0 ledger, lane registry: `generated.{0,20}basis|pc2_carrier|ddm_gb1|carrier.*[Jj]acobian`.
- All **477** canonical equation records via `tools/list_canonical_equations.py --json`, retained in the arm tree.
- pc1/pc2 artifact trees: name inventory for Jacobians/PREP; inspected pc2's Jacobian producer and its actual six-pair JSON.

The first exact index filename query missed its dated filename. The corrected
search of `CANONICAL_RESEARCH_INDEX_20260629.md` returned no matches for the
same surface expression (`INDEX_RECALL.txt`, rg exit 1). Research content search
returned 130 paths; the source list records that bounded scope.

**Beyond the charter seeds:** jc1's
`ddm_jc1_carrier_jacobian_posemetric_refit_20260816.md` explicitly labels its
derivative STE-relaxed and reports linear predictions missing realized
refit ratios by 134-1,065x on an older body. Its n600 bank is coefficient-space
and stale for this task. `ddm_ra3_subspace_trust_region_refit_20260816.md` likewise
limits its averaged pullback to a local relaxation. These changed the plan from
computing an apparent proof to checking whether a proof is identifiable at all.
The canonical equation's express exclusion of d_pose predictions independently
supports this change. DAG hits on Gabor/directional features concern other
vehicles and do not establish a live PoseNet lattice result. No matching live
n600 spatial derivative certificate was found in these searched sources.

## Retention, verification and boundaries

Durable root:
`/Volumes/VertigoDataTier/pact/ddm_gb1_generated_carrier_basis/`.

- `input_audit_v2/AUDIT.json`: final source/retention manifest, live archive copy,
  every parsed byte field and ndarray, source snapshots, exact command, hashes.
- `source_verification/HISTORICAL_ROWS.json`, `PR1_DEMAND.json`: recomputed
  historical aggregates plus hashed per-pair source copies.
- `input_audit/`: earlier completed audit retained; v2 additionally captures
  its producer source and explicitly labels base codes.
- `equations_recall_20260909.json`: complete registry recall; routing receipt
  records its lossless move into the charter-owned tree.

`experiments/ddm_gb1_pricing_input_audit.py` is an input/custody audit, not a
generated-basis implementation. It parsed the real archive, refused stale
pointer custody, and verified every retained file via `--resume-from`.
One process, one numeric thread; initial load 16.49, Vertigo free 11 GiB,
APDataStore free 47 GiB. Only small archive/metadata outputs were created.
No raw inflation, scorer cache, training, detached job, network dispatch,
Modal call, or deletion. Upstream, submissions, shared index and other arms'
trees were not modified by this arm. No memory was written.

Two Python review passes covered all five current functions. The real retained
manifest passes resume validation; an intentionally corrupted hash in a copy of
that manifest is refused (`verification/CONTROLS.json`). The source parser also
refused the actual stale rc2 archive. These controls validate the audit's custody
behavior; they supply no generated-basis or d_pose validation.

All six integration hooks are research-only/N/A: no sensitivity measurement,
Pareto move, bit allocation policy, or deployable candidate exists; the equation
anchor and task fire order are the consumers. The proof-vs-surrogate distinction
is the disambiguation; no duplicate probe is launched to settle it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer `.omx/state/canonical_task_status.jsonl`, task `ddm_gb1_generated_carrier_basis_lattice_priced_closed_form_20260909::VALID_BOUND_INPUTS`.** On harvest of this receipt, resolve the proof gate first: supply a certified lattice/remainder construction or explicitly revise the charter to permit a surrogate screen. Only after that gate and MAIN's scorer-slot allocation, collect retained live n600 spatial/directional derivatives in chunks <=120 with <=2 CPU processes; re-read the pointer, price all 36 cells with counted fitted information, apply the written margin, and admit at most one family for the original full build/measure/seal chain. No solver may fire from the present zero bound or null matrix.

## LIVE-HYPOTHESES

- A generic Zernike, steerable or Gabor generator with counted selected parameters
  may supply better lattice placements. Poor reconstruction of the old field
  does not exclude a different witness with the same six pose outputs.
- A compact counted Jacobian-informed transform may help if its lattice benefit
  exceeds its payload cost. This remains plausible; exact fitted vectors are
  neither generic nor free.

## DEAD-ENDS

- **FORMULATION:** using rounded/locally polished STE residuals as a certified
  lower bound. The inequality is reversed for feasible candidate values and the
  nonlinear remainder is uncontrolled.
- **FORMULATION:** regenerating video-fitted singular vectors from only an
  algorithm seed. The receiver lacks the fitted information.
- **INSTANCE:** reusing the pc2 six-pair aggregate as an n600 spatial Jacobian.
  It has neither the population coverage nor the necessary variable-space data.
- **FORMULATION:** using +/-2 as a global reach limit or pc2's loss ratios as a
  universal lattice tax. The source solver repeats moves and paired effects
  have both signs. These closures do not close the generated carrier families.

Own-vehicle frontier read at this audit: **S 0.13882326433317044 @ 181,373 B
[contest-CUDA T4 n600], unchanged by ddm_gb1.**
