# ddm_gf2 — static/dynamic-factorized generator form: CEILING-REFUSED

Owner: MAIN. Charter: `.omx/research/charters/ddm_gf2_static_dynamic_generator_form_20260903.md`.
Operating contract: `docs/operating_manual_craft_handoff.md` plus `.omx/tmp/codex_runs/_common_contract.md`.
Axis: `[macOS-CPU scorer-free exact field measurement, n600]`. Score claim: false. Pointer moved: false.

Receipts:

- Fixed-point fit and real-coder payloads: `/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/converged_v3/`
- Global lower-bound audit: `/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/globality_audit_v1/`
- Fit runner: `experiments/ddm_gf2_static_dynamic_generator_form.py`
- Globality runner: `experiments/ddm_gf2_alignment_globality_audit.py`

## Result first

| quantity | n600 result | authority |
|---|---:|---|
| Best observed shared field + independently exhaustive shifts for that field | 3,072,488 mismatches / 117,964,800 = 2.60458% | MEASURED upper bound on the unknown global optimum |
| Certified lower bound for every shared categorical field and every per-pair integer shift in `[−12,+12]^2` | **923,953 mismatches** | MEASURED, global for the declared family |
| Largest mismatch count compatible with the charter's optimistic repair gate | 292,264 | exact arithmetic |
| Certified excess over the gate | **631,689 mismatches** | exact subtraction |
| Typed decision | **CEILING-REFUSED** | FORMULATION scope below |

The earlier fit reached a coordinate-descent fixed point and exhaustively selected the best shift for its final
retained field. It did **not** solve the joint field/shift global optimum. That fixed point is therefore no longer
the authority for refusal. The independent audit closes the epistemic gap: on the 360×488 common interior it
records the five-class histogram under all 625 allowed shifts for every frame. For any two frames rendered from
one shared categorical field, their combined errors are at least the total-variation distance between their
shifted target histograms. The audit takes the exact minimum over all 625×625 shift pairs and sums it over 300
disjoint frame pairs. Border errors are discarded. The resulting 923,953 is thus a lower bound for **every**
field and shift assignment in the declared family, not a local-fit result. The exact optimum remains unmeasured
and lies between 923,953 and 3,072,488; that uncertainty cannot change the ceiling decision.

The lower-bound histogram tensor was independently materialized twice and was byte-identical: 7,500,000 B,
SHA-256 `6c13293602abc5868f2198da4928111e64b56fd0b7dad51ff27474cdcc106818`. The selected pairing's exact
bound was also byte-identical across the repeat; its per-pair lower bounds range from 256 to 8,240, with median
2,760. The selected pairing was `extreme_center_class_3`; even the weaker interval-only bound for that pairing
was 866,307. The audit took 25.45 s and peaked at 231,047,168 B RSS, below the 20 GB cap. It invoked no scorer,
Modal, Metal, or MPS.

## Static ceiling and per-class mismatch

The fit gave the static term a full 512×384 categorical lattice, which is strictly more flexible than a GF1 SDF
or born-small generated static term. Its retained final shift is the exact best member of the declared rigid
family for that retained field. These class counts describe that observed field; they are not a classwise
decomposition of the certified global lower bound.

| target class | observed mismatches |
|---|---:|
| Road | 486,108 |
| Lane | 691,095 |
| Undrivable | 400,352 |
| Movable | 1,394,814 |
| MyCar | 100,119 |
| **Total** | **3,072,488** |

Unaligned per-site modal mismatches were 3,126,748. The retained fixed point reduced that count by only 54,260
(1.735%). Exact empirical residual entropy was 1,578,452.26 B when conditioned by lattice site and
2,880,286.44 B when conditioned only by rendered static class.

## Ceiling arithmetic

The charter grants the entire 71,404.5 B packet to dynamic repairs and charges zero bytes for the shared field,
rigid offsets, coder tags, and container framing:

`floor(71,404.5 / 0.2909) = 245,460 repairs; 245,460 + 46,804 = 292,264 mismatches.`

The certified lower bound is 923,953, or 3.161× that limit. Even at the optimistic generic rate, reducing only
the certified lower bound to 46,804 mismatches costs at least
`(923,953 − 46,804) × 0.2909 = 255,162.6441 B`, before any static, warp, or container bytes. That is already
3.573× the full packet cap.

## Packet, residual, and failure split

| retained section / route | selected coder bytes |
|---|---:|
| Shared categorical field | 258 |
| 600 rigid `(dy,dx)` offsets | 193 |
| Static + rigid packet | **451** |
| Domain-matched pixel-time residual | **335,096** |
| Generic frame-raster residual | **461,188** |
| Static + rigid + domain residual | **335,547** |
| Replacement cap | **85,020** |

The domain-matched route is 250,527 B over the replacement cap. All raw fields, offsets, decodes, residuals,
coder outputs, repeats, checkpoints, and manifests are retained in the fit store. The globality audit adds its
histogram, translation, pairing, repeat, result, and manifest payloads; its manifest has 9 entries totaling
15,037,143 B and independently re-hashed without error.

Failure is in the **static scene assumption**, not in the byte cost of encoding the chosen static field or its
rigid offsets. A single field plus rigid motion leaves at least 923,953 errors globally. The observed fit leaves
most error in Movable, Lane, and Road targets. No sparse dynamic events were fitted because the charter says to
stop when the ceiling is refused. Consequently there are no three 47–71 KB fit points and no packet→mismatch
curve: those builds were forbidden by the ceiling-first gate, not attempted and failed.

The 335,096 B domain residual is a physical retained coder result for the observed fit. The 0.2909 B/site line
is only the charter's optimistic comparison rate; it is not a claim that every parametric dynamic mechanism must
cost that much.

## Decision and boundaries

**CEILING-REFUSED. verdict_scope: FORMULATION** — one categorical field shared across all 600 frames, arbitrary
per-pair integer translations `(dy,dx) ∈ [−12,+12]^2`, and corrections charged by the charter's optimistic
0.2909 B/site gate. The result also refuses any less expressive GF1-family static term under the same rigid and
repair assumptions.

This does **not** close multiple counted GOP fields, time-conditioned low-rank factors, non-rigid warps,
parametric boundary-motion atoms, or a dynamic generator that repairs many sites per coded parameter. None of
those objects was fitted here. The prior memo's derived per-GOP closure is withdrawn: consecutive-pair IoU is
not a lower bound on the jointly optimized multi-field mismatch count.

Canonical-equation relation: GF2 is an in-domain use of
`decoder_derivable_ideal_savings_ceiling_v1` from
`tac.canonical_equations.ddm_lv3_current_arc_laws_20260901`. The ceiling is used only to refuse a build after a
valid global lower bound; it is not presented as a physical byte claim. The sister
`generator_form_fit_error_entanglement_v1` law remains non-transferable because its 2.178× ratio is inseparable
from the measured GF1 fit error.

## RECALL EVIDENCE

Bounded repo recall searched `.omx/research/`, `.omx/state/`, the canonical equation registry, the canonical
research index, the current DAG, and task ledgers for `static/dynamic`, `shared scene`, `per-site modal`,
`boundary motion`, `NerVast`, `71,404.5`, `292,264`, `0.2909 B/site`, and `ddm_gf2`.

- Charter seeds consumed: GF1 capacity, RN1 inequalities, OL1 online scan, BZ2D amplification, DDS1 geometry,
  LTG1/BLP1 Lane costs, and MC1 motion compensation.
- Beyond the seeds, `factorized_4d_kplanes_observability_20260630T180753Z.md` keeps a time-conditioned
  static/time factorization plausible, but it does not price a current byte-closed GF2 packet.
- `frozen_partition_topology_ego_deformation_20260623.md` reports low-dimensional coarse boundary deformation
  alongside 62–77% non-ego residual and fine-island high-rank structure. It supports testing structured motion,
  but contradicts treating a global rigid warp as sufficient.
- `order_exploit_rate_budget_20260627T053101Z.md` records prior failures of global integer shift and block-shift
  forms plus a 373–406 KB temporal literal. This raised the bar against another rigid-motion build; the new
  global lower bound now closes that exact family on the current retained field.
- The canonical research index contained no additional direct GF2 result in the searched scope. This is bounded
  absence, not a claim that no external or unindexed result exists.

## NEXT_IF_RESUMED

- **QUEUED-CONDITIONAL** — owner: a generator-form arm assigned by MAIN; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/time_conditioned_successor/`; fire trigger:
  closed-form arithmetic on the current n600 field predicts a real-coded packet ≤ 71,404.5 B and either ≤ 46,804
  mismatches or packet + domain residual ≤ 85,020 B, while explicitly pricing the LTG1/BLP1 boundary obligations.

## LIVE-HYPOTHESES

- A counted small-GOP or time-conditioned factorized field could escape the single-static-field theorem because
  it changes the shared sufficient statistic over time. K-Planes/HexPlane/DS-NeRV-style separation is plausible
  for non-rigid scene evolution, but remains untested here and must survive real 47–71 KB packet pricing.
- Parametric boundary-motion atoms could repair many correlated sites per byte, escaping the generic 0.2909
  B/site arithmetic. Coarse boundary motion has measured low-dimensional structure, but any build must also pay
  for the measured 62–77% non-ego component and fine-island residue rather than assuming them away.

## DEAD-ENDS

- One shared categorical field plus any allowed per-pair integer translation is closed: every member has at
  least 923,953 mismatches, 631,689 above the ceiling gate.
- Treating the coordinate-descent fixed point as the global best is closed: it is only an observed upper bound;
  the independent histogram-TV theorem is the refusal authority.
- A generic or literal residual on the retained rigid-static fit is closed: the better domain-matched payload is
  335,096 B, and static + offsets + residual is 335,547 B versus the 85,020 B cap.
- Declaring small-GOP fields closed from consecutive-pair IoU alone is closed as an inference: IoU does not bound
  the jointly optimized multi-field result.

Own-vehicle frontier: **S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]** — GF2 did not move it.
