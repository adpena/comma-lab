# DDM GF3 time-conditioned factorized-field closed-form pricing — 2026-09-09

## Decision

**PRICING-CLOSED. verdict_scope: FORMULATION** — the discrete `L ∈ {2,3,5,10,20,50}` family of one
full 384×512 categorical field per GOP plus one global integer translation `(dy,dx) ∈ [-12,+12]^2`
per frame, with generic repairs charged at the charter's optimistic `0.2909 B/site` comparison rate.
No row passes either `{packet ≤ 71,404.5 B AND certified mismatches ≤ 46,804}` or the optimistic
`packet + repair charge ≤ 85,020 B` gate.

The closest lower-bound pricing row is **L=10 at 97,813.3582 B**, still **12,793.3582 B over** the
85,020 B replacement cap. The best physically achieved packet-plus-domain-residual row is **L=50 at
347,500 B**, **262,480 B over**. No trainer was built and no scorer was invoked.

Axis: `[macOS-CPU scorer-free exact field measurement, n600]`. `score_claim=false`; `pointer_moved=false`.

## Closed-form and physical table

The certified bound is the retained GF2-form histogram-total-variation proof. The observed fit is an
achievable one-sweep upper bound, never an optimum. Packet and physical residual bytes are selected from a
real Brotli q11 / zlib-9 / LZMA2-extreme coder race and decoded again from the persisted selected payload.
The optimistic total uses `(bound - 46,804) × 0.2909 B/site`; it is deliberately more favorable than the
measured residual and is **not** a physical-coder claim.

| L | certified lower-bound mismatches / 105,408,000 | observed fit mismatches | real packet B | physical domain residual B to 46,804 | physical total B | optimistic bound-price total B | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 154,586 | 731,190 | 157,124 | 303,500 | 460,624 | 188,477.7838 | PRICING-REFUSED |
| 3 | 162,633 | 778,199 | 137,832 | 338,848 | 476,680 | 171,526.6561 | PRICING-REFUSED |
| 5 | 208,152 | 1,006,260 | 78,428 | 354,192 | 432,620 | 125,364.1332 | PRICING-REFUSED |
| 10 | 273,402 | 1,281,107 | 31,896 | 355,124 | 387,020 | **97,813.3582** | PRICING-REFUSED |
| 20 | 409,920 | 1,631,346 | 14,820 | 348,580 | 363,400 | 120,450.4444 | PRICING-REFUSED |
| 50 | 661,753 | 2,266,749 | 5,280 | 342,220 | **347,500** | 184,168.6641 | PRICING-REFUSED |

LZMA2-extreme won the physical packet at L=2,3,5,10,20; Brotli q11 won at L=50. The shipping HPAC
RC64 was not run: it requires receiver-model probabilities and therefore prices a different object. A static
or 32-bit coder was not relabelled RC64. All three generic real coders and their byte-identical repeats remain
retained for every packet and residual.

The prior-law quantitative claim `bound(L=10) ≥ 400,000` is **falsified**: the certified result is 273,402.
Its qualitative rate/accuracy tradeoff is confirmed. Short GOPs have permissive error bounds but packets that
are too large; longer GOPs have cheap packets but certified error that is too large.

## Certified theorem

For target frames `T_i,T_j`, allowed translations `a,b`, and any one shared GOP field, the two frames'
combined errors on the common 360×488 interior are at least

`TV(hist(T_i shifted by a), hist(T_j shifted by b))`.

For each frame pair, GF3 takes the exact minimum over all `625×625` allowed translation pairs. Within each
GOP it evaluates six deterministic disjoint pairings (fixed extremes and five center-histogram class orders),
sums the pair bounds, and retains the maximum. Every pairing bound applies to every shared field and every
global-offset assignment, so their maximum remains a lower bound. Odd GOPs discard one frame; every row
discards all border errors. These relaxations can only weaken the bound.

Primary and independently repeated certificates are byte-identical at every L. The audit is a bound on the
global optimum; it does **not** claim to solve or observe that optimum.

An earlier retained proof leg under each physical `L_*` directory grants every frame a different translation
at every lattice cell before taking a plurality. That cellwise relaxation is mathematically valid but too weak
for adjudication: it would have admitted L=10 on `17,067` mismatches. It is retained as a transparent
preliminary result and **superseded for the decision** by `certified_globality_v2`; it is not deleted,
overwritten, or presented as the load-bearing bound.

## Exact object and source re-derivation

The measured object is the GF2 exact n600 DALI-lineage argmax field:

- path: `/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/converged_v3/retained/source_afr1_jbp1_field.u8`
- bytes: `117,964,800`
- SHA-256: `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`
- geometry: `600×384×512`, five categorical classes
- lineage gate: `verify_gt_lineage(axis="contest_cuda", declared_lineage="dali") → VERIFIED`

Every charter number was re-read at its source:

| source fact | re-derived value | role here |
|---|---:|---|
| GF1 generator packet | 47,603 B | `1.5× = 71,404.5 B` direct packet cap |
| GF1 mismatch count | 1,325,033 | RN1 requires a 28.31× reduction to 46,804 |
| GF1 exact correction stream | 385,448 B | generator byte ratio cannot be separated from fit error |
| GF2 certified L=600 bound | 923,953 mismatches | prior rigid-static closure; not rerun |
| GF2 observed fit / packet / residual | 3,072,488 / 451 B / 335,096 B | retained ancestor and domain-order reference |
| OL1 NerVast projection | 18,994 B | projected shared-scene price only, not a measured GF3 row |
| LTG1 exact Lane packet | 233,262 B = 11,148 topology + 221,717 shape + 397 container | exact boundary alternative |
| BLP1 predictor weights | 60,191 B before residual | already 24,147 B over GF1's 36,044 B Lane stream and 38,492 B over the 21,699 B door |

LTG1 and BLP1 are comparison obligations, not addends to the whole-field GOP packet: adding them would
double-count Lane, which is already present in each full categorical field. Their measured prices reinforce
the refusal rather than provide a missing cheap boundary component.

## Physical receiver and custody

The physical member for each L is a zero-offset exact GOP plurality, followed by exhaustive best global
integer offset per frame for that fixed field and one exact plurality update at the fixed offsets. The final
field and offsets are packetized, real-coded, reopened from the retained selected payload, parsed, rendered,
and matched to the retained observed decode. A dense residual orders all 600 symbols for each `(y,x)`
contiguously, corrects a deterministic mismatch prefix down to exactly 46,804 remaining mismatches, and is
likewise decoded from the retained selected payload. This is an achievable upper bound and an exact receiver
round trip; it is not a coordinate-descent optimum.

Retained store:
`/Volumes/VertigoDataTier/pact/ddm_gf3_time_conditioned_pricing/`

- final audit result: `certified_globality_v2/RESULT.json`, 16,728 B, SHA-256
  `5a539a7f4e1dcf4d70b15a3514547185cf1d29b20b70b75bb3f0fb3a8737e0c1`
- final audit manifest: `certified_globality_v2/MANIFEST.json`, 7,050 B, SHA-256
  `d6e166bbad53e3fd32f3c772bc124300a963ebf977fed95c9ba248deb57822df`
- audit manifest: 27 entries, 4,302,611 B, independently rehashed with zero failures
- full retained tree: 200 files, 1,708,196 KiB by `du -sk`; no payload was deleted or moved
- each physical L manifest: 25 entries; all entries independently rehashed before the final decision
- deterministic repeats: source reachability cache, both proof certificates, fitted member, raw packet,
  every coded payload, packet decode, raw residual, and residual decode
- resource maxima: physical runs peaked at 3,437,182,976 B RSS; the final audit peaked at 115,277,824 B;
  every numerical-library thread setting was `1`

No scorer, training, Modal, Metal/MPS, `upstream/` mutation, submission mutation, or frontier mutation occurred.

## RECALL EVIDENCE

Searches were by content over the full required corpus, not just charter names:

- `rg -n -i "factorized[_ -]?4d|k-?planes|time[- ]conditioned|shared[- ]field|boundary[- ]motion" .omx/research .omx/state src/tac/canonical_equations docs`
- `rg -n "LTG1|BLP1" .omx/research/`
- `rg -n "ddm_gf2|generator_form_fit_error_entanglement|decoder_derivable_ideal_savings_ceiling|procedural_predictor_plus_residual" .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_* .omx/state`
- `.venv/bin/python tools/list_canonical_equations.py --json`
- exact source reads of GF1 at commit `5b884ec957`, GF2 memo/runner/final message, RN1, OL1, LTG1, and BLP1

Beyond the charter seeds, the search found
`.omx/research/factorized_4d_kplanes_observability_20260630T180753Z.md`, which documents K-Planes/HexPlane
static-time factorization and its smooth-dense-field assumptions, and
`.omx/research/multiscale_lowfreq_seg_repair_verdict_20260611.md`, where a cheap shared correction field
cancels under direction disagreement. This changed the plan by making the negative boundary explicit:
time-conditioned low-rank planes, non-rigid warps, and boundary atoms remain outside this result; none is
silently killed by the full-field GOP theorem. The second memo was supporting negative signal only and did
not replace the exact GF2-form proof.

The equation inventory contains `decoder_derivable_ideal_savings_ceiling_v1`,
`generator_form_fit_error_entanglement_v1`, and
`procedural_predictor_plus_residual_correction_savings_v1`. GF3 is an empirical anchor on the first law:
the optimistic charge is used only to refuse a build and never promoted to a physical byte or score claim.

## Disposition and boundary

No build charter fires. All six grid rows are `PRICING-REFUSED`; the scorer step is `FOLDED` because the
closed-form gate failed. L=600 remains closed by GF2 and was not rerun. Coordinate-descent-as-optimum,
consecutive-IoU extrapolation, and generic residual repair of a rigid-static field remain closed.

`QUEUED-CONDITIONAL` — owner: MAIN; consumer store:
`/Volumes/VertigoDataTier/pact/ddm_gf3_time_conditioned_pricing/certified_globality_v2/`; fire trigger: a
**different** representation outside this verdict scope—time-conditioned low-rank factors, non-rigid warp,
or parametric boundary atoms—first supplies a certified current-n600 bound plus a receiver-closed real packet
showing either `{≤71,404.5 B and ≤46,804 mismatches}` or `packet + physical domain residual ≤85,020 B`.
Until that trigger, do not build or train another full-field GOP member.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]
