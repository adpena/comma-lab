# DDM IS1 — full inverse solve to the end

`research_only=true` · `execution_allowed=false` ·
`evidence_axis=[macOS-CPU frozen-scorer advisory]` · `score_claim=false` ·
`promotion_eligible=false` · `main_review_required=true`

Lane:
`lane_ddm_is1_full_inverse_solve_to_the_end_20260724`

Pointer:
`0.1910828242 [contest-CPU] UNMOVED`

## Outcome

The right objective is not the #613 box and not minimum bytes at a fixed
distortion point. It is:

\[
\min_{f,z}\left[
100d_{\rm seg}(R\,D_f(z))
+\sqrt{10d_{\rm pose}(R\,D_f(z))}
+25B_{\rm real}(f,z)/37{,}545{,}489
\right],
\]

where the free choices are jointly **which score-equivalent/tolerance-region
solution** \(D_f(z)\) to emit and **how to describe it**. The real-coded
distortion knee is an output.

The prospective path ranking is:

1. **(d) score-quotient functional representation** — best aligned family:
   fit a compact function only to the scorer plane/at-risk margins and
   at-most-six-dimensional Pose statistics, with exact scorer and real coder
   in the objective. Count parameters, temporal latents, and interface
   placements. Human fidelity and scorer-null camera detail are absent by
   construction.
2. **(b) solve-as-oracle description** — mandatory dense supervision and
   empirical testbed. Describe the solved object or its diff; do not re-predict
   and repaint the scene. The raw exact residual is rate-dead, but it identifies
   the content that (d) and (a) must compress.
3. **(a) metric-active MS2 typed quotient solve** — the deterministic
   comparator/compressor after oracle decomposition and #669c re-homing. It has
   never produced a metric-active row.
4. **(c) #366 descent** — not a standalone representation family. It may be
   the fitting engine inside (d), or a residual finisher after (a)/(b).

No one has measured which reaches the target cheapest because none has emitted
a same-currency receiver-closed n600 candidate. The ranking is actionable but
prospective, not a score forecast.

**Training-necessary residual: empty on current evidence.** The 25 stopped RG3
rows prove a lower bound on missing counted content in the current vocabulary;
they do not prove that content requires a learned representation. Under (d),
training has an honest role as the numerical search for entropy-penalized
parameters/latents—including those 25 placements—not as a fallback repair
after “solving failed.”

## New exact n600 measurement

This lane built and ran a deterministic, resumable exact-residual price
instrument:

- source: V12 predictor scorer planes and the SHA-bound exact solved n600
  scorer planes;
- object: solved minus predictor, exact signed-int16 L3 raster residual;
- stages: 50 immutable 12-pair checkpoints, all preserved;
- coders: real deterministic zlib-9 and LZMA preset 0 per stage; LZMA won all
  50;
- exact container bytes: **497,662,180**;
- derived rate contribution: **331.3728181832976**;
- exact delta parse-back: 50/50;
- predictor + decoded delta equals solved uint8 planes: 50/50;
- real camera preimage → actual \(R\) → rounded uint8 identity: 50/50;
- maximum pre-round \(R\) absolute error:
  `5.684341886080802e-14`;
- compressed payloads persisted: none; byte count derives from
  generated-and-parseback-verified records;
- candidate archive: none;
- new frozen-scorer invocation: none.

This is the first true-price row in this lane, but only for
`RESIDUAL × L3_raster` before re-homing. Its verdict is:

`FORMULATION:EXACT_REVERSIBLE_L3_RASTER_RESIDUAL_RATE_DEAD`.

It does not price or kill SKELETON, CONNECTION, FIBER, the functional family,
MS2, training, or the paradigm.

| Type | Candidate home | Exact bytes | Authority |
|---|---|---:|---|
| GAUGE | `L3_raster` quotient coordinates | 0 | structural zero in this coordinate system, not an exchange-rate measurement |
| CONNECTION | `NULL` | `NULL` | no receiver-closed oracle-diff generator |
| SKELETON | `NULL` | `NULL` | no receiver-closed oracle-diff generator |
| FIBER | `NULL` | `NULL` | no receiver-closed oracle-diff generator |
| RESIDUAL | `L3_raster` | 497,662,180 | measured exact reversible n600 upper bound before re-homing |

Every inherited RD1/DR2B/C1/MENU1/V19C exchange rate is downgraded to
`upper-bound, proposal-search-channel`. Those values cannot close a path or
support new box arithmetic.

The resume replay accepted all 50 checkpoints under exact config/module/tool
custody. The receipt itself changed only at
`storage_preflight.observations[0].free_bytes`, as expected for a runtime
observation. After replacing that field with the declared runtime sentinel,
the before/after scientific hashes are identical:
`2b72204310413c1e8b0cef1ea94229b56635f1251b4c9d876b5054316cab54a3`.

## The 159× pipeline confound

The exact solve has:

- `d_seg=0.0001519690619574653`;
- `17,927` errors from exact decimal multiplication by
  `600*512*384=117,964,800`;
- `d_pose=0.00010184327939026322`;
- a demonstrated uint8-lattice realization through the real \(R\).

The best measured described Seg base `ws1_seglex96` has 2,845,843 errors, so:

\[
2{,}845{,}843/17{,}927=158.746192893401\ldots .
\]

The directive’s 159× statement is the correct rounded ratio. Its 17,931
intermediate count is not repeated because the SHA-bound exact arithmetic is
17,927.

Quantization is therefore exonerated as the binding family: the solved object
passes the same uint8/\(R\) gate. The loss is in the
description→RGB-regeneration path that re-predicts and repaints a scene rather
than describing the solved score object.

V14 and PT1 localize pieces of that path but do not form one common-payload
telescoping experiment:

- v14: exact mask promise `d_seg=0.000282948812` becomes
  `d_seg=0.027470296224` under fixed-prototype RGB projection;
- PT1: hard placement moves 2,648,079 flat-control errors to 2,592,874, and
  its 30-byte amplitude-statistics arm reaches 1,016,725 errors;
- WS2: `ws1_seglex96` is 2,845,843 errors at 138,031 bytes.

Those are real controlled endpoints in their own chains. Their differences
must not be multiplied into a fake causal factorization, and their bytes are
not oracle-diff true prices.

The 2,709,004 errors between `ws1_seglex96` and the #613 Seg allowance are a
measured endpoint gap—not a required correction workload, not #366’s job
size, and not a property of the description family.

At the exact-solve distortion point, the old strict `<0.15` arithmetic gives:

- Seg contribution: `0.01519690619574653`;
- Pose contribution: `0.03191289385033316`;
- continuous byte bound: `154522.5148231086`;
- whole-byte bound: `B <= 154522`;
- total at 154,522 bytes: `0.14999965720042385`;
- total at 154,524 bytes: `0.15000098891833010`;
- total at 200,000 bytes: `0.18028159067051396`.

This is a diagnostic tangent at one over-precise solution, not the objective.
The #613 box (`<=200 KB`, `d_seg<=0.00116`, `d_pose<=0.00161`) is a useful
representation gate, not the optimum.

## The 25-row empirical demand set

RG3 source commit `4a1728d9ae` changed the target bucket in 11 of RG2’s 36
exact hard-block evaluations. The remaining 25 all carry:

`NO_TARGET_BUCKET_EVENT_CHANGED_BY_ANY_COUNTED_RG3_MAGNITUDE_OR_SIGN`.

The family gradient is:

- class birth: `0/10`;
- finer boundary: `3/9`, leaving 6;
- Fisher cell: `8/17`, leaving 9.

The correct question is not “what coordinate can reach this bucket?” It is
“what does the solved object say here, and what is the cheapest legal way to
say it?”

The per-row ledger is
`ddm_is1_rg3_solution_demand_25_20260724.json`. Its 16 boundary rows are
candidate `SKELETON × L3_raster`: exact class-interface placement is counted
video information. Its 9 existing-cell rows are candidate
`FIBER × L4_scorer_feature`: the solved object specifies a score-visible
within-cell class/margin choice. The latter deliberately reopens RG3’s
historical SKELETON tag for #669c re-homing.

| Candidate home | Exact rows | Per-row real-coded bytes |
|---|---:|---:|
| `SKELETON × L3_raster` interface placement | 16 | `NULL` |
| `FIBER × L4_scorer_feature` within-cell choice | 9 | `NULL` |

All 25 prices are honestly `NULL`: no per-row receiver-closed solved-value
generator and real coder exists. The exact next measurement is now bounded:
materialize these solved values, run #669c, then code and parse back each row.
This is an exact blocker, not permission to speculate or launch RG4.

The measured ordering says refinement of existing structure paid more often
than conjuring a missing class interface. The class-birth `0/10` result
supports the scoped law that generic receiver geometry cannot supply
video-specific interface placement for free.

## Four-family comparison in one currency

| Family | Exact counted evidence | Exact n600 scorer evidence | Main blocker | Rank |
|---|---|---|---|---:|
| (d) score-quotient function | minimal params + latents + 25 placements = `NULL`; prior compact generator archive was 83,838 B | prior archive `d_seg=0.003455691874`, `d_pose=63.030915737`, wrong objective/two-plane content | no same-scope exact-S/coder-in-loss functional fit | 1 prospective |
| (b) oracle diff / solved-object description | raw L3 residual 497,662,180 B; other type/layer rows `NULL` | oracle endpoint already known; this lane invoked no scorer | #669c and 25-row solved-value prices | 2 |
| (a) MS2 typed quotient | `NULL`; metric-active measured pair count 0 | none | complete metric custody, #669c, real coder, E5 | 3 |
| (c) #366 descent | no independent representation price | no post-re-homing endpoint | must be bound to (d) as fitter or to an exact residual as finisher | method, not family |

The 83,838-byte row is consumed only as the representational lesson that a
small counted function can beat direct-plane storage by orders of magnitude.
It is not reused as an old-lineage vehicle, a score-quotient optimum, or a
contest score. The C2/M1 arc remains formulation-scoped and supplies no
same-currency n600 payload.

For (d), the minimal counted payload is presently:

`B(params) + B(temporal latents) + B(25 interface/margin placements) +
B(exceptions) = NULL`.

Any numeric minimum would be fake until the exact functional receiver,
real-coder-in-loss fit, Pose leg, and E5 parse-back exist.

## Training verdict and #366 job description

Two questions must remain separate:

1. **What information must be counted?** At least the content represented by
   the 25 RG3 demand rows under the current vocabulary. Sixteen are
   interface-placement candidates; nine are existing-cell margin candidates.
2. **Must a learned model represent it?** Unproven. Explicit typed
   descriptions, a score-quotient function, or another deterministic
   min-description family may carry it.

Therefore:

- `training_necessary_residual = EMPTY_ON_CURRENT_EVIDENCE`;
- `training_family_verdict = OPEN`;
- `training_as_optimizer_for_(d) = ADMISSIBLE_AFTER_MAIN_REVIEW`;
- `#366_standalone_full_closer = NOT_AUTHORIZED`;
- `#366_functional_role = entropy-penalized exact-S fitting engine`;
- `#366_finisher_role = only the nonzero visible residue after deterministic
  re-homing/MS2/E5 exhaustion`.

A future training-necessity claim must name the exact remaining quotient
coordinates, show all 25 demand rows priced, enumerate exhausted deterministic
families, bind exact archive bytes, and survive the real receiver/frozen
scorers. No training was launched here.

## Actionable continuation

1. MAIN reviews the IS1 tool, receipt, demand ledger, and equation/DAG changes.
2. Materialize each of the 25 solved-value demands rather than another
   reachability vocabulary.
3. Run #669c and fill the SKELETON/FIBER/CONNECTION type × layer price rows with
   real parse-back coders.
4. In parallel at the representation level, bind the #559 score-plane emitter,
   at-risk margins, Pose statistics, and 25-row latents into a counted
   score-quotient functional contract.
5. Fit it by exact \(S\) with real coder price; distortion is allowed to move
   to the knee.
6. Compare that candidate to metric-active MS2 on identical exact bytes through
   E5. Only then decide whether #366 is the winning fitter, a finisher, or
   unnecessary.

No paid dispatch, GPU, training, exact contest evaluation, frontier mutation,
or pointer update is authorized by this landing.

## Custody

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| delegated authority | 8,112 | `64740beb925c59d85dc9049531041469b5dc81036fa0bb4f0718a9809a813e3b` |
| Directive 1 | 2,205 | `646de874043180a8e3d3228340edc0539778add9dfaf13af99cacdbf78c069e3` |
| Directive 2 | 2,626 | `b443720b3c14ebd13e5a17b72aa3d529edec4914f312820d870768ba1dc47351` |
| Directive 3 | 913 | `de65d7adc552c2fb30dad274fe4da9ede465321a102bca8f5dee083fb55ef625` |
| Directive 4 | 2,323 | `793d53dde4c7b73b1e2fb91a48343377e354cb4d9aba7732993c7a079b7d5e42` |
| Directive 5 | 2,295 | `048166c6bd7749ad2be77912c6dd3edec177b23d239967182d1e25e84faf247d` |
| Directive 6 | 1,708 | `c34a68f1afde74050ea2305f2b437aa05b21b6cdbd8648c274caea8d55d327c6` |
| Directive 7 | 2,633 | `7a9cc45f0d4f60487fb12afcc8c59c46436d908907cf28a8422999bec614ca0e` |
| MS1 receipt | 8,800 | `1b7063a44574b0839ede08c807f348ad417be0492ac32d68634b124b9c2b1e97` |
| MS2 receipt | 32,848 | `04060edf9834b661f12a9794e50ceadf7dd4ab114baf55a15555537abc71e419` |
| RG2 support summary | 394,792 | `15b12224e3abb0d93f4fb9693402794d27969783b1d796114f0208277fe5a9ed` |
| RG3 summary at `4a1728d9ae` | git object | `3d4c4fb635ec37668cbf6037cefca63fe7c08a9ad950e6724ae023deb0473fd2` |
| y-hat RD table | 7,735 | `74e04312e90330d1a4c03e49db5090b134c4cf5894ffd78165dd576ab5c796e3` |
| WS2 receipt | 9,598 | `05581b02cc6ce789b6219302ebd888f1665ab4c3882038ce29e9be18f6174ea1` |
| IS1 exact residual receipt | 19,317 | `a03c7e07bc273f6813db7dc54d480e4f0f7931eb14c3e712fb892b27761bc863` |
| IS1 resume verification | 1,425 | `abe6b77b511a8101455562cca958cf79297ec8d95be41cd77029b59ee84b8b72` |

The RG3 summary was read from the immutable sister commit without changing
branches or copying its landing. The multi-gigabyte source planes were
streamed through their existing SHA-bound production loaders; this lane
persisted only small checkpoints and receipts.

## Triality and system wire-in

- DSL/code:
  `ddm_is1_oracle_diff_exact_residual.py`, its strict config, CLI, and tests;
  the sealed `ddm_min_description_contract`.
- DAG:
  `ddm_is1_full_inverse_solve_to_the_end_DAG_FEED_20260724.md`.
- Equations:
  `ddm_is1_full_inverse_solve_to_the_end_canonical_equations_20260724.md`.
- Sensitivity/bit allocation:
  the 25-row demand ledger replaces speculative vocabulary with exact rows;
  prices remain null until real coding.
- Pareto:
  optimize exact \(S\); the #613 box and 154,522-byte tangent remain gates,
  not the objective.
- Cathedral/autopilot:
  rank (d), hold standalone #366 authority, and do not respawn RG4.
- Continual learning:
  class-interface placement and non-telescoping pipeline laws are durable.
- Probe disambiguation:
  #669c adjudicates SKELETON/FIBER/CONNECTION homes; exact bytes arbitrate
  (d) versus MS2.

## STORES CONSULTED

- SHA-bound delegated authority and all seven amendments;
- `CLAUDE.md`, `AGENTS.md`, operating manual, and v7.5/v8 contracts;
- current lane, subagent, frontier, and relevant canonical state;
- MS1–MS6, RG1/RG2, V14, PT1, A1, V19C, MENU1, WS2, y-hat, M1/C2, and
  score-native doctrine artifacts;
- RG3 source commit `4a1728d9ae` and its 25-row summary;
- exact production source loaders and solved/predictor custody;
- both live inboxes after every major milestone.

## MAIN landing review required

MAIN must independently review:

1. the exact residual encoder/decoder, source hashes, real-\(R\) identity, and
   the runtime-only resume hash difference;
2. the 25-row solution-demand re-homing and every `NULL` price;
3. the min-\(S\)-over-solutions objective and four-family ranking;
4. the strict separation between training as optimizer and training as
   representational necessity;
5. the old-lineage ban around the compact-function lesson;
6. the non-telescoping 159× diagnosis and exact 17,927 arithmetic;
7. preservation of `score_claim=false`, pointer immobility, and absence of any
   candidate or contest-axis claim.
