# ddm_r012 — rate-representation routes on the rc2 body

- **Date:** 2026-08-21
- **Arm:** `ddm_r012_rate_representation`
- **Base:** `rc2_composed`, archive
  `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`
- **Authority anchor:** **S 0.14827847122030852 @ 180,456 B
  `[contest-CUDA T4, n600]`**, `d_seg=0.00020139`, `d_pose=6.37e-06`.
- **This arm's activity:** recall, exact arithmetic, scorer-free custody inspection, and routing.
  It launched no scorer, no governed job, no Modal call, and materialized no payload.
- **Overall disposition:** **BLOCKED(rate route is at least 42,382 B short at current
  distortion; fx5's measured top rung is owned by a sibling and lacks its recorded decode receipt
  and seal).**

## ANSWER FIRST

**The honest composable measured ceiling is 88 B, not 42.5 KB: fx5 contributes a measured
70 B on rc2 and dx1 can contribute 18 B after a receiver-format transfer.** The two touch disjoint
streams. At unchanged distortion that would give 180,368 B and
`S=0.14821987563243377`, still **42,382 B short** of the exact sub-0.12 rate requirement.

Rate alone therefore does not reach sub-0.12. More strongly, the measured 88 B composition cannot
reach sub-0.12 even if every rc2 distortion term is eliminated: zero distortion still requires a
minimum 238 B cut, leaving a 150 B irreducible rate deficit.

The top measured rc2 rung is fx5's 19-member generic corrector. Its candidate archive and retained
copy are present and byte-identical at **180,386 B**, sha256
`4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`, a measured **-70 B**.
However, the recorded `FX5_DECODE_IDENTITY.json` and
`CANDIDATE_SEAL_fx5_e1_19member.json` were absent when inspected, while the full decode status said
`running`. The rung is therefore **BLOCKED(fx5 sibling decode/seal in progress)**, not
`READY_TO_FIRE`. This arm did not touch fx5's surface.

## EXACT GOAL ARITHMETIC -- correction to the charter

The canonical rate derivative is

`r = 25 / 37,545,489 = 6.658589531221713e-7 S/B`.

On rc2, the distortion contribution is

`D = 100(0.00020139) + sqrt(10(6.37e-06)) = 0.028120227975693966`.

Consequently the exact unchanged-distortion condition is

`D + r B < 0.12`, hence `B < 137,986.8388 B`.

For an integer-sized archive the real bar is therefore **archive <=137,986 B**, or a cut of at
least **42,470 B**. The charter's `-42,463 B / archive <137,993 B` arithmetic is seven bytes too
weak and cannot be used as an admission line. A 42,463 B cut still needs a
`4.102493881758e-06 S` distortion win.

The other hard endpoint is also exact: with `D=0`, rc2's 180,456 B archive remains above 0.12;
sub-0.12 requires **archive <=180,218 B**, a cut of at least **238 B**.

## ROUTE TABLE -- bookable credit first

`Expected -B` is the credit this campaign may honestly book on rc2 now. An ancestor-body or
unbuilt number is shown as an anchor but books zero until transfer. Distortion prices are score
deltas, because one byte is worth only `6.658589531221713e-7 S`.

| rank | rung | evidence label | expected -B on rc2 | distortion price | decode-wall delta against 323.5 s | conflicts and disposition |
|---:|---|---|---:|---|---|---|
| 1 | fx5 19-member token corrector | **MEASURED-on-rc2** `[macOS-CPU scorer-free exact bytes]` | **70 B** | zero **only after** decoded-token identity; generic code adds no counted payload | **MODELED +89 s**, leaving about 234.5 s; current local full decode was still running, not a T4 wall result | Token stream; disjoint from dx1. **BLOCKED(fx5 decode receipt and seal absent; sibling owns surface).** |
| 2 | dx1 adaptive-context Rice/CABAC coefficient recode | **MEASURED-elsewhere-owes-transfer**; rc2 coefficient lattice and 9,829 B Rice payload were verified byte-identical to dx1's body | **18 B after transfer**, zero booked before it | zero if coefficient decode is identical; otherwise refused | unmeasured | Same CAP1 carrier container/header as rr5 and the fixed-metadata idea, but a disjoint coefficient stream from rr5's basis stream. Needs CABAC receiver plus an unambiguous flag. **QUEUED-WITH-FIRE-ORDER after fx5 closes.** |
| 3 | CAP1 fixed-metadata pack | **MEASURED-elsewhere-owes-transfer** | **0 booked**; **79 B ancestor anchor** | zero only with bit-identical parse-back | unmeasured | CP135 measurement, while rc2 ships our re-solved 22,296 B carrier. Competes with dx1/rr5 for container signalling; do not sum until one joint build proves the format. **QUEUED behind dx1 as the alternative.** |
| 4 | warm-lineage / curriculum-inherited width distillation | **MODELED** | **0 booked**; target scale 4--12 KB is not a credit | must be smaller than its real byte credit; no admissible current estimate | unknown; a narrower decoder may help, but no wall receipt exists | Rewrites the model/body, so every fx5, dx1, CAP1, and token price must be remeasured. Fresh D56/F64 is not a proxy for this mechanism. **QUEUED only if a stratified checkpoint comes within 1.5x of its byte-derived score bar.** |
| 5 | WD2/WD3 measured narrow archive | **MEASURED-elsewhere-owes-transfer** | **0 booked**; gross 17,372 B versus hv1 | D56/F64 n120 instances lose **+1.4806/+1.1093 S**; the earlier scorer-aware narrow row's seg damage was about six times its bar | unmeasured | Fresh-init, 65-epoch family instances. **CLOSED(INSTANCE); FAMILY parked, not killed for warm-lineage/longer/curriculum births.** W96 correctly did not fire. |
| 6 | carrier rank/atom truncation and refit | **MEASURED-elsewhere-owes-transfer** | **0 booked**; feasible rungs returned only 913--1,847 B and rank-4 gross was 14,709 B | functional misses were 1,498--3,139x; best trust-region refit still 35.5x its bar | unmeasured | Same carrier payload as dx1/CAP1. **CLOSED(FAMILY across ra3+rr1's six treatments).** |
| 7 | mz2 mixed q3/q4 and FiLM-row sparsity | **MEASURED-elsewhere-owes-transfer** | **0 booked**; retained gross anchors 823 B and 130--2,051 B | q3/q4 lost +0.07549 S net; deeper mp2 prune lost +0.03617 S net, and keep25 no longer has positive credit versus rc2 | unmeasured | Shares learned carrier tensors with CAP1/dx1 and changes the render. **CLOSED for current-body transfer.** Retained mz2 inventory: 167 files, 8,183,925 B, manifest sha `156112d0...`; no payload was discarded by this arm. |
| 8 | rc4/fs2/fs3 token drop plus carrier re-solve | **MEASURED-on-lineage** | **0 booked**; real credits 1,022 B (fs2) and 664 B (fs3) | fs2 remains +0.005515 S even with pose free; fs3 measured +0.0357952 S net, with pose 81x its rate credit | receiver class unchanged, but no independent wall delta | Shares the jg5 edit field and forces carrier re-solve; fx5's token-coder delta would need remeasurement. **CLOSED(FORMULATION for current confidence-threshold/drop construction).** |
| 9 | task #869 token-by-token waterfill/adaptive map | **MEASURED-elsewhere-owes-transfer** | **0 booked**; old IX2 projection -113,555 B is discarded | later gx1 measured **+0.009002 S** on the changed/live operating point | unmeasured | Task-lossy ancestor projection, not a current archive. **CLOSED(INSTANCE and number non-transferable).** |
| 10 | more memoryless section/coder work | **MEASURED-on-lineage** | **0 booked**; best residual section-coding census was about 5 B, packaging slack 210 B total | zero only for exact recodes | mechanism-dependent | All four shipped sections are at their measured memoryless bounds; SMEVR adds 5,183 B. **CLOSED(FORMULATION except body-dependent coding already consumed by fx5/dx1).** |

### Composition accounting

The only current composable measured route is:

`fx5 token corrector 70 B + dx1 coefficient coder 18 B = 88 B`.

It is honest to call 70 B **measured and built**, but not receiver-closed until fx5's live decode
finishes. It is honest to call dx1's 18 B **measured elsewhere with an exact rc2 input-identity
premise**, but not built on rc2. Therefore 88 B is the **measured composable ceiling**, not a ready
candidate.

CAP1's 79 B is an alternative ancestor-body anchor, not a third additive term. Even granting its
full transfer and composing it with fx5 would yield only 149 B, still below the unavoidable 238 B
zero-distortion cut. Width changes the body and invalidates all small-delta prices until they are
remeasured; the remaining rows either fail joint score or are closed.

The prior-law falsifier was a composable measured route of at least 42,470 B. **It did not occur.**
The old projection stack over-promised by far more than 5--10x after body transfer, distortion
pricing, and stream-conflict removal. That is a result about this rc2 routing census, not a universal
law for future representations.

## JOINT RATE--DISTORTION FRONTIER TO SUB-0.12

Here `distortion win` means a decrease in
`100*d_seg + sqrt(10*d_pose)` from rc2, and `minimum cut` is the smallest integer byte cut that makes
the strict inequality `S<0.12` true.

| distortion win | minimum rate cut | largest admissible archive |
|---:|---:|---:|
| 0 | **42,470 B** | 137,986 B |
| 0.001 | 40,968 B | 139,488 B |
| 0.005 | 34,961 B | 145,495 B |
| 0.010 | 27,451 B | 153,005 B |
| 0.015 | 19,942 B | 160,514 B |
| 0.020 | 12,433 B | 168,023 B |
| full current seg term, 0.020139 | **12,225 B** | 168,231 B |
| 0.025 | 4,924 B | 175,532 B |
| 0.028 | 419 B | 180,037 B |
| full current distortion, 0.028120227975694 | **238 B** | 180,218 B |

The inverse routing form is equally important:

| rate cut supplied | minimum distortion win still required |
|---:|---:|
| 70 B, fx5 only | 0.028231861093589968 |
| **88 B, measured composable ceiling** | **0.028219875632433770** |
| 238 B | 0.028119996789465443 |
| 1,000 B | 0.027612612267186350 |
| 4,000 B | 0.025615035407819835 |
| 8,000 B | 0.022951599595331150 |
| 12,000 B | 0.020288163782842464 |
| 20,000 B | 0.014961292157865093 |
| 30,000 B | 0.008302702626643380 |

At the 88 B measured ceiling, the required distortion win exceeds all distortion rc2 has by
`9.9647656739804e-05 S`. Thus jo1 cannot make this measured rate route sufficient even with a
perfect distortion result. A new rate representation must first supply at least 150 additional
bytes, and the practical asks remain 12,225 B if jo1 removes all current seg error or 42,470 B if
distortion stays fixed.

## TOP-RUNG BUILD AND CUSTODY STATUS

The top measured rc2 rung was already being built on the explicitly protected sibling surface
`/Volumes/APDataStore/pact/ddm_fx5/`. This arm performed only read-only custody checks:

| required fact | observed fact | disposition |
|---|---|---|
| rc2-derived candidate archive | `/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5/archive.zip`, **180,386 B**, sha `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841` | present |
| retained payload copy | `/Volumes/APDataStore/pact/ddm_fx5/retained/candidate_e1_19member.zip`, **180,386 B**, same sha | present, payload retained |
| build manifest | `candidate_runtime_fx5/FX5_BUILD_MANIFEST.json`, base sha `df7fd266...`, 19 members, zero counted payload for new members | present |
| structured Python/C parity | `retained/FX5_PARITY_E1.json`, 393,216 rows, verdict `IDENTICAL` `[macOS-CPU advisory / implementation parity only]` | present, not the n600 receiver verdict |
| real n600 decoded-token identity receipt | recorded path `retained/FX5_DECODE_IDENTITY.json` | **absent at inspection** |
| candidate seal | recorded path `CANDIDATE_SEAL_fx5_e1_19member.json` | **absent at inspection** |
| live decode | `decode_r1/launch/resource_safe_run_status.json`: `status=running`, elapsed 809.494 s at the final custody sample (`2026-08-21T14:38:23Z`) | incomplete; not a T4 wall receipt |

The sibling memo says the rung is sealed, but the bytes on disk are the authority and do not yet
support that statement. **Build disposition: BLOCKED(fx5 owner must finish or classify its n600
decode, write the identity receipt, then make the seal).** Duplicating or repairing it here would
violate the charter's no-touch constraint and create two owners for one candidate.

No r012 payload directory was created because this arm generated no payload. There is nothing to
retain or certify under `/Volumes/APDataStore/pact/ddm_r012_rate_representation/`.

## FALSIFIERS REGISTERED BEFORE ANY FUTURE MEASUREMENT

1. **fx5:** refuse if the full decoded token sha is not rc2's
   `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`, if the real candidate
   archive is not 180,386 B with the sha above, or if contest-T4 charged decode exceeds 822 s.
   A local running timer does not settle the T4 wall.
2. **dx1 transfer:** refuse if the container flag or alignment leaves less than 18 B net archive
   credit, if the decoded coefficient lattice changes by one value, if the flag collides with rr5's
   reserved `0x08`, or if the composed receiver does not finish under 822 s on T4.
3. **CAP1 alternative:** refuse if a current-body parse-back is not byte-identical or its total net
   archive credit is nonpositive after signalling. Do not add it to dx1 until one shared-container
   build proves both formats coexist.
4. **warm-lineage width:** do not spend the full governed burn unless a stratified-random n>=120
   checkpoint lands within 1.5x of its real serialized byte-derived score bar. Refuse the instance
   if realized joint score damage exceeds its real archive credit; do not transfer that refusal to
   longer-budget or curriculum-born families without matching the mechanism.
5. **campaign falsifier:** a receiver-closed, mutually composable rc2 route of at least 42,470 B
   would overturn this memo's rate-alone conclusion. A route of at least 238 B would overturn only
   the stronger claim that the current small-rate stack cannot work even with zero distortion.

## FOLLOW-ON DISPOSITIONS

- **QUEUED-WITH-FIRE-ORDER:** fx5 owner finishes the already-running local decode, writes the
  identity receipt and candidate seal, then MAIN alone may fire the T4 row if the seal verifies.
  Consumer store: `/Volumes/APDataStore/pact/ddm_fx5/t4_row_r1/`.
- **QUEUED-WITH-FIRE-ORDER:** after fx5 is resolved, a rate-representation successor may build the
  dx1 CABAC decoder and CAP1 flag on a copy under
  `/Volumes/APDataStore/pact/ddm_r012_rate_representation/dx1_rc2/`; the fire trigger is a
  byte-identical full decode, >=18 B net archive credit, and a seal that composes the rr5 flag.
- **FOLDED:** the joint routing table above is the consumer for jo1. Once jo1 returns a measured
  distortion win, read the corresponding residual-byte ask directly from this table; no speculative
  rate promise is needed.
- **FOLDED:** CAP1 fixed metadata remains the alternative format experiment inside the dx1 receiver
  rebuild, not a separate additive row.
- **FOLDED:** width distillation remains parked behind its existing reactivation gate; no governed
  ticket is READY because fresh D56/F64 failed and no warm-lineage checkpoint presently clears the
  preflight bar.

## RECALL EVIDENCE

### Surfaces and queries

I searched the full research corpus, canonical equation registry, canonical indexes/DAG, live board,
and task ledger rather than using the charter as the corpus. The content queries included:

- `rfo2|route order|width.distill|D56|F64|W96|mixed q3|FiLM.row|token.by.token waterfill|adaptive.map|direction-dependent|2.24|token.drop|carrier re.solve|first.order rate|over.promise|180,456|df7fd266`
- `#869|token waterfill|adaptive map`
- `carrier rank|atom truncation|q3/q4|19-member|CABAC`
- exact task IDs and dispositions in `.omx/state/canonical_task_status.jsonl`, plus rate entries in
  `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` FEED blocks.

The equation registry was read through
`.venv/bin/python tools/list_canonical_equations.py --json`; the score/rate derivative and
byte/flip price used here come from the registered canonical equations, not a copied charter
constant.

Principal evidence consumed:

- `.omx/research/ddm_rfo2_fresh_eyes_gestalt_synergy_20260815.md`
- `.omx/research/ddm_wd3_n120_family_disposition_20260816.md` and the D56/F64 verdict/design files
- `.omx/research/ddm_mz2_frozen_section_representation_attack_20260815.md`
- `.omx/research/ddm_mp2_mixed_precision_receiver_close_20260815.md`
- `.omx/research/ddm_fs2_rc4_drop_carrier_resolve_20260820.md`
- `.omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md`
- `.omx/research/ddm_jg5_pose_resolve_on_edited_renders_20260819.md`
- `.omx/research/ddm_rc4_rung4_token_drop_verdict_20260816.md`
- `.omx/research/ddm_gs2_gestalt_synthesis_vs_rc2_20260821.md`
- `.omx/research/ddm_eq1_equations_lineages_vs_rc2_20260821.md`
- `.omx/research/ddm_pw2_week_signal_composition_matrix_20260821.md`
- `.omx/research/ddm_na11_negative_regrade_vs_rc2_20260821.md`
- `.omx/research/ddm_rc2_t4_row_sixteenth_move_20260820.md`
- `.omx/research/ddm_na4_20260805/NA4_RECEIPT.md`
- `.omx/research/ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md`
- the read-only sibling memo `.omx/research/ddm_fx5_composed_rate_candidate_20260821.md`, checked
  against the actual APDataStore files rather than accepted as current custody truth.

### Findings beyond the charter seeds and what changed

1. **Exact arithmetic changed the campaign line:** the unchanged-distortion cut is 42,470 B, seven
   bytes larger than the charter headline, and the zero-distortion floor is 238 B.
2. **PW2's later six-treatment carrier census changed the route:** it closes the broad
   22,161 B carrier-shave projection as a current measured route; only 913--1,847 B returned in
   function-preserving rungs and even those failed their quality bars.
3. **Eq1 and gx1 changed #869's status:** the old -113,555 B IX2 projection is task-lossy and a later
   operating-point transfer measured +0.009002 S, so it cannot enter an rc2 sum.
4. **fx5's live artifact state changed the build disposition:** its 70 B archive is real and
   retained, but the claimed identity receipt and seal are absent while its decode is running.
   Therefore this memo reports `BLOCKED`, not the sibling memo's premature sealed state.
5. **dx1's coefficient object survives rc2 unchanged:** rr5 recodes the basis, not the coefficient
   stream. This kept dx1 as the only additional measured composable rung and reduced its blocker to
   a receiver/container implementation.
6. **The rate-axis prefix audit is near-neutral (0.989--1.030x),** but this memo did not promote any
   prefix result or bank a negative from n=8; all relied-on negative rows are n120 or full-lineage
   receiver measurements with their stated scope.

## BOUNDARIES

**Measured or exact in this receipt:** rc2's authority components and archive bytes from its
contest-CUDA receipt; the exact score arithmetic; fx5's two on-disk archive sizes and hashes; the
presence of its manifest/parity file; the absence of its recorded identity/seal paths at inspection;
the route numbers explicitly marked measured in their cited bodies; and the algebraic joint frontier.

**Not measured by this arm:** no new d_seg, d_pose, score, T4 decode time, width result, dx1 transfer,
CAP1 transfer, or joint candidate. fx5's unchanged distortion is conditional on the incomplete real
decode-identity gate. The +89 s fx5 wall delta is modeled, not observed on T4. Ancestor-body credits
are not silently rebased. No conclusion here promotes a macOS or subset row to contest authority.

## LIVE-HYPOTHESES

- **dx1 may still return its full 18 B on rc2** because the coefficient lattice and Rice payload are
  byte-identical to its measured body; only receiver signalling and decode closure remain.
- **CAP1 fixed metadata may beat dx1 as the carrier-format choice** because its ancestor measurement
  was 79 B, but rc2's re-solved carrier and shared signalling make that a transfer hypothesis, not
  additive credit.
- **Warm-lineage or curriculum-inherited width reduction may escape the fresh-birth failure** because
  D56/F64 tested fresh 65-epoch births, not a narrowed descendant of the conditioned shipping trunk.
  It is worth pursuing only through the stated stratified early gate.
- **A genuinely new representation is still required** because even perfect distortion leaves a
  238 B rate floor and the measured composable stack supplies only 88 B; the practical target is much
  larger unless jo1 eliminates nearly all current distortion.

## DEAD-ENDS

- **Fresh D56/F64 and unconditional W96 escalation:** closed at INSTANCE scope by reproduced n120
  score losses; W96 lacks measured capacity pressure.
- **Current carrier rank/atom truncation-refit family:** closed across six treatments because
  function-preserving rate returns were small and quality misses remained at least 35.5x.
- **Current q3/q4 and deeper FiLM sparsity transfer:** closed because pose/score loss dwarfs the
  measured byte credit and keep25 is larger than rc2.
- **Current confidence-threshold token drop plus carrier repair:** closed at FORMULATION scope by
  real re-encode prices and same-instrument pose loss; the required carrier rescue is 696x versus the
  measured 8x precedent.
- **Task #869's -113,555 B projection:** discarded as an ancestor, task-lossy number contradicted at
  the later operating point.
- **More memoryless section coding or packaging archaeology:** closed on this lineage; the residual
  measured room is bytes, not the tens of kilobytes the goal needs.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` -- UNMOVED by ddm_r012.**
