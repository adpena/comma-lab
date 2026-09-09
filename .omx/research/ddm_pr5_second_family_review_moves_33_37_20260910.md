# ddm_pr5 — second-family review of pointer moves 33–37 and gs3 Addendum 18

`[no-triality] [p0-ledger-ok]` · owner `ddm_pr5` · 2026-09-09.
`research_only=true`; `score_claim=false`; review axis only.

## Verdict

The five exact `[contest-CUDA T4 n600]` scores all reproduce from receipt
components within `2.78e-17`, so the charter's P0 arithmetic stop did not fire.
The canonical pointer is correctly bound to move 37. The projection-residual
prediction also holds: exact, optimistic by `1.8446260297011463e-6`,
pessimistic by `1.0876437430418218e-6`, machine-exact, exact.

Addendum 18's correction fixes its original 83-KB attribution, unshipped-field,
and bound-direction errors, but the source audit is not clean. Six additional
scope errors remain: the 36-KB “flag mass” is another ideal-cost attribution;
RW1's 240-cell endpoint is a partial-screen extrapolation; RP1's 217-B full-pass
number is an n120 projection; exact rate-token/residual-cell identity was never
joined; “coder-level ... done” closes live context designs the packets leave
open; and rank 6/6 does not admit every future representation through a carrier
re-solve. The requested public-state assertions also fail: PR #140 carries the
23rd pointer-move archive, so move 37 is **14**, not 13, moves ahead, and the
public PR body explicitly names Claude and Codex.

## Exact score re-derivations

Every primary receipt reports `passed: true`, `returncode: 0`, the canonical
`archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda` path, and an
empty validation-error list; its component and axis fields are at receipt
lines 63–73 (for example RC2 at
`/Volumes/APDataStore/pact/ddm_rc2_t4_hpac_semistatic_mixing_20260909/MODAL_REMOTE_RESULT.json:63-73`).
The complete machine-readable arithmetic is retained in
`ddm_pr5_20260910/score_recomputation.json:1`.

`S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489`.

| move | receipt components `(d_seg, d_pose, B)` | derived terms `(seg, pose, rate)` | recomputed S | recomputed − receipt | result |
|---:|---|---|---:|---:|---|
| 33 rc2 | `0.00010913, 5.10e-6, 181414` | `0.010913, 0.0071414284285428505, 0.12079613612170559` | `0.13885056455024844` | `0` | PASS |
| 34 pc2 | `0.00010913, 5.10e-6, 181373` | `0.010913, 0.0071414284285428505, 0.12076883590462759` | `0.13882326433317044` | `0` | PASS |
| 35 sj1 | `0.00010698, 5.05e-6, 181521` | `0.010698, 0.007106335201775948, 0.12086738302968966` | `0.13867171823146562` | `0` | PASS |
| 36 cmp1 | `0.00010698, 5.05e-6, 180772` | `0.010698, 0.007106335201775948, 0.12036865467380116` | `0.13817298987557710` | `-2.7755575615628914e-17` | PASS |
| 37 cmp2 | `0.00010698, 5.05e-6, 180388` | `0.010698, 0.007106335201775948, 0.12011296483580225` | `0.13791730003757818` | `0` | PASS |

The packets independently print the same term decompositions at rc2:13-16,
pc2:13-16, sj1:13-16, cmp1:13-16, and cmp2:13-16. Archive sizes and SHA-256s
also match the fired custody rows at each packet's lines 7 and 43–47. The live
pointer names cmp2's lane, archive SHA, and score at
`.omx/state/canonical_frontier_pointer.json:3-15`, repeats the 180,388-B CUDA
anchor at lines 38–59, and has no checkpoint-maturity refusals at lines 62–71.

## Projection versus exact

Residual means `realized S - projected S`; a positive residual is optimistic,
a negative residual is pessimistic.

| move | pre-fire projection | exact S | residual | classification | source-grounded cause |
|---:|---:|---:|---:|---|---|
| 33 rc2 | `0.13885056455024844` | `0.13885056455024844` | `0` | exact | Lossless rate-only recode; the seal fixes `-231 B`, unchanged distortion, and the exact projection (`/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/SEAL_ddm_rc2_hpac_semistatic_mixing_contest_cuda.json:21-25`); T4 reproduced both distortion prints (rc2 packet:20-27). |
| 34 pc2 | `0.13882141970714074` | `0.13882326433317044` | `+1.8446260297011463e-6` | optimistic | The local instrument used `d_pose=5.0901316014139157e-6`, while the authority receipt exposes only the `5.10e-6` print; the seal states both values and the print gate (`SEAL_ddm_pc2_carrier_scales_resolve.json:21-27`), and the packet classifies the residual at lines 25–31. |
| 35 sj1 | `0.13867280587520867` | `0.13867171823146562` | `-1.0876437430418218e-6` | pessimistic | The shipped-field seg prediction was exact and T4's printed pose was slightly more favorable; the pre-fire projection and band are in `ddm_sj1_multipass_token_predistortion_20260905.md:1231-1237`, and the realized classification is at the move packet:25-31. |
| 36 cmp1 | `0.13817298987557710` | `0.13817298987557713` | `+2.7755575615628914e-17` | exact to floating-point order | The two lossless section deltas composed to exactly `-749 B` and held both distortion prints (cmp1 packet:18-31); the seal's rate-only rule is `SEAL_ddm_cmp1_rc3_tc1_composed.json:4-16`. |
| 37 cmp2 | `0.13791730003757818` | `0.13791730003757818` | `0` | exact | Build A was a `-384 B` lossless recode at held distortion; the derived projection is in `ddm_cmp2_compose_sm1_fe1_20260909.md:182-189`, and the exact custody replay is at the move packet:18-31. |

All referenced seals in that table live below their named
`/Volumes/VertigoDataTier/pact/` arm directories. No projection is relabelled an
exact score.

## Addendum 18 source audit

The audited claims are gs3 Addendum 18 §§I–IV at
`.omx/research/ddm_gs3_gestalt_after_submission_20260903.md:1031-1073`; its
correction is at lines 1077–1080.

Citation key used below: `gs3` is
`.omx/research/ddm_gs3_gestalt_after_submission_20260903.md`; `bnd1` is
`.omx/research/ddm_bnd1_boundary_representation_closed_form_pricing_20260909.md`;
`rw1` is
`.omx/research/ddm_rw1_boundary_local_renderer_weight_foldback_20260909.md`;
`rp1` is
`.omx/research/ddm_rp1_rate_directed_token_predistortion_20260909.md`; and a
`move-N packet` is the corresponding
`.omx/research/*_pointer_move_N_20260909.md` source named in the exact-score
table.

| class | Addendum claim | source verdict and corrected statement |
|---|---|---|
| ATTRIBUTION — corrected | “83 KB sits on” the 235,044 mispredicted tokens and is boundary description (gs3:1063-1069). | Corrected by gs3:1077-1080. `83,258.63239658909 = 666,069.0591727127/8` B is `sum(-log2 p(symbol))/8`, not a serialized substream; the actual whole retained envelope is 119,784 B (`ddm_bnd1_boundary_representation_closed_form_pricing_20260909.md:27-30,43-50`; producer math `experiments/ddm_rp1_rate_rank.py:705-722`). |
| DIFFERENT-OBJECT — corrected | `12,377` pass-5 cells were joined to the move-37 rate population as shipped (gs3:1062-1065). | Corrected by gs3:1077-1080. Move 37 ships pass 4: 12,614 cells, field SHA `813bf1e6…`; pass 5's 12,377 cells have SHA `48b5852e…` and are unshipped (`ddm_bnd1...md:51-58`). The subtraction is `12,614 - 237 = 12,377`, but it does not establish a same-field join. |
| BOUND-DIRECTION — corrected | A representation had to beat “83 KB” and an achieved code implied a `>=78 KB` floor (gs3:1069; corrected discussion gs3:1077-1080). | An achieved length `L` proves `optimal_length <= L`, never `>= L`; empirical entropy of a distribution is not a shortest-program lower bound for this structured object (`ddm_bnd1...md:63-69`). No family was closed. |
| ATTRIBUTION — still present | The remaining 117.7 M positions contain a detachable `~36 KB` prediction “flag mass” untouchable by any field change (gs3:1063). | `36,519.39071475216 = 292,155.1257180173/8` B is the same `-log2 p(symbol)` attribution, now over positions where the stored symbol is the row argmax (`MISPREDICTED_CENSUS.json:300-311`; producer fields at `experiments/ddm_rp1_mispredicted_census.py:148-164`). It is not an emitted flag substream. Holding a probability row fixed, changing its argmax symbol cannot improve that local first-order term; context, predictor, or representation changes can change the rows, so “untouchable by any field change” is not established. |
| PREFIX / partial screen — still present | One RW1 int4 code step “breaks 240–455 cells” as a measured formulation fact (gs3:1039). | The 240 endpoint is `48` extra flips on a seeded 120-pair screen multiplied by five (`ddm_rw1_boundary_local_renderer_weight_foldback_20260909.md:500-527`). RW1 later proves this screen invalid for a global actuator because the checked case reverses sign outside the screen (`rw1:745-775`). The n600-safe conclusion is 0 accepted repairs in 150 searches and 0 confirmed repairs across tested routes, not an n600-measured 240-cell minimum. |
| PREFIX / partial sample — still present | RP1's “full pass -217 B” appears in the measured column (gs3:1045). | The exact completed partial encode is `119,784 -> 119,752 = -32 B` for 217 accepted changes (`ddm_rp1_rate_directed_token_predistortion_20260909.md:254-272`). The `217.0 B` is a full-pass projection: an unbiased interleaved n120 sample is multiplied to n600 and then multiplied by a measured `0.2663` selected payout (`rp1:274-289`). RP1 explicitly says a real subset re-encode and pose leg are owed. |
| DIFFERENT-OBJECT / unjoined populations — still present | The rate tokens “are the boundary population” of the seg errors and the representation “stores the boundary twice” (gs3:1063-1065). | The defensible overlap is only a lower bound: `209,179 / 235,044 = 88.99567740508161%` of the full misprediction population is on an edge, using 213,733 retained locations and treating all 21,311 missing locations as off-edge (`ddm_bnd1...md:21-30,73-96`). GT class-pair edges, tangent runs, and the exact joint residual-cell intersection remain unmeasured. Broad edge support is not population identity or proof of two detachable boundary stores. |
| DIFFERENT-OBJECT / formulation-to-family — still present | “The positives ... closed the coder level” and “the coder-level squeeze is done” (gs3:1052,1063). | The measured chain covered every existing archive section and recovered about 3.2 KB; it did not close all coder designs. RC2 leaves a richer shared low-parameter mixer live behind a >=150-B counted-design gate (move-33 packet:49-56), and CMP1 leaves a joint-context tail architecture live (move-36 packet:49-56). “Covered every section with the tested chain” is the source-supported statement. |
| DIFFERENT-OBJECT / universal pose transfer — still present | Rank 6/6 means “every candidate representation” is admitted per pair through carrier re-solve (gs3:1071). | Rank 6/6 over all pairs is a structural precondition on the incumbent carrier. Recovery for a global renderer change was explicitly not measured and remains owed (`ddm_rw1...md:238-246,831-834`). Prior pair-confined re-solves do not prove recovery for an arbitrary new representation. |

The prior-law prediction was only partly right. The additional 36-KB attribution
error exists, but neither a `12.75x` nor an `8.94x` claim appears in Addendum 18
§§I–IV. Its `msr1 8.94%` is labelled a historical “Prior” and “at the time” at
gs3:1069; read that way, it is not a transfer claim. BND1 independently confirms
that neither historical `msr1 8.94%` nor `lb1 8.94x` can price the new local draw
(`ddm_bnd1...md:126-139`).

## “What this does NOT claim” cross-check

- **RC2: one contradiction.** Its packet keeps a richer shared mixer live behind
  a counted >=150-B gate (move-33 packet:49-56), so Addendum 18's coder-level
  closure at gs3:1052/1063 is too broad. The unchanged-distortion, no-CPU, and
  no-render-composition boundaries are otherwise respected.
- **PC2: clean.** Addendum 18 does not promote the refused rank cut, the 7-B
  reader-dependent ITEM 2, or a CPU row; those exclusions are explicit at the
  move-34 packet:49-57.
- **SJ1: clean after its field correction.** Addendum 18 does not claim pass-4
  convergence or promote the full field; the packet excludes both at move-35
  lines 49-57. Its use of the unshipped pass-5 residual is already corrected at
  gs3:1077-1080.
- **CMP1: one contradiction.** Its joint-context tail architecture remains a
  live hypothesis (move-36 packet:49-56), again contradicting universal
  coder-level closure. No general additivity or CPU claim is imported.
- **CMP2: clean.** Addendum 18 does not promote Build B, transfer the favorable
  121-B container interaction, or claim CPU authority; the packet expressly
  excludes all three at move-37 lines 49-56.

Separately, gs3:1062's “No actuator on the shipped object reaches it” must be
read as “no listed measured formulation reaches it.” RW1 says the renderer
paradigm is untouched and names finer-grid and realized-selection cures
(`ddm_rw1...md:450-467,564-567`). As an unqualified family closure it would
contradict that source.

## PR #140 status

The requested “13 moves behind” assertion does not verify. The public PR still
carries AFR1's archive, and the custody registry calls AFR1 the **23rd pointer
move** (`.omx/state/active_lane_dispatch_claims.md:142-145`). FS2 independently
records that same public archive as move 23 (`ddm_fs2_pointer_move_25_20260904.md:43-46`).
The current packet is move 37 (move-37 packet:1-7), hence `37 - 23 = 14` moves
behind. The score gap is
`0.14797617125559104 - 0.13791730003757818 = 0.010058871218012855 S`;
the posted score is sourced at `ddm_pr140_submission_posted_20260903.md:45-47,73-74`.
Move 37's own line-57 “13 moves” statement is off by one.

The no-public-AI-attribution assertion also fails. The checked-in posted-body
snapshot says, in public-facing text, “I used coding agents (Claude as
orchestrator of Codex subagents)” at
`experiments/results/ddm_fr2_final_review_20260903/pr_body_FINAL_POSTED.md:75`.
The live unauthenticated GitHub page still showed the same sentence; retained
live-check receipt: `ddm_pr5_20260910/pr140_public_audit.md:1-21`. The narrower
claim that the two commits have no AI/co-author trailer is recorded at
`ddm_pr140_submission_posted_20260903.md:24-28`; it does not make the broader
public-text statement true. Nothing was published or edited by this review.

## RECALL EVIDENCE

The required source floor was read in full: charter, common contract, PROGRAM,
AGENTS/CLAUDE, operating handoff, live hot state, canonical pointer, all five
move packets and authority receipts, Addendum 18 plus correction, BND1, GB2,
and the per-landing typed-verdict reference at
`ddm_gc16_from_here_20260803.md:466-485`.

Independent content searches covered `.omx/research/` and retained receipts
with queries including `boundary representation|flag mass|mispredicted-token`,
`one code-step|240|455|screen`, `full pass|217 B|first-order`,
`rank 6/6|global recovery|per-pair re-solve`, `12.75|8.94`, and
`PR #140|23rd pointer move|AI attribution|Co-Authored-By`. The canonical
equation export (`tools/list_canonical_equations.py --json`) was searched for
`boundary|contour|offset|tail|context|pose|prefix`; the research index/DAG,
design/SPEC corpus, canonical task ledger, and dispatch registry were searched
for the same surfaces.

Beyond the charter seeds, RW1's own screen retraction at rw1:745-775 changed
the 240-cell row from measured to partial-screen extrapolation; RP1's full-pass
derivation at rp1:274-289 changed 217 B from measured to projected; the dispatch
registry at `active_lane_dispatch_claims.md:142-145` changed the requested PR
lag from 13 to 14; and the public-body snapshot at
`pr_body_FINAL_POSTED.md:68-75` falsified the requested public-attribution
condition. The equation/index/DAG/SPEC search found no same-object receipt that
repairs those four classifications. BND1's complete-population limits at
bnd1:73-111 prevented broad edge support from being upgraded to exact set
identity.

## Boundaries

This was a read-only review plus memo/evidence write. No code, scorer, archive,
pricing run, Modal dispatch, pointer mutation, submission mutation, or
`upstream/` write occurred. Existing exact rows remain `[contest-CUDA T4 n600]`;
all new statements here are receipt-derived review findings, `score_claim=false`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer
  `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md` plus
  `.omx/state/canonical_task_status.jsonl`; fire when MAIN harvests this memo:
  append a correction that labels the 36-KB quantity an attribution, the 240
  cells and 217 B as partial-sample projections, the boundary identity as
  unjoined, and the coder/pose closures as formulation-scoped.
- **FOLDED** — owner operator; consumer existing
  `ddm_ps2_pr140_update_operator_decision_gate_20260904` in
  `.omx/state/canonical_task_status.jsonl`; fire only on an explicit one-line
  publication decision. Any public-body attribution edit needs explicit scope;
  this review neither infers authority nor changes PR #140.

## LIVE-HYPOTHESES

- Richer counted shared model-row mixing may still save at least 150 B because
  RC2 measures a remaining order-1 gap and leaves precisely that gated design
  open; no tested row closes it.
- Joint tail contexts may still save bytes because CMP1 names them outside the
  tested section-local coder family; no exact price exists yet.
- A causal segment grammar may pay because at least 88.99567740508161% of the
  full misprediction population is edge-supported under the conservative
  retained census, but its topology, address cost, exact residual intersection,
  draw, pose, and whole-tail price remain unmeasured.

## DEAD-ENDS

- Treating either 83,258.632 B or 36,519.391 B of ideal coder attribution as a
  detachable serialized payload: the producer sums logarithms; only the whole
  119,784-B envelope is retained.
- Joining pass 5's 12,377 residual cells to move 37: pass 5 and shipped pass 4
  have different field hashes.
- Calling RW1's 240-cell endpoint n600-measured: it is a fivefold extrapolation
  from a 120-pair screen that RW1 itself invalidated for global actuators.
- Calling RP1's 217-B full pass measured: it is an n120 extrapolation whose real
  subset encode and pose leg were explicitly owed.
- Proving exact rate/seg population identity from edge enrichment: the exact
  joint intersection and class-pair/tangent grammar were not measured.
- Treating a serializer length or ideal attribution as a universal lower bound:
  an achieved length supplies the opposite inequality.
- Treating rank 6/6 as proof that arbitrary new representations are pose-safe:
  it is only a structural precondition on the incumbent carrier.
- Reporting PR #140 as 13 moves stale or free of public AI attribution: the
  receipts give `37 - 23 = 14`, and the live public body names Claude and Codex.

Own-vehicle frontier, **unchanged by this review**: **S 0.13791730003757818 @
180,388 B [contest-CUDA T4 n600]**, archive SHA-256
`670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`.
