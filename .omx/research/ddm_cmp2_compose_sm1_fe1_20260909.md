# ddm_cmp2 — semantic composition on cmp1: A sealed, B scorer proof owed

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: ddm_cmp2. `research_only=true`;
`score_claim=false`. Byte measurements: **[macOS-CPU advisory / scorer-free EXACT
byte measurement]**. The exact frontier is unchanged by this arm.

**SEAL READY — A ONLY:** 180,388 B, **384 B smaller** than the live cmp1 base.
The complete charter remains **PARTIAL**: B is retained at 180,372 B before its
required carrier re-solve, but its fresh seg-neutrality and pose proofs are owed.
The common contract reserves scorer work to MAIN when the arm does not own the
slot; this charter says the local scorer lane belongs to MAIN. No scorer was run.
The one seal therefore selects admissible A and explicitly excludes fe1's code.
This is not a claim that B failed or that its distortion was measured.

Root store (ROOT below): `/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/`.
Seal: `ROOT/SEAL_ddm_cmp2_sm1_fe1_composed.json`.
Candidate runtime: `ROOT/candidate_runtime/`.
Archive SHA: `670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`.
The candidate ID required by the charter is `ddm_cmp2_sm1_fe1_composed`; the seal's
notes state that it contains **SM1 only, with fe1 excluded**.

## RECALL EVIDENCE

Read the complete charter and common contract, PROGRAM, the identical
CLAUDE/AGENTS governing sections including NO FAKE, payload retention, storage,
scorer ownership, pointer authority, checkpoints, review and serializer rules,
and the operating manual and current hot state. The first checkpoint read found
no predecessor for ddm_cmp2. The common contract's August frontier paragraph is
historical; the live pointer is move 36, cmp1, 180,772 B.

Original content recall queried `.omx/research/` with
`semantic.{0,30}mixer|container.break|pose.base.{0,30}configuration`; queried the
canonical research index, `sub015_DAG_*`, design documents under `docs/`, and
`.omx/state/canonical_task_status.jsonl` with
`semantic.{0,30}mixer|container.break|pair.331`. Enumerated the canonical equations
with `tools/list_canonical_equations.py --json`; the retained full registry is
`ROOT/EQUATIONS_RECALL.json`. Read both named equation bodies and anchors.
Read sm1's codec/stager and memo, cmp1's composition/stager and memo, fe1's exact
handoff and rebuild refusal contract, plus the named pose-base and container
lottery memories. No September 8–9 `_directive_` filename was found in the
bounded research filename search.

Beyond the charter seeds:

- The task ledger still contains fe1 ITEM_5 as pending even though fe1's memo
  closes its broad greedy byte search. No new greedy search was launched here.
- `ddm_scg2_seal_custody_followon_digest_naming_20260909.md` distinguishes the
  seal runtime digest from Modal's upload projection. Runtime checks here use
  `tac.candidate_seal.measure_runtime_digest`; no mismatch is inferred against
  the pointer's differently defined runtime hash.
- `ddm_rw1_boundary_local_renderer_weight_foldback_20260909.md` and the canonical
  edit-fee equation bound the empirical fee to its measured section and edit
  domain. They do not authorize a transferred fe1 byte credit or a pose claim.
- The source handoff explicitly says fe1 had **no parse-back of the moved
  shipped bytes**. Its 21→21 render receipt cannot be treated as the requested
  current-candidate neutrality proof. This changed the disposition to a typed
  MAIN scorer queue rather than an inherited B admission.
- The source semantic section is itself under Brotli even after SM1 recoding.
  Thus arithmetic per-symbol cost and full-container cost are separate
  measurements. The claim that the arithmetic coder replaces the lottery is
  tested, not assumed.

The Codex memory registry quick pass used serializer/hash and honest uncommitted
handoff precedents; current CLI behavior and files were checked directly.

## Source binding and prediction before encoding

`INPUTS.json` captures the live pointer, source runtime file hashes, seed,
command, host and SSD preflight. Initial SSD free space was about 69 GiB; the
write probe succeeded. Source is the read-only cmp1 `candidate_runtime` tree.
Its archive SHA is
`66b8d5bb8996f893f867a51e21d35ae8c8a705783fe1089dab1dffae0831b3c0`.

The null ZIP rebuild is byte-identical, including member metadata. The current
semantic SM3R body equals sm1's source body byte-for-byte. VERIFIED-AT-SOURCE:
30,246 B original semantic container versus 29,983 B sm1 container = 263 B;
fe1's retained move-35 archive, selected by its full SHA rather than length,
is 181,460 B versus its actual 181,521 B base = 61 B. The latter is historical
evidence only and is never added to this candidate's prediction.

Before encoding, `PLAN.json` records A = **180,509 B**, saving 263 ±30 B, and B
as another 0–80 B predicted saving. The original SM1 weights at the original
CK2/q11/window24 shape produce exactly **180,509 B** here: no interaction with
cmp1's neighboring sections. The below-200-B falsifier does not fire.

## Exact builds and container sampling

Two bodies: A is unchanged; B changes only the signed 3-bit code
`frame_embed[331,6]` from −1 to 0. Metadata, fp16 scales, HPAC, carrier and tail
stay fixed. Five counted mixer vectors were sampled for each body: the retained
SM1 winner plus four deterministic perturbations of four weights by ±1, using
NumPy seeds `20260909 + sample`. No new semantic codes were searched. Each of
the ten variants has two complete independent encodes and exact decoded-body
identity. Every rider, range stream, actual frequency trace, metadata, weights,
decoded body and full archive is retained.

Each variant is priced in 30 real containers: CK2 on/off × Brotli q9/q10/q11 ×
windows 16/18/20/22/24. All **300 full archives** are retained and parsed back.
Selection minimizes exact archive bytes; SHA breaks length ties by a specific
byte identity. No raw-range ranking is substituted.

| Sample | A archive B | B archive B, carrier unresolved |
|---|---:|---:|
| 0, original SM1 vector | 180,490 | 180,412 |
| 1 | 180,449 | 180,393 |
| 2 | 180,416 | 180,478 |
| 3 | **180,388** | **180,372** |
| 4 | 180,492 | 180,476 |

Both selected shapes use CK2, q10, window16. A's 384 B gain decomposes as:
263 B original SM1 credit + 19 B container search on that vector + 102 B from
the sampled-vector winner at its own container optimum. This is a same-object
measured decomposition, not additive credit from ancestor archives.

Selected B SHA:
`6e500244bf4dc4cc78e6afc7473cf44d95b73ff6977178d7299ece9a0e808fbf`.
It is **not admitted**. The current carrier is retained unchanged; a fresh
re-solve may change its bytes and distortion, so 180,372 B is not the final
pose-resolved B price.

## Actual per-symbol price, not a histogram model

The encoder's actual integer-frequency decisions are retained for all 231,168
bit events per encode. The edited symbol occupies events 9,882–9,884 (zero
based). `RESULT.json` records separate costs for its three bits and all causal
adaptation afterward. No modelled 1.49-bit order-zero price is used.

| Matched A/B mixer | Changed-symbol Δbits | Whole-stream Δbits | Range ΔB | Searched archive ΔB |
|---|---:|---:|---:|---:|
| Original SM1 vector | −1.418239269 | −1.328608241 | 0 | −78 |
| Selected sample 3 | −1.537546599 | −1.303058107 | 0 | −16 |

At the original fixed CK2/q11/window24 shape the first row's archive delta is
−29 B, not −78 B. **The arithmetic price does not replace the outer-container
effect.** Roughly 1.3 bits of total probability cost change rounds to zero whole
range bytes, while the range-byte pattern changes and Brotli prices it
differently. This closes the charter's replacement premise at INSTANCE scope;
it does not close arithmetic coding or the pair-331 move as a family.

## Public boundary, section census, and retained seal

The actual F26 entrypoint reaches its token-decoder call after loading all
semantic weights. A has **38/38 bit-identical tensors** against the current base.
B has 37 identical tensors and exactly one differing scalar in the remaining
tensor: `[331,6]`, −0.873046875 → 0. This verifies the intended code change but
does not measure its rendered flipped-cell count or its pose effect.

Both staged readers independently encode their selected body twice more and
produce the exact selected full archive bytes. `STAGED_TWINS_A.json` and
`STAGED_TWINS_B.json` bind those retained archives. The receiver's own codec is
used in these final twins, not just the experiment-library decoder.

| Section | Base B | A B | Identity |
|---|---:|---:|---|
| RX1 header | 14 | 14 | only semantic length changes |
| HPAC | 11,911 | 11,911 | byte-identical |
| semantic | 30,246 | 29,862 | recoded; identical public weights |
| carrier | 18,586 | 18,586 | byte-identical |
| tail | 119,915 | 119,915 | byte-identical |
| ZIP framing | 100 | 100 | same source metadata |

The changed reader census contains only SM1 dispatch seams, the new generic
SM1 reader, archive pins and manifest. No learned data moves into code. All
24 mixer coefficients remain counted inside the semantic section. The model,
field, carrier and generic generator mechanisms are retained prior work; this
arm contributes current-base composition, sampled exact pricing and proofs.

Both A and the live-base public smoke probes reached actual token decoding;
both shell probes reached the CUDA gate. `_public_smoke_problems` returned no
problems. Probe subprocesses are sequential, bounded at 60 seconds and killed
and reaped as process groups; no renderer/scorer completion is claimed. Public
CPU entrypoints require four threads, with at most two computational processes
active across the arm. No detached jobs, Modal fire, or upstream/live/sibling
tree writes were performed.

`make_candidate_seal.py` created the one A seal; a separate `validate_seal`
returned **SEAL VALID**. The pointer was re-read at stage/seal/validation.
`POINTER_REFUSAL.json` also verifies refusal against a retained malformed pointer
copy; the actual pointer was never modified. A changed live pointer requires a
new retained generation and rebase before firing.

## Score boundary and equations

`BASE_COMPONENTS.json` verifies the bound existing cmp1 T4 receipt and recomputes
its S from `d_seg=0.00010698`, `d_pose=0.00000505`, and 180,772 B, preserving the
receipt's eight-decimal component precision. Rate delta for A:
`−384 × 25 / 37,545,489 = −0.0002556898379989124`.
Conditional A projection: **S 0.13791730003757818**, holding those base
distortions. This is DERIVED, not a new contest score; sub-0.12 is not achieved.

The canonical equation registry now carries `ddm_cmp2_lossless_A_20260909` on
`model_section_adaptive_recode_ceiling_v1` and `ddm_cmp2_single_code_B_20260909`
on `model_section_edit_container_break_fee_v1`. `EQUATION_ANCHORS.json` preserves
the appended events. Provenance names and hashes the actual `RESULT.json`
source artifact; the archive is separately bound in its empirical output.
Shared registry files were already dirty and are not swept into this arm's
commit. The exact new events are preserved for MAIN's separate reconciliation.

Six integration hooks: sensitivity and Pareto use A's measured −384 B at
construction-identical model state; bit allocation N/A because no depths or
capacity changed; exact dispatch routes through the MAIN-owned seal/fire order;
continual learning receives the two equation anchors; disambiguation separates
symbol cost, range bytes, container bytes and unmeasured scorer debt.
Canonical tools are reused for source geometry, public decoding, seal and
serializer because these are the receiver/custody authorities, not new models.

## Verification, durability and scope

Real-input verification covers null identity, 20 initial encodes, 300 containers,
four further staged-reader encodes, A/B public tensor proofs, the public smoke
pair, exact archive/section census, pointer refusal and seal validation. Correctness
lint passes with only C408 style excluded. Two source review passes cover byte
custody, causal frequency observation, selected-code geometry, public identity,
resume behavior, pointer guards and the seal's exclusion of B. The shared
assumption challenged was that a per-symbol arithmetic price eliminates the
outer-container lottery; the measurements show it does not.

All stages checkpoint immutable artifacts atomically and resume from the owned
ROOT. No bulk is deleted. Storage preflight blocks below 256 MiB free. The
manifest and commands retain payload paths, bytes and hashes. Source revision
history in PLAN and the final code receipt distinguish initial implementation
from later staging/review additions. No writes to memories were made.

Reproduction: run `experiments/ddm_cmp2_compose.py` with `--resume-from ROOT`;
stages `init`; `encode --build A|B --sample 0..4 --trial 1|2`; matching `sweep`;
`stage`; `proof`; `stage-b`; `proof-b`; `staged-twins --build A|B`; `combine`;
`seal`. Each invocation is a separate process; use the retained final source
and imported helper pins. Existing source-stage records are immutable. MAIN
must add the distinct B scorer/re-solve leg before any B admission.

Serializer outcome and any sandbox fallback bundle are recorded in
`ROOT/SERIALIZER_RESULT.json` and `ROOT/serializer_fallback/`. The final
completion receipt distinguishes an actual main-branch landing from fallback
custody. No commit is claimed merely because a bundle exists.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store ROOT/B_ADMISSION.json;
  trigger:** MAIN harvests this handoff while it can use its scorer lane. On the
  current base's own renders/configuration, measure its n600 pose vector; prove
  B's shipped parse-back has exactly the base's n600 flipped-cell count; re-solve
  pair 331 from live carrier coefficients with the unmoved-render control;
  retain the per-pair receipt, rebuilt carrier and all final B bytes. Compare
  the actual pose-resolved B against A, without carrying −61 B or old pose.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store ROOT/MAIN_EXACT_RESULT.json
  and the canonical frontier pointer; trigger:** B is either closed or fully
  admitted, and the selected seal validates against the still-current pointer.
  Fire exactly one uniquely claimed T4 n600 evaluation. If B wins, replace A's
  selection with freshly proved and sealed B before that single fire. Retain and
  harvest the result; recompute components and apply canonical admission.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN or assigned successor; consumer store
  ROOT/REBASE_HANDOFF.json; trigger:** the pointer changes before selection or
  firing. Rebase to the new live tree, retaining this generation, and repeat
  all affected byte/public/scorer proofs and seal validation.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store ROOT/LANDING_RECEIPT.json
  and canonical equation/lane stores; trigger:** this handoff is harvested in a
  Git-writable context. Apply any serializer fallback only for its named files,
  and reconcile this arm's two equation events and lane row without sibling
  working-tree content.

## LIVE-HYPOTHESES

- A's exact T4 row should realize its 384 B rate saving: the full public semantic
  state and all remaining sections are identical. Actual T4 score/runtime remain
  unmeasured for this archive.
- B may retain its extra 16 B after the pair-331 solve: the code change is exact
  and prior renders were neutral, but neither the new parse-back neutrality nor
  current-base pose payment has been established.

## DEAD-ENDS

- **FOLDED, INSTANCE:** carrying fe1's −61 B; real SM1 pricing gives a different
  result, and the final carrier has not yet been re-solved.
- **FOLDED, INSTANCE:** treating arithmetic per-symbol cost as full-archive cost;
  both measured mixer comparisons save zero whole range bytes while Brotli
  yields different archive deltas.
- **FOLDED, INSTANCE:** admitting B from model-only or ancestor render identity;
  the public proof changes one real scalar and does not establish scorer neutrality.
- **FOLDED, INSTANCE:** length-only stream identity; every selected archive is
  pinned by bytes/SHA and reproduced by full staged twins.

OWN-VEHICLE FRONTIER: S 0.13817298987557713 @ 180,772 B
[contest-CUDA T4 n600], unchanged by this arm.
