# ddm_cmp1 — compose RC3 model recoding and TC1 tail mixing on live sj1 pass 4

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: ddm_cmp1. Local charter deliverables complete; MAIN evaluation queued.
Axis: `[macOS-CPU advisory / scorer-free EXACT byte measurement]`; `score_claim=false`.
No scorer or Modal dispatch is authorized for this arm. MAIN consumes the completed seal.

The base is lane `ddm_sj1_t4_token_predistortion_pass4_20260909`, archive
`b0ca809ce2c657dfce97e73148a83b9b20c128461ced4c1f6ce1b386ddfd1d20`, **181,521 B**,
at `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass4/candidate_runtime`.
The canonical pointer was read before materialization. Its score is **0.13867171823146562
[contest-CUDA T4 n600]**. The common contract's August frontier paragraph is historical;
it is not the current base. The exact pointer is unchanged by this arm. It was re-read at seal creation and validation.

## RECALL EVIDENCE

Read the complete charter and common contract, PROGRAM.md, operating manual,
CLAUDE.md/AGENTS.md governing no-fake, retention, storage, review and serializer rules,
live hot state, canonical pointer, source seals and both predecessor memos.
Checkpoint read for `ddm_cmp1` found no predecessor record.

Content recall searched `.omx/research/` memos and arm receipts with
`shared.mixer|model.section.*recode|container.break|tc1|rc3`; searched the canonical
research index, `sub015_DAG_*` FEED blocks, `docs/` design/spec material and
`.omx/state/canonical_task_status.jsonl` with
`shared.mixer|model.section.*recode|token.tail|container.break`. Enumerated the actual
canonical-equations registry through `tools/list_canonical_equations.py --json`.
Relevant equations are `model_section_adaptive_recode_ceiling_v1` and
`token_tail_context_mixing_bound_v1`. The registry's existing empirical anchors were
read, preserving their oracle-versus-counted-coder scope.

Findings beyond the charter's seed memos:

- `ddm_scg2_seal_custody_followon_digest_naming_20260909.md`: seal runtime digests and
  Modal upload-projection digests have different definitions. This changes the custody
  check to name `tac.candidate_seal.measure_runtime_digest` explicitly, and forbids
  comparing it directly with the historical Modal-projected tree hash.
- The current sj1 move-35 packet locates the admitted **subset** field, not its full
  proposal field. The source has all 600 admitted planes, with non-admitted pairs
  carrying the base. This changes input selection to `admission_pass4/field_admitted.npz`.
- The task ledger's rc1 ITEM_1 is model-prior work; it does not own or supersede the
  distinct token-tail composition. The historical DAG envelope references do not
  provide a competing current-object composition.
- fe1's actual body confirms that two 30,246 B streams differed at 30,129 bytes;
  container shape must be established by exact identity. Its unchanged-body container
  search found no free gain on that older object. Neither its numeric optimum nor its
  old archive SHA is transferred as a current-object control.
- Read the JG2 full-state checkpoint implementation, including its historical omitted
  corrector-state defect. The new streaming encoder captures all dynamic corrector
  arrays and the TC1 mixer, together with both RC64 states in one atomic NPZ.

No additional current-object composition was found in the searched index/design
surfaces. No upstream or live/sibling-tree writes are required.

## Closed-form prediction and verified source bytes

Before encoding: `B_expected = 181521 - 201 - 549 = 180771 B`, saving 750 B ±20 B.
The charter falsifier is a saving below 700 B. The 549 B predecessor tail saving is a
prediction for this changed field, not a measured transfer.

`/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/INPUTS.json` records source facts,
SHA-256, the original field NPZ, the full pointer capture and the prediction.
The actual current sections are:

| Section | Bytes |
|---|---:|
| header | 14 |
| HPAC | 12,112 |
| semantic | 30,246 |
| carrier | 18,586 |
| tail | 120,463 |
| ZIP framing | 100 |

VERIFIED-AT-SOURCE: current HPAC equals tc1's old HPAC. RC3's retained HPAC is 11,911 B,
giving exactly 201 B. The retained tc1 predecessor source/candidate tails differ by
549 B. No claim of 549 B on the new field is made before its encode completes.

The source null rebuild is **byte-identical** to the complete current archive, including
ZIP metadata. The full admitted field is 117,964,800 uint8 symbols with SHA-256
`361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94`.
The unchanged 35 counted int8 weights have SHA-256
`35d56667911d1b593434e30c5334915ca516cbf8f547d54b32c091246ca75f3b`.

## Implementation and current verification

`experiments/ddm_cmp1_compose.py` reuses the shipped sparse HPAC trajectory and JG2
corrector capture. It streams the actual float32 coding rows into the unchanged TC1
mixer and independently encodes both the original control and mixed candidate.
Two separate runs recompute their own complete trajectories. No dense HPAC shortcut,
constant-probability replacement, parameter fitting or prefix-based research verdict
is introduced. The original rows reach the original coder without reconstruction.

The restart implementation control passed: frame 1 saved, fresh process resumed to
frame 2, versus an independent uninterrupted two-frame run. All **164 checkpoint
arrays**, the original coder stream and the mixed coder stream match exactly.
Receipt: `RESUME_CONTROL.json` in the arm store. This is a real-input implementation
test, not a population compression result. Full n600 encode and receiver proofs also pass; see the measured composition and public proof below.

The copied current parser and the composed RC3 parser separately restore **17,770 B**
of identical IHS1 model content, SHA-256
`817281908d993d89f62349fa466ad53f204b7e83d36e29ba0edc30cf2cb8f085`.
The model-only current-base archive is **181,320 B**, exactly **201 B smaller**.
Receipts: `MODEL_PROOF_source.json`, `MODEL_PROOF_candidate.json`, and
`READER_SOURCE_CENSUS.json` records the early reader census; `RESULT.json` and
`READER_SOURCE_CENSUS_FINAL.json` provide the completed composition census. The malformed-pointer control separately
refuses a retained copy naming a changed archive; the actual pointer stays untouched.

An independent source reviewer found no confirmed correctness defect in the frozen
streaming encoder or the composed staging code. A provisional concern about the
canonical provenance builder's parameter name was withdrawn after checking its body:
the parameter becomes `source_sha256`. The final staging code names the archive
path/hash together explicitly, while its equation anchor separately cites RESULT.json.
The independent review is source-level evidence; the full real-input runs below supply execution evidence.

Source code and small manifests stay in the repository/Vertigo arm tree. Every
materialized coder stream is retained. Complete compressed checkpoint payloads are
under `/Volumes/APDataStore/pact/ddm_cmp1_compose/encode/`; their source/config/build/
input-bound receipts stay on Vertigo. The Vertigo arm stays below 1 GiB even though
the volume now has more free space than the charter's stale estimate. No bulk is
deleted; resumption uses the last complete atomic stage. Every write path checks disk
space. Detached launches use the canonical launcher, best-effort nice 10 and a
780-second process-group watchdog. At most two one-thread encode workers run together.

## Measured composition

Both independently computed full-n600 sparse trajectories reproduce the shipped
120,367 B original RC64 stream byte-identically. Their new mixed streams are also
byte-identical: 119,779 B, SHA-256
`ffcd64bd3013f313538d3a426fdccdd8ac82c5fbcb19b382d7f935c5163bb427`.
No old-field payload was substituted. All encoder streams and every complete stage
checkpoint remain in custody, including independent matching checkpoints at 190 and 460.

| Same live base, exact byte axis | Archive bytes | Saving |
|---|---:|---:|
| Live sj1 pass 4 / exact null rebuild | 181,521 | 0 |
| RC3 model only | 181,320 | 201 |
| TC1 current-field tail only | 180,973 | 548 |
| RC3 + TC1 composed | **180,772** | **749** |

Candidate archive SHA-256:
`66b8d5bb8996f893f867a51e21d35ae8c8a705783fe1089dab1dffae0831b3c0`.
The counted tail is 119,915 B: retained 96 B prefix + 5 B rider header + 35 B
unchanged mixer weights + 119,779 B RC64 stream. HPAC is 11,911 B. Semantic
30,246 B and carrier 18,586 B are byte-identical. Header changes are the tail
reader flag and HPAC length; ZIP framing remains 100 B.

Forty composed shapes and forty same-base TC1-only shapes were actually retained
and parsed back: plain RC64 or Brotli q9/q10/q11 with windows 22/23/24, each inside
ZIP STORE or DEFLATE levels 1/6/9. Plain RC64 with ZIP STORE uniquely wins at
180,772 B. The nearest Brotli alternatives are 180,777 B; equal lengths across
windows have different SHA-256 values, so length is not an identity control.
The independent repeat stream was separately repacked into the exact winning
archive bytes. There is **zero measured container interaction** against the two
same-base standalone savings. The one-byte miss against the older 750 B prediction
is the changed field's tail saving (548 versus 549), not container interaction.
The 730–770 B prediction interval holds; the below-700 B falsifier did not fire.

The rate-only delta is `-749 * 25 / 37545489 = -0.0004987283558885063`.
Conditional projected S is **0.1381729898755771**. This is arithmetic on the live
exact row, not a measured new score, and does not reach sub-0.12. `BASE_COMPONENTS.json`
recomputes the source score from the bound exact receipt's distortion components.

The actual public receiver saved frame 1, resumed to frame 2, and matched a fresh
two-frame decode across all **100 checkpoint arrays** and the whole checkpoint
file. `PUBLIC_RESUME_CONTROL.json` is an implementation restart check, not a
prefix-based research verdict. `PUBLIC_SMOKE_VALIDATOR.json` reports no problems
for both roles: actual f26 token-entry reach and actual bash inflate.sh CUDA-gate
reach. Full n600 public-field identity also passed, as detailed below.

## Public identity and seal

The actual `runtime.f26_inflate.inflate_archive` CPU token phase restored all **600
planes / 117,964,800 uint8 symbols**, with exact field SHA-256
`361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94`.
This is the admitted live field, proved independently from the encoder. The receiver
stopped at its durable token-stage checkpoint before RGB rendering. No scorer,
contest-CPU evaluation, CUDA execution, or Modal fire was performed by this arm.

The 780-second watchdog stopped the first full public leg at a retained 450-frame
checkpoint; the resumed suffix finished successfully. Its 260.7523469924927-second
token timing is **suffix-only**, not an n600 runtime claim. Logit/CDF digests in that
receipt are also suffix-only; the decoded token hash covers all n600. Watchdog peak
RSS printed zero because sandbox process visibility was unavailable; that is not a
measured zero-memory claim. Best-effort nice was recorded as unapplied. The two
one-thread encoders finished before the public decoder and smoke pair ran.

**SEAL READY:**
`/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/SEAL_ddm_cmp1_rc3_tc1_composed.json`.
Candidate `ddm_cmp1_rc3_tc1_composed`, axis `contest_cuda`, **180,772 B**, archive SHA
`66b8d5bb8996f893f867a51e21d35ae8c8a705783fe1089dab1dffae0831b3c0`.
Seal creation and a separate `validate_seal` call both returned **SEAL_VALID** with
no problems, against the unchanged live pointer. Runtime digest, using
`tac.candidate_seal.measure_runtime_digest`, is
`19185aea0ad1a18aa2f65748a3a63fae61ffd15f98ddae889209cafbf6479754`
(47 shippable files, 984,278 B). The nine receiver pins include both entrypoints,
f26, RC3/RC2 model readers, TC1 tail reader and checkpoint helper. The admit bar is
strictly negative exact dS against live sj1 pass 4; its report-8dp bound is computed
from the bound source receipt by the canonical seal tool.

The mechanisms are retained RC3 and TC1 work. This arm contributes the current-base
composition, full re-encoding, independent controls, container measurement and custody
handoff. It does not claim a newly invented codec or a new measured contest score.

## Retained evidence and reproducibility

Root: `/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/`.

- `INPUTS.json`, `BASE_COMPONENTS.json`: live pointer, source objects, seed 20260909,
  real source savings, pre-encode prediction and exact source score components.
- `RESULT.json`: all 80 archive rows, source/candidate section census, twin identity,
  standalone comparisons, archive facts and conditional score arithmetic.
- `retained/`: every source section, field, model body, original null archive, every
  container candidate and independently packed repeat archive; no losing payload discarded.
- `encode/{primary,repeat}/`: both complete original/mixed streams, envelopes and
  source/config/build-bound receipts. Complete NPZ state payloads remain under
  `/Volumes/APDataStore/pact/ddm_cmp1_compose/encode/`.
- `MODEL_PROOF_{source,candidate}.json`, `RESUME_CONTROL.json`,
  `STALE_POINTER_CONTROL.json`, `PUBLIC_RESUME_CONTROL.json`,
  `CONTAINER_TIE_NEGATIVE_CONTROL.json`: actual implementation and identity controls.
- `public_identity/PUBLIC_FIELD_IDENTITY.json` and
  `public_identity/public_stage_checkpoint/tokens_cpu_stage_complete.u8`: full actual
  public field and its hash/size receipt; complete receiver checkpoints are retained.
- `PUBLIC_SMOKE.json`, `PUBLIC_SMOKE_VALIDATOR.json`, `SEAL_COMMAND.json`,
  `SEAL_VALIDATION.json`: paired public entry evidence and exact seal reproduction.
- `RUN_PROVENANCE.json`, `SOURCE_REVIEW_FINAL.json`, `source_release/final/`:
  executed source, command/log custody, final source hashes and review evidence.
- `SERIALIZER_RESULT.json`: actual serializer return, commit/bundle/receipt custody.
  The shared staged index is not touched. An SSD bundle is retained if shared Git
  object writes are denied; MAIN must land that bundle before claiming a shared commit.

Two review passes cover each final Python source. AST parsing, shell syntax, Ruff
F checks excluding the frozen unused-import F401, real encoder and public-receiver
restart controls, full original-stream controls, independent encodes and archives,
full receiver identity, and seal validation passed. The equation API rejected the
initial signed residual before any equation write; the final normalized magnitude is
`abs(749-750)/750 = 0.0013333333333333333`, independently reviewed and actually accepted.
The signed one-byte prediction miss remains in `interaction_vs_prior_savings`.

## Equations and harvest disposition

`update_equation_with_empirical_anchor` wrote anchor
`ddm_cmp1_rc3_tc1_live_sj1_composition_20260909` to both
`model_section_adaptive_recode_ceiling_v1` and
`token_tail_context_mixing_bound_v1`. Each anchors the **180,772 B** composition,
**749 B** saving, full identity evidence and exact archive path/hash; `score_claim=false`.
`EQUATION_ANCHORS.json` and the repository file
`.omx/research/ddm_cmp1_equation_anchors_20260909.json` preserve this arm's portable
anchors. The shared registry also contains unrelated uncommitted SM1 work, so the
serializer includes the portable receipt instead of absorbing the shared whole file.
MAIN's harvest must preserve/replay these two anchors through the canonical helper.

`FIRE_ORDER.json` is **QUEUED-WITH-A-FIRE-ORDER**, owner **MAIN**, consumer
`/Volumes/APDataStore/pact/ddm_cmp1_t4_rc3_tc1_composed_20260909/MODAL_REMOTE_RESULT.json`.
It records the exact single-axis command and archive/seal hashes. Trigger: harvest
the source bundle and anchors, validate the seal against the unchanged live pointer
and runtime, claim the unique lane, then fire one exact contest-CUDA n600 evaluation.
No dispatch is claimed. The charter's existing FE1-after-promotion composition order
remains folded into MAIN's existing FE1 ownership; this arm starts no FE1 work.

Live frontier unchanged: **S = 0.13867171823146562 @ 181,521 B
[contest-CUDA T4 n600]**, lane `ddm_sj1_t4_token_predistortion_pass4_20260909`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** harvest the serializer bundle and preserve both
  equation anchors via the canonical helper. Consumer stores: shared Git HEAD and
  `.omx/state/canonical_equations_registry.jsonl`. Fire trigger: this arm's verified
  bundle/receipt and portable anchors are harvested, before candidate dispatch.
- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** execute the exact command in `FIRE_ORDER.json`.
  Consumer store: `/Volumes/APDataStore/pact/ddm_cmp1_t4_rc3_tc1_composed_20260909/MODAL_REMOTE_RESULT.json`.
  Fire trigger: harvested source/anchors, current SEAL_VALID, unchanged base/runtime,
  and MAIN's unique active lane claim.

## LIVE-HYPOTHESES

- The composed candidate should improve exact contest-CUDA S by
  0.0004987283558885063: all scorer-driving model/field/semantic/carrier content is
  identical and the archive is 749 B smaller. Full T4 rendering, timing and exact
  evaluator confirmation remain untested; MAIN's queued row is the consumer.

## DEAD-ENDS

- **INSTANCE:** copying the predecessor tail payload onto live sj1 is invalid because
  pass 4 changed the admitted field. Both full new encodes reproduce the current
  original stream; the older 549 B saving becomes 548 B here.
- **INSTANCE:** tested Brotli/DEFLATE container variants provide no extra saving on
  this composed body. The unique plain/STORE winner is at least 5 B smaller.
- **INSTANCE / identity method:** equal archive lengths do not prove identical bytes.
  Two current 180,777 B window variants differ at 9 positions; exact SHA/bytes govern.
- **INSTANCE:** a container-interaction explanation for the one-byte prediction miss
  is closed by same-base standalone pricing: 201 + 548 = 749, with zero interaction.
