# JCB1 SCMDL batched suffix pricing — TERMINAL-REFUSED at the source and supersession gates

**Date:** 2026-09-02  
**Charter:** `.omx/research/charters/ddm_jcb1_scmdl_batched_suffix_pricing_20260902.md`  
**Axis:** `[macOS-CPU advisory / scorer-free rate custody audit]`  
**Typed status:** `TERMINAL-REFUSED-SOURCE-AND-SUPERSESSION-GATE`  
**Score claim:** `false`  
**Pointer moved:** `false`

JCB1 did not build or launch a batched pricer. The charter's first fire condition cannot be
proved from the named RXC1 receipts: those receipts contain no three candidate-specific XOV1
source hashes. More importantly, two of the three requested rows were already closed by the
binding DDS1 ceiling and the third already has a retained exact physical price from JBP1. The
2026-09-02 charter post-dates those results but neither refutes nor explicitly supersedes them.
Silently reconstructing missing pins, reopening closed rows, or wrapping three different causal
graphs in one outer loop and calling that one amortized suffix traversal would be a fake result.

No scientific payload was materialized. No scorer, archive evaluator, Modal job, network call, or
remote process ran.

## Fire-trigger verification

The charter requires "the three XOV1 source hashes named in the rxc1 receipts" to match their live
objects and directs a typed refusal rather than re-derivation if the gate fails. The bounded source
census covered `PREFLIGHT.json`, `BASELINE.json`, `NULL_REPLAY.json`, `SCREEN.json`,
`MANIFEST.json`, every JSON receipt below the RXC1 consumer store, and the owning gen-3 memo.

| Required source surface | What is actually pinned | Live verification | Typed outcome |
|---|---|---|---|
| `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/PREFLIGHT.json` | AFR1 archive and field; JC1/CM1 memos; RXC1, JG2, and RC64 sources; four runtime files | no `xov1`, `gf1`, `peel`, `5506`, or B/H/W candidate pin occurs in the retained RXC1 JSON corpus | `MISSING-REQUIRED-PINS` |
| owning memo `ddm_rxc1_gen3_gate1_verdict_20260901.md` | one aggregate XOV1 memo hash, `ad093da3...ce6345` | live XOV1 memo is exactly `ad093da3358996cb30700a2d0976af2436b10ea570eeada276580336b6ce6345` | aggregate memo pin passes; it is not three object pins |
| retained JG2 executable source | 58,160 B, `e762bead...f76e9` | retained copy still matches; live `experiments/ddm_jg2_tail_reencode.py` is `7fe0769d...f0404` and is an unrelated dirty-worktree modification | `LIVE-SOURCE-DRIFT`; preserved, not worked around |

For diagnosis only, the live XOV1 custody still verifies against XOV1's own receipts: GF1 packet
47,603 B at `87d79345...e1497`, retained GF1 field 117,964,800 B at
`4026c4e2...a5d19`, and B/H/W records 7,965,356 B at `f75f06fe...819aa`. The XOV1 result and
manifest remain `59003d28...094a` and `50bc34e0...a98c`. Those facts cannot be substituted for
the charter's specifically named RXC1 receipt gate; doing so would be the forbidden re-derivation
around missing provenance.

## Supersession reconciliation

The required full-corpus recall found two results that post-date RXC1's queued three-row roster and
pre-date this charter. Neither is listed in the new charter, and the charter supplies no evidence
that overturns either one.

| Requested JCB1 row | Current source authority | Current disposition |
|---|---|---|
| born context/expert with counted GF1 packet | `ddm_dds1_ceiling_readjudication_20260901.md`, commit `a19f9f2555`: full-tuple all-live ideal-coder ceiling about 613 B versus the 47,603 B packet, 77.6x underwater | `CLOSED-BY-CEILING`; a physical replay cannot make the declared context family repay even its packet |
| generator-conditioned peel chain with counted GF1 packet | same DDS1 re-adjudication: candidate 2 is the same packet-dependent tuple family and is its third independent death | `CLOSED-BY-CEILING`; no re-entry without a source-level refutation of the ceiling |
| 5,506-record directed B/H/W support | `ddm_jbp1_joint_batch_price_verdict_20260901.md`, commit `625de245ee`: one exact full-state shipped-G/M re-encode already retained | `MEASURED-REFUSED`; 177,052 B archive, -2,950 B versus AFR1, 7.4641707% of the 39,522.14 B demand, leaving 36,572.14 B unpaid |

The prior Candidate-3 physical artifacts were re-verified live, not recomputed: archive 177,052 B,
SHA-256 `d515e3351784f0d2543045d1d5a9bde03c69bdd190337ec1055cd6bd49d24d9d`;
RC64 stream 110,461 B, SHA-256
`e92115f2bf93596d769ef2b2166c850eb9ac6c8811341e8f44562854ea4649f8`.
Its scientific axis remains `[macOS-CPU advisory / scorer-free EXACT byte measurement]`; this
turn only revalidated file facts. It is not a new JCB1 price and no distortion conclusion follows.

The later timestamp on the JCB1 charter is not, by itself, a provenance-bearing refutation. Under
the common contract and the #1195/#1201 reconciliation rule, a closed candidate can re-enter only
through an explicit correction or new source evidence. None is present.

## Causal batching audit

The three alternatives are not three probability heads over one reusable deterministic suffix:

- The born expert codes unchanged AFR1 `X` while consuming the HPAC model trajectory and the
  packet-decoded GF1 class/distance tuple.
- D3C's peel form is four independently framed binary rungs. Its declared context includes the
  causal binary plane plus the complete already-decoded coarse field at spatial and temporal
  neighbours, including the next-frame coarse value. It is a different graph and decode schedule,
  not another head on the AFR1 five-class traversal.
- The B/H/W row changes `X` at 5,506 sites across 567 pairs. JG2's source documents and executes
  the resulting intra-frame HPAC, prior-frame, boundary-bucket, and persistent-corrector cascades.
  Its causal suffix is therefore not the unchanged-field suffix used by the born row.

A program could place all three loops under one Python `for frame` statement, but it would still
execute the distinct causal states and model evaluations. Reporting that wrapper as one amortized
suffix traversal would be a mechanism substitution, not batching. No executable source was written
under the name `experiments/ddm_jcb1_batched_suffix_pricer.py` because a file with that claim and no
valid shared traversal would violate NO-FAKE.

## Required deliverable rows

| Row | Build / run status | Identity admission | Physical price | Wall clock / amortization | Typed result |
|---|---|---|---|---|---|
| source trigger | read-only census complete | not applicable | not applicable | bounded audit only | `REFUSED-MISSING-REQUIRED-PINS` |
| batched runner | `NOT-BUILT` after trigger refusal | no candidate was admitted | none | not measured | `REFUSED-NO-FAKE` |
| born expert | `NOT-RUN` | no new payload | none | not measured | `FOLDED-CLOSED-BY-CEILING` |
| conditioned peel | `NOT-RUN` | no new payload | none | not measured | `FOLDED-CLOSED-BY-CEILING` |
| B/H/W 5,506 | `NOT-RUN`; exact JBP1 row already exists | prior JBP1 null and exact row remain retained | prior exact 177,052 B, -2,950 B | prior row 698.737711 s; no JCB1 amortization measurement | `FOLDED-MEASURED-REFUSED` |
| review passes | no new Python file exists | not applicable | not applicable | not applicable | `NOT-FIRED` |

The prior-law prediction, "all three prices in at most about twice one 897.675 s replay," was not
tested and is not claimed false by timing. Its prerequisite roster and shared-mechanism premise did
not survive source review. The bounded-reset fallback trigger (`batched cost >= 3x`) therefore did
not fire; inventing a timing from the causal-topology census would be another fake measurement.

## RECALL EVIDENCE

The recall searched the full `.omx/research/` corpus and arm-final receipts by content for
`SCMDL`, `batch-of-proposals`, `batched suffix`, `bounded reset`, `born expert`,
`generator-conditioned`, `generated_class`, `cross_parent_bhw`, `5506`, `D3C`, `GF1`, `RXC1`,
`JBP1`, `DDS1`, `causal schedule`, and `decoder derivable`. It also searched
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC documents, the canonical task
ledger and live hot state for task #1374, and ran
`.venv/bin/python tools/list_canonical_equations.py --json`.

Beyond the charter seeds, the decisive findings were:

- DDS1's append-only re-adjudication closed both packet-dependent requested candidates using a
  derived ideal-coder ceiling rather than the earlier arbitrary 10% screen convention. This removed
  two rows from the runnable roster.
- JBP1 had already extended RXC1 to the exact B/H/W object and retained its complete field overlay,
  stream, archive, ledger, and checkpoints. This removed the only remaining row from the unmeasured
  denominator.
- D3C's source and memo show that the peel candidate is a four-rung binary causal graph with
  complete-coarse-field conditioning, while JG2 shows that changing `X` changes the forward HPAC
  trajectory. This prevented a fake shared-traversal implementation.
- `token_rate_model_direction_dependence_v1`,
  `greedy_set_average_vs_marginal_price_v1`, and
  `decoder_causal_condition_transport_v1` reinforce that direction, marginal set, and conditioning
  prices do not transfer without a physical receiver-causal object. They supplied no licence to
  infer a missing batch price.

The recall therefore changed the plan from build-and-run to fail-closed reconciliation. The bounded
search did not find a post-DDS1 source refutation, an explicit JCB1 supersession statement, three
candidate pins in RXC1 receipts, or an executable one-trajectory representation of all three rows.
These are scoped absences in the named surfaces, not global nonexistence claims.

## Verification and boundaries

- The XOV1 memo, result, run manifest, GF1 packet/field, B/H/W records, retained RXC1 sources, live
  JG2 source, and prior JBP1 Candidate-3 stream/archive were hashed from their bytes this turn.
- The staged index was empty at entry and was not used. Unrelated dirty-worktree files, including
  the live JG2 modification, were preserved.
- `upstream/` writes: 0. Scorer runs: 0. Contest evaluations: 0. Modal calls: 0. Network calls: 0.
- New candidate payloads: 0. Nothing was measured and discarded. The AP store was read-only for
  this terminal audit.
- No fit or surrogate was run. The exact deterministic coder graph and already-retained exact row
  resolved the reachable questions before any fitted stage, satisfying CLOSED-FORM-FIRST.
- The mandated serializer could not write the managed checkout's Git object database
  (`unable to create temporary file: Operation not permitted`). Its AP fallback correctly refused
  because the volume was below the 40 GiB reserve. The intended commit is retained instead as a
  bundle, format patch, and JSONL receipt under
  `/Volumes/VertigoDataTier/pact/ddm_jc1/restartable_exact_coder/jcb1_serializer_fallback/`;
  the shared index remained empty.

## NEXT_IF_RESUMED

- **Disposition: `QUEUED-WITH-A-FIRE-ORDER`. Owner: MAIN charter/provenance owner. Consumer store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`. Fire trigger: a corrected successor charter explicitly reconciles DDS1 commit `a19f9f2555` and JBP1 commit `625de245ee`, pins each admitted candidate's exact source object in a new immutable receipt, and names one receiver-executable causal graph rather than three incompatible graphs.** Only then build and review a pricer for the surviving unmeasured row.
- **Disposition: `QUEUED-STRUCTURAL-LOCALITY`. Owner: task #1374 bounded-reset causal-state builder assigned by MAIN. Consumer store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`. Fire trigger: a counted reset grammar, reset overhead, receiver parser, and exact-vs-full identity screen are all retained and source-pinned.** Measure that changed model on a bounded screen before using it for any outer-loop price.

## LIVE-HYPOTHESES

- A counted bounded-reset causal state may make future exact prices local enough to support a wider
  SCMDL roster. RXC1 proved exact restartability, and its 0/32 terminal reconvergence result locates
  the cost in persistent adaptive state; a receiver-visible reset changes that state law directly.
- A genuinely new decoder-causal conditioning object could reorder the current rate economics.
  QX4 and the canonical transport law show that conditioning changes rankings, but no such object is
  present in the three requested rows after DDS1/JBP1 reconciliation.
- A fitted cross-group schedule for one of SFP1's still-blocked changed fields may remain worth a
  physical price. JBP1 refused fixed-G/M stand-ins precisely because schedule and trained causal mask
  are coupled; that executable fitted object remains absent rather than measured negative.

## DEAD-ENDS

- Re-running the GF1 packet-dependent born expert or conditioned peel without first refuting DDS1 is
  closed: their optimistic all-live ceiling is about 613 B against a 47,603 B counted packet.
- Re-running the 5,506-edit support under shipped G/M is closed on this exact object: JBP1 retained a
  177,052 B archive that pays only 7.464% of the 39,522.14 B cut demand.
- Treating the live dirty JG2 file as the RXC1-pinned executable is closed: its SHA-256 differs from
  the retained source receipt, and the unrelated modification was not ours to overwrite or bless.
- Calling three separate causal-state/model traversals "one batched suffix" is closed as an
  implementation claim. A shared outer loop does not amortize the HPAC trajectory, the four-rung
  D3C graph, or the edited-field feedback cascade.
- Substituting XOV1's own current hashes for the missing three RXC1 receipt pins is closed by the
  charter's explicit do-not-re-derive instruction.
- The amortization prediction and its >=3x falsifier remain untested. No timing-based batching or
  bounded-reset verdict may be inherited from this source refusal.

**OWN-VEHICLE FRONTIER: UNMOVED — S `0.14797617125559104` @ `180,002 B` `[contest-CUDA T4 n600]`, AFR1 archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.**
