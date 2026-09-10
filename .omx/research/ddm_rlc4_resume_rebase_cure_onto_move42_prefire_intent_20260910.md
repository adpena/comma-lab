# ddm_rlc4 — real move42 cure; frozen intent refused on manifest-bound receiver identity

**STOP — `PREFIRE_RISK_EVIDENCE_REFUSED: receiver risk endpoints differ`.** The real
candidate is **180,178 B**, saves **60 B**, and its complete cold public output is
byte-identical to move42. The unchanged producer passed the non-timing checks,
then refused the timing-risk endpoints. **No valid intent, seal, authorization,
dispatch, exact candidate score, or pointer move exists.** This is an INSTANCE
contract/custody conflict, not a negative verdict on the compression mechanism.

`research_only=true; score_claim=false` · `[no-triality] [p0-ledger-ok]`

| Quantity | Result | Authority |
|---|---|---|
| Candidate archive | 180,178 B; `eaf17a7038a4671bf39c3071192b15eec7d7513d471d520b6d13bc9acf94a60b` | Real serialized bytes |
| Source archive | 180,238 B; `f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f` | Move42, rehashed |
| Candidate RC64 twins | 119,529 B each; `e47a5c2402972c25d9718f55a2fe7bd806a1ded7ced55f9474f1e6e845483914` | Two independent full-n600 encoder states |
| Source RC64 control | 119,613 B; reconstructed exactly | Full-n600 real encoder |
| Counted contents | 40 B fitted weights + 19 B geometry; 60 B including variant | Parsed real rider |
| Cold public raw | 3,662,409,600 B; `1db04341d5ae9972d6296831a9e822d50a5d0f25ef77af378eae75b69e917159` | Literal `inflate.sh`, all 600 pairs, macOS-CPU advisory |
| Raw comparison | Every byte equal to retained move42 raw; no resume; token cache DISABLED | Actual public execution and full byte comparison |
| Manifest | 49 rows; all hashes pass; generated outside tree | Re-read exact dependencies |
| Census / smokes | 51 shipped files covered; four candidate/frontier checks PASS | Source/custody review and actual setup/gate smokes |
| Runtime digest | `816f72e31639c8365ef6d5aa3f0738a235fed8d6dfd879d15d1838a5fa847333` | `measure_runtime_digest`, 51 files, 1,009,537 B |
| Actual normalized receiver | `27948d3d5ac0c32bd6eda8c0cfd08b2b488cfaa60c44ce3e98ba2689d7edbb8a` | `measure_receiver_digest` |
| Required normalized receiver | `b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d` | Frozen pr12/RLC4 gate, live RLC1 reference revalidated |
| Historical risk arithmetic | 1,032.7250802956457 s, below 1,260 s | Diagnostic projection only; risk gate REFUSED |
| New public shell wall | 999.3320274169964 s | Loaded macOS-CPU diagnostic, not T4 authority or a timing window |
| Assigned store | 6,261,596,774 B over 2,115 files at final storage census; below 8 GiB | Filesystem measurement |

Candidate custody: `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime/archive.zip`.
Evidence packet: `.omx/research/ddm_rlc4_20260910/`.

## The exact conflict and real producer result

The six receiver paths reproduce RLC1's reviewed cure on move42. With the archive,
the mechanism delta is the chartered seven paths. The separately required manifest
refresh is the eighth changed file. Comparing the actual normalized receiver rows
with the timed RLC1 reference finds **one differing path out of 50:
`MANIFEST.sha256`**. The other 49 normalized shipped files match exactly.

The manifest differs on exactly the `inflate.py` hash row:

```text
RLC1:  93303ee37fb9161de3c8adfd346d8738b6fe07168dbc4a89820c3ea2e97f38c9  inflate.py
RLC4:  c186a94a704b32fad506cb0963665ab62e3bc855195837180c5c1cef7512f76e  inflate.py
```

Those hashes differ because the archive SHA/size pins correctly name different
counted bytes. The receiver digest normalizes those two assignments in `inflate.py`
but hashes `MANIFEST.sha256` verbatim. The regenerated manifest SHA is
`7c2b977a968dbd0512ce817dd7951e06d817b125875aa410a4c0f0d863345147`, versus RLC1's
`98993a00b6f2eb0f0ef8454bfbeea40ccdae7c18012a5a25cdfd645dfa4fc8f4`.
`RECEIVER_MANIFEST_CONFLICT.json` and `MANIFEST_REFERENCE_DIFF.txt` preserve the proof.
Keeping RLC1's manifest would retain the required digest but recreate pr9's stale
manifest failure. I did not do that, alter the timed reference, or patch either contract.

The literal charter CLI first returned **rc 1** with
`ModuleNotFoundError: No module named 'experiments'` while `validate_prefire_risk`
called the unchanged T4-leg validator. Its full argv and traceback are retained.
The same CLI with only the actual repository root and `src` added to `PYTHONPATH`
loaded the real frozen dependency and returned **rc 3**:

```text
PREFIRE_RISK_EVIDENCE_REFUSED: receiver risk endpoints differ
```

No source, validator, hash, endpoint, or semantic gate was changed for that invocation.
The typed receipt is `PREFIRE_REFUSAL_acb9c1ab3cb3443881b2af2538547635.json`.
The producer's call order proves `_pf_contract`, `_pf_identity`, `_pf_pointer`, and
`_pf_evidence` completed before the risk refusal. The failed new intent was deleted
by the frozen producer itself. **Intent path, file SHA-256, bytes, and canonical
digest are therefore absent**, explicitly null in `DELIVERY_STATUS.json`.
Normal `validate_seal` and normal `--seal` controls were **not exercised on a valid
intent**, because none exists. No fixture or fabricated replacement was substituted.

## Real mechanism and literal census

Reused `experiments/ddm_rlc3_move42_trace.py` unchanged from its landed source.
Its two full encodes reconstruct move42's shipped stream; retained field SHA is
`d5248c775e49d8ac4299d2b0b60ffe5cffbeaafe348ef57986a169d3f118322b`.
The rebase delegates to RLC1's landed real encoder with explicit move42 bindings,
two distinct candidate arithmetic states, and a third source-control state. It
retains every frame checkpoint, both envelopes/streams/riders, both complete member
payloads, both full archives, the source archive reconstruction, and the cold raw.
No length-only proxy or ledger sum supplies the candidate price.

Measured accounting: token stream **−84 B**, additional fitted weights **+5 B**,
counted geometry **+19 B**, net archive **−60 B**. The prefix retains move42's
header, HPAC, semantic, carrier, and tail header. The raw comparison independently
rules out silently reverting its field/carrier to move40.

Literal-census acceptance: **Lane class and row band counted; all remaining 14
geometry fields counted; all 40 fitted weights counted; integer Q16 geometry
selected by the receiver; TC4 maps absent; no new scorer/GT/source-video content
in the cure; inherited online mixer floating arithmetic remains reachable.**
The 17 geometry fields parse as
`[1,128,320,16,4,4,36,2,2,3,24,0,1,2,4,8,16]` from the 19-byte struct.

The independent enumeration covers 8,114 occurrences across the shipped text
dependencies: Python NUMBER and digit-bearing STRING tokens, plus conservative
numeric text runs in C/shell, including flags, identifiers and comments. The six
changed receiver files contain 628 such occurrences. This broader denominator is
not compared numerically with pr9's 443 numeric-only occurrences. The census also
hash-binds all 51 shipped files, checks exact reference bytes outside the two archive
pins/manifest, and retains each enumerated occurrence with source location.
`LITERAL_CENSUS.json`, `LITERAL_OCCURRENCES.json`, and `LITERAL_CENSUS_COUNTS.json`
contain the counts and scope. CLEAR applies to the rebased RLC1 cure, not an invented
fresh compliance proof for every historical dependency or a whole-integer receiver.

## Risk, conditional arithmetic, and boundaries

The completed move40 `t4_direct` leg still validates at **990.053829427 s**. Its
exact 1,553-byte file SHA is `ed929b24cc876bf8ffabc3004b856decbb5e73d0fee13c1d3c87d91d659f9521`.
The risk receipt retains the actually refused RLC1 g3/g4 diagnostics at
829.0324737499905 and 831.502915124991 s, and the reviewed move40 diagnostic
denominator 797.1459791249945 s. Thus the observed fraction is
0.043099930125356956 and the projection is **1,032.7250802956457 s**.
The numeric ceiling passes; receiver correspondence fails. No diagnostic became
timing authority, and the new 999.332 s public wall is not substituted into this chain.

DERIVED, CONDITIONAL ONLY:
`25*(180178-180238)/37545489 = -0.00003995153718733028`;
holding move42's reported components gives **S 0.13743655372199698**.
The strict `-2e-5` bar permits at most 180,207 B. This archive clears the byte bar
by 29 B, but **there is no candidate contest-CUDA/CPU score**, no timing clearance,
and no sub-0.12 achievement. Full raw identity here is same-host macOS-CPU advisory,
not a new cross-host or CUDA determinism result.

No Modal, cost authorization, first-measurement fire, completion, scorer, GT decode,
timing window, PR edit, or pointer update occurred. No `upstream/`, sealed tree,
sj1 pass6 directory, protected file, or frozen contract source was edited. No
payload was deleted, moved, or deduplicated. The public shell's ordinary native
scratch cleanup ran only after the exact builds/source/argv were retained. The
shared staged index is checked independently of the serializer's isolated index.

## RECALL EVIDENCE

Read the RLC4 charter, inherited RLC3/RLC2 charters, full common contract, PROGRAM,
operating handoff, live hot state, governing NO-FAKE/retention/review/serializer
clauses, pr9/pr12, the frozen producer/consumer code, and current dated directives.
The Codex memory-registry search `resume|rebase|prefire|move42|rlc4` found no relevant
entry in that scope; no unverified memory supplied a task fact.

Own corpus queries: `counted.{0,40}(rider|geometry)|prefire_intent|receiver.delta`
over all `.omx/research/` (135 matched paths at intake);
`rule.?118|prefire|receiver.*timing` in docs;
`counted.rider|rule.?118|prefire|receiver.*timing` over the canonical research index
and all `sub015_DAG_*` files; and `rlc[1234]|prefire|counted.rider` against the
canonical task ledger, lane registry, and active-dispatch claim store. Initial
guessed task/claim filenames did not exist; the actual stores were located and
searched, with outputs retained. The complete canonical-equation export and
selected integer-determinism/frontier rows are retained.

Beyond the charter seeds, the index/DAG preserve the generic/free versus selected/
counted boundary, and the integer-determinism equation does not transfer a timing
or archive verdict. The task store still contains historical timing orders;
the current charter's suspension governs, so none was relaunched. The current
disqualification guard independently records why an uncounted receiver cannot
be promoted. The consequential source-level finding beyond the expected rebase
was the manifest's inclusion in normalized receiver identity; it required retaining
a real contract refusal instead of claiming the intent path had closed.

## Custody, source review, and MAIN landing

Freeze commit: `a475431997d0e0c66563448524e44c2ca8ddb384`.
Freeze receipt SHA: `59158b8fce89e12c06eb4ae3e1cb71f347daa78a2cc5e8a061e1899be1150c2a`.
pr12 memo SHA: `50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc`.
rlc2 STOP SHA: `76ba13cf2ef8f91d731a8526d65f3bd96d7d3799c7002e9a99ed56e77fa24caf`.
All charter, trace-script, producer and source pins are in `SOURCE_PINS.json` and
the actual production receipts; no hash is inferred from a filename.

The four new Python wrappers have two recorded source-review passes and passed
Ruff/syntax checks. The actual trace, triple encode, archive parse-back, full cold
public execution, complete byte comparison, four smokes, census, and real producer
invocations are the functional verification. The wrapper source is intentionally
unlanded until the charter's final serializer attempt; the receipts disclose this
and name the checked-out landed mechanism commit plus exact wrapper source hashes.

The final serializer is attempted once, after this complete deliverable. Its
actual rc, bundle/format-patch references, independently verified commit/file set,
and shared-index comparison are recorded in `FINAL_SERIALIZER_STATUS.json` and
`FINAL_HANDOFF.json`. Those post-attempt records necessarily sit outside the
commit they describe. MAIN must land the verified bundle before treating its
source/evidence as main-branch custody. No permission or prior landing is fabricated.

The six solver hooks are explicitly N/A here except the owned pre-fire consumer:
no new sensitivity map, Pareto model, bit-allocation change, empirical score
posterior, or optimizer ambiguity was measured. The deployment hook is the
blocked real intent chain in `RESUME_FIRE_ORDER.json`; the raw/code/manifest
comparisons disambiguate receiver identity from the manifest bookkeeping failure.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `.omx/research/ddm_rlc4_20260910/FINAL_HANDOFF.json`; fire trigger: Git-writable MAIN harvests the verified final serializer bundle. Land its exact source and evidence, preserving unrelated work.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / pr12-family; consumer store: `.omx/research/ddm_rlc4_20260910/RECEIVER_MANIFEST_CONFLICT.json`; fire trigger: harvest of the actual rc3 refusal and one-path normalized diff. Adjudicate the manifest/digest conflict, freeze any change prospectively, and explicitly decide how already-produced evidence may be reused without retimestamping it; include the bare-CLI import failure in readiness.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / newly chartered producer; consumer store: `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42`; fire trigger: prospective contract resolution and a resumed charter while the pointer remains move42. Resume the retained candidate chain under the resolved ordering, produce a committed valid intent and both normal-seal refusal controls; MAIN alone owns subsequent authorization, fire, harvest, and completion.

Frontier unchanged: **composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)**.

## LIVE-HYPOTHESES

- This retained 180,178-byte archive may clear the exact T4 score bar once its
  custody gate is resolved: both real encoders and the full macOS public raw proof
  support the 60-byte rate credit. T4 scoring remains untested.
- The cured receiver may finish below 1,260 seconds on T4: the completed source
  T4 leg and retained RLC1 diagnostics motivate the 1,032.725-second risk estimate.
  That is not candidate timing authority or a confidence bound.

## DEAD-ENDS

- A valid refreshed manifest cannot retain the required old receiver digest for
  this rebased tree: the actual normalized diff isolates its changed `inflate.py`
  hash row. Restoring a stale manifest would repeat pr9's failed condition.
- Calling the real first-fire producer an open door is closed for this instance:
  with correct module loading it returned the typed receiver-endpoint refusal.
- Copying the move40 rider as a move42 result is closed; move42 required fresh
  full encoding, which is now retained and has a different stream.
- Inherited timing for changed receiver code, additional local calibration
  windows, self-authorized T4 fire, and an uncounted-constant cure remain outside
  the approved path. No such workaround was attempted.
- Stopping at a first Git-object denial is superseded by this charter: production
  and the real refusal were completed before the single final serializer attempt.
