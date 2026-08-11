# ddm_lp135 — lossless-pack reconciliation and closure

**Date:** 2026-08-10  
**Axis:** `[custodied archive bytes; scorer-free reconciliation]`  
**Verdict scope:** exact PR130, PR135/F26, CP135, LC2, and LC2 Route-B objects named below  
**Status:** COMPLETE — the charter's `>=90% ALREADY-BANKED` falsifier fired; both named rows had already been executed at optimal form by CP135

## Result first

No new encoder, scorer, or archive build ran. The required reconciliation found that CP135 already
inherits **4,328 / 4,328 B (100%)** of FD135's PR130→PR135 reduction, then adds another 472 B. More
importantly, CP135 had already executed both chartered successor rows on the exact F26 probability and
CAP1 objects, with every payload retained:

- same-state F26 ANS loses to RC64 by **6 B** on the control state and **9 B** on the HP3-step2 state;
- the complete FD135 CAP1 fixed-field hypothesis packs **22,223→22,183 raw B** and saves **79 B** in
  the complete control archive, with strict pack/unpack identity and deterministic repeat archives.

The prior-law prediction that at least 1,000 B of FD135 residue would survive into the composed base is
therefore **FALSIFIED on this exact instance**. It confused the 4,328-B PR130→PR135 ancestry delta with
unbanked successor headroom. There is no unmeasured LP135 byte credit to add to CP135.

## Reconciliation table

Every delta uses the explicit PR130 archive baseline: **191,052 B**, SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.

| Exact archive | Bytes | Delta vs explicit PR130 baseline | FD135 reduction already banked | Share of FD135 4,328 B | Residue to FD135 endpoint | Reading |
|---|---:|---:|---:|---:|---:|---|
| PR135/F26 | 186,724 | −4,328 B | 4,328 B by exact ancestry | 100.000% | 0 B | FD135's measured endpoint |
| CP135 composed | **186,252** | **−4,800 B** | **4,328 B by exact ancestry** | **100.000%** | **0 B** | PR135 plus a separately measured −472 B; no FD135 delta remains to transfer |
| LC2 | 187,226 | −3,826 B | 3,826 B of baseline-relative magnitude | 88.401% | 502 B by magnitude only | Different CPR1 semantic/carrier/HPAC wire; this is not inheritance of FD135 mechanisms |
| LC2 Route-B RC64 | **187,222** | **−3,830 B** | 3,830 B of baseline-relative magnitude | 88.494% | 498 B by magnitude only | Already-landed 4-B same-state LC2 RC64 win; still a different wire from F26 |

The LC2 percentages are a common-baseline magnitude reconciliation, not Shapley or mechanism credits.
LC2 keeps PR130's 23,054-B CPR1 carrier and has no CAP1 section; its ANS/temporal/split/CX2 stack is not
byte- or state-identical to FD135's IHS2/WANS1/CAP1/RC64/CBQ stack. Treating the 502/498-B magnitude
difference as transferable residue would repeat the cross-state comparison that CP135's exact race
closed.

## Row 3 — `f26_same_state_ans_race`

**Disposition: FOLDED / ALREADY-BANKED NEGATIVE.** Consumer evidence is the pre-existing CP135 store
`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/coders/`; the chartered duplicate consumer
path was not created because that would duplicate settled payloads and apparatus.

| Exact F26 probability state | RC64 token / archive | ANS token / archive | Complete-archive ANS delta | Equality proof |
|---|---:|---:|---:|---|
| control | 114,706 / 186,468 B | 114,712 / 186,474 B | **+6 B** | 117,964,800 symbols; event-order SHA `8eb51ab7…1366`; spatial SHA `c5c7671d…ece`; ANS terminal state empty |
| HP3-step2 | 115,231 / 186,252 B | 115,240 / 186,261 B | **+9 B** | same 117,964,800-symbol, event-order, and spatial hashes; ANS terminal state empty |

All four complete archives have byte-identical repeat archives. Principal archive SHAs are control RC64
`7b5feee4…2edd`, control ANS `8b7ff9b7…fcad`, HP3 RC64 `6eb1a3b7…edb6`, and HP3 ANS
`ac5d6a3a…e70f`. The negative is scoped to **INSTANCE(exact F26 control and HP3-step2 probability
states)**; it does not kill ANS on future probability states.

The cross-state observation that LC2's ANS payload was 178 B smaller than F26's RC64 payload is not a
same-state comparison and produced no transferable win. Conversely, on the exact LC2 state, Route-B
RC64 had already beaten ANS by 4 B and produced the retained 187,222-B archive.

## Row 4 — `cap1_metadata_pack`

**Disposition: FOLDED / ALREADY-BANKED POSITIVE.** Consumer evidence is the pre-existing CP135 packed
CAP1 store below `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/models/` and its complete
candidate archives.

The landed pack is the full FD135 hypothesis, not a subset:

- 12 q8 factors: one u8 base plus twelve 7-bit deltas;
- 12 signed biases: 6 bits each;
- 32 Huffman lengths: 4 bits each;
- 12 Rice `k` values: one u8 base plus twelve 1-bit deltas.

The exact F26 physical carrier/selector source is **22,223 B**, SHA-256
`e16dee51c22266d412cfb5807d4479b7704b9a169f6d39c0aea4a25f40a36f50`; the packed form is
**22,183 B**, SHA-256 `30c33886dcf40684a5895c48e292d11a9180380f9d1219c0c6de81754bbb3aab`.
The inverse restores the source byte-for-byte. On the explicit split-model control baseline, the complete
archive moves **186,547→186,468 B (−79 B)**. The selected HP3 archive carries the same inverse and is
186,252 B. Both control and HP3 CAP1 candidates have byte-identical repeat archives.

CAP1 cannot be applied to LC2 as the same row: LC2 exposes a distinct 23,054-B CPR1 carrier and receiver
grammar, not CAP1. A CPR1-specific representation would be a new mechanism with its own exact census,
pack/unpack proof, and complete-archive race; no CAP1 byte credit transfers to it.

## Two final lossless archives

No new win remained to compose. The required two-base closure is therefore the already-retained best
receiver-equal archive for each base:

| Base | Final archive | Delta from that base | Receiver proof | Repeat proof |
|---|---|---:|---|---|
| PR135/F26 composed | `186,252 B`, SHA `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6` | −472 B vs PR135 | CP135 restores semantic/carrier/residual bytes and all 117,964,800 tokens; later contest-CUDA row preserved PR135 distortions bit-for-bit | `archive.repeat.zip` has the same bytes and SHA |
| LC2 | `187,222 B`, SHA `b3365410a423fa6ae4d53e9a86fc2bd38bc59793ea2b437fc161bdcca11712b0` | −4 B vs LC2 | RC64P literal parse-back produced the canonical 3,662,409,600-B raw SHA `a18eb42a…353`; all token symbols and terminal state passed | `archive.repeat.zip` has the same bytes and SHA |

The CP135 row is already measured at **S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4,
n600]**. This LP135 unit is scorer-free and did not run or claim a new score. The LC2 Route-B value
`0.16959633225649604` is only a rate derivation from the measured LC2 anchor; no evaluator row on the
Route-B archive was run here.

## Payload custody

LP135 materialized no new payload. It re-hashed the retained archives in place and found the expected
byte counts and SHAs. No existing payload was moved, deleted, or copied. The authoritative retained
payload roots remain:

- `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/` — both F26 coders, decoded full-symbol streams,
  probability exports, CAP1 source/packed streams, every candidate archive, and repeats;
- `/Volumes/VertigoDataTier/pact/ddm_rc64p_20260810/route_b/` — LC2 RC64 payload, final archive, repeat,
  checkpoints, and parse-back receipts;
- `/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/` and
  `/Volumes/APDataStore/pact/ddm_lc2_20260810/` — LC2 source/repeat archives and canonical raw custody.

## RECALL EVIDENCE

Recall preceded adjudication and covered the full contract surface:

- read `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, and live
  `.omx/state/main_hot_state.md`;
- read FD135, RC64P, CP135 including ADDENDUM 3, LC2's final findings/receipt, and the exact machine
  receipts under the three SSD stores;
- ran `tools/corpus_query.py` over all seven stores (`research,equations,memory,dag,council,tasks,docs`)
  with queries for `RC64 cap1 F26 container CBQ lossless same-state ANS reconciliation`, `m46 baseline
  law already banked reconciliation baseline delta bytes`, `cp135 lc2 F26 CAP1 metadata pack exact
  probability export`, `RC64 F26 receiver equality 117964800 terminal state`, and `CAP1 predictor
  length Rice metadata bit packing`;
- listed the canonical equations and selected the exact rate law `lambda_rate=25/37,545,489`, the
  registered categorical range-coder row, and the same-axis gap-decomposition boundary;
- searched `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG, live task/hot-state surfaces, and the landed
  CP135 implementation/tests by content.

The beyond-seed finding that changed the plan was decisive: CP135 had not merely supplied tooling. It
had already executed both LP135 rows at full n600/complete-archive form, retained every payload, and
landed the exact evaluator confirmation. That changed the ordered work from duplicate encoding to the
charter's own ALREADY-BANKED close.

## Boundaries and dispositions

**MEASURED/re-verified:** exact archive bytes and SHA-256s; common-baseline deltas; complete-archive
ANS/RC64 sizes on two F26 states; retained full-symbol equality receipts; CAP1 source/packed sizes and
inverse equality; repeat-archive identity; exact wire incompatibility between CAP1/F26 and CPR1/LC2.

**NOT measured here:** any scorer component, evaluator row, runtime benchmark, new ANS/RC64 state, new
CAP1/CPR1 representation, or new archive. No Modal, scorer lane, public-PR mutation, or upstream edit was
used.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN roadmap owner; consumer store: `.omx/state/main_hot_state.md`
  plus the active JS1/rate-allocation plan; fire trigger: before MAIN next allocates rate work or launches
  a row that relies on the stale “fd135 residue >=1KB” premise; action: remove that premise and route any
  further rate work to a representation- or learned-state change, not these exact F26 coder/metadata rows.

## LIVE-HYPOTHESES

- A representation-changing current-base rate move can still beat CP135 because HP3's learned-state
  change produced a net complete-archive win even while its fresh token stream grew; exact same-state
  coder swaps are saturated, but the probability object and section geometry are not globally fixed.
- A CPR1-specific metadata representation may exist on LC2 because its structured 23,054-B carrier is a
  different object and CX2 already showed that reversible coordinate changes can pay there. CAP1's
  40-raw-byte result gives no transferable byte estimate; this requires a new field census before it is
  actionable.
- ANS may beat RC64 on a future probability state because the sign already differs across states/objects:
  RC64 wins by 4 B on LC2 and 6–9 B on the two tested F26 states, while neither recurrence dominates by
  theorem for every probability sequence. This is not a reason to retry either settled F26 state.

## DEAD-ENDS

- Exact F26 control-state ANS: closed at INSTANCE scope because it is 6 B larger than RC64 after full
  117,964,800-symbol equality and complete repeat-identical archive construction.
- Exact F26 HP3-step2 ANS: closed at INSTANCE scope because it is 9 B larger than RC64 under the same
  full-symbol and complete-archive proof.
- More of the named CAP1 fixed-field pack: closed at INSTANCE/FORMULATION scope because CP135 implements
  all four FD135 fields, restores 22,223 source bytes exactly, and already banks the full measured −79 B
  whole-archive effect.
- Applying CAP1 directly to LC2: closed as a representation mismatch because LC2 has a CPR1 carrier and
  no CAP1 section; importing the F26 byte credit would be a cross-wire fake.
- Treating FD135's 4,328 B as unbanked CP135 headroom: closed by exact ancestry; CP135 contains all 4,328
  B and another 472 B.
- Launching a duplicate coder or copying payloads into the charter's nominal consumer directories:
  closed because the landed CP135/RC64P stores already retain the full optimal-form payloads and proofs.
