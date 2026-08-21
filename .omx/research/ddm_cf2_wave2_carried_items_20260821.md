# ddm_cf2 — the rv17 wave-2 carried items, both landed

`date_utc: 2026-08-21` · `owner: ddm_cf2` · `axis: [macOS-CPU advisory / scorer-free EXACT byte
measurement]` · `score_claim: false` · cost $0 · no Modal, no launches, no `upstream/` edits,
frozen gen6 packet untouched.

## ANSWER FIRST

Both wave-2 carried items are done, and neither moved the score.

**F2 — the equations leg is PAID.** Two laws registered through the locked canonical helper:
`token_rate_model_direction_dependence_v1` (4 anchors) and
`greedy_set_average_vs_marginal_price_v1` (2 anchors). Registry-wide empirical-anchor
round-trip audit: **0 violations**. `check_evidence_authority_claims_are_custodied` PASS
strict, including its `[catalog-351-anchor-roundtrip+canonical-producer-identity]` leg — the
gate the #1149 red belongs to.

**F1 — the typed receipt writer is built and ready for its trigger.** `src/comma_lab/packet_receipts.py`
validates a `DOC_DIVERGENCE_RECEIPT` record and re-parses its own serialized bytes **before**
the target path is opened, so a refusal leaves the directory byte-identical. The R15
trailing-comma class is unrepresentable: a note that is not a `str` cannot be constructed.

**One finding worth stating plainly, because it changes what F1's wording could honestly
claim:** the charter said *"wire the existing append path(s) through it."* **There is no
in-repo append path.** `grep -rn DOC_DIVERGENCE_RECEIPT` over the whole repo returns research
memos plus two POST-write verifiers (`verify_receipt_chain.py`, `verify_citations.py`) in the
packet prep directory — and no writer. That absence *is* wave-2 F1's finding
(*"no shared receipt writer"*), and R16's own `known_defect_in_predecessor` field says why it
matters: *"receipt writers are unreviewed one-shot scripts."* So this module is the FIRST
sanctioned append path, not a replacement for one. I have not claimed a rewiring that did not
happen.

**Pointer UNMOVED: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4 n600]`.** This unit paid
apparatus debt. It produced no exact row and is not goal progress.

---

## §1 F2 — the two registrations

Both are `verdict_scope: FORMULATION` — they govern **jg3-class edit-configuration re-selection
under real prices, on the rc2-lineage token field**. Neither is a family law. Neither is a score.

### (a) `token_rate_model_direction_dependence_v1`

> The `-log2 p` autoregressive rate model may be trusted in one direction of travel and not
> the other.

| direction | denominator | realized / modelled | anchors |
|---|---|---:|---|
| **AWAY** from the model argmax | the **ACTUAL flat price** 4.1379 b/tok | **0.9217 – 0.9274** | 2 |
| **TOWARD** the argmax, `u = 7.75` | the **MODELLED credit** | **0.0872** (recapture 91.3%) | 1 |
| **TOWARD** the argmax, `u = 12.0` | the **MODELLED credit** | **−0.0145** (costs 37 B) | 1 |

The away band is two objects on one lineage, not noise: jg3's own configuration prices at
**3.8137673394495413 b/tok** (5,196.258 B / 10,900 tokens, from jg4's retained per-frame
code-bit arrays) = an **8.50% overcharge**; jg5's shipped 455-edit set prices at
**3.837300670210307** (4,151 B / 8,654 tokens) = **7.84%**. Both were re-derived in code and
both agree with their receipt's own published field. Consumer rule, unchanged and the real
law: **price a token-field lever by a REAL re-encode, never by the model.**

**The 4.718 is not a price.** It is `ddm_jg1`/`ddm_jg3`'s `LogitPrice` **RANKER** — ordering
only, per its own docstring. jg5's published `realized_over_modelled = 0.8133320623591156` and
the mirror's `0.7721920938004125` are both *ranker-relative*. The module names that quantity
`ranker_relative_ratio()` so it cannot be read as a trust factor, and `domain_of_validity.excluded`
refuses the use outright. This makes the W2-F3/F4/F14 cure structural rather than editorial.

**The u=12 anchor carries the equation's largest residual (~1.44 b/tok) on purpose.** Predicting
it from the u=7.75 factor is badly wrong, and that recorded miss is the evidence that the
toward-direction factor does **not** transfer across thresholds — only its sign and order of
magnitude do. The excluded list says so in as many words.

### (b) `greedy_set_average_vs_marginal_price_v1`

The split wave-3 round-1's item-4 note asked for, honored:

* **FRAMING (arithmetic, true by definition):** the average of a set is not the value of its
  marginal member. Carries no empirical content on its own.
* **EMPIRICAL, FORMULATION-scoped:** the **degradation direction**. Because jg3's greedy orders
  candidate sites by gain, a denser configuration adds exactly the sites it already ranked
  worst — so price and yield degrade **together**.

| side of the admission cut | marginal | set average | ratio |
|---|---:|---:|---:|
| **ADD** (round 2's reopen, +300 tokens / +223 B) | **5.9467** b/tok | 2.6573 | **2.24×** |
| **REMOVE** (the mirror, −997 tokens / −664 B) | **5.3280** b/tok | **3.6432** | **1.46×** |

Yields move the other way in the same measurement: the 38 pairs' shipped edits yield
876/569 = **1.5395** cells/token, the marginal sites 116/300 = **0.3867** — **3.98× worse**. A
flat prior fitted to a set average is therefore simultaneously too dear for the set and too
cheap for its margin, and correcting only the first error admits exactly the configurations the
second forbids. That is why the stopping rule is vindicated in **both** directions.

**The 5.3280 travels with its warning, everywhere it goes.** It is a **TREND EDGE, not a
plateau**: the running credit fell monotonically **7.1862 → 6.1227 → 5.9451 → 5.5617 → 5.3280**
and came to rest **1.9%** above its kill line. The `domain_of_validity.excluded` list forbids
carrying it as a plateau value.

### Anchors, artifacts and shas

| anchor_id | source artifact | sha256 (first 16) |
|---|---|---|
| `cf2_away_jg5_shipped_455_20260820` | `/Volumes/APDataStore/pact/ddm_jg5/retained/final/S1_encode_jg5_subset455.json` | `0d22f394f12c7dac` |
| `cf2_away_jg3_configuration_20260820` | `.omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md` | `d817e545ccb0380a` |
| `cf2_toward_rung4_u7p75_20260820` | `.omx/research/ddm_fs2_rc4_drop_carrier_resolve_20260820.md` | `f864de4d6f2b3861` |
| `cf2_toward_rung4_u12_20260820` | `.omx/research/ddm_fs2_rc4_drop_carrier_resolve_20260820.md` | `f864de4d6f2b3861` |
| `cf2_marginal_add_round2_reopen_20260820` | `.omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md` | `d817e545ccb0380a` |
| `cf2_marginal_removal_mirror_drop137_20260821` | `/Volumes/APDataStore/pact/ddm_fs3/reencode/retained/S1_encode_fs3_drop137.json` | `01fcf55abc9f7c70` |

Referenced retained payloads (ALWAYS-KEEP-THE-PAYLOAD): the mirror candidate archive
`candidate_fs3_drop137.zip` 179,916 B sha `12a4ae4153123c32…`; its token stream
`tail_fs3_drop137.bin` 113,183 B sha `c8790d3124c9099a…`; the sha-receipted shipped-455
baseline stream 113,847 B sha `b9243abd2e38f9ae…`; both encodes' controls byte-identical
(`15054e5da33640bc…`).

### Round-trip verification (the #1149 red)

Three independent instruments, run in this order:

1. **In-memory**, per builder: `to_dict() → json → _equation_from_dict → to_dict()` compared as
   complete JSON objects — anchors AND full payload identical. (Test:
   `test_every_anchor_json_round_trips_exactly`.)
2. **Dry-run write** to a temp ledger through `populate_cf2_token_price_laws`, then the
   registry's own `audit_empirical_anchor_roundtrip_fidelity` → **0 violations**. (Test:
   `test_registration_and_canonical_roundtrip_audit`.)
3. **Real registry, read back through the loader**: `audit_empirical_anchor_roundtrip_fidelity()`
   over the whole live ledger → **0 violations registry-wide**, and `get_equation_by_id` returns
   `0.8133320623591156` / `0.7721920938004125` bit-identical to the receipts' own fields.

⚠ **These three are not fully independent** (wave-2 law 6: independence is a property of the
instruments, not the count). (1) and (2) share `_equation_from_dict`. (3) is the one that reads
bytes that actually went to disk, and it is the one that matters. I am reporting three steps,
not three confirmations.

### My own corrected outputs

**Review round 2 caught a sign inconsistency I shipped in round 1.** The removal anchor's
`inputs` carried the signed operation (`delta_tokens: -997`, `delta_bytes: -664`) while its
`predicted_output` and `empirical_output` carried magnitudes — a row that says two different
things about the same direction of travel, in a law whose entire content is that direction and
aggregation are part of the claim. Corrected: all four now carry the same sign (a removal
credits negative bits, `-3632.27` predicted against `-5312` realized). The **residual is
unchanged at 1679.7273083453047** — this was a presentation defect, not an arithmetic one. The
correction landed as a NEW `registered` event; the predecessor row is preserved per Catalog
#110/#113 APPEND-ONLY and is never edited.

Two further round-1 defects in the F1 module are recorded in §2 (the R11-F2 mis-scoping and the
omitted first-link legacy fields) — both surfaced by running the schema against the real chain
rather than by re-reading my own code, which is the honest lesson: the artifact found them, I
did not.

### Not marker-only (NO-FAKE #1/#2)

Every evaluator computes. `bits_per_token` is the single shared instrument every constant in
both laws was derived through — no decimal in the module is typed, each is computed from a
receipt's own `(bytes, tokens)` pair. The tests are behavioral, not constant-checking:
`predict_realized_bits_per_token` is asserted to **scale** with its input and to **differ by
direction**; `direction_trust_factor` **raises** on an unmeasured direction rather than
defaulting; `greedy_margin_degrades_both_terms` is asserted **false** in both falsifying
directions. Replacing any function body with `return CONSTANT` fails at least one test.

---

## §2 F1 — the typed receipt writer

`src/comma_lab/packet_receipts.py` (+ `tests/test_comma_lab_packet_receipts.py`, 28 tests).

### The defect it makes unrepresentable

R16's own `known_defect_in_predecessor` field, verbatim: R15's
`repo_only_docs["verify_citations.py"].note` was serialized as a **single-element JSON LIST**
instead of a string — a trailing-comma slip whose tuple-guard checked the enclosing dict rather
than the note value. Content intact, type wrong, append-only forbids editing it. The list is
truthy, iterates, and serializes, so every check that asked *"is it there?"* passed. The
question that catches it is *"is it a `str`?"* — and that is the one guard the module puts in
front of every note.

### Schema DERIVED from the 14 real receipts, not invented

Every field name, optionality and nested shape was read off `DOC_DIVERGENCE_RECEIPT.json` and
`_R4`..`_R16` — including three legacy fields that appear **only** in the chain's first link
(`authoritative_source`, `corrections_applied`, `repo_only_corrected_docs`, whose
`frozen_counterpart` is nullable and is preserved as `None` rather than dropped).

Two things the derivation forced me to change after the first draft, both of which I had wrong:

1. **I had modelled R11-F2 as a construction invariant.** It is not: R4–R8 legitimately predate
   the rule and must still PARSE. It is now a **write-time policy**
   (`check_publish_source_declared`, called from `serialize_receipt`) — refused going forward,
   accepted historically. `verify_receipt_chain.py` applies it to the LATEST receipt only, and
   this is that rule's pre-write mirror.
2. **I had omitted the first link's legacy fields**, which made the chain's oldest receipt fail
   my own schema. A schema derived from the chain must model the whole chain.

### The executed control

```
$ .venv/bin/python -m comma_lab.packet_receipts --check <gen6_receipts>
ok   DOC_DIVERGENCE_RECEIPT.json … ok DOC_DIVERGENCE_RECEIPT_R16.json   (13 ok)
FAIL DOC_DIVERGENCE_RECEIPT_R15.json: repo_only_docs.note: expected a string, got list (…).
     A one-element list here is the R15 trailing-comma slip -- drop the comma.
13/14 receipts parse against packet_doc_divergence_receipt_v1
```

**13/14, and the single refusal is R15 — the exact defect R16 recorded.** That is a stronger
control than 14/14 would have been: it shows the schema matches the real chain *and* that it
catches the historical defect at the source, with a message naming the cause.

### Both directions, and the ordering claim tested rather than asserted

* **Positive:** a valid record writes; the bytes re-parse to an identical payload; serializing
  the same record twice is byte-stable; optional sections are **omitted, not nulled**.
* **Negative:** every refusal test **snapshots the directory before and asserts it after**.
  *"It raised"* and *"it wrote nothing"* are different facts, and only the second is the cure —
  validation, serialization and re-parse all complete before the target path is opened.
* **Append-only:** refuses an existing path, and refuses any name whose rank does not strictly
  exceed the chain head.
* Also refused: non-`str` notes at all three entry types, bad sha shape (incl. uppercase), bad
  date form, empty author/reason, unknown top-level and nested fields, bad `publish_source`
  enum, a string where `review_lineage` expects a list, and a receipt tracking zero documents.

### What is NOT claimed

* No prior append path was rewired, because none existed (§ANSWER FIRST).
* `verify_receipt_chain.py` was **not** edited to consume this schema. It is
  receipt-**tracked** (`repo_only_docs`), so changing its sha would break the live chain check
  and OWE an R17 append — MAIN's act at the swap boundary, not mine. Sharing the derivation
  between writer and verifier (the R16-F1 pattern) is the natural next step and belongs in that
  receipt-bearing edit.
* Nothing here was run against the frozen packet in write mode. The only contact with the SSD
  custody tree was **reading** the 14 receipts.

---

## §3 TRIALITY

* **DAG** — `FEED-cf2` appended to `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`,
  recording the equations-leg debt as PAID and F1 as built-awaiting-trigger.
* **equations** — `token_rate_model_direction_dependence_v1` +
  `greedy_set_average_vs_marginal_price_v1` in
  `.omx/state/canonical_equations_registry.jsonl`, appended via `register_canonical_equation`
  under the registry lock. Producers `experiments.ddm_jg2_tail_reencode` +
  `experiments.ddm_fs3_jg5_real_price_reopen`; consumers the jg-family waterfill solvers
  (`ddm_jg3_joint_solve`, `ddm_fs3_jg3_repriced_rescreen`, `ddm_fs3_compose_reopen_candidate`,
  `ddm_fs2_drop_ladder`, `ddm_fs2_jg5_on_candidate`) and `tac.ddm_costate_organ`'s pricing rows.
* **DSL** — **N/A with reason.** Neither law is a trainer lever; both are pricing predicates the
  solvers consume directly. No `Lever` factory is owed and none was invented.

Wave-2's F2 row is discharged. Its scope note asked specifically for the **Series B /
price-based** form rather than the round-4 range, and that is what (a) carries: the away
direction is stated against the ACTUAL price with both measured objects named, and the ranker is
excluded from being a denominator at all.

---

## §4 STATUS

| item | state | evidence |
|---|---|---|
| **F2 (a)** `token_rate_model_direction_dependence_v1` | **REGISTERED**, 4 anchors | live ledger; loader read-back exact |
| **F2 (b)** `greedy_set_average_vs_marginal_price_v1` | **REGISTERED**, 2 anchors | live ledger; loader read-back exact |
| Anchor round-trip (the #1149 red) | **0 violations registry-wide** | `audit_empirical_anchor_roundtrip_fidelity()` |
| `check_evidence_authority_claims_are_custodied` | **PASS strict** | incl. catalog-351 anchor-roundtrip leg |
| Evaluators do real work | **verified** | 22 behavioral tests, scaling + raise + falsify |
| **F1** typed schema + writer | **BUILT**, awaiting MAIN's swap-boundary trigger | `src/comma_lab/packet_receipts.py` |
| F1 schema derived from real rows | **13/14 parse; R15 refused** | CLI `--check` over the gen6 chain |
| F1 refusal writes nothing | **verified** | before/after directory snapshots |
| F1 append-only guards | **verified** | existing-path + non-advancing-rank refusals |
| F1 wire existing append path | **N/A — none existed** | repo-wide grep; that absence is the finding |
| `verify_receipt_chain.py` shared derivation | **NOT DONE, deliberately** | receipt-tracked; owes an R17 append (MAIN) |
| ruff / ty | **clean on all four new files** | 19 `__init__.py` F401 = unchanged baseline |
| tests | **50 pass** (28 F1 + 22 F2) | plus 50 adjacent registry tests |
| Exact pointer | **UNMOVED** | 0.14827847122030852 @ 180,456 B `[contest-CUDA T4 n600]` |

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4 n600]`** — gen6 frozen, #1111
operator-HELD. UNMOVED by `ddm_cf2`, which paid apparatus debt and measured nothing new.
