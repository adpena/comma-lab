# ddm_rc1x — the rc64 pin was never stale: two roles, one name, two stalled arms

Date: 2026-08-19 · Arm: `ddm_rc1x` · Authority: exact byte measurement + compiled
round-trip · **Score claim: false** · **Pointer moved: false** · **Byte-closed: YES**
(on the rr4/D1 lineage; NOT on the ck2 body — §5)

`verdict_scope`: **INSTANCE** — the `ddm_pq2_compress_e2e` rebuild recipe and the
rr4/hv1 prepared base. The role distinction it names is a property of the coder and
generalises; the byte count does not.

## Conclusion first, including the correction to my own charter

**The `rc64_source_sha256` pin is CORRECT and its file has been on disk since
2026-08-10. I was dispatched to cure a "stale pin" and there is no stale pin.** The
blocker was that nothing named the pin's **role** or its **location**, so a search keyed
on the file name found the wrong body and concluded the pin was unclearable.

Two distinct C bodies wear the name `rc64_backend.c`:

| role | bytes | sha256 (16) | exports | where |
|---|---:|---|---|---|
| **ENCODER** | 12,222 | `5c75e2c70b89f148` | `rc64_encoder_*` **and** `rc64_decoder_*` | `<VertigoDataTier>/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/` |
| **SHIPPED receiver** | 5,638 | `05839d1416e68a49` | `rc64_decoder_*` **only** | every archive's `runtime/entropy/rc64_backend.c` (232 copies) |

The encoder role is what `ddm_rr2_encoder_byteclose` extends with a 2,603 B
checkpoint/resume block and compiles into the 14,825 B `rc64_backend_checkpoint.c`. The
shipped body exports **no encoder symbol at all**, so it can never satisfy that pin.

**`ddm_ma1` §7 reported: "I hashed every `rc64_backend.c` on both trees: 158 copies, 2
distinct contents (`05839d14…` ×157, `b249b77b…` ×1), and neither matches … Clearing this
pin is the first owed step." That claim is WITHDRAWN — measured.** The scan missed the
encoder body. Worse, the premise was already falsified by our own receipts:
**`ddm_fx2`'s byte-close chain USED that exact file successfully on 2026-08-17**
(`/Volumes/APDataStore/pact/ddm_fx2/byteclose_a/`, archive 180,450 B,
`archive_repeat_byte_identical: true`), two days before ma1 declared the pin unclearable.
`ddm_fx2`'s own driver hardcodes `TAC_PQ2_RC64_SOURCE=<VertigoDataTier>/…/rc64_backend.c`.
This is the **negative-existence claim** failure class: "I did not find it" was reported as
"it does not exist," and no one checked the sibling arm that had already found it.

**The measured byte-close.** With the roles named, ma1's law byte-closes on the first run:

| quantity | fx2 D1 (landed 08-17) | **ma1 (this arm)** | delta |
|---|---:|---:|---:|
| token stream | 109,801 B | **109,696 B** | **−105 B** |
| archive | 180,450 B | **180,345 B** | **−105 B** |

`code_bytes` measured **109,695.85533299884**; ma1's memo §1 predicted
**109,695.8553**. The prediction is reproduced to the last quoted digit, on the first
attempt, through the real coder rather than a sector model.

## 1. Controls — the instrument before any verdict

| control | expectation | measured | verdict |
|---|---|---|---|
| **P1** encoder source recoverable | subtracting the 2,603 B extension from the retained 14,825 B checkpoint yields the pin | sha `5c75e2c7…`, **byte-identical** | **pass** |
| **P2** independent on-disk copy | the VertigoDataTier body equals the recovered body | `cmp` **IDENTICAL**, 12,222 B | **pass** |
| **P3** shipped decoder decodes encoder output | 20,000 symbols, exact | status 0, **0 mismatches** | **pass** |
| **P4** positive control | encoder body's own decoder, identical path | status 0, **0 mismatches** | **pass** |
| **P5** negative control | one flipped payload bit must break it | first mismatch **10,066**, 3,492 wrong | **pass** |
| **A** shipped body as encoder source | REFUSE | refused, names both shas | **pass** |
| **B** encoder body as shipped member | REFUSE | refused, names both shas | **pass** |
| **C** both roles correct | PASS | `VERIFIED … 180,345` | **pass** |
| **D** determinism | second build byte-identical | `archive_repeat_byte_identical: true` | **pass** |
| **E** identity of the other sections | 7 non-token sections unchanged | `every_other_section_byte_identical: true` | **pass** |

P3 is the load-bearing one: it proves the two role bodies are **one coder**, so pinning
them separately is bookkeeping, not a fork. P5 proves P3 is not vacuous — the harness can
tell a good stream from a bad one. My first P3 attempt *failed* with 7,102 mismatches; the
positive control located the fault in my own harness (I fed the decoder the 4-byte
`TOKEN_MAGIC` frame the wrapper strips), not in the bodies. **Without P4 I would have
reported a false negative and "confirmed" that the roles were incompatible.**

## 2. The cure, and what it deliberately does not do

`experiments/ddm_pq2_compress_e2e.py` (commit `a6e07d42df`):

* the recipe comment now **names the role** of `rc64_source_sha256`, the location of the
  encoder body, and the receipt that the pin already worked;
* a new OPTIONAL key `rc64_shipped_member_sha256` (`RECEIVER_RECIPE_KEYS`) makes the
  shipped decoder a **verified input** so both halves are custodied by name;
* declared-but-blank refuses instead of silently skipping (`_is_sha256`) — a pin that
  checks nothing is the silent-instrument failure this edit exists to end;
* **no sha in `RR4_RECIPE` changed.** rr4's input set is byte-for-byte what it was; the
  test suite asserts that explicitly.

17 tests (`src/tac/micro_edit/tests/test_rc1x_rc64_role_recipe.py`). **Not vacuous:**
deleting the blank-pin guard fails 6 of them; the suite also asserts the structural fact
the arms missed (the shipped body exports no `rc64_encoder_*` symbol).

## 3. The byte-closed row

`/Volumes/APDataStore/pact/ddm_rc1x/byteclose_ma1/retained/archive.zip`

* archive **180,345 B**, sha `a0b2bdb1cd300177563b113ae7dec3db006d76bc869f3ce115e0dee05e7bc9d1`
* token stream **109,696 B**, sha `15054e5da33640bcb2e9d4589615c3b89b1312ce27fd9aa8e2a0ec0284b506f2`
* member 180,245 B, sha `0bf8c66282552c1d925b60017795f3761eeb2c32e00fc195aec74812c8c3de3e`
* corrector `ddm_ma1_within_miss_corrector`, sha `88cacf14a574f00d204133053c456d9c89b270c91d61b4579007826ae2c54625`
* `delta_S = −0.0016073835128369216` vs the 182,759 B prepared base;
  `bytes_saved_vs_frontier = 2,414`
* determinism repeat byte-identical; all 7 non-token sections byte-identical
* rebuilt and re-verified through the **cured two-role recipe**
  (`RESULT_rc1x_e2e_verify.json`), both rc64 roles reported `sha256_matches: true`

Against fx2's landed D1 row this is **−105 B / ΔS −6.99e-05**, which is **20.0×** the
−3.5e-6 admit bar.

## 4. Honest limits

* **This is NOT the ck2 body.** ck2 is a **container transform**: it borrows the sz1 token
  tail *verbatim* (109,897 B) and changes only the semantic/carrier serialization
  (`ddm_sa3_rebase_sz1.py --row ck2_plane2`). ma1 changes the tail. Composing them needs a
  tail-override that `ddm_sa3_rebase_sz1.py` does not expose — a named build step, not an
  inference. My charter's expected ≈176,420 B assumed the pin was the only blocker; it was
  not. **The −105 B is measured on the rr4/D1 lineage and projected, not measured, on ck2.**
* **No advisory was fired and no seal was drafted.** Both were charter steps and both are
  owed. The advisory is cheap; the seal is blocked — see §5.
* **Parse-back is still running** at the time of writing; the decoded-field falsifier
  (target `9ba2e52b…`) is therefore **PENDING**, exactly as the build receipt says. No
  decode-identity claim may be made until it lands.
* **Selection is inherited.** ma1 chose its cell on the scored clip; I re-ran its law, I
  did not re-open that choice.

## 5. A blocker MAIN owns: the frontier pointer cannot supply an admission bar

`.omx/state/canonical_frontier_pointer.json` carries
`our_local_frontier_contest_cuda.extra.archive_bytes = null` while
`archive_sha256 = 0aa1cada…`. `tac.candidate_seal.read_frontier_archive_identity` refuses
on it, and `src/tac/tests/test_candidate_seal_pin_consistency.py::test_live_pointer_supplies_a_usable_bar`
**fails on HEAD, independent of this arm** (verified by stashing my diff). Any
`make_candidate_seal.py` run against the live pointer will refuse. The ck2 refresh wrote
the sha without the byte count; the fix is MAIN's because the pointer is the frontier SoT
and its tolerance-0 rule exists to refuse exactly this kind of silent edit.

## 6. Custody

`/Volumes/APDataStore/pact/ddm_rc1x/`

| path | sha256 (16) | contents |
|---|---|---|
| `retained/rc64_backend_encoder_role.c` | `5c75e2c70b89f148` | the encoder body, under its role name |
| `retained/rc64_backend_shipped_receiver_role.c` | `05839d1416e68a49` | the shipped decoder body |
| `retained/rc64_role_identity_proof.py` | `7a410d47ef97aefc` | P1–P5, re-runnable |
| `RC64_ROLE_IDENTITY_PROOF.json` | `b95887326ebf5093` | the measured proof |
| `recipe/ma1_within_miss_over_rr4_d1.recipe.json` | — | the two-role recipe |
| `ma1_byteclose_driver.sh` | — | the launch, with the role note inline |
| `byteclose_ma1/retained/archive.zip` | `a0b2bdb1cd300177` | **the byte-closed candidate** |
| `byteclose_ma1/retained/token_stream.bin` | `15054e5da33640bc` | the re-encoded tail |
| `RESULT_rc1x_e2e_verify.json` | — | verify through the cured recipe |

## 7. NEXT_IF_RESUMED, ranked

1. **Land the parse-back verdict** (running) — decoded field must equal `9ba2e52b…`.
2. **Fire the local CPU advisory** on `a0b2bdb1…`. ma1's law is a rate-only re-encode of
   the same tokens, so decoded state must be **bit-identical** and the advisory must
   reproduce fx2's distortion exactly; anything else refuses the row.
3. **Fix the pointer's missing `archive_bytes`** (MAIN) — it blocks every seal.
4. **Add a tail-override to `ddm_sa3_rebase_sz1.py`** so ma1 composes onto ck2. That is the
   real remaining work for the −105 B on the shipping body.
5. **Sweep for the sibling class**: any other recipe/driver pin whose file is named
   generically and located off-tree. The genus is *name-without-role*, and it cost two arms
   a byte-close each.

## STORES CONSULTED

`.omx/research/ddm_ma1_model_axis_miss_cost_20260819.md` §1/§6/§7/§8 (the −104.584 B law,
the composition table, the §7 blocker this arm falsifies) ·
`.omx/research/ddm_fx2_t4_sealed_fire_order_20260818.json` (repeats the same blocker claim)
· `/Volumes/APDataStore/pact/ddm_fx2/byteclose_a_driver.sh` + `RESULT_build.json`
(**read at source** — the receipt that falsifies both) ·
`/Volumes/APDataStore/pact/ddm_pq2/e2e_smoke/RESULT_pq2_e2e.json` (the 12,222 B input row) ·
`experiments/ddm_rr2_encoder_byteclose.py:143-149,243-249` (**read at source**: the
hardcoded `RC64_SOURCE_SHA` and the concatenate-then-compile) ·
`experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py:130-148,233-244,286-320`
(`quantize_probabilities`, `TOKEN_MAGIC` framing — the fault my P4 control located) ·
`/Volumes/APDataStore/pact/ddm_ck2/{seal/CANDIDATE_SEAL_ck2_r1.json,CK2_CUSTODY_MANIFEST.json}`
· `/Volumes/APDataStore/pact/ddm_ck1/{GENERATION_RECEIPT.json,compile/SA3_REBASE.json}` ·
`experiments/ddm_sa3_rebase_sz1.py` (the container path; `sz1["tail"]` verbatim) ·
`.omx/state/canonical_frontier_pointer.json` (§5) · memories
`[[the_denominator_and_the_falsifier_can_both_be_vacuous_20260816]]` (§2 blank-pin guard) ·
`[[measured_object_vs_named_object_20260816]]` (**the genus of this whole arm**) ·
`[[negative-existence-claims]]` (§Conclusion) ·
`[[hand_assembled_dispatch_is_the_error_factory_20260817]]` (the launch guard refused my
hand-rolled nohup and was right).
