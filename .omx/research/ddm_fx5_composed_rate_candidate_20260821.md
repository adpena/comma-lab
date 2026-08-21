# ddm_fx5 — the fx2 19-member rebuild, built and byte-closed on the rc2 body

- **arm** `ddm_fx5` · **date** 2026-08-21 · **fire authority** MAIN only. This arm dispatched
  nothing and spent nothing.
- **axis** every byte number below is `[macOS-CPU advisory / scorer-free EXACT byte
  measurement]` through the real encoder and the real receiver. **`score_claim: false`** ·
  **`promotion_eligible: false`** — no scorer ran here and no row was fired.
- **base** `rc2_composed`, the SIXTEENTH move: **S 0.14827847122030852 @ 180,456 B
  `[contest-CUDA T4, n600]`**, archive
  `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`.
- **Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` —
  UNMOVED by this arm.** Only MAIN's fire can move it.

---

## ANSWER FIRST

**The fx2 19-member build is BUILT, byte-closed and sealed on the rc2 body. It is worth
−70 B — MEASURED, not projected — which is ΔS −4.6610126718551995e-05, 13.3× the −3.5e-6
admission bar, at ZERO counted bytes.**

| quantity | rc2 base | fx5 candidate | delta |
|---|---:|---:|---:|
| token stream | 113,847 B | 113,777 B | **−70 B** |
| archive | 180,456 B | **180,386 B** | **−70 B** |
| d_seg | 0.00020139 | 0.00020139 | **0 by construction** |
| d_pose | 6.37e-06 | 6.37e-06 | **0 by construction** |
| S | 0.14827847122030852 | **0.14823186109358996** | **−4.661013e-05** |

The candidate S is RECOMPOSED from components and reproduces `base + ΔS_rate` to the last
digit, so the arithmetic has its own control.

**The −70 B is 80.9% of the −86.58 B na11's R2 quoted, and the gap is real rather than a
measurement error. I did not get the number I pre-registered and I say so first.** §3 is why.

**Only ONE of the two chartered legs is in the candidate.** The dx1 −18 B CABAC re-code is
NOT built. Its premise is now VERIFIED on rc2 for the first time (§4) — that is the part
worth inheriting — but shipping it needs a CAP1 container flag and a receiver CABAC decoder,
which is a receiver-format change I could not build AND verify inside this arm. A smaller
honest candidate beats a composed broken one, so the seal carries the fx2 leg alone.

---

## §1 WHAT WAS BUILT

`ddm_fx2` raced the probability-model axis and measured its best architecture, **E1** — 19
members on the `cls_boundary_agree_homog_ubin8` mixer context — at **−797.42 B** against the
live rr4 law. It shipped the 13-member **D1** build at −710.84 B instead, and named its own
door in the memo: *"it becomes the pick the moment somebody measures real T4 headroom instead
of projecting it."* `ddm_rc2` measured it: **498.476 s charged ≤ 822 s ceiling, 323.5 s of
slack** on the shipping object.

E1 and D1 differ in exactly ONE dimension — the member set. Mixer context, count buckets,
learning rate, the SSE switch and ma1's within-miss sector are identical. So the build is a
PATCH over the rc2 receiver, not a new receiver:

* `runtime/fx2_model_axis_corrector.py` — `SHIPPED_CONFIG["families"]` 13 → 19.
* `runtime/native_free_corrector.py` — the guard's `EXPECTED_SHIPPED_CONFIG`, likewise.
* `runtime/f26_corrector_native.c` — `N_FAMILIES` 13 → 19, four new rule cases, four
  extended tables. **+54 / −8 lines.** Members 17 and 18 reuse rules the C already had, so
  only four rules were transcribed.
* `inflate.py` — the archive pin, repinned from disk after the splice (§5).

Builder: `experiments/ddm_fx5_build_e1_runtime.py` (committed `5995f7daec`, extended
`a59f006c5a`). It pins all three source files by sha256 BEFORE patching and requires every
anchor to appear **exactly once** — a patch that matched zero times would produce a tree that
looks built and behaves like the base, which is the inert-flag failure class. Re-running it
reproduces all three patched files byte-identically.

**Rule 118.** Every new member is generic receiver code reading only already-decoded symbols.
Nothing is transmitted, learned-and-shipped, or video-derived. The archive carries **zero**
extra bytes for the six new members: the whole −70 B is token stream at +0 B of counted
payload. This is the far-favourable side of the placement law
([[the_counted_byte_is_not_fungible_placement_beats_amount_20260816]] / na11's L6).

---

## §2 THE CONTROLS — three of them, and one is a negative control

No number above is admissible unless these hold. All three are mine, re-run in my own hands.

**C1 — the encoder inverts the shipping decoder on THIS body.** `ddm_jg2_tail_reencode
--stage control` re-encoded the UNEDITED rc2 token field under the shipped 13-member law:

| | value |
|---|---|
| shipped token stream | 113,847 B, sha `b9243abd2e38f9ae27e318a2fc608a0b9275ae0bc24c4a2499bbd9fcdbd7eb40` |
| re-encoded | 113,847 B, sha `b9243abd…` — **identical** |
| `byte_identical` | **true**, `prefix_bytes_matching` 113,847 |

This also proves, for free, that the **Python** 13-member corrector reproduces what the **C**
13-member corrector decoded on the fired T4 row, over the whole 600-frame field.

**C2 — Python == C for the NEW 19-member config.** `experiments/ddm_fx5_parity_e1.py` drives
both correctors in lockstep over the receiver's own 190-group wavefront, comparing
`coding_row` **byte-exact** (never `allclose`: one float32 ULP moves an RC64 frequency by up
to 128 counts) and every family/mixer/miss/run table at each frame boundary.
**Verdict `IDENTICAL`** — 380 groups, 393,216 rows, all 19 members.

**C3 — the negative control: the detector is NOT vacuous.** I injected one plausible
transcription slip into the C (`RULE_HOMOG_SPATIAL4`, factor order swapped), rebuilt, and
re-ran C2. It **FAILED at frame 0**, naming `counts/15` — which is exactly the family I
sabotaged. A parity check that has never been shown to fail is not evidence ([[m50]]).

---

## §3 WHY −70 B AND NOT −86.58 B — the honest accounting

na11's R2 row quoted **−86.58 B**, correctly attributed: it is fx2's own measured
`E1 (−797.42) − D1 (−710.84)`. I pre-registered it and measured **−70 B**, a realized/projected
ratio of **0.809**.

**Neither number is wrong. They were measured on DIFFERENT BASES, and the base differs in two
ways at once — which is exactly why I cannot tell you the mechanism.**

| | fx2's base | rc2 (this base) |
|---|---|---|
| corrector | fx2 D1 hit-event model | `Ma1WithinMissCorrector` = D1 **plus** ma1's within-miss sector |
| token field / stream | the pre-jg5 field, 110,512 B | jg5's 455-edit waterfilled field, **113,847 B** |

Two candidate causes, and **I did not separate them**:

1. **Sector overlap.** fx2's §6 decomposition is `hit-event bits + within-miss bits = total`.
   The six new members buy in the hit-event sector; ma1's law works the miss sector. If they
   share any of the same code length, the second one in returns less.
2. **A different field.** jg5's waterfill changed the tokens the model is predicting. A member
   set raced against one field has no guarantee of the same return against another.

Isolating these needs an E1-vs-D1 A/B on a body with ma1 but WITHOUT jg5's edits — a run I did
not make. **So: the 80.9% is MEASURED and the reason for it is UNMEASURED, and I will not
assert one.** The direction is at least unsurprising — fx2 measured its own two axes composing
at 79%, not 100% — but a plausible story is not a measurement.

**What transfers as a law is the smaller, harder claim: a delta measured on one base does not
transfer to another without measurement**
([[a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803]]). na11 quoted
−86.58 B correctly and attributed it correctly; the arm that spends it still has to re-measure
it, and this arm is the receipt for why.

**It does not change the verdict.** −70 B clears the bar 13.3×.

**One thing this arm did NOT do:** it did not re-race the member ladder on the rc2 body. fx2's
own average-vs-marginal caveat applies — member 1 returned 340.82 B for +13.1 s, members 2–11
returned 21.9 B each for 11.4 s each, a 13.5× collapse — so **do not extrapolate a 20th
member** from this row.

---

## §4 THE dx1 LEG — premise VERIFIED on rc2, and NOT BUILT

The charter asked for the dx1 −18 B CABAC re-code composed into the same candidate. I checked
its premise first, and the check was worth running because **the obvious reading is that rc2
destroyed it.** `pq2`'s own NOT_EXPRESSIBLE registry says rc2 carries *"the RR5 lossless
re-encode of the carrier body under an adaptive arithmetic basis"* — and dx1's −18 B is a
carrier re-code. Two carrier re-codes on one body is a collision.

**It is not a collision, and the reason is in rr5's own docstring:** rr5 re-codes the 27,648
five-bit **BASIS** symbols and states *"the coefficient stream must be left alone (Rice wins
by 415 B)"*. dx1 targets the Rice-coded **COEFFICIENT** stream. Disjoint regions.

**Measured, apples-to-apples, both bodies through the same parser:**

| | rc2 | jg5 (dx1's base) |
|---|---|---|
| CAP1 carrier blob | 22,296 B | 22,296 B |
| Rice payload | **9,829 B / 78,628 bits** | 9,829 B / 78,628 bits |
| Rice `ks` | `[9,9,9,8,8,9,9,9,9,9,9,9]` | identical |
| base code lattice sha256 | `1a5b7a46930f653b…` | `1a5b7a46930f653b…` — **identical** |

**dx1's −18 B transfers to rc2 exactly.** Its object is byte-identical on both bodies.

**Why it is not in the candidate.** The winner is *"adaptive-ctx Rice (CABAC prefix, cap=8)"*
— 9,811 B, side table 0 B, `decode_identity: true`. Shipping it needs (a) a CABAC decoder in
the receiver, (b) a CAP1 container signal so the receiver knows which coder wrote the
coefficient stream, and (c) that signal composed with rr5's existing reserved flag `0x08` on
the same carrier. That is a receiver-FORMAT change, and each verification cycle for it is a
full n600 decode. I could have built it; I could not have built **and verified** it inside
this arm, and an unverified format change spliced into a sealed candidate is exactly the
composed-broken-candidate the charter told me to refuse.

**Handed forward, priced and unblocked:** −18 B = ΔS −1.1985e-05, 3.4× the bar, $0 of compute,
premise verified above, payload retained at
`/Volumes/VertigoDataTier/pact/ddm_dx1/retained/dx1_payload_adaptive-ctx_Rice_CABAC_prefix_cap8.bin`
(sha `b93131a52674abb4…`). dx1's own recommendation was *"fold the −18 B into the receiver
revision when the tree is being rebuilt for reasons that already justify a seal chain"* — that
condition is now met and the named blocker is the CAP1 flag, not the coder.

---

## §5 WHAT THE BUILD SURFACED THAT NOBODY HAD WRITTEN DOWN

1. **`pq2_compress_e2e.py` REFUSES the rc2 body by name** (`NOT_EXPRESSIBLE`), and it is right
   to: it cannot REPRODUCE rc2's semantic/carrier sections from upstream inputs. But that is
   not the operation a model-only candidate needs. Taking rc2's own sections **verbatim** and
   re-encoding only the token stream is expressible, and `ddm_jg2_tail_reencode` already does
   exactly that with a splice-into-the-pointer-body guard. **A refusal to REBUILD a body is not
   a refusal to DERIVE from it.**
2. **`inflate.py` pins its own archive sha and the pin FIRED** on the first candidate decode
   (`archive.zip does not match the promoted F26 artifact`). That guard is correct and I did
   not bypass it — the builder now repins from disk, with no flag to pass a digest by hand, so
   a repin cannot be pointed at bytes nobody has.
3. **`inflate.sh` calls bare `python`, which this host does not have** (only `python3`). A
   host-environment matter, not a candidate defect — the contest runner provides `python` —
   but it cost one launch cycle and is worth knowing before the next local decode.

---

## §6 DECODE IDENTITY — the definitive Python-vs-C gate

The encoder ran the **Python** 19-member corrector; the receiver runs the **C** one. So a
byte-identical decoded token field proves Python == C over 600 frames of real accumulated
state — strictly stronger than §2's C2, which cannot reach a table state only 600 frames of
accumulation produces.

**The proof only works if the decoder really ran the C.** If it had silently fallen back to
the Python corrector, an identical field would prove nothing about the C — encoder and decoder
would both be Python. So the binding is verified DIRECTLY, not inferred from a silent stderr:
the live decode process carries
`F26_CORRECTOR_NATIVE_LIBRARY=/var/folders/…/tmp.9xuizc1lpL/f26_corrector_native.so`, the
library `inflate.sh` compiled from **this tree's** 19-member `.c`, and
`native_free_corrector` refuses to bind at all unless the live `SHIPPED_CONFIG` matches the
19 families compiled into it.

| anchor | required | MEASURED | verdict |
|---|---|---|---|
| decoded token field sha256 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | **IDENTICAL** |
| corrector actually bound | native, 19 members | `F26_CORRECTOR_NATIVE_LIBRARY` exported, guard passed | **NATIVE** |
| coded token stream sha256 | must DIFFER (it is the −70 B) | `b9243abd…` → `e2af55e641c4f2d3…` | **differs, as required** |
| `hpac_blob` sha256 | unchanged | identical to rc2 | **carried verbatim** |
| `residual_payload` sha256 | unchanged | identical to rc2 | **carried verbatim** |
| `0.raw` sha256 / bytes | `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` / 3,662,409,600 | see receipt | render stage |

**Different bytes in, identical tokens out, everything else carried verbatim — which is the
definition of a lossless rate win.** The token anchor is the same digest rc2's contest-CUDA T4
receipt records as `decoded_token_sha256`, so local CPU and contest CUDA already agree on it
for the base, which is what makes it usable as a candidate anchor here.

**Consequence for the score: `d_seg` and `d_pose` cannot move.** The render is a deterministic
function of the token field; the token field is byte-identical; so the two distortion legs are
unchanged and the entire ΔS is the rate term. That is why falsifier 1 below is a REFUSAL and
not a regression check.

**State at seal time, stated exactly.** The token-field row is **MEASURED and PASSED** — that
is the decisive gate, and it is the same discriminator rc2's own memo calls *"stronger than
the component-level form"*. The `0.raw` row is **still landing**: the render reached the full
3,662,409,600 B, then the host stalled in uninterruptible I/O flushing 3.4 GB to an SSD at 99%
capacity (31 GiB free). That is a storage condition on this machine, not a property of the
candidate, and it cannot change the raw bytes — the render is a deterministic function of a
token field already proven identical. A detached writer
(`ddm_fx5_decode_receipt`) is bound to the completion artifact and lands the raw sha into the
receipt when the flush completes.

**I sealed on the token gate and say so rather than implying both rows were closed.** If the
raw sha comes back anything other than
`7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`, falsifier 3 fires and the
candidate is refused before any fire.

**Receipt: `/Volumes/APDataStore/pact/ddm_fx5/retained/FX5_DECODE_IDENTITY.json`** (verdict
field: `DECODE_IDENTICAL` when both rows close, `TOKEN_IDENTICAL_RAW_PENDING` in between,
`REFUSED` if the token row ever fails).

---

## §7 THE SEAL AND THE FIRE ORDER

**Status: SEALED · `SEAL_VALID` · READY_TO_FIRE.**

| | |
|---|---|
| seal | `/Volumes/APDataStore/pact/ddm_fx5/CANDIDATE_SEAL_fx5_e1_19member.json` |
| seal sha256 | `3dcef9b3c986b6cf747d0862ce7dfe6fbbe8414216bb48a6ecb45783ddb8fb5f` |
| candidate id | `ddm_fx5_e1_19member_model_axis` `[contest_cuda]` |
| archive | **180,386 B**, sha `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841` |
| runtime tree | 38 files, 672,500 B, digest `f1dcb517a332886b…` |
| receivers pinned | 8, per-file: `inflate.py` `inflate.sh` `runtime/residual_archive.py` `runtime/f26_corrector_native.c` `runtime/native_free_corrector.py` `runtime/fx2_model_axis_corrector.py` `runtime/free_corrector.py` `runtime/rr5_arith_basis.py` |
| admit bar | net ΔS < −3.5e-06 vs `contest_cuda` 0.14827847, tolerance 0 |

Written by `tools/make_candidate_seal.py` — every sha MEASURED from disk, none hand-typed;
the archive sha was passed as `--verify-archive-sha` (checked against the bytes, never stored
as the value) and the report-8dp bound is COMPUTED from rc2's own T4 receipt.

**MAIN fires with:**

```
.venv/bin/python tools/fire_modal_auth_eval.py \
  --seal /Volumes/APDataStore/pact/ddm_fx5/CANDIDATE_SEAL_fx5_e1_19member.json \
  --output-dir /Volumes/APDataStore/pact/ddm_fx5/t4_row_r1 \
  --lane-id lane_ddm_fx5_e1_19member_cuda_20260821 \
  --instance-job-id ddm_fx5_e1_19member_t4_r1 \
  --axis cuda
```

**Pre-registered falsifiers** (each refuses the candidate):

1. **Any** `d_seg` or `d_pose` delta ≠ 0 at report precision. The token FIELD is unchanged —
   `tokens_changed: 0` — so only the probability law moved; a distortion delta means the
   decode desynchronised and the row is void, not merely worse.
2. Decoded token sha256 ≠ `cc10a7b0…` — same class, caught earlier.
3. `token_decoder.free_corrector` ≠ `NativeFreeCorrector` — the Python fallback costs the
   decode wall, not identity, but it invalidates the wall projection below.
4. **Charged decode wall > 822 s.** rc2 measured 498.476 s charged. fx2's serial timing
   projected E1 at **+89 s** over D1, landing at ~587.5 s — **234.5 s under the ceiling**.
   That +89 s is a Python-numpy marginal carried across to a **native** corrector, so it is a
   CEILING rather than an estimate; the true native marginal should be smaller, and the six
   added members are 6/13 more table updates against a corrector that is already only part of
   the stage.

   **This is the one number in the candidate that is PROJECTED rather than measured, it is the
   binding constraint, and the local decode CANNOT settle it.** rc2's own two local n600
   decodes of the SAME bytes measured **1,250.9 s and 1,681.8 s** — a **1.34× spread** on an
   unchanged object. Local run-to-run variance is larger than the effect being tested, so no
   local wall I report can bound the T4 marginal. I record my local wall in the decode receipt
   as custody, not as evidence. **The T4 row is the only place this constraint is settled, and
   MAIN should read the harness's own `Wall budget` verdict on that row before treating the
   candidate as shippable.**
5. Archive bytes ≠ 180,386, or archive sha ≠ `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`.
6. The seal's own COMPUTED report-8dp bound against the rc2 base receipt.

Admission bar: net ΔS < −3.5e-6. Expected: **−4.661013e-05**, 13.3× over.

---

## STORES CONSULTED

`.omx/research/ddm_fx2_model_axis_all_sections_20260818.md` §5 §8 §9 (the E1/D1 race table, the
decode-margin refusal, the selection caveats) · `/Volumes/APDataStore/pact/ddm_fx2/race/E1_compose_19x_homogctx.json`
(E1's exact 19-member list, read at source rather than retyped) ·
`.omx/research/ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md` §1–§3 §6 §8 +
`/Volumes/VertigoDataTier/pact/ddm_dx1/retained/DX1_RECODE_RACE.json` (the −18 B, its control,
its retained payloads) · `.omx/research/ddm_na11_negative_regrade_vs_rc2_20260821.md` §3 R2 (the
reopen, the exchange rates, the 323.5 s slack) · `.omx/research/ddm_rc2_t4_row_sixteenth_move_20260820.md`
(the base row, the measured wall, the rv17 F1 correction) ·
`/Volumes/APDataStore/pact/ddm_rc2/CANDIDATE_SEAL_rc2_composed.json` (the receiver pins this
build inherits) · `experiments/ddm_jg2_tail_reencode.py` (the encoder, read at source: its
control-is-the-proof contract is what makes §2 C1 admissible) ·
`experiments/ddm_rr8_corrector_parity.py` (the lockstep protocol §2 C2 mirrors) ·
`experiments/ddm_pq2_compress_e2e.py` `NOT_EXPRESSIBLE` (the refusal §5.1 reads correctly) ·
`runtime/rr5_arith_basis.py` (verified at source: rr5 codes the BASIS, not the coefficients —
the fact that saves the dx1 premise).

## RETENTION (P0)

`/Volumes/APDataStore/pact/ddm_fx5/RETENTION_MANIFEST.json` — **20 payloads, 21,268,363 B,
each with sha256 + bytes.** Both re-encoded token streams are kept, not only the winner's:
`tail_e1_19member.bin` (113,777 B — the −70 B itself, not its length) and
`tail_control_600.bin` (113,847 B — the byte-identity proof). Plus the candidate archive, both
encoder + corrector checkpoints (resumable), both per-frame bit ledgers, the built `.so`, the
three patched receiver sources, and every verdict JSON. **Nothing was measured and discarded.**

The 3,662,409,600 B raw render and the 117,964,800 B decoded token field under
`decode_r1/inflated/` are certified REBUILDABLE — the command, the runtime-tree digest and the
archive sha are all in the launch manifest and `FX5_DECODE_IDENTITY.json`, so they regenerate
deterministically. They may be cold-stored or deleted **after** that receipt lands, not before.
APDataStore was at 31 GiB free at seal time; Vertigo is full and was used read-only.

---

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` — UNMOVED
by ddm_fx5.** This arm produced a sealed candidate and claims no row.
