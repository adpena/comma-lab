# ddm_rr2 — the rr1 free corrector is byte-closed: 181,161 B, on target to the byte

**Date:** 2026-08-17
**Base:** hv1 ep0634, `S = 0.15959729295498598` @ 182,759 B `[contest-CUDA T4, n600]`,
archive sha256 `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.
**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]`. `score_claim: false`,
`promotable: false`. No Modal, no dispatch, no exact eval. **Borrowed pointer UNMOVED.**
**Store:** `/Volumes/APDataStore/pact/ddm_rr2_encoder_build/`.

---

## ANSWER

**The build landed on the pre-registered target exactly: `archive.zip` = 181,161 B against a
target of 181,161 ± 2 B, token stream = 110,512 B against a target of 110,512 B — 0 B deviation
on both.** Saving vs the frontier archive: **1,598 B**, `ΔS = −1.0640426e-3`. `ddm_rr1`'s
projection was not merely close; it was exact, because the two unknowns it carried — the repack
layer and the coder tax on corrected distributions — both came out inside a byte.

**The encoder is proven to be the exact inverse of the shipping decoder, not argued to be.** A
bypass control encoded the already-decoded field through the receiver's own probability rows and
reproduced the shipped 112,110 B token stream **byte-identically** (sha `73a87889…`), at a code
length of **112,109.57757858852 B** — reproducing to the last digit the number `dc1`, `hm1` and
`rc4` measured independently. Without that control the corrected stream would be a plausible
artifact; with it, the only thing that changed between the two runs is the probability model.

**Every other section is byte-identical, checked by re-parsing the built archive with the
receiver's own parser rather than by comparing the builder's own slices to themselves.** Seven
of eight parsed sections match the frontier exactly; only `token_stream` differs. A separate
control rebuilt the frontier archive from its own parsed sections and reproduced sha
`80d9c8c6…` byte-identically, so the repack layer contributes zero bytes of its own.

**What this is NOT.** 1,598 B is **11.09% of the 14,414 B bar**. This does not reach sub-0.15 and
does not move the pointer. It is one measured, composable, zero-distortion rate row, delivered
byte-closed so the next arm can stack on it instead of re-deriving it.

---

## 1. The pre-registered target, and what it returned

`ddm_rr1` §4 fixed the target and the falsifier before any build existed. Both are reproduced
verbatim from that memo:

```
TARGET:        archive.zip == 181,161 B  (+/- 2 B)
               token stream == 110,512 B
               every other section byte-identical to 80d9c8c6...
FALSIFIED IF:  the built archive exceeds 181,250 B, or the decoded token field is not
               bit-identical to sha 9ba2e52b...
```

| row | target | measured | verdict |
|---|---:|---:|---|
| `archive.zip` bytes | 181,161 ± 2 | **181,161** | **HIT, 0 B deviation** |
| token stream bytes | 110,512 | **110,512** | **HIT, 0 B deviation** |
| every other section byte-identical | yes | **7 of 7 non-token sections identical** | **HIT** |
| falsifier: archive > 181,250 B | must not fire | 181,161 | **did not fire** |
| falsifier: decoded field ≠ `9ba2e52b…` | must not fire | see §4 | see §4 |

Built archive sha256 `48d5d469d0d87e72d3465e5a76602ae73e3ce4c331d7491895bde240fcc9eb42`,
181,161 B; member 181,061 B sha `81282d6cfabffbdd28df95d7bf1f534e5fdfaec37c7676a72ce084c49c9908ca`;
token stream 110,512 B sha `72a905cc53dfe366fea01ce50d5114fac239e62e0e167079f2f9f979bf944280`.
A repeat build is byte-identical, and a repeat encode reproduces the token stream byte-identically.

---

## 2. The controls, which are the load-bearing part

A byte count is cheap to produce and easy to produce wrongly. Three controls make this one hard
to fake, and each is fail-closed — the scripts refuse to emit a verdict without them.

### 2.1 The repack layer is exact (zero bytes of its own)

Re-emitting the frontier archive from its own parsed sections, with no change at all, reproduced
`80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e` at 182,759 B —
**byte-identical**. The ZIP member is STORED, so this also fixes the framing at exactly 100 B
(30 + 1 + 46 + 1 + 22). Consequence: the candidate archive differs from the frontier by exactly
the token-stream delta and nothing else.

### 2.2 The encoder is the exact inverse of the shipping decoder

The bypass encode replays the shipped decode order — 600 frames × 190 causal patch groups — and
hands the RC64 encoder the receiver's own probability rows, unmodified.

| quantity | value |
|---|---|
| bypass stream bytes | **112,110** (shipped: 112,110) |
| bypass stream sha256 | **`73a878891a31c3668a0403f842740f21598999fee5c8afd8982fb2ca31125829`** — MATCH |
| code length | **112,109.57757858852 B** |
| independent prior measurement (`dc1`, `hm1`, `rc4`) | 112,109.57757858852 B |
| realised coder overhead | **+0.42242141148017254 B** |

This proves, in one measurement, that the group order, the symbol order, the float32 probability
quantization, the arithmetic coder recurrence and the terminal flush all agree with the shipped
decoder. `ddm_rr1` §2.8 predicted the overhead constant at +0.42242 B; it is that number.

### 2.3 The corrector reproduces `ddm_rr1`'s H1 rung independently

| quantity | `ddm_rr1` stage 3 H1 | `ddm_rr2` encoder |
|---|---:|---:|
| code length | 110,511.28 B | **110,511.27764 B** |
| bytes saved | 1,598.30 | **1,598** (realised stream bytes) |
| contexts | 51,200 | **51,200** |
| warm contexts (count ≥ 32) | 9,613 | **9,613** |

The warm-context count is the sharper of the two: it is a discrete integer that depends on the
whole context construction, the smoothing, the cold-context floor and the per-group update order.
Reproducing 9,613 exactly from an independent implementation is a strong equality check.

### 2.4 The section table, checked by re-parsing rather than by self-comparison

The first version of this check compared the builder's own concatenated slices against themselves
— a tautology that would have passed no matter what was written. It was replaced with a real
check: the built archive is re-opened and parsed by `runtime.residual_archive.read_residual_archive`,
the receiver's own parser, and each parsed section is hashed.

| section | byte-identical to frontier |
|---|---|
| `compressed_models` (`e35d1237…`) | yes |
| `semantic_blob` (`b489c735…`) | yes |
| `carrier_blob` (`196f0e51…`) | yes |
| `hpac_blob` (`e8c0cfd7…`) | yes |
| `compensation_blob` (`38792b49…`) | yes |
| `residual_payload` RCF1 (`74775aab…`) | yes |
| `table_codes` (`76afdc3c…`) | yes |
| `token_stream` | **no — this is the whole change** |

---

## 3. The coder tax, which `ddm_rr1` §5.5 left assumed

`ddm_dc1` measured RC64 at 1.00000 of cross-entropy **on the shipped probability tables**. The
corrected tables are different distributions, and `ddm_rr1` said plainly that it had not
re-measured the tax. Row 1 of its build order settles it by construction, and here is the number:

| stream | code length (B) | stream (B) | realised overhead (B) |
|---|---:|---:|---:|
| shipped probabilities (bypass) | 112,109.57757858852 | 112,110 | **+0.42242** |
| corrected probabilities | 110,511.27763690 | 110,512 | **+0.72236** |

The tax grew by **+0.29994 B** across the whole 600-frame stream. It is a terminal flush effect,
not a rate: the corrected distributions cost a third of a byte more to flush and nothing more.
`DELTA_CLIP = 4.0` bounds the odds shift to 16×, which keeps every RC64 frequency far above the
1-in-2³¹ floor, and no row ever tripped the encoder's positivity or normalization guard across
117,964,800 coded positions.

---

## 4. The receiver, and the distortion proof

`ddm_rr1` §4 step 1 named the insertion point exactly. The patch is five insertions into
`runtime/residual_archive.decode_production_tokens`, applied by exact-string replacement with a
fail-closed occurrence check — the build refuses if any anchor appears other than exactly once:

1. `from .free_corrector import FreeCorrector`
2. `corrector = FreeCorrector(runtime.EVAL_H * runtime.EVAL_W)`
3. `corrector.begin_frame(boundary)` at the frame boundary
4. the group decode now consumes `corrector.coding_row(state)` and calls `corrector.observe(...)`
5. `corrector.end_frame(...)` once the frame is complete

Nothing else in the module changed. The two existing digests — `corrected_quantized_logit_sha256`
and `corrected_cdf_input_sha256` — still hash the **uncorrected** arrays, so they remain live
controls that the base probability path was not disturbed.

**PARSE-BACK VERDICT: the decoded token field is BIT-IDENTICAL.** The candidate archive was
decoded end to end by its own generated runtime — `bash inflate.sh` on the extracted member,
`F26_TOKEN_DECODER=python`, torch 2.12.1, arm64 CPU, 4 threads — and returned:

| digest | value | verdict |
|---|---|---|
| `decoded_token_sha256` | `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` | **MATCH — falsifier did not fire** |
| `token_stream_sha256` | `72a905cc…` (the new 110,512 B stream) | the candidate's, as expected |
| `archive_sha256` | `48d5d469…` at 181,161 B | the candidate's, as expected |
| `corrected_quantized_logit_sha256` | `562ac652b372faa020d0fc5e2ed9b7b61625169e0f5c2041d4fe99196055b8c7` | **MATCH to the frontier** |
| `corrected_cdf_input_sha256` | `dd48843b021763e78524caf3dcd01e944045e7bd0ffd93b451dec83548f083b7` | **MATCH to the frontier** |

The last two are worth their own sentence. They hash the receiver's **uncorrected** logits and
CDF inputs, and they still equal the frontier's values. So the corrector did not perturb the base
probability path at all — it is purely additive on top of an unchanged base, and the shorter
stream comes from the correction and from nothing else. `decoder_bit_position` fell from 896,939
to 884,153, consistent with the shorter stream.

**And the proof did not have to stop at the token field.** The parse-back ran the whole inflate,
and the resulting 3.66 GB output was compared byte for byte against the frontier archive's own
CPU inflate (`ddm_hv1_base_advisory_n600_cpu`, 2026-08-15, same machine, same python decoder,
same 4-thread config):

| | bytes | sha256 |
|---|---:|---|
| frontier base CPU inflate | 3,662,409,600 | `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` |
| **rr2 candidate CPU inflate** | 3,662,409,600 | **`e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`** |

**BYTE-IDENTICAL.** `upstream/evaluate.py` reads exactly this file. So `d_seg` and `d_pose` are
not "unable to have moved by inference" — they are provably the same numbers, measured on the
bytes the scorer consumes. `S_candidate = S_frontier + Δrate` is exact arithmetic over identical
distortion terms, which is a stronger statement than `ddm_rr1` §4 step 3 asked for.

### 4.1 One function used twice, enforced rather than asserted

`ddm_rr1` §4 step 2: *"The two must be one function used twice, not two implementations, or they
will drift."* The candidate runtime's `runtime/free_corrector.py` is a byte-identical copy of
`experiments/ddm_rr2_free_corrector.py`, the module the encoder imports —
sha `ddc915986b3fbdd82116299fdc25774a4eede98028331a3941d16412d10cb2cb`, 10,686 B on both sides.
The build compares the two sha256s and refuses if they differ.

### 4.2 The second decode path is refused, not trusted

The shipping runtime exposes a second token decoder behind `F26_TOKEN_DECODER=native-hpac`
(`runtime/f26_hpac_native.py::decode_production_tokens_native`). This arm did not patch it.
Leaving it reachable would leave an unproven decode path inside a candidate, so the candidate
generation **refuses** it with a fail-closed raise rather than silently decoding a different
field. The native lowering of the corrector is owed (§7 row 2); it is a decode-speed item, not a
correctness one — hv1's own `[contest-CUDA T4]` inflate ran 364.111 s inside the 1,800 s budget
on the python path, 4.944× headroom.

---

## 5. Legality under rule 118, audited rather than asserted

The rate term charges `archive.zip` bytes only; `inflate.py` and `runtime/` are unsized. The
question rule 118 actually asks is whether anything **video-derived** hides in the free code. An
AST audit of the module that enters the runtime:

| probe | result |
|---|---|
| `bytes` literals | **0** |
| literal sequences longer than 8 elements | 1 — a 9-name `__slots__` tuple |
| string literals over 400 chars | 2 — the module and class docstrings |
| module-level constants | **10**, all named and generic |
| distinct numeric literals in the whole file | **11**: 0.0, 1e-9, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 32.0, 64.0, 255.0 |

The ten constants are `NUM_CLASSES=5`, `U_STEP=0.5`, `U_BINS=64`, `RUN_LEVELS=8`, `RUN_CAP=255`,
`BOUNDARY_LEVELS=5`, `KT_ALPHA=0.5`, `MIN_COUNT=32.0`, `DELTA_CLIP=4.0`, `PROB_EPS=1e-9`.
`NUM_CLASSES` and `BOUNDARY_LEVELS` are the format's own alphabets. The rest are first-principles
estimator choices — a Krichevsky-Trofimov smoother, a power-of-two bin width, a cold-context
floor, a symmetric odds clamp — and **none was swept against the scored clip**. A table with
1,598 B of video-derived content cannot hide in eleven numeric literals; this is the audit, not
the assertion.

`ddm_rr1` §5.6 stated the boundary and it is unchanged: sweeping these constants on video 0 and
keeping the argmax turns them into video-derived scalars that must then be counted in
`archive.zip`. They are frozen for exactly that reason.

---

## 6. Honest limits

1. **This is not an exact-eval row.** No Modal, no T4, no `upstream/evaluate.py`. The
   `S = 0.15853325` figure is the frontier's own measured components with the rate term
   recomputed on a real, stat'd `archive.zip`. It is exact **arithmetic** on an advisory axis,
   not an exact **score**. The borrowed pointer did not move and this arm did not move it.
2. **11.09% of the bar.** The remaining ~12,816 B still has no measured supplier on the rate
   axis; `ddm_rr1` §3 puts ~87% of the bar there and that is unchanged.
3. **The encoder replays cached logits, not a live HPAC forward.** It consumes `ddm_hm1`'s
   retained pre-correction logits. That is sound only because the decoded field is unchanged, so
   the decode trajectory the logits were cached along is the trajectory that is replayed — and
   §2.2 is what turns that argument into a measurement rather than leaving it an argument.
4. **The decode-cost comparison is load-confounded.** The candidate's token stage decoded in
   792.045 s against the frontier base run's 566.607 s on the same machine and the same code
   path — **+225.4 s (+39.8%)**. But the machine carried two other heavy jobs throughout (load
   average 5.4–7.4) and the base run's load is unrecorded, so that delta is an **upper bound on
   the corrector's cost, not an attribution**. My own instrument cannot resolve it either: an
   identical corrected encode ran 32.5 s on one pass and 52.4 s on a repeat, so the load noise
   band (±20 s) exceeds the effect. A clean decode-time row belongs on the contest device, where
   hv1's own `[contest-CUDA T4]` inflate measured 364.111 s inside the 1,800 s budget (4.944×
   headroom). **That row is still owed** (`ddm_rr1` NEXT #8).
5. **Composition with any future token field is unmeasured**, exactly as `ddm_rr1` §3 stated. A
   Schur-compensated `rc4` drop changes the field, and the free credit must be re-measured on it.
6. `verdict_scope`: everything here is **INSTANCE** — this archive, this vehicle, this estimator
   form. Nothing is closed and nothing is generalized.

---

## 7. NEXT_IF_RESUMED

Bars must be read from
`tac.canonical_equations.sub015_pure_rate_archive_byte_bar_20260816.pure_rate_byte_bar_from_pointer()`,
never from a literal.

| # | row | disposition | owner | fire condition |
|---|---|---|---|---|
| 1 | **Bundle this 1,598 B with other credit before any T4 fire.** `ddm_rr1` §4 step 5 already said 11.09% of the bar does not justify a solo dispatch, and that is still true. The nearest composable partner is `ra2`'s CPR1 inner coder (~230–278 B, zero distortion), whose own gate condition — "fire when a ≥2 KB rung is in flight" — is now closer to met than it was. | **QUEUED** | rate owner | when a second rate row lands |
| 2 | **Native lowering of the corrector** into `f26_hpac_native.py`, so the `native-hpac` decode path stops being refused. Decode-speed only; the python path already fits the budget with 4.944× headroom. | **QUEUED** | runtime owner | before any run that needs the native path |
| 3 | **Shrinkage / mixing estimator** (`ddm_rr1` NEXT #4). Stage 3 showed the count-based per-context estimator turned over on context richness; an estimator that shares strength across sparse contexts is a different family and is not bounded by any row here. The receiver plumbing this row was waiting on now exists. | **QUEUED** | ddm_rr1 successor | now unblocked |
| 4 | **Re-measure the free credit on any new token field** (`ddm_rr1` NEXT #5). Binding on the Schur-compensated drop, on any HPAC capacity change, and on any semantic-width change. Not additive with them. | **QUEUED** | whoever moves the field | at that arm's harvest |

**Retracted / not claimed:** no exact eval, no score claim, no promotion, no pointer move.

---

## Artifacts (ALWAYS KEEP THE PAYLOAD)

Store root `/Volumes/APDataStore/pact/ddm_rr2_encoder_build/`, every file with sha256 and byte
count in `RETENTION_MANIFEST.json`. Nothing was measured and discarded: both token streams, the
learned corrector statistics, the per-frame code ledger, the member, the archive and its
determinism repeat are all on disk.

**Landed instruments** (each fail-closed):

* `experiments/ddm_rr2_free_corrector.py` — the corrector. One implementation; the encoder
  imports it and the receiver ships a byte-identical copy.
* `experiments/ddm_rr2_encoder_byteclose.py` — repack control, bypass control, corrected encode,
  archive build. Refuses to continue if the repack is not exact or the bypass does not reproduce
  the shipped stream.
* `experiments/ddm_rr2_receiver_close.py` — candidate runtime generation and the parse-back
  proof. Refuses on any patch anchor that is not unique, on a corrector copy that is not
  byte-identical, and on a source tree not bound to the frontier archive.
