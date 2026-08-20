# ddm_rr2 — RECEIPT

**Arm:** ddm_rr2 (encoder build for the `ddm_rr1` free decode-time corrector)
**Date:** 2026-08-17
**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]`
`score_claim: false` · `promotable: false` · no Modal · no T4 · no exact eval
**Memo:** `.omx/research/ddm_rr2_encoder_byteclose_20260817.md`
**Store:** `/Volumes/APDataStore/pact/ddm_rr2_encoder_build/`

---

## 1. VERDICT — the pre-registered target was hit to the byte

| pre-registered row (`ddm_rr1` §4) | target | measured | verdict |
|---|---:|---:|---|
| `archive.zip` bytes | 181,161 ± 2 | **181,161** | **HIT (0 B deviation)** |
| token stream bytes | 110,512 | **110,512** | **HIT (0 B deviation)** |
| every other section byte-identical to `80d9c8c6…` | yes | **7 of 7** | **HIT** |
| FALSIFIER: archive > 181,250 B | must not fire | 181,161 | **did not fire** |
| FALSIFIER: decoded field ≠ `9ba2e52b…` | must not fire | `9ba2e52b…` | **did not fire** |

**ADJUDICATION: NOT FALSIFIED. The byte-close is complete.**

---

## 2. Built artifacts

| artifact | path | bytes | sha256 |
|---|---|---:|---|
| candidate `archive.zip` | `retained/archive.zip` | **181,161** | `48d5d469d0d87e72d3465e5a76602ae73e3ce4c331d7491895bde240fcc9eb42` |
| determinism repeat | `work/archive.repeat.zip` | 181,161 | `48d5d469…` — **byte-identical** |
| candidate member `p` | `retained/member.bin` | 181,061 | `81282d6cfabffbdd28df95d7bf1f534e5fdfaec37c7676a72ce084c49c9908ca` |
| **new token stream** | `retained/token_stream.bin` | **110,512** | `72a905cc53dfe366fea01ce50d5114fac239e62e0e167079f2f9f979bf944280` |
| bypass-control token stream | `work/token_stream_bypass.bin` | 112,110 | `73a878891a31c3668a0403f842740f21598999fee5c8afd8982fb2ca31125829` |
| corrected stream, encode round 1 | `work/token_stream_corrected.round1.bin` | 110,512 | `72a905cc…` — **byte-identical repeat** |
| learned corrector statistics | `retained/corrector_final_state_corrected.npz` | 3,202,157 | `8d2cfc52640a3433921a2eab7dbb7926bab4bbc2d89b71b999d0b47338417a30` |
| per-frame code ledger | `retained/bits_per_frame_corrected.npy` | 4,928 | `c0f2d07babf0a4bf4a75e9c29bb6cdb534ff86a3dfd66d7115fccf0b9013b759` |
| control repack of the frontier | `work/control_repack.zip` | 182,759 | `80d9c8c6…` — **byte-identical to the frontier** |
| RC64 encoder library | `work/librc64_rr2.dylib` | 34,688 | `570902010451854f2072731cac0ee2f50d438ac2ca969e9e17ef22e31eb6dcf2` |
| candidate runtime tree | `candidate_runtime/` | — | see §5 |

Full listing with sha256 + byte count for every file: `RETENTION_MANIFEST.json`.
**Nothing was measured and discarded.**

---

## 3. Byte arithmetic

| quantity | value |
|---|---:|
| frontier `archive.zip` | 182,759 B |
| candidate `archive.zip` | **181,161 B** |
| bytes saved | **1,598** |
| ΔS | **−0.0010640426070892298** |
| S (frontier components, rate term recomputed) | **0.15853325034789675** |
| bar to sub-0.15 | 14,414 B strict (14,413.402 B continuous) |
| share of the bar supplied | **11.087%** |

**The bar is not closed. ~12,816 B still has no measured supplier.** This is one composable
rate row, not the goal.

---

## 4. Controls (all fail-closed; scripts refuse a verdict without them)

### 4.1 Repack layer is exact
Re-emitting the frontier from its own parsed sections reproduced
`80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e` at 182,759 B — byte-identical.
The candidate therefore differs from the frontier by exactly the token-stream delta.

### 4.2 The encoder is the exact inverse of the shipping decoder
Encoding the already-decoded field through the receiver's own probability rows, with no
correction, reproduced the shipped token stream **byte-identically**:

| quantity | value |
|---|---|
| bytes | 112,110 (shipped: 112,110), delta **0** |
| sha256 | `73a87889…` — **MATCH** |
| code length | **112,109.57757858852 B** |
| same number measured independently by `dc1`, `hm1`, `rc4` | 112,109.57757858852 B |
| realised coder overhead | **+0.42242141148017254 B** (`ddm_rr1` §2.8 predicted +0.42242) |

### 4.3 The corrector reproduces `ddm_rr1`'s H1 rung

| quantity | `ddm_rr1` stage 3 H1 | `ddm_rr2` |
|---|---:|---:|
| code length | 110,511.28 B | **110,511.27764 B** |
| contexts | 51,200 | **51,200** |
| warm contexts (count ≥ 32) | 9,613 | **9,613** |
| touched contexts | — | 15,386 |

### 4.4 Per-section identity, checked by re-parsing (not by self-comparison)
The built archive is re-opened and parsed by the receiver's own
`runtime.residual_archive.read_residual_archive`, and each parsed section hashed:

| section | sha256 | identical |
|---|---|---|
| `compressed_models` | `e35d12371fa79747…` | yes |
| `semantic_blob` | `b489c73567046e64…` | yes |
| `carrier_blob` | `196f0e5136f4d6bf…` | yes |
| `hpac_blob` | `e8c0cfd73d3275ad…` | yes |
| `compensation_blob` | `38792b4953318117…` | yes |
| `residual_payload` (RCF1) | `74775aab04c7615c…` | yes |
| `table_codes` | `76afdc3ceda1212a…` | yes |
| `token_stream` | `72a905cc…` vs `73a87889…` | **no — the intended change** |

*(The first version of this check compared the builder's own slices against themselves — a
tautology. It was replaced before any verdict was quoted.)*

### 4.5 Coder tax, which `ddm_rr1` §5.5 left assumed

| stream | code length (B) | stream (B) | realised overhead (B) |
|---|---:|---:|---:|
| shipped probabilities | 112,109.57757858852 | 112,110 | +0.42242 |
| corrected probabilities | 110,511.27763690 | 110,512 | **+0.72236** |

Tax grew **+0.29994 B** over the whole stream — a terminal flush effect, not a rate. No row
tripped the encoder's positivity or normalization guard across 117,964,800 coded positions.

---

## 5. Receiver parse-back — the distortion proof

Decoded end to end by the candidate's own generated runtime: `bash inflate.sh` on the extracted
member, `F26_TOKEN_DECODER=python`, torch 2.12.1, arm64 CPU, 4 threads / 1 interop thread.

| digest | value | verdict |
|---|---|---|
| **`decoded_token_sha256`** | **`9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`** | **BIT-IDENTICAL** |
| `archive_sha256` (bound by the receiver) | `48d5d469…` @ 181,161 B | the candidate |
| `token_stream_sha256` | `72a905cc…` | the new 110,512 B stream |
| `corrected_quantized_logit_sha256` | `562ac652b372faa020d0fc5e2ed9b7b61625169e0f5c2041d4fe99196055b8c7` | **MATCH to frontier** |
| `corrected_cdf_input_sha256` | `dd48843b021763e78524caf3dcd01e944045e7bd0ffd93b451dec83548f083b7` | **MATCH to frontier** |
| `decoder_bit_position` | 884,153 (frontier: 896,939) | shorter stream, as expected |
| `residual_payload_sha256` | `74775aab…` | unchanged |
| `hpac_blob_sha256` | `e8c0cfd7…` | unchanged |

The last two digests hash the receiver's **uncorrected** logits and CDF inputs. They still equal
the frontier's values, so the corrector did not perturb the base probability path at all — it is
purely additive, and the shorter stream comes from the correction and nothing else.

Process `returncode = 0`, total wall clock 1,325.137 s for the whole inflate.

### 5.0 The scorer's own input bytes are identical — the distortion proof, measured

The parse-back ran to the end, not only through the token stage, and the fully inflated output
was compared against the frontier archive's own CPU inflate (`ddm_hv1_base_advisory_n600_cpu`,
2026-08-15, same machine, same python decoder, same 4-thread config):

| | bytes | sha256 |
|---|---:|---|
| frontier base CPU inflate | 3,662,409,600 | `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` |
| **rr2 candidate CPU inflate** | 3,662,409,600 | **`e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`** |

**BYTE-IDENTICAL over all 3.66 GB.** `upstream/evaluate.py` reads exactly this file, so `d_seg`
and `d_pose` are not merely "unable to have moved by inference" — they are provably the same
numbers, measured at the byte level on the CPU axis. `S_candidate = S_frontier + Δrate` is
therefore exact arithmetic over identical distortion terms.

### 5.1 One function used twice — enforced

| copy | path | bytes | sha256 |
|---|---|---:|---|
| encoder imports | `experiments/ddm_rr2_free_corrector.py` | 10,686 | `ddc915986b3fbdd82116299fdc25774a4eede98028331a3941d16412d10cb2cb` |
| receiver ships | `candidate_runtime/runtime/free_corrector.py` | 10,686 | `ddc915986b3fbdd82116299fdc25774a4eede98028331a3941d16412d10cb2cb` |

The build compares the two sha256s and refuses if they differ.

### 5.2 Receiver patch

`runtime/residual_archive.py` `47d0c7b9…` (23,714 B) → `df07a53c4309959556367c3c325ad3f945da0b2e013f9bc80b9f3a9a6b13a645`
(24,203 B). Five insertions, applied by exact-string replacement with a fail-closed
occurrence check; nothing else in the module changed.

### 5.3 The second decode path is refused, not trusted

The shipping runtime exposes `F26_TOKEN_DECODER=native-hpac`
(`runtime/f26_hpac_native.py::decode_production_tokens_native`). This arm did not patch it, so the
candidate generation **raises** on it rather than decoding a different field with an unpatched
model. Native lowering of the corrector is owed; it is a decode-speed item, not a correctness one.

---

## 6. Rule-118 payload cleanliness of the free module

| probe | result |
|---|---|
| `bytes` literals | **0** |
| literal sequences longer than 8 elements | 1 — a 9-name `__slots__` tuple |
| string literals over 400 chars | 2 — module and class docstrings |
| module-level constants | **10**, all named and generic |
| distinct numeric literals in the whole file | **11**: 0.0, 1e-9, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 32.0, 64.0, 255.0 |

Constants: `NUM_CLASSES=5`, `U_STEP=0.5`, `U_BINS=64`, `RUN_LEVELS=8`, `RUN_CAP=255`,
`BOUNDARY_LEVELS=5`, `KT_ALPHA=0.5`, `MIN_COUNT=32.0`, `DELTA_CLIP=4.0`, `PROB_EPS=1e-9`.
None was swept against the scored clip (`ddm_rr1` §5.6 boundary, unchanged). 1,598 B of
video-derived content cannot hide in eleven numeric literals.

---

## 7. Decode cost — measured, and load-confounded

| run | token-stage decode | machine |
|---|---:|---|
| frontier base, 2026-08-15 | 566.607 s | same machine, python decoder, 4 threads |
| **rr2 candidate, 2026-08-17** | **792.045 s** | same machine + corrector, **load average 5.4–7.4** |
| delta | +225.4 s (+39.8%) | **CONFOUNDED** |

The machine carried two other heavy jobs throughout, and the base run's load is unrecorded, so
this delta is an **upper bound on the corrector's cost, not an attribution**. My own instrument
cannot resolve it either: a full corrected encode over 600 frames ran 32.5 s on one pass and
52.4 s on an identical repeat, so the load noise band (±20 s) exceeds the effect being measured.
A clean decode-time row belongs on the contest device, where hv1's own `[contest-CUDA T4]` inflate
measured 364.111 s inside the 1,800 s budget (4.944× headroom). **That row is owed.**

---

## 8. Honest limits

1. **Not an exact-eval row.** `S = 0.15853325` is the frontier's measured components with the
   rate term recomputed on a real, stat'd archive. Exact arithmetic on an advisory axis, not an
   exact score. The borrowed pointer did not move.
2. **11.087% of the bar.** ~12,816 B still unsupplied.
3. **The encoder replays `ddm_hm1`'s cached logits, not a live HPAC forward.** Sound only because
   the decoded field is unchanged — and §4.2 is what turns that argument into a measurement.
4. **Decode cost is load-confounded** (§7).
5. **Composition with any future token field is unmeasured** (`ddm_rr1` §3).
6. `verdict_scope`: **INSTANCE** throughout — this archive, this vehicle, this estimator form.
   Nothing closed, nothing generalized.

---

## 9. Reproduce

```bash
.venv/bin/python experiments/ddm_rr2_encoder_byteclose.py --stage control-repack
.venv/bin/python experiments/ddm_rr2_encoder_byteclose.py --stage control-encode
.venv/bin/python experiments/ddm_rr2_encoder_byteclose.py --stage encode
.venv/bin/python experiments/ddm_rr2_encoder_byteclose.py --stage build
.venv/bin/python experiments/ddm_rr2_receiver_close.py --stage build
.venv/bin/python experiments/ddm_rr2_receiver_close.py --stage parseback
```

Provenance: git `f7e29a124c`, numpy 1.26.4, python 3.13.12, torch 2.12.1, arm64 Darwin.
