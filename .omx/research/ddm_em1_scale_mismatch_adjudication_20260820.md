# ddm_em1 — the fx4 encoder/decoder scale mismatch: adjudicated NOT LIVE

**Arm:** `ddm_em1` · **Date:** 2026-08-20 · **Charter:** task #1147, the fx4-unbundled
encoder/decoder SCALE-MISMATCH finding.
**Authority:** `[local advisory]` — source adjudication plus one read-only decode of the shipping
archive bytes. No scorer run, no new archive, no exact row.

> **Pointer delta: NONE.** Own-vehicle frontier unchanged: **S 0.14827847122030852 @ 180,456 B
> `[contest-CUDA T4 n600]`**, archive `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`.
> This unit produced no frontier progress and none was available — see §4. Saying otherwise would be
> the means-as-ends fake.

---

## 0. VERDICT

**The asymmetry is REAL where it was filed, and NOT LIVE on the rc2 shipping lineage.**

`verdict_scope: SITE-SET × LINEAGE` — the narrowest level that carries the evidence:

- **REAL** at the 13 `*_as_renderer.py` variant-A helpers. Verified at source, not inferred.
- **NOT LIVE** on the lineage that produced archive `df7fd266…` / runtime tree `fdd57749…`, on
  **three independent legs**: it is unreachable there (§1), the *class* is absent from the shipping
  encode/decode pair (§2), and the counterfactual has the **wrong sign** and is bounded at
  ≤ 0.197 % of codes (§3).

This is **not** a claim that the asymmetry does not exist. It exists, off-lineage.

The charter's step 1 disposition applies: honest scoped negative, stop. **Task #1147 closes.**

---

## 1. LEG 1 — REACHABILITY: the filed sites are absent from the shipping packet

The filed construction, verified at source:

`src/tac/ffnerv_as_renderer.py:448-455`
```python
def _quantize_per_tensor_int8_with_fp16_scale(...):
    scale = max(max_abs, 1e-8) / 127.0                                      # :452  fp32
    scale_fp16 = torch.tensor([scale], dtype=torch.float16).clamp(...)      # :453  STORED
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)            # :454  quantized with :452
    return q, scale_fp16
```

Line 454 divides by the **fp32** `scale`; line 453 stores its **fp16** narrowing. The decoder can
only multiply by what was stored. Asymmetric, exactly as `ddm_fx4` filed it.

The shipping packet is self-contained by contest rule, so a reference sweep over it is decisive.
Swept `/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed/`
(**46 files**, read-only, never edited):

| pattern | files matching |
|---|---|
| `_quantize_per_tensor_int8_with_fp16_scale` | **0** |
| `as_renderer` | **0** |
| `ffnerv` · `hinerv` · `mnerv` · `dsnerv` · `tcnerv` · `cnerv` · `blocknerv` · `nervdc` · `ego_nerv` · `e_nerv` · `vqvae_as_full_renderer` | **0** each |

Nothing in the shipped decode side, the shipped `compress.py`, or `cpr1/` reaches the filed code.

---

## 2. LEG 2 — CLASS: the shipping encode/decode pair is already symmetric

Reachability alone answers the narrow question. A single-site check is partial (fractal-audit
discipline), so I swept the *class* — "quantize with a precision the decoder never sees" — across
every quantize/store/dequantize triple on the shipping lineage.

**The live semantic section is `SM3R`**, measured by decoding the real archive: magic `SM3R`,
**36,130 B**. That selects the SM3R encoder/decoder pair, not the WANS1 fallback.

| Shipping surface | Encoder | Decoder | Symmetric? |
|---|---|---|---|
| **SM3R semantic (LIVE)** | `experiments/ddm_sm3_semantic_representation.py:204-207` | `cpr1/ddm_mp2_semantic_receiver.py:105-113` | **YES** |
| WANS1 semantic (fallback, unused by rc2) | same sm3/sd1 lineage | `runtime/entropy/renderer_weight_codec.py:210,220` | **YES** |
| pose carrier basis + coefficients | `pack_semantic_pose.py:84-87, 182-184` | `cpr1/carrier_codec.py:180-188` | **YES** (fp32 `<f4`, no narrowing at all) |
| pose semantic weights | `pack_semantic_pose.py:125-129` | `pack_semantic_pose.py:157-170` | **YES** |
| semantic training-time quantizers | `train_semantic_quantized.py:55-61` · `evaluate_semantic_quantization.py:43-49` | same value (STE) | **YES** |

The live pair, verbatim:

`experiments/ddm_sm3_semantic_representation.py:204-207` — **encoder**
```python
scales = source.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / limit   # :204 fp32
scales = scales.to(torch.float16).clamp(min=_FP16_MIN_POSITIVE)                     # :205 NARROW FIRST
codes  = (source / scales.float()).round().clamp(-limit, limit).to(torch.int8)      # :206 quantize with :205
restored = codes.float() * scales.float()                                           # :207 same value
```
…and `:210` stores `scales.astype("<f2")` — bit-exactly the value used on line 206.

`cpr1/ddm_mp2_semantic_receiver.py:108-112` — **decoder**
```python
scales = np.frombuffer(scale_view, dtype="<f2").copy()      # :108
restored = codes.reshape(template.shape).float()            # :111
restored *= torch.from_numpy(scales).float().reshape(...)   # :112
```

Encoder and decoder use **the same fp16 number**. The shipping lineage already implements the very
fix `ddm_fx4` proposed for variant A (`quantize with float(scale_fp16)`); it was never asymmetric.

The pose carrier is stronger still: it stores `<f4`, so the stored scale *is* the encoder's fp32
scale, bit for bit. No narrowing exists, so no mismatch is representable.

---

## 3. LEG 3 — MAGNITUDE: the counterfactual is bounded, and points the wrong way

Even a not-live finding deserves a price, so the negative can be trusted rather than assumed.

**Measured on the real shipping bytes.** I decoded archive `df7fd266…` (sha verified before use)
with the **frozen** shipping runtime and captured the real per-tensor `(codes, fp16 scales)` by
wrapping the receiver's own `_decode_quantized` — real inputs, no synthetic fixture.

```
quantized tensors captured ............ 16
total quantized elements .............. 59,376
max |code| over all tensors ........... 7          (the 4-bit limit is reached)
any zero scale anywhere ............... false
worst-case flip fraction upper bound .. 0.001969   (0.197 %)
worst-case flipped codes upper bound .. 116.9      of 59,376
```

Derivation of the bound (distribution-free; needs no access to the original fp32 weights):
the shipping code is `k = round(w/s16)`, so `w/s16 ∈ [k−0.5, k+0.5)`. Variant A would compute
`round(w/s32)` with `fp16(s32) = s16`, and fp16 round-to-nearest gives `|s32/s16 − 1| ≤ 2⁻¹¹`.
The argument therefore shifts by at most `|k|·2⁻¹¹`, so the per-code worst-case flip fraction is
`min(1, 2·|k|·2⁻¹¹)`. Summing over the real codes gives the table above.

**The sign matters more than the size.** `round(w/s16)` is by construction the nearest code *under
the scale the decoder will actually apply*. Variant A's `round(w/s32)` is nearest under a scale the
decoder never sees, so every one of those ≤ 117 flips moves the reconstruction **away** from `w`.
The available direction is variant-A → shipping, and the shipping side is already there.

**There is no frontier row here**: no bytes to move, and the only reachable change is a loss.

Bonus, from the same real decode: `any zero scale anywhere = false` confirms the sister `ddm_fx4`
ROW-1 fp16 zero-scale collapse is likewise not realized in the shipped bytes.

---

## 4. WHY THIS IS NOT FRONTIER PROGRESS

The charter framed this as potentially archive-byte-moving. It is not. `ddm_fx4`'s "would change
archive bytes for *every* tensor" is correct **for variant A**, whose encoder would have to be
re-run; it does not transfer to rc2, whose encoder never had the defect. Adjudicating first — rather
than fixing first — is what kept this from becoming a byte-moving A/B against a baseline that was
already optimal.

---

## 5. PROVENANCE PINS

| Artifact | Identity |
|---|---|
| shipping archive | `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`, 180,456 B (re-hashed before use) |
| archive member | `p`, 180,356 B, stored, sha `83fa979c1118499b7dd6083cb20bb66f3f8f47e32cfc16ff30ea66449d81cdf3` |
| runtime tree | `fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2` |
| frozen packet | `/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed/` — read-only; **nothing under it was modified** |
| decode runtime used | frozen `runtime/` + `cpr1/` copied to scratch; copies re-hashed and confirmed byte-identical to the frozen originals before import |

**Tracked-vs-frozen runtime.** `submissions/robust_current/jg5_sub015_runtime/runtime/runtime/`
is the **jg5** generation, not rc2. Of 25 shipped `.py` files: **22 identical, 1 differing**
(`residual_archive.py`), **2 frozen-only** (`native_free_corrector.py`, `rr5_arith_basis.py`).
That is the expected rc2-over-jg5 delta, not drift. Where they differ I used the **frozen** copy as
authority; `renderer_weight_codec.py` and `f26_inflate.py` are byte-identical either way.

---

## 6. RETAINED PAYLOAD (P0 — nothing measured was discarded)

`/Volumes/APDataStore/pact/ddm_em1/retained/`

| File | sha256 | bytes |
|---|---|---|
| `rc2_real_sm3r_codes_and_scales.npz` (28 real arrays: codes + scales for all 16 quantized tensors) | `cdd4535249002d740db73acebd61dbedb4734b0cca24a0d71213a33057ec385a` | 38,735 |
| `em1_scale_symmetry_probe_v2.json` (per-tensor report) | `d4d250a236fc7a7d66b4f464b6db0ff874463fe727d2280764f14308ed4405dc` | — |
| `em1_scale_symmetry_probe.json` (v1, SM3R-magic discovery run) | `f5970c67f97ea63723c6ef65308fd3e22b2b2a39b810f0bef722b448ee6f29e7` | — |

The captured codes/scales are the real shipped values; they are kept so the next consumer can prove
byte-identity rather than re-decode.

---

## 7. NON-VACUITY

A negative is worthless if the instrument could not have fired. Denominators, stated:

- reachability sweep scanned **46 packet files**, not zero;
- the class sweep found and read **5 encoder/decoder pairs**, not zero;
- the magnitude probe captured **16 tensors / 59,376 real elements**, `max|k| = 7` — the bound sits
  at its worst case, not at a degenerate zero;
- the probe asserts the archive sha **before** decoding, so it cannot silently measure other bytes.

The one control I did **not** get to run: re-encode idempotency on the shipping tensors. It is not
sound here — the decoder's output lies exactly on the reconstruction lattice, so re-deriving `amax`
yields a different scale whenever `max|k| < limit`. Reporting it as a passing control would have
been a false green. The source reading in §2 is the load-bearing evidence, and it is direct.

---

## 8. WHAT I DID NOT DO

- **No fix, no A/B, no seal, no Modal.** Charter step 1 disposition: not live ⇒ stop.
- **Did not touch the frozen packet** or `submissions/`.
- **Did not fix the 13 off-lineage variant-A sites.** They stay asymmetric. Off the shipping
  lineage this is dead weight, not damage — but §3's mechanism says any future vehicle reusing
  `*_as_renderer.py` would inherit a systematic bias of ≤ `|k|·2⁻¹¹` in the wrong direction. The
  one-line cure is `q = (tensor / scale_fp16.float()).round()…`, matching `ddm_sm3:206`. Priced for
  MAIN as **low value, low cost, latent-only**; not claimed as score work.

## 9. NEXT-IF-RESUMED

1. Nothing on this row. #1147 is closed by §0.
2. *Optional, unrelated to score:* fold the 13 variant-A sites onto the `ddm_sm3:204-207` form and
   extend the existing `test_fp16_scale_floor_guard.py` class sweep to require narrow-before-quantize.
   This is correctness hygiene for future lineages, **not** a frontier row — do not let it be
   reported as one.
