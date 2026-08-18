# Cross-section joint coding on the sz1 pointer — REFUTED, with a mechanism

`verdict_scope`: **INSTANCE** — the four coded sections of the sz1 pointer archive
(sha `debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a`, 179,930 B),
measured byte-exact through real Brotli. No entropy estimates.

Receipt: `/Volumes/APDataStore/pact/ddm_xs1/XS1_JOINT_CODING.json` + the four raw
payloads persisted beside it (P0 ALWAYS KEEP THE PAYLOAD — written before measuring).

## 1. The pointer's byte table (exact, closes to the byte)

Member `p` = 179,830 B; + 100 B ZIP overhead (30+1 local · 46+1 central · 22 EOCD —
the minimum for a one-member ZIP) = 179,930 B.

| section | coded B | share | raw B | ratio |
|---|---:|---:|---:|---:|
| `token_stream` | 109,801 | 61.0% | — (rc64-coded) | — |
| `semantic_blob` | 34,243 | 19.0% | 36,051 | 0.950 |
| `carrier_blob` | 22,161 | 12.3% | 22,242 | **0.996** |
| `hpac_blob` | 13,515 | 7.5% | 17,952 | 0.753 |
| `residual` + `compensation` | 136 | 0.08% | 136 | 1.000 |
| ZIP overhead | 100 | 0.06% | — | — |

`compressed_models` = 69,933 B coded holds 76,281 B raw across **four separately-coded
sub-streams**. `section` = 109,897 B (residual 96 + tokens 109,801). 69,933 + 109,897 =
179,830 ✓ exact.

## 2. The hypothesis and its refutation

**Hypothesis:** four independent coder streams pay four independent model warmups and
cannot match across each other. All four derive from the same 600-pair video, so joint
coding should recover cross-section mutual information. The receiver splits back by the
length prefixes it already stores — free under rule 118, decode-identical by construction.

**MEASURED, same instrument (brotli q11 / lgwin 24) throughout:**

| arm | bytes | vs independent |
|---|---:|---:|
| independent (control) | 70,594 | — |
| joint, best of all 24 orders | 70,799 | **+205** |
| joint, worst order | 71,066 | +472 |

All 24 orders LOSE. Conditional costs `cost(X|Y) = brotli(Y‖X) − brotli(Y)`:

| pair | conditional | solo | mutual gain |
|---|---:|---:|---:|
| carrier \| semantic | 23,490 | 22,192 | **−1,298** |
| semantic \| carrier | 36,105 | 34,807 | **−1,298** |
| hpac \| carrier | 13,934 | 13,555 | −379 |
| hpac \| semantic | 13,855 | 13,555 | −300 |
| semantic \| hpac | 35,092 | 34,807 | −285 |
| carrier \| hpac | 22,443 | 22,192 | −251 |

**Every** ordered pair is negative. Cross-section mutual information as Brotli can
exploit it is not merely absent — it is *anti-present*.

**Mechanism:** these streams are already near-entropy and mutually *dissimilar*.
Concatenation forces one adaptive model and one shared context/Huffman assignment across
statistically different sources; the model spends more adapting than it recovers in
matches. Four-stream independence is correct design, not an oversight.

## 3. Two facts the control produced that are worth more than the hypothesis

**(a) The shipped coder beats generic Brotli by 661 B.** Same-instrument independent
total is 70,594 B; the shipped `compressed_models` is **69,933 B**. The container uses
per-stream tuned coders that a generic brotli-q11 pass cannot match. Any future "just
re-Brotli it" proposal starts 661 B in the hole.

**(b) `carrier_blob` is incompressible: ratio 0.9978** (22,242 → 22,192, 50 B on 22 KB).
It is already AR1-arithmetic-coded upstream (`coefficient_ar1_codec`). Brotli sees
nothing there. `semantic_blob` at 0.950 is nearly as tight.

Concordant with #996 (coder axis closed on all 4 sections vs their own memoryless bound)
and #1060 (all 38/38 semantic tensors receiver-required, 0 derive-at-decode, every exact
recoding +340 B). The lossless rate axis on this vehicle is measured shut at the
section level; rate progress must come from **representation** (fewer/smaller symbols),
not from **coding**.

## 4. What this does NOT close

- The sz1 win itself (−520 B) was a within-section byte-ORDER permutation at offset 49,
  length 8,284. That mechanism is alive and only three sections were ever probed
  (`extension_other_sections.json`: semantic PAYS −520, carrier NO WIN +12, hpac NOT
  MEASURABLE — its re-Brotli control missed the shipped length by 40 B, so the
  instrument was never calibrated on it).
- `token_stream` (61.0% of the archive) was never probed for byte-planing. Expected
  null — it is an rc64 arithmetic-coded bitstream, near-incompressible by construction —
  but that is a *derivation*, not a measurement, and it is stated as such here.

## 5. Routing

Joint/shared-dictionary coding across sections: **CLOSED at INSTANCE scope** with a
named mechanism. Do not re-propose without new evidence that the sections' statistics
have converged.
