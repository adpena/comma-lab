# ddm_bz2d — bz2 born-object distortion: REFUSED at 99.68×, and the transfer law that closes the family

**Date:** 2026-08-30 · **Task:** #1333 · **Axis:** `[macOS-CPU frozen-scorer advisory]` ·
`score_claim=false` · `promotable=false`
**Own-vehicle frontier: lb1 — S 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], UNMOVED.**

## 1. Verdict

The pre-registered falsifier (written before the number, `PREREGISTERED_ADJUDICATION.md`) said:

> realized seg + pose ≤ **0.052840** ⇒ THE CROSS intersection is non-empty at n=4 and bz2 is a
> sub-0.12 candidate outright. > 0.052840 ⇒ bz2 fails, and the magnitude decides the
> FIT-vs-RENDER question for the whole small-body family.

**MEASURED: realized seg + pose = 5.266919383676105 — 99.68× the bar. bz2 FAILS.**

Recomputed FROM COMPONENTS (#877; the row's own `final_score` field reads the rounded 5.37):

| term | value | contribution |
|---|---:|---:|
| d_seg | 0.01299522 | 100·d_seg = **1.299522** |
| d_pose | 1.5740242 | √(10·d_pose) = **3.9673973836761047** |
| rate (measurement vehicle, 157,024 B) | 0.004182233450202234 | 25·r = **0.10455583625505584** |
| **S (vehicle)** | | **5.371475219931161** |

The distortion legs are **bz2's own** (see §2), so bz2's own body carries them at its own rate:
**S(bz2) = 5.334079249405913** at 100,862 B. Not a score — an advisory row on the PyAV lineage.

## 2. Why these numbers ARE bz2's — the control that made it a measurement

The measurement vehicle is bz2's token field re-encoded through lb1's shipped HPAC model and
spliced into lb1's body (157,024 B, sha `34c4efd96c406f34024d9f72fb3f66ce47aaf6a09f725a2a6232495af368f0eb`).

Two independent proofs that its render **is** bz2's render:

1. **Token-identity control PASSED** (`TOKEN_IDENTITY_CONTROL.json`, tool
   `experiments/ddm_bz2d_token_identity_control.py`, committed `2a80b5d430`). The shipping runtime's
   decoded field is byte-identical to bz2's retained `archive_parseback_tokens.u8` —
   117,964,800 B, sha `968ffca296302616…`. The control is non-vacuous by construction: it refuses
   self-comparison by path-resolve **and** inode, and all four directions were executed (same-path
   rc=2 · symlink rc=2 · distinct-but-equal rc=0 · one-flipped-byte rc=3 reporting
   `first_difference_offset 4097`, `pairs_differing 1` at pair 4 — exactly where the byte was injected).
2. **The RX1M header is field-for-field identical to lb1's**, parsed with the shipped parser
   (`runtime/residual_archive.py::_decode_rx1_models`, format `<4sBBBBHHH`):
   `ver=1 codec=2 table_mode=0 reserved=0b11010 hpac=13515 semantic=30856 carrier=22010`.
   Only the token stream differs. That is the construction proof, not a narration of one.

Because the renderer is deterministic, identical tokens ⇒ identical frames ⇒ **this row's
d_seg/d_pose are bz2's distortion, measured through the real inflate/evaluate chain rather than
transferred from a sibling** ([[m143]]: cross-regime transfer has cost three arms).

## 3. THE TRANSFER LAW — token error → argmax error AMPLIFIES (the reusable product)

bz2's field is a GT fit: `gt_mismatch_fraction` = **0.011229510837131076** (1.1230% of the
117,964,800 native token positions disagree with ground truth;
`NATIVE_ANCESTOR_COMPARISON.json`, read at source).

**Measured ratio: d_seg / token_error = 0.01299522 / 0.011229510837131076 = 1.1572382972400272.**

The render → R → uint8 → SegNet chain is a mild **amplifier**, not an attenuator. The
pre-registration explicitly refused to quote an attenuation factor and said "the attenuation is
unquantified; the measurement is the first number that will constrain it." It now does — and the
constraint points the wrong way for the whole family.

What that licenses, as a bar on ANY GT-fit body:

| distortion budget | required d_seg | required token accuracy | bz2's shortfall |
|---|---:|---:|---:|
| all 0.052840 to seg (pose free) | ≤ 0.00052840 | token error ≤ **0.0457%** | **24.6×** |
| lb1-like pose held (0.007981) | ≤ 0.00044859 | token error ≤ **0.0388%** | **29.0×** |

98.877% token-correct is not close. Sub-0.12 needs ~99.96%.

## 4. Pose: destroyed, and the mechanism is our own measured law

d_pose = **1.5740242** — 247,100× lb1's CUDA 6.37e-6. The #1142 GT-lineage fork spans
0.887–1,627× per pair, so this is a REGIME read, not a like-for-like number. But it is robust
across the *entire* fork range: even dividing by the most generous 1,627× leaves **152× worse**.

Mechanism = **[[#1222]]**: PoseNet scores the FRAMES, so the RENDERER carries pose. Swapping the
token field while keeping the pose carrier byte-identical does **not** preserve pose — different
tokens render different frames. The pre-registration had already retracted my earlier
"construction-protected pose" hypothesis on exactly this ground, *before* the number landed; the
measurement confirms the retraction was right.

**Consequence beyond bz2:** every "swap the field, keep the machinery" composition inherits this.
The pose carrier is not a pose guarantee.

## 5. Matched-lineage comparison (both PyAV, same GT table, same instrument)

Comparing this PyAV row against lb1's CUDA d_seg would be the wrong-object error. The matched
baseline is bo2's dx2-base row (`perclass_base_dx2.json`,
`gt_lineage: "PYAV_YUV420_TO_RGB via AVVideoDataset (authority=False)"` — byte-identical lineage
string to this advisory's own log): `avg_segnet_dist_recomputed` = **0.00034740024142795135**.

**bz2 / dx2-base = 37.407×** on the matched instrument. The ratio is the transferable quantity;
the absolute PyAV d_seg is not.

## 6. THE CROSS: intersection MEASURED EMPTY at n=4

| body | bytes | rate | distortion (seg+pose) | verdict |
|---|---:|---:|---:|---|
| lb1 pointer (CUDA authority) | 180,083 | 0.119910 | **0.028120** | 42,097 B OVER cap |
| born-small qbt2b r10 | 121,928 | 0.081187 | 0.327712 | 8.4× over budget |
| NR1 K32 | 122,250 | ~0.0814 | 27.716026 | 984× over |
| **bz2 (this row)** | **100,862** | **0.067160** | **5.266919** | **99.68× over** |

Four measured bodies. Every byte-feasible one is distortion-dead; the only distortion-feasible one
is 42,097 B over. **{byte-feasible} ∩ {distortion-feasible} is measured EMPTY at n=4.**

And bz2 sharpens the [[the-cross-two-objects-each-hold-one-half-of-sub012]] no-trend law: bz2 and
born-small sit **21,066 B apart in rate and 16.1× apart in distortion**; bz2 and NR1 sit
21,388 B apart and 5.26× apart. Among small bodies, byte position predicts nothing about distortion.

## 7. THE REPRESENTATION RACE — the positive product, measured at source

Holding the FIELD fixed (bit-identical object) and swapping only the REPRESENTATION. All figures
parsed from the containers today, payload-level (zip overhead 100 B on each, cancels):

- lb1 payload 179,983 B = hdr 14 + hpac 13,515 + semantic 30,856 + carrier 22,010 + **tokens 113,588**
- vehicle payload 156,924 B = the identical 66,395 B models block + **tokens 90,529**
- bz2 payload 100,762 B = **generator 47,779** + semantic 30,856 + carrier 22,010 + framing 117

bz2's semantic renderer and pose carrier are **byte-identical to lb1's**, proven by substring hit
inside bz2's payload at offsets **21** and **30,877**, shas `39d1be52ba629334…` / `932b979f5181b331…`.

**Same field, two representations:**

| representation | bytes |
|---|---:|
| {RX1M hdr 14 + HPAC model 13,515 + coded tokens 90,529} | **104,058** |
| HG1C generator packet | **47,779** |

**The generator is 2.178× cheaper for a bit-identical object.**

The −79,221 B archive delta decomposes with **zero remainder**:

| leg | mechanism | bytes |
|---|---|---:|
| A | bz2's field is intrinsically cheaper to code (same model, same coder) | **−23,059** |
| B | generator representation beats {model + coded tokens} | **−56,279** |
| C | framing residue | **+117** |
| | **sum** | **−79,221** ✓ (= 100,862 − 180,083) |

Leg B survives bz2's death as a candidate: it is a fact about *representations*, measured on a
bit-identical field. **71.0% of the rate advantage is representational, not field-quality.** A
successor body that wants born-small's rate should inherit the generator FORM, not the GT-fit field.

## 8. What is NOT claimed

No score. PyAV GT lineage, not the DALI table (#1142): d_seg primary (seg fork 1.43×), d_pose
REGIME-ONLY (pose fork 0.887–1,627×). bz2's own fire order asked for a DALI-lineage terminal; this
run does not satisfy that half, and substituting one lineage for the other silently would be the
wrong-object error. The pose conclusion is robust across the whole fork range (§4) — the seg
conclusion is 24.6–29.0× clear of its bar, far outside any lineage effect.

## 9. Routing

- **bz2 CLOSED**, verdict_scope **FORMULATION**: GT-fit generator bodies at ~98.88% token accuracy,
  rendered through the dx2/lb1 semantic renderer. Reactivation requires a field at ≥99.96% token
  accuracy AND a pose mechanism that survives a field swap — both named, neither in hand.
- **The transfer law (§3) and the representation race (§7) are the durable products.** They bind
  every future small-body proposal: price the token accuracy against the 0.0388–0.0457% bar
  *before* building, and inherit the generator FORM rather than the fitted field.
- Per-class seg decomposition running on the matched bo2 instrument (`--gt-argmax-cache`
  guarantees the same GT object); it will attribute the 37.407× by class for the successor design.
