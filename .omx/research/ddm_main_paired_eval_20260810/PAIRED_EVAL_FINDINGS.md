# Paired local exact eval — PR130 base vs cp2 rank-1 composed

**Axis:** `[macOS-CPU advisory; upstream AV GT; immutable evaluate.py; n600]` · `score_claim=false`
**Not** the contest-CUDA/DALI axis that carries the 0.172141297 bar. The DELTA is the finding.

## The headline: the composed −8,688 B was never a win

| | base (PR130 CPR1) | cp2 rank-1 composed |
|---|---:|---:|
| archive bytes | 191,052 | 182,364 (−8,688) |
| d_seg | 0.00042735 | 0.01473595 (**34.5×** worse) |
| d_pose | 0.00015911 | 3.47795892 (**21,859×** worse) |
| rate term | 0.12721375 | 0.12142875 |
| **S** | **0.2098374** | **7.4924** |

The −8,688 B buys 0.005785 S of rate and costs ~7.28 S of distortion. **Not bankable.**

## Harness validated by prediction, not by assertion

Before running, the base row was PREDICTED from ai1's independently-measured
AV-GT row: ai1's realized raw was byte-identical to the base raw, so its
distortion IS the base distortion — 0.208229 − rate(188,636 B) = 0.082635 —
giving base = 0.082635 + rate(191,052 B) = **0.209838**.
Measured: **0.2098374**. Match to 6 significant figures.

That cross-validates three things at once: hp3's retained raw is the base
render, ai1's axis label was correct, and this harness is wired correctly.
The catastrophic candidate number is therefore the CANDIDATE, not the instrument.

## Attribution: exact, and $0 (no decode, no scorer)

Direct comparison of the two retained raws, by frame parity:

| frame parity | mean abs delta | max abs delta | pixels changed |
|---|---:|---:|---:|
| EVEN (pose carrier) | 0.0000 | 0 | 0.00% |
| ODD (semantic) | 67.71 | 240 | 99.26% |

Even frames are BYTE-IDENTICAL. So `ai1`'s ANS + temporal-reversion token leg
is confirmed losslessAGAIN, independently. 100% of the damage is the semantic
render path = `sm3`'s pointwise low-rank r32.

## Characterization: defect, not graceful degradation

Odd-frame statistics vs base, sampled at frames 1/3/101/601:

- base mean ~130.4, sd ~68.5
- cand mean ~77.6, sd ~49.5, 239-245 unique values, correlation **+0.30 to +0.32**

Structure survives (not a collapse), but a CONSISTENT DC shift (-52) and
variance compression (0.73x) on every frame, with correlation ~0.31, is the
signature of a factorization/dequant defect -- e.g. low-rank applied without
centering, so rank-32 spends itself representing the mean instead of the
structure. A genuinely over-aggressive rank-32 approximation degrades
gracefully; it does not shift the DC of all 600 frames by the same amount.

**VERDICT SCOPE: INSTANCE** (this pointwise-low-rank-r32 implementation).
NOT a family verdict on low-rank semantic coding. The named next measurement is
whether the SM3R packer/receiver centers before factorizing.

## Corrected rate ledger on the PR130 base

| candidate | section | delta bytes | distortion status |
|---|---|---:|---|
| ai1 ANS + temporal_reversion | tokens | -2,416 | LOSSLESS (raw byte-identical, twice confirmed) |
| hp3 requant frame_embed step2 | hpac | -8 | LOSSLESS (raw byte-identical) |
| sm3 pointwise low-rank r32 | semantic | -6,272 | **REFUTED** (S 0.2098 -> 7.4924) |
| sm3 vector/scale VQ32 | semantic | -4,648 | UNMEASURED (same receiver, same suspicion) |
| SD1 mixed q3/q4 | semantic | -848 | UNMEASURED |

**Bankable today: -2,424 B** (ai1 + hp3), = 7.3% of the 33,252 B sub-0.15 rate
target -- NOT the 26.1% the composed figure implied. Every downstream budget
that counted sm3's -6,272 B must be re-derived.

## What this vindicates

`cp2` did NOT overclaim: its receipt recorded `d_seg_status: UNMEASURED` and
`d_pose_status: UNMEASURED` explicitly and honestly. The exposure was MAIN
carrying "-8,688 B = 26.1% of the rate target" forward as if byte-closure implied
score-safety. Byte-closure and parse-back prove the BYTES round-trip. They say
nothing about whether the reconstructed model still works. Those are different
claims and this is what the difference costs.

## Custody

- harness + arms: `/Volumes/VertigoDataTier/pact/ddm_main_paired_eval_20260810/`
- base raw sha256 `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353` (= established base raw hash)
- cand raw sha256 `46ca24e7004c5a3ea42a118981a4fdf6a523e9d5b56cf6baff4444a062176f32` (matches cp2's own receipt)
- both raws retained; both reports retained; no payload discarded.

## ADDENDUM — the defect read at source, and a zero-byte-cost discriminator

Read at source, `sm3r_receiver.py::_decode_lowrank`:

```python
left,  remaining = _decode_standard_q4("factor.left",  left_template,  remaining)
right, remaining = _decode_standard_q4("factor.right", right_template, remaining)
restored[name] = (left @ right).reshape(value.shape)
```

Two facts, both confirmed in code:
1. **No centering / no mean term.** Rank-32 must also represent the DC structure.
2. **Both FACTORS are int4-quantized**, then multiplied. Reconstruction is
   `dequant_q4(L) @ dequant_q4(R)` -- quantization error in each factor multiplies
   and accumulates across the 32 rank terms. The standard path quantizes W DIRECTLY,
   where per-element error is bounded.

Targets are `coord_mix.weight` + `blocks.{0..3}.pw.weight` -- the pointwise-conv
capacity of the semantic model.

STATUS: this is a code-level READ plus the pixel-level signature (consistent DC
shift, variance compression, corr ~0.31). It is a strong hypothesis for the
mechanism, NOT yet a measurement of it. Labelled accordingly.

### Byte arithmetic (derived from the receiver's own format)

standard q4 = `rows*cols/2 + rows*2` · lowrank q4 = `(rows*r)/2 + (r*cols)/2 + (rows+r)*2`
Break-even rank: **r* = rows*cols/(rows+cols)** (= n/2 for square n x n), so r=32
only saves bytes when the tensor exceeds 64 wide.

| shape | std q4 | r32-int4 | r16-int8 (same bits/factor-element x2) |
|---|---:|---:|---:|
| 128x128 | 8,448 | 4,416 | ~4,384 |
| 96x96 | 4,800 | 3,328 | ~3,296 |

### The discriminator (ZERO byte cost)

**r16-int8 costs the same bytes as r32-int4** but quantizes each factor 16x finer.

- If the defect is FACTOR-QUANTIZATION COMPOUNDING -> r16-int8 beats r32-int4 decisively.
- If it is RANK INSUFFICIENCY -> r16-int8 is worse.

Either outcome is a mechanism verdict, at equal bytes, on the same receiver.
Add centering (store a per-row mean, tiny) as the second arm.

This is the named next measurement for the -6,272 B that the paired eval withdrew.
It is worth running because that figure is 2.6x the entire currently-bankable saving.

---

## ADDENDUM 2 (MAIN, 2026-08-10) — a correction to my own corrected ledger, and one scorer pass closed by arithmetic

### The correction: -2,424 B assumed an additivity the producing arm had already disclaimed

Addendum 1 stated the bankable saving as **-2,424 B = ai1 -2,416 + hp3 -8**. That addition is not
licensed. `ddm_hp3`'s own report says so at line 123, verbatim:

> require a new real archive because the `-8 B` result is not additive with a changed token coder.

The mechanism is in hp3's decomposition: its -8 B is not one number, it is
**-548 B (joint model XZ) + 516 B (token Range stream) + 24 B (seek checkpoint)**. The +516 B
token term is measured against the BASE Range coder. `ai1` REPLACES that coder with ANS. Under a
different coder the token term is a different number, so the net is unknown until a real composed
archive is built and stat'd.

**Corrected ledger:**

| claim | bytes | share of the 33,252 B sub-0.15 rate target | status |
|---|---:|---:|---|
| `ai1` ANS token re-code | **-2,416** | **7.27%** | BANKABLE — lossless by construction (same symbols, fewer bytes; canonical token sha reproduced) |
| `hp3` frame-embed step2 | -8 | 0.02% | **NOT ADDITIVE** with ai1 — needs a real composed archive |
| `sm3` low-rank r32 | -6,272 | — | REFUTED (S 0.2098374 -> 7.4924) |

The magnitude of this correction is negligible (7.27% vs 7.29%). **The kind is not.** It is the
same assumed-additivity error that the paired eval had just punished on the distortion axis, made
again on the rate axis in the same document, against an explicit written warning from the arm that
produced the number. Recording it here because a ledger that quietly rounds in its own favour is
worse than one that is 8 B smaller.

### hp3's queued exact eval is CLOSED BY ARITHMETIC — do not spend the slot

`hp3` measured its realized raw output at sha256 `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`
— **byte-identical to the base raw**. That is a measurement, not an assumption: the re-quantization
changes 2,371 of 4,800 int8 frame-embedding values and none of them survive to a rendered pixel.

`upstream/evaluate.py` computes d_seg and d_pose as deterministic functions of
`submission_dir/inflated/<name>.raw` against a fixed GT. Identical raw bytes therefore give
**identical distortion terms by construction**. The only quantity that moves is `archive.zip` size,
and that delta is exactly -8 B. So both axes are already fully determined:

- `[contest-CUDA, DALI GT]`: S = 0.172141297491896447 - 0.000005327 = **0.172135970620271**
- `[macOS-CPU advisory, AV GT]`: S = 0.209837400 - 0.000005327 = **0.209832073**

A scorer pass would consume the sole slot for ~7 minutes to reproduce two numbers we hold exactly.
**Row closed. Not deferred — closed.** It reopens only if the composed-with-ai1 archive changes the
realized raw, which is precisely the composition measurement named above.

### What the fleet is doing about all of it

Four arms spawned on the two live rate levers and the two screening debts:

- `ddm_sm4` — the r16-int8 vs r32-int4 equal-byte discriminator from Addendum 1. Recovers the
  -6,272 B if the defect is factor-quantization compounding rather than rank insufficiency.
- `ddm_hm1` — HPAC model capacity vs token bytes as ONE joint budget. tokens are 61.23% of the
  archive and rc2 closed the coder axis on every section; the model is the surviving lever.
- `ddm_sv3` — screens the two still-unmeasured semantic candidates (VQ32 -4,648 B, SD1 -848 B)
  with the refuted low-rank as a mandatory positive control.
- `ddm_pz2` — the pose section as a representation problem: 23,384 B for 3,600 score-relevant
  scalars, with the measured 6.83x AV-vs-DALI pose gap as a hard no-transfer constraint.

**BASE: PR130 CPR1 S = 0.172141297491896447 @ 191,052 B `[contest-CUDA, DALI GT, n600]` — UNMOVED.**

---

## Addendum 3 — the CUDA row came back, and it is REFUSED for a real reason

`fc-01KZNSY6WYB5YXZQFXS2N0YASW` / `ap-H0plwfCQHJCEmjprn1vyEV`, Tesla T4, 665.16 s, n600, on the ai1
ANS+temporal_reversion archive sha `0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84`,
188,636 B. Full result + 8 artifacts retained at
`/Volumes/VertigoDataTier/pact/ddm_main_ai1_exact_20260810/cuda/` (RESULT.json sha `26960f5d9e23a08b…`,
ARTIFACT_MANIFEST.json carries per-file bytes + sha256).

**The row is `score_claim=False`, `evidence_grade='auth-eval env mismatch advisory'`, rc=10.**
It is NOT a contest-CUDA row and the pointer does NOT move.

### What it measured

| quantity | ai1 archive (this run) | PR130 base (contest bot, 2026-07-21) | delta |
|---|---:|---:|---:|
| d_seg | 0.00029661 | 0.00029660 | +1e-08 |
| d_pose | 0.00002332 | 0.00002331 | +1e-08 |
| bytes | 188,636 | 191,052 | **-2,416** |
| S | 0.170536856816211 | 0.172141297491896 | **-0.001604** |

**The distortion-invariance argument is now MEASURED on the CUDA axis, not just derived.** Our
archive's inflated raw is byte-identical to PR130's base raw, and on Modal T4 CUDA it reproduces the
contest bot's own published PR130 distortion values to their full displayed precision. Two
independently-produced numbers on two different machines agree at 1e-8.

### The 4.275e-06 residual is display rounding in MY derivation input, not measurement noise

I derived 0.170532582261153 = bar − 25·2416/W. Measured came back 0.170536856816211. I initially
read the 4.275e-06 gap as ~5 flipped tie pixels. It is not. Recompute with the bot's *displayed*
d_seg/d_pose and our bytes: 0.170532582261153, matching my derivation to **2.8e-17**. The entire
residual is the bot's 8-decimal display truncation propagating into the bar I derived from. The
derivation was exact; its input was rounded. Corrected, and it strengthens rather than weakens the
result.

### The refusal mechanism, named exactly

`experiments/contest_auth_eval.py:673-712` requires proven parity against `upstream/.venv/bin/python`
before it will stamp `contest-CUDA`. On Modal it recorded
`OSError(8, 'Exec format error')` for that reference. Two independent defects:

1. **A macOS binary is being shipped into the Linux image.** Locally
   `upstream/.venv/bin/python -> /opt/homebrew/opt/python@3.11/bin/python3.11`. The upload
   dereferences the symlink, so the container holds a Mach-O executable at that path — it `exists()`,
   so the gate tries it, and Linux refuses to exec it. This is the concrete form of #836's "2 lost
   symlinks" drift, surfacing as a dispatch blocker.

2. **The image's packages materially diverge from `upstream/uv.lock`.** Measured:

   | package | Modal image ran | upstream/uv.lock pins |
   |---|---|---|
   | torch | 2.5.1 | **2.9.0+cu126** |
   | torchvision | 0.20.1 | **0.24.0** |
   | timm | 1.0.27 | **1.0.22** |
   | numpy | 1.26.4 | **2.3.4** |

   Even with defect 1 fixed, parity would fail on the merits. Fixing only the symlink would convert an
   honest refusal into a passing gate over a still-divergent environment — the worse outcome.

**The gate was right and the fail-closed design paid for itself.** Note the empirical counterweight,
recorded so it is not mistaken for a dismissal: torch 2.5.1 and whatever the contest bot runs agree to
1e-8 on this exact input, so the divergence is not *detectably* moving the score here. That is one
data point on one archive, and it is not a licence to relabel the axis.

### What this costs and what it buys

The candidate is worth **-0.001604 S** against the bar, on measured distortion and measured bytes. It
needs a contest-CUDA row to move the pointer, and that needs the locked environment built inside the
image (`uv sync` from `upstream/uv.lock`, then evaluate through that interpreter so
`eval_python == upstream_ref` and the mismatch block is skipped by identity rather than by assertion).
`--upstream-python` exists as a bypass; using it here would be an operator assertion of a parity that
is measurably false.

The CPU axis (`ap-NcyMr0ASXxEDRSjP3oseaw`) produced **0 tasks** and stopped — no ledger row, nothing
ran, nothing charged. It needs re-firing after the environment fix, not before.

**BASE: PR130 CPR1 S = 0.172141297491896447 @ 191,052 B `[contest-CUDA, DALI GT, n600]` — UNMOVED.**
Best candidate 0.170536856816211 @ 188,636 B `[CUDA env-mismatch advisory, MEASURED]`.
