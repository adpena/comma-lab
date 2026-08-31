# TWENTY-THIRD POINTER MOVE — S 0.14797617125559104 @ 180,002 B, lossless, distortion BIT-IDENTICAL · plus a NO-TOY erratum against two of my own numbers from today

Date: 2026-08-31 · Author: MAIN
Axis: **[contest-CUDA T4 n600]** — exact `upstream/evaluate.py` on the exact shipped bytes.
`score_claim=true` for §1 (authority row). §2 is an erratum, `score_claim=false`.

## 1. THE ROW — ADMITTED

Call `fc-01M1C2ZZQEQWNE0FT06R3WZJCS` · lane `ddm_afr1_tile48_groupbin8_cuda_n600_20260831` ·
archive **cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25**, **180,002 B** ·
runtime tree `6cdfa27dd1e9b46f…` · `n_samples 600` · `score_axis contest_cuda`.

**Recomputed from components (#877 — the printed `final_score: 0.15` is a rounded display and is
NEVER the number):**

```
rate = 25 × 180,002 / 37,545,489 = 0.11985594327989708
seg  = 100 × 0.00020139          = 0.020139
pose = sqrt(10 × 6.37e-06)       = 0.007981227975693965
S                                 = 0.14797617125559104   ← matches the receipt field EXACTLY
```

| | lb1 (prior pointer) | afr1 | delta |
|---|---:|---:|---:|
| archive | 180,083 B | **180,002 B** | **−81 B** |
| d_seg | 0.00020139 | **0.00020139** | **IDENTICAL** |
| d_pose | 6.37e-06 | **6.37e-06** | **IDENTICAL** |
| S | 0.14803010583079396 | **0.14797617125559104** | **−5.3934575202918555e-05** |

**The lossless claim realized exactly.** Projected rate-only ΔS was
`−5.393457520289588e-05`; realized `−5.3934575202918555e-05` — agreement to **12 significant
figures**, and both distortion terms came back bit-identical, which is precisely what afr1's
receiver-identity proof (600/600 pairs, 3,662,409,600/3,662,409,600 raw bytes, 0 differing)
guaranteed. Admit bar `−1.997576859366514e-05` → **2.700× the bar.**

The row was NECESSARY, not ceremonial: `rr2` (#1096) measured device-scoped decode desync on this
coder family (CPU-prob encode × CUDA-prob decode → S 27.83). macOS decode-identity does not imply
CUDA decode-identity. Only this row could admit it.

Sub-0.12 gap now **0.027976171255591042**. Pointer + effective_frontier updated; lane closed
`completed_admitted`.

## 2. NO-TOY ERRATUM — the operator's law applied to my OWN two numbers from today

Operator, 2026-08-31: *"No naive or toy."* Applied retroactively, it convicts two things I wrote in
the last hour. Recording both, because the law is a CHARTER-TIME law and both defects were born in
my charter/arithmetic, not in the arms.

### 2a. `ddm_lfb1`'s 9.262× is TOY-PRICED — my charter invited it

I chartered lfb1 to price the Lane fold at **gf1's measured 0.2909 B/correction**. That is a
**GENERIC clustered-residual rate**, applied to Lane — an object that is thin, parametric,
geometric and temporally coherent. Pricing a structured payload with a generic residual estimator
is exactly the defect **#1202** already caught in my own `ef1` charter ("races generic estimators
against a domain-tuned learned model"). **Second instance, same genus, mine both times.**

⚠ **CORRECTION to my own headline.** I wrote *"D3 stays closed at 9.262×."* That overstates what was
measured. What is honest:

| statement | status |
|---|---|
| 690,874 Lane positions require restoration (0.5857% of the field) | **MEASURED**, double-controlled |
| generic-residual price = 200,975 B = 9.262× the bar | **TOY-PRICED UPPER ESTIMATE** — do not cite as the Lane carriage price |
| `gf1`'s PARAMETRIC lane stream 36,044 B = **1.661×** the bar | **the real measured Lane price, and the honest reason D3 stays closed** |

lfb1's own verdict said this correctly — *"not a new coder measurement and closes no Lane-carrier
family."* The arm was honest; my headline was not. **D3 stays closed on 1.661×, not 9.262×.**

### 2b. `ddm_hyb1` §4a priced B's tokens by AREA — label corrected

I wrote `0.734 × 113,492` for the spatial split's token cost. Area-proportional token pricing is a
**toy model**. I did label the result a lower bound and gave the sign argument (B holds the hard
content, so content-weighting only worsens it), and the margin is 1.72× — so the **verdict stands**
and the direction is safe. But the row's label must read **"toy-priced, sign-safe bound"**, not a
clean measurement. Corrected here rather than left to be read as measured.

### 2c. What the law changes going forward

Any charter that prices a STRUCTURED object must race a **domain-matched** coder, not a generic
per-correction rate inherited from a different object. The open question is untouched and correctly
stated: **is there a Lane carrier below 21,699 B?** — with `gf1`'s 36,044 B parametric stream as the
incumbent to beat by 1.661×, and no toy figure standing in for it.

## 3. Denominator

Exact rows fired: **1**. Admitted: **1**. Pointer moves: **1** (the 23rd). Own numbers corrected
under the no-toy law: **2** (lfb1's 9.262× headline · hyb1's area-priced bound). Dollars: ~$0.16.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; MOVED −5.3934575202918555e-05.`
