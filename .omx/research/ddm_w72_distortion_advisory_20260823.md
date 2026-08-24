# ddm_w72 — the 25.67%-of-demand renderer rung is DEAD, and pose is 65.3% of what kills it

**Verdict: REFUSED.** `nested_group_dense_w72` measures **S 6.864979038642395** on
`[macOS-CPU env-mismatch advisory]` vs the dx2 pointer's 0.14821987563243377 — **46.3× worse**.
The 10,879 B it buys are real and the rate credit lands to eight significant figures; the
distortion it pays is ~928× that credit.

`verdict_scope`: **INSTANCE** — this rung, this archive
(`a731065431f1b134a5a2ceb51c969666e68def57bc0ca2c4a51dc7e2fb45d2f6`, 169,489 B), this axis.
It does not by itself close the renderer family; see §4 for exactly what it does close.

STORES CONSULTED: `.omx/research/ddm_rj1_renderer_joint_move_20260823.md` (the rung table and
the W64 prior) · `ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange rate, CITED not
re-derived) · `ddm_ar1b_archive_residue_purchase_20260822.md` (renderer = 30,856 B of residue) ·
tasks #1222 / #1224 (the renderer-carries-pose pair) · #1147/CPU→CUDA seg-transfer law ·
#1140/#1142 (GT-lineage fork) · the amplification-exponent-16.7 prohibition.

---

## 1. The measurement

| quantity | dx2 pointer | W72 | ratio |
|---|---:|---:|---:|
| `d_seg` | 0.00020139 | **0.02351655** | 116.8× |
| `d_pose` | 0.00000637 | **1.93641210** | 303,989× |
| archive B | 180,368 | 169,489 | −10,879 |
| **S** | 0.14821987563243377 | **6.864979038642395** | **46.3×** |

S recomputed from components, never from the printed 2-dp display (which reads 6.86 and
differs from canonical by 0.004979).

Receipt: `/Volumes/APDataStore/pact/ddm_w72_distortion_advisory/attempt_r1/contest_auth_eval.json`
(`score_claim: false`, `promotable: false`, `evidence_grade: auth-eval env mismatch advisory`).
Raw sha `c3eec3a8d09a4bed9e1107090a2925fb6b1da14792d798af5290f31a04807cc3`, 3,662,409,600 B.
n=600. Inflate 771.7 s rc=0, evaluate 421.0 s, total 1,193.9 s. Cost $0 (local).

## 2. The rate credit is EXACT; the distortion is 928× it

- realized rate credit **0.007243832 S** vs rj1's published **0.007243880061** — agreement to
  6.6e-9, i.e. the byte saving is exactly what rj1 said it was. The rung's *rate* claim is sound.
- seg cost **+2.331516 S = 321.9×** the credit.
- pose cost **+4.392487 S = 606.4×** the credit.
- On the seg budget alone (Δd_seg < 7.2439e-5 for break-even): **322× over**.

There is no axis caveat that survives two orders of magnitude. The CPU→CUDA seg-transfer law
(#1147) makes CPU seg deltas *upper bounds*, which is the direction that could rescue a marginal
row — it cannot rescue 322×. The pose column rides the PyAV GT lineage (#1140/#1142) and is
therefore not decision-usable on its own; at 303,989× base it does not need to be.

## 3. Pose is 65.3% of the damage — the THIRD independent confirmation

`pose 4.392487 / (seg 2.331516 + pose 4.392487) = 65.3%`.

The inflate receipt records `decoded_token_sha256 = cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`
— **bit-identical to dx2's categorical field**. W72's token stream is the pointer's token stream.
Every byte of the 10,879 B saving, and every unit of the distortion, comes from the **renderer**.
That makes this a clean isolation of the renderer axis, and it lands on the same side as:

- **#1224 / rj1**: renderer re-representation REFUSED 3.51×, d_pose 97.70% of it (contest-CUDA).
- **#1222 / mf1**: zero-byte boundary pull ΔS +0.77834455 with pose 95.37% of it.
- **this row**: pose 65.3% of a 46.3× refusal, on a rung whose tokens never moved.

Three measurements, three magnitudes, one direction: **PoseNet scores the FRAMES, so the
renderer is the pose carrier.** Perturbing it is priced in pose first.

## 4. What this closes, and what it does not

rj1 retained three rungs. Their status is now:

| rung | bytes bought | % of the 42,382 B demand | status |
|---|---:|---:|---|
| `nested_group_dense_w72` | 10,879 | 25.669% | **MEASURED DEAD (this row, 46.3×)** |
| `pointwise_svd_w96_r32` | 5,191 | 12.248% | dead unconditionally (rj1; #1225) |
| `film_amortized_flat_w96` | 1,078 | 2.544% | **UNMEASURED — stays open** (ny1/#1225) |

Two prior rungs on this axis are now measured dead at different widths — W64 at d_seg 0.03182023
(rj1) and W72 at 0.02351655 here. W72 is genuinely better than W64, as more width should be, and
it is still 322× over budget. **I do not extrapolate to W96**: the amplification exponent is ~16.7
and interpolating distortion between rungs is forbidden. What I state instead is a ceiling —
even at *zero* distortion, `film_amortized_flat_w96` buys **1,078 B, 2.544% of the demand**. It
cannot be the route; it can only ever be a rider on one.

So: the renderer *re-representation* axis is closed on both rungs that were large enough to
matter, without closing the renderer family — which is exactly what rj2 was chartered to test
by a different mechanism (joint optimization against both frozen scorers with in-compile
compensation and carrier re-solve, not post-hoc re-representation).

## 5. What it means for rj2

rj2's PRIOR-LAW PREDICTION was "REFUTED for a pointer move, CONFIRMED for mechanism recovery."
This row sharpens the *reason*: a renderer change that leaves the tokens alone pays 65.3% of its
distortion in pose, and the compensation machinery rj2 is required to carry (qs5 in-compile Schur
compensation, proven; up2/up3 + jg2 carrier re-solve, proven) is precisely the machinery that
addresses that term. This row does not predict rj2 succeeds. It measures the size of the debt
rj2's two required mechanisms have to service: **4.39 S of pose on a 25.67%-of-demand rung.**

## 6. Own-vehicle frontier

dx2 — S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`, archive sha
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` — **UNMOVED by this row**.
Gap to 0.12 = 0.028220 ⇒ 42,382 B at fixed distortion, or 150 B at zero distortion.
