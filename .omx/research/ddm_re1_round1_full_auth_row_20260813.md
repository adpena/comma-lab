# re1 Round-1 full evaluate.py row — HONEST LOSS + the family law (2026-08-13, MAIN)

> ⚠ ERRATA 2026-08-13 (`ddm_errata_8dp_band_instrument_mixing_20260813.md`): the
> '+4.03e-6 WORSE' verdict is RETRACTED as a signed result — the pose leg is exactly ONE
> 8dp report ULP (+6.02e-6); both canonicals carry declared ±3.5e-6 bounds. Sign
> INDETERMINATE; resolution = dual-axis worker base+candidate legs (qs2 fire-order).


## The row [contest-CUDA T4, n600 — full evaluate.py, score_claim: true]
- Archive: re1 Round-1, sha `7be3eb94b229306278a6ed204e2c716d7aafa98f6f93c82a5d2be18822467dfa`
  @ 186,252 B. Dispatch `fc-01KZY85HNP3RX6GMQXSM6RBQYG` (paired dispatcher, CUDA leg;
  CPU leg refused by single-flight — harmless, F26 would have refused it anyway).
- **S (recomputed from components): 0.16195916412468953**
  - avg d_seg 0.00029641 (−2 flips vs cp135's 0.00029643 — the sign-gate result CONFIRMED
    at the full instrument: seg leg −1.695e-6 S)
  - avg d_pose ≈ 6.895e-06 (vs base 6.8856e-06: **+9.5e-9 d_pose → +5.7e-6 S**)
  - rate identical (byte-equal archive).
- vs cp135 floor 0.16195513827824176: **+4.03e-6 WORSE. Round 1 DEAD at INSTANCE scope
  on the complete-S instrument.** The seg win did not pay its pose interaction.
- Receipt: `experiments/results/modal_auth_eval/ddm_re1_round1_full_auth_20260813_cuda/`
  (contest_auth_eval.json + provenance + report.txt).

## THE FAMILY LAW (two independent measurements, same shape)
| candidate | seg effect | pose effect (S) | net |
|---|---|---|---|
| JO1 six events (+1 B) | ~neutral | +2.05e-4 | +2.16e-4 LOSS |
| re1 Round-1 (0 B) | −1.7e-6 | +5.7e-6 | +4.0e-6 LOSS |

Mechanism (recalled, ddm_pz1): PoseNet and SegNet consume the IDENTICAL
`interpolate(x, segnet_model_input_size)` output — the shared D. No seg-targeted cell
edit is pose-invisible by construction; the measured exchange matches hv1's marginal
price (603 S per unit d_pose). Per-flip seg value = 100/117,964,800 = 8.477e-7 S;
measured per-cell pose damage = 5.7e-6 … 3.4e-5 S — **7–40× the seg gain.**

**LAW (FAMILY scope, cp135 base, HP3 semantic-cell closure): unprojected semantic-cell
edits are POSE-DOMINATED. Admission requires pose-null projection (Q3 — #837 measured
the exactly-pose-null frame_1 subspace SEG-REACHABLE) or a per-candidate pose-vector
screen BEFORE compile.** Existence remains proven (edits survive realization; the sign
gate works); the selection rule was the missing piece.

## Consequences (executed)
1. **js6's 200-proposal bank fire-order AMENDED**: no proposal compiles to a T4 candidate
   without (a) Q3 pose-null projection of the cell edit, or (b) a pose-vector screen.
   The efficient instrument: extend the re1t/sa1 T4 worker to retain PoseNet 6-vectors
   per pair IN THE SAME dispatch as the seg field (PoseNet forward on already-decoded
   frames ≈ free on T4) — one dispatch, both axes, no more seg-only provisional gates.
2. re1 Round-2 (+1 B) stays UNFIRED — same closure, no pose screen, presumptively same
   family fate; reopens only pose-screened.
3. Cadence ledger: this IS a byte-closed complete-S row (honest negative, banked).
   #381 spend ≈ $2.4 + 0.16 (sign gate) + ~0.16 (this row) ≈ **$2.7 of $20**.
