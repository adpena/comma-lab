# CRF Sweep Results (2026-04-10)

Single-video test (video 0). SVT-AV1 with film-grain=22, keyint=-1, sharpness=1.

| CRF | Video Size | Savings vs CRF 34 | Projected Score* | Rate Term |
|-----|-----------|-------------------|-----------------|-----------|
| 34  | 847.4 KB  | baseline          | 1.366           | 0.608     |
| 35  | 791.2 KB  | -6.6%             | 1.328           | 0.570     |
| 36  | 758.1 KB  | -10.5%            | 1.305           | 0.548     |

*Projected score assumes postfilter holds distortion constant (seg=0.00610, pose=0.00218).

## Rate savings in score terms
- CRF 35: saves **0.038 points** on rate
- CRF 36: saves **0.061 points** on rate

## Key question
Does the postfilter trained on CRF 34 artifacts compensate for CRF 35/36's
additional compression? If yes, these are free points. If not, we retrain.

## Note on preset
CRF 34 and 35 were encoded with preset 0 (maximum quality, very slow).
CRF 36 was encoded with preset 4 (faster) — may have slightly different
rate-distortion characteristics at the same CRF value.

## Next step
Test current postfilter on CRF 35 archive via proxy eval (zero-cost test).
