Design completed. **No launch, training, or scorer run occurred; no successor is yet proven feasible.**

These are **conditional scenarios, not measured scores**:

| Rank | Construction | Archive bytes | Conditional S |
|---|---|---:|---:|
| 1, unresolved | Coarse field + fixed regularizer + residual | 112,194–150,971 | 0.091826–0.117646 |
| 1, unresolved | Coarse field + counted pattern table + residual | 115,710–154,487 | 0.094167–0.119987 |
| Counterfactual | Exact 48 KB generator + pre-distortion edits | 120,406–145,976 | 0.097293–0.114319 |
| 2, weak hypothesis | Eight-parameter Lane generator | 148,906–154,507 | 0.116271–0.120000 |

The **one $0 rung** compares the first two constructions using real coded packets and exact n600 field reconstruction. **Falsifier: both complete packets exceed 94,292 B.**

Exact command sketch, queued under the charter’s no-launch constraint:

```bash
sed -n '/^# SO1_FIRST_RUNG_BEGIN$/,/^# SO1_FIRST_RUNG_END$/p' \
  .omx/research/ddm_so1_successor_object_with_preimage_freedom_20260912.md \
  | .venv/bin/python - --resume-from /Volumes/VertigoDataTier/pact/ddm_so1_first_rung
```

[Design memo](/Users/adpena/Projects/pact/.omx/research/ddm_so1_successor_object_with_preimage_freedom_20260912.md)  
SHA-256: `f482fe77c19d66a5034a838aa832d9cd037c66e0130fb424b6a81bfb1a8ca7d9`

Serializer **rc=17**, Git-object write denied. The [fallback bundle](/Volumes/VertigoDataTier/pact/ddm_so1/receipts/serializer/20260912T125832.306742Z-19519/intended-commit.bundle) is verified; all 12 declared hashes match. Shared index unchanged. MAIN must land fallback commit `e071c05dffd1826cd5d2bbc6e516c0c1544d0a77`.

composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN landing owner; consumer [LANDING.json](/Users/adpena/Projects/pact/.omx/research/ddm_so1_20260912/LANDING.json); trigger verified rc17 bundle, now satisfied. Land the declared files.
- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN/SO1 byte-successor; consumer `/Volumes/VertigoDataTier/pact/ddm_so1_first_rung/RESULT.json`; trigger harvested design, revalidated pins, and governed resource admission. Run the single comparison.
- **FOLDED into success branch** — owner MAIN receiver owner; consumer that store’s `receiver/`; trigger exact packet ≤94,261 B. Build and validate receiver identity and timing before evaluation.

## LIVE-HYPOTHESES

- Counted fine-pattern tables may preserve recurring pre-distortion cheaply because coarse spatial neighbors expose useful boundary context.
- Fixed regularization may code better despite more errors because clustered residuals can be cheaper.
- A different Lane positional generator remains plausible because Lane dominates surprise, provided topology and residuals are fully charged.

## DEAD-ENDS

- Unmodified HG1 as an exact 48 KB generator: its measured residual already costs 385,448 B.
- Treating marginal edit prices as standalone sparse entropy: independent addresses remain unpaid.
- Removing one-third of bytes from Lane’s surprise share: D3B disproves that accounting shortcut.
- Reusing the born n32 distortion, roughly 700-second slack, or FB1’s old axis bound: later receipts and current arithmetic contradict those premises.